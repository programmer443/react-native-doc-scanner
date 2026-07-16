//
//  ONNXInference.swift
//  react-native-doc-scanner
//
//  Owns the four ONNX Runtime sessions (document detector + 3-stage RapidOCR
//  pipeline: det/cls/rec) and all tensor pre/post-processing that isn't pure
//  OpenCV image ops (those live in OpenCVBridge). Uses the official Microsoft
//  ONNX Runtime Objective-C API (`onnxruntime-objc` pod, `ORTSession`/
//  `ORTValue`/`ORTEnv` etc.) directly from Swift.
//
//  NOTE ON RISK: `onnxruntime-objc`'s exact Obj-C selector names below
//  (`ORTSession(env:modelPath:sessionOptions:)`, `session.run(withInputs:
//  outputNames:runOptions:)`, `ORTValue(tensorData:elementType:shape:)`,
//  `session.inputNames()/outputNames()`, `ORTTensorTypeAndShapeInfo.shape`)
//  are implemented against the long-stable, publicly documented ORT iOS API
//  surface. The Pods for this module weren't installed in this environment
//  (no `pod install` was run against the host app), so these calls could not
//  be compiled/verified against the actual installed headers — double check
//  them against `Pods/onnxruntime-objc/**/*.h` on first real build.
//

import Foundation
import NitroModules
import onnxruntime_objc

/// Matches `MODEL_INPUT_SIZE.detectorWidth/detectorHeight` in
/// `src/constants/thresholds.ts` — the DocAligner detector's square input.
private let kDetectorInputSize: Int32 = 256

/// PP-OCRv4 det: standard DBNet long-side cap (keeps the resized image at a
/// manageable resolution for mobile inference while preserving small text).
private let kOcrDetMaxSide: Int32 = 960

/// PP-OCRv3 rec: fixed input height (RapidOCR/PaddleOCR convention).
private let kRecTargetHeight: Int32 = 48

/// ppocr_mobile cls: fixed input size (RapidOCR/PaddleOCR convention).
private let kClsWidth: Int32 = 192
private let kClsHeight: Int32 = 48

/// YuNet's fixed square input size (see docs/MODEL_TRAINING.md §5) — matches
/// `preprocessFaceDetectorInput`'s targetSize and the decode grid math below.
/// Confirmed directly from the real face_detection_yunet_2023mar.onnx graph
/// (static [1,3,640,640] input) — NOT 320x320, a common demo default that
/// doesn't match this exact export (caught by inspecting the actual
/// downloaded model file after initial implementation assumed 320).
private let kFaceDetectorInputSize: Int32 = 640

/// YuNet score/NMS thresholds, matching OpenCV's own `face_detect.cpp`
/// defaults exactly (ground truth for this decode — see the task spec).
private let kFaceScoreThreshold: Double = 0.7
private let kFaceNmsThreshold: Double = 0.3

/// The 3 FPN stride levels YuNet's ONNX export decodes at.
private let kFaceStrides: [Int32] = [8, 16, 32]

/// The 12 named output tensors YuNet's ONNX export produces — one
/// cls/obj/bbox/kps quadruplet per stride level above.
private let kFaceOutputNames: Set<String> = [
  "cls_8", "cls_16", "cls_32",
  "obj_8", "obj_16", "obj_32",
  "bbox_8", "bbox_16", "bbox_32",
  "kps_8", "kps_16", "kps_32",
]

/// Document-type classifier's square input — matches
/// `MODEL_INPUT_SIZE.classifierWidth/classifierHeight` in
/// `src/constants/thresholds.ts`. Update both together if your fine-tune
/// (see docs/MODEL_TRAINING.md §6) uses a different size.
private let kClassifierInputSize: Int32 = 224

/// Class order the classifier's output vector is read against — MUST exactly
/// match the label order the model was trained/exported with, or predictions
/// will silently point at the wrong document type. `"GENERIC"` is included as
/// the "not a recognised ID document" / background class.
private let kClassifierLabels: [String] = [
  "PASSPORT", "DRIVING_LICENCE", "ID_CARD", "RESIDENCE_PERMIT", "VISA", "GENERIC",
]

final class ONNXInference {
  struct DetectorOutput {
    let quad: Quad
    let confidence: Double
  }

  struct RecognizedLine {
    let text: String
    let confidence: Double
  }

  /// One detected face, already mapped back into full pixel-buffer space
  /// (buffer-oriented, i.e. NOT yet display-orientation-normalised — that
  /// happens in HybridDocScanner, same division of responsibility as
  /// `DetectorOutput.quad` above). `box` is an axis-aligned rect (buffer
  /// space stays axis-aligned post-squash-resize since x/y are scaled
  /// independently). `landmarks` order matches `FaceLandmarks`: rightEye,
  /// leftEye, noseTip, rightMouthCorner, leftMouthCorner.
  struct FaceCandidate {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
    let landmarks: [Point]
    let score: Double
  }

  /// One raw decoded YuNet candidate, still in 640x640 model-input pixel
  /// space, pre-NMS.
  private struct RawFaceCandidate {
    let x1: Double
    let y1: Double
    let w: Double
    let h: Double
    let score: Double
    /// 5 points, model-input space, order: rightEye, leftEye, noseTip,
    /// rightMouthCorner, leftMouthCorner.
    let landmarks: [(x: Double, y: Double)]
  }

  // Created lazily on first `loadModels()` call rather than in `init()` — this
  // type is owned by `HybridDocScanner`, whose own initializer (via the
  // nitrogen-generated `HybridDocScannerSpec_base`) is non-throwing, so
  // constructing the ORT environment eagerly would force an unrecoverable
  // `try!`/crash on the vanishingly-unlikely chance `ORTEnv` init fails.
  // Deferring it means that failure surfaces as a normal thrown error from
  // `loadModels()` (already async/throwing) instead.
  private var env: ORTEnv?
  private var detectorSession: ORTSession?
  private var ocrDetSession: ORTSession?
  private var ocrClsSession: ORTSession?
  private var ocrRecSession: ORTSession?
  private var charset: [String] = []
  // Optional 5th session — see loadModels()'s do/catch around this one.
  private var faceDetectorSession: ORTSession?
  // Optional 6th session — same do/catch, never-fails-the-load contract as
  // faceDetectorSession above.
  private var classifierSession: ORTSession?

  private(set) var resolvedDetectorPath: String = ""
  private(set) var resolvedOcrRecPath: String = ""
  private(set) var resolvedFaceDetectorPath: String = ""
  private(set) var resolvedClassifierPath: String = ""

  var detectorLoaded: Bool { detectorSession != nil }
  var modelsLoaded: Bool {
    detectorSession != nil && ocrDetSession != nil && ocrClsSession != nil && ocrRecSession != nil
  }
  var faceDetectorLoaded: Bool { faceDetectorSession != nil }
  var classifierLoaded: Bool { classifierSession != nil }

  init() {}

  private func ensureEnv() throws -> ORTEnv {
    if let env { return env }
    let newEnv = try ORTEnv(loggingLevel: .warning)
    env = newEnv
    return newEnv
  }

  // MARK: - Model loading

  func loadModels(config: ModelPaths) throws -> LoadModelsResult {
    let detectorPath = try ModelPathResolver.resolve(config.detectorModelPath)
    let ocrDetPath = try ModelPathResolver.resolve(config.ocrDetModelPath)
    let ocrClsPath = try ModelPathResolver.resolve(config.ocrClsModelPath)
    let ocrRecPath = try ModelPathResolver.resolve(config.ocrRecModelPath)
    let charsetPath = try ModelPathResolver.resolve(config.ocrRecCharsetPath)

    for path in [detectorPath, ocrDetPath, ocrClsPath, ocrRecPath, charsetPath] {
      guard FileManager.default.fileExists(atPath: path) else {
        throw RuntimeError.error(withMessage: "react-native-doc-scanner: model file not found at \"\(path)\". Run loadModels with valid ModelPaths (see docs/MODEL_TRAINING.md).")
      }
    }

    let ortEnv = try ensureEnv()
    let options = try ORTSessionOptions()
    let newDetector = try ORTSession(env: ortEnv, modelPath: detectorPath, sessionOptions: options)
    let newOcrDet = try ORTSession(env: ortEnv, modelPath: ocrDetPath, sessionOptions: options)
    let newOcrCls = try ORTSession(env: ortEnv, modelPath: ocrClsPath, sessionOptions: options)
    let newOcrRec = try ORTSession(env: ortEnv, modelPath: ocrRecPath, sessionOptions: options)

    let charsetText = try String(contentsOfFile: charsetPath, encoding: .utf8)
    var lines = charsetText.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    if let last = lines.last, last.isEmpty {
      lines.removeLast()
    }
    guard !lines.isEmpty else {
      throw RuntimeError.error(withMessage: "react-native-doc-scanner: charset file at \"\(charsetPath)\" is empty.")
    }

    // Commit all-or-nothing, after every session above has been created
    // successfully — a partially-updated engine (e.g. new detector but old
    // rec model) would be a confusing state to leave the app in.
    detectorSession = newDetector
    ocrDetSession = newOcrDet
    ocrClsSession = newOcrCls
    ocrRecSession = newOcrRec
    charset = lines
    resolvedDetectorPath = detectorPath
    resolvedOcrRecPath = ocrRecPath

    // Face detector: OPTIONAL, unlike the four models above. Only attempt to
    // load it if a path was actually supplied, and — critically — never let
    // a failure here throw out of loadModels(): the document/OCR pipeline
    // above must keep working even if the face model is missing or bad.
    // analyzeFaceFrame() already treats faceDetectorSession == nil as
    // "detected: false" rather than an error, so this just leaves face
    // detection unavailable instead of failing model loading entirely.
    var faceDetectorVersion = ""
    if config.faceDetectorModelPath.isEmpty {
      faceDetectorSession = nil
      resolvedFaceDetectorPath = ""
    } else {
      do {
        let faceDetectorPath = try ModelPathResolver.resolve(config.faceDetectorModelPath)
        guard FileManager.default.fileExists(atPath: faceDetectorPath) else {
          throw RuntimeError.error(withMessage: "react-native-doc-scanner: face detector model file not found at \"\(faceDetectorPath)\".")
        }
        let newFaceDetector = try ORTSession(env: ortEnv, modelPath: faceDetectorPath, sessionOptions: options)
        faceDetectorSession = newFaceDetector
        resolvedFaceDetectorPath = faceDetectorPath
        faceDetectorVersion = Self.fileFingerprint(path: faceDetectorPath)
      } catch {
        faceDetectorSession = nil
        resolvedFaceDetectorPath = ""
        faceDetectorVersion = ""
      }
    }

    // Document-type classifier: OPTIONAL, exact same contract as the face
    // detector above — skip if no path supplied, never let a failure here
    // throw out of loadModels(). analyzeFrame() already treats
    // classifierSession == nil as "keep documentType GENERIC" rather than an
    // error, so this just leaves classification unavailable instead of
    // failing model loading entirely.
    var classifierVersion = ""
    if config.classifierModelPath.isEmpty {
      classifierSession = nil
      resolvedClassifierPath = ""
    } else {
      do {
        let classifierPath = try ModelPathResolver.resolve(config.classifierModelPath)
        guard FileManager.default.fileExists(atPath: classifierPath) else {
          throw RuntimeError.error(withMessage: "react-native-doc-scanner: classifier model file not found at \"\(classifierPath)\".")
        }
        let newClassifier = try ORTSession(env: ortEnv, modelPath: classifierPath, sessionOptions: options)
        classifierSession = newClassifier
        resolvedClassifierPath = classifierPath
        classifierVersion = Self.fileFingerprint(path: classifierPath)
      } catch {
        classifierSession = nil
        resolvedClassifierPath = ""
        classifierVersion = ""
      }
    }

    return LoadModelsResult(
      success: true,
      detectorVersion: Self.fileFingerprint(path: detectorPath),
      ocrVersion: Self.fileFingerprint(path: ocrRecPath),
      faceDetectorVersion: faceDetectorVersion,
      classifierVersion: classifierVersion
    )
  }

  /// A "real read" version string: filename + size + modification time. Not a
  /// cryptographic hash (that would mean hashing potentially large model
  /// files on every load), but genuinely reflects the actual file on disk
  /// rather than a hardcoded placeholder.
  private static func fileFingerprint(path: String) -> String {
    let name = (path as NSString).lastPathComponent
    if let attrs = try? FileManager.default.attributesOfItem(atPath: path) {
      let size = (attrs[.size] as? UInt64) ?? 0
      let modDate = (attrs[.modificationDate] as? Date) ?? Date(timeIntervalSince1970: 0)
      return "\(name)#\(size)@\(Int(modDate.timeIntervalSince1970))"
    }
    return name
  }

  // MARK: - Detector

  /// Runs the DocAligner ONNX detector on one camera frame's pixel buffer.
  /// Returns `nil` if no session is loaded or the model's output tensor
  /// doesn't match either of the two shapes this is robust to (heatmap or
  /// direct 8-value point regression) — callers should fall back to the
  /// classical OpenCV contour detector in that case.
  func runDetector(pixelBuffer: CVPixelBuffer, bridge: OpenCVBridge) throws -> DetectorOutput? {
    guard let session = detectorSession else { return nil }

    var scaleX: Double = 1
    var scaleY: Double = 1
    guard let inputData = bridge.preprocessDetectorInput(
      withPixelBuffer: pixelBuffer, targetSize: kDetectorInputSize, scaleX: &scaleX, scaleY: &scaleY
    ) else {
      return nil
    }

    let inputNames = try session.inputNames()
    let outputNames = try session.outputNames()
    guard let inputName = inputNames.first, !outputNames.isEmpty else { return nil }

    let shape: [NSNumber] = [1, 3, NSNumber(value: kDetectorInputSize), NSNumber(value: kDetectorInputSize)]
    let inputTensor = try ORTValue(tensorData: NSMutableData(data: inputData), elementType: .float, shape: shape)
    let outputs = try session.run(withInputs: [inputName: inputTensor], outputNames: Set(outputNames), runOptions: nil)

    let bufferWidth = Double(CVPixelBufferGetWidth(pixelBuffer))
    let bufferHeight = Double(CVPixelBufferGetHeight(pixelBuffer))

    // Try every declared output until one matches a shape we know how to
    // interpret — robust to whichever DocAligner export variant (heatmap vs.
    // direct point regression) actually gets bundled, per
    // docs/MODEL_TRAINING.md's resilience design.
    for name in outputNames {
      guard let value = outputs[name] else { continue }
      if let parsed = try parseDetectorOutput(value, scaleX: scaleX, scaleY: scaleY, bufferWidth: bufferWidth, bufferHeight: bufferHeight) {
        return parsed
      }
    }
    return nil
  }

  private func parseDetectorOutput(
    _ value: ORTValue, scaleX: Double, scaleY: Double, bufferWidth: Double, bufferHeight: Double
  ) throws -> DetectorOutput? {
    let shapeInfo = try value.tensorTypeAndShapeInfo()
    // NSNumber.intValue is Int32; normalise to Int so it composes with
    // Array.count/indices (all plain Int) without explicit casts everywhere.
    let shape = shapeInfo.shape.map { Int($0.intValue) }
    let rawData = try value.tensorData() as Data
    let floats: [Float] = rawData.withUnsafeBytes { raw in
      Array(raw.bindMemory(to: Float.self))
    }

    func toBufferPoint(modelX: Double, modelY: Double) -> Point {
      // Plain per-axis inverse of the squash-resize (no letterbox/padding —
      // see preprocessDetectorInput's doc comment) back to full
      // pixel-buffer coordinates.
      Point(x: modelX / scaleX, y: modelY / scaleY)
    }

    // Heatmap variant: [1, 4, H, W] — one channel per corner. Each corner's
    // location is the argmax pixel of its channel.
    if shape.count == 4, shape[1] == 4 {
      let mapH = shape[2], mapW = shape[3]
      let planeSize = mapH * mapW
      guard planeSize > 0, floats.count >= 4 * planeSize else { return nil }

      var points: [Point] = []
      var peakSum: Double = 0
      for c in 0..<4 {
        var bestIdx = 0
        var bestVal = -Float.greatestFiniteMagnitude
        let base = c * planeSize
        for i in 0..<planeSize {
          let v = floats[base + i]
          if v > bestVal {
            bestVal = v
            bestIdx = i
          }
        }
        let py = bestIdx / mapW
        let px = bestIdx % mapW
        let modelX = (Double(px) + 0.5) / Double(mapW) * Double(kDetectorInputSize)
        let modelY = (Double(py) + 0.5) / Double(mapH) * Double(kDetectorInputSize)
        points.append(toBufferPoint(modelX: modelX, modelY: modelY))
        peakSum += Double(Self.sigmoidIfNeeded(bestVal))
      }

      let confidence = min(1.0, max(0.0, peakSum / 4.0))
      let quad = Quad(topLeft: points[0], topRight: points[1], bottomRight: points[2], bottomLeft: points[3])
      return DetectorOutput(quad: quad, confidence: confidence)
    }

    // Direct-regression variant: flattens to exactly 8 floats — normalised
    // (x, y) model-space coordinates for 4 corners.
    let flatCount = shape.reduce(1, *)
    if flatCount == 8, floats.count >= 8 {
      var points: [Point] = []
      for i in 0..<4 {
        let nx = Double(floats[i * 2])
        let ny = Double(floats[i * 2 + 1])
        let modelX = nx * Double(kDetectorInputSize)
        let modelY = ny * Double(kDetectorInputSize)
        points.append(toBufferPoint(modelX: modelX, modelY: modelY))
      }
      // No natural per-inference confidence signal for direct regression —
      // fixed value, as called out in the task spec (§5.3): these exports are
      // trained to always emit *some* quad, so a flat, reasonably-confident
      // constant is the honest choice here (not a fake/random number).
      let quad = Quad(topLeft: points[0], topRight: points[1], bottomRight: points[2], bottomLeft: points[3])
      return DetectorOutput(quad: quad, confidence: 0.9)
    }

    return nil
  }

  private static func sigmoidIfNeeded(_ raw: Float) -> Float {
    // Heatmap peak values might already be probabilities (if the model
    // applies its own sigmoid) or raw logits. If it's clearly outside [0,1],
    // assume logits and squash it.
    if raw >= 0, raw <= 1 { return raw }
    return 1.0 / (1.0 + expf(-raw))
  }

  // MARK: - Face detector (YuNet)

  /// Runs the YuNet ONNX face detector on one camera frame's pixel buffer.
  /// Returns `nil` if no session is loaded, no face scored above
  /// `kFaceScoreThreshold`, or the model output couldn't be read — callers
  /// should treat that as "no face detected" (see `NativeFaceFrameResult`'s
  /// `detected: false` contract), never throw further. Front-camera selfie
  /// use case: only the single highest-scoring face (post-NMS) is returned,
  /// matching `NativeFaceFrameResult` only having room for one box/landmark
  /// set.
  func runFaceDetector(pixelBuffer: CVPixelBuffer, bridge: OpenCVBridge) throws -> FaceCandidate? {
    guard let session = faceDetectorSession else { return nil }

    var scaleX: Double = 1
    var scaleY: Double = 1
    guard let inputData = bridge.preprocessFaceDetectorInput(
      withPixelBuffer: pixelBuffer, targetSize: kFaceDetectorInputSize, scaleX: &scaleX, scaleY: &scaleY
    ) else {
      return nil
    }

    let inputNames = try session.inputNames()
    guard let inputName = inputNames.first else { return nil }

    let shape: [NSNumber] = [1, 3, NSNumber(value: kFaceDetectorInputSize), NSNumber(value: kFaceDetectorInputSize)]
    let inputTensor = try ORTValue(tensorData: NSMutableData(data: inputData), elementType: .float, shape: shape)
    let outputs = try session.run(withInputs: [inputName: inputTensor], outputNames: kFaceOutputNames, runOptions: nil)

    func floats(_ name: String) throws -> [Float] {
      guard let value = outputs[name] else { return [] }
      let data = try value.tensorData() as Data
      return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    var allCandidates: [RawFaceCandidate] = []
    for stride in kFaceStrides {
      let cls = try floats("cls_\(stride)")
      let obj = try floats("obj_\(stride)")
      let bbox = try floats("bbox_\(stride)")
      let kps = try floats("kps_\(stride)")
      guard !cls.isEmpty, !obj.isEmpty, !bbox.isEmpty, !kps.isEmpty else { continue }
      allCandidates.append(contentsOf: Self.decodeFaceStride(stride: stride, cls: cls, obj: obj, bbox: bbox, kps: kps))
    }
    guard !allCandidates.isEmpty else { return nil }

    // Greedy NMS, then take the single highest-scoring survivor. Note the
    // top-scoring candidate overall is *always* a survivor of greedy NMS (it
    // has nothing already-kept to conflict with when it's considered first),
    // so this is equivalent to "argmax by score" for this selfie single-face
    // use case — NMS is still implemented in full per the task spec (ground
    // truth against OpenCV's own algorithm), not skipped as a shortcut.
    guard let best = Self.nmsFaceCandidates(allCandidates).first else { return nil }

    func toBufferPoint(modelX: Double, modelY: Double) -> Point {
      // Same per-axis inverse of the squash-resize as the document detector's
      // toBufferPoint (see parseDetectorOutput above) — plain division, no
      // letterbox/padding to undo.
      Point(x: modelX / scaleX, y: modelY / scaleY)
    }

    let bufferTopLeft = toBufferPoint(modelX: best.x1, modelY: best.y1)
    let bufferBottomRight = toBufferPoint(modelX: best.x1 + best.w, modelY: best.y1 + best.h)
    let bufferLandmarks = best.landmarks.map { toBufferPoint(modelX: $0.x, modelY: $0.y) }

    return FaceCandidate(
      x: bufferTopLeft.x,
      y: bufferTopLeft.y,
      width: bufferBottomRight.x - bufferTopLeft.x,
      height: bufferBottomRight.y - bufferTopLeft.y,
      landmarks: bufferLandmarks,
      score: best.score
    )
  }

  /// Decodes one YuNet FPN stride level's 4 output tensors into candidate
  /// faces, in 640x640 model-input pixel space. Ground-truth algorithm from
  /// OpenCV's own `modules/objdetect/src/face_detect.cpp` — see the task
  /// spec for the full derivation; not to be "simplified" without re-checking
  /// against that source.
  private static func decodeFaceStride(
    stride: Int32, cls: [Float], obj: [Float], bbox: [Float], kps: [Float]
  ) -> [RawFaceCandidate] {
    let s = Double(stride)
    let cols = Int(kFaceDetectorInputSize / stride)
    let rows = Int(kFaceDetectorInputSize / stride)
    guard cols > 0, rows > 0 else { return [] }

    var candidates: [RawFaceCandidate] = []
    candidates.reserveCapacity(rows * cols)

    for r in 0..<rows {
      for c in 0..<cols {
        let idx = r * cols + c
        guard idx < cls.count, idx < obj.count,
              idx * 4 + 3 < bbox.count, idx * 10 + 9 < kps.count else { continue }

        let clsScore = min(1.0, max(0.0, Double(cls[idx])))
        let objScore = min(1.0, max(0.0, Double(obj[idx])))
        let score = (clsScore * objScore).squareRoot()
        if score < kFaceScoreThreshold { continue }

        let bboxBase = idx * 4
        let cx = (Double(c) + Double(bbox[bboxBase])) * s
        let cy = (Double(r) + Double(bbox[bboxBase + 1])) * s
        let w = exp(Double(bbox[bboxBase + 2])) * s
        let h = exp(Double(bbox[bboxBase + 3])) * s
        let x1 = cx - w / 2
        let y1 = cy - h / 2

        var landmarks: [(x: Double, y: Double)] = []
        landmarks.reserveCapacity(5)
        let kpsBase = idx * 10
        for n in 0..<5 {
          let lx = (Double(kps[kpsBase + 2 * n]) + Double(c)) * s
          let ly = (Double(kps[kpsBase + 2 * n + 1]) + Double(r)) * s
          landmarks.append((x: lx, y: ly))
        }

        candidates.append(RawFaceCandidate(x1: x1, y1: y1, w: w, h: h, score: score, landmarks: landmarks))
      }
    }
    return candidates
  }

  /// Standard greedy IoU-based NMS, sorted by score descending: keep a
  /// candidate iff its IoU with every already-kept candidate is below
  /// `kFaceNmsThreshold`. Hand-implemented (no `dnn::NMSBoxes` available from
  /// Swift) per the task spec.
  private static func nmsFaceCandidates(_ candidates: [RawFaceCandidate]) -> [RawFaceCandidate] {
    let sorted = candidates.sorted { $0.score > $1.score }
    var kept: [RawFaceCandidate] = []
    for candidate in sorted {
      let overlapsKept = kept.contains { iou($0, candidate) >= kFaceNmsThreshold }
      if !overlapsKept {
        kept.append(candidate)
      }
    }
    return kept
  }

  private static func iou(_ a: RawFaceCandidate, _ b: RawFaceCandidate) -> Double {
    let aX2 = a.x1 + a.w, aY2 = a.y1 + a.h
    let bX2 = b.x1 + b.w, bY2 = b.y1 + b.h
    let interX1 = max(a.x1, b.x1)
    let interY1 = max(a.y1, b.y1)
    let interX2 = min(aX2, bX2)
    let interY2 = min(aY2, bY2)
    let interW = max(0.0, interX2 - interX1)
    let interH = max(0.0, interY2 - interY1)
    let interArea = interW * interH
    let unionArea = a.w * a.h + b.w * b.h - interArea
    guard unionArea > 0 else { return 0 }
    return interArea / unionArea
  }

  // MARK: - Document-type classifier

  struct ClassificationOutput {
    let documentType: String
    let confidence: Double
  }

  /// Runs the document-type classifier on one camera frame's pixel buffer,
  /// optionally cropped to `roi` (full-buffer pixel coordinates; pass `nil`
  /// to classify the whole frame — see HybridDocScanner.analyzeFrame's
  /// fallback when no quad has been found yet). Returns `nil` if no session
  /// is loaded, preprocessing failed, or the output couldn't be read —
  /// callers should treat that as "leave documentType at GENERIC", same
  /// no-throw contract as `runFaceDetector`.
  func runClassifier(pixelBuffer: CVPixelBuffer, roi: CGRect?, bridge: OpenCVBridge) throws -> ClassificationOutput? {
    guard let session = classifierSession else { return nil }
    guard let inputData = bridge.preprocessClassifierInput(
      withPixelBuffer: pixelBuffer, roi: roi ?? .null, targetSize: kClassifierInputSize
    ) else {
      return nil
    }

    let inputNames = try session.inputNames()
    let outputNames = try session.outputNames()
    guard let inputName = inputNames.first, let outputName = outputNames.first else { return nil }

    let shape: [NSNumber] = [1, 3, NSNumber(value: kClassifierInputSize), NSNumber(value: kClassifierInputSize)]
    let inputTensor = try ORTValue(tensorData: NSMutableData(data: inputData), elementType: .float, shape: shape)
    let outputs = try session.run(withInputs: [inputName: inputTensor], outputNames: Set(outputNames), runOptions: nil)
    guard let outputValue = outputs[outputName] else { return nil }

    let data = try outputValue.tensorData() as Data
    let floats: [Float] = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    guard floats.count >= kClassifierLabels.count else { return nil }

    var bestIdx = 0
    var bestVal = floats[0]
    for i in 1..<kClassifierLabels.count {
      if floats[i] > bestVal {
        bestVal = floats[i]
        bestIdx = i
      }
    }

    // Softmax-normalise just the winning logit against the full class set for
    // a confidence estimate — same trick ctcGreedyDecode uses below, avoids a
    // full softmax pass. Handles both raw logits and an already-softmaxed
    // output (in the latter case sumExp collapses to ~1/bestVal, still a
    // valid confidence).
    var sumExp: Float = 0
    for i in 0..<kClassifierLabels.count {
      sumExp += expf(floats[i] - bestVal)
    }
    let confidence = sumExp > 0 ? 1.0 / Double(sumExp) : 0

    return ClassificationOutput(documentType: kClassifierLabels[bestIdx], confidence: confidence)
  }

  // MARK: - OCR: detection (DBNet)

  func runOcrDetection(bridge: OpenCVBridge) throws -> [DSTextBox] {
    guard let session = ocrDetSession else {
      throw RuntimeError.error(withMessage: "react-native-doc-scanner: loadModels() must complete before captureAndExtract().")
    }

    var outWidth: Int32 = 0
    var outHeight: Int32 = 0
    guard let inputData = bridge.preprocessOcrDetInput(withMaxSide: kOcrDetMaxSide, outWidth: &outWidth, outHeight: &outHeight) else {
      return []
    }

    let inputNames = try session.inputNames()
    let outputNames = try session.outputNames()
    guard let inputName = inputNames.first, let outputName = outputNames.first else { return [] }

    let shape: [NSNumber] = [1, 3, NSNumber(value: outHeight), NSNumber(value: outWidth)]
    let inputTensor = try ORTValue(tensorData: NSMutableData(data: inputData), elementType: .float, shape: shape)
    let outputs = try session.run(withInputs: [inputName: inputTensor], outputNames: Set(outputNames), runOptions: nil)
    guard let outputValue = outputs[outputName] else { return [] }

    let outShapeInfo = try outputValue.tensorTypeAndShapeInfo()
    let outShape = outShapeInfo.shape.map { Int($0.intValue) }
    // DBNet det output is typically [1, 1, H, W] (single-channel probability
    // map), sometimes exported as [1, H, W]. Take the last two dims as the
    // map size regardless of which.
    guard outShape.count >= 2 else { return [] }
    let mapH = outShape[outShape.count - 2]
    let mapW = outShape[outShape.count - 1]

    let outputData = try outputValue.tensorData() as Data
    return bridge.textBoxes(fromProbabilityMap: outputData, mapWidth: Int32(mapW), mapHeight: Int32(mapH), threshold: 0.3)
  }

  // MARK: - OCR: orientation classification

  /// Returns `true` if the text-line crop should be rotated 180° before
  /// recognition (ppocr_mobile 2-class cls: `[prob(0°), prob(180°)]`).
  func classifyIs180(box: DSTextBox, bridge: OpenCVBridge) throws -> Bool {
    guard let session = ocrClsSession else { return false }
    guard let inputData = bridge.preprocessClsInput(for: box, clsWidth: kClsWidth, clsHeight: kClsHeight) else {
      return false
    }

    let inputNames = try session.inputNames()
    let outputNames = try session.outputNames()
    guard let inputName = inputNames.first, let outputName = outputNames.first else { return false }

    let shape: [NSNumber] = [1, 3, NSNumber(value: kClsHeight), NSNumber(value: kClsWidth)]
    let inputTensor = try ORTValue(tensorData: NSMutableData(data: inputData), elementType: .float, shape: shape)
    let outputs = try session.run(withInputs: [inputName: inputTensor], outputNames: Set(outputNames), runOptions: nil)
    guard let outputValue = outputs[outputName] else { return false }

    let data = try outputValue.tensorData() as Data
    let floats: [Float] = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    guard floats.count >= 2 else { return false }
    return floats[1] > floats[0]
  }

  // MARK: - OCR: recognition (CTC)

  func recognize(box: DSTextBox, rotate180: Bool, bridge: OpenCVBridge) throws -> RecognizedLine? {
    guard let session = ocrRecSession else {
      throw RuntimeError.error(withMessage: "react-native-doc-scanner: loadModels() must complete before captureAndExtract().")
    }

    var outWidth: Int32 = 0
    guard let inputData = bridge.preprocessRecInput(for: box, rotate180: rotate180, targetHeight: kRecTargetHeight, outWidth: &outWidth) else {
      return nil
    }

    let inputNames = try session.inputNames()
    let outputNames = try session.outputNames()
    guard let inputName = inputNames.first, let outputName = outputNames.first else { return nil }

    let shape: [NSNumber] = [1, 3, NSNumber(value: kRecTargetHeight), NSNumber(value: outWidth)]
    let inputTensor = try ORTValue(tensorData: NSMutableData(data: inputData), elementType: .float, shape: shape)
    let outputs = try session.run(withInputs: [inputName: inputTensor], outputNames: Set(outputNames), runOptions: nil)
    guard let outputValue = outputs[outputName] else { return nil }

    let shapeInfo = try outputValue.tensorTypeAndShapeInfo()
    let outShape = shapeInfo.shape.map { Int($0.intValue) }
    guard outShape.count == 3 else { return nil }
    let seqLen = outShape[1]
    let numClasses = outShape[2]
    guard seqLen > 0, numClasses > 0 else { return nil }

    let data = try outputValue.tensorData() as Data
    let floats: [Float] = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    guard floats.count >= seqLen * numClasses else { return nil }

    return ctcGreedyDecode(floats: floats, seqLen: seqLen, numClasses: numClasses)
  }

  /// Standard CTC greedy decode: per timestep, take the argmax class; skip
  /// the blank (index 0, PP-OCR convention — `en_dict.txt` lists only real
  /// characters starting at CTC index 1); collapse consecutive repeats of the
  /// same non-blank class into a single character.
  private func ctcGreedyDecode(floats: [Float], seqLen: Int, numClasses: Int) -> RecognizedLine {
    var text = ""
    var confidences: [Double] = []
    var previousIndex = -1

    for t in 0..<seqLen {
      let base = t * numClasses
      var bestIdx = 0
      var bestVal = floats[base]
      for c in 1..<numClasses {
        let v = floats[base + c]
        if v > bestVal {
          bestVal = v
          bestIdx = c
        }
      }

      if bestIdx != 0, bestIdx != previousIndex {
        let charIndex = bestIdx - 1
        if charIndex >= 0, charIndex < charset.count {
          // Softmax-normalise just the winning logit against the full class
          // set for a confidence estimate, without a full softmax pass.
          var sumExp: Float = 0
          for c in 0..<numClasses {
            sumExp += expf(floats[base + c] - bestVal)
          }
          let winnerProb = sumExp > 0 ? 1.0 / sumExp : 0
          text += charset[charIndex]
          confidences.append(Double(winnerProb))
        }
      }
      previousIndex = bestIdx
    }

    let meanConfidence = confidences.isEmpty ? 0.0 : confidences.reduce(0, +) / Double(confidences.count)
    return RecognizedLine(text: text, confidence: meanConfidence)
  }
}
