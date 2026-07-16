//
//  HybridDocScanner.swift
//  react-native-doc-scanner
//
//  Swift implementation of the `DocScanner` Nitro HybridObject (see
//  src/specs/DocScanner.nitro.ts and the generated
//  nitrogen/generated/ios/swift/HybridDocScannerSpec.swift). Wires together:
//   - ONNXInference: ONNX Runtime sessions for the detector + 3-stage OCR.
//   - OpenCVBridge: real OpenCV C++ ops (quality analysis, classical detector
//     fallback, perspective correction, DBNet pre/post-processing).
//
//  `analyzeFrame` is the hot path (called once per camera frame, synchronously,
//  from the VisionCamera frame-processor thread — never touch UIKit/main-
//  thread-only APIs here). `loadModels`/`captureAndExtract` are async
//  (Promise-returning) and safe to do heavier work.
//

import CoreGraphics
import CoreVideo
import Foundation
import NitroModules
import VisionCamera

/// Mirrors `SCANNER_THRESHOLDS.minClassificationConfidence` in
/// src/constants/thresholds.ts — keep both in sync if you retune this.
private let kClassifierMinConfidence: Double = 0.6

class HybridDocScanner: HybridDocScannerSpec {
  // One long-lived bridge for the per-frame path — it remembers the previous
  // frame's downscaled grayscale buffer for motion scoring (see
  // OpenCVBridge.mm's `_previousGray`), so this must NOT be recreated per call.
  private let frameBridge = OpenCVBridge()
  private let onnx = ONNXInference()

  // MARK: - loadModels

  func loadModels(config: ModelPaths) throws -> Promise<LoadModelsResult> {
    return Promise<LoadModelsResult>.async { [weak self] in
      guard let self else {
        throw RuntimeError.error(withMessage: "react-native-doc-scanner: HybridDocScanner was disposed before loadModels() completed.")
      }
      return try self.onnx.loadModels(config: config)
    }
  }

  // MARK: - analyzeFrame

  func analyzeFrame(frame: any HybridFrameSpec) throws -> NativeFrameResult {
    let nativeBuffer = try frame.getNativeBuffer()
    defer { nativeBuffer.release() }

    guard let rawPointer = UnsafeMutableRawPointer(bitPattern: UInt(nativeBuffer.pointer)) else {
      return Self.emptyResult(frame: frame)
    }
    // `getNativeBuffer()` hands us a +1-retained CVPixelBuffer (VisionCamera's
    // `CVPixelBuffer.asNativeBuffer()`); take it unretained here and release
    // the extra retain via `nativeBuffer.release()` in the `defer` above,
    // exactly matching the ownership contract documented on
    // `Frame.getNativeBuffer()` in react-native-vision-camera.
    let pixelBuffer = Unmanaged<CVPixelBuffer>.fromOpaque(rawPointer).takeUnretainedValue()

    let bufferWidth = Double(CVPixelBufferGetWidth(pixelBuffer))
    let bufferHeight = Double(CVPixelBufferGetHeight(pixelBuffer))
    guard bufferWidth > 0, bufferHeight > 0 else {
      return Self.emptyResult(frame: frame)
    }

    let quality = frameBridge.analyzeQuality(withPixelBuffer: pixelBuffer, roi: .null)

    var bufferQuad: Quad?
    var confidence: Double = 0

    // Primary path: DocAligner ONNX detector, if loaded.
    if onnx.detectorLoaded, let result = try? onnx.runDetector(pixelBuffer: pixelBuffer, bridge: frameBridge) {
      bufferQuad = result.quad
      confidence = result.confidence
    }

    // Fallback path: classical OpenCV contour detection — used whenever the
    // ONNX model isn't loaded yet, fails to run, or its output didn't match
    // either shape ONNXInference knows how to interpret.
    if bufferQuad == nil {
      if let fallback = frameBridge.detectDocumentQuad(withPixelBuffer: pixelBuffer),
         fallback.detected, let dsQuad = fallback.quad {
        bufferQuad = Self.quad(fromDSQuad: dsQuad)
        confidence = fallback.confidence
      }
    }

    guard let detectedQuad = bufferQuad, confidence > 0 else {
      let (frameWidth, frameHeight) = Self.orientedDimensions(
        bufferWidth: bufferWidth, bufferHeight: bufferHeight, orientation: frame.orientation)
      return NativeFrameResult(
        detected: false,
        documentType: "GENERIC",
        confidence: 0,
        quad: nil,
        frameWidth: frameWidth,
        frameHeight: frameHeight,
        blurScore: quality.blurScore,
        brightness: quality.brightness,
        glareRatio: quality.glareRatio,
        motionScore: quality.motionScore,
        distanceRatio: 0,
        perspectiveSkewDeg: 0,
        outOfFrameRatio: 0
      )
    }

    // Document-type classification: only attempted once a document is
    // already confidently found (the guard above already filters out
    // no-document frames, so this never runs on background/empty frames —
    // a 6th ONNX session is real added per-frame cost). Crops to the
    // detected quad's bounding box, in the same raw buffer pixel space the
    // quad is already in at this point (pre-orientation-normalisation).
    // Only trusts the classifier's own result if its confidence clears
    // kClassifierMinConfidence; otherwise leaves documentType at GENERIC
    // rather than report a low-confidence guess. If no classifier is loaded
    // (classifierModelPath was empty/failed to load — see
    // ONNXInference.loadModels), documentType stays GENERIC exactly as
    // before this feature existed.
    var documentType = "GENERIC"
    if onnx.classifierLoaded {
      let roi = Self.boundingRect(ofQuad: detectedQuad)
      if let classification = try? onnx.runClassifier(pixelBuffer: pixelBuffer, roi: roi, bridge: frameBridge),
         classification.confidence >= kClassifierMinConfidence {
        documentType = classification.documentType
      }
    }

    // Orientation contract: quad/frameWidth/frameHeight must be normalised to
    // the preview's display orientation (accounting for frame.orientation +
    // isMirrored) before returning — see the doc comment on
    // `NativeFrameResult` in src/vision/nativeFrameResult.ts. Done here in
    // Swift, not punted to JS.
    let (orientedQuad, frameWidth, frameHeight) = Self.normalizeOrientation(
      quad: detectedQuad, bufferWidth: bufferWidth, bufferHeight: bufferHeight,
      orientation: frame.orientation, isMirrored: frame.isMirrored
    )

    let distanceRatio = Self.quadArea(orientedQuad) / max(1.0, frameWidth * frameHeight)
    let perspectiveSkewDeg = Self.perspectiveSkewDeg(orientedQuad)
    let outOfFrameRatio = Self.outOfFrameRatio(orientedQuad, frameWidth: frameWidth, frameHeight: frameHeight)

    return NativeFrameResult(
      detected: true,
      documentType: documentType,
      confidence: confidence,
      quad: orientedQuad,
      frameWidth: frameWidth,
      frameHeight: frameHeight,
      blurScore: quality.blurScore,
      brightness: quality.brightness,
      glareRatio: quality.glareRatio,
      motionScore: quality.motionScore,
      distanceRatio: distanceRatio,
      perspectiveSkewDeg: perspectiveSkewDeg,
      outOfFrameRatio: outOfFrameRatio
    )
  }

  // MARK: - analyzeFaceFrame

  func analyzeFaceFrame(frame: any HybridFrameSpec) throws -> NativeFaceFrameResult {
    let nativeBuffer = try frame.getNativeBuffer()
    defer { nativeBuffer.release() }

    guard let rawPointer = UnsafeMutableRawPointer(bitPattern: UInt(nativeBuffer.pointer)) else {
      return Self.emptyFaceResult(frame: frame)
    }
    // Same +1-retained/`nativeBuffer.release()` ownership contract as
    // `analyzeFrame` above — see its comment for the full explanation.
    let pixelBuffer = Unmanaged<CVPixelBuffer>.fromOpaque(rawPointer).takeUnretainedValue()

    let bufferWidth = Double(CVPixelBufferGetWidth(pixelBuffer))
    let bufferHeight = Double(CVPixelBufferGetHeight(pixelBuffer))
    guard bufferWidth > 0, bufferHeight > 0 else {
      return Self.emptyFaceResult(frame: frame)
    }

    // Quality analysis runs on every frame that has a valid pixel buffer,
    // regardless of whether a face is actually found this frame — lighting/
    // blur/motion guidance should be available even before a face is
    // detected. Reuses `frameBridge`, the SAME long-lived instance
    // `analyzeFrame` uses, so its motion-detection `_previousGray` state
    // stays continuous regardless of which detector mode the caller is
    // currently driving.
    let quality = frameBridge.analyzeQuality(withPixelBuffer: pixelBuffer, roi: .null)

    guard onnx.faceDetectorLoaded,
          let candidate = try? onnx.runFaceDetector(pixelBuffer: pixelBuffer, bridge: frameBridge)
    else {
      // No face detector loaded (faceDetectorModelPath was empty/failed to
      // load — see ONNXInference.loadModels), or no face found this frame:
      // both report detected: false, never throw, per NativeFaceFrameResult's
      // contract. Quality metrics are still returned (computed above).
      let (frameWidth, frameHeight) = Self.orientedDimensions(
        bufferWidth: bufferWidth, bufferHeight: bufferHeight, orientation: frame.orientation)
      return NativeFaceFrameResult(
        detected: false,
        confidence: 0,
        box: nil,
        landmarks: nil,
        frameWidth: frameWidth,
        frameHeight: frameHeight,
        blurScore: quality.blurScore,
        brightness: quality.brightness,
        glareRatio: quality.glareRatio,
        motionScore: quality.motionScore
      )
    }

    // Orientation contract identical to analyzeFrame's quad handling — see
    // normalizeFaceOrientation's doc comment for how it differs for a
    // box+landmarks vs. a Quad.
    let (orientedBox, orientedLandmarks, frameWidth, frameHeight) = Self.normalizeFaceOrientation(
      boxX: candidate.x, boxY: candidate.y, boxWidth: candidate.width, boxHeight: candidate.height,
      landmarks: candidate.landmarks,
      bufferWidth: bufferWidth, bufferHeight: bufferHeight,
      orientation: frame.orientation, isMirrored: frame.isMirrored
    )

    return NativeFaceFrameResult(
      detected: true,
      confidence: candidate.score,
      box: orientedBox,
      landmarks: orientedLandmarks,
      frameWidth: frameWidth,
      frameHeight: frameHeight,
      blurScore: quality.blurScore,
      brightness: quality.brightness,
      glareRatio: quality.glareRatio,
      motionScore: quality.motionScore
    )
  }

  // MARK: - captureAndExtract

  func captureAndExtract(photoPath: String, documentType: String, quad: Quad?) throws -> Promise<RawOcrResultNative> {
    return Promise<RawOcrResultNative>.async { [weak self] in
      guard let self else {
        throw RuntimeError.error(withMessage: "react-native-doc-scanner: HybridDocScanner was disposed before captureAndExtract() completed.")
      }
      return try self.performCaptureAndExtract(photoPath: photoPath, documentType: documentType, quad: quad)
    }
  }

  private func performCaptureAndExtract(photoPath: String, documentType: String, quad: Quad?) throws -> RawOcrResultNative {
    guard onnx.modelsLoaded else {
      throw RuntimeError.error(withMessage: "react-native-doc-scanner: loadModels() must succeed before captureAndExtract() — no models are loaded.")
    }

    let cleanPath = photoPath.hasPrefix("file://") ? String(photoPath.dropFirst("file://".count)) : photoPath

    // Fresh, short-lived bridge instance for this single capture — distinct
    // from `frameBridge`, which must keep living across per-frame calls.
    let captureBridge = OpenCVBridge()
    let imageSize = captureBridge.loadCaptureImage(atPath: cleanPath)
    guard imageSize.width > 0, imageSize.height > 0 else {
      throw RuntimeError.error(withMessage: "react-native-doc-scanner: could not read captured photo at \"\(cleanPath)\".")
    }

    var rectifiedImagePath = cleanPath
    if let quad {
      // NOTE (risk, see final report): `quad` is expressed in the pixel space
      // of whatever frame `analyzeFrame` last measured it against, which may
      // be a different resolution than this full-resolution captured photo
      // (VisionCamera's frame-processor stream and photo output can be
      // configured to different resolutions). Lacking any frame-dimensions
      // parameter in this method's fixed spec, we treat `quad` as already
      // being in the loaded photo's own pixel space (the literal reading of
      // `captureAndExtract`'s contract) and defensively clamp it into bounds
      // so a mismatched quad degrades gracefully instead of corrupting the
      // warp or crashing.
      let clamped = Self.clampQuad(quad, width: Double(imageSize.width), height: Double(imageSize.height))
      if Self.quadArea(clamped) > 16 {
        let outputPath = (NSTemporaryDirectory() as NSString)
          .appendingPathComponent("docscanner_rectified_\(UUID().uuidString).jpg")
        let dsQuad = Self.dsQuad(fromQuad: clamped)
        if captureBridge.rectifyCaptureImage(with: dsQuad, outputPath: outputPath) {
          rectifiedImagePath = outputPath
        }
        // If rectification fails we deliberately keep going with the
        // original (already-loaded) image rather than throwing — a failed
        // warp shouldn't block OCR on an otherwise perfectly readable photo.
      }
    }

    let boxes = try onnx.runOcrDetection(bridge: captureBridge)

    let currentSize = captureBridge.currentImageSize()
    let width = max(1.0, Double(currentSize.width))
    let height = max(1.0, Double(currentSize.height))

    var lines: [OcrTextLineNative] = []
    lines.reserveCapacity(boxes.count)

    for box in boxes {
      let rotate180 = (try? onnx.classifyIs180(box: box, bridge: captureBridge)) ?? false
      guard let recognized = try onnx.recognize(box: box, rotate180: rotate180, bridge: captureBridge),
            !recognized.text.isEmpty else {
        continue
      }
      let bbox = box.boundingBox
      lines.append(OcrTextLineNative(
        text: recognized.text,
        confidence: recognized.confidence,
        x: Double(bbox.origin.x) / width,
        y: Double(bbox.origin.y) / height,
        width: Double(bbox.size.width) / width,
        height: Double(bbox.size.height) / height
      ))
    }

    let fullText = lines.map(\.text).joined(separator: "\n")
    let overallConfidence = lines.isEmpty ? 0.0 : lines.reduce(0.0) { $0 + $1.confidence } / Double(lines.count)

    return RawOcrResultNative(
      fullText: fullText,
      lines: lines,
      confidence: overallConfidence,
      rectifiedImagePath: rectifiedImagePath
    )
  }

  // MARK: - Geometry helpers (pure Swift — no OpenCV needed)

  private static func emptyResult(frame: any HybridFrameSpec) -> NativeFrameResult {
    let bufferWidth = frame.width
    let bufferHeight = frame.height
    let (frameWidth, frameHeight) = orientedDimensions(bufferWidth: bufferWidth, bufferHeight: bufferHeight, orientation: frame.orientation)
    return NativeFrameResult(
      detected: false, documentType: "GENERIC", confidence: 0, quad: nil,
      frameWidth: frameWidth, frameHeight: frameHeight,
      blurScore: 0, brightness: 0, glareRatio: 0, motionScore: 0,
      distanceRatio: 0, perspectiveSkewDeg: 0, outOfFrameRatio: 0
    )
  }

  /// All-zero/empty face result for the earliest fast-path failure (no
  /// pixel buffer at all) — mirrors `emptyResult` above, including NOT
  /// running quality analysis (there's no pixel buffer to analyse). Once a
  /// pixel buffer IS available, `analyzeFaceFrame` always computes real
  /// quality metrics even on a no-face frame — see its body.
  private static func emptyFaceResult(frame: any HybridFrameSpec) -> NativeFaceFrameResult {
    let bufferWidth = frame.width
    let bufferHeight = frame.height
    let (frameWidth, frameHeight) = orientedDimensions(bufferWidth: bufferWidth, bufferHeight: bufferHeight, orientation: frame.orientation)
    return NativeFaceFrameResult(
      detected: false, confidence: 0, box: nil, landmarks: nil,
      frameWidth: frameWidth, frameHeight: frameHeight,
      blurScore: 0, brightness: 0, glareRatio: 0, motionScore: 0
    )
  }

  private static func orientedDimensions(bufferWidth: Double, bufferHeight: Double, orientation: CameraOrientation) -> (Double, Double) {
    switch orientation {
    case .up, .down:
      return (bufferWidth, bufferHeight)
    case .left, .right:
      return (bufferHeight, bufferWidth)
    @unknown default:
      return (bufferWidth, bufferHeight)
    }
  }

  /// Normalises a quad + its buffer dimensions from raw sensor/buffer pixel
  /// space into the preview's display orientation, per the orientation
  /// contract on `NativeFrameResult`. `orientation` describes how the raw
  /// pixel data is rotated *relative to* the desired upright display (see
  /// react-native-vision-camera's `CameraOrientation` doc comment); the
  /// per-case normalized-space mapping below mirrors VisionCamera's own
  /// internal `FrameCoordinateSystemConverter.getFrameToCameraMatrix` (not
  /// public API, so reimplemented here from its documented semantics rather
  /// than imported).
  private static func normalizeOrientation(
    quad: Quad, bufferWidth: Double, bufferHeight: Double, orientation: CameraOrientation, isMirrored: Bool
  ) -> (Quad, Double, Double) {
    let (displayWidth, displayHeight) = orientedDimensions(bufferWidth: bufferWidth, bufferHeight: bufferHeight, orientation: orientation)

    func mapPoint(_ p: Point) -> Point {
      orientedPoint(
        p, bufferWidth: bufferWidth, bufferHeight: bufferHeight,
        displayWidth: displayWidth, displayHeight: displayHeight,
        orientation: orientation, isMirrored: isMirrored
      )
    }

    // Map all four points first, THEN re-derive which physical corner is
    // visually top-left/top-right/etc. — a 90° rotation permutes which
    // original corner ends up where, so simply carrying forward the
    // pre-rotation labels would mislabel corners after a left/right rotation.
    let mapped = [mapPoint(quad.topLeft), mapPoint(quad.topRight), mapPoint(quad.bottomRight), mapPoint(quad.bottomLeft)]
    let reordered = reorderClockwiseFromTopLeft(mapped)

    return (reordered, displayWidth, displayHeight)
  }

  /// Maps one point from raw sensor/buffer pixel space into the preview's
  /// display orientation (rotate per `orientation`, then mirror per
  /// `isMirrored`) — the single-point core of `normalizeOrientation` above,
  /// factored out so `normalizeFaceOrientation` below can reuse the exact
  /// same per-orientation-case math on a bounding box's corners + landmark
  /// points, which (unlike a `Quad`'s named corners) don't need any
  /// corner-identity reordering afterwards.
  private static func orientedPoint(
    _ p: Point, bufferWidth: Double, bufferHeight: Double, displayWidth: Double, displayHeight: Double,
    orientation: CameraOrientation, isMirrored: Bool
  ) -> Point {
    let xn = bufferWidth > 0 ? p.x / bufferWidth : 0
    let yn = bufferHeight > 0 ? p.y / bufferHeight : 0
    var xn2: Double
    var yn2: Double
    switch orientation {
    case .up:
      xn2 = xn
      yn2 = yn
    case .down:
      xn2 = 1 - xn
      yn2 = 1 - yn
    case .left:
      xn2 = yn
      yn2 = 1 - xn
    case .right:
      xn2 = 1 - yn
      yn2 = xn
    @unknown default:
      xn2 = xn
      yn2 = yn
    }
    if isMirrored {
      // Front camera: mirror horizontally in the already-oriented display
      // space, to match what the preview shows the user (see
      // docs/TROUBLESHOOTING.md's "Bounding box drawn mirrored" entry).
      xn2 = 1 - xn2
    }
    return Point(x: xn2 * displayWidth, y: yn2 * displayHeight)
  }

  /// Face-frame counterpart of `normalizeOrientation` above: applies the
  /// IDENTICAL per-point rotation/mirror transform (`orientedPoint`), but to
  /// a bounding box + 5 semantically-fixed landmarks instead of a `Quad`.
  /// A box has no per-corner "identity" the way a quad's named corners do, so
  /// its 4 geometric corners are mapped and then re-collapsed into a fresh
  /// axis-aligned rect (rather than reordered like a quad's corners). Each
  /// landmark's identity (e.g. "right eye") does NOT change under rotation —
  /// only its on-screen position does — so landmarks are mapped directly,
  /// with no reordering step at all.
  private static func normalizeFaceOrientation(
    boxX: Double, boxY: Double, boxWidth: Double, boxHeight: Double,
    // Order: rightEye, leftEye, noseTip, rightMouthCorner, leftMouthCorner —
    // matches ONNXInference.FaceCandidate.landmarks and FaceLandmarks' field
    // order.
    landmarks: [Point],
    bufferWidth: Double, bufferHeight: Double, orientation: CameraOrientation, isMirrored: Bool
  ) -> (BoundingBox, FaceLandmarks, Double, Double) {
    let (displayWidth, displayHeight) = orientedDimensions(bufferWidth: bufferWidth, bufferHeight: bufferHeight, orientation: orientation)

    func mapPoint(_ p: Point) -> Point {
      orientedPoint(
        p, bufferWidth: bufferWidth, bufferHeight: bufferHeight,
        displayWidth: displayWidth, displayHeight: displayHeight,
        orientation: orientation, isMirrored: isMirrored
      )
    }

    let corners = [
      Point(x: boxX, y: boxY),
      Point(x: boxX + boxWidth, y: boxY),
      Point(x: boxX + boxWidth, y: boxY + boxHeight),
      Point(x: boxX, y: boxY + boxHeight),
    ].map(mapPoint)
    let xs = corners.map(\.x)
    let ys = corners.map(\.y)
    let minX = xs.min() ?? 0, maxX = xs.max() ?? 0
    let minY = ys.min() ?? 0, maxY = ys.max() ?? 0
    let orientedBox = BoundingBox(x: minX, y: minY, width: max(0, maxX - minX), height: max(0, maxY - minY))

    let mappedLandmarks = landmarks.map(mapPoint)
    // Defensive: FaceCandidate.landmarks is always built with exactly 5
    // entries (see ONNXInference.decodeFaceStride), but guard rather than
    // force-index in case that invariant is ever broken by a future change.
    let orientedLandmarks: FaceLandmarks
    if mappedLandmarks.count == 5 {
      orientedLandmarks = FaceLandmarks(
        rightEye: mappedLandmarks[0],
        leftEye: mappedLandmarks[1],
        noseTip: mappedLandmarks[2],
        rightMouthCorner: mappedLandmarks[3],
        leftMouthCorner: mappedLandmarks[4]
      )
    } else {
      let zero = Point(x: 0, y: 0)
      orientedLandmarks = FaceLandmarks(rightEye: zero, leftEye: zero, noseTip: zero, rightMouthCorner: zero, leftMouthCorner: zero)
    }

    return (orientedBox, orientedLandmarks, displayWidth, displayHeight)
  }

  /// Orders 4 arbitrary points into [topLeft, topRight, bottomRight,
  /// bottomLeft] using the standard sum/difference heuristic (same one
  /// OpenCVBridge.mm's `OrderQuadPoints` uses on the C++ side): top-left has
  /// the smallest x+y, bottom-right the largest x+y, top-right the smallest
  /// y-x, bottom-left the largest y-x.
  private static func reorderClockwiseFromTopLeft(_ points: [Point]) -> Quad {
    let sums = points.map { $0.x + $0.y }
    let diffs = points.map { $0.y - $0.x }
    let tlIdx = sums.indices.min(by: { sums[$0] < sums[$1] })!
    let brIdx = sums.indices.max(by: { sums[$0] < sums[$1] })!
    let trIdx = diffs.indices.min(by: { diffs[$0] < diffs[$1] })!
    let blIdx = diffs.indices.max(by: { diffs[$0] < diffs[$1] })!
    return Quad(topLeft: points[tlIdx], topRight: points[trIdx], bottomRight: points[brIdx], bottomLeft: points[blIdx])
  }

  /// Axis-aligned bounding box of a quad's 4 corners, in whatever pixel space
  /// the quad itself is in — used to crop the classifier's input to the
  /// detected document region (see `preprocessClassifierInput`'s `roi`).
  private static func boundingRect(ofQuad quad: Quad) -> CGRect {
    let xs = [quad.topLeft.x, quad.topRight.x, quad.bottomRight.x, quad.bottomLeft.x]
    let ys = [quad.topLeft.y, quad.topRight.y, quad.bottomRight.y, quad.bottomLeft.y]
    let minX = xs.min() ?? 0, maxX = xs.max() ?? 0
    let minY = ys.min() ?? 0, maxY = ys.max() ?? 0
    return CGRect(x: minX, y: minY, width: max(0, maxX - minX), height: max(0, maxY - minY))
  }

  private static func quadArea(_ quad: Quad) -> Double {
    let pts = [quad.topLeft, quad.topRight, quad.bottomRight, quad.bottomLeft]
    var area = 0.0
    for i in 0..<pts.count {
      let p1 = pts[i]
      let p2 = pts[(i + 1) % pts.count]
      area += p1.x * p2.y - p2.x * p1.y
    }
    return area.magnitude / 2
  }

  private static func perspectiveSkewDeg(_ quad: Quad) -> Double {
    func angleDeviation(_ a: Point, _ b: Point, _ c: Point) -> Double {
      let v1x = a.x - b.x, v1y = a.y - b.y
      let v2x = c.x - b.x, v2y = c.y - b.y
      let dot = v1x * v2x + v1y * v2y
      let mag1 = (v1x * v1x + v1y * v1y).squareRoot()
      let mag2 = (v2x * v2x + v2y * v2y).squareRoot()
      guard mag1 > 1e-6, mag2 > 1e-6 else { return 0 }
      let cosAngle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
      let angleDeg = acos(cosAngle) * 180.0 / Double.pi
      return (angleDeg - 90.0).magnitude
    }
    let d0 = angleDeviation(quad.bottomLeft, quad.topLeft, quad.topRight)
    let d1 = angleDeviation(quad.topLeft, quad.topRight, quad.bottomRight)
    let d2 = angleDeviation(quad.topRight, quad.bottomRight, quad.bottomLeft)
    let d3 = angleDeviation(quad.bottomRight, quad.bottomLeft, quad.topLeft)
    return max(d0, d1, d2, d3)
  }

  private static func outOfFrameRatio(_ quad: Quad, frameWidth: Double, frameHeight: Double) -> Double {
    let xs = [quad.topLeft.x, quad.topRight.x, quad.bottomRight.x, quad.bottomLeft.x]
    let ys = [quad.topLeft.y, quad.topRight.y, quad.bottomRight.y, quad.bottomLeft.y]
    let minX = xs.min() ?? 0, maxX = xs.max() ?? 0
    let minY = ys.min() ?? 0, maxY = ys.max() ?? 0
    let bboxArea = max(0, maxX - minX) * max(0, maxY - minY)
    guard bboxArea > 0 else { return 0 }
    let insideMinX = max(0, minX), insideMaxX = min(frameWidth, maxX)
    let insideMinY = max(0, minY), insideMaxY = min(frameHeight, maxY)
    let insideArea = max(0, insideMaxX - insideMinX) * max(0, insideMaxY - insideMinY)
    return max(0, min(1, 1 - insideArea / bboxArea))
  }

  private static func clampQuad(_ quad: Quad, width: Double, height: Double) -> Quad {
    func clampPoint(_ p: Point) -> Point {
      Point(x: min(max(p.x, 0), max(0, width - 1)), y: min(max(p.y, 0), max(0, height - 1)))
    }
    return Quad(
      topLeft: clampPoint(quad.topLeft), topRight: clampPoint(quad.topRight),
      bottomRight: clampPoint(quad.bottomRight), bottomLeft: clampPoint(quad.bottomLeft)
    )
  }

  private static func quad(fromDSQuad dsQuad: DSQuad) -> Quad {
    Quad(
      topLeft: Point(x: Double(dsQuad.topLeft.x), y: Double(dsQuad.topLeft.y)),
      topRight: Point(x: Double(dsQuad.topRight.x), y: Double(dsQuad.topRight.y)),
      bottomRight: Point(x: Double(dsQuad.bottomRight.x), y: Double(dsQuad.bottomRight.y)),
      bottomLeft: Point(x: Double(dsQuad.bottomLeft.x), y: Double(dsQuad.bottomLeft.y))
    )
  }

  private static func dsQuad(fromQuad quad: Quad) -> DSQuad {
    DSQuad(
      topLeft: CGPoint(x: CGFloat(quad.topLeft.x), y: CGFloat(quad.topLeft.y)),
      topRight: CGPoint(x: CGFloat(quad.topRight.x), y: CGFloat(quad.topRight.y)),
      bottomRight: CGPoint(x: CGFloat(quad.bottomRight.x), y: CGFloat(quad.bottomRight.y)),
      bottomLeft: CGPoint(x: CGFloat(quad.bottomLeft.x), y: CGFloat(quad.bottomLeft.y))
    )
  }
}
