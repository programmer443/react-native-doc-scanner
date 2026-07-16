package com.margelo.nitro.docscanner

import android.content.Context
import android.util.Log
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.NitroModules
import com.margelo.nitro.camera.CameraOrientation
import com.margelo.nitro.camera.HybridFrameSpec
import com.margelo.nitro.camera.PixelFormat
import com.margelo.nitro.core.Promise
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max

/**
 * Kotlin implementation of the `DocScanner` Nitro HybridObject (see
 * `nitrogen/generated/android/kotlin/com/margelo/nitro/docscanner/HybridDocScannerSpec.kt`,
 * generated from `src/specs/DocScanner.nitro.ts`).
 *
 * Orchestrates:
 *  - [FrameConversion] + [OpenCvQualityAnalyzer] + [DocumentDetector] + [DocumentClassifier] for
 *    `analyzeFrame` (called on every VisionCamera frame, from a worklet/frame-processor thread).
 *  - [FrameConversion] + [OpenCvQualityAnalyzer] + [FaceDetector] for `analyzeFaceFrame` (same
 *    per-frame worklet-thread contract, used by the selfie capture guide instead of
 *    `analyzeFrame` — only one of the two runs per frame).
 *  - [OnnxEngine] for `loadModels`.
 *  - [OcrPipeline] for `captureAndExtract` (called once per capture, off the frame path).
 */
@DoNotStrip
@Keep
class HybridDocScanner : HybridDocScannerSpec() {
    private val onnxEngine = OnnxEngine()
    private val documentDetector = DocumentDetector()
    private val faceDetector = FaceDetector()
    private val documentClassifier = DocumentClassifier()
    // Shared with analyzeFrame's path on purpose: OpenCvQualityAnalyzer keeps previous-frame
    // state for motion scoring, which should stay continuous regardless of which detector mode
    // (document vs. face) is currently active on a given frame.
    private val qualityAnalyzer = OpenCvQualityAnalyzer()
    private val ocrPipeline = OcrPipeline()

    init {
        OpenCvBootstrap.ensureInitialized()
    }

    // -----------------------------------------------------------------
    // analyzeFrame — called on every camera frame
    // -----------------------------------------------------------------

    override fun analyzeFrame(frame: HybridFrameSpec): NativeFrameResult {
        return try {
            analyzeFrameInternal(frame)
        } catch (e: Exception) {
            Log.e(TAG, "analyzeFrame failed: ${e.message}", e)
            emptyResult(frame)
        }
    }

    private fun analyzeFrameInternal(frame: HybridFrameSpec): NativeFrameResult {
        val rawOriented = FrameConversion.toOrientedGrayMat(frame)
        try {
            val working = downscaleToWorkingResolution(rawOriented, QUALITY_WORKING_LONG_EDGE)
            try {
                val frameWidth = working.cols()
                val frameHeight = working.rows()

                val detection = documentDetector.detect(onnxEngine.detectorSession, working, MODEL_INPUT_SIZE)
                val quality = qualityAnalyzer.analyze(working, detection?.quad, frameWidth, frameHeight)

                // Document-type classification: only attempted once a document is already
                // confidently found (detection != null already filters out no-document frames —
                // a 6th ONNX session is real added per-frame cost). Only trusts the classifier's
                // own result if its confidence clears MIN_CLASSIFICATION_CONFIDENCE; otherwise
                // leaves documentType at GENERIC rather than report a low-confidence guess. If no
                // classifier is loaded (classifierModelPath was empty/failed to load — see
                // OnnxEngine.load), documentType stays GENERIC exactly as before this feature
                // existed.
                var documentType = "GENERIC"
                if (detection != null) {
                    val classification = documentClassifier.classify(
                        onnxEngine.classifierSession, working, detection.quad, CLASSIFIER_MODEL_INPUT_SIZE,
                    )
                    if (classification != null && classification.confidence >= MIN_CLASSIFICATION_CONFIDENCE) {
                        documentType = classification.documentType
                    }
                }

                return NativeFrameResult(
                    detected = detection != null,
                    documentType = documentType,
                    confidence = detection?.confidence ?: 0.0,
                    quad = detection?.quad,
                    frameWidth = frameWidth.toDouble(),
                    frameHeight = frameHeight.toDouble(),
                    blurScore = quality.blurScore,
                    brightness = quality.brightness,
                    glareRatio = quality.glareRatio,
                    motionScore = quality.motionScore,
                    distanceRatio = quality.distanceRatio,
                    perspectiveSkewDeg = quality.perspectiveSkewDeg,
                    outOfFrameRatio = quality.outOfFrameRatio,
                )
            } finally {
                if (working !== rawOriented) working.release()
            }
        } finally {
            rawOriented.release()
        }
    }

    private fun emptyResult(frame: HybridFrameSpec): NativeFrameResult {
        val width = try { frame.width } catch (_: Exception) { 0.0 }
        val height = try { frame.height } catch (_: Exception) { 0.0 }
        return NativeFrameResult(
            detected = false,
            documentType = "GENERIC",
            confidence = 0.0,
            quad = null,
            frameWidth = width,
            frameHeight = height,
            blurScore = 0.0,
            brightness = 0.0,
            glareRatio = 0.0,
            motionScore = 0.0,
            distanceRatio = 0.0,
            perspectiveSkewDeg = 0.0,
            outOfFrameRatio = 0.0,
        )
    }

    // All detection/quality analysis — and the frameWidth/frameHeight reported back to JS —
    // operate on a working-resolution copy of the oriented frame (long edge capped below),
    // not the raw sensor resolution. This keeps per-frame CPU cost bounded for real-time
    // analysis; the JS/Skia overlay only ever needs frameWidth/frameHeight as a scale
    // reference against the preview view, never true sensor pixels, so this is safe.
    private fun downscaleToWorkingResolution(src: Mat, longEdge: Int): Mat {
        val currentLongEdge = max(src.cols(), src.rows())
        if (currentLongEdge <= longEdge) return src
        val scale = longEdge.toDouble() / currentLongEdge
        val out = Mat()
        Imgproc.resize(src, out, Size(src.cols() * scale, src.rows() * scale), 0.0, 0.0, Imgproc.INTER_AREA)
        return out
    }

    // -----------------------------------------------------------------
    // analyzeFaceFrame — called on every camera frame, selfie-capture mode
    // -----------------------------------------------------------------

    override fun analyzeFaceFrame(frame: HybridFrameSpec): NativeFaceFrameResult {
        return try {
            analyzeFaceFrameInternal(frame)
        } catch (e: Exception) {
            Log.e(TAG, "analyzeFaceFrame failed: ${e.message}", e)
            emptyFaceResult(frame)
        }
    }

    private fun analyzeFaceFrameInternal(frame: HybridFrameSpec): NativeFaceFrameResult {
        // `FrameConversion.toOrientedGrayMat` already applies sensor-rotation + front-camera
        // mirroring while extracting the Y-plane (see its doc comment) — by the time we get
        // `rawOriented` back, it's already in display-orientation space, exactly like
        // `analyzeFrameInternal` above. Detection then runs directly on this oriented Mat, so
        // the resulting box/landmark coordinates are already display-oriented too; no separate
        // post-hoc rotation step is needed here (mirrors analyzeFrame's own convention).
        val rawOriented = FrameConversion.toOrientedGrayMat(frame)
        try {
            val working = downscaleToWorkingResolution(rawOriented, QUALITY_WORKING_LONG_EDGE)
            try {
                val frameWidth = working.cols()
                val frameHeight = working.rows()

                val detection = faceDetector.detect(onnxEngine.faceDetectorSession, working, FACE_MODEL_INPUT_SIZE)
                // Same shared analyzer instance/state as analyzeFrame — quality metrics (blur,
                // brightness, glare, motion) are useful even before a face is found, matching
                // analyzeFrame's own "populate quality metrics regardless of detection" behavior.
                val quality = qualityAnalyzer.analyze(working, quad = null, frameWidth, frameHeight)

                return NativeFaceFrameResult(
                    detected = detection != null,
                    confidence = detection?.confidence ?: 0.0,
                    box = detection?.box,
                    landmarks = detection?.landmarks,
                    frameWidth = frameWidth.toDouble(),
                    frameHeight = frameHeight.toDouble(),
                    blurScore = quality.blurScore,
                    brightness = quality.brightness,
                    glareRatio = quality.glareRatio,
                    motionScore = quality.motionScore,
                )
            } finally {
                if (working !== rawOriented) working.release()
            }
        } finally {
            rawOriented.release()
        }
    }

    private fun emptyFaceResult(frame: HybridFrameSpec): NativeFaceFrameResult {
        val width = try { frame.width } catch (_: Exception) { 0.0 }
        val height = try { frame.height } catch (_: Exception) { 0.0 }
        return NativeFaceFrameResult(
            detected = false,
            confidence = 0.0,
            box = null,
            landmarks = null,
            frameWidth = width,
            frameHeight = height,
            blurScore = 0.0,
            brightness = 0.0,
            glareRatio = 0.0,
            motionScore = 0.0,
        )
    }

    // -----------------------------------------------------------------
    // loadModels
    // -----------------------------------------------------------------

    override fun loadModels(config: ModelPaths): Promise<LoadModelsResult> {
        return Promise.async {
            onnxEngine.load(requireContext(), config)
        }
    }

    // -----------------------------------------------------------------
    // captureAndExtract — called once per capture
    // -----------------------------------------------------------------

    override fun captureAndExtract(photoPath: String, documentType: String, quad: Quad?): Promise<RawOcrResultNative> {
        // `documentType` is intentionally unused here — it only affects which JS-side parser
        // (`parseMrz`/`parseDrivingLicence`/`parseIdCard`) runs over the raw OCR output; native
        // extraction is document-agnostic. Kept as a parameter only because it's part of the spec.
        return Promise.async {
            if (!onnxEngine.isLoaded) {
                throw IllegalStateException(
                    "react-native-doc-scanner: captureAndExtract() called before loadModels() completed.",
                )
            }
            ocrPipeline.extract(requireContext(), onnxEngine, photoPath, quad)
        }
    }

    private fun requireContext(): Context {
        return NitroModules.applicationContext
            ?: throw IllegalStateException(
                "react-native-doc-scanner: NitroModules.applicationContext is null — " +
                    "the host app's ReactApplicationContext isn't available yet.",
            )
    }

    // -----------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------

    override fun dispose() {
        super.dispose()
        onnxEngine.close()
        qualityAnalyzer.resetMotionState()
    }

    companion object {
        private const val TAG = "DocScanner"

        // Mirrors `MODEL_INPUT_SIZE.detectorWidth/Height` in src/constants/thresholds.ts — keep
        // these in sync if that value ever changes.
        private const val MODEL_INPUT_SIZE = 256

        // YuNet's own fixed training/export input size — confirmed directly from the real
        // face_detection_yunet_2023mar.onnx graph (`onnx.load(...).graph.input`): static
        // [1,3,640,640], NOT 320x320 (that was an incorrect assumption during initial
        // implementation, based on common demo defaults rather than this exact export's real
        // shape — caught by inspecting the actual downloaded model file). 640 is already a
        // multiple of 32, so no letterbox padding is needed for the stride-8/16/32 grid math.
        // The flattened output shapes confirm it too: cls_8/obj_8/bbox_8/kps_8 have 6400 =
        // 80*80 = (640/8)*(640/8) entries; cls_16 has 1600 = 40*40 = (640/16)^2; cls_32 has
        // 400 = 20*20 = (640/32)^2. See FaceDetector.
        private const val FACE_MODEL_INPUT_SIZE = 640

        // See downscaleToWorkingResolution's comment above.
        private const val QUALITY_WORKING_LONG_EDGE = 480

        // Mirrors `MODEL_INPUT_SIZE.classifierWidth/Height` in src/constants/thresholds.ts —
        // keep in sync if that value ever changes.
        private const val CLASSIFIER_MODEL_INPUT_SIZE = 224

        // Mirrors `SCANNER_THRESHOLDS.minClassificationConfidence` in
        // src/constants/thresholds.ts — keep in sync if you retune this.
        private const val MIN_CLASSIFICATION_CONFIDENCE = 0.6
    }
}

/**
 * Converts a VisionCamera `Frame` (`HybridFrameSpec`, from `react-native-vision-camera`'s own
 * Nitro spec — see `nitrogen/generated/android/kotlin/com/margelo/nitro/camera/HybridFrameSpec.kt`
 * in that package) into a single-channel grayscale OpenCV `Mat`, already rotated/mirrored into
 * the preview's display orientation.
 *
 * This deliberately uses only `HybridFrameSpec`'s own public members (`width`/`height`/
 * `orientation`/`isMirrored`/`pixelFormat`/`getPlanes()`) rather than downcasting to
 * VisionCamera's internal `NativeFrame`/`ImageProxy` bridge — those members are already
 * everything needed, and staying on the public cross-module Nitro surface means this code
 * doesn't depend on VisionCamera's internal Android implementation details.
 *
 * For all 8-bit YUV pixel formats (the VisionCamera default), plane 0 is always the
 * full-resolution luma (Y) plane, which is exactly what every quality metric in
 * [OpenCvQualityAnalyzer] and both paths in [DocumentDetector] need — this avoids having to
 * de-interleave the (device-specific, semi-planar-or-planar, varying pixelStride) chroma
 * planes at all. Single-plane RGB/RGBA/BGRA frames (`pixelFormat: 'rgb'` in JS) are also
 * supported, converted to gray via OpenCV. 10-bit YUV variants are explicitly rejected with a
 * descriptive error rather than silently misinterpreting the byte layout.
 */
internal object FrameConversion {
    fun toOrientedGrayMat(frame: HybridFrameSpec): Mat {
        val rawWidth = frame.width.toInt()
        val rawHeight = frame.height.toInt()
        if (rawWidth <= 0 || rawHeight <= 0) {
            throw IllegalStateException(
                "react-native-doc-scanner: received an invalid Frame with width=$rawWidth height=$rawHeight.",
            )
        }

        val gray = when {
            isEightBitYuv(frame.pixelFormat) -> extractYPlaneAsGray(frame, rawWidth, rawHeight)
            isRgbLike(frame.pixelFormat) -> extractRgbAsGray(frame, rawWidth, rawHeight)
            else -> throw IllegalStateException(
                "react-native-doc-scanner: unsupported Frame pixelFormat ${frame.pixelFormat} " +
                    "— only 8-bit YUV and RGB/BGR/RGBA pixel formats are supported for analysis.",
            )
        }

        val mirrored = if (frame.isMirrored) {
            val out = Mat()
            Core.flip(gray, out, 1)
            gray.release()
            out
        } else {
            gray
        }

        return when (frame.orientation) {
            CameraOrientation.UP -> mirrored
            CameraOrientation.RIGHT -> Mat().also {
                Core.rotate(mirrored, it, Core.ROTATE_90_CLOCKWISE)
                mirrored.release()
            }
            CameraOrientation.DOWN -> Mat().also {
                Core.rotate(mirrored, it, Core.ROTATE_180)
                mirrored.release()
            }
            CameraOrientation.LEFT -> Mat().also {
                Core.rotate(mirrored, it, Core.ROTATE_90_COUNTERCLOCKWISE)
                mirrored.release()
            }
        }
    }

    private fun isEightBitYuv(format: PixelFormat): Boolean {
        return format == PixelFormat.YUV_420_8_BIT_VIDEO ||
            format == PixelFormat.YUV_420_8_BIT_FULL ||
            format == PixelFormat.YUV_422_8_BIT_VIDEO ||
            format == PixelFormat.YUV_422_8_BIT_FULL ||
            format == PixelFormat.YUV_444_8_BIT_VIDEO
    }

    private fun isRgbLike(format: PixelFormat): Boolean {
        return format == PixelFormat.RGB_RGBA_8_BIT ||
            format == PixelFormat.RGB_BGRA_8_BIT ||
            format == PixelFormat.RGB_RGB_8_BIT
    }

    private fun extractYPlaneAsGray(frame: HybridFrameSpec, width: Int, height: Int): Mat {
        val planes = frame.getPlanes()
        if (planes.isEmpty()) {
            throw IllegalStateException("react-native-doc-scanner: Frame reported zero planes for a YUV pixel format.")
        }
        val yPlane = planes[0]
        val rowStride = yPlane.bytesPerRow.toInt()
        if (rowStride < width) {
            throw IllegalStateException(
                "react-native-doc-scanner: Y plane rowStride ($rowStride) is smaller than frame width ($width).",
            )
        }

        val buffer = yPlane.getPixelBuffer().getBuffer(false)
        buffer.rewind()
        val required = rowStride * height
        if (buffer.remaining() < required) {
            throw IllegalStateException(
                "react-native-doc-scanner: Y plane buffer too small (${buffer.remaining()} bytes, need $required).",
            )
        }
        val bytes = ByteArray(required)
        buffer.get(bytes)

        val padded = Mat(height, rowStride, CvType.CV_8UC1)
        padded.put(0, 0, bytes)
        return if (rowStride == width) {
            padded
        } else {
            val cropped = Mat(padded, Rect(0, 0, width, height)).clone()
            padded.release()
            cropped
        }
    }

    /** Single-plane RGB/RGBA/BGRA frame (VisionCamera `pixelFormat: 'rgb'`) — convert straight to gray. */
    private fun extractRgbAsGray(frame: HybridFrameSpec, width: Int, height: Int): Mat {
        val planes = frame.getPlanes()
        val plane = planes.firstOrNull()
            ?: throw IllegalStateException("react-native-doc-scanner: Frame reported zero planes for an RGB pixel format.")

        val channels = if (frame.pixelFormat == PixelFormat.RGB_RGB_8_BIT) 3 else 4
        val rowStride = plane.bytesPerRow.toInt()
        val paddedWidth = rowStride / channels
        val expectedStride = width * channels

        val buffer = plane.getPixelBuffer().getBuffer(false)
        buffer.rewind()
        val required = rowStride * height
        if (buffer.remaining() < required) {
            throw IllegalStateException(
                "react-native-doc-scanner: RGB plane buffer too small (${buffer.remaining()} bytes, need $required).",
            )
        }
        val bytes = ByteArray(required)
        buffer.get(bytes)

        val matType = if (channels == 3) CvType.CV_8UC3 else CvType.CV_8UC4
        val full = Mat(height, paddedWidth, matType)
        full.put(0, 0, bytes)
        val cropped = if (rowStride == expectedStride) full else Mat(full, Rect(0, 0, width, height)).clone()
        if (rowStride != expectedStride) full.release()

        val gray = Mat()
        val code = when (frame.pixelFormat) {
            PixelFormat.RGB_BGRA_8_BIT -> Imgproc.COLOR_BGRA2GRAY
            PixelFormat.RGB_RGBA_8_BIT -> Imgproc.COLOR_RGBA2GRAY
            else -> Imgproc.COLOR_RGB2GRAY
        }
        Imgproc.cvtColor(cropped, gray, code)
        cropped.release()
        return gray
    }
}
