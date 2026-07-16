package com.margelo.nitro.docscanner

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log

/**
 * Owns the ONNX Runtime environment and the sessions this module needs: the DocAligner
 * document-corner detector, the three-stage RapidOCR (PP-OCR) pipeline used by `OcrPipeline`
 * (text detection (DB), orientation classification, and text recognition (CTC)), the
 * optional YuNet face detector used by `analyzeFaceFrame` for the selfie capture guide, and the
 * optional document-type classifier used by `analyzeFrame` (see `DocumentClassifier`).
 *
 * Sessions are created directly from in-memory byte arrays
 * (`OrtEnvironment.createSession(ByteArray, SessionOptions)`) rather than
 * filesystem paths, since `bundle://` models live inside the APK's assets
 * and are not directly addressable as a `File` — see `ModelPathResolver`.
 */
class OnnxEngine {
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()

    var detectorSession: OrtSession? = null
        private set
    var ocrDetSession: OrtSession? = null
        private set
    var ocrClsSession: OrtSession? = null
        private set
    var ocrRecSession: OrtSession? = null
        private set
    var faceDetectorSession: OrtSession? = null
        private set
    var classifierSession: OrtSession? = null
        private set
    var charset: List<String> = emptyList()
        private set

    var detectorVersion: String = ""
        private set
    var ocrVersion: String = ""
        private set
    var faceDetectorVersion: String = ""
        private set
    var classifierVersion: String = ""
        private set

    val isLoaded: Boolean
        get() = detectorSession != null && ocrDetSession != null && ocrClsSession != null && ocrRecSession != null

    /**
     * Loads (or reloads) the four required models (detector + 3-stage OCR), plus the optional
     * YuNet face detector when `config.faceDetectorModelPath` is non-empty. Runs synchronously
     * on whatever thread it's called from — `HybridDocScanner` wraps this in `Promise.async` so
     * it never blocks the JS thread. Throws a real, descriptive exception (which nitrogen turns
     * into a rejected JS Promise) on any failure of the four *required* models — missing asset,
     * corrupt/incompatible ONNX graph, etc. — rather than returning a silent `success: false`.
     * The face detector is intentionally NOT part of that contract: it's optional, so a missing
     * or corrupt face model never fails this call (see the try/catch around it below) — it just
     * leaves `faceDetectorSession` null and `faceDetectorVersion` empty, and `analyzeFaceFrame`
     * always reports `detected: false` in that case.
     */
    fun load(context: Context, config: ModelPaths): LoadModelsResult {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
        }

        val detectorBytes = ModelPathResolver.readBytes(context, config.detectorModelPath)
        val ocrDetBytes = ModelPathResolver.readBytes(context, config.ocrDetModelPath)
        val ocrClsBytes = ModelPathResolver.readBytes(context, config.ocrClsModelPath)
        val ocrRecBytes = ModelPathResolver.readBytes(context, config.ocrRecModelPath)
        val charsetLines = ModelPathResolver.readCharset(context, config.ocrRecCharsetPath)

        if (charsetLines.isEmpty()) {
            throw IllegalStateException(
                "react-native-doc-scanner: charset file \"${config.ocrRecCharsetPath}\" resolved to zero " +
                    "characters — OCR recognition cannot decode anything without it.",
            )
        }

        val newDetector = createSessionOrThrow(detectorBytes, options, "detector", config.detectorModelPath)
        val newOcrDet = try {
            createSessionOrThrow(ocrDetBytes, options, "OCR det", config.ocrDetModelPath)
        } catch (e: Exception) {
            newDetector.close()
            throw e
        }
        val newOcrCls = try {
            createSessionOrThrow(ocrClsBytes, options, "OCR cls", config.ocrClsModelPath)
        } catch (e: Exception) {
            newDetector.close()
            newOcrDet.close()
            throw e
        }
        val newOcrRec = try {
            createSessionOrThrow(ocrRecBytes, options, "OCR rec", config.ocrRecModelPath)
        } catch (e: Exception) {
            newDetector.close()
            newOcrDet.close()
            newOcrCls.close()
            throw e
        }

        // Face detector: optional, additive, and never allowed to fail the overall load — the
        // four required sessions above already succeeded and must keep working even if this
        // model is missing/corrupt/absent (e.g. not yet fetched by scripts/fetch-models.sh).
        var newFaceDetector: OrtSession? = null
        var newFaceDetectorVersion = ""
        if (config.faceDetectorModelPath.isNotEmpty()) {
            try {
                val faceBytes = ModelPathResolver.readBytes(context, config.faceDetectorModelPath)
                newFaceDetector = createSessionOrThrow(faceBytes, options, "face detector", config.faceDetectorModelPath)
                newFaceDetectorVersion = ModelPathResolver.versionLabel(config.faceDetectorModelPath, faceBytes)
            } catch (e: Exception) {
                Log.w(
                    TAG,
                    "Face detector model failed to load (non-fatal — analyzeFaceFrame will report " +
                        "detected=false until a valid model is configured): ${e.message}",
                )
                newFaceDetector?.close()
                newFaceDetector = null
                newFaceDetectorVersion = ""
            }
        }

        // Document-type classifier: same optional/additive/never-fails-the-load contract as the
        // face detector above.
        var newClassifier: OrtSession? = null
        var newClassifierVersion = ""
        if (config.classifierModelPath.isNotEmpty()) {
            try {
                val classifierBytes = ModelPathResolver.readBytes(context, config.classifierModelPath)
                newClassifier = createSessionOrThrow(classifierBytes, options, "classifier", config.classifierModelPath)
                newClassifierVersion = ModelPathResolver.versionLabel(config.classifierModelPath, classifierBytes)
            } catch (e: Exception) {
                Log.w(
                    TAG,
                    "Document classifier model failed to load (non-fatal — documentType will stay " +
                        "GENERIC until a valid model is configured): ${e.message}",
                )
                newClassifier?.close()
                newClassifier = null
                newClassifierVersion = ""
            }
        }

        // Every session constructed successfully — swap the old ones out atomically-ish
        // (single-threaded call site) and close them to free native memory.
        detectorSession?.close()
        ocrDetSession?.close()
        ocrClsSession?.close()
        ocrRecSession?.close()
        faceDetectorSession?.close()
        classifierSession?.close()

        detectorSession = newDetector
        ocrDetSession = newOcrDet
        ocrClsSession = newOcrCls
        ocrRecSession = newOcrRec
        faceDetectorSession = newFaceDetector
        classifierSession = newClassifier
        charset = charsetLines

        detectorVersion = ModelPathResolver.versionLabel(config.detectorModelPath, detectorBytes)
        ocrVersion = ModelPathResolver.versionLabel(config.ocrRecModelPath, ocrRecBytes)
        faceDetectorVersion = newFaceDetectorVersion
        classifierVersion = newClassifierVersion

        Log.i(
            TAG,
            "Loaded ONNX models — detector=$detectorVersion ocr=$ocrVersion charset=${charset.size} chars " +
                "faceDetector=${if (faceDetectorVersion.isEmpty()) "(not configured)" else faceDetectorVersion} " +
                "classifier=${if (classifierVersion.isEmpty()) "(not configured)" else classifierVersion}",
        )

        return LoadModelsResult(
            success = true,
            detectorVersion = detectorVersion,
            ocrVersion = ocrVersion,
            faceDetectorVersion = faceDetectorVersion,
            classifierVersion = classifierVersion,
        )
    }

    private fun createSessionOrThrow(
        bytes: ByteArray,
        options: OrtSession.SessionOptions,
        label: String,
        path: String,
    ): OrtSession {
        return try {
            environment.createSession(bytes, options)
        } catch (e: Exception) {
            throw IllegalStateException(
                "react-native-doc-scanner: failed to create ONNX Runtime session for the $label model " +
                    "(\"$path\"): ${e.message}",
                e,
            )
        }
    }

    /** Releases all native ONNX Runtime resources — called from `HybridDocScanner.dispose()`. */
    fun close() {
        detectorSession?.close()
        ocrDetSession?.close()
        ocrClsSession?.close()
        ocrRecSession?.close()
        faceDetectorSession?.close()
        classifierSession?.close()
        detectorSession = null
        ocrDetSession = null
        ocrClsSession = null
        ocrRecSession = null
        faceDetectorSession = null
        classifierSession = null
    }

    companion object {
        private const val TAG = "DocScanner/OnnxEngine"
    }
}
