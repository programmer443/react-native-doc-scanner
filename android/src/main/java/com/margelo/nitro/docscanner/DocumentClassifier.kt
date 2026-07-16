package com.margelo.nitro.docscanner

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.exp

/** Result of one classification pass. */
internal data class RawClassification(val documentType: String, val confidence: Double)

/**
 * Document-type classifier (PASSPORT/DRIVING_LICENCE/ID_CARD/RESIDENCE_PERMIT/VISA vs GENERIC) —
 * optional, see docs/MODEL_TRAINING.md §6. No off-the-shelf Apache/MIT-licensed ONNX model with
 * this exact taxonomy exists today (checked HF `prithivMLmods/Document-Type-Detection`: wrong
 * classes, general document genre not ID type; `logasanjeev/indian-id-validator`: YOLO11-based,
 * the same AGPL-vs-commercial tension already flagged for the main detector in
 * docs/MODEL_TRAINING.md §1) — this expects a model you've fine-tuned yourself and exported to
 * ONNX (see §6 for the recommended path).
 *
 * Same "grayscale replicated to 3 identical channels" hot-path tradeoff as [DocumentDetector]/
 * [FaceDetector] — see their doc comments; this runs on the same already-oriented grayscale Mat.
 * Crops to the detected quad's bounding box before resizing (higher-signal input than the whole
 * frame), falling back to the whole frame if no quad was found yet.
 */
class DocumentClassifier {
    internal fun classify(session: OrtSession?, orientedGray: Mat, quad: Quad?, modelInputSize: Int): RawClassification? {
        if (session == null) return null
        return try {
            runOnnxClassifier(session, orientedGray, quad, modelInputSize)
        } catch (e: Exception) {
            Log.w(TAG, "Document classifier failed: ${e.message}")
            null
        }
    }

    private fun runOnnxClassifier(session: OrtSession, orientedGray: Mat, quad: Quad?, modelInputSize: Int): RawClassification? {
        val env = OrtEnvironment.getEnvironment()
        val cropped = cropToQuad(orientedGray, quad)
        val squash = SquashResize.build(cropped, modelInputSize)
        if (cropped !== orientedGray) cropped.release()
        try {
            val bgr = Mat()
            Imgproc.cvtColor(squash.mat, bgr, Imgproc.COLOR_GRAY2BGR)

            val plane = modelInputSize * modelInputSize
            val pixelBytes = ByteArray(plane * 3)
            bgr.get(0, 0, pixelBytes)
            bgr.release()

            // NCHW, ImageNet mean/std normalised RGB — standard preprocessing for an
            // ImageNet-pretrained backbone (MobileNetV3/EfficientNet-Lite). Update alongside
            // whatever backbone docs/MODEL_TRAINING.md §6 actually documents if it differs.
            val chw = FloatArray(3 * plane)
            for (i in 0 until plane) {
                val b = (pixelBytes[i * 3].toInt() and 0xFF) / 255f
                val g = (pixelBytes[i * 3 + 1].toInt() and 0xFF) / 255f
                val r = (pixelBytes[i * 3 + 2].toInt() and 0xFF) / 255f
                chw[i] = (r - MEAN[0]) / STD[0]
                chw[plane + i] = (g - MEAN[1]) / STD[1]
                chw[plane * 2 + i] = (b - MEAN[2]) / STD[2]
            }

            val inputName = session.inputNames.iterator().next()
            OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(chw),
                longArrayOf(1, 3, modelInputSize.toLong(), modelInputSize.toLong()),
            ).use { inputTensor ->
                session.run(mapOf(inputName to inputTensor)).use { results ->
                    val outputName = session.outputNames.iterator().next()
                    val outputValue = results.get(outputName)
                    if (!outputValue.isPresent) {
                        Log.w(TAG, "Classifier session produced no output named \"$outputName\".")
                        return null
                    }
                    val tensor = outputValue.get() as? OnnxTensor ?: run {
                        Log.w(TAG, "Classifier output was not an OnnxTensor.")
                        return null
                    }
                    val buffer = tensor.floatBuffer
                    if (buffer.remaining() < LABELS.size) return null

                    var bestIdx = 0
                    var bestVal = buffer.get(0)
                    for (i in 1 until LABELS.size) {
                        val v = buffer.get(i)
                        if (v > bestVal) {
                            bestVal = v
                            bestIdx = i
                        }
                    }

                    // Softmax-normalise just the winning logit against the full class set for a
                    // confidence estimate — same trick used by OcrPipeline's CTC decode, avoids a
                    // full softmax pass. Handles both raw logits and an already-softmaxed output.
                    var sumExp = 0.0
                    for (i in LABELS.indices) {
                        sumExp += exp((buffer.get(i) - bestVal).toDouble())
                    }
                    val confidence = if (sumExp > 0.0) 1.0 / sumExp else 0.0

                    return RawClassification(documentType = LABELS[bestIdx], confidence = confidence)
                }
            }
        } finally {
            squash.mat.release()
        }
    }

    /**
     * Crops to `quad`'s axis-aligned bounding box (clamped to the Mat's bounds), or returns [src]
     * unchanged if `quad` is null (classify the whole frame — same fallback used when no quad has
     * been found yet). Returns an owned clone, same `Mat(source, Rect(...)).clone()` convention as
     * HybridDocScanner's other crop sites.
     */
    private fun cropToQuad(src: Mat, quad: Quad?): Mat {
        if (quad == null) return src
        val xs = listOf(quad.topLeft.x, quad.topRight.x, quad.bottomRight.x, quad.bottomLeft.x)
        val ys = listOf(quad.topLeft.y, quad.topRight.y, quad.bottomRight.y, quad.bottomLeft.y)
        val minX = xs.min().coerceIn(0.0, src.cols().toDouble())
        val minY = ys.min().coerceIn(0.0, src.rows().toDouble())
        val maxX = xs.max().coerceIn(0.0, src.cols().toDouble())
        val maxY = ys.max().coerceIn(0.0, src.rows().toDouble())
        val width = (maxX - minX).toInt()
        val height = (maxY - minY).toInt()
        if (width <= 0 || height <= 0) return src
        return Mat(src, Rect(minX.toInt(), minY.toInt(), width, height)).clone()
    }

    companion object {
        private const val TAG = "DocScanner/DocumentClassifier"

        // Class order the classifier's output vector is read against — MUST exactly match the
        // label order the model was trained/exported with, or predictions will silently point at
        // the wrong document type. "GENERIC" is included as the "not a recognised ID document" /
        // background class.
        private val LABELS = arrayOf("PASSPORT", "DRIVING_LICENCE", "ID_CARD", "RESIDENCE_PERMIT", "VISA", "GENERIC")

        // ImageNet mean/std — matches the standard preprocessing for an ImageNet-pretrained
        // backbone (MobileNetV3/EfficientNet-Lite). Update alongside whatever backbone
        // docs/MODEL_TRAINING.md §6 actually documents.
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
