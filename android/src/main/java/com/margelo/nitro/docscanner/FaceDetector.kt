package com.margelo.nitro.docscanner

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** Result of one face-detection pass, in the same pixel space as the Mat that was analyzed. */
internal data class RawFaceDetection(
    val box: BoundingBox,
    val landmarks: FaceLandmarks,
    val confidence: Double,
)

/** One decoded candidate face, still in 640x640 model-input pixel space, pre-NMS. */
private data class FaceCandidate(
    val x: Double,
    val y: Double,
    val w: Double,
    val h: Double,
    val score: Double,
    // Index order: 0=rightEye, 1=leftEye, 2=noseTip, 3=rightMouthCorner, 4=leftMouthCorner
    // (subject's own left/right, per OpenCV's FaceDetectorYN convention).
    val landmarkX: DoubleArray,
    val landmarkY: DoubleArray,
)

/**
 * YuNet ONNX face detector (MIT-licensed, from OpenCV's `opencv_zoo`), used for the selfie
 * capture guide's real-time face box + 5-point landmarks. Chosen over SCRFD because SCRFD's
 * pretrained weights require a paid InsightFace commercial license.
 *
 * Decode algorithm ported directly from OpenCV's own `modules/objdetect/src/face_detect.cpp`
 * (the reference implementation `cv::FaceDetectorYN` wraps) — see docs/MODEL_TRAINING.md §5.
 * Unlike [DocumentDetector], this is a single fixed ONNX graph with a well-known, stable output
 * contract (no heatmap/direct-regression branching, no classical fallback): YuNet always
 * produces the same 12 named output tensors, so a plain single-path decode is correct here.
 */
class FaceDetector {
    internal fun detect(session: OrtSession?, orientedGray: Mat, modelInputSize: Int): RawFaceDetection? {
        if (session == null) return null
        return try {
            runOnnxDetector(session, orientedGray, modelInputSize)
        } catch (e: Exception) {
            Log.w(TAG, "YuNet face detector failed: ${e.message}")
            null
        }
    }

    private fun runOnnxDetector(session: OrtSession, gray: Mat, modelInputSize: Int): RawFaceDetection? {
        val env = OrtEnvironment.getEnvironment()
        val squash = SquashResize.build(gray, modelInputSize)
        try {
            val bgr = Mat()
            Imgproc.cvtColor(squash.mat, bgr, Imgproc.COLOR_GRAY2BGR)

            val plane = modelInputSize * modelInputSize
            val pixelBytes = ByteArray(plane * 3)
            bgr.get(0, 0, pixelBytes)
            bgr.release()

            // NCHW, raw 0..255 pixel values, BGR channel order PRESERVED (deliberately NOT
            // reordered to RGB, deliberately NOT divided by 255 or mean/std-normalised) — this
            // is the opposite convention from DocumentDetector's DocAligner preprocessing right
            // above, and that's intentional, not a copy/paste slip: confirmed against OpenCV's
            // own face_detect.cpp, which calls `dnn::blobFromImage(pad_image)` with no
            // scalefactor/mean args, i.e. raw BGR bytes cast to float.
            //
            // Like DocumentDetector, the input here is grayscale (extracted from the frame's
            // Y-plane for per-frame cost, not genuine RGB) replicated to fake-BGR via
            // COLOR_GRAY2BGR — R/G/B (here B/G/R) carry no distinct information regardless of
            // ordering. Same accepted 30-60fps hot-path speed/accuracy tradeoff as the document
            // detector, flagged here for the same reason.
            val chw = FloatArray(3 * plane)
            for (i in 0 until plane) {
                val b = (pixelBytes[i * 3].toInt() and 0xFF).toFloat()
                val g = (pixelBytes[i * 3 + 1].toInt() and 0xFF).toFloat()
                val r = (pixelBytes[i * 3 + 2].toInt() and 0xFF).toFloat()
                chw[i] = b
                chw[plane + i] = g
                chw[plane * 2 + i] = r
            }

            val inputName = session.inputNames.iterator().next()
            OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(chw),
                longArrayOf(1, 3, modelInputSize.toLong(), modelInputSize.toLong()),
            ).use { inputTensor ->
                session.run(mapOf(inputName to inputTensor)).use { results ->
                    val candidates = mutableListOf<FaceCandidate>()
                    for (stride in STRIDES) {
                        candidates.addAll(decodeStride(results, stride, modelInputSize))
                    }
                    if (candidates.isEmpty()) return null

                    val kept = greedyNms(candidates, NMS_THRESHOLD)
                    // Selfie-specific simplification: only one dominant face is expected/wanted,
                    // so just take the highest-scoring survivor — NativeFaceFrameResult only has
                    // room for one box/landmarks set anyway.
                    val best = kept.maxByOrNull { it.score } ?: return null

                    val box = BoundingBox(
                        x = (best.x / squash.scaleX).coerceIn(0.0, gray.cols().toDouble()),
                        y = (best.y / squash.scaleY).coerceIn(0.0, gray.rows().toDouble()),
                        width = best.w / squash.scaleX,
                        height = best.h / squash.scaleY,
                    )
                    val landmarks = FaceLandmarks(
                        rightEye = mapPoint(best, 0, squash),
                        leftEye = mapPoint(best, 1, squash),
                        noseTip = mapPoint(best, 2, squash),
                        rightMouthCorner = mapPoint(best, 3, squash),
                        leftMouthCorner = mapPoint(best, 4, squash),
                    )
                    return RawFaceDetection(box = box, landmarks = landmarks, confidence = best.score)
                }
            }
        } finally {
            squash.mat.release()
        }
    }

    private fun mapPoint(candidate: FaceCandidate, index: Int, squash: SquashResize): Point {
        return Point(
            x = candidate.landmarkX[index] / squash.scaleX,
            y = candidate.landmarkY[index] / squash.scaleY,
        )
    }

    /**
     * Decodes one stride level's 4 named output tensors (`cls_$stride`, `obj_$stride`,
     * `bbox_$stride`, `kps_$stride`) into candidate faces, still in [inputSize]x[inputSize]
     * model-input pixel space. Grid indexing is row-major (`idx = r * cols + c`), matching how
     * YuNet's anchor-free head lays out its flattened per-cell outputs.
     */
    private fun decodeStride(results: OrtSession.Result, stride: Int, inputSize: Int): List<FaceCandidate> {
        val cols = inputSize / stride
        val rows = inputSize / stride
        val cls = namedFloatBuffer(results, "cls_$stride")
        val obj = namedFloatBuffer(results, "obj_$stride")
        val bbox = namedFloatBuffer(results, "bbox_$stride")
        val kps = namedFloatBuffer(results, "kps_$stride")

        val candidates = mutableListOf<FaceCandidate>()
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val idx = r * cols + c
                val clsScore = cls.get(idx).toDouble().coerceIn(0.0, 1.0)
                val objScore = obj.get(idx).toDouble().coerceIn(0.0, 1.0)
                val score = sqrt(clsScore * objScore)
                if (score < SCORE_THRESHOLD) continue

                val bx = bbox.get(idx * 4).toDouble()
                val by = bbox.get(idx * 4 + 1).toDouble()
                val bw = bbox.get(idx * 4 + 2).toDouble()
                val bh = bbox.get(idx * 4 + 3).toDouble()

                val cx = (c + bx) * stride
                val cy = (r + by) * stride
                val w = exp(bw) * stride
                val h = exp(bh) * stride
                val x1 = cx - w / 2.0
                val y1 = cy - h / 2.0

                val landmarkX = DoubleArray(5)
                val landmarkY = DoubleArray(5)
                for (n in 0 until 5) {
                    landmarkX[n] = (kps.get(idx * 10 + 2 * n).toDouble() + c) * stride
                    landmarkY[n] = (kps.get(idx * 10 + 2 * n + 1).toDouble() + r) * stride
                }

                candidates.add(FaceCandidate(x1, y1, w, h, score, landmarkX, landmarkY))
            }
        }
        return candidates
    }

    private fun namedFloatBuffer(results: OrtSession.Result, name: String): FloatBuffer {
        val value = results.get(name)
        if (!value.isPresent) {
            throw IllegalStateException("YuNet session produced no output named \"$name\".")
        }
        val tensor = value.get() as? OnnxTensor
            ?: throw IllegalStateException("YuNet output \"$name\" was not an OnnxTensor.")
        return tensor.floatBuffer
    }

    /**
     * Standard greedy IoU-based NMS: process candidates in descending score order, keep a
     * candidate only if its IoU with every already-kept candidate is below [iouThreshold].
     * Hand-rolled rather than relying on an OpenCV NMSBoxes helper — this is ~15 lines and
     * avoids a dependency on a Java API shape that may not match these plain double boxes.
     */
    private fun greedyNms(candidates: List<FaceCandidate>, iouThreshold: Double): List<FaceCandidate> {
        val sorted = candidates.sortedByDescending { it.score }
        val kept = mutableListOf<FaceCandidate>()
        for (candidate in sorted) {
            val overlapsKept = kept.any { iou(it, candidate) >= iouThreshold }
            if (!overlapsKept) kept.add(candidate)
        }
        return kept
    }

    private fun iou(a: FaceCandidate, b: FaceCandidate): Double {
        val interX1 = max(a.x, b.x)
        val interY1 = max(a.y, b.y)
        val interX2 = min(a.x + a.w, b.x + b.w)
        val interY2 = min(a.y + a.h, b.y + b.h)
        val interArea = max(0.0, interX2 - interX1) * max(0.0, interY2 - interY1)
        val unionArea = a.w * a.h + b.w * b.h - interArea
        return if (unionArea <= 0.0) 0.0 else interArea / unionArea
    }

    companion object {
        private const val TAG = "DocScanner/FaceDetector"
        private val STRIDES = intArrayOf(8, 16, 32)
        private const val SCORE_THRESHOLD = 0.7
        private const val NMS_THRESHOLD = 0.3
    }
}
