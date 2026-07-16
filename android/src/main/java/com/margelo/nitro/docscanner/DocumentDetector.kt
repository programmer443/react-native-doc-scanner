package com.margelo.nitro.docscanner

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.max

/** Result of one detection pass, in the same pixel space as the Mat that was analyzed. */
internal data class RawDetection(val quad: Quad, val confidence: Double)

/**
 * Locates a document's 4 corners in an already-oriented grayscale frame.
 *
 * Primary path: the DocAligner ONNX corner detector (`runOnnxDetector`). Its exact exported
 * output layout is not pinned down upstream (see docs/MODEL_TRAINING.md), so this branches at
 * runtime on the *actual* output tensor shape reported by the loaded session:
 *  - `[1, 4, H, W]` (4 per-corner heatmaps)      -> per-channel argmax.
 *  - 8 total elements (`[1,8]`/`[1,4,2]`/`[8]`)  -> direct (x,y) pair regression.
 *  - anything else, a missing session, or a runtime exception
 *                                                 -> classical OpenCV contour fallback.
 *
 * Fallback: `runContourFallback` is the standard "OpenCV document scanner" algorithm
 * (blur -> Canny -> contours -> approxPolyDP -> largest convex quadrilateral). It is a
 * complete, independently-functioning detector in its own right, not a stub — it runs
 * whenever the ONNX path is unavailable or produces something we don't recognize.
 *
 * `documentType` is intentionally not produced here — this module has no separate document
 * *classifier*, only a corner detector. `HybridDocScanner` always reports "GENERIC" for it;
 * the JS `OcrService` branches on the app-supplied `documentType` argument to `captureAndExtract`
 * for field parsing, not on this value (it's informational only, e.g. for a confidence badge).
 */
class DocumentDetector {
    internal fun detect(session: OrtSession?, orientedGray: Mat, modelInputSize: Int): RawDetection? {
        if (session != null) {
            try {
                val result = runOnnxDetector(session, orientedGray, modelInputSize)
                if (result != null) return result
            } catch (e: Exception) {
                Log.w(TAG, "ONNX detector failed, falling back to contour detection: ${e.message}")
            }
        }
        return runContourFallback(orientedGray)
    }

    // ---------------------------------------------------------------------
    // Primary: DocAligner ONNX
    // ---------------------------------------------------------------------

    private fun runOnnxDetector(session: OrtSession, gray: Mat, modelInputSize: Int): RawDetection? {
        val env = OrtEnvironment.getEnvironment()
        val squash = SquashResize.build(gray, modelInputSize)
        try {
            val bgr = Mat()
            Imgproc.cvtColor(squash.mat, bgr, Imgproc.COLOR_GRAY2BGR)

            val plane = modelInputSize * modelInputSize
            val pixelBytes = ByteArray(plane * 3)
            bgr.get(0, 0, pixelBytes)
            bgr.release()

            // NCHW, 0..1 scale, no mean/std normalisation — confirmed against DocAligner's
            // actual heatmap_reg/infer.py: `img[None] / 255.`, no further normalisation.
            // Channel order is moot here specifically: the input is grayscale (extracted
            // from the frame's Y-plane for per-frame cost, not the original RGB DocAligner
            // was trained on — see toOrientedGrayMat) replicated to 3 identical channels,
            // so R/G/B carry no distinct information regardless of ordering. This is a
            // deliberate speed/accuracy tradeoff for the 30-60fps hot path, not an oversight
            // — a genuinely color-fed detector pass would need a separate, slower path.
            val chw = FloatArray(3 * plane)
            for (i in 0 until plane) {
                val b = (pixelBytes[i * 3].toInt() and 0xFF) / 255f
                val g = (pixelBytes[i * 3 + 1].toInt() and 0xFF) / 255f
                val r = (pixelBytes[i * 3 + 2].toInt() and 0xFF) / 255f
                chw[i] = r
                chw[plane + i] = g
                chw[plane * 2 + i] = b
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
                        Log.w(TAG, "DocAligner session produced no output named \"$outputName\".")
                        return null
                    }
                    val tensor = outputValue.get() as? OnnxTensor ?: run {
                        Log.w(TAG, "DocAligner output was not an OnnxTensor.")
                        return null
                    }
                    val shape = tensor.info.shape
                    val totalElements = shape.fold(1L) { acc, d -> acc * max(d, 1L) }
                    val buffer = tensor.floatBuffer

                    val (modelPoints, confidence) = when {
                        shape.size == 4 && shape[1] == 4L -> decodeHeatmap(buffer, shape, modelInputSize)
                        totalElements == 8L -> decodeDirectRegression(buffer, modelInputSize)
                        else -> {
                            Log.w(TAG, "Unrecognized DocAligner output shape ${shape.joinToString()}; falling back to contour detection.")
                            return null
                        }
                    }

                    val framePoints = modelPoints.map { (mx, my) ->
                        Point(
                            x = (mx / squash.scaleX).coerceIn(0.0, gray.cols().toDouble()),
                            y = (my / squash.scaleY).coerceIn(0.0, gray.rows().toDouble()),
                        )
                    }
                    return RawDetection(orderQuadPoints(framePoints), confidence)
                }
            }
        } finally {
            squash.mat.release()
        }
    }

    private fun decodeHeatmap(buffer: FloatBuffer, shape: LongArray, modelInputSize: Int): Pair<List<Pair<Double, Double>>, Double> {
        val h = shape[2].toInt()
        val w = shape[3].toInt()
        val corners = mutableListOf<Pair<Double, Double>>()
        var peakSum = 0.0
        for (c in 0 until 4) {
            var bestVal = Float.NEGATIVE_INFINITY
            var bestIdx = 0
            val base = c * h * w
            for (idx in 0 until h * w) {
                val v = buffer.get(base + idx)
                if (v > bestVal) {
                    bestVal = v
                    bestIdx = idx
                }
            }
            val py = bestIdx / w
            val px = bestIdx % w
            val modelX = (px + 0.5) * (modelInputSize.toDouble() / w)
            val modelY = (py + 0.5) * (modelInputSize.toDouble() / h)
            corners.add(Pair(modelX, modelY))
            peakSum += sigmoidIfLogit(bestVal.toDouble())
        }
        return Pair(corners, (peakSum / 4.0).coerceIn(0.0, 1.0))
    }

    private fun decodeDirectRegression(buffer: FloatBuffer, modelInputSize: Int): Pair<List<Pair<Double, Double>>, Double> {
        val corners = mutableListOf<Pair<Double, Double>>()
        for (i in 0 until 4) {
            val x = buffer.get(i * 2).toDouble()
            val y = buffer.get(i * 2 + 1).toDouble()
            // Direct-regression exports commonly normalise outputs to 0..1; detect that
            // range and scale to model-input pixel space, otherwise assume raw pixels.
            val scaledX = if (x in -0.01..1.01) x * modelInputSize else x
            val scaledY = if (y in -0.01..1.01) y * modelInputSize else y
            corners.add(Pair(scaledX, scaledY))
        }
        // No activation signal to derive a confidence from for direct regression outputs —
        // a documented reasonable constant, per spec.
        return Pair(corners, 0.7)
    }

    private fun sigmoidIfLogit(v: Double): Double {
        // Heatmap peaks are usually already 0..1 (sigmoid applied during export). If a value
        // clearly outside that range shows up, treat it as a raw logit and apply sigmoid.
        return if (v in 0.0..1.0) v else 1.0 / (1.0 + Math.exp(-v))
    }

    // ---------------------------------------------------------------------
    // Fallback: classical OpenCV contour detection
    // ---------------------------------------------------------------------

    private fun runContourFallback(gray: Mat): RawDetection? {
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        val edges = Mat()
        Imgproc.Canny(blurred, edges, 75.0, 200.0)
        Imgproc.dilate(edges, edges, Mat())
        blurred.release()

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
        edges.release()
        hierarchy.release()

        val frameArea = (gray.rows() * gray.cols()).toDouble()
        var best: RawDetection? = null
        var bestArea = 0.0

        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            if (area >= frameArea * 0.1 && area > bestArea) {
                val contour2f = MatOfPoint2f(*contour.toArray())
                val arcLength = Imgproc.arcLength(contour2f, true)
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(contour2f, approx, 0.02 * arcLength, true)
                contour2f.release()

                val approxPoints = approx.toArray()
                approx.release()

                if (approxPoints.size == 4) {
                    val intPoints = MatOfPoint(*approxPoints)
                    val isConvex = Imgproc.isContourConvex(intPoints)
                    intPoints.release()

                    if (isConvex) {
                        val points = approxPoints.map { Point(x = it.x, y = it.y) }
                        val quad = orderQuadPoints(points)
                        val areaRatio = (area / frameArea).coerceIn(0.0, 1.0)
                        val skew = OpenCvQualityAnalyzer.computePerspectiveSkewDeg(quad)
                        val angleScore = (1.0 - (skew / 45.0)).coerceIn(0.0, 1.0)
                        val areaScore = (areaRatio / 0.5).coerceIn(0.0, 1.0)
                        val confidence = (0.5 * areaScore + 0.5 * angleScore).coerceIn(0.0, 1.0)
                        best = RawDetection(quad, confidence)
                        bestArea = area
                    }
                }
            }
            contour.release()
        }

        return best
    }

    companion object {
        private const val TAG = "DocScanner/DocumentDetector"

        /**
         * Orders 4 arbitrary points (from either the ONNX or contour path, in unknown winding
         * order) into topLeft/topRight/bottomRight/bottomLeft using the standard sum/difference
         * heuristic: min(x+y) = topLeft, max(x+y) = bottomRight, min(y-x) = topRight,
         * max(y-x) = bottomLeft. Robust regardless of the source's original point order.
         */
        fun orderQuadPoints(points: List<Point>): Quad {
            require(points.size == 4) { "orderQuadPoints requires exactly 4 points, got ${points.size}" }
            val bySum = points.sortedBy { it.x + it.y }
            val byDiff = points.sortedBy { it.y - it.x }
            return Quad(
                topLeft = bySum.first(),
                topRight = byDiff.first(),
                bottomRight = bySum.last(),
                bottomLeft = byDiff.last(),
            )
        }
    }
}

/**
 * A plain (aspect-distorting) resize of a Mat to a square [targetSize] —
 * deliberately NOT a letterbox. Matches DocAligner's own preprocessing
 * (`cb.imresize(img, size=img_size_infer)`, no padding): letterboxing here
 * would feed the model a different distribution than it was trained/exported
 * on, and — more visibly — the inverse coordinate mapping back to frame
 * space would be systematically wrong for any non-square frame (i.e. every
 * camera frame).
 */
internal class SquashResize private constructor(
    val mat: Mat,
    val scaleX: Double,
    val scaleY: Double,
) {
    companion object {
        fun build(src: Mat, targetSize: Int): SquashResize {
            val scaleX = targetSize.toDouble() / src.cols()
            val scaleY = targetSize.toDouble() / src.rows()

            val resized = Mat()
            Imgproc.resize(src, resized, Size(targetSize.toDouble(), targetSize.toDouble()))

            return SquashResize(resized, scaleX, scaleY)
        }
    }
}
