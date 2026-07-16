package com.margelo.nitro.docscanner

import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Loads OpenCV's native library exactly once for the process. `OpenCVLoader.initLocal()`
 * statically links against the `org.opencv:opencv` Maven Central AAR (no external "OpenCV
 * Manager" app required). Every entry point that touches `org.opencv.*` calls
 * [ensureInitialized] first; it's cheap to call repeatedly once initialized.
 */
object OpenCvBootstrap {
    private const val TAG = "DocScanner/OpenCV"

    @Volatile
    private var initialized = false

    @Synchronized
    fun ensureInitialized() {
        if (initialized) return
        initialized = try {
            OpenCVLoader.initLocal()
        } catch (t: Throwable) {
            Log.e(TAG, "OpenCVLoader.initLocal() threw an exception", t)
            false
        }
        if (initialized) {
            Log.i(TAG, "OpenCV native library loaded (${Core.VERSION}).")
        } else {
            Log.e(
                TAG,
                "OpenCVLoader.initLocal() returned false — OpenCV failed to load. " +
                    "All quality analysis and document detection will be unavailable.",
            )
        }
    }
}

/** Flat, non-Nitro bag of the quality metrics this analyzer produces for one frame. */
internal data class QualityMetrics(
    val blurScore: Double,
    val brightness: Double,
    val glareRatio: Double,
    val motionScore: Double,
    val distanceRatio: Double,
    val perspectiveSkewDeg: Double,
    val outOfFrameRatio: Double,
)

/**
 * Blur/brightness/glare/motion/perspective/distance/out-of-frame quality metrics, computed
 * with OpenCV's Java API against a single already-oriented grayscale frame Mat.
 *
 * Threading assumption: this class keeps `previousMotionGray` as mutable instance state for
 * frame-to-frame motion diffing. VisionCamera drives a given frame processor instance from a
 * single dedicated camera thread, so `analyze()` is expected to be called serially, one frame
 * at a time, for the lifetime of one `HybridDocScanner` — it is NOT safe to call concurrently
 * from multiple threads.
 */
class OpenCvQualityAnalyzer {
    private var previousMotionGray: Mat? = null

    internal fun analyze(gray: Mat, quad: Quad?, frameWidth: Int, frameHeight: Int): QualityMetrics {
        OpenCvBootstrap.ensureInitialized()

        val blurScore = computeBlurScore(gray)
        val brightness = computeBrightness(gray)
        val glareRatio = computeGlareRatio(gray)
        val motionScore = computeMotionScore(gray)

        val frameArea = frameWidth.toDouble() * frameHeight.toDouble()
        val distanceRatio = if (quad != null && frameArea > 0.0) quadArea(quad) / frameArea else 0.0
        val perspectiveSkewDeg = if (quad != null) computePerspectiveSkewDeg(quad) else 0.0
        val outOfFrameRatio = if (quad != null) computeOutOfFrameRatio(quad, frameWidth, frameHeight) else 0.0

        return QualityMetrics(
            blurScore = blurScore,
            brightness = brightness,
            glareRatio = glareRatio.coerceIn(0.0, 1.0),
            motionScore = motionScore.coerceIn(0.0, 1.0),
            distanceRatio = distanceRatio.coerceIn(0.0, 4.0),
            perspectiveSkewDeg = perspectiveSkewDeg,
            outOfFrameRatio = outOfFrameRatio.coerceIn(0.0, 1.0),
        )
    }

    /** Call when a camera session restarts so motion scoring doesn't diff against a stale frame. */
    fun resetMotionState() {
        previousMotionGray?.release()
        previousMotionGray = null
    }

    private fun computeBlurScore(gray: Mat): Double {
        val laplacian = Mat()
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F)
        val mean = MatOfDouble()
        val stddev = MatOfDouble()
        Core.meanStdDev(laplacian, mean, stddev)
        val sigma = stddev.toArray().firstOrNull() ?: 0.0
        laplacian.release()
        mean.release()
        stddev.release()
        return sigma * sigma
    }

    private fun computeBrightness(gray: Mat): Double {
        return Core.mean(gray).`val`[0]
    }

    private fun computeGlareRatio(gray: Mat): Double {
        val thresholded = Mat()
        Imgproc.threshold(gray, thresholded, 250.0, 255.0, Imgproc.THRESH_BINARY)
        val total = (gray.rows() * gray.cols()).toDouble()
        val ratio = if (total > 0) Core.countNonZero(thresholded) / total else 0.0
        thresholded.release()
        return ratio
    }

    // Motion is diffed on a small, fixed-size downscale of whatever resolution `analyze()`
    // is called with, so the cost of this step never scales with the caller's working
    // resolution.
    private val motionDownscaleLongEdge = 160

    private fun computeMotionScore(gray: Mat): Double {
        val longEdge = max(gray.rows(), gray.cols())
        val scale = motionDownscaleLongEdge.toDouble() / longEdge.toDouble()
        val downscaled = Mat()
        if (scale < 1.0) {
            Imgproc.resize(
                gray,
                downscaled,
                Size(gray.cols() * scale, gray.rows() * scale),
                0.0,
                0.0,
                Imgproc.INTER_AREA,
            )
        } else {
            gray.copyTo(downscaled)
        }

        val prev = previousMotionGray
        val score = if (prev != null && prev.rows() == downscaled.rows() && prev.cols() == downscaled.cols()) {
            val diff = Mat()
            Core.absdiff(prev, downscaled, diff)
            val motion = Core.mean(diff).`val`[0] / 255.0
            diff.release()
            motion
        } else {
            0.0
        }

        prev?.release()
        previousMotionGray = downscaled
        return score
    }

    companion object {
        /** Shoelace formula — plain arithmetic, no OpenCV call needed. */
        fun quadArea(quad: Quad): Double {
            val pts = listOf(quad.topLeft, quad.topRight, quad.bottomRight, quad.bottomLeft)
            var area = 0.0
            for (i in pts.indices) {
                val a = pts[i]
                val b = pts[(i + 1) % pts.size]
                area += a.x * b.y - b.x * a.y
            }
            return abs(area) / 2.0
        }

        /** Max deviation (degrees) of any of the quad's 4 interior corner angles from 90°. */
        fun computePerspectiveSkewDeg(quad: Quad): Double {
            val pts = listOf(quad.topLeft, quad.topRight, quad.bottomRight, quad.bottomLeft)
            var maxDeviation = 0.0
            for (i in pts.indices) {
                val prev = pts[(i - 1 + pts.size) % pts.size]
                val curr = pts[i]
                val next = pts[(i + 1) % pts.size]
                val v1x = prev.x - curr.x
                val v1y = prev.y - curr.y
                val v2x = next.x - curr.x
                val v2y = next.y - curr.y
                val mag1 = sqrt(v1x * v1x + v1y * v1y)
                val mag2 = sqrt(v2x * v2x + v2y * v2y)
                if (mag1 < 1e-6 || mag2 < 1e-6) continue
                val cosAngle = ((v1x * v2x + v1y * v2y) / (mag1 * mag2)).coerceIn(-1.0, 1.0)
                val angleDeg = Math.toDegrees(acos(cosAngle))
                val deviation = abs(angleDeg - 90.0)
                if (deviation > maxDeviation) maxDeviation = deviation
            }
            return maxDeviation
        }

        /** Fraction of the quad's own axis-aligned bounding box that falls outside the frame. */
        fun computeOutOfFrameRatio(quad: Quad, frameWidth: Int, frameHeight: Int): Double {
            val xs = listOf(quad.topLeft.x, quad.topRight.x, quad.bottomRight.x, quad.bottomLeft.x)
            val ys = listOf(quad.topLeft.y, quad.topRight.y, quad.bottomRight.y, quad.bottomLeft.y)
            val minX = xs.min()
            val maxX = xs.max()
            val minY = ys.min()
            val maxY = ys.max()
            val boxArea = max(0.0, maxX - minX) * max(0.0, maxY - minY)
            if (boxArea <= 0.0) return 0.0

            val clampedMinX = minX.coerceIn(0.0, frameWidth.toDouble())
            val clampedMaxX = maxX.coerceIn(0.0, frameWidth.toDouble())
            val clampedMinY = minY.coerceIn(0.0, frameHeight.toDouble())
            val clampedMaxY = maxY.coerceIn(0.0, frameHeight.toDouble())
            val insideArea = max(0.0, clampedMaxX - clampedMinX) * max(0.0, clampedMaxY - clampedMinY)
            val outsideArea = (boxArea - insideArea).coerceAtLeast(0.0)
            return outsideArea / boxArea
        }

        /**
         * Perspective-corrects [src] using [quad] (expressed in [src]'s own pixel coordinates)
         * into an upright rectangle sized to the quad's own average width/height, so callers
         * don't need to guess a canonical output size.
         */
        fun warpToQuad(src: Mat, quad: Quad): Mat {
            val topWidth = distance(quad.topLeft, quad.topRight)
            val bottomWidth = distance(quad.bottomLeft, quad.bottomRight)
            val leftHeight = distance(quad.topLeft, quad.bottomLeft)
            val rightHeight = distance(quad.topRight, quad.bottomRight)
            val outWidth = max(topWidth, bottomWidth).coerceAtLeast(1.0)
            val outHeight = max(leftHeight, rightHeight).coerceAtLeast(1.0)

            val srcPoints = MatOfPoint2f(
                org.opencv.core.Point(quad.topLeft.x, quad.topLeft.y),
                org.opencv.core.Point(quad.topRight.x, quad.topRight.y),
                org.opencv.core.Point(quad.bottomRight.x, quad.bottomRight.y),
                org.opencv.core.Point(quad.bottomLeft.x, quad.bottomLeft.y),
            )
            val dstPoints = MatOfPoint2f(
                org.opencv.core.Point(0.0, 0.0),
                org.opencv.core.Point(outWidth, 0.0),
                org.opencv.core.Point(outWidth, outHeight),
                org.opencv.core.Point(0.0, outHeight),
            )
            val transform = Imgproc.getPerspectiveTransform(srcPoints, dstPoints)
            val warped = Mat()
            Imgproc.warpPerspective(src, warped, transform, Size(outWidth, outHeight))
            srcPoints.release()
            dstPoints.release()
            transform.release()
            return warped
        }

        private fun distance(a: Point, b: Point): Double {
            val dx = a.x - b.x
            val dy = a.y - b.y
            return sqrt(dx * dx + dy * dy)
        }
    }
}
