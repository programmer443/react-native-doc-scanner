package com.margelo.nitro.docscanner

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.RotatedRect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.max

/**
 * The RapidOCR (PP-OCR) 3-stage pipeline behind `captureAndExtract`: text detection (DB),
 * orientation classification, and text recognition (CTC) — plus the perspective correction
 * that turns a captured photo + detected quad into a rectified, OCR-ready image.
 *
 * Runs once per capture, not per-frame, so it can afford full-resolution work.
 *
 * Preprocessing note: PP-OCR's exact published preprocessing differs slightly between det and
 * cls/rec stages. This implementation uses the standard, widely-documented RapidOCR/PaddleOCR
 * convention — RGB channel order throughout; ImageNet mean/std normalisation for the det (DB)
 * stage; simple `(x/255 - 0.5) / 0.5` normalisation for cls and rec — rather than assuming an
 * unverified exact match to any specific exported graph.
 */
class OcrPipeline {
    fun extract(context: Context, engine: OnnxEngine, photoPath: String, quad: Quad?): RawOcrResultNative {
        val detSession = engine.ocrDetSession
            ?: throw IllegalStateException("react-native-doc-scanner: call loadModels() before captureAndExtract() — OCR detection session is not loaded.")
        val clsSession = engine.ocrClsSession
            ?: throw IllegalStateException("react-native-doc-scanner: call loadModels() before captureAndExtract() — OCR classification session is not loaded.")
        val recSession = engine.ocrRecSession
            ?: throw IllegalStateException("react-native-doc-scanner: call loadModels() before captureAndExtract() — OCR recognition session is not loaded.")
        val charset = engine.charset
        if (charset.isEmpty()) {
            throw IllegalStateException("react-native-doc-scanner: OCR charset is empty — call loadModels() first.")
        }

        OpenCvBootstrap.ensureInitialized()

        val cleanPath = photoPath.removePrefix("file://")
        val file = File(cleanPath)
        if (!file.exists() || !file.isFile) {
            throw IllegalStateException("react-native-doc-scanner: captured photo not found at \"$cleanPath\".")
        }
        val bitmap = BitmapFactory.decodeFile(cleanPath)
            ?: throw IllegalStateException(
                "react-native-doc-scanner: failed to decode captured photo at \"$cleanPath\" (unsupported or corrupt image).",
            )

        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        bitmap.recycle()
        val rgbFull = Mat()
        Imgproc.cvtColor(rgba, rgbFull, Imgproc.COLOR_RGBA2RGB)
        rgba.release()

        val rectified: Mat
        val rectifiedImagePath: String
        if (quad != null) {
            rectified = OpenCvQualityAnalyzer.warpToQuad(rgbFull, quad)
            rgbFull.release()
            rectifiedImagePath = saveRectifiedImage(context, rectified)
        } else {
            rectified = rgbFull
            rectifiedImagePath = cleanPath
        }

        try {
            val lineQuads = try {
                detectTextLines(detSession, rectified)
            } catch (e: Exception) {
                Log.e(TAG, "Text-line detection failed: ${e.message}", e)
                emptyList()
            }

            val lines = mutableListOf<OcrTextLineNative>()
            for (lineQuad in lineQuads) {
                var crop = OpenCvQualityAnalyzer.warpToQuad(rectified, lineQuad)
                try {
                    crop = classifyAndFixOrientation(clsSession, crop)
                    val (text, confidence) = recognizeText(recSession, charset, crop)
                    if (text.isNotBlank()) {
                        val bbox = normalizedBoundingBox(lineQuad, rectified.cols(), rectified.rows())
                        lines.add(
                            OcrTextLineNative(
                                text = text,
                                confidence = confidence,
                                x = bbox[0],
                                y = bbox[1],
                                width = bbox[2],
                                height = bbox[3],
                            ),
                        )
                    }
                } finally {
                    crop.release()
                }
            }

            val fullText = lines.joinToString("\n") { it.text }
            val overallConfidence = if (lines.isNotEmpty()) lines.sumOf { it.confidence } / lines.size else 0.0

            return RawOcrResultNative(
                fullText = fullText,
                lines = lines.toTypedArray(),
                confidence = overallConfidence,
                rectifiedImagePath = rectifiedImagePath,
            )
        } finally {
            rectified.release()
        }
    }

    // ---------------------------------------------------------------------
    // Stage 1: text detection (DB / PP-OCRv4 det)
    // ---------------------------------------------------------------------

    private fun detectTextLines(session: OrtSession, rectifiedRgb: Mat): List<Quad> {
        val (resized, scaleX, scaleY) = resizeToMultipleOf32(rectifiedRgb, DET_MAX_SIDE)
        try {
            val env = OrtEnvironment.getEnvironment()
            val inputName = session.inputNames.iterator().next()
            buildInputTensor(env, resized, IMAGENET_MEAN, IMAGENET_STD).use { inputTensor ->
                session.run(mapOf(inputName to inputTensor)).use { results ->
                    val outputName = session.outputNames.iterator().next()
                    val outputOpt = results.get(outputName)
                    if (!outputOpt.isPresent) return emptyList()
                    val tensorOut = outputOpt.get() as? OnnxTensor ?: return emptyList()
                    val shape = tensorOut.info.shape

                    val h: Int
                    val w: Int
                    when (shape.size) {
                        4 -> {
                            h = shape[2].toInt()
                            w = shape[3].toInt()
                        }
                        3 -> {
                            h = shape[1].toInt()
                            w = shape[2].toInt()
                        }
                        else -> throw IllegalStateException("Unexpected DB det output shape ${shape.joinToString()}")
                    }

                    val buffer = tensorOut.floatBuffer
                    val maskBytes = ByteArray(h * w)
                    for (i in 0 until h * w) {
                        maskBytes[i] = if (buffer.get(i) > DET_PROB_THRESHOLD) 0xFF.toByte() else 0
                    }
                    val probMap = Mat(h, w, CvType.CV_8UC1)
                    probMap.put(0, 0, maskBytes)

                    val contours = mutableListOf<MatOfPoint>()
                    val hierarchy = Mat()
                    Imgproc.findContours(probMap, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
                    probMap.release()
                    hierarchy.release()

                    val boxes = mutableListOf<Pair<Quad, Double>>()
                    for (contour in contours) {
                        val area = Imgproc.contourArea(contour)
                        if (area >= MIN_TEXT_CONTOUR_AREA) {
                            val contour2f = MatOfPoint2f(*contour.toArray())
                            val rotatedRect = Imgproc.minAreaRect(contour2f)
                            contour2f.release()

                            // Lightweight stand-in for DB's "unclip" polygon expansion (which
                            // upstream computes via Vatti-clipping/pyclipper): grow the tight
                            // box by a fixed ratio around its own center. This is structurally
                            // equivalent in effect (recovers margin lost by the shrunk-text
                            // training target) but not bit-for-bit identical to the reference
                            // postprocessing.
                            val expanded = RotatedRect(
                                rotatedRect.center,
                                Size(rotatedRect.size.width * UNCLIP_RATIO, rotatedRect.size.height * UNCLIP_RATIO),
                                rotatedRect.angle,
                            )
                            val ptsArr = arrayOf(
                                org.opencv.core.Point(),
                                org.opencv.core.Point(),
                                org.opencv.core.Point(),
                                org.opencv.core.Point(),
                            )
                            expanded.points(ptsArr)

                            val scaledPoints = ptsArr.map { Point(x = it.x / scaleX, y = it.y / scaleY) }
                            val lineQuad = DocumentDetector.orderQuadPoints(scaledPoints)
                            val yCenter = (lineQuad.topLeft.y + lineQuad.topRight.y + lineQuad.bottomRight.y + lineQuad.bottomLeft.y) / 4.0
                            boxes.add(Pair(lineQuad, yCenter))
                        }
                        contour.release()
                    }

                    // Spec requirement: sorted top-to-bottom by vertical center. Break ties by
                    // horizontal position for deterministic reading order.
                    return boxes.sortedWith(compareBy({ it.second }, { it.first.topLeft.x }))
                        .map { it.first }
                }
            }
        } finally {
            resized.release()
        }
    }

    // ---------------------------------------------------------------------
    // Stage 2: orientation classification (ppocr_mobile cls)
    // ---------------------------------------------------------------------

    private fun classifyAndFixOrientation(session: OrtSession, crop: Mat): Mat {
        val env = OrtEnvironment.getEnvironment()
        val (targetH, targetW) = staticInputSize(session, defaultH = 48, defaultW = 192)

        val resized = Mat()
        Imgproc.resize(crop, resized, Size(targetW.toDouble(), targetH.toDouble()))
        val inputTensor = buildInputTensor(env, resized, HALF_MEAN, HALF_STD)
        resized.release()

        inputTensor.use { tensor ->
            val inputName = session.inputNames.iterator().next()
            session.run(mapOf(inputName to tensor)).use { results ->
                val outputName = session.outputNames.iterator().next()
                val outputOpt = results.get(outputName)
                if (!outputOpt.isPresent) return crop
                val tensorOut = outputOpt.get() as? OnnxTensor ?: return crop
                val buffer = tensorOut.floatBuffer
                if (buffer.remaining() < 2) return crop

                val classNormal = buffer.get(0)
                val classFlipped = buffer.get(1)
                return if (classFlipped > classNormal) {
                    val rotated = Mat()
                    Core.rotate(crop, rotated, Core.ROTATE_180)
                    crop.release()
                    rotated
                } else {
                    crop
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Stage 3: text recognition (PP-OCRv3 rec, CTC)
    // ---------------------------------------------------------------------

    private fun recognizeText(session: OrtSession, charset: List<String>, crop: Mat): Pair<String, Double> {
        val env = OrtEnvironment.getEnvironment()
        val (targetH, _) = staticInputSize(session, defaultH = 48, defaultW = 320)

        val aspect = crop.cols().toDouble() / crop.rows().toDouble().coerceAtLeast(1.0)
        val targetW = (targetH * aspect).toInt().coerceIn(1, MAX_REC_WIDTH)

        val resized = Mat()
        Imgproc.resize(crop, resized, Size(targetW.toDouble(), targetH.toDouble()))
        val inputTensor = buildInputTensor(env, resized, HALF_MEAN, HALF_STD)
        resized.release()

        inputTensor.use { tensor ->
            val inputName = session.inputNames.iterator().next()
            session.run(mapOf(inputName to tensor)).use { results ->
                val outputName = session.outputNames.iterator().next()
                val outputOpt = results.get(outputName)
                if (!outputOpt.isPresent) return Pair("", 0.0)
                val tensorOut = outputOpt.get() as? OnnxTensor ?: return Pair("", 0.0)
                val shape = tensorOut.info.shape
                if (shape.size != 3) {
                    Log.w(TAG, "Unexpected rec output shape ${shape.joinToString()}")
                    return Pair("", 0.0)
                }
                val seqLen = shape[1].toInt()
                val numClasses = shape[2].toInt()
                return ctcGreedyDecode(tensorOut.floatBuffer, seqLen, numClasses, charset)
            }
        }
    }

    private fun ctcGreedyDecode(buffer: FloatBuffer, seqLen: Int, numClasses: Int, charset: List<String>): Pair<String, Double> {
        val sb = StringBuilder()
        var lastIndex = -1
        var confSum = 0.0
        var confCount = 0

        for (t in 0 until seqLen) {
            val base = t * numClasses
            var bestIdx = 0
            var bestVal = Float.NEGATIVE_INFINITY
            for (c in 0 until numClasses) {
                val v = buffer.get(base + c)
                if (v > bestVal) {
                    bestVal = v
                    bestIdx = c
                }
            }

            // Cheap per-timestep confidence: softmax probability of just the winning class,
            // without materialising a full softmax over the class axis.
            var sumExp = 0.0
            for (c in 0 until numClasses) {
                sumExp += Math.exp((buffer.get(base + c) - bestVal).toDouble())
            }
            val prob = if (sumExp > 0.0) 1.0 / sumExp else 0.0

            if (bestIdx != 0 && bestIdx != lastIndex) {
                val char = charIndexToString(bestIdx, charset, numClasses)
                if (char != null) {
                    sb.append(char)
                    confSum += prob
                    confCount += 1
                }
            }
            lastIndex = bestIdx
        }

        val confidence = if (confCount > 0) confSum / confCount else 0.0
        return Pair(sb.toString(), confidence)
    }

    /**
     * PP-OCR/RapidOCR CTC label convention: index 0 is the reserved blank, indices
     * `1..charset.size` map to the dict file in order, and the final index (`numClasses - 1`)
     * is a trailing space — verified against the shape of `en_dict.txt` as shipped by RapidOCR.
     */
    private fun charIndexToString(index: Int, charset: List<String>, numClasses: Int): String? {
        if (index <= 0) return null
        val charIdx = index - 1
        if (charIdx < charset.size) return charset[charIdx]
        if (index == numClasses - 1) return " "
        return null
    }

    // ---------------------------------------------------------------------
    // Shared helpers
    // ---------------------------------------------------------------------

    private fun staticInputSize(session: OrtSession, defaultH: Int, defaultW: Int): Pair<Int, Int> {
        val info = session.inputInfo.values.firstOrNull()?.info as? TensorInfo
        val shape = info?.shape
        val h = shape?.getOrNull(2)?.takeIf { it > 0 }?.toInt() ?: defaultH
        val w = shape?.getOrNull(3)?.takeIf { it > 0 }?.toInt() ?: defaultW
        return Pair(h, w)
    }

    private fun buildInputTensor(env: OrtEnvironment, rgbMat: Mat, mean: FloatArray, std: FloatArray): OnnxTensor {
        val h = rgbMat.rows()
        val w = rgbMat.cols()
        val pixelBytes = ByteArray(h * w * 3)
        rgbMat.get(0, 0, pixelBytes)
        val plane = h * w
        val chw = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val r = (pixelBytes[i * 3].toInt() and 0xFF) / 255f
            val g = (pixelBytes[i * 3 + 1].toInt() and 0xFF) / 255f
            val b = (pixelBytes[i * 3 + 2].toInt() and 0xFF) / 255f
            chw[i] = (r - mean[0]) / std[0]
            chw[plane + i] = (g - mean[1]) / std[1]
            chw[plane * 2 + i] = (b - mean[2]) / std[2]
        }
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), longArrayOf(1, 3, h.toLong(), w.toLong()))
    }

    private fun resizeToMultipleOf32(mat: Mat, maxSide: Int): Triple<Mat, Double, Double> {
        val longSide = max(mat.rows(), mat.cols())
        val scale = if (longSide > maxSide) maxSide.toDouble() / longSide else 1.0
        var targetW = (mat.cols() * scale).toInt()
        var targetH = (mat.rows() * scale).toInt()
        targetW = (((targetW + 31) / 32) * 32).coerceAtLeast(32)
        targetH = (((targetH + 31) / 32) * 32).coerceAtLeast(32)

        val resized = Mat()
        Imgproc.resize(mat, resized, Size(targetW.toDouble(), targetH.toDouble()))
        val scaleX = targetW.toDouble() / mat.cols()
        val scaleY = targetH.toDouble() / mat.rows()
        return Triple(resized, scaleX, scaleY)
    }

    private fun normalizedBoundingBox(quad: Quad, width: Int, height: Int): DoubleArray {
        val xs = listOf(quad.topLeft.x, quad.topRight.x, quad.bottomRight.x, quad.bottomLeft.x)
        val ys = listOf(quad.topLeft.y, quad.topRight.y, quad.bottomRight.y, quad.bottomLeft.y)
        val minX = xs.min().coerceIn(0.0, width.toDouble())
        val maxX = xs.max().coerceIn(0.0, width.toDouble())
        val minY = ys.min().coerceIn(0.0, height.toDouble())
        val maxY = ys.max().coerceIn(0.0, height.toDouble())
        val safeWidth = width.toDouble().coerceAtLeast(1.0)
        val safeHeight = height.toDouble().coerceAtLeast(1.0)
        return doubleArrayOf(minX / safeWidth, minY / safeHeight, (maxX - minX) / safeWidth, (maxY - minY) / safeHeight)
    }

    private fun saveRectifiedImage(context: Context, mat: Mat): String {
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bitmap)
        val outFile = File(context.cacheDir, "doc-scanner-rectified-${System.currentTimeMillis()}.jpg")
        FileOutputStream(outFile).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 92, out)
        }
        bitmap.recycle()
        return outFile.absolutePath
    }

    companion object {
        private const val TAG = "DocScanner/OcrPipeline"

        private const val DET_MAX_SIDE = 960
        private const val DET_PROB_THRESHOLD = 0.3f
        private const val MIN_TEXT_CONTOUR_AREA = 16.0
        private const val UNCLIP_RATIO = 1.6
        private const val MAX_REC_WIDTH = 320

        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
        private val HALF_MEAN = floatArrayOf(0.5f, 0.5f, 0.5f)
        private val HALF_STD = floatArrayOf(0.5f, 0.5f, 0.5f)
    }
}
