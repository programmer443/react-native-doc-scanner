package com.margelo.nitro.docscanner

import android.content.Context
import java.io.File
import java.io.FileNotFoundException
import java.util.zip.CRC32

/**
 * Resolves the `bundle://<filename>` / absolute-path convention used by
 * `ModelPaths` (see `src/models/modelRegistry.ts`) into real bytes.
 *
 * - `bundle://<filename>` -> read from this package's Android assets, at
 *   `models/<filename>`. The package's `android/build.gradle` wires the
 *   shared `<package-root>/assets/models/` directory (also used by the iOS
 *   podspec's `s.resources`) into this module's asset sourceSet, so the same
 *   files fetched by `scripts/fetch-models.sh` are what get packaged here —
 *   no separate Android-only copy of the weights is needed.
 * - anything else -> treated as an absolute filesystem path (e.g. a model an
 *   app downloaded at runtime into its own files dir).
 */
object ModelPathResolver {
    private const val BUNDLE_PREFIX = "bundle://"
    private const val ASSETS_MODELS_DIR = "models"

    fun readBytes(context: Context, path: String): ByteArray {
        return if (path.startsWith(BUNDLE_PREFIX)) {
            val fileName = path.removePrefix(BUNDLE_PREFIX)
            val assetPath = "$ASSETS_MODELS_DIR/$fileName"
            try {
                context.assets.open(assetPath).use { it.readBytes() }
            } catch (e: FileNotFoundException) {
                throw IllegalStateException(
                    "react-native-doc-scanner: could not find bundled model asset " +
                        "\"$assetPath\" in android assets. Did you run scripts/fetch-models.sh " +
                        "(or copy the DocAligner weights per docs/MODEL_TRAINING.md) into " +
                        "$fileName, and rebuild the app so Gradle re-packages assets/models/?",
                    e,
                )
            }
        } else {
            val file = File(path)
            if (!file.exists() || !file.isFile) {
                throw IllegalStateException(
                    "react-native-doc-scanner: model path \"$path\" is not a bundle:// asset " +
                        "and does not exist as an absolute file on disk.",
                )
            }
            file.readBytes()
        }
    }

    /**
     * Reads a plain-text charset file (one recognizable character per line, as shipped by
     * RapidOCR's `en_dict.txt`) into an in-memory list for CTC decoding (see OcrPipeline).
     */
    fun readCharset(context: Context, path: String): List<String> {
        val bytes = readBytes(context, path)
        return bytes
            .toString(Charsets.UTF_8)
            .split('\n')
            .map { it.trimEnd('\r') }
            .filter { it.isNotEmpty() }
    }

    /**
     * A real, content-derived identifier for `LoadModelsResult.detectorVersion` /
     * `ocrVersion` — combines the filename with a CRC32 of the actual bytes that were
     * loaded, so it changes whenever the underlying weights change (useful for app teams
     * that swap in fine-tuned models but keep the same filename).
     */
    fun versionLabel(path: String, bytes: ByteArray): String {
        val name = path.substringAfterLast('/').removePrefix(BUNDLE_PREFIX)
        val crc = CRC32()
        crc.update(bytes)
        return "$name@${java.lang.Long.toHexString(crc.value)}"
    }
}
