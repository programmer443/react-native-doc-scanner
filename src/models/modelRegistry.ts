/**
 * Describes one swappable set of ONNX weights: a document detector plus the
 * three-stage OCR pipeline (detection → orientation classification →
 * recognition), mirroring RapidOCR's PP-OCR export layout.
 *
 * Path convention (resolved natively — see ios/DocScannerModule.swift and
 * android DocScannerModule.kt):
 *   - `bundle://<filename>`  → shipped inside this package (podspec
 *     resources on iOS, `android/src/main/assets/models/` on Android).
 *   - anything else          → treated as an absolute filesystem path, so
 *     apps can download/replace models at runtime without any UI change.
 */
export interface ModelConfig {
  id: string;
  label: string;
  /** SPDX identifier, surfaced so app teams don't lose track of what they shipped. */
  license: string;
  detectorModelPath: string;
  ocrDetModelPath: string;
  ocrClsModelPath: string;
  ocrRecModelPath: string;
  ocrRecCharsetPath: string;
  /** YuNet ONNX face detector, used by useSelfieCapture. Empty string disables face capture. */
  faceDetectorModelPath: string;
  /**
   * Document-type classifier (PASSPORT/DRIVING_LICENCE/ID_CARD/RESIDENCE_PERMIT/VISA
   * vs GENERIC) — see docs/MODEL_TRAINING.md §6. Empty string disables classification;
   * `analyzeFrame` then always reports `documentType: "GENERIC"`, as it does today.
   */
  classifierModelPath: string;
}

/**
 * Default model: DocAligner (Apache-2.0, corner/quad regression) for
 * detection, RapidOCR's PP-OCRv4 (Apache-2.0, ONNX export of PaddleOCR) for
 * text recognition, YuNet (MIT, opencv_zoo) for face detection. See
 * docs/MODEL_TRAINING.md for how to fetch these weights into assets/models/
 * and how to fine-tune your own detector.
 */
export const DEFAULT_MODEL: ModelConfig = {
  id: 'default-docaligner-rapidocr-v1',
  label: 'DocAligner + RapidOCR (bundled default)',
  license: 'Apache-2.0',
  detectorModelPath: 'bundle://docaligner_fastvit_heatmap.onnx',
  ocrDetModelPath: 'bundle://ch_PP-OCRv4_det_infer.onnx',
  ocrClsModelPath: 'bundle://ch_ppocr_mobile_v2.0_cls_infer.onnx',
  ocrRecModelPath: 'bundle://en_PP-OCRv3_rec_infer.onnx',
  ocrRecCharsetPath: 'bundle://en_dict.txt',
  faceDetectorModelPath: 'bundle://face_detection_yunet_2023mar.onnx',
  // No bundled default yet — no off-the-shelf Apache/MIT model with the right
  // taxonomy exists (see docs/MODEL_TRAINING.md §6). Fine-tune your own and
  // register it, or leave this empty to keep today's GENERIC-only behavior.
  classifierModelPath: '',
};

const registry = new Map<string, ModelConfig>([[DEFAULT_MODEL.id, DEFAULT_MODEL]]);

export function registerModel(config: ModelConfig): void {
  registry.set(config.id, config);
}

export function getModel(id: string): ModelConfig | undefined {
  return registry.get(id);
}

export function listModels(): ModelConfig[] {
  return Array.from(registry.values());
}
