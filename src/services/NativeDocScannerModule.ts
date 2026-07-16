import { DocScannerNative } from '../vision/docScannerNative';
import type { ModelPaths, LoadModelsResult } from '../specs/DocScanner.nitro';
import type { RawOcrResult } from '../types/ocr';
import type { Quad } from '../types/detection';

/**
 * Thin wrapper around the native `DocScanner` Nitro HybridObject's async
 * methods (model loading + post-capture OCR). Per-frame detection goes
 * through `vision/docScannerNative.ts`'s `analyzeFrame` directly from a
 * worklet instead — see hooks/useDocumentScanner.ts.
 */
export const NativeDocScannerModule = {
  loadModels: (config: ModelPaths): Promise<LoadModelsResult> => DocScannerNative.loadModels(config),

  captureAndExtract: (photoPath: string, documentType: string, quad: Quad | null): Promise<RawOcrResult> =>
    DocScannerNative.captureAndExtract(photoPath, documentType, quad ?? undefined),
};
