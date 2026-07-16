/**
 * Console logging for the scan → OCR → parse pipeline, tagged so it's easy
 * to filter in `adb logcat` (Android) or the Xcode/Console.app device log
 * (iOS) — both keep forwarding `console.*` output from the JS engine even
 * when Metro isn't attached, so this is visible on a standalone installed
 * build, not just during `pnpm run ios`/`android`.
 *
 * Grep for `[DocScanner]` to isolate this package's output, e.g.:
 *   adb logcat | grep '\[DocScanner\]'
 */
import type { DocumentType } from '../types/detection';
import type { OcrExtractionResult, RawOcrResult } from '../types/ocr';

const TAG = '[DocScanner]';

export function logCaptureStarted(documentType: DocumentType): void {
  console.warn(`${TAG} capture started — documentType=${documentType}`);
}

export function logRawOcrResult(documentType: DocumentType, raw: RawOcrResult): void {
  console.warn(
    `${TAG} raw OCR — documentType=${documentType} confidence=${raw.confidence.toFixed(3)} ` +
      `lines=${raw.lines.length} rectifiedImagePath=${raw.rectifiedImagePath}`,
  );
  console.warn(`${TAG} raw OCR fullText:\n${raw.fullText || '(empty)'}`);
  raw.lines.forEach((line, i) => {
    console.warn(
      `${TAG}   line[${i}] conf=${line.confidence.toFixed(3)} bbox=(${line.x.toFixed(3)},${line.y.toFixed(3)},${line.width.toFixed(3)},${line.height.toFixed(3)}) text="${line.text}"`,
    );
  });
}

export function logExtractionResult(documentType: DocumentType, result: OcrExtractionResult): void {
  console.warn(
    `${TAG} extraction result — documentType=${documentType} success=${result.success} mrzValid=${result.mrzValid}`,
  );
  console.warn(`${TAG} parsed fields:`, JSON.stringify(result.data, null, 2));
}
