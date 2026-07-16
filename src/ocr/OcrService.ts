import { DocumentType } from '../types/detection';
import type { Quad } from '../types/detection';
import type { OcrExtractionResult, StructuredDocumentData } from '../types/ocr';
import { NativeDocScannerModule } from '../services/NativeDocScannerModule';
import { parseMrz } from './parsers/mrzParser';
import { parseDrivingLicence } from './parsers/parseDrivingLicence';
import { parseIdCard } from './parsers/parseIdCard';
import { logCaptureStarted, logRawOcrResult, logExtractionResult } from '../utils/scannerLogger';

const emptyStructuredData = (documentType: DocumentType): StructuredDocumentData => ({
  documentType,
  name: '',
  documentNumber: '',
  dob: '',
  expiry: '',
  nationality: '',
  address: '',
  mrz: '',
  confidence: 0,
});

/**
 * Runs the native perspective-correction + RapidOCR pipeline on a captured
 * photo, then structures the result into the JSON shape apps consume. This
 * is the only place that knows how each document type's raw OCR maps to
 * fields — UI code only ever sees `StructuredDocumentData`.
 *
 * Logs the raw OCR output and the parsed fields at every call (see
 * `utils/scannerLogger.ts`) — visible via `adb logcat`/Xcode device console
 * even on a standalone build with no Metro attached, so field-parsing
 * accuracy can be checked against real captures without extra tooling.
 */
export async function extractDocumentData(
  photoPath: string,
  documentType: DocumentType,
  quad: Quad | null,
): Promise<OcrExtractionResult> {
  logCaptureStarted(documentType);
  const result = await runExtraction(photoPath, documentType, quad);
  logExtractionResult(documentType, result);
  return result;
}

async function runExtraction(
  photoPath: string,
  documentType: DocumentType,
  quad: Quad | null,
): Promise<OcrExtractionResult> {
  const raw = await NativeDocScannerModule.captureAndExtract(photoPath, documentType, quad);
  logRawOcrResult(documentType, raw);

  if (!raw.fullText.trim()) {
    return {
      success: false,
      data: emptyStructuredData(documentType),
      raw,
      mrzValid: null,
    };
  }

  const data = emptyStructuredData(documentType);
  data.confidence = raw.confidence;
  let mrzValid: boolean | null;

  switch (documentType) {
    case DocumentType.PASSPORT:
    case DocumentType.VISA: {
      const mrz = parseMrz(raw.fullText);
      data.name = [mrz.data.firstName, mrz.data.lastName].filter(Boolean).join(' ').trim();
      data.documentNumber = mrz.kind === 'visa' ? mrz.data.visaNumber ?? '' : mrz.data.documentNumber;
      data.dob = mrz.data.dateOfBirth;
      data.expiry = mrz.kind === 'visa' ? mrz.data.visaExpiryDate ?? '' : mrz.data.expiryDate;
      data.nationality = mrz.data.nationality;
      data.mrz = mrz.data.mrzRaw;
      mrzValid = mrz.ok ? mrz.validChecks === 3 : null;
      return { success: mrz.ok, data, raw, mrzValid };
    }

    case DocumentType.DRIVING_LICENCE: {
      const licence = parseDrivingLicence(raw.fullText, raw.lines);
      data.name = licence.data.fullName;
      data.documentNumber = licence.data.licenceNumber;
      data.dob = licence.data.dateOfBirth;
      data.expiry = licence.data.expiryDate;
      data.address = licence.data.address;
      data.nationality = licence.data.country;
      return { success: licence.ok, data, raw, mrzValid: null };
    }

    case DocumentType.ID_CARD:
    case DocumentType.RESIDENCE_PERMIT: {
      const idCard = parseIdCard(raw.fullText, raw.lines);
      data.name = idCard.data.fullName;
      data.documentNumber = idCard.data.idNumber;
      data.dob = idCard.data.dateOfBirth;
      data.expiry = idCard.data.expiryDate;
      data.address = idCard.data.address;
      data.nationality = idCard.data.nationality;
      const mrz = parseMrz(raw.fullText);
      data.mrz = mrz.ok ? mrz.data.mrzRaw : '';
      mrzValid = mrz.ok ? mrz.validChecks === 3 : null;
      return { success: idCard.ok, data, raw, mrzValid };
    }

    default:
      return { success: false, data, raw, mrzValid: null };
  }
}
