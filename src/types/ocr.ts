import type { DocumentType } from './detection';

export interface OcrTextLine {
  text: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Raw output of the native capture+extract call, before field parsing. */
export interface RawOcrResult {
  fullText: string;
  lines: OcrTextLine[];
  /** Mean recognition confidence (0-1) across all lines. */
  confidence: number;
  /** Path to the perspective-corrected image the OCR actually ran on. */
  rectifiedImagePath: string;
}

export interface StructuredDocumentData {
  documentType: DocumentType | '';
  name: string;
  documentNumber: string;
  dob: string;
  expiry: string;
  nationality: string;
  address: string;
  mrz: string;
  confidence: number;
}

export interface OcrExtractionResult {
  success: boolean;
  data: StructuredDocumentData;
  raw: RawOcrResult;
  /** Populated only when a passport/visa MRZ checksum was validated. */
  mrzValid: boolean | null;
}
