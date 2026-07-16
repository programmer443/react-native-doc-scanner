/**
 * National ID card parser.
 *
 * Most modern ID cards carry a TD1 MRZ (same 3×30 format as a residence
 * permit), so we try that first — it's far more reliable than label parsing
 * since it's fixed-width and check-digit validated. Cards without an MRZ
 * fall back to generic label/regex heuristics: a long alphanumeric ID
 * number, a labelled "date of birth", and the largest uppercase text block
 * in the top half of the card as the name.
 */

import type { OcrTextLine } from '../../types/ocr';
import { parseMrz } from './mrzParser';

export interface IdCardData {
  fullName: string;
  idNumber: string;
  dateOfBirth: string;
  expiryDate: string;
  address: string;
  nationality: string;
}

export interface IdCardParseResult {
  ok: boolean;
  data: IdCardData;
}

const ID_NUMBER_REGEX = /\b[A-Z0-9]{6,15}\b/;

const DATE_REGEX = /\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b/g;

const extractDates = (text: string): string[] => {
  const out: string[] = [];
  const re = new RegExp(DATE_REGEX.source, 'g');
  let match = re.exec(text);
  while (match !== null) {
    const [, dd, mm, yyyy] = match;
    const m = parseInt(mm, 10);
    const d = parseInt(dd, 10);
    if (m >= 1 && m <= 12 && d >= 1 && d <= 31) {
      out.push(`${yyyy}-${mm.padStart(2, '0')}-${dd.padStart(2, '0')}`);
    }
    match = re.exec(text);
  }
  return out;
};

const isLikelyName = (s: string): boolean => /^[A-Z][A-Z\s'-]{2,40}$/.test(s.trim());

const emptyIdCardData = (): IdCardData => ({
  fullName: '',
  idNumber: '',
  dateOfBirth: '',
  expiryDate: '',
  address: '',
  nationality: '',
});

export const parseIdCard = (fullText: string, lines: OcrTextLine[]): IdCardParseResult => {
  const mrz = parseMrz(fullText);
  if (mrz.ok && mrz.kind === 'identity') {
    return {
      ok: true,
      data: {
        fullName: [mrz.data.firstName, mrz.data.lastName].filter(Boolean).join(' ').trim(),
        idNumber: mrz.data.documentNumber,
        dateOfBirth: mrz.data.dateOfBirth,
        expiryDate: mrz.data.expiryDate,
        address: '',
        nationality: mrz.data.nationality,
      },
    };
  }

  // ── Fallback: no MRZ readable — use label/regex heuristics ──────────
  const data = emptyIdCardData();
  const upper = fullText.toUpperCase();

  const idMatch = ID_NUMBER_REGEX.exec(
    upper
      .split('\n')
      .find((l) => /ID|NUMBER|NO[.:]/.test(l)) ??
      upper,
  );
  if (idMatch) data.idNumber = idMatch[0];

  const dates = extractDates(upper);
  if (dates.length > 0) {
    const sorted = [...dates].sort();
    data.dateOfBirth = sorted[0] ?? '';
    if (sorted.length > 1) data.expiryDate = sorted[sorted.length - 1] ?? '';
  }

  const nameCandidates = lines
    .filter(
      (l) =>
        l.y < 0.55 &&
        isLikelyName(l.text) &&
        !/IDENTITY|CARD|NATIONAL|REPUBLIC|GOVERNMENT/.test(l.text.toUpperCase()),
    )
    .sort((a, b) => b.text.length - a.text.length);
  if (nameCandidates[0]) data.fullName = nameCandidates[0].text.trim();

  const addressIdx = lines.findIndex((l) => /ADDRESS/.test(l.text.toUpperCase()));
  if (addressIdx !== -1) {
    data.address = lines
      .slice(addressIdx, addressIdx + 3)
      .map((l) => l.text.trim())
      .filter(Boolean)
      .join(', ');
  }

  return { ok: !!(data.idNumber || data.fullName), data };
};
