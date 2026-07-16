/**
 * UK (DVLA) driving licence parser.
 *
 * The DVLA photocard places fields in fixed positions on the front of the
 * card. Even with imperfect OCR we can recover every field by combining:
 *
 *   1. A regex for the licence number (5 letters, 6 date-encoded digits,
 *      2 letters, 1 digit, 2 letters — 16 chars total).
 *   2. Numbered-label heuristics — DVLA prints "1." (surname), "2." (given
 *      names), "3." (DOB/place), "4a/b/c" (issue/expiry/authority), "5"
 *      (licence number), "8" (address).
 *   3. Date regex variants (UK format dd.mm.yyyy or dd/mm/yyyy).
 *
 * Every field is optional, so a partial OCR still pre-fills what was
 * readable. Other-country licence formats can be added alongside this one —
 * OcrService picks the parser by locale/documentType, not by hardcoding UK.
 */

import type { OcrTextLine } from '../../types/ocr';

const DVLA_LICENCE_REGEX = /\b([A-Z0-9]{5}[A-Z0-9]{6}[A-Z0-9]{2}\d[A-Z0-9]{2})(?:\s?\d{0,2})?\b/;

const normalizeLicenceNumber = (raw: string): string => {
  const out = raw.split('');
  const letterToDigit: Record<string, string> = {
    O: '0',
    Q: '0',
    I: '1',
    L: '1',
    Z: '2',
    S: '5',
    B: '8',
    G: '6',
  };
  const digitToLetter: Record<string, string> = {
    '0': 'O',
    '1': 'I',
    '5': 'S',
    '8': 'B',
    '6': 'G',
    '2': 'Z',
  };
  for (let i = 0; i < out.length; i++) {
    const isLetterPos = i <= 4 || i === 11 || i === 12 || i === 14 || i === 15;
    if (isLetterPos && /[0-9]/.test(out[i])) out[i] = digitToLetter[out[i]] ?? out[i];
    else if (!isLetterPos && /[A-Z]/.test(out[i])) out[i] = letterToDigit[out[i]] ?? out[i];
  }
  return out.join('');
};

const UK_DATE_REGEX = /\b(\d{2})[./-](\d{2})[./-](\d{4})\b/g;

const toIso = (dd: string, mm: string, yyyy: string): string => `${yyyy}-${mm}-${dd}`;

const extractAllDates = (text: string): string[] => {
  const out: string[] = [];
  const re = new RegExp(UK_DATE_REGEX.source, 'g');
  let match = re.exec(text);
  while (match !== null) {
    const [, dd, mm, yyyy] = match;
    const m = parseInt(mm, 10);
    const d = parseInt(dd, 10);
    if (m >= 1 && m <= 12 && d >= 1 && d <= 31) {
      out.push(toIso(dd, mm, yyyy));
    }
    match = re.exec(text);
  }
  return out;
};

const linesAfter = (lines: OcrTextLine[], predicate: (l: OcrTextLine) => boolean, count = 2): OcrTextLine[] => {
  const idx = lines.findIndex(predicate);
  if (idx === -1) return [];
  return lines.slice(idx + 1, idx + 1 + count);
};

const cleanField = (s: string): string =>
  s
    .replace(/^[\d.)\s:]+/, '')
    .replace(/\s+/g, ' ')
    .trim();

const isLikelyName = (s: string): boolean => /^[A-Z][A-Z\s'-]+$/.test(s.trim()) && s.length >= 2;

const yearOfBirthCheck = (s: string): boolean => {
  const m = /(\d{4})/.exec(s);
  if (!m) return false;
  const y = parseInt(m[1], 10);
  return y >= 1900 && y <= new Date().getFullYear();
};

export interface DrivingLicenceData {
  fullName: string;
  licenceNumber: string;
  dateOfBirth: string;
  expiryDate: string;
  issueDate: string;
  address: string;
  categories: string[];
  issuingAuthority: string;
  country: string;
}

export interface DrivingLicenceParseResult {
  ok: boolean;
  data: DrivingLicenceData;
}

const emptyDrivingLicenceData = (): DrivingLicenceData => ({
  fullName: '',
  licenceNumber: '',
  dateOfBirth: '',
  expiryDate: '',
  issueDate: '',
  address: '',
  categories: [],
  issuingAuthority: '',
  country: 'United Kingdom',
});

/** `lines` positions are normalised (0..1) so spatial heuristics work at any resolution. */
export const parseDrivingLicence = (fullText: string, lines: OcrTextLine[]): DrivingLicenceParseResult => {
  const data = emptyDrivingLicenceData();
  const upper = fullText.toUpperCase();

  // ── 1. Licence number ────────────────────────────────────────────
  const licMatch = DVLA_LICENCE_REGEX.exec(upper.replace(/\s+/g, ' '));
  if (licMatch) {
    data.licenceNumber = normalizeLicenceNumber(licMatch[1]);
  }

  // ── 2. Dates ────────────────────────────────────────────────────
  const allDates = extractAllDates(upper);
  if (allDates.length > 0) {
    const sorted = [...allDates].sort();
    data.dateOfBirth = sorted[0] ?? '';
    if (sorted.length === 2) {
      data.expiryDate = sorted[1] ?? '';
    } else if (sorted.length >= 3) {
      data.issueDate = sorted[1] ?? '';
      data.expiryDate = sorted[sorted.length - 1] ?? '';
    }
  }

  // Cross-check DOB against licence number (digits 6-7 of licence = YY).
  if (data.licenceNumber && data.dateOfBirth) {
    const yyFromLic = data.licenceNumber.substring(5, 7);
    const yyFromDob = data.dateOfBirth.substring(2, 4);
    if (yyFromLic !== yyFromDob && yearOfBirthCheck(data.dateOfBirth)) {
      const fullYear = parseInt(yyFromLic, 10);
      const reconYear = fullYear > new Date().getFullYear() % 100 ? 1900 + fullYear : 2000 + fullYear;
      const month = data.licenceNumber.substring(7, 9);
      const day = data.licenceNumber.substring(9, 11);
      const m = parseInt(month, 10);
      const correctedMonth = m > 50 ? String(m - 50).padStart(2, '0') : month;
      if (parseInt(correctedMonth, 10) >= 1 && parseInt(correctedMonth, 10) <= 12) {
        data.dateOfBirth = `${reconYear}-${correctedMonth}-${day}`;
      }
    }
  }

  // ── 3. Name fields ──────────────────────────────────────────────
  const surnameLines = linesAfter(lines, (l) => /^1\.?\s*$/.test(l.text.trim()), 1);
  const givenLines = linesAfter(lines, (l) => /^2\.?\s*$/.test(l.text.trim()), 1);
  let surname = surnameLines[0]?.text ?? '';
  let given = givenLines[0]?.text ?? '';

  if (!surname) {
    const m = /(?:^|\n)\s*1[\s.)\-:]*([A-Z][A-Z\s'-]{1,40})/.exec(upper);
    if (m) surname = m[1];
  }
  if (!given) {
    const m = /(?:^|\n)\s*2[\s.)\-:]*([A-Z][A-Z\s'-]{1,40})/.exec(upper);
    if (m) given = m[1];
  }

  if (data.licenceNumber && !surname) {
    const prefix = data.licenceNumber.slice(0, 5).replace(/9+$/, '');
    if (prefix.length >= 3) surname = prefix;
  }

  if (!surname || !given) {
    const candidates = lines
      .filter(
        (l) =>
          l.y < 0.55 &&
          isLikelyName(l.text) &&
          !/DRIVING|LICEN[CS]E|UNITED|KINGDOM/.test(l.text.toUpperCase()),
      )
      .sort((a, b) => a.y - b.y);
    if (!surname && candidates[0]) surname = candidates[0].text;
    if (!given && candidates[1]) given = candidates[1].text;
  }

  const cleanSurname = cleanField(surname);
  const cleanGiven = cleanField(given);
  data.fullName = [cleanGiven, cleanSurname].filter(Boolean).join(' ').trim();

  // ── 4. Address (field 8) ────────────────────────────────────────
  const addressIdx = lines.findIndex((l) => /^8[.)\s:]+/.test(l.text.trim()));
  if (addressIdx !== -1) {
    const addrLines = lines
      .slice(addressIdx, addressIdx + 4)
      .map((l) => cleanField(l.text))
      .filter(Boolean);
    data.address = addrLines.join(', ');
  }

  // ── 5. Categories (field 9) ─────────────────────────────────────
  const catMatch = upper.match(/\b(?:A1|A2|AM|B1|BE|C1E|C1|CE|D1E|D1|DE|[ABCDEFGHKMNPQ])\b(?=[^A-Z])/g);
  if (catMatch) {
    data.categories = Array.from(new Set(catMatch)).slice(0, 12);
  }

  return { ok: !!(data.licenceNumber || data.fullName), data };
};
