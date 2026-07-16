/**
 * ICAO 9303 Machine Readable Zone (MRZ) parser for passports and similar
 * travel documents. Supports:
 *
 *   TD3   — Passport (2 lines × 44 chars).
 *   TD1   — ID card / residence permit (3 lines × 30 chars; on some cards
 *           the MRZ is on the back panel, not the front).
 *   MRV-A — Visa vignette, ICAO 9303 part 7 (2 lines × 44 chars).
 *   MRV-B — Visa vignette, smaller sticker (2 lines × 36 chars).
 *
 * MRV results are tagged `kind: 'visa'` and land in `visaNumber`/
 * `visaExpiryDate` — never in `documentNumber`/`expiryDate` — since a visa's
 * own number/expiry describe the visa, not the travel document it's stuck
 * inside.
 *
 * Each MRZ embeds check digits computed with the ICAO weighting scheme
 * (7, 3, 1 repeating). We validate the digits and try targeted character
 * substitutions when one fails — most OCR errors on MRZ are unambiguous
 * pairs like O↔0, I/L↔1, S↔5, B↔8, G↔6, Z↔2, Q↔0.
 */

// ── Char → numeric weight (ICAO 9303) ───────────────────────────────────────

const WEIGHTS = [7, 3, 1];

const charValue = (c: string): number => {
  if (c >= '0' && c <= '9') return c.charCodeAt(0) - 48;
  if (c >= 'A' && c <= 'Z') return c.charCodeAt(0) - 55;
  if (c === '<') return 0;
  return 0;
};

const computeCheck = (input: string): number => {
  let sum = 0;
  for (let i = 0; i < input.length; i++) {
    sum += charValue(input[i]) * WEIGHTS[i % 3];
  }
  return sum % 10;
};

// ── OCR autocorrect ─────────────────────────────────────────────────────────

const LETTER_TO_DIGIT: Record<string, string> = {
  O: '0',
  Q: '0',
  D: '0',
  I: '1',
  L: '1',
  T: '1',
  Z: '2',
  E: '3',
  A: '4',
  S: '5',
  G: '6',
  B: '8',
};

const DIGIT_TO_LETTER: Record<string, string> = {
  '0': 'O',
  '1': 'I',
  '2': 'Z',
  '5': 'S',
  '6': 'G',
  '8': 'B',
};

const toDigits = (s: string): string =>
  s
    .split('')
    .map((c) => LETTER_TO_DIGIT[c] ?? c)
    .join('');

const toLetters = (s: string): string =>
  s
    .split('')
    .map((c) => DIGIT_TO_LETTER[c] ?? c)
    .join('');

const reconcileNumeric = (raw: string, check: string): { value: string; ok: boolean } => {
  const checkDigit = LETTER_TO_DIGIT[check] ?? check;
  const candidates = [raw, toDigits(raw)];
  for (const cand of candidates) {
    if (computeCheck(cand).toString() === checkDigit) {
      return { value: cand, ok: true };
    }
  }
  return { value: toDigits(raw), ok: false };
};

// ── Helpers ─────────────────────────────────────────────────────────────────

const yyToYear = (yy: number, kind: 'dob' | 'expiry'): number => {
  if (kind === 'expiry') return 2000 + yy;
  const currentYY = new Date().getFullYear() % 100;
  return yy <= currentYY ? 2000 + yy : 1900 + yy;
};

const parseMrzDate = (s: string, kind: 'dob' | 'expiry'): string => {
  if (!/^\d{6}$/.test(s)) return '';
  const yy = parseInt(s.slice(0, 2), 10);
  const mm = parseInt(s.slice(2, 4), 10);
  const dd = parseInt(s.slice(4, 6), 10);
  if (mm < 1 || mm > 12 || dd < 1 || dd > 31) return '';
  const yyyy = yyToYear(yy, kind);
  return `${yyyy}-${String(mm).padStart(2, '0')}-${String(dd).padStart(2, '0')}`;
};

const cleanName = (n: string): string =>
  toLetters(n).replace(/</g, ' ').replace(/\s+/g, ' ').trim();

const stripFillers = (s: string): string => s.replace(/<+$/g, '').replace(/</g, ' ').trim();

const mrzSexToLabel = (sex: string): string => {
  if (sex === 'M') return 'Male';
  if (sex === 'F') return 'Female';
  return '';
};

// ── MRZ detection ───────────────────────────────────────────────────────────

const normalizeMrzLine = (raw: string): string =>
  raw
    .toUpperCase()
    .replace(/\s+/g, '')
    .replace(/[«»]/g, '<')
    .replace(/[{[(]/g, '<')
    .replace(/[}\])]/g, '<');

export type MrzFormat = 'TD3' | 'TD1' | 'MRV_A' | 'MRV_B';
export type MrzKind = 'identity' | 'visa';

interface MrzBlock {
  format: MrzFormat;
  lines: string[];
}

const findMrzBlock = (text: string): MrzBlock | null => {
  const lines = text
    .split(/\r?\n/)
    .map(normalizeMrzLine)
    .filter((l) => l.length > 0);

  // TD3 passport / MRV-A visa vignette: two consecutive lines of length 44.
  for (let i = 0; i < lines.length - 1; i++) {
    const a = lines[i];
    const b = lines[i + 1];
    if (a.length >= 40 && a.length <= 48 && b.length >= 40 && b.length <= 48 && a.includes('<<')) {
      const firstChar = a.charAt(0);
      if (firstChar === 'P' || firstChar === 'F' || firstChar === 'R') {
        return { format: 'TD3', lines: [a.padEnd(44, '<').slice(0, 44), b.padEnd(44, '<').slice(0, 44)] };
      }
      if (firstChar === 'V') {
        return { format: 'MRV_A', lines: [a.padEnd(44, '<').slice(0, 44), b.padEnd(44, '<').slice(0, 44)] };
      }
    }
  }

  // TD1 ID card / residence permit: three consecutive lines of length 30.
  for (let i = 0; i < lines.length - 2; i++) {
    const a = lines[i];
    const b = lines[i + 1];
    const c = lines[i + 2];
    if (
      a.length >= 28 &&
      a.length <= 32 &&
      b.length >= 28 &&
      b.length <= 32 &&
      c.length >= 28 &&
      c.length <= 32 &&
      (a.startsWith('I') || a.startsWith('A') || a.startsWith('C'))
    ) {
      return {
        format: 'TD1',
        lines: [
          a.padEnd(30, '<').slice(0, 30),
          b.padEnd(30, '<').slice(0, 30),
          c.padEnd(30, '<').slice(0, 30),
        ],
      };
    }
  }

  // MRV-B visa vignette: two consecutive lines of length 36, document code V.
  for (let i = 0; i < lines.length - 1; i++) {
    const a = lines[i];
    const b = lines[i + 1];
    if (
      a.length >= 33 &&
      a.length <= 39 &&
      b.length >= 33 &&
      b.length <= 39 &&
      a.includes('<<') &&
      a.charAt(0) === 'V'
    ) {
      return { format: 'MRV_B', lines: [a.padEnd(36, '<').slice(0, 36), b.padEnd(36, '<').slice(0, 36)] };
    }
  }

  return null;
};

// ── Public API ──────────────────────────────────────────────────────────────

export interface MrzData {
  firstName: string;
  lastName: string;
  documentNumber: string;
  nationality: string;
  dateOfBirth: string;
  gender: string;
  expiryDate: string;
  issuingAuthority: string;
  mrzRaw: string;
  visaNumber?: string;
  visaExpiryDate?: string;
  identityFormat?: 'TD3' | 'TD1';
}

export interface MrzParseResult {
  ok: boolean;
  /** 'identity' = passport/ID biodata page. 'visa' = a visa vignette. */
  kind: MrzKind;
  format: MrzFormat | null;
  data: MrzData;
  /** Number of check digits that matched (0-3). Higher = more trustworthy. */
  validChecks: number;
}

const emptyMrzData = (): MrzData => ({
  firstName: '',
  lastName: '',
  documentNumber: '',
  nationality: '',
  dateOfBirth: '',
  gender: '',
  expiryDate: '',
  issuingAuthority: '',
  mrzRaw: '',
});

interface Td3LikeFields {
  issuingAuthority: string;
  lastName: string;
  firstName: string;
  documentNumber: string;
  documentNumberOk: boolean;
  nationality: string;
  dateOfBirth: string;
  dateOfBirthOk: boolean;
  gender: string;
  expiryDate: string;
  expiryOk: boolean;
}

const extractTd3LikeFields = (l1: string, l2: string): Td3LikeFields => {
  const issuingAuthority = toLetters(stripFillers(l1.slice(2, 5)));
  const nameField = l1.slice(5);
  const [surnameRaw, ...given] = nameField.split('<<');

  const doc = reconcileNumeric(l2.slice(0, 9), l2.charAt(9));
  const nationality = toLetters(stripFillers(l2.slice(10, 13)));
  const dob = reconcileNumeric(l2.slice(13, 19), l2.charAt(19));
  const gender = mrzSexToLabel(l2.charAt(20));
  const exp = reconcileNumeric(l2.slice(21, 27), l2.charAt(27));

  return {
    issuingAuthority,
    lastName: cleanName(surnameRaw),
    firstName: cleanName(given.join(' ')),
    documentNumber: stripFillers(doc.value),
    documentNumberOk: doc.ok,
    nationality,
    dateOfBirth: parseMrzDate(dob.value, 'dob'),
    dateOfBirthOk: dob.ok,
    gender,
    expiryDate: parseMrzDate(exp.value, 'expiry'),
    expiryOk: exp.ok,
  };
};

const parseTd3 = (l1: string, l2: string): MrzParseResult => {
  const data = emptyMrzData();
  data.mrzRaw = `${l1}\n${l2}`;
  data.identityFormat = 'TD3';

  const f = extractTd3LikeFields(l1, l2);
  data.issuingAuthority = f.issuingAuthority;
  data.lastName = f.lastName;
  data.firstName = f.firstName;
  data.documentNumber = f.documentNumber;
  data.nationality = f.nationality;
  data.dateOfBirth = f.dateOfBirth;
  data.gender = f.gender;
  data.expiryDate = f.expiryDate;

  const validChecks = (f.documentNumberOk ? 1 : 0) + (f.dateOfBirthOk ? 1 : 0) + (f.expiryOk ? 1 : 0);
  const hasAny =
    data.documentNumber || data.dateOfBirth || data.expiryDate || data.firstName || data.lastName;

  return { ok: !!hasAny, kind: 'identity', format: 'TD3', data, validChecks };
};

const parseMrv = (l1: string, l2: string, format: 'MRV_A' | 'MRV_B'): MrzParseResult => {
  const data = emptyMrzData();
  data.mrzRaw = `${l1}\n${l2}`;

  const f = extractTd3LikeFields(l1, l2);
  data.issuingAuthority = f.issuingAuthority;
  data.lastName = f.lastName;
  data.firstName = f.firstName;
  data.nationality = f.nationality;
  data.dateOfBirth = f.dateOfBirth;
  data.gender = f.gender;
  data.visaNumber = f.documentNumber;
  data.visaExpiryDate = f.expiryDate;

  const validChecks = (f.documentNumberOk ? 1 : 0) + (f.dateOfBirthOk ? 1 : 0) + (f.expiryOk ? 1 : 0);
  const hasAny = data.visaNumber || data.visaExpiryDate || data.firstName || data.lastName;

  return { ok: !!hasAny, kind: 'visa', format, data, validChecks };
};

const parseTd1 = (l1: string, l2: string, l3: string): MrzParseResult => {
  const data = emptyMrzData();
  data.mrzRaw = `${l1}\n${l2}\n${l3}`;
  data.identityFormat = 'TD1';

  data.issuingAuthority = toLetters(stripFillers(l1.slice(2, 5)));

  const docNum = reconcileNumeric(l1.slice(5, 14), l1.charAt(14));
  data.documentNumber = stripFillers(docNum.value);

  const dob = reconcileNumeric(l2.slice(0, 6), l2.charAt(6));
  data.dateOfBirth = parseMrzDate(dob.value, 'dob');

  data.gender = mrzSexToLabel(l2.charAt(7));

  const exp = reconcileNumeric(l2.slice(8, 14), l2.charAt(14));
  data.expiryDate = parseMrzDate(exp.value, 'expiry');

  data.nationality = toLetters(stripFillers(l2.slice(15, 18)));

  const [surname, ...given] = l3.split('<<');
  data.lastName = cleanName(surname);
  data.firstName = cleanName(given.join(' '));

  const validChecks = (docNum.ok ? 1 : 0) + (dob.ok ? 1 : 0) + (exp.ok ? 1 : 0);
  const hasAny =
    data.documentNumber || data.dateOfBirth || data.expiryDate || data.firstName || data.lastName;

  return { ok: !!hasAny, kind: 'identity', format: 'TD1', data, validChecks };
};

export const parseMrz = (text: string): MrzParseResult => {
  const block = findMrzBlock(text);
  if (!block) {
    return { ok: false, kind: 'identity', format: null, data: emptyMrzData(), validChecks: 0 };
  }
  if (block.format === 'TD3') return parseTd3(block.lines[0], block.lines[1]);
  if (block.format === 'TD1') return parseTd1(block.lines[0], block.lines[1], block.lines[2]);
  return parseMrv(block.lines[0], block.lines[1], block.format);
};
