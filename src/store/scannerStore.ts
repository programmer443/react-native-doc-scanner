import { create } from 'zustand';
import { createMMKV } from 'react-native-mmkv';
import { DocumentType, type Quad } from '../types/detection';
import { GuidanceCode, type CaptureStage, type GuidanceState } from '../types/guidance';
import { GUIDANCE_MESSAGES } from '../constants/messages';
import type { OcrExtractionResult } from '../types/ocr';

const settingsStorage = createMMKV({ id: 'react-native-doc-scanner-settings' });
const LAST_DOCUMENT_TYPE_KEY = 'lastDocumentType';

export const IDLE_GUIDANCE: GuidanceState = {
  code: GuidanceCode.NO_DOCUMENT,
  message: GUIDANCE_MESSAGES[GuidanceCode.NO_DOCUMENT],
  isValid: false,
  confidence: 0,
  flags: {
    hasDocument: false,
    // Matches what evaluateGuidance actually computes when hasDocument is
    // false: isWrongDocumentType short-circuits on hasDocument, so this stays
    // true (no document to mismatch) rather than reporting a false violation.
    isCorrectDocumentType: true,
    isCentered: false,
    isSharp: false,
    hasNoGlare: false,
    hasGoodBrightness: false,
    isStill: false,
    isCorrectSize: false,
    isCorrectPerspective: false,
    isFullyInFrame: false,
  },
};

export interface FrameSize {
  width: number;
  height: number;
}

interface ScannerState {
  documentType: DocumentType;
  guidance: GuidanceState;
  /** Latest detected quad in frame pixel space, or null when nothing is detected. */
  quad: Quad | null;
  /** Native frame's (orientation-normalised) pixel dimensions — see nativeFrameResult.ts's orientation contract. */
  frameSize: FrameSize;
  captureStage: CaptureStage;
  stabilityProgress: number;
  lastResult: OcrExtractionResult | null;
  lastError: string | null;

  setDocumentType: (documentType: DocumentType) => void;
  setGuidance: (guidance: GuidanceState, quad: Quad | null, frameSize: FrameSize) => void;
  setCaptureStage: (stage: CaptureStage) => void;
  setStabilityProgress: (progress: number) => void;
  setLastResult: (result: OcrExtractionResult | null) => void;
  setLastError: (error: string | null) => void;
  reset: () => void;
}

/**
 * In-memory, high-frequency scanner state (guidance/quality updates arrive
 * at ~15Hz from the frame processor). Only `documentType` is persisted —
 * per-frame guidance is intentionally never written to MMKV.
 */
export const useScannerStore = create<ScannerState>((set) => ({
  documentType:
    (settingsStorage.getString(LAST_DOCUMENT_TYPE_KEY) as DocumentType | undefined) ??
    DocumentType.PASSPORT,
  guidance: IDLE_GUIDANCE,
  quad: null,
  frameSize: { width: 0, height: 0 },
  captureStage: 'idle',
  stabilityProgress: 0,
  lastResult: null,
  lastError: null,

  setDocumentType: (documentType) => {
    settingsStorage.set(LAST_DOCUMENT_TYPE_KEY, documentType);
    set({ documentType });
  },
  setGuidance: (guidance, quad, frameSize) => set({ guidance, quad, frameSize }),
  setCaptureStage: (captureStage) => set({ captureStage }),
  setStabilityProgress: (stabilityProgress) => set({ stabilityProgress }),
  setLastResult: (lastResult) => set({ lastResult, lastError: null }),
  setLastError: (lastError) => set({ lastError }),
  reset: () =>
    set({
      guidance: IDLE_GUIDANCE,
      quad: null,
      frameSize: { width: 0, height: 0 },
      captureStage: 'idle',
      stabilityProgress: 0,
      lastResult: null,
      lastError: null,
    }),
}));
