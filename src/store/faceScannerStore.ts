import { create } from 'zustand';
import type { BoundingBox } from '../types/detection';
import { FaceGuidanceCode, type FaceGuidanceState } from '../types/faceGuidance';
import { FACE_GUIDANCE_MESSAGES } from '../constants/faceMessages';
import type { CaptureStage } from '../types/guidance';
import type { FrameSize } from './scannerStore';

export const IDLE_FACE_GUIDANCE: FaceGuidanceState = {
  code: FaceGuidanceCode.NO_FACE,
  message: FACE_GUIDANCE_MESSAGES[FaceGuidanceCode.NO_FACE],
  isValid: false,
  confidence: 0,
  flags: {
    hasFace: false,
    isCentered: false,
    isCorrectSize: false,
    isLevel: false,
    isSharp: false,
    hasNoGlare: false,
    hasGoodBrightness: false,
    isStill: false,
  },
};

interface FaceScannerState {
  guidance: FaceGuidanceState;
  /** Latest detected face box in frame pixel space, or null when nothing is detected. */
  box: BoundingBox | null;
  /** Native frame's (orientation-normalised) pixel dimensions. */
  frameSize: FrameSize;
  captureStage: CaptureStage;
  stabilityProgress: number;
  lastCapturedPath: string | null;
  lastError: string | null;

  setGuidance: (guidance: FaceGuidanceState, box: BoundingBox | null, frameSize: FrameSize) => void;
  setCaptureStage: (stage: CaptureStage) => void;
  setStabilityProgress: (progress: number) => void;
  setLastCapturedPath: (path: string | null) => void;
  setLastError: (error: string | null) => void;
  reset: () => void;
}

/**
 * In-memory, high-frequency selfie-scanner state (guidance updates arrive at
 * ~15Hz from the frame processor) — a self-contained sibling to
 * `scannerStore.ts` (documents), not shared with it, since a screen is
 * either scanning a document or guiding a selfie, never both at once.
 */
export const useFaceScannerStore = create<FaceScannerState>((set) => ({
  guidance: IDLE_FACE_GUIDANCE,
  box: null,
  frameSize: { width: 0, height: 0 },
  captureStage: 'idle',
  stabilityProgress: 0,
  lastCapturedPath: null,
  lastError: null,

  setGuidance: (guidance, box, frameSize) => set({ guidance, box, frameSize }),
  setCaptureStage: (captureStage) => set({ captureStage }),
  setStabilityProgress: (stabilityProgress) => set({ stabilityProgress }),
  setLastCapturedPath: (lastCapturedPath) => set({ lastCapturedPath, lastError: null }),
  setLastError: (lastError) => set({ lastError }),
  reset: () =>
    set({
      guidance: IDLE_FACE_GUIDANCE,
      box: null,
      frameSize: { width: 0, height: 0 },
      captureStage: 'idle',
      stabilityProgress: 0,
      lastCapturedPath: null,
      lastError: null,
    }),
}));
