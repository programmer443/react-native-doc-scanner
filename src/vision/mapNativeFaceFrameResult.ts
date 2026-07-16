import type { FaceDetectionResult, FaceFrameAnalysis } from '../types/face';
import type { NativeFaceFrameResult } from './nativeFaceFrameResult';

/**
 * Pure, worklet-safe mapper — runs on the frame-processing thread, so it
 * must not reference anything outside plain JS (no MMKV, no native modules).
 * Mirrors `mapNativeFrameResult.ts`'s contract exactly.
 */
export function mapNativeFaceFrameResult(
  native: NativeFaceFrameResult,
  timestampMs: number,
): FaceFrameAnalysis {
  'worklet';

  const detection: FaceDetectionResult = {
    detected: native.detected,
    confidence: native.confidence,
    box: native.detected ? (native.box ?? null) : null,
    landmarks: native.detected ? (native.landmarks ?? null) : null,
    frameWidth: native.frameWidth,
    frameHeight: native.frameHeight,
  };

  return {
    detection,
    quality: {
      blurScore: native.blurScore,
      brightness: native.brightness,
      glareRatio: native.glareRatio,
      motionScore: native.motionScore,
    },
    timestampMs,
  };
}
