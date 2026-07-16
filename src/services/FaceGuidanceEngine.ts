import type { FaceFrameAnalysis } from '../types/face';
import { FaceGuidanceCode, type FaceGuidanceState, type FaceQualityFlags } from '../types/faceGuidance';
import { FACE_GUIDANCE_MESSAGES } from '../constants/faceMessages';
import { FACE_THRESHOLDS } from '../constants/faceThresholds';
import { boxCenterOffsetRatio, eyeLineTiltDeg } from '../utils/geometry';

/**
 * Turns one FaceFrameAnalysis into a single, prioritised human instruction
 * plus the full flag set the auto-capture gate needs. Mirrors
 * `GuidanceEngine.ts`'s document-scanning priority order (nothing to detect
 * first, then distance/framing, then stability, then image quality).
 *
 * Pure function — safe to call from a worklet (no allocation beyond plain
 * objects, no closures over native modules).
 */
export function evaluateFaceGuidance(analysis: FaceFrameAnalysis): FaceGuidanceState {
  'worklet';
  const { detection, quality } = analysis;
  const t = FACE_THRESHOLDS;

  const hasFace = detection.detected && detection.confidence >= t.minDetectionConfidence;

  const frameArea = detection.frameWidth * detection.frameHeight;
  const sizeRatio = hasFace && detection.box && frameArea > 0 ? (detection.box.width * detection.box.height) / frameArea : 0;
  const isCorrectSize = !hasFace || (sizeRatio >= t.minFaceSizeRatio && sizeRatio <= t.maxFaceSizeRatio);

  const isCentered =
    !hasFace ||
    !detection.box ||
    boxCenterOffsetRatio(detection.box, detection.frameWidth, detection.frameHeight) <= t.maxCenterOffsetRatio;

  const tiltDeg = hasFace && detection.landmarks ? Math.abs(eyeLineTiltDeg(detection.landmarks)) : 0;
  const isLevel = tiltDeg <= t.maxEyeTiltDeg;

  const isStill = quality.motionScore <= t.maxMotionScore;
  const isSharp = quality.blurScore >= t.minBlurScore;
  const hasGoodBrightness = quality.brightness >= t.minBrightness && quality.brightness <= t.maxBrightness;
  const hasNoGlare = quality.glareRatio <= t.maxGlareRatio;

  const flags: FaceQualityFlags = {
    hasFace,
    isCentered,
    isCorrectSize,
    isLevel,
    isSharp,
    hasNoGlare,
    hasGoodBrightness,
    isStill,
  };

  const isValid = Object.values(flags).every(Boolean);

  let code: FaceGuidanceCode;
  if (!hasFace) {
    code = FaceGuidanceCode.NO_FACE;
  } else if (sizeRatio < t.minFaceSizeRatio) {
    code = FaceGuidanceCode.MOVE_CLOSER;
  } else if (sizeRatio > t.maxFaceSizeRatio) {
    code = FaceGuidanceCode.MOVE_FARTHER;
  } else if (!isCentered) {
    code = FaceGuidanceCode.CENTER_FACE;
  } else if (!isLevel) {
    code = FaceGuidanceCode.LEVEL_HEAD;
  } else if (!isStill) {
    code = FaceGuidanceCode.HOLD_STILL;
  } else if (!isSharp) {
    code = FaceGuidanceCode.BLURRY;
  } else if (quality.brightness < t.minBrightness) {
    code = FaceGuidanceCode.LOW_LIGHT;
  } else if (quality.brightness > t.maxBrightness) {
    code = FaceGuidanceCode.OVEREXPOSED;
  } else if (!hasNoGlare) {
    code = FaceGuidanceCode.REDUCE_GLARE;
  } else {
    code = FaceGuidanceCode.READY;
  }

  return {
    code,
    message: FACE_GUIDANCE_MESSAGES[code],
    isValid,
    flags,
    confidence: detection.confidence,
  };
}
