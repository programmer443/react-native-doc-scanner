import { DocumentType, type FrameAnalysis } from '../types/detection';
import { GuidanceCode, type GuidanceState, type QualityFlags } from '../types/guidance';
import { GUIDANCE_MESSAGES } from '../constants/messages';
import { SCANNER_THRESHOLDS } from '../constants/thresholds';
import { centerOffsetRatio, quadRotationDeg } from '../utils/geometry';

/**
 * Turns one FrameAnalysis into a single, prioritised human instruction plus
 * the full flag set the auto-capture gate needs. Priority follows the order
 * a user should fix things in: nothing to detect first, then wrong document
 * type, then distance/framing, then stability, then image quality — matching
 * how banking-grade scanners sequence guidance so only one instruction is
 * ever shown at a time.
 *
 * `expectedDocumentType`, if given, is compared against the native
 * classifier's result (`detection.documentType` — see docs/MODEL_TRAINING.md
 * §6). Omit it (or leave the classifier model unloaded, which keeps
 * `documentType` at GENERIC) to skip this check entirely.
 *
 * Pure function — safe to call from a worklet (no allocation beyond plain
 * objects, no closures over native modules).
 */
export function evaluateGuidance(
  analysis: FrameAnalysis,
  expectedDocumentType?: DocumentType,
): GuidanceState {
  'worklet';
  const { detection, quality } = analysis;
  const t = SCANNER_THRESHOLDS;

  const hasDocument = detection.detected && detection.confidence >= t.minDetectionConfidence;

  // Only fires once the classifier has confidently identified *some* document
  // type (GENERIC means "no classifier loaded" or "below its own confidence
  // threshold" — see HybridDocScanner.swift/.kt's documentType contract — so
  // GENERIC is never treated as a mismatch, only an unconfirmed guess).
  const isWrongDocumentType =
    hasDocument &&
    !!expectedDocumentType &&
    detection.documentType !== DocumentType.GENERIC &&
    detection.documentType !== expectedDocumentType;

  const isCorrectSize =
    !hasDocument ||
    (quality.distanceRatio >= t.minDistanceRatio && quality.distanceRatio <= t.maxDistanceRatio);

  const isFullyInFrame = !hasDocument || quality.outOfFrameRatio <= t.maxOutOfFrameRatio;

  const isCentered =
    !hasDocument ||
    !detection.quad ||
    centerOffsetRatio(detection.quad, detection.frameWidth, detection.frameHeight) <=
      t.maxCenterOffsetRatio;

  const rotationDeg = hasDocument && detection.quad ? Math.abs(quadRotationDeg(detection.quad)) : 0;
  const isRotationOk = rotationDeg <= t.maxRotationDeg;
  const isCorrectPerspective = quality.perspectiveSkewDeg <= t.maxPerspectiveSkewDeg;

  const isStill = quality.motionScore <= t.maxMotionScore;
  const isSharp = quality.blurScore >= t.minBlurScore;
  const hasGoodBrightness = quality.brightness >= t.minBrightness && quality.brightness <= t.maxBrightness;
  const hasNoGlare = quality.glareRatio <= t.maxGlareRatio;

  const flags: QualityFlags = {
    hasDocument,
    isCorrectDocumentType: !isWrongDocumentType,
    isCentered,
    isSharp,
    hasNoGlare,
    hasGoodBrightness,
    isStill,
    isCorrectSize,
    isCorrectPerspective: isCorrectPerspective && isRotationOk,
    isFullyInFrame,
  };

  const isValid = Object.values(flags).every(Boolean);

  let code: GuidanceCode;
  if (!hasDocument) {
    code = GuidanceCode.NO_DOCUMENT;
  } else if (isWrongDocumentType) {
    code = GuidanceCode.WRONG_DOCUMENT_TYPE;
  } else if (quality.distanceRatio < t.minDistanceRatio) {
    code = GuidanceCode.MOVE_CLOSER;
  } else if (quality.distanceRatio > t.maxDistanceRatio) {
    code = GuidanceCode.MOVE_FARTHER;
  } else if (!isFullyInFrame) {
    code = GuidanceCode.PARTIALLY_OUT_OF_FRAME;
  } else if (!isCentered) {
    code = GuidanceCode.CENTER_DOCUMENT;
  } else if (!isRotationOk) {
    code = GuidanceCode.ALIGN_DOCUMENT;
  } else if (!isCorrectPerspective) {
    code = GuidanceCode.TILT_PHONE;
  } else if (!isStill) {
    code = GuidanceCode.HOLD_STILL;
  } else if (!isSharp) {
    code = GuidanceCode.BLURRY;
  } else if (quality.brightness < t.minBrightness) {
    code = GuidanceCode.LOW_LIGHT;
  } else if (quality.brightness > t.maxBrightness) {
    code = GuidanceCode.OVEREXPOSED;
  } else if (!hasNoGlare) {
    code = GuidanceCode.REDUCE_GLARE;
  } else {
    code = GuidanceCode.READY;
  }

  return {
    code,
    message: GUIDANCE_MESSAGES[code],
    isValid,
    flags,
    confidence: detection.confidence,
  };
}
