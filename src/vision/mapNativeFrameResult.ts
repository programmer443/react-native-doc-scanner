import { DocumentType } from '../types/detection';
import type { DetectionResult, FrameAnalysis } from '../types/detection';
import type { NativeFrameResult } from './nativeFrameResult';

/**
 * Pure, worklet-safe mapper — runs on the frame-processing thread, so it
 * must not reference anything outside plain JS (no MMKV, no native modules).
 */
export function mapNativeFrameResult(native: NativeFrameResult, timestampMs: number): FrameAnalysis {
  'worklet';

  const quad = native.detected ? (native.quad ?? null) : null;

  const boundingBox = quad
    ? {
        x: Math.min(quad.topLeft.x, quad.bottomLeft.x),
        y: Math.min(quad.topLeft.y, quad.topRight.y),
        width:
          Math.max(quad.topRight.x, quad.bottomRight.x) - Math.min(quad.topLeft.x, quad.bottomLeft.x),
        height:
          Math.max(quad.bottomLeft.y, quad.bottomRight.y) - Math.min(quad.topLeft.y, quad.topRight.y),
      }
    : null;

  const documentType = (Object.values(DocumentType) as string[]).includes(native.documentType)
    ? (native.documentType as DocumentType)
    : DocumentType.GENERIC;

  const detection: DetectionResult = {
    detected: native.detected,
    documentType,
    confidence: native.confidence,
    quad,
    boundingBox,
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
      distanceRatio: native.distanceRatio,
      perspectiveSkewDeg: native.perspectiveSkewDeg,
      outOfFrameRatio: native.outOfFrameRatio,
    },
    timestampMs,
  };
}
