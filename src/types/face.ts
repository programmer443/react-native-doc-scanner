import type { BoundingBox, Point } from './detection';

/**
 * 5-point face landmarks (YuNet's output order): right eye, left eye, nose
 * tip, right mouth corner, left mouth corner — "right"/"left" from the
 * subject's own perspective, matching OpenCV's FaceDetectorYN convention.
 * In the same normalised frame space as the accompanying `BoundingBox` (see
 * `vision/nativeFaceFrameResult.ts`'s orientation-contract doc comment).
 */
export interface FaceLandmarks {
  rightEye: Point;
  leftEye: Point;
  noseTip: Point;
  rightMouthCorner: Point;
  leftMouthCorner: Point;
}

/** Raw per-frame face-detector output, produced natively by `analyzeFaceFrame`. */
export interface FaceDetectionResult {
  detected: boolean;
  confidence: number;
  box: BoundingBox | null;
  landmarks: FaceLandmarks | null;
  frameWidth: number;
  frameHeight: number;
}

export interface FaceQualityMetrics {
  /** Laplacian variance of the whole frame; higher = sharper. */
  blurScore: number;
  /** Mean luminance 0-255 of the whole frame. */
  brightness: number;
  /** Fraction (0-1) of the frame classified as blown-out glare. */
  glareRatio: number;
  /** 0-1 motion magnitude vs. the previous frame (0 = perfectly still). */
  motionScore: number;
}

/** One fully-analysed camera frame: face detection + quality, as returned by the native plugin. */
export interface FaceFrameAnalysis {
  detection: FaceDetectionResult;
  quality: FaceQualityMetrics;
  timestampMs: number;
}
