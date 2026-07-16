/** Documents this scanner knows how to guide, detect, and extract fields from. */
export enum DocumentType {
  PASSPORT = 'PASSPORT',
  DRIVING_LICENCE = 'DRIVING_LICENCE',
  ID_CARD = 'ID_CARD',
  RESIDENCE_PERMIT = 'RESIDENCE_PERMIT',
  VISA = 'VISA',
  GENERIC = 'GENERIC',
}

export interface Point {
  x: number;
  y: number;
}

/** Four corners of a detected document quad, in frame pixel space, clockwise from top-left. */
export interface Quad {
  topLeft: Point;
  topRight: Point;
  bottomRight: Point;
  bottomLeft: Point;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Raw per-frame detector output, produced natively by the frame processor plugin. */
export interface DetectionResult {
  detected: boolean;
  documentType: DocumentType;
  confidence: number;
  quad: Quad | null;
  boundingBox: BoundingBox | null;
  /** Width/height of the analysed frame, for normalising box/quad ratios. */
  frameWidth: number;
  frameHeight: number;
}

export interface QualityMetrics {
  /** Laplacian variance of the document ROI; higher = sharper. */
  blurScore: number;
  /** Mean luminance 0-255 of the document ROI. */
  brightness: number;
  /** Fraction (0-1) of the ROI classified as blown-out glare. */
  glareRatio: number;
  /** 0-1 motion magnitude vs. the previous frame (0 = perfectly still). */
  motionScore: number;
  /** documentArea / frameArea, used to judge "too close" / "too far". */
  distanceRatio: number;
  /** Max deviation (degrees) of any quad corner angle from 90°; 0 = perfect rectangle. */
  perspectiveSkewDeg: number;
  /** Fraction (0-1) of the document quad estimated to be outside the visible frame. */
  outOfFrameRatio: number;
}

/** One fully-analysed camera frame: detection + quality, as returned by the native plugin. */
export interface FrameAnalysis {
  detection: DetectionResult;
  quality: QualityMetrics;
  timestampMs: number;
}
