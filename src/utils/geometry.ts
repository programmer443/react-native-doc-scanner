import type { BoundingBox, Point, Quad } from '../types/detection';
import type { FaceLandmarks } from '../types/face';

/** Euclidean distance between two points. */
export function distance(a: Point, b: Point): number {
  'worklet';
  return Math.hypot(b.x - a.x, b.y - a.y);
}

/** Centroid of a quad. */
export function quadCenter(quad: Quad): Point {
  'worklet';
  return {
    x: (quad.topLeft.x + quad.topRight.x + quad.bottomRight.x + quad.bottomLeft.x) / 4,
    y: (quad.topLeft.y + quad.topRight.y + quad.bottomRight.y + quad.bottomLeft.y) / 4,
  };
}

/** Shoelace-formula area of a quad. */
export function quadArea(quad: Quad): number {
  'worklet';
  const pts = [quad.topLeft, quad.topRight, quad.bottomRight, quad.bottomLeft];
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const p1 = pts[i];
    const p2 = pts[(i + 1) % pts.length];
    area += p1.x * p2.y - p2.x * p1.y;
  }
  return Math.abs(area) / 2;
}

/**
 * Normalised distance (0-1) of the quad's centroid from the frame's centre,
 * where 0 = perfectly centred and 1 = centroid at a frame corner.
 */
export function centerOffsetRatio(quad: Quad, frameWidth: number, frameHeight: number): number {
  'worklet';
  const center = quadCenter(quad);
  const frameCenter = { x: frameWidth / 2, y: frameHeight / 2 };
  const maxOffset = Math.hypot(frameWidth / 2, frameHeight / 2);
  if (maxOffset === 0) return 0;
  return Math.min(1, distance(center, frameCenter) / maxOffset);
}

/**
 * In-plane rotation of the document, in degrees, measured from the top edge
 * (topLeft → topRight) against the horizontal. 0 = perfectly upright.
 * Distinct from perspectiveSkewDeg, which captures keystone distortion from
 * a tilted camera rather than rotation within the frame plane.
 */
export function quadRotationDeg(quad: Quad): number {
  'worklet';
  const dx = quad.topRight.x - quad.topLeft.x;
  const dy = quad.topRight.y - quad.topLeft.y;
  const radians = Math.atan2(dy, dx);
  return (radians * 180) / Math.PI;
}

/** Centroid of a bounding box. */
export function boxCenter(box: BoundingBox): Point {
  'worklet';
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

/**
 * Normalised distance (0-1) of the box's centroid from the frame's centre,
 * where 0 = perfectly centred and 1 = centroid at a frame corner.
 */
export function boxCenterOffsetRatio(box: BoundingBox, frameWidth: number, frameHeight: number): number {
  'worklet';
  const center = boxCenter(box);
  const frameCenter = { x: frameWidth / 2, y: frameHeight / 2 };
  const maxOffset = Math.hypot(frameWidth / 2, frameHeight / 2);
  if (maxOffset === 0) return 0;
  return Math.min(1, distance(center, frameCenter) / maxOffset);
}

/**
 * Head tilt (roll), in degrees, measured as the eye-to-eye line's deviation
 * from horizontal. 0 = eyes level. Only the magnitude is meaningful (the
 * only caller, `FaceGuidanceEngine`, takes `Math.abs()` of this).
 *
 * Deliberately uses `Math.abs(dx)` in the `atan2` call, NOT signed `dx`.
 * The native side mirrors front-camera landmark positions to match the
 * selfie preview (see `HybridDocScanner`'s `orientedPoint`/`isMirrored`
 * handling) without relabeling which point is "leftEye" vs "rightEye" —
 * so depending on the camera, `dx` can legitimately come back negative
 * for a level face. With signed `dx`, a sign flip sends `atan2(~0, dx)`
 * from ~0° to ~180° (a real bug this fixed: the front camera reported
 * "hold your head level" on every frame, since a level face always
 * measured as ~180° tilted). Using `Math.abs(dx)` measures deviation from
 * horizontal regardless of which eye ends up with the larger x.
 */
export function eyeLineTiltDeg(landmarks: FaceLandmarks): number {
  'worklet';
  const dx = Math.abs(landmarks.leftEye.x - landmarks.rightEye.x);
  const dy = landmarks.leftEye.y - landmarks.rightEye.y;
  return (Math.atan2(dy, dx) * 180) / Math.PI;
}
