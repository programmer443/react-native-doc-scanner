import type { DocumentType, Quad } from '../types/detection';

/**
 * Struct returned by the native `DocScanner` Nitro HybridObject's
 * `analyzeFrame` method on every call. This is a genuine Nitro struct (plain
 * TS interface — nitrogen generates the matching Swift/Kotlin/C++ types), not
 * a hand-flattened JSI object, so nested types like `Quad` are fine.
 *
 * iOS: HybridDocScanner.swift builds this.
 * Android: HybridDocScanner.kt builds this.
 *
 * IMPORTANT — orientation contract: `quad`/`frameWidth`/`frameHeight` must
 * already be normalised to the preview's display orientation (i.e.
 * accounting for `frame.orientation` / sensor rotation and front-camera
 * mirroring) before returning, so `frameWidth`/`frameHeight` describe the
 * buffer *as the user sees it* (portrait-upright ⇒ width < height). This
 * keeps all downstream JS (guidance math, Skia overlay scaling) a plain
 * linear scale from frame space to view space — no rotation/mirroring logic
 * in JS.
 */
export interface NativeFrameResult {
  detected: boolean;
  documentType: string;
  confidence: number;
  quad: Quad | undefined;
  frameWidth: number;
  frameHeight: number;
  blurScore: number;
  brightness: number;
  glareRatio: number;
  motionScore: number;
  distanceRatio: number;
  perspectiveSkewDeg: number;
  outOfFrameRatio: number;
}

export const EMPTY_NATIVE_FRAME_RESULT: NativeFrameResult = {
  detected: false,
  documentType: 'GENERIC' as DocumentType,
  confidence: 0,
  quad: undefined,
  frameWidth: 0,
  frameHeight: 0,
  blurScore: 0,
  brightness: 0,
  glareRatio: 0,
  motionScore: 0,
  distanceRatio: 0,
  perspectiveSkewDeg: 0,
  outOfFrameRatio: 0,
};
