import type { BoundingBox } from '../types/detection';
import type { FaceLandmarks } from '../types/face';

/**
 * Struct returned by the native `DocScanner` Nitro HybridObject's
 * `analyzeFaceFrame` method on every call — mirrors `NativeFrameResult`'s
 * contract exactly (see that file's doc comment for the full orientation
 * explanation): `box`/`landmarks`/`frameWidth`/`frameHeight` are already
 * normalised to the preview's display orientation (sensor rotation +
 * front-camera mirroring already applied natively), so downstream JS never
 * does rotation math.
 *
 * iOS: HybridDocScanner.swift builds this.
 * Android: HybridDocScanner.kt builds this.
 */
export interface NativeFaceFrameResult {
  detected: boolean;
  confidence: number;
  box: BoundingBox | undefined;
  landmarks: FaceLandmarks | undefined;
  frameWidth: number;
  frameHeight: number;
  blurScore: number;
  brightness: number;
  glareRatio: number;
  motionScore: number;
}

export const EMPTY_NATIVE_FACE_FRAME_RESULT: NativeFaceFrameResult = {
  detected: false,
  confidence: 0,
  box: undefined,
  landmarks: undefined,
  frameWidth: 0,
  frameHeight: 0,
  blurScore: 0,
  brightness: 0,
  glareRatio: 0,
  motionScore: 0,
};
