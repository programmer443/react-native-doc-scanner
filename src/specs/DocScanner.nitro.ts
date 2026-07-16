import type { HybridObject } from 'react-native-nitro-modules';
import type { Frame } from 'react-native-vision-camera';
import type { Quad } from '../types/detection';
import type { NativeFrameResult } from '../vision/nativeFrameResult';
import type { NativeFaceFrameResult } from '../vision/nativeFaceFrameResult';

export interface ModelPaths {
  detectorModelPath: string;
  ocrDetModelPath: string;
  ocrClsModelPath: string;
  ocrRecModelPath: string;
  ocrRecCharsetPath: string;
  /**
   * YuNet ONNX face detector (see docs/MODEL_TRAINING.md §5) — optional so
   * document-only consumers aren't forced to bundle it. Pass `''` (or omit
   * fetching it) to skip loading; `analyzeFaceFrame` then always reports
   * `detected: false` instead of throwing.
   */
  faceDetectorModelPath: string;
  /**
   * Document-type classifier (see docs/MODEL_TRAINING.md §6) — optional, same
   * convention as `faceDetectorModelPath`. Pass `''` to skip loading; `analyzeFrame`
   * then always reports `documentType: "GENERIC"`, same as today.
   */
  classifierModelPath: string;
}

export interface LoadModelsResult {
  success: boolean;
  detectorVersion: string;
  ocrVersion: string;
  /** Empty string when `faceDetectorModelPath` was empty/not loaded. */
  faceDetectorVersion: string;
  /** Empty string when `classifierModelPath` was empty/not loaded. */
  classifierVersion: string;
}

export interface OcrTextLineNative {
  text: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface RawOcrResultNative {
  fullText: string;
  lines: OcrTextLineNative[];
  confidence: number;
  rectifiedImagePath: string;
}

/**
 * Native document-scanner engine: real-time detection + quality analysis
 * (per-frame, called synchronously from a `useFrameOutput` worklet) plus the
 * post-capture perspective-correction + OCR pipeline (async, called once per
 * capture). One HybridObject backs both — see ios/HybridDocScanner.swift and
 * android/.../HybridDocScanner.kt.
 */
export interface DocScanner extends HybridObject<{ ios: 'swift'; android: 'kotlin' }> {
  /**
   * Runs the YOLO/DocAligner detector + OpenCV quality analysis (blur,
   * brightness, glare, motion, perspective) on one camera frame. Must be
   * called from a worklet on the camera's own frame-processing thread (see
   * `useFrameOutput`) — this is a synchronous, zero-copy JSI call, never an
   * async bridge round trip.
   *
   * `NativeFrameResult.detection.documentType` reflects the document-type
   * classifier's result (see `classifierModelPath`) once a document is
   * confidently detected; it stays `"GENERIC"` if no classifier is loaded.
   */
  analyzeFrame(frame: Frame): NativeFrameResult;

  /**
   * Runs the YuNet ONNX face detector + the same OpenCV quality analysis
   * `analyzeFrame` uses (blur, brightness, glare, motion — computed over the
   * whole frame here rather than a document ROI) on one camera frame. Same
   * synchronous, zero-copy worklet-thread contract as `analyzeFrame`; call
   * this instead of `analyzeFrame`, not alongside it — only one model needs
   * to run per frame depending on whether the screen is scanning a document
   * or guiding a selfie.
   */
  analyzeFaceFrame(frame: Frame): NativeFaceFrameResult;

  /**
   * (Re)loads the ONNX models used for detection + OCR (+ face detection, if
   * `faceDetectorModelPath` is set, and document-type classification, if
   * `classifierModelPath` is set).
   */
  loadModels(config: ModelPaths): Promise<LoadModelsResult>;

  /**
   * Runs OpenCV perspective correction (when `quad` is provided) followed by
   * the RapidOCR detection→classification→recognition pipeline on a
   * full-resolution captured photo.
   */
  captureAndExtract(
    photoPath: string,
    documentType: string,
    quad: Quad | undefined,
  ): Promise<RawOcrResultNative>;
}
