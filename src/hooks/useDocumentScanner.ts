import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useFrameOutput, usePhotoOutput } from 'react-native-vision-camera';
import { useSharedValue, runOnJS } from 'react-native-reanimated';
import { DocScannerNative } from '../vision/docScannerNative';
import { mapNativeFrameResult } from '../vision/mapNativeFrameResult';
import { evaluateGuidance } from '../services/GuidanceEngine';
import { AutoCaptureController } from '../services/AutoCaptureController';
import { extractDocumentData } from '../ocr/OcrService';
import { ModelManager } from '../models/ModelManager';
import { useScannerStore } from '../store/scannerStore';
import { useHapticFeedback } from './useHapticFeedback';
import { SCANNER_THRESHOLDS } from '../constants/thresholds';
import { DocumentType, type Quad } from '../types/detection';
import type { GuidanceState } from '../types/guidance';
import type { OcrExtractionResult } from '../types/ocr';

export interface UseDocumentScannerOptions {
  documentType: DocumentType;
  /** Auto-capture once every quality gate passes for ~1s. Defaults to true. */
  autoCapture?: boolean;
  onCaptured?: (result: OcrExtractionResult) => void;
  onError?: (error: Error) => void;
}

/**
 * Top-level hook wiring the camera outputs, native detection, guidance
 * engine, auto-capture gate, and OCR extraction into one surface.
 *
 * Built on VisionCamera Core's session/output model: `useFrameOutput`'s
 * `onFrame` runs as a worklet on the camera's own capture thread and calls
 * `DocScannerNative.analyzeFrame(frame)` — a direct, synchronous Nitro JSI
 * call into native ONNX/OpenCV code, never a JS-bridge round trip. Motion
 * detection needs consecutive frames, so analysis runs on every frame; only
 * the hop back to the JS thread (via `runOnJS`, ~15Hz) is throttled.
 *
 * Returns `frameOutput`/`photoOutput` for the screen to pass into
 * `<Camera outputs={[frameOutput, photoOutput]} />` — this hook does not
 * render anything itself.
 */
export function useDocumentScanner({
  documentType,
  autoCapture = true,
  onCaptured,
  onError,
}: UseDocumentScannerOptions) {
  const capturingRef = useRef(false);
  const lastQuadRef = useRef<Quad | null>(null);
  const captureRef = useRef<() => Promise<void>>(async () => {});
  const haptic = useHapticFeedback();
  const [modelReady, setModelReady] = useState(false);

  const guidance = useScannerStore((s) => s.guidance);
  const quad = useScannerStore((s) => s.quad);
  const frameSize = useScannerStore((s) => s.frameSize);
  const captureStage = useScannerStore((s) => s.captureStage);
  const stabilityProgress = useScannerStore((s) => s.stabilityProgress);
  const lastResult = useScannerStore((s) => s.lastResult);
  const lastError = useScannerStore((s) => s.lastError);
  const setGuidance = useScannerStore((s) => s.setGuidance);
  const setCaptureStage = useScannerStore((s) => s.setCaptureStage);
  const setStabilityProgress = useScannerStore((s) => s.setStabilityProgress);
  const setLastResult = useScannerStore((s) => s.setLastResult);
  const setLastError = useScannerStore((s) => s.setLastError);
  const resetStore = useScannerStore((s) => s.reset);

  useEffect(() => {
    let cancelled = false;
    ModelManager.activate()
      .then(() => {
        if (!cancelled) setModelReady(true);
      })
      .catch((e) => onError?.(e instanceof Error ? e : new Error(String(e))));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const autoCaptureController = useMemo(
    () =>
      new AutoCaptureController({
        onStageChange: setCaptureStage,
        onProgress: setStabilityProgress,
        onCapture: () => {
          captureRef.current();
        },
      }),
    [setCaptureStage, setStabilityProgress],
  );

  const photoOutput = usePhotoOutput({});

  const capture = useCallback(async () => {
    if (capturingRef.current) return;
    capturingRef.current = true;
    haptic('captureStarted');

    try {
      const photoFile = await photoOutput.capturePhotoToFile({ flashMode: 'off' }, {});
      autoCaptureController.markProcessing();

      const filePath = photoFile.filePath.startsWith('file://')
        ? photoFile.filePath
        : `file://${photoFile.filePath}`;
      const result = await extractDocumentData(filePath, documentType, lastQuadRef.current);

      if (!result.success) {
        throw new Error('Could not read the document. Retake the photo or try manual entry.');
      }

      setLastResult(result);
      autoCaptureController.markCompleted();
      haptic('success');
      onCaptured?.(result);
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      setLastError(err.message);
      haptic('error');
      onError?.(err);
      autoCaptureController.markFailed();
    } finally {
      capturingRef.current = false;
    }
  }, [autoCaptureController, documentType, haptic, onCaptured, onError, photoOutput, setLastError, setLastResult]);

  useEffect(() => {
    captureRef.current = capture;
  }, [capture]);

  // Runs on the JS thread — `runOnJS` schedules it from the worklet below,
  // throttled so React only re-renders ~15x/sec instead of every frame.
  const onFrameAnalysedJS = useCallback(
    (guidanceState: GuidanceState, detectedQuad: Quad | null, frameWidth: number, frameHeight: number) => {
      lastQuadRef.current = detectedQuad;
      setGuidance(guidanceState, detectedQuad, { width: frameWidth, height: frameHeight });
      if (autoCapture) {
        autoCaptureController.update(guidanceState.isValid);
      }
    },
    [autoCapture, autoCaptureController, setGuidance],
  );

  const lastForwardedAtMs = useSharedValue(0);

  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',
    onFrame(frame) {
      'worklet';
      const nativeResult = DocScannerNative.analyzeFrame(frame);
      const analysis = mapNativeFrameResult(nativeResult, Date.now());
      const guidanceState = evaluateGuidance(analysis, documentType);
      frame.dispose();

      const now = Date.now();
      if (now - lastForwardedAtMs.value >= SCANNER_THRESHOLDS.guidanceThrottleMs) {
        lastForwardedAtMs.value = now;
        runOnJS(onFrameAnalysedJS)(
          guidanceState,
          analysis.detection.quad,
          analysis.detection.frameWidth,
          analysis.detection.frameHeight,
        );
      }
    },
  });

  const reset = useCallback(() => {
    resetStore();
    autoCaptureController.reset();
    lastQuadRef.current = null;
  }, [autoCaptureController, resetStore]);

  return {
    frameOutput,
    photoOutput,
    modelReady,
    guidance,
    quad,
    frameSize,
    captureStage,
    stabilityProgress,
    lastResult,
    lastError,
    reset,
  };
}
