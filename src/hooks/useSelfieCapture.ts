import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useFrameOutput, usePhotoOutput } from 'react-native-vision-camera';
import { useSharedValue, runOnJS } from 'react-native-reanimated';
import { DocScannerNative } from '../vision/docScannerNative';
import { mapNativeFaceFrameResult } from '../vision/mapNativeFaceFrameResult';
import { evaluateFaceGuidance } from '../services/FaceGuidanceEngine';
import { AutoCaptureController } from '../services/AutoCaptureController';
import { ModelManager } from '../models/ModelManager';
import { useFaceScannerStore } from '../store/faceScannerStore';
import { useHapticFeedback } from './useHapticFeedback';
import { FACE_THRESHOLDS } from '../constants/faceThresholds';
import type { BoundingBox } from '../types/detection';
import type { FaceGuidanceState } from '../types/faceGuidance';

export interface UseSelfieCaptureOptions {
  /** Auto-capture once every quality gate passes for ~1s. Defaults to true — this flow has no manual-capture fallback. */
  autoCapture?: boolean;
  onCaptured?: (photo: { path: string }) => void;
  onError?: (error: Error) => void;
}

/**
 * Selfie-guide counterpart to `useDocumentScanner` — same camera-output/
 * frame-processor/auto-capture architecture, but detects a face (YuNet ONNX)
 * instead of a document, and captures a plain photo with no OCR step
 * afterward (see `docs/MODEL_TRAINING.md` §5 for the face detector).
 *
 * Returns `frameOutput`/`photoOutput` for the screen to pass into
 * `<Camera outputs={[frameOutput, photoOutput]} />` — this hook does not
 * render anything itself.
 */
export function useSelfieCapture({
  autoCapture = true,
  onCaptured,
  onError,
}: UseSelfieCaptureOptions) {
  const capturingRef = useRef(false);
  const captureRef = useRef<() => Promise<void>>(async () => {});
  const haptic = useHapticFeedback();
  const [modelReady, setModelReady] = useState(false);

  const guidance = useFaceScannerStore((s) => s.guidance);
  const box = useFaceScannerStore((s) => s.box);
  const frameSize = useFaceScannerStore((s) => s.frameSize);
  const captureStage = useFaceScannerStore((s) => s.captureStage);
  const stabilityProgress = useFaceScannerStore((s) => s.stabilityProgress);
  const lastCapturedPath = useFaceScannerStore((s) => s.lastCapturedPath);
  const lastError = useFaceScannerStore((s) => s.lastError);
  const setGuidance = useFaceScannerStore((s) => s.setGuidance);
  const setCaptureStage = useFaceScannerStore((s) => s.setCaptureStage);
  const setStabilityProgress = useFaceScannerStore((s) => s.setStabilityProgress);
  const setLastCapturedPath = useFaceScannerStore((s) => s.setLastCapturedPath);
  const setLastError = useFaceScannerStore((s) => s.setLastError);
  const resetStore = useFaceScannerStore((s) => s.reset);

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
      new AutoCaptureController(
        {
          onStageChange: setCaptureStage,
          onProgress: setStabilityProgress,
          onCapture: () => {
            captureRef.current();
          },
        },
        FACE_THRESHOLDS.autoCaptureStableMs,
      ),
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

      setLastCapturedPath(filePath);
      autoCaptureController.markCompleted();
      haptic('success');
      onCaptured?.({ path: filePath });
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      setLastError(err.message);
      haptic('error');
      onError?.(err);
      autoCaptureController.markFailed();
    } finally {
      capturingRef.current = false;
    }
  }, [autoCaptureController, haptic, onCaptured, onError, photoOutput, setLastCapturedPath, setLastError]);

  useEffect(() => {
    captureRef.current = capture;
  }, [capture]);

  // Runs on the JS thread — `runOnJS` schedules it from the worklet below,
  // throttled so React only re-renders ~15x/sec instead of every frame.
  const onFrameAnalysedJS = useCallback(
    (guidanceState: FaceGuidanceState, detectedBox: BoundingBox | null, frameWidth: number, frameHeight: number) => {
      setGuidance(guidanceState, detectedBox, { width: frameWidth, height: frameHeight });
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
      const nativeResult = DocScannerNative.analyzeFaceFrame(frame);
      const analysis = mapNativeFaceFrameResult(nativeResult, Date.now());
      const guidanceState = evaluateFaceGuidance(analysis);
      frame.dispose();

      const now = Date.now();
      if (now - lastForwardedAtMs.value >= FACE_THRESHOLDS.guidanceThrottleMs) {
        lastForwardedAtMs.value = now;
        runOnJS(onFrameAnalysedJS)(
          guidanceState,
          analysis.detection.box,
          analysis.detection.frameWidth,
          analysis.detection.frameHeight,
        );
      }
    },
  });

  const reset = useCallback(() => {
    resetStore();
    autoCaptureController.reset();
  }, [autoCaptureController, resetStore]);

  return {
    frameOutput,
    photoOutput,
    modelReady,
    guidance,
    box,
    frameSize,
    captureStage,
    stabilityProgress,
    lastCapturedPath,
    lastError,
    reset,
  };
}
