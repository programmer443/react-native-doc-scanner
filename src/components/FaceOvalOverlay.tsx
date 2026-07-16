import { useEffect, useMemo } from 'react';
import { StyleSheet, View } from 'react-native';
import { Canvas, Path, Skia, FillType } from '@shopify/react-native-skia';
import {
  useSharedValue,
  useDerivedValue,
  withTiming,
  withRepeat,
  interpolateColor,
} from 'react-native-reanimated';
import type { BoundingBox } from '../types/detection';
import type { CaptureStage } from '../types/guidance';

export interface FaceOvalOverlayProps {
  /** Rendered size of the camera preview this overlay sits on top of. */
  width: number;
  height: number;
  /** Latest detected face box, in the native frame's (orientation-normalised) pixel space. */
  box: BoundingBox | null;
  frameWidth: number;
  frameHeight: number;
  isValid: boolean;
  captureStage: CaptureStage;
  /** Guide-oval width as a fraction of the overlay width, 0-1. */
  guideWidthRatio?: number;
  /** Guide-oval aspect ratio (width / height) — defaults to a natural face proportion. */
  guideAspectRatio?: number;
}

const INVALID_COLOR = '#FF4D4F';
const VALID_COLOR = '#33D690';
const STROKE_WIDTH = 4;

/**
 * Skia-drawn selfie overlay: dimmed mask with an oval cutout, an animated
 * oval outline tracking the detected face (green when every quality gate
 * passes, red otherwise), and a scanning pulse while stabilizing/capturing.
 * Pure presentation — all detection/quality logic lives in
 * FaceGuidanceEngine; this component only visualises it. Mirrors
 * `ScannerOverlay.tsx`'s structure/animation approach for a document quad.
 */
export function FaceOvalOverlay({
  width,
  height,
  box,
  frameWidth,
  frameHeight,
  isValid,
  captureStage,
  guideWidthRatio = 0.72,
  guideAspectRatio = 0.78,
}: FaceOvalOverlayProps) {
  const guide = useMemo(() => {
    const guideWidth = width * guideWidthRatio;
    const guideHeight = guideWidth / guideAspectRatio;
    const cx = width / 2;
    const cy = height / 2;
    return { cx, cy, rx: guideWidth / 2, ry: guideHeight / 2 };
  }, [width, height, guideWidthRatio, guideAspectRatio]);

  const validProgress = useSharedValue(0);
  useEffect(() => {
    validProgress.value = withTiming(isValid ? 1 : 0, { duration: 200 });
  }, [isValid, validProgress]);

  const ovalColor = useDerivedValue(() =>
    interpolateColor(validProgress.value, [0, 1], [INVALID_COLOR, VALID_COLOR]),
  );

  const scaleX = frameWidth > 0 ? width / frameWidth : 0;
  const scaleY = frameHeight > 0 ? height / frameHeight : 0;

  const cx = useSharedValue(guide.cx);
  const cy = useSharedValue(guide.cy);
  const rx = useSharedValue(guide.rx);
  const ry = useSharedValue(guide.ry);

  useEffect(() => {
    const target = box
      ? {
          cx: (box.x + box.width / 2) * scaleX,
          cy: (box.y + box.height / 2) * scaleY,
          rx: (box.width / 2) * scaleX,
          ry: (box.height / 2) * scaleY,
        }
      : { cx: guide.cx, cy: guide.cy, rx: guide.rx, ry: guide.ry };

    const duration = 120;
    cx.value = withTiming(target.cx, { duration });
    cy.value = withTiming(target.cy, { duration });
    rx.value = withTiming(target.rx, { duration });
    ry.value = withTiming(target.ry, { duration });
  }, [box, scaleX, scaleY, guide, cx, cy, rx, ry]);

  const trackedOvalPath = useDerivedValue(() => {
    const path = Skia.Path.Make();
    path.addOval({
      x: cx.value - rx.value,
      y: cy.value - ry.value,
      width: rx.value * 2,
      height: ry.value * 2,
    });
    return path;
  });

  const maskPath = useMemo(() => {
    const path = Skia.Path.Make();
    path.addRect({ x: 0, y: 0, width, height });
    path.addOval({
      x: guide.cx - guide.rx,
      y: guide.cy - guide.ry,
      width: guide.rx * 2,
      height: guide.ry * 2,
    });
    path.setFillType(FillType.EvenOdd);
    return path;
  }, [width, height, guide]);

  const guideOvalPath = useMemo(() => {
    const path = Skia.Path.Make();
    path.addOval({
      x: guide.cx - guide.rx,
      y: guide.cy - guide.ry,
      width: guide.rx * 2,
      height: guide.ry * 2,
    });
    return path;
  }, [guide]);

  const isScanning = captureStage === 'stabilizing' || captureStage === 'processing' || captureStage === 'capturing';
  const pulseProgress = useSharedValue(0);
  useEffect(() => {
    if (isScanning) {
      pulseProgress.value = withRepeat(withTiming(1, { duration: 900 }), -1, true);
    } else {
      pulseProgress.value = 0;
    }
  }, [isScanning, pulseProgress]);

  const pulseOpacity = useDerivedValue(() => (isScanning ? 0.25 + pulseProgress.value * 0.5 : 0));

  return (
    <View pointerEvents="none" style={[StyleSheet.absoluteFill, { width, height }]}>
      <Canvas style={StyleSheet.absoluteFill}>
        {/* Dimmed mask everywhere except the oval cutout. */}
        <Path path={maskPath} color="rgba(0,0,0,0.55)" />

        {/* Static guide oval marking where a face should sit. */}
        <Path
          path={guideOvalPath}
          style="stroke"
          strokeWidth={STROKE_WIDTH}
          color="rgba(255,255,255,0.6)"
        />

        {/* Live tracked oval — green when every quality gate passes. */}
        <Path
          path={trackedOvalPath}
          style="stroke"
          strokeWidth={STROKE_WIDTH}
          color={ovalColor}
        />

        {/* Scanning pulse while stabilizing/capturing/processing. */}
        <Path
          path={guideOvalPath}
          style="stroke"
          strokeWidth={STROKE_WIDTH * 2}
          color="rgba(51,214,144,0.9)"
          opacity={pulseOpacity}
        />
      </Canvas>
    </View>
  );
}
