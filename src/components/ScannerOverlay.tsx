import { useEffect, useMemo } from 'react';
import { StyleSheet, View } from 'react-native';
import { Canvas, Group, Path, Skia, Line, vec } from '@shopify/react-native-skia';
import {
  useSharedValue,
  useDerivedValue,
  withTiming,
  withRepeat,
  interpolateColor,
} from 'react-native-reanimated';
import type { Quad } from '../types/detection';
import type { CaptureStage } from '../types/guidance';

export interface ScannerOverlayProps {
  /** Rendered size of the camera preview this overlay sits on top of. */
  width: number;
  height: number;
  /** Latest detected quad, in the native frame's (orientation-normalised) pixel space. */
  quad: Quad | null;
  frameWidth: number;
  frameHeight: number;
  isValid: boolean;
  captureStage: CaptureStage;
  /** Guide-rectangle size as a fraction of the overlay width, 0-1. */
  guideWidthRatio?: number;
  /** Guide-rectangle aspect ratio (width / height) — defaults to an ID-card/passport bio-page ratio. */
  guideAspectRatio?: number;
}

const INVALID_COLOR = '#FF4D4F';
const VALID_COLOR = '#33D690';
const CORNER_LENGTH = 28;
const CORNER_RADIUS = 16;
const STROKE_WIDTH = 4;

/**
 * Skia-drawn camera overlay: dimmed mask, animated guide rectangle with
 * corner brackets, a live quad tracking the detected document (green when
 * every quality gate passes, red otherwise), and a scanning sweep while
 * capturing/processing. Pure presentation — all detection/quality logic
 * lives in GuidanceEngine; this component only visualises it.
 */
export function ScannerOverlay({
  width,
  height,
  quad,
  frameWidth,
  frameHeight,
  isValid,
  captureStage,
  guideWidthRatio = 0.86,
  guideAspectRatio = 1.42,
}: ScannerOverlayProps) {
  const guide = useMemo(() => {
    const guideWidth = width * guideWidthRatio;
    const guideHeight = guideWidth / guideAspectRatio;
    const x = (width - guideWidth) / 2;
    const y = (height - guideHeight) / 2;
    return { x, y, width: guideWidth, height: guideHeight };
  }, [width, height, guideWidthRatio, guideAspectRatio]);

  const validProgress = useSharedValue(0);
  useEffect(() => {
    validProgress.value = withTiming(isValid ? 1 : 0, { duration: 200 });
  }, [isValid, validProgress]);

  const borderColor = useDerivedValue(() =>
    interpolateColor(validProgress.value, [0, 1], [INVALID_COLOR, VALID_COLOR]),
  );

  const scaleX = frameWidth > 0 ? width / frameWidth : 0;
  const scaleY = frameHeight > 0 ? height / frameHeight : 0;

  const tlX = useSharedValue(guide.x);
  const tlY = useSharedValue(guide.y);
  const trX = useSharedValue(guide.x + guide.width);
  const trY = useSharedValue(guide.y);
  const brX = useSharedValue(guide.x + guide.width);
  const brY = useSharedValue(guide.y + guide.height);
  const blX = useSharedValue(guide.x);
  const blY = useSharedValue(guide.y + guide.height);

  useEffect(() => {
    const target = quad
      ? {
          tlX: quad.topLeft.x * scaleX,
          tlY: quad.topLeft.y * scaleY,
          trX: quad.topRight.x * scaleX,
          trY: quad.topRight.y * scaleY,
          brX: quad.bottomRight.x * scaleX,
          brY: quad.bottomRight.y * scaleY,
          blX: quad.bottomLeft.x * scaleX,
          blY: quad.bottomLeft.y * scaleY,
        }
      : {
          tlX: guide.x,
          tlY: guide.y,
          trX: guide.x + guide.width,
          trY: guide.y,
          brX: guide.x + guide.width,
          brY: guide.y + guide.height,
          blX: guide.x,
          blY: guide.y + guide.height,
        };

    const duration = 120;
    tlX.value = withTiming(target.tlX, { duration });
    tlY.value = withTiming(target.tlY, { duration });
    trX.value = withTiming(target.trX, { duration });
    trY.value = withTiming(target.trY, { duration });
    brX.value = withTiming(target.brX, { duration });
    brY.value = withTiming(target.brY, { duration });
    blX.value = withTiming(target.blX, { duration });
    blY.value = withTiming(target.blY, { duration });
  }, [quad, scaleX, scaleY, guide, tlX, tlY, trX, trY, brX, brY, blX, blY]);

  const quadPath = useDerivedValue(() => {
    const path = Skia.Path.Make();
    path.moveTo(tlX.value, tlY.value);
    path.lineTo(trX.value, trY.value);
    path.lineTo(brX.value, brY.value);
    path.lineTo(blX.value, blY.value);
    path.close();
    return path;
  });

  const isScanning = captureStage === 'stabilizing' || captureStage === 'processing' || captureStage === 'capturing';
  const sweepProgress = useSharedValue(0);
  useEffect(() => {
    if (isScanning) {
      sweepProgress.value = withRepeat(withTiming(1, { duration: 1400 }), -1, true);
    } else {
      sweepProgress.value = 0;
    }
  }, [isScanning, sweepProgress]);

  const sweepP1 = useDerivedValue(() => vec(guide.x + 8, guide.y + sweepProgress.value * guide.height));
  const sweepP2 = useDerivedValue(() =>
    vec(guide.x + guide.width - 8, guide.y + sweepProgress.value * guide.height),
  );
  const sweepOpacity = useDerivedValue(() =>
    isScanning ? interpolateOpacityEdges(sweepProgress.value) : 0,
  );

  const cornerPath = useMemo(() => buildCornerBracketsPath(guide, CORNER_LENGTH, CORNER_RADIUS), [guide]);

  return (
    <View pointerEvents="none" style={[StyleSheet.absoluteFill, { width, height }]}>
      <Canvas style={StyleSheet.absoluteFill}>
        {/* Dimmed mask outside the guide rectangle, as 4 opaque strips. */}
        <Group color="rgba(0,0,0,0.55)">
          <Path path={rectPath(0, 0, width, guide.y)} />
          <Path path={rectPath(0, guide.y + guide.height, width, height - guide.y - guide.height)} />
          <Path path={rectPath(0, guide.y, guide.x, guide.height)} />
          <Path
            path={rectPath(guide.x + guide.width, guide.y, width - guide.x - guide.width, guide.height)}
          />
        </Group>

        {/* Static corner brackets marking the ideal guide rectangle. */}
        <Path
          path={cornerPath}
          style="stroke"
          strokeWidth={STROKE_WIDTH}
          color="rgba(255,255,255,0.9)"
          strokeCap="round"
          strokeJoin="round"
        />

        {/* Live tracked quad — green when every quality gate passes. */}
        <Path path={quadPath} style="stroke" strokeWidth={STROKE_WIDTH} color={borderColor} strokeJoin="round" />

        {/* Scanning sweep line while stabilizing/capturing/processing. */}
        <Line p1={sweepP1} p2={sweepP2} color="rgba(51,214,144,0.9)" strokeWidth={2} opacity={sweepOpacity} />
      </Canvas>
    </View>
  );
}

function rectPath(x: number, y: number, w: number, h: number) {
  'worklet';
  const path = Skia.Path.Make();
  path.addRect({ x, y, width: Math.max(0, w), height: Math.max(0, h) });
  return path;
}

function buildCornerBracketsPath(
  guide: { x: number; y: number; width: number; height: number },
  length: number,
  radius: number,
) {
  const path = Skia.Path.Make();
  const { x, y, width: w, height: h } = guide;

  // Top-left
  path.moveTo(x, y + length);
  path.arcToTangent(x, y, x + radius, y, radius);
  path.lineTo(x + length, y);

  // Top-right
  path.moveTo(x + w - length, y);
  path.arcToTangent(x + w, y, x + w, y + radius, radius);
  path.lineTo(x + w, y + length);

  // Bottom-right
  path.moveTo(x + w, y + h - length);
  path.arcToTangent(x + w, y + h, x + w - radius, y + h, radius);
  path.lineTo(x + w - length, y + h);

  // Bottom-left
  path.moveTo(x + length, y + h);
  path.arcToTangent(x, y + h, x, y + h - radius, radius);
  path.lineTo(x, y + h - length);

  return path;
}

function interpolateOpacityEdges(progress: number): number {
  'worklet';
  // Fade in/out near the top and bottom of the sweep so it doesn't hard-cut.
  const fadeZone = 0.12;
  if (progress < fadeZone) return progress / fadeZone;
  if (progress > 1 - fadeZone) return (1 - progress) / fadeZone;
  return 1;
}
