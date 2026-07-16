import { useEffect } from 'react';
import { StyleSheet, View } from 'react-native';
import { Canvas, Path, Skia, Group } from '@shopify/react-native-skia';
import { useDerivedValue, useSharedValue, withTiming } from 'react-native-reanimated';
import { SCANNER_THEME } from '../constants/theme';

export interface CaptureProgressRingProps {
  size?: number;
  strokeWidth?: number;
  /** 0-1 progress through the auto-capture stability window. */
  progress: number;
  /** Show a filled/success ring regardless of `progress`. */
  completed?: boolean;
}

/** Circular progress ring for the ~1s auto-capture stability countdown. */
export function CaptureProgressRing({
  size = 64,
  strokeWidth = 5,
  progress,
  completed = false,
}: CaptureProgressRingProps) {
  const animatedProgress = useSharedValue(0);

  useEffect(() => {
    animatedProgress.value = withTiming(completed ? 1 : progress, { duration: 100 });
  }, [progress, completed, animatedProgress]);

  const radius = (size - strokeWidth) / 2;
  const center = size / 2;

  const trackPath = useDerivedValue(() => {
    const path = Skia.Path.Make();
    path.addCircle(center, center, radius);
    return path;
  });

  const progressPath = useDerivedValue(() => {
    const path = Skia.Path.Make();
    const sweep = 360 * animatedProgress.value;
    path.addArc({ x: center - radius, y: center - radius, width: radius * 2, height: radius * 2 }, -90, sweep);
    return path;
  });

  return (
    <View style={{ width: size, height: size }}>
      <Canvas style={StyleSheet.absoluteFill}>
        <Group>
          <Path path={trackPath} style="stroke" strokeWidth={strokeWidth} color={SCANNER_THEME.border} />
          <Path
            path={progressPath}
            style="stroke"
            strokeWidth={strokeWidth}
            strokeCap="round"
            color={SCANNER_THEME.accent}
          />
        </Group>
      </Canvas>
    </View>
  );
}
