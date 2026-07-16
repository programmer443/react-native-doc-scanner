import { useEffect, useMemo } from 'react';
import { StyleSheet } from 'react-native';
import { Canvas, Path, Skia, Circle } from '@shopify/react-native-skia';
import Animated, { useSharedValue, useAnimatedStyle, withTiming, withSpring, Easing } from 'react-native-reanimated';
import { SCANNER_THEME } from '../constants/theme';

export interface SuccessCheckAnimationProps {
  size?: number;
  /** Mount this component only when `visible` transitions true — it animates in once. */
  visible: boolean;
}

/** Circular success checkmark shown when a document is fully captured + extracted. */
export function SuccessCheckAnimation({ size = 88, visible }: SuccessCheckAnimationProps) {
  const scale = useSharedValue(0);
  const checkProgress = useSharedValue(0);

  useEffect(() => {
    if (visible) {
      scale.value = withSpring(1, { damping: 12, stiffness: 180 });
      checkProgress.value = withTiming(1, { duration: 320, easing: Easing.out(Easing.cubic) });
    } else {
      scale.value = 0;
      checkProgress.value = 0;
    }
  }, [visible, scale, checkProgress]);

  const containerStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
    opacity: scale.value,
  }));

  const checkPath = useMemo(() => {
    const path = Skia.Path.Make();
    path.moveTo(size * 0.28, size * 0.53);
    path.lineTo(size * 0.44, size * 0.68);
    path.lineTo(size * 0.74, size * 0.34);
    return path;
  }, [size]);

  if (!visible) return null;

  return (
    <Animated.View style={[{ width: size, height: size }, containerStyle]}>
      <Canvas style={StyleSheet.absoluteFill}>
        <Circle cx={size / 2} cy={size / 2} r={size / 2 - 2} color={SCANNER_THEME.accent} opacity={0.16} />
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={size / 2 - 2}
          style="stroke"
          strokeWidth={2}
          color={SCANNER_THEME.accent}
        />
        <Path
          path={checkPath}
          style="stroke"
          strokeWidth={size * 0.07}
          strokeCap="round"
          strokeJoin="round"
          color={SCANNER_THEME.accent}
          start={0}
          end={checkProgress}
        />
      </Canvas>
    </Animated.View>
  );
}
