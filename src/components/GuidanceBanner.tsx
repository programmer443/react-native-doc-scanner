import { useEffect } from 'react';
import { StyleSheet, Text } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withSequence,
} from 'react-native-reanimated';
import { SCANNER_THEME } from '../constants/theme';

export interface GuidanceBannerProps {
  /** Any guidance-code enum's string value — only used to key the pulse animation, never switched on. */
  code: string;
  message: string;
  isValid: boolean;
}

/** Single-line, always-current instruction banner — the text spec calls out (e.g. "Hold still", "Scanning..."). */
export function GuidanceBanner({ code, message, isValid }: GuidanceBannerProps) {
  const pulse = useSharedValue(1);

  useEffect(() => {
    pulse.value = withSequence(withTiming(1.04, { duration: 120 }), withTiming(1, { duration: 160 }));
  }, [code, pulse]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulse.value }],
  }));

  const accentColor = isValid ? SCANNER_THEME.accent : SCANNER_THEME.textPrimary;

  return (
    <Animated.View style={[styles.container, animatedStyle]}>
      <Animated.View style={[styles.dot, { backgroundColor: accentColor }]} />
      <Text style={[styles.text, { color: accentColor }]} numberOfLines={1}>
        {message}
      </Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'center',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 24,
    backgroundColor: SCANNER_THEME.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: SCANNER_THEME.border,
    gap: 8,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  text: {
    fontSize: 15,
    fontWeight: '600',
    letterSpacing: 0.2,
  },
});
