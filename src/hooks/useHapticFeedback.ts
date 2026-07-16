import { useCallback } from 'react';
import { Vibration } from 'react-native';

export type HapticEvent = 'guidanceChanged' | 'captureStarted' | 'success' | 'error';

type HapticFeedbackModule = {
  trigger: (type: string, options?: Record<string, unknown>) => void;
};

// `react-native-haptic-feedback` is an optional peer dep — most banking-app
// teams already have a haptics library of choice, so we use it when present
// and fall back to core RN `Vibration` (works everywhere, no extra native
// linking) rather than forcing one specific package on every consumer.
let hapticModule: HapticFeedbackModule | null = null;
try {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  hapticModule = require('react-native-haptic-feedback').default as HapticFeedbackModule;
} catch {
  hapticModule = null;
}

const EVENT_TO_TRIGGER: Record<HapticEvent, string> = {
  guidanceChanged: 'impactLight',
  captureStarted: 'impactMedium',
  success: 'notificationSuccess',
  error: 'notificationError',
};

const FALLBACK_DURATION_MS: Record<HapticEvent, number> = {
  guidanceChanged: 8,
  captureStarted: 15,
  success: 25,
  error: 40,
};

/** Fires haptic feedback for a scanner lifecycle event, iOS/Android-safe either way. */
export function useHapticFeedback() {
  return useCallback((event: HapticEvent) => {
    if (hapticModule) {
      hapticModule.trigger(EVENT_TO_TRIGGER[event], {
        enableVibrateFallback: true,
        ignoreAndroidSystemSettings: false,
      });
      return;
    }
    Vibration.vibrate(FALLBACK_DURATION_MS[event]);
  }, []);
}
