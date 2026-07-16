import { useState } from 'react';
import { StyleSheet, View, Text, useWindowDimensions, ActivityIndicator } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import { useSelfieCapture } from '../hooks/useSelfieCapture';
import { useScannerPermissions } from '../hooks/useScannerPermissions';
import { FaceOvalOverlay } from '../components/FaceOvalOverlay';
import { GuidanceBanner } from '../components/GuidanceBanner';
import { CaptureProgressRing } from '../components/CaptureProgressRing';
import { SuccessCheckAnimation } from '../components/SuccessCheckAnimation';
import { SCANNER_THEME } from '../constants/theme';

export interface SelfieCaptureScreenProps {
  onCaptured: (photo: { path: string }) => void;
  onCancel?: () => void;
}

/**
 * Ready-to-use selfie-guide screen wiring the front camera, native face
 * detection, guidance UI, and auto-capture into one component — mirrors
 * `DocumentScannerScreen.tsx`'s structure exactly. Apps that want a custom
 * layout should compose the same pieces (useSelfieCapture + FaceOvalOverlay
 * + GuidanceBanner, etc.) directly instead of this screen.
 */
export function SelfieCaptureScreen({ onCaptured, onCancel }: SelfieCaptureScreenProps) {
  const { width, height } = useWindowDimensions();
  const { hasPermission } = useScannerPermissions();
  const device = useCameraDevice('front');
  const [showSuccess, setShowSuccess] = useState(false);

  const {
    frameOutput,
    photoOutput,
    modelReady,
    guidance,
    box,
    frameSize,
    captureStage,
    stabilityProgress,
    lastError,
  } = useSelfieCapture({
    onCaptured: (photo) => {
      setShowSuccess(true);
      setTimeout(() => onCaptured(photo), 900);
    },
  });

  if (!hasPermission) {
    return (
      <View style={styles.centered}>
        <Text style={styles.message}>Camera permission is required to take a selfie.</Text>
      </View>
    );
  }

  if (!device || !modelReady) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator color={SCANNER_THEME.accent} />
        <Text style={styles.message}>{!device ? 'No front camera available' : 'Loading scanner models...'}</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        outputs={[frameOutput, photoOutput]}
      />

      <FaceOvalOverlay
        width={width}
        height={height}
        box={box}
        frameWidth={frameSize.width}
        frameHeight={frameSize.height}
        isValid={guidance.isValid}
        captureStage={captureStage}
      />

      <View style={styles.bottomBar} pointerEvents="box-none">
        <GuidanceBanner code={guidance.code} message={guidance.message} isValid={guidance.isValid} />
        {captureStage === 'stabilizing' && (
          <View style={styles.progressRing}>
            <CaptureProgressRing progress={stabilityProgress} />
          </View>
        )}
        {lastError && <Text style={styles.error}>{lastError}</Text>}
      </View>

      {showSuccess && (
        <View style={styles.successOverlay} pointerEvents="none">
          <SuccessCheckAnimation visible={showSuccess} />
        </View>
      )}

      {onCancel && (
        <Text style={styles.cancel} onPress={onCancel}>
          Cancel
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: SCANNER_THEME.background,
  },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: SCANNER_THEME.background,
    gap: 12,
    padding: 24,
  },
  message: {
    color: SCANNER_THEME.textSecondary,
    fontSize: 15,
    textAlign: 'center',
  },
  bottomBar: {
    position: 'absolute',
    bottom: 56,
    left: 0,
    right: 0,
    alignItems: 'center',
    gap: 16,
  },
  progressRing: {
    alignItems: 'center',
  },
  error: {
    color: SCANNER_THEME.danger,
    fontSize: 13,
    textAlign: 'center',
    paddingHorizontal: 24,
  },
  successOverlay: {
    ...StyleSheet.absoluteFill,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(11,13,16,0.72)',
  },
  cancel: {
    position: 'absolute',
    top: 56,
    right: 20,
    color: SCANNER_THEME.textSecondary,
    fontSize: 14,
    fontWeight: '600',
  },
});
