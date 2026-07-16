import { useState } from 'react';
import { StyleSheet, View, Text, useWindowDimensions, ActivityIndicator } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import { useDocumentScanner } from '../hooks/useDocumentScanner';
import { useScannerPermissions } from '../hooks/useScannerPermissions';
import { ScannerOverlay } from '../components/ScannerOverlay';
import { GuidanceBanner } from '../components/GuidanceBanner';
import { ConfidenceBadge } from '../components/ConfidenceBadge';
import { CaptureProgressRing } from '../components/CaptureProgressRing';
import { SuccessCheckAnimation } from '../components/SuccessCheckAnimation';
import { SCANNER_THEME } from '../constants/theme';
import { DocumentType } from '../types/detection';
import type { OcrExtractionResult } from '../types/ocr';

export interface DocumentScannerScreenProps {
  documentType: DocumentType;
  onCaptured: (result: OcrExtractionResult) => void;
  onCancel?: () => void;
}

/**
 * Ready-to-use scanning screen wiring the camera, native detection, guidance
 * UI, and auto-capture into one component. Apps that want a custom layout
 * should compose the same pieces (useDocumentScanner + ScannerOverlay +
 * GuidanceBanner, etc.) directly instead of this screen.
 */
export function DocumentScannerScreen({ documentType, onCaptured, onCancel }: DocumentScannerScreenProps) {
  const { width, height } = useWindowDimensions();
  const { hasPermission } = useScannerPermissions();
  const device = useCameraDevice('back');
  const [showSuccess, setShowSuccess] = useState(false);

  const {
    frameOutput,
    photoOutput,
    modelReady,
    guidance,
    quad,
    frameSize,
    captureStage,
    stabilityProgress,
    lastError,
  } = useDocumentScanner({
    documentType,
    onCaptured: (result) => {
      setShowSuccess(true);
      setTimeout(() => onCaptured(result), 900);
    },
  });

  if (!hasPermission) {
    return (
      <View style={styles.centered}>
        <Text style={styles.message}>Camera permission is required to scan documents.</Text>
      </View>
    );
  }

  if (!device || !modelReady) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator color={SCANNER_THEME.accent} />
        <Text style={styles.message}>{!device ? 'No camera available' : 'Loading scanner models...'}</Text>
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

      <ScannerOverlay
        width={width}
        height={height}
        quad={quad}
        frameWidth={frameSize.width}
        frameHeight={frameSize.height}
        isValid={guidance.isValid}
        captureStage={captureStage}
      />

      <View style={styles.topBar} pointerEvents="box-none">
        <ConfidenceBadge
          documentType={documentType}
          confidence={guidance.confidence}
          detected={guidance.flags.hasDocument}
        />
      </View>

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
  topBar: {
    position: 'absolute',
    top: 56,
    left: 20,
    right: 20,
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
