import { StyleSheet, Text, View } from 'react-native';
import { SCANNER_THEME } from '../constants/theme';
import { DocumentType } from '../types/detection';

export interface ConfidenceBadgeProps {
  documentType: DocumentType;
  confidence: number;
  detected: boolean;
}

const DOCUMENT_LABELS: Record<DocumentType, string> = {
  [DocumentType.PASSPORT]: 'Passport',
  [DocumentType.DRIVING_LICENCE]: 'Driving Licence',
  [DocumentType.ID_CARD]: 'ID Card',
  [DocumentType.RESIDENCE_PERMIT]: 'Residence Permit',
  [DocumentType.VISA]: 'Visa',
  [DocumentType.GENERIC]: 'Document',
};

/** Small pill showing the detector's document-type label + confidence %. */
export function ConfidenceBadge({ documentType, confidence, detected }: ConfidenceBadgeProps) {
  if (!detected) return null;

  return (
    <View style={styles.container}>
      <Text style={styles.label}>{DOCUMENT_LABELS[documentType]}</Text>
      <Text style={styles.confidence}>{Math.round(confidence * 100)}%</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 14,
    backgroundColor: SCANNER_THEME.surfaceElevated,
    gap: 6,
  },
  label: {
    color: SCANNER_THEME.textPrimary,
    fontSize: 12,
    fontWeight: '600',
  },
  confidence: {
    color: SCANNER_THEME.accent,
    fontSize: 12,
    fontWeight: '700',
  },
});
