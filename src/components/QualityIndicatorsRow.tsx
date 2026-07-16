import { StyleSheet, Text, View } from 'react-native';
import { SCANNER_THEME } from '../constants/theme';
import { describeQualityIndicators } from '../opencv/qualityAnalysis';
import type { QualityMetrics } from '../types/detection';

export interface QualityIndicatorsRowProps {
  metrics: QualityMetrics;
}

/** Row of small pass/fail dots for blur, brightness, glare, stability, distance, perspective. */
export function QualityIndicatorsRow({ metrics }: QualityIndicatorsRowProps) {
  const indicators = describeQualityIndicators(metrics);

  return (
    <View style={styles.row}>
      {indicators.map((indicator) => (
        <View key={indicator.key} style={styles.item}>
          <View
            style={[
              styles.dot,
              { backgroundColor: indicator.passed ? SCANNER_THEME.accent : SCANNER_THEME.danger },
            ]}
          />
          <Text style={styles.label}>{indicator.label}</Text>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 12,
    paddingHorizontal: 16,
  },
  item: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  label: {
    color: SCANNER_THEME.textSecondary,
    fontSize: 11,
    fontWeight: '500',
  },
});
