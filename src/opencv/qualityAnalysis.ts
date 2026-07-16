import { SCANNER_THRESHOLDS } from '../constants/thresholds';
import type { QualityMetrics } from '../types/detection';

/**
 * JS-side interpretation of the OpenCV-derived metrics the native frame
 * processor computes (Laplacian variance, brightness histogram, glare ratio,
 * motion, perspective). The heavy lifting (cv::Laplacian, cv::cvtColor,
 * cv::warpPerspective, etc.) happens natively — this module only turns those
 * numbers into UI-friendly signals.
 */

export interface QualityIndicator {
  key: keyof QualityMetrics;
  label: string;
  passed: boolean;
  /** 0-1, how far into the acceptable range the reading is. */
  score: number;
}

export function describeQualityIndicators(metrics: QualityMetrics): QualityIndicator[] {
  const t = SCANNER_THRESHOLDS;

  return [
    {
      key: 'blurScore',
      label: 'Sharpness',
      passed: metrics.blurScore >= t.minBlurScore,
      score: clamp01(metrics.blurScore / (t.minBlurScore * 2)),
    },
    {
      key: 'brightness',
      label: 'Lighting',
      passed: metrics.brightness >= t.minBrightness && metrics.brightness <= t.maxBrightness,
      score: clamp01(
        1 -
          Math.abs(metrics.brightness - (t.minBrightness + t.maxBrightness) / 2) /
            ((t.maxBrightness - t.minBrightness) / 2),
      ),
    },
    {
      key: 'glareRatio',
      label: 'Glare',
      passed: metrics.glareRatio <= t.maxGlareRatio,
      score: clamp01(1 - metrics.glareRatio / (t.maxGlareRatio * 4)),
    },
    {
      key: 'motionScore',
      label: 'Stability',
      passed: metrics.motionScore <= t.maxMotionScore,
      score: clamp01(1 - metrics.motionScore / (t.maxMotionScore * 4)),
    },
    {
      key: 'distanceRatio',
      label: 'Distance',
      passed: metrics.distanceRatio >= t.minDistanceRatio && metrics.distanceRatio <= t.maxDistanceRatio,
      score: clamp01(
        1 -
          Math.abs(metrics.distanceRatio - (t.minDistanceRatio + t.maxDistanceRatio) / 2) /
            ((t.maxDistanceRatio - t.minDistanceRatio) / 2),
      ),
    },
    {
      key: 'perspectiveSkewDeg',
      label: 'Perspective',
      passed: metrics.perspectiveSkewDeg <= t.maxPerspectiveSkewDeg,
      score: clamp01(1 - metrics.perspectiveSkewDeg / (t.maxPerspectiveSkewDeg * 2)),
    },
  ];
}

/** Composite 0-100 score for a single "confidence meter" style UI element. */
export function qualityScore(metrics: QualityMetrics): number {
  const indicators = describeQualityIndicators(metrics);
  const average = indicators.reduce((sum, i) => sum + i.score, 0) / indicators.length;
  return Math.round(average * 100);
}

function clamp01(n: number): number {
  if (Number.isNaN(n)) return 0;
  return Math.min(1, Math.max(0, n));
}
