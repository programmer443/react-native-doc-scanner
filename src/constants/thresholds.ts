/**
 * All tunable quality gates in one place. Values were chosen against 720p/1080p
 * VisionCamera frame-processor output on mid-range Android + iPhone 12+ and are
 * intentionally conservative — loosen them if auto-capture feels too strict for
 * your document mix.
 */
export const SCANNER_THRESHOLDS = {
  /** Minimum detector confidence to treat a frame as "document detected". */
  minDetectionConfidence: 0.55,

  /** Minimum classifier confidence before trusting its documentType over GENERIC. */
  minClassificationConfidence: 0.6,

  /** Laplacian variance below this is treated as blurry. */
  minBlurScore: 60,

  /** Acceptable mean-luminance band (0-255). */
  minBrightness: 60,
  maxBrightness: 220,

  /** Fraction of the ROI blown out by glare before we warn. */
  maxGlareRatio: 0.08,

  /** Frame-to-frame motion magnitude (0-1) below which the document is "still". */
  maxMotionScore: 0.04,

  /** documentArea / frameArea band for "correct distance". */
  minDistanceRatio: 0.35,
  maxDistanceRatio: 0.92,

  /** Normalised (0-1) distance of the document centroid from frame centre before we ask to re-centre. */
  maxCenterOffsetRatio: 0.18,

  /** Max corner-angle deviation from 90° before perspective is "skewed" (phone not parallel to document). */
  maxPerspectiveSkewDeg: 12,

  /** Max in-plane rotation (degrees) before the document needs re-aligning. */
  maxRotationDeg: 8,

  /** Fraction of the quad allowed to fall outside the visible frame. */
  maxOutOfFrameRatio: 0.02,

  /** Consecutive-valid duration required before auto-capture fires, per spec ("~1 second"). */
  autoCaptureStableMs: 1000,

  /** How often (ms) the JS thread is updated from the worklet frame processor. */
  guidanceThrottleMs: 66, // ~15 Hz UI updates; detector itself can run every frame
} as const;

export const MODEL_INPUT_SIZE = {
  /** DocAligner heatmap/point regression input — square, per upstream model card. */
  detectorWidth: 256,
  detectorHeight: 256,
  /** Document-type classifier input — see docs/MODEL_TRAINING.md §6. Update if your fine-tune uses a different size. */
  classifierWidth: 224,
  classifierHeight: 224,
} as const;
