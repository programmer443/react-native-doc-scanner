/**
 * Tunable quality gates for selfie capture — a self-contained sibling to
 * `SCANNER_THRESHOLDS` (documents), not shared with it, since face framing/
 * lighting tolerances are genuinely different from document scanning (a
 * face fills a rounder, more central region of frame; selfie lighting is
 * typically front-lit and more forgiving of a wider brightness band).
 */
// Original values below were untested guesses (never run against a real
// face/selfie), and two of them turned out to be actively broken: the tilt
// check had a sign bug that made every level face measure as ~180° (see
// `eyeLineTiltDeg`'s doc comment), and thresholds here were generally tuned
// tighter than a fast, forgiving production KYC selfie flow (Onfido/Jumio/
// Persona-style) actually needs — those prioritise capturing a recognisable
// face quickly over a cosmetically "perfect" shot, since the downstream
// face-match step is itself tolerant of minor blur/tilt/off-centering.
// Values below are widened accordingly, and grounded in a real measurement
// where noted (see `minBlurScore`).
export const FACE_THRESHOLDS = {
  /** Minimum detector confidence to treat a frame as "face detected". */
  minDetectionConfidence: 0.6,

  /**
   * Laplacian variance below this is treated as blurry. Measured directly
   * against a real, studio-quality reference face photo at graduated blur
   * levels (same grid-max methodology used to calibrate the document/face
   * sharpness metrics elsewhere in this app): sharp ≈ 130-234, noticeably
   * soft ≈ 38, clearly blurred ≈ 3-8. A real phone selfie (lower-res front
   * camera, imperfect lighting) will score below this idealised reference
   * even in focus, so 15 leaves real margin below "sharp" while still
   * rejecting genuinely unusable frames.
   */
  minBlurScore: 15,

  /** Acceptable mean-luminance band (0-255) — selfies are usually front-lit, so a bit wider than documents. */
  minBrightness: 40,
  maxBrightness: 240,

  /** Fraction of the frame blown out by glare before we warn. */
  maxGlareRatio: 0.15,

  /** Frame-to-frame motion magnitude (0-1) below which the face is "still" — widened for natural handheld jitter. */
  maxMotionScore: 0.08,

  /** faceBoxArea / frameArea band for "correct distance" — a face fills noticeably less of the frame than a document does. */
  minFaceSizeRatio: 0.06,
  maxFaceSizeRatio: 0.65,

  /** Normalised (0-1) distance of the face centroid from frame centre before we ask to re-centre. */
  maxCenterOffsetRatio: 0.3,

  /**
   * Max head tilt (roll, from the eye-to-eye line), in degrees, before we
   * ask the user to level their head. 12° was unrealistically strict for
   * natural head position during a selfie; 25° matches (slightly exceeds)
   * the tolerance the old native-OS-detector flow used successfully.
   */
  maxEyeTiltDeg: 25,

  /** Consecutive-valid duration required before auto-capture fires — shortened for a snappier, more production-app-like capture. */
  autoCaptureStableMs: 600,

  /** How often (ms) the JS thread is updated from the worklet frame processor. */
  guidanceThrottleMs: 66,
} as const;

export const FACE_MODEL_INPUT_SIZE = {
  /**
   * YuNet's square input — confirmed from the real face_detection_yunet_2023mar.onnx
   * graph (static [1,3,640,640] input; NOT 320x320, a common demo default that doesn't
   * match this exact export). Mirrored as `FACE_MODEL_INPUT_SIZE` in
   * android/.../HybridDocScanner.kt and `kFaceDetectorInputSize` in
   * ios/ONNXInference.swift — keep in sync if this ever changes.
   */
  detectorWidth: 640,
  detectorHeight: 640,
} as const;
