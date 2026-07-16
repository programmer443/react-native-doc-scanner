/** Every real-time instruction the selfie guide can surface. UI copy lives in constants/faceMessages.ts. */
export enum FaceGuidanceCode {
  NO_FACE = 'NO_FACE',
  MOVE_CLOSER = 'MOVE_CLOSER',
  MOVE_FARTHER = 'MOVE_FARTHER',
  CENTER_FACE = 'CENTER_FACE',
  LEVEL_HEAD = 'LEVEL_HEAD',
  HOLD_STILL = 'HOLD_STILL',
  BLURRY = 'BLURRY',
  LOW_LIGHT = 'LOW_LIGHT',
  OVEREXPOSED = 'OVEREXPOSED',
  REDUCE_GLARE = 'REDUCE_GLARE',
  READY = 'READY',
  CAPTURING = 'CAPTURING',
  EXTRACTING = 'EXTRACTING',
  COMPLETED = 'COMPLETED',
}

export interface FaceQualityFlags {
  hasFace: boolean;
  isCentered: boolean;
  isCorrectSize: boolean;
  isLevel: boolean;
  isSharp: boolean;
  hasNoGlare: boolean;
  hasGoodBrightness: boolean;
  isStill: boolean;
}

/** Result of running FaceGuidanceEngine over one FaceFrameAnalysis. */
export interface FaceGuidanceState {
  code: FaceGuidanceCode;
  message: string;
  /** True only when every FaceQualityFlags entry passes — the auto-capture gate. */
  isValid: boolean;
  flags: FaceQualityFlags;
  confidence: number;
}
