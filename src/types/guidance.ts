/** Every real-time instruction the scanner can surface. UI copy lives in constants/messages.ts. */
export enum GuidanceCode {
  NO_DOCUMENT = 'NO_DOCUMENT',
  WRONG_DOCUMENT_TYPE = 'WRONG_DOCUMENT_TYPE',
  MOVE_CLOSER = 'MOVE_CLOSER',
  MOVE_FARTHER = 'MOVE_FARTHER',
  CENTER_DOCUMENT = 'CENTER_DOCUMENT',
  ALIGN_DOCUMENT = 'ALIGN_DOCUMENT',
  TILT_PHONE = 'TILT_PHONE',
  HOLD_STILL = 'HOLD_STILL',
  BLURRY = 'BLURRY',
  LOW_LIGHT = 'LOW_LIGHT',
  OVEREXPOSED = 'OVEREXPOSED',
  REDUCE_GLARE = 'REDUCE_GLARE',
  PARTIALLY_OUT_OF_FRAME = 'PARTIALLY_OUT_OF_FRAME',
  READY = 'READY',
  CAPTURING = 'CAPTURING',
  SCANNING = 'SCANNING',
  EXTRACTING_TEXT = 'EXTRACTING_TEXT',
  COMPLETED = 'COMPLETED',
}

export interface QualityFlags {
  hasDocument: boolean;
  /** False only when the classifier confidently identified a document type that doesn't match `expectedDocumentType`. Always true if no classifier is loaded (documentType stays GENERIC) or no `expectedDocumentType` was given. */
  isCorrectDocumentType: boolean;
  isCentered: boolean;
  isSharp: boolean;
  hasNoGlare: boolean;
  hasGoodBrightness: boolean;
  isStill: boolean;
  isCorrectSize: boolean;
  isCorrectPerspective: boolean;
  isFullyInFrame: boolean;
}

/** Result of running GuidanceEngine over one FrameAnalysis. */
export interface GuidanceState {
  code: GuidanceCode;
  message: string;
  /** True only when every QualityFlags entry passes — the auto-capture gate. */
  isValid: boolean;
  flags: QualityFlags;
  confidence: number;
}

export type CaptureStage = 'idle' | 'stabilizing' | 'capturing' | 'processing' | 'completed' | 'failed';

export interface AutoCaptureState {
  stage: CaptureStage;
  /** 0-1 progress through the post-stability capture delay, for the UI ring. */
  stabilityProgress: number;
}
