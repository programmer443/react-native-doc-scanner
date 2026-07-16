import { GuidanceCode } from '../types/guidance';

export const GUIDANCE_MESSAGES: Record<GuidanceCode, string> = {
  [GuidanceCode.NO_DOCUMENT]: 'Document not detected',
  [GuidanceCode.WRONG_DOCUMENT_TYPE]: "This doesn't look like the right document type",
  [GuidanceCode.MOVE_CLOSER]: 'Move closer',
  [GuidanceCode.MOVE_FARTHER]: 'Move farther',
  [GuidanceCode.CENTER_DOCUMENT]: 'Center document in frame',
  [GuidanceCode.ALIGN_DOCUMENT]: 'Align document',
  [GuidanceCode.TILT_PHONE]: 'Tilt phone slightly',
  [GuidanceCode.HOLD_STILL]: 'Hold still',
  [GuidanceCode.BLURRY]: 'Image is blurry',
  [GuidanceCode.LOW_LIGHT]: 'Increase lighting',
  [GuidanceCode.OVEREXPOSED]: 'Reduce lighting',
  [GuidanceCode.REDUCE_GLARE]: 'Reduce glare',
  [GuidanceCode.PARTIALLY_OUT_OF_FRAME]: 'Document partially outside frame',
  [GuidanceCode.READY]: 'Perfect — hold still',
  [GuidanceCode.CAPTURING]: 'Capturing...',
  [GuidanceCode.SCANNING]: 'Scanning...',
  [GuidanceCode.EXTRACTING_TEXT]: 'Extracting text...',
  [GuidanceCode.COMPLETED]: 'Completed',
};
