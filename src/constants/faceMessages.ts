import { FaceGuidanceCode } from '../types/faceGuidance';

export const FACE_GUIDANCE_MESSAGES: Record<FaceGuidanceCode, string> = {
  [FaceGuidanceCode.NO_FACE]: 'Position your face in the frame',
  [FaceGuidanceCode.MOVE_CLOSER]: 'Move closer',
  [FaceGuidanceCode.MOVE_FARTHER]: 'Move farther',
  [FaceGuidanceCode.CENTER_FACE]: 'Center your face in the frame',
  [FaceGuidanceCode.LEVEL_HEAD]: 'Hold your head level',
  [FaceGuidanceCode.HOLD_STILL]: 'Hold still',
  [FaceGuidanceCode.BLURRY]: 'Image is blurry',
  [FaceGuidanceCode.LOW_LIGHT]: 'Increase lighting',
  [FaceGuidanceCode.OVEREXPOSED]: 'Reduce lighting',
  [FaceGuidanceCode.REDUCE_GLARE]: 'Reduce glare',
  [FaceGuidanceCode.READY]: 'Perfect — hold still',
  [FaceGuidanceCode.CAPTURING]: 'Capturing...',
  [FaceGuidanceCode.EXTRACTING]: 'Processing...',
  [FaceGuidanceCode.COMPLETED]: 'Completed',
};
