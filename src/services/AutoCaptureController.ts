import { SCANNER_THRESHOLDS } from '../constants/thresholds';
import type { CaptureStage } from '../types/guidance';

export interface AutoCaptureCallbacks {
  onStageChange: (stage: CaptureStage) => void;
  /** 0-1 progress through the stability window, for the capture-progress ring. */
  onProgress: (progress: number) => void;
  /** Fired exactly once when the stability window completes — take the photo here. */
  onCapture: () => void;
}

/**
 * Auto-capture gate: fires `onCapture` once all guidance flags have been
 * simultaneously true for `stableDurationMs` (~1s per spec). Runs on the JS
 * thread, fed by throttled guidance updates from the frame processor — no
 * timers of its own, so it can't fire early on a burst of stale updates.
 */
export class AutoCaptureController {
  private stableSinceMs: number | null = null;
  private stage: CaptureStage = 'idle';
  // Set on a failed capture; cleared the next time guidance goes invalid.
  // While set, a still-valid frame is NOT allowed to re-arm the stability
  // timer — without this, a document sitting motionless in a persistently-
  // failing frame (bad angle, glare, unreadable text, etc.) retriggers a
  // real photo capture — and its native OS shutter sound — every
  // `stableDurationMs`, in a tight loop. Requiring an invalid→valid
  // transition first means the real camera shutter only ever fires again
  // once the situation has actually changed (the user moved/adjusted the
  // document), not on every stability-window tick.
  private waitingForInvalidBeforeRetry = false;

  constructor(
    private readonly callbacks: AutoCaptureCallbacks,
    private readonly stableDurationMs: number = SCANNER_THRESHOLDS.autoCaptureStableMs,
  ) {}

  /** Feed the latest guidance validity in. Call on every (throttled) guidance update. */
  update(isValid: boolean, now: number = Date.now()): void {
    if (this.stage === 'capturing' || this.stage === 'processing' || this.stage === 'completed') {
      return;
    }

    if (!isValid) {
      this.waitingForInvalidBeforeRetry = false;
      this.stableSinceMs = null;
      this.callbacks.onProgress(0);
      this.setStage('idle');
      return;
    }

    if (this.waitingForInvalidBeforeRetry) {
      return;
    }

    if (this.stableSinceMs === null) {
      this.stableSinceMs = now;
      this.setStage('stabilizing');
    }

    const elapsed = now - this.stableSinceMs;
    const progress = Math.min(1, elapsed / this.stableDurationMs);
    this.callbacks.onProgress(progress);

    if (progress >= 1) {
      this.setStage('capturing');
      this.callbacks.onCapture();
    }
  }

  markProcessing(): void {
    this.setStage('processing');
  }

  markCompleted(): void {
    this.setStage('completed');
  }

  /**
   * A capture was attempted (the native shutter has already fired) but
   * extraction failed. Goes to a resting `failed` stage rather than
   * straight back to `idle` — see `waitingForInvalidBeforeRetry` above —
   * so the same still-valid frame can't immediately retrigger another real
   * capture.
   */
  markFailed(): void {
    this.stableSinceMs = null;
    this.callbacks.onProgress(0);
    this.waitingForInvalidBeforeRetry = true;
    this.setStage('failed');
  }

  reset(): void {
    this.stableSinceMs = null;
    this.waitingForInvalidBeforeRetry = false;
    this.callbacks.onProgress(0);
    this.setStage('idle');
  }

  getStage(): CaptureStage {
    return this.stage;
  }

  private setStage(stage: CaptureStage): void {
    if (this.stage === stage) return;
    this.stage = stage;
    this.callbacks.onStageChange(stage);
  }
}
