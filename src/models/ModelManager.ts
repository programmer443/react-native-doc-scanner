import { createMMKV } from 'react-native-mmkv';
import { NativeDocScannerModule } from '../services/NativeDocScannerModule';
import { DEFAULT_MODEL, getModel, registerModel, type ModelConfig } from './modelRegistry';

const ACTIVE_MODEL_KEY = 'react-native-doc-scanner.activeModelId';

const storage = createMMKV({ id: 'react-native-doc-scanner' });

let loadedModelId: string | null = null;
let loadInFlight: Promise<ModelConfig> | null = null;

/**
 * Owns which ONNX model set is active. This is the single place an app swaps
 * in its own fine-tuned detector or a newer RapidOCR export — everything
 * downstream (frame processor, capture pipeline, UI) reads results generically
 * and never assumes a specific model file, satisfying "replace the model
 * without changing the UI".
 */
export const ModelManager = {
  /** Registers a custom model set (e.g. one your app downloaded or fine-tuned) so it can be activated later. */
  register(config: ModelConfig): void {
    registerModel(config);
  },

  getActiveModelId(): string {
    return storage.getString(ACTIVE_MODEL_KEY) ?? DEFAULT_MODEL.id;
  },

  /**
   * Loads a model set into the native detector/OCR engines and remembers the
   * choice across app restarts. Safe to call multiple times concurrently —
   * concurrent callers await the same in-flight load.
   */
  async activate(modelId: string = ModelManager.getActiveModelId()): Promise<ModelConfig> {
    const config = getModel(modelId);
    if (!config) {
      throw new Error(`react-native-doc-scanner: no model registered with id "${modelId}".`);
    }

    if (loadedModelId === modelId) {
      return config;
    }

    if (loadInFlight) {
      await loadInFlight;
      if (loadedModelId === modelId) return config;
    }

    loadInFlight = NativeDocScannerModule.loadModels({
      detectorModelPath: config.detectorModelPath,
      ocrDetModelPath: config.ocrDetModelPath,
      ocrClsModelPath: config.ocrClsModelPath,
      ocrRecModelPath: config.ocrRecModelPath,
      ocrRecCharsetPath: config.ocrRecCharsetPath,
      faceDetectorModelPath: config.faceDetectorModelPath,
      classifierModelPath: config.classifierModelPath,
    }).then((result) => {
      if (!result.success) {
        throw new Error('react-native-doc-scanner: native model load reported failure.');
      }
      loadedModelId = modelId;
      storage.set(ACTIVE_MODEL_KEY, modelId);
      return config;
    });

    try {
      await loadInFlight;
    } finally {
      loadInFlight = null;
    }

    return config;
  },

  isLoaded(modelId: string = ModelManager.getActiveModelId()): boolean {
    return loadedModelId === modelId;
  },
};
