import { NitroModules } from 'react-native-nitro-modules';
import type { DocScanner } from '../specs/DocScanner.nitro';

/**
 * Singleton native engine — one HybridObject instance backs both the
 * per-frame `analyzeFrame` worklet call and the async model-loading/capture
 * calls, matching the `nitro.json` "DocScanner" autolinking entry.
 */
export const DocScannerNative = NitroModules.createHybridObject<DocScanner>('DocScanner');
