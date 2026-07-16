# react-native-doc-scanner

Real-time AI document scanner for React Native — built on **VisionCamera
Core** (the Nitro-Modules-based rewrite of react-native-vision-camera, v5+)
with ONNX Runtime document detection, OpenCV quality analysis, and PaddleOCR
(via RapidOCR's ONNX export) field extraction for passports, driving
licences, national ID cards, residence permits, and visas.

> **Requires VisionCamera Core (react-native-vision-camera ^5.0.0+ / "Nitro"
> architecture)**, not the classic v3/v4 frame-processor-plugin API. If your
> app is still on VisionCamera v4, either upgrade it or don't add this
> package yet — there's no shim between the two architectures.

Guides the user live ("Move closer", "Hold still", "Reduce glare", ...),
auto-captures once every quality gate passes for ~1s, applies perspective
correction, and returns structured JSON:

```json
{
  "documentType": "PASSPORT",
  "name": "JANE ANN DOE",
  "documentNumber": "123456789",
  "dob": "1990-05-12",
  "expiry": "2030-05-11",
  "nationality": "GBR",
  "address": "",
  "mrz": "P<GBRDOE<<JANE<ANN<<<<<<<<<<<<<<<<<<<<<<<<<\n1234567897GBR9005128F3005116<<<<<<<<<<<<<<02",
  "confidence": 0.94
}
```

## Why this stack

- **Detection**: [DocAligner](https://github.com/DocsaidLab/DocAligner)
  (Apache-2.0) — corner/quad regression, exported to ONNX. *Not* Ultralytics
  YOLO11: its pretrained weights are AGPL-3.0, which is a real licensing
  question for a proprietary app (see docs/MODEL_TRAINING.md). Swappable —
  bring your own fine-tuned or YOLO-based model without touching the UI.
- **OCR**: [RapidOCR](https://github.com/RapidAI/RapidOCR) (Apache-2.0) — the
  actual PaddleOCR PP-OCR nets, exported to ONNX, run via the platform-native
  ONNX Runtime SDKs.
- **Quality analysis**: OpenCV (Laplacian variance for blur, histogram for
  brightness/glare, frame-diff for motion, corner geometry for perspective) —
  runs natively, every frame.
- **Camera integration**: this package's native engine is a `DocScanner`
  **Nitro HybridObject** (`src/specs/DocScanner.nitro.ts`), not a classic
  bridge/TurboModule. Real-time analysis is called directly and synchronously
  from a `useFrameOutput` worklet (`DocScannerNative.analyzeFrame(frame)`) —
  zero-copy JSI, no bridge serialization, no per-frame async round trip.
  Model loading and post-capture OCR are async HybridObject methods on the
  same object.
- **Why not the `onnxruntime-react-native` npm package**: its New
  Architecture support is unverified, and per-frame inference has to run
  natively regardless (a JS-bridge round trip per frame isn't fast enough).
  Instead, both the real-time detector and the post-capture OCR pipeline call
  the official `onnxruntime-objc` / `onnxruntime-android` SDKs directly from
  this package's own Nitro module.

## Installation

```sh
# From your app (if consuming as a pnpm workspace package, this is already linked):
pnpm add react-native-doc-scanner
pnpm add react-native-vision-camera react-native-vision-camera-worklets \
  react-native-nitro-modules react-native-reanimated react-native-worklets \
  react-native-mmkv react-native-gesture-handler @shopify/react-native-skia zustand

cd ios && pod install
```

`react-native-vision-camera-worklets` is required for `useFrameOutput`'s
per-frame worklet callback (VisionCamera Core doesn't bundle a worklets
runtime itself — see docs/TROUBLESHOOTING.md).

### iOS configuration

Add camera usage description to `ios/<App>/Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>We use the camera to scan your documents.</string>
```

Ensure your Podfile has (most VisionCamera + New Architecture apps already do):

```ruby
use_frameworks! :linkage => :static
$RNFirebaseAsStaticFramework = true # only if you also use Firebase

# HybridDocScanner.swift does `import onnxruntime_objc` directly — under
# static linkage this pod needs an explicit module map, same requirement as
# Firebase/Google pods commonly already have in RN apps.
pod 'onnxruntime-objc', :modular_headers => true
```

### Android configuration

`android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
```

`android/build.gradle` (root) — make sure both are present under `allprojects.repositories`:

```groovy
allprojects {
  repositories {
    google()
    mavenCentral()
  }
}
```

Minimum `minSdkVersion 24` (ONNX Runtime Android's floor).

### Babel

Your app's Reanimated/worklets Babel plugin must already be configured (it's
what turns `useFrameOutput`'s `onFrame` callback, and this package's
worklet-marked helper functions, into worklets) — if `react-native-reanimated`
already works elsewhere in your app, this is already satisfied. Don't add a
second/conflicting worklets plugin.

## Fetching the bundled models

The default model weights aren't committed to the repo (they're binary and
several MB each) — fetch them once per docs/MODEL_TRAINING.md §1:

```sh
cd packages/react-native-doc-scanner
./scripts/fetch-models.sh   # see docs/MODEL_TRAINING.md if you'd rather do it by hand
```

## Usage

### Drop-in screen

```tsx
import { DocumentScannerScreen, DocumentType } from 'react-native-doc-scanner';

function ScanPassport({ navigation }: Props) {
  return (
    <DocumentScannerScreen
      documentType={DocumentType.PASSPORT}
      onCaptured={(result) => {
        console.log(result.data); // StructuredDocumentData
        navigation.navigate('PassportDetails', { data: result.data });
      }}
      onCancel={() => navigation.goBack()}
    />
  );
}
```

### Composing your own screen

```tsx
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import {
  useDocumentScanner,
  ScannerOverlay,
  GuidanceBanner,
  QualityIndicatorsRow,
  DocumentType,
} from 'react-native-doc-scanner';

function CustomScanScreen() {
  const device = useCameraDevice('back');
  const { frameOutput, photoOutput, guidance, quad, frameSize, captureStage } = useDocumentScanner({
    documentType: DocumentType.DRIVING_LICENCE,
    onCaptured: (result) => console.log(result.data),
  });

  return (
    <>
      <Camera style={{ flex: 1 }} device={device!} isActive outputs={[frameOutput, photoOutput]} />
      <ScannerOverlay width={390} height={844} quad={quad} frameWidth={frameSize.width}
        frameHeight={frameSize.height} isValid={guidance.isValid} captureStage={captureStage} />
      <GuidanceBanner code={guidance.code} message={guidance.message} isValid={guidance.isValid} />
    </>
  );
}
```

### Replacing the model later

```ts
import { ModelManager } from 'react-native-doc-scanner';

ModelManager.register({ id: 'custom-v1', label: 'Custom', license: 'Apache-2.0', ...paths });
await ModelManager.activate('custom-v1');
```

No UI or hook code changes — see docs/MODEL_TRAINING.md.

## Performance tuning

- `constants/thresholds.ts` — every quality gate and the `guidanceThrottleMs`
  UI update rate are tunable in one place.
- Detection + quality analysis run on every frame natively (required for
  accurate motion detection); only the JS/React update rate is throttled.
- `useFrameOutput({ pixelFormat: 'yuv', ... })` avoids a redundant color-space
  conversion before native code runs — don't switch to `'rgb'` unless you
  have a reason to.

## Architecture

```
src/
  types/        Detection, guidance, and OCR result shapes
  constants/    Tunable thresholds, guidance copy, theme
  models/       ModelManager + swappable model registry
  specs/        DocScanner.nitro.ts — the Nitro HybridObject spec (source of truth for the native contract)
  vision/       HybridObject instantiation, native<->JS frame result mapping
  services/     GuidanceEngine, AutoCaptureController, native async bridge wrapper
  ocr/          Capture-time OCR orchestration + MRZ/licence/ID field parsers
  opencv/       JS-side interpretation of native OpenCV quality metrics
  store/        zustand + MMKV scanner state
  hooks/        useDocumentScanner, useHapticFeedback, useScannerPermissions
  components/   ScannerOverlay (Skia), GuidanceBanner, indicators, animations
  screens/      Ready-to-use DocumentScannerScreen
  utils/        Pure geometry helpers (worklet-safe)
ios/            HybridDocScanner.swift + ONNX Runtime/OpenCV glue
android/        HybridDocScanner.kt + ONNX Runtime/OpenCV glue
nitro.json      Nitro codegen config (autolinking, module names)
docs/           Model training/swapping guide, troubleshooting
```

Run `pnpm --filter react-native-doc-scanner specs` (`tsc && nitrogen`) after
editing `src/specs/DocScanner.nitro.ts` to regenerate the platform bridging
code under `nitrogen/generated/`.

See docs/TROUBLESHOOTING.md for build/runtime issues and docs/MODEL_TRAINING.md
for model sourcing, fine-tuning, and licensing details.
