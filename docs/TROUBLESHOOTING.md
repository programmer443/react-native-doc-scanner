# Troubleshooting

## `NitroModules.createHybridObject<DocScanner>('DocScanner')` throws / returns null

The JS side (`vision/docScannerNative.ts`) throws a module-not-found error
when the native `DocScanner` HybridObject never registered.

- Run `pnpm --filter react-native-doc-scanner run specs` (`tsc && nitrogen`)
  at least once — this generates `nitrogen/generated/`, which both the
  podspec (`load "nitrogen/generated/ios/DocScanner+autolinking.rb"`) and
  `android/build.gradle` (`apply from: "../nitrogen/generated/android/DocScanner+autolinking.gradle"`)
  depend on. If that directory doesn't exist yet, the native build won't even
  compile the registration glue.
- Rebuild the app after installing/updating the package — HybridObjects
  register at native module init time, Metro fast refresh isn't enough.
- iOS: confirm `pod install` picked up `react-native-doc-scanner.podspec`
  (check `Pods/Pods.xcodeproj` for a `react-native-doc-scanner` target, and
  that `nitrogen/generated/ios/` contains a `DocScanner+autolinking.rb`).
- Android: confirm `nitrogen/generated/android/DocScanner+autolinking.gradle`
  and `.cmake` exist, and that the CMake build actually ran (check
  `android/build/generated/source/nitrogen` or the Gradle sync log for
  `DocScanner` CMake targets).

## Nitro cross-module reference to `Frame` (from `react-native-vision-camera`) fails to compile

`src/specs/DocScanner.nitro.ts` imports `Frame` from `react-native-vision-camera` —
a HybridObject type owned by a different package. This requires
`react-native-vision-camera`'s own `nitrogen/generated` C++ headers to be
visible at compile time:

- Android: `android/build.gradle` depends on
  `implementation project(":react-native-vision-camera")` for exactly this —
  if Gradle can't resolve that project, check the host app's
  `settings.gradle` autolinking actually includes `react-native-vision-camera`
  (it should, automatically, since it's already a dependency of the app).
- iOS: `add_nitrogen_files(s)` in the podspec should pull in the right search
  paths automatically; if you see "no member named 'Frame' in namespace" or
  similar, verify `react-native-vision-camera` itself built successfully
  first (its own `nitrogen/generated` must exist too).

## iOS: `OpenCV2` pod fails to build, or `pod install` can't find a compatible version

Note the pod is named **`OpenCV2`**, not `OpenCV` — the pod literally named
"OpenCV" on the CocoaPods trunk only ever published OpenCV 2.4.x (last in
2018) and has no version satisfying a `~> 4.x` constraint at all (confirmed
via `pod trunk info OpenCV`); `OpenCV2` is the one that actually ships OpenCV
4.x, up to 4.3.0 (April 2020) — still community-maintained and not updated
since, so it can fail on newer Xcode toolchains (module map / bitcode-related
errors), **and confirmed to have no `arm64-simulator` slice** — only
`ios-arm64` (device) and `ios-x86_64-simulator` (Intel Mac simulator). On an
Apple Silicon Mac this makes the **Simulator** build fail at link time
(`building for 'iOS-simulator', but linking in object file ... built for
'iOS'`) — the same category of issue as ClearQuoteSDK's xcframework
constraint. A **device** build is unaffected (the `ios-arm64` slice is
device-only and present). Since VisionCamera needs a real camera anyway,
this generally doesn't block real testing — just build for a physical
device instead of the simulator. If you hit that:

1. Build/run on a physical device instead of the Simulator (`npx react-native
   run-ios --device`) — sidesteps the missing slice entirely, **or**
2. For Apple-Silicon Simulator specifically: pin the Simulator build to
   `x86_64` (Rosetta) rather than `arm64`, **or**
3. Try pinning an older, known-good Xcode command-line toolchain for just the
   pod build (`xcode-select` a secondary Xcode install), **or**
4. Vendor `opencv2.framework`/`OpenCV.xcframework` directly: download the
   official prebuilt iOS framework from https://opencv.org/releases/, drop it
   under `ios/Frameworks/`, remove the `OpenCV2` pod dependency from
   `react-native-doc-scanner.podspec`, and add
   `s.vendored_frameworks = "ios/Frameworks/opencv2.xcframework"` instead.
   The OpenCV glue code doesn't need to change — it only depends on the
   `<opencv2/...>` C++ headers, not on how the binary was linked in.

## Android: `org.opencv:opencv` / ONNX Runtime AAR not resolving

Both are on Maven Central, so this is almost always a missing
`mavenCentral()` in the **host app's** `android/build.gradle`
`allprojects { repositories { ... } }` block (not just this package's own
`build.gradle`, which pnpm/gradle won't apply to the root project
automatically).

## `useFrameOutput` throws "react-native-vision-camera-worklets is not installed"

Install `react-native-vision-camera-worklets` — VisionCamera Core doesn't
bundle a worklets runtime itself; `useFrameOutput`'s `onFrame` worklet
requires this companion package (which itself depends on
`react-native-worklets`, the same engine Reanimated 4 runs on).

## Frame analysis runs but the camera feels sluggish / dropped frames

- Confirm `pixelFormat: 'yuv'` on `useFrameOutput` — native code converts
  from YUV itself; requesting `'rgb'` adds a redundant conversion pass before
  your native code even runs.
- The detector + OpenCV quality analysis run on **every** frame by design
  (motion detection needs consecutive frames) — if a low-end Android device
  can't keep up, the usual fix is downscaling the frame before inference
  (resize to `MODEL_INPUT_SIZE` in native code, which the bundled
  implementation already does for the detector — if you're seeing slowness,
  check the OpenCV quality-metric pass isn't operating on the full-resolution
  buffer instead of a downscaled copy).
- Watch for `onFrameDropped` warnings (`'out-of-buffers'` means `analyzeFrame`
  is taking longer than one frame interval — profile the native code, not
  the JS side).
- `guidanceThrottleMs` in `constants/thresholds.ts` controls how often
  results cross to JS/React — this does **not** affect native frame
  throughput, only UI update frequency. Lowering it further won't help a
  native-side perf problem and will just cost more JS-thread work.
- Make sure `frame.dispose()` is called every time in `onFrame` (see
  `hooks/useDocumentScanner.ts`) — a held-onto `Frame` stalls the camera
  pipeline exactly like the classic VisionCamera API.

## Bounding box / quad drawn in the wrong place or mirrored

The native `analyzeFrame` implementation is contractually required to return
`quad`/`frameWidth`/`frameHeight` already normalised to the preview's display
orientation (see the comment in `vision/nativeFrameResult.ts`). If corners
look rotated or mirrored:

- Confirm the native code accounts for `frame.orientation` (iOS) /
  `image.imageInfo.rotationDegrees` (Android, via CameraX) before returning
  coordinates — this normalisation must happen natively, not in JS.
- Front camera: mirror horizontally (`x' = frameWidth - x`) before returning,
  to match what `isMirrored` shows the user in the preview.

## MRZ reads inconsistently (passport/visa)

This is expected on some captures — the recognizer isn't MRZ-font-specialised
(see docs/MODEL_TRAINING.md §1). `mrzParser.ts`'s check-digit autocorrection
recovers most single-character OCR mistakes, but a persistently unreadable
MRZ usually means the bottom of the passport page isn't fully in frame, or
there's glare across the MRZ band specifically — both are things
`GuidanceEngine` already warns about (`PARTIALLY_OUT_OF_FRAME`,
`REDUCE_GLARE`), so check those flags were actually green at capture time.

## Worklet / Reanimated version conflicts

This package's `onFrame` worklet (`hooks/useDocumentScanner.ts`) calls plain
functions marked `'worklet'` from other files (`GuidanceEngine`,
`mapNativeFrameResult`, `geometry.ts`), and uses `runOnJS`/`useSharedValue`
from `react-native-reanimated`. That only works if your app's Babel config
runs the worklets/reanimated plugin over your **entire** source tree (the
default), not just files that directly call a camera hook — double-check
`babel.config.js` doesn't scope the plugin to a subdirectory, and that you
don't have both `react-native-worklets-core` (the old package, unrelated to
VisionCamera Core) and `react-native-worklets` fighting over the same
worklet registration.
