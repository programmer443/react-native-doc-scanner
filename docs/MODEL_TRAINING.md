# Model training & swapping guide

This package ships a default detector (DocAligner) and OCR pipeline (RapidOCR),
but every model is swappable at runtime through `ModelManager` — nothing in the
UI, frame processor contract, or OCR parsers assumes a specific model file.
This doc covers where the bundled weights come from, how to fine-tune your
own detector, and how to plug a replacement in.

## 1. Bundled default: DocAligner + RapidOCR

| Model | Role | License | Source |
|---|---|---|---|
| DocAligner (`heatmap_reg`, FastViT-SA24 + BiFPN backbone, 256×256 input) | Document corner/quad detection | Apache-2.0 | https://github.com/DocsaidLab/DocAligner |
| RapidOCR `ch_PP-OCRv4_det` | Text-region detection (script-agnostic) | Apache-2.0 | https://huggingface.co/SWHL/RapidOCR |
| RapidOCR `ch_ppocr_mobile_v2.0_cls` | Text-orientation classification | Apache-2.0 | https://huggingface.co/SWHL/RapidOCR |
| RapidOCR `en_PP-OCRv3_rec` | Text recognition (Latin script) | Apache-2.0 | https://huggingface.co/SWHL/RapidOCR |

**Why not Ultralytics YOLO11 for detection?** Ultralytics' pretrained weights
are AGPL-3.0. That's a real constraint for a proprietary app: distributing an
AGPL-licensed model inside a closed-source binary generally obligates you to
offer recipients the corresponding source (or buy Ultralytics' Enterprise
license). DocAligner's Apache-2.0 license has no such obligation. If your org
has already cleared AGPL or holds an Enterprise license, see §3 below for the
YOLO11 path — the architecture supports it identically.

**DocAligner weight availability**: upstream serves ONNX weights from
hardcoded Google Drive file IDs inside the Python package, not a stable CDN
URL. Don't point production builds at Google Drive directly. Instead:

```bash
pip install docaligner-docsaid  # real PyPI name is "docaligner-docsaid", not "docaligner-onnx"
python - <<'PY'
from docaligner import DocAligner
# Triggers the one-time download (from the hardcoded Google Drive IDs) to
# ~/.cache/docaligner
DocAligner()
PY
# Copy the resulting .onnx files into this package's assets:
find ~/.cache/docaligner -name "*.onnx" -exec cp {} packages/react-native-doc-scanner/assets/models/ \;
```

`DocAligner()`'s default constructor downloads the `heatmap_reg` family —
currently `fastvit_sa24_h_e_bifpn_256_fp32.onnx` (FastViT-SA24 + BiFPN,
256×256 input, outputs a `[N,4,H,W]` per-corner heatmap — the shape both
native detectors already handle). Rename the copied file to
`docaligner_fastvit_heatmap.onnx` to match `modelRegistry.ts`'s
`DEFAULT_MODEL`, or update that path if you keep a different filename or use
the `point_reg` family instead (flat 8-float direct corner regression,
handled by the same native code's other branch).

**RapidOCR weights**:

```bash
mkdir -p packages/react-native-doc-scanner/assets/models
# General-purpose detector — text detection is script-agnostic, so RapidOCR
# pairs this same PP-OCRv4 detector with English recognition too.
curl -L -o assets/models/ch_PP-OCRv4_det_infer.onnx \
  https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx
# 2-class orientation classifier — only ever published under PP-OCRv1.
curl -L -o assets/models/ch_ppocr_mobile_v2.0_cls_infer.onnx \
  https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv1/ch_ppocr_mobile_v2.0_cls_infer.onnx
curl -L -o assets/models/en_PP-OCRv3_rec_infer.onnx \
  https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv3/en_PP-OCRv3_rec_infer.onnx
# Verified byte-identical to PaddleOCR's own release/2.6 en_dict.txt (the
# release line PP-OCRv3 shipped under).
curl -L -o assets/models/en_dict.txt \
  https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.1/paddle/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile/en_dict.txt
```

Or just run `scripts/fetch-models.sh`, which does exactly the above. Verify
the file browser paths on `huggingface.co/SWHL/RapidOCR` before running these
if you hit a 404 — HF repo layouts occasionally get reorganised between
releases (this repo has no `det/`, `rec/`, or `cls/` subdirectories — files
sit flat inside each `PP-OCRv*/` folder).

**Known limitation**: RapidOCR's recognizer is trained on general Latin text,
not the OCR-B monospace MRZ font specifically (see PaddleOCR issue #8852 for
community reports of it underperforming Tesseract on MRZ). That's why MRZ
extraction goes through `ocr/parsers/mrzParser.ts`'s check-digit
autocorrection rather than trusting raw OCR text — treat the OCR engine as
"good enough to get characters roughly right" and let the ICAO check digits
do the real validation.

## 2. Fine-tuning DocAligner on your own document mix

DocAligner's corner-regression approach generalises reasonably well out of
the box, but if your users scan documents with unusual borders, holograms, or
extreme glare, fine-tune on your own captures plus MIDV-500/2020:

```bash
git clone https://github.com/DocsaidLab/DocAligner
cd DocAligner
pip install -r requirements.txt
# Follow the repo's `train.py` instructions, pointing --data at a directory
# of (image, 4-corner-quad) annotation pairs. Mix in your own labelled
# captures alongside MIDV-500/2020 (see licensing note below) for
# device/lighting conditions your users actually hit.
python train.py --config configs/lcnet050.yaml --data /path/to/your/dataset
python export_onnx.py --checkpoint runs/best.ckpt --output docaligner_custom.onnx
```

### MIDV-500 / MIDV-2020 — dataset access and licensing

- **MIDV-500**: `ftp://smartengines.com/midv-500/` (or the `fcakyon/midv500`
  Python package, which wraps that download). Paper: arXiv:1807.05786.
- **MIDV-2020**: `l3i-share.univ-lr.fr/MIDV2020/midv2020.html` — gated behind
  a request form; you'll receive sFTP credentials by email.
- **Licensing — read before using commercially**: neither dataset grants an
  unambiguous commercial-use license. MIDV-500's paper only states the
  source images are "public domain or under public copyright licenses" (no
  CC0/CC-BY grant), and the MIDV-2020 portal explicitly asks you to seek
  permission before commercial use. Treat both as research/fine-tuning aids
  for internal model development, and get written permission (or use only
  your own captured documents) before shipping a model trained on them in a
  commercial product — this is a legal question for your team, not something
  to route around technically.

## 3. Alternative: YOLO11 detector (if AGPL/Enterprise is cleared)

If your organisation has cleared the AGPL-3.0 obligations (or holds an
Ultralytics Enterprise license), you can swap in a YOLO11 bounding-box
detector instead of DocAligner's corner regression:

```bash
pip install ultralytics
yolo export model=yolo11n.pt format=onnx imgsz=256
# Fine-tune on MIDV-500/2020 (converted to YOLO bounding-box format) or your
# own labelled captures:
yolo detect train model=yolo11n.pt data=midv500.yaml epochs=100 imgsz=256
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

A bounding-box model only gives you `boundingBox`, not `quad` — the native
detector code computes `quad` as the box's four corners in that case (no
perspective correction benefit, but centering/distance/size guidance all
still work identically).

## 4. Swapping in your model

Register and activate it from JS — no native rebuild needed if you're only
changing which files load, since the native side always loads whatever paths
`ModelManager` passes it:

```ts
import { ModelManager } from 'react-native-doc-scanner';

ModelManager.register({
  id: 'acme-v2',
  label: 'Acme fine-tuned detector v2',
  license: 'Apache-2.0',
  detectorModelPath: '/data/user/0/com.acme.app/files/models/detector_v2.onnx', // downloaded at runtime
  ocrDetModelPath: 'bundle://ch_PP-OCRv4_det_infer.onnx', // keep the bundled OCR nets
  ocrClsModelPath: 'bundle://ch_ppocr_mobile_v2.0_cls_infer.onnx',
  ocrRecModelPath: 'bundle://en_PP-OCRv3_rec_infer.onnx',
  ocrRecCharsetPath: 'bundle://en_dict.txt',
});

await ModelManager.activate('acme-v2');
```

Any path not prefixed `bundle://` is treated as an absolute filesystem path —
so an app can download a model update over the network, save it to
`RNFS.DocumentDirectoryPath`/app-internal storage, and activate it without an
app store release.

## 5. Face detection: YuNet (selfie capture guide)

| Model | Role | License | Source |
|---|---|---|---|
| YuNet (`face_detection_yunet_2023mar`, 320×320 input) | Real-time single-face box + 5-point landmarks for the selfie capture guide | MIT | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet |

`analyzeFaceFrame` (native `DocScanner` HybridObject — see
`android/.../HybridDocScanner.kt` and `ios/HybridDocScanner.swift`) runs this
model instead of the DocAligner document detector when the screen is guiding
a selfie rather than a document scan. It reports WIDER-Face AP of
0.834/0.824/0.708 (easy/medium/hard) — plenty for a guided, cooperative
front-camera capture, where the subject is close, mostly front-facing, and
prompted by on-screen feedback.

**Why not SCRFD?** SCRFD (from InsightFace) is the other commonly-recommended
lightweight ONNX face detector, but InsightFace's pretrained SCRFD weights are
gated behind a paid commercial license for production use — the same
proprietary-app constraint that ruled out Ultralytics YOLO11 for document
detection in §1. YuNet has no such restriction: it's MIT-licensed, ships
directly in OpenCV's own `opencv_zoo`, and is the detector backing OpenCV's
`cv::FaceDetectorYN` C++/Python API.

**Fetching the weights**:

```bash
mkdir -p packages/react-native-doc-scanner/assets/models
curl -L -o assets/models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

Or run `scripts/fetch-models.sh`, which fetches it as one additive step at
the end (alongside the RapidOCR weights) — safe to skip if your app only uses
the document-scanning flow.

**Wiring it in**: pass `faceDetectorModelPath: 'bundle://face_detection_yunet_2023mar.onnx'`
in the `ModelPaths` object given to `loadModels`/`ModelManager.register`.
This field is optional — leave it `''` (or omit fetching the file) and
`analyzeFaceFrame` will simply always report `detected: false` rather than
throwing; document scanning and OCR are entirely unaffected either way, since
loading this model is wrapped in its own try/catch on both platforms and
never fails the rest of `loadModels`.

## 6. Document-type classifier

`analyzeFrame`'s `documentType` field (PASSPORT/DRIVING_LICENCE/ID_CARD/
RESIDENCE_PERMIT/VISA vs GENERIC) is real, not a stub: both platforms run an
optional 6th ONNX model — same `classifierModelPath` / try-catch-never-fails
convention as the YuNet face detector above — and, once loaded, the
`GuidanceEngine` surfaces a `WRONG_DOCUMENT_TYPE` guidance code (and blocks
auto-capture) whenever it doesn't match the `documentType` the app asked
`useDocumentScanner` to scan for. With no classifier loaded, `documentType`
stays `GENERIC` and this check is skipped entirely — exactly today's
behavior.

**No bundled default weight ships with this package.** Unlike DocAligner/
RapidOCR/YuNet above, there's no off-the-shelf Apache/MIT-licensed ONNX model
with the right taxonomy to fetch:

- HF `prithivMLmods/Document-Type-Detection` — Apache-2.0, SigLIP2-base,
  224×224 — but classifies document *genre* (Advertisement/Hand-Written/
  Invoice/Letter/News-Article/Resume), not ID document type. Wrong classes
  for this use case.
- HF `logasanjeev/indian-id-validator` — card states MIT on the fine-tuned
  weights, but the model is built on Ultralytics YOLO11, the same AGPL-3.0
  licensing question §1 above already ruled out for the main detector. An
  MIT label on a fine-tune doesn't obviously clear the underlying Ultralytics
  AGPL obligation — get legal sign-off before using it, don't assume it's
  clean because the card says so. It's also Indian-documents-only (Aadhaar/
  PAN/Passport/Voter ID/Driving Licence), not the UK/EU/US/PK/BD coverage
  this app's `parseDrivingLicenceByCountry` already handles.

**Recommended path**: fine-tune a permissively-licensed ImageNet-pretrained
backbone — MobileNetV3-Small or EfficientNet-Lite0, both small enough for
real-time per-frame inference — as a 6-class classifier: PASSPORT,
DRIVING_LICENCE, ID_CARD, RESIDENCE_PERMIT, VISA, GENERIC (background/
not-a-recognized-document). Training data options, same commercial-use
diligence as the MIDV-500/2020 caveat in §2 — check each source's license
before shipping a model trained on it:

- Roboflow Universe per-document-type projects (driving-license, passport,
  ID-card, etc. — search `class:"driving license"` style queries) — useful
  as classification data even though most are published as object-detection
  datasets: treat each project's images as one class.
- Your own app-captured samples, which will always be the highest-signal
  data for your actual users' devices/lighting/document mix.

```bash
# Sketch — adapt to whatever training script you use:
# 1. Assemble a directory-per-class image set (PASSPORT/, DRIVING_LICENCE/, ...).
# 2. Fine-tune an ImageNet-pretrained MobileNetV3-Small/EfficientNet-Lite0
#    classification head on it.
# 3. Export to ONNX at the input size src/constants/thresholds.ts's
#    MODEL_INPUT_SIZE.classifierWidth/classifierHeight expects (224x224 by
#    default), preprocessed with ImageNet mean/std normalisation
#    (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) — see
#    OpenCVBridge's preprocessClassifierInput / DocumentClassifier.kt.
python export_onnx.py --checkpoint runs/best.ckpt --output classifier_v1.onnx
```

**Wiring it in**, identical pattern to §4:

```ts
import { ModelManager } from 'react-native-doc-scanner';

ModelManager.register({
  id: 'acme-v2',
  // ...other paths...
  classifierModelPath: '/data/user/0/com.acme.app/files/models/classifier_v1.onnx',
});
await ModelManager.activate('acme-v2');
```

**Label order matters**: the native decode (`kClassifierLabels` in
`ios/ONNXInference.swift`, `LABELS` in `android/.../DocumentClassifier.kt`)
reads the model's output vector by fixed index — `["PASSPORT",
"DRIVING_LICENCE", "ID_CARD", "RESIDENCE_PERMIT", "VISA", "GENERIC"]`. Your
training pipeline's class order must match this exactly, or predictions will
silently point at the wrong document type rather than error.

**Output contract note**: the ONNX graph exposes 12 named output tensors
(`cls_8`/`cls_16`/`cls_32`, `obj_8`/`obj_16`/`obj_32`, `bbox_8`/`bbox_16`/
`bbox_32`, `kps_8`/`kps_16`/`kps_32` — one triplet-of-triplets per stride
level), decoded natively rather than through OpenCV's `dnn`/`FaceDetectorYN`
C++ wrapper (this package doesn't link `opencv_dnn`). The decode math
(per-stride anchor-free grid, `score = sqrt(cls * obj)`, exponential box-size
regression, 5-point landmark offsets) is ported directly from OpenCV's own
`modules/objdetect/src/face_detect.cpp` — see `FaceDetector.kt` (Android) and
`ONNXInference.swift`'s "Face detector (YuNet)" section, decoded from the raw
ORT tensors and mapped back to buffer space by `HybridDocScanner.swift`'s
`analyzeFaceFrame`/`normalizeFaceOrientation` (iOS), for the implementation
and inline citations. Landmark order is `[rightEye, leftEye, noseTip,
rightMouthCorner, leftMouthCorner]`, "right"/"left" from the subject's own
perspective (OpenCV's documented `FaceDetectorYN` convention).
