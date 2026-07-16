#!/usr/bin/env bash
# Fetches the default RapidOCR ONNX weights into assets/models/.
#
# DocAligner's detector weights are NOT fetched by this script — they're
# downloaded on first use by the `docaligner-docsaid` Python package (from
# Hugging Face) rather than served at a stable static URL. Run the
# `pip install docaligner-docsaid` snippet in docs/MODEL_TRAINING.md §1 once,
# then copy the resulting .onnx file here yourself.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p assets/models

BASE_URL="https://huggingface.co/SWHL/RapidOCR/resolve/main"

echo "Fetching RapidOCR ONNX models (Apache-2.0, huggingface.co/SWHL/RapidOCR)..."

# General-purpose (script-agnostic) text-region detector — text detection
# doesn't need language-specific training, so the general PP-OCRv4 detector
# is RapidOCR's own recommended pairing for English recognition too. (Real
# repo layout is flat per version folder — no det/rec/cls subdirectories.)
curl -L --fail -o assets/models/ch_PP-OCRv4_det_infer.onnx \
  "${BASE_URL}/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx"

# 2-class text-line orientation classifier — unchanged since PP-OCR v1, only
# ever published under the PP-OCRv1 folder (verified via the HF repo's file
# tree; PP-OCRv2/v3 do not have a "_cls_infer.onnx" of their own).
curl -L --fail -o assets/models/ch_ppocr_mobile_v2.0_cls_infer.onnx \
  "${BASE_URL}/PP-OCRv1/ch_ppocr_mobile_v2.0_cls_infer.onnx"

curl -L --fail -o assets/models/en_PP-OCRv3_rec_infer.onnx \
  "${BASE_URL}/PP-OCRv3/en_PP-OCRv3_rec_infer.onnx"

# Verified byte-identical to PaddleOCR's own release/2.6 en_dict.txt (the
# release line PP-OCRv3 shipped under) — safe to pair with the v3 rec model
# above despite coming from RapidOCR's current (v4-era) default config.
curl -L --fail -o assets/models/en_dict.txt \
  "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.1/paddle/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile/en_dict.txt"

echo "Done. Still missing: assets/models/docaligner_fastvit_heatmap.onnx — see docs/MODEL_TRAINING.md §1."

# YuNet face detector (MIT, opencv_zoo) — used by analyzeFaceFrame for the selfie capture
# guide. Optional: only fetch it if you're using the selfie flow, see docs/MODEL_TRAINING.md
# §5. Left as a separate, additive block since this fetches into the same shared
# assets/models/ directory used by both the Android and iOS halves of this package.
curl -L --fail -o assets/models/face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

echo "Done. Fetched assets/models/face_detection_yunet_2023mar.onnx (MIT, opencv_zoo)."

# Document-type classifier: NOT fetched by this script — unlike the models above, there's no
# off-the-shelf Apache/MIT ONNX model with the right taxonomy (PASSPORT/DRIVING_LICENCE/ID_CARD/
# RESIDENCE_PERMIT/VISA vs GENERIC) to point a curl at. Fine-tune your own and register it via
# ModelManager — see docs/MODEL_TRAINING.md §6. Fully optional: leave classifierModelPath empty
# and documentType simply stays GENERIC, exactly as it did before this feature existed.
