#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-google/vit-base-patch16-224}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1}"
REPORT_COST="${REPORT_COST:-0}"
GELU_METHOD="${GELU_METHOD:-poly}"

EXTRA_ARGS=()
if [[ "${REPORT_COST}" != "0" ]]; then
  EXTRA_ARGS+=(--report_cost)
fi

python run_image_classification_private.py \
  --model_name_or_path "${MODEL_NAME}" \
  --max_eval_samples "${EVAL_SAMPLES}" \
  --gelu_method "${GELU_METHOD}" \
  --seed 42 \
  "${EXTRA_ARGS[@]}"
