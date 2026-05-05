#!/usr/bin/env bash
set -euo pipefail

NUM_IMAGES="${NUM_IMAGES:-1}"
CONFIG="${CONFIG:-$(pwd)/examples/python/conf/2pc.json}"
GELU_MODE="${GELU_MODE:-seg3_gelu}"

if [[ ! -x "./bazel-bin/examples/python/ml/flax_vit/flax_vit" ]]; then
  bazel build -c opt //examples/python/ml/flax_vit:flax_vit >/dev/null
fi

BAZEL_BIN_DIR="$(bazel info -c opt bazel-bin)"
VIT_BIN="${BAZEL_BIN_DIR}/examples/python/ml/flax_vit/flax_vit"

if [[ ! -x "${VIT_BIN}" ]]; then
  echo "ERROR: ViT binary not found at ${VIT_BIN}" >&2
  echo "Try: bazel build -c opt //examples/python/ml/flax_vit:flax_vit" >&2
  exit 1
fi

env SPU_BB_SET_IEQUAL_BITS=16 \
  "${VIT_BIN}" \
  --config "${CONFIG}" \
  --gelu-mode "${GELU_MODE}" \
  --num-images "${NUM_IMAGES}" \
  --spu-only
