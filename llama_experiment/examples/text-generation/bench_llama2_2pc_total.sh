#!/usr/bin/env bash
set -euo pipefail

LEN_DATA="${LEN_DATA:-8}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-hf}"
SILU_METHOD="${SILU_METHOD:-fourier}"
FP16="${FP16:-1}"
REPORT_COST="${REPORT_COST:-1}"
MAX_LAYERS="${MAX_LAYERS:-8}"
LOG_FILE="${LOG_FILE:-/tmp/shaft_llama2_2pc_total.log}"

EXTRA_ARGS=()
if [[ "${FP16}" == "1" ]]; then
  EXTRA_ARGS+=(--fp16)
fi
if [[ "${REPORT_COST}" != "0" ]]; then
  EXTRA_ARGS+=(--report_cost)
fi
if [[ -n "${MAX_LAYERS}" ]]; then
  EXTRA_ARGS+=(--max_layers "${MAX_LAYERS}")
fi

python run_generation_private.py \
  --model_type=llama \
  --model_name_or_path="${MODEL_NAME}" \
  --len_data "${LEN_DATA}" \
  --length 1 \
  --estimate_mode total \
  --silu_method "${SILU_METHOD}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"

python - "${LOG_FILE}" <<'PY'
import re
import sys

log_path = sys.argv[1]
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

matches = re.findall(
    r"running time:\s*([0-9]+(?:\.[0-9]+)?)s,\s*comm byte:\s*([0-9]+(?:\.[0-9]+)?)\s*GB",
    text,
)
if not matches:
    raise SystemExit("Could not parse SHAFT latency/comm from log.")

latency_sec, comm_gb = matches[-1]
print(
    "BENCH_RESULT system=SHAFT model=Llama2-7B "
    f"latency_sec={latency_sec} comm_gb={comm_gb}"
)
PY
