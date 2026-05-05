#!/usr/bin/env bash
set -euo pipefail

LEN_DATA="${LEN_DATA:-128}"
MODEL_NAME="${MODEL_NAME:-openai-community/gpt2}"
REPORT_COST="${REPORT_COST:-1}"
GELU_METHOD="${GELU_METHOD:-fourier}"
LOG_FILE="${LOG_FILE:-/tmp/shaft_gpt2_2pc_total.log}"

EXTRA_ARGS=()
if [[ "$REPORT_COST" != "0" ]]; then
  EXTRA_ARGS+=(--report_cost)
fi
if [[ -n "$GELU_METHOD" ]]; then
  EXTRA_ARGS+=(--gelu_method "$GELU_METHOD")
fi

python run_generation_private.py \
  --model_type=gpt2 \
  --model_name_or_path="${MODEL_NAME}" \
  --len_data "${LEN_DATA}" \
  --length 1 \
  --estimate_mode total \
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
print(f"BENCH_RESULT system=SHAFT latency_sec={latency_sec} comm_gb={comm_gb}")
PY
