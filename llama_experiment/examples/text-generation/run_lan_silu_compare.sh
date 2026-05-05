#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_lan_silu_compare.sh
# Optional env vars:
#   DEV=lo RTT_MS=0.5 WORLD_SIZE=2 LEN_DATA=8 MAX_LAYERS=16 MODEL_NAME=meta-llama/Llama-2-7b-hf

DEV="${DEV:-lo}"
RTT_MS="${RTT_MS:-0.5}"
WORLD_SIZE="${WORLD_SIZE:-2}"
LEN_DATA="${LEN_DATA:-8}"
MAX_LAYERS="${MAX_LAYERS:-16}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-hf}"
PY_BIN="${PY_BIN:-$(command -v python3 || command -v python)}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_PY="${SCRIPT_DIR}/run_generation_private.py"

if [[ -z "${PY_BIN}" ]]; then
  echo "ERROR: python3/python not found. Set PY_BIN explicitly." >&2
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "ERROR: sudo is required for tc setup." >&2
  exit 1
fi
if ! command -v tc >/dev/null 2>&1; then
  echo "ERROR: tc not found." >&2
  exit 1
fi

cleanup_tc() {
  sudo tc qdisc del dev "${DEV}" root 2>/dev/null || true
}

setup_tc() {
  local bw_gbps="$1"
  local one_way_ms
  one_way_ms=$(python - <<PY
rtt = float("${RTT_MS}")
print(rtt / 2.0)
PY
)

  cleanup_tc
  sudo tc qdisc add dev "${DEV}" root handle 1: htb default 10
  sudo tc class add dev "${DEV}" parent 1: classid 1:10 htb rate "${bw_gbps}gbit" ceil "${bw_gbps}gbit"
  sudo tc qdisc add dev "${DEV}" parent 1:10 handle 10: netem delay "${one_way_ms}ms"
}

run_one() {
  local bw="$1"
  local method="$2"
  local log_file="/tmp/shaft_lan_${bw}gbps_${method}.log"

  echo "=== RUN bw=${bw}Gbps rtt=${RTT_MS}ms method=${method} ==="

  WORLD_SIZE="${WORLD_SIZE}" \
  CUDA_VISIBLE_DEVICES=0 \
  SHAFT_CUDA_DEVICES=0 \
  PYTORCH_ALLOC_CONF=max_split_size_mb:64 \
  FP16=1 \
  SILU_METHOD="${method}" \
  CRYPTEN_ONNX_FORCE_DISK=1 \
  CRYPTEN_KEEP_PYTORCH_MODEL=0 \
  "${PY_BIN}" "${RUN_PY}" \
    --model_type llama \
    --model_name_or_path "${MODEL_NAME}" \
    --len_data "${LEN_DATA}" --length 1 \
    --estimate_mode total --fp16 --report_cost \
    --max_layers "${MAX_LAYERS}" \
    --silu_method "${method}" \
    2>&1 | tee "${log_file}"

  python - "${log_file}" "${bw}" "${method}" <<'PY'
import re, sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(errors='ignore')
bw = sys.argv[2]
method = sys.argv[3]

keys = [
    "total_running_time",
    "total_comm_time",
    "total_comm_bytes",
    "total_comm_rounds",
    "silu_time",
    "silu_comm_time",
    "silu_comm_bytes",
    "silu_comm_rounds",
]

def vals(k):
    float_re = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    return [float(x) for x in re.findall(rf"{re.escape(k)}:\s*{float_re}", log)]

run_line = re.findall(
    r"running time:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)s,\s*comm byte:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*GB",
    log,
)
rt = float(run_line[-1][0]) if run_line else None

allv = {k: vals(k) for k in keys}
tr = allv["total_running_time"]
if not tr:
    raise SystemExit("Could not parse total_running_time from log.")
idx = 0
if rt is not None:
    idx = min(range(len(tr)), key=lambda i: abs(tr[i] - rt))
pick = {}
for k in keys:
    seq = allv[k]
    if not seq:
        pick[k] = float("nan")
    else:
        pick[k] = seq[idx] if idx < len(seq) else seq[-1]
print(
    "LAN_RESULT "
    f"bw_gbps={bw} method={method} "
    f"total_time_sec={pick['total_running_time']:.6f} "
    f"total_comm_gb={pick['total_comm_bytes']:.6f} "
    f"total_rounds={int(pick['total_comm_rounds'])} "
    f"silu_time_sec={pick['silu_time']:.6f} "
    f"silu_comm_sec={pick['silu_comm_time']:.6f} "
    f"silu_comm_gb={pick['silu_comm_bytes']:.6f} "
    f"silu_rounds={int(pick['silu_comm_rounds'])}"
)
PY
}

trap cleanup_tc EXIT

for bw in 1 10; do
  setup_tc "${bw}"
  run_one "${bw}" d2poly
  run_one "${bw}" fourier
done

echo "=== DONE ==="
