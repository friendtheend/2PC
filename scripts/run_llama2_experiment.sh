#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run LLaMA2 SHAFT 8-layer profiling plus BriLLMFlow online linear operators and
collect logs/CSV.

Usage:
  bash scripts/run_llama2_experiment.sh

Common environment overrides:
  OUT=results/llama2_experiment_test
  SHAFT_ROOT=/path/to/shaft/tree        # default: ./llama_experiment
  MODEL_NAME=meta-llama/Llama-2-7b-hf
  LEN_DATA=8
  MAX_LAYERS=8
  FP16=1
  RUN_SHAFT=1
  RUN_FOURIER=1
  RUN_D2POLY=1
  RUN_BRILLM=1
  BIN=/path/to/gpu_matrix_beaver

Outputs:
  $OUT/env.txt
  $OUT/commands.txt
  $OUT/shaft_llama2_seq8_8layer_fourier.log
  $OUT/shaft_llama2_seq8_8layer_d2poly.log
  $OUT/shaft_summary.csv
  $OUT/brillm_online_llama_*.log
  $OUT/brillm_llama2_raw_online.csv
EOF
  exit 0
fi

SHAFT_ROOT="${SHAFT_ROOT:-$ROOT/llama_experiment}"
OUT="${OUT:-${ROOT}/results/llama2_experiment_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-hf}"
LEN_DATA="${LEN_DATA:-8}"
MAX_LAYERS="${MAX_LAYERS:-8}"
FP16="${FP16:-1}"
RUN_SHAFT="${RUN_SHAFT:-1}"
RUN_FOURIER="${RUN_FOURIER:-1}"
RUN_D2POLY="${RUN_D2POLY:-1}"
RUN_BRILLM="${RUN_BRILLM:-1}"
BIN="${BIN:-$ROOT/brillmflow_2pc/BMT/build/gpu_matrix_beaver}"
mkdir -p "$OUT"

{
  date
  uname -a
  echo "SHAFT_ROOT=$SHAFT_ROOT"
  echo "MODEL_NAME=$MODEL_NAME"
  echo "LEN_DATA=$LEN_DATA"
  echo "MAX_LAYERS=$MAX_LAYERS"
  echo "FP16=$FP16"
  echo "BIN=$BIN"
  nvidia-smi || true
  tc qdisc show || true
} > "$OUT/env.txt" 2>&1
: > "$OUT/commands.txt"

if [[ "$RUN_SHAFT" == "1" ]]; then
  cd "$SHAFT_ROOT/examples/text-generation"
  run_shaft() {
    local method="$1" log="$2"
    local -a cmd=(
      env WORLD_SIZE=2
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
      SHAFT_CUDA_DEVICES="${SHAFT_CUDA_DEVICES:-0,1}"
      PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:64}"
      FP16="$FP16"
      SILU_METHOD="$method"
      CRYPTEN_ONNX_FORCE_DISK=1
      CRYPTEN_KEEP_PYTORCH_MODEL=0
      MODEL_NAME="$MODEL_NAME"
      LEN_DATA="$LEN_DATA"
      MAX_LAYERS="$MAX_LAYERS"
      REPORT_COST=1
      bash bench_llama2_2pc_total.sh
    )
    printf '%q ' "${cmd[@]}" >> "$OUT/commands.txt"
    echo >> "$OUT/commands.txt"
    "${cmd[@]}" 2>&1 | tee "$log"
  }
  [[ "$RUN_FOURIER" == "1" ]] && run_shaft fourier "$OUT/shaft_llama2_seq8_8layer_fourier.log"
  [[ "$RUN_D2POLY" == "1" ]] && run_shaft d2poly "$OUT/shaft_llama2_seq8_8layer_d2poly.log"
fi

if [[ "$RUN_BRILLM" == "1" ]]; then
  if [[ ! -x "$BIN" ]]; then
    echo "Missing BriLLMFlow online binary: $BIN" >&2
    echo "Attempting to build it with scripts/build_brillm_online.sh..." >&2
    "$ROOT/scripts/build_brillm_online.sh"
  fi
  if [[ ! -x "$BIN" ]]; then
    echo "Still missing BriLLMFlow online binary after build attempt: $BIN" >&2
    exit 1
  fi
  OUT="$OUT" BIN="$BIN" "$ROOT/scripts/run_llama2_online_ops.sh"
fi

shaft_logs=()
[[ -f "$OUT/shaft_llama2_seq8_8layer_fourier.log" ]] && shaft_logs+=("llama2_fourier=$OUT/shaft_llama2_seq8_8layer_fourier.log")
[[ -f "$OUT/shaft_llama2_seq8_8layer_d2poly.log" ]] && shaft_logs+=("llama2_d2poly=$OUT/shaft_llama2_seq8_8layer_d2poly.log")
if [[ "${#shaft_logs[@]}" -gt 0 ]]; then
  python "$ROOT/scripts/parse_shaft_report.py" --out "$OUT/shaft_summary.csv" "${shaft_logs[@]}"
fi

if [[ -f "$OUT/brillm_online_llama_W_QKV.log" ]]; then
  python "$ROOT/scripts/parse_brillm_logs.py" \
    --kind online \
    --out "$OUT/brillm_llama2_raw_online.csv" \
    llama_W_QKV="$OUT/brillm_online_llama_W_QKV.log" \
    llama_W_O="$OUT/brillm_online_llama_W_O.log" \
    llama_W_gateup="$OUT/brillm_online_llama_W_gateup.log" \
    llama_W_down="$OUT/brillm_online_llama_W_down.log" \
    llama_QKT="$OUT/brillm_online_llama_QKT.log" \
    llama_scoresV="$OUT/brillm_online_llama_scoresV.log"
fi

echo "OUT=$OUT"
[[ -f "$OUT/shaft_summary.csv" ]] && echo "SHAFT CSV=$OUT/shaft_summary.csv"
[[ -f "$OUT/brillm_llama2_raw_online.csv" ]] && echo "BriLLMFlow CSV=$OUT/brillm_llama2_raw_online.csv"
