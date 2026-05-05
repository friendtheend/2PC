#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run a GPT2-base seq=128 SHAFT vs BriLLMFlow online comparison and generate a
paper-style table.

Usage:
  bash scripts/run_gpt2_hybrid_table.sh

Common environment overrides:
  OUT=results/gpt2_hybrid_test
  SHAFT_ROOT=/path/to/shaft/tree        # default: ./vit_experiment
  MODEL_NAME=openai-community/gpt2
  LEN_DATA=128
  LENGTH=1
  GELU_METHOD=fourier|d2poly            # default: fourier vanilla SHAFT
  RUN_SHAFT=1
  RUN_SHAFT_COMP=1
  RUN_BRILLM=1
  BIN=/path/to/gpu_matrix_beaver
  BRILLM_CUDA_VISIBLE_DEVICES=rank|0    # default rank; use 0 for single-GPU two-rank runs

Outputs:
  $OUT/shaft_gpt2_l128_comp.log
  $OUT/shaft_gpt2_l128_report.log
  $OUT/brillm_online_gpt2_*.log
  $OUT/gpt2_brillm_raw_online.csv
  $OUT/gpt2_brillm_perop.csv
  $OUT/gpt2_hybrid_summary.csv
  $OUT/gpt2_table.tex
  $OUT/table5.tex
EOF
  exit 0
fi

SHAFT_ROOT="${SHAFT_ROOT:-$ROOT/vit_experiment}"
OUT="${OUT:-${ROOT}/results/gpt2_hybrid_$(date +%Y%m%d_%H%M%S)}"
BIN="${BIN:-${ROOT}/brillmflow_2pc/BMT/build/gpu_matrix_beaver}"
MODEL_NAME="${MODEL_NAME:-openai-community/gpt2}"
LEN_DATA="${LEN_DATA:-128}"
LENGTH="${LENGTH:-1}"
GELU_METHOD="${GELU_METHOD:-fourier}"
RUN_SHAFT="${RUN_SHAFT:-1}"
RUN_SHAFT_COMP="${RUN_SHAFT_COMP:-1}"
RUN_BRILLM="${RUN_BRILLM:-1}"
BRILLM_CUDA_VISIBLE_DEVICES="${BRILLM_CUDA_VISIBLE_DEVICES:-rank}"
ENABLE_OP_PROFILE="${ENABLE_OP_PROFILE:-1}"

mkdir -p "$OUT"

# Ensure local SHAFT/CrypTen modules are importable without pip-installing
# crypten into the active environment.
export PYTHONPATH="$SHAFT_ROOT:${PYTHONPATH:-}"

{
  date
  uname -a
  pwd
  echo "SHAFT_ROOT=$SHAFT_ROOT"
  echo "MODEL_NAME=$MODEL_NAME"
  echo "LEN_DATA=$LEN_DATA"
  echo "LENGTH=$LENGTH"
  echo "GELU_METHOD=$GELU_METHOD"
  echo "BIN=$BIN"
  echo "BRILLM_CUDA_VISIBLE_DEVICES=$BRILLM_CUDA_VISIBLE_DEVICES"
  echo "ENABLE_OP_PROFILE=$ENABLE_OP_PROFILE"
  nvidia-smi || true
  tc qdisc show || true
} > "$OUT/env.txt" 2>&1

if [[ ! -x "$BIN" ]]; then
  echo "Missing BriLLMFlow online binary: $BIN" >&2
  echo "Attempting to build it with scripts/build_brillm_online.sh..." >&2
  "$ROOT/scripts/build_brillm_online.sh"
fi
if [[ ! -x "$BIN" ]]; then
  echo "Still missing BriLLMFlow online binary after build attempt: $BIN" >&2
  exit 1
fi

SHAFT_LOG="$OUT/shaft_gpt2_l128_report.log"
if [[ "$RUN_SHAFT" == "1" ]]; then
  cd "$SHAFT_ROOT/examples/text-generation"
  if [[ "$RUN_SHAFT_COMP" == "1" ]]; then
    SHAFT_COMP_ARGS=(
      python run_generation_private.py
      --model_type=gpt2
      --model_name_or_path="$MODEL_NAME"
      --len_data "$LEN_DATA"
      --comp
      --length "$LENGTH"
    )
    if [[ -n "$GELU_METHOD" ]]; then
      SHAFT_COMP_ARGS+=(--gelu_method "$GELU_METHOD")
    fi
    printf '%q ' "${SHAFT_COMP_ARGS[@]}" > "$OUT/shaft_comp_command.txt"
    echo >> "$OUT/shaft_comp_command.txt"
    "${SHAFT_COMP_ARGS[@]}" 2>&1 | tee "$OUT/shaft_gpt2_l128_comp.log"
  fi

  SHAFT_ARGS=(
    python run_generation_private.py
    --model_type=gpt2
    --model_name_or_path="$MODEL_NAME"
    --len_data "$LEN_DATA"
    --length "$LENGTH"
    --estimate_mode total
    --report_cost
  )
  if [[ -n "$GELU_METHOD" ]]; then
    SHAFT_ARGS+=(--gelu_method "$GELU_METHOD")
  fi
  printf '%q ' "${SHAFT_ARGS[@]}" > "$OUT/shaft_command.txt"
  echo >> "$OUT/shaft_command.txt"
  if [[ "$ENABLE_OP_PROFILE" == "1" ]]; then
    CRYPTEN_PROFILE_LINEAR_OPS=1 \
    CRYPTEN_PROFILE_RANK0_ONLY=1 \
    "${SHAFT_ARGS[@]}" 2>&1 | tee "$SHAFT_LOG"
    python "$ROOT/scripts/parse_shaft_op_profile.py" \
      --log "$SHAFT_LOG" \
      --raw-out "$OUT/shaft_op_profile_raw.csv" \
      --agg-out "$OUT/shaft_op_profile_agg.csv" || true
  else
    "${SHAFT_ARGS[@]}" 2>&1 | tee "$SHAFT_LOG"
  fi
fi

if [[ "$RUN_BRILLM" == "1" ]]; then
  cd "$ROOT"
  PCG="$OUT/brillm_dummy_pcg"
  mkdir -p "$PCG"
  MPIRUN=(mpirun -np 2 --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo)

  run_one() {
    local label="$1" M="$2" K="$3" N="$4" iters="$5"
    local prefix="$PCG/${label}_party"
    local log="$OUT/brillm_online_${label}.log"
    "$ROOT/scripts/create_dummy_matrix_pcg.py" --prefix "$prefix" --M "$M" --K "$K" --N "$N" --bits 64 > "$OUT/create_${label}.txt"
    echo "==== $label ($M,$K,$N), iters=$iters ====" | tee "$log"
    "${MPIRUN[@]}" bash -lc "if [[ '$BRILLM_CUDA_VISIBLE_DEVICES' == rank ]]; then export CUDA_VISIBLE_DEVICES=\${OMPI_COMM_WORLD_LOCAL_RANK}; else export CUDA_VISIBLE_DEVICES='$BRILLM_CUDA_VISIBLE_DEVICES'; fi; exec '$BIN' --pcg '$prefix' --iterations '$iters'" >> "$log" 2>&1
    grep -E "Matrix:|Total time:|PRG time:|Comm time:|GEMM time:|Communication per iter" "$log" | tail -10
  }

  run_one gpt2_W_QKV    128 768  768  1000
  run_one gpt2_W_O      128 768  768  1000
  run_one gpt2_W_1      128 768  3072 500
  run_one gpt2_W_2      128 3072 768  500
  run_one gpt2_QKT      128 64   128  1000
  run_one gpt2_scoresV  128 128  64   1000
fi

python "$ROOT/scripts/parse_brillm_logs.py" \
  --kind online \
  --out "$OUT/gpt2_brillm_raw_online.csv" \
  gpt2_W_QKV="$OUT/brillm_online_gpt2_W_QKV.log" \
  gpt2_W_O="$OUT/brillm_online_gpt2_W_O.log" \
  gpt2_W_1="$OUT/brillm_online_gpt2_W_1.log" \
  gpt2_W_2="$OUT/brillm_online_gpt2_W_2.log" \
  gpt2_QKT="$OUT/brillm_online_gpt2_QKT.log" \
  gpt2_scoresV="$OUT/brillm_online_gpt2_scoresV.log"

python "$ROOT/scripts/make_gpt2_hybrid_table.py" \
  --shaft-log "$SHAFT_LOG" \
  --brillm-csv "$OUT/gpt2_brillm_raw_online.csv" \
  --out-dir "$OUT"

echo "OUT=$OUT"
echo "LaTeX: $OUT/table5.tex"
echo "Summary: $OUT/gpt2_hybrid_summary.csv"
