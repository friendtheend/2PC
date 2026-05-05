#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run ViT-B/16 SHAFT private inference experiments and collect logs/CSV.

Usage:
  bash scripts/run_vit_experiment.sh

Common environment overrides:
  OUT=results/vit_experiment_test
  SHAFT_ROOT=/path/to/shaft/tree        # default: ./vit_experiment
  MODEL_NAME=google/vit-base-patch16-224
  EVAL_SAMPLES=1
  RUN_D2POLY=1
  RUN_FOURIER=1
  RUN_COMP_COMM=1

Outputs:
  $OUT/env.txt
  $OUT/commands.txt
  $OUT/vit_d2poly_report.log
  $OUT/vit_fourier_report.log
  $OUT/vit_comp.log
  $OUT/vit_comm.log
  $OUT/shaft_summary.csv
EOF
  exit 0
fi

SHAFT_ROOT="${SHAFT_ROOT:-$ROOT/vit_experiment}"
OUT="${OUT:-${ROOT}/results/vit_experiment_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-google/vit-base-patch16-224}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1}"
RUN_D2POLY="${RUN_D2POLY:-1}"
RUN_FOURIER="${RUN_FOURIER:-1}"
RUN_COMP_COMM="${RUN_COMP_COMM:-1}"
mkdir -p "$OUT"

{
  date
  uname -a
  echo "SHAFT_ROOT=$SHAFT_ROOT"
  echo "MODEL_NAME=$MODEL_NAME"
  echo "EVAL_SAMPLES=$EVAL_SAMPLES"
  nvidia-smi || true
  tc qdisc show || true
} > "$OUT/env.txt" 2>&1
: > "$OUT/commands.txt"

cd "$SHAFT_ROOT/examples/image-classification"

run_report() {
  local method="$1" log="$2"
  local -a cmd=(
    env GELU_METHOD="$method" REPORT_COST=1 EVAL_SAMPLES="$EVAL_SAMPLES" MODEL_NAME="$MODEL_NAME"
    bash bench_vitb16_2pc_total.sh
  )
  printf '%q ' "${cmd[@]}" >> "$OUT/commands.txt"
  echo >> "$OUT/commands.txt"
  "${cmd[@]}" 2>&1 | tee "$log"
}

if [[ "$RUN_D2POLY" == "1" ]]; then
  run_report d2poly "$OUT/vit_d2poly_report.log"
fi
if [[ "$RUN_FOURIER" == "1" ]]; then
  run_report fourier "$OUT/vit_fourier_report.log"
fi
if [[ "$RUN_COMP_COMM" == "1" ]]; then
  env GELU_METHOD=fourier python run_image_classification_private.py \
    --model_name_or_path "$MODEL_NAME" \
    --max_eval_samples "$EVAL_SAMPLES" \
    --gelu_method fourier \
    --comp \
    --seed 42 2>&1 | tee "$OUT/vit_comp.log"
  env GELU_METHOD=fourier python run_image_classification_private.py \
    --model_name_or_path "$MODEL_NAME" \
    --max_eval_samples "$EVAL_SAMPLES" \
    --gelu_method fourier \
    --seed 42 2>&1 | tee "$OUT/vit_comm.log"
fi

logs=()
[[ -f "$OUT/vit_d2poly_report.log" ]] && logs+=("vit_d2poly=$OUT/vit_d2poly_report.log")
[[ -f "$OUT/vit_fourier_report.log" ]] && logs+=("vit_fourier=$OUT/vit_fourier_report.log")
[[ -f "$OUT/vit_comp.log" ]] && logs+=("vit_comp=$OUT/vit_comp.log")
[[ -f "$OUT/vit_comm.log" ]] && logs+=("vit_comm=$OUT/vit_comm.log")

python "$ROOT/scripts/parse_shaft_report.py" --out "$OUT/shaft_summary.csv" "${logs[@]}"

echo "OUT=$OUT"
echo "CSV=$OUT/shaft_summary.csv"
