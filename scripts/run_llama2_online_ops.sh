#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/brillmflow_2pc/BMT/build/gpu_matrix_beaver}"
OUT="${OUT:-$ROOT/results/rerun_llama2_online_$(date +%Y%m%d_%H%M%S)}"
BRILLM_CUDA_VISIBLE_DEVICES="${BRILLM_CUDA_VISIBLE_DEVICES:-rank}"
BRILLM_LLAMA_MODE="${BRILLM_LLAMA_MODE:-real_forward}"
PCG="$OUT/dummy_pcg"
mkdir -p "$PCG"

if [[ ! -x "$BIN" ]]; then
  echo "Missing binary: $BIN" >&2
  echo "Run scripts/build_brillm_online.sh first." >&2
  exit 1
fi

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

if [[ "$BRILLM_LLAMA_MODE" == "real_forward" ]]; then
  # LLaMA-2-7B, seq=8 real-forward-shape operators.
  run_one llama_W_QKV    8 4096  4096  100
  run_one llama_W_O      8 4096  4096  100
  run_one llama_W_gateup 8 4096 11008  100
  run_one llama_W_down   8 11008 4096  100
  run_one llama_QKT      8 128      8 1000
  run_one llama_scoresV  8 8      128 1000
elif [[ "$BRILLM_LLAMA_MODE" == "stress" ]]; then
  # Legacy stress-shape mode (kept as backup/comparison).
  run_one llama_W_QKV    4096 4096  4096  2
  run_one llama_W_O      4096 4096  4096  2
  run_one llama_W_gateup 4096 4096 11008  2
  run_one llama_W_down   4096 11008 4096  2
  run_one llama_QKT      128 128     128 10
  run_one llama_scoresV  128 128     128 10
else
  echo "Unsupported BRILLM_LLAMA_MODE=$BRILLM_LLAMA_MODE (expected real_forward|stress)" >&2
  exit 1
fi

# Keep mode-specific backup copies in addition to canonical names.
for f in "$OUT"/brillm_online_llama_*.log; do
  cp -f "$f" "${f%.log}_${BRILLM_LLAMA_MODE}.log"
done

echo "OUT=$OUT"
echo "BRILLM_LLAMA_MODE=$BRILLM_LLAMA_MODE"
