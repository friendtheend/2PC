#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/brillmflow_2pc/BMT/build/gpu_matrix_beaver}"
OUT="${OUT:-$ROOT/results/rerun_llama2_online_$(date +%Y%m%d_%H%M%S)}"
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
  "${MPIRUN[@]}" bash -lc "export CUDA_VISIBLE_DEVICES=\${OMPI_COMM_WORLD_LOCAL_RANK}; exec '$BIN' --pcg '$prefix' --iterations '$iters'" >> "$log" 2>&1
  grep -E "Matrix:|Total time:|PRG time:|Comm time:|GEMM time:|Communication per iter" "$log" | tail -10
}

# LLaMA-2-7B, seq=8 forward-shape operators.
run_one llama_W_QKV    8 4096  4096  100
run_one llama_W_O      8 4096  4096  100
run_one llama_W_gateup 8 4096 11008  100
run_one llama_W_down   8 11008 4096  100
run_one llama_QKT      8 128      8 1000
run_one llama_scoresV  8 8      128 1000

echo "OUT=$OUT"
