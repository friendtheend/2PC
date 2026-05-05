#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/offline_experiment/build/pcg_matrix_beaver"
OUT="${OUT:-$ROOT/results/offline_all_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

if [[ ! -x "$BIN" ]]; then
  echo "Missing executable: $BIN"
  echo "Build first: cd $ROOT/offline_experiment && MPICXX=/usr/bin/mpicxx make offline && chmod +x build/pcg_matrix_beaver"
  exit 1
fi

MPIRUN_BIN="${MPIRUN:-/usr/bin/mpirun}"
PE="${PCG_PE:-48}"
OMP_T="${PCG_OMP_THREADS:-96}"
SLOTS="${PCG_SLOTS:-192}"
BITS="${PCG_BITS:-64}"
CH="${PCG_CHANNELS:-20}"
BATCH="${PCG_BATCH:-256}"

echo "$(hostname) slots=${SLOTS}" > "$OUT/hostfile"
cat > "$OUT/env.txt" <<EOF
ROOT=$ROOT
BIN=$BIN
OUT=$OUT
MPIRUN=$MPIRUN_BIN
PE=$PE
OMP_THREADS=$OMP_T
SLOTS=$SLOTS
BITS=$BITS
CHANNELS=$CH
BATCH=$BATCH
EOF

run_one() {
  local model="$1" label="$2" M="$3" K="$4" N="$5"
  local log="$OUT/${model}_${label}.log"
  echo "==== ${model}_${label} M=$M K=$K N=$N ===="
  (
    cd "$ROOT/offline_experiment"
    unset OMP_DYNAMIC OMP_WAIT_POLICY OMP_PLACES OMP_PROC_BIND
    export OMP_NUM_THREADS="$OMP_T"
    "$MPIRUN_BIN" -np 2 --hostfile "$OUT/hostfile" \
      --map-by ppr:2:node:PE="$PE" --bind-to core --report-bindings \
      --mca pml ob1 --mca btl vader,self \
      "$BIN" \
      --M "$M" --K "$K" --N "$N" --bits "$BITS" --channels "$CH" --batch "$BATCH" --no-verify \
      2>&1 | tee "$log"
  )
}

# ---------------- LLaMA2 (offline table stress shapes) ----------------
run_one llama2 W_QKV    4096  4096  4096
run_one llama2 QKT       256   256   256
run_one llama2 scoresV  1024  1024  1024
run_one llama2 W_O      4096  4096  4096
run_one llama2 W_gateup 4096  4096 11008
run_one llama2 W_down   4096 11008  4096

# ---------------- BERT-base ----------------
run_one bert W_QKV      768   768   768
run_one bert QKT        128    64   128
run_one bert scoresV    128   128    64
run_one bert W_O        768   768   768
run_one bert W_1        768   768  3072
run_one bert W_2        768  3072   768

# ---------------- GPT2-base (same linear family as bert) ----------------
run_one gpt2 W_QKV      768   768   768
run_one gpt2 QKT        128    64   128
run_one gpt2 scoresV    128   128    64
run_one gpt2 W_O        768   768   768
run_one gpt2 W_1        768   768  3072
run_one gpt2 W_2        768  3072   768

echo
echo "DONE: $OUT"
echo "Quick extract:"
echo "  rg -n \"Done:|Time: PRG|Communication:|Throughput:\" $OUT/*.log"

