#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="$ROOT_DIR/offline_experiment"
if [[ -x /usr/bin/mpirun ]]; then
  MPIRUN_BIN="${MPIRUN:-/usr/bin/mpirun}"
else
  MPIRUN_BIN="${MPIRUN:-mpirun}"
fi

cd "$EXP_DIR"

if [[ ! -x build/pcg_matrix_beaver ]]; then
  echo "Missing build/pcg_matrix_beaver in $EXP_DIR"
  echo "Please build it first (make offline) or copy a validated binary."
  exit 1
fi

# Auto-detect topology. You can override with env vars below.
CORES_PER_SOCKET="$(lscpu | awk -F: '/Core\(s\) per socket/{gsub(/ /,"",$2); print $2; exit}')"
THREADS_PER_CORE="$(lscpu | awk -F: '/Thread\(s\) per core/{gsub(/ /,"",$2); print $2; exit}')"
SOCKETS="$(lscpu | awk -F: '/Socket\(s\)/{gsub(/ /,"",$2); print $2; exit}')"
LOGICAL_CPUS_NPROC="$(command nproc 2>/dev/null || true)"
LOGICAL_CPUS_AFF="$(python3 - <<'PY'
import os
try:
    print(len(os.sched_getaffinity(0)))
except Exception:
    print("")
PY
)"
LOGICAL_CPUS_GETCONF="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
LOGICAL_CPUS="${LOGICAL_CPUS_AFF:-}"
if [[ -z "${LOGICAL_CPUS}" ]]; then
  LOGICAL_CPUS="${LOGICAL_CPUS_GETCONF:-}"
fi
if [[ -z "${LOGICAL_CPUS}" ]]; then
  LOGICAL_CPUS="${LOGICAL_CPUS_NPROC:-64}"
fi

if [[ -z "${CORES_PER_SOCKET}" || -z "${THREADS_PER_CORE}" || -z "${SOCKETS}" ]]; then
  echo "Failed to parse lscpu topology."
  exit 1
fi

# MPI binding: PE = physical cores per rank (default: one socket per rank).
PCG_PE="${PCG_PE:-$CORES_PER_SOCKET}"
# OpenMP threads per rank (default: same as PE, no SMT oversubscription).
PCG_OMP_THREADS="${PCG_OMP_THREADS:-$PCG_PE}"
# Hostfile slots (default: logical CPU count).
PCG_SLOTS="${PCG_SLOTS:-$LOGICAL_CPUS}"

# Workload params (overridable)
PCG_M="${PCG_M:-1024}"
PCG_K="${PCG_K:-64}"
PCG_N="${PCG_N:-1024}"
PCG_BITS="${PCG_BITS:-64}"
PCG_CHANNELS="${PCG_CHANNELS:-20}"
PCG_BATCH="${PCG_BATCH:-256}"

echo "Topology: sockets=$SOCKETS, cores/socket=$CORES_PER_SOCKET, threads/core=$THREADS_PER_CORE, logical_cpus=$LOGICAL_CPUS"
echo "Config: PE=$PCG_PE, OMP_THREADS=$PCG_OMP_THREADS, slots=$PCG_SLOTS, MKN=${PCG_M}x${PCG_K}x${PCG_N}, channels=$PCG_CHANNELS, batch=$PCG_BATCH"
echo "MPI: $MPIRUN_BIN"
"$MPIRUN_BIN" --version 2>/dev/null | head -1 || true
if [[ -n "${LOGICAL_CPUS_NPROC}" && "${LOGICAL_CPUS_NPROC}" != "${LOGICAL_CPUS}" ]]; then
  echo "Note: nproc=$LOGICAL_CPUS_NPROC but affinity-based cpu count=$LOGICAL_CPUS (using affinity count)."
fi
if (( PCG_OMP_THREADS > PCG_PE )); then
  echo "Note: OMP threads ($PCG_OMP_THREADS) > PE cores/rank ($PCG_PE). This is SMT/oversubscription mode and may reduce throughput."
fi

echo "$(hostname) slots=$PCG_SLOTS" > hostfile

unset OMP_DYNAMIC OMP_WAIT_POLICY OMP_PLACES OMP_PROC_BIND
export OMP_NUM_THREADS="$PCG_OMP_THREADS"

exec "$MPIRUN_BIN" -np 2 --hostfile hostfile \
  --map-by ppr:2:node:PE="$PCG_PE" --bind-to core --report-bindings \
  --mca pml ob1 --mca btl vader,self \
  ./build/pcg_matrix_beaver \
  --M "$PCG_M" --K "$PCG_K" --N "$PCG_N" --bits "$PCG_BITS" --channels "$PCG_CHANNELS" --batch "$PCG_BATCH" --no-verify
