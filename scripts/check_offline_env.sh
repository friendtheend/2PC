#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="$ROOT_DIR/offline_experiment"
BIN="$EXP_DIR/build/pcg_matrix_beaver"

MPIRUN_BIN="${MPIRUN:-$(command -v mpirun || true)}"
MPICXX_BIN="${MPICXX:-$(command -v mpicxx || true)}"

echo "== Offline Environment Check =="
echo "Root: $ROOT_DIR"
echo "Experiment dir: $EXP_DIR"
echo

echo "== Toolchain =="
printf "%-12s %s\n" "mpirun" "${MPIRUN_BIN:-MISSING}"
printf "%-12s %s\n" "mpicxx" "${MPICXX_BIN:-MISSING}"
printf "%-12s %s\n" "python3" "$(command -v python3 || echo MISSING)"
printf "%-12s %s\n" "lscpu" "$(command -v lscpu || echo MISSING)"
echo

if [[ -n "${MPIRUN_BIN:-}" ]]; then
  echo "mpirun version:"
  "$MPIRUN_BIN" --version 2>/dev/null | head -1 || true
fi
if [[ -n "${MPICXX_BIN:-}" ]]; then
  echo "mpicxx version:"
  "$MPICXX_BIN" --version 2>/dev/null | head -1 || true
fi
echo

echo "== CPU Topology / Availability =="
lscpu | awk -F: '/Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU\(s\)/{gsub(/^[ \t]+/,"",$2); printf "%-24s %s\n",$1,$2}'
echo "nproc: $(command nproc 2>/dev/null || echo N/A)"
echo "Cpus_allowed_list: $(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')"
python3 - <<'PY'
import os
print("os.cpu_count():", os.cpu_count())
try:
    print("sched_getaffinity:", len(os.sched_getaffinity(0)))
except Exception as e:
    print("sched_getaffinity: ERROR", e)
PY
echo

echo "== Offline Binary =="
if [[ -f "$BIN" ]]; then
  ls -l "$BIN"
  file "$BIN" || true
  if [[ -x "$BIN" ]]; then
    echo "binary executable: YES"
  else
    echo "binary executable: NO (run: chmod +x $BIN)"
  fi
  echo "linked MPI libs:"
  ldd "$BIN" | grep -E 'libmpi|open-rte|open-pal' || true
else
  echo "binary missing: $BIN"
  echo "build command:"
  echo "  cd $EXP_DIR && MPICXX=/usr/bin/mpicxx make offline && chmod +x build/pcg_matrix_beaver"
fi
echo

echo "== libOTe / deps (expected by offline_experiment Makefile) =="
missing=0
for hdr in \
  /usr/local/include/libOTe/TwoChooseOne/Iknp/IknpOtExtSender.h \
  /usr/local/include/cryptoTools/Crypto/PRNG.h \
  /usr/local/include/coproto/Socket/AsioSocket.h; do
  if [[ -f "$hdr" ]]; then
    echo "OK header: $hdr"
  else
    echo "MISSING header: $hdr"
    missing=1
  fi
done
for lib in \
  /usr/local/lib/libKyberOT.a \
  /usr/local/lib/liblibOTe.a \
  /usr/local/lib/libSimplestOT.a \
  /usr/local/lib/libcryptoTools.a \
  /usr/local/lib/libcoproto.a; do
  if [[ -f "$lib" ]]; then
    echo "OK lib: $lib"
  else
    echo "MISSING lib: $lib"
    missing=1
  fi
done
echo

echo "== MPI consistency hint =="
if [[ -f "$BIN" ]]; then
  bin_mpi="$(ldd "$BIN" | awk '/libmpi\.so/{print $3; exit}')"
  echo "binary links libmpi: ${bin_mpi:-unknown}"
  echo "runtime mpirun: ${MPIRUN_BIN:-unknown}"
  echo "If you see 'Need 2 MPI processes', force:"
  echo "  MPICXX=/usr/bin/mpicxx (compile) and MPIRUN=/usr/bin/mpirun (run)"
fi
echo

if [[ $missing -eq 0 ]]; then
  echo "Environment check: PASS (core deps found)"
else
  echo "Environment check: WARN (some deps missing)"
fi
