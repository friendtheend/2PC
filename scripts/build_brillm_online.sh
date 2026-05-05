#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BMT="$ROOT/brillmflow_2pc/BMT"
mkdir -p "$BMT/build"

NVCC="${NVCC:-$(command -v nvcc)}"
MPICXX="${MPICXX:-$(command -v mpicxx)}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"

if [[ -z "${NVCC}" || ! -x "${NVCC}" ]]; then
  echo "Missing nvcc. Install CUDA toolkit or set NVCC=/path/to/nvcc." >&2
  exit 1
fi
if [[ -z "${MPICXX}" || ! -x "${MPICXX}" ]]; then
  echo "Missing mpicxx. Install OpenMPI or set MPICXX=/path/to/mpicxx." >&2
  exit 1
fi

MPI_INC="$($MPICXX -showme:compile 2>/dev/null | grep -oE '\-I[^ ]+' | head -1 | sed 's/-I//')"
MPI_LIB="$($MPICXX -showme:link 2>/dev/null | grep -oE '\-L[^ ]+' | head -1 | sed 's/-L//')"

if [[ -z "${MPI_INC}" || -z "${MPI_LIB}" ]]; then
  echo "Could not infer OpenMPI include/lib flags from: $MPICXX -showme" >&2
  echo "This build helper expects OpenMPI's mpicxx wrapper." >&2
  exit 1
fi

"$NVCC" -O3 -std=c++17 -arch="$CUDA_ARCH" \
  "$BMT/gpu_matrix_beaver_online.cu" \
  -I"$MPI_INC" -L"$MPI_LIB" -lmpi \
  -o "$BMT/build/gpu_matrix_beaver"

echo "$BMT/build/gpu_matrix_beaver"
