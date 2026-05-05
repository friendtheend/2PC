#!/bin/bash
# build_gpu.sh - 编译 GPU 加速版本的 BMT（Offline + Online）

set -e

# 检测 CUDA 路径
CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
if [ ! -d "$CUDA_PATH" ]; then
    echo "Error: CUDA not found at $CUDA_PATH"
    echo "Please set CUDA_PATH environment variable"
    exit 1
fi

# 检测 libOTe 路径
LIBOTE_PATH=${LIBOTE_PATH:-/usr/local}

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Building GPU-accelerated BMT Generator                      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  CUDA_PATH:   $(printf '%-45s' "$CUDA_PATH")║"
echo "║  LIBOTE_PATH: $(printf '%-45s' "$LIBOTE_PATH")║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 创建 build 目录
mkdir -p build
cd build

# GPU 架构选项
GPU_ARCH="-gencode arch=compute_80,code=sm_80 \
          -gencode arch=compute_86,code=sm_86 \
          -gencode arch=compute_89,code=sm_89 \
          -gencode arch=compute_90,code=sm_90"

# ============================================================
# 编译标志 (与原 Makefile 一致)
# ============================================================
CXXFLAGS="-O3 -std=c++20 -Wall -fcoroutines -fopenmp -march=native -pthread"
CXXFLAGS="$CXXFLAGS -I${LIBOTE_PATH}/include"
CXXFLAGS="$CXXFLAGS -I${CUDA_PATH}/include"
CXXFLAGS="$CXXFLAGS -DCOPROTO_ENABLE_BOOST"

LDFLAGS="-no-pie"

# 静态库 (与原 Makefile 一致的顺序!)
LIBS="${LIBOTE_PATH}/lib/libKyberOT.a \
      ${LIBOTE_PATH}/lib/liblibOTe.a \
      ${LIBOTE_PATH}/lib/libSimplestOT.a \
      ${LIBOTE_PATH}/lib/libcryptoTools.a \
      ${LIBOTE_PATH}/lib/libcoproto.a \
      -lsodium \
      -lboost_system -lboost_thread -lboost_filesystem \
      -lssl -lcrypto -ldl -lpthread -lm \
      -L${CUDA_PATH}/lib64 -lcudart"

# ============================================================
# 1. 编译 Online 阶段（纯 CUDA，无依赖）
# ============================================================
echo "=== [1/3] Compiling Online Phase (PRG expansion) ==="
${CUDA_PATH}/bin/nvcc -O3 -std=c++17 \
    ${GPU_ARCH} \
    ../PRG_GPU_Gen_BMT.cu -o pcg_online_gpu \
    -I${CUDA_PATH}/include \
    -L${CUDA_PATH}/lib64 \
    -lcudart
echo "✓ pcg_online_gpu built"
echo ""

# ============================================================
# 2. 编译 Offline 阶段的 CUDA kernels
# ============================================================
echo "=== [2/3] Compiling CUDA kernels for Offline Phase ==="
${CUDA_PATH}/bin/nvcc -c -O3 -std=c++17 \
    ${GPU_ARCH} \
    ../gpu_kernels.cu -o gpu_kernels.o \
    -I${CUDA_PATH}/include
echo "✓ gpu_kernels.o built"
echo ""

# ============================================================
# 3. 编译 Offline 阶段主程序（多通道 GPU 版本）
# ============================================================
echo "=== [3/3] Compiling Offline Phase (Multi-channel GPU) ==="
mpicxx ${CXXFLAGS} \
    ../PCG_gpu.cpp gpu_kernels.o -o pcg_offline_gpu \
    ${LDFLAGS} ${LIBS}
echo "✓ pcg_offline_gpu built"
echo ""

# ============================================================
# 可选：编译 CPU 版本用于对比
# ============================================================
if [ -f "../PCG_multibit_parallel.cpp" ]; then
    echo "=== [Optional] Compiling CPU version for comparison ==="
    mpicxx ${CXXFLAGS} \
        ../PCG_multibit_parallel.cpp -o pcg_offline_cpu \
        ${LDFLAGS} ${LIBS}
    echo "✓ pcg_offline_cpu built"
    echo ""
fi

# ============================================================
# 完成
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Build Successful!                                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Binaries in ./build/                                        ║"
echo "║                                                              ║"
echo "║  pcg_offline_gpu  - GPU-accelerated offline phase            ║"
echo "║  pcg_online_gpu   - GPU PRG expansion (standalone)           ║"
echo "║  pcg_offline_cpu  - CPU version (comparison)                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Usage:"
echo ""
echo "  # Step 1: Generate corrections (offline phase)"
echo "  mpirun -np 2 --oversubscribe --mca btl vader,self \\"
echo "    ./build/pcg_offline_gpu --num_triples 10000000 --bits 64 \\"
echo "    --output build/offline_party"
echo ""
echo "  # Step 2: Expand to full BMT (online phase)"
echo "  ./build/pcg_online_gpu build/offline_party0.bin build/offline_party1.bin"
echo ""