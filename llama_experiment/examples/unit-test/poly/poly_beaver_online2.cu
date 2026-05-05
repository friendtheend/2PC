// poly_beaver_online2.cu
// 2-party MPC online benchmark for polynomial activation:
//    y = a*x^2 + b*x + c
// where x is secret-shared (uint64 ring), x^2 is computed via Beaver elemwise multiply.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_86 poly_beaver_online2.cu \
//        -I"${CONDA_PREFIX}/include" \
//        -L"${CONDA_PREFIX}/lib" -lmpi \
//        -Xlinker --allow-shlib-undefined \
//        -Xlinker -rpath -Xlinker "${CONDA_PREFIX}/lib" \
//        -o poly_beaver_online2
//
// Run:
/*
# ViT-base, batch=1
mpirun -np 2 \
  --bind-to core --map-by core \
  --mca pml ob1 --mca btl vader,self \
  ./poly_beaver_online2 --rows 197 --cols 3072 --iters 10

# LLaMA-7B, batch=1
mpirun -np 2 \
  --bind-to core --map-by core \
  --mca pml ob1 --mca btl vader,self \
  ./poly_beaver_online2 --rows 8 --cols 11008 --iters 10
*/

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <random>

using u64 = uint64_t;

#define CUDA_CHECK(call) do {                                  \
  cudaError_t err = (call);                                    \
  if (err != cudaSuccess) {                                    \
    fprintf(stderr, "[CUDA] %s:%d: %s\n",                      \
            __FILE__, __LINE__, cudaGetErrorString(err));      \
    std::exit(1);                                              \
  }                                                            \
} while (0)

static int g_rows = 128;
static int g_cols = 3072;
static int g_iters = 10;
static int g_bits = 64;
static u64 g_mask = ~0ULL;

// ------------------------ Kernels ------------------------

// d = x - a
__global__ void k_sub_mod(const u64* __restrict__ x,
                          const u64* __restrict__ a,
                          u64* __restrict__ out,
                          size_t n,
                          u64 mask) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = tid; i < n; i += stride) {
    out[i] = (x[i] - a[i]) & mask;
  }
}

// Beaver elemwise multiply combine:
// z = C + d_open*B + A*e_open + (party0 ? d_open*e_open : 0)
__global__ void k_beaver_combine_elemmul(const u64* __restrict__ C,
                                        const u64* __restrict__ A,
                                        const u64* __restrict__ B,
                                        const u64* __restrict__ d_open,
                                        const u64* __restrict__ e_open,
                                        u64* __restrict__ z_share,
                                        size_t n,
                                        int is_party0,
                                        u64 mask) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  for (size_t i = tid; i < n; i += stride) {
    u64 z = C[i];

    // d_open * B
    __uint128_t p1 = (__uint128_t)d_open[i] * (__uint128_t)B[i];
    z = (z + ((u64)p1 & mask)) & mask;

    // A * e_open
    __uint128_t p2 = (__uint128_t)A[i] * (__uint128_t)e_open[i];
    z = (z + ((u64)p2 & mask)) & mask;

    // party0 adds d_open * e_open
    if (is_party0) {
      __uint128_t p3 = (__uint128_t)d_open[i] * (__uint128_t)e_open[i];
      z = (z + ((u64)p3 & mask)) & mask;
    }

    z_share[i] = z;
  }
}

// y = a*x2 + b*x + c  (public coefficients)
// 优化：ax² 和 bx 并行计算（无数据依赖）
__global__ void k_poly_eval(const u64* __restrict__ x_share,
                           const u64* __restrict__ x2_share,
                           u64* __restrict__ y_share,
                           size_t n,
                           u64 a, u64 b, u64 c,
                           int is_party0,
                           u64 mask) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  for (size_t i = tid; i < n; i += stride) {
    // ax² 和 bx 并行计算（两个独立的乘法，无依赖）
    __uint128_t p_ax2 = (__uint128_t)a * (__uint128_t)x2_share[i];  // ax²
    __uint128_t p_bx  = (__uint128_t)b * (__uint128_t)x_share[i];   // bx

    u64 ax2 = (u64)p_ax2 & mask;
    u64 bx  = (u64)p_bx  & mask;

    // 合并: y = ax² + bx + c
    // c 只有 party0 加（保证 MPC 正确性）
    u64 y = (ax2 + bx) & mask;
    if (is_party0) {
      y = (y + c) & mask;
    }

    y_share[i] = y;
  }
}

// ------------------------ Helpers ------------------------

static u64 parse_u64(const char* s) {
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 0);
  return (u64)v;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  if (world != 2) {
    if (rank == 0) fprintf(stderr, "Need exactly 2 MPI processes.\n");
    MPI_Finalize();
    return 1;
  }

  // args
  u64 a = 3;   // arbitrary public coeffs
  u64 b = 5;
  u64 c = 7;

  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--rows")) g_rows = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--cols")) g_cols = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters")) g_iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--bits")) g_bits = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--a")) a = parse_u64(argv[++i]);
    else if (!std::strcmp(argv[i], "--b")) b = parse_u64(argv[++i]);
    else if (!std::strcmp(argv[i], "--c")) c = parse_u64(argv[++i]);
  }

  g_mask = (g_bits >= 64) ? ~0ULL : ((1ULL << g_bits) - 1);

  const size_t n = (size_t)g_rows * (size_t)g_cols;
  const int is_party0 = (rank == 0) ? 1 : 0;

  // pick GPU
  CUDA_CHECK(cudaSetDevice(0));

  // host buffers (pinned for faster D2H/H2D)
  u64 *h_d = nullptr, *h_e = nullptr, *h_d_open = nullptr, *h_e_open = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_d,      n * sizeof(u64)));
  CUDA_CHECK(cudaMallocHost(&h_e,      n * sizeof(u64)));
  CUDA_CHECK(cudaMallocHost(&h_d_open, n * sizeof(u64)));
  CUDA_CHECK(cudaMallocHost(&h_e_open, n * sizeof(u64)));

  // device buffers
  u64 *d_x = nullptr, *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  u64 *d_d = nullptr, *d_e = nullptr;
  u64 *d_d_open = nullptr, *d_e_open = nullptr;
  u64 *d_x2 = nullptr, *d_y = nullptr;

  CUDA_CHECK(cudaMalloc(&d_x,      n * sizeof(u64)));
  CUDA_CHECK(cudaMalloc(&d_A,      n * sizeof(u64)));
  CUDA_CHECK(cudaMalloc(&d_B,      n * sizeof(u64)));
  CUDA_CHECK(cudaMalloc(&d_C,      n * sizeof(u64)));
  CUDA_CHECK(cudaMalloc(&d_d,      n * sizeof(u64)));
  CUDA_CHECK(cudaMalloc(&d_e,      n * sizeof(u64)));
  CUDA_CHECK(cudaMalloc(&d_d_open, n * sizeof(u64)));
  CUDA_CHECK(cudaMalloc(&d_e_open, n * sizeof(u64)));
  CUDA_CHECK(cudaMalloc(&d_x2,     n * sizeof(u64)));
  CUDA_CHECK(cudaMalloc(&d_y,      n * sizeof(u64)));

  // initialize x share + offline triple shares (outside timed loop)
  std::mt19937_64 rng(12345 + rank * 999);
  std::vector<u64> hx(n), hA(n), hB(n), hC(n);
  for (size_t i = 0; i < n; i++) {
    hx[i] = rng() & g_mask;
    hA[i] = rng() & g_mask;
    hB[i] = rng() & g_mask;
    hC[i] = rng() & g_mask;  // not necessarily A*B, correctness not needed
  }

  CUDA_CHECK(cudaMemcpy(d_x, hx.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_A, hA.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, hB.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, hC.data(), n * sizeof(u64), cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = 256;

  // warmup
  k_sub_mod<<<blocks, threads>>>(d_x, d_A, d_d, n, g_mask);
  k_sub_mod<<<blocks, threads>>>(d_x, d_B, d_e, n, g_mask);
  CUDA_CHECK(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // timing
  double total_ms_sum = 0.0;
  double comm_ms_sum  = 0.0;

  // bytes/rounds: open(d,e) 算作 1 轮，发送 2*n*8 bytes
  const double bytes_per_iter = 2.0 * (double)n * 8.0;
  const double rounds_per_iter = 1.0;

  for (int it = 0; it < g_iters; it++) {
    MPI_Barrier(MPI_COMM_WORLD);

    auto t_iter0 = std::chrono::high_resolution_clock::now();

    // (1) d = x - A, e = x - B  (GPU)
    k_sub_mod<<<blocks, threads>>>(d_x, d_A, d_d, n, g_mask);
    k_sub_mod<<<blocks, threads>>>(d_x, d_B, d_e, n, g_mask);

    // (2) D2H
    CUDA_CHECK(cudaMemcpy(h_d, d_d, n * sizeof(u64), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e, d_e, n * sizeof(u64), cudaMemcpyDeviceToHost));

    // (3) open via MPI_Allreduce (SUM mod 2^k)
    auto t_comm0 = std::chrono::high_resolution_clock::now();
    MPI_Allreduce(h_d, h_d_open, (int)n, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(h_e, h_e_open, (int)n, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    auto t_comm1 = std::chrono::high_resolution_clock::now();

    double comm_ms = std::chrono::duration<double, std::milli>(t_comm1 - t_comm0).count();
    comm_ms_sum += comm_ms;

    // (4) H2D
    CUDA_CHECK(cudaMemcpy(d_d_open, h_d_open, n * sizeof(u64), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_open, h_e_open, n * sizeof(u64), cudaMemcpyHostToDevice));

    // (5) Beaver combine => x2 share
    k_beaver_combine_elemmul<<<blocks, threads>>>(
        d_C, d_A, d_B, d_d_open, d_e_open, d_x2, n, is_party0, g_mask);

    // (6) Poly eval: y = a*x2 + b*x + c （并行优化版本）
    k_poly_eval<<<blocks, threads>>>(d_x, d_x2, d_y, n, a, b, c, is_party0, g_mask);

    CUDA_CHECK(cudaDeviceSynchronize());

    auto t_iter1 = std::chrono::high_resolution_clock::now();
    double iter_ms = std::chrono::duration<double, std::milli>(t_iter1 - t_iter0).count();
    total_ms_sum += iter_ms;
  }

  if (rank == 0) {
    double avg_s = (total_ms_sum / (double)g_iters) / 1000.0;
    double avg_bytes_MB = (bytes_per_iter / 1048576.0);
    double avg_rounds = rounds_per_iter;

    // CrypTen-like one-liner
    printf("(%d, %d) time: %.4fs, bytes: %.0f MB, rounds: %.0f\n",
           g_rows, g_cols, avg_s, avg_bytes_MB, avg_rounds);

    // 可选：详细分解
    printf("[breakdown] avg comm: %.2f ms / iter\n", comm_ms_sum / (double)g_iters);
    printf("[breakdown] avg compute+copy: %.2f ms / iter\n",
           (total_ms_sum - comm_ms_sum) / (double)g_iters);
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_d));
  CUDA_CHECK(cudaFree(d_e));
  CUDA_CHECK(cudaFree(d_d_open));
  CUDA_CHECK(cudaFree(d_e_open));
  CUDA_CHECK(cudaFree(d_x2));
  CUDA_CHECK(cudaFree(d_y));

  CUDA_CHECK(cudaFreeHost(h_d));
  CUDA_CHECK(cudaFreeHost(h_e));
  CUDA_CHECK(cudaFreeHost(h_d_open));
  CUDA_CHECK(cudaFreeHost(h_e_open));

  MPI_Finalize();
  return 0;
}
