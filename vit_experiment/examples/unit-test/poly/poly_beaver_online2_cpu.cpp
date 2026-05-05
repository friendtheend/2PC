// poly_beaver_online2_cpu.cpp
// 2-party MPC online benchmark (CPU-only) for polynomial activation:
//    y = a*x^2 + b*x + c
// x secret-shared in ring 2^k, x^2 via Beaver elemwise multiply.
//
// Build:
//   mpicxx -O3 -std=c++17 poly_beaver_online2_cpu.cpp -o poly_beaver_online2_cpu
// Optional OpenMP:
//   mpicxx -O3 -std=c++17 -fopenmp poly_beaver_online2_cpu.cpp -o poly_beaver_online2_cpu
//
// Run:
/*
# ViT-base, batch=1
mpirun -np 2 \
  --bind-to core --map-by core \
  --mca pml ob1 --mca btl vader,self \
  ./poly_beaver_online2_cpu --rows 197 --cols 3072 --iters 10

# LLaMA-7B, batch=1
mpirun -np 2 \
  --bind-to core --map-by core \
  --mca pml ob1 --mca btl vader,self \
  ./poly_beaver_online2_cpu --rows 8 --cols 11008 --iters 10
*/

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <random>

using u64 = uint64_t;

static int g_rows = 128;
static int g_cols = 3072;
static int g_iters = 10;
static int g_bits = 64;
static u64 g_mask = ~0ULL;

static u64 parse_u64(const char* s) {
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 0);
  return (u64)v;
}

// out = (x - a) mod 2^k
static void sub_mod_cpu(const u64* x, const u64* a, u64* out, size_t n, u64 mask) {
  #pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    out[i] = (x[i] - a[i]) & mask;
  }
}

// Beaver combine elemwise:
// z = C + d_open*B + A*e_open + (party0 ? d_open*e_open : 0) mod 2^k
static void beaver_combine_elemmul_cpu(
    const u64* C,
    const u64* A,
    const u64* B,
    const u64* d_open,
    const u64* e_open,
    u64* z_share,
    size_t n,
    int is_party0,
    u64 mask) {

  #pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    u64 z = C[i];

    unsigned __int128 p1 = (unsigned __int128)d_open[i] * (unsigned __int128)B[i];
    z = (z + ((u64)p1 & mask)) & mask;

    unsigned __int128 p2 = (unsigned __int128)A[i] * (unsigned __int128)e_open[i];
    z = (z + ((u64)p2 & mask)) & mask;

    if (is_party0) {
      unsigned __int128 p3 = (unsigned __int128)d_open[i] * (unsigned __int128)e_open[i];
      z = (z + ((u64)p3 & mask)) & mask;
    }

    z_share[i] = z;
  }
}

// y = a*x2 + b*x + c (c only added by party0) mod 2^k
static void poly_eval_cpu(
    const u64* x_share,
    const u64* x2_share,
    u64* y_share,
    size_t n,
    u64 a, u64 b, u64 c,
    int is_party0,
    u64 mask) {

  #pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    unsigned __int128 p_ax2 = (unsigned __int128)a * (unsigned __int128)x2_share[i];
    unsigned __int128 p_bx  = (unsigned __int128)b * (unsigned __int128)x_share[i];

    u64 ax2 = (u64)p_ax2 & mask;
    u64 bx  = (u64)p_bx  & mask;

    u64 y = (ax2 + bx) & mask;
    if (is_party0) {
      y = (y + c) & mask;
    }
    y_share[i] = y;
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  if (world != 2) {
    if (rank == 0) std::fprintf(stderr, "Need exactly 2 MPI processes.\n");
    MPI_Finalize();
    return 1;
  }

  u64 a = 3;
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

  // CPU buffers
  std::vector<u64> x(n), A(n), B(n), C(n);
  std::vector<u64> d(n), e(n);
  std::vector<u64> d_open(n), e_open(n);
  std::vector<u64> x2(n), y(n);

  // init (outside timed loop)
  std::mt19937_64 rng(12345 + rank * 999);
  for (size_t i = 0; i < n; i++) {
    x[i] = rng() & g_mask;
    A[i] = rng() & g_mask;
    B[i] = rng() & g_mask;
    C[i] = rng() & g_mask; // benchmark only (not necessarily A*B)
  }

  // warmup
  sub_mod_cpu(x.data(), A.data(), d.data(), n, g_mask);
  sub_mod_cpu(x.data(), B.data(), e.data(), n, g_mask);
  MPI_Barrier(MPI_COMM_WORLD);

  double total_ms_sum = 0.0;
  double comm_ms_sum  = 0.0;

  // bytes/rounds: open(d,e) counted as 1 round, total bytes = 2*n*8
  const double bytes_per_iter  = 2.0 * (double)n * 8.0;
  const double rounds_per_iter = 1.0;

  // 注意：MPI_Allreduce count 参数是 int，如果 n 很大（>2^31-1）要分块；你现在维度一般不会到那。
  if (n > (size_t)INT32_MAX) {
    if (rank == 0) {
      std::fprintf(stderr, "n too large for MPI_Allreduce count int; need chunking.\n");
    }
    MPI_Finalize();
    return 1;
  }

  for (int it = 0; it < g_iters; it++) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_iter0 = std::chrono::high_resolution_clock::now();

    // (1) d = x - A, e = x - B  (CPU)
    sub_mod_cpu(x.data(), A.data(), d.data(), n, g_mask);
    sub_mod_cpu(x.data(), B.data(), e.data(), n, g_mask);

    // (2) open via MPI_Allreduce (SUM mod 2^k, mask later is fine)
    auto t_comm0 = std::chrono::high_resolution_clock::now();
    MPI_Allreduce(d.data(), d_open.data(), (int)n, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(e.data(), e_open.data(), (int)n, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    auto t_comm1 = std::chrono::high_resolution_clock::now();

    double comm_ms = std::chrono::duration<double, std::milli>(t_comm1 - t_comm0).count();
    comm_ms_sum += comm_ms;

    // (3) Beaver combine => x2 share
    beaver_combine_elemmul_cpu(
        C.data(), A.data(), B.data(),
        d_open.data(), e_open.data(),
        x2.data(), n, is_party0, g_mask);

    // (4) Poly eval
    poly_eval_cpu(x.data(), x2.data(), y.data(), n, a, b, c, is_party0, g_mask);

    auto t_iter1 = std::chrono::high_resolution_clock::now();
    double iter_ms = std::chrono::duration<double, std::milli>(t_iter1 - t_iter0).count();
    total_ms_sum += iter_ms;
  }

  if (rank == 0) {
    double avg_s = (total_ms_sum / (double)g_iters) / 1000.0;
    double avg_bytes_MB = (bytes_per_iter / 1048576.0);
    double avg_rounds = rounds_per_iter;

    std::printf("(%d, %d) time: %.4fs, bytes: %.0f MB, rounds: %.0f\n",
                g_rows, g_cols, avg_s, avg_bytes_MB, avg_rounds);

    // 可选 breakdown
     std::printf("[breakdown] avg comm: %.2f ms / iter\n", comm_ms_sum / (double)g_iters);
     std::printf("[breakdown] avg compute: %.2f ms / iter\n",
                 (total_ms_sum - comm_ms_sum) / (double)g_iters);
  }

  MPI_Finalize();
  return 0;
}
