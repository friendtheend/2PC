// gpu_kernels.cu - GPU 加速的 BMT 生成实现
#include "gpu_kernels.cuh"
#include <cstdio>
#include <cstring>

// ============================================================
// ChaCha20 GPU 实现
// ============================================================
__device__ __forceinline__ void qr_gpu(u32 &a, u32 &b, u32 &c, u32 &d) {
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8)  | (d >> 24);
    c += d; b ^= c; b = (b << 7)  | (b >> 25);
}

__device__ void chacha20_block_gpu(u64 seed_hi, u64 seed_lo, u64 idx, u32 out[16]) {
    u32 s0 = (u32)seed_lo, s1 = (u32)(seed_lo >> 32);
    u32 s2 = (u32)seed_hi, s3 = (u32)(seed_hi >> 32);
    
    u32 key[8] = { 
        s0 ^ 0xA5A5A5A5, s1 ^ 0x3C6EF372, s2 ^ 0x9E3779B9, s3 ^ 0xC3EFE9DB,
        s0 ^ s2, s1 ^ s3, s0 ^ s3, s1 ^ s2 
    };
    u32 nonce[3] = { 0xDEADBEEF, 0xFEEDFACE, 0x12345678 };
    
    // 初始状态
    u32 s[16] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
        (u32)idx, nonce[0], nonce[1], nonce[2]
    };
    
    // 复制到输出
    #pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = s[i];
    
    // 20 轮 (10 次双轮)
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        qr_gpu(out[0], out[4], out[8],  out[12]);
        qr_gpu(out[1], out[5], out[9],  out[13]);
        qr_gpu(out[2], out[6], out[10], out[14]);
        qr_gpu(out[3], out[7], out[11], out[15]);
        qr_gpu(out[0], out[5], out[10], out[15]);
        qr_gpu(out[1], out[6], out[11], out[12]);
        qr_gpu(out[2], out[7], out[8],  out[13]);
        qr_gpu(out[3], out[4], out[9],  out[14]);
    }
    
    // 加上初始状态
    #pragma unroll
    for (int i = 0; i < 16; ++i) out[i] += s[i];
}

// ============================================================
// Kernel: 生成 a, b
// ============================================================
__global__ void gen_ab_kernel(
    u64 seed_hi, u64 seed_lo,
    u64 start_idx,
    u64* __restrict__ a_out,
    u64* __restrict__ b_out,
    size_t count,
    int k_bits)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    
    u32 out[16];
    chacha20_block_gpu(seed_hi, seed_lo, start_idx + i, out);
    
    u64 mask = (k_bits >= 64) ? ~0ULL : ((1ULL << k_bits) - 1);
    a_out[i] = (((u64)out[0] << 32) | out[1]) & mask;
    b_out[i] = (((u64)out[2] << 32) | out[3]) & mask;
}

// ============================================================
// Kernel: 构建 choice bits
// ============================================================
__global__ void build_choices_kernel(
    const u64* __restrict__ x_vec,
    u8* __restrict__ choices,
    size_t count,
    int k_bits,
    u64 mask)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = count * k_bits;
    if (idx >= total) return;
    
    size_t t = idx / k_bits;
    int b = idx % k_bits;
    
    u64 x = x_vec[t] & mask;
    choices[idx] = (x >> b) & 1;
}

// ============================================================
// Kernel: 构建 OT 消息对 (Sender)
// 输出格式: msgs[t * k * 2 + b * 2 + 0] = r
//          msgs[t * k * 2 + b * 2 + 1] = r + y * 2^b
// ============================================================
__global__ void build_ot_messages_kernel(
    const u64* __restrict__ y_vec,
    const u64* __restrict__ randoms,
    u64* __restrict__ msgs,
    u64* __restrict__ sum_r,
    size_t count,
    int k_bits,
    u64 mask)
{
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    
    u64 y = y_vec[t] & mask;
    u64 sum = 0;
    
    for (int b = 0; b < k_bits; ++b) {
        u64 r = randoms[t * k_bits + b] & mask;
        sum += r;
        
        size_t msg_idx = t * k_bits * 2 + b * 2;
        msgs[msg_idx + 0] = r;
        msgs[msg_idx + 1] = (r + (y << b)) & mask;
    }
    
    sum_r[t] = sum;
}

// ============================================================
// Kernel: 聚合 blocks (Receiver)
// recv_data: 每个 OT 返回一个 u64 (block 的 lo ^ hi)
// ============================================================
__global__ void aggregate_blocks_kernel(
    const u64* __restrict__ recv_data,
    u64* __restrict__ out,
    size_t count,
    int k_bits,
    u64 mask)
{
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    
    // 使用 128-bit 累加避免溢出
    unsigned long long acc_lo = 0;
    unsigned long long acc_hi = 0;
    
    for (int b = 0; b < k_bits; ++b) {
        u64 val = recv_data[t * k_bits + b];
        acc_lo += val;
        if (acc_lo < val) acc_hi++;  // 进位
    }
    
    out[t] = acc_lo & mask;
}

// ============================================================
// Kernel: 计算 sender 输出 (-sum_r)
// ============================================================
__global__ void compute_sender_out_kernel(
    const u64* __restrict__ sum_r,
    u64* __restrict__ out,
    size_t count,
    u64 mask)
{
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    
    out[t] = (0ULL - sum_r[t]) & mask;
}

// ============================================================
// Host 函数实现
// ============================================================

GPUContext* gpu_init(size_t max_batch_size, int k_bits) {
    GPUContext* ctx = new GPUContext();
    ctx->max_batch_size = max_batch_size;
    ctx->k_bits = k_bits;
    
    size_t num_ots = max_batch_size * k_bits;
    
    // 创建 stream
    CUDA_CHECK(cudaStreamCreate(&ctx->stream));
    
    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&ctx->d_a, max_batch_size * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&ctx->d_b, max_batch_size * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&ctx->d_randoms, num_ots * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&ctx->d_out, max_batch_size * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&ctx->d_choices, num_ots * sizeof(u8)));
    CUDA_CHECK(cudaMalloc(&ctx->d_msgs, num_ots * 2 * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&ctx->d_sum_r, max_batch_size * sizeof(u64)));
    
    // 分配 pinned host memory
    CUDA_CHECK(cudaMallocHost(&ctx->h_a, max_batch_size * sizeof(u64)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_b, max_batch_size * sizeof(u64)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_randoms, num_ots * sizeof(u64)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_out, max_batch_size * sizeof(u64)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_choices, num_ots * sizeof(u8)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_msgs, num_ots * 2 * sizeof(u64)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_sum_r, max_batch_size * sizeof(u64)));
    
    return ctx;
}

void gpu_cleanup(GPUContext* ctx) {
    if (!ctx) return;
    
    cudaStreamDestroy(ctx->stream);
    
    cudaFree(ctx->d_a);
    cudaFree(ctx->d_b);
    cudaFree(ctx->d_randoms);
    cudaFree(ctx->d_out);
    cudaFree(ctx->d_choices);
    cudaFree(ctx->d_msgs);
    cudaFree(ctx->d_sum_r);
    
    cudaFreeHost(ctx->h_a);
    cudaFreeHost(ctx->h_b);
    cudaFreeHost(ctx->h_randoms);
    cudaFreeHost(ctx->h_out);
    cudaFreeHost(ctx->h_choices);
    cudaFreeHost(ctx->h_msgs);
    cudaFreeHost(ctx->h_sum_r);
    
    delete ctx;
}

void gpu_gen_ab(GPUContext* ctx, u64 seed_hi, u64 seed_lo, u64 start_idx, size_t count) {
    const int block_size = 256;
    const int num_blocks = (count + block_size - 1) / block_size;
    
    gen_ab_kernel<<<num_blocks, block_size, 0, ctx->stream>>>(
        seed_hi, seed_lo, start_idx,
        ctx->d_a, ctx->d_b, count, ctx->k_bits
    );
    
    // 异步复制回 host
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_a, ctx->d_a, count * sizeof(u64), 
                               cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_b, ctx->d_b, count * sizeof(u64), 
                               cudaMemcpyDeviceToHost, ctx->stream));
}

void gpu_build_choices(GPUContext* ctx, size_t count, size_t offset) {
    int k = ctx->k_bits;
    u64 mask = (k >= 64) ? ~0ULL : ((1ULL << k) - 1);
    
    size_t total = count * k;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    build_choices_kernel<<<num_blocks, block_size, 0, ctx->stream>>>(
        ctx->d_a + offset, ctx->d_choices, count, k, mask
    );
    
    // 异步复制回 host
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_choices, ctx->d_choices, total * sizeof(u8),
                               cudaMemcpyDeviceToHost, ctx->stream));
}

void gpu_build_ot_messages(GPUContext* ctx, size_t count, size_t offset, u64 mask) {
    int k = ctx->k_bits;
    size_t num_ots = count * k;
    
    // 先把 randoms 复制到 GPU
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_randoms, ctx->h_randoms, num_ots * sizeof(u64),
                               cudaMemcpyHostToDevice, ctx->stream));
    
    const int block_size = 256;
    const int num_blocks = (count + block_size - 1) / block_size;
    
    build_ot_messages_kernel<<<num_blocks, block_size, 0, ctx->stream>>>(
        ctx->d_b + offset, ctx->d_randoms, ctx->d_msgs, ctx->d_sum_r,
        count, k, mask
    );
    
    // 异步复制回 host
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_msgs, ctx->d_msgs, num_ots * 2 * sizeof(u64),
                               cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_sum_r, ctx->d_sum_r, count * sizeof(u64),
                               cudaMemcpyDeviceToHost, ctx->stream));
}

void gpu_aggregate_blocks(GPUContext* ctx, const u64* recv_data, size_t count, u64 mask) {
    int k = ctx->k_bits;
    size_t num_ots = count * k;
    
    // 复制接收到的数据到 GPU
    // 注意：这里 recv_data 可能不是 pinned memory，所以用同步复制
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_randoms, recv_data, num_ots * sizeof(u64),
                               cudaMemcpyHostToDevice, ctx->stream));
    
    const int block_size = 256;
    const int num_blocks = (count + block_size - 1) / block_size;
    
    aggregate_blocks_kernel<<<num_blocks, block_size, 0, ctx->stream>>>(
        ctx->d_randoms, ctx->d_out, count, k, mask
    );
    
    // 异步复制回 host
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_out, ctx->d_out, count * sizeof(u64),
                               cudaMemcpyDeviceToHost, ctx->stream));
}

void gpu_compute_sender_out(GPUContext* ctx, size_t count, u64 mask) {
    const int block_size = 256;
    const int num_blocks = (count + block_size - 1) / block_size;
    
    compute_sender_out_kernel<<<num_blocks, block_size, 0, ctx->stream>>>(
        ctx->d_sum_r, ctx->d_out, count, mask
    );
    
    // 异步复制回 host
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_out, ctx->d_out, count * sizeof(u64),
                               cudaMemcpyDeviceToHost, ctx->stream));
}

void gpu_sync(GPUContext* ctx) {
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}
