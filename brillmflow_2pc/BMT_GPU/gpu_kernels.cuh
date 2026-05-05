// gpu_kernels.cuh - GPU 加速的 BMT 生成
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

using u64 = uint64_t;
using u32 = uint32_t;
using u8  = uint8_t;

// CUDA 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// GPU 上下文
struct GPUContext {
    // 设备指针
    u64* d_a;           // a values
    u64* d_b;           // b values
    u64* d_randoms;     // random values for OT
    u64* d_out;         // output (corrections or OT results)
    u8*  d_choices;     // choice bits
    u64* d_msgs;        // OT message pairs (存储为 u64, 每对 2 个)
    u64* d_sum_r;       // sum of randoms per triple
    
    // 主机 pinned memory (加速传输)
    u64* h_a;
    u64* h_b;
    u64* h_randoms;
    u64* h_out;
    u8*  h_choices;
    u64* h_msgs;
    u64* h_sum_r;
    
    size_t max_batch_size;
    int k_bits;
    
    cudaStream_t stream;
};

// 初始化 GPU 上下文
GPUContext* gpu_init(size_t max_batch_size, int k_bits);

// 释放 GPU 上下文
void gpu_cleanup(GPUContext* ctx);

// GPU 生成 a, b (ChaCha20)
void gpu_gen_ab(GPUContext* ctx, 
                u64 seed_hi, u64 seed_lo, 
                u64 start_idx, size_t count);

// GPU 构建 choice bits (Receiver 侧)
// offset: d_a 数组中的起始偏移
void gpu_build_choices(GPUContext* ctx, size_t count, size_t offset = 0);

// GPU 构建 OT 消息对 (Sender 侧)
// randoms 已经在 h_randoms 中准备好
// offset: d_b 数组中的起始偏移
void gpu_build_ot_messages(GPUContext* ctx, size_t count, size_t offset, u64 mask);

// GPU 聚合 blocks (Receiver 侧)
// recv_blocks 是从 OT 收到的 blocks (每个 16 bytes, 但我们只用低 8 bytes XOR 高 8 bytes)
void gpu_aggregate_blocks(GPUContext* ctx, const u64* recv_data, size_t count, u64 mask);

// GPU 计算 sender 的输出 (-sum_r)
void gpu_compute_sender_out(GPUContext* ctx, size_t count, u64 mask);

// 同步并获取结果
void gpu_sync(GPUContext* ctx);
