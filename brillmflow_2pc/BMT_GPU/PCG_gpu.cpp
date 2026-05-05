// PCG_gpu_multichannel.cpp - GPU 加速的 BMT Offline Phase (多通道版本)
//
// 特点:
//   - GPU 加速 ChaCha20 生成 a, b
//   - GPU 加速 OT 消息构建和聚合
//   - 多通道并行 OT 通信 (8 通道默认)
//
// 运行:
//   mpirun -np 2 --oversubscribe --mca btl vader,self \
//     ./pcg_offline_gpu --num_triples 10000000 --bits 64 --channels 8 \
//     --output offline_party

// ============================================================
// libsodium noclamp (SimplestOT 需要)
// ============================================================
#include <sodium/crypto_scalarmult_ed25519.h>

extern "C" int crypto_scalarmult_noclamp(
    unsigned char* q, const unsigned char* n, const unsigned char* p) {
    return crypto_scalarmult_ed25519_noclamp(q, n, p);
}
extern "C" int crypto_scalarmult_base_noclamp(
    unsigned char* q, const unsigned char* n) {
    return crypto_scalarmult_ed25519_base_noclamp(q, n);
}

// ============================================================
// Includes
// ============================================================
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <mpi.h>
#include <omp.h>

#include <libOTe/TwoChooseOne/Iknp/IknpOtExtSender.h>
#include <libOTe/TwoChooseOne/Iknp/IknpOtExtReceiver.h>
#include <coproto/Socket/Socket.h>
#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Crypto/PRNG.h>
#include <macoro/sync_wait.h>
#include <macoro/task.h>

#include "gpu_kernels.cuh"

using namespace osuCrypto;
using u64 = uint64_t;
using u32 = uint32_t;
using u8  = uint8_t;

// ============================================================
// 配置
// ============================================================
static int g_k_bits = 64;
static size_t g_start_index = 0;
static int g_num_channels = 8;
static size_t g_max_ot_per_call = 4000000;  // 可通过参数调整

// ============================================================
// 统计
// ============================================================
struct Stats {
    std::atomic<double> gpu_gen_time{0};
    std::atomic<double> gpu_build_time{0};
    std::atomic<double> gpu_agg_time{0};
    std::atomic<double> ot_comm_time{0};
    std::atomic<double> file_time{0};
    std::atomic<size_t> ot_calls{0};
    std::atomic<size_t> total_ots{0};
    std::atomic<size_t> bytes_sent{0};
    std::atomic<size_t> bytes_recv{0};
};
static Stats g_stats;

// ============================================================
// block 转 u64
// ============================================================
static inline u64 blk2u64(const block& b) {
    u64 lo, hi;
    std::memcpy(&lo, &b, 8);
    std::memcpy(&hi, (const u8*)&b + 8, 8);
    return lo ^ hi;
}

// ============================================================
// MPI 多通道 Socket
// ============================================================
class MPIChannelSocket {
    int rank_, peer_, ch_;
    int stag_, rtag_;
    bool closed_ = false;
    std::mutex smtx_, rmtx_;
    std::vector<u8> rbuf_;
    size_t rpos_ = 0;

public:
    MPIChannelSocket(int rank, int channel) 
        : rank_(rank), peer_(1-rank), ch_(channel) {
        // 每个通道用不同的 tag 范围
        int base = 1000 + channel * 100;
        stag_ = (rank == 0) ? base : base + 50;
        rtag_ = (rank == 0) ? base + 50 : base;
    }
    
    ~MPIChannelSocket() { closed_ = true; }
    
    void close() { closed_ = true; }

    macoro::task<std::tuple<std::error_code, size_t>>
    send(std::span<const u8> d, macoro::stop_token) {
        if (closed_) {
            co_return std::make_tuple(std::make_error_code(std::errc::broken_pipe), size_t(0));
        }
        std::lock_guard<std::mutex> lk(smtx_);
        const u8* p = d.data();
        size_t rem = d.size(), tot = 0;
        while (rem > 0) {
            int c = (int)std::min<size_t>(rem, INT_MAX);
            MPI_Send((void*)p, c, MPI_BYTE, peer_, stag_, MPI_COMM_WORLD);
            p += c; rem -= c; tot += c;
        }
        g_stats.bytes_sent += tot;
        co_return std::make_tuple(std::error_code{}, tot);
    }

    void fill_rbuf() {
        MPI_Status st;
        MPI_Probe(peer_, rtag_, MPI_COMM_WORLD, &st);
        int cnt; MPI_Get_count(&st, MPI_BYTE, &cnt);
        if (cnt <= 0) return;
        size_t old = rbuf_.size();
        rbuf_.resize(old + cnt);
        MPI_Recv(rbuf_.data() + old, cnt, MPI_BYTE, peer_, rtag_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        g_stats.bytes_recv += cnt;
    }

    macoro::task<std::tuple<std::error_code, size_t>>
    recv(std::span<u8> d, macoro::stop_token) {
        if (closed_) {
            co_return std::make_tuple(std::make_error_code(std::errc::broken_pipe), size_t(0));
        }
        std::lock_guard<std::mutex> lk(rmtx_);
        size_t need = d.size(), got = 0;
        while (got < need) {
            if (rpos_ < rbuf_.size()) {
                size_t take = std::min(rbuf_.size() - rpos_, need - got);
                std::memcpy(d.data() + got, rbuf_.data() + rpos_, take);
                rpos_ += take; got += take;
            } else {
                rbuf_.clear(); rpos_ = 0;
                fill_rbuf();
            }
        }
        co_return std::make_tuple(std::error_code{}, got);
    }
};

// ============================================================
// 通道上下文 (每个通道独立的 GPU 缓冲区)
// ============================================================
struct ChannelContext {
    int channel_id;
    
    // GPU 缓冲区指针 (指向主 GPUContext 的偏移位置)
    u64* h_a;
    u64* h_b;
    u64* h_randoms;
    u64* h_out;
    u8*  h_choices;
    u64* h_msgs;
    u64* h_sum_r;
    
    // 当前处理的数据范围
    size_t start_idx;
    size_t count;
    
    // Socket
    std::unique_ptr<coproto::Socket> sock;
    
    // 结果
    std::vector<u64> cross1;
    std::vector<u64> cross2;
};

// ============================================================
// 扩展的 GPU 上下文 (支持多通道)
// ============================================================
struct MultiChannelGPUContext {
    GPUContext* gpu;
    int num_channels;
    size_t per_channel_size;
    
    // 每个通道的缓冲区
    std::vector<u64*> h_a_per_ch;
    std::vector<u64*> h_b_per_ch;
    std::vector<u64*> h_randoms_per_ch;
    std::vector<u64*> h_out_per_ch;
    std::vector<u8*>  h_choices_per_ch;
    std::vector<u64*> h_msgs_per_ch;
    std::vector<u64*> h_sum_r_per_ch;
};

MultiChannelGPUContext* mc_gpu_init(size_t total_batch_size, int k_bits, int num_channels) {
    auto* mc = new MultiChannelGPUContext();
    mc->num_channels = num_channels;
    mc->per_channel_size = (total_batch_size + num_channels - 1) / num_channels;
    
    // 初始化主 GPU 上下文
    mc->gpu = gpu_init(total_batch_size, k_bits);
    
    // 为每个通道分配独立的 host 缓冲区
    size_t max_ots_per_ch = mc->per_channel_size * k_bits;
    
    mc->h_a_per_ch.resize(num_channels);
    mc->h_b_per_ch.resize(num_channels);
    mc->h_randoms_per_ch.resize(num_channels);
    mc->h_out_per_ch.resize(num_channels);
    mc->h_choices_per_ch.resize(num_channels);
    mc->h_msgs_per_ch.resize(num_channels);
    mc->h_sum_r_per_ch.resize(num_channels);
    
    for (int ch = 0; ch < num_channels; ++ch) {
        cudaMallocHost(&mc->h_a_per_ch[ch], mc->per_channel_size * sizeof(u64));
        cudaMallocHost(&mc->h_b_per_ch[ch], mc->per_channel_size * sizeof(u64));
        cudaMallocHost(&mc->h_randoms_per_ch[ch], max_ots_per_ch * sizeof(u64));
        cudaMallocHost(&mc->h_out_per_ch[ch], mc->per_channel_size * sizeof(u64));
        cudaMallocHost(&mc->h_choices_per_ch[ch], max_ots_per_ch * sizeof(u8));
        cudaMallocHost(&mc->h_msgs_per_ch[ch], max_ots_per_ch * 2 * sizeof(u64));
        cudaMallocHost(&mc->h_sum_r_per_ch[ch], mc->per_channel_size * sizeof(u64));
    }
    
    return mc;
}

void mc_gpu_cleanup(MultiChannelGPUContext* mc) {
    for (int ch = 0; ch < mc->num_channels; ++ch) {
        cudaFreeHost(mc->h_a_per_ch[ch]);
        cudaFreeHost(mc->h_b_per_ch[ch]);
        cudaFreeHost(mc->h_randoms_per_ch[ch]);
        cudaFreeHost(mc->h_out_per_ch[ch]);
        cudaFreeHost(mc->h_choices_per_ch[ch]);
        cudaFreeHost(mc->h_msgs_per_ch[ch]);
        cudaFreeHost(mc->h_sum_r_per_ch[ch]);
    }
    gpu_cleanup(mc->gpu);
    delete mc;
}

// ============================================================
// Gilboa OT (单通道，使用通道专用缓冲区)
// ============================================================
static void gilboa_channel(
    ChannelContext& ctx,
    GPUContext* gpu,
    u64 ot_seed,
    int role,
    const u64* x_vec,  // receiver 用 a, sender 用 b
    size_t count,
    std::vector<u64>& out)
{
    const int k = g_k_bits;
    const u64 mask = (k >= 64) ? ~0ULL : ((1ULL << k) - 1);
    
    if (count == 0) { out.clear(); return; }
    
    const size_t max_triples = g_max_ot_per_call / k;
    const size_t num_chunks = (count + max_triples - 1) / max_triples;
    
    out.resize(count);
    
    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        size_t start = chunk * max_triples;
        size_t end = std::min(start + max_triples, count);
        size_t chunk_size = end - start;
        size_t num_ots = chunk_size * k;
        
        u64 chunk_seed = ot_seed + chunk * 999983ULL;
        
        if (role == 0) {
            // ============ Receiver ============
            
            // 构建 choice bits
            for (size_t t = 0; t < chunk_size; ++t) {
                u64 x = x_vec[start + t] & mask;
                for (int b = 0; b < k; ++b) {
                    ctx.h_choices[t * k + b] = (x >> b) & 1;
                }
            }
            
            // 转换为 BitVector
            BitVector choices(num_ots);
            for (size_t i = 0; i < num_ots; ++i) {
                choices[i] = ctx.h_choices[i];
            }
            
            // OT 接收
            auto t1 = std::chrono::high_resolution_clock::now();
            PRNG prng(block(chunk_seed ^ 0x67696C62ULL, ctx.channel_id));
            IknpOtExtReceiver recver;
            std::vector<block> recv(num_ots);
            macoro::sync_wait(recver.receiveChosen(choices, recv, prng, *ctx.sock));
            auto t2 = std::chrono::high_resolution_clock::now();
            g_stats.ot_comm_time += std::chrono::duration<double>(t2 - t1).count();
            
            // 聚合
            for (size_t t = 0; t < chunk_size; ++t) {
                u64 sum = 0;
                for (int b = 0; b < k; ++b) {
                    sum += blk2u64(recv[t * k + b]);
                }
                out[start + t] = sum & mask;
            }
            
        } else {
            // ============ Sender ============
            
            // 生成随机数并构建消息对
            PRNG prng(block(chunk_seed ^ 0x67696C62ULL ^ 0x5353454EULL, ctx.channel_id));
            std::vector<std::array<block, 2>> msgs(num_ots);
            std::vector<u64> sum_r(chunk_size, 0);
            
            for (size_t t = 0; t < chunk_size; ++t) {
                u64 y = x_vec[start + t] & mask;
                for (int b = 0; b < k; ++b) {
                    u64 r = prng.get<u64>() & mask;
                    sum_r[t] += r;
                    msgs[t * k + b][0] = block(r, 0);
                    msgs[t * k + b][1] = block((r + (y << b)) & mask, 0);
                }
            }
            
            // OT 发送
            auto t1 = std::chrono::high_resolution_clock::now();
            PRNG prng2(block(chunk_seed ^ 0x67696C62ULL ^ 0x5353454EULL, ctx.channel_id));
            IknpOtExtSender sender;
            macoro::sync_wait(sender.sendChosen(msgs, prng2, *ctx.sock));
            auto t2 = std::chrono::high_resolution_clock::now();
            g_stats.ot_comm_time += std::chrono::duration<double>(t2 - t1).count();
            
            // 输出 -sum_r
            for (size_t t = 0; t < chunk_size; ++t) {
                out[start + t] = (0ULL - sum_r[t]) & mask;
            }
        }
        
        g_stats.ot_calls++;
        g_stats.total_ots += num_ots;
    }
}

// ============================================================
// 通道线程函数
// ============================================================
static void channel_thread_func(
    ChannelContext& ctx,
    GPUContext* gpu,
    int party,
    u64 ot_seed_base,
    const u64* h_a,
    const u64* h_b)
{
    // Cross term 1: Party 0 是 receiver (用 a), Party 1 是 sender (用 b)
    {
        int role = (party == 0) ? 0 : 1;
        const u64* x_vec = (role == 0) ? (h_a + ctx.start_idx) : (h_b + ctx.start_idx);
        u64 seed1 = ot_seed_base + ctx.channel_id * 2;
        gilboa_channel(ctx, gpu, seed1, role, x_vec, ctx.count, ctx.cross1);
    }
    
    // Cross term 2: 角色交换
    {
        int role = (party == 0) ? 1 : 0;
        const u64* x_vec = (role == 0) ? (h_a + ctx.start_idx) : (h_b + ctx.start_idx);
        u64 seed2 = ot_seed_base + ctx.channel_id * 2 + 1;
        gilboa_channel(ctx, gpu, seed2, role, x_vec, ctx.count, ctx.cross2);
    }
}

// ============================================================
// 文件头
// ============================================================
struct Header {
    int party;
    size_t num_triples;
    u64 seed_hi, seed_lo;
    int k_bits;
    size_t start_index;
    
    void write(std::ofstream& f) const {
        f.write((const char*)&party, sizeof(party));
        f.write((const char*)&num_triples, sizeof(num_triples));
        f.write((const char*)&seed_hi, sizeof(seed_hi));
        f.write((const char*)&seed_lo, sizeof(seed_lo));
        f.write((const char*)&k_bits, sizeof(k_bits));
        f.write((const char*)&start_index, sizeof(start_index));
    }
    
    static size_t size() {
        return sizeof(int) + sizeof(size_t) + 2*sizeof(u64) + sizeof(int) + sizeof(size_t);
    }
};

// ============================================================
// 多通道 GPU 版本生成
// ============================================================
static void generate_multichannel_gpu(
    int party,
    size_t num_triples,
    u64 seed_hi, u64 seed_lo,
    u64 ot_seed,
    const std::string& output_file)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    
    std::ofstream ofs(output_file, std::ios::binary);
    if (!ofs) {
        std::fprintf(stderr, "[Party %d] Cannot open %s\n", party, output_file.c_str());
        return;
    }
    
    // 写入文件头
    Header hdr;
    hdr.party = party;
    hdr.num_triples = num_triples;
    hdr.seed_hi = seed_hi;
    hdr.seed_lo = seed_lo;
    hdr.k_bits = g_k_bits;
    hdr.start_index = g_start_index;
    hdr.write(ofs);
    
    // 初始化多通道 GPU 上下文
    // batch_size=2000000 测试最优
    const size_t BATCH_SIZE = 2000000;
    MultiChannelGPUContext* mc = mc_gpu_init(BATCH_SIZE, g_k_bits, g_num_channels);
    
    // 创建通道上下文
    std::vector<ChannelContext> channels(g_num_channels);
    for (int ch = 0; ch < g_num_channels; ++ch) {
        channels[ch].channel_id = ch;
        channels[ch].h_a = mc->h_a_per_ch[ch];
        channels[ch].h_b = mc->h_b_per_ch[ch];
        channels[ch].h_randoms = mc->h_randoms_per_ch[ch];
        channels[ch].h_out = mc->h_out_per_ch[ch];
        channels[ch].h_choices = mc->h_choices_per_ch[ch];
        channels[ch].h_msgs = mc->h_msgs_per_ch[ch];
        channels[ch].h_sum_r = mc->h_sum_r_per_ch[ch];
        
        // 创建 socket (每个通道一个)
        channels[ch].sock = std::make_unique<coproto::Socket>(
            coproto::make_socket_tag{}, 
            std::make_unique<MPIChannelSocket>(party, ch)
        );
    }
    
    // 结果缓冲区
    std::vector<u64> all_corrections(num_triples, 0);
    
    std::fprintf(stderr, "[Party %d] GPU mode, %d channels, batch_size=%zu\n", 
                 party, g_num_channels, BATCH_SIZE);
    
    const u64 mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);
    
    // 按批次处理
    for (size_t batch_start = 0; batch_start < num_triples; batch_start += BATCH_SIZE) {
        size_t batch_end = std::min(batch_start + BATCH_SIZE, num_triples);
        size_t batch_size = batch_end - batch_start;
        u64 global_start = g_start_index + batch_start;
        
        // GPU 生成 a, b
        auto t1 = std::chrono::high_resolution_clock::now();
        gpu_gen_ab(mc->gpu, seed_hi, seed_lo, global_start, batch_size);
        gpu_sync(mc->gpu);
        auto t2 = std::chrono::high_resolution_clock::now();
        g_stats.gpu_gen_time += std::chrono::duration<double>(t2 - t1).count();
        
        // 分配给各通道
        size_t per_ch = (batch_size + g_num_channels - 1) / g_num_channels;
        for (int ch = 0; ch < g_num_channels; ++ch) {
            size_t ch_start = ch * per_ch;
            size_t ch_end = std::min(ch_start + per_ch, batch_size);
            if (ch_start >= batch_size) {
                channels[ch].start_idx = 0;
                channels[ch].count = 0;
            } else {
                channels[ch].start_idx = ch_start;
                channels[ch].count = ch_end - ch_start;
            }
        }
        
        // 并行运行各通道的 OT (使用 OpenMP)
        u64 ot_seed_batch = ot_seed + batch_start * 2 * g_num_channels;
        
        #pragma omp parallel for num_threads(g_num_channels) schedule(static, 1)
        for (int ch = 0; ch < g_num_channels; ++ch) {
            if (channels[ch].count > 0) {
                channel_thread_func(
                    channels[ch],
                    mc->gpu,
                    party,
                    ot_seed_batch,
                    mc->gpu->h_a,
                    mc->gpu->h_b
                );
            }
        }
        
        // 收集结果
        for (int ch = 0; ch < g_num_channels; ++ch) {
            size_t ch_start = channels[ch].start_idx;
            size_t ch_count = channels[ch].count;
            for (size_t i = 0; i < ch_count; ++i) {
                u64 corr = (channels[ch].cross1[i] + channels[ch].cross2[i]) & mask;
                all_corrections[batch_start + ch_start + i] = corr;
            }
        }
        
        // 进度
        auto t_now = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t_now - t_start).count();
        std::fprintf(stderr, "\r[Party %d] %zu/%zu (%.1f%%), %.1f K/s    ",
                     party, batch_end, num_triples, 
                     100.0 * batch_end / num_triples, 
                     batch_end / sec / 1000.0);
    }
    
    // 写入文件
    auto t_file_start = std::chrono::high_resolution_clock::now();
    ofs.write((const char*)all_corrections.data(), all_corrections.size() * sizeof(u64));
    ofs.close();
    auto t_file_end = std::chrono::high_resolution_clock::now();
    g_stats.file_time += std::chrono::duration<double>(t_file_end - t_file_start).count();
    
    // 清理
    mc_gpu_cleanup(mc);
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    
    std::fprintf(stderr, "\n[Party %d] Done: %.2f s, %.2f K/s\n", 
                 party, total_sec, num_triples / total_sec / 1000.0);
    std::fprintf(stderr, "[Party %d] OT calls: %zu, Total OTs: %zu\n",
                 party, g_stats.ot_calls.load(), g_stats.total_ots.load());
    std::fprintf(stderr, "[Party %d] Time breakdown:\n", party);
    std::fprintf(stderr, "  GPU gen (ChaCha):  %.2fs\n", g_stats.gpu_gen_time.load());
    std::fprintf(stderr, "  GPU build:         %.2fs\n", g_stats.gpu_build_time.load());
    std::fprintf(stderr, "  GPU aggregate:     %.2fs\n", g_stats.gpu_agg_time.load());
    std::fprintf(stderr, "  OT communication:  %.2fs\n", g_stats.ot_comm_time.load());
    std::fprintf(stderr, "  File I/O:          %.2fs\n", g_stats.file_time.load());
    
    size_t sent = g_stats.bytes_sent.load();
    size_t recv = g_stats.bytes_recv.load();
    double sent_mb = sent / (1024.0 * 1024.0);
    double recv_mb = recv / (1024.0 * 1024.0);
    std::fprintf(stderr, "[Party %d] Communication: Sent=%.2f MB, Recv=%.2f MB\n",
                 party, sent_mb, recv_mb);
    std::fprintf(stderr, "[Party %d] Bytes per triple: %.2f bytes\n",
                 party, (double)(sent + recv) / num_triples);
}

// ============================================================
// 验证 (CPU 版本的 gen_ab)
// ============================================================
static inline void qr(u32 &a, u32 &b, u32 &c, u32 &d) {
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8)  | (d >> 24);
    c += d; b ^= c; b = (b << 7)  | (b >> 25);
}

static void chacha20_block(const u32 key[8], const u32 nonce[3], u32 counter, u32 out[16]) {
    u32 s[16] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
        counter, nonce[0], nonce[1], nonce[2]
    };
    for (int i = 0; i < 16; ++i) out[i] = s[i];
    for (int r = 0; r < 10; ++r) {
        qr(out[0],out[4],out[8],out[12]); qr(out[1],out[5],out[9],out[13]);
        qr(out[2],out[6],out[10],out[14]); qr(out[3],out[7],out[11],out[15]);
        qr(out[0],out[5],out[10],out[15]); qr(out[1],out[6],out[11],out[12]);
        qr(out[2],out[7],out[8],out[13]); qr(out[3],out[4],out[9],out[14]);
    }
    for (int i = 0; i < 16; ++i) out[i] += s[i];
}

static void gen_ab_cpu(u64 seed_hi, u64 seed_lo, u64 idx, u64& a, u64& b) {
    u32 s0 = (u32)seed_lo, s1 = (u32)(seed_lo >> 32);
    u32 s2 = (u32)seed_hi, s3 = (u32)(seed_hi >> 32);
    u32 key[8] = { s0^0xA5A5A5A5, s1^0x3C6EF372, s2^0x9E3779B9, s3^0xC3EFE9DB,
                   s0^s2, s1^s3, s0^s3, s1^s2 };
    u32 nonce[3] = { 0xDEADBEEF, 0xFEEDFACE, 0x12345678 };
    u32 out[16];
    chacha20_block(key, nonce, (u32)idx, out);
    u64 mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);
    a = (((u64)out[0] << 32) | out[1]) & mask;
    b = (((u64)out[2] << 32) | out[3]) & mask;
}

bool verify_sampled(int party, u64 seed_hi, u64 seed_lo,
                    const std::string& my_file, size_t num_triples)
{
    const size_t SAMPLES = 10000;
    
    u64 peer_seed[2], my_seed[2] = { seed_hi, seed_lo };
    MPI_Sendrecv(my_seed, 2, MPI_UNSIGNED_LONG_LONG, 1 - party, 9000,
                 peer_seed, 2, MPI_UNSIGNED_LONG_LONG, 1 - party, 9000,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    std::vector<size_t> indices(SAMPLES);
    if (party == 0) {
        std::mt19937_64 rng(12345);
        for (size_t i = 0; i < SAMPLES; ++i)
            indices[i] = rng() % num_triples;
        std::sort(indices.begin(), indices.end());
    }
    MPI_Bcast(indices.data(), SAMPLES, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    
    std::ifstream ifs(my_file, std::ios::binary);
    std::vector<u64> my_corr(SAMPLES);
    for (size_t i = 0; i < SAMPLES; ++i) {
        ifs.seekg(Header::size() + indices[i] * sizeof(u64));
        ifs.read((char*)&my_corr[i], sizeof(u64));
    }
    ifs.close();
    
    std::vector<u64> peer_corr(SAMPLES);
    MPI_Sendrecv(my_corr.data(), SAMPLES, MPI_UNSIGNED_LONG_LONG, 1 - party, 9001,
                 peer_corr.data(), SAMPLES, MPI_UNSIGNED_LONG_LONG, 1 - party, 9001,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    if (party != 0) return true;
    
    u64 mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);
    size_t bad = 0;
    
    for (size_t i = 0; i < SAMPLES; ++i) {
        size_t idx = indices[i];
        size_t global_idx = g_start_index + idx;
        
        u64 a0, b0, a1, b1;
        gen_ab_cpu(seed_hi, seed_lo, global_idx, a0, b0);
        gen_ab_cpu(peer_seed[0], peer_seed[1], global_idx, a1, b1);
        
        u64 c0 = ((a0 * b0) + my_corr[i]) & mask;
        u64 c1 = ((a1 * b1) + peer_corr[i]) & mask;
        u64 a = (a0 + a1) & mask;
        u64 b = (b0 + b1) & mask;
        u64 c = (c0 + c1) & mask;
        u64 exp = (a * b) & mask;
        
        if (c != exp) ++bad;
    }
    
    if (bad == 0) {
        std::fprintf(stderr, "[Verify] ✓ All %zu samples OK\n", SAMPLES);
        return true;
    } else {
        std::fprintf(stderr, "[Verify] ✗ %zu mismatches\n", bad);
        return false;
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        if (rank == 0) std::fprintf(stderr, "Need exactly 2 processes\n");
        MPI_Finalize();
        return 1;
    }
    
    // 解析参数
    size_t num_triples = 10000;
    std::string output_prefix = "offline_party";
    bool do_verify = true;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--num_triples" && i + 1 < argc) {
            num_triples = std::stoull(argv[++i]);
        } else if (arg == "--bits" && i + 1 < argc) {
            g_k_bits = std::stoi(argv[++i]);
        } else if (arg == "--channels" && i + 1 < argc) {
            g_num_channels = std::stoi(argv[++i]);
        } else if (arg == "--start" && i + 1 < argc) {
            g_start_index = std::stoull(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (arg == "--max-ot" && i + 1 < argc) {
            g_max_ot_per_call = std::stoull(argv[++i]);
        } else if (arg == "--no-verify") {
            do_verify = false;
        }
    }
    
    // 生成种子
    u64 seed_hi, seed_lo;
    if (rank == 0) {
        std::random_device rd;
        seed_hi = ((u64)rd() << 32) | rd();
        seed_lo = ((u64)rd() << 32) | rd();
    }
    MPI_Bcast(&seed_hi, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seed_lo, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    
    // 生成 OT 种子 (双方相同)
    u64 ot_seed = seed_hi ^ seed_lo ^ 0xDEADBEEF12345678ULL;
    
    // 输出文件名
    std::string output_file = output_prefix + std::to_string(rank) + ".bin";
    
    if (rank == 0) {
        size_t total_ots = num_triples * g_k_bits * 2;  // cross1 + cross2
        size_t est_ot_calls = (total_ots + g_max_ot_per_call - 1) / g_max_ot_per_call;
        
        std::fprintf(stderr, "╔══════════════════════════════════════════════════════════════╗\n");
        std::fprintf(stderr, "║  BMT Offline Phase (GPU + Multi-channel)                     ║\n");
        std::fprintf(stderr, "╠══════════════════════════════════════════════════════════════╣\n");
        std::fprintf(stderr, "║  Bit width:         %8d                               ║\n", g_k_bits);
        std::fprintf(stderr, "║  Channels:          %8d                               ║\n", g_num_channels);
        std::fprintf(stderr, "║  Triples:      %13zu                               ║\n", num_triples);
        std::fprintf(stderr, "║  Max OT/call:  %13zu                               ║\n", g_max_ot_per_call);
        std::fprintf(stderr, "║  Est OT calls: %13zu                               ║\n", est_ot_calls);
        std::fprintf(stderr, "╚══════════════════════════════════════════════════════════════╝\n");
    }
    
    // 握手
    int dummy = rank;
    MPI_Sendrecv(&dummy, 1, MPI_INT, 1-rank, 0,
                 &dummy, 1, MPI_INT, 1-rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::fprintf(stderr, "[Party %d] Handshake OK\n", rank);
    
    // 生成
    generate_multichannel_gpu(rank, num_triples, seed_hi, seed_lo, ot_seed, output_file);
    
    // 验证
    MPI_Barrier(MPI_COMM_WORLD);
    if (do_verify) {
        verify_sampled(rank, seed_hi, seed_lo, output_file, num_triples);
    }
    
    MPI_Finalize();
    return 0;
}