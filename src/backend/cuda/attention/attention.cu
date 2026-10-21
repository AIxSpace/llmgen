// lab: module-serving-unfused -- attention.cu: 分页 KV cache 上的因果 GQA 注意力(head_dim 固定 128)
//
// 对照 module-style: 那里 CUDA 设备上的注意力其实跑的是 CPU 标量实现(PagedAttentionImpl 无 CUDA 分支).
// 与 module-style 一样, 注意力节点先把本步 k/v 写进分页 cache(store_kv), 再算注意力. 这里三个 kernel:
//   store_kv: 本步 k/v → 分页 cache 的 slot.
//   decode : split-K flash-decoding. grid = (kv_head, seq, part), 一个 block 处理一个 KV 头的一段 token,
//            GQA 组内 G 个 query 头**共享同一次 KV 读取**(Qwen3-0.6B: G=2, KV 流量减半).
//            各 part 的 (未归一化 o, max, sum) 写 scratch, 再由 reduce kernel 合并.
//   prefill: tensor core FlashAttention-2(mma.sync). grid = (query tile, kv_head), 一个 tile = 一条序列的 64 个 query
//            × GQA 组内全部 query 头; K/V 按 64 行一块用 cp.async 双缓冲从分页 cache 搬进 shared memory. 对 q_len=1 的行同样成立, 所以 chunked prefill
//            与 prefill/decode 混批也走它.
// KV cache 布局 [num_blocks, nkv, block_size, 128] (见 cuda_shim.h).
#include "../utils.cuh"

// ------------------------------------------------------------------
// store_kv: k/v [T, nkv, 128] → 分页 cache 的 slot(块内布局 [kv_head][slot][head_dim]); 一个 warp 一个 KV 头
// ------------------------------------------------------------------
__global__ void store_kv_kernel(const bf16 *k, const bf16 *v, const int *slots, bf16 *kc, bf16 *vc, int nkv,
                                int block_size) {
    const int HD = 128;
    const int t = blockIdx.x, h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (h >= nkv) return;
    const int slot = slots[t];
    if (slot < 0) return;
    const long blk = slot / block_size, off = slot % block_size;
    const long dst = ((blk * nkv + h) * block_size + off) * HD;
    const long src = ((long)t * nkv + h) * HD;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        kc[dst + lane + 32 * j] = k[src + lane + 32 * j];
        vc[dst + lane + 32 * j] = v[src + lane + 32 * j];
    }
}

extern "C" void cuda_store_kv(const cuda_bf16_t *k, const cuda_bf16_t *v, const int *slot_mapping, cuda_bf16_t *k_cache,
                            cuda_bf16_t *v_cache, int T, int nkv, int hd, int block_size) {
    if (T <= 0) return;
    if (hd != 128 || nkv > 32) {
        fprintf(stderr, "[serve] store_kv: 仅支持 head_dim=128 且 nkv<=32\n");
        return;
    }
    store_kv_kernel<<<T, nkv * 32, 0, global_cuda_stream>>>((const bf16 *)k, (const bf16 *)v, slot_mapping, (bf16 *)k_cache,
                                                   (bf16 *)v_cache, nkv, block_size);
}

#define HD 128

// ==================================================================
// decode
// ==================================================================
template <int G>
__global__ void __launch_bounds__(128) attn_decode_kernel(
    float *part_o, float *part_ml, const bf16 *q, const bf16 *kc, const bf16 *vc,
    const int *block_tables, const int *ctx_lens, int nq, int nkv, int max_blocks, int block_size,
    float scale, int part_size, int max_parts) {
    const int kvh = blockIdx.x, s = blockIdx.y, part = blockIdx.z;
    const int ctx = ctx_lens[s];
    const int t0 = part * part_size;
    if (t0 >= ctx) return;
    const int n = min(ctx - t0, part_size);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;

    extern __shared__ float smem[];
    float *sc = smem;                  // [G][part_size] 分数 → 概率
    float *red = smem + G * part_size; // [8][G][HD] PV 部分和
    __shared__ float q_s[G][HD];
    __shared__ float wred[4][G];
    __shared__ float stat_m[G], stat_l[G];

    for (int i = tid; i < G * HD; i += 128) {
        int g = i / HD, d = i % HD;
        q_s[g][d] = bf2f(q[(long)s * nq * HD + (long)(kvh * G + g) * HD + d]) * scale;
    }
    __syncthreads();

    // ---- phase 1: QK^T. 4 个 lane 一个 token, 每 lane 负责 4 个 16B 分块(32 维) ----
    const int lg = tid & 3, grp = tid >> 2; // 32 组
    float qr[G][32];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int k = 0; k < 8; ++k) qr[g][j * 8 + k] = q_s[g][(lg + 4 * j) * 8 + k];

    const int *bt = block_tables + (long)s * max_blocks;
    const long head_stride = (long)block_size * HD;
    // 循环次数必须 warp 内一致(下面有全 mask 的 __shfl_xor_sync), 越界 token 用 valid 屏蔽而不是提前退出
    for (int base = 0; base < n; base += 32) {
        const int i = base + grp;
        const bool valid = i < n;
        float acc[G];
#pragma unroll
        for (int g = 0; g < G; ++g) acc[g] = 0.f;
        if (valid) {
            int t = t0 + i;
            long blk = bt[t / block_size];
            const bf16x8 *kr = (const bf16x8 *)(kc + (blk * nkv + kvh) * head_stride + (long)(t % block_size) * HD);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float f[8];
                unpack8(kr[lg + 4 * j], f);
#pragma unroll
                for (int g = 0; g < G; ++g)
#pragma unroll
                    for (int k = 0; k < 8; ++k) acc[g] += qr[g][j * 8 + k] * f[k];
            }
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            acc[g] += __shfl_xor_sync(0xffffffffu, acc[g], 1);
            acc[g] += __shfl_xor_sync(0xffffffffu, acc[g], 2);
        }
        if (valid && lg == 0)
#pragma unroll
            for (int g = 0; g < G; ++g) sc[g * part_size + i] = acc[g];
    }
    __syncthreads();

    // ---- phase 2: 每个头的 max / exp / sum ----
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float m = -INFINITY;
        for (int i = tid; i < n; i += 128) m = fmaxf(m, sc[g * part_size + i]);
        m = warp_max(m);
        if (lane == 0) wred[wid][g] = m;
    }
    __syncthreads();
    if (tid < G) stat_m[tid] = fmaxf(fmaxf(wred[0][tid], wred[1][tid]), fmaxf(wred[2][tid], wred[3][tid]));
    __syncthreads();
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float m = stat_m[g], l = 0.f;
        for (int i = tid; i < n; i += 128) {
            float p = __expf(sc[g * part_size + i] - m);
            sc[g * part_size + i] = p;
            l += p;
        }
        l = warp_sum(l);
        if (lane == 0) wred[wid][g] = l;
    }
    __syncthreads();
    if (tid < G) stat_l[tid] = wred[0][tid] + wred[1][tid] + wred[2][tid] + wred[3][tid];

    // ---- phase 3: P·V. 16 线程一行(每线程 8 维), 8 个 token 槽并行 ----
    const int dl = tid & 15, ts = tid >> 4;
    float o[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int k = 0; k < 8; ++k) o[g][k] = 0.f;
    for (int i = ts; i < n; i += 8) {
        int t = t0 + i;
        long blk = bt[t / block_size];
        const bf16x8 *vr = (const bf16x8 *)(vc + (blk * nkv + kvh) * head_stride + (long)(t % block_size) * HD);
        float f[8];
        unpack8(vr[dl], f);
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float p = sc[g * part_size + i];
#pragma unroll
            for (int k = 0; k < 8; ++k) o[g][k] += p * f[k];
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int k = 0; k < 8; ++k) red[(ts * G + g) * HD + dl * 8 + k] = o[g][k];
    __syncthreads();
    for (int idx = tid; idx < G * HD; idx += 128) {
        int g = idx / HD, d = idx % HD;
        float acc = 0.f;
#pragma unroll
        for (int r = 0; r < 8; ++r) acc += red[(r * G + g) * HD + d];
        long h = kvh * G + g;
        part_o[(((long)s * nq + h) * max_parts + part) * HD + d] = acc;
    }
    if (tid < G) {
        long h = kvh * G + tid;
        float *ml = part_ml + (((long)s * nq + h) * max_parts + part) * 2;
        ml[0] = stat_m[tid];
        ml[1] = stat_l[tid];
    }
}

__global__ void attn_decode_reduce_kernel(bf16 *out, const float *part_o, const float *part_ml,
                                          const int *ctx_lens, int nq, int part_size, int max_parts) {
    const int h = blockIdx.x, s = blockIdx.y, d = threadIdx.x;
    const int ctx = ctx_lens[s];
    bf16 *o = out + (long)s * nq * HD + (long)h * HD;
    if (ctx <= 0) {
        o[d] = f2bf(0.f);
        return;
    }
    const int np = (ctx + part_size - 1) / part_size;
    const float *ml = part_ml + ((long)s * nq + h) * max_parts * 2;
    const float *po = part_o + ((long)s * nq + h) * max_parts * HD;
    float M = -INFINITY;
    for (int p = 0; p < np; ++p) M = fmaxf(M, ml[2 * p]);
    float num = 0.f, den = 0.f;
    for (int p = 0; p < np; ++p) {
        float w = __expf(ml[2 * p] - M);
        num += w * po[(long)p * HD + d];
        den += w * ml[2 * p + 1];
    }
    o[d] = f2bf(num / den);
}

extern "C" void cuda_attn_decode(cuda_bf16_t *out, const cuda_bf16_t *q, const cuda_bf16_t *k_cache, const cuda_bf16_t *v_cache,
                               const int *block_tables, const int *ctx_lens,
                               int S, int nq, int nkv, int hd, int max_blocks, int block_size, float scale,
                               float *part_o, float *part_ml, int part_size, int max_parts) {
    if (S <= 0) return;
    if (hd != HD) {
        fprintf(stderr, "[serve] attn_decode: 仅支持 head_dim=128\n");
        return;
    }
    const int G = nq / nkv;
    // 回退开关 SV_ATTN_PART: 段长至少为该值(只允许调大: max_parts 与 scratch 按默认段长分配, 段越长需要的段越少).
    // 例如 SV_ATTN_PART=100000 → 每条序列只有 1 段, 即不做 split-K, 复现第 11 课 bs=1 时注意力并行度不足的问题.
    static const int part_min = [] {
        int v = cuda_env_int("SV_ATTN_PART", 0);
        if (v > 0) fprintf(stderr, "[serve] 回退开关 SV_ATTN_PART=%d\n", v);
        return v;
    }();
    if (part_min > part_size) {
        const int smem_cap = (96 * 1024 / 4 - 8 * G * HD) / G; // [G][part_size] 分数必须放进 96 KB 动态 smem
        part_size = part_min < smem_cap ? part_min : smem_cap;
    }
    dim3 grid(nkv, S, max_parts);
    size_t smem = (size_t)(G * part_size + 8 * G * HD) * sizeof(float);
#define LAUNCH_DEC(GG)                                                                                          \
    do {                                                                                                        \
        static bool attr_set = false;                                                                           \
        if (!attr_set) {                                                                                        \
            cudaFuncSetAttribute(attn_decode_kernel<GG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); \
            attr_set = true;                                                                                    \
        }                                                                                                       \
        attn_decode_kernel<GG><<<grid, 128, smem, global_cuda_stream>>>(part_o, part_ml, (const bf16 *)q,             \
            (const bf16 *)k_cache, (const bf16 *)v_cache, block_tables, ctx_lens, nq, nkv, max_blocks,          \
            block_size, scale, part_size, max_parts);                                                           \
    } while (0)
    switch (G) {
        case 1: LAUNCH_DEC(1); break;
        case 2: LAUNCH_DEC(2); break;
        case 4: LAUNCH_DEC(4); break;
        case 8: LAUNCH_DEC(8); break;
        default: fprintf(stderr, "[serve] attn_decode: 不支持的 GQA 组大小 %d\n", G); return;
    }
#undef LAUNCH_DEC
    attn_decode_reduce_kernel<<<dim3(nq, S), HD, 0, global_cuda_stream>>>((bf16 *)out, part_o, part_ml, ctx_lens,
                                                                   nq, part_size, max_parts);
}

// ==================================================================
// prefill / 混合: tensor core 版 FlashAttention-2(mma.sync m16n8k16 BF16, f32 累加)
//
// 一个 block = 一个 query tile(64 行) × 一个 KV 头的 G 个 query 头, 每个头 4 个 warp 各管 16 行.
//   S = Q·K^T : Q 片段常驻寄存器(ldmatrix), K tile(64 key)经 ldmatrix 喂 mma
//   online softmax 在 mma 累加器布局上直接做(每线程持 2 行 × 若干列, quad 内 shuffle 求行 max)
//   O += P·V  : P 从 S 累加器原地打包成 A 片段(FA2 经典技巧), V 用 ldmatrix.trans 取 B 片段
// shared memory 行宽补到 136 个 bf16(272B), ldmatrix 的 8 行落在不同 bank, 无冲突.
// ==================================================================
#define TC_BQ 64
#define TC_BK 64
#define SROW 136

__device__ __forceinline__ void ldsm_x4(uint32_t *r, const void *p) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t *r, const void *p) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t *>(&v);
}

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem, bool valid) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
    int n = valid ? 16 : 0; // src-size=0 → 目标 16B 填零
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// 一个 block = 一个 query tile(64 行) × 一个 KV 头的全部 G 个 query 头(GQA 组共享同一份 K/V tile).
// 4·G 个 warp: warp w 负责 head (w / 4) 的第 (w % 4)·16 .. +15 行.
// K/V tile 用 cp.async 双缓冲: 计算第 i 块时第 i+1 块已在路上.
template <int G>
__global__ void __launch_bounds__(128 * G) attn_prefill_tc_kernel(
    bf16 *out, const bf16 *q, const bf16 *kc, const bf16 *vc, const int *block_tables,
    const int *ctx_lens, const int *cu_q, const int *tile_seq, const int *tile_q0,
    int nq, int nkv, int max_blocks, int block_size, float scale) {
    constexpr int NT = 128 * G;
    const int tile = blockIdx.x, kvh = blockIdx.y;
    const int s = tile_seq[tile], q0 = tile_q0[tile];
    const int q_beg = cu_q[s], q_len = cu_q[s + 1] - q_beg, ctx = ctx_lens[s];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int g = w >> 2, wr = w & 3; // 本 warp 的头(组内)与行段
    const int h = kvh * G + g;

    extern __shared__ __align__(16) unsigned char smraw[];
    bf16 *buf = (bf16 *)smraw; // [2 stage][K,V][64][SROW]; Q 暂存复用 stage 1
    auto Kst = [&](int st) { return buf + (st * 2 + 0) * TC_BK * SROW; };
    auto Vst = [&](int st) { return buf + (st * 2 + 1) * TC_BK * SROW; };

    const int *bt = block_tables + (long)s * max_blocks;
    const long head_stride = (long)block_size * HD;
    const int kend = ctx - q_len + min(q0 + TC_BQ, q_len);
    const int n_tiles = (kend + TC_BK - 1) / TC_BK;

    auto prefetch = [&](int it, int st) {
        const int kb = it * TC_BK;
        bf16 *Kd = Kst(st), *Vd = Vst(st);
        for (int i = tid; i < TC_BK * 16; i += NT) {
            int r = i >> 4, c = i & 15, t = kb + r;
            bool ok = t < kend;
            long base = 0;
            if (ok) {
                long blk = bt[t / block_size];
                base = (blk * nkv + kvh) * head_stride + (long)(t % block_size) * HD;
            }
            cp_async16(Kd + r * SROW + c * 8, kc + base + c * 8, ok);
            cp_async16(Vd + r * SROW + c * 8, vc + base + c * 8, ok);
        }
    };

    // ---- Q: G 个头 × 64 行 → 暂存到 stage 1 区(G·64 行 ≤ 2·64 行) → 寄存器 A 片段 ----
    {
        bf16 *Qs = Kst(1);
        for (int i = tid; i < G * TC_BQ * 16; i += NT) {
            int r = i >> 4, c = i & 15; // r = gg·64 + row
            int gg = r / TC_BQ, qi = q0 + (r % TC_BQ);
            bool ok = qi < q_len;
            const bf16 *src = q + (long)(q_beg + (ok ? qi : 0)) * nq * HD + (long)(kvh * G + gg) * HD + c * 8;
            cp_async16(Qs + r * SROW + c * 8, src, ok);
        }
        cp_async_commit();
    }
    if (n_tiles > 0) prefetch(0, 0);
    cp_async_commit();
    cp_async_wait<1>(); // Q 到齐(tile 0 可仍在路上)
    __syncthreads();
    uint32_t qa[8][4];
    {
        const bf16 *Qs = Kst(1) + (g * TC_BQ + wr * 16) * SROW;
#pragma unroll
        for (int j = 0; j < 8; ++j) ldsm_x4(qa[j], Qs + (lane & 15) * SROW + j * 16 + (lane >> 4) * 8);
    }
    __syncthreads(); // stage 1 之后要被 K/V 覆盖

    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[i][e] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const int qr0 = q0 + wr * 16 + (lane >> 2), qr1 = qr0 + 8;
    const int pos0 = ctx - q_len + qr0, pos1 = pos0 + 8;
    const bool v0 = qr0 < q_len, v1 = qr1 < q_len;
    const int warp_pos_max = ctx - q_len + min(q0 + wr * 16 + 15, q_len - 1); // 本 warp 能看到的最远 key
    const float sl2 = scale * 1.4426950408889634f;

    for (int it = 0; it < n_tiles; ++it) {
        if (it + 1 < n_tiles) prefetch(it + 1, (it + 1) & 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const int kb = it * TC_BK;
        const bf16 *Ks = Kst(it & 1), *Vs = Vst(it & 1);

        if (kb <= warp_pos_max) { // 整块都在对角线之上的 warp 跳过(因果)
            float sc[8][4];
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int e = 0; e < 4; ++e) sc[i][e] = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
#pragma unroll
                for (int np = 0; np < 4; ++np) {
                    uint32_t kb4[4];
                    ldsm_x4(kb4, Ks + (np * 16 + (lane & 7) + ((lane >> 4) << 3)) * SROW + j * 16 + ((lane >> 3) & 1) * 8);
                    mma_bf16(sc[2 * np], qa[j], kb4[0], kb4[1]);
                    mma_bf16(sc[2 * np + 1], qa[j], kb4[2], kb4[3]);
                }
            }
            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    int key = kb + nt * 8 + (lane & 3) * 2 + (e & 1);
                    bool ok = (e < 2) ? (v0 && key <= pos0) : (v1 && key <= pos1);
                    float v = ok ? sc[nt][e] * sl2 : -INFINITY;
                    sc[nt][e] = v;
                    if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
                }
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
            const float r0 = (mn0 == -INFINITY) ? 0.f : mn0, r1 = (mn1 == -INFINITY) ? 0.f : mn1;
            const float c0 = exp2f(m0 - r0), c1 = exp2f(m1 - r1);
            float ls0 = 0.f, ls1 = 0.f;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                sc[nt][0] = exp2f(sc[nt][0] - r0);
                sc[nt][1] = exp2f(sc[nt][1] - r0);
                sc[nt][2] = exp2f(sc[nt][2] - r1);
                sc[nt][3] = exp2f(sc[nt][3] - r1);
                ls0 += sc[nt][0] + sc[nt][1];
                ls1 += sc[nt][2] + sc[nt][3];
            }
            l0 = l0 * c0 + ls0; // 各线程只累加自己那几列, 结尾再 quad 内求和(c 对整行一致, 可交换)
            l1 = l1 * c1 + ls1;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                o[i][0] *= c0;
                o[i][1] *= c0;
                o[i][2] *= c1;
                o[i][3] *= c1;
            }
            m0 = mn0;
            m1 = mn1;
#pragma unroll
            for (int j2 = 0; j2 < 4; ++j2) {
                uint32_t pa[4] = {pack_bf2(sc[2 * j2][0], sc[2 * j2][1]), pack_bf2(sc[2 * j2][2], sc[2 * j2][3]),
                                  pack_bf2(sc[2 * j2 + 1][0], sc[2 * j2 + 1][1]), pack_bf2(sc[2 * j2 + 1][2], sc[2 * j2 + 1][3])};
#pragma unroll
                for (int dp = 0; dp < 8; ++dp) {
                    uint32_t vb[4];
                    ldsm_x4_t(vb, Vs + (j2 * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * SROW + dp * 16 + (lane >> 4) * 8);
                    mma_bf16(o[2 * dp], pa, vb[0], vb[1]);
                    mma_bf16(o[2 * dp + 1], pa, vb[2], vb[3]);
                }
            }
        }
        __syncthreads(); // 本 stage 读完, 下一轮的 prefetch 才能覆盖它
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = v0 ? 1.f / l0 : 0.f, i1 = v1 ? 1.f / l1 : 0.f;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        int d = nt * 8 + (lane & 3) * 2;
        if (v0) *(uint32_t *)(out + (long)(q_beg + qr0) * nq * HD + (long)h * HD + d) = pack_bf2(o[nt][0] * i0, o[nt][1] * i0);
        if (v1) *(uint32_t *)(out + (long)(q_beg + qr1) * nq * HD + (long)h * HD + d) = pack_bf2(o[nt][2] * i1, o[nt][3] * i1);
    }
}

extern "C" int cuda_attn_prefill_tile(void) { return TC_BQ; }

extern "C" void cuda_attn_prefill(cuda_bf16_t *out, const cuda_bf16_t *q, const cuda_bf16_t *k_cache, const cuda_bf16_t *v_cache,
                                const int *block_tables, const int *ctx_lens, const int *cu_q,
                                const int *tile_seq, const int *tile_q0, int n_tiles,
                                int nq, int nkv, int hd, int max_blocks, int block_size, float scale) {
    if (n_tiles <= 0) return;
    if (hd != HD) {
        fprintf(stderr, "[serve] attn_prefill: 仅支持 head_dim=128\n");
        return;
    }
    const int G = nq / nkv;
    const size_t smem = (size_t)(2 * 2 * TC_BK) * SROW * sizeof(bf16); // 双缓冲 K/V(Q 暂存复用其中一半)
    if (G * TC_BQ > 2 * TC_BK) {
        fprintf(stderr, "[serve] attn_prefill: GQA 组 %d 过大(Q 暂存放不下)\n", G);
        return;
    }
#define LAUNCH_PF(GG)                                                                                        \
    do {                                                                                                     \
        static bool attr_set = false;                                                                        \
        if (!attr_set) {                                                                                     \
            cudaFuncSetAttribute(attn_prefill_tc_kernel<GG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
            attr_set = true;                                                                                 \
        }                                                                                                    \
        attn_prefill_tc_kernel<GG><<<dim3(n_tiles, nkv), 128 * GG, smem, global_cuda_stream>>>(                     \
            (bf16 *)out, (const bf16 *)q, (const bf16 *)k_cache, (const bf16 *)v_cache, block_tables, ctx_lens, \
            cu_q, tile_seq, tile_q0, nq, nkv, max_blocks, block_size, scale);                                  \
    } while (0)
    switch (G) {
        case 1: LAUNCH_PF(1); break;
        case 2: LAUNCH_PF(2); break;
        default: fprintf(stderr, "[serve] attn_prefill: 不支持的 GQA 组大小 %d\n", G); return;
    }
#undef LAUNCH_PF
}
