#include <stdio.h>

#include "../utils.cuh"
#include "atten_common.cuh"

template <typename T, int HEAD_DIM=128>
__global__ void flash_attention_warp_kernel(
    T* __restrict__ out,
    const T* __restrict__ Q,
    int64_t B, int64_t num_q_heads, int64_t seq_len, int64_t num_kv_heads,
    float scale_val, int causal, int num_kv_groups,
    const int32_t* __restrict__ page_k_ids,
    const int32_t* __restrict__ page_v_ids,
    const T* __restrict__ k_pool_base,
    const T* __restrict__ v_pool_base,
    int32_t block_stride_elems,
    int32_t total_cached, int32_t q_start
) {
    const int bh = blockIdx.x;
    if (bh >= B * num_q_heads) return;
    const int head_idx  = bh % num_q_heads;
    const int batch_idx = bh / num_q_heads;
    const int kv_head_idx = head_idx / num_kv_groups;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int q_pos = warp_id + blockIdx.y * static_cast<int>(blockDim.x / 32);
    const bool active = q_pos < seq_len; // C++ 在这里 return; 本 lab 的越界 warp 仍要参与搬 tile 与 __syncthreads

    const int64_t q_base  = (batch_idx * num_q_heads + head_idx) * seq_len * HEAD_DIM;
    const int64_t out_base = q_base;

    constexpr int KV_TILE = 32;
    extern __shared__ char smem_[];
    T* shared_K = reinterpret_cast<T*>(smem_);
    T* shared_V = reinterpret_cast<T*>(smem_ + KV_TILE * HEAD_DIM * sizeof(T));

    constexpr int VEC = HEAD_DIM / 32;
    float q_reg[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        int d_idx = lane_id + i * 32;
        q_reg[i] = active ? float(Q[q_base + q_pos * HEAD_DIM + d_idx]) : 0.f;
    }
    float m_i = -INFINITY;
    float l_i = 0.f;
    float o_reg[VEC] = {0};

    for (int kv_tile = 0; kv_tile < total_cached; kv_tile += KV_TILE) {
        int tile_size = KV_TILE < static_cast<int>(total_cached - kv_tile) ? KV_TILE : static_cast<int>(total_cached - kv_tile);

        // C++: shared_K[i] = K[kv_base + (kv_tile + k) * HEAD_DIM + d]; 这里按逻辑位置查页表取块内 token-major 行
        for (int i = threadIdx.x; i < tile_size * HEAD_DIM; i += blockDim.x) {
            int k = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            int kv_pos = kv_tile + k;
            int blk = kv_pos / PAGE_BLOCK_SIZE;
            int p = kv_pos % PAGE_BLOCK_SIZE;
            int64_t kv_off = (int64_t(p) * num_kv_heads + kv_head_idx) * HEAD_DIM + d;
            shared_K[i] = k_pool_base[int64_t(page_k_ids[blk]) * block_stride_elems + kv_off];
            shared_V[i] = v_pool_base[int64_t(page_v_ids[blk]) * block_stride_elems + kv_off];
        }
        __syncthreads();

        if (active) {
            for (int k = 0; k < tile_size; ++k) {
                int kv_pos = kv_tile + k;
                if (causal && kv_pos > q_start + q_pos) continue;

                float score_partial = 0.f;
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    int d_idx = lane_id + i * 32;
                    score_partial += q_reg[i] * float(shared_K[k * HEAD_DIM + d_idx]);
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    score_partial += __shfl_down_sync(0xffffffff, score_partial, offset);
                }
                float score = __shfl_sync(0xffffffff, score_partial, 0);
                score *= scale_val;

                float m_prev = m_i;
                m_i = fmaxf(m_i, score);
                float exp_score = __expf(score - m_i);
                float exp_prev  = __expf(m_prev - m_i);
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    int d_idx = lane_id + i * 32;
                    float v = float(shared_V[k * HEAD_DIM + d_idx]);
                    o_reg[i] = o_reg[i] * exp_prev + exp_score * v;
                }
                l_i = l_i * exp_prev + exp_score;
            }
        }
        __syncthreads();
    }

    if (!active) return;
    float inv_l = 1.f / l_i;
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        int d_idx = lane_id + i * 32;
        out[out_base + q_pos * HEAD_DIM + d_idx] = T(o_reg[i] * inv_l);
    }
}

template <typename T, int HD>
static void launch_flash(dim3 grid_dim, int threads, void* out, const void* Q,
                         int64_t B, int64_t num_q_heads, int64_t seq_len, int64_t num_kv_heads,
                         float scale_val, int causal, int num_kv_groups,
                         const int32_t* d_page_k_ids, const int32_t* d_page_v_ids,
                         const void* k_pool_base, const void* v_pool_base,
                         int32_t block_stride_elems, int32_t total_cached, int32_t q_start) {
    constexpr int KV_TILE = 32;
    const size_t smem = 2 * KV_TILE * HD * sizeof(T);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(flash_attention_warp_kernel<T, HD>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }
    flash_attention_warp_kernel<T, HD><<<grid_dim, threads, smem>>>(
        static_cast<T*>(out), static_cast<const T*>(Q),
        B, num_q_heads, seq_len, num_kv_heads, scale_val, causal, num_kv_groups,
        d_page_k_ids, d_page_v_ids,
        static_cast<const T*>(k_pool_base), static_cast<const T*>(v_pool_base),
        block_stride_elems, total_cached, q_start);
}

extern "C" void cuda_flash_attention(void *out, const void *Q, int dtype,
                                     int32_t head_dim, int32_t num_q_heads, int32_t seq_len, int32_t num_kv_heads,
                                     int32_t total_cached, float scale, int causal, int32_t num_kv_groups,
                                     const int32_t *h_page_k_ids, const int32_t *h_page_v_ids, int32_t num_blocks,
                                     const void *k_pool_base, const void *v_pool_base, int32_t block_stride_elems,
                                     int32_t q_start, int dev_id) {
    cudaSetDevice(dev_id);
    int64_t B = 1; // 注意力张量是 4D [B, H, S, D](同 C++), 只支持 batch=1(C++ 取 Q->dims[0], 恒为 1)

    // ===== 强约束（当前 kernel 版本）=====
    if (head_dim != 128 && head_dim != 256) {
        fprintf(stderr, "flash_attention_warp_kernel requires head_dim in {128, 256}\n");
        return;
    }
    if (seq_len <= 0) {
        fprintf(stderr, "invalid sequence length\n");
        return;
    }

    int32_t *d_page_k_ids = nullptr, *d_page_v_ids = nullptr;
    upload_page_table(h_page_k_ids, h_page_v_ids, num_blocks, &d_page_k_ids, &d_page_v_ids);

    // ===== kernel 配置 =====
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    const int blocks_x = static_cast<int>(B * num_q_heads);
    const int blocks_y = static_cast<int>((seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    const dim3 grid_dim(blocks_x, blocks_y);

    // ===== launch with runtime head_dim dispatch =====
#define FLASH_ARGS grid_dim, THREADS, out, Q, B, num_q_heads, seq_len, num_kv_heads, scale, causal, num_kv_groups, \
                   d_page_k_ids, d_page_v_ids, k_pool_base, v_pool_base, block_stride_elems, total_cached, q_start
    switch (dtype) {
        case GT_F32:
            if (head_dim == 256) launch_flash<float, 256>(FLASH_ARGS); else launch_flash<float, 128>(FLASH_ARGS);
            break;
        case GT_F16:
            if (head_dim == 256) launch_flash<__half, 256>(FLASH_ARGS); else launch_flash<__half, 128>(FLASH_ARGS);
            break;
        case GT_BF16:
            if (head_dim == 256) launch_flash<__nv_bfloat16, 256>(FLASH_ARGS); else launch_flash<__nv_bfloat16, 128>(FLASH_ARGS);
            break;
        default:
            fprintf(stderr, "FlashAttention: unsupported dtype\n");
    }
#undef FLASH_ARGS

    cudaFree(d_page_k_ids);
    cudaFree(d_page_v_ids);

    // ===== error check =====
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attention_warp_kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}