#include <stdio.h>

#include "../utils.cuh"
#include "atten_common.cuh"

template <typename T>
__global__ void paged_attention_kernel(
    T *out,
    const T *Q,
    int32_t head_dim, int32_t num_q_heads, int32_t seq_len,
    int32_t num_kv_heads, int32_t total_cached,
    float scale_val, int causal, int num_kv_groups,
    const int32_t *page_k_ids,
    const int32_t *page_v_ids,
    int32_t num_blocks,
    const T *k_pool_base,
    const T *v_pool_base,
    int32_t block_stride_elems,
    int32_t q_start
) {
    constexpr int32_t MAX_HEAD_DIM = 256;
    constexpr int32_t MAX_HEAD_DIM_REGS = MAX_HEAD_DIM / 32;
    int32_t head_dim_regs = head_dim / 32;

    int32_t bh = blockIdx.x;
    if (bh >= num_q_heads) return;
    int32_t head_idx  = bh % num_q_heads;
    int32_t batch_idx = bh / num_q_heads;
    int32_t kv_head_idx = head_idx / num_kv_groups;

    int32_t warp_id = threadIdx.x / 32;
    int32_t lane_id = threadIdx.x % 32;
    int32_t q_pos = warp_id + blockIdx.y * (blockDim.x / 32);
    if (q_pos >= seq_len) return;

    int64_t q_base = (int64_t(batch_idx) * num_q_heads + head_idx) * seq_len * head_dim;

    float q_reg[MAX_HEAD_DIM_REGS];
    for (int i = 0; i < head_dim_regs; ++i) {
        int d = lane_id + i * 32;
        q_reg[i] = float(Q[q_base + q_pos * head_dim + d]);
    }

    float m_i = -INFINITY;
    float l_i = 0.f;
    float o_reg[MAX_HEAD_DIM_REGS] = {0};

    constexpr int32_t BLOCK_SIZE = PAGE_BLOCK_SIZE;

    for (int32_t blk = 0; blk < num_blocks; ++blk) {
        int32_t k_blk_id = page_k_ids[blk];
        int32_t v_blk_id = page_v_ids[blk];
        if (k_blk_id < 0 || v_blk_id < 0) break;

        const T* k_blk_ptr = k_pool_base + int64_t(k_blk_id) * block_stride_elems;
        const T* v_blk_ptr = v_pool_base + int64_t(v_blk_id) * block_stride_elems;

        int32_t blk_start = blk * BLOCK_SIZE;
        int32_t blk_end = min(blk_start + BLOCK_SIZE, total_cached);

        for (int32_t p = 0; p < blk_end - blk_start; ++p) {
            int32_t kv_pos = blk_start + p;
            if (causal && kv_pos > q_start + q_pos) continue;

            int64_t kv_off = (int64_t(p) * num_kv_heads + kv_head_idx) * head_dim;

            float score_partial = 0.f;
            for (int i = 0; i < head_dim_regs; ++i) {
                int d = lane_id + i * 32;
                score_partial += q_reg[i] * float(k_blk_ptr[kv_off + d]);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                score_partial += __shfl_down_sync(0xffffffff, score_partial, offset);
            float score = __shfl_sync(0xffffffff, score_partial, 0) * scale_val;

            float m_prev = m_i;
            m_i = fmaxf(m_i, score);
            float exp_score = __expf(score - m_i);
            float exp_prev  = __expf(m_prev - m_i);
            for (int i = 0; i < head_dim_regs; ++i) {
                int d = lane_id + i * 32;
                o_reg[i] = o_reg[i] * exp_prev + exp_score * float(v_blk_ptr[kv_off + d]);
            }
            l_i = l_i * exp_prev + exp_score;
        }
    }

    float inv_l = 1.f / l_i;
    for (int i = 0; i < head_dim_regs; ++i) {
        int d = lane_id + i * 32;
        out[q_base + q_pos * head_dim + d] = T(o_reg[i] * inv_l);
    }
}

extern "C" void cuda_paged_attention(void *out, const void *Q, int dtype,
                                     int32_t head_dim, int32_t num_q_heads, int32_t seq_len, int32_t num_kv_heads,
                                     int32_t total_cached, float scale, int causal, int32_t num_kv_groups,
                                     const int32_t *h_page_k_ids, const int32_t *h_page_v_ids, int32_t num_blocks,
                                     const void *k_pool_base, const void *v_pool_base, int32_t block_stride_elems,
                                     int32_t q_start, int dev_id) {
    cudaSetDevice(dev_id);

    if (head_dim % 32 != 0) {
        fprintf(stderr, "paged_attention requires head_dim to be a multiple of 32\n");
        return;
    }

    int32_t *d_page_k_ids = nullptr, *d_page_v_ids = nullptr;
    upload_page_table(h_page_k_ids, h_page_v_ids, num_blocks, &d_page_k_ids, &d_page_v_ids);

    constexpr int32_t WARPS_PER_BLOCK = 4;
    constexpr int32_t THREADS = WARPS_PER_BLOCK * 32;
    int32_t blocks_x = num_q_heads;
    int32_t blocks_y = (seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid_dim(blocks_x, blocks_y);

    switch (dtype) {
        case GT_F32:
            paged_attention_kernel<float><<<grid_dim, THREADS>>>(
                static_cast<float*>(out), static_cast<const float*>(Q),
                head_dim, num_q_heads, seq_len, num_kv_heads, total_cached,
                scale, causal, num_kv_groups,
                d_page_k_ids, d_page_v_ids, num_blocks,
                static_cast<const float*>(k_pool_base), static_cast<const float*>(v_pool_base),
                block_stride_elems, q_start);
            break;
        case GT_F16:
            paged_attention_kernel<__half><<<grid_dim, THREADS>>>(
                static_cast<__half*>(out), static_cast<const __half*>(Q),
                head_dim, num_q_heads, seq_len, num_kv_heads, total_cached,
                scale, causal, num_kv_groups,
                d_page_k_ids, d_page_v_ids, num_blocks,
                static_cast<const __half*>(k_pool_base), static_cast<const __half*>(v_pool_base),
                block_stride_elems, q_start);
            break;
        case GT_BF16:
            paged_attention_kernel<__nv_bfloat16><<<grid_dim, THREADS>>>(
                static_cast<__nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(Q),
                head_dim, num_q_heads, seq_len, num_kv_heads, total_cached,
                scale, causal, num_kv_groups,
                d_page_k_ids, d_page_v_ids, num_blocks,
                static_cast<const __nv_bfloat16*>(k_pool_base), static_cast<const __nv_bfloat16*>(v_pool_base),
                block_stride_elems, q_start);
            break;
        default:
            fprintf(stderr, "paged_attention: unsupported dtype\n");
    }

    cudaFree(d_page_k_ids);
    cudaFree(d_page_v_ids);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "paged_attention_kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}