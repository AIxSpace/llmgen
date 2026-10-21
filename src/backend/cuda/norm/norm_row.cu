// lab: module-serving-unfused -- normalization.cu -- RMSNorm
// 对照 module-style backend/cuda/normalization.cu 与 C++ src/backend/cuda/normalization.cu(RmsNormImpl<CUDA>).
// 行宽 = 权重长度: attn_norm / ffn_norm / final_norm 按 hidden(一个 block 一行), q_norm / k_norm 按 head_dim=128(一个 warp 一行).
// 数值同 Qwen3RMSNorm: (x·rstd).to(bf16) · w → bf16.

#include "../utils.cuh"

// ------------------------------------------------------------------
// rmsnorm: x 看成 [rows, D], out = (x·rstd).to(bf16) · w
//   D = 128(每头 QK-norm): 一个 warp 一行, 一个 block 4 行
//   其余(hidden):          一个 block 一行
// ------------------------------------------------------------------
__global__ void rmsnorm_row_kernel(bf16x8 *out, const bf16x8 *x, const float *w, int D8, float eps) {
    const long t = blockIdx.x;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D8; i += blockDim.x) {
        float v[8];
        unpackbf16x8(x[t * D8 + i], v);
#pragma unroll
        for (int k = 0; k < 8; ++k) ss += v[k] * v[k];
    }
    const float rstd = rsqrtf(block_sum(ss) / (float)(D8 * 8) + eps);
    for (int i = threadIdx.x; i < D8; i += blockDim.x) {
        float v[8];
        unpackbf16x8(x[t * D8 + i], v);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = round_bf(v[k] * rstd) * w[i * 8 + k];
        out[t * D8 + i] = packbf16x8(v);
    }
}

__global__ void rmsnorm_head_kernel(bf16 *out, const bf16 *x, const float *w, long rows, float eps) {
    const int HD = 128;

    const long r = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);

    const int lane = threadIdx.x & 31;

    if (r >= rows) return;

    const bf16 *xr = x + r * HD;
    float v[4];
    float ss = 0.f;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = bf2f(xr[lane + 32 * j]);
        ss += v[j] * v[j];
    }

    const float rs = rsqrtf(warp_sum(ss) / (float)HD + eps);
    bf16 *o = out + r * HD;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        o[lane + 32 * j] = f2bf(round_bf(v[j] * rs) * w[lane + 32 * j]);
    }
}

// 短行怕归约浪费，长行怕并行不足，于是用两个 kernel 各取所长
extern "C" void cuda_rmsnorm(cuda_bf16_t *out, const cuda_bf16_t *x, const float *w, long rows, int D, float eps) {
    if (rows <= 0) {
        return;
    }
    if (D == 128) {
        // head qk_norm
        const int rows_per_block = 4;
        rmsnorm_head_kernel<<<(unsigned)((rows + rows_per_block - 1) / rows_per_block), rows_per_block * 32, 0, global_cuda_stream>>>((bf16 *)out, (const bf16 *)x, w, rows, eps);
        return;
    }
    rmsnorm_row_kernel<<<(unsigned)rows, 128, 0, global_cuda_stream>>>((bf16x8 *)out, (const bf16x8 *)x, w, D / 8, eps);
}