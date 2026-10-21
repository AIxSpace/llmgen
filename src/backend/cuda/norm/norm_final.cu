#include "../utils.cuh"

// RMSNorm:x 看成 [rows, D], out = (x·rstd).to(bf16) · w。对照 lab module-serving-unfused/backend/cuda/normalization.cu
// 行宽 D = 权重长度, Qwen3 里只有两种:
//   D = hidden(1024):attn_norm / ffn_norm / final_norm, 一个 block(128 线程)一行, 每线程一个 bf16x8, 只读一次就覆盖整行
//   D = 128(head_dim):q_norm / k_norm 每头一行, 一个 warp 一行, 每个 lane 管 4 个元素存在寄存器里, 不读第二遍
// 舍入点同 PyTorch 的 Qwen3RMSNorm:先把 x·rstd 舍入成 bf16, 再乘权重

__global__ void rmsnorm_row_kernel(bf16x8 *out, const bf16x8 *x, const float *w, int D8, float eps) {
    const long t = blockIdx.x;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D8; i += blockDim.x) {
        float v[8];
        unpack8(x[t * D8 + i], v);
#pragma unroll
        for (int k = 0; k < 8; ++k) ss += v[k] * v[k];
    }
    const float rstd = rsqrtf(block_sum(ss) / (float)(D8 * 8) + eps); // block_sum 把结果广播给所有线程
    for (int i = threadIdx.x; i < D8; i += blockDim.x) {
        float v[8];
        unpack8(x[t * D8 + i], v); // D = 1024 时每线程只有一个 bf16x8, 这次读命中 L1
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = round_bf(v[k] * rstd) * w[i * 8 + k];
        out[t * D8 + i] = pack8(v);
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
        v[j] = bf2f(xr[lane + 32 * j]); // lane, lane+32, lane+64, lane+96:相邻 lane 读相邻地址, 合并访存
        ss += v[j] * v[j];
    }
    const float rs = rsqrtf(warp_sum(ss) / (float)HD + eps); // xor 蝶形归约, 32 个 lane 都拿到总和
    bf16 *o = out + r * HD;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[lane + 32 * j] = f2bf(round_bf(v[j] * rs) * w[lane + 32 * j]);
}

extern "C" void cuda_rmsnorm(void *out, const void *x, const float *w, long rows, int D, float eps) {
    if (rows <= 0) return;
    if (D == 128) {
        const int rows_per_block = 4;
        rmsnorm_head_kernel<<<(unsigned)((rows + rows_per_block - 1) / rows_per_block), rows_per_block * 32, 0,
                              global_cuda_stream>>>((bf16 *)out, (const bf16 *)x, w, rows, eps);
        return;
    }
    rmsnorm_row_kernel<<<(unsigned)rows, 128, 0, global_cuda_stream>>>((bf16x8 *)out, (const bf16x8 *)x, w, D / 8, eps);
}
