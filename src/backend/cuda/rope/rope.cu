// lab: module-style - rope.cu -- ApplyRopeImpl<CUDA>(NEOX 半旋转)
// 对照 src/backend/cuda/rope.cu(apply_rope_kernel: 每线程一个元素, i < rope_half 的线程旋转一对)。
// cos/sin 来自 rope_cache 节点([max_seq, rope_dim] F32, 由 RopeCacheImpl<CPU> 预算, 经 memcpy 代理常驻显存),
// 不移植 RopeCacheImpl<CUDA>: C++ 的 kernel::rope_cache 恒走 CPU 特化, 从不调用 CUDA 版。

#include "../utils.cuh"

// ─── RoPE 应用内核 ─────────────────────────────────────────────────
// 对 Q/K 的前 rope_dim 维应用半旋转编码，剩余维度原样复制

template <typename T>
__global__ void apply_rope_kernel(
    T* __restrict__ out,
    const T* __restrict__ input,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int64_t B, int64_t n_heads, int64_t seq_len,
    int64_t head_dim, int64_t rope_half, int64_t rope_dim,
    int64_t head_stride, int64_t batch_stride,
    int64_t cos_stride, int64_t start_pos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * n_heads * seq_len * head_dim) {
        return;
    }

    int i = idx % head_dim;
    int tmp = idx / head_dim;
    int s = tmp % seq_len;
    tmp /= seq_len;
    int h = tmp % n_heads;
    int b = tmp / n_heads;

    const T* row_in = input + b * batch_stride + h * head_stride + s * head_dim;
    T* row_out = out + b * batch_stride + h * head_stride + s * head_dim;
    int64_t pos = start_pos + s;

    if (i < rope_half) {
        float x0 = float(row_in[i]);
        float x1 = float(row_in[i + rope_half]);
        float c = cos_cache[pos * cos_stride + i];
        float si = sin_cache[pos * cos_stride + i];
        row_out[i] = T(x0 * c - x1 * si);
        row_out[i + rope_half] = T(x0 * si + x1 * c);
    } else if (i >= rope_dim) {
        row_out[i] = row_in[i];
    }
}

extern "C" void cuda_apply_rope(void *out, const void *x, const float *cos, const float *sin, int dtype,
                                int64_t B, int64_t n_heads, int64_t seq_len,
                                int64_t head_dim, int64_t rope_dim, int64_t start_pos, int device_id) {
    cudaSetDevice(device_id);

    int64_t rope_half = rope_dim / 2;

    int64_t head_stride  = seq_len * head_dim;
    int64_t batch_stride = n_heads * head_stride;
    int64_t cos_stride   = rope_dim;

    int64_t total = B * n_heads * seq_len * head_dim;
    constexpr int THREADS = 256;
    int blocks = (total + THREADS - 1) / THREADS;

    switch (dtype) {
        case GT_F32:
            apply_rope_kernel<<<blocks, THREADS>>>(
                static_cast<float*>(out), static_cast<const float*>(x), cos, sin,
                B, n_heads, seq_len, head_dim, rope_half, rope_dim,
                head_stride, batch_stride, cos_stride, start_pos);
            break;
        case GT_F16:
            apply_rope_kernel<<<blocks, THREADS>>>(
                static_cast<__half*>(out), static_cast<const __half*>(x), cos, sin,
                B, n_heads, seq_len, head_dim, rope_half, rope_dim,
                head_stride, batch_stride, cos_stride, start_pos);
            break;
        case GT_BF16:
            apply_rope_kernel<<<blocks, THREADS>>>(
                static_cast<__nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(x), cos, sin,
                B, n_heads, seq_len, head_dim, rope_half, rope_dim,
                head_stride, batch_stride, cos_stride, start_pos);
            break;
        default:
            fprintf(stderr, "cuda::apply_rope: unsupported dtype\n");
            return;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda::apply_rope kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}