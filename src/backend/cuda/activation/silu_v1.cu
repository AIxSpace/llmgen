#include "../utils.cuh"

// ------------------------------------------------------------------
// silu: out = x / (1 + e^-x), n 个元素(n 为 8 的倍数)
// ------------------------------------------------------------------
__global__ void silu_kernel(bf16x8 *output, const bf16x8 *input, size_t n8) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) {
        float x[8];
        unpackbf16x8(input[i], x);

#pragma unroll
        for (int k = 0; k < 8; ++k) {
            x[k] = x[k] / (1.f + __expf(-x[k]));
        }

        output[i] = packbf16x8(x);
    }
}

extern "C" void cuda_silu(cuda_bf16_t *output, const cuda_bf16_t *x, size_t n) {
    const size_t n8 = n / 8;
    if (n8 <= 0) return;
    silu_kernel<<<grid_for(n8, 256), 256, 0, global_cuda_stream>>>((bf16x8 *)output, (const bf16x8 *)x, n8);
}