#include "../utils.cuh"

__global__ void mul_kernel(bf16x8 *output, const bf16x8 *input1, const bf16x8 *input2, size_t n8) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) {
        float x[8], y[8];
        unpackbf16x8(input1[i], x);
        unpackbf16x8(input2[i], y);

#pragma unroll
        for (int k = 0; k < 8; ++k) {
            x[k] *= y[k];
        }

        output[i] = packbf16x8(x);
    }
}

extern "C" void cuda_mul(cuda_bf16_t *output, const cuda_bf16_t *input1, const cuda_bf16_t *input2, size_t n) {
    const size_t n8 = n / 8;
    if (n8 <= 0) return;
    mul_kernel<<<grid_for(n8, 256), 256, 0, global_cuda_stream>>>((bf16x8 *)output, (const bf16x8 *)input1, (const bf16x8 *)input2, n8);
}