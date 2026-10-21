#include "../utils.cuh"

template<typename T>
__global__ void mul_kernel(const T* input1, const T *input2, T *output, size_t size) {
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < size) {
        output[global_idx] = input1[global_idx] * input2[global_idx];
    }
}

extern "C" void cuda_mul(void *out, const void *in1, const void *in2, int dtype, size_t numel, int device_id) {
    cudaSetDevice(device_id);

    constexpr int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    switch (dtype) {
        case GT_F16:
            mul_kernel<<<blocks, threads>>>(static_cast<const __half*>(in1), static_cast<const __half*>(in2), static_cast<__half*>(out), numel);
            break;
        case GT_BF16:
            mul_kernel<<<blocks, threads>>>(static_cast<const __nv_bfloat16*>(in1), static_cast<const __nv_bfloat16*>(in2), static_cast<__nv_bfloat16*>(out), numel);
            break;
        case GT_F32:
            mul_kernel<<<blocks, threads>>>(static_cast<const float*>(in1), static_cast<const float*>(in2), static_cast<float*>(out), numel);
            break;
        default:
            fprintf(stderr, "cuda_mul not implemented\n");
    }
}