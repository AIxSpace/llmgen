#include "../utils.cuh"

// output =  x / (1 + exp(-x))

template <typename T>
__global__ void silu_kernel(const T *input, T *output, size_t size) {
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_idx < size) {
        float x = float(input[global_idx]);
        output[global_idx] = T(x / (1.0f + expf(-x))); // 强制类型转换
    }
}

extern "C" void cuda_silu(void *out, const void *in, int dtype, size_t numel, int device_id) {
    cudaSetDevice(device_id);

    constexpr int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    switch (dtype) {
        case GT_F16:
            silu_kernel<<<blocks, threads>>>(static_cast<const __half*>(in), static_cast<__half*>(out), numel);
            break;
        case GT_BF16:
            silu_kernel<<<blocks, threads>>>(static_cast<const __nv_bfloat16*>(in), static_cast<__nv_bfloat16*>(out), numel);
            break;
        case GT_F32:
            silu_kernel<<<blocks, threads>>>(static_cast<const float*>(in), static_cast<float*>(out), numel);
            break;
        default:
            fprintf(stderr, "cuda_silu not implemented\n");
    }
}