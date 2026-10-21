#include "../utils.cuh"

template <typename T>
__global__ void softmax_kernel(const T* __restrict__ input, T* __restrict__ output, size_t outer_dim, size_t axis_dim, size_t inner_dim) {
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t total = outer_dim * inner_dim;

    if (global_idx >= total) return;

    // 计算索引
    size_t outer_idx = global_idx / inner_dim;
    size_t inner_idx = global_idx % inner_dim;

    const size_t base_offset = outer_idx * axis_dim * inner_dim + inner_idx;

    // 计算最大值 
    float max_val = -INFINITY;
    for (size_t i = 0; i < axis_dim; ++i) {
        float val = static_cast<float>(input[base_offset + i * inner_dim]);
        max_val = fmaxf(val, max_val);
    }

    // 指数计算
    float sum = 0.0f;
    for (size_t i = 0; i < axis_dim; ++i) {
        float val = static_cast<float>(input[base_offset + i * inner_dim]);
        sum += expf(val - max_val);
    }

    // softmax
    for (size_t i = 0; i < axis_dim; ++i) {
        float val = static_cast<float>(input[base_offset + i * inner_dim]);
        float result = expf(val - max_val) / sum;
        output[base_offset + i * inner_dim] = static_cast<T>(result);
    }
}

// 对照 SoftmaxImpl<Device::CUDA>::execute(Tensor* t, int32_t dev_id)
// outer / axis / inner 由 Zig 侧按 C++ 的方式从 dims 与 axis 算好传进来.
extern "C" void cuda_softmax(void *out, const void *in, int dtype, size_t outer_dim, size_t axis_dim, size_t inner_dim, size_t numel, int dev_id) {
    cudaSetDevice(dev_id);

    constexpr int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    switch (dtype) {
        case GT_F16:
            softmax_kernel<<<blocks, threads>>>(static_cast<const __half*>(in), static_cast<__half*>(out), outer_dim, axis_dim, inner_dim);
            break;
        case GT_BF16:
            softmax_kernel<<<blocks, threads>>>(static_cast<const __nv_bfloat16*>(in), static_cast<__nv_bfloat16*>(out), outer_dim, axis_dim, inner_dim);
            break;
        case GT_F32:
            softmax_kernel<<<blocks, threads>>>(static_cast<const float*>(in), static_cast<float*>(out), outer_dim, axis_dim, inner_dim);
            break;
        default:
            fprintf(stderr, "cuda::softmax not implemented\n");
    }
}