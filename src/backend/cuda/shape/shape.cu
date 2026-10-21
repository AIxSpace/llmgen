// lab: module-style - shape.cu -- PermuteImpl<CUDA>(本 lab 的 ReshapePermute / PermuteReshape 都落到这里)
// 对照 src/backend/cuda/shape.cu(permute_kernel: 每线程一个输入元素, 按 strides 拆坐标 → 换轴 → 写输出)。
// C++ 图里是 Reshape(零拷贝视图, 只改 dims/strides)+ Permute 两个节点; 本 lab 是一个融合节点,
// 由 Zig 侧按"reshape 之后的视图"算好 in/out strides, kernel 与 C++ 逐字一致。

#include "../utils.cuh"

#define TENSOR_MAX_DIMS 4

template <typename T, const int MAX_DIM = TENSOR_MAX_DIMS>
__global__ void permute_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    const uint64_t* in_strides,
    const uint64_t* out_strides,
    const int64_t* axes,
    int ndim,
    size_t numel
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;
    // 计算多维坐标 old_idx
    int coord[MAX_DIM]; // 假设 <= 8 维
    int tmp = tid;
    for (int i = 0; i < ndim; ++i) {
        coord[i] = tmp / in_strides[i];
        tmp %= in_strides[i];
    }
    // 生成新坐标 new_coord
    int new_coord[MAX_DIM];
    for (int i = 0; i < ndim; ++i) {
        new_coord[i] = coord[axes[i]];
    }
    // 计算输出位置
    int out_index = 0;
    for (int i = 0; i < ndim; ++i) {
        out_index += new_coord[i] * out_strides[i];
    }
    dst[out_index] = src[tid];
}

// 对照 PermuteImpl<Device::CUDA>::execute(Tensor* out, int32_t dev_id)
extern "C" void cuda_permute(void *out, const void *in, int dtype, const uint64_t *in_strides, const uint64_t *out_strides,
                             const int64_t *axes, int ndim, size_t numel, int device_id) {
    cudaSetDevice(device_id);

    // 如果 ndim == 1, 就是恒等映射，无需计算
    if (ndim <= 1) return;

    int64_t* d_axes;
    uint64_t* d_in_strides, * d_out_strides;
    cudaMallocManaged(&d_axes, sizeof(int64_t) * ndim);
    cudaMallocManaged(&d_in_strides, sizeof(uint64_t) * ndim);
    cudaMallocManaged(&d_out_strides, sizeof(uint64_t) * ndim);

    for(int i = 0; i < ndim; ++i){
        d_in_strides[i] = in_strides[i];
        d_out_strides[i] = out_strides[i];
        d_axes[i] = axes[i];
    }

    constexpr int threads = 256;
    size_t total = numel;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    switch (dtype) {
        case GT_F32:
            permute_kernel<<<blocks, threads>>>(static_cast<float*>(out), static_cast<const float*>(in),
                                                d_in_strides, d_out_strides, d_axes, ndim, total);
            break;
        case GT_F16:
            permute_kernel<<<blocks, threads>>>(static_cast<__half*>(out), static_cast<const __half*>(in),
                                                d_in_strides, d_out_strides, d_axes, ndim, total);
            break;
        case GT_BF16:
            permute_kernel<<<blocks, threads>>>(static_cast<__nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(in),
                                                d_in_strides, d_out_strides, d_axes, ndim, total);
            break;
        default:
            fprintf(stderr, "cuda::permute: unsupported dtype\n");
    }
    cudaFree(d_in_strides);
    cudaFree(d_out_strides);
    cudaFree(d_axes);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda::permute kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}