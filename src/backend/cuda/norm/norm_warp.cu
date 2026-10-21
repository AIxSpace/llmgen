#include "../utils.cuh"

// rms = x[1,seq,2560] - 平方-> [1,seq,2560] -mean-> [1,seq,1] -sqrt-> [1,seq,1]
// out = (x/rms) * w[2560]

#define WARP_THREAD_SIZE 32

template<typename T, typename W>
__global__ void rmsnorm_warp_kernel(T *out, const T *x , const W *weight, int rows, int hidden_size, float eps){
    int warp_id = threadIdx.x / WARP_THREAD_SIZE; // 因为一个warp是32个线程，所以每个warp处理一行
    int lane_id = threadIdx.x % WARP_THREAD_SIZE; // 算出当前线程在warp中的编号

    int cur_row = blockIdx.x * (blockDim.x / 32) + warp_id; // 算出当前线程处理的行号，共rows行
    if (cur_row >= rows) {
        return; // 超出行数，直接返回
    }

    const T *x_row = x + (long)cur_row * hidden_size; // 输入地址
    T *out_row = out + (long)cur_row * hidden_size;   // 输出地址

    // 1. 计算平方和 - 32 个线程交错读取整个 hidden_size 维度
    float sum = 0.f;
    for (int i = lane_id; i < hidden_size; i += WARP_THREAD_SIZE) {
        float v = float(x_row[i]);
        sum += v * v;
    }

    // warp reduce
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    float mean_sq = __shfl_sync(0xffffffff, sum, 0) / hidden_size; // 平均值
    float inv_rms = rsqrtf(mean_sq + eps);

    // 2. scale
    for (int i = lane_id; i < hidden_size; i += WARP_THREAD_SIZE) {
        float v = float(x_row[i]);
        float o = v * inv_rms * float(weight[i]); // scale
        out_row[i] = T(o);
    }
}

template<typename T>
static void launch_rmsnorm(int blocks, int threads, void* out, const void* x, const void* w, int wdtype,
                           int rows, int hidden_size, float eps) {
    T *output = static_cast<T*>(out);
    const T* input = static_cast<const T*>(x);

    switch (wdtype) {
        case GT_F16:  rmsnorm_warp_kernel<<<blocks, threads, 0, global_cuda_stream>>>(output, input, static_cast<const __half*>(w), rows, hidden_size, eps); break;
        case GT_BF16: rmsnorm_warp_kernel<<<blocks, threads, 0, global_cuda_stream>>>(output, input, static_cast<const __nv_bfloat16*>(w), rows, hidden_size, eps); break;
        case GT_F32:  rmsnorm_warp_kernel<<<blocks, threads, 0, global_cuda_stream>>>(output, input, static_cast<const float*>(w), rows, hidden_size, eps); break;
        default: fprintf(stderr, "RMSNorm: unsupported weight dtype %d\n", wdtype);
    }
}

extern "C" void cuda_rms_norm(void *out, const void *x, const void *w, int dtype, int wdtype,
                              int rows, int hidden_size, float eps, int device_id) {
    cudaSetDevice(device_id);

    constexpr int WARPS_PER_BLOCK = 4;
    int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    constexpr int THREADS = WARPS_PER_BLOCK * WARP_THREAD_SIZE;

    switch (dtype) {
        case GT_F16:  launch_rmsnorm<__half>(blocks, THREADS, out, x, w, wdtype, rows, hidden_size, eps); break;
        case GT_BF16: launch_rmsnorm<__nv_bfloat16>(blocks, THREADS, out, x, w, wdtype, rows, hidden_size, eps); break;
        case GT_F32:  launch_rmsnorm<float>(blocks, THREADS, out, x, w, wdtype, rows, hidden_size, eps); break;
        default: fprintf(stderr, "RMSNorm only supports fp32/fp16/bf16\n"); return;
    }

    cudaError_t err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        fprintf(stderr, "rmsnorm kernel failed: %s\n", cudaGetErrorString(err));
    }
}