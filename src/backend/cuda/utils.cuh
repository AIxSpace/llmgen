#ifndef GENLLM_CUDA_COMMON_CUH
#define GENLLM_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// GGML dtype 编码
#define GT_F32  0
#define GT_F16  1
#define GT_BF16 30

// 设备编码(对照 types.DeviceType)
#define DEV_CPU  0
#define DEV_CUDA 1

typedef unsigned short cuda_bf16_t;

extern cudaStream_t global_cuda_stream;
extern cublasHandle_t global_cublas_handle;

// todo: - 这里这样转换的指标计算依据
// 8 个 bf16 = 16 字节的向量化读写
struct __align__(16) bf16x8 {
    __nv_bfloat162 pairs[4]; // 2 个 bf16 * 4 = 8 个
};

__device__ __forceinline__ bf16x8 packbf16x8(const float *f) {
    bf16x8 value;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        value.pairs[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]); // float -> bf16float x 2
    }
    return value;
}

__device__ __forceinline__ void unpackbf16x8(const bf16x8 &value, float *f) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f2v = __bfloat1622float2(value.pairs[i]); // bf16 x 2 => float x 2
        f[2 * i] = f2v.x;
        f[2 * i + 1] = f2v.y;
    }
}

// todo: - delete unpack8 and pack8
__device__ __forceinline__ void unpack8(const bf16x8 &v, float *f) { 
    unpackbf16x8(v, f); 
}

__device__ __forceinline__ bf16x8 pack8(const float *f) {
    return packbf16x8(f); 
}

typedef __nv_bfloat16 bf16;
__device__ __forceinline__ float bf2f(bf16 x) { 
    return __bfloat162float(x); 
}
__device__ __forceinline__ bf16 f2bf(float x) { 
    return __float2bfloat16(x); 
}

__device__ __forceinline__ float round_bf(float x) { 
    return __bfloat162float(__float2bfloat16(x)); 
}

static inline unsigned grid_for(size_t n, int threads) {
    size_t b = (n + threads - 1) / threads;
    return (unsigned)(b > 65535L * 8 ? 65535L * 8 : b);
}

// todo: - this should change more ...
static inline void checkCuda(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "[cuda_shim] %s: %s\n", what, cudaGetErrorString(e));
    }
}

#define CUDA_CHECK(expr)                                                                        \
    do {                                                                                      \
        cudaError_t _e = (expr);                                                              \
        if (_e != cudaSuccess) fprintf(stderr, "[serve] %s:%d %s: %s\n", __FILE__, __LINE__, \
                                       #expr, cudaGetErrorString(_e));                        \
    } while (0)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    return v;
}

// 块内求和(blockDim.x 为 32 的倍数, ≤ 1024); 结果广播给所有线程
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float shared[32];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    v = warp_sum(v);
    __syncthreads();

    if (lane_id == 0) {
        shared[warp_id] = v;
    }
    __syncthreads();

    int nw = (blockDim.x + 31) / 32;
    float r = (lane_id < nw) ? shared[lane_id] : 0.f;

    return warp_sum(r);
}

// todo: - loadWeightF32 和 dequant_weigh 有什么区别 ?
// 统一转 FP32
__device__ __forceinline__ float loadWeightF32(const void *base, int dtype, long i) {
    switch (dtype) {
        case GT_F32: {
            return ((const float *)base)[i];
        }
        case GT_BF16: { 
            unsigned int b = (unsigned int)((const unsigned short *)base)[i] << 16; float f; memcpy(&f, &b, 4); return f; 
        }
        case GT_F16:  {
            return __half2float(((const __half *)base)[i]); // 半精度转单精度
        }
        default: return 0.0f;
    }
}

// 设备端按 dtype 读第 i 个权重为 f32(对照 dtype_traits 的 dtype → float 转换)。
__device__ __forceinline__ float dequant_weight(const void *base, int dtype, long i) {
    switch (dtype) {
        case GT_F32:  return ((const float *)base)[i];
        case GT_BF16: return __bfloat162float(((const __nv_bfloat16 *)base)[i]);
        case GT_F16:  return __half2float(((const __half *)base)[i]);
        default:      return 0.0f;
    }
}
static inline int cuda_env_int(const char *name, int def) {
    const char *v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}

#endif // GENLLM_CUDA_COMMON_CUH
