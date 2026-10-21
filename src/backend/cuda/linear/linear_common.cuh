#pragma once
#include "cuda_common.cuh"

// 反量化
static __global__ void dequant_weight_kernel(float *wf32, const void *w, int dtype, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        wf32[idx] = loadWeightF32(w, dtype, idx);
    }
}