#include <stdio.h>

#include "../utils.cuh"

extern "C" void *cuda_malloc(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }

    void *p = nullptr;
    cudaError_t err = cudaMalloc(&p, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "CudaMemoryResource: cudaMalloc %zu bytes failed: %s\n", bytes, cudaGetErrorString(err));
        return nullptr;
    }
    return p;
}

extern "C" void cuda_free(void *device_data_ptr) { 
    if (device_data_ptr) {
        cudaFree(device_data_ptr);
    }
}

extern "C" void *cuda_host_alloc(size_t bytes) {
    void *p = nullptr;
    if (cudaMallocHost(&p, bytes) != cudaSuccess) return nullptr;
    return p;
}

extern "C" void cuda_host_free(void *p) { if (p) cudaFreeHost(p); }