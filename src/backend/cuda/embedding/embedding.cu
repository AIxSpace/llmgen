#include "../utils.cuh"

__global__ void embedding_kernel(bf16x8 *output, const int *input_ids, const bf16x8 *weight, int H8) {
    const int idx = blockIdx.x;
    const int id = input_ids[idx];

    const bf16x8 *src = weight + id * H8;
    bf16x8 *dst = output + idx * H8;
    
    for (int i = threadIdx.x; i < H8; i += blockDim.x) {
        dst[i] = src[i];
    }
}

extern "C" void cuda_embedding(cuda_bf16_t *output, const int *input_ids, const cuda_bf16_t *weight, int T, int H) {
    if (T <= 0) return;
    int H8 = H / 8;
    embedding_kernel<<<T, 128, 0, global_cuda_stream>>>((bf16x8 *)output, input_ids, (const bf16x8 *)weight, H8);
}