#include <cstdlib>
#include "../utils.cuh"

extern "C" void cuda_memcpy(void *dst, const void *src, size_t nbytes, int src_device_type, int dst_device_type, int dev_id) {
    cudaSetDevice(dev_id);
    
    if (!src) {
        fprintf(stderr, "MemcpyImpl<CUDA>: source tensor has no data\n");
        return;
    }

    // 所有拷贝都排进全局流: 它是 NonBlocking 流, 和默认流之间没有隐式同步, 混用就会有数据竞争

    // cpu -> cuda: 可分页内存的 async H2D 会先拷进驱动的中转缓冲再返回, 调用方可以马上复用 src
    if (src_device_type == DEV_CPU && dst_device_type == DEV_CUDA) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, global_cuda_stream));
        return;
    }

    // cuda -> cpu: host 马上要读结果, 拷完同步一次
    if (src_device_type == DEV_CUDA && dst_device_type == DEV_CPU) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost, global_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(global_cuda_stream));
        return;
    }

    // cuda -> cuda
    if (src_device_type == DEV_CUDA && dst_device_type == DEV_CUDA) {
        cudaPointerAttributes src_attr, dst_attr;
        cudaPointerGetAttributes(&src_attr, src);
        cudaPointerGetAttributes(&dst_attr, dst);

        if (src_attr.device == dst_attr.device) {
            CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, global_cuda_stream));
        } else {
            CUDA_CHECK(cudaStreamSynchronize(global_cuda_stream)); // 跨设备走 host 中转, 先等源数据算完
            // 跨设备：CUDA:0 → host → CUDA:1
            int dst_id = dst_attr.device;
            void *temp = malloc(nbytes);

            cudaSetDevice(src_attr.device);
            cudaMemcpy(temp, src, nbytes, cudaMemcpyDeviceToHost);

            cudaSetDevice(dst_id);
            cudaMemcpy(dst, temp, nbytes, cudaMemcpyHostToDevice);

            free(temp);
        }
        return;
    }

    fprintf(stderr, "cuda_memcpy unsupported copy (%d -> %d)\n", src_device_type, dst_device_type);
}

// ------------------------------------------------------------------
// gather_rows: out[s] = x[idx[s]]
// ------------------------------------------------------------------
__global__ void gather_rows_kernel(bf16x8 *out, const bf16x8 *x, const int *idx, int H8) {
    int s = blockIdx.x;
    long src = idx[s];
    for (int i = threadIdx.x; i < H8; i += blockDim.x) out[(long)s * H8 + i] = x[src * H8 + i];
}

extern "C" void cuda_gather_rows(cuda_bf16_t *out, const cuda_bf16_t *x, const int *idx, int S, int H) {
    if (S <= 0) return;
    gather_rows_kernel<<<S, 128, 0, global_cuda_stream>>>((bf16x8 *)out, (const bf16x8 *)x, idx, H / 8);
}