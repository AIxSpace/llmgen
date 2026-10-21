// lab: module-serving-unfused -- sample.cu: GPU 端采样(每步只把 S 个 int32 拷回 host)
//
// 对照 module-style: logits 在 host, CPU 标量 argmax 扫整个 151936 词表.
// 这里两段式: 每行拆 32 个 block 各扫一段词表出候选, 再由一个 warp 合并; temperature>0 用 Gumbel-max(argmax(logit/T + G), G = -log(-log U))
// —— 与 nano-vllm Sampler 的 argmax(softmax(logit/T) / Exp(1)) 同分布; temperature<=0 退化为 greedy.
// 随机数: 无状态计数器哈希(seed, 行, 词 id) → CUDA Graph 回放时只需更新 device 上的 seed.

#include "../utils.cuh"

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// argmax 归约: 并列取小下标(与 torch.argmax 一致)
__device__ __forceinline__ void argmax_merge(float &best, int &best_i, float ov, int oi) {
    if (ov > best || (ov == best && oi < best_i)) { best = ov; best_i = oi; }
}
__device__ __forceinline__ void warp_argmax(float &best, int &best_i) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, best_i, o);
        argmax_merge(best, best_i, ov, oi);
    }
}

// 第一段: grid = (S, SAMPLE_SPLIT), 每个 block 扫词表的一段, 输出段内 (最大值, 下标)
#define SAMPLE_SPLIT 32
#define SAMPLE_THREADS 256
__global__ void __launch_bounds__(SAMPLE_THREADS) sample_partial_kernel(float *pv, int *pi, const bf16 *logits, const float *temps,
                                                                        const unsigned long long *seed_ptr, int V) {
    const int s = blockIdx.x, part = blockIdx.y;
    const int chunk = (V + SAMPLE_SPLIT - 1) / SAMPLE_SPLIT;
    const int beg = part * chunk, end = min(V, beg + chunk);
    const bf16 *row = logits + (long)s * V;
    const float temp = temps[s];
    const unsigned long long seed = splitmix64(*seed_ptr ^ ((unsigned long long)s << 40));
    float best = -INFINITY;
    int best_i = 0x7fffffff;
    if (temp > 0.f) {
        const float inv_t = 1.f / temp;
        for (int i = beg + threadIdx.x; i < end; i += SAMPLE_THREADS) {
            unsigned long long r = splitmix64(seed + (unsigned long long)i);
            float u = ((float)(r >> 40) + 0.5f) * (1.0f / 16777216.0f); // (0,1)
            argmax_merge(best, best_i, bf2f(row[i]) * inv_t - __logf(-__logf(u)), i);
        }
    } else {
        for (int i = beg + threadIdx.x; i < end; i += SAMPLE_THREADS) argmax_merge(best, best_i, bf2f(row[i]), i);
    }
    warp_argmax(best, best_i);
    __shared__ float sv[SAMPLE_THREADS / 32];
    __shared__ int si[SAMPLE_THREADS / 32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { sv[wid] = best; si[wid] = best_i; }
    __syncthreads();
    if (wid == 0) {
        best = lane < SAMPLE_THREADS / 32 ? sv[lane] : -INFINITY;
        best_i = lane < SAMPLE_THREADS / 32 ? si[lane] : 0x7fffffff;
        warp_argmax(best, best_i);
        if (lane == 0) {
            pv[s * SAMPLE_SPLIT + part] = best;
            pi[s * SAMPLE_SPLIT + part] = best_i;
        }
    }
}

// 第二段: 一个 warp 合并 SAMPLE_SPLIT 个候选
__global__ void sample_final_kernel(int *out, const float *pv, const int *pi) {
    const int s = blockIdx.x, lane = threadIdx.x;
    float best = pv[s * SAMPLE_SPLIT + lane];
    int best_i = pi[s * SAMPLE_SPLIT + lane];
    warp_argmax(best, best_i);
    if (lane == 0) out[s] = best_i;
}

extern "C" void cuda_sample(int *out_ids, const cuda_bf16_t *logits, const float *temps, const unsigned long long *seed,
                          int S, int V, void *scratch) {
    if (S <= 0) return;
    float *pv = (float *)scratch;
    int *pi = (int *)(pv + (long)S * SAMPLE_SPLIT);
    sample_partial_kernel<<<dim3(S, SAMPLE_SPLIT), SAMPLE_THREADS, 0, global_cuda_stream>>>(pv, pi, (const bf16 *)logits, temps, seed, V);
    sample_final_kernel<<<S, 32, 0, global_cuda_stream>>>(out_ids, pv, pi);
}

extern "C" size_t cuda_sample_scratch_bytes(int max_S) { return (size_t)max_S * SAMPLE_SPLIT * 8; }