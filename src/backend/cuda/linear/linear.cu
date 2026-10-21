// lab: module-serving-unfused -- linear.cu: y[M,N] = x[M,K] · W[N,K]^T
//
// 对照 module-style linear.cu: 那里每次调用都把 BF16 权重反量化成 f32 临时显存再 cublasSgemm(无 tensor core).
// 这里权重常驻显存且保持 BF16, 分两条路:
//   M ≤ SV_GEMV_MAX_M(decode 极小批): 自写 GEMV —— 一个 warp 负责一行输出, 16B 向量化顺序读权重行, M 个输入行复用同一次权重读取.
//     decode 是纯带宽受限(每步把 1.2GB 权重读一遍), M=1~2 时它比 cuBLAS 快 ~10%(cuda-course 第 05 课).
//   其余: cuBLAS GemmEx(A/B=BF16, C=BF16 或 f32, compute=32F) → tensor core.
//     M 变大后 GEMV 每个 lane 要为 M 行各解包一次 x、各累加一份, 计算和寄存器压力上来:
//     M=8 时 lm_head 只剩 487 GB/s, cuBLAS 仍有 864 GB/s; 而 CUDA Graph 的 decode 桶 1,2,4,8 会把 5~7 条序列补到 8,
//     所以阈值卡在 cuBLAS 反超之前. cuBLAS 用 cuda_init 里的固定 workspace, 照样能录进 CUDA Graph.
//
// 行主 → 列主换算(cuBLAS 路): y 行主 [M,N] == 列主 [N,M]
//   C_cm[N,M] = op(W)^T[N,K] · op(x)[K,M];  W 行主 [N,K] = 列主 [K,N] → OP_T, lda=K; x 行主 [M,K] = 列主 [K,M] → OP_N, ldb=K

#include "../utils.cuh"

#define GEMV_WARPS 8
#ifndef SV_GEMV_MAX_M
#define SV_GEMV_MAX_M 2 // 3090 实测见 README.md 的「GEMV 阈值」一节
#endif

template <int M, bool OUT_F32>
__global__ void __launch_bounds__(GEMV_WARPS * 32) gemv_kernel(void *y, const bf16x8 *x, const bf16x8 *w, int N, int K8) {
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * GEMV_WARPS + (threadIdx.x >> 5);
    if (n >= N) return; // 整个 warp 一起退出(n 在 warp 内一致), 不影响下面的 warp shuffle
    const bf16x8 *wr = w + (long)n * K8;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll 4
    for (int k = lane; k < K8; k += 32) {
        float wf[8];
        unpack8(wr[k], wf);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float xf[8];
            unpack8(x[(long)m * K8 + k], xf);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[m] += wf[e] * xf[e];
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float v = warp_sum(acc[m]);
        if (lane == 0) {
            if (OUT_F32) ((float *)y)[(long)m * N + n] = v;
            else ((bf16 *)y)[(long)m * N + n] = f2bf(v);
        }
    }
}

template <int M>
static void launch_gemv(void *y, int y_f32, const cuda_bf16_t *x, const cuda_bf16_t *w, int N, int K) {
    dim3 grid((N + GEMV_WARPS - 1) / GEMV_WARPS);
    if (y_f32) {
        gemv_kernel<M, true><<<grid, GEMV_WARPS * 32, 0, global_cuda_stream>>>(y, (const bf16x8 *)x, (const bf16x8 *)w, N, K / 8);
    } else {
        gemv_kernel<M, false><<<grid, GEMV_WARPS * 32, 0, global_cuda_stream>>>(y, (const bf16x8 *)x, (const bf16x8 *)w, N, K / 8);
    }
}

// 环境变量 SV_GEMV_MAX_M 覆盖编译期默认值(0 = 全走 cuBLAS, 最大 8), 首次调用读一次
static int gemv_max_m() {
    static const int m = [] {
        int v = cuda_env_int("SV_GEMV_MAX_M", SV_GEMV_MAX_M);
        v = v < 0 ? 0 : (v > 8 ? 8 : v);
        if (v != SV_GEMV_MAX_M) fprintf(stderr, "[serve] 回退开关 SV_GEMV_MAX_M=%d\n", v);
        return v;
    }();
    return m;
}

extern "C" void cuda_linear(void *y, int y_f32, const cuda_bf16_t *x, const cuda_bf16_t *w, int M, int N, int K) {
    if (M <= 0) return;
    if (M <= gemv_max_m() && K % 8 == 0) {
        switch (M) {
            case 1: launch_gemv<1>(y, y_f32, x, w, N, K); return;
            case 2: launch_gemv<2>(y, y_f32, x, w, N, K); return;
            case 3: launch_gemv<3>(y, y_f32, x, w, N, K); return;
            case 4: launch_gemv<4>(y, y_f32, x, w, N, K); return;
            case 5: launch_gemv<5>(y, y_f32, x, w, N, K); return;
            case 6: launch_gemv<6>(y, y_f32, x, w, N, K); return;
            case 7: launch_gemv<7>(y, y_f32, x, w, N, K); return;
            case 8: launch_gemv<8>(y, y_f32, x, w, N, K); return;
        }
    }
    const float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasGemmEx(global_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha,
                                     w, CUDA_R_16BF, K,
                                     x, CUDA_R_16BF, K,
                                     &beta,
                                     y, y_f32 ? CUDA_R_32F : CUDA_R_16BF, N,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) fprintf(stderr, "[serve] cublasGemmEx failed: %d (M=%d N=%d K=%d)\n", (int)st, M, N, K);
}