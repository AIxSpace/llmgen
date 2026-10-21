#include "../utils.cuh"

// ------------------------------------------------------------------
// rope.cu -- NEOX RoPE, token-major: x / out [T, n_head, 128]
// 一个 block 一个 token, 一个 warp 一个头; lane 负责 i ∈ {lane, lane+32} 及其配对 i+64
// 位置从本步元数据的 positions[t] 读, 不再是"起点 + 序号"。
// cos / sin 是 llmgen 的两张 [max_seq, 128] f32 表(RopeCacheImpl<CPU> 算一次, 经拷贝代理常驻显存),
// 每行前后两半存同一个角度, 这里只读前半 i ∈ [0, 64)。和 backend/cpu/rope.zig 同一套表、同一个公式。
// ------------------------------------------------------------------

__global__ void rope_kernel(bf16 *output, const bf16 *x, const float *cos_tab, const float *sin_tab, const int *positions, int n_head) {
    const int HEAD_DIM = 128; // 每个 head 的维度
    const int HALF_DIM = 64;  // 旋转配对的间距

    const int token = blockIdx.x;       // 一个 block = 一个 token
    const int head = threadIdx.x / 32;  // 一个 warp = 一个 head
    const int lane = threadIdx.x % 32;  // warp 内的 lane

    if (head >= n_head) {
        return;
    }

    // 该 token / head 在 x、output 里的起始偏移
    const long head_base = ((long)token * n_head + head) * HEAD_DIM;

    // 按 position 取两张表的同一行
    const long row = (long)positions[token] * HEAD_DIM;
    const float *cos_row = cos_tab + row;
    const float *sin_row = sin_tab + row;

    // 每个 lane 负责两个旋转对: 下标 lane 和 lane+32, 合起来覆盖 0..63
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int i = lane + 32 * half; // 0..63, 它的配对是 i + HALF_DIM
        const int i_pair = i + HALF_DIM;

        const float x_lo = bf2f(x[head_base + i]);
        const float x_hi = bf2f(x[head_base + i_pair]);
        const float c = cos_row[i];
        const float s = sin_row[i];

        output[head_base + i] = f2bf(x_lo * c - x_hi * s);
        output[head_base + i_pair] = f2bf(x_hi * c + x_lo * s);
    }
}

extern "C" void cuda_rope(cuda_bf16_t *output, const cuda_bf16_t *x, const float *cos_tab, const float *sin_tab,
                          const int *positions, int T, int n_head, int hd) {
    if (T <= 0) {
        return;
    }

    if (hd != 128 || n_head > 32) {
        fprintf(stderr, "[serve] rope: 仅支持 head_dim=128 且 n_head<=32\n");
        return;
    }

    rope_kernel<<<T, n_head * 32, 0, global_cuda_stream>>>((bf16 *)output, (const bf16 *)x, cos_tab, sin_tab, positions, n_head);
}
