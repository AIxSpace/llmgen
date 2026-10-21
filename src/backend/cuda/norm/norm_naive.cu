// 【口诀】RMSNorm
//   平方求均值，
//   加上 eps 再开方；
//   倒数乘回原向量，
//   最后再乘 gain。
//   不减均值不加偏，
//   比 LayerNorm 省一步。
// 【公式】
//   out = x / sqrt( mean(x²) + eps ) * gain
//
// RMSNorm: 沿最后一维归一化
//   out[r, h] = x[r, h] / sqrt( mean_h(x[r, :]²) + eps ) * gain[h]
//
// 维度:
//   x    : [rows, hidden]  f32   (更高维的张量把前面几维压平成 rows)
//   gain : [hidden]        原始字节(f32/bf16/f16), 所有行共用
//   out  : [rows, hidden]  f32
//
// Qwen3-0.6B 里的两种调用 (S = token 数):
//   attn_norm / ffn_norm : x = [S, 1024]          → rows = S,       hidden = 1024
//   q_norm               : x = [16, S, 128]       → rows = 16 × S,  hidden = 128
//   k_norm               : x = [ 8, S, 128]       → rows =  8 × S,  hidden = 128
//
// 线程映射: 一个线程负责一整行 (教学版, 累加顺序与 CPU 完全一致 → 对拍误差极小)
//   例: rows = 15 (S=15 的 attn_norm), block = 128 → grid = 1, 只有 15 个线程干活
#include "../../utils.cuh"

__global__ void rmsnorm_row_kernel(float *out, const float *x, const void *gain, int gdtype,
                                   int rows, int hidden, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // 这个线程负责第几行
    if (row >= rows) return;                         // 越界线程直接退出

    // 行首指针: 第 row 行从 row × hidden 开始, 连续 hidden 个 float
    const float *x_row = x + (long)row * hidden;
    float *out_row = out + (long)row * hidden;

    // ① 平方和 ss = Σ_h x[row, h]²          (标量)
    float ss = 0.f;
    for (int h = 0; h < hidden; ++h) ss += x_row[h] * x_row[h];

    // ② scale = 1 / sqrt(ss / hidden + eps)  (标量, 本行共用)
    float scale = 1.f / sqrtf(ss / (float)hidden + eps);

    // ③ out[row, h] = x[row, h] × scale × gain[h]   ([hidden] → [hidden])
    //    gain 按 dtype 在设备端解码成 f32
    for (int h = 0; h < hidden; ++h) out_row[h] = x_row[h] * scale * loadWeightF32(gain, gdtype, h);
}

extern "C" void cuda_rmsnorm(float *out_dev, const float *x_dev, const void *gain_dev, int gdtype, int rows, int hidden, float eps) {
    cuda_init();

    int tb = 128;                                   // 每个 block 128 个线程
    rmsnorm_row_kernel<<<(rows + tb - 1) / tb, tb>>>(out_dev, x_dev, gain_dev, gdtype, rows, hidden, eps);

    checkCuda(cudaGetLastError(), "rmsnorm launch"); // 捕获启动错误(配置非法等)
    cudaDeviceSynchronize();                         // 等 kernel 跑完(执行期错误在这里暴露)
}