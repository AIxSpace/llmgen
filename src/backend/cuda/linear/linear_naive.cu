#include "linear_common.cuh"

// __global__ void linear_naive_kernel(float *y, const float *x, const float *wf32,
//                                     int seq, int in_dim, int out_dim) {
//     int o = blockIdx.x * blockDim.x + threadIdx.x; // 输出通道
//     int s = blockIdx.y;                            // 序列位置
//     if (o >= out_dim || s >= seq) return;
//     float acc = 0.f;
//     const float *xr = x + (long)s * in_dim;
//     const float *wr = wf32 + (long)o * in_dim;
//     for (int i = 0; i < in_dim; ++i) acc += xr[i] * wr[i]; // 与 CPU 同序
//     y[(long)s * out_dim + o] = acc;
// }

// extern "C" void cuda_linear(float *y_dev, const float *x_dev, const void *w_dev, int wdtype,
//                             int seq, int in_dim, int out_dim) {
//     cuda_init();

//     long wn = (long)out_dim * in_dim;
//     float *wf32 = nullptr;
//     cudaMalloc(&wf32, wn * sizeof(float));
//     int tb = 256;

//     dequant_weight_kernel<<<(unsigned)((wn + tb - 1) / tb), tb>>>(wf32, w_dev, wdtype, wn);

//     dim3 block_size(256, 1);
//     dim3 grid_size((out_dim + 255) / 256, seq);
 
//     // 朴素版本                                
//     linear_naive_kernel<<<grid_size, block_size>>>(y_dev, x_dev, wf32, seq, in_dim, out_dim);

//     cudaDeviceSynchronize();
    
//     cudaFree(wf32);
// }