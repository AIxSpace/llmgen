#include "cuda_common.cuh"

// extern "C" void cuda_linear_hosted(float *y_host, const float *x_host, const void *w_host, int wdtype,
//                                    int seq, int in_dim, int out_dim) {
//     cuda_init();

//     long xn = (long)seq * in_dim, yn = (long)seq * out_dim, wn = (long)out_dim * in_dim;

//     long wbytes = wn * (wdtype == GT_F32 ? 4 : 2);

//     float *xd; float *yd; void *wd;

//     cudaMalloc(&xd, xn * 4); 
//     cudaMalloc(&yd, yn * 4); 
//     cudaMalloc(&wd, wbytes);

//     cudaMemcpy(xd, x_host, xn * 4, cudaMemcpyHostToDevice);
//     cudaMemcpy(wd, w_host, wbytes, cudaMemcpyHostToDevice);

//     cuda_linear(yd, xd, wd, wdtype, seq, in_dim, out_dim);

//     cudaMemcpy(y_host, yd, yn * 4, cudaMemcpyDeviceToHost);

//     cudaFree(xd); 
//     cudaFree(yd); 
//     cudaFree(wd);
// }
