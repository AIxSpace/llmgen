#include "linear_common.cuh"

// extern "C" void cuda_linear(float *y_dev, const float *x_dev, const void *w_dev, int wdtype,
//                             int seq, int in_dim, int out_dim) {
//     cuda_init();
    
//     long wn = (long)out_dim * in_dim;
//     float *wf32 = nullptr;
//     cudaMalloc(&wf32, wn * sizeof(float));
    
//     int tb = 256;
//     dequant_weight_kernel<<<(unsigned)((wn + tb - 1) / tb), tb>>>(wf32, w_dev, wdtype, wn);

//     // cuBLAS 列主序: 行主 A[m,n] == 列主 A[n,m].
//     //   C_cm[out,seq] = op(A=Wf32)^T[out,in] * op(B=x)[in,seq]
//     //   Wf32 行主[out,in]=列主[in,out], lda=in; x 行主[seq,in]=列主[in,seq], ldb=in; y 列主[out,seq]=行主[seq,out], ldc=out
//     float alpha = 1.f, beta = 0.f;
//     cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
//                 out_dim, seq, in_dim,
//                 &alpha, wf32, in_dim, x_dev, in_dim,
//                 &beta, y_dev, out_dim);

//     cudaDeviceSynchronize();
//     cudaFree(wf32);
// }

