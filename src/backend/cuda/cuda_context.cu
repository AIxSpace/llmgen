#include "utils.cuh"
#include <cublas_v2.h>

// static bool g_inited = false;

cudaStream_t global_cuda_stream = nullptr;
cublasHandle_t global_cublas_handle = nullptr;

static void *global_cublas_workspace = nullptr;

static cudaGraph_t global_capture_graph = nullptr;

extern "C" int cuda_init(size_t ws_bytes) {
    if (global_cuda_stream) return 0;

    // cuda device
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        fprintf(stderr, "[serve] no CUDA device\n");
        return 1;
    }
    cudaSetDevice(0);

    // setup global cuda stream
    checkCuda(cudaStreamCreateWithFlags(&global_cuda_stream, cudaStreamNonBlocking), "cudaStreamCreate");

    // setup cublas and workspace
    if (cublasCreate(&global_cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[serve] cublasCreate failed\n");
        return 2;
    }
    cublasSetStream(global_cublas_handle, global_cuda_stream);

    // cuda graph use workspace
    if (ws_bytes > 0) {
        cudaMalloc(&global_cublas_workspace, ws_bytes);
        cublasSetWorkspace(global_cublas_handle, global_cublas_workspace, ws_bytes); 
    }
    cublasSetMathMode(global_cublas_handle, CUBLAS_DEFAULT_MATH);

    return 0;
}

extern "C" int cuda_device_count(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
}

extern "C" void cuda_device_info(int dev, char *name, int name_cap, size_t *total_mem, int *sm) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        if (name && name_cap > 0) name[0] = '\0';
        if (total_mem) *total_mem = 0;
        if (sm) *sm = 0;
        return;
    }
    if (name && name_cap > 0) {
        strncpy(name, prop.name, name_cap - 1);
        name[name_cap - 1] = '\0';
    }
    if (total_mem) *total_mem = prop.totalGlobalMem;
    if (sm) *sm = prop.major * 10 + prop.minor;
}

// 当前设备的空闲 / 总显存(含本进程的 CUDA 上下文和别的进程已占用的部分);KV 预算按它算
extern "C" void cuda_mem_info(size_t *free_mem, size_t *total_mem) {
    if (cudaMemGetInfo(free_mem, total_mem) != cudaSuccess) {
        *free_mem = 0;
        *total_mem = 0;
    }
}

extern "C" void cuda_h2d(void *dst, const void *src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
}

// 同步 D2H:排进全局流再等它做完(测试 / 调试用)
extern "C" void cuda_d2h(void *dst, const void *src, size_t n) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, global_cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(global_cuda_stream));
}

extern "C" void cuda_h2d_async(void *dst, const void *src, size_t n) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, global_cuda_stream));
}
extern "C" void cuda_d2h_async(void *dst, const void *src, size_t n) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, global_cuda_stream));
}

extern "C" void cuda_d2d(void *dst, const void *src, size_t bytes) {
    checkCuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), "D2D");
}

// kernel 发射是异步的, 错误要等同步之后才能取到(测试里用)
extern "C" int cuda_last_error(void) {
    return (int)cudaGetLastError();
}

extern "C" void cuda_sync(void) { 
    CUDA_CHECK(cudaStreamSynchronize(global_cuda_stream)); 
}

// ---- CUDA Graph:在 global_cuda_stream 上录制 / 回放 ----
// ThreadLocal:只拦截本线程在录制期间的非法调用(同步、cudaMalloc 等), 其他线程不受影响
extern "C" int cuda_graph_begin(void) {
    return (int)cudaStreamBeginCapture(global_cuda_stream, cudaStreamCaptureModeThreadLocal);
}

// 结束录制并实例化;返回 cudaGraphExec_t(原始的 cudaGraph_t 实例化后就不需要了)
extern "C" void *cuda_graph_end(void) {
    CUDA_CHECK(cudaStreamEndCapture(global_cuda_stream, &global_capture_graph));
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec, global_capture_graph, 0));
    cudaGraphDestroy(global_capture_graph);
    global_capture_graph = nullptr;
    return (void *)exec;
}

extern "C" void cuda_graph_launch(void *exec) {
    CUDA_CHECK(cudaGraphLaunch((cudaGraphExec_t)exec, global_cuda_stream));
}

extern "C" void cuda_graph_destroy(void *exec) {
    if (exec) cudaGraphExecDestroy((cudaGraphExec_t)exec);
}