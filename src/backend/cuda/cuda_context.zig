pub extern fn cuda_init(cublas_workspace_bytes: usize) c_int;
pub extern fn cuda_sync() void;
pub extern fn cuda_last_error() c_int; //               cudaGetLastError, 0 = 没有错误

pub extern fn cuda_graph_begin() c_int; //  cudaStreamBeginCapture, 0 = 成功
pub extern fn cuda_graph_end() ?*anyopaque; //  EndCapture + Instantiate → cudaGraphExec_t
pub extern fn cuda_graph_launch(exec: ?*anyopaque) void;
pub extern fn cuda_graph_destroy(exec: ?*anyopaque) void;

extern fn cuda_device_count() c_int;
extern fn cuda_mem_info(free_mem: *usize, total_mem: *usize) void;
extern fn cuda_device_info(dev: c_int, name: [*c]u8, name_cap: c_int, total_mem: *usize, sm: *c_int) void;

pub const CudaContext = struct {
    inited: bool = false,

    var g_inst: ?CudaContext = null;

    pub fn instance() *CudaContext {
        if (g_inst == null) {
            g_inst = .{};
            _ = cuda_init(32 << 20); // cuBLAS 固定 workspace:录 CUDA Graph 期间 cuBLAS 不能自己 cudaMalloc
            g_inst.?.inited = true;
        }
        return &g_inst.?;
    }

    pub fn deviceCount(_: *CudaContext) i32 {
        return @intCast(cuda_device_count());
    }

    pub fn deviceInfo(_: *CudaContext, dev: i32, name_buf: []u8) struct { total_mem: usize, sm: i32 } {
        var mem: usize = 0;
        var sm: c_int = 0;
        cuda_device_info(
            @intCast(dev),
            name_buf.ptr,
            @intCast(name_buf.len),
            &mem,
            &sm,
        );

        return .{
            .total_mem = mem,
            .sm = @intCast(sm),
        };
    }

    pub fn sync(_: *CudaContext) void {
        cuda_sync();
    }

    /// 已用显存 = 总量 - 空闲(cudaMemGetInfo, 当前设备)。包含 CUDA 上下文本身和别的进程
    pub fn usedMemory(_: *CudaContext) usize {
        var free_mem: usize = 0;
        var total_mem: usize = 0;
        cuda_mem_info(&free_mem, &total_mem);
        return total_mem - free_mem;
    }
};
