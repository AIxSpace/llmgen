pub const EmbeddingImpl = @import("embedding/embedding.zig").EmbeddingImpl;
pub const AddImpl = @import("arithmetic/arithmetic.zig").AddImpl;
pub const MulImpl = @import("arithmetic/arithmetic.zig").MulImpl;
pub const SiluImpl = @import("activation/activation.zig").SiluImpl;

pub const ReshapeImpl = @import("shape/shape.zig").ReshapeImpl;
pub const PermuteImpl = @import("shape/shape.zig").PermuteImpl;

pub const ApplyRopeImpl = @import("rope/rope.zig").ApplyRopeImpl;
pub const RmsNormImpl = @import("norm/norm.zig").RmsNormImpl;
pub const LinearImpl = @import("linear/linear.zig").LinearImpl;

pub const PagedAttentionImpl = @import("attention/attention_v1.zig").PagedAttentionImpl;
pub const FlashAttentionImpl = @import("attention/attention_v1.zig").FlashAttentionImpl;
pub const SoftmaxImpl = @import("softmax/softmax.zig").SoftmaxImpl;
pub const MemcpyImpl = @import("memory/memcpy.zig").MemcpyImpl;

pub const memory = @import("memory/memory.zig");
pub const cuda_host_alloc = memory.cuda_host_alloc;
pub const cuda_host_free = memory.cuda_host_free;
pub const cuda_h2d = memory.cuda_h2d; //  同步 H2D(加载权重用)
pub const cuda_h2d_async = memory.cuda_h2d_async;
pub const cuda_d2h_async = memory.cuda_d2h_async;
pub const cuda_sample_scratch_bytes = @import("sample/sample.zig").cuda_sample_scratch_bytes;

pub const GatherLastImpl = @import("memory/memcpy.zig").GatherLastImpl;
pub const SampleImpl = @import("sample/sample.zig").SampleImpl;

pub const CudaContext = @import("cuda_context.zig").CudaContext;

pub const cuda_graph_begin = @import("cuda_context.zig").cuda_graph_begin;
pub const cuda_graph_end = @import("cuda_context.zig").cuda_graph_end;
pub const cuda_graph_launch = @import("cuda_context.zig").cuda_graph_launch;
pub const cuda_graph_destroy = @import("cuda_context.zig").cuda_graph_destroy;

// todo: - 下面的方法还有用吗？没用删了 - 只在测试中用了

/// H2D:dst 是设备 []u8, src 是 host []u8。
pub fn cudaH2D(dst_dev: []u8, src_host: []const u8) void {
    _ = CudaContext.instance();
    memory.cuda_h2d(
        dst_dev.ptr,
        src_host.ptr,
        @min(dst_dev.len, src_host.len),
    );
}
/// D2H:dst 是 host []u8, src 是设备 []u8。
pub fn cudaD2H(dst_host: []u8, src_dev: []const u8) void {
    _ = CudaContext.instance();
    memory.cuda_d2h(
        dst_host.ptr,
        src_dev.ptr,
        @min(dst_host.len, src_dev.len),
    );
}
/// D2D:分页 KV 写入用(对照 C++ device_memcpy)。
pub fn cudaD2D(dst_dev: []u8, src_dev: []const u8) void {
    memory.cuda_d2d(
        dst_dev.ptr,
        src_dev.ptr,
        @min(dst_dev.len, src_dev.len),
    );
}
