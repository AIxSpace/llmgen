const Tensor = @import("../../../Tensor.zig");
const common = @import("../../cpu/common.zig");

pub extern fn cuda_sample(
    out_ids: ?*anyopaque,
    logits: ?*const anyopaque,
    temps: ?*const anyopaque,
    seed: ?*const anyopaque,
    S: c_int,
    V: c_int,
    scratch: ?*anyopaque,
) void;

pub extern fn cuda_sample_scratch_bytes(max_S: c_int) usize;

pub const SampleImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const logits = out_tensor.src[0].?; // [1, S, V]
        cuda_sample(
            out_tensor.data,
            logits.data,
            common.g_step.temps,
            common.g_step.seed,
            common.g_step.n_seq,
            @intCast(logits.dims[2]),
            common.g_step.sample_scratch,
        );
    }
};

// backend/cpu/sample.zig 是同一个算法的 CPU 版, 见"CPU 同步"一节
