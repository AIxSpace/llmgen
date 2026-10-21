const std = @import("std");

const types = @import("../../types.zig");
const Tensor = @import("../../Tensor.zig");

const common = @import("common.zig");

pub const MemcpyImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        // binary(out_tensor, mul);
        const src = out_tensor.src[0].?; // copy src to out_tensor
        const n = out_tensor.bytes();
        @memcpy(out_tensor.data.?[0..n], src.data.?[0..n]);
    }
};

// ---- src/backend/cpu/memcpy.zig(派发表要求每个算子都有 CPU 特化) ----
pub const GatherLastImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        const s = &common.g_step;
        const row = @as(usize, @intCast(s.hidden)) * x.data_type.size();
        const last: [*]const i32 = @ptrCast(@alignCast(s.last.?));
        for (0..@intCast(s.n_seq)) |i| {
            const from = @as(usize, @intCast(last[i])) * row;
            @memcpy(out_tensor.data.?[i * row ..][0..row], x.data.?[from..][0..row]);
        }
    }
};
