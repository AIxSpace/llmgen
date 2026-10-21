const std = @import("std");
const common = @import("common.zig");

const types = @import("../../types.zig");
const Tensor = @import("../../Tensor.zig");

pub const SoftmaxImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        const r = out_tensor.rank(); // 非 0 维度个数

        const last = @as(usize, @intCast(out_tensor.dims[r - 1]));
        const rows = out_tensor.numElements() / last;

        for (0..rows) |row| {
            const base = row * last;
            var mx: f32 = -std.math.inf(f32);
            for (0..last) |j| {
                mx = @max(mx, common.elemF32(x, base + j));
            }
            var sum: f32 = 0;
            for (0..last) |j| {
                sum += @exp(common.elemF32(x, base + j) - mx);
            }
            for (0..last) |j| {
                // dest_data[base + j] /= sum;
                common.storeF32(out_tensor, base + j, @exp(common.elemF32(x, base + j) - mx) / sum);
            }
        }
    }
};
