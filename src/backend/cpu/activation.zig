const std = @import("std");

const common = @import("common.zig");

const types = @import("../../types.zig");
const Tensor = @import("../../Tensor.zig");

pub const SiluImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        const num = out_tensor.numElements();

        for (0..num) |i| {
            const v = common.elemF32(x, i);
            const value = v / (1.0 + @exp(-v));
            common.storeF32(out_tensor, i, value);
        }
    }
};
