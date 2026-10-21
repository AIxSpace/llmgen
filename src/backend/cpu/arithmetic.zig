const std = @import("std");

const common = @import("common.zig");

const types = @import("../../types.zig");
const Tensor = @import("../../Tensor.zig");

pub const MulImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        binary(out_tensor, mul);
    }

    fn mul(x: f32, y: f32) f32 {
        return x * y;
    }
};

pub const AddImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        binary(out_tensor, add);
    }

    fn add(x: f32, y: f32) f32 {
        return x + y;
    }
};

fn binary(out_tensor: *Tensor, comptime f: fn (f32, f32) f32) void {
    const a = out_tensor.src[0].?;
    const b = out_tensor.src[1].?;
    const n = out_tensor.numElements();
    const bn = b.numElements();

    for (0..n) |i| {
        const bi = if (bn == n) i else i % bn;
        const v = f(common.elemF32(a, i), common.elemF32(b, bi));
        common.storeF32(out_tensor, i, v);
    }
}
