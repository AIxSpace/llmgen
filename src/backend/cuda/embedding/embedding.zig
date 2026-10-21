const std = @import("std");

const Tensor = @import("../../../Tensor.zig");

const utils = @import("../utils.zig");

pub extern fn cuda_embedding(output: ?*anyopaque, input_ids: ?*const anyopaque, weight: ?*const anyopaque, T: c_int, H: c_int) void;

pub const EmbeddingImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const input_ids = out_tensor.src[0].?;
        const weight = out_tensor.src[1].?;

        if (out_tensor.data_type != .GGML_TYPE_BF16 or weight.data_type != .GGML_TYPE_BF16) {
            return error.EmbeddingNeedsBf16;
        }

        const hidden = weight.dims[1];
        if (@mod(hidden, 8) != 0) {
            // kernel 按 bf16x8 也就是 16 字节一次搬运，H/8 是整除，所以要先检查 hidden 能被 8 整除
            return error.EmbeddingHiddenNotMultipleOf8; // kernel 按 bf16x8 搬
        }

        cuda_embedding(
            out_tensor.data,
            input_ids.data,
            weight.data,
            @intCast(out_tensor.dims[1]),
            @intCast(hidden),
        );
    }
};
