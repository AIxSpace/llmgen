const types = @import("../../../types.zig");

const Tensor = @import("../../../Tensor.zig");

const utils = @import("../utils.zig");

pub extern fn cuda_add(output: ?*anyopaque, input1: ?*const anyopaque, input2: ?*const anyopaque, n: usize) void;

pub const AddImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        const y = out_tensor.src[1].?;
        const num = out_tensor.numElements();

        if (y.numElements() != num) {
            return error.UnsupportedBroadcast;
        }

        if (out_tensor.data_type != .GGML_TYPE_BF16 or num % 8 != 0) {
            return error.ElementwiseNeedsBf16x8;
        }

        cuda_add(out_tensor.data, x.data, y.data, num);
    }
};

pub extern fn cuda_mul(output: ?*anyopaque, input1: ?*const anyopaque, input2: ?*const anyopaque, n: usize) void;

pub const MulImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        const y = out_tensor.src[1].?;
        const num = out_tensor.numElements();

        if (y.numElements() != num) {
            return error.UnsupportedBroadcast;
        }

        if (out_tensor.data_type != .GGML_TYPE_BF16 or num % 8 != 0) {
            return error.ElementwiseNeedsBf16x8;
        }

        cuda_mul(out_tensor.data, x.data, y.data, num);
    }
};
