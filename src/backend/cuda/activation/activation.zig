const types = @import("../../../types.zig");
const Tensor = @import("../../../Tensor.zig");
const utils = @import("../utils.zig");

pub extern fn cuda_silu(output: ?*anyopaque, x: ?*const anyopaque, n: usize) void;

pub const SiluImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;

        const num = out_tensor.numElements();

        if (out_tensor.data_type != .GGML_TYPE_BF16 or num % 8 != 0) {
            return error.ElementwiseNeedsBf16x8;
        }

        cuda_silu(out_tensor.data, x.data, num);
    }
};
