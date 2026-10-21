const types = @import("../../../types.zig");

const Tensor = @import("../../../Tensor.zig");

const utils = @import("../utils.zig");

const common = @import("../../cpu/common.zig");

pub extern fn cuda_rope(
    out: ?*anyopaque,
    x: ?*const anyopaque,
    cos_tab: ?*const anyopaque,
    sin_tab: ?*const anyopaque,
    positions: ?*const anyopaque,
    T: c_int,
    n_head: c_int,
    hd: c_int,
) void;

/// src[0] = x [1, T, n_head, hd] bf16, src[1] = cos / src[2] = sin [max_seq, hd] f32(显存)
pub const ApplyRopeImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        const cos = out_tensor.src[1].?;
        const sin = out_tensor.src[2].?;

        if (x.data_type != .GGML_TYPE_BF16) {
            return error.RopeNeedsBf16;
        }

        cuda_rope(
            out_tensor.data,
            x.data,
            cos.data,
            sin.data,
            common.g_step.pos,
            common.g_step.n_tok,
            @intCast(x.dims[2]),
            @intCast(x.dims[3]),
        );
    }
};
