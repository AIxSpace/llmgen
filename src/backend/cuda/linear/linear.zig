const types = @import("../../../types.zig");

const Tensor = @import("../../../Tensor.zig");

const utils = @import("../utils.zig");

pub extern fn cuda_linear(
    y: ?*anyopaque,
    y_f32: c_int,
    x: ?*const anyopaque,
    w: ?*const anyopaque,
    M: c_int,
    N: c_int,
    K: c_int,
) void;

/// src[0]=x[B,M,K](B=1), src[1]=W[N, K]
pub const LinearImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        const w = out_tensor.src[1].?;

        if (x.data_type != .GGML_TYPE_BF16 or w.data_type != .GGML_TYPE_BF16) return error.LinearNeedsBf16;

        const y_f32: c_int = @intFromBool(out_tensor.data_type == .GGML_TYPE_F32);

        cuda_linear(
            out_tensor.data,
            y_f32,
            x.data,
            w.data,
            @intCast(x.dims[1]),
            @intCast(w.dims[0]),
            @intCast(w.dims[1]),
        );
    }
};

pub const MatmulImpl = struct {
    pub fn execute(out: *Tensor, dev_id: i32) !void {
        _ = out;
        _ = dev_id;
        return error.NotImplemented;
    }
};
