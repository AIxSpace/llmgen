const types = @import("../../../types.zig");
const Tensor = @import("../../../Tensor.zig");

const utils = @import("../utils.zig");

pub extern fn cuda_softmax(
    out: ?*anyopaque,
    in: ?*const anyopaque,
    dtype: c_int,
    outer_dim: usize,
    axis_dim: usize,
    inner_dim: usize,
    numel: usize,
    dev_id: c_int,
) void;

/// 由 axis 拆出 outer / axis / inner 三段.
/// 本 lab 的 softmax 与 cpu/softmax.zig 一致恒沿末维(C++ 从 op_params[0] 取 axis).
pub const SoftmaxImpl = struct {
    pub fn execute(device_id: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        const dims = x.rank();
        const axis = dims - 1;
        var outer_dim: usize = 1;
        var inner_dim: usize = 1;
        const axis_dim: usize = @intCast(x.dims[axis]);

        for (0..axis) |i| {
            outer_dim *= @intCast(x.dims[i]);
        }

        for (axis + 1..dims) |i| {
            inner_dim *= @intCast(x.dims[i]);
        }

        cuda_softmax(
            out_tensor.data,
            x.data,
            utils.dataTypeEnumRawValue(out_tensor.data_type),
            outer_dim,
            axis_dim,
            inner_dim,
            out_tensor.numElements(),
            device_id,
        );
    }
};
