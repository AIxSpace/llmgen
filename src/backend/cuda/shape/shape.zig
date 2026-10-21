const types = @import("../../../types.zig");
const Tensor = @import("../../../Tensor.zig");

const utils = @import("../utils.zig");

pub extern fn cuda_permute(
    out: ?*anyopaque,
    in: ?*const anyopaque,
    dtype: c_int,
    in_strides: [*]const u64,
    out_strides: [*]const u64,
    axes: [*]const i64,
    ndim: c_int,
    numel: usize,
    device_id: c_int,
) void;

// 纯视图, 共享数据 + 重算行主序 strides, 不发射 kernel
pub const ReshapeImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        out_tensor.data = x.data;
        out_tensor.offset = x.offset;
        var stride: u64 = 1;
        var i: usize = types.TENSOR_MAX_DIMS;
        while (i > 0) {
            i -= 1;
            if (out_tensor.dims[i] == 0) {
                out_tensor.strides[i] = 0;
            } else {
                out_tensor.strides[i] = stride;
                stride *= @intCast(out_tensor.dims[i]);
            }
        }
    }
};

// 有效维数取自 x.dims, axes 取自 op_params, strides 取自 x / out.
pub const PermuteImpl = struct {
    pub fn execute(device_id: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?; // [batch,seq,head,dim]

        // 有效维度数
        var ndim: usize = 0;
        while (ndim < types.TENSOR_MAX_DIMS and x.dims[ndim] != 0) ndim += 1;

        var axes: [types.TENSOR_MAX_DIMS]i64 = undefined;
        for (0..ndim) |i| axes[i] = @intFromFloat(out_tensor.op_params[i]);

        cuda_permute(
            out_tensor.data,
            x.data,
            utils.dataTypeEnumRawValue(out_tensor.data_type),
            &x.strides,
            &out_tensor.strides,
            &axes,
            @intCast(ndim),
            out_tensor.numElements(),
            device_id,
        );
    }
};
