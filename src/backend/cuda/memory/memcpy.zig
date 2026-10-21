const types = @import("../../../types.zig");

const Tensor = @import("../../../Tensor.zig");

const utils = @import("../utils.zig");

const common = @import("../../cpu/common.zig");

pub extern fn cuda_memcpy(
    dst: ?*anyopaque,
    src: ?*const anyopaque,
    nbytes: usize,
    src_device_type: c_int,
    dst_device_type: c_int,
    device_id: c_int,
) void;

pub const MemcpyImpl = struct {
    pub fn execute(device_id: i32, out_tensor: *Tensor) !void {
        const src = out_tensor.src[0] orelse return error.NoSource;
        const nbytes = out_tensor.bytes();

        cuda_memcpy(
            out_tensor.data,
            src.data,
            nbytes,
            utils.deviceTypeEnumRawValue(src.device_type),
            utils.deviceTypeEnumRawValue(out_tensor.device_type),
            device_id,
        );
    }
};

pub extern fn cuda_gather_rows(
    out: ?*anyopaque,
    x: ?*const anyopaque,
    idx: ?*const anyopaque,
    S: c_int,
    H: c_int,
) void;

/// out[s] = x[last[s]], 行宽 = hidden
pub const GatherLastImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        cuda_gather_rows(
            out_tensor.data,
            out_tensor.src[0].?.data,
            common.g_step.last,
            common.g_step.n_seq,
            common.g_step.hidden,
        );
    }
};
