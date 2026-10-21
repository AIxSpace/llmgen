const std = @import("std");
const types = @import("types.zig");

name: []const u8 = "",

data: ?[*]u8 = null,
data_type: types.GGMLDataType = .GGML_TYPE_F32,
offset: usize = 0,

layer_id: i32 = -1,
src: [types.TENSOR_MAX_SRC]?*Self = .{null} ** types.TENSOR_MAX_SRC,
tensor_type: types.TensorType = .TENSOR_TYPE_UNKNOWN,

op_type: types.OperationType = .OP_TYPE_NONE,
op_params: [types.TENSOR_MAX_OP_PARAMS]f32 = .{0} ** 4,

dims: [types.TENSOR_MAX_DIMS]i64 = .{0} ** 4,
strides: [types.TENSOR_MAX_DIMS]u64 = .{0} ** 4,

device_type: types.DeviceType = .CPU,
device_handle: usize = 0, // todo: - 无用，后续删除

const Self = @This();

// 非 0 前导维度的个数
pub fn rank(self: *const Self) usize {
    var r: usize = 0;
    for (self.dims) |d| {
        if (d == 0) break;
        r += 1;
    }
    return r;
}

pub fn isComputed(self: *const Self) bool {
    return self.op_type != .OP_TYPE_NONE and self.tensor_type != .TENSOR_TYPE_WEIGHT and self.tensor_type != .TENSOR_TYPE_INPUT;
}

/// 对照 bytes_at():把负维(-1 动态序列维)解析成 resolve 后算字节.
pub fn bytesAt(self: *const Self, resolve: i64) usize {
    var total: usize = 1;
    for (self.dims) |d| {
        if (d == 0) break;
        total *= @intCast(if (d < 0) resolve else d);
    }
    return total * self.data_type.size();
}

pub fn numElements(self: *const Self) usize {
    var total: usize = 1;
    for (self.dims) |d| {
        if (d == 0) break;
        total *= @abs(d); // 各维度相乘
    }
    return total;
}

pub fn bytes(self: *const Self) usize {
    return self.numElements() * self.data_type.size();
}

/// 两个动态维分别解析:DYN → n_tok, DYN_S → n_seq。建池时用它, 避免把 [S, vocab] 当成 [max_seq, vocab]
pub fn bytesAtTS(self: *const Self, n_tok: i64, n_seq: i64) usize {
    var total: usize = 1;
    for (self.dims) |d| {
        if (d == 0) break;
        total *= @intCast(if (d == -2) n_seq else if (d < 0) n_tok else d);
    }
    return total * self.data_type.size();
}
