const std = @import("std");

const types = @import("../types.zig");
const TensorType = types.TensorType;
const GGMLDataType = types.GGMLDataType;

const Tensor = @import("../Tensor.zig");

const gguf = @import("../gguf.zig");

pub const DYN: i64 = -1; // 张量动态维度标记
pub const DYN_S: i64 = -2; // 第二个动态维:本步的序列数 S(DYN 是本步的 token 总数 T)

pub fn placeholder(allocator: std.mem.Allocator, name: []const u8, layer_id: i32, tensor_type: TensorType, data_type: GGMLDataType, dims: [4]i64) !*Tensor {
    const t = try newTensor(
        allocator,
        .{
            .name = name,
            .data_type = data_type,
            .layer_id = layer_id,
            .tensor_type = tensor_type,
            .op_type = .OP_TYPE_NONE,
            .dims = dims,
        },
    );
    computeStrides(t);
    return t;
}

pub fn weightPlaceholder(allocator: std.mem.Allocator, tensor_info: *const gguf.TensorInfo, name: []const u8, layer_id: i32) !*Tensor {
    const t = try newTensor(
        allocator,
        .{
            .name = name,
            .data_type = tensor_info.dtype,
            .offset = tensor_info.offset,
            .layer_id = layer_id,
            .tensor_type = .TENSOR_TYPE_WEIGHT,
            .op_type = .OP_TYPE_NONE,
            .dims = tensor_info.dimensions,
        },
    );
    computeStrides(t);
    return t;
}

pub fn inferLinearOutput(input_dims: [4]i64, weight_dims: [4]i64, transpose: bool) ![4]i64 {
    var out = [_]i64{ 0, 0, 0, 0 };
    // 第 1 步：从后往前找 input 第一个非 0 的位置，得到 input 的实际维度数 ndim
    var ndim: usize = 0;
    var i: usize = types.TENSOR_MAX_DIMS;
    while (i > 0) {
        i -= 1;
        if (input_dims[i] != 0) {
            ndim = i + 1;
            break;
        }
    }

    if (ndim == 0) return error.EmptyInputShape;

    // 第 2 步：权重的输出维度 = weight_dims[0]
    var weight_out = weight_dims[0];
    var weight_in = weight_dims[1];

    if (!transpose) {
        std.mem.swap(i64, &weight_out, &weight_in);
    }

    // x [seq=17, in=1024], weight_dims [1024, 4096], 1024 必须对应起来
    const last_dim = input_dims[ndim - 1];
    if (last_dim > 0 and weight_in > 0 and last_dim != weight_in) {
        return error.LinearDimensionMismatch;
    }

    for (0..ndim - 1) |k| {
        out[k] = input_dims[k];
    }
    out[ndim - 1] = weight_out;
    return out;
}

pub fn elementwise(allocator: std.mem.Allocator, name: []const u8, layer_id: i32, op_type: types.OperationType, dims: [4]i64, sources: []const *Tensor) !*Tensor {
    const t = try newTensor(allocator, .{
        .name = name,
        .data_type = sources[0].data_type,
        .layer_id = layer_id,
        .src = srcs(sources),
        .tensor_type = .TENSOR_TYPE_ACTIVATION,
        .op_type = op_type,
        .dims = dims,
    });
    computeStrides(t);
    return t;
}

// 这个函数的作用是根据张量的维度 dims 计算其步长 strides。步长(stride)描述的是:在内存中沿某一维度移动一个单位时,需要跳过的元素个数(注意这里是元素个数,不是字节数)
// 例子 1:embd,dims = [17, 1024, 0, 0]

// 从右往左:

// ┌─────┬─────────┬──────────────────┬────────────┬───────────────┐
// │  i  │ dims[i] │       计算       │ strides[i] │  累积 stride  │
// ├─────┼─────────┼──────────────────┼────────────┼───────────────┤
// │ 3   │ 0       │ 跳过             │ 0          │ 1             │
// ├─────┼─────────┼──────────────────┼────────────┼───────────────┤
// │ 2   │ 0       │ 跳过             │ 0          │ 1             │
// ├─────┼─────────┼──────────────────┼────────────┼───────────────┤
// │ 1   │ 1024    │ 写入 stride=1    │ 1          │ 1×1024 = 1024 │
// ├─────┼─────────┼──────────────────┼────────────┼───────────────┤
// │ 0   │ 17      │ 写入 stride=1024 │ 1024       │ 1024×17       │
// └─────┴─────────┴──────────────────┴────────────┴───────────────┘

// 得 strides = [1024, 1, 0, 0]。含义:

// - 第 0 维(token 行)步长 1024 → 跨到下一个 token 要跳 1024 个元素(一整行 hidden);
// - 第 1 维(hidden 通道)步长 1 → 相邻通道紧挨着。

// 于是元素 (r, c) 的线性下标 = r*1024 + c*1,正是行主序连续排布。

// 例子 2:logits,dims = [17, 151936, 0, 0]

// 同样从右往左:strides[1] = 1,strides[0] = 151936。

// todo: - compute strides 不是每个地方都必须的 ?
pub fn computeStrides(t: *Tensor) void {
    var stride: u64 = 1;
    var index: usize = types.TENSOR_MAX_DIMS;
    while (index > 0) {
        index -= 1;
        if (t.dims[index] == 0) {
            t.strides[index] = 0;
        } else {
            t.strides[index] = stride;
            stride *= @abs(t.dims[index]);
        }
    }
}

pub fn srcs(list: []const *Tensor) [types.TENSOR_MAX_SRC]?*Tensor {
    var src = [_]?*Tensor{null} ** types.TENSOR_MAX_SRC;
    for (list, 0..) |t, i| {
        src[i] = t;
    }
    return src;
}

pub fn newTensor(allocator: std.mem.Allocator, tensor: Tensor) !*Tensor {
    const t = try allocator.create(Tensor);
    errdefer allocator.destroy(t);
    t.* = tensor;
    return t;
}
