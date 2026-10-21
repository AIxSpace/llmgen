const std = @import("std");

const Tensor = @import("../Tensor.zig");

const base = @import("base.zig");

/// x [1, T, H] → [1, S, H]:每条序列取它本步最后一个 token 的那一行。
/// 取哪一行在执行期由元数据的 last[S] 决定。
pub fn gatherLast(
    allocator: std.mem.Allocator,
    x: *Tensor,
    name: []const u8,
    layer_id: i32,
) !*Tensor {
    const t = try base.newTensor(allocator, .{
        .name = name,
        .layer_id = layer_id,
        .data_type = x.data_type,
        .tensor_type = .TENSOR_TYPE_ACTIVATION,
        .op_type = .OP_TYPE_GATHER_LAST,
        .src = base.srcs(&.{x}),
        .dims = .{ 1, base.DYN_S, x.dims[2], 0 },
    });
    base.computeStrides(t);
    return t;
}

/// logits [1, S, V] bf16 → token id [1, S] i32。温度和随机种子在执行期从元数据读。
pub fn sample(allocator: std.mem.Allocator, logits: *Tensor, name: []const u8, layer_id: i32) !*Tensor {
    const t = try base.newTensor(allocator, .{
        .name = name,
        .layer_id = layer_id,
        .data_type = .GGML_TYPE_I32,
        .tensor_type = .TENSOR_TYPE_OUTPUT,
        .op_type = .OP_TYPE_SAMPLE,
        .src = base.srcs(&.{logits}),
        .dims = .{ 1, base.DYN_S, 0, 0 },
    });
    base.computeStrides(t);
    return t;
}
