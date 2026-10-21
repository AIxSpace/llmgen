const std = @import("std");

const types = @import("../types.zig");

const Tensor = @import("../Tensor.zig");

const base = @import("base.zig");

const module = @import("module.zig");
const Ctx = module.Ctx;

pub const HeadKind = enum {
    query,
    kv,
};

pub fn HeadSplit(comptime kind: HeadKind) type {
    return struct {
        allocator: std.mem.Allocator,

        layer_id: i32,
        name: []const u8,

        n_head: i64,
        head_dim: i64,

        const Self = @This();

        pub fn init(ctx: Ctx) !Self {
            const n_head = ctx.config_params.n_head;
            const head_kv_count = ctx.config_params.head_count_kv;
            const head_dim = ctx.config_params.head_dim;
            const spec_entry = ctx.find_spec(ctx.spec_key);

            return .{
                .allocator = ctx.trace.node_allocator,
                .layer_id = ctx.layer_id,
                .name = try ctx.trace.name(spec_entry.tag, ctx.layer_id),
                .n_head = switch (kind) {
                    .query => n_head,
                    .kv => head_kv_count,
                },
                .head_dim = head_dim,
            };
        }

        pub fn forward(self: *Self, x: *Tensor) !*Tensor {
            // [1, T, n*hd] → reshape [1, T, n, hd]:零拷贝视图
            return reshape(
                self.allocator,
                x,
                &.{ 1, base.DYN, self.n_head, self.head_dim },
                self.name,
                self.layer_id,
            );
        }
    };
}

// 类似 pytorch 中的 view
/// reshape:[B,M,N] -> [B,M,X,Y](X*Y = N)。建 TENSOR_TYPE_VIEW 节点, 共享底层数据(零拷贝, Executor 走 executeView)。
/// new_shape 里至多一个 -1:源含动态维时保持 -1, 否则按元素数推断。
pub fn reshape(allocator: std.mem.Allocator, input: *Tensor, new_shape_in: []const i64, name: []const u8, layer_id: i32) !*Tensor {
    var new_shape = [_]i64{ 0, 0, 0, 0 };
    if (new_shape_in.len > types.TENSOR_MAX_DIMS) return error.ReshapeTooManyDims;
    for (new_shape_in, 0..) |d, i| new_shape[i] = d;

    var src_elements: usize = 1;
    var src_has_dynamic = false;
    for (input.dims) |d| {
        if (d == 0) continue;
        if (d == -1) src_has_dynamic = true else src_elements *= @intCast(d);
    }
    var known_elements: usize = 1;
    var infer_idx: ?usize = null;
    for (new_shape_in, 0..) |d, i| {
        if (d == -1) infer_idx = i else known_elements *= @intCast(d);
    }
    if (infer_idx) |idx| {
        if (src_has_dynamic) {
            new_shape[idx] = -1; // 保持动态
        } else {
            if (src_elements % known_elements != 0) return error.ReshapeInvalid;
            new_shape[idx] = @intCast(src_elements / known_elements);
        }
    }
    const t = try base.newTensor(allocator, .{
        .name = name,
        .layer_id = layer_id,
        .data_type = input.data_type,
        .data = input.data, // 共享内存(零拷贝)
        .offset = input.offset,
        .dims = new_shape,
        .tensor_type = .TENSOR_TYPE_VIEW,
        .op_type = .OP_TYPE_RESHAPE,
        .src = base.srcs(&.{input}),
    });

    base.computeStrides(t);

    return t;
}

/// permute:perm[i] = 新第 i 维取自源第 perm[i] 维;perm 存进 op_params 给 kernel 读。
pub fn permute(allocator: std.mem.Allocator, input: *Tensor, perm: []const usize, name: []const u8, layer_id: i32) !*Tensor {
    var src_dims: [types.TENSOR_MAX_DIMS]i64 = undefined;
    var src_ndim: usize = 0;
    for (input.dims) |d| {
        if (d != 0) {
            src_dims[src_ndim] = d;
            src_ndim += 1;
        }
    }
    if (perm.len != src_ndim) return error.PermuteDimMismatch;
    var dims = [_]i64{ 0, 0, 0, 0 };
    var params = [_]f32{0} ** types.TENSOR_MAX_OP_PARAMS;
    for (perm, 0..) |p, i| {
        if (p >= src_ndim) return error.PermuteIndexOutOfRange;
        dims[i] = src_dims[p];
        params[i] = @floatFromInt(p);
    }
    const t = try base.newTensor(allocator, .{
        .name = name,
        .layer_id = layer_id,
        .data_type = input.data_type,
        .dims = dims,
        .tensor_type = .TENSOR_TYPE_ACTIVATION,
        .op_type = .OP_TYPE_PERMUTE,
        .op_params = params,
        .src = base.srcs(&.{input}),
    });
    base.computeStrides(t);
    return t;
}

/// 先 reshape(视图节点 name_reshape)再 permute(kernel 节点 name_permute), 返回后者。
pub fn reshapePermute(allocator: std.mem.Allocator, input: *Tensor, new_shape: []const i64, new_perm: []const usize, name: []const u8, layer_id: i32) !*Tensor {
    const reshaped = try reshape(
        allocator,
        input,
        new_shape,
        try std.fmt.allocPrint(allocator, "{s}_reshape", .{name}),
        layer_id,
    );
    return permute(
        allocator,
        reshaped,
        new_perm,
        try std.fmt.allocPrint(allocator, "{s}_permute", .{name}),
        layer_id,
    );
}

/// 先 permute(kernel 节点 name_permute)再 reshape(视图节点 name_reshape), 返回后者。
pub fn permuteReshape(allocator: std.mem.Allocator, input: *Tensor, new_perm: []const usize, new_shape: []const i64, name: []const u8, layer_id: i32) !*Tensor {
    const result = try permute(
        allocator,
        input,
        new_perm,
        try std.fmt.allocPrint(allocator, "{s}_permute", .{name}),
        layer_id,
    );
    return reshape(
        allocator,
        result,
        new_shape,
        try std.fmt.allocPrint(allocator, "{s}_reshape", .{name}),
        layer_id,
    );
}
