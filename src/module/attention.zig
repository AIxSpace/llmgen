const std = @import("std");

const types = @import("../types.zig");

const Tensor = @import("../Tensor.zig");

const gguf = @import("../gguf.zig");
const TensorInfo = gguf.TensorInfo;

const base = @import("base.zig");
const shape = @import("shape.zig");

const module = @import("module.zig");
const Ctx = module.Ctx;

pub const Attention = struct {
    allocator: std.mem.Allocator,

    layer_id: i32,
    q_dim: i64,

    name_attn: []const u8,
    name_attn_traspose: []const u8,

    op: types.OperationType,

    const Self = @This();

    pub fn init(ctx: Ctx) !Attention {
        return .{
            .allocator = ctx.trace.node_allocator,
            .layer_id = ctx.layer_id,
            .q_dim = ctx.config_params.n_head * ctx.config_params.head_dim,
            .name_attn = try ctx.trace.name("attn", ctx.layer_id),
            .name_attn_traspose = try ctx.trace.name("attn_tr", ctx.layer_id),
            .op = ctx.attn_op,
        };
    }

    pub fn forward(self: *Self, q: *Tensor, k: *Tensor, v: *Tensor) !*Tensor {
        const attn = try attention(
            self.allocator,
            q,
            k,
            v,
            null,
            -1,
            true,
            0,
            self.name_attn,
            self.layer_id,
            self.op,
        );

        // [1, T, n_head, hd] 本来就是 token-major, reshape 成 [1, T, q_dim] 只是换个看法
        return shape.reshape(
            self.allocator,
            attn,
            &.{ 1, base.DYN, self.q_dim },
            self.name_attn_traspose,
            self.layer_id,
        );
    }
};

pub fn attention(
    allocator: std.mem.Allocator,
    q_rope: *Tensor,
    k_rope: *Tensor,
    v_4d: *Tensor,
    mask: ?*Tensor,
    scale: f32,
    causal: bool,
    num_kv_groups_in: i32,
    name: []const u8,
    layer_id: i32,
    op_type: types.OperationType,
) !*Tensor {
    const n_heads = q_rope.dims[2]; // [1, T, n, hd]
    const n_kv_heads = k_rope.dims[2];
    if (n_kv_heads == 0 or @rem(n_heads, n_kv_heads) != 0) {
        return error.HeadsNotDivisible;
    }

    const num_kv_gropus = if (num_kv_groups_in < 1) @as(i32, @intCast(@divTrunc(n_heads, n_kv_heads))) else num_kv_groups_in;
    const head_dim = q_rope.dims[3];

    var params = [_]f32{0} ** types.TENSOR_MAX_OP_PARAMS;
    params[0] = @floatFromInt(head_dim);
    params[1] = if (scale < 0) 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))) else scale;
    params[2] = if (causal) 1 else 0;
    params[3] = @floatFromInt(num_kv_gropus);

    const t = try base.newTensor(allocator, .{
        .name = name,
        .data_type = q_rope.data_type,
        .layer_id = layer_id,
        .src = blk: {
            var s = base.srcs(&.{ q_rope, k_rope, v_4d });
            s[3] = mask;
            break :blk s;
        },
        .tensor_type = .TENSOR_TYPE_ACTIVATION,
        .op_type = op_type,
        .op_params = params,
        .dims = q_rope.dims,
    });

    base.computeStrides(t);

    return t;
}
