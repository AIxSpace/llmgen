const std = @import("std");
const types = @import("../types.zig");
const Tensor = @import("../Tensor.zig");
const base = @import("base.zig");
const module = @import("module.zig");
const Ctx = module.Ctx;

/// RoPE 的 cos / sin 表(两个 CACHE 节点)。
pub const RopeTables = struct { cos: *Tensor, sin: *Tensor };

/// 模型级:建 [context_length, head_dim] 的 cos / sin 表(对照 module-style RopeCache / C++ OpFactory::rope_cache)。
/// 名字前缀取 spec 的 tag("rope" → rope_cos / rope_sin)。
pub const RopeCache = struct {
    allocator: std.mem.Allocator,
    prefix: []const u8,
    max_seq: i64,
    rope_dim: i64,
    theta: f32,
    layer_id: i32,

    pub fn init(ctx: Ctx) !RopeCache {
        return .{
            .allocator = ctx.trace.node_allocator,
            .prefix = ctx.find_spec(ctx.spec_key).tag,
            .max_seq = ctx.config_params.context_length,
            .rope_dim = ctx.config_params.head_dim,
            .theta = ctx.config_params.rope_freq_base,
            .layer_id = ctx.layer_id,
        };
    }

    pub fn forward(self: *RopeCache) !RopeTables {
        return ropeCache(self.allocator, self.max_seq, self.rope_dim, self.theta, self.prefix, self.layer_id);
    }
};

/// 层级:q / k 一起进, 各出一个 APPLY_ROPE 节点(v 不做);表来自构造时的 Ctx.rope。
pub const Rope = struct {
    allocator: std.mem.Allocator,
    tables: RopeTables,
    layer_id: i32,

    pub fn init(ctx: Ctx) !Rope {
        return .{
            .allocator = ctx.trace.node_allocator,
            .tables = ctx.rope orelse return error.NoRopeTables,
            .layer_id = ctx.layer_id,
        };
    }

    pub fn forward(self: *Rope, q: *Tensor, k: *Tensor) !struct { q: *Tensor, k: *Tensor } {
        return .{
            .q = try applyRope(self.allocator, q, self.tables, self.layer_id),
            .k = try applyRope(self.allocator, k, self.tables, self.layer_id),
        };
    }
};

/// op_params[0]=theta, [1]=rope_dim, [2]=max_seq_len;输出恒为 F32。由 RopeCacheImpl<CPU> 算一次。
pub fn ropeCache(allocator: std.mem.Allocator, max_seq_len: i64, rope_dim: i64, theta: f32, prefix: []const u8, layer_id: i32) !RopeTables {
    var params = [_]f32{0} ** types.TENSOR_MAX_OP_PARAMS;
    params[0] = theta;
    params[1] = @floatFromInt(rope_dim);
    params[2] = @floatFromInt(max_seq_len);
    var out: [2]*Tensor = undefined;
    for ([_][]const u8{ "cos", "sin" }, 0..) |kind, i| {
        const t = try base.newTensor(allocator, .{
            .name = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, kind }), // RopeCacheImpl 靠名字里的 "_cos" 区分
            .layer_id = layer_id,
            .data_type = .GGML_TYPE_F32,
            .dims = .{ max_seq_len, rope_dim, 0, 0 },
            .tensor_type = .TENSOR_TYPE_CACHE,
            .op_type = .OP_TYPE_ROPE_CACHE,
            .op_params = params,
        });
        base.computeStrides(t);
        out[i] = t;
    }
    return .{ .cos = out[0], .sin = out[1] };
}

/// src = [x, cos, sin];op_params[0]=head_dim, [1]=rope_dim, [2]=start_pos(Executor 每步改写)。
/// 节点名 = 输入名 ++ "_rope"(对照 C++)。
pub fn applyRope(allocator: std.mem.Allocator, x: *Tensor, tables: RopeTables, layer_id: i32) !*Tensor {
    const head_dim = x.dims[3]; // [B, n_heads, seq, head_dim]
    const rope_dim = tables.cos.dims[1];
    if (head_dim <= 0 or @mod(head_dim, 2) != 0) return error.BadHeadDim;
    if (rope_dim > head_dim or @mod(rope_dim, 2) != 0) return error.BadRopeDim;
    var params = [_]f32{0} ** types.TENSOR_MAX_OP_PARAMS;
    params[0] = @floatFromInt(head_dim);
    params[1] = @floatFromInt(rope_dim);
    params[2] = 0; // start_pos
    const t = try base.newTensor(allocator, .{
        .name = try std.fmt.allocPrint(allocator, "{s}_rope", .{x.name}),
        .layer_id = layer_id,
        .data_type = x.data_type,
        .dims = x.dims,
        .tensor_type = .TENSOR_TYPE_ACTIVATION,
        .op_type = .OP_TYPE_APPLY_ROPE,
        .op_params = params,
        .src = base.srcs(&.{ x, tables.cos, tables.sin }),
    });

    base.computeStrides(t);

    return t;
}
