const std = @import("std");

const types = @import("../types.zig");

const Tensor = @import("../Tensor.zig");

const gguf = @import("../gguf.zig");
const TensorInfo = gguf.TensorInfo;

const base = @import("base.zig");
const module = @import("module.zig");
const Ctx = module.Ctx;

pub const ResidualAdd = struct {
    allocator: std.mem.Allocator,

    layer_id: i32,
    name: []const u8,

    const Self = @This();

    pub fn init(ctx: Ctx) !Self {
        const spec_entry = ctx.find_spec(ctx.spec_key);

        return .{
            .allocator = ctx.trace.node_allocator,
            .layer_id = ctx.layer_id,
            .name = try ctx.trace.name(spec_entry.tag, ctx.layer_id),
        };
    }

    pub fn forward(self: *Self, a: *Tensor, b: *Tensor) !*Tensor {
        return add(
            self.allocator,
            a,
            b,
            self.name,
            self.layer_id,
        );
    }
};

pub fn add(allocator: std.mem.Allocator, a: *Tensor, b: *Tensor, name: []const u8, layer_id: i32) !*Tensor {
    const t = try base.newTensor(allocator, .{
        .name = name,
        .data_type = a.data_type,
        .layer_id = layer_id,
        .tensor_type = .TENSOR_TYPE_ACTIVATION,
        .src = base.srcs(&.{ a, b }),
        .op_type = .OP_TYPE_ADD,
        .dims = a.dims,
    });

    base.computeStrides(t);

    return t;
}

pub const Mul = struct {
    allocator: std.mem.Allocator,

    layer_id: i32,
    name: []const u8,

    const Self = @This();

    pub fn init(ctx: Ctx) !Self {
        const spec_entry = ctx.find_spec(ctx.spec_key);

        return .{
            .allocator = ctx.trace.node_allocator,
            .layer_id = ctx.layer_id,
            .name = try ctx.trace.name(spec_entry.tag, ctx.layer_id),
        };
    }

    pub fn forward(self: *Self, a: *Tensor, b: *Tensor) !*Tensor {
        return mul(
            self.allocator,
            a,
            b,
            self.name,
            self.layer_id,
        );
    }
};

pub fn mul(
    allocator: std.mem.Allocator,
    a_tensor: *Tensor,
    b_tensor: *Tensor,
    name: []const u8,
    layer_id: i32,
) !*Tensor {
    for (a_tensor.dims, b_tensor.dims) |da, db| {
        if (da != db and da > 0 and db > 0) {
            return error.MulShapeMismatch;
        }
    }
    return base.elementwise(
        allocator,
        name,
        layer_id,
        .OP_TYPE_MUL,
        a_tensor.dims,
        &.{ a_tensor, b_tensor },
    );
}
