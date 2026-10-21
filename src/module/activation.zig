const std = @import("std");

const Tensor = @import("../Tensor.zig");

const gguf = @import("../gguf.zig");
const TensorInfo = gguf.TensorInfo;

const base = @import("base.zig");
const module = @import("module.zig");
const Ctx = module.Ctx;

pub const Silu = struct {
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

    pub fn forward(self: *Self, x: *Tensor) !*Tensor {
        return silu(
            self.allocator,
            x,
            self.name,
            self.layer_id,
        );
    }
};

pub fn silu(
    allocator: std.mem.Allocator,
    input_tensor: *Tensor,
    name: []const u8,
    layer_id: i32,
) !*Tensor {
    return base.elementwise(
        allocator,
        name,
        layer_id,
        .OP_TYPE_SILU,
        input_tensor.dims,
        &.{input_tensor},
    );
}

pub fn sigmoid(
    allocator: std.mem.Allocator,
    input: *Tensor,
    name: []const u8,
    layer_id: i32,
) !*Tensor {
    return base.elementwise(
        allocator,
        name,
        layer_id,
        .OP_TYPE_SIGMOID,
        input.dims,
        &.{input},
    );
}
