const std = @import("std");

const types = @import("../types.zig");

const Tensor = @import("../Tensor.zig");

const gguf = @import("../gguf.zig");
const TensorInfo = gguf.TensorInfo;

const trace = @import("trace.zig");
const Trace = trace.Trace;

const base = @import("base.zig");
const module = @import("module.zig");
const Ctx = module.Ctx;

pub const Embedding = struct {
    allocator: std.mem.Allocator,

    layer_id: i32,
    name: []const u8,
    weight_info: *const TensorInfo,

    const Self = @This();

    pub fn init(ctx: Ctx) !Self {
        const spec_entry = ctx.find_spec(ctx.spec_key);

        return .{
            .allocator = ctx.trace.node_allocator,
            .layer_id = ctx.layer_id,
            .name = try ctx.trace.name(spec_entry.tag, ctx.layer_id),
            .weight_info = try ctx.trace.weight(spec_entry.gguf, ctx.layer_id),
        };
    }

    pub fn forward(self: *Self, input_ids: *Tensor) !*Tensor {
        // embedding_lookup 默认 GGML_TYPE_BF16:非量化权重时输出跟随权重 dtype, 量化权重时用 activation_dtype
        return embeddingLookup(
            self.allocator,
            input_ids,
            self.weight_info,
            false,
            self.name,
            self.layer_id,
            .GGML_TYPE_BF16,
        );
    }
};

pub fn embeddingLookup(
    allocator: std.mem.Allocator,
    input_ids: *Tensor,
    weight_info: *const gguf.TensorInfo,
    transpose: bool,
    name: []const u8,
    layer_id: i32,
    activation_dtype: types.GGMLDataType,
) !*Tensor {
    const wt = try base.weightPlaceholder(
        allocator,
        weight_info,
        weight_info.name,
        layer_id,
    );
    const hidden = if (transpose) weight_info.dimensions[0] else weight_info.dimensions[1];
    var params = [_]f32{0} ** types.TENSOR_MAX_OP_PARAMS;
    params[0] = if (transpose) 1 else 0; // transpose

    const t = try base.newTensor(allocator, .{
        .name = name,
        .layer_id = layer_id,
        .data_type = if (weight_info.dtype.isQuantized()) activation_dtype else weight_info.dtype, // ← 输出 dtype 规则,
        .dims = .{ input_ids.dims[0], input_ids.dims[1], hidden, 0 },
        .tensor_type = .TENSOR_TYPE_ACTIVATION,
        .op_type = .OP_TYPE_EMBEDDING,
        .op_params = params,
        .src = base.srcs(&.{ input_ids, wt }),
    });

    base.computeStrides(t);

    return t;
}
