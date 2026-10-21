const std = @import("std");

const types = @import("../types.zig");
const TensorType = types.TensorType;
const GGMLDataType = types.GGMLDataType;

const gguf = @import("../gguf.zig");
const Tensor = @import("../Tensor.zig");
const GGUFInfo = gguf.GGUFInfo;

pub const Trace = @import("trace.zig").Trace;

pub const Embedding = @import("embedding.zig").Embedding;
pub const Linear = @import("linear.zig").Linear;
pub const RmsNorm = @import("norm.zig").RmsNorm;
pub const Attention = @import("attention.zig").Attention;
pub const ResidualAdd = @import("arithmetic.zig").ResidualAdd;
pub const Mul = @import("arithmetic.zig").Mul;
pub const Silu = @import("activation.zig").Silu;
pub const RopeCache = @import("rope.zig").RopeCache; // ← 替换原来的 RoPE
pub const Rope = @import("rope.zig").Rope;
pub const RopeTables = @import("rope.zig").RopeTables;
pub const HeadSplit = @import("shape.zig").HeadSplit;

pub const SpecEntry = struct {
    gguf: []const u8,
    tag: []const u8,
};

pub const SpecFn = *const fn (name: []const u8) SpecEntry;

pub const ConfigParams = struct {
    n_head: i64,
    head_count_kv: i64,
    head_dim: i64,

    rms_eps: f32,

    rope_freq_base: f32,
    context_length: i64, // rope 表的行数
};

pub const Ctx = struct {
    layer_id: i32,
    find_spec: SpecFn,
    spec_key: []const u8 = "",
    gguf_info: *const GGUFInfo, // todo: - this should not owned
    trace: *Trace,
    config_params: ConfigParams,
    rope: ?RopeTables = null,

    attn_op: types.OperationType = .OP_TYPE_PAGED_ATTN,

    const Self = @This();

    pub fn useSpec(self: Self, key: []const u8) Self {
        var c = self;
        c.spec_key = key;
        return c;
    }

    pub fn atLayer(self: Self, layer_id: i32) Self {
        var c = self;
        c.layer_id = layer_id;
        c.spec_key = "";
        return c;
    }
};

pub fn Module(comptime Self: type) type {
    return struct {
        pub fn init(ctx: Ctx) !Self {
            var self: Self = undefined;
            inline for (@typeInfo(Self).@"struct".fields) |f| {
                if (comptime !@hasDecl(f.type, "init")) continue; // 如果没有 init 直接跳过
                var child_ctx = ctx;
                child_ctx.spec_key = f.name;
                @field(self, f.name) = try f.type.init(child_ctx);
            }
            return self;
        }
    };
}
