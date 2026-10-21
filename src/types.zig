const std = @import("std");

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" 小端

pub const TENSOR_MAX_SRC: usize = 8;
pub const TENSOR_MAX_DIMS: usize = 4; // tensor max dim
pub const TENSOR_MAX_OP_PARAMS: usize = 4;

pub const DeviceType = enum(u8) {
    CPU = 0,
    CUDA = 1,
    SYCL = 3,
    VULKAN = 4,
    UNKNOWN = 255,
};

pub const GGMLDataType = enum(u8) {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27,
    GGML_TYPE_F64 = 28,
    GGML_TYPE_BF16 = 30,
    GGML_TYPE_COUNT = 40,
    _,

    pub fn size(self: GGMLDataType) usize {
        return switch (self) {
            .GGML_TYPE_F64, .GGML_TYPE_I64 => 8,
            .GGML_TYPE_F32, .GGML_TYPE_I32 => 4,
            .GGML_TYPE_F16, .GGML_TYPE_BF16, .GGML_TYPE_I16 => 2,
            .GGML_TYPE_I8 => 1,
            else => 0, // 量化类型走 MemoryBlock
        };
    }

    // 是否量化
    pub fn isQuantized(self: GGMLDataType) bool {
        return switch (self) {
            .GGML_TYPE_F32,
            .GGML_TYPE_F16,
            .GGML_TYPE_BF16,
            .GGML_TYPE_F64,
            .GGML_TYPE_I8,
            .GGML_TYPE_I16,
            .GGML_TYPE_I32,
            .GGML_TYPE_I64,
            => false,
            else => true,
        };
    }

    pub fn name(self: GGMLDataType) []const u8 {
        return @tagName(self);
    }
};

pub const GGUFType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

/// 张量生命周期分类(对照 utils.hpp: enum class TensorType).
pub const TensorType = enum(u8) {
    TENSOR_TYPE_UNKNOWN = 0,
    TENSOR_TYPE_WEIGHT,
    TENSOR_TYPE_CACHE,
    TENSOR_TYPE_KV_CACHE,
    TENSOR_TYPE_INPUT,
    TENSOR_TYPE_ACTIVATION,
    TENSOR_TYPE_OUTPUT,
    TENSOR_TYPE_VIEW,
};

pub const OperationType = enum(u8) {
    OP_TYPE_NONE = 0,
    OP_TYPE_MEMCPY,
    OP_TYPE_ADD,
    OP_TYPE_SUB,
    OP_TYPE_MUL,
    OP_TYPE_DIV,
    OP_TYPE_MAT_MUL,
    OP_TYPE_TRANSPOSE,
    OP_TYPE_RESHAPE,
    OP_TYPE_PERMUTE,
    OP_TYPE_VIEW,
    OP_TYPE_CONCAT,
    OP_TYPE_REPEAT,
    OP_TYPE_SOFTMAX,
    OP_TYPE_RMS_NORM,
    OP_TYPE_LAYER_NORM,
    OP_TYPE_GELU,
    OP_TYPE_SILU,
    OP_TYPE_RELU,
    OP_TYPE_EMBEDDING,
    OP_TYPE_LINEAR,
    OP_TYPE_APPLY_ROPE,
    OP_TYPE_ROPE_CACHE,
    OP_TYPE_FLASH_ATTN,
    OP_TYPE_PAGED_ATTN,
    OP_TYPE_CAUSAL_CONV1D,
    OP_TYPE_SSM_SCAN,
    OP_TYPE_SPLIT,
    OP_TYPE_SIGMOID,
    OP_TYPE_GATHER_LAST, // 每条序列取最后一个 token 的行
    OP_TYPE_SAMPLE, // GPU 采样
};

test "datatype-bytes-quantized" {
    try std.testing.expectEqual(4, GGMLDataType.GGML_TYPE_F32.size());
    try std.testing.expectEqual(2, GGMLDataType.GGML_TYPE_F16.size());
    try std.testing.expectEqual(2, GGMLDataType.GGML_TYPE_BF16.size());
    try std.testing.expectEqual(1, GGMLDataType.GGML_TYPE_I8.size());
    try std.testing.expect(!GGMLDataType.GGML_TYPE_F32.isQuantized());
    try std.testing.expect(GGMLDataType.GGML_TYPE_Q4_K.isQuantized());
    try std.testing.expectEqual(30, @intFromEnum(GGMLDataType.GGML_TYPE_BF16));
}
