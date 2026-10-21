const std = @import("std");
const types = @import("../../types.zig");

const Tensor = @import("../../Tensor.zig");

const paged = @import("../../memory/page_atten.zig");

pub var global_allocator: ?std.mem.Allocator = null; // 只剩 NaiveAttentionImpl 在用

pub const StepState = struct {
    n_tok: c_int = 0, // 本趟 token 数 T(CUDA Graph 桶里 = S)
    n_seq: c_int = 0, // 本趟序列数 S(含 CUDA Graph 桶的 padding 行)
    n_tiles: c_int = 0, // prefill 注意力的 query tile 数
    decode: bool = false,
    // 模型超参
    hidden: c_int = 0,
    n_head: c_int = 0,
    n_kv_head: c_int = 0,
    head_dim: c_int = 0,
    attn_scale: f32 = 0,
    // 分页 KV
    block_size: c_int = 0,
    max_blocks: c_int = 0, // 块表列数
    max_model_len: usize = 0,
    // 本步元数据(设备指针; CPU / Metal 下是 host 指针. 布局见 executor.zig 的 Layout)
    seed: ?*anyopaque = null,
    pos: ?*anyopaque = null,
    slots: ?*anyopaque = null,
    cu_q: ?*anyopaque = null,
    ctx: ?*anyopaque = null,
    last: ?*anyopaque = null,
    temps: ?*anyopaque = null,
    tile_seq: ?*anyopaque = null,
    tile_q0: ?*anyopaque = null,
    block_tables: ?*anyopaque = null,
    // 常驻 workspace
    cos_sin: ?*anyopaque = null,
    part_o: ?*anyopaque = null,
    part_ml: ?*anyopaque = null,
    sample_scratch: ?*anyopaque = null,
};

pub var g_step: StepState = .{};

/// 本设备的分页 KV 管理器。单设备, Executor.init 里设一次。
pub var g_attn_mgr: ?*paged.PagedAttentionManager = null;

/// 本层的 K / V 显存地址
pub fn kvOf(layer_id: i32) !*paged.LayerState {
    const mgr = g_attn_mgr orelse return error.NoAttentionManager;
    return mgr.getLayer(layer_id) orelse return error.LayerNotActive;
}

// ---- 16 位浮点 <-> f32 ----
pub fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16); // bf16 就是 f32 的高 16 位
}

/// f32 → bf16, round-to-nearest-even(对照 C++ float_to_bf16_bits)。
pub fn f32ToBf16(f: f32) u16 {
    const u: u32 = @bitCast(f);
    if ((u & 0x7FFF_FFFF) > 0x7F80_0000) return 0x7FC0; // NaN 保持 NaN(不让舍入把尾数进位成 inf)
    const rounding_bias: u32 = 0x7FFF + ((u >> 16) & 1); // 0x7FFF + 保留位最低位 = 平局时向偶数舍入
    return @intCast((u +% rounding_bias) >> 16);
}

pub fn f16ToF32(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

pub fn f32ToF16(f: f32) u16 {
    const h: f16 = @floatCast(f);
    return @bitCast(h);
}

/// 按 t.data_type 读第 i 个元素为 f32。
pub fn elemF32(t: *const Tensor, i: usize) f32 {
    return loadF32(t.data.?, t.data_type, i);
}

/// 按 dtype 从原始指针读第 i 个元素(给分页 KV 块这类不是 Tensor 的缓冲用)。
pub fn loadF32(base: [*]const u8, data_type: types.GGMLDataType, i: usize) f32 {
    return switch (data_type) {
        .GGML_TYPE_F32 => @as([*]const f32, @ptrCast(@alignCast(base)))[i],
        .GGML_TYPE_BF16 => bf16ToF32(@as([*]const u16, @ptrCast(@alignCast(base)))[i]),
        .GGML_TYPE_F16 => f16ToF32(@as([*]const u16, @ptrCast(@alignCast(base)))[i]),
        else => unsupportedDtype("loadF32", data_type),
    };
}

/// 按 t.data_type 把 f32 写回第 i 个元素
pub fn storeF32(t: *Tensor, i: usize, v: f32) void {
    const tensor_data = t.data.?;
    switch (t.data_type) {
        .GGML_TYPE_F32 => @as([*]f32, @ptrCast(@alignCast(tensor_data)))[i] = v,
        .GGML_TYPE_BF16 => @as([*]u16, @ptrCast(@alignCast(tensor_data)))[i] = f32ToBf16(v),
        .GGML_TYPE_F16 => @as([*]u16, @ptrCast(@alignCast(tensor_data)))[i] = f32ToF16(v),
        else => unsupportedDtype("storeF32", t.data_type),
    }
}

/// 不支持的 dtype(量化等)大声失败, 不静默按 0 算错(原来的 elemF32 是 else => 0)。
pub fn unsupportedDtype(comptime where: []const u8, data_type: types.GGMLDataType) noreturn {
    std.debug.panic(where ++ ": unsupported dtype {s}", .{@tagName(data_type)});
}

pub fn dimAt(t: *const Tensor, i: usize) usize {
    return @intCast(t.dims[i]);
}

pub fn outTensorDataPtr(t: *Tensor) [*]f32 {
    return @ptrCast(@alignCast(t.data.?));
}

pub fn i32DataPtr(t: *const Tensor) [*]const i32 {
    return @ptrCast(@alignCast(t.data.?));
}
