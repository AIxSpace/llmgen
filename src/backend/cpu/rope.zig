//! cpu/rope.zig -- ApplyRopeImpl / RopeCacheImpl<CPU>(由 module-style 改名而来)

const std = @import("std");
const types = @import("../../types.zig");
const Tensor = @import("../../Tensor.zig");
const c = @import("common.zig");

/// NEOX RoPE, token-major:x / out [1, T, n_head, head_dim]。
/// 第 t 个 token 的位置从本步元数据 g_step.pos[t] 读(第 05 步起不再是"起点 + 序号"),
/// 所以同一步里可以混着不同序列、不同位置的 token(第 19 步批处理直接可用)。
/// src[1] = cos / src[2] = sin [max_seq, rope_dim] F32, 每行前后两半存同一个角度, 只读前半。
/// op_params[1] = rope_dim:只旋转前 rope_dim 维, 其余原样复制。
pub const ApplyRopeImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        const cos_data: [*]const f32 = @ptrCast(@alignCast(out_tensor.src[1].?.data.?));
        const sin_data: [*]const f32 = @ptrCast(@alignCast(out_tensor.src[2].?.data.?));
        const pos: [*]const i32 = @ptrCast(@alignCast(c.g_step.pos.?));

        const n_tok: usize = @intCast(c.g_step.n_tok);
        const n_head = c.dimAt(x, 2);
        const head_dim = c.dimAt(x, 3);
        const rope_dim: usize = @intFromFloat(out_tensor.op_params[1]);
        const half = rope_dim / 2;

        for (0..n_tok) |t| {
            const p: usize = @intCast(pos[t]);
            const cos_row = cos_data + p * rope_dim;
            const sin_row = sin_data + p * rope_dim;
            for (0..n_head) |h| {
                const base = (t * n_head + h) * head_dim;
                for (0..half) |i| {
                    const x0 = c.elemF32(x, base + i);
                    const x1 = c.elemF32(x, base + i + half);
                    c.storeF32(out_tensor, base + i, x0 * cos_row[i] - x1 * sin_row[i]);
                    c.storeF32(out_tensor, base + i + half, x1 * cos_row[i] + x0 * sin_row[i]);
                }
                for (rope_dim..head_dim) |i| c.storeF32(out_tensor, base + i, c.elemF32(x, base + i));
            }
        }
    }
};

/// 对照 RopeCacheImpl<Device::CPU>::execute:预计算 [max_seq, head_dim] 的 cos 或 sin 表(输出始终为 F32).
/// op_params[0]=theta, [1]=head_dim(rope_dim), [2]=max_seq;按名字是否含 "_cos" 决定算哪张表.
/// inv_freq / angle 用 f64 算, 前后两半存同一个值(NEOX 配对 i 与 i+half 共用一个角度).
pub const RopeCacheImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const theta: f64 = out_tensor.op_params[0];
        const head_dim: usize = @intFromFloat(out_tensor.op_params[1]);
        const max_seq: usize = @intFromFloat(out_tensor.op_params[2]);
        const is_cos = std.mem.indexOf(u8, out_tensor.name, "_cos") != null;

        const half_dim = head_dim / 2;
        const dst: [*]f32 = @ptrCast(@alignCast(out_tensor.data.?));

        var inv_freqs: [512]f64 = undefined;
        if (half_dim > inv_freqs.len) return error.HeadDimTooLarge;
        for (0..half_dim) |i| {
            const exponent = (2.0 * @as(f64, @floatFromInt(i))) / @as(f64, @floatFromInt(head_dim));
            inv_freqs[i] = 1.0 / std.math.pow(f64, theta, exponent);
        }

        for (0..max_seq) |s| {
            const pos: f64 = @floatFromInt(s);
            const base_idx = s * head_dim; // 输出 [max_seq, head_dim]
            for (0..half_dim) |i| {
                const angle = pos * inv_freqs[i];
                const value: f32 = @floatCast(if (is_cos) @cos(angle) else @sin(angle));
                dst[base_idx + i] = value;
                dst[base_idx + i + half_dim] = value;
            }
        }
    }
};
