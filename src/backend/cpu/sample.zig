//! cpu/sample.zig -- SampleImpl<CPU>: 和 cuda/sample/sample.cu 同一个算法
//! temperature > 0 用 Gumbel-max: argmax(logit / T − log(−log U)), U 由无状态哈希 splitmix64(seed, 行, 词 id) 给出;
//! temperature ≤ 0 退化为 greedy argmax。并列取小下标(与 torch.argmax 一致)。
const std = @import("std");
const Tensor = @import("../../Tensor.zig");
const c = @import("common.zig");

pub fn splitmix64(x0: u64) u64 {
    var x = x0 +% 0x9E3779B97F4A7C15;
    x = (x ^ (x >> 30)) *% 0xBF58476D1CE4E5B9;
    x = (x ^ (x >> 27)) *% 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// src[0] = logits [1, S, V] bf16 → out [1, S] i32
pub const SampleImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const s = &c.g_step;
        const logits = out_tensor.src[0].?;
        const vocab = c.dimAt(logits, 2);
        const out: [*]i32 = @ptrCast(@alignCast(out_tensor.data.?));
        const temps: [*]const f32 = @ptrCast(@alignCast(s.temps.?));
        const step_seed = @as(*const u64, @ptrCast(@alignCast(s.seed.?))).*;

        for (0..@intCast(s.n_seq)) |row| {
            const temp = temps[row];
            const seed = splitmix64(step_seed ^ (@as(u64, row) << 40));
            var best = -std.math.inf(f32);
            var best_i: usize = 0;
            for (0..vocab) |i| {
                var val = c.elemF32(logits, row * vocab + i);
                if (temp > 0) {
                    const r = splitmix64(seed +% i);
                    const u = (@as(f32, @floatFromInt(r >> 40)) + 0.5) * (1.0 / 16777216.0); // (0, 1)
                    val = val / temp - @log(-@log(u));
                }
                if (val > best) { // 严格大于:并列保留小下标
                    best = val;
                    best_i = i;
                }
            }
            out[row] = @intCast(best_i);
        }
    }
};
