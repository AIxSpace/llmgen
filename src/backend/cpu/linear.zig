const std = @import("std");
const common = @import("common.zig");

const types = @import("../../types.zig");
const Tensor = @import("../../Tensor.zig");

pub const LinearImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        const weight = out_tensor.src[1].?;

        const in_dim = common.dimAt(weight, 1);
        const out_dim = common.dimAt(weight, 0);

        const seq_len = common.dimAt(x, 1); // [1, S, H]

        for (0..seq_len) |seq_idx| {
            for (0..out_dim) |out_idx| {
                var acc: f32 = 0;
                const wrow = out_idx * in_dim;
                for (0..in_dim) |in_idx| {
                    acc += common.elemF32(x, seq_idx * in_dim + in_idx) * common.elemF32(weight, wrow + in_idx);
                }
                common.storeF32(out_tensor, seq_idx * out_dim + out_idx, acc);
            }
        }
    }
};

/// 对照 MatmulImpl<CPU>.两个 f32 激活相乘 A[m,k]@B[k,n](B 行主序 [k,n]).
pub const MatmulImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const a = out_tensor.src[0].?;
        const b = out_tensor.src[1].?;

        const m = common.dimAt(a, 0); // A 的行
        const k = common.dimAt(a, 1); // A 的列 = B 的行（隐式约束）
        const n = common.dimAt(b, 1); // B 的列

        for (0..m) |m_idx| {
            for (0..n) |n_idx| {
                var acc: f32 = 0;
                for (0..k) |k_idx| {
                    acc += common.elemF32(a, m_idx * k + k_idx) * common.elemF32(b, k_idx * n + n_idx);
                }
                common.storeF32(out_tensor, m_idx * n + n_idx, acc);
            }
        }
    }
};
