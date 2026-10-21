///! reshape 是视图(Executor 走 executeView, 正常不会派发到这里);permute 真正搬数据.
const std = @import("std");

const common = @import("common.zig");

const types = @import("../../types.zig");
const Tensor = @import("../../Tensor.zig");

/// 对照 ReshapeImpl<Device::CPU>::execute:out 与 src 共享数据, 按 out.dims 重算行主序 strides.
pub const ReshapeImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        out_tensor.data = x.data;
        out_tensor.offset = x.offset;
        var stride: u64 = 1;
        var i: usize = types.TENSOR_MAX_DIMS;

        while (i > 0) {
            i -= 1;
            if (out_tensor.dims[i] == 0) {
                out_tensor.strides[i] = 0;
            } else {
                out_tensor.strides[i] = stride;
                stride *= @intCast(out_tensor.dims[i]);
            }
        }
    }
};

/// 对照 PermuteImpl<Device::CPU>::execute:perm 取自 op_params(建图时存入), 源偏移按 x.strides 算, 按 dtype 字节拷贝.
///   最内维未被置换:源和输出都是行内连续, 逐行 memcpy(快速路径);
///   最内维被置换:退化为逐元素拷贝(慢速路径).
pub const PermuteImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        // 有效维度数
        var ndim: usize = 0;
        while (ndim < types.TENSOR_MAX_DIMS and x.dims[ndim] != 0) ndim += 1;
        // Read perm from op_params (stored during graph build)
        var perm: [types.TENSOR_MAX_DIMS]usize = undefined;
        for (0..ndim) |i| perm[i] = @intFromFloat(out_tensor.op_params[i]);

        const elem_sz = out_tensor.data_type.size();
        const total = out_tensor.numElements();
        const dst = out_tensor.data.?;
        const src = x.data.?;
        const inner_dim: usize = @intCast(out_tensor.dims[ndim - 1]);
        const inner_bytes = inner_dim * elem_sz;
        if (perm[ndim - 1] == ndim - 1) {
            // ── 快速路径: 最内维未被置换，源和输出都是行内连续 ──
            const outer_count = total / inner_dim;
            for (0..outer_count) |r| {
                var remaining = r;
                var src_offset: usize = 0;
                var d: usize = ndim - 1;
                while (d > 0) {
                    d -= 1;
                    const dim_sz: usize = @intCast(out_tensor.dims[d]);
                    const coord = remaining % dim_sz;
                    remaining /= dim_sz;
                    src_offset += coord * x.strides[perm[d]];
                }
                @memcpy(dst[r * inner_bytes ..][0..inner_bytes], src[src_offset * elem_sz ..][0..inner_bytes]);
            }
        } else {
            // ── 慢速路径: 最内维被置换，退化为逐元素拷贝 ──
            for (0..total) |idx| {
                var remaining = idx;
                var src_offset: usize = 0;
                var d: usize = ndim;
                while (d > 0) {
                    d -= 1;
                    const dim_sz: usize = @intCast(out_tensor.dims[d]);
                    const coord = remaining % dim_sz;
                    remaining /= dim_sz;
                    src_offset += coord * x.strides[perm[d]];
                }
                @memcpy(dst[idx * elem_sz ..][0..elem_sz], src[src_offset * elem_sz ..][0..elem_sz]);
            }
        }
    }
};
