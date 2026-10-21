//! RmsNormImpl<CUDA>:委托 normalization.cu 的 cuda_rmsnorm。
//! 行宽 = 权重长度:attn_norm / ffn_norm / final_norm 按 hidden, q_norm / k_norm 按 head_dim(每头一行)。
const Tensor = @import("../../../Tensor.zig");

pub extern fn cuda_rmsnorm(out: ?*anyopaque, x: ?*const anyopaque, w: ?*const anyopaque, rows: c_long, D: c_int, eps: f32) void;

pub const RmsNormImpl = struct {
    /// src[0] = x(BF16), src[1] = norm 权重(F32);op_params[0] = eps。op_params[1](w_offset)Qwen3 恒为 0, CUDA 不支持
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        const w = out_tensor.src[1].?;

        // kernel 把权重当 float* 读、x 按 bf16x8 读。llama.cpp 转出来的 GGUF 里 1 维张量(norm 权重)本来就是 F32
        if (w.data_type != .GGML_TYPE_F32 or x.data_type != .GGML_TYPE_BF16) {
            return error.RmsNormNeedsBf16AndF32Gain;
        }

        if (out_tensor.op_params[1] != 0) {
            return error.RmsNormWeightOffsetUnsupported;
        }

        const d = w.numElements();

        if (d != 128 and d % 8 != 0) {
            return error.RmsNormBadWidth; // hidden 行按 bf16x8 读
        }

        cuda_rmsnorm(out_tensor.data, x.data, w.data, @intCast(x.numElements() / d), @intCast(d), out_tensor.op_params[0]);
    }
};
