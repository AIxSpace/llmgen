const std = @import("std");
const common = @import("common.zig");

const types = @import("../../types.zig");
const Tensor = @import("../../Tensor.zig");

// RMSNorm 计算过程 out = x / sqrt(mean(x^2)+eps) * gain.

// ## 代码做了什么

// 对输入 `x`（形状 `[seq_len, hidden_size]`）**逐行**做三步：

// 1. **求均方根**：`ss = Σ(x[i]²)`，再除以 `hidden_size` 得到均值 `mean(x²)`
// 2. **算缩放系数**：`scale = 1 / sqrt(mean(x²) + eps)`（`eps` 防止除零）
// 3. **归一化 + 增益**：`out[i] = x[i] * scale * gain[i]`（`gain` 是可学习参数，形状 `[hidden]`，对所有行共享）

// ## 实际例子

// 假设 `seq_len=1, hidden_size=4, eps=0.01`：

// ```
// x    = [2.0,  -4.0,  1.0,  3.0]
// gain = [1.0,   1.0,  0.5,  2.0]
// ```

// **第一步：求平方和**

// ```
// ss = 2² + (-4)² + 1² + 3² = 4 + 16 + 1 + 9 = 30
// ```

// **第二步：算 scale**

// ```
// mean(x²) = 30 / 4 = 7.5
// scale    = 1 / sqrt(7.5 + 0.01) = 1 / sqrt(7.51) ≈ 1 / 2.7404 ≈ 0.3649
// ```

// **第三步：逐元素乘 scale 再乘 gain**

// ```
// out[0] = 2.0  * 0.3649 * 1.0 = 0.7298
// out[1] = -4.0 * 0.3649 * 1.0 = -1.4596
// out[2] = 1.0  * 0.3649 * 0.5 = 0.1825
// out[3] = 3.0  * 0.3649 * 2.0 = 2.1894
// ```

// 结果：`out ≈ [0.73, -1.46, 0.18, 2.19]`

// 验证一下：归一化后每行的 RMS（`sqrt(mean(out²))`，gain=1 的列除外）正好是 1，即向量的“能量”被标准化了，方向（比例关系）保持不变——`x` 里最大的 `-4` 归一化后仍是绝对值最大的 `-1.46`。

// ## 几个关键点

// - **为什么叫 RMS**：RMS = Root Mean Square（均方根）。LayerNorm 要减均值再除标准差（两个统计量），RMSNorm 省掉减均值，只除均方根，少了 `mean(x)` 的计算，这正是 LLaMA 等模型用它来提速的原因。
// - **外层循环 `for (0..seq_len)`**：每行（每个 token 的 hidden 向量）独立归一化，行与行之间互不影响，天然可并行。
// - **`eps` 的作用**：如果某行 `x` 全是 0，`mean(x²)=0`，没有 `eps` 会除以 0 得到 inf；加上 `1e-6` 这类小量保证数值稳定。
// - **`gain` 的作用**：纯归一化会把向量强行压到固定“长度”，可能损失表达能力；`gain`（也叫 weight）让网络可以学出每个通道的缩放，这里例子中 `gain[3]=2.0` 就把第 3 维放大了一倍。
// - **`elemF32` 的意义**：输入可能是 F32/BF16/F16 混布的（LLaMA 权重常是 BF16），这个工具函数统一读成 f32 参与计算，而输出 `dest_data` 固定是 f32。
pub const RmsNormImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const x = out_tensor.src[0].?;
        const gain = out_tensor.src[1].?;

        const r = out_tensor.rank();

        const last = common.dimAt(out_tensor, r - 1);

        const rows = out_tensor.numElements() / last;

        const eps = out_tensor.op_params[0];
        const w_offset = out_tensor.op_params[1];

        const round_bf16 = out_tensor.data_type == .GGML_TYPE_BF16; // 新增

        for (0..rows) |s| {
            var ss: f32 = 0; // 平方和
            for (0..last) |hidx| {
                const v = common.elemF32(x, s * last + hidx);
                ss += v * v; // 求平方
            }

            const scale = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(last)) + eps);

            for (0..last) |hidx| {
                // 舍入点同 PyTorch 的 Qwen3RMSNorm((x·rstd).to(input_dtype) · w)和 CUDA kernel:
                // bf16 激活先把 x·scale 舍入成 bf16 再乘权重;f32 激活本来就不丢精度, 不舍入
                var normed = common.elemF32(x, s * last + hidx) * scale;
                if (round_bf16) normed = common.bf16ToF32(common.f32ToBf16(normed));
                const v = normed * (common.elemF32(gain, hidx) + w_offset);
                common.storeF32(out_tensor, s * last + hidx, v);
            }
        }
    }
};
