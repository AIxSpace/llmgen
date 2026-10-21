//! 第 8 章 - sampling -- softmax / argmax / top-p 采样(自包含)
//!
//! 对照 include/utils/tools.hpp 的 ops::Softmax + src/core/executor.cpp 的 sample / sample_argmax / sample_top_p.
//! 这里只放与 logits 存储无关的纯计算:Executor 先把 logits 最后一行(bf16)读成 f32, 再调这些函数.
//! 随机数:C++ 用 thread_local std::mt19937{random_device{}()}(每次运行不同);本 lab 由调用方传入
//! 固定 seed 的 std.Random(见 scheduler.Config.seed), 让 A/B 两版图在 logits 相同时抽到同一 token.

const std = @import("std");

/// 对照 sample_argmax 的比较循环:取最大 logit 的下标(并列取先出现的).
pub fn argmax(logits: []const f32) usize {
    var best: usize = 0;
    var best_val: f32 = logits[0];
    for (logits[1..], 1..) |v, i| {
        if (v > best_val) {
            best_val = v;
            best = i;
        }
    }
    return best;
}

/// 对照 ops::Softmax(span, temperature):减 max → exp((x - max) / T) → 归一化, 并把每个概率钳到 >= 1e-6.
/// 返回的数组归调用方 free.
pub fn softmax(alloc: std.mem.Allocator, vec: []const f32, temperature: f32) ![]f32 {
    const inv_temp = 1.0 / temperature;

    // find max in float domain
    var max_val: f32 = vec[0];
    for (vec[1..]) |v| max_val = @max(max_val, v);

    // exp and sum
    const probs = try alloc.alloc(f32, vec.len);
    var sum: f32 = 0;
    for (vec, 0..) |v, i| {
        probs[i] = @exp((v - max_val) * inv_temp);
        sum += probs[i];
    }

    // normalize, clamp to avoid zero probability
    for (probs) |*p| p.* = @max(p.* / sum, 1e-6);
    return probs;
}

/// 对照 sample() / sample_top_p() 共用的后半段:给定 softmax 概率, top-p ∈ (0, 1) 时按概率降序截断到
/// 累积 >= top_p 的前缀, 在截断集合内按归一化概率抽样(抽不到落回概率最大者);否则直接在全分布上抽样(抽不到返回 0).
pub fn sampleFromProbs(alloc: std.mem.Allocator, probs: []const f32, top_p: f32, rng: std.Random) !usize {
    const vocab_size = probs.len;

    // ========== 2. Top-p 过滤 ==========
    if (top_p > 0.0 and top_p < 1.0) {
        const Pair = struct { p: f32, id: usize };
        const sorted_probs = try alloc.alloc(Pair, vocab_size);
        defer alloc.free(sorted_probs);
        for (sorted_probs, probs, 0..) |*sp, p, i| sp.* = .{ .p = p, .id = i };
        std.sort.pdq(Pair, sorted_probs, {}, struct {
            fn gt(_: void, a: Pair, b: Pair) bool {
                return a.p > b.p;
            }
        }.gt);

        var cumsum: f32 = 0;
        var cutoff: usize = sorted_probs.len;
        for (sorted_probs, 0..) |sp, i| {
            cumsum += sp.p;
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        var truncated_sum: f32 = 0;
        for (sorted_probs[0..cutoff]) |sp| truncated_sum += sp.p;

        const rand_val = rng.float(f32); // 对照 uniform_real_distribution<float>{0, 1}
        var cum_prob: f32 = 0;
        for (sorted_probs[0..cutoff]) |sp| {
            cum_prob += sp.p / truncated_sum;
            if (rand_val <= cum_prob) return sp.id;
        }
        return sorted_probs[0].id;
    }

    // ========== 3. 直接从全分布采样 ==========
    const rand_val = rng.float(f32);
    var cum_prob: f32 = 0;
    for (probs, 0..) |p, i| {
        cum_prob += p;
        if (rand_val <= cum_prob) return i;
    }
    return 0;
}

const testing = std.testing;

test "argmax" {
    try testing.expectEqual(@as(usize, 2), argmax(&[_]f32{ 1, 3, 5, 2 }));
}

test "softmax:归一 + 温度越低越尖 + 钳到 1e-6" {
    const logits = [_]f32{ 1, 2, 3, -100 };
    const p1 = try softmax(testing.allocator, &logits, 1.0);
    defer testing.allocator.free(p1);
    const p2 = try softmax(testing.allocator, &logits, 0.5);
    defer testing.allocator.free(p2);
    try testing.expectApproxEqAbs(@as(f32, 1), p1[0] + p1[1] + p1[2], 1e-5);
    try testing.expect(p2[2] > p1[2]);
    try testing.expectEqual(@as(f32, 1e-6), p1[3]); // exp 下溢后被钳住
}

test "top-p:极小 top_p 只剩概率最大者" {
    const logits = [_]f32{ 1, 9, 2 };
    const probs = try softmax(testing.allocator, &logits, 1.0);
    defer testing.allocator.free(probs);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..20) |_| try testing.expectEqual(@as(usize, 1), try sampleFromProbs(testing.allocator, probs, 0.01, prng.random()));
}

test "固定 seed 可复现" {
    const logits = [_]f32{ 1, 2, 3, 4, 5 };
    const probs = try softmax(testing.allocator, &logits, 0.8);
    defer testing.allocator.free(probs);
    var p1 = std.Random.DefaultPrng.init(7);
    var p2 = std.Random.DefaultPrng.init(7);
    for (0..10) |_| {
        const a = try sampleFromProbs(testing.allocator, probs, 0.9, p1.random());
        const b = try sampleFromProbs(testing.allocator, probs, 0.9, p2.random());
        try testing.expectEqual(a, b);
    }
}
