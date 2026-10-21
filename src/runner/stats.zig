const std = @import("std");

const batching = @import("batching.zig");
const Sequence = batching.Sequence;
const Entry = batching.Entry;
const BlockManager = @import("block.zig").BlockManager;

const common = @import("../profiler/common.zig");

// queue_time = scheduled_ns   − arrival_ns
// TTFT       = first_token_ns − arrival_ns          (包含排队)
// TPOT       = (finish_ns − first_token_ns) / (completion_tokens − 1)
// E2E        = finish_ns − arrival_ns

pub const EngineStats = struct {
    steps: u64 = 0,
    prefill_steps: u64 = 0, //  含至少一个 prefill 块的 step
    decode_steps: u64 = 0, //   全是 decode 的 step
    sum_seqs: u64 = 0, //       Σ 每步序列数 → 平均 batch
    sum_tokens: u64 = 0, //     Σ 每步 token 数
    sched_ns: u64 = 0, //       schedule() 累计耗时
    exec_ns: u64 = 0, //        execute() 累计耗时
    sum_block_used: u64 = 0, // Σ 每步已用块 → 平均块利用率
    peak_block_used: u64 = 0,

    pub fn onStep(self: *EngineStats, entries: []const Entry, sched_ns: u64, exec_ns: u64, bm: *const BlockManager) void {
        self.steps += 1;
        self.sched_ns += sched_ns;
        self.exec_ns += exec_ns;
        self.sum_seqs += entries.len;

        var has_prefill = false;
        for (entries) |e| {
            self.sum_tokens += e.q_len;
            if (e.q_start < e.seq.num_prompt_tokens) has_prefill = true;
        }
        if (has_prefill) self.prefill_steps += 1 else self.decode_steps += 1;

        const used = bm.num_blocks - bm.freeBlocksNum();
        self.sum_block_used += used;
        self.peak_block_used = @max(self.peak_block_used, used);
    }

    /// seqs: 所有序列(只统计已结束的); duration_ns: 从 Engine 启动到现在
    pub fn print(
        self: *const EngineStats,
        allocator: std.mem.Allocator,
        seqs: []const *Sequence,
        num_blocks: usize,
        num_preemptions: usize,
        duration_ns: u64,
    ) !void {
        const steps_f: f64 = @floatFromInt(@max(self.steps, 1));
        const blocks_f: f64 = @floatFromInt(@max(num_blocks, 1));
        std.debug.print("\n=== 引擎指标 ===\n", .{});
        std.debug.print("steps             : {d} (prefill {d} / decode {d})\n", .{ self.steps, self.prefill_steps, self.decode_steps });
        std.debug.print("平均 batch        : {d:.2} seqs, {d:.1} tokens\n", .{
            @as(f64, @floatFromInt(self.sum_seqs)) / steps_f,
            @as(f64, @floatFromInt(self.sum_tokens)) / steps_f,
        });
        std.debug.print("调度开销          : {d:.2}%\n", .{
            @as(f64, @floatFromInt(self.sched_ns)) / @as(f64, @floatFromInt(@max(self.sched_ns + self.exec_ns, 1))) * 100,
        });
        std.debug.print("块利用率 均值/峰值 : {d:.1}% / {d:.1}%\n", .{
            @as(f64, @floatFromInt(self.sum_block_used)) / steps_f / blocks_f * 100,
            @as(f64, @floatFromInt(self.peak_block_used)) / blocks_f * 100,
        });
        std.debug.print("抢占次数          : {d}\n", .{num_preemptions});

        // 请求级分布
        var ttft: std.ArrayList(u64) = .empty;
        defer ttft.deinit(allocator);
        var queue: std.ArrayList(u64) = .empty;
        defer queue.deinit(allocator);
        var good: usize = 0;
        const slo_ttft: u64 = 500 * std.time.ns_per_ms;
        const slo_tpot: u64 = 50 * std.time.ns_per_ms;

        for (seqs) |s| {
            if (s.finish_ns == 0) continue;
            const t = s.first_token_ns - s.arrival_ns;
            try ttft.append(allocator, t);
            try queue.append(allocator, s.scheduled_ns - s.arrival_ns);
            const n = s.numCompletionTokens();
            const tpot: u64 = if (n > 1) (s.finish_ns - s.first_token_ns) / @as(u64, @intCast(n - 1)) else 0;
            if (t <= slo_ttft and tpot <= slo_tpot) good += 1;
        }
        std.mem.sort(u64, ttft.items, {}, std.sort.asc(u64));
        std.mem.sort(u64, queue.items, {}, std.sort.asc(u64));
        std.debug.print("TTFT P50/P99      : {d:.1} / {d:.1} ms\n", .{
            common.nsToMs(common.percentile(ttft.items, 50)),
            common.nsToMs(common.percentile(ttft.items, 99)),
        });
        std.debug.print("排队 P50/P99      : {d:.1} / {d:.1} ms\n", .{
            common.nsToMs(common.percentile(queue.items, 50)),
            common.nsToMs(common.percentile(queue.items, 99)),
        });
        std.debug.print("Goodput           : {d:.2} req/s ({d}/{d} 满足 SLO)\n", .{
            common.perSec(good, duration_ns), good, ttft.items.len,
        });
    }
};
