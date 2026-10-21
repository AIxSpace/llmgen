const std = @import("std");

const common = @import("common.zig");
const Report = @import("reporter.zig").Report;

const qwen3 = @import("../models/qwen3.zig");
const MemoryManager = @import("../memory/manager.zig").MemoryManager;

// TTFT = t(第一次 sample 返回) − t(generate 开始)
// E2E  = t(generate 返回) − t(generate 开始)

// TPOT 是平均值，ITL 是分布
// ITL：Inter-Token Latency 指相邻两个生成 token 之间的延迟，即模型流式输出过程中“下一个 token 要等多久”。
// TPOT：Time Per Output Token（也常写作 Time Per Token） 指平均每生成一个输出 token 所需的时间，通常可视为 ITL 的平均指标。
// ITL[i] = t(第 i 个 token 产出) − t(第 i−1 个 token 产出)       i ≥ 1
// TPOT   = (t_last − TTFT) / len(ITL)                      = mean(ITL)
//          t_last 是最后一个 token(含 eos)的产出时刻; 不用 E2E, 因为它尾部可能含不产出 token 的工作

// prefill 吞吐 = prompt_tokens / prefill_s
// decode  吞吐 = len(ITL) / (t_last − TTFT)_s = 1 / TPOT
// 总吞吐       = (prompt_tokens + gen_tokens) / E2E_s

// E2E ≈ forward_ns + sample_ns + detok_ns + 杂项
// forward_ns = Σ 算子 ns (OpProfiler) + 框架开销

// 框架开销 = resolveDims + allocateOutput + resetStepActivations
//          + executeView + Timer 本身 + CUDA sync 等待

pub const GenMetrics = struct {
    enabled: bool = false,
    prompt_tokens_len: usize = 0, // prefill tokens num
    gen_tokens_len: usize = 0, // all generate tokens num (not contains eos)

    prefill_ns: u64 = 0, // prefill time
    ttft_ns: u64 = 0, // ttft time
    e2e_ns: u64 = 0, // e2e time

    last_emit_ns: u64 = 0, // 上一个 token 的时间
    itl_ns: std.ArrayList(u64) = .empty, // 每个 token 的间隔延迟

    sample_ns: u64 = 0, // sample time
    detok_ns: u64 = 0, // decode + print time
    forward_ns: u64 = 0, // forward + decode time

    const Self = @This();

    pub fn init() Self {
        return .{
            .enabled = (std.posix.getenv("LLMGEN_METRICS") != null),
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.itl_ns.deinit(allocator);
    }

    pub fn reset(self: *Self) void {
        self.prompt_tokens_len = 0;
        self.gen_tokens_len = 0;

        self.prefill_ns = 0;
        self.ttft_ns = 0;
        self.e2e_ns = 0;

        self.last_emit_ns = 0;
        self.itl_ns.clearRetainingCapacity();

        self.sample_ns = 0;
        self.detok_ns = 0;
        self.forward_ns = 0;
    }

    pub fn onGenToken(self: *Self, allocator: std.mem.Allocator, now_ns: u64) !void {
        if (self.ttft_ns == 0) {
            self.ttft_ns = now_ns; // TTFT 首 token 延迟
        } else {
            try self.itl_ns.append(allocator, now_ns - self.last_emit_ns);
        }
        self.last_emit_ns = now_ns;
    }

    pub fn tpotNs(self: *const Self) u64 {
        const n = self.itl_ns.items.len;
        if (n == 0) {
            return 0;
        }

        return self.decodeNs() / n;
    }

    /// decode 阶段耗时: 首产生 token 到最后一个 token
    pub fn decodeNs(self: *const Self) u64 {
        return self.last_emit_ns - self.ttft_ns;
    }

    pub fn print(self: *Self) void {
        std.debug.print("\n=== 生成指标 ===\n", .{});
        std.debug.print("prompt tokens len : {d}\n", .{self.prompt_tokens_len});
        std.debug.print("gen tokens len    : {d}\n", .{self.gen_tokens_len});
        std.debug.print("prefill           : {d:.2} ms\n", .{common.nsToMs(self.prefill_ns)});
        std.debug.print("TTFT              : {d:.2} ms\n", .{common.nsToMs(self.ttft_ns)});
        std.debug.print("E2E               : {d:.2} ms\n", .{common.nsToMs(self.e2e_ns)});

        const itl = self.itl_ns.items;
        if (itl.len > 0) {
            const k = @max(itl.len / 10, 1);
            std.debug.print("TPOT          : {d:.2} ms\n", .{common.nsToMs(self.tpotNs())});
            std.debug.print("ITL 前10%均值  : {d:.2} ms\n", .{common.nsToMs(common.mean(itl[0..k]))});
            std.debug.print("ITL 后10%均值  : {d:.2} ms\n", .{common.nsToMs(common.mean(itl[itl.len - k ..]))});

            std.mem.sort(u64, itl, {}, std.sort.asc(u64)); // 先排序
            std.debug.print("ITL P50/P90/P99/max : {d:.2} / {d:.2} / {d:.2} / {d:.2} ms\n", .{
                common.nsToMs(common.percentile(itl, 50)),
                common.nsToMs(common.percentile(itl, 90)),
                common.nsToMs(common.percentile(itl, 99)),
                common.nsToMs(itl[itl.len - 1]),
            });
        }

        std.debug.print(
            "prefill 吞吐  : {d:.1} tok/s\n",
            .{common.perSec(self.prompt_tokens_len, self.prefill_ns)},
        );
        if (itl.len > 0) {
            std.debug.print(
                "decode  吞吐  : {d:.1} tok/s\n",
                .{common.perSec(itl.len, self.decodeNs())},
            );
        }
        std.debug.print(
            "总吞吐        : {d:.1} tok/s\n",
            .{common.perSec(self.prompt_tokens_len + self.gen_tokens_len, self.e2e_ns)},
        );

        self.printBreakdown();
    }

    /// 时间账本: E2E = forward + sample + detok + 其它
    fn printBreakdown(self: *const Self) void {
        const accounted = self.forward_ns + self.sample_ns + self.detok_ns;
        const other = self.e2e_ns -| accounted; // 饱和减法, 防止计时误差导致下溢

        std.debug.print("\n=== 时间拆解 ===\n", .{});
        self.printShare("forward", self.forward_ns);
        self.printShare("sample", self.sample_ns);
        self.printShare("detok+print", self.detok_ns);
        self.printShare("其它", other);

        const samples = self.itl_ns.items.len + 1; // sample 调用次数(含 eos)
        std.debug.print("sample 平均       : {d:.3} ms/次\n", .{
            common.nsToMs(self.sample_ns) / @as(f64, @floatFromInt(samples)),
        });
    }

    fn printShare(self: *const Self, name: []const u8, ns: u64) void {
        const percentage: f64 = if (self.e2e_ns == 0) 0 else @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(self.e2e_ns)) * 100;
        std.debug.print("{s:<17} : {d:>10.2} ms ({d:>5.1}%)\n", .{ name, common.nsToMs(ns), percentage });
    }

    /// decode 阶段框架开销, decode_op_ns 来自 OpProfiler.opNs(.decode)
    /// prefill() 只做 setRopeStart + forward, 所以 forward_ns − prefill_ns ≈ decode 阶段的 forward 总耗时
    pub fn printOverhead(self: *const Self, decode_op_ns: u64) void {
        const decode_fwd = self.forward_ns -| self.prefill_ns;
        const overhead = decode_fwd -| decode_op_ns;
        const pct: f64 = if (decode_fwd == 0) 0 else @as(f64, @floatFromInt(overhead)) / @as(f64, @floatFromInt(decode_fwd)) * 100;
        std.debug.print("\n=== decode 框架开销 ===\n", .{});
        std.debug.print("decode forward    : {d:.2} ms\n", .{common.nsToMs(decode_fwd)});
        std.debug.print("  其中算子        : {d:.2} ms\n", .{common.nsToMs(decode_op_ns)});
        std.debug.print("  框架开销        : {d:.2} ms ({d:.1}%)\n", .{ common.nsToMs(overhead), pct });
    }

    /// 必须在 print() 之后调用: print 会把 itl_ns 排序, 分位数依赖有序
    pub fn toReport(self: *const Self, backend: []const u8) Report {
        const itl = self.itl_ns.items;
        return .{
            .backend = backend,
            .prompt_tokens = self.prompt_tokens_len,
            .gen_tokens = self.gen_tokens_len,
            .prefill_ms = common.nsToMs(self.prefill_ns),
            .ttft_ms = common.nsToMs(self.ttft_ns),
            .tpot_ms = common.nsToMs(self.tpotNs()),
            .itl_p50_ms = common.nsToMs(common.percentile(itl, 50)),
            .itl_p99_ms = common.nsToMs(common.percentile(itl, 99)),
            .prefill_tps = common.perSec(self.prompt_tokens_len, self.prefill_ns),
            .decode_tps = common.perSec(itl.len, self.decodeNs()),
            .e2e_ms = common.nsToMs(self.e2e_ns),
        };
    }
};

// 单 token FLOPs(ctx)   = 2·N_mm + 4·L·q_dim·ctx   ≈ 1.19 GFLOP + 229K·ctx FLOP
// prefill(P) 总 FLOPs  = 2·N_mm·P + 4·L·q_dim · P(P+1)/2          (因果 mask)
// decode 第 c 步 FLOPs = 2·N_mm   + 4·L·q_dim · c                 (c = 当前上下文长度)

// KV 每 token 字节     = 2(K,V) · L · kv_dim · kv_elem  = 2 × 28 × 1024 × 2B = 112 KiB (BF16)
// decode 第 c 步字节   ≈ N_mm·w_elem + c·112KiB        ≈ 1.19 GB + c × 0.115 MB

// MFU(prefill) = prefill 总 FLOPs / prefill_s / 峰值 FLOPS
// MBU(decode)  = Σ decode 每步字节 / decode_s / 峰值带宽

/// 由模型结构推导理论计算量和访存量
pub const ModelCost = struct {
    n_layers: f64,
    q_dim: f64,
    kv_dim: f64,
    n_mm: f64, //    参与矩阵乘的参数量
    w_elem: f64, //  权重每元素字节(BF16 = 2)
    kv_elem: f64, // KV 每元素字节(以 page_atten 的 data_type 为准)

    pub fn fromConfig(c: qwen3.Config, w_elem: f64, kv_elem: f64) ModelCost {
        const d: f64 = @floatFromInt(c.embedding_length);
        const q: f64 = @floatFromInt(c.qHeadsDim());
        const kv: f64 = @floatFromInt(c.kvHeadsDim());
        const f: f64 = @floatFromInt(c.feed_forward_length);
        const v: f64 = @floatFromInt(c.vocab_size);
        const l: f64 = @floatFromInt(c.block_count);
        const per_layer = d * q + 2 * d * kv + q * d + 3 * d * f;
        return .{
            .n_layers = l,
            .q_dim = q,
            .kv_dim = kv,
            .n_mm = l * per_layer + d * v,
            .w_elem = w_elem,
            .kv_elem = kv_elem,
        };
    }

    pub fn prefillFlops(self: ModelCost, p: f64) f64 {
        return 2 * self.n_mm * p + 4 * self.n_layers * self.q_dim * p * (p + 1) / 2;
    }

    /// decode 从上下文 c0 开始做 n 次 forward: c 依次为 c0, c0+1, ..., c0+n-1
    pub fn decodeFlops(self: ModelCost, c0: f64, n: f64) f64 {
        const sum_c = n * c0 + n * (n - 1) / 2;
        return 2 * self.n_mm * n + 4 * self.n_layers * self.q_dim * sum_c;
    }

    pub fn decodeBytes(self: ModelCost, c0: f64, n: f64) f64 {
        const sum_c = n * c0 + n * (n - 1) / 2;
        const kv_tok = 2 * self.n_layers * self.kv_dim * self.kv_elem;
        return self.n_mm * self.w_elem * n + kv_tok * sum_c;
    }
};

pub fn printEfficiency(metrics: *const GenMetrics, cost: ModelCost) void {
    const p: f64 = @floatFromInt(metrics.prompt_tokens_len);
    const n: f64 = @floatFromInt(metrics.itl_ns.items.len); // decode forward 次数
    const prefill_s = @as(f64, @floatFromInt(metrics.prefill_ns)) / 1e9;
    const decode_s = @as(f64, @floatFromInt(metrics.decodeNs())) / 1e9;

    const pf = cost.prefillFlops(p);
    std.debug.print("\n=== 硬件效率 ===\n", .{});
    std.debug.print("N_mm              : {d:.1} M 参数\n", .{cost.n_mm / 1e6});
    std.debug.print("prefill           : {d:.1} GFLOP, {d:.3} TFLOPS\n", .{ pf / 1e9, pf / prefill_s / 1e12 });
    if (envF64("LLMGEN_PEAK_TFLOPS")) |peak| {
        std.debug.print("MFU(prefill)      : {d:.1}%\n", .{pf / prefill_s / (peak * 1e12) * 100});
    }

    if (n == 0 or decode_s == 0) return;
    const df = cost.decodeFlops(p + 1, n);
    const db = cost.decodeBytes(p + 1, n);
    std.debug.print("decode            : {d:.3} TFLOPS, {d:.1} GB/s\n", .{ df / decode_s / 1e12, db / decode_s / 1e9 });
    if (envF64("LLMGEN_PEAK_GBPS")) |peak| {
        std.debug.print("MBU(decode)       : {d:.1}%\n", .{db / decode_s / (peak * 1e9) * 100});
    }
}

fn envF64(name: []const u8) ?f64 {
    const s = std.posix.getenv(name) orelse return null;
    return std.fmt.parseFloat(f64, s) catch null;
}

// 池利用率     = peak_ / capacity            (看峰值; 激活池每层都会回卷, 当前值没意义)
// KV 块利用率   = Σ(num_blocks − freeNum) / Σ num_blocks          (按层求和)
// 块内碎片率   = 1 − Σ num_cached / (Σ 已用块数 × block_capacity)
// 单层最大 token ≈ num_blocks × block_capacity

/// 打印各设备内存池和分页 KV 的使用情况, 要在 resetKVCache 之前调用
pub fn printMemory(mm: *MemoryManager) void {
    std.debug.print("\n=== 内存 ===\n", .{});
    var buf: [256]u8 = undefined;

    var it = mm.devices_map.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        const pools = &e.value_ptr.*.pools;
        std.debug.print("[{s}#{d}]\n", .{ @tagName(key.device_type), key.device_id });
        if (pools.weight) |*p| std.debug.print("  {s}\n", .{p.formatUsage(&buf)});
        if (pools.activation) |*p| std.debug.print("  {s}\n", .{p.formatUsage(&buf)});
        if (pools.kv_cache) |*p| std.debug.print("  {s}\n", .{p.formatUsage(&buf)});
    }

    // KV 块的占用现在看 Executor 的 block_manager:已用 = num_blocks - freeBlocksNum()
}

pub const Stats = struct {
    mean: f64 = 0,
    p50: f64 = 0,
    p90: f64 = 0,
    p99: f64 = 0,
    max: f64 = 0,
    n: usize = 0,
};

/// 分位数: 线性插值(同 numpy.percentile 默认).
fn percentile(sorted: []const f64, q: f64) f64 {
    if (sorted.len == 0) return 0;
    if (sorted.len == 1) return sorted[0];
    const pos = q * @as(f64, @floatFromInt(sorted.len - 1));
    const lo: usize = @intFromFloat(@floor(pos));
    const hi = @min(lo + 1, sorted.len - 1);
    const frac = pos - @as(f64, @floatFromInt(lo));
    return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

/// 会就地排序 xs.
pub fn stats(xs: []f64) Stats {
    if (xs.len == 0) return .{};
    std.sort.pdq(f64, xs, {}, std.sort.asc(f64));
    var sum: f64 = 0;
    for (xs) |x| sum += x;
    return .{
        .mean = sum / @as(f64, @floatFromInt(xs.len)),
        .p50 = percentile(xs, 0.5),
        .p90 = percentile(xs, 0.9),
        .p99 = percentile(xs, 0.99),
        .max = xs[xs.len - 1],
        .n = xs.len,
    };
}

/// 一次压测运行的逐请求 token 时间戳收集器.
pub const Recorder = struct {
    alloc: std.mem.Allocator,
    t0_ns: i128 = 0,
    first: []i128, // 每个请求首 token 时刻(0 = 尚未产出)
    prev: []i128, // 上一个 token 时刻
    count: []usize,
    itl_ms: std.ArrayListUnmanaged(f64) = .empty,

    pub fn init(allocator: std.mem.Allocator, n: usize) !Recorder {
        const r = Recorder{
            .alloc = allocator,
            .first = try allocator.alloc(i128, n),
            .prev = try allocator.alloc(i128, n),
            .count = try allocator.alloc(usize, n),
        };
        @memset(r.first, 0);
        @memset(r.prev, 0);
        @memset(r.count, 0);
        return r;
    }

    pub fn deinit(self: *Recorder) void {
        self.alloc.free(self.first);
        self.alloc.free(self.prev);
        self.alloc.free(self.count);
        self.itl_ms.deinit(self.alloc);
    }

    pub fn start(self: *Recorder, now: i128) void {
        self.t0_ns = now;
    }

    pub fn token(self: *Recorder, req: usize, now: i128) !void {
        if (self.count[req] == 0) {
            self.first[req] = now;
        } else {
            try self.itl_ms.append(self.alloc, @as(f64, @floatFromInt(now - self.prev[req])) / 1e6);
        }
        self.prev[req] = now;
        self.count[req] += 1;
    }

    /// TTFT 样本(ms), 调用方负责 free.
    pub fn ttftMs(self: *const Recorder, alloc: std.mem.Allocator) ![]f64 {
        const out = try alloc.alloc(f64, self.first.len);
        for (self.first, 0..) |f, i| out[i] = @as(f64, @floatFromInt(f - self.t0_ns)) / 1e6;
        return out;
    }
};
