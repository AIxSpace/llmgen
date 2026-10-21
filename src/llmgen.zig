const std = @import("std");
const builtin = @import("builtin");
const opts = @import("build_options");

const backend = @import("backend/backend.zig");
const DeviceManager = backend.DeviceManager;

const gguf = @import("gguf.zig");
const GGUFParser = gguf.GGUFParser;

const ModelFactory = @import("models/factory.zig").ModelFactory;
const qwen3 = @import("models/qwen3.zig");

const scheduler = @import("graph/scheduler.zig");
const GraphScheduler = scheduler.GraphScheduler;

const executor = @import("executor.zig");
const Executor = executor.Executor;

const tkz = @import("tokenizer.zig");

const Engine = @import("runner/engine.zig").Engine;

const Sequence = @import("runner/batching.zig").Sequence;

const workload = @import("bench/workload.zig");

const metrics = @import("profiler/metrics.zig");

const cli = @import("cli.zig");
const recorder = @import("bench/recorder.zig");

const cuda = if (opts.have_cuda) @import("backend/cuda/cuda.zig") else struct {};

pub fn run() !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer {
        if (builtin.mode == .Debug) {
            std.debug.assert(debug_allocator.deinit() == .ok);
        }
    }

    const allocator = if (builtin.mode == .Debug)
        debug_allocator.allocator() // 用来抓泄露，Release 模式换掉，避免启动性能问题
    else
        std.heap.smp_allocator;

    const cli_opts = try cli.parseArgs(allocator);
    defer cli.freeOptions(allocator, cli_opts);

    var load_timer = try std.time.Timer.start(); // 加载耗时:从这里到 Engine 建好

    // 1. gguf:只解析头部, 权重留在磁盘上
    const model_file_path = cli_opts.model orelse std.posix.getenv("LLMGEN_MODEL") orelse return error.NoModelPath;
    var gguf_parser = try GGUFParser.init(allocator, model_file_path);
    defer gguf_parser.deinit(); // 头部缓冲区要活到 tokenizer 析构之后(词表字符串指向它), 所以放在最前面 defer
    const gguf_info = gguf_parser.getInfo();

    // 2. model -> graph
    try ModelFactory.init(allocator);
    defer ModelFactory.deinit(allocator);
    const qwen3_model = try ModelFactory.createFromGGUF(allocator, gguf_info);
    defer qwen3_model.deinit();

    const compute_graph = try qwen3_model.buildGraph(gguf_info);
    if (builtin.mode == .Debug) try compute_graph.exportDot("docs/qwen3-graph.dot"); // 调试用, Release 不写文件

    const model_cfg = try qwen3.Config.fromGGUF(gguf_info);

    // 3. scheduler:原来一个 max_seq_len 管四件事, 现在由命令行分别给(KV 容量由调度器按显存预算算)
    var graph_scheduler = GraphScheduler.init(allocator, compute_graph, .{
        .max_model_len = cli_opts.max_model_len,
        .max_num_seqs = cli_opts.max_num_seqs,
        .max_tokens = cli_opts.maxTokens(),
        .block_size = cli_opts.block_size,
        .gpu_memory_utilization = cli_opts.gpu_util,
        .num_kv_blocks = cli_opts.kv_blocks,
        .vocab_size = qwen3_model.vocabSize(),
        .memory_headroom = 0.05,
        .temperature = cli_opts.temp, // 只有 --single 的 generate 路径读它;Engine 路径由每条请求自己带
        .model_cfg = model_cfg,
    });
    defer graph_scheduler.deinit();

    try DeviceManager.instance().printDevices();
    const devices = DeviceManager.instance().getDevices();
    try graph_scheduler.schedule(devices);

    for (graph_scheduler.assignments.items) |a| {
        std.debug.print("layers {d}..{d} -> {s}#{d}\n", .{ a.start_layer, a.end_layer, @tagName(a.device_type), a.device_id });
    }

    // 4. load weights:磁盘 → pinned 暂存区 → 显存
    const memory_manager = graph_scheduler.getMemoryManager();
    try memory_manager.loadWeights(&gguf_parser, compute_graph);

    // 5. executor(含录 CUDA Graph)
    var runtime_executor = try Executor.init(allocator, compute_graph, &graph_scheduler, memory_manager, model_cfg, .{
        .eager = cli_opts.eager,
        .sync_each = cli_opts.sync_each,
    });
    defer runtime_executor.deinit();

    // 6. tokenizer
    var tokenizer = try tkz.Tokenizer.init(allocator, gguf_info);
    defer tokenizer.deinit();

    const eos: i32 = @intCast(gguf_info.metadata.getInt("tokenizer.ggml.eos_token_id", -1));

    // 7. 服务引擎:排队 → schedule → execute → postprocess。KV 块借用 executor.block_manager
    const engine = try Engine.init(allocator, &runtime_executor, .{
        .eos = eos,
        .policy = cli_opts.policy,
        .chunk_tokens = cli_opts.chunk,
        .max_num_batched_tokens = cli_opts.max_batched,
    });
    defer engine.deinit();

    const load_s = @as(f64, @floatFromInt(load_timer.read())) / 1e9;
    std.debug.print("[engine] 加载 {d:.2} s | 调度策略 {s}", .{ load_s, @tagName(cli_opts.policy) });
    if (cli_opts.policy == .chunked) std.debug.print(", chunk {d} token", .{cli_opts.chunk});
    std.debug.print("\n", .{});

    switch (cli_opts.cmd) {
        .bench => {
            const path = cli_opts.arg0 orelse return error.MissingWorkload;
            try bench(allocator, engine, &graph_scheduler, path, cli_opts, load_s);
        },
        .chat => {
            if (cli_opts.interactive) {
                return repl(allocator, engine, &tokenizer, cli_opts);
            }

            // 不给 prompt 时用对拍基线的 prompt(~/llmgen-golden.txt 就是它的 greedy 输出)
            const prompt_text = cli_opts.arg0 orelse "请详细介绍一下北京的历史和著名景点。";
            const templated = try tkz.qwenChatTemplate(allocator, prompt_text, false);
            defer allocator.free(templated);
            const prompt_ids = try tokenizer.encode(templated);
            defer allocator.free(prompt_ids);

            // 单条序列最多 max_model_len 个 token
            if (prompt_ids.len >= cli_opts.max_model_len) return error.PromptTooLong;
            const max_new = @min(cli_opts.max_new, cli_opts.max_model_len - prompt_ids.len);

            if (cli_opts.single) {
                // 单请求的旧路径(Executor.generate):LLMGEN_METRICS / LLMGEN_PROFILE 的打点都在这条路上
                runtime_executor.use_kv_cache = !cli_opts.no_kv_cache;
                const out_ids = try runtime_executor.generate(prompt_ids, max_new, cli_opts.temp, eos, &tokenizer);
                defer allocator.free(out_ids);
                printTokenIds(out_ids);
                if (runtime_executor.metrics.enabled) {
                    // BF16 权重; KV 的类型以 graph/scheduler.zig 里 initLayer 传入的 data_type 为准
                    const cost = metrics.ModelCost.fromConfig(model_cfg, 2, 2);
                    metrics.printEfficiency(&runtime_executor.metrics, cost);
                }
                return;
            }
            try chat(
                allocator,
                engine,
                &tokenizer,
                prompt_ids,
                max_new,
                cli_opts.temp,
                cli_opts.dup,
            );
        },
    }
}

/// 对拍基线:greedy 下这一行在每一步优化前后都应该完全相同
fn printTokenIds(ids: []const i32) void {
    std.debug.print("\ntoken ids:", .{});
    for (ids) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
}

/// 同一个 prompt 提交 dup 份, 流式打印第 0 份, 结束后打印第 0 份的 token ids;dup > 1 时检查各份输出完全相同
fn chat(allocator: std.mem.Allocator, engine: *Engine, tokenizer: *tkz.Tokenizer, prompt_ids: []const i32, max_new: usize, temp: f32, dup: usize) !void {
    const seqs = try allocator.alloc(*Sequence, dup);
    defer allocator.free(seqs);

    for (seqs) |*s| {
        s.* = try engine.addRequest(prompt_ids, .{
            .temperature = temp,
            .max_tokens = max_new,
            .ignore_eos = false,
        });
    }

    var timer = try std.time.Timer.start();
    var first_ns: u64 = 0;
    while (!engine.isFinished()) {
        _ = try engine.runStep();

        for (engine.lastEmitted()) |e| {
            if (e.seq != seqs[0]) {
                continue;
            }

            if (first_ns == 0) {
                first_ns = timer.read();
            }

            if (e.token_id == engine.eos) {
                continue;
            }

            const token_str = try tokenizer.decode(&[_]i32{e.token_id});
            defer allocator.free(token_str);

            std.debug.print("{s}", .{token_str});
        }
    }
    const end_ns = timer.read();

    // postprocess 会把 EOS 也追加进序列(据此判断结束);generate 遇到 EOS 不追加。打印时去掉, 和 golden 同口径
    var first = seqs[0].all_token_ids.items[seqs[0].num_prompt_tokens..];
    const n_out = first.len;
    if (first.len > 0 and first[first.len - 1] == engine.eos) first = first[0 .. first.len - 1];
    const dec_ms = @as(f64, @floatFromInt(end_ns - first_ns)) / 1e6;
    std.debug.print("\n---\n生成 {d} token | TTFT {d:.2} ms | decode {d:.3} ms/token ({d:.1} tok/s)", .{
        n_out,
        @as(f64, @floatFromInt(first_ns)) / 1e6,
        if (n_out > 1) dec_ms / @as(f64, @floatFromInt(n_out - 1)) else 0,
        if (n_out > 1) @as(f64, @floatFromInt(n_out - 1)) * 1e3 / dec_ms else 0,
    });
    printTokenIds(first);
    if (dup > 1) {
        var same: usize = 0;
        for (seqs) |s| {
            if (std.mem.startsWith(i32, s.all_token_ids.items[s.num_prompt_tokens..], first) and
                s.all_token_ids.items.len - s.num_prompt_tokens == n_out)
            {
                same += 1;
            }
        }
        std.debug.print("[chat] 批内一致性:{d}/{d} 份输出与第 0 份完全相同\n", .{ same, dup });
        if (same != dup) return error.BatchOutputsDiffer;
    }
    engine.releaseFinishedSeqs();
}

/// chat -i:交互式多轮对话。每轮把整段历史重新套模板、encode、提交一条请求(没有前缀缓存, 历史越长 TTFT 越大);
/// 超出 max_model_len 时从最早的一问一答开始丢
fn repl(allocator: std.mem.Allocator, engine: *Engine, tokenizer: *tkz.Tokenizer, o: cli.Options) !void {
    var history: std.ArrayList(tkz.ChatMessage) = .empty;
    defer {
        for (history.items) |m| allocator.free(m.content);
        history.deinit(allocator);
    }
    var think = false;

    // 每轮至少给回答留这么多 token, 剩下的给历史
    const reserve = @min(o.max_new, o.max_model_len / 2);

    // takeDelimiterInclusive 0.15.1 / 0.15.2 行为一致(takeDelimiter 0.15.2 才有)
    var in_buf: [64 * 1024]u8 = undefined;
    var stdin_reader = std.fs.File.stdin().reader(&in_buf);
    const stdin = &stdin_reader.interface;

    std.debug.print("[chat] 交互模式:/think 切换思考模式, /clear 清空历史, /exit 退出\n", .{});
    while (true) {
        std.debug.print("\n>>> ", .{});
        const raw = stdin.takeDelimiterInclusive('\n') catch |err| switch (err) {
            error.EndOfStream => break, // Ctrl-D
            else => return err,
        };
        const line = std.mem.trim(u8, raw, " \t\r\n");
        if (line.len == 0) continue;

        if (std.mem.eql(u8, line, "/exit") or std.mem.eql(u8, line, "/quit")) break;
        if (std.mem.eql(u8, line, "/clear")) {
            for (history.items) |m| allocator.free(m.content);
            history.clearRetainingCapacity();
            std.debug.print("[chat] 历史已清空\n", .{});
            continue;
        }
        if (std.mem.eql(u8, line, "/think")) {
            think = !think;
            std.debug.print("[chat] 思考模式:{s}\n", .{if (think) "开" else "关"});
            continue;
        }

        // line 指向 stdin 的缓冲区, 下次读就失效, 先复制
        try history.append(allocator, .{ .role = "user", .content = try allocator.dupe(u8, line) });

        // 套模板 + encode;放不下就丢最早的一问一答, 只剩本轮还放不下就放弃本轮
        var prompt_ids: []i32 = undefined;
        while (true) {
            const templated = try tkz.qwenChatTemplateMulti(allocator, o.system, history.items, think);
            defer allocator.free(templated);
            prompt_ids = try tokenizer.encode(templated);
            if (prompt_ids.len + reserve <= o.max_model_len or history.items.len < 3) break;
            allocator.free(prompt_ids);
            for (history.items[0..2]) |m| allocator.free(m.content);
            try history.replaceRange(allocator, 0, 2, &.{});
            std.debug.print("[chat] 上下文超长, 丢弃最早一轮对话\n", .{});
        }
        defer allocator.free(prompt_ids);
        if (prompt_ids.len >= o.max_model_len) {
            std.debug.print("[chat] 输入过长({d} token ≥ max_model_len {d}), 本轮忽略\n", .{ prompt_ids.len, o.max_model_len });
            allocator.free(history.pop().?.content);
            continue;
        }
        const max_new = @min(o.max_new, o.max_model_len - prompt_ids.len);

        const seq = try engine.addRequest(prompt_ids, .{ .temperature = o.temp, .max_tokens = max_new, .ignore_eos = false });

        var timer = try std.time.Timer.start();
        var first_ns: u64 = 0;
        while (!engine.isFinished()) {
            _ = try engine.runStep();
            for (engine.lastEmitted()) |e| {
                if (e.seq != seq) continue;
                if (first_ns == 0) first_ns = timer.read();
                if (e.token_id == engine.eos) continue;
                const token_str = try tokenizer.decode(&[_]i32{e.token_id});
                defer allocator.free(token_str);
                std.debug.print("{s}", .{token_str});
            }
        }
        const end_ns = timer.read();

        // 回答存进历史:去掉 EOS;思考模式下只留 </think> 之后的正文(同 Qwen3 官方模板)
        var out = seq.all_token_ids.items[seq.num_prompt_tokens..];
        const n_out = out.len;
        if (out.len > 0 and out[out.len - 1] == engine.eos) out = out[0 .. out.len - 1];
        const reply = try tokenizer.decode(out);
        defer allocator.free(reply);
        var content: []const u8 = reply;
        if (std.mem.lastIndexOf(u8, content, "</think>")) |i| content = content[i + "</think>".len ..];
        content = std.mem.trim(u8, content, " \t\r\n");
        try history.append(allocator, .{ .role = "assistant", .content = try allocator.dupe(u8, content) });
        engine.releaseFinishedSeqs();

        const dec_ms = @as(f64, @floatFromInt(end_ns - first_ns)) / 1e6;
        std.debug.print("\n--- prompt {d} token | 生成 {d} token | TTFT {d:.2} ms | {d:.1} tok/s\n", .{
            prompt_ids.len,
            n_out,
            @as(f64, @floatFromInt(first_ns)) / 1e6,
            if (n_out > 1) @as(f64, @floatFromInt(n_out - 1)) * 1e3 / dec_ms else 0,
        });
    }
}

/// 一轮请求同时提交, 跑到全部结束, 返回墙钟秒数。口径同 lab:
/// TTFT = 提交(t0)到首 token 回到 host;ITL = 同一请求相邻 token 的间隔(所有请求的样本合在一起)
fn runRound(allocator: std.mem.Allocator, engine: *Engine, reqs: []const workload.Request, ttft_ms: *std.ArrayList(f64), itl_ms: *std.ArrayList(f64), n_steps: *usize) !f64 {
    var timer = try std.time.Timer.start();
    const base_id = engine.next_id; // 本轮第 k 条请求的 seq.id = base_id + k
    const last_emit = try allocator.alloc(u64, reqs.len); // 每条请求上一个 token 回到 host 的时刻, 0 = 还没出 token
    defer allocator.free(last_emit);
    @memset(last_emit, 0);

    for (reqs) |r| {
        _ = try engine.addRequest(r.prompt_ids, .{ .temperature = r.temperature, .max_tokens = r.max_tokens, .ignore_eos = r.ignore_eos });
    }
    while (!engine.isFinished()) {
        _ = try engine.runStep();
        n_steps.* += 1;
        const now = timer.read();
        for (engine.lastEmitted()) |e| {
            const k: usize = @intCast(e.seq.id - base_id);
            const ms = @as(f64, @floatFromInt(now - last_emit[k])) / 1e6;
            if (last_emit[k] == 0) try ttft_ms.append(allocator, ms) else try itl_ms.append(allocator, ms);
            last_emit[k] = now;
        }
    }
    const wall_s = @as(f64, @floatFromInt(timer.read())) / 1e9;
    engine.releaseFinishedSeqs();
    return wall_s;
}

/// 按 workload 逐轮跑(轮内请求同时提交), 输出和 lab / nano-vllm 同字段的 JSON。吞吐 = 一轮的输出 token 数 ÷ 各轮墙钟的中位数
fn bench(
    allocator: std.mem.Allocator,
    engine: *Engine,
    sched: *GraphScheduler,
    path: []const u8,
    o: cli.Options,
    load_s: f64,
) !void {
    const parsed = try workload.load(allocator, path);
    defer parsed.deinit();
    const wl = parsed.value;
    if (wl.rounds.len == 0 or wl.rounds[0].len == 0) return error.EmptyWorkload;
    const reqs0 = wl.rounds[0];

    var in_tok: usize = 0;
    var out_tok: usize = 0; // workload 都是 ignore_eos, 每条恰好生成 max_tokens 个
    for (reqs0) |r| {
        in_tok += r.prompt_ids.len;
        out_tok += r.max_tokens;
    }
    std.debug.print("[bench] {s}: {d} 请求 × {d} 轮, 输入 {d} / 输出 {d} token\n", .{ wl.name, reqs0.len, wl.rounds.len, in_tok, out_tok });

    // 预热(同 lab / nano-vllm bench.py:一条短请求):GPU 升频、cuBLAS 初始化、第一次 graph launch
    {
        _ = try engine.addRequest(reqs0[0].prompt_ids[0..@min(8, reqs0[0].prompt_ids.len)], .{
            .temperature = 1.0,
            .max_tokens = 64,
            .ignore_eos = true,
        });

        while (!engine.isFinished()) {
            _ = try engine.runStep();
        }

        engine.releaseFinishedSeqs();
    }

    var ttft_ms: std.ArrayList(f64) = .empty;
    defer ttft_ms.deinit(allocator);
    var itl_ms: std.ArrayList(f64) = .empty;
    defer itl_ms.deinit(allocator);
    const walls = try allocator.alloc(f64, wl.rounds.len);
    defer allocator.free(walls);
    var n_steps: usize = 0;
    const pre0 = engine.batch_scheduler.num_preemptions;
    const graph0 = engine.executor.n_graph_steps;
    const eager0 = engine.executor.n_eager_steps;

    for (wl.rounds, 0..) |reqs, i| {
        walls[i] = try runRound(allocator, engine, reqs, &ttft_ms, &itl_ms, &n_steps);
        std.debug.print("[bench] 第 {d} 轮: {d:.3} s, {d:.1} tok/s\n", .{ i + 1, walls[i], @as(f64, @floatFromInt(out_tok)) / walls[i] });
    }

    // 所有请求都结束后, 块应该全部还回来;不等说明 postprocess / preempt 漏还了块
    const bm = engine.block_manager;
    if (bm.freeBlocksNum() != bm.num_blocks) std.debug.print("[bench] KV 块泄漏:空闲 {d}/{d}\n", .{ bm.freeBlocksNum(), bm.num_blocks });

    const wall = recorder.stats(walls).p50;
    const mb = 1024.0 * 1024.0;
    const dev = engine.executor.input_tensor.device_type;
    const dev_pools = sched.memory_manager.get(dev, 0);
    const weight_bytes: usize = if (dev_pools) |p| (if (p.weight) |w| w.used() else 0) else 0;
    const device_used: usize = if (opts.have_cuda and dev == .CUDA) cuda.CudaContext.instance().usedMemory() else 0;

    const Summary = struct {
        engine: []const u8,
        workload: []const u8,
        n_requests: usize,
        repeat: usize,
        input_tokens: usize,
        output_tokens: usize,
        wall_s: f64,
        throughput_out: f64,
        throughput_total: f64,
        ttft_ms: recorder.Stats,
        itl_ms: recorder.Stats,
        peak_rss_mb: f64,
        device_used_mb: f64,
        weights_mb: f64,
        kv_cache_mb: f64,
        kv_tokens: usize,
        load_s: f64,
        preemptions: usize,
        steps: usize,
        graph_steps: usize,
        eager_steps: usize,
        policy: []const u8,
    };
    const summary = Summary{
        .engine = if (dev == .CUDA) "llmgen" else "llmgen-cpu",
        .workload = wl.name,
        .n_requests = reqs0.len,
        .repeat = wl.rounds.len,
        .input_tokens = in_tok,
        .output_tokens = out_tok,
        .wall_s = wall,
        .throughput_out = @as(f64, @floatFromInt(out_tok)) / wall,
        .throughput_total = @as(f64, @floatFromInt(out_tok + in_tok)) / wall,
        .ttft_ms = recorder.stats(ttft_ms.items),
        .itl_ms = recorder.stats(itl_ms.items),
        .peak_rss_mb = recorder.peakRssMb(),
        .device_used_mb = @as(f64, @floatFromInt(device_used)) / mb,
        .weights_mb = @as(f64, @floatFromInt(weight_bytes)) / mb,
        .kv_cache_mb = @as(f64, @floatFromInt(sched.kv_bytes)) / mb,
        .kv_tokens = sched.num_kv_blocks * sched.config.block_size,
        .load_s = load_s,
        .preemptions = engine.batch_scheduler.num_preemptions - pre0,
        .steps = n_steps,
        .graph_steps = engine.executor.n_graph_steps - graph0,
        .eager_steps = engine.executor.n_eager_steps - eager0,
        .policy = @tagName(engine.batch_scheduler.batch_config.policy),
    };
    const js = try std.json.Stringify.valueAlloc(allocator, summary, .{ .whitespace = .indent_2 });
    defer allocator.free(js);
    std.debug.print("{s}\n", .{js});
    if (o.out) |p| try std.fs.cwd().writeFile(.{ .sub_path = p, .data = js });
}

test {
    _ = @import("runner/batching.zig");
    _ = @import("runner/block.zig");
    _ = @import("module/linear.zig");
    _ = @import("module/norm.zig");
    _ = @import("profiler/profiler.zig");
    _ = @import("sampling.zig");
    _ = @import("backend/cpu/attention.zig");
    if (opts.have_cuda) {
        _ = @import("backend/cuda/kernel_test.zig");
    }
}

test "bf16 RNE:平局向偶数, NaN 保持 NaN" {
    const c = @import("backend/cpu/common.zig");
    // 1.0 精确可表示
    try std.testing.expectEqual(@as(u16, 0x3F80), c.f32ToBf16(1.0));
    // 0x3F808000 正好在 0x3F80 和 0x3F81 中间, 偶数是 0x3F80
    try std.testing.expectEqual(@as(u16, 0x3F80), c.f32ToBf16(@bitCast(@as(u32, 0x3F80_8000))));
    // 0x3F818000 在 0x3F81 和 0x3F82 中间, 偶数是 0x3F82
    try std.testing.expectEqual(@as(u16, 0x3F82), c.f32ToBf16(@bitCast(@as(u32, 0x3F81_8000))));
    // 往返
    try std.testing.expectEqual(@as(f32, -2.5), c.bf16ToF32(c.f32ToBf16(-2.5)));
    try std.testing.expect(std.math.isNan(c.bf16ToF32(c.f32ToBf16(std.math.nan(f32)))));
}
