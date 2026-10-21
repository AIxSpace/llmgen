const std = @import("std");
const opts = @import("build_options");
const cuda = if (opts.have_cuda) @import("backend/cuda/cuda.zig") else struct {};

const types = @import("types.zig");
const Tensor = @import("Tensor.zig");

const common = @import("backend/cpu/common.zig");

const qwen3 = @import("models/qwen3.zig");

const graph = @import("graph/graph.zig");
const ComputeGraph = graph.ComputeGraph;

const scheduler = @import("graph/scheduler.zig");
const GraphScheduler = scheduler.GraphScheduler;

const module = @import("module/module.zig");
const module_base = @import("module/base.zig");
const computeStrides = module_base.computeStrides;

const pools = @import("memory/pools.zig");
const MemoryPool = pools.MemoryPool;
const MemoryManager = @import("memory/manager.zig").MemoryManager;
const PagedAttentionManager = @import("memory/page_atten.zig").PagedAttentionManager;
const memory_plan = @import("memory/plan.zig");

const ops = @import("ops.zig");

const profiler = @import("profiler/profiler.zig");
const OpProfiler = profiler.OpProfiler;
const metrics_mod = @import("profiler/metrics.zig");
const GenMetrics = metrics_mod.GenMetrics;
const report = @import("profiler/reporter.zig");

const sampling = @import("sampling.zig");

const tok = @import("tokenizer.zig");
const Tokenizer = tok.Tokenizer;

const batching = @import("./runner/batching.zig");
const Entry = batching.Entry;

const Sequence = @import("runner/batching.zig").Sequence;
const BlockManager = @import("runner/block.zig").BlockManager;
const cuda_attn = @import("backend/cuda/attention/attention_v1.zig");

const MAX_BUCKETS: usize = 64; // CUDA Graph 最多录几个桶

/// 原来一个 max_seq 管的四件事里, Executor 用到的三件 + 块大小(都来自 GraphScheduler.Config)
pub const Limits = struct {
    max_model_len: usize, // 单条序列最长 token 数:块表列数、位置上限
    max_tokens: usize, // 单趟前向最多 token 数 T:元数据 token 段、激活 token 区
    max_seqs: usize, // 一步最多几条序列 S:元数据序列段、序列区、CUDA Graph 桶
    block_size: usize, // KV 分页块大小

    pub fn of(c: scheduler.GraphScheduler.Config) Limits {
        return .{
            .max_model_len = c.max_model_len,
            .max_tokens = c.max_tokens,
            .max_seqs = c.max_num_seqs,
            .block_size = c.block_size,
        };
    }
    /// 块表列数:一条最长的序列占多少块
    pub fn maxBlocks(self: Limits) usize {
        return (self.max_model_len + self.block_size - 1) / self.block_size;
    }
};

/// prefill 注意力的 query tile 大小:CUDA 向 kernel 查询(= TC_BQ), CPU 的 attention 不读 tile 元数据, 给个默认值
fn prefillTile(device_type: types.DeviceType) usize {
    if (opts.have_cuda and device_type == .CUDA) {
        return @intCast(cuda_attn.cuda_attn_prefill_tile());
    }
    return 64;
}

/// Executor 常驻 workspace 各段的字节数(都从激活池切, markPersistent 固定)。
/// 调度器建池时用 total() 把它算进激活池容量, allocWorkspace 按同一份数切, 两边不会对不上。
pub const Workspace = struct {
    meta: usize, //        一步的元数据(Layout 按最大规模)
    part_o: usize = 0, //  decode split-K scratch:各段未归一化的输出 [S, nq, max_parts, hd] f32
    part_ml: usize = 0, // 各段的 (max, sum)                    [S, nq, max_parts, 2] f32
    sample: usize = 0, //  两段式 argmax 的中间候选

    pub fn of(cfg: qwen3.Config, lim: Limits, device_type: types.DeviceType) Workspace {
        const tile = prefillTile(device_type);
        var w: Workspace = .{
            .meta = Layout.init(lim.max_tokens, lim.max_seqs, lim.max_tokens / tile + lim.max_seqs, lim.maxBlocks()).total,
        };
        if (opts.have_cuda and device_type == .CUDA) {
            const nq: usize = @intCast(cfg.head_count);
            const hd: usize = @intCast(cfg.key_length);
            var cap: usize = 0;
            var s: usize = 1;
            // 各批大小档的 part_size 不同, 取 S × parts 的最大值
            while (s <= lim.max_seqs) : (s += 1) cap = @max(cap, s * cuda_attn.maxParts(s, lim.max_model_len));
            w.part_o = cap * nq * hd * 4;
            w.part_ml = cap * nq * 2 * 4;
            w.sample = cuda.cuda_sample_scratch_bytes(@intCast(lim.max_seqs));
        }
        return w;
    }

    /// 各段按 POOL_ALIGN 对齐后的总和, 再留一点余量给后面 CACHE 节点的对齐
    pub fn total(self: Workspace) usize {
        var sum: usize = 0;
        inline for (std.meta.fields(Workspace)) |f| sum += std.mem.alignForward(usize, @field(self, f.name), pools.POOL_ALIGN);
        return sum + 4 * pools.POOL_ALIGN;
    }
};

pub fn dispatchKernel(tensor: *Tensor, device_id: i32) !void {
    switch (tensor.op_type) {
        .OP_TYPE_NONE => {},
        .OP_TYPE_MEMCPY => try ops.kernel.memcpy(tensor, device_id),
        .OP_TYPE_EMBEDDING => try ops.kernel.embedding(tensor, device_id),
        .OP_TYPE_RMS_NORM => try ops.kernel.rmsNorm(tensor, device_id),
        .OP_TYPE_LINEAR => try ops.kernel.linear(tensor, device_id),
        .OP_TYPE_RESHAPE => try ops.kernel.reshape(tensor, device_id),
        .OP_TYPE_PERMUTE => try ops.kernel.permute(tensor, device_id),
        .OP_TYPE_APPLY_ROPE => try ops.kernel.applyRope(tensor, device_id),
        .OP_TYPE_ROPE_CACHE => try ops.kernel.ropeCache(tensor, device_id),
        .OP_TYPE_PAGED_ATTN => try ops.kernel.pagedAttention(tensor, device_id),
        .OP_TYPE_FLASH_ATTN => try ops.kernel.flashAttention(tensor, device_id),
        .OP_TYPE_SILU => try ops.kernel.silu(tensor, device_id),
        .OP_TYPE_MUL => try ops.kernel.mul(tensor, device_id),
        .OP_TYPE_ADD => try ops.kernel.add(tensor, device_id),
        .OP_TYPE_SOFTMAX => try ops.kernel.softmax(tensor, device_id),
        .OP_TYPE_GATHER_LAST => try ops.kernel.gatherLast(tensor, device_id),
        .OP_TYPE_SAMPLE => try ops.kernel.sample(tensor, device_id),
        .OP_TYPE_TRANSPOSE, .OP_TYPE_VIEW => {},
        else => return error.UnsupportedOp,
    }
}

/// 一步的元数据在 host/device 上的紧凑布局(字节偏移). 同一 (T,S,tiles) 布局恒定 → CUDA Graph 可复用.
const Layout = struct {
    seed: usize = 0,
    ids: usize,
    pos: usize,
    slots: usize,
    cu_q: usize,
    ctx: usize,
    last: usize,
    temps: usize,
    tile_seq: usize,
    tile_q0: usize,
    bt: usize,
    total: usize,

    fn init(n_tok: usize, n_seq: usize, n_tiles: usize, max_blocks: usize) Layout {
        var offset: usize = 8; // 前 8 字节留给 seed（u64）

        var l: Layout = undefined;
        l.seed = 0;

        // token ids
        l.ids = offset;
        offset += 4 * n_tok;

        // 每个 token 在自己序列里的位置
        l.pos = offset;
        offset += 4 * n_tok;

        // 每个 token 的 K/V 写到 KV cache 的哪个槽，块号 × 16 + 块内偏移；-1 表示不写
        l.slots = offset;
        offset += 4 * n_tok;

        // 每条序列的 query 在 T 里的前缀和：序列 s 占 [cu_q[s], cu_q[s+1])
        l.cu_q = offset;
        offset += 4 * (n_seq + 1);

        // 每条序列算完这步后 KV 里有多少 token
        l.ctx = offset;
        offset += 4 * n_seq;

        // 每条序列最后一个 token 在 T 里的行号
        l.last = offset;
        offset += 4 * n_seq;

        // 采样温度、本步随机种子
        l.temps = offset;
        offset += 4 * n_seq;

        // prefill 注意力把 query 切成 64 行一块，每块属于哪条序列、从第几个 query 开始
        l.tile_seq = offset;
        offset += 4 * n_tiles;

        l.tile_q0 = offset;
        offset += 4 * n_tiles;

        // 每条序列的逻辑块 → 物理块
        l.bt = offset;
        offset += 4 * n_seq * max_blocks;

        l.total = offset;

        return l;
    }
};

/// 执行方式的开关(命令行的 --eager / --sync-each)
pub const ExecOptions = struct {
    eager: bool = false, //     不录也不回放 CUDA Graph, 全部逐 kernel 发射
    sync_each: bool = false, // 每个 kernel 之后同步, 定位出错的 kernel(隐含 eager)
};

pub const Executor = struct {
    allocator: std.mem.Allocator,

    compute_graph: *ComputeGraph,
    graph_scheduler: *GraphScheduler,

    memory_manager: *MemoryManager,

    limit: Limits,
    seq_pos: usize = 0, // has processed token pos
    cur_tokens: usize = 0, // 本次 forward 的 token 数, 给 opCost 算注意力上下文

    model_cfg: qwen3.Config, // init 多收一个参数, llmgen.zig 里传 qwen3.Config.fromGGUF(gguf_info)

    max_blocks: usize, // 块表列数 = ceil(max_model_len / block_size)

    meta_host: [*]u8 = undefined, // 元数据的 host 缓冲(CUDA 下是 pinned)
    meta_pinned: bool = false, //    meta_host 是 cudaMallocHost 来的, deinit 要还
    meta_dev: usize = 0, //  元数据在显存里的地址(从 activation 池切, 常驻)

    step_seed: u64 = 0x5eed,

    original_dims: std.AutoHashMapUnmanaged(*Tensor, [4]i64) = .empty, // 原始维度快照

    input_tensor: *Tensor,
    output_tensor: *Tensor,

    use_kv_cache: bool = true,

    persistent_computed: bool = false,
    persistent_nodes: std.ArrayList(*Tensor) = .empty, // CACHE 节点(rope 表及其拷贝代理), 只算一次

    plan: *const memory_plan.MemoryPlan, // = &graph_scheduler.plan.?, 每步节点的地址由它算
    tok_arena: usize = 0, //       token 区基址:每 token 一行的激活都住这里
    seq_arena: usize = 0, //       序列区基址:gather_last 之后每序列一行的激活

    apply_rope_tensors: std.ArrayList(*Tensor) = .empty,

    prng: std.Random.DefaultPrng, // Smampler use

    profiler: OpProfiler,
    metrics: GenMetrics,

    sync_each: bool = false, // 是否开启每次同步，方便调试用
    pf_tile: usize = 64, // prefill 注意力的 query tile 大小, 必须和 kernel 的 TC_BQ 一致(init 里向 CUDA 查询)

    // CUDA Graph:纯 decode 步按批大小分桶, 每个桶录一张图, 之后每步只调一次 cudaGraphLaunch   (放在 pf_tile 后面)
    enforce_eager: bool = false, // --eager / 逐算子同步 / profile / 非 CUDA:不录也不回放, 全部逐 kernel 发射
    buckets: [MAX_BUCKETS]usize = undefined, // 每个桶的批大小(升序)
    n_buckets: usize = 0,
    graphs: [MAX_BUCKETS]?*anyopaque = [_]?*anyopaque{null} ** MAX_BUCKETS, // cudaGraphExec_t
    graph_out: [MAX_BUCKETS]usize = [_]usize{0} ** MAX_BUCKETS, // 每个桶回放时采样结果所在的显存地址
    n_graph_steps: usize = 0,
    n_eager_steps: usize = 0,

    block_manager: BlockManager, // KV 块池的分配:generate 自己用, Engine 借用(engine.block_manager 指向它)

    // execute 切成多趟时用(一步的 token 数超过 lim.max_tokens)
    pass_out: std.ArrayList(i32) = .empty, // 各条目的采样结果(按 entries 顺序)
    pass_group: std.ArrayList(Entry) = .empty, // 当前这一趟的条目片段
    pass_owner: std.ArrayList(usize) = .empty, // 每个片段属于 entries 里的第几条

    part_o: ?*anyopaque = null, //  decode split-K 的 scratch:各段未归一化的输出
    part_ml: ?*anyopaque = null, // 各段的 (max, sum)

    out_host: [*]i32 = undefined, //      采样结果的 pinned host 缓冲, max_seqs 个 i32
    sample_scratch: ?*anyopaque = null, // 两段式 argmax 的中间候选

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        compute_graph: *ComputeGraph,
        graph_scheduler: *GraphScheduler,
        memory_manager: *MemoryManager,
        model_cfg: qwen3.Config,
        exec_opts: ExecOptions,
    ) !Self {
        const lim = Limits.of(graph_scheduler.config);

        // cudaDeviceSynchronize():提前建好 CUDA 上下文
        if (opts.have_cuda) {
            cuda.CudaContext.instance().sync();
        }

        var input_tensor: ?*Tensor = null;
        for (compute_graph.getAllTensors()) |t| {
            if (t.tensor_type == .TENSOR_TYPE_INPUT and std.mem.eql(u8, t.name, "input_ids")) {
                input_tensor = t;
            }
        }
        const outs = compute_graph.getExecutionOutputs();
        if (input_tensor == null or outs.len == 0) {
            return error.BadGraph;
        }

        var original_dims: std.AutoHashMapUnmanaged(*Tensor, [4]i64) = .empty;
        errdefer original_dims.deinit(allocator);
        for (compute_graph.getAllTensors()) |t| {
            var dynamic = false;
            for (t.dims) |d| {
                if (d < 0) dynamic = true;
            }
            if (dynamic) {
                try original_dims.put(allocator, t, t.dims);
            }
        }

        const plan: *const memory_plan.MemoryPlan = if (graph_scheduler.plan) |*p| p else return error.NoMemoryPlan;

        var self: Self = .{
            .allocator = allocator,
            .plan = plan,
            .compute_graph = compute_graph,
            .graph_scheduler = graph_scheduler,
            .memory_manager = memory_manager,
            .model_cfg = model_cfg,
            .original_dims = original_dims,
            .input_tensor = input_tensor.?,
            .output_tensor = outs[0],
            .prng = std.Random.DefaultPrng.init(graph_scheduler.config.seed),
            .profiler = OpProfiler.init(),
            .metrics = GenMetrics.init(),
            // KV 是所有序列共享的块池:块数由调度器按显存预算算好(第 19/20 步 Engine 借用这一份, 不再另建)
            .block_manager = try BlockManager.init(allocator, graph_scheduler.num_kv_blocks, lim.block_size),
            .max_blocks = lim.maxBlocks(),
            .limit = lim,
        };
        errdefer self.deinit();

        // CACHE 节点只算一次(持久段);其余计算节点就是 graph_scheduler.step_nodes, 已经按规划排好
        for (compute_graph.getExecutionOrder()) |t| {
            if (t.isComputed() and t.tensor_type == .TENSOR_TYPE_CACHE) try self.persistent_nodes.append(allocator, t);
        }

        for (compute_graph.getAllTensors()) |tensor| {
            if (tensor.op_type == .OP_TYPE_APPLY_ROPE) {
                try self.apply_rope_tensors.append(allocator, tensor);
            }
        }

        // tile 大小决定元数据里 tile_seq / tile_q0 两段的长度, 要在 allocWorkspace 之前定下来
        self.pf_tile = prefillTile(self.input_tensor.device_type);

        try self.allocWorkspace();

        std.debug.print("[executor] KV {d} 块 × {d} token = {d} token ({d:.0} MB) | 单条最长 {d} | 单趟 {d} token / {d} 条\n", .{
            graph_scheduler.num_kv_blocks,
            lim.block_size,
            graph_scheduler.num_kv_blocks * lim.block_size,
            @as(f64, @floatFromInt(graph_scheduler.kv_bytes)) / (1024.0 * 1024.0),
            lim.max_model_len,
            lim.max_tokens,
            lim.max_seqs,
        });

        self.sync_each = exec_opts.sync_each;

        common.g_attn_mgr = memory_manager.getAttentionManager(input_tensor.?.device_type, 0);

        // 录制要跑真 kernel(KV 管理器、元数据都得先就绪), 所以放在 init 最后
        self.enforce_eager = exec_opts.eager or self.sync_each or self.profiler.enabled or
            !(opts.have_cuda and self.input_tensor.device_type == .CUDA);

        if (opts.have_cuda and !self.enforce_eager) {
            try self.captureGraphs();
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (opts.have_cuda) {
            for (self.graphs[0..self.n_buckets]) |g| {
                cuda.cuda_graph_destroy(g);
            }
        }

        // cudaMallocHost 分配的是锁页内存。只有锁页内存才能真正做异步 DMA：可分页内存的 "async"
        if (opts.have_cuda and self.meta_pinned) {
            cuda.cuda_host_free(self.meta_host);
            cuda.cuda_host_free(self.out_host);
        }

        self.original_dims.deinit(self.allocator);
        self.persistent_nodes.deinit(self.allocator);
        self.apply_rope_tensors.deinit(self.allocator);
        self.metrics.deinit(self.allocator);
        self.block_manager.deinit();

        self.pass_out.deinit(self.allocator);
        self.pass_group.deinit(self.allocator);
        self.pass_owner.deinit(self.allocator);
    }

    /// temperature ≤ 0 是 greedy;eos < 0 表示 ignore_eos(词表里没有负数 id, 永远不会命中)
    pub fn generate(self: *Self, prompt: []const i32, max_tokens: usize, temperature: f32, eos: i32, tokenizer: ?*Tokenizer) ![]i32 {
        if (prompt.len == 0) return error.EmptyPrompt;
        if (max_tokens == 0) return error.BadMaxTokens;

        // 超长的请求在入口就拒掉, 不要等到中途 OutOfBlocks
        if (prompt.len + max_tokens > self.limit.max_model_len) {
            return error.SequenceTooLong;
        }
        // prompt 一趟 prefill 完(超过 max_tokens 要切成多趟, 第 19 步的 execute 才支持)
        if (prompt.len > self.limit.max_tokens) {
            return error.PromptTooLong;
        }

        self.seq_pos = 0;

        var total_timer = try std.time.Timer.start();

        // op profiler
        self.profiler.reset();

        // metrics
        self.metrics.reset();
        self.metrics.prompt_tokens_len = prompt.len;

        // 单请求也用 Sequence 表示:token_ids = prompt + 已生成。第 19 步多条序列时 forward 不用再改。
        var seq = try Sequence.init(
            self.allocator,
            0,
            prompt,
            .{
                .temperature = temperature,
                .max_tokens = max_tokens,
                .ignore_eos = eos < 0,
            },
        );
        defer {
            self.block_manager.deallocate(&seq); // 块还给空闲栈
            seq.deinit();
        }

        if (!try self.block_manager.ensure(&seq, prompt.len)) {
            return error.OutOfBlocks;
        }

        var next = (try self.forward(&.{.{
            .seq = &seq,
            .q_start = 0,
            .q_len = prompt.len,
        }}))[0];

        self.seq_pos = prompt.len;
        self.metrics.prefill_ns = total_timer.read();
        self.profiler.phase = .decode; // 进入 decode 阶段 profiler 统计

        var idx: usize = 0;

        var step_timer = try std.time.Timer.start();
        while (idx < max_tokens) : (idx += 1) {
            step_timer.reset(); // 采样已经在 forward 里做完了, sample_ns 从此恒为 0

            // 时间戳一直记(一次 Timer.read + 一次 append, 几十纳秒);enabled 只控制要不要打印。第一个 token 时顺带记下 TTFT
            try self.metrics.onGenToken(self.allocator, total_timer.read());

            if (next == eos) {
                break;
            }

            try seq.all_token_ids.append(self.allocator, next);
            self.metrics.gen_tokens_len += 1;

            if (tokenizer) |tk| {
                const s = try tk.decode(&[_]i32{next});
                defer self.allocator.free(s);
                std.debug.print("{s}", .{s});
            }
            self.metrics.detok_ns += step_timer.lap();

            // if (output.items.len >= self.max_seq) {
            //     break;
            // }

            // 最后一个 token 已产出, 再 forward 一次也不会被 sample, 直接结束
            if (idx + 1 == max_tokens) {
                break;
            }

            if (!try self.block_manager.ensure(&seq, seq.len())) {
                return error.OutOfBlocks;
            }

            // 整段重跑(调试用)时从 0 开始:同样的 token 写回同样的 slot, KV 不用先清
            const q_start: usize = if (self.use_kv_cache) seq.len() - 1 else 0;
            next = (try self.forward(&.{.{
                .seq = &seq,
                .q_start = q_start,
                .q_len = seq.len() - q_start,
            }}))[0];

            self.seq_pos = seq.len();
        }

        self.metrics.e2e_ns = total_timer.read();
        if (self.metrics.enabled) {
            std.debug.print("[executor] 累计 graph 步 {d} / eager 步 {d}\n", .{ self.n_graph_steps, self.n_eager_steps });
            self.metrics.print();
            metrics_mod.printMemory(self.memory_manager);
            const backend = if (opts.have_cuda) "cuda" else "cpu";
            report.writeJson(self.metrics.toReport(backend)) catch |err| {
                std.debug.print("写 metrics json 失败: {s}\n", .{@errorName(err)});
            };
        }

        if (self.profiler.enabled) {
            self.profiler.printPhase(.prefill, self.metrics.prefill_ns);
            self.profiler.printPhase(.decode, self.metrics.e2e_ns - self.metrics.prefill_ns);
            if (self.metrics.enabled) {
                self.metrics.printOverhead(self.profiler.opNs(.decode));
            }
        }

        return self.allocator.dupe(i32, seq.all_token_ids.items[prompt.len..]);
    }

    /// 一趟前向:返回 entries 里每一条的采样 token(按 entries 顺序)。token 总数必须 ≤ lim.max_tokens, 条目数 ≤ lim.max_seqs
    fn forward(self: *Self, entries: []const Entry) ![]const i32 {
        var timer = try std.time.Timer.start();
        defer self.metrics.forward_ns += timer.read(); // prefill + decode

        var n_tok: usize = 0;
        var decode = true; // 只有"每条都是 1 个 token 且在序列末尾"才算纯 decode 步
        for (entries) |e| {
            n_tok += e.q_len;
            if (!(e.q_len == 1 and e.emits())) decode = false;
        }
        const n_seq = entries.len;

        // 纯 decode 步找一个装得下的桶;prefill / --eager / profile / 逐算子同步时不用图
        const bi: ?usize = if (decode and !self.enforce_eager) self.bucketFor(n_seq) else null;
        const seq_num = if (bi) |b| self.buckets[b] else n_seq; // 含 padding 行的序列数
        const token_num = if (decode) seq_num else n_tok; // decode 时每条 1 个 token

        var n_tiles: usize = 0;
        if (!decode) {
            for (entries) |e| {
                n_tiles += (e.q_len + self.pf_tile - 1) / self.pf_tile;
            }
        }
        self.cur_tokens = n_tok;

        ops.setGlobalAllocator(self.allocator);

        // 回放时 Layout 必须和录制时完全相同(Layout.init(Sn, Sn, 0, ..)), 否则各段偏移对不上
        const layout = Layout.init(
            token_num,
            seq_num,
            n_tiles,
            self.max_blocks,
        );
        self.fillMeta(layout, entries, seq_num, n_tiles);
        self.uploadMeta(layout.total); // 一次性拷到显存

        // opts.have_cuda 是 comptime:CPU 构建不分析这段, 不需要 cuda_graph_* 符号
        if (opts.have_cuda) {
            if (bi) |b| {
                // 回放:一次调用, 整步的 kernel 按录制时的参数排进 stream。
                // runNodes / dispatchKernel / 各 Impl 都不执行, 所以 resolveDims / bindStep / plan.bind 也不用调
                cuda.cuda_graph_launch(self.graphs[b]);
                self.n_graph_steps += 1;
                return self.fetchOutAt(self.graph_out[b], n_seq);
            }
        }

        self.resolveDims(@intCast(seq_num), @intCast(token_num));
        // 把 devAt(layout.xxx) 这些地址交给 kernel
        self.bindStep(layout, token_num, seq_num, n_tiles, decode);
        self.plan.bind(self.tok_arena, self.seq_arena, token_num, seq_num);

        try self.runPersistentOnce();

        // 每步节点按执行序派发, 不再分配也不再回卷
        try self.runNodes();
        self.n_eager_steps += 1;

        return self.fetchOutAt(@intFromPtr(self.output_tensor.data.?), n_seq);
    }

    /// 只算一次的 CACHE 节点(rope 表及其拷贝代理):从 workspace 之后 bump, 之后再也不回卷。
    /// 录图之前必须先算好:它们不能被录进图里每步重算
    fn runPersistentOnce(self: *Self) !void {
        if (self.persistent_computed) return;
        self.memory_manager.resetAllActivations();
        for (self.persistent_nodes.items) |t| try self.executorTensor(t);
        self.persistent_computed = true;
    }

    fn bucketFor(self: *const Self, s: usize) ?usize {
        for (self.buckets[0..self.n_buckets], 0..) |b, i| {
            if (b >= s) return i;
        }
        return null;
    }

    /// 纯 decode 步按批大小分桶录制。桶:1, 2, 4, 8, 16, 32, 48, ... ≤ MAX_SEQS, 再补上 max_seqs 本身
    fn captureGraphs(self: *Self) !void {
        var list: std.ArrayList(usize) = .empty;
        defer list.deinit(self.allocator);

        for ([_]usize{ 1, 2, 4, 8 }) |b| {
            if (b <= self.limit.max_seqs) {
                try list.append(self.allocator, b);
            }
        }
        var b: usize = 16;

        while (b <= self.limit.max_seqs) : (b += 16) {
            try list.append(self.allocator, b);
        }

        if (list.items[list.items.len - 1] != self.limit.max_seqs) {
            try list.append(self.allocator, self.limit.max_seqs);
        }

        if (list.items.len > MAX_BUCKETS) {
            return error.TooManyBuckets;
        }

        // RoPE 表等只算一次的节点先算好, 不能录进图里
        try self.runPersistentOnce();

        var timer = try std.time.Timer.start();
        for (list.items, 0..) |bs, i| {
            const lay = Layout.init(bs, bs, 0, self.max_blocks);
            // 预热一次:让驱动加载好 kernel、cuBLAS 选好算法。内容无所谓:全是 padding 行, ctx = 0 / slot = -1, 不读写 KV
            self.fillMeta(lay, &.{}, bs, 0);
            self.uploadMeta(lay.total);
            self.resolveDims(@intCast(bs), @intCast(bs));
            self.bindStep(lay, bs, bs, 0, true);
            self.plan.bind(self.tok_arena, self.seq_arena, bs, bs);
            try self.runNodes();
            cuda.CudaContext.instance().sync();

            if (cuda.cuda_graph_begin() != 0) return error.GraphCaptureFailed;
            try self.runNodes(); // 这一遍 kernel 不执行, 只被记下来
            const g = cuda.cuda_graph_end() orelse return error.GraphCaptureFailed;
            self.buckets[i] = bs;
            self.graphs[i] = g;
            self.graph_out[i] = @intFromPtr(self.output_tensor.data.?);
            self.n_buckets = i + 1;
        }
        std.debug.print("[executor] CUDA Graph: 录了 {d} 个桶(最大批 {d}), 耗时 {d:.1} ms\n", .{
            self.n_buckets,
            self.buckets[self.n_buckets - 1],
            @as(f64, @floatFromInt(timer.read())) / 1e6,
        });
    }

    fn runNodes(self: *Self) !void {
        for (self.graph_scheduler.step_nodes.items) |t| {
            try self.executorTensor(t);
        }
    }

    /// 取回采样结果(前 n_seq 个, padding 行的不要)。回放时 output_tensor.data 不一定是这个桶的地址, 所以由调用方给
    fn fetchOutAt(self: *Self, out_addr: usize, n_seq: usize) []const i32 {
        const out: ?*anyopaque = @ptrFromInt(out_addr);
        if (opts.have_cuda and self.output_tensor.device_type == .CUDA) {
            cuda.cuda_d2h_async(self.out_host, out, n_seq * @sizeOf(i32));
            cuda.CudaContext.instance().sync(); // 一步唯一的一次同步
            return self.out_host[0..n_seq];
        }
        return @as([*]const i32, @ptrCast(@alignCast(out)))[0..n_seq];
    }

    fn executorTensor(self: *Self, tensor: *Tensor) !void {
        // 视图和它的根同址, plan.bind 已经设好(和 plan.build 用同一个判据:tensor_type == VIEW)
        if (tensor.tensor_type == .TENSOR_TYPE_VIEW) return;

        const device_id = self.graph_scheduler.deviceIdForLayer(tensor.layer_id) orelse 0;

        // 只算一次的 CACHE 节点(rope 表及其拷贝代理)不在规划里, 仍然从池里 bump 一块
        if (tensor.tensor_type == .TENSOR_TYPE_CACHE) {
            try self.allocateOutput(tensor, device_id);
        }

        // 需要分析时开启，不要影响指标 metrics 评估
        if (self.profiler.enabled) {
            // 算子级计时必须等 kernel 真做完, 所以 profile 模式下仍然逐算子同步;这个模式下的吞吐不能当成绩
            var timer = try std.time.Timer.start();
            try dispatchKernel(tensor, @intCast(device_id));
            if (opts.have_cuda and tensor.device_type != .CPU) {
                // 这里目前是同步操作，所以 OpProfiler 能算准
                cuda.CudaContext.instance().sync(); // cuda 同步操作
            }
            const ns = timer.read();
            const ctx: u64 = @intCast(self.seq_pos + self.cur_tokens); // forward 期间 seq_pos 还没更新
            self.profiler.record(
                tensor.op_type,
                tensor.device_type,
                ns,
                profiler.opCost(tensor, ctx),
            );
        } else {
            try dispatchKernel(tensor, @intCast(device_id));

            // 需要逐算子同步的时候才开启
            if (self.sync_each and opts.have_cuda and tensor.device_type != .CPU) {
                cuda.CudaContext.instance().sync();
            }
        }
    }

    fn allocWorkspace(self: *Self) !void {
        const device_type = self.input_tensor.device_type;
        const device_pools = self.memory_manager.get(device_type, 0) orelse return error.NoDevicePools;
        const pool = &(device_pools.activation orelse return error.NoActivationPool);
        const carve = struct {
            fn f(p: *MemoryPool, bytes: usize) !usize {
                return @intFromPtr((try p.allocate(bytes, pools.POOL_ALIGN)).ptr);
            }
        }.f;

        const ws = Workspace.of(self.model_cfg, self.limit, device_type);
        self.tok_arena = try carve(pool, self.plan.tokArenaBytes(self.limit.max_tokens));
        self.seq_arena = try carve(pool, self.plan.seqArenaBytes(self.limit.max_seqs));
        self.meta_dev = try carve(pool, ws.meta);

        if (opts.have_cuda and device_type == .CUDA) {
            self.part_o = @ptrFromInt(try carve(pool, ws.part_o));
            self.part_ml = @ptrFromInt(try carve(pool, ws.part_ml));
            self.sample_scratch = @ptrFromInt(try carve(pool, ws.sample));
            self.meta_host = @ptrCast(cuda.cuda_host_alloc(ws.meta) orelse return error.PinnedAllocFailed);
            self.out_host = @ptrCast(@alignCast(cuda.cuda_host_alloc(self.limit.max_seqs * @sizeOf(i32)) orelse return error.PinnedAllocFailed));
            self.meta_pinned = true;
        } else {
            self.meta_host = @ptrFromInt(self.meta_dev); // CPU 的池 host 可寻址, 直接往里写
        }

        // 以上全是常驻区:resetAllActivations 只回卷到这里
        pool.markPersistent();

        const mb = 1024.0 * 1024.0;
        std.debug.print("[executor] 激活 token 区 {d:.1} MB ({d} B/token) + 序列区 {d:.2} MB | workspace {d:.2} MB\n", .{
            @as(f64, @floatFromInt(self.plan.tokArenaBytes(self.limit.max_tokens))) / mb,
            self.plan.tok_row_bytes,
            @as(f64, @floatFromInt(self.plan.seqArenaBytes(self.limit.max_seqs))) / mb,
            @as(f64, @floatFromInt(ws.total())) / mb,
        });
    }

    fn i32At(self: *Self, off: usize) [*]i32 {
        return @ptrCast(@alignCast(self.meta_host + off));
    }
    fn f32At(self: *Self, off: usize) [*]f32 {
        return @ptrCast(@alignCast(self.meta_host + off));
    }
    fn devAt(self: *const Self, off: usize) ?*anyopaque {
        return @ptrFromInt(self.meta_dev + off);
    }

    fn uploadMeta(self: *Self, bytes: usize) void {
        if (opts.have_cuda and self.input_tensor.device_type == .CUDA) {
            cuda.cuda_h2d_async(self.devAt(0), self.meta_host, bytes);
        }
    }

    /// 把本步的 Entry 列表写成紧凑元数据。Sn ≥ entries.len:多出来的是 CUDA Graph 桶的 padding 行
    fn fillMeta(self: *Self, layout: Layout, entries: []const Entry, seq_num: usize, n_tiles: usize) void {
        // 随机种子
        self.step_seed +%= 0x9E3779B97F4A7C15;
        std.mem.writeInt(u64, self.meta_host[0..8], self.step_seed, .little);

        // meta_host（一块连续内存）
        // ┌──────┬─────┬─────┬───────┬──────┬─────┬──────┬───────┬──────────┬─────────┬──────────────┐
        // │ seed │ ids │ pos │ slots │ cu_q │ ctx │ last │ temps │ tile_seq │ tile_q0 │  bt (S × mb) │
        // │  8B  │ 4T  │ 4T  │  4T   │4(S+1)│ 4S  │  4S  │  4S   │  4·tiles │ 4·tiles │   4·S·mb     │
        // └──────┴─────┴─────┴───────┴──────┴─────┴──────┴───────┴──────────┴─────────┴──────────────┘
        //  ← 逐 token 的 3 段 →  ← 逐序列的 4 段 →         ← prefill tile →    ← 块表 →
        // 一次性上传到 GPU
        // 对应的缓冲区偏移
        // T=6，S=3，n_tiles=3，mb=4：
        // seed     [  0,   8)
        // ids      [  8,  32)    6 × 4
        // pos      [ 32,  56)
        // slots    [ 56,  80)
        // cu_q     [ 80,  96)    4 × 4
        // ctx      [ 96, 108)
        // last     [108, 120)
        // temps    [120, 132)
        // tile_seq [132, 144)
        // tile_q0  [144, 156)
        // bt       [156, 204)    3 × 4 × 4
        // total = 204 字节  → 一次 H2D 拷贝

        const mb = self.max_blocks;
        const ids = self.i32At(layout.ids);
        const pos = self.i32At(layout.pos);
        const slots = self.i32At(layout.slots);
        const cu_q = self.i32At(layout.cu_q);
        const ctx = self.i32At(layout.ctx);
        const last = self.i32At(layout.last);
        const temps = self.f32At(layout.temps);
        const tseq = self.i32At(layout.tile_seq);
        const tq0 = self.i32At(layout.tile_q0);
        const bt = self.i32At(layout.bt);

        var t: usize = 0; // ★ 全局游标：下一个 token 写到第几行
        var tile: usize = 0; // prefill tile 的游标
        cu_q[0] = 0;
        for (entries, 0..) |entry_item, seq_idx| { // seq_idx = 第几条序列
            // 把这条序列本步要算的 q_len 个 token 依次追加到第 t 行
            const seq = entry_item.seq;
            for (0..entry_item.q_len) |j| {
                const p = entry_item.q_start + j;
                ids[t] = seq.all_token_ids.items[p]; // token ids
                pos[t] = @intCast(p); // position
                slots[t] = seq.slotOf(p, self.limit.block_size); // block slot
                t += 1;
            }

            // 这条序列写完了，记录它的边界和属性
            cu_q[seq_idx + 1] = @intCast(t);
            ctx[seq_idx] = @intCast(entry_item.q_start + entry_item.q_len);
            last[seq_idx] = @intCast(t - 1);
            temps[seq_idx] = seq.params.temperature;

            // 块表：拷进第 seq_idx 行，不足 max_blocks 的部分补 0
            const row = bt + seq_idx * mb;
            const nb = seq.block_table.items.len;
            @memcpy(row[0..nb], seq.block_table.items);
            @memset(row[nb..mb], 0);

            // prefill：把这条序列的 query 按 pf_tile(64) 切块
            if (n_tiles > 0) {
                var q0: usize = 0;
                while (q0 < entry_item.q_len) : (q0 += self.pf_tile) {
                    tseq[tile] = @intCast(seq_idx);
                    tq0[tile] = @intCast(q0);
                    tile += 1;
                }
            }
        }

        // padding 行:ctx = 0(注意力输出 0)、slot = -1(不写 KV);采样出来的 token 没人读
        var s = entries.len;
        while (s < seq_num) : (s += 1) {
            ids[t] = 0;
            pos[t] = 0;
            slots[t] = -1;
            t += 1;
            cu_q[s + 1] = @intCast(t);
            ctx[s] = 0;
            last[s] = @intCast(t - 1);
            temps[s] = 0;
            @memset((bt + s * mb)[0..mb], 0);
        }
    }

    /// 把本步的规模和各段元数据的显存地址交给 kernel(经 common.g_step)。
    fn bindStep(self: *Self, lay: Layout, n_tok: usize, n_seq: usize, n_tiles: usize, decode: bool) void {
        self.input_tensor.data = @ptrCast(self.devAt(lay.ids));
        const cfg = self.model_cfg;

        ops.setStep(.{
            .n_tok = @intCast(n_tok),
            .n_seq = @intCast(n_seq),
            .n_tiles = @intCast(n_tiles),
            .decode = decode,
            .hidden = @intCast(cfg.embedding_length),
            .n_head = @intCast(cfg.head_count),
            .n_kv_head = @intCast(cfg.head_count_kv),
            .head_dim = @intCast(cfg.key_length),
            .attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.key_length))),
            .block_size = @intCast(self.limit.block_size),
            .max_blocks = @intCast(self.max_blocks),
            .max_model_len = self.limit.max_model_len,
            .seed = self.devAt(lay.seed),
            .pos = self.devAt(lay.pos),
            .slots = self.devAt(lay.slots),
            .cu_q = self.devAt(lay.cu_q),
            .ctx = self.devAt(lay.ctx),
            .last = self.devAt(lay.last),
            .temps = self.devAt(lay.temps),
            .tile_seq = self.devAt(lay.tile_seq),
            .tile_q0 = self.devAt(lay.tile_q0),
            .block_tables = self.devAt(lay.bt),
            .part_o = self.part_o,
            .part_ml = self.part_ml,
            .sample_scratch = self.sample_scratch,
        });
    }

    // 1. 找到设备内存池
    // 2. 找到对应的激活池
    // 3. 填充数据和 offset
    fn allocateOutput(self: *Self, out_tensor: *Tensor, device_id: usize) !void {
        const device_pools = self.memory_manager.get(out_tensor.device_type, device_id) orelse return error.NoDevicePools;
        const activation_pool = &(device_pools.activation orelse return error.NoActivationPool);

        const nbytes = out_tensor.bytes();
        if (nbytes == 0) {
            return error.ZeroByteTensor; // 维度没解析(还含 -1)
        }

        const memory_block = try activation_pool.allocate(out_tensor.bytes(), 32);

        out_tensor.data = memory_block.ptr;
        out_tensor.offset = memory_block.offset;
    }

    fn sampleLast(self: *Self) i32 {
        const tensor = self.output_tensor;
        const r = tensor.rank();
        const vocab_size: usize = @intCast(tensor.dims[r - 1]);
        const rows = tensor.numElements() / vocab_size;
        const last = (rows - 1) * vocab_size;

        // logits 现在是 bf16:按 dtype 读成 f32 再 argmax
        var best_idx: usize = 0;
        var best_val: f32 = common.elemF32(tensor, last);
        for (1..vocab_size) |i| {
            const v = common.elemF32(tensor, last + i);
            if (v > best_val) {
                best_val = v;
                best_idx = i;
            }
        }
        return @intCast(best_idx);
    }

    /// 把 logits 最后一个 token 位读成 f32(logits 是 bf16, 在 host 上的回拷代理里)。调用方 free。
    fn lastLogits(self: *Self) ![]f32 {
        const t = self.output_tensor;
        const vocab: usize = @intCast(t.dims[t.rank() - 1]);
        const last = (t.numElements() / vocab - 1) * vocab;
        const row = try self.allocator.alloc(f32, vocab);
        for (row, 0..) |*v, i| v.* = common.elemF32(t, last + i);
        return row;
    }

    /// greedy(对拍 / 调试用)
    pub fn sampleArgmax(self: *Self) !i32 {
        const row = try self.lastLogits();
        defer self.allocator.free(row);
        return @intCast(sampling.argmax(row));
    }

    /// 对照 C++ sample():softmax(带温度) → top-p 截断后抽样
    pub fn sample(self: *Self) !i32 {
        const cfg = self.graph_scheduler.config;
        const row = try self.lastLogits();
        defer self.allocator.free(row);
        const probs = try sampling.softmax(self.allocator, row, cfg.temperature);
        defer self.allocator.free(probs);
        return @intCast(try sampling.sampleFromProbs(self.allocator, probs, cfg.top_p, self.prng.random()));
    }

    // 每次 forward 前,把整张图里所有带 -1(动态 seq_len)的张量,按本步实际的 token 数把维度填实,并重算 strides。 它是"单张图同时服务 prefill 和 decode"的开关
    // 为什么需要它
    // 建图时只建一张图,但图里 seq_len 这个维度建图时还不知道多长,就用 -1 占位(比如 input_ids 是 [-1,0,0,0],logits 是 [-1, vocab, 0,0])。而一次生成里 seq_len 会变两种值:
    // - prefill:一次喂整段 prompt → seq_len = prompt 长度
    // - decode:每步喂 1 个 token → seq_len = 1
    /// 把动态维填实:DYN(-1) → 本步 token 总数, DYN_S(-2) → 本步序列数
    pub fn resolveDims(self: *Self, n_seq: i64, n_tok: i64) void {
        var it = self.original_dims.iterator();
        while (it.next()) |entry| {
            const t = entry.key_ptr.*;
            t.dims = entry.value_ptr.*;
            for (&t.dims) |*d| {
                if (d.* == module_base.DYN_S) {
                    d.* = n_seq;
                } else if (d.* < 0) {
                    d.* = n_tok;
                }
            }
            computeStrides(t);
        }
    }

    /// 跑一步. 返回 entries 里每一条的采样 token(按 entries 顺序; 不 emits 的条目其值无意义).
    /// token 数超过单趟上限时切成多趟(条目的 query 段可被切开, 只有末段的采样结果有意义).
    pub fn execute(self: *Executor, entries: []const Entry) ![]const i32 {
        if (entries.len > self.limit.max_seqs) {
            return error.TooManySeqsInStep;
        }

        var total: usize = 0;
        for (entries) |e| {
            total += e.q_len;
        }

        // 单次 forward 可以完成，覆盖大部分情况
        if (total <= self.limit.max_tokens) {
            return self.forward(entries);
        }

        // 拆成多次
        try self.pass_out.resize(self.allocator, entries.len);
        self.pass_group.clearRetainingCapacity();
        self.pass_owner.clearRetainingCapacity();

        var room = self.limit.max_tokens; // 当前这趟还可以装多少 token
        for (entries, 0..) |e, idx| {
            var start = e.q_start;
            var left = e.q_len;
            while (left > 0) {
                const take = @min(left, room);
                try self.pass_group.append(self.allocator, .{ .seq = e.seq, .q_start = start, .q_len = take });
                try self.pass_owner.append(self.allocator, idx); // 记录片段和 entry 对应关系
                start += take;
                left -= take;
                room -= take;
                if (room == 0) {
                    try self.flushPass(entries);
                    room = self.limit.max_tokens;
                }
            }
        }
        try self.flushPass(entries);
        return self.pass_out.items;
    }

    fn flushPass(self: *Executor, entries: []const Entry) !void {
        if (self.pass_group.items.len == 0) {
            return;
        }

        const toks = try self.forward(self.pass_group.items);

        for (self.pass_group.items, self.pass_owner.items, toks) |piece, owner, tk| {
            const e = entries[owner];
            if (piece.q_start + piece.q_len == e.q_start + e.q_len) {
                self.pass_out.items[owner] = tk;
            }
        }

        self.pass_group.clearRetainingCapacity();
        self.pass_owner.clearRetainingCapacity();
    }
};
