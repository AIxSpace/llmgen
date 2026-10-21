const std = @import("std");

const batching = @import("batching.zig");
const Sequence = batching.Sequence;
const BatchScheduler = batching.BatchScheduler;
const Policy = batching.Policy;
const Entry = batching.Entry;
const SamplingParams = batching.SamplingParams;

const block = @import("block.zig");
const BlockManager = block.BlockManager;

const Executor = @import("../executor.zig").Executor;
const EngineStats = @import("stats.zig").EngineStats;

/// 调度相关的配置。KV 的块数 / 块大小 / 单条最长 / 单步最多序列数都在 Executor.lim 和它的 BlockManager 里, 这里不重复
pub const EngineConfig = struct {
    policy: Policy = .prefill_first,
    chunk_tokens: usize = 2048,
    max_num_batched_tokens: usize = 16384, // 一步最多排多少 token;超过 Executor 的 max_tokens 时 execute 切成多趟
    eos: i32 = -1,
};

pub const Emitted = struct {
    seq: *Sequence,
    token_id: i32,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,

    cfg: EngineConfig,

    all_seqs: std.ArrayList(*Sequence) = .empty,
    next_id: u64 = 0, // sequence next id
    eos: i32,

    last: []const Entry = &.{},
    emitted: std.ArrayList(Emitted) = .empty,

    batch_scheduler: BatchScheduler,
    block_manager: *BlockManager, // 借用 executor.block_manager:KV 只有一个池, 块的分配也只能有一份
    executor: *Executor,

    clock: std.time.Timer, // 所有 Sequence 时间戳的共同起点
    stats: EngineStats = .{},

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, executor: *Executor, cfg: EngineConfig) !*Self {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);

        const executor_block_manager = &executor.block_manager;
        self.* = .{
            .allocator = allocator,
            .cfg = cfg,
            .batch_scheduler = BatchScheduler.init(allocator, executor_block_manager, .{
                .max_scheduled_len = executor.limit.max_seqs, // 不能超过 Executor 按它分配的元数据 / 序列区 / CUDA Graph 桶
                .max_batched_tokens_len = cfg.max_num_batched_tokens,
                .policy = cfg.policy,
                .chunk_tokens = cfg.chunk_tokens,
                .eos = cfg.eos,
            }),
            .block_manager = executor_block_manager,
            .executor = executor,
            .eos = cfg.eos,
            .clock = try std.time.Timer.start(),
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        for (self.all_seqs.items) |s| {
            self.block_manager.deallocate(s); // 没跑完的序列(比如出错提前退出)把块还回去
            s.deinit();
            self.allocator.destroy(s);
        }
        self.all_seqs.deinit(self.allocator);
        self.emitted.deinit(self.allocator);
        self.batch_scheduler.deinit();
        self.allocator.destroy(self);
    }

    pub fn addRequest(self: *Self, prompt: []const i32, params: SamplingParams) !*Sequence {
        if (prompt.len == 0 or params.max_tokens == 0) {
            return error.EmptyRequest;
        }

        if (prompt.len + params.max_tokens > self.executor.limit.max_model_len) {
            return error.SequenceTooLong;
        }

        const seq = try self.allocator.create(Sequence);
        errdefer self.allocator.destroy(seq);

        seq.* = try Sequence.init(self.allocator, self.next_id, prompt, params);
        errdefer seq.deinit();

        seq.arrival_ns = self.clock.read();

        try self.batch_scheduler.addSeq(seq);
        try self.all_seqs.append(self.allocator, seq);

        self.next_id += 1;

        return seq;
    }

    pub fn runStep(self: *Self) !usize {
        var t = try std.time.Timer.start();
        const entries = try self.batch_scheduler.schedule();
        const sched_ns = t.lap(); // 调度开销

        self.last = entries;
        if (entries.len == 0) {
            return error.SchedulerStuck;
        }

        const now0 = self.clock.read();
        for (entries) |e| {
            if (e.seq.scheduled_ns == 0) e.seq.scheduled_ns = now0;
        }

        const out_token_ids = try self.executor.execute(entries);
        const exec_ns = t.lap(); // forward + sample

        self.emitted.clearRetainingCapacity();
        var buf: std.ArrayList(i32) = .empty;
        defer buf.deinit(self.allocator);

        const now = self.clock.read();
        for (entries, 0..) |e, idx| {
            if (e.emits()) {
                if (e.seq.first_token_ns == 0) {
                    e.seq.first_token_ns = now;
                }
                try buf.append(self.allocator, out_token_ids[idx]);
                try self.emitted.append(self.allocator, .{
                    .seq = e.seq,
                    .token_id = out_token_ids[idx],
                });
            }
        }

        const n = try self.batch_scheduler.postprocess(entries, buf.items);
        for (entries) |e| {
            if (e.seq.status == .finished and e.seq.finish_ns == 0) e.seq.finish_ns = now;
        }

        self.stats.onStep(entries, sched_ns, exec_ns, self.block_manager);
        return n;
    }

    pub fn isFinished(self: *Self) bool {
        return self.batch_scheduler.isFinished();
    }

    pub fn lastEntries(self: *const Self) []const Entry {
        return self.last;
    }

    pub fn lastEmitted(self: *const Self) []const Emitted {
        return self.emitted.items;
    }

    pub fn releaseFinishedSeqs(self: *Self) void {
        var i: usize = 0;
        while (i < self.all_seqs.items.len) {
            const seq = self.all_seqs.items[i];
            if (seq.status == .finished) {
                seq.deinit();
                self.allocator.destroy(seq);
                _ = self.all_seqs.swapRemove(i);
            } else i += 1;
        }

        self.last = &.{};
        self.emitted.clearRetainingCapacity();
    }
};
