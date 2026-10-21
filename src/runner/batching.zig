const std = @import("std");

const block = @import("block.zig");
const BlockManager = block.BlockManager;

pub const SamplingParams = struct {
    temperature: f32 = 1.0, // <=0 -> greedy
    max_tokens: usize = 64,
    ignore_eos: bool = false,
};

pub const Status = enum {
    waiting,
    running,
    finished,
};

pub const Sequence = struct {
    allocator: std.mem.Allocator,

    id: u64,
    status: Status = .waiting,

    all_token_ids: std.ArrayList(i32) = .empty, // 包含 prompt 和 decode 结果 token_ids

    num_prompt_tokens: usize, // prompt tokens num
    num_computed_tokens: usize = 0, // 已经计算过的 tokens 的数量

    block_table: std.ArrayList(i32) = .empty,

    params: SamplingParams, // 采样参数

    arrival_ns: u64 = 0, // addRequest - 添加到 request 的时间
    scheduled_ns: u64 = 0, // 第一次被 schedule 选中
    first_token_ns: u64 = 0, // 第一次 emits() 产出 token
    finish_ns: u64 = 0, // status 变成 finished

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, id: u64, prompt: []const i32, params: SamplingParams) !Self {
        var instance: Sequence = .{
            .allocator = allocator,
            .id = id,
            .num_prompt_tokens = prompt.len,
            .params = params,
        };
        try instance.all_token_ids.appendSlice(allocator, prompt);
        return instance;
    }

    pub fn deinit(self: *Self) void {
        self.all_token_ids.deinit(self.allocator);
        self.block_table.deinit(self.allocator);
    }

    pub fn len(self: *const Self) usize {
        return self.all_token_ids.items.len;
    }

    pub fn lastToken(self: *const Self) i32 {
        return self.all_token_ids.items[self.all_token_ids.items.len - 1];
    }

    pub fn numCompletionTokens(self: *const Self) usize {
        return self.len() - self.num_prompt_tokens;
    }

    // 容纳 n 个 token 需要的块数
    pub fn blocksFor(tokens_num: usize, block_size: usize) usize {
        return (tokens_num + block_size - 1) / block_size;
    }

    // pos token 在 kv cache 中的全局 slot
    pub fn slotOf(self: *const Self, pos: usize, block_size: usize) i32 {
        const block_idx = self.block_table.items[pos / block_size]; // block inex
        return block_idx * @as(i32, @intCast(block_size)) + @as(i32, @intCast(pos % block_size));
    }
};

pub const Policy = enum {
    prefill_first,
    chunked,
};

pub const BatchConfig = struct {
    max_scheduled_len: usize = 256, // scheduled max entries
    max_batched_tokens_len: usize = 16384, // max batched tokens len
    policy: Policy = .prefill_first,
    chunk_tokens: usize = 2048,
    eos: i32 = -1,
};

pub const Entry = struct {
    seq: *Sequence,
    q_start: usize,
    q_len: usize, // 这条序列目前已知的 token 总数

    // 这个 entry 算到了序列当前已有的最后一个 token, 所以本步会产出一个新 token
    // decode 永远为真，prefill 最后一块也为真, prefill 的中间块则为 false
    pub fn emits(self: Entry) bool {
        return (self.q_start + self.q_len) == self.seq.len();
    }
};

pub const BatchScheduler = struct {
    allocator: std.mem.Allocator,

    batch_config: BatchConfig,
    num_preemptions: usize = 0,

    waiting: std.ArrayList(*Sequence) = .empty, // FIFO
    running: std.ArrayList(*Sequence) = .empty,
    scheduled: std.ArrayList(Entry) = .empty, // 不断放入 scheduled 然后进行调度处理

    block_manager: *BlockManager,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, block_manager: *BlockManager, cfg: BatchConfig) Self {
        return .{
            .allocator = allocator,
            .batch_config = cfg,
            .block_manager = block_manager,
        };
    }

    pub fn deinit(self: *Self) void {
        self.waiting.deinit(self.allocator);
        self.running.deinit(self.allocator);
        self.scheduled.deinit(self.allocator);
    }

    // 批次都已经处理完毕
    pub fn isFinished(self: *const Self) bool {
        return self.waiting.items.len == 0 and self.running.items.len == 0;
    }

    // 添加 sequence
    pub fn addSeq(self: *Self, s: *Sequence) !void {
        s.status = .waiting;
        try self.waiting.append(self.allocator, s);
    }

    // 开始调度，把调度好的 Entry 返回
    pub fn schedule(self: *Self) ![]const Entry {
        self.scheduled.clearRetainingCapacity(); // 每次清空可调度元素

        switch (self.batch_config.policy) {
            .prefill_first => try self.schedulePrefillFirst(), // prefill first
            .chunked => try self.scheduleChunked(), // chunked prefill
        }

        return self.scheduled.items;
    }

    // 优先处理 prefill，有空闲资源才会去处理 decode，会出现 decode 饥饿的现象
    fn schedulePrefillFirst(self: *Self) !void {
        // prefill
        var prefill_tokens_len: usize = 0;

        // 有等待的，还可以被调度
        while (self.waiting.items.len > 0 and self.scheduled.items.len < self.batch_config.max_scheduled_len) {
            const cur_seq = self.waiting.items[0];
            const cur_seq_tokens_len = cur_seq.len();

            if (prefill_tokens_len + cur_seq_tokens_len > self.batch_config.max_batched_tokens_len) { // 处理的 token 数量大于最大的批次的 token 数量
                break;
            }

            if (!try self.block_manager.ensure(cur_seq, cur_seq_tokens_len)) { // 能否分配新的 block
                break;
            }

            _ = self.waiting.orderedRemove(0); // 从 waiting 取一个放到 running 里
            cur_seq.status = .running;
            try self.running.append(self.allocator, cur_seq);

            try self.scheduled.append(
                self.allocator,
                .{ .seq = cur_seq, .q_start = 0, .q_len = cur_seq_tokens_len },
            );

            prefill_tokens_len += cur_seq_tokens_len;
        }

        // prefill_first: 优先处理 prefill，prefill 没有了再处理 decode
        if (self.scheduled.items.len > 0) {
            return;
        }

        // decode
        var i: usize = 0;
        while (i < self.running.items.len and self.scheduled.items.len < self.batch_config.max_scheduled_len) {
            const cur_running_seq = self.running.items[i];
            var is_decode_tail = false;

            // 先来的先保住，后来的先牺牲
            while (!self.block_manager.canEnsure(cur_running_seq, cur_running_seq.len())) { // block 块不够用
                if (self.running.items.len - 1 > i) { // i 后面还有别的序列
                    try self.preempt(self.running.pop().?); // 退回到 waiting，释放它的块
                } else {
                    // 如果 i 自己是队尾
                    _ = self.running.orderedRemove(i);
                    try self.preempt(cur_running_seq); // 把自己放到 waiting 队列
                    is_decode_tail = true; // 本轮 decode 调度结束
                    break;
                }
            }

            if (is_decode_tail) {
                break;
            }

            _ = try self.block_manager.ensure(cur_running_seq, cur_running_seq.len());

            try self.scheduled.append(self.allocator, .{
                .seq = cur_running_seq,
                .q_start = cur_running_seq.len() - 1,
                .q_len = 1,
            });

            i += 1;
        }
    }

    // 切成块，prefill 和 decode 混在一起，雨露均沾
    fn scheduleChunked(self: *Self) !void {
        var max_budget = self.batch_config.max_batched_tokens_len; // 这轮批次最多算多少 token
        var prefill_budget = self.batch_config.chunk_tokens; // chunk token 上限，生效的留给 decode
        const preemptions_before = self.num_preemptions; // 记录本轮开始时的抢占次数

        // running - decode 未完成或者 prefill 未完成的继续
        var i: usize = 0;
        while (i < self.running.items.len and self.scheduled.items.len < self.batch_config.max_scheduled_len and max_budget > 0) {
            // 处理 running 的 sequence - 先处理 running 的 seq
            const cur_seq = self.running.items[i];
            const remaining = cur_seq.len() - cur_seq.num_computed_tokens;
            const is_doing_prefill = (remaining > 1); // prefill 还没完 - decode remaining 始终为 1

            var q_len = @min(remaining, max_budget);
            if (is_doing_prefill) {
                q_len = @min(q_len, prefill_budget); // 限制 prefill 的 token 数
            }
            if (q_len == 0) { // budget 超限，下一轮
                i += 1;
                continue;
            }

            const upto_tokens_len = cur_seq.num_computed_tokens + q_len; // 将要处理的 tokens 的数量

            var is_decode_tail = false;
            while (!self.block_manager.canEnsure(cur_seq, upto_tokens_len)) {
                if (self.running.items.len - 1 > i) {
                    try self.preempt(self.running.pop().?); // running 后面还有，优先牺牲后来的，保住先来的
                } else {
                    _ = self.running.orderedRemove(i); // 自己就是队尾, 牺牲自己
                    try self.preempt(cur_seq);
                    is_decode_tail = true; // 本轮调度结束了
                    break;
                }
            }

            if (is_decode_tail) {
                break;
            }

            _ = try self.block_manager.ensure(cur_seq, upto_tokens_len);
            try self.scheduled.append(self.allocator, .{
                .seq = cur_seq,
                .q_start = cur_seq.num_computed_tokens,
                .q_len = q_len,
            });

            max_budget -= q_len;
            if (is_doing_prefill) {
                prefill_budget -= q_len;
            }

            i += 1;
        }

        if (self.num_preemptions != preemptions_before) {
            return; // 发生了抢占，避免再次从 waiting 中恢复出来，所以跳过
        }

        // 从 waiting 接新客
        while (self.waiting.items.len > 0 and self.scheduled.items.len < self.batch_config.max_scheduled_len and max_budget > 0 and prefill_budget > 0) {
            const seq = self.waiting.items[0];
            const q_len = @min(
                seq.len(),
                @min(max_budget, prefill_budget),
            );

            if (!try self.block_manager.ensure(seq, q_len)) {
                break;
            }

            // 还有预算，放入 running
            _ = self.waiting.orderedRemove(0);
            seq.status = .running;
            seq.num_computed_tokens = 0;

            try self.running.append(self.allocator, seq);
            try self.scheduled.append(self.allocator, .{
                .seq = seq,
                .q_start = 0,
                .q_len = q_len,
            });

            max_budget -= q_len;
            prefill_budget -= q_len;
        }
    }

    fn preempt(self: *Self, seq: *Sequence) !void {
        seq.status = .waiting;
        self.block_manager.deallocate(seq);
        try self.waiting.insert(self.allocator, 0, seq);
        self.num_preemptions += 1;
    }

    // 回填结果 token_ids
    pub fn postprocess(self: *Self, entries: []const Entry, tokens: []const i32) !usize {
        var k: usize = 0;
        var n_finished: usize = 0;

        for (entries) |e| {
            const seq = e.seq;
            seq.num_computed_tokens = e.q_start + e.q_len;

            const emits = e.emits();
            if (!emits) {
                continue;
            }

            const token_id = tokens[k];
            k += 1;
            try seq.all_token_ids.append(self.allocator, token_id);

            const hit_eos = !seq.params.ignore_eos and token_id == self.batch_config.eos; // 碰到结束符
            if (hit_eos or seq.numCompletionTokens() >= seq.params.max_tokens) {
                seq.status = .finished;
                self.block_manager.deallocate(seq);
                for (self.running.items, 0..) |running_seq, idx| {
                    if (running_seq == seq) {
                        _ = self.running.orderedRemove(idx);
                        break;
                    }
                }
                n_finished += 1;
            }
        }

        return n_finished;
    }
};
