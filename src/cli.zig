//! 命令行(对照 lab module-serving-unfused/main.zig 的 Options / parseArgs)
//!
//!   llmgen chat   [prompt] [max_new] [选项]    greedy(默认)生成并流式打印, 最后一行打印 token ids(对拍用)
//!   llmgen bench <workload.json>    [选项]    按 workload 压测, 打印同 lab / nano-vllm 口径的 JSON
//!
//! 选项:
//!   --model <path>              GGUF 路径(默认取环境变量 LLMGEN_MODEL)
//!   --temp <f>                  chat 的采样温度(默认 0 = greedy)
//!   --max-model-len <n>         单条序列最长 token 数(默认 4096)
//!   --max-num-seqs <n>          一步最多几条序列(默认 256)
//!   --max-batched-tokens <n>    调度器一步最多排多少 token(默认 16384)
//!   --pass-tokens <n>           单趟前向的 token 上限(默认 CUDA 8192 / CPU 2048)
//!   --block-size <n>            KV 分页块大小(默认 16)
//!   --gpu-util <f>              进程最多用总显存的比例(默认 0.9)
//!   --kv-blocks <n>             固定 KV 块数(默认按显存算;CPU 512)
//!   --policy prefill_first|chunked   调度策略(默认 prefill_first)
//!   --chunk <n>                 chunked 每步新 prefill 的 token 上限(默认 2048)
//!   --out <path>                bench 的 JSON 另存一份
//!   --dup <n>                   chat:同一个 prompt 提交 n 份一起跑, 检查输出完全相同
//!   --eager                     不录 / 不回放 CUDA Graph
//!   --sync-each                 每个 kernel 后同步(定位出错的 kernel;隐含 --eager)
//!   --single                    chat 走单请求的旧路径 Executor.generate(LLMGEN_METRICS / LLMGEN_PROFILE 的打点在这条路上)
//!   --no-kv-cache               配合 --single:每步整段重算(调试用)
//!   -i, --interactive           chat:交互式多轮对话(/think 切换思考模式, /clear 清空历史, /exit 退出)
//!   --system <text>             配合 -i:system prompt

const std = @import("std");
const opts = @import("build_options");
const Policy = @import("runner/batching.zig").Policy;

pub const Command = enum { chat, bench };

pub const Options = struct {
    cmd: Command = .chat,
    arg0: ?[]const u8 = null, // chat:prompt;bench:workload 路径
    max_new: usize = 2000, //   chat 最多生成多少 token(再按 max_model_len 截断)
    model: ?[]const u8 = null,
    out: ?[]const u8 = null,
    temp: f32 = 0,
    eager: bool = false,
    sync_each: bool = false,
    single: bool = false, // 单请求老模式
    no_kv_cache: bool = false,
    interactive: bool = false,
    system: ?[]const u8 = null, //   -i 的 system prompt
    dup: usize = 1,
    max_num_seqs: usize = 256,
    max_batched: usize = 16384,
    max_model_len: usize = 4096,
    block_size: usize = 16,
    gpu_util: f32 = 0.9,
    kv_blocks: usize = 0, //         0 = CUDA 按 gpu_util 自动算、CPU 用 HOST_KV_BLOCKS(调度器里)
    policy: Policy = .prefill_first,
    chunk: usize = 2048,
    pass_tokens: ?usize = null, //   null = CUDA 8192 / CPU 2048

    /// 单趟前向的 token 上限:至少要能放下一步 decode(max_num_seqs 个 token)
    pub fn maxTokens(self: Options) usize {
        const pass = self.pass_tokens orelse (if (opts.have_cuda) @as(usize, 8192) else 2048);
        return @max(pass, self.max_num_seqs);
    }
};

/// 位置参数:cmd、arg0、max_new;"--xxx" 是选项, 不带值的开关见 flags
pub fn parseArgs(allocator: std.mem.Allocator) !Options {
    var it = try std.process.argsWithAllocator(allocator);
    defer it.deinit();
    _ = it.next(); // 程序名

    var o = Options{};
    var pos: usize = 0;
    while (it.next()) |a| {
        if (std.mem.eql(u8, a, "-i")) {
            o.interactive = true;
        } else if (std.mem.startsWith(u8, a, "--")) {
            const key = a[2..];
            // 不带值的开关
            if (std.mem.eql(u8, key, "eager")) {
                o.eager = true;
            } else if (std.mem.eql(u8, key, "sync-each")) {
                o.sync_each = true;
            } else if (std.mem.eql(u8, key, "single")) {
                o.single = true;
            } else if (std.mem.eql(u8, key, "no-kv-cache")) {
                o.no_kv_cache = true;
            } else if (std.mem.eql(u8, key, "interactive")) {
                o.interactive = true;
            } else {
                // 带值的选项:值要复制出来, 迭代器 deinit 后原字符串就失效了
                const v = try allocator.dupe(u8, it.next() orelse return error.MissingOptionValue);
                if (std.mem.eql(u8, key, "model")) o.model = v //
                else if (std.mem.eql(u8, key, "out")) o.out = v //
                else if (std.mem.eql(u8, key, "system")) o.system = v //
                else if (std.mem.eql(u8, key, "temp")) o.temp = try std.fmt.parseFloat(f32, v) //
                else if (std.mem.eql(u8, key, "dup")) o.dup = try std.fmt.parseInt(usize, v, 10) //
                else if (std.mem.eql(u8, key, "max-num-seqs")) o.max_num_seqs = try std.fmt.parseInt(usize, v, 10) //
                else if (std.mem.eql(u8, key, "max-batched-tokens")) o.max_batched = try std.fmt.parseInt(usize, v, 10) //
                else if (std.mem.eql(u8, key, "max-model-len")) o.max_model_len = try std.fmt.parseInt(usize, v, 10) //
                else if (std.mem.eql(u8, key, "block-size")) o.block_size = try std.fmt.parseInt(usize, v, 10) //
                else if (std.mem.eql(u8, key, "gpu-util")) o.gpu_util = try std.fmt.parseFloat(f32, v) //
                else if (std.mem.eql(u8, key, "kv-blocks")) o.kv_blocks = try std.fmt.parseInt(usize, v, 10) //
                else if (std.mem.eql(u8, key, "policy")) o.policy = std.meta.stringToEnum(Policy, v) orelse return error.UnknownPolicy //
                else if (std.mem.eql(u8, key, "chunk")) o.chunk = try std.fmt.parseInt(usize, v, 10) //
                else if (std.mem.eql(u8, key, "pass-tokens")) o.pass_tokens = try std.fmt.parseInt(usize, v, 10) //
                else {
                    std.debug.print("未知选项 --{s}\n", .{key});
                    return error.UnknownOption;
                }
            }
        } else {
            switch (pos) {
                0 => o.cmd = std.meta.stringToEnum(Command, a) orelse {
                    std.debug.print("未知子命令 {s}(可选 chat / bench)\n", .{a});
                    return error.UnknownCommand;
                },
                1 => o.arg0 = try allocator.dupe(u8, a),
                2 => o.max_new = try std.fmt.parseInt(usize, a, 10),
                else => return error.TooManyArgs,
            }
            pos += 1;
        }
    }
    if (o.sync_each) o.eager = true; // 逐 kernel 同步和 CUDA Graph 回放不能同时用
    return o;
}

/// parseArgs 里 dupe 出来的字符串
pub fn freeOptions(allocator: std.mem.Allocator, o: Options) void {
    if (o.arg0) |s| allocator.free(s);
    if (o.model) |s| allocator.free(s);
    if (o.out) |s| allocator.free(s);
    if (o.system) |s| allocator.free(s);
}
