//! bench/workload.zig -- 读 bench/gen_workload.py 生成的 JSON(格式同 module-serving-unfused 的 main.zig)
//! {"name": ..., "rounds": [[req, ...], ...]}:各轮依次跑, 轮内请求同时提交。

const std = @import("std");

pub const Request = struct {
    prompt_ids: []const i32,
    max_tokens: usize,
    temperature: f32 = 0.6,
    ignore_eos: bool = true,
};

pub const Workload = struct {
    name: []const u8 = "workload",
    rounds: []const []const Request,
};

pub fn load(allocator: std.mem.Allocator, path: []const u8) !std.json.Parsed(Workload) {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 256 << 20);
    defer allocator.free(bytes);

    // .alloc_always:解析结果不引用 bytes, 上面可以放心 free
    return std.json.parseFromSlice(
        Workload,
        allocator,
        bytes,
        .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_always,
        },
    );
}
