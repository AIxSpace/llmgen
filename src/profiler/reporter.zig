const std = @import("std");

pub const Report = struct {
    backend: []const u8,
    prompt_tokens: usize,
    gen_tokens: usize,
    prefill_ms: f64,
    ttft_ms: f64,
    tpot_ms: f64,
    itl_p50_ms: f64,
    itl_p99_ms: f64,
    prefill_tps: f64,
    decode_tps: f64,
    e2e_ms: f64,
};

/// 路径来自 LLMGEN_METRICS_JSON, 没设就什么都不做
pub fn writeJson(report: Report) !void {
    const path = std.posix.getenv("LLMGEN_METRICS_JSON") orelse return;
    var file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    var buf: [4096]u8 = undefined;
    var w = file.writer(&buf);
    try w.interface.print("{f}\n", .{std.json.fmt(report, .{ .whitespace = .indent_2 })});
    try w.interface.flush();
}
