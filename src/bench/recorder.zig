//! 压测统计(对照 lab module-serving-unfused/metrics.zig):分位数用线性插值, 同 numpy.percentile 的默认值

const std = @import("std");

pub const Stats = struct {
    mean: f64 = 0,
    p50: f64 = 0,
    p90: f64 = 0,
    p99: f64 = 0,
    max: f64 = 0,
    n: usize = 0,
};

/// sorted 必须升序;q ∈ [0, 1]
fn percentile(sorted: []const f64, q: f64) f64 {
    if (sorted.len == 0) return 0;
    if (sorted.len == 1) return sorted[0];
    const pos = q * @as(f64, @floatFromInt(sorted.len - 1));
    const lo: usize = @intFromFloat(@floor(pos));
    const hi = @min(lo + 1, sorted.len - 1);
    const frac = pos - @as(f64, @floatFromInt(lo));
    return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

/// 会就地排序 xs
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

/// 峰值常驻内存(MB):Linux 读 /proc/self/status 的 VmHWM;没有 /proc(macOS)退回 getrusage
pub fn peakRssMb() f64 {
    var b: [4096]u8 = undefined;
    const s = std.fs.cwd().readFile("/proc/self/status", &b) catch {
        const ru = std.posix.getrusage(std.posix.rusage.SELF);
        const div: f64 = if (@import("builtin").os.tag.isDarwin()) 1024.0 * 1024.0 else 1024.0; // macOS 单位是字节
        return @as(f64, @floatFromInt(ru.maxrss)) / div;
    };
    var it = std.mem.splitScalar(u8, s, '\n');
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, "VmHWM:")) {
            var toks = std.mem.tokenizeAny(u8, line[6..], " \tkB");
            const kb = std.fmt.parseInt(u64, toks.next() orelse return 0, 10) catch return 0;
            return @as(f64, @floatFromInt(kb)) / 1024.0;
        }
    }
    return 0;
}
