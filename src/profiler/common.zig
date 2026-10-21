const std = @import("std");

// ns -> ms
pub fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

// 求分位数
pub fn percentile(sorted: []const u64, p: f64) u64 {
    if (sorted.len == 0) return 0;
    const rank = p / 100.0 * @as(f64, @floatFromInt(sorted.len - 1));
    return sorted[@intFromFloat(@round(rank))];
}

pub fn mean(xs: []const u64) u64 {
    var sum: u64 = 0;
    for (xs) |x| sum += x;
    return sum / xs.len;
}

pub fn perSec(count: usize, ns: u64) f64 {
    if (ns == 0) return 0;
    return @as(f64, @floatFromInt(count)) / (@as(f64, @floatFromInt(ns)) / 1e9);
}
