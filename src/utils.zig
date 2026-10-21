const std = @import("std");

pub fn toUsize(value: anytype) usize {
    return @as(usize, @intCast(value));
}
