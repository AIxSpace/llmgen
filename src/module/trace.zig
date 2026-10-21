const std = @import("std");

const gguf = @import("../gguf.zig");
const GGUFInfo = gguf.GGUFInfo;
const TensorInfo = gguf.TensorInfo;

const graph = @import("../graph/graph.zig");
const ComputeGraph = graph.ComputeGraph;

pub const Trace = struct {
    allocator: std.mem.Allocator,
    node_allocator: std.mem.Allocator,
    gguf_info: *const GGUFInfo,
    names: std.ArrayList([]u8) = .empty,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        node_allocator: std.mem.Allocator,
        gguf_info: *const GGUFInfo,
    ) Self {
        return .{
            .allocator = allocator,
            .node_allocator = node_allocator,
            .gguf_info = gguf_info,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.names.items) |n| {
            self.allocator.free(n);
        }
        self.names.deinit(self.allocator);
    }

    pub fn name(self: *Self, tag: []const u8, layer: i32) ![]const u8 {
        const str = if (layer >= 0)
            try std.fmt.allocPrint(self.allocator, "l{d}.{s}", .{ layer, tag })
        else
            try self.allocator.dupe(u8, tag);
        errdefer self.allocator.free(str);
        try self.names.append(self.allocator, str);
        return str;
    }

    pub fn weight(self: *Self, name_or_suffix: []const u8, layer: i32) !*const TensorInfo {
        if (layer < 0) return self.gguf_info.findTensor(name_or_suffix) orelse error.MissingWeight;

        const combined_weight_name = try std.fmt.allocPrint(self.allocator, "blk.{d}.{s}", .{
            layer,
            name_or_suffix,
        });
        errdefer self.allocator.free(combined_weight_name);

        try self.names.append(self.allocator, combined_weight_name);

        return self.gguf_info.findTensor(combined_weight_name) orelse error.MissingWeight;
    }
};
