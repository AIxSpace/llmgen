const std = @import("std");
const qwen3 = @import("qwen3.zig");
const gguf = @import("../gguf.zig");
const ComputeGraph = @import("../graph/graph.zig").ComputeGraph;

pub const ModelBase = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const Self = @This();

    pub const VTable = struct {
        printInfo: *const fn (ptr: *anyopaque) void,
        chatTemplate: *const fn (ptr: *anyopaque, allocator: std.mem.Allocator, prompt: []const u8, think: bool) anyerror![]u8,
        vocabSize: *const fn (ptr: *anyopaque) usize,
        maxSeqLen: *const fn (ptr: *anyopaque) i64,
        buildGraph: *const fn (ptr: *anyopaque, gguf_info: *const gguf.GGUFInfo) anyerror!*ComputeGraph,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn printInfo(self: Self) void {
        return self.vtable.printInfo(self.ptr);
    }

    pub fn chatTemplate(self: Self, allocator: std.mem.Allocator, prompt: []const u8, think: bool) ![]u8 {
        return self.vtable.chatTemplate(self.ptr, allocator, prompt, think);
    }

    pub fn vocabSize(self: Self) usize {
        return self.vtable.vocabSize(self.ptr);
    }

    pub fn maxSeqLen(self: Self) i64 {
        return self.vtable.maxSeqLen(self.ptr);
    }

    pub fn buildGraph(self: Self, gguf_info: *const gguf.GGUFInfo) !*ComputeGraph {
        return self.vtable.buildGraph(self.ptr, gguf_info);
    }

    pub fn deinit(self: Self) void {
        self.vtable.deinit(self.ptr);
    }
};

pub const ModelFactory = struct {
    const ModelCreator = *const fn (allocator: std.mem.Allocator) anyerror!ModelBase;
    var models_registry: std.StringHashMapUnmanaged(ModelCreator) = .empty;

    pub fn registerModel(allocator: std.mem.Allocator, arch_name: []const u8, creator: ModelCreator) !void {
        try models_registry.put(allocator, arch_name, creator);
    }

    pub fn init(allocator: std.mem.Allocator) !void {
        try registerModel(allocator, "qwen3", qwen3.Model.create);
    }

    pub fn deinit(allocator: std.mem.Allocator) void {
        models_registry.deinit(allocator);
        models_registry = .empty;
    }

    pub fn createFromArch(allocator: std.mem.Allocator, arch_name: []const u8) !ModelBase {
        const creator = models_registry.get(arch_name) orelse return error.UnknownArch;
        return creator(allocator);
    }

    pub fn createFromGGUF(allocator: std.mem.Allocator, gguf_info: *const gguf.GGUFInfo) !ModelBase {
        const arch_name = gguf_info.getModelArchitecture();
        return createFromArch(allocator, arch_name);
    }
};
