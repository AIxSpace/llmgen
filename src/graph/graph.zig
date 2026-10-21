const std = @import("std");

const types = @import("../types.zig");

const Tensor = @import("../Tensor.zig");

pub const ComputeGraph = struct {
    tensors: std.ArrayList(*Tensor) = .empty,
    outputs: std.ArrayList(*Tensor) = .empty,

    exec_order: std.ArrayList(*Tensor) = .empty,
    levels: std.ArrayList([]*Tensor) = .empty,

    layer_groups: std.AutoHashMapUnmanaged(i32, std.ArrayList(*Tensor)) = .empty,
    max_layer: i32 = -1,

    allocator: std.mem.Allocator,
    node_arena: std.heap.ArenaAllocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .node_arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.clearOrder();

        self.clearLayerGroups();
        self.layer_groups.deinit(self.allocator);

        self.exec_order.deinit(self.allocator);
        self.levels.deinit(self.allocator);

        self.outputs.deinit(self.allocator);

        self.tensors.deinit(self.allocator);
        self.node_arena.deinit();
    }

    pub fn nodeAllocator(self: *Self) std.mem.Allocator {
        return self.node_arena.allocator();
    }

    pub fn addTensor(self: *Self, tensor: *Tensor) !void {
        try self.tensors.append(self.allocator, tensor);
    }

    pub fn indexOf(self: *const Self, target: *Tensor) ?usize {
        for (self.tensors.items, 0..) |t, i| {
            if (t == target) return i;
        }
        return null;
    }

    pub fn getAllTensors(self: *const Self) []*Tensor {
        return self.tensors.items;
    }

    pub fn getExecutionOrder(self: *const Self) []*Tensor {
        return self.exec_order.items;
    }

    pub fn getExecutionLevels(self: *const Self) []*Tensor {
        return self.tensors.items;
    }

    pub fn getExecutionOutputs(self: *const Self) []*Tensor {
        return self.outputs.items;
    }

    pub fn getLayerGroups(self: *Self) *std.AutoHashMapUnmanaged(i32, std.ArrayList(*Tensor)) {
        return &self.layer_groups;
    }

    pub fn getMaxLayer(self: *const Self) i32 {
        return self.max_layer;
    }

    pub fn buildFromOutputs(self: *Self, outs: []const *Tensor) !void {
        self.clear();

        for (outs) |out| {
            out.tensor_type = .TENSOR_TYPE_OUTPUT;
            try self.outputs.append(self.allocator, out);
        }

        try self.reverseBfsCollect(self.outputs.items);
        try self.topologicSort();
    }

    pub fn rebuildOrder(self: *Self) !void {
        self.clearOrder();
        try self.topologicSort();
    }

    // reverse bfs collect 从 outputs 沿 src 反向遍历，收集可触达的所有节点
    fn reverseBfsCollect(self: *Self, seeds: []const *Tensor) !void {
        var visited: std.AutoHashMapUnmanaged(*Tensor, void) = .empty;
        defer visited.deinit(self.allocator);

        var queue: std.ArrayList(*Tensor) = .empty;
        defer queue.deinit(self.allocator);

        for (seeds) |t| {
            if (!visited.contains(t)) {
                try visited.put(self.allocator, t, {});
                try queue.append(self.allocator, t);
            }
        }

        var head: usize = 0;
        while (head < queue.items.len) : (head += 1) {
            const cur = queue.items[head];
            try self.tensors.append(self.allocator, cur);
            for (cur.src) |maybe_src| {
                const src = maybe_src orelse continue;
                if (!visited.contains(src)) {
                    try visited.put(self.allocator, src, {});
                    try queue.append(self.allocator, src);
                }
            }
        }
    }

    // Kahn 分层
    fn topologicSort(self: *Self) !void {
        var in_deg: std.AutoHashMapUnmanaged(*Tensor, u32) = .empty;
        defer in_deg.deinit(self.allocator);

        var children: std.AutoHashMapUnmanaged(*Tensor, std.ArrayList(*Tensor)) = .empty;
        defer {
            var it = children.valueIterator();
            while (it.next()) |list| {
                list.deinit(self.allocator);
            }
            children.deinit(self.allocator);
        }

        // 找出 parent -> children 关系
        // 确认 degree
        for (self.tensors.items) |t| {
            var deg: u32 = 0;
            for (t.src) |maybe_src| {
                const src = maybe_src orelse continue;
                deg += 1;
                const entry = try children.getOrPutValue(self.allocator, src, .empty);
                try entry.value_ptr.append(self.allocator, t);
            }
            try in_deg.put(self.allocator, t, deg);
        }

        // 收集当前所有 in_degree==0 的节点作为同一 level - 一开始的 level
        var ready: std.ArrayList(*Tensor) = .empty;
        defer ready.deinit(self.allocator);
        for (self.tensors.items) |t| {
            if (in_deg.get(t).? == 0) {
                try ready.append(self.allocator, t);
            }
        }

        self.max_layer = -1;
        var processed: usize = 0;
        while (ready.items.len > 0) {
            std.sort.insertion(*Tensor, ready.items, {}, lessByLayer);
            var cur_level: std.ArrayList(*Tensor) = .empty;
            errdefer cur_level.deinit(self.allocator);

            var next: std.ArrayList(*Tensor) = .empty;
            defer next.deinit(self.allocator);

            for (ready.items) |t| {
                processed += 1;
                self.max_layer = @max(self.max_layer, t.layer_id);
                if (t.isComputed()) {
                    // std.debug.print("\ncompute tensor name = {s}\n", .{t.name});
                    try cur_level.append(self.allocator, t);
                    try self.exec_order.append(self.allocator, t); // -> 找出 exec_order
                }
                const consumers = children.getPtr(t) orelse continue;
                for (consumers.items) |consumer| {
                    const deg = in_deg.getPtr(consumer).?;
                    deg.* -= 1;
                    if (deg.* == 0) {
                        try next.append(self.allocator, consumer); // 明确下一步计算的 tensors
                    }
                }
            }

            if (cur_level.items.len > 0) {
                try self.levels.append(self.allocator, try cur_level.toOwnedSlice(self.allocator)); // -> 找出 levels
            } else {
                cur_level.deinit(self.allocator);
            }

            ready.clearRetainingCapacity();
            try ready.appendSlice(self.allocator, next.items); // 下一层级
        }
        if (processed != self.tensors.items.len) {
            return error.GraphCycle;
        }

        for (self.tensors.items) |t| {
            const entry = try self.layer_groups.getOrPutValue(self.allocator, t.layer_id, .empty);
            try entry.value_ptr.append(self.allocator, t);
        }
    }

    pub fn insertMemcpy(self: *Self, origin_tensor: *Tensor, device_type: types.DeviceType) !*Tensor {
        const node_allocator = self.nodeAllocator();
        const proxy_tensor = try node_allocator.create(Tensor);
        proxy_tensor.* = .{
            .name = try std.fmt.allocPrint(node_allocator, "copy_{s}", .{origin_tensor.name}),
            .data_type = origin_tensor.data_type,
            .tensor_type = origin_tensor.tensor_type,
            .layer_id = origin_tensor.layer_id,
            .op_type = .OP_TYPE_MEMCPY,
            .dims = origin_tensor.dims,
            .strides = origin_tensor.strides,
            .device_type = device_type,
        };
        proxy_tensor.src[0] = origin_tensor;
        try self.addTensor(proxy_tensor);
        return proxy_tensor;
    }

    /// 对照 replace_output:把外部输出 old 换成 new(采样从 new 读)。
    pub fn replaceOutput(self: *Self, old: *Tensor, new: *Tensor) void {
        for (self.outputs.items) |*out| {
            if (out.* == old) out.* = new;
        }
    }

    pub fn clear(self: *Self) void {
        self.clearOrder();
        self.tensors.clearRetainingCapacity();
        self.outputs.clearRetainingCapacity();
        self.max_layer = -1;
    }

    fn clearOrder(self: *Self) void {
        for (self.levels.items) |level| {
            self.allocator.free(level);
        }
        self.levels.clearRetainingCapacity();
        self.exec_order.clearRetainingCapacity();
    }

    fn clearLayerGroups(self: *Self) void {
        var it = self.layer_groups.valueIterator();
        while (it.next()) |list| {
            list.deinit(self.allocator);
        }
        self.layer_groups.clearRetainingCapacity();
    }

    fn lessByLayer(_: void, a: *Tensor, b: *Tensor) bool {
        return a.layer_id < b.layer_id;
    }

    pub fn exportDot(self: *const Self, path: []const u8) !void {
        const dot = try self.renderDot(self.allocator);
        defer self.allocator.free(dot);
        try std.fs.cwd().writeFile(.{ .sub_path = path, .data = dot });
    }

    pub fn renderDot(self: *const Self, allocator: std.mem.Allocator) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);

        try out.appendSlice(allocator,
            \\digraph ComputeGraph {
            \\  rankdir=TB;
            \\  node [shape=box, style=filled];
            \\
            \\
        );

        // 画节点
        for (self.tensors.items, 0..) |t, i| {
            const color: []const u8 = switch (t.tensor_type) {
                .TENSOR_TYPE_WEIGHT => "#FFD54F",
                .TENSOR_TYPE_INPUT => "#90CAF9",
                .TENSOR_TYPE_OUTPUT => "#FF0000",
                .TENSOR_TYPE_CACHE, .TENSOR_TYPE_KV_CACHE => "#4A91D8",
                else => "#C8E6C9",
            };

            var real_dims: [types.TENSOR_MAX_DIMS]i64 = undefined;
            var ndims: usize = 0;
            for (t.dims) |d| {
                if (d != 0) {
                    real_dims[ndims] = d;
                    ndims += 1;
                }
            }

            const line = if (t.layer_id >= 0)
                try std.fmt.allocPrint(allocator, "  n{d} [fillcolor=\"{s}\", label=\"[L{d},{s}] {s}\\n{s}{any}", .{ i, color, t.layer_id, @tagName(t.data_type), t.name, @tagName(t.device_type), real_dims[0..ndims] })
            else
                try std.fmt.allocPrint(allocator, "  n{d} [fillcolor=\"{s}\", label=\"[Global,{s}] {s}\\n{s}{any}", .{ i, color, @tagName(t.data_type), t.name, @tagName(t.device_type), real_dims[0..ndims] });
            defer allocator.free(line);
            try out.appendSlice(allocator, line);

            if (t.op_type != .OP_TYPE_NONE) {
                const op_line = try std.fmt.allocPrint(allocator, "\\n{s}", .{@tagName(t.op_type)});
                defer allocator.free(op_line);
                try out.appendSlice(allocator, op_line);
            }

            try out.appendSlice(allocator, "\"];\n");
        }

        try out.appendSlice(allocator, "\n");

        // 画边
        for (self.tensors.items, 0..) |t, i| {
            for (t.src) |maybe_src| {
                const src = maybe_src orelse continue;
                const src_id = self.indexOf(src) orelse return error.ForeignTensor;
                const line = if (t.tensor_type == .TENSOR_TYPE_VIEW)
                    try std.fmt.allocPrint(allocator, "  n{d} -> n{d} [style=dashed];\n", .{ src_id, i })
                else
                    try std.fmt.allocPrint(allocator, "  n{d} -> n{d};\n", .{ src_id, i });
                defer allocator.free(line);
                try out.appendSlice(allocator, line);
            }
        }

        try out.appendSlice(allocator, "}\n");

        return out.toOwnedSlice(allocator);
    }
};
