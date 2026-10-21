const std = @import("std");
const opts = @import("build_options");

const Tensor = @import("../Tensor.zig");

const types = @import("../types.zig");
const DeviceType = types.DeviceType;

const gguf = @import("../gguf.zig");
const GGUFParser = gguf.GGUFParser;

const graph = @import("../graph/graph.zig");
const ComputeGraph = graph.ComputeGraph;

const res = @import("memory_resource.zig");
const CpuMemoryResource = res.CpuMemoryResource;
const CudaMemoryResource = res.CudaMemoryResource;

const pools = @import("pools.zig");
const DevicePools = pools.DevicePools;
const MemoryPool = pools.MemoryPool;

const paged_atten = @import("page_atten.zig");
const PagedAttentionManager = paged_atten.PagedAttentionManager;

const cuda = if (opts.have_cuda) @import("../backend/cuda/cuda.zig") else struct {};

/// 权重上传用的 pinned 暂存区:磁盘 → pinned → 显存, 所有权重轮流用这一块
const STAGE_BYTES: usize = 32 << 20;

const DeviceKey = struct {
    device_type: DeviceType,
    device_id: usize,
};

const DeviceSlot = struct {
    cpu_resource: CpuMemoryResource,
    cuda_resource: if (opts.have_cuda) CudaMemoryResource else void,
    pools: DevicePools = .{},
};

pub var global_mem_manager: ?*MemoryManager = null;

pub const MemoryManager = struct {
    allocator: std.mem.Allocator,
    devices_map: std.AutoHashMapUnmanaged(DeviceKey, *DeviceSlot) = .empty,
    layer_device_map: std.AutoHashMapUnmanaged(usize, DeviceKey) = .empty, // layer device ids map
    paged_atten_managers_map: std.AutoHashMapUnmanaged(DeviceKey, *PagedAttentionManager) = .empty,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        if (global_mem_manager == self) {
            global_mem_manager = null;
        }

        var pam_it = self.paged_atten_managers_map.valueIterator();
        while (pam_it.next()) |paged_atten_maanger| {
            paged_atten_maanger.*.deinit();
            self.allocator.destroy(paged_atten_maanger.*);
        }
        self.paged_atten_managers_map.deinit(self.allocator);

        var it = self.devices_map.valueIterator();
        while (it.next()) |device_value| {
            device_value.*.pools.deinit();
            self.allocator.destroy(device_value.*);
        }
        self.devices_map.deinit(self.allocator);

        self.layer_device_map.deinit(self.allocator);
    }

    pub fn get(self: *Self, device_type: DeviceType, device_id: usize) ?*DevicePools {
        const device_key: DeviceKey = .{
            .device_type = device_type,
            .device_id = device_id,
        };
        if (self.devices_map.get(device_key)) |device_slot| {
            return &device_slot.pools;
        }
        return null;
    }

    fn makeResource(slot: *DeviceSlot, device_type: DeviceType) !res.IMemoryResource {
        return switch (device_type) {
            .CPU => slot.cpu_resource.asIMemoryResource(),
            .CUDA => if (opts.have_cuda) slot.cuda_resource.asIMemoryResource() else error.UnSupportedDevice,
            else => error.UnSupportedDevice,
        };
    }

    pub fn getOrCreate(self: *Self, device_type: DeviceType, device_id: usize, weight_cap: usize, activation_cap: usize, kv_cap: usize) !*DevicePools {
        const device_key = DeviceKey{
            .device_type = device_type,
            .device_id = device_id,
        };
        if (self.devices_map.get(device_key)) |device_value| {
            return &device_value.pools;
        }

        const device_slot = try self.allocator.create(DeviceSlot);
        errdefer self.allocator.destroy(device_slot);
        device_slot.* = .{
            .cpu_resource = CpuMemoryResource.init(self.allocator),
            .cuda_resource = if (opts.have_cuda) CudaMemoryResource.init(device_id) else {},
        };

        if (weight_cap > 0) {
            device_slot.pools.weight = try MemoryPool.init(try makeResource(device_slot, device_type), weight_cap, "weight");
        }
        if (activation_cap > 0) {
            device_slot.pools.activation = try MemoryPool.init(try makeResource(device_slot, device_type), activation_cap, "activation");
        }
        if (kv_cap > 0) {
            device_slot.pools.kv_cache = try MemoryPool.init(try makeResource(device_slot, device_type), kv_cap, "kv_cache");
        }

        try self.devices_map.put(self.allocator, device_key, device_slot);

        return &device_slot.pools;
    }

    pub fn registerLayerDevice(self: *Self, layer_id: usize, device_type: DeviceType, device_id: usize) !void {
        const device_key: DeviceKey = .{
            .device_type = device_type,
            .device_id = device_id,
        };
        try self.layer_device_map.put(self.allocator, layer_id, device_key);
    }

    pub fn getLayerDevice(self: *const Self, layer_id: usize) ?DeviceKey {
        return self.layer_device_map.get(layer_id);
    }

    pub fn resetAllActivations(self: *Self) void {
        var it = self.devices_map.valueIterator();
        while (it.next()) |device_value| {
            device_value.*.pools.resetActivation();
        }
    }

    pub fn createAttentionManager(
        self: *Self,
        device_type: DeviceType,
        device_id: usize,
        block_size: usize,
    ) !*PagedAttentionManager {
        const device_key: DeviceKey = .{
            .device_type = device_type,
            .device_id = device_id,
        };

        if (self.paged_atten_managers_map.get(device_key)) |pam| {
            return pam;
        }

        const device_pools = self.get(device_type, device_id) orelse return error.NoDevicePools;
        if (device_pools.kv_cache == null) {
            return error.NoKVCachePool;
        }

        const paged_atten_mananger = try self.allocator.create(PagedAttentionManager);
        errdefer self.allocator.destroy(paged_atten_mananger);

        paged_atten_mananger.* = PagedAttentionManager.init(
            self.allocator,
            &(device_pools.kv_cache.?),
            block_size,
        );

        try self.paged_atten_managers_map.put(
            self.allocator,
            device_key,
            paged_atten_mananger,
        );

        return paged_atten_mananger;
    }

    pub fn getAttentionManager(self: *Self, device_type: DeviceType, device_id: usize) ?*PagedAttentionManager {
        return self.paged_atten_managers_map.get(.{
            .device_type = device_type,
            .device_id = device_id,
        });
    }

    // todo: - understand - 理解这个逻辑

    /// 对照 C++ load_weights:按设备分成 cpu / gpu 两组, 各按 gguf offset 排序(顺序读盘);
    ///   CPU:bump 一块, 拷入磁盘字节;同名权重(tied embedding)在同一设备上共享一块。
    ///   CUDA:按 getLayerDevice(layer_id) 找卡, bump 一块显存, H2D 上传。
    pub fn loadWeights(self: *Self, parser: *GGUFParser, compute_graph: *ComputeGraph) !void {
        const Entry = struct {
            tensor: *Tensor,
            gguf_offset: u64,
            fn lessThan(_: void, a: @This(), b: @This()) bool {
                return a.gguf_offset < b.gguf_offset;
            }
        };
        var cpu_weights: std.ArrayList(Entry) = .empty;
        defer cpu_weights.deinit(self.allocator);
        var gpu_weights: std.ArrayList(Entry) = .empty;
        defer gpu_weights.deinit(self.allocator);

        for (compute_graph.tensors.items) |t| {
            if (t.tensor_type != .TENSOR_TYPE_WEIGHT or t.data != null) continue;
            const info = parser.getInfo().findTensor(t.name) orelse return error.MissingWeight;
            switch (t.data_type) {
                .GGML_TYPE_F32, .GGML_TYPE_F16, .GGML_TYPE_BF16 => {},
                else => return error.UnsupportedWeightDtype, // 量化权重 size() 为 0, 不拦会静默算错
            }
            const e = Entry{ .tensor = t, .gguf_offset = info.offset };
            if (t.device_type == .CPU) {
                try cpu_weights.append(self.allocator, e);
            } else {
                try gpu_weights.append(self.allocator, e);
            }
        }

        // 同名权重只在"同一设备"上共享:CPU 上的 embd 和 GPU 上的 lm_head 都叫 token_embd.weight, 各存一份
        const Bound = struct { ptr: [*]u8, offset: usize };
        var bound: std.StringHashMapUnmanaged(Bound) = .empty;
        defer bound.deinit(self.allocator);

        std.mem.sort(Entry, cpu_weights.items, {}, Entry.lessThan);

        for (cpu_weights.items) |e| {
            if (bound.get(e.tensor.name)) |b| {
                e.tensor.data = b.ptr;
                e.tensor.offset = b.offset;
                continue;
            }
            const device_pools = self.get(.CPU, 0) orelse return error.NoDevicePools;
            const wp = &(device_pools.weight orelse return error.NoWeightPool);
            // CPU 池 host 可寻址:从磁盘直接 pread 进池里, 不经过中间缓冲
            const info = parser.getInfo().findTensor(e.tensor.name).?;
            const block = try wp.allocate(info.bytes(), 32);
            try parser.readTensor(info, block.ptr[0..info.bytes()]);
            e.tensor.data = block.ptr;
            e.tensor.offset = block.offset;
            try bound.put(
                self.allocator,
                e.tensor.name,
                .{ .ptr = block.ptr, .offset = block.offset },
            );
        }

        if (gpu_weights.items.len == 0) {
            return;
        }

        if (!opts.have_cuda) {
            return error.UnsupportedDevice;
        }

        std.mem.sort(Entry, gpu_weights.items, {}, Entry.lessThan);

        // 一块 pinned 暂存区, 所有权重轮流用:pread 进 pinned, 再 H2D。从 pageable 内存 H2D 时驱动内部还要先拷到它自己的 pinned 区, 等于多拷一遍
        const stage: [*]u8 = @ptrCast(cuda.cuda_host_alloc(STAGE_BYTES) orelse return error.PinnedAllocFailed);
        defer cuda.cuda_host_free(stage);

        // 同一设备上的同名权重只上传一次:embedding 和 lm_head 共用 token_embd.weight
        var bound_gpu: std.StringHashMapUnmanaged(Bound) = .empty;
        defer bound_gpu.deinit(self.allocator);

        for (gpu_weights.items) |entry| {
            if (bound_gpu.get(entry.tensor.name)) |b| {
                entry.tensor.data = b.ptr;
                entry.tensor.offset = b.offset;
                continue;
            }
            // 全局节点(layer_id < 0)没登记过"层 → 设备", 直接用张量自己的 device_type, 设备号 0
            const key: DeviceKey = if (entry.tensor.layer_id < 0)
                .{ .device_type = entry.tensor.device_type, .device_id = 0 }
            else
                self.getLayerDevice(@intCast(entry.tensor.layer_id)) orelse return error.LayerNotRegistered;

            const device_pools = self.get(key.device_type, key.device_id) orelse return error.NoDevicePools;
            const wp = &(device_pools.weight orelse return error.NoWeightPool);

            const info = parser.getInfo().findTensor(entry.tensor.name).?;
            const total = info.bytes();
            const block = try wp.allocate(total, 32);
            // 按暂存区大小分块:pread → pinned → H2D(同步拷贝, 返回时暂存区就能复用)
            const file_off = parser.tensorFileOffset(info);
            var done: usize = 0;
            while (done < total) {
                const chunk = @min(STAGE_BYTES, total - done);
                const n = try parser.file.preadAll(stage[0..chunk], file_off + done);
                if (n != chunk) return error.ShortRead;
                cuda.cuda_h2d(block.ptr + done, stage, chunk); // ← 显存:只能 H2D, 不能 @memcpy
                done += chunk;
            }

            entry.tensor.data = block.ptr;
            entry.tensor.offset = block.offset;

            // todo: - 这里避免 outOfMemory
            try bound_gpu.put(
                self.allocator,
                entry.tensor.name,
                .{
                    .ptr = block.ptr,
                    .offset = block.offset,
                },
            );
        }
    }
};
