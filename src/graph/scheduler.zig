const std = @import("std");
const types = @import("../types.zig");

const DeviceType = types.DeviceType;

const backend = @import("../backend/backend.zig");
const BackendInfo = backend.BackendInfo;

const graph = @import("graph.zig");
const ComputeGraph = graph.ComputeGraph;

const pools = @import("../memory/pools.zig");
const manager = @import("../memory/manager.zig");
const MemoryManager = manager.MemoryManager;

const memory_plan = @import("../memory/plan.zig");
const executor = @import("../executor.zig");
const qwen3 = @import("../models/qwen3.zig");

const Tensor = @import("../Tensor.zig");

/// KV 预算里给 CUDA Graph、cuBLAS 等杂项留的余量
const KV_RESERVE: usize = 32 << 20;
/// CPU 不按比例算(backend.zig 里 CPU 总内存是写死的 16 GB), 固定给 512 块 = 8192 token, 约 0.9 GB
const HOST_KV_BLOCKS: usize = 512;

pub const GraphScheduler = struct {
    pub const Config = struct {
        vocab_size: usize = 0,
        max_model_len: usize = 1024, // 单条序列最长 token 数:块表列数、位置上限
        max_num_seqs: usize = 1, // 一步最多几条序列 S = 动态维 DYN_S 的上界:序列区、元数据、CUDA Graph 桶
        max_tokens: usize = 1024, // 单趟前向最多 token 数 T = 动态维 DYN 的上界:激活 token 区
        block_size: usize = 16, // KV 分页块大小(token)
        gpu_memory_utilization: f32 = 0.9, // 进程最多用到总显存的这个比例, 权重和激活之外剩下的全给 KV
        /// 0 = CUDA 按 gpu_memory_utilization 自动算、CPU 用 HOST_KV_BLOCKS;> 0 = 固定块数(做"对齐 KV 容量"的对比用)
        num_kv_blocks: usize = 0,
        memory_headroom: f32 = 0.05,
        temperature: f32 = 0.8, // 采样温度
        top_p: f32 = 0.9, // top-p 截断
        seed: u64 = 42, // [labs] 固定 seed, 让两版实现在 logits 相同时抽到同一 token
        model_cfg: ?qwen3.Config = null, // 算 Executor workspace(元数据 / split-K / 采样 scratch)要用头数和 head_dim
    };

    pub const LayerCost = struct {
        layer_id: i32 = -1,
        weight_bytes: usize = 0,
        activation_bytes: usize = 0,
        kv_cache_bytes: usize = 0,

        pub fn total(self: LayerCost) usize {
            return self.weight_bytes + self.activation_bytes + self.kv_cache_bytes;
        }
    };

    pub const LayerAssignment = struct {
        start_layer: i32,
        end_layer: i32,
        device_type: DeviceType,
        device_id: usize,
        bytes: usize,
    };

    allocator: std.mem.Allocator,
    compute_graph: *ComputeGraph,
    config: Config,

    assignments: std.ArrayList(LayerAssignment) = .empty,
    memory_manager: MemoryManager,

    /// 0 = CUDA 按 gpu_memory_utilization 自动算、CPU 用 HOST_KV_BLOCKS;> 0 = 固定块数(做"对齐 KV 容量"的对比用)
    num_kv_blocks: usize = 0, // 实际切了多少块(每层 K、V 各这么多块;块号所有层通用)
    kv_bytes: usize = 0, // KV 总字节(所有层的 K + V)

    plan: ?memory_plan.MemoryPlan = null, // 激活的活跃区间规划(createMemoryPools 里建)
    step_nodes: std.ArrayList(*Tensor) = .empty, // 每步都算的节点(执行序), 和 plan.slots 一一对应;Executor 也用这一份

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, compute_graph: *ComputeGraph, config: Config) Self {
        return .{
            .allocator = allocator,
            .compute_graph = compute_graph,
            .config = config,
            .memory_manager = MemoryManager.init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.plan) |*p| p.deinit();
        self.step_nodes.deinit(self.allocator);

        self.memory_manager.deinit();
        self.assignments.deinit(self.allocator);
    }

    pub fn getMemoryManager(self: *Self) *MemoryManager {
        return &self.memory_manager;
    }

    pub fn schedule(self: *Self, device_backend_infos: []const BackendInfo) !void {
        manager.global_mem_manager = &self.memory_manager;

        // 1. 估算每层的内存开销 (权重 + 激活 + KVCache)
        var costs = try self.estimateLayerCosts();
        defer costs.deinit(self.allocator);

        // 2. 记录内传在 layer 上分配的 cost 情况
        try self.assignLayers(costs.items, device_backend_infos);

        // 3. 分配 layers 和 tensors 与 DeviceType 的对照关系
        try self.applyAssignment();

        // 4. 全局节 (input_ids / embedding) 跟第 0 层走，整张图落在同一设备上
        self.assignGlobalNodes(self.deviceTypeForLayer(0) orelse .CPU);

        // 5. 为跨设备情况添加拷贝节点，方便在不同设备间搬运数据
        try self.insertCopyEdges();

        // 6. 预先申请好内存池
        try self.createMemoryPools(device_backend_infos);

        // 7. KV Cache 内存初始化, kv cache 分页内存初始化
        try self.initializeKVCache(&self.memory_manager);

        // print summary
    }

    /// 按层累加:权重 bytes()、激活 bytesAtTS(max_seq, max_num_seqs)(跳过视图)、KV 按注意力节点算;按 layer_id 升序(含 -1)。
    pub fn estimateLayerCosts(self: *Self) !std.ArrayList(LayerCost) {
        var costs: std.ArrayList(LayerCost) = .empty;
        errdefer costs.deinit(self.allocator);

        // 分层放设备时的估算:激活按单趟上限 (max_tokens, max_num_seqs), KV 至少要放下一条最长的序列
        const max_tok: i64 = @intCast(self.config.max_tokens);
        const max_seqs: i64 = @intCast(self.config.max_num_seqs);
        const page_block_size = self.config.block_size;
        const max_blocks = (self.config.max_model_len + page_block_size - 1) / page_block_size;

        var layer_grops_it = self.compute_graph.getLayerGroups().iterator();
        while (layer_grops_it.next()) |entry| {
            const layer_id = entry.key_ptr.*;
            const tensor_items = entry.value_ptr.items;

            var layer_const = LayerCost{ .layer_id = layer_id };

            for (tensor_items) |t| {
                // 权重
                if (t.tensor_type == .TENSOR_TYPE_WEIGHT) {
                    layer_const.weight_bytes += t.bytes();
                    continue;
                }

                // 激活
                if (!t.isComputed() or t.tensor_type == .TENSOR_TYPE_VIEW) {
                    continue;
                }
                layer_const.activation_bytes += t.bytesAtTS(max_tok, max_seqs);

                // kv cache
                const is_attn = t.op_type == .OP_TYPE_PAGED_ATTN or t.op_type == .OP_TYPE_FLASH_ATTN;
                if (is_attn and t.src[1] != null) {
                    const n_kv: usize = @intCast(t.src[1].?.dims[2]); // k: [1. T, n_kv, hd]
                    const head_dim: usize = @intCast(t.dims[3]);
                    layer_const.kv_cache_bytes += 2 * max_blocks * page_block_size * n_kv * head_dim * t.data_type.size();
                }
            }
            try costs.append(self.allocator, layer_const);
        }
        std.sort.pdq(LayerCost, costs.items, {}, struct {
            fn lt(_: void, a: LayerCost, b: LayerCost) bool {
                return a.layer_id < b.layer_id;
            }
        }.lt);

        return costs;
    }

    /// 按层顺序填充设备:每段占用 = Σ(weight + kv) + max(activation)(激活层间复用, 不累加)。
    /// 设备队列 = 非 CPU 设备按可用内存从大到小, 用完回退 CPU;某设备连一层都放不下 → 报错。
    /// layer -1(embedding / rope 表)也参与估算, 但 applyAssignment 会跳过它, 由 assignGlobalNodes 放到 CPU。
    pub fn assignLayers(self: *Self, costs: []const LayerCost, device_backend_infos: []const BackendInfo) !void {
        self.assignments.clearRetainingCapacity();
        if (costs.len == 0) return;

        var dev_queue: std.ArrayList(*const BackendInfo) = .empty;
        defer dev_queue.deinit(self.allocator);
        var cpu_dev: ?*const BackendInfo = null;
        for (device_backend_infos) |*d| {
            if (d.device_type == .CPU) cpu_dev = d else try dev_queue.append(self.allocator, d);
        }
        std.sort.pdq(*const BackendInfo, dev_queue.items, {}, struct {
            fn gt(_: void, a: *const BackendInfo, b: *const BackendInfo) bool {
                return a.availableMemory() > b.availableMemory();
            }
        }.gt);

        var dev_idx: usize = 0;
        var layer_idx: usize = 0;
        while (layer_idx < costs.len) {
            const cur = if (dev_idx < dev_queue.items.len) dev_queue.items[dev_idx] else (cpu_dev orelse return error.NoDeviceForRemainingLayers);
            const budget = capacity(cur.*, self.config.memory_headroom);
            const seg_start = costs[layer_idx].layer_id;
            var seg_end = seg_start;
            var sum_wkv: usize = 0;
            var max_act: usize = 0;
            var seg_empty = true;
            while (layer_idx < costs.len) {
                const c = costs[layer_idx];
                const cur_act = @max(max_act, c.activation_bytes);
                if (sum_wkv + c.weight_bytes + c.kv_cache_bytes + cur_act > budget and !seg_empty) break;
                sum_wkv += c.weight_bytes + c.kv_cache_bytes;
                max_act = cur_act;
                seg_end = c.layer_id;
                seg_empty = false;
                layer_idx += 1;
            }
            if (seg_empty) return error.LayerDoesNotFit;
            try self.assignments.append(self.allocator, .{
                .start_layer = seg_start,
                .end_layer = seg_end,
                .device_type = cur.device_type,
                .device_id = cur.device_id,
                .bytes = sum_wkv + max_act,
            });
            dev_idx += 1;
        }
    }

    pub fn applyAssignment(self: *Self) !void { // ← 返回 !void, schedule 里加 try
        // 逐层登记 layer → 设备(loadWeights 按它找权重池)
        for (self.assignments.items) |assign| {
            var l = assign.start_layer;
            while (l <= assign.end_layer) : (l += 1) {
                if (l < 0) continue;
                try self.memory_manager.registerLayerDevice(@intCast(l), assign.device_type, assign.device_id);
            }
        }
        for (self.compute_graph.tensors.items) |t| {
            if (t.layer_id < 0) continue;
            if (self.deviceTypeForLayer(t.layer_id)) |device_type| {
                t.device_type = device_type;
            }
        }
    }

    pub fn assignGlobalNodes(self: *Self, device_type: DeviceType) void {
        for (self.compute_graph.tensors.items) |t| {
            if (t.layer_id >= 0) {
                continue;
            }
            t.device_type = if (t.op_type == .OP_TYPE_ROPE_CACHE) .CPU else device_type; // 除了 ROPE CACHE 其他全上 GPU
        }
    }

    pub fn insertCopyEdges(self: *Self) !void {
        const Proxy = struct {
            src: *Tensor,
            device_type: DeviceType,
            proxy: *Tensor,
        };

        var proxies: std.ArrayList(Proxy) = .empty;
        defer proxies.deinit(self.allocator);

        const n = self.compute_graph.tensors.items.len;

        var idx: usize = 0;
        while (idx < n) : (idx += 1) {
            const cur_tensor = self.compute_graph.tensors.items[idx];
            if (cur_tensor.device_type == .UNKNOWN) continue;
            for (&cur_tensor.src) |*p_src| {
                const src_tensor = p_src.* orelse continue;
                // 如果在同类型的设备上直接跳过
                if (src_tensor.device_type == .UNKNOWN or src_tensor.device_type == cur_tensor.device_type) {
                    continue;
                }

                var found_tensor: ?*Tensor = null;

                // 如果已经找到代理 Tensor 则返回
                for (proxies.items) |proxy_item| {
                    if (proxy_item.src == src_tensor and proxy_item.device_type == cur_tensor.device_type) {
                        found_tensor = proxy_item.proxy;
                        break;
                    }
                }

                if (found_tensor == null) {
                    const proxy_tensor = try self.compute_graph.insertMemcpy(src_tensor, cur_tensor.device_type);
                    try proxies.append(self.allocator, .{
                        .src = src_tensor,
                        .device_type = cur_tensor.device_type,
                        .proxy = proxy_tensor,
                    });
                    found_tensor = proxy_tensor;
                }

                p_src.* = found_tensor.?;
            }
        }

        // 图的输出是 S 个 token id, 留在显存里;Executor 自己用一次异步 D2H 取回(fetchOut)

        try self.compute_graph.rebuildOrder();
    }

    pub fn createMemoryPools(self: *Self, device_backend_infos: []const BackendInfo) !void {
        const max_tok: i64 = @intCast(self.config.max_tokens);
        const max_seqs: i64 = @intCast(self.config.max_num_seqs);

        // 1. 每步都算的节点 → 活跃区间规划(只依赖图, 和 T / S 无关)
        self.step_nodes.clearRetainingCapacity();
        for (self.compute_graph.getExecutionOrder()) |t| {
            if (t.isComputed() and t.tensor_type != .TENSOR_TYPE_CACHE) {
                try self.step_nodes.append(self.allocator, t);
            }
        }
        if (self.step_nodes.items.len == 0) return error.EmptyGraph;
        // 两个区只有一个基址, 所以每步节点必须都在同一个设备上(按层拆到多设备要按设备各做一份规划, 暂不支持)
        const step_device = self.step_nodes.items[0].device_type;
        for (self.step_nodes.items) |t| {
            if (t.device_type != step_device) return error.MultiDeviceStepUnsupported;
        }
        if (self.plan) |*p| {
            p.deinit();
        }
        self.plan = try memory_plan.build(
            self.allocator,
            self.step_nodes.items,
            self.compute_graph.getExecutionOutputs(),
        );
        try self.plan.?.check(self.step_nodes.items); // 启动时自检一次:活跃区间相交的节点偏移不重叠

        for (device_backend_infos) |backend_info| {
            var weight_cap: usize = 0;
            var cache_bytes: usize = 0;
            var n_attn: usize = 0; //         本设备上的注意力层数
            var kv_block_bytes: usize = 0; // 一层 K(或 V)一块的字节数
            var used_any = false;

            var seen: std.StringHashMapUnmanaged(void) = .empty;
            defer seen.deinit(self.allocator);

            for (self.compute_graph.tensors.items) |t| {
                if (t.device_type != backend_info.device_type) continue;
                used_any = true;
                if (t.tensor_type == .TENSOR_TYPE_WEIGHT) {
                    if (seen.contains(t.name)) continue; // 同设备上同名权重(tied embedding)只存一份
                    weight_cap += std.mem.alignForward(usize, t.bytes(), 32); // 权重是固定的
                    try seen.put(self.allocator, t.name, {});
                    continue;
                }
                if (!t.isComputed() or t.tensor_type == .TENSOR_TYPE_VIEW) continue;
                if (t.tensor_type == .TENSOR_TYPE_CACHE) {
                    // 只算一次的节点(rope 表及其拷贝代理)不在规划里, 常驻在激活池, 由 Executor 逐个 bump
                    cache_bytes += std.mem.alignForward(
                        usize,
                        t.bytesAtTS(max_tok, max_seqs),
                        32,
                    );
                    continue;
                }

                const is_attn = t.op_type == .OP_TYPE_PAGED_ATTN or t.op_type == .OP_TYPE_FLASH_ATTN;
                if (is_attn and t.src[1] != null) {
                    const head_dim: usize = @intCast(t.dims[3]); // [1, T, n_head, hd]
                    const n_kv: usize = @intCast(t.src[1].?.dims[2]); // k: [1, T, n_kv, hd]
                    kv_block_bytes = self.config.block_size * n_kv * head_dim * t.data_type.size();
                    n_attn += 1;
                }
            }
            if (!used_any) continue;

            // 激活池 = 只算一次的 CACHE 节点 + (跑每步节点的设备)token 区峰值 + 序列区峰值 + Executor 常驻 workspace。
            // 容量是算出来的, 不再需要 64 MB 兜底
            var act_cap = cache_bytes;
            if (backend_info.device_type == step_device) {
                const pl = &self.plan.?;
                const model = self.config.model_cfg orelse return error.NoModelConfig;
                act_cap += std.mem.alignForward(usize, pl.tokArenaBytes(self.config.max_tokens), pools.POOL_ALIGN);
                act_cap += std.mem.alignForward(usize, pl.seqArenaBytes(self.config.max_num_seqs), pools.POOL_ALIGN);
                act_cap += executor.Workspace.of(model, executor.Limits.of(self.config), step_device).total();
            }

            // KV:权重和激活分完之后, 剩下的预算全给它
            var kv_cap: usize = 0;
            if (n_attn > 0) {
                const per_block_all = kv_block_bytes * 2 * n_attn; // 一个块号在所有层的 K + V 里占的总字节
                var n = self.config.num_kv_blocks;
                if (n == 0 and backend_info.device_type == .CPU) n = HOST_KV_BLOCKS;
                if (n == 0) {
                    // used_memory 是建池之前读的, 已经包含 CUDA 上下文本身(几百 MB)和别的进程
                    const budget = @as(f64, @floatFromInt(backend_info.total_memory)) * self.config.gpu_memory_utilization -
                        @as(f64, @floatFromInt(backend_info.used_memory + weight_cap + act_cap + KV_RESERVE));
                    if (budget <= 0) return error.NoMemoryForKvCache;
                    n = @intFromFloat(budget / @as(f64, @floatFromInt(per_block_all)));
                }
                if (n == 0) return error.NoMemoryForKvCache;
                if (n * self.config.block_size < self.config.max_model_len) return error.KvCacheSmallerThanOneSequence;
                self.num_kv_blocks = n;
                self.kv_bytes = n * per_block_all;
                kv_cap = 2 * n_attn * (std.mem.alignForward(usize, n * kv_block_bytes, pools.POOL_ALIGN) + pools.POOL_ALIGN);
            }
            _ = try self.memory_manager.getOrCreate(
                backend_info.device_type,
                backend_info.device_id,
                weight_cap,
                act_cap,
                kv_cap,
            );
        }
    }

    pub fn initializeKVCache(self: *Self, memory_manager: *MemoryManager) !void {
        for (self.assignments.items) |assign| {
            const device_pools = memory_manager.get(assign.device_type, assign.device_id) orelse continue;
            if (device_pools.kv_cache == null) continue;
            const page_atten_manager = try memory_manager.createAttentionManager(
                assign.device_type,
                assign.device_id,
                self.config.block_size,
            );
            var layer_idx = assign.start_layer;
            while (layer_idx <= assign.end_layer) : (layer_idx += 1) {
                for (self.compute_graph.getAllTensors()) |t| {
                    const is_atten = t.op_type == .OP_TYPE_PAGED_ATTN or t.op_type == .OP_TYPE_FLASH_ATTN;

                    if (!is_atten or t.src[1] == null) continue;
                    if (t.layer_id != layer_idx) continue;

                    const head_dim: i32 = @intCast(t.dims[3]); // (1, T, n_head, hd)
                    const n_kv_heads: i32 = @intCast(t.src[1].?.dims[2]); // k: (1, T, n_kv, hd)

                    try page_atten_manager.initLayer(
                        t.layer_id,
                        n_kv_heads,
                        head_dim,
                        t.data_type,
                    );

                    try page_atten_manager.reserveLayer(t.layer_id, self.num_kv_blocks);
                }
            }
        }
    }

    fn capacity(backend_info: BackendInfo, headroom: f32) usize {
        const current_device_available_mem: f32 = @floatFromInt(backend_info.availableMemory());
        return @intFromFloat(current_device_available_mem * (1.0 - headroom));
    }

    pub fn deviceTypeForLayer(self: *const Self, layer_id: i32) ?types.DeviceType {
        for (self.assignments.items) |layer_assignment| {
            if (layer_assignment.start_layer <= layer_id and layer_assignment.end_layer >= layer_id) {
                return layer_assignment.device_type;
            }
        }
        return null;
    }

    pub fn deviceIdForLayer(self: *const Self, layer_id: i32) ?usize {
        for (self.assignments.items) |layer_assignment| {
            if (layer_assignment.start_layer <= layer_id and layer_assignment.end_layer >= layer_id) {
                return layer_assignment.device_id;
            }
        }
        return null;
    }
};
