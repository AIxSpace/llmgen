//! memory/page_atten.zig -- 每层 K/V 的分页缓存(由 module-serving-unfused 的 paged.zig 改名而来)
//! 块内布局 [kv_head][slot][head_dim]。块的分配 / 回收是序列级的, 由 runner/block.zig 的 BlockManager 管,
//! 所有层共用同一张块表;这里只持有每层 K / V 的显存地址。

const std = @import("std");
const types = @import("../types.zig");
const P = @import("pools.zig");

pub const DataType = types.GGMLDataType;

/// 每块 token 数的缺省值(调度器算池容量时用)
pub const PAGE_BLOCK_SIZE: i32 = 16;

/// 每层的 KV 分页状态(对照 PagedAttentionManager::LayerState).k / v 是设备指针.
pub const LayerState = struct {
    active: bool = false,
    n_kv_heads: i32 = 0,
    head_dim: i32 = 0,
    dtype: DataType = .GGML_TYPE_BF16,
    num_blocks: usize = 0,
    k: ?*anyopaque = null,
    v: ?*anyopaque = null,

    /// 一个 token 的 K(或 V)字节数 = n_kv_heads × head_dim × elem.
    pub fn tokenBytes(self: *const LayerState) usize {
        return @as(usize, @intCast(self.n_kv_heads)) * @as(usize, @intCast(self.head_dim)) * self.dtype.size();
    }
};

/// 对照 class PagedAttentionManager:pool_ + layers_(layer_id → LayerState).
pub const PagedAttentionManager = struct {
    alloc: std.mem.Allocator,
    pool: *P.MemoryPool, // KV 块的 backing(kv_cache 池)
    block_size: usize, // 每块 token 数(module-style 固定 PAGE_BLOCK_SIZE=16, 这里可配)
    layers: std.AutoHashMapUnmanaged(i32, *LayerState) = .empty,

    pub fn init(alloc: std.mem.Allocator, pool: *P.MemoryPool, block_size: usize) PagedAttentionManager {
        return .{ .alloc = alloc, .pool = pool, .block_size = block_size };
    }
    pub fn deinit(self: *PagedAttentionManager) void {
        var it = self.layers.valueIterator();
        while (it.next()) |ls| self.alloc.destroy(ls.*);
        self.layers.deinit(self.alloc);
    }

    pub fn getLayer(self: *PagedAttentionManager, layer_id: i32) ?*LayerState {
        return self.layers.get(layer_id);
    }
    pub fn isActive(self: *PagedAttentionManager, layer_id: i32) bool {
        if (self.layers.get(layer_id)) |ls| return ls.active;
        return false;
    }

    /// 对照 init_layer:只设该层元数据(n_kv/head_dim/dtype),active=true;显存由 reserveLayer 切.
    pub fn initLayer(self: *PagedAttentionManager, layer_id: i32, n_kv_heads: i32, head_dim: i32, dtype: DataType) !void {
        const gop = try self.layers.getOrPut(self.alloc, layer_id);
        if (!gop.found_existing) gop.value_ptr.* = try self.alloc.create(LayerState);
        gop.value_ptr.*.* = .{ .active = true, .n_kv_heads = n_kv_heads, .head_dim = head_dim, .dtype = dtype };
    }

    /// 一层 K(或 V)一块的字节数.
    pub fn blockBytes(self: *const PagedAttentionManager, ls: *const LayerState) usize {
        return self.block_size * ls.tokenBytes();
    }

    /// 对照 reserve_layer:为该层从 KV 池切出 K/V 各 num_blocks 个块.
    pub fn reserveLayer(self: *PagedAttentionManager, layer_id: i32, num_blocks: usize) !void {
        const ls = self.layers.get(layer_id) orelse return error.LayerNotActive;
        if (!ls.active) return;
        const per_pool = self.blockBytes(ls) * num_blocks;
        const kblk = try self.pool.allocate(per_pool, P.POOL_ALIGN);
        const vblk = try self.pool.allocate(per_pool, P.POOL_ALIGN);
        ls.num_blocks = num_blocks;
        ls.k = @ptrCast(kblk.ptr);
        ls.v = @ptrCast(vblk.ptr);
    }
};
