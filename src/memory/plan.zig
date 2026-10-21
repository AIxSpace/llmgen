const std = @import("std");
const Tensor = @import("../Tensor.zig");
const base = @import("../module/base.zig"); // DYN / DYN_S

const Alloc = std.mem.Allocator;

pub const Scale = enum { tok, seq };

pub const Slot = struct {
    node: *Tensor,
    scale: Scale,
    row_bytes: usize, // 对齐后的"每行"字节(单位偏移)
    off: usize, //       在对应区里的"每行"偏移
    until: usize, //     最后一次被使用的执行序下标(图输出 = order.len);check 用
    view_of: ?usize = null, // 视图节点:根节点在 slots 里的下标(不占空间, 地址 = 根的地址)
};

/// 对齐到 256 字节:off × T 仍是 256 的倍数(cuBLAS / 16B 向量化读写都满意)
const ROW_ALIGN: usize = 256;

pub const MemoryPlan = struct {
    alloc: Alloc,
    slots: []Slot, //       与传入的 order 一一对应
    tok_row_bytes: usize, // token 区峰值(每 token 字节)
    seq_row_bytes: usize, // 序列区峰值(每序列字节)

    pub fn deinit(self: *MemoryPlan) void {
        self.alloc.free(self.slots);
    }

    /// 把计划落到具体的 (T, S):给每个节点的 data 赋地址
    pub fn bind(self: *const MemoryPlan, tok_base: usize, seq_base: usize, n_tok: usize, n_seq: usize) void {
        for (self.slots) |s| {
            if (s.view_of) |r| {
                s.node.data = self.slots[r].node.data; // 执行序:根一定在视图之前
                s.node.offset = self.slots[r].node.offset;
                continue;
            }
            const addr = switch (s.scale) {
                .tok => tok_base + s.off * n_tok,
                .seq => seq_base + s.off * n_seq,
            };
            s.node.data = @ptrFromInt(addr);
            s.node.offset = 0; // 规划出来的地址不再对应池里的 offset, 没有 kernel 读它
        }
    }

    pub fn tokArenaBytes(self: *const MemoryPlan, max_tok: usize) usize {
        return self.tok_row_bytes * max_tok;
    }
    pub fn seqArenaBytes(self: *const MemoryPlan, max_seqs: usize) usize {
        return self.seq_row_bytes * max_seqs;
    }

    /// 自检(不依赖 build 的活跃区间分析, 重新从图上推一遍):
    ///   1. 每个节点读的激活(经视图归到根), 它的 until 必须 ≥ 这个节点的位置
    ///   2. 同一个区里活跃区间相交的两个节点, 偏移区间不相交
    pub fn check(self: *const MemoryPlan, order: []const *Tensor) !void {
        for (order, 0..) |n, j| {
            for (n.src) |ms| {
                const s = ms orelse continue;
                const si = indexOf(order, s) orelse continue; // 权重 / 输入 / CACHE 不在规划里
                const r = self.slots[si].view_of orelse si;
                if (self.slots[r].until < j) return error.PlanLivenessBroken;
            }
        }
        for (self.slots, 0..) |x, i| {
            if (x.view_of != null) continue;
            for (self.slots[i + 1 ..], i + 1..) |y, j| {
                if (y.view_of != null or x.scale != y.scale) continue;
                if (x.until < j) continue; // y 开始时 x 已经死了
                const disjoint = x.off + x.row_bytes <= y.off or y.off + y.row_bytes <= x.off;
                if (!disjoint) return error.PlanOverlap;
            }
        }
    }
};

fn indexOf(order: []const *Tensor, t: *const Tensor) ?usize {
    for (order, 0..) |n, i| {
        if (n == t) return i;
    }
    return null;
}

/// 行字节 & 所属区:DYN → token 区, DYN_S → 序列区;其余维相乘 × dtype
fn rowOf(node: *const Tensor) !struct { scale: Scale, bytes: usize } {
    var scale: ?Scale = null;
    var n: usize = 1;
    for (node.dims) |d| {
        if (d == 0) break;
        if (d == base.DYN) {
            scale = .tok;
        } else if (d == base.DYN_S) {
            scale = .seq;
        } else n *= @intCast(d);
    }
    const sc = scale orelse return error.StaticActivationUnsupported;
    return .{ .scale = sc, .bytes = std.mem.alignForward(usize, n * node.data_type.size(), ROW_ALIGN) };
}

const Live = struct { off: usize, size: usize, until: usize };

/// first-fit:在当前存活区间(按偏移排序)之间找第一个放得下的空隙
fn place(live: *std.ArrayList(Live), alloc: Alloc, size: usize, until: usize) !usize {
    std.sort.insertion(Live, live.items, {}, struct {
        fn lt(_: void, a: Live, b: Live) bool {
            return a.off < b.off;
        }
    }.lt);
    var cur: usize = 0;
    for (live.items) |l| {
        if (l.off >= cur + size) break;
        cur = @max(cur, l.off + l.size);
    }
    try live.append(alloc, .{ .off = cur, .size = size, .until = until });
    return cur;
}

/// order:每步都算的节点(执行序, 不含权重 / 输入 / 只算一次的 CACHE);outputs:图输出, 活到一步结束
pub fn build(alloc: Alloc, order: []const *Tensor, outputs: []const *Tensor) !MemoryPlan {
    // 每个节点在执行序里的位置
    var pos: std.AutoHashMapUnmanaged(*const Tensor, usize) = .empty;
    defer pos.deinit(alloc);
    for (order, 0..) |n, i| {
        try pos.put(alloc, n, i);
    }

    // 视图 → 根节点下标(视图的视图也归到同一个根);非视图的根是自己
    const root = try alloc.alloc(usize, order.len);
    defer alloc.free(root);
    for (order, 0..) |n, i| {
        root[i] = i;
        if (n.tensor_type == .TENSOR_TYPE_VIEW) {
            const si = pos.get(n.src[0] orelse return error.ViewWithoutSource) orelse return error.ViewOfNonActivation;
            root[i] = root[si];
        }
    }

    // 最后一次被消费的位置(记在根上)
    const last_use = try alloc.alloc(usize, order.len);
    defer alloc.free(last_use);
    for (order, 0..) |_, i| last_use[i] = i;
    for (order, 0..) |n, i| {
        for (n.src) |ms| {
            const s = ms orelse continue;
            if (pos.get(s)) |si| last_use[root[si]] = @max(last_use[root[si]], i);
        }
    }
    // 图输出活到最后(采样结果要拷回 host)
    for (outputs) |o| {
        if (pos.get(o)) |oi| last_use[root[oi]] = order.len;
    }

    const slots = try alloc.alloc(Slot, order.len);
    errdefer alloc.free(slots);
    var live_tok: std.ArrayList(Live) = .empty;
    defer live_tok.deinit(alloc);
    var live_seq: std.ArrayList(Live) = .empty;
    defer live_seq.deinit(alloc);
    var peak_tok: usize = 0;
    var peak_seq: usize = 0;

    for (order, 0..) |n, i| {
        // 释放:最后使用点 < i 的区间(使用点 == i 的是本节点的输入, 不能与输出重叠)
        inline for (.{ &live_tok, &live_seq }) |lv| {
            var k: usize = 0;
            while (k < lv.items.len) {
                if (lv.items[k].until < i) _ = lv.swapRemove(k) else k += 1;
            }
        }
        if (root[i] != i) {
            const rs = slots[root[i]];
            slots[i] = .{ .node = n, .scale = rs.scale, .row_bytes = 0, .off = rs.off, .until = rs.until, .view_of = root[i] };
            continue;
        }
        const r = try rowOf(n);
        const lv = if (r.scale == .tok) &live_tok else &live_seq;
        const off = try place(lv, alloc, r.bytes, last_use[i]);
        if (r.scale == .tok) peak_tok = @max(peak_tok, off + r.bytes) else peak_seq = @max(peak_seq, off + r.bytes);
        slots[i] = .{ .node = n, .scale = r.scale, .row_bytes = r.bytes, .off = off, .until = last_use[i] };
    }

    return .{ .alloc = alloc, .slots = slots, .tok_row_bytes = peak_tok, .seq_row_bytes = peak_seq };
}
