//! cpu/attention.zig -- PagedAttentionImpl<CPU>(和 cuda/attention/attention.cu 同语义, 给 CUDA 当对拍参考)
//! 两件事:
//!   1. store_kv:本步 k / v [T, nkv, hd] 按 slots[t] 写进本层分页 cache(slot < 0 跳过)
//!   2. 在分页 KV 上做因果 GQA + 在线 softmax
//! decode / prefill / 混合批共用一条路径:第 s 条序列的 query 段是 [cu_q[s], cu_q[s+1]), 算完后 KV 总长 ctx[s],
//! 段内第 j 个 query 的位置 = ctx − q_len + j(chunked prefill 的后半段同样成立)。ctx = 0 的 padding 行输出 0。
//! cache 布局 [block][kv_head][slot][hd], 和 .cu 一致。
const std = @import("std");
const Tensor = @import("../../Tensor.zig");
const c = @import("common.zig");

const MAX_HD = 256;

inline fn bf2f(x: u16) f32 {
    return c.bf16ToF32(x);
}

inline fn f2bf(x: f32) u16 {
    return c.f32ToBf16(x);
}

fn bf16Ptr(p: ?*anyopaque) [*]u16 {
    return @ptrCast(@alignCast(p.?));
}

fn i32Ptr(p: ?*anyopaque) [*]const i32 {
    return @ptrCast(@alignCast(p.?));
}

/// k / v [T, nkv, hd] → cache 里 slot 对应的位置
pub fn storeKv(k: [*]const u16, v: [*]const u16, slots: [*]const i32, kc: [*]u16, vc: [*]u16, n_tok: usize, nkv: usize, hd: usize, bs: usize) void {
    for (0..n_tok) |t| {
        if (slots[t] < 0) continue;
        const slot: usize = @intCast(slots[t]);
        const blk = slot / bs;
        const off = slot % bs;
        for (0..nkv) |h| {
            const dst = ((blk * nkv + h) * bs + off) * hd;
            const src = (t * nkv + h) * hd;
            @memcpy(kc[dst..][0..hd], k[src..][0..hd]);
            @memcpy(vc[dst..][0..hd], v[src..][0..hd]);
        }
    }
}

pub const Args = struct {
    out: [*]u16,
    q: [*]const u16,
    kc: [*]const u16,
    vc: [*]const u16,
    block_tables: [*]const i32,
    ctx_lens: [*]const i32,
    cu_q: [*]const i32,
    n_seq: usize,
    nq: usize,
    nkv: usize,
    hd: usize,
    max_blocks: usize,
    bs: usize,
    scale: f32,
};

/// 分页因果 GQA(不写 KV)。PagedAttentionImpl 和单测共用。
pub fn attend(a: Args) void {
    std.debug.assert(a.hd <= MAX_HD);
    const group = a.nq / a.nkv;
    var acc: [MAX_HD]f32 = undefined;

    for (0..a.n_seq) |s| {
        const q_beg: usize = @intCast(a.cu_q[s]);
        const q_end: usize = @intCast(a.cu_q[s + 1]);
        const q_len = q_end - q_beg;
        const ctx: usize = @intCast(a.ctx_lens[s]);
        const bt = a.block_tables + s * a.max_blocks;

        for (q_beg..q_end) |t| {
            const j = t - q_beg; // 段内第 j 个 query
            for (0..a.nq) |h| {
                const o = a.out + (t * a.nq + h) * a.hd;
                if (ctx < q_len) { // padding 行:没有可看的 key
                    @memset(o[0..a.hd], 0);
                    continue;
                }
                const n_keys = ctx - q_len + j + 1; // 因果:看得到位置 0..pos
                const kvh = h / group;
                const qr = a.q + (t * a.nq + h) * a.hd;

                @memset(acc[0..a.hd], 0);
                var m: f32 = -std.math.inf(f32); // 运行最大值
                var l: f32 = 0; //                    运行分母

                for (0..n_keys) |p| {
                    const blk: usize = @intCast(bt[p / a.bs]);
                    const row = ((blk * a.nkv + kvh) * a.bs + p % a.bs) * a.hd;
                    var dot: f32 = 0;
                    for (0..a.hd) |d| {
                        dot += bf2f(qr[d]) * bf2f(a.kc[row + d]);
                    }
                    const score = dot * a.scale;

                    const m_new = @max(m, score);

                    const corr = @exp(m - m_new); // 旧的累加量按新最大值重新缩放
                    const w = @exp(score - m_new);
                    l = l * corr + w;
                    for (0..a.hd) |d| {
                        acc[d] = acc[d] * corr + w * bf2f(a.vc[row + d]);
                    }
                    m = m_new;
                }

                for (0..a.hd) |d| {
                    o[d] = f2bf(acc[d] / l);
                }
            }
        }
    }
}

/// src[0] = q [1, T, nq, hd], src[1] = k [1, T, nkv, hd], src[2] = v [1, T, nkv, hd] → out [1, T, nq, hd], 全是 bf16
pub const PagedAttentionImpl = struct {
    pub fn execute(_: i32, out: *Tensor) !void {
        if (out.data_type != .GGML_TYPE_BF16) return error.AttentionNeedsBf16;
        const s = &c.g_step;
        const kv = try c.kvOf(out.layer_id);
        const n_tok: usize = @intCast(s.n_tok);
        const nkv: usize = @intCast(s.n_kv_head);
        const hd: usize = @intCast(s.head_dim);
        const bs: usize = @intCast(s.block_size);
        const kc = bf16Ptr(kv.k);
        const vc = bf16Ptr(kv.v);

        storeKv(bf16Ptr(out.src[1].?.data), bf16Ptr(out.src[2].?.data), i32Ptr(s.slots), kc, vc, n_tok, nkv, hd, bs);

        attend(.{
            .out = bf16Ptr(out.data),
            .q = bf16Ptr(out.src[0].?.data),
            .kc = kc,
            .vc = vc,
            .block_tables = i32Ptr(s.block_tables),
            .ctx_lens = i32Ptr(s.ctx),
            .cu_q = i32Ptr(s.cu_q),
            .n_seq = @intCast(s.n_seq),
            .nq = @intCast(s.n_head),
            .nkv = nkv,
            .hd = hd,
            .max_blocks = @intCast(s.max_blocks),
            .bs = bs,
            .scale = s.attn_scale,
        });
    }
};

/// 旧的 FLASH_ATTN 图节点(LLMGEN_FLASH=1)走同一个实现
pub const FlashAttentionImpl = PagedAttentionImpl;

// ============================================================
// 单测:和朴素 f32 参考对拍(分页 + GQA + 因果 + 混批 + padding 行)
// ============================================================
const testing = std.testing;

fn refOne(q: []const u16, kc: []const u16, vc: []const u16, bt: []const i32, nq: usize, nkv: usize, hd: usize, bs: usize, t_q: usize, h: usize, n_keys: usize, out: []f32) !void {
    const kvh = h / (nq / nkv);
    const scores = try testing.allocator.alloc(f32, n_keys);
    defer testing.allocator.free(scores);
    var mx: f32 = -std.math.inf(f32);
    for (0..n_keys) |p| {
        const blk: usize = @intCast(bt[p / bs]);
        const row = ((blk * nkv + kvh) * bs + p % bs) * hd;
        var dot: f32 = 0;
        for (0..hd) |d| dot += bf2f(q[(t_q * nq + h) * hd + d]) * bf2f(kc[row + d]);
        scores[p] = dot / @sqrt(@as(f32, @floatFromInt(hd)));
        mx = @max(mx, scores[p]);
    }
    var sum: f32 = 0;
    for (scores) |*sc| {
        sc.* = @exp(sc.* - mx);
        sum += sc.*;
    }
    @memset(out, 0);
    for (0..n_keys) |p| {
        const blk: usize = @intCast(bt[p / bs]);
        const row = ((blk * nkv + kvh) * bs + p % bs) * hd;
        for (0..hd) |d| out[d] += scores[p] / sum * bf2f(vc[row + d]);
    }
}

test "PagedAttention<CPU>: 分页 + GQA + 因果, 混批(prefill 后半段 + decode + padding)对上 f32 参考" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    const nq = 4;
    const nkv = 2;
    const hd = 64;
    const bs = 16;
    const n_blocks = 24;
    const max_blocks = 8;
    const kc = try a.alloc(u16, n_blocks * nkv * bs * hd);
    defer a.free(kc);
    const vc = try a.alloc(u16, n_blocks * nkv * bs * hd);
    defer a.free(vc);
    for (kc) |*e| e.* = f2bf(rng.float(f32) * 2 - 1);
    for (vc) |*e| e.* = f2bf(rng.float(f32) * 2 - 1);

    // 序列 0: ctx=70, 本步 q_len=30(chunked prefill 后半段); 序列 1: ctx=41 的 decode; 序列 2: padding(ctx=0)
    var bt = [_]i32{0} ** (3 * max_blocks);
    for (0..max_blocks) |i| {
        bt[i] = @intCast((i * 5 + 1) % n_blocks);
        bt[max_blocks + i] = @intCast((i * 7 + 2) % n_blocks);
    }
    const cu_q = [_]i32{ 0, 30, 31, 32 };
    const ctx = [_]i32{ 70, 41, 0 };
    const n_tok = 32;
    const q = try a.alloc(u16, n_tok * nq * hd);
    defer a.free(q);
    for (q) |*e| e.* = f2bf(rng.float(f32) * 2 - 1);
    const out = try a.alloc(u16, n_tok * nq * hd);
    defer a.free(out);

    attend(.{
        .out = out.ptr,
        .q = q.ptr,
        .kc = kc.ptr,
        .vc = vc.ptr,
        .block_tables = &bt,
        .ctx_lens = &ctx,
        .cu_q = &cu_q,
        .n_seq = 3,
        .nq = nq,
        .nkv = nkv,
        .hd = hd,
        .max_blocks = max_blocks,
        .bs = bs,
        .scale = 1.0 / @sqrt(@as(f32, hd)),
    });

    var ref: [hd]f32 = undefined;
    for (0..n_tok) |t| {
        for (0..nq) |h| {
            const got = out[(t * nq + h) * hd ..][0..hd];
            if (t == 31) { // padding 行
                for (got) |g| try testing.expectEqual(@as(f32, 0), bf2f(g));
                continue;
            }
            const s: usize = if (t < 30) 0 else 1;
            const n_keys: usize = if (s == 0) 70 - 30 + t + 1 else 41;
            try refOne(q, kc, vc, bt[s * max_blocks ..][0..max_blocks], nq, nkv, hd, bs, t, h, n_keys, &ref);
            for (ref, got) |r, g| try testing.expectApproxEqAbs(r, bf2f(g), 1e-2);
        }
    }
}

test "storeKv: 按 slot 写进分页 cache, slot < 0 跳过" {
    const nkv = 2;
    const hd = 4;
    const bs = 2;
    var k: [3 * nkv * hd]u16 = undefined;
    var v: [3 * nkv * hd]u16 = undefined;
    for (0..k.len) |i| {
        k[i] = f2bf(@floatFromInt(i));
        v[i] = f2bf(@floatFromInt(100 + i));
    }
    var kc = [_]u16{0} ** (3 * nkv * bs * hd);
    var vc = [_]u16{0} ** (3 * nkv * bs * hd);
    const slots = [_]i32{ 5, -1, 0 }; // token0 → 块 2 偏移 1; token1 跳过; token2 → 块 0 偏移 0
    storeKv(&k, &v, &slots, &kc, &vc, 3, nkv, hd, bs);
    // token0 的 kv 头 1 → dst = ((2*2+1)*2+1)*4 = 44
    try testing.expectEqual(k[1 * hd], kc[44]);
    try testing.expectEqual(v[1 * hd + 3], vc[47]);
    // token2 的 kv 头 0 → dst = 0
    try testing.expectEqual(k[2 * nkv * hd], kc[0]);
    // token1 没写: 块 1 仍为 0
    for (kc[16..32]) |e| try testing.expectEqual(@as(u16, 0), e);
}
