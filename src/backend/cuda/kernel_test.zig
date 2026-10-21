//! CUDA kernel 对拍测试(仅 -Dcuda 时编译;make llmgen-cuda-test)。对照 lab module-serving-unfused/kernel_test.zig
//!
//! 绕过 Impl 直接调 C 接口:随机数据 → GPU kernel 与写在本文件里的 f32 CPU 参考逐元素比较(BF16 输入, 容差按 BF16 精度给)。
//! 覆盖:GEMM 两条路(M ≤ SV_GEMV_MAX_M 自写 GEMV / 其余 cuBLAS)、分页 decode 注意力(split-K)、tensor core prefill 注意力、
//!       细粒度逐元素 / norm / rope / 写 KV kernel、embedding / gather_last / greedy 采样。

const std = @import("std");
const cuda_ctx = @import("cuda_context.zig");
const mem = @import("memory/memory.zig");
const memcpy = @import("memory/memcpy.zig");
const attn = @import("attention/attention_v1.zig");
const sample = @import("sample/sample.zig");
const testing = std.testing;

/// 把 llmgen 分散在各个 *.zig 里的 extern 收到一处, 测试里统一写 cu.xxx(和 lab 同样的写法)
const cu = struct {
    const cuda_malloc = mem.cuda_malloc;
    const cuda_free = mem.cuda_free;
    const cuda_h2d = mem.cuda_h2d;
    const cuda_host_alloc = mem.cuda_host_alloc;
    const cuda_host_free = mem.cuda_host_free;
    const cuda_d2h_async = mem.cuda_d2h_async;
    const cuda_sync = cuda_ctx.cuda_sync;
    const cuda_linear = @import("linear/linear.zig").cuda_linear;
    const cuda_add = @import("arithmetic/arithmetic.zig").cuda_add;
    const cuda_mul = @import("arithmetic/arithmetic.zig").cuda_mul;
    const cuda_silu = @import("activation/activation.zig").cuda_silu;
    const cuda_rmsnorm = @import("norm/norm.zig").cuda_rmsnorm;
    const cuda_rope = @import("rope/rope.zig").cuda_rope;
    const cuda_embedding = @import("embedding/embedding.zig").cuda_embedding;
    const cuda_gather_rows = memcpy.cuda_gather_rows;
    const cuda_sample = sample.cuda_sample;
    const cuda_sample_scratch_bytes = sample.cuda_sample_scratch_bytes;
    const cuda_store_kv = attn.cuda_store_kv;
    const cuda_attn_decode = attn.cuda_attn_decode;
    const cuda_attn_prefill = attn.cuda_attn_prefill;
    const cuda_attn_prefill_tile = attn.cuda_attn_prefill_tile;

    /// kernel 发射是异步的:先同步, 再取 cudaGetLastError, 把"参数非法 / 越界"之类的错误变成测试失败
    fn check(what: []const u8) !void {
        cuda_ctx.cuda_sync();
        const e = cuda_ctx.cuda_last_error();
        if (e != 0) {
            std.debug.print("[kernel_test] {s}: cuda error {d}\n", .{ what, e });
            return error.CudaError;
        }
    }
};

/// CUDA 上下文(stream / cuBLAS / workspace)由单例懒初始化
fn initCuda() void {
    _ = cuda_ctx.CudaContext.instance();
}

fn toBf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16); // 就近舍入
}
fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn randBf16(alloc: std.mem.Allocator, rng: std.Random, n: usize, scale: f32) ![]u16 {
    const v = try alloc.alloc(u16, n);
    for (v) |*e| e.* = toBf16((rng.float(f32) * 2 - 1) * scale);
    return v;
}

fn upload(data: anytype) !*anyopaque {
    const bytes = std.mem.sliceAsBytes(data);
    const p = cu.cuda_malloc(bytes.len) orelse return error.OutOfDeviceMemory;
    cu.cuda_h2d(p, bytes.ptr, bytes.len);
    return p;
}

fn download(comptime T: type, alloc: std.mem.Allocator, dev: *anyopaque, n: usize) ![]T {
    const out = try alloc.alloc(T, n);
    cu.cuda_sync();
    const host: [*]u8 = @ptrCast(cu.cuda_host_alloc(n * @sizeOf(T)) orelse return error.PinnedAllocFailed);
    defer cu.cuda_host_free(host);
    cu.cuda_d2h_async(host, dev, n * @sizeOf(T));
    cu.cuda_sync();
    @memcpy(std.mem.sliceAsBytes(out), host[0 .. n * @sizeOf(T)]);
    return out;
}

test "GEMM: GEMV(小 M) 与 cuBLAS 两条路都对上 CPU 参考" {
    initCuda();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(1);
    const rng = prng.random();
    const N = 384;
    const K = 512;
    for ([_]usize{ 1, 2, 3, 4, 8, 33 }) |M| {
        const x = try randBf16(a, rng, M * K, 1);
        defer a.free(x);
        const w = try randBf16(a, rng, N * K, 0.1);
        defer a.free(w);
        const dx = try upload(x);
        defer cu.cuda_free(dx);
        const dw = try upload(w);
        defer cu.cuda_free(dw);
        const dy = cu.cuda_malloc(M * N * 4).?;
        defer cu.cuda_free(dy);
        cu.cuda_linear(dy, 1, dx, dw, @intCast(M), N, K);
        const y = try download(f32, a, dy, M * N);
        defer a.free(y);
        for (0..M) |m| for (0..N) |n| {
            var ref: f32 = 0;
            for (0..K) |k| ref += fromBf16(x[m * K + k]) * fromBf16(w[n * K + k]);
            try testing.expectApproxEqAbs(ref, y[m * N + n], 1e-2);
        };
    }
}

/// CPU 参考: 分页 KV([blocks, nkv, bs, 128]) 上的因果 GQA 注意力, 一条序列.
fn refAttention(q: []const u16, kc: []const u16, vc: []const u16, bt: []const i32, nq: usize, nkv: usize, bs: usize, q_len: usize, ctx: usize, out: []f32) void {
    const hd = 128;
    const G = nq / nkv;
    const scale = 1.0 / @sqrt(@as(f32, hd));
    var scores: [4096]f32 = undefined;
    for (0..q_len) |i| {
        const pos = ctx - q_len + i;
        for (0..nq) |h| {
            const kvh = h / G;
            var mx: f32 = -std.math.inf(f32);
            for (0..pos + 1) |t| {
                const base = ((@as(usize, @intCast(bt[t / bs])) * nkv + kvh) * bs + t % bs) * hd;
                var d: f32 = 0;
                for (0..hd) |e| d += fromBf16(q[(i * nq + h) * hd + e]) * fromBf16(kc[base + e]);
                scores[t] = d * scale;
                mx = @max(mx, scores[t]);
            }
            var sum: f32 = 0;
            for (0..pos + 1) |t| {
                scores[t] = @exp(scores[t] - mx);
                sum += scores[t];
            }
            for (0..hd) |e| {
                var acc: f32 = 0;
                for (0..pos + 1) |t| {
                    const base = ((@as(usize, @intCast(bt[t / bs])) * nkv + kvh) * bs + t % bs) * hd;
                    acc += scores[t] * fromBf16(vc[base + e]);
                }
                out[(i * nq + h) * hd + e] = acc / sum;
            }
        }
    }
}

test "注意力: decode(split-K) 与 tensor core prefill 都对上 CPU 参考(分页 + GQA + 因果)" {
    initCuda();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const nq = 4;
    const nkv = 2;
    const hd = 128;
    const bs = 16;
    const n_blocks = 32;
    const ctx = 150; // 跨多个块且不对齐 16 / 64
    const max_blocks = 16;
    const kc = try randBf16(a, rng, n_blocks * nkv * bs * hd, 1);
    defer a.free(kc);
    const vc = try randBf16(a, rng, n_blocks * nkv * bs * hd, 1);
    defer a.free(vc);
    // 打乱的块表: 验证分页寻址
    var bt = [_]i32{0} ** max_blocks;
    for (0..max_blocks) |i| bt[i] = @intCast((i * 7 + 3) % n_blocks);
    const dk = try upload(kc);
    defer cu.cuda_free(dk);
    const dv = try upload(vc);
    defer cu.cuda_free(dv);
    const dbt = try upload(@as([]const i32, &bt));
    defer cu.cuda_free(dbt);
    const ctx_arr = [_]i32{ctx};
    const dctx = try upload(@as([]const i32, &ctx_arr));
    defer cu.cuda_free(dctx);

    // ---- prefill: 最后 q_len 个位置作为本步 query(模拟 chunked prefill 的后半段) ----
    const q_len = 100;
    const q = try randBf16(a, rng, q_len * nq * hd, 1);
    defer a.free(q);
    const ref = try a.alloc(f32, q_len * nq * hd);
    defer a.free(ref);
    refAttention(q, kc, vc, &bt, nq, nkv, bs, q_len, ctx, ref);

    const dq = try upload(q);
    defer cu.cuda_free(dq);
    const dout = cu.cuda_malloc(q_len * nq * hd * 2).?;
    defer cu.cuda_free(dout);
    const tile: usize = @intCast(cu.cuda_attn_prefill_tile());
    var tseq: [8]i32 = undefined;
    var tq0: [8]i32 = undefined;
    var nt: usize = 0;
    var q0: usize = 0;
    while (q0 < q_len) : (q0 += tile) {
        tseq[nt] = 0;
        tq0[nt] = @intCast(q0);
        nt += 1;
    }
    const cuq = [_]i32{ 0, q_len };
    const dcuq = try upload(@as([]const i32, &cuq));
    defer cu.cuda_free(dcuq);
    const dts = try upload(@as([]const i32, tseq[0..nt]));
    defer cu.cuda_free(dts);
    const dtq = try upload(@as([]const i32, tq0[0..nt]));
    defer cu.cuda_free(dtq);
    cu.cuda_attn_prefill(dout, dq, dk, dv, dbt, dctx, dcuq, dts, dtq, @intCast(nt), nq, nkv, hd, max_blocks, bs, 1.0 / @sqrt(@as(f32, hd)));
    try cu.check("attn_prefill");
    const got = try download(u16, a, dout, q_len * nq * hd);
    defer a.free(got);
    for (ref, got) |r, g| try testing.expectApproxEqAbs(r, fromBf16(g), 2e-2);

    // ---- decode: 只取最后一个位置, 用不同 part_size 覆盖 split-K 合并 ----
    for ([_]c_int{ 64, 128, 512 }) |ps| {
        const dq1 = try upload(q[(q_len - 1) * nq * hd ..]);
        defer cu.cuda_free(dq1);
        const d1 = cu.cuda_malloc(nq * hd * 2).?;
        defer cu.cuda_free(d1);
        const max_parts = @divTrunc(4096 + ps - 1, ps);
        const po = cu.cuda_malloc(@intCast(nq * max_parts * hd * 4)).?;
        defer cu.cuda_free(po);
        const pml = cu.cuda_malloc(@intCast(nq * max_parts * 2 * 4)).?;
        defer cu.cuda_free(pml);
        cu.cuda_attn_decode(d1, dq1, dk, dv, dbt, dctx, 1, nq, nkv, hd, max_blocks, bs, 1.0 / @sqrt(@as(f32, hd)), po, pml, ps, max_parts);
        try cu.check("attn_decode");
        const g1 = try download(u16, a, d1, nq * hd);
        defer a.free(g1);
        for (ref[(q_len - 1) * nq * hd ..], g1) |r, g| try testing.expectApproxEqAbs(r, fromBf16(g), 2e-2);
    }
}

test "细粒度算子: add / mul / silu / rmsnorm(hidden 与每头) / rope / store_kv 对上 CPU 参考" {
    initCuda();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const T = 5;
    const H = 1024;
    const nkv = 8;
    const hd = 128;
    const bs = 16;
    const n = T * H;

    const x = try randBf16(a, rng, n, 2);
    defer a.free(x);
    const y = try randBf16(a, rng, n, 2);
    defer a.free(y);
    const w = try a.alloc(f32, H);
    defer a.free(w);
    for (w) |*e| e.* = 0.5 + rng.float(f32);
    const dx = try upload(x);
    defer cu.cuda_free(dx);
    const dy = try upload(y);
    defer cu.cuda_free(dy);
    const dw = try upload(w);
    defer cu.cuda_free(dw);
    const dout = cu.cuda_malloc(n * 2).?;
    defer cu.cuda_free(dout);

    // 逐元素: 结果 bf16 舍入 → 与 CPU f32 结果差在 bf16 末位以内
    inline for (.{ "add", "mul", "silu" }) |op| {
        if (comptime std.mem.eql(u8, op, "add")) cu.cuda_add(dout, dx, dy, n);
        if (comptime std.mem.eql(u8, op, "mul")) cu.cuda_mul(dout, dx, dy, n);
        if (comptime std.mem.eql(u8, op, "silu")) cu.cuda_silu(dout, dx, n);
        const g = try download(u16, a, dout, n);
        defer a.free(g);
        for (x, y, g) |xa, yb, gv| {
            const fa = fromBf16(xa);
            const fb = fromBf16(yb);
            const want: f32 = if (comptime std.mem.eql(u8, op, "add")) fa + fb else if (comptime std.mem.eql(u8, op, "mul")) fa * fb else fa / (1 + @exp(-fa));
            try testing.expectApproxEqAbs(want, fromBf16(gv), 1e-2 * @max(1, @abs(want)));
        }
    }

    // rmsnorm: 行宽 hidden(block 一行)与 head_dim(warp 一行, 行数 = T·H/128)两条路
    for ([_]usize{ H, hd }) |D| {
        const rows = n / D;
        cu.cuda_rmsnorm(dout, dx, dw, @intCast(rows), @intCast(D), 1e-6);
        const g = try download(u16, a, dout, n);
        defer a.free(g);
        for (0..rows) |r| {
            var ss: f32 = 0;
            for (x[r * D ..][0..D]) |e| ss += fromBf16(e) * fromBf16(e);
            const rstd = 1 / @sqrt(ss / @as(f32, @floatFromInt(D)) + 1e-6);
            for (0..D) |i| {
                const want = fromBf16(toBf16(fromBf16(x[r * D + i]) * rstd)) * w[i]; // 舍入点同 kernel / PyTorch
                try testing.expectApproxEqAbs(want, fromBf16(g[r * D + i]), 2e-2 * @max(1, @abs(want)));
            }
        }
    }

    // rope: x 看成 [T, H/128 头, 128], NEOX 配对 (i, i+64)
    const nh = H / hd;
    // llmgen 是两张表(cos / sin 各 [位置, hd]), kernel 只读前半;lab 是一张 [cos(hd/2) | sin(hd/2)]
    const cos_t = try a.alloc(f32, 64 * hd);
    defer a.free(cos_t);
    const sin_t = try a.alloc(f32, 64 * hd);
    defer a.free(sin_t);
    for (0..64) |p| for (0..hd / 2) |i| {
        const ang = @as(f32, @floatFromInt(p)) / std.math.pow(f32, 10000, @as(f32, @floatFromInt(2 * i)) / hd);
        cos_t[p * hd + i] = @cos(ang);
        cos_t[p * hd + hd / 2 + i] = @cos(ang);
        sin_t[p * hd + i] = @sin(ang);
        sin_t[p * hd + hd / 2 + i] = @sin(ang);
    };
    const pos = [_]i32{ 3, 17, 0, 42, 9 };
    const dcos = try upload(cos_t);
    defer cu.cuda_free(dcos);
    const dsin = try upload(sin_t);
    defer cu.cuda_free(dsin);
    const dpos = try upload(&pos);
    defer cu.cuda_free(dpos);
    cu.cuda_rope(dout, dx, dcos, dsin, dpos, T, nh, hd);
    try cu.check("rope");
    {
        const g = try download(u16, a, dout, n);
        defer a.free(g);
        for (0..T) |t| for (0..nh) |h| for (0..hd / 2) |i| {
            const base = (t * nh + h) * hd;
            const p: usize = @intCast(pos[t]);
            const c = cos_t[p * hd + i];
            const sn = sin_t[p * hd + i];
            const xa = fromBf16(x[base + i]);
            const xb = fromBf16(x[base + i + hd / 2]);
            try testing.expectApproxEqAbs(xa * c - xb * sn, fromBf16(g[base + i]), 2e-2);
            try testing.expectApproxEqAbs(xb * c + xa * sn, fromBf16(g[base + i + hd / 2]), 2e-2);
        };
    }

    // store_kv: k/v[T, nkv, 128] → cache 块内 [kv_head][slot][hd]; slot < 0 不写(逐位拷贝)
    const slots = [_]i32{ 3, 17, -1, 42, 9 };
    const dslots = try upload(&slots);
    defer cu.cuda_free(dslots);
    const kv_n = 4 * nkv * bs * hd; // 4 块
    const zeros = try a.alloc(u16, kv_n);
    defer a.free(zeros);
    @memset(zeros, 0);
    const dkc = try upload(zeros);
    defer cu.cuda_free(dkc);
    const dvc = try upload(zeros);
    defer cu.cuda_free(dvc);
    cu.cuda_store_kv(dx, dy, dslots, dkc, dvc, T, nkv, hd, bs);
    const kc = try download(u16, a, dkc, kv_n);
    defer a.free(kc);
    const vc = try download(u16, a, dvc, kv_n);
    defer a.free(vc);
    var written: usize = 0;
    for (0..T) |t| {
        if (slots[t] < 0) continue;
        const slot: usize = @intCast(slots[t]);
        const blk = slot / bs;
        const off = slot % bs;
        for (0..nkv) |h| {
            const dst = ((blk * nkv + h) * bs + off) * hd;
            const src = (t * nkv + h) * hd;
            try testing.expectEqualSlices(u16, x[src..][0..hd], kc[dst..][0..hd]);
            try testing.expectEqualSlices(u16, y[src..][0..hd], vc[dst..][0..hd]);
            written += hd;
        }
    }
    var nonzero: usize = 0;
    for (kc) |e| {
        if (e != 0) nonzero += 1;
    }
    try testing.expect(nonzero <= written); // slot=-1 的行没写
}

test "首尾算子: embedding / gather_last / greedy 采样 对上 CPU 参考" {
    initCuda();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(9);
    const rng = prng.random();
    const V = 1000; // 词表(不是 2 的幂, 覆盖采样分段的尾巴)
    const H = 256;
    const T = 6;

    // embedding:out[t] = W[ids[t]], 逐位拷贝
    const wte = try randBf16(a, rng, V * H, 1);
    defer a.free(wte);
    const ids = [_]i32{ 7, 0, 999, 7, 123, 500 };
    const dwte = try upload(wte);
    defer cu.cuda_free(dwte);
    const dids = try upload(&ids);
    defer cu.cuda_free(dids);
    const demb = cu.cuda_malloc(T * H * 2).?;
    defer cu.cuda_free(demb);
    cu.cuda_embedding(demb, dids, dwte, T, H);
    try cu.check("embedding");
    const emb = try download(u16, a, demb, T * H);
    defer a.free(emb);
    for (0..T) |t| {
        const id: usize = @intCast(ids[t]);
        try testing.expectEqualSlices(u16, wte[id * H ..][0..H], emb[t * H ..][0..H]);
    }

    // gather_last:按行号取出 S 行(每条序列最后一个 token), 逐位拷贝
    const S = 3;
    const last = [_]i32{ 1, 3, 5 };
    const dlast = try upload(&last);
    defer cu.cuda_free(dlast);
    const dg = cu.cuda_malloc(S * H * 2).?;
    defer cu.cuda_free(dg);
    cu.cuda_gather_rows(dg, demb, dlast, S, H);
    try cu.check("gather_rows");
    const g = try download(u16, a, dg, S * H);
    defer a.free(g);
    for (0..S) |s| {
        const r: usize = @intCast(last[s]);
        try testing.expectEqualSlices(u16, emb[r * H ..][0..H], g[s * H ..][0..H]);
    }

    // greedy 采样(温度 0):每行取 argmax;把最大值放在不同的段里
    const logits = try randBf16(a, rng, S * V, 4);
    defer a.free(logits);
    const peak = [_]usize{ 0, 517, V - 1 };
    for (0..S) |s| logits[s * V + peak[s]] = toBf16(10);
    const temps = [_]f32{ 0, 0, 0 };
    const seed = [_]u64{0x5eed};
    const dlog = try upload(logits);
    defer cu.cuda_free(dlog);
    const dtemps = try upload(&temps);
    defer cu.cuda_free(dtemps);
    const dseed = try upload(&seed);
    defer cu.cuda_free(dseed);
    const dout = cu.cuda_malloc(S * 4).?;
    defer cu.cuda_free(dout);
    const scratch = cu.cuda_malloc(cu.cuda_sample_scratch_bytes(S)).?;
    defer cu.cuda_free(scratch);
    cu.cuda_sample(dout, dlog, dtemps, dseed, S, V, scratch);
    try cu.check("sample");
    const got = try download(i32, a, dout, S);
    defer a.free(got);
    for (0..S) |s| try testing.expectEqual(@as(i32, @intCast(peak[s])), got[s]);
}
