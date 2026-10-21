// 废弃了
const std = @import("std");
const Tensor = @import("../../Tensor.zig");
const ops = @import("../../ops.zig");
const c = @import("../cpu/common.zig");

const cuda = @import("cuda.zig");

const CudaMemoryResource = @import("../../memory/memory_resource.zig").CudaMemoryResource;

/// 一个住在显存里的张量:t 是描述(data 指向显存), buf 是那块显存。
pub const DevTensor = struct {
    t: Tensor,
    buf: []u8,
    res: *CudaMemoryResource,

    pub fn fromHost(r: *CudaMemoryResource, host: *const Tensor) !DevTensor {
        var d = try empty(r, host.*);
        cuda.cudaH2D(d.buf, host.data.?[0..host.bytes()]);
        d.t.strides = host.strides;
        return d;
    }

    pub fn empty(r: *CudaMemoryResource, like: Tensor) !DevTensor {
        const buf = try r.asIMemoryResource().allocate(like.bytes(), 32);
        var t = like;
        t.data = buf.ptr;
        t.device_type = .CUDA;
        t.src = .{null} ** t.src.len;
        return .{ .t = t, .buf = buf, .res = r };
    }

    pub fn toHost(self: *const DevTensor, dst: []u8) void {
        cuda.cudaD2H(dst, self.buf);
    }
    pub fn deinit(self: *DevTensor) void {
        self.res.asIMemoryResource().deallocate(self.buf);
    }
};

/// out 是 host 输出模板(data 指向 CPU 结果缓冲, device_type = .CPU);srcs 是 host 输入。
/// run 是 ops.kernel.xxx。按各自 dtype 读成 f32 后比较, 容差 tol。
pub fn expectMatch(run: anytype, out: *Tensor, srcs: []const *Tensor, tol: f32) !void {
    const allocator = std.testing.allocator;
    for (srcs, 0..) |s, i| out.src[i] = s;
    try run(out, 0); // CPU 参考

    var r = CudaMemoryResource.init(0);
    var devs: [8]DevTensor = undefined;
    for (srcs, 0..) |s, i| devs[i] = try DevTensor.fromHost(&r, s);
    defer for (devs[0..srcs.len]) |*d| d.deinit();
    var dout = try DevTensor.empty(&r, out.*);
    defer dout.deinit();
    for (0..srcs.len) |i| dout.t.src[i] = &devs[i].t;
    try run(&dout.t, 0); // CUDA
    cuda.CudaContext.instance().sync();

    const host = try allocator.alloc(u8, out.bytes());
    defer allocator.free(host);
    dout.toHost(host);
    var got = out.*;
    got.data = host.ptr;
    var max_diff: f32 = 0;
    for (0..out.numElements()) |i| {
        const d = @abs(c.elemF32(out, i) - c.elemF32(&got, i));
        max_diff = @max(max_diff, d);
        if (d > tol) {
            std.debug.print("mismatch@{d}: cpu={d} cuda={d}\n", .{ i, c.elemF32(out, i), c.elemF32(&got, i) });
            return error.CudaMismatch;
        }
    }
    std.debug.print("[cuda-test] {s}: n={d} max_abs_diff={d}\n", .{ out.name, out.numElements(), max_diff });
}

/// 随机 bf16 张量(值域 [-1, 1)), 数据由调用方 free。
pub fn randBf16(allocator: std.mem.Allocator, rnd: std.Random, n: usize) ![]u16 {
    const v = try allocator.alloc(u16, n);
    for (v) |*x| x.* = c.f32ToBf16(rnd.float(f32) * 2 - 1);
    return v;
}

test "MemcpyImpl<CUDA>:CPU → CUDA → CPU 往返" {
    var r = CudaMemoryResource.init(0);
    var host_src = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var host_dst = [_]f32{0} ** 6;
    var src = Tensor{ .name = "src", .data = @ptrCast(&host_src), .dims = .{ 6, 0, 0, 0 } };
    var mid = try DevTensor.empty(&r, src);
    defer mid.deinit();
    mid.t.op_type = .OP_TYPE_MEMCPY;
    mid.t.src[0] = &src;
    try ops.kernel.memcpy(&mid.t, 0); // 两端设备不同 → 派发给 CUDA 侧 → H2D

    var back = Tensor{ .name = "back", .data = @ptrCast(&host_dst), .dims = .{ 6, 0, 0, 0 }, .op_type = .OP_TYPE_MEMCPY };
    back.src[0] = &mid.t;
    try ops.kernel.memcpy(&back, 0); // out 在 CPU, src 在 CUDA → 仍派发给 CUDA 侧 → D2H
    try std.testing.expectEqualSlices(f32, &host_src, &host_dst);
}

test "LinearImpl<CUDA> 与 CPU 对拍(bf16 x · bf16 W)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const M = 7; // seq
    const K = 256; // in
    const N = 96; // out
    const xs = try randBf16(allocator, prng.random(), M * K);
    defer allocator.free(xs);
    const ws = try randBf16(allocator, prng.random(), N * K);
    defer allocator.free(ws);
    const ys = try allocator.alloc(u16, M * N);
    defer allocator.free(ys);
    var x = Tensor{ .name = "x", .data = @ptrCast(xs.ptr), .data_type = .GGML_TYPE_BF16, .dims = .{ 1, M, K, 0 } };
    var w = Tensor{ .name = "w", .data = @ptrCast(ws.ptr), .data_type = .GGML_TYPE_BF16, .dims = .{ N, K, 0, 0 } };
    var y = Tensor{ .name = "linear", .data = @ptrCast(ys.ptr), .data_type = .GGML_TYPE_BF16, .dims = .{ 1, M, N, 0 } };
    // 容差:K=256 个 bf16 乘积求和, 输出再舍入成 bf16(相对误差 ~2^-8)
    try expectMatch(ops.kernel.linear, &y, &.{ &x, &w }, 5e-2);
}

test "RmsNormImpl<CUDA> 与 CPU 对拍(bf16 x, f32 gain;两种形状)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    // 形状 1:input_norm [1, S=5, 1024];形状 2:q_norm [1, 16, S=3, 128]
    const shapes = [_][4]i64{ .{ 1, 5, 1024, 0 }, .{ 1, 16, 3, 128 } };
    for (shapes) |dims| {
        const hidden: usize = @intCast(if (dims[3] != 0) dims[3] else dims[2]);
        var n: usize = 1;
        for (dims) |d| if (d != 0) {
            n *= @intCast(d);
        };
        const xs = try randBf16(allocator, prng.random(), n);
        defer allocator.free(xs);
        const gs = try allocator.alloc(f32, hidden);
        defer allocator.free(gs);
        for (gs) |*g| g.* = prng.random().float(f32) + 0.5;
        const ys = try allocator.alloc(u16, n);
        defer allocator.free(ys);
        var x = Tensor{ .name = "x", .data = @ptrCast(xs.ptr), .data_type = .GGML_TYPE_BF16, .dims = dims };
        var g = Tensor{ .name = "g", .data = @ptrCast(gs.ptr), .data_type = .GGML_TYPE_F32, .dims = .{ @intCast(hidden), 0, 0, 0 } };
        var y = Tensor{ .name = "rms_norm", .data = @ptrCast(ys.ptr), .data_type = .GGML_TYPE_BF16, .dims = dims };
        y.op_params[0] = 1e-6;
        try expectMatch(ops.kernel.rmsNorm, &y, &.{ &x, &g }, 3e-2);
    }
}

test "Add / Mul / SiLU / Softmax <CUDA> 与 CPU 对拍(bf16)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const dims = [4]i64{ 1, 3, 3072, 0 }; // SwiGLU 中间维
    const n: usize = 3 * 3072;
    const as = try randBf16(allocator, prng.random(), n);
    defer allocator.free(as);
    const bs = try randBf16(allocator, prng.random(), n);
    defer allocator.free(bs);
    const ys = try allocator.alloc(u16, n);
    defer allocator.free(ys);
    var a = Tensor{ .name = "a", .data = @ptrCast(as.ptr), .data_type = .GGML_TYPE_BF16, .dims = dims };
    var b = Tensor{ .name = "b", .data = @ptrCast(bs.ptr), .data_type = .GGML_TYPE_BF16, .dims = dims };

    var y = Tensor{ .name = "add", .data = @ptrCast(ys.ptr), .data_type = .GGML_TYPE_BF16, .dims = dims };
    try expectMatch(ops.kernel.add, &y, &.{ &a, &b }, 1e-2);
    y.name = "mul";
    try expectMatch(ops.kernel.mul, &y, &.{ &a, &b }, 1e-2);
    y.name = "silu";
    y.src = .{null} ** y.src.len;
    try expectMatch(ops.kernel.silu, &y, &.{&a}, 1e-2);
    y.name = "softmax";
    try expectMatch(ops.kernel.softmax, &y, &.{&a}, 1e-3);
}

test "Permute {0,2,1,3} / ApplyRope <CUDA> 与 CPU 对拍(bf16)" {
    const allocator = std.testing.allocator;
    const base = @import("../../module/base.zig");
    var prng = std.Random.DefaultPrng.init(13);
    const S = 5;
    const H = 16;
    const D = 128;
    const n: usize = S * H * D;
    const xs = try randBf16(allocator, prng.random(), n);
    defer allocator.free(xs);
    const ys = try allocator.alloc(u16, n);
    defer allocator.free(ys);

    // permute:[1, S, H, D] → [1, H, S, D]
    var x = Tensor{ .name = "x", .data = @ptrCast(xs.ptr), .data_type = .GGML_TYPE_BF16, .dims = .{ 1, S, H, D } };
    base.computeStrides(&x);
    var p = Tensor{ .name = "permute", .data = @ptrCast(ys.ptr), .data_type = .GGML_TYPE_BF16, .dims = .{ 1, H, S, D } };
    base.computeStrides(&p);
    for ([_]f32{ 0, 2, 1, 3 }, 0..) |v, i| p.op_params[i] = v;
    try expectMatch(ops.kernel.permute, &p, &.{&x}, 0); // 纯搬运, 必须逐位相同

    // rope:x 当作 [1, H, S, D], 表在 host 上先算好(和图里一样恒为 F32)
    const max_seq = 64;
    const cs = try allocator.alloc(f32, max_seq * D);
    defer allocator.free(cs);
    const ss = try allocator.alloc(f32, max_seq * D);
    defer allocator.free(ss);
    var cos = Tensor{ .name = "rope_cos", .data = @ptrCast(cs.ptr), .dims = .{ max_seq, D, 0, 0 }, .op_type = .OP_TYPE_ROPE_CACHE };
    var sin = Tensor{ .name = "rope_sin", .data = @ptrCast(ss.ptr), .dims = .{ max_seq, D, 0, 0 }, .op_type = .OP_TYPE_ROPE_CACHE };
    for ([_]*Tensor{ &cos, &sin }) |t| {
        t.op_params = .{ 1000000, D, max_seq, 0 };
        try ops.kernel.ropeCache(t, 0);
    }
    var xr = Tensor{ .name = "xr", .data = @ptrCast(xs.ptr), .data_type = .GGML_TYPE_BF16, .dims = .{ 1, H, S, D } };
    var r = Tensor{ .name = "apply_rope", .data = @ptrCast(ys.ptr), .data_type = .GGML_TYPE_BF16, .dims = .{ 1, H, S, D } };
    r.op_params = .{ D, D, 7, 0 }; // start_pos = 7
    try expectMatch(ops.kernel.applyRope, &r, &.{ &xr, &cos, &sin }, 1e-2);
}
