const std = @import("std");

const types = @import("../types.zig");

const Tensor = @import("../Tensor.zig");

const common = @import("common.zig");

const OperationType = types.OperationType;
const DeviceType = types.DeviceType;

pub const Phase = enum {
    prefill,
    decode,
};

const Stat = struct {
    calls: u64 = 0, // 调用次数
    ns: u64 = 0, // 消耗时间
    flops: u64 = 0, // 累计浮点运算数
    bytes: u64 = 0, // 累计访存字节(估算
};

const PerDevice = std.EnumArray(
    DeviceType, // 设备类型
    Stat, // 统计指标
);

const Table = std.EnumArray(
    OperationType, // 算子操作类型
    PerDevice, // 在设备上的指标
);

const PhaseTable = std.EnumArray(
    Phase,
    Table,
);

pub const Cost = struct {
    flops: u64 = 0,
    bytes: u64 = 0,
};

/// 估算一个算子的 FLOPs 和访存字节
/// attn_ctx: 注意力看到的上下文长度(executor 传入 seq_pos + 本步 token 数)
pub fn opCost(tensor: *const Tensor, attn_ctx: u64) Cost {
    // 默认: 读入所有 src + 写出, FLOPs ≈ 输出元素数(适用于逐元素算子)
    var cost: Cost = .{ .bytes = tensor.bytes() }; // 写出
    for (tensor.src) |maybe| {
        if (maybe) |s| {
            cost.bytes += s.bytes(); // 读入
        }
    }

    switch (tensor.op_type) {
        .OP_TYPE_LINEAR => {
            const x = tensor.src[0].?;
            const k: u64 = @intCast(x.dims[x.rank() - 1]);
            cost.flops = 2 * tensor.numElements() * k; // 输出每个元素 = K 次乘加
        },
        .OP_TYPE_EMBEDDING => {
            // src = [input_ids, 词表]: 只读 token 对应的行, 不是整张词表
            cost.bytes = tensor.src[0].?.bytes() + 2 * tensor.bytes(); // ids + 读出的行 + 写出
            cost.flops = 0; // 纯拷贝
        },
        .OP_TYPE_APPLY_ROPE => {
            // src = [x, cos, sin], x: [B, n_heads, seq, head_dim], cos/sin: [context_length, rope_dim] f32
            // cos/sin 表只读本步用到的 seq 行
            const x = tensor.src[0].?;
            const tokens: u64 = @intCast(x.dims[2]);
            const rope_dim: u64 = @intCast(tensor.src[1].?.dims[1]);
            cost.bytes = x.bytes() + tensor.bytes() + 2 * tokens * rope_dim * 4;
            cost.flops = tensor.numElements();
        },
        .OP_TYPE_PAGED_ATTN, .OP_TYPE_FLASH_ATTN => {
            // 输出 [tokens, q_dim]: QK^T 与 AV 各 2*q_dim*ctx / token
            cost.flops = 4 * tensor.numElements() * attn_ctx;
            // 历史 K/V 在分页池里, 不在 src 中, 要单独加上: K 和 V 各 ctx × kv_dim 个元素
            const k = tensor.src[1].?; // (B, n_kv, seq, head_dim)
            const kv_dim: u64 = @intCast(k.dims[1] * k.dims[3]);
            cost.bytes += 2 * attn_ctx * kv_dim * k.data_type.size();
        },
        else => cost.flops = tensor.numElements(),
    }
    return cost;
}

pub const OpProfiler = struct {
    enabled: bool = false,
    phase: Phase = .prefill,
    stats: PhaseTable = PhaseTable.initFill(Table.initFill(PerDevice.initFill(.{}))),

    const Self = @This();

    /// LLMGEN_PROFILE 设了就开启; 关闭时 executor 连 Timer 都不启动, 零开销
    pub fn init() Self {
        const profile_env = std.posix.getenv("LLMGEN_PROFILE");
        std.debug.print("profile env = {any}\n", .{profile_env});
        return .{
            .enabled = (profile_env != null),
        };
    }

    pub fn reset(self: *Self) void {
        self.stats = PhaseTable.initFill(Table.initFill(PerDevice.initFill(.{})));
        self.phase = .prefill;
    }

    pub fn record(self: *Self, op: OperationType, device: DeviceType, ns: u64, cost: Cost) void {
        const stat = self.stats.getPtr(self.phase).getPtr(op).getPtr(device);
        stat.calls += 1;
        stat.ns += ns;
        stat.flops += cost.flops;
        stat.bytes += cost.bytes;
    }

    pub fn printPhase(self: *Self, phase: Phase, phase_ns: u64) void {
        const table = self.stats.getPtr(phase);
        const Row = struct {
            op: OperationType,
            device: DeviceType,
            stat: Stat,
        };

        var rows: [Table.len * PerDevice.len]Row = undefined;
        var n: usize = 0;
        var op_ns: u64 = 0;

        for (std.enums.values(OperationType)) |op| {
            for (std.enums.values(DeviceType)) |device| {
                const stat = table.get(op).get(device);
                if (stat.calls == 0) continue;
                rows[n] = .{
                    .op = op,
                    .device = device,
                    .stat = stat,
                };
                n += 1;
                op_ns += stat.ns;
            }
        }

        // 按耗时从大到小
        std.mem.sort(Row, rows[0..n], {}, struct {
            fn lessThan(_: void, a: Row, b: Row) bool {
                return a.stat.ns > b.stat.ns;
            }
        }.lessThan);

        const total_ms = common.nsToMs(phase_ns);

        std.debug.print("\n=== 算子耗时 [{s}] (共 {d:.1} ms) ===\n", .{ @tagName(phase), total_ms });
        std.debug.print("{s:<14} {s:<6} {s:>8} {s:>12} {s:>9} {s:>6} {s:>9} {s:>8} {s:>7}\n", .{ "op", "device", "calls", "total_ms", "avg_us", "pct", "GFLOP/s", "GB/s", "AI" });
        for (rows[0..n]) |row| {
            const ms = common.nsToMs(row.stat.ns);
            const avg_us = @as(f64, @floatFromInt(row.stat.ns)) / @as(f64, @floatFromInt(row.stat.calls)) / 1e3; // 平均耗时
            const ns_f: f64 = @floatFromInt(@max(row.stat.ns, 1));
            const flops_f: f64 = @floatFromInt(row.stat.flops);
            const bytes_f: f64 = @floatFromInt(@max(row.stat.bytes, 1));

            std.debug.print("{s:<14} {s:<6} {d:>8} {d:>12.1} {d:>9.1} {d:>5.1}% {d:>9.1} {d:>8.1} {d:>7.2}\n", .{
                @tagName(row.op)["OP_TYPE_".len..],
                @tagName(row.device),
                row.stat.calls,
                ms,
                avg_us,
                ms / total_ms * 100,
                flops_f / ns_f, // FLOP/ns = GFLOP/s
                bytes_f / ns_f, // B/ns    = GB/s
                flops_f / bytes_f, // AI
            });
        }

        // 采样 / tokenizer decode / 打印 / 激活池分配 等不在算子里的时间
        const other_ns = phase_ns -| op_ns;
        std.debug.print("{s:<14} {s:<6} {s:>8} {d:>12.1} {s:>9} {d:>5.1}%\n", .{
            "(其它)",
            "",
            "",
            common.nsToMs(other_ns),
            "",
            common.nsToMs(other_ns) / total_ms * 100,
        });
    }

    /// 某个阶段所有算子的耗时之和, 用于算框架开销
    pub fn opNs(self: *const Self, phase: Phase) u64 {
        const table = self.stats.getPtrConst(phase);
        var sum: u64 = 0;
        for (std.enums.values(OperationType)) |op| {
            for (std.enums.values(DeviceType)) |device| {
                sum += table.get(op).get(device).ns;
            }
        }
        return sum;
    }
};

test "OpProfiler 累加与 reset" {
    var p = OpProfiler{ .enabled = true };
    p.record(.OP_TYPE_LINEAR, .CUDA, 100, .{ .flops = 10, .bytes = 4 });
    p.record(.OP_TYPE_LINEAR, .CUDA, 50, .{});
    p.record(.OP_TYPE_LINEAR, .CPU, 7, .{});

    const s = p.stats.get(.prefill).get(.OP_TYPE_LINEAR).get(.CUDA);

    try std.testing.expectEqual(@as(u64, 2), s.calls);
    try std.testing.expectEqual(@as(u64, 150), s.ns);
    try std.testing.expectEqual(@as(u64, 7), p.stats.get(.prefill).get(.OP_TYPE_LINEAR).get(.CPU).ns);
    try std.testing.expectEqual(@as(u64, 157), p.opNs(.prefill));
    try std.testing.expectEqual(@as(u64, 0), p.opNs(.decode));
    try std.testing.expectEqual(@as(u64, 10), p.stats.get(.prefill).get(.OP_TYPE_LINEAR).get(.CUDA).flops);

    p.reset();
    try std.testing.expectEqual(@as(u64, 0), p.stats.get(.prefill).get(.OP_TYPE_LINEAR).get(.CUDA).calls);
}
