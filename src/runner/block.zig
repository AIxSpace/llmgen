const std = @import("std");
const batching = @import("batching.zig");
const Sequence = batching.Sequence;

pub const BlockManager = struct {
    allocator: std.mem.Allocator,

    num_blocks: usize, // blocks 数量
    block_size: usize, // 每个 block 包含 16 个 token

    free: std.ArrayList(i32) = .empty, // 空闲块

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, num_blocks: usize, block_size: usize) !Self {
        var self = BlockManager{
            .allocator = allocator,
            .num_blocks = num_blocks,
            .block_size = block_size,
        };

        // 构造栈结构
        try self.free.ensureTotalCapacity(allocator, num_blocks);
        var i: usize = num_blocks;
        while (i > 0) {
            i -= 1;
            self.free.appendAssumeCapacity(@intCast(i));
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.free.deinit(self.allocator);
    }

    pub fn freeBlocksNum(self: *const Self) usize {
        return self.free.items.len;
    }

    // 确定 block 块是否够用
    pub fn ensure(self: *Self, seq: *Sequence, tokens_num: usize) !bool {
        const need_blocks_num = Sequence.blocksFor(tokens_num, self.block_size);
        const have_blocks_num = seq.block_table.items.len;

        // std.debug.print("need blocks num = {}, have blocks num = {}\n", .{ need_blocks_num, have_blocks_num });

        if (need_blocks_num <= have_blocks_num) {
            return true;
        }

        const new_blokcs_num = need_blocks_num - have_blocks_num;
        if (new_blokcs_num > self.free.items.len) return false;

        try seq.block_table.ensureUnusedCapacity(self.allocator, new_blokcs_num);

        for (0..new_blokcs_num) |_| {
            seq.block_table.appendAssumeCapacity(self.free.pop().?);
        }

        return true;
    }

    pub fn canEnsure(self: *const Self, seq: *const Sequence, n: usize) bool {
        const need_blocks_num = Sequence.blocksFor(n, self.block_size);
        const have_blocks_num = seq.block_table.items.len;
        return need_blocks_num <= have_blocks_num or need_blocks_num - have_blocks_num <= self.free.items.len;
    }

    // 释放 seq 占用的 blocks
    pub fn deallocate(self: *Self, seq: *Sequence) void {
        var i = seq.block_table.items.len;
        while (i > 0) {
            i -= 1;
            self.free.appendAssumeCapacity(seq.block_table.items[i]);
        }
        seq.block_table.clearRetainingCapacity();
        seq.num_computed_tokens = 0;
    }
};

test "block-manager-sequence" {
    const allocator = std.testing.allocator;

    var block_manager = try BlockManager.init(allocator, 4, 16); // 4 * 16 = 64
    defer block_manager.deinit();

    var seq = try Sequence.init(allocator, 0, &[_]i32{0} ** 20, .{});
    defer seq.deinit();

    std.debug.print("run block manager unit testing..\n", .{});

    try std.testing.expect(try block_manager.ensure(&seq, 20)); // 占用  2 块
    try std.testing.expectEqual(2, seq.block_table.items.len);
    try std.testing.expectEqual(2, block_manager.freeBlocksNum());

    var free_blocks_num = block_manager.freeBlocksNum();
    std.debug.print("free blocks num = {}\n", .{free_blocks_num});

    try std.testing.expect(try block_manager.ensure(&seq, 32)); // 扔占 2 块
    try std.testing.expectEqual(2, block_manager.freeBlocksNum());

    free_blocks_num = block_manager.freeBlocksNum();
    std.debug.print("free blocks num = {}\n", .{free_blocks_num});

    try std.testing.expect(try block_manager.ensure(&seq, 33)); // 占 3 块

    try std.testing.expect(!block_manager.canEnsure(&seq, 65)); // 需 5 块，不够了
    try std.testing.expect(!try block_manager.ensure(&seq, 65)); // 需 5 块，不够了
    try std.testing.expectEqual(3, seq.block_table.items.len); // sequence 已经占了 3 块 block

    try std.testing.expectEqual(seq.block_table.items[1] * 16 + 3, seq.slotOf(19, 16)); // 第 19 个 token = 第 1 块偏倚 3

    block_manager.deallocate(&seq);
    try std.testing.expectEqual(4, block_manager.freeBlocksNum());
    try std.testing.expectEqual(0, seq.block_table.items.len);
}
