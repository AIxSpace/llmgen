// 为什么需要它:GPT-2/Qwen 用"字节级 BPE".词表里的 token 不是原始字节,
// 而是把每个字节先映射成一个"可见 unicode 字符"后的字符串.比如空格(0x20)
// 映射成 'Ġ'(U+0120).这样词表里不会出现控制字符/不可打印字节,BPE 在
// 可见字符上做合并.encode 前要 bytes→unicode,decode 后要 unicode→bytes.

const std = @import("std");

/// 计算 GPT-2 的字节→码点表(comptime 生成,零运行时开销).
/// 规则:可打印 ASCII 段 + Latin-1 段直接映射成自身;其余 68 个字节顺序映射到 256+n.
fn buildByteToCp() [256]u21 {
    @setEvalBranchQuota(20000);
    var table: [256]u21 = undefined;
    var is_direct = [_]bool{false} ** 256;
    // 直接映射的字节范围(与 GPT-2 一致)
    for (33..127) |b| is_direct[b] = true;
    for (161..173) |b| is_direct[b] = true;
    for (174..256) |b| is_direct[b] = true;

    var n: u21 = 0;
    for (0..256) |b| {
        if (is_direct[b]) {
            table[b] = @intCast(b);
        } else {
            table[b] = 256 + n;
            n += 1;
        }
    }
    return table;
}

/// 码点→字节的反查(comptime).码点上限约 323,用 512 长数组.
/// 用 u16 存:合法值 0..255,无效用 0xFFFF--因为字节 255 本身映射到 cp 255,
/// 值就是 0xFF,若用 u8 的 0xFF 当哨兵会和它撞车(曾导致字节 255 丢失).
pub const CP_INVALID: u16 = 0xFFFF;
fn buildCpToByte() [512]u16 {
    @setEvalBranchQuota(20000);
    var table = [_]u16{CP_INVALID} ** 512;
    const fwd = buildByteToCp();
    for (0..256) |b| table[fwd[b]] = @intCast(b);
    return table;
}

pub const byte_to_cp: [256]u21 = buildByteToCp();
pub const cp_to_byte: [512]u16 = buildCpToByte();

// ---------- UTF-8 ----------

/// 一个 UTF-8 首字节对应的序列长度.
pub fn utf8Len(c: u8) u3 {
    if (c < 0x80) return 1;
    if (c & 0xE0 == 0xC0) return 2;
    if (c & 0xF0 == 0xE0) return 3;
    if (c & 0xF8 == 0xF0) return 4;
    return 1;
}

/// 解码一个 UTF-8 字符,返回码点与字节数.容错:非法序列按单字节处理.
pub fn utf8Decode(s: []const u8) struct { cp: u21, len: u3 } {
    const c = s[0];
    if (c < 0x80) return .{ .cp = c, .len = 1 };
    if (c & 0xE0 == 0xC0 and s.len >= 2)
        return .{ .cp = (@as(u21, c & 0x1F) << 6) | (s[1] & 0x3F), .len = 2 };
    if (c & 0xF0 == 0xE0 and s.len >= 3)
        return .{ .cp = (@as(u21, c & 0x0F) << 12) | (@as(u21, s[1] & 0x3F) << 6) | (s[2] & 0x3F), .len = 3 };
    if (c & 0xF8 == 0xF0 and s.len >= 4)
        return .{ .cp = (@as(u21, c & 0x07) << 18) | (@as(u21, s[1] & 0x3F) << 12) | (@as(u21, s[2] & 0x3F) << 6) | (s[3] & 0x3F), .len = 4 };
    return .{ .cp = c, .len = 1 };
}

/// 把码点编码成 UTF-8 追加到 out.
pub fn utf8Encode(out: *std.ArrayListUnmanaged(u8), alloc: std.mem.Allocator, cp: u21) !void {
    if (cp < 0x80) {
        try out.append(alloc, @intCast(cp));
    } else if (cp < 0x800) {
        try out.append(alloc, @intCast(0xC0 | (cp >> 6)));
        try out.append(alloc, @intCast(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        try out.append(alloc, @intCast(0xE0 | (cp >> 12)));
        try out.append(alloc, @intCast(0x80 | ((cp >> 6) & 0x3F)));
        try out.append(alloc, @intCast(0x80 | (cp & 0x3F)));
    } else {
        try out.append(alloc, @intCast(0xF0 | (cp >> 18)));
        try out.append(alloc, @intCast(0x80 | ((cp >> 12) & 0x3F)));
        try out.append(alloc, @intCast(0x80 | ((cp >> 6) & 0x3F)));
        try out.append(alloc, @intCast(0x80 | (cp & 0x3F)));
    }
}

/// 原始字节串 → GPT-2 unicode 字符串(每字节映射成一个可见码点的 UTF-8).
/// 调用方负责 free 返回值.
pub fn bytesToUnicode(alloc: std.mem.Allocator, raw: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(alloc);
    for (raw) |b| {
        try utf8Encode(&out, alloc, byte_to_cp[b]);
    }
    return out.toOwnedSlice(alloc);
}

/// GPT-2 unicode 字符串 → 原始字节串(反查).无法反查的码点跳过.
pub fn unicodeToBytes(alloc: std.mem.Allocator, mapped: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(alloc);
    var i: usize = 0;
    while (i < mapped.len) {
        const d = utf8Decode(mapped[i..]);
        i += d.len;
        if (d.cp < cp_to_byte.len and cp_to_byte[d.cp] != CP_INVALID)
            try out.append(alloc, @intCast(cp_to_byte[d.cp]));
    }
    return out.toOwnedSlice(alloc);
}

const testing = std.testing;

test "可打印 ASCII 直接映射自身" {
    try testing.expectEqual(@as(u21, 'A'), byte_to_cp['A']);
    try testing.expectEqual(@as(u21, '!'), byte_to_cp['!']);
}

test "空格映射到 U+0120 (Ġ)" {
    // 空格 0x20=32:非直接段的字节按 0,1,..,32 顺序映射 → 256+32 = 288 = U+0120
    try testing.expectEqual(@as(u21, 288), byte_to_cp[' ']);
}

test "反查表是正向表的逆" {
    for (0..256) |b| {
        const cp = byte_to_cp[b];
        try testing.expectEqual(@as(u16, @intCast(b)), cp_to_byte[cp]);
    }
}

test "bytes↔unicode 往返:任意字节都可逆" {
    const alloc = testing.allocator;
    const raw = "Hello, 世界! \n\t";
    const mapped = try bytesToUnicode(alloc, raw);
    defer alloc.free(mapped);
    const back = try unicodeToBytes(alloc, mapped);
    defer alloc.free(back);
    try testing.expectEqualStrings(raw, back);
}

test "全 256 字节往返无损" {
    const alloc = testing.allocator;
    var raw: [256]u8 = undefined;
    for (0..256) |i| raw[i] = @intCast(i);
    const mapped = try bytesToUnicode(alloc, &raw);
    defer alloc.free(mapped);
    const back = try unicodeToBytes(alloc, mapped);
    defer alloc.free(back);
    try testing.expectEqualSlices(u8, &raw, back);
}
