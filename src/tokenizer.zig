const std = @import("std");
const bu = @import("byte_unicode.zig");
const gguf = @import("gguf.zig");
const GGUFInfo = gguf.GGUFInfo;

pub const Pair = struct {
    left: []const u8,
    right: []const u8,
};

const PairCtx = struct {
    pub fn hash(_: PairCtx, p: Pair) u64 {
        var h = std.hash.Wyhash.init(0);
        h.update(p.left);
        h.update(&[_]u8{0});
        h.update(p.right);
        return h.final();
    }
    pub fn eql(_: PairCtx, a: Pair, b: Pair) bool {
        return std.mem.eql(u8, a.left, b.left) and std.mem.eql(u8, a.right, b.right);
    }
};

const Symbol = struct {
    start: usize,
    len: usize,
    prev: i32,
    next: i32,
};

pub const Tokenizer = struct {
    alloc: std.mem.Allocator,

    vocab: [][]const u8, // id → token 文本(借用 GGUF 缓冲)
    vocab_map: std.StringHashMapUnmanaged(i32) = .empty, // token 文本 → id

    bpe_ranks: std.HashMapUnmanaged(Pair, i32, PairCtx, 80) = .empty,
    added: []AddedToken, // 特殊/多片 token,按长度降序做最长匹配

    bos_id: i32 = -1,
    eos_id: i32 = -1,
    pad_id: i32 = -1,
    add_bos: bool = false,
    add_eos: bool = false,

    const AddedToken = struct {
        text: []const u8,
        id: i32,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, gguf_info: *GGUFInfo) !Self {
        const gguf_metadata = gguf_info.metadata;

        const tokens_v = gguf_metadata.get("tokenizer.ggml.tokens") orelse return error.NoTokens;
        if (tokens_v != .array) return error.NoTokens;
        const tokens = tokens_v.array.items;

        const vocab = try allocator.alloc([]const u8, tokens.len);
        errdefer allocator.free(vocab);

        var self = Tokenizer{
            .alloc = allocator,
            .vocab = vocab,
            .added = &.{},
        };
        errdefer {
            self.vocab_map.deinit(allocator);
            self.bpe_ranks.deinit(allocator);
        }

        try self.vocab_map.ensureTotalCapacity(allocator, @intCast(tokens.len));
        for (tokens, 0..) |v, i| {
            const s = v.string;
            vocab[i] = s;
            try self.vocab_map.put(allocator, s, @intCast(i));
        }

        // merges:"left right" → rank = 行号
        if (gguf_metadata.get("tokenizer.ggml.merges")) |merges_v| {
            if (merges_v == .array) {
                const merges = merges_v.array.items;
                try self.bpe_ranks.ensureTotalCapacity(allocator, @intCast(merges.len));
                for (merges, 0..) |m, i| {
                    const s = m.string;
                    const sp = std.mem.indexOfScalar(u8, s, ' ') orelse continue;
                    try self.bpe_ranks.put(
                        allocator,
                        .{
                            .left = s[0..sp],
                            .right = s[sp + 1 ..],
                        },
                        @intCast(i),
                    );
                }
            }
        }

        // special token
        var added_list: std.ArrayListUnmanaged(AddedToken) = .empty;
        errdefer added_list.deinit(allocator);
        for (vocab, 0..) |tok, i| {
            if (tok.len == 0) continue;
            const pieces = try preTokenize(allocator, tok); // 预分词
            defer allocator.free(pieces);
            if (pieces.len > 1) try added_list.append(allocator, .{ .text = tok, .id = @intCast(i) });
        }
        const added = try added_list.toOwnedSlice(allocator);
        std.sort.pdq(AddedToken, added, {}, struct {
            fn lt(_: void, a: AddedToken, b: AddedToken) bool {
                return a.text.len > b.text.len;
            }
        }.lt);

        self.added = added;

        self.bos_id = @intCast(gguf_metadata.getInt("tokenizer.ggml.bos_token_id", -1));
        self.eos_id = @intCast(gguf_metadata.getInt("tokenizer.ggml.eos_token_id", -1));
        self.pad_id = @intCast(gguf_metadata.getInt("tokenizer.ggml.padding_token_id", -1));
        self.add_bos = gguf_metadata.getInt("tokenizer.ggml.add_bos_token", 0) == 1;
        self.add_eos = gguf_metadata.getInt("tokenizer.ggml.add_eos_token", 0) == 1;

        return self;
    }

    pub fn deinit(self: *Tokenizer) void {
        self.vocab_map.deinit(self.alloc);
        self.bpe_ranks.deinit(self.alloc);
        self.alloc.free(self.vocab);
        self.alloc.free(self.added);
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.vocab.len;
    }

    fn tokenId(self: *const Tokenizer, text: []const u8) ?i32 {
        return self.vocab_map.get(text);
    }
    fn rank(self: *const Tokenizer, left: []const u8, right: []const u8) ?i32 {
        return self.bpe_ranks.get(.{ .left = left, .right = right });
    }

    fn bpe(self: *const Tokenizer, out: *std.ArrayListUnmanaged(i32), word: []const u8) !void {
        const alloc = self.alloc;
        if (word.len == 0) return;
        var syms: std.ArrayListUnmanaged(Symbol) = .empty;
        defer syms.deinit(alloc);
        var i: usize = 0;
        while (i < word.len) {
            const l = bu.utf8Len(word[i]);
            const idx: i32 = @intCast(syms.items.len);
            try syms.append(alloc, .{ .start = i, .len = l, .prev = idx - 1, .next = idx + 1 });
            i += l;
        }
        const n: i32 = @intCast(syms.items.len);
        if (n == 0) return;
        syms.items[@intCast(n - 1)].next = -1;
        const slice = struct {
            fn of(wd: []const u8, s: Symbol) []const u8 {
                return wd[s.start .. s.start + s.len];
            }
        }.of;
        while (true) {
            var best_rank: i32 = std.math.maxInt(i32);
            var best: i32 = -1;
            var k: i32 = 0;
            while (k < n) : (k += 1) {
                const s = syms.items[@intCast(k)];
                if (s.next < 0 or s.prev == -2) continue;
                const j = s.next;
                if (self.rank(slice(word, s), slice(word, syms.items[@intCast(j)]))) |r| {
                    if (r < best_rank) {
                        best_rank = r;
                        best = k;
                    }
                }
            }
            if (best < 0) break;
            const bi: usize = @intCast(best);
            const j: usize = @intCast(syms.items[bi].next);
            syms.items[bi].len += syms.items[j].len;
            syms.items[bi].next = syms.items[j].next;
            if (syms.items[j].next >= 0)
                syms.items[@intCast(syms.items[j].next)].prev = best;
            syms.items[j].prev = -2;
        }
        var cur: i32 = 0;
        while (cur >= 0) {
            const s = syms.items[@intCast(cur)];
            if (self.tokenId(slice(word, s))) |id| try out.append(alloc, id);
            cur = s.next;
        }
    }

    fn encodeChunk(self: *const Tokenizer, out: *std.ArrayListUnmanaged(i32), text: []const u8) !void {
        const pieces = try preTokenize(self.alloc, text);
        defer self.alloc.free(pieces);
        for (pieces) |piece| {
            const mapped = try bu.bytesToUnicode(self.alloc, piece);
            defer self.alloc.free(mapped);
            try self.bpe(out, mapped);
        }
    }

    /// 文本 → token id 序列.调用方 free 返回值.
    pub fn encode(self: *const Tokenizer, text: []const u8) ![]i32 {
        var out: std.ArrayListUnmanaged(i32) = .empty;
        errdefer out.deinit(self.alloc);

        if (self.add_bos and self.bos_id >= 0) {
            try out.append(self.alloc, self.bos_id);
        }

        var pos: usize = 0;
        while (pos < text.len) {
            var earliest: usize = std.math.maxInt(usize);
            var matched_id: i32 = -1;
            var matched_len: usize = 0;
            for (self.added) |added_token| {
                const real_str = try bu.unicodeToBytes(self.alloc, added_token.text);
                defer self.alloc.free(real_str);
                // std.debug.print("added token = {s}, id = {d}\n", .{ real_str, added_token.id });
                if (std.mem.indexOfPos(u8, text, pos, added_token.text)) |found| {
                    if (found < earliest) {
                        earliest = found;
                        matched_id = added_token.id;
                        matched_len = added_token.text.len;
                    }
                }
            }

            if (earliest == std.math.maxInt(usize)) {
                try self.encodeChunk(&out, text[pos..]);
                break;
            }

            if (earliest > pos) try self.encodeChunk(&out, text[pos..earliest]);
            try out.append(self.alloc, matched_id);

            pos = earliest + matched_len;
        }
        if (self.add_eos and self.eos_id >= 0) try out.append(self.alloc, self.eos_id);
        return out.toOwnedSlice(self.alloc);
    }

    fn tokenToText(self: *const Tokenizer, out: *std.ArrayListUnmanaged(u8), id: i32) !void {
        if (id < 0 or id >= self.vocab.len) return;
        const tok = self.vocab[@intCast(id)];
        // 特殊字节 token <0xHH>
        if (tok.len == 6 and tok[0] == '<' and tok[1] == '0' and tok[2] == 'x' and tok[5] == '>') {
            const hi = std.fmt.charToDigit(tok[3], 16) catch 0;
            const lo = std.fmt.charToDigit(tok[4], 16) catch 0;
            try out.append(self.alloc, @intCast(hi * 16 + lo));
            return;
        }
        const bytes = try bu.unicodeToBytes(self.alloc, tok);
        defer self.alloc.free(bytes);
        try out.appendSlice(self.alloc, bytes);
    }

    /// token id 序列 → 文本.遇到 eos 停止.调用方 free 返回值.
    pub fn decode(self: *const Tokenizer, ids: []const i32) ![]u8 {
        var out: std.ArrayListUnmanaged(u8) = .empty;
        errdefer out.deinit(self.alloc);
        for (ids) |id| {
            if (id == self.eos_id) break;
            try self.tokenToText(&out, id);
        }
        return out.toOwnedSlice(self.alloc);
    }
};

/// Qwen3 chat 模板(对照 Qwen3.hpp).think=false 时塞空 <think> 块.
pub fn qwenChatTemplate(allocator: std.mem.Allocator, prompt: []const u8, think: bool) ![]u8 {
    if (think) {
        return std.fmt.allocPrint(allocator, "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n", .{prompt});
    }
    return std.fmt.allocPrint(allocator, "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", .{prompt});
}

pub const ChatMessage = struct {
    role: []const u8, // "user" / "assistant"
    content: []const u8,
};

/// 多轮对话模板(Qwen3):history 里的 assistant 只放正文(不带 <think> 块), 最后补上本轮 assistant 的开头;
/// think = false 时同 qwenChatTemplate 补一个空 think 块
pub fn qwenChatTemplateMulti(allocator: std.mem.Allocator, system: ?[]const u8, history: []const ChatMessage, think: bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    if (system) |s| try out.print(allocator, "<|im_start|>system\n{s}<|im_end|>\n", .{s});
    for (history) |m| try out.print(allocator, "<|im_start|>{s}\n{s}<|im_end|>\n", .{ m.role, m.content });
    try out.appendSlice(allocator, "<|im_start|>assistant\n");
    if (!think) try out.appendSlice(allocator, "<think>\n\n</think>\n\n");
    return out.toOwnedSlice(allocator);
}

// ---------- pre_tokenize(BPE 前切分,同 Day02)----------
fn isLetter(cp: u21) bool {
    return (cp >= 'a' and cp <= 'z') or (cp >= 'A' and cp <= 'Z');
}
fn isDigit(cp: u21) bool {
    return cp >= '0' and cp <= '9';
}
fn isSpace(cp: u21) bool {
    return cp == ' ' or cp == '\t' or cp == '\n' or cp == '\r' or cp == 0x0C or cp == 0x0B;
}
fn isNewline(cp: u21) bool {
    return cp == '\n' or cp == '\r';
}
fn matchContraction(s: []const u8) usize {
    if (s.len < 2 or s[0] != '\'') return 0;
    return switch (s[1]) {
        's', 't', 'd', 'm', 'S', 'T', 'D', 'M' => 2,
        'l', 'L' => if (s.len >= 3 and (s[2] == 'l' or s[2] == 'L')) 3 else 0,
        'v', 'V' => if (s.len >= 3 and (s[2] == 'e' or s[2] == 'E')) 3 else 0,
        'r', 'R' => if (s.len >= 3 and (s[2] == 'e' or s[2] == 'E')) 3 else 0,
        else => 0,
    };
}
pub fn preTokenize(alloc: std.mem.Allocator, text: []const u8) ![][]const u8 {
    var pieces: std.ArrayListUnmanaged([]const u8) = .empty;
    errdefer pieces.deinit(alloc);
    var p: usize = 0;
    const end = text.len;
    while (p < end) {
        const d = bu.utf8Decode(text[p..]);
        const cp = d.cp;
        const clen = d.len;
        if (cp == '\'') {
            const m = matchContraction(text[p..]);
            if (m > 0) {
                try pieces.append(alloc, text[p .. p + m]);
                p += m;
                continue;
            }
        }
        const letter = isLetter(cp);
        const digit = isDigit(cp);
        const space = isSpace(cp);
        const newline = isNewline(cp);
        if (letter) {
            const start = p;
            p += clen;
            while (p < end) {
                const nn = bu.utf8Decode(text[p..]);
                if (isLetter(nn.cp)) p += nn.len else break;
            }
            try pieces.append(alloc, text[start..p]);
            continue;
        }
        if (!space and !newline and !digit and !letter and p + clen < end) {
            const n0 = bu.utf8Decode(text[p + clen ..]);
            if (isLetter(n0.cp)) {
                const start = p;
                p += clen;
                while (p < end) {
                    const nn = bu.utf8Decode(text[p..]);
                    if (isLetter(nn.cp)) p += nn.len else break;
                }
                try pieces.append(alloc, text[start..p]);
                continue;
            }
        }
        if (digit) {
            const start = p;
            var count: usize = 0;
            while (p < end and count < 3) {
                const nn = bu.utf8Decode(text[p..]);
                if (isDigit(nn.cp)) {
                    p += nn.len;
                    count += 1;
                } else break;
            }
            try pieces.append(alloc, text[start..p]);
            continue;
        }
        if (!space and !newline and !digit and !letter) {
            const start = p;
            while (p < end) {
                const nn = bu.utf8Decode(text[p..]);
                if (!isSpace(nn.cp) and !isLetter(nn.cp) and !isDigit(nn.cp)) p += nn.len else break;
            }
            while (p < end) {
                const nn = bu.utf8Decode(text[p..]);
                if (isNewline(nn.cp)) p += nn.len else break;
            }
            try pieces.append(alloc, text[start..p]);
            continue;
        }
        if (cp == ' ' and p + 1 < end) {
            const n0 = bu.utf8Decode(text[p + 1 ..]);
            if (!isSpace(n0.cp) and !isNewline(n0.cp) and !isLetter(n0.cp) and !isDigit(n0.cp)) {
                const start = p;
                p += 1;
                while (p < end) {
                    const nn = bu.utf8Decode(text[p..]);
                    if (!isSpace(nn.cp) and !isLetter(nn.cp) and !isDigit(nn.cp)) p += nn.len else break;
                }
                while (p < end) {
                    const nn = bu.utf8Decode(text[p..]);
                    if (isNewline(nn.cp)) p += nn.len else break;
                }
                try pieces.append(alloc, text[start..p]);
                continue;
            }
        }
        if (space or newline) {
            const start = p;
            while (p < end) {
                const nn = bu.utf8Decode(text[p..]);
                if (isSpace(nn.cp) and !isNewline(nn.cp)) p += nn.len else break;
            }
            if (p < end and isNewline(bu.utf8Decode(text[p..]).cp)) {
                while (p < end) {
                    const nn = bu.utf8Decode(text[p..]);
                    if (isNewline(nn.cp)) p += nn.len else break;
                }
                try pieces.append(alloc, text[start..p]);
                continue;
            }
            p = end;
            try pieces.append(alloc, text[start..p]);
            continue;
        }
        try pieces.append(alloc, text[p .. p + clen]);
        p += clen;
    }
    return pieces.toOwnedSlice(alloc);
}

const testing = std.testing;

test "chat 模板格式正确" {
    const alloc = testing.allocator;
    const s = try qwenChatTemplate(alloc, "你好", true);
    defer alloc.free(s);
    try testing.expectEqualStrings("<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n", s);
}
