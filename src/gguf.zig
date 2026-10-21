const std = @import("std");
const types = @import("types.zig");

const ByteReader = struct {
    data: []const u8,
    pos: usize = 0, // update pos after read

    const Self = @This();

    pub const ReadError = error{
        OutOfBounds,
        BadString,
    };

    pub fn init(data: []const u8) Self {
        return .{
            .data = data,
        };
    }

    pub fn take(self: *Self, n: usize) ReadError![]const u8 {
        if (self.pos + n > self.data.len) return ReadError.OutOfBounds;
        const data_slice = self.data[self.pos .. self.pos + n];
        self.pos += n;
        return data_slice;
    }
    pub fn readU8(self: *Self) ReadError!u8 {
        return (try self.take(1))[0];
    }

    pub fn readI8(self: *Self) ReadError!i8 {
        return @bitCast(try self.readU8());
    }

    pub fn readU16(self: *Self) ReadError!u16 {
        const buffer = (try self.take(2))[0..2];
        return std.mem.readInt(u16, buffer, .little);
    }

    pub fn readI16(self: *Self) ReadError!i16 {
        return @bitCast(try self.readU16());
    }

    pub fn readU32(self: *Self) ReadError!u32 {
        const buffer = (try self.take(4))[0..4];
        return std.mem.readInt(u32, buffer, .little);
    }

    pub fn readI32(self: *Self) ReadError!i32 {
        return @bitCast(try self.readU32());
    }

    pub fn readU64(self: *Self) ReadError!u64 {
        const buffer = (try self.take(8))[0..8];
        return std.mem.readInt(u64, buffer, .little);
    }

    pub fn readI64(self: *Self) ReadError!i64 {
        return @bitCast(try self.readU64());
    }

    pub fn readF32(self: *Self) ReadError!f32 {
        return @bitCast(try self.readU32());
    }

    pub fn readF64(self: *Self) ReadError!f64 {
        return @bitCast(try self.readU64());
    }

    pub fn readBool(self: *Self) ReadError!bool {
        return try self.readU8() != 0;
    }

    pub fn remaining(self: *const Self) usize {
        return self.data.len - self.pos;
    }

    pub fn readGGUFString(self: *Self) ReadError![]const u8 {
        const len = try self.readU64();
        if (len > self.remaining()) {
            return ReadError.BadString;
        }
        return self.take(@intCast(len));
    }
};

pub const MetadataValue = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    uint64: u64,
    int64: i64,
    float32: f32,
    float64: f64,
    boolean: bool,
    string: []const u8,
    array: Array,

    const Array = struct {
        elem_type: types.GGUFType,
        items: []MetadataValue,
    };

    fn asInt(self: MetadataValue) ?i64 {
        return switch (self) {
            .uint8 => |v| v,
            .int8 => |v| v,
            .uint16 => |v| v,
            .int16 => |v| v,
            .uint32 => |v| v,
            .int32 => |v| v,
            .uint64 => |v| @intCast(v),
            .int64 => |v| v,
            .boolean => |v| @intFromBool(v),
            else => null,
        };
    }

    fn asFloat(self: MetadataValue) ?f32 {
        return switch (self) {
            .float32 => |v| v,
            .float64 => |v| @floatCast(v),
            else => if (self.asInt()) |i| @floatFromInt(i) else null,
        };
    }
};

pub const Metadata = struct {
    allocator: std.mem.Allocator,
    kvmap: std.StringHashMapUnmanaged(MetadataValue) = .empty,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.kvmap.valueIterator();
        while (it.next()) |v| {
            freeValue(self.allocator, v.*);
        }
        self.kvmap.deinit(self.allocator);
    }

    pub fn freeValue(allocator: std.mem.Allocator, v: MetadataValue) void {
        if (v == .array) {
            for (v.array.items) |child| {
                freeValue(allocator, child);
            }
            allocator.free(v.array.items);
        }
    }

    pub fn get(self: *const Self, key: []const u8) ?MetadataValue {
        return self.kvmap.get(key);
    }

    pub fn getInt(self: *const Self, key: []const u8, default: i64) i64 {
        const metadata_value = self.kvmap.get(key) orelse return default;
        return metadata_value.asInt() orelse default;
    }

    pub fn getFloat(self: *const Self, key: []const u8, default: f32) f32 {
        const metadata_value = self.kvmap.get(key) orelse return default;
        return metadata_value.asFloat() orelse default;
    }

    pub fn getString(self: *const Self, key: []const u8) ?[]const u8 {
        const metadata_value = self.kvmap.get(key) orelse return null;
        return if (metadata_value == .string) metadata_value.string else null;
    }

    pub fn getArray(self: *const Self, key: []const u8) ?MetadataValue.Array {
        const metadata_value = self.kvmap.get(key) orelse return null;
        return if (metadata_value == .array) metadata_value.array else null;
    }
};

pub const TensorInfo = struct {
    transpose: bool = false,
    dtype: types.GGMLDataType = .GGML_TYPE_F32,
    offset: u64 = 0,
    name: []const u8 = "unknown",
    dimensions: [types.TENSOR_MAX_DIMS]i64 = .{ 0, 0, 0, 0 }, // todo: - this place need use shape
    n_dims: u32 = 0,

    const Self = @This();

    pub fn numElements(self: *const Self) usize {
        var n: usize = 1;
        for (self.dimensions[0..self.n_dims]) |d| {
            n *= @intCast(d);
        }
        return n;
    }

    pub fn bytes(self: *const Self) usize {
        return self.numElements() * self.dtype.size();
    }
};

pub const GGUFInfo = struct {
    version: u32 = 0,
    tensor_count: u64 = 0,
    metadata_kv_count: u64 = 0,
    metadata: Metadata,
    data_offset: u64 = 0,
    tensor_info: []TensorInfo = &.{},
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        self.metadata.deinit();
        self.allocator.free(self.tensor_info);
    }

    pub fn getModelArchitecture(self: *const Self) []const u8 {
        return self.metadata.getString("general.architecture") orelse "unknown";
    }

    pub fn getModelName(self: *const Self) []const u8 {
        return self.metadata.getString("general.name") orelse "unknown";
    }

    pub fn findTensor(self: *const Self, name: []const u8) ?*const TensorInfo {
        for (self.tensor_info) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    pub fn show(self: *Self) void {
        std.debug.print("{s:-<14} GGUF 文件信息 {s:-<14}\n", .{ "", "" });

        // display gguf basic
        std.debug.print("gguf version:                      {d}\n", .{self.version});
        std.debug.print("model arch:                        {s}\n", .{self.getModelArchitecture()});
        std.debug.print("model name:                        {s}\n", .{self.getModelName()});
        std.debug.print("kv_count:                          {d}\n", .{self.metadata_kv_count});
        std.debug.print("tensor_count:                      {d}\n", .{self.tensor_count});
        std.debug.print("data_offset:                       {d}\n", .{self.data_offset});

        // display meatata kv map
        var it = self.metadata.kvmap.iterator();
        while (it.next()) |entry| {
            const key = entry.key_ptr.*;
            const v = entry.value_ptr.*;
            switch (v) {
                .string => |s| std.debug.print(
                    "metadata: {s} = \"{s}\"\n",
                    .{ key, s },
                ),
                .array => |a| std.debug.print(
                    "metadata: {s} = [{s} array, len={d}]\n",
                    .{ key, @tagName(a.elem_type), a.items.len },
                ),
                inline else => |x| std.debug.print(
                    "metadata: {s} = {any}\n",
                    .{ key, x },
                ),
            }
        }

        // display tensor infos
        std.debug.print("{s:-<26} {s:-<8} {s:-<14}\n", .{ "", "", "" });
        std.debug.print("{s:<26} {s:<12}     {s}\n", .{ "名称", "数据类型", "维度" });
        std.debug.print("{s:-<26} {s:-<8} {s:-<14}\n", .{ "", "", "" });
        for (self.tensor_info) |info| {
            std.debug.print("{s:<26} {s:<8}     {any}\n", .{
                info.name,
                info.dtype.name(),
                info.dimensions[0..info.n_dims],
            });
        }
        std.debug.print("{s:-<26} {s:-<8} {s:-<14}\n", .{ "", "", "" });
    }
};

pub const GGUFParser = struct {
    gguf_info: GGUFInfo,
    buffer: []u8, //         文件前缀(只到头部能完整解析为止);metadata 里的字符串(tokenizer 的词表等)指向这里, 要活到最后
    data_offset: u64 = 0, // 张量数据区在文件里的起点(已按 general.alignment 对齐)
    file: std.fs.File, //    张量数据留在磁盘上, 由 readTensor / loadWeights 按需 pread
    allocator: std.mem.Allocator,

    const ParseError = error{
        UnknownGGUFType,
        UnknownDataType,
        BadMagic,
        TooManyDims,
        DimensionTooLarge,
        OutOfMemory,
    } || ByteReader.ReadError;

    /// 第一次读多少字节来解析头部;不够就翻倍重读。Qwen3 的头部(15 万词表 + 15 万 merges)约 6 MB
    const HEADER_PROBE: usize = 16 << 20;

    const Self = @This();

    /// 只读文件前缀直到头部能完整解析(不够就翻倍重读), 张量数据留在磁盘上 —— 1.5 GB 的权重不经过进程堆
    pub fn init(allocator: std.mem.Allocator, filename: []const u8) !Self {
        const file = try std.fs.cwd().openFile(filename, .{});
        errdefer file.close();

        const file_size: usize = @intCast((try file.stat()).size);
        var want: usize = @min(file_size, HEADER_PROBE);
        while (true) {
            const buffer = try allocator.alloc(u8, want);
            const n = try file.preadAll(buffer, 0);
            if (parse(allocator, buffer[0..n])) |gguf_info| {
                return .{
                    .gguf_info = gguf_info,
                    .buffer = buffer,
                    .data_offset = gguf_info.data_offset,
                    .file = file,
                    .allocator = allocator,
                };
            } else |err| {
                allocator.free(buffer);
                // 前缀截断在某个字段 / 字符串中间:读到头了才是真错误, 否则读两倍再试
                const truncated = err == ParseError.OutOfBounds or err == ParseError.BadString;
                if (!truncated or want >= file_size) return err;
                want = @min(file_size, want * 2);
            }
        }
    }

    pub fn deinit(self: *Self) void {
        self.gguf_info.deinit();
        self.allocator.free(self.buffer);
        self.file.close();
    }

    pub fn getInfo(self: *Self) *GGUFInfo {
        return &self.gguf_info;
    }

    pub fn getDataOffset(self: *const Self) u64 {
        return self.data_offset; // 原来写成了 self.dataOffset(字段不存在, 没人调用所以没报错)
    }

    /// 张量在文件里的绝对偏移
    pub fn tensorFileOffset(self: *const Self, tensor_info: *const TensorInfo) u64 {
        return self.data_offset + tensor_info.offset;
    }

    /// 把某张量的原始磁盘字节(未反量化)读进 dst(长度必须 == tensor_info.bytes())
    pub fn readTensor(self: *const Self, tensor_info: *const TensorInfo, dst: []u8) !void {
        std.debug.assert(dst.len == tensor_info.bytes());
        const n = try self.file.preadAll(dst, self.tensorFileOffset(tensor_info));
        if (n != dst.len) return error.ShortRead;
    }

    fn parse(allocator: std.mem.Allocator, buffer: []u8) !GGUFInfo {
        var byte_reader = ByteReader.init(buffer);

        // 1、header
        const magic = try byte_reader.readU32(); // "GGUF" MAGIC STRING
        if (magic != types.GGUF_MAGIC) return ParseError.BadMagic;

        const version = try byte_reader.readU32(); // gguf version
        const tensor_count = try byte_reader.readU64(); // tensors count
        const metadata_kv_count = try byte_reader.readU64(); // metadata kv count

        // 2、metadata
        var metadata = try parseMetadata(allocator, &byte_reader, metadata_kv_count);
        errdefer metadata.deinit();

        const alignment: u64 = @intCast(metadata.getInt("general.alignment", 32)); // use default value

        // 3、tensors
        const tensors = try parseTensors(allocator, &byte_reader, tensor_count);

        // weights data region start offset
        const data_offset = std.mem.alignForward(u64, byte_reader.pos, alignment);

        return .{
            .version = version,
            .tensor_count = tensor_count,
            .metadata_kv_count = metadata_kv_count,
            .metadata = metadata,
            .data_offset = data_offset,
            .tensor_info = tensors,
            .allocator = allocator,
        };
    }

    fn parseMetadata(allocator: std.mem.Allocator, byte_reader: *ByteReader, kv_count: u64) ParseError!Metadata {
        var metadata = Metadata.init(allocator);
        errdefer metadata.deinit();

        try metadata.kvmap.ensureTotalCapacity(allocator, @intCast(kv_count));

        var i: u64 = 0;
        while (i < kv_count) : (i += 1) {
            const key = try byte_reader.readGGUFString(); // key len + key str
            const type_raw = try byte_reader.readU32(); // gguf type
            const gguf_type = std.meta.intToEnum(types.GGUFType, type_raw) catch return ParseError.UnknownGGUFType; // map to enum
            const value = try readMetadataValue(allocator, byte_reader, gguf_type); // read value by gguf type
            try metadata.kvmap.put(allocator, key, value);
        }

        return metadata;
    }

    fn readMetadataValue(allocator: std.mem.Allocator, byte_reader: *ByteReader, gguf_type: types.GGUFType) ParseError!MetadataValue {
        return switch (gguf_type) {
            .uint8 => .{ .uint8 = try byte_reader.readU8() },
            .int8 => .{ .int8 = try byte_reader.readI8() },
            .uint16 => .{ .uint16 = try byte_reader.readU16() },
            .int16 => .{ .int16 = try byte_reader.readI16() },
            .uint32 => .{ .uint32 = try byte_reader.readU32() },
            .int32 => .{ .int32 = try byte_reader.readI32() },
            .uint64 => .{ .uint64 = try byte_reader.readU64() },
            .int64 => .{ .int64 = try byte_reader.readI64() },
            .float32 => .{ .float32 = try byte_reader.readF32() },
            .float64 => .{ .float64 = try byte_reader.readF64() },
            .boolean => .{ .boolean = try byte_reader.readBool() },
            .string => .{ .string = try byte_reader.readGGUFString() },
            .array => blk: { // array 处递归读取
                const elem_raw = try byte_reader.readU32();
                const elem_type = std.meta.intToEnum(types.GGUFType, elem_raw) catch return ParseError.UnknownGGUFType;
                const count = try byte_reader.readU64();
                const items = try allocator.alloc(MetadataValue, @intCast(count));
                errdefer allocator.free(items);
                for (items) |*it| {
                    it.* = try readMetadataValue(allocator, byte_reader, elem_type);
                }
                break :blk .{
                    .array = .{
                        .elem_type = elem_type,
                        .items = items,
                    },
                };
            },
        };
    }

    fn parseTensors(allocator: std.mem.Allocator, byte_reader: *ByteReader, tensor_count: u64) ParseError![]TensorInfo {
        const tensors = try allocator.alloc(TensorInfo, @intCast(tensor_count));
        errdefer allocator.free(tensors);

        for (tensors) |*gguf_tensor_info| { // parse to tensor info
            const name = try byte_reader.readGGUFString();
            const n_dims = try byte_reader.readU32(); // dims numbers
            if (n_dims > types.TENSOR_MAX_DIMS) return ParseError.TooManyDims;
            var dims = [_]i64{ 0, 0, 0, 0 };
            for (0..n_dims) |i| {
                const dim = try byte_reader.readU64();
                if (dim == 0 or dim > std.math.maxInt(i64)) {
                    return ParseError.DimensionTooLarge;
                }
                dims[n_dims - i - 1] = @intCast(dim); // 反转维度顺序，参考 pytorch 的维度标准
            }

            const ggml_data_type = std.meta.intToEnum(types.GGMLDataType, try byte_reader.readU32()) catch return ParseError.UnknownDataType;
            // const dtype = try ggml_type.toDataType();
            const offset = try byte_reader.readU64();

            gguf_tensor_info.* = .{
                .name = name,
                .n_dims = n_dims,
                .dimensions = dims,
                .dtype = ggml_data_type,
                .offset = offset,
            };
        }

        return tensors;
    }
};
