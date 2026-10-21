const std = @import("std");
const types = @import("../types.zig");
const DeviceType = types.DeviceType;

const res = @import("memory_resource.zig");
const IMemoryResource = res.IMemoryResource;
const MemoryBlock = res.MemoryBlock;

pub const POOL_ALIGN: usize = 64;

pub const MemoryPool = struct {
    resource: IMemoryResource,
    buffer: []u8,
    used_: usize = 0, // bump 游标
    peak_: usize = 0, // 最大
    cursor_: usize = 0, // 水位
    name: []const u8,

    const Self = @This();

    pub fn init(resource: IMemoryResource, cap: usize, name: []const u8) !Self {
        const buffer = try resource.allocate(cap, POOL_ALIGN);
        return .{
            .resource = resource,
            .buffer = buffer,
            .name = name,
        };
    }

    pub fn deinit(self: *Self) void {
        self.resource.deallocate(self.buffer);
    }

    pub fn allocate(self: *Self, size: usize, alignment: usize) error{OutOfPoolMemory}!MemoryBlock {
        const offset = std.mem.alignForward(usize, self.used_, alignment);
        if (offset + size > self.buffer.len) return error.OutOfPoolMemory;

        self.used_ = offset + size;
        self.peak_ = @max(self.peak_, self.used_);

        return .{
            .ptr = self.buffer.ptr + offset,
            .size = size,
            .offset = offset,
            .device_handle = self.resource.deviceHandle(),
        };
    }

    pub fn reset(self: *Self) void {
        self.used_ = self.cursor_;
    }

    pub fn resetTo(self: *Self, pos: usize) void {
        std.debug.assert(pos <= self.buffer.len);
        self.used_ = pos;
    }

    pub fn markPersistent(self: *Self) void {
        self.cursor_ = self.used_;
    }

    pub fn used(self: *const Self) usize {
        return self.used_;
    }

    fn peak(self: *const Self) usize {
        return self.peak_;
    }

    fn capacity(self: *const Self) usize {
        return self.buffer.len;
    }

    /// 利用率
    fn utilization(self: *const Self) f64 {
        return if (self.buffer.len > 0) @as(f64, @floatFromInt(self.used_)) / @as(f64, @floatFromInt(self.buffer.len)) else 0;
    }

    pub fn getDeviceType(self: *const Self) DeviceType {
        return self.resource.getDeviceType();
    }

    fn deviceId(self: *const Self) usize {
        return self.resource.id();
    }

    pub fn formatUsage(self: *const Self, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "{s}: used={d}/{d},peak={d} ({d:.1}%)", .{
            self.name,
            self.used_,
            self.buffer.len,
            self.peak_,
            self.utilization() * 100,
        }) catch "";
    }
};

pub const DevicePools = struct {
    weight: ?MemoryPool = null, // 权重池
    activation: ?MemoryPool = null, // 激活池
    kv_cache: ?MemoryPool = null, // kv cache 池

    const Self = @This();

    pub fn deinit(self: *Self) void {
        if (self.weight) |*p| {
            p.deinit();
        }
        if (self.activation) |*p| {
            p.deinit();
        }
        if (self.kv_cache) |*p| {
            p.deinit();
        }
    }

    pub fn resetActivation(self: *Self) void {
        if (self.activation) |*p| {
            p.reset();
        }
    }
};
