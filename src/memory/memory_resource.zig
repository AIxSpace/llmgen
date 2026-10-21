const std = @import("std");
const opts = @import("build_options");

const types = @import("../types.zig");
const DeviceType = types.DeviceType;

const cuda = if (opts.have_cuda) @import("../backend/cuda/cuda.zig") else struct {};

const cuda_memory = cuda.memory;
const CudaContext = cuda.CudaContext;

pub const MemoryBlock = struct {
    ptr: [*]u8,
    size: usize = 0,
    offset: usize = 0,
    device_handle: usize = 0,
};

pub const IMemoryResource = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const Self = @This();

    pub const VTable = struct {
        allocate: *const fn (ptr: *anyopaque, size: usize, alignment: usize) anyerror![]u8,
        deallocate: *const fn (ptr: *anyopaque, buf: []u8) void,
        id: *const fn (ptr: *anyopaque) usize,
        getDeviceType: *const fn (ptr: *anyopaque) DeviceType,
        deviceHandle: *const fn (ptr: *anyopaque) usize,
    };

    pub fn allocate(self: Self, size: usize, alignment: usize) ![]u8 {
        return self.vtable.allocate(self.ptr, size, alignment);
    }

    pub fn deallocate(self: Self, buf: []u8) void {
        return self.vtable.deallocate(self.ptr, buf);
    }

    pub fn id(self: Self) usize {
        return self.vtable.id(self.ptr);
    }

    pub fn getDeviceType(self: Self) DeviceType {
        return self.vtable.getDeviceType(self.ptr);
    }

    pub fn deviceHandle(self: Self) usize {
        return self.vtable.deviceHandle(self.ptr);
    }
};

pub const CpuMemoryResource = struct {
    backing_allocator: std.mem.Allocator,
    lock_memory: bool = false,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .backing_allocator = allocator,
        };
    }

    const vtable = IMemoryResource.VTable{
        .allocate = allocateImpl,
        .deallocate = deallocateImpl,
        .id = idImpl,
        .getDeviceType = getDeviceTypeImpl,
        .deviceHandle = deviceHandleImpl,
    };

    pub fn asIMemoryResource(self: *Self) IMemoryResource {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    fn allocateImpl(ptr: *anyopaque, size: usize, alignment: usize) ![]u8 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        _ = alignment; // autofix
        return self.backing_allocator.alignedAlloc(u8, .fromByteUnits(64), size);
    }

    pub fn deallocateImpl(ptr: *anyopaque, buf: []u8) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.backing_allocator.free(@as([]align(64) u8, @alignCast(buf)));
    }

    pub fn idImpl(_: *anyopaque) usize {
        return 0;
    }

    pub fn getDeviceTypeImpl(_: *anyopaque) DeviceType {
        return .CPU;
    }

    pub fn deviceHandleImpl(_: *anyopaque) usize {
        return 0;
    }
};

pub const CudaMemoryResource = struct {
    device_id: usize = 0,

    pub fn init(device_id: usize) CudaMemoryResource {
        _ = CudaContext.instance();
        return .{ .device_id = device_id };
    }

    pub fn asIMemoryResource(self: *CudaMemoryResource) IMemoryResource {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable = IMemoryResource.VTable{
        .allocate = allocateImpl,
        .deallocate = deallocateImpl,
        .id = idImpl,
        .getDeviceType = deviceTypeImpl,
        .deviceHandle = deviceHandleImpl,
    };

    fn allocateImpl(ptr: *anyopaque, size: usize, alignment: usize) anyerror![]u8 {
        _ = ptr;
        _ = alignment; // cudaMalloc 已按 256 字节对齐
        const p = cuda_memory.cuda_malloc(size) orelse return error.OutOfMemory;
        return @as([*]u8, @ptrCast(p))[0..size];
    }

    fn deallocateImpl(ptr: *anyopaque, buf: []u8) void {
        _ = ptr;
        cuda_memory.cuda_free(@ptrCast(buf.ptr));
    }

    fn idImpl(ptr: *anyopaque) usize {
        const self: *CudaMemoryResource = @ptrCast(@alignCast(ptr));
        return self.device_id;
    }

    fn deviceTypeImpl(_: *anyopaque) DeviceType {
        return .CUDA;
    }

    fn deviceHandleImpl(_: *anyopaque) usize {
        return 0;
    }
};
