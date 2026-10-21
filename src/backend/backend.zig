const std = @import("std");
const opts = @import("build_options");

const types = @import("../types.zig");

const cuda = if (opts.have_cuda) @import("cuda/cuda.zig") else struct {};

const DeviceType = types.DeviceType;

pub const BackendInfo = struct {
    device_id: usize = 0,
    device_type: DeviceType = .CPU,
    total_memory: usize = 0,
    used_memory: usize = 0,
    compute_power: f32 = 0,
    bandwidth: f32 = 0,

    pub fn availableMemory(self: BackendInfo) usize {
        return self.total_memory - self.used_memory;
    }
};

pub const BackendProvider = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const Self = @This();

    pub const VTable = struct {
        isAvailable: *const fn (ptr: *anyopaque) bool,
        getDeviceCount: *const fn (ptr: *anyopaque) i32,
        getBackendType: *const fn (ptr: *anyopaque) DeviceType,
        getBackendName: *const fn (ptr: *anyopaque) []const u8,
        getBackendInfo: *const fn (ptr: *anyopaque, device_id: i32) BackendInfo,
    };

    pub fn isAvailable(self: Self) bool {
        return self.vtable.isAvailable(self.ptr);
    }

    pub fn getDeviceCount(self: Self) i32 {
        return self.vtable.getDeviceCount(self.ptr);
    }

    pub fn getBackendType(self: Self) DeviceType {
        return self.vtable.getBackendType(self.ptr);
    }

    pub fn getBackendName(self: Self) []const u8 {
        return self.vtable.getBackendName(self.ptr);
    }

    pub fn getBackendInfo(self: Self, device_id: i32) BackendInfo {
        return self.vtable.getBackendInfo(self.ptr, device_id);
    }
};

pub const CPUBackendProvider = struct {
    const Self = @This();

    var self_instance = .{};

    const vtable = BackendProvider.VTable{
        .isAvailable = vIsAvailable,
        .getDeviceCount = vGetDeviceCount,
        .getBackendType = vGetBackendType,
        .getBackendName = vGetBackendName,
        .getBackendInfo = vGetBackendInfo,
    };

    pub fn provider() BackendProvider {
        return .{
            .ptr = &self_instance,
            .vtable = &vtable,
        };
    }

    fn vIsAvailable(_: *anyopaque) bool {
        return true;
    }

    fn vGetDeviceCount(_: *anyopaque) i32 {
        return 1;
    }

    fn vGetBackendType(_: *anyopaque) DeviceType {
        return .CPU;
    }

    fn vGetBackendName(_: *anyopaque) []const u8 {
        return "CPU";
    }

    fn vGetBackendInfo(_: *anyopaque, device_id: i32) BackendInfo {
        // todo: - fix this place need use real system func to get real memory or system info
        return .{
            .device_id = @intCast(device_id),
            .total_memory = 16 * 1024 * 1024 * 1024,
            .compute_power = 1,
            .bandwidth = 50,
            .device_type = .CPU,
        };
    }
};

pub const CUDABackendProvider = if (opts.have_cuda) struct {
    var self_instance: CUDABackendProvider = .{};
    var name_buf: [128]u8 = undefined;

    const vtable = BackendProvider.VTable{
        .isAvailable = vIsAvailable,
        .getDeviceCount = vGetDeviceCount,
        .getBackendType = vGetBackendType,
        .getBackendName = vGetBackendName,
        .getBackendInfo = vGetBackendInfo,
    };

    pub fn provider() BackendProvider {
        return .{ .ptr = &self_instance, .vtable = &vtable };
    }

    fn vIsAvailable(_: *anyopaque) bool {
        return cuda.CudaContext.instance().deviceCount() > 0;
    }
    fn vGetDeviceCount(_: *anyopaque) i32 {
        return cuda.CudaContext.instance().deviceCount();
    }
    fn vGetBackendType(_: *anyopaque) DeviceType {
        return .CUDA;
    }
    fn vGetBackendName(_: *anyopaque) []const u8 {
        _ = cuda.CudaContext.instance().deviceInfo(0, &name_buf);
        return std.mem.sliceTo(&name_buf, 0);
    }
    fn vGetBackendInfo(_: *anyopaque, device_id: i32) BackendInfo {
        const info = cuda.CudaContext.instance().deviceInfo(device_id, &name_buf);
        return .{
            .device_id = @intCast(device_id),
            .device_type = .CUDA,
            .total_memory = info.total_mem,
            .used_memory = cuda.CudaContext.instance().usedMemory(), // 单卡:cudaMemGetInfo 查的是当前设备
            .compute_power = @floatFromInt(info.sm),
            .bandwidth = 936, // todo: - this place dynamic get
        };
    }
} else struct {};

const BackedRegistry = struct {
    providers: std.ArrayList(BackendProvider) = .empty,

    const Self = @This();

    var global_instance: ?Self = null;
    pub fn getInstance() *Self {
        if (global_instance == null) {
            global_instance = .{};
            global_instance.?.registerProvider(CPUBackendProvider.provider());
            if (opts.have_cuda) {
                global_instance.?.registerProvider(CUDABackendProvider.provider());
            }
        }
        return &global_instance.?;
    }

    pub fn registerProvider(self: *Self, provider: BackendProvider) void {
        if (!provider.isAvailable()) {
            return;
        }
        self.providers.append(std.heap.page_allocator, provider) catch return; // todo: - allocator change
    }

    pub fn getProviders(self: *Self) []BackendProvider {
        return self.providers.items;
    }
};

pub const DeviceManager = struct {
    device_backend_infos: std.ArrayList(BackendInfo) = .empty,
    discovered: bool = false,

    const Self = @This();

    var g_instance: ?Self = null;
    pub fn instance() *Self {
        if (g_instance == null) {
            g_instance = .{};
        }
        return &g_instance.?;
    }

    pub fn getDevices(self: *Self) []const BackendInfo {
        if (!self.discovered) {
            for (BackedRegistry.getInstance().getProviders()) |provider| {
                var device_id: i32 = 0;
                while (device_id < provider.getDeviceCount()) : (device_id += 1) {
                    self.device_backend_infos.append(std.heap.page_allocator, provider.getBackendInfo(device_id)) catch break;
                }
            }
            self.discovered = true;
        }
        return self.device_backend_infos.items;
    }

    pub fn printDevices(_: *Self) !void {
        std.debug.print("=== DeviceManager 设备发现 ===\n", .{});
        for (BackedRegistry.getInstance().getProviders()) |provider| {
            if (!provider.isAvailable()) {
                continue;
            }

            var device_id: i32 = 0;
            while (device_id < provider.getDeviceCount()) : (device_id += 1) {
                const info = provider.getBackendInfo(device_id);
                std.debug.print("   [{s} #{d}] mem={d:.1}GB compute={d:.0} bw={d:.0}GB/s\n", .{
                    provider.getBackendName(),
                    device_id,
                    @as(f64, @floatFromInt(info.total_memory)) / 1e9,
                    info.compute_power,
                    info.bandwidth,
                });
            }
        }
    }
};
