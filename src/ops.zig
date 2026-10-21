const std = @import("std");
const opts = @import("build_options");

const types = @import("types.zig");
const DeviceType = types.DeviceType;

const Tensor = @import("Tensor.zig");

const CPUBackend = @import("backend/cpu/cpu.zig");
const CUDABackend = if (opts.have_cuda) @import("backend/cuda/cuda.zig") else struct {};

const common = @import("backend/cpu/common.zig");

pub fn setGlobalAllocator(allocator: std.mem.Allocator) void {
    common.global_allocator = allocator;
}

pub fn setStep(step: common.StepState) void {
    common.g_step = step;
}

// 动态匹配的后端算子
fn Backend(comptime D: DeviceType) type {
    return switch (D) {
        .CPU => CPUBackend,
        .CUDA => if (opts.have_cuda) CUDABackend else @compileError("CUDA backend not enabled (-Dcuda=true)"),
        else => @compileError("no this backend: " ++ @tagName(D)),
    };
}

fn backendGeneric(comptime D: types.DeviceType, comptime name: []const u8) type {
    const B = Backend(D);
    if (!@hasDecl(B, name)) {
        @compileError(name ++ "<" ++ @tagName(D) ++ "> 未实现");
    }
    return @field(B, name);
}

pub fn MemcpyImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "MemcpyImpl");
}

pub fn EmbeddingImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "EmbeddingImpl");
}

pub fn RmsNormImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "RmsNormImpl");
}

pub fn LinearImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "LinearImpl");
}

pub fn MatmulImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "MatmulImpl");
}

pub fn ApplyRopeImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "ApplyRopeImpl");
}

pub fn PagedAttentionImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "PagedAttentionImpl");
}

pub fn NaiveAttentionImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "NaiveAttentionImpl");
}

pub fn FlashAttentionImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "FlashAttentionImpl");
}

pub fn SiluImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "SiluImpl");
}

pub fn AddImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "AddImpl");
}

pub fn MulImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "MulImpl");
}

pub fn ReshapeImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "ReshapeImpl");
}

pub fn PermuteImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "PermuteImpl");
}

pub fn SoftmaxImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "SoftmaxImpl");
}

pub fn RopeCacheImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "RopeCacheImpl");
}

pub fn GatherLastImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "GatherLastImpl");
}

pub fn SampleImpl(comptime D: DeviceType) type {
    return backendGeneric(D, "SampleImpl");
}

pub const kernel = struct {
    pub fn memcpy(out_tensor: *Tensor, device_id: i32) !void {
        var device_type = out_tensor.device_type;
        const dst_tensor = out_tensor;

        if (out_tensor.src[0]) |src| {
            if (src.device_type != dst_tensor.device_type) {
                if (dst_tensor.device_type != .CPU) {
                    device_type = dst_tensor.device_type;
                } else {
                    device_type = src.device_type;
                }
            }
        }
        try device.dispatchOp(MemcpyImpl, device_type, device_id, out_tensor);
    }

    pub fn embedding(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(EmbeddingImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn rmsNorm(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(RmsNormImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn linear(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(LinearImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn reshape(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(ReshapeImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn permute(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(PermuteImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn applyRope(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(ApplyRopeImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn pagedAttention(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(PagedAttentionImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn flashAttention(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(FlashAttentionImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn silu(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(SiluImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn mul(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(MulImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn add(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(AddImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn matmul(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(MatmulImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn softmax(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(SoftmaxImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn ropeCache(out_tensor: *Tensor, device_id: i32) !void {
        // try device.dispatchOp(RopeCacheImpl, .CPU, device_id, out_tensor); // always use cpu
        try RopeCacheImpl(.CPU).execute(device_id, out_tensor); // always use cpu
    }

    pub fn gatherLast(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(GatherLastImpl, out_tensor.device_type, device_id, out_tensor);
    }

    pub fn sample(out_tensor: *Tensor, device_id: i32) !void {
        try device.dispatchOp(SampleImpl, out_tensor.device_type, device_id, out_tensor);
    }
};

const device = struct {
    // todo: - device id and device type save here

    pub fn dispatchOp(comptime Impl: fn (comptime DeviceType) type, device_type: DeviceType, device_id: i32, out_tensor: *Tensor) !void {
        switch (device_type) {
            .CPU => try Impl(.CPU).execute(device_id, out_tensor),
            .CUDA => {
                if (comptime opts.have_cuda) {
                    try Impl(.CUDA).execute(device_id, out_tensor);
                } else {
                    return error.UnsupportedDevice;
                }
            },
            else => return error.UnsupportedDevice,
        }
    }
};
