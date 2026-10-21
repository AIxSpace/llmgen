const std = @import("std");
const types = @import("../types.zig");

const Tensor = @import("../Tensor.zig");

const gguf = @import("../gguf.zig");
const TensorInfo = gguf.TensorInfo;

const base = @import("base.zig");
const module = @import("module.zig");
const Ctx = module.Ctx;

pub const RmsNorm = struct {
    allocator: std.mem.Allocator,

    layer_id: i32,
    name: []const u8,
    weight_info: *const TensorInfo,

    rms_eps: f32,

    const Self = @This();

    pub fn init(ctx: Ctx) !Self {
        const spec_entry = ctx.find_spec(ctx.spec_key);

        return .{
            .allocator = ctx.trace.node_allocator,
            .layer_id = ctx.layer_id,
            .name = try ctx.trace.name(spec_entry.tag, ctx.layer_id),
            .weight_info = try ctx.trace.weight(spec_entry.gguf, ctx.layer_id),
            .rms_eps = ctx.config_params.rms_eps,
        };
    }

    pub fn forward(self: *Self, x: *Tensor) !*Tensor {
        return rmsNorm(
            self.allocator,
            x,
            self.weight_info,
            self.rms_eps,
            self.name,
            self.layer_id,
            0,
        );
    }
};

pub fn rmsNorm(
    allocator: std.mem.Allocator,
    input_tensor: *Tensor,
    weight_tensor_info: *const gguf.TensorInfo,
    eps: f32,
    name: []const u8,
    layer_id: i32,
    w_offset: f32,
) !*Tensor {
    const weight_tensor = try base.weightPlaceholder(
        allocator,
        weight_tensor_info,
        weight_tensor_info.name,
        layer_id,
    );

    var params = [_]f32{0} ** types.TENSOR_MAX_OP_PARAMS;
    params[0] = eps;
    params[1] = w_offset;

    const t = try base.newTensor(
        allocator,
        .{
            .name = name,
            .data_type = input_tensor.data_type,
            .layer_id = layer_id,
            .src = base.srcs(&.{ input_tensor, weight_tensor }),
            .tensor_type = .TENSOR_TYPE_ACTIVATION,
            .op_type = .OP_TYPE_RMS_NORM,
            .op_params = params,
            .dims = input_tensor.dims,
        },
    );

    base.computeStrides(t);

    return t;
}

const executor = @import("../executor.zig");

test "module-norm->ops-norm" {
    const allocator = std.testing.allocator;

    // 1. read test data

    const file = std.fs.cwd().openFile("testdata/norm_case.bin", .{}) catch |e| {
        std.debug.print("read data file error: {any}", .{e});
        return;
    };
    defer file.close();

    const file_size = (try file.stat()).size;
    const buffer = try allocator.alloc(u8, @intCast(file_size));
    defer allocator.free(buffer);

    {
        var offset: usize = 0;
        while (offset < buffer.len) {
            const n = try file.read(buffer[offset..]);
            if (n == 0) break;
            offset += n;
        }
    }

    // 2. parse test data

    const seq_len: usize = @intCast(std.mem.readInt(i32, buffer[0..4], .little));
    const hidden_size: usize = @intCast(std.mem.readInt(i32, buffer[4..8], .little));
    const eps: f32 = @bitCast(std.mem.readInt(u32, buffer[8..12], .little));

    std.debug.print("seq_len = {}, hidden_size = {}, eps = {}\n", .{ seq_len, hidden_size, eps });

    const x_num = seq_len * hidden_size; // x -> [seq_len, hidden_size]
    const w_num = hidden_size; // [hidden_size] (gain, 1 维)
    const y_num = seq_len * hidden_size; // [seq_len, hidden_size]

    const x_data = try allocator.alloc(f32, x_num);
    defer allocator.free(x_data);

    const w_data = try allocator.alloc(f32, w_num);
    defer allocator.free(w_data);

    const y_ref = try allocator.alloc(f32, y_num); // 对照数据
    defer allocator.free(y_ref);

    var start_pos: usize = 12;
    @memcpy(std.mem.sliceAsBytes(x_data), buffer[start_pos .. start_pos + x_num * 4]);
    start_pos += x_num * 4;
    @memcpy(std.mem.sliceAsBytes(w_data), buffer[start_pos .. start_pos + w_num * 4]);
    start_pos += w_num * 4;
    @memcpy(std.mem.sliceAsBytes(y_ref), buffer[start_pos .. start_pos + y_num * 4]);

    // 3. prepare module data

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const node_allocator = arena.allocator();

    var x_t: Tensor = .{
        .name = "x",
        .data = @ptrCast(x_data.ptr),
        .data_type = .GGML_TYPE_F32,
        .tensor_type = .TENSOR_TYPE_ACTIVATION,
        .dims = .{ @intCast(seq_len), @intCast(hidden_size), 0, 0 },
    };

    const weight_info: gguf.TensorInfo = .{
        .name = "test.weight",
        .dtype = .GGML_TYPE_F32,
        .offset = 0,
        .n_dims = 1,
        .dimensions = .{ @intCast(hidden_size), 0, 0, 0 },
    };

    // 4. module.forward

    var norm_module: RmsNorm = .{
        .allocator = node_allocator,
        .layer_id = 0,
        .name = "test.norm",
        .weight_info = &weight_info,
        .rms_eps = eps,
    };
    const out_t = try norm_module.forward(&x_t);

    out_t.src[1].?.data = @ptrCast(w_data.ptr); // 补上权重占位的数据 (gain)

    const y_got = try allocator.alloc(f32, y_num);
    defer allocator.free(y_got);
    out_t.data = @ptrCast(y_got.ptr);

    // 5. executor
    try executor.dispatchKernel(out_t, 0);

    // 6. compare
    var max_diff: f32 = 0;
    for (0..y_num) |i| {
        const distance = @abs(y_got[i] - y_ref[i]);
        if (distance > max_diff) max_diff = distance;
        const tol = 1e-3 + 1e-3 * @abs(y_ref[i]);
        if (distance > tol) {
            std.debug.print("mismatch@{d}: got={d} exp={d} diff={d}\n", .{
                i,
                y_got[i],
                y_ref[i],
                distance,
            });
            return error.NormMismatch;
        }
    }

    std.debug.print("[module-norm] seq={d} hidden_size={d} eps={d} max_abs_diff={d}\n", .{
        seq_len,
        hidden_size,
        eps,
        max_diff,
    });
}
