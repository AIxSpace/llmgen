const std = @import("std");

const Tensor = @import("../Tensor.zig");

const gguf = @import("../gguf.zig");
const TensorInfo = gguf.TensorInfo;

const executor = @import("../executor.zig");

const trace = @import("trace.zig");
const Trace = trace.Trace;

const base = @import("base.zig");
const module = @import("module.zig");
const Ctx = module.Ctx;

pub const Linear = struct {
    allocator: std.mem.Allocator,

    layer_id: i32,
    name: []const u8,
    weight_info: *const TensorInfo,

    const Self = @This();

    pub fn init(ctx: Ctx) !Self {
        const spec_entry = ctx.find_spec(ctx.spec_key);

        return .{
            .allocator = ctx.trace.node_allocator,
            .layer_id = ctx.layer_id,
            .name = try ctx.trace.name(spec_entry.tag, ctx.layer_id),
            .weight_info = try ctx.trace.weight(spec_entry.gguf, ctx.layer_id),
        };
    }

    pub fn forward(self: *Self, x: *Tensor) !*Tensor {
        return linear(
            self.allocator,
            x,
            self.weight_info,
            self.name,
            self.layer_id,
        );
    }
};

pub fn linear(
    allocator: std.mem.Allocator,
    input_tensor: *Tensor,
    weight_tensor_info: *const gguf.TensorInfo,
    name: []const u8,
    layer_id: i32,
) !*Tensor {
    const weight_tensor = try base.weightPlaceholder(
        allocator,
        weight_tensor_info,
        weight_tensor_info.name,
        layer_id,
    );
    const out_dims = try base.inferLinearOutput(
        input_tensor.dims,
        weight_tensor_info.dimensions,
        true,
    );
    const t = try base.newTensor(
        allocator,
        .{
            .name = name,
            .data_type = input_tensor.data_type,
            .layer_id = layer_id,
            .src = base.srcs(&.{ input_tensor, weight_tensor }),
            .tensor_type = .TENSOR_TYPE_ACTIVATION,
            .op_type = .OP_TYPE_LINEAR,
            .dims = out_dims,
        },
    );

    base.computeStrides(t);

    return t;
}

test "module-linear->ops-linear" {
    const allocator = std.testing.allocator;

    // 1. read test data

    const file = std.fs.cwd().openFile("testdata/linear_case.bin", .{}) catch |e| {
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
    const in_dim: usize = @intCast(std.mem.readInt(i32, buffer[4..8], .little));
    const out_dim: usize = @intCast(std.mem.readInt(i32, buffer[8..12], .little));

    std.debug.print("seq_len = {}, in_dim = {}, out_dim = {}\n", .{
        seq_len,
        in_dim,
        out_dim,
    });

    const x_num = seq_len * in_dim; // [seq_len, in_dim]
    const w_num = out_dim * in_dim; // [out_dim, in_dim]
    const y_num = seq_len * out_dim; // [seq_len, out_dim]

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
        .dims = .{ 1, @intCast(seq_len), @intCast(in_dim), 0 },
    };

    const weight_info: gguf.TensorInfo = .{
        .name = "test.weight",
        .dtype = .GGML_TYPE_F32,
        .offset = 0,
        .n_dims = 2,
        .dimensions = .{ @intCast(out_dim), @intCast(in_dim), 0, 0 },
    };

    // 4. module.forward

    var linear_module: Linear = .{
        .allocator = node_allocator,
        .layer_id = 0,
        .name = "test.linear",
        .weight_info = &weight_info,
    };
    const out_t = try linear_module.forward(&x_t);

    out_t.src[1].?.data = @ptrCast(w_data.ptr); // 补上权重占位的数据

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
            return error.LinearMismatch;
        }
    }

    std.debug.print("[module-linear] seq={d} in={d} out={d} out_t.dims={any} max_abs_diff={d}\n", .{
        seq_len,
        in_dim,
        out_dim,
        out_t.dims,
        max_diff,
    });
}
