const std = @import("std");
const common = @import("common.zig");

const types = @import("../../types.zig");
const Tensor = @import("../../Tensor.zig");

pub const EmbeddingImpl = struct {
    pub fn execute(_: i32, out_tensor: *Tensor) !void {
        const token_ids = out_tensor.src[0].?;
        const weight = out_tensor.src[1].?;

        // [vocab, hidden]
        const vocab_size = common.dimAt(weight, 0);
        const hidden = common.dimAt(weight, 1);

        const seq_len = common.dimAt(out_tensor, 1);

        const token_ids_array = common.i32DataPtr(token_ids);

        for (0..seq_len) |s| {
            const id = token_ids_array[s];
            if (id < 0 or id >= @as(i32, @intCast(vocab_size))) return error.TokenOutOfRange;
            const row: usize = @as(usize, @intCast(id)) * hidden;
            for (0..hidden) |h| {
                common.storeF32(out_tensor, s * hidden + h, common.elemF32(weight, row + h));
            }
        }
    }
};
