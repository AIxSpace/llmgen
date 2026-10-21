//! cuda/attention.zig -- PagedAttentionImpl<CUDA>
//! 先 cuda_store_kv 把本步 k / v 写进本层分页 cache, 再做注意力。
//! decode: split-K flash-decoding(第 13 步讲);prefill: tensor core FlashAttention-2(第 14 步讲)。
const std = @import("std");

const Tensor = @import("../../../Tensor.zig");

const common = @import("../../cpu/common.zig");

pub extern fn cuda_store_kv(k: ?*const anyopaque, v: ?*const anyopaque, slot_mapping: ?*const anyopaque, k_cache: ?*anyopaque, v_cache: ?*anyopaque, T: c_int, nkv: c_int, hd: c_int, block_size: c_int) void;
pub extern fn cuda_attn_decode(
    out: ?*anyopaque,
    q: ?*const anyopaque,
    k_cache: ?*const anyopaque,
    v_cache: ?*const anyopaque,
    block_tables: ?*const anyopaque,
    ctx_lens: ?*const anyopaque,
    S: c_int,
    nq: c_int,
    nkv: c_int,
    hd: c_int,
    max_blocks: c_int,
    block_size: c_int,
    scale: f32,
    part_o: ?*anyopaque,
    part_ml: ?*anyopaque,
    part_size: c_int,
    max_parts: c_int,
) void;
pub extern fn cuda_attn_prefill(
    out: ?*anyopaque,
    q: ?*const anyopaque,
    k_cache: ?*const anyopaque,
    v_cache: ?*const anyopaque,
    block_tables: ?*const anyopaque,
    ctx_lens: ?*const anyopaque,
    cu_q: ?*const anyopaque,
    tile_seq: ?*const anyopaque,
    tile_q0: ?*const anyopaque,
    n_tiles: c_int,
    nq: c_int,
    nkv: c_int,
    hd: c_int,
    max_blocks: c_int,
    block_size: c_int,
    scale: f32,
) void;
pub extern fn cuda_attn_prefill_tile() c_int;

/// decode split-K 的分段长度: 批越小分段越细, 保证 grid = nkv × S × parts 足够铺满 82 个 SM。
pub fn partSize(s: usize) usize {
    return if (s >= 64) 512 else if (s >= 16) 256 else if (s >= 4) 128 else 64;
}
pub fn maxParts(s: usize, max_model_len: usize) usize {
    const ps = partSize(s);
    return (max_model_len + ps - 1) / ps;
}

/// src[0] = q [1, T, nq, hd], src[1] = k [1, T, nkv, hd], src[2] = v [1, T, nkv, hd] → out [1, T, nq, hd]
pub const PagedAttentionImpl = struct {
    pub fn execute(_: i32, out: *Tensor) !void {
        const global_step = &common.g_step;
        const kv = try common.kvOf(out.layer_id);

        cuda_store_kv(
            out.src[1].?.data,
            out.src[2].?.data,
            global_step.slots,
            kv.k,
            kv.v,
            global_step.n_tok,
            global_step.n_kv_head,
            global_step.head_dim,
            global_step.block_size,
        );

        // decode 阶段
        if (global_step.decode) {
            const n_seq: usize = @intCast(global_step.n_seq);
            cuda_attn_decode(
                out.data,
                out.src[0].?.data,
                kv.k,
                kv.v,
                global_step.block_tables,
                global_step.ctx,
                global_step.n_seq,
                global_step.n_head,
                global_step.n_kv_head,
                global_step.head_dim,
                global_step.max_blocks,
                global_step.block_size,
                global_step.attn_scale,
                global_step.part_o,
                global_step.part_ml,
                @intCast(partSize(n_seq)),
                @intCast(maxParts(n_seq, global_step.max_model_len)),
            );
        } else {
            cuda_attn_prefill(
                out.data,
                out.src[0].?.data,
                kv.k,
                kv.v,
                global_step.block_tables,
                global_step.ctx,
                global_step.cu_q,
                global_step.tile_seq,
                global_step.tile_q0,
                global_step.n_tiles,
                global_step.n_head,
                global_step.n_kv_head,
                global_step.head_dim,
                global_step.max_blocks,
                global_step.block_size,
                global_step.attn_scale,
            );
        }
    }
};

/// 旧的 FLASH_ATTN 图节点(LLMGEN_FLASH=1)走同一个实现
pub const FlashAttentionImpl = PagedAttentionImpl;
