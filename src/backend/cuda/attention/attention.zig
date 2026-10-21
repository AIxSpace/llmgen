// 对照 src/backend/cuda/attention.cu。两者的 KV 都已由 Zig 侧 append_kv_from_tensor 写进分页块(显存),
// 页表(page_k_ids / page_v_ids)在 host, 这里 cudaMalloc + H2D 上传后发射 kernel(与 C++ PagedAttention 一致)。
//
//   paged_attention_kernel      与 C++ 逐字一致(一个 warp 一个 query, 逐页逐 token online softmax)。
//   flash_attention_warp_kernel 沿用 C++ 的结构(一个 warp 一个 query, KV 按 KV_TILE=32 搬进 shared memory,
//                               online softmax)。与 C++ 的差异: C++ 只对本步的 Q/K/V 计算(不读 cache), 本 lab
//                               与 cpu/attention.zig 的 FlashAttentionImpl 同语义 —— K/V 来自分页 cache 的全部历史,
//                               所以 tile 从分页块按逻辑位置搬, 因果判断带 q_start 偏移; 另外越界 warp 不提前 return
//                               (要参与 tile 搬运和 __syncthreads)。

const types = @import("../../../types.zig");

const Tensor = @import("../../../Tensor.zig");

const page_atten = @import("../../../memory/page_atten.zig");
const MemoryManager = @import("../../../memory/manager.zig");
const utils = @import("../utils.zig");

const Kind = enum {
    paged,
    flash,
};

pub extern fn cuda_paged_attention(
    out: ?*anyopaque,
    Q: ?*const anyopaque,
    dtype: c_int,
    head_dim: i32,
    num_q_heads: i32,
    seq_len: i32,
    num_kv_heads: i32,
    total_cached: i32,
    scale: f32,
    causal: c_int,
    num_kv_groups: i32,
    h_page_k_ids: [*]const i32,
    h_page_v_ids: [*]const i32,
    num_blocks: i32,
    k_pool_base: ?*const anyopaque,
    v_pool_base: ?*const anyopaque,
    block_stride_elems: i32,
    q_start: i32,
    dev_id: c_int,
) void;

pub extern fn cuda_flash_attention(
    out: ?*anyopaque,
    Q: ?*const anyopaque,
    dtype: c_int,
    head_dim: i32,
    num_q_heads: i32,
    seq_len: i32,
    num_kv_heads: i32,
    total_cached: i32,
    scale: f32,
    causal: c_int,
    num_kv_groups: i32,
    h_page_k_ids: [*]const i32,
    h_page_v_ids: [*]const i32,
    num_blocks: i32,
    k_pool_base: ?*const anyopaque,
    v_pool_base: ?*const anyopaque,
    block_stride_elems: i32,
    q_start: i32,
    dev_id: c_int,
) void;

fn attentionExecute(comptime kind: Kind, out: *Tensor, device_id: i32) !void {
    const mgr = (MemoryManager.global_mem_manager orelse return).getAttentionManager(out.device_type, @intCast(device_id)) orelse return;

    const layer_id = out.layer_id;
    if (!mgr.isActive(layer_id)) {
        return;
    }

    const Q = out.src[0].?; // [B, num_q_heads, seq_len, head_dim]
    const K = out.src[1].?; // [B, num_kv_heads, seq_len, head_dim]
    const V = out.src[2].?; // [B, num_kv_heads, seq_len, head_dim]

    const num_q_heads: i32 = @intCast(Q.dims[1]);
    const seq_len: i32 = @intCast(Q.dims[2]);
    const head_dim: i32 = @intCast(Q.dims[3]);

    const num_kv_heads: i32 = @intCast(K.dims[1]);
    const num_kv_groups: i32 = @intFromFloat(out.op_params[3]);

    const scale = out.op_params[1];
    const causal: c_int = if (out.op_params[2] != 0) 1 else 0;

    try mgr.appendKVFromTensor(layer_id, K.data.?[0..K.bytes()], V.data.?[0..V.bytes()], num_kv_heads, seq_len, head_dim, K.data_type);

    const layer = mgr.getLayerState(layer_id).?;
    const total_cached: i32 = layer.num_cached;
    const num_blocks: i32 = @intCast(layer.page_table.items.len);
    if (total_cached <= 0 or num_blocks <= 0) {
        return;
    }

    const block_stride_elems: i32 = page_atten.PAGE_BLOCK_SIZE * num_kv_heads * head_dim;

    const n: usize = @intCast(num_blocks);
    const h_page_k_ids = try mgr.allocator.alloc(i32, n);
    defer mgr.allocator.free(h_page_k_ids);
    const h_page_v_ids = try mgr.allocator.alloc(i32, n);
    defer mgr.allocator.free(h_page_v_ids);
    for (layer.page_table.items, 0..) |e, i| {
        h_page_k_ids[i] = e.k_block_id;
        h_page_v_ids[i] = e.v_block_id;
    }

    const k_pool_base = layer.k_pool.blockData(0).ptr;
    const v_pool_base = layer.v_pool.blockData(0).ptr;
    const q_start = total_cached - seq_len;

    const attention_kernel_launch = switch (kind) {
        .paged => cuda_paged_attention,
        .flash => cuda_flash_attention,
    };

    attention_kernel_launch(
        out.data,
        Q.data,
        utils.dataTypeEnumRawValue(out.data_type),
        head_dim,
        num_q_heads,
        seq_len,
        num_kv_heads,
        total_cached,
        scale,
        causal,
        num_kv_groups,
        h_page_k_ids.ptr,
        h_page_v_ids.ptr,
        num_blocks,
        k_pool_base,
        v_pool_base,
        block_stride_elems,
        q_start,
        device_id,
    );
}

pub const PagedAttentionImpl = struct {
    pub fn execute(device_id: i32, out_tensor: *Tensor) !void {
        try attentionExecute(.paged, out_tensor, device_id);
    }
};

pub const FlashAttentionImpl = struct {
    pub fn execute(device_id: i32, out_tensor: *Tensor) !void {
        try attentionExecute(.flash, out_tensor, device_id);
    }
};
