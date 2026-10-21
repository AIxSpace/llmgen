const std = @import("std");

const Tensor = @import("../Tensor.zig");

const gguf = @import("../gguf.zig");
const GGUFInfo = gguf.GGUFInfo;

const types = @import("../types.zig");

const ModelBase = @import("factory.zig").ModelBase;

const graph = @import("../graph/graph.zig");
const ComputeGraph = graph.ComputeGraph;

const module = @import("../module/module.zig");
const module_base = @import("../module/base.zig");
const SpecFn = module.SpecFn;
const Ctx = module.Ctx;
const Module = module.Module;
const Trace = module.Trace;
const SpectEntry = module.SpecEntry;
const Embedding = module.Embedding;
const Linear = module.Linear;
const RmsNorm = module.RmsNorm;
const Attention = module.Attention;
const ResidualAdd = module.ResidualAdd;
const Mul = module.Mul;
const Silu = module.Silu;
const RopeCache = module.RopeCache;
const Rope = module.Rope;
const HeadSplit = module.HeadSplit;
const serving = @import("../module/serving.zig");

const STATE_DCIT = std.StaticStringMap(SpectEntry).initComptime(.{
    // 模型进出
    .{ "embed", SpectEntry{ .gguf = "token_embd.weight", .tag = "embd" } },
    .{ "final_norm", SpectEntry{ .gguf = "output_norm.weight", .tag = "final_norm" } },
    .{ "lm_head", SpectEntry{ .gguf = "token_embd.weight", .tag = "logits" } },

    // 注意力
    .{ "norm", SpectEntry{ .gguf = "attn_norm.weight", .tag = "attn_norm" } },
    .{ "wq", SpectEntry{ .gguf = "attn_q.weight", .tag = "q" } },
    .{ "wk", SpectEntry{ .gguf = "attn_k.weight", .tag = "k" } },
    .{ "wv", SpectEntry{ .gguf = "attn_v.weight", .tag = "v" } },
    .{ "wo", SpectEntry{ .gguf = "attn_output.weight", .tag = "o" } },
    .{ "attn_q_norm", SpectEntry{ .gguf = "attn_q_norm.weight", .tag = "q_norm" } },
    .{ "attn_k_norm", SpectEntry{ .gguf = "attn_k_norm.weight", .tag = "k_norm" } },

    // FFN
    .{ "ffn_norm", SpectEntry{ .gguf = "ffn_norm.weight", .tag = "ffn_norm" } },
    .{ "w_gate", SpectEntry{ .gguf = "ffn_gate.weight", .tag = "gate" } },
    .{ "w_up", SpectEntry{ .gguf = "ffn_up.weight", .tag = "up" } },
    .{ "w_down", SpectEntry{ .gguf = "ffn_down.weight", .tag = "down" } },

    // 无权重算子
    .{ "attn_res", SpectEntry{ .gguf = "", .tag = "attn_res" } },
    .{ "ffn_res", SpectEntry{ .gguf = "", .tag = "ffn_res" } },

    .{ "q_hp", SpectEntry{ .gguf = "", .tag = "q_hp" } },
    .{ "k_hp", SpectEntry{ .gguf = "", .tag = "k_hp" } },
    .{ "v_hp", SpectEntry{ .gguf = "", .tag = "v_hp" } },

    .{ "rope_cache", SpectEntry{ .gguf = "", .tag = "rope" } },

    .{ "silu", SpectEntry{ .gguf = "", .tag = "silu" } },
    .{ "swiglu", SpectEntry{ .gguf = "", .tag = "swiglu" } },
    .{ "act", SpectEntry{ .gguf = "", .tag = "act" } }, // DenseMlp
});

pub fn spec(name: []const u8) SpectEntry {
    return STATE_DCIT.get(name) orelse std.debug.panic("STATE_DICT has no spec mapping: {s}", .{name});
}

pub const Config = struct {
    block_count: i64, // 层数
    embedding_length: i64,
    head_count: i64, // Q head count
    head_count_kv: i64, // KV head count
    key_length: i64, // head_dim
    value_length: i64,
    feed_forward_length: i64, // ffn 中间维
    context_length: i64,
    vocab_size: i64,
    rms_eps: f32,
    rope_freq_base: f32,

    const Self = @This();

    pub fn qHeadsDim(self: Self) i64 {
        return self.head_count * self.key_length; // Q 投影输出维
    }

    pub fn kvHeadsDim(self: Self) i64 {
        return self.head_count_kv * self.key_length; // KV 投影输出维
    }

    pub fn fromGGUF(gguf_info: *const gguf.GGUFInfo) !Self {
        const model_arch = gguf_info.getModelArchitecture();
        std.debug.print("the model arch = {s}\n", .{model_arch});

        const getIntValue = struct {
            fn func(info: *const gguf.GGUFInfo, arch: []const u8, name: []const u8, default: i64) i64 {
                var buf: [64]u8 = undefined;
                const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, name }) catch return default;
                return info.metadata.getInt(key, default);
            }
        }.func;

        const getFloatValue = struct {
            fn func(info: *const gguf.GGUFInfo, arch: []const u8, name: []const u8, default: f32) f32 {
                var buf: [64]u8 = undefined;
                const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, name }) catch return default;
                return info.metadata.getFloat(key, default);
            }
        }.func;

        var vocab_size = getIntValue(gguf_info, model_arch, "vocab_size", 0);
        if (vocab_size == 0) {
            if (gguf_info.metadata.get("tokenizer.ggml.tokens")) |metadata_value| {
                if (metadata_value == .array) {
                    vocab_size = @intCast(metadata_value.array.items.len);
                }
            }
        }

        return .{
            .block_count = getIntValue(gguf_info, model_arch, "block_count", 0),
            .embedding_length = getIntValue(gguf_info, model_arch, "embedding_length", 0),
            .head_count = getIntValue(gguf_info, model_arch, "attention.head_count", 0),
            .head_count_kv = getIntValue(gguf_info, model_arch, "attention.head_count_kv", 0),
            .key_length = getIntValue(gguf_info, model_arch, "attention.key_length", 0),
            .value_length = getIntValue(gguf_info, model_arch, "attention.value_length", 0),
            .feed_forward_length = getIntValue(gguf_info, model_arch, "feed_forward_length", 0),
            .context_length = getIntValue(gguf_info, model_arch, "context_length", 0),
            .vocab_size = vocab_size,
            .rms_eps = getFloatValue(gguf_info, model_arch, "attention.layer_norm_rms_epsilon", 1e-6),
            .rope_freq_base = getFloatValue(gguf_info, model_arch, "rope.rope_freq_base", 10000),
        };
    }
};

const Qwen3MLP = struct {
    const Self = @This();

    w_gate: Linear,
    w_up: Linear,
    silu: Silu,
    swiglu: Mul,
    w_down: Linear,

    pub const init = Module(Self).init;

    pub fn forward(self: *Self, x: *Tensor) !*Tensor {
        const gate = try self.w_gate.forward(x);

        const up = try self.w_up.forward(x);

        const act = try self.silu.forward(gate);
        const glu = try self.swiglu.forward(act, up);

        return self.w_down.forward(glu);
    }
};

const Qwen3Attention = struct {
    norm: RmsNorm,

    wq: Linear,
    wk: Linear,
    wv: Linear,

    q_hp: HeadSplit(.query),
    k_hp: HeadSplit(.kv),
    v_hp: HeadSplit(.kv),

    attn_q_norm: RmsNorm,
    attn_k_norm: RmsNorm,

    rope: Rope, // 2 个

    attention: Attention, // 2 个

    wo: Linear,

    const Self = @This();

    pub const init = Module(Self).init;

    fn forward(self: *Self, x: *Tensor) !*Tensor {
        const x_norm = try self.norm.forward(x);

        const q0 = try self.wq.forward(x_norm);
        const k0 = try self.wk.forward(x_norm);
        const v0 = try self.wv.forward(x_norm);

        const q1 = try self.q_hp.forward(q0);
        const k1 = try self.k_hp.forward(k0);
        const v1 = try self.v_hp.forward(v0);

        const q2 = try self.attn_q_norm.forward(q1);
        const k2 = try self.attn_k_norm.forward(k1);

        const qk = try self.rope.forward(q2, k2);

        const attn_score = try self.attention.forward(qk.q, qk.k, v1);

        return self.wo.forward(attn_score);
    }
};

const Qwen3DecoderLayer = struct {
    qwen3_attn: Qwen3Attention, // 13
    attn_res: ResidualAdd, // 1
    ffn_norm: RmsNorm, // 1
    mlp: Qwen3MLP, // 5
    ffn_res: ResidualAdd, // 1

    config: Config,

    const Self = @This();

    pub const init = Module(Self).init;

    fn forward(self: *Self, x_in: *Tensor) !*Tensor {
        const a = try self.qwen3_attn.forward(x_in);

        var x = try self.attn_res.forward(x_in, a);

        const fnorm = try self.ffn_norm.forward(x);

        const m = try self.mlp.forward(fnorm);

        x = try self.ffn_res.forward(x, m);

        return x;
    }
};

const Qwen3Model = struct {
    allocator: std.mem.Allocator,

    config: Config,
    trace: Trace,

    rope_cache: RopeCache,
    embed: Embedding,
    layers: std.ArrayList(Qwen3DecoderLayer) = .empty,
    final_norm: RmsNorm,
    lm_head: Linear,

    const Self = @This();

    fn init(
        allocator: std.mem.Allocator,
        node_allocator: std.mem.Allocator,
        gguf_info: *const GGUFInfo,
        config: Config,
    ) !Self {
        var self: Self = .{
            .allocator = allocator,
            .config = config,
            .trace = Trace.init(allocator, node_allocator, gguf_info),
            .rope_cache = undefined,
            .embed = undefined,
            .final_norm = undefined,
            .lm_head = undefined,
        };
        errdefer self.deinit();

        var ctx: Ctx = .{
            .layer_id = -1,
            .gguf_info = gguf_info,
            .trace = &self.trace,
            .find_spec = &spec,
            .config_params = .{
                .n_head = config.head_count,
                .head_count_kv = config.head_count_kv,
                .head_dim = config.key_length,
                .rms_eps = config.rms_eps,
                .rope_freq_base = config.rope_freq_base,
                .context_length = config.context_length,
            },
        };

        if (std.posix.getenv("LLMGEN_FLASH") != null) ctx.attn_op = .OP_TYPE_FLASH_ATTN;

        self.rope_cache = try RopeCache.init(ctx.useSpec("rope_cache"));
        ctx.rope = try self.rope_cache.forward();

        self.embed = try Embedding.init(ctx.useSpec("embed"));
        self.final_norm = try RmsNorm.init(ctx.useSpec("final_norm"));
        self.lm_head = try Linear.init(ctx.useSpec("lm_head"));

        // final_norm / logits 的 layer_id = block_count(跟随最后一层上 GPU, 不是全局节点)
        self.final_norm.layer_id = @intCast(config.block_count);
        self.lm_head.layer_id = @intCast(config.block_count);

        var layer_id: i32 = 0;
        while (layer_id < config.block_count) : (layer_id += 1) {
            var decode_layer = try Qwen3DecoderLayer.init(ctx.atLayer(@intCast(layer_id)));
            decode_layer.config = self.config;
            try self.layers.append(allocator, decode_layer);
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.layers.deinit(self.allocator);
        self.trace.deinit();
    }

    // input_ids, embedding, final_norm, lm_head 未分配 layer，默认在 CPU 执行
    fn forward(self: *Self, input_ids: *Tensor) !*Tensor {
        var x = try self.embed.forward(input_ids);

        for (self.layers.items) |*qwen3_layer| {
            x = try qwen3_layer.forward(x);
        }

        const fx = try self.final_norm.forward(x);

        // 只有每条序列最后一个 token 的隐状态要过 lm_head
        const last = try serving.gatherLast(
            self.trace.node_allocator,
            fx,
            "gather_last",
            self.final_norm.layer_id,
        );
        const logits = try self.lm_head.forward(last);

        // 图的输出从 logits 变成采样出的 token id
        return serving.sample(
            self.trace.node_allocator,
            logits,
            "sample",
            self.final_norm.layer_id,
        );
    }
};

pub const Model = struct {
    allocator: std.mem.Allocator,

    config: Config = undefined,

    names: std.ArrayList([]u8) = .empty,

    compute_graph: ?*ComputeGraph = null,

    qwen3_model: ?*Qwen3Model = null,

    const Self = @This();

    const vtable = ModelBase.VTable{
        .printInfo = vPrintInfo,
        .chatTemplate = vChatTemplate,
        .vocabSize = vVocabSize,
        .maxSeqLen = vMaxSeqLen,
        .buildGraph = vBuildGraph,
        .deinit = vDeinit,
    };

    pub fn vPrintInfo(ptr: *anyopaque) void {
        return cast(ptr).printInfo();
    }

    pub fn vChatTemplate(ptr: *anyopaque, allocator: std.mem.Allocator, prompt: []const u8, think: bool) anyerror![]u8 {
        return cast(ptr).chatTemplate(allocator, prompt, think);
    }

    pub fn vVocabSize(ptr: *anyopaque) usize {
        return cast(ptr).vocabSize();
    }

    pub fn vMaxSeqLen(ptr: *anyopaque) i64 {
        return cast(ptr).maxSeqLen();
    }

    pub fn vBuildGraph(ptr: *anyopaque, gguf_info: *const gguf.GGUFInfo) anyerror!*ComputeGraph {
        return cast(ptr).buildGraph(gguf_info);
    }

    pub fn vDeinit(ptr: *anyopaque) void {
        const self = cast(ptr);
        const allocator = self.allocator;
        self.deinit();
        allocator.destroy(self);
    }

    inline fn cast(ptr: *anyopaque) *Self {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn create(allocator: std.mem.Allocator) !ModelBase {
        const self = try allocator.create(Model);
        self.* = .{
            .allocator = allocator,
        };
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    pub fn printInfo(self: *Self) void {
        const cfg = self.config;
        std.debug.print("qwen3 model: layers={d}, n_embd={d}, heads = {d}/{d}(GQA), head_dim={d}, ffn={d}, vocab={d}, eps={e}, rope={d}\n", .{
            cfg.block_count,
            cfg.embedding_length,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.key_length,
            cfg.feed_forward_length,
            cfg.vocab_size,
            cfg.rms_eps,
            cfg.rope_freq_base,
        });
    }

    pub fn chatTemplate(_: *Self, allocator: std.mem.Allocator, prompt: []const u8, think: bool) ![]u8 {
        if (think) {
            return std.fmt.allocPrint(allocator, "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n", .{prompt});
        }
        return std.fmt.allocPrint(allocator, "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", .{prompt});
    }

    pub fn vocabSize(self: *Self) usize {
        return @intCast(self.config.vocab_size);
    }

    pub fn maxSeqLen(self: *Self) i64 {
        return self.config.context_length;
    }

    pub fn buildGraph(self: *Self, gguf_info: *const gguf.GGUFInfo) !*ComputeGraph {
        try self.parseConfig(gguf_info);

        const compute_graph = try self.allocator.create(ComputeGraph);
        errdefer self.allocator.destroy(compute_graph);

        compute_graph.* = ComputeGraph.init(self.allocator);
        errdefer compute_graph.deinit();
        const node_allocator = compute_graph.nodeAllocator();

        const qwen3_model = try self.allocator.create(Qwen3Model);
        errdefer {
            if (self.qwen3_model == null) {
                qwen3_model.deinit();
                self.allocator.destroy(qwen3_model);
            }
        }
        qwen3_model.* = try Qwen3Model.init(
            self.allocator,
            node_allocator,
            gguf_info,
            self.config,
        );

        const input_ids = try module_base.placeholder(
            node_allocator,
            "input_ids",
            -1,
            .TENSOR_TYPE_INPUT,
            .GGML_TYPE_I32,
            .{ 1, -1, 0, 0 },
        );
        const logits = try qwen3_model.forward(input_ids);

        // const logits = try self.buildLayers(node_allocator, gguf_info);

        try compute_graph.buildFromOutputs(&.{logits});

        self.compute_graph = compute_graph;
        self.qwen3_model = qwen3_model;

        return compute_graph;
    }

    pub fn deinit(self: *Self) void {
        if (self.qwen3_model) |qwen3_model| {
            qwen3_model.deinit();
            self.allocator.destroy(qwen3_model);
        }

        if (self.compute_graph) |compute_graph| {
            compute_graph.deinit();
            self.allocator.destroy(compute_graph);
        }

        for (self.names.items) |n| {
            self.allocator.free(n);
        }

        self.names.deinit(self.allocator);
    }

    fn buildLayers(self: *Self, node_allocator: std.mem.Allocator, gguf_info: *const gguf.GGUFInfo) !*Tensor {
        const cfg = self.config;

        const input_ids = try module.placeholder(node_allocator, "input_ids", -1, .TENSOR_TYPE_INPUT, .GGML_TYPE_I32, .{ -1, 0, 0, 0 });

        const token_embedding_weight_tensor_info = try self.getTensorInfo(gguf_info, "token_embd.weight", -1);
        var x = try module.embeddingLookup(node_allocator, input_ids, token_embedding_weight_tensor_info, false, "embd", -1, .GGML_TYPE_F32);

        var layer_idx: i32 = 0;
        while (layer_idx < cfg.block_count) : (layer_idx += 1) {
            x = try self.buildQwen3Layer(node_allocator, gguf_info, x, layer_idx);
        }

        const output_weight_tensor_info = try self.getTensorInfo(gguf_info, "output_norm.weight", -1);
        const fx = try module.rmsNorm(node_allocator, x, output_weight_tensor_info, cfg.rms_eps, "final_norm", -1, 0);

        const logits = try module.linear(
            node_allocator,
            fx,
            token_embedding_weight_tensor_info,
            "logits",
            -1,
        );
        logits.tensor_type = .TENSOR_TYPE_OUTPUT;

        return logits;
    }

    pub fn buildQwen3Layer(self: *Self, node_allocator: std.mem.Allocator, gguf_info: *const gguf.GGUFInfo, x_in: *Tensor, layer_idx: i32) !*Tensor {
        const cfg = self.config;
        const rms_eps = cfg.rms_eps;

        const attn_norm_weight_tensor_info = try self.getTensorInfo(gguf_info, "blk.{d}.attn_norm.weight", layer_idx);
        const x_norm = try module.rmsNorm(node_allocator, x_in, attn_norm_weight_tensor_info, rms_eps, try self.nameOfLayer("l{d}.attn_norm", layer_idx), layer_idx, 0);

        const w_q_weight_tensor_info = try self.getTensorInfo(gguf_info, "blk.{d}.attn_q.weight", layer_idx);
        const w_k_weight_tensor_info = try self.getTensorInfo(gguf_info, "blk.{d}.attn_k.weight", layer_idx);
        const w_v_weight_tensor_info = try self.getTensorInfo(gguf_info, "blk.{d}.attn_v.weight", layer_idx);

        const q_flat = try module.linear(
            node_allocator,
            x_norm,
            w_q_weight_tensor_info,
            try self.nameOfLayer("l{d}.q", layer_idx),
            layer_idx,
        );
        const k_flat = try module.linear(
            node_allocator,
            x_norm,
            w_k_weight_tensor_info,
            try self.nameOfLayer("l{d}.k", layer_idx),
            layer_idx,
        );
        const v_flat = try module.linear(
            node_allocator,
            x_norm,
            w_v_weight_tensor_info,
            try self.nameOfLayer("l{d}.v", layer_idx),
            layer_idx,
        );

        const q_norm_weight_tensor_info = try self.getTensorInfo(gguf_info, "blk.{d}.attn_q_norm.weight", layer_idx);
        const k_norm_weight_tensor_info = try self.getTensorInfo(gguf_info, "blk.{d}.attn_k_norm.weight", layer_idx);

        const q = try module.rmsNorm(node_allocator, q_flat, q_norm_weight_tensor_info, rms_eps, try self.nameOfLayer("l{d}.q_norm", layer_idx), layer_idx, 0);
        const k = try module.rmsNorm(node_allocator, k_flat, k_norm_weight_tensor_info, rms_eps, try self.nameOfLayer("l{d}.k_norm", layer_idx), layer_idx, 0);

        const attn = try module.attention(
            node_allocator,
            q,
            k,
            v_flat,
            null,
            -1,
            true,
            0,
            try self.nameOfLayer("l{d}.attn", layer_idx),
            layer_idx,
            .OP_TYPE_PAGED_ATTN,
        );

        const attn_out_weight_tensor_info = try self.getTensorInfo(gguf_info, "blk.{d}.attn_output.weight", layer_idx);
        const o = try module.linear(
            node_allocator,
            attn,
            attn_out_weight_tensor_info,
            try self.nameOfLayer("l{d}.attn_res", layer_idx),
            layer_idx,
        );

        var x = try module.add(node_allocator, x_in, o, try self.nameOfLayer("l{d}.attn_res", layer_idx), layer_idx);

        // -- FFN (SwiGLU) --
        const ffn_norm_weight_tensor_info = try self.getTensorInfo(gguf_info, "blk.{d}.ffn_norm.weight", layer_idx);
        const fn_ = try module.rmsNorm(node_allocator, x, ffn_norm_weight_tensor_info, rms_eps, try self.nameOfLayer("l{d}.ffn_norm", layer_idx), layer_idx, 0);

        const ffn_gate_weight_tensor_info = try self.getTensorInfo(gguf_info, "blk.{d}.ffn_gate.weight", layer_idx);
        const gate = try module.linear(
            node_allocator,
            fn_,
            ffn_gate_weight_tensor_info,
            try self.nameOfLayer("l{d}.gate", layer_idx),
            layer_idx,
        );

        const ffn_up_weight_tensor_info = try self.getTensorInfo(gguf_info, "blk.{d}.ffn_up.weight", layer_idx);
        const up = try module.linear(
            node_allocator,
            fn_,
            ffn_up_weight_tensor_info,
            try self.nameOfLayer("l{d}.up", layer_idx),
            layer_idx,
        );

        const act = try module.silu(
            node_allocator,
            gate,
            try self.nameOfLayer("l{d}.silu", layer_idx),
            layer_idx,
        );
        const swiglu = try module.mul(node_allocator, act, up, try self.nameOfLayer("l{d}.swiglu", layer_idx), layer_idx);

        const ffn_down_weight_tensor_info = try self.getTensorInfo(gguf_info, "blk.{d}.ffn_down.weight", layer_idx);
        const down = try module.linear(
            node_allocator,
            swiglu,
            ffn_down_weight_tensor_info,
            try self.nameOfLayer("l{d}.down", layer_idx),
            layer_idx,
        );

        x = try module.add(node_allocator, x, down, try self.nameOfLayer("l{d}.ffn_res", layer_idx), layer_idx);
        return x;
    }

    fn getTensorInfo(self: *Self, gguf_info: *const gguf.GGUFInfo, comptime fmt: []const u8, layer_idx: i32) !*const gguf.TensorInfo {
        const has_layer_idx = comptime std.mem.indexOfScalar(u8, fmt, '{') != null;
        const weight_name: []const u8 = if (has_layer_idx)
            try self.weightNameOfLayer(fmt, layer_idx)
        else
            fmt;

        return gguf_info.findTensor(weight_name) orelse error.MissingWeight;
    }

    fn weightNameOfLayer(self: *Self, comptime fmt: []const u8, layer_idx: i32) ![]const u8 {
        return try self.nameOfLayer(fmt, layer_idx);
    }

    fn nameOfLayer(self: *Self, comptime fmt: []const u8, layer_idx: i32) ![]const u8 {
        const name_str = try std.fmt.allocPrint(self.allocator, fmt, .{layer_idx});
        try self.names.append(self.allocator, name_str);
        return name_str;
    }

    pub fn parseConfig(self: *Self, gguf_info: *const gguf.GGUFInfo) !void {
        const cfg = try Config.fromGGUF(gguf_info);
        self.config = cfg;
        self.printInfo();
    }
};
