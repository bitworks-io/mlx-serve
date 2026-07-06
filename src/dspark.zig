//! DeepSeek-AI DSpark speculative-decoding drafter — Qwen3 target.
//!
//! Port of `deepspec/modeling/dspark/qwen3/{modeling,config}.py` +
//! `deepspec/modeling/dspark/markov_head.py` +
//! `deepspec/modeling/dspark/common.py` (`AcceptRatePredictor`) from
//! DeepSeek-AI's DeepSpec repo (https://github.com/deepseek-ai/DeepSpec).
//! Full port spec: see DSPARK_PORT_SPEC.md (not checked into this repo).
//!
//! Architecture summary (see the spec for exact citations):
//!   - 5-layer, NON-autoregressive, cross-attention-fused backbone. One
//!     forward proposes ALL `block_size` (7) draft tokens in parallel via
//!     mask-token infill — fundamentally unlike drafter.zig/mtp.zig's
//!     autoregressive chains.
//!   - Per-layer `Qwen3DSparkAttention`: Q from the draft/noise stream only;
//!     K/V computed TWICE (once from the fused target-hidden context, once
//!     from the draft block itself) and CONCATENATED along the sequence
//!     axis before QK-norm + RoPE. Attention is fully non-causal (no mask)
//!     over the concatenated ctx+block span.
//!   - `target_hidden_states` = concat of 5 RAW (pre-final-norm) target
//!     decoder-layer outputs at `target_layer_ids`, projected via `fc` +
//!     `hidden_norm` ONCE before the layer loop (all 5 backbone layers
//!     reuse the same projected tensor as K/V source).
//!   - Markov head (`VanillaMarkov`, this checkpoint's only variant):
//!     rank-256 bigram correction, `bias = markov_w2 @ markov_w1[prev_tok]`,
//!     applied sequentially over the 7 already-computed base-logit rows
//!     (cheap — no transformer forward).
//!   - Confidence head: single linear over
//!     `concat(backbone_hidden, markov_w1[prev_tok])` → sigmoid
//!     accept-probability estimate, used to truncate the proposal length.
//!
//! MILESTONE 1 (this file, current scope): config parse, full weight
//! loading (all 64 tensors — embedding/output, fusion/projection, 5
//! backbone decoder layers, Markov head, confidence head), target-binding
//! validation, and the single-round BACKBONE FORWARD (`proposeBlock`) that
//! turns `(anchor_token, target_hidden_taps)` into base logits + Markov-
//! corrected sampled tokens + confidence-gated proposal length. All tensors
//! in this checkpoint are dense bf16 (no quantization) — no QLinear/inferBits
//! machinery needed, unlike mtp.zig's sidecar.
//!
//! MILESTONE 2 (explicitly OUT of scope here — see `// MILESTONE 2:` marks
//! below): the persistent-across-rounds draft KV cache lifecycle (snapshot/
//! restore in lockstep with the target's cache), the Leviathan verify/accept
//! integration into generate.zig (`Generator.nextDspark`), the NEW target-side
//! multi-layer raw-hidden-state capture point in transformer.zig
//! (`capture_hidden_layers`), scheduler dispatch, the `--dspark` CLI flag,
//! and the four-HTTP-surface per-request `enable_dspark`/`confidence_threshold`
//! plumbing.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
const log = @import("log.zig");

const Weights = model_mod.Weights;
const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;

/// Draft-block size (max tokens proposed per round). Checkpoint-specific
/// (`config.json`'s `block_size`); this port doesn't hardcode 7.
pub const DEFAULT_BLOCK_SIZE: u32 = 7;

/// Only `VanillaMarkov` is implemented (this checkpoint's only variant —
/// no `gate_proj`/`joint_proj` tensors in the safetensors header). A future
/// checkpoint with `markov_head_type: "gated"` or `"rnn"` must fail loudly
/// rather than silently mis-load.
pub const MarkovHeadType = enum {
    vanilla,

    pub fn fromString(s: []const u8) !MarkovHeadType {
        if (std.mem.eql(u8, s, "vanilla")) return .vanilla;
        return error.UnsupportedMarkovHeadType;
    }
};

/// Draft config — a checkpoint-specific deep-copy-plus-overrides of the
/// target's own HF config (`qwen3/config.py:9-56`, `build_draft_config`).
/// Every field here is read from the drafter's OWN `config.json` at load
/// time (never hardcoded), since a different DSpark checkpoint (e.g. a
/// 4B/14B target pairing) has different `hidden_size`/`vocab_size`/
/// `target_layer_ids`/etc.
pub const DsparkConfig = struct {
    hidden_size: u32,
    vocab_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    block_size: u32,
    mask_token_id: u32,
    /// 0-indexed target decoder-layer indices tapped for the fused context
    /// (owned; e.g. `[1, 9, 17, 25, 33]`). Must never include the target's
    /// LAST layer (its `output_hidden_states` slot is post-final-norm, not
    /// a raw per-layer output — mixing the two is a silent train/inference
    /// mismatch per the reference's own `assert_no_final_target_layer`).
    target_layer_ids: []u32,
    num_target_layers: u32,
    markov_rank: u32,
    markov_head_type: MarkovHeadType,
    confidence_head_with_markov: bool,
    enable_confidence_head: bool,

    pub fn deinit(self: *DsparkConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.target_layer_ids);
    }

    /// Number of tapped target layers (`target_layer_ids.len`) — the
    /// multiplier on `hidden_size` for the `fc` projection's input dim
    /// (`fc.weight [hidden, num_taps*hidden]`).
    pub fn numTaps(self: *const DsparkConfig) u32 {
        return @intCast(self.target_layer_ids.len);
    }

    /// Confidence-head input dim: `hidden_size` alone, or
    /// `hidden_size + markov_rank` when `confidence_head_with_markov`.
    pub fn confidenceInputDim(self: *const DsparkConfig) u32 {
        if (self.confidence_head_with_markov) return self.hidden_size + self.markov_rank;
        return self.hidden_size;
    }

    /// Validate `target_layer_ids` are in-range for a target with
    /// `target_num_layers` decoder layers and never include the last one
    /// (`base_evaluator.py:100-112`, `assert_no_final_target_layer`).
    pub fn validateTargetLayerIds(self: *const DsparkConfig, target_num_layers: u32) !void {
        if (target_num_layers == 0) return error.DsparkTargetMismatch;
        const last: u32 = target_num_layers - 1;
        for (self.target_layer_ids) |lid| {
            if (lid >= target_num_layers) return error.DsparkTargetMismatch;
            if (lid == last) return error.DsparkTargetMismatch;
        }
    }
};

/// Parse a `DsparkConfig` from a `config.json`-shaped `std.json.Value`
/// object (or a synthetic one in tests). Missing required numeric fields
/// error rather than silently defaulting — a DSpark checkpoint's shape is
/// load-bearing (byte-identical vocab/hidden alignment with the target),
/// unlike optional multimodal `text_config` fields elsewhere in the codebase.
pub fn parseConfig(allocator: std.mem.Allocator, root: std.json.Value) !DsparkConfig {
    const obj = switch (root) {
        .object => |o| o,
        else => return error.InvalidDsparkConfig,
    };

    const getU32 = struct {
        fn call(o: std.json.ObjectMap, key: []const u8) !u32 {
            const v = o.get(key) orelse return error.MissingDsparkConfigField;
            return switch (v) {
                .integer => |i| @intCast(i),
                .float => |f| @intFromFloat(f),
                else => return error.InvalidDsparkConfig,
            };
        }
    }.call;

    const getF32 = struct {
        fn call(o: std.json.ObjectMap, key: []const u8, default: f32) f32 {
            const v = o.get(key) orelse return default;
            return switch (v) {
                .integer => |i| @floatFromInt(i),
                .float => |f| @floatCast(f),
                else => default,
            };
        }
    }.call;

    const getBool = struct {
        fn call(o: std.json.ObjectMap, key: []const u8, default: bool) bool {
            const v = o.get(key) orelse return default;
            return switch (v) {
                .bool => |b| b,
                else => default,
            };
        }
    }.call;

    var rope_theta: f32 = 1_000_000.0;
    if (obj.get("rope_parameters")) |rp| {
        if (rp == .object) {
            if (rp.object.get("rope_theta")) |rt| {
                rope_theta = switch (rt) {
                    .integer => |i| @floatFromInt(i),
                    .float => |f| @floatCast(f),
                    else => rope_theta,
                };
            }
        }
    } else {
        rope_theta = getF32(obj, "rope_theta", rope_theta);
    }

    const target_layer_ids_val = obj.get("target_layer_ids") orelse return error.MissingDsparkConfigField;
    if (target_layer_ids_val != .array) return error.InvalidDsparkConfig;
    const src_ids = target_layer_ids_val.array.items;
    var ids = try allocator.alloc(u32, src_ids.len);
    errdefer allocator.free(ids);
    for (src_ids, 0..) |v, i| {
        ids[i] = switch (v) {
            .integer => |n| @intCast(n),
            .float => |f| @intFromFloat(f),
            else => return error.InvalidDsparkConfig,
        };
    }

    const markov_head_type_str = blk: {
        const v = obj.get("markov_head_type") orelse break :blk "vanilla";
        break :blk switch (v) {
            .string => |s| s,
            else => "vanilla",
        };
    };

    return .{
        .hidden_size = try getU32(obj, "hidden_size"),
        .vocab_size = try getU32(obj, "vocab_size"),
        .num_hidden_layers = try getU32(obj, "num_hidden_layers"),
        .num_attention_heads = try getU32(obj, "num_attention_heads"),
        .num_key_value_heads = try getU32(obj, "num_key_value_heads"),
        .head_dim = try getU32(obj, "head_dim"),
        .intermediate_size = try getU32(obj, "intermediate_size"),
        .rms_norm_eps = getF32(obj, "rms_norm_eps", 1e-6),
        .rope_theta = rope_theta,
        .block_size = getU32(obj, "block_size") catch DEFAULT_BLOCK_SIZE,
        .mask_token_id = try getU32(obj, "mask_token_id"),
        .target_layer_ids = ids,
        .num_target_layers = getU32(obj, "num_target_layers") catch 0,
        .markov_rank = try getU32(obj, "markov_rank"),
        .markov_head_type = try MarkovHeadType.fromString(markov_head_type_str),
        .confidence_head_with_markov = getBool(obj, "confidence_head_with_markov", false),
        .enable_confidence_head = getBool(obj, "enable_confidence_head", false),
    };
}

// ── Weights ──

/// One dense bf16 linear, pre-transposed to `[in, out]` for plain matmul.
/// This checkpoint carries no quantization (section 6 of the spec — all 64
/// tensors are BF16), so unlike mtp.zig's `QLinear` there is no scales/biases
/// slot at all.
const Linear = struct {
    w_t: mlx.mlx_array,

    fn deinit(self: *Linear) void {
        _ = mlx.mlx_array_free(self.w_t);
    }
};

pub const DsparkLayer = struct {
    input_ln_w: mlx.mlx_array,
    q_proj: Linear, // [hidden, hidden]
    k_proj: Linear, // [hidden, kv_dim]
    v_proj: Linear, // [hidden, kv_dim]
    o_proj: Linear, // [hidden, hidden]
    q_norm_w: mlx.mlx_array, // [head_dim]
    k_norm_w: mlx.mlx_array, // [head_dim]
    post_attn_ln_w: mlx.mlx_array,
    gate_proj: Linear, // [hidden, intermediate]
    up_proj: Linear, // [hidden, intermediate]
    down_proj: Linear, // [intermediate, hidden]

    fn deinit(self: *DsparkLayer) void {
        _ = mlx.mlx_array_free(self.input_ln_w);
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.o_proj.deinit();
        _ = mlx.mlx_array_free(self.q_norm_w);
        _ = mlx.mlx_array_free(self.k_norm_w);
        _ = mlx.mlx_array_free(self.post_attn_ln_w);
        self.gate_proj.deinit();
        self.up_proj.deinit();
        self.down_proj.deinit();
    }
};

pub const DsparkModel = struct {
    config: DsparkConfig,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,

    // Embedding / output. Per spec section 7, these are byte-identical to
    // the target's own embed_tokens/lm_head by training-time construction,
    // but this port loads the drafter's OWN copies (the spec's recommended
    // "start simple/safe" option) rather than refcount-sharing the target's
    // — sharing is a documented follow-up optimization, not a milestone-1
    // requirement.
    embed_tokens_w: mlx.mlx_array, // [vocab, hidden]
    lm_head_w_t: mlx.mlx_array, // [hidden, vocab], pre-transposed
    norm_w: mlx.mlx_array, // [hidden] — final backbone norm

    // Fusion / context projection.
    fc_w_t: mlx.mlx_array, // [num_taps*hidden, hidden], pre-transposed
    hidden_norm_w: mlx.mlx_array, // [hidden]

    layers: []DsparkLayer,

    // Markov head (VanillaMarkov only, per config validation at load).
    markov_w1: mlx.mlx_array, // [vocab, markov_rank] embedding table
    markov_w2_t: mlx.mlx_array, // [markov_rank, vocab], pre-transposed

    // Confidence head.
    confidence_proj_w_t: mlx.mlx_array, // [confidence_input_dim, 1], pre-transposed
    confidence_proj_b: mlx.mlx_array, // [1]

    pub fn deinit(self: *DsparkModel) void {
        _ = mlx.mlx_array_free(self.embed_tokens_w);
        _ = mlx.mlx_array_free(self.lm_head_w_t);
        _ = mlx.mlx_array_free(self.norm_w);
        _ = mlx.mlx_array_free(self.fc_w_t);
        _ = mlx.mlx_array_free(self.hidden_norm_w);
        for (self.layers) |*lw| lw.deinit();
        self.allocator.free(self.layers);
        _ = mlx.mlx_array_free(self.markov_w1);
        _ = mlx.mlx_array_free(self.markov_w2_t);
        _ = mlx.mlx_array_free(self.confidence_proj_w_t);
        _ = mlx.mlx_array_free(self.confidence_proj_b);
        self.config.deinit(self.allocator);
    }

    /// Validate the drafter against the bound target and record nothing
    /// beyond the check itself for milestone 1 (no embed/lm_head sharing,
    /// no KV-layer-type mapping needed — DSpark's cross-attention K/V come
    /// from the FUSED tapped-hidden projection, not a per-layer-type target
    /// KV alias the way drafter.zig's Gemma4 drafter needs).
    pub fn bind(self: *const DsparkModel, target: *Transformer) !void {
        if (self.config.hidden_size != target.config.hidden_size) {
            log.err("[dspark] config mismatch: hidden_size={d}, target.hidden_size={d}\n", .{
                self.config.hidden_size, target.config.hidden_size,
            });
            return error.DsparkTargetMismatch;
        }
        if (self.config.vocab_size != target.config.vocab_size) {
            log.err("[dspark] vocab_size mismatch: drafter={d}, target={d}\n", .{
                self.config.vocab_size, target.config.vocab_size,
            });
            return error.DsparkTargetMismatch;
        }
        // num_target_layers in the drafter's config.json is advisory (spec
        // section 7); validate target_layer_ids against the ACTUAL bound
        // target's layer count, which is authoritative.
        self.config.validateTargetLayerIds(target.config.num_hidden_layers) catch {
            log.err("[dspark] target_layer_ids {any} incompatible with target num_hidden_layers={d}\n", .{
                self.config.target_layer_ids, target.config.num_hidden_layers,
            });
            return error.DsparkTargetMismatch;
        };
        if (!std.mem.eql(u8, target.config.model_type, "qwen3") and
            !std.mem.startsWith(u8, target.config.model_type, "qwen3_5") and
            !std.mem.startsWith(u8, target.config.model_type, "qwen3_moe"))
        {
            log.err("[dspark] unsupported target model_type={s} (DSpark targets Qwen3 family)\n", .{target.config.model_type});
            return error.UnsupportedDsparkArch;
        }
    }
};

fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[dspark] missing tensor: {s}\n", .{key});
        return error.MissingDsparkWeight;
    };
    var owned = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&owned, arr));
    return owned;
}

fn ownAndTranspose2D(w: *const Weights, key: []const u8, s: mlx.mlx_stream) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[dspark] missing tensor: {s}\n", .{key});
        return error.MissingDsparkWeight;
    };
    const axes = [_]c_int{ 1, 0 };
    var t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&t, arr, &axes, 2, s));
    return t;
}

fn loadLinear(w: *const Weights, key_prefix: []const u8, s: mlx.mlx_stream) !Linear {
    var buf: [256]u8 = undefined;
    const key = try std.fmt.bufPrint(&buf, "{s}.weight", .{key_prefix});
    return .{ .w_t = try ownAndTranspose2D(w, key, s) };
}

fn loadLayer(w: *const Weights, allocator: std.mem.Allocator, idx: u32, s: mlx.mlx_stream) !DsparkLayer {
    _ = allocator;
    var buf: [256]u8 = undefined;
    const p = struct {
        fn f(b: []u8, i: u32, suffix: []const u8) []const u8 {
            return std.fmt.bufPrint(b, "layers.{d}.{s}", .{ i, suffix }) catch unreachable;
        }
    }.f;

    return .{
        .input_ln_w = try ownWeight(w, p(&buf, idx, "input_layernorm.weight")),
        .q_proj = try loadLinear(w, p(&buf, idx, "self_attn.q_proj"), s),
        .k_proj = try loadLinear(w, p(&buf, idx, "self_attn.k_proj"), s),
        .v_proj = try loadLinear(w, p(&buf, idx, "self_attn.v_proj"), s),
        .o_proj = try loadLinear(w, p(&buf, idx, "self_attn.o_proj"), s),
        .q_norm_w = try ownWeight(w, p(&buf, idx, "self_attn.q_norm.weight")),
        .k_norm_w = try ownWeight(w, p(&buf, idx, "self_attn.k_norm.weight")),
        .post_attn_ln_w = try ownWeight(w, p(&buf, idx, "post_attention_layernorm.weight")),
        .gate_proj = try loadLinear(w, p(&buf, idx, "mlp.gate_proj"), s),
        .up_proj = try loadLinear(w, p(&buf, idx, "mlp.up_proj"), s),
        .down_proj = try loadLinear(w, p(&buf, idx, "mlp.down_proj"), s),
    };
}

/// True when `model_dir` carries a DSpark sidecar we know how to load,
/// mirroring `mtp.hasMtpSidecar`'s subdirectory convention
/// (`<model_dir>/dspark/model.safetensors` + `config.json`).
pub fn hasDsparkSidecar(io: std.Io, model_dir: []const u8) bool {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const parent = std.fmt.bufPrint(&buf, "{s}/dspark", .{model_dir}) catch return false;
    var dir = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch return false;
    defer dir.close(io);
    const st = dir.statFile(io, "model.safetensors", .{}) catch return false;
    return st.size > 0;
}

/// Load the DSpark drafter from `<model_dir>/dspark/{config.json,model.safetensors}`.
pub fn loadDspark(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    model_dir: []const u8,
) !DsparkModel {
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dspark_dir = try std.fmt.bufPrint(&dir_buf, "{s}/dspark", .{model_dir});

    // Parse config.json.
    var cfg_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cfg_path = try std.fmt.bufPrint(&cfg_path_buf, "{s}/config.json", .{dspark_dir});
    var dir = try std.Io.Dir.openDirAbsolute(io, dspark_dir, .{});
    defer dir.close(io);
    const cfg_file = try dir.openFile(io, "config.json", .{});
    defer cfg_file.close(io);
    const cfg_bytes = try cfg_file.readToEndAlloc(io, allocator, 1 << 20);
    defer allocator.free(cfg_bytes);
    _ = cfg_path;

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, cfg_bytes, .{});
    defer parsed.deinit();
    var cfg = try parseConfig(allocator, parsed.value);
    errdefer cfg.deinit(allocator);

    if (cfg.markov_head_type != .vanilla) {
        log.err("[dspark] unsupported markov_head_type (only 'vanilla' is implemented)\n", .{});
        return error.UnsupportedMarkovHeadType;
    }

    var weights = try model_mod.loadWeights(io, allocator, dspark_dir);
    defer weights.deinit();

    const layers = try allocator.alloc(DsparkLayer, cfg.num_hidden_layers);
    errdefer allocator.free(layers);
    var li: u32 = 0;
    while (li < cfg.num_hidden_layers) : (li += 1) {
        layers[li] = try loadLayer(&weights, allocator, li, s);
    }

    var m = DsparkModel{
        .config = cfg,
        .allocator = allocator,
        .s = s,
        .embed_tokens_w = try ownWeight(&weights, "embed_tokens.weight"),
        .lm_head_w_t = try ownAndTranspose2D(&weights, "lm_head.weight", s),
        .norm_w = try ownWeight(&weights, "norm.weight"),
        .fc_w_t = try ownAndTranspose2D(&weights, "fc.weight", s),
        .hidden_norm_w = try ownWeight(&weights, "hidden_norm.weight"),
        .layers = layers,
        .markov_w1 = try ownWeight(&weights, "markov_head.markov_w1.weight"),
        .markov_w2_t = try ownAndTranspose2D(&weights, "markov_head.markov_w2.weight", s),
        .confidence_proj_w_t = try ownAndTranspose2D(&weights, "confidence_head.proj.weight", s),
        .confidence_proj_b = try ownWeight(&weights, "confidence_head.proj.bias"),
    };
    errdefer m.deinit();

    // Materialize all weights now so first-round latency doesn't pay for it.
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        const top = [_]mlx.mlx_array{
            m.embed_tokens_w, m.lm_head_w_t,        m.norm_w,
            m.fc_w_t,         m.hidden_norm_w,       m.markov_w1,
            m.markov_w2_t,    m.confidence_proj_w_t, m.confidence_proj_b,
        };
        for (top) |a| _ = mlx.mlx_vector_array_append_value(eval_vec, a);
        for (m.layers) |*lw| {
            const arrs = [_]mlx.mlx_array{
                lw.input_ln_w,       lw.q_proj.w_t,   lw.k_proj.w_t,
                lw.v_proj.w_t,       lw.o_proj.w_t,   lw.q_norm_w,
                lw.k_norm_w,         lw.post_attn_ln_w,
                lw.gate_proj.w_t,    lw.up_proj.w_t,  lw.down_proj.w_t,
            };
            for (arrs) |a| _ = mlx.mlx_vector_array_append_value(eval_vec, a);
        }
        _ = mlx.mlx_eval(eval_vec);
    }

    log.info("[dspark] loaded DSpark drafter (layers={d}, block_size={d}, taps={d})\n", .{
        m.config.num_hidden_layers, m.config.block_size, m.config.numTaps(),
    });
    return m;
}

// ── Forward: Qwen3DSparkAttention + decoder layer (backbone) ──

inline fn rmsNormFn(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, s));
    return out;
}

fn linearFwd(x: mlx.mlx_array, lin: *const Linear, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&out, x, lin.w_t, s));
    return out;
}

/// `noise_embedding` = `embed_tokens([anchor_token_id] ++ [mask_token_id] * (block_size-1))`.
/// Shape `[1, block_size, hidden]`.
pub fn buildNoiseEmbedding(
    self: *const DsparkModel,
    anchor_token_id: u32,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const bs = self.config.block_size;
    var ids = [_]i32{0} ** 64; // block_size is checkpoint-fixed and small (7)
    std.debug.assert(bs <= 64);
    ids[0] = @intCast(anchor_token_id);
    var i: u32 = 1;
    while (i < bs) : (i += 1) ids[i] = @intCast(self.config.mask_token_id);

    const id_shape = [_]c_int{@intCast(bs)};
    const id_arr = mlx.mlx_array_new_data(&ids, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(id_arr);

    var gathered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered);
    try mlx.check(mlx.mlx_take_axis(&gathered, self.embed_tokens_w, id_arr, 0, s));

    const hidden: c_int = @intCast(self.config.hidden_size);
    const out_shape = [_]c_int{ 1, @intCast(bs), hidden };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, gathered, &out_shape, 3, s));
    return out;
}

/// Project the raw multi-layer target-hidden concat through `fc` + `hidden_norm`
/// (`qwen3/modeling.py:372-373`): `hidden_norm(fc(target_hidden_states))`.
/// `raw_taps` is `[1, ctx_len, num_taps*hidden]`. Returns `[1, ctx_len, hidden]`.
pub fn projectTargetContext(
    self: *const DsparkModel,
    raw_taps: mlx.mlx_array,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    var proj = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&proj, raw_taps, self.fc_w_t, s));
    defer _ = mlx.mlx_array_free(proj);
    return rmsNormFn(proj, self.hidden_norm_w, self.config.rms_norm_eps, s);
}

/// One `Qwen3DSparkAttention` + decoder-layer forward
/// (`qwen3/modeling.py:43-198`, verbatim per the spec).
///
/// `hidden`         — `[1, q_len, hidden]`, the draft/noise stream (residual
///                     stream input to this layer).
/// `target_ctx`     — `[1, ctx_len, hidden]`, this round's FRESH projected
///                     target context (output of `projectTargetContext`).
/// `cache`          — this layer's persistent K/V cache. MILESTONE 2: the
///                     draft cache is expected to already hold prior rounds'
///                     committed k_ctx/v_ctx + k_noise/v_noise; milestone 1
///                     exercises a single round against a fresh cache.
/// `rope_offset`    — RoPE position of the FIRST position in the concatenated
///                     `[target_ctx; hidden]` span (matches K's length —
///                     resolved in spec section 1, no ambiguity).
///
/// Q is projected from `hidden` ONLY (never from `target_ctx`). K/V are each
/// computed from BOTH `target_ctx` and `hidden`, concatenated along the
/// sequence axis (context first, draft-block second) BEFORE QK-norm + RoPE.
/// Attention is fully non-causal (`mask_mode = ""`) — every draft position
/// attends to every other draft position plus the full target context.
pub fn attentionLayerFwd(
    self: *const DsparkModel,
    layer: *const DsparkLayer,
    hidden: mlx.mlx_array,
    target_ctx: mlx.mlx_array,
    cache: *KVCache,
    layer_idx: u32,
    rope_offset: c_int,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const cfg = &self.config;
    const h_count: c_int = @intCast(cfg.num_attention_heads);
    const kv_h: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const eps = cfg.rms_norm_eps;

    const h_shape = mlx.getShape(hidden);
    const q_len: c_int = h_shape[1];
    const ctx_shape = mlx.getShape(target_ctx);
    const ctx_len: c_int = ctx_shape[1];
    const flat_shape = [_]c_int{ 1, q_len, h_count * hd };

    // ── Pre-attn norm + residual save ──
    const residual = hidden;
    const normed = try rmsNormFn(hidden, layer.input_ln_w, eps, s);
    defer _ = mlx.mlx_array_free(normed);

    // Q from noise stream ONLY.
    const q_proj = try linearFwd(normed, &layer.q_proj, s);
    defer _ = mlx.mlx_array_free(q_proj);
    var q_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_r);
    {
        const q_shape = [_]c_int{ 1, q_len, h_count, hd };
        try mlx.check(mlx.mlx_reshape(&q_r, q_proj, &q_shape, 4, s));
    }
    const q_normed = try rmsNormFn(q_r, layer.q_norm_w, eps, s);
    defer _ = mlx.mlx_array_free(q_normed);

    // K/V from BOTH target_ctx and hidden, concatenated along seq axis.
    const k_ctx = try linearFwd(target_ctx, &layer.k_proj, s);
    defer _ = mlx.mlx_array_free(k_ctx);
    const k_noise = try linearFwd(normed, &layer.k_proj, s);
    defer _ = mlx.mlx_array_free(k_noise);
    const v_ctx = try linearFwd(target_ctx, &layer.v_proj, s);
    defer _ = mlx.mlx_array_free(v_ctx);
    const v_noise = try linearFwd(normed, &layer.v_proj, s);
    defer _ = mlx.mlx_array_free(v_noise);

    var k_cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_cat);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, k_ctx);
        _ = mlx.mlx_vector_array_append_value(vec, k_noise);
        try mlx.check(mlx.mlx_concatenate_axis(&k_cat, vec, 1, s));
    }
    var v_cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_cat);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, v_ctx);
        _ = mlx.mlx_vector_array_append_value(vec, v_noise);
        try mlx.check(mlx.mlx_concatenate_axis(&v_cat, vec, 1, s));
    }

    const kv_len: c_int = ctx_len + q_len;
    var k_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_r);
    var v_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_r);
    {
        const kv_shape = [_]c_int{ 1, kv_len, kv_h, hd };
        try mlx.check(mlx.mlx_reshape(&k_r, k_cat, &kv_shape, 4, s));
        try mlx.check(mlx.mlx_reshape(&v_r, v_cat, &kv_shape, 4, s));
    }

    // QK-norm AFTER K-concat (same weights for all positions).
    const k_normed = try rmsNormFn(k_r, layer.k_norm_w, eps, s);
    defer _ = mlx.mlx_array_free(k_normed);

    // Transpose to [1, heads, seq, head_dim] for RoPE + SDPA.
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var q_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_t);
    var k_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_t);
    var v_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_t);
    try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, s));
    try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, s));
    try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, s));

    // RoPE applied to Q at offset (ctx_len positions ahead of the block
    // start, since Q only spans the block) and to the FULL concatenated K
    // starting at `rope_offset` (spans ctx_len + q_len — spec section 1,
    // resolved: cos/sin exactly cover K's length using true absolute
    // sequence positions).
    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    var k_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rope);
    const q_rope_offset = rope_offset + ctx_len;
    try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, hd, false, mlx.mlx_optional_float.some(cfg.rope_theta), 1.0, q_rope_offset, .{ .ctx = null }, s));
    try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, hd, false, mlx.mlx_optional_float.some(cfg.rope_theta), 1.0, rope_offset, .{ .ctx = null }, s));

    // MILESTONE 2: cache.update appends this round's K/V to the persistent
    // draft cache (which already holds prior rounds' committed ctx/noise
    // K/V). Milestone 1 exercises one round against a fresh cache, so
    // `kv_view` here spans exactly this round's concatenated ctx+block.
    var kv_view = try cache.update(layer_idx, k_rope, v_t, s, 0);
    defer kv_view.deinit();

    var attn_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_out);
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    // Fully non-causal: attention_mask=None, is_causal=False in the reference
    // (spec section 2) — every draft position attends to every context AND
    // every other draft position (past and future within the block).
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, kv_view.k, kv_view.v, attn_scale, "", none_mask, .{ .ctx = null }, s));

    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm, 4, s));
    var attn_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_flat);
    try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, s));

    const o_out = try linearFwd(attn_flat, &layer.o_proj, s);
    defer _ = mlx.mlx_array_free(o_out);

    var h1 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h1);
    try mlx.check(mlx.mlx_add(&h1, residual, o_out, s));

    // ── MLP: down(silu(gate(x)) * up(x)), pre-norm residual ──
    const ff_normed = try rmsNormFn(h1, layer.post_attn_ln_w, eps, s);
    defer _ = mlx.mlx_array_free(ff_normed);
    const g = try linearFwd(ff_normed, &layer.gate_proj, s);
    defer _ = mlx.mlx_array_free(g);
    const up = try linearFwd(ff_normed, &layer.up_proj, s);
    defer _ = mlx.mlx_array_free(up);
    var g_sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_sig);
    try mlx.check(mlx.mlx_sigmoid(&g_sig, g, s));
    var g_silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_silu);
    try mlx.check(mlx.mlx_multiply(&g_silu, g, g_sig, s));
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_multiply(&act, g_silu, up, s));
    const mlp_out = try linearFwd(act, &layer.down_proj, s);
    defer _ = mlx.mlx_array_free(mlp_out);

    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, h1, mlp_out, s));
    _ = mlx.mlx_array_free(h1);
    return out;
}

pub const BackboneOut = struct {
    /// `[1, block_size, hidden]` — post-final-norm backbone output.
    hidden: mlx.mlx_array,
    /// `[1, block_size, vocab]` — base (pre-Markov) logits.
    base_logits: mlx.mlx_array,
};

/// One full backbone forward: all `num_hidden_layers` (5) decoder layers
/// over the draft block, then final norm + lm_head. This is ONE parallel
/// forward proposing the whole block (spec section 2) — no per-token loop
/// at this layer (the Markov-corrected SEQUENTIAL sample walk over the
/// resulting per-position base logits is a separate, cheap step — see
/// `sampleBlockTokens` below, not part of this transformer forward).
///
/// MILESTONE 2: `draft_cache` is expected to be the request's PERSISTENT
/// 5-layer cache (grows across rounds); this milestone's test coverage is
/// limited to hermetic config/math checks (no real checkpoint available),
/// so the live multi-round behavior is unverified pending an equivalence
/// test against the reference (see report).
pub fn proposeBlock(
    self: *const DsparkModel,
    anchor_token_id: u32,
    raw_target_taps: mlx.mlx_array,
    draft_cache: *KVCache,
    rope_offset: c_int,
    s: mlx.mlx_stream,
) !BackboneOut {
    const target_ctx = try projectTargetContext(self, raw_target_taps, s);
    defer _ = mlx.mlx_array_free(target_ctx);

    var hidden = try buildNoiseEmbedding(self, anchor_token_id, s);
    errdefer _ = mlx.mlx_array_free(hidden);

    for (self.layers, 0..) |*layer, i| {
        const next = try attentionLayerFwd(self, layer, hidden, target_ctx, draft_cache, @intCast(i), rope_offset, s);
        _ = mlx.mlx_array_free(hidden);
        hidden = next;
    }

    const post_norm = try rmsNormFn(hidden, self.norm_w, self.config.rms_norm_eps, s);
    _ = mlx.mlx_array_free(hidden);
    hidden = .{ .ctx = null }; // disarm the errdefer above — ownership transferred to post_norm
    errdefer _ = mlx.mlx_array_free(post_norm);

    var logits = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&logits, post_norm, self.lm_head_w_t, s));

    return .{ .hidden = post_norm, .base_logits = logits };
}

// ── Markov head (VanillaMarkov) ──

/// `bias = markov_w2 @ markov_w1[prev_token_id]` (`markov_head.py:8-32`).
/// `prev_token_ids` is `[n]` int32; returns `[n, vocab]` bias.
pub fn markovStepBias(
    self: *const DsparkModel,
    prev_token_ids: mlx.mlx_array,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    var latent = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(latent);
    try mlx.check(mlx.mlx_take_axis(&latent, self.markov_w1, prev_token_ids, 0, s));

    var bias = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&bias, latent, self.markov_w2_t, s));
    return bias;
}

// ── Confidence head ──

/// `sigmoid(Linear(concat(hidden, markov_w1[prev_token_id])))` per position
/// (`qwen3/modeling.py:292-307`). `hidden` is `[1, n, hidden]`,
/// `prev_token_ids` is `[n]` int32 (the shift-by-1 predecessor per
/// position — anchor for position 0). Returns `[1, n, 1]` confidence
/// probabilities (already sigmoided).
pub fn confidenceProbs(
    self: *const DsparkModel,
    hidden: mlx.mlx_array,
    prev_token_ids: mlx.mlx_array,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    if (!self.config.confidence_head_with_markov) {
        var logits = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&logits, hidden, self.confidence_proj_w_t, s));
        defer _ = mlx.mlx_array_free(logits);
        var biased = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&biased, logits, self.confidence_proj_b, s));
        defer _ = mlx.mlx_array_free(biased);
        var probs = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sigmoid(&probs, biased, s));
        return probs;
    }

    var prev_emb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prev_emb);
    try mlx.check(mlx.mlx_take_axis(&prev_emb, self.markov_w1, prev_token_ids, 0, s));

    const h_shape = mlx.getShape(hidden);
    const n: c_int = h_shape[1];
    const prev_shape = [_]c_int{ 1, n, @intCast(self.config.markov_rank) };
    var prev_emb_3d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prev_emb_3d);
    try mlx.check(mlx.mlx_reshape(&prev_emb_3d, prev_emb, &prev_shape, 3, s));

    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, hidden);
        _ = mlx.mlx_vector_array_append_value(vec, prev_emb_3d);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 2, s));
    }

    var logits = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&logits, cat, self.confidence_proj_w_t, s));
    defer _ = mlx.mlx_array_free(logits);
    var biased = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&biased, logits, self.confidence_proj_b, s));
    defer _ = mlx.mlx_array_free(biased);
    var probs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&probs, biased, s));
    return probs;
}

/// `_confident_prefix_length` (`draft_ops.py:82-93`) — pure math, no mlx
/// calls: given a slice of per-position sigmoid confidences (already
/// computed) and a threshold, find the first position below threshold and
/// return that as the proposal length; `threshold <= 0` or all-confident
/// both return `block_size` unchanged.
pub fn confidentPrefixLength(confidences: []const f32, block_size: u32, threshold: f32) u32 {
    if (threshold <= 0.0) return block_size;
    for (confidences, 0..) |c, i| {
        if (c < threshold) return @intCast(i);
    }
    return block_size;
}

// MILESTONE 2: the Markov-corrected sequential sample walk
// (`sample_block_tokens`, spec section 3) — 7 cheap sequential steps over
// the already-computed `base_logits`, each step: apply `markovStepBias`
// using the PREVIOUS step's sampled token, sample, feed forward. Needs a
// sampler (temperature/argmax) which lives in generate.zig; wiring that in
// is part of milestone 2's `Generator.nextDspark`, not this file.

// MILESTONE 2: the Leviathan verify/accept loop, the persistent draft-cache
// snapshot/restore (in lockstep with the target's KVCache), the NEW
// transformer.zig `capture_hidden_layers` multi-layer raw-hidden capture
// point, scheduler dispatch, the `--dspark` CLI flag, and per-request
// `enable_dspark`/`confidence_threshold` on all four HTTP surfaces.

// ── Tests ──

const testing = std.testing;

test "dspark: parseConfig reads a synthetic config.json" {
    const allocator = testing.allocator;
    const json_text =
        \\{
        \\  "architectures": ["Qwen3DSparkModel"],
        \\  "attention_bias": false,
        \\  "block_size": 7,
        \\  "confidence_head_with_markov": true,
        \\  "enable_confidence_head": true,
        \\  "head_dim": 128,
        \\  "hidden_size": 4096,
        \\  "intermediate_size": 12288,
        \\  "mask_token_id": 151669,
        \\  "markov_head_type": "vanilla",
        \\  "markov_rank": 256,
        \\  "model_type": "qwen3",
        \\  "num_attention_heads": 32,
        \\  "num_hidden_layers": 5,
        \\  "num_key_value_heads": 8,
        \\  "num_target_layers": 36,
        \\  "rms_norm_eps": 1e-06,
        \\  "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        \\  "target_layer_ids": [1, 9, 17, 25, 33],
        \\  "vocab_size": 151936
        \\}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    var cfg = try parseConfig(allocator, parsed.value);
    defer cfg.deinit(allocator);

    try testing.expectEqual(@as(u32, 4096), cfg.hidden_size);
    try testing.expectEqual(@as(u32, 151936), cfg.vocab_size);
    try testing.expectEqual(@as(u32, 5), cfg.num_hidden_layers);
    try testing.expectEqual(@as(u32, 32), cfg.num_attention_heads);
    try testing.expectEqual(@as(u32, 8), cfg.num_key_value_heads);
    try testing.expectEqual(@as(u32, 128), cfg.head_dim);
    try testing.expectEqual(@as(u32, 12288), cfg.intermediate_size);
    try testing.expectEqual(@as(f32, 1e-6), cfg.rms_norm_eps);
    try testing.expectEqual(@as(f32, 1_000_000.0), cfg.rope_theta);
    try testing.expectEqual(@as(u32, 7), cfg.block_size);
    try testing.expectEqual(@as(u32, 151669), cfg.mask_token_id);
    try testing.expectEqual(@as(u32, 256), cfg.markov_rank);
    try testing.expectEqual(MarkovHeadType.vanilla, cfg.markov_head_type);
    try testing.expect(cfg.confidence_head_with_markov);
    try testing.expect(cfg.enable_confidence_head);
    try testing.expectEqual(@as(usize, 5), cfg.target_layer_ids.len);
    try testing.expectEqual(@as(u32, 1), cfg.target_layer_ids[0]);
    try testing.expectEqual(@as(u32, 33), cfg.target_layer_ids[4]);
    try testing.expectEqual(@as(u32, 36), cfg.num_target_layers);
    try testing.expectEqual(@as(u32, 5), cfg.numTaps());
    try testing.expectEqual(@as(u32, 4096 + 256), cfg.confidenceInputDim());
}

test "dspark: parseConfig defaults markov_head_type to vanilla and rejects unknown" {
    const allocator = testing.allocator;
    const json_text =
        \\{
        \\  "hidden_size": 128, "vocab_size": 256, "num_hidden_layers": 1,
        \\  "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
        \\  "intermediate_size": 256, "mask_token_id": 5, "markov_rank": 8,
        \\  "target_layer_ids": [0]
        \\}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();
    var cfg = try parseConfig(allocator, parsed.value);
    defer cfg.deinit(allocator);
    try testing.expectEqual(MarkovHeadType.vanilla, cfg.markov_head_type);

    try testing.expectError(error.UnsupportedMarkovHeadType, MarkovHeadType.fromString("gated"));
    try testing.expectError(error.UnsupportedMarkovHeadType, MarkovHeadType.fromString("rnn"));
}

test "dspark: parseConfig errors on missing required field" {
    const allocator = testing.allocator;
    const json_text =
        \\{ "hidden_size": 128, "vocab_size": 256 }
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();
    try testing.expectError(error.MissingDsparkConfigField, parseConfig(allocator, parsed.value));
}

test "dspark: validateTargetLayerIds rejects out-of-range and last-layer ids" {
    const allocator = testing.allocator;
    const ids_ok = try allocator.dupe(u32, &[_]u32{ 1, 9, 17, 25, 33 });
    var cfg = DsparkConfig{
        .hidden_size = 4096,
        .vocab_size = 151936,
        .num_hidden_layers = 5,
        .num_attention_heads = 32,
        .num_key_value_heads = 8,
        .head_dim = 128,
        .intermediate_size = 12288,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .block_size = 7,
        .mask_token_id = 151669,
        .target_layer_ids = ids_ok,
        .num_target_layers = 36,
        .markov_rank = 256,
        .markov_head_type = .vanilla,
        .confidence_head_with_markov = true,
        .enable_confidence_head = true,
    };
    defer cfg.deinit(allocator);

    // Valid against a 36-layer target.
    try cfg.validateTargetLayerIds(36);
    // Out of range against a smaller target.
    try testing.expectError(error.DsparkTargetMismatch, cfg.validateTargetLayerIds(10));
    // Zero layers is always invalid.
    try testing.expectError(error.DsparkTargetMismatch, cfg.validateTargetLayerIds(0));

    // Including the target's LAST layer (0-indexed 35 of 36) must be rejected.
    allocator.free(cfg.target_layer_ids);
    cfg.target_layer_ids = try allocator.dupe(u32, &[_]u32{ 1, 9, 35 });
    try testing.expectError(error.DsparkTargetMismatch, cfg.validateTargetLayerIds(36));
}

test "dspark: confidentPrefixLength boundary cases" {
    // threshold <= 0 -> always full block_size regardless of confidences.
    try testing.expectEqual(@as(u32, 7), confidentPrefixLength(&[_]f32{ 0.01, 0.01, 0.01 }, 7, 0.0));
    try testing.expectEqual(@as(u32, 7), confidentPrefixLength(&[_]f32{ 0.01, 0.01, 0.01 }, 7, -1.0));

    // All confident (none below threshold) -> full block_size.
    try testing.expectEqual(@as(u32, 7), confidentPrefixLength(&[_]f32{ 0.9, 0.95, 0.99 }, 7, 0.5));

    // First position already below threshold -> truncate to 0.
    try testing.expectEqual(@as(u32, 0), confidentPrefixLength(&[_]f32{ 0.1, 0.9, 0.9 }, 7, 0.5));

    // Truncate at first low-confidence position (index 2).
    try testing.expectEqual(@as(u32, 2), confidentPrefixLength(&[_]f32{ 0.9, 0.8, 0.3, 0.9 }, 7, 0.5));

    // Empty slice with threshold enabled -> full block_size (nothing to
    // compare, so it never finds a below-threshold position).
    try testing.expectEqual(@as(u32, 7), confidentPrefixLength(&[_]f32{}, 7, 0.5));
}

test "dspark: DsparkConfig.numTaps / confidenceInputDim without markov" {
    const allocator = testing.allocator;
    const ids = try allocator.dupe(u32, &[_]u32{ 0, 1 });
    var cfg = DsparkConfig{
        .hidden_size = 512,
        .vocab_size = 1000,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 128,
        .intermediate_size = 1024,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .block_size = 4,
        .mask_token_id = 99,
        .target_layer_ids = ids,
        .num_target_layers = 10,
        .markov_rank = 64,
        .markov_head_type = .vanilla,
        .confidence_head_with_markov = false,
        .enable_confidence_head = false,
    };
    defer cfg.deinit(allocator);
    try testing.expectEqual(@as(u32, 2), cfg.numTaps());
    try testing.expectEqual(@as(u32, 512), cfg.confidenceInputDim());
}
