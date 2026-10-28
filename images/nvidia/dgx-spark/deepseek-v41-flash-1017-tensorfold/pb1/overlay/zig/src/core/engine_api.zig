//! The engine a server drives (``Engine``), and ``LaneHost``: the lane core served on one thread, rounds left to it.
const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

/// The server's name for one request, unique while the process lives.
pub const Id = u64;

/// One request's sampling (Python's exact_sampling.Sampling); a request without one decodes greedily.
pub const Sampling = lanes.Sampling;

/// A prompt's reproducible seed (``seed_for``), for requests that name none.
pub const seedFor = lanes.sampling.seedFor;

/// The request's stop strings, checked on the engine's thread after each committed token (the server decodes the tail).
pub const Stop = struct {
    ctx: *anyopaque,
    check: *const fn (ctx: *anyopaque, emitted: []const u32) bool,
};

/// A prompt's image (a vision family's): its span and the family's payload (lanes.stream.Image).
pub const Image = lanes.stream.Image;

/// A vision family's image input (lanes.stream.Vision): the server renders each image part as the placeholder and the
/// family's prepare sizes its span.
pub const Vision = lanes.stream.Vision;

/// tool_choice "required" or a named function: outside a think block the answer opens a call to an offered tool.
pub const CallGate = struct {
    /// The token that opens a call, the template text before the name, and the mark that ends the name.
    opener: u32,
    lead: []const u8 = "",
    tail: []const u8 = "",
    /// Offered names (empty: any); the gate completes a name its prefix starts.
    names: []const []const u8 = &.{},
    think_open: ?u32 = null,
    think_end: ?u32 = null,
};

/// response_format and the guided_* fields: every token keeps the reply inside this grammar.
pub const Structure = struct {
    kind: enum { json, json_schema, regex, choice, grammar },
    /// The schema (JSON text), regex, choices (a JSON array) or EBNF; empty for any JSON object.
    text: []const u8 = "",
    /// With thinking on, the grammar starts after this token.
    after: ?u32 = null,
};

pub const draft_depths = 16;
pub const DraftDepth = struct { drafted: u32 = 0, reached: u32 = 0, kept: u32 = 0 };

/// A reply to decode. The request and every slice in it stay valid until its ``finished`` event.
pub const Request = struct {
    prompt: []const u32,
    /// Prefix-cache namespace only; no effect on model tokens or sampling.
    cache_namespace: ?[32]u8 = null,
    /// The prompt's images in order, each span inside the prompt (its positions hold the image token).
    images: []const Image = &.{},
    /// The prompt as the prompt cache keys it, one id for each of its tokens: an image position's id made from the picture's
    /// content (two pictures never share a kept state, one picture always does); empty: the prompt itself.
    cache_key: []const u32 = &.{},
    max_tokens: u32,
    sampling: ?Sampling = null,
    /// Tokens that end the reply as ``stop`` (the token is delivered); empty when the request ignores EOS.
    eos: []const u32 = &.{},
    stop: ?Stop = null,
    /// false: one token a round and no drafts, the serial reference drafted replies must equal.
    drafts: bool = true,
    /// Background work yields to foreground arrivals.
    background: bool = false,
    /// Prompt prefix lengths worth keeping for a later turn: the rendered history, then shared system blocks.
    history_len: u32 = 0,
    shared_prefixes: []const u32 = &.{},
    /// Where the prompt's prefill chunks start after 0 (Python's PrefillPlan at ``Info.prefill_step``); empty: the engine's own.
    chunks: []const u32 = &.{},
    /// Most reply tokens inside a think block before the engine writes ``think_close`` (0: no limit).
    think_budget: u32 = 0,
    think_close: []const u32 = &.{},
    think_end: ?u32 = null,
    /// The offered tools as JSON text, for drafters that propose a call's structure; empty: no tools.
    tools_json: []const u8 = "",
    /// Set only when ``Info`` says the engine enforces it.
    call: ?CallGate = null,
    structure: ?Structure = null,
    /// Stop a short exact cycle while the think block is open.
    loop_guard: bool = false,
    loop_dry_windows: u32 = 10,
};

pub const Reason = enum { stop, length, cancelled, failed };

/// What a finished reply's rounds did, for its runtime fields and /metrics.
pub const Stats = struct {
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    depths: [draft_depths]DraftDepth = @splat(.{}),
    /// The narrowest verify window of the reply's rounds (drafted replies: 2 or more).
    min_rows: u32 = 0,
    prefill_widths: []const u32 = &.{},
    prefill_raised: []const bool = &.{},
    /// The loop period that ended or interrupted the think block, if any.
    loop_period: ?u32 = null,
    loop_novelty: ?f64 = null,
    /// Prompt prefill duration measured by the engine host, in seconds.
    prefill_seconds: ?f64 = null,
    /// The drafter's own counters as a JSON object, or empty.
    telemetry_json: []const u8 = "",
};

pub const Event = union(enum) {
    /// The prompt is in the cache: ``cached`` of its tokens came from a kept prefix.
    prefilled: u32,
    /// Tokens committed for this request, in order; valid only during the call.
    tokens: []const u32,
    /// The reply ended; nothing follows. ``message`` says why a failed reply failed.
    finished: struct { reason: Reason, stats: Stats = .{}, message: []const u8 = "" },
};

/// Where a request's events go. Called on the engine's thread: it must return at once and not re-enter the engine.
pub const Sink = struct {
    ctx: *anyopaque,
    event: *const fn (ctx: *anyopaque, id: Id, event: *const Event) void,
};

/// A reasoning effort name and the number a family's chat template takes for it (DeepSeek-V4.1's: "medium" -> 62).
pub const Effort = struct { name: []const u8, value: i64 };

/// What the engine serves, fixed once it has loaded.
pub const Info = struct {
    name: []const u8 = "lanes",
    /// Enforces per-request prefix namespaces (salted requests never use
    /// unsalted learned disk states). Unsupported engines refuse cache_salt.
    cache_salt: bool = false,
    lanes: u32 = 1,
    /// Prompt plus reply tokens one request may use (0: no limit).
    context_window: u32 = 0,
    context_fitted: bool = false,
    /// What the engine enforces; the server refuses requests that need more.
    call_gates: bool = false,
    /// The engine observes loop_guard requests in its lane stream.
    loop_guard: bool = false,
    structures: bool = false,
    /// Prompt rows a prefill chunk at most, for the server's chunk starts (0: the engine cuts prompts itself).
    prefill_step: u32 = 0,
    /// A line the server prints once at startup (the engine's memory plan); empty: none.
    startup: []const u8 = "",
    /// The reasoning efforts the family's template takes as numbers: each name passes as asked and the template hears
    /// its number. Empty: the names the template quotes, as text (the nearest one for any other name).
    efforts: []const Effort = &.{},
    /// A vision family's image input; null: the server refuses images.
    vision: ?Vision = null,
};

/// A checkpoint family an engine reads: its config ``model_type`` and weight formats, as gate entries name them.
pub const Family = struct { model_type: []const u8, formats: []const []const u8 };

/// What a server asks of the engine it opens: the checkpoint, and the serve flags an engine reads.
pub const Open = struct {
    dir: []const u8,
    model_type: []const u8,
    context: ?i64 = null,
    lanes: u32 = 8,
    /// --parallel named a number: an engine whose memory fits fewer streams refuses instead of serving fewer.
    lanes_fixed: bool = false,
    drafts: bool = true,
    speed_up: ?[]const u8 = null,
    prompt_cache_gib: ?f64 = null,
    prompt_cache_over_cap: bool = false,
    /// --learn: where shared prompt states are kept on disk for later sessions and servers (null: off).
    learn: ?[]const u8 = null,
    /// --learn-gib: what learned states may take on disk, every model and build together.
    learn_gib: f64 = 32,
    /// --device and --segments (CUDA); null: the backend's environment fallback, then its default.
    device: ?u32 = null,
    segments: ?u32 = null,
    /// PETABRIDGE (k2a-pb1) spill P5: 0014. The prompt cache's disk tier and parked-request scheduling (spill v2);
    /// every switch off. Nothing reads it yet (P6 binds it in the backend; P7 passes the flags).
    spill: Spill = .{},
};

/// PETABRIDGE (k2a-pb1) spill P5: 0014's Spill, verbatim.
/// The prompt cache's durable tier and its scheduling (wojo's GLM 0088/0110/0111): --spill-*, with TF_SPILL_* for the
/// rest. Off unless both a directory and a cap are given; every rank of an engine must be opened with the same values
/// (the family checks them before any spill command).
pub const Spill = struct {
    /// --spill-dir: this rank's local store (a fresh native namespace under it; never the model or Engram mount).
    dir: ?[]const u8 = null,
    /// --spill-gib: what the store may hold on this rank (0: off).
    gib: f64 = 0,
    /// --spill-min-free-gib: no write leaves less free on the store's file system.
    min_free_gib: f64 = 50,
    /// --spill-quota: one conversation's share of the cap (0: none).
    quota: f64 = 0,
    /// --spill-min-tokens: shorter kept states are not written.
    min_tokens: u32 = 8192,
    /// TF_SPILL_TURNS: old turns kept on a conversation's active path before they go first (0: until room is needed).
    turns: u32 = 4,
    /// TF_SPILL_LOAD_MS: one service's budget for every restore together, then staged writes (at least 1).
    load_ms: u32 = 40,
    /// TF_SPILL_LOAD_WAIT_MS: how long a request waits parked on a restore before it is admitted without it.
    load_wait_ms: u32 = 30_000,
    /// TF_SPILL_FLUSH_MS: a clean shutdown's bound for draining writes.
    flush_ms: u32 = 60_000,

    /// The disk tier is on.
    pub fn enabled(s: Spill) bool {
        return s.dir != null and s.gib > 0;
    }

    /// Whether the disk tier was asked for and `engine` did not take it: a backend that wires it configures its lane
    /// host (LaneHost.configure), whose status then carries the cache's counters.
    pub fn unbound(s: Spill, engine: Engine) bool {
        if (!s.enabled()) return false;
        var st: Status = .{};
        engine.status(&st, &.{});
        return st.cache == null;
    }
};

/// PETABRIDGE (k2a-pb1) spill P5: 0014's CacheStats, verbatim.
/// The prompt cache's counters for /health and /metrics, read from HTTP threads as one copy (LaneHost publishes it on
/// its engine thread). Null in Status unless a tier or grouping is on. Totals unless in `gauges`.
pub const CacheStats = struct {
    memory_hits: u64 = 0,
    memory_misses: u64 = 0,
    kept: u64 = 0,
    evicted: u64 = 0,
    refused: u64 = 0,
    failed: u64 = 0,
    entries: u64 = 0,
    held_bytes: u64 = 0,
    disk_lookups: u64 = 0,
    disk_hits: u64 = 0,
    disk_hits_committed: u64 = 0,
    restores_begun: u64 = 0,
    restores_shared: u64 = 0,
    restores_failed: u64 = 0,
    restores_cancelled: u64 = 0,
    restores_dropped: u64 = 0,
    restores_refused: u64 = 0,
    restores_active: u64 = 0,
    restoring_bytes: u64 = 0,
    restore_blocks_done: u64 = 0,
    restore_blocks_total: u64 = 0,
    restored_tokens: u64 = 0,
    restore_waits: u64 = 0,
    restore_wait_timeouts: u64 = 0,
    restore_wait_ns: u64 = 0,
    restore_wait_max_ns: u64 = 0,
    writes_enqueued: u64 = 0,
    shared_promoted: u64 = 0,
    flushes: u64 = 0,
    flush_timeouts: u64 = 0,
    flush_unpublished: u64 = 0,
    parked: u64 = 0,
    chats_continued: u64 = 0,
    chats_continued_prefill_tokens: u64 = 0,
    cached_tokens: u64 = 0,
    computed_tokens: u64 = 0,
    /// The tier's own counters, each rank's (null: no tier).
    spill: ?prompt_cache.RankCounts = null,

    pub const gauges = [_][]const u8{ "entries", "held_bytes", "restores_active", "restoring_bytes", "restore_blocks_done", "restore_blocks_total", "parked" };

    /// Each field's /metrics help (a missing one fails the build).
    pub const help = .{
        .memory_hits = "Prompt passes that resumed a kept state in memory.",
        .memory_misses = "Prompt lookups that found no usable kept state.",
        .kept = "Prompt states kept in memory.",
        .evicted = "Kept states evicted from memory for room.",
        .refused = "Prompt states not kept: past the memory budget.",
        .failed = "Kept states that failed to save or restore.",
        .entries = "Prompt states kept in memory now.",
        .held_bytes = "Bytes the kept states hold now.",
        .disk_lookups = "Lookups that asked the disk tier.",
        .disk_hits = "Lookups whose durable point was longer than the best usable memory state.",
        .disk_hits_committed = "Disk restores every rank installed, adopted as kept states.",
        .restores_begun = "Disk restore jobs started.",
        .restores_shared = "Requests that joined a disk restore already in flight.",
        .restores_failed = "Disk restores aborted on every rank (missing, damaged, guarded).",
        .restores_cancelled = "Disk restores stopped: no request waited for them any more, or shutdown.",
        .restores_dropped = "Installed disk restores not adopted: no room, or memory already had them.",
        .restores_refused = "Disk restores not begun: the memory budget could not hold their destination.",
        .restores_active = "Disk restore jobs in flight.",
        .restoring_bytes = "Memory the disk restores in flight reserved, charged to the prompt cache's budget.",
        .restore_blocks_done = "Blocks the restores in flight have copied so far.",
        .restore_blocks_total = "Blocks the restores in flight copy in all.",
        .restored_tokens = "Prompt tokens restored from disk.",
        .restore_waits = "Requests parked on a disk restore.",
        .restore_wait_timeouts = "Requests admitted without their disk restore after TF_SPILL_LOAD_WAIT_MS.",
        .restore_wait_ns = "Nanoseconds requests waited for disk restores.",
        .restore_wait_max_ns = "The longest wait for a disk restore, in nanoseconds.",
        .writes_enqueued = "Kept states enqueued for the disk tier at their keep.",
        .shared_promoted = "Disk points promoted to shared prefixes.",
        .flushes = "Clean-shutdown flushes of the disk tier.",
        .flush_timeouts = "Flushes that reached their deadline with points unpublished.",
        .flush_unpublished = "Points left unpublished at a flush deadline.",
        .parked = "Requests parked on a disk restore now (reading it, or deferred to ask again).",
        .chats_continued = "Streams that continued a conversation without resuming from its own turn state.",
        .chats_continued_prefill_tokens = "Prompt tokens those streams prefilled.",
        .cached_tokens = "Prompt tokens passes resumed from kept states.",
        .computed_tokens = "Prompt tokens passes computed.",
    };
};

/// An opened engine; ``close`` stops its thread and frees its backend.
pub const Opened = struct { engine: Engine, close: *const fn (ctx: *anyopaque) void, ctx: *anyopaque };

pub const Memory = struct { active: u64 = 0, cache: u64 = 0, peak: u64 = 0 };

/// A backend's words for its own refusals (a request it cannot serve); null: the error's name.
pub const Explain = struct {
    ctx: ?*anyopaque = null,
    text: *const fn (ctx: ?*anyopaque, err: anyerror) ?[]const u8,
};

/// A backend's own memory counts for LaneHost; read from HTTP threads, so it must not touch the GPU.
pub const MemorySource = struct {
    ctx: ?*anyopaque = null,
    read: *const fn (ctx: ?*anyopaque, reset_peak: bool) ?Memory,
};

/// Completed prompt passes, available before the generated reply finishes.
pub const PrefillTelemetry = struct {
    input: u64 = 0,
    cached: u64 = 0,
    computed: u64 = 0,
    seconds: f64 = 0,
    count: u64 = 0,
    active: u32 = 0,
    progress: u64 = 0,
    pending: u64 = 0,
};

pub const AdaptiveTelemetry = struct {
    rounds: u64 = 0,
    cost_updates: u64 = 0,
    acceptance_updates: u64 = 0,
    probes: u64 = 0,
    rejected_times: u64 = 0,
    prediction_samples: u64 = 0,
    prediction_abs_error_microseconds: u64 = 0,
    verify_microseconds: u64 = 0,
    last_rows: u64 = 0,
    last_streams: u64 = 0,
    last_context_tokens: u64 = 0,
    last_predicted_verify_microseconds: u64 = 0,
    last_verify_microseconds: u64 = 0,
};

pub const Status = struct {
    adaptive: ?AdaptiveTelemetry = null,
    /// Requests in prefill or decode, and those waiting for a lane.
    running: u32 = 0,
    waiting: u32 = 0,
    decode_tokens_per_second: f64 = 0,
    prefill_tokens_per_second: f64 = 0,
    /// Requests that gave a lane up to a later one; null when the engine does not count them.
    preemptions: ?u64 = null,
    warming: bool = false,
    /// Live streams written to the caller's buffer: tokens each holds.
    streams: usize = 0,
    /// Generated tokens held by live streams, excluding their prompts.
    generation_tokens: u64 = 0,
    token_pool: ?lanes.backend.TokenPool = null,
    prefill_telemetry: ?PrefillTelemetry = null,
    /// PETABRIDGE (k2a-pb1) spill P5: 0014. The prompt cache's counters, when its tier or grouping is on (null: off,
    /// which is every engine until P6 configures one).
    cache: ?CacheStats = null,
};

pub const SubmitError = error{ Closed, Busy };
pub const ResetCacheError = error{ Closed, Busy, Unsupported };

pub const Engine = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        info: *const fn (ctx: *anyopaque) Info,
        submit: *const fn (ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void,
        /// Ends a request between rounds; its ``finished`` (cancelled) still arrives. Unknown ids are ignored.
        cancel: *const fn (ctx: *anyopaque, id: Id) void,
        status: *const fn (ctx: *anyopaque, out: *Status, stream_tokens: []u32) void,
        /// Device memory in bytes, the peak since the last reset; null when the backend keeps no count.
        memory: *const fn (ctx: *anyopaque, reset_peak: bool) ?Memory,
        /// The engine's queue as a keepalive target, or null when the backend is not Metal.
        keepalive: ?*const fn (ctx: *anyopaque) ?keepalive.Target = null,
        /// One logit per label, and the vocabulary logsumexp. Null until a family scores decisions.
        score: ?*const fn (ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64) error{Failed}!f64 = null,
        /// Idle-only reset, serialized on the engine thread; no weight reload.
        reset_cache: ?*const fn (ctx: *anyopaque) ResetCacheError!void = null,
    };

    pub fn info(e: Engine) Info {
        return e.vtable.info(e.ctx);
    }
    pub fn submit(e: Engine, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        return e.vtable.submit(e.ctx, id, request, sink);
    }
    pub fn cancel(e: Engine, id: Id) void {
        e.vtable.cancel(e.ctx, id);
    }

    pub fn resetCache(e: Engine) ResetCacheError!void {
        const f = e.vtable.reset_cache orelse return error.Unsupported;
        return f(e.ctx);
    }
    pub fn status(e: Engine, out: *Status, stream_tokens: []u32) void {
        e.vtable.status(e.ctx, out, stream_tokens);
    }
    pub fn memory(e: Engine, reset_peak: bool) ?Memory {
        return e.vtable.memory(e.ctx, reset_peak);
    }
    /// The engine's queue as a keepalive target, or null when the backend is not Metal.
    pub fn keepaliveTarget(e: Engine) ?keepalive.Target {
        const f = e.vtable.keepalive orelse return null;
        return f(e.ctx);
    }

    /// The decision hook, or Unsupported when this engine does not score.
    pub fn score(e: Engine, prompt: []const u32, labels: []const u32, logits: []f64) error{ Failed, Unsupported }!f64 {
        const f = e.vtable.score orelse return error.Unsupported;
        return f(e.ctx, prompt, labels, logits);
    }
};

/// A backend's own driver for a lone greedy stream (the GPU round), which LaneHost runs while the stream is alone.
pub const Lone = struct {
    ctx: *anyopaque,
    sampled: bool = false, // it drives sampled streams too
    /// Prefills `s` and decodes it until it finishes (false) or `hooks.yield` hands it to the lane core (true).
    run: *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: LoneHooks) anyerror!bool,
};

/// What a lone driver tells the host between rounds: tokens landed; and asks: hand the stream over now?
pub const LoneHooks = struct {
    ctx: *anyopaque,
    committed: *const fn (ctx: *anyopaque) void,
    yield: *const fn (ctx: *anyopaque) bool,
};

/// The lane core served to the HTTP threads (lane_host.zig).
pub const LaneHost = @import("lane_host.zig").LaneHost;

/// Exact prompt reuse between requests, for any family (prompt_cache.zig).
pub const prompt_cache = @import("prompt_cache.zig");

/// Learned prompt-cache states on disk (prompt_imprint.zig).
pub const prompt_imprint = @import("prompt_imprint.zig");

/// The idle keepalive's ticker and target contract.
pub const keepalive = @import("keepalive.zig");

test {
    _ = @import("lane_host.zig");
    _ = @import("lane_host_reuse_test.zig");
    _ = @import("prompt_cache.zig");
    _ = @import("lane_host_tier_test.zig"); // PETABRIDGE (k2a-pb1) spill P5: 0014's host tests (tip, adapted)
    _ = @import("spill_p5_k2a_test.zig"); // PETABRIDGE (k2a-pb1) spill P5: the k2a fit's own tests
    _ = @import("prompt_imprint.zig");
    _ = @import("keepalive.zig");
}
