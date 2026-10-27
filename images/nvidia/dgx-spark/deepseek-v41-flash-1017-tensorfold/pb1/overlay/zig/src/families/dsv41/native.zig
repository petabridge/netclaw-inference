//! DeepSeek-V4.1-Flash behind the native CUDA family interface, tensor parallel over two GPUs: the server's process is
//! rank 0 (the lane core over lanes.zig's Backend on model.zig), rank 1 is `tf-dsv41-lanes` following it on the other
//! node (the same model directory, rank cache, kit, pool and port). The ranks link over TCP and NCCL; the inputs
//! beyond the checkpoint come from the environment:
//!   TF_TP_WORLD (2), TF_TP_MASTER, TF_TP_PORT   the ranks: rank 0's fabric address and port (rank 1 connects)
//!   TF_DS_RANK_CACHE                            the lane's per-rank weight files (the Python lane's TF_DS_RANK_CACHE)
//!   TF_DS_KIT                                   cubins/ (the served extension cubins), the served RoPE tables
//!                                               (rope-*.f32, as many rows as the context) and engram.json
//!   TF_DS_ENGRAM, TF_DS_TOKEN_MAP               the Engram tables and the compressed token map
//!   TF_DS_ARENA_GIB                             a first block of device memory for the buffers (unset: they take what
//!                                               they need, the pool's caches by the context)
//!   TF_DS_POOL_TOKENS                          optional aggregate pool, independent of --context per request;
//!                                               followers must use the same value via --pool
//!   TF_DS_ROUND_MS                              the lane core's round costs by rows (ms, comma separated; the
//!                                               served ROUND_MS unless set: a four-node lane's rounds cost less)
//!   TF_DS_ROUND_ROWS, TF_DS_STREAMS             a round's rows at most (16; the four-node lane's 48) and the pool's
//!                                               streams (4; its 16): --parallel at most the streams
//!   TF_RDMA_DEVICES                             the decode gathers over the RDMA ring on these devices (else NCCL)
//!   TF_DS_GRAPHS=1                              the rounds' stretches as CUDA graphs
//!   TF_DS_HC_SIDE=1                             the mixes' side work on a stream of its own
//!   TF_DS_L2_PREFETCH=1                         the paced L2 prefetch of the next kernels' weights
//!   TF_DS_ENGRAM_AIO=1                          a round's Engram reads by Linux AIO on O_DIRECT
//!   TF_DS_FIX_K=K                               a measurement: every drafted stream verifies K drafts a round
//! The kernel set (TENSORFOLD_CUDA_KERNELS) is the recorded Triton set (aot_pack.py). The pool holds TF_DS_POOL_TOKENS
//! (default --context)
//! positions for every stream together (each takes an extent: its prompt, its max_tokens and a round's rows).
//! TF_TP_WORLD=4: the four nodes of the exact 2D split (prompt2d.zig), ranks 1-3 `tf-dsv41-lanes` following on the
//! other nodes; TF_RDMA_DEVICES (and TF_DS_RDMA_KERNELS: rdma_gather.cu's fatbin, unset: the embedded one) put its
//! decode-size exchanges on the RDMA rings, as the four-node Python lane's TF_RDMA_DEVICES does.
//! PETABRIDGE (k2a-pb1) spill P6: the spill tier (MiaAI-Lab 0016/0019's wiring, spill_tier.zig; off unless the
//! server's --spill-dir and --spill-gib): rank 0's switches come through the server (api.Spill: --spill-dir,
//! --spill-gib, --spill-min-free-gib, --spill-min-tokens, --spill-quota, TF_SPILL_TURNS, TF_SPILL_FLUSH_MS), its
//! staging cap from TF_DS_SPILL_STAGING_MIB (128). Each follower runs its own store on its own disk with its own
//! switches (tf-dsv41-lanes --spill-dir/--spill-gib/..., or TF_DS_SPILL_DIR, TF_DS_SPILL_GIB, ...); the ranks'
//! fingerprints (rank_identity.zig: TF_DS_ENGINE_ID, TF_DS_RECIPE_PATCHES and TF_DS_RANK_CACHE_SHA or
//! TF_DS_MODEL_REVISION on every rank, TF_DS_ENGRAM_REVISION with Engram) must agree at startup, a difference naming
//! the rank and field. TF_TP_WORLD 2 or 3 (TP3: every rank writes its own full replicated copy); TF_DS_KV_SHARD must be
//! unset or 0 (KV sharding is not ported). A broken agreement exits rank 0 with status 75 (0020's minimal piece) so the
//! coordinator relaunches every rank.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const model = @import("model.zig");
const lanes_mod = @import("lanes.zig");
const round = @import("round.zig");
const link = @import("link.zig");
const rank_cache = @import("rank_cache.zig");
const vision_mod = @import("vision.zig");
// PETABRIDGE (k2a-pb1) spill P6
const spill_tier = @import("spill_tier.zig");
const rank_wire = @import("rank_wire.zig");
const rank_identity = @import("rank_identity.zig");
const fingerprint = @import("fingerprint.zig");
const prompt = @import("prompt.zig");
const core = @import("core");

pub const model_type = "deepseek_v41";
pub const formats: []const []const u8 = &.{"exl3-mul1"};
/// The Python lane's default window; the server's --context moves it up to the model's 1,048,576.
pub const default_context: i64 = 262144;
pub const max_segments: u32 = 1;
/// The served build's prompt chunk (TF_DS_PREFILL_CHUNK).
pub const prompt_rows: u32 = 2048;
/// The served lane's reasoning effort numbers (text.zig): the server passes each name and the template hears its number.
pub const efforts = @import("text.zig").efforts;

pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    /// PETABRIDGE (k2a-pb1) spill P6: the spill tier (null: off; the served path and its bytes unchanged).
    spill: ?spill_tier.Settings = null,
};

/// Separate the HTTP request window from the aggregate placement plane. The
/// legacy default is unchanged; an explicit pool leaves room for the aligned
/// placement's verification tail even for a request filling its whole window.
/// RoPE uses FP32 positions, exact only below 2^24; this is an arithmetic bound,
/// not a memory-fit promise. Every rank still needs matching pool and RoPE files.
pub fn poolTokens(context: usize, value: ?[]const u8) !usize {
    const text = value orelse return context;
    if (text.len == 0) return error.BadPool;
    for (text) |c| if (c < '0' or c > '9') return error.BadPool;
    const count = std.fmt.parseInt(usize, text, 10) catch return error.BadPool;
    const limit = (1 << 24) - round.max_rows - 1;
    if (context > limit or count > limit) return error.BadPool;
    const need = std.mem.alignForward(usize, context + round.max_rows + 2, model.chunk_rows);
    if (count < need) return error.BadPool;
    return count;
}

test "aggregate pool is independent of request context and keeps legacy default" {
    try std.testing.expectEqual(@as(usize, 401408), try poolTokens(401408, null));
    try std.testing.expectEqual(@as(usize, 1048576), try poolTokens(401408, "1048576"));
    try std.testing.expectEqual(@as(usize, 3300000), try poolTokens(401408, "3300000"));
    try std.testing.expectEqual(@as(usize, 403456), try poolTokens(401408, "403456"));
    for ([_][]const u8{ "", "0", "401408", "403455", "-1", "+1048576", "1_048_576", " 1048576", "1048576 ", "16777216", "999999999999999999999999" }) |value|
        try std.testing.expectError(error.BadPool, poolTokens(401408, value));
}

/// A lone drafted stream's own driver: decodes it until it finishes (false) or `yield` hands it over (true).
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates for its own sequence.
    stream_bytes: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    /// Kept prompt states for the host's prompt cache (core/prompt_cache Snapshots: `ptr` and its functions).
    snaps: ?Snaps = null,
    /// Images in requests (the server renders their placeholders; the tower runs on this rank).
    vision: ?lanes.stream.Vision = null,
    /// PETABRIDGE (k2a-pb1) spill P6: the spill tier (the prompt cache's Tier), when Options.spill asked for it, and
    /// the prompt cache's durable rules with it (Rules.grid: the chunk; Rules.window: the replay window).
    tier: ?spill_tier.prompt_cache.Tier = null,
    grid: u32 = 0,
    replay: u32 = 0,
};

pub const Snaps = struct {
    ptr: *anyopaque,
    bytes: *const fn (*anyopaque, u32) u64,
    save: *const fn (*anyopaque, ?*anyopaque, u32) anyerror!*anyopaque,
    restore: *const fn (*anyopaque, ?*anyopaque, *anyopaque) anyerror!void,
    drop: *const fn (*anyopaque, *anyopaque) void,
    live: *const fn (*anyopaque, *anyopaque) bool,
    compatible: *const fn (*anyopaque, *anyopaque, usize) bool,
    exact_end: bool = true,
};

/// The rank's model and the backend over it (what `open` hands the server, freed by `deinit`).
const State = struct {
    gpa: std.mem.Allocator,
    m: *model.Model,
    lanes: lanes_mod.Lanes,
    vision: ?*vision_mod.Vision = null,
    // PETABRIDGE (k2a-pb1) spill P6
    staging: ?cuda.HostBuffer = null, // the spill tier's pinned staging (allocated before the pool)
    events: []cuda.Event = &.{}, // one a staging slot
    stopping: bool = false, // deinit began: a broken agreement then ends nothing (the process is ending anyway)
};

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// An empty variable as none (a container passes TF_RDMA_DEVICES empty when it is unset).
fn nonEmpty(v: ?[]const u8) ?[]const u8 {
    const s = v orelse return null;
    return if (s.len == 0) null else s;
}

/// The device bytes this rank's weights take, for the host's memory check: its file in TF_DS_RANK_CACHE (the
/// checkpoint holds both ranks'); 0 when there is none to read (open then says what to set).
pub fn weightBytes(io: std.Io, dir: []const u8) u64 {
    _ = dir;
    var d = std.Io.Dir.cwd().openDir(io, getenv("TF_DS_RANK_CACHE") orelse return 0, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var buf: [512]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const world = std.fmt.parseInt(u32, getenv("TF_TP_WORLD") orelse "2", 10) catch return 0;
    const name = rank_cache.find(fba.allocator(), io, d, 0, world) catch return 0;
    const st = d.statFile(io, name, .{}) catch return 0;
    return st.size;
}

/// Loads rank 0 and links followers against TF_TP_MASTER:TF_TP_PORT; the pool holds TF_DS_POOL_TOKENS or `o.context`
/// positions. Needs `ctx` current.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    const aggregate_pool = try poolTokens(o.context, getenv("TF_DS_POOL_TOKENS"));
    const world = std.fmt.parseInt(u32, getenv("TF_TP_WORLD") orelse "2", 10) catch return error.BadWorld;
    try model.checkWorldMode(world, getenv("TF_DS_EXPERIMENTAL_TP3"));
    const master = link.parseIp(getenv("TF_TP_MASTER") orelse return error.NoMaster) catch return error.BadMaster;
    const port = std.fmt.parseInt(u16, getenv("TF_TP_PORT") orelse "29620", 10) catch return error.BadPort;
    const arena_gib = if (getenv("TF_DS_ARENA_GIB")) |t| std.fmt.parseInt(usize, t, 10) catch return error.BadArena else 0;
    // PETABRIDGE (k2a-pb1) spill P6: the tier's refusals from the environment, before the staging, weights and pool
    if (o.spill != null) try spillPreflight(getenvFn);
    const st = try gpa.create(State);
    errdefer gpa.destroy(st);
    st.gpa = gpa;
    st.staging = null;
    st.events = &.{};
    st.stopping = false;
    // the spill tier's staging comes out of memory before the pool is sized (MiaAI-Lab 0016: budgeted first)
    if (o.spill) |sp| st.staging = try spillStaging(ctx, sp);
    errdefer if (st.staging) |*b| b.free();
    const mo: model.Options = .{
        .model_dir = dir,
        .cache_dir = getenv("TF_DS_RANK_CACHE") orelse return error.NoRankCache,
        .kit_dir = getenv("TF_DS_KIT") orelse return error.NoKit,
        .aot_dir = kernels,
        .rank = 0,
        .world = world,
        .master = master,
        .port = port,
        .engram_dir = getenv("TF_DS_ENGRAM"),
        .token_map = getenv("TF_DS_TOKEN_MAP"),
        .rdma_devices = nonEmpty(getenv("TF_RDMA_DEVICES")),
        .rdma_kernels = getenv("TF_DS_RDMA_KERNELS"),
        .graphs = if (getenv("TF_DS_GRAPHS")) |v| !std.mem.eql(u8, v, "0") else false,
        .side = if (getenv("TF_DS_HC_SIDE")) |v| !std.mem.eql(u8, v, "0") else false,
        .prefetch = if (getenv("TF_DS_L2_PREFETCH")) |v| !std.mem.eql(u8, v, "0") else false,
        .engram_aio = if (getenv("TF_DS_ENGRAM_AIO")) |v| !std.mem.eql(u8, v, "0") else false,
        .pool = aggregate_pool,
        .drafts = o.drafts,
        .arena_bytes = arena_gib << 30,
        .round_rows = if (getenv("TF_DS_ROUND_ROWS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadRoundRows else round.default_rows,
        .round_ms = if (getenv("TF_DS_ROUND_MS")) |v| lanes_mod.parseMs(std.heap.page_allocator, v) catch return error.BadRoundMs else null,
        .streams = if (getenv("TF_DS_STREAMS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadStreams else model.max_streams_default,
    };
    st.m = try model.Model.open(gpa, io, ctx, mo);
    errdefer st.m.close();
    st.lanes = lanes_mod.Lanes.init(gpa, st.m);
    // PETABRIDGE (k2a-pb1) spill P6: a start refused past here (a fingerprint mismatch, a store that does not open)
    // frees the frames the agreement wrote through the lanes (0016); the followers end on their own
    errdefer st.lanes.wire.deinit(gpa);
    // the ranks' agreement and this rank's store, while each follower scans its own (opened at the agreement)
    if (o.spill) |sp| {
        st.events = try openSpill(gpa, io, st.m, &st.lanes, mo, sp, st.staging.?);
        // 0020's minimal piece: a broken agreement (rank_wire's Leader.fail, a rank that stopped answering) exits
        // rank 0 non-zero, so the coordinator relaunches every rank instead of the tier hanging requests
        st.lanes.spill.?.on_broken = spillBroken;
        st.lanes.spill.?.broken_ctx = st;
    }
    errdefer if (st.lanes.spill) |t| {
        t.destroy();
        st.lanes.spill = null;
        for (st.events) |*e| e.deinit();
        if (st.events.len > 0) gpa.free(st.events);
        st.events = &.{};
    };
    if (getenv("TF_DS_ADAPTIVE_LEARNING")) |value| {
        if (!std.mem.eql(u8, value, "0") and !std.mem.eql(u8, value, "1")) return error.BadAdaptiveLearning;
        st.lanes.adaptive_learning = std.mem.eql(u8, value, "1");
    }
    st.lanes.logit_dump_dir = getenv("TF_DS_CACHE_LOGITS_DIR");
    st.lanes.short_continuation = if (getenv("TF_DS_CACHE_SHORT_CONTINUATION")) |v| std.mem.eql(u8, v, "1") else false;
    if (getenv("TF_DS_FIX_K")) |t| st.lanes.fix_k = std.fmt.parseInt(usize, t, 10) catch return error.BadFixK;
    // a vision checkpoint's tower on this rank, from the kit's vision/ (torch's attention cubin, the routing bias)
    st.vision = null;
    const diagnostic_no_vision = if (getenv("TF_DS_DIAGNOSTIC_NO_VISION")) |v| std.mem.eql(u8, v, "1") else false;
    if (diagnostic_no_vision) std.log.warn("DIAGNOSTIC ONLY: vision initialization omitted, model not qualified", .{});
    if (st.m.cfg.vision and !diagnostic_no_vision) st.vision = try vision_mod.Vision.open(gpa, io, st.m, dir, getenv("TF_DS_KIT").?);
    if (st.m.ring) |r| r.dumpRegistrations("after HTTP family init");
    errdefer if (st.vision) |v| v.close();
    st.lanes.vision = st.vision;
    // the pool is the model's (preallocated): a stream takes no device memory of its own
    const L = lanes_mod.Lanes;
    const snaps: Snaps = .{ .ptr = &st.lanes, .bytes = L.snapBytesFn, .save = L.snapSaveFn, .restore = L.snapRestoreFn, .drop = L.snapDropFn, .live = L.snapLiveFn, .compatible = L.snapCompatibleFn };
    // PETABRIDGE (k2a-pb1) spill P6: the tier and its durable rules (null and 0 with the tier off: nothing binds)
    const tier: ?spill_tier.prompt_cache.Tier = if (st.lanes.spill) |t| t.tier() else null;
    return .{ .backend = st.lanes.backend(), .facts = st.lanes.facts(), .rows = @intCast(st.m.eng.round_rows), .stream_bytes = 0, .ctx = st, .deinit = deinitFn, .snaps = snaps, .vision = if (st.vision) |v| v.hook() else null, .tier = tier, .grid = if (tier != null) model.chunk_rows else 0, .replay = if (tier != null) @intCast(st.m.cfg.window) else 0 };
}

fn deinitFn(ptr: *anyopaque) void {
    const st: *State = @ptrCast(@alignCast(ptr));
    // PETABRIDGE (k2a-pb1) spill P6: the tier after the prompt cache's store (cuda.zig frees it first) and before
    // the family: its store closes; a break from here on is the shutdown's own, not a cause to exit
    st.stopping = true;
    if (st.lanes.spill) |t| t.destroy();
    st.lanes.spill = null;
    for (st.events) |*e| e.deinit();
    if (st.events.len > 0) st.gpa.free(st.events);
    if (st.staging) |*b| b.free();
    st.lanes.deinit(); // rank 1 hears done
    if (st.vision) |v| v.close();
    st.m.close();
    st.gpa.destroy(st);
}

// -- PETABRIDGE (k2a-pb1) spill P6: the spill tier's startup (rank 0 here, the followers in tf-dsv41-lanes) -----------
// MiaAI-Lab 0016's, with 0019's spill half (a follower's store opens on its own thread at the agreement, rank 0 waits
// while its scan progresses), fitted: worlds 2 and 3, TF_DS_KV_SHARD refused unless unset or 0, no shard planner.

/// The worlds the spill tier runs on: the TP2 pair and plain TP3 (rank_wire and spill_tier take up to 4, but our world
/// 4 is the 2D split, whose ranks hold different shapes: not qualified).
pub fn spillWorld(world: u32) bool {
    return world == 2 or world == 3;
}

/// What refuses the spill tier from the environment alone, checked on every rank before anything loads (minutes on
/// the Sparks): TF_TP_WORLD 2 or 3, TF_DS_KV_SHARD unset or "0" (KV sharding is not ported: any other value,
/// "1" or not a number, is refused), and the identity the fingerprint needs (rank_identity.Extra.env:
/// TF_DS_ENGINE_ID, TF_DS_RECIPE_PATCHES, TF_DS_RANK_CACHE_SHA or TF_DS_MODEL_REVISION; TF_DS_ENGRAM_REVISION with
/// TF_DS_ENGRAM).
pub fn spillPreflight(env: *const fn ([:0]const u8) ?[]const u8) !void {
    const world = std.fmt.parseInt(u32, env("TF_TP_WORLD") orelse "2", 10) catch return error.BadWorld;
    if (!spillWorld(world)) return error.SpillNeedsTwoOrThreeRanks;
    if (env("TF_DS_KV_SHARD")) |v| if (!std.mem.eql(u8, std.mem.trim(u8, v, " "), "0") and std.mem.trim(u8, v, " ").len > 0) return error.SpillWithKvShard;
    _ = try rank_identity.Extra.env(env);
    if (nonEmpty(env("TF_DS_ENGRAM")) != null and (env("TF_DS_ENGRAM_REVISION") orelse @as([]const u8, "")).len == 0) return error.NoEngramId;
}

/// The spill tier's switches from the server's (rank 0; api.Spill's fields) and TF_DS_SPILL_STAGING_MIB.
pub fn spillSettings(dir: []const u8, gib: f64, min_free_gib: f64, quota: f64, min_tokens: u32, turns: u32, flush_ms: u32) !spill_tier.Settings {
    var s: spill_tier.Settings = .{ .dir = dir, .gib = gib, .min_free_gib = min_free_gib, .quota = quota, .min_tokens = min_tokens, .turns = turns, .flush_ms = flush_ms };
    if (getenv("TF_DS_SPILL_STAGING_MIB")) |v| s.staging_mib = std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch return error.BadSpillSetting;
    if (s.staging_mib < spill_tier.min_staging_mib) return error.BadSpillSetting;
    return s;
}

/// The pinned staging a rank's spill tier works in (its cap; allocated before the model's pool).
pub fn spillStaging(ctx: *const cuda.Context, s: spill_tier.Settings) !cuda.HostBuffer {
    var b = try cuda.HostBuffer.alloc(ctx.d, @intCast(s.stagingBytes()));
    if (@intFromPtr(b.bytes.ptr) % 4096 != 0) {
        b.free();
        return error.UnalignedStaging;
    }
    return b;
}

fn stagingSlice(b: cuda.HostBuffer) []align(4096) u8 {
    return @alignCast(b.bytes);
}

pub fn getenvFn(name: [:0]const u8) ?[]const u8 {
    return getenv(name);
}

/// This rank's fingerprint exchange (rank_identity's inputs: the model options it opened with, the opened model, the
/// recipe's identity from the environment, its switches and its state layout). Always replicated here.
pub fn rankExchange(gpa: std.mem.Allocator, io: std.Io, mo: model.Options, m: *model.Model, s: ?spill_tier.Settings, layout: *const model.Model.spill.StateLayout) !fingerprint.Exchange {
    const envp = try rank_identity.Extra.envFor(getenvFn, s != null);
    var fbuf: [256]u8 = undefined;
    const image_cache = if (getenv("TF_DS_IMAGE_CACHE")) |v| !std.mem.eql(u8, v, "0") else true;
    const features = try rank_identity.featureString(&fbuf, .{ .spill = s != null, .spill_min_tokens = if (s) |x| x.min_tokens else 0, .kv_shard = false, .image_cache = image_cache, .streams = @intCast(m.streams), .round_rows = @intCast(m.eng.round_rows), .pool = m.pool_cap, .drafts = m.drafting() });
    var pbuf: [model.Model.spill.StateLayout.max_encoded]u8 = undefined;
    const lp = try rank_identity.layoutPart(layout, m.world, &pbuf);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const rc_name = try rank_cache.find(arena.allocator(), io, m.cache_dir, m.rank, m.world);
    const x: rank_identity.Extra = .{
        .engine_id = envp.engine_id,
        .recipe_patches = envp.recipe_patches,
        .build_flags = envp.build_flags,
        .model_revision = envp.model_revision,
        .ablation = envp.ablation,
        .rank_cache_sha = envp.rank_cache_sha,
        .engram_revision = envp.engram_revision,
        .features = features,
        .native_format = core.spill.format.version,
        .chunk_rows = model.chunk_rows,
        .replay_floor = spill_replay_floor,
        .idx_split = spill_idx_split,
        .planes = lp.planes,
        .rank_layout = lp.rank_layout,
        .owner_layout = lp.owner_layout,
        .owner_version = lp.owner_version,
        .parts = lp.parts,
        .persist = s != null,
        .gpu_capability = try m.ctx.capability(),
        .cuda_driver = try m.ctx.d.version(),
        .rank_cache_name = rc_name,
    };
    var g = try rank_identity.gather(gpa, io, mo, m, x);
    defer g.deinit();
    const fp = try g.build(gpa);
    return fp.exchange();
}

/// PETABRIDGE (k2a-pb1) spill P6: MiaAI's base exports these two math switches from prompt.zig (prompt.replay_floor,
/// prompt.idx_split); ours has neither switch. Our arithmetic: the served bounded replay with its floor at the replay
/// (fillChunk: floor = replay, replayCut) and no split index pass. Fixed per build, so every rank agrees; they only
/// enter the fingerprint (a build that changes either must change them, which renames the store namespace).
const spill_replay_floor = true;
const spill_idx_split = false;

/// One reusable event a staging slot (spill_tier.eventCount).
pub fn spillEvents(gpa: std.mem.Allocator, m: *model.Model, n: usize) ![]cuda.Event {
    const evs = try gpa.alloc(cuda.Event, n);
    var made: usize = 0;
    errdefer {
        for (evs[0..made]) |*e| e.deinit();
        gpa.free(evs);
    }
    for (evs) |*e| {
        e.* = try cuda.Event.init(m.ctx.d, false);
        made += 1;
    }
    return evs;
}

fn agreeSink(ctx: *anyopaque, sub: []const u8) anyerror!void {
    return lanes_mod.Lanes.spillFrame(ctx, rank_wire.lanes_kind, sub);
}

/// Rank 0: the fingerprint agreement with every follower (before anything spill-related crosses), then this rank's
/// store under <spill-dir>/native-spill-<compat>/rank0 and the tier on `ln` (ln.spill), then every follower's store
/// open (each scans its own on its own thread since the agreement: 0019). The events it returns are the caller's
/// (freed after the tier).
pub fn openSpill(gpa: std.mem.Allocator, io: std.Io, m: *model.Model, ln: *lanes_mod.Lanes, mo: model.Options, s: spill_tier.Settings, staging: cuda.HostBuffer) ![]cuda.Event {
    if (!spillWorld(m.world)) return error.SpillNeedsTwoOrThreeRanks;
    const layout = try m.spillLayout();
    const plan = try spill_tier.stagingPlan(s, &layout);
    const fx = try rankExchange(gpa, io, mo, m, s, &layout);
    const lead = try gpa.create(rank_wire.Leader);
    var lead_owned = true;
    defer if (lead_owned) gpa.destroy(lead);
    lead.* = rank_wire.Leader.init(@intCast(m.world), rank_wire.freshEpoch(io), fx);
    const wire = ln.spillWire();
    const a = lead.agree(.{ .ctx = ln, .send = agreeSink }, wire.links) catch |e| {
        var buf: [192]u8 = undefined;
        if (lead.broken) |b| std.log.err("dsv41 spill: {s}", .{rank_wire.describeBroken(b, &buf)});
        // a follower without the tier ends at the hello (SpillNotStarted on its side): rank 0 sees its link go down
        if (e != error.FingerprintMismatch) std.log.err("dsv41 spill: no agreement with the followers; if one printed \"follow\": \"SpillNotStarted\", start it with the spill tier too (TF_DS_SPILL_DIR and TF_DS_SPILL_GIB)", .{});
        return if (e == error.FingerprintMismatch) error.SpillFingerprintMismatch else e;
    };
    const events = try spillEvents(gpa, m, spill_tier.eventCount(plan));
    errdefer {
        for (events) |*e| e.deinit();
        gpa.free(events);
    }
    var gens = try m.spillGenerations(gpa);
    const compat = spill_tier.storeCompat(a, 0, &layout);
    const max_tokens: u32 = @intCast(m.pool_cap);
    const side = lanes_mod.SpillTier.S.create(gpa, io, .{ .m = m }, layout, gens, s, 0, compat, max_tokens, stagingSlice(staging), events, .{}) catch |e| {
        gens.deinit(gpa);
        return e;
    };
    var boot: [16]u8 = undefined;
    io.random(&boot);
    const t = lanes_mod.SpillTier.create(gpa, side, ln.spillFamily(), wire, lead, s, boot) catch |e| {
        side.destroy();
        return e;
    };
    lead_owned = false;
    // every follower scans its store on its own thread since the agreement (rank 0's scan above ran meanwhile):
    // waited for while its scan makes progress, before any spill command
    t.awaitFollowers(null, 50 * std.time.ns_per_ms) catch |e| {
        t.destroy();
        return e;
    };
    ln.spill = t;
    std.log.info("dsv41 spill: rank 0's store under {s} ({d:.1} GiB, staging {d} MiB); {d} ranks agreed", .{ s.dir, s.gib, s.stagingBytes() >> 20, m.world });
    return events;
}

/// A follower's spill side, made before its model's pool (staging) is sized and opened at rank 0's agreement. The
/// staging and events stay the caller's.
pub fn followerSpill(gpa: std.mem.Allocator, io: std.Io, m: *model.Model, mo: model.Options, s: spill_tier.Settings, staging: cuda.HostBuffer, events: *[]cuda.Event) !*lanes_mod.SpillFollower {
    if (!spillWorld(m.world)) return error.SpillNeedsTwoOrThreeRanks;
    const layout = try m.spillLayout();
    const plan = try spill_tier.stagingPlan(s, &layout);
    const fx = try rankExchange(gpa, io, mo, m, s, &layout);
    events.* = try spillEvents(gpa, m, spill_tier.eventCount(plan));
    errdefer {
        for (events.*) |*e| e.deinit();
        gpa.free(events.*);
        events.* = &.{};
    }
    var gens = try m.spillGenerations(gpa);
    errdefer gens.deinit(gpa);
    const max_tokens: u32 = @intCast(m.pool_cap);
    // (no family yet: lanes.followWith attaches its kept states)
    return lanes_mod.SpillFollower.create(gpa, io, @intCast(m.rank), fx, null, .{ .dev = .{ .m = m }, .layout = layout, .gens = gens, .settings = s, .max_tokens = max_tokens, .staging = stagingSlice(staging), .events = events.* });
}

/// 0020's minimal piece: the spill tier's agreement broke (it logged the rank and cause): rank 0 exits with 75 (the
/// broken-pair status), so the coordinator, which restarts every rank when one container exits, relaunches the
/// trio; a request is never left waiting on a tier that can no longer agree. Not during deinit (a follower that
/// already stopped at shutdown). _exit: no atexit handler may wait on the device or a peer.
fn spillBroken(ctx: ?*anyopaque, err: anyerror) void {
    const st: *State = @ptrCast(@alignCast(ctx.?));
    if (st.stopping) return;
    var buf: [192]u8 = undefined;
    const lead = if (st.lanes.spill) |t| t.lead else null;
    const cause = if (lead) |l| (if (l.broken) |b| rank_wire.describeBroken(b, &buf) else @errorName(err)) else @errorName(err);
    std.log.err("dsv41 spill: the ranks' agreement broke ({s}): rank 0 exits with status {d} so every rank restarts", .{ cause, spill_exit_status });
    std.c._exit(spill_exit_status);
}

/// rank 0's exit status when the spill tier's agreement broke (0020's broken-pair status).
pub const spill_exit_status = 75;

/// A request or a start this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    if (rank_wire.explain(err)) |t| return t; // PETABRIDGE (k2a-pb1) spill P6: the ranks' agreement's own words
    return switch (err) {
        // PETABRIDGE (k2a-pb1) spill P6
        error.SpillWithKvShard => "the spill tier (--spill-dir/--spill-gib) does not run with TF_DS_KV_SHARD (KV sharding is not ported): leave it unset or 0",
        error.SpillNeedsTwoOrThreeRanks => "the spill tier runs on TP2 or TP3 (TF_TP_WORLD=2 or 3), every rank with it on",
        error.SpillFingerprintMismatch => "the ranks' configurations differ (the log names the rank and field): start every rank from the same image, model, kit and TF_DS_* settings, with the spill tier on every rank (TF_DS_SPILL_DIR/TF_DS_SPILL_GIB on the followers)",
        error.SpillNotStarted => "rank 0 runs the spill tier (--spill-dir/--spill-gib): start this rank with it too (TF_DS_SPILL_DIR and TF_DS_SPILL_GIB, or tf-dsv41-lanes --spill-dir/--spill-gib)",
        error.BadSpillSetting => "TF_DS_SPILL_STAGING_MIB: the spill staging cap in MiB, at least 16 (default 128); the follower's TF_DS_SPILL_* values must be numbers (quota 0 to 1)",
        error.StagingTooSmall => "the spill staging cap (TF_DS_SPILL_STAGING_MIB) holds too few slots for this model's blocks and small state: raise it",
        error.SpillLayoutMismatch => "the spill state layout does not match the loaded model or its allocation (an engine bug; the startup 'spill layout:' line says why)",
        error.FollowerScanStalled => "a follower's spill store scan stopped making progress (its log names the store): check its spill disk (TF_DS_SPILL_DIR), or start it with a fresh directory",
        error.SpillRankZeroInServer => "tf-dsv41-lanes runs the spill tier as a follower only: rank 0's tier runs in the server (tensorfold-native --spill-dir/--spill-gib)",
        error.NotPortedYet => "the native DeepSeek-V4.1 engine supports world2/4 and this fork's experimental world3",
        error.ExperimentalTP3Required => "TP3 requires TF_DS_EXPERIMENTAL_TP3=1 on every rank and the padded TP3 checkpoint; GPU/numerical qualification is still required",
        error.NoMaster, error.BadMaster => "set TF_TP_MASTER to rank 0's fabric address (rank 1: tf-dsv41-lanes MODEL CACHE 1 2 MASTER PORT KIT; ranks 1-3 of four: ... G 4 ...)",
        error.NoRdmaKernels => "TF_RDMA_DEVICES at world 4 needs TF_DS_RDMA_KERNELS (rdma_gather.cu's fatbin) in a build without the kernel images",
        error.NoRankCache => "set TF_DS_RANK_CACHE to the lane's per-rank weight files",
        error.NoKit => "set TF_DS_KIT to the kernel kit (cubins/, the RoPE tables, engram.json)",
        error.NoFreeSlot => "this engine serves TF_DS_STREAMS streams at once (default 4): --parallel at most that",
        error.BadRoundRows => "TF_DS_ROUND_ROWS: a round's rows at most, 16 .. 48",
        error.BadStreams => "TF_DS_STREAMS: the pool's streams, 1 .. 16",
        error.BadPool => "TF_DS_POOL_TOKENS: aggregate positions, at least alignUp(--context + 50, 2048), below 2^24 - 48; use the same --pool on followers and sufficient RoPE tables",
        error.ContextFull => "the streams' prompts and budgets fill the context's pool: lower max_tokens, or retry when a stream ends",
        error.PromptTooLong => "the prompt and its max_tokens do not fit the context window",
        error.NoVisionKit => "this checkpoint reads images: its kit (TF_DS_KIT) needs vision/torch_fmha_sm120.cubin and vision/gate_bias_vl.safetensors",
        else => null,
    };
}

test "refusals name what to set" {
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoMaster).?, "TF_TP_MASTER") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoFreeSlot).?, "TF_DS_STREAMS") != null);
    try std.testing.expectEqual(@as(?[]const u8, null), explain(null, error.OutOfMemory));
}

test "PETABRIDGE spill P6: the tier's refusals from the environment (worlds 2 and 3; TF_DS_KV_SHARD unset or 0 only) come first and say what to set" {
    const E = struct {
        var kv: ?[]const u8 = null;
        var world: ?[]const u8 = null;
        var weights: ?[]const u8 = "rev-1";
        var engram: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_DS_ENGINE_ID")) return "engine";
            if (std.mem.eql(u8, name, "TF_DS_RECIPE_PATCHES")) return "patches";
            if (std.mem.eql(u8, name, "TF_DS_MODEL_REVISION")) return weights;
            if (std.mem.eql(u8, name, "TF_DS_KV_SHARD")) return kv;
            if (std.mem.eql(u8, name, "TF_TP_WORLD")) return world;
            if (std.mem.eql(u8, name, "TF_DS_ENGRAM")) return engram;
            return null;
        }
    };
    try spillPreflight(E.get); // (TF_TP_WORLD unset: 2)
    E.world = "3";
    try spillPreflight(E.get); // TP3
    E.world = "2";
    try spillPreflight(E.get);
    for ([_][]const u8{ "4", "1" }) |w| {
        E.world = w;
        try std.testing.expectError(error.SpillNeedsTwoOrThreeRanks, spillPreflight(E.get));
    }
    E.world = "x";
    try std.testing.expectError(error.BadWorld, spillPreflight(E.get));
    E.world = "3";
    E.kv = "0";
    try spillPreflight(E.get);
    E.kv = "";
    try spillPreflight(E.get);
    for ([_][]const u8{ "1", "maybe", "2" }) |v| {
        E.kv = v;
        try std.testing.expectError(error.SpillWithKvShard, spillPreflight(E.get));
    }
    E.kv = null;
    E.weights = null;
    try std.testing.expectError(error.NoWeightsId, spillPreflight(E.get));
    E.weights = "rev-1";
    E.engram = "/engram";
    try std.testing.expectError(error.NoEngramId, spillPreflight(E.get));
    E.engram = null;
    for ([_]anyerror{ error.SpillWithKvShard, error.SpillNeedsTwoOrThreeRanks, error.NoWeightsId, error.NoEngramId, error.SpillNotStarted, error.NoEngineId, error.SpillFingerprintMismatch, error.FollowerScanStalled, error.SpillRankZeroInServer }) |err|
        try std.testing.expect(explain(null, err) != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SpillNotStarted).?, "TF_DS_SPILL_DIR") != null);
    try std.testing.expect(spillWorld(2) and spillWorld(3) and !spillWorld(1) and !spillWorld(4));
    // the server's switches carried whole; the staging cap's floor
    const s = try spillSettings("/spill", 128, 300, 0.25, 8192, 4, 45000);
    try std.testing.expectEqualStrings("/spill", s.dir);
    try std.testing.expectEqual(@as(u32, 128), s.staging_mib);
    try std.testing.expect(s.enabled());
}
