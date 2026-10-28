//! One TP rank of DeepSeek-V4.1-Flash loaded for serving: the weights, kernels, caches and buffers the layer gate sets
//! up (layer_main.zig), for a pool of draft.max_streams streams (MultiDecoder's slots: each stream's window rings,
//! positional stores and drafter rings in its slot, its compressed rows in its extent of one shared plane), and the
//! primitives a lane round runs on them. Both ranks run the same primitives in the same order (lanes.zig sends them
//! to rank 1): a prompt's chunks into its stream's slot and extent (fill), one round over every stream's window and
//! the windows' taps absorbed (verify), and the drafter's batched pass (pass). Every primitive is the gates' own code
//! path (prompt.zig, round.zig, draft.zig) with no checks.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const rank_cache = @import("rank_cache.zig");
const weights = @import("weights.zig");
const prompt = @import("prompt.zig");
const prompt2d = @import("prompt2d.zig");
const round = @import("round.zig");
const draft = @import("draft.zig");
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const exl3_prefill = @import("exl3_prefill.zig");
const exl3_linear = @import("exl3_linear.zig");
const exl3_experts = @import("exl3_experts.zig");
const ops_mod = @import("ops.zig");
const exact = @import("exact.zig");
const cublas = @import("cublas.zig");
const comm_mod = @import("comm.zig");
const link = @import("link.zig");
const ring2d = @import("ring2d.zig");
const rdma = @import("rdma.zig");

/// The served prompt chunk (TF_DS_PREFILL_CHUNK): a prompt runs in chunks that start at its multiples.
pub const chunk_rows = 2048;
/// The pool's streams (--parallel 4).
pub const max_streams = draft.max_streams;
/// The pool's streams when unset (--parallel 4).
pub const max_streams_default = draft.default_streams;
/// The RDMA ring's slot (the served 4.25 MiB: a 16-row round's logits half fits).
pub const ring_bytes = 4456448;

/// TP3 remains opt-in until its GPU/numerical campaign passes. Both HTTP and
/// follower/lane entrypoints share this policy, before weight loading.
pub fn checkWorldMode(world: u32, experimental: ?[]const u8) !void {
    if (world == 2 or world == 4) return;
    if (world != 3) return error.NotPortedYet;
    if (!std.mem.eql(u8, experimental orelse "", "1")) return error.ExperimentalTP3Required;
}

fn experimentalTP3() ?[]const u8 {
    return if (std.c.getenv("TF_DS_EXPERIMENTAL_TP3")) |v| std.mem.span(v) else null;
}

test "experimental TP3 requires exact opt-in while legacy worlds remain available" {
    try checkWorldMode(2, null);
    try checkWorldMode(4, null);
    try checkWorldMode(3, "1");
    for ([_]?[]const u8{ null, "", "0", "true", "01", "1 " }) |value|
        try std.testing.expectError(error.ExperimentalTP3Required, checkWorldMode(3, value));
    for ([_]u32{ 0, 1, 5, 6 }) |world|
        try std.testing.expectError(error.NotPortedYet, checkWorldMode(world, "1"));
}

/// Avoid silently falling back to NCCL for a full TP3 verification window's
/// head logits. Pinned host send/receive buffers scale with slots*(world+1).
pub fn tp3RingBytes(local_vocab: usize, rows: usize) !usize {
    if (local_vocab == 0 or local_vocab % 128 != 0 or rows < round.default_rows or rows > round.max_rows) return error.BadRoundRows;
    const values = try std.math.mul(usize, local_vocab, rows);
    const bytes = try std.math.mul(usize, values, 4);
    return @max(ring_bytes, bytes); // divisible by64 because vocab is128-aligned
}

test "TP3 RDMA capacity includes C8 K5 head without reducing legacy floor" {
    try std.testing.expectEqual(ring_bytes, try tp3RingBytes(43136, 16));
    const bytes = try tp3RingBytes(43136, 48);
    try std.testing.expectEqual(@as(usize, 8282112), bytes);
    try std.testing.expect(bytes >= 48 * 43136 * 4);
    try std.testing.expectEqual(@as(usize, 0), bytes % 64);
    try std.testing.expectEqual(@as(usize, 61210624), (bytes - ring_bytes) * 4 * (3 + 1));
    try std.testing.expectError(error.BadRoundRows, tp3RingBytes(43136, 49));
    try std.testing.expectError(error.BadRoundRows, tp3RingBytes(43135, 48));
}

pub const Options = struct {
    model_dir: []const u8, // config.json
    cache_dir: []const u8, // the lane's per-rank weight file (TF_DS_RANK_CACHE)
    kit_dir: []const u8, // a gate dir's layout: aot/, cubins/, rope-{plain,compressed}-{cos,sin}.f32, engram.json
    aot_dir: ?[]const u8 = null, // the recorded Triton set (aot.json and its cubins; null: kit_dir/aot)
    rank: u32,
    world: u32,
    master: [4]u8,
    port: u16,
    engram_dir: ?[]const u8 = null, // the Engram tables (none: a checkpoint without Engram layers)
    token_map: ?[]const u8 = null, // the compressed token map (the lane's JSON cache)
    pool: usize = 1 << 18, // positions of the streams' shared plane (their extents together)
    drafts: bool = true, // load the DSpark drafter
    rdma_devices: ?[]const u8 = null, // the decode gathers over RDMA on these devices (comma separated), else NCCL: TP2's
    // ring (rdma.zig), or at world 4 the 2D split's decode-size exchanges over its rings (ring2d.zig)
    rdma_kernels: ?[]const u8 = null, // world 4: the rings' kernels' image (rdma_gather.cu's fatbin; null: the embedded one)
    graphs: bool = false, // the rounds' stretches as CUDA graphs (round.Graphs)
    side: bool = false, // the mixes' side work on a stream of its own (round.Round.useSide)
    prefetch: bool = false, // the paced L2 prefetch (round.Round.usePrefetch)
    engram_aio: bool = false, // a round's Engram reads by Linux AIO on O_DIRECT (else the reader pool)
    arena_bytes: usize = 0, // the device arena's first block (0: it grows from 256 MiB blocks as the buffers ask)
    round_rows: usize = round.default_rows, // a round's rows at most (TF_DS_ROUND_ROWS; the four-node lane's 48)
    streams: usize = draft.default_streams, // the pool's streams (--parallel; the four-node lane's 16)
    round_ms: ?[]const f64 = null, // the lane core's round costs by rows (TF_DS_ROUND_MS; null: the served ROUND_MS)
    vision_bias: ?[]const u8 = null, // gate_bias_vl.safetensors (null: the kit's vision/): the image-span routing bias
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    host: std.heap.ArenaAllocator, // allocations that live as long as the model
    io: std.Io,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    cfg: Config,
    rank: u32,
    world: u32,
    streams: usize, // the pool's streams (Options.streams)
    round_ms: []const f64, // the lane core's round costs by rows (Options.round_ms, else draft.round_ms)
    pool_cap: usize,
    cache_dir: std.Io.Dir,
    cache_file: std.Io.File,
    ix: rank_cache.Index,
    w: weights.Weights,
    fan: prompt2d.Fan,
    comm: comm_mod.Comm,
    rdma_mod: ?cuda.Module = null, // the RDMA kernels (TP2's ring or world 4's rings)
    ring: ?*rdma.Ring = null,
    set: cuda.aot.Set,
    pf: exl3_prefill.Kernels,
    lg: exl3_linear.Kernels,
    exk: exl3_experts.Kernels,
    ops: ops_mod.Ops,
    ex: exact.Exact,
    rope_buf: cuda.DeviceBuffer,
    arena: prompt.Arena,
    eng: prompt.Engine,
    ch: prompt.Chunk,
    caches: prompt.Caches,
    blas: cublas.Blas,
    rings: []u64, // each layer's window rings, max_streams slots
    ring_view: []u64, // a filling stream's slot of them
    rd: round.Round,
    graphs: ?round.Graphs = null,
    pgraphs: ?draft.PassGraphs = null, // the drafter pass as CUDA graphs (with graphs)
    dpool: ?draft.Pool = null,
    dr: ?draft.Drafter = null,
    tables: ?engram_io.Tables = null,
    epool: ?*engram_io.Pool = null,
    aio: ?*engram_aio.Aio = null,
    paio: ?*engram_aio.Aio = null, // the prompt chunks' Engram reads, on a ring of their own
    eh: ?prompt.EngramHost = null,
    host_pos: []i64,
    ids64: []i64,
    vocab: usize,
    logits: []f32, // round.max_rows rows of logits on the host
    amax: u64, // u32 [round.max_rows]: a round's greedy tokens on the device (roundArgmax)
    amax_host: [round.max_rows]u32 = undefined,
    // the stream a fill is running: its slot and its view of the caches (its extent from base)
    fill_slot: usize = 0,
    // the prompt a fill's chunks come from (the caller's, set for the fill) and the thread reading the next chunk's
    // Engram rows during this one (fillChunk), joined before the prompt can change (joinAhead)
    fill_seq: []const i32 = &.{},
    ahead: ?std.Thread = null,
    view: prompt.Caches = undefined,
    // the filling prompt's image spans and their rows (setImages; cleared by fillBegin)
    spans: std.ArrayList(prompt.Span) = .empty,
    img_buf: ?cuda.DeviceBuffer = null,
    // profiling (lanes' --profile): the last round's forward and absorb, synchronized apart (ns)
    prof: bool = false,
    timer: ?*round.PhaseTimer = null, // a profile's GPU time by phase (eager rounds)
    ptimer: ?*round.PhaseTimer = null, // the same for prompt chunks (fills), a chunk a 'round'
    t_enqueue: u64 = 0, // the host's time to issue the forward (the GPU idle before it)
    // Model.useClock: the rounds' GPU clock gaps summed (ns; gap i = clock i + 1 - clock i: the forward's start to
    // the first stretch, then each stretch, then the wait before the next), and the rounds counted
    clk_ph: [256]u64 = @splat(0), // by the phase of the clock that ends it
    clk_cnt: [256]u64 = @splat(0),
    clk_rounds: u64 = 0,
    t_forward: u64 = 0,
    t_absorb: u64 = 0,
    // world 4: the 2D split's exchanges (prompt2d.zig, TP4's), with the RDMA rings when asked
    two: ?prompt2d.Two = null,
    rdma_rings: ?ring2d.Rings = null,

    /// Loads rank o.rank's weights from the lane's rank cache, links the ranks (rank 0 listens on o.port), opens NCCL
    /// and sets up every buffer. Needs `ctx` current on this thread.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, o: Options) !*Model {
        try checkWorldMode(o.world, experimentalTP3());
        if (o.rank >= o.world) return error.UnsupportedRank;
        if (o.round_rows < round.default_rows or o.round_rows > round.max_rows) return error.BadRoundRows;
        if (o.world == 3) std.debug.print("EXPERIMENTAL TP3: GPU/numerical qualification pending; rank {d}\n", .{o.rank});
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        if (std.c.getenv("TF_DS_REPLAY_FLOOR")) |v| prompt.replay_floor = std.mem.eql(u8, std.mem.span(v), "1"); // every rank alike
        prompt.short_continuation_gather = if (std.c.getenv("TF_DS_CACHE_SHORT_CONTINUATION")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false; // same opt-in on every rank
        m.spans = .empty;
        m.img_buf = null;
        m.host = std.heap.ArenaAllocator.init(gpa);
        errdefer m.host.deinit();
        const a = m.host.allocator();
        m.io = io;
        m.ctx = ctx;
        m.rank = o.rank;
        m.world = o.world;
        if (o.streams == 0 or o.streams > max_streams) return error.BadStreams;
        m.streams = o.streams;
        m.round_ms = o.round_ms orelse &draft.round_ms;
        if (m.round_ms.len == 0) return error.BadRoundMs;
        m.pool_cap = o.pool;
        m.fill_slot = 0;
        m.prof = false;
        m.timer = null;
        m.ptimer = null;
        m.t_enqueue = 0;
        m.t_forward = 0;
        m.t_absorb = 0;
        m.dpool = null;
        m.dr = null;
        m.tables = null;
        m.epool = null;
        m.aio = null;
        m.paio = null;
        m.eh = null;
        m.rdma_mod = null;
        m.ring = null;
        m.graphs = null;
        m.pgraphs = null;
        m.two = null;
        m.rdma_rings = null;
        const d = ctx.d;
        m.cfg = try Config.read(a, io, o.model_dir);
        try m.cfg.checkWorld(o.world);
        const c = &m.cfg;
        m.stream = try cuda.Stream.init(d, true);
        errdefer m.stream.deinit();

        // the weights (and the DSpark blocks) from the lane's rank cache
        m.cache_dir = try std.Io.Dir.cwd().openDir(io, o.cache_dir, .{ .iterate = true });
        errdefer m.cache_dir.close(io);
        const cache_name = try rank_cache.find(a, io, m.cache_dir, o.rank, o.world);
        m.ix = try rank_cache.open(gpa, io, m.cache_dir, cache_name);
        errdefer m.ix.deinit();
        m.cache_file = try m.cache_dir.openFile(io, cache_name, .{});
        errdefer m.cache_file.close(io);
        const sp = try prompt2d.split(o.rank, o.world);
        m.w = try weights.load(gpa, d, .{ .cache = .{ .file = m.cache_file, .index = &m.ix, .io = io } }, c.*, sp, o.drafts);
        errdefer m.w.deinit();
        if (o.drafts and m.w.dspark == null) return error.NoDrafter;

        // the ranks' link and NCCL (rank 0 listens: rank 1 retries its connect while rank 0 loads)
        m.fan = try prompt2d.Fan.open(io, o.master, o.port, o.rank, o.world);
        errdefer m.fan.close();
        m.comm = try m.fan.comm(o.rank, o.world);
        errdefer m.comm.deinit();
        // TP2: the decode gathers' RDMA ring (both ranks connect before either returns)
        if (sp.pair == null and o.rdma_devices != null) {
            const list = o.rdma_devices.?;
            if (!cuda.kernels.available) return error.NoKernelImages;
            m.rdma_mod = try cuda.Module.load(d, cuda.kernels.dsv41_rdma);
            var devices: std.ArrayList([]const u8) = .empty;
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |dev| try devices.append(a, dev);
            const bytes = if (o.world == 3) try tp3RingBytes(m.w.head.n, o.round_rows) else ring_bytes;
            m.ring = try openRingSized(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, &m.comm, m.stream, bytes);
        }
        errdefer if (m.ring) |r| r.destroy(gpa);
        errdefer if (m.rdma_mod) |*x| x.unload();

        // kernels: the recorded Triton set, the served extension cubins, the torch-op images
        m.set = try cuda.aot.Set.load(gpa, io, d, ctx.device, o.aot_dir orelse try std.fs.path.join(a, &.{ o.kit_dir, "aot" }));
        errdefer m.set.deinit();
        m.pf = try exl3_prefill.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear.cubin"));
        errdefer m.pf.unload();
        m.lg = try exl3_linear.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear_grouped.cubin"));
        errdefer m.lg.unload();
        m.exk = try exl3_experts.Kernels.load(d, try readKit(a, io, o.kit_dir, "experts.cubin"), try readKit(a, io, o.kit_dir, "experts_cb.cubin"), exl3_linear.codebook_mul1);
        errdefer m.exk.unload();
        if (!cuda.kernels.available) return error.NoKernelImages;
        m.ops = try ops_mod.Ops.load(d, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
        errdefer m.ops.unload();
        m.ex = try exact.Exact.load(d, cuda.kernels.dsv41_torch);
        errdefer m.ex.unload();

        // the RoPE tables (the served build's bits: torch CPU pow and polar), every position of the pool
        const half = c.rope_dim / 2;
        const rope_rows = o.pool + round.max_rows + 1;
        m.rope_buf = try cuda.DeviceBuffer.alloc(d, 4 * rope_rows * half * 4);
        errdefer m.rope_buf.free();
        {
            const part = try gpa.alloc(u8, rope_rows * half * 4);
            defer gpa.free(part);
            for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }, 0..) |name, j| {
                var f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ o.kit_dir, name }), .{});
                defer f.close(io);
                if (try f.readPositionalAll(io, part, 0) != part.len) return error.ShortRopeTable; // a table of fewer rows than the pool
                try m.rope_buf.upload(j * part.len, part);
            }
        }
        const tbl = rope_rows * half * 4;
        const rope: prompt.Rope = .{ .cos = m.rope_buf.ptr, .sin = m.rope_buf.ptr + tbl };
        const rope_c: prompt.Rope = .{ .cos = m.rope_buf.ptr + 2 * tbl, .sin = m.rope_buf.ptr + 3 * tbl };

        m.arena = try prompt.Arena.init(d, o.arena_bytes);
        errdefer m.arena.deinit();
        // world 4 (layer_main's setup): the 2D exchanges' buffers, and with rdma_devices their rings (whose infos go
        // round over NCCL world 4)
        errdefer if (m.rdma_rings) |*x| x.close(gpa);
        if (sp.pair != null) {
            m.two = try prompt2d.Two.init(c, &m.w, &m.arena, sp, chunk_rows);
            if (o.rdma_devices) |devs| {
                const image: []const u8 = if (o.rdma_kernels) |path|
                    try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(1 << 26), .@"16", null)
                else if (cuda.kernels.available) cuda.kernels.dsv41_rdma else return error.NoRdmaKernels;
                m.rdma_mod = try cuda.Module.load(d, image);
                var devices: std.ArrayList([]const u8) = .empty;
                var dit = std.mem.splitScalar(u8, devs, ',');
                while (dit.next()) |x| try devices.append(a, x);
                m.rdma_rings = try ring2d.Rings.open(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, .{ .max_bytes = ring_bytes, .gid_index = try rdma.servedGidIndex() }, &m.comm, m.stream);
                m.two.?.rings = &m.rdma_rings.?;
            }
        }
        m.eng = .{ .d = d, .s = m.stream, .t = .{ .set = &m.set, .stream = m.stream }, .blas = undefined, .comm = &m.comm, .pf = &m.pf, .lin = &m.lg, .ex = &m.exk, .ops = &m.ops, .exact = &m.ex, .c = c, .w = &m.w, .world = sp.world, .plain = rope, .compressed = rope_c, .ring = m.ring, .two = if (m.two) |*t| t else null };
        m.eng.round_rows = o.round_rows;
        m.ch = try prompt.Chunk.init(&m.eng, &m.arena, chunk_rows, o.pool);
        m.caches = try prompt.Caches.init(&m.eng, &m.arena, o.pool, m.streams);
        m.blas = try cublas.Blas.open(m.stream, m.ch.blas_ws);
        errdefer m.blas.close();
        m.eng.blas = &m.blas;
        m.rings = try a.alloc(u64, c.layers);
        m.ring_view = try a.alloc(u64, c.layers);
        for (m.rings) |*r| {
            r.* = try m.arena.take(m.streams * m.eng.ringBytes());
            try d.check(d.api.cuMemsetD8Async(r.*, 0, m.streams * m.eng.ringBytes(), m.stream.handle), "cuMemsetD8Async");
        }
        m.rd = try round.Round.init(&m.eng, &m.arena, a, o.pool);
        m.amax = try m.arena.take(round.max_rows * 4);
        if (o.graphs) m.graphs = round.Graphs.init(gpa);
        if (o.side) try m.rd.useSide(&m.eng);
        if (o.prefetch) try m.rd.usePrefetch(&m.eng);
        if (o.drafts) {
            m.dpool = try draft.Pool.init(&m.eng, &m.arena, m.streams);
            m.dr = try draft.Drafter.init(&m.eng, &m.arena, sp);
            if (o.graphs) {
                m.pgraphs = draft.PassGraphs.init(gpa);
                m.dr.?.graphs = &m.pgraphs.?;
            }
        }
        try m.stream.synchronize();
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize"); // the setup's legacy-stream copies too

        // Engram on the host: the compressed token map, the multipliers, the tables and 64 readers
        if (o.engram_dir) |edir| {
            const map_text = try std.Io.Dir.cwd().readFileAlloc(io, o.token_map orelse return error.NoTokenMap, a, .limited(1 << 26));
            const map = try std.json.parseFromSliceLeaky([]i32, a, map_text, .{});
            const EngramJson = struct { multipliers: [][]i64 };
            const ej_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ o.kit_dir, "engram.json" }), a, .limited(1 << 26));
            const ej = try std.json.parseFromSliceLeaky(EngramJson, a, ej_text, .{ .ignore_unknown_fields = true });
            var mult: [engram.max_layers][engram.max_ngram]i64 = @splat(@splat(0));
            for (ej.multipliers, 0..) |row, l| for (row, 0..) |v, k| {
                mult[l][k] = v;
            };
            m.tables = try engram_io.Tables.open(gpa, io, edir);
            m.epool = try engram_io.Pool.init(gpa, io, 64);
            m.eh = try prompt.EngramHost.init(a, c, engram.Hasher.init(c.*, map, mult), &m.tables.?, m.epool.?, sp.rank, sp.world, chunk_rows);
            if (o.engram_aio) {
                // a round's reads, 2 x 12 x its rows at most: 1,024 slots for each 16 rows of the round-row setting
                m.aio = try engram_aio.Aio.init(gpa, io, 1024 * ((o.round_rows + 15) / 16));
                m.eh.?.aio = m.aio;
                // a prompt chunk's reads (2 x 12 x 2,048 at most): a ring of 4,096 slots (32 MiB), 1,024 rows a batch
                m.paio = try engram_aio.Aio.init(gpa, io, 4096);
                m.eh.?.paio = m.paio;
            }
        } else if (c.engram_layers.slice().len > 0) return error.NoEngramTables;

        // a vision checkpoint's image routing bias (without it a prompt with images is refused; text runs as ever)
        if (c.vision) {
            const path = o.vision_bias orelse try std.fs.path.join(a, &.{ o.kit_dir, "vision", "gate_bias_vl.safetensors" });
            m.loadVisionBias(path) catch |err| if (o.vision_bias != null or err != error.FileNotFound) return err;
        }

        m.host_pos = try a.alloc(i64, chunk_rows);
        m.ids64 = try a.alloc(i64, chunk_rows);
        m.vocab = sp.world * (if (m.two) |t| t.hw[0] + t.hw[1] else m.w.head.n); // 2D: both pairs' vocabulary parts
        m.logits = try a.alloc(f32, round.max_rows * m.vocab);
        // PETABRIDGE (k2a-pb1): spill layout self-check (P4), every rank, every start: one "spill layout: " line; never
        // fails the start; host-only unless TF_DS_SPILL_SELF_CHECK=1
        m.spillStartupCheck();
        return m;
    }

    pub fn close(m: *Model) void {
        m.joinAhead();
        m.stream.synchronize() catch {};
        if (m.img_buf) |*b| b.free();
        m.spans.deinit(m.gpa);
        if (m.graphs) |*g| g.deinit();
        if (m.pgraphs) |*g| g.deinit();
        if (m.timer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        if (m.ptimer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        m.rd.dropSide(&m.eng);
        m.rd.dropPrefetch();
        if (m.aio) |x| x.deinit();
        if (m.paio) |x| x.deinit();
        if (m.epool) |p| p.deinit(m.gpa);
        if (m.tables) |*t| t.close();
        if (m.rdma_rings) |*r| r.close(m.gpa);
        m.blas.close();
        m.arena.deinit();
        m.rope_buf.free();
        m.ex.unload();
        m.ops.unload();
        m.exk.unload();
        m.lg.unload();
        m.pf.unload();
        m.set.deinit();
        if (m.ring) |r| {
            r.stop();
            r.destroy(m.gpa);
        }
        if (m.rdma_mod) |*x| x.unload();
        m.comm.deinit();
        m.fan.close();
        m.w.deinit();
        m.cache_file.close(m.io);
        m.ix.deinit();
        m.cache_dir.close(m.io);
        m.stream.deinit();
        m.host.deinit();
        m.gpa.destroy(m);
    }

    /// TP2's RDMA ring for the decode gathers: created on both ranks, the queue-pair infos exchanged over NCCL,
    /// connected, and started once both are (an all-gather as the barrier).
    pub fn openRing(gpa: std.mem.Allocator, d: *const cuda.Driver, k: rdma.Kernels, devices: []const []const u8, rank: u32, comm: *const comm_mod.Comm, stream: cuda.Stream) !*rdma.Ring {
        return openRingSized(gpa, d, k, devices, rank, comm, stream, ring_bytes);
    }

    pub fn openRingSized(gpa: std.mem.Allocator, d: *const cuda.Driver, k: rdma.Kernels, devices: []const []const u8, rank: u32, comm: *const comm_mod.Comm, stream: cuda.Stream, bytes: usize) !*rdma.Ring {
        const world: usize = comm.world;
        if (world < 2 or world > rdma.max_ranks or rank >= world) return error.BadRdmaWorld;
        const mask_text = if (std.c.getenv("TF_RDMA_PEER_MASKS")) |v| std.mem.span(v) else null;
        const peer_masks = try rdma.parsePeerMasks(mask_text, rank, comm.world, @intCast(devices.len));
        const r = try rdma.Ring.create(gpa, d, k, devices, rank, comm.world, .{ .max_bytes = bytes, .gid_index = try rdma.servedGidIndex(), .peer_masks = peer_masks });
        errdefer r.destroy(gpa);
        const per = @sizeOf(rdma.Info);
        var mine: [rdma.max_ranks]rdma.Info = undefined;
        for (0..world) |peer_rank| mine[peer_rank] = r.info(@intCast(peer_rank));
        const send_bytes = world * per;
        var dev = try cuda.DeviceBuffer.alloc(d, (world + 1) * send_bytes);
        defer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(mine[0..world]));
        try comm.allGather(dev.ptr, dev.ptr + send_bytes, send_bytes, .u8, stream);
        try stream.synchronize();
        var all: [rdma.max_ranks * rdma.max_ranks]rdma.Info = undefined;
        try dev.download(send_bytes, std.mem.sliceAsBytes(all[0 .. world * world]));
        var peers: [rdma.max_ranks]rdma.Info = undefined;
        for (0..world) |peer_rank| peers[peer_rank] = all[peer_rank * world + rank];
        try r.connect(peers[0..world]);
        // every queue pair ready on both ranks before either sends
        try comm.allGather(dev.ptr, dev.ptr + send_bytes, 1, .u8, stream);
        try stream.synchronize();
        try r.start();
        return r;
    }

    fn readKit(a: std.mem.Allocator, io: std.Io, kit: []const u8, name: []const u8) ![]const u8 {
        return std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ kit, "cubins", name }), a, .limited(1 << 28), .@"16", null);
    }

    /// The link to the other rank (rank 0's to rank 1, rank 1's to rank 0).
    pub fn peer(m: *const Model) link.Link {
        return m.fan.links[0].?;
    }

    /// Rank 0's links to its followers: rank 1's, or ranks 1-3's on the four-node split.
    pub fn followers(m: *const Model) []const ?link.Link {
        return m.fan.links[0 .. m.world - 1];
    }

    pub fn drafting(m: *const Model) bool {
        return m.dr != null;
    }

    // -- the primitives (both ranks, in the same order) ----------------------------------------------------------

    /// A stream's prompt is about to fill pool slot `slot`, its compressed rows in its extent from position `base`.
    /// Wait for the next chunk's Engram read-ahead, if one runs (before the prompt it reads changes or goes).
    pub fn joinAhead(m: *Model) void {
        if (m.ahead) |th| th.join();
        m.ahead = null;
    }

    /// A kept prompt state's own bytes (the prompt cache's snapshots; the compressed plane stays in the stream's
    /// extent): one slot's window ring of every layer and positional stores of every ratio-2 layer.
    pub fn snapBytes(m: *const Model) usize {
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var n: usize = m.rings.len * m.eng.ringBytes();
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            if (k != 0) n += pb;
            if (sc != 0) n += pb;
        }
        // A complete prompt also resumes its exact head distribution and drafter rings.
        if (m.dpool) |p| n += p.stages * p.ring * p.hd * 2;
        n += m.vocab * 4;
        return n;
    }

    /// Slot `slot`'s own state into `buf` (save: true) or back from it, in the stream's order (no wait): with the
    /// extent's compressed rows, the state a prompt pass stands in at a chunk end.
    pub fn snapCopy(m: *Model, slot: usize, buf: u64, save: bool) !void {
        const rb = m.eng.ringBytes();
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var at = buf;
        for (m.rings) |r| {
            try m.copyState(r + slot * rb, at, rb, save);
            at += rb;
        }
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            for ([_]u64{ k, sc }) |live| if (live != 0) {
                try m.copyState(live + slot * pb, at, pb, save);
                at += pb;
            };
        }
        if (m.dpool) |p| {
            const db = p.ring * p.hd * 2;
            for (p.rings[0..p.stages]) |r| {
                try m.copyState(r + slot * db, at, db, save);
                at += db;
            }
        }
        try m.copyState(m.promptLogits(), at, m.vocab * 4, save);
    }

    /// The compressed rows of positions [0, at) of the extent at `from` copied into the extent at `to` (a kept prompt
    /// resumed in another extent while a live stream holds its own), in the stream's order: every kv-source layer's
    /// codes and scales and its index keys, the rows view() places at those positions.
    pub fn copyExtent(m: *Model, from: usize, to: usize, at: usize) !void {
        const c = m.eng.c;
        const cs = &m.caches;
        for (m.eng.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            if (from % r != 0 or to % r != 0) return error.BadView;
            const n = at / r;
            try m.copySpan(cs.comp_codes[i], from / r, to / r, n, c.head_dim / 2);
            try m.copySpan(cs.comp_scales[i], from / r, to / r, n, c.head_dim / 16);
            if (lay.idx_wk != null) {
                try m.copySpan(cs.idx_codes[i], from / r, to / r, n, c.index_head_dim / 2);
                try m.copySpan(cs.idx_scales[i], from / r, to / r, n, c.index_head_dim / 32);
            }
        }
    }

    fn copySpan(m: *Model, plane: u64, from: usize, to: usize, n: usize, row: usize) !void {
        const d = m.ctx.d;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(plane + to * row, plane + from * row, n * row, m.stream.handle), "cuMemcpyDtoDAsync");
    }

    fn copyState(m: *Model, live: u64, kept: u64, bytes: usize, save: bool) !void {
        const d = m.ctx.d;
        const dst = if (save) kept else live;
        const src = if (save) live else kept;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, m.stream.handle), "cuMemcpyDtoDAsync");
    }

    // -- PETABRIDGE (k2a-pb1): spill v2's checkpoint state (MiaAI-Lab 0016's model side, spill port step P4) ---------
    // A kept point's state on this rank as the spill tier stores it: the descriptor derived from the loaded weights and
    // the live allocation, and the planes' addresses and copies for spill.Packer. Nothing here runs on the served path
    // except spillStartupCheck's host-only layout check at the end of open (one log line); the tier is not wired (P6).
    // Our KV is replicated on every rank (no TF_DS_KV_SHARD), so the layout is always `.replicated` (0016's form; 0019's
    // owner_sharded branch is not taken).

    pub const spill = @import("spill_state.zig");

    comptime {
        std.debug.assert(draft.max_streams <= spill.max_slots);
    }

    /// This rank's state planes as spill.Packer reads and writes them: their device addresses (the whole pool's
    /// compressed planes, a slot's rings and raw stores) and copies in the model's stream's order.
    pub const SpillDev = struct {
        m: *Model,

        pub const Event = cuda.Event;

        /// Row 0 of a block plane over the whole pool (the extent at base b starts b / ratio rows in, as view() puts it).
        pub fn blockPlane(dv: SpillDev, p: spill.Plane) u64 {
            const cs = &dv.m.caches;
            return switch (p.kind) {
                .comp_codes => cs.comp_codes[p.layer],
                .comp_scales => cs.comp_scales[p.layer],
                .idx_codes => cs.idx_codes[p.layer],
                .idx_scales => cs.idx_scales[p.layer],
                .ring, .raw_kv, .raw_score => 0,
            };
        }

        /// Slot `slot`'s rows of a small plane (snapCopy's addresses).
        pub fn slotPlane(dv: SpillDev, p: spill.Plane, slot: usize) u64 {
            const m = dv.m;
            const base: u64 = switch (p.kind) {
                .ring => m.rings[p.layer],
                .raw_kv => m.caches.raw_kv[p.layer],
                .raw_score => m.caches.raw_score[p.layer],
                .comp_codes, .comp_scales, .idx_codes, .idx_scales => 0,
            };
            return if (base == 0) 0 else base + slot * p.slotBytes();
        }

        pub fn capacity(dv: SpillDev) usize {
            return dv.m.caches.cap;
        }

        pub fn slots(dv: SpillDev) usize {
            return dv.m.streams;
        }

        pub fn d2h(dv: SpillDev, dst: []u8, src: u64) !void {
            const d = dv.m.ctx.d;
            try d.check(d.api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, dv.m.stream.handle), "cuMemcpyDtoHAsync");
        }

        pub fn h2d(dv: SpillDev, dst: u64, src: []const u8) !void {
            const d = dv.m.ctx.d;
            try d.check(d.api.cuMemcpyHtoDAsync_v2(dst, src.ptr, src.len, dv.m.stream.handle), "cuMemcpyHtoDAsync");
        }

        pub fn fill(dv: SpillDev, dst: u64, value: u8, bytes: usize) !void {
            const d = dv.m.ctx.d;
            try d.check(d.api.cuMemsetD8Async(dst, value, bytes, dv.m.stream.handle), "cuMemsetD8Async");
        }

        pub fn record(dv: SpillDev, ev: cuda.Event) !void {
            try ev.record(dv.m.stream);
        }

        /// Whether the copies before `ev` completed (spill_tier polls it; a device error reads as done, and the next
        /// copy on the stream reports it).
        pub fn done(_: SpillDev, ev: cuda.Event) bool {
            return ev.done() catch true;
        }

        /// The copies before `ev` completed (a cancel, a drop, a shutdown: bounded by what was enqueued).
        pub fn wait(_: SpillDev, ev: cuda.Event) void {
            ev.synchronize() catch {};
        }
    };

    /// PETABRIDGE (k2a-pb1): the bytes our Snap holds past MiaAI's small state. k2a's snapBytes/snapCopy add, after the
    /// rings and raw stores (the spill small state, in the same order), every DSpark stage's ring of the slot and then
    /// the prompt logits, so that a "complete" point resumes with no replay. A spilled point is always pre-replay
    /// (spill.eligible: at % 2048 == 0, at <= len - window), so this tail is never stored: the resumed pass's bounded
    /// replay absorbs the drafter's whole read set (spill.checkReplayRebuilds; spillCheckResumed checks it on the pool)
    /// and its final chunk's head writes the logits. Decision (a) of the port plan, section 3.3.
    pub const SpillSnapTail = struct {
        draft: usize, // snapCopy's dpool part: stages x ring x head_dim x 2 (0 without the drafter)
        logits: usize, // snapCopy's last copy: vocab x 4

        pub fn bytes(t: SpillSnapTail) usize {
            return t.draft + t.logits;
        }
    };

    /// PETABRIDGE (k2a-pb1): this model's Snap tail (snapBytes' own terms after its rings and raw stores).
    pub fn spillSnapTail(m: *const Model) SpillSnapTail {
        return .{ .draft = if (m.dpool) |p| p.stages * p.ring * p.hd * 2 else 0, .logits = m.vocab * 4 };
    }

    /// PETABRIDGE (k2a-pb1): MiaAI's descriptor derivation alone (spillLayout's first half): the loaded layers as
    /// Caches.init reads them, no check against the allocation.
    fn spillLayoutOf(m: *const Model) !spill.StateLayout {
        const layers = m.eng.w.layers;
        if (layers.len > spill.max_layers) return error.SpillLayoutMismatch;
        var shapes: [spill.max_layers]spill.LayerShape = undefined;
        for (layers, shapes[0..layers.len]) |lay, *s| {
            const kv = lay.comp_wkv != null;
            s.* = .{ .ratio = if (kv) lay.ratio else 0, .kv = kv, .idx = kv and lay.idx_wk != null, .raw = kv and lay.ratio > 1 };
        }
        return spill.StateLayout.init(shapes[0..layers.len], spill.Geometry.of(m.eng.c, chunk_rows, prompt.ring_extra, prompt.raw_rows), .replicated);
    }

    /// PETABRIDGE (k2a-pb1): MiaAI's checks of a descriptor against the live allocation, each with its reason (null: it
    /// matches). The one change: small_bytes is snapBytes() less our Snap tail (MiaAI's own check is
    /// `small_bytes != snapBytes()`), so the small state is exactly a prefix of our Snap.
    fn spillLayoutFault(m: *const Model, l: *const spill.StateLayout) ?[]const u8 {
        if (m.rings.len != m.eng.w.layers.len) return "rings != layers";
        const snap = m.snapBytes();
        const tail = m.spillSnapTail().bytes();
        if (tail > snap or l.small_bytes != snap - tail) return "small state != snapBytes - drafter rings - logits";
        const dv: SpillDev = .{ .m = @constCast(m) };
        for (l.blockPlanes()) |p| if (dv.blockPlane(p) == 0) return "a block plane is not allocated";
        for (l.smallPlanes()) |p| {
            if (dv.slotPlane(p, 0) == 0) return "a small plane is not allocated";
            if (p.kind == .ring and p.slotBytes() != m.eng.ringBytes()) return "ring plane bytes != ringBytes";
        }
        return null;
    }

    /// The spill state descriptor of this rank, derived from the loaded layers (Caches.init's reading of them) and
    /// checked against the live allocation: snapCopy's bytes, a ring per layer, every plane present.
    /// PETABRIDGE (k2a-pb1): snapCopy's bytes less our Snap tail (spillSnapTail): the small state is the Snap's prefix.
    pub fn spillLayout(m: *const Model) !spill.StateLayout {
        const l = try m.spillLayoutOf();
        if (m.spillLayoutFault(&l) != null) return error.SpillLayoutMismatch;
        return l;
    }

    /// The pool's extent and slot generations (one counter a chunk of positions), sized once at setup.
    pub fn spillGenerations(m: *const Model, gpa: std.mem.Allocator) !spill.Generations {
        return spill.Generations.init(gpa, m.caches.cap, chunk_rows);
    }

    /// A packer over this rank's planes, `l` from spillLayout and `g` from spillGenerations (both the caller's, alive
    /// while it is used). Its calls enqueue on the model's stream and record the caller's event: no allocation, no wait.
    pub fn spillPacker(m: *Model, l: *const spill.StateLayout, g: *spill.Generations) spill.Packer(SpillDev) {
        return .{ .dev = .{ .m = m }, .layout = l, .gens = g, .rank = m.rank };
    }

    /// PETABRIDGE (k2a-pb1): a Snap buffer (snapBytes) whose first small_bytes a spilled point filled (Packer.unpackSnap)
    /// gets its tail zeroed, in the stream's order. A later snapCopy(.., false) of it then writes zeros, not another
    /// state's leftover bytes, into the slot's drafter rings and the prompt logits, which the resumed pass rebuilds
    /// (SpillSnapTail). A restore straight into a slot (Packer.unpackSmall) never touches the tail and needs no call.
    pub fn spillClearSnapTail(m: *Model, l: *const spill.StateLayout, buf: u64) !void {
        const snap = m.snapBytes();
        if (l.small_bytes > snap) return error.SpillLayoutMismatch;
        if (snap == l.small_bytes) return;
        const d = m.ctx.d;
        try d.check(d.api.cuMemsetD8Async(buf + l.small_bytes, 0, snap - l.small_bytes, m.stream.handle), "cuMemsetD8Async");
    }

    /// After a restored prompt's fill: the drafter's rings were rebuilt by its replay through the prompt's end
    /// (spill.checkReplayRebuilds' rule, here as the pool records it), so the first draft reads no unwritten row.
    pub fn spillCheckResumed(m: *const Model, slot: usize, len: usize) !void {
        if (slot >= m.streams) return error.BadSlot;
        if (m.dpool) |*p| if (p.absorbed[slot] != len) return error.DraftNotRebuilt;
    }

    /// A compressed plane as copyExtent copies it: its address, row bytes (copyExtent's own widths from the config)
    /// and ratio.
    const SpillRefPlane = struct { plane: u64, row: usize, ratio: usize };

    /// copyExtent's planes in its order (every kv layer: codes, scales, then the index keys' codes and scales), read
    /// from the loaded layers and the config, not from the spill layout.
    fn spillRefPlanes(m: *const Model, out: *[4 * spill.max_layers]SpillRefPlane) ![]SpillRefPlane {
        const c = m.eng.c;
        const cs = &m.caches;
        var k: usize = 0;
        for (m.eng.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            if (k + 4 > out.len) return error.SpillLayoutMismatch;
            const r: usize = lay.ratio;
            out[k] = .{ .plane = cs.comp_codes[i], .row = c.head_dim / 2, .ratio = r };
            out[k + 1] = .{ .plane = cs.comp_scales[i], .row = c.head_dim / 16, .ratio = r };
            k += 2;
            if (lay.idx_wk != null) {
                out[k] = .{ .plane = cs.idx_codes[i], .row = c.index_head_dim / 2, .ratio = r };
                out[k + 1] = .{ .plane = cs.idx_scales[i], .row = c.index_head_dim / 32, .ratio = r };
                k += 2;
            }
        }
        return out[0..k];
    }

    /// Stage 2's completeness check on one instance (a GPU tool, not a serving path), against the engine's own resume
    /// primitives rather than the layout itself. The point at `at` of slot `from_slot` and the extent at `from`:
    ///   - reference: copyExtent(from, ref, at) into a third extent and snapCopy(from_slot) into a scratch Snap buffer;
    ///   - packed, unpacked into slot `to_slot` and the extent at `to` (both overwritten, as is `ref`), packed again.
    /// Then (round trip) both packs are equal; (blocks) the restored rows of `to` equal copyExtent's rows at `ref`, and
    /// the packed blocks, cut with offsets computed from copyExtent's loop and the config's row widths, equal them;
    /// (small) a snapCopy of the restored slot equals the reference Snap, and the packed small state equals it too.
    /// Every argument is checked before anything is enqueued; its own pinned staging, device scratch and event, none
    /// freed before the stream is idle.
    /// PETABRIDGE (k2a-pb1): unchanged from 0016. Its Snap comparisons read the first small_bytes of each snapCopy, which
    /// with our Snap tail (SpillSnapTail) is exactly the small state: the tail (drafter rings, logits) is not compared.
    pub fn spillSelfCheck(m: *Model, gpa: std.mem.Allocator, from_slot: usize, from: usize, to_slot: usize, to: usize, ref: usize, at: usize) !void {
        if (from_slot >= m.streams or to_slot >= m.streams or from_slot == to_slot) return error.BadSlot;
        if (at == 0) return error.BadRange;
        for ([_]usize{ from, to, ref, at }) |v| if (v % chunk_rows != 0) return error.BadRange;
        const cap = m.caches.cap;
        for ([_]usize{ from, to, ref }) |b| if (b >= cap or at > cap - b) return error.BadRange;
        for ([_][2]usize{ .{ from, to }, .{ from, ref }, .{ to, ref } }) |p| if (p[0] < p[1] + at and p[1] < p[0] + at) return error.BadRange;
        const l = try m.spillLayout();
        var refs_buf: [4 * spill.max_layers]SpillRefPlane = undefined;
        const refs = try m.spillRefPlanes(&refs_buf);
        var ref_bytes: usize = 0;
        for (refs) |rp| ref_bytes += at / rp.ratio * rp.row;
        const nb = try l.blocks(at);
        const blk = try std.math.mul(usize, nb, l.block_bytes);
        if (ref_bytes != blk) return error.SpillLayoutMismatch; // the layout's block bytes are copyExtent's
        const small = l.small_bytes;
        var g = try m.spillGenerations(gpa);
        defer g.deinit(gpa);
        var pk = m.spillPacker(&l, &g);

        var a = try cuda.HostBuffer.alloc(m.ctx.d, blk + small);
        defer a.free();
        var b = try cuda.HostBuffer.alloc(m.ctx.d, blk + small);
        defer b.free();
        var rows = try cuda.HostBuffer.alloc(m.ctx.d, 2 * blk); // copyExtent's rows at ref, then the restored rows at to
        defer rows.free();
        var snaps = try cuda.HostBuffer.alloc(m.ctx.d, 2 * small); // the reference Snap, then the restored slot's
        defer snaps.free();
        var sa = try cuda.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
        defer sa.free();
        var sb = try cuda.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
        defer sb.free();
        var ev = try cuda.Event.init(m.ctx.d, false);
        defer ev.deinit();
        defer m.stream.synchronize() catch {}; // (runs first: nothing above is freed under an enqueued copy)

        // the engine's own copies of the point
        try m.snapCopy(from_slot, sa.ptr, true);
        try m.copyExtent(from, ref, at);
        // pack, restore elsewhere, pack again
        _ = try pk.packBlocks(try g.pin(from, from + at), 0, nb, a.bytes[0..blk], ev);
        _ = try pk.packSmall(try g.pinSlot(from_slot), a.bytes[blk..][0..small], ev);
        try ev.synchronize();
        const dst = try g.reserve(to, to + at);
        const ds = try g.reserveSlot(to_slot);
        const ub = try pk.unpackBlocks(dst, 0, nb, a.bytes[0..blk], ev);
        const us = try pk.unpackSmall(ds, a.bytes[blk..][0..small], ev);
        _ = try pk.packBlocks(dst, 0, nb, b.bytes[0..blk], ev);
        _ = try pk.packSmall(ds, b.bytes[blk..][0..small], ev);
        try m.snapCopy(to_slot, sb.ptr, true);
        // plain copies of each plane's rows [ref / r, (ref + at) / r) and [to / r, (to + at) / r), plane after plane
        const d = m.ctx.d;
        var off: usize = 0;
        for ([_]usize{ ref, to }) |base| for (refs) |rp| {
            const n = at / rp.ratio * rp.row;
            try d.check(d.api.cuMemcpyDtoHAsync_v2(rows.bytes[off..].ptr, rp.plane + base / rp.ratio * rp.row, n, m.stream.handle), "cuMemcpyDtoHAsync");
            off += n;
        };
        try d.check(d.api.cuMemcpyDtoHAsync_v2(snaps.bytes.ptr, sa.ptr, small, m.stream.handle), "cuMemcpyDtoHAsync");
        try d.check(d.api.cuMemcpyDtoHAsync_v2(snaps.bytes[small..].ptr, sb.ptr, small, m.stream.handle), "cuMemcpyDtoHAsync");
        try m.stream.synchronize();
        if (!g.valid(ub) or !g.valid(us)) return error.StaleExtent;

        if (!std.mem.eql(u8, a.bytes[0 .. blk + small], b.bytes[0 .. blk + small])) return error.SpillRoundTripMismatch;
        const ref_rows = rows.bytes[0..blk];
        if (!std.mem.eql(u8, ref_rows, rows.bytes[blk..][0..blk])) return error.SpillRestoreMismatch;
        // the packed blocks against copyExtent's rows: block k's plane j is that plane's rows from k * 2048 / ratio
        var in_block: usize = 0; // plane j's offset in a block (copyExtent's order and widths)
        var in_rows: usize = 0; // plane j's offset in ref_rows
        for (refs) |rp| {
            const per = chunk_rows / rp.ratio * rp.row;
            for (0..nb) |k| {
                if (!std.mem.eql(u8, b.bytes[k * l.block_bytes + in_block ..][0..per], ref_rows[in_rows + k * per ..][0..per])) return error.SpillLayoutMismatch;
            }
            in_block += per;
            in_rows += at / rp.ratio * rp.row;
        }
        if (in_block != l.block_bytes) return error.SpillLayoutMismatch;
        const ref_snap = snaps.bytes[0..small];
        if (!std.mem.eql(u8, ref_snap, snaps.bytes[small..][0..small])) return error.SpillRestoreMismatch;
        if (!std.mem.eql(u8, ref_snap, b.bytes[blk..][0..small])) return error.SpillLayoutMismatch;
    }

    /// PETABRIDGE (k2a-pb1): the opt-in switch for the startup check's device round trip (spillSelfCheck on the fresh
    /// pool). Unset, or anything but "1" (production today, spill off): the startup check is host-only.
    pub const spill_self_check_env = "TF_DS_SPILL_SELF_CHECK";

    /// PETABRIDGE (k2a-pb1): the spill layout's startup check, once a rank at the end of open: one line on stderr,
    /// "spill layout: ok ..." or "spill layout: MISMATCH ...". It never fails and never stops the start. With spill off
    /// it is host arithmetic over the loaded layers and the allocation's addresses: no device call, no allocation.
    /// With TF_DS_SPILL_SELF_CHECK=1 it also runs spillSelfCheck's device round trip on the fresh pool (see
    /// spillStartupRoundTrip), its outcome in the same line.
    fn spillStartupCheck(m: *Model) void {
        const want = if (std.c.getenv(spill_self_check_env)) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        var buf: [512]u8 = undefined;
        std.debug.print("{s}\n", .{m.spillCheckLine(&buf, want)});
    }

    /// PETABRIDGE (k2a-pb1): spillStartupCheck's line (host-only unless `round_trip`). Grep for "spill layout: ".
    pub fn spillCheckLine(m: *Model, buf: []u8, round_trip: bool) []const u8 {
        const snap = m.snapBytes();
        const tail = m.spillSnapTail();
        const l = m.spillLayoutOf() catch |err|
            return std.fmt.bufPrint(buf, "spill layout: MISMATCH rank {d} (descriptor refused: {s}; snap {d} bytes, drafter {d}, logits {d})", .{ m.rank, @errorName(err), snap, tail.draft, tail.logits }) catch "spill layout: MISMATCH";
        if (m.spillLayoutFault(&l)) |why|
            return std.fmt.bufPrint(buf, "spill layout: MISMATCH rank {d} ({s}; small {d} bytes, snap {d}, drafter {d}, logits {d}, block {d} bytes)", .{ m.rank, why, l.small_bytes, snap, tail.draft, tail.logits, l.block_bytes }) catch "spill layout: MISMATCH";
        var trip_buf: [128]u8 = undefined;
        var verdict: []const u8 = "ok";
        const trip: []const u8 = if (!round_trip) "round trip skipped: spill off" else if (m.spillStartupRoundTrip(&l)) |_| "round trip ok" else |err| switch (err) {
            error.SpillRoundTripMismatch, error.SpillRestoreMismatch, error.SpillLayoutMismatch, error.StaleExtent => blk: {
                verdict = "MISMATCH";
                break :blk std.fmt.bufPrint(&trip_buf, "round trip failed: {s}", .{@errorName(err)}) catch "round trip failed";
            },
            else => std.fmt.bufPrint(&trip_buf, "round trip not run: {s}", .{@errorName(err)}) catch "round trip not run",
        };
        return std.fmt.bufPrint(buf, "spill layout: {s} rank {d} (small {d} bytes = snap {d} - drafter {d} - logits {d}; block {d} bytes, {d} B/token; planes {d}+{d}; {s})", .{ verdict, m.rank, l.small_bytes, snap, tail.draft, tail.logits, l.block_bytes, l.token_bytes, l.n_block, l.n_small, trip }) catch "spill layout: ok";
    }

    /// PETABRIDGE (k2a-pb1): TF_DS_SPILL_SELF_CHECK=1's device round trip at open, while the pool is fresh (open has
    /// cleared it and nothing has run). Slot 0 and the extent at 0 are seeded with a byte pattern (block planes by
    /// plain copies, the small state by snapCopy from a scratch Snap whose tail is zero), spillSelfCheck packs them,
    /// restores them into slot 1 and the extent at 2,048 and compares against copyExtent into the extent at 4,096 and
    /// snapCopy. Then every byte it touched goes back to a fresh pool's: the three extents' block rows to their
    /// sentinels (Caches.clear's values), slots 0 and 1's rings and raw stores to zero (open's), slot 0's drafter rings
    /// zero (Pool.init's, written by the seed) and the prompt logits zero (every prompt's head writes them before they
    /// are read). About 40 MB of pinned host and scratch, freed before it returns.
    fn spillStartupRoundTrip(m: *Model, l: *const spill.StateLayout) !void {
        const at = chunk_rows;
        const bases = [_]usize{ 0, chunk_rows, 2 * chunk_rows }; // from, to, ref
        if (m.streams < 2) return error.TooFewStreams;
        if (m.caches.cap < 3 * chunk_rows) return error.PoolTooSmall;
        const checked = m.spillStartupSeeded(l, bases, at);
        // (also after a failed seed or check: whatever it wrote goes back to the fresh pool's bytes)
        const dv: SpillDev = .{ .m = m };
        for (bases) |base| for (l.blockPlanes()) |p| try dv.fill(dv.blockPlane(p) + base / p.ratio * p.row, p.sentinel, p.spanBytes(at));
        for ([_]usize{ 0, 1 }) |s| for (l.smallPlanes()) |p| try dv.fill(dv.slotPlane(p, s), 0, p.slotBytes());
        try m.stream.synchronize();
        return checked;
    }

    /// PETABRIDGE (k2a-pb1): spillStartupRoundTrip's seed of slot 0 and the extent at bases[0], then spillSelfCheck.
    fn spillStartupSeeded(m: *Model, l: *const spill.StateLayout, bases: [3]usize, at: usize) !void {
        const d = m.ctx.d;
        const dv: SpillDev = .{ .m = m };
        const snap = m.snapBytes();
        const block_bytes = l.block_bytes; // (at is one block)
        var hb = try cuda.HostBuffer.alloc(d, block_bytes + snap);
        defer hb.free();
        var sd = try cuda.DeviceBuffer.alloc(d, snap);
        defer sd.free();
        defer m.stream.synchronize() catch {}; // (runs first: the stream idle before the staging goes, whatever failed)
        var x: u32 = 0x9e3779b9;
        for (hb.bytes[0 .. block_bytes + l.small_bytes]) |*byte| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            byte.* = @truncate(x);
        }
        @memset(hb.bytes[block_bytes + l.small_bytes ..], 0); // the Snap tail: zeros (the drafter rings' and logits' fresh bytes)
        var off: usize = 0;
        for (l.blockPlanes()) |p| {
            const n = p.spanBytes(at);
            try dv.h2d(dv.blockPlane(p) + bases[0] / p.ratio * p.row, hb.bytes[off..][0..n]);
            off += n;
        }
        try dv.h2d(sd.ptr, hb.bytes[block_bytes..][0..snap]);
        try m.snapCopy(0, sd.ptr, false);
        try m.stream.synchronize();
        try m.spillSelfCheck(m.gpa, 0, bases[0], 1, bases[1], bases[2], at);
    }

    /// Each layer's gate_bias_vl (f32 [experts], the checkpoint's extra file) on the device: Gate.forward's bias at
    /// image-span positions.
    fn loadVisionBias(m: *Model, path: []const u8) !void {
        const core = @import("core");
        var f = try core.safetensors.File.open(m.gpa, m.io, path);
        defer f.close(m.io);
        var nb: [64]u8 = undefined;
        for (m.w.layers, 0..) |*lay, i| {
            const t = f.get(try std.fmt.bufPrint(&nb, "layers.{d}.ffn.gate.bias_vl", .{i})) orelse return error.MissingVisionBias;
            if (!t.is(.f32, &.{m.cfg.experts})) return error.BadVisionBias;
            const dst = try m.arena.take(t.bytes.len);
            const tmp = try m.gpa.dupe(u8, t.bytes); // (a copy out of the file mapping first: see vit.zig)
            defer m.gpa.free(tmp);
            try m.ctx.d.check(m.ctx.d.api.cuMemcpyHtoD_v2(dst, tmp.ptr, tmp.len), "cuMemcpyHtoD");
            lay.gate_b_vl = dst;
        }
    }

    /// The filling prompt's image spans (absolute positions, in order) and their rows bf16 [tokens, hidden] from the
    /// host, after fillBegin and before its chunks.
    pub fn setImages(m: *Model, spans: []const prompt.Span, rows: []const u8) !void {
        if (rows.len > 0) {
            if (m.img_buf == null or m.img_buf.?.len < rows.len) {
                if (m.img_buf) |*b| b.free();
                m.img_buf = null;
                m.img_buf = try cuda.DeviceBuffer.alloc(m.ctx.d, rows.len);
            }
            const d = m.ctx.d;
            try d.check(d.api.cuMemcpyHtoDAsync_v2(m.img_buf.?.ptr, rows.ptr, rows.len, m.stream.handle), "cuMemcpyHtoDAsync");
            try m.stream.synchronize(); // (the caller's rows may go once this returns)
        }
        m.spans.clearRetainingCapacity();
        try m.spans.appendSlice(m.gpa, spans);
        m.ch.spans = m.spans.items;
        m.ch.img_rows = if (m.img_buf) |b| b.ptr else 0;
    }

    pub fn fillBegin(m: *Model, slot: usize, base: usize) !void {
        if (slot >= m.streams) return error.BadSlot;
        m.spans.clearRetainingCapacity();
        m.ch.spans = &.{};
        m.view = try m.caches.view(&m.eng, slot, base);
        for (m.rings, m.ring_view) |r, *v| v.* = r + slot * m.eng.ringBytes();
        m.fill_slot = slot;
        // PETABRIDGE (k2a-pb1): a read-ahead the previous pass left joined holds that prompt's Engram rows. A pass that
        // ended early (a cancel between chunks, or the prefill interleave's error) leaves ahead_start/ahead_n/ready
        // set, and fillChunk's joined-read-ahead reuse (MiaAI-Lab 0005) would take them for this prompt's first chunk
        // when it starts at the same position (a resume there): another prompt's rows, silently. Never this pass's.
        m.joinAhead();
        if (m.eh) |*eh| {
            eh.ahead_ready = .{ false, false };
            eh.ahead_err = null;
        }
    }

    /// The prompt's chunk of n rows from position `start` (seq: the stream's ids through the chunk's end) through every
    /// layer with the served bounded replay (replay = the prompt's length - the window: an encoder-only chunk stops at
    /// the decoder's first layer). A chunk that reaches the head leaves the logits of its last row in ch.head_g and,
    /// drafting, its DSpark taps absorbed into the slot's drafter rings. True when it reached the head.
    pub fn fillChunk(m: *Model, seq: []const i32, start: usize, n: usize, replay: usize) !bool {
        if (n == 0 or n > chunk_rows or seq.len != start + n) return error.BadChunk;
        const e = &m.eng;
        const c = e.c;
        const ch = &m.ch;
        // an image position's id is negative in the sequence (its n-grams dead for Engram): its embedding's id is the
        // image token's, which its span row then replaces
        const image_id: i64 = if (c.image_token) |t| t else 0;
        for (0..n) |j| m.ids64[j] = if (seq[start + j] < 0) image_id else seq[start + j];
        var shared: prompt.Shared = .{};
        const pt = m.ptimer;
        if (pt) |t| try t.mark(m.stream, .start);
        try prompt.begin(e, ch, m.ids64[0..n], start, m.host_pos);
        if (ch.spans.len > 0) try prompt.embedSpans(e, ch);
        if (pt) |t| try t.mark(m.stream, .embed);
        const taps = m.dr != null;
        var floor: usize = 0;
        var kv_done: ?usize = null;
        // Engram rows read ahead on a thread while the GPU runs the layers before them (the served reader pool's
        // overlap): this chunk's both layers' during the chunk before it (m.ahead), else the first layer's from the
        // chunk's start and the next's once the first is applied
        var ahead: ?std.Thread = m.ahead;
        m.ahead = null;
        defer if (ahead) |th| th.join();
        if (m.eh) |*eh| {
            if (ahead != null and (eh.ahead_start != start or eh.ahead_n != n)) {
                ahead.?.join();
                ahead = null;
            }
            // a read-ahead of this chunk already joined (a round ran between two chunks): its rows stand ready
            const joined = ahead == null and eh.ahead_start == start and eh.ahead_n == n and eh.ahead_ready[0];
            if (ahead == null and !joined) {
                eh.ahead_start = start;
                eh.ahead_n = n;
                eh.ahead_ready = .{ false, false };
                eh.ahead_err = null;
                ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, @as(usize, 0) }) catch null;
            }
        }
        for (0..c.layers) |li| {
            if (li == c.layers / 2) {
                if (!try prompt.replayCut(e, ch, &m.view, &shared, li, replay, m.host_pos)) {
                    // encoder-only: no synchronize either (as at the chunk's end below), so the next chunk's host steps
                    // run while this one's last layers do; the profile alone waits
                    if (pt) |t| {
                        try m.stream.synchronize();
                        try t.finish();
                    }
                    return false;
                }
                floor = replay;
                kv_done = li;
            }
            if (e.w.layers[li].engram_wkv != null) {
                const eh = if (m.eh) |*x| x else return error.NoEngramTables;
                if (ahead) |th| {
                    th.join();
                    ahead = null;
                }
                if (eh.ahead_err) |err| return err;
                try prompt.engramApply(e, ch, eh, li, seq);
                const j = std.mem.indexOfScalar(u16, c.engram_layers.slice(), @intCast(li)).?;
                const ne = @min(2, c.engram_layers.slice().len);
                if (j + 1 < ne) {
                    if (!eh.ahead_ready[j + 1])
                        ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, j + 1 }) catch null;
                } else if (m.fill_seq.len > start + n and m.fill_seq.ptr == seq.ptr) {
                    // the next chunk's rows while this one's layers run
                    const nn = @min(chunk_rows, m.fill_seq.len - start - n);
                    eh.ahead_start = start + n;
                    eh.ahead_n = nn;
                    eh.ahead_ready = .{ false, false };
                    eh.ahead_err = null;
                    m.ahead = std.Thread.spawn(.{}, prompt.engramReadAheadAll, .{ eh, c, m.fill_seq[0 .. start + n + nn], start + n, nn }) catch null;
                }
                if (pt) |t| try t.mark(m.stream, .engram);
            }
            // a DSpark tap reads the streams after the layer's Engram (_forward_k's order)
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j);
            try prompt.attnMixes(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .mix_attn);
            try prompt.attention(e, ch, &m.view, &shared, li, m.ring_view[li], floor, kv_done != null and kv_done.? == li);
            if (pt) |t| try t.mark(m.stream, .attn);
            try prompt.gather(e, ch, ch.pa, ch.ga);
            if (pt) |t| try t.mark(m.stream, .gather_a);
            try prompt.ffnMixes(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .mix_ffn);
            try prompt.moe(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .moe);
            try prompt.gather(e, ch, ch.pm, ch.gm);
            if (pt) |t| try t.mark(m.stream, .gather_m);
            prompt.endLayer(ch);
        }
        try prompt.head(e, ch);
        if (pt) |t| try t.mark(m.stream, .head);
        if (m.dr) |*dr| {
            // its taps (the tap layers side by side) into the slot's drafter rings
            const dw = c.hidden;
            for (0..c.dspark_taps.slice().len) |j| try e.ops.copyRows(e.s, prompt.tapRows(e, ch, j), dw * 2, dr.at + j * dw * 2, dr.taps_w * 2, dw * 2, ch.n);
            try dr.absorb(e, ch, &m.dpool.?, m.fill_slot, dr.at, ch.n, ch.start);
        }
        // no synchronize: the next chunk's host steps (its first Engram layer's hashing, table reads and decode, ~0.14 s
        // at 2,048 rows) run while this chunk's kernels do; readers of the results (the prompt's logits) synchronize.
        // The profile (--profile 2) times a chunk to its end, so it alone waits.
        if (pt) |t| {
            try m.stream.synchronize();
            try t.finish();
        }
        return true;
    }

    /// The prompt's logits (its last row's, the ranks' columns in rank order) after a fill's last chunk.
    pub fn promptLogits(m: *const Model) u64 {
        return m.ch.head_g;
    }

    /// One round over every stream's window (rows.windows, each with its stream's slot and extent), then each window
    /// whose stream drafts has its rows' taps absorbed into its slot's drafter rings at its positions (the served
    /// eager absorb: rows past the ones the round keeps sit where the drafter never reads, and the next window
    /// overwrites them). absorb[i]: window i's stream drafts.
    pub fn verify(m: *Model, rows: round.Rows, absorb: []const bool) !void {
        m.joinAhead(); // a prompt chunk's Engram read-ahead shares the host buffers a round's Engram reads use
        const e = &m.eng;
        if (m.prof) try m.stream.synchronize(); // (the GPU idle, so the issue time shows whether the host keeps it fed)
        const t0 = m.now();
        try round.forward(e, &m.rd, &m.ch, &m.caches, m.rings, if (m.eh) |*x| x else null, &.{}, rows, m.pool_cap, null, if (m.graphs) |*g| g else null);
        if (m.prof) {
            m.t_enqueue = m.now() - t0;
            try m.stream.synchronize();
            m.t_forward = m.now() - t0;
            if (m.timer) |t| try t.finish();
            if (m.rd.clk != 0) {
                var h: [1 + 2 * round.Round.max_clocks]u64 = undefined;
                const d = m.ctx.d;
                try d.check(d.api.cuMemcpyDtoH_v2(&h, m.rd.clk, h.len * 8), "cuMemcpyDtoH");
                const n = @min(h[0], round.Round.max_clocks);
                for (1..n) |i| {
                    const tag = h[2 + 2 * i] & 0xff;
                    m.clk_ph[tag] += h[1 + 2 * i] -% h[1 + 2 * (i - 1)];
                    m.clk_cnt[tag] += 1;
                }
                m.clk_rounds += 1;
            }
        }
        const t1 = m.now();
        if (m.dr) |*dr| {
            const wins = rows.windows orelse return error.BadRound;
            var items: [max_streams]draft.Item = undefined;
            var n: usize = 0;
            for (wins, absorb) |win, yes| {
                if (!yes) continue;
                items[n] = .{ .slot = @intCast(rows.slots.?[win.row]), .taps = m.rd.taps + win.row * dr.taps_w * 2, .n = win.n, .start = @intCast(rows.pos[win.row]) };
                n += 1;
            }
            if (n > 0) try dr.absorbMany(e, &m.ch, &m.dpool.?, items[0..n]);
        }
        try m.stream.synchronize();
        if (m.prof) m.t_absorb = m.now() - t1;
    }

    /// The rounds' GPU clocks (a profile): the forward's start and each stretch's ends, for its gaps.
    pub fn useClock(m: *Model, marks: u8) !void {
        m.rd.clk_marks = marks == 1 or marks == 3;
        m.rd.clk_ratio = if (marks == 3) 2 else null;
        m.rd.clk_layers = marks == 2 or marks == 3;
        if (m.rd.clk == 0) m.rd.clk = try m.arena.take((1 + 2 * round.Round.max_clocks) * 8);
    }

    /// The rounds' GPU time by phase (eager rounds; --profile 2): a timer on the round.
    pub fn usePhaseTimer(m: *Model) !void {
        if (m.timer != null) return;
        const t = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(t);
        t.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.timer = t;
        const pt = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(pt);
        pt.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.ptimer = pt;
        m.rd.timer = t;
    }

    /// The monotonic clock (ns).
    pub fn now(m: *const Model) u64 {
        return @intCast(std.Io.Timestamp.now(m.io, .awake).nanoseconds);
    }

    /// The round's logits, R rows of the vocabulary, on the host (after verify).
    /// The round's greedy tokens, R rows (the device's argmax of the round's logits, as sampling.argmax), on the host.
    pub fn roundArgmax(m: *Model, R: usize) ![]const u32 {
        const d = m.ctx.d;
        try m.ops.argmaxRows(m.stream, m.rd.logits, m.vocab, m.cfg.tokenVocab(), R, m.amax);
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(&m.amax_host, m.amax, R * 4), "cuMemcpyDtoH");
        return m.amax_host[0..R];
    }

    pub fn roundLogits(m: *Model, R: usize) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.rd.logits, R * m.vocab * 4), "cuMemcpyDtoH");
        m.cfg.maskHostLogits(m.logits[0 .. R * m.vocab]);
        return m.logits[0 .. R * m.vocab];
    }

    /// The prompt's logits on the host (after a fill's last chunk).
    pub fn promptLogitsHost(m: *Model) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.ch.head_g, m.vocab * 4), "cuMemcpyDtoH");
        m.cfg.maskHostLogits(m.logits[0..m.vocab]);
        return m.logits[0..m.vocab];
    }

    /// The drafter's batched pass for N streams: each one's pending token at position q0 in its slot (its rings hold
    /// its positions below q0), `steps` drafts each (the drafter's drafts and confidences on the host).
    pub fn pass(m: *Model, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize) !void {
        if (m.dr == null) return error.NoDrafter;
        const dr = &m.dr.?;
        const pool = &m.dpool.?;
        for (q0, slots) |q, s| pool.absorbed[@intCast(s)] = @intCast(q);
        try dr.pass(&m.eng, &m.ch, pool, tokens, q0, slots, steps, null);
    }
};

// -- PETABRIDGE (k2a-pb1): spill port P4's host tests (no device: the model's planes are fake addresses nothing reads) --

/// A Model with no device behind it, enough for spillLayout, snapBytes and spillCheckLine (host arithmetic over the
/// loaded layers and the allocation's addresses): our checkpoint's layers as weights.load sets them (a kv source's
/// compressor, an index source's keys on a kv source), the pool's planes at distinct nonzero addresses where
/// Caches.init and open allocate them, the drafter pool's sizes as draft.Pool.init sets them, and open's vocabulary
/// columns (the world times this rank's head columns).
const SpillShell = struct {
    m: Model,
    cfg: Config,
    w: weights.Weights,
    layers: [64]weights.Layer,
    rings: [64]u64,

    fn init(sh: *SpillShell, vocab: usize, drafts: bool, streams: usize) !void {
        var why: @import("config.zig").Why = .{};
        sh.cfg = try @import("config.zig").parse(std.testing.allocator, @import("config.zig").test_config, &why);
        const c = &sh.cfg;
        const fake: weights.Linear = .{ .words = 1, .suh = 1, .svh = 1, .k = 1, .n = 1, .k2 = 1 };
        sh.w = .{ .gpa = std.testing.allocator };
        for (sh.layers[0..c.layers], 0..) |*lay, i| {
            lay.* = .{ .idx = @intCast(i), .ratio = c.ratios[i] };
            if (c.kv_sources.has(i)) lay.comp_wkv = fake; // weights.zig: kv_sources.has(i)
            if (c.index_sources.has(i) and c.kv_sources.has(i)) lay.idx_wk = fake; // an index source that is a kv source
        }
        sh.w.layers = sh.layers[0..c.layers];
        sh.m = undefined;
        sh.m.eng.c = c;
        sh.m.eng.w = &sh.w;
        sh.m.rank = 0;
        sh.m.streams = streams;
        sh.m.vocab = vocab;
        sh.m.caches = .{ .cap = 1 << 18, .slots = streams };
        var next: u64 = 1 << 40;
        const step: u64 = 1 << 32;
        for (sh.layers[0..c.layers], 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            sh.m.caches.comp_codes[i] = next;
            sh.m.caches.comp_scales[i] = next + step;
            next += 2 * step;
            if (lay.idx_wk != null) {
                sh.m.caches.idx_codes[i] = next;
                sh.m.caches.idx_scales[i] = next + step;
                next += 2 * step;
            }
            if (lay.ratio > 1) {
                sh.m.caches.raw_kv[i] = next;
                sh.m.caches.raw_score[i] = next + step;
                next += 2 * step;
            }
        }
        for (sh.rings[0..c.layers]) |*r| {
            r.* = next;
            next += step;
        }
        sh.m.rings = sh.rings[0..c.layers];
        // draft.Pool.init: a ring plane a DSpark block (the checkpoint's draft layers), window + RING_EXTRA rows
        sh.m.dpool = if (drafts) .{ .stages = c.draft_layers, .slots = streams, .ring = c.window + draft.ring_extra, .hd = c.head_dim } else null;
    }
};

// TP2's columns (2 x 64,640: the 129,280 vocabulary) and TP3's (3 x 43,136: the padded 129,408 one).
const spill_test_vocabs = [_]usize{ 2 * 64640, 3 * 43136 };

test "PETABRIDGE spill P4: our layout is MiaAI's (890 B a token) and the small state is snapBytes less the drafter rings and logits" {
    for (spill_test_vocabs) |vocab| for ([_]bool{ true, false }) |drafts| {
        var sh: SpillShell = undefined;
        try sh.init(vocab, drafts, 4);
        const m = &sh.m;
        const l = try m.spillLayout();
        // the descriptor from the loaded layers is the one config.json gives (spill_state's own fixture numbers)
        const from_cfg = try Model.spill.StateLayout.fromConfig(&sh.cfg, Model.spill.Geometry.of(&sh.cfg, chunk_rows, prompt.ring_extra, prompt.raw_rows), .replicated);
        try std.testing.expectEqualSlices(u8, &from_cfg.digest(), &l.digest());
        try std.testing.expectEqual(@as(usize, 890), l.token_bytes);
        try std.testing.expectEqual(@as(usize, 1_822_720), l.block_bytes);
        try std.testing.expectEqual(@as(usize, 6_684_672), l.small_bytes);
        try std.testing.expectEqual(@as(usize, 16), l.n_block);
        try std.testing.expectEqual(@as(usize, 46), l.n_small);
        // our Snap: the small state, then 3 stages x 144 x 512 x 2 drafter bytes, then the logits
        const tail = m.spillSnapTail();
        try std.testing.expectEqual(@as(usize, if (drafts) 3 * 144 * 512 * 2 else 0), tail.draft);
        try std.testing.expectEqual(vocab * 4, tail.logits);
        try std.testing.expectEqual(l.small_bytes + tail.bytes(), m.snapBytes());
        // MiaAI's own check (small_bytes == snapBytes) does not hold here: our Snap is larger by exactly the tail
        try std.testing.expect(l.small_bytes != m.snapBytes());
        if (drafts and vocab == 3 * 43136) try std.testing.expectEqual(@as(usize, 7_644_672), m.snapBytes());
    };
}

test "PETABRIDGE spill P4: the small planes are snapCopy's first copies, in its order, at its addresses" {
    var sh: SpillShell = undefined;
    try sh.init(3 * 43136, true, 4);
    const m = &sh.m;
    const l = try m.spillLayout();
    const dv: Model.SpillDev = .{ .m = m };
    for (0..m.streams) |slot| {
        // snapCopy's walk (rings, then each layer's raw kv and raw score where allocated): live address and length
        var want: [Model.spill.max_planes][2]u64 = undefined;
        var n: usize = 0;
        const rb = m.eng.ringBytes();
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        for (m.rings) |r| {
            want[n] = .{ r + slot * rb, rb };
            n += 1;
        }
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| for ([_]u64{ k, sc }) |live| if (live != 0) {
            want[n] = .{ live + slot * pb, pb };
            n += 1;
        };
        try std.testing.expectEqual(n, l.n_small);
        var off: usize = 0;
        for (l.smallPlanes(), 0..) |p, i| {
            try std.testing.expectEqual(want[i][0], dv.slotPlane(p, slot));
            try std.testing.expectEqual(want[i][1], p.slotBytes());
            try std.testing.expectEqual(off, l.smallOffset(i)); // its offset in the Snap buffer, as snapCopy advances
            off += p.slotBytes();
        }
        // what snapCopy copies after them (the drafter rings, then the logits) is the tail
        try std.testing.expectEqual(l.small_bytes, off);
        try std.testing.expectEqual(m.snapBytes() - off, m.spillSnapTail().bytes());
    }
    // the block planes are the pool's own (Caches.init's) addresses
    for (l.blockPlanes()) |p| {
        const cs = &m.caches;
        const want = switch (p.kind) {
            .comp_codes => cs.comp_codes[p.layer],
            .comp_scales => cs.comp_scales[p.layer],
            .idx_codes => cs.idx_codes[p.layer],
            .idx_scales => cs.idx_scales[p.layer],
            else => unreachable,
        };
        try std.testing.expectEqual(want, dv.blockPlane(p));
    }
}

test "PETABRIDGE spill P4: the startup line is ok for our layout, MISMATCH (never an error) for a broken one" {
    var buf: [512]u8 = undefined;
    {
        var sh: SpillShell = undefined;
        try sh.init(3 * 43136, true, 4);
        const line = sh.m.spillCheckLine(&buf, false);
        try std.testing.expectEqualStrings("spill layout: ok rank 0 (small 6684672 bytes = snap 7644672 - drafter 442368 - logits 517632; block 1822720 bytes, 890 B/token; planes 16+46; round trip skipped: spill off)", line);
    }
    {
        // a ratio-2 source without its raw stores: snapBytes no longer covers the layout's small state
        var sh: SpillShell = undefined;
        try sh.init(3 * 43136, true, 4);
        sh.m.caches.raw_kv[2] = 0;
        sh.m.caches.raw_score[2] = 0;
        try std.testing.expectError(error.SpillLayoutMismatch, sh.m.spillLayout());
        const line = sh.m.spillCheckLine(&buf, false);
        try std.testing.expect(std.mem.startsWith(u8, line, "spill layout: MISMATCH rank 0 (small state != snapBytes - drafter rings - logits;"));
    }
    {
        // a block plane missing from the allocation
        var sh: SpillShell = undefined;
        try sh.init(3 * 43136, false, 4);
        sh.m.caches.idx_codes[8] = 0;
        try std.testing.expectError(error.SpillLayoutMismatch, sh.m.spillLayout());
        try std.testing.expect(std.mem.startsWith(u8, sh.m.spillCheckLine(&buf, false), "spill layout: MISMATCH rank 0 (a block plane is not allocated;"));
    }
    {
        // a ring per layer
        var sh: SpillShell = undefined;
        try sh.init(3 * 43136, true, 4);
        sh.m.rings = sh.rings[0 .. sh.cfg.layers - 1];
        try std.testing.expectError(error.SpillLayoutMismatch, sh.m.spillLayout());
        try std.testing.expect(std.mem.startsWith(u8, sh.m.spillCheckLine(&buf, false), "spill layout: MISMATCH rank 0 (rings != layers;"));
    }
    {
        // a kv source whose ratio the spill format does not describe: the descriptor refuses it
        var sh: SpillShell = undefined;
        try sh.init(3 * 43136, true, 4);
        sh.layers[20].ratio = 4;
        try std.testing.expectError(error.BadLayout, sh.m.spillLayout());
        try std.testing.expect(std.mem.startsWith(u8, sh.m.spillCheckLine(&buf, false), "spill layout: MISMATCH rank 0 (descriptor refused: BadLayout;"));
    }
}

test "PETABRIDGE spill P4: a disk point's resumed pass absorbs the drafter's whole read set (our chunking), so its rings are rebuilt" {
    var sh: SpillShell = undefined;
    try sh.init(3 * 43136, true, 4);
    const c = &sh.cfg;
    const window = c.window;
    const ring = sh.m.dpool.?.ring;
    try std.testing.expectEqual(@as(usize, 128), window);
    try std.testing.expectEqual(@as(usize, 144), ring);
    var checked: usize = 0;
    var len: usize = window + 1;
    while (len < 6 * chunk_rows + 700) : (len += 37) {
        var at: usize = chunk_rows;
        while (at < len) : (at += chunk_rows) {
            if (!Model.spill.eligible(at, len, window, chunk_rows)) continue;
            // MiaAI's rule (2048-token chunks from `at`)
            try Model.spill.checkReplayRebuilds(at, len, window, chunk_rows, ring);
            // ours: lanes.zig's prefill loop from a pre-replay point (replay = len - window, the split at last_end),
            // each chunk past the replay reaching the head and draft.absorb keeping its last min(n, ring) rows
            const replay = len - window;
            const last_end = std.mem.alignBackward(usize, @min(replay, len - 1), 2);
            var got: [2]usize = .{ 0, 0 }; // the newest contiguous absorbed run
            var start = at;
            var absorbed: usize = 0;
            while (start < len) {
                const end = if (start < last_end) @min(len, last_end) else len;
                const n = @min(chunk_rows, end - start);
                const first = @max(start, replay);
                if (first < start + n) {
                    const rows = start + n - first; // the chunk's rows after prompt.replayCut
                    try std.testing.expect(rows <= 128); // draft.absorb's ported limit
                    const keep = @min(rows, ring);
                    const lo = start + n - keep;
                    got = if (got[1] != 0 and lo <= got[1]) .{ @min(got[0], lo), start + n } else .{ lo, start + n };
                    absorbed = start + n; // Pool.absorbed[slot]
                }
                start += n;
            }
            // the drafter's first pass reads the window's newest positions below len (draft.zig: 128 newest absorbed)
            try std.testing.expectEqual(len, absorbed);
            try std.testing.expectEqual(len, got[1]);
            try std.testing.expect(got[0] <= len - window);
            checked += 1;
        }
    }
    try std.testing.expect(checked > 500);
}

test "PETABRIDGE spill P4: spillCheckResumed is DraftNotRebuilt until the pool absorbed the prompt's end" {
    var sh: SpillShell = undefined;
    try sh.init(3 * 43136, true, 4);
    const m = &sh.m;
    m.dpool.?.absorbed = @splat(0);
    m.dpool.?.absorbed[1] = 4096;
    try m.spillCheckResumed(1, 4096);
    try std.testing.expectError(error.DraftNotRebuilt, m.spillCheckResumed(1, 4100));
    try std.testing.expectError(error.DraftNotRebuilt, m.spillCheckResumed(0, 4096));
    try std.testing.expectError(error.BadSlot, m.spillCheckResumed(4, 4096));
    m.dpool = null; // no drafter: nothing to rebuild
    try m.spillCheckResumed(0, 4096);
}
