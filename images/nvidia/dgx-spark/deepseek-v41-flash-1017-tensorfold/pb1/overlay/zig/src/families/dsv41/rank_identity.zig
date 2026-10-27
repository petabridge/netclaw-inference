//! A DeepSeek-V4.1 rank's fingerprint inputs, gathered from what both commands already hold: the model.Options they
//! open the model with (native.open on rank 0, tf-dsv41-lanes on ranks 1-3: the same model dir, rank cache, kit and
//! switches), the opened Model, and the identity the recipe exports (TF_DS_ENGINE_ID and friends). Port of the
//! inputs wojo's GLM 0088/0110 spill signature takes (`build_id`, `weights_id`, the planes and switches), extended to
//! PORT-SPILL-V2.md's list. The Options and Model are taken as `anytype` so this file stays host-testable (the tests
//! use stand-ins with the same field names); round 2 passes `o: model.Options` and `m: *model.Model`.
const std = @import("std");
const fingerprint = @import("fingerprint.zig");
const rank_wire = @import("rank_wire.zig");
const spill_state = @import("spill_state.zig");
const parts = @import("parts.zig");
const rank_cache = @import("rank_cache.zig");

const Io = std.Io;

/// What the Options and Model do not hold: the recipe's identity and the spill/shard side's layout.
pub const Extra = struct {
    engine_id: []const u8, // TF_DS_ENGINE_ID: the engine commit / image content digest (required)
    recipe_patches: []const u8, // TF_DS_RECIPE_PATCHES: the applied patch series' digest (required)
    build_flags: []const u8 = "", // TF_DS_BUILD_FLAGS: the recipe's chosen build flags
    model_revision: []const u8 = "", // TF_DS_MODEL_REVISION: the checkpoint's revision id (config.json is hashed too)
    ablation: []const u8 = "off", // TF_DS_ABLATION_ID: the ablation overlay's identity when one is enabled
    rank_cache_sha: []const u8 = "", // TF_DS_RANK_CACHE_SHA: a recorded content digest of this rank's file, if any
    engram_revision: []const u8 = "", // TF_DS_ENGRAM_REVISION: the Engram tables' revision or digest (required with Engram)
    features: []const u8, // the switches, canonical (featureString)
    native_format: u32, // core/spill's format version
    chunk_rows: u32, // model.chunk_rows
    replay_floor: bool, // prompt.replay_floor
    idx_split: bool, // prompt.idx_split
    planes: []const u8, // the common part of spill_state's layout descriptor
    rank_layout: fingerprint.Digest, // spill_state.StateLayout.digest() on this rank
    owner_layout: fingerprint.OwnerLayout = .replicated,
    owner_version: u16 = 0, // parts.layout_version when owner_sharded
    parts: u8 = 1,
    gpu_capability: u32, // cuda Context.capability()
    cuda_driver: i32, // cuda Driver.version()
    rank_cache_name: []const u8, // rank_cache.find's name for this rank
    /// Whether the agreement guards state on disk (the spill tier): the recipe's identity is then required (envFor).
    /// TF_DS_KV_SHARD alone agrees the pair's configuration with the same fingerprint, but persists nothing, so an
    /// identity the recipe did not export reads "unset" there instead of refusing the start.
    persist: bool = true,

    /// The env half of Extra (the rest is passed by the caller): error.NoEngineId / error.NoRecipePatches when the
    /// recipe did not export them, error.NoWeightsId when neither TF_DS_RANK_CACHE_SHA nor TF_DS_MODEL_REVISION is
    /// set (a spill or shard start then refuses: an unidentified engine or set of weights must never read or agree on
    /// stored state; the rank cache's header alone does not tell two checkpoints or quants of one shape apart).
    /// TF_DS_ENGRAM_REVISION is read here and required by gather when the model runs with Engram tables.
    pub fn env(get: *const fn ([:0]const u8) ?[]const u8) !EnvPart {
        return envFor(get, true);
    }

    /// env, or with `persist` false (TF_DS_KV_SHARD without the spill tier: nothing on disk to guard) every missing
    /// identity "unset" instead of a refusal; both ranks still must agree on what they hold.
    pub fn envFor(get: *const fn ([:0]const u8) ?[]const u8, persist: bool) !EnvPart {
        if (!persist) return .{
            .engine_id = nonEmpty(get("TF_DS_ENGINE_ID")) orelse "unset",
            .recipe_patches = nonEmpty(get("TF_DS_RECIPE_PATCHES")) orelse "unset",
            .build_flags = get("TF_DS_BUILD_FLAGS") orelse "",
            .model_revision = get("TF_DS_MODEL_REVISION") orelse "",
            .ablation = nonEmpty(get("TF_DS_ABLATION_ID")) orelse "off",
            .rank_cache_sha = get("TF_DS_RANK_CACHE_SHA") orelse "",
            .engram_revision = get("TF_DS_ENGRAM_REVISION") orelse "",
        };
        const e: EnvPart = .{
            .engine_id = nonEmpty(get("TF_DS_ENGINE_ID")) orelse return error.NoEngineId,
            .recipe_patches = nonEmpty(get("TF_DS_RECIPE_PATCHES")) orelse return error.NoRecipePatches,
            .build_flags = get("TF_DS_BUILD_FLAGS") orelse "",
            .model_revision = get("TF_DS_MODEL_REVISION") orelse "",
            .ablation = nonEmpty(get("TF_DS_ABLATION_ID")) orelse "off",
            .rank_cache_sha = get("TF_DS_RANK_CACHE_SHA") orelse "",
            .engram_revision = get("TF_DS_ENGRAM_REVISION") orelse "",
        };
        try weightsIdentified(e.model_revision, e.rank_cache_sha);
        return e;
    }
};

pub const EnvPart = struct {
    engine_id: []const u8,
    recipe_patches: []const u8,
    build_flags: []const u8,
    model_revision: []const u8,
    ablation: []const u8,
    rank_cache_sha: []const u8,
    engram_revision: []const u8,
};

/// The weights' content must be identified by a recorded digest or revision, not only by their shapes.
fn weightsIdentified(model_revision: []const u8, rank_cache_sha: []const u8) !void {
    if (model_revision.len == 0 and rank_cache_sha.len == 0) return error.NoWeightsId;
}

fn nonEmpty(v: ?[]const u8) ?[]const u8 {
    const s = v orelse return null;
    return if (s.len == 0) null else s;
}

/// spill_state's owner layout as the fingerprint (and rank_wire's has) carry it.
pub fn ownerLayout(o: spill_state.OwnerLayout) fingerprint.OwnerLayout {
    return switch (o) {
        .replicated => .replicated,
        .owner_sharded => .owner_sharded,
    };
}

/// The state layout's part of Extra: `planes` (its canonical bytes, in `buf`: the common field, since the ranks of a
/// TP pair hold the same planes; owner_sharded, its part size too) and `rank_layout` (its digest: this rank's own
/// field), with its owner layout, the owner mapping's version and its parts (1 and 0 when replicated).
pub const LayoutPart = struct { planes: []const u8, rank_layout: fingerprint.Digest, owner_layout: fingerprint.OwnerLayout, owner_version: u16 = 0, parts: u8 = 1 };

/// `l` from Model.spillLayout (owner_sharded and cut into its parts under TF_DS_KV_SHARD=1: then `world` parts, one a
/// rank); `buf` must outlive the build. A sharded layout without its part size is refused (it could not pack).
pub fn layoutPart(l: *const spill_state.StateLayout, world: u32, buf: *[spill_state.StateLayout.max_encoded]u8) !LayoutPart {
    var lp: LayoutPart = .{ .planes = try l.encode(buf), .rank_layout = l.digest(), .owner_layout = ownerLayout(l.owner_layout) };
    if (l.owner_layout == .owner_sharded) {
        if (l.part_rows == 0 or world < 2 or world > parts.max_parts) return error.BadOwnerLayout;
        lp.owner_version = parts.layout_version;
        lp.parts = @intCast(world);
    }
    return lp;
}

/// The switches that change state or protocol, canonical ("name=value;" in this order): every rank must hold the same.
pub const Features = struct {
    spill: bool = false,
    spill_min_tokens: u32 = 0,
    kv_shard: bool = false,
    kv_carry: bool = false,
    twins: bool = false,
    part_rows: u64 = 0, // TF_DS_KV_SHARD: positions a part (the pool / world)
    engram_lanes: u8 = 1,
    image_cache: bool = true,
    streams: u32,
    round_rows: u32,
    pool: u64,
    drafts: bool,
};

/// TF_DS_KV_SHARD adds "part=N;" (the part size: the owner layout's, also in the planes field); off, the bytes are
/// what they always were.
pub fn featureString(buf: []u8, f: Features) ![]const u8 {
    const head = try std.fmt.bufPrint(buf, "spill={d};spill_min={d};kv_shard={d};kv_carry={d};twins={d};engram_lanes={d};image_cache={d};streams={d};round_rows={d};pool={d};drafts={d};", .{
        @intFromBool(f.spill), f.spill_min_tokens, @intFromBool(f.kv_shard), @intFromBool(f.kv_carry), @intFromBool(f.twins),
        f.engram_lanes, @intFromBool(f.image_cache), f.streams, f.round_rows, f.pool, @intFromBool(f.drafts),
    });
    if (!f.kv_shard) return head;
    const tail = try std.fmt.bufPrint(buf[head.len..], "part={d};", .{f.part_rows});
    return buf[0 .. head.len + tail.len];
}

/// The split of every rank (prompt2d.split's rule: world 2 is plain TP2; world 4 is two TP2 pairs).
// PETABRIDGE (k2a-pb1): world 3 is plain TP3, as our prompt2d.split has it (`1, 2, 3 => .{ .rank = g, .world = world }`)
fn splitText(buf: []u8, world: u32) ![]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    for (0..world) |g| switch (world) {
        2, 3 => try w.print("{d}:rank={d};", .{ g, g }), // PETABRIDGE (k2a-pb1): + world 3 (plain TP3)
        4 => try w.print("{d}:rank={d},pair={d};", .{ g, g % 2, g / 2 }),
        else => return error.UnsupportedWorld,
    };
    return w.buffered();
}

fn rankSplitText(buf: []u8, rank: u32, world: u32) ![]const u8 {
    return switch (world) {
        2, 3 => std.fmt.bufPrint(buf, "rank={d}", .{rank}), // PETABRIDGE (k2a-pb1): + world 3 (plain TP3)
        4 => std.fmt.bufPrint(buf, "rank={d},pair={d}", .{ rank % 2, rank / 2 }),
        else => error.UnsupportedWorld,
    };
}

/// This rank's weight file's identity: its name, size and index (rank_cache.zig's format: the tensors, then the JSON
/// index of names, dtypes, shapes and offsets, then the index's offset and the magic word), and the recorded content
/// digest when the recipe gives one. The tens of GB of content are not read.
pub fn rankCacheIdentity(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, name: []const u8, recorded: []const u8) ![]u8 {
    var file = try dir.openFile(io, name, .{});
    defer file.close(io);
    const size = try file.length(io);
    if (size < 16) return error.BadRankCache;
    var tail: [16]u8 = undefined;
    if (try file.readPositionalAll(io, &tail, size - 16) != 16) return error.BadRankCache;
    if (!std.mem.eql(u8, tail[8..], rank_cache.magic)) return error.BadRankCache;
    const at = std.mem.readInt(u64, tail[0..8], .little);
    if (at > size - 16 or size - 16 - at > 1 << 30) return error.BadRankCache;
    const index = try gpa.alloc(u8, @intCast(size - 16 - at));
    defer gpa.free(index);
    if (try file.readPositionalAll(io, index, at) != index.len) return error.BadRankCache;
    var d: fingerprint.Digest = undefined;
    std.crypto.hash.sha2.Sha256.hash(index, &d, .{});
    return std.fmt.allocPrint(gpa, "{s}|{d}|{x}|{s}", .{ name, size, &d, recorded });
}

fn exists(io: Io, dir: Io.Dir, path: []const u8) bool {
    dir.access(io, path, .{}) catch return false;
    return true;
}

fn fileOr(io: Io, dir: Io.Dir, path: []const u8) fingerprint.Source {
    return if (exists(io, dir, path)) .{ .file = .{ .dir = dir, .path = path } } else .none;
}

fn treeOr(io: Io, dir: Io.Dir, path: []const u8, keep: ?*const fn ([]const u8) bool) fingerprint.Source {
    return if (exists(io, dir, path)) .{ .tree = .{ .dir = dir, .path = path, .keep = keep } } else .none;
}

fn isRope(p: []const u8) bool {
    return std.mem.startsWith(u8, p, "rope-") and std.mem.indexOfScalar(u8, p, '/') == null;
}

fn isExl3(p: []const u8) bool {
    return std.mem.indexOf(u8, p, "exl3") != null;
}

/// A folder's file names and sizes, not their contents (the Engram tables: hundreds of GB), as text.
fn listing(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, path: []const u8) ![]u8 {
    var root = try dir.openDir(io, path, .{ .iterate = true });
    defer root.close(io);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    var it = root.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file) continue;
        const st = try root.statFile(io, e.name, .{});
        const line = try std.fmt.allocPrint(gpa, "{s}:{d};", .{ e.name, st.size });
        names.append(gpa, line) catch |err| {
            gpa.free(line);
            return err;
        };
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn lt(_: void, x: []u8, y: []u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    return std.mem.join(gpa, "", names.items);
}

/// Owns the strings and directories an Inputs points into (free with deinit after fingerprint.build).
pub const Gathered = struct {
    in: fingerprint.Inputs,
    arena: std.heap.ArenaAllocator,
    dirs: [3]?Io.Dir = .{ null, null, null },
    io: Io,

    pub fn deinit(g: *Gathered) void {
        for (g.dirs) |d| if (d) |x| x.close(g.io);
        g.arena.deinit();
    }

    /// This rank's fingerprint.
    pub fn build(g: *Gathered, gpa: std.mem.Allocator) !fingerprint.Fingerprint {
        return fingerprint.build(gpa, g.io, g.in);
    }
};

/// The inputs from model.Options `o`, the opened model `m` (its cfg: window, KV sources, ratios, vision; drafting())
/// and `x`.
pub fn gather(gpa: std.mem.Allocator, io: Io, o: anytype, m: anytype, x: Extra) !Gathered {
    if (x.persist) {
        try weightsIdentified(x.model_revision, x.rank_cache_sha);
        // the Engram tables (hundreds of GB) are identified by their recorded revision, not read
        if (o.engram_dir != null and x.engram_revision.len == 0) return error.NoEngramId;
    }
    var g: Gathered = .{ .in = undefined, .arena = .init(gpa), .io = io };
    errdefer g.deinit();
    const a = g.arena.allocator();
    const cwd = Io.Dir.cwd();
    const model_dir = try cwd.openDir(io, o.model_dir, .{});
    g.dirs[0] = model_dir;
    const kit = try cwd.openDir(io, o.kit_dir, .{});
    g.dirs[1] = kit;
    const cache = try cwd.openDir(io, o.cache_dir, .{});
    g.dirs[2] = cache;

    // the checkpoint: its recorded revision and its config.json's content
    const config_digest = if (exists(io, model_dir, "config.json")) try fingerprint.fileDigest(io, model_dir, "config.json") else @as(fingerprint.Digest, @splat(0));
    const model_revision = try std.fmt.allocPrint(a, "{s}|{x}", .{ x.model_revision, &config_digest });

    const aot: fingerprint.Source = if (o.aot_dir) |p| .{ .tree = .{ .dir = cwd, .path = p } } else treeOr(io, kit, "aot", null);
    const rdma: fingerprint.Source = if (o.rdma_kernels) |p| .{ .file = .{ .dir = cwd, .path = p } } else if (o.rdma_devices != null) .{ .bytes = "embedded" } else .none;

    const engram: []const u8 = if (o.engram_dir) |p| blk: {
        const ej = if (exists(io, kit, "engram.json")) try fingerprint.fileDigest(io, kit, "engram.json") else @as(fingerprint.Digest, @splat(0));
        break :blk try std.fmt.allocPrint(a, "{s}|{x}|{s}", .{ x.engram_revision, &ej, try listing(a, io, cwd, p) });
    } else "off";

    const vision: []const u8 = if (m.cfg.vision) blk: {
        const bias = if (o.vision_bias) |p| try fingerprint.fileDigest(io, cwd, p) else if (exists(io, kit, "vision/gate_bias_vl.safetensors")) try fingerprint.fileDigest(io, kit, "vision/gate_bias_vl.safetensors") else @as(fingerprint.Digest, @splat(0));
        const vdir = if (exists(io, kit, "vision")) try fingerprint.treeDigest(a, io, kit, "vision", null) else @as(fingerprint.Digest, @splat(0));
        break :blk try std.fmt.allocPrint(a, "{x}|{x}", .{ &bias, &vdir });
    } else "off";

    const dspark: []const u8 = if (o.drafts and m.drafting()) try std.fmt.allocPrint(a, "on|streams={d}|round_rows={d}", .{ o.streams, o.round_rows }) else "off";
    const math = try std.fmt.allocPrint(a, "idx_split={d};graphs={d};side={d};prefetch={d};engram_aio={d};", .{ @intFromBool(x.idx_split), @intFromBool(o.graphs), @intFromBool(o.side), @intFromBool(o.prefetch), @intFromBool(o.engram_aio) });

    const kv = m.cfg.kv_sources.slice();
    const layers = try a.alloc(u32, kv.len);
    const ratios = try a.alloc(u32, kv.len);
    for (kv, layers, ratios) |l, *lo, *ro| {
        lo.* = l;
        ro.* = m.cfg.ratios[l];
    }

    const sbuf = try a.alloc(u8, 128);
    const rbuf = try a.alloc(u8, 64);
    if (o.world > fingerprint.max_world or o.rank >= o.world) return error.BadRank;
    g.in = .{
        .native_format = x.native_format,
        .wire = try a.dupe(u8, &rank_wire.stampBytes()),
        .engine_commit = x.engine_id,
        .recipe_patches = x.recipe_patches,
        .build_flags = x.build_flags,
        .kit_aot = aot,
        .kit_cuda = treeOr(io, kit, "cubins", null),
        .kit_exl3 = treeOr(io, kit, ".", isExl3),
        .kit_rdma = rdma,
        .model_revision = model_revision,
        .rank_cache = try rankCacheIdentity(a, io, cache, x.rank_cache_name, x.rank_cache_sha),
        .ablation = x.ablation,
        .dspark = dspark,
        .tokenizer = fileOr(io, model_dir, "tokenizer.json"),
        .template = if (exists(io, model_dir, "chat_template.jinja")) fileOr(io, model_dir, "chat_template.jinja") else fileOr(io, model_dir, "tokenizer_config.json"),
        .engram = engram,
        .token_map = if (o.token_map) |p| .{ .file = .{ .dir = cwd, .path = p } } else .none,
        .rope = .{ .tree = .{ .dir = kit, .path = ".", .keep = isRope } },
        .vision = vision,
        .gpu_capability = x.gpu_capability,
        .cuda_driver = x.cuda_driver,
        .math = math,
        .chunk_rows = x.chunk_rows,
        .replay_floor = x.replay_floor,
        .window = @intCast(m.cfg.window),
        .source_layers = layers,
        .ratios = ratios,
        .planes = x.planes,
        .world = @intCast(o.world),
        .split = try splitText(sbuf, o.world),
        .owner_layout = x.owner_layout,
        .owner_version = x.owner_version,
        .parts = x.parts,
        .features = x.features,
        .rank = @intCast(o.rank),
        .rank_split = try rankSplitText(rbuf, o.rank, o.world),
        .rank_layout = x.rank_layout,
    };
    return g;
}

// -- tests ------------------------------------------------------------------------------------------------------------

const testing = std.testing;

const FakeIds = struct {
    v: [4]u16 = .{ 2, 8, 14, 20 },
    pub fn slice(s: *const FakeIds) []const u16 {
        return &s.v;
    }
};

const FakeModel = struct {
    cfg: struct { vision: bool = false, window: usize = 128, kv_sources: FakeIds = .{}, ratios: [43]u8 = blk: {
        var r: [43]u8 = @splat(0);
        for (2..20) |i| r[i] = 2;
        for (20..40) |i| r[i] = 1;
        break :blk r;
    } } = .{},
    draft_on: bool = true,
    pub fn drafting(m: *const FakeModel) bool {
        return m.draft_on;
    }
};

/// model.Options' fields gather reads.
const FakeOptions = struct {
    model_dir: []const u8,
    cache_dir: []const u8,
    kit_dir: []const u8,
    aot_dir: ?[]const u8 = null,
    rank: u32,
    world: u32 = 2,
    engram_dir: ?[]const u8 = null,
    token_map: ?[]const u8 = null,
    drafts: bool = true,
    rdma_devices: ?[]const u8 = null,
    rdma_kernels: ?[]const u8 = null,
    graphs: bool = false,
    side: bool = false,
    prefetch: bool = false,
    engram_aio: bool = false,
    round_rows: usize = 16,
    streams: usize = 8,
    vision_bias: ?[]const u8 = null,
};

/// A rank cache file in rank_cache.zig's format: `body` (the tensors' bytes), the JSON `index`, its offset, the magic.
fn writeRankCache(io: Io, dir: Io.Dir, name: []const u8, index: []const u8, body: []const u8) !void {
    var buf: [256]u8 = undefined;
    @memcpy(buf[0..body.len], body);
    @memcpy(buf[body.len..][0..index.len], index);
    std.mem.writeInt(u64, buf[body.len + index.len ..][0..8], body.len, .little);
    @memcpy(buf[body.len + index.len + 8 ..][0..8], rank_cache.magic);
    try dir.writeFile(io, .{ .sub_path = name, .data = buf[0 .. body.len + index.len + 16] });
}

fn testEnv(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_DS_ENGINE_ID")) return "629c524";
    if (std.mem.eql(u8, name, "TF_DS_RECIPE_PATCHES")) return "e65785c";
    if (std.mem.eql(u8, name, "TF_DS_MODEL_REVISION")) return "deepseek-v4.1-flash-1017@exl3-3.5";
    if (std.mem.eql(u8, name, "TF_DS_ENGRAM_REVISION")) return "engram-2026-10-17";
    return null;
}

fn noWeightsEnv(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_DS_ENGINE_ID")) return "629c524";
    if (std.mem.eql(u8, name, "TF_DS_RECIPE_PATCHES")) return "e65785c";
    return null;
}

fn noEnv(_: [:0]const u8) ?[]const u8 {
    return null;
}

test "two ranks of one install agree on the common digest, and a kit change is named" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const d = tmp.dir;
    for ([_][]const u8{ "model", "cache", "kitA/cubins", "kitA/aot", "kitB/cubins", "kitB/aot", "engram" }) |p| try d.createDirPath(io, p);
    try d.writeFile(io, .{ .sub_path = "model/config.json", .data = "{\"model_type\": \"deepseek_v41\"}" });
    try d.writeFile(io, .{ .sub_path = "model/tokenizer.json", .data = "tok" });
    try d.writeFile(io, .{ .sub_path = "engram/shard0.bin", .data = "0123" });
    for ([_][]const u8{ "kitA", "kitB" }) |k| {
        var kd = try d.openDir(io, k, .{});
        defer kd.close(io);
        try kd.writeFile(io, .{ .sub_path = "cubins/a.cubin", .data = "A" });
        try kd.writeFile(io, .{ .sub_path = "aot/aot.json", .data = "{}" });
        try kd.writeFile(io, .{ .sub_path = "rope-plain-cos.f32", .data = "rope" });
        try kd.writeFile(io, .{ .sub_path = "engram.json", .data = "{}" });
    }
    try writeRankCache(io, d, "cache/rank0.st", "{\"w\":0}", "weights0");
    try writeRankCache(io, d, "cache/rank1.st", "{\"w\":1}", "weights1");
    var root_buf: [1024]u8 = undefined;
    const root = root_buf[0..try d.realPath(io, &root_buf)];
    const path = struct {
        fn of(a: std.mem.Allocator, r: []const u8, p: []const u8) ![]u8 {
            return std.fs.path.join(a, &.{ r, p });
        }
    }.of;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const envp = try Extra.env(testEnv);
    var fbuf: [256]u8 = undefined;
    const features = try featureString(&fbuf, .{ .spill = true, .streams = 8, .round_rows = 16, .pool = 1 << 20, .drafts = true });
    const base: Extra = .{
        .engine_id = envp.engine_id,
        .recipe_patches = envp.recipe_patches,
        .model_revision = envp.model_revision,
        .engram_revision = envp.engram_revision,
        .features = features,
        .native_format = 1,
        .chunk_rows = 2048,
        .replay_floor = false,
        .idx_split = false,
        .planes = "planes",
        .rank_layout = @splat(1),
        .gpu_capability = 121,
        .cuda_driver = 13000,
        .rank_cache_name = "rank0.st",
    };
    const m: FakeModel = .{};
    var fps: [3]fingerprint.Fingerprint = undefined;
    for (0..3) |i| {
        const o: FakeOptions = .{
            .model_dir = try path(a, root, "model"),
            .cache_dir = try path(a, root, "cache"),
            .kit_dir = try path(a, root, if (i == 2) "kitB" else "kitA"),
            .rank = @intCast(i % 2),
            .engram_dir = try path(a, root, "engram"),
        };
        var x = base;
        x.rank_cache_name = if (i % 2 == 0) "rank0.st" else "rank1.st";
        x.rank_layout = @splat(@intCast(1 + i % 2));
        var g = try gather(gpa, io, o, &m, x);
        defer g.deinit();
        fps[i] = try g.build(gpa);
        if (i == 1) try d.writeFile(io, .{ .sub_path = "kitB/cubins/a.cubin", .data = "B" }); // rank "2" sees a changed kit
    }
    // ranks 0 and 1: same common digest, different rank digests
    try testing.expectEqualSlices(u8, &fps[0].common(), &fps[1].common());
    try testing.expect(!std.mem.eql(u8, &fps[0].rankDigest(), &fps[1].rankDigest()));
    // a third start with another cubin: kit_cuda is the field named
    const x0 = fps[0].exchange();
    const x2 = fps[2].exchange();
    try testing.expectEqual(@as(?u16, @intFromEnum(fingerprint.Field.kit_cuda)), x0.differs(&x2));
    try testing.expectError(error.NoEngineId, Extra.env(noEnv));
    try testing.expectError(error.NoWeightsId, Extra.env(noWeightsEnv));

    // rank 0 again, its file regenerated from other weights of the same shapes (same header and size): only the
    // recorded digest tells them apart, and it changes rank_cache; another Engram revision changes engram
    const o0: FakeOptions = .{
        .model_dir = try path(a, root, "model"),
        .cache_dir = try path(a, root, "cache"),
        .kit_dir = try path(a, root, "kitA"),
        .rank = 0,
        .engram_dir = try path(a, root, "engram"),
    };
    var x = base;
    x.rank_cache_sha = "sha256:1111";
    try writeRankCache(io, d, "cache/rank0.st", "{\"w\":0}", "weights9");
    var g1 = try gather(gpa, io, o0, &m, x);
    defer g1.deinit();
    const f1 = try g1.build(gpa);
    x.rank_cache_sha = "sha256:2222";
    var g2 = try gather(gpa, io, o0, &m, x);
    defer g2.deinit();
    const f2 = try g2.build(gpa);
    // (a rank field: each rank's own, so it moves the rank digest and its store's compat, not the common digest)
    try testing.expect(!std.mem.eql(u8, &f1.get(.rank_cache), &f2.get(.rank_cache)));
    try testing.expect(!std.mem.eql(u8, &f1.rankDigest(), &f2.rankDigest()));
    try testing.expectEqualSlices(u8, &f1.common(), &f2.common());
    x.engram_revision = "engram-2026-10-18";
    var g3 = try gather(gpa, io, o0, &m, x);
    defer g3.deinit();
    const f3 = try g3.build(gpa);
    try testing.expectEqual(@as(?u16, @intFromEnum(fingerprint.Field.engram)), f2.exchange().differs(&f3.exchange()));
    // Engram without its revision, weights without any content identity: refused
    x.engram_revision = "";
    try testing.expectError(error.NoEngramId, gather(gpa, io, o0, &m, x));
    x = base;
    x.model_revision = "";
    try testing.expectError(error.NoWeightsId, gather(gpa, io, o0, &m, x));
}

test "the rank cache's identity reads its index (the file's footer), not its weights" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeRankCache(io, tmp.dir, "a", "{\"t\":1}", "xxxx");
    try writeRankCache(io, tmp.dir, "b", "{\"t\":1}", "yyyy");
    const ia = try rankCacheIdentity(gpa, io, tmp.dir, "a", "");
    defer gpa.free(ia);
    const ib = try rankCacheIdentity(gpa, io, tmp.dir, "b", "");
    defer gpa.free(ib);
    // same size and index: only the name differs (a recorded digest is how content differences are caught)
    try testing.expectEqualStrings(ia[1..], ib[1..]);
    try writeRankCache(io, tmp.dir, "c", "{\"t\":2}", "xxxx");
    const ic = try rankCacheIdentity(gpa, io, tmp.dir, "c", "");
    defer gpa.free(ic);
    try testing.expect(!std.mem.eql(u8, ia[1..], ic[1..]));
    try tmp.dir.writeFile(io, .{ .sub_path = "bad", .data = &.{ 0xff, 0xff, 0, 0, 0, 0, 0, 0, 1 } });
    try testing.expectError(error.BadRankCache, rankCacheIdentity(gpa, io, tmp.dir, "bad", ""));
    // the file the loader itself reads (rank_cache.writeFake, weights.py's layout) is identified, and its index is what counts
    var fake = std.heap.ArenaAllocator.init(gpa);
    defer fake.deinit();
    try rank_cache.writeFake(fake.allocator(), io, tmp.dir, "rank0of2-0123456789abcdef0123.bin", &.{.{ .key = "w", .dtype = "bfloat16", .shape = &.{ 2, 4 }, .fill = 7 }});
    const iw = try rankCacheIdentity(gpa, io, tmp.dir, "rank0of2-0123456789abcdef0123.bin", "sha256:ab");
    defer gpa.free(iw);
    try testing.expect(std.mem.endsWith(u8, iw, "|sha256:ab"));
    var ix = try rank_cache.open(gpa, io, tmp.dir, "rank0of2-0123456789abcdef0123.bin");
    defer ix.deinit();
    var sb: [64]u8 = undefined;
    try testing.expectEqualStrings("0:rank=0,pair=0;1:rank=1,pair=0;2:rank=0,pair=1;3:rank=1,pair=1;", try splitText(&sb, 4));
}

test "the state layout feeds planes, rank_layout and owner_layout; the two owner layouts never agree" {
    const config = @import("config.zig");
    var why: config.Why = .{};
    const c = try config.parse(testing.allocator, config.test_config, &why);
    const g = spill_state.Geometry.of(&c, 2048, 16, 64);
    const rep = try spill_state.StateLayout.fromConfig(&c, g, .replicated);
    const sh = try spill_state.StateLayout.fromConfig(&c, g, .owner_sharded);
    var b1: [spill_state.StateLayout.max_encoded]u8 = undefined;
    var b2: [spill_state.StateLayout.max_encoded]u8 = undefined;
    const p1 = try layoutPart(&rep, 2, &b1);
    try testing.expectError(error.BadOwnerLayout, layoutPart(&sh, 2, &b2)); // (not cut into parts: it could not pack)
    const cut = try sh.withParts(1 << 20);
    const p2 = try layoutPart(&cut, 2, &b2);
    try testing.expectEqual(fingerprint.OwnerLayout.replicated, p1.owner_layout);
    try testing.expectEqual(fingerprint.OwnerLayout.owner_sharded, p2.owner_layout);
    try testing.expectEqual(@as(u8, 1), p1.parts);
    try testing.expectEqual(@as(u16, 0), p1.owner_version);
    try testing.expectEqual(@as(u8, 2), p2.parts);
    try testing.expectEqual(parts.layout_version, p2.owner_version);
    try testing.expectEqualSlices(u8, &rep.digest(), &p1.rank_layout);
    try testing.expect(!std.mem.eql(u8, p1.planes, p2.planes));
    try testing.expect(!std.mem.eql(u8, &p1.rank_layout, &p2.rank_layout));
    // another part size: other planes (the common field), so a pool change is a mismatch at the hello
    var b3: [spill_state.StateLayout.max_encoded]u8 = undefined;
    const p3 = try layoutPart(&(try sh.withParts(1 << 19)), 2, &b3);
    try testing.expect(!std.mem.eql(u8, p2.planes, p3.planes));
}

test "the features: off as they always were; TF_DS_KV_SHARD adds its switches and the part size" {
    var b1: [256]u8 = undefined;
    var b2: [256]u8 = undefined;
    const off = try featureString(&b1, .{ .spill = true, .spill_min_tokens = 8192, .streams = 8, .round_rows = 16, .pool = 1 << 20, .drafts = true });
    try testing.expectEqualStrings("spill=1;spill_min=8192;kv_shard=0;kv_carry=0;twins=0;engram_lanes=1;image_cache=1;streams=8;round_rows=16;pool=1048576;drafts=1;", off);
    const on = try featureString(&b2, .{ .spill = true, .spill_min_tokens = 8192, .kv_shard = true, .kv_carry = true, .twins = false, .part_rows = 1050624, .streams = 8, .round_rows = 16, .pool = 2101248, .drafts = true });
    try testing.expectEqualStrings("spill=1;spill_min=8192;kv_shard=1;kv_carry=1;twins=0;engram_lanes=1;image_cache=1;streams=8;round_rows=16;pool=2101248;drafts=1;part=1050624;", on);
}

test "TF_DS_KV_SHARD alone: a missing recipe identity reads unset, never a refusal; with the spill tier it is refused" {
    const e = try Extra.envFor(noEnv, false);
    try testing.expectEqualStrings("unset", e.engine_id);
    try testing.expectEqualStrings("unset", e.recipe_patches);
    try testing.expectError(error.NoEngineId, Extra.envFor(noEnv, true));
    const k = try Extra.envFor(testEnv, false);
    try testing.expectEqualStrings("629c524", k.engine_id);
}

// PETABRIDGE (k2a-pb1): TP3 (P3). The split text of three ranks, and three gathered fingerprints: rank 0 serves from
// a folder whose chat template is the recipe's overlay (k2a's head_chat_template), ranks 1 and 2 read the plain
// checkpoint. The template is a rank field, so the three still agree; a follower's other tokenizer is a common field.
test "PETABRIDGE TP3: three ranks' split, and the head's own chat template agrees while a common field does not" {
    var sb: [128]u8 = undefined;
    try testing.expectEqualStrings("0:rank=0;1:rank=1;2:rank=2;", try splitText(&sb, 3));
    try testing.expectEqualStrings("0:rank=0;1:rank=1;", try splitText(&sb, 2)); // (unchanged)
    var rb: [64]u8 = undefined;
    try testing.expectEqualStrings("rank=2", try rankSplitText(&rb, 2, 3));
    try testing.expectEqualStrings("rank=1,pair=1", try rankSplitText(&rb, 3, 4)); // (unchanged)
    try testing.expectError(error.UnsupportedWorld, splitText(&sb, 1));
    try testing.expectError(error.UnsupportedWorld, rankSplitText(&rb, 0, 5));

    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const d = tmp.dir;
    for ([_][]const u8{ "served", "model", "other", "cache", "kit/cubins", "kit/aot" }) |p| try d.createDirPath(io, p);
    for ([_][]const u8{ "served", "model", "other" }) |m| {
        var md = try d.openDir(io, m, .{});
        defer md.close(io);
        try md.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\": \"deepseek_v41\"}" });
        try md.writeFile(io, .{ .sub_path = "tokenizer.json", .data = if (std.mem.eql(u8, m, "other")) "tok2" else "tok" });
        try md.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"chat_template\": \"plain\"}" });
    }
    try d.writeFile(io, .{ .sub_path = "served/chat_template.jinja", .data = "{{ k2a head overlay }}" });
    try d.writeFile(io, .{ .sub_path = "kit/cubins/a.cubin", .data = "A" });
    try d.writeFile(io, .{ .sub_path = "kit/aot/aot.json", .data = "{}" });
    try d.writeFile(io, .{ .sub_path = "kit/rope-plain-cos.f32", .data = "rope" });
    for (0..3) |r| {
        var nb: [32]u8 = undefined;
        var ib: [16]u8 = undefined;
        try writeRankCache(io, d, try std.fmt.bufPrint(&nb, "cache/rank{d}of3.bin", .{r}), try std.fmt.bufPrint(&ib, "{{\"w\":{d}}}", .{r}), "weights");
    }
    var root_buf: [1024]u8 = undefined;
    const root = root_buf[0..try d.realPath(io, &root_buf)];
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const envp = try Extra.env(testEnv);
    var fbuf: [256]u8 = undefined;
    const features = try featureString(&fbuf, .{ .spill = true, .streams = 8, .round_rows = 48, .pool = 1 << 20, .drafts = true });
    const m: FakeModel = .{};
    // ranks 0, 1, 2 of the trio, then rank 2 again reading a checkpoint with another tokenizer
    var fps: [4]fingerprint.Fingerprint = undefined;
    for (0..4) |i| {
        const rank: u32 = if (i == 3) 2 else @intCast(i);
        const dir = if (i == 0) "served" else if (i == 3) "other" else "model";
        const o: FakeOptions = .{
            .model_dir = try std.fs.path.join(a, &.{ root, dir }),
            .cache_dir = try std.fs.path.join(a, &.{ root, "cache" }),
            .kit_dir = try std.fs.path.join(a, &.{ root, "kit" }),
            .rank = rank,
            .world = 3,
            .round_rows = 48,
        };
        const x: Extra = .{
            .engine_id = envp.engine_id,
            .recipe_patches = envp.recipe_patches,
            .model_revision = envp.model_revision,
            .features = features,
            .native_format = 1,
            .chunk_rows = 2048,
            .replay_floor = false,
            .idx_split = false,
            .planes = "planes",
            .rank_layout = @splat(1), // (replicated KV: every rank holds the same layout)
            .gpu_capability = 121,
            .cuda_driver = 13000,
            .rank_cache_name = try std.fmt.allocPrint(a, "rank{d}of3.bin", .{rank}),
        };
        var g = try gather(gpa, io, o, &m, x);
        defer g.deinit();
        try testing.expectEqual(@as(u16, 3), g.in.world);
        fps[i] = try g.build(gpa);
    }
    // the head's template differs from the followers' (theirs agree), and it is a rank field
    try testing.expectEqual(fingerprint.Scope.rank, fingerprint.scope(.template));
    try testing.expect(!std.mem.eql(u8, &fps[0].get(.template), &fps[1].get(.template)));
    try testing.expectEqualSlices(u8, &fps[1].get(.template), &fps[2].get(.template));
    // so the three agree on the common digest and on every common field; each has its own rank digest
    try testing.expectEqualSlices(u8, &fps[0].common(), &fps[1].common());
    try testing.expectEqualSlices(u8, &fps[0].common(), &fps[2].common());
    try testing.expectEqualSlices(u8, &fps[0].get(.tp_split), &fps[2].get(.tp_split));
    const x0 = fps[0].exchange();
    for (fps[1..3]) |*f| {
        try testing.expectEqual(@as(?u16, null), x0.differs(&f.exchange()));
        try testing.expect(!std.mem.eql(u8, &fps[0].rankDigest(), &f.rankDigest()));
    }
    try testing.expect(!std.mem.eql(u8, &fps[1].rankDigest(), &fps[2].rankDigest()));
    // rank 2 on a checkpoint with another tokenizer: a common field, named
    try testing.expectEqual(@as(?u16, @intFromEnum(fingerprint.Field.tokenizer)), x0.differs(&fps[3].exchange()));
}
