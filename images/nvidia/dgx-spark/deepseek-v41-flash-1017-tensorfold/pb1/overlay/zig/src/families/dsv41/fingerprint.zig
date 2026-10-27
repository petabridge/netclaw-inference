//! The ranks' compatibility fingerprint. Port of wojo's GLM 0088/0110 spill identity (`build_id`, `weights_id` and
//! the spill signature of PR #78/#124) and 0118's owner-layout identity (PR #152), as PORT-SPILL-V2.md "Persistent
//! format and exactness identity" asks of the native engine: one SHA-256 digest per field over everything that can
//! change the bytes a kept state holds or the commands the ranks exchange, a common digest over the fields every rank
//! must share, and a per-rank digest over the fields that are each rank's own (its weight file, its rank's split, its
//! state layout). The fields are named, so a startup disagreement says which one differs (rank_wire.zig's hello).
//!
//! Every field must be given (Builder.finish refuses a missing one): a value a configuration lacks is given as an
//! explicit "off"/"none", never left out, so a later input can never silently join an old digest. A field may be
//! built from several parts (each length-prefixed, so ("ab","c") and ("a","bc") differ). Files are hashed by content
//! (putFile), kit folders as a sorted manifest of paths, sizes and content digests (putTree); huge immutable inputs
//! (the rank cache, Engram tables) are identified by their recorded revisions/hashes, given as bytes.
//! Paths, inodes and mtimes are never part of a digest: a copied, identical file must agree.
const std = @import("std");
const builtin = @import("builtin");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Digest = [32]u8;

/// Bumped when the digest construction itself changes (every stored point and every peer then disagrees).
pub const version: u32 = 2; // 2: the chat template became a per-rank field

/// Whether every rank must hold the same value (`common`) or each rank its own (`rank`, exchanged in rank order).
pub const Scope = enum(u8) { common = 1, rank = 2 };

/// The fingerprint's fields, by stable id (ids are on the wire and in digests: never renumber, only append).
pub const Field = enum(u16) {
    native_format = 1, // the spill store's on-disk format version (core/spill format)
    wire_version = 2, // rank_wire.version, its header and the shard-op schema (rank_wire.stamp)
    engine_commit = 3, // the engine's source commit
    recipe_patches = 4, // the recipe's patch series digest (zig/patches, in apply order)
    build_flags = 5, // compiler and build flags (the builtin facts are added by Builder.init)
    kit_aot = 6, // the AOT kernel set (TENSORFOLD_CUDA_KERNELS: aot.json and its cubins)
    kit_cuda = 7, // the kit's served extension cubins (TF_DS_KIT/cubins)
    kit_exl3 = 8, // the EXL3 kernels' images
    kit_rdma = 9, // the RDMA gather fatbin (TF_DS_RDMA_KERNELS or the embedded image; "none" without RDMA)
    model_revision = 10, // the checkpoint's revision (config, index, recorded revision id)
    rank_cache = 11, // this rank's generated weight file identity
    ablation = 12, // the ablation overlay's identity when enabled, else "off"
    dspark = 13, // DSpark (drafter) weights and policy, else "off"
    tokenizer = 14, // tokenizer identity
    template = 15, // chat template identity
    engram = 16, // Engram shard revisions and hashes (engram.json), else "off"
    token_map = 17, // the compressed token map
    rope = 18, // the served RoPE tables
    vision = 19, // vision preprocessing and routing bias, else "off"
    gpu = 20, // GPU architecture (compute capability)
    cuda = 21, // CUDA driver version and the math/runtime choices that change bits
    chunk_rows = 22, // the prompt chunk (model.chunk_rows)
    replay = 23, // bounded replay policy and floor (prompt.replay_floor, the window)
    source_layers = 24, // KV-source layer mapping and ratios
    planes = 25, // plane order, widths, dtypes, ratios and ring layouts (common part of the state layout)
    tp_split = 26, // TP world and the whole rank split plan
    owner_layout = 27, // replicated or owner_sharded, owner mapping version, parts
    features = 28, // the switches that change state or protocol (spill, shard, carry, idx split, engram lanes ...)
    rank_split = 29, // this rank's index and its part of the split
    rank_layout = 30, // this rank's state layout digest (spill_state.StateLayout.digest)
};

pub const n_fields = @typeInfo(Field).@"enum".field_names.len;

pub fn scope(f: Field) Scope {
    return switch (f) {
        // the chat template is rank 0's alone (it renders prompts; followers take ids, and the served folder carries the
        // recipe's template while a follower may read the plain checkpoint): a kept state depends on the ids, not on how
        // they were made, so the ranks need not agree on it
        .rank_cache, .rank_split, .rank_layout, .template => .rank,
        else => .common,
    };
}

pub fn name(f: Field) []const u8 {
    return @tagName(f);
}

/// A field by its wire id, or null for an id this build does not know (a peer's newer field).
pub fn fieldOf(id: u16) ?Field {
    return std.enums.fromInt(Field, id);
}

fn index(f: Field) usize {
    inline for (@typeInfo(Field).@"enum".field_values, 0..) |v, i| {
        if (@intFromEnum(f) == v) return i;
    }
    unreachable;
}

/// The fields in id order (the order every digest walks them in).
pub const all: [n_fields]Field = blk: {
    var out: [n_fields]Field = undefined;
    for (@typeInfo(Field).@"enum".field_values, 0..) |v, i| out[i] = @enumFromInt(v);
    break :blk out;
};

/// Builds the per-field digests. Feed each field with part/int/putFile/putTree, then finish.
pub const Builder = struct {
    h: [n_fields]?Sha256 = @splat(null),

    /// A builder with build_flags already holding the compiler's own facts (Zig version, optimize mode, target
    /// triple; not the CPU model): the caller adds the flags it chose (-D options, recipe flags).
    pub fn init() Builder {
        var b: Builder = .{};
        b.part(.build_flags, builtin.zig_version_string);
        b.part(.build_flags, @tagName(builtin.mode));
        b.part(.build_flags, @tagName(builtin.cpu.arch));
        // not builtin.cpu.model: a GB10 has Cortex-X925 and A725 cores and the native build resolves the model per
        // compiler process (one image's server and follower binaries differed), while no GPU arithmetic or kept byte
        // depends on the host CPU model
        b.part(.build_flags, @tagName(builtin.os.tag));
        b.part(.build_flags, @tagName(builtin.abi));
        return b;
    }

    fn hasher(b: *Builder, f: Field) *Sha256 {
        const i = index(f);
        if (b.h[i] == null) {
            var s = Sha256.init(.{});
            s.update("tf-dsv41-fingerprint\x00");
            var head: [6]u8 = undefined;
            std.mem.writeInt(u32, head[0..4], version, .little);
            std.mem.writeInt(u16, head[4..6], @intFromEnum(f), .little);
            s.update(&head);
            b.h[i] = s;
        }
        return &b.h[i].?;
    }

    /// One part of field `f`: its length (u64) then its bytes.
    pub fn part(b: *Builder, f: Field, bytes: []const u8) void {
        const s = b.hasher(f);
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, bytes.len, .little);
        s.update(&len);
        s.update(bytes);
    }

    /// An integer part (little-endian, its width fixed by T: u32 3 and u64 3 differ).
    pub fn int(b: *Builder, f: Field, comptime T: type, v: T) void {
        var buf: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &buf, v, .little);
        b.part(f, &buf);
    }

    /// A list of integers as one part (its count, then each).
    pub fn ints(b: *Builder, f: Field, comptime T: type, vs: []const T) void {
        b.int(f, u64, vs.len);
        for (vs) |v| b.int(f, T, v);
    }

    /// A file's content digest as one part (streamed; its name and location are not part of it).
    pub fn putFile(b: *Builder, f: Field, io: std.Io, dir: std.Io.Dir, sub_path: []const u8) !void {
        b.part(f, &try fileDigest(io, dir, sub_path));
    }

    /// A folder's manifest as one part: every regular file below it, sorted by relative path ('/' separated), each
    /// its path, size and content digest. `keep` (null: all) picks the files by relative path. An empty match counts
    /// (an empty kit folder differs from a populated one, and both from "none").
    pub fn putTree(b: *Builder, f: Field, gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, sub_path: []const u8, keep: ?*const fn ([]const u8) bool) !void {
        b.part(f, &try treeDigest(gpa, io, dir, sub_path, keep));
    }

    /// The digests; error.MissingField names (in `missing`) the first field never given.
    pub fn finish(b: *const Builder, missing: ?*Field) !Fingerprint {
        var fp: Fingerprint = .{};
        for (all, 0..) |f, i| {
            var s = b.h[i] orelse {
                if (missing) |m| m.* = f;
                return error.MissingField;
            };
            s.final(&fp.digests[i]);
        }
        return fp;
    }
};

/// A file's SHA-256 over its content.
pub fn fileDigest(io: std.Io, dir: std.Io.Dir, sub_path: []const u8) !Digest {
    var file = try dir.openFile(io, sub_path, .{});
    defer file.close(io);
    var s = Sha256.init(.{});
    var buf: [64 << 10]u8 = undefined;
    var at: u64 = 0;
    while (true) {
        const n = try file.readPositional(io, &.{&buf}, at);
        if (n == 0) break;
        s.update(buf[0..n]);
        at += n;
    }
    var out: Digest = undefined;
    s.final(&out);
    return out;
}

/// A folder's manifest digest (Builder.putTree).
pub fn treeDigest(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, sub_path: []const u8, keep: ?*const fn ([]const u8) bool) !Digest {
    var root = try dir.openDir(io, sub_path, .{ .iterate = true });
    defer root.close(io);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    var walk = try root.walk(gpa);
    defer walk.deinit();
    while (try walk.next(io)) |e| {
        if (e.kind != .file) continue;
        const rel = try gpa.dupe(u8, e.path);
        // one separator on every host, so a manifest is the same wherever it is built
        for (rel) |*ch| if (ch.* == std.fs.path.sep) {
            ch.* = '/';
        };
        if (keep) |k| if (!k(rel)) {
            gpa.free(rel);
            continue;
        };
        names.append(gpa, rel) catch |err| {
            gpa.free(rel);
            return err;
        };
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn lt(_: void, x: []u8, y: []u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    var s = Sha256.init(.{});
    s.update("tf-dsv41-tree\x00");
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, names.items.len, .little);
    s.update(&len);
    for (names.items) |n| {
        std.mem.writeInt(u64, &len, n.len, .little);
        s.update(&len);
        s.update(n);
        const st = try root.statFile(io, n, .{});
        std.mem.writeInt(u64, &len, st.size, .little);
        s.update(&len);
        s.update(&try fileDigest(io, root, n));
    }
    var out: Digest = undefined;
    s.final(&out);
    return out;
}

/// Every field's digest, in id order.
pub const Fingerprint = struct {
    digests: [n_fields]Digest = undefined,

    pub fn get(fp: *const Fingerprint, f: Field) Digest {
        return fp.digests[index(f)];
    }

    /// The digest every rank must share: H(version, then each common field's id and digest in id order).
    pub fn common(fp: *const Fingerprint) Digest {
        return fp.over(.common);
    }

    /// This rank's own digest: the same over the rank-scoped fields.
    pub fn rankDigest(fp: *const Fingerprint) Digest {
        return fp.over(.rank);
    }

    fn over(fp: *const Fingerprint, sc: Scope) Digest {
        var s = Sha256.init(.{});
        s.update(if (sc == .common) "tf-dsv41-common\x00" else "tf-dsv41-rank\x00");
        var v: [4]u8 = undefined;
        std.mem.writeInt(u32, &v, version, .little);
        s.update(&v);
        for (all, 0..) |f, i| {
            if (scope(f) != sc) continue;
            var id: [2]u8 = undefined;
            std.mem.writeInt(u16, &id, @intFromEnum(f), .little);
            s.update(&id);
            s.update(&fp.digests[i]);
        }
        var out: Digest = undefined;
        s.final(&out);
        return out;
    }

    /// What rank_wire's hello carries: every field's id, scope and digest, and the two summaries.
    pub fn exchange(fp: *const Fingerprint) Exchange {
        var x: Exchange = .{ .common = fp.common(), .rank = fp.rankDigest(), .n = n_fields };
        for (all, 0..) |f, i| x.fields[i] = .{ .id = @intFromEnum(f), .scope = scope(f), .digest = fp.digests[i] };
        return x;
    }
};

/// The most fields a peer's exchange may list (a newer peer may know more fields than this build; the bound keeps the
/// wire's parse fixed).
pub const max_fields = 64;

pub const Entry = struct { id: u16, scope: Scope, digest: Digest };

/// A rank's fingerprint as exchanged at startup.
pub const Exchange = struct {
    common: Digest,
    rank: Digest,
    n: u16 = 0,
    fields: [max_fields]Entry = undefined,

    pub fn entries(x: *const Exchange) []const Entry {
        return x.fields[0..x.n];
    }

    fn find(x: *const Exchange, id: u16) ?Entry {
        for (x.entries()) |e| if (e.id == id) return e;
        return null;
    }

    /// The first common field (by id) on which `x` and `y` disagree, a field one lists and the other lacks, or a
    /// field the two scope differently; null when every common field agrees (then their common digests agree too,
    /// unless the construction differs, which `differs` reports as wire_version). Rank fields are each rank's own.
    pub fn firstDifference(x: *const Exchange, y: *const Exchange) ?u16 {
        var best: ?u16 = null;
        for ([_]*const Exchange{ x, y }, [_]*const Exchange{ y, x }) |a, b| {
            for (a.entries()) |e| {
                const o = b.find(e.id) orelse {
                    best = minId(best, e.id);
                    continue;
                };
                if (o.scope != e.scope) {
                    best = minId(best, e.id);
                    continue;
                }
                if (e.scope == .common and !std.mem.eql(u8, &e.digest, &o.digest)) best = minId(best, e.id);
            }
        }
        return best;
    }

    /// Whether the two disagree at all on what must be common: a differing field (its id), or, with every listed
    /// field equal, differing common digests (a construction difference, reported as wire_version).
    pub fn differs(x: *const Exchange, y: *const Exchange) ?u16 {
        if (x.firstDifference(y)) |id| return id;
        if (!std.mem.eql(u8, &x.common, &y.common)) return @intFromEnum(Field.wire_version);
        return null;
    }
};

fn minId(a: ?u16, b: u16) u16 {
    return if (a) |x| @min(x, b) else b;
}

/// A field id in words ("engine_commit", or "field 41" for one this build does not know).
pub fn describe(id: u16, buf: []u8) []const u8 {
    if (fieldOf(id)) |f| return name(f);
    return std.fmt.bufPrint(buf, "field {d}", .{id}) catch "field ?";
}

/// What the ranks agreed at startup: the common digest and every rank's own, in rank order.
pub const Agreement = struct {
    common: Digest,
    world: u16,
    ranks: [max_world]Digest = @splat(@splat(0)),

    /// The digest of the whole pair (common, then the per-rank digests in order).
    pub fn pair(a: *const Agreement) Digest {
        var s = Sha256.init(.{});
        s.update("tf-dsv41-pair\x00");
        s.update(&a.common);
        var w: [2]u8 = undefined;
        std.mem.writeInt(u16, &w, a.world, .little);
        s.update(&w);
        for (a.ranks[0..a.world]) |r| s.update(&r);
        var out: Digest = undefined;
        s.final(&out);
        return out;
    }

    /// The `common` a rank's spill store is opened with (core/spill Config.compat via format.compatDigest): the
    /// pair's agreement narrowed to that rank, so a store is only ever read by the rank that wrote it, under the
    /// same common configuration and its own weights/layout.
    pub fn storeCommon(a: *const Agreement, rank: u16) Digest {
        std.debug.assert(rank < a.world);
        var s = Sha256.init(.{});
        s.update("tf-dsv41-store\x00");
        s.update(&a.common);
        var w: [4]u8 = undefined;
        std.mem.writeInt(u16, w[0..2], rank, .little);
        std.mem.writeInt(u16, w[2..4], a.world, .little);
        s.update(&w);
        s.update(&a.ranks[rank]);
        var out: Digest = undefined;
        s.final(&out);
        return out;
    }
};

/// The most ranks a run links (the four-node split).
pub const max_world = 4;

/// The inputs a DeepSeek-V4.1 rank's fingerprint is built from, gathered by native.zig / lanes_main.zig (round 2):
/// strings and byte strings are given as they are (the builder length-prefixes them), files and folders are hashed
/// here. "off"/"none" for an input the configuration does not use.
pub const Inputs = struct {
    native_format: u32, // core/spill's format version
    wire: []const u8, // rank_wire.stampBytes()
    engine_commit: []const u8, // full commit id of the engine source
    recipe_patches: []const u8, // digest (hex) of the applied recipe patches, in order
    build_flags: []const u8, // the chosen build options (-D..., recipe zig_flags.py), "" when none
    kit_aot: Source, // TENSORFOLD_CUDA_KERNELS
    kit_cuda: Source, // TF_DS_KIT/cubins
    kit_exl3: Source, // the EXL3 kernel images
    kit_rdma: Source, // TF_DS_RDMA_KERNELS, the embedded image's bytes, or none
    model_revision: []const u8, // the checkpoint's revision id and config.json bytes (as given)
    rank_cache: []const u8, // this rank's weight file identity (its index header, size, recorded digest)
    ablation: []const u8, // overlay identity, or "off"
    dspark: []const u8, // drafter weights identity and policy (streams, depth), or "off"
    tokenizer: Source,
    template: Source,
    engram: []const u8, // engram.json's revisions and shard hashes, or "off"
    token_map: Source,
    rope: Source, // the served RoPE tables (TF_DS_KIT/rope-*.f32)
    vision: []const u8, // vision preprocessing + routing bias identity, or "off"
    gpu_capability: u32, // 10 * major + minor (cuda Context.capability)
    cuda_driver: i32, // cuda Driver.version: 1000 * major + 10 * minor
    math: []const u8, // math/runtime choices that change bits (graphs, side stream, L2 prefetch, AIO ...)
    chunk_rows: u32,
    replay_floor: bool,
    window: u32, // the bounded replay's window (Lanes.usable's)
    source_layers: []const u32, // KV-source layer ids, in plane order
    ratios: []const u32, // their compression ratios
    planes: []const u8, // the common part of the state layout (spill_state's descriptor bytes)
    world: u16,
    split: []const u8, // the whole rank split plan (every rank's part), as bytes
    owner_layout: OwnerLayout,
    owner_version: u16, // parts.layout_version (0 when replicated)
    parts: u8, // owner parts (1 when replicated)
    features: []const u8, // the switches, as a canonical "name=value;..." string
    rank: u16,
    rank_split: []const u8, // this rank's part of the split
    rank_layout: Digest, // spill_state.StateLayout.digest() on this rank
};

pub const OwnerLayout = enum(u8) { replicated = 1, owner_sharded = 2 };

/// A file, a folder or literal bytes standing for an input; none when the configuration has no such input.
pub const Source = union(enum) {
    none,
    bytes: []const u8,
    file: struct { dir: std.Io.Dir, path: []const u8 },
    tree: struct { dir: std.Io.Dir, path: []const u8, keep: ?*const fn ([]const u8) bool = null },
};

fn putSource(b: *Builder, f: Field, gpa: std.mem.Allocator, io: std.Io, s: Source) !void {
    switch (s) {
        .none => b.part(f, "none"),
        .bytes => |v| {
            b.part(f, "bytes");
            b.part(f, v);
        },
        .file => |v| {
            b.part(f, "file");
            try b.putFile(f, io, v.dir, v.path);
        },
        .tree => |v| {
            b.part(f, "tree");
            try b.putTree(f, gpa, io, v.dir, v.path, v.keep);
        },
    }
}

/// Builds a rank's fingerprint from its inputs (every field given).
pub fn build(gpa: std.mem.Allocator, io: std.Io, in: Inputs) !Fingerprint {
    var b = Builder.init();
    b.int(.native_format, u32, in.native_format);
    b.part(.wire_version, in.wire);
    b.part(.engine_commit, in.engine_commit);
    b.part(.recipe_patches, in.recipe_patches);
    b.part(.build_flags, in.build_flags);
    try putSource(&b, .kit_aot, gpa, io, in.kit_aot);
    try putSource(&b, .kit_cuda, gpa, io, in.kit_cuda);
    try putSource(&b, .kit_exl3, gpa, io, in.kit_exl3);
    try putSource(&b, .kit_rdma, gpa, io, in.kit_rdma);
    b.part(.model_revision, in.model_revision);
    b.part(.rank_cache, in.rank_cache);
    b.part(.ablation, in.ablation);
    b.part(.dspark, in.dspark);
    try putSource(&b, .tokenizer, gpa, io, in.tokenizer);
    try putSource(&b, .template, gpa, io, in.template);
    b.part(.engram, in.engram);
    try putSource(&b, .token_map, gpa, io, in.token_map);
    try putSource(&b, .rope, gpa, io, in.rope);
    b.part(.vision, in.vision);
    b.int(.gpu, u32, in.gpu_capability);
    b.int(.cuda, i32, in.cuda_driver);
    b.part(.cuda, in.math);
    b.int(.chunk_rows, u32, in.chunk_rows);
    b.int(.replay, u8, @intFromBool(in.replay_floor));
    b.int(.replay, u32, in.window);
    if (in.source_layers.len != in.ratios.len) return error.BadSourceLayers;
    b.ints(.source_layers, u32, in.source_layers);
    b.ints(.source_layers, u32, in.ratios);
    b.part(.planes, in.planes);
    b.int(.tp_split, u16, in.world);
    b.part(.tp_split, in.split);
    b.int(.owner_layout, u8, @intFromEnum(in.owner_layout));
    b.int(.owner_layout, u16, in.owner_version);
    b.int(.owner_layout, u8, in.parts);
    if (in.owner_layout == .replicated and (in.parts != 1 or in.owner_version != 0)) return error.BadOwnerLayout;
    b.part(.features, in.features);
    if (in.rank >= in.world or in.world > max_world) return error.BadRank;
    b.int(.rank_split, u16, in.rank);
    b.part(.rank_split, in.rank_split);
    b.part(.rank_layout, &in.rank_layout);
    return b.finish(null);
}

// -- tests ------------------------------------------------------------------------------------------------------------

fn testInputs() Inputs {
    return .{
        .native_format = 1,
        .wire = "rank-wire v1",
        .engine_commit = "629c524a0ea57512a5342abb498d1b7c25353343",
        .recipe_patches = "0001 0002 0004 0005 0006",
        .build_flags = "-Doptimize=ReleaseFast",
        .kit_aot = .{ .bytes = "aot" },
        .kit_cuda = .{ .bytes = "cubins" },
        .kit_exl3 = .{ .bytes = "exl3" },
        .kit_rdma = .none,
        .model_revision = "rev-1017",
        .rank_cache = "rank0-of-2.safetensors 123",
        .ablation = "off",
        .dspark = "off",
        .tokenizer = .{ .bytes = "tok" },
        .template = .{ .bytes = "tpl" },
        .engram = "engram.json",
        .token_map = .{ .bytes = "map" },
        .rope = .{ .bytes = "rope" },
        .vision = "off",
        .gpu_capability = 121,
        .cuda_driver = 13000,
        .math = "graphs=1",
        .chunk_rows = 2048,
        .replay_floor = false,
        .window = 128,
        .source_layers = &.{ 2, 8, 14, 20 },
        .ratios = &.{ 2, 2, 2, 1 },
        .planes = "planes",
        .world = 2,
        .split = "tp2",
        .owner_layout = .replicated,
        .owner_version = 0,
        .parts = 1,
        .features = "spill=1;shard=0",
        .rank = 0,
        .rank_split = "rank0",
        .rank_layout = @splat(7),
    };
}

/// A copy of `in` with field `f` changed (one input that feeds it).
fn mutate(in: Inputs, f: Field) Inputs {
    var o = in;
    switch (f) {
        .native_format => o.native_format += 1,
        .wire_version => o.wire = "rank-wire v2",
        .engine_commit => o.engine_commit = "629c524a0ea57512a5342abb498d1b7c25353344",
        .recipe_patches => o.recipe_patches = "0001 0002 0004 0005",
        .build_flags => o.build_flags = "-Doptimize=Debug",
        .kit_aot => o.kit_aot = .{ .bytes = "aot2" },
        .kit_cuda => o.kit_cuda = .none,
        .kit_exl3 => o.kit_exl3 = .{ .bytes = "exl3b" },
        .kit_rdma => o.kit_rdma = .{ .bytes = "rdma" },
        .model_revision => o.model_revision = "rev-1018",
        .rank_cache => o.rank_cache = "rank0-of-2.safetensors 124",
        .ablation => o.ablation = "overlay-abc",
        .dspark => o.dspark = "dspark depth 4",
        .tokenizer => o.tokenizer = .{ .bytes = "tok2" },
        .template => o.template = .{ .bytes = "tpl2" },
        .engram => o.engram = "engram.json v2",
        .token_map => o.token_map = .{ .bytes = "map2" },
        .rope => o.rope = .{ .bytes = "rope2" },
        .vision => o.vision = "bias",
        .gpu => o.gpu_capability = 120,
        .cuda => o.cuda_driver = 13010,
        .chunk_rows => o.chunk_rows = 4096,
        .replay => o.replay_floor = true,
        .source_layers => o.ratios = &.{ 2, 2, 1, 1 },
        .planes => o.planes = "planes2",
        .tp_split => o.split = "tp2b",
        .owner_layout => {
            o.owner_layout = .owner_sharded;
            o.owner_version = 1;
            o.parts = 2;
        },
        .features => o.features = "spill=1;shard=1",
        .rank_split => o.rank_split = "rank0b",
        .rank_layout => o.rank_layout = @splat(8),
    }
    return o;
}

test "the fingerprint is stable and every field changes its digest" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const base = try build(gpa, io, testInputs());
    const again = try build(gpa, io, testInputs());
    try std.testing.expectEqualSlices(u8, &base.common(), &again.common());
    try std.testing.expectEqualSlices(u8, &base.rankDigest(), &again.rankDigest());
    const bx = base.exchange();
    for (all) |f| {
        const fp = try build(gpa, io, mutate(testInputs(), f));
        // the field itself, and no other field
        for (all) |g| {
            const same = std.mem.eql(u8, &fp.get(g), &base.get(g));
            try std.testing.expect(same == (g != f));
        }
        const cs = std.mem.eql(u8, &fp.common(), &base.common());
        const rs = std.mem.eql(u8, &fp.rankDigest(), &base.rankDigest());
        try std.testing.expect(cs == (scope(f) == .rank));
        try std.testing.expect(rs == (scope(f) == .common));
        const fx = fp.exchange();
        if (scope(f) == .common) {
            try std.testing.expectEqual(@as(?u16, @intFromEnum(f)), bx.differs(&fx));
            try std.testing.expectEqual(@as(?u16, @intFromEnum(f)), fx.differs(&bx));
        } else {
            try std.testing.expectEqual(@as(?u16, null), bx.differs(&fx));
        }
    }
}

test "a field never given is refused by name, parts are length-delimited" {
    var b = Builder.init();
    for (all) |f| if (f != .engram) b.part(f, "x");
    var missing: Field = .native_format;
    try std.testing.expectError(error.MissingField, b.finish(&missing));
    try std.testing.expectEqual(Field.engram, missing);
    b.part(.engram, "");
    _ = try b.finish(null);

    var x = Builder.init();
    var y = Builder.init();
    x.part(.engram, "ab");
    x.part(.engram, "c");
    y.part(.engram, "a");
    y.part(.engram, "bc");
    for (all) |f| if (f != .engram) {
        x.part(f, "");
        y.part(f, "");
    };
    const fx = try x.finish(null);
    const fy = try y.finish(null);
    try std.testing.expect(!std.mem.eql(u8, &fx.get(.engram), &fy.get(.engram)));
    // u32 3 and u64 3 differ
    var p = Builder.init();
    var q = Builder.init();
    p.int(.chunk_rows, u32, 3);
    q.int(.chunk_rows, u64, 3);
    for (all) |f| if (f != .chunk_rows) {
        p.part(f, "");
        q.part(f, "");
    };
    try std.testing.expect(!std.mem.eql(u8, &(try p.finish(null)).get(.chunk_rows), &(try q.finish(null)).get(.chunk_rows)));
}

test "the digest construction is pinned" {
    // a change to how digests are built must bump `version` (stored points and peers then disagree on purpose)
    var b: Builder = .{}; // without the builtin facts: the same on every host
    for (all) |f| b.int(f, u16, @intFromEnum(f));
    const fp = try b.finish(null);
    var hex: [64]u8 = undefined;
    _ = std.fmt.bufPrint(&hex, "{x}", .{&fp.common()}) catch unreachable;
    try std.testing.expectEqualStrings(pinned_common, &hex);
}

const pinned_common = "93e91fba5b567b84c16b12657de343df9f4811ffede0a529ff3d6aefeecaf05b";

test "files and folders are hashed by content, not by name or place" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "kit/a/sub");
    try tmp.dir.createDirPath(io, "kit2/a/sub");
    try tmp.dir.writeFile(io, .{ .sub_path = "kit/a/x.cubin", .data = "one" });
    try tmp.dir.writeFile(io, .{ .sub_path = "kit/a/sub/y.cubin", .data = "two" });
    try tmp.dir.writeFile(io, .{ .sub_path = "kit2/a/x.cubin", .data = "one" });
    try tmp.dir.writeFile(io, .{ .sub_path = "kit2/a/sub/y.cubin", .data = "two" });
    const t1 = try treeDigest(gpa, io, tmp.dir, "kit", null);
    const t2 = try treeDigest(gpa, io, tmp.dir, "kit2", null);
    try std.testing.expectEqualSlices(u8, &t1, &t2);
    try tmp.dir.writeFile(io, .{ .sub_path = "kit2/a/sub/y.cubin", .data = "twO" });
    try std.testing.expect(!std.mem.eql(u8, &t1, &try treeDigest(gpa, io, tmp.dir, "kit2", null)));
    try tmp.dir.writeFile(io, .{ .sub_path = "kit2/a/sub/y.cubin", .data = "two" });
    try tmp.dir.writeFile(io, .{ .sub_path = "kit2/a/extra.cubin", .data = "" });
    try std.testing.expect(!std.mem.eql(u8, &t1, &try treeDigest(gpa, io, tmp.dir, "kit2", null)));
    const only = struct {
        fn cubinsBelowSub(p: []const u8) bool {
            return std.mem.startsWith(u8, p, "a/sub/");
        }
    }.cubinsBelowSub;
    try std.testing.expectEqualSlices(u8, &try treeDigest(gpa, io, tmp.dir, "kit", only), &try treeDigest(gpa, io, tmp.dir, "kit2", only));
    const f1 = try fileDigest(io, tmp.dir, "kit/a/x.cubin");
    const f2 = try fileDigest(io, tmp.dir, "kit2/a/x.cubin");
    try std.testing.expectEqualSlices(u8, &f1, &f2);
    var expect: Digest = undefined;
    Sha256.hash("one", &expect, .{});
    try std.testing.expectEqualSlices(u8, &expect, &f1);
}

test "the agreement narrows to each rank's store" {
    var a: Agreement = .{ .common = @splat(1), .world = 2 };
    a.ranks[0] = @splat(2);
    a.ranks[1] = @splat(3);
    const s0 = a.storeCommon(0);
    const s1 = a.storeCommon(1);
    try std.testing.expect(!std.mem.eql(u8, &s0, &s1));
    var b = a;
    b.ranks[1] = @splat(4);
    try std.testing.expectEqualSlices(u8, &s0, &b.storeCommon(0)); // rank 0's store does not depend on rank 1's weights
    try std.testing.expect(!std.mem.eql(u8, &a.pair(), &b.pair()));
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("engine_commit", describe(3, &buf));
    try std.testing.expectEqualStrings("field 999", describe(999, &buf));
}
