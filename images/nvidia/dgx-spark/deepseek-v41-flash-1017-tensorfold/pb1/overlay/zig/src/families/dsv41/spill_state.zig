//! Port of wojo's GLM 0110 (PR #124; BadAd84's block-tier design) on its 0088 foundation (PR #78): the checkpoint
//! state a spilled DeepSeek-V4.1 kept point holds, laid out as this native engine keeps it. A kept point is the
//! existing pre-replay checkpoint (Lanes.usable: 0 < at < len, at % 2048 == 0, at <= len - window); its state is
//!
//!   - blocks: every kv-source layer's compressed rows of positions [0, at) of the stream's extent, as copyExtent
//!     copies them (packed FP4 codes and E8M0 scales, then the index keys' codes and scales), cut into shared
//!     2048-token blocks, each block plane after plane in layer order (TF_DS_KV_SHARD's owner_sharded layout: the
//!     codes and scales on their part's owner only, so a rank's block of another part's point is its index rows);
//!   - small state: the slot's own part, byte for byte as Model.snapCopy keeps it (every layer's BF16 window ring of
//!     window + RING_EXTRA rows, then each ratio-2 source's FP32 raw kv and raw score stores of RAW rows).
//!
//! The rest of a resumed prompt's state is rebuilt, not saved (`inventory`): the Engram sequence from the request's
//! ids, the drafter's rings by the resumed replay. Guard rows past an extent (Caches' cap / ratio + 2) are never
//! state. This file is host-only (std and config.zig): the layout, its digest, the extent generations and the
//! packing order; Model.SpillDev (model.zig) gives the packer the device addresses and the stream's copies.
const std = @import("std");
const Config = @import("config.zig").Config;

/// The native spill state format (bumped with any change to the plane order, the encoding or the digest input).
pub const format_version: u32 = 1;
/// Planes a layout holds at most: 4 block planes for each of up to 8 kv sources, a ring for each of up to 64 layers
/// and 2 raw stores for each of up to 8 sources.
pub const max_planes = 4 * 8 + 64 + 2 * 8;
/// Layers a layout describes at most (Caches' arrays).
pub const max_layers = 64;
/// Pool slots the generations track (draft.max_streams; model.zig checks it).
pub const max_slots = 16;

/// Whose rows a rank's planes hold. `replicated`: every rank holds every row. `owner_sharded` (0118, TF_DS_KV_SHARD=1):
/// the pool is cut into parts of `part_rows` positions, part p's compressed codes and scales live on rank p alone (those
/// planes are `owner_only`, in that rank's part-local plane: shard.Shard.planeBase), the index planes, rings and raw
/// stores stay replicated. Each rank then packs only what it holds: a point in its own part as every plane (block_bytes
/// a block), a point in another part as the replicated planes alone (foreign_block_bytes), the replicated planes once
/// (spill-shard round 3; GLM 0118's spill_blocks `_present`). The digest covers the layout and the part size, so the
/// two layouts (and two part sizes) never share a store.
pub const OwnerLayout = enum(u8) { replicated = 0, owner_sharded = 1 };

/// A block plane a rank does not hold for a point (an owner_only plane of another part's point): no staging offset.
pub const absent = std.math.maxInt(usize);

pub const Kind = enum(u8) { comp_codes = 0, comp_scales = 1, idx_codes = 2, idx_scales = 3, ring = 4, raw_kv = 5, raw_score = 6 };

/// A plane's element encoding (the digest's, not a kernel's): FP4 E2M1 two a byte, E8M0 scales, BF16, FP32.
pub const DType = enum(u8) { fp4x2 = 0, e8m0 = 1, bf16 = 2, f32 = 3 };

/// One state plane. Block planes (ratio 1 or 2) hold a row per `ratio` positions of the extent; small planes
/// (ratio 0) hold `rows` rows per slot.
pub const Plane = struct {
    kind: Kind,
    layer: u16,
    ratio: u8 = 0,
    row: u32, // bytes a row
    rows: u32 = 0, // small planes: a slot's rows
    dtype: DType,
    sentinel: u8 = 0, // a fresh pool's byte (Caches.clear: the index scales' 127, E8M0 1.0; zeros elsewhere)
    owner_only: bool = false, // owner_sharded: only the point's owner rank holds (and stores) these rows

    pub fn isBlock(p: Plane) bool {
        return p.ratio != 0;
    }

    /// A block plane's bytes for `tokens` positions (a multiple of the ratio).
    pub fn spanBytes(p: Plane, tokens: usize) usize {
        return tokens / p.ratio * p.row;
    }

    /// A small plane's bytes for one slot.
    pub fn slotBytes(p: Plane) usize {
        return @as(usize, p.rows) * p.row;
    }
};

/// What a target layer carries, as Caches.init reads it from the loaded weights: a kv source (compressor), its
/// ratio, an index source over it, ratio-2 raw stores.
pub const LayerShape = struct { ratio: u8 = 0, kv: bool = false, idx: bool = false, raw: bool = false };

/// The dimensions the planes are sized by (from the loaded model: config.json, prompt.ring_extra, prompt.raw_rows,
/// model.chunk_rows).
pub const Geometry = struct {
    block_tokens: usize,
    window: usize,
    ring_extra: usize,
    raw_rows: usize,
    head_dim: usize,
    index_head_dim: usize,

    pub fn of(c: *const Config, block_tokens: usize, ring_extra: usize, raw_rows: usize) Geometry {
        return .{ .block_tokens = block_tokens, .window = c.window, .ring_extra = ring_extra, .raw_rows = raw_rows, .head_dim = c.head_dim, .index_head_dim = c.index_head_dim };
    }

    /// A window ring's rows (window + RING_EXTRA).
    pub fn ringRows(g: Geometry) usize {
        return g.window + g.ring_extra;
    }
};

/// Every layer's shape as weights.zig loads it from config.json (kv sources compress at their ratio; an index source
/// that is also a kv source keeps index keys; ratio-2 sources keep raw stores).
pub fn shapesOf(c: *const Config, out: []LayerShape) ![]LayerShape {
    if (c.layers > out.len) return error.TooManyLayers;
    for (out[0..c.layers], 0..) |*s, i| {
        const kv = c.kv_sources.has(i);
        s.* = .{ .ratio = if (kv) c.ratios[i] else 0, .kv = kv, .idx = kv and c.index_sources.has(i), .raw = kv and c.ratios[i] > 1 };
    }
    return out[0..c.layers];
}

/// The ordered state descriptor of one rank: block planes first (layer order; codes, scales, index codes, index
/// scales), then the small planes in snapCopy's order (every layer's ring, then each layer's raw kv and raw score).
pub const StateLayout = struct {
    geo: Geometry,
    owner_layout: OwnerLayout = .replicated,
    planes: [max_planes]Plane = undefined,
    n_block: usize = 0,
    n_small: usize = 0,
    offsets: [max_planes]usize = undefined, // a block plane's offset in a block; a small plane's in the small state
    token_bytes: usize = 0, // block bytes a position (890 for the served checkpoint)
    block_bytes: usize = 0, // a block's bytes (block_tokens positions; owner_sharded: a point of the rank's own part)
    small_bytes: usize = 0, // the small state's bytes (Model.snapBytes)
    /// owner_sharded: positions a part (withParts; 0: not yet cut, the Packer refuses). Replicated: 0.
    part_rows: usize = 0,
    /// A block plane's offset in a block of a point another rank owns (`absent` for its owner_only planes).
    foreign_offsets: [max_planes]usize = undefined,
    /// A block's bytes for a point another rank owns (its replicated planes; block_bytes when replicated).
    foreign_block_bytes: usize = 0,
    /// The owner_only bytes a position (the compressed codes and scales a carry sends: 720 for the served checkpoint;
    /// 0 when replicated).
    owner_token_bytes: usize = 0,

    pub fn init(shapes: []const LayerShape, g: Geometry, owner: OwnerLayout) !StateLayout {
        // owner_sharded: the same planes, the compressed codes/scales marked owner_only (0118); packing it needs the
        // part size (withParts) and the packing rank (Packer.rank)
        const own = owner == .owner_sharded;
        if (shapes.len == 0 or shapes.len > max_layers) return error.BadLayout;
        if (g.block_tokens == 0 or g.head_dim == 0 or g.head_dim % 16 != 0 or g.index_head_dim % 32 != 0 or g.raw_rows == 0) return error.BadLayout;
        var l: StateLayout = .{ .geo = g, .owner_layout = owner };
        const hd: u32 = @intCast(g.head_dim);
        const ihd: u32 = @intCast(g.index_head_dim);
        for (shapes, 0..) |s, i| {
            if ((s.idx or s.raw) and !s.kv) return error.BadLayout;
            if (!s.kv) continue;
            if (s.ratio == 0 or s.ratio > 2 or g.block_tokens % s.ratio != 0) return error.BadLayout;
            if (s.raw != (s.ratio > 1)) return error.BadLayout; // Caches.init: raw stores exactly at ratio > 1
            const li: u16 = @intCast(i);
            try l.add(.{ .kind = .comp_codes, .layer = li, .ratio = s.ratio, .row = hd / 2, .dtype = .fp4x2, .owner_only = own });
            try l.add(.{ .kind = .comp_scales, .layer = li, .ratio = s.ratio, .row = hd / 16, .dtype = .e8m0, .owner_only = own });
            if (s.idx) {
                if (ihd == 0) return error.BadLayout;
                try l.add(.{ .kind = .idx_codes, .layer = li, .ratio = s.ratio, .row = ihd / 2, .dtype = .fp4x2 });
                try l.add(.{ .kind = .idx_scales, .layer = li, .ratio = s.ratio, .row = ihd / 32, .dtype = .e8m0, .sentinel = 127 });
            }
        }
        const ring_rows: u32 = @intCast(g.ringRows());
        for (0..shapes.len) |i| try l.add(.{ .kind = .ring, .layer = @intCast(i), .row = hd * 2, .rows = ring_rows, .dtype = .bf16 });
        for (shapes, 0..) |s, i| if (s.raw) {
            try l.add(.{ .kind = .raw_kv, .layer = @intCast(i), .row = hd * 4, .rows = @intCast(g.raw_rows), .dtype = .f32 });
            try l.add(.{ .kind = .raw_score, .layer = @intCast(i), .row = hd * 4, .rows = @intCast(g.raw_rows), .dtype = .f32 });
        };
        var at: usize = 0;
        for (l.blockPlanes(), 0..) |p, i| {
            l.offsets[i] = at;
            at += p.spanBytes(g.block_tokens);
        }
        l.block_bytes = at;
        if (at % g.block_tokens != 0) return error.BadLayout;
        l.token_bytes = at / g.block_tokens;
        at = 0;
        for (l.blockPlanes(), 0..) |p, i| {
            if (p.owner_only) {
                l.foreign_offsets[i] = absent;
                continue;
            }
            l.foreign_offsets[i] = at;
            at += p.spanBytes(g.block_tokens);
        }
        l.foreign_block_bytes = at;
        l.owner_token_bytes = (l.block_bytes - at) / g.block_tokens; // (every span a multiple of the block's rows)
        at = 0;
        for (l.smallPlanes(), l.n_block..) |p, i| {
            l.offsets[i] = at;
            at += p.slotBytes();
        }
        l.small_bytes = at;
        return l;
    }

    /// An owner_sharded layout cut into parts of `part_rows` positions (shard.Shard.part: the pool / world, whole
    /// blocks): what the Packer needs to find a point's owner and its part-local rows.
    pub fn withParts(l: StateLayout, part_rows: usize) !StateLayout {
        if (l.owner_layout != .owner_sharded or part_rows == 0 or part_rows % l.geo.block_tokens != 0) return error.BadLayout;
        var o = l;
        o.part_rows = part_rows;
        return o;
    }

    /// The part holding pool position `pos` (owner_sharded; 0 when replicated).
    pub fn partOf(l: *const StateLayout, pos: usize) u32 {
        if (l.owner_layout == .replicated or l.part_rows == 0) return 0;
        return @intCast(pos / l.part_rows);
    }

    /// A block's bytes on a rank that owns the point (`own`; always when replicated) or does not.
    pub fn blockBytesOf(l: *const StateLayout, own: bool) usize {
        return if (own or l.owner_layout == .replicated) l.block_bytes else l.foreign_block_bytes;
    }

    /// Block plane i's offset in a block on a rank that owns the point or does not (`absent`: that rank has none).
    pub fn blockOffsetOf(l: *const StateLayout, i: usize, own: bool) usize {
        return if (own or l.owner_layout == .replicated) l.offsets[i] else l.foreign_offsets[i];
    }

    /// The layout of a config.json (host checks; the server derives it from the loaded weights, Model.spillLayout).
    pub fn fromConfig(c: *const Config, g: Geometry, owner: OwnerLayout) !StateLayout {
        var buf: [max_layers]LayerShape = undefined;
        return init(try shapesOf(c, &buf), g, owner);
    }

    fn n(l: *const StateLayout) usize {
        return l.n_block + l.n_small;
    }

    fn add(l: *StateLayout, p: Plane) !void {
        const i = l.n();
        if (i == max_planes) return error.BadLayout;
        l.planes[i] = p;
        if (p.isBlock()) {
            if (l.n_small != 0) return error.BadLayout; // block planes first
            l.n_block += 1;
        } else l.n_small += 1;
    }

    pub fn blockPlanes(l: *const StateLayout) []const Plane {
        return l.planes[0..l.n_block];
    }

    pub fn smallPlanes(l: *const StateLayout) []const Plane {
        return l.planes[l.n_block..][0..l.n_small];
    }

    /// A small plane's offset in the small state (i: its index in smallPlanes()).
    pub fn smallOffset(l: *const StateLayout, i: usize) usize {
        return l.offsets[l.n_block + i];
    }

    /// A block plane's offset in a block.
    pub fn blockOffset(l: *const StateLayout, i: usize) usize {
        return l.offsets[i];
    }

    /// The whole blocks of a point at `at`: refused unless `at` is a block end (a partial block would be a mutable
    /// tail; native points have none).
    pub fn blocks(l: *const StateLayout, at: usize) !usize {
        if (at % l.geo.block_tokens != 0) return error.MutableTail;
        return at / l.geo.block_tokens;
    }

    /// A point's bytes on one rank (its blocks, shared or not, and its small state; the owner's under owner_sharded).
    /// error.Overflow for an `at` no pool holds (a parsed manifest's).
    pub fn pointBytes(l: *const StateLayout, at: usize) !usize {
        return l.pointBytesOf(at, true);
    }

    /// pointBytes on a rank that owns the point or does not (owner_sharded).
    pub fn pointBytesOf(l: *const StateLayout, at: usize, own: bool) !usize {
        return std.math.add(usize, try std.math.mul(usize, try l.blocks(at), l.blockBytesOf(own)), l.small_bytes);
    }

    /// The bytes encode() writes at most.
    pub const max_encoded = 16 + 4 + 1 + 6 * 4 + 2 * 4 + 3 * 8 + max_planes * 16 + 8;

    /// The layout's canonical bytes (the digest's input; the compatibility fingerprint takes the digest): magic,
    /// format version, owner layout, geometry, plane counts and sizes, then every plane's kind, dtype, ratio,
    /// sentinel, layer, flags (bit 0: owner_only), row bytes and slot rows (16 bytes a plane), all little-endian;
    /// owner_sharded adds the part size (u64) last, so a replicated layout's bytes are what they always were.
    pub fn encode(l: *const StateLayout, buf: []u8) ![]u8 {
        var w: std.Io.Writer = .fixed(buf);
        w.writeAll("tf-dsv41-state\x00\x00") catch return error.NoSpace;
        put(&w, u32, format_version) catch return error.NoSpace;
        put(&w, u8, @backingInt(l.owner_layout)) catch return error.NoSpace;
        const g = l.geo;
        for ([_]usize{ g.block_tokens, g.window, g.ring_extra, g.raw_rows, g.head_dim, g.index_head_dim }) |v| put(&w, u32, @intCast(v)) catch return error.NoSpace;
        put(&w, u32, @intCast(l.n_block)) catch return error.NoSpace;
        put(&w, u32, @intCast(l.n_small)) catch return error.NoSpace;
        for ([_]usize{ l.token_bytes, l.block_bytes, l.small_bytes }) |v| put(&w, u64, v) catch return error.NoSpace;
        for (l.planes[0..l.n()]) |p| {
            put(&w, u8, @backingInt(p.kind)) catch return error.NoSpace;
            put(&w, u8, @backingInt(p.dtype)) catch return error.NoSpace;
            put(&w, u8, p.ratio) catch return error.NoSpace;
            put(&w, u8, p.sentinel) catch return error.NoSpace;
            put(&w, u16, p.layer) catch return error.NoSpace;
            put(&w, u16, @intFromBool(p.owner_only)) catch return error.NoSpace;
            put(&w, u32, p.row) catch return error.NoSpace;
            put(&w, u32, p.rows) catch return error.NoSpace; // (0 for block planes: their rows follow the point's `at`)
        }
        if (l.owner_layout == .owner_sharded) put(&w, u64, l.part_rows) catch return error.NoSpace;
        return w.buffered();
    }

    fn put(w: *std.Io.Writer, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try w.writeAll(&b);
    }

    /// SHA-256 of encode(): the layout's part of the compatibility fingerprint (with the owner layout in it, replicated
    /// and owner-sharded stores never share a namespace).
    pub fn digest(l: *const StateLayout) [32]u8 {
        var buf: [max_encoded]u8 = undefined;
        const bytes = l.encode(&buf) catch unreachable; // sized for max_planes
        var out: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
        return out;
    }
};

// -- extent and slot generations ---------------------------------------------------------------------------------

/// A pinned range of pool positions [lo, hi) (a point's extent rows, or a restore's reserved destination) and the
/// generation stamp it was pinned at.
pub const ExtentRef = struct { lo: usize, hi: usize, stamp: u64 };

/// A pinned slot (its rings and raw stores) and its generation.
pub const SlotRef = struct { slot: usize, stamp: u64 };

/// A kept Snap's device buffer and its identity (Lanes' Snap.id: assigned from next_snap, never reused, never 0). The
/// buffer's address alone is not an identity: a dropped Snap's address can come back for a new one.
pub const SnapRef = struct { id: u64, buf: u64 };

/// Generations of the pool's positions (a counter for each `bucket` positions) and of its slots, bumped by whatever
/// is about to change their bytes: a placement that stales a kept state, a stream that writes over rows, a restore
/// that reserves a destination, a slot taken by another stream. A pin's stamp is the sum of its buckets' counters
/// (they only grow), so it holds exactly while no bump reached them. Engine thread only; sized once (init), no
/// allocation after.
pub const Generations = struct {
    bucket: usize,
    cap: usize,
    gens: []u64,
    slots: [max_slots]u64 = @splat(0),

    pub fn init(gpa: std.mem.Allocator, cap: usize, bucket: usize) !Generations {
        if (bucket == 0 or cap == 0) return error.BadGenerations;
        const g = try gpa.alloc(u64, (cap + bucket - 1) / bucket);
        @memset(g, 0);
        return .{ .bucket = bucket, .cap = cap, .gens = g };
    }

    pub fn deinit(g: *Generations, gpa: std.mem.Allocator) void {
        gpa.free(g.gens);
        g.* = undefined;
    }

    fn span(g: *const Generations, lo: usize, hi: usize) [2]usize {
        const a = @min(lo / g.bucket, g.gens.len);
        const b = @min(hi / g.bucket + @intFromBool(hi % g.bucket != 0), g.gens.len);
        return .{ a, b };
    }

    /// Positions [lo, hi) are about to change: every pin over them goes stale.
    pub fn bump(g: *Generations, lo: usize, hi: usize) void {
        if (hi <= lo) return;
        const s = g.span(lo, hi);
        for (g.gens[s[0]..s[1]]) |*x| x.* += 1;
    }

    pub fn stamp(g: *const Generations, lo: usize, hi: usize) u64 {
        const s = g.span(lo, hi);
        var sum: u64 = 0;
        for (g.gens[s[0]..s[1]]) |x| sum += x;
        return sum;
    }

    /// [lo, hi) as it stands now (a point's rows while they are its own).
    pub fn pin(g: *const Generations, lo: usize, hi: usize) !ExtentRef {
        if (hi <= lo or hi > g.cap) return error.BadRange;
        return .{ .lo = lo, .hi = hi, .stamp = g.stamp(lo, hi) };
    }

    /// A restore's destination: every earlier pin over it goes stale, then it is pinned for the restore.
    pub fn reserve(g: *Generations, lo: usize, hi: usize) !ExtentRef {
        if (hi <= lo or hi > g.cap) return error.BadRange;
        g.bump(lo, hi);
        return g.pin(lo, hi);
    }

    pub fn current(g: *const Generations, r: ExtentRef) bool {
        return r.hi <= g.cap and g.stamp(r.lo, r.hi) == r.stamp;
    }

    pub fn bumpSlot(g: *Generations, slot: usize) void {
        if (slot < max_slots) g.slots[slot] += 1;
    }

    pub fn pinSlot(g: *const Generations, slot: usize) !SlotRef {
        if (slot >= max_slots) return error.BadSlot;
        return .{ .slot = slot, .stamp = g.slots[slot] };
    }

    pub fn reserveSlot(g: *Generations, slot: usize) !SlotRef {
        if (slot >= max_slots) return error.BadSlot;
        g.bumpSlot(slot);
        return g.pinSlot(slot);
    }

    pub fn currentSlot(g: *const Generations, r: SlotRef) bool {
        return r.slot < max_slots and g.slots[r.slot] == r.stamp;
    }

    /// Whether a ticket's pins still hold (an unpack's destination before the restore is published).
    pub fn valid(g: *const Generations, t: Ticket) bool {
        if (t.extent) |r| if (!g.current(r)) return false;
        if (t.slot) |r| if (!g.currentSlot(r)) return false;
        return true;
    }
};

pub const Op = enum(u8) { pack_blocks, unpack_blocks, pack_small, unpack_small, pack_snap, unpack_snap, clear_extent, clear_slot };

/// What an enqueued copy did (returned once its completion event is recorded). The bytes are final when the event
/// completes; a pack's bytes belong to the point whose pin equals `extent`/`slot` (belongsTo), or to the Snap
/// `snap_id` (belongsToSnap), never to a later occupant of the same base or address; an unpack is published only while
/// Generations.valid(ticket) holds.
pub const Ticket = struct {
    op: Op,
    extent: ?ExtentRef = null,
    slot: ?SlotRef = null,
    snap_id: u64 = 0, // pack_snap/unpack_snap: the Snap's id (0: none)
    first_block: usize = 0,
    blocks: usize = 0,
    bytes: usize = 0,

    /// A completed pack's bytes attach to the point pinned as `r` only (same range, same generation).
    pub fn belongsTo(t: Ticket, r: ExtentRef) bool {
        const e = t.extent orelse return false;
        return e.lo == r.lo and e.hi == r.hi and e.stamp == r.stamp;
    }

    /// A completed pack_snap's bytes attach to the Snap with id `id` only.
    pub fn belongsToSnap(t: Ticket, id: u64) bool {
        return t.op == .pack_snap and id != 0 and t.snap_id == id;
    }
};

// -- packing ------------------------------------------------------------------------------------------------------

/// The pack and unpack order over a device `Dev` (Model.SpillDev on the GPU; a host fake in the tests). Dev gives
///   Event; blockPlane(Plane) u64 (row 0 of the whole plane); slotPlane(Plane, slot) u64; capacity() usize
///   (positions); slots() usize; d2h([]u8, u64); h2d(u64, []const u8); fill(u64, u8, usize); record(Event)
/// with every copy in one stream's order. Each call checks everything (sizes, ranges, overflow, pins) before it
/// enqueues anything, then enqueues its copies into caller-owned staging (pinned host memory for the copies to be
/// asynchronous), records `done` after them and returns: no allocation, no wait. A call that fails after its first
/// enqueue still records `done` (best effort) before returning the error, so the caller always has an event to wait on
/// before it recycles the staging; a failed unpack's destination is half written and must be cleared (clearExtent /
/// clearSlot with the same refs) before anything reads it. Staging holds blocks one after another, each its block
/// planes in order (blockOffset), and the small state in smallPlanes' order (snapCopy's).
pub fn Packer(comptime Dev: type) type {
    return struct {
        const Self = @This();
        dev: Dev,
        layout: *const StateLayout,
        gens: *Generations,
        /// owner_sharded: this rank (it owns part `rank`: its owner_only planes hold that part's rows, from row 0).
        rank: u32 = 0,

        /// Whether this rank holds every plane of the extent at `base` (always when replicated).
        pub fn owns(p: *const Self, base: usize) bool {
            return p.layout.owner_layout == .replicated or p.layout.partOf(base) == p.rank;
        }

        /// owner_sharded: [lo, hi) inside one part (an extent never crosses one: Model.request_cap, parts.zig).
        fn checkPart(p: *const Self, lo: usize, hi: usize) !void {
            const l = p.layout;
            if (l.owner_layout == .replicated) return;
            if (l.part_rows == 0) return error.NoParts;
            if (l.partOf(lo) != l.partOf(hi - 1)) return error.CrossesPart;
        }

        /// The device address of plane `pl`'s row holding position `pos` (a ratio multiple): the pool-wide row, or for
        /// an owner_only plane of a sharded layout the row of its part's local plane (shard.Shard.planeBase's).
        fn rowAddr(p: *const Self, pl: Plane, pos: usize) u64 {
            const l = p.layout;
            const at = if (pl.owner_only and l.owner_layout == .owner_sharded) pos - @as(usize, l.partOf(pos)) * l.part_rows else pos;
            return p.dev.blockPlane(pl) + at / pl.ratio * pl.row;
        }

        /// The block calls' range checks (also port_kernels' one-launch pack, unpack and clear): owner_sharded, the
        /// extent inside one part and `staged` this rank's bytes for it (blockBytesOf: its own part's or another's).
        pub fn checkBlocks(p: *const Self, r: ExtentRef, first_block: usize, count: usize, staged: usize) !void {
            const l = p.layout;
            const bt = l.geo.block_tokens;
            if (r.hi <= r.lo or r.hi > p.dev.capacity()) return error.BadRange;
            try p.checkPart(r.lo, r.hi);
            if (count == 0 or staged != try std.math.mul(usize, count, l.blockBytesOf(p.owns(r.lo)))) return error.BadStaging;
            if (r.lo % bt != 0) return error.BadView; // extents start at chunk ends (placeSoft's sizes); ratio-aligned
            if (try std.math.mul(usize, try std.math.add(usize, first_block, count), bt) > r.hi - r.lo) return error.BadRange;
        }

        /// Blocks [first_block, first_block + count) of the point whose rows are pinned as `src` (src.lo: its extent's
        /// base) into `dst`. error.StaleExtent: the rows were given away since the pin (cancel the unpublished write).
        pub fn packBlocks(p: *Self, src: ExtentRef, first_block: usize, count: usize, dst: []u8, done: Dev.Event) !Ticket {
            try p.checkBlocks(src, first_block, count, dst.len);
            if (!p.gens.current(src)) return error.StaleExtent;
            errdefer p.dev.record(done) catch {};
            try p.blockCopies(src.lo, first_block, count, dst, null);
            try p.dev.record(done);
            return .{ .op = .pack_blocks, .extent = src, .first_block = first_block, .blocks = count, .bytes = dst.len };
        }

        /// The staged blocks back into the reserved destination `dst` (Generations.reserve). The restore publishes
        /// only while Generations.valid(ticket) holds after `done`.
        pub fn unpackBlocks(p: *Self, dst: ExtentRef, first_block: usize, count: usize, src: []const u8, done: Dev.Event) !Ticket {
            try p.checkBlocks(dst, first_block, count, src.len);
            if (!p.gens.current(dst)) return error.StaleExtent;
            errdefer p.dev.record(done) catch {};
            try p.blockCopies(dst.lo, first_block, count, null, src);
            try p.dev.record(done);
            return .{ .op = .unpack_blocks, .extent = dst, .first_block = first_block, .blocks = count, .bytes = src.len };
        }

        fn blockCopies(p: *Self, base: usize, first_block: usize, count: usize, out: ?[]u8, in: ?[]const u8) !void {
            const l = p.layout;
            const bt = l.geo.block_tokens;
            const own = p.owns(base);
            const bb = l.blockBytesOf(own);
            for (0..count) |k| {
                const pos = base + (first_block + k) * bt; // < r.hi (checkBlocks)
                for (l.blockPlanes(), 0..) |pl, i| {
                    const in_block = l.blockOffsetOf(i, own);
                    if (in_block == absent) continue; // (another rank's owner_only rows: not here)
                    const dev = p.rowAddr(pl, pos);
                    const len = pl.spanBytes(bt);
                    const off = k * bb + in_block; // < staged (checkBlocks)
                    if (out) |o| try p.dev.d2h(o[off..][0..len], dev) else try p.dev.h2d(dev, in.?[off..][0..len]);
                }
            }
        }

        fn checkSmall(p: *const Self, slot: usize, staged: usize) !void {
            if (staged != p.layout.small_bytes) return error.BadStaging;
            if (slot >= p.dev.slots() or slot >= max_slots) return error.BadSlot;
        }

        /// The live slot's small state (at a keep, before the slot moves on) into `dst`.
        pub fn packSmall(p: *Self, slot: SlotRef, dst: []u8, done: Dev.Event) !Ticket {
            try p.checkSmall(slot.slot, dst.len);
            if (!p.gens.currentSlot(slot)) return error.StaleSlot;
            errdefer p.dev.record(done) catch {};
            try p.smallCopies(slot.slot, dst, null);
            try p.dev.record(done);
            return .{ .op = .pack_small, .slot = slot, .bytes = dst.len };
        }

        /// A staged small state into the reserved slot (Generations.reserveSlot).
        pub fn unpackSmall(p: *Self, slot: SlotRef, src: []const u8, done: Dev.Event) !Ticket {
            try p.checkSmall(slot.slot, src.len);
            if (!p.gens.currentSlot(slot)) return error.StaleSlot;
            errdefer p.dev.record(done) catch {};
            try p.smallCopies(slot.slot, null, src);
            try p.dev.record(done);
            return .{ .op = .unpack_small, .slot = slot, .bytes = src.len };
        }

        fn smallCopies(p: *Self, slot: usize, out: ?[]u8, in: ?[]const u8) !void {
            const l = p.layout;
            for (l.smallPlanes(), 0..) |pl, i| {
                const dev = p.dev.slotPlane(pl, slot);
                const off = l.smallOffset(i);
                const len = pl.slotBytes();
                if (out) |o| try p.dev.d2h(o[off..][0..len], dev) else try p.dev.h2d(dev, in.?[off..][0..len]);
            }
        }

        /// A kept Snap's device buffer (snapCopy's bytes, the small state's order) into `dst`: one copy, which may run
        /// after the slot has moved on. The Packer cannot see Lanes.snaps, so the caller holds the identity rule: enqueue
        /// only while the Snap with `snap.id` is live and still owns `snap.buf`, keep it alive (snapDropFn waits the
        /// pending pack's event before freeing the buffer) until `done`, and attach the bytes only to that id
        /// (Ticket.belongsToSnap).
        pub fn packSnap(p: *Self, snap: SnapRef, dst: []u8, done: Dev.Event) !Ticket {
            if (dst.len != p.layout.small_bytes) return error.BadStaging;
            if (snap.id == 0 or snap.buf == 0) return error.BadSnap;
            errdefer p.dev.record(done) catch {};
            try p.dev.d2h(dst, snap.buf);
            try p.dev.record(done);
            return .{ .op = .pack_snap, .snap_id = snap.id, .bytes = dst.len };
        }

        /// A staged small state into a fresh Snap's device buffer (a restored point kept in memory again).
        pub fn unpackSnap(p: *Self, src: []const u8, snap: SnapRef, done: Dev.Event) !Ticket {
            if (src.len != p.layout.small_bytes) return error.BadStaging;
            if (snap.id == 0 or snap.buf == 0) return error.BadSnap;
            errdefer p.dev.record(done) catch {};
            try p.dev.h2d(snap.buf, src);
            try p.dev.record(done);
            return .{ .op = .unpack_snap, .snap_id = snap.id, .bytes = src.len };
        }

        /// A failed restore's reserved destination `dst` (Generations.reserve's ref) back to a fresh pool's bytes
        /// (Caches.clear: zeros, the index scales' sentinel 127), so no later kernel reads a half-restored row; bumps
        /// their generation first. error.StaleExtent, nothing enqueued, when the reservation was voided (a placement or
        /// another stream took the rows: they are its bytes now).
        pub fn clearExtent(p: *Self, dst: ExtentRef, done: Dev.Event) !Ticket {
            const l = p.layout;
            if (dst.hi <= dst.lo or dst.hi > p.dev.capacity()) return error.BadRange;
            try p.checkPart(dst.lo, dst.hi);
            for (l.blockPlanes()) |pl| if (dst.lo % pl.ratio != 0 or dst.hi % pl.ratio != 0) return error.BadView;
            if (!p.gens.current(dst)) return error.StaleExtent;
            p.gens.bump(dst.lo, dst.hi);
            errdefer p.dev.record(done) catch {};
            // (owner_sharded: another rank's owner_only rows are not here; this rank's local rows at the same offsets
            // are another extent of its own part, never touched)
            const own = p.owns(dst.lo);
            for (l.blockPlanes(), 0..) |pl, i| if (l.blockOffsetOf(i, own) != absent)
                try p.dev.fill(p.rowAddr(pl, dst.lo), pl.sentinel, pl.spanBytes(dst.hi - dst.lo));
            try p.dev.record(done);
            return .{ .op = .clear_extent, .extent = try p.gens.pin(dst.lo, dst.hi), .bytes = 0 };
        }

        /// A failed restore's reserved slot `s` (Generations.reserveSlot's ref) back to a fresh pool's bytes (rings and
        /// raw stores zero); bumps its generation. error.StaleSlot, nothing enqueued, when another stream took the slot.
        pub fn clearSlot(p: *Self, s: SlotRef, done: Dev.Event) !Ticket {
            if (s.slot >= p.dev.slots() or s.slot >= max_slots) return error.BadSlot;
            if (!p.gens.currentSlot(s)) return error.StaleSlot;
            p.gens.bumpSlot(s.slot);
            errdefer p.dev.record(done) catch {};
            for (p.layout.smallPlanes()) |pl| try p.dev.fill(p.dev.slotPlane(pl, s.slot), pl.sentinel, pl.slotBytes());
            try p.dev.record(done);
            return .{ .op = .clear_slot, .slot = try p.gens.pinSlot(s.slot), .bytes = 0 };
        }
    };
}

// -- state rebuilt rather than saved ------------------------------------------------------------------------------

pub const Source = enum { saved, rebuilt_from_ids, rebuilt_by_replay, immutable_input, not_persisted };

pub const Item = struct { what: []const u8, source: Source, rule: []const u8 };

/// Every piece of a resumed prompt's state and where it comes from on a restore.
pub const inventory = [_]Item{
    .{ .what = "compressed rows (comp codes, comp scales) of each kv source", .source = .saved, .rule = "block planes, rows [0, at) of the extent at the ratio (copyExtent's)" },
    .{ .what = "index rows (idx codes, idx scales) of each index kv source", .source = .saved, .rule = "block planes; unwritten scales are 127 (clearExtent restores that)" },
    .{ .what = "window rings, every target layer", .source = .saved, .rule = "small state: window + RING_EXTRA rows at position % rows (snapCopy's)" },
    .{ .what = "ratio-2 raw kv and raw score stores", .source = .saved, .rule = "small state: RAW rows at position % RAW (snapCopy's)" },
    .{ .what = "Engram n-gram sequence (Lanes.seqs[slot])", .source = .rebuilt_from_ids, .rule = "from the validated request's raw ids on every rank, image positions -1 (checkEngramSeq)" },
    .{ .what = "Engram read-ahead rows and AIO state", .source = .not_persisted, .rule = "joined and dropped at the resumed pass's fillBegin (Model.resetEngramAhead)" },
    .{ .what = "Engram tables, token map, multipliers, RoPE tables", .source = .immutable_input, .rule = "identified by the compatibility fingerprint, never copied" },
    .{ .what = "DSpark stage rings and absorbed position", .source = .rebuilt_by_replay, .rule = "the resumed bounded replay absorbs [len - window, len) (checkReplayRebuilds, Model.spillCheckResumed)" },
    .{ .what = "held drafts, confidences, pending token, sampler", .source = .not_persisted, .rule = "a pre-replay point has no decode state" },
    .{ .what = "picture span rows", .source = .not_persisted, .rule = "a picture before `at` is in the saved rows; one straddling or after it runs the tower again (prefillFn)" },
};

/// Lanes.usable's rule without its liveness: a point at `at` resumes a prompt of `len` tokens.
pub fn eligible(at: usize, len: usize, window: usize, block_tokens: usize) bool {
    return at > 0 and at < len and at % block_tokens == 0 and at <= len -| window;
}

/// The drafter positions a resume from `at` absorbs as Model.fillChunk runs the rest of a `len`-token prompt in
/// `chunk`-token chunks (each chunk cut to its rows at or past replay = len - window, the last `ring` of them
/// absorbed when any remain): the newest contiguous run [lo, hi), null when no chunk reaches the head.
pub fn replayAbsorbed(at: usize, len: usize, window: usize, chunk: usize, ring: usize) ?[2]usize {
    const replay = len -| window;
    var start = at;
    var got: ?[2]usize = null;
    while (start < len) {
        const n = @min(chunk, len - start);
        const first = @max(start, replay);
        if (first < start + n) {
            const keep = @min(start + n - first, ring);
            const lo = start + n - keep;
            got = if (got) |g| (if (lo <= g[1]) .{ @min(g[0], lo), start + n } else .{ lo, start + n }) else .{ lo, start + n };
        }
        start += n;
    }
    return got;
}

/// A restore from `at` rebuilds the drafter's read set (its `read` newest positions below len, the window) by the
/// resumed replay alone: refused for an ineligible point or one whose replay would leave a gap.
pub fn checkReplayRebuilds(at: usize, len: usize, window: usize, chunk: usize, ring: usize) !void {
    if (!eligible(at, len, window, chunk)) return error.NotEligible;
    const r = replayAbsorbed(at, len, window, chunk, ring) orelse return error.ReplayGap;
    if (r[1] != len or r[0] > len - window) return error.ReplayGap;
}

/// The Engram sequence a restored slot runs with is the request's own: its raw ids, image span positions negative
/// (no n-gram reaches into or across a picture). `images`: [at, tokens] of each span.
pub fn checkEngramSeq(seq: []const i32, ids: []const u32, images: []const [2]usize) !void {
    if (seq.len != ids.len) return error.EngramSeqMismatch;
    for (seq, ids, 0..) |q, id, i| {
        const in_image = for (images) |sp| {
            if (i >= sp[0] and i < sp[0] + sp[1]) break true;
        } else false;
        if (in_image) {
            if (q >= 0) return error.EngramSeqMismatch;
        } else if (q < 0 or @as(u32, @intCast(q)) != id) return error.EngramSeqMismatch;
    }
}

// -- tests ---------------------------------------------------------------------------------------------------------

const config = @import("config.zig");

/// The fixture checkpoint's layout (config.zig's test_config; RING_EXTRA 16, RAW 64, 2048-token blocks).
fn fixtureLayout(owner: OwnerLayout) !StateLayout {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, config.test_config, &why);
    return StateLayout.fromConfig(&c, Geometry.of(&c, 2048, 16, 64), owner);
}

test "the fixture checkpoint's state layout: 288/68-byte rows, 890 B a token, 1,822,720 B a block, 6,684,672 B small" {
    const l = try fixtureLayout(.replicated);
    // four kv sources (layers 2, 8, 14 at ratio 2; 20 at ratio 1), each with index keys
    try std.testing.expectEqual(@as(usize, 16), l.n_block);
    var comp_row: [64]usize = @splat(0);
    var idx_row: [64]usize = @splat(0);
    for (l.blockPlanes()) |p| switch (p.kind) {
        .comp_codes, .comp_scales => comp_row[p.layer] += p.row,
        .idx_codes, .idx_scales => idx_row[p.layer] += p.row,
        else => return error.TestUnexpectedResult,
    };
    for ([_]u16{ 2, 8, 14, 20 }, [_]u8{ 2, 2, 2, 1 }) |li, r| {
        try std.testing.expectEqual(@as(usize, 288), comp_row[li]);
        try std.testing.expectEqual(@as(usize, 68), idx_row[li]);
        for (l.blockPlanes()) |p| if (p.layer == li) try std.testing.expectEqual(r, p.ratio);
    }
    try std.testing.expectEqual(@as(usize, 890), l.token_bytes);
    try std.testing.expectEqual(@as(usize, 1_822_720), l.block_bytes);
    // 40 rings of 144 x 512 x 2, then two 64 x 512 x 4 stores for each of three ratio-2 sources
    try std.testing.expectEqual(@as(usize, 46), l.n_small);
    var rings: usize = 0;
    var raws: usize = 0;
    for (l.smallPlanes()) |p| switch (p.kind) {
        .ring => {
            rings += 1;
            try std.testing.expectEqual(@as(usize, 144 * 512 * 2), p.slotBytes());
        },
        .raw_kv, .raw_score => {
            raws += 1;
            try std.testing.expectEqual(@as(usize, 64 * 512 * 4), p.slotBytes());
        },
        else => return error.TestUnexpectedResult,
    };
    try std.testing.expectEqual(@as(usize, 40), rings);
    try std.testing.expectEqual(@as(usize, 6), raws);
    try std.testing.expectEqual(@as(usize, 6_684_672), l.small_bytes);
    // snapCopy's order: every ring first, then each source's kv and score
    try std.testing.expectEqual(Kind.ring, l.smallPlanes()[39].kind);
    try std.testing.expectEqual(Kind.raw_kv, l.smallPlanes()[40].kind);
    try std.testing.expectEqual(Kind.raw_score, l.smallPlanes()[41].kind);
    try std.testing.expectEqual(@as(u16, 2), l.smallPlanes()[40].layer);
    // index scales' sentinel, zeros elsewhere
    for (l.blockPlanes()) |p| try std.testing.expectEqual(@as(u8, if (p.kind == .idx_scales) 127 else 0), p.sentinel);
    // a full 1M plane range is 890 MiB a rank; the largest eligible point at the current window
    try std.testing.expectEqual(@as(usize, 890 << 20), 1_048_576 * l.token_bytes);
    try std.testing.expectEqual(@as(usize, 511 * 1_822_720 + 6_684_672), try l.pointBytes(1_046_528));
    try std.testing.expectError(error.MutableTail, l.blocks(2049));
}

test "the layout digest is stable, covers the geometry and the owner layout" {
    const a = try fixtureLayout(.replicated);
    const b = try fixtureLayout(.replicated);
    try std.testing.expectEqualSlices(u8, &a.digest(), &b.digest());
    var buf: [StateLayout.max_encoded]u8 = undefined;
    const enc = try a.encode(&buf);
    try std.testing.expect(std.mem.startsWith(u8, enc, "tf-dsv41-state"));
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, config.test_config, &why);
    // another RING_EXTRA, RAW or block size is another layout
    for ([_]Geometry{ Geometry.of(&c, 2048, 8, 64), Geometry.of(&c, 2048, 16, 32), Geometry.of(&c, 4096, 16, 64) }) |g| {
        const o = try StateLayout.fromConfig(&c, g, .replicated);
        try std.testing.expect(!std.mem.eql(u8, &a.digest(), &o.digest()));
    }
    // another plane set (a source without index keys) is another layout
    var shapes: [max_layers]LayerShape = undefined;
    const s = try shapesOf(&c, &shapes);
    s[20].idx = false;
    const o = try StateLayout.init(s, Geometry.of(&c, 2048, 16, 64), .replicated);
    try std.testing.expectEqual(@as(usize, 822), o.token_bytes);
    try std.testing.expect(!std.mem.eql(u8, &a.digest(), &o.digest()));
    // an owner-sharded store never shares a replicated one's namespace: same planes and sizes, the compressed
    // codes/scales owner_only, another digest
    const sh = try fixtureLayout(.owner_sharded);
    try std.testing.expectEqual(a.block_bytes, sh.block_bytes);
    try std.testing.expectEqual(a.small_bytes, sh.small_bytes);
    for (sh.planes[0 .. sh.n_block + sh.n_small], a.planes[0 .. a.n_block + a.n_small]) |q, r| {
        try std.testing.expectEqual(q.kind == .comp_codes or q.kind == .comp_scales, q.owner_only);
        try std.testing.expect(!r.owner_only);
    }
    try std.testing.expect(!std.mem.eql(u8, &a.digest(), &sh.digest()));
    // a malformed shape is refused
    s[20].idx = true;
    s[20].raw = true; // raw stores at ratio 1
    try std.testing.expectError(error.BadLayout, StateLayout.init(s, Geometry.of(&c, 2048, 16, 64), .replicated));
}

/// Shared test counters (FakeDev is passed by value).
const Counts = struct { records: usize = 0, copies: usize = 0, fail_at: ?usize = null };

/// A host stand-in for the device: one byte array, the planes carved from the config with Caches.init's and
/// Model.open's own formulas (not from the layout under test): per kv source (cap / ratio + 2) rows of head_dim / 2,
/// head_dim / 16, index_head_dim / 2, index_head_dim / 32 bytes, a ring of slots x (window + 16) x head_dim x 2 per
/// layer, and slots x 64 x head_dim x 4 raw kv/score per ratio-2 source. Copies immediate; `fail_at` fails that copy.
const FakeDev = struct {
    mem: []u8,
    cap: usize,
    nslots: usize,
    comp_codes: [max_layers]u64 = @splat(0),
    comp_scales: [max_layers]u64 = @splat(0),
    idx_codes: [max_layers]u64 = @splat(0),
    idx_scales: [max_layers]u64 = @splat(0),
    rings: [max_layers]u64 = @splat(0),
    raw_kv: [max_layers]u64 = @splat(0),
    raw_score: [max_layers]u64 = @splat(0),
    ring_stride: usize,
    raw_stride: usize,
    n: *Counts,

    pub const Event = u32;
    const origin: u64 = 4096; // device address of mem[0] (0 stays "no plane")

    fn make(gpa: std.mem.Allocator, c: *const config.Config, cap: usize, nslots: usize, n: *Counts) !FakeDev {
        return makeParts(gpa, c, cap, nslots, 0, n);
    }

    /// TF_DS_KV_SHARD's planes (prompt.Caches.initShard): each kv source's codes and scales only `part / ratio + 2`
    /// rows (this rank's part), the index keys over the whole pool. `part` 0: replicated (make).
    fn makeParts(gpa: std.mem.Allocator, c: *const config.Config, cap: usize, nslots: usize, part: usize, n: *Counts) !FakeDev {
        const hd = c.head_dim;
        const ihd = c.index_head_dim;
        var d: FakeDev = .{ .mem = &.{}, .cap = cap, .nslots = nslots, .n = n, .ring_stride = (c.window + 16) * hd * 2, .raw_stride = 64 * hd * 4 };
        var at: usize = origin;
        const take = struct {
            fn f(x: *usize, len: usize) u64 {
                defer x.* += len;
                return x.*;
            }
        }.f;
        for (0..c.layers) |i| {
            if (!c.kv_sources.has(i)) continue;
            const rows = cap / c.ratios[i] + 2;
            const crows = if (part > 0) part / c.ratios[i] + 2 else rows;
            d.comp_codes[i] = take(&at, crows * hd / 2);
            d.comp_scales[i] = take(&at, crows * hd / 16);
            if (c.index_sources.has(i)) {
                d.idx_codes[i] = take(&at, rows * ihd / 2);
                d.idx_scales[i] = take(&at, rows * ihd / 32);
            }
            if (c.ratios[i] > 1) {
                d.raw_kv[i] = take(&at, nslots * d.raw_stride);
                d.raw_score[i] = take(&at, nslots * d.raw_stride);
            }
        }
        for (0..c.layers) |i| d.rings[i] = take(&at, nslots * d.ring_stride);
        d.mem = try gpa.alloc(u8, at - origin);
        for (d.mem, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 7);
        return d;
    }

    fn bytes(d: FakeDev, a: u64, len: usize) []u8 {
        return d.mem[a - origin ..][0..len];
    }

    fn copy(d: FakeDev) !void {
        defer d.n.copies += 1;
        if (d.n.fail_at) |f| if (d.n.copies == f) return error.FakeCopyFailed;
    }

    pub fn blockPlane(d: FakeDev, p: Plane) u64 {
        return switch (p.kind) {
            .comp_codes => d.comp_codes[p.layer],
            .comp_scales => d.comp_scales[p.layer],
            .idx_codes => d.idx_codes[p.layer],
            .idx_scales => d.idx_scales[p.layer],
            else => 0,
        };
    }
    pub fn slotPlane(d: FakeDev, p: Plane, slot: usize) u64 {
        return switch (p.kind) {
            .ring => d.rings[p.layer] + slot * d.ring_stride,
            .raw_kv => d.raw_kv[p.layer] + slot * d.raw_stride,
            .raw_score => d.raw_score[p.layer] + slot * d.raw_stride,
            else => 0,
        };
    }
    pub fn capacity(d: FakeDev) usize {
        return d.cap;
    }
    pub fn slots(d: FakeDev) usize {
        return d.nslots;
    }
    pub fn d2h(d: FakeDev, dst: []u8, src: u64) !void {
        try d.copy();
        @memcpy(dst, d.bytes(src, dst.len));
    }
    pub fn h2d(d: FakeDev, dst: u64, src: []const u8) !void {
        try d.copy();
        @memcpy(d.bytes(dst, src.len), src);
    }
    pub fn fill(d: FakeDev, dst: u64, v: u8, len: usize) !void {
        try d.copy();
        @memset(d.bytes(dst, len), v);
    }
    pub fn record(d: FakeDev, _: Event) !void {
        d.n.records += 1;
    }

    /// Host replica of Model.copyExtent's plane loop: every kv layer's (plane, row bytes, ratio), codes, scales, then
    /// the index keys' codes and scales.
    fn extentPlanes(d: FakeDev, c: *const config.Config, out: *[4 * max_layers][3]usize) [][3]usize {
        var k: usize = 0;
        for (0..c.layers) |i| {
            if (d.comp_codes[i] == 0) continue;
            const r: usize = c.ratios[i];
            for ([_][2]usize{ .{ d.comp_codes[i], c.head_dim / 2 }, .{ d.comp_scales[i], c.head_dim / 16 }, .{ d.idx_codes[i], c.index_head_dim / 2 }, .{ d.idx_scales[i], c.index_head_dim / 32 } }) |pr| {
                if (pr[0] == 0) continue;
                out[k] = .{ pr[0], pr[1], r };
                k += 1;
            }
        }
        return out[0..k];
    }

    /// The staged blocks [first, first + count) of the extent at `base` as the format defines them, built
    /// independently: block after block, each copyExtent's planes in order, block_tokens / ratio rows each.
    fn expectBlocks(d: FakeDev, c: *const config.Config, out: *std.ArrayList(u8), gpa: std.mem.Allocator, base: usize, first: usize, count: usize, bt: usize) !void {
        var buf: [4 * max_layers][3]usize = undefined;
        for (first..first + count) |k| for (d.extentPlanes(c, &buf)) |pl| {
            const lo = (base + k * bt) / pl[2];
            try out.appendSlice(gpa, d.bytes(pl[0] + lo * pl[1], bt / pl[2] * pl[1]));
        };
    }

    /// expectBlocks for an owner-sharded rank: the codes and scales of a point in its own part (`own`) from the part's
    /// local rows ((base - part_start) / ratio, prompt.Caches.view's planeBase), none of another part's point; the
    /// index keys from the pool rows either way. Built from the config, not from the layout under test.
    fn expectBlocksSharded(d: FakeDev, c: *const config.Config, out: *std.ArrayList(u8), gpa: std.mem.Allocator, base: usize, first: usize, count: usize, bt: usize, part: usize, own: bool) !void {
        const start = base / part * part;
        for (first..first + count) |k| for (0..c.layers) |i| {
            if (d.comp_codes[i] == 0) continue;
            const r: usize = c.ratios[i];
            const pos = base + k * bt;
            if (own) {
                try out.appendSlice(gpa, d.bytes(d.comp_codes[i] + (pos - start) / r * (c.head_dim / 2), bt / r * (c.head_dim / 2)));
                try out.appendSlice(gpa, d.bytes(d.comp_scales[i] + (pos - start) / r * (c.head_dim / 16), bt / r * (c.head_dim / 16)));
            }
            if (d.idx_codes[i] != 0) {
                try out.appendSlice(gpa, d.bytes(d.idx_codes[i] + pos / r * (c.index_head_dim / 2), bt / r * (c.index_head_dim / 2)));
                try out.appendSlice(gpa, d.bytes(d.idx_scales[i] + pos / r * (c.index_head_dim / 32), bt / r * (c.index_head_dim / 32)));
            }
        };
    }

    /// Host replica of Model.snapCopy's two loops (save): every layer's ring, then each layer's raw kv and raw score.
    fn expectSnap(d: FakeDev, c: *const config.Config, out: *std.ArrayList(u8), gpa: std.mem.Allocator, slot: usize) !void {
        for (d.rings[0..c.layers]) |r| try out.appendSlice(gpa, d.bytes(r + slot * d.ring_stride, d.ring_stride));
        for (d.raw_kv[0..c.layers], d.raw_score[0..c.layers]) |k, sc| for ([_]u64{ k, sc }) |live| if (live != 0) {
            try out.appendSlice(gpa, d.bytes(live + slot * d.raw_stride, d.raw_stride));
        };
    }
};

fn fixtureConfig() !config.Config {
    var why: config.Why = .{};
    return config.parse(std.testing.allocator, config.test_config, &why);
}

test "packed bytes are copyExtent's rows and snapCopy's bytes, built independently; a round trip moves them, guard rows untouched" {
    const gpa = std.testing.allocator;
    const c = try fixtureConfig();
    const l = try fixtureLayout(.replicated);
    const bt = l.geo.block_tokens;
    const cap = 4 * bt;
    var n: Counts = .{};
    const dev = try FakeDev.make(gpa, &c, cap, 2, &n);
    defer gpa.free(dev.mem);
    // the fake's planes end where the layout's sizes say a pool of this cap ends (guard rows included)
    var want_mem: usize = 0;
    for (l.blockPlanes()) |p| want_mem += (cap / p.ratio + 2) * p.row;
    for (l.smallPlanes()) |p| want_mem += 2 * p.slotBytes();
    try std.testing.expectEqual(want_mem, dev.mem.len);
    const before = try gpa.dupe(u8, dev.mem);
    defer gpa.free(before);
    var g = try Generations.init(gpa, cap, bt);
    defer g.deinit(gpa);
    var pk: Packer(FakeDev) = .{ .dev = dev, .layout = &l, .gens = &g };

    // a point of two blocks at base 0, slot 0; and its second block alone from an extent at base 2048
    const src = try g.pin(0, 2 * bt);
    const staged = try gpa.alloc(u8, 2 * l.block_bytes);
    defer gpa.free(staged);
    const t0 = try pk.packBlocks(src, 0, 2, staged, 0);
    try std.testing.expect(t0.belongsTo(src));
    var want: std.ArrayList(u8) = .empty;
    defer want.deinit(gpa);
    try dev.expectBlocks(&c, &want, gpa, 0, 0, 2, bt);
    try std.testing.expectEqualSlices(u8, want.items, staged);
    want.clearRetainingCapacity();
    const one = try gpa.alloc(u8, l.block_bytes);
    defer gpa.free(one);
    _ = try pk.packBlocks(try g.pin(bt, 3 * bt), 1, 1, one, 0);
    try dev.expectBlocks(&c, &want, gpa, bt, 1, 1, bt);
    try std.testing.expectEqualSlices(u8, want.items, one);
    want.clearRetainingCapacity();

    const small = try gpa.alloc(u8, l.small_bytes);
    defer gpa.free(small);
    _ = try pk.packSmall(try g.pinSlot(0), small, 0);
    try dev.expectSnap(&c, &want, gpa, 0);
    try std.testing.expectEqualSlices(u8, want.items, small);
    // a Snap buffer holds snapCopy's bytes: it packs to the same staging in one copy
    // (a second fake whose memory is the Snap's device buffer, at the fake's origin)
    const snapdev = try gpa.alloc(u8, l.small_bytes);
    defer gpa.free(snapdev);
    var snapped: FakeDev = dev;
    snapped.mem = snapdev;
    @memcpy(snapdev, want.items);
    var spk: Packer(FakeDev) = .{ .dev = snapped, .layout = &l, .gens = &g };
    const fromsnap = try gpa.alloc(u8, l.small_bytes);
    defer gpa.free(fromsnap);
    const copies = n.copies;
    const ts = try spk.packSnap(.{ .id = 7, .buf = FakeDev.origin }, fromsnap, 0);
    try std.testing.expectEqual(copies + 1, n.copies);
    try std.testing.expect(ts.belongsToSnap(7) and !ts.belongsToSnap(8));
    try std.testing.expectEqualSlices(u8, small, fromsnap);
    try std.testing.expectError(error.BadSnap, spk.packSnap(.{ .id = 0, .buf = FakeDev.origin }, fromsnap, 0));

    // restored at base 2048 (a moved resume), slot 1
    const recs = n.records;
    const dst = try g.reserve(2 * bt, 4 * bt);
    const t1 = try pk.unpackBlocks(dst, 0, 2, staged, 0);
    _ = try pk.unpackSmall(try g.reserveSlot(1), small, 0);
    try std.testing.expect(g.valid(t1));
    try std.testing.expectEqual(recs + 2, n.records);
    var buf: [4 * max_layers][3]usize = undefined;
    for (dev.extentPlanes(&c, &buf)) |pl| {
        const rows = 2 * bt / pl[2] * pl[1];
        try std.testing.expectEqualSlices(u8, dev.bytes(pl[0], rows), dev.bytes(pl[0] + rows, rows));
        // guard rows past the plane's cap / ratio stay as they were
        const guard = cap / pl[2] * pl[1];
        try std.testing.expectEqualSlices(u8, before[pl[0] - FakeDev.origin + guard ..][0 .. 2 * pl[1]], dev.bytes(pl[0] + guard, 2 * pl[1]));
    }
    var again: std.ArrayList(u8) = .empty;
    defer again.deinit(gpa);
    try dev.expectSnap(&c, &again, gpa, 1);
    try std.testing.expectEqualSlices(u8, small, again.items);

    // staging sizes and ranges are checked before any copy
    const c0 = n.copies;
    try std.testing.expectError(error.BadStaging, pk.packBlocks(src, 0, 1, staged, 0));
    try std.testing.expectError(error.BadRange, pk.packBlocks(src, 1, 2, staged, 0));
    try std.testing.expectError(error.BadStaging, pk.packSmall(try g.pinSlot(0), staged[0..10], 0));
    try std.testing.expectError(error.BadSlot, pk.packSmall(try g.pinSlot(2), small, 0));
    try std.testing.expectEqual(c0, n.copies);
}

test "a manifest's sizes cannot overflow the checks" {
    const gpa = std.testing.allocator;
    const c = try fixtureConfig();
    const l = try fixtureLayout(.replicated);
    const bt = l.geo.block_tokens;
    var n: Counts = .{};
    const dev = try FakeDev.make(gpa, &c, 4 * bt, 2, &n);
    defer gpa.free(dev.mem);
    var g = try Generations.init(gpa, 4 * bt, bt);
    defer g.deinit(gpa);
    var pk: Packer(FakeDev) = .{ .dev = dev, .layout = &l, .gens = &g };
    const staged = try gpa.alloc(u8, 2 * l.block_bytes);
    defer gpa.free(staged);
    const src = try g.pin(0, 2 * bt);
    const max = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, pk.packBlocks(src, max - 1, 2, staged, 0));
    try std.testing.expectError(error.Overflow, pk.packBlocks(src, max / bt, 2, staged, 0));
    try std.testing.expectError(error.Overflow, pk.packBlocks(src, 0, max / 2, staged, 0));
    try std.testing.expectError(error.Overflow, l.pointBytes(max / bt * bt));
    // a hand-made ref with hi <= lo
    try std.testing.expectError(error.BadRange, pk.packBlocks(.{ .lo = 2 * bt, .hi = bt, .stamp = 0 }, 0, 1, staged[0..l.block_bytes], 0));
    try std.testing.expectEqual(@as(usize, 0), n.copies);
    // generations take any range without wrapping
    g.bump(max - 1, max);
    try std.testing.expect(g.current(src));
}

test "an owner-sharded layout: 720 owner-only bytes a token (a carry's), 170 replicated, the part size in its digest" {
    const rep = try fixtureLayout(.replicated);
    const raw = try fixtureLayout(.owner_sharded);
    try std.testing.expectError(error.BadLayout, rep.withParts(1 << 20));
    try std.testing.expectError(error.BadLayout, raw.withParts(1000));
    const sh = try raw.withParts(1 << 20);
    // the owner's block is the replicated one (every plane), another part's point only its index rows
    try std.testing.expectEqual(rep.block_bytes, sh.blockBytesOf(true));
    try std.testing.expectEqual(@as(usize, 170 * 2048), sh.blockBytesOf(false));
    try std.testing.expectEqual(@as(usize, 720), sh.owner_token_bytes);
    try std.testing.expectEqual(@as(usize, 0), rep.owner_token_bytes);
    try std.testing.expectEqual(rep.block_bytes, rep.blockBytesOf(false)); // (replicated: every rank every plane)
    try std.testing.expectEqual(sh.owner_token_bytes + sh.foreign_block_bytes / 2048, sh.token_bytes);
    try std.testing.expectEqual(@as(usize, 511 * 170 * 2048 + 6_684_672), try sh.pointBytesOf(1_046_528, false));
    // a foreign block keeps the replicated planes in order, densely
    var at: usize = 0;
    for (sh.blockPlanes(), 0..) |p, i| {
        if (p.owner_only) {
            try std.testing.expectEqual(absent, sh.blockOffsetOf(i, false));
            continue;
        }
        try std.testing.expectEqual(at, sh.blockOffsetOf(i, false));
        at += p.spanBytes(2048);
    }
    try std.testing.expectEqual(sh.foreign_block_bytes, at);
    // the part size is in the digest (another TF_DS_POOL is another store); unset parts and the replicated layout differ
    const sh2 = try raw.withParts(1 << 19);
    try std.testing.expect(!std.mem.eql(u8, &sh.digest(), &sh2.digest()));
    try std.testing.expect(!std.mem.eql(u8, &sh.digest(), &raw.digest()));
    try std.testing.expect(!std.mem.eql(u8, &sh.digest(), &rep.digest()));
    var b1: [StateLayout.max_encoded]u8 = undefined;
    var b2: [StateLayout.max_encoded]u8 = undefined;
    try std.testing.expectEqual((try rep.encode(&b1)).len + 8, (try sh.encode(&b2)).len); // (the replicated bytes unchanged)
    try std.testing.expectEqual(@as(u32, 1), sh.partOf(1 << 20));
    try std.testing.expectEqual(@as(u32, 0), rep.partOf(1 << 20));
}

test "owner-sharded packing: each rank packs only what it holds (its part's codes from local rows, the index keys once), round trips, clears only its own" {
    const gpa = std.testing.allocator;
    const c = try fixtureConfig();
    const raw = try fixtureLayout(.owner_sharded);
    const bt = raw.geo.block_tokens;
    const part = 2 * bt;
    const cap = 2 * part;
    const l = try raw.withParts(part);
    var n: Counts = .{};
    var devs: [2]FakeDev = undefined;
    var gens: [2]Generations = undefined;
    var pks: [2]Packer(FakeDev) = undefined;
    for (0..2) |r| {
        devs[r] = try FakeDev.makeParts(gpa, &c, cap, 2, part, &n);
        gens[r] = try Generations.init(gpa, cap, bt);
    }
    defer for (0..2) |r| {
        gpa.free(devs[r].mem);
        gens[r].deinit(gpa);
    };
    for (0..2) |r| pks[r] = .{ .dev = devs[r], .layout = &l, .gens = &gens[r], .rank = @intCast(r) };
    // the fake's planes: per source a part's codes and scales (part / ratio + 2 rows), the index keys pool-wide
    var want_mem: usize = 0;
    for (l.blockPlanes()) |p| want_mem += (if (p.owner_only) part else cap) / p.ratio * p.row + 2 * p.row;
    for (l.smallPlanes()) |p| want_mem += 2 * p.slotBytes();
    try std.testing.expectEqual(want_mem, devs[0].mem.len);
    const before = [2][]u8{ try gpa.dupe(u8, devs[0].mem), try gpa.dupe(u8, devs[1].mem) };
    defer for (before) |b| gpa.free(b);

    // a point of one block at base 2048 in part 0 (rank 0 owns it) and one at base 4096 + 2048 in part 1
    for ([_]usize{ bt, part + bt }) |base| for (0..2) |r| {
        const own = base / part == r;
        try std.testing.expectEqual(own, pks[r].owns(base));
        const src = try gens[r].pin(base, base + bt);
        const staged = try gpa.alloc(u8, l.blockBytesOf(own));
        defer gpa.free(staged);
        try std.testing.expectError(error.BadStaging, pks[r].packBlocks(src, 0, 1, staged[0 .. staged.len - 1], 0));
        _ = try pks[r].packBlocks(src, 0, 1, staged, 0);
        var want: std.ArrayList(u8) = .empty;
        defer want.deinit(gpa);
        try devs[r].expectBlocksSharded(&c, &want, gpa, base, 0, 1, bt, part, own);
        try std.testing.expectEqualSlices(u8, want.items, staged);
        // back into the other free extent of the same part (a restore into the recorded part): the same rows there
        const to = base / part * part + (if (base % part == 0) bt else 0);
        const dst = try gens[r].reserve(to, to + bt);
        const t = try pks[r].unpackBlocks(dst, 0, 1, staged, 0);
        try std.testing.expect(gens[r].valid(t));
        var again: std.ArrayList(u8) = .empty;
        defer again.deinit(gpa);
        try devs[r].expectBlocksSharded(&c, &again, gpa, to, 0, 1, bt, part, own);
        try std.testing.expectEqualSlices(u8, want.items, again.items);
        // a clear of that destination: the index rows to fresh bytes on both ranks, the codes on the owner only (the
        // other rank's local rows at the same offsets are an extent of its own part)
        const mem0 = try gpa.dupe(u8, devs[r].mem);
        defer gpa.free(mem0);
        _ = try pks[r].clearExtent(try gens[r].reserve(to, to + bt), 0);
        const start = to / part * part;
        for (0..c.layers) |i| {
            if (devs[r].comp_codes[i] == 0) continue;
            const ratio: usize = c.ratios[i];
            const rows = bt / ratio;
            const lr = (to - start) / ratio;
            for ([_][2]usize{ .{ devs[r].comp_codes[i], c.head_dim / 2 }, .{ devs[r].comp_scales[i], c.head_dim / 16 } }) |pl| {
                const got = devs[r].bytes(pl[0] + lr * pl[1], rows * pl[1]);
                if (own) {
                    for (got) |b| try std.testing.expectEqual(@as(u8, 0), b);
                } else try std.testing.expectEqualSlices(u8, mem0[pl[0] - FakeDev.origin + lr * pl[1] ..][0 .. rows * pl[1]], got);
            }
            for (devs[r].bytes(devs[r].idx_scales[i] + to / ratio * (c.index_head_dim / 32), rows * (c.index_head_dim / 32))) |b| try std.testing.expectEqual(@as(u8, 127), b);
        }
    };
    // an extent across the parts is refused before any copy, and so is a layout never cut into parts
    const copies = n.copies;
    const staged = try gpa.alloc(u8, l.block_bytes);
    defer gpa.free(staged);
    try std.testing.expectError(error.CrossesPart, pks[0].packBlocks(try gens[0].pin(bt, part + bt), 0, 1, staged, 0));
    try std.testing.expectError(error.CrossesPart, pks[0].clearExtent(try gens[0].reserve(bt, part + bt), 0));
    var uncut: Packer(FakeDev) = .{ .dev = devs[0], .layout = &raw, .gens = &gens[0] };
    try std.testing.expectError(error.NoParts, uncut.packBlocks(try gens[0].pin(0, bt), 0, 1, staged, 0));
    try std.testing.expectEqual(copies, n.copies);
    // nothing outside the touched extents moved (the guard rows past a part's rows included)
    for (0..2) |r| for (l.blockPlanes()) |pl| {
        const rows_total = (if (pl.owner_only) part else cap) / pl.ratio;
        const a = pks[r].dev.blockPlane(pl) - FakeDev.origin;
        try std.testing.expectEqualSlices(u8, before[r][a + rows_total * pl.row ..][0 .. 2 * pl.row], devs[r].mem[a + rows_total * pl.row ..][0 .. 2 * pl.row]);
    };
}
test "a pin goes stale when its rows or slot are given away; a delayed completion never attaches to the new occupant" {
    const gpa = std.testing.allocator;
    const c = try fixtureConfig();
    const l = try fixtureLayout(.replicated);
    const bt = l.geo.block_tokens;
    var n: Counts = .{};
    const dev = try FakeDev.make(gpa, &c, 4 * bt, 2, &n);
    defer gpa.free(dev.mem);
    var g = try Generations.init(gpa, 4 * bt, bt);
    defer g.deinit(gpa);
    var pk: Packer(FakeDev) = .{ .dev = dev, .layout = &l, .gens = &g };
    const staged = try gpa.alloc(u8, l.block_bytes);
    defer gpa.free(staged);

    const old = try g.pin(0, 2 * bt);
    const t = try pk.packBlocks(old, 0, 1, staged, 0);
    // a placement stales the point (a stream writes over its rows), and a new point keeps the same base
    g.bump(bt, 3 * bt);
    try std.testing.expect(!g.current(old));
    const new = try g.pin(0, 2 * bt);
    try std.testing.expect(t.belongsTo(old) and !t.belongsTo(new)); // the old pack's bytes are the old point's
    try std.testing.expectError(error.StaleExtent, pk.packBlocks(old, 0, 1, staged, 0)); // later packs are refused
    // a bump elsewhere leaves a pin alone
    const far = try g.pin(3 * bt, 4 * bt);
    g.bump(0, bt);
    try std.testing.expect(g.current(far));
    // a restore's destination: an overlapping bump during it voids the publication
    const dst = try g.reserve(2 * bt, 3 * bt);
    const u = try pk.unpackBlocks(dst, 0, 1, staged, 0);
    try std.testing.expect(g.valid(u));
    g.bump(2 * bt + 5, 2 * bt + 6);
    try std.testing.expect(!g.valid(u));
    // a slot taken by another stream
    const s0 = try g.pinSlot(0);
    g.bumpSlot(0);
    const small = try gpa.alloc(u8, l.small_bytes);
    defer gpa.free(small);
    try std.testing.expectError(error.StaleSlot, pk.packSmall(s0, small, 0));
    // generations bounds
    try std.testing.expectError(error.BadRange, g.pin(0, 4 * bt + 1));
    try std.testing.expectError(error.BadRange, g.pin(5, 5));
}

test "a failed copy still records its event" {
    const gpa = std.testing.allocator;
    const c = try fixtureConfig();
    const l = try fixtureLayout(.replicated);
    const bt = l.geo.block_tokens;
    var n: Counts = .{};
    const dev = try FakeDev.make(gpa, &c, 4 * bt, 2, &n);
    defer gpa.free(dev.mem);
    var g = try Generations.init(gpa, 4 * bt, bt);
    defer g.deinit(gpa);
    var pk: Packer(FakeDev) = .{ .dev = dev, .layout = &l, .gens = &g };
    const staged = try gpa.alloc(u8, l.block_bytes);
    defer gpa.free(staged);
    const small = try gpa.alloc(u8, l.small_bytes);
    defer gpa.free(small);
    n.fail_at = 5; // the sixth copy of the first block
    try std.testing.expectError(error.FakeCopyFailed, pk.unpackBlocks(try g.reserve(0, bt), 0, 1, staged, 0));
    try std.testing.expectEqual(@as(usize, 1), n.records);
    n.fail_at = n.copies + 3;
    try std.testing.expectError(error.FakeCopyFailed, pk.unpackSmall(try g.reserveSlot(1), small, 0));
    try std.testing.expectEqual(@as(usize, 2), n.records);
    // a check that fails enqueues nothing and records nothing
    try std.testing.expectError(error.BadStaging, pk.unpackSmall(try g.reserveSlot(1), staged, 0));
    try std.testing.expectEqual(@as(usize, 2), n.records);
}

test "a failed restore's destination is cleared to a fresh pool's bytes, index scales 127, only while it is still reserved" {
    const gpa = std.testing.allocator;
    const c = try fixtureConfig();
    const l = try fixtureLayout(.replicated);
    const bt = l.geo.block_tokens;
    var n: Counts = .{};
    const dev = try FakeDev.make(gpa, &c, 4 * bt, 2, &n);
    defer gpa.free(dev.mem);
    const before = try gpa.dupe(u8, dev.mem);
    defer gpa.free(before);
    var g = try Generations.init(gpa, 4 * bt, bt);
    defer g.deinit(gpa);
    var pk: Packer(FakeDev) = .{ .dev = dev, .layout = &l, .gens = &g };
    const dst = try g.reserve(bt, 3 * bt);
    const t = try pk.clearExtent(dst, 0);
    try std.testing.expect(!g.current(dst) and g.valid(t));
    var buf: [4 * max_layers][3]usize = undefined;
    for (dev.extentPlanes(&c, &buf), 0..) |pl, k| {
        const lo = bt / pl[2] * pl[1];
        const len = 2 * bt / pl[2] * pl[1];
        // Caches.clear's bytes: the index keys' scales (each source's 4th plane) 127, zeros elsewhere
        const idx_scales = pl[1] == c.index_head_dim / 32 and k % 4 == 3;
        for (dev.bytes(pl[0] + lo, len)) |b| try std.testing.expectEqual(@as(u8, if (idx_scales) 127 else 0), b);
        // the rows on either side keep their bytes
        try std.testing.expectEqualSlices(u8, before[pl[0] - FakeDev.origin ..][0..lo], dev.bytes(pl[0], lo));
        try std.testing.expectEqualSlices(u8, before[pl[0] - FakeDev.origin + lo + len ..][0 .. 2 * pl[1]], dev.bytes(pl[0] + lo + len, 2 * pl[1]));
    }
    const ds = try g.reserveSlot(1);
    _ = try pk.clearSlot(ds, 0);
    for (l.smallPlanes()) |p| {
        for (dev.bytes(dev.slotPlane(p, 1), p.slotBytes())) |b| try std.testing.expectEqual(@as(u8, 0), b);
        try std.testing.expectEqualSlices(u8, before[dev.slotPlane(p, 0) - FakeDev.origin ..][0..p.slotBytes()], dev.bytes(dev.slotPlane(p, 0), p.slotBytes()));
    }
    try std.testing.expectError(error.BadView, pk.clearExtent(.{ .lo = 1, .hi = 3 * bt, .stamp = 0 }, 0)); // a ratio-2 plane's half row

    // an abort after the reservation was voided: the new occupant's rows and slot are left alone
    @memcpy(dev.mem, before);
    const voided = try g.reserve(2 * bt, 3 * bt);
    const vslot = try g.reserveSlot(0);
    g.bump(2 * bt, 4 * bt); // a placement gave [2048*2, 2048*4) to another stream
    g.bumpSlot(0); // and slot 0 to another stream
    const c0 = n.copies;
    const r0 = n.records;
    try std.testing.expectError(error.StaleExtent, pk.clearExtent(voided, 0));
    try std.testing.expectError(error.StaleSlot, pk.clearSlot(vslot, 0));
    try std.testing.expectEqual(c0, n.copies);
    try std.testing.expectEqual(r0, n.records);
    try std.testing.expectEqualSlices(u8, before, dev.mem);
}

test "an eligible point's resumed replay rebuilds the drafter's read set; an ineligible one is refused" {
    const window = 128;
    const chunk = 2048;
    const ring = window + 16;
    var len: usize = chunk + 1;
    while (len < 6 * chunk) : (len += 61) {
        var at: usize = chunk;
        while (at < len) : (at += chunk) {
            if (eligible(at, len, window, chunk)) {
                try checkReplayRebuilds(at, len, window, chunk, ring);
                const r = replayAbsorbed(at, len, window, chunk, ring).?;
                try std.testing.expectEqual(len, r[1]);
                try std.testing.expect(r[0] <= len - window and len - r[0] <= ring); // the 128 newest, within the ring
            } else try std.testing.expectError(error.NotEligible, checkReplayRebuilds(at, len, window, chunk, ring));
        }
    }
    // the boundary: at == len - window is eligible, one block later is not
    try checkReplayRebuilds(2 * chunk, 2 * chunk + window, window, chunk, ring);
    try std.testing.expect(!eligible(2 * chunk, 2 * chunk + window - 1, window, chunk));
    try std.testing.expect(!eligible(0, 3 * chunk, window, chunk));
    try std.testing.expect(!eligible(chunk + 1, 3 * chunk, window, chunk));
    // a replay that skipped the head would leave the drafter without its window
    try std.testing.expectEqual(@as(?[2]usize, null), replayAbsorbed(chunk, chunk, window, chunk, ring));
    // inventory: every saved item is in the layout, every other one says how it comes back
    var saved: usize = 0;
    for (inventory) |it| {
        if (it.source == .saved) saved += 1;
        try std.testing.expect(it.rule.len > 0);
    }
    try std.testing.expectEqual(@as(usize, 4), saved);
}

test "the Engram sequence is the request's raw ids with image spans dead" {
    const ids = [_]u32{ 5, 6, 129264, 129264, 7 };
    try checkEngramSeq(&.{ 5, 6, -1, -1, 7 }, &ids, &.{.{ 2, 2 }});
    try std.testing.expectError(error.EngramSeqMismatch, checkEngramSeq(&.{ 5, 6, 129264, 129264, 7 }, &ids, &.{.{ 2, 2 }}));
    try std.testing.expectError(error.EngramSeqMismatch, checkEngramSeq(&.{ 5, 6, -1, -1, 8 }, &ids, &.{.{ 2, 2 }}));
    try std.testing.expectError(error.EngramSeqMismatch, checkEngramSeq(&.{ 5, 6, -1 }, &ids, &.{.{ 2, 2 }}));
}

test "switch off: the replicated layout's digest is round 2's (no store namespace moved); a replicated packer packs as before" {
    // pinned from port/integration 98370ef's spill_state (before owner_sharded packing existed)
    const l = try fixtureLayout(.replicated);
    try std.testing.expectEqualStrings("7d85b94781dcb47b8c5aa7ac0cf676c09e77629f9d1259d83714ae075a5d6873", &std.fmt.bytesToHex(l.digest(), .lower));
    try std.testing.expectEqual(l.block_bytes, l.foreign_block_bytes);
    try std.testing.expectEqual(@as(usize, 0), l.owner_token_bytes);
    for (0..l.n_block) |i| try std.testing.expectEqual(l.blockOffset(i), l.blockOffsetOf(i, false));
}
