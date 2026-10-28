//! The ranks' agreement protocol for the spill tier and KV ownership. Port of wojo's GLM 0110 (PR #124: HAS, LOAD,
//! LOADPART, LOADED; BadAd84's block tier) and 0118 (PR #152: owner epochs, copy/move/carry, borrowed twins and their
//! UNTWIN/XMOVE), carried as ONE lanes.zig Kind (`rank_wire = 10`) so the follower's ordered command stream stays the
//! only path from rank 0 to the followers, and a reply channel back over the same link.
//!
//! A rank_wire frame is the lanes frame's payload after its kind and step: a fixed 32-byte header (magic, version, op,
//! flags, sender rank, engine epoch, transaction id, sequence number), then the op's fixed-schema body, all
//! little-endian, nothing after it. Parsing is bounded (no field sizes an allocation; every count is checked against a
//! constant before it is read) and canonical (a frame decodes only if re-encoding it gives the same bytes).
//!
//! Who sends what: rank 0 sends commands in the lanes stream; a follower answers an asking command (hello, has,
//! status, validate, shard_layout, shard_check, shard_xfer, stats) with exactly one reply frame on the same link,
//! straight from its command loop,
//! before it reads the next command. Rank 0 never has two asks outstanding and collects every follower's reply with a
//! deadline (Leader.collect). Every frame names the engine epoch and a transaction; a stale epoch, a skipped or
//! repeated sequence number, a transaction out of order or a phase out of order fails closed (the pair ends: there is
//! no local fallback while the peer follows another command sequence). docs/port/rank-wire.md is the specification.
const std = @import("std");
const link = @import("link.zig");
const fingerprint = @import("fingerprint.zig");
const parts = @import("parts.zig");

pub const Digest = fingerprint.Digest;
pub const OwnerLayout = fingerprint.OwnerLayout;

/// The lanes.zig Kind value that carries these frames (round 2: `rank_wire = 10` in `Kind`).
pub const lanes_kind: u8 = 10;
/// "RW", the first two bytes of every rank_wire frame.
pub const magic: u16 = 0x5752;
/// This protocol's version (in every header and in the fingerprint's wire_version field). 2: kept ids in reserve and
/// commit, the keep/share/touch/forget notices, the status ask, shard layout carry/twins, the shard stats set.
pub const version: u16 = 2;
pub const header_len = 32;
/// A command frame at most (inside the follower's receive buffer, whose 64 KiB slack beyond the prompt ids it fits).
pub const max_frame: usize = 60 << 10;
/// A reply frame at most.
pub const max_reply: usize = 4 << 10;
pub const max_world = fingerprint.max_world;
/// Restores in flight at once (each a transaction from has to commit/abort).
pub const max_restores = 16;
/// Blocks (2,048 tokens each) a point may have: far past the 1M window's 511.
pub const max_blocks: u32 = 1 << 16;
/// Ownership parts at most (parts.max_parts).
pub const max_parts = parts.max_parts;
/// Shard ops in one shard_ops frame (a longer plan goes in several frames of one transaction).
pub const max_shard_ops = 512;
/// Lineage forks one keep carries at most (a lineage with more is never written while followers run: rank 0's tier
/// refuses the keep; format.Limits.max_forks is 4096, far past any conversation's resumes in a 1M window).
pub const max_keep_forks = 2048;
/// Pictures one keep carries at most (spill-pictures' max_note_media: a point with more is not written while followers
/// run). Its frame with max_keep_forks forks too still fits max_frame (checked below).
pub const max_keep_media = 256;
pub const no_rank: u16 = 0xffff;
pub const no_block: u32 = 0xffff_ffff;

pub const Op = enum(u8) {
    // startup agreement
    hello = 1, // rank 0 -> all, asks: its fingerprint exchange
    hello_reply = 2, // follower -> rank 0: ok or the first differing field, and its own exchange
    agreed = 3, // rank 0 -> all: the common digest and every rank's digest, in rank order
    // spill restore (one transaction: has, reserve, start, progress*, validate, commit | abort at any point)
    has = 16, // asks: does this rank hold the committed point under this compat digest?
    has_reply = 17,
    reserve = 18, // reserve the destination slot/extent (nothing is read yet)
    start = 19, // start this rank's disk reads into its slots
    progress = 20, // copy blocks below `upto` into the extent (rank 0 copied that far)
    validate = 21, // asks: copy the rest, install the small state, report
    validate_reply = 22,
    commit = 23, // every rank validated: publish
    abort = 24, // drop the restore (reason, failing rank), release the reservation
    status = 25, // asks: how far this rank's copies of the restore are, answered at once from memory (never waits)
    status_reply = 26,
    // spill writes and point metadata (notices, each its own transaction)
    write_staged = 32, // a kept point's pending rows were staged before its source changes
    write_cancelled = 33, // a kept point's unpublished write was cancelled (reason)
    keep = 34, // rank 0 kept a state and writes it: every follower writes its own copy (write at keep)
    share = 35, // a point is shared (a prefix other conversations resume): kept longest at eviction
    touch = 36, // a point was used (its LRU position)
    forget = 37, // every rank drops a point
    // KV ownership
    shard_layout = 48, // asks: this rank's capacity and layout version (part agreement)
    shard_layout_reply = 49,
    shard_set = 50, // the agreed part size and the first owner epoch
    shard_ops = 51, // a plan's ops (parts.Op records), in order, one owner epoch a plan
    shard_check = 52, // asks: the planner digest at this owner epoch
    shard_check_reply = 53,
    shard_xfer = 54, // asks: did this rank's side of a stream's copy/carry/move transfer succeed?
    shard_xfer_reply = 55,
    shard_finish = 56, // rank 0's ok for a stream's copy/carry/move: every rank calls parts.finish(stream, ok) with it
    // counters
    stats = 64, // asks: this rank's counters (cache-tier's per-rank TierCounts ride this reply)
    stats_reply = 65,

    /// The reply an asking command expects; null for a command that is not answered (and for replies).
    pub fn reply(op: Op) ?Op {
        return switch (op) {
            .hello => .hello_reply,
            .has => .has_reply,
            .validate => .validate_reply,
            .status => .status_reply,
            .shard_layout => .shard_layout_reply,
            .shard_check => .shard_check_reply,
            .shard_xfer => .shard_xfer_reply,
            .stats => .stats_reply,
            else => null,
        };
    }

    pub fn isReply(op: Op) bool {
        return switch (op) {
            .hello_reply, .has_reply, .validate_reply, .status_reply, .shard_layout_reply, .shard_check_reply, .shard_xfer_reply, .stats_reply => true,
            else => false,
        };
    }
};

/// A follower's answer to has. `incomplete`: this rank's own write of the point is still queued (staged, not yet
/// published): rank 0 aborts the restore without `forget`, since the write publishes there soon and the point is
/// then restorable on every rank; `missing`, `damaged` and `incompatible` make every rank forget it (rank 0's copy
/// could never restore on the pair); `disabled`: this rank runs without the spill tier.
pub const Absence = enum(u8) { present = 0, missing = 1, incompatible = 2, incomplete = 3, damaged = 4, disabled = 5 };
/// Why a restore failed or was dropped.
pub const Failure = enum(u8) {
    none = 0,
    presence = 1, // a rank lacks the point (has)
    reserve = 2, // the destination could not be reserved
    read = 3, // a disk read failed
    crc = 4, // a block's CRC or digest did not match
    truncated = 5, // a file was shorter than its manifest
    missing = 6, // a file was gone
    cancelled = 7, // the request went away
    timeout = 8, // a rank's own copy deadline passed (reported, not silence)
    unpack = 9, // the small state or a block did not install
    superseded = 10, // a better source appeared
    shutdown = 11,
};
/// Why an unpublished write was cancelled.
pub const CancelWhy = enum(u8) { overwrite = 1, evict = 2, drop = 3, slot_reuse = 4, carry = 5, staging_full = 6, shutdown = 7, io = 8 };
pub const Status = enum(u8) { ok = 0, mismatch = 1 };

/// parts.Reason (shard-parts' planner): which rule chose a carry.
pub const Reason = parts.Reason;

/// A KV-ownership op: parts.Op itself (shard-parts' planner). The codec is reflective, so the frame layout follows the
/// type and `shard_schema` changes with it (a peer built with another Op layout is refused at hello and per frame).
pub const ShardOp = parts.Op;

/// The shard-op records' schema (field names, wire types, enum values): in every shard_ops frame and in the
/// fingerprint, so ranks built with different op layouts never exchange ops.
pub const shard_schema: u32 = schemaOf(ShardOp);

// -- the messages ---------------------------------------------------------------------------------------------------

pub const Hello = struct { world: u16, fx: fingerprint.Exchange };
pub const HelloReply = struct { status: Status, differ: u16, fx: fingerprint.Exchange };
pub const Agreed = struct { world: u16, common: Digest, ranks: [max_world]Digest };
pub const Has = struct { point: Digest, compat: Digest, at: u64, blocks: u32, layout: OwnerLayout, part: u8 };
pub const HasReply = struct { answer: Absence, point: Digest, compat: Digest };
/// `kept`: rank 0's Snap id of the destination (every rank installs the restored state under it; fixed for the
/// transaction). `generation`: the destination extent's parts generation when the point is owner_sharded (round 3's
/// planRestore), 0 when replicated (the Tracker refuses anything else).
pub const Reserve = struct { point: Digest, kept: u64, slot: u32, base: u64, size: u64, at: u64, blocks: u32, generation: u64, part: u8 };
pub const Start = struct { point: Digest, blocks: u32 };
pub const Progress = struct { upto: u32 };
pub const Validate = struct { upto: u32, shared: bool };
pub const ValidateReply = struct { ok: bool, why: Failure, copied: u32, bad_block: u32, point: Digest };
/// `kept` is the reserve's (Tracker: WrongPoint otherwise).
pub const Commit = struct { point: Digest, at: u64, kept: u64, shared: bool };
pub const Abort = struct { why: Failure, rank: u16, forget: bool };
/// The status ask names its restore by the header's txn alone.
pub const RestoreStatus = struct {};
/// `copied`: blocks this rank has copied into the destination, in order (enqueued); `small_ready`: the small state
/// installed and every copy of the restore completed on the device; `why`: the restore's recorded failure (none: no
/// failure so far). Complete = why none, copied = blocks, small_ready: validate then answers at once.
pub const RestoreStatusReply = struct { copied: u32, small_ready: bool, why: Failure, point: Digest };
/// `kept`: rank 0's Snap id of the kept state whose write is staged (Snap ids are never reused on a rank).
pub const WriteStaged = struct { point: Digest, kept: u64, upto: u32 };
pub const WriteCancelled = struct { point: Digest, kept: u64, why: CancelWhy };
/// A lineage fork (core/spill format.Fork, field for field).
pub const Fork = struct { name: [16]u8, at: u32 };
/// The point a kept state forks from (core/spill format.Origin, field for field).
pub const Origin = struct { key: Digest, at: u32 };
/// A picture before a kept point's cut (core/spill format.Media, field for field): its prompt positions
/// [start, start + len) and the full SHA-256 of its payload.
pub const Media = struct { start: u32, len: u32, digest: Digest };
/// Write at keep (lanes' Snap `kept`, kept from `slot` at `base` with `at` positions): every follower writes its own
/// copy of point `point` from the ids it holds in that slot, under the same lineage (`root`, the part salt `part`,
/// `forks`), `origin`, conversation `group` and `shared` flag. Sent for a dedup on rank 0 too (a follower that lost
/// the point writes it again). `media`: the point's pictures before `at` (in order, not overlapping, each starting
/// before `at`), and `placeholder` the model id at their raw positions: null exactly when there are none (a text
/// point). A rank that does not keep pictures skips such a keep (counted), as for any keep it cannot write.
pub const Keep = struct { slot: u32, kept: u64, base: u64, at: u32, point: Digest, group: u64, shared: bool, root: [16]u8, part: ?u32, origin: ?Origin, placeholder: ?u32 = null, forks: []const Fork, media: []const Media = &.{} };
pub const Share = struct { point: Digest };
pub const Touch = struct { point: Digest };
pub const Forget = struct { point: Digest };
/// `carry`, `twins`: rank 0's switches (TF_DS_KV_SHARD_CARRY / _TWINS).
pub const ShardLayout = struct { version: u16, parts: u8, cap: u64, carry: bool, twins: bool, schema: u32 };
/// The follower's own layout and switches (rank 0 compares them with its own: kv_lanes.Kv.start).
pub const ShardLayoutReply = struct { version: u16, parts: u8, cap: u64, carry: bool, twins: bool, schema: u32 };
pub const ShardSet = struct { version: u16, parts: u8, part_rows: u64, owner_epoch: u64 };
pub const ShardOps = struct { owner_epoch: u64, index: u16, last: bool, schema: u32 = shard_schema, ops: []const ShardOp };
pub const ShardCheck = struct { owner_epoch: u64, digest: Digest };
pub const ShardCheckReply = struct { owner_epoch: u64, digest: Digest };
/// `owner_epoch` is the epoch of the plan whose copy/carry/move op started the stream's transfer (its transaction).
pub const ShardXfer = struct { owner_epoch: u64, stream: u64 };
pub const ShardXferReply = struct { owner_epoch: u64, stream: u64, ok: bool };
pub const ShardFinish = struct { owner_epoch: u64, stream: u64, ok: bool };
/// Which counter set a stats ask wants (the reply's values are that set's fields in declaration order): `tier`, core
/// prompt_cache.TierCounts (the spill tier's); `shard`, core lanes.kv_parts.Rank (this rank's own KV-ownership
/// counters); `open`, the follower's store open after the agreement (open_values: whether it is open, and how far its
/// scan got, which grows while the scan works). A rank without that side answers n = 0.
pub const StatsSet = enum(u8) { tier = 1, shard = 2, open = 3 };
/// A stats reply of set `open`: values[0] 1 once this rank's store is open (0 while its scan runs), values[1] the
/// names its scan read so far.
pub const open_values: u8 = 2;
/// Counters a stats reply carries at most (core TierCounts has 37).
pub const max_stats = 64;
pub const Stats = struct { set: StatsSet };
/// `values[0..n]` the counters, the rest zero (one canonical form).
pub const StatsReply = struct { set: StatsSet, n: u8, values: [max_stats]u64 };

pub const Msg = union(Op) {
    hello: Hello,
    hello_reply: HelloReply,
    agreed: Agreed,
    has: Has,
    has_reply: HasReply,
    reserve: Reserve,
    start: Start,
    progress: Progress,
    validate: Validate,
    validate_reply: ValidateReply,
    commit: Commit,
    abort: Abort,
    status: RestoreStatus,
    status_reply: RestoreStatusReply,
    write_staged: WriteStaged,
    write_cancelled: WriteCancelled,
    keep: Keep,
    share: Share,
    touch: Touch,
    forget: Forget,
    shard_layout: ShardLayout,
    shard_layout_reply: ShardLayoutReply,
    shard_set: ShardSet,
    shard_ops: ShardOps,
    shard_check: ShardCheck,
    shard_check_reply: ShardCheckReply,
    shard_xfer: ShardXfer,
    shard_xfer_reply: ShardXferReply,
    shard_finish: ShardFinish,
    stats: Stats,
    stats_reply: StatsReply,
};

pub const Head = struct {
    op: Op,
    from: u16, // the sending rank
    epoch: u64, // the engine epoch (rank 0's, from hello on)
    txn: u64, // the transaction (0: hello and agreed)
    seq: u64, // rank 0's command sequence number; a reply repeats its command's
};

pub const Frame = struct { head: Head, msg: Msg };

// -- encoding -------------------------------------------------------------------------------------------------------

const W = struct {
    buf: []u8,
    n: usize = 0,

    fn bytes(w: *W, b: []const u8) !void {
        if (b.len > w.buf.len - w.n) return error.FrameTooLong;
        @memcpy(w.buf[w.n..][0..b.len], b);
        w.n += b.len;
    }

    fn int(w: *W, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try w.bytes(&b);
    }
};

const R = struct {
    b: []const u8,
    at: usize = 0,

    fn take(r: *R, n: usize) ![]const u8 {
        if (n > r.b.len - r.at) return error.ShortFrame;
        const s = r.b[r.at..][0..n];
        r.at += n;
        return s;
    }

    fn int(r: *R, comptime T: type) !T {
        return std.mem.readInt(T, (try r.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }
};

/// The wire integer of a field type: usize/isize as 64 bits (the same on every host), others rounded up to bytes.
fn WireInt(comptime T: type) type {
    if (T == usize) return u64;
    if (T == isize) return i64;
    const i = @typeInfo(T).int;
    return @Int(i.signedness, (i.bits + 7) / 8 * 8);
}

fn put(w: *W, comptime T: type, v: T) !void {
    if (T == fingerprint.Exchange) return putExchange(w, &v);
    switch (@typeInfo(T)) {
        .int => try w.int(WireInt(T), v),
        .bool => try w.int(u8, @intFromBool(v)),
        .@"enum" => |e| try put(w, e.tag_type, @intFromEnum(v)),
        .optional => |o| if (v) |x| {
            try w.int(u8, 1);
            try put(w, o.child, x);
        } else try w.int(u8, 0),
        .array => |a| {
            if (a.child == u8) return w.bytes(&v);
            for (v) |x| try put(w, a.child, x);
        },
        .@"struct" => |s| inline for (s.field_names, s.field_types) |n, FT| try put(w, FT, @field(v, n)),
        .@"union" => |u| switch (v) {
            inline else => |payload, tag| {
                const i = comptime fieldIndex(u.field_names, @tagName(tag));
                try w.int(u8, i);
                try put(w, @TypeOf(payload), payload);
            },
        },
        else => @compileError("rank_wire: no wire form for " ++ @typeName(T)),
    }
}

fn get(r: *R, comptime T: type) !T {
    if (T == fingerprint.Exchange) return getExchange(r);
    switch (@typeInfo(T)) {
        .int => return std.math.cast(T, try r.int(WireInt(T))) orelse error.BadValue,
        .bool => return switch (try r.int(u8)) {
            0 => false,
            1 => true,
            else => error.BadValue,
        },
        .@"enum" => |e| return std.enums.fromInt(T, try get(r, e.tag_type)) orelse error.BadValue,
        .optional => |o| return switch (try r.int(u8)) {
            0 => null,
            1 => try get(r, o.child),
            else => error.BadValue,
        },
        .array => |a| {
            var out: T = undefined;
            if (a.child == u8) {
                @memcpy(&out, try r.take(a.len));
            } else {
                for (&out) |*x| x.* = try get(r, a.child);
            }
            return out;
        },
        .@"struct" => |s| {
            var out: T = undefined;
            inline for (s.field_names, s.field_types) |n, FT| @field(out, n) = try get(r, FT);
            return out;
        },
        .@"union" => |u| {
            const tag = try r.int(u8);
            inline for (u.field_names, u.field_types, 0..) |n, FT, i| {
                if (tag == i) return @unionInit(T, n, try get(r, FT));
            }
            return error.BadValue;
        },
        else => @compileError("rank_wire: no wire form for " ++ @typeName(T)),
    }
}

/// The most items a message's slice field `name` carries (each slice field has its bound here and its buffer in Bufs,
/// under the same name).
fn sliceMax(comptime name: []const u8) usize {
    if (comptime std.mem.eql(u8, name, "forks")) return max_keep_forks;
    if (comptime std.mem.eql(u8, name, "media")) return max_keep_media;
    @compileError("rank_wire: no bound for the slice field " ++ name);
}

/// A message with slice fields (keep's forks and media): every field in declaration order, each slice as a u16 count
/// and its items (at most sliceMax(its name)).
fn putSliced(w: *W, comptime T: type, v: T) !void {
    const s = @typeInfo(T).@"struct";
    inline for (s.field_names, s.field_types) |n, FT| {
        if (comptime @typeInfo(FT) == .pointer) {
            const xs = @field(v, n);
            if (xs.len > (comptime sliceMax(n))) return error.BadCount;
            try w.int(u16, @intCast(xs.len));
            for (xs) |x| try put(w, @typeInfo(FT).pointer.child, x);
        } else try put(w, FT, @field(v, n));
    }
}

/// putSliced's message read back, each slice's items into the Bufs field of its name (more than its bound or than
/// that buffer holds: BadCount, before any item is read).
fn getSliced(r: *R, comptime T: type, bufs: Bufs) !T {
    var out: T = undefined;
    const s = @typeInfo(T).@"struct";
    inline for (s.field_names, s.field_types) |n, FT| {
        if (comptime @typeInfo(FT) == .pointer) {
            const buf = @field(bufs, n);
            const k = try r.int(u16);
            if (k > (comptime sliceMax(n)) or k > buf.len) return error.BadCount;
            for (buf[0..k]) |*x| x.* = try get(r, @typeInfo(FT).pointer.child);
            @field(out, n) = buf[0..k];
        } else @field(out, n) = try get(r, FT);
    }
    return out;
}

fn fieldIndex(comptime names: []const [:0]const u8, comptime n: []const u8) u8 {
    for (names, 0..) |x, i| if (std.mem.eql(u8, x, n)) return i;
    unreachable;
}

/// The most bytes a value of T takes on the wire (T without slices).
fn maxSize(comptime T: type) usize {
    if (T == fingerprint.Exchange) return 64 + 2 + fingerprint.max_fields * 35;
    return switch (@typeInfo(T)) {
        .int => @sizeOf(WireInt(T)),
        .bool => 1,
        .@"enum" => |e| maxSize(e.tag_type),
        .optional => |o| 1 + maxSize(o.child),
        .pointer => 2, // a trailing slice's count (its items are bounded separately)
        .array => |a| a.len * maxSize(a.child),
        .@"struct" => |s| blk: {
            var n: usize = 0;
            for (s.field_types) |FT| n += maxSize(FT);
            break :blk n;
        },
        .@"union" => |u| blk: {
            var n: usize = 0;
            for (u.field_types) |FT| n = @max(n, maxSize(FT));
            break :blk 1 + n;
        },
        else => @compileError("rank_wire: no wire form for " ++ @typeName(T)),
    };
}

fn putExchange(w: *W, x: *const fingerprint.Exchange) !void {
    if (x.n > fingerprint.max_fields) return error.BadValue;
    try w.bytes(&x.common);
    try w.bytes(&x.rank);
    try w.int(u16, x.n);
    for (x.entries()) |e| {
        try w.int(u16, e.id);
        try w.int(u8, @intFromEnum(e.scope));
        try w.bytes(&e.digest);
    }
}

fn getExchange(r: *R) !fingerprint.Exchange {
    var x: fingerprint.Exchange = .{ .common = undefined, .rank = undefined };
    @memset(&x.fields, .{ .id = 0, .scope = .common, .digest = @splat(0) });
    @memcpy(&x.common, try r.take(32));
    @memcpy(&x.rank, try r.take(32));
    x.n = try r.int(u16);
    if (x.n > fingerprint.max_fields) return error.BadCount;
    var last: u16 = 0;
    for (x.fields[0..x.n]) |*e| {
        e.id = try r.int(u16);
        if (e.id <= last) return error.BadValue; // ids strictly increasing, none 0: one canonical form
        last = e.id;
        e.scope = std.enums.fromInt(fingerprint.Scope, try r.int(u8)) orelse return error.BadValue;
        @memcpy(&e.digest, try r.take(32));
    }
    return x;
}

comptime {
    // a full plan frame fits a command frame; a hello reply fits a reply frame
    std.debug.assert(header_len + 8 + 2 + 1 + 4 + 2 + max_shard_ops * maxSize(ShardOp) <= max_frame);
    std.debug.assert(header_len + maxSize(HelloReply) <= max_reply);
    std.debug.assert(header_len + maxSize(Hello) <= max_frame);
    // a keep with its most forks and its most pictures fits a command frame; every reply fits a reply frame
    std.debug.assert(header_len + maxSize(Keep) + max_keep_forks * maxSize(Fork) + max_keep_media * maxSize(Media) <= max_frame);
    for (@typeInfo(Msg).@"union".field_names, @typeInfo(Msg).@"union".field_types) |n, FT| {
        const op = @field(Op, n);
        if (op.isReply()) std.debug.assert(header_len + maxSize(FT) <= max_reply);
    }
}

/// Every message type's wire schema (field names, wire types, enum values, in declaration order): in the
/// fingerprint's wire_version field (stampBytes), so two builds whose messages differ in any layout never agree.
pub const msg_schema: u32 = schemaOf(Msg);

/// The frame for `head` and `msg` in `buf` (max_frame for a command, max_reply for a reply); its bytes.
pub fn encode(buf: []u8, head: Head, msg: Msg) ![]u8 {
    if (head.op != std.meta.activeTag(msg)) return error.OpMismatch;
    try check(msg);
    const limit: usize = if (head.op.isReply()) max_reply else max_frame;
    var w: W = .{ .buf = buf[0..@min(buf.len, limit)] };
    try w.int(u16, magic);
    try w.int(u16, version);
    try w.int(u8, @intFromEnum(head.op));
    try w.int(u8, 0);
    try w.int(u16, head.from);
    try w.int(u64, head.epoch);
    try w.int(u64, head.txn);
    try w.int(u64, head.seq);
    switch (msg) {
        .shard_ops => |m| {
            if (m.ops.len > max_shard_ops) return error.BadCount;
            try w.int(u64, m.owner_epoch);
            try w.int(u16, m.index);
            try w.int(u8, @intFromBool(m.last));
            try w.int(u32, m.schema);
            try w.int(u16, @intCast(m.ops.len));
            for (m.ops) |o| try put(&w, ShardOp, o);
        },
        .keep => |m| try putSliced(&w, Keep, m),
        inline else => |m| try put(&w, @TypeOf(m), m),
    }
    return buf[0..w.n];
}

/// Where a decoded frame's variable parts go: a shard_ops frame's ops, a keep's forks and media (the message's
/// slices point there; a slice field's buffer has its name). A frame with more than its buffer holds is BadCount
/// (never an overrun).
pub const Bufs = struct { ops: []ShardOp = &.{}, forks: []Fork = &.{}, media: []Media = &.{} };

/// A frame's header and message, checked: bounds, magic, version, a known op, the op's exact schema and its values.
/// A shard_ops frame's ops go into `ops` (the message's slice points there); a keep with forks needs decodeInto.
pub fn decode(bytes: []const u8, ops: []ShardOp) !Frame {
    return decodeInto(bytes, .{ .ops = ops });
}

/// decode with every variable part's buffer (Bufs).
pub fn decodeInto(bytes: []const u8, bufs: Bufs) !Frame {
    const ops = bufs.ops;
    if (bytes.len > max_frame) return error.FrameTooLong;
    var r: R = .{ .b = bytes };
    if (try r.int(u16) != magic) return error.BadMagic;
    if (try r.int(u16) != version) return error.BadVersion;
    const op = std.enums.fromInt(Op, try r.int(u8)) orelse return error.BadOp;
    if (try r.int(u8) != 0) return error.BadFlags;
    if (op.isReply() and bytes.len > max_reply) return error.FrameTooLong;
    const head: Head = .{ .op = op, .from = try r.int(u16), .epoch = try r.int(u64), .txn = try r.int(u64), .seq = try r.int(u64) };
    const msg: Msg = switch (op) {
        .shard_ops => blk: {
            const owner_epoch = try r.int(u64);
            const index_ = try r.int(u16);
            const last = try get(&r, bool);
            const schema = try r.int(u32);
            if (schema != shard_schema) return error.SchemaMismatch;
            const n = try r.int(u16);
            if (n > max_shard_ops or n > ops.len) return error.BadCount;
            for (ops[0..n]) |*o| o.* = try get(&r, ShardOp);
            break :blk .{ .shard_ops = .{ .owner_epoch = owner_epoch, .index = index_, .last = last, .schema = schema, .ops = ops[0..n] } };
        },
        .keep => .{ .keep = try getSliced(&r, Keep, bufs) },
        inline else => |t| @unionInit(Msg, @tagName(t), try get(&r, @FieldType(Msg, @tagName(t)))),
    };
    if (r.at != bytes.len) return error.TrailingBytes;
    try check(msg);
    return .{ .head = head, .msg = msg };
}

fn zeroDigest(d: *const Digest) bool {
    return std.mem.allEqual(u8, d, 0);
}

/// A message's own values (what needs no state): counts, ranges, reason codes that must agree with flags.
pub fn check(msg: Msg) !void {
    switch (msg) {
        .hello => |m| if (m.world < 2 or m.world > max_world) return error.BadWorld,
        .hello_reply => |m| if ((m.status == .ok) != (m.differ == 0)) return error.BadValue,
        .agreed => |m| {
            if (m.world < 2 or m.world > max_world) return error.BadWorld;
            for (m.ranks[m.world..]) |*d| if (!zeroDigest(d)) return error.BadValue;
        },
        .has => |m| {
            if (m.blocks == 0 or m.blocks > max_blocks or m.at == 0) return error.BadValue;
            if (m.part >= max_parts or (m.layout == .replicated and m.part != 0)) return error.BadPart;
        },
        .has_reply => {},
        .reserve => |m| {
            if (m.blocks == 0 or m.blocks > max_blocks or m.at == 0 or m.size < m.at) return error.BadValue;
            if (m.part >= max_parts) return error.BadPart;
        },
        .start => |m| if (m.blocks == 0 or m.blocks > max_blocks) return error.BadValue,
        .progress => |m| if (m.upto == 0 or m.upto > max_blocks) return error.BadValue,
        .validate => |m| if (m.upto == 0 or m.upto > max_blocks) return error.BadValue,
        .validate_reply => |m| {
            if (m.ok != (m.why == .none)) return error.BadValue;
            if (m.copied > max_blocks or (m.bad_block != no_block and m.bad_block >= max_blocks)) return error.BadValue;
        },
        .commit => |m| if (m.at == 0) return error.BadValue,
        .status => {},
        .status_reply => |m| if (m.copied > max_blocks) return error.BadValue,
        .abort => |m| {
            if (m.why == .none) return error.BadValue;
            if (m.rank != no_rank and m.rank >= max_world) return error.BadValue;
        },
        .write_staged => |m| if (m.upto > max_blocks) return error.BadValue,
        .write_cancelled, .share, .touch, .forget => {},
        .keep => |m| {
            if (m.at == 0 or m.kept == 0) return error.BadValue;
            if (m.forks.len > max_keep_forks or m.media.len > max_keep_media) return error.BadCount;
            // pictures: a placeholder exactly when there are any; each non-empty, in order, not overlapping, starting
            // before the cut (one canonical form for every point)
            if ((m.placeholder != null) != (m.media.len > 0)) return error.BadValue;
            var end: u64 = 0;
            for (m.media) |x| {
                if (x.len == 0 or x.start < end or x.start >= m.at) return error.BadValue;
                end = @as(u64, x.start) + x.len;
            }
        },
        .shard_layout => |m| {
            if (m.parts == 0 or m.parts > max_parts or m.version == 0) return error.BadPart;
            if (m.schema != shard_schema) return error.SchemaMismatch;
        },
        .shard_layout_reply => |m| if (m.parts == 0 or m.parts > max_parts or m.version == 0) return error.BadPart,
        .shard_set => |m| if (m.parts == 0 or m.parts > max_parts or m.version == 0 or m.part_rows == 0) return error.BadPart,
        .shard_ops => |m| {
            if (m.ops.len > max_shard_ops) return error.BadCount;
            if (m.schema != shard_schema) return error.SchemaMismatch;
        },
        .shard_check, .shard_check_reply, .shard_xfer, .shard_xfer_reply, .shard_finish, .stats => {},
        .stats_reply => |m| {
            if (m.n > max_stats) return error.BadCount;
            for (m.values[m.n..]) |v| if (v != 0) return error.BadValue;
        },
    }
}

// -- ordering: the transactions and their phases ------------------------------------------------------------------

/// A restore transaction's phase.
pub const Phase = enum(u8) { has, reserved, started, validating };

/// An open restore as every rank's tracker holds it: what has named (point, at, blocks, layout, part) and what reserve
/// fixed (the destination's kept id), so a later frame of the transaction that names another value is refused
/// (WrongPoint).
const Restore = struct { txn: u64 = 0, phase: Phase = .has, point: Digest = @splat(0), at: u64 = 0, blocks: u32 = 0, layout: OwnerLayout = .replicated, part: u8 = 0, kept: u64 = 0, upto: u32 = 0 };

/// Streams whose copy/carry/move is in flight at once (draft.max_streams).
pub const max_loads = 16;

/// A stream's transfer a plan started (a copy, carry or move op), open until its shard_finish.
const Load = struct { txn: u64 = 0, epoch: u64 = 0, stream: u64 = 0 };

/// The stream a shard op starts a transfer for (parts: the ops that set `loading`), or null.
pub fn loadOf(op: ShardOp) ?u64 {
    return switch (op) {
        .copy => |o| o.stream,
        .carry => |o| o.stream,
        .move => |o| o.stream,
        else => null,
    };
}

/// The order every rank checks the command stream against (rank 0 checks what it sends, a follower what it reads):
/// hello first (seq 0, setting the epoch), agreed next, then every command's seq one more than the last; a command
/// that opens a transaction (has, the spill notices write_staged/write_cancelled/keep/share/touch/forget,
/// shard_layout, shard_set, shard_check, stats, a plan's first shard_ops) takes the next transaction id; the rest name
/// an open one, in its phase order. Any break fails closed.
pub const Tracker = struct {
    epoch: u64 = 0,
    hello: bool = false,
    agreed: bool = false,
    next_seq: u64 = 0,
    last_txn: u64 = 0,
    restores: [max_restores]Restore = @splat(.{}),
    owner_epoch: u64 = 0,
    shard_set: bool = false,
    plan_txn: u64 = 0, // an open plan (shard_ops frames before its last)
    plan_next: u16 = 0,
    loads: [max_loads]Load = @splat(.{}), // transfers awaiting their shard_finish

    fn find(t: *Tracker, txn: u64) !*Restore {
        if (txn == 0) return error.UnknownTxn;
        for (&t.restores) |*x| if (x.txn == txn) return x;
        return error.UnknownTxn;
    }

    fn findLoad(t: *Tracker, stream: u64) ?*Load {
        for (&t.loads) |*x| if (x.txn != 0 and x.stream == stream) return x;
        return null;
    }

    /// Transfers awaiting their finish.
    pub fn loading(t: *const Tracker) usize {
        var n: usize = 0;
        for (t.loads) |x| n += @intFromBool(x.txn != 0);
        return n;
    }

    /// The point an open restore names, or null.
    pub fn pointOf(t: *const Tracker, txn: u64) ?Digest {
        if (txn == 0) return null;
        for (t.restores) |x| if (x.txn == txn) return x.point;
        return null;
    }

    /// The blocks an open restore names, or null.
    pub fn blocksOf(t: *const Tracker, txn: u64) ?u32 {
        if (txn == 0) return null;
        for (t.restores) |x| if (x.txn == txn) return x.blocks;
        return null;
    }

    /// Restores in flight.
    pub fn open(t: *const Tracker) usize {
        var n: usize = 0;
        for (t.restores) |x| n += @intFromBool(x.txn != 0);
        return n;
    }

    fn opens(msg: Msg) bool {
        return switch (msg) {
            .has, .write_staged, .write_cancelled, .keep, .share, .touch, .forget, .shard_layout, .shard_set, .shard_check, .stats => true,
            .shard_ops => |m| m.index == 0,
            else => false,
        };
    }

    /// Admits command `f` or says why it is out of order (nothing changes on an error).
    pub fn admit(t: *Tracker, f: *const Frame) !void {
        var n = t.*;
        try n.step(f);
        t.* = n;
    }

    fn step(t: *Tracker, f: *const Frame) !void {
        const h = f.head;
        if (h.op.isReply()) return error.UnexpectedReply;
        if (h.from != 0) return error.WrongRank;
        if (h.seq != t.next_seq) return error.OutOfOrder;
        switch (f.msg) {
            .hello => {
                if (t.hello or h.seq != 0) return error.OutOfOrder;
                if (h.epoch == 0) return error.StaleEpoch;
                if (h.txn != 0) return error.OutOfOrder;
                t.epoch = h.epoch;
                t.hello = true;
                t.next_seq += 1;
                return;
            },
            else => {},
        }
        if (!t.hello) return error.NotAgreed;
        if (h.epoch != t.epoch) return error.StaleEpoch;
        if (f.msg == .agreed) {
            if (t.agreed or h.txn != 0) return error.OutOfOrder;
            t.agreed = true;
            t.next_seq += 1;
            return;
        }
        if (!t.agreed) return error.NotAgreed;
        // a plan's frames are contiguous: nothing else between its first and its last
        if (t.plan_txn != 0 and !(f.msg == .shard_ops and h.txn == t.plan_txn)) return error.OutOfOrder;
        if (opens(f.msg)) {
            if (h.txn <= t.last_txn) return error.StaleTxn;
            if (h.txn != t.last_txn + 1) return error.OutOfOrder;
        }
        switch (f.msg) {
            .hello, .agreed => unreachable,
            .has => |m| {
                const slot = for (&t.restores) |*x| {
                    if (x.txn == 0) break x;
                } else return error.TooManyRestores;
                slot.* = .{ .txn = h.txn, .phase = .has, .point = m.point, .at = m.at, .blocks = m.blocks, .layout = m.layout, .part = m.part };
            },
            .reserve => |m| {
                const x = try t.find(h.txn);
                if (x.phase != .has) return error.OutOfOrder;
                if (!std.mem.eql(u8, &m.point, &x.point) or m.blocks != x.blocks or m.at != x.at or m.part != x.part) return error.WrongPoint;
                if (x.layout == .replicated and m.generation != 0) return error.BadValue; // (an owner_sharded extent's only)
                x.kept = m.kept;
                x.phase = .reserved;
            },
            .start => |m| {
                const x = try t.find(h.txn);
                if (x.phase != .reserved) return error.OutOfOrder;
                if (!std.mem.eql(u8, &m.point, &x.point) or m.blocks != x.blocks) return error.WrongPoint;
                x.phase = .started;
            },
            .progress => |m| {
                const x = try t.find(h.txn);
                if (x.phase != .started) return error.OutOfOrder;
                if (m.upto <= x.upto or m.upto > x.blocks) return error.OutOfOrder; // strictly forward, within the point
                x.upto = m.upto;
            },
            .status => {
                const x = try t.find(h.txn);
                if (x.phase != .started) return error.OutOfOrder; // (between start and validate)
            },
            .validate => |m| {
                const x = try t.find(h.txn);
                if (x.phase != .started) return error.OutOfOrder;
                if (m.upto != x.blocks) return error.OutOfOrder; // validate copies the rest: through the last block
                x.upto = m.upto;
                x.phase = .validating;
            },
            .commit => |m| {
                const x = try t.find(h.txn);
                if (x.phase != .validating) return error.OutOfOrder;
                if (!std.mem.eql(u8, &m.point, &x.point) or m.at != x.at or m.kept != x.kept) return error.WrongPoint;
                x.* = .{};
            },
            .abort => {
                const x = try t.find(h.txn);
                x.* = .{};
            },
            .write_staged, .write_cancelled, .keep, .share, .touch, .forget, .shard_layout => {},
            .shard_set => |m| {
                if (m.owner_epoch <= t.owner_epoch) return error.StaleEpoch;
                if (t.loading() != 0) return error.OutOfOrder; // a new layout under transfers in flight
                t.owner_epoch = m.owner_epoch;
                t.shard_set = true;
            },
            .shard_ops => |m| {
                if (!t.shard_set) return error.NotAgreed;
                if (m.index == 0) {
                    if (m.owner_epoch != t.owner_epoch + 1) return if (m.owner_epoch <= t.owner_epoch) error.StaleEpoch else error.OutOfOrder;
                    t.owner_epoch = m.owner_epoch;
                    if (!m.last) {
                        t.plan_txn = h.txn;
                        t.plan_next = 1;
                    }
                } else {
                    if (h.txn != t.plan_txn or m.index != t.plan_next) return error.OutOfOrder;
                    if (m.owner_epoch != t.owner_epoch) return error.StaleEpoch;
                    t.plan_next += 1;
                    if (m.last) {
                        t.plan_txn = 0;
                        t.plan_next = 0;
                    }
                }
                // the transfers this plan starts: one each stream, never one a stream already has in flight
                for (m.ops) |o| if (loadOf(o)) |stream| {
                    if (t.findLoad(stream)) |x| {
                        if (x.txn != h.txn) return error.StreamLoading;
                        continue; // (a plan may name its stream twice)
                    }
                    const slot = for (&t.loads) |*x| {
                        if (x.txn == 0) break x;
                    } else return error.TooManyLoads;
                    slot.* = .{ .txn = h.txn, .epoch = m.owner_epoch, .stream = stream };
                };
            },
            .shard_xfer => |m| {
                const x = t.findLoad(m.stream) orelse return error.UnknownStream;
                if (x.txn != h.txn) return error.OutOfOrder;
                if (x.epoch != m.owner_epoch) return error.StaleEpoch;
            },
            .shard_finish => |m| {
                const x = t.findLoad(m.stream) orelse return error.UnknownStream;
                if (x.txn != h.txn) return error.OutOfOrder;
                if (x.epoch != m.owner_epoch) return error.StaleEpoch;
                x.* = .{};
            },
            .stats => {},
            .shard_check => |m| {
                if (!t.shard_set) return error.NotAgreed;
                if (m.owner_epoch != t.owner_epoch) return error.StaleEpoch;
            },
            .hello_reply, .has_reply, .validate_reply, .status_reply, .shard_layout_reply, .shard_check_reply, .shard_xfer_reply, .stats_reply => unreachable,
        }
        if (opens(f.msg)) t.last_txn = h.txn;
        t.next_seq += 1;
    }
};

// -- rank 0 ---------------------------------------------------------------------------------------------------------

/// How long rank 0 waits for every follower's reply, by the asking op. A missed deadline ends the pair.
pub const Timeouts = struct {
    hello_ns: u64 = 120 * std.time.ns_per_s, // a follower may still be finishing its own start
    has_ns: u64 = 10 * std.time.ns_per_s, // a manifest lookup on its own disk
    validate_ns: u64 = follower_copy_ns + 30 * std.time.ns_per_s, // its copies through the last block, then the check
    shard_ns: u64 = 10 * std.time.ns_per_s, // shard_layout, shard_check
    xfer_ns: u64 = 30 * std.time.ns_per_s, // shard_xfer: the follower waits its side's transfer events first
    stats_ns: u64 = 2 * std.time.ns_per_s, // stats: from memory
    status_ns: u64 = 2 * std.time.ns_per_s, // status: from memory (no device wait, at most one ready block enqueued)
    /// Not an ask's deadline: how long rank 0 polls a restore's status (from when its own copies finished) before it
    /// gives the restore up as `timeout` without asking validate (the spill tier reads it from its Leader).
    status_wait_ns: u64 = status_wait_ns,

    pub fn of(t: Timeouts, op: Op) u64 {
        return switch (op) {
            .hello => t.hello_ns,
            .has => t.has_ns,
            .validate => t.validate_ns,
            .status => t.status_ns,
            .shard_xfer => t.xfer_ns,
            .stats => t.stats_ns,
            else => t.shard_ns,
        };
    }
};

/// A follower's own bound on finishing a restore's copies when validate asks (its disk reads included): past it the
/// follower answers `timeout`, never silence. Rank 0's validate deadline is longer by a margin.
pub const follower_copy_ns: u64 = 60 * std.time.ns_per_s;

/// A follower's budget for each command that is not an ask (reserve, start, progress; GLM's TF_GLM_SPILL_LOAD_MS
/// default): it copies what its reads already delivered and leaves the rest to its idle service and validate (serve's
/// rule 1). A status answer has no budget: it reports from memory.
pub const follower_progress_ns: u64 = 40 * std.time.ns_per_ms;

/// Timeouts.status_wait_ns's default: how long rank 0 polls a restore's status (from when its own copies finished)
/// before it gives the restore up as `timeout` without asking validate: a follower's disk that slow fails the
/// restore, never the decode rounds.
pub const status_wait_ns: u64 = follower_copy_ns;

/// How rank 0's frames reach the followers: lanes.zig begins a `rank_wire` frame, appends `sub`, and flushes it to
/// every follower (Model.followers), in the ordered command stream.
pub const Sink = struct {
    ctx: *anyopaque,
    send: *const fn (ctx: *anyopaque, sub: []const u8) anyerror!void,
};

/// Why the pair cannot go on, for the log and explain().
pub const Broken = struct { rank: u16, err: anyerror, field: u16 = 0 };

pub const Leader = struct {
    t: Tracker = .{},
    world: u16,
    epoch: u64,
    fx: fingerprint.Exchange,
    timeouts: Timeouts = .{},
    asked: ?Head = null,
    asked_msg: Msg = undefined, // the outstanding ask (its echo fields are checked in collect)
    agreement: ?fingerprint.Agreement = null,
    pair: Digest = @splat(0), // agreement.pair(): what every has carries as `compat`
    broken: ?Broken = null,
    buf: [max_frame]u8 = undefined,
    rbuf: [max_world - 1][max_reply]u8 = undefined,

    /// `epoch`: this engine instance's (nonzero, fresh at every start: freshEpoch). `fx`: rank 0's fingerprint.
    /// About 80 KiB: allocate it (gpa.create), never as a stack local.
    pub fn init(world: u16, epoch: u64, fx: fingerprint.Exchange) Leader {
        std.debug.assert(world >= 2 and world <= max_world and epoch != 0);
        return .{ .world = world, .epoch = epoch, .fx = fx };
    }

    /// The `compat` every has carries: the agreed pair's digest (Agreement.pair()), a value every rank holds. Each
    /// rank checks its own store's compat (opened with storeCommon(rank)) locally; this one only proves the ranks
    /// agreed the same pair.
    pub fn compat(l: *const Leader) Digest {
        return l.pair;
    }

    /// The transaction id a command takes: the next one for a command that opens one (`txn` must be 0), else `txn`.
    fn txnFor(l: *const Leader, msg: Msg, txn: u64) !u64 {
        if (msg == .hello or msg == .agreed) return if (txn == 0) 0 else error.OutOfOrder;
        if (Tracker.opens(msg)) return if (txn == 0) l.t.last_txn + 1 else error.OutOfOrder;
        return txn;
    }

    const Prepared = struct { head: Head, t: Tracker };

    /// Every check a command meets before it is sent, nothing changed: the header it would take and the tracker
    /// after it.
    fn prepare(l: *const Leader, txn: u64, msg: Msg) !Prepared {
        if (l.asked != null) return error.ReplyOutstanding; // never two asks, nor a command between an ask and its replies
        const op = std.meta.activeTag(msg);
        if (op.isReply()) return error.UnexpectedReply;
        try check(msg);
        const head: Head = .{ .op = op, .from = 0, .epoch = l.epoch, .txn = try l.txnFor(msg, txn), .seq = l.t.next_seq };
        var t = l.t;
        try t.admit(&.{ .head = head, .msg = msg });
        if (msg == .has and !std.mem.eql(u8, &msg.has.compat, &l.pair)) return error.WrongCompat;
        return .{ .head = head, .t = t };
    }

    /// Would post/ask take this command now? Nothing is sent, nothing changes. Rank 0 calls it BEFORE it changes its
    /// own state for a command it will post (Planner.apply/keepOp/finish, its own commit or abort, a staged write):
    /// a post that is refused ends the pair, because rank 0 has already acted and the followers have not.
    pub fn admissible(l: *const Leader, txn: u64, msg: Msg) !void {
        if (l.broken != null) return error.PairBroken;
        _ = try l.prepare(txn, msg);
    }

    /// Sends a command that is not answered; its header (the transaction a has/write/plan opened is `head.txn`).
    /// Any refusal (out of order, a bad value, the send failing) ends the pair: error.PairBroken, the cause in
    /// `broken.err` (call admissible first for a refusal that does not).
    pub fn post(l: *Leader, sink: Sink, txn: u64, msg: Msg) !Head {
        if (std.meta.activeTag(msg).reply() != null) return error.MustAsk;
        return l.emit(sink, txn, msg);
    }

    fn emit(l: *Leader, sink: Sink, txn: u64, msg: Msg) !Head {
        if (l.broken != null) return error.PairBroken;
        const is_ask = std.meta.activeTag(msg).reply() != null;
        // an ask refused before it is sent changes nothing anywhere (rank 0 has not acted on it): no break
        const p = l.prepare(txn, msg) catch |err| return if (is_ask) err else l.fail(0, err);
        const bytes = encode(&l.buf, p.head, msg) catch |err| return l.fail(0, err);
        sink.send(sink.ctx, bytes) catch |err| return l.fail(0, err);
        l.t = p.t;
        if (is_ask) {
            l.asked = p.head;
            l.asked_msg = msg;
        }
        return p.head;
    }

    /// Records why the pair ended (the first cause only: a later call keeps the first rank and cause) and returns
    /// error.PairBroken, the one error rank_wire surfaces for it (explain maps it; `broken` holds the rank and the
    /// cause; error.Timeout reads as ReplyTimeout). Public for rank 0's own causes the protocol cannot see (kv_lanes:
    /// its own side of a cross-part transfer failed or timed out): nothing more is sent after it.
    pub fn fail(l: *Leader, rank: u16, err: anyerror) anyerror {
        if (l.broken == null) l.broken = .{ .rank = rank, .err = if (err == error.Timeout) error.ReplyTimeout else err };
        return error.PairBroken;
    }

    /// Sends an asking command and collects every follower's reply into `replies` (rank 1 first); its header.
    /// The whole engine waits here (rank 0's round loop included) until every follower answered: up to the ask's
    /// Timeouts entry, which is a full engine stall when a follower is slow.
    pub fn ask(l: *Leader, sink: Sink, links: []const ?link.Link, txn: u64, msg: Msg, replies: []Frame) !Head {
        if (std.meta.activeTag(msg).reply() == null) return error.NotAnAsk;
        // sized before anything is sent: a mis-sized call must not leave an ask outstanding at the followers
        if (links.len != l.world - 1 or replies.len < links.len) return error.BadWorld;
        const head = try l.emit(sink, txn, msg);
        try l.collect(links, replies);
        return head;
    }

    /// Every follower's reply to the outstanding ask, within its deadline: links[i] is rank i + 1's (Model.followers()).
    /// A missing, late, garbled or mismatched reply (header or echoed fields) ends the pair (`broken` names the rank).
    pub fn collect(l: *Leader, links: []const ?link.Link, replies: []Frame) !void {
        const asked = l.asked orelse return error.NothingAsked;
        if (links.len != l.world - 1 or replies.len < links.len) return l.fail(no_rank, error.BadWorld);
        const want = asked.op.reply().?;
        const deadline = monotonicNs() +| l.timeouts.of(asked.op);
        var got: [max_world - 1]bool = @splat(false);
        var left: usize = links.len;
        while (left > 0) {
            const now = monotonicNs();
            if (now >= deadline) return l.fail(firstMissing(got[0..links.len]), error.ReplyTimeout);
            var waiting: [max_world - 1]link.Link = undefined;
            var which: [max_world - 1]usize = undefined;
            var n: usize = 0;
            for (links, 0..) |x, i| if (!got[i]) {
                waiting[n] = x orelse return l.fail(@intCast(i + 1), error.NoLink);
                which[n] = i;
                n += 1;
            };
            var ready: [max_world - 1]bool = undefined;
            const k = link.readableAny(waiting[0..n], ready[0..n], deadline - now) catch |err| return l.fail(no_rank, err);
            if (k == 0) continue; // (the deadline check above ends it)
            for (ready[0..n], 0..) |r, j| if (r) {
                const i = which[j];
                const rank: u16 = @intCast(i + 1);
                const bytes = links[i].?.recvWithin(&l.rbuf[i], deadline -| monotonicNs()) catch |err| return l.fail(rank, err);
                const f = decode(bytes, &.{}) catch |err| return l.fail(rank, err);
                if (f.head.op != want) return l.fail(rank, error.UnexpectedReply);
                if (f.head.epoch != l.epoch) return l.fail(rank, error.StaleEpoch);
                if (f.head.seq != asked.seq or f.head.txn != asked.txn) return l.fail(rank, error.OutOfOrder);
                if (f.head.from != rank) return l.fail(rank, error.WrongRank);
                echo(&l.t, asked, &l.asked_msg, f.msg) catch |err| return l.fail(rank, err);
                replies[i] = f;
                got[i] = true;
                left -= 1;
            };
        }
        l.asked = null;
    }

    fn firstMissing(got: []const bool) u16 {
        for (got, 0..) |g, i| if (!g) return @intCast(i + 1);
        return no_rank;
    }

    /// Startup agreement, before anything spill- or shard-related crosses: hello with rank 0's fingerprint, every
    /// follower's verdict and exchange, then agreed with the common digest and the per-rank digests in rank order.
    /// A difference is error.FingerprintMismatch, `broken` naming the first rank and field (describeBroken); any
    /// other failure is error.PairBroken.
    pub fn agree(l: *Leader, sink: Sink, links: []const ?link.Link) !fingerprint.Agreement {
        if (links.len != l.world - 1) return error.BadWorld;
        var replies: [max_world - 1]Frame = undefined;
        _ = try l.ask(sink, links, 0, .{ .hello = .{ .world = l.world, .fx = l.fx } }, replies[0..links.len]);
        var a: fingerprint.Agreement = .{ .common = l.fx.common, .world = l.world };
        a.ranks[0] = l.fx.rank;
        for (replies[0..links.len], 0..) |f, i| {
            const r = f.msg.hello_reply;
            const rank: u16 = @intCast(i + 1);
            // either side's verdict: the follower's, and rank 0's own reading of the follower's exchange
            const differ = if (r.status != .ok) r.differ else (l.fx.differs(&r.fx) orelse 0);
            if (differ != 0) {
                l.broken = .{ .rank = rank, .err = error.FingerprintMismatch, .field = differ };
                return error.FingerprintMismatch;
            }
            a.ranks[rank] = r.fx.rank;
        }
        _ = try l.post(sink, 0, .{ .agreed = .{ .world = l.world, .common = a.common, .ranks = a.ranks } });
        l.agreement = a;
        l.pair = a.pair();
        return a;
    }
};

/// The fields a reply must repeat from its ask, and what the ask's transaction fixes. Both sides check: a follower
/// before it answers (serve), rank 0 as it collects, so a buggy handler is caught on either rank.
fn echo(t: *const Tracker, asked: Head, ask_msg: *const Msg, r: Msg) !void {
    switch (r) {
        .hello_reply => {},
        .has_reply => |x| {
            if (!std.mem.eql(u8, &x.point, &ask_msg.has.point)) return error.WrongPoint;
            if (!std.mem.eql(u8, &x.compat, &ask_msg.has.compat)) return error.WrongCompat;
        },
        .validate_reply => |x| {
            const p = t.pointOf(asked.txn) orelse return error.UnknownTxn;
            if (!std.mem.eql(u8, &x.point, &p)) return error.WrongPoint;
            if (x.ok and x.copied != ask_msg.validate.upto) return error.BadEcho; // ok: every block copied
        },
        .status_reply => |x| {
            const p = t.pointOf(asked.txn) orelse return error.UnknownTxn;
            if (!std.mem.eql(u8, &x.point, &p)) return error.WrongPoint;
            if (x.copied > t.blocksOf(asked.txn).?) return error.BadEcho; // never past the point
        },
        .shard_layout_reply => |x| if (x.schema != ask_msg.shard_layout.schema or x.version != ask_msg.shard_layout.version) return error.BadEcho,
        .shard_check_reply => |x| if (x.owner_epoch != ask_msg.shard_check.owner_epoch) return error.BadEcho,
        .shard_xfer_reply => |x| if (x.stream != ask_msg.shard_xfer.stream or x.owner_epoch != ask_msg.shard_xfer.owner_epoch) return error.WrongStream,
        .stats_reply => |x| if (x.set != ask_msg.stats.set) return error.OpMismatch,
        else => return error.UnexpectedReply,
    }
}

/// "rank 1's fingerprint differs in engine_commit" and the like, for the start's refusal.
pub fn describeBroken(b: Broken, buf: []u8) []const u8 {
    var fb: [32]u8 = undefined;
    if (b.err == error.FingerprintMismatch)
        return std.fmt.bufPrint(buf, "rank {d}'s configuration differs from rank 0's in {s}", .{ b.rank, fingerprint.describe(b.field, &fb) }) catch "the ranks' configurations differ";
    return std.fmt.bufPrint(buf, "rank {d}: {s}", .{ b.rank, @errorName(b.err) }) catch "the ranks disagree";
}

/// A start or request this protocol refuses, in words (native.zig's explain: `if (rank_wire.explain(err)) |t| return t;`).
/// Only rank_wire's own error names match (a Leader failure always surfaces as error.PairBroken, its cause in
/// `broken`), so a link or ordering error from any other code is never explained as a spill/KV-shard one. The log
/// line from describeBroken names the rank, the cause and, for a fingerprint, the field.
pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.FingerprintMismatch => "the ranks' builds, kits, models or TF_DS_* settings differ (the log names the field): give every rank the same image, kit, model and environment",
        error.PairBroken => "a rank stopped answering, or the ranks' spill/KV-shard command streams diverged (the log names the rank and why): the pair cannot go on; restart both ranks",
        error.NoEngineId => "the spill tier and TF_DS_KV_SHARD need the engine's identity: set TF_DS_ENGINE_ID (the recipe exports it)",
        error.NoRecipePatches => "the spill tier and TF_DS_KV_SHARD need the recipe's identity: set TF_DS_RECIPE_PATCHES (the recipe exports it)",
        error.NoWeightsId => "the spill tier and TF_DS_KV_SHARD need the weights' content identity: set TF_DS_RANK_CACHE_SHA (this rank's file's digest) or TF_DS_MODEL_REVISION (the checkpoint and quant revision)",
        error.NoEngramId => "the spill tier and TF_DS_KV_SHARD with Engram need the tables' identity: set TF_DS_ENGRAM_REVISION (the Engram shards' revision or digest)",
        else => null,
    };
}

/// A fresh engine epoch (nonzero): rank 0 takes one at every start.
pub fn freshEpoch(io: std.Io) u64 {
    while (true) {
        var b: [8]u8 = undefined;
        io.random(&b);
        const e = std.mem.readInt(u64, &b, .little);
        if (e != 0) return e;
    }
}

/// Every follower said present.
pub fn allPresent(replies: []const Frame) bool {
    for (replies) |f| if (f.msg.has_reply.answer != .present) return false;
    return true;
}

/// Every follower's status: the first one that recorded a failure (abort with it, no validate), whether all are
/// complete (validate answers at once), and the fewest blocks any of them copied.
pub const StatusSummary = struct { failed: ?struct { rank: u16, why: Failure } = null, complete: bool = true, copied: u32 = no_block };

pub fn statusOf(replies: []const Frame, blocks: u32) StatusSummary {
    var out: StatusSummary = .{};
    for (replies, 0..) |f, i| {
        const s = f.msg.status_reply;
        if (s.why != .none and out.failed == null) out.failed = .{ .rank = @intCast(i + 1), .why = s.why };
        if (s.copied < blocks or !s.small_ready) out.complete = false;
        out.copied = @min(out.copied, s.copied);
    }
    if (out.copied == no_block) out.copied = blocks;
    return out;
}

/// The first follower whose validation failed (its rank and reason), or null when every one passed.
pub fn firstFailure(replies: []const Frame) ?struct { rank: u16, why: Failure, bad_block: u32 } {
    for (replies, 0..) |f, i| {
        const v = f.msg.validate_reply;
        if (!v.ok) return .{ .rank = @intCast(i + 1), .why = v.why, .bad_block = v.bad_block };
    }
    return null;
}

/// The fingerprint's wire_version field: this protocol's version, header size, the shard-op schema and every
/// message's schema (msg_schema).
pub fn stampBytes() [12]u8 {
    var b: [12]u8 = undefined;
    std.mem.writeInt(u16, b[0..2], version, .little);
    std.mem.writeInt(u16, b[2..4], header_len, .little);
    std.mem.writeInt(u32, b[4..8], shard_schema, .little);
    std.mem.writeInt(u32, b[8..12], msg_schema, .little);
    return b;
}

// -- followers ------------------------------------------------------------------------------------------------------

pub const Follower = struct {
    t: Tracker = .{},
    rank: u16,
    fx: fingerprint.Exchange,
    owed: ?Head = null, // an asking command not yet answered
    owed_msg: Msg = undefined, // that ask (serve checks the reply's echo against it)
    agreement: ?fingerprint.Agreement = null,
    pair: Digest = @splat(0), // agreement.pair(): the compat every has must carry
    mismatch: u16 = 0, // the first differing field rank 0's hello showed
    ops: [max_shard_ops]ShardOp = undefined,
    forks: [max_keep_forks]Fork = undefined,
    media: [max_keep_media]Media = undefined,
    rbuf: [max_reply]u8 = undefined,

    /// About 90 KiB (`ops`, `forks`, `media`): allocate it (gpa.create, or a field of a heap-held side), never as a stack local
    /// in follow().
    pub fn init(rank: u16, fx: fingerprint.Exchange) Follower {
        std.debug.assert(rank >= 1 and rank < max_world);
        return .{ .rank = rank, .fx = fx };
    }

    /// A rank_wire command (the lanes frame's rest), decoded and admitted in order. An asking command must be
    /// answered (reply) before the next is accepted.
    pub fn accept(f: *Follower, payload: []const u8) !Frame {
        if (f.owed != null) return error.ReplyOwed;
        const fr = try decodeInto(payload, .{ .ops = &f.ops, .forks = &f.forks, .media = &f.media });
        try f.t.admit(&fr);
        if (fr.head.op.reply() != null) {
            f.owed = fr.head;
            f.owed_msg = fr.msg;
        }
        return fr;
    }

    /// The one answer to the outstanding ask, sent to rank 0 at once.
    pub fn reply(f: *Follower, l: link.Link, msg: Msg) !void {
        const asked = f.owed orelse return error.NothingAsked;
        if (asked.op.reply().? != std.meta.activeTag(msg)) return error.OpMismatch;
        try echo(&f.t, asked, &f.owed_msg, msg);
        const head: Head = .{ .op = std.meta.activeTag(msg), .from = f.rank, .epoch = asked.epoch, .txn = asked.txn, .seq = asked.seq };
        const bytes = try encode(&f.rbuf, head, msg);
        f.owed = null;
        try l.send(bytes);
    }
};

fn need(comptime H: type, comptime name_: []const u8) bool {
    return @hasDecl(H, name_);
}

/// The follower's one dispatch case: decodes and admits `payload`, runs the handler's method for it, answers an ask
/// on `l`. `h` is a pointer to the family's spill/shard side; each method it lacks fails closed (error.Unsupported:
/// a feature this rank was not started with, which the fingerprint should already have refused). Methods:
///   has(Head, Has) !HasReply                validate(Head, Validate) !ValidateReply
///   status(Head, RestoreStatus) !RestoreStatusReply   (at once, from memory: no device or disk wait; at most
///     handing on completed copies and enqueuing one block already read)
///   reserve/start/progress/commit/abort(Head, <msg>) !void
///   writeStaged(Head, WriteStaged) !void    writeCancelled(Head, WriteCancelled) !void
///   keep(Head, Keep) !void   share(Head, Share) !void   touch(Head, Touch) !void   forget(Head, Forget) !void
///     (keep's `forks` and `media` point into the Follower's buffers: valid during the call only)
///   shardLayout(Head, ShardLayout) !ShardLayoutReply   shardSet(Head, ShardSet) !void
///   shardOps(Head, ShardOps) !void          shardCheck(Head, ShardCheck) !ShardCheckReply
///   shardXfer(Head, ShardXfer) !ShardXferReply   shardFinish(Head, ShardFinish) !void
///   stats(Head, Stats) !StatsReply
///   agreed(fingerprint.Agreement) !void     (optional: the startup agreement, e.g. to open the spill store)
/// hello and agreed are the Follower's own: a mismatch is answered, then error.FingerprintMismatch ends this rank.
/// has: serve has already checked `m.compat` against the agreed pair (Agreement.pair(); a difference ends the rank:
/// the streams diverged); the handler checks presence under its OWN store's compat (storeCommon(rank)) and echoes
/// `point` and `compat`. Every reply's echo fields (has: point, compat; validate: the transaction's point, and
/// copied = upto when ok; status: the transaction's point, copied <= blocks; shard_layout: version, schema;
/// shard_check: owner_epoch; shard_xfer: stream, owner_epoch; stats: set) are checked here before sending, and again
/// on rank 0.
///
/// The rules the handlers keep (rank 0's round loop keeps sending while a follower works; only an ask waits):
///   1. The commands that are not asks never wait on the disk: reserve, start, progress and the notices (keep, share,
///      touch, forget, write_staged, write_cancelled) take at most `follower_progress_ns` each (GLM's
///      TF_GLM_SPILL_LOAD_MS: 40 ms); a status answer takes no wait at all (from memory). progress copies only the
///      blocks whose reads already completed (or waits within that budget) and leaves the rest to the idle service;
///      rank 0 asks validate (which has the `follower_copy_ns` bound and its own reply) only once every status said
///      complete.
///   2. A read, CRC, truncation or missing-file failure in reserve/start/progress is RECORDED (the restore's state)
///      and reported by validate_reply (why, bad_block), never returned as an error. The same for a transfer's
///      failure in shardOps (shard_xfer_reply.ok = false).
///   3. A handler error is kept for a protocol or invariant break (a state the commands cannot reach, an op the
///      planner refuses, out of memory). It is never answered: this rank ends, and rank 0's collect sees the link go
///      down (error.PairBroken, cause PeerDown).
pub fn serve(f: *Follower, l: link.Link, payload: []const u8, h: anytype) !void {
    const H = @typeInfo(@TypeOf(h)).pointer.child;
    const fr = try f.accept(payload);
    const head = fr.head;
    switch (fr.msg) {
        .hello => |m| {
            const differ = f.fx.differs(&m.fx) orelse 0;
            f.mismatch = differ;
            try f.reply(l, .{ .hello_reply = .{ .status = if (differ == 0) .ok else .mismatch, .differ = differ, .fx = f.fx } });
            if (differ != 0) return error.FingerprintMismatch;
        },
        .agreed => |m| {
            if (!std.mem.eql(u8, &m.common, &f.fx.common) or !std.mem.eql(u8, &m.ranks[f.rank], &f.fx.rank) or f.rank >= m.world)
                return error.FingerprintMismatch;
            var a: fingerprint.Agreement = .{ .common = m.common, .world = m.world };
            a.ranks = m.ranks;
            f.agreement = a;
            f.pair = a.pair();
            if (comptime need(H, "agreed")) try h.agreed(a);
        },
        .has => |m| {
            if (comptime !need(H, "has")) return error.Unsupported;
            if (!std.mem.eql(u8, &m.compat, &f.pair)) return error.WrongCompat; // rank 0 agreed another pair
            try f.reply(l, .{ .has_reply = try h.has(head, m) });
        },
        .reserve => |m| if (comptime need(H, "reserve")) try h.reserve(head, m) else return error.Unsupported,
        .start => |m| if (comptime need(H, "start")) try h.start(head, m) else return error.Unsupported,
        .progress => |m| if (comptime need(H, "progress")) try h.progress(head, m) else return error.Unsupported,
        .validate => |m| {
            if (comptime !need(H, "validate")) return error.Unsupported;
            try f.reply(l, .{ .validate_reply = try h.validate(head, m) });
        },
        .status => |m| {
            if (comptime !need(H, "status")) return error.Unsupported;
            try f.reply(l, .{ .status_reply = try h.status(head, m) });
        },
        .commit => |m| if (comptime need(H, "commit")) try h.commit(head, m) else return error.Unsupported,
        .abort => |m| if (comptime need(H, "abort")) try h.abort(head, m) else return error.Unsupported,
        .write_staged => |m| if (comptime need(H, "writeStaged")) try h.writeStaged(head, m) else return error.Unsupported,
        .write_cancelled => |m| if (comptime need(H, "writeCancelled")) try h.writeCancelled(head, m) else return error.Unsupported,
        .keep => |m| if (comptime need(H, "keep")) try h.keep(head, m) else return error.Unsupported,
        .share => |m| if (comptime need(H, "share")) try h.share(head, m) else return error.Unsupported,
        .touch => |m| if (comptime need(H, "touch")) try h.touch(head, m) else return error.Unsupported,
        .forget => |m| if (comptime need(H, "forget")) try h.forget(head, m) else return error.Unsupported,
        .shard_layout => |m| {
            if (comptime !need(H, "shardLayout")) return error.Unsupported;
            try f.reply(l, .{ .shard_layout_reply = try h.shardLayout(head, m) });
        },
        .shard_set => |m| if (comptime need(H, "shardSet")) try h.shardSet(head, m) else return error.Unsupported,
        .shard_ops => |m| if (comptime need(H, "shardOps")) try h.shardOps(head, m) else return error.Unsupported,
        .shard_check => |m| {
            if (comptime !need(H, "shardCheck")) return error.Unsupported;
            const r = try h.shardCheck(head, m);
            try f.reply(l, .{ .shard_check_reply = r });
        },
        .shard_xfer => |m| {
            if (comptime !need(H, "shardXfer")) return error.Unsupported;
            try f.reply(l, .{ .shard_xfer_reply = try h.shardXfer(head, m) });
        },
        .shard_finish => |m| if (comptime need(H, "shardFinish")) try h.shardFinish(head, m) else return error.Unsupported,
        .stats => |m| {
            if (comptime !need(H, "stats")) return error.Unsupported;
            try f.reply(l, .{ .stats_reply = try h.stats(head, m) });
        },
        .hello_reply, .has_reply, .validate_reply, .status_reply, .shard_layout_reply, .shard_check_reply, .shard_xfer_reply, .stats_reply => unreachable, // (admit)
    }
}

fn monotonicNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

// -- the schema digest ----------------------------------------------------------------------------------------------

/// A type's wire schema as a 32-bit FNV-1a of its canonical description (field names, wire types, enum values).
pub fn schemaOf(comptime T: type) u32 {
    return comptime blk: {
        @setEvalBranchQuota(200_000);
        var h = std.hash.Fnv1a_32.init();
        feed(&h, T);
        break :blk h.final();
    };
}

fn feed(comptime h: *std.hash.Fnv1a_32, comptime T: type) void {
    switch (@typeInfo(T)) {
        .int => h.update(@typeName(WireInt(T))),
        .bool => h.update("bool"),
        .optional => |o| {
            h.update("?");
            feed(h, o.child);
        },
        .pointer => |p| {
            h.update("[]");
            feed(h, p.child);
        },
        .@"enum" => |e| {
            h.update("enum(");
            feed(h, e.tag_type);
            for (e.field_names, e.field_values) |n, v| h.update(std.fmt.comptimePrint("{s}={d},", .{ n, v }));
            h.update(")");
        },
        .array => |a| {
            h.update(std.fmt.comptimePrint("[{d}]", .{a.len}));
            feed(h, a.child);
        },
        .@"struct" => |s| {
            h.update("struct{");
            for (s.field_names, s.field_types) |n, FT| {
                h.update(n);
                h.update(":");
                feed(h, FT);
                h.update(",");
            }
            h.update("}");
        },
        .@"union" => |u| {
            h.update("union{");
            // the tag's names and numbers (the op byte on the wire is its value: a renumbered op changes the stamp)
            if (u.tag_type) |TT| feed(h, TT);
            for (u.field_names, u.field_types) |n, FT| {
                h.update(n);
                h.update(":");
                feed(h, FT);
                h.update(",");
            }
            h.update("}");
        },
        else => @compileError("rank_wire: no schema for " ++ @typeName(T)),
    }
}

// -- tests ------------------------------------------------------------------------------------------------------------

const testing = std.testing;

fn randDigest(rnd: std.Random) Digest {
    var d: Digest = undefined;
    rnd.bytes(&d);
    return d;
}

fn randEnum(comptime E: type, rnd: std.Random) E {
    const vals = @typeInfo(E).@"enum".field_values;
    const i = rnd.uintLessThan(usize, vals.len);
    inline for (vals, 0..) |v, j| if (i == j) return @enumFromInt(v);
    unreachable;
}

fn randExchange(rnd: std.Random) fingerprint.Exchange {
    var x: fingerprint.Exchange = .{ .common = randDigest(rnd), .rank = randDigest(rnd) };
    @memset(&x.fields, .{ .id = 0, .scope = .common, .digest = @splat(0) });
    x.n = rnd.uintAtMost(u16, fingerprint.max_fields);
    var id: u16 = 0;
    for (x.fields[0..x.n]) |*e| {
        id += 1 + rnd.uintLessThan(u16, 3);
        e.* = .{ .id = id, .scope = randEnum(fingerprint.Scope, rnd), .digest = randDigest(rnd) };
    }
    return x;
}

/// Any value of a plain record type (ints, bools, enums, arrays, structs, unions), uniformly over its bits.
fn randValue(comptime T: type, rnd: std.Random) T {
    switch (@typeInfo(T)) {
        .int => return if (T == usize) rnd.int(u64) else rnd.int(T),
        .bool => return rnd.boolean(),
        .@"enum" => return randEnum(T, rnd),
        .array => |a| {
            var out: T = undefined;
            for (&out) |*x| x.* = randValue(a.child, rnd);
            return out;
        },
        .@"struct" => |s| {
            var out: T = undefined;
            inline for (s.field_names, s.field_types) |n, FT| @field(out, n) = randValue(FT, rnd);
            return out;
        },
        .@"union" => |u| {
            const i = rnd.uintLessThan(usize, u.field_names.len);
            inline for (u.field_names, u.field_types, 0..) |n, FT, j| if (i == j) return @unionInit(T, n, randValue(FT, rnd));
            unreachable;
        },
        else => @compileError("randValue: " ++ @typeName(T)),
    }
}

/// A valid message of each op, its numbers random within what check() allows (a keep's forks in `forks_buf`, its
/// pictures in `media_buf`).
fn randMsg(op: Op, rnd: std.Random, ops_buf: []ShardOp) Msg {
    return randMsgWith(op, rnd, ops_buf, &.{}, &.{});
}

/// A keep's pictures: in order, not overlapping; the cut past the last one's start.
fn randMedia(rnd: std.Random, media_buf: []Media, at: *u32) []const Media {
    const n = rnd.uintAtMost(usize, @min(media_buf.len, max_keep_media));
    var cursor: u32 = 0;
    for (media_buf[0..n]) |*m| {
        m.* = .{ .start = cursor + rnd.uintAtMost(u32, 64), .len = rnd.intRangeAtMost(u32, 1, 600), .digest = randDigest(rnd) };
        cursor = m.start + m.len;
    }
    if (n > 0) at.* = media_buf[n - 1].start + rnd.intRangeAtMost(u32, 1, 1 << 20);
    return media_buf[0..n];
}

fn randMsgWith(op: Op, rnd: std.Random, ops_buf: []ShardOp, forks_buf: []Fork, media_buf: []Media) Msg {
    const world = rnd.intRangeAtMost(u16, 2, max_world);
    const blocks = rnd.intRangeAtMost(u32, 1, max_blocks);
    return switch (op) {
        .hello => .{ .hello = .{ .world = world, .fx = randExchange(rnd) } },
        .hello_reply => blk: {
            const ok = rnd.boolean();
            break :blk .{ .hello_reply = .{ .status = if (ok) .ok else .mismatch, .differ = if (ok) 0 else rnd.intRangeAtMost(u16, 1, 0xffff), .fx = randExchange(rnd) } };
        },
        .agreed => blk: {
            var ranks: [max_world]Digest = @splat(@splat(0));
            for (ranks[0..world]) |*d| d.* = randDigest(rnd);
            break :blk .{ .agreed = .{ .world = world, .common = randDigest(rnd), .ranks = ranks } };
        },
        .has => blk: {
            const layout = randEnum(OwnerLayout, rnd);
            break :blk .{ .has = .{ .point = randDigest(rnd), .compat = randDigest(rnd), .at = rnd.intRangeAtMost(u64, 1, 1 << 40), .blocks = blocks, .layout = layout, .part = if (layout == .replicated) 0 else rnd.uintLessThan(u8, max_parts) } };
        },
        .has_reply => .{ .has_reply = .{ .answer = randEnum(Absence, rnd), .point = randDigest(rnd), .compat = randDigest(rnd) } },
        .reserve => blk: {
            const at = rnd.intRangeAtMost(u64, 1, 1 << 40);
            break :blk .{ .reserve = .{ .point = randDigest(rnd), .kept = rnd.int(u64), .slot = rnd.int(u32), .base = rnd.int(u64), .size = at + rnd.uintAtMost(u64, 1 << 20), .at = at, .blocks = blocks, .generation = rnd.int(u64), .part = rnd.uintLessThan(u8, max_parts) } };
        },
        .start => .{ .start = .{ .point = randDigest(rnd), .blocks = blocks } },
        .progress => .{ .progress = .{ .upto = blocks } },
        .validate => .{ .validate = .{ .upto = blocks, .shared = rnd.boolean() } },
        .validate_reply => blk: {
            const ok = rnd.boolean();
            var why = randEnum(Failure, rnd);
            if (!ok and why == .none) why = .crc;
            break :blk .{ .validate_reply = .{ .ok = ok, .why = if (ok) .none else why, .copied = rnd.uintAtMost(u32, max_blocks), .bad_block = if (ok) no_block else rnd.uintLessThan(u32, max_blocks), .point = randDigest(rnd) } };
        },
        .commit => .{ .commit = .{ .point = randDigest(rnd), .at = rnd.intRangeAtMost(u64, 1, 1 << 40), .kept = rnd.int(u64), .shared = rnd.boolean() } },
        .status => .{ .status = .{} },
        .status_reply => .{ .status_reply = .{ .copied = rnd.uintAtMost(u32, max_blocks), .small_ready = rnd.boolean(), .why = randEnum(Failure, rnd), .point = randDigest(rnd) } },
        .abort => blk: {
            var why = randEnum(Failure, rnd);
            if (why == .none) why = .read;
            break :blk .{ .abort = .{ .why = why, .rank = if (rnd.boolean()) no_rank else rnd.uintLessThan(u16, max_world), .forget = rnd.boolean() } };
        },
        .write_staged => .{ .write_staged = .{ .point = randDigest(rnd), .kept = rnd.int(u64), .upto = rnd.uintAtMost(u32, max_blocks) } },
        .write_cancelled => .{ .write_cancelled = .{ .point = randDigest(rnd), .kept = rnd.int(u64), .why = randEnum(CancelWhy, rnd) } },
        .keep => blk: {
            const n = rnd.uintAtMost(usize, @min(forks_buf.len, max_keep_forks));
            for (forks_buf[0..n]) |*f| f.* = randValue(Fork, rnd);
            var at = rnd.intRangeAtMost(u32, 1, std.math.maxInt(u32));
            const media = randMedia(rnd, media_buf, &at);
            break :blk .{ .keep = .{ .slot = rnd.int(u32), .kept = rnd.intRangeAtMost(u64, 1, std.math.maxInt(u64)), .base = rnd.int(u64), .at = at, .point = randDigest(rnd), .group = rnd.int(u64), .shared = rnd.boolean(), .root = randValue([16]u8, rnd), .part = if (rnd.boolean()) rnd.int(u32) else null, .origin = if (rnd.boolean()) randValue(Origin, rnd) else null, .placeholder = if (media.len > 0) rnd.int(u32) else null, .forks = forks_buf[0..n], .media = media } };
        },
        .share => .{ .share = .{ .point = randDigest(rnd) } },
        .touch => .{ .touch = .{ .point = randDigest(rnd) } },
        .forget => .{ .forget = .{ .point = randDigest(rnd) } },
        .shard_layout => .{ .shard_layout = .{ .version = rnd.intRangeAtMost(u16, 1, 9), .parts = rnd.intRangeAtMost(u8, 1, max_parts), .cap = rnd.int(u64), .carry = rnd.boolean(), .twins = rnd.boolean(), .schema = shard_schema } },
        .shard_layout_reply => .{ .shard_layout_reply = .{ .version = rnd.intRangeAtMost(u16, 1, 9), .parts = rnd.intRangeAtMost(u8, 1, max_parts), .cap = rnd.int(u64), .carry = rnd.boolean(), .twins = rnd.boolean(), .schema = rnd.int(u32) } },
        .shard_set => .{ .shard_set = .{ .version = 1, .parts = rnd.intRangeAtMost(u8, 1, max_parts), .part_rows = rnd.intRangeAtMost(u64, 1, 1 << 40), .owner_epoch = rnd.int(u64) } },
        .shard_ops => blk: {
            const n = rnd.uintAtMost(usize, @min(ops_buf.len, max_shard_ops));
            for (ops_buf[0..n]) |*o| o.* = randValue(ShardOp, rnd);
            break :blk .{ .shard_ops = .{ .owner_epoch = rnd.int(u64), .index = rnd.int(u16), .last = rnd.boolean(), .ops = ops_buf[0..n] } };
        },
        .shard_check => .{ .shard_check = .{ .owner_epoch = rnd.int(u64), .digest = randDigest(rnd) } },
        .shard_check_reply => .{ .shard_check_reply = .{ .owner_epoch = rnd.int(u64), .digest = randDigest(rnd) } },
        .shard_xfer => .{ .shard_xfer = .{ .owner_epoch = rnd.int(u64), .stream = rnd.int(u64) } },
        .shard_xfer_reply => .{ .shard_xfer_reply = .{ .owner_epoch = rnd.int(u64), .stream = rnd.int(u64), .ok = rnd.boolean() } },
        .shard_finish => .{ .shard_finish = .{ .owner_epoch = rnd.int(u64), .stream = rnd.int(u64), .ok = rnd.boolean() } },
        .stats => .{ .stats = .{ .set = randEnum(StatsSet, rnd) } },
        .stats_reply => blk: {
            var v: [max_stats]u64 = @splat(0);
            const n = rnd.uintAtMost(u8, max_stats);
            for (v[0..n]) |*x| x.* = rnd.int(u64);
            break :blk .{ .stats_reply = .{ .set = randEnum(StatsSet, rnd), .n = n, .values = v } };
        },
    };
}

fn randHead(op: Op, rnd: std.Random) Head {
    return .{ .op = op, .from = rnd.int(u16), .epoch = rnd.int(u64), .txn = rnd.int(u64), .seq = rnd.int(u64) };
}

const all_ops = blk: {
    const vals = @typeInfo(Op).@"enum".field_values;
    var out: [vals.len]Op = undefined;
    for (vals, 0..) |v, i| out[i] = @enumFromInt(v);
    break :blk out;
};

test "every message round-trips, and nothing but its exact bytes decodes" {
    var prng = std.Random.DefaultPrng.init(0x5257_0001);
    const rnd = prng.random();
    const gpa = testing.allocator;
    const src_ops = try gpa.alloc(ShardOp, max_shard_ops);
    defer gpa.free(src_ops);
    const dst_ops = try gpa.alloc(ShardOp, max_shard_ops);
    defer gpa.free(dst_ops);
    const src_forks = try gpa.alloc(Fork, max_keep_forks);
    defer gpa.free(src_forks);
    const dst_forks = try gpa.alloc(Fork, max_keep_forks);
    defer gpa.free(dst_forks);
    const src_media = try gpa.alloc(Media, max_keep_media);
    defer gpa.free(src_media);
    const dst_media = try gpa.alloc(Media, max_keep_media);
    defer gpa.free(dst_media);
    const buf = try gpa.alloc(u8, max_frame + 64);
    defer gpa.free(buf);
    const again = try gpa.alloc(u8, max_frame);
    defer gpa.free(again);
    const bufs: Bufs = .{ .ops = dst_ops, .forks = dst_forks, .media = dst_media };
    for (0..40) |_| for (all_ops) |op| {
        const head = randHead(op, rnd);
        const msg = randMsgWith(op, rnd, src_ops, src_forks, src_media);
        const bytes = try encode(buf, head, msg);
        const f = try decodeInto(bytes, bufs);
        try testing.expectEqualDeep(head, f.head);
        try testing.expectEqualDeep(msg, f.msg);
        // canonical: the decoded frame encodes to the same bytes
        try testing.expectEqualSlices(u8, bytes, try encode(again, f.head, f.msg));
        // every truncation fails, one more byte fails
        var cut: usize = 0;
        while (cut < bytes.len) : (cut += 1 + rnd.uintLessThan(usize, 1 + bytes.len / 64)) {
            if (decodeInto(bytes[0..cut], bufs)) |_| return error.TruncatedFrameDecoded else |_| {}
        }
        buf[bytes.len] = 0;
        try testing.expectError(error.TrailingBytes, decodeInto(buf[0 .. bytes.len + 1], bufs));
    };
}

test "garbled frames never decode to something else, random bytes never crash the parser" {
    var prng = std.Random.DefaultPrng.init(0x5257_0002);
    const rnd = prng.random();
    const gpa = testing.allocator;
    var src_ops: [64]ShardOp = undefined;
    var dst_ops: [64]ShardOp = undefined;
    var src_forks: [64]Fork = undefined;
    var dst_forks: [64]Fork = undefined;
    var src_media: [32]Media = undefined;
    var dst_media: [32]Media = undefined;
    const bufs: Bufs = .{ .ops = &dst_ops, .forks = &dst_forks, .media = &dst_media };
    const buf = try gpa.alloc(u8, max_frame);
    defer gpa.free(buf);
    const again = try gpa.alloc(u8, max_frame);
    defer gpa.free(again);
    var decoded: usize = 0;
    var refused: usize = 0;
    for (0..3000) |_| {
        const op = all_ops[rnd.uintLessThan(usize, all_ops.len)];
        const bytes = try encode(buf, randHead(op, rnd), randMsgWith(op, rnd, &src_ops, &src_forks, &src_media));
        // flip 1-3 random bytes: either refused, or a frame whose canonical bytes are exactly the garbled ones
        const flips = 1 + rnd.uintLessThan(usize, 3);
        for (0..flips) |_| bytes[rnd.uintLessThan(usize, bytes.len)] ^= @as(u8, 1) << rnd.int(u3);
        if (decodeInto(bytes, bufs)) |f| {
            decoded += 1;
            try testing.expectEqualSlices(u8, bytes, try encode(again, f.head, f.msg));
        } else |_| refused += 1;
    }
    try testing.expect(refused > 0 and decoded > 0);
    // pure noise, including plausible headers
    for (0..20000) |i| {
        const n = rnd.uintAtMost(usize, 4096);
        rnd.bytes(buf[0..n]);
        if (i % 2 == 0 and n >= header_len) {
            std.mem.writeInt(u16, buf[0..2], magic, .little);
            std.mem.writeInt(u16, buf[2..4], version, .little);
            buf[4] = @intFromEnum(all_ops[rnd.uintLessThan(usize, all_ops.len)]);
            buf[5] = 0;
        }
        if (decodeInto(buf[0..n], bufs)) |f| {
            try testing.expectEqualSlices(u8, buf[0..n], try encode(again, f.head, f.msg));
        } else |_| {}
    }
}

test "frames past their bound are refused both ways" {
    const gpa = testing.allocator;
    const big = try gpa.alloc(u8, max_frame + 1);
    defer gpa.free(big);
    @memset(big, 0);
    try testing.expectError(error.FrameTooLong, decode(big, &.{}));
    // a reply longer than max_reply
    var buf: [max_reply]u8 = undefined;
    const ok = try encode(&buf, .{ .op = .has_reply, .from = 1, .epoch = 1, .txn = 1, .seq = 1 }, .{ .has_reply = .{ .answer = .present, .point = @splat(1), .compat = @splat(2) } });
    const long = try gpa.alloc(u8, max_reply + 1);
    defer gpa.free(long);
    @memcpy(long[0..ok.len], ok);
    @memset(long[ok.len..], 0);
    try testing.expectError(error.FrameTooLong, decode(long, &.{}));
    // more ops than a frame carries
    const ops = try gpa.alloc(ShardOp, max_shard_ops + 1);
    defer gpa.free(ops);
    for (ops) |*o| o.* = .{ .drop = .{ .kept = 1, .gen = 1 } };
    const fbuf = try gpa.alloc(u8, max_frame);
    defer gpa.free(fbuf);
    const head: Head = .{ .op = .shard_ops, .from = 0, .epoch = 1, .txn = 1, .seq = 2 };
    try testing.expectError(error.BadCount, encode(fbuf, head, .{ .shard_ops = .{ .owner_epoch = 1, .index = 0, .last = true, .ops = ops } }));
    const full = try encode(fbuf, head, .{ .shard_ops = .{ .owner_epoch = 1, .index = 0, .last = true, .ops = ops[0..max_shard_ops] } });
    try testing.expectError(error.BadCount, decode(full, ops[0..10])); // a receiver with less room refuses, never overruns
    // bad values the schema forbids
    var b2: [256]u8 = undefined;
    const v = try encode(&b2, .{ .op = .validate, .from = 0, .epoch = 1, .txn = 1, .seq = 1 }, .{ .validate = .{ .upto = 3, .shared = true } });
    v[v.len - 1] = 2; // a bool that is neither 0 nor 1
    try testing.expectError(error.BadValue, decode(v, &.{}));
    try testing.expectError(error.BadValue, encode(&b2, .{ .op = .abort, .from = 0, .epoch = 1, .txn = 1, .seq = 1 }, .{ .abort = .{ .why = .none, .rank = 1, .forget = false } }));
    try testing.expectError(error.OpMismatch, encode(&b2, head, .{ .progress = .{ .upto = 1 } }));
}

test "a shard op's schema follows its type" {
    const A = union(enum) { x: struct { a: u64 } };
    const B = union(enum) { x: struct { a: u32 } };
    const C = union(enum) { x: struct { b: u64 } };
    const D = union(enum) { x: struct { a: usize } };
    try testing.expect(schemaOf(A) != schemaOf(B));
    try testing.expect(schemaOf(A) != schemaOf(C));
    try testing.expectEqual(schemaOf(A), schemaOf(D)); // usize is 64 bits on the wire on every host
    const E1 = union(enum) { x: struct { r: enum(u2) { a, b } } };
    const E2 = union(enum) { x: struct { r: enum(u2) { a, c } } };
    try testing.expect(schemaOf(E1) != schemaOf(E2));
    // a frame with another schema is refused
    var buf: [256]u8 = undefined;
    const bytes = try encode(&buf, .{ .op = .shard_ops, .from = 0, .epoch = 1, .txn = 1, .seq = 1 }, .{ .shard_ops = .{ .owner_epoch = 1, .index = 0, .last = true, .ops = &.{.{ .drop = .{ .kept = 5, .gen = 2 } }} } });
    std.mem.writeInt(u32, bytes[header_len + 11 ..][0..4], shard_schema +% 1, .little);
    var ops: [4]ShardOp = undefined;
    try testing.expectError(error.SchemaMismatch, decode(bytes, &ops));
}

/// A command frame as the tracker sees it.
fn cmd(t: *const Tracker, op_msg: Msg, txn: u64) Frame {
    return .{ .head = .{ .op = std.meta.activeTag(op_msg), .from = 0, .epoch = t.epoch, .txn = txn, .seq = t.next_seq }, .msg = op_msg };
}

fn testFx(seed: u8) fingerprint.Exchange {
    var b: fingerprint.Builder = .{};
    for (fingerprint.all) |fl| b.int(fl, u8, seed);
    const fp = b.finish(null) catch unreachable;
    return fp.exchange();
}

fn agreedTracker() !Tracker {
    var t: Tracker = .{};
    const fx = testFx(1);
    var hello = cmd(&t, .{ .hello = .{ .world = 2, .fx = fx } }, 0);
    hello.head.epoch = 77;
    try t.admit(&hello);
    try t.admit(&cmd(&t, .{ .agreed = .{ .world = 2, .common = fx.common, .ranks = .{ fx.rank, fx.rank, @splat(0), @splat(0) } } }, 0));
    return t;
}

test "the tracker admits a restore in order and refuses it out of order" {
    const p: Digest = @splat(9);
    var t = try agreedTracker();
    // nothing before hello, nothing but hello first
    {
        var fresh: Tracker = .{};
        try testing.expectError(error.NotAgreed, fresh.admit(&cmd(&fresh, .{ .progress = .{ .upto = 1 } }, 1)));
        var h = cmd(&fresh, .{ .hello = .{ .world = 2, .fx = testFx(1) } }, 0);
        try testing.expectError(error.StaleEpoch, fresh.admit(&h)); // epoch 0
        h.head.epoch = 5;
        try fresh.admit(&h);
        try testing.expectError(error.NotAgreed, fresh.admit(&cmd(&fresh, .{ .has = .{ .point = p, .compat = p, .at = 4096, .blocks = 2, .layout = .replicated, .part = 0 } }, 1)));
    }
    const has: Msg = .{ .has = .{ .point = p, .compat = p, .at = 6144, .blocks = 3, .layout = .replicated, .part = 0 } };
    // a skipped txn id, a repeated one
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, has, 2)));
    try t.admit(&cmd(&t, has, 1));
    try testing.expectError(error.StaleTxn, t.admit(&cmd(&t, has, 1)));
    // a stale epoch, a skipped seq, a repeated seq
    {
        var f = cmd(&t, .{ .reserve = .{ .point = p, .kept = 1, .slot = 0, .base = 0, .size = 8192, .at = 6144, .blocks = 3, .generation = 0, .part = 0 } }, 1);
        f.head.epoch = 76;
        try testing.expectError(error.StaleEpoch, t.admit(&f));
        f.head.epoch = t.epoch;
        f.head.seq += 1;
        try testing.expectError(error.OutOfOrder, t.admit(&f));
        f.head.seq -= 2;
        try testing.expectError(error.OutOfOrder, t.admit(&f));
        f.head.seq += 1;
        f.head.from = 1;
        try testing.expectError(error.WrongRank, t.admit(&f));
    }
    // phases out of order
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .start = .{ .point = p, .blocks = 3 } }, 1)));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .validate = .{ .upto = 3, .shared = false } }, 1)));
    try testing.expectError(error.UnknownTxn, t.admit(&cmd(&t, .{ .start = .{ .point = p, .blocks = 3 } }, 9)));
    try testing.expectError(error.WrongPoint, t.admit(&cmd(&t, .{ .reserve = .{ .point = @splat(8), .kept = 1, .slot = 0, .base = 0, .size = 8192, .at = 6144, .blocks = 3, .generation = 0, .part = 0 } }, 1)));
    try t.admit(&cmd(&t, .{ .reserve = .{ .point = p, .kept = 1, .slot = 0, .base = 0, .size = 8192, .at = 6144, .blocks = 3, .generation = 0, .part = 0 } }, 1));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .progress = .{ .upto = 1 } }, 1))); // before start
    try t.admit(&cmd(&t, .{ .start = .{ .point = p, .blocks = 3 } }, 1));
    try t.admit(&cmd(&t, .{ .progress = .{ .upto = 1 } }, 1));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .progress = .{ .upto = 1 } }, 1))); // not forward
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .progress = .{ .upto = 4 } }, 1))); // past the point
    try t.admit(&cmd(&t, .{ .progress = .{ .upto = 2 } }, 1));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .commit = .{ .point = p, .at = 6144, .kept = 1, .shared = false } }, 1)));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .validate = .{ .upto = 2, .shared = false } }, 1))); // not through the end
    try t.admit(&cmd(&t, .{ .validate = .{ .upto = 3, .shared = true } }, 1));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .progress = .{ .upto = 3 } }, 1)));
    try t.admit(&cmd(&t, .{ .commit = .{ .point = p, .at = 6144, .kept = 1, .shared = true } }, 1));
    try testing.expectError(error.UnknownTxn, t.admit(&cmd(&t, .{ .abort = .{ .why = .read, .rank = 1, .forget = true } }, 1))); // closed
    try testing.expectEqual(@as(usize, 0), t.open());
    // abort from any phase; at most max_restores open
    for (0..max_restores) |i| try t.admit(&cmd(&t, has, 2 + i));
    try testing.expectError(error.TooManyRestores, t.admit(&cmd(&t, has, 2 + max_restores)));
    try t.admit(&cmd(&t, .{ .abort = .{ .why = .presence, .rank = 1, .forget = false } }, 5));
    try t.admit(&cmd(&t, has, 2 + max_restores));
    try testing.expectEqual(@as(usize, max_restores), t.open());
    // replies are never commands
    var r = cmd(&t, .{ .has_reply = .{ .answer = .present, .point = p, .compat = p } }, 1);
    try testing.expectError(error.UnexpectedReply, t.admit(&r));
    r = cmd(&t, .{ .hello = .{ .world = 2, .fx = testFx(1) } }, 0);
    try testing.expectError(error.OutOfOrder, t.admit(&r)); // hello once
}

test "the tracker keeps a plan's frames together and owner epochs forward" {
    var t = try agreedTracker();
    const ops = [_]ShardOp{.{ .drop = .{ .kept = 1, .gen = 1 } }};
    try testing.expectError(error.NotAgreed, t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 1, .index = 0, .last = true, .ops = &ops } }, 1)));
    try t.admit(&cmd(&t, .{ .shard_layout = .{ .version = 1, .parts = 2, .cap = 1 << 20, .carry = true, .twins = true, .schema = shard_schema } }, 1));
    try t.admit(&cmd(&t, .{ .shard_set = .{ .version = 1, .parts = 2, .part_rows = 1 << 19, .owner_epoch = 10 } }, 2));
    try testing.expectError(error.StaleEpoch, t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 10, .index = 0, .last = true, .ops = &ops } }, 3)));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 12, .index = 0, .last = true, .ops = &ops } }, 3)));
    try t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 11, .index = 0, .last = false, .ops = &ops } }, 3));
    // nothing between a plan's frames
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .write_cancelled = .{ .point = @splat(1), .kept = 1, .why = .carry } }, 4)));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 11, .index = 2, .last = true, .ops = &ops } }, 3)));
    try testing.expectError(error.StaleEpoch, t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 12, .index = 1, .last = false, .ops = &ops } }, 3)));
    try t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 11, .index = 1, .last = false, .ops = &ops } }, 3));
    try t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 11, .index = 2, .last = true, .ops = &ops } }, 3));
    try testing.expectError(error.StaleEpoch, t.admit(&cmd(&t, .{ .shard_check = .{ .owner_epoch = 10, .digest = @splat(0) } }, 4)));
    try t.admit(&cmd(&t, .{ .shard_check = .{ .owner_epoch = 11, .digest = @splat(0) } }, 4));
    try t.admit(&cmd(&t, .{ .write_staged = .{ .point = @splat(1), .kept = 3, .upto = 1 } }, 5));
    try testing.expectError(error.StaleEpoch, t.admit(&cmd(&t, .{ .shard_set = .{ .version = 1, .parts = 2, .part_rows = 1 << 19, .owner_epoch = 11 } }, 6)));
}

// -- the reply channel over a loopback link --------------------------------------------------------------------------

const TestSink = struct {
    l: link.Link,
    fn send(ctx: *anyopaque, sub: []const u8) anyerror!void {
        const s: *TestSink = @ptrCast(@alignCast(ctx));
        try s.l.send(sub);
    }
    fn sink(s: *TestSink) Sink {
        return .{ .ctx = s, .send = send };
    }
};

/// A follower's spill side for the tests: holds `held`, fails validation of block `bad` when set.
const FakeSide = struct {
    held: Digest = @splat(9),
    bad: ?u32 = null,
    copied: u32 = 0,
    committed: u32 = 0,
    aborted: u32 = 0,
    silent: bool = false, // has: never answers (a stuck rank)
    garble: bool = false,
    has_err: bool = false, // has: the handler fails (the rank ends without answering)
    agreed_seen: bool = false,
    store_compat: Digest = @splat(0), // the store this rank opened at agreed: storeCommon(rank)
    written_under: ?Digest = null, // the compat the held point's manifest records (null: this store's own)
    // the KV-ownership side: a real parts.Planner, fed only the ops rank 0 sends
    pl: ?parts.Planner = null,
    cap: u64 = 32 * parts.chunk_rows,
    xfer_fail: bool = false,
    finished: u32 = 0,
    // the restore's copies as status reports them: complete once the clock passes `ready_at` (a slow disk)
    ready_at: u64 = 0,
    status_why: Failure = .none,
    statuses: u32 = 0,
    validates: u32 = 0,
    blocks: u32 = 0,
    // the notices
    keeps: u32 = 0,
    keep_forks: u32 = 0,
    keep_origin: bool = false,
    shares: u32 = 0,
    touches: u32 = 0,
    forgets: u32 = 0,
    // PETABRIDGE (k2a-pb1): the world-3 tests' (P3): a has answer to give instead of the held/compat check, and what
    // each abort carried (how many said forget, the last one's rank)
    answer: ?Absence = null,
    abort_forgets: u32 = 0,
    abort_rank: u16 = 0,

    rank: u16 = 1,

    pub fn agreed(s: *FakeSide, a: fingerprint.Agreement) !void {
        s.agreed_seen = true;
        s.store_compat = a.storeCommon(s.rank);
    }
    /// As round 2's handler: m.compat (the pair) was checked by serve; presence is this rank's own store's, under
    /// its own compat; point and compat echo.
    pub fn has(s: *FakeSide, _: Head, m: Has) !HasReply {
        if (s.has_err) return error.DiskGone;
        if (s.answer) |a| return .{ .answer = a, .point = m.point, .compat = m.compat }; // PETABRIDGE (k2a-pb1): P3 tests
        const answer: Absence = if (!std.mem.eql(u8, &m.point, &s.held))
            .missing
        else if (!std.mem.eql(u8, &(s.written_under orelse s.store_compat), &s.store_compat))
            .incompatible
        else
            .present;
        return .{ .answer = answer, .point = m.point, .compat = m.compat };
    }
    pub fn reserve(s: *FakeSide, _: Head, m: Reserve) !void {
        s.blocks = m.blocks;
    }
    /// From memory, at once: how far the "disk" got by now.
    pub fn status(s: *FakeSide, _: Head, _: RestoreStatus) !RestoreStatusReply {
        s.statuses += 1;
        const done = monotonicNs() >= s.ready_at;
        return .{ .copied = if (done) s.blocks else s.blocks / 2, .small_ready = done, .why = s.status_why, .point = s.held };
    }
    pub fn keep(s: *FakeSide, _: Head, m: Keep) !void {
        s.keeps += 1;
        s.keep_forks = @intCast(m.forks.len);
        s.keep_origin = m.origin != null;
    }
    pub fn share(s: *FakeSide, _: Head, _: Share) !void {
        s.shares += 1;
    }
    pub fn touch(s: *FakeSide, _: Head, _: Touch) !void {
        s.touches += 1;
    }
    pub fn forget(s: *FakeSide, _: Head, _: Forget) !void {
        s.forgets += 1;
    }
    pub fn start(_: *FakeSide, _: Head, _: Start) !void {}
    pub fn progress(s: *FakeSide, _: Head, m: Progress) !void {
        s.copied = m.upto;
    }
    pub fn validate(s: *FakeSide, _: Head, m: Validate) !ValidateReply {
        s.validates += 1;
        s.copied = m.upto;
        if (s.bad) |b| return .{ .ok = false, .why = .crc, .copied = b, .bad_block = b, .point = s.held };
        return .{ .ok = true, .why = .none, .copied = m.upto, .bad_block = no_block, .point = s.held };
    }
    pub fn commit(s: *FakeSide, _: Head, _: Commit) !void {
        s.committed += 1;
    }
    pub fn abort(s: *FakeSide, _: Head, m: Abort) !void { // PETABRIDGE (k2a-pb1): `m` read for the P3 tests
        s.aborted += 1;
        if (m.forget) s.abort_forgets += 1; // PETABRIDGE (k2a-pb1): P3 tests
        s.abort_rank = m.rank; // PETABRIDGE (k2a-pb1): P3 tests
    }

    pub fn shardLayout(s: *FakeSide, _: Head, m: ShardLayout) !ShardLayoutReply {
        return .{ .version = parts.layout_version, .parts = m.parts, .cap = s.cap, .carry = m.carry, .twins = m.twins, .schema = shard_schema };
    }
    pub fn shardSet(s: *FakeSide, _: Head, m: ShardSet) !void {
        if (m.version != parts.layout_version) return error.LayoutVersion;
        if (s.pl) |*x| x.deinit(testing.allocator);
        s.pl = try parts.Planner.init(.{ .cap = @intCast(m.part_rows * m.parts), .parts = m.parts, .carry_bytes = 720 });
    }
    pub fn shardOps(s: *FakeSide, _: Head, m: ShardOps) !void {
        try s.pl.?.applyOps(testing.allocator, m.ops);
    }
    pub fn shardXfer(s: *FakeSide, _: Head, m: ShardXfer) !ShardXferReply {
        return .{ .owner_epoch = m.owner_epoch, .stream = m.stream, .ok = !s.xfer_fail };
    }
    pub fn shardFinish(s: *FakeSide, _: Head, m: ShardFinish) !void {
        _ = try s.pl.?.finish(m.stream, m.ok);
        s.finished += 1;
    }
    pub fn shardCheck(s: *FakeSide, _: Head, m: ShardCheck) !ShardCheckReply {
        return .{ .owner_epoch = m.owner_epoch, .digest = s.pl.?.digest() };
    }
    pub fn stats(s: *FakeSide, _: Head, m: Stats) !StatsReply {
        var v: [max_stats]u64 = @splat(0);
        v[0] = s.committed;
        v[1] = s.aborted;
        v[2] = s.finished;
        return .{ .set = m.set, .n = 3, .values = v };
    }
};

/// A hand-built wrong reply to has (rank 0's collect must refuse each).
const BadReply = enum { epoch, seq, txn, from, op, point };

const Peer = struct {
    port: u16,
    fx: fingerprint.Exchange,
    rank: u16 = 1,
    side: FakeSide = .{},
    err: ?anyerror = null,
    frames: u32 = 0,
    bad: ?BadReply = null,
    delays_ms: [4]u32 = @splat(0), // the i-th has waits this long before it is served
    hases: u32 = 0,

    fn run(p: *Peer) void {
        var l = link.Link.connect(testing.io, .{ 127, 0, 0, 1 }, p.port, 10) catch |e| {
            p.err = e;
            return;
        };
        defer l.close();
        p.side.rank = p.rank;
        var f = Follower.init(p.rank, p.fx);
        var buf: [max_frame]u8 = undefined;
        while (true) {
            const bytes = l.recv(&buf) catch |e| {
                p.err = e;
                return;
            };
            if (bytes.len == 0) return; // the test's "done"
            p.frames += 1;
            const is_has = bytes.len > 4 and bytes[4] == @intFromEnum(Op.has);
            if (is_has) {
                if (p.hases < p.delays_ms.len and p.delays_ms[p.hases] > 0)
                    std.Io.sleep(testing.io, .fromMilliseconds(p.delays_ms[p.hases]), .awake) catch {};
                p.hases += 1;
            }
            if (p.bad != null and is_has) {
                p.sendBad(l, bytes) catch |e| {
                    p.err = e;
                };
                return;
            }
            if (p.side.silent and bytes.len > 4 and bytes[4] == @intFromEnum(Op.has)) {
                // a stuck rank: takes the ask, answers nothing, holds the link open
                std.Io.sleep(testing.io, .fromMilliseconds(600), .awake) catch {};
                return;
            }
            if (p.side.garble and bytes.len > 4 and bytes[4] == @intFromEnum(Op.has)) {
                l.send("XX garbage") catch {};
                return;
            }
            serve(&f, l, bytes, &p.side) catch |e| {
                p.err = e;
                return;
            };
        }
    }

    fn sendBad(p: *Peer, l: link.Link, bytes: []const u8) !void {
        const fr = try decode(bytes, &.{});
        var head = fr.head;
        head.op = .has_reply;
        head.from = p.rank;
        var msg: Msg = .{ .has_reply = .{ .answer = .present, .point = fr.msg.has.point, .compat = fr.msg.has.compat } };
        switch (p.bad.?) {
            .epoch => head.epoch +%= 1,
            .seq => head.seq += 1,
            .txn => head.txn += 1,
            .from => head.from += 1,
            .op => {
                head.op = .stats_reply;
                msg = .{ .stats_reply = .{ .set = .tier, .n = 0, .values = @splat(0) } };
            },
            .point => msg.has_reply.point[0] ^= 1,
        }
        var rb: [max_reply]u8 = undefined;
        try l.send(try encode(&rb, head, msg));
    }
};

fn testPort(io: std.Io) u16 {
    var b: [2]u8 = undefined;
    io.random(&b);
    return 30000 + std.mem.readInt(u16, &b, .little) % 20000;
}

test "rank 0 and a follower agree, restore, and fail closed over a loopback link" {
    const io = testing.io;
    const gpa = testing.allocator;
    // 1. matching fingerprints: agreement, a present point, a restore committed
    {
        const port = testPort(io);
        const peer = try gpa.create(Peer);
        defer gpa.destroy(peer);
        peer.* = .{ .port = port, .fx = testFx(1) };
        const th = try std.Thread.spawn(.{}, Peer.run, .{peer});
        var l = try link.Link.listen(.{ 127, 0, 0, 1 }, port);
        defer l.close();
        var ts: TestSink = .{ .l = l };
        const leader = try gpa.create(Leader);
        defer gpa.destroy(leader);
        leader.* = Leader.init(2, freshEpoch(io), testFx(1));
        const links = [_]?link.Link{l};
        const a = try leader.agree(ts.sink(), &links);
        try testing.expectEqualSlices(u8, &testFx(1).common, &a.common);
        var replies: [1]Frame = undefined;
        const p: Digest = @splat(9);
        try testing.expectEqualSlices(u8, &a.pair(), &leader.compat());
        const has = try leader.ask(ts.sink(), &links, 0, .{ .has = .{ .point = p, .compat = leader.compat(), .at = 6144, .blocks = 3, .layout = .replicated, .part = 0 } }, &replies);
        try testing.expect(allPresent(&replies));
        try testing.expectEqualSlices(u8, &a.pair(), &replies[0].msg.has_reply.compat);
        // no command while an ask is outstanding, no second ask
        _ = try leader.post(ts.sink(), has.txn, .{ .reserve = .{ .point = p, .kept = 4, .slot = 1, .base = 0, .size = 8192, .at = 6144, .blocks = 3, .generation = 0, .part = 0 } });
        _ = try leader.post(ts.sink(), has.txn, .{ .start = .{ .point = p, .blocks = 3 } });
        _ = try leader.post(ts.sink(), has.txn, .{ .progress = .{ .upto = 2 } });
        try testing.expectError(error.OutOfOrder, leader.admissible(has.txn, .{ .progress = .{ .upto = 2 } })); // checked, nothing sent
        _ = try leader.ask(ts.sink(), &links, has.txn, .{ .validate = .{ .upto = 3, .shared = false } }, &replies);
        try testing.expect(firstFailure(&replies) == null);
        _ = try leader.post(ts.sink(), has.txn, .{ .commit = .{ .point = p, .at = 6144, .kept = 4, .shared = false } });
        // an absent point: has says missing, rank 0 aborts the transaction everywhere
        const q: Digest = @splat(3);
        const h2 = try leader.ask(ts.sink(), &links, 0, .{ .has = .{ .point = q, .compat = leader.compat(), .at = 2048, .blocks = 1, .layout = .replicated, .part = 0 } }, &replies);
        try testing.expect(!allPresent(&replies));
        try testing.expectEqual(Absence.missing, replies[0].msg.has_reply.answer);
        _ = try leader.post(ts.sink(), h2.txn, .{ .abort = .{ .why = .presence, .rank = 1, .forget = false } });
        // a has whose compat is not the agreed pair is refused before it is sent (an ask: the pair goes on)
        try testing.expectError(error.WrongCompat, leader.ask(ts.sink(), &links, 0, .{ .has = .{ .point = p, .compat = p, .at = 2048, .blocks = 1, .layout = .replicated, .part = 0 } }, &replies));
        try testing.expect(leader.broken == null);
        // a mis-sized ask is refused before anything is sent (no ask left outstanding at the follower)
        try testing.expectError(error.BadWorld, leader.ask(ts.sink(), &links, 0, .{ .stats = .{ .set = .tier } }, replies[0..0]));
        try testing.expect(leader.asked == null and leader.broken == null);
        // the point is held, but written under another store compat: this rank's own check says incompatible
        peer.side.written_under = @splat(0x44);
        const h3 = try leader.ask(ts.sink(), &links, 0, .{ .has = .{ .point = p, .compat = leader.compat(), .at = 2048, .blocks = 1, .layout = .replicated, .part = 0 } }, &replies);
        try testing.expectEqual(Absence.incompatible, replies[0].msg.has_reply.answer);
        _ = try leader.post(ts.sink(), h3.txn, .{ .abort = .{ .why = .presence, .rank = 1, .forget = false } });
        try l.send(&.{}); // done
        th.join();
        try testing.expectEqual(@as(?anyerror, null), peer.err);
        try testing.expect(peer.side.agreed_seen);
        try testing.expectEqual(@as(u32, 1), peer.side.committed);
        try testing.expectEqual(@as(u32, 2), peer.side.aborted); // the missing point, the incompatible one
        try testing.expectEqual(@as(u32, 3), peer.side.copied);
    }
    // 2. a differing fingerprint: both sides refuse, naming the field
    {
        const port = testPort(io);
        const peer = try gpa.create(Peer);
        defer gpa.destroy(peer);
        var other = testFx(1);
        other.fields[2].digest[0] ^= 1; // engine_commit (id 3)
        other.common[0] ^= 1;
        peer.* = .{ .port = port, .fx = other };
        const th = try std.Thread.spawn(.{}, Peer.run, .{peer});
        var l = try link.Link.listen(.{ 127, 0, 0, 1 }, port);
        defer l.close();
        var ts: TestSink = .{ .l = l };
        const leader = try gpa.create(Leader);
        defer gpa.destroy(leader);
        leader.* = Leader.init(2, 5, testFx(1));
        try testing.expectError(error.FingerprintMismatch, leader.agree(ts.sink(), &[_]?link.Link{l}));
        th.join();
        try testing.expectEqual(@as(?anyerror, error.FingerprintMismatch), peer.err);
        var buf: [128]u8 = undefined;
        try testing.expectEqualStrings("rank 1's configuration differs from rank 0's in engine_commit", describeBroken(leader.broken.?, &buf));
        try testing.expectError(error.PairBroken, leader.post(ts.sink(), 0, .{ .write_staged = .{ .point = @splat(0), .kept = 1, .upto = 0 } }));
    }
    // 3. a follower that never answers: rank 0's wait is bounded and ends the pair
    // 4. a garbled reply ends the pair
    for ([_]bool{ true, false }) |silent| {
        const port = testPort(io);
        const peer = try gpa.create(Peer);
        defer gpa.destroy(peer);
        peer.* = .{ .port = port, .fx = testFx(1) };
        peer.side.silent = silent;
        peer.side.garble = !silent;
        const th = try std.Thread.spawn(.{}, Peer.run, .{peer});
        var l = try link.Link.listen(.{ 127, 0, 0, 1 }, port);
        defer l.close();
        var ts: TestSink = .{ .l = l };
        const leader = try gpa.create(Leader);
        defer gpa.destroy(leader);
        leader.* = Leader.init(2, 6, testFx(1));
        leader.timeouts.has_ns = 150 * std.time.ns_per_ms;
        _ = try leader.agree(ts.sink(), &[_]?link.Link{l});
        var replies: [1]Frame = undefined;
        const p: Digest = @splat(9);
        const t0 = monotonicNs();
        const r = leader.ask(ts.sink(), &[_]?link.Link{l}, 0, .{ .has = .{ .point = p, .compat = leader.compat(), .at = 2048, .blocks = 1, .layout = .replicated, .part = 0 } }, &replies);
        try testing.expectError(error.PairBroken, r);
        if (silent) {
            try testing.expectEqual(@as(anyerror, error.ReplyTimeout), leader.broken.?.err);
            try testing.expect(monotonicNs() - t0 < 2 * std.time.ns_per_s);
        } else {
            try testing.expectEqual(@as(anyerror, error.BadMagic), leader.broken.?.err);
        }
        try testing.expectEqual(@as(u16, 1), leader.broken.?.rank);
        th.join();
    }
}

test "every refusal a start can meet is explained, and nothing else" {
    for ([_]anyerror{ error.FingerprintMismatch, error.PairBroken, error.NoEngineId, error.NoRecipePatches, error.NoWeightsId, error.NoEngramId }) |e| try testing.expect(explain(e) != null);
    // generic names other code returns too (link's PeerDown, a Timeout, ordering errors) are never taken as rank_wire's
    for ([_]anyerror{ error.OutOfMemory, error.PeerDown, error.Timeout, error.ReplyTimeout, error.OutOfOrder, error.StaleEpoch, error.BadMagic, error.BadVersion }) |e| try testing.expect(explain(e) == null);
}

test "a follower answers each ask once and refuses a reply that does not fit the ask" {
    var f = Follower.init(1, testFx(1));
    var buf: [max_frame]u8 = undefined;
    const fx = testFx(1);
    const hello = try encode(&buf, .{ .op = .hello, .from = 0, .epoch = 3, .txn = 0, .seq = 0 }, .{ .hello = .{ .world = 2, .fx = fx } });
    _ = try f.accept(hello);
    const agreed = try encode(&buf, .{ .op = .agreed, .from = 0, .epoch = 3, .txn = 0, .seq = 1 }, .{ .agreed = .{ .world = 2, .common = fx.common, .ranks = .{ fx.rank, fx.rank, @splat(0), @splat(0) } } });
    try testing.expectError(error.ReplyOwed, f.accept(agreed)); // hello is still owed its answer
    var g = Follower.init(1, fx);
    try testing.expectError(error.NothingAsked, g.reply(undefined, .{ .has_reply = .{ .answer = .present, .point = @splat(0), .compat = @splat(0) } }));
    try testing.expectError(error.OpMismatch, f.reply(undefined, .{ .has_reply = .{ .answer = .present, .point = @splat(0), .compat = @splat(0) } }));
}

test "the tracker opens a transfer at its plan and closes it at its finish, atomically" {
    var t = try agreedTracker();
    try t.admit(&cmd(&t, .{ .shard_set = .{ .version = 1, .parts = 2, .part_rows = 1 << 19, .owner_epoch = 10 } }, 1));
    const cp: ShardOp = .{ .copy = .{ .kept = 7, .kept_gen = 3, .stream = 5, .src = 0, .dst = 4096, .rows = 2048 } };
    const mv: ShardOp = .{ .move = .{ .stream = 6, .gen = 4, .src = 0, .src_part = 0, .dst = 1 << 19, .dst_part = 1, .size = 4096, .rows = 2048, .hold_gen = 9, .tick = 0 } };
    try t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 11, .index = 0, .last = true, .ops = &.{ cp, cp } } }, 2)); // a stream named twice: one transfer
    try testing.expectEqual(@as(usize, 1), t.loading());
    // a second transfer of a stream in flight is refused, and the refusal changes nothing
    const before = t;
    try testing.expectError(error.StreamLoading, t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 12, .index = 0, .last = true, .ops = &.{ mv, cp } } }, 3)));
    try testing.expectEqualDeep(before, t);
    try t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 12, .index = 0, .last = true, .ops = &.{mv} } }, 3));
    try testing.expectEqual(@as(usize, 2), t.loading());
    // no new layout under transfers in flight
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .shard_set = .{ .version = 1, .parts = 2, .part_rows = 1 << 19, .owner_epoch = 20 } }, 4)));
    // the ask and the finish name the transfer's own plan: its transaction and its owner epoch
    try testing.expectError(error.UnknownStream, t.admit(&cmd(&t, .{ .shard_xfer = .{ .owner_epoch = 11, .stream = 9 } }, 2)));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .shard_xfer = .{ .owner_epoch = 11, .stream = 5 } }, 3)));
    try testing.expectError(error.StaleEpoch, t.admit(&cmd(&t, .{ .shard_xfer = .{ .owner_epoch = 12, .stream = 5 } }, 2)));
    try t.admit(&cmd(&t, .{ .shard_xfer = .{ .owner_epoch = 11, .stream = 5 } }, 2));
    try t.admit(&cmd(&t, .{ .shard_finish = .{ .owner_epoch = 12, .stream = 6, .ok = false } }, 3)); // any order between streams
    try t.admit(&cmd(&t, .{ .shard_finish = .{ .owner_epoch = 11, .stream = 5, .ok = true } }, 2));
    try testing.expectError(error.UnknownStream, t.admit(&cmd(&t, .{ .shard_finish = .{ .owner_epoch = 11, .stream = 5, .ok = true } }, 2)));
    try testing.expectEqual(@as(usize, 0), t.loading());
    try t.admit(&cmd(&t, .{ .shard_set = .{ .version = 1, .parts = 2, .part_rows = 1 << 19, .owner_epoch = 20 } }, 4));
    // at most max_loads at once
    var ops: [max_loads + 1]ShardOp = undefined;
    for (&ops, 0..) |*o, i| o.* = .{ .copy = .{ .kept = 7, .kept_gen = 3, .stream = 100 + i, .src = 0, .dst = 4096, .rows = 2048 } };
    try testing.expectError(error.TooManyLoads, t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 21, .index = 0, .last = true, .ops = &ops } }, 5)));
    try t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 21, .index = 0, .last = true, .ops = ops[0..max_loads] } }, 5));
    // stats opens its own transaction
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .stats = .{ .set = .tier } }, 7)));
    try t.admit(&cmd(&t, .{ .stats = .{ .set = .tier } }, 6));
}

/// Rank 0's half of a shard plan: its ops (in frames), each transfer asked and finished, then the digests compared.
fn runPlan(leader: *Leader, sink: Sink, links: []const ?link.Link, pl: *parts.Planner, epoch: *u64, ops: []const ShardOp, loads: *u32) !void {
    const gpa = testing.allocator;
    epoch.* += 1;
    var txn: u64 = 0;
    var i: usize = 0;
    var index_: u16 = 0;
    while (true) : (index_ += 1) {
        const n = @min(ops.len - i, max_shard_ops);
        const h = try leader.post(sink, txn, .{ .shard_ops = .{ .owner_epoch = epoch.*, .index = index_, .last = i + n == ops.len, .ops = ops[i..][0..n] } });
        txn = h.txn;
        i += n;
        if (i == ops.len) break;
    }
    var replies: [max_world - 1]Frame = undefined;
    var seen: [max_loads]u64 = undefined;
    var ns: usize = 0;
    for (ops) |o| if (loadOf(o)) |stream| {
        if (std.mem.indexOfScalar(u64, seen[0..ns], stream) != null) continue;
        seen[ns] = stream;
        ns += 1;
        _ = try leader.ask(sink, links, txn, .{ .shard_xfer = .{ .owner_epoch = epoch.*, .stream = stream } }, replies[0..links.len]);
        var ok = true; // (rank 0's own side: done)
        for (replies[0..links.len]) |r| ok = ok and r.msg.shard_xfer_reply.ok;
        _ = try leader.post(sink, txn, .{ .shard_finish = .{ .owner_epoch = epoch.*, .stream = stream, .ok = ok } });
        _ = try pl.finish(stream, ok);
        loads.* += 1;
    };
    _ = try leader.ask(sink, links, 0, .{ .shard_check = .{ .owner_epoch = epoch.*, .digest = pl.digest() } }, replies[0..links.len]);
    for (replies[0..links.len]) |r| try testing.expectEqualSlices(u8, &pl.digest(), &r.msg.shard_check_reply.digest);
    _ = gpa;
}

test "two ranks' planners stay equal over the wire: layout agreement, plans, transfers, finishes, digests" {
    const io = testing.io;
    const gpa = testing.allocator;
    const C = parts.chunk_rows;
    const port = testPort(io);
    const peer = try gpa.create(Peer);
    defer gpa.destroy(peer);
    peer.* = .{ .port = port, .fx = testFx(1) };
    peer.side.cap = 40 * C; // smaller than rank 0's: the pool agreed is the smaller, in whole parts
    defer if (peer.side.pl) |*x| x.deinit(gpa);
    const th = try std.Thread.spawn(.{}, Peer.run, .{peer});
    var l = try link.Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var ts: TestSink = .{ .l = l };
    const leader = try gpa.create(Leader);
    defer gpa.destroy(leader);
    leader.* = Leader.init(2, freshEpoch(io), testFx(1));
    const links = [_]?link.Link{l};
    _ = try leader.agree(ts.sink(), &links);
    // part agreement: every rank's capacity, the pool every rank can hold, the parts and the first owner epoch
    var replies: [1]Frame = undefined;
    _ = try leader.ask(ts.sink(), &links, 0, .{ .shard_layout = .{ .version = parts.layout_version, .parts = 2, .cap = 64 * C, .carry = true, .twins = true, .schema = shard_schema } }, &replies);
    const lr = replies[0].msg.shard_layout_reply;
    try testing.expectEqual(parts.layout_version, lr.version);
    try testing.expectEqual(shard_schema, lr.schema);
    const cap = try parts.agree(&.{ 64 * C, @intCast(lr.cap) }, 2);
    try testing.expectEqual(@as(usize, 40 * C), cap);
    var epoch: u64 = 1;
    _ = try leader.post(ts.sink(), 0, .{ .shard_set = .{ .version = parts.layout_version, .parts = 2, .part_rows = cap / 2, .owner_epoch = epoch } });
    var pl = try parts.Planner.init(.{ .cap = cap, .parts = 2, .carry_bytes = 720 });
    defer pl.deinit(gpa);
    var loads: u32 = 0;
    // a stream, a state it keeps, its release; then two requests resume that state: the second copies (or carries) it
    {
        var p = try pl.planAdmit(gpa, .{ .stream = 1, .slot = 1, .size = 8 * C });
        defer p.deinit(gpa);
        try testing.expectEqual(parts.Status.placed, p.status);
        try pl.apply(gpa, &p);
        try runPlan(leader, ts.sink(), &links, &pl, &epoch, p.ops.items, &loads);
    }
    {
        const k = try pl.keepOp(7, 1, 4 * C);
        try pl.applyOps(gpa, &.{k});
        try runPlan(leader, ts.sink(), &links, &pl, &epoch, &.{k}, &loads);
        var p = try pl.planRelease(gpa, 1);
        defer p.deinit(gpa);
        try pl.apply(gpa, &p);
        try runPlan(leader, ts.sink(), &links, &pl, &epoch, p.ops.items, &loads);
    }
    for ([_]u64{ 10, 11 }) |stream| {
        var p = try pl.planAdmit(gpa, .{ .stream = stream, .slot = @intCast(stream), .size = 8 * C, .source = 7 });
        defer p.deinit(gpa);
        try testing.expectEqual(parts.Status.placed, p.status);
        try pl.apply(gpa, &p);
        try runPlan(leader, ts.sink(), &links, &pl, &epoch, p.ops.items, &loads);
    }
    try testing.expect(loads > 0);
    // the follower's counters ride a stats reply
    _ = try leader.ask(ts.sink(), &links, 0, .{ .stats = .{ .set = .tier } }, &replies);
    try testing.expectEqual(@as(u8, 3), replies[0].msg.stats_reply.n);
    try testing.expectEqual(@as(u64, loads), replies[0].msg.stats_reply.values[2]);
    try l.send(&.{}); // done
    th.join();
    try testing.expectEqual(@as(?anyerror, null), peer.err);
    try testing.expectEqualSlices(u8, &pl.digest(), &peer.side.pl.?.digest());
}

test "the tracker refuses a reserve or commit that names another at, part or kept state" {
    const p: Digest = @splat(9);
    var t = try agreedTracker();
    try t.admit(&cmd(&t, .{ .has = .{ .point = p, .compat = p, .at = 6144, .blocks = 3, .layout = .owner_sharded, .part = 1 } }, 1));
    const res: Reserve = .{ .point = p, .kept = 7, .slot = 0, .base = 0, .size = 8192, .at = 6144, .blocks = 3, .generation = 0, .part = 1 };
    var r = res;
    r.at = 4096;
    try testing.expectError(error.WrongPoint, t.admit(&cmd(&t, .{ .reserve = r }, 1)));
    r = res;
    r.part = 0;
    try testing.expectError(error.WrongPoint, t.admit(&cmd(&t, .{ .reserve = r }, 1)));
    try t.admit(&cmd(&t, .{ .reserve = res }, 1));
    try t.admit(&cmd(&t, .{ .start = .{ .point = p, .blocks = 3 } }, 1));
    try t.admit(&cmd(&t, .{ .validate = .{ .upto = 3, .shared = false } }, 1));
    try testing.expectError(error.WrongPoint, t.admit(&cmd(&t, .{ .commit = .{ .point = p, .at = 6144, .kept = 8, .shared = false } }, 1)));
    try testing.expectError(error.WrongPoint, t.admit(&cmd(&t, .{ .commit = .{ .point = p, .at = 2048, .kept = 7, .shared = false } }, 1)));
    try t.admit(&cmd(&t, .{ .commit = .{ .point = p, .at = 6144, .kept = 7, .shared = false } }, 1));
}

/// A loopback pair: one Peer thread per follower (ranks 1..world-1), rank 0's links and a Leader agreed with them.
const Pair = struct {
    peers: [max_world - 1]*Peer = undefined,
    threads: [max_world - 1]std.Thread = undefined,
    joined: [max_world - 1]bool = @splat(false),
    links: [max_world - 1]?link.Link = @splat(null),
    leader: *Leader,
    n: usize,

    fn sendAll(ctx: *anyopaque, sub: []const u8) anyerror!void {
        const s: *Pair = @ptrCast(@alignCast(ctx));
        for (s.links[0..s.n]) |x| try x.?.send(sub);
    }
    fn sink(s: *Pair) Sink {
        return .{ .ctx = s, .send = sendAll };
    }
    fn ls(s: *const Pair) []const ?link.Link {
        return s.links[0..s.n];
    }

    /// `setup(peer)` adjusts each Peer before its thread starts.
    fn start(s: *Pair, world: u16, setup: anytype) !void {
        const gpa = testing.allocator;
        const io = testing.io;
        s.n = world - 1;
        for (0..s.n) |i| {
            s.peers[i] = try gpa.create(Peer);
            s.peers[i].* = .{ .port = testPort(io), .fx = testFx(1), .rank = @intCast(i + 1) };
            setup.f(s.peers[i]);
            s.threads[i] = try std.Thread.spawn(.{}, Peer.run, .{s.peers[i]});
        }
        for (0..s.n) |i| s.links[i] = try link.Link.listen(.{ 127, 0, 0, 1 }, s.peers[i].port);
        s.leader.* = Leader.init(world, freshEpoch(io), testFx(1));
    }

    /// Waits for follower i + 1's thread to end (it ends by itself on an error).
    fn join(s: *Pair, i: usize) void {
        if (s.joined[i]) return;
        s.threads[i].join();
        s.joined[i] = true;
    }

    /// "done" to every follower, the threads joined, everything freed.
    fn finish(s: *Pair) void {
        for (s.links[0..s.n]) |x| x.?.send(&.{}) catch {};
        for (0..s.n) |i| s.join(i);
        for (s.links[0..s.n]) |*x| x.*.?.close();
        for (s.peers[0..s.n]) |p| testing.allocator.destroy(p);
        testing.allocator.destroy(s.leader);
    }
};

const NoSetup = struct {
    fn f(_: *Peer) void {}
};

fn newPair() !Pair {
    return .{ .leader = try testing.allocator.create(Leader), .n = 0 };
}

test "rank 0 refuses every malformed reply: stale epoch, wrong seq/txn/sender/op, a wrong echo" {
    const want = [_]anyerror{ error.StaleEpoch, error.OutOfOrder, error.OutOfOrder, error.WrongRank, error.UnexpectedReply, error.WrongPoint };
    for (std.enums.values(BadReply), want) |bad, e| {
        var s = try newPair();
        defer s.finish();
        const Setup = struct {
            var b: BadReply = undefined;
            fn f(p: *Peer) void {
                p.bad = b;
            }
        };
        Setup.b = bad;
        try s.start(2, Setup);
        _ = try s.leader.agree(s.sink(), s.ls());
        var replies: [1]Frame = undefined;
        try testing.expectError(error.PairBroken, s.leader.ask(s.sink(), s.ls(), 0, .{ .has = .{ .point = @splat(9), .compat = s.leader.compat(), .at = 2048, .blocks = 1, .layout = .replicated, .part = 0 } }, &replies));
        try testing.expectEqual(e, s.leader.broken.?.err);
        try testing.expectEqual(@as(u16, 1), s.leader.broken.?.rank);
        try testing.expectError(error.PairBroken, s.leader.ask(s.sink(), s.ls(), 0, .{ .stats = .{ .set = .tier } }, &replies));
    }
}

test "a follower whose handler fails ends, and rank 0 sees its link go down" {
    var s = try newPair();
    defer s.finish();
    const Setup = struct {
        fn f(p: *Peer) void {
            p.side.has_err = true;
        }
    };
    try s.start(2, Setup);
    _ = try s.leader.agree(s.sink(), s.ls());
    var replies: [1]Frame = undefined;
    try testing.expectError(error.PairBroken, s.leader.ask(s.sink(), s.ls(), 0, .{ .has = .{ .point = @splat(9), .compat = s.leader.compat(), .at = 2048, .blocks = 1, .layout = .replicated, .part = 0 } }, &replies));
    try testing.expectEqual(@as(anyerror, error.PeerDown), s.leader.broken.?.err);
    try testing.expectEqual(@as(u16, 1), s.leader.broken.?.rank);
    s.join(0);
    try testing.expectEqual(@as(?anyerror, error.DiskGone), s.peers[0].err);
}

test "a follower refuses an agreed whose digests are not the ones it answered hello with" {
    var s = try newPair();
    defer s.finish();
    try s.start(2, NoSetup);
    var replies: [1]Frame = undefined;
    _ = try s.leader.ask(s.sink(), s.ls(), 0, .{ .hello = .{ .world = 2, .fx = s.leader.fx } }, &replies);
    const fx = s.leader.fx;
    _ = try s.leader.post(s.sink(), 0, .{ .agreed = .{ .world = 2, .common = @splat(0xab), .ranks = .{ fx.rank, fx.rank, @splat(0), @splat(0) } } });
    s.join(0);
    try testing.expectEqual(@as(?anyerror, error.FingerprintMismatch), s.peers[0].err);
}

test "a follower refuses a has whose compat is not the agreed pair" {
    var f = Follower.init(1, testFx(1));
    var side: FakeSide = .{};
    const fx = testFx(1);
    var buf: [max_frame]u8 = undefined;
    _ = try f.accept(try encode(&buf, .{ .op = .hello, .from = 0, .epoch = 3, .txn = 0, .seq = 0 }, .{ .hello = .{ .world = 2, .fx = fx } }));
    f.owed = null; // (its reply is not sent here)
    try serve(&f, undefined, try encode(&buf, .{ .op = .agreed, .from = 0, .epoch = 3, .txn = 0, .seq = 1 }, .{ .agreed = .{ .world = 2, .common = fx.common, .ranks = .{ fx.rank, fx.rank, @splat(0), @splat(0) } } }), &side);
    try testing.expect(side.agreed_seen);
    const has = try encode(&buf, .{ .op = .has, .from = 0, .epoch = 3, .txn = 1, .seq = 2 }, .{ .has = .{ .point = @splat(9), .compat = fx.common, .at = 2048, .blocks = 1, .layout = .replicated, .part = 0 } });
    try testing.expectError(error.WrongCompat, serve(&f, undefined, has, &side)); // fails before any reply
}

test "a refused post ends the pair (rank 0 already acted); admissible refuses without ending it" {
    const gpa = testing.allocator;
    const C = parts.chunk_rows;
    var s = try newPair();
    defer s.finish();
    try s.start(2, NoSetup);
    const l = s.leader;
    _ = try l.agree(s.sink(), s.ls());
    var replies: [1]Frame = undefined;
    _ = try l.ask(s.sink(), s.ls(), 0, .{ .shard_layout = .{ .version = parts.layout_version, .parts = 2, .cap = 32 * C, .carry = true, .twins = true, .schema = shard_schema } }, &replies);
    _ = try l.post(s.sink(), 0, .{ .shard_set = .{ .version = parts.layout_version, .parts = 2, .part_rows = 16 * C, .owner_epoch = 1 } });
    defer if (s.peers[0].side.pl) |*x| x.deinit(gpa);
    var pl = try parts.Planner.init(.{ .cap = 32 * C, .parts = 2, .carry_bytes = 720 });
    defer pl.deinit(gpa);
    var p = try pl.planAdmit(gpa, .{ .stream = 1, .slot = 1, .size = 8 * C });
    defer p.deinit(gpa);
    // the plan's frame names a skipped owner epoch: admissible says so and nothing ends
    const bad: Msg = .{ .shard_ops = .{ .owner_epoch = 3, .index = 0, .last = true, .ops = p.ops.items } };
    try testing.expectError(error.OutOfOrder, l.admissible(0, bad));
    try testing.expect(l.broken == null);
    // rank 0 applies it anyway and posts: the followers' planners would diverge, so the pair ends
    try pl.apply(gpa, &p);
    try testing.expectError(error.PairBroken, l.post(s.sink(), 0, bad));
    try testing.expectEqual(@as(anyerror, error.OutOfOrder), l.broken.?.err);
    try testing.expectEqual(@as(u16, 0), l.broken.?.rank);
    try testing.expectError(error.PairBroken, l.admissible(0, .{ .stats = .{ .set = .tier } }));
    try testing.expectError(error.PairBroken, l.ask(s.sink(), s.ls(), 0, .{ .stats = .{ .set = .tier } }, &replies));
}

test "rank 0 collects three ranks' replies in rank order and names the late one" {
    var s = try newPair();
    defer s.finish();
    const Setup = struct {
        fn f(p: *Peer) void {
            if (p.rank == 2) p.delays_ms = .{ 150, 500, 0, 0 }; // rank 2: late, then too late
        }
    };
    try s.start(3, Setup);
    const l = s.leader;
    const a = try l.agree(s.sink(), s.ls());
    try testing.expectEqual(@as(u16, 3), a.world);
    var replies: [2]Frame = undefined;
    const has: Msg = .{ .has = .{ .point = @splat(9), .compat = l.compat(), .at = 2048, .blocks = 1, .layout = .replicated, .part = 0 } };
    const h = try l.ask(s.sink(), s.ls(), 0, has, &replies);
    try testing.expectEqual(@as(u16, 1), replies[0].head.from);
    try testing.expectEqual(@as(u16, 2), replies[1].head.from);
    try testing.expect(allPresent(&replies));
    _ = try l.post(s.sink(), h.txn, .{ .abort = .{ .why = .cancelled, .rank = no_rank, .forget = false } });
    l.timeouts.has_ns = 250 * std.time.ns_per_ms;
    try testing.expectError(error.PairBroken, l.ask(s.sink(), s.ls(), 0, has, &replies));
    try testing.expectEqual(@as(anyerror, error.ReplyTimeout), l.broken.?.err);
    try testing.expectEqual(@as(u16, 2), l.broken.?.rank);
}

test "the notices open their own transactions; status asks only a started restore; a replicated reserve has no generation" {
    var t = try agreedTracker();
    const p: Digest = @splat(4);
    const forks = [_]Fork{.{ .name = @splat(1), .at = 2048 }};
    const keep: Msg = .{ .keep = .{ .slot = 1, .kept = 9, .base = 0, .at = 4096, .point = p, .group = 3, .shared = false, .root = @splat(2), .part = null, .origin = null, .forks = &forks } };
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, keep, 2))); // a skipped transaction
    try t.admit(&cmd(&t, keep, 1));
    try testing.expectError(error.StaleTxn, t.admit(&cmd(&t, .{ .touch = .{ .point = p } }, 1)));
    {
        var f = cmd(&t, .{ .share = .{ .point = p } }, 2);
        f.head.epoch +%= 1;
        try testing.expectError(error.StaleEpoch, t.admit(&f));
        f.head.epoch = t.epoch;
        f.head.seq += 1;
        try testing.expectError(error.OutOfOrder, t.admit(&f));
    }
    try t.admit(&cmd(&t, .{ .share = .{ .point = p } }, 2));
    try t.admit(&cmd(&t, .{ .touch = .{ .point = p } }, 3));
    try t.admit(&cmd(&t, .{ .forget = .{ .point = p } }, 4));
    try t.admit(&cmd(&t, .{ .write_staged = .{ .point = p, .kept = 9, .upto = 1 } }, 5));
    try t.admit(&cmd(&t, .{ .write_cancelled = .{ .point = p, .kept = 9, .why = .drop } }, 6));
    // a restore: status only between start and validate, and only of an open restore
    const has: Msg = .{ .has = .{ .point = p, .compat = p, .at = 4096, .blocks = 2, .layout = .replicated, .part = 0 } };
    try t.admit(&cmd(&t, has, 7));
    try testing.expectError(error.UnknownTxn, t.admit(&cmd(&t, .{ .status = .{} }, 99)));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .status = .{} }, 7)));
    try testing.expectError(error.BadValue, t.admit(&cmd(&t, .{ .reserve = .{ .point = p, .kept = 12, .slot = 0, .base = 0, .size = 4096, .at = 4096, .blocks = 2, .generation = 5, .part = 0 } }, 7)));
    try t.admit(&cmd(&t, .{ .reserve = .{ .point = p, .kept = 12, .slot = 0, .base = 0, .size = 4096, .at = 4096, .blocks = 2, .generation = 0, .part = 0 } }, 7));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .status = .{} }, 7))); // before start
    try t.admit(&cmd(&t, .{ .start = .{ .point = p, .blocks = 2 } }, 7));
    try t.admit(&cmd(&t, .{ .status = .{} }, 7));
    try t.admit(&cmd(&t, .{ .keep = keep.keep }, 8)); // notices interleave with a restore
    try t.admit(&cmd(&t, .{ .status = .{} }, 7));
    try t.admit(&cmd(&t, .{ .validate = .{ .upto = 2, .shared = false } }, 7));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .status = .{} }, 7))); // after validate
    try testing.expectError(error.WrongPoint, t.admit(&cmd(&t, .{ .commit = .{ .point = p, .at = 4096, .kept = 13, .shared = false } }, 7)));
    try t.admit(&cmd(&t, .{ .commit = .{ .point = p, .at = 4096, .kept = 12, .shared = false } }, 7));
    // an owner_sharded point's reserve names its extent's generation
    try t.admit(&cmd(&t, .{ .has = .{ .point = p, .compat = p, .at = 4096, .blocks = 2, .layout = .owner_sharded, .part = 1 } }, 9));
    try t.admit(&cmd(&t, .{ .reserve = .{ .point = p, .kept = 14, .slot = 0, .base = 0, .size = 4096, .at = 4096, .blocks = 2, .generation = 5, .part = 1 } }, 9));
    // nothing between a plan's frames, notices included
    try t.admit(&cmd(&t, .{ .shard_set = .{ .version = 1, .parts = 2, .part_rows = 1 << 19, .owner_epoch = 1 } }, 10));
    const ops = [_]ShardOp{.{ .drop = .{ .kept = 1, .gen = 1 } }};
    try t.admit(&cmd(&t, .{ .shard_ops = .{ .owner_epoch = 2, .index = 0, .last = false, .ops = &ops } }, 11));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .touch = .{ .point = p } }, 12)));
    try testing.expectError(error.OutOfOrder, t.admit(&cmd(&t, .{ .status = .{} }, 9)));
}

test "a keep's forks are bounded both ways and its optionals have one canonical form" {
    const gpa = testing.allocator;
    const forks = try gpa.alloc(Fork, max_keep_forks + 1);
    defer gpa.free(forks);
    for (forks, 0..) |*f, i| f.* = .{ .name = @splat(@truncate(i)), .at = @intCast(i * 2048) };
    const buf = try gpa.alloc(u8, max_frame);
    defer gpa.free(buf);
    const head: Head = .{ .op = .keep, .from = 0, .epoch = 1, .txn = 1, .seq = 2 };
    var k: Keep = .{ .slot = 0, .kept = 1, .base = 0, .at = 2048, .point = @splat(1), .group = 0, .shared = false, .root = @splat(0), .part = 7, .origin = .{ .key = @splat(3), .at = 1024 }, .forks = forks };
    try testing.expectError(error.BadCount, encode(buf, head, .{ .keep = k }));
    k.forks = forks[0..max_keep_forks];
    const full = try encode(buf, head, .{ .keep = k });
    try testing.expect(full.len <= max_frame);
    var few: [8]Fork = undefined;
    try testing.expectError(error.BadCount, decodeInto(full, .{ .forks = &few })); // less room: refused, never overrun
    try testing.expectError(error.BadCount, decode(full, &.{})); // (decode has no forks buffer)
    const back = try gpa.alloc(Fork, max_keep_forks);
    defer gpa.free(back);
    const f = try decodeInto(full, .{ .forks = back });
    try testing.expectEqual(@as(?u32, 7), f.msg.keep.part);
    try testing.expectEqualSlices(u8, &k.origin.?.key, &f.msg.keep.origin.?.key);
    try testing.expectEqual(@as(usize, max_keep_forks), f.msg.keep.forks.len);
    try testing.expectEqualDeep(forks[max_keep_forks - 1], f.msg.keep.forks[max_keep_forks - 1]);
    // no forks, no part, no origin: decode needs no buffer; the absent optionals are one zero byte each
    var b2: [256]u8 = undefined;
    k.forks = &.{};
    k.part = null;
    k.origin = null;
    const small = try encode(&b2, head, .{ .keep = k });
    const g = try decode(small, &.{});
    try testing.expect(g.msg.keep.part == null and g.msg.keep.origin == null);
    // part's flag sits after root: neither 0 nor 1 is refused
    const flag_at = header_len + 4 + 8 + 8 + 4 + 32 + 8 + 1 + 16;
    try testing.expectEqual(@as(u8, 0), small[flag_at]);
    small[flag_at] = 2;
    try testing.expectError(error.BadValue, decode(small, &.{}));
    small[flag_at] = 1; // present, but its payload is missing
    try testing.expectError(error.ShortFrame, decode(small, &.{}));
    // values check() forbids
    k.at = 0;
    try testing.expectError(error.BadValue, encode(&b2, head, .{ .keep = k }));
    k.at = 2048;
    k.kept = 0;
    try testing.expectError(error.BadValue, encode(&b2, head, .{ .keep = k }));
}

test "Leader.fail keeps the first cause and every later call is PairBroken" {
    var l = Leader.init(2, 3, testFx(1));
    try testing.expectEqual(@as(anyerror, error.PairBroken), l.fail(1, error.Timeout));
    try testing.expectEqual(@as(anyerror, error.ReplyTimeout), l.broken.?.err); // (a Timeout reads as ReplyTimeout)
    try testing.expectEqual(@as(u16, 1), l.broken.?.rank);
    try testing.expectEqual(@as(anyerror, error.PairBroken), l.fail(0, error.TransferFailed));
    try testing.expectEqual(@as(anyerror, error.ReplyTimeout), l.broken.?.err); // the first cause stays
    try testing.expectEqual(@as(u16, 1), l.broken.?.rank);
    try testing.expectError(error.PairBroken, l.admissible(0, .{ .stats = .{ .set = .tier } }));
    var nothing: [1]u8 = undefined;
    const sink: Sink = .{ .ctx = &nothing, .send = struct {
        fn f(_: *anyopaque, _: []const u8) anyerror!void {
            return error.MustNotSend;
        }
    }.f };
    try testing.expectError(error.PairBroken, l.post(sink, 0, .{ .touch = .{ .point = @splat(0) } }));
    var m = Leader.init(2, 4, testFx(1));
    try testing.expectEqual(@as(anyerror, error.PairBroken), m.fail(0, error.TransferTimeout));
    try testing.expectEqual(@as(anyerror, error.TransferTimeout), m.broken.?.err);
    try testing.expectEqual(@as(u16, 0), m.broken.?.rank);
}

test "a keep's pictures: the most forks and pictures fit one frame, bounds and order refused both ways" {
    const gpa = testing.allocator;
    const forks = try gpa.alloc(Fork, max_keep_forks + 1);
    defer gpa.free(forks);
    for (forks, 0..) |*f, i| f.* = .{ .name = @splat(@truncate(i)), .at = @intCast(i) };
    const media = try gpa.alloc(Media, max_keep_media + 1);
    defer gpa.free(media);
    for (media, 0..) |*m, i| m.* = .{ .start = @intCast(i * 600), .len = 576, .digest = @splat(@truncate(i)) };
    const at: u32 = (max_keep_media + 1) * 600;
    const buf = try gpa.alloc(u8, max_frame);
    defer gpa.free(buf);
    const head: Head = .{ .op = .keep, .from = 0, .epoch = 1, .txn = 3, .seq = 4 };
    const base: Keep = .{ .slot = 1, .kept = 9, .base = 0, .at = at, .point = @splat(7), .group = 2, .shared = false, .root = @splat(1), .part = null, .origin = null, .placeholder = 129264, .forks = forks[0..max_keep_forks], .media = media[0..max_keep_media] };
    const full = try encode(buf, head, .{ .keep = base });
    try testing.expect(full.len <= max_frame);
    const got_f = try gpa.alloc(Fork, max_keep_forks);
    defer gpa.free(got_f);
    const got_m = try gpa.alloc(Media, max_keep_media);
    defer gpa.free(got_m);
    const f = try decodeInto(full, .{ .forks = got_f, .media = got_m });
    try testing.expectEqualDeep(Msg{ .keep = base }, f.msg);
    // a receiver with room for fewer pictures (or none) refuses, never overruns
    try testing.expectError(error.BadCount, decodeInto(full, .{ .forks = got_f, .media = got_m[0..10] }));
    try testing.expectError(error.BadCount, decodeInto(full, .{ .forks = got_f }));
    // more than a keep carries, either slice
    var k = base;
    k.media = media;
    try testing.expectError(error.BadCount, encode(buf, head, .{ .keep = k }));
    k = base;
    k.forks = forks;
    try testing.expectError(error.BadCount, encode(buf, head, .{ .keep = k }));
    // a text point: no pictures, no placeholder (the bytes of a v2 keep without pictures carry two empty fields)
    k = base;
    k.media = &.{};
    k.placeholder = null;
    const text = try encode(buf, head, .{ .keep = k });
    try testing.expectEqual(@as(usize, 0), (try decodeInto(text, .{ .forks = got_f })).msg.keep.media.len);
    // one canonical form: a placeholder without pictures, pictures without one, out of order, overlapping, empty,
    // or starting at or past the cut
    k.placeholder = 5;
    try testing.expectError(error.BadValue, encode(buf, head, .{ .keep = k }));
    const two = [_]Media{ .{ .start = 10, .len = 4, .digest = @splat(1) }, .{ .start = 20, .len = 4, .digest = @splat(2) } };
    k = base;
    k.at = 100;
    k.media = &two;
    k.placeholder = null;
    try testing.expectError(error.BadValue, encode(buf, head, .{ .keep = k }));
    k.placeholder = 5;
    _ = try encode(buf, head, .{ .keep = k });
    const bad = [_][2]Media{
        .{ two[1], two[0] }, // out of order
        .{ two[0], .{ .start = 12, .len = 4, .digest = @splat(2) } }, // overlapping
        .{ two[0], .{ .start = 20, .len = 0, .digest = @splat(2) } }, // empty
        .{ two[0], .{ .start = 100, .len = 4, .digest = @splat(2) } }, // at the cut
    };
    for (&bad) |*b| {
        k.media = b;
        try testing.expectError(error.BadValue, encode(buf, head, .{ .keep = k }));
    }
    // the same refusal on decode: a valid frame with its second picture's start moved past the cut
    k.media = &two;
    const ok = try encode(buf, head, .{ .keep = k });
    const off = ok.len - 40; // (the last picture's start: the frame's trailing item)
    std.mem.writeInt(u32, ok[off..][0..4], 200, .little);
    try testing.expectError(error.BadValue, decodeInto(ok, .{ .forks = got_f, .media = got_m }));
}

test "the union tag's numbers are in the schema: a renumbered op changes the stamp" {
    const T1 = enum(u8) { a = 1, b = 2 };
    const T2 = enum(u8) { a = 1, b = 3 };
    const U1 = union(T1) { a: u32, b: u64 };
    const U2 = union(T2) { a: u32, b: u64 };
    try testing.expect(schemaOf(U1) != schemaOf(U2));
    const U3 = union(T1) { a: u32, b: u64 };
    try testing.expectEqual(schemaOf(U1), schemaOf(U3));
}

test "every message's layout is in the wire stamp" {
    const A = struct { x: u32, y: ?u32 };
    const B = struct { x: u32, y: u32 };
    const C = struct { x: u32, y: []const u32 };
    try testing.expect(schemaOf(A) != schemaOf(B));
    try testing.expect(schemaOf(B) != schemaOf(C));
    const st = stampBytes();
    try testing.expectEqual(version, std.mem.readInt(u16, st[0..2], .little));
    try testing.expectEqual(shard_schema, std.mem.readInt(u32, st[4..8], .little));
    try testing.expectEqual(msg_schema, std.mem.readInt(u32, st[8..12], .little));
}

/// Rank 0's half of a restore over the status asks: has, reserve, start, progress, then status polls between
/// "rounds" (a touch notice each, as lanes frames go on) until every follower is complete, validate, commit. Each
/// status ask's time is recorded; the longest is returned.
fn restoreByStatus(l: *Leader, sink: Sink, links: []const ?link.Link, point: Digest, blocks: u32, rounds: *u32) !struct { ok: bool, longest_ns: u64, failed: ?Failure = null } {
    var replies: [max_world - 1]Frame = undefined;
    const rs = replies[0..links.len];
    const h = try l.ask(sink, links, 0, .{ .has = .{ .point = point, .compat = l.compat(), .at = @as(u64, blocks) * 2048, .blocks = blocks, .layout = .replicated, .part = 0 } }, rs);
    if (!allPresent(rs)) return error.NotPresent;
    _ = try l.post(sink, h.txn, .{ .reserve = .{ .point = point, .kept = 77, .slot = 0, .base = 0, .size = @as(u64, blocks) * 2048, .at = @as(u64, blocks) * 2048, .blocks = blocks, .generation = 0, .part = 0 } });
    _ = try l.post(sink, h.txn, .{ .start = .{ .point = point, .blocks = blocks } });
    _ = try l.post(sink, h.txn, .{ .progress = .{ .upto = blocks } }); // (rank 0's own copies are done)
    var longest: u64 = 0;
    const give_up = monotonicNs() + 5 * std.time.ns_per_s;
    while (true) {
        const t0 = monotonicNs();
        _ = try l.ask(sink, links, h.txn, .{ .status = .{} }, rs);
        longest = @max(longest, monotonicNs() - t0);
        const st = statusOf(rs, blocks);
        if (st.failed) |f| {
            _ = try l.post(sink, h.txn, .{ .abort = .{ .why = f.why, .rank = f.rank, .forget = false } });
            return .{ .ok = false, .longest_ns = longest, .failed = f.why };
        }
        if (st.complete) break;
        if (monotonicNs() > give_up) return error.TestTimedOut;
        // a decode round goes on meanwhile: the ordered stream keeps flowing
        _ = try l.post(sink, 0, .{ .touch = .{ .point = point } });
        rounds.* += 1;
        std.Io.sleep(testing.io, .fromMilliseconds(5), .awake) catch {};
    }
    _ = try l.ask(sink, links, h.txn, .{ .validate = .{ .upto = blocks, .shared = false } }, rs);
    if (firstFailure(rs) != null) return .{ .ok = false, .longest_ns = longest };
    _ = try l.post(sink, h.txn, .{ .commit = .{ .point = point, .at = @as(u64, blocks) * 2048, .kept = 77, .shared = false } });
    return .{ .ok = true, .longest_ns = longest };
}

test "a slow follower's restore: status answers at once, rank 0 keeps its rounds, validate only when complete" {
    var s = try newPair();
    defer s.finish();
    const Setup = struct {
        fn f(p: *Peer) void {
            p.side.ready_at = monotonicNs() + 400 * std.time.ns_per_ms; // its "disk" needs 400 ms for the point
        }
    };
    try s.start(2, Setup);
    const l = s.leader;
    _ = try l.agree(s.sink(), s.ls());
    var rounds: u32 = 0;
    const r = try restoreByStatus(l, s.sink(), s.ls(), @splat(9), 3, &rounds);
    try testing.expect(r.ok);
    try testing.expect(rounds >= 10); // rank 0 went on while the follower copied (~400 ms / 5 ms polls)
    try testing.expect(r.longest_ns < 200 * std.time.ns_per_ms); // never a stall for the follower's disk
    // a keep with forks and an origin reaches the follower's handler; the other notices too
    const forks = [_]Fork{ .{ .name = @splat(1), .at = 2048 }, .{ .name = @splat(2), .at = 4096 } };
    _ = try l.post(s.sink(), 0, .{ .keep = .{ .slot = 0, .kept = 5, .base = 0, .at = 6144, .point = @splat(9), .group = 1, .shared = true, .root = @splat(3), .part = null, .origin = .{ .key = @splat(8), .at = 2048 }, .forks = &forks } });
    _ = try l.post(s.sink(), 0, .{ .share = .{ .point = @splat(9) } });
    _ = try l.post(s.sink(), 0, .{ .forget = .{ .point = @splat(9) } });
    var replies: [1]Frame = undefined;
    _ = try l.ask(s.sink(), s.ls(), 0, .{ .stats = .{ .set = .tier } }, &replies); // (the follower read everything before)
    const side = &s.peers[0].side;
    try testing.expectEqual(@as(u32, 1), side.validates);
    try testing.expect(side.statuses >= rounds);
    try testing.expectEqual(@as(u32, 1), side.committed);
    try testing.expectEqual(rounds, side.touches);
    try testing.expectEqual(@as(u32, 1), side.keeps);
    try testing.expectEqual(@as(u32, 2), side.keep_forks);
    try testing.expect(side.keep_origin);
    try testing.expectEqual(@as(u32, 1), side.shares);
    try testing.expectEqual(@as(u32, 1), side.forgets);
}

test "a follower's status that records a failure aborts the restore without validate" {
    var s = try newPair();
    defer s.finish();
    const Setup = struct {
        fn f(p: *Peer) void {
            p.side.status_why = .crc;
            p.side.ready_at = std.math.maxInt(u64);
        }
    };
    try s.start(2, Setup);
    const l = s.leader;
    _ = try l.agree(s.sink(), s.ls());
    var rounds: u32 = 0;
    const r = try restoreByStatus(l, s.sink(), s.ls(), @splat(9), 2, &rounds);
    try testing.expect(!r.ok);
    try testing.expectEqual(@as(?Failure, .crc), r.failed);
    var replies: [1]Frame = undefined;
    _ = try l.ask(s.sink(), s.ls(), 0, .{ .stats = .{ .set = .tier } }, &replies);
    try testing.expectEqual(@as(u32, 0), s.peers[0].side.validates);
    try testing.expectEqual(@as(u32, 1), s.peers[0].side.aborted);
    try testing.expect(l.broken == null);
}

test "a status reply naming another point or past the point ends the pair" {
    var t = try agreedTracker();
    const p: Digest = @splat(5);
    try t.admit(&cmd(&t, .{ .has = .{ .point = p, .compat = p, .at = 4096, .blocks = 2, .layout = .replicated, .part = 0 } }, 1));
    try t.admit(&cmd(&t, .{ .reserve = .{ .point = p, .kept = 3, .slot = 0, .base = 0, .size = 4096, .at = 4096, .blocks = 2, .generation = 0, .part = 0 } }, 1));
    try t.admit(&cmd(&t, .{ .start = .{ .point = p, .blocks = 2 } }, 1));
    const asked: Head = .{ .op = .status, .from = 0, .epoch = t.epoch, .txn = 1, .seq = t.next_seq };
    const am: Msg = .{ .status = .{} };
    try echo(&t, asked, &am, .{ .status_reply = .{ .copied = 2, .small_ready = true, .why = .none, .point = p } });
    try testing.expectError(error.WrongPoint, echo(&t, asked, &am, .{ .status_reply = .{ .copied = 1, .small_ready = false, .why = .none, .point = @splat(6) } }));
    try testing.expectError(error.BadEcho, echo(&t, asked, &am, .{ .status_reply = .{ .copied = 3, .small_ready = false, .why = .none, .point = p } }));
}

// -- PETABRIDGE (k2a-pb1): world 3 (TP3: rank 0 and two followers), P3 of the spill port ------------------------------

/// PETABRIDGE (k2a-pb1): a rank's exchange for the world-3 tests, built like testFx (every field fed `1`), plus: each
/// rank-scoped field also fed `rank` (so the three ranks' own digests differ, as their weight files and splits do),
/// the template fed `template` (a rank field: the head's chat template overlay), and one common field fed an odd
/// value when `odd` names it (a follower started from another kit, tokenizer, ...).
fn rankFx(rank: u8, template: u8, odd: ?fingerprint.Field) fingerprint.Exchange {
    var b: fingerprint.Builder = .{};
    for (fingerprint.all) |fl| {
        b.int(fl, u8, 1);
        if (fingerprint.scope(fl) == .rank) b.int(fl, u8, rank);
        if (fl == .template) b.int(fl, u8, template);
        if (odd) |o| if (o == fl) b.int(fl, u8, 0xee);
    }
    return (b.finish(null) catch unreachable).exchange();
}

/// PETABRIDGE (k2a-pb1): field `f`'s digest in an exchange (a test's look at one field).
fn fxDigest(x: fingerprint.Exchange, f: fingerprint.Field) Digest {
    for (x.entries()) |e| if (e.id == @intFromEnum(f)) return e.digest;
    unreachable;
}

/// PETABRIDGE (k2a-pb1): "done" to every follower and their threads joined, so a test can read their sides
/// (Pair.finish then only closes and frees; its second "done" goes nowhere).
fn stopAll(s: *Pair) void {
    for (s.links[0..s.n]) |x| x.?.send(&.{}) catch {};
    for (0..s.n) |i| s.join(i);
}

/// PETABRIDGE (k2a-pb1): rank 0's verdict on a has's replies, as tip spill_tier.zig's restore makes it (MiaAI 0019,
/// `refused` / `forget`): the first follower that is not present is named; missing, damaged or incompatible on any
/// follower forgets the point on every rank, incomplete (still writing) or disabled keeps it.
fn presenceVerdict(replies: []const Frame) ?struct { rank: u16, forget: bool } {
    var refused: ?u16 = null;
    var forget = false;
    for (replies, 0..) |f, i| {
        const ans = f.msg.has_reply.answer;
        if (ans == .present) continue;
        if (refused == null) refused = @intCast(i + 1);
        if (ans == .missing or ans == .damaged or ans == .incompatible) forget = true;
    }
    return if (refused) |r| .{ .rank = r, .forget = forget } else null;
}

test "PETABRIDGE TP3: rank 0 and two followers agree on world 3, every rank's own digest in rank order, and restore" {
    var s = try newPair();
    defer s.finish();
    const Setup = struct {
        fn f(p: *Peer) void {
            p.fx = rankFx(@intCast(p.rank), 0xb0, null);
        }
    };
    try s.start(3, Setup);
    const l = s.leader;
    l.fx = rankFx(0, 0xb0, null);
    const a = try l.agree(s.sink(), s.ls());
    try testing.expectEqual(@as(u16, 3), a.world);
    try testing.expectEqualSlices(u8, &l.fx.common, &a.common);
    for (0..3) |r| try testing.expectEqualSlices(u8, &rankFx(@intCast(r), 0xb0, null).rank, &a.ranks[r]);
    try testing.expectEqualSlices(u8, &@as(Digest, @splat(0)), &a.ranks[3]); // (no fourth rank)
    try testing.expect(!std.mem.eql(u8, &a.ranks[1], &a.ranks[2]));
    try testing.expectEqualSlices(u8, &a.pair(), &l.compat());
    // a whole restore across the three ranks
    var replies: [2]Frame = undefined;
    const p: Digest = @splat(9);
    const has = try l.ask(s.sink(), s.ls(), 0, .{ .has = .{ .point = p, .compat = l.compat(), .at = 6144, .blocks = 3, .layout = .replicated, .part = 0 } }, &replies);
    try testing.expect(allPresent(&replies));
    try testing.expectEqual(@as(u16, 1), replies[0].head.from);
    try testing.expectEqual(@as(u16, 2), replies[1].head.from);
    try testing.expect(presenceVerdict(&replies) == null);
    _ = try l.post(s.sink(), has.txn, .{ .reserve = .{ .point = p, .kept = 4, .slot = 1, .base = 0, .size = 8192, .at = 6144, .blocks = 3, .generation = 0, .part = 0 } });
    _ = try l.post(s.sink(), has.txn, .{ .start = .{ .point = p, .blocks = 3 } });
    _ = try l.post(s.sink(), has.txn, .{ .progress = .{ .upto = 2 } });
    _ = try l.ask(s.sink(), s.ls(), has.txn, .{ .validate = .{ .upto = 3, .shared = false } }, &replies);
    try testing.expect(firstFailure(&replies) == null);
    _ = try l.post(s.sink(), has.txn, .{ .commit = .{ .point = p, .at = 6144, .kept = 4, .shared = false } });
    try testing.expectEqual(@as(usize, 0), l.t.open());
    stopAll(&s);
    try testing.expect(l.broken == null);
    for (s.peers[0..2], 1..) |peer, r| {
        try testing.expectEqual(@as(?anyerror, null), peer.err);
        try testing.expect(peer.side.agreed_seen);
        // each follower's store is opened under its own narrowing of the agreement
        try testing.expectEqualSlices(u8, &a.storeCommon(@intCast(r)), &peer.side.store_compat);
        try testing.expectEqual(@as(u32, 1), peer.side.committed);
        try testing.expectEqual(@as(u32, 3), peer.side.copied);
    }
    try testing.expect(!std.mem.eql(u8, &s.peers[0].side.store_compat, &s.peers[1].side.store_compat));
}

test "PETABRIDGE TP3: one follower missing or damaged a point aborts the restore with forget on both followers" {
    var s = try newPair();
    defer s.finish();
    try s.start(3, NoSetup);
    const l = s.leader;
    _ = try l.agree(s.sink(), s.ls());
    const Case = struct { a1: ?Absence, a2: ?Absence, rank: u16, forget: bool };
    const cases = [_]Case{
        .{ .a1 = null, .a2 = .missing, .rank = 2, .forget = true }, // rank 2 pruned it (its own cap/floor/turns)
        .{ .a1 = .damaged, .a2 = null, .rank = 1, .forget = true },
        .{ .a1 = null, .a2 = .damaged, .rank = 2, .forget = true },
        .{ .a1 = null, .a2 = .incompatible, .rank = 2, .forget = true },
        .{ .a1 = .incomplete, .a2 = null, .rank = 1, .forget = false }, // still writing: kept everywhere
        .{ .a1 = .incomplete, .a2 = .missing, .rank = 1, .forget = true }, // the first refuser named, any loss forgets
    };
    var replies: [2]Frame = undefined;
    var forgets: u32 = 0;
    for (cases, 1..) |c, n| {
        s.peers[0].side.answer = c.a1;
        s.peers[1].side.answer = c.a2;
        const h = try l.ask(s.sink(), s.ls(), 0, .{ .has = .{ .point = @splat(9), .compat = l.compat(), .at = 4096, .blocks = 2, .layout = .replicated, .part = 0 } }, &replies);
        try testing.expect(!allPresent(&replies));
        const v = presenceVerdict(&replies).?;
        try testing.expectEqual(c.rank, v.rank);
        try testing.expectEqual(c.forget, v.forget);
        _ = try l.post(s.sink(), h.txn, .{ .abort = .{ .why = .presence, .rank = v.rank, .forget = v.forget } });
        forgets += @intFromBool(v.forget);
        try testing.expectEqual(@as(usize, 0), l.t.open()); // closed on rank 0
        // an ask after the abort: both followers took it (their trackers closed the restore too) and say so
        _ = try l.ask(s.sink(), s.ls(), 0, .{ .stats = .{ .set = .tier } }, &replies);
        for (replies, s.peers[0..2]) |r, peer| {
            try testing.expectEqual(@as(u64, n), r.msg.stats_reply.values[1]); // aborted, on both followers
            try testing.expectEqual(forgets, peer.side.abort_forgets);
            try testing.expectEqual(c.rank, peer.side.abort_rank);
        }
    }
    // the pair goes on: every rank holds it again, a restore commits on all three
    s.peers[0].side.answer = null;
    s.peers[1].side.answer = null;
    const h = try l.ask(s.sink(), s.ls(), 0, .{ .has = .{ .point = @splat(9), .compat = l.compat(), .at = 4096, .blocks = 2, .layout = .replicated, .part = 0 } }, &replies);
    try testing.expect(allPresent(&replies));
    _ = try l.post(s.sink(), h.txn, .{ .reserve = .{ .point = @splat(9), .kept = 7, .slot = 0, .base = 0, .size = 4096, .at = 4096, .blocks = 2, .generation = 0, .part = 0 } });
    _ = try l.post(s.sink(), h.txn, .{ .start = .{ .point = @splat(9), .blocks = 2 } });
    _ = try l.ask(s.sink(), s.ls(), h.txn, .{ .validate = .{ .upto = 2, .shared = false } }, &replies);
    try testing.expect(firstFailure(&replies) == null);
    _ = try l.post(s.sink(), h.txn, .{ .commit = .{ .point = @splat(9), .at = 4096, .kept = 7, .shared = false } });
    stopAll(&s);
    try testing.expect(l.broken == null);
    for (s.peers[0..2]) |peer| {
        try testing.expectEqual(@as(?anyerror, null), peer.err);
        try testing.expectEqual(@as(u32, cases.len), peer.side.aborted);
        try testing.expectEqual(@as(u32, 5), peer.side.abort_forgets);
        try testing.expectEqual(@as(u32, 1), peer.side.committed);
    }
}

test "PETABRIDGE TP3: a per-rank field (the head's chat template) never breaks agreement; a common one names rank and field" {
    var buf: [128]u8 = undefined;
    // 1. rank 0 renders with the recipe's template overlay, the followers hold the plain one: agreed
    {
        var s = try newPair();
        defer s.finish();
        const Setup = struct {
            fn f(p: *Peer) void {
                p.fx = rankFx(@intCast(p.rank), 0xb0, null);
            }
        };
        try s.start(3, Setup);
        s.leader.fx = rankFx(0, 0xa0, null);
        try testing.expectEqual(fingerprint.Scope.rank, fingerprint.scope(.template));
        try testing.expectEqualSlices(u8, &s.leader.fx.common, &rankFx(1, 0xb0, null).common); // (the common digest is blind to it)
        const a = try s.leader.agree(s.sink(), s.ls());
        try testing.expectEqual(@as(u16, 3), a.world);
        try testing.expectEqualSlices(u8, &rankFx(0, 0xa0, null).rank, &a.ranks[0]);
        // ... but it is in rank 0's own digest, so rank 0's store compat moves with the template (followers' do not)
        try testing.expect(!std.mem.eql(u8, &rankFx(0, 0xa0, null).rank, &rankFx(0, 0xb0, null).rank));
        // each rank's own Engram shard listing (engram_shard, a rank field rankFx gives per rank) differs on every
        // rank too, and the three agreed: each rank's own digest (its store namespace) carries its shards
        try testing.expectEqual(fingerprint.Scope.rank, fingerprint.scope(.engram_shard));
        for ([_]u8{ 0, 1, 2 }) |r| for ([_]u8{ 0, 1, 2 }) |q| if (r != q) {
            try testing.expect(!std.mem.eql(u8, &fxDigest(rankFx(r, 0xb0, null), .engram_shard), &fxDigest(rankFx(q, 0xb0, null), .engram_shard)));
        };
        try testing.expectEqualSlices(u8, &fxDigest(rankFx(0, 0xa0, null), .engram), &fxDigest(rankFx(2, 0xb0, null), .engram));
        try testing.expect(!std.mem.eql(u8, &a.storeCommon(1), &a.storeCommon(2)));
        stopAll(&s);
        for (s.peers[0..2]) |peer| {
            try testing.expectEqual(@as(?anyerror, null), peer.err);
            try testing.expect(peer.side.agreed_seen);
        }
    }
    // 2. rank 2 started from another kit (a common field): refused on rank 0 and rank 2, naming rank 2 and the field;
    //    rank 1 answered ok and is never sent agreed
    // 3. both followers differ in common fields: the first rank is named (rank 1, its field)
    const Odd = struct { odd1: ?fingerprint.Field, odd2: ?fingerprint.Field, rank: u16, field: fingerprint.Field, text: []const u8 };
    for ([_]Odd{
        .{ .odd1 = null, .odd2 = .kit_cuda, .rank = 2, .field = .kit_cuda, .text = "rank 2's configuration differs from rank 0's in kit_cuda" },
        .{ .odd1 = .tokenizer, .odd2 = .engine_commit, .rank = 1, .field = .tokenizer, .text = "rank 1's configuration differs from rank 0's in tokenizer" },
        // (another Engram revision on rank 1: still common, still named, while every rank's shards differ)
        .{ .odd1 = .engram, .odd2 = null, .rank = 1, .field = .engram, .text = "rank 1's configuration differs from rank 0's in engram" },
    }) |c| {
        var s = try newPair();
        defer s.finish();
        const Setup = struct {
            var odd: [2]?fingerprint.Field = .{ null, null };
            fn f(p: *Peer) void {
                // (the followers' templates differ from rank 0's too: never the field named)
                p.fx = rankFx(@intCast(p.rank), 0xb0, odd[p.rank - 1]);
            }
        };
        Setup.odd = .{ c.odd1, c.odd2 };
        try s.start(3, Setup);
        s.leader.fx = rankFx(0, 0xa0, null);
        try testing.expectError(error.FingerprintMismatch, s.leader.agree(s.sink(), s.ls()));
        try testing.expectEqual(c.rank, s.leader.broken.?.rank);
        try testing.expectEqual(@intFromEnum(c.field), s.leader.broken.?.field);
        try testing.expectEqualStrings(c.text, describeBroken(s.leader.broken.?, &buf));
        try testing.expectError(error.PairBroken, s.leader.post(s.sink(), 0, .{ .touch = .{ .point = @splat(1) } }));
        stopAll(&s);
        for (s.peers[0..2], [_]?fingerprint.Field{ c.odd1, c.odd2 }) |peer, odd| {
            // a follower whose own reading of rank 0's hello differs ends itself; one that matched waits, then stops
            try testing.expectEqual(@as(?anyerror, if (odd != null) error.FingerprintMismatch else null), peer.err);
            try testing.expect(!peer.side.agreed_seen);
        }
    }
}
