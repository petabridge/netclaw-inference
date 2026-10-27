//! Port of wojo's GLM 0118 (PR #152, the KV-shard placement core at fork main 8641c90: parts, borrowed twins, the
//! cross-owner carry and the part-salted spill identity) to the native DeepSeek-V4.1 lanes: the pure host-side
//! ownership planner. No CUDA, no I/O, no clock: rank 0 plans, applies the plan's operation list and sends the same
//! list to the followers, which apply it to their own planner and so hold the same parts (round 2's lanes.zig encodes
//! the list on the wire and runs the transfers it names).
//!
//! With conversation ownership the shared compressed pool (Model.pool_cap positions) is cut into `parts` equal parts,
//! one a rank: part p is [p * part_rows, (p + 1) * part_rows) and its compressed codes and scales live on rank p only.
//! An extent never crosses a part's end. What lives in a part:
//!   - live extents: a stream's reservation [base, base + size), sized up front (prompt + max_new + a round's rows,
//!     chunk-aligned: lanes.prefillFn's `size`), never grown (native has no GLM GROW);
//!   - kept states: the prompt cache's saved points (lanes.Snap), rows [base, base + at) held while they live, and
//!     the extent of the stream that kept them, [base, end), held back from placement while room allows (placeSoft's
//!     growth slack, so the conversation's next turn resumes in place);
//!   - twins: a filling prompt's borrowed copy of its rows in every other part (all or none, free rows only), so the
//!     other ranks attend locally; given back before the first decode round, at release, or when their rows are
//!     needed, never while a prompt chunk reads them (pinned);
//!   - holds: a reserved destination nothing else may use (a spill restore landing; a live move in flight).
//!
//! Placement keeps the native policy inside a part (lanes.placeSoft's three tiers, placeKept's oldest-first victims
//! that never take rows a live stream holds, staleOver, and 629c524's shared-prefix guard, keptOver) and chooses the
//! part GLM's way: no eviction anywhere first, the part with fewer live streams then more free rows; else the one part
//! whose needed evictions are the oldest; never a placement that evicts and still fails. A cache hit is carried to
//! another part (a byte transfer between the two owners; the source stays) only to avoid a concrete cost: its own part
//! would drop it (`rescued`, the prompt would run from 0), would evict kept states while another part has the room
//! free with a block of headroom for each stream there and the arriving one (`pressure`), would evict another
//! conversation's longer state where the copy has no free rows (`others`), or would make it wait for the live stream
//! holding it (`paused`). Never to even the parts out, never under carry_min tokens, never rows something pins. A live
//! stream moves whole (its reservation and the kept states inside it) to another part only when a request would
//! otherwise wait for room, at most once per relocate_hold rounds, never straight back, and never out of a part
//! another stream just entered. With one part (the switch off) every decision is the native one, byte for byte.
//!
//! Planning never mutates (plan* take *const); apply() checks and applies a plan's ops atomically (a failure leaves the
//! planner as it was); counters count at apply, once (a refusal once a reason a request). The source of a copy or carry
//! in flight is held until finish: no room, no victim, never staled, dropped or moved. Every op that names an existing extent carries its generation,
//! and every op that makes one carries the generation it gets, so a stale or out-of-order frame fails closed.
//!
//! GLM's Python check (tools/kv_shard_check.py) and its mutation runner (tools/kv_shard_mutants.sh) are the tests at
//! the bottom: one suite of checks, run unmutated (all pass, no section raises) and once per Mutant (each must fail
//! the check written for it, `killer`; a section that raises is a crash, not a kill).
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

/// A pool block: model.chunk_rows (kept self-contained so `zig test` runs this file alone).
// (lanes.zig asserts parts.chunk_rows == model.chunk_rows and parts.growth_rows == growth_rows.)
pub const chunk_rows: usize = 2048;
/// lanes.growth_rows: room left past a kept state's extent while placements fit so.
pub const growth_rows: usize = 16 * chunk_rows;
/// The fewest rows a carry moves (GLM CARRY_MIN): one block; a smaller state is cheaper to prefill again.
pub const carry_min: usize = 2048;
/// Rounds (Planner.tick) a moved live stream stays put, may not go back, and in which no stream leaves a part another
/// one entered (GLM RELOCATE_HOLD).
pub const relocate_hold: i64 = 256;
/// The descriptor layout (spill identity and the wire's extent records carry it).
pub const layout_version: u16 = 1;
pub const max_parts = 8;
pub const no_slot: u32 = std.math.maxInt(u32);
const never: i64 = -(1 << 40);

/// Why a carry or a move was not made (GLM carry.refused, counted once a decision).
pub const Refusal = enum(u3) { pinned, twin, loading, partial, no_target, small };
/// Which rule chose an admission's carry.
pub const Reason = enum(u2) { rescued, pressure, others, paused };

pub const Kind = enum(u8) { live = 1, kept = 2, twin = 3, hold = 4 };

/// An extent as the spill identity and the follower see it.
pub const Desc = struct {
    version: u16 = layout_version,
    kind: Kind,
    id: u64, // the stream (live, twin), the kept state, the hold
    part: u8, // its owner: rank `part` holds its compressed rows
    base: usize,
    reserved: usize, // rows held from base: a live stream's size, a kept state's extent (end - base), a twin's, a hold's
    valid: usize, // rows with content: a kept state's `at`, a live stream's position (rank 0's)
    slot: u32 = no_slot,
    generation: u64,
    stale: bool = false,
};

pub const Live = struct {
    stream: u64,
    slot: u32,
    part: u8,
    base: usize,
    size: usize,
    gen: u64,
    valid: usize = 0, // rank 0 (setValid): the stream's cache length
    loading: bool = false, // a copy, carry or move into or of it in flight
    partial: bool = false, // rank 0 (setPartial): a prompt chunk in flight (its twins pinned)
    decoding: bool = false, // rank 0 (setDecoding)
    moved_at: i64 = never, // the tick it last moved to another part in
    came_from: ?u8 = null, // ...and the part it left then
};

pub const Kept = struct {
    id: u64,
    part: u8,
    base: usize,
    at: usize,
    end: usize,
    gen: u64,
    stale: bool = false,
    pending: bool = false, // rank 0 (setPending): spill writes still read its rows
};

pub const Twin = struct { stream: u64, part: u8, base: usize, size: usize, gen: u64 };

pub const HoldWhy = enum(u8) { restore = 1, move = 2 };
pub const Hold = struct { id: u64, why: HoldWhy, part: u8, base: usize, size: usize, gen: u64 };

const FlightKind = enum(u8) { copy, carry, move };
const Flight = struct { stream: u64, kind: FlightKind, rows: usize, kept: u64 = 0, dst: usize = 0, dst_part: u8 = 0 };

/// What rank 0 sends the followers, in order (rank_wire's shard_ops). copy: the kept rows into a new extent of the same
/// part (the source stays); carry: the same into another part, a transfer between the two owners (the source stays;
/// the destination is the stream's and counts as loading until finish); move: a live extent to another base with its
/// kept states (the source rows freed on finish; across parts a transfer between the owners, its spill writes staged
/// first). stage: a kept state's waiting spill writes copied to the host before its rows change owner or are
/// overwritten.
// rank_wire encodes a plan's ops as shard_ops frames sent before any transfer they name; the follower decodes and
// applies them on its own planner, then runs the transfers in the same order.
pub const Op = union(enum) {
    stage: struct { kept: u64, gen: u64 },
    stale: struct { kept: u64, gen: u64 },
    untwin: struct { stream: u64, gen: u64, revoke: bool },
    admit: struct { stream: u64, slot: u32, part: u8, base: usize, size: usize, gen: u64 },
    @"resume": struct { kept: u64, gen: u64, stream: u64 },
    copy: struct { kept: u64, kept_gen: u64, stream: u64, src: usize, dst: usize, rows: usize },
    carry: struct { kept: u64, kept_gen: u64, stream: u64, src: usize, src_part: u8, dst: usize, dst_part: u8, rows: usize, why: Reason },
    move: struct { stream: u64, gen: u64, src: usize, src_part: u8, dst: usize, dst_part: u8, size: usize, rows: usize, hold_gen: u64, tick: i64 },
    twin: struct { stream: u64, part: u8, base: usize, size: usize, gen: u64 },
    keep: struct { kept: u64, stream: u64, part: u8, base: usize, at: usize, end: usize, gen: u64 },
    drop: struct { kept: u64, gen: u64 },
    release: struct { stream: u64, gen: u64 },
    hold: struct { id: u64, part: u8, base: usize, size: usize, gen: u64 },
    restored: struct { id: u64, gen: u64, kept: u64, at: usize, kept_gen: u64 },
    unhold: struct { id: u64, gen: u64 },
};

pub const Mode = enum { fresh, in_place, copy, carry };
pub const Status = enum { placed, full, relocate };

/// An admission (or a smaller decision) as rank 0 made it, not yet applied. `full`: no room now (lanes'
/// error.ContextFull: the host tries again); `relocate`: apply it, run the move, finish it, then plan the request again.
pub const Plan = struct {
    status: Status = .full,
    mode: Mode = .fresh,
    stream: u64 = 0,
    part: u8 = 0,
    base: usize = 0,
    size: usize = 0,
    source: ?u64 = null, // the kept state it resumes from
    dropped: bool = false, // a hit given up: the prompt runs from 0 (lanes' reuse_failed)
    why: ?Reason = null,
    refused: [6]bool = @splat(false),
    no_part: bool = false,
    reloc_why: ?Refusal = null, // a relocation refused (counted once a reason a request)
    restore: bool = false, // planRestore's (its refusals are not a request's)
    ops: std.ArrayList(Op) = .empty,

    pub fn deinit(p: *Plan, gpa: Allocator) void {
        p.ops.deinit(gpa);
    }

    fn refuse(p: *Plan, r: Refusal) void {
        p.refused[@intFromEnum(r)] = true;
    }
};

/// What a request asks of the planner (lanes.prefillFn's inputs).
pub const Request = struct {
    stream: u64,
    slot: u32,
    size: usize, // the reservation: alignForward(len + max_new + round_rows + 2, chunk_rows)
    source: ?u64 = null, // the kept state it resumes from (lanes checked usable())
    patient: bool = false, // held(source) and lanes' held_wait clock still runs (lanes.patient)
    twins: bool = true, // a fresh prompt may borrow twins
};

pub const Counters = struct {
    hits: u64 = 0, // admissions that carried a kept state into another part
    rescued: u64 = 0,
    pressure: u64 = 0,
    others: u64 = 0,
    paused: u64 = 0,
    live: u64 = 0, // live streams moved to another part
    tokens: u64 = 0, // tokens carried or moved across parts
    bytes: u64 = 0, // ...and the bytes the owners sent (Config.carry_bytes a token)
    failed: u64 = 0, // transfers that finished failed
    staged: u64 = 0,
    staled: u64 = 0,
    twins_made: u64 = 0,
    twins_revoked: u64 = 0,
    refused: [6]u64 = @splat(0), // rank 0, by Refusal
    no_part: u64 = 0, // rank 0: admissions no part could make room for

    pub fn reason(c: *Counters, r: Reason) *u64 {
        return switch (r) {
            .rescued => &c.rescued,
            .pressure => &c.pressure,
            .others => &c.others,
            .paused => &c.paused,
        };
    }

    pub fn refusals(c: *const Counters, r: Refusal) u64 {
        return c.refused[@intFromEnum(r)];
    }
};

/// Faithful mutants of the guards (tests only: the suite must fail on each; GLM's kv_shard_mutants.sh).
pub const Mutant = enum {
    none,
    evict_any_part,
    cross_part,
    no_part_end_inplace,
    no_growth_slack,
    no_prefix_guard,
    failed_mutates,
    pressure_no_eviction_condition,
    pressure_no_headroom,
    pressure_may_evict,
    small_ignored,
    small_uses_size,
    min_0,
    rescue_off,
    others_off,
    counted_at_plan,
    never_counted,
    carry_rows_minus_1,
    carry_before_admit,
    pins_no_loading,
    pins_no_pinned,
    pins_no_twin,
    pins_no_partial,
    no_cooldown,
    move_back_allowed,
    no_entered_guard,
    hold_1,
    no_cheapest,
    refusal_recounted,
    no_headroom_in_move,
    no_stage_on_move,
    no_stage_on_stale,
    twin_not_all_or_none,
    twin_evicts_kept,
    twin_pinned_room,
    twin_kept_at_decode,
    salt_stacks,
    salt_unsharded,
    no_salt,
    no_generation_check,
    copy_across_parts,
    carry_when_off,
    carry_releases_source,
    move_keeps_source,
    order_ignores_live,
    cheapest_ignored,
    flight_source_unprotected,
    move_end_unclamped,
};

pub const Config = struct {
    cap: usize, // pool positions over every part (Model.pool_cap)
    parts: u8 = 1, // 1: unsharded (TF_DS_KV_SHARD off); else one a rank
    carry: bool = true, // carries and moves across parts (GLM TF_GLM_KV_SHARD_CARRY)
    twins: bool = true, // borrowed twins for filling prompts
    // lanes.carryBytes over the model's kv sources: the owner-only planes' bytes a token (four KV sources' codes and
    // scales, (3/2 + 1) x 288 = 720 at this checkpoint); index planes copy locally.
    carry_bytes: usize = 0, // bytes the owners send a carried token (the sharded planes' rows)
    mutant: Mutant = .none,

    pub fn check(c: Config) !void {
        if (c.parts == 0 or c.parts > max_parts) return error.BadParts;
        if (c.cap == 0 or c.cap % (@as(usize, c.parts) * chunk_rows) != 0) return error.BadPoolSize;
    }

    pub fn partRows(c: Config) usize {
        return c.cap / c.parts;
    }
};

/// The pool every rank can hold (GLM shard.agree): the smallest rank's capacity in whole parts of whole blocks, so
/// both ranks cut the same parts. Refuses a part below one block.
pub fn agree(caps: []const usize, parts: u8) !usize {
    if (caps.len == 0 or parts == 0) return error.BadParts;
    var m: usize = std.math.maxInt(usize);
    for (caps) |c| m = @min(m, c);
    const unit = @as(usize, parts) * chunk_rows;
    const cap = m / unit * unit;
    if (cap == 0) return error.PoolTooSmall;
    return cap;
}

/// The native kept-point rule (Lanes.usable, without its liveness): 0 < at < len, a block end, before the replay.
pub fn eligible(at: usize, len: usize, window: usize) bool {
    return at > 0 and at < len and at % chunk_rows == 0 and at <= len -| window;
}

/// Rows of a plane of `ratio` tokens a row that hold tokens [base, base + n), base ratio-aligned: the last row may be
/// partly filled and travels whole (GLM's partial pooled-key row).
pub const Span = struct { row: usize, rows: usize };
pub fn planeSpan(base: usize, n: usize, ratio: usize) !Span {
    if (ratio == 0) return error.BadRatio;
    if (base % ratio != 0) return error.Unaligned;
    return .{ .row = base / ratio, .rows = (n + ratio - 1) / ratio };
}

/// The first chunk-aligned base in [lo, hi) for `size` rows that overlaps none of `used` (lanes.place in a part).
pub fn placeIn(used: []const [2]usize, size: usize, lo: usize, hi: usize) ?usize {
    var base = lo;
    while (base + size <= hi) {
        var clash: ?usize = null;
        for (used) |u| {
            if (base < u[1] and u[0] < base + size) clash = @max(clash orelse 0, u[1]);
        }
        const past = clash orelse return base;
        base = std.mem.alignForward(usize, past, chunk_rows);
    }
    return null;
}

/// One of placeSoft's tiers in [lo, hi): 0 holds kept states' extents plus growth_rows back, 1 their extents, 2 their
/// rows only. `kept`: {base, at, end}.
pub fn placeTier(gpa: Allocator, hard: []const [2]usize, kept: []const [3]usize, size: usize, lo: usize, hi: usize, tier: u2) !?usize {
    var used: std.ArrayList([2]usize) = .empty;
    defer used.deinit(gpa);
    try used.ensureTotalCapacity(gpa, hard.len + kept.len);
    used.appendSliceAssumeCapacity(hard);
    for (kept) |k| {
        const rows_end = k[0] + k[1];
        const ext = @max(k[2], rows_end);
        used.appendAssumeCapacity(.{ k[0], switch (tier) {
            0 => @min(ext + growth_rows, @max(hi, ext)),
            1 => ext,
            else => rows_end,
        } });
    }
    return placeIn(used.items, size, lo, hi);
}

// -- spill identity: the part salt -----------------------------------------------------------------------------------

/// GLM spill_blocks.in_part: `lineage` ("head|fork@at|...") for rows in part `part`: the first computation's name
/// carries the part ("head#p"), replacing one it has (never stacked); the forks after it are untouched. Unsharded
/// stores never call it.
pub fn inPart(buf: []u8, lineage: []const u8, part: u8) ![]u8 {
    return inPartM(buf, lineage, part, .none);
}

fn inPartM(buf: []u8, lineage: []const u8, part: u8, mutant: Mutant) ![]u8 {
    const bar = std.mem.indexOfScalar(u8, lineage, '|') orelse lineage.len;
    const head = lineage[0..bar];
    const plain = if (mutant == .salt_stacks) head else head[0 .. std.mem.indexOfScalar(u8, head, '#') orelse head.len];
    return std.fmt.bufPrint(buf, "{s}#{d}{s}", .{ plain, part, lineage[bar..] });
}

/// The same salt for a digest lineage (spill-store's chained digests): the salted root is always derived from the
/// unsalted one, so re-salting replaces the part and never stacks. `part` null: the root unchanged (unsharded names
/// stay what they were).
pub const Root = struct {
    plain: [32]u8,
    part: ?u8 = null,

    pub fn inPart(r: Root, part: ?u8) Root {
        return .{ .plain = r.plain, .part = part };
    }

    pub fn digest(r: Root) [32]u8 {
        const p = r.part orelse return r.plain;
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update("tf-dsv41-part\x00");
        h.update(&r.plain);
        h.update(&[_]u8{p});
        return h.finalResult();
    }
};

// -- the planner ------------------------------------------------------------------------------------------------------

const Target = struct { part: u8, base: usize };
const Order = struct { ids: [max_parts]u8 = undefined, n: usize = 0 };
const Opts = struct { twins_free: bool = false, gone: ?u64 = null };
const Sim = struct { base: ?usize = null, twins_free: bool = false, source_lost: bool = false, wait: bool = false };
const HitOutcome = enum { placed, wait, full, fresh };

/// The ops of a plan being built, and the generations the extents it makes will get.
const Emit = struct {
    plan: *Plan,
    gpa: Allocator,
    gen: u64,

    fn op(e: *Emit, o: Op) !void {
        try e.plan.ops.append(e.gpa, o);
    }

    fn next(e: *Emit) u64 {
        const g = e.gen;
        e.gen += 1;
        return g;
    }
};

pub const Planner = struct {
    cfg: Config,
    part_rows: usize,
    lives: std.ArrayList(Live) = .empty,
    kept: std.ArrayList(Kept) = .empty, // oldest first (lanes.snaps' order: the victims' order)
    twins: std.ArrayList(Twin) = .empty,
    holds: std.ArrayList(Hold) = .empty,
    flights: std.ArrayList(Flight) = .empty,
    next_gen: u64 = 1,
    ticks: i64 = 0, // rank 0: decode rounds (tick)
    entered: [max_parts]i64 = @splat(never),
    counters: Counters = .{},
    /// Rank 0: the reasons already counted for a request not yet placed, by stream (GLM lane.xwhy, kept per request
    /// so requests waiting in turn do not recount one another's): bit r a Refusal, bit no_part_bit `no_part`.
    seen: std.ArrayList(Seen) = .empty,

    const Seen = struct { stream: u64, bits: u8 };
    const no_part_bit: u3 = 6;

    pub fn init(cfg: Config) !Planner {
        try cfg.check();
        return .{ .cfg = cfg, .part_rows = cfg.partRows() };
    }

    pub fn deinit(self: *Planner, gpa: Allocator) void {
        self.lives.deinit(gpa);
        self.kept.deinit(gpa);
        self.twins.deinit(gpa);
        self.holds.deinit(gpa);
        self.flights.deinit(gpa);
        self.seen.deinit(gpa);
    }

    /// A copy of the extents (rank 0's `seen` is not copied: applyOps hands it over).
    fn clone(self: *const Planner, gpa: Allocator) !Planner {
        var c = self.*;
        c.lives = .empty;
        c.kept = .empty;
        c.twins = .empty;
        c.holds = .empty;
        c.flights = .empty;
        c.seen = .empty;
        errdefer c.deinit(gpa);
        c.lives = try self.lives.clone(gpa);
        c.kept = try self.kept.clone(gpa);
        c.twins = try self.twins.clone(gpa);
        c.holds = try self.holds.clone(gpa);
        c.flights = try self.flights.clone(gpa);
        return c;
    }

    inline fn mut(self: *const Planner, m: Mutant) bool {
        return builtin.is_test and self.cfg.mutant == m;
    }

    fn sharded(self: *const Planner) bool {
        return self.cfg.parts > 1;
    }

    fn carrying(self: *const Planner) bool {
        return self.sharded() and (self.cfg.carry or self.mut(.carry_when_off));
    }

    fn holdRounds(self: *const Planner) i64 {
        return if (self.mut(.hold_1)) 1 else relocate_hold;
    }

    pub fn lo(self: *const Planner, p: u8) usize {
        return @as(usize, p) * self.part_rows;
    }

    /// A part's end as placement sees it.
    fn hi(self: *const Planner, p: u8) usize {
        return if (self.mut(.cross_part)) self.cfg.cap else self.partEnd(p);
    }

    pub fn partEnd(self: *const Planner, p: u8) usize {
        return self.lo(p) + self.part_rows;
    }

    pub fn partOf(self: *const Planner, row: usize) u8 {
        return @intCast(@min(row / self.part_rows, self.cfg.parts - 1));
    }

    // -- lookups ------------------------------------------------------------------------------------------------------

    pub fn liveIndex(self: *const Planner, stream: u64) ?usize {
        for (self.lives.items, 0..) |l, i| if (l.stream == stream) return i;
        return null;
    }

    pub fn keptIndex(self: *const Planner, id: u64) ?usize {
        for (self.kept.items, 0..) |k, i| if (k.id == id) return i;
        return null;
    }

    fn holdIndex(self: *const Planner, why: HoldWhy, id: u64) ?usize {
        for (self.holds.items, 0..) |h, i| if (h.why == why and h.id == id) return i;
        return null;
    }

    fn flightIndex(self: *const Planner, stream: u64) ?usize {
        for (self.flights.items, 0..) |f, i| if (f.stream == stream) return i;
        return null;
    }

    /// A kept state's view (lanes copies base, end and part back into its Snap after finish moves it).
    pub fn keptAt(self: *const Planner, id: u64) ?Kept {
        const i = self.keptIndex(id) orelse return null;
        return self.kept.items[i];
    }

    /// Whether an in-flight copy or carry reads kept state `id`: its rows stay, unmoved and unstaled, until finish.
    pub fn readBy(self: *const Planner, id: u64) bool {
        for (self.flights.items) |f| if (f.kind != .move and f.kept == id) return true;
        return false;
    }

    /// readBy as planning honours it (a test mutant drops the guard; applyOps always checks it).
    fn reading(self: *const Planner, id: u64) bool {
        return !self.mut(.flight_source_unprotected) and self.readBy(id);
    }

    /// Whether a kept state an in-flight copy or carry reads, `except` aside, has rows in [a, b).
    fn clashReading(self: *const Planner, a: usize, b: usize, except: ?usize) bool {
        for (self.kept.items, 0..) |k, i| {
            if (i == except or k.stale) continue;
            if (k.base < b and a < k.base + k.at and self.reading(k.id)) return true;
        }
        return false;
    }

    /// Whether a live extent holds a kept state an in-flight copy or carry reads.
    fn readsIn(self: *const Planner, l: Live) bool {
        for (self.kept.items) |k| {
            if (!k.stale and k.part == l.part and l.base <= k.base and k.base < l.base + l.size and self.reading(k.id)) return true;
        }
        return false;
    }

    pub fn hasTwins(self: *const Planner, stream: u64) bool {
        for (self.twins.items) |t| if (t.stream == stream) return true;
        return false;
    }

    /// Whether a twin is pinned: its stream has a prompt chunk in flight that reads it.
    fn twinPinned(self: *const Planner, t: Twin) bool {
        const i = self.liveIndex(t.stream) orelse return false;
        return self.lives.items[i].partial;
    }

    fn clashLive(self: *const Planner, a: usize, b: usize, gone: ?u64) bool {
        for (self.lives.items) |l| if (gone != l.stream and a < l.base + l.size and l.base < b) return true;
        return false;
    }

    fn clashHolds(self: *const Planner, a: usize, b: usize) bool {
        for (self.holds.items) |h| if (a < h.base + h.size and h.base < b) return true;
        return false;
    }

    fn clashPinned(self: *const Planner, a: usize, b: usize) bool {
        for (self.twins.items) |t| if (self.twinPinned(t) and a < t.base + t.size and t.base < b) return true;
        return false;
    }

    /// The live stream whose extent holds a kept state's rows (the stream that kept it, or one that resumed it in place).
    fn holder(self: *const Planner, k: Kept) ?usize {
        for (self.lives.items, 0..) |l, i| if (l.base <= k.base and k.base < l.base + l.size) return i;
        return null;
    }

    /// Whether a kept state's rows lie in a live stream's extent (lanes: that stream holds the room).
    pub fn held(self: *const Planner, id: u64) bool {
        const i = self.keptIndex(id) orelse return false;
        const k = self.kept.items[i];
        return !k.stale and self.clashLive(k.base, k.base + k.at, null);
    }

    // -- room ---------------------------------------------------------------------------------------------------------

    /// The extent of the live stream a relocation would move away (its kept states go with it).
    fn goneRange(self: *const Planner, o: Opts) ?[2]usize {
        const s = o.gone orelse return null;
        const l = self.lives.items[self.liveIndex(s) orelse return null];
        return .{ l.base, l.base + l.size };
    }

    fn hard(self: *const Planner, gpa: Allocator, out: *std.ArrayList([2]usize), o: Opts) !void {
        out.clearRetainingCapacity();
        for (self.lives.items) |l| if (o.gone != l.stream) try out.append(gpa, .{ l.base, l.base + l.size });
        for (self.holds.items) |h| try out.append(gpa, .{ h.base, h.base + h.size });
        for (self.twins.items) |t| {
            if (o.twins_free and (self.mut(.twin_pinned_room) or !self.twinPinned(t))) continue;
            try out.append(gpa, .{ t.base, t.base + t.size });
        }
        // the rows an in-flight copy or carry reads are no room at any tier (but those that would leave with `gone`:
        // its move is refused as `pinned`, so a relocation still sees why it is wanted)
        const gone = self.goneRange(o);
        for (self.kept.items) |k| {
            if (k.stale or !self.reading(k.id)) continue;
            if (gone) |g| if (g[0] <= k.base and k.base < g[1]) continue;
            try out.append(gpa, .{ k.base, k.base + k.at });
        }
    }

    /// The non-stale kept states of part p (lanes.placeKept's `kept`), {base, at, end}.
    fn keptOf(self: *const Planner, gpa: Allocator, out: *std.ArrayList([3]usize), p: u8, stale: []const bool, o: Opts) !void {
        out.clearRetainingCapacity();
        const gone = self.goneRange(o);
        for (self.kept.items, 0..) |k, i| {
            if (k.stale or stale[i] or k.part != p) continue;
            if (gone) |g| if (g[0] <= k.base and k.base < g[1]) continue; // (they move with it)
            try out.append(gpa, .{ k.base, k.at, k.end });
        }
    }

    fn tierOn(self: *const Planner, tier: u2) bool {
        return !(tier == 0 and self.mut(.no_growth_slack));
    }

    /// lanes.placeSoft in part p.
    fn placeSoftIn(self: *const Planner, gpa: Allocator, p: u8, size: usize, o: Opts, stale: []const bool) !?usize {
        var h: std.ArrayList([2]usize) = .empty;
        defer h.deinit(gpa);
        var k: std.ArrayList([3]usize) = .empty;
        defer k.deinit(gpa);
        try self.hard(gpa, &h, o);
        try self.keptOf(gpa, &k, p, stale, o);
        var tier: u2 = 0;
        while (tier < 3) : (tier += 1) {
            if (!self.tierOn(tier)) continue;
            if (try placeTier(gpa, h.items, k.items, size, self.lo(p), self.hi(p), tier)) |b| return b;
        }
        return null;
    }

    fn liveIn(self: *const Planner, p: u8) usize {
        var n: usize = 0;
        for (self.lives.items) |l| n += @intFromBool(l.part == p);
        return n;
    }

    /// The rows of part p nothing holds: no live stream, twin, hold or kept state's rows.
    pub fn freeIn(self: *const Planner, gpa: Allocator, p: u8) !usize {
        var spans: std.ArrayList([2]usize) = .empty;
        defer spans.deinit(gpa);
        try self.hard(gpa, &spans, .{});
        for (self.kept.items) |k| if (!k.stale) try spans.append(gpa, .{ k.base, k.base + k.at });
        return self.part_rows - covered(spans.items, self.lo(p), self.partEnd(p));
    }

    /// The parts in GLM's order: fewer live streams, then more free rows, then the lower index.
    fn order(self: *const Planner, gpa: Allocator) !Order {
        var o: Order = .{};
        var live: [max_parts]usize = undefined;
        var free: [max_parts]usize = undefined;
        for (0..self.cfg.parts) |q| {
            live[q] = if (self.mut(.order_ignores_live)) 0 else self.liveIn(@intCast(q));
            free[q] = try self.freeIn(gpa, @intCast(q));
            // insertion by (live, -free, q)
            var j = o.n;
            while (j > 0) : (j -= 1) {
                const r = o.ids[j - 1];
                if (live[r] < live[q] or (live[r] == live[q] and free[r] >= free[q])) break;
                o.ids[j] = r;
            }
            o.ids[j] = @intCast(q);
            o.n += 1;
        }
        return o;
    }

    fn headroom(self: *const Planner, q: u8) usize {
        return chunk_rows * (self.liveIn(q) + 1);
    }

    /// The oldest kept state of part p that may give its room: not stale, not `protect`, not in a live stream's extent,
    /// not read by an in-flight copy or carry.
    fn victim(self: *const Planner, p: u8, stale: []const bool, protect: ?usize) ?usize {
        for (self.kept.items, 0..) |k, i| {
            if (k.stale or stale[i] or i == protect) continue;
            if (!self.mut(.evict_any_part) and k.part != p) continue;
            if (self.clashLive(k.base, k.base + k.at, null)) continue;
            if (self.reading(k.id)) continue;
            return i;
        }
        return null;
    }

    /// lanes.placeKept in part p, simulated: victims oldest first, `source` last; with `patient` and the source held by
    /// a live stream, nothing evicted and the copy must leave room for another of its size (else wait).
    fn simPlaceKept(self: *const Planner, gpa: Allocator, p: u8, size: usize, source: ?usize, patient: bool, stale: []bool, victims: *std.ArrayList(usize)) !Sim {
        const is_held = if (source) |si| blk: {
            const k = self.kept.items[si];
            break :blk !stale[si] and self.clashLive(k.base, k.base + k.at, null);
        } else false;
        while (true) {
            var got = try self.placeSoftIn(gpa, p, size, .{}, stale);
            var tf = false;
            if (got == null) {
                got = try self.placeSoftIn(gpa, p, size, .{ .twins_free = true }, stale);
                tf = got != null;
            }
            if (is_held and patient) {
                const b = got orelse return .{ .wait = true };
                return if (try self.roomAfter(gpa, p, b, size, tf, stale)) .{ .base = b, .twins_free = tf } else .{ .wait = true };
            }
            if (got) |b| return .{ .base = b, .twins_free = tf, .source_lost = if (source) |si| stale[si] else false };
            const v = self.victim(p, stale, source) orelse blk: {
                const si = source orelse return .{};
                const k = self.kept.items[si];
                if (stale[si] or k.stale or self.clashLive(k.base, k.base + k.at, null) or self.reading(k.id)) return .{};
                break :blk si;
            };
            stale[v] = true;
            try victims.append(gpa, v);
        }
    }

    /// lanes.roomAfter in part p: with `size` taken at b, another extent of `size` still fits.
    fn roomAfter(self: *const Planner, gpa: Allocator, p: u8, b: usize, size: usize, tf: bool, stale: []const bool) !bool {
        var rows: std.ArrayList([2]usize) = .empty;
        defer rows.deinit(gpa);
        try self.hard(gpa, &rows, .{ .twins_free = tf });
        for (self.kept.items, 0..) |k, i| if (!k.stale and !stale[i] and k.part == p) try rows.append(gpa, .{ k.base, k.base + k.at });
        try rows.append(gpa, .{ b, b + size });
        return placeIn(rows.items, size, self.lo(p), self.hi(p)) != null;
    }

    /// Evicting the oldest kept states of part q until `size` fits there: the base, the victims (appended) and the cost
    /// (the newest victim's place in the kept order, plus one; 0 when none is needed). Null when even all leave too little.
    fn evictCost(self: *const Planner, gpa: Allocator, q: u8, size: usize, protect: ?usize, stale: []const bool, victims: *std.ArrayList(usize)) !?struct { base: usize, cost: usize } {
        const local = try gpa.dupe(bool, stale);
        defer gpa.free(local);
        const mark = victims.items.len;
        var cost: usize = 0;
        while (true) {
            if (try self.placeSoftIn(gpa, q, size, .{ .twins_free = true }, local)) |b| return .{ .base = b, .cost = cost };
            const v = self.victim(q, local, protect) orelse {
                victims.shrinkRetainingCapacity(mark);
                return null;
            };
            local[v] = true;
            try victims.append(gpa, v);
            cost = v + 1;
        }
    }

    /// The part where making `size` rows evicts only the oldest kept states (GLM _cheaper_half), `skip` aside.
    fn cheapest(self: *const Planner, gpa: Allocator, size: usize, skip: ?u8, protect: ?usize, stale: []const bool, victims: *std.ArrayList(usize)) !?Target {
        var best: ?struct { t: Target, cost: usize, v: []usize } = null;
        defer if (best) |b| gpa.free(b.v);
        var scratch: std.ArrayList(usize) = .empty;
        defer scratch.deinit(gpa);
        for (0..self.cfg.parts) |qi| {
            const q: u8 = @intCast(qi);
            if (q == skip) continue;
            scratch.clearRetainingCapacity();
            const r = try self.evictCost(gpa, q, size, protect, stale, &scratch) orelse continue;
            const better = if (best) |b| (if (self.mut(.cheapest_ignored)) false else r.cost < b.cost) else true;
            if (better) {
                const v = try gpa.dupe(usize, scratch.items);
                if (best) |b| gpa.free(b.v);
                best = .{ .t = .{ .part = q, .base = r.base }, .cost = r.cost, .v = v };
            }
        }
        const b = best orelse return null;
        try victims.appendSlice(gpa, b.v);
        return b.t;
    }

    /// Free rows (no eviction, twins kept) for `size` in a part other than p, the lightest first; with `head`, plus a
    /// block for each live stream there and the arriving one.
    fn freeTarget(self: *const Planner, gpa: Allocator, p: u8, size: usize, head: bool, stale: []const bool) !?Target {
        const o = try self.order(gpa);
        for (o.ids[0..o.n]) |q| {
            if (q == p) continue;
            const want = size + if (head) self.headroom(q) else 0;
            if (want > self.part_rows) continue;
            if (try self.placeSoftIn(gpa, q, want, .{}, stale)) |b| return .{ .part = q, .base = b };
        }
        return null;
    }

    /// Whether a kept state other than `except` and its shorter siblings has rows in [a, b) (lanes.keptOver: 629c524).
    fn keptOver(self: *const Planner, a: usize, b: usize, except: usize, stale: []const bool) bool {
        const e = self.kept.items[except];
        for (self.kept.items, 0..) |k, i| {
            if (i == except or k.stale or stale[i]) continue;
            if (k.base == e.base and k.at <= e.at) continue;
            if (k.base < b and a < k.base + k.at) return true;
        }
        return false;
    }

    /// lanes.staleOver: the kept states with rows in [a, b), `except` aside, given up (appended, marked). Rows an
    /// in-flight copy or carry reads are no room at any tier and no victim, so no placement reaches them: one that
    /// does is a planning bug, refused (error.InFlight) rather than planned over the rows being read.
    fn staleOver(self: *const Planner, gpa: Allocator, a: usize, b: usize, except: ?usize, stale: []bool, victims: *std.ArrayList(usize)) !void {
        for (self.kept.items, 0..) |k, i| {
            if (i == except or k.stale or stale[i]) continue;
            if (k.base < b and a < k.base + k.at) {
                if (self.reading(k.id)) return error.InFlight;
                stale[i] = true;
                try victims.append(gpa, i);
            }
        }
    }

    /// Why a live stream's rows cannot move or be carried from now (GLM _pins), or null. GLM's names: `twin` for a
    /// stream holding twins (mid-fill or not), `partial` for one with a prompt chunk in flight. GLM's `pinned` (the
    /// extent is itself a pinned twin) has no native case, as a kept state never lies in a twin: see pinsMove.
    fn pinsLive(self: *const Planner, l: Live) ?Refusal {
        if (l.loading and !self.mut(.pins_no_loading)) return .loading;
        if (self.hasTwins(l.stream) and !self.mut(.pins_no_twin)) return .twin;
        if (l.partial and !self.mut(.pins_no_partial)) return .partial;
        return null;
    }

    /// pinsLive for a move, plus `pinned`: a kept state in its extent is read by an in-flight copy or carry (a reader
    /// holds the rows the move would free). Reading does not stop a carry of the same rows, so pinsKept leaves it out.
    fn pinsMove(self: *const Planner, l: Live) ?Refusal {
        if (self.pinsLive(l)) |w| return w;
        if (!self.mut(.pins_no_pinned) and self.readsIn(l)) return .pinned;
        return null;
    }

    fn pinsKept(self: *const Planner, si: usize) ?Refusal {
        const li = self.holder(self.kept.items[si]) orelse return null;
        return self.pinsLive(self.lives.items[li]);
    }

    /// Rows of a live extent in use: its position and the kept states in it (rank 0's count: GLM _used(pending=False)).
    fn usedRows(self: *const Planner, li: usize) usize {
        const l = self.lives.items[li];
        var n = l.valid;
        for (self.kept.items) |k| if (!k.stale and k.part == l.part and l.base <= k.base and k.base < l.base + l.size) {
            n = @max(n, k.base - l.base + k.at);
        };
        return @min(n, l.size);
    }

    // -- planning (rank 0, read only) ---------------------------------------------------------------------------------

    /// Where a request's extent goes and how its rows get there (lanes.prefillFn's placement), without changing anything.
    pub fn planAdmit(self: *const Planner, gpa: Allocator, req: Request) !Plan {
        if (req.size == 0 or req.size % chunk_rows != 0) return error.BadSize;
        if (req.size > self.part_rows) return error.PromptTooLong;
        if (self.liveIndex(req.stream) != null) return error.StreamLive;
        var plan: Plan = .{ .stream = req.stream, .size = req.size };
        errdefer plan.deinit(gpa);
        var e: Emit = .{ .plan = &plan, .gpa = gpa, .gen = self.next_gen };
        const stale = try gpa.alloc(bool, self.kept.items.len);
        defer gpa.free(stale);
        @memset(stale, false);
        var victims: std.ArrayList(usize) = .empty;
        defer victims.deinit(gpa);
        var fresh = true;
        if (req.source) |id| {
            const si = self.keptIndex(id) orelse return error.UnknownKept;
            if (req.size <= self.kept.items[si].at) return error.BadSize; // (len > at: the reservation holds the rows)
            if (self.kept.items[si].stale) {
                plan.dropped = true;
            } else switch (try self.planHit(gpa, req, si, &e, stale, &victims)) {
                .placed, .wait, .full => fresh = false,
                .fresh => {
                    plan.dropped = true;
                    plan.ops.clearRetainingCapacity();
                    e.gen = self.next_gen;
                    @memset(stale, false);
                    victims.clearRetainingCapacity();
                },
            }
        }
        if (fresh) try self.planFresh(gpa, req, &e, stale, &victims);
        if (plan.status == .full and self.mut(.failed_mutates)) {
            // the native placeKept's way: the victims it took stay stale though nothing was placed
            for (victims.items) |v| {
                const k = self.kept.items[v];
                try e.op(.{ .stale = .{ .kept = k.id, .gen = k.gen } });
            }
        }
        if (self.mut(.counted_at_plan)) if (plan.why) |w| if (plan.mode == .carry) {
            const c = &@constCast(self).counters;
            c.reason(w).* += 1;
            c.hits += 1;
        };
        return plan;
    }

    fn planHit(self: *const Planner, gpa: Allocator, req: Request, si: usize, e: *Emit, stale: []bool, victims: *std.ArrayList(usize)) !HitOutcome {
        const plan = e.plan;
        const sn = self.kept.items[si];
        const p = sn.part;
        const end = sn.base + req.size;
        // in place unless the extent runs past its part, a live stream (or a hold, or a pinned twin) has rows there, or
        // another kept state has rows past the kept ones (lanes.prefillFn's `moved`)
        const end_over = end > (if (self.mut(.no_part_end_inplace)) self.cfg.cap else self.partEnd(p));
        const live_over = self.clashLive(sn.base, end, null) or self.clashHolds(sn.base, end) or self.clashPinned(sn.base, end) or
            self.clashReading(sn.base + sn.at, end, si);
        const kept_over = !self.mut(.no_prefix_guard) and self.keptOver(sn.base + sn.at, end, si, stale);
        if (!end_over and !live_over and !kept_over) {
            try self.emitPlaced(gpa, e, req, victims, stale, .{ .part = p, .base = sn.base, .stale_from = sn.base + sn.at, .except = si, .twins_free = true, .transfer = .{ .@"resume" = .{ .kept = sn.id, .gen = sn.gen, .stream = req.stream } } });
            plan.mode = .in_place;
            plan.source = sn.id;
            return .placed;
        }
        const by_kept = !end_over and !live_over and kept_over;
        const r = try self.simPlaceKept(gpa, p, req.size, si, req.patient, stale, victims);
        const cheap = r.base != null and victims.items.len == 0 and !r.source_lost;
        if (cheap and !self.mut(.pressure_no_eviction_condition)) return self.hitCopy(gpa, e, req, si, r, victims, stale);
        if (self.carrying()) carry: {
            const why: Reason = if (r.wait) .paused else if (r.base == null or r.source_lost) .rescued else if (by_kept) .others else .pressure;
            if (why == .rescued and self.mut(.rescue_off)) break :carry;
            if (why == .others and self.mut(.others_off)) break :carry;
            const min: usize = if (self.mut(.min_0)) 0 else carry_min;
            const small = if (self.mut(.small_uses_size)) req.size < min else sn.at < min;
            if (small and !self.mut(.small_ignored)) {
                plan.refuse(.small);
                break :carry;
            }
            if (self.pinsKept(si)) |w| {
                plan.refuse(w);
                break :carry;
            }
            const none = try gpa.alloc(bool, self.kept.items.len);
            defer gpa.free(none);
            @memset(none, false);
            var tv: std.ArrayList(usize) = .empty;
            defer tv.deinit(gpa);
            const t: ?Target = switch (why) {
                .others => try self.freeTarget(gpa, p, req.size, false, none),
                .pressure, .paused => if (self.mut(.pressure_may_evict))
                    try self.cheapest(gpa, req.size, p, si, none, &tv)
                else
                    try self.freeTarget(gpa, p, req.size, !self.mut(.pressure_no_headroom), none),
                .rescued => (try self.freeTarget(gpa, p, req.size, true, none)) orelse try self.cheapest(gpa, req.size, p, si, none, &tv),
            };
            const tg = t orelse {
                plan.refuse(.no_target);
                break :carry;
            };
            for (tv.items) |v| none[v] = true;
            const rows = if (self.mut(.carry_rows_minus_1)) sn.at - 1 else sn.at;
            try self.emitPlaced(gpa, e, req, &tv, none, .{ .part = tg.part, .base = tg.base, .stale_from = tg.base, .except = si, .twins_free = true, .transfer = .{ .carry = .{ .kept = sn.id, .kept_gen = sn.gen, .stream = req.stream, .src = sn.base, .src_part = p, .dst = tg.base, .dst_part = tg.part, .rows = rows, .why = why } } });
            plan.mode = .carry;
            plan.why = why;
            plan.source = sn.id;
            return .placed;
        }
        if (cheap) return self.hitCopy(gpa, e, req, si, r, victims, stale);
        if (r.wait) {
            plan.status = .full;
            return .wait;
        }
        if (r.base) |b| {
            if (!r.source_lost) return self.hitCopy(gpa, e, req, si, r, victims, stale);
            if (!self.sharded()) {
                // native: the source gave its room too, so the prompt runs from 0 where it was placed
                plan.dropped = true;
                try self.emitPlaced(gpa, e, req, victims, stale, .{ .part = p, .base = b, .stale_from = b, .twins_free = r.twins_free });
                return .placed;
            }
        } else if (!self.sharded()) {
            plan.status = .full;
            return .full;
        }
        return .fresh;
    }

    fn hitCopy(self: *const Planner, gpa: Allocator, e: *Emit, req: Request, si: usize, r: Sim, victims: *std.ArrayList(usize), stale: []bool) !HitOutcome {
        const sn = self.kept.items[si];
        const b = r.base.?;
        try self.emitPlaced(gpa, e, req, victims, stale, .{ .part = sn.part, .base = b, .stale_from = b, .except = si, .twins_free = r.twins_free, .transfer = .{ .copy = .{ .kept = sn.id, .kept_gen = sn.gen, .stream = req.stream, .src = sn.base, .dst = b, .rows = sn.at } } });
        e.plan.mode = .copy;
        e.plan.source = sn.id;
        return .placed;
    }

    fn planFresh(self: *const Planner, gpa: Allocator, req: Request, e: *Emit, stale: []bool, victims: *std.ArrayList(usize)) !void {
        const plan = e.plan;
        const o = try self.order(gpa);
        var found: ?Target = null;
        var tf = false;
        // no eviction: the tiers over the parts in order (twins kept, then given back), then the cheapest part's evictions
        for ([_]bool{ false, true }) |free_twins| {
            var tier: u2 = 0;
            outer: while (tier < 3) : (tier += 1) {
                if (!self.tierOn(tier)) continue;
                for (o.ids[0..o.n]) |q| {
                    if (try self.placeTierIn(gpa, q, req.size, .{ .twins_free = free_twins }, stale, tier)) |b| {
                        found = .{ .part = q, .base = b };
                        tf = free_twins;
                        break :outer;
                    }
                }
            }
            if (found != null) break;
        }
        if (found == null) {
            found = try self.cheapest(gpa, req.size, null, null, stale, victims);
            tf = true;
            for (victims.items) |v| stale[v] = true;
        }
        if (found) |t| {
            try self.emitPlaced(gpa, e, req, victims, stale, .{ .part = t.part, .base = t.base, .stale_from = t.base, .twins_free = tf, .twins = self.sharded() and self.cfg.twins and req.twins });
            plan.mode = .fresh;
            return;
        }
        plan.status = .full;
        if (self.sharded()) {
            if (try self.planRelocate(gpa, req, e, stale)) {
                plan.status = .relocate;
                return;
            }
            plan.no_part = true;
        }
    }

    fn placeTierIn(self: *const Planner, gpa: Allocator, p: u8, size: usize, o: Opts, stale: []const bool, tier: u2) !?usize {
        var h: std.ArrayList([2]usize) = .empty;
        defer h.deinit(gpa);
        var k: std.ArrayList([3]usize) = .empty;
        defer k.deinit(gpa);
        try self.hard(gpa, &h, o);
        try self.keptOf(gpa, &k, p, stale, o);
        return placeTier(gpa, h.items, k.items, size, self.lo(p), self.hi(p), tier);
    }

    const Placed = struct {
        part: u8,
        base: usize,
        stale_from: usize,
        except: ?usize = null,
        twins_free: bool = false,
        transfer: ?Op = null,
        twins: bool = false,
    };

    /// The ops of a placement, in order: staged writes and evictions (the victims, then what staleOver finds), the
    /// twins given back, the extent, its transfer, its own twins.
    fn emitPlaced(self: *const Planner, gpa: Allocator, e: *Emit, req: Request, victims: *std.ArrayList(usize), stale: []bool, a: Placed) !void {
        const plan = e.plan;
        const end = a.base + req.size;
        try self.staleOver(gpa, a.stale_from, end, a.except, stale, victims);
        try self.emitStale(e, victims.items);
        if (a.twins_free) try self.emitUntwins(e, a.base, end);
        const admit: Op = .{ .admit = .{ .stream = req.stream, .slot = req.slot, .part = a.part, .base = a.base, .size = req.size, .gen = 0 } };
        if (self.mut(.carry_before_admit) and a.transfer != null and a.transfer.? == .carry) {
            try e.op(a.transfer.?);
            var ad = admit;
            ad.admit.gen = e.next();
            try e.op(ad);
        } else {
            var ad = admit;
            ad.admit.gen = e.next();
            try e.op(ad);
            if (a.transfer) |t| try e.op(t);
        }
        plan.status = .placed;
        plan.part = a.part;
        plan.base = a.base;
        if (a.twins) try self.planTwins(gpa, e, req, a.part, stale);
    }

    fn emitStale(self: *const Planner, e: *Emit, victims: []const usize) !void {
        for (victims) |v| {
            const k = self.kept.items[v];
            if (k.pending and !self.mut(.no_stage_on_stale)) try e.op(.{ .stage = .{ .kept = k.id, .gen = k.gen } });
            try e.op(.{ .stale = .{ .kept = k.id, .gen = k.gen } });
        }
    }

    /// The revocable twins in [a, b) given back, each stream's all at once.
    fn emitUntwins(self: *const Planner, e: *Emit, a: usize, b: usize) !void {
        var done: [64]u64 = undefined;
        var n: usize = 0;
        for (self.twins.items) |t| {
            if (!(a < t.base + t.size and t.base < b)) continue;
            if (self.twinPinned(t) and !self.mut(.twin_pinned_room)) continue;
            if (std.mem.indexOfScalar(u64, done[0..n], t.stream) != null) continue;
            if (n == done.len) return error.TooManyTwins;
            done[n] = t.stream;
            n += 1;
            const l = self.lives.items[self.liveIndex(t.stream) orelse return error.UnknownStream];
            try e.op(.{ .untwin = .{ .stream = t.stream, .gen = l.gen, .revoke = true } });
        }
    }

    /// A fresh prompt's twins: one in every other part, free rows only (no kept state gives room), all or none.
    fn planTwins(self: *const Planner, gpa: Allocator, e: *Emit, req: Request, p: u8, stale: []bool) !void {
        var bases: [max_parts]?usize = @splat(null);
        var all = true;
        var h: std.ArrayList([2]usize) = .empty;
        defer h.deinit(gpa);
        for (0..self.cfg.parts) |qi| {
            const q: u8 = @intCast(qi);
            if (q == p) continue;
            bases[qi] = if (self.mut(.twin_evicts_kept)) blk: {
                try self.hard(gpa, &h, .{});
                break :blk placeIn(h.items, req.size, self.lo(q), self.hi(q));
            } else try self.placeSoftIn(gpa, q, req.size, .{}, stale);
            if (bases[qi] == null) all = false;
        }
        if (!all and !self.mut(.twin_not_all_or_none)) return;
        var v: std.ArrayList(usize) = .empty;
        defer v.deinit(gpa);
        for (0..self.cfg.parts) |qi| {
            const b = bases[qi] orelse continue;
            if (self.mut(.twin_evicts_kept)) {
                v.clearRetainingCapacity();
                try self.staleOver(gpa, b, b + req.size, null, stale, &v);
                try self.emitStale(e, v.items);
            }
            try e.op(.{ .twin = .{ .stream = req.stream, .part = @intCast(qi), .base = b, .size = req.size, .gen = e.next() } });
        }
    }

    /// A request no part has room for would wait: move one live stream whole to a part with room for it (and a
    /// block of headroom a stream there plus one), the cheapest whose leaving lets the request in without evicting, if
    /// it may move (nothing pins it, not within relocate_hold rounds of its last move, not back where it came from,
    /// not out of a part another stream just entered). One move a plan.
    fn planRelocate(self: *const Planner, gpa: Allocator, req: Request, e: *Emit, stale: []const bool) !bool {
        if (!self.carrying()) return false;
        const plan = e.plan;
        const hold = self.holdRounds();
        var best: ?struct { li: usize, rows: usize, t: Target } = null;
        var pin: ?Refusal = null;
        var wanted = false;
        for (0..self.cfg.parts) |pi| {
            const p: u8 = @intCast(pi);
            if (!self.mut(.no_entered_guard) and self.ticks - self.entered[pi] < hold) continue;
            for (self.lives.items, 0..) |l, li| {
                if (l.part != p or l.stream == req.stream) continue;
                if (try self.placeSoftIn(gpa, p, req.size, .{ .gone = l.stream }, stale) == null) continue;
                wanted = true;
                if (self.pinsMove(l)) |w| {
                    if (pin == null) pin = w;
                    continue;
                }
                if (!l.decoding) continue;
                if (!self.mut(.no_cooldown) and self.ticks - l.moved_at < hold) continue;
                const t = try self.moveTo(gpa, l, stale) orelse continue;
                const rows = self.usedRows(li);
                const better = if (best) |b| (if (self.mut(.no_cheapest)) rows > b.rows else rows < b.rows) else true;
                if (better) best = .{ .li = li, .rows = rows, .t = t };
            }
        }
        const b = best orelse {
            if (wanted) plan.reloc_why = pin orelse .no_target;
            return false;
        };
        const l = self.lives.items[b.li];
        if (!self.mut(.no_stage_on_move)) for (self.kept.items) |k| {
            if (!k.stale and k.pending and k.part == l.part and l.base <= k.base and k.base < l.base + l.size)
                try e.op(.{ .stage = .{ .kept = k.id, .gen = k.gen } });
        };
        try e.op(.{ .move = .{ .stream = l.stream, .gen = l.gen, .src = l.base, .src_part = l.part, .dst = b.t.base, .dst_part = b.t.part, .size = l.size, .rows = b.rows, .hold_gen = e.next(), .tick = self.ticks } });
        return true;
    }

    /// The part a live stream would move to: room for its whole reservation and headroom, never back into the part it
    /// left within relocate_hold rounds; the lightest first.
    fn moveTo(self: *const Planner, gpa: Allocator, l: Live, stale: []const bool) !?Target {
        const o = try self.order(gpa);
        for (o.ids[0..o.n]) |q| {
            if (q == l.part) continue;
            if (!self.mut(.move_back_allowed) and l.came_from == q and self.ticks - l.moved_at < self.holdRounds()) continue;
            const want = l.size + if (self.mut(.no_headroom_in_move)) 0 else self.headroom(q);
            if (want > self.part_rows) continue;
            if (try self.placeSoftIn(gpa, q, want, .{}, stale)) |b| return .{ .part = q, .base = b };
        }
        return null;
    }

    /// A spill restore's destination in the part its point was written from (GLM: back into the recorded half),
    /// evicting there only; kept invisible (a hold) until restoredOp or unholdOp.
    pub fn planRestore(self: *const Planner, gpa: Allocator, id: u64, size: usize, part: u8) !Plan {
        if (part >= self.cfg.parts) return error.BadPart;
        if (size == 0 or size % chunk_rows != 0 or size > self.part_rows) return error.BadSize;
        if (self.holdIndex(.restore, id) != null) return error.HoldExists;
        var plan: Plan = .{ .size = size, .part = part, .restore = true };
        errdefer plan.deinit(gpa);
        var e: Emit = .{ .plan = &plan, .gpa = gpa, .gen = self.next_gen };
        const stale = try gpa.alloc(bool, self.kept.items.len);
        defer gpa.free(stale);
        @memset(stale, false);
        var victims: std.ArrayList(usize) = .empty;
        defer victims.deinit(gpa);
        var found = try self.placeSoftIn(gpa, part, size, .{}, stale);
        if (found == null) {
            const r = try self.evictCost(gpa, part, size, null, stale, &victims) orelse {
                plan.no_part = self.sharded();
                return plan;
            };
            found = r.base;
            for (victims.items) |v| stale[v] = true;
        }
        const b = found.?;
        try self.staleOver(gpa, b, b + size, null, stale, &victims);
        try self.emitStale(&e, victims.items);
        try self.emitUntwins(&e, b, b + size);
        try e.op(.{ .hold = .{ .id = id, .part = part, .base = b, .size = size, .gen = e.next() } });
        plan.status = .placed;
        plan.base = b;
        return plan;
    }

    pub fn restoredOp(self: *const Planner, id: u64, kept: u64, at: usize) !Op {
        const h = self.holds.items[self.holdIndex(.restore, id) orelse return error.UnknownHold];
        return .{ .restored = .{ .id = id, .gen = h.gen, .kept = kept, .at = at, .kept_gen = self.next_gen } };
    }

    pub fn unholdOp(self: *const Planner, id: u64) !Op {
        const h = self.holds.items[self.holdIndex(.restore, id) orelse return error.UnknownHold];
        return .{ .unhold = .{ .id = id, .gen = h.gen } };
    }

    /// The live stream's state at `at` kept (lanes.snapSaveFn).
    pub fn keepOp(self: *const Planner, id: u64, stream: u64, at: usize) !Op {
        const l = self.lives.items[self.liveIndex(stream) orelse return error.UnknownStream];
        return .{ .keep = .{ .kept = id, .stream = stream, .part = l.part, .base = l.base, .at = at, .end = l.base + l.size, .gen = self.next_gen } };
    }

    /// A kept state dropped (lanes.snapDropFn). error.InFlight while a copy or carry reads it: lanes drops it after
    /// that stream's finish.
    pub fn dropOp(self: *const Planner, id: u64) !Op {
        const k = self.kept.items[self.keptIndex(id) orelse return error.UnknownKept];
        if (self.readBy(id)) return error.InFlight;
        return .{ .drop = .{ .kept = id, .gen = k.gen } };
    }

    /// Before a stream's first decode round: its twins given back (decode reads the owner's rows only).
    pub fn planDecode(self: *const Planner, gpa: Allocator, stream: u64) !Plan {
        const l = self.lives.items[self.liveIndex(stream) orelse return error.UnknownStream];
        var plan: Plan = .{ .stream = stream, .status = .placed };
        errdefer plan.deinit(gpa);
        if (self.hasTwins(stream) and !self.mut(.twin_kept_at_decode)) try plan.ops.append(gpa, .{ .untwin = .{ .stream = stream, .gen = l.gen, .revoke = false } });
        return plan;
    }

    /// A stream ends: its twins, then its extent (its kept states stay).
    pub fn planRelease(self: *const Planner, gpa: Allocator, stream: u64) !Plan {
        const l = self.lives.items[self.liveIndex(stream) orelse return error.UnknownStream];
        if (self.flightIndex(stream) != null) return error.InFlight;
        var plan: Plan = .{ .stream = stream, .status = .placed };
        errdefer plan.deinit(gpa);
        if (self.hasTwins(stream)) try plan.ops.append(gpa, .{ .untwin = .{ .stream = stream, .gen = l.gen, .revoke = false } });
        try plan.ops.append(gpa, .{ .release = .{ .stream = stream, .gen = l.gen } });
        return plan;
    }

    /// Whether a request of `size` rows could be placed now in some part, evicting every kept state that may go and
    /// giving back every twin that may (pinned twins, live streams and holds are no room).
    pub fn fits(self: *const Planner, gpa: Allocator, size: usize) !bool {
        var h: std.ArrayList([2]usize) = .empty;
        defer h.deinit(gpa);
        try self.hard(gpa, &h, .{ .twins_free = true });
        for (0..self.cfg.parts) |qi| {
            const q: u8 = @intCast(qi);
            if (placeIn(h.items, size, self.lo(q), self.hi(q)) != null) return true;
        }
        return false;
    }

    // -- rank 0's own facts -------------------------------------------------------------------------------------------

    pub fn tick(self: *Planner) void {
        self.ticks += 1;
    }

    pub fn setValid(self: *Planner, stream: u64, n: usize) void {
        if (self.liveIndex(stream)) |i| self.lives.items[i].valid = @min(n, self.lives.items[i].size);
    }

    pub fn setPartial(self: *Planner, stream: u64, on: bool) void {
        if (self.liveIndex(stream)) |i| self.lives.items[i].partial = on;
    }

    pub fn setDecoding(self: *Planner, stream: u64, on: bool) void {
        if (self.liveIndex(stream)) |i| self.lives.items[i].decoding = on;
    }

    pub fn setPending(self: *Planner, id: u64, on: bool) void {
        if (self.keptIndex(id)) |i| self.kept.items[i].pending = on;
    }

    // -- applying (every rank) ----------------------------------------------------------------------------------------

    /// Rank 0: a plan's ops applied, then its refusals counted, each reason once a request (a request that waits and
    /// is planned again, or another request planned in between, counts nothing twice; placed, it starts over). Every
    /// plan is applied, `full` ones too (no ops: only the counting). All or nothing.
    pub fn apply(self: *Planner, gpa: Allocator, plan: *const Plan) !void {
        try self.seen.ensureUnusedCapacity(gpa, 1);
        try self.applyOps(gpa, plan.ops.items);
        var bits: u8 = 0;
        for (plan.refused, 0..) |r, i| bits |= @as(u8, @intFromBool(r)) << @intCast(i);
        if (plan.reloc_why) |w| bits |= @as(u8, 1) << @intFromEnum(w);
        if (plan.no_part) bits |= @as(u8, 1) << no_part_bit;
        if (plan.restore) {
            // a restore is no request: its rare refusal counts each time
            self.countBits(bits, 0);
            return;
        }
        const found = for (self.seen.items, 0..) |s, i| {
            if (s.stream == plan.stream) break i;
        } else null;
        const had: u8 = if (found) |i| self.seen.items[i].bits else 0;
        self.countBits(bits, if (self.mut(.refusal_recounted)) 0 else had);
        if (plan.status == .placed) {
            if (found) |i| _ = self.seen.swapRemove(i);
        } else if (found) |i| {
            self.seen.items[i].bits = had | bits;
        } else if (bits != 0) {
            self.seen.appendAssumeCapacity(.{ .stream = plan.stream, .bits = bits });
        }
    }

    fn countBits(self: *Planner, bits: u8, had: u8) void {
        const new = bits & ~had;
        for (0..self.counters.refused.len) |i| self.counters.refused[i] += (new >> @intCast(i)) & 1;
        self.counters.no_part += (new >> no_part_bit) & 1;
    }

    /// Rank 0: a waiting request given up (its client gone) forgets what was counted for it.
    pub fn forget(self: *Planner, stream: u64) void {
        for (self.seen.items, 0..) |s, i| if (s.stream == stream) {
            _ = self.seen.swapRemove(i);
            return;
        };
    }

    /// Every rank: rank 0's ops in order, checked; all or nothing (a failure leaves the planner as it was).
    pub fn applyOps(self: *Planner, gpa: Allocator, ops: []const Op) !void {
        var c = try self.clone(gpa);
        errdefer c.deinit(gpa);
        for (ops) |op| try c.applyOp(gpa, op);
        c.seen = self.seen;
        self.seen = .empty;
        self.deinit(gpa);
        self.* = c;
    }

    fn sameGen(self: *const Planner, have: u64, want: u64) !void {
        if (have != want and !self.mut(.no_generation_check)) return error.StaleGeneration;
    }

    fn newGen(self: *Planner, g: u64) !void {
        if (g != self.next_gen) return error.OutOfStep;
        self.next_gen += 1;
    }

    /// [base, base + size) a whole, aligned range of part `part`, clear of live streams, holds and twins (`skip`
    /// aside) and, with `kept_too`, of kept rows.
    fn checkRange(self: *const Planner, part: u8, base: usize, size: usize, skip: ?u64, kept_too: bool) !void {
        if (part >= self.cfg.parts) return error.BadPart;
        if (size == 0 or base % chunk_rows != 0 or size % chunk_rows != 0) return error.BadRange;
        if (base < self.lo(part) or base + size > self.partEnd(part)) return error.CrossesPart;
        const end = base + size;
        for (self.lives.items) |l| if (l.stream != skip and base < l.base + l.size and l.base < end) return error.Overlap;
        for (self.holds.items) |h| if (base < h.base + h.size and h.base < end) return error.Overlap;
        for (self.twins.items) |t| if (base < t.base + t.size and t.base < end) return error.Overlap;
        if (kept_too) for (self.kept.items) |k| if (!k.stale and base < k.base + k.at and k.base < end) return error.Overlap;
    }

    fn applyOp(self: *Planner, gpa: Allocator, op: Op) !void {
        const c = &self.counters;
        switch (op) {
            .stage => |s| {
                const k = &self.kept.items[self.keptIndex(s.kept) orelse return error.UnknownKept];
                try self.sameGen(k.gen, s.gen);
                k.pending = false;
                c.staged += 1;
            },
            .stale => |s| {
                const k = &self.kept.items[self.keptIndex(s.kept) orelse return error.UnknownKept];
                try self.sameGen(k.gen, s.gen);
                if (k.pending) return error.Unstaged; // its spill writes still read the rows
                if (self.readBy(s.kept)) return error.InFlight; // a copy or carry still reads the rows
                k.stale = true;
                c.staled += 1;
            },
            .untwin => |u| {
                const l = self.lives.items[self.liveIndex(u.stream) orelse return error.UnknownStream];
                try self.sameGen(l.gen, u.gen);
                if (!self.hasTwins(u.stream)) return error.NoTwin;
                // a chunk reads them (rank 0 knows); a release or a stream's first decode round is never mid-chunk
                if (u.revoke and l.partial) return error.Pinned;
                var i: usize = 0;
                while (i < self.twins.items.len) {
                    if (self.twins.items[i].stream == u.stream) _ = self.twins.orderedRemove(i) else i += 1;
                }
                if (u.revoke) c.twins_revoked += 1;
            },
            .admit => |a| {
                if (self.liveIndex(a.stream) != null) return error.StreamLive;
                for (self.lives.items) |l| if (l.slot == a.slot and a.slot != no_slot) return error.SlotTaken;
                try self.checkRange(a.part, a.base, a.size, null, false);
                // kept rows it overlaps are given up first (stale ops), but for the state it resumes in place and its
                // shorter siblings (same base): an admission over live kept rows is a plan that forgot staleOver
                for (self.kept.items) |k| if (!k.stale and k.base != a.base and a.base < k.base + k.at and k.base < a.base + a.size) return error.Overlap;
                try self.newGen(a.gen);
                try self.lives.append(gpa, .{ .stream = a.stream, .slot = a.slot, .part = a.part, .base = a.base, .size = a.size, .gen = a.gen });
            },
            .@"resume" => |r| {
                const k = &self.kept.items[self.keptIndex(r.kept) orelse return error.UnknownKept];
                try self.sameGen(k.gen, r.gen);
                const l = &self.lives.items[self.liveIndex(r.stream) orelse return error.UnknownStream];
                if (k.stale or l.base != k.base or l.part != k.part) return error.BadOp;
                k.end = @max(k.end, l.base + l.size);
                l.valid = k.at;
            },
            .copy => |x| {
                const k = self.kept.items[self.keptIndex(x.kept) orelse return error.UnknownKept];
                try self.sameGen(k.gen, x.kept_gen);
                const l = &self.lives.items[self.liveIndex(x.stream) orelse return error.UnknownStream];
                if (k.part != l.part and !self.mut(.copy_across_parts)) return error.BadOp; // across parts: a carry
                if (k.stale or x.src != k.base or x.dst != l.base or x.rows > k.at or x.rows > l.size) return error.BadOp;
                l.loading = true;
                try self.flights.append(gpa, .{ .stream = x.stream, .kind = .copy, .rows = x.rows, .kept = x.kept });
            },
            .carry => |x| {
                const k = self.kept.items[self.keptIndex(x.kept) orelse return error.UnknownKept];
                try self.sameGen(k.gen, x.kept_gen);
                const l = &self.lives.items[self.liveIndex(x.stream) orelse return error.UnknownStream];
                if (k.part == l.part and !self.mut(.copy_across_parts)) return error.BadOp; // in one part: a copy
                if (k.stale or x.src != k.base or x.src_part != k.part or x.dst != l.base or x.dst_part != l.part or x.rows > k.at or x.rows > l.size) return error.BadOp;
                l.loading = true;
                try self.flights.append(gpa, .{ .stream = x.stream, .kind = .carry, .rows = x.rows, .kept = x.kept });
                if (!self.mut(.never_counted) and !self.mut(.counted_at_plan)) {
                    c.hits += 1;
                    c.reason(x.why).* += 1;
                }
                c.tokens += x.rows;
                c.bytes += x.rows * self.cfg.carry_bytes;
            },
            .move => |m| {
                const l = &self.lives.items[self.liveIndex(m.stream) orelse return error.UnknownStream];
                try self.sameGen(l.gen, m.gen);
                if (self.hasTwins(m.stream) or l.loading) return error.Pinned;
                if (m.src != l.base or m.src_part != l.part or m.size != l.size or m.rows > m.size) return error.BadOp;
                try self.checkRange(m.dst_part, m.dst, m.size, m.stream, true);
                for (self.kept.items) |k| if (!k.stale and k.part == l.part and l.base <= k.base and k.base < l.base + l.size) {
                    if (k.pending) return error.Unstaged;
                    if (self.readBy(k.id)) return error.InFlight; // a copy or carry reads rows the move would free
                };
                try self.newGen(m.hold_gen);
                try self.holds.append(gpa, .{ .id = m.stream, .why = .move, .part = m.dst_part, .base = m.dst, .size = m.size, .gen = m.hold_gen });
                l.loading = true;
                try self.flights.append(gpa, .{ .stream = m.stream, .kind = .move, .rows = m.rows, .dst = m.dst, .dst_part = m.dst_part });
                if (m.dst_part != m.src_part) {
                    l.moved_at = m.tick;
                    l.came_from = m.src_part;
                    self.entered[m.dst_part] = m.tick;
                    c.live += 1;
                    c.tokens += m.rows;
                    c.bytes += m.rows * self.cfg.carry_bytes;
                }
            },
            .twin => |t| {
                const l = self.lives.items[self.liveIndex(t.stream) orelse return error.UnknownStream];
                if (t.part == l.part or t.size != l.size) return error.BadOp;
                try self.checkRange(t.part, t.base, t.size, null, true);
                try self.newGen(t.gen);
                if (!self.hasTwins(t.stream)) c.twins_made += 1;
                try self.twins.append(gpa, .{ .stream = t.stream, .part = t.part, .base = t.base, .size = t.size, .gen = t.gen });
            },
            .keep => |k| {
                const l = self.lives.items[self.liveIndex(k.stream) orelse return error.UnknownStream];
                if (self.keptIndex(k.kept) != null) return error.KeptExists;
                if (k.at == 0 or k.at % chunk_rows != 0 or k.at > l.size or k.base != l.base or k.part != l.part or k.end != l.base + l.size) return error.BadOp;
                try self.newGen(k.gen);
                try self.kept.append(gpa, .{ .id = k.kept, .part = k.part, .base = k.base, .at = k.at, .end = k.end, .gen = k.gen });
            },
            .drop => |d| {
                const i = self.keptIndex(d.kept) orelse return error.UnknownKept;
                try self.sameGen(self.kept.items[i].gen, d.gen);
                if (self.readBy(d.kept)) return error.InFlight;
                _ = self.kept.orderedRemove(i);
            },
            .release => |r| {
                const i = self.liveIndex(r.stream) orelse return error.UnknownStream;
                try self.sameGen(self.lives.items[i].gen, r.gen);
                if (self.hasTwins(r.stream)) return error.Twinned;
                if (self.flightIndex(r.stream) != null) return error.InFlight;
                _ = self.lives.orderedRemove(i);
            },
            .hold => |h| {
                if (self.holdIndex(.restore, h.id) != null) return error.HoldExists;
                try self.checkRange(h.part, h.base, h.size, null, true);
                try self.newGen(h.gen);
                try self.holds.append(gpa, .{ .id = h.id, .why = .restore, .part = h.part, .base = h.base, .size = h.size, .gen = h.gen });
            },
            .restored => |r| {
                const i = self.holdIndex(.restore, r.id) orelse return error.UnknownHold;
                const h = self.holds.items[i];
                try self.sameGen(h.gen, r.gen);
                if (self.keptIndex(r.kept) != null) return error.KeptExists;
                if (r.at == 0 or r.at % chunk_rows != 0 or r.at > h.size) return error.BadOp;
                try self.newGen(r.kept_gen);
                _ = self.holds.orderedRemove(i);
                try self.kept.append(gpa, .{ .id = r.kept, .part = h.part, .base = h.base, .at = r.at, .end = h.base + h.size, .gen = r.kept_gen });
            },
            .unhold => |u| {
                const i = self.holdIndex(.restore, u.id) orelse return error.UnknownHold;
                try self.sameGen(self.holds.items[i].gen, u.gen);
                _ = self.holds.orderedRemove(i);
            },
        }
    }

    /// Every rank, in the same order: a stream's copy, carry or move done (`ok`) or failed. A copy or carry that
    /// failed leaves the stream's extent with nothing resumed (its prompt runs from 0) and the source as it was; a move
    /// that failed leaves the stream where it was. A move that worked frees its old rows and takes its kept states
    /// along (each a new generation, its spill writes named for the new part from now on).
    ///
    /// `ok` is rank 0's verdict (sent on the wire), never a rank's own transfer result: the ranks must end alike.
    /// Checked before anything changes (an unknown flight, stream or hold changes nothing). After a move that worked,
    /// lanes copies each moved kept state's base, end and part (keptAt) back into its Snap.
    pub fn finish(self: *Planner, stream: u64, ok: bool) !struct { resumed: bool } {
        const fi = self.flightIndex(stream) orelse return error.NoFlight;
        const li = self.liveIndex(stream) orelse return error.UnknownStream;
        const f = self.flights.items[fi];
        const hold_i: ?usize = if (f.kind == .move) (self.holdIndex(.move, stream) orelse return error.UnknownHold) else null;
        _ = self.flights.orderedRemove(fi);
        const l = &self.lives.items[li];
        l.loading = false;
        if (!ok) self.counters.failed += 1;
        switch (f.kind) {
            .copy, .carry => {
                l.valid = if (ok) f.rows else 0;
                if (ok and f.kind == .carry and self.mut(.carry_releases_source)) {
                    if (self.keptIndex(f.kept)) |ki| self.kept.items[ki].stale = true;
                }
            },
            .move => {
                const h = self.holds.orderedRemove(hold_i.?);
                if (ok and !self.mut(.move_keeps_source)) {
                    const old = l.base;
                    for (self.kept.items) |*k| if (k.part == l.part and old <= k.base and k.base < old + l.size) {
                        k.base = k.base - old + f.dst;
                        // an end a larger stream left (resume) stops at the moved extent's: never past the part
                        const end = k.end - old + f.dst;
                        k.end = if (self.mut(.move_end_unclamped)) end else @min(end, f.dst + l.size);
                        k.part = f.dst_part;
                        k.gen = self.next_gen;
                        self.next_gen += 1;
                    };
                    l.base = f.dst;
                    l.part = f.dst_part;
                    l.gen = h.gen;
                }
            },
        }
        return .{ .resumed = ok };
    }

    // -- views --------------------------------------------------------------------------------------------------------

    /// The spill part salt of a kept state's blocks: its part when sharded, null (names unchanged) when not.
    pub fn saltOf(self: *const Planner, id: u64) ?u8 {
        if (self.mut(.no_salt)) return null;
        if (!self.sharded() and !self.mut(.salt_unsharded)) return null;
        const i = self.keptIndex(id) orelse return null;
        return self.kept.items[i].part;
    }

    /// Every extent, in a fixed order (live, kept, twin, hold; each in its list's order).
    pub fn descs(self: *const Planner, gpa: Allocator) ![]Desc {
        var out: std.ArrayList(Desc) = .empty;
        errdefer out.deinit(gpa);
        for (self.lives.items) |l| try out.append(gpa, .{ .kind = .live, .id = l.stream, .part = l.part, .base = l.base, .reserved = l.size, .valid = l.valid, .slot = l.slot, .generation = l.gen });
        for (self.kept.items) |k| try out.append(gpa, .{ .kind = .kept, .id = k.id, .part = k.part, .base = k.base, .reserved = k.end - k.base, .valid = k.at, .generation = k.gen, .stale = k.stale });
        for (self.twins.items) |t| try out.append(gpa, .{ .kind = .twin, .id = t.stream, .part = t.part, .base = t.base, .reserved = t.size, .valid = 0, .generation = t.gen });
        for (self.holds.items) |h| try out.append(gpa, .{ .kind = .hold, .id = h.id, .part = h.part, .base = h.base, .reserved = h.size, .valid = 0, .generation = h.gen });
        return out.toOwnedSlice(gpa);
    }

    /// What every rank must agree on (rank 0's own facts left out): extents, flights, generations, entries.
    pub fn digest(self: *const Planner) [32]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        const put = struct {
            fn f(hh: *std.crypto.hash.sha2.Sha256, v: u64) void {
                var b: [8]u8 = undefined;
                std.mem.writeInt(u64, &b, v, .little);
                hh.update(&b);
            }
        }.f;
        put(&h, layout_version);
        put(&h, self.cfg.parts);
        put(&h, self.cfg.cap);
        put(&h, self.next_gen);
        for (self.lives.items) |l| {
            for ([_]u64{ 1, l.stream, l.slot, l.part, l.base, l.size, l.gen, @intFromBool(l.loading) }) |v| put(&h, v);
        }
        for (self.kept.items) |k| {
            for ([_]u64{ 2, k.id, k.part, k.base, k.at, k.end, k.gen, @intFromBool(k.stale) }) |v| put(&h, v);
        }
        for (self.twins.items) |t| {
            for ([_]u64{ 3, t.stream, t.part, t.base, t.size, t.gen }) |v| put(&h, v);
        }
        for (self.holds.items) |x| {
            for ([_]u64{ 4, x.id, @intFromEnum(x.why), x.part, x.base, x.size, x.gen }) |v| put(&h, v);
        }
        for (self.flights.items) |f| {
            for ([_]u64{ 5, f.stream, @intFromEnum(f.kind), f.rows, f.kept, f.dst, f.dst_part }) |v| put(&h, v);
        }
        for (self.entered) |t| put(&h, @bitCast(t));
        return h.finalResult();
    }

    pub const PartStats = struct { free: usize, live_rows: usize, kept_rows: usize, twin_rows: usize, hold_rows: usize, live: usize };

    /// A part's rows by what holds them (health): kept rows outside live extents only.
    pub fn partStats(self: *const Planner, gpa: Allocator, p: u8) !PartStats {
        var s: PartStats = .{ .free = try self.freeIn(gpa, p), .live_rows = 0, .kept_rows = 0, .twin_rows = 0, .hold_rows = 0, .live = self.liveIn(p) };
        for (self.lives.items) |l| if (l.part == p) {
            s.live_rows += l.size;
        };
        for (self.twins.items) |t| if (t.part == p) {
            s.twin_rows += t.size;
        };
        for (self.holds.items) |h| if (h.part == p) {
            s.hold_rows += h.size;
        };
        var spans: std.ArrayList([2]usize) = .empty;
        defer spans.deinit(gpa);
        for (self.kept.items) |k| if (!k.stale and k.part == p and !self.clashLive(k.base, k.base + k.at, null)) try spans.append(gpa, .{ k.base, k.base + k.at });
        s.kept_rows = covered(spans.items, self.lo(p), self.partEnd(p));
        return s;
    }
};

/// The rows of [lo, hi) that the spans cover (overlaps counted once).
fn covered(spans: [][2]usize, lo: usize, hi: usize) usize {
    std.mem.sort([2]usize, spans, {}, struct {
        fn f(_: void, a: [2]usize, b: [2]usize) bool {
            return a[0] < b[0];
        }
    }.f);
    var n: usize = 0;
    var at = lo;
    for (spans) |s| {
        const a = @max(s[0], at);
        const b = @min(s[1], hi);
        if (b > a) {
            n += b - a;
            at = b;
        }
    }
    return n;
}

// -- tests: GLM's tools/kv_shard_check.py and tools/kv_shard_mutants.sh, adapted to native reservations -----------------
//
// One suite of checks (`suite`), run unmutated (every check passes) and once per Mutant (each must fail its `killer`).
// GLM's GROW cases become the native up-front reservations: a live stream never grows, so a "pause" is a request that
// waits for room (lanes' error.ContextFull), and a live move is how a request avoids it. The decode/prompt kernels'
// equality, the transport and the ranks' collectives are shard-attn's and round 2's (GPU, wire), not checked here.

const C = chunk_rows;

const T = struct {
    gpa: Allocator,
    m: Mutant,
    verbose: bool,
    pass: usize = 0,
    fail: usize = 0,
    crash: usize = 0, // sections that raised (a crash, not a kill: kv_shard_mutants.sh)
    failed: [512][]const u8 = undefined, // the failing checks' names (string literals)
    nfailed: usize = 0,

    fn check(t: *T, name: []const u8, ok: bool) void {
        if (ok) {
            t.pass += 1;
        } else {
            t.fail += 1;
            if (t.nfailed < t.failed.len) {
                t.failed[t.nfailed] = name;
                t.nfailed += 1;
            }
            if (t.verbose) std.debug.print("FAIL {s}\n", .{name});
        }
    }

    fn section(t: *T, name: []const u8, f: *const fn (*T) anyerror!void) void {
        f(t) catch |err| {
            t.crash += 1;
            if (t.verbose) std.debug.print("CRASH {s}: raised {s}\n", .{ name, @errorName(err) });
        };
    }

    /// Whether a check whose name starts with `prefix` failed.
    fn killedBy(t: *const T, prefix: []const u8) bool {
        for (t.failed[0..t.nfailed]) |n| if (std.mem.startsWith(u8, n, prefix)) return true;
        return false;
    }
};

const Fx = struct {
    pl: Planner,
    gpa: Allocator,

    fn init(t: *T, parts: u8, part_rows: usize) !Fx {
        return initCfg(t, .{ .cap = @as(usize, parts) * part_rows, .parts = parts });
    }

    fn initCfg(t: *T, cfg: Config) !Fx {
        var c = cfg;
        c.mutant = t.m;
        c.carry_bytes = 720;
        return .{ .gpa = t.gpa, .pl = try Planner.init(c) };
    }

    fn deinit(f: *Fx) void {
        f.pl.deinit(f.gpa);
    }

    fn gen(f: *Fx) u64 {
        const g = f.pl.next_gen;
        f.pl.next_gen += 1;
        return g;
    }

    fn live(f: *Fx, stream: u64, base: usize, size: usize) !void {
        try f.pl.lives.append(f.gpa, .{ .stream = stream, .slot = @intCast(stream), .part = f.pl.partOf(base), .base = base, .size = size, .gen = f.gen(), .decoding = true });
    }

    fn kept(f: *Fx, id: u64, base: usize, at: usize, end: usize) !void {
        try f.pl.kept.append(f.gpa, .{ .id = id, .part = f.pl.partOf(base), .base = base, .at = at, .end = end, .gen = f.gen() });
    }

    fn twin(f: *Fx, stream: u64, base: usize, size: usize) !void {
        try f.pl.twins.append(f.gpa, .{ .stream = stream, .part = f.pl.partOf(base), .base = base, .size = size, .gen = f.gen() });
    }

    fn hold(f: *Fx, id: u64, base: usize, size: usize) !void {
        try f.pl.holds.append(f.gpa, .{ .id = id, .why = .restore, .part = f.pl.partOf(base), .base = base, .size = size, .gen = f.gen() });
    }

    fn lv(f: *Fx, stream: u64) *Live {
        return &f.pl.lives.items[f.pl.liveIndex(stream).?];
    }

    fn kp(f: *Fx, id: u64) *Kept {
        return &f.pl.kept.items[f.pl.keptIndex(id).?];
    }

    fn plan(f: *Fx, req: Request) !Plan {
        return f.pl.planAdmit(f.gpa, req);
    }

    fn apply(f: *Fx, p: *const Plan) !void {
        try f.pl.apply(f.gpa, p);
    }

    /// Plan and apply; the plan is the caller's to free.
    fn admit(f: *Fx, req: Request) !Plan {
        var p = try f.plan(req);
        errdefer p.deinit(f.gpa);
        try f.apply(&p);
        return p;
    }

    fn release(f: *Fx, stream: u64) !void {
        var p = try f.pl.planRelease(f.gpa, stream);
        defer p.deinit(f.gpa);
        try f.apply(&p);
    }
};

fn rq(stream: u64, size: usize, source: ?u64) Request {
    return .{ .stream = stream, .slot = @intCast(stream), .size = size, .source = source };
}

fn count(p: *const Plan, tag: std.meta.Tag(Op)) usize {
    var n: usize = 0;
    for (p.ops.items) |o| n += @intFromBool(o == tag);
    return n;
}

fn index(p: *const Plan, tag: std.meta.Tag(Op)) ?usize {
    for (p.ops.items, 0..) |o, i| if (o == tag) return i;
    return null;
}

/// The kept states a plan gives up, in order.
fn staled(p: *const Plan, buf: []u64) []u64 {
    var n: usize = 0;
    for (p.ops.items) |o| if (o == .stale and n < buf.len) {
        buf[n] = o.stale.kept;
        n += 1;
    };
    return buf[0..n];
}

fn staledAre(p: *const Plan, want: []const u64) bool {
    var buf: [64]u64 = undefined;
    return std.mem.eql(u64, staled(p, &buf), want);
}

fn carryOf(p: *const Plan) ?@FieldType(Op, "carry") {
    for (p.ops.items) |o| if (o == .carry) return o.carry;
    return null;
}

// 1-8: parts, eviction in one part, placement order, sizes ------------------------------------------------------------

fn secParts(t: *T) !void {
    const P = 64 * C;
    {
        // 1. two parts full of kept states, LRU interleaved: the part whose needed victims are oldest, evictions there only
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.kept(1, 0, 20 * C, 20 * C);
        try f.kept(2, P, 20 * C, P + 20 * C);
        try f.kept(3, 20 * C, 20 * C, 40 * C);
        try f.kept(4, P + 20 * C, 30 * C, P + 50 * C);
        try f.kept(5, 40 * C, 24 * C, 64 * C);
        try f.kept(6, P + 50 * C, 14 * C, P + 64 * C);
        var p = try f.plan(rq(1, 30 * C, null));
        defer p.deinit(t.gpa);
        t.check("1 room in the part losing older states", p.status == .placed and p.part == 0);
        t.check("1 evictions only there (a0, a1), oldest first", staledAre(&p, &.{ 1, 3 }));
        t.check("1 no twin where the other part has no free rows", count(&p, .twin) == 0);
        try f.apply(&p);
    }
    {
        // 1b. the same with the other part's states older: the other part
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.kept(2, P, 20 * C, P + 20 * C);
        try f.kept(1, 0, 20 * C, 20 * C);
        try f.kept(4, P + 20 * C, 20 * C, P + 40 * C);
        try f.kept(3, 20 * C, 30 * C, 50 * C);
        try f.kept(6, P + 40 * C, 24 * C, P + 64 * C);
        try f.kept(5, 50 * C, 14 * C, 64 * C);
        var p = try f.plan(rq(1, 30 * C, null));
        defer p.deinit(t.gpa);
        t.check("1 mirrored: the part whose newest needed victim is oldest (part 1)", p.part == 1 and staledAre(&p, &.{ 2, 4 }));
    }
    {
        // 2. an explicit part (a restore into its recorded part): evicts only there, though the oldest is elsewhere
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.kept(1, 0, 64 * C, 64 * C);
        try f.kept(2, P, 32 * C, P + 32 * C);
        try f.kept(3, P + 32 * C, 32 * C, P + 64 * C);
        f.kp(2).pending = true;
        var p = try f.pl.planRestore(t.gpa, 9, 16 * C, 1);
        defer p.deinit(t.gpa);
        t.check("2 explicit part evicts only its own", p.status == .placed and p.base == P and staledAre(&p, &.{2}));
        t.check("2 a kept state with spill writes waiting is staged before it goes", index(&p, .stage) != null and index(&p, .stage).? < index(&p, .stale).?);
        try f.apply(&p);
        t.check("2 the hold is in place, the rest kept", f.pl.holds.items.len == 1 and !f.kp(1).stale and !f.kp(3).stale and f.kp(2).stale);
    }
    {
        // 3. one pool: the oldest first, as lanes.placeKept
        var f = try Fx.init(t, 1, 128 * C);
        defer f.deinit();
        try f.kept(1, 0, 64 * C, 64 * C);
        try f.kept(2, 64 * C, 32 * C, 96 * C);
        try f.kept(3, 96 * C, 32 * C, 128 * C);
        var p = try f.plan(rq(1, 16 * C, null));
        defer p.deinit(t.gpa);
        t.check("3 one pool: oldest first", staledAre(&p, &.{1}) and p.base == 0);
        t.check("3 one pool: no twins", count(&p, .twin) == 0);
    }
    {
        // 4. a new stream to the part with fewer live streams, then the one with more free rows
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.live(1, 0, 8 * C);
        try f.live(2, 8 * C, 8 * C);
        try f.kept(3, P, 40 * C, P + 40 * C);
        var p = try f.plan(rq(10, 4 * C, null));
        defer p.deinit(t.gpa);
        t.check("4 fewer live streams first", p.part == 1);
        try f.release(1);
        try f.release(2);
        var p2 = try f.plan(rq(10, 4 * C, null));
        defer p2.deinit(t.gpa);
        t.check("4 tie: more free rows", p2.part == 0);
    }
    {
        // 5. (GLM's grow, native: the reservation is up front) resuming in place never runs past the part's end
        var f = try Fx.init(t, 2, 32 * C);
        defer f.deinit();
        try f.kept(7, 24 * C, 8 * C, 32 * C);
        var p = try f.plan(rq(10, 12 * C, 7));
        defer p.deinit(t.gpa);
        t.check("5 past its part's end the hit is copied within the part, not resumed in place", p.mode == .copy and p.part == 0 and p.base + 12 * C <= 32 * C);
        try f.apply(&p);
    }
    {
        // 6. random: evictions in one part, the placement's; none beyond the need
        var prng: std.Random.DefaultPrng = .init(5);
        const rnd = prng.random();
        var bad: usize = 0;
        var placed: usize = 0;
        for (0..300) |_| {
            var f = try Fx.init(t, 2, P);
            defer f.deinit();
            var id: u64 = 1;
            for (0..2) |h| {
                var at = h * P;
                while (true) {
                    const n = (rnd.uintLessThan(usize, 19) + 1) * C;
                    if (at + n > (h + 1) * P) break;
                    if (rnd.float(f64) < 0.8) {
                        try f.kept(id, at, n, at + n);
                        id += 1;
                    }
                    at += n;
                }
            }
            rnd.shuffle(Kept, f.pl.kept.items);
            const need = (rnd.uintLessThan(usize, 63) + 1) * C;
            var p = try f.plan(rq(1000, need, null));
            defer p.deinit(t.gpa);
            if (p.status != .placed) continue;
            placed += 1;
            var buf: [64]u64 = undefined;
            const v = staled(&p, &buf);
            for (v) |k| if (f.kp(k).part != p.part) {
                bad += 1;
            };
            if (v.len > 0) {
                // without the newest victim, the part has no room for it
                const st = try t.gpa.alloc(bool, f.pl.kept.items.len);
                defer t.gpa.free(st);
                @memset(st, false);
                for (v[0 .. v.len - 1]) |k| st[f.pl.keptIndex(k).?] = true;
                if (try f.pl.placeSoftIn(t.gpa, p.part, need, .{ .twins_free = true }, st) != null) bad += 1;
            }
            f.apply(&p) catch {
                bad += 1;
            };
        }
        t.check("6 random: evictions in one part only, none beyond the need, every plan applies", bad == 0 and placed > 100);
    }
    for ([_]u8{ 3, 4 }) |N| {
        // 7. N parts: placement, cuts, eviction in the part losing the oldest
        const Q = 32 * C;
        var f = try Fx.init(t, N, Q);
        defer f.deinit();
        t.check("7 N parts", f.pl.cfg.parts == N and f.pl.part_rows == Q);
        var ok = true;
        for (0..N) |q| ok = ok and try f.pl.freeIn(t.gpa, @intCast(q)) == Q;
        t.check("7 free rows cut at every part's end", ok);
        try f.live(1, 0, 8 * C);
        try f.live(2, Q, 8 * C);
        var p = try f.plan(rq(10, 4 * C, null));
        defer p.deinit(t.gpa);
        t.check("7 new stream to a part with no live stream", p.part == 2);
        const cross = f.pl.applyOps(t.gpa, &.{.{ .admit = .{ .stream = 11, .slot = 11, .part = 0, .base = Q - C, .size = 2 * C, .gen = f.pl.next_gen } }});
        t.check("7 no extent across a part's end", if (cross) |_| false else |err| err == error.CrossesPart);
        var g = try Fx.init(t, N, Q);
        defer g.deinit();
        for (0..2) |j| for (0..N) |h| try g.kept(h * 10 + j, h * Q + j * 16 * C, 16 * C, h * Q + (j + 1) * 16 * C);
        var p2 = try g.plan(rq(10, 24 * C, null));
        defer p2.deinit(t.gpa);
        t.check("7 eviction in one part, the one losing the oldest", p2.part == 0 and staledAre(&p2, &.{ 0, 1 }));
    }
    {
        // 7b. placement never crosses a part's end (both parts equally light, part 0 first)
        var f = try Fx.init(t, 2, 32 * C);
        defer f.deinit();
        try f.live(1, 0, 24 * C);
        try f.live(2, 40 * C, 24 * C);
        var p = try f.plan(rq(10, 16 * C, null));
        defer p.deinit(t.gpa);
        t.check("7 16C where each part has 8C free: no room (never across the parts' end)", p.status != .placed);
    }
    {
        // 8. sizes: the ranks agree on the smallest pool in whole parts; bad sizes refused
        t.check("8 agree: the smallest rank's pool in whole parts of whole blocks", try agree(&.{ 2_000_000, 1_999_000 }, 2) == 1_999_000 / (2 * C) * (2 * C));
        t.check("8 agree: a pool below a part of one block is refused", if (agree(&.{ 4096, 100 }, 2)) |_| false else |_| true);
        t.check("8 config: a pool not cut in whole blocks is refused", if (Planner.init(.{ .cap = 3 * C, .parts = 2 })) |_| false else |_| true);
        var f = try Fx.init(t, 2, 32 * C);
        defer f.deinit();
        t.check("8 a request larger than a part is refused before placement", if (f.plan(rq(1, 33 * C, null))) |_| false else |err| err == error.PromptTooLong);
    }
}

// 9: twins ------------------------------------------------------------------------------------------------------------

fn secTwins(t: *T) !void {
    const P = 32 * C;
    {
        // a pinned twin (the fill mid-chunk): placement goes around it; the room comes from the kept state
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.live(1, 0, 8 * C);
        f.lv(1).partial = true;
        try f.twin(1, P + 8 * C, 8 * C);
        try f.kept(5, P + 24 * C, 8 * C, P + 32 * C);
        var p = try f.pl.planRestore(t.gpa, 9, 10 * C, 1);
        defer p.deinit(t.gpa);
        try f.apply(&p);
        t.check("9 twin: a pinned twin stays where it is, never given back", f.pl.twins.items.len == 1 and f.pl.twins.items[0].base == P + 8 * C and count(&p, .untwin) == 0);
        t.check("9 twin: the room came from evicting the kept state", p.base == P + 16 * C and staledAre(&p, &.{5}));
    }
    {
        // free 8 | pinned twin 8 | free 8 | live 8: 16 free, never together; nothing evicted, nothing moves
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.live(1, 0, 8 * C);
        f.lv(1).partial = true;
        try f.twin(1, P + 8 * C, 8 * C);
        try f.live(2, P + 24 * C, 8 * C);
        var p = try f.pl.planRestore(t.gpa, 9, 12 * C, 1);
        defer p.deinit(t.gpa);
        t.check("9 twin: its rows are no room; none found, no op", p.status == .full and p.ops.items.len == 0);
        f.lv(1).partial = false;
        var q = try f.pl.planRestore(t.gpa, 9, 12 * C, 1);
        defer q.deinit(t.gpa);
        try f.apply(&q);
        t.check("9 twin: between chunks it is given back (UNTWIN, counted revoked) and the room is found", q.status == .placed and count(&q, .untwin) == 1 and f.pl.twins.items.len == 0 and f.pl.counters.twins_revoked == 1 and q.base == P);
    }
    {
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.live(1, 0, 8 * C);
        try f.twin(1, P + 8 * C, 8 * C);
        const r = f.pl.applyOps(t.gpa, &.{.{ .move = .{ .stream = 1, .gen = f.lv(1).gen, .src = 0, .src_part = 0, .dst = P + 16 * C, .dst_part = 1, .size = 8 * C, .rows = 8 * C, .hold_gen = f.pl.next_gen, .tick = 0 } }});
        t.check("9 twin: a stream holding twins is never moved", if (r) |_| false else |err| err == error.Pinned);
    }
    {
        // fits: free rows spread over parts are no room; a part's kept rows are; a pinned twin's are not
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.live(1, 0, 20 * C);
        try f.live(2, P, 20 * C);
        t.check("9 fits: free rows spread over parts are not room for one request", !try f.pl.fits(t.gpa, 16 * C));
        var g = try Fx.init(t, 2, P);
        defer g.deinit();
        try g.live(1, 0, 16 * C);
        try g.kept(3, 16 * C, 8 * C, 24 * C);
        try g.live(2, P, 20 * C);
        t.check("9 fits: free and kept rows of ONE part cover it", try g.pl.fits(t.gpa, 16 * C));
        var h = try Fx.init(t, 2, P);
        defer h.deinit();
        try h.live(1, 0, 20 * C);
        h.lv(1).partial = true;
        try h.twin(1, P + 12 * C, 20 * C);
        try h.live(2, P, 12 * C);
        t.check("9 fits: a pinned twin's rows are no room", !try h.pl.fits(t.gpa, 16 * C));
        h.lv(1).partial = false;
        t.check("9 fits: a twin between chunks is room (it is given back)", try h.pl.fits(t.gpa, 16 * C));
    }
    {
        // a fresh prompt's twins: one in every other part, all or none, released before decode and at release
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        var p = try f.admit(rq(1, 8 * C, null));
        defer p.deinit(t.gpa);
        t.check("9 twins: a fresh prompt borrows a twin in the other part", count(&p, .twin) == 1 and f.pl.twins.items.len == 1 and f.pl.twins.items[0].part == 1 and f.pl.counters.twins_made == 1);
        var d = try f.pl.planDecode(t.gpa, 1);
        defer d.deinit(t.gpa);
        try f.apply(&d);
        t.check("9 twins: given back before the first decode round", f.pl.twins.items.len == 0 and f.pl.counters.twins_revoked == 0);
        var q = try f.admit(rq(2, 8 * C, null));
        defer q.deinit(t.gpa);
        const had = f.pl.twins.items.len;
        try f.release(2);
        t.check("9 twins: and at release", had == 1 and f.pl.twins.items.len == 0);
        var g = try Fx.init(t, 3, P);
        defer g.deinit();
        try g.live(9, 2 * P, 32 * C);
        var r = try g.plan(rq(1, 8 * C, null));
        defer r.deinit(t.gpa);
        t.check("9 twins: all or none (one part full: no twin anywhere)", r.status == .placed and count(&r, .twin) == 0);
        var h = try Fx.init(t, 2, P);
        defer h.deinit();
        try h.kept(4, P, 32 * C, 2 * P);
        var s = try h.plan(rq(1, 8 * C, null));
        defer s.deinit(t.gpa);
        t.check("9 twins: never evict a kept state (the other part holds only kept rows)", s.part == 0 and count(&s, .twin) == 0 and count(&s, .stale) == 0);
        var u = try Fx.init(t, 1, 2 * P);
        defer u.deinit();
        var w = try u.plan(rq(1, 8 * C, null));
        defer w.deinit(t.gpa);
        t.check("9 twins: none with one part", count(&w, .twin) == 0);
        var k = try Fx.init(t, 2, P);
        defer k.deinit();
        try k.kept(7, 0, 8 * C, 16 * C);
        var x = try k.plan(rq(1, 12 * C, 7));
        defer x.deinit(t.gpa);
        t.check("9 twins: a resumed prompt borrows none", x.mode == .in_place and count(&x, .twin) == 0);
    }
    {
        // random layouts: a pinned twin never moves, is never given back or overlapped; nothing evicted for nothing
        var prng: std.Random.DefaultPrng = .init(11);
        const rnd = prng.random();
        var bad: usize = 0;
        var seen: usize = 0;
        for (0..300) |_| {
            var f = try Fx.init(t, 2, P);
            defer f.deinit();
            try f.live(1, 0, 8 * C);
            var at: usize = P;
            var id: u64 = 10;
            var tw = false;
            while (true) {
                const n = (rnd.uintLessThan(usize, 6) + 1) * C;
                if (at + n > 2 * P) break;
                switch (rnd.uintLessThan(u8, 5)) {
                    1, 2 => try f.kept(id, at, n, at + n),
                    3 => try f.live(id, at, n),
                    4 => if (!tw) {
                        try f.twin(1, at, n);
                        tw = true;
                    },
                    else => {},
                }
                id += 1;
                at += n + if (rnd.boolean()) 0 else C;
            }
            if (!tw) continue;
            seen += 1;
            const pinned = rnd.boolean();
            f.lv(1).partial = pinned;
            rnd.shuffle(Kept, f.pl.kept.items);
            const base0 = f.pl.twins.items[0].base;
            const need = (rnd.uintLessThan(usize, 19) + 1) * C;
            var p = if (rnd.boolean()) try f.pl.planRestore(t.gpa, 99, need, 1) else try f.plan(rq(500, need, null));
            defer p.deinit(t.gpa);
            if (p.status == .full and count(&p, .stale) > 0) bad += 1;
            f.apply(&p) catch {
                bad += 1;
                continue;
            };
            if (pinned) {
                if (f.pl.twins.items.len == 0 or f.pl.twins.items[0].base != base0) bad += 1;
                const tb = base0;
                const te = base0 + f.pl.twins.items[0].size;
                for (f.pl.lives.items) |l| if (l.base < te and tb < l.base + l.size) {
                    bad += 1;
                };
                for (f.pl.holds.items) |h| if (h.base < te and tb < h.base + h.size) {
                    bad += 1;
                };
            }
        }
        t.check("9 twin random: a pinned twin never moves, is never given back or overlapped; no eviction for nothing", bad == 0 and seen > 100);
    }
    {
        // no part can make the room: none is evicted, the refusal counted once
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.live(1, 0, 24 * C);
        try f.live(2, P, 24 * C);
        try f.kept(3, 24 * C, 8 * C, 32 * C);
        try f.kept(4, P + 24 * C, 8 * C, 2 * P);
        var p = try f.plan(rq(10, 16 * C, null));
        defer p.deinit(t.gpa);
        const before = f.pl.digest();
        try f.apply(&p);
        t.check("9 room: none when no part can make it, and no kept state is evicted", p.status != .placed and count(&p, .stale) == 0 and std.mem.eql(u8, &before, &f.pl.digest()));
        t.check("9 room: the refusal is counted", f.pl.counters.no_part == 1);
        var q = try f.plan(rq(10, 8 * C, null));
        defer q.deinit(t.gpa);
        t.check("9 room: still evicts the oldest when a part can (8C: one state)", q.status == .placed and count(&q, .stale) == 1);
    }
}

// 10: the carry -------------------------------------------------------------------------------------------------------

/// Part 0: a live stream (24C) and the kept hit (8C, its stream gone): full. Part 1 empty.
fn rescueLayout(t: *T, at: usize) !Fx {
    var f = try Fx.init(t, 2, 32 * C);
    errdefer f.deinit();
    try f.live(1, 0, 24 * C);
    try f.kept(7, 24 * C, at, 32 * C);
    return f;
}

/// Part 0: a live stream (24C) and another (8C) whose extent holds the hit.
fn heldLayout(t: *T) !Fx {
    var f = try Fx.init(t, 2, 32 * C);
    errdefer f.deinit();
    try f.live(1, 0, 24 * C);
    try f.live(3, 24 * C, 8 * C);
    try f.kept(7, 24 * C, 8 * C, 32 * C);
    return f;
}

/// Part 0: a live stream (10C), one (8C) whose extent holds the hit, another conversation's kept state y (8C), 6C free:
/// a request of 14C fits only by evicting y. Part 1: `live1` rows of a live stream.
fn pressureLayout(t: *T, live1: usize, hit_at: usize) !Fx {
    var f = try Fx.init(t, 2, 32 * C);
    errdefer f.deinit();
    try f.live(1, 0, 10 * C);
    try f.live(3, 10 * C, 8 * C);
    try f.kept(7, 10 * C, hit_at, 18 * C);
    try f.kept(8, 18 * C, 8 * C, 26 * C);
    if (live1 > 0) try f.live(2, 32 * C, live1);
    return f;
}

fn secRescue(t: *T) !void {
    const P = 32 * C;
    {
        var f = try rescueLayout(t, 8 * C);
        defer f.deinit();
        var p = try f.plan(rq(50, 16 * C, 7));
        defer p.deinit(t.gpa);
        t.check("10 rescue: a full part no longer drops the hit: carried into the part with room", p.mode == .carry and p.why == .rescued and p.part == 1 and !p.dropped and count(&p, .stale) == 0);
        const cy = carryOf(&p);
        t.check("10 rescue: the carry names the hit's rows, all of them, part 0 to part 1", cy != null and cy.?.rows == 8 * C and cy.?.src == 24 * C and cy.?.src_part == 0 and cy.?.dst_part == 1);
        t.check("10 rescue: the extent first, then the carry (the follower holds the ADMIT before the rows travel)", index(&p, .admit) != null and index(&p, .carry) != null and index(&p, .admit).? < index(&p, .carry).? and count(&p, .carry) == 1 and count(&p, .copy) == 0);
        t.check("10 counters: nothing counted at planning", f.pl.counters.hits == 0 and f.pl.counters.rescued == 0);
        try f.apply(&p);
        const c = f.pl.counters;
        t.check("10 rescue: counted once at apply as rescued, no refusal, no no_part", c.hits == 1 and c.rescued == 1 and c.pressure == 0 and c.others == 0 and c.no_part == 0 and c.tokens == 8 * C and c.bytes == 8 * C * 720);
        const r = try f.pl.finish(50, true);
        t.check("10 rescue: finished, the stream resumes; the hit's kept state stays (a carry copies)", r.resumed and f.lv(50).valid == 8 * C and !f.kp(7).stale);
    }
    {
        // the other part's free rows lack headroom: the pressure rule would not, the rescue (evicting as needed) does
        var f = try rescueLayout(t, 8 * C);
        defer f.deinit();
        try f.live(2, P, 16 * C);
        var p = try f.plan(rq(50, 16 * C, 7));
        defer p.deinit(t.gpa);
        t.check("10 rescue: no headroom in the other part's free rows: still kept by the rescue", p.mode == .carry and p.part == 1 and p.why == .rescued);
    }
    {
        // the other part full of kept states: the oldest give their room, the hit stays
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.live(1, 0, 24 * C);
        try f.kept(20, P, 16 * C, P + 16 * C);
        try f.kept(21, P + 16 * C, 16 * C, 2 * P);
        try f.kept(7, 24 * C, 8 * C, 32 * C);
        var p = try f.plan(rq(50, 16 * C, 7));
        defer p.deinit(t.gpa);
        t.check("10 rescue: a part full of kept states gives up its oldest, the hit travels", p.mode == .carry and p.base == P and staledAre(&p, &.{20}));
        try f.apply(&p);
    }
    {
        var f = try heldLayout(t);
        defer f.deinit();
        var p = try f.plan(rq(50, 16 * C, 7));
        defer p.deinit(t.gpa);
        t.check("10 rescue: also when the hit lies in a live stream's extent", p.mode == .carry and p.why == .rescued and p.part == 1 and count(&p, .stale) == 0);
        var r = f.pl.planAdmit(t.gpa, .{ .stream = 50, .slot = 50, .size = 16 * C, .source = 7, .patient = true }) catch unreachable;
        defer r.deinit(t.gpa);
        t.check("10 paused: a request that would wait for the stream holding its hit is carried instead", r.mode == .carry and r.why == .paused and r.part == 1);
        try f.apply(&r);
        t.check("10 paused: counted as paused", f.pl.counters.paused == 1 and f.pl.counters.hits == 1);
    }
    for ([_]bool{ true, false }) |on| {
        var f = try Fx.initCfg(t, .{ .cap = 2 * P, .parts = 2, .carry = on });
        defer f.deinit();
        try f.live(1, 0, 24 * C);
        try f.live(2, P, 32 * C);
        try f.kept(7, 24 * C, 8 * C, 32 * C);
        var p = try f.plan(rq(50, 16 * C, 7));
        defer p.deinit(t.gpa);
        try f.apply(&p);
        t.check("10 rescue: no room in any part is full, counted once, nothing evicted", p.status == .full and f.pl.counters.no_part == 1 and count(&p, .stale) == 0);
    }
    {
        var f = try Fx.initCfg(t, .{ .cap = 2 * P, .parts = 2, .carry = false });
        defer f.deinit();
        try f.live(1, 0, 24 * C);
        try f.kept(7, 24 * C, 8 * C, 32 * C);
        var p = try f.plan(rq(50, 16 * C, 7));
        defer p.deinit(t.gpa);
        try f.apply(&p);
        const c = f.pl.counters;
        t.check("10 carry off: the hit is dropped as before (a fresh prompt in the other part)", p.dropped and p.mode == .fresh and p.part == 1 and c.hits == 0 and c.rescued == 0 and count(&p, .carry) == 0);
    }
    {
        var f = try rescueLayout(t, 1024);
        defer f.deinit();
        var p = try f.plan(rq(50, 16 * C, 7));
        defer p.deinit(t.gpa);
        try f.apply(&p);
        t.check("10 rescue: a hit under carry_min is dropped as before (cheaper to recompute)", p.dropped and count(&p, .carry) == 0 and f.pl.counters.refusals(.small) == 1);
    }
}

fn secPins(t: *T) !void {
    const P = 32 * C;
    const Pin = struct { why: Refusal, setup: *const fn (*Fx) anyerror!void };
    const pins = [_]Pin{
        .{ .why = .loading, .setup = struct {
            fn f(x: *Fx) !void {
                x.lv(3).loading = true;
            }
        }.f },
        .{ .why = .twin, .setup = struct {
            fn f(x: *Fx) !void {
                try x.twin(3, 32 * C + 24 * C, 8 * C);
            }
        }.f },
        // mid-fill with twins is GLM's `twin` too (its `pinned` is a pinned twin's own extent, no native case)
        .{ .why = .twin, .setup = struct {
            fn f(x: *Fx) !void {
                try x.twin(3, 32 * C + 24 * C, 8 * C);
                x.lv(3).partial = true;
            }
        }.f },
        .{ .why = .partial, .setup = struct {
            fn f(x: *Fx) !void {
                x.lv(3).partial = true;
            }
        }.f },
    };
    for (pins) |pin| {
        var f = try heldLayout(t);
        defer f.deinit();
        try pin.setup(&f);
        var p = try f.plan(rq(50, 16 * C, 7));
        defer p.deinit(t.gpa);
        try f.apply(&p);
        const c = f.pl.counters;
        var others: u64 = 0;
        for (c.refused, 0..) |n, i| if (i != @intFromEnum(pin.why)) {
            others += n;
        };
        t.check("10 pins: rows something pins are not carried (the rescue): dropped as before, refusal counted once", p.mode != .carry and count(&p, .carry) == 0 and c.refusals(pin.why) == 1 and others == 0 and c.hits == 0);
        var g = try pressureLayout(t, 0, 8 * C);
        defer g.deinit();
        try pin.setup(&g);
        var q = try g.plan(rq(50, 14 * C, 7));
        defer q.deinit(t.gpa);
        try g.apply(&q);
        t.check("10 pins: ...nor by the pressure rule: placed in its part as before", q.mode == .copy and q.part == 0 and g.pl.counters.refusals(pin.why) == 1 and g.pl.counters.pressure == 0);
    }
    _ = P;
}

fn secPressure(t: *T) !void {
    const P = 32 * C;
    {
        var f = try pressureLayout(t, 0, 8 * C);
        defer f.deinit();
        var p = try f.plan(rq(50, 14 * C, 7));
        defer p.deinit(t.gpa);
        t.check("10 pressure: placing it would evict y and the other part has the room free: carried, y kept", p.mode == .carry and p.why == .pressure and p.part == 1 and count(&p, .stale) == 0);
        try f.apply(&p);
        t.check("10 pressure: counted as pressure only", f.pl.counters.pressure == 1 and f.pl.counters.rescued == 0 and f.pl.counters.others == 0 and !f.kp(8).stale);
    }
    {
        // free rows without the headroom (a block for the stream there and the arriving one) are no room
        var f = try pressureLayout(t, 17 * C, 8 * C);
        defer f.deinit();
        var p = try f.plan(rq(50, 14 * C, 7));
        defer p.deinit(t.gpa);
        try f.apply(&p);
        t.check("10 pressure: without headroom no carry: y evicted and the hit copied in its part as before", p.mode == .copy and p.part == 0 and staledAre(&p, &.{8}) and f.pl.counters.refusals(.no_target) == 1);
        var g = try pressureLayout(t, 16 * C, 8 * C);
        defer g.deinit();
        var q = try g.plan(rq(50, 14 * C, 7));
        defer q.deinit(t.gpa);
        t.check("10 pressure: ...and with the headroom free it carries", q.mode == .carry and q.part == 1);
    }
    {
        // a target part with only kept rows: the pressure rule never evicts there
        var f = try pressureLayout(t, 0, 8 * C);
        defer f.deinit();
        try f.kept(30, P, 32 * C, 2 * P);
        var p = try f.plan(rq(50, 14 * C, 7));
        defer p.deinit(t.gpa);
        t.check("10 pressure: no free target: no eviction in another part, y evicted at home as before", p.mode == .copy and staledAre(&p, &.{8}));
    }
    {
        // its part fits the request without evicting: never carried, however empty the other part
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.live(1, 0, 10 * C);
        try f.live(3, 10 * C, 8 * C);
        try f.kept(7, 10 * C, 8 * C, 18 * C);
        var p = try f.plan(rq(50, 14 * C, 7));
        defer p.deinit(t.gpa);
        t.check("10 no pressure: its part fits it without evicting: copied there, no carry though part 1 is empty", p.mode == .copy and p.part == 0 and count(&p, .stale) == 0);
        var g = try Fx.init(t, 2, P);
        defer g.deinit();
        for (0..5) |i| try g.live(10 + i, i * 4 * C, 4 * C);
        try g.kept(7, 20 * C, 8 * C, 28 * C);
        var q = try g.plan(rq(50, 10 * C, 7));
        defer q.deinit(t.gpa);
        t.check("10 no pressure: 5 live streams against none is no reason (no balance carry)", q.mode == .in_place);
    }
    {
        var f = try pressureLayout(t, 0, 1024);
        defer f.deinit();
        var p = try f.plan(rq(50, 14 * C, 7));
        defer p.deinit(t.gpa);
        try f.apply(&p);
        t.check("10 pressure: a hit under carry_min is never carried: refused as small, y evicted as before", p.mode == .copy and f.pl.counters.refusals(.small) == 1 and staledAre(&p, &.{8}));
        var g = try pressureLayout(t, 0, C);
        defer g.deinit();
        var q = try g.plan(rq(50, 14 * C, 7));
        defer q.deinit(t.gpa);
        t.check("10 pressure: a hit of exactly carry_min tokens carries", q.mode == .carry);
    }
    {
        // no ping-pong: carried to part 1, the next turn resumes there; once part 1 fills it is not carried back
        var f = try rescueLayout(t, 8 * C);
        defer f.deinit();
        var p = try f.admit(rq(50, 16 * C, 7));
        defer p.deinit(t.gpa);
        _ = try f.pl.finish(50, true);
        f.pl.setValid(50, 12 * C);
        try f.pl.applyOps(t.gpa, &.{try f.pl.keepOp(70, 50, 12 * C)});
        try f.release(50);
        var q = try f.admit(rq(51, 20 * C, 70));
        defer q.deinit(t.gpa);
        t.check("10 no ping-pong: the next turn resumes in the part it was carried to", q.mode == .in_place and q.part == 1);
        try f.release(51);
        try f.live(9, P + 20 * C, 12 * C);
        var r = try f.plan(rq(52, 28 * C, 70));
        defer r.deinit(t.gpa);
        t.check("10 no ping-pong: a turn its part cannot fit is not carried back into the part it left (no room there)", r.mode != .carry);
    }
}

fn secOthers(t: *T) !void {
    const P = 32 * C;
    const Lay = struct {
        fn make(tt: *T, live0: usize, live1: usize, hit_at: usize) !Fx {
            var f = try Fx.init(tt, 2, 32 * C);
            errdefer f.deinit();
            try f.live(1, 0, live0);
            if (live1 > 0) try f.live(2, 32 * C, live1);
            try f.kept(7, live0, hit_at, live0 + 16 * C); // the shared prefix (4C)
            try f.kept(9, live0, 16 * C, live0 + 16 * C); // another conversation's longer state in the same rows
            return f;
        }
    };
    {
        var f = try Lay.make(t, 16 * C, 0, 4 * C);
        defer f.deinit();
        var p = try f.plan(rq(50, 10 * C, 7));
        defer p.deinit(t.gpa);
        t.check("10 others: the part has no free rows for the copy: another part's carry it, the other conversation keeps its state", p.mode == .carry and p.why == .others and p.part == 1 and count(&p, .stale) == 0);
        try f.apply(&p);
        t.check("10 others: counted as others only", f.pl.counters.others == 1 and f.pl.counters.rescued == 0 and f.pl.counters.pressure == 0 and !f.kp(9).stale);
    }
    {
        var f = try Lay.make(t, 4 * C, 0, 4 * C);
        defer f.deinit();
        var p = try f.plan(rq(50, 10 * C, 7));
        defer p.deinit(t.gpa);
        t.check("10 others: free rows in its own part still take the copy, nothing carried", p.mode == .copy and p.part == 0 and count(&p, .stale) == 0);
    }
    {
        var f = try Lay.make(t, 16 * C, 32 * C, 4 * C);
        defer f.deinit();
        var p = try f.plan(rq(50, 10 * C, 7));
        defer p.deinit(t.gpa);
        try f.apply(&p);
        t.check("10 others: no free rows anywhere: the other state is evicted, the copy made at home, as before", p.mode == .copy and staledAre(&p, &.{9}) and f.pl.counters.refusals(.no_target) == 1);
    }
    {
        var f = try Lay.make(t, 16 * C, 0, 1024);
        defer f.deinit();
        var p = try f.plan(rq(50, 10 * C, 7));
        defer p.deinit(t.gpa);
        try f.apply(&p);
        t.check("10 others: a hit under carry_min is not carried (small), the other state evicted as before", p.mode == .copy and f.pl.counters.refusals(.small) == 1 and staledAre(&p, &.{9}));
    }
    _ = P;
}

// the native reference: lanes.zig's placement at 629c524 + the recipe, verbatim in shape, on plain arrays -------------

const RefSnap = struct { base: usize, at: usize, end: usize, stale: bool = false };

fn refClashes(used: []const [2]usize, a: usize, b: usize) bool {
    for (used) |u| if (a < u[1] and u[0] < b) return true;
    return false;
}

fn refPlaceSoft(gpa: Allocator, live: []const [2]usize, kept: []const [3]usize, size: usize, cap: usize) !?usize {
    var used: std.ArrayList([2]usize) = .empty;
    defer used.deinit(gpa);
    try used.appendSlice(gpa, live);
    for (kept) |k| try used.append(gpa, .{ k[0], @max(k[2], k[0] + k[1]) + growth_rows });
    if (placeIn(used.items, size, 0, cap)) |b| return b;
    used.shrinkRetainingCapacity(live.len);
    for (kept) |k| try used.append(gpa, .{ k[0], @max(k[2], k[0] + k[1]) });
    if (placeIn(used.items, size, 0, cap)) |b| return b;
    used.shrinkRetainingCapacity(live.len);
    for (kept) |k| try used.append(gpa, .{ k[0], k[0] + k[1] });
    return placeIn(used.items, size, 0, cap);
}

fn refPlaceKept(gpa: Allocator, live: []const [2]usize, snaps: []RefSnap, size: usize, cap: usize, source: ?usize, patient: bool) !?usize {
    var kept: std.ArrayList([3]usize) = .empty;
    defer kept.deinit(gpa);
    var rows: std.ArrayList([2]usize) = .empty;
    defer rows.deinit(gpa);
    const held_ = if (source) |s| !snaps[s].stale and refClashes(live, snaps[s].base, snaps[s].base + snaps[s].at) else false;
    while (true) {
        kept.clearRetainingCapacity();
        rows.clearRetainingCapacity();
        try rows.appendSlice(gpa, live);
        for (snaps) |sn| if (!sn.stale) {
            try kept.append(gpa, .{ sn.base, sn.at, sn.end });
            try rows.append(gpa, .{ sn.base, sn.base + sn.at });
        };
        const got = try refPlaceSoft(gpa, live, kept.items, size, cap);
        if (held_ and patient) {
            const b = got orelse return null;
            try rows.append(gpa, .{ b, b + size });
            return if (placeIn(rows.items, size, 0, cap) != null) b else null;
        }
        if (got) |b| return b;
        var victim: ?usize = null;
        for (snaps, 0..) |sn, i| if (!sn.stale and i != source and !refClashes(live, sn.base, sn.base + sn.at)) {
            victim = i;
            break;
        };
        if (victim == null) if (source) |s| if (!snaps[s].stale and !refClashes(live, snaps[s].base, snaps[s].base + snaps[s].at)) {
            victim = s;
        };
        snaps[victim orelse return null].stale = true;
    }
}

fn refKeptOver(snaps: []const RefSnap, a: usize, b: usize, except: usize) bool {
    const e = snaps[except];
    for (snaps, 0..) |sn, i| if (i != except and !sn.stale and (sn.base != e.base or sn.at > e.at) and sn.base < b and a < sn.base + sn.at) return true;
    return false;
}

fn refAdmit(gpa: Allocator, live: []const [2]usize, snaps: []RefSnap, size: usize, cap: usize, source: ?usize, patient: bool) !?struct { base: usize, kept: bool, moved: bool } {
    var moved = false;
    if (source) |s| {
        const sn = snaps[s];
        moved = sn.base + size > cap or refClashes(live, sn.base, sn.base + size) or refKeptOver(snaps, sn.base + sn.at, sn.base + size, s);
    }
    const base = if (source != null and !moved) snaps[source.?].base else (try refPlaceKept(gpa, live, snaps, size, cap, source, patient) orelse return null);
    var kept = source;
    if (kept) |s| if (!moved) {
        snaps[s].end = @max(snaps[s].end, base + size);
    };
    if (kept) |s| if (moved and snaps[s].stale) {
        kept = null;
    };
    const a = if (kept != null and !moved) base + snaps[kept.?].at else base;
    for (snaps, 0..) |*sn, i| if (i != kept and sn.base < base + size and a < sn.base + sn.at) {
        sn.stale = true;
    };
    return .{ .base = base, .kept = kept != null, .moved = moved };
}

/// A random pool: live streams (some holding kept states), kept states (one or two at a base), free rows; with parts,
/// twins of filling prompts, holds (restores), pins and pending spill writes.
fn randLayout(f: *Fx, rnd: std.Random, heavy: bool) !void {
    const parts = f.pl.cfg.parts;
    const P = f.pl.part_rows;
    var id: u64 = 1;
    var stream: u64 = 100;
    var twin_spans: std.ArrayList([2]usize) = .empty;
    defer twin_spans.deinit(f.gpa);
    for (0..parts) |h| {
        var at = h * P;
        while (true) {
            const n = (rnd.uintLessThan(usize, 8) + 1) * C;
            if (at + n > (h + 1) * P) break;
            const kinds = if (heavy) "LLLKFHTS" else "LKKKFHTS";
            switch (kinds[rnd.uintLessThan(usize, kinds.len)]) {
                'L' => {
                    try f.live(stream, at, n);
                    f.lv(stream).valid = rnd.uintLessThan(usize, n);
                    f.lv(stream).decoding = rnd.float(f64) < 0.7;
                    if (parts > 1 and rnd.float(f64) < 0.1) f.lv(stream).partial = true;
                    stream += 1;
                },
                'S' => {
                    try f.live(stream, at, n);
                    const k = (rnd.uintLessThan(usize, n / C) + 1) * C;
                    try f.kept(id, at, k, at + n);
                    f.kp(id).pending = parts > 1 and rnd.float(f64) < 0.2;
                    f.lv(stream).valid = k + rnd.uintLessThan(usize, n - k + 1);
                    id += 1;
                    stream += 1;
                },
                'K' => {
                    const a = (rnd.uintLessThan(usize, n / C) + 1) * C;
                    try f.kept(id, at, a, at + n);
                    f.kp(id).pending = parts > 1 and rnd.float(f64) < 0.2;
                    id += 1;
                    if (rnd.float(f64) < 0.3) {
                        const b = (rnd.uintLessThan(usize, n / C) + 1) * C;
                        if (b != a) {
                            try f.kept(id, at, b, at + n);
                            id += 1;
                        }
                    }
                },
                'H' => if (parts > 1) try f.hold(1000 + id, at, n),
                'T' => if (parts > 1) try twin_spans.append(f.gpa, .{ at, n }),
                else => {},
            }
            at += n + if (rnd.uintLessThan(u8, 3) == 0) C else 0;
        }
    }
    for (twin_spans.items) |ts| {
        // a twin of a filling stream in another part
        var homes: [64]u64 = undefined;
        var nh: usize = 0;
        for (f.pl.lives.items) |l| if (l.part != f.pl.partOf(ts[0]) and nh < homes.len) {
            homes[nh] = l.stream;
            nh += 1;
        };
        if (nh == 0) continue;
        const s = homes[rnd.uintLessThan(usize, nh)];
        try f.twin(s, ts[0], ts[1]);
        f.lv(s).decoding = false;
    }
    rnd.shuffle(Kept, f.pl.kept.items);
}

fn randRequest(f: *Fx, rnd: std.Random) Request {
    var r: Request = .{ .stream = 1, .slot = 1, .size = @min((rnd.uintLessThan(usize, 18) + 2) * C, f.pl.part_rows) };
    var live_kept: [256]u64 = undefined;
    var n: usize = 0;
    for (f.pl.kept.items) |k| if (!k.stale and n < live_kept.len) {
        live_kept[n] = k.id;
        n += 1;
    };
    if (n > 0 and rnd.float(f64) < 0.9) {
        r.source = live_kept[rnd.uintLessThan(usize, n)];
        r.size = @min(@max(r.size, f.kp(r.source.?).at + C), f.pl.part_rows);
        if (r.size <= f.kp(r.source.?).at) r.source = null;
    }
    r.patient = rnd.float(f64) < 0.25;
    return r;
}

fn secNative(t: *T) !void {
    // with one part (the switch off) every decision is lanes.zig's: base, resume, and the states it stales
    var bad: usize = 0;
    var kinds = [_]usize{ 0, 0, 0, 0 }; // in place, copy, dropped, full
    for (0..500) |seed| {
        var prng: std.Random.DefaultPrng = .init(seed);
        const rnd = prng.random();
        var f = try Fx.init(t, 1, 64 * C);
        defer f.deinit();
        try randLayout(&f, rnd, seed % 2 == 1);
        const req = randRequest(&f, rnd);
        var live: std.ArrayList([2]usize) = .empty;
        defer live.deinit(t.gpa);
        for (f.pl.lives.items) |l| try live.append(t.gpa, .{ l.base, l.base + l.size });
        const snaps = try t.gpa.alloc(RefSnap, f.pl.kept.items.len);
        defer t.gpa.free(snaps);
        for (f.pl.kept.items, snaps) |k, *s| s.* = .{ .base = k.base, .at = k.at, .end = k.end };
        const src: ?usize = if (req.source) |id| f.pl.keptIndex(id) else null;
        const held_ = if (req.source) |id| f.pl.held(id) else false;
        const want = try refAdmit(t.gpa, live.items, snaps, req.size, f.pl.cfg.cap, src, req.patient and held_);
        var r = req;
        r.patient = req.patient and held_;
        var p = try f.plan(r);
        defer p.deinit(t.gpa);
        const w = want orelse {
            kinds[3] += 1;
            if (p.status != .full) {
                if (t.verbose and bad == 0) std.debug.print("  native seed {d}: plan {s} {s} base {d}, want full\n", .{ seed, @tagName(p.status), @tagName(p.mode), p.base / C });
                bad += 1;
            }
            continue;
        };
        const resumed = p.source != null and !p.dropped;
        if (p.status != .placed or p.base != w.base or resumed != w.kept or (p.mode == .in_place) != (w.kept and !w.moved)) {
            if (t.verbose and bad == 0) std.debug.print("  native seed {d}: plan {s} {s} base {d} resumed {} / want base {d} kept {} moved {}\n", .{ seed, @tagName(p.status), @tagName(p.mode), p.base / C, resumed, w.base / C, w.kept, w.moved });
            bad += 1;
            continue;
        }
        if (count(&p, .carry) + count(&p, .twin) + count(&p, .move) > 0) bad += 1;
        kinds[if (p.mode == .in_place) 0 else if (p.dropped) 2 else 1] += 1;
        if (count(&p, .carry) + count(&p, .twin) + count(&p, .move) > 0 and t.verbose and bad == 0) std.debug.print("  native seed {d}: sharded ops\n", .{seed});
        f.apply(&p) catch |err| {
            if (t.verbose and bad == 0) std.debug.print("  native seed {d}: apply {s} ({s} {s})\n", .{ seed, @errorName(err), @tagName(p.status), @tagName(p.mode) });
            bad += 1;
            continue;
        };
        for (f.pl.kept.items, snaps) |k, s| if (k.stale != s.stale or k.end != s.end) {
            if (t.verbose and bad == 0) std.debug.print("  native seed {d}: kept {d} stale {} end {d} / want stale {} end {d}\n", .{ seed, k.id, k.stale, k.end / C, s.stale, s.end / C });
            bad += 1;
            break;
        };
    }
    t.check("10 one part: 500 random pools place, resume, copy and stale exactly as lanes.zig", bad == 0);
    t.check("10 one part: the sweep reaches in-place, copy, dropped and full", kinds[0] > 20 and kinds[1] > 20 and kinds[2] > 5 and kinds[3] > 5);
}

fn miss(t: *T, bad: *usize, where: usize, seed: usize) void {
    if (t.verbose and bad.* == 0) std.debug.print("  first miss: check {d}, seed {d}\n", .{ where, seed });
    bad.* += 1;
}

fn secInvariants(t: *T) !void {
    // carry on against carry off on the same pool: a carry only where carry off evicted, dropped the hit or waited;
    // otherwise the same plan; nothing pinned or small carried; pressure/others/paused carries evict nothing
    var bad: usize = 0;
    var carried: usize = 0;
    var kinds = [_]usize{ 0, 0, 0, 0 };
    var same: usize = 0;
    for (0..700) |seed| {
        for ([_]bool{ false, true }) |heavy| {
            var pa: std.Random.DefaultPrng = .init(seed);
            var pb: std.Random.DefaultPrng = .init(seed);
            var a = try Fx.initCfg(t, .{ .cap = 2 * 32 * C, .parts = 2 });
            defer a.deinit();
            var b = try Fx.initCfg(t, .{ .cap = 2 * 32 * C, .parts = 2, .carry = false });
            defer b.deinit();
            try randLayout(&a, pa.random(), heavy);
            try randLayout(&b, pb.random(), heavy);
            const req = randRequest(&a, pa.random());
            var on = try a.plan(req);
            defer on.deinit(t.gpa);
            var off = try b.plan(req);
            defer off.deinit(t.gpa);
            if (on.mode == .carry) {
                carried += 1;
                kinds[@intFromEnum(on.why.?)] += 1;
                const si = a.pl.keptIndex(req.source.?).?;
                const k = a.pl.kept.items[si];
                if (on.part == k.part or a.pl.pinsKept(si) != null or k.at < carry_min) miss(t, &bad, 1, seed);
                if (!(off.status != .placed or off.dropped or count(&off, .stale) > 0)) miss(t, &bad, 2, seed);
                if (on.why.? != .rescued and count(&on, .stale) > 0) miss(t, &bad, 3, seed);
            } else {
                const st = if (on.status == .relocate) Status.full else on.status;
                if (st != off.status or on.mode != off.mode or on.part != off.part or on.base != off.base or on.dropped != off.dropped) {
                    miss(t, &bad, 4, seed);
                } else if (on.status != .relocate and on.ops.items.len != off.ops.items.len) miss(t, &bad, 5, seed) else same += 1;
                if (on.mode == .copy and on.part != a.pl.kept.items[a.pl.keptIndex(on.source.?).?].part) miss(t, &bad, 6, seed);
            }
            a.apply(&on) catch |err| {
                if (t.verbose and bad == 0) std.debug.print("  invariants apply {s}: {s} {s} dropped {}\n", .{ @errorName(err), @tagName(on.status), @tagName(on.mode), on.dropped });
                miss(t, &bad, 7, seed);
            };
        }
    }
    t.check("10 random pools: a carry only where carry off evicted, dropped the hit or waited; else the same plan; never pinned or small; pressure/others/paused evict nothing", bad == 0);
    t.check("10 random pools: the carry happens in every kind (the sweep reaches them)", carried > 40 and kinds[@intFromEnum(Reason.rescued)] >= 3 and kinds[@intFromEnum(Reason.pressure)] >= 3 and kinds[@intFromEnum(Reason.paused)] >= 1 and same > 200);
}

fn secTwoRanks(t: *T) !void {
    // rank 0 plans and applies, rank 1 applies the ops it is sent: the same pool, the same generations
    var bad: usize = 0;
    for (0..300) |seed| {
        var p0: std.Random.DefaultPrng = .init(seed + 7);
        var p1: std.Random.DefaultPrng = .init(seed + 7);
        var r0 = try Fx.init(t, 2, 32 * C);
        defer r0.deinit();
        var r1 = try Fx.init(t, 2, 32 * C);
        defer r1.deinit();
        try randLayout(&r0, p0.random(), seed % 2 == 0);
        try randLayout(&r1, p1.random(), seed % 2 == 0);
        const req = randRequest(&r0, p0.random());
        var p = try r0.plan(req);
        defer p.deinit(t.gpa);
        r0.apply(&p) catch {
            miss(t, &bad, 8, seed);
            continue;
        };
        r1.pl.applyOps(t.gpa, p.ops.items) catch {
            miss(t, &bad, 9, seed);
            continue;
        };
        if (!std.mem.eql(u8, &r0.pl.digest(), &r1.pl.digest())) miss(t, &bad, 10, seed);
        if (r0.pl.counters.hits != r1.pl.counters.hits or r0.pl.counters.tokens != r1.pl.counters.tokens or r0.pl.counters.live != r1.pl.counters.live) miss(t, &bad, 11, seed);
        // the transfers end alike on both
        while (r0.pl.flights.items.len > 0) {
            const s = r0.pl.flights.items[0].stream;
            _ = try r0.pl.finish(s, true);
            _ = try r1.pl.finish(s, true);
        }
        if (!std.mem.eql(u8, &r0.pl.digest(), &r1.pl.digest())) miss(t, &bad, 12, seed);
    }
    t.check("10 two ranks: the follower applying rank 0's ops ends with rank 0's pool and generations", bad == 0);
    // malformed frames fail closed
    var f = try Fx.init(t, 2, 32 * C);
    defer f.deinit();
    try f.kept(7, 0, 8 * C, 8 * C);
    try f.live(5, 8 * C, 8 * C);
    try f.live(6, 32 * C, 8 * C);
    const g7 = f.kp(7).gen;
    const in_one = f.pl.applyOps(t.gpa, &.{.{ .carry = .{ .kept = 7, .kept_gen = g7, .stream = 5, .src = 0, .src_part = 0, .dst = 8 * C, .dst_part = 0, .rows = 8 * C, .why = .rescued } }});
    t.check("10 follower: a carry inside one part fails closed", if (in_one) |_| false else |err| err == error.BadOp);
    const across = f.pl.applyOps(t.gpa, &.{.{ .copy = .{ .kept = 7, .kept_gen = g7, .stream = 6, .src = 0, .dst = 32 * C, .rows = 8 * C } }});
    t.check("10 follower: a copy across parts fails closed (that is a carry)", if (across) |_| false else |err| err == error.BadOp);
    const nokept = f.pl.applyOps(t.gpa, &.{.{ .carry = .{ .kept = 99, .kept_gen = 1, .stream = 6, .src = 0, .src_part = 0, .dst = 32 * C, .dst_part = 1, .rows = 8 * C, .why = .rescued } }});
    t.check("10 follower: a carry of no kept state fails closed", if (nokept) |_| false else |err| err == error.UnknownKept);
    const stale_gen = f.pl.applyOps(t.gpa, &.{.{ .stale = .{ .kept = 7, .gen = g7 + 100 } }});
    t.check("10 follower: a stale generation fails closed (a frame for a reused extent)", if (stale_gen) |_| false else |err| err == error.StaleGeneration);
    const order = f.pl.applyOps(t.gpa, &.{.{ .admit = .{ .stream = 9, .slot = 9, .part = 1, .base = 48 * C, .size = 4 * C, .gen = f.pl.next_gen + 1 } }});
    t.check("10 follower: an extent whose generation is out of step fails closed", if (order) |_| false else |err| err == error.OutOfStep);
    const over = f.pl.applyOps(t.gpa, &.{.{ .admit = .{ .stream = 9, .slot = 9, .part = 0, .base = 8 * C, .size = 4 * C, .gen = f.pl.next_gen } }});
    t.check("10 follower: an extent over a live one fails closed", if (over) |_| false else |err| err == error.Overlap);
    const before = f.pl.digest();
    const half = f.pl.applyOps(t.gpa, &.{ .{ .stale = .{ .kept = 7, .gen = g7 } }, .{ .stale = .{ .kept = 99, .gen = 0 } } });
    t.check("10 follower: a frame that fails midway changes nothing", half == error.UnknownKept and std.mem.eql(u8, &before, &f.pl.digest()) and !f.kp(7).stale);
}

fn secRelocate(t: *T) !void {
    const P = 32 * C;
    const Lay = struct {
        /// Part 0: stream 1 (8C, 6C used), 4C free, stream 2 (20C); part 1: stream 3 (21C), 11C free. A request of
        /// 12C fits nowhere; stream 1 leaving part 0 lets it in, and part 1 has its 8C and the headroom (2 blocks).
        fn make(tt: *T, parts: u8, part1: usize) !Fx {
            var f = try Fx.init(tt, parts, 32 * C);
            errdefer f.deinit();
            try f.live(1, 0, 8 * C);
            f.lv(1).valid = 6 * C;
            try f.live(2, 12 * C, 20 * C);
            f.lv(2).valid = 18 * C;
            try f.live(3, 32 * C, part1);
            f.pl.ticks = 1000;
            return f;
        }
    };
    {
        var f = try Lay.make(t, 2, 21 * C);
        defer f.deinit();
        var p = try f.plan(rq(60, 12 * C, null));
        defer p.deinit(t.gpa);
        var mv: ?@FieldType(Op, "move") = null;
        for (p.ops.items) |o| if (o == .move) {
            mv = o.move;
        };
        t.check("10 move: a request that would wait has a live stream moved whole to the part with room", p.status == .relocate and count(&p, .move) == 1 and mv != null and mv.?.stream == 1 and mv.?.dst_part == 1 and mv.?.size == 8 * C and mv.?.rows == 6 * C);
        try f.apply(&p);
        t.check("10 move: counted once at apply (live, tokens), the destination held, the source still there", f.pl.counters.live == 1 and f.pl.counters.tokens == 6 * C and f.pl.holds.items.len == 1 and f.lv(1).part == 0);
        _ = try f.pl.finish(1, true);
        t.check("10 move: finished, the stream is in part 1 with its whole reservation, its old rows free", f.lv(1).part == 1 and f.lv(1).size == 8 * C and f.pl.holds.items.len == 0 and f.lv(1).came_from == 0 and f.lv(1).moved_at == 1000);
        var q = try f.plan(rq(60, 12 * C, null));
        defer q.deinit(t.gpa);
        t.check("10 move: then the request is placed where the stream was", q.status == .placed and q.part == 0 and q.base == 0);
    }
    {
        var f = try Lay.make(t, 2, 23 * C);
        defer f.deinit();
        var p = try f.plan(rq(60, 12 * C, null));
        defer p.deinit(t.gpa);
        t.check("10 move: free rows without the headroom are no room: it waits as before", p.status == .full and count(&p, .move) == 0);
    }
    {
        var f = try Lay.make(t, 2, 21 * C);
        defer f.deinit();
        f.lv(1).moved_at = 1000 - relocate_hold + 1;
        var p = try f.plan(rq(60, 12 * C, null));
        defer p.deinit(t.gpa);
        t.check("10 move: a stream that moved less than relocate_hold rounds ago is not moved again", p.status == .full and count(&p, .move) == 0);
        f.pl.tick();
        var q = try f.plan(rq(60, 12 * C, null));
        defer q.deinit(t.gpa);
        t.check("10 move: ...and may once the window has passed", q.status == .relocate);
        f.pl.entered[0] = f.pl.ticks - relocate_hold + 1;
        var r = try f.plan(rq(60, 12 * C, null));
        defer r.deinit(t.gpa);
        t.check("10 move: no stream leaves a part another entered within the window", r.status == .full);
    }
    {
        var f = try Lay.make(t, 2, 21 * C);
        defer f.deinit();
        f.lv(1).came_from = 1;
        f.lv(1).moved_at = 1000 - relocate_hold + 1;
        const none = try t.gpa.alloc(bool, 0);
        defer t.gpa.free(none);
        t.check("10 move: never back into the part it just left", try f.pl.moveTo(t.gpa, f.lv(1).*, none) == null);
        f.lv(1).moved_at = 1000 - relocate_hold;
        const to = try f.pl.moveTo(t.gpa, f.lv(1).*, none);
        t.check("10 move: ...once the window has passed it may", to != null and to.?.part == 1);
    }
    {
        // three parts: the part it left is not the only one with room; inside the window only the cooldown stops it
        var f = try Lay.make(t, 3, 21 * C);
        defer f.deinit();
        try f.live(4, 64 * C, 21 * C);
        f.lv(1).came_from = 1;
        f.lv(1).moved_at = 1000 - relocate_hold + 1;
        var p = try f.plan(rq(60, 12 * C, null));
        defer p.deinit(t.gpa);
        const none = try t.gpa.alloc(bool, 0);
        defer t.gpa.free(none);
        const to = try f.pl.moveTo(t.gpa, f.lv(1).*, none);
        t.check("10 move: inside the window it stays put though a third part has the room", p.status == .full and to != null and to.?.part == 2);
    }
    {
        // the refusal is counted once a reason a request, again when the reason changes
        var f = try Lay.make(t, 2, 21 * C);
        defer f.deinit();
        f.lv(1).partial = true;
        for (0..2) |_| {
            var p = try f.plan(rq(60, 12 * C, null));
            defer p.deinit(t.gpa);
            try f.apply(&p);
        }
        t.check("10 move: a stream with a chunk in flight is not moved, refused once (partial)", f.pl.counters.refusals(.partial) == 1 and f.pl.holds.items.len == 0);
        f.lv(1).partial = false;
        f.lv(1).moved_at = f.pl.ticks - 1;
        var p = try f.plan(rq(60, 12 * C, null));
        defer p.deinit(t.gpa);
        try f.apply(&p);
        t.check("10 move: and counted again when the reason is new (no_target)", f.pl.counters.refusals(.no_target) == 1 and f.pl.counters.refusals(.partial) == 1);
    }
    {
        // the cheapest stream whose leaving lets the request in moves
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.live(1, 0, 8 * C);
        f.lv(1).valid = 7 * C;
        try f.live(5, 12 * C, 8 * C);
        f.lv(5).valid = 2 * C;
        try f.live(2, 20 * C, 12 * C);
        try f.live(3, P, 21 * C);
        f.pl.ticks = 1000;
        var p = try f.plan(rq(60, 12 * C, null));
        defer p.deinit(t.gpa);
        var who: u64 = 0;
        for (p.ops.items) |o| if (o == .move) {
            who = o.move.stream;
        };
        t.check("10 move: of the streams whose leaving lets it in, the one with the fewest rows to carry moves", p.status == .relocate and who == 5 and count(&p, .move) == 1);
    }
    {
        // kept states travel with the extent; their spill writes staged first; a failed move leaves all as it was
        var f = try Lay.make(t, 2, 21 * C);
        defer f.deinit();
        try f.kept(80, 0, 4 * C, 8 * C);
        f.kp(80).pending = true;
        const g80 = f.kp(80).gen;
        var p = try f.plan(rq(60, 12 * C, null));
        defer p.deinit(t.gpa);
        t.check("10 move: the extent's kept state with writes waiting is staged before the move", index(&p, .stage) != null and index(&p, .move) != null and index(&p, .stage).? < index(&p, .move).?);
        try f.apply(&p);
        var g = try f.pl.clone(t.gpa);
        defer g.deinit(t.gpa);
        _ = try g.finish(1, false);
        const l1 = g.lives.items[g.liveIndex(1).?];
        t.check("10 move: failed, the stream is where it was, the destination freed", l1.part == 0 and l1.base == 0 and g.holds.items.len == 0 and g.counters.failed == 1);
        _ = try f.pl.finish(1, true);
        const k = f.kp(80).*;
        t.check("10 move: its kept states moved with it (a new generation, the new part's salt)", k.part == 1 and k.base == f.lv(1).base and k.gen != g80 and f.pl.saltOf(80) == 1);
    }
    {
        var f = try Fx.initCfg(t, .{ .cap = 2 * P, .parts = 2, .carry = false });
        defer f.deinit();
        try f.live(1, 0, 8 * C);
        try f.live(2, 12 * C, 20 * C);
        try f.live(3, P, 21 * C);
        f.pl.ticks = 1000;
        var p = try f.plan(rq(60, 12 * C, null));
        defer p.deinit(t.gpa);
        t.check("10 move: carry off: never", p.status == .full and count(&p, .move) == 0);
    }
}

fn secThrash(t: *T) !void {
    // many rounds of streams coming and going in a pool that cannot hold them: moves do not answer one another
    var total: usize = 0;
    for (0..3) |variant| {
        var prng: std.Random.DefaultPrng = .init(variant + 40);
        const rnd = prng.random();
        var f = try Fx.init(t, 2, 48 * C);
        defer f.deinit();
        const Mv = struct { tick: i64, stream: u64, from: u8, to: u8 };
        var moves: std.ArrayList(Mv) = .empty;
        defer moves.deinit(t.gpa);
        var next: u64 = 1;
        var waiting: ?Request = null;
        for (0..2500) |_| {
            f.pl.tick();
            var i: usize = 0;
            while (i < f.pl.lives.items.len) {
                const l = f.pl.lives.items[i];
                if (rnd.uintLessThan(u32, 120) == 0) {
                    try f.release(l.stream);
                    continue;
                }
                f.pl.setValid(l.stream, l.valid + 256);
                i += 1;
            }
            if (waiting == null and rnd.boolean()) {
                waiting = .{ .stream = next, .slot = @intCast(next % 4096), .size = (rnd.uintLessThan(usize, 16) + 4) * C };
                next += 1;
            }
            const req = waiting orelse continue;
            for (0..2) |_| {
                var p = try f.plan(req);
                defer p.deinit(t.gpa);
                try f.apply(&p);
                switch (p.status) {
                    .placed => {
                        var d = try f.pl.planDecode(t.gpa, req.stream);
                        defer d.deinit(t.gpa);
                        try f.apply(&d);
                        f.pl.setDecoding(req.stream, true);
                        waiting = null;
                        break;
                    },
                    .relocate => for (p.ops.items) |o| if (o == .move) {
                        try moves.append(t.gpa, .{ .tick = f.pl.ticks, .stream = o.move.stream, .from = o.move.src_part, .to = o.move.dst_part });
                        _ = try f.pl.finish(o.move.stream, true);
                    },
                    .full => break,
                }
            }
        }
        var gaps = true;
        var back: usize = 0;
        var counter: usize = 0;
        for (moves.items, 0..) |a, i| for (moves.items[i + 1 ..]) |b| {
            const dt = b.tick - a.tick;
            if (b.stream == a.stream and dt < relocate_hold) {
                gaps = false;
                if (b.to == a.from) back += 1;
            }
            if (b.stream != a.stream and dt < relocate_hold and b.from == a.to) counter += 1;
        };
        t.check("10 thrash: a stream moves at most once a window, never back within it, no stream leaves a part another just entered", gaps and back == 0 and counter == 0);
        total += moves.items.len;
    }
    t.check("10 thrash: the sweep moves streams (it reaches the rule)", total >= 3);
}

fn secSpill(t: *T) !void {
    var b1: [128]u8 = undefined;
    var b2: [128]u8 = undefined;
    var b3: [128]u8 = undefined;
    const L = "boot-3|boot-9@2048";
    const s0 = try inPartM(&b1, L, 0, t.m);
    const s01 = try inPartM(&b2, s0, 1, t.m);
    const s1 = try inPartM(&b3, L, 1, t.m);
    t.check("10 spill: the part salts the first computation's name, forks untouched", std.mem.eql(u8, s1, "boot-3#1|boot-9@2048"));
    t.check("10 spill: salting a salted lineage replaces the part, never stacks", std.mem.eql(u8, s01, s1) and std.mem.count(u8, s01, "#") == 1);
    var b4: [128]u8 = undefined;
    t.check("10 spill: an empty lineage salts to the part alone", std.mem.eql(u8, try inPartM(&b4, "", 1, t.m), "#1"));
    var b5: [128]u8 = undefined;
    t.check("10 spill: the public salt is the unmutated one", std.mem.eql(u8, try inPart(&b5, "a#3|b@2048", 2), "a#2|b@2048"));
    const r: Root = .{ .plain = @splat(7) };
    t.check("10 spill: digest roots: parts differ, the same part agrees, re-salting replaces, unsalted unchanged", !std.mem.eql(u8, &r.inPart(0).digest(), &r.inPart(1).digest()) and std.mem.eql(u8, &r.inPart(0).inPart(1).digest(), &r.inPart(1).digest()) and std.mem.eql(u8, &r.inPart(null).digest(), &r.plain));
    var u = try Fx.init(t, 1, 32 * C);
    defer u.deinit();
    try u.kept(7, 0, 8 * C, 8 * C);
    t.check("10 spill: an unsharded store's names are unchanged (no salt)", u.pl.saltOf(7) == null);
    // a carried conversation's later states are named for its new part; the state it came from keeps its own
    var f = try rescueLayout(t, 8 * C);
    defer f.deinit();
    var p = try f.admit(rq(50, 16 * C, 7));
    defer p.deinit(t.gpa);
    _ = try f.pl.finish(50, true);
    try f.pl.applyOps(t.gpa, &.{try f.pl.keepOp(71, 50, 12 * C)});
    t.check("10 spill: a carried state's later blocks are its new part's, the source's stay its own", f.pl.saltOf(71) == 1 and f.pl.saltOf(7) == 0);
}

fn secMisc(t: *T) !void {
    {
        // a refused placement leaves the planner as it was (lanes' placeKept used to stale its victims anyway)
        var f = try Fx.init(t, 1, 32 * C);
        defer f.deinit();
        try f.kept(1, 24 * C, 4 * C, 28 * C);
        try f.live(3, 0, 24 * C);
        try f.kept(7, 0, 8 * C, 24 * C);
        var p = try f.plan(rq(50, 16 * C, 7));
        defer p.deinit(t.gpa);
        const before = f.pl.digest();
        try f.apply(&p);
        t.check("misc: a placement that fails changes nothing", p.status == .full and std.mem.eql(u8, &before, &f.pl.digest()) and !f.kp(1).stale);
    }
    {
        // lanes.zig's growth slack, one part: a kept state keeps room to grow while the pool allows
        var f = try Fx.init(t, 1, 128 * C);
        defer f.deinit();
        try f.kept(1, 0, 48 * C, 50 * C);
        var p = try f.plan(rq(5, 4 * C, null));
        defer p.deinit(t.gpa);
        t.check("misc: a placement leaves a kept state room to grow (placeSoft)", p.base == 50 * C + growth_rows);
    }
    {
        // 629c524: a prefix two conversations share: resuming the shorter in place would stale the longer
        var f = try Fx.init(t, 1, 1 << 20);
        defer f.deinit();
        try f.kept(1, 0, 18432, 430080);
        try f.kept(2, 0, 395264, 430080);
        var p = try f.plan(rq(5, 55296, 1));
        defer p.deinit(t.gpa);
        t.check("misc: the shared prefix's second conversation is copied, the first one's later state kept", p.mode == .copy and count(&p, .stale) == 0);
        var q = try f.plan(rq(5, 432128, 2));
        defer q.deinit(t.gpa);
        t.check("misc: the first conversation's next turn resumes in place", q.mode == .in_place);
    }
    {
        // a restore's destination: in its recorded part, invisible until done, then a kept state
        var f = try Fx.init(t, 2, 32 * C);
        defer f.deinit();
        var p = try f.pl.planRestore(t.gpa, 9, 8 * C, 1);
        defer p.deinit(t.gpa);
        try f.apply(&p);
        var q = try f.plan(rq(5, 32 * C, null));
        defer q.deinit(t.gpa);
        t.check("misc: a hold is no room (a 32C request does not fit its part)", q.part == 0 and f.pl.holds.items[0].part == 1);
        try f.pl.applyOps(t.gpa, &.{try f.pl.restoredOp(9, 44, 6 * C)});
        t.check("misc: restored, it is a kept state of its part", f.pl.holds.items.len == 0 and f.kp(44).part == 1 and f.kp(44).at == 6 * C);
        var r = try f.pl.planRestore(t.gpa, 10, 8 * C, 0);
        defer r.deinit(t.gpa);
        try f.apply(&r);
        try f.pl.applyOps(t.gpa, &.{try f.pl.unholdOp(10)});
        t.check("misc: an abandoned restore frees its rows", f.pl.holds.items.len == 0);
        f.pl.setPending(44, true);
        try f.pl.applyOps(t.gpa, &.{try f.pl.dropOp(44)});
        t.check("misc: a dropped kept state is gone", f.pl.keptIndex(44) == null);
    }
    {
        t.check("misc: the native kept-point rule", eligible(2048, 4096, 1024) and !eligible(2048, 2048, 0) and !eligible(1024, 4096, 0) and !eligible(4096, 6000, 2048) and !eligible(0, 4096, 0));
        const a = try planeSpan(0, 3001, 4);
        const b = try planeSpan(2048, 2049, 2);
        t.check("misc: a plane's rows of a token range include the partly filled last row", a.rows == 751 and a.row == 0 and b.row == 1024 and b.rows == 1025 and (try planeSpan(4096, 7, 1)).rows == 7);
        t.check("misc: a plane span refuses a zero ratio and an unaligned base (an error, never UB)", planeSpan(0, 1, 0) == error.BadRatio and planeSpan(3, 1, 2) == error.Unaligned);
    }
    {
        // copy, carry and move: a copy and a carry leave the source; a failed carry leaves nothing resumed
        var f = try rescueLayout(t, 8 * C);
        defer f.deinit();
        var p = try f.admit(rq(50, 16 * C, 7));
        defer p.deinit(t.gpa);
        t.check("misc: an in-flight carry pins its stream (loading) and keeps it from release", f.lv(50).loading and if (f.pl.planRelease(t.gpa, 50)) |_| false else |err| err == error.InFlight);
        const r = try f.pl.finish(50, false);
        t.check("misc: a failed carry: nothing resumed, the source intact, counted", !r.resumed and f.lv(50).valid == 0 and !f.kp(7).stale and f.pl.counters.failed == 1);
    }
    {
        // descriptors: the layout version, owner part, base, reserved, valid, slot, generation
        var f = try Fx.init(t, 2, 32 * C);
        defer f.deinit();
        var p = try f.admit(rq(5, 8 * C, null));
        defer p.deinit(t.gpa);
        const ds = try f.pl.descs(t.gpa);
        defer t.gpa.free(ds);
        t.check("misc: descriptors name every extent", ds.len == 2 and ds[0].kind == .live and ds[0].version == layout_version and ds[0].slot == 5 and ds[0].reserved == 8 * C and ds[1].kind == .twin and ds[1].part == 1);
        const st = try f.pl.partStats(t.gpa, 1);
        t.check("misc: part stats", st.twin_rows == 8 * C and st.free == 24 * C and st.live == 0);
    }
}

// lifetimes: in-flight sources, moved ends, refusal counting, fail-closed finish, release mid-chunk --------------------

fn secLifetimes(t: *T) !void {
    const P = 32 * C;
    {
        // an in-flight carry's source: no victim, no room, no stale or drop until finish
        var f = try rescueLayout(t, 8 * C);
        defer f.deinit();
        var p = try f.admit(rq(50, 16 * C, 7));
        defer p.deinit(t.gpa);
        try f.live(2, P + 16 * C, 16 * C);
        var q = try f.plan(rq(60, 8 * C, null));
        defer q.deinit(t.gpa);
        t.check("life source: an in-flight carry's source is no victim and no room: the request waits, nothing staled", q.status != .placed and count(&q, .stale) == 0);
        try f.apply(&q);
        const g7 = f.kp(7).gen;
        t.check("life source: still kept after the waiting plan", !f.kp(7).stale);
        const st = f.pl.applyOps(t.gpa, &.{.{ .stale = .{ .kept = 7, .gen = g7 } }});
        t.check("life source frame: a frame staling a source being read fails closed", if (st) |_| false else |err| err == error.InFlight);
        const dr = f.pl.applyOps(t.gpa, &.{.{ .drop = .{ .kept = 7, .gen = g7 } }});
        t.check("life source frame: ...and one dropping it (dropOp refuses too)", (if (dr) |_| false else |err| err == error.InFlight) and f.pl.dropOp(7) == error.InFlight);
        _ = try f.pl.finish(50, true);
        var r = try f.plan(rq(60, 8 * C, null));
        defer r.deinit(t.gpa);
        t.check("life source: finished, the source may give its room again", r.status == .placed and r.base == 24 * C and staledAre(&r, &.{7}));
    }
    {
        // a live stream whose extent holds a copy's or carry's source is not moved (refused `pinned`)
        var f = try heldLayout(t);
        defer f.deinit();
        var p = try f.admit(rq(50, 16 * C, 7));
        defer p.deinit(t.gpa);
        try f.live(2, P + 16 * C, 12 * C);
        f.pl.ticks = 1000;
        var q = try f.plan(rq(60, 8 * C, null));
        defer q.deinit(t.gpa);
        try f.apply(&q);
        t.check("life pinned: a stream holding a source being read is not moved, refused as pinned", q.status == .full and count(&q, .move) == 0 and f.pl.counters.refusals(.pinned) == 1);
        var g = try Fx.init(t, 2, P);
        defer g.deinit();
        try g.live(3, 0, 8 * C);
        try g.kept(7, 0, 4 * C, 8 * C);
        try g.live(50, 8 * C, 8 * C);
        try g.pl.applyOps(t.gpa, &.{.{ .copy = .{ .kept = 7, .kept_gen = g.kp(7).gen, .stream = 50, .src = 0, .dst = 8 * C, .rows = 4 * C } }});
        const mv = g.pl.applyOps(t.gpa, &.{.{ .move = .{ .stream = 3, .gen = g.lv(3).gen, .src = 0, .src_part = 0, .dst = P, .dst_part = 1, .size = 8 * C, .rows = 4 * C, .hold_gen = g.pl.next_gen, .tick = 0 } }});
        t.check("life pinned frame: a frame moving the rows a copy reads fails closed", if (mv) |_| false else |err| err == error.InFlight);
    }
    {
        // a moved kept state's extent stops at its stream's: an end a larger stream left never runs past the part
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.live(1, 0, 8 * C);
        f.lv(1).valid = 6 * C;
        try f.kept(80, 0, 4 * C, 20 * C);
        try f.live(2, 12 * C, 20 * C);
        try f.live(3, P, 21 * C);
        f.pl.ticks = 1000;
        var p = try f.plan(rq(60, 12 * C, null));
        defer p.deinit(t.gpa);
        try f.apply(&p);
        _ = try f.pl.finish(1, true);
        const k = f.pl.keptAt(80).?;
        t.check("life end: moved, a kept state's end is its stream's extent's, inside the part", p.status == .relocate and k.part == 1 and k.base == f.lv(1).base and k.end == f.lv(1).base + 8 * C and k.end <= f.pl.partEnd(1));
        // random: part 0 holds stream 1 (a rows, its kept states with ends a larger stream left), a gap g, stream 2;
        // a request of a + g fits only once stream 1 moves to part 1, which has its rows and the headroom
        var bad: usize = 0;
        var moved: usize = 0;
        for (0..300) |seed| {
            var prng: std.Random.DefaultPrng = .init(seed + 900);
            const rnd = prng.random();
            const a = (rnd.uintLessThan(usize, 9) + 2) * C;
            const gap = (rnd.uintLessThan(usize, 4) + 3) * C;
            const free1 = a + 2 * C + rnd.uintLessThan(usize, gap / C - 2) * C;
            var g = try Fx.init(t, 2, P);
            defer g.deinit();
            try g.live(1, 0, a);
            g.lv(1).valid = rnd.uintLessThan(usize, a);
            for (0..rnd.uintLessThan(usize, 3) + 1) |j| {
                const at = (rnd.uintLessThan(usize, a / C) + 1) * C;
                if (g.pl.kept.items.len > 0 and g.pl.kept.items[g.pl.kept.items.len - 1].at == at) continue;
                try g.kept(80 + j, 0, at, at + rnd.uintLessThan(usize, P / C - at / C + 1) * C);
            }
            try g.live(2, a + gap, P - a - gap);
            try g.live(3, P, P - free1);
            g.lv(2).decoding = false; // (only stream 1 may move)
            g.lv(3).decoding = false;
            g.pl.ticks = 1000;
            var q = try g.plan(rq(60, a + gap, null));
            defer q.deinit(t.gpa);
            if (q.status != .relocate) {
                bad += 1;
                continue;
            }
            g.apply(&q) catch {
                bad += 1;
                continue;
            };
            _ = try g.pl.finish(1, true);
            moved += 1;
            const ds = try g.pl.descs(t.gpa);
            defer t.gpa.free(ds);
            for (ds) |d| if (d.base < g.pl.lo(d.part) or d.base + d.reserved > g.pl.partEnd(d.part)) {
                bad += 1;
            };
        }
        t.check("life end random: after random moves every extent (kept ones too) stays inside its part", bad == 0 and moved == 300);
    }
    {
        // refusals: once a reason a request, though requests wait in turn and are planned again
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.live(1, 0, 24 * C);
        try f.kept(7, 24 * C, 1024, 32 * C);
        try f.live(2, P, 32 * C);
        for (0..3) |_| for ([_]u64{ 50, 51 }) |s| {
            var p = try f.plan(rq(s, 16 * C, 7));
            defer p.deinit(t.gpa);
            try f.apply(&p);
        };
        const c = f.pl.counters;
        t.check("life refusals: two requests waiting in turn count each reason once each", c.refusals(.small) == 2 and c.refusals(.no_target) == 2 and c.no_part == 2);
        var g = try Fx.init(t, 2, P);
        defer g.deinit();
        try g.live(1, 0, 8 * C);
        g.lv(1).partial = true;
        try g.live(2, 12 * C, 20 * C);
        try g.live(3, P, 21 * C);
        g.pl.ticks = 1000;
        for (0..3) |_| for ([_]u64{ 60, 61 }) |s| {
            var p = try g.plan(rq(s, 12 * C, null));
            defer p.deinit(t.gpa);
            try g.apply(&p);
        };
        t.check("life refusals: a relocation refused for two alternating requests counts twice, not six times", g.pl.counters.refusals(.partial) == 2);
        g.lv(1).partial = false;
        g.pl.forget(61);
        var p = try g.plan(rq(60, 12 * C, null));
        defer p.deinit(t.gpa);
        try g.apply(&p);
        _ = try g.pl.finish(1, true);
        var q = try g.plan(rq(60, 12 * C, null));
        defer q.deinit(t.gpa);
        try g.apply(&q);
        t.check("life refusals: placed, the request's record is gone (and a forgotten one's)", q.status == .placed and g.pl.seen.items.len == 0);
    }
    {
        // finish checks before it changes anything
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.live(1, 0, 8 * C);
        f.lv(1).valid = 6 * C;
        try f.live(2, 12 * C, 20 * C);
        try f.live(3, P, 21 * C);
        f.pl.ticks = 1000;
        var p = try f.plan(rq(60, 12 * C, null));
        defer p.deinit(t.gpa);
        try f.apply(&p);
        const h = f.pl.holds.pop().?;
        const before = f.pl.digest();
        const r = f.pl.finish(1, false);
        t.check("life finish: an unknown hold fails before anything changes", (if (r) |_| false else |err| err == error.UnknownHold) and std.mem.eql(u8, &before, &f.pl.digest()) and f.pl.counters.failed == 0 and f.lv(1).loading);
        try f.pl.holds.append(t.gpa, h);
        _ = try f.pl.finish(1, true);
    }
    {
        // a stream released with a chunk in flight (a failed fillChunk): its twins and extent still go
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        var p = try f.admit(rq(1, 8 * C, null));
        defer p.deinit(t.gpa);
        f.pl.setPartial(1, true);
        const ok = if (f.release(1)) |_| true else |_| false;
        t.check("life release: a stream mid-chunk is still released, its twins with it", ok and f.pl.twins.items.len == 0 and f.pl.liveIndex(1) == null);
    }
    {
        // generations and alignment on the remaining ops
        var f = try Fx.init(t, 2, P);
        defer f.deinit();
        try f.kept(7, 0, 8 * C, 8 * C);
        const d = f.pl.applyOps(t.gpa, &.{.{ .drop = .{ .kept = 7, .gen = f.kp(7).gen + 9 } }});
        t.check("life gen: a drop for a reused kept id fails closed", if (d) |_| false else |err| err == error.StaleGeneration);
        var p = try f.pl.planRestore(t.gpa, 9, 8 * C, 1);
        defer p.deinit(t.gpa);
        try f.apply(&p);
        var op = try f.pl.restoredOp(9, 44, 3 * C);
        op.restored.at = 3 * C + 5;
        const r = f.pl.applyOps(t.gpa, &.{op});
        t.check("life gen: a restore published at an unaligned point fails closed", if (r) |_| false else |err| err == error.BadOp);
    }
}

const sections = .{
    .{ "1-8 parts", secParts },
    .{ "9 twins", secTwins },
    .{ "10 rescue", secRescue },
    .{ "10 pins", secPins },
    .{ "10 pressure", secPressure },
    .{ "10 others", secOthers },
    .{ "10 native", secNative },
    .{ "10 invariants", secInvariants },
    .{ "10 two ranks", secTwoRanks },
    .{ "10 relocate", secRelocate },
    .{ "10 thrash", secThrash },
    .{ "10 spill", secSpill },
    .{ "misc", secMisc },
    .{ "lifetimes", secLifetimes },
};

fn suite(gpa: Allocator, m: Mutant, verbose: bool) T {
    var t: T = .{ .gpa = gpa, .m = m, .verbose = verbose };
    inline for (sections) |s| t.section(s[0], s[1]);
    return t;
}

test "kv shard checks (GLM tools/kv_shard_check.py, native reservations)" {
    const t = suite(std.testing.allocator, .none, true);
    if (t.fail > 0) std.debug.print("{d} of {d} checks failed\n", .{ t.fail, t.fail + t.pass });
    try std.testing.expectEqual(@as(usize, 0), t.fail);
    try std.testing.expectEqual(@as(usize, 0), t.crash);
    try std.testing.expect(t.pass > 120);
}

/// The check written for each mutant (its name's start): the mutant must fail it, not merely some check or a crash.
fn killer(m: Mutant) []const u8 {
    return switch (m) {
        .none => unreachable,
        .evict_any_part => "1 evictions only there",
        .cross_part => "1 no twin where the other part has no free rows",
        .no_part_end_inplace => "5 past its part's end",
        .no_growth_slack => "misc: a placement leaves a kept state room to grow",
        .no_prefix_guard => "misc: the shared prefix's second conversation",
        .failed_mutates => "misc: a placement that fails changes nothing",
        .pressure_no_eviction_condition => "10 no pressure: its part fits it without evicting",
        .pressure_no_headroom => "10 pressure: without headroom no carry",
        .pressure_may_evict => "10 pressure: no free target",
        .small_ignored => "10 rescue: a hit under carry_min",
        .small_uses_size => "10 pressure: a hit under carry_min",
        .min_0 => "10 others: a hit under carry_min",
        .rescue_off => "10 rescue: a full part no longer drops the hit",
        .others_off => "10 others: the part has no free rows for the copy",
        .counted_at_plan => "10 counters: nothing counted at planning",
        .never_counted => "10 pressure: counted as pressure only",
        .carry_rows_minus_1 => "10 rescue: the carry names the hit's rows",
        .carry_before_admit => "10 rescue: the extent first, then the carry",
        .pins_no_loading => "10 pins: rows something pins are not carried",
        .pins_no_pinned => "life pinned: a stream holding a source being read",
        .pins_no_twin => "10 pins: rows something pins are not carried",
        .pins_no_partial => "10 move: a stream with a chunk in flight is not moved",
        .no_cooldown => "10 move: a stream that moved less than relocate_hold",
        .move_back_allowed => "10 move: never back into the part it just left",
        .no_entered_guard => "10 move: no stream leaves a part another entered",
        .hold_1 => "10 thrash: a stream moves at most once a window",
        .no_cheapest => "10 move: of the streams whose leaving lets it in",
        .refusal_recounted => "life refusals: a relocation refused for two alternating requests",
        .no_headroom_in_move => "10 move: free rows without the headroom are no room",
        .no_stage_on_move => "10 move: the extent's kept state with writes waiting is staged",
        .no_stage_on_stale => "2 a kept state with spill writes waiting is staged",
        .twin_not_all_or_none => "9 twins: all or none",
        .twin_evicts_kept => "9 twins: never evict a kept state",
        .twin_pinned_room => "10 two ranks: the follower applying rank 0's ops",
        .twin_kept_at_decode => "9 twins: given back before the first decode round",
        .salt_stacks => "10 spill: salting a salted lineage",
        .salt_unsharded => "10 spill: an unsharded store's names are unchanged",
        .no_salt => "10 spill: a carried state's later blocks",
        .no_generation_check => "10 follower: a stale generation fails closed",
        .copy_across_parts => "10 follower: a copy across parts fails closed",
        .carry_when_off => "10 move: carry off: never",
        .carry_releases_source => "10 rescue: finished, the stream resumes",
        .move_keeps_source => "10 move: finished, the stream is in part 1",
        .order_ignores_live => "4 fewer live streams first",
        .cheapest_ignored => "1 mirrored: the part whose newest needed victim is oldest",
        .flight_source_unprotected => "life source: an in-flight carry's source is no victim",
        .move_end_unclamped => "life end random: after random moves",
    };
}

test "every mutant is killed by its own check (GLM tools/kv_shard_mutants.sh)" {
    var survived: usize = 0;
    for (std.enums.values(Mutant)) |m| {
        if (m == .none) continue;
        var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
        defer arena.deinit();
        const t = suite(arena.allocator(), m, false);
        if (!t.killedBy(killer(m))) {
            std.debug.print("SURVIVED {s}: \"{s}\" passed ({d} checks failed, {d} sections crashed)\n", .{ @tagName(m), killer(m), t.fail, t.crash });
            survived += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), survived);
}

test "every public declaration compiles" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Planner);
}
