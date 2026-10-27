//! DeepSeek-V4.1-Flash behind TensorFold 1.0's lane core (core/lanes): the round loop picks each round's streams and
//! windows and keeps the accepted rows; this backend runs them on the port's model (model.zig) with the served lane's
//! arithmetic. Each stream has a pool slot (its window rings, positional stores and drafter rings) and an extent of
//! the shared compressed plane (first fit, chunk-aligned, as large as its prompt, its budget and a round's rows), and
//! its ids by position on both ranks (Engram's n-grams). verify is one round over every window (round.zig) with each
//! row drawn by the served sampler (sampling.zig) at its keyed position, and the drafting streams' window rows
//! absorbed into their drafter rings at once (the served eager absorb: the rows past the kept ones sit where the
//! drafter never reads, and the next window writes over them); draft is one batched drafter pass (draft.zig) for
//! every stream asked, the drafts held for the next round with their confidences, whose sigmoids are each draft's
//! chance of landing once the earlier ones did (probabilities: their running products, the chance that a draft and
//! every earlier one land; the round loop's allocation trims the windows by them). The caches are
//! positional, so keep only checks that a path is a prefix. Rank 0 runs the core; before each primitive it sends rank
//! 1 the primitive's inputs over the ranks' link, and rank 1 (follow) runs the same primitive: both ranks issue the
//! same kernels and collectives in the same order. The drafted reply is the serial one: every row is drawn with its
//! stream's key at its own position, whatever the round's composition.
const std = @import("std");
const lanes = @import("lanes");
const model = @import("model.zig");
const round = @import("round.zig");
const draft = @import("draft.zig");
const sampling = @import("sampling.zig");
const penalty_pool = @import("penalty_pool.zig");
const link = @import("link.zig");
const cuda = @import("cuda");
// PETABRIDGE (k2a-pb1) spill P6: MiaAI-Lab 0016/0019's spill tier (off unless native.zig / tf-dsv41-lanes open it)
const spill_tier = @import("spill_tier.zig");

const be = lanes.backend;
const Model = model.Model;
const max_streams = model.max_streams;
const max_rows = round.max_rows;

/// The spill tier on rank 0 (the prompt cache's Tier) and on a follower, over this rank's planes (spill_tier.zig).
/// PETABRIDGE (k2a-pb1) spill P6: 0016 verbatim.
pub const SpillTier = spill_tier.Leader(Model.SpillDev);
pub const SpillFollower = spill_tier.Follower(Model.SpillDev);

/// A window of the pending row and the drafter's block: the widest one the gates checked row for row.
pub const window_rows = 6;

/// The image rows a fill_images frame carries at most.
pub const image_piece = 32 << 10;

/// A gate's diagnostic (the tools' --logits-sha 1): each prompt's logits' sha256 on stderr.
pub var logits_sha = false;

/// The primitives rank 0 sends rank 1 (each frame: kind u8, step u64, its fields little-endian).
/// PETABRIDGE (k2a-pb1) spill P6: with the spill tier on (only then), rank_wire's frames (spill_tier: the ranks'
/// agreement, the keep/share/touch/forget notices, every restore's transaction, guards): rank_wire v2 (0019), so 0016's
/// untracked Kind 11 is not taken.
pub const Kind = enum(u8) { fill_begin = 1, fill_chunk = 2, verify = 3, pass = 4, snap_save = 5, snap_restore = 6, snap_drop = 7, fill_images = 8, done = 9, rank_wire = 10 };

comptime {
    std.debug.assert(@intFromEnum(Kind.rank_wire) == @import("rank_wire.zig").lanes_kind);
}

/// A frame being written.
pub const Wire = struct {
    buf: std.ArrayList(u8) = .empty,

    pub fn begin(w: *Wire, gpa: std.mem.Allocator, kind: Kind, step: u64) !void {
        w.buf.clearRetainingCapacity();
        try w.int(gpa, u8, @backingInt(kind));
        try w.int(gpa, u64, step);
    }

    pub fn int(w: *Wire, gpa: std.mem.Allocator, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try w.buf.appendSlice(gpa, &b);
    }

    pub fn ids(w: *Wire, gpa: std.mem.Allocator, v: []const i32) !void {
        try w.int(gpa, u32, @intCast(v.len));
        try w.buf.ensureUnusedCapacity(gpa, v.len * 4);
        for (v) |x| try w.int(gpa, i32, x);
    }

    pub fn deinit(w: *Wire, gpa: std.mem.Allocator) void {
        w.buf.deinit(gpa);
    }
};

/// A frame being read.
pub const Reader = struct {
    b: []const u8,
    at: usize = 0,

    pub fn int(r: *Reader, comptime T: type) !T {
        const n = @sizeOf(T);
        if (r.at + n > r.b.len) return error.ShortFrame;
        const v = std.mem.readInt(T, r.b[r.at..][0..n], .little);
        r.at += n;
        return v;
    }

    /// The frame's bytes after those read.
    pub fn rest(r: *Reader) []const u8 {
        const b = r.b[r.at..];
        r.at = r.b.len;
        return b;
    }

    /// A list of ids written by Wire.ids, into `out` (replacing it).
    pub fn ids(r: *Reader, gpa: std.mem.Allocator, out: *std.ArrayList(i32)) !void {
        const n = try r.int(u32);
        try out.resize(gpa, n);
        for (out.items) |*x| x.* = try r.int(i32);
    }
};

/// A window of a round as both ranks build it: its stream's slot and extent, its first position, whether its rows'
/// taps go into the drafter, its rows' ids.
pub const Win = struct { slot: usize, base: usize, end: usize, start: usize, absorb: bool, ids: []const i32 };

/// The rows of a round (round.Rows' slices point here).
pub const Built = struct {
    ids: [max_rows]i64 = undefined,
    pos: [max_rows]i64 = undefined,
    slots: [max_rows]i64 = undefined,
    bases: [max_rows]i64 = undefined,
    ends: [max_rows]i64 = undefined,
    wins: [max_streams]round.Window = undefined,
    absorb: [max_streams]bool = undefined,
    r: usize = 0,
    n: usize = 0,

    /// Each window's ids written at its positions in its slot's sequence (the rows past them dropped: the rejected
    /// rows of the last round), then the round's rows in window order.
    pub fn build(b: *Built, gpa: std.mem.Allocator, seqs: *[max_streams]std.ArrayList(i32), wins: []const Win) !round.Rows {
        if (wins.len == 0 or wins.len > max_streams) return error.BadRound;
        b.r = 0;
        b.n = wins.len;
        for (wins, 0..) |w, i| {
            if (w.slot >= max_streams or w.ids.len == 0) return error.BadRound;
            if (b.r + w.ids.len > max_rows) return error.WindowTooWide;
            const seq = &seqs[w.slot];
            if (seq.items.len < w.start) return error.SequenceGap;
            try seq.resize(gpa, w.start + w.ids.len);
            @memcpy(seq.items[w.start..], w.ids);
            for (w.ids, 0..) |id, j| {
                b.ids[b.r + j] = id;
                b.pos[b.r + j] = @intCast(w.start + j);
                b.slots[b.r + j] = @intCast(w.slot);
                b.bases[b.r + j] = @intCast(w.base);
                b.ends[b.r + j] = @intCast(w.end);
            }
            b.wins[i] = .{ .row = b.r, .n = w.ids.len, .seq = seq.items[0 .. w.start + w.ids.len] };
            b.absorb[i] = w.absorb;
            b.r += w.ids.len;
        }
        const R = b.r;
        return .{ .ids = b.ids[0..R], .pos = b.pos[0..R], .slots = b.slots[0..R], .bases = b.bases[0..R], .ends = b.ends[0..R], .windows = b.wins[0..b.n] };
    }
};

/// The first chunk-aligned extent of `size` positions in [0, cap) that overlaps none of `used`.
/// Whether [lo, hi) meets any of `used`.
fn clashes(used: []const [2]usize, lo: usize, hi: usize) bool {
    for (used) |u| if (lo < u[1] and u[0] < hi) return true;
    return false;
}

/// Whether one of the prompt cache's marks has its chunk end at `at` (each kept at the chunk end at or before it).
fn markEnds(marks: []const u32, at: usize) bool {
    for (marks) |w| if (w / model.chunk_rows * model.chunk_rows == at) return true;
    return false;
}

pub fn place(used: []const [2]usize, size: usize, cap: usize) ?usize {
    var base: usize = 0;
    while (base + size <= cap) {
        var clash: ?usize = null;
        for (used) |u| {
            if (base < u[1] and u[0] < base + size) clash = @max(clash orelse 0, u[1]);
        }
        const past = clash orelse return base;
        base = std.mem.alignForward(usize, past, model.chunk_rows);
    }
    return null;
}

/// Soft growth space, adapted from Bertholomus TensorFold PR26 (Apache-2.0); no extra KV allocation.
pub const growth_rows = 16 * model.chunk_rows;

/// Prefer growth room and full extents; under pressure fall back to the exact retained rows.
pub fn placeSoft(gpa: std.mem.Allocator, live: []const [2]usize, kept: []const [3]usize, size: usize, cap: usize) !?usize {
    var used: std.ArrayList([2]usize) = .empty;
    defer used.deinit(gpa);
    try used.ensureTotalCapacity(gpa, live.len + kept.len);
    used.appendSliceAssumeCapacity(live);
    for (kept) |k| used.appendAssumeCapacity(.{ k[0], @max(k[2], k[0] + k[1]) +| growth_rows });
    if (place(used.items, size, cap)) |base| return base;
    used.shrinkRetainingCapacity(live.len);
    for (kept) |k| used.appendAssumeCapacity(.{ k[0], @max(k[2], k[0] + k[1]) });
    if (place(used.items, size, cap)) |base| return base;
    used.shrinkRetainingCapacity(live.len);
    for (kept) |k| used.appendAssumeCapacity(.{ k[0], k[0] + k[1] });
    return place(used.items, size, cap);
}

/// The token a row's logits draw at absolute position `position`: greedy without sampling (or at temperature 0),
/// else the served sampler with the stream's key.
pub fn draw(row: []const f32, position: u64, s: ?lanes.Sampling) !u32 {
    const sm = s orelse return @intCast(sampling.argmax(row));
    if (sm.temperature <= 0) return @intCast(sampling.argmax(row));
    const t = try sampling.sampleRow(row, position, .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p });
    return @intCast(t);
}

/// Where rank 0's backend calls spend their time (ns, summed; --profile): a round's frame to rank 1, its forward
/// and absorb (synchronized apart), the logits' copy to the host and the draws; the drafter passes; the prefills.
pub const Profile = struct {
    send: u64 = 0,
    enqueue: u64 = 0,
    forward: u64 = 0,
    absorb: u64 = 0,
    logits: u64 = 0,
    sample: u64 = 0,
    pass: u64 = 0,
    prefill: u64 = 0,
    rounds: u64 = 0,
    rows: u64 = 0,
    passes: u64 = 0,
    pass_streams: u64 = 0,
    prefills: u64 = 0,
};

/// A round-cost table "30.0,35.2,..." (TF_DS_ROUND_MS's form): ms by rows from 1.
pub fn parseMs(a: std.mem.Allocator, text: []const u8) ![]const f64 {
    var out: std.ArrayList(f64) = .empty;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |x| try out.append(a, try std.fmt.parseFloat(f64, std.mem.trim(u8, x, " ")));
    if (out.items.len == 0) return error.BadRoundMs;
    return out.items;
}

/// A stream's place in the pool and the drafts it holds.
const Lane = struct {
    slot: usize,
    base: usize,
    end: usize,
    from_snap: u64 = 0, // the kept state this stream's pass resumed from (0: none); retention
    last_snap: u64 = 0, // the last state this stream kept; retention
    filled: usize = 0, // confirmed prefill positions; observation only
    first: u64 = 0, // the prompt's draw (a handle)
    held: [draft.max_block]u32 = undefined,
    confs: [draft.max_block]f32 = undefined,
    nheld: usize = 0,
};

extern "c" fn exp(x: f64) f64;

/// A kept prompt state (the prompt cache's saved state, kept at a chunk end the prompt pass reached before its
/// replay): the slot's own part on the device (Model.snapCopy), and the extent its compressed rows stay in,
/// [base, base + at), held back from placement while it lives. Stale: a placement needed that room (its restore fails
/// and the prompt runs from 0).
pub const Snap = struct {
    id: u64,
    at: usize,
    base: usize,
    end: usize = 0,
    buf: cuda.DeviceBuffer,
    stale: bool = false,
    complete: bool = false,
    decoded: bool = false,
    // PETABRIDGE (k2a-pb1) retention: how often a prompt pass resumed this state (a conversation's newest state carries
    // its conversation's count forward), when it was last kept or resumed (retentionTick), and whether a later state of
    // the same conversation superseded it (a pass that resumed here, or an earlier mark of that pass, kept further on)
    resumes: u32 = 0,
    used: u64 = 0,
    superseded: bool = false,
    // PETABRIDGE (k2a-pb1) spill P6: MiaAI-Lab 0016's fields
    slot: u32 = 0, // the slot it was kept from (the spill tier's keep notice: a follower reads the prompt's ids there)
    loading: bool = false, // a restore's destination: its rows held back, not usable until every rank installed it
    restored: bool = false, // read back from the spill tier
};

/// PETABRIDGE (k2a-pb1) spill P6 review: the positions a write of positions [lo, hi) can touch in the pool, for the
/// spill tier's bump. Compression ratios here are 1 or 2 (a ratio-2 plane stores position p at row p/2): an odd lo
/// completes the group whose row also holds lo-1, and an incomplete group's row at the end holds up to the next even
/// position, so both ends widen to even. This makes the bump cover those rows directly instead of relying on placement
/// keeping kept states off a live extent and on kept points being even.
fn bumpSpan(lo: usize, hi: usize) [2]usize {
    return .{ lo & ~@as(usize, 1), std.mem.alignForward(usize, hi, 2) };
}

test "spill bump span widens to the ratio-2 groups a write can touch" {
    try std.testing.expectEqual([2]usize{ 4096, 4098 }, bumpSpan(4096, 4098));
    try std.testing.expectEqual([2]usize{ 4096, 4100 }, bumpSpan(4097, 4099));
    try std.testing.expectEqual([2]usize{ 0, 2 }, bumpSpan(0, 1));
    try std.testing.expectEqual([2]usize{ 6142, 6144 }, bumpSpan(6143, 6144));
}

/// PETABRIDGE (k2a-pb1) retention. Upstream's placeKept gave up kept states oldest first, so an agent's long main
/// conversation, idle while its subagents and one-shot jobs ran, was always the oldest and always went first
/// (MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks#27). With retention on (TF_DS_RETENTION unset or not "0") the
/// victim is the least recently used state of the lowest tier: never resumed, then superseded (an older prefix of a
/// conversation that moved on; a shared prefix that many conversations resume stays recently used), then the resumed
/// conversations' newest states. Rank 0 alone decides placement (the workers get the base in their frames), so this
/// changes no frame. Engine thread only.
var retention_clock: u64 = 0;
var retention_on: ?bool = null;

fn retentionTick() u64 {
    retention_clock += 1;
    return retention_clock;
}

fn retention() bool {
    if (retention_on == null) retention_on = if (std.c.getenv("TF_DS_RETENTION")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    return retention_on.?;
}

/// A kept state's retention tier: the lowest goes first.
fn retentionTier(sn: *const Snap) u2 {
    if (sn.resumes == 0) return 0;
    if (sn.superseded) return 1;
    return 2;
}

/// Whether `a` goes before `b`: the lower tier, then the least recently used (ties: the caller's oldest-first order).
fn retentionBefore(a: *const Snap, b: *const Snap) bool {
    const ta = retentionTier(a);
    const tb = retentionTier(b);
    if (ta != tb) return ta < tb;
    return a.used < b.used;
}

/// First tokens a handle names (each prompt's draw, read back right after its prefill).
const ring = 1024;

pub const Lanes = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    peer: ?link.Link, // rank 1 (null: one rank); every follower gets each frame (Model.followers)
    vision: ?*@import("vision.zig").Vision = null, // rank 0's tower (a vision checkpoint served with its kit's vision/)
    step: u64 = 0,
    logit_dump_dir: ?[]const u8 = null, // closed-window diagnostics only; no production I/O by default
    logit_dump_serial: u64 = 0,
    short_continuation: bool = false, // experimental full-decoder resume; disabled unless explicitly qualified
    wire: Wire = .{},
    streams: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    wc: [window_rows]lanes.config.Cost = undefined, // the window costs (init)
    sc: [max_rows]lanes.config.Cost = undefined, // the shared forwards' costs (init)
    used: [max_streams]bool = @splat(false),
    seqs: [max_streams]std.ArrayList(i32) = @splat(.empty),
    built: Built = .{},
    prof: ?Profile = null,
    adaptive_learning: bool = false, // active online learning, opt-in
    fix_k: ?usize = null, // a measurement (tf-dsv41-lanes --fix-k): K drafts a stream, chances 1 to K and 0 past it
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    snaps: std.ArrayList(*Snap) = .empty, // kept prompt states, oldest first
    next_snap: u64 = 1,
    sample_workspace: sampling.Workspace = .{},
    penalty_workers: ?*penalty_pool.Pool = null,
    penalty_disabled: bool = false,
    /// PETABRIDGE (k2a-pb1) spill P6: --spill-dir/--spill-gib: the spill tier (native.zig opens it; null: off, nothing
    /// here changes: no bump, no guard, no realignment, no frame).
    spill: ?*SpillTier = null,

    fn drawToken(self: *Lanes, row: []const f32, position: u64, s: ?lanes.Sampling) !u32 {
        const sm = s orelse return @intCast(sampling.argmax(row));
        const t = try sampling.sampleRowWithWorkspace(&self.sample_workspace, self.gpa, row, position, .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p });
        return @intCast(t);
    }

    pub fn init(gpa: std.mem.Allocator, m: *Model) Lanes {
        var ln: Lanes = .{ .gpa = gpa, .m = m, .peer = if (m.world > 1) m.peer() else null };
        // the round costs by rows: ROUND_MS[min(rows, len) - 1] (the served table, or TF_DS_ROUND_MS)
        const t = m.round_ms;
        for (&ln.wc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        for (&ln.sc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        return ln;
    }

    /// Tells rank 1 to stop, then frees the host state (the model is the caller's).
    pub fn deinit(self: *Lanes) void {
        for (self.snaps.items) |sn| {
            sn.buf.free();
            self.gpa.destroy(sn);
        }
        self.snaps.deinit(self.gpa);
        if (self.penalty_workers) |p| p.destroy(self.gpa);
        self.sample_workspace.deinit(self.gpa);
        if (self.peer != null) self.send(.done) catch {};
        self.streams.deinit(self.gpa);
        for (&self.seqs) |*q| q.deinit(self.gpa);
        self.wire.deinit(self.gpa);
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = &.{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .probabilities = probabilitiesFn,
            .release = releaseFn,
            .token_pool = tokenPoolFn,
        } };
    }

    /// The facts the round loop reads: windows of the pending row and up to the drafter's block, several streams'
    /// windows in one forward of at most 16 rows (four streams), the drafter's chances, the served lane's round ms.
    pub fn facts(self: *const Lanes) lanes.Model {
        const drafting = self.m.drafting();
        return .{
            .adaptive_learning = self.adaptive_learning,
            .exact_width = window_rows,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = if (self.m.dr) |dr| @intCast(dr.n) else 1,
            .window_costs = self.wc[0..],
            .mtp_step_ms = draft.draft_ms,
            .streams_exact = true,
            .hidden_rows = true,
            .batch_rows = @intCast(self.m.eng.round_rows),
            .max_streams = @intCast(self.m.streams),
            .shared_costs = self.sc[0..self.m.eng.round_rows],
            .draft_probabilities = drafting,
            .draft_streams = drafting,
        };
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn value(self: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
    }

    /// The frame being built goes to rank 1 (after `begin` and the fields), or to ranks 1-3 on the four-node split.
    fn flush(self: *Lanes) !void {
        if (self.peer == null) return;
        for (self.m.followers()) |l| try l.?.send(self.wire.buf.items);
    }

    fn begin(self: *Lanes, kind: Kind) !bool {
        if (self.peer == null) return false;
        try self.wire.begin(self.gpa, kind, self.step);
        self.step += 1;
        return true;
    }

    fn send(self: *Lanes, kind: Kind) !void {
        if (try self.begin(kind)) try self.flush();
    }

    /// Union of occupied intervals, without allocation or double-counting overlapping snapshots.
    fn retained(self: *const Lanes, active: []const [2]usize) usize {
        var cursor: usize = 0;
        var total: usize = 0;
        while (true) {
            var begin_: ?usize = null;
            for (active) |r| if (r[1] > r[0] and r[1] > cursor) {
                begin_ = @min(begin_ orelse r[0], @max(cursor, r[0]));
            };
            for (self.snaps.items) |s| if (!s.stale and s.at > 0 and s.base + s.at > cursor) {
                begin_ = @min(begin_ orelse s.base, @max(cursor, s.base));
            };
            const first = begin_ orelse break;
            var end = first;
            while (true) {
                const before = end;
                for (active) |r| if (r[0] <= end and r[1] > end) {
                    end = r[1];
                };
                for (self.snaps.items) |s| if (!s.stale and s.base <= end and s.base + s.at > end) {
                    end = s.base + s.at;
                };
                if (end == before) break;
            }
            if (end == first) break;
            total += end - first;
            cursor = end;
        }
        return total;
    }

    fn tokenPoolFn(ptr: *anyopaque) be.TokenPool {
        const self = of(ptr);
        var ranges: [max_streams][2]usize = undefined;
        var count: usize = 0;
        var active: usize = 0;
        var reserved: usize = 0;
        var it = self.streams.iterator();
        while (it.next()) |entry| {
            const lane = entry.value_ptr;
            const filled = @min(lane.end - lane.base, @max(lane.filled, entry.key_ptr.*.cache_len));
            ranges[count] = .{ lane.base, lane.base + filled };
            active += filled;
            reserved += lane.end - lane.base;
            count += 1;
        }
        const total = self.retained(ranges[0..count]);
        return .{ .capacity = self.m.pool_cap, .active = active, .cached = total -| active, .reserved = reserved };
    }

    /// GPU completion events are queried on the engine thread, never synchronized for telemetry.
    const Progress = struct {
        marks: std.ArrayList(struct { event: cuda.Event, at: u32 }) = .empty,
        consumed: usize = 0,
        fn deinit(p: *Progress, gpa: std.mem.Allocator) void {
            for (p.marks.items) |*mark| mark.event.deinit();
            p.marks.deinit(gpa);
        }
        fn record(p: *Progress, self: *Lanes, s: *lanes.Stream, at: usize) void {
            if (s.prefill_progress == null) return;
            var event = cuda.Event.init(self.m.stream.d, false) catch return;
            event.record(self.m.stream) catch {
                event.deinit();
                return;
            };
            p.marks.append(self.gpa, .{ .event = event, .at = @intCast(at) }) catch {
                event.deinit();
                return;
            };
            p.poll(self, s);
        }
        fn poll(p: *Progress, self: *Lanes, s: *lanes.Stream) void {
            var at: ?u32 = null;
            while (p.consumed < p.marks.items.len) {
                const mark = &p.marks.items[p.consumed];
                if (!(mark.event.done() catch false)) break;
                at = mark.at;
                p.consumed += 1;
            }
            if (at) |n| {
                self.streams.getPtr(s).?.filled = n;
                if (s.prefill_progress) |hook| hook.at(hook.ptr, s, n);
            }
        }
    };

    // -- the vtable ---------------------------------------------------------------------------------------------

    /// The lowest free slot, the first extent that fits, the prompt in chunks of the served size (at multiples of it:
    /// the chunks the gates checked; the host's chunk plan is not followed), the head's draw of the first token.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        const t0 = m.now();
        defer if (self.prof) |*p| {
            p.prefill += m.now() - t0;
            p.prefills += 1;
        };
        const ids = s.prompt();
        const len = ids.len;
        if (len == 0) return error.EmptyPrompt;
        const size = std.mem.alignForward(usize, len + s.max_new + m.eng.round_rows + 2, model.chunk_rows);
        if (size > m.pool_cap) return error.PromptTooLong;
        if (self.streams.fetchRemove(s)) |old| self.used[old.value.slot] = false;
        const slot = std.mem.indexOfScalar(bool, self.used[0..self.m.streams], false) orelse return error.NoFreeSlot;
        // PETABRIDGE (k2a-pb1) spill P6 (write path W1, slot reuse): the slot's own state is about to change
        if (self.spill) |t| t.bumpSlot(slot);
        var taken: [max_streams][2]usize = undefined;
        var nt: usize = 0;
        var it = self.streams.valueIterator();
        while (it.next()) |l| {
            taken[nt] = .{ l.base, l.end };
            nt += 1;
        }
        // a kept state of this prompt's prefix: the pass resumes at its chunk end, in its extent, or in one of its own
        // with the kept rows copied there when a live stream holds that extent (a burst's second request on a prompt)
        s.cached = 0;
        var kept: ?*Snap = null;
        var moved = false;
        if (s.reuse.saved) |p| {
            const sn: *Snap = @ptrCast(@alignCast(p));
            // (a prompt with images resumes a kept state too: the cache keys the image positions by the picture's content, so the
            // state's rows are this prompt's own; the pictures before the cut need no rows again, below)
            if (self.usable(sn, len)) {
                kept = sn;
                moved = sn.base + size > m.pool_cap or clashes(taken[0..nt], sn.base, sn.base + size) or
                    self.overwritesKept(sn.base + sn.at, sn.base + size, sn);
            } else s.reuse_failed = true;
        }
        const base = if (kept != null and !moved) kept.?.base else try self.placeKept(taken[0..nt], size, kept);
        if (kept) |sn| if (!moved) {
            sn.end = @max(sn.end, base + size);
        };
        if (kept) |sn| if (moved and sn.stale) { // the placement needed the kept rows' room: the prompt runs from 0
            kept = null;
            s.reuse_failed = true;
        };
        // the rows this stream writes are no other kept state's
        self.staleOver(if (kept != null and !moved) base + kept.?.at else base, base + size, kept);
        if (kept) |sn| {
            sn.resumes +|= 1;
            sn.used = retentionTick();
        }
        try self.streams.put(gpa, s, .{ .slot = slot, .base = base, .end = base + size, .from_snap = if (kept) |sn| sn.id else 0 });
        self.used[slot] = true;
        errdefer {
            _ = self.streams.remove(s);
            self.used[slot] = false;
        }
        const seq = &self.seqs[slot];
        try seq.resize(gpa, len);
        for (seq.items, ids) |*q, id| q.* = @intCast(id);
        // images: each span's positions dead in the sequence (negative: no Engram n-gram reaches into or across one),
        // its rows ready (a gate's) or the tower's (rank 0), all of them sent to rank 1 after the fill's start
        var spans: std.ArrayList(@import("prompt.zig").Span) = .empty;
        defer spans.deinit(gpa);
        var rows: std.ArrayList(u8) = .empty;
        defer rows.deinit(gpa);
        if (s.images.len > 0) {
            const image_id = m.cfg.image_token orelse return error.NoVision;
            const row_bytes = m.cfg.hidden * 2;
            var at_row: usize = 0;
            // a resumed pass starts at the kept state's end: a picture wholly before it is already in the state (its positions stay
            // dead in the sequence for Engram's n-grams, but it needs no rows and no run of the tower)
            const resume_at: usize = if (kept) |sn| sn.at else 0;
            for (s.images) |img| {
                if (img.tokens == 0 or @as(usize, img.at) + img.tokens > len) return error.BadImageSpan;
                for (seq.items[img.at..][0..img.tokens]) |*q| {
                    if (q.* != @as(i64, image_id)) return error.BadImageSpan;
                    q.* = -1;
                }
                if (@as(usize, img.at) + img.tokens <= resume_at) continue;
                if (img.rows) {
                    if (img.bytes.len != img.tokens * row_bytes) return error.BadImageRows;
                    try rows.appendSlice(gpa, img.bytes);
                } else {
                    // a prepared picture (vision.zig): the tower's span rows, here on rank 0
                    const v = self.vision orelse return error.NoVisionTower;
                    if (try v.rows(img.bytes, gpa, &rows) != img.tokens) return error.BadImageSpan;
                }
                try spans.append(gpa, .{ .at = img.at, .len = img.tokens, .row = at_row });
                at_row += img.tokens;
            }
        }
        if (try self.begin(.fill_begin)) {
            try self.wire.int(gpa, u32, @intCast(slot));
            try self.wire.int(gpa, u64, base);
            try self.wire.ids(gpa, seq.items);
            try self.flush();
        }
        try m.fillBegin(slot, base);
        if (spans.items.len > 0) {
            // the spans and their rows, image_piece bytes of rows a frame (a follower's frame buffer holds 64 KiB and
            // the ids of a full pool)
            var off: usize = 0;
            while (off < rows.items.len) {
                const n = @min(image_piece, rows.items.len - off);
                if (try self.begin(.fill_images)) {
                    try self.wire.int(gpa, u32, @intCast(spans.items.len));
                    for (spans.items) |sp| {
                        try self.wire.int(gpa, u64, sp.at);
                        try self.wire.int(gpa, u64, sp.len);
                    }
                    try self.wire.int(gpa, u64, off);
                    try self.wire.buf.appendSlice(gpa, rows.items[off..][0..n]);
                    try self.flush();
                }
                off += n;
            }
            try m.setImages(spans.items, rows.items);
        }
        // A fully decoded prefix has valid preceding window keys. For a short
        // append, preserve that decoder history instead of discarding it at the
        // new prompt's CED boundary. Encoder-only snapshots retain bounded replay.
        // This path is opt-in and changes continuation numerics; qualify separately.
        const replay = if (kept) |sn| (if (self.shortResume(sn, len)) sn.at -| m.cfg.window else len -| m.cfg.window) else len -| m.cfg.window;
        var start: usize = 0;
        if (kept) |sn| {
            if (try self.begin(.snap_restore)) {
                try self.wire.int(gpa, u64, sn.id);
                try self.wire.int(gpa, u32, @intCast(slot));
                try self.wire.int(gpa, u64, sn.base);
                try self.wire.int(gpa, u64, base);
                try self.wire.int(gpa, u64, sn.at);
                try self.flush();
            }
            if (base != sn.base) {
                // PETABRIDGE (k2a-pb1) spill P6 (write path W4, copyExtent): the destination rows change
                if (self.spill) |t| t.bump(base, base + sn.at);
                try m.copyExtent(sn.base, base, sn.at);
            }
            try m.snapCopy(slot, sn.buf.ptr, false);
            start = sn.at;
            s.cached = @intCast(sn.at);
        }
        // where this pass keeps states: the cache's marks and its last chunk end before the replay (the served lane's
        // kept prompt), each at the chunk end at or before it
        const last_end = std.mem.alignBackward(usize, @min(replay, len - 1), 2);
        // PETABRIDGE (k2a-pb1) spill P6: the grid realignment (plan 3.4), only with the tier on and for a text prompt:
        // the first chunk after an off-grid resume ends at the next 2,048 boundary (spillChunk), so every later chunk
        // end is on the grid, and the pass also keeps the grid point at or before last_end (spillGridKeep), which the
        // tier may write. k2a's own keeps (off-grid last_end, complete, continuation) stay as they are, memory only.
        const realign = self.spill != null and s.images.len == 0;
        const grid_keep: usize = if (realign) spillGridKeep(start, last_end, self.spill.?.min_tokens) else 0;
        var head = start == len;
        m.fill_seq = seq.items;
        defer {
            m.joinAhead();
            m.fill_seq = &.{};
        }
        self.streams.getPtr(s).?.filled = start;
        if (s.prefill_progress) |hook| hook.at(hook.ptr, s, @intCast(start));
        var progress: Progress = .{};
        defer progress.deinit(gpa);
        while (start < len) {
            if (start > 0 and s.isCancelled()) return error.Cancelled;
            // Keep bulk GEMMs large; split only once at the final encoder checkpoint.
            const n = chunkLen(start, len, last_end, realign);
            if (try self.begin(.fill_chunk)) {
                try self.wire.int(gpa, u64, start);
                try self.wire.int(gpa, u32, @intCast(n));
                try self.wire.int(gpa, u64, replay);
                try self.flush();
            }
            // PETABRIDGE (k2a-pb1) spill P6 (write path W2, fillChunk): the chunk's rows change (images included: their
            // span rows are embedded into these same rows)
            if (self.spill) |t| {
                const span = bumpSpan(base + start, base + start + n); // (P6 review: ratio-2 groups at both ends)
                t.bump(span[0], span[1]);
            }
            head = try m.fillChunk(seq.items[0 .. start + n], start, n, replay);
            start += n;
            progress.record(self, s, start);
            if (s.reuse.hook) |k| if (start < len and start <= replay and (start == last_end or markEnds(s.reuse.marks, start) or start == grid_keep)) k.at(k.ptr, s, @intCast(start));
            // prefill interleave: the other streams' rounds between this pass's chunks (verify frames go to rank 1 in
            // order; every state a pass carries from chunk to chunk is the slot's, except the Engram read-ahead, which
            // a round's own Engram reads would race: joined first, its rows kept for the next chunk)
            if (start < len and s.images.len == 0) if (s.between) |b| if (b.wanted(b.ptr)) {
                m.joinAhead();
                try b.run(b.ptr);
            };
        }
        if (!head) return error.NoPromptLogits;
        // PETABRIDGE (k2a-pb1) spill P6: a state read back from disk: the drafter's rings were rebuilt by the replay
        // before the first draft (0016; P4's spillCheckResumed)
        if (self.spill != null and m.drafting()) if (kept) |sn| if (sn.restored) try m.spillCheckResumed(slot, len);
        // Full identical prompts reuse the complete state and logits. A later longer
        // prompt can use its encoder state only when bounded replay starts after it.
        if (s.images.len == 0) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(len));
        const logits = try m.promptLogitsHost();
        progress.poll(self, s);
        self.streams.getPtr(s).?.filled = len;
        if (logits_sha) {
            var dg: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(logits), &dg, .{});
            std.debug.print("{{\"prompt_logits\": \"{s}\", \"sha256\": \"{s}\"}}\n", .{ s.id, std.fmt.bytesToHex(dg, .lower) });
        }
        if (self.logit_dump_dir) |dir| {
            self.logit_dump_serial += 1;
            var name: [4096]u8 = undefined;
            const path = try std.fmt.bufPrintSentinel(&name, "{s}/prompt-{d}.f32", .{ dir, self.logit_dump_serial }, 0);
            const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
            if (fd < 0) return error.LogitDiagnosticOpen;
            defer _ = std.c.close(fd);
            const bytes = std.mem.sliceAsBytes(logits);
            var written: usize = 0;
            while (written < bytes.len) {
                const n = std.c.write(fd, bytes.ptr + written, bytes.len - written);
                if (n <= 0) return error.LogitDiagnosticWrite;
                written += @intCast(n);
            }
            std.debug.print("{{\"cache_logit_dump\":{d},\"prompt_tokens\":{d},\"cached_tokens\":{d}}}\n", .{ self.logit_dump_serial, len, s.cached });
        }
        const token = try self.drawToken(logits, len, s.sampling);
        self.streams.getPtr(s).?.first = self.take(token);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        const l = self.streams.getPtr(s) orelse return error.UnknownStream;
        if (position != s.prompt_len) return error.PositionMismatch;
        return l.first;
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = .{ ptr, s, feed, position };
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// Every window as one round at its stream's cache length: the pending token, the held drafts (or the host's), each
    /// row drawn at its keyed position.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        if (windows.len > max_streams) return error.WindowTooWide;
        var ids: [max_streams][max_rows]i32 = undefined;
        var wins: [max_streams]Win = undefined;
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = self.streams.getPtr(w.stream) orelse return error.UnknownStream;
            const rows = w.rows();
            if (rows > window_rows) return error.WindowTooWide;
            const start: usize = @intCast(w.stream.cache_len);
            for (w.positions, 0..) |p, r| if (p != start + 1 + r) return error.PositionMismatch;
            if (w.held > l.nheld) return error.HeldMismatch;
            ids[k][0] = @intCast(w.pending);
            for (l.held[0..w.held], 0..) |t, j| ids[k][1 + j] = @intCast(t);
            for (w.tokens, 0..) |t, j| ids[k][1 + w.held + j] = @intCast(t);
            wins[k] = .{ .slot = l.slot, .base = l.base, .end = l.end, .start = start, .absorb = m.drafting() and w.stream.drafts, .ids = ids[k][0..rows] };
        }
        const t0 = m.now();
        if (try self.begin(.verify)) {
            try self.wire.int(gpa, u32, @intCast(windows.len));
            for (wins[0..windows.len]) |w| {
                try self.wire.int(gpa, u32, @intCast(w.slot));
                try self.wire.int(gpa, u64, w.base);
                try self.wire.int(gpa, u64, w.end);
                try self.wire.int(gpa, u64, w.start);
                try self.wire.int(gpa, u8, @intFromBool(w.absorb));
                try self.wire.ids(gpa, w.ids);
            }
            try self.flush();
        }
        const t1 = m.now();
        // PETABRIDGE (k2a-pb1) spill P6 (write path W3, verify/draft rows: each window's rows, the pending row and the
        // drafts past cache_len included; also every round the 0005 interleave runs between a pass's chunks)
        // (widened by bumpSpan; plus the extent's last two positions, w.end being the extent's absolute end: an
        // incomplete ratio-2 group's row is written to the extent's last row as scratch, round.zig's ctarget2)
        if (self.spill) |t| for (wins[0..windows.len]) |w| {
            const span = bumpSpan(w.base + w.start, w.base + w.start + w.ids.len);
            t.bump(span[0], span[1]);
            t.bump(w.end -| 2, w.end);
        };
        const rows = try self.built.build(gpa, &self.seqs, wins[0..windows.len]);
        // Bounded two-row lookahead: all stream anchors before second rows.
        // Full-vocabulary unconstrained draws only; other modes use baseline.
        var requests: [penalty_pool.max_jobs]penalty_pool.Request = undefined;
        var count: usize = 0;
        for (0..2) |depth| for (windows) |w| {
            if (depth >= w.rows()) continue;
            if (w.stream.sampling) |sm| {
                if (sm.temperature <= 0 or !std.math.isFinite(sm.temperature) or
                    (sm.top_k != 0 and sm.top_k < m.vocab) or (sm.top_p > 0 and sm.top_p < 1)) continue;
                if (count == requests.len) break;
                requests[count] = .{ .seed = sm.seed, .position = w.positions[depth] };
                count += 1;
            }
        };
        if (count > 0 and self.penalty_workers == null and !self.penalty_disabled) {
            self.penalty_workers = penalty_pool.Pool.create(gpa, m.io, m.vocab) catch null;
            if (self.penalty_workers == null) self.penalty_disabled = true;
        }
        var active_pool: ?*penalty_pool.Pool = null;
        if (count > 0) if (self.penalty_workers) |p| {
            if (p.start(requests[0..count])) |_| {
                active_pool = p;
                self.sample_workspace.penalty_cache = p.cache();
            } else |_| {
                self.penalty_disabled = true; // preserve ordinary sampling if cache cannot start
            }
        };
        defer {
            self.sample_workspace.penalty_cache = null;
            if (active_pool) |p| p.finish(); // also on GPU/transport/sampling failure
        }
        try m.verify(rows, self.built.absorb[0..windows.len]);
        const t2 = m.now();
        // every row greedy: the device's argmax (R tokens back); else the logits for the host's draws
        var greedy = true;
        for (windows) |w| {
            if (w.stream.sampling) |sm| if (sm.temperature > 0) {
                greedy = false;
            };
        }
        const tokens: []const u32 = if (greedy) try m.roundArgmax(rows.ids.len) else &.{};
        const logits: []const f32 = if (greedy) &.{} else try m.roundLogits(rows.ids.len);
        const t3 = m.now();
        defer if (self.prof) |*p| {
            p.send += t1 - t0;
            p.enqueue += m.t_enqueue;
            p.forward += m.t_forward;
            p.absorb += m.t_absorb;
            p.logits += t3 - t2;
            p.sample += m.now() - t3;
            p.rounds += 1;
            p.rows += rows.ids.len;
        };
        const V = m.vocab;
        for (windows, out, 0..) |w, *o, k| {
            const row0 = self.built.wins[k].row;
            const l = self.streams.getPtr(w.stream).?;
            @memcpy(o.drafts[0..w.held], l.held[0..w.held]);
            @memcpy(o.drafts[w.held..][0..w.tokens.len], w.tokens);
            if (greedy) {
                @memcpy(o.sampled[0..w.rows()], tokens[row0..][0..w.rows()]);
            } else {
                const sm: ?sampling.Sampling = if (w.stream.sampling) |s| .{ .seed = s.seed, .temperature = s.temperature, .top_k = s.top_k, .top_p = s.top_p, .min_p = s.min_p } else null;
                _ = try sampling.sampleWindow(&self.sample_workspace, gpa, logits[row0 * V ..][0 .. w.rows() * V], V, w.positions, o.drafts[0 .. w.rows() - 1], sm, w.prefix_only, o.sampled[0..w.rows()]);
            }
            l.nheld = 0;
        }
    }

    /// The caches are positional: the next window starts at the stream's new length and writes over the rows past it.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        _ = ptr;
        for (windows, paths) |_, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
        }
    }

    /// One drafter pass for every stream asked: its pending token at its new length, `depth` drafts held.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        if (m.dr == null) return error.NoDraftHead;
        const block = m.dr.?.n;
        var tokens: [max_streams]i64 = undefined;
        var q0: [max_streams]i64 = undefined;
        var slots: [max_streams]i64 = undefined;
        var depths: [max_streams]usize = undefined;
        var asked: [max_streams]*Lane = undefined;
        var n: usize = 0;
        var steps: usize = 0;
        for (requests) |r| {
            if (r.lanes != null or r.early) return error.TreesNotBuilt;
            const l = self.streams.getPtr(r.stream) orelse return error.UnknownStream;
            l.nheld = 0;
            if (r.depth == 0) continue;
            if (n == max_streams) return error.TooManyStreams;
            if (r.position != r.stream.cache_len + 1) return error.PositionMismatch;
            const token: u32 = if (r.rows == null) self.value(r.first orelse return error.NoFirstToken) else (if (r.follow.len > 0) r.follow[r.follow.len - 1] else return error.NoFollowToken);
            tokens[n] = token;
            q0[n] = @intCast(r.position - 1);
            slots[n] = @intCast(l.slot);
            depths[n] = @min(r.depth, block);
            asked[n] = l;
            steps = @max(steps, depths[n]);
            n += 1;
        }
        if (n == 0) return;
        if (try self.begin(.pass)) {
            try self.wire.int(gpa, u32, @intCast(n));
            try self.wire.int(gpa, u32, @intCast(steps));
            for (0..n) |i| {
                try self.wire.int(gpa, i64, tokens[i]);
                try self.wire.int(gpa, i64, q0[i]);
                try self.wire.int(gpa, i64, slots[i]);
            }
            try self.flush();
        }
        const t0 = m.now();
        try m.pass(tokens[0..n], q0[0..n], slots[0..n], steps);
        if (self.prof) |*p| {
            p.pass += m.now() - t0;
            p.passes += 1;
            p.pass_streams += n;
        }
        const d = &m.dr.?;
        for (asked[0..n], depths[0..n], 0..) |l, k, i| {
            for (0..k) |j| {
                l.held[j] = @intCast(d.drafts[i][j]);
                l.confs[j] = d.confs[i][j];
            }
            l.nheld = k;
        }
    }

    /// The chance that each held draft lands with every earlier one, as the round loop's allocation reads it: the
    /// running product of the confidences' sigmoids (multi.py's conf rule: a sigmoid is the draft's survival once
    /// the earlier drafts survived, and the prefix survivals' sum is the expected kept drafts; draft.chooseK).
    fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const self = of(ptr);
        const l = self.streams.getPtr(s) orelse return false;
        if (out.len > l.nheld) return false;
        if (self.fix_k) |k| {
            for (out, 0..) |*o, j| o.* = if (j < k) 1.0 else 0.0;
            return true;
        }
        survivals(l.confs[0..out.len], out);
        return true;
    }

    /// Whether a kept state still restores for a prompt of `len` tokens: live, not stale, a chunk end before the
    /// prompt's replay (the rows before it ran the encoder layers alone, as a fresh pass of this prompt runs them).
    fn usable(self: *const Lanes, sn: *const Snap, len: usize) bool {
        if (std.mem.indexOfScalar(*Snap, self.snaps.items, @constCast(sn)) == null) return false;
        if (sn.stale or sn.at == 0) return false;
        if (sn.loading) return false; // PETABRIDGE (k2a-pb1) spill P6: a restore's destination, not yet installed
        if (sn.complete and sn.at == len) return true;
        if (self.shortResume(sn, len)) return true;
        return sn.at < len and sn.at % 2 == 0 and sn.at <= len -| self.m.cfg.window;
    }

    fn shortResume(self: *const Lanes, sn: *const Snap, len: usize) bool {
        return self.short_continuation and (sn.complete or sn.decoded) and
            sn.at > 0 and sn.at % 2 == 0 and sn.at < len and len - sn.at <= self.m.cfg.window;
    }

    pub fn snapLiveFn(ptr: *anyopaque, saved: *anyopaque) bool {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        return std.mem.indexOfScalar(*Snap, self.snaps.items, sn) != null and !sn.stale and !sn.loading; // PETABRIDGE: !loading (P6)
    }

    pub fn snapCompatibleFn(ptr: *anyopaque, saved: *anyopaque, len: usize) bool {
        return of(ptr).usable(@ptrCast(@alignCast(saved)), len);
    }

    /// The first extent that fits around the live streams and the kept states; the oldest kept states go stale
    /// until it does.
    fn placeKept(self: *Lanes, live: []const [2]usize, size: usize, protected: ?*Snap) !usize {
        const gpa = self.gpa;
        var kept: std.ArrayList([3]usize) = .empty;
        defer kept.deinit(gpa);
        while (true) {
            kept.clearRetainingCapacity();
            for (self.snaps.items) |sn| if (!sn.stale) try kept.append(gpa, .{ sn.base, sn.at, sn.end });
            if (try placeSoft(gpa, live, kept.items, size, self.m.pool_cap)) |b| return b;
            const oldest = victim(self.snaps.items, live, protected) orelse if (protected) |sn| blk: {
                // True memory pressure only: give up the source last, then the caller
                // explicitly falls back to a cold prefill. Never evict it just by age.
                if (!sn.stale and !clashes(live, sn.base, sn.base + sn.at)) break :blk sn;
                return error.ContextFull;
            } else return error.ContextFull;
            // PETABRIDGE (k2a-pb1) spill P6: its unpublished write stages or cancels before the rows go
            if (self.spill) |t| t.guardSnap(oldest.id, 0);
            oldest.stale = true;
        }
    }

    /// The kept state placement gives up next, `protected` (a pass's restore source) aside and never one whose rows a
    /// live stream still owns: by retention order (retention()), else the oldest, as upstream.
    fn victim(snaps: []const *Snap, live: []const [2]usize, protected: ?*Snap) ?*Snap {
        var best: ?*Snap = null;
        for (snaps) |sn| {
            // PETABRIDGE (k2a-pb1) spill P6: never a restore's destination (loading: its rows are being read in)
            if (sn.stale or sn.loading or sn == protected or clashes(live, sn.base, sn.base + sn.at)) continue;
            if (!retention()) return sn;
            if (best == null or retentionBefore(sn, best.?)) best = sn;
        }
        return best;
    }

    /// Would the writable tail destroy another retained prefix? In-place growth
    /// must obey the same reservations as placement of a fresh extent.
    fn overwritesKept(self: *const Lanes, lo: usize, hi: usize, except: ?*Snap) bool {
        for (self.snaps.items) |sn| {
            if (!sn.stale and sn != except and sn.base < hi and lo < sn.base + sn.at) return true;
        }
        return false;
    }

    /// Kept states with rows in [lo, hi) go stale (a stream writes there), `except` aside.
    fn staleOver(self: *Lanes, lo: usize, hi: usize, except: ?*Snap) void {
        for (self.snaps.items) |sn| if (sn != except and sn.base < hi and lo < sn.base + sn.at) {
            // PETABRIDGE (k2a-pb1) spill P6 (0016): its unpublished write stages what it still reads before `lo`, or is
            // cancelled, before the stream writes there
            if (!sn.stale) if (self.spill) |t| t.guardSnap(sn.id, @intCast(@max(lo, sn.base) - sn.base));
            sn.stale = true;
        };
    }

    /// The prompt cache's Snapshots functions (native.zig hands them to the CUDA host).
    pub fn snapBytesFn(ptr: *anyopaque, at: u32) u64 {
        _ = at;
        return of(ptr).m.snapBytes();
    }

    /// The live state at `at` (the pass stands at that chunk end): the slot's own part copied, the extent's rows kept.
    pub fn snapSaveFn(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = of(ptr);
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const l = self.streams.getPtr(s) orelse return error.UnknownStream;
        const complete = at == s.prompt_len and s.emitted().len == 0;
        const continuation = s.finished and (s.reason == .stop or s.reason == .length) and at <= s.cache_len;
        if (at == 0 or (!complete and (at % 2 != 0 or (!continuation and at > s.prompt_len -| self.m.cfg.window)))) return error.NotAChunkEnd;
        const sn = try self.gpa.create(Snap);
        errdefer self.gpa.destroy(sn);
        sn.* = .{ .id = self.next_snap, .at = at, .base = l.base, .end = l.end, .buf = try cuda.DeviceBuffer.alloc(self.m.ctx.d, self.m.snapBytes()), .complete = complete, .decoded = complete or continuation, .slot = @intCast(l.slot) }; // PETABRIDGE: .slot (P6)
        errdefer sn.buf.free();
        try self.snaps.append(self.gpa, sn);
        self.next_snap += 1;
        // retention: the conversation's count carries forward; the state this one grew from (this pass's last keep,
        // else the state the pass resumed from) is superseded when this one reaches further
        sn.used = retentionTick();
        const prior = if (l.last_snap != 0) l.last_snap else l.from_snap;
        for (self.snaps.items) |o| {
            if (o == sn) continue;
            if (o.id == l.from_snap) sn.resumes = @max(sn.resumes, o.resumes);
            if (o.id == prior and o.at < at) o.superseded = true;
        }
        l.last_snap = sn.id;
        if (try self.begin(.snap_save)) {
            try self.wire.int(self.gpa, u64, sn.id);
            try self.wire.int(self.gpa, u32, @intCast(l.slot));
            try self.flush();
        }
        try self.m.snapCopy(l.slot, sn.buf.ptr, true);
        return sn;
    }

    pub fn snapRestoreFn(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }

    pub fn snapDropFn(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        const i = std.mem.indexOfScalar(*Snap, self.snaps.items, sn) orelse return;
        _ = self.snaps.orderedRemove(i);
        // PETABRIDGE (k2a-pb1) spill P6 (0016, rule b): a pending small-state capture of its buffer done first, its
        // unpublished write guarded from 0, before the buffer goes (the followers do the same at their snap_drop)
        if (self.spill) |t| t.dropSnap(sn.id);
        if (self.begin(.snap_drop) catch false) {
            self.wire.int(self.gpa, u64, sn.id) catch {};
            self.flush() catch |err| std.log.err("dsv41: rank 1 kept a dropped prompt state: {s}", .{@errorName(err)});
        }
        sn.buf.free();
        self.gpa.destroy(sn);
    }

    // -- PETABRIDGE (k2a-pb1) spill P6: the spill tier's family and wire (spill_tier.Family, spill_tier.Wire) ---------
    // MiaAI-Lab 0016's, fitted: restores' destinations enter retention's probation (resumes 0, used now), never a
    // victim while loading; a Snap buffer's tail past the small state (our drafter rings and logits) is zeroed before
    // the restore lands (P4: the replay rebuilds both); ineligible states (complete, decoded, off-grid) are not written.

    pub fn spillFamily(self: *Lanes) spill_tier.Family {
        return .{ .ptr = self, .reserve = spillReserve, .install = spillInstall, .discard = spillDiscard, .info = spillInfo };
    }

    pub fn spillWire(self: *Lanes) spill_tier.Wire {
        return .{ .ctx = self, .frame = spillFrame, .links = self.m.followers() };
    }

    /// A restore's destination: an extent placed like a stream's (kept states give their room by retention order,
    /// guarded), a fresh Snap holding it back (loading: not usable until every rank installed it) and its buffer.
    /// `room`: the waiting prompt's length; when an extent of it and a growth margin fits without taking any kept
    /// state's room, the Snap's extent (`end`) holds that much, so the resume runs in place (no copyExtent).
    fn spillReserve(ptr: *anyopaque, at: usize, room: usize) anyerror!spill_tier.Dest {
        const self = of(ptr);
        var taken: [max_streams][2]usize = undefined;
        var nt: usize = 0;
        var it = self.streams.valueIterator();
        while (it.next()) |l| {
            taken[nt] = .{ l.base, l.end };
            nt += 1;
        }
        const want = @min(std.mem.alignForward(usize, @max(room, at) + growth_rows, model.chunk_rows), self.m.pool_cap);
        const roomy = if (want > at) try self.placeFree(taken[0..nt], want) else null;
        const base = roomy orelse try self.placeKept(taken[0..nt], at, null);
        self.staleOver(base, base + at, null);
        const sn = try self.gpa.create(Snap);
        errdefer self.gpa.destroy(sn);
        // (probation: never resumed, used now; retention gives it a conversation's count only once resumed)
        sn.* = .{ .id = self.next_snap, .at = at, .base = base, .end = base + (if (roomy != null) want else at), .buf = try cuda.DeviceBuffer.alloc(self.m.ctx.d, self.m.snapBytes()), .loading = true, .used = retentionTick() };
        errdefer sn.buf.free();
        if (self.spill) |t| try self.m.spillClearSnapTail(&t.side.layout, sn.buf.ptr);
        try self.snaps.append(self.gpa, sn);
        self.next_snap += 1;
        return .{ .saved = sn, .id = sn.id, .base = base, .buf = sn.buf.ptr };
    }

    /// An extent of `size` around the live streams and the kept states' rows (placeSoft's rules: their held extents
    /// first), staling nobody: null when none.
    fn placeFree(self: *Lanes, live: []const [2]usize, size: usize) !?usize {
        var kept: std.ArrayList([3]usize) = .empty;
        defer kept.deinit(self.gpa);
        for (self.snaps.items) |sn| if (!sn.stale) try kept.append(self.gpa, .{ sn.base, sn.at, sn.end });
        return placeSoft(self.gpa, live, kept.items, size, self.m.pool_cap);
    }

    fn spillInstall(_: *anyopaque, saved: *anyopaque) void {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        sn.loading = false;
        sn.restored = true;
        // retention: a restored state enters probation (never resumed) as recently used, never inheriting a count
        sn.resumes = 0;
        sn.superseded = false;
        sn.used = retentionTick();
    }

    /// A restore that did not commit: its Snap goes (the followers drop theirs with the abort, no snap_drop).
    fn spillDiscard(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        if (std.mem.indexOfScalar(*Snap, self.snaps.items, sn)) |i| _ = self.snaps.orderedRemove(i);
        sn.buf.free();
        self.gpa.destroy(sn);
    }

    fn spillInfo(_: *anyopaque, saved: *anyopaque) spill_tier.SnapInfo {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        return .{ .id = sn.id, .base = sn.base, .at = sn.at, .buf = sn.buf.ptr, .slot = sn.slot, .eligible = spillEligible(sn) };
    }

    /// A spill frame (Kind rank_wire) to every follower, in the ordered stream.
    pub fn spillFrame(ctx: *anyopaque, kind: u8, sub: []const u8) anyerror!void {
        const self = of(ctx);
        const k: Kind = switch (kind) {
            10 => .rank_wire,
            else => return error.BadFrame,
        };
        if (try self.begin(k)) {
            try self.wire.buf.appendSlice(self.gpa, sub);
            try self.flush();
        }
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        if (!self.streams.contains(s)) return;
        // Only accepted target rows. Speculative rows beyond cache_len are excluded
        // from the key and compressed extent; the ring's extra rows protect the tail.
        if (s.images.len == 0 and s.finished and (s.reason == .stop or s.reason == .length)) {
            const at = std.mem.alignBackward(usize, @intCast(s.cache_len), 2);
            if (at > s.prompt_len and at <= s.context.items.len) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(at));
        }
        const kv = self.streams.fetchRemove(s).?;
        self.used[kv.value.slot] = false;
    }
};

// -- PETABRIDGE (k2a-pb1) spill P6: the grid realignment and the disk eligibility rule (pure; tested below) ----------

/// The spill tier's grid (spill_state's block, the store's `Rules.grid`): only points on it go to disk.
pub const spill_grid = model.chunk_rows;

/// A prompt pass's next chunk from `start` (prefillFn's loop): k2a's rule (at most chunk_rows, the bulk cut once at
/// the final encoder checkpoint `last_end`), and with `realign` (the tier on, a text prompt) spillChunk's cut too.
/// `realign` false: exactly k2a's arithmetic.
pub fn chunkLen(start: usize, len: usize, last_end: usize, realign: bool) usize {
    // Keep bulk GEMMs large; split only once at the final encoder checkpoint.
    const end = if (start < last_end) @min(len, last_end) else len;
    const n = @min(model.chunk_rows, end - start);
    return if (realign) spillChunk(start, n, last_end) else n;
}

/// A pass's chunk with the tier on (plan 3.4): `n` (k2a's: at most chunk_rows, cut at last_end) cut once more at the
/// next grid boundary while `start` is off the grid and before `last_end`, so that after an off-grid resume every
/// later chunk end is on the grid until the cut at last_end. On the grid, or at or past last_end: `n` unchanged.
pub fn spillChunk(start: usize, n: usize, last_end: usize) usize {
    if (start % spill_grid == 0 or start >= last_end) return n;
    return @min(n, std.mem.alignForward(usize, start, spill_grid) - start);
}

/// The grid point a pass starting at `start` also keeps with the tier on: the grid point at or before `last_end`,
/// when it lies past the resume point, below last_end itself (k2a keeps that one already) and long enough for the
/// tier (`min_tokens`); 0: none. After spillChunk's realignment it is always one of the pass's chunk ends.
pub fn spillGridKeep(start: usize, last_end: usize, min_tokens: u32) usize {
    const g = std.mem.alignBackward(usize, last_end, spill_grid);
    if (g == 0 or g <= start or g == last_end or g < min_tokens) return 0;
    return g;
}

/// Whether a kept state may go to disk: a plain pre-replay prompt state on the grid (the only kind a restore resumes
/// with a bounded replay that rebuilds the drafter rings and the logits: P4), never k2a's complete, decoded or
/// continuation states, a restore's destination still loading, or a stale one.
pub fn spillEligible(sn: *const Snap) bool {
    return sn.at > 0 and sn.at % spill_grid == 0 and !sn.complete and !sn.decoded and !sn.loading and !sn.stale;
}

/// Prefix survivals: out[j] = the product of sigmoid(conf[i]) for i <= j, in float64 (libm's exp, as chooseK).
pub fn survivals(conf: []const f32, out: []f64) void {
    var surv: f64 = 1.0;
    for (out, conf[0..out.len]) |*o, cf| {
        surv *= 1.0 / (1.0 + exp(-@as(f64, cf)));
        o.* = surv;
    }
}

/// A primitive this rank refused: rank 0 runs the same one on the same inputs and refuses it at the same step (its
/// request ends with the error), so the follower reports it and takes the next frame instead of ending.
fn refused(err: anyerror) void {
    std.debug.print("{{\"follow_refused\": \"{s}\"}}\n", .{@errorName(err)});
}

/// A follower's verify times with Model.prof (a profile on ranks 1-3: rounds, issue and wall ns summed).
pub var follow_prof: struct { rounds: u64 = 0, enqueue: u64 = 0, forward: u64 = 0 } = .{};

/// Rank 1 (each of ranks 1-3 on the four-node split): the primitives rank 0 sends, run in order until it says done.
pub fn follow(gpa: std.mem.Allocator, m: *Model) !void {
    return followWith(gpa, m, null);
}

/// PETABRIDGE (k2a-pb1) spill P6: a follower's kept states as its spill side sees them (spill_tier.FollowerFamily);
/// MiaAI-Lab 0016's, plus our Snap tail zeroed in a restore's fresh buffer (P4: the replay rebuilds it).
const FollowKept = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    kept: *std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer),
    loading: std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer) = .empty, // restores' destinations, not yet installed
    seqs: *[max_streams]std.ArrayList(i32),
    layout: ?model.Model.spill.StateLayout = null,

    fn family(f: *FollowKept) spill_tier.FollowerFamily {
        return .{ .ptr = f, .reserve = reserve, .install = install, .discard = discard, .buf = buf, .ids = ids };
    }

    fn of(p: *anyopaque) *FollowKept {
        return @ptrCast(@alignCast(p));
    }

    fn reserve(p: *anyopaque, id: u64) anyerror!u64 {
        const f = of(p);
        if (f.kept.contains(id) or f.loading.contains(id)) return error.SnapshotExists;
        try f.loading.ensureUnusedCapacity(f.gpa, 1);
        var b = try cuda.DeviceBuffer.alloc(f.m.ctx.d, f.m.snapBytes());
        if (f.layout) |*l| f.m.spillClearSnapTail(l, b.ptr) catch |e| {
            b.free();
            return e;
        };
        f.loading.putAssumeCapacity(id, b);
        return b.ptr;
    }

    fn install(p: *anyopaque, id: u64) anyerror!void {
        const f = of(p);
        const kv = f.loading.fetchRemove(id) orelse return error.UnknownSnapshot;
        f.kept.put(f.gpa, id, kv.value) catch |e| {
            var b = kv.value;
            b.free();
            return e;
        };
    }

    fn discard(p: *anyopaque, id: u64) void {
        const f = of(p);
        var kv = f.loading.fetchRemove(id) orelse return;
        kv.value.free();
    }

    fn buf(p: *anyopaque, id: u64) ?u64 {
        return if (of(p).kept.get(id)) |b| b.ptr else null;
    }

    fn ids(p: *anyopaque, slot: u32) []const i32 {
        const f = of(p);
        return if (slot < max_streams) f.seqs[slot].items else &.{};
    }

    fn deinit(f: *FollowKept) void {
        var it = f.loading.valueIterator();
        while (it.next()) |b| b.free();
        f.loading.deinit(f.gpa);
    }
};

/// follow, with this rank's spill side when the tier is on (`spill`: made before the model's pool, its store opened at
/// the startup agreement rank 0 starts; the caller destroys it after this returns). PETABRIDGE (k2a-pb1) spill P6:
/// MiaAI-Lab 0016's followWith, with 0019's v2 (no Kind 11). Null: follow exactly as before (no new frame is valid).
pub fn followWith(gpa: std.mem.Allocator, m: *Model, spill: ?*SpillFollower) !void {
    const peer = m.peer();
    const buf = try gpa.alloc(u8, (m.pool_cap + 1) * 4 + (1 << 16));
    defer gpa.free(buf);
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    var win_ids: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&win_ids) |*q| q.deinit(gpa);
    var built: Built = .{};
    var step: u64 = 0;
    var slot: usize = 0;
    var img_rows: std.ArrayList(u8) = .empty; // a prompt's image rows as their pieces come
    defer img_rows.deinit(gpa);
    var kept: std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer) = .empty; // rank 0's kept states, this rank's part
    defer {
        var ki = kept.valueIterator();
        while (ki.next()) |b| b.free();
        kept.deinit(gpa);
    }
    // PETABRIDGE (k2a-pb1) spill P6 (0016): the spill side's view of the kept states, and its cleanup order
    var fk: FollowKept = .{ .gpa = gpa, .m = m, .kept = &kept, .seqs = &seqs };
    defer fk.deinit();
    if (spill) |sd| {
        fk.layout = try m.spillLayout();
        sd.attach(fk.family());
    }
    // on every exit, while fk and kept are alive: open restores ended (their buffers discarded through fk), the
    // family let go, then every copy still on the device waited for (no copy may read a buffer the defers above free)
    defer if (spill) |sd| {
        sd.detach();
        sd.quiesce();
    };
    var fill_base: usize = 0; // the filling prompt's extent (fill_begin's): its chunks' pool rows
    while (true) {
        // PETABRIDGE (k2a-pb1) spill P6: with spill writes or restores pending, they progress while rank 0 sends
        // nothing (2 ms polls)
        if (spill) |sd| while (sd.busy() and !try peer.readable(2 * std.time.ns_per_ms)) sd.service(std.time.ns_per_ms);
        // the next primitive, polled for 20 ms before the read blocks (rank 0 sends it a fraction of a millisecond
        // after this rank's GPU goes idle in a decode; a sleeping thread would add its core's wake-up to every frame)
        var r: Reader = .{ .b = try peer.recvSpin(buf, 20 * std.time.ns_per_ms) };
        const kind: Kind = switch (try r.int(u8)) {
            1 => .fill_begin,
            2 => .fill_chunk,
            3 => .verify,
            4 => .pass,
            5 => .snap_save,
            6 => .snap_restore,
            7 => .snap_drop,
            8 => .fill_images,
            9 => .done,
            10 => .rank_wire, // PETABRIDGE (k2a-pb1) spill P6: only with the tier on (else SpillNotStarted below)
            else => return error.BadFrame,
        };
        if (try r.int(u64) != step) return error.OutOfStep;
        step += 1;
        if (spill) |sd| sd.service(std.time.ns_per_ms); // (rank 0 pumps between its primitives alike)
        // a chunk's read-ahead of the next is done before anything but that chunk (a prompt's ids may change)
        if (kind != .fill_chunk and kind != .snap_save and kind != .snap_restore and kind != .fill_images and kind != .rank_wire) {
            m.joinAhead();
            m.fill_seq = &.{};
        }
        switch (kind) {
            .fill_begin => {
                slot = try r.int(u32);
                const base = try r.int(u64);
                if (slot >= max_streams) return error.BadSlot;
                img_rows.clearRetainingCapacity();
                try r.ids(gpa, &seqs[slot]);
                fill_base = @intCast(base);
                if (spill) |sd| sd.bumpSlot(slot); // PETABRIDGE (k2a-pb1) spill P6 (follower W1: slot reuse)
                m.fillBegin(slot, @intCast(base)) catch |err| refused(err);
                m.fill_seq = seqs[slot].items;
            },
            .fill_chunk => {
                const start: usize = @intCast(try r.int(u64));
                const n: usize = try r.int(u32);
                const replay: usize = @intCast(try r.int(u64));
                if (start + n > seqs[slot].items.len) return error.BadChunk;
                m.fill_seq = seqs[slot].items; // (a round between two chunks cleared it: the next chunk's read-ahead again)
                if (spill) |sd| { // PETABRIDGE spill P6 (follower W2; review: ratio-2 groups at both ends)
                    const span = bumpSpan(fill_base + start, fill_base + start + n);
                    sd.bump(span[0], span[1]);
                }
                _ = m.fillChunk(seqs[slot].items[0 .. start + n], start, n, replay) catch |err| refused(err);
            },
            .verify => {
                const nw = try r.int(u32);
                if (nw == 0 or nw > max_streams) return error.BadFrame;
                var wins: [max_streams]Win = undefined;
                for (0..nw) |k| {
                    const s: usize = try r.int(u32);
                    const base: usize = @intCast(try r.int(u64));
                    const end: usize = @intCast(try r.int(u64));
                    const start: usize = @intCast(try r.int(u64));
                    const absorb = try r.int(u8) != 0;
                    try r.ids(gpa, &win_ids[k]);
                    wins[k] = .{ .slot = s, .base = base, .end = end, .start = start, .absorb = absorb, .ids = win_ids[k].items };
                }
                const rows = try built.build(gpa, &seqs, wins[0..nw]);
                // PETABRIDGE (k2a-pb1) spill P6 (follower W3: verify/draft rows)
                if (spill) |sd| for (wins[0..nw]) |w| { // (review: ratio-2 groups, and the extent's scratch row)
                    const span = bumpSpan(w.base + w.start, w.base + w.start + w.ids.len);
                    sd.bump(span[0], span[1]);
                    sd.bump(w.end -| 2, w.end);
                };
                m.verify(rows, built.absorb[0..nw]) catch |err| refused(err);
                if (m.prof) {
                    follow_prof.rounds += 1;
                    follow_prof.enqueue += m.t_enqueue;
                    follow_prof.forward += m.t_forward;
                }
            },
            .pass => {
                const n = try r.int(u32);
                const steps = try r.int(u32);
                if (n == 0 or n > max_streams) return error.BadFrame;
                var tokens: [max_streams]i64 = undefined;
                var q0: [max_streams]i64 = undefined;
                var slots: [max_streams]i64 = undefined;
                for (0..n) |i| {
                    tokens[i] = try r.int(i64);
                    q0[i] = try r.int(i64);
                    slots[i] = try r.int(i64);
                }
                m.pass(tokens[0..n], q0[0..n], slots[0..n], steps) catch |err| refused(err);
            },
            .fill_images => {
                const count = try r.int(u32);
                var spans: std.ArrayList(@import("prompt.zig").Span) = .empty;
                defer spans.deinit(gpa);
                var row: usize = 0;
                for (0..count) |_| {
                    const at: usize = @intCast(try r.int(u64));
                    const n: usize = @intCast(try r.int(u64));
                    try spans.append(gpa, .{ .at = at, .len = n, .row = row });
                    row += n;
                }
                const off: usize = @intCast(try r.int(u64));
                const piece = r.rest();
                const total = row * m.cfg.hidden * 2;
                if (off != img_rows.items.len or off + piece.len > total or piece.len == 0) return error.BadFrame;
                try img_rows.appendSlice(gpa, piece);
                if (img_rows.items.len == total) {
                    m.setImages(spans.items, img_rows.items) catch |err| refused(err);
                    img_rows.clearRetainingCapacity();
                }
            },
            .snap_save => {
                const id = try r.int(u64);
                const s: usize = try r.int(u32);
                if (s >= max_streams) return error.BadSlot;
                var b = try cuda.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
                m.snapCopy(s, b.ptr, true) catch |err| {
                    b.free();
                    refused(err);
                    continue;
                };
                try kept.put(gpa, id, b);
            },
            .snap_restore => {
                const id = try r.int(u64);
                const s: usize = try r.int(u32);
                if (s >= max_streams) return error.BadSlot;
                const from: usize = @intCast(try r.int(u64));
                const to: usize = @intCast(try r.int(u64));
                const at: usize = @intCast(try r.int(u64));
                const b = kept.get(id) orelse return error.UnknownSnapshot; // rank 0 restores only states both ranks keep
                if (from != to) {
                    if (spill) |sd| sd.bump(to, to + at); // PETABRIDGE (k2a-pb1) spill P6 (follower W4: copyExtent)
                    m.copyExtent(from, to, at) catch |err| refused(err);
                }
                m.snapCopy(s, b.ptr, false) catch |err| refused(err);
            },
            .snap_drop => {
                const id = try r.int(u64);
                // PETABRIDGE (k2a-pb1) spill P6 (0016): its write guarded, its small-state capture done, before the free
                if (spill) |sd| sd.dropSnap(id);
                if (kept.fetchRemove(id)) |kv| {
                    var b = kv.value;
                    b.free();
                }
            },
            // PETABRIDGE (k2a-pb1) spill P6: rank_wire's frames (0019 v2: every spill command, notices included)
            .rank_wire => try (spill orelse return error.SpillNotStarted).wireFrame(peer, r.rest()),
            .done => {
                if (spill) |sd| {
                    const left = sd.flush(sd.flush_ns);
                    if (left > 0) std.log.info("dsv41 spill: {d} kept states were still unpublished at the flush deadline", .{left});
                }
                return;
            },
        }
    }
}

test "draft chances are the prefix survivals chooseK sums" {
    var out: [3]f64 = undefined;
    survivals(&.{ 0.0, 0.0, 2.0 }, &out);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), out[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.25 / (1.0 + @exp(@as(f64, -2.0))), out[2], 1e-12);
}

test "extents take the first chunk-aligned gap" {
    const used = [_][2]usize{ .{ 0, 4096 }, .{ 6144, 8192 } };
    try std.testing.expectEqual(@as(?usize, 4096), place(&used, 2048, 1 << 20));
    try std.testing.expectEqual(@as(?usize, 8192), place(&used, 4096, 1 << 20));
    try std.testing.expectEqual(@as(?usize, 0), place(&.{}, 2048, 2048));
    try std.testing.expectEqual(@as(?usize, null), place(&used, 4096, 10240));
}

test "a frame's fields read back as written" {
    const gpa = std.testing.allocator;
    var w: Wire = .{};
    defer w.deinit(gpa);
    try w.begin(gpa, .verify, 7);
    try w.int(gpa, u64, 1 << 40);
    try w.ids(gpa, &.{ 1, -2, 129279 });
    var r: Reader = .{ .b = w.buf.items };
    try std.testing.expectEqual(@as(u8, @backingInt(Kind.verify)), try r.int(u8));
    try std.testing.expectEqual(@as(u64, 7), try r.int(u64));
    try std.testing.expectEqual(@as(u64, 1 << 40), try r.int(u64));
    var got: std.ArrayList(i32) = .empty;
    defer got.deinit(gpa);
    try r.ids(gpa, &got);
    try std.testing.expectEqualSlices(i32, &.{ 1, -2, 129279 }, got.items);
    try std.testing.expectError(error.ShortFrame, r.int(u8));
}

test "a round's rows: each window at its positions, its ids written over the rejected rows" {
    const gpa = std.testing.allocator;
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    try seqs[0].appendSlice(gpa, &.{ 10, 11, 12, 13, 99, 98 }); // positions 4.. hold a rejected window
    try seqs[2].appendSlice(gpa, &.{ 20, 21 });
    var b: Built = .{};
    const rows = try b.build(gpa, &seqs, &.{
        .{ .slot = 0, .base = 0, .end = 4096, .start = 4, .absorb = true, .ids = &.{ 14, 15 } },
        .{ .slot = 2, .base = 4096, .end = 8192, .start = 2, .absorb = false, .ids = &.{22} },
    });
    try std.testing.expectEqualSlices(i64, &.{ 14, 15, 22 }, rows.ids);
    try std.testing.expectEqualSlices(i64, &.{ 4, 5, 2 }, rows.pos);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 2 }, rows.slots.?);
    try std.testing.expectEqualSlices(i64, &.{ 4096, 4096, 8192 }, rows.ends.?);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 12, 13, 14, 15 }, rows.windows.?[0].seq);
    try std.testing.expectEqualSlices(i32, &.{ 20, 21, 22 }, rows.windows.?[1].seq);
    try std.testing.expectEqual(@as(usize, 2), rows.windows.?[1].row);
    try std.testing.expectError(error.SequenceGap, b.build(gpa, &seqs, &.{.{ .slot = 1, .base = 0, .end = 2048, .start = 3, .absorb = false, .ids = &.{1} }}));
}

test "greedy draws take the first largest logit" {
    const row = [_]f32{ 0.5, 2.0, -1.0, 2.0 };
    try std.testing.expectEqual(@as(u32, 1), try draw(&row, 9, null));
    try std.testing.expectEqual(@as(u32, 1), try draw(&row, 9, .{ .seed = 3, .temperature = 0 }));
}

test "prefix growth relocates instead of overwriting a retained neighbour" {
    var a: Snap = .{ .id = 1, .at = 6144, .base = 0, .buf = undefined };
    var b: Snap = .{ .id = 2, .at = 6144, .base = 6144, .buf = undefined };
    var m: Model = undefined;
    m.pool_cap = 3300000;
    var l: Lanes = undefined;
    l.gpa = std.testing.allocator;
    l.m = &m;
    l.snaps = .empty;
    l.spill = null; // PETABRIDGE (k2a-pb1) spill P6: the tier field (off) of an undefined Lanes
    defer l.snaps.deinit(l.gpa);
    try l.snaps.appendSlice(l.gpa, &.{ &a, &b });
    try std.testing.expect(l.overwritesKept(6144, 10240, &a));
    try std.testing.expectEqual(@as(usize, 12288 + growth_rows), try l.placeKept(&.{}, 10240, &a));
    try std.testing.expect(!a.stale and !b.stale);
    try std.testing.expect(!l.overwritesKept(12288, 22528, &a));
}

test "retention: placement gives up a never-resumed state before a resumed conversation, whatever their age" {
    retention_on = true;
    defer retention_on = null;
    // main: the oldest, resumed by later turns; one_shot: newer, never resumed
    var main: Snap = .{ .id = 1, .at = 6144, .base = 0, .buf = undefined, .resumes = 3, .used = 10 };
    var one_shot: Snap = .{ .id = 2, .at = 6144, .base = 6144, .buf = undefined, .used = 20 };
    var m: Model = undefined;
    m.pool_cap = 16384;
    var l: Lanes = undefined;
    l.gpa = std.testing.allocator;
    l.m = &m;
    l.snaps = .empty;
    l.spill = null; // PETABRIDGE (k2a-pb1) spill P6: the tier field (off) of an undefined Lanes
    defer l.snaps.deinit(l.gpa);
    try l.snaps.appendSlice(l.gpa, &.{ &main, &one_shot });
    // room for 10240 means giving up one of them: the one-shot goes, the older main conversation stays
    try std.testing.expectEqual(@as(usize, 6144), try l.placeKept(&.{}, 10240, null));
    try std.testing.expect(!main.stale and one_shot.stale);
}

test "retention: tiers are never resumed, then superseded, then live conversations, least recently used first in each" {
    retention_on = true;
    defer retention_on = null;
    var head_old: Snap = .{ .id = 1, .at = 2048, .base = 0, .buf = undefined, .resumes = 2, .used = 1 };
    var shared: Snap = .{ .id = 2, .at = 2048, .base = 2048, .buf = undefined, .resumes = 40, .used = 90, .superseded = true };
    var prefix: Snap = .{ .id = 3, .at = 2048, .base = 4096, .buf = undefined, .resumes = 1, .used = 5, .superseded = true };
    var fresh: Snap = .{ .id = 4, .at = 2048, .base = 6144, .buf = undefined, .used = 80 };
    var stale_one: Snap = .{ .id = 5, .at = 2048, .base = 8192, .buf = undefined, .used = 0, .stale = true };
    const all = [_]*Snap{ &head_old, &shared, &prefix, &fresh, &stale_one };
    try std.testing.expectEqual(@as(?*Snap, &fresh), Lanes.victim(&all, &.{}, null));
    fresh.stale = true;
    try std.testing.expectEqual(@as(?*Snap, &prefix), Lanes.victim(&all, &.{}, null));
    prefix.stale = true;
    try std.testing.expectEqual(@as(?*Snap, &shared), Lanes.victim(&all, &.{}, null));
    shared.stale = true;
    try std.testing.expectEqual(@as(?*Snap, &head_old), Lanes.victim(&all, &.{}, null));
    // never the restore source, nor a state whose rows a live stream owns
    try std.testing.expectEqual(@as(?*Snap, null), Lanes.victim(&all, &.{}, &head_old));
    try std.testing.expectEqual(@as(?*Snap, null), Lanes.victim(&all, &.{.{ 0, 4096 }}, null));
}

test "retention off (TF_DS_RETENTION=0) gives up the oldest, as upstream" {
    retention_on = false;
    defer retention_on = null;
    var main: Snap = .{ .id = 1, .at = 2048, .base = 0, .buf = undefined, .resumes = 9, .used = 50 };
    var one_shot: Snap = .{ .id = 2, .at = 2048, .base = 2048, .buf = undefined };
    const all = [_]*Snap{ &main, &one_shot };
    try std.testing.expectEqual(@as(?*Snap, &main), Lanes.victim(&all, &.{}, null));
}

test "placement keeps restore source until genuine pool pressure requires cold fallback" {
    var a: Snap = .{ .id = 1, .at = 6144, .base = 0, .buf = undefined };
    var b: Snap = .{ .id = 2, .at = 6144, .base = 6144, .buf = undefined };
    var m: Model = undefined;
    m.pool_cap = 16384;
    var l: Lanes = undefined;
    l.gpa = std.testing.allocator;
    l.m = &m;
    l.snaps = .empty;
    l.spill = null; // PETABRIDGE (k2a-pb1) spill P6: the tier field (off) of an undefined Lanes
    defer l.snaps.deinit(l.gpa);
    try l.snaps.appendSlice(l.gpa, &.{ &a, &b });
    try std.testing.expectEqual(@as(usize, 6144), try l.placeKept(&.{}, 10240, &a));
    try std.testing.expect(!a.stale and b.stale);
    m.pool_cap = 10240;
    try std.testing.expectEqual(@as(usize, 0), try l.placeKept(&.{}, 10240, &a));
    try std.testing.expect(a.stale);
}

test "full prompt snapshots are exact-length only inside bounded replay by default" {
    var m: Model = undefined;
    m.cfg.window = 128;
    var sn: Snap = .{ .id = 1, .at = 8191, .base = 0, .buf = undefined, .complete = true };
    var l: Lanes = undefined;
    l.m = &m;
    l.gpa = std.testing.allocator;
    l.snaps = .empty;
    l.spill = null; // PETABRIDGE (k2a-pb1) spill P6: the tier field (off) of an undefined Lanes
    defer l.snaps.deinit(l.gpa);
    try l.snaps.append(l.gpa, &sn);
    l.short_continuation = false;
    try std.testing.expect(l.usable(&sn, 8191));
    try std.testing.expect(!l.usable(&sn, 8192));
    try std.testing.expect(!l.usable(&sn, 8400)); // pending ratio-2 row: not a prompt chunk start
    sn.at = 8190;
    try std.testing.expect(l.usable(&sn, 8318));
    try std.testing.expect(!l.usable(&sn, 8317));
    sn.stale = true;
    try std.testing.expect(!l.usable(&sn, 8190));
}

test "opt-in short continuation requires a live aligned fully decoded prefix" {
    var m: Model = undefined;
    m.cfg.window = 128;
    var sn: Snap = .{ .id = 1, .at = 9126, .base = 0, .buf = undefined, .decoded = true };
    var l: Lanes = undefined;
    l.m = &m;
    l.gpa = std.testing.allocator;
    l.snaps = .empty;
    l.spill = null; // PETABRIDGE (k2a-pb1) spill P6: the tier field (off) of an undefined Lanes
    l.short_continuation = false;
    defer l.snaps.deinit(l.gpa);
    try l.snaps.append(l.gpa, &sn);
    try std.testing.expect(!l.usable(&sn, 9145)); // actual workflow miss remains the default
    l.short_continuation = true;
    try std.testing.expect(l.usable(&sn, 9145));
    try std.testing.expect(l.shortResume(&sn, 9254));
    try std.testing.expect(!l.shortResume(&sn, 9255));
    try std.testing.expect(l.usable(&sn, 9255)); // ordinary encoder reuse, no fast continuation
    try std.testing.expect(!l.usable(&sn, 9126)); // accepted state does not carry its exact head logits
    try std.testing.expect(!l.usable(&sn, 9125));
    sn.decoded = false;
    try std.testing.expect(!l.usable(&sn, 9145)); // encoder-only remains protected
    sn.complete = true;
    try std.testing.expect(l.usable(&sn, 9126));
    try std.testing.expect(l.usable(&sn, 9145));
    sn.at = 9127;
    try std.testing.expect(!l.usable(&sn, 9145)); // ratio-2 partial group cannot start a fill chunk
    try std.testing.expect(l.usable(&sn, 9127)); // exact head-only restore is still allowed
    sn.at = 9126;
    sn.stale = true;
    try std.testing.expect(!l.usable(&sn, 9145));
}

test "telemetry unions snapshot overlaps and excludes stale and live overlap" {
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = undefined, .peer = null };
    defer ln.snaps.deinit(std.testing.allocator);
    var a: Snap = .{ .id = 1, .base = 100, .at = 200, .buf = undefined };
    var b: Snap = .{ .id = 2, .base = 200, .at = 200, .buf = undefined };
    var stale: Snap = .{ .id = 3, .base = 1000, .at = 200, .buf = undefined, .stale = true };
    try ln.snaps.append(std.testing.allocator, &a);
    try ln.snaps.append(std.testing.allocator, &b);
    try ln.snaps.append(std.testing.allocator, &stale);
    const active = [_][2]usize{ .{ 0, 150 }, .{ 500, 550 } };
    try std.testing.expectEqual(@as(usize, 450), ln.retained(&active));
    try std.testing.expectEqual(@as(usize, 300), ln.retained(&.{}));
    b.stale = true;
    try std.testing.expectEqual(@as(usize, 350), ln.retained(&active));
}

test "a short shared sibling cannot overwrite the main conversation's longer prefix" {
    var common: Snap = .{ .id = 1, .at = 8192, .base = 0, .buf = undefined };
    var main: Snap = .{ .id = 2, .at = 98304, .base = 0, .end = 131072, .buf = undefined };
    var m: Model = undefined;
    m.pool_cap = 524288;
    var l: Lanes = undefined;
    l.gpa = std.testing.allocator;
    l.m = &m;
    l.snaps = .empty;
    l.spill = null; // PETABRIDGE (k2a-pb1) spill P6: the tier field (off) of an undefined Lanes
    defer l.snaps.deinit(l.gpa);
    try l.snaps.appendSlice(l.gpa, &.{ &common, &main });
    // A1 retained main; B1 resumes common. This invariant already holds in CDEF.
    try std.testing.expect(l.overwritesKept(common.at, 16384, &common));
    const b = try l.placeKept(&.{}, 16384, &common);
    l.staleOver(b, b + 16384, &common);
    try std.testing.expect(!main.stale and !common.stale);
    try std.testing.expect(b >= main.end);
    // A2 can still reuse main in place after B1 finishes, without changing its prefix.
    try std.testing.expect(!l.overwritesKept(main.at, main.end, &main));
}

test "soft reservations yield to packing before any kept prefix is evicted" {
    const c = model.chunk_rows;
    const a = std.testing.allocator;
    const kept = [_][3]usize{.{ 0, 4 * c, 8 * c }};
    try std.testing.expectEqual(@as(?usize, 8 * c + growth_rows), try placeSoft(a, &.{}, &kept, 4 * c, 128 * c));
    try std.testing.expectEqual(@as(?usize, 8 * c), try placeSoft(a, &.{}, &kept, 4 * c, 12 * c));
    try std.testing.expectEqual(@as(?usize, 4 * c), try placeSoft(a, &.{}, &kept, 4 * c, 8 * c));
    try std.testing.expectEqual(@as(?usize, null), try placeSoft(a, &.{.{ 4 * c, 8 * c }}, &kept, 4 * c, 8 * c));
}

test "pool pressure cannot evict a retained prefix whose rows remain owned by a live stream" {
    var snap: Snap = .{ .id = 1, .at = 8192, .base = 0, .end = 16384, .buf = undefined };
    var m: Model = undefined;
    m.pool_cap = 16384;
    var l: Lanes = undefined;
    l.gpa = std.testing.allocator;
    l.m = &m;
    l.snaps = .empty;
    l.spill = null; // PETABRIDGE (k2a-pb1) spill P6: the tier field (off) of an undefined Lanes
    defer l.snaps.deinit(l.gpa);
    try l.snaps.append(l.gpa, &snap);
    try std.testing.expectError(error.ContextFull, l.placeKept(&.{.{ 0, 16384 }}, 2048, null));
    try std.testing.expect(!snap.stale);
}


test "K2a empty active interval cannot stop union before later retained snapshots" {
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = undefined, .peer = null };
    defer ln.snaps.deinit(std.testing.allocator);
    var a: Snap = .{ .id = 1, .base = 10, .at = 10, .buf = undefined };
    var b: Snap = .{ .id = 2, .base = 100, .at = 20, .buf = undefined };
    try ln.snaps.appendSlice(ln.gpa, &.{ &a, &b });
    const active = [_][2]usize{ .{ 0, 0 }, .{ 30, 30 }, .{ 200, 200 }, .{ 15, 25 } };
    try std.testing.expectEqual(@as(usize, 35), ln.retained(&active));
    try std.testing.expectEqual(@as(usize, 30), ln.retained(&.{.{ 30, 30 }}));
    a.stale = true;
    b.stale = true;
    try std.testing.expectEqual(@as(usize, 0), ln.retained(&.{ .{ 0, 0 }, .{ 10, 10 } }));
}

test "K2a retained union agrees with independent bitmap oracle in 10000 interval cases" {
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = undefined, .peer = null };
    defer ln.snaps.deinit(std.testing.allocator);
    var snaps: [8]Snap = undefined;
    for (&snaps, 0..) |*sn, i| {
        sn.* = .{ .id = @intCast(i), .base = 0, .at = 0, .buf = undefined };
        try ln.snaps.append(ln.gpa, sn);
    }
    var prng = std.Random.DefaultPrng.init(0x4b32564b);
    const rng = prng.random();
    for (0..10000) |_| {
        var occupied: [256]bool = undefined;
        @memset(&occupied, false);
        var active: [8][2]usize = undefined;
        for (&active) |*r| {
            r[0] = rng.uintLessThan(usize, 256);
            r[1] = r[0] + rng.uintLessThan(usize, 257 - r[0]);
            for (r[0]..r[1]) |j| occupied[j] = true;
        }
        for (&snaps) |*sn| {
            sn.base = rng.uintLessThan(usize, 256);
            sn.at = rng.uintLessThan(usize, 257 - sn.base);
            sn.stale = rng.boolean();
            if (!sn.stale) for (sn.base..sn.base + sn.at) |j| { occupied[j] = true; };
        }
        var expected: usize = 0;
        for (occupied) |yes| if (yes) { expected += 1; };
        try std.testing.expectEqual(expected, ln.retained(&active));
    }
}

test "PETABRIDGE spill P6: a restore's destination is held back from placement, never a victim, never usable or live until installed, and installs into retention's probation" {
    // (MiaAI-Lab 0016's test, fitted to k2a's retention victim and usable)
    retention_on = true;
    defer retention_on = null;
    const c = model.chunk_rows;
    var m: Model = undefined;
    m.pool_cap = 16 * c;
    m.cfg.window = 128;
    const gpa = std.testing.allocator;
    var ln: Lanes = .{ .gpa = gpa, .m = &m, .peer = null };
    defer ln.snaps.deinit(gpa);
    // a restore reading into [0, 4c), an ordinary kept state at [4c, 8c)
    var loading: Snap = .{ .id = 1, .at = 4 * c, .base = 0, .end = 4 * c, .buf = undefined, .loading = true };
    var kept: Snap = .{ .id = 2, .at = 4 * c, .base = 4 * c, .end = 8 * c, .buf = undefined, .resumes = 5, .used = 99 };
    try ln.snaps.append(gpa, &loading);
    try ln.snaps.append(gpa, &kept);
    try std.testing.expect(!ln.usable(&loading, 8 * c)); // not until every rank installed it
    try std.testing.expect(!Lanes.snapLiveFn(&ln, &loading));
    try std.testing.expect(ln.usable(&kept, 9 * c));
    // never a victim, even as the lowest tier and the least recently used: the ordinary state goes
    try std.testing.expectEqual(@as(?*Snap, &kept), Lanes.victim(ln.snaps.items, &.{}, null));
    const base = try ln.placeKept(&.{}, 12 * c, null);
    try std.testing.expect(kept.stale and !loading.stale);
    try std.testing.expectEqual(4 * c, base);
    // with nothing else to give, the destination still holds: the pool is full for this request
    kept.stale = false;
    try std.testing.expectError(error.ContextFull, ln.placeKept(&.{.{ 4 * c, 16 * c }}, 4 * c, null));
    // installed: usable, live, restored, in probation (never resumed), recently used
    loading.resumes = 7;
    loading.superseded = true;
    Lanes.spillInstall(&ln, &loading);
    try std.testing.expect(!loading.loading and loading.restored);
    try std.testing.expectEqual(@as(u32, 0), loading.resumes);
    try std.testing.expect(!loading.superseded and loading.used > 0);
    try std.testing.expect(ln.usable(&loading, 8 * c));
    try std.testing.expect(Lanes.snapLiveFn(&ln, &loading));
    // its family info: a plain grid state, eligible; the frame kinds before rank_wire keep their values
    const info = Lanes.spillInfo(&ln, &loading);
    try std.testing.expect(info.eligible and info.at == 4 * c and info.base == 0);
    try std.testing.expectEqual(@as(u8, 9), @intFromEnum(Kind.done));
    try std.testing.expectEqual(@as(u8, 10), @intFromEnum(Kind.rank_wire));
    try std.testing.expect(ln.spill == null);
}

test "PETABRIDGE spill P6: with the tier off nothing spill-related runs: no guard, the k2a placement and staleness as before" {
    const c = model.chunk_rows;
    var m: Model = undefined;
    m.pool_cap = 16 * c;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null };
    defer ln.snaps.deinit(ln.gpa);
    var a: Snap = .{ .id = 1, .at = 6 * c, .base = 0, .buf = undefined };
    var b: Snap = .{ .id = 2, .at = 6 * c, .base = 6 * c, .buf = undefined };
    try ln.snaps.appendSlice(ln.gpa, &.{ &a, &b });
    try std.testing.expect(ln.spill == null);
    // (a guard would dereference the tier: these paths ran without one)
    ln.staleOver(0, 2 * c, null);
    try std.testing.expect(a.stale and !b.stale);
    a.stale = false;
    try std.testing.expectEqual(@as(usize, 6 * c), try ln.placeKept(&.{}, 10 * c, &a));
    try std.testing.expect(!a.stale and b.stale);
    // a continuation, complete or off-grid state stays memory-only whatever the tier: spillInfo says ineligible
    var cont: Snap = .{ .id = 3, .at = 4 * c, .base = 0, .buf = undefined, .decoded = true };
    try std.testing.expect(!Lanes.spillInfo(&ln, &cont).eligible);
    var off: Snap = .{ .id = 4, .at = 4 * c + 6, .base = 0, .buf = undefined };
    try std.testing.expect(!Lanes.spillInfo(&ln, &off).eligible);
}
