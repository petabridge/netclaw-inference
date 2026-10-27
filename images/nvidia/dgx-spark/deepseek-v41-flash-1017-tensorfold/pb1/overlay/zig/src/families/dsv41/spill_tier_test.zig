//! spill_tier's host tests: the Tier over a real core/spill store on a scratch directory (TF_SPILL_TEST_DIR), the
//! packing over a byte-array device whose copies complete after a chosen number of event polls, and for two ranks a
//! follower thread over a loopback link (rank_wire's own harness). Covers write at keep and publication, lookup,
//! restore (rows and small state byte for byte), a failure on either rank (abort everywhere, the point forgotten on
//! every rank), cancel mid-restore (issued copies waited for, destination cleared to its sentinels), a stale extent
//! during staging, the guard, presence refused, and the shutdown flush.
//! PETABRIDGE (k2a-pb1) spill P6: MiaAI-Lab's tip tests, fitted (every change marked): the owner-sharded and picture
//! tests went with those features (not ported), 0020's Tier.full checks too; added: the refusal rules (pictures, a
//! state the family marks ineligible, off-grid), the owner-sharded refusal at create, and world 3 (two followers over
//! loopback: written on every rank, a rank missing, damaged or out of room aborts everywhere).
const std = @import("std");
const core = @import("core");
const tier = @import("spill_tier.zig");
const state = @import("spill_state.zig");
const rank_wire = @import("rank_wire.zig");
const fingerprint = @import("fingerprint.zig");
const link = @import("link.zig");

const spill = core.spill;
const format = spill.format;
const pc = tier.prompt_cache;
const testing = std.testing;
const io = testing.io;
const gpa = testing.allocator;

const bt = 16; // block tokens
const cap = 16 * bt; // pool positions
const max_snaps = 16;

fn testLayout() !state.StateLayout {
    const shapes = [_]state.LayerShape{ .{ .ratio = 2, .kv = true, .idx = true, .raw = true }, .{ .ratio = 1, .kv = true, .idx = true } };
    return state.StateLayout.init(&shapes, .{ .block_tokens = bt, .window = 8, .ring_extra = 4, .raw_rows = 4, .head_dim = 64, .index_head_dim = 64 }, .replicated);
}

/// The owner-sharded twin: two parts of half the pool, each rank's codes and scales its own part's only.
const part_rows = cap / 2;
fn shardedLayout() !state.StateLayout {
    const shapes = [_]state.LayerShape{ .{ .ratio = 2, .kv = true, .idx = true, .raw = true }, .{ .ratio = 1, .kv = true, .idx = true } };
    const l = try state.StateLayout.init(&shapes, .{ .block_tokens = bt, .window = 8, .ring_extra = 4, .raw_rows = 4, .head_dim = 64, .index_head_dim = 64 }, .owner_sharded);
    return l.withParts(part_rows);
}

/// The device: block planes over the pool, then Snap buffers; copies immediate, events complete after `delay` polls.
const Sim = struct {
    mem: []u8,
    layout: state.StateLayout,
    planes: [state.max_planes]u64 = @splat(0),
    snap_base: u64 = 0,
    snap_used: [max_snaps]bool = @splat(false),
    delay: u32 = 0,
    pending: [1024]u32 = @splat(0),
    waits: u32 = 0,
    rank: u8,

    const origin: u64 = 4096;

    fn init(rank: u8) !*Sim {
        return initWith(rank, try testLayout());
    }

    /// `layout` owner_sharded: an owner_only plane holds this rank's part's rows only (part_rows / ratio + 2), from its
    /// part's first position (prompt.Caches.initShard).
    fn initWith(rank: u8, layout: state.StateLayout) !*Sim {
        const s = try gpa.create(Sim);
        s.* = .{ .mem = &.{}, .layout = layout, .rank = rank };
        var at: u64 = origin;
        for (s.layout.blockPlanes(), 0..) |p, i| {
            s.planes[i] = at;
            at += (@as(usize, if (s.sharded(p)) part_rows else cap) / p.ratio + 2) * p.row;
        }
        s.snap_base = at;
        at += max_snaps * s.layout.small_bytes;
        s.mem = try gpa.alloc(u8, at - origin);
        for (s.mem, 0..) |*b, i| b.* = @truncate((i *% 2654435761 +% @as(usize, rank) *% 977) >> 5);
        return s;
    }

    fn deinit(s: *Sim) void {
        gpa.free(s.mem);
        gpa.destroy(s);
    }

    fn bytes(s: *Sim, a: u64, len: usize) []u8 {
        return s.mem[a - origin ..][0..len];
    }

    fn allocSnap(s: *Sim) !u64 {
        for (&s.snap_used, 0..) |*u, i| if (!u.*) {
            u.* = true;
            return s.snap_base + i * s.layout.small_bytes;
        };
        return error.NoSnapRoom;
    }

    fn freeSnap(s: *Sim, buf: u64) void {
        s.snap_used[(buf - s.snap_base) / s.layout.small_bytes] = false;
    }

    fn sharded(s: *const Sim, p: state.Plane) bool {
        return p.owner_only and s.layout.owner_layout == .owner_sharded;
    }

    /// Whether this rank holds plane i's rows at pool position `pos` (an owner_only plane: its own part's only).
    fn holds(s: *const Sim, i: usize, pos: usize) bool {
        const p = s.layout.blockPlanes()[i];
        return !s.sharded(p) or pos / part_rows == s.rank;
    }

    /// A block plane's rows of pool positions [lo, hi) (an owner_only plane's: its part's local rows).
    fn rows(s: *Sim, i: usize, lo: usize, hi: usize) []u8 {
        const p = s.layout.blockPlanes()[i];
        const at = if (s.sharded(p)) lo - lo / part_rows * part_rows else lo;
        return s.bytes(s.planes[i] + at / p.ratio * p.row, (hi - lo) / p.ratio * p.row);
    }

    fn sameRows(s: *Sim, a: usize, b: usize, n: usize) bool {
        for (0..s.layout.n_block) |i| if (s.holds(i, a) and s.holds(i, b)) {
            if (!std.mem.eql(u8, s.rows(i, a, a + n), s.rows(i, b, b + n))) return false;
        };
        return true;
    }

    fn clobber(s: *Sim, lo: usize, hi: usize) void {
        for (0..s.layout.n_block) |i| if (s.holds(i, lo)) @memset(s.rows(i, lo, hi), 0xA5);
    }

    fn cleared(s: *Sim, lo: usize, hi: usize) bool {
        for (s.layout.blockPlanes(), 0..) |p, i| if (s.holds(i, lo)) for (s.rows(i, lo, hi)) |b| if (b != p.sentinel) return false;
        return true;
    }
};

const Dev = struct {
    sim: *Sim,
    pub const Event = u32;

    pub fn blockPlane(d: Dev, p: state.Plane) u64 {
        for (d.sim.layout.blockPlanes(), 0..) |q, i| if (q.kind == p.kind and q.layer == p.layer) return d.sim.planes[i];
        return 0;
    }
    pub fn slotPlane(d: Dev, p: state.Plane, slot: usize) u64 {
        _ = .{ p, slot };
        return d.sim.snap_base;
    }
    pub fn capacity(_: Dev) usize {
        return cap;
    }
    pub fn slots(_: Dev) usize {
        return 4;
    }
    pub fn d2h(d: Dev, dst: []u8, src: u64) !void {
        @memcpy(dst, d.sim.bytes(src, dst.len));
    }
    pub fn h2d(d: Dev, dst: u64, src: []const u8) !void {
        @memcpy(d.sim.bytes(dst, src.len), src);
    }
    pub fn fill(d: Dev, dst: u64, v: u8, len: usize) !void {
        @memset(d.sim.bytes(dst, len), v);
    }
    pub fn record(d: Dev, ev: u32) !void {
        d.sim.pending[ev] = d.sim.delay;
    }
    pub fn done(d: Dev, ev: u32) bool {
        if (d.sim.pending[ev] == 0) return true;
        d.sim.pending[ev] -= 1;
        return false;
    }
    pub fn wait(d: Dev, ev: u32) void {
        if (d.sim.pending[ev] > 0) d.sim.waits += 1;
        d.sim.pending[ev] = 0;
    }
};

const L = tier.Leader(Dev);
const F = tier.Follower(Dev);

fn bigFree(_: ?*anyopaque) u64 {
    return 1 << 40;
}

/// One rank's pieces: its device, staging, events and a scratch store directory.
const Rank = struct {
    sim: *Sim,
    staging: []align(4096) u8,
    events: []u32,
    path: [256]u8 = undefined,
    root: [:0]const u8 = undefined,
    settings: tier.Settings,
    scan_ns: u64 = 0, // the follower's store scan: this long a name (a slow disk)
    stall_ns: u64 = 0, // ... and this long once, at its first name (a scan that stops)
    stalled: bool = false,

    fn init(rank: u8, tag: []const u8, staging_bytes: u64) !*Rank {
        return initWith(rank, tag, staging_bytes, try testLayout());
    }

    fn initWith(rank: u8, tag: []const u8, staging_bytes: u64, layout: state.StateLayout) !*Rank {
        const r = try gpa.create(Rank);
        r.* = .{ .sim = try Sim.initWith(rank, layout), .staging = &.{}, .events = &.{}, .settings = undefined };
        var tb: [64]u8 = undefined;
        r.root = try spill.fsio.testRoot(&r.path, try std.fmt.bufPrint(&tb, "tier-{s}-r{d}", .{ tag, rank }));
        const fd = try spill.fsio.freshDir(gpa, r.root);
        _ = std.c.close(fd);
        r.settings = .{ .dir = r.root, .gib = 1, .min_free_gib = 0, .min_tokens = bt, .staging_bytes = if (staging_bytes > 0) staging_bytes else 1 << 20, .readers = 2 };
        const plan = try tier.stagingPlan(r.settings, &r.sim.layout);
        r.staging = try gpa.alignedAlloc(u8, .fromByteUnits(4096), plan.bytes());
        r.events = try gpa.alloc(u32, tier.eventCount(plan));
        for (r.events, 0..) |*e, i| e.* = @intCast(i);
        return r;
    }

    fn deinit(r: *Rank) void {
        spill.fsio.dropDir(gpa, r.root);
        gpa.free(r.staging);
        gpa.free(r.events);
        r.sim.deinit();
        gpa.destroy(r);
    }

    fn compat(r: *Rank, a: fingerprint.Agreement, rank: u16) format.Digest {
        return tier.storeCompat(a, rank, &r.sim.layout);
    }

    fn side(r: *Rank, rank: u16, a: fingerprint.Agreement) !*L.S {
        const gens = try state.Generations.init(gpa, cap, bt);
        return L.S.create(gpa, io, .{ .sim = r.sim }, r.sim.layout, gens, r.settings, rank, r.compat(a, rank), cap, r.staging, r.events, .{ .free = bigFree });
    }

    fn opening(r: *Rank) !tier.Opening(Dev) {
        return .{ .dev = .{ .sim = r.sim }, .layout = r.sim.layout, .gens = try state.Generations.init(gpa, cap, bt), .settings = r.settings, .max_tokens = cap, .staging = r.staging, .events = r.events, .hooks = .{ .ctx = r, .free = bigFree, .scanned = slowScan } };
    }

    fn slowScan(ctx: ?*anyopaque, _: []const u8) void {
        const r: *Rank = @ptrCast(@alignCast(ctx.?));
        if (r.stall_ns > 0 and !r.stalled) {
            r.stalled = true;
            std.Io.sleep(io, .fromNanoseconds(r.stall_ns), .awake) catch {};
        }
        if (r.scan_ns > 0) std.Io.sleep(io, .fromNanoseconds(r.scan_ns), .awake) catch {};
    }

    /// `n` stray names in this rank's store under the test agreement's compat (its open scans each, then removes
    /// it as an orphan).
    fn strays(r: *Rank, rank: u16, n: usize) !void {
        const s = try r.side(rank, agreement());
        defer s.destroy();
        for (0..n) |i| {
            var nb: [32]u8 = undefined;
            const name = try std.fmt.bufPrintSentinel(&nb, "stray{d}", .{i}, 0);
            const fd = std.c.openat(s.store.blocks_fd, name, .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(std.c.mode_t, 0o600));
            if (fd < 0) return error.Unexpected;
            _ = std.c.close(fd);
        }
    }

    /// Flips a byte in every block file of this rank's store (damage CRC catches on the next read).
    fn corrupt(r: *Rank) !void {
        var root = try std.Io.Dir.cwd().openDir(io, r.root, .{ .iterate = true });
        defer root.close(io);
        var it = root.iterate();
        while (try it.next(io)) |ns| {
            if (ns.kind != .directory) continue;
            var sub_buf: [256]u8 = undefined;
            const sub = try std.fmt.bufPrint(&sub_buf, "{s}/rank{d}/blocks", .{ ns.name, r.sim.rank });
            var bd = root.openDir(io, sub, .{ .iterate = true }) catch continue;
            defer bd.close(io);
            var bit = bd.iterate();
            while (try bit.next(io)) |b| {
                if (b.kind != .file) continue;
                var f = try bd.openFile(io, b.name, .{ .mode = .read_write });
                defer f.close(io);
                var one: [1]u8 = undefined;
                _ = try f.readPositionalAll(io, &one, 7);
                one[0] ^= 0xff;
                try f.writePositionalAll(io, &one, 7);
            }
        }
    }
};

/// A kept state as the test's family holds it (lanes.zig's Snap stands in).
const Snap = struct { id: u64, base: usize, at: usize, buf: u64, slot: u32 = 0, installed: bool = false, part: ?u8 = null, eligible: bool = true }; // PETABRIDGE: eligible

/// Rank 0's family: restores get a fresh Snap at the next free base the test sets.
const Fam0 = struct {
    sim: *Sim,
    next_id: u64 = 1000,
    next_base: usize = 8 * bt,
    installed: u32 = 0,
    discarded: u32 = 0,
    reserved: u32 = 0,
    last_room: usize = 0,
    live: std.ArrayList(*Snap) = .empty,

    fn family(f: *Fam0) tier.Family {
        return .{ .ptr = f, .reserve = reserve, .install = install, .discard = discard, .info = info };
    }
    fn of(p: *anyopaque) *Fam0 {
        return @ptrCast(@alignCast(p));
    }
    fn reserve(p: *anyopaque, at: usize, room: usize) anyerror!tier.Dest { // PETABRIDGE: 0019's part dropped
        const f = of(p);
        f.reserved += 1;
        f.last_room = room;
        const sn = try gpa.create(Snap);
        sn.* = .{ .id = f.next_id, .base = f.next_base, .at = at, .buf = try f.sim.allocSnap() };
        f.next_id += 1;
        try f.live.append(gpa, sn);
        return .{ .saved = sn, .id = sn.id, .base = sn.base, .buf = sn.buf };
    }
    fn install(p: *anyopaque, saved: *anyopaque) void {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        sn.installed = true;
        of(p).installed += 1;
    }
    fn discard(p: *anyopaque, saved: *anyopaque) void {
        const f = of(p);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        f.discarded += 1;
        f.sim.freeSnap(sn.buf);
        for (f.live.items, 0..) |x, i| if (x == sn) {
            _ = f.live.swapRemove(i);
            break;
        };
        gpa.destroy(sn);
    }
    fn info(_: *anyopaque, saved: *anyopaque) tier.SnapInfo {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        return .{ .id = sn.id, .base = sn.base, .at = sn.at, .buf = sn.buf, .slot = sn.slot, .part = sn.part, .eligible = sn.eligible };
    }
    fn deinit(f: *Fam0) void {
        for (f.live.items) |sn| gpa.destroy(sn);
        f.live.deinit(gpa);
    }
};

fn testIds(out: []u32, seed: u32) void {
    for (out, 0..) |*x, i| x.* = seed *% 7919 +% @as(u32, @intCast(i)) *% 31 +% 3;
}

fn agreement() fingerprint.Agreement {
    var a: fingerprint.Agreement = .{ .common = @splat(1), .world = 2 };
    a.ranks[0] = @splat(10);
    a.ranks[1] = @splat(11);
    return a;
}

fn testFx(rank: u8) fingerprint.Exchange {
    var x: fingerprint.Exchange = .{ .common = @splat(1), .rank = @splat(10 + rank) };
    @memset(&x.fields, .{ .id = 0, .scope = .common, .digest = @splat(0) });
    return x;
}

/// Pumps until `n` writes were published (or a second passed).
fn pumpUntil(ld: *L, n: u64) !void {
    const t = ld.tier();
    var k: usize = 0;
    while (k < 1000) : (k += 1) {
        t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
        if (ld.side.store.counts().committed >= n) return;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.TestTimedOut;
}

fn pollUntilDone(t: pc.Tier, job: u64) pc.Poll {
    var k: usize = 0;
    while (k < 5000) : (k += 1) {
        const p = t.vtable.poll(t.ptr, job, std.time.ns_per_ms);
        if (p != .pending) return p;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return .{ .failed = error.TestTimedOut };
}

/// Staging back to none (reads still in flight on a reader give their slots back when they end).
fn stagingDrains(st: *spill.Store) !void {
    var k: usize = 0;
    while (k < 2000) : (k += 1) {
        if (st.counts().staging_bytes == 0) return;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.StagingHeld;
}

fn want(ids: []const u32) pc.Want {
    return .{ .key = ids, .beyond = 0, .max_at = @intCast(ids.len), .grid = bt };
}

/// A kept state at `base` holding `at` positions: its small state in a fresh buffer (the device's own bytes there).
fn keptSnap(sim: *Sim, id: u64, base: usize, at: usize) !*Snap {
    const sn = try gpa.create(Snap);
    sn.* = .{ .id = id, .base = base, .at = at, .buf = try sim.allocSnap() };
    return sn;
}

fn keep(t: pc.Tier, sn: *Snap, ids: []const u32, lineage: []const pc.Fork) ?u64 {
    return t.vtable.keep(t.ptr, &.{ .saved = sn, .key = ids, .at = @intCast(sn.at), .bytes = 0, .group = 9, .lineage = lineage });
}

// -- one rank -----------------------------------------------------------------------------------------------------

test "one rank: a kept state written at keep, found, restored byte for byte into a fresh extent" {
    const r = try Rank.init(0, "one", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0, agreement()), fam.family(), null, null, r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [4 * bt + 5]u32 = undefined;
    testIds(&ids, 1);
    const sn = try keptSnap(r.sim, 1, 0, 4 * bt);
    defer gpa.destroy(sn);
    const lin = [_]pc.Fork{.{ .name = 5, .at = 0 }};
    const id = keep(t, sn, &ids, &lin) orelse return error.NotKept;
    try testing.expect(t.vtable.busy.?(t.ptr));
    try pumpUntil(ld, 1);
    // found (the store's index, no I/O), at its own length, as the prompt cache asks
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    try testing.expectEqual(id, p.id);
    try testing.expectEqual(@as(u32, 4 * bt), p.at);
    try testing.expectEqual(@as(u64, 9), p.group);
    // a prompt that differs inside the point does not find it
    var other = ids;
    other[3] +%= 1;
    try testing.expect(t.vtable.lookup(t.ptr, &want(&other)) == null);
    // restore: the rows land in the family's fresh extent, the small state in its fresh buffer
    r.sim.delay = 2;
    const job = try t.vtable.begin(t.ptr, &p);
    const done = pollUntilDone(t, job);
    try testing.expect(done == .ready);
    const got: *Snap = @ptrCast(@alignCast(done.ready.saved));
    try testing.expect(got.installed);
    try testing.expect(r.sim.sameRows(0, got.base, 4 * bt));
    try testing.expectEqualSlices(u8, r.sim.bytes(sn.buf, r.sim.layout.small_bytes), r.sim.bytes(got.buf, r.sim.layout.small_bytes));
    // the point's lineage comes back as names that map to the same stored lineage
    try testing.expectEqual(@as(usize, 1), done.ready.lineage.len);
    try testing.expectEqualSlices(u8, &ld.name16(5), &ld.name16(done.ready.lineage[0].name));
    try testing.expectEqual(@as(u32, 1), fam.installed);
    try testing.expectEqual(@as(u64, 1), ld.side.cached.writes_committed);
    try stagingDrains(ld.side.store);
    fam.sim.freeSnap(got.buf);
}

test "one rank: rows that change before they are staged void the write; a guard stages them first" {
    const r = try Rank.init(0, "stale", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0, agreement()), fam.family(), null, null, r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 2);
    // 1. a placement takes the rows without a guard (the generation moves): the stage is refused, nothing publishes
    const a = try keptSnap(r.sim, 1, 0, 4 * bt);
    defer gpa.destroy(a);
    _ = keep(t, a, &ids, &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    ld.bump(0, 4 * bt);
    var k: usize = 0;
    while (k < 200 and t.vtable.busy.?(t.ptr)) : (k += 1) {
        t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try testing.expectEqual(@as(u64, 0), ld.side.store.counts().committed);
    try testing.expectEqual(@as(u64, 1), ld.side.store.counts().write_failed);
    try testing.expect(t.vtable.lookup(t.ptr, &want(&ids)) == null);
    // 2. the guard before the change: every block staged now, so the point publishes the bytes from before it
    testIds(&ids, 3);
    const b = try keptSnap(r.sim, 2, 4 * bt, 4 * bt);
    defer gpa.destroy(b);
    var before: [2][]u8 = undefined;
    before[0] = try gpa.dupe(u8, r.sim.rows(0, 4 * bt, 8 * bt));
    defer gpa.free(before[0]);
    _ = keep(t, b, &ids, &.{.{ .name = 2, .at = 0 }}) orelse return error.NotKept;
    ld.guardSnap(b.id, 0);
    ld.bump(4 * bt, 8 * bt);
    r.sim.clobber(4 * bt, 8 * bt);
    try pumpUntil(ld, 1);
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    fam.next_base = 12 * bt;
    const done = pollUntilDone(t, try t.vtable.begin(t.ptr, &p));
    try testing.expect(done == .ready);
    const got: *Snap = @ptrCast(@alignCast(done.ready.saved));
    try testing.expectEqualSlices(u8, before[0], r.sim.rows(0, got.base, got.base + 4 * bt));
    try testing.expect(ld.side.store.counts().guard_staged > 0);
    fam.sim.freeSnap(got.buf);
}

test "one rank: a guard without staging room cancels the unpublished write" {
    // staging for 4 block slots: 2 writes, 2 reads (and 6 small slots)
    const plan_small = std.mem.alignForward(u64, (try testLayout()).small_bytes, 4096);
    const r = try Rank.init(0, "guardcancel", 6 * plan_small + 4 * 4096);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0, agreement()), fam.family(), null, null, r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    try testing.expectEqual(@as(usize, 2), ld.side.plan.write);
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 4);
    const a = try keptSnap(r.sim, 1, 0, 4 * bt);
    defer gpa.destroy(a);
    _ = keep(t, a, &ids, &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    ld.guardSnap(a.id, 0); // four blocks left, two write slots: cancelled, not staged half
    ld.dropSnap(a.id);
    var k: usize = 0;
    while (k < 200 and t.vtable.busy.?(t.ptr)) : (k += 1) {
        t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    const c = ld.side.store.counts();
    try testing.expectEqual(@as(u64, 1), c.cancelled_guard);
    try testing.expectEqual(@as(u64, 0), c.committed);
    try stagingDrains(ld.side.store);
}

test "one rank: a damaged block fails the restore, forgets the point and clears the destination" {
    const r = try Rank.init(0, "damage", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0, agreement()), fam.family(), null, null, r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [3 * bt]u32 = undefined;
    testIds(&ids, 5);
    const a = try keptSnap(r.sim, 1, 0, 3 * bt);
    defer gpa.destroy(a);
    _ = keep(t, a, &ids, &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    try pumpUntil(ld, 1);
    try r.corrupt();
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    const done = pollUntilDone(t, try t.vtable.begin(t.ptr, &p));
    try testing.expectEqual(error.SpillDamaged, done.failed);
    try testing.expectEqual(@as(u32, 1), fam.discarded);
    try testing.expect(r.sim.cleared(8 * bt, 11 * bt));
    try testing.expect(t.vtable.lookup(t.ptr, &want(&ids)) == null);
    try stagingDrains(ld.side.store);
}

test "one rank: a cancel mid-restore waits for issued copies, clears the destination and releases everything" {
    const r = try Rank.init(0, "cancel", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0, agreement()), fam.family(), null, null, r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 6);
    const a = try keptSnap(r.sim, 1, 0, 4 * bt);
    defer gpa.destroy(a);
    _ = keep(t, a, &ids, &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    try pumpUntil(ld, 1);
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    r.sim.delay = 1000; // copies stay on the "device"
    const job = try t.vtable.begin(t.ptr, &p);
    var k: usize = 0;
    while (k < 1000 and ld.jobs.get(job).?.rx.copied == 0) : (k += 1) {
        try testing.expect(t.vtable.poll(t.ptr, job, std.time.ns_per_ms) == .pending);
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try testing.expect(ld.jobs.get(job).?.rx.inflight > 0);
    const waits = r.sim.waits;
    t.vtable.cancel(t.ptr, job);
    try testing.expect(r.sim.waits > waits);
    try testing.expectEqual(@as(u32, 1), fam.discarded);
    try testing.expect(r.sim.cleared(8 * bt, 12 * bt));
    try stagingDrains(ld.side.store);
    try testing.expect(ld.side.copies.items.len == 0);
    // the point is still there (a cancel is not damage)
    try testing.expect(t.vtable.lookup(t.ptr, &want(&ids)) != null);
}

test "one rank: the shutdown flush stages and publishes every queued write" {
    const r = try Rank.init(0, "flush", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0, agreement()), fam.family(), null, null, r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [2][4 * bt]u32 = undefined;
    var snaps: [2]*Snap = undefined;
    for (0..2) |i| {
        testIds(&ids[i], @intCast(10 + i));
        snaps[i] = try keptSnap(r.sim, i + 1, i * 4 * bt, 4 * bt);
        _ = keep(t, snaps[i], &ids[i], &.{.{ .name = i + 1, .at = 0 }}) orelse return error.NotKept;
    }
    defer for (snaps) |s| gpa.destroy(s);
    r.sim.delay = 3;
    try testing.expectEqual(@as(u64, 0), t.vtable.flush(t.ptr, 5 * std.time.ns_per_s));
    try testing.expectEqual(@as(u64, 2), ld.side.store.counts().committed);
    try testing.expect(!t.vtable.busy.?(t.ptr));
}

test "a keep crosses rank_wire with its lineage and origin intact; a lineage past one keep's forks is refused" {
    const forks = [_]format.Fork{ .{ .name = @splat(4), .at = 32 }, .{ .name = @splat(5), .at = 48 } };
    const lin: format.Lineage = .{ .root = @splat(1), .part = 2, .forks = &forks };
    var wf: [4]rank_wire.Fork = undefined;
    const m = try tier.keepMsg(3, 77, 4096, 64, @splat(9), 5, true, lin, .{ .key = @splat(8), .at = 32 }, &wf, .{});
    var buf: [rank_wire.max_frame]u8 = undefined;
    const bytes = try rank_wire.encode(&buf, .{ .op = .keep, .from = 0, .epoch = 1, .txn = 1, .seq = 2 }, .{ .keep = m });
    var back: [4]rank_wire.Fork = undefined;
    const f = try rank_wire.decodeInto(bytes, .{ .forks = &back });
    const k = f.msg.keep;
    try testing.expectEqual(@as(u64, 77), k.kept);
    try testing.expectEqual(@as(u32, 3), k.slot);
    var out: [4]format.Fork = undefined;
    try testing.expect((try tier.keepLineage(k, &out)).eql(lin));
    try testing.expectEqual(@as(u32, 32), tier.keepOrigin(k).?.at);
    var one: [1]format.Fork = undefined;
    try testing.expectError(error.TooManyForks, tier.keepLineage(k, &one));
    var wf1: [1]rank_wire.Fork = undefined;
    try testing.expectError(error.TooManyForks, tier.keepMsg(3, 77, 4096, 64, @splat(9), 5, true, lin, null, &wf1, .{}));
    const none = try tier.keepMsg(3, 77, 4096, 64, @splat(9), 5, false, .{ .root = @splat(1) }, null, &wf, .{});
    try testing.expect(none.part == null and none.origin == null and none.forks.len == 0);
}

test "follower settings: flags over the environment, off without a directory or a cap" {
    const Env = struct {
        fn flagNone(_: []const u8) ?[]const u8 {
            return null;
        }
        fn flagSome(n: []const u8) ?[]const u8 {
            if (std.mem.eql(u8, n, "--spill-gib")) return "8";
            return null;
        }
        fn env(n: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, n, "TF_DS_SPILL_DIR")) return "/nvme/spill";
            if (std.mem.eql(u8, n, "TF_DS_SPILL_GIB")) return "2";
            if (std.mem.eql(u8, n, "TF_DS_SPILL_MIN_TOKENS")) return "4096";
            return null;
        }
        fn none(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn bad(n: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, n, "TF_DS_SPILL_DIR")) return "/x";
            if (std.mem.eql(u8, n, "TF_DS_SPILL_GIB")) return "lots";
            return null;
        }
    };
    const s = (try tier.Settings.follower(Env.flagSome, Env.env)).?;
    try testing.expectEqualStrings("/nvme/spill", s.dir);
    try testing.expectEqual(@as(f64, 8), s.gib);
    try testing.expectEqual(@as(u32, 4096), s.min_tokens);
    try testing.expect(try tier.Settings.follower(Env.flagNone, Env.none) == null);
    try testing.expectError(error.BadSpillSetting, tier.Settings.follower(Env.flagNone, Env.bad));
}

// -- two ranks over a loopback link ---------------------------------------------------------------------------------

/// Test frames beside the protocol's: [kind u8][payload].
const T = struct {
    const snap: u8 = 20; // id u64, slot u32, base u64, at u32: the follower keeps a state (its own bytes there)
    const ids: u8 = 21; // slot u32, then u32 ids (i32 bits): the slot's sequence as this rank holds it
    const query: u8 = 24; // key [32]: the follower answers [present u8][restores open u8][committed u32][aborted u32][skipped u32]
    const verify: u8 = 25; // src u64, dst u64, at u32, src snap u64, dst snap u64: the follower answers [ok u8]
    const corrupt: u8 = 23;
    const cleared: u8 = 26; // lo u64, hi u64: the follower answers [its rows there are fresh-pool sentinels u8]
    const fail_reserve: u8 = 27; // on u8: the follower's family refuses the next reserves
    const clobber: u8 = 28; // lo u64, hi u64: the follower keeps a copy of its rows there, bumps them and overwrites them
    const saved: u8 = 29; // dst u64: the follower answers [its rows at dst equal the copy clobber kept u8]
    const forget: u8 = 30; // key [32]: the follower's own store drops the point (as its cap or floor would)
    const hold: u8 = 31; // on u8: the follower's device holds every copy (a slow rank: its events never complete) or lets them go
    const done: u8 = 0xff;
};

/// The follower's family over the test's kept states.
const FamF = struct {
    sim: *Sim,
    kept: std.AutoHashMapUnmanaged(u64, u64) = .empty, // id -> buf
    loading: std.AutoHashMapUnmanaged(u64, u64) = .empty,
    ids: [4][]i32 = @splat(&.{}),
    committed: u32 = 0,
    aborted: u32 = 0,
    refuse: bool = false,

    fn family(f: *FamF) tier.FollowerFamily {
        return .{ .ptr = f, .reserve = reserve, .install = install, .discard = discard, .buf = buf, .ids = idsOf };
    }
    fn of(p: *anyopaque) *FamF {
        return @ptrCast(@alignCast(p));
    }
    fn reserve(p: *anyopaque, id: u64) anyerror!u64 {
        const f = of(p);
        if (f.refuse) return error.NoSnapRoom;
        const b = try f.sim.allocSnap();
        try f.loading.put(gpa, id, b);
        return b;
    }
    fn install(p: *anyopaque, id: u64) anyerror!void {
        const f = of(p);
        const kv = f.loading.fetchRemove(id) orelse return error.UnknownSnapshot;
        try f.kept.put(gpa, id, kv.value);
        f.committed += 1;
    }
    fn discard(p: *anyopaque, id: u64) void {
        const f = of(p);
        const kv = f.loading.fetchRemove(id) orelse return;
        f.sim.freeSnap(kv.value);
        f.aborted += 1;
    }
    fn buf(p: *anyopaque, id: u64) ?u64 {
        return of(p).kept.get(id);
    }
    fn idsOf(p: *anyopaque, slot: u32) []const i32 {
        return of(p).ids[slot];
    }
    fn deinit(f: *FamF) void {
        f.kept.deinit(gpa);
        f.loading.deinit(gpa);
        for (f.ids) |x| gpa.free(x);
    }
};

const Peer = struct {
    port: u16,
    err: ?anyerror = null,
    rank: *Rank = undefined,
    id: u8 = 1, // PETABRIDGE (k2a-pb1) spill P6: this follower's rank (world 3: 1 or 2)

    fn run(p: *Peer) void {
        p.loop() catch |e| {
            p.err = e;
        };
    }

    fn loop(p: *Peer) !void {
        var l = try link.Link.connect(io, .{ 127, 0, 0, 1 }, p.port, 10);
        defer l.close();
        var fam: FamF = .{ .sim = p.rank.sim };
        defer fam.deinit();
        const f = try F.create(gpa, io, p.id, testFx(p.id), fam.family(), try p.rank.opening());
        defer f.destroy();
        const buf = try gpa.alloc(u8, rank_wire.max_frame + 64);
        defer gpa.free(buf);
        var keep_rows: std.ArrayList(u8) = .empty; // T.clobber's copy, every block plane's rows in order
        defer keep_rows.deinit(gpa);
        var keep_n: usize = 0;
        while (true) {
            if (!try l.readable(2 * std.time.ns_per_ms)) {
                f.service(std.time.ns_per_ms);
                continue;
            }
            const bytes = try l.recv(buf);
            if (bytes.len == 0) return error.EmptyFrame;
            f.service(std.time.ns_per_ms);
            const body = bytes[1..];
            var r: Reader = .{ .b = body };
            switch (bytes[0]) {
                rank_wire.lanes_kind => try f.wireFrame(l, body),
                T.snap => {
                    const id = try r.int(u64);
                    _ = try r.int(u32);
                    _ = try r.int(u64);
                    _ = try r.int(u32);
                    try fam.kept.put(gpa, id, try fam.sim.allocSnap());
                },
                T.ids => {
                    const slot = try r.int(u32);
                    const n = (body.len - 4) / 4;
                    gpa.free(fam.ids[slot]);
                    fam.ids[slot] = try gpa.alloc(i32, n);
                    for (fam.ids[slot]) |*x| x.* = @bitCast(try r.int(u32)); // (pictures' positions negative)
                },
                T.query => {
                    var key: format.Digest = undefined;
                    @memcpy(&key, body[0..32]);
                    var out: [16]u8 = undefined;
                    out[0] = @intFromBool(if (f.side) |s| s.store.presence(key) else false);
                    out[1] = @intCast(f.restores.count());
                    std.mem.writeInt(u32, out[2..6], fam.committed, .little);
                    std.mem.writeInt(u32, out[6..10], fam.aborted, .little);
                    std.mem.writeInt(u32, out[10..14], @intCast(f.skipped), .little);
                    try l.send(out[0..14]);
                },
                T.verify => {
                    const src = try r.int(u64);
                    const dst = try r.int(u64);
                    const at = try r.int(u32);
                    const sid = try r.int(u64);
                    const did = try r.int(u64);
                    const sim = fam.sim;
                    const ok = sim.sameRows(src, dst, at) and std.mem.eql(u8, sim.bytes(fam.kept.get(sid).?, sim.layout.small_bytes), sim.bytes(fam.kept.get(did).?, sim.layout.small_bytes));
                    try l.send(&.{@intFromBool(ok)});
                },
                T.corrupt => try p.rank.corrupt(),
                T.cleared => {
                    const lo = try r.int(u64);
                    const hi = try r.int(u64);
                    try l.send(&.{@intFromBool(fam.sim.cleared(lo, hi))});
                },
                T.fail_reserve => fam.refuse = (try r.int(u8)) != 0,
                T.clobber => {
                    const lo = try r.int(u64);
                    const hi = try r.int(u64);
                    keep_rows.clearRetainingCapacity();
                    for (0..fam.sim.layout.n_block) |i| try keep_rows.appendSlice(gpa, fam.sim.rows(i, lo, hi));
                    keep_n = hi - lo;
                    f.bump(lo, hi);
                    fam.sim.clobber(lo, hi);
                },
                T.saved => {
                    const dst = try r.int(u64);
                    var off: usize = 0;
                    var ok = true;
                    for (0..fam.sim.layout.n_block) |i| {
                        const rows = fam.sim.rows(i, dst, dst + keep_n);
                        ok = ok and std.mem.eql(u8, rows, keep_rows.items[off..][0..rows.len]);
                        off += rows.len;
                    }
                    try l.send(&.{@intFromBool(ok)});
                },
                T.hold => {
                    if ((try r.int(u8)) != 0) {
                        fam.sim.delay = std.math.maxInt(u32);
                    } else {
                        fam.sim.delay = 0;
                        @memset(&fam.sim.pending, 0);
                    }
                },
                T.forget => {
                    var key: format.Digest = undefined;
                    @memcpy(&key, body[0..32]);
                    if (f.side) |sd| sd.store.forget(key, &.{}, &.{});
                },
                T.done => {
                    _ = f.flush(5 * std.time.ns_per_s);
                    return;
                },
                else => return error.BadFrame,
            }
        }
    }
};

const Reader = struct {
    b: []const u8,
    at: usize = 0,
    fn int(r: *Reader, comptime U: type) !U {
        if (r.at + @sizeOf(U) > r.b.len) return error.Short;
        defer r.at += @sizeOf(U);
        return std.mem.readInt(U, r.b[r.at..][0..@sizeOf(U)], .little);
    }
};

/// Rank 0's end of the link: lanes' frames as [kind][payload].
const Rank0Wire = struct {
    l: link.Link,
    buf: std.ArrayList(u8) = .empty,

    fn frame(ctx: *anyopaque, kind: u8, sub: []const u8) anyerror!void {
        const w: *Rank0Wire = @ptrCast(@alignCast(ctx));
        w.buf.clearRetainingCapacity();
        try w.buf.append(gpa, kind);
        try w.buf.appendSlice(gpa, sub);
        try w.l.send(w.buf.items);
    }

    fn send(w: *Rank0Wire, kind: u8, parts: []const []const u8) !void {
        w.buf.clearRetainingCapacity();
        try w.buf.append(gpa, kind);
        for (parts) |x| try w.buf.appendSlice(gpa, x);
        try w.l.send(w.buf.items);
    }

    /// A one-byte answer to a test frame (T.cleared, T.saved, T.verify).
    fn yes(w: *Rank0Wire, kind: u8, parts: []const []const u8) !bool {
        try w.send(kind, parts);
        var b: [8]u8 = undefined;
        const got = try w.l.recvWithin(&b, 5 * std.time.ns_per_s);
        if (got.len != 1) return error.BadReply;
        return got[0] != 0;
    }

    fn query(w: *Rank0Wire, key: format.Digest) ![14]u8 {
        try w.send(T.query, &.{&key});
        var b: [64]u8 = undefined;
        const got = try w.l.recvWithin(&b, 5 * std.time.ns_per_s);
        if (got.len != 14) return error.BadReply;
        return got[0..14].*;
    }

    /// The follower holds `key` complete (its writes run while it idles).
    fn waitPresent(w: *Rank0Wire, key: format.Digest, present: bool) !void {
        var k: usize = 0;
        while (k < 2000) : (k += 1) {
            if (((try w.query(key))[0] != 0) == present) return;
            std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
        }
        return error.TestTimedOut;
    }
};

fn le(comptime U: type, v: U) [@sizeOf(U)]u8 {
    var b: [@sizeOf(U)]u8 = undefined;
    std.mem.writeInt(U, &b, v, .little);
    return b;
}

fn testPort() u16 {
    var b: [2]u8 = undefined;
    io.random(&b);
    return 30000 + std.mem.readInt(u16, &b, .little) % 20000;
}

/// A kept state on both ranks (the follower's own bytes at the same base, the same prompt in its slot), then keep.
fn keepBoth(w: *Rank0Wire, t: pc.Tier, sim: *Sim, id: u64, base: usize, ids: []const u32, follower_ids: []const u32) !*Snap {
    return keepBothIn(w, t, sim, id, base, ids, follower_ids, null);
}

/// keepBoth with the kept state's part (owner_sharded: the planner's salt).
fn keepBothIn(w: *Rank0Wire, t: pc.Tier, sim: *Sim, id: u64, base: usize, ids: []const u32, follower_ids: []const u32, part: ?u8) !*Snap {
    const sn = try keptSnap(sim, id, base, ids.len);
    sn.part = part;
    errdefer gpa.destroy(sn);
    try w.send(T.snap, &.{ &le(u64, id), &le(u32, 0), &le(u64, base), &le(u32, @intCast(ids.len)) });
    var raw: std.ArrayList(u8) = .empty;
    defer raw.deinit(gpa);
    try raw.appendSlice(gpa, &le(u32, 0));
    for (follower_ids) |x| try raw.appendSlice(gpa, &le(u32, x));
    try w.send(T.ids, &.{raw.items});
    _ = keep(t, sn, ids, &.{.{ .name = id, .at = 0 }}) orelse return error.NotKept;
    return sn;
}

/// Both ranks over loopback: the follower thread, rank 0's end of the link and the startup agreement. `deinit` ends
/// the follower (done) and joins it before anything it uses is freed, also when an assertion failed on the way.
const Pair = struct {
    r0: *Rank,
    r1: *Rank,
    peer: *Peer,
    th: ?std.Thread = null,
    l: link.Link = undefined,
    listening: bool = false,
    w: Rank0Wire = undefined,
    links: [1]?link.Link = .{null},
    lead: ?*rank_wire.Leader = null, // until the tier takes it
    a: fingerprint.Agreement = undefined,

    /// How the follower's store scans at the agreement (tests of rank 0's awaitFollowers), and rank 0's hello bound.
    const Scan = struct { strays: usize = 0, scan_ns: u64 = 0, stall_ns: u64 = 0, hello_ns: ?u64 = null };

    fn start(tag: []const u8, staging_bytes: u64, epoch: u64) !*Pair {
        return startWith(tag, staging_bytes, epoch, try testLayout());
    }

    fn startWith(tag: []const u8, staging_bytes: u64, epoch: u64, layout: state.StateLayout) !*Pair {
        return startScan(tag, staging_bytes, epoch, layout, .{});
    }

    fn startScan(tag: []const u8, staging_bytes: u64, epoch: u64, layout: state.StateLayout, scan: Scan) !*Pair {
        const p = try gpa.create(Pair);
        p.* = .{ .r0 = undefined, .r1 = undefined, .peer = undefined };
        p.r0 = Rank.initWith(0, tag, staging_bytes, layout) catch |e| {
            gpa.destroy(p);
            return e;
        };
        p.r1 = Rank.initWith(1, tag, staging_bytes, layout) catch |e| {
            p.r0.deinit();
            gpa.destroy(p);
            return e;
        };
        p.peer = gpa.create(Peer) catch |e| {
            p.r1.deinit();
            p.r0.deinit();
            gpa.destroy(p);
            return e;
        };
        const port = testPort();
        p.peer.* = .{ .port = port, .rank = p.r1 };
        errdefer p.deinit();
        if (scan.strays > 0) try p.r1.strays(1, scan.strays);
        p.r1.scan_ns = scan.scan_ns;
        p.r1.stall_ns = scan.stall_ns;
        p.th = try std.Thread.spawn(.{}, Peer.run, .{p.peer});
        p.l = try link.Link.listen(.{ 127, 0, 0, 1 }, port);
        p.listening = true;
        p.w = .{ .l = p.l };
        p.links = .{p.l};
        const lead = try gpa.create(rank_wire.Leader);
        p.lead = lead;
        lead.* = rank_wire.Leader.init(2, epoch, testFx(0));
        if (scan.hello_ns) |h| lead.timeouts.hello_ns = h;
        p.a = try lead.agree(.{ .ctx = &p.w, .send = struct {
            fn f(ctx: *anyopaque, sub: []const u8) anyerror!void {
                return Rank0Wire.frame(ctx, rank_wire.lanes_kind, sub);
            }
        }.f }, &p.links);
        return p;
    }

    fn wire(p: *Pair) tier.Wire {
        return .{ .ctx = &p.w, .frame = Rank0Wire.frame, .links = &p.links };
    }

    /// The tier over rank 0's side, owning the Leader from here.
    fn leader(p: *Pair, fam: *Fam0, boot: u8) !*L {
        const ld = try L.create(gpa, try p.r0.side(0, p.a), fam.family(), p.wire(), p.lead, p.r0.settings, @splat(boot));
        p.lead = null;
        errdefer ld.destroy();
        try ld.awaitFollowers(null, std.time.ns_per_ms); // (native.openSpill's order)
        return ld;
    }

    /// The follower ended cleanly (done) and reported no error.
    fn finish(p: *Pair) !void {
        const th = p.th orelse return;
        try p.w.send(T.done, &.{});
        th.join();
        p.th = null;
        if (p.peer.err) |e| return e;
    }

    fn deinit(p: *Pair) void {
        if (p.th) |th| {
            // (an assertion failed with the follower still running: it ends at done, or when its link fails)
            if (p.listening) p.w.send(T.done, &.{}) catch {};
            th.join();
        }
        if (p.listening) {
            p.l.close();
            p.w.buf.deinit(gpa);
        }
        if (p.lead) |l| gpa.destroy(l);
        gpa.destroy(p.peer);
        p.r1.deinit();
        p.r0.deinit();
        gpa.destroy(p);
    }
};

test "two ranks: written on both, restored on both after every rank says present; damage on either rank aborts everywhere" {
    const pr = try Pair.start("pair", 0, 77);
    defer pr.deinit();
    const r0 = pr.r0;
    const w = &pr.w;
    var fam: Fam0 = .{ .sim = r0.sim };
    defer fam.deinit();
    const ld = try pr.leader(&fam, 4);
    var ld_alive = true;
    defer if (ld_alive) ld.destroy();
    const t = ld.tier();

    // 1. a point written on both ranks, then restored on both: rows and small state equal their sources
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 21);
    const sn = try keepBoth(w, t, r0.sim, 1, 0, &ids, &ids);
    defer gpa.destroy(sn);
    try pumpUntil(ld, 1);
    const key = format.pointKey(&ids, bt, &.{});
    try w.waitPresent(key, true);
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    const done = pollUntilDone(t, try t.vtable.begin(t.ptr, &p));
    try testing.expect(done == .ready);
    const got: *Snap = @ptrCast(@alignCast(done.ready.saved));
    defer r0.sim.freeSnap(got.buf);
    try testing.expect(r0.sim.sameRows(0, got.base, 4 * bt));
    try testing.expectEqual(@as(usize, ids.len), fam.last_room); // the waiting prompt's length, the room hint
    try testing.expect(try w.yes(T.verify, &.{ &le(u64, 0), &le(u64, got.base), &le(u32, 4 * bt), &le(u64, 1), &le(u64, got.id) }));
    const q1 = try w.query(key);
    try testing.expectEqual(@as(u8, 0), q1[1]); // no restore left open on the follower
    try testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, q1[2..6], .little)); // committed there too
    // every rank's counters: rank 1's ride its stats reply
    ld.stats_at = 0;
    t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
    var counts: [pc.max_ranks]pc.TierCounts = undefined;
    try testing.expectEqual(@as(usize, 2), t.vtable.counts(t.ptr, &counts));
    try testing.expectEqual(@as(u64, 1), counts[1].writes_committed);
    try testing.expectEqual(@as(u64, 1), counts[1].reads);

    // 2. damage on rank 1: its validation fails, rank 0 aborts with forget, the point goes on both ranks, both
    // destinations cleared to their sentinels
    var ids2: [3 * bt]u32 = undefined;
    testIds(&ids2, 22);
    const sn2 = try keepBoth(w, t, r0.sim, 2, 4 * bt, &ids2, &ids2);
    defer gpa.destroy(sn2);
    try pumpUntil(ld, 2);
    const key2 = format.pointKey(&ids2, bt, &.{});
    try w.waitPresent(key2, true);
    try w.send(T.corrupt, &.{});
    fam.next_base = 12 * bt;
    const p2 = t.vtable.lookup(t.ptr, &want(&ids2)) orelse return error.NotFound;
    const done2 = pollUntilDone(t, try t.vtable.begin(t.ptr, &p2));
    try testing.expectEqual(error.SpillDamaged, done2.failed);
    try testing.expect(!ld.side.store.presence(key2));
    try w.waitPresent(key2, false);
    try testing.expect(r0.sim.cleared(12 * bt, 15 * bt));
    try testing.expect(try w.yes(T.cleared, &.{ &le(u64, 12 * bt), &le(u64, 15 * bt) }));
    // (rank 1's corruption also damaged its copy of the first point: forgetting widens to its aliases only there)

    // 3. a point only rank 0 holds (the follower's prompt differs: it never wrote it): presence refused, and since
    // the follower answered missing (not still writing), the point is dropped on rank 0 too
    var ids3: [2 * bt]u32 = undefined;
    testIds(&ids3, 23);
    var wrong = ids3;
    wrong[1] +%= 1;
    const sn3 = try keepBoth(w, t, r0.sim, 3, 8 * bt, &ids3, &wrong);
    defer gpa.destroy(sn3);
    try pumpUntil(ld, 3);
    const key3 = format.pointKey(&ids3, bt, &.{});
    const q3 = try w.query(key3);
    try testing.expectEqual(@as(u8, 0), q3[0]);
    try testing.expect(std.mem.readInt(u32, q3[10..14], .little) >= 1); // skipped: its ids are not rank 0's
    const p3 = t.vtable.lookup(t.ptr, &want(&ids3)) orelse return error.NotFound;
    const reserved = fam.reserved;
    try testing.expectError(error.SpillNotOnEveryRank, t.vtable.begin(t.ptr, &p3));
    try testing.expectEqual(@as(u64, 1), ld.presence_refused);
    try testing.expectEqual(reserved, fam.reserved); // refused before any placement
    try testing.expect(!ld.side.store.presence(key3));
    try testing.expect(t.vtable.lookup(t.ptr, &want(&ids3)) == null); // never offered again
    try stagingDrains(ld.side.store);

    // 4. damage on rank 0, after the follower started: abort everywhere with forget, both destinations cleared
    var ids4: [2 * bt]u32 = undefined;
    testIds(&ids4, 24);
    const sn4 = try keepBoth(w, t, r0.sim, 4, 10 * bt, &ids4, &ids4);
    defer gpa.destroy(sn4);
    try pumpUntil(ld, 4);
    const key4 = format.pointKey(&ids4, bt, &.{});
    try w.waitPresent(key4, true);
    try r0.corrupt();
    fam.next_base = 14 * bt;
    const p4 = t.vtable.lookup(t.ptr, &want(&ids4)) orelse return error.NotFound;
    const done4 = pollUntilDone(t, try t.vtable.begin(t.ptr, &p4));
    try testing.expectEqual(error.SpillDamaged, done4.failed);
    try w.waitPresent(key4, false);
    const q4 = try w.query(key4);
    try testing.expectEqual(@as(u8, 0), q4[1]);
    try testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, q4[6..10], .little)); // both failed restores discarded
    try testing.expect(r0.sim.cleared(14 * bt, 16 * bt));
    try testing.expect(try w.yes(T.cleared, &.{ &le(u64, 14 * bt), &le(u64, 16 * bt) }));

    // 5. the shutdown flush on rank 0 (the follower drains its own at done)
    try testing.expectEqual(@as(u64, 0), t.vtable.flush(t.ptr, 5 * std.time.ns_per_s));
    ld.destroy();
    ld_alive = false;
    try pr.finish();
}

test "two ranks: a cancel mid-restore aborts the follower's transaction and releases its destination" {
    const pr = try Pair.start("pcancel", 0, 78);
    defer pr.deinit();
    const r0 = pr.r0;
    const w = &pr.w;
    var fam: Fam0 = .{ .sim = r0.sim };
    defer fam.deinit();
    const ld = try pr.leader(&fam, 5);
    var ld_alive = true;
    defer if (ld_alive) ld.destroy();
    const t = ld.tier();
    var ids: [6 * bt]u32 = undefined;
    testIds(&ids, 31);
    const sn = try keepBoth(w, t, r0.sim, 1, 0, &ids, &ids);
    defer gpa.destroy(sn);
    try pumpUntil(ld, 1);
    const key = format.pointKey(&ids, bt, &.{});
    try w.waitPresent(key, true);
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    r0.sim.delay = 1000;
    const job = try t.vtable.begin(t.ptr, &p);
    var k: usize = 0;
    while (k < 1000 and ld.jobs.get(job).?.rx.copied == 0) : (k += 1) {
        _ = t.vtable.poll(t.ptr, job, std.time.ns_per_ms);
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    t.vtable.cancel(t.ptr, job);
    try testing.expect(r0.sim.cleared(8 * bt, 14 * bt));
    const q = try w.query(key);
    try testing.expectEqual(@as(u8, 0), q[1]); // the follower's transaction ended
    try testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, q[6..10], .little)); // its destination discarded
    try testing.expectEqual(@as(u8, 1), q[0]); // the point stays (not damage)
    try testing.expect(try w.yes(T.cleared, &.{ &le(u64, 8 * bt), &le(u64, 14 * bt) }));
    ld.destroy();
    ld_alive = false;
    try pr.finish();
}

test "two ranks: a follower that cannot reserve fails validate (reserve), both ranks abort, the point stays" {
    const pr = try Pair.start("pres", 0, 79);
    defer pr.deinit();
    const r0 = pr.r0;
    const w = &pr.w;
    var fam: Fam0 = .{ .sim = r0.sim };
    defer fam.deinit();
    const ld = try pr.leader(&fam, 6);
    var ld_alive = true;
    defer if (ld_alive) ld.destroy();
    const t = ld.tier();
    var ids: [3 * bt]u32 = undefined;
    testIds(&ids, 41);
    const sn = try keepBoth(w, t, r0.sim, 1, 0, &ids, &ids);
    defer gpa.destroy(sn);
    try pumpUntil(ld, 1);
    const key = format.pointKey(&ids, bt, &.{});
    try w.waitPresent(key, true);
    try w.send(T.fail_reserve, &.{&[_]u8{1}});
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    const done = pollUntilDone(t, try t.vtable.begin(t.ptr, &p));
    try testing.expectEqual(error.SpillReserve, done.failed);
    try testing.expectEqual(@as(u32, 1), fam.discarded);
    try testing.expect(r0.sim.cleared(8 * bt, 11 * bt));
    const q = try w.query(key);
    try testing.expectEqual(@as(u8, 0), q[1]); // no transaction left open there
    try testing.expectEqual(@as(u8, 1), q[0]); // not damage: the point stays on both
    try testing.expect(ld.side.store.presence(key));
    try stagingDrains(ld.side.store);
    // with room again the same point restores on both
    try w.send(T.fail_reserve, &.{&[_]u8{0}});
    const again = pollUntilDone(t, try t.vtable.begin(t.ptr, &(t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound)));
    try testing.expect(again == .ready);
    const got: *Snap = @ptrCast(@alignCast(again.ready.saved));
    defer r0.sim.freeSnap(got.buf);
    try testing.expect(try w.yes(T.verify, &.{ &le(u64, 0), &le(u64, got.base), &le(u32, 3 * bt), &le(u64, 1), &le(u64, got.id) }));
    ld.destroy();
    ld_alive = false;
    try pr.finish();
}

test "two ranks: rank 0's guard stages the follower's copy before the rows change there; a dedup keep repairs a follower that lost the point" {
    const pr = try Pair.start("pguard", 4 << 20, 80);
    defer pr.deinit();
    const r0 = pr.r0;
    const w = &pr.w;
    var fam: Fam0 = .{ .sim = r0.sim };
    defer fam.deinit();
    const ld = try pr.leader(&fam, 7);
    var ld_alive = true;
    defer if (ld_alive) ld.destroy();
    const t = ld.tier();
    // 1. kept, then guarded at once (write_staged) and the rows overwritten on both ranks (generations bumped): each
    // rank's point publishes the bytes from before the change
    var ids: [8 * bt]u32 = undefined;
    testIds(&ids, 51);
    const before0 = try gpa.dupe(u8, r0.sim.rows(0, 0, 8 * bt));
    defer gpa.free(before0);
    const sn = try keepBoth(w, t, r0.sim, 1, 0, &ids, &ids);
    defer gpa.destroy(sn);
    ld.guardSnap(sn.id, 0);
    ld.bump(0, 8 * bt);
    r0.sim.clobber(0, 8 * bt);
    try w.send(T.clobber, &.{ &le(u64, 0), &le(u64, 8 * bt) });
    try pumpUntil(ld, 1);
    const key = format.pointKey(&ids, bt, &.{});
    try w.waitPresent(key, true);
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    const done = pollUntilDone(t, try t.vtable.begin(t.ptr, &p));
    try testing.expect(done == .ready);
    const got: *Snap = @ptrCast(@alignCast(done.ready.saved));
    defer r0.sim.freeSnap(got.buf);
    try testing.expectEqualSlices(u8, before0, r0.sim.rows(0, got.base, got.base + 8 * bt));
    try testing.expect(try w.yes(T.saved, &.{&le(u64, got.base)}));
    try stagingDrains(ld.side.store);

    // 2. the follower's store drops the point on its own (its cap, its floor): rank 0's next keep of the same prompt
    // dedups there, and its note makes the follower write the point again, so the pair can restore it
    try w.send(T.forget, &.{&key});
    try w.waitPresent(key, false);
    const sn2 = try keepBoth(w, t, r0.sim, 2, 0, &ids, &ids);
    defer gpa.destroy(sn2);
    try w.waitPresent(key, true);
    try testing.expectEqual(@as(u64, 1), ld.side.store.counts().committed); // rank 0 wrote nothing new
    ld.destroy();
    ld_alive = false;
    try pr.finish();
}

test "one rank: a restore past the cap is refused before anything is placed or read" {
    const r = try Rank.init(0, "toomany", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0, agreement()), fam.family(), null, null, r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    try testing.expectEqual(@as(u32, 2), ld.max_restores);
    var ids: [3][2 * bt]u32 = undefined;
    var snaps: [3]*Snap = undefined;
    for (0..3) |i| {
        testIds(&ids[i], @intCast(60 + i));
        snaps[i] = try keptSnap(r.sim, i + 1, i * 2 * bt, 2 * bt);
        _ = keep(t, snaps[i], &ids[i], &.{.{ .name = i + 1, .at = 0 }}) orelse return error.NotKept;
    }
    defer for (snaps) |s| gpa.destroy(s);
    try pumpUntil(ld, 3);
    r.sim.delay = 1000;
    var jobs: [2]u64 = undefined;
    for (0..2) |i| {
        fam.next_base = (8 + 4 * i) * bt;
        jobs[i] = try t.vtable.begin(t.ptr, &(t.vtable.lookup(t.ptr, &want(&ids[i])) orelse return error.NotFound));
    }
    const p3 = t.vtable.lookup(t.ptr, &want(&ids[2])) orelse return error.NotFound;
    try testing.expectError(error.TooManyRestores, t.vtable.begin(t.ptr, &p3));
    try testing.expectEqual(@as(u32, 2), fam.reserved); // no third destination placed
    for (jobs) |j| t.vtable.cancel(t.ptr, j);
    try stagingDrains(ld.side.store);
    try testing.expect(t.vtable.lookup(t.ptr, &want(&ids[2])) != null); // still there for later
}

test "owner-sharded and replicated stores never share a namespace; an owner-sharded side is refused (KV sharding is not ported)" {
    // PETABRIDGE (k2a-pb1) spill P6: the tip's first half (pure digests) as it was; its second half (and the two-rank
    // owner-sharded test) went: our Side.create refuses any owner_sharded layout before its store opens
    const a = agreement();
    const rep = try testLayout();
    const sh = try shardedLayout();
    try testing.expect(!std.mem.eql(u8, &tier.storeCompat(a, 0, &rep), &tier.storeCompat(a, 0, &sh)));
    try testing.expectEqual(format.OwnerLayout.owner_sharded, tier.storeLayout(&sh));
    try testing.expectEqual(rank_wire.OwnerLayout.replicated, tier.wireLayout(&rep));
    const r = try Rank.initWith(0, "sharded", 0, try shardedLayout());
    defer r.deinit();
    const gens = try state.Generations.init(gpa, cap, bt);
    var g2 = gens;
    try testing.expectError(error.SpillWithKvShard, L.S.create(gpa, io, .{ .sim = r.sim }, r.sim.layout, gens, r.settings, 0, r.compat(a, 0), cap, r.staging, r.events, .{ .free = bigFree }));
    g2.deinit(gpa);
}

test "two ranks: a slow follower never stalls rank 0's polls; validate is asked once, when every rank is complete" {
    const pr = try Pair.start("pslow", 0, 81);
    defer pr.deinit();
    const r0 = pr.r0;
    const w = &pr.w;
    var fam: Fam0 = .{ .sim = r0.sim };
    defer fam.deinit();
    const ld = try pr.leader(&fam, 8);
    var ld_alive = true;
    defer if (ld_alive) ld.destroy();
    const t = ld.tier();
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 71);
    const sn = try keepBoth(w, t, r0.sim, 1, 0, &ids, &ids);
    defer gpa.destroy(sn);
    try pumpUntil(ld, 1);
    const key = format.pointKey(&ids, bt, &.{});
    try w.waitPresent(key, true);
    // rank 1's device now holds every copy it enqueues: its restore cannot complete until released
    try w.send(T.hold, &.{&[_]u8{1}});
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    const job = try t.vtable.begin(t.ptr, &p);
    var polls: usize = 0;
    var longest: u64 = 0;
    const t_end = monotonic() + 300 * std.time.ns_per_ms;
    while (monotonic() < t_end) : (polls += 1) {
        const t0 = monotonic();
        const r = t.vtable.poll(t.ptr, job, std.time.ns_per_ms);
        longest = @max(longest, monotonic() - t0);
        try testing.expect(r == .pending);
        // a decode round's frames go on meanwhile (the ordered stream: a test frame the follower answers)
        _ = try w.query(key);
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
    try testing.expect(polls >= 20);
    try testing.expect(longest < 100 * std.time.ns_per_ms); // a status ask, never the follower's copy time
    try testing.expect(ld.status_asks >= 10);
    try testing.expectEqual(@as(u64, 0), ld.validate_asks);
    try testing.expectEqual(@as(u8, 1), (try w.query(key))[1]); // its restore is still open there
    // released: the follower completes, status says so, validate once, committed on both
    try w.send(T.hold, &.{&[_]u8{0}});
    const done = pollUntilDone(t, job);
    try testing.expect(done == .ready);
    const got: *Snap = @ptrCast(@alignCast(done.ready.saved));
    defer r0.sim.freeSnap(got.buf);
    try testing.expectEqual(@as(u64, 1), ld.validate_asks);
    try testing.expect(r0.sim.sameRows(0, got.base, 4 * bt));
    try testing.expect(try w.yes(T.verify, &.{ &le(u64, 0), &le(u64, got.base), &le(u32, 4 * bt), &le(u64, 1), &le(u64, got.id) }));
    const q = try w.query(key);
    try testing.expectEqual(@as(u8, 0), q[1]);
    try testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, q[2..6], .little));
    ld.destroy();
    ld_alive = false;
    try pr.finish();
}
// PETABRIDGE (k2a-pb1) spill P6: the tip's picture tests (0021) went with the pictures; the refusal test below
// ("PETABRIDGE refusals") checks that a picture point is never written.

test "two ranks: a follower still incomplete past status_wait gives the restore up as timeout, never validate; the point restores again" {
    const pr = try Pair.start("pwait", 0, 83);
    defer pr.deinit();
    const r0 = pr.r0;
    const w = &pr.w;
    var fam: Fam0 = .{ .sim = r0.sim };
    defer fam.deinit();
    pr.lead.?.timeouts.status_wait_ns = 50 * std.time.ns_per_ms; // (Timeouts: the tier reads its Leader's)
    const ld = try pr.leader(&fam, 9);
    var ld_alive = true;
    defer if (ld_alive) ld.destroy();
    const t = ld.tier();
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 73);
    const sn = try keepBoth(w, t, r0.sim, 1, 0, &ids, &ids);
    defer gpa.destroy(sn);
    try pumpUntil(ld, 1);
    const key = format.pointKey(&ids, bt, &.{});
    try w.waitPresent(key, true);
    // rank 1's copies never complete while held: past 50 ms of status polls, rank 0 gives the restore up itself
    try w.send(T.hold, &.{&[_]u8{1}});
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    const job = try t.vtable.begin(t.ptr, &p);
    const t0 = monotonic();
    const r = pollUntilDone(t, job);
    try testing.expect(r == .failed);
    try testing.expectEqual(@as(anyerror, error.SpillTimeout), r.failed);
    try testing.expect(monotonic() - t0 >= 50 * std.time.ns_per_ms);
    try testing.expect(ld.status_asks >= 2);
    try testing.expectEqual(@as(u64, 0), ld.validate_asks); // never asked: the restore was given up, not validated
    try testing.expectEqual(@as(?rank_wire.Broken, null), ld.lead.?.broken); // the pair goes on
    try testing.expectEqual(@as(?anyerror, null), ld.broken);
    // the follower's abort ran (its copies waited for before the destination went): nothing open there, the point
    // kept (a timeout is not damage)
    const q = try w.query(key);
    try testing.expectEqual(@as(u8, 1), q[0]);
    try testing.expectEqual(@as(u8, 0), q[1]);
    try testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, q[6..10], .little));
    // released: the same point restores on both ranks
    try w.send(T.hold, &.{&[_]u8{0}});
    const p2 = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    fam.next_base = 8 * bt;
    const job2 = try t.vtable.begin(t.ptr, &p2);
    const done = pollUntilDone(t, job2);
    try testing.expect(done == .ready);
    const got: *Snap = @ptrCast(@alignCast(done.ready.saved));
    defer r0.sim.freeSnap(got.buf);
    try testing.expectEqual(@as(u64, 1), ld.validate_asks);
    try testing.expect(r0.sim.sameRows(0, got.base, 4 * bt));
    try testing.expect(try w.yes(T.verify, &.{ &le(u64, 0), &le(u64, got.base), &le(u32, 4 * bt), &le(u64, 1), &le(u64, got.id) }));
    ld.destroy();
    ld_alive = false;
    try pr.finish();
}

test "two ranks: a follower that completes within the poll rank 0 got ready in commits in that same poll" {
    const pr = try Pair.start("pfirst", 0, 85);
    defer pr.deinit();
    const r0 = pr.r0;
    const w = &pr.w;
    var fam: Fam0 = .{ .sim = r0.sim };
    defer fam.deinit();
    const ld = try pr.leader(&fam, 10);
    var ld_alive = true;
    defer if (ld_alive) ld.destroy();
    const t = ld.tier();
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 75);
    const sn = try keepBoth(w, t, r0.sim, 1, 0, &ids, &ids);
    defer gpa.destroy(sn);
    try pumpUntil(ld, 1);
    const key = format.pointKey(&ids, bt, &.{});
    try w.waitPresent(key, true);
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    const job = try t.vtable.begin(t.ptr, &p);
    // a poll with the core's usual budget: the follower's copies lag rank 0's by its reads and a frame, and the
    // repeated status asks inside the budget see it complete without leaving the poll
    var r: pc.Poll = undefined;
    var k: usize = 0;
    while (k < 2000) : (k += 1) {
        r = t.vtable.poll(t.ptr, job, 40 * std.time.ns_per_ms);
        if (r != .pending) break;
        try testing.expectEqual(@as(u64, 0), ld.status_asks); // (pending only while rank 0's own copies run)
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try testing.expect(r == .ready);
    const got: *Snap = @ptrCast(@alignCast(r.ready.saved));
    defer r0.sim.freeSnap(got.buf);
    try testing.expectEqual(@as(u64, 1), ld.ready_first_poll);
    try testing.expectEqual(@as(u64, 1), ld.validate_asks);
    ld.destroy();
    ld_alive = false;
    try pr.finish();
}

test "two ranks: a follower store scan far longer than the hello bound opens; rank 0 waits while it progresses" {
    // 40 names at 25 ms each: a one-second scan against a 150 ms hello bound (a large store on the Sparks: minutes
    // against 120 s), which used to fail the start at rank 0's first ask after the agreement
    const hello_ns = 150 * std.time.ns_per_ms;
    const pr = try Pair.startScan("scanslow", 0, 81, try testLayout(), .{ .strays = 40, .scan_ns = 25 * std.time.ns_per_ms, .hello_ns = hello_ns });
    defer pr.deinit();
    const r0 = pr.r0;
    var fam: Fam0 = .{ .sim = r0.sim };
    defer fam.deinit();
    const t0 = monotonic();
    const ld = try pr.leader(&fam, 5); // (awaitFollowers: the hello bound without progress, not in all)
    defer ld.destroy();
    try testing.expect(monotonic() - t0 > 4 * hello_ns);
    // the follower's store is open and serves: a point written on both ranks restores on both
    const t = ld.tier();
    var ids: [2 * bt]u32 = undefined;
    testIds(&ids, 61);
    const sn = try keepBoth(&pr.w, t, r0.sim, 1, 0, &ids, &ids);
    defer gpa.destroy(sn);
    try pumpUntil(ld, 1);
    const key = format.pointKey(&ids, bt, &.{});
    try pr.w.waitPresent(key, true);
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    const done = pollUntilDone(t, try t.vtable.begin(t.ptr, &p));
    try testing.expect(done == .ready);
    const got: *Snap = @ptrCast(@alignCast(done.ready.saved));
    defer r0.sim.freeSnap(got.buf);
    try testing.expect(r0.sim.sameRows(0, got.base, 2 * bt));
    try pr.finish();
}

test "two ranks: a follower store scan that stops making progress fails the start, named" {
    const pr = try Pair.startScan("scanstall", 0, 82, try testLayout(), .{ .strays = 2, .stall_ns = 700 * std.time.ns_per_ms });
    defer pr.deinit();
    var fam: Fam0 = .{ .sim = pr.r0.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try pr.r0.side(0, pr.a), fam.family(), pr.wire(), pr.lead, pr.r0.settings, @splat(6));
    pr.lead = null;
    defer ld.destroy();
    try testing.expectError(error.FollowerScanStalled, ld.awaitFollowers(150 * std.time.ns_per_ms, std.time.ns_per_ms));
    // the follower's open still finishes (its scan was only slow): it ends cleanly at done
    try pr.finish();
}

fn monotonic() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

// -- PETABRIDGE (k2a-pb1) spill P6: the refusal rules, and world 3 (TP3: two followers over loopback) -----------------

test "PETABRIDGE refusals: a picture, an ineligible state (complete, decoded, off-grid, loading, stale), off the grid or short never reaches the disk" {
    const r = try Rank.init(0, "refuse", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0, agreement()), fam.family(), null, null, r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [4 * bt + 3]u32 = undefined;
    testIds(&ids, 91);
    // a picture before the cut (pb1 0006's content key: Kept.images): refused, counted, nothing queued
    const a = try keptSnap(r.sim, 1, 0, 4 * bt);
    defer gpa.destroy(a);
    try testing.expect(t.vtable.keep(t.ptr, &.{ .saved = a, .key = &ids, .at = 4 * bt, .bytes = 0, .images = true }) == null);
    try testing.expectEqual(@as(u64, 1), ld.pictures_refused);
    // the family's own verdict (Lanes.spillInfo: complete, decoded, off-grid, loading or stale): refused, counted
    a.eligible = false;
    try testing.expect(keep(t, a, &ids, &.{.{ .name = 1, .at = 0 }}) == null);
    try testing.expectEqual(@as(u64, 1), ld.ineligible_refused);
    a.eligible = true;
    // off the block grid, or a key shorter than the cut: refused before the family is asked
    const off = try keptSnap(r.sim, 2, 4 * bt, 2 * bt + 2);
    defer gpa.destroy(off);
    try testing.expect(keep(t, off, &ids, &.{.{ .name = 2, .at = 0 }}) == null);
    try testing.expect(t.vtable.keep(t.ptr, &.{ .saved = a, .key = ids[0 .. 4 * bt - 1], .at = 4 * bt, .bytes = 0 }) == null);
    try testing.expectEqual(@as(u64, 0), ld.side.store.counts().queued);
    // the same state, eligible and text only: written
    _ = keep(t, a, &ids, &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    try pumpUntil(ld, 1);
    // the pure rule, each case (Leader.refusal)
    const info: tier.SnapInfo = .{ .id = 1, .base = 0, .at = 4 * bt, .buf = 1, .slot = 0 };
    const kept: pc.Kept = .{ .saved = a, .key = &ids, .at = 4 * bt, .bytes = 0 };
    try testing.expectEqual(@as(?L.Refusal, null), L.refusal(bt, bt, &kept, info));
    var k2 = kept;
    k2.images = true;
    try testing.expectEqual(@as(?L.Refusal, .picture), L.refusal(bt, bt, &k2, info));
    k2 = kept;
    k2.at = 4 * bt - 2;
    try testing.expectEqual(@as(?L.Refusal, .off_grid), L.refusal(bt, bt, &k2, info));
    try testing.expectEqual(@as(?L.Refusal, .short), L.refusal(bt, 8 * bt, &kept, info));
    var info2 = info;
    info2.eligible = false;
    try testing.expectEqual(@as(?L.Refusal, .ineligible), L.refusal(bt, bt, &kept, info2));
    info2 = info;
    info2.at = 2 * bt;
    try testing.expectEqual(@as(?L.Refusal, .wrong_state), L.refusal(bt, bt, &kept, info2));
    // a follower's ids: text only (a picture's negative position is not rank 0's text point)
    var seq = [_]i32{ 5, 6, 7, 8 };
    var out: [4]u32 = undefined;
    try testing.expect(tier.followerIds(&seq, 4, &out));
    try testing.expectEqualSlices(u32, &.{ 5, 6, 7, 8 }, &out);
    seq[2] = -1;
    try testing.expect(!tier.followerIds(&seq, 4, &out));
    try testing.expect(!tier.followerIds(seq[0..3], 4, &out));
    // a keep notice never carries pictures from this port
    var wf: [1]rank_wire.Fork = undefined;
    const media = [_]format.Media{.{ .start = 1, .len = 2, .digest = @splat(1) }};
    try testing.expectError(error.TooManyPictures, tier.keepMsg(0, 1, 0, 2 * bt, @splat(9), 0, false, .{ .root = @splat(1) }, null, &wf, .{ .media = &media, .placeholder = 3 }));
}

/// Rank 0 and two followers over loopback (TP3): the follower threads (ranks 1 and 2, each its own port, store,
/// device and sequence), rank 0's links and its Leader agreed with both on world 3. `ws[i]`: the test frames to
/// follower i + 1 alone (its own answers on its own link); rank 0's protocol frames go to both (frame).
const Trio = struct {
    r: [3]*Rank,
    peers: [2]*Peer,
    th: [2]?std.Thread = .{ null, null },
    links: [2]?link.Link = .{ null, null },
    ws: [2]Rank0Wire = undefined,
    fan: std.ArrayList(u8) = .empty,
    lead: ?*rank_wire.Leader = null,
    a: fingerprint.Agreement = undefined,

    fn start(tag: []const u8, epoch: u64) !*Trio {
        const p = try gpa.create(Trio);
        p.* = .{ .r = undefined, .peers = undefined };
        var made: usize = 0;
        errdefer {
            for (p.r[0..made]) |x| x.deinit();
            gpa.destroy(p);
        }
        for (0..3) |i| {
            p.r[i] = try Rank.init(@intCast(i), tag, 0);
            made += 1;
        }
        var peers_made: usize = 0;
        errdefer for (p.peers[0..peers_made]) |x| gpa.destroy(x);
        for (0..2) |i| {
            p.peers[i] = try gpa.create(Peer);
            p.peers[i].* = .{ .port = testPort(), .rank = p.r[i + 1], .id = @intCast(i + 1) };
            peers_made += 1;
        }
        for (&p.ws) |*w| w.* = .{ .l = undefined };
        errdefer p.stop();
        for (0..2) |i| p.th[i] = try std.Thread.spawn(.{}, Peer.run, .{p.peers[i]});
        for (0..2) |i| {
            p.links[i] = try link.Link.listen(.{ 127, 0, 0, 1 }, p.peers[i].port);
            p.ws[i].l = p.links[i].?;
        }
        const lead = try gpa.create(rank_wire.Leader);
        p.lead = lead;
        lead.* = rank_wire.Leader.init(3, epoch, testFx(0));
        p.a = try lead.agree(.{ .ctx = p, .send = fanAgree }, &p.links);
        return p;
    }

    fn fanAgree(ctx: *anyopaque, sub: []const u8) anyerror!void {
        return frame(ctx, rank_wire.lanes_kind, sub);
    }

    /// A lanes frame of `kind` to every follower, in the ordered stream (Lanes.flush's fan-out).
    fn frame(ctx: *anyopaque, kind: u8, sub: []const u8) anyerror!void {
        const p: *Trio = @ptrCast(@alignCast(ctx));
        p.fan.clearRetainingCapacity();
        try p.fan.append(gpa, kind);
        try p.fan.appendSlice(gpa, sub);
        for (p.links) |l| try l.?.send(p.fan.items);
    }

    fn wire(p: *Trio) tier.Wire {
        return .{ .ctx = p, .frame = frame, .links = &p.links };
    }

    fn leader(p: *Trio, fam: *Fam0, boot: u8) !*L {
        const ld = try L.create(gpa, try p.r[0].side(0, p.a), fam.family(), p.wire(), p.lead, p.r[0].settings, @splat(boot));
        p.lead = null;
        errdefer ld.destroy();
        try ld.awaitFollowers(null, std.time.ns_per_ms);
        return ld;
    }

    /// A kept state on every rank (each follower's own bytes at the same base, `fids[i]` in its slot 0), then keep.
    fn keepAll(p: *Trio, t: pc.Tier, id: u64, base: usize, ids: []const u32, fids: [2][]const u32) !*Snap {
        const sn = try keptSnap(p.r[0].sim, id, base, ids.len);
        errdefer gpa.destroy(sn);
        for (&p.ws, fids) |*w, f| {
            try w.send(T.snap, &.{ &le(u64, id), &le(u32, 0), &le(u64, base), &le(u32, @intCast(ids.len)) });
            var raw: std.ArrayList(u8) = .empty;
            defer raw.deinit(gpa);
            try raw.appendSlice(gpa, &le(u32, 0));
            for (f) |x| try raw.appendSlice(gpa, &le(u32, x));
            try w.send(T.ids, &.{raw.items});
        }
        _ = keep(t, sn, ids, &.{.{ .name = id, .at = 0 }}) orelse return error.NotKept;
        return sn;
    }

    fn waitAll(p: *Trio, key: format.Digest, present: bool) !void {
        for (&p.ws) |*w| try w.waitPresent(key, present);
    }

    /// Both followers ended cleanly (done) with no error.
    fn finish(p: *Trio) !void {
        for (&p.ws, 0..) |*w, i| if (p.th[i] != null) try w.send(T.done, &.{});
        for (&p.th) |*th| if (th.*) |x| {
            x.join();
            th.* = null;
        };
        for (p.peers) |x| if (x.err) |e| return e;
    }

    fn stop(p: *Trio) void {
        for (0..2) |i| if (p.th[i]) |x| {
            if (p.links[i]) |l| l.send(&[_]u8{T.done}) catch {};
            x.join();
            p.th[i] = null;
        };
        for (&p.links) |*l| if (l.*) |*x| {
            x.close();
            l.* = null;
        };
    }

    fn deinit(p: *Trio) void {
        p.stop();
        for (&p.ws) |*w| w.buf.deinit(gpa);
        p.fan.deinit(gpa);
        if (p.lead) |l| gpa.destroy(l);
        for (p.peers) |x| gpa.destroy(x);
        for (p.r) |x| x.deinit();
        gpa.destroy(p);
    }
};

test "PETABRIDGE TP3: written on all three ranks, restored on all three; rank 2 missing, rank 2 out of room, rank 1 damaged or rank 2's ids differing aborts on every rank" {
    const tr = try Trio.start("trio", 93);
    defer tr.deinit();
    try testing.expectEqual(@as(u16, 3), tr.a.world);
    const r0 = tr.r[0];
    var fam: Fam0 = .{ .sim = r0.sim };
    defer fam.deinit();
    const ld = try tr.leader(&fam, 4);
    var ld_alive = true;
    defer if (ld_alive) ld.destroy();
    const t = ld.tier();

    // 1. a point every rank writes (from the keep notice, each from its own ids), restored on every rank: rows and
    // small state equal their sources on rank 0 and on both followers
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 101);
    const sn = try tr.keepAll(t, 1, 0, &ids, .{ &ids, &ids });
    defer gpa.destroy(sn);
    try pumpUntil(ld, 1);
    const key = format.pointKey(&ids, bt, &.{});
    try tr.waitAll(key, true);
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    const done = pollUntilDone(t, try t.vtable.begin(t.ptr, &p));
    try testing.expect(done == .ready);
    const got: *Snap = @ptrCast(@alignCast(done.ready.saved));
    defer r0.sim.freeSnap(got.buf);
    try testing.expect(r0.sim.sameRows(0, got.base, 4 * bt));
    for (&tr.ws) |*w| {
        try testing.expect(try w.yes(T.verify, &.{ &le(u64, 0), &le(u64, got.base), &le(u32, 4 * bt), &le(u64, 1), &le(u64, got.id) }));
        const q = try w.query(key);
        try testing.expectEqual(@as(u8, 0), q[1]); // no restore left open
        try testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, q[2..6], .little)); // committed there
    }
    // every rank's counters (both followers' ride their stats replies)
    ld.stats_at = 0;
    t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
    var counts: [pc.max_ranks]pc.TierCounts = undefined;
    try testing.expectEqual(@as(usize, 3), t.vtable.counts(t.ptr, &counts));
    for (counts[1..3]) |c| {
        try testing.expectEqual(@as(u64, 1), c.writes_committed);
        try testing.expectEqual(@as(u64, 1), c.reads);
    }

    // 2. rank 2's store lost a point (its own cap or floor): the has fails there, presence is refused before any
    // placement and every rank forgets it (rank 1 and rank 0 too), never offered again
    var ids2: [3 * bt]u32 = undefined;
    testIds(&ids2, 102);
    const sn2 = try tr.keepAll(t, 2, 4 * bt, &ids2, .{ &ids2, &ids2 });
    defer gpa.destroy(sn2);
    try pumpUntil(ld, 2);
    const key2 = format.pointKey(&ids2, bt, &.{});
    try tr.waitAll(key2, true);
    try tr.ws[1].send(T.forget, &.{&key2});
    try tr.ws[1].waitPresent(key2, false);
    const reserved = fam.reserved;
    const p2 = t.vtable.lookup(t.ptr, &want(&ids2)) orelse return error.NotFound;
    try testing.expectError(error.SpillNotOnEveryRank, t.vtable.begin(t.ptr, &p2));
    try testing.expectEqual(reserved, fam.reserved);
    try testing.expect(!ld.side.store.presence(key2));
    try tr.ws[0].waitPresent(key2, false); // rank 1 forgot its copy with the abort
    try testing.expect(t.vtable.lookup(t.ptr, &want(&ids2)) == null);

    // 3. rank 2 cannot reserve: validate fails (reserve) on rank 2, the restore aborts on all three, the point stays
    var ids3: [2 * bt]u32 = undefined;
    testIds(&ids3, 103);
    const sn3 = try tr.keepAll(t, 3, 8 * bt, &ids3, .{ &ids3, &ids3 });
    defer gpa.destroy(sn3);
    try pumpUntil(ld, 3);
    const key3 = format.pointKey(&ids3, bt, &.{});
    try tr.waitAll(key3, true);
    try tr.ws[1].send(T.fail_reserve, &.{&[_]u8{1}});
    fam.next_base = 12 * bt;
    const p3 = t.vtable.lookup(t.ptr, &want(&ids3)) orelse return error.NotFound;
    const done3 = pollUntilDone(t, try t.vtable.begin(t.ptr, &p3));
    try testing.expectEqual(error.SpillReserve, done3.failed);
    try testing.expect(r0.sim.cleared(12 * bt, 14 * bt));
    try testing.expect(try tr.ws[0].yes(T.cleared, &.{ &le(u64, 12 * bt), &le(u64, 14 * bt) })); // rank 1's destination
    for (&tr.ws) |*w| {
        const q = try w.query(key3);
        try testing.expectEqual(@as(u8, 1), q[0]); // not damage: kept everywhere
        try testing.expectEqual(@as(u8, 0), q[1]);
    }
    try tr.ws[1].send(T.fail_reserve, &.{&[_]u8{0}});

    // 4. rank 1's copy of a fresh point damaged: its validate fails, every rank forgets the point, every destination
    // cleared (a fresh point: a read of the point just aborted above may still be in flight there, with good bytes)
    var ids4: [2 * bt]u32 = undefined;
    testIds(&ids4, 104);
    const sn4 = try tr.keepAll(t, 4, 14 * bt, &ids4, .{ &ids4, &ids4 });
    defer gpa.destroy(sn4);
    try pumpUntil(ld, 4);
    const key4 = format.pointKey(&ids4, bt, &.{});
    try tr.waitAll(key4, true);
    try tr.ws[0].send(T.corrupt, &.{});
    fam.next_base = 12 * bt;
    const p4 = t.vtable.lookup(t.ptr, &want(&ids4)) orelse return error.NotFound;
    const done4 = pollUntilDone(t, try t.vtable.begin(t.ptr, &p4));
    try testing.expect(done4 == .failed);
    try testing.expectEqual(error.SpillDamaged, done4.failed);
    try testing.expect(!ld.side.store.presence(key4));
    try tr.waitAll(key4, false);
    try testing.expect(r0.sim.cleared(12 * bt, 14 * bt));
    for (&tr.ws) |*w| try testing.expect(try w.yes(T.cleared, &.{ &le(u64, 12 * bt), &le(u64, 14 * bt) }));
    try testing.expect(t.vtable.lookup(t.ptr, &want(&ids4)) == null);

    // 5. a point only rank 1's ids match: rank 2 skipped its keep (its prompt differs), presence refused and forgotten
    var ids5: [2 * bt]u32 = undefined;
    testIds(&ids5, 105);
    var wrong = ids5;
    wrong[3] +%= 1;
    const sn5 = try tr.keepAll(t, 5, 10 * bt, &ids5, .{ &ids5, &wrong });
    defer gpa.destroy(sn5);
    try pumpUntil(ld, 5);
    const key5 = format.pointKey(&ids5, bt, &.{});
    try tr.ws[0].waitPresent(key5, true);
    try testing.expect(std.mem.readInt(u32, (try tr.ws[1].query(key5))[10..14], .little) >= 1); // skipped there
    const p5 = t.vtable.lookup(t.ptr, &want(&ids5)) orelse return error.NotFound;
    try testing.expectError(error.SpillNotOnEveryRank, t.vtable.begin(t.ptr, &p5));
    try tr.waitAll(key5, false);
    try testing.expect(!ld.side.store.presence(key5));

    try testing.expectEqual(@as(u64, 0), t.vtable.flush(t.ptr, 5 * std.time.ns_per_s));
    ld.destroy();
    ld_alive = false;
    try tr.finish();
}

test "PETABRIDGE TP3: rank 0's guard stages both followers' copies before the rows change there" {
    const tr = try Trio.start("trioguard", 94);
    defer tr.deinit();
    const r0 = tr.r[0];
    var fam: Fam0 = .{ .sim = r0.sim };
    defer fam.deinit();
    const ld = try tr.leader(&fam, 5);
    var ld_alive = true;
    defer if (ld_alive) ld.destroy();
    const t = ld.tier();
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 111);
    const before0 = try gpa.dupe(u8, r0.sim.rows(0, 0, 4 * bt));
    defer gpa.free(before0);
    const sn = try tr.keepAll(t, 1, 0, &ids, .{ &ids, &ids });
    defer gpa.destroy(sn);
    // the guard (write_staged to both), then the rows change on every rank (each bumps its own generations)
    ld.guardSnap(sn.id, 0);
    ld.bump(0, 4 * bt);
    r0.sim.clobber(0, 4 * bt);
    for (&tr.ws) |*w| try w.send(T.clobber, &.{ &le(u64, 0), &le(u64, 4 * bt) });
    try pumpUntil(ld, 1);
    const key = format.pointKey(&ids, bt, &.{});
    try tr.waitAll(key, true);
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    const done = pollUntilDone(t, try t.vtable.begin(t.ptr, &p));
    try testing.expect(done == .ready);
    const got: *Snap = @ptrCast(@alignCast(done.ready.saved));
    defer r0.sim.freeSnap(got.buf);
    try testing.expectEqualSlices(u8, before0, r0.sim.rows(0, got.base, got.base + 4 * bt));
    for (&tr.ws) |*w| try testing.expect(try w.yes(T.saved, &.{&le(u64, got.base)}));
    ld.destroy();
    ld_alive = false;
    try tr.finish();
}
