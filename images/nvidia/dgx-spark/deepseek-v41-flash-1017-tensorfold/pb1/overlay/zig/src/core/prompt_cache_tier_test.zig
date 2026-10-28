//! The prompt cache's durable tier and 0111 continuation (wojo's GLM 0110/0111): a fake in-memory tier, and a family
//! whose state is a running hash of the prompt, so a restored state equals a fresh pass exactly when it should.
//! The continuation cases translate tools/chat_continued_check.py at fork main 8641c90.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const FakeTier = @import("prompt_cache_fake_tier.zig").FakeTier;
const Allocator = std.mem.Allocator;
const Store = pc.Store;
const Entry = pc.Entry;
const gpa = std.testing.allocator;

/// A family over host memory: its live state is a position and a running hash of the prompt's tokens.
const Sum = struct {
    at: u32 = 0,
    sum: u64 = 0,
    live: usize = 0,
    stale: ?pc.Saved = null, // `usable` refuses this state

    const State = struct { at: u32, sum: u64 };

    fn snapshots(f: *Sum) pc.Snapshots {
        // PETABRIDGE (k2a-pb1) spill P5: k2a has no Snapshots.usable; its `live` (in the family's list, not stale) is
        // the same test, so the stale state is refused there (liveFn, below).
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .live = liveFn } };
    }
    fn of(ptr: *anyopaque) *Sum {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytesFn(_: *anyopaque, at: u32) u64 {
        return 100 + @as(u64, at);
    }
    fn saveFn(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f = of(ptr);
        if (f.at != at) return error.NotAtMark;
        const st = try gpa.create(State);
        st.* = .{ .at = f.at, .sum = f.sum };
        f.live += 1;
        return st;
    }
    fn restoreFn(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) anyerror!void {
        const st: *State = @ptrCast(@alignCast(saved));
        of(ptr).at = st.at;
        of(ptr).sum = st.sum;
    }
    fn dropFn(ptr: *anyopaque, saved: pc.Saved) void {
        of(ptr).live -= 1;
        gpa.destroy(@as(*State, @ptrCast(@alignCast(saved))));
    }
    // PETABRIDGE (k2a-pb1) spill P5: the tip's usableFn(ptr, saved, len) as k2a's live(ptr, saved) (it never read len)
    fn liveFn(ptr: *anyopaque, saved: pc.Saved) bool {
        return of(ptr).stale != saved;
    }
    /// The live state run on to `to` tokens of `prompt` (a fresh pass's arithmetic).
    fn feed(f: *Sum, prompt: []const u32, to: u32) void {
        for (prompt[f.at..to]) |t| f.sum = f.sum *% 31 +% t;
        f.at = to;
    }
    fn reset(f: *Sum) void {
        f.at = 0;
        f.sum = 0;
    }
    fn fresh(prompt: []const u32, to: u32) u64 {
        var x: Sum = .{};
        x.feed(prompt, to);
        return x.sum;
    }

    fn codec(f: *Sum) @import("prompt_cache_fake_tier.zig").Codec {
        return .{ .ctx = f, .pack = pack, .unpack = unpack };
    }
    fn pack(_: *anyopaque, a: Allocator, saved: pc.Saved) anyerror![]u8 {
        return a.dupe(u8, std.mem.asBytes(@as(*State, @ptrCast(@alignCast(saved)))));
    }
    fn unpack(ctx: *anyopaque, bytes: []const u8) anyerror!pc.Saved {
        if (bytes.len != @sizeOf(State)) return error.SizeMismatch;
        const st = try gpa.create(State);
        @memcpy(std.mem.asBytes(st), bytes);
        of(ctx).live += 1;
        return st;
    }
};

fn seq(comptime n: usize, base: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*t, i| t.* = base + @as(u32, @intCast(i));
    return out;
}

/// Keep `prompt`'s state at `at` as a pass would: the family stands there.
fn keepAt(s: *Store, f: *Sum, prompt: []const u32, at: u32, info: pc.KeepInfo) bool {
    f.reset();
    f.feed(prompt, at);
    return s.keepWith(prompt, at, null, &.{}, info);
}

/// Poll `r` until it is no longer pending.
fn settle(s: *Store, r: *pc.Restore) pc.Restored {
    while (true) {
        const st = s.restorePoll(r, 1000);
        if (st != .pending) return st;
    }
}

test "without a tier the store asks nothing and counts nothing new" {
    var f: Sum = .{};
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const p = seq(40, 1);
    try std.testing.expect(keepAt(&s, &f, &p, 16, .{}));
    const pr = s.probe(&p, &.{});
    try std.testing.expectEqual(@as(u32, 16), pr.memory.?.at);
    try std.testing.expect(pr.durable == null);
    try std.testing.expectEqual(pc.Stats{}, s.stats);
    try std.testing.expect(s.entries.items[0].point == null and s.entries.items[0].group == 0);
    try std.testing.expectEqual(pc.Join{}, s.join(&p, null, true)); // grouping off: group 0, no roots
    try std.testing.expectEqual(@as(u32, 0), s.roots.count());
    try std.testing.expectEqual(@as(u64, 0), s.flush(0));
}

test "a keep enqueues its write with conversation and lineage; a state that left memory restores, equal to a fresh pass" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec(), .polls = 3 };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 150); // room for one state
    defer s.deinit();
    s.setTier(tier.tier(), 7);
    try std.testing.expect(s.groups);
    const a = seq(40, 1);
    const g = s.join(&a, null, true).group;
    try std.testing.expectEqual(@as(u64, 7 << 32) + 1, g);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{ .group = g }));
    try std.testing.expectEqual(@as(usize, 1), tier.queue.items.len);
    try std.testing.expectEqual(g, tier.queue.items[0].group);
    try std.testing.expectEqual(@as(?u64, null), tier.queue.items[0].parent);
    s.pump(1000);
    try std.testing.expectEqual(@as(usize, 1), tier.points.items.len);
    const b = seq(40, 1000); // another conversation's state takes the only room
    try std.testing.expect(keepAt(&s, &f, &b, 16, .{}));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    var a2: [48]u32 = undefined;
    @memcpy(a2[0..40], &a);
    for (a2[40..], 0..) |*t, i| t.* = 500 + @as(u32, @intCast(i));
    const pr = s.probe(&a2, &.{});
    try std.testing.expect(pr.memory == null);
    const p = pr.durable.?;
    try std.testing.expectEqual(@as(u32, 16), p.at);
    try std.testing.expectEqual(g, p.group);
    const r = try s.restoreBegin(p, &a2);
    try std.testing.expectEqual(pc.Restored.pending, s.restorePoll(r, 1000));
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, r));
    try std.testing.expectEqual(@as(usize, 0), s.restoring.items.len);
    const e = s.find(&a2, &.{}).?; // the restored state, `b`'s evicted for its room
    try std.testing.expect(e.restored and e.point.? == p.id and e.group == g and !e.shared);
    try s.family.vtable.restore(s.family.ptr, null, e.saved);
    f.feed(&a2, 48);
    try std.testing.expectEqual(Sum.fresh(&a2, 48), f.sum);
    s.resumed(e, &a2, true);
    try std.testing.expectEqual(@as(u64, 1), tier.touched);
    // a later keep of this pass forks the restored point's lineage at the resume point
    try std.testing.expect(keepAt(&s, &f, &a2, 32, .{ .group = g, .parent = e.point, .fork = 16 }));
    try std.testing.expectEqual(p.id, tier.queue.items[0].parent.?);
    try std.testing.expectEqual(@as(u32, 16), tier.queue.items[0].fork);
    try std.testing.expectEqual(@as(u64, 1), s.stats.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 16), s.stats.restored_tokens);
    try std.testing.expectEqual(@as(u64, 1), s.stats.restores_begun);
    try std.testing.expectEqual(@as(u64, 1), s.stats.disk_hits);
}

test "a stale long memory state never shadows a durable point; ineligible points are not offered" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .grid = 8, .window = 4 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(64, 1);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{}));
    try std.testing.expect(keepAt(&s, &f, &a, 32, .{}));
    s.pump(0);
    try std.testing.expect(keepAt(&s, &f, &a, 48, .{})); // memory only (never pumped)
    // the 48 state goes stale: the lookup takes 32 from memory, and the durable 48 is not on disk yet
    f.stale = s.entries.items[2].saved;
    var pr = s.probe(&a, &.{});
    try std.testing.expectEqual(@as(u32, 32), pr.memory.?.at);
    try std.testing.expect(pr.durable == null);
    s.pump(0);
    pr = s.probe(&a, &.{});
    try std.testing.expectEqual(@as(u32, 48), pr.durable.?.at); // not shadowed by its stale memory twin
    // eligibility: a prompt whose replay rows reach past 48 (len 50, window 4) gets 32 from memory, no point
    pr = s.probe(a[0..50], &.{});
    try std.testing.expectEqual(@as(u32, 32), pr.memory.?.at);
    try std.testing.expect(pr.durable == null);
    // a lax tier offering off-grid or ineligible points: the store checks them itself
    tier.lax = true;
    _ = try tier.put(a[0..20], 20, std.mem.asBytes(&Sum.State{ .at = 20, .sum = 0 }), 0, 0, false);
    pr = s.probe(a[0..50], &.{});
    try std.testing.expect(pr.durable == null); // 48 is past the replay rows of 50, 20 is off the grid
    try std.testing.expect(!pc.Rules.eligible(.{ .grid = 8, .window = 4 }, 47, 64));
    try std.testing.expect(pc.Rules.eligible(.{ .grid = 8, .window = 4 }, 48, 52));
    try std.testing.expect(!pc.Rules.eligible(.{ .grid = 8, .window = 4 }, 48, 51));
    try std.testing.expectEqual(@as(?u32, 48), pc.Rules.maxAt(.{ .grid = 8, .window = 4 }, 52));
    try std.testing.expectEqual(@as(?u32, null), pc.Rules.maxAt(.{ .grid = 8, .window = 4 }, 9));
    try std.testing.expectEqual(@as(?u32, 9), pc.Rules.maxAt(.{}, 10));
}

test "one restore per point: a second request joins it, and it is cancelled once" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec(), .polls = 5 };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    _ = try tier.put(a[0..16], 16, std.mem.asBytes(&Sum.State{ .at = 16, .sum = Sum.fresh(&a, 16) }), 0, 3, true);
    const p = s.probe(&a, &.{}).durable.?;
    try std.testing.expect(p.shared and p.group == 3);
    const r1 = try s.restoreBegin(p, &a);
    const r2 = try s.restoreBegin(p, a[0..30]);
    try std.testing.expectEqual(r1, r2);
    try std.testing.expectEqual(@as(u64, 1), s.stats.restores_shared);
    try std.testing.expectEqual(@as(usize, 1), tier.jobs.items.len);
    s.restoreCancel(r1);
    try std.testing.expectEqual(@as(usize, 0), tier.jobs.items.len);
    try std.testing.expectEqual(@as(u64, 1), tier.cancels);
    try std.testing.expectEqual(@as(u64, 0), tier.misuse);
    try std.testing.expectEqual(@as(u64, 1), s.stats.restores_cancelled);
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    // an adopted shared point is a shared entry: other conversations resume it, its own never supersedes it
    const r3 = try s.restoreBegin(p, &a);
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, r3));
    try std.testing.expect(s.entries.items[0].shared and s.entries.items[0].restored);
    try std.testing.expectEqual(@as(usize, 1), f.live);
}

test "a restore that fails anywhere leaves nothing; one memory already holds is dropped, its copy freed" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    _ = try tier.put(a[0..16], 16, std.mem.asBytes(&Sum.State{ .at = 16, .sum = Sum.fresh(&a, 16) }), 0, 0, false);
    const p = s.probe(&a, &.{}).durable.?;
    tier.fail_next = error.CrcMismatch;
    try std.testing.expectEqual(pc.Restored.failed, settle(&s, try s.restoreBegin(p, &a)));
    try std.testing.expectEqual(@as(u64, 1), s.stats.restores_failed);
    try std.testing.expectEqual(@as(u64, 1), tier.c.crc_errors);
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    const r = try s.restoreBegin(p, &a);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{})); // a pass kept the same state meanwhile
    try std.testing.expectEqual(pc.Restored.dropped, settle(&s, r));
    try std.testing.expectEqual(@as(u64, 1), s.stats.restores_dropped);
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqual(@as(usize, 1), f.live);
    // too large for the whole budget: never offered
    s.budget = 100;
    try std.testing.expect(s.probe(&a, &.{}).durable == null);
}

test "a stale memory twin of a durable point is replaced by its restore; an evicted state restores too" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec(), .polls = 2 };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{}));
    s.pump(1000); // on disk too
    // the family stales the kept state (KV sharding gave its rows to a placement): the entry stays in memory
    f.stale = s.entries.items[0].saved;
    const pr = s.probe(&a, &.{});
    try std.testing.expect(pr.memory == null); // never resumed
    const p = pr.durable.?;
    try std.testing.expectEqual(@as(u32, 16), p.at);
    // the restore of the same point replaces the stale twin instead of being dropped (a recompute)
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, try s.restoreBegin(p, &a)));
    try std.testing.expectEqual(@as(u64, 1), s.stats.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 0), s.stats.restores_dropped);
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expect(s.entries.items[0].restored and s.entries.items[0].saved != f.stale.?);
    try std.testing.expectEqual(@as(usize, 1), f.live); // the stale state was given back to the family
    const e = s.find(&a, &.{}).?;
    try s.family.vtable.restore(s.family.ptr, null, e.saved);
    f.feed(&a, 40);
    try std.testing.expectEqual(Sum.fresh(&a, 40), f.sum); // the restored state is the prompt's
    f.stale = null;
    // a usable copy kept while the restore ran still wins: that restore is dropped, its copy freed
    const c = seq(40, 2000);
    try std.testing.expect(keepAt(&s, &f, &c, 16, .{}));
    s.pump(1000);
    f.stale = s.entries.items[s.entries.items.len - 1].saved;
    const pcp = s.probe(&c, &.{}).durable.?;
    const r = try s.restoreBegin(pcp, &c);
    f.stale = null; // (the same state usable again: as if a pass kept it anew meanwhile)
    try std.testing.expectEqual(pc.Restored.dropped, settle(&s, r));
    try std.testing.expectEqual(@as(u64, 1), s.stats.restores_dropped);
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expectEqual(@as(usize, 2), f.live);
    // an evicted state (its entry gone from memory for another's room): its restore is adopted, nothing dropped
    var g: Sum = .{};
    var tier2: FakeTier = .{ .gpa = gpa, .codec = g.codec(), .polls = 2 };
    defer tier2.deinit();
    var s2 = Store.init(gpa, g.snapshots(), .{ .min_prompt = 0 }, 150); // room for one state
    defer s2.deinit();
    s2.setTier(tier2.tier(), 0);
    const b = seq(40, 1000);
    try std.testing.expect(keepAt(&s2, &g, &b, 16, .{}));
    s2.pump(1000);
    try std.testing.expect(keepAt(&s2, &g, &c, 16, .{})); // evicts b's
    try std.testing.expectEqual(@as(u64, 1), s2.counts.evicted);
    const pb = s2.probe(&b, &.{}).durable.?;
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s2, try s2.restoreBegin(pb, &b)));
    try std.testing.expectEqual(@as(u64, 1), s2.stats.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 0), s2.stats.restores_dropped);
}

test "a write still queued is staged before its memory state is dropped; a new shared prefix is marked on disk" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 4 }, 150);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    const b = seq(40, 900);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{}));
    try std.testing.expect(keepAt(&s, &f, &b, 16, .{})); // evicts a's state before its write was staged
    try std.testing.expectEqual(@as(u64, 1), tier.c.forced_staging);
    try std.testing.expectEqual(@as(usize, 1), tier.points.items.len);
    const st: *const Sum.State = @ptrCast(@alignCast(tier.points.items[0].bytes.ptr));
    try std.testing.expectEqual(Sum.fresh(&a, 16), st.sum); // the bytes it wrote are a's, staged in time
    // b's state turns out to be a shared cut: kept again, it is promoted on disk (made stale first so the lookup plans it)
    f.stale = s.entries.items[0].saved;
    s.budget = 1 << 20; // (room for the lookup's marks: it evicts nothing)
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    _ = try s.lookup(arena.allocator(), &b, 30, &.{16}, &.{});
    f.stale = null;
    try std.testing.expect(keepAt(&s, &f, &b, 16, .{}));
    try std.testing.expect(s.entries.items[0].shared);
    try std.testing.expectEqual(@as(u64, 1), tier.shared_marks);
    try std.testing.expectEqual(@as(u64, 1), s.stats.shared_promoted);
    try std.testing.expect(tier.queue.items[0].shared);
}

test "flush stops restores, drains writes, and reports what its deadline left unpublished" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec(), .polls = 100 };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    const b = seq(40, 300);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{}));
    try std.testing.expect(s.busy());
    try std.testing.expectEqual(@as(u64, 0), s.flush(1000));
    try std.testing.expect(!s.busy());
    try std.testing.expectEqual(@as(usize, 1), tier.points.items.len);
    const c = seq(40, 600);
    _ = try tier.put(c[0..16], 16, std.mem.asBytes(&Sum.State{ .at = 16, .sum = Sum.fresh(&c, 16) }), 0, 0, false);
    _ = try s.restoreBegin(s.probe(&c, &.{}).durable orelse return error.TestUnexpectedResult, &c);
    try std.testing.expect(keepAt(&s, &f, &b, 16, .{}));
    tier.stuck = true;
    try std.testing.expectEqual(@as(u64, 1), s.flush(1000));
    try std.testing.expectEqual(@as(usize, 0), s.restoring.items.len);
    try std.testing.expectEqual(@as(u64, 1), tier.cancels);
    try std.testing.expectEqual(@as(u64, 1), s.stats.flush_timeouts);
    try std.testing.expectEqual(@as(u64, 1), s.stats.flush_unpublished);
    try std.testing.expectEqual(@as(u64, 2), s.stats.flushes);
}

test "forgetting a point drops it on disk and unlinks its memory twin" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{}));
    s.pump(0);
    const id = s.entries.items[0].point.?;
    s.forgetPoint(id);
    try std.testing.expect(s.entries.items[0].point == null);
    try std.testing.expect(tier.point(id) == null);
    try std.testing.expectEqual(@as(u64, 0), tier.released);
}

// -- 0111: chat_continued_check.py -----------------------------------------------------------------------------------

const system = seq(900, 100000); // a 900-token system prompt every conversation shares
const opening = seq(100, 200000); // and the same 100-token opening message

/// A conversation's prompts: the first, then each turn appending a reply and a tool result of its own (50 tokens).
fn convo(a: Allocator, seed: u32, turns: usize) ![][]u32 {
    const out = try a.alloc([]u32, turns + 1);
    var ids: std.ArrayList(u32) = .empty;
    try ids.appendSlice(a, &system);
    try ids.appendSlice(a, &opening);
    out[0] = try a.dupe(u32, ids.items);
    for (0..turns) |t| {
        for (0..50) |k| try ids.append(a, seed * 1000 + @as(u32, @intCast(t)) * 50 + @as(u32, @intCast(k)));
        out[t + 1] = try a.dupe(u32, ids.items);
    }
    return out;
}

const Kind = struct { own: bool = true, images: bool = false, restored: bool = false, at: ?u32 = null };

/// A kept state with `last` as its evidence (as the store records it at keep), in use order; it stands at `kind.at`
/// (default: one short of its evidence's end).
fn state(s: *Store, f: *Sum, last: []const u32, group: u64, kind: Kind) !*Entry {
    const e = try gpa.create(Entry);
    errdefer gpa.destroy(e);
    const at: u32 = kind.at orelse @intCast(last.len - 1);
    f.at = at;
    const saved = try Sum.saveFn(f, null, at);
    s.clock += 1;
    e.* = .{ .tokens = try gpa.dupe(u32, last[0..at]), .at = at, .saved = saved, .bytes = 0, .born = @intCast(last.len), .used = s.clock, .last = try gpa.dupe(u32, last), .shared = !kind.own, .group = group, .images = kind.images, .restored = kind.restored };
    try s.entries.append(gpa, e);
    return e;
}

fn cat(a: Allocator, x: []const u32, y: []const u32) ![]u32 {
    const out = try a.alloc(u32, x.len + y.len);
    @memcpy(out[0..x.len], x);
    @memcpy(out[x.len..], y);
    return out;
}

test "common prefix" {
    const big = try gpa.alloc(u32, 500000);
    defer gpa.free(big);
    for (big, 0..) |*t, i| t.* = @intCast(i);
    const other = try gpa.dupe(u32, big);
    defer gpa.free(other);
    other[123457] = std.math.maxInt(u32);
    try std.testing.expectEqual(@as(usize, 2), pc.commonPrefix(&.{ 1, 2, 3, 4 }, &.{ 1, 2, 9 }));
    try std.testing.expectEqual(@as(usize, 3), pc.commonPrefix(&.{ 1, 2, 3 }, &.{ 1, 2, 3, 4 }));
    try std.testing.expectEqual(@as(usize, 0), pc.commonPrefix(&.{}, &.{1}));
    try std.testing.expectEqual(@as(usize, 123457), pc.commonPrefix(big, other));
}

test "continued conversations: a rewind continues, another conversation or an edit starts its own, the newest match wins" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    var f: Sum = .{};
    var s = Store.init(gpa, f.snapshots(), .{}, 1 << 30);
    defer s.deinit();
    s.groups = true;
    const a = try convo(ar, 7, 6);
    const b = try convo(ar, 8, 6);
    try s.roots.put(gpa, 1, @intCast(a[0].len));
    _ = try state(&s, &f, &system, 0, .{ .own = false });
    _ = try state(&s, &f, a[5], 1, .{});
    const newest = try state(&s, &f, a[6], 1, .{});
    // issue #104: a rewind before the kept turns continues its conversation
    try std.testing.expectEqual(newest, s.continuedChat(try cat(ar, a[3], &.{ 9, 9, 9 })).?);
    // another conversation behind the same system prompt and opening starts its own
    try std.testing.expect(s.continuedChat(b[2]) == null);
    // a prompt that has not repeated the first reply yet starts its own
    const fives: [10]u32 = @splat(5);
    try std.testing.expect(s.continuedChat(try cat(ar, a[0], &fives)) == null);
    // an edit of the first message starts a new conversation
    var edited: std.ArrayList(u32) = .empty;
    try edited.appendSlice(ar, &system);
    try edited.appendSlice(ar, opening[0..50]);
    try edited.appendNTimes(ar, 0, 50);
    try edited.appendSlice(ar, a[2][a[0].len..]);
    try std.testing.expect(s.continuedChat(edited.items) == null);
}

test "continued conversations skip shared, pictured, restored and forgotten conversations' states" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    const a = try convo(ar, 7, 6);
    const prompt = try cat(ar, a[2], &.{1});
    const kinds = [_]struct { group: u64, kind: Kind }{ .{ .group = 1, .kind = .{ .own = false } }, .{ .group = 1, .kind = .{ .images = true } }, .{ .group = 1, .kind = .{ .restored = true } }, .{ .group = 2, .kind = .{} } };
    for (kinds) |k| {
        var f: Sum = .{};
        var s = Store.init(gpa, f.snapshots(), .{}, 1 << 30);
        defer s.deinit();
        s.groups = true;
        try s.roots.put(gpa, 1, @intCast(a[0].len));
        _ = try state(&s, &f, a[4], k.group, k.kind);
        try std.testing.expect(s.continuedChat(prompt) == null);
    }
    var f: Sum = .{};
    var s = Store.init(gpa, f.snapshots(), .{}, 1 << 30);
    defer s.deinit();
    s.groups = true;
    try s.roots.put(gpa, 1, @intCast(a[0].len));
    try s.roots.put(gpa, 2, @intCast(a[0].len));
    _ = try state(&s, &f, a[3], 1, .{});
    _ = try state(&s, &f, a[4], 2, .{});
    try std.testing.expectEqual(@as(u64, 2), s.continuedChat(prompt).?.group); // the most recently used match wins
    s.entries.items[0].used = s.entries.items[1].used; // a tie of use: the later entry, both ranks alike
    try std.testing.expectEqual(@as(u64, 2), s.continuedChat(prompt).?.group);
}

test "join: an own turn keeps its conversation, a continuation names it, anything else starts one" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    var f: Sum = .{};
    var s = Store.init(gpa, f.snapshots(), .{}, 1 << 30);
    defer s.deinit();
    s.groups = true;
    s.next_group = 40;
    const a = try convo(ar, 7, 6);
    const shared = try state(&s, &f, &system, 0, .{ .own = false });
    const first = s.join(a[0], shared, true); // a new conversation behind a shared prefix
    try std.testing.expectEqual(@as(u64, 41), first.group);
    try std.testing.expect(!first.continued);
    try std.testing.expectEqual(@as(u32, @intCast(a[0].len)), s.roots.get(41).?);
    const own = try state(&s, &f, a[3], 41, .{});
    const turn = s.join(a[4], own, true); // its own turn state: the conversation, not counted
    try std.testing.expectEqual(@as(u64, 41), turn.group);
    try std.testing.expect(!turn.continued);
    const rewound = s.join(try cat(ar, a[2], &.{9}), shared, true); // back before its kept turn, from the shared prefix
    try std.testing.expectEqual(@as(u64, 41), rewound.group);
    try std.testing.expect(rewound.continued and rewound.source == .shared);
    const pictured = s.join(try cat(ar, a[2], &.{9}), shared, false); // pictures or draft:false: never matched
    try std.testing.expectEqual(@as(u64, 42), pictured.group);
    // conversations no kept state is in are forgotten once there are many
    for (0..80) |_| _ = s.join(&.{ 1, 2, 3 }, null, true);
    try std.testing.expect(s.roots.count() <= 4 * 16 + 1);
    try std.testing.expect(s.roots.contains(41)); // a kept state still names it
}

test "a continued request is counted with its prefill; a large one's line says where it resumed and why" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    var f: Sum = .{};
    var s = Store.init(gpa, f.snapshots(), .{}, 1 << 30);
    defer s.deinit();
    s.groups = true;
    const long = try convo(ar, 7, 200);
    try s.roots.put(gpa, 1, @intCast(long[0].len));
    _ = try state(&s, &f, long[200], 1, .{});
    const shared = try state(&s, &f, &system, 0, .{ .own = false });
    const prompt = try cat(ar, long[150], &.{9});
    const j = s.join(prompt, shared, true);
    try std.testing.expect(j.continued and j.group == 1);
    s.continued(j, system.len, prompt.len);
    try std.testing.expectEqual(@as(u64, 1), s.stats.chats_continued);
    try std.testing.expectEqual(@as(u64, prompt.len - system.len), s.stats.chats_continued_prefill_tokens);
    var buf: [512]u8 = undefined;
    const line = pc.continuedLine(&buf, j, system.len, prompt.len);
    try std.testing.expect(std.mem.indexOf(u8, line, "conversation 1: resumed at 900 of") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "from a shared prefix") != null);
    const back = try std.fmt.allocPrint(ar, "went back to token {d}", .{long[150].len});
    try std.testing.expect(std.mem.indexOf(u8, line, back) != null);
    try std.testing.expectEqual(@as(u32, @intCast(long[200].len - 1)), j.prior_len); // the state's own ids, not its evidence's
    // a prompt that repeats the state's ids but leaves its evidence after them: not a rewind (stale, off the grid...)
    const past = s.join(try cat(ar, long[200][0 .. long[200].len - 1], &.{ 9, 9 }), shared, true);
    try std.testing.expect(past.continued and past.back >= past.prior_len);
    try std.testing.expect(std.mem.indexOf(u8, pc.continuedLine(&buf, past, system.len, 9000), "could not serve it") != null);
    // a small prefill: counted (the log threshold is chat_log_tokens); a request that did not continue: not counted
    s.continued(j, @intCast(prompt.len - 10), prompt.len);
    s.continued(.{ .group = 5 }, 0, prompt.len);
    try std.testing.expectEqual(@as(u64, 2), s.stats.chats_continued);
    try std.testing.expectEqual(@as(u64, prompt.len - system.len + 10), s.stats.chats_continued_prefill_tokens);
    const nothing = pc.continuedLine(&buf, .{ .group = 3, .continued = true, .prior_len = 50, .back = 50 }, 0, 9000);
    try std.testing.expect(std.mem.indexOf(u8, nothing, "from nothing") != null and std.mem.indexOf(u8, nothing, "could not serve it") != null);
}

test "the log threshold: a continued request's prefill of chat_log_tokens or more" {
    const j: pc.Join = .{ .group = 1, .continued = true };
    try std.testing.expect(!pc.shouldLog(j, 100, 100 + pc.chat_log_tokens - 1));
    try std.testing.expect(pc.shouldLog(j, 100, 100 + pc.chat_log_tokens));
    try std.testing.expect(!pc.shouldLog(.{ .group = 1 }, 0, 1 << 20)); // not continued
    try std.testing.expect(!pc.shouldLog(j, 9000, 100)); // nothing prefilled
}

test "join: a hit on another conversation's state inside the shared system prompt is no evidence of that conversation" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    var f: Sum = .{};
    var s = Store.init(gpa, f.snapshots(), .{}, 1 << 30);
    defer s.deinit();
    s.groups = true;
    s.next_group = 10;
    const a = try convo(ar, 7, 3);
    const ga = s.join(a[0], null, true).group;
    // A's turn-1 state on the grid inside the system prompt (as dsv41 keeps a 2048 floor), its evidence A's prompt
    const inside = try state(&s, &f, a[0], ga, .{ .at = 800 });
    // B: the same system prompt, its own opening; it resumes A's state, and is a conversation of its own
    const b0 = try cat(ar, &system, &seq(100, 300000));
    const jb = s.join(b0, inside, true);
    try std.testing.expect(jb.group != ga and !jb.continued);
    // A's next turn resuming the same state is A's own turn
    const ja = s.join(a[1], inside, true);
    try std.testing.expectEqual(ga, ja.group);
    try std.testing.expect(!ja.continued);
    // a conversation whose first prompt was forgotten keeps the hit-alone rule
    _ = s.roots.remove(ga);
    try std.testing.expectEqual(ga, s.join(b0, inside, true).group);
}

test "with a grid an off-grid shared cut marks the state the family keeps on the grid as shared, on disk too" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 4, .grid = 8 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const p = seq(40, 1);
    const l = try s.lookup(arena.allocator(), &p, 30, &.{21}, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 21, 30 }, l.marks);
    try std.testing.expectEqual(@as(usize, 1), s.shared_keys.items.len);
    try std.testing.expect(keepAt(&s, &f, &p, 16, .{})); // dsv41 keeps the mark at 21 at its chunk end, 16
    try std.testing.expect(s.entries.items[0].shared);
    try std.testing.expect(tier.queue.items[0].shared);
}

/// Sum with spare storage: a dropped state's bytes stay spare, a save takes spare first, trim frees it.
const Pool = struct {
    sum: Sum = .{},
    spare_: u64 = 0,

    fn snapshots(p: *Pool) pc.Snapshots {
        return .{ .ptr = p, .vtable = &.{ .bytes = Sum.bytesFn, .save = save, .restore = restore, .drop = drop, .spare = spare, .trim = trim, .reuses = reuses } };
    }
    fn of(ptr: *anyopaque) *Pool {
        return @ptrCast(@alignCast(ptr));
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const p = of(ptr);
        if (p.spare_ >= Sum.bytesFn(ptr, at)) p.spare_ -= Sum.bytesFn(ptr, at);
        return Sum.saveFn(&p.sum, owner, at);
    }
    fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: pc.Saved) anyerror!void {
        return Sum.restoreFn(&of(ptr).sum, owner, saved);
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const st: *Sum.State = @ptrCast(@alignCast(saved));
        of(ptr).spare_ += Sum.bytesFn(ptr, st.at);
        Sum.dropFn(&of(ptr).sum, saved);
    }
    fn spare(ptr: *anyopaque) u64 {
        return of(ptr).spare_;
    }
    fn trim(ptr: *anyopaque, room: u64) void {
        of(ptr).spare_ = @min(of(ptr).spare_, room);
    }
    fn reuses(ptr: *anyopaque, at: u32) bool {
        return of(ptr).spare_ >= Sum.bytesFn(ptr, at);
    }
};

test "a restore's destination is charged to the budget while it runs; spare storage in its way is trimmed, not every state" {
    var pool: Pool = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = pool.sum.codec(), .polls = 2 };
    defer tier.deinit();
    var s = Store.init(gpa, pool.snapshots(), .{ .min_prompt = 0 }, 400); // states of 100 + at bytes
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    const b = seq(40, 300);
    const c = seq(40, 600);
    pool.sum.reset();
    pool.sum.feed(&a, 10);
    try std.testing.expect(s.keepWith(&a, 10, null, &.{}, .{}));
    pool.sum.reset();
    pool.sum.feed(&b, 10);
    try std.testing.expect(s.keepWith(&b, 10, null, &.{}, .{}));
    pool.sum.reset();
    pool.sum.feed(&c, 10);
    try std.testing.expect(s.keepWith(&c, 10, null, &.{}, .{}));
    try std.testing.expectEqual(@as(u64, 330), s.held);
    const d = seq(40, 900);
    _ = try tier.put(d[0..20], 20, std.mem.asBytes(&Sum.State{ .at = 20, .sum = Sum.fresh(&d, 20) }), 0, 0, false);
    const p = s.probe(&d, &.{}).durable.?;
    const r = try s.restoreBegin(p, &d); // 120 bytes: one state goes, its spare storage is trimmed to fit
    try std.testing.expectEqual(@as(u64, 120), s.restoring_bytes);
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expect(s.held + s.restoring_bytes + s.spare() <= s.budget);
    try std.testing.expectEqual(@as(u64, 400 - 220 - 120 - s.spare()), s.room());
    // a second restore the budget cannot hold beside the first and the states it must keep: refused, nothing begun
    const e2 = seq(400, 2000);
    _ = try tier.put(e2[0..300], 300, std.mem.asBytes(&Sum.State{ .at = 300, .sum = 0 }), 0, 0, false);
    const p2 = s.probe(&e2, &.{}).durable.?;
    try std.testing.expectError(error.NoRoom, s.restoreBegin(p2, &e2));
    try std.testing.expectEqual(@as(u64, 1), s.stats.restores_refused);
    try std.testing.expectEqual(@as(usize, 1), s.restoring.items.len);
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len); // refused before evicting anything
    // adopted beside both remaining states: no eviction past what the reservation already made
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, r));
    try std.testing.expectEqual(@as(u64, 0), s.restoring_bytes);
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(s.held + s.spare() <= s.budget);
    // a cancelled restore gives its charge back (a's state was staged to disk when it was evicted)
    const r3 = try s.restoreBegin(s.probe(&a, &.{}).durable.?, &a);
    try std.testing.expectEqual(@as(u64, 110), s.restoring_bytes);
    s.restoreCancel(r3);
    try std.testing.expectEqual(@as(u64, 0), s.restoring_bytes);
    try std.testing.expectEqual(s.entries.items.len, pool.sum.live);
}

test "lineage: a fresh pass starts its own, a resumed one forks its state's at the resume point, kept states carry it to disk and back" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec(), .ranks = 2 };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 150); // room for one state
    defer s.deinit();
    s.setTier(tier.tier(), 3);
    const a = seq(48, 1);
    const fresh = try s.passLineage(null);
    defer gpa.free(fresh);
    try std.testing.expectEqual(@as(usize, 1), fresh.len);
    try std.testing.expectEqual(@as(u32, 0), fresh[0].at);
    try std.testing.expectEqual(@as(u64, 3) << 32, fresh[0].name >> 32 << 32);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{ .lineage = fresh })); // below any write minimum it is still carried
    const e = s.entries.items[0];
    try std.testing.expectEqualSlices(pc.Fork, fresh, e.lineage);
    try std.testing.expectEqualSlices(pc.Fork, fresh, tier.queue.items[0].lineage);
    const forked = try s.passLineage(e);
    defer gpa.free(forked);
    try std.testing.expectEqual(@as(usize, 2), forked.len);
    try std.testing.expectEqual(fresh[0], forked[0]);
    try std.testing.expectEqual(@as(u32, 16), forked[1].at);
    try std.testing.expect(forked[1].name != fresh[0].name);
    // its state leaves memory (staged to disk first) and comes back with its lineage
    const b = seq(40, 900);
    try std.testing.expect(keepAt(&s, &f, &b, 16, .{}));
    const r = try s.restoreBegin(s.probe(&a, &.{}).durable.?, &a);
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, r));
    try std.testing.expectEqualSlices(pc.Fork, fresh, s.find(&a, &.{}).?.lineage);
    // without a tier there is no lineage to carry
    var plain = Store.init(gpa, f.snapshots(), .{}, 1 << 20);
    defer plain.deinit();
    try std.testing.expectEqual(@as(usize, 0), (try plain.passLineage(null)).len);
    // every rank's counters
    const rc = s.tierCounts().?;
    try std.testing.expectEqual(@as(u32, 2), rc.n);
    try std.testing.expectEqual(@as(u64, 1), rc.slice()[1].crc_errors);
}

// PETABRIDGE (k2a-pb1) spill P5: the tip's last two tests ("a restore refused for room or capacity: ..." and "a NoRoom
// is worth waiting for only when the restores in flight hold the rest of the budget") are 0020's (Store.restoreBeginOver,
// Store.deferrable, Tier.full: TF_SPILL_DEFER_MS) and are left out with it.
