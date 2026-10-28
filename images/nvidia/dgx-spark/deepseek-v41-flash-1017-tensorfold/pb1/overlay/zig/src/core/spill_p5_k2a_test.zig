//! PETABRIDGE (k2a-pb1) spill P5: the tests of 0014's fit to k2a's prompt cache and lane host (no CUDA, no disk tier:
//! the in-memory FakeTier). Salted requests stay memory-only; dropping a memory entry releases its point and never
//! forgets it (eviction, prune_dead, dead_at_keep, a failed resume, a peer's refusal); the --learn imprint and the tier
//! never both hold or read one shared cut; restore polling and the 0005 prefill interleave share a fill's `between`.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const imprint = @import("prompt_imprint.zig");
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const ft = @import("prompt_cache_fake_tier.zig");
const ht = @import("lane_host_tier_test.zig");
const Store = pc.Store;
const Request = api.Request;
const Reason = api.Reason;
const gpa = std.testing.allocator;
const io = std.testing.io;

/// A family over host memory with k2a's liveness: a state is a position and a running hash; `dead` states are not live
/// (k2a's `live`: gone from the family's list, or stale), and it can learn states to disk for --learn.
const Fam = struct {
    at: u32 = 0,
    sum: u64 = 0,
    live_n: usize = 0,
    dead: [4]?pc.Saved = @splat(null),

    const State = struct { at: u32, sum: u64 };

    fn snapshots(f: *Fam) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .live = liveFn } };
    }
    /// With learned states on disk too (write/read/forget), as glm's host binds them.
    fn learned(f: *Fam) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .live = liveFn, .write = writeFn, .read = readFn, .forget = forgetFn } };
    }
    fn of(ptr: *anyopaque) *Fam {
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
        f.live_n += 1;
        return st;
    }
    fn restoreFn(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) anyerror!void {
        const st: *State = @ptrCast(@alignCast(saved));
        of(ptr).at = st.at;
        of(ptr).sum = st.sum;
    }
    fn dropFn(ptr: *anyopaque, saved: pc.Saved) void {
        const f = of(ptr);
        for (&f.dead) |*d| if (d.* == saved) {
            d.* = null;
        };
        f.live_n -= 1;
        gpa.destroy(@as(*State, @ptrCast(@alignCast(saved))));
    }
    fn liveFn(ptr: *anyopaque, saved: pc.Saved) bool {
        for (of(ptr).dead) |d| if (d == saved) return false;
        return true;
    }
    fn kill(f: *Fam, saved: pc.Saved) void {
        for (&f.dead) |*d| if (d.* == null) {
            d.* = saved;
            return;
        };
        unreachable;
    }
    fn feed(f: *Fam, prompt: []const u32, to: u32) void {
        f.at = 0;
        f.sum = 0;
        for (prompt[0..to]) |t| f.sum = f.sum *% 31 +% t;
        f.at = to;
    }
    fn codec(f: *Fam) ft.Codec {
        return .{ .ctx = f, .pack = pack, .unpack = unpack };
    }
    fn pack(_: *anyopaque, a: std.mem.Allocator, saved: pc.Saved) anyerror![]u8 {
        return a.dupe(u8, std.mem.asBytes(@as(*State, @ptrCast(@alignCast(saved)))));
    }
    fn unpack(ctx: *anyopaque, bytes: []const u8) anyerror!pc.Saved {
        if (bytes.len != @sizeOf(State)) return error.SizeMismatch;
        const st = try gpa.create(State);
        @memcpy(std.mem.asBytes(st), bytes);
        of(ctx).live_n += 1;
        return st;
    }
    fn file(buf: []u8, dir: []const u8, key: u64) ![:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.bin", .{ dir, key }, 0);
    }
    fn writeFn(_: *anyopaque, saved: pc.Saved, dir: [:0]const u8, key: u64) anyerror!void {
        var path: [512]u8 = undefined;
        const fd = std.c.open(try file(&path, dir, key), .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.WriteFailed;
        defer _ = std.c.close(fd);
        try imprint.writeAll(fd, std.mem.asBytes(@as(*State, @ptrCast(@alignCast(saved)))));
    }
    fn readFn(ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32) anyerror!pc.Saved {
        var path: [512]u8 = undefined;
        const fd = std.c.open(try file(&path, dir, key), .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.ReadFailed;
        defer _ = std.c.close(fd);
        const st = try gpa.create(State);
        errdefer gpa.destroy(st);
        if (!imprint.readAt(fd, std.mem.asBytes(st), 0) or st.at != at) return error.ReadFailed;
        of(ptr).live_n += 1;
        return st;
    }
    fn forgetFn(_: *anyopaque, dir: [:0]const u8, key: u64) void {
        var path: [512]u8 = undefined;
        _ = std.c.unlink(file(&path, dir, key) catch return);
    }
};

/// The FakeTier behind a spy that counts what the store asks of it: `forget` must never come from a memory drop.
const Spy = struct {
    inner: pc.Tier,
    forgets: u64 = 0,
    releases: u64 = 0,
    keeps: u64 = 0,
    lookups: u64 = 0,
    begins: u64 = 0,

    fn tier(s: *Spy) pc.Tier {
        return .{ .ptr = s, .vtable = &vt };
    }
    const vt: pc.Tier.VTable = .{ .lookup = lookup, .begin = begin, .poll = poll, .cancel = cancel, .keep = keep, .release = release, .share = share, .touch = touch, .forget = forget, .pump = pump, .busy = busy, .flush = flush, .counts = counts };
    fn of(p: *anyopaque) *Spy {
        return @ptrCast(@alignCast(p));
    }
    fn lookup(p: *anyopaque, w: *const pc.Want) ?pc.Point {
        const s = of(p);
        s.lookups += 1;
        return s.inner.vtable.lookup(s.inner.ptr, w);
    }
    fn begin(p: *anyopaque, pt: *const pc.Point) anyerror!u64 {
        const s = of(p);
        s.begins += 1;
        return s.inner.vtable.begin(s.inner.ptr, pt);
    }
    fn poll(p: *anyopaque, job: u64, b: u64) pc.Poll {
        return of(p).inner.vtable.poll(of(p).inner.ptr, job, b);
    }
    fn cancel(p: *anyopaque, job: u64) void {
        of(p).inner.vtable.cancel(of(p).inner.ptr, job);
    }
    fn keep(p: *anyopaque, k: *const pc.Kept) ?u64 {
        const s = of(p);
        s.keeps += 1;
        return s.inner.vtable.keep(s.inner.ptr, k);
    }
    fn release(p: *anyopaque, id: u64, saved: pc.Saved) void {
        const s = of(p);
        s.releases += 1;
        s.inner.vtable.release.?(s.inner.ptr, id, saved);
    }
    fn share(p: *anyopaque, id: u64) void {
        of(p).inner.vtable.share.?(of(p).inner.ptr, id);
    }
    fn touch(p: *anyopaque, id: u64) void {
        of(p).inner.vtable.touch.?(of(p).inner.ptr, id);
    }
    fn forget(p: *anyopaque, id: u64) void {
        const s = of(p);
        s.forgets += 1;
        s.inner.vtable.forget(s.inner.ptr, id);
    }
    fn pump(p: *anyopaque, b: u64) void {
        of(p).inner.vtable.pump.?(of(p).inner.ptr, b);
    }
    fn busy(p: *anyopaque) bool {
        return of(p).inner.vtable.busy.?(of(p).inner.ptr);
    }
    fn flush(p: *anyopaque, b: u64) u64 {
        return of(p).inner.vtable.flush(of(p).inner.ptr, b);
    }
    fn counts(p: *anyopaque, out: []pc.TierCounts) usize {
        return of(p).inner.vtable.counts(of(p).inner.ptr, out);
    }
};

fn seq(comptime n: usize, base: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*t, i| t.* = base + @as(u32, @intCast(i));
    return out;
}

fn keepAt(s: *Store, f: *Fam, ns: pc.Namespace, prompt: []const u32, at: u32) bool {
    f.feed(prompt, at);
    return s.keepScoped(ns, prompt, at, null, &.{});
}

test "PETABRIDGE spill P5: a salted keep never reaches the tier, a salted state is no twin, and its lookups stay in memory" {
    var f: Fam = .{};
    var tier: ft.FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var spy: Spy = .{ .inner = tier.tier() };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer s.deinit();
    s.setTier(spy.tier(), 0);
    const ns = pc.namespaceFor("tenant-a");
    const a = seq(40, 1);
    // salted keeps, a plain one and a shared cut alike: memory only, no point, nothing enqueued
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    _ = try s.lookupScoped(ns, arena.allocator(), &a, 30, &.{16}, &.{}); // (plans 16 as the salted shared cut)
    try std.testing.expect(keepAt(&s, &f, ns, &a, 16));
    try std.testing.expect(keepAt(&s, &f, ns, &a, 30));
    try std.testing.expectEqual(@as(u64, 0), spy.keeps);
    try std.testing.expectEqual(@as(usize, 0), tier.queue.items.len);
    for (s.entries.items) |e| try std.testing.expect(e.point == null and e.group == 0 and e.lineage.len == 0);
    try std.testing.expect(s.entries.items[0].shared);
    try std.testing.expectEqual(@as(u64, 0), s.stats.writes_enqueued);
    // the same tokens un-salted are written
    try std.testing.expect(keepAt(&s, &f, null, &a, 16));
    try std.testing.expectEqual(@as(u64, 1), spy.keeps);
    try std.testing.expectEqual(@as(u64, 1), s.stats.writes_enqueued);
    s.pump(0);
    // the un-salted memory state at 16 shadows its own point; the salted states are not in the un-salted lookup at all
    try std.testing.expect(s.probe(&a, &.{}).durable == null);
    try std.testing.expectEqual(@as(u32, 16), s.probe(&a, &.{}).memory.?.at);
    // a peer's refusal drops that memory copy (released, not forgotten); the point then restores, and never treats a
    // salted state of the same tokens as its twin (the salted ones stay, unchanged)
    const salted0 = s.entries.items[0];
    s.forget(&a, 16);
    try std.testing.expectEqual(@as(u64, 1), spy.releases);
    const r = try s.restoreBegin(s.probe(&a, &.{}).durable.?, &a);
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, r));
    try std.testing.expect(std.mem.indexOfScalar(*pc.Entry, s.entries.items, salted0) != null);
    try std.testing.expect(salted0.namespace != null and salted0.point == null);
    const own = s.find(&a, &.{}).?;
    try std.testing.expect(own.namespace == null and own.restored);
    try std.testing.expect(s.findScoped(ns, &a, &.{}).? != own); // the salted lookup still sees only its own
    try std.testing.expect(s.findScoped(ns, &a, &.{}).?.namespace != null);
    try std.testing.expectEqual(@as(u64, 0), spy.forgets);
}

fn settle(s: *Store, r: *pc.Restore) pc.Restored {
    while (true) {
        const st = s.restorePoll(r, 1000);
        if (st != .pending) return st;
    }
}

test "PETABRIDGE spill P5: dropping memory releases the disk point and never forgets it (evictBudget, prune_dead, dead_at_keep, failed resume, a peer's refusal)" {
    var f: Fam = .{};
    var tier: ft.FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var spy: Spy = .{ .inner = tier.tier() };
    // three states of 116 bytes fit, a fourth does not
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 3 * 116 + 50);
    defer s.deinit();
    s.setTier(spy.tier(), 0);
    s.prune_dead = true;
    const a = seq(40, 1);
    const b = seq(40, 1000);
    const c = seq(40, 2000);
    const d = seq(40, 3000);
    const e = seq(40, 4000);
    try std.testing.expect(keepAt(&s, &f, null, &a, 16));
    try std.testing.expect(keepAt(&s, &f, null, &b, 16));
    try std.testing.expect(keepAt(&s, &f, null, &c, 16));
    const ids = [_]u64{ s.entries.items[0].point.?, s.entries.items[1].point.?, s.entries.items[2].point.? };
    // 1. evictBudget, a valid victim (the oldest, a's): its write was still queued, so the release staged it
    try std.testing.expect(keepAt(&s, &f, null, &d, 16));
    try std.testing.expectEqual(@as(u64, 1), s.audit.valid_budget_evictions);
    try std.testing.expectEqual(@as(u64, 1), spy.releases);
    try std.testing.expectEqual(@as(u64, 1), tier.c.forced_staging);
    try std.testing.expect(tier.point(ids[0]) != null);
    s.pump(0);
    // 2. prune_dead: c's state is dead (the family staled it); under pressure it goes first, released, its point kept
    f.kill(s.entries.items[1].saved); // c's (b's at 0, c's at 1, d's at 2)
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    _ = try s.lookup(arena.allocator(), &e, 16, &.{}, &.{}); // (its reserve for the mark at 16)
    try std.testing.expectEqual(@as(u64, 1), s.audit.dead_at_reserve);
    try std.testing.expectEqual(@as(u64, 2), spy.releases);
    try std.testing.expect(tier.point(ids[2]) != null);
    // 3. dead_at_keep: b's state dies; the next keep drops it first, released, its point kept
    f.kill(s.entries.items[0].saved);
    try std.testing.expect(keepAt(&s, &f, null, &e, 16));
    try std.testing.expectEqual(@as(u64, 1), s.audit.dead_at_keep);
    try std.testing.expectEqual(@as(u64, 3), spy.releases);
    try std.testing.expect(tier.point(ids[1]) != null);
    // 4. a failed resume of a memory state (its copy failed): dropped from memory, released
    const d_entry = s.find(&d, &.{}).?;
    const d_id = d_entry.point.?;
    s.resumed(d_entry, &d, false);
    try std.testing.expectEqual(@as(u64, 4), spy.releases);
    // 5. a peer that cannot resume a state (Store.forget, flashnext's speed-up peer): released too
    s.forget(&e, 16);
    try std.testing.expectEqual(@as(u64, 5), spy.releases);
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    // never a forget: every point is still on the tier, and restores of each come back
    try std.testing.expectEqual(@as(u64, 0), spy.forgets);
    s.pump(0);
    try std.testing.expectEqual(@as(usize, 5), tier.points.items.len);
    try std.testing.expect(tier.point(d_id) != null);
    for ([_][]const u32{ &a, &b, &c }) |p| {
        try std.testing.expectEqual(pc.Restored.adopted, settle(&s, try s.restoreBegin(s.probe(p, &.{}).durable.?, p)));
    }
    try std.testing.expectEqual(@as(u64, 3), s.stats.disk_hits_committed);
    // only forgetPoint forgets (the family's all-rank abort)
    s.forgetPoint(ids[0]);
    try std.testing.expectEqual(@as(u64, 1), spy.forgets);
    try std.testing.expect(tier.point(ids[0]) == null);
    // and a reset is refused while a tier can repopulate memory
    try std.testing.expectError(error.DurableTierEnabled, s.clear());
}

test "PETABRIDGE spill P5: the --learn imprint and the tier never both hold or read one shared cut" {
    var tmp_buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&tmp_buf, "/tmp/tf-p5-imprint-{d}", .{std.c.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(gpa, root, 5, 1 << 30);
    defer im.deinit();
    var f: Fam = .{};
    var tier: ft.FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var spy: Spy = .{ .inner = tier.tier() };
    var s = Store.init(gpa, f.learned(), .{ .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    s.setTier(spy.tier(), 0);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = seq(40, 1);
    // the pass plans 16 as a shared cut and 30 as its history
    const l = try s.lookup(arena.allocator(), &a, 30, &.{16}, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 16, 30 }, l.marks);
    try std.testing.expect(keepAt(&s, &f, null, &a, 16)); // the shared cut: the imprint learns it, the tier does not
    try std.testing.expect(im.has(imprint.Imprint.keyOf(a[0..16])));
    try std.testing.expect(s.entries.items[0].shared and s.entries.items[0].point == null);
    try std.testing.expectEqual(@as(u64, 0), spy.keeps);
    try std.testing.expect(keepAt(&s, &f, null, &a, 30)); // the history's: the tier writes it, the imprint does not
    try std.testing.expectEqual(@as(u64, 1), spy.keeps);
    try std.testing.expect(!im.has(imprint.Imprint.keyOf(a[0..30])));
    // a tier point at a cut the imprint holds (an older process wrote both) is never restored: recall reads it
    _ = try tier.put(a[0..16], 16, std.mem.asBytes(&Fam.State{ .at = 16, .sum = 0 }), 0, 0, true);
    var fresh = Store.init(gpa, f.learned(), .{ .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer fresh.deinit();
    fresh.imprint = &im;
    fresh.setTier(spy.tier(), 0);
    const longer = a ++ [_]u32{ 9, 9 };
    try std.testing.expect(fresh.probe(longer[0..20], &.{}).durable == null);
    const lookups = spy.lookups;
    try std.testing.expect(lookups > 0); // (it asked; the store refused the answer)
    const got = try fresh.lookup(arena.allocator(), longer[0..20], 20, &.{}, &.{});
    try std.testing.expectEqual(@as(u32, 16), got.entry.?.at); // read from the imprint
    try std.testing.expect(got.entry.?.point == null);
    try std.testing.expectEqual(@as(u64, 0), spy.begins);
    // without --learn the same point restores from the tier
    var plain = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer plain.deinit();
    plain.setTier(spy.tier(), 0);
    try std.testing.expectEqual(@as(u32, 16), plain.probe(longer[0..20], &.{}).durable.?.at);
}

const along = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };

test "PETABRIDGE spill P5: a salted request is never parked, never looked up on disk, never written, and joins no conversation" {
    const r = try ht.Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    _ = try r.point(&ht.t2, 8);
    var b: ht.Box = .{};
    defer b.deinit();
    const salted: Request = .{ .prompt = &ht.t2, .max_tokens = 6, .history_len = 14, .cache_namespace = pc.namespaceFor("tenant-b") };
    try r.host.engine().submit(1, &salted, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 0), b.cached); // the point was there: memory-only means a cold pass
    try b.exact(&ht.t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 0), s.disk_lookups);
    try std.testing.expectEqual(@as(u64, 0), s.restore_waits);
    try std.testing.expectEqual(@as(u64, 0), s.restores_begun);
    try std.testing.expectEqual(@as(u64, 0), s.writes_enqueued); // its keep at 14 stays in memory
    try std.testing.expectEqual(@as(u64, 1), s.kept);
    r.stop();
    try std.testing.expectEqual(@as(u64, 0), r.tier.begins);
    try std.testing.expectEqual(@as(usize, 1), r.tier.points.items.len); // only the published point
    try std.testing.expectEqual(@as(usize, 0), r.tier.queue.items.len);
    try std.testing.expectEqual(@as(usize, 1), r.store.entries.items.len);
    const e = r.store.entries.items[0];
    try std.testing.expect(e.namespace != null and e.point == null and e.group == 0 and e.lineage.len == 0);
}

test "PETABRIDGE spill P5: with the 0005 interleave on, a parked restore is polled between another pass's chunks while a decoding stream keeps its tokens" {
    const Pre = struct {
        fn f(rig: *ht.Rig) void {
            rig.host.interleave_ns = 5 * std.time.ns_per_ms;
        }
    };
    // (the boxes outlive the rig: a failed check stops the host, and its streams, before they go)
    var a: ht.Box = .{};
    defer a.deinit();
    var b: ht.Box = .{};
    defer b.deinit();
    var c: ht.Box = .{};
    defer c.deinit();
    const r = try ht.Rig.createWith(.{ .load_ms = 5 }, Pre.f);
    defer r.destroy();
    const e = r.host.engine();
    // A decodes on
    const areq: Request = .{ .prompt = &along, .max_tokens = 100000 };
    try e.submit(1, &areq, a.sink());
    while (a.count() == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    // C parks on a restore that cannot finish yet
    r.tier.gate.store(false, .release);
    _ = try r.point(&ht.t2, 8);
    const creq: Request = .{ .prompt = &ht.t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(2, &creq, c.sink());
    try r.until(struct {
        fn ok(st: api.CacheStats) bool {
            return st.restore_waits == 1;
        }
    }.ok);
    // B's six-chunk pass: at each chunk, A's tokens and the tier's polls so far (the engine thread's)
    const Watch = struct {
        tier: *ft.FakeTier,
        a: *ht.Box,
        polled: [6]u64 = @splat(0),
        tokens: [6]usize = @splat(0),
        fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len != 10 or chunk >= w.polled.len) return;
            w.polled[chunk] = @atomicLoad(u64, &w.tier.polled, .monotonic);
            w.tokens[chunk] = w.a.count();
        }
    };
    var watch: Watch = .{ .tier = &r.tier, .a = &a };
    r.target.prefill_chunks = 6;
    r.target.prefill_hook = Watch.call;
    r.target.prefill_hook_ctx = &watch;
    const other = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8 };
    const breq: Request = .{ .prompt = &other, .max_tokens = 1 };
    try e.submit(3, &breq, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    // C's own pass in one chunk: the fake backend reads a resumed state after its chunks, where dsv41 copies it in
    // before its first one (and between() reports the resumed entry, freeing the stream's hold on it)
    r.target.prefill_chunks = 1;
    r.target.prefill_hook = null;
    try std.testing.expect(watch.tokens[5] > watch.tokens[0]); // the interleave's rounds ran inside B's pass
    try std.testing.expect(watch.polled[5] > watch.polled[0]); // and so did the restore's service, after them
    try std.testing.expect(c.finished() == null); // C is still parked, holding no lane
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, c.wait());
    try std.testing.expectEqual(@as(?u32, 8), c.cached);
    try c.exact(&ht.t2);
    try b.exact(&other);
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
}

test "PETABRIDGE spill P5: without a tier the host and the store change nothing (no status counters, no parking, no between hook without the interleave)" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const fk: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return fk.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const fk: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            fk.drop(saved);
        }
    };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = lh_init(&core);
    host.cache = &store;
    host.configure(.{ .dir = "/x", .gib = 1 }); // asked, never bound: still off
    try host.start();
    defer host.stop();
    const Hook = struct {
        between_seen: bool = false,
        fn call(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            if (s.between != null) w.between_seen = true;
        }
    };
    var hook: Hook = .{};
    target.prefill_chunks = 4;
    target.prefill_hook = Hook.call;
    target.prefill_hook_ctx = &hook;
    var b: ht.Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &ht.t2, .max_tokens = 3, .history_len = 14 };
    try host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expect(!hook.between_seen); // 0005's rule: no hook with the interleave off
    var st: api.Status = .{};
    host.engine().status(&st, &.{});
    try std.testing.expect(st.cache == null);
    try std.testing.expectEqual(pc.Stats{}, store.stats);
    try std.testing.expect(!store.groups and store.restoring.items.len == 0 and store.roots.count() == 0);
    for (store.entries.items) |e| try std.testing.expect(e.point == null and e.group == 0 and e.lineage.len == 0 and !e.restored);
    try std.testing.expectEqual(@as(usize, 0), host.ready.items.len + host.loading.items.len);
}

fn lh_init(core: *lanes.Engine) @import("lane_host.zig").LaneHost {
    var h = @import("lane_host.zig").LaneHost.init(gpa, io, core, .{ .lanes = 2 });
    h.interleave_ns = 0; // (the environment's TF_DS_PREFILL_INTERLEAVE_MS aside)
    return h;
}
