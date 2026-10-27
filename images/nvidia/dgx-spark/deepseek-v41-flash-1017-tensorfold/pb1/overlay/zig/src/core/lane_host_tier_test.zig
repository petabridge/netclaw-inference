//! LaneHost with a durable tier (wojo's GLM 0110's scheduler hooks): requests parked on restores, their service
//! between rounds, between prompt chunks (the interleave off) and at idle, cancellation and shutdown, through the fake
//! lane backend and an in-memory tier.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const lh = @import("lane_host.zig");
const ft = @import("prompt_cache_fake_tier.zig");
const LaneHost = lh.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;
const gpa = std.testing.allocator;
const io = std.testing.io;

pub const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    done: ?Reason = null,

    pub fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .prefilled => |c| b.cached = c,
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| b.done = f.reason,
        }
    }
    pub fn sink(b: *Box) api.Sink {
        return .{ .ctx = b, .event = event };
    }
    pub fn wait(b: *Box) Reason {
        while (true) {
            b.mutex.lockUncancelable(io);
            const d = b.done;
            b.mutex.unlock(io);
            if (d) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
    }
    pub fn deinit(b: *Box) void {
        b.tokens.deinit(gpa);
    }
    pub fn finished(b: *Box) ?Reason {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.done;
    }
    pub fn count(b: *Box) usize {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.tokens.items.len;
    }
    /// The reply equals the fake target's over `prompt` (any slip in the restored history changes it).
    pub fn exact(b: *Box, prompt: []const u32) !void {
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, prompt);
        for (b.tokens.items) |t| {
            try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
            try history.append(gpa, t);
        }
    }
};

/// The fake backend, a prompt cache over its states, an in-memory tier and the host, at stable addresses.
pub const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    store: pc.Store,
    tier: ft.FakeTier,
    host: LaneHost,
    stopped: bool = false,

    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        f.drop(saved);
    }
    fn pack(_: *anyopaque, a: std.mem.Allocator, saved: pc.Saved) anyerror![]u8 {
        const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
        return a.dupe(u8, std.mem.sliceAsBytes(kept.items));
    }
    fn unpack(_: *anyopaque, b: []const u8) anyerror!pc.Saved {
        const kept = try gpa.create(std.ArrayList(u32));
        kept.* = .empty;
        const n = b.len / 4;
        try kept.resize(gpa, n);
        @memcpy(std.mem.sliceAsBytes(kept.items), b[0 .. n * 4]);
        return kept;
    }

    pub fn create(sp: api.Spill) !*Rig {
        return createWith(sp, null);
    }

    /// `pre` runs on the host before its thread starts (the robustness tests' policies).
    pub fn createWith(sp: api.Spill, pre: ?*const fn (*Rig) void) !*Rig {
        const r = try gpa.create(Rig);
        errdefer gpa.destroy(r);
        r.* = .{ .cfg = undefined, .target = .{ .gpa = gpa }, .core = undefined, .store = undefined, .tier = .{ .gpa = gpa, .codec = .{ .ctx = r, .pack = pack, .unpack = unpack } }, .host = undefined };
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
        r.store = pc.Store.init(gpa, .{ .ptr = &r.target, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
        r.store.setTier(r.tier.tier(), 0);
        r.host = LaneHost.init(gpa, io, &r.core, .{ .lanes = 2 });
        r.host.interleave_ns = 0;
        r.host.cache = &r.store;
        r.host.configure(sp);
        if (pre) |f| f(r);
        try r.host.start();
        return r;
    }

    pub fn stop(r: *Rig) void {
        if (r.stopped) return;
        r.stopped = true;
        r.host.stop();
    }

    pub fn destroy(r: *Rig) void {
        r.stop();
        r.store.deinit();
        r.tier.deinit();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(gpa);
        gpa.destroy(r);
    }

    /// A published point of `key[0..at]` whose state is those tokens (the fake backend's state is its history).
    pub fn point(r: *Rig, key: []const u32, at: u32) !u64 {
        return r.tier.put(key[0..at], at, std.mem.sliceAsBytes(key[0..at]), 0, 0, false);
    }

    pub fn stats(r: *Rig) api.CacheStats {
        var st: api.Status = .{};
        r.host.engine().status(&st, &.{});
        return st.cache.?;
    }

    /// Waits (5 s at most) until the published counters satisfy `ok`.
    pub fn until(r: *Rig, comptime ok: fn (api.CacheStats) bool) !void {
        for (0..5000) |_| {
            if (ok(r.stats())) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

pub const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
pub const t2 = t1 ++ [_]u32{ 7, 7, 5, 3, 8, 8 };

test "a turn whose state left memory waits parked for its restore, resumes it exactly, and writes its own keeps forked from it" {
    const r = try Rig.create(.{ .load_ms = 5, .flush_ms = 1000 });
    defer r.destroy();
    r.tier.polls = 3;
    const id = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.writes_enqueued == 1;
        }
    }.ok);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restore_waits);
    try std.testing.expectEqual(@as(u64, 1), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 8), s.restored_tokens);
    try std.testing.expectEqual(@as(u64, 1), s.memory_hits);
    try std.testing.expectEqual(@as(u64, 8), s.cached_tokens);
    try std.testing.expectEqual(@as(u64, 8), s.computed_tokens);
    try std.testing.expectEqual(@as(u64, 0), s.parked);
    try std.testing.expect(s.spill != null);
    r.stop();
    try std.testing.expectEqual(@as(u64, 1), r.tier.flushed); // the clean shutdown drained the tier
    const kept = for (r.tier.points.items) |p| {
        if (p.at == 14) break p;
    } else return error.TestUnexpectedResult;
    try std.testing.expectEqual(id, kept.parent.?); // its lineage forks the restored point at the resume point
    try std.testing.expectEqual(@as(u32, 8), kept.fork);
    try std.testing.expect(kept.group != 0); // a conversation of its own (the restored state is no turn's own)
    try std.testing.expectEqual(@as(usize, 1), kept.lineage.len); // the point carried none: a lineage of its own
}

test "two requests on one point share its restore; it stops once, when the last of them is cancelled" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const e = r.host.engine();
    try e.submit(1, &req, a.sink());
    try e.submit(2, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 2;
        }
    }.ok);
    var st: api.Status = .{};
    e.status(&st, &.{});
    try std.testing.expectEqual(@as(u32, 2), st.waiting);
    try std.testing.expectEqual(@as(u32, 0), st.running);
    try std.testing.expectEqual(@as(u64, 1), st.cache.?.restores_begun);
    try std.testing.expectEqual(@as(u64, 1), st.cache.?.restores_shared);
    try std.testing.expectEqual(@as(u64, 2), st.cache.?.parked);
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
    try std.testing.expectEqual(@as(u64, 1), r.stats().restores_active); // still wanted by the other
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, b.wait());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_cancelled == 1 and s.restores_active == 0 and s.parked == 0;
        }
    }.ok);
    r.stop();
    try std.testing.expectEqual(@as(u64, 1), r.tier.cancels);
    try std.testing.expectEqual(@as(u64, 0), r.tier.misuse);
    try std.testing.expectEqual(@as(usize, 0), r.tier.jobs.items.len);
}

test "a restore that fails on any rank prefills from the start, exactly" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.fail_next = error.CrcMismatch;
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 0), b.cached);
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_failed);
    try std.testing.expectEqual(@as(u64, 0), s.disk_hits_committed);
}

test "restores progress between another request's prompt chunks with the interleave off, then their request resumes" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const e = r.host.engine();
    try e.submit(1, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
    const Watch = struct {
        tier: *ft.FakeTier,
        seen: [6]u64 = @splat(0),
        fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len == 12 and chunk < w.seen.len) w.seen[chunk] = w.tier.polled; // (the engine thread's)
        }
    };
    var watch: Watch = .{ .tier = &r.tier };
    r.target.prefill_chunks = 6;
    r.target.prefill_hook = Watch.call;
    r.target.prefill_hook_ctx = &watch;
    const other = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    var o: Box = .{};
    defer o.deinit();
    const oreq: Request = .{ .prompt = &other, .max_tokens = 1 };
    try e.submit(2, &oreq, o.sink());
    try std.testing.expectEqual(Reason.length, o.wait());
    try std.testing.expect(watch.seen[5] > watch.seen[0]); // polled inside the other's pass, between its chunks
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
}

test "shutdown with requests parked on a restore that never ends: they are cancelled, the restore stops once, the tier flushes" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, a.sink());
    try r.host.engine().submit(2, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 2;
        }
    }.ok);
    r.stop();
    try std.testing.expectEqual(Reason.cancelled, a.wait());
    try std.testing.expectEqual(Reason.cancelled, b.wait());
    try std.testing.expectEqual(@as(u64, 1), r.tier.cancels);
    try std.testing.expectEqual(@as(u64, 0), r.tier.misuse);
    try std.testing.expectEqual(@as(u64, 1), r.tier.flushed);
}

test "without a tier nothing is parked and status carries no cache counters" {
    const gpa_ = gpa;
    var cfg = try lanes.Config.init(gpa_, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa_);
    var target: lanes.fake.Fake = .{ .gpa = gpa_ };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa_, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var store = pc.Store.init(gpa_, .{ .ptr = &target, .vtable = &.{ .bytes = Rig.bytes, .save = Rig.save, .restore = Rig.restore, .drop = Rig.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa_, io, &core, .{ .lanes = 2 });
    host.cache = &store;
    host.configure(.{});
    try host.start();
    defer host.stop();
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 3, .history_len = 14 };
    try host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    var st: api.Status = .{};
    host.engine().status(&st, &.{});
    try std.testing.expect(st.cache == null);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expectEqual(@as(u64, 0), store.entries.items[0].group);
}

test "a request parked on a restore past TF_SPILL_LOAD_WAIT_MS is admitted without it, and the restore stops" {
    const r = try Rig.create(.{ .load_ms = 5, .load_wait_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 0), b.cached);
    try b.exact(&t2);
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_wait_timeouts == 1 and s.restores_cancelled == 1 and s.restores_active == 0 and s.restoring_bytes == 0;
        }
    }.ok);
    r.stop();
    try std.testing.expectEqual(@as(u64, 1), r.tier.cancels);
    try std.testing.expectEqual(@as(u64, 0), r.tier.misuse);
}

test "restores are serviced between decode rounds while another stream decodes" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    const e = r.host.engine();
    const along = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    var a: Box = .{};
    defer a.deinit();
    const areq: Request = .{ .prompt = &along, .max_tokens = 100000 };
    try e.submit(1, &areq, a.sink());
    while (a.count() == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(2, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
    const p0 = @atomicLoad(u64, &r.tier.polled, .monotonic);
    const n0 = a.count();
    var grew = false;
    for (0..5000) |_| {
        if (a.count() > n0 + 32 and @atomicLoad(u64, &r.tier.polled, .monotonic) > p0 + 8) {
            grew = true;
            break;
        }
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(grew); // polled round after round while the core is never idle
    try std.testing.expect(a.finished() == null);
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
}

test "spill switches a backend never bound are refused, not ignored" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var tier: ft.FakeTier = .{ .gpa = gpa, .codec = .{ .ctx = &target, .pack = Rig.pack, .unpack = Rig.unpack } };
    defer tier.deinit();
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Rig.bytes, .save = Rig.save, .restore = Rig.restore, .drop = Rig.drop } }, .{}, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, io, &core, .{ .lanes = 2 });
    const disk: api.Spill = .{ .dir = "/x", .gib = 1 };
    try std.testing.expect(!(api.Spill{}).unbound(host.engine())); // nothing asked
    try std.testing.expect(disk.unbound(host.engine())); // asked, never configured
    host.configure(disk);
    try std.testing.expect(disk.unbound(host.engine())); // configured without a prompt cache: no tier to serve it
    host.cache = &store;
    host.configure(disk);
    try std.testing.expect(disk.unbound(host.engine())); // a prompt cache the backend bound no tier to
    store.setTier(tier.tier(), 0);
    host.configure(disk);
    try std.testing.expect(!disk.unbound(host.engine()));
}
