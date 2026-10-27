//! An in-memory prompt-cache Tier for tests (no disk, no CUDA; one lock for the engine thread and the test's): writes are queued at keep and published
//! by `pump`, a restore needs `polls` polls, and every misuse (a job cancelled twice, a poll of a gone job) is recorded.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Allocator = std.mem.Allocator;

/// How the fake turns a family's state into bytes and back (the family's packing, in round 2).
pub const Codec = struct {
    ctx: *anyopaque,
    pack: *const fn (ctx: *anyopaque, a: Allocator, saved: pc.Saved) anyerror![]u8,
    unpack: *const fn (ctx: *anyopaque, bytes: []const u8) anyerror!pc.Saved,
};

pub const FakeTier = struct {
    gpa: Allocator,
    codec: Codec,
    points: std.ArrayList(Pt) = .empty,
    queue: std.ArrayList(Pending) = .empty,
    jobs: std.ArrayList(Job) = .empty,
    next_id: u64 = 1,
    next_job: u64 = 1,
    /// Polls a restore takes before it is ready (1: the first).
    polls: u32 = 1,
    /// The next restore fails with this error (on its first poll).
    fail_next: ?anyerror = null,
    /// The next `refuse` begins are refused with `refuse_err` before anything is reserved (a tier at its capacity:
    /// dsv41's TooManyRestores, the store's TooMany/NoSlot); `refused` counts them.
    refuse: u32 = 0,
    refuse_err: anyerror = error.TooManyRestores,
    refused: u64 = 0,
    begins: u64 = 0,
    /// The next `full` asks of Tier.VTable.full answer true (the tier at its cap before any room is made; `fulls`
    /// counts the asks).
    full: u32 = 0,
    fulls: u64 = 0,
    /// Shorter states are not written.
    min_tokens: u32 = 0,
    /// A flush cannot drain: its queue is left unpublished.
    stuck: bool = false,
    /// Lookups ignore `beyond`, `max_at` and `grid` (the store must check the answer itself).
    lax: bool = false,
    /// While false every poll stays pending (another thread opens it).
    gate: std.atomic.Value(bool) = .init(true),
    c: pc.TierCounts = .{},
    polled: u64 = 0,
    cancels: u64 = 0,
    misuse: u64 = 0, // a cancel or poll of a job that is not pending
    touched: u64 = 0,
    released: u64 = 0,
    shared_marks: u64 = 0,
    flushed: u64 = 0,
    /// Ranks whose counters `counts` reports (rank r's `crc_errors` is r, to tell them apart).
    ranks: u32 = 1,
    /// The engine thread's calls and the test thread's `put`/`point` (a test adds a point while the host still
    /// publishes an earlier keep: two appends at once lost one list, a leak the allocator caught now and then).
    mu: std.Io.Mutex = .init,

    pub const Pt = struct { id: u64, key: []u32, at: u32, group: u64, shared: bool, bytes: []u8, parent: ?u64, fork: u32, size: u64, lineage: []pc.Fork = &.{} };
    pub const Pending = struct { id: u64, key: []u32, at: u32, group: u64, shared: bool, saved: pc.Saved, parent: ?u64, fork: u32, size: u64, lineage: []pc.Fork };
    const Job = struct { id: u64, point: u64, left: u32, fail: ?anyerror };

    pub fn deinit(t: *FakeTier) void {
        for (t.points.items) |p| {
            t.gpa.free(p.key);
            t.gpa.free(p.bytes);
            t.gpa.free(p.lineage);
        }
        for (t.queue.items) |q| {
            t.gpa.free(q.key);
            t.gpa.free(q.lineage);
        }
        t.points.deinit(t.gpa);
        t.queue.deinit(t.gpa);
        t.jobs.deinit(t.gpa);
    }

    pub fn tier(t: *FakeTier) pc.Tier {
        // PETABRIDGE (k2a-pb1) spill P5: the tip's `.full = isFull` is left out: Tier.full is 0020's (TF_SPILL_DEFER_MS),
        // not ported. isFull, `full` and `fulls` stay, unused, so 0020 can wire them back unchanged.
        return .{ .ptr = t, .vtable = &.{ .lookup = lookup, .begin = begin, .poll = poll, .cancel = cancel, .keep = keep, .release = release, .share = share, .touch = touch, .forget = forget, .pump = pump, .busy = busy, .flush = flush, .counts = counts } };
    }

    fn lock(t: *FakeTier) void {
        t.mu.lockUncancelable(std.testing.io);
    }

    fn unlock(t: *FakeTier) void {
        t.mu.unlock(std.testing.io);
    }

    fn of(ptr: *anyopaque) *FakeTier {
        return @ptrCast(@alignCast(ptr));
    }

    /// A published point of `key` (its ids, `at` plus lookahead) whose state packs to `bytes`: its name.
    pub fn put(t: *FakeTier, key: []const u32, at: u32, bytes: []const u8, size: u64, group: u64, shared: bool) !u64 {
        t.lock();
        defer t.unlock();
        const k = try t.gpa.dupe(u32, key);
        errdefer t.gpa.free(k);
        const b = try t.gpa.dupe(u8, bytes);
        errdefer t.gpa.free(b);
        const id = t.next_id;
        try t.points.append(t.gpa, .{ .id = id, .key = k, .at = at, .group = group, .shared = shared, .bytes = b, .parent = null, .fork = 0, .size = size });
        t.next_id += 1;
        return id;
    }

    /// The published point named `id`, if any.
    pub fn point(t: *FakeTier, id: u64) ?*Pt {
        t.lock();
        defer t.unlock();
        return t.find(id);
    }

    fn find(t: *FakeTier, id: u64) ?*Pt {
        for (t.points.items) |*p| if (p.id == id) return p;
        return null;
    }

    fn lookup(ptr: *anyopaque, want: *const pc.Want) ?pc.Point {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        var best: ?*Pt = null;
        for (t.points.items) |*p| {
            if (p.key.len > want.key.len or !std.mem.eql(u32, p.key, want.key[0..p.key.len])) continue;
            if (!t.lax and (p.at <= want.beyond or p.at > want.max_at or (want.grid > 0 and p.at % want.grid != 0))) continue;
            if (best == null or p.at > best.?.at) best = p;
        }
        const p = best orelse {
            t.c.compatible_misses += 1;
            return null;
        };
        return .{ .id = p.id, .at = p.at, .tokens = @intCast(p.key.len), .bytes = p.size, .group = p.group, .shared = p.shared };
    }

    fn isFull(ptr: *anyopaque) bool {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        t.fulls += 1;
        if (t.full == 0) return false;
        t.full -= 1;
        return true;
    }

    fn begin(ptr: *anyopaque, p: *const pc.Point) anyerror!u64 {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        t.begins += 1;
        if (t.refuse > 0) {
            t.refuse -= 1;
            t.refused += 1;
            return t.refuse_err;
        }
        if (t.find(p.id) == null) {
            t.c.presence_refused += 1;
            return error.PointMissing;
        }
        const id = t.next_job;
        t.next_job += 1;
        try t.jobs.append(t.gpa, .{ .id = id, .point = p.id, .left = @max(t.polls, 1), .fail = t.fail_next });
        t.fail_next = null;
        return id;
    }

    fn job(t: *FakeTier, id: u64) ?usize {
        for (t.jobs.items, 0..) |j, i| if (j.id == id) return i;
        return null;
    }

    fn poll(ptr: *anyopaque, id: u64, _: u64) pc.Poll {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        t.polled += 1;
        const i = t.job(id) orelse {
            t.misuse += 1;
            return .{ .failed = error.NoSuchJob };
        };
        const j = &t.jobs.items[i];
        if (j.fail) |err| {
            _ = t.jobs.orderedRemove(i);
            t.c.read_failed += 1;
            if (err == error.CrcMismatch) t.c.crc_errors += 1;
            return .{ .failed = err };
        }
        if (!t.gate.load(.acquire)) return .{ .pending = .{ .total = @max(t.polls, 1) } };
        j.left -= 1;
        if (j.left > 0) return .{ .pending = .{ .done = @max(t.polls, 1) - j.left, .total = @max(t.polls, 1) } };
        const pid = j.point;
        _ = t.jobs.orderedRemove(i);
        const p = t.find(pid) orelse return .{ .failed = error.PointMissing };
        const saved = t.codec.unpack(t.codec.ctx, p.bytes) catch |err| return .{ .failed = err };
        t.c.reads += 1;
        t.c.read_bytes += p.bytes.len;
        return .{ .ready = .{ .saved = saved, .bytes = p.size, .lineage = p.lineage } };
    }

    fn cancel(ptr: *anyopaque, id: u64) void {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        t.cancels += 1;
        const i = t.job(id) orelse {
            t.misuse += 1;
            return;
        };
        _ = t.jobs.orderedRemove(i);
    }

    fn keep(ptr: *anyopaque, k: *const pc.Kept) ?u64 {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        if (k.at < t.min_tokens) return null;
        const key = t.gpa.dupe(u32, k.key) catch return null;
        const lin = t.gpa.dupe(pc.Fork, k.lineage) catch {
            t.gpa.free(key);
            return null;
        };
        const id = t.next_id;
        t.queue.append(t.gpa, .{ .id = id, .key = key, .at = k.at, .group = k.group, .shared = k.shared, .saved = k.saved, .parent = k.parent, .fork = k.fork, .size = k.bytes, .lineage = lin }) catch {
            t.gpa.free(key);
            t.gpa.free(lin);
            return null;
        };
        t.next_id += 1;
        t.c.writes_queued += 1;
        return id;
    }

    /// Queued write `i` published (its state packed now).
    fn publish(t: *FakeTier, i: usize) void {
        const q = t.queue.orderedRemove(i);
        const bytes = t.codec.pack(t.codec.ctx, t.gpa, q.saved) catch {
            t.gpa.free(q.key);
            t.gpa.free(q.lineage);
            t.c.writes_failed += 1;
            return;
        };
        t.points.append(t.gpa, .{ .id = q.id, .key = q.key, .at = q.at, .group = q.group, .shared = q.shared, .bytes = bytes, .parent = q.parent, .fork = q.fork, .size = q.size, .lineage = q.lineage }) catch {
            t.gpa.free(q.key);
            t.gpa.free(q.lineage);
            t.gpa.free(bytes);
            t.c.writes_failed += 1;
            return;
        };
        t.c.writes_committed += 1;
        t.c.written_bytes += bytes.len;
    }

    fn release(ptr: *anyopaque, id: u64, _: pc.Saved) void {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        t.released += 1;
        for (t.queue.items, 0..) |q, i| if (q.id == id) {
            t.c.forced_staging += 1;
            return t.publish(i);
        };
    }

    fn share(ptr: *anyopaque, id: u64) void {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        t.shared_marks += 1;
        if (t.find(id)) |p| p.shared = true;
        for (t.queue.items) |*q| if (q.id == id) {
            q.shared = true;
        };
    }

    fn touch(ptr: *anyopaque, _: u64) void {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        t.touched += 1;
    }

    fn forget(ptr: *anyopaque, id: u64) void {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        for (t.points.items, 0..) |p, i| if (p.id == id) {
            t.gpa.free(p.key);
            t.gpa.free(p.bytes);
            t.gpa.free(p.lineage);
            _ = t.points.orderedRemove(i);
            return;
        };
    }

    fn pump(ptr: *anyopaque, _: u64) void {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        while (t.queue.items.len > 0) t.publish(0);
    }

    fn busy(ptr: *anyopaque) bool {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        return t.queue.items.len > 0;
    }

    fn flush(ptr: *anyopaque, _: u64) u64 {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        t.flushed += 1;
        if (!t.stuck) {
            while (t.queue.items.len > 0) t.publish(0); // (pump's, under this lock)
            return 0;
        }
        const left = t.queue.items.len;
        for (t.queue.items) |q| {
            t.gpa.free(q.key);
            t.gpa.free(q.lineage);
        }
        t.queue.clearRetainingCapacity();
        t.c.writes_cancelled += left;
        return left;
    }

    fn counts(ptr: *anyopaque, out: []pc.TierCounts) usize {
        const t = of(ptr);
        t.lock();
        defer t.unlock();
        var c = t.c;
        c.points = t.points.items.len;
        c.writes_pending = t.queue.items.len;
        const n = @min(out.len, t.ranks);
        for (out[0..n], 0..) |*o, r| {
            o.* = c;
            if (r > 0) o.crc_errors = r;
        }
        return n;
    }
};
