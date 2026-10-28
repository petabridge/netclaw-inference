//! PETABRIDGE (k2a-pb1) spill P6: host tests of the dsv41 wiring that need no GPU: the grid realignment's arithmetic
//! (lanes.chunkLen, spillChunk, spillGridKeep: plan 3.4) over prefillFn's own chunk rule, and the disk eligibility rule
//! (lanes.spillEligible). The tier itself is tested in spill_tier_test.zig (loopback, worlds 2 and 3).
const std = @import("std");
const lanes = @import("lanes.zig");
const model = @import("model.zig");

const testing = std.testing;
const G = model.chunk_rows;
const window = 128; // the checkpoint's replay window (config: window)

/// prefillFn's walk of one pass (its loop over chunkLen), from `start` to `len`: the chunk ends in `ends`, and
/// whether the keep hook's grid point (spillGridKeep) was one of them. `realign`: the tier on, a text prompt.
const Walk = struct { ends: [4096]usize = undefined, n: usize = 0, grid_keep: usize = 0, grid_keep_hit: bool = false };

fn walk(start0: usize, len: usize, realign: bool, min_tokens: u32) Walk {
    var w: Walk = .{};
    const replay = len -| window;
    const last_end = std.mem.alignBackward(usize, @min(replay, len - 1), 2);
    w.grid_keep = if (realign) lanes.spillGridKeep(start0, last_end, min_tokens) else 0;
    var start = start0;
    while (start < len) {
        const n = lanes.chunkLen(start, len, last_end, realign);
        std.debug.assert(n > 0 and n <= G);
        start += n;
        w.ends[w.n] = start;
        w.n += 1;
        if (start == w.grid_keep and start < len and start <= replay) w.grid_keep_hit = true;
    }
    return w;
}

/// k2a's loop as it was before P6 (the reference for "tier off: unchanged").
fn k2aChunk(start: usize, len: usize, last_end: usize) usize {
    const end = if (start < last_end) @min(len, last_end) else len;
    return @min(G, end - start);
}

test "PETABRIDGE P6 grid realignment: tier off is k2a's chunking exactly; on, after an off-grid resume every chunk end before last_end is on the 2,048 grid" {
    var prng = std.Random.DefaultPrng.init(0x5f11_9e6);
    const rnd = prng.random();
    var cases: usize = 0;
    var realigned: usize = 0;
    var kept_grid: usize = 0;
    while (cases < 20000) : (cases += 1) {
        const len = rnd.intRangeAtMost(usize, 2, 300_000);
        const replay = len -| window;
        const last_end = std.mem.alignBackward(usize, @min(replay, len - 1), 2);
        // k2a resumes at any even position before the replay (or 0: a fresh pass)
        const start0 = if (rnd.boolean() or replay < 2) 0 else std.mem.alignBackward(usize, rnd.uintLessThan(usize, replay), 2);
        // 1. tier off: chunkLen(.., false) is k2a's rule, chunk for chunk
        var s = start0;
        while (s < len) {
            const a = lanes.chunkLen(s, len, last_end, false);
            try testing.expectEqual(k2aChunk(s, len, last_end), a);
            s += a;
        }
        const off = walk(start0, len, false, 8192);
        const on = walk(start0, len, true, 8192);
        try testing.expectEqual(@as(usize, 0), off.grid_keep);
        // 2. tier on: the pass covers the same rows, every chunk at most chunk_rows
        try testing.expectEqual(len, on.ends[on.n - 1]);
        // 3. an on-grid resume (0 included) or one at or past last_end: no change at all
        if (start0 % G == 0 or start0 >= last_end) {
            try testing.expectEqual(off.n, on.n);
            try testing.expectEqualSlices(usize, off.ends[0..off.n], on.ends[0..on.n]);
        } else {
            realigned += 1;
            // the first chunk ends at the next grid boundary or at last_end, whichever comes first
            try testing.expectEqual(@min(std.mem.alignForward(usize, start0, G), last_end, start0 + G), on.ends[0]);
            // one extra chunk at most
            try testing.expect(on.n <= off.n + 1);
        }
        // 4. every chunk end before last_end is on the grid once the first chunk is done
        for (on.ends[0..on.n]) |e| if (e < last_end) try testing.expect(e % G == 0);
        // 5. the grid point at or before last_end, when kept, is one of the pass's chunk ends (the keep hook fires)
        if (on.grid_keep != 0) {
            kept_grid += 1;
            try testing.expect(on.grid_keep_hit);
            try testing.expect(on.grid_keep % G == 0 and on.grid_keep > start0 and on.grid_keep < last_end and on.grid_keep >= 8192);
            try testing.expect(on.grid_keep <= len -| window); // a pre-replay point: spill.eligible's rule
        }
    }
    try testing.expect(realigned > 1000 and kept_grid > 1000);
}

test "PETABRIDGE P6 grid realignment: the plan's examples" {
    // a resume at 5,000 of a 100,000-token prompt: 1,144 rows to 6,144, then whole chunks on the grid
    const last_end = std.mem.alignBackward(usize, 100_000 - window, 2);
    try testing.expectEqual(@as(usize, 1144), lanes.chunkLen(5000, 100_000, last_end, true));
    try testing.expectEqual(@as(usize, 2048), lanes.chunkLen(5000, 100_000, last_end, false));
    try testing.expectEqual(@as(usize, 2048), lanes.chunkLen(6144, 100_000, last_end, true));
    // the grid keep: alignBackward(last_end, 2048), past the resume point, under last_end and at least min_tokens
    try testing.expectEqual(@as(usize, 98304), lanes.spillGridKeep(5000, last_end, 8192));
    try testing.expectEqual(@as(usize, 0), lanes.spillGridKeep(98304, last_end, 8192)); // (the resume point itself)
    try testing.expectEqual(@as(usize, 0), lanes.spillGridKeep(0, 8000, 8192)); // under min_tokens
    try testing.expectEqual(@as(usize, 0), lanes.spillGridKeep(0, 4096, 0)); // last_end on the grid: k2a keeps it already
    try testing.expectEqual(@as(usize, 4096), lanes.spillGridKeep(0, 4100, 0));
    // spillChunk alone: on the grid or at or past last_end, unchanged
    try testing.expectEqual(@as(usize, 2048), lanes.spillChunk(4096, 2048, 10_000));
    try testing.expectEqual(@as(usize, 100), lanes.spillChunk(10_000, 100, 10_000));
    try testing.expectEqual(@as(usize, 2), lanes.spillChunk(4094, 2048, 10_000));
    try testing.expectEqual(@as(usize, 6), lanes.spillChunk(4090, 6, 4096)); // already cut at last_end
}

test "PETABRIDGE P6 disk eligibility: only plain pre-replay grid states; complete, decoded, continuation, loading, stale and off-grid stay memory-only" {
    var sn: lanes.Snap = .{ .id = 1, .at = 4 * G, .base = 0, .buf = undefined };
    try testing.expect(lanes.spillEligible(&sn));
    sn.complete = true;
    try testing.expect(!lanes.spillEligible(&sn));
    sn.complete = false;
    sn.decoded = true; // a post-decode continuation (releaseFn's keep) or a complete prompt
    try testing.expect(!lanes.spillEligible(&sn));
    sn.decoded = false;
    sn.loading = true;
    try testing.expect(!lanes.spillEligible(&sn));
    sn.loading = false;
    sn.stale = true;
    try testing.expect(!lanes.spillEligible(&sn));
    sn.stale = false;
    for ([_]usize{ 0, 4 * G - 2, 4 * G + 2, 1000 }) |at| {
        sn.at = at;
        try testing.expect(!lanes.spillEligible(&sn)); // off the grid (k2a's even points) or nothing
    }
    sn.at = G;
    try testing.expect(lanes.spillEligible(&sn));
    // a restored state may be resumed and kept again later, but it is no new disk point by itself: the prompt cache
    // adopts it without a keep, so only the flags above decide
    sn.restored = true;
    try testing.expect(lanes.spillEligible(&sn));
}
