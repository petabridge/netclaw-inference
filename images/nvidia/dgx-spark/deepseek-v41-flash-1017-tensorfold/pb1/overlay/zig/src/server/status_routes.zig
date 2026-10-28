//! GET /health, /v1/models, /metrics, and the opt-in dashboard.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const live = @import("live.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// Experimental benchmark controls are opt-in and loopback-bound in serve.run.
pub fn benchmark(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.benchmark_control) return routes.unknown(conn, a);
    var state: api.Status = .{};
    srv.engine.status(&state, &.{});
    const body = std.fmt.allocPrint(a, "{{\"schema\":\"tensorfold.benchmark-control.v1\",\"engine_generation\":\"{d}\",\"running\":{d},\"waiting\":{d},\"preparing\":{d},\"reset_supported\":{s},\"cache_salt_supported\":{s}}}", .{ srv.generation_ns, state.running, state.waiting, srv.preparing.load(.acquire), if (srv.engine.vtable.reset_cache != null) "true" else "false", if (srv.info.cache_salt) "true" else "false" }) catch return;
    conn.sendJson(200, body);
}

pub fn resetPrefix(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.benchmark_control) return routes.unknown(conn, a);
    if (srv.preparing.load(.acquire) != 0) return conn.sendJson(409, "{\"success\":false,\"error\":\"requests_preparing\"}");
    srv.engine.resetCache() catch |err| {
        const code: u16 = switch (err) {
            error.Busy => 409,
            error.Closed => 503,
            error.Unsupported => 501,
        };
        const body = std.fmt.allocPrint(a, "{{\"success\":false,\"error\":\"{s}\"}}", .{@errorName(err)}) catch return;
        return conn.sendJson(code, body);
    };
    const body = std.fmt.allocPrint(a, "{{\"success\":true,\"engine_generation\":\"{d}\",\"scope\":\"in-memory reusable prefix states; learned disk states refused\"}}", .{srv.generation_ns}) catch return;
    conn.sendJson(200, body);
}

/// With keys on, health names nothing about the model.
pub fn health(srv: *Server, conn: *Conn, a: Allocator) !void {
    if (srv.keys) |k| if (k.enabled()) return conn.sendJson(200, "{\"status\": \"ok\"}");
    const o = try json.newObject(a);
    try o.put(a, "status", .{ .string = "ok" });
    try o.put(a, "model", .{ .string = srv.config.served_name });
    const ids = try a.alloc(Value, srv.config.model_ids.len);
    for (srv.config.model_ids, ids) |id, *slot| slot.* = .{ .string = id };
    try o.put(a, "model_ids", .{ .array = ids });
    try o.put(a, "max_batch_size", try json.intValue(a, srv.info.lanes));
    var status: api.Status = .{};
    srv.engine.status(&status, &.{});
    try o.put(a, "warming", .{ .bool = status.warming });
    const memory = try json.newObject(a);
    if (srv.engine.memory(std.mem.indexOf(u8, conn.path, "reset_peak=1") != null)) |mem| {
        try memory.put(a, "active", try json.intValue(a, mem.active));
        try memory.put(a, "cache", try json.intValue(a, mem.cache));
        try memory.put(a, "peak", try json.intValue(a, mem.peak));
    }
    try o.put(a, "memory", .{ .object = memory });
    try o.put(a, "live", try live.snapshot(a, srv.engine));
    if (status.cache) |c| try putCache(a, o, c); // PETABRIDGE (k2a-pb1) spill P6: 0014 (null with no tier)
    routes.sendValue(conn, a, 200, .{ .object = o });
}

/// The prompt cache's counters, when its tier or grouping is on: the GLM engine's top-level names (chats_continued,
/// chats_continued_prefill_tokens), then every counter under prompt_cache, the tier's own under prompt_cache.spill (an
/// array, one object per rank, rank 0 first). PETABRIDGE (k2a-pb1) spill P6: 0014, verbatim.
pub fn putCache(a: Allocator, o: *json.Object, c: api.CacheStats) Allocator.Error!void {
    try o.put(a, "chats_continued", try json.intValue(a, c.chats_continued));
    try o.put(a, "chats_continued_prefill_tokens", try json.intValue(a, c.chats_continued_prefill_tokens));
    const pc = try json.newObject(a);
    const info = @typeInfo(api.CacheStats).@"struct";
    inline for (info.field_names, info.field_types) |name, T| if (T == u64) try pc.put(a, name, try json.intValue(a, @field(c, name)));
    if (c.spill) |rc| {
        const ranks = rc.slice();
        const arr = try a.alloc(json.Value, ranks.len);
        for (ranks, arr, 0..) |t, *v, r| {
            const sp = try json.newObject(a);
            try sp.put(a, "rank", try json.intValue(a, r));
            inline for (@typeInfo(@TypeOf(t)).@"struct".field_names) |name| try sp.put(a, name, try json.intValue(a, @field(t, name)));
            v.* = .{ .object = sp };
        }
        try pc.put(a, "spill", .{ .array = arr });
    }
    try o.put(a, "prompt_cache", .{ .object = pc });
}

pub fn models(srv: *Server, conn: *Conn, a: Allocator) !void {
    const created = std.Io.Clock.real.now(srv.io).toSeconds();
    const data = try a.alloc(Value, srv.config.model_ids.len);
    for (srv.config.model_ids, data) |id, *slot| {
        const m = try json.newObject(a);
        try m.put(a, "id", .{ .string = id });
        try m.put(a, "object", .{ .string = "model" });
        try m.put(a, "created", try json.intValue(a, created));
        try m.put(a, "owned_by", .{ .string = "tensorfold" });
        slot.* = .{ .object = m };
    }
    const o = try json.newObject(a);
    try o.put(a, "object", .{ .string = "list" });
    try o.put(a, "data", .{ .array = data });
    routes.sendValue(conn, a, 200, .{ .object = o });
}

/// Prometheus text, version 0.0.4.
pub fn metrics(srv: *Server, conn: *Conn, a: Allocator) void {
    var out: std.Io.Writer.Allocating = .init(a);
    srv.metrics.render(srv.io, &out.writer, srv.engine, srv.info.context_window) catch return;
    const body = out.written();
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/plain; version=0.0.4; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch return) catch return;
    conn.finish(body) catch {};
}

pub fn dashboard(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.dashboard) return routes.unknown(conn, a);
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/html; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{dashboard_page.len}) catch return) catch return;
    conn.finish(dashboard_page) catch {};
}

pub fn stats(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.dashboard) return routes.unknown(conn, a);
    var status: api.Status = .{};
    srv.engine.status(&status, &.{});
    const memory = srv.engine.memory(false);
    srv.metrics.mutex.lockUncancelable(srv.io);
    const prompt = srv.metrics.prompt;
    const generation = srv.metrics.generation;
    const drafted = srv.metrics.drafted;
    const accepted = srv.metrics.accepted;
    const rounds = srv.metrics.rounds;
    srv.metrics.mutex.unlock(srv.io);
    const body = statsSnapshot(a, status, memory, prompt, generation, drafted, accepted, rounds) catch return;
    conn.quiet_log = true;
    routes.sendValue(conn, a, 200, body);
}

pub fn statsSnapshot(a: Allocator, status: api.Status, memory: ?api.Memory, prompt: u64, generation: u64, drafted: u64, accepted: u64, rounds: u64) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "running", try json.intValue(a, status.running));
    try o.put(a, "waiting", try json.intValue(a, status.waiting));
    try o.put(a, "decode_tokens_per_second", .{ .float = status.decode_tokens_per_second });
    try o.put(a, "prefill_tokens_per_second", .{ .float = status.prefill_tokens_per_second });
    try o.put(a, "generation_tokens_running", try json.intValue(a, status.generation_tokens));
    try o.put(a, "prompt_tokens_total", try json.intValue(a, prompt));
    try o.put(a, "generation_tokens_total", try json.intValue(a, generation));
    try o.put(a, "mtp_drafted_total", try json.intValue(a, drafted));
    try o.put(a, "mtp_accepted_total", try json.intValue(a, accepted));
    try o.put(a, "decode_rounds_total", try json.intValue(a, rounds));
    const mem = try json.newObject(a);
    if (memory) |m| {
        try mem.put(a, "active", try json.intValue(a, m.active));
        try mem.put(a, "cache", try json.intValue(a, m.cache));
        try mem.put(a, "peak", try json.intValue(a, m.peak));
    }
    try o.put(a, "memory", .{ .object = mem });
    return .{ .object = o };
}

const dashboard_page = @embedFile("dashboard.html");

test "health's prompt cache counters: the GLM names on top, every counter below, the tier's under spill" {
    // PETABRIDGE (k2a-pb1) spill P6: 0014's test, with a third rank (TP3)
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const o = try json.newObject(a);
    var ranks: api.prompt_cache.RankCounts = .{ .n = 3 };
    ranks.ranks[0].points = 5;
    ranks.ranks[1].points = 4;
    ranks.ranks[2].points = 3;
    try putCache(a, o, .{ .chats_continued = 1, .disk_hits_committed = 3, .spill = ranks });
    try std.testing.expectEqualStrings("1", o.get("chats_continued").?.int);
    const pc = o.get("prompt_cache").?.object;
    try std.testing.expectEqualStrings("3", pc.get("disk_hits_committed").?.int);
    const spill = pc.get("spill").?.array;
    try std.testing.expectEqual(@as(usize, 3), spill.len);
    try std.testing.expectEqualStrings("5", spill[0].object.get("points").?.int);
    try std.testing.expectEqualStrings("1", spill[1].object.get("rank").?.int);
    try std.testing.expectEqualStrings("4", spill[1].object.get("points").?.int);
    try std.testing.expectEqualStrings("3", spill[2].object.get("points").?.int);
    const none = try json.newObject(a);
    try putCache(a, none, .{});
    try std.testing.expect(none.get("prompt_cache").?.object.get("spill") == null);
}

test "stats snapshot exposes counters and omits unknown memory" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const value = try statsSnapshot(arena.allocator(), .{ .running = 1, .generation_tokens = 2 }, null, 3, 4, 5, 6, 7);
    const object = value.object;
    for ([_][]const u8{ "running", "waiting", "generation_tokens_running", "prompt_tokens_total", "generation_tokens_total", "mtp_drafted_total", "mtp_accepted_total", "decode_rounds_total", "memory" }) |key|
        try std.testing.expect(object.contains(key));
    try std.testing.expectEqual(@as(usize, 0), object.get("memory").?.object.count());
}
