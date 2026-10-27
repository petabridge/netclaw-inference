//! ``tensorfold serve`` flags: one table for parsing them and for ``capabilities --json``.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

pub const Kind = enum {
    value, // --flag VALUE or --flag=VALUE
    store_true, // --flag
    append, // repeatable --flag VALUE
};

pub const Flag = struct {
    name: []const u8,
    kind: Kind = .value,
    /// argparse's choices; empty: any value of the flag's type.
    choices: []const []const u8 = &.{},
    /// This binary honours the flag (and lists it in ``capabilities``); the rest are refused as unsupported.
    native: bool = false,
    /// The values it honours when only some of the choices.
    native_values: ?[]const []const u8 = null,
};

const backend_values: []const []const u8 = if (builtin.os.tag == .macos) &.{ "auto", "mlx" } else &.{ "auto", "cuda" };

/// The CUDA build's own flags and variables (the Metal build refuses them).
const cuda_build = builtin.os.tag == .linux;

/// Every Python serve flag (``cli_args.build_parser``), the ones this binary serves marked native.
pub const flags = [_]Flag{
    .{ .name = "--host", .native = true },
    .{ .name = "--port", .native = true },
    .{ .name = "--name", .native = true },
    .{ .name = "--alias", .kind = .append, .native = true },
    .{ .name = "--api-key", .kind = .append, .native = true },
    .{ .name = "--api-key-file", .native = true },
    .{ .name = "--metrics-open", .kind = .store_true, .native = true },
    .{ .name = "--dashboard", .kind = .store_true, .native = true },
    .{ .name = "--benchmark-control", .kind = .store_true, .native = true },
    .{ .name = "--vision", .kind = .store_true },
    .{ .name = "--vision-urls", .kind = .store_true },
    .{ .name = "--vision-offload", .kind = .store_true },
    .{ .name = "--vision-max-images" },
    .{ .name = "--vision-image-tokens" },
    .{ .name = "--context", .native = true },
    .{ .name = "--speed-up", .native = true },
    .{ .name = "--max-tokens", .native = true },
    .{ .name = "--temperature", .native = true },
    .{ .name = "--top-p", .native = true },
    .{ .name = "--top-k", .native = true },
    .{ .name = "--min-p", .native = true },
    .{ .name = "--thinking", .kind = .store_true, .native = true },
    .{ .name = "--no-thinking", .kind = .store_true, .native = true },
    .{ .name = "--reasoning-effort", .choices = &.{ "minimal", "low", "medium", "high", "xhigh", "max" }, .native = true },
    .{ .name = "--thinking-budget", .native = true },
    .{ .name = "--loop-guard", .kind = .store_true, .native = true },
    .{ .name = "--no-drafts", .kind = .store_true, .native = true },
    .{ .name = "--keep-warm", .native = true },
    .{ .name = "--compact-at", .native = true },
    .{ .name = "--compact-keep", .native = true },
    .{ .name = "--compact-memory", .native = true },
    .{ .name = "--drafter" },
    .{ .name = "--drafter-bits" },
    .{ .name = "--mtp-drafts" },
    .{ .name = "--mtp-confidence" },
    .{ .name = "--lane-kernels", .choices = &.{ "auto", "on", "off" } },
    .{ .name = "--prompt-cache-gib", .native = true },
    .{ .name = "--prompt-cache-over-cap", .kind = .store_true, .native = true },
    .{ .name = "--checkpoint-slots" },
    // PETABRIDGE (k2a-pb1) spill P6 (MiaAI-Lab 0014): the prompt cache's disk tier (spill v2): every one off unless given
    .{ .name = "--spill-gib", .native = true },
    .{ .name = "--spill-dir", .native = true },
    .{ .name = "--spill-min-free-gib", .native = true },
    .{ .name = "--spill-min-tokens", .native = true },
    .{ .name = "--spill-quota", .native = true },
    .{ .name = "--snapshot-dir", .native = true, .native_values = &.{"none"} },
    .{ .name = "--max-snapshots", .native = true, .native_values = &.{"0"} },
    .{ .name = "--parallel", .native = true },
    .{ .name = "--decode-share" },
    .{ .name = "--prefill-pass" },
    .{ .name = "--pass-cache-gib" },
    .{ .name = "--mlx-cache-gib" },
    .{ .name = "--ssd-experts" },
    .{ .name = "--ple-on-ssd", .kind = .store_true },
    .{ .name = "--no-update-check", .kind = .store_true, .native = true },
    .{ .name = "--backend", .choices = &.{ "auto", "mlx", "cuda" }, .native = true, .native_values = backend_values },
    .{ .name = "--tp", .choices = &.{ "1", "2" } },
    .{ .name = "--rank", .choices = &.{ "0", "1" } },
    .{ .name = "--master" },
    .{ .name = "--master-port" },
    .{ .name = "--kv-dtype", .choices = &.{ "bf16", "int8", "int4" } },
    .{ .name = "--prefill-fp8", .kind = .store_true },
    .{ .name = "--no-prefill-fp8", .kind = .store_true },
    .{ .name = "--precision", .choices = &.{ "checkpoint", "full" } },
    // the CUDA CLI's own: the GPU ordinal, and whole prompt chunks a call runs as staggered segments
    .{ .name = "--device", .native = cuda_build },
    .{ .name = "--segments", .native = cuda_build },
    // the native server's own: shared prompt states (a harness) kept on disk for later sessions and servers
    .{ .name = "--learn", .kind = .store_true, .native = true },
    .{ .name = "--learn-dir", .native = true },
    .{ .name = "--learn-gib", .native = true },
};

/// The variables this binary honours as the Python engine does, then the CUDA build's.
pub const env = [_][]const u8{ "TENSORFOLD_API_KEY", "TENSORFOLD_NO_LIVE", "TENSORFOLD_SEED_SALT", "TENSORFOLD_REQUEST_LOG", "TENSORFOLD_NO_UPDATE_CHECK", "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE" } ++
    spill_env ++
    (if (cuda_build) [_][]const u8{ "TF_CUDA_DEVICE", "TF_CUDA_SEGMENTS", "TENSORFOLD_CUDA_KERNELS", "TENSORFOLD_MEMORY_RESERVE_GIB", "TENSORFOLD_CUDA_MEMORY_LIMIT_GB" } else [_][]const u8{});

/// The prompt cache tier's variables (api.Spill): its knobs beyond the flags. PETABRIDGE (k2a-pb1) spill P6: 0014.
pub const spill_env = [_][]const u8{ "TF_SPILL_TURNS", "TF_SPILL_LOAD_MS", "TF_SPILL_LOAD_WAIT_MS", "TF_SPILL_FLUSH_MS" };

pub const Args = struct {
    model: []const u8 = "",
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    name: []const u8 = "",
    alias: []const []const u8 = &.{},
    api_key: []const []const u8 = &.{},
    api_key_file: ?[]const u8 = null,
    metrics_open: bool = false,
    dashboard: bool = false,
    context: ?i64 = null,
    speed_up: ?[]const u8 = null, // speed-up mode: this Mac's settings for the two-Mac link (Flash Next)
    prompt_cache_gib: ?f64 = null, // kept prompt states' budget (0: none; null: what 70% of RAM leaves past the loaded server)
    prompt_cache_over_cap: bool = false, // a --prompt-cache-gib past that is kept, not refused
    learn: bool = false, // keep shared prompt states on disk (--learn-dir: where; it implies --learn)
    learn_dir: ?[]const u8 = null,
    learn_gib: f64 = 32, // disk for learned states, every model and build together
    max_tokens: i64 = 4096,
    temperature: ?f64 = null,
    top_p: ?f64 = null,
    top_k: ?i64 = null,
    min_p: ?f64 = null,
    thinking: bool = true,
    reasoning_effort: ?[]const u8 = null,
    benchmark_control: bool = false,
    thinking_budget: i64 = 0,
    loop_guard: bool = false,
    no_drafts: bool = false,
    keep_warm: i64 = 900, // seconds the idle keepalive runs after the last request ends (0: off)
    compact_auto: bool = false,
    compact_fraction: ?f64 = null,
    compact_keep: ?u32 = null,
    compact_memory: ?[]const u8 = null,
    parallel: []const u8 = "auto",
    backend: []const u8 = "auto",
    device: ?u32 = null,
    segments: ?u32 = null,
    // PETABRIDGE (k2a-pb1) spill P6: 0014's disk tier switches (off unless --spill-dir and --spill-gib)
    spill_dir: ?[]const u8 = null, // this rank's local store for kept prompt states (with --spill-gib)
    spill_gib: f64 = 0, // what it may hold (0: off)
    spill_min_free_gib: f64 = 50, // no write leaves less free on its file system
    spill_min_tokens: u32 = 8192, // shorter kept states are not written
    spill_quota: f64 = 0, // one conversation's share of --spill-gib (0: none)
};

/// A usage error's message (argparse's ``error:`` line); the caller exits 2.
pub const Usage = struct { message: []const u8 = "" };

fn find(name: []const u8) ?Flag {
    for (flags) |f| if (std.mem.eql(u8, f.name, name)) return f;
    return null;
}

fn fail(u: *Usage, a: Allocator, comptime fmt: []const u8, args: anytype) error{ Usage, OutOfMemory } {
    u.message = try std.fmt.allocPrint(a, fmt, args);
    return error.Usage;
}

/// ``serve`` arguments as the switch passes them: the model and full flag names.
pub fn parse(a: Allocator, argv: []const []const u8, u: *Usage) error{ Usage, OutOfMemory }!Args {
    var out: Args = .{};
    var alias: std.ArrayList([]const u8) = .empty;
    var keys: std.ArrayList([]const u8) = .empty;
    var model: ?[]const u8 = null;
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const token = argv[i];
        if (!std.mem.startsWith(u8, token, "--")) {
            if (model != null) return fail(u, a, "unrecognized arguments: {s}", .{token});
            model = token;
            continue;
        }
        const eq = std.mem.indexOfScalar(u8, token, '=');
        const name = token[0 .. eq orelse token.len];
        const flag = find(name) orelse return fail(u, a, "unrecognized arguments: {s}", .{token});
        var value: ?[]const u8 = null;
        if (flag.kind != .store_true) {
            if (eq) |e| value = token[e + 1 ..] else {
                i += 1;
                if (i >= argv.len) return fail(u, a, "argument {s}: expected one argument", .{name});
                value = argv[i];
            }
            if (flag.choices.len > 0) for (flag.choices) |c| {
                if (std.mem.eql(u8, c, value.?)) break;
            } else return fail(u, a, "argument {s}: invalid choice: '{s}'", .{ name, value.? });
        } else if (eq != null) return fail(u, a, "argument {s}: ignored explicit argument '{s}'", .{ name, token[eq.? + 1 ..] });
        if (!flag.native) return fail(u, a, "{s} is not served by the native engine yet; serve this command with --engine python", .{name});
        if (flag.native_values) |allowed| if (value) |v| for (allowed) |x| {
            if (std.mem.eql(u8, x, v)) break;
        } else return fail(u, a, "{s} {s} is not served by the native engine; serve this command with --engine python", .{ name, v });
        try apply(a, &out, name, value, u, &alias, &keys);
    }
    out.model = model orelse return fail(u, a, "the following arguments are required: model", .{});
    if (out.spill_gib > 0 and out.spill_dir == null) return fail(u, a, "--spill-gib needs --spill-dir: a local directory for this rank's stored prompt states", .{});
    out.alias = alias.items;
    out.api_key = keys.items;
    return out;
}

fn int(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!i64 {
    return std.fmt.parseInt(i64, std.mem.trim(u8, v, " "), 10) catch fail(u, a, "argument {s}: invalid int value: '{s}'", .{ name, v });
}

fn float(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!f64 {
    return @import("fields.zig").pyFloat(v) orelse fail(u, a, "argument {s}: invalid float value: '{s}'", .{ name, v });
}

/// A finite GiB count from 0 (off) to below 2^34, whose bytes fit a u64.
fn gib(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!f64 {
    const g = try float(u, a, name, v);
    if (!std.math.isFinite(g) or g < 0 or g >= 1 << 34) return fail(u, a, "argument {s}: expected GiB from 0 (off) to below 2^34: '{s}'", .{ name, v });
    return g;
}

fn apply(a: Allocator, out: *Args, name: []const u8, value: ?[]const u8, u: *Usage, alias: *std.ArrayList([]const u8), keys: *std.ArrayList([]const u8)) error{ Usage, OutOfMemory }!void {
    if (std.mem.eql(u8, name, "--benchmark-control")) {
        out.benchmark_control = true;
        return;
    }
    const v = value orelse "";
    const is = struct {
        fn f(x: []const u8, y: []const u8) bool {
            return std.mem.eql(u8, x, y);
        }
    }.f;
    if (try cudaFlag(a, out, name, v, u)) return;
    if (try spillFlag(a, out, name, v, u)) return; // PETABRIDGE (k2a-pb1) spill P6: 0014
    if (is(name, "--host")) out.host = v else if (is(name, "--port")) {
        const p = try int(u, a, name, v);
        if (p < 0 or p > 65535) return fail(u, a, "argument --port: invalid port: '{s}'", .{v});
        out.port = @intCast(p);
    } else if (is(name, "--name")) out.name = v else if (is(name, "--alias")) try alias.append(a, v) else if (is(name, "--api-key")) try keys.append(a, v) else if (is(name, "--api-key-file")) out.api_key_file = v else if (is(name, "--metrics-open")) out.metrics_open = true else if (is(name, "--dashboard")) out.dashboard = true else if (is(name, "--context")) out.context = try int(u, a, name, v) else if (is(name, "--speed-up")) out.speed_up = v else if (is(name, "--prompt-cache-gib")) out.prompt_cache_gib = try gib(u, a, name, v) else if (is(name, "--prompt-cache-over-cap")) out.prompt_cache_over_cap = true else if (is(name, "--learn")) out.learn = true else if (is(name, "--learn-dir")) {
        out.learn = true;
        out.learn_dir = v;
    } else if (is(name, "--learn-gib")) {
        out.learn = true;
        out.learn_gib = try gib(u, a, name, v);
    } else if (is(name, "--max-tokens")) out.max_tokens = try int(u, a, name, v) else if (is(name, "--temperature")) out.temperature = try float(u, a, name, v) else if (is(name, "--top-p")) out.top_p = try float(u, a, name, v) else if (is(name, "--top-k")) out.top_k = try int(u, a, name, v) else if (is(name, "--min-p")) out.min_p = try float(u, a, name, v) else if (is(name, "--thinking")) out.thinking = true else if (is(name, "--no-thinking")) out.thinking = false else if (is(name, "--reasoning-effort")) out.reasoning_effort = v else if (is(name, "--thinking-budget")) out.thinking_budget = try int(u, a, name, v) else if (is(name, "--loop-guard")) out.loop_guard = true else if (is(name, "--no-drafts")) out.no_drafts = true else if (is(name, "--keep-warm")) out.keep_warm = try int(u, a, name, v) else if (is(name, "--compact-at")) {
        if (std.mem.eql(u8, v, "auto")) out.compact_auto = true else {
            const f = try float(u, a, name, v);
            if (!(f > 0 and f <= 1)) return fail(u, a, "argument --compact-at: expected auto or a fraction in (0, 1]: '{s}'", .{v});
            out.compact_fraction = f;
        }
    } else if (is(name, "--compact-keep")) {
        const n = try int(u, a, name, v);
        if (n < 0) return fail(u, a, "argument --compact-keep: expected a token count from 0: '{s}'", .{v});
        out.compact_keep = @intCast(n);
    } else if (is(name, "--compact-memory")) out.compact_memory = v else if (is(name, "--parallel")) out.parallel = v else if (is(name, "--backend")) out.backend = v;
}

/// The CUDA build's --device and --segments; false for any other flag.
fn cudaFlag(a: Allocator, out: *Args, name: []const u8, v: []const u8, u: *Usage) error{ Usage, OutOfMemory }!bool {
    if (std.mem.eql(u8, name, "--device")) {
        out.device = std.math.cast(u32, try int(u, a, name, v)) orelse return fail(u, a, "argument --device: a GPU ordinal from 0: '{s}'", .{v});
    } else if (std.mem.eql(u8, name, "--segments")) {
        const n = try int(u, a, name, v);
        out.segments = if (n >= 1) @intCast(@min(n, std.math.maxInt(u32))) else return fail(u, a, "argument --segments: a count from 1: '{s}'", .{v});
    } else return false;
    return true;
}

/// The disk tier's flags; false for any other flag. PETABRIDGE (k2a-pb1) spill P6: 0014, verbatim.
fn spillFlag(a: Allocator, out: *Args, name: []const u8, v: []const u8, u: *Usage) error{ Usage, OutOfMemory }!bool {
    const is = struct {
        fn f(x: []const u8, y: []const u8) bool {
            return std.mem.eql(u8, x, y);
        }
    }.f;
    if (is(name, "--spill-gib")) out.spill_gib = try gib(u, a, name, v) else if (is(name, "--spill-dir")) {
        if (v.len == 0) return fail(u, a, "argument --spill-dir: expected a directory", .{});
        out.spill_dir = v;
    } else if (is(name, "--spill-min-free-gib")) out.spill_min_free_gib = try gib(u, a, name, v) else if (is(name, "--spill-min-tokens")) {
        out.spill_min_tokens = std.math.cast(u32, try int(u, a, name, v)) orelse return fail(u, a, "argument --spill-min-tokens: a token count from 0: '{s}'", .{v});
    } else if (is(name, "--spill-quota")) {
        const q = try float(u, a, name, v);
        if (!(q >= 0 and q <= 1)) return fail(u, a, "argument --spill-quota: a share of --spill-gib from 0 (none) to 1: '{s}'", .{v});
        out.spill_quota = q;
    } else return false;
    return true;
}

/// The engine's api.Spill from the flags, then the variables in `env` (`get(name)`: its value or null). An invalid
/// variable keeps the default, and so does 0 for the restore budgets (a 0 ns budget never progresses a restore).
pub fn spill(args: Args, env_: anytype) api.Spill {
    var s: api.Spill = .{ .dir = args.spill_dir, .gib = args.spill_gib, .min_free_gib = args.spill_min_free_gib, .quota = args.spill_quota, .min_tokens = args.spill_min_tokens };
    if (env_.get("TF_SPILL_TURNS")) |v| s.turns = std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch s.turns;
    if (env_.get("TF_SPILL_LOAD_MS")) |v| s.load_ms = positive(v) orelse s.load_ms;
    if (env_.get("TF_SPILL_LOAD_WAIT_MS")) |v| s.load_wait_ms = positive(v) orelse s.load_wait_ms;
    if (env_.get("TF_SPILL_FLUSH_MS")) |v| s.flush_ms = std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch s.flush_ms;
    return s;
}

/// A variable's whole number from 1 (null: invalid or 0).
fn positive(v: []const u8) ?u32 {
    const n = std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch return null;
    return if (n == 0) null else n;
}

/// The process environment, for `spill`.
pub const ProcessEnv = struct {
    pub fn get(_: ProcessEnv, name: [:0]const u8) ?[]const u8 {
        const v = std.c.getenv(name) orelse return null;
        return std.mem.span(v);
    }
};

/// ``--parallel``: "auto" is up to 8 requests at once; a number caps it.
pub fn parallel(text: []const u8) ?u32 {
    const t = std.mem.trim(u8, text, " ");
    if (std.ascii.eqlIgnoreCase(t, "auto")) return 8;
    const n = std.fmt.parseInt(i64, t, 10) catch return null;
    return @intCast(@max(1, @min(n, 4096)));
}

/// ``--parallel`` named a number (an engine that fits fewer refuses), not "auto" (it serves what fits).
pub fn parallelFixed(text: []const u8) bool {
    return !std.ascii.eqlIgnoreCase(std.mem.trim(u8, text, " "), "auto") and parallel(text) != null;
}

/// What ``capabilities --json`` reports about the engine side: its version, chip, backends and families.
pub const Engines = struct {
    version: []const u8,
    chip: ?[]const u8 = null,
    backends: []const []const u8 = &.{},
    /// model_type to the weight formats it reads, as gate entries name them.
    families: []const api.Family = &.{},
};

/// The native capabilities document: the table's native flags and the honoured variables.
pub fn capabilities(w: *std.Io.Writer, e: Engines) !void {
    try w.print("{{\"schema\": 1, \"engine\": \"zig\", \"version\": \"{s}\", \"chip\": ", .{e.version});
    if (e.chip) |c| try w.print("\"{s}\"", .{c}) else try w.writeAll("null");
    try w.writeAll(", \"backends\": [");
    for (e.backends, 0..) |b, i| try w.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", b });
    try w.writeAll("], \"families\": {");
    for (e.families, 0..) |f, i| {
        try w.print("{s}\"{s}\": [", .{ if (i > 0) ", " else "", f.model_type });
        for (f.formats, 0..) |fmt, j| try w.print("{s}\"{s}\"", .{ if (j > 0) ", " else "", fmt });
        try w.writeAll("]");
    }
    try w.writeAll("}, \"serve\": {\"flags\": {");
    var first = true;
    for (flags) |f| {
        if (!f.native) continue;
        try w.print("{s}\"{s}\": {{", .{ if (first) "" else ", ", f.name });
        first = false;
        const values = f.native_values orelse f.choices;
        if (values.len > 0) {
            try w.writeAll("\"values\": [");
            for (values, 0..) |v, j| try w.print("{s}\"{s}\"", .{ if (j > 0) ", " else "", v });
            try w.writeAll("]");
        }
        try w.writeAll("}");
    }
    try w.writeAll("}}, \"env\": [");
    for (env, 0..) |name, i| try w.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", name });
    try w.writeAll("]}\n");
}

test "parse and capabilities share the table" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    const args = try parse(a, &.{ "/models/x", "--port", "9000", "--api-key=k1", "--api-key", "k2", "--no-thinking", "--reasoning-effort", "low" }, &u);
    try std.testing.expectEqual(@as(u16, 9000), args.port);
    try std.testing.expectEqual(@as(usize, 2), args.api_key.len);
    try std.testing.expect(!args.thinking);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--drafter", "x" }, &u));
    try std.testing.expectEqualStrings("max", (try parse(a, &.{ "m", "--reasoning-effort", "max" }, &u)).reasoning_effort.?);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--reasoning-effort", "invalid" }, &u));
    try std.testing.expectEqual(@as(?f64, 12.5), (try parse(a, &.{ "m", "--prompt-cache-gib", "12.5" }, &u)).prompt_cache_gib);
    try std.testing.expectEqual(@as(?f64, 0), (try parse(a, &.{ "m", "--prompt-cache-gib", "0" }, &u)).prompt_cache_gib);
    try std.testing.expect(!(try parse(a, &.{ "m", "--prompt-cache-gib", "16" }, &u)).prompt_cache_over_cap);
    try std.testing.expect((try parse(a, &.{ "m", "--prompt-cache-gib", "16", "--prompt-cache-over-cap" }, &u)).prompt_cache_over_cap);
    try std.testing.expect((try parse(a, &.{ "m", "--learn" }, &u)).learn);
    const dir = try parse(a, &.{ "m", "--learn-dir", "/x/y" }, &u);
    try std.testing.expect(dir.learn and std.mem.eql(u8, "/x/y", dir.learn_dir.?));
    const capped = try parse(a, &.{ "m", "--learn-gib", "8" }, &u);
    try std.testing.expect(capped.learn and capped.learn_gib == 8);
    for ([_][]const u8{ "1e300", "inf", "nan", "-1", "17179869184" }) |bad| try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--prompt-cache-gib", bad }, &u));
    var out: std.Io.Writer.Allocating = .init(a);
    try capabilities(&out.writer, .{ .version = "0.6.5" });
    const doc = out.written();
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--no-thinking\": {}") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--drafter\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--compact-at\": {}") != null);
    const on = try parse(a, &.{ "m", "--compact-at", "auto", "--compact-keep", "100", "--compact-memory", "notes" }, &u);
    try std.testing.expect(on.compact_auto);
    try std.testing.expectEqual(@as(?u32, 100), on.compact_keep);
    try std.testing.expectEqualStrings("notes", on.compact_memory.?);
    try std.testing.expectEqual(@as(?f64, 0.5), (try parse(a, &.{ "m", "--compact-at", "0.5" }, &u)).compact_fraction);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--compact-at", "0" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--compact-at", "2" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--compact-keep", "-1" }, &u));
}

test "--device and --segments: CUDA builds serve them, values checked" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    if (cuda_build) {
        const args = try parse(a, &.{ "m", "--device", "1", "--segments=2" }, &u);
        try std.testing.expectEqual(@as(?u32, 1), args.device);
        try std.testing.expectEqual(@as(?u32, 2), args.segments);
    } else try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--device", "1" }, &u));
    var out: Args = .{};
    try std.testing.expect(try cudaFlag(a, &out, "--device", "0", &u));
    try std.testing.expectEqual(@as(?u32, 0), out.device);
    try std.testing.expectError(error.Usage, cudaFlag(a, &out, "--device", "-1", &u));
    try std.testing.expectError(error.Usage, cudaFlag(a, &out, "--segments", "0", &u));
    try std.testing.expect(try cudaFlag(a, &out, "--segments", "4", &u));
    try std.testing.expectEqual(@as(?u32, 4), out.segments);
    try std.testing.expect(!try cudaFlag(a, &out, "--port", "1", &u));
    try std.testing.expect(parallelFixed("3") and !parallelFixed("auto") and !parallelFixed("x"));
}

test "spill flags and variables: off by default, checked, passed whole" {
    // PETABRIDGE (k2a-pb1) spill P6: 0014's test, verbatim
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    const Env = struct {
        pairs: []const [2][]const u8 = &.{},
        pub fn get(e: @This(), name: [:0]const u8) ?[]const u8 {
            for (e.pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
            return null;
        }
    };
    const off = spill(try parse(a, &.{"m"}, &u), Env{});
    try std.testing.expect(!off.enabled());
    try std.testing.expectEqual(api.Spill{}, off);
    const args = try parse(a, &.{ "m", "--spill-dir", "/nvme/spill", "--spill-gib", "64", "--spill-min-free-gib=20", "--spill-min-tokens", "4096", "--spill-quota", "0.25" }, &u);
    const on = spill(args, Env{ .pairs = &.{ .{ "TF_SPILL_TURNS", "2" }, .{ "TF_SPILL_LOAD_MS", "25" }, .{ "TF_SPILL_FLUSH_MS", "x" } } });
    try std.testing.expect(on.enabled());
    try std.testing.expectEqualStrings("/nvme/spill", on.dir.?);
    try std.testing.expectEqual(@as(f64, 64), on.gib);
    try std.testing.expectEqual(@as(f64, 20), on.min_free_gib);
    try std.testing.expectEqual(@as(u32, 4096), on.min_tokens);
    try std.testing.expectEqual(@as(f64, 0.25), on.quota);
    try std.testing.expectEqual(@as(u32, 2), on.turns);
    try std.testing.expectEqual(@as(u32, 25), on.load_ms);
    try std.testing.expectEqual(@as(u32, 60_000), on.flush_ms); // an invalid value keeps the default
    const zero = spill(try parse(a, &.{"m"}, &u), Env{ .pairs = &.{ .{ "TF_SPILL_LOAD_MS", "0" }, .{ "TF_SPILL_LOAD_WAIT_MS", "250" } } });
    try std.testing.expectEqual(@as(u32, 40), zero.load_ms); // 0 would never progress a restore: the default
    try std.testing.expectEqual(@as(u32, 250), zero.load_wait_ms);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--spill-gib", "8" }, &u)); // no directory
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--spill-quota", "1.5" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--spill-min-tokens", "-1" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--spill-dir", "" }, &u));
    try std.testing.expect(!spill(try parse(a, &.{ "m", "--spill-dir", "/x" }, &u), Env{}).enabled()); // a directory alone stays off
    var out: std.Io.Writer.Allocating = .init(a);
    try capabilities(&out.writer, .{ .version = "0.6.5" });
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\"--spill-gib\": {}") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\"TF_SPILL_LOAD_MS\"") != null);
}
