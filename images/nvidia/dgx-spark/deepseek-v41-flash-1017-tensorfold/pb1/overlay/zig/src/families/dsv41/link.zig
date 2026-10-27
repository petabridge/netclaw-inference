//! Rank 0 to rank 1 over TCP (the fabric addresses): length-prefixed frames for each round's step and NCCL's id, as
//! multi.py's Link sends them. One connection, TCP_NODELAY, SO_BUSY_POLL, blocking reads (recvSpin polls first).
const std = @import("std");
const c = std.c;

pub const Link = struct {
    fd: c.fd_t,

    fn socket() !c.fd_t {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        tune(fd);
        return fd;
    }

    /// TCP_NODELAY, and SO_BUSY_POLL: a read polls the NIC's receive queue itself (up to 50 us) instead of waiting for
    /// its interrupt, whose core may be leaving a deep idle state (rank 0 to rank 1, 64-byte frames: round trip 1.0 ms
    /// blocking, 0.72 ms polling the socket, 11 us with SO_BUSY_POLL).
    fn tune(fd: c.fd_t) void {
        const one: c_int = 1;
        const busy_us: c_int = 50;
        _ = c.setsockopt(fd, c.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&one), @sizeOf(c_int));
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.BUSY_POLL, std.mem.asBytes(&busy_us), @sizeOf(c_int));
    }

    fn addr(ip: [4]u8, port: u16) c.sockaddr.in {
        return .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip) };
    }

    /// Rank 0: listen on `ip`:`port` and take the one peer's connection.
    pub fn listen(ip: [4]u8, port: u16) !Link {
        return listenTimeout(ip, port, null);
    }

    /// Bounded accept used by the experimental multi-peer startup. Null keeps
    /// the original standalone listen contract.
    pub fn listenTimeout(ip: [4]u8, port: u16, timeout_ms: ?u32) !Link {
        const fd = try socket();
        errdefer _ = c.close(fd);
        const one: c_int = 1;
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, std.mem.asBytes(&one), @sizeOf(c_int));
        var a = addr(ip, port);
        if (c.bind(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) != 0) return error.BindFailed;
        if (c.listen(fd, 1) != 0) return error.ListenFailed;
        if (timeout_ms) |ms| try waitReady(fd, c.POLL.IN, monotonicNs() + @as(u64, ms) * std.time.ns_per_ms);
        const peer = c.accept(fd, null, null);
        if (peer < 0) return error.AcceptFailed;
        _ = c.close(fd);
        tune(peer);
        return .{ .fd = peer };
    }

    /// Rank 1: connect to rank 0, retrying for up to `seconds` while it starts.
    pub fn connect(io: std.Io, ip: [4]u8, port: u16, seconds: u32) !Link {
        const deadline = monotonicNs() + @as(u64, seconds) * std.time.ns_per_s;
        while (monotonicNs() < deadline) {
            const fd = try socket();
            const flags = c.fcntl(fd, c.F.GETFL, @as(c_int, 0));
            const nonblock: c_int = @bitCast(std.posix.O{ .NONBLOCK = true });
            if (flags < 0 or c.fcntl(fd, c.F.SETFL, flags | nonblock) != 0) {
                _ = c.close(fd);
                return error.SocketFailed;
            }
            var a = addr(ip, port);
            const rc = c.connect(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in));
            var connected = rc == 0;
            if (!connected and c.errno(rc) == .INPROGRESS) {
                const ready = if (waitReady(fd, c.POLL.OUT, deadline)) |_| true else |_| false;
                if (ready) {
                    var socket_error: c_int = 0;
                    var size: c.socklen_t = @sizeOf(c_int);
                    connected = c.getsockopt(fd, c.SOL.SOCKET, c.SO.ERROR, &socket_error, &size) == 0 and socket_error == 0;
                }
            }
            if (connected) {
                if (c.fcntl(fd, c.F.SETFL, flags) != 0) {
                    _ = c.close(fd);
                    return error.SocketFailed;
                }
                return .{ .fd = fd };
            }
            _ = c.close(fd);
            const now = monotonicNs();
            if (now >= deadline) break;
            std.Io.sleep(io, .fromNanoseconds(@intCast(@min(deadline - now, 250 * std.time.ns_per_ms))), .awake) catch {};
        }
        return error.ConnectFailed;
    }

    fn waitReady(fd: c.fd_t, events: i16, deadline: u64) !void {
        var fds = [_]c.pollfd{.{ .fd = fd, .events = events, .revents = 0 }};
        while (true) {
            const now = monotonicNs();
            if (now >= deadline) return error.StartupTimeout;
            const ms = @min((deadline - now + std.time.ns_per_ms - 1) / std.time.ns_per_ms, std.math.maxInt(c_int));
            const rc = c.poll(&fds, 1, @intCast(ms));
            if (rc > 0) {
                if (fds[0].revents & (events | c.POLL.ERR | c.POLL.HUP) != 0) return;
                return error.PeerDown;
            }
            if (rc < 0 and c.errno(rc) != .INTR) return error.PeerDown;
        }
    }

    pub fn close(l: *Link) void {
        _ = c.close(l.fd);
        l.* = undefined;
    }

    fn writeAll(l: Link, bytes: []const u8) !void {
        var done: usize = 0;
        while (done < bytes.len) {
            const n = c.send(l.fd, bytes.ptr + done, bytes.len - done, c.MSG.NOSIGNAL);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    fn readAll(l: Link, out: []u8) !void {
        var done: usize = 0;
        while (done < out.len) {
            const n = c.recv(l.fd, out.ptr + done, out.len - done, 0);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    /// One frame: its length (u32, little-endian), then its bytes.
    pub fn send(l: Link, bytes: []const u8) !void {
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(bytes.len), .little);
        try l.writeAll(&len);
        try l.writeAll(bytes);
    }

    /// recv, polling the socket without blocking for up to `spin_ns` first: a frame that comes within it is read
    /// without the receiving thread's sleep and wake-up (a core's deepest idle state takes 433 us to leave on GB10; the
    /// follower's frames come a fraction of a millisecond after its GPU goes idle), and after it the read blocks.
    pub fn recvSpin(l: Link, buf: []u8, spin_ns: u64) ![]u8 {
        var len: [4]u8 = undefined;
        var got: usize = 0;
        const t0 = monotonicNs();
        var polls: u32 = 0;
        while (got < len.len) {
            const n = c.recv(l.fd, @as([*]u8, &len) + got, len.len - got, c.MSG.DONTWAIT);
            if (n > 0) {
                got += @intCast(n);
                continue;
            }
            if (n == 0) return error.PeerDown;
            switch (c.errno(n)) {
                .AGAIN => {},
                .INTR => continue,
                else => return error.PeerDown,
            }
            polls +%= 1;
            if (polls % 64 == 0 and monotonicNs() -% t0 > spin_ns) break;
            std.atomic.spinLoopHint();
        }
        if (got < len.len) try l.readAll(len[got..]);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }

    pub fn monotonicNs() u64 {
        var ts: c.timespec = undefined;
        _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    // PETABRIDGE (k2a-pb1): readable, recvWithin and readAllBy are MiaAI 0016's link.zig hunk, verbatim; rank_wire.zig
    // (P3) needs them for its bounded reply reads. Nothing outside rank_wire calls them yet.
    /// Whether a byte can be read within `timeout_ns` (poll(2); a hang-up or socket error counts as readable, so the
    /// read that follows reports it). Never consumes anything.
    pub fn readable(l: Link, timeout_ns: u64) !bool {
        var one = [_]Link{l};
        var ready = [_]bool{false};
        return try readableAny(&one, &ready, timeout_ns) > 0;
    }

    /// recv with a deadline over the whole frame (rank-to-rank replies, rank_wire.zig): error.Timeout when the frame
    /// does not arrive in full within `timeout_ns`. After a timeout the link may hold part of a frame and is no longer
    /// usable: the caller ends the pair (there is no resynchronizing a byte stream).
    pub fn recvWithin(l: Link, buf: []u8, timeout_ns: u64) ![]u8 {
        const deadline = monotonicNs() +| timeout_ns;
        var len: [4]u8 = undefined;
        try l.readAllBy(&len, deadline);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAllBy(buf[0..n], deadline);
        return buf[0..n];
    }

    fn readAllBy(l: Link, out: []u8, deadline: u64) !void {
        var done: usize = 0;
        while (done < out.len) {
            const now = monotonicNs();
            if (now >= deadline) return error.Timeout;
            if (!try l.readable(deadline - now)) return error.Timeout;
            const n = c.recv(l.fd, out.ptr + done, out.len - done, c.MSG.DONTWAIT);
            if (n > 0) {
                done += @intCast(n);
                continue;
            }
            if (n == 0) return error.PeerDown;
            switch (c.errno(n)) {
                .AGAIN, .INTR => {},
                else => return error.PeerDown,
            }
        }
    }

    /// The next frame into `buf` (a frame longer than `buf` is refused); its bytes.
    pub fn recv(l: Link, buf: []u8) ![]u8 {
        var len: [4]u8 = undefined;
        try l.readAll(&len);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }
};

// PETABRIDGE (k2a-pb1): readableAny is MiaAI 0016's, verbatim (rank_wire.zig's Leader polls its followers with it, P3)
/// Which of `links` have a byte to read (or a hang-up/error to report) within `timeout_ns`: their flags in `ready`
/// (one per link), the count. 0 when none did in time. A timeout of 0 only polls.
pub fn readableAny(links: []const Link, ready: []bool, timeout_ns: u64) !usize {
    std.debug.assert(ready.len == links.len and links.len <= 8);
    var fds: [8]c.pollfd = undefined;
    for (links, 0..) |l, i| fds[i] = .{ .fd = l.fd, .events = c.POLL.IN, .revents = 0 };
    const deadline = Link.monotonicNs() +| timeout_ns;
    while (true) {
        const now = Link.monotonicNs();
        const left = if (deadline > now) deadline - now else 0;
        // poll's milliseconds, rounded up so a short wait still waits; capped well inside c_int
        const ms: c_int = @intCast(@min((left + std.time.ns_per_ms - 1) / std.time.ns_per_ms, 1 << 30));
        const n = c.poll(&fds, @intCast(links.len), ms);
        if (n < 0) {
            if (c.errno(n) == .INTR) continue;
            return error.PollFailed;
        }
        var count: usize = 0;
        for (ready, 0..) |*r, i| {
            r.* = fds[i].revents != 0;
            if (r.*) count += 1;
        }
        if (count > 0 or Link.monotonicNs() >= deadline) return count;
    }
}

/// "192.0.2.7" as four bytes.
pub fn parseIp(text: []const u8) ![4]u8 {
    var out: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, text, '.');
    for (&out) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
    if (it.next() != null) return error.BadAddress;
    return out;
}

test "addresses parse as four bytes" {
    try std.testing.expectEqual([4]u8{ 192, 0, 2, 7 }, try parseIp("192.0.2.7"));
    try std.testing.expectError(error.BadAddress, parseIp("192.0.2"));
}

test "missing listener peer times out without waiting for a kernel TCP timeout" {
    const began = Link.monotonicNs();
    try std.testing.expectError(error.StartupTimeout, Link.listenTimeout(.{ 127, 0, 0, 1 }, 0, 30));
    try std.testing.expect(Link.monotonicNs() - began < 5 * std.time.ns_per_s);
    try std.testing.expectError(error.ConnectFailed, Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, 0, 0));
}

test "frames cross a loopback link in order" {
    const io = std.testing.io;
    const port: u16 = 29000 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 500)));
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            defer l.close();
            l.send("step 1") catch return;
            l.send(&.{}) catch return;
            l.send("polled") catch return;
            l.send("late") catch return;
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("step 1", try l.recv(&buf));
    try std.testing.expectEqual(@as(usize, 0), (try l.recv(&buf)).len);
    try std.testing.expectEqualStrings("polled", try l.recvSpin(&buf, 1000 * std.time.ns_per_ms));
    try std.testing.expectEqualStrings("late", try l.recvSpin(&buf, 0));
    t.join();
}

// PETABRIDGE (k2a-pb1): MiaAI 0016's test of recvWithin/readable/readableAny, verbatim (P3)
test "a reply that does not come in time is a timeout, a readable link says so" {
    const io = std.testing.io;
    const port: u16 = 29500 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 400)));
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            defer l.close();
            var buf: [8]u8 = undefined;
            _ = l.recv(&buf) catch return; // the go-ahead
            // half a frame, then nothing for a while, then the rest
            l.writeAll(&.{ 5, 0, 0, 0, 'h' }) catch return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(300), .awake) catch {};
            l.writeAll("ello") catch return;
            _ = l.recv(&buf) catch return;
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var buf: [16]u8 = undefined;
    try std.testing.expect(!try l.readable(20 * std.time.ns_per_ms));
    var ready = [_]bool{true};
    try std.testing.expectEqual(@as(usize, 0), try readableAny(&.{l}, &ready, 0));
    try std.testing.expect(!ready[0]);
    try l.send("go");
    try std.testing.expectError(error.Timeout, l.recvWithin(&buf, 100 * std.time.ns_per_ms));
    try std.testing.expect(try l.readable(2 * std.time.ns_per_s)); // the rest arrives
    try l.send("end");
    t.join();
    // the peer is gone: readable (a hang-up), then the read says so
    try std.testing.expect(try l.readable(std.time.ns_per_s));
    var rest: [8]u8 = undefined;
    _ = c.recv(l.fd, &rest, rest.len, 0); // "ello"
    try std.testing.expectError(error.PeerDown, l.recvWithin(&buf, std.time.ns_per_s));
}
