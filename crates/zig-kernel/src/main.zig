// zig-kernel: ARM NEON hot loops for The Rotation
// Exports C ABI functions callable from Rust via FFI.

const TILE: usize = 16;
const CACHE_LINE: usize = 64;

comptime {
    if (TILE % 4 != 0) @compileError("TILE must be multiple of 4 for NEON");
    if (CACHE_LINE != 64) @compileError("CACHE_LINE must be 64 bytes for ARM");
}

fn trit_at(val: u128, idx: usize) u8 {
    return @truncate((val >> @as(u7, @intCast(idx * 2))) & 0x3);
}

export fn tensor_pack(src: [*]const i8) callconv(.C) u128 {
    var result: u128 = 0;
    for (0..64) |i| {
        const v = src[i];
        const bits: u128 = switch (v) {
            1 => 0b01,
            -1 => 0b10,
            else => 0b00,
        };
        result |= @as(u128, bits) << @as(u7, @intCast(i * 2));
    }
    return result;
}

export fn tensor_unpack(val: u128, dst: [*]i8) callconv(.C) void {
    for (0..64) |i| {
        const bits = (val >> @as(u7, @intCast(i * 2))) & 0x3;
        dst[i] = switch (bits) { 0b01 => 1, 0b10 => -1, else => 0 };
    }
}

export fn matmul_ternary_16x16(
    rows: [*]const u128,
    cols: [*]const u128,
    out: [*]f32,
) callconv(.C) void {
    for (0..TILE) |i| {
        const row = rows[i];
        for (0..TILE) |j| {
            const col = cols[j];
            var pos_match: u32 = 0;
            var neg_match: u32 = 0;
            for (0..64) |k| {
                const rbits = (row >> @as(u7, @intCast(k * 2))) & 0x3;
                const cbits = (col >> @as(u7, @intCast(k * 2))) & 0x3;
                if (rbits != 0 and cbits != 0) {
                    if (rbits == cbits) { pos_match += 1; } else { neg_match += 1; }
                }
            }
            out[i * TILE + j] = @floatFromInt(pos_match - neg_match);
        }
    }
}

export fn attractor_64(
    values: [*]const f32,
    threshold: f32,
    output: [*]i8,
) callconv(.C) void {
    const V: type = @Vector(4, f32);
    const IV: type = @Vector(4, i32);

    var base: usize = 0;
    while (base < 64) : (base += 4) {
        const v: V = values[base..][0..4].*;
        const abs_v: V = @abs(v);
        const threshold_v: V = @as(V, @splat(threshold));
        const mask: @Vector(4, bool) = abs_v > threshold_v;
        const zero_v: V = @as(V, @splat(0.0));
        const pos_v: V = @as(V, @splat(1.0));
        const neg_v: V = @as(V, @splat(-1.0));
        const sign: V = @select(f32, v > zero_v, pos_v, neg_v);
        const result: V = @select(f32, mask, sign, zero_v);
        const fi: IV = @intFromFloat(result);
        const ni: @Vector(4, i8) = @truncate(fi);
        output[base..][0..4].* = ni;
    }
}

export fn pid_batch(
    errors: [*]const f32,
    prevs: [*]const f32,
    integrals: [*]f32,
    kp: f32,
    ki: f32,
    kd: f32,
    dt: f32,
    clamp_val: f32,
    output: [*]f32,
) callconv(.C) void {
    const V: type = @Vector(4, f32);
    const kp_v: V = @as(V, @splat(kp));
    const ki_v: V = @as(V, @splat(ki));
    const kd_v: V = @as(V, @splat(kd));
    const dt_v: V = @as(V, @splat(dt));
    const clamp_v: V = @as(V, @splat(clamp_val));
    const zero_v: V = @as(V, @splat(0.0));
    const inv_dt: f32 = if (dt > 1e-10) @as(f32, 1.0) / dt else 0.0;
    const inv_dt_v: V = @as(V, @splat(inv_dt));

    var ch: usize = 0;
    while (ch < 16) : (ch += 4) {
        const err: V = errors[ch..][0..4].*;
        const prev: V = prevs[ch..][0..4].*;
        var int_vals: V = integrals[ch..][0..4].*;

        const p: V = kp_v * err;
        const int_delta: V = ki_v * err * dt_v;
        int_vals = int_vals + int_delta;
        int_vals = @max(int_vals, -clamp_v);
        int_vals = @min(int_vals, clamp_v);
        integrals[ch..][0..4].* = int_vals;

        const d_error: V = err - prev;
        const raw_deriv: V = if (dt > 1e-10) d_error * inv_dt_v else zero_v;
        const d: V = kd_v * raw_deriv;

        var out_val: V = p + int_vals + d;
        out_val = @max(out_val, -clamp_v);
        out_val = @min(out_val, clamp_v);
        output[ch..][0..4].* = out_val;
    }
}

const RingBuf = extern struct {
    head: u64 align(CACHE_LINE),
    tail: u64 align(CACHE_LINE),
};

export fn ringbuf_push(
    rb: *volatile RingBuf,
    slots: [*]f32,
    capacity: u32,
    val: f32,
) callconv(.C) i32 {
    const tail = @atomicLoad(u64, &rb.tail, .acquire);
    const head = @atomicLoad(u64, &rb.head, .acquire);
    if (tail -% head >= capacity) return 1;
    const mask = capacity - 1;
    const idx = tail & mask;
    slots[idx] = val;
    @atomicStore(u64, &rb.tail, tail +% 1, .release);
    return 0;
}

export fn ringbuf_pop(
    rb: *volatile RingBuf,
    slots: [*]const f32,
    capacity: u32,
    val: [*]f32,
) callconv(.C) i32 {
    const head = @atomicLoad(u64, &rb.head, .acquire);
    const tail = @atomicLoad(u64, &rb.tail, .acquire);
    if (head == tail) return 1;
    const mask = capacity - 1;
    const idx = head & mask;
    val[0] = slots[idx];
    @atomicStore(u64, &rb.head, head +% 1, .release);
    return 0;
}

// Tests
const testing = @import("std").testing;

test "tensor_pack_unpack_roundtrip" {
    var src: [64]i8 = undefined;
    for (0..64) |i| {
        src[i] = switch (i % 3) { 0 => 1, 1 => -1, else => 0 };
    }
    const val = tensor_pack(&src);
    var dst: [64]i8 = undefined;
    tensor_unpack(val, &dst);
    try testing.expectEqualSlices(i8, &src, &dst);
}

test "matmul_ternary_16x16" {
    var a: [16]u128 = @splat(0);
    var b: [16]u128 = @splat(0);
    var c: [256]f32 = undefined;
    var src: [64]i8 = @splat(1);
    a[0] = tensor_pack(&src);
    matmul_ternary_16x16(&a, &b, &c);
    for (c) |v| try testing.expectEqual(@as(f32, 0.0), v);
}

test "attractor_64" {
    var values: [64]f32 = @splat(1.0);
    var output: [64]i8 = undefined;
    attractor_64(&values, 0.5, &output);
    for (output) |v| try testing.expectEqual(@as(i8, 1), v);
}

test "pid_batch" {
    var errors: [16]f32 = @splat(1.0);
    var prevs: [16]f32 = @splat(0.0);
    var integrals: [16]f32 = @splat(0.0);
    var output: [16]f32 = undefined;
    pid_batch(&errors, &prevs, &integrals, 1.0, 0.1, 0.05, 0.1, 100.0, &output);
    try testing.expectApproxEqRel(@as(f32, 1.51), output[0], 0.01);
}

test "ringbuf_push_pop" {
    var rb: RingBuf = .{ .head = 0, .tail = 0 };
    var slots: [64]f32 = undefined;
    var val: f32 = undefined;
    _ = ringbuf_push(&rb, &slots, 64, 42.0);
    _ = ringbuf_pop(&rb, &slots, 64, @ptrCast(&val));
    try testing.expectEqual(@as(f32, 42.0), val);
}

test "ringbuf_full" {
    var rb: RingBuf = .{ .head = 0, .tail = 0 };
    var slots: [4]f32 = undefined;
    try testing.expectEqual(@as(i32, 0), ringbuf_push(&rb, &slots, 4, 1.0));
    try testing.expectEqual(@as(i32, 0), ringbuf_push(&rb, &slots, 4, 2.0));
    try testing.expectEqual(@as(i32, 0), ringbuf_push(&rb, &slots, 4, 3.0));
    try testing.expectEqual(@as(i32, 0), ringbuf_push(&rb, &slots, 4, 4.0));
    try testing.expectEqual(@as(i32, 1), ringbuf_push(&rb, &slots, 4, 5.0));
}

test "ringbuf_empty_pop" {
    var rb: RingBuf = .{ .head = 0, .tail = 0 };
    var slots: [4]f32 = undefined;
    var val: f32 = undefined;
    try testing.expectEqual(@as(i32, 1), ringbuf_pop(&rb, &slots, 4, @ptrCast(&val)));
}
