//! Rust FFI bindings for the Zig kernel (zig-kernel).
//!
//! These are `extern "C"` wrappers around the Zig functions exported
//! as `callconv(.C)` from `crates/zig-kernel/src/main.zig`.
//!
//! ## Linking
//!
//! The Zig static library (`librotation_zig.a`) is linked via `build.rs`
//! which runs `zig build` in `crates/zig-kernel/` and passes the output
//! to rustc via `cargo:rustc-link-search` and `cargo:rustc-link-lib`.
//!
//! ## ABI compatibility
//!
//! All exported Zig functions use `callconv(.C)` which is the standard
//! C ABI for the target (AAPCS64 on aarch64). The Rust side matches
//! with `extern "C"` and equivalent types.
//!
//! ## Type mapping
//!
//! | Zig type          | Rust type          |
//! |-------------------|--------------------|
//! | u128              | u128               |
//! | [*]const i8       | *const i8          |
//! | [*]f32            | *mut f32           |
//! | f32               | f32                |
//! | i32               | i32                |
//! | u32               | u32                |
//! | *volatile RingBuf | *const RingBuffer  |

#![allow(non_snake_case, dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

// ── Extern declarations ─────────────────────────────────────────────────────

extern "C" {
    /// Pack 64 i8 values into u128 (2 bits each).
    pub fn tensor_pack(src: *const i8) -> u128;

    /// Unpack u128 to 64 i8 values.
    pub fn tensor_unpack(val: u128, dst: *mut i8);

    /// 16x16 ternary matmul: rows x cols -> out (float).
    pub fn matmul_ternary_16x16(rows: *const u128, cols: *const u128, out: *mut f32);

    /// Attractor step: threshold compare + sign selection.
    pub fn attractor_64(values: *const f32, threshold: f32, output: *mut i8);

    /// Batch PID update for 16 channels.
    pub fn pid_batch(
        errors: *const f32,
        prevs: *const f32,
        integrals: *mut f32,
        kp: f32,
        ki: f32,
        kd: f32,
        dt: f32,
        clamp_val: f32,
        output: *mut f32,
    );

    /// Push to ring buffer. Returns 0 on success, 1 if full.
    pub fn ringbuf_push(rb: *const RingBuffer, slots: *mut f32, capacity: u32, val: f32) -> i32;

    /// Pop from ring buffer. Returns 0 on success, 1 if empty.
    pub fn ringbuf_pop(rb: *const RingBuffer, slots: *const f32, capacity: u32, val: *mut f32) -> i32;
}

// ── Rust re-exports (mirrors Zig types) ──────────────────────────────────────

/// Lock-free SPSC ring buffer (mirrors Zig `RingBuf`).
#[repr(C, align(64))]
pub struct RingBuffer {
    pub head: AtomicU64,
    _pad1: [u8; 64 - 8],
    pub tail: AtomicU64,
    _pad2: [u8; 64 - 8],
}

impl RingBuffer {
    /// Create a new zeroed ring buffer.
    pub const fn new() -> Self {
        RingBuffer {
            head: AtomicU64::new(0),
            _pad1: [0u8; 64 - 8],
            tail: AtomicU64::new(0),
            _pad2: [0u8; 64 - 8],
        }
    }
}

/// Safe wrapper: pack 64 i8 values.
#[inline]
pub fn pack_ternary(src: &[i8; 64]) -> u128 {
    unsafe { tensor_pack(src.as_ptr()) }
}

/// Safe wrapper: unpack u128 to 64 i8 values.
#[inline]
pub fn unpack_ternary(val: u128, dst: &mut [i8; 64]) {
    unsafe { tensor_unpack(val, dst.as_mut_ptr()) }
}

/// Safe wrapper: 16x16 ternary matmul.
#[inline]
pub fn matmul_ternary(a: &[u128; 16], b: &[u128; 16], c: &mut [f32; 256]) {
    unsafe { matmul_ternary_16x16(a.as_ptr(), b.as_ptr(), c.as_mut_ptr()) }
}

/// Safe wrapper: attractor step.
#[inline]
pub fn attractor(values: &[f32; 64], threshold: f32, output: &mut [i8; 64]) {
    unsafe { attractor_64(values.as_ptr(), threshold, output.as_mut_ptr()) }
}

/// Safe wrapper: batch PID.
#[inline]
pub fn pid_batch_16(
    errors: &[f32; 16],
    prevs: &[f32; 16],
    integrals: &mut [f32; 16],
    kp: f32, ki: f32, kd: f32, dt: f32, clamp: f32,
    output: &mut [f32; 16],
) {
    unsafe {
        pid_batch(
            errors.as_ptr(),
            prevs.as_ptr(),
            integrals.as_mut_ptr(),
            kp, ki, kd, dt, clamp,
            output.as_mut_ptr(),
        );
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_cross_language() {
        let src: [i8; 64] = core::array::from_fn(|i| match i % 3 {
            0 => 1,
            1 => -1,
            _ => 0,
        });
        let packed = pack_ternary(&src);
        let mut unpacked = [0i8; 64];
        unpack_ternary(packed, &mut unpacked);
        assert_eq!(src, unpacked);
    }

    #[test]
    fn test_matmul_cross_language() {
        // All rows zero → all cols zero → all output zero
        let a = [0u128; 16];
        let b = [0u128; 16];
        let mut c = [0.0f32; 256];
        matmul_ternary(&a, &b, &mut c);
        for &v in &c {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_attractor_cross_language() {
        let values = [1.0f32; 64];
        let mut output = [0i8; 64];
        attractor(&values, 0.5, &mut output);
        for &v in &output {
            assert_eq!(v, 1);
        }
    }

    #[test]
    fn test_pid_cross_language() {
        let errors = [1.0f32; 16];
        let prevs = [0.0f32; 16];
        let mut integrals = [0.0f32; 16];
        let mut output = [0.0f32; 16];
        pid_batch_16(&errors, &prevs, &mut integrals, 1.0, 0.1, 0.05, 0.1, 100.0, &mut output);
        assert!((output[0] - 1.51).abs() < 0.01);
    }

    #[test]
    fn test_ringbuf_cross_language() {
        let rb = RingBuffer::new();
        let mut slots = [0.0f32; 64];
        let mut val = 0.0f32;

        unsafe {
            assert_eq!(ringbuf_push(&rb, slots.as_mut_ptr(), 64, 42.0), 0);
            assert_eq!(ringbuf_pop(&rb, slots.as_ptr(), 64, &mut val), 0);
        }
        assert_eq!(val, 42.0);
    }
}
