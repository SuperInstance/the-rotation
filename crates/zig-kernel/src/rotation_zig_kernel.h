#ifndef ROTATION_ZIG_KERNEL_H
#define ROTATION_ZIG_KERNEL_H

#include <stdint.h>
#include <stdalign.h>

/// Zig kernel exports (C ABI) for the Rotation engine.
/// Compatable with Rust FFI, C++, C, Zig, and any language that can
/// call C functions.

#ifdef __cplusplus
extern "C" {
#endif

// ── Ternary Tensor Operations ────────────────────────────────────────────────

/// Pack 64 i8 values into a u128 (2 bits each: 00=0, 01=+1, 10=-1).
uint128_t tensor_pack(const int8_t src[64]);

/// Unpack u128 to 64 i8 values.
void tensor_unpack(uint128_t val, int8_t dst[64]);

/// 16x16 ternary matmul: rows x cols -> float out[256].
void matmul_ternary_16x16(const uint128_t rows[16],
                          const uint128_t cols[16],
                          float out[256]);

// ── NEON-Optimized Kernels ───────────────────────────────────────────────────

/// Attractor step: for each of 64 f32 values, output sign(i) if |v| > threshold.
void attractor_64(const float values[64], float threshold, int8_t output[64]);

/// Batch PID update for 16 independent controllers.
void pid_batch(const float errors[16], const float prevs[16],
               float integrals[16],
               float kp, float ki, float kd,
               float dt, float clamp_val,
               float output[16]);

// ── Lock-Free Concurrent Ring Buffer ─────────────────────────────────────────

/// Ring buffer descriptor (cache-line padded).
typedef struct {
    volatile uint64_t head;
    char _pad1[64 - 8];
    volatile uint64_t tail;
    char _pad2[64 - 8];
} ring_buffer_t;

/// Push to ring buffer. Returns 0 on success, 1 if full.
int ringbuf_push(volatile ring_buffer_t *rb,
                 float *slots, uint32_t capacity, float val);

/// Pop from ring buffer. Returns 0 on success, 1 if empty.
int ringbuf_pop(volatile ring_buffer_t *rb,
                const float *slots, uint32_t capacity, float *val);

#ifdef __cplusplus
}
#endif

#endif // ROTATION_ZIG_KERNEL_H
