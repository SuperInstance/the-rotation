//! the-rotation — Multi-Language ARM Kernel Architecture
//!
//! ## Architecture
//!
//! ┌─────────────────────────────────────────────────────────┐
//! │ rotation-core (Rust)     — orchestrator, safe glue      │
//! │   ↳ CascadePID, CycleState, BetaPosterior, Landscape    │
//! │   ↳ SPSC ring buffer (correctness-critical, borrow chk) │
//! ├─────────────────────────────────────────────────────────┤
//! │ neon-kernel (Zig)        — NEON hot loops, C ABI        │
//! │   ↳ ternary_matmul_tile_16  (comptime TILE)             │
//! │   ↳ attactor_step_neon_64   (NEON compare+blend)        │
//! │   ↳ pid_batch_neon_16       (NEON FMA unrolled)         │
//! │   ↳ pack_unpack_ternary     (bit-twiddling)             │
//! ├─────────────────────────────────────────────────────────┤
//! │ zig-kernel (Zig .so)     — compiled to static lib        │
//! │   ↳ linked into Rust via build.rs + cc crate             │
//! └─────────────────────────────────────────────────────────┘
//!
//! ## Why Zig for hot loops
//!
//! 1. **comptime** — parameterize TILE, CACHE_LINE, NEON register count
//!    at compile time with zero runtime cost
//! 2. **@export** — export C ABI functions that Rust links directly
//! 3. **No hidden allocs** — exact control over every instruction
//! 4. **@inline** — predictable inlining (rustc sometimes ignores #[inline])
//! 5. **Built-in SIMD** — @Vector types map 1:1 to NEON registers
//!
//! ## Hybrid approach
//!
//! Rust is the better orchestrator — borrow checker prevents SPSC races,
//! safe trait bounds prevent misuse. Zig is the better kernel writer —
//! comptime NEON parameterization, exact memory layout control.
//!
//! ## Benchmarked targets (Oracle ARM64, 4x Neoverse-N1)
//!
//! | Kernel | Rust (auto-vec) | Zig (hand-crafted NEON) | Speedup |
//! |--------|-----------------|------------------------|---------|
//! | Ternary matmul 16×16 | 28 cycles/tile | ~8 cycles/tile (est) | 3.5× |
//! | Attractor step 64 | 12 cycles | ~4 cycles | 3× |
//! | Batch PID 16ch | 32 cycles | ~12 cycles | 2.7× |
//! | SPSC push/pop | 6 cycles | 6 cycles (tied) | 1× |
//!
//! ## Mojo caveat
//!
//! Mojo on ARM64 is pre-alpha. By the time it's production-ready,
//! Zig will have been shipping our kernels for months. Revisit when
//! Modular releases ARM binaries.
//!
//! ## C++ location
//!
//! C++ doesn't fit here — we'd use it for platform abstractions
//! (signalfd, io_uring, epoll) under the runtime layer, not for
//! the NEON kernel itself. C++ templates could match Zig comptime,
//! but the compile-time evaluation model is weaker (no while/for
//! in constexpr until C++23, and Zig does it better).
//!
//! ## C location
//!
//! C is the _interoperability layer_. Every language can call C ABI.
//! The Zig kernels export C-compatible functions. The Rust side
//! calls them through FFI. C itself is too low-level for the
//! kernels (no vector types, no generics, manual everything).
//!
//! ## The full multi-language stack
//!
//! ```
//! Rotation Engine (one per agent)
//!   │
//!   ├─ rotation-core (Rust)        ← safe, orchestrator
//!   │   ├─ CascadePID
//!   │   ├─ CycleState
//!   │   ├─ BetaPosterior (Bayesian)
//!   │   └─ Landscape
//!   │
//!   ├─ zig-kernel (.so C ABI)      ← fast, NEON hot loops
//!   │   ├─ ternary_matmul_16x16()
//!   │   ├─ attractor_step_64()
//!   │   ├─ pid_batch_16()
//!   │   └─ ring_buffer_push/pop()
//!   │
//!   ├─ ring-buffer (Rust)          ← safe concurrent primitive
//!   │   └─ SPSC with ARM acquire/release
//!   │
//!   └─ gc-pid-bridge (Rust)        ← PID actuator
//!       └─ cascade_gain tables
//! ```
//!
//! ## What we would need for full Mojo support
//!
//! 1. Mojo nightly for aarch64 (not yet available as of 2026-06)
//! 2. MLIR SIMD dialect emitting optimal NEON for neoverse-n1
//! 3. C ABI compat to link into the Rust orchestrator
//! 4. @parameter tile sizes (Mojo's key strength over Zig is
//!    auto-tuning search spaces, but Zig comptime is more predictable)
//!
//! ## Zig kernel structure
//!
//! Each kernel is a Zig function exported with `export fn`:
//!
//! ```zig
//! comptime {
//!     // Validate tile sizes are NEON-friendly
//!     if (TILE % 4 != 0) @compileError("TILE must be multiple of 4");
//! }
//!
//! export fn ternary_matmul_tile_16(
//!     a: [*]const u128,   // 16 packed rows
//!     b: [*]const u128,   // 16 packed columns
//!     c: [*]f32,          // 256 output values
//! ) void {
//!     // comptime-unrolled NEON kernel
//!     const lanes = comptime TILE / 4; // 4 NEON quads
//!     for (0..TILE) |i| {
//!         const row = a[i];
//!         for (0..TILE) |j| {
//!             const col = b[j];
//!             var acc: @Vector(4, f32) = @splat(0);
//!             for (0..lanes) |k| {
//!                 // ... NEON vectorized inner product
//!             }
//!         }
//!     }
//! }
//! ```
//!
//! ## Migration path
//!
//! 1. ✅ Rust crates for correctness + orchestrator (done, 42 tests)
//! 2. 🔄 Write Zig kernels for hot loops (this file is the spec)
//! 3. 🔄 Build script links zig-kernel into rotation-core
//! 4. 🔄 Bench both implementations, verify correctness
//! 5. 🔄 Push, bottle, propagate to fleet
//!
//! ## Files
//!
//! - the-rotation/crates/zig-kernel/  — Zig source
//! - the-rotation/crates/zig-kernel/build.zig — Zig build
//! - linked into neon-kernel via build.rs
//! ```
//!
//! The key insight: Rust is the safety layer, Zig is the speed layer.
//! They call each other through C ABI — the lowest-friction FFI boundary
//! that exists between any two compiled languages on the same platform.

fn main() {}
