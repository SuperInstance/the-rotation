# The Rotation — Recursive Self-Improvement Engine

ARM64-optimized low-level crates for the Rotation meta-controller.

## Crates

| Crate | Path | Description | NEON |
|-------|------|-------------|------|
| `neon-kernel` | `crates/neon-kernel/` | SIMD ternary matmul, batch PID, attractor step, lock-free ring buffer | ✅ |
| `log-tensor` | `crates/log-tensor/` | Tensor cycle closure, bottleneck detection, forgetting factor adaptation | Pure |
| `pid-cascade` | `crates/pid-cascade/` | Two-level cascade PID (resource → cognitive), ratio conversion | Pure |
| `attractor` | `crates/attractor/` | Potential energy landscape, basin dynamics, pluripotent differentiation | NEON |
| `rotation-core` | `crates/rotation-core/` | Orchestrator: one Rotation pass, state management, bridge to fleet | Mixed |

## Feature flags

- `neon`: enable ARM NEON accelerated kernels (default on aarch64)
- `no_std`: disable std dependencies for embedded targets
- `benchmarks`: enable criterion benchmarks

## Performance targets (vs naive f32 on Oracle ARM64)

- Ternary matmul 16×16: ~4.0× (add/sub only, no multiply)
- Batch PID 16-channel: ~3.5× (NEON SDOT + FMA)
- Attractor step 64-element: ~6.0× (NEON compare + blend + threshold)
- Cycle error 4×4 tensor: ~8.0× (NEON pairwise reduction)
- Lock-free SPSC channel: ~2 GB/s throughput per core

## Integration

```
rotation-core
├── neon-kernel (SIMD hot paths)
├── log-tensor (cycle validation)
├── pid-cascade (two-level control)
└── attractor (landscape dynamics)
```
