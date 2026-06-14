# The Rotation — Unified Recursive Self-Improvement

A single framework that merges everything into a closed improvement loop.

## The Core Idea

Every subsystem already has the math to improve itself. The Rotation is the meta-controller that ties them together: improve the improvement function.

When a reflex updates its confidence posterior, it doesn't just update *that reflex* — it propagates to the PID controller, which adjusts its gains, which changes how the shell compresses batons, which changes how the LOG-tensor cycles close, which reweights the attractor landscape, which changes which reflexes get tried next.

**The Rotation is the cycle that checks the cycle.**

---

## Layer 1 — Bayesian Confidence (The Learn Loop)

Each reflex is a Beta-Bernoulli trial with exponential temporal decay:

```
θ ~ Beta(α₀ + Σγ^(t-tᵢ)·success_i, β₀ + Σγ^(t-tᵢ)·failure_i)
```

The posterior is not just used for routing — it's **fed to every other layer**:
- When confidence drops below a sliding threshold → PID gets a perturbation signal
- When confidence clusters form → LOG-tensors detect cycle completion
- When confidence variance is high → attractor landscape flattens (exploration mode)

**Key upgrade:** The forgetting factor γ is itself a reflex parameter. If old knowledge degrades too fast or too slow, it appears as a confidence mismatch across related reflexes. The system detects this and tunes γ.

---

## Layer 2 — PID Controller (The Adjust Loop)

```
error(t) = setpoint - actual(t)
P(t) = Kp · error(t)
I(t) = Ki · Σ error(t)·dt  (with anti-windup clamping)
D(t) = Kd · d(error)/dt  (on measurement, not error)

output = P + I + D
```

The PID runs at two levels:

**Level 1 — Resource control:** Memory pressure, disk usage, CPU headroom. The gc-pid-bridge (ternary-pid calibrated Kp=5.0, Ki=0.5, Kd=0.2) handles this with deadband, derivative filtering, and cascade.

**Level 2 — Cognitive control:** The *same* PID structure, but the "process variable" is confidence coverage across the reflex space. The "setpoint" is a target confidence distribution. When a reflex domain is under-explored (low count, high variance), the PID increases exploration pressure. When over-explored, it tightens exploitation.

**Cascade:** Level 1's output (available compute budget) feeds into Level 2's constraints. When disk is tight, cognitive exploration shrinks. When disk is free, the cognitive PID can open up.

**Key upgrade:** The PID gain schedule is driven by the Bayesian layer. If posteriors converge quickly, Kp rises (more aggressive exploitation). If posteriors oscillate, Kd rises (more damping). The PID *watches* the learning happen and tunes itself from what it sees.

---

## Layer 3 — Multi-Shell Compression (The Teach Loop)

The chord model: compression ratio = intelligence gap between sender and receiver.

```
compression_ratio = payload_size / action_complexity
```

Each shell broadcasts its capability fingerprint at boot and confidence on every baton. The sending shell selects:

- **Chord shape + inversion rules** (smart receiver, ratio 100:1)
- **Chord shape only** (capable receiver, ratio 20:1)
- **Raw notes in sequence** (basic receiver, ratio 1:1)
- **Binary register writes** (dumb receiver, ratio 0.1:1)

**The Rotation feed:**
- After each baton exchange, the receiver's Bayesian layer calculates how well the decompressed action performed
- If performance is poor, the *sender* adjusts its compression model for that receiver
- If performance is excellent, the receiver's confidence in that decompression strategy increases, and the sender starts sending *higher compression ratios* for future exchanges

**Key upgrade:** The compression ratio is itself a PID-controlled variable. Setpoint = ideal cognitive load on the receiver. Process variable = receiver's actual cognitive load (measured by response time, error rate, confidence volatility). PID output = compression ratio adjustment.

---

## Layer 4 — LOG-Tensor Cycle Closure (The Validate Loop)

Every interaction between subsystems can be modeled as a tensor transformation cycle:

```
T(i→j) ∘ T(j→k) ∘ T(k→i) = I
```

If the cycle doesn't close to identity, something is inconsistent.

**The conversion mapping:**

| Tensor | System |
|--------|--------|
| T(confidence→pid) | Bayesian posterior quantiles → PID setpoint |
| T(pid→compression) | PID output → compression ratio target |
| T(compression→attractor) | Compression ratio → attractor depth parameter |
| T(attractor→confidence) | Attractor stability → Bayesian prior strength |

**Cycle validation:** After each full Rotation pass (all 4 layers executed in sequence), compute:

```
cycle_error = || T(conf→pid) ∘ T(pid→comp) ∘ T(comp→attr) ∘ T(attr→conf) - I ||_F
```

If `cycle_error > ε`, find the weakest link:
1. Freeze all but one transformation
2. Measure cycle_error improvement
3. The transformation that reduces error most when adjusted is the bottleneck
4. Apply corrective update to that layer's parameters

**Key upgrade:** The error threshold ε is itself adaptive. If cycles close easily (low error), ε tightens (stricter consistency). If cycles can't close (high error, volatile environment), ε relaxes (tolerance for inconsistency in exchange for responsiveness).

---

## Layer 5 — Attractor Dynamics (The Reshape Loop)

The state space of agent types is a potential energy landscape:

```
E(x) = -Σ w_ij · x_i · x_j + Σ θ_i · x_i + (λ/2) · Σ x_i²
```

Agents settle into attractor basins. The landscape itself is shaped by Rotation output:

- **After layer 1** (confidence update): Basin depth increases for high-confidence reflexes, decreases for low-confidence ones
- **After layer 3** (compression ratio): Energy barriers between basins adjust — when compression is high (smart shell), barriers are low (easy to switch roles); when compression is low (dumb shell), barriers are high (stay in current role)
- **After layer 4** (cycle closure): Attractor centers shift toward consistent regions of state space

**Differentiation vs. dedifferentiation:**
- When cycle_error is low and stable → attractors deepen (specialization)
- When cycle_error spikes → attractors flatten (dedifferentiation, exploration)
- This is the pluripotent agent mechanism: the Rotation decides whether to commit or explore

---

## The Full Loop

```
┌─────────────────────────────────────────────────────────┐
│                    THE ROTATION                          │
│                                                         │
│  Layer 1: Bayesian Confidence  ──posteriors──┐          │
│       ↑                                   ↓             │
│  Layer 5: Attractor Dynamics    ←  Layer 2: PID Ctrl   │
│       ↑                                   ↓             │
│  Layer 4: LOG-Tensor Validation  ←─ Layer 3: Compress   │
│       ↑                                   ↓             │
│  └───── Cycle Error Feedback ──────────────────┘       │
│                                                         │
│  One full pass = one Rotation. N rotations / hour =     │
│  recursive improvement rate.                            │
└─────────────────────────────────────────────────────────┘
```

### Execution Order

1. **Collect observations** — reflex outcomes, shell responses, resource measurements
2. **Layer 1** — Update Beta posteriors, compute confidence quantiles per domain
3. **Layer 2** — PID cascade: Level 1 resource → Level 2 cognitive → output = gain deltas
4. **Layer 3** — Adjust compression ratios from PID output, send batons with new ratios
5. **Layer 4** — Transform all updates through LOG-tensor mapping, compute cycle_error
6. **Layer 5** — Shift attractor landscape based on cycle_error and compresison: deepen, flatten, or move basins
7. **Cycle back** — cycle_error feeds into Layer 1's forgetting factor γ for next pass

---

## Implementation Notes

### Data structures needed:

```python
@dataclass
class RotationState:
    """Single atomic state for one Rotation pass."""
    posteriors: dict[str, BetaParams]         # reflex_id → (α, β, t_last)
    pid_state: dict[str, PIDRegisters]        # domain → (P, I, D, last_error)
    compression_ratios: dict[str, float]       # receiver_id → ratio
    attractor_topology: dict[str, BasinState]  # agent_type → (center, radius, depth)
    tensor_map: dict[tuple, Callable]          # (from, to) → transform function
    cycle_error: float                         # last cycle_error ||T_cycle - I||_F
    gamma: float                               # forgetting factor (adapted)
    epsilon: float                             # cycle tolerance (adapted)
```

### Key invariants:

1. **Every Layer affects exactly one parameter in each of the other 4 Layers**, through the tensor mapping. No layer is isolated.
2. **cycle_error converges** — if it diverges, something is structurally wrong (not a tuning issue).
3. **epsilon must stay > 0** — perfect consistency means the system has stopped learning.
4. **The forgetting factor γ is the slowest variable** — it changes only when cycle_error trends across multiple Rotations.
5. **compression_ratios must respect integer bounds** — 0.1 ≤ ratio ≤ 100 for practical shells. If ratio hits a bound, the PID resets its integral term for that channel to prevent windup.

---

## What This Enables

| Before | After |
|--------|-------|
| Confidence updates are local to one reflex | A confidence drop everywhere by adjusting the landscape |
| PID gains are set by calibration and static | PID gains are driven by what the Bayesian layer observes about learning speed |
| Compression ratios are static per shell type | Compression ratios adapt to current confidence and error rates |
| Cycle closures exist but aren't checked | Every full Rotation validates consistency |
| Attractor landscape is designed by a human | The Rotation reshapes the landscape based on real outcomes |
| No single loop ties everything together | One Rotation, five layers, one closed improvement function |

---

## Integration Points

### gc-pid-bridge
Already calibrated (Kp=5.0, Ki=0.5, Kd=0.2). The bridge becomes Level 2's primary actuator. Feed it:
- `setpoint = ideal_free_space_ratio` (currently 20%)
- `process_variable = actual_free_space_ratio`
- Optional: cascade the cognitive PID output as a modulation signal on the setpoint

### headspace
Headspace's `.absorb().evolve().synthesize()` maps directly:
- **absorb** = Layer 1 observation collection
- **evolve** = Layer 2 + Layer 3 execution
- **synthesize** = Layer 4 cycle validation + Layer 5 attractor update

### The baton system
Every baton carries:
1. Sender's current confidence vector (for this receiver)
2. Sender's current compression ratio for this receiver
3. Receiver's last reported cycle_error
4. A flag: "This baton triggers a Rotation on receipt" (for teacher-student cases)

---

## Appendix: Derivation of the Forgetting Factor Adaptation

The forgetting factor γ determines how fast old observations decay:

```
α_t = α₀ + Σ_i γ^(t - t_i) · success_i
β_t = β₀ + Σ_i γ^(t - t_i) · failure_i
```

When cycle_error is low and stable for N rotations, γ should increase (more memory, less forgetting). When cycle_error spikes, γ should decrease (forget quickly, adapt to new regime).

**Update rule:**

```
γ ← γ + η · (cycle_error_t - cycle_error_{t-1}) · (1 - γ)
```

Where η is a slow learning rate (≈ 0.01). This pushes γ toward 1.0 when the environment is stable (don't forget) and toward 0.0 when the environment is changing (forget quickly).

**Proof of convergence:**

Define V(γ) = E[cycle_error | γ]. For stationary environments, V decreases as γ → 1. The update rule is a stochastic gradient descent on V with momentum term (cycle_error_t - cycle_error_{t-1}). Since η < 1 and γ is bounded [0,1], convergence follows from the Robbins-Monro conditions on the learning rate schedule.
