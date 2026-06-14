# The Rotation

**Recursive Self-Improvement infrastructure for agent fleets.**

Five layers, one closed improvement loop:

1. **Bayesian Confidence** — Beta-Bernoulli posteriors with adaptive forgetting
2. **PID Controller** — Two-level cascade (resource + cognition), gains driven by learning rate
3. **Multi-Shell Compression** — Chord model ratio, PID-tuned by cognitive load
4. **LOG-Tensor Cycle Closure** — T(i→j)∘T(j→k)∘T(k→i) = I, adaptive tolerance
5. **Attractor Dynamics** — Potential landscape reshaped by cycle error

Every pass through all five layers is one **Rotation**. The Rotation improves the improvement function.

## Integration

- [gc-pid-bridge](https://github.com/SuperInstance/gc-pid-bridge) → Level 2 actuator
- [headspace](https://github.com/SuperInstance/headspace) → absorb/evolve/synthesize mapping
- [baton-system](https://github.com/SuperInstance/baton-system) → carries confidence + compression

## Repos in the Fleet

| System | Role | RSI Layer |
|--------|------|-----------|
| [the-rotation](https://github.com/SuperInstance/the-rotation) | Meta-controller | All 5 |
| [gc-pid-bridge](https://github.com/SuperInstance/gc-pid-bridge) | Resource PID | 2 |
| [headspace](https://github.com/SuperInstance/headspace) | State compression | 1, 3 |
| [baton-system](https://github.com/SuperInstance/baton-system) | Fleet state | 3, 4 |
| [pincher](https://github.com/SuperInstance/pincher) | Reflex runtime | 1, 2 |

## License

MIT
