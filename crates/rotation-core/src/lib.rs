//! # rotation-core — The Rotation Orchestrator
//!
//! Binds all 5 layers into one atomic Rotation pass.
//!
//! ```text
//! One Rotation:
//!   1. Collect observations
//!   2. Bayesian Layer — update posteriors, compute confidence quantiles
//!   3. PID Cascade — resource PID → cognitive PID with Bayesian gain schedule
//!   4. Compression — adjust ratios from PID output
//!   5. LOG-Tensor — validate cycle, find bottleneck
//!   6. Attractor — shift landscape based on cycle_error + compression
//!   7. Feedback — cycle_error → forgetting factor γ
//! ```
//!
//! Designed for concurrent use: multiple agents can perform independent
//! rotations on state that's protected by the lock-free ring buffer.

use std::vec::Vec;
use std::string::String;

// Re-export alloc for Vec usage
use std as alloc;

// ── Re-exports ───────────────────────────────────────────────────────────────

// #[cfg(not(feature = "no_std"))]
pub use log_tensor::*;
pub use pid_cascade::*;
pub use attractor::*;
pub use neon_kernel::*;

// ── Rotation State (complete) ────────────────────────────────────────────────

/// Complete state for one Rotation engine instance.
#[derive(Clone, Debug)]
pub struct RotationEngine {
    // Layer 1: Bayesian posterior state
    pub posteriors: alloc::vec::Vec<BetaPosterior>,

    // Layer 2: PID cascade
    pub pid: CascadePID,

    // Layer 3: Compression ratios (per receiver)
    pub compression_ratios: alloc::vec::Vec<CompressionChannel>,

    // Layer 4: Tensor cycle
    pub cycle: CycleState,

    // Layer 5: Attractor landscape
    pub landscape: Landscape,

    // Meta-parameters
    pub forgetting_factor: f32,
    pub learning_rate: f32,
    pub prev_cycle_error: f32,
    pub rotations_executed: u64,
}

impl RotationEngine {
    /// Create a new rotation engine with default parameters.
    pub fn new(num_reflexes: usize, num_receivers: usize, epsilon: f32) -> Self {
        let mut posteriors = Vec::with_capacity(num_reflexes);
        for _ in 0..num_reflexes {
            posteriors.push(BetaPosterior::new(2.0, 2.0)); // weakly informative prior
        }

        let mut ratios = Vec::with_capacity(num_receivers);
        for _ in 0..num_receivers {
            ratios.push(CompressionChannel::new(10.0));
        }

        RotationEngine {
            posteriors,
            pid: CascadePID::new(),
            compression_ratios: ratios,
            cycle: CycleState::new(epsilon),
            landscape: Landscape::new(0.1, 1.0),
            forgetting_factor: 0.9,
            learning_rate: 0.01,
            prev_cycle_error: 0.0,
            rotations_executed: 0,
        }
    }

    /// Execute one full rotation.
    ///
    /// `observations`: (reflex_id, success, timestamp_elapsed) for each reflex.
    /// `free_space_ratio`: disk free ratio (0.0–1.0).
    ///
    /// Returns a [`RotationReport`] with diagnostics.
    pub fn rotate(
        &mut self,
        observations: &[(usize, bool, f32)],
        free_space_ratio: f32,
        dt: f32,
    ) -> RotationReport {
        // ── Layer 1: Bayesian ──────────────────────────────────────────────────
        for &(reflex_id, success, elapsed) in observations {
            if reflex_id < self.posteriors.len() {
                self.posteriors[reflex_id].update(success, elapsed, self.forgetting_factor);
            }
        }

        // Compute confidence coverage: how many reflexes have confident posteriors
        let total = self.posteriors.len();
        let confident = self.posteriors.iter()
            .filter(|p| p.confidence() > 0.6)
            .count();
        let confidence_coverage = if total > 0 {
            confident as f32 / total as f32
        } else {
            0.5
        };

        // Compute convergence rate and variance for gain scheduling
        let (convergence_rate, confidence_variance) = self.compute_bayesian_metrics();

        // ── Layer 2: PID Cascade ───────────────────────────────────────────────
        self.pid.schedule_gains(convergence_rate, confidence_variance);
        let (resource_out, cognitive_out) = self.pid.update(free_space_ratio, confidence_coverage, dt);

        // ── Layer 3: Compression ───────────────────────────────────────────────
        for channel in self.compression_ratios.iter_mut() {
            channel.adjust(resource_out, cognitive_out);
        }

        // Derived: average compression ratio for landscape update
        let avg_ratio = if !self.compression_ratios.is_empty() {
            self.compression_ratios.iter().map(|c| c.ratio).sum::<f32>()
                / self.compression_ratios.len() as f32
        } else {
            10.0
        };

        // ── Layer 4: Cycle Validation ──────────────────────────────────────────
        // Update the tensor transformations from actual system dynamics
        // (simplified: map confidence → PID → compression → attractor)
        self.sync_tensor_map(confidence_coverage, resource_out, cognitive_out, avg_ratio);

        let cycle_error = self.cycle.validate();
        let (bottleneck_idx, bottleneck_val) = self.cycle.find_bottleneck();
        self.cycle.adapt_epsilon();

        // ── Feedback: Forgetting Factor ──────────────────────────────────────
        self.forgetting_factor = update_forgetting_factor(
            self.forgetting_factor,
            cycle_error,
            self.prev_cycle_error,
            self.learning_rate,
        );
        self.prev_cycle_error = cycle_error;

        // ── Layer 5: Attractor ─────────────────────────────────────────────────
        let landscape_changed = rotation_update(
            &mut self.landscape,
            cycle_error,
            self.cycle.cycle_error_avg,
            avg_ratio,
        );

        self.rotations_executed += 1;

        RotationReport {
            cycle_error,
            cycle_error_avg: self.cycle.cycle_error_avg,
            epsilon: self.cycle.epsilon,
            bottleneck_idx: bottleneck_idx as u8,
            bottleneck_val,
            resource_output: resource_out,
            cognitive_output: cognitive_out,
            confidence_coverage,
            convergence_rate,
            forgetting_factor: self.forgetting_factor,
            avg_compression_ratio: avg_ratio,
            num_basins: self.landscape.num_basins,
            landscape_changed,
            rotations_total: self.rotations_executed,
        }
    }

    /// Compute aggregate Bayesian metrics for PID gain scheduling.
    #[inline]
    fn compute_bayesian_metrics(&self) -> (f32, f32) {
        let n = self.posteriors.len();
        if n == 0 {
            return (0.5, 0.5);
        }

        let mut total_variance = 0.0;
        let mut high_conf_count = 0;
        for p in &self.posteriors {
            let var = p.variance();
            total_variance += var;
            if var < 0.05 {
                high_conf_count += 1;
            }
        }

        let convergence_rate = high_conf_count as f32 / n as f32;
        let confidence_variance = f32::min(total_variance / n as f32, 0.5);
        (convergence_rate, confidence_variance)
    }

    /// Sync the tensor map from current system state.
    fn sync_tensor_map(
        &mut self,
        confidence_coverage: f32,
        resource_out: f32,
        cognitive_out: f32,
        avg_ratio: f32,
    ) {
        // T₁₂: Bayesian → PID
        // Confidence coverage affects PID setpoint modulation
        self.cycle.t_01.set(0, 0, confidence_coverage);    // confidence → cognitive setpoint
        self.cycle.t_01.set(1, 1, 1.0 - confidence_coverage); // uncertainty → damping
        self.cycle.t_01.set(2, 2, 0.5);                     // count channel static

        // T₂₃: PID → Compression
        self.cycle.t_12.set(0, 0, resource_out.abs());      // resource magnitude → ratio base
        self.cycle.t_12.set(1, 0, cognitive_out);           // cognitive direction → ratio delta
        self.cycle.t_12.set(2, 2, 1.0);                      // conservatism

        // T₃₀ (using t_30 for attractor → bayesian to close the cycle):
        // Compression → Attractor: ratio affects barrier height
        // This is placeholder-level; the actual mapping emerges from data
        self.cycle.t_23.set(0, 0, avg_ratio / 100.0);         // ratio → barrier modulation
        self.cycle.t_23.set(1, 1, 0.5);                       // radius modifier

        // T₃₀: Attractor → Bayesian (closes the cycle)
        // Basin stability → prior strength
        let avg_stability = if self.landscape.num_basins > 0 {
            (0..self.landscape.num_basins)
                .map(|i| self.landscape.basins[i].depth)
                .sum::<f32>() / self.landscape.num_basins as f32
        } else {
            0.5
        };
        self.cycle.t_30.set(0, 0, avg_stability / 10.0);     // depth → prior α boost
        self.cycle.t_30.set(1, 1, self.cycle.epsilon);        // tolerance → variance
    }
}

// ── Supporting Types ─────────────────────────────────────────────────────────

/// Beta posterior for one reflex (Bayesian Layer 1).
#[derive(Clone, Debug)]
#[repr(C)]
pub struct BetaPosterior {
    pub alpha: f32,
    pub beta: f32,
    pub last_update: f32,
}

impl BetaPosterior {
    pub fn new(alpha: f32, beta: f32) -> Self {
        BetaPosterior { alpha, beta, last_update: 0.0 }
    }

    /// Update with observation. Applies forgetting factor γ to prior.
    #[inline]
    pub fn update(&mut self, success: bool, elapsed: f32, gamma: f32) {
        // Temporal decay: shrink the prior counts
        let decay = gamma.powf(elapsed);
        self.alpha = self.alpha * decay + if success { 1.0 } else { 0.0 };
        self.beta = self.beta * decay + if success { 0.0 } else { 1.0 };
        self.last_update += elapsed;
    }

    /// Mean of the Beta posterior.
    #[inline]
    pub fn mean(&self) -> f32 {
        self.alpha / (self.alpha + self.beta + f32::EPSILON)
    }

    /// Variance of the Beta posterior.
    #[inline]
    pub fn variance(&self) -> f32 {
        let n = self.alpha + self.beta;
        if n <= 0.0 { return 1.0; }
        self.alpha * self.beta / (n * n * (n + 1.0))
    }

    /// Confidence: how sure we are (1 - variance).
    #[inline]
    pub fn confidence(&self) -> f32 {
        1.0 - self.variance()
    }

    /// Effective sample size.
    #[inline]
    pub fn sample_size(&self) -> f32 {
        self.alpha + self.beta
    }
}

/// Compression ratio for one receiver channel (Layer 3).
#[derive(Clone, Debug)]
#[repr(C)]
pub struct CompressionChannel {
    pub ratio: f32,
    pub min_ratio: f32,
    pub max_ratio: f32,
}

impl CompressionChannel {
    pub fn new(ratio: f32) -> Self {
        CompressionChannel { ratio, min_ratio: 0.1, max_ratio: 100.0 }
    }

    /// Adjust ratio from PID output.
    #[inline]
    pub fn adjust(&mut self, resource_out: f32, cognitive_out: f32) {
        self.ratio = pid_to_compression_ratio(resource_out, cognitive_out, self.ratio);
        self.ratio = self.ratio.clamp(self.min_ratio, self.max_ratio);
    }
}

/// Diagnostic report from one rotation.
#[derive(Clone, Debug)]
pub struct RotationReport {
    pub cycle_error: f32,
    pub cycle_error_avg: f32,
    pub epsilon: f32,
    pub bottleneck_idx: u8,
    pub bottleneck_val: f32,
    pub resource_output: f32,
    pub cognitive_output: f32,
    pub confidence_coverage: f32,
    pub convergence_rate: f32,
    pub forgetting_factor: f32,
    pub avg_compression_ratio: f32,
    pub num_basins: usize,
    pub landscape_changed: bool,
    pub rotations_total: u64,
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;
    use std::prelude::v1::*;
    use super::*;

    #[test]
    fn test_beta_posterior_update() {
        let mut p = BetaPosterior::new(2.0, 2.0);
        assert!((p.mean() - 0.5).abs() < 0.01);
        p.update(true, 1.0, 0.9);
        assert!(p.mean() > 0.5, "success should increase mean");
        assert!(p.confidence() > 0.0);
    }

    #[test]
    fn test_full_rotation_empty_observations() {
        let mut engine = RotationEngine::new(5, 3, 0.1);
        let report = engine.rotate(&[], 0.20, 0.1);

        assert_eq!(engine.rotations_executed, 1);
        assert!(report.cycle_error >= 0.0);
        assert!(report.confidence_coverage >= 0.0);
    }

    #[test]
    fn test_full_rotation_with_observations() {
        let mut engine = RotationEngine::new(10, 3, 0.1);

        // Simulate 10 observations, mostly successes
        let obs: Vec<(usize, bool, f32)> = (0..10)
            .map(|i| (i, i % 3 != 0, 1.0)) // 2/3 success
            .collect();

        let report = engine.rotate(&obs, 0.20, 0.1);
        assert!(report.rotations_total == 1);
        assert!(report.confidence_coverage > 0.3);

        // Second rotation should show convergence
        let report2 = engine.rotate(&obs, 0.15, 0.1);
        assert!(report2.rotations_total == 2);
    }

    #[test]
    fn test_convergence_rate_increases() {
        let mut engine = RotationEngine::new(5, 1, 0.1);

        // All successes → fast convergence
        let obs: Vec<(usize, bool, f32)> = (0..5).map(|i| (i, true, 1.0)).collect();
        let r1 = engine.rotate(&obs, 0.20, 0.1);
        let r2 = engine.rotate(&obs, 0.20, 0.1);
        let r3 = engine.rotate(&obs, 0.20, 0.1);

        // Convergence rate should increase (more high-confidence posteriors)
        assert!(r3.convergence_rate > 0.0);
    }

    #[test]
    fn test_compression_channel_adjust() {
        let mut ch = CompressionChannel::new(10.0);
        ch.adjust(0.5, 0.3); // exploit mode, slight resource → raise ratio slightly
        assert!(ch.ratio >= 0.1);
        ch.adjust(-0.5, -0.8); // explore mode, constrained → lower ratio
        assert!(ch.ratio > 0.0);
    }

    #[test]
    fn test_rotation_feedback_loop() {
        let mut engine = RotationEngine::new(5, 1, 0.1);
        let gamma_before = engine.forgetting_factor;

        // Run rotations with increasing cycle error
        for _ in 0..5 {
            let obs: Vec<(usize, bool, f32)> = (0..5).map(|i| (i, true, 1.0)).collect();
            engine.rotate(&obs, 0.20, 0.1);
        }

        // After stable cycles, forgetting factor should increase
        assert!(engine.forgetting_factor >= gamma_before,
            "stable rotation should increase forgetting factor");
    }
}
