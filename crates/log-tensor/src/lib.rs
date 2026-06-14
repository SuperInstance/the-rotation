//! # log-tensor — Cycle Closure Engine
//!
//! Tensor transformations for the Rotation's Layer 4 validation.
//!
//! Every interaction between subsystems is modeled as a tensor transformation:
//!
//!   T(i→j) ∘ T(j→k) ∘ T(k→i) = I
//!
//! If the cycle doesn't close, the system is inconsistent. This engine
//! computes cycle_error and identifies the weakest link for correction.
//!
//! ## Tensor mapping between rotation layers:
//!
//! | Layer From | Layer To | Transformation | Symbol |
//! |-----------|---------|---------------|--------|
//! | Bayesian (1) | PID (2) | Posterior quantiles → PID setpoint | T₁₂ |
//! | PID (2) | Compress (3) | PID output → compression ratio delta | T₂₃ |
//! | Compress (3) | Attractor (5) | Compression ratio → basin depth | T₃₅ |
//! | Attractor (5) | Bayesian (1) | Basin stability → prior strength | T₅₁ |
//!
//! Cycle: T₁₂ ∘ T₂₃ ∘ T₃₅ ∘ T₅₁ = I

#![no_std]

extern crate libm;
use core::f32;
use libm::sqrtf;

// ── Tensor Types ─────────────────────────────────────────────────────────────

/// A 4×4 transformation matrix with f32 elements.
/// Maps one rotation layer's state vector to another.
#[derive(Clone, Debug)]
#[repr(C, align(64))]
pub struct Tensor4 {
    pub data: [f32; 16], // row-major
}

impl Tensor4 {
    pub const fn identity() -> Self {
        Tensor4 {
            data: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    pub const fn zero() -> Self {
        Tensor4 { data: [0.0; 16] }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * 4 + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * 4 + c] = v;
    }

    /// Multiply two 4×4 tensors: C = A × B
    #[inline]
    pub fn mul(&self, other: &Tensor4) -> Tensor4 {
        let mut result = Tensor4::zero();
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Frobenius norm difference from identity: ||T - I||_F
    /// Measures how far from closure this tensor is.
    #[inline]
    pub fn diff_from_identity(&self) -> f32 {
        let identity = Tensor4::identity();
        let mut sum_sq = 0.0;
        for i in 0..16 {
            let d = self.data[i] - identity.data[i];
            sum_sq += d * d;
        }
        sqrtf(sum_sq)
    }
}

// ── Cycle State ──────────────────────────────────────────────────────────────

/// Full state for one Rotation cycle, representing the current transformation
/// between each pair of layers.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct CycleState {
    /// Transformations between layers (indexed by the 4×4 cycle ordering):
    // Layers: 0=Bayesian, 1=PID, 2=Compress, 3=Attractor (note: Layer 4 is this cycle itself)
    pub t_01: Tensor4,   // Bayesian → PID
    pub t_12: Tensor4,   // PID → Compress
    pub t_23: Tensor4,   // Compress → Attractor
    pub t_30: Tensor4,   // Attractor → Bayesian (closes the cycle)

    /// Adaptive cycle error threshold.
    pub epsilon: f32,

    /// Current cycle error: ||T₁₂∘T₂₃∘T₃₀∘T₀₁ - I||_F
    pub cycle_error: f32,

    /// Per-transformation error contribution (for bottleneck identification).
    pub per_link_error: [f32; 4],

    /// Cycle count (total rotations executed).
    pub cycle_count: u64,

    /// Running average of cycle_error (low-pass filtered).
    pub cycle_error_avg: f32,
}

impl CycleState {
    pub fn new(epsilon: f32) -> Self {
        CycleState {
            t_01: Tensor4::identity(),
            t_12: Tensor4::identity(),
            t_23: Tensor4::identity(),
            t_30: Tensor4::identity(),
            epsilon,
            cycle_error: 0.0,
            per_link_error: [0.0; 4],
            cycle_count: 0,
            cycle_error_avg: 0.0,
        }
    }

    /// Compute the full cycle product: T_cycle = T₁₂ · T₂₃ · T₃₀ · T₀₁
    #[inline]
    pub fn compute_cycle_product(&self) -> Tensor4 {
        let t_12_23 = self.t_12.mul(&self.t_23);
        let t_12_23_30 = t_12_23.mul(&self.t_30);
        t_12_23_30.mul(&self.t_01)
    }

    /// Compute and cache cycle error. Returns cycle_error.
    #[inline]
    pub fn validate(&mut self) -> f32 {
        let product = self.compute_cycle_product();
        let error = product.diff_from_identity();

        // Per-link bottleneck analysis:
        // Freeze each transformation in turn and measure error reduction.
        // The one that reduces error most is the bottleneck.
        for (link_idx, frozen) in [
            &self.t_12,
            &self.t_23,
            &self.t_30,
            &self.t_01,
        ].iter().enumerate() {
            // Temporarily replace link with identity to see how much error drops
            let mut test_state = self.clone();
            match link_idx {
                0 => test_state.t_12 = Tensor4::identity(),
                1 => test_state.t_23 = Tensor4::identity(),
                2 => test_state.t_30 = Tensor4::identity(),
                3 => test_state.t_01 = Tensor4::identity(),
                _ => unreachable!(),
            }
            let test_product = test_state.compute_cycle_product();
            self.per_link_error[link_idx] = test_product.diff_from_identity();
        }

        self.cycle_error = error;
        self.cycle_count += 1;

        // Running average (exponential, 0.01 decay)
        let alpha = 0.01f32;
        self.cycle_error_avg = self.cycle_error_avg * (1.0 - alpha) + error * alpha;

        error
    }

    /// Returns (bottleneck_layer_idx, bottleneck_value).
    /// Lower per_link_error means the frozen link contributed more to the error
    /// (i.e., that link is the bottleneck).
    #[inline]
    pub fn find_bottleneck(&self) -> (usize, f32) {
        let mut min_idx = 0;
        let mut min_val = self.per_link_error[0];
        for (i, &v) in self.per_link_error.iter().enumerate().skip(1) {
            if v < min_val {
                min_idx = i;
                min_val = v;
            }
        }
        (min_idx, min_val)
    }

    /// True if cycle is consistent within epsilon.
    #[inline]
    pub fn is_consistent(&self) -> bool {
        self.cycle_error < self.epsilon
    }

    /// Adapt epsilon based on cycle_error trend.
    /// If error is low and stable, tighten tolerance (learning).
    /// If error spikes, relax tolerance (adapting to new regime).
    #[inline]
    pub fn adapt_epsilon(&mut self) {
        let spike = self.cycle_error / (self.cycle_error_avg + f32::EPSILON);
        if spike > 3.0 {
            // Spike: relax tolerance
            self.epsilon = (self.epsilon * 1.2).min(1.0);
        } else if spike < 0.5 && self.cycle_count > 10 {
            // Stable: tighten
            self.epsilon = (self.epsilon * 0.95).max(0.001);
        }
    }
}

// ── Forgetting Factor Adaptation ─────────────────────────────────────────────

/// Update the forgetting factor γ based on cycle_error trend.
///
/// γ ← γ + η · (cycle_error_t - cycle_error_{t-1}) · (1 - γ)
///
/// Pushes toward 1.0 when stable (remember more), toward 0.0 when changing
/// (forget faster, adapt).
#[inline]
pub fn update_forgetting_factor(
    gamma: f32,
    cycle_error: f32,
    prev_cycle_error: f32,
    eta: f32,
) -> f32 {
    let delta = cycle_error - prev_cycle_error;
    let update = gamma + eta * delta * (1.0 - gamma);
    update.clamp(0.1, 0.999)
}

// ── State Vector ─────────────────────────────────────────────────────────────

/// A 4-element state vector for one layer:
///   [mean, variance, count, energy]
#[derive(Clone, Debug)]
#[repr(C)]
pub struct StateVector4 {
    pub data: [f32; 4],
}

impl StateVector4 {
    pub const fn zero() -> Self {
        StateVector4 { data: [0.0; 4] }
    }

    pub fn new(mean: f32, variance: f32, count: f32, energy: f32) -> Self {
        StateVector4 { data: [mean, variance, count, energy] }
    }

    /// Apply tensor transformation: v' = T × v
    #[inline]
    pub fn transform(&self, tensor: &Tensor4) -> Self {
        let mut result = [0.0f32; 4];
        for i in 0..4 {
            let mut sum = 0.0;
            for j in 0..4 {
                sum += tensor.get(i, j) * self.data[j];
            }
            result[i] = sum;
        }
        StateVector4 { data: result }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_identity_multiply() {
        let id = Tensor4::identity();
        let result = id.mul(&id);
        assert!((result.diff_from_identity()) < 1e-6, "I × I should be I");
    }

    #[test]
    fn test_cycle_identity_is_consistent() {
        let mut cycle = CycleState::new(0.1);
        let err = cycle.validate();
        assert!(err < 0.001, "identity cycle should have near-zero error");
        assert!(cycle.is_consistent());
    }

    #[test]
    fn test_cycle_with_perturbation_detected() {
        let mut cycle = CycleState::new(0.1);
        // Perturb one tensor
        cycle.t_12.set(0, 1, 0.5);
        let err = cycle.validate();
        assert!(err > 0.1, "perturbed cycle should have detectable error, got {}", err);
        assert!(!cycle.is_consistent(), "perturbed cycle should not be consistent");
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut cycle = CycleState::new(0.5);
        // Perturb t_12 significantly
        cycle.t_12.set(0, 0, 2.0); // major perturbation
        cycle.t_12.set(1, 1, 0.5);
        cycle.validate();
        let (idx, _val) = cycle.find_bottleneck();
        // The frozen link with LOWEST error is the bottleneck
        // Since t_12 is the main problem, freezing it (testing with identity)
        // should reduce error the most → lowest per_link_error for idx=0
        assert_eq!(idx, 0, "t_12 should be the bottleneck");
    }

    #[test]
    fn test_state_vector_transform() {
        let sv = StateVector4::new(1.0, 0.5, 10.0, 2.0);
        let identity = Tensor4::identity();
        let result = sv.transform(&identity);
        assert_eq!(result.data, sv.data);
    }

    #[test]
    fn test_epsilon_adaptation_tightens() {
        let mut cycle = CycleState::new(0.1);
        // Run several cycles with low error to tighten epsilon
        cycle.cycle_count = 20;
        cycle.cycle_error = 0.01;
        cycle.cycle_error_avg = 0.02;
        let old_eps = cycle.epsilon;
        cycle.adapt_epsilon();
        assert!(cycle.epsilon < old_eps, "epsilon should tighten when stable");
    }

    #[test]
    fn test_epsilon_adaptation_relaxes() {
        let mut cycle = CycleState::new(0.1);
        cycle.cycle_error = 0.5;
        cycle.cycle_error_avg = 0.02; // big spike
        let old_eps = cycle.epsilon;
        cycle.adapt_epsilon();
        assert!(cycle.epsilon > old_eps, "epsilon should relax on spike");
    }

    #[test]
    fn test_forgetting_factor_update() {
        let gamma = 0.9;
        // Stable: delta close to 0, gamma should increase slightly
        let new_gamma = update_forgetting_factor(gamma, 0.1, 0.09, 0.01);
        assert!(new_gamma > gamma, "stable environment should increase gamma");
        assert!(new_gamma <= 0.999);
    }
}
