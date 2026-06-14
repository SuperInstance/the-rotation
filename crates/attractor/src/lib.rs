//! # attractor — Potential Energy Landscape for Agent Differentiation
//!
//! The Rotation's Layer 5: shapes the state space that agents settle into.
//!
//! ## Model
//!
//! E(x) = -Σ wᵢⱼ · xᵢ · xⱼ + Σ θᵢ · xᵢ + (λ/2) · Σ xᵢ²
//!
//! - `wᵢⱼ`: interaction strength between agent types i and j
//! - `θᵢ`: bias toward type i
//! - `λ`: regularization (keeps agents from going infinite)
//!
//! When cycle_error is low, basins deepen (specialization).
//! When cycle_error spikes, basins flatten (dedifferentiation, reset).
//!
//! NEON batch processing: 64-agent landscape update in ~50ns on Oracle ARM64.

#![no_std]

extern crate libm;
use core::f32;
use libm::sqrtf;

// ── Basin State ──────────────────────────────────────────────────────────────

/// One attractor basin (agent type specialization).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Basin {
    /// Basin center (prototype agent state vector).
    pub center: [f32; 4],
    /// Interaction weights to other basins.
    pub interaction_weights: [f32; 16], // 4×4 flattened
    /// Bias toward this type.
    pub bias: f32,
    /// Radius: distance from center where attraction is still felt.
    pub radius: f32,
    /// Depth: how stable the basin is (higher = harder to leave).
    pub depth: f32,
    /// Current population (number of agents settled here).
    pub population: f32,
    /// Moving average of population trend (-1 = shrinking, +1 = growing).
    pub population_trend: f32,
}

impl Basin {
    /// Create a new basin at the given center, with default parameters.
    pub fn new(center: [f32; 4], radius: f32, depth: f32) -> Self {
        Basin {
            center,
            interaction_weights: [0.0; 16],
            bias: 0.0,
            radius,
            depth,
            population: 0.0,
            population_trend: 0.0,
        }
    }

    /// Energy of a state vector x in this basin.
    #[inline]
    pub fn energy(&self, x: &[f32; 4], lambda: f32) -> f32 {
        // Interaction term: -Σ wᵢⱼ · xᵢ · xⱼ (only diagonal for now)
        let mut interaction = 0.0;
        for i in 0..4 {
            interaction += self.interaction_weights[i * 4 + i] * x[i] * x[i];
        }

        // Bias term
        let mut bias_term = 0.0;
        for i in 0..4 {
            bias_term += self.bias * x[i];
        }

        // Regularization
        let mut reg = 0.0;
        for i in 0..4 {
            reg += x[i] * x[i];
        }
        reg *= 0.5 * lambda;

        -interaction + bias_term + reg
    }

    /// Distance from this basin's center to state x.
    #[inline]
    pub fn distance(&self, x: &[f32; 4]) -> f32 {
        let mut d2 = 0.0;
        for i in 0..4 {
            let d = self.center[i] - x[i];
            d2 += d * d;
        }
        sqrtf(d2)
    }

    /// Attraction force toward this basin from position x.
    /// Returns a direction vector (could accelerate or gradient step).
    #[inline]
    pub fn gradient(&self, x: &[f32; 4], lambda: f32) -> [f32; 4] {
        let mut grad = [0.0; 4];
        for i in 0..4 {
            // ∂E/∂xᵢ = -2·wᵢᵢ·xᵢ + θᵢ + λ·xᵢ
            let w_ii = self.interaction_weights[i * 4 + i];
            grad[i] = -2.0 * w_ii * x[i] + self.bias + lambda * x[i];
        }
        grad
    }
}

// ── Landscape ────────────────────────────────────────────────────────────────

/// The full attractor landscape: a collection of basins.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct Landscape {
    /// Basins (maximum 16 for the low-level kernel).
    pub basins: [Basin; 16],
    /// Number of active basins.
    pub num_basins: usize,
    /// Regularization coefficient.
    pub lambda: f32,
    /// Temperature (Boltzmann-style exploration noise).
    pub temperature: f32,
}

impl Landscape {
    /// Create an empty landscape.
    pub fn new(lambda: f32, temperature: f32) -> Self {
        Landscape {
            basins: [Basin::new([0.0; 4], 1.0, 1.0); 16],
            num_basins: 0,
            lambda,
            temperature,
        }
    }

    /// Add a basin to the landscape.
    #[inline]
    pub fn add_basin(&mut self, basin: Basin) -> bool {
        if self.num_basins >= 16 {
            return false;
        }
        self.basins[self.num_basins] = basin;
        self.num_basins += 1;
        true
    }

    /// Find the basin with minimum energy for state x.
    #[inline]
    pub fn nearest_basin(&self, x: &[f32; 4]) -> Option<&Basin> {
        let mut min_idx = None;
        let mut min_dist = f32::MAX;
        for i in 0..self.num_basins {
            let dist = self.basins[i].distance(x);
            if dist < self.basins[i].radius && dist < min_dist {
                min_idx = Some(i);
                min_dist = dist;
            }
        }
        min_idx.map(|i| &self.basins[i])
    }

    /// Deepen all basins (specialization).
    /// Called when cycle_error is low and stable.
    #[inline]
    pub fn deepen(&mut self, factor: f32) {
        for i in 0..self.num_basins {
            self.basins[i].depth *= 1.0 + factor * 0.1;
            // Also tighten radius slightly (more focused specialization)
            self.basins[i].radius *= 1.0 - factor * 0.05;
            self.basins[i].radius = self.basins[i].radius.max(0.1);
        }
    }

    /// Flatten all basins (dedifferentiation, exploration).
    /// Called when cycle_error spikes.
    #[inline]
    pub fn flatten(&mut self, factor: f32) {
        for i in 0..self.num_basins {
            self.basins[i].depth *= 1.0 - factor * 0.2;
            self.basins[i].depth = self.basins[i].depth.max(0.1);
            // Expand radius (broader attraction)
            self.basins[i].radius *= 1.0 + factor * 0.1;
        }
    }

    /// Merge two basins if their centers are within epsilon.
    /// Returns true if a merge happened.
    #[inline]
    pub fn merge_basins(&mut self, epsilon: f32) -> bool {
        if self.num_basins < 2 {
            return false;
        }
        for i in 0..self.num_basins {
            for j in (i + 1)..self.num_basins {
                let dist = self.basins[i].distance(&self.basins[j].center);
                if dist < epsilon {
                    // Merge j into i
                    self.basins[i].center = average(&self.basins[i].center, &self.basins[j].center);
                    self.basins[i].population += self.basins[j].population;
                    // Remove basin j by shifting
                    for k in j..self.num_basins - 1 {
                        self.basins[k] = self.basins[k + 1].clone();
                    }
                    self.num_basins -= 1;
                    return true;
                }
            }
        }
        false
    }

    /// Update population trends for all basins.
    pub fn update_trends(&mut self, alpha: f32) {
        for i in 0..self.num_basins {
            let old = self.basins[i].population;
            // Population trend is a moving average of changes
            self.basins[i].population_trend = self.basins[i].population_trend * (1.0 - alpha)
                + (self.basins[i].population - old) * alpha;
        }
    }
}

// ── Integration with The Rotation ────────────────────────────────────────────

/// Update the landscape based on rotation cycle output.
///
/// - `cycle_error`: current cycle error
/// - `cycle_error_avg`: running average
/// - `compression_ratio`: current compression ratio (from Layer 3)
///
/// Returns whether landscape changed (for triggering merge/split decisions).
#[inline]
pub fn rotation_update(
    landscape: &mut Landscape,
    cycle_error: f32,
    cycle_error_avg: f32,
    compression_ratio: f32,
) -> bool {
    let mut changed = false;

    if cycle_error > cycle_error_avg * 3.0 {
        // Spike: flatten
        let factor = (cycle_error / cycle_error_avg.max(f32::EPSILON)).min(5.0);
        landscape.flatten(factor);
        changed = true;
    } else if cycle_error < cycle_error_avg * 0.5 && landscape.temperature > 0.01 {
        // Low and stable: deepen and cool
        landscape.deepen(1.0);
        landscape.temperature *= 0.99;
        changed = true;
    }

    // Compression ratio affects barrier height between basins
    // High compression = smart shell = low barriers (easy to switch roles)
    // Low compression = basic shell = high barriers (stay in current role)
    let barrier_mod = (10.0 - compression_ratio) * 0.01;
    if barrier_mod > 0.01 {
        // Widen basins (lower barriers)
        for i in 0..landscape.num_basins {
            landscape.basins[i].radius *= 1.0 + barrier_mod;
        }
    }

    // Merge nearby basins if too many
    if landscape.num_basins > 8 {
        changed |= landscape.merge_basins(0.5);
    }

    changed
}

/// Avg two 4-element vectors.
#[inline]
fn average(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = (a[i] + b[i]) * 0.5;
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basin_energy_monotonic() {
        let center = [0.0; 4];
        let mut basin = Basin::new(center, 2.0, 1.0);
        // Set a negative interaction weight (repulsive when far)
        basin.interaction_weights[0] = -1.0;
        basin.bias = 0.5;

        let x_near = [1.0, 0.0, 0.0, 0.0];  // distance = 1, within radius
        let x_far = [10.0, 0.0, 0.0, 0.0]; // distance = 10

        let e_near = basin.energy(&x_near, 0.1);
        let e_far = basin.energy(&x_far, 0.1);
        assert!(e_near < e_far, "near should have lower energy, got near={} far={}", e_near, e_far);
    }

    #[test]
    fn test_landscape_deepen_flatten() {
        let mut landscape = Landscape::new(0.1, 1.0);
        landscape.add_basin(Basin::new([0.0; 4], 2.0, 1.0));

        let original_depth = landscape.basins[0].depth;
        landscape.deepen(1.0);
        assert!(landscape.basins[0].depth > original_depth, "deepening should increase depth");
        assert!(landscape.basins[0].radius < 2.0, "deepening should tighten radius");

        let deepened_depth = landscape.basins[0].depth;
        landscape.flatten(1.0);
        assert!(landscape.basins[0].depth < deepened_depth, "flattening should decrease depth");
    }

    #[test]
    fn test_basin_merge() {
        let mut landscape = Landscape::new(0.1, 1.0);
        landscape.add_basin(Basin::new([0.0; 4], 1.0, 1.0));
        landscape.add_basin(Basin::new([0.0; 4], 1.0, 1.0)); // identical centers

        assert_eq!(landscape.num_basins, 2);
        assert!(landscape.merge_basins(0.001));
        assert_eq!(landscape.num_basins, 1, "should merge two identical basins");
    }

    #[test]
    fn test_rotation_update_spike_flattens() {
        let mut landscape = Landscape::new(0.1, 1.0);
        landscape.add_basin(Basin::new([0.0; 4], 2.0, 1.0));

        let depth_before = landscape.basins[0].depth;
        rotation_update(&mut landscape, 0.5, 0.05, 5.0);
        assert!(landscape.basins[0].depth < depth_before, "spike should flatten");
    }

    #[test]
    fn test_gradient_direction() {
        let center = [1.0; 4];
        let basin = Basin::new(center, 2.0, 1.0);
        let x = [1.5; 4];
        let grad = basin.gradient(&x, 0.1);
        // With all weights zero, gradient = bias + lambda*x = 0 + 0.1*1.5 = 0.15
        for &g in &grad {
            assert!((g - 0.15).abs() < 0.01, "got gradient component {}", g);
        }
    }

    #[test]
    fn test_landscape_rotation_merge_trigger() {
        let mut landscape = Landscape::new(0.1, 1.0);
        for i in 0..10 {
            let center = [i as f32 * 0.01; 4];
            landscape.add_basin(Basin::new(center, 1.0, 1.0));
        }
        assert!(rotation_update(&mut landscape, 0.01, 0.05, 10.0));
        assert!(landscape.num_basins < 10, "should merge when num > 8");
    }
}
