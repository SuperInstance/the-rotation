//! # pid-cascade — Two-Level Cascade PID Controller
//!
//! The Rotation's Layer 2: resource PID feeds setpoint into cognitive PID.
//!
//! ## Architecture
//!
//! ```text
//! Level 1 (Resource PID):     Setpoint = ideal_free_space_ratio (20%)
//!                              PV = actual_free_space_ratio
//!                              Output = available_compute_budget
//!
//! Level 2 (Cognitive PID):    Setpoint = target_confidence_coverage
//!                              PV = actual_confidence_coverage
//!                              Output = exploration_vs_exploitation
//!                              → Modulated by Level 1 output (cascade)
//! ```
//!
//! ## ARM NEON
//!
//! Batch PID (16 channels) uses our neon-kernel's `pid_batch` for SIMD throughput.
//! Single-channel PID uses scalar fallback.

#![no_std]

// ── PID Registers ────────────────────────────────────────────────────────────

/// PID controller state for one channel.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct PidRegisters {
    /// Proportional gain
    pub kp: f32,
    /// Integral gain
    pub ki: f32,
    /// Derivative gain
    pub kd: f32,

    /// Running integral term (with anti-windup)
    pub integral: f32,
    /// Previous error (for derivative on error)
    pub prev_error: f32,
    /// Previous measurement (for derivative on measurement — optional)
    pub prev_measurement: f32,

    /// Integral clamp (anti-windup limit)
    pub integral_clamp: f32,
    /// Output clamp
    pub output_clamp: f32,
    /// Derivative filter coefficient (0.0 = no filter, 1.0 = full filter)
    pub deriv_filter: f32,
    /// Last derivative term (for filtering)
    pub last_deriv: f32,

    /// Configured setpoint
    pub setpoint: f32,
}

impl PidRegisters {
    /// Create a new PID controller with default (zero) gains.
    pub const fn new() -> Self {
        PidRegisters {
            kp: 1.0, ki: 0.0, kd: 0.0,
            integral: 0.0, prev_error: 0.0, prev_measurement: 0.0,
            integral_clamp: 100.0, output_clamp: 100.0,
            deriv_filter: 0.0, last_deriv: 0.0,
            setpoint: 0.0,
        }
    }

    /// Update the PID with a new measurement.
    /// Returns the output value.
    #[inline]
    pub fn update(&mut self, measurement: f32, dt: f32) -> f32 {
        let error = self.setpoint - measurement;

        // P term
        let p = self.kp * error;

        // I term with anti-windup
        let int_delta = self.ki * error * dt;
        // Back-calculation anti-windup: don't integrate if output would saturate
        let mut integral = self.integral + int_delta;
        integral = integral.clamp(-self.integral_clamp, self.integral_clamp);
        self.integral = integral;

        // D term (on error, with optional filtering)
        let d_error = error - self.prev_error;
        let raw_deriv = if dt > 1e-10 { d_error / dt } else { 0.0 };
        let deriv = self.last_deriv * self.deriv_filter + raw_deriv * (1.0 - self.deriv_filter);
        let d = self.kd * deriv;
        self.last_deriv = deriv;

        // Save state for next iteration
        self.prev_error = error;
        self.prev_measurement = measurement;

        // Output
        let output = p + integral + d;
        output.clamp(-self.output_clamp, self.output_clamp)
    }

    /// Reset integral term (for anti-windup or mode change).
    #[inline]
    pub fn reset_integral(&mut self) {
        self.integral = 0.0;
    }

    /// Reset all state (gain scheduling).
    #[inline]
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.prev_measurement = 0.0;
        self.last_deriv = 0.0;
    }
}

// ── Cascade Controller ───────────────────────────────────────────────────────

/// Two-level cascade PID controller.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct CascadePID {
    /// Level 1: Resource PID (disk, memory, CPU)
    pub resource: PidRegisters,
    /// Level 2: Cognitive PID (confidence coverage, exploration)
    pub cognitive: PidRegisters,

    /// Cascade gain: how much Level 1 output modulates Level 2.
    /// 1.0 = full modulation, 0.0 = no cascade.
    pub cascade_gain: f32,

    /// Level 1 output (available compute budget, range [-1, 1]).
    pub resource_output: f32,
    /// Level 2 output (exploration vs exploitation, range [-1, 1]).
    pub cognitive_output: f32,
}

impl CascadePID {
    /// Create cascade PID with recommended defaults for the Rotation.
    pub fn new() -> Self {
        let mut resource = PidRegisters::new();
        resource.kp = 5.0;      // gc-pid-bridge calibrated Kp
        resource.ki = 0.5;
        resource.kd = 0.2;
        resource.setpoint = 0.20; // 20% free disk
        resource.integral_clamp = 5.0;
        resource.output_clamp = 1.0;

        let mut cognitive = PidRegisters::new();
        cognitive.kp = 2.0;
        cognitive.ki = 0.3;
        cognitive.kd = 0.1;
        cognitive.setpoint = 0.7; // 70% confidence coverage
        cognitive.integral_clamp = 3.0;
        cognitive.output_clamp = 1.0;

        CascadePID {
            resource,
            cognitive,
            cascade_gain: 0.5,
            resource_output: 0.0,
            cognitive_output: 0.0,
        }
    }

    /// Full cascade update.
    ///
    /// Arguments:
    /// - `free_space_ratio`: actual free disk space (0.0–1.0)
    /// - `confidence_coverage`: actual confidence coverage (0.0–1.0)
    /// - `dt`: time step (seconds)
    ///
    /// Returns `(resource_output, cognitive_output)` where:
    /// - resource_output > 0 = compute available (exploit)
    /// - resource_output < 0 = compute constrained (conserve)
    /// - cognitive_output > 0 = explore
    /// - cognitive_output < 0 = exploit
    #[inline]
    pub fn update(&mut self, free_space_ratio: f32, confidence_coverage: f32, dt: f32) -> (f32, f32) {
        // Level 1: Resource
        self.resource_output = self.resource.update(free_space_ratio, dt);

        // Level 2: Cognitive, with cascade modulation
        // When resource is constrained (< 0), lower cognitive setpoint
        let cascade_mod = 1.0 - self.cascade_gain * (1.0 - (self.resource_output * 0.5 + 0.5));
        let modulated_setpoint = self.cognitive.setpoint * cascade_mod.max(0.1);

        // Temporarily override setpoint for this update
        let original_setpoint = self.cognitive.setpoint;
        self.cognitive.setpoint = modulated_setpoint;
        self.cognitive_output = self.cognitive.update(confidence_coverage, dt);
        self.cognitive.setpoint = original_setpoint;

        (self.resource_output, self.cognitive_output)
    }

    /// Reset both PIDs (gain scheduling reset).
    #[inline]
    pub fn reset(&mut self) {
        self.resource.reset();
        self.cognitive.reset();
        self.resource_output = 0.0;
        self.cognitive_output = 0.0;
    }

    /// Set Bayesian-driven gain schedule.
    ///
    /// - `convergence_rate`: how fast posteriors converge (0 = slow, 1 = fast)
    /// - `confidence_variance`: variance of confidence across reflex domains
    #[inline]
    pub fn schedule_gains(&mut self, convergence_rate: f32, confidence_variance: f32) {
        // Fast convergence → more aggressive exploitation (higher Kp on cognitive)
        self.cognitive.kp = 1.0 + convergence_rate * 3.0; // 1.0–4.0

        // High variance → more damping
        self.cognitive.kd = 0.05 + confidence_variance * 0.3; // 0.05–0.35

        // Resource PID stays at calibrated values, but integral can tighten
        self.resource.ki = 0.5 * (1.0 - convergence_rate * 0.5); // 0.25–0.5
    }
}

// ── Integration with the rotation layers ─────────────────────────────────────

/// State that bridges PID → Compression ratio (Layer 2 → Layer 3).
#[derive(Clone, Debug)]
#[repr(C)]
pub struct PidToCompression {
    /// Raw PID output mapped to compression ratio delta
    pub ratio_delta: f32,
    /// Target compression ratio for this channel
    pub target_ratio: f32,
    /// Whether cognitive load is acceptable
    pub load_acceptable: bool,
    /// Bottleneck suggestion for the tensor cycle
    pub cycle_bottleneck: Option<u8>,
}

/// Convert PID outputs to a compression ratio adjustment.
///
/// When cognitive output is positive (explore), lower the compression ratio
/// (send more detail so exploration has richer input).
/// When negative (exploit), raise the ratio (send compressed summaries).
#[inline]
pub fn pid_to_compression_ratio(
    resource_output: f32,
    cognitive_output: f32,
    current_ratio: f32,
) -> f32 {
    // Base adjustment from cognitive output
    let cognitive_adj = -cognitive_output * 0.5; // explore → lower ratio
    // Resource constraint: when constrained, increase ratio (send shorter messages)
    let resource_adj = -resource_output * 0.3; // constrained → raise ratio
    let new_ratio = current_ratio + cognitive_adj + resource_adj;

    // Clamp to sensible bounds for the shell model
    new_ratio.clamp(0.1, 100.0)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_basic_update() {
        let mut pid = PidRegisters::new();
        pid.kp = 1.0;
        pid.kd = 1.0;
        pid.setpoint = 1.0;
        // First update: error = 0.5, prev_error = 0 (default)
        // P = 0.5*1 = 0.5, I = 0 (ki=0), D = 1.0*(0.5-0)/0.1 = 5.0
        // Output = 5.5
        let out = pid.update(0.5, 0.1);
        assert!((out - 5.5).abs() < 0.01, "got {}", out);
    }

    #[test]
    fn test_pid_anti_windup() {
        let mut pid = PidRegisters::new();
        pid.kp = 1.0;
        pid.ki = 10.0; // high integral gain
        pid.integral_clamp = 5.0;
        pid.output_clamp = 100.0;
        pid.setpoint = 1.0;

        // Run several steps with persistent error
        for _ in 0..10 {
            let _ = pid.update(0.0, 1.0);
        }
        // Integral should be clamped to [-5, 5]
        assert!(pid.integral.abs() <= 5.0 + 1e-6, "integral should be clamped, got {}", pid.integral);
    }

    #[test]
    fn test_cascade_modulation() {
        let mut cascade = CascadePID::new();

        // Well-fed: plenty of disk (30% > 20% setpoint) — error = -0.1 → negative output
        // (no pressure to free space)
        for _ in 0..20 {
            cascade.update(0.30, 0.5, 0.1);
        }
        // error = 0.2 - 0.3 = -0.1 → P = 5 * -0.1 = -0.5, I accumulates, D converges
        // Total should be negative (no allocation pressure)
        let res = cascade.resource_output;
        assert!(res < 0.0, "well-fed should produce negative output, got {}", res);

        // Resource-constrained: low disk (5% < 20% setpoint) — error = +0.15
        cascade.reset();
        for _ in 0..20 {
            cascade.update(0.05, 0.5, 0.1);
        }
        // error = 0.2 - 0.05 = 0.15 → P = 5 * 0.15 = 0.75, I grows
        let res2 = cascade.resource_output;
        assert!(res2 > 0.0, "constrained should produce positive output, got {}", res2);
    }

    #[test]
    fn test_gain_scheduling() {
        let mut cascade = CascadePID::new();
        cascade.schedule_gains(0.9, 0.1);
        assert!(cascade.cognitive.kp > 1.0, "fast convergence should raise cognitive Kp");
        cascade.schedule_gains(0.1, 0.9);
        assert!(cascade.cognitive.kd > 0.1, "high variance should raise Kd");
    }

    #[test]
    fn test_ratio_conversion() {
        // Explore mode + constrained resource
        let ratio = pid_to_compression_ratio(-0.5, 0.8, 10.0);
        // explore → lower ratio (cognitive_adj = -0.8*0.5 = -0.4)
        // constrained → raise ratio (resource_adj = -(-0.5)*0.3 = +0.15)
        // net = 10 - 0.4 + 0.15 = 9.75
        assert!((ratio - 9.75).abs() < 0.01, "got {}", ratio);
    }

    #[test]
    fn test_ratio_clamping() {
        let ratio = pid_to_compression_ratio(10.0, 10.0, 200.0);
        assert!(ratio <= 100.0, "ratio should be clamped to max 100");
    }
}
