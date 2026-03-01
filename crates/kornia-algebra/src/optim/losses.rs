//! Robust loss functions for least squares optimization.

use crate::optim::FactorError;

/// Trait for robust loss functions that scale residuals by weight.
pub trait RobustLoss: Send + Sync {
    /// Compute loss weight for a squared residual norm.
    fn weight(&self, squared_norm: f32) -> f32;

    /// Evaluate the robust cost function ρ(s²).
    ///
    /// Returns the true robust loss value for the given squared residual norm.
    /// Used by the optimizer for step acceptance and termination checks.
    fn rho(&self, squared_norm: f32) -> f32 {
        squared_norm
    }
}

/// Identity loss: weight always 1.0 (L2 baseline).
#[derive(Debug, Clone, Copy)]
pub struct IdentityLoss;

impl RobustLoss for IdentityLoss {
    fn weight(&self, _squared_norm: f32) -> f32 {
        1.0
    }

    fn rho(&self, squared_norm: f32) -> f32 {
        squared_norm
    }
}

/// Huber loss: smooth transition from quadratic to linear weighting.
#[derive(Debug, Clone, Copy)]
pub struct HuberLoss {
    delta: f32,
}

impl HuberLoss {
    /// Create new Huber loss. Returns error if delta <= 0.
    pub fn new(delta: f32) -> Result<Self, FactorError> {
        if delta <= 0.0 {
            return Err(FactorError::InvalidParameters(
                "Huber delta must be positive".into(),
            ));
        }
        Ok(HuberLoss { delta })
    }
}

impl RobustLoss for HuberLoss {
    fn weight(&self, squared_norm: f32) -> f32 {
        let delta_sq = self.delta * self.delta;
        if squared_norm <= delta_sq {
            1.0
        } else {
            self.delta / squared_norm.sqrt()
        }
    }

    fn rho(&self, squared_norm: f32) -> f32 {
        let s = squared_norm.sqrt();
        if s <= self.delta {
            squared_norm
        } else {
            2.0 * self.delta * s - self.delta * self.delta
        }
    }
}

/// Cauchy loss: aggressive outlier rejection via 1/(1 + s²/σ²).
#[derive(Debug, Clone, Copy)]
pub struct CauchyLoss {
    scale: f32,
}

impl CauchyLoss {
    /// Create new Cauchy loss. Returns error if scale <= 0.
    pub fn new(scale: f32) -> Result<Self, FactorError> {
        if scale <= 0.0 {
            return Err(FactorError::InvalidParameters(
                "Cauchy scale must be positive".into(),
            ));
        }
        Ok(CauchyLoss { scale })
    }
}

impl RobustLoss for CauchyLoss {
    fn weight(&self, squared_norm: f32) -> f32 {
        let scale_sq = self.scale * self.scale;
        1.0 / (1.0 + squared_norm / scale_sq)
    }

    fn rho(&self, squared_norm: f32) -> f32 {
        let scale_sq = self.scale * self.scale;
        scale_sq * (1.0 + squared_norm / scale_sq).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_loss_always_one() {
        let loss = IdentityLoss;
        assert_eq!(loss.weight(0.0), 1.0);
        assert_eq!(loss.weight(1.0), 1.0);
        assert_eq!(loss.weight(100.0), 1.0);
    }

    #[test]
    fn test_huber_loss_weights() {
        let huber = HuberLoss::new(1.0).unwrap();
        assert_eq!(huber.weight(0.0), 1.0); // Zero: quadratic region
        assert_eq!(huber.weight(1.0), 1.0); // At threshold
        assert!((huber.weight(4.0) - 0.5).abs() < 1e-6); // Large: 1/sqrt(4)
        assert!(huber.weight(10000.0).is_finite()); // Very large: no NaN
    }

    #[test]
    fn test_huber_loss_parameter_validation() {
        assert!(HuberLoss::new(0.5).is_ok());
        assert!(HuberLoss::new(2.0).is_ok());
    }

    #[test]
    fn test_huber_loss_zero_delta_error() {
        assert!(HuberLoss::new(0.0).is_err());
    }

    #[test]
    fn test_huber_loss_negative_delta_error() {
        assert!(HuberLoss::new(-1.0).is_err());
    }

    #[test]
    fn test_cauchy_loss_weights() {
        let cauchy = CauchyLoss::new(1.0).unwrap();
        assert_eq!(cauchy.weight(0.0), 1.0); // Zero: max weight
        assert!((cauchy.weight(0.01) - (1.0 / 1.01)).abs() < 1e-4); // Small
        assert!((cauchy.weight(10.0) - (1.0 / 11.0)).abs() < 1e-4); // Large
        assert!(cauchy.weight(100000.0) > 0.0 && cauchy.weight(100000.0) < 0.00002);
        // Very large
    }

    #[test]
    fn test_cauchy_loss_scales() {
        let c1 = CauchyLoss::new(1.0).unwrap();
        let c2 = CauchyLoss::new(2.0).unwrap();
        assert!(c2.weight(4.0) > c1.weight(4.0)); // Larger scale → larger weight
    }

    #[test]
    fn test_cauchy_loss_zero_scale_error() {
        assert!(CauchyLoss::new(0.0).is_err());
    }

    #[test]
    fn test_cauchy_loss_negative_scale_error() {
        assert!(CauchyLoss::new(-1.0).is_err());
    }

    #[test]
    fn test_huber_vs_cauchy_outlier_behavior() {
        let huber = HuberLoss::new(1.0).unwrap();
        let cauchy = CauchyLoss::new(1.0).unwrap();
        let squared_norm = 100.0;
        let w_huber = huber.weight(squared_norm);
        let w_cauchy = cauchy.weight(squared_norm);
        assert!((w_huber - 0.1).abs() < 1e-6);
        assert!(w_cauchy < w_huber); // Cauchy more aggressive
    }

    // --- rho() tests ---

    #[test]
    fn test_identity_rho_equals_squared_norm() {
        let loss = IdentityLoss;
        assert_eq!(loss.rho(0.0), 0.0);
        assert_eq!(loss.rho(1.0), 1.0);
        assert_eq!(loss.rho(25.0), 25.0);
    }

    #[test]
    fn test_huber_rho_quadratic_region() {
        let huber = HuberLoss::new(2.0).unwrap();
        // s = sqrt(1.0) = 1.0 <= delta=2.0 → rho = s²
        assert_eq!(huber.rho(1.0), 1.0);
        // At boundary: s² = 4.0, s = 2.0 = delta → rho = s² = 4.0
        assert_eq!(huber.rho(4.0), 4.0);
    }

    #[test]
    fn test_huber_rho_linear_region() {
        let huber = HuberLoss::new(1.0).unwrap();
        // s² = 9.0, s = 3.0 > delta=1.0 → rho = 2*1*3 - 1 = 5.0
        assert!((huber.rho(9.0) - 5.0).abs() < 1e-6);
        // s² = 100.0, s = 10.0 → rho = 2*1*10 - 1 = 19.0
        assert!((huber.rho(100.0) - 19.0).abs() < 1e-5);
    }

    #[test]
    fn test_huber_rho_less_than_l2() {
        let huber = HuberLoss::new(1.0).unwrap();
        // In linear region, rho(s²) < s²
        assert!(huber.rho(100.0) < 100.0);
    }

    #[test]
    fn test_cauchy_rho_values() {
        let cauchy = CauchyLoss::new(1.0).unwrap();
        // rho(0) = 1 * ln(1) = 0
        assert_eq!(cauchy.rho(0.0), 0.0);
        // rho(1) = 1 * ln(2) ≈ 0.6931
        assert!((cauchy.rho(1.0) - 2.0_f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_cauchy_rho_less_than_l2() {
        let cauchy = CauchyLoss::new(1.0).unwrap();
        assert!(cauchy.rho(10.0) < 10.0);
        assert!(cauchy.rho(100.0) < 100.0);
    }

    #[test]
    fn test_rho_ordering_identity_huber_cauchy() {
        let identity = IdentityLoss;
        let huber = HuberLoss::new(1.0).unwrap();
        let cauchy = CauchyLoss::new(1.0).unwrap();
        let s2 = 25.0;
        // For outliers: Cauchy < Huber < Identity
        assert!(cauchy.rho(s2) < huber.rho(s2));
        assert!(huber.rho(s2) < identity.rho(s2));
    }
}
