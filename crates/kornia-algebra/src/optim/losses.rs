//! Robust loss functions for least squares optimization.

/// Trait for robust loss functions that scale residuals by weight.
pub trait RobustLoss: Send + Sync {
    /// Compute loss weight for a squared residual norm.
    fn weight(&self, squared_norm: f32) -> f32;
}

/// Identity loss: weight always 1.0 (L2 baseline).
#[derive(Debug, Clone, Copy)]
pub struct IdentityLoss;

impl RobustLoss for IdentityLoss {
    fn weight(&self, _squared_norm: f32) -> f32 {
        1.0
    }
}

/// Huber loss: smooth transition from quadratic to linear weighting.
#[derive(Debug, Clone, Copy)]
pub struct HuberLoss {
    pub delta: f32,
}

impl HuberLoss {
    /// Create new Huber loss. Panics if delta <= 0.
    pub fn new(delta: f32) -> Self {
        assert!(delta > 0.0, "Huber delta must be positive, got {}", delta);
        HuberLoss { delta }
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
}

/// Cauchy loss: aggressive outlier rejection via 1/(1 + s/σ²).
#[derive(Debug, Clone, Copy)]
pub struct CauchyLoss {
    pub scale: f32,
}

impl CauchyLoss {
    /// Create new Cauchy loss. Panics if scale <= 0.
    pub fn new(scale: f32) -> Self {
        assert!(scale > 0.0, "Cauchy scale must be positive, got {}", scale);
        CauchyLoss { scale }
    }
}

impl RobustLoss for CauchyLoss {
    fn weight(&self, squared_norm: f32) -> f32 {
        let scale_sq = self.scale * self.scale;
        1.0 / (1.0 + squared_norm / scale_sq)
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
        let huber = HuberLoss::new(1.0);
        assert_eq!(huber.weight(0.0), 1.0); // Zero: quadratic region
        assert_eq!(huber.weight(1.0), 1.0); // At threshold
        assert!((huber.weight(4.0) - 0.5).abs() < 1e-6); // Large: 1/sqrt(4)
        assert!(huber.weight(10000.0).is_finite()); // Very large: no NaN
    }

    #[test]
    fn test_huber_loss_parameter_validation() {
        assert!(HuberLoss::new(0.5).delta > 0.0);
        assert!(HuberLoss::new(2.0).delta > 0.0);
    }

    #[test]
    #[should_panic(expected = "Huber delta must be positive")]
    fn test_huber_loss_zero_delta_panics() {
        HuberLoss::new(0.0);
    }

    #[test]
    #[should_panic(expected = "Huber delta must be positive")]
    fn test_huber_loss_negative_delta_panics() {
        HuberLoss::new(-1.0);
    }

    #[test]
    fn test_cauchy_loss_weights() {
        let cauchy = CauchyLoss::new(1.0);
        assert_eq!(cauchy.weight(0.0), 1.0); // Zero: max weight
        assert!((cauchy.weight(0.01) - (1.0 / 1.01)).abs() < 1e-4); // Small
        assert!((cauchy.weight(10.0) - (1.0 / 11.0)).abs() < 1e-4); // Large
        assert!(cauchy.weight(100000.0) > 0.0 && cauchy.weight(100000.0) < 0.00002);
        // Very large
    }

    #[test]
    fn test_cauchy_loss_scales() {
        let c1 = CauchyLoss::new(1.0);
        let c2 = CauchyLoss::new(2.0);
        assert!(c2.weight(4.0) > c1.weight(4.0)); // Larger scale → larger weight
    }

    #[test]
    #[should_panic(expected = "Cauchy scale must be positive")]
    fn test_cauchy_loss_zero_scale_panics() {
        CauchyLoss::new(0.0);
    }

    #[test]
    #[should_panic(expected = "Cauchy scale must be positive")]
    fn test_cauchy_loss_negative_scale_panics() {
        CauchyLoss::new(-1.0);
    }

    #[test]
    fn test_huber_vs_cauchy_outlier_behavior() {
        let huber = HuberLoss::new(1.0);
        let cauchy = CauchyLoss::new(1.0);
        let squared_norm = 100.0;
        let w_huber = huber.weight(squared_norm);
        let w_cauchy = cauchy.weight(squared_norm);
        assert!((w_huber - 0.1).abs() < 1e-6);
        assert!(w_cauchy < w_huber); // Cauchy more aggressive
    }
}
