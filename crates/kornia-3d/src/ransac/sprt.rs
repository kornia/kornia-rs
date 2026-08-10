//! Core mathematical logic for Wald's Sequential Probability Ratio Test (SPRT) in RANSAC.
//!
//! SPRT allows the RANSAC driver to reject "bad" model hypotheses early by evaluating points
//! sequentially and stopping as soon as the Log-Likelihood Ratio (LLR) exceeds a calculated threshold.

/// Default probability of accepting a bad model (Type II error).
pub const DEFAULT_BETA: f64 = 0.1;

/// Default probability that a bad model accepts a point by chance.
pub const DEFAULT_CHANCE_PROB: f64 = 0.01;

/// Configuration for SPRT thresholds and timing.
#[derive(Debug, Clone, Copy)]
pub struct SPRTConfig {
    /// Expected inlier ratio ($\epsilon$).
    pub epsilon: f64,
    /// Probability of a good model being rejected (Type I error, $\alpha$).
    pub delta: f64,
    /// Model instantiation time ($t_M$).
    pub t_M: f64,
    /// Single point evaluation time ($t_m$).
    pub t_m: f64,
}

impl SPRTConfig {
    /// Calculate the decision threshold $A = \ln((1 - \beta) / \alpha)$.
    pub fn calculate_threshold(&self, beta: f64) -> f64 {
        ((1.0 - beta) / self.delta).ln()
    }
}

/// State for a single SPRT session during model verification.
#[derive(Debug, Clone, Copy)]
pub struct SPRTState {
    /// Current log-likelihood ratio.
    pub llr: f64,
    /// Number of points tested so far.
    pub num_tested: usize,
    /// Precomputed threshold for rejection.
    pub decision_threshold: f64,
}

impl SPRTState {
    /// Create a new SPRT state based on a config.
    pub fn new(config: &SPRTConfig, beta: f64) -> Self {
        Self {
            llr: 0.0,
            num_tested: 0,
            decision_threshold: config.calculate_threshold(beta),
        }
    }

    /// Update the LLR based on whether the current point is an inlier.
    ///
    /// LLR update:
    /// - Inlier: $\ln(\delta_{chance} / \epsilon)$
    /// - Outlier: $\ln((1 - \delta_{chance}) / (1 - \epsilon))$
    pub fn update(&mut self, is_inlier: bool, epsilon: f64, delta_chance: f64) {
        // Clamp probabilities to avoid ln(0)
        let epsilon = epsilon.clamp(1e-10, 1.0 - 1e-10);
        let delta_chance = delta_chance.clamp(1e-10, 1.0 - 1e-10);

        let step = if is_inlier {
            (delta_chance / epsilon).ln()
        } else {
            ((1.0 - delta_chance) / (1.0 - epsilon)).ln()
        };
        self.llr += step;
        self.num_tested += 1;
    }

    /// Check if the current LLR exceeds the decision threshold, meaning the model should be rejected.
    pub fn is_rejected(&self) -> bool {
        self.llr > self.decision_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_threshold_calculation() {
        let config = SPRTConfig {
            epsilon: 0.5,
            delta: 0.01,
            t_M: 1.0,
            t_m: 0.1,
        };
        // A = ln((1 - 0.1) / 0.01) = ln(90) approx 4.4998
        let threshold = config.calculate_threshold(DEFAULT_BETA);
        assert!((threshold - 4.4998).abs() < 1e-4);
    }

    #[test]
    fn test_likelihood_ratio_inliers() {
        let config = SPRTConfig {
            epsilon: 0.3,
            delta: 0.01,
            t_M: 1.0,
            t_m: 0.1,
        };
        let mut state = SPRTState::new(&config, DEFAULT_BETA);
        let initial_llr = state.llr;

        // For inliers: ln(0.01 / 0.3) approx -3.401
        state.update(true, config.epsilon, DEFAULT_CHANCE_PROB);
        assert!(state.llr < initial_llr);
        assert!((state.llr - (-3.401)).abs() < 1e-3);
    }

    #[test]
    fn test_likelihood_ratio_outliers() {
        let config = SPRTConfig {
            epsilon: 0.3,
            delta: 0.01,
            t_M: 1.0,
            t_m: 0.1,
        };
        let mut state = SPRTState::new(&config, DEFAULT_BETA);
        let initial_llr = state.llr;

        // For outliers: ln((1 - 0.01) / (1 - 0.3)) = ln(0.99 / 0.7) approx 0.3466
        state.update(false, config.epsilon, DEFAULT_CHANCE_PROB);
        assert!(state.llr > initial_llr);
        assert!((state.llr - 0.3466).abs() < 1e-3);
    }

    #[test]
    fn test_dynamic_epsilon_update() {
        let config = SPRTConfig {
            epsilon: 0.3,
            delta: 0.01,
            t_M: 1.0,
            t_m: 0.1,
        };
        let mut state = SPRTState::new(&config, DEFAULT_BETA);

        // Test with epsilon = 0.3
        state.update(true, 0.3, DEFAULT_CHANCE_PROB);
        let llr_1 = state.llr;

        // Test with epsilon = 0.5
        let mut state2 = SPRTState::new(&config, DEFAULT_BETA);
        state2.update(true, 0.5, DEFAULT_CHANCE_PROB);
        let llr_2 = state2.llr;

        // ln(0.01 / 0.5) is more negative than ln(0.01 / 0.3)
        assert!(llr_2 < llr_1);
    }

    #[test]
    fn test_rejection_logic() {
        let config = SPRTConfig {
            epsilon: 0.3,
            delta: 0.01,
            t_M: 1.0,
            t_m: 0.1,
        };
        let mut state = SPRTState::new(&config, DEFAULT_BETA);

        // Force rejection by adding many outliers
        for _ in 0..20 {
            state.update(false, config.epsilon, DEFAULT_CHANCE_PROB);
        }
        assert!(state.is_rejected());

        // Force non-rejection by adding many inliers
        let mut state_good = SPRTState::new(&config, DEFAULT_BETA);
        for _ in 0..20 {
            state_good.update(true, config.epsilon, DEFAULT_CHANCE_PROB);
        }
        assert!(!state_good.is_rejected());
    }
}
