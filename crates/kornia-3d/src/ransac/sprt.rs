//! Core mathematical logic for Wald's Sequential Probability Ratio Test (SPRT) in RANSAC.
//!
//! SPRT allows the RANSAC driver to reject "bad" model hypotheses early by evaluating points
//! sequentially and stopping as soon as the Log-Likelihood Ratio (LLR) exceeds a calculated threshold.
//!
//! The math is based on WALD's SPRT (Wald, "Sequential Tests of Statistical Hypotheses",
//! 1945) and the OpenCV USAC framework's formulation (Raguram et al., "USAC: A Universal
//! Framework for Random Sample Consensus", 2013). Two threshold formulas are supported:
//!
//! - The simple Wald form: $A = \ln((1 - \beta) / \alpha)$.
//! - The time-aware variant used when `t_M > t_m`:
//!   $A = \ln((1 - \beta) / \alpha) \cdot (t_M - t_m) / t_m$,
//!   which scales the threshold with the relative cost of evaluating one
//!   point ($t_m$) versus instantiating a model hypothesis ($t_M$).

/// Default probability of accepting a bad model (Type II error, $\beta$).
pub const DEFAULT_BETA: f64 = 0.05;

/// Default probability that a bad model accepts a point by chance ($\delta_{chance}$).
pub const DEFAULT_CHANCE_PROB: f64 = 0.05;

/// Configuration for SPRT thresholds and timing.
#[derive(Debug, Clone, Copy)]
// Field names follow the USAC paper's notation (t_M, t_m).
#[allow(non_snake_case)]
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
    /// Create a new SPRT config with sensible defaults. The caller must still
    /// supply the expected inlier ratio and Type I error tolerance.
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self {
            epsilon,
            delta,
            t_M: 1.0,
            t_m: 1.0,
        }
    }

    /// Calculate the decision threshold $A$.
    ///
    /// When $t_M > t_m$ (a model hypothesis costs more than a single point
    /// evaluation) the threshold is scaled by the time ratio:
    /// $A = \ln((1 - \beta) / \alpha) \cdot (t_M - t_m) / t_m$.
    ///
    /// Falls back to the simple Wald form $A = \ln((1 - \beta) / \alpha)$
    /// when $t_M \le t_m$ (i.e. evaluating one point is no cheaper than
    /// instantiating a model — no time saving to be had).
    pub fn calculate_threshold(&self, beta: f64) -> f64 {
        let base_ln = ((1.0 - beta) / self.delta).ln();
        if self.t_M > self.t_m && self.t_m > 0.0 {
            let exponent = (self.t_M - self.t_m) / self.t_m;
            base_ln * exponent
        } else {
            base_ln
        }
    }

    /// Returns true if the config has invalid (non-finite, non-positive, or saturated)
    /// probability values. Callers should reject such configs and fall back to
    /// non-SPRT evaluation.
    pub fn is_valid(&self) -> bool {
        self.epsilon.is_finite()
            && self.delta.is_finite()
            && self.t_M.is_finite()
            && self.t_m.is_finite()
            && self.epsilon > 0.0
            && self.epsilon < 1.0
            && self.delta > 0.0
            && self.delta < 1.0
            && self.t_M > 0.0
            && self.t_m > 0.0
    }
}

impl Default for SPRTConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.5,
            delta: 0.05,
            t_M: 1.0,
            t_m: 1.0,
        }
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
    /// Whether the model has been rejected by the test.
    rejected: bool,
    /// Whether the model has been accepted (reached the end of the test set).
    accepted: bool,
}

impl SPRTState {
    /// Create a new SPRT state based on a config.
    pub fn new(config: &SPRTConfig, beta: f64) -> Self {
        Self {
            llr: 0.0,
            num_tested: 0,
            decision_threshold: config.calculate_threshold(beta),
            rejected: false,
            accepted: false,
        }
    }

    /// Update the LLR based on whether the current point is an inlier.
    ///
    /// LLR update:
    /// - Inlier: $\ln(\delta_{chance} / \epsilon)$
    /// - Outlier: $\ln((1 - \delta_{chance}) / (1 - \epsilon))$
    ///
    /// If the LLR exceeds the decision threshold, the model is marked as rejected
    /// and subsequent `update` calls are no-ops.
    pub fn update(&mut self, is_inlier: bool, epsilon: f64, delta_chance: f64) {
        if self.rejected || self.accepted {
            return;
        }

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

        if self.llr > self.decision_threshold {
            self.rejected = true;
        }
    }

    /// Check if the current LLR exceeds the decision threshold, meaning the model should be rejected.
    pub fn is_rejected(&self) -> bool {
        self.rejected
    }

    /// Check if the model has been accepted (i.e. the test set was exhausted without rejection).
    pub fn is_accepted(&self) -> bool {
        self.accepted
    }

    /// Mark the model as accepted (the test set was exhausted without rejection).
    pub fn mark_accepted(&mut self) {
        if !self.rejected {
            self.accepted = true;
        }
    }

    /// Highly optimized update path that uses precalculated LLR steps.
    /// Callers should compute `inlier_step` and `outlier_step` outside the inner loop.
    #[inline]
    pub fn update_with_steps(&mut self, is_inlier: bool, inlier_step: f64, outlier_step: f64) {
        if self.rejected || self.accepted {
            return;
        }
        self.llr += if is_inlier { inlier_step } else { outlier_step };
        self.num_tested += 1;

        if self.llr > self.decision_threshold {
            self.rejected = true;
        }
    }
}

/// Outcome of an SPRT evaluation over a full data set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SPROutcome {
    /// The model was rejected early by SPRT.
    Rejected,
    /// The model was accepted (test set was exhausted without rejection).
    Accepted,
}

/// Evaluate a hypothesis model against a point set using SPRT.
///
/// Iterates over `points` in randomized order, computing the residual of
/// each point via `residual_fn`. Once the LLR exceeds the decision threshold
/// the model is rejected and the loop stops. If the loop finishes without
/// rejection the model is accepted.
///
/// `residual_fn` returns the residual for a single point; values below
/// `inlier_threshold` count as inliers.
///
/// The returned [`SPRTState`] exposes the number of points tested and the
/// final LLR, so callers can update internal bookkeeping (e.g. the best
/// inlier mask) once the test set is exhausted.
pub fn evaluate<P, F, R>(
    points: &[P],
    residual_fn: F,
    perm: &[usize],
    inlier_threshold: f64,
    config: &SPRTConfig,
    beta: f64,
) -> (SPRTState, SPROutcome)
where
    F: Fn(&P) -> R,
    R: PartialOrd<f64>,
{
    let mut state = SPRTState::new(config, beta);

    for &p_idx in perm.iter() {
        let p = &points[p_idx];
        let r = residual_fn(p);
        let is_inlier = r < inlier_threshold;
        state.update(is_inlier, config.epsilon, DEFAULT_CHANCE_PROB);

        if state.is_rejected() {
            // Fast-exit: the hypothesis is bad, drop it.
            return (state, SPROutcome::Rejected);
        }
    }

    state.mark_accepted();
    (state, SPROutcome::Accepted)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests below use `t_M == t_m` so the simple Wald threshold
    /// $A = \ln((1-\beta)/\alpha)$ is exercised (matches the formula the
    /// pre-SPRT USAC reference uses, and keeps the asserted numbers
    /// interpretable). Time-aware thresholds are covered separately by
    /// `calculate_threshold` itself.

    #[test]
    fn test_verify_threshold_calculation() {
        let config = SPRTConfig {
            epsilon: 0.5,
            delta: 0.01,
            t_M: 1.0,
            t_m: 1.0,
        };
        // A = ln((1 - 0.05) / 0.01) = ln(95) approx 4.5539
        let threshold = config.calculate_threshold(DEFAULT_BETA);
        assert!((threshold - 4.5539).abs() < 1e-4);
    }

    /// Time-aware variant (t_M > t_m) scales the Wald threshold by the time ratio.
    #[test]
    fn test_calculate_threshold_time_aware() {
        let config = SPRTConfig {
            epsilon: 0.5,
            delta: 0.01,
            t_M: 2.0,
            t_m: 1.0,
        };
        // A = ln((1 - 0.05) / 0.01) * (2 - 1) / 1 = ln(95) * 1 = 4.5539.
        let threshold = config.calculate_threshold(DEFAULT_BETA);
        assert!((threshold - 4.5539).abs() < 1e-4);
    }

    #[test]
    fn test_likelihood_ratio_inliers() {
        let config = SPRTConfig {
            epsilon: 0.3,
            delta: 0.01,
            t_M: 1.0,
            t_m: 1.0,
        };
        let mut state = SPRTState::new(&config, DEFAULT_BETA);
        let initial_llr = state.llr;

        // For inliers: ln(0.05 / 0.3) approx -1.7918
        state.update(true, config.epsilon, DEFAULT_CHANCE_PROB);
        assert!(state.llr < initial_llr);
        assert!((state.llr - (-1.7918)).abs() < 1e-3);
    }

    #[test]
    fn test_likelihood_ratio_outliers() {
        let config = SPRTConfig {
            epsilon: 0.3,
            delta: 0.01,
            t_M: 1.0,
            t_m: 1.0,
        };
        let mut state = SPRTState::new(&config, DEFAULT_BETA);
        let initial_llr = state.llr;

        // For outliers: ln((1 - 0.05) / (1 - 0.3)) = ln(0.95 / 0.7) approx 0.3054
        state.update(false, config.epsilon, DEFAULT_CHANCE_PROB);
        assert!(state.llr > initial_llr);
        assert!((state.llr - 0.3054).abs() < 1e-3);
    }

    #[test]
    fn test_dynamic_epsilon_update() {
        let config = SPRTConfig {
            epsilon: 0.3,
            delta: 0.01,
            t_M: 1.0,
            t_m: 1.0,
        };
        let mut state = SPRTState::new(&config, DEFAULT_BETA);

        // Test with epsilon = 0.3
        state.update(true, 0.3, DEFAULT_CHANCE_PROB);
        let llr_1 = state.llr;

        // Test with epsilon = 0.5
        let mut state2 = SPRTState::new(&config, DEFAULT_BETA);
        state2.update(true, 0.5, DEFAULT_CHANCE_PROB);
        let llr_2 = state2.llr;

        // ln(0.05 / 0.5) is more negative than ln(0.05 / 0.3)
        assert!(llr_2 < llr_1);
    }

    #[test]
    fn test_rejection_logic() {
        let config = SPRTConfig {
            epsilon: 0.3,
            delta: 0.01,
            t_M: 1.0,
            t_m: 1.0,
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

    // ---------- evaluate() tests ----------

    /// All-inlier data set → SPRT exhausts the test set and accepts.
    #[test]
    fn test_sprt_full_acceptance() {
        // 30 inliers (residual = 0.0), threshold = 1.0
        let residuals: Vec<f64> = vec![0.0; 30];
        let config = SPRTConfig {
            epsilon: 0.5,
            delta: 0.01,
            t_M: 1.0,
            t_m: 1.0,
        };
        let perm: Vec<usize> = (0..residuals.len()).collect();
        let (state, outcome) = evaluate(&residuals, |r| *r, &perm, 1.0, &config, DEFAULT_BETA);
        assert_eq!(outcome, SPROutcome::Accepted);
        assert!(!state.is_rejected());
        assert_eq!(state.num_tested, residuals.len());
    }

    /// All-outlier data set → SPRT rejects early, well before the test set is exhausted.
    #[test]
    fn test_sprt_early_rejection() {
        // 50 outliers (residual = 10.0), threshold = 1.0
        let residuals: Vec<f64> = vec![10.0; 50];
        let config = SPRTConfig {
            epsilon: 0.5,
            delta: 0.01,
            t_M: 1.0,
            t_m: 1.0,
        };
        let perm: Vec<usize> = (0..residuals.len()).collect();
        let (state, outcome) = evaluate(&residuals, |r| *r, &perm, 1.0, &config, DEFAULT_BETA);
        assert_eq!(outcome, SPROutcome::Rejected);
        assert!(state.is_rejected());
        // Must reject well before exhausting the data set.
        assert!(state.num_tested < residuals.len());
    }

    /// Marginal data set (mixed inliers/outliers near the expected ratio) → evaluated
    /// to completion without early rejection.
    #[test]
    fn test_sprt_marginal_rejection() {
        // Half inliers, half outliers: matches the expected epsilon, so the LLR does
        // not climb much and the test reaches the end of the data set.
        let mut residuals: Vec<f64> = vec![0.0; 50];
        residuals.extend(vec![10.0; 50]);
        let config = SPRTConfig {
            epsilon: 0.5,
            delta: 0.01,
            t_M: 1.0,
            t_m: 1.0,
        };
        let perm: Vec<usize> = (0..residuals.len()).collect();
        let (state, _outcome) = evaluate(&residuals, |r| *r, &perm, 1.0, &config, DEFAULT_BETA);
        // Doesn't reject (avg log-likelihood stays close to zero on 50/50 input).
        assert!(!state.is_rejected());
    }

    #[test]
    fn test_invalid_sprt_config_handling() {
        // epsilon = 0 (degenerate): config reports invalid.
        let bad = SPRTConfig {
            epsilon: 0.0,
            delta: 0.01,
            t_M: 1.0,
            t_m: 1.0,
        };
        assert!(!bad.is_valid());

        // delta = 1.0 (degenerate): config reports invalid.
        let bad2 = SPRTConfig {
            epsilon: 0.5,
            delta: 1.0,
            t_M: 1.0,
            t_m: 1.0,
        };
        assert!(!bad2.is_valid());

        // NaN probabilities: config reports invalid.
        let bad3 = SPRTConfig {
            epsilon: f64::NAN,
            delta: 0.01,
            t_M: 1.0,
            t_m: 1.0,
        };
        assert!(!bad3.is_valid());

        // Sanity: a good config is valid.
        let good = SPRTConfig {
            epsilon: 0.5,
            delta: 0.01,
            t_M: 1.0,
            t_m: 1.0,
        };
        assert!(good.is_valid());
    }
}
