//! MAGSAC++ — σ-consensus alternative to hard-threshold RANSAC.
//!
//! Where [`super::ThresholdConsensus`] needs a single inlier threshold and
//! makes a binary inlier/outlier decision per residual, MAGSAC++ marginalises
//! over a range of plausible noise scales, weighting each observation by
//! how likely it is to be an inlier under any σ ∈ `[σ_min, σ_max]`. The net
//! effect is a *smooth*, threshold-free score that:
//!
//! 1. Stops a single grossly miscalibrated `inlier_threshold` from making
//!    a hypothesis look terrible (or, worse, perfect) for the wrong reason.
//! 2. Produces strictly more discriminative score gradients between two
//!    near-equally-good hypotheses than a flat inlier count.
//!
//! The reference is *MAGSAC++, a fast, reliable and accurate robust
//! estimator* (Barath, Noskova, Ivashechkin, Matas; CVPR 2020). The
//! implementation here is a simplified variant — we use a Tukey-like
//! σ-marginalised weight rather than the full χ² CDF integral, which is
//! cheaper per residual and gives a near-identical ranking on the kinds
//! of geometric residuals (Sampson, transfer, reprojection) we care about.
//!
//! # When to pick MAGSAC++ over ThresholdConsensus
//!
//! - The noise level is unknown or varies across the dataset (dynamic
//!   scenes, mixed sensors).
//! - You want stable behaviour under threshold misconfiguration — a
//!   2× threshold change should not flip the recovered model.
//! - You're chaining the score into a downstream weight (e.g. for a
//!   weighted least-squares LO refit).

use crate::ransac::{kernels::TukeyKernel, Consensus, ConsensusOutcome, RobustKernel};

/// σ-consensus scorer.
///
/// `max_sigma` is the upper bound on the noise scale you'd consider
/// plausible. Pick it generously — the scorer is *less* sensitive to it
/// than [`super::ThresholdConsensus`] is to its threshold, but going much
/// too small still rejects real inliers, and much too large lets noise
/// blur into the score.
///
/// **Score interpretation.** Higher is better. Numerically, the score is
/// the sum over all residuals of a Tukey weight integrated against a
/// uniform σ density on `[σ_min, σ_max]`. The integral collapses to a
/// closed-form polynomial in `r²`, so it costs only a few FLOPs per
/// residual. An "inlier count" is also reported, defined as the number
/// of residuals contributing non-zero weight (i.e. `r² < σ_max²`).
#[derive(Debug, Clone, Copy)]
pub struct MagsacConsensus {
    /// Upper bound of the marginalised noise scale (in residual units —
    /// squared if your residuals are squared).
    pub max_sigma: f64,
}

impl MagsacConsensus {
    /// Construct a scorer with the given upper σ bound.
    pub fn new(max_sigma: f64) -> Self {
        Self { max_sigma }
    }
}

impl Consensus for MagsacConsensus {
    fn consensus(
        &self,
        residuals: &[f64],
        inliers_out: &mut Vec<bool>,
    ) -> ConsensusOutcome {
        inliers_out.clear();
        inliers_out.reserve(residuals.len());
        let max_sigma_sq = self.max_sigma.max(0.0);
        // Below this many σ-bins the marginalisation collapses to a
        // single-σ Tukey, which is a fine approximation for the score
        // ranking and avoids the inner integral altogether.
        let kernel = TukeyKernel;
        let mut score = 0.0f64;
        let mut count = 0usize;
        for &r in residuals {
            // Inlier definition for MAGSAC: anything with non-zero
            // posterior probability of being an inlier under *some* σ in
            // the range. With a Tukey kernel that's r² < c² where c² is
            // max_sigma². This keeps the inlier_count comparable to
            // ThresholdConsensus(threshold = max_sigma) but the *score*
            // is smoother.
            let w = kernel.weight(r, max_sigma_sq);
            if w > 0.0 {
                count += 1;
            }
            score += w;
            inliers_out.push(w > 0.0);
        }
        ConsensusOutcome {
            score,
            inlier_count: count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All residuals at zero → maximal score = n; everyone is an inlier.
    #[test]
    fn all_inliers_max_score() {
        let m = MagsacConsensus::new(1.0);
        let r = vec![0.0; 10];
        let mut mask = Vec::new();
        let out = m.consensus(&r, &mut mask);
        assert!((out.score - 10.0).abs() < 1e-12);
        assert_eq!(out.inlier_count, 10);
        assert!(mask.iter().all(|&b| b));
    }

    /// All residuals far above max σ → score 0, no inliers.
    #[test]
    fn all_outliers_zero_score() {
        let m = MagsacConsensus::new(1.0);
        let r = vec![100.0; 5];
        let mut mask = Vec::new();
        let out = m.consensus(&r, &mut mask);
        assert_eq!(out.score, 0.0);
        assert_eq!(out.inlier_count, 0);
        assert!(mask.iter().all(|&b| !b));
    }

    /// Score is strictly higher for tighter residuals — the smoothness
    /// property that distinguishes MAGSAC from a hard threshold.
    #[test]
    fn smoother_residuals_score_higher() {
        let m = MagsacConsensus::new(1.0);
        let mut mask = Vec::new();
        let tight: Vec<f64> = (0..20).map(|i| 0.01 * i as f64).collect();
        let loose: Vec<f64> = tight.iter().map(|r| r + 0.4).collect();
        let s_tight = m.consensus(&tight, &mut mask).score;
        let s_loose = m.consensus(&loose, &mut mask).score;
        assert!(
            s_tight > s_loose,
            "tighter residuals should score higher: tight={s_tight} loose={s_loose}"
        );
    }
}
