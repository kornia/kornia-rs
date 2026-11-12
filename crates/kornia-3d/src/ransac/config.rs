//! RANSAC driver configuration.

/// Which consensus strategy the driver should apply.
///
/// The driver instantiates the corresponding [`super::Consensus`] impl
/// internally — callers picking a kind here don't need to construct the
/// scorer themselves. Hand-rolled scorers can still be passed directly to
/// the lower-level driver entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsensusKind {
    /// Classic hard-threshold RANSAC. Uses [`super::ThresholdConsensus`]
    /// with [`RansacConfig::inlier_threshold`].
    Threshold,
    /// MAGSAC++ σ-consensus. Threshold-free; reserved for a follow-up impl.
    Magsac,
}

/// Top-level RANSAC loop configuration.
///
/// Defaults are tuned for two-view geometry with normalised pixel residuals;
/// callers solving very different problems (e.g. PnP in metric units) should
/// override `inlier_threshold` and likely `max_iters`.
#[derive(Debug, Clone)]
pub struct RansacConfig {
    /// Hard cap on hypotheses drawn before the driver returns the best so far.
    pub max_iters: u32,
    /// Target probability that at least one drawn sample is all-inlier.
    /// Used to adapt `max_iters` downwards once a high-inlier hypothesis lands.
    /// A value outside (0, 1), or NaN, disables adaptive stopping, so the
    /// driver draws all `max_iters` samples.
    pub confidence: f64,
    /// Inlier residual cutoff. Interpreted by the active [`super::Consensus`].
    pub inlier_threshold: f64,
    /// Run a local-optimisation refit every `lo_every` *accepted* hypotheses
    /// (LO-RANSAC). `0` disables LO entirely.
    pub lo_every: u32,
    /// Evaluate hypotheses in parallel chunks via `rayon`.
    pub parallel: bool,
    /// Consensus strategy applied to per-hypothesis residuals.
    pub consensus: ConsensusKind,
    /// Enable Wald's SPRT to short-circuit bad hypotheses early.
    /// When `Some`, the driver verifies a hypothesis by sequentially evaluating
    /// points against the SPRT threshold and stops as soon as the model is
    /// rejected, instead of scoring the full residual vector in one batch.
    pub sprt: Option<super::SPRTConfig>,
}

/// Whether `confidence` is a usable RANSAC success probability.
///
/// Adaptive stopping needs `ln(1 - confidence)` to be finite and negative,
/// which holds exactly for finite values strictly between zero and one. The
/// PnP and two-view entry points reject any other value, and so do the Python
/// bindings; the generic driver instead treats it as "never stop early" (see
/// [`RansacConfig::confidence`]).
///
/// # Arguments
///
/// * `confidence` - Target probability that at least one drawn sample is
///   all-inlier.
///
/// # Returns
///
/// `true` if `confidence` is finite and `0 < confidence < 1`.
///
/// # Errors
///
/// This function is infallible.
///
/// # Example
///
/// ```rust
/// use kornia_3d::ransac::is_valid_confidence;
///
/// assert!(is_valid_confidence(0.999));
/// assert!(!is_valid_confidence(1.0));
/// assert!(!is_valid_confidence(f64::NAN));
/// ```
#[inline]
pub fn is_valid_confidence(confidence: f64) -> bool {
    confidence.is_finite() && confidence > 0.0 && confidence < 1.0
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            max_iters: 1000,
            confidence: 0.999,
            inlier_threshold: 1.0,
            lo_every: 0,
            parallel: false,
            consensus: ConsensusKind::Threshold,
            sprt: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_finite_confidences_strictly_inside_the_unit_interval_are_valid() {
        for confidence in [f64::MIN_POSITIVE, 0.5, 0.99, 1.0 - f64::EPSILON] {
            assert!(is_valid_confidence(confidence), "{confidence}");
        }
        for confidence in [0.0, -0.0, 1.0, -0.1, 1.5, f64::NAN, f64::INFINITY] {
            assert!(!is_valid_confidence(confidence), "{confidence}");
        }
    }
}
