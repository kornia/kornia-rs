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
        }
    }
}
