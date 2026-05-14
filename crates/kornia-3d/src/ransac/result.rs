//! RANSAC driver output.

/// Result of a RANSAC run.
///
/// Holds owned data only — no borrows, no trait objects — so the type can be
/// returned from `kornia-py` bindings as a `#[pyclass(frozen)]` without
/// lifetime gymnastics.
#[derive(Debug, Clone)]
pub struct RansacResult<M> {
    /// Best-scoring model, or `None` if no hypothesis produced any inliers.
    pub model: Option<M>,
    /// Inlier mask aligned to the input sample slice. Empty iff `model` is
    /// `None`.
    pub inliers: Vec<bool>,
    /// Number of hypotheses actually evaluated (may be < `max_iters` due to
    /// early termination from the confidence bound).
    pub num_iters: u32,
    /// Best consensus score observed. Higher is better. Units depend on the
    /// active consensus strategy (inlier count for threshold, σ-weighted sum
    /// for MAGSAC++).
    pub score: f64,
}

impl<M> RansacResult<M> {
    /// Number of inliers in [`Self::inliers`]. O(n) over the mask.
    pub fn inlier_count(&self) -> usize {
        self.inliers.iter().filter(|&&b| b).count()
    }
}
