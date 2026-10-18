//! Candidate scores and the incumbent hypothesis.
//!
//! A [`Score`] separates the *objective*, which orders candidates, from the
//! *support*, the number of inliers at the run's threshold. Termination,
//! output masks and SPRT calibration read the support; only the incumbent's
//! replacement rule reads the objective.

/// Tie rule shared by pruning and acceptance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ties {
    /// A candidate must beat the incumbent strictly.
    Strict,
    /// A candidate that ties the incumbent replaces it.
    AcceptTies,
}

/// A totally ordered ranking objective.
pub(crate) trait Objective: Copy {
    /// Objective of the empty incumbent, before any candidate was accepted.
    fn worst() -> Self;

    /// Whether `self` replaces `incumbent` under `ties`.
    fn beats(&self, incumbent: &Self, ties: Ties) -> bool;
}

/// Higher-is-better scalar score from a [`crate::ransac::Consensus`].
///
/// NaN compares false under both tie rules, so a NaN score never replaces
/// the incumbent, as in the generic driver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MaxScore(pub f64);

impl Objective for MaxScore {
    #[inline]
    fn worst() -> Self {
        Self(f64::NEG_INFINITY)
    }

    #[inline]
    fn beats(&self, incumbent: &Self, ties: Ties) -> bool {
        match ties {
            Ties::Strict => self.0 > incumbent.0,
            Ties::AcceptTies => self.0 >= incumbent.0,
        }
    }
}

/// Inlier count first, then the lower sum of inlier residuals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CountThenResidual {
    /// Number of inliers.
    pub count: usize,
    /// Sum of the inliers' residuals.
    pub residual: f64,
}

impl Objective for CountThenResidual {
    #[inline]
    fn worst() -> Self {
        Self {
            count: 0,
            residual: f64::INFINITY,
        }
    }

    #[inline]
    fn beats(&self, incumbent: &Self, ties: Ties) -> bool {
        self.count > incumbent.count
            || (self.count == incumbent.count
                && match ties {
                    Ties::Strict => self.residual < incumbent.residual,
                    Ties::AcceptTies => self.residual <= incumbent.residual,
                })
    }
}

/// Plain inlier count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InlierCount(pub usize);

impl Objective for InlierCount {
    #[inline]
    fn worst() -> Self {
        Self(0)
    }

    #[inline]
    fn beats(&self, incumbent: &Self, ties: Ties) -> bool {
        match ties {
            Ties::Strict => self.0 > incumbent.0,
            Ties::AcceptTies => self.0 >= incumbent.0,
        }
    }
}

/// Score of one candidate model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Score<O> {
    /// Ranking objective.
    pub objective: O,
    /// Number of inliers at the run's threshold.
    pub support: usize,
}

/// The best hypothesis so far: model, score and inlier mask.
///
/// The engine replaces it only through [`Incumbent::consider`], and through
/// [`Incumbent::consider_keeping_mask`] for two legacy final fits. Either way
/// `score.support` is the support of the mask in [`Self::inliers`].
#[derive(Debug, Clone)]
pub(crate) struct Incumbent<M, O> {
    /// Best model, or `None` before the first accepted candidate.
    pub model: Option<M>,
    /// Score of [`Self::model`]; [`Objective::worst`] with zero support
    /// while there is no model. After [`Self::consider_keeping_mask`] the
    /// objective is the new model's and the support the kept mask's.
    pub score: Score<O>,
    /// Inlier mask of [`Self::model`], or of the previous model after
    /// [`Self::consider_keeping_mask`].
    pub inliers: Vec<bool>,
}

impl<M: Clone, O: Objective> Incumbent<M, O> {
    /// An empty incumbent that owns `inliers` as its mask buffer.
    pub(crate) fn new(inliers: Vec<bool>) -> Self {
        Self {
            model: None,
            score: Score {
                objective: O::worst(),
                support: 0,
            },
            inliers,
        }
    }

    /// The incumbent's score once a model exists, for pruning raw candidates.
    #[inline]
    pub(crate) fn bound(&self) -> Option<&Score<O>> {
        self.model.as_ref().map(|_| &self.score)
    }

    /// Replace the incumbent if `score` beats it under `ties`.
    ///
    /// On replacement the candidate mask is swapped into place and `mask`
    /// receives the previous incumbent's buffer for reuse.
    #[inline]
    pub(crate) fn consider(
        &mut self,
        model: &M,
        score: Score<O>,
        mask: &mut Vec<bool>,
        ties: Ties,
    ) -> bool {
        if !score.objective.beats(&self.score.objective, ties) {
            return false;
        }
        self.model = Some(model.clone());
        self.score = score;
        std::mem::swap(&mut self.inliers, mask);
        true
    }

    /// Replace the model and the objective if `objective` beats the
    /// incumbent's under `ties`, but keep the inlier mask and its support.
    ///
    /// Reproduces two legacy final fits that return the pre-refit mask: the
    /// homography DLT refit and the PnP refit. Fixing those stale masks is a
    /// separate, deliberate behaviour change (#1171 phase B). Keeping the
    /// support with the mask means every reader of `score.support` still
    /// sees the size of the returned consensus set.
    #[inline]
    pub(crate) fn consider_keeping_mask(&mut self, model: &M, objective: O, ties: Ties) -> bool {
        if !objective.beats(&self.score.objective, ties) {
            return false;
        }
        self.model = Some(model.clone());
        self.score.objective = objective;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctr(count: usize, residual: f64) -> CountThenResidual {
        CountThenResidual { count, residual }
    }

    #[test]
    fn max_score_tie_rules_and_nan() {
        assert!(MaxScore(2.0).beats(&MaxScore(1.0), Ties::Strict));
        assert!(!MaxScore(1.0).beats(&MaxScore(1.0), Ties::Strict));
        assert!(MaxScore(1.0).beats(&MaxScore(1.0), Ties::AcceptTies));
        assert!(MaxScore(0.0).beats(&MaxScore::worst(), Ties::Strict));
        assert!(!MaxScore(f64::NAN).beats(&MaxScore::worst(), Ties::Strict));
        assert!(!MaxScore(f64::NAN).beats(&MaxScore::worst(), Ties::AcceptTies));
    }

    #[test]
    fn count_then_residual_orders_lexicographically() {
        assert!(ctr(5, 9.0).beats(&ctr(4, 0.0), Ties::Strict));
        assert!(ctr(4, 1.0).beats(&ctr(4, 2.0), Ties::Strict));
        assert!(!ctr(4, 2.0).beats(&ctr(4, 2.0), Ties::Strict));
        assert!(ctr(4, 2.0).beats(&ctr(4, 2.0), Ties::AcceptTies));
        assert!(!ctr(3, 0.0).beats(&ctr(4, 9.0), Ties::AcceptTies));
        // An empty incumbent loses to any finite candidate, even one with no
        // inliers, exactly like the legacy two-view loops' (0, +inf) start.
        assert!(ctr(0, 0.0).beats(&CountThenResidual::worst(), Ties::Strict));
    }

    #[test]
    fn inlier_count_needs_support_to_beat_the_empty_incumbent() {
        assert!(!InlierCount(0).beats(&InlierCount::worst(), Ties::Strict));
        assert!(InlierCount(1).beats(&InlierCount::worst(), Ties::Strict));
        assert!(InlierCount(3).beats(&InlierCount(3), Ties::AcceptTies));
    }

    #[test]
    fn consider_swaps_masks_only_on_replacement() {
        let mut best = Incumbent::<u8, InlierCount>::new(vec![false; 3]);
        assert!(best.bound().is_none());

        let mut mask = vec![true, false, true];
        let two = Score {
            objective: InlierCount(2),
            support: 2,
        };
        assert!(best.consider(&7, two, &mut mask, Ties::Strict));
        assert_eq!(best.model, Some(7));
        assert_eq!(best.inliers, [true, false, true]);
        assert_eq!(mask, [false; 3]);
        assert_eq!(best.bound(), Some(&two));

        let mut tie = vec![false, true, true];
        assert!(!best.consider(&8, two, &mut tie, Ties::Strict));
        assert_eq!(best.model, Some(7));
        assert_eq!(tie, [false, true, true]);

        assert!(best.consider(&8, two, &mut tie, Ties::AcceptTies));
        assert_eq!(best.model, Some(8));
        assert_eq!(best.inliers, [false, true, true]);
    }

    #[test]
    fn consider_keeping_mask_leaves_the_mask_and_its_support_in_place() {
        let mut best = Incumbent::<u8, InlierCount>::new(vec![false; 3]);
        let mut mask = vec![true, false, true];
        let two = Score {
            objective: InlierCount(2),
            support: 2,
        };
        assert!(best.consider(&1, two, &mut mask, Ties::Strict));
        // A refit that would count three inliers replaces the model, but the
        // mask and its support of two stay.
        assert!(best.consider_keeping_mask(&2, InlierCount(3), Ties::AcceptTies));
        assert!(!best.consider_keeping_mask(&3, InlierCount(2), Ties::AcceptTies));
        assert_eq!(best.model, Some(2));
        assert_eq!(best.score.objective, InlierCount(3));
        assert_eq!(best.score.support, 2);
        assert_eq!(best.inliers, [true, false, true]);
    }
}
