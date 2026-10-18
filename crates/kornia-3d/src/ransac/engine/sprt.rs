//! Wald's SPRT as a [`Verifier`] for raw minimal candidates.

use super::{PointResidual, Verifier, VerifyContext, Workspace};
use crate::ransac::sprt::{SPRTConfig, SPRTState, DEFAULT_BETA, DEFAULT_CHANCE_PROB};
use rand::{seq::SliceRandom, Rng, RngExt};

/// Draws between reshuffles of the base evaluation order.
const RESHUFFLE_PERIOD: usize = 16;

/// Sequential probability ratio test over residuals computed on demand.
///
/// Points are visited in a random order: one base permutation, reshuffled
/// every [`RESHUFFLE_PERIOD`] draws and rotated by a random offset for every
/// candidate. That is O(1) per candidate instead of an O(n) shuffle (at
/// n = 600 the shuffle alone costs ~2 µs, more than the point evaluations it
/// saves on a typical rejected hypothesis), and rotation keeps the per-visit
/// marginal distribution the drift analysis relies on: each point occupies
/// each visit slot with probability 1/n.
///
/// The reshuffle runs once per *candidate* on qualifying draws, so a draw
/// with several roots reshuffles several times. Both legacy SPRT loops did
/// this, and their seeded outputs depend on it; changing it to once per draw
/// changes the RNG stream.
///
/// A passing candidate has had every residual written to
/// [`Workspace::residuals`], so exact scoring can reuse them.
pub(crate) struct SprtVerifier {
    config: SPRTConfig,
    threshold: f64,
    sample_size: usize,
    order: Vec<usize>,
    reshuffle_phase: usize,
}

impl SprtVerifier {
    /// Verify candidates against `config` with a strict `residual < threshold`
    /// inlier test.
    ///
    /// The point order is reshuffled on draws where
    /// `draws % RESHUFFLE_PERIOD == reshuffle_phase`: the generic driver
    /// counted draws from zero (`reshuffle_phase = 1`), the PnP loop from one
    /// (`reshuffle_phase = 0`).
    ///
    /// `config` must be valid ([`SPRTConfig::is_valid`]); callers fall back
    /// to unverified scoring otherwise. The run's [`Workspace::residuals`]
    /// must hold one entry per observation.
    pub(crate) fn new(
        config: SPRTConfig,
        threshold: f64,
        population: usize,
        sample_size: usize,
        reshuffle_phase: usize,
    ) -> Self {
        Self {
            config,
            threshold,
            sample_size,
            order: (0..population).collect(),
            reshuffle_phase,
        }
    }
}

impl<P: PointResidual, R: Rng> Verifier<P, R> for SprtVerifier {
    fn verify(
        &mut self,
        problem: &P,
        model: &P::Model,
        cx: &VerifyContext,
        ws: &mut Workspace,
        rng: &mut R,
    ) -> bool {
        let n = self.order.len();
        debug_assert_eq!(
            ws.residuals.len(),
            n,
            "SPRT needs one residual slot per point"
        );
        // The LLR steps need the inlier ratio `eps`. Once a model with at
        // least a minimal sample's support exists, use its *observed* ratio,
        // never clamped up to the prior: a prior above the true ratio makes
        // even a good hypothesis look bad. Before that, cap the caller's
        // prior at 0.3: an optimistic prior gives good hypotheses a positive
        // LLR drift, so SPRT would reject the very models it should find.
        let reliable = cx.best_support >= self.sample_size;
        let eps = if reliable {
            (cx.best_support as f64 / n as f64).clamp(0.05, 0.95)
        } else {
            self.config.epsilon.min(0.3)
        };
        let eps = eps.clamp(1e-10, 1.0 - 1e-10);
        let delta = DEFAULT_CHANCE_PROB.clamp(1e-10, 1.0 - 1e-10);
        let inlier_step = (delta / eps).ln();
        let outlier_step = ((1.0 - delta) / (1.0 - eps)).ln();
        // Before a reliable incumbent exists the prior is unverified: every
        // hypothesis gets a minimum look at the data before rejection, so a
        // mismatched prior cannot starve the run of its first good model.
        let grace = if reliable {
            0
        } else {
            (n / 4).max(self.sample_size * 4)
        };

        if cx.draws % RESHUFFLE_PERIOD == self.reshuffle_phase {
            self.order.shuffle(rng);
        }
        let start = rng.random_range(0..n);

        // Bounding the residuals by the problem's size lets one index check
        // cover both the residual and the observation lookup.
        let residuals = &mut ws.residuals[..problem.num_observations()];
        let mut state = SPRTState::new(&self.config, DEFAULT_BETA);
        for k in 0..n {
            let index = self.order[(start + k) % n];
            // Lazy residuals: a rejected hypothesis never touches the rest.
            let residual = problem.residual(model, index);
            residuals[index] = residual;
            state.update_with_steps(residual < self.threshold, inlier_step, outlier_step);
            if state.is_rejected() && state.num_tested >= grace {
                // Log a copy: borrowing `state` for the formatter would pin
                // it to memory for the whole walk.
                let tested = state.num_tested;
                log::debug!(
                    "SPRT rejected hypothesis at iteration {} after {tested} points",
                    cx.draws
                );
                return false;
            }
        }
        ws.residuals_complete = true;
        true
    }
}
