//! When the sampling loop stops.
//!
//! Both policies here use the exact adaptive bound for sampling without
//! replacement ([`adaptive_max_iters`]). They differ only in the legacy
//! details of how the bound is updated, which phase A of #1171 preserves.

use super::Termination;

/// Recompute the adaptive iteration cap for distinct sampling without
/// replacement. Returns the minimum of `current` and the analytic estimate.
/// Never increases the cap and never returns zero.
#[inline]
pub(crate) fn adaptive_max_iters(
    inlier_count: usize,
    population: usize,
    sample_size: usize,
    confidence: f64,
    current: usize,
) -> usize {
    if inlier_count < sample_size
        || sample_size > population
        || confidence <= 0.0
        || confidence >= 1.0
    {
        return current;
    }
    if inlier_count == population {
        return 1;
    }
    let mut log_all_inlier = 0.0;
    for j in 0..sample_size {
        log_all_inlier += ((inlier_count - j) as f64).ln() - ((population - j) as f64).ln();
    }
    if !log_all_inlier.is_finite() || log_all_inlier >= 0.0 {
        return current;
    }
    let log_fail_per_draw = (-log_all_inlier.exp()).ln_1p();
    if !log_fail_per_draw.is_finite() || log_fail_per_draw >= 0.0 {
        return current;
    }
    let raw = (-confidence).ln_1p() / log_fail_per_draw;
    if !raw.is_finite() || raw <= 0.0 {
        return current;
    }
    let ceiled = raw.ceil() as usize;
    ceiled.min(current).max(1)
}

/// Adaptive cap of the generic driver (`ransac::run`).
///
/// Every accepted candidate with non-zero support tightens the cap. The cap
/// may drop below the number of draws already made, which ends the loop
/// after the current draw.
#[derive(Debug, Clone)]
pub(crate) struct GenericCap {
    max_draws: u32,
    population: usize,
    sample_size: usize,
    confidence: f64,
}

impl GenericCap {
    /// A cap of `max_draws` draws, tightened at `confidence`.
    pub(crate) fn new(
        max_draws: u32,
        population: usize,
        sample_size: usize,
        confidence: f64,
    ) -> Self {
        Self {
            max_draws,
            population,
            sample_size,
            confidence,
        }
    }
}

impl Termination for GenericCap {
    #[inline]
    fn done(&self, draws: usize) -> bool {
        draws >= self.max_draws as usize
    }

    #[inline]
    fn on_new_best(&mut self, support: usize, _draws: usize) {
        if support > 0 {
            let tightened = adaptive_max_iters(
                support,
                self.population,
                self.sample_size,
                self.confidence,
                self.max_draws as usize,
            ) as u32;
            if tightened < self.max_draws {
                self.max_draws = tightened;
            }
        }
    }
}

/// Adaptive cap of the dedicated two-view loops (`pose::ransac_*`).
///
/// The cap never drops below the draws already made, full support stops
/// after the current draw, and the homography loop additionally stops after
/// `stagnation` draws without a new best.
///
/// Every new best tightens the current cap. The legacy essential and
/// homography loops tightened the configured maximum instead; that gives the
/// same cap in every state the loops reach, because a new best never has
/// less support than the last one, the adaptive bound does not grow with
/// support, and the cap never drops below the draws made.
#[derive(Debug, Clone)]
pub(crate) struct TwoViewCap {
    cap: usize,
    population: usize,
    sample_size: usize,
    confidence: f64,
    stagnation: Option<usize>,
    last_improvement: usize,
}

impl TwoViewCap {
    /// A cap of `max_draws` draws, tightened at `confidence`.
    pub(crate) fn new(
        max_draws: usize,
        population: usize,
        sample_size: usize,
        confidence: f64,
    ) -> Self {
        Self {
            cap: max_draws,
            population,
            sample_size,
            confidence,
            stagnation: None,
            last_improvement: 0,
        }
    }

    /// Also stop after `limit` consecutive draws without a new best.
    pub(crate) fn with_stagnation(mut self, limit: usize) -> Self {
        self.stagnation = Some(limit);
        self
    }
}

impl Termination for TwoViewCap {
    #[inline]
    fn done(&self, draws: usize) -> bool {
        draws >= self.cap
            || self
                .stagnation
                .is_some_and(|limit| draws - self.last_improvement >= limit)
    }

    #[inline]
    fn on_new_best(&mut self, support: usize, draws: usize) {
        self.last_improvement = draws;
        if support == self.population {
            // Finish this draw's remaining candidates, then stop.
            self.cap = draws;
            return;
        }
        self.cap = adaptive_max_iters(
            support,
            self.population,
            self.sample_size,
            self.confidence,
            self.cap,
        )
        .max(draws);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_cap_uses_without_replacement_probability() {
        // q = C(8, 8) / C(10, 8) = 1/45, so 0.999999 confidence needs
        // ceil(log(1-p) / log(1-q)) = 615 draws. The w^k approximation
        // incorrectly gives only 76 draws here.
        let eight = adaptive_max_iters(8, 10, 8, 0.999_999, 10_000);
        assert_eq!(eight, 615);

        // Smaller minimal samples have a higher all-inlier probability for
        // the same inlier population, hence require no more draws.
        let seven = adaptive_max_iters(8, 10, 7, 0.999_999, 10_000);
        assert!(seven < eight, "seven-point cap {seven} must beat {eight}");
    }

    #[test]
    fn test_adaptive_cap_preserves_edge_cases_and_existing_cap() {
        assert_eq!(adaptive_max_iters(7, 10, 8, 0.999, 500), 500);
        assert_eq!(adaptive_max_iters(10, 10, 8, 0.999, 500), 1);
        assert_eq!(adaptive_max_iters(8, 10, 8, 1.0, 500), 500);
        assert_eq!(adaptive_max_iters(8, 10, 8, 0.999_999, 100), 100);
    }

    #[test]
    fn generic_cap_tightens_below_completed_draws_and_ignores_zero_support() {
        let mut cap = GenericCap::new(1000, 10, 8, 0.999);
        cap.on_new_best(0, 5);
        assert!(!cap.done(999));
        // Full support gives the analytic bound 1, below the draws made.
        cap.on_new_best(10, 5);
        assert!(cap.done(5));
        assert!(cap.done(1));
    }

    #[test]
    fn two_view_cap_never_drops_below_completed_draws() {
        let mut cap = TwoViewCap::new(1000, 10, 8, 0.999_999);
        cap.on_new_best(8, 700);
        // The analytic bound (615) is below the draws made (700).
        assert!(!cap.done(699));
        assert!(cap.done(700));
    }

    #[test]
    fn two_view_cap_full_support_stops_after_the_current_draw() {
        let mut cap = TwoViewCap::new(1000, 50, 4, 0.99);
        cap.on_new_best(50, 3);
        assert!(!cap.done(2));
        assert!(cap.done(3));
    }

    #[test]
    fn two_view_cap_tightens_the_current_cap() {
        let mut cap = TwoViewCap::new(1000, 10, 8, 0.999_999);
        // min(1000, 615) = 615 at 8 inliers.
        cap.on_new_best(8, 1);
        assert!(!cap.done(614) && cap.done(615));
        // An equal-support best, e.g. a lower residual, keeps it.
        cap.on_new_best(8, 2);
        assert!(!cap.done(614) && cap.done(615));
        // More support tightens it further: q = C(9, 8) / C(10, 8) = 1/5.
        cap.on_new_best(9, 3);
        assert!(!cap.done(61) && cap.done(62));
    }

    #[test]
    fn two_view_cap_stagnation_counts_draws_since_the_last_best() {
        let mut cap = TwoViewCap::new(10_000, 1000, 4, 0.99).with_stagnation(200);
        assert!(!cap.done(199));
        assert!(cap.done(200));
        cap.on_new_best(10, 150);
        assert!(!cap.done(349));
        assert!(cap.done(350));
    }
}
