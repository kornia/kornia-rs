//! Shared valid-span computation for the warp fast paths.
//!
//! `warp_affine`, `warp_affine_u8`, and `warp_perspective_u8` all derive, per
//! destination row, the half-open column range `[lo, hi)` whose inverse-mapped
//! source coordinates land inside the image, so everything outside can be
//! zero-filled with a memset and the inner loop can run unchecked.
//!
//! Each source-bounds condition reduces to a linear constraint `a*x + b >= 0`
//! or `a*x + b < 0` on the destination column `x`. The subtlety this module
//! exists to centralize: **which end of the solution interval is inclusive
//! flips with the sign of `a`**, and integer boundaries must be rounded
//! accordingly:
//!
//! ```text
//! a*x + b >= 0,  a > 0:  x >= -b/a  (inclusive)  → lo = ceil(-b/a)
//! a*x + b >= 0,  a < 0:  x <= -b/a  (inclusive)  → hi = floor(-b/a) + 1
//! a*x + b <  0,  a > 0:  x <  -b/a  (strict)     → hi = ceil(-b/a)
//! a*x + b <  0,  a < 0:  x >  -b/a  (strict)     → lo = floor(-b/a) + 1
//! ```
//!
//! Applying `ceil` uniformly (as three hand-maintained copies of this logic
//! once did) drops a valid column — or admits an invalid one — whenever the
//! boundary lands exactly on an integer, which is the common case for
//! axis-aligned transforms (flips, 90° rotations, integer translations).

/// Intersect the half-open integer range `[lo, hi)` with the solutions of one
/// linear constraint on `x`:
///
/// * `ge = true`:  `a*x + b >= 0`
/// * `ge = false`: `a*x + b <  0`
///
/// `eps` is the degenerate-slope threshold: when `|a| < eps` (or `a == 0`) the
/// constraint does not depend on `x`, so it is either vacuous or infeasible —
/// infeasible collapses the range to empty (`hi = lo`). Callers pick `eps` to
/// match their coordinate scale (see call sites).
pub(crate) fn constrain_span(a: f32, b: f32, ge: bool, eps: f32, lo: &mut i64, hi: &mut i64) {
    if a.abs() < eps || a == 0.0 {
        let feasible = if ge { b >= 0.0 } else { b < 0.0 };
        if !feasible {
            *hi = *lo;
        }
        return;
    }
    let k = -b / a;
    // Inclusive bound ⇔ (ge XOR a<0): see the table in the module doc.
    match (ge, a > 0.0) {
        // x >= k (inclusive lower)
        (true, true) => *lo = (*lo).max(k.ceil() as i64),
        // x <= k (inclusive upper)
        (true, false) => *hi = (*hi).min(k.floor() as i64 + 1),
        // x < k (strict upper)
        (false, true) => *hi = (*hi).min(k.ceil() as i64),
        // x > k (strict lower)
        (false, false) => *lo = (*lo).max(k.floor() as i64 + 1),
    }
}

/// Valid destination-column span for one row of an (inverse-mapped) affine
/// warp: the `x` for which `0 <= d*x + s0 < upper` holds on both axes,
/// clamped to `[0, dst_w)`. Returns an empty span as `(0, 0)`.
pub(crate) fn affine_valid_span(
    axes: [(f32, f32, f32); 2], // (d, s0, upper) per axis
    dst_w: usize,
    eps: f32,
) -> (usize, usize) {
    let (mut lo, mut hi) = (0i64, dst_w as i64);
    for (d, s0, upper) in axes {
        constrain_span(d, s0, true, eps, &mut lo, &mut hi); // d*x + s0 >= 0
        constrain_span(d, s0 - upper, false, eps, &mut lo, &mut hi); // d*x + s0 - upper < 0
        if lo >= hi {
            return (0, 0);
        }
    }
    let lo = lo.clamp(0, dst_w as i64);
    let hi = hi.clamp(0, dst_w as i64);
    if lo >= hi {
        (0, 0)
    } else {
        (lo as usize, hi as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(a: f32, b: f32, ge: bool, dst_w: i64) -> (i64, i64) {
        let (mut lo, mut hi) = (0i64, dst_w);
        constrain_span(a, b, ge, 1e-6, &mut lo, &mut hi);
        (lo, hi)
    }

    /// Inclusive bounds at exact integers must keep the boundary column.
    #[test]
    fn inclusive_integer_boundary_is_kept() {
        // x >= 3 (a>0, ge): lo = 3, x = 3 valid.
        assert_eq!(span(1.0, -3.0, true, 10), (3, 10));
        // x <= 3 (a<0, ge): valid x ∈ [0, 3], half-open hi = 4.
        assert_eq!(span(-1.0, 3.0, true, 10), (0, 4));
    }

    /// Strict bounds at exact integers must exclude the boundary column.
    #[test]
    fn strict_integer_boundary_is_excluded() {
        // x < 3 (a>0, lt): hi = 3.
        assert_eq!(span(1.0, -3.0, false, 10), (0, 3));
        // x > 3 (a<0, lt): lo = 4.
        assert_eq!(span(-1.0, 3.0, false, 10), (4, 10));
    }

    /// Non-integer boundaries: ceil and floor+1 agree, both ends behave.
    #[test]
    fn fractional_boundaries() {
        // x >= 2.5 → lo = 3;  x <= 2.5 → hi = 3;  x < 2.5 → hi = 3;  x > 2.5 → lo = 3.
        assert_eq!(span(1.0, -2.5, true, 10), (3, 10));
        assert_eq!(span(-1.0, 2.5, true, 10), (0, 3));
        assert_eq!(span(1.0, -2.5, false, 10), (0, 3));
        assert_eq!(span(-1.0, 2.5, false, 10), (3, 10));
    }

    /// Degenerate slope: constraint is x-independent — vacuous or infeasible.
    #[test]
    fn degenerate_slope() {
        assert_eq!(span(0.0, 1.0, true, 10), (0, 10)); // 1 >= 0 always
        assert_eq!(span(0.0, -1.0, true, 10), (0, 0)); // -1 >= 0 never
        assert_eq!(span(0.0, -1.0, false, 10), (0, 10)); // -1 < 0 always
        assert_eq!(span(0.0, 1.0, false, 10), (0, 0)); // 1 < 0 never
    }

    /// The horizontal-flip case that motivated this module: dst x = 3 maps to
    /// src x = 0.0 exactly (d = -1, s0 = 3, upper = 4) — column 3 is valid.
    /// The y axis of a horizontal flip does not depend on the column
    /// (d = 0, source row fixed at 0.5, in bounds), so it constrains nothing.
    #[test]
    fn flip_keeps_edge_column() {
        let (lo, hi) = affine_valid_span([(-1.0, 3.0, 4.0), (0.0, 0.5, 2.0)], 4, 1e-6);
        assert_eq!((lo, hi), (0, 4), "flip must keep the src-x==0 column");
    }
}
