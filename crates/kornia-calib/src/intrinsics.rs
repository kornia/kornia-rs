//! Focal-length estimation from a single planar-fiducial view (Zhang's method).
//!
//! Bundle adjustment ([`crate::calibrate_apriltag`]) treats camera intrinsics as
//! **fixed** — it only solves poses and points. So a camera whose intrinsics are
//! unknown (a factory-blank OAK reporting `fx = 0`, an RGB-only sensor) can only
//! enter the rig with a *guessed* focal, which biases every reprojection.
//!
//! This module recovers a real focal from the geometry already in hand: the tag
//! (or full AprilGrid board) is planar, so the mapping board→image is a
//! homography `H = K·[r₁ r₂ t]`. Enforcing that `r₁` and `r₂` are orthonormal
//! columns of a rotation gives two scalar constraints on `K`, and — under the
//! usual single-view assumptions (principal point known, zero skew, square
//! pixels) — each constraint yields an estimate of `f²`.
//!
//! # What is and isn't observable from one view
//!
//! - **Focal** — observable. Recovered here.
//! - **Principal point** — *not* separably observable from a single homography;
//!   the caller must supply it (image centre is the standard default).
//! - **Radial/tangential distortion** — *not* observable from one snapshot. It
//!   is only well-conditioned when the board sweeps the full field of view over
//!   many frames (multi-view Zhang). Estimating it from one view fits noise, so
//!   it is deliberately left to a future multi-view path.

use kornia_3d::pose::homography_dlt;
use kornia_algebra::Vec2F64;

/// Plausibility window for a recovered focal length, in pixels. Anything outside
/// is treated as a degenerate solve (fronto-parallel board, near-collinear
/// corners) and rejected rather than trusted.
const FOCAL_MIN_PX: f64 = 100.0;
const FOCAL_MAX_PX: f64 = 20_000.0;

/// Estimate a single focal length (`fx = fy = f`, in pixels) from ONE planar
/// view of a fiducial, given the principal point.
///
/// `board_pts` are the fiducial's corner coordinates on its own plane (metres,
/// z = 0 implied); `image_pts` are the matching observed pixels, in the same
/// order. Pass **all** covisible corners of an AprilGrid board for a
/// least-squares homography — more points make the estimate far steadier than
/// the 4 corners of a lone tag.
///
/// Returns `None` when the view is too degenerate to constrain the focal (fewer
/// than 4 points, a failed homography, or no positive `f²` root inside the
/// plausibility window).
///
/// # Assumptions
///
/// Principal point fixed at `(cx, cy)`, zero skew, square pixels. These are the
/// standard single-homography assumptions; relax them only with multiple views.
pub fn estimate_focal(
    board_pts: &[Vec2F64],
    image_pts: &[Vec2F64],
    cx: f64,
    cy: f64,
) -> Option<f64> {
    if board_pts.len() < 4 || board_pts.len() != image_pts.len() {
        return None;
    }
    let h = homography_dlt(board_pts, image_pts).ok()?;

    // Column-major [f64; 9]: cols 0,1 are the first two homography columns
    // h₁ = (h11, h21, h31), h₂ = (h12, h22, h32).
    let a = h.to_cols_array();
    let (h11, h21, h31) = (a[0], a[1], a[2]);
    let (h12, h22, h32) = (a[3], a[4], a[5]);

    // Move the principal point to the origin so K reduces to diag(f, f, 1) and
    // ω = K⁻ᵀK⁻¹ = diag(1/f², 1/f², 1). Columns of K⁻¹H are then
    // (u_j/f, v_j/f, w_j) with u,v,w below.
    let (u1, v1, w1) = (h11 - cx * h31, h21 - cy * h31, h31);
    let (u2, v2, w2) = (h12 - cx * h32, h22 - cy * h32, h32);

    // Two rotation-column constraints, each an independent estimate of f²:
    //   orthogonality  r₁·r₂ = 0        → f² = -(u1 u2 + v1 v2)/(w1 w2)
    //   equal-norm     ‖r₁‖ = ‖r₂‖      → f² = (u1²+v1² − u2²−v2²)/(w2² − w1²)
    // Each degenerates on a different fronto-parallel condition (w1 w2 → 0, or
    // w1 ≈ w2), so we keep whichever roots are positive and average them.
    let mut roots: Vec<f64> = Vec::with_capacity(2);
    let denom_ortho = w1 * w2;
    if denom_ortho.abs() > 1e-12 {
        let f2 = -(u1 * u2 + v1 * v2) / denom_ortho;
        if f2 > 0.0 {
            roots.push(f2);
        }
    }
    let denom_norm = w2 * w2 - w1 * w1;
    if denom_norm.abs() > 1e-12 {
        let f2 = (u1 * u1 + v1 * v1 - u2 * u2 - v2 * v2) / denom_norm;
        if f2 > 0.0 {
            roots.push(f2);
        }
    }
    if roots.is_empty() {
        return None;
    }
    let f = (roots.iter().sum::<f64>() / roots.len() as f64).sqrt();
    (FOCAL_MIN_PX..=FOCAL_MAX_PX).contains(&f).then_some(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Project a planar board through a known K (f, cx, cy) and pose, then check
    // estimate_focal recovers f. Synthetic ground truth per the geometry skill:
    // build (R, t), project, round-trip, assert.
    fn project(
        board: &[Vec2F64],
        f: f64,
        cx: f64,
        cy: f64,
        // camera rotation as three columns (world→cam) and translation
        r: [[f64; 3]; 3],
        t: [f64; 3],
    ) -> Vec<Vec2F64> {
        board
            .iter()
            .map(|p| {
                // board point (x, y, 0) in world → camera
                let (x, y) = (p.x, p.y);
                let xc = r[0][0] * x + r[0][1] * y + t[0];
                let yc = r[1][0] * x + r[1][1] * y + t[1];
                let zc = r[2][0] * x + r[2][1] * y + t[2];
                Vec2F64::new(f * xc / zc + cx, f * yc / zc + cy)
            })
            .collect()
    }

    fn grid(n: usize, step: f64) -> Vec<Vec2F64> {
        let mut v = Vec::new();
        for i in 0..n {
            for j in 0..n {
                v.push(Vec2F64::new(i as f64 * step, j as f64 * step));
            }
        }
        v
    }

    #[test]
    fn recovers_focal_from_tilted_board() {
        let (f, cx, cy) = (900.0, 640.0, 360.0);
        let board = grid(4, 0.05);
        // Tilt ~25° about x so the board is NOT fronto-parallel (both
        // constraints well-conditioned), pushed 0.6 m in front.
        let a = 25f64.to_radians();
        let (c, s) = (a.cos(), a.sin());
        let r = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]];
        let img = project(&board, f, cx, cy, r, [-0.075, -0.075, 0.6]);
        let est = estimate_focal(&board, &img, cx, cy).expect("focal");
        assert!((est - f).abs() < 5.0, "estimated {est}, want {f}");
    }

    #[test]
    fn four_corner_tag_recovers_focal() {
        let (f, cx, cy) = (1032.0, 640.0, 360.0);
        let s = 0.08;
        let tag = vec![
            Vec2F64::new(0.0, 0.0),
            Vec2F64::new(s, 0.0),
            Vec2F64::new(s, s),
            Vec2F64::new(0.0, s),
        ];
        let a = 30f64.to_radians();
        let (c, sn) = (a.cos(), a.sin());
        // tilt about the y axis this time
        let r = [[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]];
        let img = project(&tag, f, cx, cy, r, [-0.04, -0.04, 0.5]);
        let est = estimate_focal(&tag, &img, cx, cy).expect("focal");
        assert!((est - f).abs() < 10.0, "estimated {est}, want {f}");
    }

    #[test]
    fn rejects_too_few_points() {
        let p = vec![Vec2F64::new(0.0, 0.0); 3];
        assert!(estimate_focal(&p, &p, 0.0, 0.0).is_none());
    }
}
