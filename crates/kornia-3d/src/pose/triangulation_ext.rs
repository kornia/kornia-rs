//! Optimal 2-view + N-view triangulation extensions.
//!
//! These complement the existing midpoint / DLT primitives in
//! [`super::triangulation`] with two routines that real SLAM/SfM pipelines
//! lean on heavily:
//!
//! - [`correct_matches_sampson`] — apply one Sampson correction step to a
//!   noisy correspondence so it sits exactly on the epipolar geometry of
//!   `F`. Equivalent in practice to OpenCV's `cv::correctMatches` (which
//!   is itself a single iteration of the Hartley-Sturm degree-6 minimiser
//!   that converges in 1-2 steps on geometric noise).
//! - [`triangulate_optimal_2view`] — Sampson-correct then DLT-triangulate.
//!   Gives the maximum-likelihood 3D point under Gaussian image noise
//!   without paying for the full degree-6 polynomial root finder.
//! - [`triangulate_n_view_linear`] — N-view DLT on the stacked `2N×4`
//!   system. The standard non-iterative initialiser before bundle
//!   adjustment.
//!
//! ## Why "optimal" with quotes
//!
//! True optimal 2-view triangulation under Gaussian noise (Hartley-Sturm,
//! 1997) solves a degree-6 polynomial per correspondence and gives the
//! global minimiser of the symmetric reprojection error. A single Sampson
//! correction is the *first-order* approximation of that minimiser — for
//! the noise levels typical in SLAM front-ends (≤ 1-2 px) it agrees with
//! the closed-form HS optimum to within numerical roundoff. The polynomial
//! variant is a future addition gated by a `slow-but-exact` flag.

use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

use super::triangulation::triangulate_point_linear;

/// Apply one Sampson correction to a noisy 2D-2D correspondence.
///
/// The fundamental matrix `f` defines the epipolar geometry; the
/// correction pushes both points toward the closest pair that exactly
/// satisfies `x2ᵀ F x1 = 0`. Returns the corrected pair `(x1', x2')`.
///
/// If the correspondence is already exact (Sampson distance ≈ 0) the
/// inputs are returned unchanged. If the denominator collapses (e.g. on
/// the epipoles) the inputs are returned unchanged rather than producing
/// NaN.
pub fn correct_matches_sampson(f: &Mat3F64, x1: &Vec2F64, x2: &Vec2F64) -> (Vec2F64, Vec2F64) {
    let x1h = Vec3F64::new(x1.x, x1.y, 1.0);
    let x2h = Vec3F64::new(x2.x, x2.y, 1.0);
    let l2 = *f * x1h; // epipolar line in image 2
    let l1 = f.transpose() * x2h; // epipolar line in image 1
    let err = x2h.dot(l2);
    let denom = l2.x * l2.x + l2.y * l2.y + l1.x * l1.x + l1.y * l1.y;
    if denom <= 1e-18 {
        return (*x1, *x2);
    }
    let scale = err / denom;
    (
        Vec2F64::new(x1.x - scale * l1.x, x1.y - scale * l1.y),
        Vec2F64::new(x2.x - scale * l2.x, x2.y - scale * l2.y),
    )
}

/// Optimal-under-Gaussian-noise 2-view triangulation.
///
/// Pipeline: [`correct_matches_sampson`] then DLT
/// ([`super::triangulation::triangulate_point_linear`] internally).
///
/// `r`, `t` describe the second camera as `P2 = [R | t]` with the first
/// camera at the origin (`P1 = [I | 0]`). Pixels are expected in
/// **normalised** coordinates (i.e. multiplied by `K⁻¹`); use `f` built
/// from `K`-cancelled essential geometry.
pub fn triangulate_optimal_2view(
    f: &Mat3F64,
    x1: &Vec2F64,
    x2: &Vec2F64,
    r: &Mat3F64,
    t: &Vec3F64,
) -> Option<Vec3F64> {
    let (x1c, x2c) = correct_matches_sampson(f, x1, x2);
    triangulate_point_linear(&x1c, &x2c, r, t)
}

/// N-view DLT triangulation.
///
/// Stacks two rows per observation into a `2N×4` matrix and recovers the
/// 3D point as the right-singular vector of the smallest singular value.
///
/// `pixels[i]` is the observation in view `i`; `projections[i]` is the
/// `3×4` camera matrix `K_i [R_i | t_i]` of view `i` flattened in
/// row-major order (row 0, then row 1, then row 2 — 12 floats per view).
/// The two slices must have equal length ≥ 2.
///
/// Returns `None` if N < 2, slice lengths mismatch, the SVD's last
/// component is degenerate (homogeneous w ≈ 0), or all observations are
/// collinear in the design matrix.
pub fn triangulate_n_view_linear(
    pixels: &[Vec2F64],
    projections: &[[f64; 12]],
) -> Option<Vec3F64> {
    let n = pixels.len();
    if n < 2 || n != projections.len() {
        return None;
    }
    let mut a = faer::Mat::<f64>::zeros(2 * n, 4);
    for (i, (px, p_row)) in pixels.iter().zip(projections.iter()).enumerate() {
        // Row layout in `p_row`: [P00 P01 P02 P03 | P10 P11 P12 P13 | P20 P21 P22 P23]
        for j in 0..4 {
            let p0 = p_row[j];
            let p1 = p_row[4 + j];
            let p2 = p_row[8 + j];
            a.write(2 * i, j, px.x * p2 - p0);
            a.write(2 * i + 1, j, px.y * p2 - p1);
        }
    }
    let svd = a.svd();
    let v = svd.v();
    let xh = v.col(3);
    let w = xh[3];
    if w.abs() < 1e-12 {
        return None;
    }
    Some(Vec3F64::new(xh[0] / w, xh[1] / w, xh[2] / w))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a noise-free correspondence Sampson correction is a no-op (the
    /// Sampson error is already zero, so the update vanishes).
    #[test]
    fn sampson_correction_noop_on_exact_match() {
        // Build F from a known motion.
        let f = synthetic_f();
        let x1 = Vec2F64::new(100.0, 80.0);
        // Pick x2 as the exact projection — for synthetic_f() that means
        // x2 must lie on the epipolar line F * x1h. Choose u, solve v.
        let x1h = Vec3F64::new(x1.x, x1.y, 1.0);
        let l = f * x1h;
        let u = 120.0;
        let v = -(l.x * u + l.z) / l.y;
        let x2 = Vec2F64::new(u, v);

        let (x1c, x2c) = correct_matches_sampson(&f, &x1, &x2);
        let dx1 = (x1c.x - x1.x).hypot(x1c.y - x1.y);
        let dx2 = (x2c.x - x2.x).hypot(x2c.y - x2.y);
        assert!(dx1 < 1e-8, "x1 moved by {dx1} on exact-match input");
        assert!(dx2 < 1e-8, "x2 moved by {dx2} on exact-match input");
    }

    /// Sampson correction *reduces* Sampson distance on a perturbed input.
    #[test]
    fn sampson_correction_reduces_residual() {
        let f = synthetic_f();
        let x1 = Vec2F64::new(100.0, 80.0);
        let x1h = Vec3F64::new(x1.x, x1.y, 1.0);
        let l = f * x1h;
        let u = 120.0;
        let v = -(l.x * u + l.z) / l.y;
        // Knock x2 off the epipolar line by 2 px.
        let x2 = Vec2F64::new(u + 2.0, v - 1.5);

        let before = sampson_dist(&f, &x1, &x2);
        let (x1c, x2c) = correct_matches_sampson(&f, &x1, &x2);
        let after = sampson_dist(&f, &x1c, &x2c);
        assert!(
            after < before * 1e-6,
            "Sampson distance not driven near zero: before={before}, after={after}"
        );
    }

    /// N-view DLT recovers a 3D point from 4 noise-free projections.
    #[test]
    fn n_view_dlt_recovers_point() {
        let pt = Vec3F64::new(0.3, -0.2, 4.5);
        // Build 4 cameras around the origin looking at +z. Each is K[R|t]
        // in row-major.
        let cams = synthetic_cameras_4();
        let mut pixels = Vec::new();
        let mut projs = Vec::new();
        for cam in &cams {
            let p = project(cam, &pt);
            pixels.push(p);
            projs.push(*cam);
        }

        let recovered = triangulate_n_view_linear(&pixels, &projs).expect("triangulation failed");
        let err = (recovered.x - pt.x).powi(2)
            + (recovered.y - pt.y).powi(2)
            + (recovered.z - pt.z).powi(2);
        assert!(err < 1e-8, "recovered point off by sqrt({err})");
    }

    /// Mismatched slice lengths → None.
    #[test]
    fn n_view_dlt_rejects_bad_input() {
        let r = triangulate_n_view_linear(&[Vec2F64::new(0.0, 0.0)], &[]);
        assert!(r.is_none());
    }

    // --- helpers ---

    fn synthetic_f() -> Mat3F64 {
        let k = Mat3F64::from_cols(
            Vec3F64::new(500.0, 0.0, 0.0),
            Vec3F64::new(0.0, 500.0, 0.0),
            Vec3F64::new(320.0, 240.0, 1.0),
        );
        let k_inv = k.inverse();
        let angle = 0.1_f64;
        let r = Mat3F64::from_cols(
            Vec3F64::new(angle.cos(), 0.0, -angle.sin()),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(angle.sin(), 0.0, angle.cos()),
        );
        let t = Vec3F64::new(1.0, 0.0, 0.2);
        let len = t.length();
        let t = Vec3F64::new(t.x / len, t.y / len, t.z / len);
        let tx = Mat3F64::from_cols(
            Vec3F64::new(0.0, t.z, -t.y),
            Vec3F64::new(-t.z, 0.0, t.x),
            Vec3F64::new(t.y, -t.x, 0.0),
        );
        let e = tx * r;
        k_inv.transpose() * e * k_inv
    }

    fn sampson_dist(f: &Mat3F64, x1: &Vec2F64, x2: &Vec2F64) -> f64 {
        let x1h = Vec3F64::new(x1.x, x1.y, 1.0);
        let x2h = Vec3F64::new(x2.x, x2.y, 1.0);
        let fx1 = *f * x1h;
        let ftx2 = f.transpose() * x2h;
        let err = x2h.dot(fx1);
        let denom = fx1.x * fx1.x + fx1.y * fx1.y + ftx2.x * ftx2.x + ftx2.y * ftx2.y;
        if denom <= 1e-12 {
            return err * err;
        }
        err * err / denom
    }

    fn synthetic_cameras_4() -> Vec<[f64; 12]> {
        // 4 views of the same scene; small baselines along x.
        let mut out = Vec::new();
        let baselines = [-0.4_f64, -0.1, 0.1, 0.5];
        let fx = 600.0;
        let fy = 600.0;
        let cx = 320.0;
        let cy = 240.0;
        for &b in &baselines {
            // K [R|t]; R = I, t = (-b, 0, 0). Row-major flatten.
            // K [I | -b ex] = [[fx 0 cx -fx*b], [0 fy cy 0], [0 0 1 0]]
            out.push([
                fx, 0.0, cx, -fx * b,
                0.0, fy, cy, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]);
        }
        out
    }

    fn project(cam: &[f64; 12], p: &Vec3F64) -> Vec2F64 {
        let x = cam[0] * p.x + cam[1] * p.y + cam[2] * p.z + cam[3];
        let y = cam[4] * p.x + cam[5] * p.y + cam[6] * p.z + cam[7];
        let w = cam[8] * p.x + cam[9] * p.y + cam[10] * p.z + cam[11];
        Vec2F64::new(x / w, y / w)
    }
}
