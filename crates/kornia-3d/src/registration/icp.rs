//! Projective point-to-plane ICP on RGB-D pyramids: association and normal equations.
//!
//! The method is the projective point-to-plane ICP of KinectFusion (Newcombe et al., "KinectFusion:
//! Real-time dense surface mapping and tracking", ISMAR 2011, section 3.5), with the
//! point-to-plane error of Chen and Medioni ("Object modelling by registration of multiple range
//! images", 1992). Each valid source pixel `p_s` is moved by the current estimate
//! `T_target_source = (R, t)`, `p = R p_s + t`, projected into the target image and matched to
//! the target vertex `v_t` and normal `n_t` of the pixel it lands on (no kd-tree). The residual
//! is `r = n_t . (p - v_t)` in metres.
//!
//! Pose updates are left-multiplied, `T <- exp(xi) T` with tangent `xi = [w, v]` (rotation first),
//! so the Jacobian row of a correspondence is `[(p x n_t)^T, n_t^T]` with the target held fixed.
//!
//! Departures from KinectFusion, each deliberate:
//!
//! * Rows are weighted `huber(r / sigma) / sigma^2` (iteratively reweighted least squares), not
//!   uniformly, so a few bad matches that pass the gates do not dominate the solve.
//! * The normal gate compares the rotated source normal `R n_s` with `n_t`, as KinectFusion
//!   does, but the distance gate is on `|p - v_t|`, the full 3D distance.
//!
//! This module holds the association pass and the accumulation of `J^T W J` and `J^T W r`; the
//! solver that iterates them comes in a later change.

// Used by the solver that lands next; until then only the tests call this module.
#![cfg_attr(not(test), allow(dead_code))]

use super::rgbd::{is_valid_normal, is_valid_vertex, RgbdLevel};
use crate::ransac::{HuberKernel, RobustKernel};
use kornia_algebra::{Mat3F64, Mat6F64, Vec3F64, Vec6F64};
use kornia_image::Image;

/// Smallest residual sigma, in metres. A row weight is `1 / sigma^2`, so a floor near
/// `f64::MIN_POSITIVE` would overflow `J^T W J`; at 1 nm the weight is `1e18`.
const MIN_SIGMA_M: f64 = 1e-9;

/// Gates and weighting of one association pass.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PassGates {
    /// Largest `|p - v_t|` of a kept correspondence, in metres.
    pub max_dist_m: f64,
    /// Smallest `(R n_s) . n_t` of a kept correspondence (cosine of the largest normal angle).
    pub cos_max_normal_angle: f64,
    /// Huber threshold on the normalised residual `r / sigma`.
    pub huber_delta: f64,
    /// One-sigma error of a residual, in metres.
    pub sigma_m: f64,
}

/// One gated correspondence of a projective-association pass.
pub(crate) struct Correspondence {
    /// Index of the source pixel in its level.
    pub src_idx: usize,
    /// Target vertex the residual is measured against.
    pub v_t: Vec3F64,
    /// Unit target normal.
    pub n_t: Vec3F64,
    /// Jacobian row `[(p x n_t)^T, n_t^T]`, the target held fixed.
    pub j: Vec6F64,
    /// Point-to-plane residual in metres.
    pub r: f64,
    /// One-sigma error of `r` in metres.
    pub sigma: f64,
}

/// Counts of one association pass, besides the correspondences themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PassCounts {
    /// Source pixels with a valid vertex and normal.
    pub num_valid: usize,
    /// Of those, the ones that projected onto valid target geometry (before gating).
    pub num_associated: usize,
}

/// Accumulated normal equations of one association pass over a level.
pub(crate) struct NormalEquations {
    /// `J^T W J`, symmetric.
    pub a: Mat6F64,
    /// `J^T W r`.
    pub b: Vec6F64,
    /// Correspondences that survived gating.
    pub num_inliers: usize,
    /// Pass counts before gating.
    pub counts: PassCounts,
    /// Unweighted sum of squared residuals over the inliers.
    pub sum_sq_residual: f64,
}

/// Huber-and-sigma weight of a row: `huber(r / sigma) / sigma^2`, the Huber weight of
/// [`HuberKernel`] on the residual normalised by its own sigma.
#[inline]
pub(crate) fn row_weight(r: f64, sigma: f64, huber_delta: f64) -> f64 {
    let rn = r / sigma;
    HuberKernel.weight(rn * rn, huber_delta * huber_delta) / (sigma * sigma)
}

/// The three channels of pixel `idx` of a vertex or normal map.
#[inline]
fn pixel3(img: &Image<f32, 3>, idx: usize) -> [f32; 3] {
    let s = &img.as_slice()[3 * idx..3 * idx + 3];
    [s[0], s[1], s[2]]
}

/// [`pixel3`] widened to `f64`.
#[inline]
fn vec3(p: [f32; 3]) -> Vec3F64 {
    Vec3F64::new(p[0] as f64, p[1] as f64, p[2] as f64)
}

/// A target vertex and normal.
struct TargetSample {
    vertex: Vec3F64,
    /// Unit length.
    normal: Vec3F64,
}

/// Nearest-pixel sample: the target vertex and normal of the pixel `(u, v)` rounds to.
fn sample_nearest(tgt: &RgbdLevel, u: f64, v: f64) -> Option<TargetSample> {
    let size = tgt.size();
    // round half up by truncation: `f64::round` is a libm call on the x86-64 baseline target
    if !(u >= -0.5 && v >= -0.5) {
        return None;
    }
    let (ui, vi) = ((u + 0.5) as usize, (v + 0.5) as usize);
    if ui >= size.width || vi >= size.height {
        return None;
    }
    let idx = vi * size.width + ui;
    let (vt, nt) = (pixel3(&tgt.vertices, idx), pixel3(&tgt.normals, idx));
    if !is_valid_vertex(&vt) || !is_valid_normal(&nt) {
        return None;
    }
    Some(TargetSample {
        vertex: vec3(vt),
        normal: vec3(nt),
    })
}

/// One projective-association pass: gate each source pixel and hand every surviving
/// correspondence to `visit`.
///
/// `rotation` and `translation` are the current estimate of `T_target_source`.
pub(crate) fn for_each_correspondence(
    src: &RgbdLevel,
    tgt: &RgbdLevel,
    rotation: &Mat3F64,
    translation: &Vec3F64,
    gates: &PassGates,
    mut visit: impl FnMut(&Correspondence),
) -> PassCounts {
    let mut counts = PassCounts {
        num_valid: 0,
        num_associated: 0,
    };
    let max_dist_sq = gates.max_dist_m * gates.max_dist_m;
    let sigma = gates.sigma_m.max(MIN_SIGMA_M);
    let vertices = src.vertices.as_slice().chunks_exact(3);
    let normals = src.normals.as_slice().chunks_exact(3);

    for (src_idx, (v_s, n_s)) in vertices.zip(normals).enumerate() {
        let (v_s, n_s) = ([v_s[0], v_s[1], v_s[2]], [n_s[0], n_s[1], n_s[2]]);
        if !is_valid_vertex(&v_s) || !is_valid_normal(&n_s) {
            continue;
        }
        counts.num_valid += 1;

        // p = T * p_s in the target frame
        let p = *rotation * vec3(v_s) + *translation;
        let Some(uv) = tgt.camera.project_to_pixel(&p, 0.0) else {
            continue;
        };
        let Some(TargetSample {
            vertex: v_t,
            normal: n_t,
        }) = sample_nearest(tgt, uv.x, uv.y)
        else {
            continue;
        };
        counts.num_associated += 1;

        // gates pass on `<=` / `>=` so that a NaN distance or cosine fails them
        let diff = p - v_t;
        let dist_ok = diff.dot(diff) <= max_dist_sq;
        let angle_ok = (*rotation * vec3(n_s)).dot(n_t) >= gates.cos_max_normal_angle;
        if !(dist_ok && angle_ok) {
            continue;
        }

        let pxn = p.cross(n_t);
        visit(&Correspondence {
            src_idx,
            v_t,
            n_t,
            j: Vec6F64::new(pxn.x, pxn.y, pxn.z, n_t.x, n_t.y, n_t.z),
            r: n_t.dot(diff),
            sigma,
        });
    }

    counts
}

/// One projective-association pass: gate, weight and accumulate `J^T W J` and `J^T W r` in
/// `f64`.
pub(crate) fn accumulate_level(
    src: &RgbdLevel,
    tgt: &RgbdLevel,
    rotation: &Mat3F64,
    translation: &Vec3F64,
    gates: &PassGates,
) -> NormalEquations {
    let mut a = Mat6F64::zeros();
    let mut b = Vec6F64::zeros();
    let (mut num_inliers, mut sum_sq_residual) = (0usize, 0.0f64);
    let counts = for_each_correspondence(src, tgt, rotation, translation, gates, |c| {
        let w = row_weight(c.r, c.sigma, gates.huber_delta);
        b += c.j * (w * c.r);
        // upper triangle only; mirrored below
        for i in 0..6 {
            for k in i..6 {
                a[(i, k)] += w * c.j[i] * c.j[k];
            }
        }
        num_inliers += 1;
        sum_sq_residual += c.r * c.r;
    });
    a.fill_lower_triangle_with_upper_triangle();

    NormalEquations {
        a,
        b,
        num_inliers,
        counts,
        sum_sq_residual,
    }
}

#[cfg(test)]
mod tests {
    use super::super::rgbd::RgbdPyramid;
    use super::super::synth::{render_depth_mm, test_camera, Scene, TEST_SIZE};
    use super::*;
    use kornia_algebra::SO3F64;

    const GATES: PassGates = PassGates {
        max_dist_m: 0.1,
        cos_max_normal_angle: 0.866, // 30 deg
        huber_delta: 1.345,
        sigma_m: 0.002,
    };

    /// Source at the origin, target camera moved by a small rotation and translation; returns
    /// both finest levels and the ground-truth `T_target_source`.
    fn moved_pair(
    ) -> Result<(RgbdPyramid, RgbdPyramid, Mat3F64, Vec3F64), Box<dyn std::error::Error>> {
        let (camera, scene) = (test_camera(), Scene::corner_and_sphere());
        let rot_wc = SO3F64::exp(Vec3F64::new(0.015, -0.01, 0.02)).matrix();
        let t_wc = Vec3F64::new(0.02, -0.01, 0.015);
        let src_depth = render_depth_mm(
            &scene,
            &camera,
            TEST_SIZE,
            &Mat3F64::IDENTITY,
            &Vec3F64::ZERO,
        )?;
        let tgt_depth = render_depth_mm(&scene, &camera, TEST_SIZE, &rot_wc, &t_wc)?;
        let src = RgbdPyramid::from_depth_mm(&src_depth, &camera, 1)?;
        let tgt = RgbdPyramid::from_depth_mm(&tgt_depth, &camera, 1)?;
        // T_target_source = T_world_cam^-1, the source camera sitting at the world origin
        let rot = rot_wc.transpose();
        Ok((src, tgt, rot, -(rot * t_wc)))
    }

    /// `J^T W J` and `J^T W r` must equal the same sums built from central differences of each
    /// row's residual under a left perturbation `exp(xi) T`, weights held fixed.
    ///
    /// The estimate is off the truth by 1 cm and 0.5 deg, so the residuals are not zero and
    /// some rows are Huber-downweighted; a wrong sign, axis order or update side in the
    /// Jacobian would break the match.
    #[test]
    fn normal_equations_match_finite_differences() -> Result<(), Box<dyn std::error::Error>> {
        let (src, tgt, rot_gt, t_gt) = moved_pair()?;
        let dr = SO3F64::exp(Vec3F64::new(0.0, 0.5f64.to_radians(), 0.0)).matrix();
        let (rot, t) = (dr * rot_gt, dr * t_gt + Vec3F64::new(0.01, 0.0, -0.005));
        let (s0, t0) = (&src.levels[0], &tgt.levels[0]);

        let eqs = accumulate_level(s0, t0, &rot, &t, &GATES);
        assert!(eqs.num_inliers > 20_000, "inliers: {}", eqs.num_inliers);
        assert!(eqs.counts.num_associated >= eqs.num_inliers);
        assert!(eqs.counts.num_valid >= eqs.counts.num_associated);

        // (source point, target point, target normal, weight) of each kept row
        let mut rows: Vec<(Vec3F64, Vec3F64, Vec3F64, f64)> = Vec::new();
        let mut downweighted = 0;
        for_each_correspondence(s0, t0, &rot, &t, &GATES, |c| {
            let w = row_weight(c.r, c.sigma, GATES.huber_delta);
            if (c.r / c.sigma).abs() > GATES.huber_delta {
                downweighted += 1;
            }
            rows.push((vec3(pixel3(&s0.vertices, c.src_idx)), c.v_t, c.n_t, w));
        });
        assert_eq!(rows.len(), eqs.num_inliers);
        assert!(downweighted > 100, "Huber branch unused: {downweighted}");

        let residual = |xi: &[f64; 6], (p, v, n, _): &(Vec3F64, Vec3F64, Vec3F64, f64)| {
            let d = SO3F64::exp(Vec3F64::new(xi[0], xi[1], xi[2])).matrix();
            let q = d * (rot * *p + t) + Vec3F64::new(xi[3], xi[4], xi[5]);
            n.dot(q - *v)
        };
        let h = 1e-6;
        let mut a_fd = Mat6F64::zeros();
        let mut b_fd = Vec6F64::zeros();
        for row in &rows {
            let r0 = residual(&[0.0; 6], row);
            let j = Vec6F64::from_fn(|k, _| {
                let mut xi = [0.0; 6];
                xi[k] = h;
                let plus = residual(&xi, row);
                xi[k] = -h;
                (plus - residual(&xi, row)) / (2.0 * h)
            });
            a_fd += j * j.transpose() * row.3;
            b_fd += j * (row.3 * r0);
        }

        let mut worst_a: f64 = 0.0;
        for i in 0..6 {
            for k in 0..6 {
                let scale = (eqs.a[(i, i)] * eqs.a[(k, k)]).sqrt();
                worst_a = worst_a.max((eqs.a[(i, k)] - a_fd[(i, k)]).abs() / scale);
            }
        }
        let b_scale = (0..6).map(|i| eqs.b[i].abs()).fold(0.0, f64::max);
        let worst_b = (eqs.b - b_fd).amax() / b_scale;
        assert!(
            worst_a < 1e-6,
            "J^T W J off finite differences: {worst_a:.3e}"
        );
        assert!(
            worst_b < 1e-6,
            "J^T W r off finite differences: {worst_b:.3e}"
        );
        assert!((eqs.a - eqs.a.transpose()).amax() == 0.0);
        Ok(())
    }

    /// At the true pose most valid pixels associate and pass the gates, and the residuals are
    /// at the millimetre quantization of the depth; tight gates reject what they should.
    #[test]
    fn gates_keep_true_matches_and_reject_far_ones() -> Result<(), Box<dyn std::error::Error>> {
        let (src, tgt, rot_gt, t_gt) = moved_pair()?;
        let (s0, t0) = (&src.levels[0], &tgt.levels[0]);

        let eqs = accumulate_level(s0, t0, &rot_gt, &t_gt, &GATES);
        assert!(
            eqs.num_inliers as f64 > 0.9 * eqs.counts.num_associated as f64,
            "{} of {} associated kept",
            eqs.num_inliers,
            eqs.counts.num_associated
        );
        let rmse = (eqs.sum_sq_residual / eqs.num_inliers as f64).sqrt();
        assert!(rmse < 2e-3, "rmse at the true pose: {rmse:.2e} m");

        // a 5 cm offset with a 1 mm distance gate keeps almost nothing
        let far = t_gt + Vec3F64::new(0.05, 0.0, 0.0);
        let tight = PassGates {
            max_dist_m: 0.001,
            ..GATES
        };
        let eqs_far = accumulate_level(s0, t0, &rot_gt, &far, &tight);
        assert!(eqs_far.num_inliers * 20 < eqs.num_inliers);
        assert_eq!(eqs_far.counts.num_valid, eqs.counts.num_valid);

        // a normal gate of -1 (any angle) keeps at least as many as the 30 deg gate
        let any_angle = PassGates {
            cos_max_normal_angle: -1.0,
            ..GATES
        };
        assert!(
            accumulate_level(s0, t0, &rot_gt, &t_gt, &any_angle).num_inliers >= eqs.num_inliers
        );
        Ok(())
    }

    #[test]
    fn row_weight_is_huber_over_sigma_squared() {
        let (sigma, delta) = (0.002, 1.345);
        assert_eq!(row_weight(0.001, sigma, delta), 1.0 / (sigma * sigma));
        // beyond the threshold the weight falls as delta / |r / sigma|
        let r = 0.01;
        let expected = (delta / (r / sigma)) / (sigma * sigma);
        assert!((row_weight(-r, sigma, delta) - expected).abs() < 1e-9 * expected);
    }
}
