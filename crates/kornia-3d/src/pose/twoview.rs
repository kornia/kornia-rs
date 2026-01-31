use crate::pose::fundamental::{fundamental_8point, sampson_distance, FundamentalError};
use crate::pose::{
    decompose_essential, decompose_homography, essential_from_fundamental, homography_4pt2d,
    HomographyError,
};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use rand::prelude::*;
use rand::SeedableRng;

/// Errors returned by two-view initialization utilities.
#[derive(thiserror::Error, Debug)]
pub enum TwoViewError {
    /// Input correspondences are invalid or insufficient.
    #[error("Need at least {required} correspondences and equal lengths")]
    InvalidInput {
        /// Minimum required correspondences for the chosen model.
        required: usize,
    },
    /// RANSAC failed to find a valid model.
    #[error("RANSAC failed to find a valid model")]
    RansacFailure,
    /// Fundamental estimation failed.
    #[error("Fundamental estimation error: {0}")]
    Fundamental(#[from] FundamentalError),
    /// Homography estimation failed.
    #[error("Homography estimation error: {0}")]
    Homography(#[from] HomographyError),
}

/// Parameters for RANSAC model estimation.
#[derive(Clone, Copy, Debug)]
pub struct RansacParams {
    /// Maximum number of RANSAC iterations.
    pub max_iterations: usize,
    /// Inlier threshold (squared pixel error).
    pub threshold: f64,
    /// Minimum number of inliers required for acceptance.
    pub min_inliers: usize,
    /// Optional RNG seed for deterministic runs.
    pub random_seed: Option<u64>,
}

impl Default for RansacParams {
    fn default() -> Self {
        Self {
            max_iterations: 2000,
            threshold: 1.0,
            min_inliers: 15,
            random_seed: Some(0),
        }
    }
}

/// Result of a RANSAC model fit.
#[derive(Clone, Debug)]
pub struct RansacResult<M> {
    /// Estimated model.
    pub model: M,
    /// Per-point inlier mask.
    pub inliers: Vec<bool>,
    /// Total inlier count.
    pub inlier_count: usize,
    /// Sum of inlier errors (lower is better).
    pub score: f64,
}

/// Two-view model used during initialization.
#[derive(Clone, Copy, Debug)]
pub enum TwoViewModel {
    /// Fundamental matrix model.
    Fundamental(Mat3F64),
    /// Homography model.
    Homography(Mat3F64),
}

/// Configuration for two-view initialization.
#[derive(Clone, Debug)]
pub struct TwoViewInitConfig {
    /// RANSAC settings for the fundamental matrix.
    pub ransac_f: RansacParams,
    /// RANSAC settings for the homography.
    pub ransac_h: RansacParams,
    /// Prefer homography when it has this ratio of inliers vs fundamental.
    pub homography_inlier_ratio: f64,
    /// Minimum parallax angle (degrees) for triangulated points.
    pub min_parallax_deg: f64,
}

impl Default for TwoViewInitConfig {
    fn default() -> Self {
        Self {
            ransac_f: RansacParams::default(),
            ransac_h: RansacParams::default(),
            homography_inlier_ratio: 0.8,
            min_parallax_deg: 1.0,
        }
    }
}

/// Output of two-view initialization.
#[derive(Clone, Debug)]
pub struct TwoViewInitResult {
    /// Selected model.
    pub model: TwoViewModel,
    /// Relative rotation from view1 to view2.
    pub rotation: Mat3F64,
    /// Relative translation direction from view1 to view2.
    pub translation: Vec3F64,
    /// Triangulated 3D points for inliers.
    pub points3d: Vec<Vec3F64>,
    /// Inlier mask used for initialization.
    pub inliers: Vec<bool>,
}

/// Estimate a fundamental matrix with RANSAC using the 8-point solver.
pub fn ransac_fundamental(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    params: &RansacParams,
) -> Result<RansacResult<Mat3F64>, TwoViewError> {
    if x1.len() != x2.len() || x1.len() < 8 {
        return Err(TwoViewError::InvalidInput { required: 8 });
    }

    let mut rng = match params.random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut tr = rand::rng();
            StdRng::from_rng(&mut tr)
        }
    };

    let n = x1.len();
    let mut best_model = None;
    let mut best_inliers = Vec::new();
    let mut best_count = 0usize;
    let mut best_score = f64::INFINITY;

    for _ in 0..params.max_iterations {
        let sample = rand::seq::index::sample(&mut rng, n, 8);
        let mut s1 = Vec::with_capacity(8);
        let mut s2 = Vec::with_capacity(8);
        for idx in sample.iter() {
            s1.push(x1[idx]);
            s2.push(x2[idx]);
        }
        let f = match fundamental_8point(&s1, &s2) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let mut inliers = vec![false; n];
        let mut count = 0usize;
        let mut score = 0.0f64;
        for i in 0..n {
            let d = sampson_distance(&f, &x1[i], &x2[i]);
            if d <= params.threshold * params.threshold {
                inliers[i] = true;
                count += 1;
                score += d;
            }
        }

        if count > best_count || (count == best_count && score < best_score) {
            best_model = Some(f);
            best_inliers = inliers;
            best_count = count;
            best_score = score;
        }
    }

    let model = match best_model {
        Some(m) if best_count >= params.min_inliers => m,
        _ => return Err(TwoViewError::RansacFailure),
    };

    Ok(RansacResult {
        model,
        inliers: best_inliers,
        inlier_count: best_count,
        score: best_score,
    })
}

/// Estimate a homography with RANSAC using the 4-point solver.
pub fn ransac_homography(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    params: &RansacParams,
) -> Result<RansacResult<Mat3F64>, TwoViewError> {
    if x1.len() != x2.len() || x1.len() < 4 {
        return Err(TwoViewError::InvalidInput { required: 4 });
    }

    let mut rng = match params.random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut tr = rand::rng();
            StdRng::from_rng(&mut tr)
        }
    };

    let n = x1.len();
    let mut best_model = None;
    let mut best_inliers = Vec::new();
    let mut best_count = 0usize;
    let mut best_score = f64::INFINITY;

    for _ in 0..params.max_iterations {
        let sample = rand::seq::index::sample(&mut rng, n, 4);
        let mut s1 = [[0.0; 2]; 4];
        let mut s2 = [[0.0; 2]; 4];
        for (i, idx) in sample.iter().enumerate() {
            s1[i] = [x1[idx].x, x1[idx].y];
            s2[i] = [x2[idx].x, x2[idx].y];
        }
        let mut h = [[0.0; 3]; 3];
        if homography_4pt2d(&s1, &s2, &mut h).is_err() {
            continue;
        }
        let h = Mat3F64::from_cols(
            Vec3F64::new(h[0][0], h[1][0], h[2][0]),
            Vec3F64::new(h[0][1], h[1][1], h[2][1]),
            Vec3F64::new(h[0][2], h[1][2], h[2][2]),
        );

        let mut inliers = vec![false; n];
        let mut count = 0usize;
        let mut score = 0.0f64;
        for i in 0..n {
            let d = homography_reproj_error(&h, &x1[i], &x2[i]);
            if d <= params.threshold * params.threshold {
                inliers[i] = true;
                count += 1;
                score += d;
            }
        }

        if count > best_count || (count == best_count && score < best_score) {
            best_model = Some(h);
            best_inliers = inliers;
            best_count = count;
            best_score = score;
        }
    }

    let model = match best_model {
        Some(m) if best_count >= params.min_inliers => m,
        _ => return Err(TwoViewError::RansacFailure),
    };

    Ok(RansacResult {
        model,
        inliers: best_inliers,
        inlier_count: best_count,
        score: best_score,
    })
}

/// Initialize a two-view geometry with model selection and triangulation.
pub fn two_view_initialize(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    k1: &Mat3F64,
    k2: &Mat3F64,
    config: &TwoViewInitConfig,
) -> Result<TwoViewInitResult, TwoViewError> {
    let res_f = ransac_fundamental(x1, x2, &config.ransac_f)?;
    let res_h = ransac_homography(x1, x2, &config.ransac_h)?;

    let use_h =
        (res_h.inlier_count as f64) > config.homography_inlier_ratio * (res_f.inlier_count as f64);

    let k1_inv = k1.inverse();
    let k2_inv = k2.inverse();

    let (model, poses, inliers) = if use_h {
        let h = res_h.model;
        (
            TwoViewModel::Homography(h),
            decompose_homography(&h, k1, k2),
            res_h.inliers,
        )
    } else {
        let f = res_f.model;
        let e = essential_from_fundamental(&f, k1, k2);
        (
            TwoViewModel::Fundamental(f),
            decompose_essential(&e),
            res_f.inliers,
        )
    };

    let mut best_pose = None;
    let mut best_count = 0usize;
    let mut best_points = Vec::new();

    let tri_params = TriangulateParams {
        k1_inv: &k1_inv,
        k2_inv: &k2_inv,
        min_parallax_deg: config.min_parallax_deg,
    };

    for (r, t) in poses {
        let (count, points) = triangulate_inliers(x1, x2, &inliers, &r, &t, &tri_params);
        if count > best_count {
            best_count = count;
            best_pose = Some((r, t));
            best_points = points;
        }
    }

    let (r, t) = match best_pose {
        Some(p) => p,
        None => return Err(TwoViewError::RansacFailure),
    };

    Ok(TwoViewInitResult {
        model,
        rotation: r,
        translation: t,
        points3d: best_points,
        inliers,
    })
}

struct TriangulateParams<'a> {
    k1_inv: &'a Mat3F64,
    k2_inv: &'a Mat3F64,
    min_parallax_deg: f64,
}

fn triangulate_inliers(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    inliers: &[bool],
    r: &Mat3F64,
    t: &Vec3F64,
    params: &TriangulateParams<'_>,
) -> (usize, Vec<Vec3F64>) {
    let mut count = 0usize;
    let mut points = Vec::new();

    for i in 0..x1.len() {
        if !inliers[i] {
            continue;
        }
        let x1n = normalize_point(params.k1_inv, &x1[i]);
        let x2n = normalize_point(params.k2_inv, &x2[i]);
        if let Some(x) = triangulate_point_linear(&x1n, &x2n, r, t) {
            let z1 = x.z;
            let x2c = transform_point(r, t, &x);
            let z2 = x2c.z;
            if z1 > 0.0 && z2 > 0.0 && parallax_ok(&x, &x2c, params.min_parallax_deg) {
                points.push(x);
                count += 1;
            }
        }
    }

    (count, points)
}

fn parallax_ok(x1: &Vec3F64, x2: &Vec3F64, min_parallax_deg: f64) -> bool {
    let dot = x1.dot(*x2);
    let n1 = x1.length();
    let n2 = x2.length();
    if n1 <= 1e-12 || n2 <= 1e-12 {
        return false;
    }
    let cos_angle = (dot / (n1 * n2)).clamp(-1.0, 1.0);
    let angle = cos_angle.acos().to_degrees();
    angle >= min_parallax_deg
}

fn normalize_point(k_inv: &Mat3F64, x: &Vec2F64) -> Vec2F64 {
    let xh = Vec3F64::new(x.x, x.y, 1.0);
    let xn = *k_inv * xh;
    Vec2F64::new(xn.x / xn.z, xn.y / xn.z)
}

fn transform_point(r: &Mat3F64, t: &Vec3F64, x: &Vec3F64) -> Vec3F64 {
    *r * *x + *t
}

fn triangulate_point_linear(
    x1: &Vec2F64,
    x2: &Vec2F64,
    r: &Mat3F64,
    t: &Vec3F64,
) -> Option<Vec3F64> {
    let p1 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let r_arr: [f64; 9] = (*r).into();
    let p2 = [
        [r_arr[0], r_arr[1], r_arr[2], t.x],
        [r_arr[3], r_arr[4], r_arr[5], t.y],
        [r_arr[6], r_arr[7], r_arr[8], t.z],
    ];

    let mut a = faer::Mat::<f64>::zeros(4, 4);
    write_dlt_row(&mut a, 0, x1.x, &p1[2], &p1[0]);
    write_dlt_row(&mut a, 1, x1.y, &p1[2], &p1[1]);
    write_dlt_row(&mut a, 2, x2.x, &p2[2], &p2[0]);
    write_dlt_row(&mut a, 3, x2.y, &p2[2], &p2[1]);

    let svd = a.svd();
    let v = svd.v();
    let xh = v.col(3);
    let w = xh[3];
    if w.abs() < 1e-12 {
        return None;
    }
    Some(Vec3F64::new(xh[0] / w, xh[1] / w, xh[2] / w))
}

fn write_dlt_row(a: &mut faer::Mat<f64>, row: usize, x: f64, p3: &[f64; 4], p1: &[f64; 4]) {
    for j in 0..4 {
        a.write(row, j, x * p3[j] - p1[j]);
    }
}

fn homography_reproj_error(h: &Mat3F64, x1: &Vec2F64, x2: &Vec2F64) -> f64 {
    let x1h = Vec3F64::new(x1.x, x1.y, 1.0);
    let hx = *h * x1h;
    if hx.z.abs() < 1e-12 {
        return f64::INFINITY;
    }
    let u = hx.x / hx.z;
    let v = hx.y / hx.z;
    let dx = u - x2.x;
    let dy = v - x2.y;
    dx * dx + dy * dy
}

/// Approximate homography decomposition to recover a relative pose.
///
/// This provides a fast, single-plane approximation and returns two candidates
/// with opposite translation directions. It is sufficient for initialization,
/// but does not recover plane normals.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ransac_fundamental_basic() {
        let f_true = Mat3F64::from_cols(
            Vec3F64::new(0.0, -0.001, 0.01),
            Vec3F64::new(0.0015, 0.0, -0.02),
            Vec3F64::new(-0.01, 0.02, 1.0),
        );
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for i in 0..50 {
            let xi = i as f64 * 1.2 - 10.0;
            let yi = i as f64 * -0.8 + 5.0;
            let x = Vec3F64::new(xi, yi, 1.0);
            let l = f_true * x;
            let xp = if l.x.abs() > 1e-12 { -l.z / l.x } else { 0.0 };
            x1.push(Vec2F64::new(x.x, x.y));
            x2.push(Vec2F64::new(xp, 0.0));
        }

        let params = RansacParams {
            max_iterations: 200,
            threshold: 1.0,
            min_inliers: 10,
            random_seed: Some(0),
        };
        let res = ransac_fundamental(&x1, &x2, &params).unwrap();
        assert!(res.inlier_count >= params.min_inliers);
    }
}
