//! # Two-View Initialization
//!
//! Recovers relative camera pose (R, t) and 3D structure from 2D correspondences.
//!
//! ## Pipeline
//!
//! ```text
//! Correspondences (pixel space)
//!     │
//!     ├─→ RANSAC + 8-point → F (fundamental, pixel space)
//!     │                         │
//!     │                         ▼
//!     │                       E = K2ᵀ F K1 (essential, metric space)
//!     │                         │
//!     │                         ▼ enforce (σ,σ,0), decompose → 4 (R,t) candidates
//!     │
//!     └─→ RANSAC + 4-point → H (homography, pixel space)
//!                               │
//!                               ▼ decompose → multiple (R,t,n) candidates
//!     │
//!     ▼
//! Model selection: compare inlier ratios (H wins when scene is planar)
//!     │
//!     ▼
//! Cheirality check: triangulate with each candidate, pick the one where
//! points are in front of both cameras
//!     │
//!     ▼
//! (R, t_direction, 3D points)   ← translation scale is lost (quotient of SE(3))
//! ```
//!
//! The output translation is a **unit vector** (direction only). Scale is irrecoverable
//! from two views alone — this is the SE(3) → essential manifold quotient in action.

use crate::pose::fundamental::{fundamental_8point, sampson_distance, FundamentalError};
use crate::pose::{
    decompose_essential, decompose_homography, enforce_essential_constraints,
    essential_from_fundamental, homography_4pt2d, HomographyError,
};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use rand::prelude::*;
use rand::SeedableRng;

/// Errors returned by two-view estimation utilities.
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
    /// Inlier threshold (pixel error). Compared against squared errors internally.
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

/// Two-view model selected during estimation.
#[derive(Clone, Copy, Debug)]
pub enum TwoViewModel {
    /// Fundamental matrix model.
    Fundamental(Mat3F64),
    /// Homography model.
    Homography(Mat3F64),
}

/// Configuration for two-view pose estimation.
#[derive(Clone, Debug)]
pub struct TwoViewConfig {
    /// RANSAC settings for the fundamental matrix.
    pub ransac_f: RansacParams,
    /// RANSAC settings for the homography.
    pub ransac_h: RansacParams,
    /// Prefer homography when it has this ratio of inliers vs fundamental.
    pub homography_inlier_ratio: f64,
    /// Minimum parallax angle (degrees) for triangulated points.
    pub min_parallax_deg: f64,
}

impl Default for TwoViewConfig {
    fn default() -> Self {
        Self {
            ransac_f: RansacParams::default(),
            ransac_h: RansacParams::default(),
            homography_inlier_ratio: 0.8,
            min_parallax_deg: 1.0,
        }
    }
}

/// Output of two-view pose estimation.
#[derive(Clone, Debug)]
pub struct TwoViewResult {
    /// Selected model.
    pub model: TwoViewModel,
    /// Relative rotation from view1 to view2.
    pub rotation: Mat3F64,
    /// Relative translation direction from view1 to view2.
    pub translation: Vec3F64,
    /// Triangulated 3D points for inliers.
    pub points3d: Vec<Vec3F64>,
    /// Index into the input `x1`/`x2` arrays for each point in `points3d`.
    pub inlier_indices: Vec<usize>,
    /// Inlier mask from the selected model's RANSAC.
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
            let mut thread_rng = rand::rng();
            StdRng::from_rng(&mut thread_rng)
        }
    };

    let n = x1.len();
    let thresh_sq = params.threshold * params.threshold;
    let mut best_model = None;
    let mut best_inliers = Vec::new();
    let mut best_count = 0usize;
    let mut best_score = f64::INFINITY;

    for _ in 0..params.max_iterations {
        let sample = rand::seq::index::sample(&mut rng, n, 8);
        let mut s1 = [Vec2F64::ZERO; 8];
        let mut s2 = [Vec2F64::ZERO; 8];
        for (i, idx) in sample.iter().enumerate() {
            s1[i] = x1[idx];
            s2[i] = x2[idx];
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
            if d <= thresh_sq {
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

        if best_count == n {
            break;
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
            let mut thread_rng = rand::rng();
            StdRng::from_rng(&mut thread_rng)
        }
    };

    let n = x1.len();
    let thresh_sq = params.threshold * params.threshold;
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
            if d <= thresh_sq {
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

        if best_count == n {
            break;
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

/// Estimate a two-view relative pose with model selection and triangulation.
pub fn two_view_estimate(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    k1: &Mat3F64,
    k2: &Mat3F64,
    config: &TwoViewConfig,
) -> Result<TwoViewResult, TwoViewError> {
    let res_f = ransac_fundamental(x1, x2, &config.ransac_f)?;
    let res_h = ransac_homography(x1, x2, &config.ransac_h)?;

    let use_h =
        (res_h.inlier_count as f64) > config.homography_inlier_ratio * (res_f.inlier_count as f64);

    let k1_inv = k1.inverse();
    let k2_inv = k2.inverse();

    let tri_params = TriangulateParams {
        k1_inv: &k1_inv,
        k2_inv: &k2_inv,
        min_parallax_deg: config.min_parallax_deg,
    };

    let mut best_pose = None;
    let mut best_count = 0usize;
    let mut best_points = Vec::new();
    let mut best_indices = Vec::new();

    let (model, inliers) = if use_h {
        let h = res_h.model;
        let poses = decompose_homography(&h, k1, k2);
        for (r, t) in &poses {
            let (count, points, indices) =
                triangulate_inliers(x1, x2, &res_h.inliers, r, t, &tri_params);
            if count > best_count {
                best_count = count;
                best_pose = Some((*r, *t));
                best_points = points;
                best_indices = indices;
            }
        }
        (TwoViewModel::Homography(h), res_h.inliers)
    } else {
        let f = res_f.model;
        let e = essential_from_fundamental(&f, k1, k2);
        let e = enforce_essential_constraints(&e);
        let poses = decompose_essential(&e);
        for (r, t) in &poses {
            let (count, points, indices) =
                triangulate_inliers(x1, x2, &res_f.inliers, r, t, &tri_params);
            if count > best_count {
                best_count = count;
                best_pose = Some((*r, *t));
                best_points = points;
                best_indices = indices;
            }
        }
        (TwoViewModel::Fundamental(f), res_f.inliers)
    };

    let (r, t) = match best_pose {
        Some(p) => p,
        None => return Err(TwoViewError::RansacFailure),
    };

    Ok(TwoViewResult {
        model,
        rotation: r,
        translation: t,
        points3d: best_points,
        inlier_indices: best_indices,
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
) -> (usize, Vec<Vec3F64>, Vec<usize>) {
    let mut count = 0usize;
    let mut points = Vec::new();
    let mut indices = Vec::new();

    for i in 0..x1.len() {
        if !inliers[i] {
            continue;
        }
        let x1n = normalize_point(params.k1_inv, &x1[i]);
        let x2n = normalize_point(params.k2_inv, &x2[i]);
        if let Some(x) = triangulate_point_linear(&x1n, &x2n, r, t) {
            let z1 = x.z;
            let x2c = *r * x + *t;
            let z2 = x2c.z;
            // True parallax: angle between the two viewing rays in the WORLD frame.
            // Ray from cam1 (at origin): x / |x|
            // Ray from cam2 (at C2 = -Rᵀt) toward x: (x - C2) / |x - C2| = Rᵀ * x2c (unnorm)
            // Using Rᵀ·x2c = x + Rᵀt removes the rotation-induced component that
            // would otherwise make this angle ≈ rotation_angle regardless of depth.
            let d2_world = r.transpose() * x2c;
            if z1 > 0.0 && z2 > 0.0 && parallax_ok(&x, &d2_world, params.min_parallax_deg) {
                points.push(x);
                indices.push(i);
                count += 1;
            }
        }
    }

    (count, points, indices)
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

/// Triangulate a single point from two views using the DLT method.
///
/// P1 = [I | 0] (first camera at origin), P2 = [R | t].
/// Builds the 4x4 linear system `A * X = 0` and solves via SVD.
fn triangulate_point_linear(
    x1: &Vec2F64,
    x2: &Vec2F64,
    r: &Mat3F64,
    t: &Vec3F64,
) -> Option<Vec3F64> {
    let r_arr: [f64; 9] = (*r).into();

    // P1 = [I|0], so rows of A for camera 1 simplify:
    //   row 0: x1.x * P1_row3 - P1_row1 = [-1, 0, x1.x, 0]
    //   row 1: x1.y * P1_row3 - P1_row2 = [0, -1, x1.y, 0]
    let mut a = faer::Mat::<f64>::zeros(4, 4);
    a.write(0, 0, -1.0);
    a.write(0, 2, x1.x);
    a.write(1, 1, -1.0);
    a.write(1, 2, x1.y);

    // r_arr is column-major: r_arr[j*3 + i] = R[i,j].
    // P2 = [R | t], so P2[row, col] = R[row, col] for col < 3, t[row] for col = 3.
    // P2 row 0: [R[0,0], R[0,1], R[0,2], tx] = [r_arr[0], r_arr[3], r_arr[6], tx]
    // P2 row 1: [R[1,0], R[1,1], R[1,2], ty] = [r_arr[1], r_arr[4], r_arr[7], ty]
    // P2 row 2: [R[2,0], R[2,1], R[2,2], tz] = [r_arr[2], r_arr[5], r_arr[8], tz]
    let p2_2 = [r_arr[2], r_arr[5], r_arr[8], t.z];
    for j in 0..4 {
        let p2_0j = if j < 3 { r_arr[j * 3] } else { t.x };
        let p2_1j = if j < 3 { r_arr[j * 3 + 1] } else { t.y };
        a.write(2, j, x2.x * p2_2[j] - p2_0j);
        a.write(3, j, x2.y * p2_2[j] - p2_1j);
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

/// Computes the squared reprojection error for mapping `x1` to `x2` via the homography `h`.
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

    #[test]
    fn test_ransac_fundamental_invalid_input() {
        let x1 = vec![Vec2F64::new(0.0, 0.0); 7];
        let x2 = vec![Vec2F64::new(0.0, 0.0); 7];
        let params = RansacParams::default();
        let err = ransac_fundamental(&x1, &x2, &params).unwrap_err();
        match err {
            TwoViewError::InvalidInput { required } => assert_eq!(required, 8),
            other => panic!("unexpected error: {other:?}"),
        }

        let x2 = vec![Vec2F64::new(0.0, 0.0); 8];
        let err = ransac_fundamental(&x1, &x2, &params).unwrap_err();
        match err {
            TwoViewError::InvalidInput { required } => assert_eq!(required, 8),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_ransac_homography_basic() {
        let h_true = Mat3F64::from_cols(
            Vec3F64::new(1.2, 0.0, 0.001),
            Vec3F64::new(0.1, 0.9, 0.002),
            Vec3F64::new(5.0, -3.0, 1.0),
        );

        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for i in 0..25 {
            let xi = (i % 5) as f64 * 2.0 - 4.0;
            let yi = (i / 5) as f64 * 1.5 - 3.0;
            let p = Vec3F64::new(xi, yi, 1.0);
            let hp = h_true * p;
            let u = hp.x / hp.z;
            let v = hp.y / hp.z;
            x1.push(Vec2F64::new(xi, yi));
            x2.push(Vec2F64::new(u, v));
        }

        let params = RansacParams {
            max_iterations: 100,
            threshold: 1e-6,
            min_inliers: 12,
            random_seed: Some(0),
        };
        let res = ransac_homography(&x1, &x2, &params).unwrap();
        assert!(res.inlier_count >= params.min_inliers);
    }

    #[test]
    fn test_ransac_homography_invalid_input() {
        let x1 = vec![Vec2F64::new(0.0, 0.0); 3];
        let x2 = vec![Vec2F64::new(0.0, 0.0); 3];
        let params = RansacParams::default();
        let err = ransac_homography(&x1, &x2, &params).unwrap_err();
        match err {
            TwoViewError::InvalidInput { required } => assert_eq!(required, 4),
            other => panic!("unexpected error: {other:?}"),
        }

        let x2 = vec![Vec2F64::new(0.0, 0.0); 4];
        let err = ransac_homography(&x1, &x2, &params).unwrap_err();
        match err {
            TwoViewError::InvalidInput { required } => assert_eq!(required, 4),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_parallax_ok_thresholds() {
        let x1 = Vec3F64::new(1.0, 0.0, 0.0);
        let x2 = Vec3F64::new(1.0, 0.0, 0.0);
        assert!(!parallax_ok(&x1, &x2, 1.0));

        let x3 = Vec3F64::new(0.0, 1.0, 0.0);
        assert!(parallax_ok(&x1, &x3, 30.0));
    }

    /// End-to-end two-view pose estimation on real EuRoC MH_01_easy images.
    ///
    /// Reads two grayscale frames, runs ORB detection + matching, estimates
    /// the relative pose via `two_view_estimate`, and compares against
    /// ground truth camera-frame pose.
    ///
    /// Frame pair: 1403636633263555584 → 1403636634263555584 (20 frames apart).
    /// Ground truth camera-frame relative pose:
    ///   - Rotation: 2.698°
    ///   - Translation: 659mm, direction [0.242, -0.233, 0.942]
    #[test]
    fn test_two_view_euroc_mh01() {
        use kornia_imgproc::features::{match_orb_descriptors, OrbDetector, OrbMatchConfig};

        // Load images.
        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let data_dir = manifest
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/data");

        let img1_u8 =
            kornia_io::png::read_image_png_mono8(data_dir.join("mh01_frame1.png")).unwrap();
        let img2_u8 =
            kornia_io::png::read_image_png_mono8(data_dir.join("mh01_frame2.png")).unwrap();

        let img1 = u8_to_f32_image(&img1_u8);
        let img2 = u8_to_f32_image(&img2_u8);

        // ORB detect + extract.
        let orb = OrbDetector::default();
        let (kps1, scales1, ori1, _) = orb.detect(&img1).unwrap();
        let (desc1, mask1) = orb.extract(&img1, &kps1, &scales1, &ori1).unwrap();
        let (kps2, scales2, ori2, _) = orb.detect(&img2).unwrap();
        let (desc2, mask2) = orb.extract(&img2, &kps2, &scales2, &ori2).unwrap();

        // Filter by valid descriptors (border mask).
        let (valid_kps1, valid_ori1, valid_desc1) = filter_by_mask(&kps1, &ori1, &desc1, &mask1);
        let (valid_kps2, valid_ori2, valid_desc2) = filter_by_mask(&kps2, &ori2, &desc2, &mask2);

        // Match descriptors.
        let match_config = OrbMatchConfig {
            nn_ratio: 0.6,
            th_low: 50,
            check_orientation: true,
            histo_length: 30,
        };
        let matches = match_orb_descriptors(
            &valid_ori1,
            &valid_desc1,
            &valid_ori2,
            &valid_desc2,
            match_config,
        );
        assert!(
            matches.len() >= 15,
            "too few ORB matches: {} (need >= 15)",
            matches.len()
        );

        // Convert matched keypoints to Vec2F64 (x=col, y=row).
        let pts1: Vec<Vec2F64> = matches
            .iter()
            .map(|&(i, _)| {
                let (row, col) = valid_kps1[i];
                Vec2F64::new(col as f64, row as f64)
            })
            .collect();
        let pts2: Vec<Vec2F64> = matches
            .iter()
            .map(|&(_, j)| {
                let (row, col) = valid_kps2[j];
                Vec2F64::new(col as f64, row as f64)
            })
            .collect();

        // EuRoC MH_01_easy cam0 intrinsics.
        let k = Mat3F64::from_cols(
            Vec3F64::new(458.654, 0.0, 0.0),
            Vec3F64::new(0.0, 457.296, 0.0),
            Vec3F64::new(367.215, 248.375, 1.0),
        );

        let config = TwoViewConfig {
            ransac_f: RansacParams {
                max_iterations: 2000,
                threshold: 1.0,
                min_inliers: 15,
                random_seed: Some(42),
            },
            ransac_h: RansacParams {
                max_iterations: 2000,
                threshold: 1.0,
                min_inliers: 8,
                random_seed: Some(42),
            },
            homography_inlier_ratio: 0.8,
            min_parallax_deg: 0.5,
        };

        let result = two_view_estimate(&pts1, &pts2, &k, &k, &config).unwrap();

        // Should select fundamental model (general motion, not planar).
        assert!(
            matches!(result.model, TwoViewModel::Fundamental(_)),
            "expected fundamental model"
        );

        // Check rotation angle: GT is 2.698°, allow 5° error.
        let r = result.rotation;
        let trace = r.col(0).x + r.col(1).y + r.col(2).z;
        let est_angle_rad = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        let est_angle_deg = est_angle_rad.to_degrees();
        let gt_angle_deg = 2.698;
        assert!(
            (est_angle_deg - gt_angle_deg).abs() < 5.0,
            "rotation error too large: estimated {est_angle_deg:.2}°, GT {gt_angle_deg}°"
        );

        // Check translation direction: GT is [0.242, -0.233, 0.942].
        // Translation can be recovered up to sign, so check both directions.
        let t = result.translation.normalize();
        let gt_t = Vec3F64::new(0.2422, -0.2330, 0.9418).normalize();
        let cos_angle = t.dot(gt_t).clamp(-1.0, 1.0);
        let t_err_deg = cos_angle.abs().acos().to_degrees();
        assert!(
            t_err_deg < 15.0,
            "translation direction error too large: {t_err_deg:.2}°"
        );

        // Should have triangulated some points.
        assert!(
            !result.points3d.is_empty(),
            "expected triangulated 3D points"
        );
    }

    fn u8_to_f32_image(
        src: &kornia_image::Image<u8, 1, kornia_tensor::CpuAllocator>,
    ) -> kornia_image::Image<f32, 1, kornia_tensor::CpuAllocator> {
        let mut dst =
            kornia_image::Image::from_size_val(src.size(), 0.0, kornia_tensor::CpuAllocator)
                .unwrap();
        src.as_slice()
            .iter()
            .zip(dst.as_slice_mut())
            .for_each(|(&s, d)| *d = s as f32 / 255.0);
        dst
    }

    type OrbFiltered = (Vec<(f32, f32)>, Vec<f32>, Vec<[u8; 32]>);

    fn filter_by_mask(
        kps: &[(f32, f32)],
        ori: &[f32],
        desc: &[[u8; 32]],
        mask: &[bool],
    ) -> OrbFiltered {
        let mut out_kps = Vec::new();
        let mut out_ori = Vec::new();
        let mut out_desc = Vec::new();
        let mut desc_idx = 0;
        for (i, &valid) in mask.iter().enumerate() {
            if valid {
                out_kps.push(kps[i]);
                out_ori.push(ori[i]);
                out_desc.push(desc[desc_idx]);
                desc_idx += 1;
            }
        }
        (out_kps, out_ori, out_desc)
    }

    #[test]
    fn test_normalize_point_identity_and_scaled() {
        let k = Mat3F64::from_cols(
            Vec3F64::new(2.0, 0.0, 0.0),
            Vec3F64::new(0.0, 3.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let k_inv = k.inverse();
        let x = Vec2F64::new(4.0, 6.0);
        let xn = normalize_point(&k_inv, &x);
        assert!((xn.x - 2.0).abs() < 1e-12);
        assert!((xn.y - 2.0).abs() < 1e-12);
    }
}
