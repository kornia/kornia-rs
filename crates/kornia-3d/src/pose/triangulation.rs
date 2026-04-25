use crate::camera::PinholeCamera;
use crate::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

/// Errors returned by triangulation utilities.
#[derive(Debug, thiserror::Error)]
pub enum TriangulationError {
    /// Point arrays have mismatched lengths.
    #[error("Mismatched point array lengths: pts1 has {pts1_len} elements, pts2 has {pts2_len}")]
    LengthMismatch {
        /// Length of the first point array.
        pts1_len: usize,
        /// Length of the second point array.
        pts2_len: usize,
    },
}

/// Configuration for triangulation-backed pose validation.
#[derive(Clone, Debug)]
pub struct TriangulationConfig {
    /// Minimum parallax angle (degrees) for triangulated points.
    pub min_parallax_deg: f64,
    /// Maximum allowed gap between the closest points on the two rays.
    pub max_midpoint_gap: f64,
    /// Maximum reprojection error when validating a triangulated point.
    pub max_reprojection_error: f64,
    /// Minimum number of positive-depth triangulated points required.
    pub min_cheirality_count: usize,
    /// When true, the two-view candidate check uses ORB-SLAM3's CheckRT:
    /// pixel-space reprojection threshold `check_rt_reproj_th_sq`, points
    /// at infinity allowed when `cosParallax >= 0.99998`, and parallax
    /// reported at sorted index `min(50, N-1)` rather than per-point filtering.
    pub use_orb_slam3_check_rt: bool,
    /// Squared reprojection threshold in pixels, applied to both images,
    /// used only when `use_orb_slam3_check_rt` is true. ORB-SLAM3 uses 4.0.
    pub check_rt_reproj_th_sq: f64,
}

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self {
            min_parallax_deg: 1.0,
            max_midpoint_gap: 1.0,
            max_reprojection_error: 2.0,
            min_cheirality_count: 1,
            use_orb_slam3_check_rt: false,
            check_rt_reproj_th_sq: 4.0,
        }
    }
}

pub(crate) struct TriangulateParams<'a> {
    pub k1_inv: &'a Mat3F64,
    pub k2_inv: &'a Mat3F64,
    pub config: &'a TriangulationConfig,
}

/// Triangulates a 3D point by midpoint between two rays with known relative pose.
///
/// This assumes a relative transform from camera 1 to camera 2:
/// `p_cam2 = r21 * p_cam1 + t21`.
///
/// `ray1_cam1` is a viewing ray in camera-1 coordinates and `ray2_cam2` is a
/// viewing ray in camera-2 coordinates. The returned point is expressed in the
/// camera-1 frame together with the distance (gap) between the closest points
/// on the two rays.
///
/// Returns `None` if the geometry is degenerate (parallel rays or invalid depth).
pub fn triangulate_midpoint_known_pose(
    ray1_cam1: &Vec3F64,
    ray2_cam2: &Vec3F64,
    r21: &Mat3F64,
    t21: &Vec3F64,
) -> Option<(Vec3F64, f64)> {
    let r21_t = r21.transpose();
    let d1 = ray1_cam1.normalize();
    let d2 = (r21_t * *ray2_cam2).normalize();
    let c2 = -(r21_t * *t21);

    let b = d1.dot(d2);
    let denom = 1.0 - b * b;
    if denom.abs() < 1e-8 {
        return None;
    }

    let w0 = -c2;
    let d = d1.dot(w0);
    let e = d2.dot(w0);
    let s1 = (b * e - d) / denom;
    let s2 = (e - b * d) / denom;
    if s1 <= 1e-8 || s2 <= 1e-8 {
        return None;
    }

    let p1 = d1 * s1;
    let p2 = c2 + d2 * s2;
    let gap = (p1 - p2).length();
    let p_cam1 = (p1 + p2) * 0.5;
    Some((p_cam1, gap))
}

pub(crate) fn triangulate_inliers(
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
            let d2_world = r.transpose() * x2c;
            if z1 > 0.0 && z2 > 0.0 && parallax_ok(&x, &d2_world, params.config.min_parallax_deg) {
                points.push(x);
                indices.push(i);
                count += 1;
            }
        }
    }

    (count, points, indices)
}

/// ORB-SLAM3-style CheckRT for a single (R, t) candidate.
///
/// Triangulates each inlier correspondence using DLT with normalized
/// coordinates, then filters against the exact criteria from
/// `TwoViewReconstruction::CheckRT`:
///
///   1. the triangulated point is finite,
///   2. `z1 > 0` OR `cosParallax >= 0.99998` (points at infinity allowed),
///   3. same for `z2`,
///   4. per-point reprojection error in image 1 ≤ `reproj_th_sq` (pixels²),
///   5. per-point reprojection error in image 2 ≤ `reproj_th_sq` (pixels²).
///
/// `parallax_deg` is reported at sorted index `min(50, N-1)` of the collected
/// cosParallax values (ascending), matching ORB-SLAM3's representative
/// parallax heuristic.
///
/// `x1` / `x2` are pixel-space observations; `k1` / `k2` are the intrinsics of
/// the corresponding cameras; `r`, `t` map points from camera 1 into camera 2.
pub fn check_rt(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    inliers: &[bool],
    k1: &Mat3F64,
    k2: &Mat3F64,
    r: &Mat3F64,
    t: &Vec3F64,
    reproj_th_sq: f64,
) -> (usize, Vec<Vec3F64>, Vec<usize>, f64) {
    let k1_inv = k1.inverse();
    let k2_inv = k2.inverse();

    // Camera-1 origin is at (0,0,0). Camera-2 origin in cam1 frame is -R^T * t.
    let o2 = -(r.transpose() * *t);

    let mut points = Vec::new();
    let mut indices = Vec::new();
    let mut cos_values: Vec<f64> = Vec::new();

    for i in 0..x1.len() {
        if !inliers[i] {
            continue;
        }
        let x1n = normalize_point(&k1_inv, &x1[i]);
        let x2n = normalize_point(&k2_inv, &x2[i]);
        let p3d = match triangulate_point_linear(&x1n, &x2n, r, t) {
            Some(p) => p,
            None => continue,
        };
        if !p3d.x.is_finite() || !p3d.y.is_finite() || !p3d.z.is_finite() {
            continue;
        }

        // Parallax at the 3D point w.r.t. both camera centres.
        let normal1 = p3d; // cam1 center is origin.
        let dist1 = normal1.length();
        let normal2 = p3d - o2;
        let dist2 = normal2.length();
        if dist1 <= 1e-12 || dist2 <= 1e-12 {
            continue;
        }
        let cos_parallax = (normal1.dot(normal2) / (dist1 * dist2)).clamp(-1.0, 1.0);

        // Depth in cam1 (origin frame).
        if p3d.z <= 0.0 && cos_parallax < 0.99998 {
            continue;
        }
        // Depth in cam2.
        let p3d_c2 = *r * p3d + *t;
        if p3d_c2.z <= 0.0 && cos_parallax < 0.99998 {
            continue;
        }

        // Reprojection error in image 1: project p3d with K1 * [I|0].
        if p3d.z.abs() < 1e-12 {
            continue;
        }
        let k1c0 = k1.col(0);
        let k1c1 = k1.col(1);
        let k1c2 = k1.col(2);
        let u1 = (k1c0.x * p3d.x + k1c1.x * p3d.y + k1c2.x * p3d.z) / p3d.z;
        let v1 = (k1c0.y * p3d.x + k1c1.y * p3d.y + k1c2.y * p3d.z) / p3d.z;
        let dx1 = u1 - x1[i].x;
        let dy1 = v1 - x1[i].y;
        if dx1 * dx1 + dy1 * dy1 > reproj_th_sq {
            continue;
        }

        // Reprojection error in image 2.
        if p3d_c2.z.abs() < 1e-12 {
            continue;
        }
        let k2c0 = k2.col(0);
        let k2c1 = k2.col(1);
        let k2c2 = k2.col(2);
        let u2 = (k2c0.x * p3d_c2.x + k2c1.x * p3d_c2.y + k2c2.x * p3d_c2.z) / p3d_c2.z;
        let v2 = (k2c0.y * p3d_c2.x + k2c1.y * p3d_c2.y + k2c2.y * p3d_c2.z) / p3d_c2.z;
        let dx2 = u2 - x2[i].x;
        let dy2 = v2 - x2[i].y;
        if dx2 * dx2 + dy2 * dy2 > reproj_th_sq {
            continue;
        }

        cos_values.push(cos_parallax);
        points.push(p3d);
        indices.push(i);
    }

    let count = indices.len();
    let parallax_deg = if cos_values.is_empty() {
        0.0
    } else {
        cos_values.sort_by(|a, b| a.total_cmp(b));
        let idx = cos_values.len().saturating_sub(1).min(50);
        cos_values[idx].clamp(-1.0, 1.0).acos().to_degrees()
    };

    (count, points, indices, parallax_deg)
}

pub(crate) fn parallax_ok(x1: &Vec3F64, x2: &Vec3F64, min_parallax_deg: f64) -> bool {
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

pub(crate) fn normalize_point(k_inv: &Mat3F64, x: &Vec2F64) -> Vec2F64 {
    let xh = Vec3F64::new(x.x, x.y, 1.0);
    let xn = *k_inv * xh;
    Vec2F64::new(xn.x / xn.z, xn.y / xn.z)
}

/// Triangulate a single point from two views using the DLT method.
///
/// P1 = [I | 0] (first camera at origin), P2 = [R | t].
/// Builds the 4x4 linear system `A * X = 0` and solves via SVD.
pub(crate) fn triangulate_point_linear(
    x1: &Vec2F64,
    x2: &Vec2F64,
    r: &Mat3F64,
    t: &Vec3F64,
) -> Option<Vec3F64> {
    let r_arr: [f64; 9] = (*r).into();

    let mut a = faer::Mat::<f64>::zeros(4, 4);
    a.write(0, 0, -1.0);
    a.write(0, 2, x1.x);
    a.write(1, 1, -1.0);
    a.write(1, 2, x1.y);

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

/// A successfully triangulated 3D point with its index in the input pair list.
#[derive(Debug, Clone)]
pub struct TriangulatedPoint {
    /// 3D position in world coordinates.
    pub position: Vec3F64,
    /// Index into the original `pairs` slice that produced this point.
    pub pair_index: usize,
}

/// Triangulate 3D points from undistorted point correspondences across two views.
///
/// For each pair of undistorted 2D observations, computes a 3D point via midpoint
/// triangulation and filters by depth, parallax, midpoint gap, and reprojection
/// error. Accepted points are returned in world coordinates.
///
/// # Arguments
///
/// * `pts1` / `pts2` — undistorted 2D points in views 1 and 2 (same length)
/// * `pose1` / `pose2` — world-to-camera poses for each view
/// * `camera` — pinhole camera (used for reprojection error checks)
/// * `config` — triangulation quality thresholds
pub fn triangulate_matched_points(
    pts1: &[Vec2F64],
    pts2: &[Vec2F64],
    pose1: &Pose3d,
    pose2: &Pose3d,
    camera: &PinholeCamera,
    config: &TriangulationConfig,
) -> Result<Vec<TriangulatedPoint>, TriangulationError> {
    if pts1.len() != pts2.len() {
        return Err(TriangulationError::LengthMismatch {
            pts1_len: pts1.len(),
            pts2_len: pts2.len(),
        });
    }

    let relative_pose = Pose3d::between(pose1, pose2);
    let r_rel = relative_pose.rotation;
    let t_rel = relative_pose.translation;
    if t_rel.length() <= 1e-8 {
        return Ok(Vec::new());
    }

    let pose1_inv = pose1.inverse();
    let reproj_th2 = config.max_reprojection_error * config.max_reprojection_error;
    let mut result = Vec::new();

    for (i, (p1, p2)) in pts1.iter().zip(pts2.iter()).enumerate() {
        let ray1 = Vec3F64::new(
            (p1.x - camera.cx) / camera.fx,
            (p1.y - camera.cy) / camera.fy,
            1.0,
        )
        .normalize();
        let ray2 = Vec3F64::new(
            (p2.x - camera.cx) / camera.fx,
            (p2.y - camera.cy) / camera.fy,
            1.0,
        )
        .normalize();

        let Some((p_cam1, gap)) = triangulate_midpoint_known_pose(&ray1, &ray2, &r_rel, &t_rel)
        else {
            continue;
        };
        if gap > config.max_midpoint_gap {
            continue;
        }

        // Depth checks in both cameras.
        if p_cam1.z <= 1e-6 {
            continue;
        }
        let p_cam2 = r_rel * p_cam1 + t_rel;
        if p_cam2.z <= 1e-6 {
            continue;
        }

        // Parallax check.
        let c2 = -(r_rel.transpose() * t_rel);
        let d1 = p_cam1.normalize();
        let d2_vec = p_cam1 - c2;
        if d2_vec.length() <= 1e-12 {
            continue;
        }
        let d2 = d2_vec.normalize();
        let parallax_deg = d1.dot(d2).clamp(-1.0, 1.0).acos().to_degrees();
        if parallax_deg < config.min_parallax_deg {
            continue;
        }

        // Reprojection error checks.
        let Some(err1) = camera.reprojection_error_sq_cam(&p_cam1, p1.x, p1.y) else {
            continue;
        };
        if err1 > reproj_th2 {
            continue;
        }
        let Some(err2) = camera.reprojection_error_sq_cam(&p_cam2, p2.x, p2.y) else {
            continue;
        };
        if err2 > reproj_th2 {
            continue;
        }

        let position = pose1_inv.transform_point(&p_cam1);
        result.push(TriangulatedPoint {
            position,
            pair_index: i,
        });
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangulate_midpoint_known_pose_simple() {
        let r21 = Mat3F64::IDENTITY;
        let t21 = Vec3F64::new(-1.0, 0.0, 0.0);

        let p_true = Vec3F64::new(0.0, 0.0, 5.0);
        let ray1 = p_true.normalize();

        let p_cam2 = r21 * p_true + t21;
        let ray2 = p_cam2.normalize();

        let (p_est, gap) = triangulate_midpoint_known_pose(&ray1, &ray2, &r21, &t21).unwrap();
        assert!((p_est - p_true).length() < 1e-6);
        assert!(gap < 1e-6);
    }

    #[test]
    fn test_triangulate_midpoint_known_pose_parallel_rays() {
        let r21 = Mat3F64::IDENTITY;
        let t21 = Vec3F64::new(-1.0, 0.0, 0.0);

        let ray1 = Vec3F64::new(0.0, 0.0, 1.0);
        let ray2 = Vec3F64::new(0.0, 0.0, 1.0);
        assert!(triangulate_midpoint_known_pose(&ray1, &ray2, &r21, &t21).is_none());
    }

    #[test]
    fn test_parallax_ok_thresholds() {
        let x1 = Vec3F64::new(1.0, 0.0, 0.0);
        let x2 = Vec3F64::new(1.0, 0.0, 0.0);
        assert!(!parallax_ok(&x1, &x2, 1.0));

        let x3 = Vec3F64::new(0.0, 1.0, 0.0);
        assert!(parallax_ok(&x1, &x3, 30.0));
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

    #[test]
    fn test_triangulate_matched_points_simple() {
        let camera = PinholeCamera {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let pose1 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);
        let pose2 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-1.0, 0.0, 0.0));

        // A point at (0, 0, 5) in world = cam1 frame.
        // In cam1: projects to (320, 240).
        // In cam2: p_cam2 = p_world + (-1, 0, 0) = (-1, 0, 5), projects to (320 - 500/5, 240) = (220, 240).
        let pts1 = vec![Vec2F64::new(320.0, 240.0)];
        let pts2 = vec![Vec2F64::new(220.0, 240.0)];

        let config = TriangulationConfig {
            min_parallax_deg: 0.1,
            max_midpoint_gap: 1.0,
            max_reprojection_error: 2.0,
            min_cheirality_count: 1,
        };

        let result =
            triangulate_matched_points(&pts1, &pts2, &pose1, &pose2, &camera, &config).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].pair_index, 0);
        assert!((result[0].position - Vec3F64::new(0.0, 0.0, 5.0)).length() < 0.05);
    }

    #[test]
    fn test_triangulate_matched_points_rejects_behind_camera() {
        let camera = PinholeCamera {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let pose1 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);
        let pose2 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-1.0, 0.0, 0.0));

        // Both points at principal point — zero parallax, degenerate.
        let pts1 = vec![Vec2F64::new(320.0, 240.0)];
        let pts2 = vec![Vec2F64::new(320.0, 240.0)];

        let config = TriangulationConfig::default();
        let result =
            triangulate_matched_points(&pts1, &pts2, &pose1, &pose2, &camera, &config).unwrap();
        assert!(result.is_empty());
    }
}
