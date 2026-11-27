use crate::camera::CameraModel;
use crate::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

/// Errors returned by triangulation utilities.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TriangulationError {
    /// Point arrays have mismatched lengths.
    #[error("Mismatched point array lengths: pts1 has {pts1_len} elements, pts2 has {pts2_len}")]
    LengthMismatch {
        /// Length of the first point array.
        pts1_len: usize,
        /// Length of the second point array.
        pts2_len: usize,
    },
    /// The per-pair reprojection thresholds do not have one entry per pair.
    #[error("Mismatched threshold length: expected {expected} entries, got {got}")]
    ThresholdLengthMismatch {
        /// Number of point pairs.
        expected: usize,
        /// Number of threshold entries.
        got: usize,
    },
    /// A per-pair reprojection threshold is negative or NaN; squaring it would
    /// turn `-1` into 1 px and NaN into a rejected pair, both silently.
    #[error("Invalid reprojection threshold {value} for pair {index}")]
    InvalidThreshold {
        /// Index of the pair.
        index: usize,
        /// The offending threshold (pixels).
        value: f64,
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
    /// Maximum ratio of second-best / best cheirality counts across the 4
    /// essential-decomposition candidates. If the runner-up retains at least
    /// this fraction of the winner's cheirality inliers the decomposition is
    /// ambiguous (near-degenerate motion: pure rotation, planar scene, or
    /// very low parallax), and `two_view_estimate` returns
    /// `TwoViewError::AmbiguousCheirality` instead of silently committing to a
    /// pose that two candidates score equally well. Set to 1.0 to disable.
    /// Default 0.7 follows ORB-SLAM3's `Initializer::ReconstructF` rule.
    pub cheirality_ambiguity_max: f64,
}

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self {
            min_parallax_deg: 1.0,
            max_midpoint_gap: 1.0,
            max_reprojection_error: 2.0,
            min_cheirality_count: 1,
            cheirality_ambiguity_max: 0.7,
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
    let cap = inliers.iter().filter(|&&b| b).count();
    let mut count = 0usize;
    let mut points = Vec::with_capacity(cap);
    let mut indices = Vec::with_capacity(cap);

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
    a[(0, 0)] = -1.0;
    a[(0, 2)] = x1.x;
    a[(1, 1)] = -1.0;
    a[(1, 2)] = x1.y;

    let p2_2 = [r_arr[2], r_arr[5], r_arr[8], t.z];
    for j in 0..4 {
        let p2_0j = if j < 3 { r_arr[j * 3] } else { t.x };
        let p2_1j = if j < 3 { r_arr[j * 3 + 1] } else { t.y };
        a[(2, j)] = x2.x * p2_2[j] - p2_0j;
        a[(3, j)] = x2.y * p2_2[j] - p2_1j;
    }

    let svd = a.svd().ok()?;
    let v = svd.V();
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

/// Triangulate 3D points from pixel correspondences across two views.
///
/// For each pair of observations, computes a 3D point via midpoint triangulation
/// of the rays `camera` unprojects and filters by depth, parallax, midpoint gap,
/// and reprojection error. Pairs `camera` cannot unproject are skipped. Accepted
/// points are returned in world coordinates.
///
/// Equals [`triangulate_matched_points_two_cameras`] with `camera` for both
/// views and `config.max_reprojection_error` for every pair.
///
/// # Arguments
///
/// * `pts1` / `pts2` — pixels in views 1 and 2 (same length), in the image
///   `camera` describes: raw pixels for a camera with distortion. Points that
///   were already undistorted need a camera with the same intrinsics and zero
///   distortion, or they are undistorted twice.
/// * `pose1` / `pose2` — world-to-camera poses for each view
/// * `camera` — camera model of both views
/// * `config` — triangulation quality thresholds; reprojection error in pixels
///
/// # Returns
///
/// The accepted points, each with the index of the pair that produced it.
///
/// # Errors
///
/// [`TriangulationError::LengthMismatch`] when `pts1` and `pts2` differ in length.
pub fn triangulate_matched_points<C: CameraModel + ?Sized>(
    pts1: &[Vec2F64],
    pts2: &[Vec2F64],
    pose1: &Pose3d,
    pose2: &Pose3d,
    camera: &C,
    config: &TriangulationConfig,
) -> Result<Vec<TriangulatedPoint>, TriangulationError> {
    triangulate_matched_points_two_cameras(pts1, pts2, pose1, pose2, camera, camera, config, None)
}

/// Triangulate 3D points from pixel correspondences across two cameras.
///
/// The two views may use different camera models (a stereo rig with a pinhole
/// and a fisheye, or two calibrations of one sensor), and each pair may carry
/// its own reprojection threshold per view, e.g. scaled by the feature's
/// pyramid level. For each pair the rays `camera1` and `camera2` unproject are
/// triangulated at their midpoint and the point is kept when it lies ahead of
/// both cameras, the rays pass within `config.max_midpoint_gap`, the parallax
/// is at least `config.min_parallax_deg`, and it reprojects into each view
/// within that view's threshold. Pairs either camera cannot unproject are
/// skipped.
///
/// # Arguments
///
/// * `pts1` / `pts2` — pixels in views 1 and 2 (same length), raw pixels of the
///   image `camera1` / `camera2` describes.
/// * `pose1` / `pose2` — world-to-camera poses of each view.
/// * `camera1` / `camera2` — camera model of each view.
/// * `config` — parallax, midpoint-gap and default reprojection thresholds.
/// * `max_reprojection_errors` — per pair, `[view 1, view 2]` reprojection
///   thresholds in pixels; `None` uses `config.max_reprojection_error` for both
///   views of every pair.
///
/// # Returns
///
/// The accepted points in world coordinates, each with the index of the pair
/// that produced it, in pair order.
///
/// # Errors
///
/// [`TriangulationError::LengthMismatch`] when `pts1` and `pts2` differ in
/// length; [`TriangulationError::ThresholdLengthMismatch`] when
/// `max_reprojection_errors` does not have one entry per pair;
/// [`TriangulationError::InvalidThreshold`] when one of its entries is
/// negative or NaN (`f64::INFINITY` accepts every reprojection).
///
/// # Example
///
/// ```rust
/// use kornia_3d::camera::{FisheyeCamera, PinholeCamera};
/// use kornia_3d::pose::{triangulate_matched_points_two_cameras, Pose3d, TriangulationConfig};
/// use kornia_3d::camera::CameraModel;
/// use kornia_algebra::{Mat3F64, Vec3F64};
///
/// let left = PinholeCamera { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0, k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0 };
/// let right = FisheyeCamera { fx: 460.0, fy: 460.0, cx: 320.0, cy: 240.0, k1: 0.03, k2: -0.02, k3: 0.0, k4: 0.0 };
/// let pose1 = Pose3d::IDENTITY;
/// let pose2 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-0.2, 0.0, 0.0));
/// let point = Vec3F64::new(0.3, -0.1, 4.0);
/// let pts1 = [left.project(&point).unwrap()];
/// let pts2 = [right.project(&pose2.transform_point(&point)).unwrap()];
///
/// let points = triangulate_matched_points_two_cameras(
///     &pts1, &pts2, &pose1, &pose2, &left, &right,
///     &TriangulationConfig::default(), Some(&[[1.0, 2.0]]),
/// )?;
/// assert!((points[0].position - point).length() < 1e-6);
/// # Ok::<(), kornia_3d::pose::TriangulationError>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn triangulate_matched_points_two_cameras<
    C1: CameraModel + ?Sized,
    C2: CameraModel + ?Sized,
>(
    pts1: &[Vec2F64],
    pts2: &[Vec2F64],
    pose1: &Pose3d,
    pose2: &Pose3d,
    camera1: &C1,
    camera2: &C2,
    config: &TriangulationConfig,
    max_reprojection_errors: Option<&[[f64; 2]]>,
) -> Result<Vec<TriangulatedPoint>, TriangulationError> {
    if pts1.len() != pts2.len() {
        return Err(TriangulationError::LengthMismatch {
            pts1_len: pts1.len(),
            pts2_len: pts2.len(),
        });
    }
    if let Some(thresholds) = max_reprojection_errors {
        if thresholds.len() != pts1.len() {
            return Err(TriangulationError::ThresholdLengthMismatch {
                expected: pts1.len(),
                got: thresholds.len(),
            });
        }
        let invalid = |t: f64| t.is_nan() || t < 0.0;
        if let Some((index, &[a, b])) = thresholds
            .iter()
            .enumerate()
            .find(|(_, &[a, b])| invalid(a) || invalid(b))
        {
            let value = if invalid(a) { a } else { b };
            return Err(TriangulationError::InvalidThreshold { index, value });
        }
    }

    let relative_pose = Pose3d::between(pose1, pose2);
    let r_rel = relative_pose.rotation;
    let t_rel = relative_pose.translation;
    if t_rel.length() <= 1e-8 {
        return Ok(Vec::new());
    }

    let pose1_inv = pose1.inverse();
    let default_th2 = config.max_reprojection_error * config.max_reprojection_error;
    let mut result = Vec::new();

    for (i, (p1, p2)) in pts1.iter().zip(pts2.iter()).enumerate() {
        let (Some(ray1), Some(ray2)) = (camera1.unproject(p1), camera2.unproject(p2)) else {
            continue;
        };

        let Some((p_cam1, gap)) = triangulate_midpoint_known_pose(&ray1, &ray2, &r_rel, &t_rel)
        else {
            continue;
        };
        if gap > config.max_midpoint_gap {
            continue;
        }

        // The midpoint solve already put the point ahead along both rays; a
        // pinhole's `project` rejects it when it is not in front of the image plane.
        let p_cam2 = r_rel * p_cam1 + t_rel;

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

        let (th2_1, th2_2) = match max_reprojection_errors {
            Some(thresholds) => {
                let [th1, th2] = thresholds[i];
                (th1 * th1, th2 * th2)
            }
            None => (default_th2, default_th2),
        };
        let reprojects1 = camera1
            .project(&p_cam1)
            .is_some_and(|pixel| (pixel - *p1).dot(pixel - *p1) <= th2_1);
        if !reprojects1 {
            continue;
        }
        let reprojects2 = camera2
            .project(&p_cam2)
            .is_some_and(|pixel| (pixel - *p2).dot(pixel - *p2) <= th2_2);
        if !reprojects2 {
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
    use crate::camera::{FisheyeCamera, PinholeCamera};

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
            cheirality_ambiguity_max: 1.0,
        };

        let result =
            triangulate_matched_points(&pts1, &pts2, &pose1, &pose2, &camera, &config).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].pair_index, 0);
        assert!((result[0].position - Vec3F64::new(0.0, 0.0, 5.0)).length() < 0.05);
    }

    #[test]
    fn test_triangulate_matched_points_fisheye_wide_angle() {
        let camera = FisheyeCamera {
            fx: 461.64,
            fy: 459.72,
            cx: 732.95,
            cy: 720.54,
            k1: 0.0344,
            k2: -0.0216,
            k3: 0.0031,
            k4: -0.0005,
        };
        let pose1 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);
        let pose2 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-0.5, 0.0, 0.0));

        // About 76°, 85° and 99° off the optical axis, beyond any pinhole image.
        let points = [
            Vec3F64::new(4.0, 1.0, 1.0),
            Vec3F64::new(-3.0, -2.0, 0.3),
            Vec3F64::new(-3.0, 1.0, -0.5),
        ];
        let pts1: Vec<_> = points.iter().map(|p| camera.project(p).unwrap()).collect();
        let pts2: Vec<_> = points
            .iter()
            .map(|p| camera.project(&pose2.transform_point(p)).unwrap())
            .collect();

        let result = triangulate_matched_points(
            &pts1,
            &pts2,
            &pose1,
            &pose2,
            &camera,
            &TriangulationConfig::default(),
        )
        .unwrap();
        assert_eq!(result.len(), points.len());
        for tp in &result {
            assert!((tp.position - points[tp.pair_index]).length() < 1e-6);
        }
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

    /// The body of `triangulate_matched_points` before it delegated to the
    /// two-camera version, kept verbatim as the bit-identity reference.
    fn triangulate_matched_points_reference<C: CameraModel + ?Sized>(
        pts1: &[Vec2F64],
        pts2: &[Vec2F64],
        pose1: &Pose3d,
        pose2: &Pose3d,
        camera: &C,
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
            let (Some(ray1), Some(ray2)) = (camera.unproject(p1), camera.unproject(p2)) else {
                continue;
            };

            let Some((p_cam1, gap)) = triangulate_midpoint_known_pose(&ray1, &ray2, &r_rel, &t_rel)
            else {
                continue;
            };
            if gap > config.max_midpoint_gap {
                continue;
            }

            // The midpoint solve already put the point ahead along both rays; a
            // pinhole's `project` rejects it when it is not in front of the image plane.
            let p_cam2 = r_rel * p_cam1 + t_rel;

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

            let reprojects = |p_cam: &Vec3F64, observed: &Vec2F64| {
                camera
                    .project(p_cam)
                    .is_some_and(|pixel| (pixel - *observed).dot(pixel - *observed) <= reproj_th2)
            };
            if !reprojects(&p_cam1, p1) || !reprojects(&p_cam2, p2) {
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

    /// Deterministic values in `[-1, 1)` (64-bit LCG, Knuth's MMIX constants).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }
    }

    fn radtan() -> PinholeCamera {
        PinholeCamera {
            fx: 458.654,
            fy: 457.296,
            cx: 367.215,
            cy: 248.375,
            k1: -0.28340811,
            k2: 0.07395907,
            p1: 0.00019359,
            p2: 1.76187114e-05,
        }
    }

    /// 200 noisy pairs seen from two poses, some of which fail each filter.
    fn seeded_pairs(camera: &PinholeCamera) -> (Vec<Vec2F64>, Vec<Vec2F64>, Pose3d, Pose3d) {
        let mut rng = Lcg(0x5eed);
        let pose1 = Pose3d::new(
            kornia_algebra::SO3F64::exp(Vec3F64::new(0.02, -0.05, 0.01)).matrix(),
            Vec3F64::new(0.1, -0.2, 0.05),
        );
        let pose2 = Pose3d::new(
            kornia_algebra::SO3F64::exp(Vec3F64::new(-0.03, 0.08, -0.02)).matrix(),
            Vec3F64::new(-0.4, -0.15, 0.1),
        );
        let (mut pts1, mut pts2) = (Vec::new(), Vec::new());
        while pts1.len() < 200 {
            let point = Vec3F64::new(3.0 * rng.next(), 2.0 * rng.next(), 6.0 + 5.0 * rng.next());
            let (Some(a), Some(b)) = (
                camera.project(&pose1.transform_point(&point)),
                camera.project(&pose2.transform_point(&point)),
            ) else {
                continue;
            };
            let noise = 3.0;
            pts1.push(a + Vec2F64::new(noise * rng.next(), noise * rng.next()));
            pts2.push(b + Vec2F64::new(noise * rng.next(), noise * rng.next()));
        }
        (pts1, pts2, pose1, pose2)
    }

    #[test]
    fn two_camera_matches_single_camera_bits() {
        let camera = radtan();
        let (pts1, pts2, pose1, pose2) = seeded_pairs(&camera);
        let config = TriangulationConfig {
            max_midpoint_gap: 0.05,
            ..TriangulationConfig::default()
        };
        let reference =
            triangulate_matched_points_reference(&pts1, &pts2, &pose1, &pose2, &camera, &config)
                .unwrap();
        let ours =
            triangulate_matched_points(&pts1, &pts2, &pose1, &pose2, &camera, &config).unwrap();
        // Some pairs pass and some fail, so both branches are compared.
        assert!(!reference.is_empty() && reference.len() < pts1.len());
        assert_eq!(ours.len(), reference.len());
        for (a, b) in ours.iter().zip(&reference) {
            assert_eq!(a.pair_index, b.pair_index);
            assert_eq!(a.position.x.to_bits(), b.position.x.to_bits());
            assert_eq!(a.position.y.to_bits(), b.position.y.to_bits());
            assert_eq!(a.position.z.to_bits(), b.position.z.to_bits());
        }
    }

    #[test]
    fn per_point_thresholds_gate_each_view() {
        let camera = radtan();
        let pose1 = Pose3d::IDENTITY;
        let pose2 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-0.5, 0.0, 0.0));
        let point = Vec3F64::new(0.2, -0.1, 5.0);
        let pixel1 = camera.project(&point).unwrap();
        let pixel2 = camera.project(&pose2.transform_point(&point)).unwrap();
        // One pixel off the epipolar line in view 1; the midpoint splits the
        // error between the views, so each view's threshold can reject it.
        let pts1 = [pixel1 + Vec2F64::new(0.0, 1.0)];
        let pts2 = [pixel2];
        let config = TriangulationConfig::default();
        let run = |thresholds: &[[f64; 2]]| {
            triangulate_matched_points_two_cameras(
                &pts1,
                &pts2,
                &pose1,
                &pose2,
                &camera,
                &camera,
                &config,
                Some(thresholds),
            )
            .unwrap()
            .len()
        };
        assert_eq!(run(&[[2.0, 2.0]]), 1);
        assert_eq!(run(&[[0.05, 2.0]]), 0);
        assert_eq!(run(&[[2.0, 0.05]]), 0);
        assert!(matches!(
            triangulate_matched_points_two_cameras(
                &pts1,
                &pts2,
                &pose1,
                &pose2,
                &camera,
                &camera,
                &config,
                Some(&[]),
            ),
            Err(TriangulationError::ThresholdLengthMismatch {
                expected: 1,
                got: 0
            })
        ));
        let invalid = |thresholds: &[[f64; 2]]| match triangulate_matched_points_two_cameras(
            &pts1,
            &pts2,
            &pose1,
            &pose2,
            &camera,
            &camera,
            &config,
            Some(thresholds),
        ) {
            Err(TriangulationError::InvalidThreshold { index, value }) => Some((index, value)),
            _ => None,
        };
        assert_eq!(invalid(&[[-1.0, 2.0]]), Some((0, -1.0)));
        assert!(invalid(&[[2.0, f64::NAN]]).is_some_and(|(i, v)| i == 0 && v.is_nan()));
        assert_eq!(run(&[[f64::INFINITY, 2.0]]), 1);
    }

    #[test]
    fn pinhole_and_fisheye_views_triangulate() {
        let pinhole = radtan();
        let fisheye = FisheyeCamera {
            fx: 461.64,
            fy: 459.72,
            cx: 732.95,
            cy: 720.54,
            k1: 0.0344,
            k2: -0.0216,
            k3: 0.0031,
            k4: -0.0005,
        };
        let pose1 = Pose3d::IDENTITY;
        let pose2 = Pose3d::new(
            kornia_algebra::SO3F64::exp(Vec3F64::new(0.0, 0.3, 0.0)).matrix(),
            Vec3F64::new(-0.3, 0.0, 0.05),
        );
        let points = [
            Vec3F64::new(0.5, 0.2, 4.0),
            Vec3F64::new(-0.8, -0.4, 6.0),
            Vec3F64::new(1.2, 0.6, 3.0),
        ];
        let pts1: Vec<_> = points.iter().map(|p| pinhole.project(p).unwrap()).collect();
        let pts2: Vec<_> = points
            .iter()
            .map(|p| fisheye.project(&pose2.transform_point(p)).unwrap())
            .collect();
        let result = triangulate_matched_points_two_cameras(
            &pts1,
            &pts2,
            &pose1,
            &pose2,
            &pinhole,
            &fisheye,
            &TriangulationConfig::default(),
            None,
        )
        .unwrap();
        assert_eq!(result.len(), points.len());
        for tp in &result {
            assert!((tp.position - points[tp.pair_index]).length() < 1e-6);
        }
        // One model for both views mis-reads view 2 and loses pairs.
        let single = triangulate_matched_points(
            &pts1,
            &pts2,
            &pose1,
            &pose2,
            &pinhole,
            &TriangulationConfig::default(),
        )
        .unwrap();
        assert!(single.len() < points.len());
    }
}
