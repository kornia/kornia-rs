//! AprilTag 6-DOF pose estimation via Lu-Hager-Mjolsness orthogonal iteration.

use crate::camera::PinholeCamera;
use crate::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

use super::homography_4pt2d;

/// A recovered tag pose with reprojection error.
#[derive(Debug, Clone)]
pub struct TagPose {
    /// The world-to-camera rigid transform.
    pub pose: Pose3d,
    /// Sum of squared reprojection errors over the 4 tag corners (pixels²).
    pub error: f64,
}

/// Two candidate poses from the planar ambiguity (one is typically degenerate).
/// `best` has the lower reprojection error.
#[derive(Debug, Clone)]
pub struct TagPosePair {
    /// Lower-error solution.
    pub best: TagPose,
    /// Higher-error solution (may be degenerate/behind-camera).
    pub second: TagPose,
}

/// Error type for tag pose estimation.
#[derive(thiserror::Error, Debug)]
pub enum AprilTagPoseError {
    /// The homography system is degenerate (coplanar-degenerate inputs).
    #[error("Homography DLT system is singular")]
    SingularHomography,
    /// SVD failed to converge.
    #[error("SVD failed to converge")]
    SvdFailed,
}

/// Decompose a planar homography (mapping tag-normalized ±1 coords to pixels) into
/// an initial pose using the camera intrinsics.
///
/// The last column of K⁻¹·H is t/scale (not metric t); OI rewrites t on its first step.
fn homography_to_pose(h: &[[f64; 3]; 3], fx: f64, fy: f64, cx: f64, cy: f64) -> Pose3d {
    let mut hk = [[0.0f64; 3]; 3];
    for j in 0..3 {
        hk[0][j] = (h[0][j] - cx * h[2][j]) / fx;
        hk[1][j] = (h[1][j] - cy * h[2][j]) / fy;
        hk[2][j] = h[2][j];
    }
    let n1 = (hk[0][0] * hk[0][0] + hk[1][0] * hk[1][0] + hk[2][0] * hk[2][0]).sqrt();
    let n2 = (hk[0][1] * hk[0][1] + hk[1][1] * hk[1][1] + hk[2][1] * hk[2][1]).sqrt();
    let scale = (n1 + n2) * 0.5;
    let r1 = Vec3F64::new(hk[0][0] / n1, hk[1][0] / n1, hk[2][0] / n1);
    let r2 = Vec3F64::new(hk[0][1] / n2, hk[1][1] / n2, hk[2][1] / n2);
    let r3 = r1.cross(r2);
    let t = Vec3F64::new(hk[0][2] / scale, hk[1][2] / scale, hk[2][2] / scale);
    Pose3d::new(Mat3F64::from_cols(r1, r2, r3), t)
}

/// Projection operator for image ray v: F = v·vᵀ / (vᵀ·v).
fn calc_f(v: Vec3F64) -> Mat3F64 {
    let s = 1.0 / v.dot(v);
    Mat3F64::from_cols(v * (v.x * s), v * (v.y * s), v * (v.z * s))
}

/// Run Lu-Hager-Mjolsness orthogonal iteration to refine a pose.
///
/// Returns `(refined_pose, sum_sq_reprojection_error)`.
fn orthogonal_iteration(
    object_pts: &[Vec3F64; 4],
    image_rays: &[Vec3F64; 4],
    init_pose: Pose3d,
    n_iters: usize,
) -> Result<(Pose3d, f64), AprilTagPoseError> {
    let f = [
        calc_f(image_rays[0]),
        calc_f(image_rays[1]),
        calc_f(image_rays[2]),
        calc_f(image_rays[3]),
    ];
    let avg_f = (f[0] + f[1] + f[2] + f[3]) * 0.25;
    let m1 = Mat3F64::IDENTITY - avg_f;
    // (I − avg_F) is singular when all image rays are identical — degenerate input.
    if m1.determinant().abs() < 1e-8 {
        return Err(AprilTagPoseError::SingularHomography);
    }
    let m1_inv = m1.inverse();

    let p_mean =
        (object_pts[0] + object_pts[1] + object_pts[2] + object_pts[3]) * 0.25;
    let p_res = [
        object_pts[0] - p_mean,
        object_pts[1] - p_mean,
        object_pts[2] - p_mean,
        object_pts[3] - p_mean,
    ];

    let mut rotation = init_pose.rotation;
    let mut translation = init_pose.translation;
    let mut error = f64::MAX;

    for _ in 0..n_iters {
        // Update translation: t = M1⁻¹ · mean((Fᵢ − I) · R·pᵢ)
        let mut m2 = Vec3F64::ZERO;
        for i in 0..4 {
            let rp = rotation * object_pts[i];
            m2 += f[i] * rp - rp; // (Fᵢ − I) · R·pᵢ
        }
        translation = m1_inv * (m2 * 0.25);

        // Update rotation via Kabsch/Procrustes
        let mut q = [Vec3F64::ZERO; 4];
        let mut q_mean = Vec3F64::ZERO;
        for i in 0..4 {
            q[i] = f[i] * (rotation * object_pts[i] + translation);
            q_mean += q[i];
        }
        q_mean *= 0.25;

        // M3 = Σᵢ (qᵢ − q̄)(pᵢ − p̄)ᵀ
        let mut m3 = Mat3F64::from_cols(Vec3F64::ZERO, Vec3F64::ZERO, Vec3F64::ZERO);
        for i in 0..4 {
            let dq = q[i] - q_mean;
            m3 += Mat3F64::from_cols(dq * p_res[i].x, dq * p_res[i].y, dq * p_res[i].z);
        }

        // SVD of M3; R = U·Vᵀ (faer — not svd3_f64 which has a Jacobi bug)
        let arr: [f64; 9] = m3.into();
        let a = faer::Mat::<f64>::from_fn(3, 3, |i, j| arr[j * 3 + i]);
        let svd = a.svd().map_err(|_| AprilTagPoseError::SvdFailed)?;
        let u_f = svd.U();
        let v_f = svd.V();
        let u = Mat3F64::from_cols(
            Vec3F64::new(u_f[(0, 0)], u_f[(1, 0)], u_f[(2, 0)]),
            Vec3F64::new(u_f[(0, 1)], u_f[(1, 1)], u_f[(2, 1)]),
            Vec3F64::new(u_f[(0, 2)], u_f[(1, 2)], u_f[(2, 2)]),
        );
        let v = Mat3F64::from_cols(
            Vec3F64::new(v_f[(0, 0)], v_f[(1, 0)], v_f[(2, 0)]),
            Vec3F64::new(v_f[(0, 1)], v_f[(1, 1)], v_f[(2, 1)]),
            Vec3F64::new(v_f[(0, 2)], v_f[(1, 2)], v_f[(2, 2)]),
        );
        let mut r_new = u * v.transpose();
        // Fix reflection (det = −1) by negating the third column, matching the C reference.
        if r_new.determinant() < 0.0 {
            r_new.z_axis = -r_new.z_axis;
        }
        rotation = r_new;

        // Error: Σᵢ ‖(I − Fᵢ)(R·pᵢ + t)‖²
        error = 0.0;
        for i in 0..4 {
            let z = rotation * object_pts[i] + translation;
            let e = z - f[i] * z;
            error += e.dot(e);
        }
    }

    Ok((Pose3d::new(rotation, translation), error))
}

/// Estimate the 6-DOF pose of a planar AprilTag from 4 corner correspondences.
///
/// # Arguments
/// * `object_pts` — 4 object points in tag metric frame: `[(-s,-s,0),(s,-s,0),(s,s,0),(-s,s,0)]`
///   where `s = tag_size / 2`. Pass them in this order (BL, BR, TR, TL matches kornia Detection.quad.corners order).
/// * `image_pts` — matching 2D image coordinates (pixels), same order as object_pts.
/// * `camera` — pinhole camera intrinsics (fx, fy, cx, cy). Distortion coefficients ignored.
/// * `n_iters` — number of orthogonal-iteration refinement steps (default: 50).
///
/// # Returns
/// `TagPosePair` with `best` (lower reprojection error) and `second` (higher error / ambiguous solution).
///
/// # Note
/// Uses the Lu-Hager-Mjolsness (1993) orthogonal iteration algorithm, matching the
/// AprilRobotics C reference implementation in `apriltag_pose.c`.
pub fn estimate_tag_pose(
    object_pts: &[Vec3F64; 4],
    image_pts: &[Vec2F64; 4],
    camera: &PinholeCamera,
    n_iters: usize,
) -> Result<TagPosePair, AprilTagPoseError> {
    let (fx, fy, cx, cy) = camera.intrinsics();

    // H maps tag-normalized corners (±1) → image pixels
    let tag_norm: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let img_arr: [[f64; 2]; 4] = [
        [image_pts[0].x, image_pts[0].y],
        [image_pts[1].x, image_pts[1].y],
        [image_pts[2].x, image_pts[2].y],
        [image_pts[3].x, image_pts[3].y],
    ];
    let mut h = [[0.0f64; 3]; 3];
    homography_4pt2d(&tag_norm, &img_arr, &mut h)
        .map_err(|_| AprilTagPoseError::SingularHomography)?;

    let init_pose = homography_to_pose(&h, fx, fy, cx, cy);

    // Unnormalized image rays: vᵢ = [(u−cx)/fx, (v−cy)/fy, 1]
    let image_rays: [Vec3F64; 4] = [
        Vec3F64::new((image_pts[0].x - cx) / fx, (image_pts[0].y - cy) / fy, 1.0),
        Vec3F64::new((image_pts[1].x - cx) / fx, (image_pts[1].y - cy) / fy, 1.0),
        Vec3F64::new((image_pts[2].x - cx) / fx, (image_pts[2].y - cy) / fy, 1.0),
        Vec3F64::new((image_pts[3].x - cx) / fx, (image_pts[3].y - cy) / fy, 1.0),
    ];

    // Pose 1: refine from H decomposition
    let (pose1, err1) = orthogonal_iteration(object_pts, &image_rays, init_pose, n_iters)?;

    // Pose 2: other planar ambiguity — negate first two R columns
    let r2_init = Mat3F64::from_cols(
        -init_pose.rotation.x_axis(),
        -init_pose.rotation.y_axis(),
        init_pose.rotation.z_axis(),
    );
    let (pose2, err2) =
        orthogonal_iteration(object_pts, &image_rays, Pose3d::new(r2_init, init_pose.translation), n_iters)?;

    let (best, second) = if err1 <= err2 {
        (TagPose { pose: pose1, error: err1 }, TagPose { pose: pose2, error: err2 })
    } else {
        (TagPose { pose: pose2, error: err2 }, TagPose { pose: pose1, error: err1 })
    };

    Ok(TagPosePair { best, second })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn project_pts(
        object_pts: &[Vec3F64; 4],
        pose: &Pose3d,
        camera: &PinholeCamera,
    ) -> [Vec2F64; 4] {
        [0, 1, 2, 3].map(|i| {
            let p = pose.transform_point(&object_pts[i]);
            Vec2F64::new(
                camera.fx * p.x / p.z + camera.cx,
                camera.fy * p.y / p.z + camera.cy,
            )
        })
    }

    fn rotation_error_rad(r_est: &Mat3F64, r_gt: &Mat3F64) -> f64 {
        let r_rel = *r_est * r_gt.transpose();
        let trace = r_rel.x_axis.x + r_rel.y_axis.y + r_rel.z_axis.z;
        ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
    }

    #[test]
    fn test_estimate_tag_pose_roundtrip() -> Result<(), AprilTagPoseError> {
        let camera = test_camera();
        let object_pts = [
            Vec3F64::new(-0.05, -0.05, 0.0),
            Vec3F64::new(0.05, -0.05, 0.0),
            Vec3F64::new(0.05, 0.05, 0.0),
            Vec3F64::new(-0.05, 0.05, 0.0),
        ];
        let angle = 10.0 * PI / 180.0;
        let r_gt = Mat3F64::from_cols(
            Vec3F64::new(angle.cos(), angle.sin(), 0.0),
            Vec3F64::new(-angle.sin(), angle.cos(), 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let t_gt = Vec3F64::new(0.01, 0.005, 0.3);
        let pose_gt = Pose3d::new(r_gt, t_gt);

        let image_pts = project_pts(&object_pts, &pose_gt, &camera);
        let result = estimate_tag_pose(&object_pts, &image_pts, &camera, 50)?;
        let best = &result.best;

        let rot_err = rotation_error_rad(&best.pose.rotation, &r_gt);
        let t_err = (best.pose.translation - t_gt).length();

        assert!(rot_err < 1e-6, "rotation error {rot_err} >= 1e-6 rad");
        assert!(t_err < 1e-4, "translation error {t_err} >= 1e-4 m");
        assert!(best.error < 1e-10, "reprojection error {} >= 1e-10", best.error);
        assert!(
            best.pose.transform_point(&object_pts[0]).z > 0.0,
            "cheirality violated"
        );
        Ok(())
    }

    #[test]
    fn test_estimate_tag_pose_noisy() -> Result<(), AprilTagPoseError> {
        let camera = test_camera();
        let object_pts = [
            Vec3F64::new(-0.05, -0.05, 0.0),
            Vec3F64::new(0.05, -0.05, 0.0),
            Vec3F64::new(0.05, 0.05, 0.0),
            Vec3F64::new(-0.05, 0.05, 0.0),
        ];
        // 20° tilt around X gives depth variation between corners, making the
        // Procrustes step well-conditioned (avoids near-frontal planar ambiguity).
        let tilt = 20.0 * PI / 180.0;
        let r_gt = Mat3F64::from_cols(
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, tilt.cos(), tilt.sin()),
            Vec3F64::new(0.0, -tilt.sin(), tilt.cos()),
        );
        let t_gt = Vec3F64::new(0.0, 0.0, 0.3);
        let pose_gt = Pose3d::new(r_gt, t_gt);

        let mut image_pts = project_pts(&object_pts, &pose_gt, &camera);
        // Deterministic zero-mean 0.5 px noise (sum_x=0, sum_y=0, max=0.5 px)
        let noise = [[0.3f64, -0.4], [-0.3, 0.4], [0.4, -0.3], [-0.4, 0.3]];
        for i in 0..4 {
            image_pts[i] = Vec2F64::new(image_pts[i].x + noise[i][0], image_pts[i].y + noise[i][1]);
        }

        let result = estimate_tag_pose(&object_pts, &image_pts, &camera, 50)?;
        let best = &result.best;

        let rot_err = rotation_error_rad(&best.pose.rotation, &r_gt);
        let t_err = (best.pose.translation - t_gt).length();
        let t_tol = 0.05 * t_gt.length();

        assert!(rot_err < PI / 180.0, "rotation error {rot_err} >= 1°");
        assert!(t_err < t_tol, "translation error {t_err} >= 5% of depth ({t_tol})");
        Ok(())
    }

    #[test]
    fn test_estimate_tag_pose_degenerate() {
        let camera = test_camera();
        let object_pts = [
            Vec3F64::new(-0.05, -0.05, 0.0),
            Vec3F64::new(0.05, -0.05, 0.0),
            Vec3F64::new(0.05, 0.05, 0.0),
            Vec3F64::new(-0.05, 0.05, 0.0),
        ];
        // All image points at the same pixel → degenerate DLT system
        let image_pts = [Vec2F64::new(320.0, 240.0); 4];
        let result = estimate_tag_pose(&object_pts, &image_pts, &camera, 50);
        assert!(
            matches!(result, Err(AprilTagPoseError::SingularHomography)),
            "expected SingularHomography, got {result:?}"
        );
    }
}
