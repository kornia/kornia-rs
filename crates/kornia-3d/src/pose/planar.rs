//! 6-DOF pose of a planar quad from 4 coplanar correspondences, via Lu-Hager-Mjolsness
//! orthogonal iteration (1993).
//!
//! Nothing here is AprilTag-specific -- a tag is one caller among many. Any 4 coplanar object
//! points with known image correspondences work: a fiducial marker, a chessboard cell, a printed
//! rectangle, a door frame.
//!
//! # Corner order is a contract, not a convenience
//!
//! The initial estimate comes from a homography fitted between a FIXED canonical square
//! `[(-1,-1), (1,-1), (1,1), (-1,1)]` and `image_pts`. The orthogonal iteration then refines
//! against the caller's real `object_pts`, so any consistent correspondence converges EVENTUALLY --
//! but the starting point, and therefore the number of iterations needed, depends on how close the
//! caller's ordering is to that canonical one.
//!
//! Measured on exact, noise-free projections of a square at 0.3 m, 20 deg tilt:
//!
//! | `object_pts` order (correspondence consistent throughout) | error at `n_iters = 50` | at 500 |
//! |---|---|---|
//! | canonical `[(-s,-s), (s,-s), (s,s), (-s,s)]`              | 0.0000 deg  | 0.0000 deg |
//! | cyclic shift by one                                       | 2.56 deg    | 0.0000 deg |
//! | reversed winding                                          | 15.58 deg   | 0.0000 deg |
//!
//! So `n_iters = 50` is only sufficient for the canonical order. Pass the corners in that order, or
//! raise `n_iters` substantially.
//!
//! Target SHAPE, by contrast, is nearly free: a 4:1 rectangle converges to 0.0000 deg at
//! `n_iters = 50`, indistinguishable from a square. Only strongly non-affine quads (a trapezoid
//! measured 0.234 deg at 50, 0.0000 deg at 5000) need more iterations for shape reasons.
//!
//! # The second solution
//!
//! [`PlanarPosePair`] returns two poses. The second is the first with its first two rotation
//! columns negated -- a 180 deg in-plane rotation, NOT the classical mirror ambiguity of a plane
//! viewed from one side.
//!
//! In practice it is usually degenerate: on every synthetic configuration measured it came back
//! BEHIND the camera with `error == f64::MAX`. The AprilRobotics C reference gates the second
//! solution on positive depth before returning it; this port does not, so callers must check
//! `second.pose.translation.z` and `second.error` themselves rather than treating it as a
//! ready-made alternative hypothesis. See the tracking issue before relying on it.
//!
use crate::camera::PinholeCamera;
use crate::pose::{homography_4pt2d, Pose3d};
use kornia_algebra::linalg::svd::svd3_f64;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

/// A recovered planar pose with its reprojection error.
#[derive(Debug, Clone)]
pub struct PlanarPose {
    /// The world-to-camera rigid transform.
    pub pose: Pose3d,
    /// Sum of squared pixel reprojection errors over the 4 tag corners (pixels²).
    /// Lower is better; < 1.0 indicates sub-pixel accuracy.
    pub error: f64,
}

/// Two candidate poses from the planar ambiguity (one is typically degenerate).
/// `best` has the lower reprojection error.
#[derive(Debug, Clone)]
pub struct PlanarPosePair {
    /// Lower-error solution.
    pub best: PlanarPose,
    /// Higher-error solution (may be degenerate/behind-camera).
    pub second: PlanarPose,
}

/// Error type for tag pose estimation.
#[derive(thiserror::Error, Debug)]
pub enum PlanarPoseError {
    /// The initial homography could not be fitted from the 4 correspondences.
    #[error("initial homography could not be fitted: {0}")]
    Homography(#[from] crate::pose::HomographyError),

    /// The orthogonal iteration's linear system `(I - avg_F)` is singular, which happens when the
    /// four image rays are (near) identical. Distinct from [`Self::Homography`]: the homography
    /// fitted, the refinement is what degenerated.
    #[error("orthogonal-iteration system is singular (degenerate image rays)")]
    SingularIteration,

    /// `n_iters` was zero. The refinement loop would never run, so the returned pose would be the
    /// raw homography decomposition carrying a sentinel error rather than a measured one --
    /// silently useless, so it is refused instead.
    #[error("n_iters must be non-zero")]
    ZeroIterations,
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
/// Returns `(refined_pose, sum_sq_reprojection_error_in_pixels)`.
fn orthogonal_iteration(
    object_pts: &[Vec3F64; 4],
    image_rays: &[Vec3F64; 4],
    image_pts: &[Vec2F64; 4],
    init_pose: Pose3d,
    n_iters: usize,
    camera: &PinholeCamera,
) -> Result<(Pose3d, f64), PlanarPoseError> {
    let (fx, fy, cx, cy) = camera.intrinsics();
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
        return Err(PlanarPoseError::SingularIteration);
    }
    let m1_inv = m1.inverse();

    let p_mean = (object_pts[0] + object_pts[1] + object_pts[2] + object_pts[3]) * 0.25;
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

        // SVD of M3 via kornia-algebra's 3×3 solver; R = U·Vᵀ with a reflection fix.
        let svd = svd3_f64(&m3);
        let mut r_new = *svd.u() * svd.v().transpose();
        // Fix reflection (det = −1) by negating the third column, matching the C reference.
        if r_new.determinant() < 0.0 {
            r_new.z_axis = -r_new.z_axis;
        }
        rotation = r_new;

        // Error: Σᵢ (Δuᵢ² + Δvᵢ²) — sum of squared pixel reprojection errors
        error = 0.0;
        for i in 0..4 {
            let p_cam = rotation * object_pts[i] + translation;
            if p_cam.z > 1e-10 {
                let u_hat = fx * p_cam.x / p_cam.z + cx;
                let v_hat = fy * p_cam.y / p_cam.z + cy;
                let du = u_hat - image_pts[i].x;
                let dv = v_hat - image_pts[i].y;
                error += du * du + dv * dv;
            } else {
                error += f64::MAX / 4.0; // degenerate: point behind camera
            }
        }
    }

    Ok((Pose3d::new(rotation, translation), error))
}

/// Estimate the 6-DOF pose of a planar quad from 4 coplanar correspondences.
///
/// # Arguments
/// * `object_pts` — 4 coplanar object points in the target frame (`z = 0`). **Order matters for
///   convergence**: pass them counter-clockwise starting from the `(-x, -y)` corner, i.e. the same
///   winding as `[(-s,-s), (s,-s), (s,s), (-s,s)]`. A consistent but differently-ordered
///   correspondence still converges, only more slowly — see the module docs for measured errors at
///   `n_iters = 50`.
/// * `image_pts` — matching 2D image coordinates (pixels), same index order as `object_pts`.
/// * `camera` — pinhole intrinsics. Distortion coefficients are IGNORED; undistort first if your
///   camera has any.
/// * `n_iters` — orthogonal-iteration refinement steps. 50 suits the canonical corner order; other
///   orderings need substantially more. Must be non-zero.
///
/// # Returns
/// [`PlanarPosePair`]: `best` (lower reprojection error) and `second`. Read the module docs before
/// using `second` — it is usually behind the camera rather than a usable alternative.
///
/// # Errors
/// [`PlanarPoseError::Homography`] if the initial homography cannot be fitted, and
/// [`PlanarPoseError::SingularIteration`] if the refinement's linear system degenerates (all four
/// image rays effectively identical).
/// [`PlanarPoseError::ZeroIterations`] if `n_iters == 0`, which would otherwise return an
/// unrefined pose carrying a sentinel error.
///
/// # Example
/// ```
/// use kornia_3d::camera::PinholeCamera;
/// use kornia_3d::pose::estimate_planar_pose;
/// use kornia_algebra::{Vec2F64, Vec3F64};
///
/// // A 10 cm square target, corners counter-clockwise from (-x, -y).
/// let s = 0.05;
/// let object_pts = [
///     Vec3F64::new(-s, -s, 0.0),
///     Vec3F64::new(s, -s, 0.0),
///     Vec3F64::new(s, s, 0.0),
///     Vec3F64::new(-s, s, 0.0),
/// ];
///
/// let camera = PinholeCamera {
///     fx: 500.0,
///     fy: 500.0,
///     cx: 320.0,
///     cy: 240.0,
///     ..PinholeCamera::IDENTITY
/// };
///
/// // Its image with the target frontal, 0.5 m away: u = fx * X / Z + cx.
/// let image_pts = [
///     Vec2F64::new(270.0, 190.0),
///     Vec2F64::new(370.0, 190.0),
///     Vec2F64::new(370.0, 290.0),
///     Vec2F64::new(270.0, 290.0),
/// ];
///
/// let pair = estimate_planar_pose(&object_pts, &image_pts, &camera, 50)?;
/// assert!((pair.best.pose.translation.z - 0.5).abs() < 1e-6);
/// # Ok::<(), kornia_3d::pose::PlanarPoseError>(())
/// ```
///
/// # Note
/// Lu-Hager-Mjolsness (1993) orthogonal iteration, matching the AprilRobotics C reference in
/// `apriltag_pose.c` — except that the reference gates the second solution on positive depth and
/// this port does not.
pub fn estimate_planar_pose(
    object_pts: &[Vec3F64; 4],
    image_pts: &[Vec2F64; 4],
    camera: &PinholeCamera,
    n_iters: usize,
) -> Result<PlanarPosePair, PlanarPoseError> {
    if n_iters == 0 {
        return Err(PlanarPoseError::ZeroIterations);
    }
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
    homography_4pt2d(&tag_norm, &img_arr, &mut h).map_err(PlanarPoseError::Homography)?;

    let init_pose = homography_to_pose(&h, fx, fy, cx, cy);

    // Unnormalized image rays: vᵢ = [(u−cx)/fx, (v−cy)/fy, 1]
    let image_rays: [Vec3F64; 4] = [
        Vec3F64::new((image_pts[0].x - cx) / fx, (image_pts[0].y - cy) / fy, 1.0),
        Vec3F64::new((image_pts[1].x - cx) / fx, (image_pts[1].y - cy) / fy, 1.0),
        Vec3F64::new((image_pts[2].x - cx) / fx, (image_pts[2].y - cy) / fy, 1.0),
        Vec3F64::new((image_pts[3].x - cx) / fx, (image_pts[3].y - cy) / fy, 1.0),
    ];

    // Pose 1: refine from H decomposition
    let (pose1, err1) = orthogonal_iteration(
        object_pts,
        &image_rays,
        image_pts,
        init_pose,
        n_iters,
        camera,
    )?;

    // Pose 2: other planar ambiguity — negate first two R columns
    let r2_init = Mat3F64::from_cols(
        -init_pose.rotation.x_axis(),
        -init_pose.rotation.y_axis(),
        init_pose.rotation.z_axis(),
    );
    let (pose2, err2) = orthogonal_iteration(
        object_pts,
        &image_rays,
        image_pts,
        Pose3d::new(r2_init, init_pose.translation),
        n_iters,
        camera,
    )?;

    let (best, second) = if err1 <= err2 {
        (
            PlanarPose {
                pose: pose1,
                error: err1,
            },
            PlanarPose {
                pose: pose2,
                error: err2,
            },
        )
    } else {
        (
            PlanarPose {
                pose: pose2,
                error: err2,
            },
            PlanarPose {
                pose: pose1,
                error: err1,
            },
        )
    };

    Ok(PlanarPosePair { best, second })
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
    fn test_estimate_planar_pose_roundtrip() -> Result<(), PlanarPoseError> {
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
        let result = estimate_planar_pose(&object_pts, &image_pts, &camera, 50)?;
        let best = &result.best;

        let rot_err = rotation_error_rad(&best.pose.rotation, &r_gt);
        let t_err = (best.pose.translation - t_gt).length();

        assert!(rot_err < 1e-6, "rotation error {rot_err} >= 1e-6 rad");
        assert!(t_err < 1e-4, "translation error {t_err} >= 1e-4 m");
        assert!(
            best.error < 1e-10,
            "reprojection error {} >= 1e-10",
            best.error
        );
        assert!(
            best.pose.transform_point(&object_pts[0]).z > 0.0,
            "cheirality violated"
        );
        Ok(())
    }

    #[test]
    fn test_estimate_planar_pose_noisy() -> Result<(), PlanarPoseError> {
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

        let result = estimate_planar_pose(&object_pts, &image_pts, &camera, 50)?;
        let best = &result.best;

        let rot_err = rotation_error_rad(&best.pose.rotation, &r_gt);
        let t_err = (best.pose.translation - t_gt).length();
        let t_tol = 0.05 * t_gt.length();

        assert!(rot_err < PI / 180.0, "rotation error {rot_err} >= 1°");
        assert!(
            t_err < t_tol,
            "translation error {t_err} >= 5% of depth ({t_tol})"
        );
        Ok(())
    }

    #[test]
    fn test_estimate_planar_pose_degenerate() {
        let camera = test_camera();
        let object_pts = [
            Vec3F64::new(-0.05, -0.05, 0.0),
            Vec3F64::new(0.05, -0.05, 0.0),
            Vec3F64::new(0.05, 0.05, 0.0),
            Vec3F64::new(-0.05, 0.05, 0.0),
        ];
        // All image points at the same pixel → degenerate DLT system
        let image_pts = [Vec2F64::new(320.0, 240.0); 4];
        let result = estimate_planar_pose(&object_pts, &image_pts, &camera, 50);
        assert!(
            matches!(
                result,
                Err(PlanarPoseError::SingularIteration) | Err(PlanarPoseError::Homography(_))
            ),
            "expected a degeneracy error, got {result:?}"
        );
    }

    /// Corner ORDER drives convergence, and `n_iters = 50` only suffices for the canonical order.
    ///
    /// This is the test the module was missing. The original docs claimed "only the index-wise
    /// correspondence matters, not the absolute order" -- measurably false at the `n_iters` every
    /// caller uses. All the other tests here pass the canonical order, which matches the hardcoded
    /// `tag_norm` used to seed the homography, so none of them can observe the effect.
    #[test]
    fn corner_order_changes_convergence_rate() {
        let camera = test_camera();
        let s = 0.05;
        let canonical = [
            Vec3F64::new(-s, -s, 0.0),
            Vec3F64::new(s, -s, 0.0),
            Vec3F64::new(s, s, 0.0),
            Vec3F64::new(-s, s, 0.0),
        ];
        // A tilted pose: at frontal the ordering effect is near-degenerate and proves nothing.
        let (ax, az) = (20.0f64.to_radians(), 15.0f64.to_radians());
        let (cx_, sx) = (ax.cos(), ax.sin());
        let (cz, sz) = (az.cos(), az.sin());
        let rx = Mat3F64::from_cols(
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, cx_, sx),
            Vec3F64::new(0.0, -sx, cx_),
        );
        let rz = Mat3F64::from_cols(
            Vec3F64::new(cz, sz, 0.0),
            Vec3F64::new(-sz, cz, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let r_gt = rz * rx;
        let t_gt = Vec3F64::new(0.0, 0.0, 0.3);

        let project = |p: &Vec3F64| {
            let c = r_gt * *p + t_gt;
            Vec2F64::new(
                camera.fx * c.x / c.z + camera.cx,
                camera.fy * c.y / c.z + camera.cy,
            )
        };
        let rot_err = |pair: &PlanarPosePair| {
            let r = pair.best.pose.rotation;
            let rel = r_gt.transpose() * r;
            let tr = (rel.x_axis.x + rel.y_axis.y + rel.z_axis.z).clamp(-1.0, 3.0);
            (((tr - 1.0) * 0.5).clamp(-1.0, 1.0)).acos().to_degrees()
        };

        // Same correspondence, three orderings of the SAME four points.
        let shifted = [canonical[1], canonical[2], canonical[3], canonical[0]];
        let reversed = [canonical[3], canonical[2], canonical[1], canonical[0]];

        for (name, obj) in [
            ("canonical", canonical),
            ("shifted", shifted),
            ("reversed", reversed),
        ] {
            let img = [
                project(&obj[0]),
                project(&obj[1]),
                project(&obj[2]),
                project(&obj[3]),
            ];
            let at_50 = rot_err(&estimate_planar_pose(&obj, &img, &camera, 50).unwrap());
            let at_2000 = rot_err(&estimate_planar_pose(&obj, &img, &camera, 2000).unwrap());

            // Every ordering converges GIVEN ENOUGH ITERATIONS -- that is the actual contract.
            assert!(
                at_2000 < 1e-3,
                "{name}: should converge with enough iterations, got {at_2000} deg"
            );
            if name == "canonical" {
                assert!(
                    at_50 < 1e-3,
                    "canonical order must converge by n_iters=50, got {at_50}"
                );
            } else {
                // ...but a non-canonical order does NOT, which is what the docs must say.
                assert!(
                    at_50 > 1e-2,
                    "{name}: expected slow convergence at n_iters=50, got {at_50} deg -- if this \
                     now converges, the homography seed was made order-independent and the module \
                     docs must be updated"
                );
            }
        }
    }

    /// `n_iters = 0` must be refused, not answered with an unrefined pose.
    #[test]
    fn zero_iterations_is_rejected() {
        let camera = test_camera();
        let s = 0.05;
        let obj = [
            Vec3F64::new(-s, -s, 0.0),
            Vec3F64::new(s, -s, 0.0),
            Vec3F64::new(s, s, 0.0),
            Vec3F64::new(-s, s, 0.0),
        ];
        let img = [
            Vec2F64::new(270.0, 190.0),
            Vec2F64::new(370.0, 190.0),
            Vec2F64::new(370.0, 290.0),
            Vec2F64::new(270.0, 290.0),
        ];
        assert!(matches!(
            estimate_planar_pose(&obj, &img, &camera, 0),
            Err(PlanarPoseError::ZeroIterations)
        ));
    }
}
