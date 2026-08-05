//! 6-DOF pose of a planar quad from 4 coplanar correspondences, via Lu-Hager-Mjolsness
//! orthogonal iteration (1993).
//!
//! Nothing here is AprilTag-specific -- a tag is one caller among many. Any 4 coplanar object
//! points with known image correspondences work: a fiducial marker, a chessboard cell, a printed
//! rectangle, a door frame.
//!
//! # Initialisation is derived from the caller's own points
//!
//! The orthogonal iteration is seeded by decomposing a homography fitted between `object_pts`
//! and `image_pts`. Because that seed is fitted from the caller's OWN points, it does not depend
//! on the order, the winding, the size or the shape they come in, and on exact data it already IS
//! the answer -- the iteration only has work to do in the presence of noise.
//!
//! The points are centred on their centroid and uniformly rescaled first. That is the standard
//! DLT conditioning step and a similarity of the object plane, so it moves the homography's third
//! column and rescales the first two by a common factor without touching the rotation those two
//! encode; it is conditioning, not correctness.
//!
//! This matters because an earlier revision seeded from a FIXED canonical square
//! `[(-1,-1), (1,-1), (1,1), (-1,1)]`, which was only correct for callers using that exact
//! winding and phase. Measured on exact projections of a square at 0.3 m / 20 deg tilt, that
//! seed left a cyclic shift by one 2.56 deg out and a reversed winding 15.58 deg out at
//! `n_iters = 50` (both converged only by ~500), and left `kornia-apriltag`'s
//! `[TR, BR, BL, TL]` corner order 4.16 deg out on perfect input.
//! `seed_is_independent_of_corner_order` pins the current behaviour: every ordering, a 4:1
//! rectangle, a trapezoid, an off-origin object frame and the same target expressed at 1e-3x and
//! 1e3x scale all land under 1e-6 deg at `n_iters = 50`.
//!
//! # The second solution
//!
//! [`PlanarPosePair::second`] is an `Option`, and it is `None` far more often than not.
//!
//! Two independent orthogonal iterations are run: one from the homography seed, one from that
//! same seed with its first two rotation columns negated (a 180 deg rotation in the object
//! plane). The two results are then sorted by reprojection error, so `best`/`second` name the
//! outcomes, not the branches -- the negation describes the SEED and nothing about the pose that
//! comes back.
//!
//! That second seed is a *heuristic* alternative starting point. It is NOT the classical planar
//! mirror ambiguity, and NOT what the AprilRobotics C reference computes: `apriltag_pose.c`
//! finds the genuine second local minimum of the same objective by solving a quartic
//! (`fix_pose_ambiguities`) and reports `HUGE_VAL` when there is no second minimum. Porting that
//! is tracked separately.
//!
//! Because of that, the second branch is usually not a real hypothesis: across depths of
//! 0.3--5 m, tilts of 0--75 deg and both noise-free and 1 px-noise inputs, it converged BEHIND
//! the camera in every configuration measured. It is now gated on cheirality at the source, so
//! `second` is `Some` only when all four object points lie in front of the camera under it --
//! which, in practice, means callers get `None` and can stop hand-rolling the check.
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
    /// Sum of squared pixel reprojection errors over the 4 correspondences (pixels²).
    /// Lower is better; < 1.0 indicates sub-pixel accuracy.
    pub error: f64,
}

/// The solutions found for one planar-pose problem.
#[derive(Debug, Clone)]
pub struct PlanarPosePair {
    /// Lower-error solution.
    pub best: PlanarPose,
    /// A second local minimum, if the second branch produced a physically possible pose.
    ///
    /// `Some` only when all four object points lie in front of the camera under it; the second
    /// branch converges behind the camera in most configurations, so expect `None`. Read the
    /// module docs before treating this as an alternative hypothesis -- it is a heuristic
    /// re-seed, not the reference implementation's quartic second minimum.
    pub second: Option<PlanarPose>,
}

/// Error type for planar pose estimation.
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

    /// The four object points collapse to a point in their own plane, so there is no frame to
    /// seed the homography from.
    #[error("object points are degenerate (zero extent in the object plane)")]
    DegenerateObjectPoints,

    /// `n_iters` was zero. The refinement loop would never run, so the returned pose would be the
    /// raw homography decomposition carrying a sentinel error rather than a measured one --
    /// silently useless, so it is refused instead.
    #[error("n_iters must be non-zero")]
    ZeroIterations,
}

/// Decompose a planar homography (mapping normalized object-plane coords to pixels) into
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

/// Seed points for the initial homography: the caller's own object x/y, centred on their
/// centroid and uniformly scaled to unit mean radius.
///
/// What makes the seed correct is that it is fitted from the CALLER'S points at all. Writing the
/// exact projection as `H = K·[r₁ r₂ t]` and substituting `q = s·(p − c)` gives
/// `H' = K·[r₁/s r₂/s (t + R·c)]`: the first two columns keep their directions and pick up a
/// COMMON factor, which [`homography_to_pose`] divides out when it normalises them. So the
/// recovered `R` is the caller's frame whatever order, winding, size or shape their points come
/// in, and `t` is rewritten by the iteration's first step anyway.
///
/// Centring and rescaling are therefore conditioning, not correctness — this is the standard DLT
/// normalisation. Measured: disabling either one on its own leaves every case in
/// `seed_is_independent_of_corner_order` passing, so neither is load-bearing for the inputs
/// tested. They are kept because they cost four sums and make the 8x8 system
/// [`homography_4pt2d`] solves independent of the unit the caller's object frame is expressed in.
fn normalized_object_seed(object_pts: &[Vec3F64; 4]) -> Option<[[f64; 2]; 4]> {
    let cx = object_pts.iter().map(|p| p.x).sum::<f64>() * 0.25;
    let cy = object_pts.iter().map(|p| p.y).sum::<f64>() * 0.25;
    let mean_r = object_pts
        .iter()
        .map(|p| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt())
        .sum::<f64>()
        * 0.25;
    if !mean_r.is_finite() || mean_r <= 0.0 {
        return None;
    }
    let s = 1.0 / mean_r;
    Some([0, 1, 2, 3].map(|i| [(object_pts[i].x - cx) * s, (object_pts[i].y - cy) * s]))
}

/// Whether every object point lies strictly in front of the camera under `pose`.
fn all_in_front(pose: &Pose3d, object_pts: &[Vec3F64; 4]) -> bool {
    object_pts
        .iter()
        .all(|p| pose.transform_point(p).z > CHEIRALITY_EPS)
}

/// Minimum camera-frame depth treated as "in front of the camera". Shared by the reprojection
/// error accumulation and the cheirality gate on the second solution so they cannot disagree.
const CHEIRALITY_EPS: f64 = 1e-10;

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
            if p_cam.z > CHEIRALITY_EPS {
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
/// * `object_pts` — 4 coplanar object points in the target's own metric frame (`z = 0`; only x/y
///   are used to seed the iteration). Order, winding and scale are free: the seed is fitted from
///   these points, so only the index-wise correspondence with `image_pts` matters. The recovered
///   pose is expressed in whatever frame they are given in.
/// * `image_pts` — matching 2D image coordinates (pixels), same index order as `object_pts`.
/// * `camera` — pinhole intrinsics. Distortion coefficients are IGNORED; undistort first if your
///   camera has any.
/// * `n_iters` — orthogonal-iteration refinement steps (50 is a good default). Must be non-zero.
///
/// # Returns
/// [`PlanarPosePair`]: `best` (lower reprojection error) and `second`, which is `Some` only when
/// the second branch converged in front of the camera — usually it does not.
///
/// # Errors
/// * [`PlanarPoseError::Homography`] — the initial homography could not be fitted.
/// * [`PlanarPoseError::SingularIteration`] — the refinement's linear system degenerated (all
///   four image rays effectively identical).
/// * [`PlanarPoseError::DegenerateObjectPoints`] — the object points have no extent in their
///   own plane.
/// * [`PlanarPoseError::ZeroIterations`] — `n_iters == 0`, which would otherwise return an
///   unrefined pose carrying a sentinel error.
///
/// # Example
/// ```
/// use kornia_3d::camera::PinholeCamera;
/// use kornia_3d::pose::estimate_planar_pose;
/// use kornia_algebra::{Vec2F64, Vec3F64};
///
/// // A 10 cm square target.
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
/// Lu-Hager-Mjolsness (1993) orthogonal iteration, as in the AprilRobotics C reference
/// `apriltag_pose.c` — except for how the second solution is obtained, see the module docs.
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

    // H maps the caller's own object plane (centred, unit mean radius) → image pixels, so the
    // seed is exact for any ordering/shape/scale rather than only for a canonical square.
    let obj_norm =
        normalized_object_seed(object_pts).ok_or(PlanarPoseError::DegenerateObjectPoints)?;
    let img_arr: [[f64; 2]; 4] = [
        [image_pts[0].x, image_pts[0].y],
        [image_pts[1].x, image_pts[1].y],
        [image_pts[2].x, image_pts[2].y],
        [image_pts[3].x, image_pts[3].y],
    ];
    let mut h = [[0.0f64; 3]; 3];
    homography_4pt2d(&obj_norm, &img_arr, &mut h).map_err(PlanarPoseError::Homography)?;

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

    // Pose 2: re-seed with the first two rotation columns negated (a 180° in-plane rotation) and
    // run an independent iteration. This is a heuristic second start, not the reference's quartic
    // second minimum — see the module docs.
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
        ((pose1, err1), (pose2, err2))
    } else {
        ((pose2, err2), (pose1, err1))
    };

    // Gate the runner-up on cheirality: a pose that puts the target behind the camera is not an
    // alternative hypothesis, and callers were silently consuming it as one.
    let second = all_in_front(&second.0, object_pts).then_some(PlanarPose {
        pose: second.0,
        error: second.1,
    });

    Ok(PlanarPosePair {
        best: PlanarPose {
            pose: best.0,
            error: best.1,
        },
        second,
    })
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

    /// Project `object_pts` from `r_gt` at `depth`, solve, and return `best`'s rotation error in
    /// degrees at `n_iters`.
    fn rot_err_deg_at(
        object_pts: &[Vec3F64; 4],
        camera: &PinholeCamera,
        r_gt: &Mat3F64,
        depth: f64,
        n_iters: usize,
    ) -> f64 {
        let pose_gt = Pose3d::new(*r_gt, Vec3F64::new(0.0, 0.0, depth));
        let image_pts = project_pts(object_pts, &pose_gt, camera);
        let pair = estimate_planar_pose(object_pts, &image_pts, camera, n_iters)
            .expect("exact projection must solve");
        rotation_error_rad(&pair.best.pose.rotation, r_gt).to_degrees()
    }

    /// The seed is fitted from the caller's OWN object points, so ordering, winding, shape and
    /// metric scale must not change what `n_iters = 50` recovers from exact data.
    ///
    /// This is the test the module lacked: every other test here passes the canonical square in
    /// the canonical order, which is exactly the ordering the old hardcoded `tag_norm` seed
    /// happened to match, so none of them could observe the effect. With that seed restored,
    /// `shifted` lands 2.56 deg out and `reversed` 15.58 deg out at 50 iterations.
    #[test]
    fn seed_is_independent_of_corner_order() {
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
        let rx = Mat3F64::from_cols(
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, ax.cos(), ax.sin()),
            Vec3F64::new(0.0, -ax.sin(), ax.cos()),
        );
        let rz = Mat3F64::from_cols(
            Vec3F64::new(az.cos(), az.sin(), 0.0),
            Vec3F64::new(-az.sin(), az.cos(), 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let r_gt = rz * rx;

        // (name, object points, depth). Depth is 0.3 m except for the scale cases, which scale
        // the target AND the distance together so the IMAGE is bit-identical: that isolates the
        // numerical question (does the object frame's unit matter?) from the physical one (a
        // 50 um target at 0.3 m really is unsolvable).
        let cases: [(&str, [Vec3F64; 4], f64); 9] = [
            ("canonical", canonical, 0.3),
            // The AprilTag decoder's [TR, BR, BL, TL] winding — the caller that motivated this.
            (
                "apriltag TR,BR,BL,TL",
                [
                    Vec3F64::new(s, -s, 0.0),
                    Vec3F64::new(s, s, 0.0),
                    Vec3F64::new(-s, s, 0.0),
                    Vec3F64::new(-s, -s, 0.0),
                ],
                0.3,
            ),
            (
                "cyclic shift by one",
                [canonical[1], canonical[2], canonical[3], canonical[0]],
                0.3,
            ),
            (
                "reversed winding",
                [canonical[3], canonical[2], canonical[1], canonical[0]],
                0.3,
            ),
            (
                "4:1 rectangle",
                [
                    Vec3F64::new(-4.0 * s, -s, 0.0),
                    Vec3F64::new(4.0 * s, -s, 0.0),
                    Vec3F64::new(4.0 * s, s, 0.0),
                    Vec3F64::new(-4.0 * s, s, 0.0),
                ],
                0.3,
            ),
            (
                "trapezoid",
                [
                    Vec3F64::new(-3.0 * s, -s, 0.0),
                    Vec3F64::new(3.0 * s, -s, 0.0),
                    Vec3F64::new(s, s, 0.0),
                    Vec3F64::new(-s, s, 0.0),
                ],
                0.3,
            ),
            // Object frame whose origin is nowhere near the target.
            (
                "offset centre",
                canonical.map(|p| p + Vec3F64::new(0.7, -0.3, 0.0)),
                0.3,
            ),
            // Same image, object frame expressed in millimetres and in kilometres.
            ("1e-3x scale", canonical.map(|p| p * 1e-3), 0.3 * 1e-3),
            ("1e3x scale", canonical.map(|p| p * 1e3), 0.3 * 1e3),
        ];

        for (name, obj, depth) in cases {
            let err = rot_err_deg_at(&obj, &camera, &r_gt, depth, 50);
            assert!(
                err < 1e-6,
                "{name}: {err} deg at n_iters = 50 — the homography seed is not \
                 order/shape/scale independent"
            );
        }
    }

    /// The second branch must never be handed back pointing away from the camera.
    ///
    /// Measured across depths 0.3/1/5 m, tilts 0-75 deg and 1 px noise, the second branch landed
    /// behind the camera in every configuration, so this asserts both halves: `None` in a
    /// concrete configuration, and — for every configuration — that a `Some` really is in front.
    #[test]
    fn second_solution_is_gated_on_cheirality() {
        let camera = test_camera();
        let s = 0.05;
        let object_pts = [
            Vec3F64::new(-s, -s, 0.0),
            Vec3F64::new(s, -s, 0.0),
            Vec3F64::new(s, s, 0.0),
            Vec3F64::new(-s, s, 0.0),
        ];

        let mut n_some = 0;
        for tilt_deg in [0.0f64, 5.0, 20.0, 45.0, 60.0, 75.0] {
            let tilt = tilt_deg.to_radians();
            let r_gt = Mat3F64::from_cols(
                Vec3F64::new(1.0, 0.0, 0.0),
                Vec3F64::new(0.0, tilt.cos(), tilt.sin()),
                Vec3F64::new(0.0, -tilt.sin(), tilt.cos()),
            );
            for depth in [0.3f64, 1.0, 5.0] {
                let pose_gt = Pose3d::new(r_gt, Vec3F64::new(0.02, -0.01, depth));
                let image_pts = project_pts(&object_pts, &pose_gt, &camera);
                let pair = estimate_planar_pose(&object_pts, &image_pts, &camera, 50)
                    .expect("exact projection must solve");
                if let Some(second) = &pair.second {
                    n_some += 1;
                    for p in &object_pts {
                        let z = second.pose.transform_point(p).z;
                        assert!(
                            z > 0.0,
                            "tilt {tilt_deg} deg / {depth} m: `second` returned with a point at \
                             z = {z}, behind the camera"
                        );
                    }
                    assert!(
                        second.error.is_finite() && second.error < f64::MAX,
                        "tilt {tilt_deg} deg / {depth} m: `second` carries a sentinel error"
                    );
                }
            }
        }
        // Guard against the loop above passing vacuously in the other direction: if a future
        // change makes the second branch usable everywhere, this fires and the docs must follow.
        assert_eq!(
            n_some, 0,
            "the second branch is documented as always degenerate on these configurations, but \
             {n_some} came back in front of the camera — update the module docs"
        );
    }

    /// `n_iters = 0` must be refused, not answered with an unrefined pose carrying `f64::MAX`.
    #[test]
    fn zero_iterations_is_rejected() {
        let camera = test_camera();
        let s = 0.05;
        let object_pts = [
            Vec3F64::new(-s, -s, 0.0),
            Vec3F64::new(s, -s, 0.0),
            Vec3F64::new(s, s, 0.0),
            Vec3F64::new(-s, s, 0.0),
        ];
        let image_pts = [
            Vec2F64::new(270.0, 190.0),
            Vec2F64::new(370.0, 190.0),
            Vec2F64::new(370.0, 290.0),
            Vec2F64::new(270.0, 290.0),
        ];
        let result = estimate_planar_pose(&object_pts, &image_pts, &camera, 0);
        assert!(
            matches!(result, Err(PlanarPoseError::ZeroIterations)),
            "expected ZeroIterations, got {result:?}"
        );
    }

    /// Object points with no extent give the seed no frame to fit; that must be an error, not a
    /// homography built from NaNs.
    #[test]
    fn degenerate_object_points_are_rejected() {
        let camera = test_camera();
        let object_pts = [Vec3F64::new(0.1, -0.2, 0.0); 4];
        let image_pts = [
            Vec2F64::new(270.0, 190.0),
            Vec2F64::new(370.0, 190.0),
            Vec2F64::new(370.0, 290.0),
            Vec2F64::new(270.0, 290.0),
        ];
        let result = estimate_planar_pose(&object_pts, &image_pts, &camera, 50);
        assert!(
            matches!(result, Err(PlanarPoseError::DegenerateObjectPoints)),
            "expected DegenerateObjectPoints, got {result:?}"
        );
    }
}
