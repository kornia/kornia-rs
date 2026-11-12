//! Projective point-to-plane ICP on RGB-D pyramids.
//!
//! The method is the projective point-to-plane ICP of KinectFusion (Newcombe et al., "KinectFusion:
//! Real-time dense surface mapping and tracking", ISMAR 2011, section 3.5), with the
//! point-to-plane error of Chen and Medioni ("Object modelling by registration of multiple range
//! images", 1992). Each valid source pixel `p_s` is moved by the current estimate
//! `T_target_source = (R, t)`, `p = R p_s + t`, projected into the target image and matched to
//! the target vertex `v_t` and normal `n_t` read at that projection (no kd-tree): the nearest
//! pixel by default, or a bilinear blend, as [`IcpPlaneCriteria::association`] selects. The
//! residual is `r = n_t . (p - v_t)` in metres.
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
//! * Each row's residual sigma comes from a depth noise model, [`IcpPlaneCriteria::depth_noise`],
//!   or a per-pixel sigma map ([`RgbdLevel::depth_sigmas`]), so the Huber threshold is
//!   dimensionless and far, noisy depth weighs less than near depth.
//!
//! The solver runs Gauss-Newton coarse to fine: at each level, starting from the coarsest, it
//! accumulates `A = J^T W J` and `b = J^T W r`, solves `A x = -b` by Cholesky and applies
//! `R <- exp(w) R`, `t <- exp(w) t + v`. A system that does not constrain all six degrees of
//! freedom still gets an answer: directions whose eigenvalue is negligible
//! ([`IcpPlaneCriteria::degeneracy_rtol`]) get a zero update and zero information, and the rest
//! is solved by eigen pseudo-inverse. The `J^T W J` of a final pass at the returned pose is
//! reported as [`IcpPlaneResult::information`].
//!
//! An optional [`IcpPlaneCriteria::rotation_prior`] adds three weighted rows to each step, and
//! an optional [`IcpPlaneCriteria::pose_prior`] six, after the unobserved directions are removed
//! from the data, so they can carry motion the geometry does not observe. They shape the solve
//! only: the conditioning, the degenerate directions and the information describe the data
//! alone.

use super::rgbd::{is_valid_normal, is_valid_vertex, RgbdIcpError, RgbdLevel, RgbdPyramid};
use crate::ransac::{HuberKernel, RobustKernel};
use kornia_algebra::{Mat3F64, Mat6F64, Vec3F64, Vec6F64, SO3F64};
use kornia_image::Image;

/// Smallest residual sigma, in metres, whatever its source (a model or a sigma map). A row
/// weight is `1 / sigma^2`, so a floor near `f64::MIN_POSITIVE` would overflow `J^T W J`; at
/// 1 nm the weight is `1e18`.
const MIN_SIGMA_M: f64 = 1e-9;

/// Lower bound on `|n . ray| / |ray|` in the residual sigma.
///
/// The first-order projection of depth noise onto the plane vanishes at grazing incidence, where
/// the error that remains is the normal's, which this model does not carry; unbounded, those rows
/// would get near-infinite weight. 0.2 is about 78 deg from the normal. Not measured.
const MIN_RAY_INCIDENCE: f64 = 0.2;

/// Relative Cholesky pivot threshold: a squared pivot of `J^T W J` below `PIVOT_RTOL` times the
/// largest diagonal of its block (rotation rows 0-2, translation rows 3-5) skips the Cholesky
/// solve for the eigen one. The blocks are separate because rotation entries grow with depth
/// squared while translation entries are O(1) from unit normals.
const PIVOT_RTOL: f64 = 1e-8;

/// Default of [`IcpPlaneCriteria::degeneracy_rtol`].
///
/// Same value as the Cholesky pivot threshold. Smallest over largest eigenvalue of the
/// depth-scaled `J^T W J` after a 2 cm slide on the synthetic 320x180 test scenes: a bare wall
/// 0 (three directions), a wall with a 6 cm bump 5.1e-8, corner and sphere 2.8e-2. The default
/// flags only an exact null space and still solves the bump. Real sensor noise lifts a wall's
/// null space far above zero; to treat real weak geometry as degenerate, calibrate this on real
/// data.
pub const DEFAULT_DEGENERACY_RTOL: f64 = 1e-8;

/// Fewest inliers that can constrain the six pose parameters.
const MIN_INLIERS: usize = 6;

/// Largest entry of `|R^T R - I|` and of `|det R - 1|` for a prior rotation to count as one.
/// Loose enough for a rotation rounded through `f32`.
const ROTATION_TOL: f64 = 1e-6;

/// Parameters of [`icp_projective_plane`].
///
/// Build it with [`Default`] and set the fields to change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IcpPlaneCriteria {
    /// Gauss-Newton iterations per pyramid level, coarsest first; one entry per level.
    pub iters_per_level: Vec<usize>,
    /// A level stops early once the update rotation `|w|` (radians) and the update translation
    /// `|v|` divided by the mean inlier depth (metres per metre) are both below this.
    pub update_tolerance: f64,
    /// Largest `|p - v_t|` of a kept correspondence, in metres.
    pub max_dist_m: f64,
    /// Largest angle between the rotated source normal and the target normal, in radians.
    pub max_normal_angle_rad: f64,
    /// Huber threshold on the normalised residual `r / sigma`.
    pub huber_delta: f64,
    /// How a projected source point reads its target vertex and normal; default
    /// [`Association::Nearest`].
    pub association: Association,
    /// Depth noise model that sets each row's `sigma`. A frame's per-pixel sigma map
    /// ([`RgbdLevel::depth_sigmas`]) takes precedence over it point by point.
    pub depth_noise: DepthNoise,
    /// Relative eigenvalue threshold below which a direction of `J^T W J` counts as unobserved.
    ///
    /// The eigen-decomposition scales the rotation rows by the inlier depth (radians to metres
    /// of arc), so rotation and translation compare in one unit. A direction whose eigenvalue
    /// is at most `degeneracy_rtol` times the largest gets a zero update and zero information.
    /// Set it between rounding noise on an exact null space and the weakest geometry still to
    /// be solved; see [`DEFAULT_DEGENERACY_RTOL`].
    pub degeneracy_rtol: f64,
    /// Optional external estimate of the rotation, e.g. integrated gyro rates; `None` by default.
    ///
    /// A scene invariant under some rotation, such as a sphere in front of a wall under roll,
    /// leaves that rotation unobserved while the residual stays perfect. The prior enters each
    /// step as three rows weighted `1 / sigma_rad^2`, added after the unobserved directions are
    /// removed from the data: ample geometry outvotes it, weak geometry defers to it, and it
    /// carries the update along directions the data do not observe. It never enters
    /// [`IcpPlaneResult::information`], the conditioning or the degenerate directions.
    ///
    /// The same as a [`PosePrior`] with zero translation information. Both may be set: their
    /// rows add, as two independent measurements would, so a rotation must not reach the solve
    /// through both from one source, or it counts twice.
    pub rotation_prior: Option<RotationPrior>,
    /// Optional external estimate of the whole transform, e.g. the motion predicted by a fused
    /// odometry; `None` by default.
    ///
    /// It enters each step as six rows weighted by the inverse of its covariance, added after
    /// the unobserved directions are removed from the data, like [`Self::rotation_prior`]: it
    /// shapes the answer where the data are weak and carries the update along directions they
    /// do not observe, such as the slide along a bare wall. It never enters
    /// [`IcpPlaneResult::information`], the conditioning or the degenerate directions.
    pub pose_prior: Option<PosePrior>,
}

impl Default for IcpPlaneCriteria {
    /// Three levels; `huber_delta` times the constant 1 cm sigma puts the Huber knee at 2 cm.
    fn default() -> Self {
        Self {
            iters_per_level: vec![10, 7, 5],
            update_tolerance: 1e-6,
            max_dist_m: 0.1,
            max_normal_angle_rad: 30.0_f64.to_radians(),
            huber_delta: 2.0,
            association: Association::Nearest,
            depth_noise: DepthNoise::Constant { sigma_m: 0.01 },
            degeneracy_rtol: DEFAULT_DEGENERACY_RTOL,
            rotation_prior: None,
            pose_prior: None,
        }
    }
}

/// How [`icp_projective_plane`] reads the target vertex and normal at a source point's
/// projection, for [`IcpPlaneCriteria::association`].
///
/// Bilinear sampling removes the snapping bias of the nearest pixel but smooths the target,
/// which far from the solution biases the coarse alignment; at the converged finest level it
/// only removes the snapping. [`Self::Hybrid`] uses each where it helps. Noise-free error of
/// the solve on the moving corner + sphere pair of the NEES test: nearest 1.75e-4 rad and
/// 0.30 mm, bilinear and hybrid 2.1e-5 rad and 0.07 mm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Association {
    /// The pixel the projection rounds to: biased by up to half a pixel of snapping, with the
    /// widest convergence range.
    #[default]
    Nearest,
    /// Bilinear blend of the four pixels around the projection, normal renormalised. A pixel
    /// with zero weight is not read; the sample is dropped when a weighted pixel is invalid,
    /// when they straddle a depth discontinuity ([`MAX_BILINEAR_DEPTH_JUMP`]) or when their
    /// normals nearly cancel.
    Bilinear,
    /// Nearest on the coarse levels and on the finest until an update falls below
    /// [`HYBRID_SWITCH_NORM`], then bilinear on the finest for up to as many iterations again.
    Hybrid,
}

/// Update size under which [`Association::Hybrid`] switches the finest level to bilinear, in
/// the units of [`IcpPlaneCriteria::update_tolerance`]: rotation in radians and translation
/// over the mean inlier depth.
///
/// 1e-3 is 0.06 deg and 2 mm at 2 m: well inside the half-pixel snapping it is there to remove
/// (~5 mm at 2 m and fx = 200) and well above the default 1e-6 tolerance, so the switch comes
/// before a level is declared converged. Not tuned.
pub const HYBRID_SWITCH_NORM: f64 = 1e-3;

/// Largest relative depth spread `(z_max - z_min) / z_min` across the weighted pixels of a
/// bilinear sample; beyond it they straddle a depth discontinuity and the sample is dropped.
///
/// A surface 80 deg from fronto-parallel spreads ~3% across one pixel at fx = 200, so 5% keeps
/// steep surfaces and still drops a foreground/background pair (a 10 cm step at 2 m). Chosen on
/// that geometry, not tuned on data.
pub const MAX_BILINEAR_DEPTH_JUMP: f64 = 0.05;

/// An external rotation estimate for [`IcpPlaneCriteria::rotation_prior`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotationPrior {
    /// Prior on the rotation of `T_target_source`, the convention of
    /// [`IcpPlaneResult::rotation`]. Must be a rotation matrix.
    pub rotation: Mat3F64,
    /// One-sigma error of [`Self::rotation`] per axis, in radians, over the interval it spans.
    ///
    /// Must be finite and positive. Scale it with the elapsed time rather than passing a
    /// per-frame constant: a prior integrated over a longer interval has drifted further. Size it
    /// against the sensor's total error, systematic terms such as a camera-to-gyro misalignment
    /// included: a tight sigma on a biased rotation pulls the solve toward the bias.
    pub sigma_rad: f64,
}

/// An external estimate of the whole transform for [`IcpPlaneCriteria::pose_prior`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PosePrior {
    /// Prior on the rotation of `T_target_source`, the convention of
    /// [`IcpPlaneResult::rotation`]. Must be a rotation matrix.
    pub rotation: Mat3F64,
    /// Prior on the translation of `T_target_source`, in metres. Must be finite.
    pub translation: Vec3F64,
    /// Covariance of the prior's error `xi = [w, v]`, rotation first (rad^2, rad m, m^2), with
    /// the true transform `exp(xi)` applied on the left of the prior: the tangent of the solver
    /// and of [`IcpPlaneResult::information`].
    ///
    /// Must be finite, symmetric and positive definite, with an inverse that does not overflow.
    /// Scale it with the interval the prior spans, as for [`RotationPrior::sigma_rad`].
    pub covariance: Mat6F64,
}

/// Depth noise model for the row weights of [`icp_projective_plane`].
///
/// Each point's depth sigma moves it along its viewing ray, so a row's residual sigma is
/// `sqrt((n . r_s)^2 sigma_s^2 + (n . r_t)^2 sigma_t^2)`, with `r = p / z` the ray of the source
/// point (rotated into the target) and of the target point, and `n` the target normal.
/// [`Self::Constant`] with no sigma map on either frame is the exception: a residual sigma used
/// as-is, the same for every row.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DepthNoise {
    /// One sigma for every row, in metres of point-to-plane residual. Where a sigma map is
    /// present on either frame, a point without a map value uses it as its depth sigma instead.
    Constant {
        /// One-sigma error in metres.
        sigma_m: f64,
    },
    /// Stereo depth: `sigma_z = z^2 * sigma_disparity_px / (focal_px * baseline_m)`.
    Stereo {
        /// Focal length in pixels at the finest level, the one the disparity was measured at.
        focal_px: f64,
        /// Stereo baseline in metres.
        baseline_m: f64,
        /// One-sigma disparity error in pixels.
        sigma_disparity_px: f64,
    },
}

impl DepthNoise {
    /// One-sigma depth error in metres at depth `z_m`.
    ///
    /// # Arguments
    ///
    /// * `z_m` - Z-depth in metres.
    ///
    /// # Returns
    ///
    /// The model's sigma; for [`Self::Constant`] the constant itself.
    pub fn depth_sigma(&self, z_m: f64) -> f64 {
        match *self {
            DepthNoise::Constant { sigma_m } => sigma_m,
            DepthNoise::Stereo {
                focal_px,
                baseline_m,
                sigma_disparity_px,
            } => z_m * z_m * sigma_disparity_px / (focal_px * baseline_m),
        }
    }

    /// The model's parameters with their field names, for validation.
    fn params(&self) -> [(&'static str, f64); 3] {
        match *self {
            // repeated so that every arm has three entries
            DepthNoise::Constant { sigma_m } => [("depth_noise.sigma_m", sigma_m); 3],
            DepthNoise::Stereo {
                focal_px,
                baseline_m,
                sigma_disparity_px,
            } => [
                ("depth_noise.focal_px", focal_px),
                ("depth_noise.baseline_m", baseline_m),
                ("depth_noise.sigma_disparity_px", sigma_disparity_px),
            ],
        }
    }
}

/// Result of [`icp_projective_plane`]: the estimate of `T_target_source`, which maps source
/// points into the target frame, `p_tgt = rotation * p_src + translation`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IcpPlaneResult {
    /// Rotation from the source to the target frame.
    pub rotation: Mat3F64,
    /// Translation from the source to the target frame, in metres.
    pub translation: Vec3F64,
    /// Root-mean-square point-to-plane residual of the inliers, in metres, from a pass at the
    /// finest level at the returned pose.
    pub rmse: f64,
    /// Gauss-Newton iterations run over all levels.
    pub iterations: usize,
    /// Correspondences that passed the gates in that same final pass.
    pub num_inliers: usize,
    /// Source pixels of the final pass that projected onto valid target geometry, before the
    /// distance and normal gates: how much evidence the solve had.
    pub num_associated: usize,
    /// [`Self::num_inliers`] over [`Self::num_associated`]: match quality alone.
    ///
    /// Not divided by all valid source pixels, which would mix quality with view overlap and
    /// fall under camera motion on solves that converged. Read it with
    /// [`Self::overlap_fraction`]: a small patch can match perfectly and still be too little to
    /// constrain a pose.
    pub inlier_fraction: f64,
    /// [`Self::num_associated`] over the valid source pixels: how much of the source view still
    /// lands on the target. It falls as the camera leaves the target view, which calls for a
    /// new target frame rather than a rejected solve.
    pub overlap_fraction: f64,
    /// Conditioning of the rotation block of `J^T W J` at the returned pose: the weakest
    /// Cholesky pivot of rows 0-2 over the strongest, in `0..=1`; `0` when the final-pass
    /// matrix does not factor.
    ///
    /// A ratio rather than a pivot magnitude, because pivots grow with the number of
    /// correspondences and the scene depth while the ratio depends on the geometry only.
    pub rotation_conditioning: f64,
    /// Conditioning of the translation block (rows 3-5), defined as
    /// [`Self::rotation_conditioning`].
    ///
    /// It is the one that falls when a single surface fills the view: translation along it is
    /// barely constrained while the residual and [`Self::inlier_fraction`] stay perfect.
    pub translation_conditioning: f64,
    /// The weaker of [`Self::rotation_conditioning`] and [`Self::translation_conditioning`]:
    /// one scalar to gate a pose on. Either block can collapse first, so gate on this rather
    /// than on one block.
    pub observability: f64,
    /// Median target depth `v_t.z` of the final pass's inliers, in metres.
    pub median_depth_m: f64,
    /// Information matrix `J^T W J` of the final pass at the returned pose, at the finest level.
    ///
    /// Tangent order `[w, v]`: rotation in radians, then translation in metres, for the left
    /// perturbation the solver applies, `R <- exp(w) R`, `t <- exp(w) t + v`, so both are
    /// expressed in the target frame. Rows are weighted `huber(r / sigma) / sigma^2` as in the
    /// solve.
    ///
    /// It holds the data term only; no prior enters it. Under the depth noise model
    /// ([`IcpPlaneCriteria::depth_noise`] or the sigma maps) its inverse is the covariance of
    /// the estimate's error `xi`, the estimate being `exp(xi)` applied on the left of the truth.
    /// It is only as calibrated as those sigmas: the default [`DepthNoise::Constant`] is a
    /// weighting, not a sensor model. It is zero along [`Self::degenerate_directions`].
    pub information: Mat6F64,
    /// Unit tangent directions `[w, v]` the data do not observe, in entries
    /// `0..num_degenerate`; the other entries are zero.
    ///
    /// They span the null space of [`Self::information`] and got no update. They come from the
    /// eigen-decomposition with rotation scaled by [`Self::median_depth_m`] and are mapped back
    /// to the unscaled tangent, so they are unit but not necessarily orthogonal.
    pub degenerate_directions: [Vec6F64; 6],
    /// Number of directions in [`Self::degenerate_directions`]: `0` when the data constrain all
    /// six degrees of freedom, `3` for a single plane.
    pub num_degenerate: usize,
}

/// Projective point-to-plane ICP between two RGB-D pyramids, coarse to fine.
///
/// # Arguments
///
/// * `src` - Source frame pyramid.
/// * `tgt` - Target frame pyramid, with as many levels as `src`.
/// * `init_rotation` - Initial rotation of `T_target_source`.
/// * `init_translation` - Initial translation of `T_target_source`, in metres.
/// * `criteria` - Gates, weighting and iteration budget.
///
/// # Returns
///
/// The estimate of `T_target_source` with the residual, match quality, overlap, conditioning
/// and information matrix of a final pass at that pose. Geometry that does not constrain all six
/// degrees of freedom is not an error: the unobserved directions get no update and are reported
/// in [`IcpPlaneResult::degenerate_directions`].
///
/// # Errors
///
/// * [`RgbdIcpError::PyramidLevelsMismatch`] if the pyramids have different level counts.
/// * [`RgbdIcpError::IterationsLevelsMismatch`] if `criteria.iters_per_level` does not have one
///   entry per level.
/// * [`RgbdIcpError::InvalidCriteria`] if a threshold of `criteria` (`degeneracy_rtol`
///   included) or a parameter of its `depth_noise` is not finite and positive, if
///   `max_normal_angle_rad` exceeds pi, or if its `rotation_prior` has a `sigma_rad` that is not
///   finite and positive (or so small that its weight overflows) or a `rotation` that is not a
///   finite rotation matrix, or if its `pose_prior` has such a `rotation`, a `translation` that
///   is not finite, or a `covariance` that is not finite, symmetric and positive definite with a
///   finite inverse.
/// * [`RgbdIcpError::TooFewCorrespondences`] if fewer than 6 correspondences pass the gates in
///   any pass, the final one included (low overlap or a bad initial guess).
/// * [`RgbdIcpError::SingularNormalEquations`] if the normal equations or their solve are not
///   finite.
///
/// # Example
///
/// ```
/// use kornia_3d::camera::PinholeCamera;
/// use kornia_3d::registration::{icp_projective_plane, IcpPlaneCriteria, RgbdPyramid};
/// use kornia_algebra::{Mat3F64, Vec3F64};
/// use kornia_image::{Image, ImageSize};
///
/// // a corrugated surface: its normals vary along both image axes
/// let size = ImageSize { width: 64, height: 48 };
/// let mut depth = Image::<u16, 1>::from_size_val(size, 0)?;
/// for (idx, d) in depth.as_slice_mut().iter_mut().enumerate() {
///     let (u, v) = ((idx % 64) as f64, (idx / 64) as f64);
///     *d = (1200.0 + 100.0 * (u / 6.0).sin() + 80.0 * (v / 5.0).cos()) as u16;
/// }
/// let camera = PinholeCamera { fx: 50.0, fy: 50.0, cx: 31.5, cy: 23.5, ..PinholeCamera::IDENTITY };
/// let pyramid = RgbdPyramid::from_depth_mm(&depth, &camera, 2)?;
///
/// let mut criteria = IcpPlaneCriteria::default();
/// criteria.iters_per_level = vec![5, 5];
/// let result =
///     icp_projective_plane(&pyramid, &pyramid, &Mat3F64::IDENTITY, &Vec3F64::ZERO, &criteria)?;
/// assert!(result.translation.length() < 1e-6);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn icp_projective_plane(
    src: &RgbdPyramid,
    tgt: &RgbdPyramid,
    init_rotation: &Mat3F64,
    init_translation: &Vec3F64,
    criteria: &IcpPlaneCriteria,
) -> Result<IcpPlaneResult, RgbdIcpError> {
    icp_projective_plane_with(
        &mut IcpWorkspace::default(),
        src,
        tgt,
        init_rotation,
        init_translation,
        criteria,
    )
}

/// Reusable buffers for [`icp_projective_plane_with`].
///
/// Each buffer is reserved by the image size, not by the number of valid pixels or inliers, so
/// after one call at a given source size a call allocates nothing, however full or sparse the
/// frames that follow are.
#[derive(Debug, Default, Clone)]
pub struct IcpWorkspace {
    /// Target depths of the final pass's inliers, for their median.
    depths: Vec<f32>,
}

/// [`icp_projective_plane`] with reused buffers, for a loop over frames.
///
/// The same result as [`icp_projective_plane`], bit for bit; the workspace holds no state that
/// carries from one call to the next.
///
/// # Arguments
///
/// * `workspace` - Buffers reused across calls; grows on the first call at a larger source size.
/// * `src` - Source frame pyramid.
/// * `tgt` - Target frame pyramid, with as many levels as `src`.
/// * `init_rotation` - Initial rotation of `T_target_source`.
/// * `init_translation` - Initial translation of `T_target_source`, in metres.
/// * `criteria` - Gates, weighting and iteration budget.
///
/// # Returns
///
/// As [`icp_projective_plane`].
///
/// # Errors
///
/// As [`icp_projective_plane`].
///
/// # Example
///
/// ```
/// use kornia_3d::camera::PinholeCamera;
/// use kornia_3d::registration::{
///     icp_projective_plane_with, IcpPlaneCriteria, IcpWorkspace, RgbdPyramid,
/// };
/// use kornia_algebra::{Mat3F64, Vec3F64};
/// use kornia_image::{Image, ImageSize};
///
/// let size = ImageSize { width: 64, height: 48 };
/// let mut depth = Image::<u16, 1>::from_size_val(size, 0)?;
/// for (idx, d) in depth.as_slice_mut().iter_mut().enumerate() {
///     let (u, v) = ((idx % 64) as f64, (idx / 64) as f64);
///     *d = (1200.0 + 100.0 * (u / 6.0).sin() + 80.0 * (v / 5.0).cos()) as u16;
/// }
/// let camera = PinholeCamera { fx: 50.0, fy: 50.0, cx: 31.5, cy: 23.5, ..PinholeCamera::IDENTITY };
/// let pyramid = RgbdPyramid::from_depth_mm(&depth, &camera, 2)?;
///
/// let mut criteria = IcpPlaneCriteria::default();
/// criteria.iters_per_level = vec![5, 5];
/// let mut workspace = IcpWorkspace::default();
/// for _ in 0..2 {
///     let (r, t) = (Mat3F64::IDENTITY, Vec3F64::ZERO);
///     let result = icp_projective_plane_with(&mut workspace, &pyramid, &pyramid, &r, &t, &criteria)?;
///     assert!(result.translation.length() < 1e-6);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn icp_projective_plane_with(
    workspace: &mut IcpWorkspace,
    src: &RgbdPyramid,
    tgt: &RgbdPyramid,
    init_rotation: &Mat3F64,
    init_translation: &Vec3F64,
    criteria: &IcpPlaneCriteria,
) -> Result<IcpPlaneResult, RgbdIcpError> {
    let num_levels = src.levels.len();
    if tgt.levels.len() != num_levels {
        return Err(RgbdIcpError::PyramidLevelsMismatch {
            src_levels: num_levels,
            tgt_levels: tgt.levels.len(),
        });
    }
    if criteria.iters_per_level.len() != num_levels {
        return Err(RgbdIcpError::IterationsLevelsMismatch {
            iters: criteria.iters_per_level.len(),
            levels: num_levels,
        });
    }
    run_icp(
        workspace,
        src,
        tgt,
        init_rotation,
        init_translation,
        criteria,
    )
}

/// The gates and prior rows of `criteria`, or the first threshold out of its range.
fn check_criteria(
    criteria: &IcpPlaneCriteria,
) -> Result<(PassGates, [Option<PriorRows>; 2]), RgbdIcpError> {
    let inf = f64::INFINITY;
    let noise = criteria.depth_noise.params().map(|(f, v)| (f, v, inf));
    for (field, value, max) in [
        ("max_dist_m", criteria.max_dist_m, inf),
        ("huber_delta", criteria.huber_delta, inf),
        ("update_tolerance", criteria.update_tolerance, inf),
        ("degeneracy_rtol", criteria.degeneracy_rtol, inf),
        // above pi the cosine gate wraps round and starts rejecting again
        (
            "max_normal_angle_rad",
            criteria.max_normal_angle_rad,
            std::f64::consts::PI,
        ),
    ]
    .into_iter()
    .chain(noise)
    {
        // written so that NaN fails
        if !(value > 0.0 && value.is_finite() && value <= max) {
            return Err(RgbdIcpError::InvalidCriteria { field, value });
        }
    }
    if let Some(prior) = &criteria.rotation_prior {
        check_rotation_prior(prior)?;
    }
    let priors = [
        criteria.rotation_prior.map(PriorRows::from_rotation),
        criteria
            .pose_prior
            .as_ref()
            .map(PriorRows::from_pose)
            .transpose()?,
    ];
    let gates = PassGates {
        // Hybrid starts on nearest; `run_icp` switches it
        bilinear: criteria.association == Association::Bilinear,
        max_dist_m: criteria.max_dist_m,
        cos_max_normal_angle: criteria.max_normal_angle_rad.cos(),
        huber_delta: criteria.huber_delta,
        depth_noise: criteria.depth_noise,
    };
    Ok((gates, priors))
}

/// `Ok` if `prior` has a finite positive sigma with a finite weight and a finite rotation matrix.
fn check_rotation_prior(prior: &RotationPrior) -> Result<(), RgbdIcpError> {
    let sigma = prior.sigma_rad;
    // written so that NaN fails; a sigma below ~1e-154 would make the weight infinite
    if !(sigma > 0.0 && sigma.is_finite() && (1.0 / (sigma * sigma)).is_finite()) {
        return Err(RgbdIcpError::InvalidCriteria {
            field: "rotation_prior.sigma_rad",
            value: sigma,
        });
    }
    check_rotation(&prior.rotation, "rotation_prior.rotation")
}

/// `Ok` if `r` is a finite rotation matrix; `field` names it in the error.
fn check_rotation(r: &Mat3F64, field: &'static str) -> Result<(), RgbdIcpError> {
    let orth = (r.transpose() * *r - Mat3F64::IDENTITY).to_cols_array();
    // `f64::max` drops NaN, hence the separate finiteness check
    let deviation = orth
        .iter()
        .fold((r.determinant() - 1.0).abs(), |m, x| m.max(x.abs()));
    if !(r.is_finite() && deviation <= ROTATION_TOL) {
        return Err(RgbdIcpError::InvalidCriteria {
            field,
            value: deviation,
        });
    }
    Ok(())
}

/// Gates and weighting of one association pass.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PassGates {
    /// Bilinear target sampling ([`sample_target`]) instead of the nearest pixel.
    pub bilinear: bool,
    /// Largest `|p - v_t|` of a kept correspondence, in metres.
    pub max_dist_m: f64,
    /// Smallest `(R n_s) . n_t` of a kept correspondence (cosine of the largest normal angle).
    pub cos_max_normal_angle: f64,
    /// Huber threshold on the normalised residual `r / sigma`.
    pub huber_delta: f64,
    /// Depth noise model behind each residual's sigma.
    pub depth_noise: DepthNoise,
}

/// One gated correspondence of a projective-association pass.
pub(crate) struct Correspondence {
    /// Index of the source pixel in its level.
    // read by the finite-difference test only, until per-pixel weights need it
    #[cfg_attr(not(test), allow(dead_code))]
    pub src_idx: usize,
    /// Target vertex the residual is measured against.
    pub v_t: Vec3F64,
    /// Unit target normal.
    // read by the finite-difference test only; the solve needs only `j`
    #[cfg_attr(not(test), allow(dead_code))]
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
    /// Sum of the inliers' target depths `v_t.z`, in metres: scales the stopping test.
    pub sum_depth_m: f64,
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

/// A valid sigma-map entry of pixel `idx` of `level`, or `None`: no map, or no value there.
#[inline]
fn map_sigma(level: &RgbdLevel, idx: usize) -> Option<f64> {
    let s = *level.depth_sigmas.as_ref()?.as_slice().get(idx)?;
    // written so that NaN fails
    (s > 0.0 && s.is_finite()).then_some(s as f64)
}

/// Depth sigma of one point: its level's map entry when present, else the model.
#[inline]
fn point_depth_sigma(level: &RgbdLevel, idx: usize, z: f64, noise: &DepthNoise) -> f64 {
    map_sigma(level, idx).unwrap_or_else(|| noise.depth_sigma(z))
}

/// Squared cosine between a normal and a ray, bounded below by [`MIN_RAY_INCIDENCE`], times
/// the ray's squared length.
#[inline]
fn projected_ray_sq(n: &Vec3F64, ray: &Vec3F64) -> f64 {
    let c = n.dot(*ray);
    (c * c).max(MIN_RAY_INCIDENCE * MIN_RAY_INCIDENCE * ray.dot(*ray))
}

/// A target vertex and normal.
struct TargetSample {
    vertex: Vec3F64,
    /// Unit length.
    normal: Vec3F64,
    /// The target's sigma-map value at the sample, when it has one.
    depth_sigma: Option<f64>,
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
        depth_sigma: map_sigma(tgt, idx),
    })
}

/// Bilinear sample of the target vertex and normal maps (and sigma map) at continuous pixel
/// `(u, v)`.
///
/// Nearest-pixel snapping leaves the matched vertex up to half a pixel off the ray, a residual
/// that on a curved or slanted surface does not average out; interpolating removes it to first
/// order. A tap with zero weight (integer `u` or `v`, the last row or column) is neither read
/// nor tested, so it cannot drop the sample. The right and lower neighbours are clamped into
/// the map, which keeps a 1-pixel-wide or -high level in bounds: there the clamped tap has zero
/// weight. `None` off the map (NaN and infinite coordinates included), when a weighted pixel is
/// invalid, when the weighted pixels straddle a depth discontinuity
/// ([`MAX_BILINEAR_DEPTH_JUMP`]) or when their normals nearly cancel. The sigma is interpolated
/// only when every weighted pixel has a map value.
fn sample_target(tgt: &RgbdLevel, u: f64, v: f64) -> Option<TargetSample> {
    let size = tgt.size();
    let (tw, th) = (size.width, size.height);
    // written so that NaN and infinity fail; a 0-pixel map fails too
    if !(u >= 0.0 && v >= 0.0 && u <= tw as f64 - 1.0 && v <= th as f64 - 1.0) {
        return None;
    }
    // u, v >= 0 here, so truncation is floor (and no libm call)
    let (u0, v0) = (u as usize, v as usize);
    let (fu, fv) = (u - u0 as f64, v - v0 as f64);
    let (u1, v1) = ((u0 + 1).min(tw - 1), (v0 + 1).min(th - 1));
    let taps = [
        (v0 * tw + u0, (1.0 - fu) * (1.0 - fv)),
        (v0 * tw + u1, fu * (1.0 - fv)),
        (v1 * tw + u0, (1.0 - fu) * fv),
        (v1 * tw + u1, fu * fv),
    ];
    let (mut z_min, mut z_max) = (f64::INFINITY, 0.0f64);
    let (mut vertex, mut normal) = (Vec3F64::ZERO, Vec3F64::ZERO);
    let mut sigma = tgt.depth_sigmas.as_ref().map(|_| 0.0);
    for (idx, w) in taps {
        if w == 0.0 {
            continue;
        }
        let (vt, nt) = (pixel3(&tgt.vertices, idx), pixel3(&tgt.normals, idx));
        if !is_valid_vertex(&vt) || !is_valid_normal(&nt) {
            return None;
        }
        let (vt, nt) = (vec3(vt), vec3(nt));
        z_min = z_min.min(vt.z);
        z_max = z_max.max(vt.z);
        vertex += vt * w;
        normal += nt * w;
        sigma = sigma.zip(map_sigma(tgt, idx)).map(|(acc, s)| acc + w * s);
    }
    let n = normal.length();
    // written so that NaN fails
    if !(z_max - z_min <= MAX_BILINEAR_DEPTH_JUMP * z_min && n >= 0.5) {
        return None;
    }
    Some(TargetSample {
        vertex,
        normal: normal / n,
        depth_sigma: sigma,
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
    let noise = &gates.depth_noise;
    // constant noise with no map on either frame is a residual sigma, the same for every row
    let constant_sigma = match *noise {
        DepthNoise::Constant { sigma_m }
            if src.depth_sigmas.is_none() && tgt.depth_sigmas.is_none() =>
        {
            Some(sigma_m.max(MIN_SIGMA_M))
        }
        _ => None,
    };
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
        let sample = if gates.bilinear {
            sample_target(tgt, uv.x, uv.y)
        } else {
            sample_nearest(tgt, uv.x, uv.y)
        };
        let Some(TargetSample {
            vertex: v_t,
            normal: n_t,
            depth_sigma: tgt_sigma,
        }) = sample
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

        let sigma = constant_sigma.unwrap_or_else(|| {
            // depth noise moves each point along its own ray `p / z`; project both onto n_t
            let z_s = v_s[2] as f64;
            let ray_s = (p - *translation) / z_s;
            let ray_t = v_t / v_t.z;
            let sig_s = point_depth_sigma(src, src_idx, z_s, noise);
            let sig_t = tgt_sigma.unwrap_or_else(|| noise.depth_sigma(v_t.z));
            let var = projected_ray_sq(&n_t, &ray_s) * sig_s * sig_s
                + projected_ray_sq(&n_t, &ray_t) * sig_t * sig_t;
            // the one floor for every source of sigma, sigma maps included
            var.sqrt().max(MIN_SIGMA_M)
        });
        // the interpolated target is held fixed within an iteration, as the nearest one is;
        // see the re-associating half of `information_matches_finite_difference_hessian`
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
/// `f64`. With `depths`, also appends each inlier's target depth `v_t.z` to it.
pub(crate) fn accumulate_level(
    src: &RgbdLevel,
    tgt: &RgbdLevel,
    rotation: &Mat3F64,
    translation: &Vec3F64,
    gates: &PassGates,
    mut depths: Option<&mut Vec<f32>>,
) -> NormalEquations {
    let mut a = Mat6F64::zeros();
    let mut b = Vec6F64::zeros();
    let (mut num_inliers, mut sum_sq_residual, mut sum_depth_m) = (0usize, 0.0f64, 0.0f64);
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
        sum_depth_m += c.v_t.z;
        if let Some(d) = depths.as_deref_mut() {
            // widened from an f32 vertex, so the cast back is exact
            d.push(c.v_t.z as f32);
        }
    });
    a.fill_lower_triangle_with_upper_triangle();

    NormalEquations {
        a,
        b,
        num_inliers,
        counts,
        sum_sq_residual,
        sum_depth_m,
    }
}

/// `Ok` if a pass kept enough inliers to constrain a pose.
fn check_inliers(eqs: &NormalEquations) -> Result<(), RgbdIcpError> {
    if eqs.num_inliers < MIN_INLIERS {
        return Err(RgbdIcpError::TooFewCorrespondences(eqs.num_inliers));
    }
    Ok(())
}

/// Solves `A x = -b` by Cholesky, or `None` on a collapsed pivot (see [`PIVOT_RTOL`]).
fn cholesky_step(a: &Mat6F64, b: &Vec6F64) -> Option<Vec6F64> {
    let chol = a.cholesky()?;
    // the pivots of the factorisation are the squared diagonal of L
    let l = chol.l_dirty();
    let block_ok = |rows: std::ops::Range<usize>| {
        let max_diag = rows.clone().map(|i| a[(i, i)]).fold(0.0, f64::max);
        // `>=` so that a NaN pivot fails
        max_diag > 0.0
            && rows
                .into_iter()
                .all(|i| l[(i, i)] * l[(i, i)] >= PIVOT_RTOL * max_diag)
    };
    (block_ok(0..3) && block_ok(3..6)).then(|| chol.solve(&(-b)))
}

/// Prior rows in information form, at the estimate `(R, t)` of `T_target_source`.
///
/// The error is `e = [log(R R_p^T), t - R R_p^T t_p]`, weighted by `info`. Under the solver's
/// left update `R <- exp(w) R`, `t <- exp(w) t + v` it becomes
/// `[log(exp(w) exp(e_r)), exp(w) e_t + v]`, so its Jacobian at `w = v = 0` is
/// `[[Jl(e_r)^-1, 0], [-[e_t]x, I]]`, with `Jl` the left Jacobian of SO(3). A rotation prior is
/// the case `t_p = 0` with zero translation information.
#[derive(Debug, Clone, Copy)]
struct PriorRows {
    rotation: Mat3F64,
    translation: Vec3F64,
    info: Mat6F64,
}

impl PriorRows {
    /// Rows of a rotation prior already checked by [`check_rotation_prior`].
    fn from_rotation(p: RotationPrior) -> Self {
        let w = 1.0 / (p.sigma_rad * p.sigma_rad);
        Self {
            rotation: p.rotation,
            translation: Vec3F64::ZERO,
            info: Mat6F64::from_diagonal(&Vec6F64::new(w, w, w, 0.0, 0.0, 0.0)),
        }
    }

    /// Rows of a pose prior, whose information is the inverse of its covariance.
    ///
    /// # Errors
    ///
    /// [`RgbdIcpError::InvalidCriteria`] if the rotation is not a finite rotation matrix, the
    /// translation is not finite, or the covariance is not finite, symmetric and positive
    /// definite with a finite inverse.
    fn from_pose(p: &PosePrior) -> Result<Self, RgbdIcpError> {
        check_rotation(&p.rotation, "pose_prior.rotation")?;
        if !p.translation.to_array().iter().all(|v| v.is_finite()) {
            return Err(RgbdIcpError::InvalidCriteria {
                field: "pose_prior.translation",
                value: p.translation.length(),
            });
        }
        let c = p.covariance;
        let field = "pose_prior.covariance";
        if let Some(&value) = c.iter().find(|v| !v.is_finite()) {
            return Err(RgbdIcpError::InvalidCriteria { field, value });
        }
        // relative, so that a covariance in any unit passes; the entries are finite here, so
        // `>` cannot let a NaN through
        let asymmetry = (c - c.transpose()).amax();
        if asymmetry > 1e-9 * c.amax() {
            return Err(RgbdIcpError::InvalidCriteria {
                field,
                value: asymmetry,
            });
        }
        let info = c.cholesky().map(|l| l.inverse());
        match info {
            // a near-singular covariance can factor and still overflow its inverse
            Some(info) if info.iter().all(|v| v.is_finite()) => Ok(Self {
                rotation: p.rotation,
                translation: p.translation,
                // symmetrise away the rounding of the inverse
                info: (info + info.transpose()) * 0.5,
            }),
            // the smallest eigenvalue says how far from positive definite it is
            _ => Err(RgbdIcpError::InvalidCriteria {
                field,
                value: c.symmetric_eigen().eigenvalues.min(),
            }),
        }
    }

    /// The error `e` and its Jacobian at `(rotation, translation)`.
    fn error_and_jacobian(&self, rotation: &Mat3F64, translation: &Vec3F64) -> (Vec6F64, Mat6F64) {
        let rr = *rotation * self.rotation.transpose();
        // SO3F64's log keeps the sign right up to pi
        let e_r = SO3F64::from_matrix(&rr).log();
        let e_t = *translation - rr * self.translation;
        let jl_inv = SO3F64::left_jacobian(e_r).inverse();
        let skew = SO3F64::hat(e_t);
        let mut jac = Mat6F64::identity();
        for i in 0..3 {
            for k in 0..3 {
                jac[(i, k)] = jl_inv.col(k)[i];
                jac[(3 + i, k)] = -skew.col(k)[i];
            }
        }
        let e = Vec6F64::new(e_r.x, e_r.y, e_r.z, e_t.x, e_t.y, e_t.z);
        (e, jac)
    }

    /// Adds `J^T info J` to `a` and `J^T info e` to `b` at `(rotation, translation)`.
    fn add_to(&self, rotation: &Mat3F64, translation: &Vec3F64, a: &mut Mat6F64, b: &mut Vec6F64) {
        let (e, jac) = self.error_and_jacobian(rotation, translation);
        let jt_info = jac.transpose() * self.info;
        *a += jt_info * jac;
        *b += jt_info * e;
    }
}

/// One Gauss-Newton step `x = [w, v]` from `A = J^T W J`, `b = J^T W r` and the priors at the
/// estimate `(rotation, translation)`.
///
/// Full-rank data take the Cholesky path on `A` plus the priors. Otherwise the unobserved
/// directions of `split` are removed from both `A` and `b`, so that a rounding-noise gradient
/// cannot push along them, the priors are added, and the sum is solved by eigen pseudo-inverse
/// in the depth-scaled tangent: a direction neither the data nor a prior observes gets a zero
/// update.
fn solve_step(
    a: &Mat6F64,
    b: &Vec6F64,
    split: &DegenerateSplit,
    priors: &[Option<PriorRows>],
    rotation: &Mat3F64,
    translation: &Vec3F64,
    rtol: f64,
) -> Result<Vec6F64, RgbdIcpError> {
    let (mut a, mut b) = if split.num_degenerate == 0 {
        (*a, *b)
    } else {
        let s = tangent_scale(split.scale);
        let b_observed = (split.projector * b.component_div(&s)).component_mul(&s);
        (split.a_observed, b_observed)
    };
    for p in priors.iter().flatten() {
        p.add_to(rotation, translation, &mut a, &mut b);
    }
    let x = if split.num_degenerate == 0 {
        // a weak pivot the eigen test still counts as observed falls back to the eigen solve
        cholesky_step(&a, &b).unwrap_or_else(|| pseudo_solve(&a, &(-b), split.scale, rtol))
    } else {
        pseudo_solve(&a, &(-b), split.scale, rtol)
    };
    if !x.iter().all(|v| v.is_finite()) {
        return Err(RgbdIcpError::SingularNormalEquations);
    }
    Ok(x)
}

/// Eigen split of `J^T W J` into observed and unobserved directions.
struct DegenerateSplit {
    /// Rotation scale of the decomposition, metres (see [`depth_scale`]).
    scale: f64,
    /// `J^T W J` with the unobserved eigen-directions removed, unscaled tangent.
    a_observed: Mat6F64,
    /// Projector onto the observed eigen-directions, in the scaled tangent.
    projector: Mat6F64,
    /// Unit unobserved directions in the unscaled tangent, entries `0..num_degenerate`.
    directions: [Vec6F64; 6],
    /// Number of unobserved directions.
    num_degenerate: usize,
}

impl DegenerateSplit {
    /// `a`, the matrix this split came from, with its unobserved directions removed: `a`
    /// itself when there are none, so full-rank data keep their exact entries.
    fn observed(&self, a: &Mat6F64) -> Mat6F64 {
        if self.num_degenerate == 0 {
            *a
        } else {
            self.a_observed
        }
    }
}

/// Rotation scale of the degeneracy test: `depth_m` when finite and positive, else 1 m.
#[inline]
fn depth_scale(depth_m: f64) -> f64 {
    if depth_m.is_finite() && depth_m > 0.0 {
        depth_m
    } else {
        1.0
    }
}

/// Diagonal of `S = diag(d, d, d, 1, 1, 1)`: radians of rotation to metres of arc at depth `d`.
#[inline]
fn tangent_scale(d: f64) -> Vec6F64 {
    Vec6F64::new(d, d, d, 1.0, 1.0, 1.0)
}

/// Symmetric eigen-decomposition of `S^-1 A S^-1` (see [`tangent_scale`]): eigenpairs in
/// ascending order and the largest eigenvalue. `a` must be finite: the eigen iteration does not
/// terminate on NaN.
fn scaled_eigen(a: &Mat6F64, d: f64) -> ([(f64, Vec6F64); 6], f64) {
    let s = tangent_scale(d);
    // symmetric by construction, as `a` is
    let m = Mat6F64::from_fn(|i, k| a[(i, k)] / (s[i] * s[k]));
    let eig = m.symmetric_eigen();
    let mut pairs: [(f64, Vec6F64); 6] =
        std::array::from_fn(|c| (eig.eigenvalues[c], eig.eigenvectors.column(c).into_owned()));
    pairs.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
    (pairs, pairs[5].0)
}

/// Splits `a` at `rtol` times its largest eigenvalue, rotation scaled by `depth_m`.
///
/// # Errors
///
/// [`RgbdIcpError::SingularNormalEquations`] if `a` is not finite.
fn split_degenerate(a: &Mat6F64, depth_m: f64, rtol: f64) -> Result<DegenerateSplit, RgbdIcpError> {
    if !a.iter().all(|v| v.is_finite()) {
        return Err(RgbdIcpError::SingularNormalEquations);
    }
    let d = depth_scale(depth_m);
    let s = tangent_scale(d);
    let (pairs, lmax) = scaled_eigen(a, d);
    let mut out = DegenerateSplit {
        scale: d,
        a_observed: Mat6F64::zeros(),
        projector: Mat6F64::zeros(),
        directions: [Vec6F64::zeros(); 6],
        num_degenerate: 0,
    };
    for (lambda, v) in pairs {
        // `lambda > ..` so that an all-zero system counts as unobserved
        if lambda > rtol * lmax && lmax > 0.0 {
            let vs = v.component_mul(&s);
            out.a_observed += vs * vs.transpose() * lambda;
            out.projector += v * v.transpose();
        } else {
            out.directions[out.num_degenerate] = v.component_div(&s).normalize();
            out.num_degenerate += 1;
        }
    }
    Ok(out)
}

/// Minimum-norm solve of `A x = rhs` in the depth-scaled tangent, dropping the eigen-directions
/// at or below `rtol` times the largest eigenvalue.
fn pseudo_solve(a: &Mat6F64, rhs: &Vec6F64, d: f64, rtol: f64) -> Vec6F64 {
    let s = tangent_scale(d);
    let rhs_scaled = rhs.component_div(&s);
    let (pairs, lmax) = scaled_eigen(a, d);
    let mut y = Vec6F64::zeros();
    if lmax > 0.0 {
        for (lambda, v) in pairs {
            if lambda > rtol * lmax {
                y += v * (v.dot(&rhs_scaled) / lambda);
            }
        }
    }
    y.component_div(&s)
}

/// Conditioning of each 3x3 block of `a`, `(rotation, translation)`: the weakest Cholesky pivot
/// of the block over its strongest, in `0..=1`, the pivots being the squared diagonal of `L`.
/// `(0, 0)` when `a` does not factor or a pivot is not finite, so a NaN never reaches the result.
fn block_conditioning(a: &Mat6F64) -> (f64, f64) {
    let Some(chol) = a.cholesky() else {
        return (0.0, 0.0);
    };
    let l = chol.l_dirty();
    let pivots: [f64; 6] = std::array::from_fn(|i| l[(i, i)] * l[(i, i)]);
    if !pivots.iter().all(|p| p.is_finite() && *p > 0.0) {
        return (0.0, 0.0);
    }
    let ratio = |block: &[f64]| {
        let lo = block.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = block.iter().copied().fold(0.0, f64::max);
        lo / hi
    };
    (ratio(&pivots[..3]), ratio(&pivots[3..]))
}

/// Median of `depths`, reordering it; `depths` must not be empty.
fn median_depth_m(depths: &mut [f32]) -> f64 {
    let mid = depths.len() / 2;
    let (_, m, _) = depths.select_nth_unstable_by(mid, f32::total_cmp);
    *m as f64
}

/// Coarse-to-fine Gauss-Newton from the initial pose; level counts already checked.
fn run_icp(
    workspace: &mut IcpWorkspace,
    src: &RgbdPyramid,
    tgt: &RgbdPyramid,
    init_rotation: &Mat3F64,
    init_translation: &Vec3F64,
    criteria: &IcpPlaneCriteria,
) -> Result<IcpPlaneResult, RgbdIcpError> {
    // Hybrid starts on nearest and switches once, on the finest level
    let (mut gates, priors) = check_criteria(criteria)?;
    let (mut rotation, mut translation) = (*init_rotation, *init_translation);
    let mut iterations = 0;
    let hybrid = criteria.association == Association::Hybrid;

    // iters_per_level is coarsest first; levels[0] is the finest
    let levels = (0..src.levels.len()).rev();
    for (&iters, level) in criteria.iters_per_level.iter().zip(levels) {
        let (s, t) = (&src.levels[level], &tgt.levels[level]);
        // the switch to bilinear brings its own budget of `iters` more iterations
        let (mut it, mut budget) = (0, iters);
        while it < budget {
            it += 1;
            let eqs = accumulate_level(s, t, &rotation, &translation, &gates, None);
            check_inliers(&eqs)?;
            // the mean, not the median: a select per iteration costs more than the scale needs
            let mean_depth = eqs.sum_depth_m / eqs.num_inliers as f64;
            let rtol = criteria.degeneracy_rtol;
            // unobserved directions are decided on the data alone, before any prior is added
            let split = split_degenerate(&eqs.a, mean_depth, rtol)?;
            let x = solve_step(
                &eqs.a,
                &eqs.b,
                &split,
                &priors,
                &rotation,
                &translation,
                rtol,
            )?;
            let (w, v) = (
                Vec3F64::new(x[0], x[1], x[2]),
                Vec3F64::new(x[3], x[4], x[5]),
            );
            let dr = SO3F64::exp(w).matrix();
            rotation = dr * rotation;
            translation = dr * translation + v;
            iterations += 1;

            // relative to the scene depth so that one tolerance fits any scale
            let below = |tol: f64| w.length() < tol && v.length() < tol * mean_depth;
            if hybrid && level == 0 && !gates.bilinear && below(HYBRID_SWITCH_NORM) {
                gates.bilinear = true;
                budget = it + iters;
                continue;
            }
            if below(criteria.update_tolerance) {
                break;
            }
        }
    }

    // the reported figures come from the returned pose, not the last pre-update pass, and from
    // the data alone: a prior must not make weak geometry look observed
    let finest = &src.levels[0];
    let depths = &mut workspace.depths;
    depths.clear();
    // by the image, not the inlier count, so a fuller frame after a sparse one never regrows it
    depths.reserve(finest.size().width * finest.size().height);
    let eqs = accumulate_level(
        finest,
        &tgt.levels[0],
        &rotation,
        &translation,
        &gates,
        Some(&mut *depths),
    );
    check_inliers(&eqs)?;
    let (rotation_conditioning, translation_conditioning) = block_conditioning(&eqs.a);
    let median_depth_m = median_depth_m(depths);
    let split = split_degenerate(&eqs.a, median_depth_m, criteria.degeneracy_rtol)?;
    let information = split.observed(&eqs.a);
    // num_valid >= num_associated >= num_inliers >= MIN_INLIERS, so no ratio divides by zero
    let PassCounts {
        num_valid,
        num_associated,
    } = eqs.counts;
    Ok(IcpPlaneResult {
        rotation,
        translation,
        rmse: (eqs.sum_sq_residual / eqs.num_inliers as f64).sqrt(),
        iterations,
        num_inliers: eqs.num_inliers,
        num_associated,
        inlier_fraction: eqs.num_inliers as f64 / num_associated as f64,
        overlap_fraction: num_associated as f64 / num_valid as f64,
        rotation_conditioning,
        translation_conditioning,
        observability: rotation_conditioning.min(translation_conditioning),
        median_depth_m,
        information,
        degenerate_directions: split.directions,
        num_degenerate: split.num_degenerate,
    })
}

#[cfg(test)]
mod tests {
    use super::super::rgbd::RgbdPyramid;
    use super::super::synth::{
        depth_m_from_mm, render_depth_mm, test_camera, Plane, Scene, Sphere, TEST_SIZE,
    };
    use super::*;
    use crate::camera::PinholeCamera;
    use kornia_algebra::SO3F64;
    use kornia_image::ImageSize;

    const GATES: PassGates = PassGates {
        bilinear: false,
        max_dist_m: 0.1,
        cos_max_normal_angle: 0.866, // 30 deg
        huber_delta: 1.345,
        depth_noise: DepthNoise::Constant { sigma_m: 0.002 },
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

    /// A gated row frozen for finite differences: (source point, target point, target normal,
    /// weight).
    type Row = (Vec3F64, Vec3F64, Vec3F64, f64);

    /// The rows of one pass at `(rot, t)` with their weights, fixed for the perturbations.
    fn gated_rows(
        src: &RgbdLevel,
        tgt: &RgbdLevel,
        rot: &Mat3F64,
        t: &Vec3F64,
        gates: &PassGates,
    ) -> Vec<Row> {
        let mut rows = Vec::new();
        for_each_correspondence(src, tgt, rot, t, gates, |c| {
            let w = row_weight(c.r, c.sigma, gates.huber_delta);
            rows.push((vec3(pixel3(&src.vertices, c.src_idx)), c.v_t, c.n_t, w));
        });
        rows
    }

    /// Residual of `row` at `exp(xi) (rot, t)`, the solver's left perturbation, target fixed.
    fn perturbed_residual(rot: &Mat3F64, t: &Vec3F64, xi: &[f64; 6], row: &Row) -> f64 {
        let (p, v, n, _) = row;
        let d = SO3F64::exp(Vec3F64::new(xi[0], xi[1], xi[2])).matrix();
        let q = d * (*rot * *p + *t) + Vec3F64::new(xi[3], xi[4], xi[5]);
        n.dot(q - *v)
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

        let eqs = accumulate_level(s0, t0, &rot, &t, &GATES, None);
        assert!(eqs.num_inliers > 20_000, "inliers: {}", eqs.num_inliers);
        assert!(eqs.counts.num_associated >= eqs.num_inliers);
        assert!(eqs.counts.num_valid >= eqs.counts.num_associated);

        let rows = gated_rows(s0, t0, &rot, &t, &GATES);
        let mut downweighted = 0;
        for_each_correspondence(s0, t0, &rot, &t, &GATES, |c| {
            if (c.r / c.sigma).abs() > GATES.huber_delta {
                downweighted += 1;
            }
        });
        assert_eq!(rows.len(), eqs.num_inliers);
        assert!(downweighted > 100, "Huber branch unused: {downweighted}");

        let residual = |xi: &[f64; 6], row: &Row| perturbed_residual(&rot, &t, xi, row);
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

        let eqs = accumulate_level(s0, t0, &rot_gt, &t_gt, &GATES, None);
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
        let eqs_far = accumulate_level(s0, t0, &rot_gt, &far, &tight, None);
        assert!(eqs_far.num_inliers * 20 < eqs.num_inliers);
        assert_eq!(eqs_far.counts.num_valid, eqs.counts.num_valid);

        // a normal gate of -1 (any angle) keeps at least as many as the 30 deg gate
        let any_angle = PassGates {
            cos_max_normal_angle: -1.0,
            ..GATES
        };
        assert!(
            accumulate_level(s0, t0, &rot_gt, &t_gt, &any_angle, None).num_inliers
                >= eqs.num_inliers
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

    /// Three-level pyramid of `scene` seen from `T_world_cam = (rot_wc, t_wc)`.
    fn pyramid(
        scene: &Scene,
        rot_wc: &Mat3F64,
        t_wc: &Vec3F64,
    ) -> Result<RgbdPyramid, Box<dyn std::error::Error>> {
        let depth = render_depth_mm(scene, &test_camera(), TEST_SIZE, rot_wc, t_wc)?;
        Ok(RgbdPyramid::from_depth_mm(&depth, &test_camera(), 3)?)
    }

    /// Angle between two rotations, in degrees.
    fn rotation_error_deg(a: &Mat3F64, b: &Mat3F64) -> f64 {
        SO3F64::from_matrix(&(a.transpose() * *b))
            .log()
            .length()
            .to_degrees()
    }

    fn icp_from_identity(
        src: &RgbdPyramid,
        tgt: &RgbdPyramid,
        criteria: &IcpPlaneCriteria,
    ) -> Result<IcpPlaneResult, RgbdIcpError> {
        icp_projective_plane(src, tgt, &Mat3F64::IDENTITY, &Vec3F64::ZERO, criteria)
    }

    #[test]
    fn test_identity_self_icp() -> Result<(), Box<dyn std::error::Error>> {
        let pyr = pyramid(
            &Scene::corner_and_sphere(),
            &Mat3F64::IDENTITY,
            &Vec3F64::ZERO,
        )?;
        let result = icp_from_identity(&pyr, &pyr, &IcpPlaneCriteria::default())?;
        let rot_err = rotation_error_deg(&result.rotation, &Mat3F64::IDENTITY);
        assert!(rot_err < 0.01, "self-ICP rotation drifted: {rot_err} deg");
        let t_err = result.translation.length();
        assert!(t_err < 1e-4, "self-ICP translation drifted: {t_err}");
        assert!(result.rmse < 1e-6, "self-ICP rmse: {}", result.rmse);
        assert!(result.num_inliers > 40_000, "{}", result.num_inliers);
        // the relative tolerance stops each level after its first, zero, step
        assert_eq!(result.iterations, 3);
        Ok(())
    }

    #[test]
    fn test_ground_truth_motion_grid() -> Result<(), Box<dyn std::error::Error>> {
        let scene = Scene::corner_and_sphere();
        let src = pyramid(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
        // (rotation vector of the target camera pose T_world_cam, deg; its translation, m)
        let cases: &[([f64; 3], [f64; 3])] = &[
            ([2.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([0.0, -2.0, 0.0], [0.0, 0.0, 0.0]),
            ([0.0, 0.0, 2.0], [0.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [0.03, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [0.0, -0.03, 0.0]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.03]),
            ([1.155, 1.155, 1.155], [0.02, -0.02, 0.015]),
        ];
        for (w_deg, t) in cases {
            let w = Vec3F64::from_array(w_deg.map(f64::to_radians));
            let (rot_wc, t_wc) = (SO3F64::exp(w).matrix(), Vec3F64::from_array(*t));
            let tgt = pyramid(&scene, &rot_wc, &t_wc)?;
            // T_target_source = T_world_cam^-1, the source camera at the world origin
            let rot_gt = rot_wc.transpose();
            let t_gt = -(rot_gt * t_wc);

            let result = icp_from_identity(&src, &tgt, &IcpPlaneCriteria::default())?;
            let rot_err = rotation_error_deg(&result.rotation, &rot_gt);
            let t_err = (result.translation - t_gt).length();
            assert!(
                rot_err < 0.2,
                "{w_deg:?}/{t:?}: rotation error {rot_err} deg"
            );
            assert!(t_err < 5e-3, "{w_deg:?}/{t:?}: translation error {t_err} m");
            assert!(result.rmse < 3e-3, "{w_deg:?}/{t:?}: rmse {}", result.rmse);
        }
        Ok(())
    }

    #[test]
    fn test_outlier_band_still_converges() -> Result<(), Box<dyn std::error::Error>> {
        let (camera, scene) = (test_camera(), Scene::corner_and_sphere());
        let src = pyramid(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
        let rot_wc = SO3F64::exp(Vec3F64::new(0.0, 1.5f64.to_radians(), 0.0)).matrix();
        let t_wc = Vec3F64::new(0.02, 0.0, 0.01);
        let mut depth = render_depth_mm(&scene, &camera, TEST_SIZE, &rot_wc, &t_wc)?;
        // a 10% band of rows biased by +60 mm: inside the 100 mm distance gate, so it exercises
        // the Huber weighting rather than the gate
        let w = TEST_SIZE.width;
        for d in &mut depth.as_slice_mut()[80 * w..(80 + TEST_SIZE.height / 10) * w] {
            if *d != 0 {
                *d += 60;
            }
        }
        let tgt = RgbdPyramid::from_depth_mm(&depth, &camera, 3)?;
        let rot_gt = rot_wc.transpose();
        let t_gt = -(rot_gt * t_wc);

        let result = icp_from_identity(&src, &tgt, &IcpPlaneCriteria::default())?;
        let rot_err = rotation_error_deg(&result.rotation, &rot_gt);
        let t_err = (result.translation - t_gt).length();
        assert!(rot_err < 0.2, "rotation error with outliers: {rot_err} deg");
        assert!(t_err < 5e-3, "translation error with outliers: {t_err} m");
        Ok(())
    }

    #[test]
    fn invalid_inputs_are_errors() -> Result<(), Box<dyn std::error::Error>> {
        let (camera, scene) = (test_camera(), Scene::corner_and_sphere());
        let pyr = pyramid(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
        let depth = render_depth_mm(
            &scene,
            &camera,
            TEST_SIZE,
            &Mat3F64::IDENTITY,
            &Vec3F64::ZERO,
        )?;
        let two = RgbdPyramid::from_depth_mm(&depth, &camera, 2)?;
        let ok = IcpPlaneCriteria::default();
        assert!(matches!(
            icp_from_identity(&pyr, &two, &ok),
            Err(RgbdIcpError::PyramidLevelsMismatch {
                src_levels: 3,
                tgt_levels: 2
            })
        ));

        for iters in [vec![], vec![5, 5]] {
            let c = IcpPlaneCriteria {
                iters_per_level: iters,
                ..ok.clone()
            };
            assert!(matches!(
                icp_from_identity(&pyr, &pyr, &c),
                Err(RgbdIcpError::IterationsLevelsMismatch { levels: 3, .. })
            ));
        }

        let bad = [
            IcpPlaneCriteria {
                max_dist_m: f64::NAN,
                ..ok.clone()
            },
            IcpPlaneCriteria {
                huber_delta: -1.0,
                ..ok.clone()
            },
            IcpPlaneCriteria {
                depth_noise: DepthNoise::Constant { sigma_m: 0.0 },
                ..ok.clone()
            },
            IcpPlaneCriteria {
                update_tolerance: f64::INFINITY,
                ..ok.clone()
            },
            IcpPlaneCriteria {
                max_normal_angle_rad: 4.0,
                ..ok.clone()
            },
            IcpPlaneCriteria {
                degeneracy_rtol: f64::NAN,
                ..ok.clone()
            },
            IcpPlaneCriteria {
                degeneracy_rtol: 0.0,
                ..ok.clone()
            },
        ];
        // every DepthNoise parameter is checked, NaN and infinity included
        let stereo = |i: usize, x: f64| {
            let mut p = [200.0, 0.075, 0.01];
            p[i] = x;
            IcpPlaneCriteria {
                depth_noise: DepthNoise::Stereo {
                    focal_px: p[0],
                    baseline_m: p[1],
                    sigma_disparity_px: p[2],
                },
                ..ok.clone()
            }
        };
        let noise_bad =
            (0..3).flat_map(|i| [0.0, -1.0, f64::NAN, f64::INFINITY].map(|x| stereo(i, x)));
        let constant_bad = [-1.0, f64::NAN, f64::INFINITY].map(|sigma_m| IcpPlaneCriteria {
            depth_noise: DepthNoise::Constant { sigma_m },
            ..ok.clone()
        });
        let bad: Vec<_> = bad
            .into_iter()
            .chain(noise_bad)
            .chain(constant_bad)
            .collect();
        for c in &bad {
            let result = icp_from_identity(&pyr, &pyr, c);
            assert!(
                matches!(result, Err(RgbdIcpError::InvalidCriteria { .. })),
                "{c:?}: {result:?}"
            );
        }

        // 5 m off: nothing projects onto the target
        let far = icp_projective_plane(
            &pyr,
            &pyr,
            &Mat3F64::IDENTITY,
            &Vec3F64::new(0.0, 0.0, -5.0),
            &ok,
        );
        assert!(
            matches!(far, Err(RgbdIcpError::TooFewCorrespondences(_))),
            "{far:?}"
        );
        Ok(())
    }

    /// One fronto-parallel plane leaves in-plane translation and the rotation about its normal
    /// unobserved. It used to be rejected as singular; it now converges along the three observed
    /// directions and reports the other three, with zero information along them.
    #[test]
    fn test_single_plane_is_degenerate() -> Result<(), Box<dyn std::error::Error>> {
        let scene = Scene {
            planes: vec![Plane {
                normal: Vec3F64::new(0.0, 0.0, 1.0),
                d: 2.0,
            }],
            spheres: vec![],
        };
        let pyr = pyramid(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
        let result = icp_from_identity(&pyr, &pyr, &IcpPlaneCriteria::default())?;
        assert_eq!(result.num_degenerate, 3, "{result:?}");
        // nothing moved: the observed directions have nothing to correct, the others no update
        assert!(rotation_error_deg(&result.rotation, &Mat3F64::IDENTITY) < 1e-6);
        assert!(result.translation.length() < 1e-9);
        assert!(result.observability < 1e-6);
        // in-plane x, y translation and z rotation, each with zero information
        for d in &result.degenerate_directions[..3] {
            assert!((d.norm() - 1.0).abs() < 1e-12);
            // in the span of w_z, v_x and v_y
            assert!(d[0].abs() + d[1].abs() + d[5].abs() < 1e-6, "{d:?}");
            let info = d.dot(&(result.information * d)) / result.information.amax();
            assert!(info.abs() < 1e-12, "information along {d:?}: {info:.3e}");
        }
        assert!(result.degenerate_directions[3..]
            .iter()
            .all(|d| *d == Vec6F64::zeros()));
        Ok(())
    }

    #[test]
    fn solve_step_zeroes_unobserved_directions_and_rejects_nan() -> Result<(), RgbdIcpError> {
        let rtol = DEFAULT_DEGENERACY_RTOL;
        let b = Vec6F64::from_element(1.0);
        let full = Mat6F64::identity();
        let id = (Mat3F64::IDENTITY, Vec3F64::ZERO);
        let x = solve_step(
            &full,
            &b,
            &split_degenerate(&full, 1.0, rtol)?,
            &[],
            &id.0,
            &id.1,
            rtol,
        )?;
        assert_eq!(x, -b);
        let mut nan = Mat6F64::identity();
        nan[(4, 4)] = f64::NAN;
        assert!(split_degenerate(&nan, 1.0, rtol).is_err());
        // positive definite, but one translation eigenvalue is 1e-10 of the largest
        let mut weak = Mat6F64::identity();
        weak[(5, 5)] = 1e-10;
        let split = split_degenerate(&weak, 1.0, rtol)?;
        assert_eq!(split.num_degenerate, 1);
        assert_eq!(split.directions[0][5].abs(), 1.0);
        let x = solve_step(&weak, &b, &split, &[], &id.0, &id.1, rtol)?;
        assert!((x - Vec6F64::new(-1.0, -1.0, -1.0, -1.0, -1.0, 0.0)).amax() < 1e-12);
        Ok(())
    }

    /// The default degeneracy threshold sits between an exact null space and the weakest
    /// geometry the solver has always solved.
    ///
    /// Smallest over largest eigenvalue of the depth-scaled `J^T W J` after a 2 cm slide: a
    /// wall has three null directions, a wall with a 6 cm bump does not, and neither does the
    /// corner and sphere.
    #[test]
    fn default_degeneracy_rtol_separates_null_space_from_weak_geometry(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let plane = || Plane {
            normal: Vec3F64::new(0.0, 0.0, 1.0),
            d: 2.0,
        };
        let wall = Scene {
            planes: vec![plane()],
            spheres: vec![],
        };
        let bump = Scene {
            planes: vec![plane()],
            spheres: vec![Sphere {
                center: Vec3F64::new(0.0, 0.0, 1.9),
                radius: 0.06,
            }],
        };
        let slide = Vec3F64::new(0.02, 0.0, 0.0);
        let mut smallest = Vec::new();
        for scene in [&wall, &bump, &Scene::corner_and_sphere()] {
            let s = pyramid(scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
            let t = pyramid(scene, &Mat3F64::IDENTITY, &slide)?;
            let c = IcpPlaneCriteria::default();
            let r = icp_from_identity(&s, &t, &c)?;
            let (gates, _) = check_criteria(&c)?;
            let eqs = accumulate_level(
                &s.levels[0],
                &t.levels[0],
                &r.rotation,
                &r.translation,
                &gates,
                None,
            );
            let (pairs, lmax) = scaled_eigen(&eqs.a, r.median_depth_m);
            smallest.push((pairs[0].0 / lmax, pairs[3].0 / lmax, r.num_degenerate));
        }
        let (wall, bump, rich) = (smallest[0], smallest[1], smallest[2]);
        assert!(wall.0 < 1e-12 && wall.2 == 3, "wall: {wall:?}");
        assert!(
            wall.1 > 1e-3,
            "wall keeps three observed directions: {wall:?}"
        );
        assert!(
            bump.0 > DEFAULT_DEGENERACY_RTOL && bump.2 == 0,
            "bump must still be solved: {bump:?}"
        );
        assert!(rich.0 > 1e-3 && rich.2 == 0, "rich: {rich:?}");
        Ok(())
    }

    /// Pyramid of `levels` levels of `scene` through `camera` at `size`, the camera translated
    /// to `t_wc` without rotation.
    fn pyramid_with(
        scene: &Scene,
        camera: &PinholeCamera,
        size: ImageSize,
        levels: usize,
        t_wc: &Vec3F64,
    ) -> Result<RgbdPyramid, Box<dyn std::error::Error>> {
        let depth = render_depth_mm(scene, camera, size, &Mat3F64::IDENTITY, t_wc)?;
        Ok(RgbdPyramid::from_depth_mm(&depth, camera, levels)?)
    }

    /// A partly overlapping view keeps a high match quality while the overlap falls.
    ///
    /// `inlier_fraction` once divided by all valid source pixels, so it fell with the view
    /// overlap rather than with the match error and rejected converged solves; quality and
    /// overlap are separate signals.
    #[test]
    fn inlier_fraction_measures_quality_not_overlap() -> Result<(), Box<dyn std::error::Error>> {
        let scene = Scene::corner_and_sphere();
        let src = pyramid(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
        // yaw far enough that a large part of the source view leaves the target frustum
        let rot_wc = SO3F64::exp(Vec3F64::new(0.0, 18f64.to_radians(), 0.0)).matrix();
        let t_wc = Vec3F64::new(0.35, 0.0, 0.0);
        let tgt = pyramid(&scene, &rot_wc, &t_wc)?;
        let rot_gt = rot_wc.transpose();
        let t_gt = -(rot_gt * t_wc);

        // started at the true rotation: translation alone has to be found
        let result = icp_projective_plane(
            &src,
            &tgt,
            &rot_gt,
            &Vec3F64::ZERO,
            &IcpPlaneCriteria::default(),
        )?;
        let rot_err = rotation_error_deg(&result.rotation, &rot_gt);
        let t_err = (result.translation - t_gt).length();
        assert!(rot_err < 0.5, "rotation error {rot_err} deg");
        assert!(t_err < 0.01, "translation error {t_err} m");
        assert!(
            result.inlier_fraction > 0.9,
            "quality fell with overlap: {} ({} associated)",
            result.inlier_fraction,
            result.num_associated
        );
        assert!(
            result.overlap_fraction < 0.7,
            "overlap missed the rotated-away view: {}",
            result.overlap_fraction
        );
        // their product is the old inliers-over-valid figure
        let inliers_over_valid = result.inlier_fraction * result.overlap_fraction;
        assert!(result.inlier_fraction > 1.5 * inliers_over_valid);
        assert!(result.num_associated > 1000, "{}", result.num_associated);
        Ok(())
    }

    /// Weak geometry shows in `observability` and in nothing else.
    ///
    /// A wall with one small bump: the bump pins the three directions the wall leaves free, but
    /// only just. Sliding along the wall then costs little residual, so the inlier fraction and
    /// the rmse look as good as on a well-conditioned scene.
    #[test]
    fn observability_exposes_weak_geometry() -> Result<(), Box<dyn std::error::Error>> {
        let weak = Scene {
            planes: vec![Plane {
                normal: Vec3F64::new(0.0, 0.0, 1.0),
                d: 2.0,
            }],
            spheres: vec![Sphere {
                center: Vec3F64::new(0.0, 0.0, 1.9),
                radius: 0.06,
            }],
        };
        let slide = Vec3F64::new(0.02, 0.0, 0.0);
        let mut measured = Vec::new();
        for scene in [weak, Scene::corner_and_sphere()] {
            let src = pyramid(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
            let tgt = pyramid(&scene, &Mat3F64::IDENTITY, &slide)?;
            let r = icp_from_identity(&src, &tgt, &IcpPlaneCriteria::default())?;
            assert!((0.0..=1.0).contains(&r.observability));
            measured.push((r.observability, r.inlier_fraction, r.rmse));
        }
        let ((weak_obs, weak_inl, weak_rmse), (rich_obs, rich_inl, _)) = (measured[0], measured[1]);
        assert!(
            weak_obs < 1e-4 && rich_obs > 1e-3,
            "weak {weak_obs:.3e}, rich {rich_obs:.3e}"
        );
        // the figures a caller would otherwise trust are no worse on the weak scene
        assert!(
            weak_inl >= rich_inl && weak_rmse < 1e-3,
            "weak inliers {weak_inl} (rich {rich_inl}), weak rmse {weak_rmse}"
        );
        Ok(())
    }

    /// The same geometry at half resolution, a quarter of the correspondences, reports the same
    /// conditioning: it is a pivot ratio, not a pivot magnitude.
    #[test]
    fn conditioning_is_scale_free() -> Result<(), Box<dyn std::error::Error>> {
        let scene = Scene::corner_and_sphere();
        let full = test_camera();
        let half = PinholeCamera {
            fx: full.fx / 2.0,
            fy: full.fy / 2.0,
            cx: (full.cx + 0.5) / 2.0 - 0.5,
            cy: (full.cy + 0.5) / 2.0 - 0.5,
            ..full
        };
        let half_size = ImageSize {
            width: TEST_SIZE.width / 2,
            height: TEST_SIZE.height / 2,
        };
        let slide = Vec3F64::new(0.02, 0.0, 0.0);
        let criteria = IcpPlaneCriteria {
            iters_per_level: vec![10, 7],
            ..Default::default()
        };
        let mut solved = Vec::new();
        for (camera, size) in [(&full, TEST_SIZE), (&half, half_size)] {
            let src = pyramid_with(&scene, camera, size, 2, &Vec3F64::ZERO)?;
            let tgt = pyramid_with(&scene, camera, size, 2, &slide)?;
            solved.push(icp_from_identity(&src, &tgt, &criteria)?);
        }
        let (a, b) = (&solved[0], &solved[1]);
        assert!(
            b.num_associated * 3 < a.num_associated,
            "{} vs {}",
            b.num_associated,
            a.num_associated
        );
        for (name, x, y) in [
            ("rotation", a.rotation_conditioning, b.rotation_conditioning),
            (
                "translation",
                a.translation_conditioning,
                b.translation_conditioning,
            ),
        ] {
            assert!(x > 0.0 && y > 0.0, "{name}: {x:.3e} vs {y:.3e}");
            assert!(x.max(y) / x.min(y) < 10.0, "{name}: {x:.3e} vs {y:.3e}");
        }
        // the median depth does not depend on the resolution either
        assert!((a.median_depth_m - b.median_depth_m).abs() < 0.05);
        Ok(())
    }

    #[test]
    fn block_conditioning_is_zero_unless_it_factors() {
        let mut a = Mat6F64::identity();
        a[(1, 1)] = 0.25;
        a[(4, 4)] = 4.0;
        assert_eq!(block_conditioning(&a), (0.25, 0.25));
        a[(4, 4)] = f64::NAN;
        assert_eq!(block_conditioning(&a), (0.0, 0.0));
        assert_eq!(block_conditioning(&Mat6F64::zeros()), (0.0, 0.0));
    }

    const TEST_STEREO: DepthNoise = DepthNoise::Stereo {
        focal_px: 200.0,
        baseline_m: 0.075,
        // ~2 mm at 2 m: small enough that central-difference normals stay meaningful
        sigma_disparity_px: 0.0075,
    };

    /// Metre depth of `scene` seen from `T_world_cam = (rot_wc, t_wc)`.
    fn depth_m(
        scene: &Scene,
        rot_wc: &Mat3F64,
        t_wc: &Vec3F64,
    ) -> Result<Image<f32, 1>, Box<dyn std::error::Error>> {
        let mm = render_depth_mm(scene, &test_camera(), TEST_SIZE, rot_wc, t_wc)?;
        Ok(depth_m_from_mm(&mm)?)
    }

    /// A per-pixel sigma map equal to the stereo model must weight like the model.
    #[test]
    fn sigma_map_matches_stereo_model() -> Result<(), Box<dyn std::error::Error>> {
        let (camera, scene) = (test_camera(), Scene::corner_and_sphere());
        let rot_wc = SO3F64::exp(Vec3F64::new(0.0, 1f64.to_radians(), 0.0)).matrix();
        let t_wc = Vec3F64::new(0.01, 0.0, 0.0);
        let ds = depth_m(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
        let dt = depth_m(&scene, &rot_wc, &t_wc)?;
        let sig = |d: &Image<f32, 1>| {
            let s = d.as_slice().iter();
            Image::<f32, 1>::new(
                d.size(),
                s.map(|&z| TEST_STEREO.depth_sigma(z as f64) as f32)
                    .collect(),
            )
        };
        let model = (
            RgbdPyramid::from_depth_m(&ds, &camera, 3)?,
            RgbdPyramid::from_depth_m(&dt, &camera, 3)?,
        );
        let mapped = (
            RgbdPyramid::from_depth_m_with_sigma(&ds, &sig(&ds)?, &camera, 3)?,
            RgbdPyramid::from_depth_m_with_sigma(&dt, &sig(&dt)?, &camera, 3)?,
        );
        // the map wins over a deliberately wrong constant model
        let wrong = DepthNoise::Constant { sigma_m: 1.0 };
        let (rot_gt, t_gt) = (rot_wc.transpose(), -(rot_wc.transpose() * t_wc));
        let finest = |p: &(RgbdPyramid, RgbdPyramid), noise| {
            let gates = PassGates {
                depth_noise: noise,
                ..GATES
            };
            accumulate_level(&p.0.levels[0], &p.1.levels[0], &rot_gt, &t_gt, &gates, None)
        };
        let (a, b) = (finest(&model, TEST_STEREO), finest(&mapped, wrong));
        assert_eq!(a.num_inliers, b.num_inliers);
        for i in 0..6 {
            let rel = (a.a[(i, i)] - b.a[(i, i)]).abs() / a.a[(i, i)];
            // the map holds the model's sigma rounded to f32
            assert!(rel < 1e-5, "J^T W J[{i}][{i}] differs by {rel:.3e}");
        }
        // without maps the wrong model is used, so the comparison above can fail
        let c = finest(&model, wrong);
        assert!(a.a[(3, 3)] > 1e3 * c.a[(3, 3)]);

        let criteria = |depth_noise| IcpPlaneCriteria {
            depth_noise,
            ..Default::default()
        };
        let rm = icp_from_identity(&model.0, &model.1, &criteria(TEST_STEREO))?;
        let rp = icp_from_identity(&mapped.0, &mapped.1, &criteria(wrong))?;
        assert!(rotation_error_deg(&rm.rotation, &rp.rotation) < 0.01);
        assert!((rm.translation - rp.translation).length() < 1e-4);
        assert!((rm.translation - t_gt).length() < 5e-3);
        Ok(())
    }

    /// A sigma map below the floor weights its rows at `MIN_SIGMA_M`, the floor of every other
    /// sigma, so `J^T W J` stays finite.
    #[test]
    fn sigma_map_shares_the_sigma_floor() -> Result<(), Box<dyn std::error::Error>> {
        let d = depth_m(
            &Scene::corner_and_sphere(),
            &Mat3F64::IDENTITY,
            &Vec3F64::ZERO,
        )?;
        let tiny = Image::<f32, 1>::from_size_val(d.size(), 1e-30)?;
        let pyr = RgbdPyramid::from_depth_m_with_sigma(&d, &tiny, &test_camera(), 1)?;
        let l0 = &pyr.levels[0];
        let (rot, t) = (Mat3F64::IDENTITY, Vec3F64::ZERO);
        let mut rows = 0;
        for_each_correspondence(l0, l0, &rot, &t, &GATES, |c| {
            assert_eq!(c.sigma, MIN_SIGMA_M);
            rows += 1;
        });
        assert!(rows > 10_000, "{rows}");
        let eqs = accumulate_level(l0, l0, &rot, &t, &GATES, None);
        assert!(eqs.a.iter().all(|x| x.is_finite()));
        Ok(())
    }

    /// A `width` x `height` target level for the samplers: vertex `(u, v, z(u, v))` with a
    /// normal facing the camera, invalid where `z` is 0, and a sigma map of `0.01 * z`.
    fn sample_level(
        width: usize,
        height: usize,
        z: impl Fn(usize, usize) -> f32,
    ) -> Result<RgbdLevel, Box<dyn std::error::Error>> {
        let size = ImageSize { width, height };
        let (mut vs, mut ns, mut ss) = (Vec::new(), Vec::new(), Vec::new());
        for v in 0..height {
            for u in 0..width {
                let d = z(u, v);
                let ok = if d > 0.0 { 1.0 } else { 0.0 };
                vs.extend([u as f32 * ok, v as f32 * ok, d]);
                ns.extend([0.0, 0.0, -ok]);
                ss.push(0.01 * d);
            }
        }
        Ok(RgbdLevel {
            camera: test_camera(),
            vertices: Image::new(size, vs)?,
            normals: Image::new(size, ns)?,
            depth_sigmas: Some(Image::new(size, ss)?),
        })
    }

    /// A bilinear sample needs only the pixels it weights: at integer `u` or on the last row
    /// and column, an invalid or far neighbour with zero weight must not drop it.
    #[test]
    fn bilinear_ignores_zero_weight_taps() -> Result<(), Box<dyn std::error::Error>> {
        // column 2 is a hole
        let holed = sample_level(
            4,
            3,
            |u, _| if u == 2 { 0.0 } else { 1.0 + 0.01 * u as f32 },
        )?;
        let s = sample_target(&holed, 1.0, 0.5).ok_or("integer u next to a hole dropped")?;
        assert!((s.vertex - Vec3F64::new(1.0, 0.5, 1.01)).length() < 1e-6);
        assert_eq!(s.normal, Vec3F64::new(0.0, 0.0, -1.0));
        assert!((s.depth_sigma.ok_or("no sigma")? - 0.0101).abs() < 1e-8);
        assert!(
            sample_target(&holed, 1.5, 0.5).is_none(),
            "weighted hole kept"
        );
        assert!(
            sample_target(&holed, 3.0, 2.0).is_some(),
            "last pixel dropped"
        );
        // a 2x depth step from column 2 on
        let step = sample_level(4, 3, |u, _| if u >= 2 { 2.0 } else { 1.0 })?;
        assert!(
            sample_target(&step, 1.0, 1.0).is_some(),
            "zero-weight jump dropped"
        );
        assert!(
            sample_target(&step, 1.5, 1.0).is_none(),
            "weighted jump kept"
        );
        Ok(())
    }

    /// Pyramid levels are at least 3x3, so this is defence in depth, on the sampler itself: a
    /// 1-pixel-wide or -high map must read only its own pixels, and non-finite coordinates
    /// give no sample.
    #[test]
    fn bilinear_handles_one_pixel_maps_and_non_finite_coordinates(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let column = sample_level(1, 3, |_, v| 1.0 + 0.01 * v as f32)?;
        let s = sample_target(&column, 0.0, 1.5).ok_or("1-wide map dropped")?;
        assert!((s.vertex - Vec3F64::new(0.0, 1.5, 1.015)).length() < 1e-6);
        assert!(sample_target(&column, 0.5, 1.0).is_none());
        let row = sample_level(3, 1, |u, _| 1.0 + 0.01 * u as f32)?;
        let s = sample_target(&row, 1.5, 0.0).ok_or("1-high map dropped")?;
        assert!((s.vertex - Vec3F64::new(1.5, 0.0, 1.015)).length() < 1e-6);
        assert!(sample_target(&row, 1.0, 0.5).is_none());
        let single = sample_level(1, 1, |_, _| 1.0)?;
        assert!(sample_target(&single, 0.0, 0.0).is_some());

        let lvl = sample_level(4, 3, |_, _| 1.0)?;
        let (nan, inf) = (f64::NAN, f64::INFINITY);
        for (u, v) in [
            (nan, 1.0),
            (1.0, nan),
            (inf, 1.0),
            (-inf, 1.0),
            (1.0, inf),
            (1.0, -inf),
        ] {
            assert!(sample_target(&lvl, u, v).is_none(), "({u}, {v}) sampled");
            assert!(sample_nearest(&lvl, u, v).is_none(), "({u}, {v}) snapped");
        }
        Ok(())
    }

    /// The reported information must be the Hessian of the cost the solver minimises.
    ///
    /// Fixes the final pass's correspondences and their weights, perturbs the estimate in the
    /// solver's left tangent and takes central second differences of `0.5 * sum w r^2`. At a
    /// converged, noise-free solve the residuals are near zero, so the Gauss-Newton `J^T W J`
    /// and the true Hessian agree; a wrong Jacobian, sign, axis order or weight would not.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "about 4.5 min in a debug build; runs in the release test job"
    )]
    fn information_matches_finite_difference_hessian() -> Result<(), Box<dyn std::error::Error>> {
        for association in [
            Association::Nearest,
            Association::Bilinear,
            Association::Hybrid,
        ] {
            fd_check(association)?;
        }
        Ok(())
    }

    /// One association of [`information_matches_finite_difference_hessian`].
    fn fd_check(association: Association) -> Result<(), Box<dyn std::error::Error>> {
        let (camera, scene) = (test_camera(), Scene::corner_and_sphere());
        let w = Vec3F64::new(1.0, 1.0, 1.0) * (1.5f64.to_radians() / 3f64.sqrt());
        let rot_wc = SO3F64::exp(w).matrix();
        let t_wc = Vec3F64::new(0.02, -0.01, 0.015);
        let src = RgbdPyramid::from_depth_m(
            &depth_m(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?,
            &camera,
            3,
        )?;
        let tgt = RgbdPyramid::from_depth_m(&depth_m(&scene, &rot_wc, &t_wc)?, &camera, 3)?;
        let criteria = IcpPlaneCriteria {
            depth_noise: TEST_STEREO,
            association,
            ..Default::default()
        };
        let res = icp_from_identity(&src, &tgt, &criteria)?;
        let info = &res.information;
        assert!((info - info.transpose()).amax() == 0.0);

        let (mut gates, _) = check_criteria(&criteria)?;
        // the final pass of Hybrid is bilinear once it has switched, which it does here
        gates.bilinear = association != Association::Nearest;
        let (rot, t) = (res.rotation, res.translation);
        let (s0, t0) = (&src.levels[0], &tgt.levels[0]);
        let rows = gated_rows(s0, t0, &rot, &t, &gates);
        assert_eq!(rows.len(), res.num_inliers);
        let cost = |xi: &[f64; 6]| -> f64 {
            rows.iter()
                .map(|row| {
                    let e = perturbed_residual(&rot, &t, xi, row);
                    0.5 * row.3 * e * e
                })
                .sum()
        };

        let h = 1e-4;
        let mut worst: f64 = 0.0;
        for j in 0..6 {
            for k in 0..6 {
                let at = |sj: f64, sk: f64| {
                    let mut xi = [0.0; 6];
                    xi[j] += sj * h;
                    xi[k] += sk * h;
                    cost(&xi)
                };
                let fd =
                    (at(1.0, 1.0) - at(1.0, -1.0) - at(-1.0, 1.0) + at(-1.0, -1.0)) / (4.0 * h * h);
                let norm = (info[(j, j)] * info[(k, k)]).sqrt();
                worst = worst.max((fd - info[(j, k)]).abs() / norm);
            }
        }
        assert!(
            worst < 1e-3,
            "{association:?}: information disagrees with the fixed-target Hessian: {worst:.3e} \
             over {} rows",
            rows.len()
        );
        if !gates.bilinear {
            // nearest-pixel association is piecewise constant: finite differences that
            // re-associate at each perturbed pose only see snapping jumps
            return Ok(());
        }

        // Re-associating at every perturbed pose, weights frozen as Gauss-Newton holds them,
        // so the interpolated target moves with the estimate: this measures what holding the
        // target fixed in the Jacobian costs: worst entry (t_y, t_y) 7.4%, from 8.9% at
        // h = 1e-3 to 7.4% at 3e-5, so not step noise. Modelling the target's motion needs
        // derivatives of the noisy maps; it is left out, so the gap is bounded here, not closed.
        let mut w_fixed = vec![f64::NAN; s0.size().width * s0.size().height];
        for_each_correspondence(s0, t0, &rot, &t, &gates, |c| {
            w_fixed[c.src_idx] = row_weight(c.r, c.sigma, gates.huber_delta);
        });
        let reassoc = |xi: &[f64; 6]| -> f64 {
            let d = SO3F64::exp(Vec3F64::new(xi[0], xi[1], xi[2])).matrix();
            let (rp, tp) = (d * rot, d * t + Vec3F64::new(xi[3], xi[4], xi[5]));
            let mut e = 0.0;
            for_each_correspondence(s0, t0, &rp, &tp, &gates, |c| {
                let w = w_fixed[c.src_idx];
                if w.is_finite() {
                    e += 0.5 * w * c.r * c.r;
                }
            });
            e
        };
        let mut worst_re = 0.0;
        for hh in [1e-3, 3e-4, 1e-4, 3e-5] {
            let (mut worst_h, mut worst_at): (f64, _) = (0.0, (0, 0));
            for j in 0..6 {
                for k in 0..6 {
                    let at = |sj: f64, sk: f64| {
                        let mut xi = [0.0; 6];
                        xi[j] += sj * hh;
                        xi[k] += sk * hh;
                        reassoc(&xi)
                    };
                    let fd = (at(1.0, 1.0) - at(1.0, -1.0) - at(-1.0, 1.0) + at(-1.0, -1.0))
                        / (4.0 * hh * hh);
                    let err = (fd - info[(j, k)]).abs() / (info[(j, j)] * info[(k, k)]).sqrt();
                    if err > worst_h {
                        (worst_h, worst_at) = (err, (j, k));
                    }
                }
            }
            eprintln!("{association:?} re-associating FD, h={hh:.0e}: worst {worst_h:.3e} at {worst_at:?}");
            if hh == 1e-4 {
                worst_re = worst_h;
            }
        }
        assert!(
            worst_re < 0.1,
            "{association:?}: fixed-target information is off the re-associated cost by \
             {worst_re:.3e}"
        );
        Ok(())
    }

    /// Error of an estimate against the truth in the solver's tangent: the estimate is
    /// `exp(xi)` on the left of the truth, `xi = [log(R R_gt^T), t - R R_gt^T t_gt]`.
    fn tangent_error(rot: &Mat3F64, t: &Vec3F64, rot_gt: &Mat3F64, t_gt: &Vec3F64) -> Vec6F64 {
        let rr = *rot * rot_gt.transpose();
        let w = SO3F64::from_matrix(&rr).log();
        let v = *t - rr * *t_gt;
        Vec6F64::new(w.x, w.y, w.z, v.x, v.y, v.z)
    }

    /// Deterministic standard normal samples (splitmix64 and Box-Muller).
    struct Gauss(u64);

    impl Gauss {
        fn uniform(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn sample(&mut self) -> f64 {
            let (u1, u2) = (self.uniform(), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// The information is a covariance to trust: the NEES of the estimate over independent
    /// noisy frames must average the 6 of a chi-square with 6 degrees of freedom.
    ///
    /// Against the truth on a moving pair, most of the NEES is there with no noise at all: the
    /// noise-free solve is off by 0.30 mm with nearest-pixel snapping and 0.07 mm with bilinear
    /// sampling, many sigmas of a 57k-row solve. The information describes the depth noise, not that bias, so the noise is
    /// judged against the truth where the bias is zero (identical poses), and against the
    /// noise-free solve where it is not.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "several minutes in a debug build; runs in the release test job"
    )]
    fn nees_is_six_on_noisy_stereo_depth() -> Result<(), Box<dyn std::error::Error>> {
        for association in [
            Association::Nearest,
            Association::Bilinear,
            Association::Hybrid,
        ] {
            nees_for(association)?;
        }
        Ok(())
    }

    /// One association of [`nees_is_six_on_noisy_stereo_depth`]; the same noise seed for each.
    fn nees_for(association: Association) -> Result<(), Box<dyn std::error::Error>> {
        let (camera, scene) = (test_camera(), Scene::corner_and_sphere());
        let criteria = IcpPlaneCriteria {
            depth_noise: TEST_STEREO,
            association,
            ..Default::default()
        };
        let mut g = Gauss(7);
        let mut noisy = |clean: &Image<f32, 1>| {
            let s = clean.as_slice().iter().map(|&z| {
                if z > 0.0 {
                    z + (TEST_STEREO.depth_sigma(z as f64) * g.sample()) as f32
                } else {
                    0.0
                }
            });
            Image::<f32, 1>::new(clean.size(), s.collect())
        };
        let axis = Vec3F64::new(0.3, 1.0, 0.2).normalize();
        let moving = (
            SO3F64::exp(axis * 1f64.to_radians()).matrix(),
            Vec3F64::new(0.02, 0.0, 0.01),
        );
        let trials = 40;
        let still = (Mat3F64::IDENTITY, Vec3F64::ZERO);
        for (name, (rot_wc, t_wc)) in [("static", still), ("moving", moving)] {
            let (rot_gt, t_gt) = (rot_wc.transpose(), -(rot_wc.transpose() * t_wc));
            let clean_src = depth_m(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
            let clean_tgt = depth_m(&scene, &rot_wc, &t_wc)?;
            let clean = icp_from_identity(
                &RgbdPyramid::from_depth_m(&clean_src, &camera, 3)?,
                &RgbdPyramid::from_depth_m(&clean_tgt, &camera, 3)?,
                &criteria,
            )?;
            let e0 = tangent_error(&clean.rotation, &clean.translation, &rot_gt, &t_gt);
            let (mut vs_truth, mut vs_clean) = (0.0, 0.0);
            for _ in 0..trials {
                let src = RgbdPyramid::from_depth_m(&noisy(&clean_src)?, &camera, 3)?;
                let tgt = RgbdPyramid::from_depth_m(&noisy(&clean_tgt)?, &camera, 3)?;
                let r = icp_from_identity(&src, &tgt, &criteria)?;
                assert_eq!(r.num_degenerate, 0);
                let e = tangent_error(&r.rotation, &r.translation, &rot_gt, &t_gt);
                let d = e - e0;
                vs_truth += e.dot(&(r.information * e)) / trials as f64;
                vs_clean += d.dot(&(r.information * d)) / trials as f64;
            }
            let (w0, v0) = (e0.fixed_rows::<3>(0).norm(), e0.fixed_rows::<3>(3).norm());
            eprintln!(
                "NEES {association:?} {name}: vs truth {vs_truth:.2}, vs noise-free solve \
                 {vs_clean:.2}; noise-free bias {w0:.2e} rad / {:.3} mm",
                v0 * 1e3
            );
            // chi2(6): mean 6, sd of a 40-trial mean ~0.55
            assert!(
                (4.0..9.0).contains(&vs_clean),
                "{association:?} {name}: NEES {vs_clean:.2} about the noise-free solve is not consistent with 6"
            );
            if name == "static" {
                assert!(
                    (4.0..9.0).contains(&vs_truth),
                    "{association:?} static: NEES {vs_truth:.2} against the truth is not consistent with 6"
                );
            }
        }
        Ok(())
    }

    /// `PriorRows` must add `J^T info J` and `J^T info e` with `J` the derivative of its error
    /// under the solver's left perturbation `exp(xi) T`, also far from the prior and near pi,
    /// where a first-order identity Jacobian or a sign-flipped log would break the match.
    #[test]
    fn prior_rows_match_finite_differences() {
        // symmetric positive definite, coupling rotation and translation
        let l = Mat6F64::from_fn(|i, k| match i.cmp(&k) {
            std::cmp::Ordering::Equal => 2.0 + i as f64,
            std::cmp::Ordering::Greater => 0.3 * (i + 2 * k) as f64 / 7.0,
            std::cmp::Ordering::Less => 0.0,
        });
        let axis = Vec3F64::new(0.2, -0.5, 0.84).normalize();
        for angle in [1e-3, 1.0, 3.1] {
            let prior = PriorRows {
                rotation: SO3F64::exp(Vec3F64::new(0.1, 0.4, -0.2)).matrix(),
                translation: Vec3F64::new(0.3, -0.1, 0.2),
                info: l * l.transpose(),
            };
            let rot = SO3F64::exp(axis * angle).matrix() * prior.rotation;
            let t = Vec3F64::new(-0.2, 0.05, 0.4);
            let (e, _) = prior.error_and_jacobian(&rot, &t);
            // the error is the left difference itself, sign included
            let e_r = Vec3F64::new(e[0], e[1], e[2]);
            assert!((e_r - axis * angle).length() < 1e-9, "{angle}: {e_r:?}");

            let (mut a, mut b) = (Mat6F64::zeros(), Vec6F64::zeros());
            prior.add_to(&rot, &t, &mut a, &mut b);
            let error_at = |xi: &Vec6F64| {
                let d = SO3F64::exp(Vec3F64::new(xi[0], xi[1], xi[2])).matrix();
                let dt = Vec3F64::new(xi[3], xi[4], xi[5]);
                prior.error_and_jacobian(&(d * rot), &(d * t + dt)).0
            };
            let h = 1e-6;
            let mut jac = Mat6F64::zeros();
            for k in 0..6 {
                let mut xi = Vec6F64::zeros();
                xi[k] = h;
                let plus = error_at(&xi);
                xi[k] = -h;
                jac.set_column(k, &((plus - error_at(&xi)) / (2.0 * h)));
            }
            let a_fd = jac.transpose() * prior.info * jac;
            let b_fd = jac.transpose() * prior.info * e;
            assert!((a - a_fd).amax() < 1e-6 * a.amax(), "{angle}: A off");
            assert!((b - b_fd).amax() < 1e-6 * b.amax(), "{angle}: b off");
        }
    }

    /// Bad sigmas and non-rotations in a rotation prior are errors, not weights.
    #[test]
    fn invalid_rotation_prior_is_an_error() -> Result<(), Box<dyn std::error::Error>> {
        let pyr = pyramid(
            &Scene::corner_and_sphere(),
            &Mat3F64::IDENTITY,
            &Vec3F64::ZERO,
        )?;
        let with = |rotation: Mat3F64, sigma_rad: f64| IcpPlaneCriteria {
            rotation_prior: Some(RotationPrior {
                rotation,
                sigma_rad,
            }),
            ..Default::default()
        };
        let id = Mat3F64::IDENTITY;
        let mut nan = id;
        nan.x_axis.y = f64::NAN;
        let reflection = Mat3F64::from_diagonal(Vec3F64::new(1.0, 1.0, -1.0));
        let cases = [
            (with(id, 0.0), "rotation_prior.sigma_rad"),
            (with(id, -1e-3), "rotation_prior.sigma_rad"),
            (with(id, f64::NAN), "rotation_prior.sigma_rad"),
            (with(id, f64::INFINITY), "rotation_prior.sigma_rad"),
            // positive, but 1 / sigma^2 overflows
            (with(id, 1e-200), "rotation_prior.sigma_rad"),
            (with(nan, 1e-3), "rotation_prior.rotation"),
            (
                with(Mat3F64::from_diagonal(Vec3F64::new(2.0, 2.0, 2.0)), 1e-3),
                "rotation_prior.rotation",
            ),
            (with(reflection, 1e-3), "rotation_prior.rotation"),
        ];
        for (c, expected) in &cases {
            let result = icp_from_identity(&pyr, &pyr, c);
            assert!(
                matches!(result, Err(RgbdIcpError::InvalidCriteria { field, .. }) if field == *expected),
                "{c:?}: {result:?}"
            );
        }
        // a rotation rounded through f32 is still a rotation
        let rounded = SO3F64::exp(Vec3F64::new(0.3, -0.2, 0.1))
            .matrix()
            .as_mat3()
            .as_dmat3();
        icp_from_identity(&pyr, &pyr, &with(rounded.into(), 1e-3))?;
        Ok(())
    }

    /// A wall with a sphere centred on the optical axis in front of it: both are invariant
    /// under roll, so rotation about the viewing axis is unobserved, while the sphere's normals
    /// pin all three translations.
    fn roll_blind_scene() -> Scene {
        Scene {
            planes: vec![Plane {
                normal: Vec3F64::new(0.0, 0.0, 1.0),
                d: 2.5,
            }],
            spheres: vec![Sphere {
                center: Vec3F64::new(0.0, 0.0, 1.8),
                radius: 0.55,
            }],
        }
    }

    /// A rotation prior must not flatter the figures it is reported alongside.
    ///
    /// The prior carries the roll the geometry cannot see, but the conditioning, the
    /// observability, the degenerate directions and the information must still describe the
    /// data alone: the same as a prior-free final pass at the returned pose. Blocks coupled
    /// through elimination would otherwise let a rotation prior lift the translation pivots too,
    /// and silently disarm a gate on unobservable geometry.
    #[test]
    fn rotation_prior_does_not_inflate_reported_conditioning(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let scene = roll_blind_scene();
        let roll = SO3F64::exp(Vec3F64::new(0.0, 0.0, 3f64.to_radians())).matrix();
        let t_wc = Vec3F64::new(0.01, 0.0, 0.0);
        let src = pyramid(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
        let tgt = pyramid(&scene, &roll, &t_wc)?;
        let rot_gt = roll.transpose();
        let criteria = IcpPlaneCriteria {
            rotation_prior: Some(RotationPrior {
                rotation: rot_gt,
                sigma_rad: 0.1f64.to_radians(),
            }),
            ..Default::default()
        };
        let with = icp_from_identity(&src, &tgt, &criteria)?;
        // the prior did carry the roll
        assert!(rotation_error_deg(&with.rotation, &rot_gt) < 0.1);
        assert!(
            with.rotation_conditioning < 1e-3 && with.observability < 1e-3,
            "prior inflated the reported conditioning: {with:?}"
        );

        // the figures equal a prior-free pass at the returned pose
        let (gates, _) = check_criteria(&criteria)?;
        let (s0, t0) = (&src.levels[0], &tgt.levels[0]);
        let eqs = accumulate_level(s0, t0, &with.rotation, &with.translation, &gates, None);
        let split = split_degenerate(&eqs.a, with.median_depth_m, criteria.degeneracy_rtol)?;
        assert_eq!(split.num_degenerate, with.num_degenerate);
        assert_eq!(with.information, split.observed(&eqs.a));
        let (rot_cond, trans_cond) = block_conditioning(&eqs.a);
        assert_eq!(
            (with.rotation_conditioning, with.translation_conditioning),
            (rot_cond, trans_cond)
        );
        Ok(())
    }

    /// The rotation prior must rescue weak geometry without harming good geometry, from a prior
    /// that is itself wrong: a real gyro carries bias, noise and a camera misalignment, and a
    /// prior fed the exact answer would prove nothing.
    #[test]
    fn rotation_prior_rescues_weak_geometry_without_harming_good(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let roll = SO3F64::exp(Vec3F64::new(0.0, 0.0, 3f64.to_radians())).matrix();
        let t_wc = Vec3F64::new(0.01, 0.0, 0.0);
        // 0.15 deg off about an oblique axis, well beyond a MEMS gyro over one frame
        let prior_err = 0.15f64.to_radians();
        let tilt = Vec3F64::new(0.3, 0.6, 0.74).normalize() * prior_err;
        let prior_rot = (SO3F64::exp(tilt).matrix() * roll).transpose();
        let rot_gt = roll.transpose();
        let mut errors = Vec::new();
        for scene in [roll_blind_scene(), Scene::corner_and_sphere()] {
            let src = pyramid(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
            let tgt = pyramid(&scene, &roll, &t_wc)?;
            let solve = |rotation_prior| -> Result<f64, RgbdIcpError> {
                let c = IcpPlaneCriteria {
                    rotation_prior,
                    ..Default::default()
                };
                let r = icp_from_identity(&src, &tgt, &c)?;
                Ok(rotation_error_deg(&r.rotation, &rot_gt))
            };
            let prior = RotationPrior {
                rotation: prior_rot,
                sigma_rad: prior_err,
            };
            errors.push((solve(None)?, solve(Some(prior))?));
        }
        let ((weak_without, weak_with), (rich_without, rich_with)) = (errors[0], errors[1]);
        // weak: without the prior the roll gets no update and stays 3 deg off; with it, the
        // answer lands near the prior's own accuracy
        assert!(
            weak_without > 2.5 && weak_with < 0.3,
            "weak: {weak_without:.3} deg without, {weak_with:.3} with"
        );
        // rich: the data outvote the wrong prior
        assert!(
            rich_with < rich_without.max(0.05) * 3.0,
            "rich: {rich_without:.4} deg without, {rich_with:.4} with"
        );
        Ok(())
    }

    /// A wall gives exactly its three blind directions zero information, and the solve takes
    /// the pose prior along them instead of leaving them where they started.
    #[test]
    fn wall_takes_prior_along_its_null_space() -> Result<(), Box<dyn std::error::Error>> {
        let wall = Scene {
            planes: vec![Plane {
                normal: Vec3F64::new(0.0, 0.0, 1.0),
                d: 2.0,
            }],
            spheres: vec![],
        };
        // roll and in-plane slide are invisible, the approach along z is not
        let rot_wc = SO3F64::exp(Vec3F64::new(0.0, 0.0, 1f64.to_radians())).matrix();
        let t_wc = Vec3F64::new(0.02, -0.01, 0.005);
        let rot_gt = rot_wc.transpose();
        let t_gt = -(rot_gt * t_wc);
        let src = pyramid(&wall, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
        let tgt = pyramid(&wall, &rot_wc, &t_wc)?;

        let check_null_space = |r: &IcpPlaneResult| {
            assert_eq!(r.num_degenerate, 3, "wall must have three blind directions");
            let dirs = &r.degenerate_directions;
            for d in &dirs[..3] {
                // only rotation about the normal and in-plane translation
                assert!(d[0].abs() + d[1].abs() + d[5].abs() < 1e-6, "{d:?}");
                let info = d.dot(&(r.information * d)) / r.information.amax();
                assert!(info.abs() < 1e-12, "information along {d:?}: {info:.3e}");
            }
            let basis = Mat3F64::from_cols(
                Vec3F64::new(dirs[0][2], dirs[0][3], dirs[0][4]),
                Vec3F64::new(dirs[1][2], dirs[1][3], dirs[1][4]),
                Vec3F64::new(dirs[2][2], dirs[2][3], dirs[2][4]),
            );
            let det = basis.determinant();
            assert!(
                det.abs() > 0.5,
                "directions do not span w_z, v_x, v_y: {det}"
            );
            assert!(r.information[(5, 5)] > 0.0, "v_z is observed");
        };

        // no prior: the blind directions get no update at all, the observed one is solved
        let bare = icp_from_identity(&src, &tgt, &IcpPlaneCriteria::default())?;
        check_null_space(&bare);
        assert!(bare.translation.x.abs() < 1e-6 && bare.translation.y.abs() < 1e-6);
        assert!(SO3F64::from_matrix(&bare.rotation).log().z.abs() < 1e-6);
        assert!((bare.translation.z - t_gt.z).abs() < 1e-4, "{bare:?}");

        // a prior 3 mm off along x: the blind directions follow it, z still follows the data
        let prior_t = t_gt + Vec3F64::new(0.003, 0.0, 0.004);
        let (var_r, var_t) = (0.2f64.to_radians().powi(2), 0.01f64.powi(2));
        let diag = Vec6F64::new(var_r, var_r, var_r, var_t, var_t, var_t);
        let criteria = IcpPlaneCriteria {
            pose_prior: Some(PosePrior {
                rotation: rot_gt,
                translation: prior_t,
                covariance: Mat6F64::from_diagonal(&diag),
            }),
            ..Default::default()
        };
        let with = icp_from_identity(&src, &tgt, &criteria)?;
        check_null_space(&with);
        let e = tangent_error(&with.rotation, &with.translation, &rot_gt, &prior_t);
        assert!(e[3].abs() < 1e-5 && e[4].abs() < 1e-5, "slide: {e:?}");
        assert!(e[2].abs() < 1e-5, "roll: {e:?}");
        // the prior is 4 mm off along the observed z too: the data must win there
        let dz = with.translation.z - t_gt.z;
        assert!(dz.abs() < 2e-4, "z follows the 4 mm-off prior: {dz}");
        Ok(())
    }

    /// A rotation prior is a pose prior with no translation information: the same weighted
    /// rows, so the same answer.
    #[test]
    fn rotation_prior_equals_pose_prior_without_translation(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let scene = roll_blind_scene();
        let roll = SO3F64::exp(Vec3F64::new(0.0, 0.0, 3f64.to_radians())).matrix();
        let src = pyramid(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
        let tgt = pyramid(&scene, &roll, &Vec3F64::new(0.01, 0.0, 0.0))?;
        let rot_gt = roll.transpose();
        let sigma = 0.1f64.to_radians();
        let a = icp_from_identity(
            &src,
            &tgt,
            &IcpPlaneCriteria {
                rotation_prior: Some(RotationPrior {
                    rotation: rot_gt,
                    sigma_rad: sigma,
                }),
                ..Default::default()
            },
        )?;
        // translation variance (1 km)^2: its information is negligible, not zero
        let diag = Vec6F64::new(sigma * sigma, sigma * sigma, sigma * sigma, 1e6, 1e6, 1e6);
        let b = icp_from_identity(
            &src,
            &tgt,
            &IcpPlaneCriteria {
                pose_prior: Some(PosePrior {
                    rotation: rot_gt,
                    // irrelevant at that variance
                    translation: Vec3F64::new(5.0, -3.0, 1.0),
                    covariance: Mat6F64::from_diagonal(&diag),
                }),
                ..Default::default()
            },
        )?;
        assert!(rotation_error_deg(&a.rotation, &b.rotation) < 1e-4);
        assert!((a.translation - b.translation).length() < 1e-6);
        // and the prior never shows in the reported conditioning
        assert!(a.rotation_conditioning < 1e-3 && b.rotation_conditioning < 1e-3);
        Ok(())
    }

    /// Bad rotations, translations and covariances in a pose prior are errors, not weights;
    /// both priors at once are not.
    #[test]
    fn invalid_pose_prior_is_an_error() -> Result<(), Box<dyn std::error::Error>> {
        let pyr = pyramid(
            &Scene::corner_and_sphere(),
            &Mat3F64::IDENTITY,
            &Vec3F64::ZERO,
        )?;
        let with =
            |rotation: Mat3F64, translation: Vec3F64, covariance: Mat6F64| IcpPlaneCriteria {
                pose_prior: Some(PosePrior {
                    rotation,
                    translation,
                    covariance,
                }),
                ..Default::default()
            };
        let (id, zero) = (Mat3F64::IDENTITY, Vec3F64::ZERO);
        let cov = Mat6F64::from_diagonal(&Vec6F64::from_element(1e-4));
        let edit = |i: usize, k: usize, v: f64| {
            let mut c = cov;
            c[(i, k)] = v;
            c
        };
        let reflection = Mat3F64::from_diagonal(Vec3F64::new(1.0, 1.0, -1.0));
        let cases = [
            (with(reflection, zero, cov), "pose_prior.rotation"),
            (
                with(id, Vec3F64::new(0.0, f64::NAN, 0.0), cov),
                "pose_prior.translation",
            ),
            (
                with(id, Vec3F64::new(f64::INFINITY, 0.0, 0.0), cov),
                "pose_prior.translation",
            ),
            (
                with(id, zero, edit(2, 2, f64::NAN)),
                "pose_prior.covariance",
            ),
            (with(id, zero, edit(0, 4, 1e-5)), "pose_prior.covariance"),
            (with(id, zero, edit(0, 0, -1e-4)), "pose_prior.covariance"),
            (with(id, zero, edit(5, 5, 0.0)), "pose_prior.covariance"),
            (with(id, zero, Mat6F64::zeros()), "pose_prior.covariance"),
            // positive definite, but the inverse overflows
            (with(id, zero, edit(3, 3, 1e-320)), "pose_prior.covariance"),
        ];
        for (c, expected) in &cases {
            let result = icp_from_identity(&pyr, &pyr, c);
            assert!(
                matches!(result, Err(RgbdIcpError::InvalidCriteria { field, .. }) if field == *expected),
                "{c:?}: {result:?}"
            );
        }
        // coupled, symmetric and positive definite passes, with a rotation prior beside it
        let mut coupled = edit(0, 4, 2e-5);
        coupled[(4, 0)] = 2e-5;
        let mut both = with(id, zero, coupled);
        both.rotation_prior = Some(RotationPrior {
            rotation: id,
            sigma_rad: 1e-3,
        });
        let r = icp_from_identity(&pyr, &pyr, &both)?;
        assert!(rotation_error_deg(&r.rotation, &id) < 1e-6 && r.translation.length() < 1e-9);
        Ok(())
    }

    /// Every field of two results, compared bit for bit.
    fn assert_bit_identical(a: &IcpPlaneResult, b: &IcpPlaneResult) {
        let bits = |r: &IcpPlaneResult| {
            let mut v: Vec<u64> = r.rotation.to_cols_array().map(f64::to_bits).to_vec();
            v.extend(r.translation.to_array().map(f64::to_bits));
            v.extend(r.information.iter().map(|x| x.to_bits()));
            v.extend(
                r.degenerate_directions
                    .iter()
                    .flatten()
                    .map(|x| x.to_bits()),
            );
            let scalars = [
                r.rmse,
                r.inlier_fraction,
                r.overlap_fraction,
                r.rotation_conditioning,
                r.translation_conditioning,
                r.observability,
                r.median_depth_m,
            ];
            v.extend(scalars.map(f64::to_bits));
            let counts = [
                r.iterations,
                r.num_inliers,
                r.num_associated,
                r.num_degenerate,
            ];
            v.extend(counts.map(|c| c as u64));
            v
        };
        assert_eq!(bits(a), bits(b));
    }

    /// A reused workspace carries nothing from one call to the next: each call equals a fresh
    /// one, whichever frame came before it.
    #[test]
    fn workspace_matches_fresh_call() -> Result<(), Box<dyn std::error::Error>> {
        let scene = Scene::corner_and_sphere();
        let src = pyramid(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
        let rot_wc = SO3F64::exp(Vec3F64::new(0.015, -0.01, 0.02)).matrix();
        let moved = pyramid(&scene, &rot_wc, &Vec3F64::new(0.02, -0.01, 0.015))?;
        let mut workspace = IcpWorkspace::default();
        // Hybrid runs both samplers
        let criteria = IcpPlaneCriteria {
            association: Association::Hybrid,
            ..Default::default()
        };
        for tgt in [&moved, &src, &moved] {
            let fresh = icp_from_identity(&src, tgt, &criteria)?;
            let (r, t) = (Mat3F64::IDENTITY, Vec3F64::ZERO);
            let reused = icp_projective_plane_with(&mut workspace, &src, tgt, &r, &t, &criteria)?;
            assert_bit_identical(&fresh, &reused);
        }
        Ok(())
    }
}
