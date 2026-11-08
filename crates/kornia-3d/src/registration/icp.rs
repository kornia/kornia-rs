//! Projective point-to-plane ICP on RGBD pyramids.
//!
//! Estimates the rigid transform `T_target_source` (`p_tgt = R * p_src + t`)
//! between two depth frames given as [`RgbdPyramid`]s, coarse-to-fine.
//! Correspondences come from projective association (no kd-tree): each source
//! vertex is transformed by the current estimate, projected into the target
//! grid with the target level intrinsics, and matched to the nearest pixel's
//! vertex/normal.
//!
//! Per correspondence the point-to-plane residual is `r = n_t . (T*p_s - v_t)`
//! with Jacobian row `[(p' x n_t)^T, n_t^T]` where `p' = T*p_s` (left
//! perturbation `T <- exp([w, t]) * T`). The target vertex and normal are
//! bilinearly interpolated at the projection of `p'` and held fixed within an
//! iteration. Maps stay `f32`; `J^T W J` / `J^T W r`
//! are accumulated in `f64` and the 6x6 system is solved by Cholesky.
//!
//! Each row is weighted by `1 / sigma^2`, the variance of its residual under
//! [`IcpPlaneCriteria::depth_noise`] (or a per-pixel sigma map, see
//! [`RgbdLevel::depth_sigmas`]), times a Huber weight on `r / sigma`. The final
//! data-only `J^T W J` is returned as [`IcpPlaneResult::information`].
//!
//! Tangent convention, shared by the update, the priors and the reported
//! information: `xi = [w, v]` (rotation first, radians then metres) acts on the
//! LEFT, `R <- exp(w) R`, `t <- exp(w) t + v`.
//!
//! A system that does not constrain all six DoF still gets an answer: directions
//! whose eigenvalue is negligible get zero information and take their update
//! from a prior, or none (see [`IcpPlaneCriteria::degeneracy_rtol`]).
//!
//! # Input size: register on decimated depth, map at full resolution
//!
//! The cost of a call is linear in the pixels of pyramid level 0, so the real-time budget
//! (20 ms on one Jetson Orin core at 15 Hz depth, about 8 ms on one x86 laptop core) is a
//! contract on the input size, not on the sensor: **level 0 is at most 160x100 (16k pixels)**.
//! Measured per call (pyramid of the new frame plus every level, 3 levels, on-CPU, x86 laptop
//! under load, `benches/bench_icp.rs`): 320x200 costs 4.3x 160x100 (43.6 vs 10.2 ms), and a
//! 320x200 frame decimated to 160x100 costs 12.6 ms, the halving included.
//!
//! A coarse-to-fine alternative that keeps a full-resolution top level (160x100 and coarser,
//! then 2 iterations at 320x200) costs 32.0 ms on the same run: every pass at the top level is
//! a 320x200 pass, and the pyramid still builds at full resolution. It is not the contract.
//!
//! Larger depth images are decimated before ICP, as RGB-D odometry systems do: build the
//! source and the keyframe pyramids with [`RgbdPyramid::rebuild_decimated_from_depth_mm`] and
//! [`super::rgbd::halvings_to_fit`]`(width, 160)`, and keep the full-resolution depth for map
//! integration, where the detail is worth its cost. The decimation is the pyramid's own
//! validity- and discontinuity-aware halving, so it neither fills holes nor averages across
//! depth edges. What it gives up is the finest level's sub-pixel detail: the snapping bias of
//! [`Association::Nearest`] doubles in metres per pixel.

use super::rgbd::{is_valid_normal, is_valid_vertex, RgbdIcpError, RgbdLevel, RgbdPyramid};
use nalgebra::Matrix6;

mod simd;

/// Parameters of [`icp_projective_plane`].
#[derive(Debug, Clone)]
pub struct IcpPlaneCriteria {
    /// Gauss-Newton iterations per pyramid level, coarsest first. Levels
    /// beyond the list reuse the last entry; an empty list runs no iterations
    /// (the result is the evaluated initial guess).
    pub iters_per_level: Vec<usize>,
    /// Convergence threshold on the twist update norm `sqrt(|w|^2 + |dt|^2)`
    /// (rad / metres); a level stops early once the update falls below it.
    pub update_tolerance: f64,
    /// Reject correspondences further apart than this in metres.
    pub max_dist_m: f64,
    /// Reject correspondences whose (rotated) source and target normals
    /// disagree by more than this angle in radians.
    pub max_normal_angle_rad: f64,
    /// Huber threshold on the NORMALISED residual `r / sigma` (dimensionless): rows beyond it
    /// get weight `delta / |r / sigma|` on top of `1 / sigma^2`. Replaces the former metric
    /// `huber_delta_m`; the defaults (`2.0` here, `sigma = 0.01` m in
    /// [`DepthNoise::Constant`]) reproduce its old 0.02 m threshold exactly.
    pub huber_delta: f64,
    /// How a projected source point picks its target vertex and normal. Default
    /// [`Association::Nearest`].
    pub association: Association,
    /// Depth noise model that sets each row's `sigma`. A frame's per-pixel sigma map
    /// ([`RgbdLevel::depth_sigmas`]) takes precedence over it point by point.
    pub depth_noise: DepthNoise,
    /// Optional independent estimate of the rotation, e.g. an integrated gyro.
    ///
    /// Depth geometry does not always determine rotation: on a swept view the rotation block of
    /// the normal equations collapses and the solve returns rotations wrong by degrees per frame
    /// (measured: 1.4-4.4 deg against a 3.0 deg truth, eventually freezing at zero) while the
    /// residual and the inlier fraction stay perfect. A gyro measures exactly that quantity, and
    /// over a single frame interval it is roughly an order of magnitude more accurate than the
    /// degraded solve, so it can carry the rotation while the geometry carries the translation.
    ///
    /// Regularises, never overrides: it enters as three weighted rows, so ample geometry
    /// outvotes it and weak geometry defers to it.
    ///
    /// Equivalent to a [`PosePrior`] with infinite translation variance; both may be set, and
    /// their rows add.
    pub rotation_prior: Option<RotationPrior>,
    /// Optional 6-DoF prior on the whole transform, e.g. the fused pose's delta.
    ///
    /// Enters as six weighted rows like [`Self::rotation_prior`], so it shapes the answer only
    /// where the data are weak, and it carries the update along directions the data cannot
    /// observe at all. It never enters [`IcpPlaneResult::information`] or the conditioning.
    pub pose_prior: Option<PosePrior>,
    /// Relative eigenvalue threshold below which a direction of the data-only `J^T W J` counts
    /// as unobserved.
    ///
    /// The eigen-decomposition runs with the rotation rows scaled by the median inlier depth
    /// (radians to metres of arc), so rotation and translation compare in one unit. A direction
    /// whose eigenvalue is at most `degeneracy_rtol` times the largest gets zero information in
    /// the result, and its update comes from the priors, or is zero without one. Set it to sit
    /// between rounding noise on an exact null space and the weakest geometry you still want
    /// solved; see the measurements on [`DEFAULT_DEGENERACY_RTOL`].
    pub degeneracy_rtol: f64,
}

/// Projective association of [`icp_projective_plane`]: how the target vertex and normal are
/// read at a source point's projection.
///
/// Measured on the same set (corner + sphere for bias and NEES; flux-odom's replay tests, run
/// against this crate, for tracking):
///
/// | | zero-noise bias, moving pair | NEES moving vs truth / vs noise-free | break recovery, no IMU | sweep: last exact / first held |
/// |---|---|---|---|---|
/// | Nearest | 1.85e-4 rad, 0.312 mm | 226 / 8.2 | 26/30 frames | 1.72e-2 / 7.16e-4 |
/// | Bilinear | 1.80e-5 rad, 0.068 mm | 117 / 6.7 | 17/30 | 5.34e-2 / 6.93e-4 |
/// | Hybrid | 1.80e-5 rad, 0.068 mm | 117 / 6.7 | 21/30 | 1.72e-2 / 7.31e-4 |
///
/// The theory: bilinear smooths the target, which far from the keyframe biases the coarse
/// alignment, while at the converged finest level it only removes snapping. Hybrid confirms it
/// on bias and on the sweep onset; it recovers the break range only partly (21 vs 26 of 30).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Association {
    /// The pixel the projection rounds to. Biased by up to half a pixel of snapping, widest
    /// convergence range.
    #[default]
    Nearest,
    /// Bilinear interpolation of the four surrounding pixels, renormalised normal; dropped when
    /// any of them is invalid or they straddle a depth discontinuity.
    Bilinear,
    /// Nearest on the coarse levels and on the finest until the update norm falls under
    /// [`HYBRID_SWITCH_NORM`], then bilinear for up to as many iterations again.
    Hybrid,
}

/// Twist update norm (rad / metres, as [`IcpPlaneCriteria::update_tolerance`]) under which
/// [`Association::Hybrid`] switches the finest level to bilinear.
///
/// 1e-3 is 1 mm / 0.06 deg: well inside the half-pixel snapping it is there to remove (~5 mm at
/// 2 m and fx = 200) and well outside the default 1e-6 convergence tolerance, so the switch
/// always happens before a level is declared converged. Not tuned.
pub const HYBRID_SWITCH_NORM: f64 = 1e-3;

/// Default of [`IcpPlaneCriteria::degeneracy_rtol`].
///
/// Same value as the Cholesky guard's `PIVOT_RTOL`. Measured smallest/largest eigenvalue of the
/// depth-scaled data matrix on the synthetic 320x180 scenes: a bare wall 0 (three directions),
/// a wall with a 6 cm bump 5.1e-8, a roll-invariant sphere-on-wall 2.7e-5, corner + sphere
/// 2.8e-2. So the default flags only an exact null space and still solves the bump, as the
/// solver always did. Real sensor noise lifts a wall's null space far above zero; to treat
/// real weak geometry as degenerate, calibrate this on real data.
pub const DEFAULT_DEGENERACY_RTOL: f64 = 1e-8;

/// Depth noise model for the row weights of [`icp_projective_plane`].
///
/// Each point's depth sigma (`z`, metres) moves it along its viewing ray, so a row's residual
/// sigma is `sqrt((n . r_s)^2 sigma_s^2 + (n . r_t)^2 sigma_t^2)` with `r = p / z` the ray of
/// the source point (rotated into the target) and of the target point, and `n` the target
/// normal. [`Self::Constant`] with no sigma map on either frame is the exception: it is a
/// residual sigma used as-is, the uniform weighting this solver always had.
#[derive(Debug, Clone, Copy, PartialEq)]
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
}

/// A 6-DoF prior for [`IcpPlaneCriteria::pose_prior`].
#[derive(Debug, Clone, Copy)]
pub struct PosePrior {
    /// Prior on the source-to-target rotation, same convention as [`IcpPlaneResult::rotation`].
    pub rotation: [[f64; 3]; 3],
    /// Prior on the source-to-target translation in metres.
    pub translation: [f64; 3],
    /// Covariance of the prior's error `xi` in the module's tangent convention
    /// (`[w, v]`, rotation first: rad^2, rad*m, m^2), where the true transform is
    /// `exp(xi)` applied on the left of the prior. Must be symmetric positive definite; scale
    /// it with the interval it spans, as for [`RotationPrior::sigma_rad`].
    pub covariance: [[f64; 6]; 6],
}

/// An external rotation estimate for [`IcpPlaneCriteria::rotation_prior`].
#[derive(Debug, Clone, Copy)]
pub struct RotationPrior {
    /// Prior on the source-to-target rotation, same convention as
    /// [`IcpPlaneResult::rotation`].
    pub rotation: [[f64; 3]; 3],
    /// One-sigma expected error of `rotation`, in radians, OVER THE INTERVAL IT SPANS.
    ///
    /// Scale it with elapsed time rather than passing a per-frame constant: a prior accumulated
    /// across held frames covers a longer interval and has drifted further, and a fixed sigma
    /// would make the constraint overconfident precisely after a stall. For a consumer MEMS gyro,
    /// bias dominates — roughly 0.5 deg/s uncalibrated, so ~0.025 deg over a 50 ms frame.
    ///
    /// The weight is `1/sigma^2`, which puts a 0.05-0.1 deg sigma one to two orders of magnitude
    /// tighter than a collapsed rotation block, and negligible against a healthy one.
    ///
    /// Size it against the sensor's TOTAL error, systematic terms included — not its bias and
    /// noise alone. Two sweeps, measured:
    ///
    /// - varying the true error at fixed sigma: the prior is comfortably ahead at sigma, still
    ///   winning around 1.5x, and loses beyond roughly 2x
    /// - varying sigma at a fixed realistic error: accuracy improved monotonically as sigma
    ///   LOOSENED (0.66 mm at 0.2 deg/s, 0.52 at 2.0, 0.24 at 5.0) with the tracked-frame count
    ///   flat across the whole range
    ///
    /// The second is counter-intuitive and decides the tuning. The dominant error there was a
    /// fixed 3 deg gyro-to-camera misalignment — systematic, not random — so a tight sigma does
    /// not pull the solve toward truth, it pulls it toward a consistently wrong attitude, while a
    /// loose one supplies just enough weight to stop the rotation block collapsing and leaves the
    /// depth free to correct the rest. Tight weighting on a biased measurement buys you the bias.
    /// The flat frame count is the same fact from the other side: a collapsed block carries
    /// almost no information, so even a very loose prior dominates it.
    ///
    /// That ordering inverts once the systematic term is calibrated away. An unbiased gyro should
    /// favour a tighter sigma, so re-measure for that case rather than inheriting this guidance.
    pub sigma_rad: f64,
}

impl Default for IcpPlaneCriteria {
    fn default() -> Self {
        Self {
            iters_per_level: vec![10, 7, 5],
            update_tolerance: 1e-6,
            max_dist_m: 0.10,
            max_normal_angle_rad: 30.0_f64.to_radians(),
            huber_delta: 2.0,
            association: Association::Nearest,
            depth_noise: DepthNoise::Constant { sigma_m: 0.01 },
            rotation_prior: None,
            pose_prior: None,
            degeneracy_rtol: DEFAULT_DEGENERACY_RTOL,
        }
    }
}

/// Result of [`icp_projective_plane`].
///
/// The transformation maps source-frame points into the target frame:
/// `p_tgt = rotation * p_src + translation`.
#[derive(Debug, Clone)]
pub struct IcpPlaneResult {
    /// Estimated rotation matrix (row-major).
    pub rotation: [[f64; 3]; 3],
    /// Estimated translation vector in metres.
    pub translation: [f64; 3],
    /// Root-mean-square point-to-plane residual (metres) of the gated
    /// correspondences at the finest level under the final transform.
    pub rmse: f64,
    /// Gated correspondences divided by *associated* ones at the finest level
    /// under the final transform: match quality alone. Deliberately not divided
    /// by all valid source pixels — that conflates quality with view overlap and
    /// collapses under camera motion, rejecting solves that converged perfectly.
    /// Pair it with [`Self::overlap_fraction`] and [`Self::num_associated`]: a
    /// small patch can match perfectly and still be too little to constrain a
    /// pose.
    pub inlier_fraction: f64,
    /// Source pixels that projected into the target frame onto valid geometry,
    /// before the distance and normal-angle gates. The observability figure: how
    /// much evidence the solve actually had.
    pub num_associated: usize,
    /// Associated pixels divided by valid source pixels: how much of the source
    /// view still overlaps the target. Falls as the camera moves away from the
    /// keyframe and is the signal to re-key, not to reject the solve.
    pub overlap_fraction: f64,
    /// Total Gauss-Newton iterations performed across all levels.
    pub iterations: usize,
    /// Conditioning of the rotation block: weakest pivot over strongest, in `0..=1`.
    pub rotation_conditioning: f64,
    /// Conditioning of the translation block: weakest pivot over strongest, in `0..=1`.
    ///
    /// This is the one that catches a camera sweeping along a wall. As the surface comes to
    /// dominate the view, translation along it stops being constrained while rotation stays
    /// perfectly determined — measured on a sweep: rotation exact to 0.01 deg, translation four
    /// times the truth, with [`Self::rmse`] at 0.4 mm and [`Self::inlier_fraction`] at 1.00
    /// throughout. Residual and inlier fraction cannot see it; they are computed over
    /// correspondences that all agree, on a surface that cannot pin the pose.
    ///
    /// Deliberately a ratio, not an absolute pivot: pivots accumulate over correspondences, so
    /// an absolute floor tracks the inlier count and the scene depth instead of the geometry.
    ///
    /// Calibrate a gate against the SLIDING regime, not against the degenerate extreme. The
    /// frames that corrupt a map sit orders of magnitude above a bare wall, so a threshold placed
    /// to catch outright degeneracy passes every pose that does damage — that mistake has already
    /// been made once here.
    ///
    /// Measured on a 320x180 depth camera sweeping a room at ~43 deg/s, 20 mm of true motion per
    /// frame. Calibrate against the ONSET of failure:
    ///
    /// | | conditioning | published motion |
    /// |---|---|---|
    /// | worst frame still exact | 1.7e-2 | error <= 0.1 mm |
    /// | first frame that slid | 7.2e-4 | 82.8 mm for 20 mm |
    ///
    /// A 24x separation with nothing in it, so a gate anywhere near the 3.5e-3 midpoint behaves
    /// the same. Take the threshold from that transition only: once a pose is wrong, every later
    /// frame's conditioning is measured against a corrupted estimate and is not independent
    /// evidence — averaging those in understates the separation.
    ///
    /// Gate on [`Self::observability`], the weaker block, not on this field alone. Measured on
    /// the same sweep, translation collapses first (7.2e-4 while rotation holds at 2.96e-2) and
    /// then the blocks SWAP: translation recovers to 1e-1 while rotation falls to 2e-3, with the
    /// published motion decaying to half of truth throughout. A gate on either block alone passes
    /// one half of that failure.
    ///
    pub translation_conditioning: f64,
    /// The weaker of the two blocks: a single scalar for callers that just need a gate.
    pub observability: f64,
    /// Data-only information `J^T W J` (6x6, tangent order `[w, v]`, see the module docs) at
    /// the finest level under the final transform, rows weighted by `1 / sigma^2` and Huber.
    ///
    /// Priors never enter it, and it is zero along [`Self::degenerate_directions`]. Its inverse,
    /// where it exists, is the covariance of the estimate's error `xi`, where the estimate is
    /// `exp(xi)` applied on the left of the truth. It is only as calibrated as the sigmas
    /// behind it: the default [`DepthNoise::Constant`] is a weighting, not a sensor model.
    pub information: [[f64; 6]; 6],
    /// Unit tangent directions (`[w, v]`) with zero information, in rows
    /// `0..num_degenerate`; the remaining rows are zero.
    ///
    /// They span the null space of [`Self::information`]. They come from the eigen-
    /// decomposition with rotation scaled by [`Self::median_depth_m`] and are mapped back to the
    /// unscaled tangent, so they are unit but not necessarily orthogonal.
    pub degenerate_directions: [[f64; 6]; 6],
    /// Number of directions in [`Self::degenerate_directions`].
    pub num_degenerate: usize,
    /// Median depth of the final pass's inliers in the target frame, metres: the scale that
    /// makes rotation comparable to translation in the degeneracy test.
    pub median_depth_m: f64,
}

/// Relative Cholesky pivot threshold of the degeneracy guard: while factoring
/// `A = J^T W J`, a squared pivot below `PIVOT_RTOL` times its *block's* max
/// diagonal (rotation rows 0-2 / translation rows 3-5 are thresholded
/// separately — rotational entries scale as `|p x n|^2` ~ depth² while
/// translational ones are O(1) from unit normals, so one global max would let
/// a collapsed translation pivot pass at large depth) means some twist
/// direction is (numerically) unobserved — e.g. a single plane leaves its two
/// in-plane translations and the in-plane rotation unconstrained, so three
/// pivots collapse to rounding noise. Such a system no longer fails: the
/// Cholesky fast path is skipped and the eigen solve (see
/// [`IcpPlaneCriteria::degeneracy_rtol`]) answers along the observed directions
/// only. The guard still zeroes the reported conditioning (see
/// `block_conditioning`).
const PIVOT_RTOL: f64 = 1e-8;

/// Projective point-to-plane ICP between two RGBD pyramids.
///
/// # Arguments
///
/// * `source` - Source frame pyramid.
/// * `target` - Target frame pyramid.
/// * `initial_rot` - Initial rotation from the source to the target frame.
/// * `initial_trans` - Initial translation from the source to the target frame.
/// * `criteria` - Gating, robustness and convergence parameters.
///
/// # Returns
///
/// The estimate with its quality figures, data-only information and degenerate directions.
/// Geometry that does not constrain all six DoF is NOT an error: the result carries the
/// unobserved directions in [`IcpPlaneResult::degenerate_directions`], zero information along
/// them, and an update along them taken from the priors (none without one).
///
/// # Errors
///
/// [`RgbdIcpError::TooFewCorrespondences`] when fewer than 6 correspondences
/// survive gating (low overlap / bad initial guess);
/// [`RgbdIcpError::InvalidNumLevels`] if either pyramid is empty;
/// [`RgbdIcpError::InvalidPriorCovariance`] if [`IcpPlaneCriteria::pose_prior`] has a
/// covariance that is not symmetric positive definite.
pub fn icp_projective_plane(
    source: &RgbdPyramid,
    target: &RgbdPyramid,
    initial_rot: [[f64; 3]; 3],
    initial_trans: [f64; 3],
    criteria: IcpPlaneCriteria,
) -> Result<IcpPlaneResult, RgbdIcpError> {
    icp_projective_plane_with(
        &mut IcpWorkspace::default(),
        source,
        target,
        initial_rot,
        initial_trans,
        &criteria,
    )
}

/// Which implementation of the per-pixel loops runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IcpKernel {
    /// Plain loops: the reference the SIMD kernel is tested against.
    Scalar,
    /// `wide` vectors: portable, SSE2/AVX on x86, NEON on aarch64. The association pass runs
    /// eight `f32` lanes and sums the normal equations in `f64` per 128-point block, so it
    /// agrees with [`Self::Scalar`] to `f32` rounding, not `f64`.
    #[default]
    Simd,
}

/// Reusable buffers for [`icp_projective_plane_with`]: after the first call at a given image
/// size, a call allocates nothing.
#[derive(Debug, Default, Clone)]
pub struct IcpWorkspace {
    /// Kernel for the association passes.
    pub kernel: IcpKernel,
    soa: Vec<simd::SourceSoa>,
    depths: Vec<f32>,
}

impl IcpWorkspace {
    /// An empty workspace running `kernel`.
    ///
    /// # Arguments
    ///
    /// * `kernel` - Kernel for the association passes.
    ///
    /// # Returns
    ///
    /// The workspace; buffers grow on first use.
    pub fn new(kernel: IcpKernel) -> Self {
        Self {
            kernel,
            ..Default::default()
        }
    }
}

/// [`icp_projective_plane`] with caller-owned buffers, for real-time loops.
///
/// Same algorithm and results (to rounding between kernels). Reusing one `workspace` across
/// calls keeps the call free of allocation once warmed.
///
/// # Arguments
///
/// * `workspace` - Buffers and kernel choice, reused across calls.
/// * `source` - Source frame pyramid.
/// * `target` - Target frame pyramid.
/// * `initial_rot` - Initial rotation from the source to the target frame.
/// * `initial_trans` - Initial translation from the source to the target frame.
/// * `criteria` - Gating, robustness and convergence parameters.
///
/// # Returns
///
/// As [`icp_projective_plane`].
///
/// # Errors
///
/// As [`icp_projective_plane`].
pub fn icp_projective_plane_with(
    workspace: &mut IcpWorkspace,
    source: &RgbdPyramid,
    target: &RgbdPyramid,
    initial_rot: [[f64; 3]; 3],
    initial_trans: [f64; 3],
    criteria: &IcpPlaneCriteria,
) -> Result<IcpPlaneResult, RgbdIcpError> {
    let num_levels = source.levels.len().min(target.levels.len());
    if num_levels == 0 {
        return Err(RgbdIcpError::InvalidNumLevels(0));
    }

    let priors = [
        criteria.rotation_prior.map(PriorRows::from_rotation),
        criteria
            .pose_prior
            .map(|p| PriorRows::from_pose(&p))
            .transpose()?,
    ];

    let mut rotation = initial_rot;
    let mut translation = initial_trans;
    let mut iterations = 0;
    let mut rotation_conditioning = 0.0;
    let mut translation_conditioning = 0.0;
    let IcpWorkspace {
        kernel,
        soa,
        depths,
    } = workspace;
    let kernel = *kernel;
    if kernel == IcpKernel::Simd && soa.len() < num_levels {
        soa.resize_with(num_levels, Default::default);
    }
    if kernel == IcpKernel::Simd {
        for (buf, level) in soa.iter_mut().zip(&source.levels).take(num_levels) {
            buf.fill(level, &criteria.depth_noise);
        }
    }
    let pass = |level: usize,
                rotation: &[[f64; 3]; 3],
                translation: &[f64; 3],
                bilinear: bool,
                depths: Option<&mut Vec<f32>>| {
        let (src, tgt) = (&source.levels[level], &target.levels[level]);
        match kernel {
            IcpKernel::Scalar => {
                accumulate_level(src, tgt, rotation, translation, criteria, bilinear, depths)
            }
            IcpKernel::Simd => simd::accumulate(
                &soa[level],
                !src.depth_sigmas.is_empty(),
                tgt,
                rotation,
                translation,
                criteria,
                bilinear,
                depths,
            ),
        }
    };
    // Hybrid starts on nearest and switches once on the finest level
    let mut bilinear = criteria.association == Association::Bilinear;

    // coarse-to-fine: level num_levels-1 down to 0 (levels[0] is finest)
    for (coarse_idx, level_idx) in (0..num_levels).rev().enumerate() {
        let iters = criteria
            .iters_per_level
            .get(coarse_idx)
            .or(criteria.iters_per_level.last())
            .copied()
            .unwrap_or(0);
        let finest = level_idx == 0;
        // Hybrid refines with bilinear on the finest level only, after nearest has converged
        // to within HYBRID_SWITCH_NORM, with its own budget of `iters` more iterations.
        let mut budget = iters;
        let mut it = 0;

        while it < budget {
            it += 1;
            let eqs = pass(level_idx, &rotation, &translation, bilinear, None);
            if eqs.num_inliers < 6 {
                return Err(RgbdIcpError::TooFewCorrespondences(eqs.num_inliers));
            }
            // Diagnostic first, on the DATA-only equations — see `block_conditioning`. A prior
            // must not be able to make weak geometry look strong to the caller.
            let (data_rot_cond, data_trans_cond) = block_conditioning(&eqs.a);
            // Unobserved directions are decided on the data alone, before any prior is added,
            // for the same reason.
            let split = split_degenerate(&eqs.a, eqs.depth_scale, criteria.degeneracy_rtol);
            let x = solve_step(
                &eqs,
                &split,
                &priors,
                &rotation,
                &translation,
                criteria.degeneracy_rtol,
            );
            // Report the finest level's last solve: the geometry the pose actually rests on,
            // which is the data-only conditioning even when a prior carried the solve.
            rotation_conditioning = data_rot_cond;
            translation_conditioning = data_trans_cond;

            let omega = [x[0], x[1], x[2]];
            let dt = [x[3], x[4], x[5]];
            let r_delta = so3_exp(&omega);
            rotation = mat3_mul(&r_delta, &rotation);
            translation = [
                mat3_row_dot(&r_delta, 0, &translation) + dt[0],
                mat3_row_dot(&r_delta, 1, &translation) + dt[1],
                mat3_row_dot(&r_delta, 2, &translation) + dt[2],
            ];
            iterations += 1;

            let update_norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if criteria.association == Association::Hybrid
                && finest
                && !bilinear
                && update_norm < HYBRID_SWITCH_NORM
            {
                bilinear = true;
                budget = it + iters;
                continue;
            }
            if update_norm < criteria.update_tolerance {
                break;
            }
        }
    }

    // final metrics: one association pass at the finest level, final pose
    let eqs = pass(0, &rotation, &translation, bilinear, Some(depths));
    let split = split_degenerate(&eqs.a, eqs.depth_scale, criteria.degeneracy_rtol);
    let information = if split.num_degenerate == 0 {
        eqs.a
    } else {
        split.a_observed
    };
    let rmse = if eqs.num_inliers > 0 {
        (eqs.sum_sq_residual / eqs.num_inliers as f64).sqrt()
    } else {
        f64::INFINITY
    };
    let inlier_fraction = if eqs.num_associated > 0 {
        eqs.num_inliers as f64 / eqs.num_associated as f64
    } else {
        0.0
    };
    let overlap_fraction = if eqs.num_valid > 0 {
        eqs.num_associated as f64 / eqs.num_valid as f64
    } else {
        0.0
    };

    Ok(IcpPlaneResult {
        rotation,
        translation,
        rmse,
        inlier_fraction,
        num_associated: eqs.num_associated,
        rotation_conditioning,
        translation_conditioning,
        observability: rotation_conditioning.min(translation_conditioning),
        overlap_fraction,
        iterations,
        information,
        degenerate_directions: split.directions,
        num_degenerate: split.num_degenerate,
        median_depth_m: eqs.depth_scale,
    })
}

/// Accumulated normal equations of one association pass over a level.
struct NormalEquations {
    /// `J^T W J` (full symmetric 6x6).
    a: [[f64; 6]; 6],
    /// `J^T W r`.
    b: [f64; 6],
    /// Correspondences that survived gating.
    num_inliers: usize,
    /// Projected inside the target onto valid geometry (pre-gating).
    num_associated: usize,
    /// Source pixels with a valid vertex and normal.
    num_valid: usize,
    /// Unweighted sum of squared residuals over the inliers.
    sum_sq_residual: f64,
    /// Target-frame depth of the inliers that scales rotation in the degeneracy test: their
    /// median on the final pass, their mean during iterations (a per-iteration select was ~0.5
    /// ms of a 9 ms pass at 320x200); 1 m when there are none.
    depth_scale: f64,
}

/// [`NormalEquations::depth_scale`]: the median of `depths` when collected, else the mean.
fn depth_scale(sum_z: f64, num_inliers: usize, depths: Option<&mut Vec<f32>>) -> f64 {
    if let Some(d) = depths.filter(|d| !d.is_empty()) {
        let mid = d.len() / 2;
        let (_, m, _) = d.select_nth_unstable_by(mid, f32::total_cmp);
        if *m > 0.0 {
            return *m as f64;
        }
    }
    if num_inliers > 0 && sum_z > 0.0 {
        sum_z / num_inliers as f64
    } else {
        1.0
    }
}

/// One gated correspondence of a projective-association pass.
struct Correspondence {
    /// Index of the source pixel in its level.
    #[cfg_attr(not(test), allow(dead_code))]
    src_idx: usize,
    /// Interpolated target vertex the residual is measured against.
    #[cfg_attr(not(test), allow(dead_code))]
    v_t: [f64; 3],
    /// Interpolated, renormalised target normal.
    #[cfg_attr(not(test), allow(dead_code))]
    n_t: [f64; 3],
    /// Jacobian row `[(p' x n_t)^T, n_t^T]`, the interpolated target held fixed.
    j: [f64; 6],
    /// Point-to-plane residual in metres.
    r: f64,
    /// One-sigma error of `r` in metres.
    sigma: f64,
    /// Depth of `p'` in the target frame.
    z: f64,
}

/// Counts of one association pass, besides the correspondences themselves.
struct PassCounts {
    num_valid: usize,
    num_associated: usize,
}

/// Lower bound on `|n . ray| / |ray|` in the residual sigma.
///
/// The first-order projection of depth noise onto the plane vanishes at grazing incidence, where
/// the error that remains is the normal's, which this model does not carry; unbounded, those rows
/// would get near-infinite weight. 0.2 is about 78 deg from the normal. Not measured.
const MIN_RAY_INCIDENCE: f64 = 0.2;

/// Huber-and-sigma weight of a row: `huber(r / sigma) / sigma^2`.
#[inline]
fn row_weight(r: f64, sigma: f64, huber_delta: f64) -> f64 {
    let rn = (r / sigma).abs();
    let huber = if rn <= huber_delta {
        1.0
    } else {
        huber_delta / rn
    };
    huber / (sigma * sigma)
}

/// Depth sigma of one point: its level's map entry when present, else the model.
#[inline]
fn point_depth_sigma(level: &RgbdLevel, idx: usize, z: f64, noise: &DepthNoise) -> f64 {
    match level.depth_sigmas.get(idx) {
        Some(&s) if s.is_finite() && s > 0.0 => s as f64,
        _ => noise.depth_sigma(z),
    }
}

/// Squared cosine between a normal and a ray, bounded below by [`MIN_RAY_INCIDENCE`], times
/// the ray's squared length.
#[inline]
fn projected_ray_sq(n: &[f64; 3], ray: &[f64; 3]) -> f64 {
    let c = n[0] * ray[0] + n[1] * ray[1] + n[2] * ray[2];
    let len_sq = ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2];
    (c * c).max(MIN_RAY_INCIDENCE * MIN_RAY_INCIDENCE * len_sq)
}

/// Largest relative depth spread `(z_max - z_min) / z_min` across the four pixels of a bilinear
/// target sample; beyond it the four straddle a depth discontinuity and the sample is dropped.
///
/// A surface 80 deg from fronto-parallel spreads ~3% across one pixel at fx = 200, so 5% keeps
/// steep surfaces and still drops any foreground/background pair (a 10 cm step at 2 m). Chosen
/// on that geometry, not tuned on data.
const MAX_BILINEAR_DEPTH_JUMP: f64 = 0.05;

/// A target vertex/normal interpolated at a sub-pixel location.
struct TargetSample {
    vertex: [f64; 3],
    /// Unit length.
    normal: [f64; 3],
    /// Interpolated sigma-map value, when all four pixels carry one.
    depth_sigma: Option<f64>,
}

/// Nearest-pixel sample: the target vertex and normal of the pixel `(u, v)` rounds to.
fn sample_nearest(tgt: &RgbdLevel, u: f64, v: f64) -> Option<TargetSample> {
    let (tw, th) = (tgt.intrinsics.width, tgt.intrinsics.height);
    // round half up by truncation: `f64::round` is a libm call on the x86-64 baseline target
    if !(u >= -0.5 && v >= -0.5) {
        return None;
    }
    let (ui, vi) = ((u + 0.5) as usize, (v + 0.5) as usize);
    if ui > tw - 1 || vi > th - 1 {
        return None;
    }
    let idx = vi * tw + ui;
    let (vt, nt) = (&tgt.vertices[idx], &tgt.normals[idx]);
    if !is_valid_vertex(vt) || !is_valid_normal(nt) {
        return None;
    }
    Some(TargetSample {
        vertex: vt.map(|c| c as f64),
        normal: nt.map(|c| c as f64),
        depth_sigma: match tgt.depth_sigmas.get(idx) {
            Some(&s) if s.is_finite() && s > 0.0 => Some(s as f64),
            _ => None,
        },
    })
}

/// Bilinear sample of the target vertex and normal maps at continuous pixel `(u, v)`.
///
/// Nearest-pixel snapping biases the residual: the matched vertex is up to half a pixel from the
/// ray, which on a curved or slanted surface leaves a residual that does not average out (measured
/// 0.18 mrad / 0.3 mm on noise-free corner + sphere). Interpolating removes it to first order.
/// `None` when any of the four pixels is invalid, when they straddle a depth discontinuity (see
/// [`MAX_BILINEAR_DEPTH_JUMP`]) or when the normals nearly cancel.
fn sample_target(tgt: &RgbdLevel, u: f64, v: f64) -> Option<TargetSample> {
    let (tw, th) = (tgt.intrinsics.width, tgt.intrinsics.height);
    if !(u >= 0.0 && v >= 0.0) || u > (tw - 1) as f64 || v > (th - 1) as f64 {
        return None;
    }
    // the last row/column has no right/lower neighbour: clamp so its weight is zero there
    // u, v >= 0 here, so truncation is floor (and no libm call)
    let u0 = (u as usize).min(tw.saturating_sub(2));
    let v0 = (v as usize).min(th.saturating_sub(2));
    let (fu, fv) = (u - u0 as f64, v - v0 as f64);
    let idx = [
        v0 * tw + u0,
        v0 * tw + u0 + 1,
        (v0 + 1) * tw + u0,
        (v0 + 1) * tw + u0 + 1,
    ];
    let wts = [
        (1.0 - fu) * (1.0 - fv),
        fu * (1.0 - fv),
        (1.0 - fu) * fv,
        fu * fv,
    ];
    let (mut z_min, mut z_max) = (f64::INFINITY, 0.0f64);
    let mut vertex = [0.0; 3];
    let mut normal = [0.0; 3];
    let (mut sig, mut sig_ok) = (0.0, !tgt.depth_sigmas.is_empty());
    for (&i, &w) in idx.iter().zip(&wts) {
        let (vt, nt) = (&tgt.vertices[i], &tgt.normals[i]);
        if !is_valid_vertex(vt) || !is_valid_normal(nt) {
            return None;
        }
        z_min = z_min.min(vt[2] as f64);
        z_max = z_max.max(vt[2] as f64);
        for k in 0..3 {
            vertex[k] += w * vt[k] as f64;
            normal[k] += w * nt[k] as f64;
        }
        if sig_ok {
            match tgt.depth_sigmas[i] {
                s if s.is_finite() && s > 0.0 => sig += w * s as f64,
                _ => sig_ok = false,
            }
        }
    }
    if z_max - z_min > MAX_BILINEAR_DEPTH_JUMP * z_min {
        return None;
    }
    let n = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if n < 0.5 {
        return None;
    }
    Some(TargetSample {
        vertex,
        normal: normal.map(|c| c / n),
        depth_sigma: sig_ok.then_some(sig),
    })
}

/// One projective-association pass: gate each source pixel and hand every surviving
/// correspondence to `visit`.
fn for_each_correspondence(
    src: &RgbdLevel,
    tgt: &RgbdLevel,
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
    criteria: &IcpPlaneCriteria,
    bilinear: bool,
    mut visit: impl FnMut(&Correspondence),
) -> PassCounts {
    let mut counts = PassCounts {
        num_valid: 0,
        num_associated: 0,
    };
    let cos_max_angle = criteria.max_normal_angle_rad.cos();
    let max_dist_sq = criteria.max_dist_m * criteria.max_dist_m;
    // Constant noise with no map anywhere is the legacy uniform weighting: a residual sigma.
    let ray_model = !src.depth_sigmas.is_empty()
        || !tgt.depth_sigmas.is_empty()
        || !matches!(criteria.depth_noise, DepthNoise::Constant { .. });
    let constant_sigma = match criteria.depth_noise {
        DepthNoise::Constant { sigma_m } => sigma_m,
        DepthNoise::Stereo { .. } => 0.0,
    };

    for (src_idx, (v_s, n_s)) in src.vertices.iter().zip(src.normals.iter()).enumerate() {
        if !is_valid_vertex(v_s) || !is_valid_normal(n_s) {
            continue;
        }
        counts.num_valid += 1;

        let p_s = [v_s[0] as f64, v_s[1] as f64, v_s[2] as f64];
        // p' = T * p_s in the target frame
        let p = [
            mat3_row_dot(rotation, 0, &p_s) + translation[0],
            mat3_row_dot(rotation, 1, &p_s) + translation[1],
            mat3_row_dot(rotation, 2, &p_s) + translation[2],
        ];
        let Some(uv) = tgt.intrinsics.project(&p) else {
            continue;
        };
        let sample = if bilinear {
            sample_target(tgt, uv[0], uv[1])
        } else {
            sample_nearest(tgt, uv[0], uv[1])
        };
        let Some(sample) = sample else {
            continue;
        };
        let (v_t, n_t) = (sample.vertex, sample.normal);
        counts.num_associated += 1;

        let diff = [p[0] - v_t[0], p[1] - v_t[1], p[2] - v_t[2]];
        if diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] > max_dist_sq {
            continue;
        }

        let n_s = [n_s[0] as f64, n_s[1] as f64, n_s[2] as f64];
        let n_s_rot = [
            mat3_row_dot(rotation, 0, &n_s),
            mat3_row_dot(rotation, 1, &n_s),
            mat3_row_dot(rotation, 2, &n_s),
        ];
        if n_s_rot[0] * n_t[0] + n_s_rot[1] * n_t[1] + n_s_rot[2] * n_t[2] < cos_max_angle {
            continue;
        }

        let r = n_t[0] * diff[0] + n_t[1] * diff[1] + n_t[2] * diff[2];
        // Jacobian row [(p' x n_t)^T, n_t^T]: the interpolated target is held fixed within an
        // iteration. Its motion through the projection is a real term: on noise-free corner +
        // sphere, re-associating finite differences exceed this information by 7.3% on t_y, and
        // modelling that motion and the normal's cut it to 1.4%. But that term is built from
        // derivatives of the noisy vertex and normal maps, and on noisy depth it inflated static
        // NEES to ~2470, against ~6.6 without it. Held fixed is the consistent choice.
        let j = [
            p[1] * n_t[2] - p[2] * n_t[1],
            p[2] * n_t[0] - p[0] * n_t[2],
            p[0] * n_t[1] - p[1] * n_t[0],
            n_t[0],
            n_t[1],
            n_t[2],
        ];
        let sigma = if ray_model {
            // Depth noise moves each point along its own ray `p / z`; project both onto n_t.
            let z_s = p_s[2];
            let ray_s = [
                (p[0] - translation[0]) / z_s,
                (p[1] - translation[1]) / z_s,
                (p[2] - translation[2]) / z_s,
            ];
            let z_t = v_t[2];
            let ray_t = [v_t[0] / z_t, v_t[1] / z_t, 1.0];
            let sig_s = point_depth_sigma(src, src_idx, z_s, &criteria.depth_noise);
            let sig_t = sample
                .depth_sigma
                .unwrap_or_else(|| criteria.depth_noise.depth_sigma(z_t));
            (projected_ray_sq(&n_t, &ray_s) * sig_s * sig_s
                + projected_ray_sq(&n_t, &ray_t) * sig_t * sig_t)
                .sqrt()
        } else {
            constant_sigma
        }
        .max(f64::MIN_POSITIVE.sqrt());

        visit(&Correspondence {
            src_idx,
            v_t,
            n_t,
            j,
            r,
            sigma,
            z: p[2],
        });
    }

    counts
}

/// One projective-association pass: gate, weight and accumulate `J^T W J`,
/// `J^T W r` in f64.
fn accumulate_level(
    src: &RgbdLevel,
    tgt: &RgbdLevel,
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
    criteria: &IcpPlaneCriteria,
    bilinear: bool,
    mut depths: Option<&mut Vec<f32>>,
) -> NormalEquations {
    let mut eqs = NormalEquations {
        a: [[0.0; 6]; 6],
        b: [0.0; 6],
        num_inliers: 0,
        num_associated: 0,
        num_valid: 0,
        sum_sq_residual: 0.0,
        depth_scale: 1.0,
    };
    if let Some(d) = depths.as_deref_mut() {
        d.clear();
    }
    let mut sum_z = 0.0;
    let counts =
        for_each_correspondence(src, tgt, rotation, translation, criteria, bilinear, |c| {
            let w = row_weight(c.r, c.sigma, criteria.huber_delta);
            for i in 0..6 {
                eqs.b[i] += w * c.j[i] * c.r;
                for k in i..6 {
                    eqs.a[i][k] += w * c.j[i] * c.j[k];
                }
            }
            eqs.num_inliers += 1;
            eqs.sum_sq_residual += c.r * c.r;
            sum_z += c.z;
            if let Some(d) = depths.as_deref_mut() {
                d.push(c.z as f32);
            }
        });
    eqs.num_valid = counts.num_valid;
    eqs.num_associated = counts.num_associated;

    // mirror the accumulated upper triangle
    for i in 1..6 {
        for k in 0..i {
            eqs.a[i][k] = eqs.a[k][i];
        }
    }

    eqs.depth_scale = depth_scale(sum_z, eqs.num_inliers, depths);
    eqs
}

/// Prior rows in information form: error `e = [log(R R_p^T), t - R R_p^T t_p]` with weight
/// `info`. A rotation prior is the same thing with zero translation information.
struct PriorRows {
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
    info: [[f64; 6]; 6],
}

impl PriorRows {
    fn from_rotation(p: RotationPrior) -> Self {
        let w = 1.0 / (p.sigma_rad * p.sigma_rad).max(f64::MIN_POSITIVE);
        let mut info = [[0.0; 6]; 6];
        for (i, row) in info.iter_mut().enumerate().take(3) {
            row[i] = w;
        }
        Self {
            rotation: p.rotation,
            translation: [0.0; 3],
            info,
        }
    }

    fn from_pose(p: &PosePrior) -> Result<Self, RgbdIcpError> {
        let c = &p.covariance;
        for (i, row) in c.iter().enumerate() {
            for (k, &a) in row.iter().enumerate() {
                let b = c[k][i];
                if !a.is_finite() || (a - b).abs() > 1e-9 * (a.abs() + b.abs()) {
                    return Err(RgbdIcpError::InvalidPriorCovariance);
                }
            }
        }
        let inv = Matrix6::from_fn(|i, k| c[i][k])
            .cholesky()
            .ok_or(RgbdIcpError::InvalidPriorCovariance)?
            .inverse();
        let mut info = [[0.0; 6]; 6];
        for (i, row) in info.iter_mut().enumerate() {
            for (k, v) in row.iter_mut().enumerate() {
                // symmetrise away the inverse's rounding
                *v = 0.5 * (inv[(i, k)] + inv[(k, i)]);
            }
        }
        Ok(Self {
            rotation: p.rotation,
            translation: p.translation,
            info,
        })
    }

    /// Add `J^T info J` and `J^T info e` for the current estimate.
    ///
    /// The error must live in the frame the update acts in. The solver applies `R <- exp(w) R`,
    /// a LEFT multiply, so the rotation error is `log(R R_prior^T)`, also on the left: then
    /// `R_new R_prior^T = exp(w) exp(e)` and its residual in `w` is `e + w` to first order, with
    /// an identity Jacobian. Writing the body-frame error `log(R_prior^T R)` here instead would
    /// agree only when the prior is near identity and would otherwise pull toward the wrong
    /// attitude. The translation error `e_t = t - R R_p^T t_p` moves to `exp(w) e_t + v`, so its
    /// Jacobian is `[-[e_t]x, I]`.
    fn add_to(
        &self,
        rotation: &[[f64; 3]; 3],
        translation: &[f64; 3],
        a: &mut [[f64; 6]; 6],
        b: &mut [f64; 6],
    ) {
        let rr = mat3_mul(rotation, &mat3_transpose(&self.rotation));
        let e_r = so3_log(&rr);
        let e_t = [
            translation[0] - mat3_row_dot(&rr, 0, &self.translation),
            translation[1] - mat3_row_dot(&rr, 1, &self.translation),
            translation[2] - mat3_row_dot(&rr, 2, &self.translation),
        ];
        let e = [e_r[0], e_r[1], e_r[2], e_t[0], e_t[1], e_t[2]];
        // J = [[I, 0], [-[e_t]x, I]]
        let mut jac = [[0.0; 6]; 6];
        for (i, row) in jac.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let skew = [
            [0.0, -e_t[2], e_t[1]],
            [e_t[2], 0.0, -e_t[0]],
            [-e_t[1], e_t[0], 0.0],
        ];
        for i in 0..3 {
            for k in 0..3 {
                jac[3 + i][k] = -skew[i][k];
            }
        }
        // info * J and info * e
        let mut ij = [[0.0; 6]; 6];
        let mut ie = [0.0; 6];
        for i in 0..6 {
            for k in 0..6 {
                ie[i] += self.info[i][k] * e[k];
                for m in 0..6 {
                    ij[i][m] += self.info[i][k] * jac[k][m];
                }
            }
        }
        for i in 0..6 {
            for k in 0..6 {
                b[i] += jac[k][i] * ie[k];
                for m in 0..6 {
                    a[i][m] += jac[k][i] * ij[k][m];
                }
            }
        }
    }
}

/// Eigen split of a data-only `J^T W J` into observed and unobserved directions.
struct DegenerateSplit {
    /// Rotation scale used for the decomposition (median depth, metres).
    scale: f64,
    /// `J^T W J` with the unobserved eigen-directions removed, unscaled tangent.
    a_observed: [[f64; 6]; 6],
    /// Projector onto the observed eigen-directions, in SCALED coordinates.
    projector: [[f64; 6]; 6],
    /// Unit unobserved directions in the unscaled tangent, rows `0..num_degenerate`.
    directions: [[f64; 6]; 6],
    num_degenerate: usize,
}

#[inline]
fn tangent_scale(d: f64) -> [f64; 6] {
    [d, d, d, 1.0, 1.0, 1.0]
}

/// Symmetric eigen-decomposition of `S^-1 A S^-1` with `S = diag(d, d, d, 1, 1, 1)`, eigenpairs
/// in ascending order.
fn scaled_eigen(a: &[[f64; 6]; 6], d: f64) -> ([(f64, [f64; 6]); 6], f64) {
    let s = tangent_scale(d);
    let m = Matrix6::from_fn(|i, k| 0.5 * (a[i][k] + a[k][i]) / (s[i] * s[k]));
    let eig = m.symmetric_eigen();
    let mut pairs: [(f64, [f64; 6]); 6] = std::array::from_fn(|c| {
        let col = eig.eigenvectors.column(c);
        (
            eig.eigenvalues[c],
            [col[0], col[1], col[2], col[3], col[4], col[5]],
        )
    });
    pairs.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
    (pairs, pairs[5].0)
}

fn split_degenerate(a: &[[f64; 6]; 6], median_depth: f64, rtol: f64) -> DegenerateSplit {
    let d = if median_depth.is_finite() && median_depth > 0.0 {
        median_depth
    } else {
        1.0
    };
    let s = tangent_scale(d);
    let (pairs, lmax) = scaled_eigen(a, d);
    let mut out = DegenerateSplit {
        scale: d,
        a_observed: [[0.0; 6]; 6],
        projector: [[0.0; 6]; 6],
        directions: [[0.0; 6]; 6],
        num_degenerate: 0,
    };
    for (lambda, v) in pairs {
        // `!(x > y)` so a NaN or an all-zero system counts as unobserved
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let unobserved = !(lambda > rtol * lmax) || lmax <= 0.0;
        if unobserved {
            let x: [f64; 6] = std::array::from_fn(|i| v[i] / s[i]);
            let n = x.iter().map(|c| c * c).sum::<f64>().sqrt();
            out.directions[out.num_degenerate] = x.map(|c| c / n);
            out.num_degenerate += 1;
        } else {
            for i in 0..6 {
                for k in 0..6 {
                    out.a_observed[i][k] += s[i] * s[k] * lambda * v[i] * v[k];
                    out.projector[i][k] += v[i] * v[k];
                }
            }
        }
    }
    out
}

/// One Gauss-Newton step `x = [w, v]` from the data equations and the priors.
///
/// Full-rank data take the Cholesky path on `A + prior`, numerically the solver it always was.
/// Otherwise the unobserved directions are removed from the data (both `A` and `b`, so a
/// rounding-noise gradient cannot push along them), the priors are added, and the sum is solved
/// by eigen pseudo-inverse in the depth-scaled tangent: a direction neither data nor prior
/// observes gets a zero update.
fn solve_step(
    eqs: &NormalEquations,
    split: &DegenerateSplit,
    priors: &[Option<PriorRows>],
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
    rtol: f64,
) -> [f64; 6] {
    let (mut a, mut b) = if split.num_degenerate == 0 {
        (eqs.a, eqs.b)
    } else {
        let s = tangent_scale(split.scale);
        let b_scaled: [f64; 6] = std::array::from_fn(|i| eqs.b[i] / s[i]);
        let b_obs: [f64; 6] = std::array::from_fn(|i| {
            s[i] * (0..6)
                .map(|k| split.projector[i][k] * b_scaled[k])
                .sum::<f64>()
        });
        (split.a_observed, b_obs)
    };
    for p in priors.iter().flatten() {
        p.add_to(rotation, translation, &mut a, &mut b);
    }
    let neg_b = b.map(|v| -v);
    if split.num_degenerate == 0 {
        if let Some((x, _, _)) = cholesky_solve_6x6(&a, &neg_b) {
            return x;
        }
    }
    pseudo_solve(&a, &neg_b, split.scale, rtol)
}

/// Minimum-norm solve of `A x = rhs` in the depth-scaled tangent, dropping eigen-directions at
/// or below `rtol` times the largest eigenvalue.
fn pseudo_solve(a: &[[f64; 6]; 6], rhs: &[f64; 6], d: f64, rtol: f64) -> [f64; 6] {
    let s = tangent_scale(d);
    let rhs_scaled: [f64; 6] = std::array::from_fn(|i| rhs[i] / s[i]);
    let (pairs, lmax) = scaled_eigen(a, d);
    let mut y = [0.0; 6];
    if lmax > 0.0 {
        for (lambda, v) in pairs {
            if lambda > rtol * lmax {
                let coef = v.iter().zip(&rhs_scaled).map(|(a, b)| a * b).sum::<f64>() / lambda;
                for i in 0..6 {
                    y[i] += coef * v[i];
                }
            }
        }
    }
    std::array::from_fn(|i| y[i] / s[i])
}

/// Solve the SPD system `A x = b` by Cholesky. Returns `None` when a squared
/// pivot falls below `PIVOT_RTOL` times its block's max diagonal (rotation /
/// translation thresholded separately — see [`PIVOT_RTOL`]) — the degeneracy
/// guard.
/// Solves `A x = b` and reports the weakest pivot relative to its block's max diagonal — the
/// observability of the least-constrained direction. A view that pins every DOF keeps this well
/// above [`PIVOT_RTOL`]; a wall filling the frame drives it toward zero along the sliding
/// direction, where the residual stays perfect while the pose drifts. Callers that must not act
/// on such a pose gate on it; the residual and the inlier fraction cannot see it.
/// Conditioning of each block WITHOUT solving: weakest pivot over strongest, `(rotation,
/// translation)`, or `(0, 0)` when the matrix does not factor at all.
///
/// Prior-free in its INPUTS, not in the pose. A prior changes the solve, so the final transform
/// differs, so the last association pass covers a slightly different correspondence set and the
/// conditioning is computed over that — measured drift of about 5% on the same frame. That is
/// benign for gating; evaluating at a fixed transform would be required to make it bit-stable.
///
/// Measured on the DATA-only normal equations, never on the prior-augmented ones. A rotation
/// prior lifts the rotation block, and because the blocks are coupled through elimination it
/// lifts the translation block's effective pivots too — measured, a 14x rise at a frame whose
/// translation error was unchanged at 3.5x the truth. Reporting that would let the regulariser
/// flatter the diagnostic and silently disarm the gate that catches unobservable translation.
/// The prior belongs in the solve; the diagnostic must keep describing the geometry.
fn block_conditioning(a: &[[f64; 6]; 6]) -> (f64, f64) {
    match cholesky_factor(a) {
        Some((_, rot, trans)) => (rot, trans),
        None => (0.0, 0.0),
    }
}

fn cholesky_solve_6x6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<([f64; 6], f64, f64)> {
    let (l, rot_cond, trans_cond) = cholesky_factor(a)?;

    // L y = b
    let mut y = [0.0; 6];
    for i in 0..6 {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    // L^T x = y
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        let mut sum = y[i];
        for k in i + 1..6 {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    Some((x, rot_cond, trans_cond))
}

fn cholesky_factor(a: &[[f64; 6]; 6]) -> Option<([[f64; 6]; 6], f64, f64)> {
    let max_diag_rot = (0..3).map(|i| a[i][i]).fold(0.0, f64::max);
    let max_diag_trans = (3..6).map(|i| a[i][i]).fold(0.0, f64::max);
    if max_diag_rot <= 0.0 || max_diag_trans <= 0.0 {
        return None;
    }

    // A = L L^T. Track each block's weakest and strongest pivot: their RATIO is the
    // conditioning. An absolute pivot is not comparable across scenes — pivots accumulate over
    // correspondences, so they grow with the inlier count and the depth, and a direction can be
    // orders of magnitude worse constrained than its neighbours while still being numerically
    // large. The ratio is scale-free and is what "one direction is unconstrained" means.
    let (mut min_rot, mut max_rot) = (f64::INFINITY, 0.0f64);
    let (mut min_trans, mut max_trans) = (f64::INFINITY, 0.0f64);
    let mut l = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                let block_scale = if i < 3 { max_diag_rot } else { max_diag_trans };
                if i < 3 {
                    min_rot = min_rot.min(sum);
                    max_rot = max_rot.max(sum);
                } else {
                    min_trans = min_trans.min(sum);
                    max_trans = max_trans.max(sum);
                }
                if sum < PIVOT_RTOL * block_scale {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    let ratio = |lo: f64, hi: f64| {
        if hi > 0.0 {
            (lo / hi).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    Some((l, ratio(min_rot, max_rot), ratio(min_trans, max_trans)))
}

/// Axis-angle of a rotation matrix: the inverse of [`so3_exp`].
fn so3_log(r: &[[f64; 3]; 3]) -> [f64; 3] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let angle = cos.acos();
    // Near zero the axis is ill-defined but the vector is not: sin(angle)/angle -> 1, so the
    // off-diagonal difference IS the axis-angle to first order.
    let scale = if angle < 1e-8 {
        0.5
    } else if angle > std::f64::consts::PI - 1e-6 {
        // Near pi the same expression loses all precision; recover the axis from the symmetric
        // part, where the diagonal stays well conditioned, and restore the sign from the skew.
        let mut axis = [
            ((r[0][0] - cos) / (1.0 - cos)).max(0.0).sqrt(),
            ((r[1][1] - cos) / (1.0 - cos)).max(0.0).sqrt(),
            ((r[2][2] - cos) / (1.0 - cos)).max(0.0).sqrt(),
        ];
        let skew = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
        for k in 0..3 {
            if skew[k] < 0.0 {
                axis[k] = -axis[k];
            }
        }
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        return if n > 0.0 {
            [
                axis[0] / n * angle,
                axis[1] / n * angle,
                axis[2] / n * angle,
            ]
        } else {
            [0.0; 3]
        };
    } else {
        angle / (2.0 * angle.sin())
    };
    [
        scale * (r[2][1] - r[1][2]),
        scale * (r[0][2] - r[2][0]),
        scale * (r[1][0] - r[0][1]),
    ]
}

fn mat3_transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut t = [[0.0; 3]; 3];
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            t[j][i] = v;
        }
    }
    t
}

#[inline]
fn mat3_row_dot(m: &[[f64; 3]; 3], row: usize, v: &[f64; 3]) -> f64 {
    m[row][0] * v[0] + m[row][1] * v[1] + m[row][2] * v[2]
}

fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (out_row, a_row) in out.iter_mut().zip(a.iter()) {
        for j in 0..3 {
            out_row[j] = a_row[0] * b[0][j] + a_row[1] * b[1][j] + a_row[2] * b[2][j];
        }
    }
    out
}

/// SO(3) exponential map (Rodrigues): `R = I + a [w]x + b [w]x^2` with
/// `a = sin(t)/t`, `b = (1 - cos(t))/t^2`, Taylor fallback near zero.
fn so3_exp(w: &[f64; 3]) -> [[f64; 3]; 3] {
    let theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    let theta = theta2.sqrt();
    let (a, b) = if theta < 1e-9 {
        (1.0 - theta2 / 6.0, 0.5 - theta2 / 24.0)
    } else {
        (theta.sin() / theta, (1.0 - theta.cos()) / theta2)
    };
    // [w]x^2 = w w^T - theta^2 I
    [
        [
            1.0 + b * (w[0] * w[0] - theta2),
            -a * w[2] + b * w[0] * w[1],
            a * w[1] + b * w[0] * w[2],
        ],
        [
            a * w[2] + b * w[1] * w[0],
            1.0 + b * (w[1] * w[1] - theta2),
            -a * w[0] + b * w[1] * w[2],
        ],
        [
            -a * w[1] + b * w[2] * w[0],
            a * w[0] + b * w[2] * w[1],
            1.0 + b * (w[2] * w[2] - theta2),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::super::rgbd::DepthIntrinsics;
    use super::super::synth::{render_depth_m, render_depth_mm, Plane, Scene, Sphere};
    use super::*;
    use crate::transforms::axis_angle_to_rotation_matrix;

    const IDENTITY_ROT: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn test_intrinsics() -> DepthIntrinsics {
        DepthIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 159.5,
            cy: 89.5,
            width: 320,
            height: 180,
        }
    }

    /// Angle in degrees between two rotation matrices.
    fn rotation_error_deg(r_a: &[[f64; 3]; 3], r_b: &[[f64; 3]; 3]) -> f64 {
        // trace(R_a^T R_b)
        let mut trace = 0.0;
        for i in 0..3 {
            for k in 0..3 {
                trace += r_a[k][i] * r_b[k][i];
            }
        }
        ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
    }

    fn translation_error(t_a: &[f64; 3], t_b: &[f64; 3]) -> f64 {
        ((t_a[0] - t_b[0]).powi(2) + (t_a[1] - t_b[1]).powi(2) + (t_a[2] - t_b[2]).powi(2)).sqrt()
    }

    /// Ground-truth `T_target_source` from the target camera pose `T_world_cam`
    /// (the source camera sits at the world origin):
    /// `R = R_wc^T`, `t = -R_wc^T t_wc`.
    fn gt_target_source(
        rot_world_cam: &[[f64; 3]; 3],
        t_world_cam: &[f64; 3],
    ) -> ([[f64; 3]; 3], [f64; 3]) {
        let mut rot = [[0.0; 3]; 3];
        for (i, rot_row) in rot.iter_mut().enumerate() {
            for (j, cell) in rot_row.iter_mut().enumerate() {
                *cell = rot_world_cam[j][i];
            }
        }
        let t = [
            -(rot[0][0] * t_world_cam[0] + rot[0][1] * t_world_cam[1] + rot[0][2] * t_world_cam[2]),
            -(rot[1][0] * t_world_cam[0] + rot[1][1] * t_world_cam[1] + rot[1][2] * t_world_cam[2]),
            -(rot[2][0] * t_world_cam[0] + rot[2][1] * t_world_cam[1] + rot[2][2] * t_world_cam[2]),
        ];
        (rot, t)
    }

    #[test]
    fn test_identity_self_icp() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let depth = render_depth_mm(&Scene::corner_and_sphere(), &intr, &IDENTITY_ROT, &[0.0; 3]);
        let pyr = RgbdPyramid::from_depth_mm(&depth, &intr, 3)?;

        let result = icp_projective_plane(
            &pyr,
            &pyr,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria::default(),
        )?;

        assert!(
            rotation_error_deg(&result.rotation, &IDENTITY_ROT) < 0.01,
            "self-ICP rotation drifted: {:?}",
            result.rotation
        );
        let t_norm = translation_error(&result.translation, &[0.0; 3]);
        assert!(t_norm < 1e-4, "self-ICP translation drifted: {t_norm}");
        assert!(result.rmse < 1e-6, "self-ICP rmse: {}", result.rmse);
        assert!(
            result.inlier_fraction > 0.9,
            "self-ICP inlier fraction: {}",
            result.inlier_fraction
        );
        Ok(())
    }

    #[test]
    fn test_ground_truth_motion_grid() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let scene = Scene::corner_and_sphere();
        let src_pyr = {
            let depth = render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]);
            RgbdPyramid::from_depth_mm(&depth, &intr, 3)?
        };

        // (axis, angle deg, translation) of the target camera pose T_world_cam
        let cases: &[([f64; 3], f64, [f64; 3])] = &[
            ([1.0, 0.0, 0.0], 2.0, [0.0, 0.0, 0.0]),
            ([0.0, 1.0, 0.0], -2.0, [0.0, 0.0, 0.0]),
            ([0.0, 0.0, 1.0], 2.0, [0.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], 0.0, [0.03, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], 0.0, [0.0, -0.03, 0.0]),
            ([1.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.03]),
            ([1.0, 1.0, 1.0], 2.0, [0.02, -0.02, 0.015]),
        ];

        for (axis, angle_deg, t_wc) in cases {
            let rot_wc = axis_angle_to_rotation_matrix(axis, angle_deg.to_radians())?;
            let tgt_depth = render_depth_mm(&scene, &intr, &rot_wc, t_wc);
            let tgt_pyr = RgbdPyramid::from_depth_mm(&tgt_depth, &intr, 3)?;
            let (rot_gt, t_gt) = gt_target_source(&rot_wc, t_wc);

            let result = icp_projective_plane(
                &src_pyr,
                &tgt_pyr,
                IDENTITY_ROT,
                [0.0; 3],
                IcpPlaneCriteria::default(),
            )?;

            let rot_err = rotation_error_deg(&result.rotation, &rot_gt);
            let t_err = translation_error(&result.translation, &t_gt);
            assert!(
                rot_err < 0.2,
                "case {axis:?}/{angle_deg} deg/{t_wc:?}: rotation error {rot_err} deg"
            );
            assert!(
                t_err < 5e-3,
                "case {axis:?}/{angle_deg} deg/{t_wc:?}: translation error {} mm",
                t_err * 1e3
            );
            assert!(
                result.inlier_fraction > 0.5,
                "case {axis:?}/{angle_deg} deg/{t_wc:?}: inlier fraction {}",
                result.inlier_fraction
            );
        }
        Ok(())
    }

    /// A partly-overlapping view must keep HIGH match quality while overlap falls.
    ///
    /// Regression guard for a live failure: `inlier_fraction` once divided by ALL valid source
    /// pixels, so it fell with view overlap rather than with match error. A moving camera then
    /// scored ~11% on solves that had converged to millimetres, every frame was rejected as
    /// "tracking lost", the pose froze and the map stopped growing. Quality and overlap are
    /// separate signals: low overlap means re-key, not reject.
    #[test]
    fn inlier_fraction_measures_quality_not_overlap() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let scene = Scene::corner_and_sphere();
        let src_depth = render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]);
        let src_pyr = RgbdPyramid::from_depth_mm(&src_depth, &intr, 3)?;

        // Yaw far enough that a large part of the source view leaves the target frustum.
        let rot_wc = axis_angle_to_rotation_matrix(&[0.0, 1.0, 0.0], 18.0_f64.to_radians())?;
        let t_wc = [0.35, 0.0, 0.0];
        let tgt_depth = render_depth_mm(&scene, &intr, &rot_wc, &t_wc);
        let tgt_pyr = RgbdPyramid::from_depth_mm(&tgt_depth, &intr, 3)?;
        let (rot_gt, t_gt) = gt_target_source(&rot_wc, &t_wc);

        // Rotation comes from the gyro prior in the live node; translation does not.
        let result = icp_projective_plane(
            &src_pyr,
            &tgt_pyr,
            rot_gt,
            [0.0; 3],
            IcpPlaneCriteria::default(),
        )?;

        // The solve is genuinely good ...
        assert!(
            rotation_error_deg(&result.rotation, &rot_gt) < 0.5,
            "rotation error {} deg",
            rotation_error_deg(&result.rotation, &rot_gt)
        );
        assert!(
            translation_error(&result.translation, &t_gt) < 0.01,
            "translation error {} m",
            translation_error(&result.translation, &t_gt)
        );
        // ... so quality stays high, even though a chunk of the view is gone ...
        // ... so quality stays high even though half the view has left the frustum ...
        assert!(
            result.inlier_fraction > 0.9,
            "quality collapsed under partial overlap: {} (associated {})",
            result.inlier_fraction,
            result.num_associated
        );
        // ... the lost view registers as overlap instead, which is what should drive re-keying ...
        assert!(
            result.overlap_fraction < 0.7,
            "overlap should register the rotated-away view: {}",
            result.overlap_fraction
        );
        // ... and the two must not be the same number: the old inliers/valid metric (their
        // product) is what sank to ~11% on hardware and rejected converged solves.
        let inliers_over_valid = result.inlier_fraction * result.overlap_fraction;
        assert!(
            result.inlier_fraction > inliers_over_valid * 1.5,
            "quality {} tracks the old inliers/valid metric {} — the split is gone",
            result.inlier_fraction,
            inliers_over_valid
        );
        assert!(
            result.num_associated > 1000,
            "too little evidence to trust the pose: {}",
            result.num_associated
        );
        Ok(())
    }

    #[test]
    fn test_outlier_band_still_converges() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let scene = Scene::corner_and_sphere();
        let src_pyr = {
            let depth = render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]);
            RgbdPyramid::from_depth_mm(&depth, &intr, 3)?
        };

        let rot_wc = axis_angle_to_rotation_matrix(&[0.0, 1.0, 0.0], 1.5f64.to_radians())?;
        let t_wc = [0.02, 0.0, 0.01];
        let mut tgt_depth = render_depth_mm(&scene, &intr, &rot_wc, &t_wc);
        // corrupt a 10% horizontal band with a +60 mm bias: inside the 100 mm
        // distance gate, so it exercises the Huber weighting rather than the gate
        let band_rows = intr.height / 10;
        for v in 80..80 + band_rows {
            for u in 0..intr.width {
                let d = &mut tgt_depth[v * intr.width + u];
                if *d != 0 {
                    *d += 60;
                }
            }
        }
        let tgt_pyr = RgbdPyramid::from_depth_mm(&tgt_depth, &intr, 3)?;
        let (rot_gt, t_gt) = gt_target_source(&rot_wc, &t_wc);

        let result = icp_projective_plane(
            &src_pyr,
            &tgt_pyr,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria::default(),
        )?;

        let rot_err = rotation_error_deg(&result.rotation, &rot_gt);
        let t_err = translation_error(&result.translation, &t_gt);
        assert!(rot_err < 0.2, "rotation error with outliers: {rot_err} deg");
        assert!(
            t_err < 5e-3,
            "translation error with outliers: {} mm",
            t_err * 1e3
        );
        Ok(())
    }

    /// Weak geometry must be visible in `observability`, because nothing else shows it.
    ///
    /// Guards a measured failure: on a near-degenerate view the tracker reported a perfect
    /// inlier fraction and a sub-millimetre residual while publishing ~3x the true motion,
    /// sliding along the surface. Residual and inlier fraction are blind to it — the pose moves
    /// through a direction the geometry does not penalise — so the caller needs the conditioning
    /// of the normal equations to refuse such a pose.
    #[test]
    fn observability_exposes_weak_geometry() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        // A wall with one small bump. The plane alone leaves three DoF free; the bump pins them,
        // but only just — the near-degenerate regime, not the exactly-singular one that comes
        // back with degenerate directions. (Two planes at any angle stay rank-deficient along their intersection,
        // so a "shallow wedge" is the wrong shape for this test.)
        let weak = Scene {
            planes: vec![Plane {
                normal: [0.0, 0.0, 1.0],
                d: 2.0,
            }],
            spheres: vec![Sphere {
                center: [0.0, 0.0, 1.9],
                radius: 0.06,
            }],
        };
        let slide = [0.02, 0.0, 0.0]; // along the surface: little residual to pay

        let mut measured: Vec<(f64, f64, f64)> = Vec::new();
        for scene in [weak, Scene::corner_and_sphere()] {
            let src = RgbdPyramid::from_depth_mm(
                &render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]),
                &intr,
                3,
            )?;
            let tgt = RgbdPyramid::from_depth_mm(
                &render_depth_mm(&scene, &intr, &IDENTITY_ROT, &slide),
                &intr,
                3,
            )?;
            let r = icp_projective_plane(
                &src,
                &tgt,
                IDENTITY_ROT,
                [0.0; 3],
                IcpPlaneCriteria::default(),
            )?;
            measured.push((r.observability, r.inlier_fraction, r.rmse));
        }
        let (weak_obs, weak_inl, weak_rmse) = measured[0];
        let (rich_obs, rich_inl, _) = measured[1];

        // Measured: 6.4e-7 on the weak scene against 8.8e-2 on the rich one — five orders of
        // magnitude at nearly identical association counts, so the margin is geometry, not scale.
        assert!(
            weak_obs < 1e-4 && rich_obs > 1e-3,
            "observability must separate the two geometries: weak {weak_obs:.3e}, rich {rich_obs:.3e}"
        );
        // The trap this field exists for: on the weak scene the numbers a caller would otherwise
        // trust are not merely acceptable, they are BETTER than on the well-conditioned one.
        assert!(
            weak_inl >= rich_inl && weak_rmse < 0.001,
            "expected the deceptive regime: weak inliers {weak_inl} (rich {rich_inl}), \
             weak rmse {weak_rmse}"
        );
        Ok(())
    }

    /// Conditioning must not move with the number of correspondences.
    ///
    /// This is why the field is a ratio and not a pivot magnitude. Pivots of `J^T W J` accumulate
    /// over correspondences, so an absolute floor tracks image resolution and scene depth rather
    /// than geometry: a gate tuned on one scene silently never fires on a denser or nearer one.
    /// Same geometry at half resolution — a quarter of the correspondences — must report the same
    /// conditioning.
    /// A rotation prior must not flatter the conditioning it is reported alongside.
    ///
    /// Measured regression: with a prior active, a frame whose translation was unobservable saw
    /// its reported translation conditioning rise 14x — over the gate that had correctly held it
    /// — while the published pose stayed wrong by 3.5x. The blocks are coupled through
    /// elimination, so lifting rotation lifts translation's effective pivots. Conditioning is
    /// therefore measured on the data-only equations while the solve uses the augmented ones.
    #[test]
    fn rotation_prior_does_not_inflate_reported_conditioning(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        // Roll-degenerate: rotation unobservable, translation pinned by the sphere.
        let scene = Scene {
            planes: vec![Plane {
                normal: [0.0, 0.0, 1.0],
                d: 2.5,
            }],
            spheres: vec![Sphere {
                center: [0.0, 0.0, 1.8],
                radius: 0.55,
            }],
        };
        let true_rot = axis_angle_to_rotation_matrix(&[0.0, 0.0, 1.0], 3.0_f64.to_radians())?;
        let src = RgbdPyramid::from_depth_mm(
            &render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]),
            &intr,
            3,
        )?;
        let tgt = RgbdPyramid::from_depth_mm(
            &render_depth_mm(&scene, &intr, &true_rot, &[0.01, 0.0, 0.0]),
            &intr,
            3,
        )?;
        let (prior_gt, _) = gt_target_source(&true_rot, &[0.01, 0.0, 0.0]);

        let with = icp_projective_plane(
            &src,
            &tgt,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria {
                rotation_prior: Some(RotationPrior {
                    rotation: prior_gt,
                    sigma_rad: 0.1_f64.to_radians(),
                }),
                ..Default::default()
            },
        )?;

        // The prior rescues the solve — without it roll is an unobserved direction and gets no
        // update at all — but the rotation it rescued must still be REPORTED as unobservable,
        // because it is.
        assert!(
            with.rotation_conditioning < 1e-3,
            "prior inflated reported rotation conditioning to {:.3e}; the geometry did not change",
            with.rotation_conditioning
        );
        Ok(())
    }

    #[test]
    fn conditioning_is_scale_free() -> Result<(), Box<dyn std::error::Error>> {
        let scene = Scene::corner_and_sphere();
        let full = test_intrinsics();
        let half = DepthIntrinsics {
            fx: full.fx / 2.0,
            fy: full.fy / 2.0,
            cx: (full.cx + 0.5) / 2.0 - 0.5,
            cy: (full.cy + 0.5) / 2.0 - 0.5,
            width: full.width / 2,
            height: full.height / 2,
        };
        let slide = [0.02, 0.0, 0.0];

        let mut solved = Vec::new();
        for intr in [&full, &half] {
            let src = RgbdPyramid::from_depth_mm(
                &render_depth_mm(&scene, intr, &IDENTITY_ROT, &[0.0; 3]),
                intr,
                2,
            )?;
            let tgt = RgbdPyramid::from_depth_mm(
                &render_depth_mm(&scene, intr, &IDENTITY_ROT, &slide),
                intr,
                2,
            )?;
            let r = icp_projective_plane(
                &src,
                &tgt,
                IDENTITY_ROT,
                [0.0; 3],
                IcpPlaneCriteria::default(),
            )?;
            solved.push(r);
        }
        let (a, b) = (&solved[0], &solved[1]);
        assert!(
            b.num_associated * 3 < a.num_associated,
            "half resolution should associate far fewer pixels: {} vs {}",
            b.num_associated,
            a.num_associated
        );
        for (name, lo, hi) in [
            ("rotation", a.rotation_conditioning, b.rotation_conditioning),
            (
                "translation",
                a.translation_conditioning,
                b.translation_conditioning,
            ),
        ] {
            let ratio = lo.max(hi) / lo.min(hi).max(f64::MIN_POSITIVE);
            assert!(
                ratio < 10.0,
                "{name} conditioning moved with correspondence count: {lo:.3e} vs {hi:.3e}"
            );
        }
        Ok(())
    }

    /// The rotation prior must rescue weak geometry WITHOUT damaging good geometry, and it must
    /// survive a prior that is itself wrong — a real gyro carries bias, noise and, on hardware
    /// with no IMU extrinsics, a fixed misalignment. A prior fed the exact answer proves nothing.
    #[test]
    fn rotation_prior_rescues_weak_geometry_without_harming_good(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        // Rotation-weak, translation-strong: a sphere centred on the optical axis in front of a
        // fronto-parallel wall. Both are invariant under ROLL, so rotation about the viewing axis
        // is unobservable, while the sphere's normals span every direction and pin all three
        // translations. This isolates the mode the prior exists for — a scene that starves
        // translation as well would be beyond any rotation prior's reach.
        let weak = Scene {
            planes: vec![Plane {
                normal: [0.0, 0.0, 1.0],
                d: 2.5,
            }],
            spheres: vec![Sphere {
                center: [0.0, 0.0, 1.8],
                radius: 0.55,
            }],
        };
        let rich = Scene::corner_and_sphere();
        // Roll about the viewing axis — the unobservable direction on the weak scene.
        let true_rot = axis_angle_to_rotation_matrix(&[0.0, 0.0, 1.0], 3.0_f64.to_radians())?;
        let true_t = [0.01, 0.0, 0.0];

        // The prior is deliberately WRONG by 0.15 deg — well beyond a MEMS gyro's error over one
        // frame, so passing here means it tolerates worse than reality.
        let prior_err_deg: f64 = 0.15;
        let prior_rot = mat3_mul(
            &axis_angle_to_rotation_matrix(&[0.3, 0.6, 0.74], prior_err_deg.to_radians())?,
            &true_rot,
        );

        let mut summary = Vec::new();
        for scene in [&weak, &rich] {
            let src = RgbdPyramid::from_depth_mm(
                &render_depth_mm(scene, &intr, &IDENTITY_ROT, &[0.0; 3]),
                &intr,
                3,
            )?;
            let tgt = RgbdPyramid::from_depth_mm(
                &render_depth_mm(scene, &intr, &true_rot, &true_t),
                &intr,
                3,
            )?;
            let (rot_gt, _) = gt_target_source(&true_rot, &true_t);

            let solve = |prior: Option<RotationPrior>| {
                let mut c = IcpPlaneCriteria {
                    rotation_prior: prior,
                    ..Default::default()
                };
                c.iters_per_level = vec![10, 7, 5];
                icp_projective_plane(&src, &tgt, IDENTITY_ROT, [0.0; 3], c)
            };
            let (prior_gt, _) = gt_target_source(&prior_rot, &true_t);
            // Without the prior roll on the weak scene is an unobserved direction — not merely
            // poorly determined, unconstrained — so the solve leaves it at the initial guess
            // (identity, 3 deg from truth). It used to be rejected as an error; an error still
            // counts as infinite so the two cases compare on one scale.
            let without = solve(None)
                .map(|r| rotation_error_deg(&r.rotation, &rot_gt))
                .unwrap_or(f64::INFINITY);
            let with = solve(Some(RotationPrior {
                rotation: prior_gt,
                sigma_rad: prior_err_deg.to_radians(),
            }))
            .map(|r| rotation_error_deg(&r.rotation, &rot_gt))
            .unwrap_or(f64::INFINITY);
            summary.push((without, with));
        }
        let (weak_without, weak_with) = summary[0];
        let (rich_without, rich_with) = summary[1];

        // Weak geometry: the prior must take over, landing near its own accuracy rather than the
        // solve's. Anything close to `prior_err_deg` means the constraint is carrying rotation.
        assert!(
            weak_without > 0.5 && weak_with < 0.3,
            "prior failed to rescue weak rotation: {weak_without} deg without, {weak_with:.3} with"
        );
        // Good geometry: the prior is wrong by 0.15 deg and must NOT drag the solve toward it.
        // Regularise, never override.
        assert!(
            rich_with < rich_without.max(0.05) * 3.0,
            "prior degraded well-conditioned geometry: {rich_without:.4} deg without, {rich_with:.4} with"
        );
        Ok(())
    }

    /// A single plane used to be rejected as `SingularNormalEquations`. It now gets the partial
    /// answer instead — the three directions the plane cannot see (in-plane translation and
    /// rotation about its normal) come back as degenerate with zero information, and the solve
    /// moves only along the other three — so this test asserts that answer rather than the error.
    #[test]
    fn test_single_plane_is_degenerate() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        // one fronto-parallel plane: in-plane translation and in-plane rotation
        // are unobservable for point-to-plane ICP
        let scene = Scene {
            planes: vec![Plane {
                normal: [0.0, 0.0, 1.0],
                d: 2.0,
            }],
            spheres: vec![],
        };
        let depth = render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]);
        let pyr = RgbdPyramid::from_depth_mm(&depth, &intr, 3)?;

        let result = icp_projective_plane(
            &pyr,
            &pyr,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria::default(),
        )?;
        assert_eq!(
            result.num_degenerate, 3,
            "single plane must report three unobserved directions"
        );
        // nothing moved: the observed directions have nothing to correct, the others no update
        assert!(rotation_error_deg(&result.rotation, &IDENTITY_ROT) < 1e-6);
        assert!(translation_error(&result.translation, &[0.0; 3]) < 1e-9);
        assert!(result.observability < 1e-6);
        Ok(())
    }

    /// `x^T A x` for a 6x6 matrix.
    fn quad(a: &[[f64; 6]; 6], x: &[f64; 6]) -> f64 {
        (0..6)
            .map(|i| (0..6).map(|k| x[i] * a[i][k] * x[k]).sum::<f64>())
            .sum()
    }

    /// Error of an estimate against the truth in the module's tangent: the estimate is
    /// `exp(xi)` on the left of the truth, `xi = [log(R R_gt^T), t - R R_gt^T t_gt]`.
    fn tangent_error(
        rot: &[[f64; 3]; 3],
        t: &[f64; 3],
        rot_gt: &[[f64; 3]; 3],
        t_gt: &[f64; 3],
    ) -> [f64; 6] {
        let rr = mat3_mul(rot, &mat3_transpose(rot_gt));
        let w = so3_log(&rr);
        [
            w[0],
            w[1],
            w[2],
            t[0] - mat3_row_dot(&rr, 0, t_gt),
            t[1] - mat3_row_dot(&rr, 1, t_gt),
            t[2] - mat3_row_dot(&rr, 2, t_gt),
        ]
    }

    /// Deterministic standard normal samples (splitmix64 + Box-Muller).
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

    const TEST_STEREO: DepthNoise = DepthNoise::Stereo {
        focal_px: 200.0,
        baseline_m: 0.075,
        // ~2 mm at 2 m: small enough that central-difference normals stay meaningful
        sigma_disparity_px: 0.0075,
    };

    /// The reported information must be the Hessian of the cost the solver minimises.
    ///
    /// Fixes the final pass's correspondences and their weights, perturbs the estimate in the
    /// module's tangent and takes central second differences of `0.5 * sum w r^2`. At a
    /// converged, noise-free solve the residuals are near zero, so the Gauss-Newton `J^T W J`
    /// and the true Hessian agree; a wrong Jacobian, sign or weight would not.
    #[test]
    fn information_matches_finite_difference_hessian() -> Result<(), Box<dyn std::error::Error>> {
        for association in [
            Association::Nearest,
            Association::Bilinear,
            Association::Hybrid,
        ] {
            eprintln!("finite differences, {association:?}:");
            fd_check(association)?;
        }
        Ok(())
    }

    /// One mode of [`information_matches_finite_difference_hessian`].
    fn fd_check(association: Association) -> Result<(), Box<dyn std::error::Error>> {
        // the final pass of Hybrid is bilinear once it has switched, which it does here
        let bilinear = association != Association::Nearest;
        let intr = test_intrinsics();
        let scene = Scene::corner_and_sphere();
        let rot_wc = axis_angle_to_rotation_matrix(&[1.0, 1.0, 1.0], 1.5_f64.to_radians())?;
        let t_wc = [0.02, -0.01, 0.015];
        let src = RgbdPyramid::from_depth_m(
            &render_depth_m(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]),
            &intr,
            3,
        )?;
        let tgt =
            RgbdPyramid::from_depth_m(&render_depth_m(&scene, &intr, &rot_wc, &t_wc), &intr, 3)?;
        let criteria = IcpPlaneCriteria {
            depth_noise: TEST_STEREO,
            association,
            ..Default::default()
        };
        let res = icp_projective_plane(&src, &tgt, IDENTITY_ROT, [0.0; 3], criteria.clone())?;
        assert_eq!(res.num_degenerate, 0);

        let (s0, t0) = (&src.levels[0], &tgt.levels[0]);
        // (source point, target point, target normal, weight)
        type Row = ([f64; 3], [f64; 3], [f64; 3], f64);
        let mut rows: Vec<Row> = Vec::new();
        for_each_correspondence(
            s0,
            t0,
            &res.rotation,
            &res.translation,
            &criteria,
            bilinear,
            |c| {
                let f = |v: &[f32; 3]| [v[0] as f64, v[1] as f64, v[2] as f64];
                rows.push((
                    f(&s0.vertices[c.src_idx]),
                    c.v_t,
                    c.n_t,
                    row_weight(c.r, c.sigma, criteria.huber_delta),
                ));
            },
        );
        let cost = |xi: &[f64; 6]| -> f64 {
            let dr = so3_exp(&[xi[0], xi[1], xi[2]]);
            let r = mat3_mul(&dr, &res.rotation);
            let t: [f64; 3] =
                std::array::from_fn(|i| mat3_row_dot(&dr, i, &res.translation) + xi[3 + i]);
            rows.iter()
                .map(|(p, v, n, w)| {
                    let q: [f64; 3] = std::array::from_fn(|i| mat3_row_dot(&r, i, p) + t[i]);
                    let e = n[0] * (q[0] - v[0]) + n[1] * (q[1] - v[1]) + n[2] * (q[2] - v[2]);
                    0.5 * w * e * e
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
                let norm = (res.information[j][j] * res.information[k][k]).sqrt();
                worst = worst.max((fd - res.information[j][k]).abs() / norm);
            }
        }
        eprintln!(
            "information vs fixed-target finite differences: worst {worst:.3e} over {} rows",
            rows.len()
        );
        assert!(
            worst < 1e-3,
            "information disagrees with the fixed-target Hessian: {worst:.3e}"
        );

        if !bilinear {
            // nearest-pixel association is piecewise constant: re-associating finite
            // differences only see snapping jumps
            return Ok(());
        }
        // Re-associating at every perturbed pose, so the interpolated target moves with the
        // estimate (weights frozen as Gauss-Newton holds them), measures what holding the target
        // fixed costs: worst entry (t_y, t_y) 7.3%, stable from h = 1e-3 to 3e-5 so not gate
        // noise. Modelling the target's motion cut it to 1.4% but broke NEES on noisy depth
        // (see `for_each_correspondence`), so the gap is bounded here, not closed.
        // weights frozen at the final pass, keyed by source pixel, as Gauss-Newton holds them
        let mut w_fixed = vec![f64::NAN; s0.vertices.len()];
        for_each_correspondence(
            s0,
            t0,
            &res.rotation,
            &res.translation,
            &criteria,
            bilinear,
            |c| {
                w_fixed[c.src_idx] = row_weight(c.r, c.sigma, criteria.huber_delta);
            },
        );
        let reassoc = |xi: &[f64; 6]| -> f64 {
            let dr = so3_exp(&[xi[0], xi[1], xi[2]]);
            let r = mat3_mul(&dr, &res.rotation);
            let t: [f64; 3] =
                std::array::from_fn(|i| mat3_row_dot(&dr, i, &res.translation) + xi[3 + i]);
            let mut e = 0.0;
            for_each_correspondence(s0, t0, &r, &t, &criteria, bilinear, |c| {
                let w = w_fixed[c.src_idx];
                if w.is_finite() {
                    e += 0.5 * w * c.r * c.r;
                }
            });
            e
        };
        let mut worst_re: f64 = 0.0;
        for hh in [1e-3, 3e-4, 1e-4, 3e-5] {
            let mut worst_h: f64 = 0.0;
            let mut worst_at = (0, 0);
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
                    let norm = (res.information[j][j] * res.information[k][k]).sqrt();
                    let err = (fd - res.information[j][k]).abs() / norm;
                    if err > worst_h {
                        worst_h = err;
                        worst_at = (j, k);
                    }
                }
            }
            eprintln!("re-associating FD, h={hh:.0e}: worst {worst_h:.3e} at {worst_at:?}");
            if hh == 1e-4 {
                worst_re = worst_h;
            }
        }
        eprintln!("information vs re-associating finite differences: worst {worst_re:.3e}");
        assert!(
            worst_re < 0.1,
            "fixed-target information is off the re-associated cost by {worst_re:.3e}"
        );
        Ok(())
    }

    /// A wall gives exactly its three blind directions zero information, and the solve takes
    /// the prior along them instead of failing.
    #[test]
    fn wall_takes_prior_along_its_null_space() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let wall = Scene {
            planes: vec![Plane {
                normal: [0.0, 0.0, 1.0],
                d: 2.0,
            }],
            spheres: vec![],
        };
        // roll and in-plane slide are invisible, the approach along z is not
        let rot_wc = axis_angle_to_rotation_matrix(&[0.0, 0.0, 1.0], 1.0_f64.to_radians())?;
        let t_wc = [0.02, -0.01, 0.005];
        let (rot_gt, t_gt) = gt_target_source(&rot_wc, &t_wc);
        let src = RgbdPyramid::from_depth_m(
            &render_depth_m(&wall, &intr, &IDENTITY_ROT, &[0.0; 3]),
            &intr,
            3,
        )?;
        let tgt =
            RgbdPyramid::from_depth_m(&render_depth_m(&wall, &intr, &rot_wc, &t_wc), &intr, 3)?;

        let check_null_space = |r: &IcpPlaneResult| {
            assert_eq!(r.num_degenerate, 3, "wall must have three blind directions");
            let scale = (0..6).map(|i| r.information[i][i]).fold(0.0, f64::max);
            let mut basis = [[0.0; 3]; 3];
            for (row, d) in r.degenerate_directions[..3].iter().enumerate() {
                // only rotation about the normal and in-plane translation
                for &i in &[0usize, 1, 5] {
                    assert!(d[i].abs() < 1e-6, "direction {d:?} leaks into axis {i}");
                }
                assert!(
                    quad(&r.information, d) <= 1e-12 * scale,
                    "information along {d:?}"
                );
                basis[row] = [d[2], d[3], d[4]];
            }
            let det = basis[0][0] * (basis[1][1] * basis[2][2] - basis[1][2] * basis[2][1])
                - basis[0][1] * (basis[1][0] * basis[2][2] - basis[1][2] * basis[2][0])
                + basis[0][2] * (basis[1][0] * basis[2][1] - basis[1][1] * basis[2][0]);
            assert!(
                det.abs() > 0.5,
                "directions do not span w_z, t_x, t_y: {det}"
            );
            assert!(r.information[5][5] > 0.0, "t_z is observed");
        };

        // No prior: the blind directions get no update at all, the observed one is solved.
        let bare = icp_projective_plane(&src, &tgt, IDENTITY_ROT, [0.0; 3], Default::default())?;
        check_null_space(&bare);
        assert!(bare.translation[0].abs() < 1e-6 && bare.translation[1].abs() < 1e-6);
        assert!(so3_log(&bare.rotation)[2].abs() < 1e-6);
        assert!((bare.translation[2] - t_gt[2]).abs() < 1e-4);

        // A prior 3 mm off along x: the blind directions follow it exactly, z still follows the
        // data, and the reported information is unchanged in kind.
        let mut prior_t = t_gt;
        prior_t[0] += 0.003;
        prior_t[2] += 0.004; // wrong along an OBSERVED axis: the data must win there
        let mut cov = [[0.0; 6]; 6];
        for (i, row) in cov.iter_mut().enumerate() {
            row[i] = if i < 3 {
                0.2_f64.to_radians().powi(2)
            } else {
                0.01_f64.powi(2)
            };
        }
        let with = icp_projective_plane(
            &src,
            &tgt,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria {
                pose_prior: Some(PosePrior {
                    rotation: rot_gt,
                    translation: prior_t,
                    covariance: cov,
                }),
                ..Default::default()
            },
        )?;
        check_null_space(&with);
        let e = tangent_error(&with.rotation, &with.translation, &rot_gt, &prior_t);
        eprintln!("wall + prior: error against the prior {e:?}");
        assert!(
            e[3].abs() < 1e-5 && e[4].abs() < 1e-5,
            "slide must follow the prior"
        );
        assert!(e[2].abs() < 1e-5, "roll must follow the prior");
        assert!(
            (with.translation[2] - t_gt[2]).abs() < 2e-4,
            "z must follow the data, not the 4 mm-off prior: {}",
            with.translation[2] - t_gt[2]
        );
        Ok(())
    }

    /// The information is a covariance you can trust: NEES of the estimate over independent
    /// noisy frames must average the 6 of a chi-square with 6 DoF.
    ///
    /// Measured against the truth on a moving pair, NEES is ~117, and most of it is there with
    /// NO noise: the noise-free solve is off by 0.017 mrad / 0.068 mm on corner + sphere (0.18
    /// mrad / 0.31 mm before bilinear association replaced nearest-pixel snapping, when this
    /// NEES was ~226). 57k rows shrink the noise sigma to ~5 um, so tens of microns of model
    /// error is still many sigmas. The information describes the depth noise, not that bias. So
    /// the noise is judged two ways: against the truth where the bias is zero (identical poses),
    /// and against the noise-free solve where it is not.
    #[test]
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

    /// One mode of [`nees_is_six_on_noisy_stereo_depth`]; same noise seed for every mode.
    fn nees_for(association: Association) -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let scene = Scene::corner_and_sphere();
        let criteria = IcpPlaneCriteria {
            depth_noise: TEST_STEREO,
            association,
            ..Default::default()
        };
        let mut g = Gauss(7);
        let mut noisy = |clean: &[f32]| -> Vec<f32> {
            clean
                .iter()
                .map(|&z| {
                    if z > 0.0 {
                        z + (TEST_STEREO.depth_sigma(z as f64) * g.sample()) as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        let moving = (
            axis_angle_to_rotation_matrix(&[0.3, 1.0, 0.2], 1.0_f64.to_radians())?,
            [0.02, 0.0, 0.01],
        );
        let trials = 40;
        for (name, (rot_wc, t_wc)) in [("static", (IDENTITY_ROT, [0.0; 3])), ("moving", moving)] {
            let (rot_gt, t_gt) = gt_target_source(&rot_wc, &t_wc);
            let clean_src = render_depth_m(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]);
            let clean_tgt = render_depth_m(&scene, &intr, &rot_wc, &t_wc);
            let clean = icp_projective_plane(
                &RgbdPyramid::from_depth_m(&clean_src, &intr, 3)?,
                &RgbdPyramid::from_depth_m(&clean_tgt, &intr, 3)?,
                IDENTITY_ROT,
                [0.0; 3],
                criteria.clone(),
            )?;
            let e0 = tangent_error(&clean.rotation, &clean.translation, &rot_gt, &t_gt);
            let (mut vs_truth, mut vs_clean) = (Vec::new(), Vec::new());
            for _ in 0..trials {
                let src = RgbdPyramid::from_depth_m(&noisy(&clean_src), &intr, 3)?;
                let tgt = RgbdPyramid::from_depth_m(&noisy(&clean_tgt), &intr, 3)?;
                let r = icp_projective_plane(&src, &tgt, IDENTITY_ROT, [0.0; 3], criteria.clone())?;
                assert_eq!(r.num_degenerate, 0);
                let e = tangent_error(&r.rotation, &r.translation, &rot_gt, &t_gt);
                let d: [f64; 6] = std::array::from_fn(|i| e[i] - e0[i]);
                vs_truth.push(quad(&r.information, &e));
                vs_clean.push(quad(&r.information, &d));
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let (m_truth, m_clean) = (mean(&vs_truth), mean(&vs_clean));
            let rot_bias = (e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2]).sqrt();
            let t_bias = (e0[3] * e0[3] + e0[4] * e0[4] + e0[5] * e0[5]).sqrt();
            eprintln!(
                "NEES {association:?} {name}, {trials} trials: vs truth {m_truth:.2}, vs \
                 noise-free solve {m_clean:.2}; noise-free bias {:.2e} rad / {:.3} mm {:?}",
                rot_bias,
                t_bias * 1e3,
                e0.map(|v| format!("{v:.1e}"))
            );
            // chi2(6): mean 6, sd of a 40-trial mean ~0.55
            assert!(
                (4.0..9.0).contains(&m_clean),
                "{name}: NEES {m_clean:.2} about the noise-free solve is not consistent with 6"
            );
            if name == "static" {
                assert!(
                    (4.0..9.0).contains(&m_truth),
                    "static: NEES {m_truth:.2} against the truth is not consistent with 6"
                );
            }
        }
        Ok(())
    }

    /// `RotationPrior` is a `PosePrior` with no translation information: the same weighted
    /// rows, so the same answer.
    #[test]
    fn rotation_prior_equals_pose_prior_without_translation(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let scene = Scene {
            planes: vec![Plane {
                normal: [0.0, 0.0, 1.0],
                d: 2.5,
            }],
            spheres: vec![Sphere {
                center: [0.0, 0.0, 1.8],
                radius: 0.55,
            }],
        };
        let true_rot = axis_angle_to_rotation_matrix(&[0.0, 0.0, 1.0], 3.0_f64.to_radians())?;
        let src = RgbdPyramid::from_depth_mm(
            &render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]),
            &intr,
            3,
        )?;
        let tgt = RgbdPyramid::from_depth_mm(
            &render_depth_mm(&scene, &intr, &true_rot, &[0.01, 0.0, 0.0]),
            &intr,
            3,
        )?;
        let (prior_rot, _) = gt_target_source(&true_rot, &[0.01, 0.0, 0.0]);
        let sigma = 0.1_f64.to_radians();
        let a = icp_projective_plane(
            &src,
            &tgt,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria {
                rotation_prior: Some(RotationPrior {
                    rotation: prior_rot,
                    sigma_rad: sigma,
                }),
                ..Default::default()
            },
        )?;
        let mut cov = [[0.0; 6]; 6];
        for (i, row) in cov.iter_mut().enumerate() {
            // translation variance huge: (1 km)^2
            row[i] = if i < 3 { sigma * sigma } else { 1e6 };
        }
        let b = icp_projective_plane(
            &src,
            &tgt,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria {
                pose_prior: Some(PosePrior {
                    rotation: prior_rot,
                    translation: [5.0, -3.0, 1.0], // irrelevant at that variance
                    covariance: cov,
                }),
                ..Default::default()
            },
        )?;
        assert!(rotation_error_deg(&a.rotation, &b.rotation) < 1e-4);
        assert!(translation_error(&a.translation, &b.translation) < 1e-6);
        // and the prior never shows in the reported information
        assert!(a.rotation_conditioning < 1e-3 && b.rotation_conditioning < 1e-3);

        let mut bad = cov;
        bad[0][0] = -1.0;
        let err = icp_projective_plane(
            &src,
            &tgt,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria {
                pose_prior: Some(PosePrior {
                    rotation: prior_rot,
                    translation: [0.0; 3],
                    covariance: bad,
                }),
                ..Default::default()
            },
        );
        assert!(matches!(err, Err(RgbdIcpError::InvalidPriorCovariance)));
        Ok(())
    }

    /// A per-pixel sigma map equal to the stereo model must weight like the model.
    #[test]
    fn sigma_map_matches_stereo_model() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let scene = Scene::corner_and_sphere();
        let rot_wc = axis_angle_to_rotation_matrix(&[0.0, 1.0, 0.0], 1.0_f64.to_radians())?;
        let t_wc = [0.01, 0.0, 0.0];
        let ds = render_depth_m(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]);
        let dt = render_depth_m(&scene, &intr, &rot_wc, &t_wc);
        let sig = |d: &[f32]| -> Vec<f32> {
            d.iter()
                .map(|&z| TEST_STEREO.depth_sigma(z as f64) as f32)
                .collect()
        };
        let model = icp_projective_plane(
            &RgbdPyramid::from_depth_m(&ds, &intr, 3)?,
            &RgbdPyramid::from_depth_m(&dt, &intr, 3)?,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria {
                depth_noise: TEST_STEREO,
                ..Default::default()
            },
        )?;
        // the map wins over the (deliberately wrong) constant model
        let mapped = icp_projective_plane(
            &RgbdPyramid::from_depth_m_with_sigma(&ds, &sig(&ds), &intr, 3)?,
            &RgbdPyramid::from_depth_m_with_sigma(&dt, &sig(&dt), &intr, 3)?,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria {
                depth_noise: DepthNoise::Constant { sigma_m: 1.0 },
                ..Default::default()
            },
        )?;
        for i in 0..6 {
            let rel = (model.information[i][i] - mapped.information[i][i]).abs()
                / model.information[i][i];
            assert!(rel < 1e-3, "information[{i}][{i}] differs by {rel:.3e}");
        }
        Ok(())
    }

    /// The default degeneracy threshold sits between an exact null space and the weakest
    /// geometry the solver has always solved.
    ///
    /// Measured smallest/largest eigenvalue of the depth-scaled data matrix after a 20 mm slide,
    /// same on mm and float depth: wall 0 (three directions), wall + 6 cm bump 5.1e-8, roll scene
    /// 2.7e-5, corner + sphere 2.8e-2. Real sensor noise lifts a wall's null space far above 0,
    /// so a caller gating on real data sets its own threshold.
    #[test]
    fn default_degeneracy_rtol_separates_null_space_from_weak_geometry(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let plane = || Plane {
            normal: [0.0, 0.0, 1.0],
            d: 2.0,
        };
        let wall = Scene {
            planes: vec![plane()],
            spheres: vec![],
        };
        let bump = Scene {
            planes: vec![plane()],
            spheres: vec![Sphere {
                center: [0.0, 0.0, 1.9],
                radius: 0.06,
            }],
        };
        let mut smallest = Vec::new();
        for scene in [&wall, &bump, &Scene::corner_and_sphere()] {
            let pyr = |t: &[f64; 3]| {
                RgbdPyramid::from_depth_mm(
                    &render_depth_mm(scene, &intr, &IDENTITY_ROT, t),
                    &intr,
                    3,
                )
            };
            let (s, t) = (pyr(&[0.0; 3])?, pyr(&[0.02, 0.0, 0.0])?);
            let c = IcpPlaneCriteria::default();
            let r = icp_projective_plane(&s, &t, IDENTITY_ROT, [0.0; 3], c.clone())?;
            let eqs = accumulate_level(
                &s.levels[0],
                &t.levels[0],
                &r.rotation,
                &r.translation,
                &c,
                false,
                Some(&mut Vec::new()),
            );
            let (pairs, lmax) = scaled_eigen(&eqs.a, eqs.depth_scale);
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

    /// The decimated input contract keeps the motion grid's accuracy: registering on the input
    /// halved once (160x90) recovers every motion within the full-resolution test's bounds.
    #[test]
    fn decimated_input_recovers_the_motion_grid() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let scene = Scene::corner_and_sphere();
        let mut scratch = super::super::rgbd::PyramidScratch::default();
        let decimated = |depth: &[u16], scratch: &mut _| -> Result<RgbdPyramid, RgbdIcpError> {
            let mut p = RgbdPyramid { levels: Vec::new() };
            let k = super::super::rgbd::halvings_to_fit(intr.width, 160);
            p.rebuild_decimated_from_depth_mm(depth, &intr, k, 3, IcpKernel::Simd, scratch)?;
            Ok(p)
        };
        let src_depth = render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]);
        let src_full = RgbdPyramid::from_depth_mm(&src_depth, &intr, 3)?;
        let src_dec = decimated(&src_depth, &mut scratch)?;
        assert_eq!(src_dec.levels[0].intrinsics.width, 160);
        let cases: &[([f64; 3], f64, [f64; 3])] = &[
            ([0.0, 1.0, 0.0], -2.0, [0.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], 0.0, [0.03, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.03]),
            ([1.0, 1.0, 1.0], 2.0, [0.02, -0.02, 0.015]),
        ];
        for (axis, angle_deg, t_wc) in cases {
            let rot_wc = axis_angle_to_rotation_matrix(axis, angle_deg.to_radians())?;
            let tgt_depth = render_depth_mm(&scene, &intr, &rot_wc, t_wc);
            let (rot_gt, t_gt) = gt_target_source(&rot_wc, t_wc);
            let full = icp_projective_plane(
                &src_full,
                &RgbdPyramid::from_depth_mm(&tgt_depth, &intr, 3)?,
                IDENTITY_ROT,
                [0.0; 3],
                IcpPlaneCriteria::default(),
            )?;
            let dec = icp_projective_plane(
                &src_dec,
                &decimated(&tgt_depth, &mut scratch)?,
                IDENTITY_ROT,
                [0.0; 3],
                IcpPlaneCriteria::default(),
            )?;
            let err = |r: &IcpPlaneResult| {
                (
                    rotation_error_deg(&r.rotation, &rot_gt),
                    translation_error(&r.translation, &t_gt) * 1e3,
                )
            };
            let (ef, ed) = (err(&full), err(&dec));
            eprintln!(
                "{axis:?}/{angle_deg} deg/{t_wc:?}: full {:.4} deg {:.3} mm, decimated {:.4} deg \
                 {:.3} mm",
                ef.0, ef.1, ed.0, ed.1
            );
            assert!(ed.0 < 0.2 && ed.1 < 5.0, "decimated: {ed:?}");
            assert_eq!(dec.num_degenerate, 0);
        }
        Ok(())
    }

    /// Largest |a - b| / max(|a|, |b|, floor) over two equally long slices.
    fn rel_diff(a: &[f64], b: &[f64], floor: f64) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(floor))
            .fold(0.0, f64::max)
    }

    /// The SIMD kernel must give the scalar kernel's answer to `f32` rounding: pose and
    /// information within 1e-4 relative, every association mode, both noise models, with and
    /// without a prior. Its `f32` projection can round a point on a pixel boundary to the
    /// neighbouring pixel, so the correspondence counts may differ by a few in 1e4.
    #[test]
    fn simd_kernel_matches_scalar() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let scene = Scene::corner_and_sphere();
        let rot_wc = axis_angle_to_rotation_matrix(&[0.3, 1.0, 0.2], 1.5_f64.to_radians())?;
        let t_wc = [0.02, -0.01, 0.015];
        let src = RgbdPyramid::from_depth_mm(
            &render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]),
            &intr,
            3,
        )?;
        let tgt =
            RgbdPyramid::from_depth_mm(&render_depth_mm(&scene, &intr, &rot_wc, &t_wc), &intr, 3)?;
        let (rot_gt, t_gt) = gt_target_source(&rot_wc, &t_wc);
        let mut worst: f64 = 0.0;
        for association in [
            Association::Nearest,
            Association::Bilinear,
            Association::Hybrid,
        ] {
            for noise in [DepthNoise::Constant { sigma_m: 0.01 }, TEST_STEREO] {
                for prior in [false, true] {
                    let criteria = IcpPlaneCriteria {
                        association,
                        depth_noise: noise,
                        rotation_prior: prior.then_some(RotationPrior {
                            rotation: rot_gt,
                            sigma_rad: 0.2_f64.to_radians(),
                        }),
                        ..Default::default()
                    };
                    let run = |kernel| {
                        icp_projective_plane_with(
                            &mut IcpWorkspace::new(kernel),
                            &src,
                            &tgt,
                            IDENTITY_ROT,
                            [0.0; 3],
                            &criteria,
                        )
                    };
                    let (a, b) = (run(IcpKernel::Scalar)?, run(IcpKernel::Simd)?);
                    let flat = |r: &IcpPlaneResult| -> Vec<f64> {
                        r.rotation.iter().flatten().copied().collect()
                    };
                    let info = |r: &IcpPlaneResult| -> Vec<f64> {
                        r.information.iter().flatten().copied().collect()
                    };
                    let scale = info(&a).iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    let d = [
                        rel_diff(&flat(&a), &flat(&b), 1.0),
                        rel_diff(&a.translation, &b.translation, 1e-2),
                        rel_diff(&info(&a), &info(&b), scale),
                        rel_diff(
                            &[a.rmse, a.median_depth_m],
                            &[b.rmse, b.median_depth_m],
                            1e-9,
                        ),
                    ];
                    let m = d.iter().fold(0.0f64, |m, v| m.max(*v));
                    let assoc = a.num_associated.abs_diff(b.num_associated);
                    eprintln!(
                        "{association:?} {noise:?} prior {prior}: {d:?}, associated {} vs \
                         {}, iterations {} vs {}",
                        a.num_associated, b.num_associated, a.iterations, b.iterations
                    );
                    assert!(
                        assoc * 1000 <= a.num_associated,
                        "{association:?}: associated {} vs {}",
                        a.num_associated,
                        b.num_associated
                    );
                    assert!(
                        m < 1e-4,
                        "{association:?} {noise:?} prior {prior}: kernels differ by {d:?}"
                    );
                    assert!(translation_error(&b.translation, &t_gt) < 5e-3);
                    worst = worst.max(m);
                }
            }
        }
        eprintln!("SIMD vs scalar ICP: worst relative difference {worst:.2e}");
        Ok(())
    }
}
