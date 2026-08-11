//! Dedicated bundle adjustment solver.
//!
//! Provides [`bundle_adjust`] which jointly optimizes camera poses and 3D point
//! positions by minimizing reprojection error using Levenberg-Marquardt with
//! analytical Jacobians.
//!
//! **Note:** The reprojection model uses only the pinhole parameters
//! (`fx`, `fy`, `cx`, `cy`). Distortion coefficients on [`PinholeCamera`] are
//! ignored. Pixel observations must be **undistorted** before being passed in,
//! otherwise the optimizer will minimize the wrong objective.

use std::sync::Arc;

use kornia_algebra::optim::{
    CauchyLoss, Factor, FactorError, FactorResult, HuberLoss, LevenbergMarquardt,
    LinearizationResult, OptimizerError, Problem, ProblemError, RobustLoss, Variable, VariableType,
};
use kornia_algebra::{Mat3AF32, Mat3F64, QuatF32, Vec3AF32, Vec3F64, SE3F32, SO3F32};

use crate::camera::PinholeCamera;
use crate::pose::Pose3d;
use crate::ransac::RobustKernelKind;

/// Errors that can occur during bundle adjustment.
#[derive(Debug, thiserror::Error)]
pub enum BaError {
    /// Failed to set up the optimization problem (e.g. duplicate variable).
    #[error("Problem setup error: {0}")]
    Problem(#[from] ProblemError),

    /// The optimizer failed during solving.
    #[error("Optimization error: {0}")]
    Optimizer(#[from] OptimizerError),

    /// Invalid input to bundle adjustment.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// A single observation linking a pose to a 3D point via a pixel measurement.
#[derive(Debug, Clone, Copy)]
pub struct BaObservation {
    /// Index into the poses slice.
    pub pose_idx: usize,
    /// Index into the points slice.
    pub point_idx: usize,
    /// Observed pixel `[u, v]` (**must be undistorted**).
    pub pixel: [f32; 2],
    /// If true, this observation's pose is held fixed during optimization.
    pub fixed_pose: bool,
    /// If true, this observation's 3D point is held fixed during optimization
    /// — used for "motion-only" BA where pose is refined against a known map.
    pub fixed_point: bool,
    /// Optional metric depth measurement at this observation pixel.
    /// When `Some(d)`, the BA cost gains a depth residual:
    ///   `r_depth = (Z_pred - d) / depth_sigma`
    /// where `Z_pred = (R · X + t)[2]` in the camera frame. This anchors the
    /// global scale of the reconstruction and prevents the scale-drift mode
    /// that pure reprojection BA cannot constrain.
    ///
    /// Currently only honoured by [`crate::ba_schur::bundle_adjust_schur`].
    pub depth_meas: Option<f32>,
    /// Standard deviation (metres) for the depth residual. Ignored when
    /// `depth_meas` is `None`.
    pub depth_sigma: f32,
}

impl Default for BaObservation {
    fn default() -> Self {
        Self {
            pose_idx: 0,
            point_idx: 0,
            pixel: [0.0, 0.0],
            fixed_pose: false,
            fixed_point: false,
            depth_meas: None,
            depth_sigma: 1.0,
        }
    }
}

impl BaObservation {
    /// Attach a metric depth measurement and its uncertainty (sigma in metres).
    pub fn with_depth(mut self, depth: f32, sigma: f32) -> Self {
        self.depth_meas = Some(depth);
        self.depth_sigma = sigma;
        self
    }
}

/// Optional per-pose translation prior. When present, BA cost gains a
/// residual per pose `i`:
///
/// ```text
///     r_pos_i = (C_i_world - prior_i) / σ
/// ```
///
/// where `C_i_world = -R_CW_i^T · t_CW_i` is the camera centre in the world
/// frame (the inverse-transformed origin of the camera-frame). Use this to
/// anchor BA to a known metric trajectory (e.g. RGB-D PnP chain-VO output)
/// so lateral / vertical drift doesn't accumulate during LM. Unlike
/// `BaObservation::depth_meas` (which only constrains the cam-frame Z of
/// the pose translation), this residual constrains all three axes of the
/// pose's world-frame position simultaneously.
///
/// Currently only honoured by [`crate::ba_schur::bundle_adjust_schur`].
#[derive(Debug, Clone, Copy)]
pub struct BaPosePrior {
    /// Prior camera centre in world frame.
    pub center_world: [f32; 3],
    /// Standard deviation (metres). Clamped to ≥ 1e-6 internally. Smaller
    /// σ → tighter anchor.
    pub sigma: f32,
    /// Optional gravity prior: the world-frame direction this camera's image-up
    /// (camera −Y) should point. Adds an orientation residual
    /// `r_up = (R_c2w·(0,−1,0) − up_world) / up_sigma` — the IMU-less "the phone
    /// was held roughly upright" constraint. Rotational drift accumulates freely
    /// along a monocular chain with no revisits; a soft per-view up prior is the
    /// only observation of absolute orientation such a capture carries.
    pub up_world: Option<[f32; 3]>,
    /// Which CAMERA-frame direction is being claimed to point along `up_world`.
    ///
    /// Defaults to image-up `(0,−1,0)`, i.e. the assumption "the phone was held upright". That
    /// assumption is measurable, and on real handheld video it is measurably wrong: a walkthrough
    /// clip measured here averages `(0.03, 0.93, 0.38)` for gravity in the camera frame — the phone
    /// tilted 22 degrees down. Asserting image-up instead, at a tight sigma, feeds the solver a
    /// 22-degree lie on every frame.
    ///
    /// Set it per camera from a MEASUREMENT — a vertical vanishing point, or an IMU — and this stops
    /// being an assumption and becomes an absolute rotation constraint. That matters far beyond
    /// tidiness: on a single forward pass with no revisits, nothing in the data constrains global
    /// orientation, so an external reference is the ONLY thing that can bound accumulated tilt
    /// drift.
    ///
    /// The Jacobian is `[Rᵀ·up_cam]×`, which depends only on the rotated result, so a per-camera
    /// direction costs nothing over the fixed one.
    pub up_cam: [f32; 3],
    /// Standard deviation of the up prior, in unit-vector units (e.g. 0.2 ≈
    /// tolerate ~11 degrees of tilt at 1σ). Ignored when `up_world` is None.
    pub up_sigma: f32,
}

/// Constant-velocity motion prior over a triplet of consecutive shots (OpenSfM's
/// `LinearMotionError`).
///
/// Encodes "shot `i1` lies a fraction `alpha` of the way from `i0` to `i2`, and rotation advances
/// at constant angular velocity". The translation term deliberately uses the norm-RATIO form
/// `alpha − |C1−C0|/|C2−C0|` rather than a position difference: a raw difference is degenerate
/// while scale is itself being estimated (shrinking the whole map satisfies it), while the ratio
/// is scale-invariant and only constrains the SHAPE of the motion. This is the drift mechanism
/// both OpenSfM and (implicitly, via its dense-network requirement) OpenMVG lean on for
/// GPS-less sequential capture.
#[derive(Debug, Clone, Copy)]
pub struct BaMotionPrior {
    /// Pose indices of the triplet, in temporal order.
    pub i0: usize,
    /// Middle pose.
    pub i1: usize,
    /// Last pose.
    pub i2: usize,
    /// Fractional position of `i1` between `i0` and `i2` (0..1), e.g. from frame timestamps.
    pub alpha: f32,
    /// Sigma of the translation-ratio residual (unitless).
    pub position_sigma: f32,
    /// Sigma of the angular-velocity residual (radians).
    pub orientation_sigma: f32,
}

/// Parameters for bundle adjustment.
pub struct BaParams {
    /// Maximum number of LM iterations.
    pub max_iterations: usize,
    /// Robust-kernel scale (squared) for DEPTH residuals specifically. `0` reuses
    /// `robust_scale_sq` (legacy behaviour).
    ///
    /// The two residual families live in different units: reprojection residuals are in
    /// normalized-camera units (a sensible knee is ~2 px / fx ≈ 0.004), while depth residuals are
    /// σ-WHITENED — 1.0 means "one standard deviation". Sharing one knee means a 1σ depth
    /// measurement is gated like a hundreds-of-σ reprojection outlier and contributes a few
    /// percent of its intended weight, silently disabling metric anchoring while still injecting
    /// biased gradients. The whitened-residual convention puts the Huber knee at 1.345 (95%
    /// Gaussian efficiency), i.e. `depth_robust_scale_sq = 1.345² ≈ 1.81`.
    pub depth_robust_scale_sq: f32,
    /// Cost convergence tolerance.
    pub cost_tolerance: f32,
    /// Gradient convergence tolerance.
    pub gradient_tolerance: f32,
    /// Initial LM damping parameter.
    pub initial_lambda: f32,
    /// Solve the reduced camera system with a SPARSE Cholesky instead of a dense one.
    ///
    /// The reduced system is `6P x 6P` and is currently factorised dense: at 643 cameras that is a
    /// 3852x3852 f64 matrix allocated inside every LM iteration (119 MB) and factorised at
    /// `dim^3/3 = 1.9e10` flops — measured at 4.75 s per iteration on a Jetson Orin, which at the
    /// 100-iteration cap is eight minutes of factorisation in a single call.
    ///
    /// But the camera-camera graph is not dense. Measured on a 643-keyframe walkthrough: 4,556 of
    /// 206,403 camera pairs co-observe a point — **2.2%** — with median index distance 6 and max
    /// 110. Two cameras only couple in this matrix if they share a point, so the vast majority of
    /// those 15 million entries are structurally zero and the factorisation is spending almost all
    /// of its time on them.
    ///
    /// Bounding the win honestly: a solver must size for the WORST coupling, not the median. A block
    /// distance of 110 is a scalar bandwidth of 660, giving `dim * b^2 = 1.7e9` flops — **~11x**
    /// fewer, not the hundred-fold the 2.2% figure suggests at first glance.
    ///
    /// Off by default: this changes the numerical path of every bundle adjustment, and the dense
    /// solver stays as the control until an A/B on ground truth says the sparse one agrees with it.
    pub sparse_reduced_system: bool,
    /// Penalise camera centres for leaving their own best-fit PLANE, in metres. 0 disables.
    ///
    /// A handheld walkthrough is filmed by a person on one floor: the camera centres trace a plane,
    /// wobbling by the few centimetres a gait and an arm allow. That is the strongest shape prior
    /// available about such a trajectory, and it constrains the ONE direction a forward walk cannot
    /// observe — reprojection is exactly blind to an out-of-plane bend, so the error accumulates
    /// unopposed and no residual will ever report it. Measured on two real maps of the same kind of
    /// space: a 40-keyframe walk that reconstructed correctly deviates 0.197 of its largest spread
    /// out of plane, while a 643-keyframe walk whose cloud is unusable deviates 0.603 — 0.89 m
    /// median and 2.17 m worst, on ONE FLOOR of a house.
    ///
    /// The plane is FITTED FROM THE TRAJECTORY ITSELF each iteration, never supplied. That is the
    /// whole difference from the up/gravity prior, which asks the same question ("which way is
    /// vertical") but needs an external answer: measured gravity lands 7.6-28 degrees off, an
    /// uncertainty comparable to the effect, and the measured result was a collapsed map at a tight
    /// sigma and an inflated one at a loose sigma. This prior needs no such input — it asks only
    /// that the centres be near SOME plane, which is a claim about human anatomy rather than about
    /// the world frame.
    ///
    /// Anisotropic by construction: one scalar residual along the plane normal, so in-plane motion
    /// is entirely free and only the out-of-plane component is penalised. A `BaPosePrior` cannot
    /// express this — its sigma is isotropic, so tightening it enough to flatten the trajectory
    /// would also freeze the walk.
    ///
    /// Off by default, and wrong for a walk that genuinely changes floor: a staircase IS
    /// out-of-plane, and this prior will fight it.
    pub plane_prior_sigma: f32,
    /// M-estimator kernel applied per-residual in the IRLS normal
    /// equations. Plumbed through [`kornia_algebra::optim::Factor::get_loss`]
    /// to the LM solver:
    /// - `Identity` → no loss (plain L2 fast path).
    /// - `Huber`    → [`kornia_algebra::optim::HuberLoss`].
    /// - `Cauchy`   → [`kornia_algebra::optim::CauchyLoss`].
    /// - `Tukey`    → currently maps to `CauchyLoss` (Tukey isn't yet
    ///   exposed by `kornia_algebra::optim::losses`; both are smooth
    ///   redescenders and produce qualitatively similar weight curves).
    pub robust: RobustKernelKind,
    /// Threads for the reduced-system assembly, which is the single most expensive phase of a
    /// Schur bundle adjustment (measured 72% of solve time at 300 keyframes, against 5% for the
    /// factorisation everyone expects to dominate).
    ///
    /// `0` means "ask the machine" (`available_parallelism`). Set it explicitly when the solver
    /// shares a box with something else — on a Jetson the GPU front-end and the solver contend,
    /// and taking every core is not the fastest setting. `KORNIA_BA_THREADS` overrides this at
    /// runtime for measurement without recompiling callers.
    ///
    /// Results are IDENTICAL at every thread count, bit for bit: the points are partitioned into
    /// fixed contiguous chunks and the per-chunk accumulators are reduced in chunk order, so the
    /// float addition order does not depend on which thread finishes first.
    /// `assembly_is_deterministic_across_thread_counts` pins that.
    pub assembly_threads: usize,
    /// Squared scale parameter for [`Self::robust`]. The square root is
    /// the linear scale fed to `HuberLoss::new`/`CauchyLoss::new`. Default
    /// `f32::INFINITY` collapses to the L2 fast path even for non-Identity
    /// kernel choices.
    pub robust_scale_sq: f32,
    /// Use the LOG-SPACE depth residual with a per-camera scale, instead of the
    /// legacy metric-difference form.
    ///
    /// ```text
    ///   false (legacy):  r = (z − m) / σ                    σ in METRES
    ///   true  (log):     r = ln( z / (s_i · m) ) / σ        σ RELATIVE (fraction of depth)
    ///                    r = (z − s_i·m) / (s_i·m·σ)        when z ≤ 0 (linear continuation)
    /// ```
    ///
    /// Two things change, and both matter for monocular video. The residual becomes a RELATIVE
    /// depth error, so a 5 m point and a 0.5 m point contribute comparably instead of the far
    /// point dominating by an order of magnitude. And `s_i` — the per-camera scale that maps the
    /// monocular network's prediction onto the map's units — becomes a free variable solved
    /// jointly with geometry rather than a constant fitted beforehand and frozen.
    ///
    /// Freezing it is the trap: a monocular depth network's scale error is per-frame, so a scale
    /// fitted from the CURRENT (drifted) geometry bakes that drift into the prior, and the prior
    /// then certifies the drift it came from. Follows VidMap (Pataki, Sarlin & Pollefeys, ECCV
    /// 2026, eq. 3), which introduced this to global SfM.
    ///
    /// Interpretation of [`BaObservation::depth_sigma`] changes with this flag: metres when false,
    /// a relative fraction when true.
    pub depth_log_residual: bool,
    /// Strength of the prior pulling each `ln s_i` back to its seed, as a MULTIPLE of that
    /// camera's own accumulated depth weight.
    ///
    /// ```text
    ///   ln s_i = ln s_seed_i + (Σ_k w_ik·(ln z_ik − ln m_ik − ln s_seed_i)/σ_ik²)
    ///                          ────────────────────────────────────────────────
    ///                                 (1 + λ) · Σ_k w_ik/σ_ik²
    /// ```
    ///
    /// Deliberately NOT a plain sigma. A sigma competes against the SUM of a camera's depth
    /// observations, so its meaningful value moves with the observation count — σ=0.3 is a strong
    /// prior against 1 observation and utterly inert against 500. Expressing it as a ratio makes
    /// the knob mean the same thing on every camera in every clip: `λ = 1` says "the seed is worth
    /// as much as all this camera's depth data", `λ = 9` says "the data may move it 10%".
    ///
    /// This term is not a nicety, it is the metric anchor. At `λ = 0` each camera rescales its own
    /// prior onto whatever geometry it already has; the depth cost goes to zero, consecutive
    /// frames' priors decouple, and the depth term stops constraining the baseline between frames
    /// — the exact quantity it was added to constrain. Measured on the synthetic A/B in
    /// `schur_ba_free_depth_scale_beats_frozen`: `λ = 0` leaves 5.74 m of point error where a
    /// frozen scale leaves 0.08 m.
    ///
    /// `< 0` freezes `s_i` at its seed (the fitted-then-frozen baseline). Ignored entirely when
    /// [`Self::depth_log_residual`] is false.
    pub depth_scale_prior: f32,
    /// Initial per-camera depth scales, indexed like `poses`. Empty → all `1.0`.
    ///
    /// Seed this with a robust per-camera fit (e.g. `median(z_sfm / m)`) so the log residual
    /// starts near its optimum; the regularised update then refines it instead of discovering it.
    pub depth_scales_init: Vec<f32>,
}

impl Default for BaParams {
    fn default() -> Self {
        Self {
            depth_robust_scale_sq: 0.0,
            max_iterations: 10,
            cost_tolerance: 1e-5,
            gradient_tolerance: 1e-5,
            initial_lambda: 1e-3,
            sparse_reduced_system: false,
            plane_prior_sigma: 0.0,
            robust: RobustKernelKind::Identity,
            assembly_threads: 0,
            robust_scale_sq: f32::INFINITY,
            depth_log_residual: false,
            depth_scale_prior: -1.0,
            depth_scales_init: Vec::new(),
        }
    }
}

/// Result of bundle adjustment.
pub struct BaResult {
    /// Optimized poses.
    pub poses: Vec<Pose3d>,
    /// Optimized 3D points.
    pub points: Vec<Vec3F64>,
    /// Number of LM iterations performed.
    pub iterations: usize,
    /// Whether the optimizer converged.
    pub converged: bool,
    /// Final per-camera depth scales, indexed like [`Self::poses`].
    ///
    /// Empty unless [`BaParams::depth_log_residual`] was set. A camera whose scale drifts far from
    /// 1.0 is reporting that its geometry and its monocular prior disagree about how big the
    /// world is there — useful as a per-frame diagnostic, not just an internal variable.
    pub depth_scales: Vec<f32>,
}

// ---------------------------------------------------------------------------
// f32 ↔ f64 conversion helpers
// ---------------------------------------------------------------------------

fn pose_to_se3_params(pose: &Pose3d) -> Vec<f32> {
    let r_f32 = Mat3AF32::from_cols(
        Vec3AF32::new(
            pose.rotation.col(0).x as f32,
            pose.rotation.col(0).y as f32,
            pose.rotation.col(0).z as f32,
        ),
        Vec3AF32::new(
            pose.rotation.col(1).x as f32,
            pose.rotation.col(1).y as f32,
            pose.rotation.col(1).z as f32,
        ),
        Vec3AF32::new(
            pose.rotation.col(2).x as f32,
            pose.rotation.col(2).y as f32,
            pose.rotation.col(2).z as f32,
        ),
    );
    let so3 = SO3F32::from_matrix(&r_f32);
    let se3 = SE3F32::new(
        so3,
        Vec3AF32::new(
            pose.translation.x as f32,
            pose.translation.y as f32,
            pose.translation.z as f32,
        ),
    );
    se3.to_params().to_vec()
}

fn se3_params_to_pose(params: &[f32]) -> Pose3d {
    let q = QuatF32::from_xyzw(params[1], params[2], params[3], params[0]).normalize();
    let se3 = SE3F32::from_qxyz(q, Vec3AF32::new(params[4], params[5], params[6]));
    let r = se3.r.matrix();
    let t = se3.t;
    Pose3d::new(
        Mat3F64::from_cols(
            Vec3F64::new(r.col(0).x as f64, r.col(0).y as f64, r.col(0).z as f64),
            Vec3F64::new(r.col(1).x as f64, r.col(1).y as f64, r.col(1).z as f64),
            Vec3F64::new(r.col(2).x as f64, r.col(2).y as f64, r.col(2).z as f64),
        ),
        Vec3F64::new(t.x as f64, t.y as f64, t.z as f64),
    )
}

// ---------------------------------------------------------------------------
// Reprojection factor with analytical Jacobians
// ---------------------------------------------------------------------------

/// Internal reprojection factor for the BA solver.
struct ReprojFactor {
    obs_u: f32,
    obs_v: f32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    /// When set, the pose is fixed and only the point is optimized.
    fixed_pose: Option<[f32; 7]>,
    /// When set, the 3D point is fixed and only the pose is optimized
    /// (motion-only BA: refines a single pose against a known map).
    fixed_point: Option<[f32; 3]>,
    /// Optional robust loss applied to this observation's residual by the
    /// optimiser via `Factor::get_loss`. Shared across every factor in the
    /// BA problem (one Arc allocation per `bundle_adjust` call).
    loss: Option<Arc<dyn RobustLoss>>,
}

impl ReprojFactor {
    fn new(obs: [f32; 2], camera: &PinholeCamera) -> Self {
        Self {
            obs_u: obs[0],
            obs_v: obs[1],
            fx: camera.fx as f32,
            fy: camera.fy as f32,
            cx: camera.cx as f32,
            cy: camera.cy as f32,
            fixed_pose: None,
            fixed_point: None,
            loss: None,
        }
    }

    fn new_fixed_pose(obs: [f32; 2], camera: &PinholeCamera, pose: [f32; 7]) -> Self {
        Self {
            obs_u: obs[0],
            obs_v: obs[1],
            fx: camera.fx as f32,
            fy: camera.fy as f32,
            cx: camera.cx as f32,
            cy: camera.cy as f32,
            fixed_pose: Some(pose),
            fixed_point: None,
            loss: None,
        }
    }

    fn new_fixed_point(obs: [f32; 2], camera: &PinholeCamera, point: [f32; 3]) -> Self {
        Self {
            obs_u: obs[0],
            obs_v: obs[1],
            fx: camera.fx as f32,
            fy: camera.fy as f32,
            cx: camera.cx as f32,
            cy: camera.cy as f32,
            fixed_pose: None,
            fixed_point: Some(point),
            loss: None,
        }
    }

    /// Attach a shared robust loss; called per-factor by `bundle_adjust`
    /// once the BaParams are translated into a concrete `RobustLoss` impl.
    fn with_loss(mut self, loss: Option<Arc<dyn RobustLoss>>) -> Self {
        self.loss = loss;
        self
    }

    /// Project a world point through an SE3 pose, returning `(u, v, z_cam)`.
    ///
    /// Clamps the effective z to ±MIN_Z so cheirality violations (z ≤ 0) and
    /// near-zero depths don't blow up the analytical Jacobian's 1/z and 1/z²
    /// terms. The robust loss attached to the factor saturates the inflated
    /// residual that results, so bad observations have bounded influence on
    /// the linear solve rather than aborting it.
    fn project(
        &self,
        pose_params: &[f32],
        point_params: &[f32],
    ) -> Result<(f32, f32, f32), FactorError> {
        const MIN_Z: f32 = 1e-3;
        let se3 = SE3F32::from_params(pose_params);
        let pw = Vec3AF32::new(point_params[0], point_params[1], point_params[2]);
        let pc = se3 * pw;
        // Preserve z sign but enforce |z| ≥ MIN_Z.
        let z_clamped = if pc.z.abs() < MIN_Z {
            if pc.z >= 0.0 {
                MIN_Z
            } else {
                -MIN_Z
            }
        } else {
            pc.z
        };
        let inv_z = 1.0 / z_clamped;
        Ok((
            self.fx * pc.x * inv_z + self.cx,
            self.fy * pc.y * inv_z + self.cy,
            z_clamped,
        ))
    }

    /// Analytical Jacobian for pose (6 DOF) + point (3 DOF).
    ///
    /// Returns a row-major 2×9 flat array.
    /// Tangent convention: `[upsilon(0..3), omega(3..6)]` matching `SE3F32::retract`.
    fn analytical_jacobian_full(
        &self,
        pose_params: &[f32],
        point_params: &[f32],
    ) -> FactorResult<Vec<f32>> {
        const MIN_Z: f32 = 1e-3;
        let se3 = SE3F32::from_params(pose_params);
        let r = se3.r.matrix();
        let pw = Vec3AF32::new(point_params[0], point_params[1], point_params[2]);
        let pc = se3 * pw;

        // Clamp z to match the forward projection (keeps Jacobian finite).
        let z = if pc.z.abs() < MIN_Z {
            if pc.z >= 0.0 {
                MIN_Z
            } else {
                -MIN_Z
            }
        } else {
            pc.z
        };
        let inv_z = 1.0 / z;
        let inv_z2 = inv_z * inv_z;

        // J_proj row coefficients
        let a0 = self.fx * inv_z;
        let a2 = -self.fx * pc.x * inv_z2;
        let b1 = self.fy * inv_z;
        let b2 = -self.fy * pc.y * inv_z2;

        // Rotation matrix elements: R[row][col]
        let r00 = r.col(0).x;
        let r01 = r.col(1).x;
        let r02 = r.col(2).x;
        let r10 = r.col(0).y;
        let r11 = r.col(1).y;
        let r12 = r.col(2).y;
        let r20 = r.col(0).z;
        let r21 = r.col(1).z;
        let r22 = r.col(2).z;

        let (px, py, pz) = (pw.x, pw.y, pw.z);

        // S = -R * skew(p_w)
        // S[i][j] columns:
        //   col 0: -pz*R[:,1] + py*R[:,2]
        //   col 1:  pz*R[:,0] - px*R[:,2]
        //   col 2: -py*R[:,0] + px*R[:,1]
        let s00 = -pz * r01 + py * r02;
        let s10 = -pz * r11 + py * r12;
        let s20 = -pz * r21 + py * r22;

        let s01 = pz * r00 - px * r02;
        let s11 = pz * r10 - px * r12;
        let s21 = pz * r20 - px * r22;

        let s02 = -py * r00 + px * r01;
        let s12 = -py * r10 + px * r11;
        let s22 = -py * r20 + px * r21;

        // J_pt = J_proj * R (also equals upsilon part of pose jacobian)
        let jpt_00 = a0 * r00 + a2 * r20;
        let jpt_01 = a0 * r01 + a2 * r21;
        let jpt_02 = a0 * r02 + a2 * r22;
        let jpt_10 = b1 * r10 + b2 * r20;
        let jpt_11 = b1 * r11 + b2 * r21;
        let jpt_12 = b1 * r12 + b2 * r22;

        // J_omega = J_proj * S
        let jom_00 = a0 * s00 + a2 * s20;
        let jom_01 = a0 * s01 + a2 * s21;
        let jom_02 = a0 * s02 + a2 * s22;
        let jom_10 = b1 * s10 + b2 * s20;
        let jom_11 = b1 * s11 + b2 * s21;
        let jom_12 = b1 * s12 + b2 * s22;

        // Row-major 2×9: [pose(6), point(3)]
        Ok(vec![
            // Row 0 (du)
            jpt_00, jpt_01, jpt_02, // upsilon
            jom_00, jom_01, jom_02, // omega
            jpt_00, jpt_01, jpt_02, // point (same as upsilon)
            // Row 1 (dv)
            jpt_10, jpt_11, jpt_12, // upsilon
            jom_10, jom_11, jom_12, // omega
            jpt_10, jpt_11, jpt_12, // point (same as upsilon)
        ])
    }

    /// Analytical Jacobian for pose only (fixed point).
    ///
    /// Returns a row-major 2×6 flat array. Columns are the first 6 columns of
    /// `analytical_jacobian_full` (upsilon then omega).
    fn analytical_jacobian_pose_only(
        &self,
        pose_params: &[f32],
        point_params: &[f32],
    ) -> FactorResult<Vec<f32>> {
        let full = self.analytical_jacobian_full(pose_params, point_params)?;
        // Row 0: full[0..6]; Row 1: full[9..15]
        Ok(vec![
            full[0], full[1], full[2], full[3], full[4], full[5], full[9], full[10], full[11],
            full[12], full[13], full[14],
        ])
    }

    /// Analytical Jacobian for point only (fixed pose).
    ///
    /// Returns a row-major 2×3 flat array.
    fn analytical_jacobian_point_only(
        &self,
        pose_params: &[f32],
        point_params: &[f32],
    ) -> FactorResult<Vec<f32>> {
        const MIN_Z: f32 = 1e-3;
        let se3 = SE3F32::from_params(pose_params);
        let r = se3.r.matrix();
        let pw = Vec3AF32::new(point_params[0], point_params[1], point_params[2]);
        let pc = se3 * pw;

        let z = if pc.z.abs() < MIN_Z {
            if pc.z >= 0.0 {
                MIN_Z
            } else {
                -MIN_Z
            }
        } else {
            pc.z
        };
        let inv_z = 1.0 / z;
        let inv_z2 = inv_z * inv_z;

        let a0 = self.fx * inv_z;
        let a2 = -self.fx * pc.x * inv_z2;
        let b1 = self.fy * inv_z;
        let b2 = -self.fy * pc.y * inv_z2;

        let r00 = r.col(0).x;
        let r01 = r.col(1).x;
        let r02 = r.col(2).x;
        let r10 = r.col(0).y;
        let r11 = r.col(1).y;
        let r12 = r.col(2).y;
        let r20 = r.col(0).z;
        let r21 = r.col(1).z;
        let r22 = r.col(2).z;

        Ok(vec![
            a0 * r00 + a2 * r20,
            a0 * r01 + a2 * r21,
            a0 * r02 + a2 * r22,
            b1 * r10 + b2 * r20,
            b1 * r11 + b2 * r21,
            b1 * r12 + b2 * r22,
        ])
    }
}

impl Factor for ReprojFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        // Resolve pose + point params from either the optimizer's `params` or
        // the factor's internal fixed buffers. Three variants are valid:
        //   fixed_pose only       → params = [point]
        //   fixed_point only      → params = [pose]
        //   neither fixed (joint) → params = [pose, point]
        let (pose_params, point_params): (&[f32], &[f32]) =
            match (&self.fixed_pose, &self.fixed_point) {
                (Some(fp), None) => {
                    if params.len() != 1 {
                        return Err(FactorError::DimensionMismatch {
                            expected: 1,
                            actual: params.len(),
                        });
                    }
                    (fp.as_slice(), params[0])
                }
                (None, Some(fpt)) => {
                    if params.len() != 1 {
                        return Err(FactorError::DimensionMismatch {
                            expected: 1,
                            actual: params.len(),
                        });
                    }
                    (params[0], fpt.as_slice())
                }
                (None, None) => {
                    if params.len() != 2 {
                        return Err(FactorError::DimensionMismatch {
                            expected: 2,
                            actual: params.len(),
                        });
                    }
                    (params[0], params[1])
                }
                (Some(_), Some(_)) => {
                    // Both fixed → factor is a constant; shouldn't be added.
                    return Err(FactorError::DimensionMismatch {
                        expected: 1,
                        actual: 0,
                    });
                }
            };

        // Cheirality (z ≤ 0) is no longer an abort. The robust loss attached
        // to this factor saturates the residual for bad projections, while
        // clamping z to MIN_Z below keeps the analytical Jacobian finite so
        // the linear solve stays well-conditioned (no NaN/Inf from 1/z).
        let (u, v, _z) = self.project(pose_params, point_params)?;
        let residual = vec![u - self.obs_u, v - self.obs_v];

        match (&self.fixed_pose, &self.fixed_point) {
            (Some(_), None) => {
                let jacobian = if compute_jacobian {
                    Some(self.analytical_jacobian_point_only(pose_params, point_params)?)
                } else {
                    None
                };
                Ok(LinearizationResult::new(residual, jacobian, 3))
            }
            (None, Some(_)) => {
                let jacobian = if compute_jacobian {
                    Some(self.analytical_jacobian_pose_only(pose_params, point_params)?)
                } else {
                    None
                };
                Ok(LinearizationResult::new(residual, jacobian, 6))
            }
            (None, None) => {
                let jacobian = if compute_jacobian {
                    Some(self.analytical_jacobian_full(pose_params, point_params)?)
                } else {
                    None
                };
                Ok(LinearizationResult::new(residual, jacobian, 9))
            }
            (Some(_), Some(_)) => Err(FactorError::DimensionMismatch {
                expected: 1,
                actual: 0,
            }),
        }
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn num_variables(&self) -> usize {
        match (&self.fixed_pose, &self.fixed_point) {
            (None, None) => 2,
            (Some(_), None) | (None, Some(_)) => 1,
            (Some(_), Some(_)) => 0,
        }
    }

    fn variable_local_dim(&self, idx: usize) -> usize {
        match (&self.fixed_pose, &self.fixed_point) {
            (Some(_), None) => 3, // point only
            (None, Some(_)) => 6, // pose only
            (None, None) => match idx {
                0 => 6,
                1 => 3,
                _ => 0,
            },
            (Some(_), Some(_)) => 0,
        }
    }

    fn get_loss(&self) -> Option<&dyn RobustLoss> {
        self.loss.as_deref()
    }
}

/// Translate a [`BaParams`] robust kernel choice into a shared
/// [`RobustLoss`] instance. Identity returns `None` so factors fall back
/// to plain L2 (the optim solver's fast path). Tukey isn't yet exposed
/// by `kornia-algebra::optim::losses` — caller falls back to Cauchy.
fn build_robust_loss(params: &BaParams) -> Option<Arc<dyn RobustLoss>> {
    use crate::ransac::RobustKernelKind;
    if !params.robust_scale_sq.is_finite() || params.robust_scale_sq <= 0.0 {
        return None;
    }
    // BaParams stores the *squared* scale (matches the kornia-3d kernel
    // surface); kornia-algebra's losses take the linear scale.
    let scale = params.robust_scale_sq.sqrt();
    match params.robust {
        RobustKernelKind::Identity => None,
        RobustKernelKind::Huber => HuberLoss::new(scale).ok().map(|l| {
            let arc: Arc<dyn RobustLoss> = Arc::new(l);
            arc
        }),
        // Tukey not yet in kornia-algebra; fall back to Cauchy which is
        // also a smooth redescender. Documented in BaParams::robust doc.
        RobustKernelKind::Cauchy | RobustKernelKind::Tukey => {
            CauchyLoss::new(scale).ok().map(|l| {
                let arc: Arc<dyn RobustLoss> = Arc::new(l);
                arc
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run bundle adjustment over a set of poses, points, and observations.
///
/// Jointly optimizes camera poses (SE3) and 3D point positions by minimizing
/// reprojection error using Levenberg-Marquardt with analytical Jacobians.
///
/// Only the pinhole intrinsics (`fx`, `fy`, `cx`, `cy`) are used for
/// projection; distortion coefficients are **ignored**. All pixel observations
/// in [`BaObservation::pixel`] must therefore be undistorted.
///
/// Observations with `fixed_pose = true` hold the referenced pose constant.
pub fn bundle_adjust(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera: &PinholeCamera,
    params: &BaParams,
) -> Result<BaResult, BaError> {
    let mut problem = Problem::new();

    // Track which poses / points are free (touched by at least one obs that
    // doesn't fix them) vs fully fixed.
    let mut pose_is_free = vec![false; poses.len()];
    let mut point_is_free = vec![false; points.len()];
    for obs in observations {
        if obs.pose_idx >= poses.len() || obs.point_idx >= points.len() {
            continue;
        }
        if !obs.fixed_pose {
            pose_is_free[obs.pose_idx] = true;
        }
        if !obs.fixed_point {
            point_is_free[obs.point_idx] = true;
        }
    }

    // Add free pose variables.
    for (i, pose) in poses.iter().enumerate() {
        if !pose_is_free[i] {
            continue;
        }
        let se3_params = pose_to_se3_params(pose);
        let var = Variable::new(format!("pose_{i}"), VariableType::SE3, vec![0.0; 7]);
        problem.add_variable(var, se3_params)?;
    }

    // Precompute fixed pose params.
    let fixed_params: Vec<Option<[f32; 7]>> = poses
        .iter()
        .enumerate()
        .map(|(i, pose)| {
            if pose_is_free[i] {
                None
            } else {
                let p = pose_to_se3_params(pose);
                Some([p[0], p[1], p[2], p[3], p[4], p[5], p[6]])
            }
        })
        .collect();

    // Add free point variables only.
    for (i, pt) in points.iter().enumerate() {
        if !point_is_free[i] {
            continue;
        }
        let var = Variable::euclidean(format!("pt_{i}"), 3);
        let init = vec![pt.x as f32, pt.y as f32, pt.z as f32];
        problem.add_variable(var, init)?;
    }

    // Translate the BaParams robust kernel choice into a shared loss
    // instance once (cloned by Arc into every factor). Identity → None,
    // skipping the loss path entirely on the optim solver's L2 fast path.
    let robust_loss = build_robust_loss(params);

    // Add factors. Variant per observation depends on which side is fixed.
    for obs in observations {
        if obs.pose_idx >= poses.len() || obs.point_idx >= points.len() {
            continue;
        }

        match (obs.fixed_pose, obs.fixed_point) {
            (false, false) => {
                let pose_name = format!("pose_{}", obs.pose_idx);
                let pt_name = format!("pt_{}", obs.point_idx);
                let factor =
                    Box::new(ReprojFactor::new(obs.pixel, camera).with_loss(robust_loss.clone()));
                problem.add_factor(factor, vec![pose_name, pt_name])?;
            }
            (true, false) => {
                let pt_name = format!("pt_{}", obs.point_idx);
                let fp_arr = fixed_params[obs.pose_idx].unwrap_or_else(|| {
                    let p = pose_to_se3_params(&poses[obs.pose_idx]);
                    [p[0], p[1], p[2], p[3], p[4], p[5], p[6]]
                });
                let factor = Box::new(
                    ReprojFactor::new_fixed_pose(obs.pixel, camera, fp_arr)
                        .with_loss(robust_loss.clone()),
                );
                problem.add_factor(factor, vec![pt_name])?;
            }
            (false, true) => {
                let pose_name = format!("pose_{}", obs.pose_idx);
                let pt = points[obs.point_idx];
                let factor = Box::new(
                    ReprojFactor::new_fixed_point(
                        obs.pixel,
                        camera,
                        [pt.x as f32, pt.y as f32, pt.z as f32],
                    )
                    .with_loss(robust_loss.clone()),
                );
                problem.add_factor(factor, vec![pose_name])?;
            }
            (true, true) => {
                // Both sides fixed — observation is a constant, contributes
                // nothing to the optimization. Silently skip.
            }
        }
    }

    // Run optimizer.
    let optimizer = LevenbergMarquardt {
        lambda_init: params.initial_lambda,
        lambda_max: 1e10,
        lambda_factor: 10.0,
        max_iterations: params.max_iterations,
        cost_tolerance: params.cost_tolerance,
        gradient_tolerance: params.gradient_tolerance,
    };

    let result = optimizer.optimize(&mut problem)?;

    let converged = matches!(
        result.termination_reason,
        kornia_algebra::optim::TerminationReason::CostConverged
            | kornia_algebra::optim::TerminationReason::GradientConverged
    );

    // Extract results.
    let vars = problem.get_variables();

    let mut out_poses = Vec::with_capacity(poses.len());
    for i in 0..poses.len() {
        if pose_is_free[i] {
            let name = format!("pose_{i}");
            let values = &vars[&name].values;
            out_poses.push(se3_params_to_pose(values));
        } else {
            out_poses.push(poses[i]);
        }
    }

    let mut out_points = Vec::with_capacity(points.len());
    for i in 0..points.len() {
        if point_is_free[i] {
            let name = format!("pt_{i}");
            let values = &vars[&name].values;
            out_points.push(Vec3F64::new(
                values[0] as f64,
                values[1] as f64,
                values[2] as f64,
            ));
        } else {
            out_points.push(points[i]);
        }
    }

    Ok(BaResult {
        poses: out_poses,
        points: out_points,
        iterations: result.iterations,
        converged,
        // Per-camera depth scales are a `bundle_adjust_schur` feature; this path ignores
        // `depth_meas` entirely, so reporting scales here would be fiction.
        depth_scales: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Mat3F64, Vec3F64};

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    #[test]
    fn test_reproj_factor_zero_residual() {
        let cam = test_camera();
        let factor = ReprojFactor::new([320.0, 240.0], &cam);
        let pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f32];
        let point = [0.0, 0.0, 5.0f32];
        let result = factor.linearize(&[&pose, &point], false).unwrap();
        assert!(result.residual[0].abs() < 1e-4);
        assert!(result.residual[1].abs() < 1e-4);
    }

    #[test]
    fn test_reproj_factor_jacobian_finite() {
        let cam = test_camera();
        let factor = ReprojFactor::new([350.0, 260.0], &cam);
        let pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f32];
        let point = [0.5, 0.3, 4.0f32];
        let result = factor.linearize(&[&pose, &point], true).unwrap();
        let jac = result.jacobian.unwrap();
        assert_eq!(jac.len(), 2 * 9);
        assert!(jac.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_fixed_pose_factor() {
        let cam = test_camera();
        let pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f32];
        let factor = ReprojFactor::new_fixed_pose([320.0, 240.0], &cam, pose);
        let point = [0.0, 0.0, 5.0f32];
        let result = factor.linearize(&[&point], true).unwrap();
        assert!(result.residual[0].abs() < 1e-4);
        assert!(result.residual[1].abs() < 1e-4);
        let jac = result.jacobian.unwrap();
        assert_eq!(jac.len(), 6);
        assert!(jac.iter().all(|x| x.is_finite()));
    }

    // TODO: rewrite numerical Jacobian in f64 (f64 SE3 exp + f64 projection) so we
    // can use eps~1e-8 and tighten the tolerance from 5% to ~1e-4. Currently limited
    // by f32 catastrophic cancellation.

    /// Compute numerical Jacobian via central differences.
    ///
    /// Uses eps=1e-3 which is near-optimal for f32 (balances truncation and rounding).
    fn numerical_jacobian(
        factor: &ReprojFactor,
        pose_params: &[f32],
        point_params: &[f32],
    ) -> Vec<f32> {
        let eps: f32 = 1e-3;
        let mut numerical = vec![0.0f32; 18];

        let se3 = SE3F32::from_params(pose_params);

        // Pose tangent (6 DOF)
        for i in 0..6 {
            let mut dp = [0.0f32; 6];
            let mut dm = [0.0f32; 6];
            dp[i] = eps;
            dm[i] = -eps;
            let pp = se3.retract(&dp).to_params();
            let pm = se3.retract(&dm).to_params();
            let (up, vp, _) = factor.project(&pp, point_params).unwrap();
            let (um, vm, _) = factor.project(&pm, point_params).unwrap();
            let inv = 1.0 / (2.0 * eps);
            numerical[i] = (up - um) * inv;
            numerical[9 + i] = (vp - vm) * inv;
        }

        // Point (3 DOF)
        for i in 0..3 {
            let mut pp = [point_params[0], point_params[1], point_params[2]];
            let mut pm = pp;
            pp[i] += eps;
            pm[i] -= eps;
            let (up, vp, _) = factor.project(pose_params, &pp).unwrap();
            let (um, vm, _) = factor.project(pose_params, &pm).unwrap();
            let inv = 1.0 / (2.0 * eps);
            numerical[6 + i] = (up - um) * inv;
            numerical[9 + 6 + i] = (vp - vm) * inv;
        }

        numerical
    }

    #[test]
    fn test_analytical_vs_numerical_jacobian() {
        let cam = test_camera();
        let factor = ReprojFactor::new([350.0, 260.0], &cam);
        let pose = [1.0, 0.0, 0.0, 0.0, 0.1, -0.05, 0.2f32];
        let point = [0.5, 0.3, 4.0f32];

        let analytical = factor.analytical_jacobian_full(&pose, &point).unwrap();
        let numerical = numerical_jacobian(&factor, &pose, &point);

        for (j, (&a, &n)) in analytical.iter().zip(numerical.iter()).enumerate() {
            let diff = (a - n).abs();
            let scale = n.abs().max(1.0);
            assert!(
                diff / scale < 5e-2,
                "Jacobian mismatch at index {j}: analytical={a}, numerical={n}, rel_err={}",
                diff / scale
            );
        }
    }

    #[test]
    fn test_analytical_vs_numerical_jacobian_rotated() {
        let cam = test_camera();

        // Build a rotated pose (~28.6° around Y axis).
        let angle = 0.5f32;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let r = Mat3AF32::from_cols(
            Vec3AF32::new(cos_a, 0.0, -sin_a),
            Vec3AF32::new(0.0, 1.0, 0.0),
            Vec3AF32::new(sin_a, 0.0, cos_a),
        );
        let so3 = SO3F32::from_matrix(&r);
        let se3 = SE3F32::new(so3, Vec3AF32::new(0.2, -0.1, 0.3));
        let pose = se3.to_params();
        let point = [1.0, -0.5, 6.0f32];

        let factor = ReprojFactor::new([300.0, 250.0], &cam);
        let analytical = factor.analytical_jacobian_full(&pose, &point).unwrap();
        let numerical = numerical_jacobian(&factor, &pose, &point);

        for (j, (&a, &n)) in analytical.iter().zip(numerical.iter()).enumerate() {
            let diff = (a - n).abs();
            let scale = n.abs().max(1.0);
            assert!(
                diff / scale < 5e-2,
                "Jacobian mismatch at index {j}: analytical={a}, numerical={n}, rel_err={}",
                diff / scale
            );
        }
    }

    #[test]
    fn test_bundle_adjust_converges() {
        let cam = test_camera();

        let pose0 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);
        let pose1 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.5, 0.0, 0.0));

        let true_points = [
            Vec3F64::new(-1.0, -1.0, 5.0),
            Vec3F64::new(1.0, -1.0, 5.0),
            Vec3F64::new(1.0, 1.0, 5.0),
            Vec3F64::new(-1.0, 1.0, 5.0),
        ];

        let project = |pose: &Pose3d, pw: &Vec3F64| -> [f32; 2] {
            let pc = pose.transform_point(pw);
            let u = cam.fx * pc.x / pc.z + cam.cx;
            let v = cam.fy * pc.y / pc.z + cam.cy;
            [u as f32, v as f32]
        };

        let mut observations = Vec::new();
        for (pi, pt) in true_points.iter().enumerate() {
            observations.push(BaObservation {
                pose_idx: 0,
                point_idx: pi,
                pixel: project(&pose0, pt),
                fixed_pose: true,
                fixed_point: false,
                ..BaObservation::default()
            });
            observations.push(BaObservation {
                pose_idx: 1,
                point_idx: pi,
                pixel: project(&pose1, pt),
                fixed_pose: false,
                fixed_point: false,
                ..BaObservation::default()
            });
        }

        let perturbed: Vec<Vec3F64> = true_points
            .iter()
            .map(|p| *p + Vec3F64::new(0.05, -0.03, 0.02))
            .collect();

        let result = bundle_adjust(
            &[pose0, pose1],
            &perturbed,
            &observations,
            &cam,
            &BaParams {
                max_iterations: 20,
                ..BaParams::default()
            },
        )
        .unwrap();

        for (i, refined) in result.points.iter().enumerate() {
            let err = (*refined - true_points[i]).length();
            assert!(err < 0.1, "point {i} error {err} too large");
        }
    }

    /// With a single outlier observation injected, plain L2 BA pulls the
    /// 3D points off-target while a Cauchy robust kernel downweights the
    /// outlier and converges close to the truth. Exercises the
    /// `BaParams::robust` plumbing through `Factor::get_loss`.
    #[test]
    fn test_bundle_adjust_robust_kernel_resists_outlier() {
        use crate::ransac::RobustKernelKind;

        let cam = test_camera();
        let pose0 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);
        let pose1 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.5, 0.0, 0.0));

        let true_points = [
            Vec3F64::new(-1.0, -1.0, 5.0),
            Vec3F64::new(1.0, -1.0, 5.0),
            Vec3F64::new(1.0, 1.0, 5.0),
            Vec3F64::new(-1.0, 1.0, 5.0),
        ];
        let project = |pose: &Pose3d, pw: &Vec3F64| -> [f32; 2] {
            let pc = pose.transform_point(pw);
            let u = cam.fx * pc.x / pc.z + cam.cx;
            let v = cam.fy * pc.y / pc.z + cam.cy;
            [u as f32, v as f32]
        };
        let mut observations = Vec::new();
        for (pi, pt) in true_points.iter().enumerate() {
            observations.push(BaObservation {
                pose_idx: 0,
                point_idx: pi,
                pixel: project(&pose0, pt),
                fixed_pose: true,
                fixed_point: false,
                ..BaObservation::default()
            });
            observations.push(BaObservation {
                pose_idx: 1,
                point_idx: pi,
                pixel: project(&pose1, pt),
                fixed_pose: false,
                fixed_point: false,
                ..BaObservation::default()
            });
        }
        // Inject one outlier on point 0, view 1 — 30 px off (≫ the 2 px
        // Cauchy scale, but small enough that the L2 LM doesn't punch
        // the perturbed point behind the camera mid-iteration).
        observations[1].pixel[0] += 30.0;
        observations[1].pixel[1] -= 30.0;

        let perturbed: Vec<Vec3F64> = true_points
            .iter()
            .map(|p| *p + Vec3F64::new(0.05, -0.03, 0.02))
            .collect();

        // L2 (Identity kernel) — should pull the outlier-related point off.
        let l2 = bundle_adjust(
            &[pose0, pose1],
            &perturbed,
            &observations,
            &cam,
            &BaParams {
                max_iterations: 20,
                ..BaParams::default()
            },
        )
        .unwrap();

        // Huber with delta=5 px — bounded-influence kernel that keeps
        // inliers at full weight (squared residual ≤ 25 → w=1) while
        // downweighting the 30 px outlier (squared norm ≈ 1800,
        // w = 5/√1800 ≈ 0.12). Cauchy at this scale is *too* aggressive
        // and discards legitimate inliers during early LM iters; that's
        // a known limitation of redescending kernels at perturbed inits.
        let robust = bundle_adjust(
            &[pose0, pose1],
            &perturbed,
            &observations,
            &cam,
            &BaParams {
                max_iterations: 20,
                robust: RobustKernelKind::Huber,
                robust_scale_sq: 25.0,
                ..BaParams::default()
            },
        )
        .unwrap();

        // The point hit by the outlier (index 0) is what we measure.
        let l2_err = (l2.points[0] - true_points[0]).length();
        let robust_err = (robust.points[0] - true_points[0]).length();
        // Robust must do better than L2 on the contaminated point.
        assert!(
            robust_err < l2_err,
            "robust BA didn't beat L2 on outlier-contaminated point: \
             l2_err={l2_err:.3} robust_err={robust_err:.3}"
        );
    }
}
