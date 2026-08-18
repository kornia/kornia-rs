//! Schur-complement bundle adjustment with dense reduced camera system.
//!
//! The standard bipartite-Schur trick from Triggs et al. (1999). Each LM
//! iteration builds the Hessian in BLOCK form
//!
//! ```text
//!     H = [ A   B  ]    A = 6P × 6P pose blocks (block-diagonal),
//!         [ Bᵀ  C  ]    C = 3N × 3N point blocks (BLOCK-DIAGONAL),
//!                       B = 6P × 3N pose-point cross terms (sparse).
//! ```
//!
//! Block-diagonal C means C⁻¹ is cheap (per-3×3 invert). The **reduced
//! camera system**
//!
//! ```text
//!     M = A − B C⁻¹ Bᵀ           (dense 6P × 6P)
//!     m = g_pose − B C⁻¹ g_point
//! ```
//!
//! is solved with `faer`'s dense Cholesky on the small matrix; points are
//! recovered by back-substitution. For our SLAM problem (~170 poses ×
//! ~3000 points × ~15000 observations) the reduced system is just
//! 1020 × 1020 — Ceres's `DENSE_SCHUR` is exactly this regime.
//!
//! No sparse-matrix dependency is needed because the only "large" object
//! the Schur trick has to manipulate (B, 6P × 3N) is never materialised:
//! we walk observations and accumulate per-point contributions into M
//! directly.
//!
//! Jacobian conventions match [`crate::ba::ReprojFactor`]:
//!
//!   * Pose tangent layout `[ρ; ω]` (upsilon then omega), 6-dim.
//!   * Point parameters are the 3-dim world coordinates.
//!   * z is clamped to `MIN_Z` to handle mid-iteration cheirality flips.
//!
//! Supports fixed-pose anchors, fixed-point gauge (motion-only BA), optional
//! per-observation depth residuals, optional per-pose translation and
//! orientation priors, optional constant-velocity motion priors over pose
//! triplets (the one residual family that makes the reduced camera system
//! non-block-diagonal — see [`bundle_adjust_schur_with_all_priors`]), and
//! the robust kernels in `BaParams::robust` via IRLS (see
//! [`bundle_adjust_schur`]). LM damping is ellipsoidal — `λ·diag(JᵀJ)`, as
//! Ceres does it — so `λ` is dimensionless here, unlike
//! [`crate::ba::bundle_adjust`], which damps by `λ·I`. Full
//! LM-with-backtracking (a gain-ratio / trust-region update) is still TODO:
//! the step test is the sign of the cost change only.
//!
//! # Instrumentation
//!
//! [`BA_CALLS`], [`BA_ITERS`] and [`BA_NANOS`] accumulate across every adjustment in the process,
//! and [`BA_LIN_NANOS`] / [`BA_ASM_NANOS`] / [`BA_FACT_NANOS`] / [`BA_TRIAL_NANOS`] split each LM
//! iteration into linearise, assemble, factor and trial. [`BA_OBS`] holds the residual-row count
//! of the most recent iteration. They are process-global with no reset, so they answer "where did
//! this process spend its solve time", not "what did this call cost".

use faer::prelude::Solve;
use faer::Mat;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec3AF32, Vec3F64, SE3F32, SO3F32};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use thiserror::Error;

use crate::ba::{BaError, BaMotionPrior, BaObservation, BaParams, BaPosePrior, BaResult};
use crate::camera::PinholeCamera;
use crate::pose::Pose3d;
use crate::ransac::RobustKernelKind;

const MIN_Z: f32 = 1e-3;

/// Total adjustments run, and total LM iterations across them. `BaResult` has always carried
/// `iterations` and `converged`, but a caller that runs hundreds of adjustments has no way to see
/// the aggregate, so "where does the solve time go" gets answered by argument instead of by
/// measurement.
pub static BA_CALLS: AtomicUsize = AtomicUsize::new(0);
/// See [`BA_CALLS`].
pub static BA_ITERS: AtomicUsize = AtomicUsize::new(0);
/// Total nanoseconds inside bundle adjustment. See [`BA_CALLS`].
pub static BA_NANOS: AtomicU64 = AtomicU64::new(0);

/// Per-phase nanoseconds within the LM iteration. Linearise = residuals, Jacobians and the
/// A/B/C/g accumulation; assemble = damping plus building the reduced camera system; factor =
/// Cholesky and back-substitution; trial = evaluating the objective at the trial point.
///
/// Nanoseconds, not microseconds: these are accumulated once per LM iteration, and `as_micros()`
/// truncates DOWNWARD every time. At this crate's own test sizes the factor phase runs ~0.4 µs, so
/// a microsecond counter reads 0 forever; the bias is worst for the shortest phase, which is the
/// one any "phase X dominates" conclusion rests on.
///
/// These are process-global and never reset, so they are an aggregate over the lifetime of the
/// process, not of a solve; under concurrent solves the four phases interleave across problems.
pub static BA_LIN_NANOS: AtomicU64 = AtomicU64::new(0);
/// See [`BA_LIN_NANOS`].
pub static BA_ASM_NANOS: AtomicU64 = AtomicU64::new(0);
/// See [`BA_LIN_NANOS`].
pub static BA_FACT_NANOS: AtomicU64 = AtomicU64::new(0);
/// See [`BA_LIN_NANOS`].
pub static BA_TRIAL_NANOS: AtomicU64 = AtomicU64::new(0);
/// Residual rows the most recent LM iteration evaluated: every in-range reprojection observation
/// (including those on fixed poses and fixed points, which are still scored) plus every depth row.
/// Pose-prior rows are not counted — they scale with the camera count, not the observation count.
///
/// Written in the linearisation pass, so it is set on every iteration that runs at all, including
/// one whose factorisation later fails.
///
/// It exists because this number is easy to get wrong from the outside and the error is silent: a
/// per-observation timing comparison built on the caller's total observation count came out 7x
/// high, and one built on a count derived from cost and RMS came out 3.5x low.
pub static BA_OBS: AtomicU64 = AtomicU64::new(0);

/// Fold one adjustment's iteration count and wall time into [`BA_ITERS`] / [`BA_NANOS`].
///
/// Must be called on EVERY exit taken after [`BA_CALLS`] was incremented — including the
/// Cholesky-failure error path — or the aggregate counts calls it never accounts for.
#[inline]
fn record_call_totals(t_ba: &std::time::Instant, iters_done: usize) {
    BA_ITERS.fetch_add(iters_done, Ordering::Relaxed);
    BA_NANOS.fetch_add(t_ba.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Ceres' `min_lm_diagonal` / `max_lm_diagonal`. The LM damping diagonal is clamped to this range
/// because extremely small or large entries of diag(JᵀJ) make the regularisation fail.
///
/// The ceiling is NOT Ceres' `1e32`: Ceres is `double`, this solver is `f32`. The damping actually
/// added is `λ · clamp(diag)` and `λ` is allowed to reach `1e10` before the loop gives up, so a
/// `1e32` ceiling overflows to `+inf` (`1e32_f32 * 1e7_f32` is already `inf`) — and an infinite
/// diagonal sails through `invert_3x3`'s `det.abs() < 1e-20` guard to yield `Some([NaN; 9])`.
/// `1e24` keeps the worst-case product at `1e34`, inside f32's `3.4e38`.
const MIN_LM_DIAGONAL: f32 = 1e-6;
const MAX_LM_DIAGONAL: f32 = 1e24;

/// Ceiling on `BaParams::initial_lambda`, so that `λ · MAX_LM_DIAGONAL` cannot overflow `f32`
/// before the loop's own `λ > 1e10` guards get a chance to fire. `1e10 · 1e24 = 1e34`, inside
/// `f32::MAX ≈ 3.4e38`.
const MAX_INITIAL_LAMBDA: f32 = 1e10;

/// IRLS weight `w = ρ'(s)` for `s = ‖r‖²`. The residual and its Jacobian rows are scaled by √w,
/// which is what makes the normal equations those of the robust problem.
///
/// `Tukey` deliberately falls back to `Cauchy`, matching `BaParams::robust`'s documented fallback
/// and `ba::build_robust_loss`. NOTE this disagrees with `impl RobustKernel for RobustKernelKind`
/// in [`crate::ransac`], which dispatches `Tukey` to the real hard redescender — the same enum
/// value means two different kernels depending on which module consumes it.
#[inline]
fn robust_weight(kind: RobustKernelKind, scale: f32, r_sq: f32) -> f32 {
    match kind {
        RobustKernelKind::Identity => 1.0,
        RobustKernelKind::Huber => {
            // Knee tested as `s <= scale²`, NOT `sqrt(s) <= scale`, to match
            // `kornia_algebra::optim::losses::HuberLoss::weight`. In f32 `sqrt` and squaring
            // round differently, so the two predicates disagree for `s` within an ulp of
            // `scale²` — the same observation weighted 1.0 by one solver and `scale/‖r‖` by the
            // other, from identical `BaParams`.
            if r_sq <= scale * scale {
                1.0
            } else {
                scale / r_sq.sqrt()
            }
        }
        RobustKernelKind::Cauchy | RobustKernelKind::Tukey => {
            let s2 = scale * scale;
            s2 / (s2 + r_sq)
        }
    }
}

/// The robust cost `½ρ(s)` that [`robust_weight`] is the derivative of: `d/ds[½ρ(s)] = ½·w(s)`.
/// `robust_weight_is_the_derivative_of_robust_cost` pins that identity.
///
/// This must be what the LM accept test compares. Accumulating the √w-scaled residual instead
/// gives `½ρ'(s)·s`, an IRLS surrogate that is NOT the objective: for Huber past the knee it is
/// `½k‖r‖` where the loss is `k‖r‖ − k²/2`, so it moves at HALF the true rate on every
/// downweighted observation, and a step's measured reduction comes out halved.
#[inline]
fn robust_cost(kind: RobustKernelKind, scale: f32, r_sq: f32) -> f32 {
    match kind {
        RobustKernelKind::Identity => 0.5 * r_sq,
        RobustKernelKind::Huber => {
            // Same knee predicate as `robust_weight` — they are a derivative pair and must
            // switch branches on the identical condition.
            if r_sq <= scale * scale {
                0.5 * r_sq
            } else {
                scale * r_sq.sqrt() - 0.5 * scale * scale
            }
        }
        // Tukey shares Cauchy's weight here, so it shares Cauchy's loss too.
        RobustKernelKind::Cauchy | RobustKernelKind::Tukey => {
            let s2 = scale * scale;
            // `ln_1p`, NOT `(1.0 + x).ln()`. In f32 the latter quantises `x` to multiples of
            // ~1.19e-7 before taking the log, which is exactly the converged regime this cost
            // has to resolve: at `s2 = 5.99, r_sq = 1e-6` it is 28.6% low, and for
            // `r_sq/s2 < 6e-8` it returns identically 0.0 — so `new_cost < cost` becomes a
            // comparison of two quantisation staircases.
            0.5 * s2 * (r_sq / s2).ln_1p()
        }
    }
}

/// `|ln s|` clamp for the per-camera depth scale — a camera may disagree with its monocular prior
/// by at most 4×. Beyond that the fit is not a scale, it is a broken pose, and letting it run
/// lets one bad camera silently switch its own depth prior off.
const MAX_ABS_LOG_DEPTH_SCALE: f32 = 1.386_294_4; // ln 4

/// Camera-frame depth `z` predicted for a point, with the mid-iteration cheirality clamp.
#[inline]
fn clamped_z(pose: &SE3F32, point: &Vec3F64) -> f32 {
    let pw = Vec3AF32::new(point.x as f32, point.y as f32, point.z as f32);
    let z = (*pose * pw).z;
    if z.abs() < MIN_Z {
        if z >= 0.0 {
            MIN_Z
        } else {
            -MIN_Z
        }
    } else {
        z
    }
}

/// Depth residual and its derivative `∂r/∂z`, for both residual forms.
///
/// Legacy (`log_mode == false`): `r = (z − m)/σ`, `∂r/∂z = 1/σ`. `σ` is in metres and `s` is
/// ignored.
///
/// Log (`log_mode == true`), following VidMap eq. 3 with `sm = s·m`:
/// ```text
///   z > 0:  r = ln(z / sm) / σ            ∂r/∂z = 1 / (z·σ)
///   z ≤ 0:  r = (z − sm) / (sm·σ)         ∂r/∂z = 1 / (sm·σ)
/// ```
/// `σ` is RELATIVE here (a fraction of depth), which is what makes the residual comparable across
/// near and far points.
///
/// The `z ≤ 0` branch is the log residual's own first-order expansion about `z = sm`, not the
/// paper's bare `z − sm`, so behind the camera it still pushes `z` back toward `sm` linearly with
/// the right slope THERE.
///
/// It is NOT continuous with the log branch at the changeover, and the expansion point is not the
/// changeover: the branches switch at `z = 0` while the expansion is about `z = sm`. At `sm = 1`,
/// `σ = 0.1` the log branch floored by `MIN_Z` gives `r² ≈ 4772` where the linear branch at
/// `z = −1e-3` gives `r² ≈ 100`, so the depth term is ~47x CHEAPER just behind the image plane
/// than just in front of it. A point being squeezed toward the plane therefore has a downhill
/// direction through it, which is the opposite of what a cheirality clamp is for. `MIN_Z` keeps
/// the Jacobian finite so this is bad conditioning rather than a NaN, and cheirality is caught
/// elsewhere — but it is a real defect in this residual and should be fixed by expanding about
/// `z = 0`, or by making the behind-camera branch strictly more expensive than the log branch at
/// `MIN_Z`. Not fixed here: this path has no in-tree caller yet, and changing it without one to
/// measure against is how the last three guesses in this file went wrong.
#[inline]
fn depth_residual(z: f32, m: f32, s: f32, sigma: f32, log_mode: bool) -> (f32, f32) {
    let inv_sigma = 1.0 / sigma.max(1e-6);
    if !log_mode {
        return ((z - m) * inv_sigma, inv_sigma);
    }
    let sm = (s * m).max(MIN_Z);
    if z > 0.0 {
        ((z / sm).ln() * inv_sigma, inv_sigma / z)
    } else {
        let inv_sm = 1.0 / sm;
        ((z - sm) * inv_sm * inv_sigma, inv_sm * inv_sigma)
    }
}

/// Regularised closed-form update of the per-camera depth scales, holding poses and points fixed.
///
/// The log residual is LINEAR in `ln s_i`, so this block of the objective
/// ```text
///   E_i(ln s) = Σ_k w_ik ((a_ik − ln s)/σ_ik)² + λ·(Σ_k w_ik/σ_ik²)·(ln s − ln s_seed_i)²
/// ```
/// with `a_ik = ln z_ik − ln m_ik`, is a one-variable weighted least squares whose exact minimiser
/// is a shrunk weighted mean:
/// ```text
///   ln s_i = ln s_seed_i + (Σ_k w_ik(a_ik − ln s_seed_i)/σ_ik²) / ((1+λ)·Σ_k w_ik/σ_ik²)
/// ```
/// Alternating this exact block update with the LM step on poses/points is block-coordinate
/// descent on the joint objective: it reaches the same stationary point as folding `s_i` into the
/// normal equations as a 7th camera DOF, without widening every 6×6 block in the Schur reduction.
///
/// The prior shrinks toward `s = 1` — an ABSOLUTE claim that the metric network's own scale is
/// right — and NOT toward the seed the caller passed in.
///
/// That distinction is the whole load-bearing part of this term, and getting it wrong is silent.
/// Callers typically re-fit the seed against the CURRENT geometry before every solve, so shrinking
/// toward the seed means "stay near wherever the geometry already is": the prior then tracks drift
/// instead of resisting it, and the scales float free exactly as if λ were 0. Measured on a real
/// 365-keyframe walk, seed-anchored λ=1 doubled the map (54.8 m → 108 m of trajectory, vertical
/// extent 5.0 m → 13.4 m) while every scale-INVARIANT metric improved — the signature of a
/// reconstruction whose shape is fine and whose gauge has come loose. An anchor recomputed from
/// the thing it anchors is not an anchor.
///
/// The seed still sets the starting value, which is worth having: it puts the log residual near
/// its optimum on iteration one instead of making the solver discover the scale.
///
/// Cameras with no forward-facing depth observations keep their current scale rather than
/// collapsing to 1.
fn update_depth_scales(
    scales: &mut [f32],
    se3s: &[SE3F32],
    xyz: &[Vec3F64],
    observations: &[BaObservation],
    prior_weight: f32,
    robust_w: &dyn Fn(f32) -> f32,
) {
    let n = scales.len();
    let mut num = vec![0.0_f64; n];
    let mut den = vec![0.0_f64; n];

    for obs in observations {
        let Some(m) = obs.depth_meas else { continue };
        if obs.pose_idx >= n || obs.point_idx >= xyz.len() || m <= 0.0 {
            continue;
        }
        let z = clamped_z(&se3s[obs.pose_idx], &xyz[obs.point_idx]);
        // Only the log branch is linear in `ln s`; behind-camera observations are excluded from
        // the closed form (they are still penalised by the LM step through the linear branch).
        if z <= 0.0 {
            continue;
        }
        let sigma = obs.depth_sigma.max(1e-6);
        let s = scales[obs.pose_idx];
        let (r, _) = depth_residual(z, m, s, sigma, true);
        let w = f64::from(robust_w(r * r));
        let inv_var = f64::from(1.0 / (sigma * sigma));
        num[obs.pose_idx] += w * inv_var * f64::from((z / m).ln());
        den[obs.pose_idx] += w * inv_var;
    }

    let shrink = 1.0 + f64::from(prior_weight.max(0.0));
    for i in 0..n {
        if den[i] <= 0.0 {
            continue;
        }
        // Shrink toward ln s = 0, i.e. s = 1. See the note above on why this must not be the seed.
        let log_s = (num[i] / den[i] / shrink) as f32;
        scales[i] = log_s
            .clamp(-MAX_ABS_LOG_DEPTH_SCALE, MAX_ABS_LOG_DEPTH_SCALE)
            .exp();
    }
}

/// Errors specific to the Schur BA driver. Wraps existing [`BaError`].
#[derive(Debug, Error)]
pub enum SchurBaError {
    /// Linear system is rank-deficient / Cholesky failed.
    #[error("Reduced camera Cholesky failed (likely rank-deficient): {0}")]
    CholeskyFailed(String),
    /// No free variables after applying anchors.
    #[error("All variables are fixed — nothing to optimise")]
    NoFreeVariables,
    /// Other BA setup error.
    #[error(transparent)]
    Ba(#[from] BaError),
}
/// The orientation-prior residual for one pose: how far the camera's IMAGE-UP axis has drifted
/// from the direction the caller claims it points, whitened by `up_sigma`.
///
/// Shared by the linearisation and the trial-cost evaluation deliberately. Those two must score
/// the SAME objective — if they drift apart, LM rejects exactly the steps this prior exists to
/// take — and the motion prior already gets that guarantee from `motion_prior_residual`. Two
/// copies of ten lines is how the up prior would lose it.
///
/// Image-up is FIXED at `(0, −1, 0)` (OpenCV convention: +Y down). A caller with a measured
/// direction rather than the upright-camera assumption expresses it by rotating `up_world`, which
/// constrains the same one degree of freedom without a second per-camera field to keep in sync.
/// `r_row1` is row 1 of the pose's rotation matrix — i.e. `[R[1][0], R[1][1], R[1][2]]`, which for
/// column-major storage is `(col0.y, col1.y, col2.y)`. `u_pred` is returned alongside the residual
/// because the Jacobian `[u_pred]×` needs it.
#[inline]
fn up_prior_residual(r_row1: [f32; 3], up_world: [f32; 3], up_sigma: f32) -> ([f32; 3], [f32; 3]) {
    let inv_su = 1.0_f32 / up_sigma.max(1e-6);
    // u_pred = Rᵀ · (0,−1,0) = minus row 1 of R, i.e. the world direction the image's up axis
    // currently points. Taking the row directly is why this needs three scalars, not the matrix.
    let u_pred = [-r_row1[0], -r_row1[1], -r_row1[2]];
    let r_up = [
        (u_pred[0] - up_world[0]) * inv_su,
        (u_pred[1] - up_world[1]) * inv_su,
        (u_pred[2] - up_world[2]) * inv_su,
    ];
    (r_up, u_pred)
}

/// 6-vector residual of one [`BaMotionPrior`] on the CURRENT pose estimates, already whitened by
/// the prior's two sigmas.
///
/// Layout: `[t_ratio, 0, 0 | w01 − α·w02]` in the general case, or
/// `[α(C2−C0) − (C1−C0) | w01 − α·w02]` in the degenerate `C0 ≈ C2` case. `C` are camera CENTRES
/// (`−Rᵀt`) — the physically meaningful quantity, not the camera-frame translations — and `w` are
/// SO(3) logs of the relative rotations.
///
/// The translation term is a norm RATIO precisely because the map's scale is a free parameter of
/// the problem: a residual on the position difference would be minimised by shrinking the entire
/// reconstruction, so it would fight the depth anchor instead of complementing it. The ratio is
/// invariant to global scale and constrains only the shape of the local motion.
///
/// The `C0 ≈ C2` fallback exists because with the endpoints coincident the ratio's denominator
/// vanishes and the residual is undefined. Its guard is ABSOLUTE (`n02 > 1e-6`) while the map's unit
/// is the bootstrap baseline, so on sequential capture — where that baseline is the SMALLEST in the
/// sequence — an operator pause gives `n02` of order 1e-3..1e-2 map units, a thousand times the
/// guard. The fallback therefore does NOT fire on the pause case; the ratio branch runs on a small
/// denominator instead.
///
/// That is survivable rather than correct. The stiffness grows as `1/n02`, but the DISPLACEMENT the
/// residual demands is `O(n02)` — sub-millimetre — so a stalled triplet reshapes local jitter and
/// cannot pull a keyframe off the trajectory. Making the guard relative to the local baseline
/// (`n02 > 1e-3 * (n01 + n02)`) would be the honest fix; it is deliberately not done here because it
/// changes the objective for every existing caller and wants its own measurement.
fn motion_prior_residual(p0: &SE3F32, p1: &SE3F32, p2: &SE3F32, mp: &BaMotionPrior) -> [f32; 6] {
    // Camera centre C = -Rᵀt, and the SO(3) log of Ra·Rbᵀ. Both come from `kornia-algebra`
    // rather than being spelled out here: `SE3F32::inverse().t` IS -Rᵀt, and `SO3F32::log()`
    // already carries the small-angle branch that a hand-rolled Rodrigues has to get right to
    // avoid dividing by a vanishing sine.
    let centre = |p: &SE3F32| p.inverse().t;
    let rel_log = |a: &SE3F32, b: &SE3F32| (a.r * b.r.inverse()).log();

    let (c0, c1, c2) = (centre(p0), centre(p1), centre(p2));
    let d01 = c1 - c0;
    let d02 = c2 - c0;
    let n01 = d01.length();
    let n02 = d02.length();
    let inv_sp = 1.0 / mp.position_sigma.max(1e-6);
    let inv_so = 1.0 / mp.orientation_sigma.max(1e-6);

    let mut r = [0.0f32; 6];
    if n02 > 1e-6 {
        r[0] = (mp.alpha - n01 / n02) * inv_sp;
    } else {
        // Stationary endpoints: fall back to the position difference (no scale in play).
        let d = d02 * mp.alpha - d01;
        r[0] = d.x * inv_sp;
        r[1] = d.y * inv_sp;
        r[2] = d.z * inv_sp;
    }
    let w = rel_log(p1, p0) - rel_log(p2, p0) * mp.alpha;
    r[3] = w.x * inv_so;
    r[4] = w.y * inv_so;
    r[5] = w.z * inv_so;
    r
}

// ── f32 ↔ f64 conversion helpers (shared shape with ba.rs) ───────────────

fn pose_to_se3(pose: &Pose3d) -> SE3F32 {
    let r = Mat3AF32::from_cols(
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
    let so3 = SO3F32::from_matrix(&r);
    SE3F32::new(
        so3,
        Vec3AF32::new(
            pose.translation.x as f32,
            pose.translation.y as f32,
            pose.translation.z as f32,
        ),
    )
}

fn se3_to_pose(se3: &SE3F32) -> Pose3d {
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

// ── Per-observation residual + analytical Jacobian (matches ReprojFactor) ──

/// One-coefficient radial distortion, evaluated at normalised image coordinates.
///
/// With `d = 1 + k1·r²` and `r² = xn² + yn²` the distorted point is `(xn·d, yn·d)`, and the
/// local 2×2 `∂(xd, yd)/∂(xn, yn)` is symmetric:
///
/// ```text
///   [ d + 2k1·xn²    2k1·xn·yn   ]
///   [ 2k1·xn·yn      d + 2k1·yn² ]
/// ```
///
/// **At `k1 = 0` every field is a hard identity**: `d = 1`, `dxd_dxn = dyd_dyn = 1`,
/// `cross = 0`. Every consumer below is written so that substituting those exact values
/// reproduces the bare pinhole *bit for bit* — `x * 1.0 == x` and `x + 0.0 == x` are exact in
/// IEEE-754, so no consumer rounds differently than it did before distortion existed. The one
/// exception is zero SIGN (`-0.0 + 0.0 == +0.0`), which needs an EXACT zero in the rotation
/// matrix — an identity pose, or a gravity-aligned rig, both common — not merely a point near an
/// optical axis. Randomised poses cannot reach it. Value-identical either way, and a sign of zero
/// cannot change any accumulation it feeds; `radial_term_is_bit_inert_at_k1_zero` covers both.
///
/// Not exact if `r²` overflows to infinity (`0.0 * inf` is NaN), which needs `|x/z| > 1.8e19`
/// — far past the point where the pinhole itself has stopped meaning anything.
struct RadialTerm {
    /// `d = 1 + k1·r²` — the isotropic scale applied to both normalised coordinates.
    d: f32,
    /// `∂xd/∂xn = d + 2k1·xn²`.
    dxd_dxn: f32,
    /// `∂yd/∂yn = d + 2k1·yn²`.
    dyd_dyn: f32,
    /// `∂xd/∂yn = ∂yd/∂xn = 2k1·xn·yn`.
    cross: f32,
    /// `r² = xn² + yn²`, kept because the `∂/∂k1` column needs it.
    r2: f32,
}

#[inline]
fn radial_term(xn: f32, yn: f32, k1: f32) -> RadialTerm {
    let r2 = xn * xn + yn * yn;
    let d = 1.0 + k1 * r2;
    let two_k1 = 2.0 * k1;
    RadialTerm {
        d,
        dxd_dxn: d + two_k1 * xn * xn,
        dyd_dyn: d + two_k1 * yn * yn,
        cross: two_k1 * xn * yn,
        r2,
    }
}

/// The distortion coefficient the solver SEEDS from when `BaParams::refine_k1` is not set.
///
/// Zero, and unconditionally so: with `k1` fixed, the model must evaluate to the bare pinhole
/// whatever `PinholeCamera::k1` happens to carry, and that is what keeps a default-parameter solve
/// bit-identical to the pre-distortion code. Only `refine_k1` makes the solver read `camera.k1`.
const BA_K1: f32 = 0.0;

/// The intrinsic state a residual is evaluated at.
///
/// A VALUE, not a borrow of the caller's [`PinholeCamera`], and that is the point. `fx` and `k1`
/// are free parameters, so the linearisation pass and the trial-cost pass run at DIFFERENT
/// intrinsics; if both read one shared `&PinholeCamera` the trial cost is computed under the old
/// model and `new_cost < cost` compares a new step against a stale objective — an LM that looks
/// convergent and is not. Naming the state at every evaluation makes that a compile error rather
/// than a review item.
///
/// Pre-casting the four `f64` fields once also hoists four casts per observation out of the hot
/// loop; the values are the same casts, so this is bit-inert.
#[derive(Clone, Copy, Debug, PartialEq)]
struct BaIntrinsics {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    k1: f32,
}

impl BaIntrinsics {
    /// Seed from the caller's camera. `k1` comes from [`BA_K1`] unless `refine_k1` frees it.
    fn seed(camera: &PinholeCamera, refine_k1: bool) -> Self {
        Self {
            fx: camera.fx as f32,
            fy: camera.fy as f32,
            cx: camera.cx as f32,
            cy: camera.cy as f32,
            k1: if refine_k1 { camera.k1 as f32 } else { BA_K1 },
        }
    }

    /// Write the fitted `fx`/`fy`/`k1` back onto a copy of the seed camera; everything this solver
    /// does not fit (`cx`, `cy`, `k2`, `p1`, `p2`) is copied through.
    fn into_camera(self, seed: &PinholeCamera) -> PinholeCamera {
        PinholeCamera {
            fx: self.fx as f64,
            fy: self.fy as f64,
            k1: self.k1 as f64,
            ..seed.clone()
        }
    }
}

/// Computes (residual, J_pose 2×6, J_point 2×3, J_intr 2×5) at the current state.
///
/// Jacobian layout (row-major flat):
///   J_pose[0..6]:  [du/dρ_x, du/dρ_y, du/dρ_z, du/dω_x, du/dω_y, du/dω_z]
///   J_pose[6..12]: [dv/dρ_x, dv/dρ_y, dv/dρ_z, dv/dω_x, dv/dω_y, dv/dω_z]
///   J_point[0..3]: [du/dx,   du/dy,   du/dz]
///   J_point[3..6]: [dv/dx,   dv/dy,   dv/dz]
///   J_intr[0..5]:  [du/dfx,  du/dfy,  du/dcx,  du/dcy,  du/dk1]
///   J_intr[5..10]: [dv/dfx,  dv/dfy,  dv/dcx,  dv/dcy,  dv/dk1]
///
/// The solver consumes only the `fx`/`fy` and `k1` columns, and only when the matching
/// `BaParams::refine_*` flag is set; `cx`/`cy` are never freed. See `intrinsic_jacobian_columns`
/// for how the selected columns are packed.
#[inline]
fn residual_and_jacobians(
    pose: &SE3F32,
    point_w: &Vec3F64,
    pixel: [f32; 2],
    intr: &BaIntrinsics,
) -> ([f32; 2], [f32; 12], [f32; 6], [f32; 10]) {
    let BaIntrinsics { fx, fy, cx, cy, k1 } = *intr;

    let pw = Vec3AF32::new(point_w.x as f32, point_w.y as f32, point_w.z as f32);
    let pc = *pose * pw;
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

    // Normalised coords, then the radial factors. At `BA_K1 == 0` this is `d = 1`, diagonal 1,
    // cross 0 (see [`RadialTerm`]).
    let xn = pc.x * inv_z;
    let yn = pc.y * inv_z;
    let rad = radial_term(xn, yn, k1);

    // u = fx·xd + cx with xd = xn·d, and likewise for v.
    //
    // Written as `fx * pc.x * inv_z * d` rather than `fx * (xn * d)` on purpose: f32 multiply is
    // NOT associative, so `fx * (pc.x * inv_z)` is a different float from the `(fx * pc.x) * inv_z`
    // this has to reproduce. Keeping the original grouping and appending `* d` makes the k1 = 0
    // case exact, since multiplying by 1.0 is.
    let u = fx * pc.x * inv_z * rad.d + cx;
    let v = fy * pc.y * inv_z * rad.d + cy;
    let r = [u - pixel[0], v - pixel[1]];

    // J_proj row coefficients (∂[u; v] / ∂[X_c]), chained through the distortion:
    //   ∂[u; v]/∂X_c = diag(fx, fy) · ∂(xd, yd)/∂(xn, yn) · ∂(xn, yn)/∂X_c
    // with ∂xn/∂X_c = [1/z, 0, -x/z²] and ∂yn/∂X_c = [0, 1/z, -y/z²].
    //
    // At k1 = 0 (diagonal 1, cross 0) a0/a2/b1/b2 are exactly the pinhole `fx·inv_z`,
    // `-fx·x·inv_z²`, `fy·inv_z`, `-fy·y·inv_z²`, and the cross columns a1/b0 are exactly zero.
    let a0 = fx * rad.dxd_dxn * inv_z;
    let a1 = fx * rad.cross * inv_z;
    let a2 = -fx * (rad.dxd_dxn * pc.x + rad.cross * pc.y) * inv_z2;
    let b0 = fy * rad.cross * inv_z;
    let b1 = fy * rad.dyd_dyn * inv_z;
    let b2 = -fy * (rad.cross * pc.x + rad.dyd_dyn * pc.y) * inv_z2;

    // Rotation matrix elements (R: world→cam).
    let rm = pose.r.matrix();
    let r00 = rm.col(0).x;
    let r01 = rm.col(1).x;
    let r02 = rm.col(2).x;
    let r10 = rm.col(0).y;
    let r11 = rm.col(1).y;
    let r12 = rm.col(2).y;
    let r20 = rm.col(0).z;
    let r21 = rm.col(1).z;
    let r22 = rm.col(2).z;

    let (px, py, pz) = (pw.x, pw.y, pw.z);

    // S = -R · skew(p_w) — for the omega part.
    let s00 = -pz * r01 + py * r02;
    let s10 = -pz * r11 + py * r12;
    let s20 = -pz * r21 + py * r22;

    let s01 = pz * r00 - px * r02;
    let s11 = pz * r10 - px * r12;
    let s21 = pz * r20 - px * r22;

    let s02 = -py * r00 + px * r01;
    let s12 = -py * r10 + px * r11;
    let s22 = -py * r20 + px * r21;

    // J_pt = J_proj · R (3 cols). The a1/b0 terms are the distortion cross-coupling: they are
    // exactly zero at k1 = 0, and adding an exact zero leaves the remaining sum untouched.
    let jpt_00 = a0 * r00 + a1 * r10 + a2 * r20;
    let jpt_01 = a0 * r01 + a1 * r11 + a2 * r21;
    let jpt_02 = a0 * r02 + a1 * r12 + a2 * r22;
    let jpt_10 = b0 * r00 + b1 * r10 + b2 * r20;
    let jpt_11 = b0 * r01 + b1 * r11 + b2 * r21;
    let jpt_12 = b0 * r02 + b1 * r12 + b2 * r22;

    // J_omega = J_proj · S (3 cols).
    let jom_00 = a0 * s00 + a1 * s10 + a2 * s20;
    let jom_01 = a0 * s01 + a1 * s11 + a2 * s21;
    let jom_02 = a0 * s02 + a1 * s12 + a2 * s22;
    let jom_10 = b0 * s00 + b1 * s10 + b2 * s20;
    let jom_11 = b0 * s01 + b1 * s11 + b2 * s21;
    let jom_12 = b0 * s02 + b1 * s12 + b2 * s22;

    // Layout J_pose 2×6 row-major: [ρ(3) | ω(3)] per row.
    let j_pose: [f32; 12] = [
        jpt_00, jpt_01, jpt_02, jom_00, jom_01, jom_02, jpt_10, jpt_11, jpt_12, jom_10, jom_11,
        jom_12,
    ];
    // J_point 2×3 row-major.
    let j_point: [f32; 6] = [jpt_00, jpt_01, jpt_02, jpt_10, jpt_11, jpt_12];

    // J_intr 2×5 row-major, columns [fx, fy, cx, cy, k1].
    let xd = xn * rad.d;
    let yd = yn * rad.d;
    let j_intr: [f32; 10] = [
        xd,
        0.0,
        1.0,
        0.0,
        fx * xn * rad.r2,
        0.0,
        yd,
        0.0,
        1.0,
        fy * yn * rad.r2,
    ];

    (r, j_pose, j_point, j_intr)
}

// ── Shared intrinsic border ──────────────────────────────────────────────

/// Widest border this solver builds: `α` (focal scale) and `k1`.
const Q_MAX: usize = 2;

/// COLMAP's `min_focal_length_ratio` / `max_focal_length_ratio`, applied to `α = fx / fx₀`.
const MIN_FOCAL_RATIO: f32 = 0.1;
/// See [`MIN_FOCAL_RATIO`].
const MAX_FOCAL_RATIO: f32 = 10.0;

/// How close to the fold `k1` may get: `1 + k1·r²  ≥  1 − K1_FOLD_MARGIN` at the largest observed
/// `r²`. At `k1 = −1/r²` the radial map collapses that radius to the principal point and the
/// Jacobian there is meaningless.
const K1_FOLD_MARGIN: f32 = 0.9;

/// Clamp a parameter step so that `x + δ` lands inside `[lo, hi]`, returning the ADMISSIBLE STEP.
///
/// Returning the step and not the new value is what makes this guard impossible to get half-right.
/// The pose and point back-substitutions and the trial intrinsics all read the one clamped `δ`, so
/// there is no way to trim the intrinsic while leaving the geometry moving against the untrimmed
/// one — an inconsistency that costs nothing in the accept test and everything in the step.
///
/// The price is that `x + clamped_step(..)` can overshoot the bound by up to an ulp OF `x`, not of
/// the bound: from `x = −50` toward a bound of `−0.9` the difference `−0.9 − (−50)` cancels. Both
/// bands here carry a wide safety cushion (COLMAP's ratio band is two orders of magnitude, the
/// fold margin is 0.1 of the fold), so an ulp is not worth a second rounding pass.
#[inline]
fn clamped_step(x: f32, delta: f64, lo: f32, hi: f32) -> f64 {
    ((x + delta as f32).clamp(lo, hi) - x) as f64
}

/// Symmetric bound on `k1` given the largest `r²` in the linearisation pass: `|k1| ≤ 0.9/r²_max`.
///
/// The barrel side is the singular one — that is what `K1_FOLD_MARGIN` is measured against — but
/// an unbounded pincushion is just as much a runaway, so the bound is applied both ways. Unbounded
/// when nothing was observed.
#[inline]
fn k1_fold_bound(r2_max: f32) -> f32 {
    if r2_max > 0.0 {
        K1_FOLD_MARGIN / r2_max
    } else {
        f32::INFINITY
    }
}

/// Which intrinsics are free, and in what column order the border stores them.
///
/// Column order is `[α, k1]` with the disabled one COMPACTED OUT, not zero-filled: a structurally
/// zero column would make the `q × q` reduced border singular and send every step down the
/// unobservable fallback, which is indistinguishable from "the data does not determine it".
#[derive(Clone, Copy)]
struct IntrinsicBlock {
    refine_focal: bool,
    refine_k1: bool,
    /// Number of free intrinsics, `0..=Q_MAX`.
    q: usize,
    /// Seed focals. `α` multiplies these; they never change during the solve.
    fx0: f32,
    fy0: f32,
}

impl IntrinsicBlock {
    fn new(params: &BaParams, seed: &BaIntrinsics) -> Self {
        Self {
            refine_focal: params.refine_focal,
            refine_k1: params.refine_k1,
            q: params.refine_focal as usize + params.refine_k1 as usize,
            fx0: seed.fx,
            fy0: seed.fy,
        }
    }

    /// Column index of `α` / `k1` in the border, or `None` when that intrinsic is fixed.
    fn col_alpha(&self) -> Option<usize> {
        self.refine_focal.then_some(0)
    }
    fn col_k1(&self) -> Option<usize> {
        self.refine_k1.then_some(self.refine_focal as usize)
    }

    /// The selected intrinsic Jacobian columns, as a 2×q block stored row-major with stride
    /// [`Q_MAX`]. Columns `q..Q_MAX` stay zero and are never read.
    ///
    /// `∂u/∂α = fx₀·∂u/∂fx` and `∂v/∂α = fy₀·∂v/∂fy` — the chain rule through `fx = α·fx₀`,
    /// `fy = α·fy₀`. No new derivation: this is a fixed linear combination of columns the
    /// finite-difference test already validates.
    #[inline]
    fn columns(&self, j_intr: &[f32; 10]) -> [f32; 2 * Q_MAX] {
        let mut j = [0.0_f32; 2 * Q_MAX];
        if let Some(a) = self.col_alpha() {
            j[a] = self.fx0 * j_intr[0];
            j[Q_MAX + a] = self.fy0 * j_intr[6];
        }
        if let Some(a) = self.col_k1() {
            j[a] = j_intr[4];
            j[Q_MAX + a] = j_intr[9];
        }
        j
    }
}

/// Second Schur elimination: reduce the bordered system to its `q × q` core and solve it.
///
/// Parameters are ordered `[δp (6P) | δi (q)]` AFTER the points have already been eliminated:
///
/// ```text
///   [ M  V ] [δp]   [m]
///   [ Vᵀ S ] [δi] = [s]
/// ```
///
/// `sol` is the multi-RHS solve `M⁻¹·[m | V]` — column 0 is `y₀ = M⁻¹m`, columns `1..=q` are
/// `Y = M⁻¹V` — against the SAME factorisation of the SAME `M` the pose-only path already needed.
/// Substituting `δp = y₀ − Y δi` into the second row gives `(S − VᵀY) δi = s − Vᵀy₀`.
///
/// One storage serves as both the border column and the border row: `Vᵀ = Eᵀ − FᵀC⁻¹Bᵀ` is exactly
/// the third row's coefficient of the original three-block system, so the bordered system is
/// symmetric by construction and needs no averaging pass.
///
/// Returns `[0; _]` when the border is unobservable — see [`solve_border`].
fn eliminate_border(
    s_mat: &[f64; Q_MAX * Q_MAX],
    s_vec: &[f64; Q_MAX],
    v_bord: &[f64],
    sol: &Mat<f64>,
    dim: usize,
    q: usize,
) -> [f64; Q_MAX] {
    if q == 0 {
        return [0.0; Q_MAX];
    }
    let mut s_red = *s_mat;
    let mut r_red = *s_vec;
    for a in 0..q {
        for i in 0..dim {
            let v = v_bord[i * q + a];
            r_red[a] -= v * sol[(i, 0)];
            for b in 0..q {
                s_red[a * Q_MAX + b] -= v * sol[(i, 1 + b)];
            }
        }
    }
    solve_border(&s_red, &r_red, q).unwrap_or([0.0; Q_MAX])
}

/// Recover the pose step from the same multi-RHS solve: `δp = y₀ − Y δi`.
///
/// Trivial, and factored out anyway so the dense-reference test drives the SOLVER's expression
/// instead of restating it. A test that recomputes `y₀ − Yδi` itself passes happily against a
/// driver that dropped the `− Yδi` term — measured: it did.
fn pose_step_from_border(sol: &Mat<f64>, d_intr: &[f64; Q_MAX], dim: usize, q: usize) -> Vec<f64> {
    (0..dim)
        .map(|i| {
            let mut v = sol[(i, 0)];
            for (a, di) in d_intr.iter().take(q).enumerate() {
                v -= sol[(i, 1 + a)] * di;
            }
            v
        })
        .collect()
}

/// Solve `S δ = s` for the `q × q` border, `q ≤ 2`.
///
/// `None` when the border is singular or has a non-positive diagonal, i.e. the intrinsic is
/// unobservable at this linearisation point (pure rotation, a single plane). That is a property of
/// the DATA, so the caller falls back to a zero intrinsic step and keeps the pose solve — it must
/// not become a factorisation error, or a degenerate segment turns a working solve into a failure.
fn solve_border(s: &[f64; Q_MAX * Q_MAX], rhs: &[f64; Q_MAX], q: usize) -> Option<[f64; Q_MAX]> {
    for a in 0..q {
        // `is_finite` first, so NaN — which compares false against everything — is caught here
        // rather than sliding through `<= 0.0`.
        if !s[a * Q_MAX + a].is_finite() || s[a * Q_MAX + a] <= 0.0 {
            return None;
        }
    }
    match q {
        0 => Some([0.0; Q_MAX]),
        1 => Some([rhs[0] / s[0], 0.0]),
        _ => {
            let (a, b, c, d) = (s[0], s[1], s[Q_MAX], s[Q_MAX + 1]);
            let det = a * d - b * c;
            // Mirrors `invert_3x3`'s guard; `1e-20` is the same "this is not a matrix any more"
            // threshold, read against a border kept in f64.
            if det.abs() < 1e-20 || !det.is_finite() {
                return None;
            }
            let x0 = (d * rhs[0] - b * rhs[1]) / det;
            let x1 = (a * rhs[1] - c * rhs[0]) / det;
            (x0.is_finite() && x1.is_finite()).then_some([x0, x1])
        }
    }
}

/// `r² = xn² + yn²` in normalised camera coordinates, the argument the radial term is a function
/// of. Only called when `k1` is free, to bound the clamp that keeps `1 + k1·r²` positive.
///
/// Recomputes the projection rather than threading `r²` out of [`residual_and_jacobians`], the
/// same trade `clamped_z` already makes; it is intrinsic-independent, so no trial/linearisation
/// discrepancy is possible.
#[inline]
fn normalised_r2(pose: &SE3F32, point_w: &Vec3F64) -> f32 {
    let pw = Vec3AF32::new(point_w.x as f32, point_w.y as f32, point_w.z as f32);
    let pc = *pose * pw;
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
    let xn = pc.x * inv_z;
    let yn = pc.y * inv_z;
    xn * xn + yn * yn
}

// ── Small block primitives (f32) ─────────────────────────────────────────

#[inline]
fn ata_6x6_into(acc: &mut [f32; 36], j: &[f32; 12]) {
    // acc += J.T @ J  where J is 2×6 row-major.
    let r0 = &j[0..6];
    let r1 = &j[6..12];
    for i in 0..6 {
        for k in 0..6 {
            acc[i * 6 + k] += r0[i] * r0[k] + r1[i] * r1[k];
        }
    }
}

#[inline]
fn ata_3x3_into(acc: &mut [f32; 9], j: &[f32; 6]) {
    let r0 = &j[0..3];
    let r1 = &j[3..6];
    for i in 0..3 {
        for k in 0..3 {
            acc[i * 3 + k] += r0[i] * r0[k] + r1[i] * r1[k];
        }
    }
}

#[inline]
fn atb_6x3_into(acc: &mut [f32; 18], jp: &[f32; 12], jx: &[f32; 6]) {
    // acc += J_pose.T @ J_point  →  6 × 3 row-major.
    let jp0 = &jp[0..6];
    let jp1 = &jp[6..12];
    let jx0 = &jx[0..3];
    let jx1 = &jx[3..6];
    for i in 0..6 {
        for k in 0..3 {
            acc[i * 3 + k] += jp0[i] * jx0[k] + jp1[i] * jx1[k];
        }
    }
}

#[inline]
fn atb_6x1_into(acc: &mut [f32; 6], j: &[f32; 12], r: &[f32; 2]) {
    // acc -= J.T @ r  (note negative for gradient convention).
    for i in 0..6 {
        acc[i] -= j[i] * r[0] + j[6 + i] * r[1];
    }
}

#[inline]
fn atb_3x1_into(acc: &mut [f32; 3], j: &[f32; 6], r: &[f32; 2]) {
    for i in 0..3 {
        acc[i] -= j[i] * r[0] + j[3 + i] * r[1];
    }
}

/// Invert a 3×3 row-major matrix. Returns None if singular.
fn invert_3x3(m: &[f32; 9]) -> Option<[f32; 9]> {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    let d = m[3];
    let e = m[4];
    let f = m[5];
    let g = m[6];
    let h = m[7];
    let i = m[8];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-20 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        (e * i - f * h) * inv_det,
        (c * h - b * i) * inv_det,
        (b * f - c * e) * inv_det,
        (f * g - d * i) * inv_det,
        (a * i - c * g) * inv_det,
        (c * d - a * f) * inv_det,
        (d * h - e * g) * inv_det,
        (b * g - a * h) * inv_det,
        (a * e - b * d) * inv_det,
    ])
}

#[inline]
fn matmul_6x3_3x3(a: &[f32; 18], b: &[f32; 9]) -> [f32; 18] {
    let mut out = [0.0_f32; 18];
    for i in 0..6 {
        for k in 0..3 {
            let mut s = 0.0_f32;
            for r in 0..3 {
                s += a[i * 3 + r] * b[r * 3 + k];
            }
            out[i * 3 + k] = s;
        }
    }
    out
}

#[inline]
fn matvec_6x3_3(a: &[f32; 18], b: &[f32; 3]) -> [f32; 6] {
    let mut out = [0.0_f32; 6];
    for i in 0..6 {
        out[i] = a[i * 3] * b[0] + a[i * 3 + 1] * b[1] + a[i * 3 + 2] * b[2];
    }
    out
}

#[inline]
fn matvec_3x3_3(a: &[f32; 9], b: &[f32; 3]) -> [f32; 3] {
    [
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
        a[3] * b[0] + a[4] * b[1] + a[5] * b[2],
        a[6] * b[0] + a[7] * b[1] + a[8] * b[2],
    ]
}

#[inline]
fn matvec_6x3t_6(a: &[f32; 18], b: &[f32; 6]) -> [f32; 3] {
    // returns a.T @ b  →  3-vector; a is stored row-major 6×3
    let mut out = [0.0_f32; 3];
    for k in 0..3 {
        out[k] = a[k] * b[0]
            + a[3 + k] * b[1]
            + a[6 + k] * b[2]
            + a[9 + k] * b[3]
            + a[12 + k] * b[4]
            + a[15 + k] * b[5];
    }
    out
}

// ── Block-sparse reduced camera system ───────────────────────────────────

/// Block-sparse storage for the reduced camera system `M = A − B C⁻¹ Bᵀ`.
///
/// The dense path materialises all `(6P)²` entries. Two cameras occupy a NONZERO 6×6 block only if
/// they share a point (or a motion prior couples them), and on a sequential capture that is a small
/// fraction of pairs — the rest of the matrix is structurally zero, and both the `O(dim²)` zeroing
/// and the `O(dim³)` dense factorisation are spent on it.
///
/// This holds one 6×6 block per COUPLED pair, addressed by a flat `(i1, i2) -> slot` table. The
/// table is still `O(P²)` — this does not escape quadratic memory, it cuts the constant by 36×,
/// one `usize` per pair instead of a 288-byte block: 3.2 MB at `P = 637` against the 117 MB dense
/// matrix it replaces. A direct index also beats a hash lookup in a loop that runs once per
/// (camera, camera, point) triple. Past a few thousand cameras the table itself would want a
/// compressed representation.
///
/// Blocks are `f64` because that is what the dense path accumulates into (`Mat::<f64>`). Matching
/// the storage type is what makes the two assemblies bit-identical rather than merely close.
///
/// NOTE the failure mode this is written to avoid: an earlier attempt accumulated into compact
/// blocks and then SCATTERED them into the dense matrix, which performed every one of the original
/// scattered writes plus the block pass on top. Compacting the accumulation only pays if the dense
/// matrix is never built at all.
struct BlockAccum {
    /// Row-major `(i1, i2) -> slot`, or `usize::MAX` for a pair that never couples.
    index: Vec<usize>,
    /// Flat row-major 6×6 blocks, one per coupled pair.
    blocks: Vec<[f64; 36]>,
    /// `(i1, i2)` for each slot, so triplet emission needs no scan of `index`.
    pairs: Vec<(usize, usize)>,
    /// Number of free cameras; the stride of `index`.
    n: usize,
}

impl BlockAccum {
    /// Build the coupling pattern once. It is a function of which cameras share a point — plus any
    /// pair coupled by a MOTION PRIOR, which no observation need connect.
    ///
    /// Covisibility alone is not the whole pattern. A constant-velocity prior (see
    /// [`BaMotionPrior`]) couples a triplet whether or not those cameras share structure, and when
    /// they do not, the block it writes has no slot to go to. Dropping such priors instead would be
    /// worse than failing: when two consecutive keyframes share NO point, the motion prior is the
    /// only thing connecting them, so discarding it severs the pose graph exactly where the
    /// geometry is weakest. Hence they are added to the pattern up front.
    ///
    /// The pattern is symmetric by construction — the covisibility double loop visits `(i1, i2)`
    /// and `(i2, i1)`, and motion pairs are inserted in both orders — which
    /// [`BlockAccum::lower_triplets`] relies on when it averages a block against its transpose.
    fn new(
        n: usize,
        b_by_point: &[Vec<(usize, [f32; 18])>],
        motion_pairs: &[(usize, usize)],
    ) -> Self {
        let mut index = vec![usize::MAX; n * n];
        let mut pairs = Vec::new();
        let touch = |index: &mut Vec<usize>, pairs: &mut Vec<(usize, usize)>, i1, i2| {
            let e = &mut index[i1 * n + i2];
            if *e == usize::MAX {
                *e = pairs.len();
                pairs.push((i1, i2));
            }
        };
        for b_for_j in b_by_point {
            for (i1, _) in b_for_j.iter() {
                for (i2, _) in b_for_j.iter() {
                    touch(&mut index, &mut pairs, *i1, *i2);
                }
            }
        }
        for &(a, b) in motion_pairs {
            if a >= n || b >= n {
                continue;
            }
            touch(&mut index, &mut pairs, a, b);
            touch(&mut index, &mut pairs, b, a);
        }
        // Every free camera carries its own A block, so the diagonal always exists — even for a
        // camera whose every observation is on a FIXED point, which contributes to A but to no B.
        for i in 0..n {
            touch(&mut index, &mut pairs, i, i);
        }
        let blocks = vec![[0.0; 36]; pairs.len()];
        Self {
            index,
            blocks,
            pairs,
            n,
        }
    }

    /// Zero every block, keeping the pattern. The pattern depends only on the observation graph,
    /// which does not change across LM iterations; the values do.
    fn clear(&mut self) {
        for b in self.blocks.iter_mut() {
            *b = [0.0; 36];
        }
    }

    /// Slot for a pair, or `None` if these two cameras never couple.
    #[inline]
    fn slot(&self, i1: usize, i2: usize) -> Option<usize> {
        let s = self.index[i1 * self.n + i2];
        (s != usize::MAX).then_some(s)
    }

    /// The LOWER TRIANGLE as triplets, symmetrised exactly as the dense path symmetrises.
    ///
    /// The dense path averages `M[i][j]` with `M[j][i]` over the whole matrix and then reads
    /// `Side::Lower`. Doing the same here — but only over the blocks that exist — reproduces that
    /// lower triangle bit for bit: floating-point addition is commutative, so `0.5 * (a + b)` is
    /// the same value whichever triangle each operand came from.
    ///
    /// Structural zeros are dropped, but an entry that happens to evaluate to `0.0` inside a live
    /// block is kept: dropping it would shrink the pattern from one LM iteration to the next for a
    /// reason that has nothing to do with the problem's structure.
    fn lower_triplets(&self) -> Vec<faer::sparse::Triplet<usize, usize, f64>> {
        let mut t = Vec::with_capacity(self.blocks.len() * 21);
        for (slot, &(i1, i2)) in self.pairs.iter().enumerate() {
            if i1 < i2 {
                continue; // emitted when its transpose slot comes round
            }
            let blk = &self.blocks[slot];
            // Symmetric counterpart. Guaranteed present for i1 > i2 (the pattern is symmetric);
            // for i1 == i2 it is this same block, transposed within itself.
            let t_blk = match self.slot(i2, i1) {
                Some(s) => &self.blocks[s],
                None => blk,
            };
            let (row0, col0) = (i1 * 6, i2 * 6);
            for r in 0..6 {
                for c in 0..6 {
                    if row0 + r < col0 + c {
                        continue;
                    }
                    // Operands in the dense path's order (upper first, then lower) so the two
                    // expressions are identical on sight as well as in value.
                    let v = if i1 == i2 && r == c {
                        blk[r * 6 + c]
                    } else {
                        0.5 * (t_blk[c * 6 + r] + blk[r * 6 + c])
                    };
                    t.push(faer::sparse::Triplet::new(row0 + r, col0 + c, v));
                }
            }
        }
        t
    }
}

/// Why a reduced-system factorisation failed.
///
/// The distinction matters to the LM loop. A `Numeric` failure is a property of the VALUES — an
/// indefinite or near-singular matrix — and raising λ usually clears it, which is why the dense
/// path has always retried. A `Structural` failure is a property of the sparsity PATTERN; damping
/// changes no pattern, so retrying only burns the iteration budget before failing anyway.
enum FactorFailure {
    Structural(String),
    Numeric(String),
}

/// Factorise the accumulated reduced system sparsely and solve for the pose step.
///
/// Only the lower triangle is built, which is all `Side::Lower` reads.
fn sparse_llt_solve(
    acc: &BlockAccum,
    dim: usize,
    rhs: &Mat<f64>,
) -> Result<Mat<f64>, FactorFailure> {
    use faer::linalg::solvers::Solve;
    // Straight from the compact blocks. Scanning a dense matrix for nonzeros would be an O(dim²)
    // pass to recover structure the assembly already knows — and it would force the dense matrix
    // to exist at all, which is the cost this path is here to avoid.
    let trips = acc.lower_triplets();
    let a = faer::sparse::SparseColMat::try_new_from_triplets(dim, dim, &trips)
        .map_err(|e| FactorFailure::Structural(format!("sparse reduced system: {e:?}")))?;
    let sym = faer::sparse::linalg::solvers::SymbolicLlt::try_new(a.symbolic(), faer::Side::Lower)
        .map_err(|e| FactorFailure::Structural(format!("symbolic Llt: {e:?}")))?;
    let llt = faer::sparse::linalg::solvers::Llt::try_new_with_symbolic(
        sym,
        a.as_ref(),
        faer::Side::Lower,
    )
    .map_err(|e| FactorFailure::Numeric(format!("sparse Llt: {e:?}")))?;
    Ok(llt.solve(rhs))
}

// ── Driver ───────────────────────────────────────────────────────────────

/// Bundle adjustment via dense Schur-complement reduction. Same argument list as
/// [`crate::ba::bundle_adjust`] but uses Schur internally: the reduced 6P×6P camera system is
/// solved with `faer`'s dense Cholesky; points are recovered by back-substitution.
///
/// Respects the `fixed_pose` and `fixed_point` flags on each observation, and honours
/// `BaParams::robust` (IRLS: residual and Jacobian rows are scaled by √ρ'(s), while the
/// accept test compares the true robust cost ½ρ(s)), plus `max_iterations`,
/// `initial_lambda` and `cost_tolerance`.
///
/// `RobustKernelKind::Tukey` maps to Cauchy, as it does in `BaParams::robust`. A non-finite or
/// non-positive `BaParams::robust_scale_sq` collapses to plain L2 for every kernel, as
/// `BaParams::robust_scale_sq` documents.
///
/// # `initial_lambda` does NOT mean the same thing here as in [`crate::ba::bundle_adjust`]
///
/// This solver damps ELLIPSOIDALLY — `A += λ·diag(A)`, `C += λ·diag(C)`, the diagonal clamped to
/// `[MIN_LM_DIAGONAL, MAX_LM_DIAGONAL]`, as Ceres' `LevenbergMarquardtStrategy` does. `λ` is
/// therefore DIMENSIONLESS: a fraction of the local curvature. [`crate::ba::bundle_adjust`] (and
/// `pgo`, and `pnp::refine`) still damp `JᵀJ + λ·I`, where `λ` carries the units of `JᵀJ`. The
/// same `BaParams { initial_lambda: 1e-3, .. }` is a ~0.1% relative damping here and an absolute
/// `1e-3` there; do not port a tuned value between the two.
///
/// `BaParams::gradient_tolerance` is NOT read by this solver — the only termination test is
/// the relative cost decrease against `cost_tolerance`. Callers needing a gradient-based
/// stopping rule do not get one here.
///
/// # Arguments
///
/// * `poses` - initial camera poses, `T_world_cam`, one per camera.
/// * `points` - initial 3-D landmark positions in world coordinates.
/// * `observations` - the 2-D measurements tying a pose to a point, each carrying its own
///   `fixed_pose` / `fixed_point` flags and optional depth measurement. Entries whose
///   `pose_idx` or `point_idx` is out of range are skipped rather than rejected.
/// * `camera` - the shared pinhole intrinsics used to project every observation.
/// * `params` - solver settings; see the caveats above on `initial_lambda` and
///   `gradient_tolerance`, which this solver interprets differently from
///   [`crate::ba::bundle_adjust`].
///
/// # Returns
///
/// A [`BaResult`] holding the refined poses and points, the iteration count, whether the
/// relative cost decrease met `cost_tolerance`, and `final_cost` — the objective
/// `Σ ½ρ(‖r‖²)` evaluated at the returned solution.
///
/// # Errors
///
/// Returns [`SchurBaError::NoFreeVariables`] if every pose and point is fixed,
/// [`SchurBaError::CholeskyFailed`] if the reduced camera system stays non-factorable after the
/// damping has been escalated past its limit, and [`SchurBaError::Ba`] for setup errors raised
/// by the shared bundle-adjustment layer.
pub fn bundle_adjust_schur(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera: &PinholeCamera,
    params: &BaParams,
) -> Result<BaResult, SchurBaError> {
    bundle_adjust_schur_with_priors(poses, points, observations, camera, params, None)
}

/// [`bundle_adjust_schur_with_priors`] plus constant-velocity motion priors over shot triplets
/// (see [`BaMotionPrior`]).
///
/// # What this adds that no other prior does
///
/// Every other residual family here — reprojection, depth, the centre and up priors — touches at
/// most ONE pose block, so the pose part of the Hessian stays block-diagonal and the reduced
/// camera system `M = A − B C⁻¹ Bᵀ` picks up off-diagonal entries only through shared points. A
/// motion residual couples THREE poses, so it writes genuine off-diagonal pose-pose blocks into
/// `M` — including between cameras that share no landmark at all. That is the whole point: it is
/// the only term that constrains a keyframe whose neighbours give it no parallax.
///
/// Jacobians are obtained by finite differences over the solver's own `retract`. The residual (a
/// norm ratio composed with an SO(3) log) has an unpleasant closed form, the cost is negligible
/// (a handful of triplets × 19 residual evaluations, against tens of thousands of observations),
/// and differentiating through `retract` itself means the perturbation convention can never drift
/// out of sync with the analytic residuals elsewhere in this file.
///
/// Motion residuals are σ-whitened, so they are gated by
/// [`BaParams::depth_robust_scale_sq`] rather than the reprojection knee — see that field.
///
/// Triplets that name an out-of-range pose, or whose three poses are ALL fixed, are skipped.
#[allow(clippy::too_many_arguments)]
pub fn bundle_adjust_schur_with_all_priors(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera: &PinholeCamera,
    params: &BaParams,
    pose_priors: Option<&[Option<BaPosePrior>]>,
    motion_priors: Option<&[BaMotionPrior]>,
) -> Result<BaResult, SchurBaError> {
    bundle_adjust_schur_impl(
        poses,
        points,
        observations,
        camera,
        params,
        pose_priors,
        motion_priors,
    )
}

/// Bundle adjustment via dense Schur-complement reduction with optional
/// per-pose translation priors.
///
/// Identical to [`bundle_adjust_schur`] but accepts a slice of
/// `Option<BaPosePrior>` of length `poses.len()` (entries may be `None` for
/// unconstrained poses). When a prior is present for pose `i`, the BA cost
/// gains a position residual
///
/// ```text
///     r_pos_i = (C_i_world − prior_i.center_world) / prior_i.sigma
/// ```
///
/// where `C_i_world = -R^T · t`. This anchors all three world-frame axes of
/// the pose translation simultaneously — the durable fix for lateral /
/// vertical drift that the per-observation depth residual alone (which only
/// constrains cam-frame Z) cannot close.
///
/// The pose-prior Jacobian decomposes into a 3×6 block per pose with no
/// coupling to point variables, so it augments only the on-diagonal
/// camera-block A_ii in the Schur reduction (B and C are untouched).
///
/// Poses marked fixed via `BaObservation::fixed_pose` have no free
/// parameters; any prior on them is silently ignored.
///
/// When a prior also carries an orientation term ([`BaPosePrior::up_world`]) that residual is
/// applied here too; it is likewise pose-only and lands in the same `A_ii` block.
pub fn bundle_adjust_schur_with_priors(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera: &PinholeCamera,
    params: &BaParams,
    pose_priors: Option<&[Option<BaPosePrior>]>,
) -> Result<BaResult, SchurBaError> {
    bundle_adjust_schur_impl(
        poses,
        points,
        observations,
        camera,
        params,
        pose_priors,
        None,
    )
}

/// `camera_seed` is named for the two things it may be used for, and it is used for exactly those
/// two: building the seed [`BaIntrinsics`] on the first line of the body, and copying the fields
/// this solver does not fit onto the reported camera at the end. `grep -c camera_seed` over the
/// body is 2, and that is a reviewable invariant.
///
/// It is the enforcement mechanism for the accept/reject hazard. Every residual evaluation names a
/// `&BaIntrinsics`, so reaching for the caller's camera in the trial pass — which would score a new
/// step against the OLD model and make `new_cost < cost` accept garbage — is a type error rather
/// than something a reviewer has to notice.
#[allow(clippy::too_many_arguments)]
fn bundle_adjust_schur_impl(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera_seed: &PinholeCamera,
    params: &BaParams,
    pose_priors: Option<&[Option<BaPosePrior>]>,
    motion_priors: Option<&[BaMotionPrior]>,
) -> Result<BaResult, SchurBaError> {
    // The ONLY read of `camera_seed` before the result is packed.
    let intr_seed = BaIntrinsics::seed(camera_seed, params.refine_k1);
    // Shared free-intrinsic border: which columns exist, and the fixed focals `α` scales.
    let ib = IntrinsicBlock::new(params, &intr_seed);
    let q = ib.q;
    // Live intrinsic state. `alpha` is the parameter; `intr.fx` is `α·fx₀` derived from it, never
    // the other way round, so the clamp band applies to the quantity it is written for.
    let mut intr = intr_seed;
    let mut alpha = 1.0_f32;

    // Validate prior slice length matches poses.
    if let Some(pp) = pose_priors {
        if pp.len() != poses.len() {
            return Err(SchurBaError::Ba(BaError::InvalidInput(format!(
                "pose_priors length {} != poses length {}",
                pp.len(),
                poses.len()
            ))));
        }
    }
    let p_total = poses.len();
    let n_total = points.len();

    // Index map: which poses / points are touched by any free observation.
    let mut pose_is_free = vec![false; p_total];
    let mut point_is_free = vec![false; n_total];
    for obs in observations {
        if obs.pose_idx >= p_total || obs.point_idx >= n_total {
            continue;
        }
        if !obs.fixed_pose {
            pose_is_free[obs.pose_idx] = true;
        }
        if !obs.fixed_point {
            point_is_free[obs.point_idx] = true;
        }
    }
    let pose_local: Vec<i64> = {
        let mut v = vec![-1_i64; p_total];
        let mut next = 0;
        for i in 0..p_total {
            if pose_is_free[i] {
                v[i] = next;
                next += 1;
            }
        }
        v
    };
    let point_local: Vec<i64> = {
        let mut v = vec![-1_i64; n_total];
        let mut next = 0;
        for i in 0..n_total {
            if point_is_free[i] {
                v[i] = next;
                next += 1;
            }
        }
        v
    };
    let n_free_poses = pose_local.iter().filter(|&&x| x >= 0).count();
    let n_free_points = point_local.iter().filter(|&&x| x >= 0).count();

    if n_free_poses == 0 {
        return Err(SchurBaError::NoFreeVariables);
    }

    // Mutable state.
    let mut se3s: Vec<SE3F32> = poses.iter().map(pose_to_se3).collect();
    let mut xyz: Vec<Vec3F64> = points.to_vec();

    // Robust-loss IRLS setup. Loop-invariant, so it is built once rather than per LM iteration.
    // The weight `w` scales the residual and Jacobian rows by √w (equivalent to multiplying the
    // observation's contribution to the normal equations by w); the accept test compares
    // `robust_cost` = ½ρ(s), which `robust_weight` is the derivative of.
    //
    // A non-finite or non-positive `robust_scale_sq` collapses to plain L2, matching what
    // `BaParams::robust_scale_sq` documents ("Default `f32::INFINITY` collapses to the L2 fast
    // path even for non-Identity kernel choices") and what `ba::build_robust_loss` enforces for
    // the non-Schur solver. Without this, `Cauchy`/`Tukey` at the DEFAULT scale give
    // `w = inf/inf` and `cost = 0.5·inf·ln(1) = inf·0`, i.e. NaN — and since `NaN < cost` is
    // false, every step is rejected and the solver returns its input poses with `Ok`.
    let robust = if params.robust_scale_sq.is_finite() && params.robust_scale_sq > 0.0 {
        params.robust
    } else {
        RobustKernelKind::Identity
    };
    // No `.max(1e-6)` floor here. `ba::build_robust_loss` computes `robust_scale_sq.sqrt()` with
    // no floor, and the guard above has already excluded the non-finite and non-positive cases, so
    // a floor could only make the two solvers disagree: at `robust_scale_sq = 1e-20` a floor gives
    // this solver a knee 1e4x wider than `ba::bundle_adjust` gets from the same `BaParams`.
    let robust_scale = params.robust_scale_sq.sqrt();
    // Separate knee for the σ-WHITENED residual families (depth measurements, motion priors).
    // `0.0` — the default — reuses the reprojection knee, so nothing changes for callers that do
    // not set it. See `BaParams::depth_robust_scale_sq` for why sharing one knee across two
    // different residual units silently throws the whitened family away.
    //
    // Bound BEFORE the closures below shadow the free functions of the same name.
    let depth_scale = if params.depth_robust_scale_sq > 0.0 {
        params.depth_robust_scale_sq.sqrt()
    } else {
        robust_scale
    };
    let depth_weight = |r_sq: f32| -> f32 { robust_weight(robust, depth_scale, r_sq) };
    let depth_cost = |r_sq: f32| -> f32 { robust_cost(robust, depth_scale, r_sq) };
    let robust_weight = |r_sq: f32| -> f32 { robust_weight(robust, robust_scale, r_sq) };
    let robust_cost = |r_sq: f32| -> f32 { robust_cost(robust, robust_scale, r_sq) };

    // Clamped, because `MAX_LM_DIAGONAL`'s overflow argument assumes a bounded λ. That bound
    // otherwise holds only via the `λ > 1e10` breaks on the reject and factorisation-failure
    // paths, which a caller-supplied `initial_lambda` reaches BEFORE any of them run:
    // `1e20 * 1e24` is +inf, the damped diagonal is inf, and the solve returns NaN poses where
    // absolute damping would merely have returned a very stiff, finite one.
    let mut lambda = params.initial_lambda.clamp(0.0, MAX_INITIAL_LAMBDA);
    let t_ba = std::time::Instant::now();
    BA_CALLS.fetch_add(1, Ordering::Relaxed);
    let mut iters_done = 0usize;
    let mut converged = false;
    // Per-view depth scale, live only in log mode. Seeded from `depth_scales_init` (a robust
    // median fit is the intended seed) so the log residual starts near its optimum instead of
    // making the solver discover the scale; missing or short entries default to 1.0.
    let log_depth = params.depth_log_residual;
    let mut dscales: Vec<f32> = vec![1.0; p_total];
    if log_depth {
        for (i, s) in params.depth_scales_init.iter().take(p_total).enumerate() {
            if *s > 0.0 {
                dscales[i] = *s;
            }
        }
    }
    // Block-sparse reduced camera system, built on the first iteration when the sparse path is
    // enabled and reused (cleared, not rebuilt) thereafter: its pattern is a function of the
    // observation graph, which does not change across LM iterations.
    let mut accum: Option<BlockAccum> = None;
    // Objective at the parameters currently held in `se3s`/`xyz`. Kept in step with them: set to
    // `cost` once it is evaluated for this iteration, and to `new_cost` when a step is accepted,
    // so every exit path — converged, budget exhausted, damping blown out — reports the value at
    // the parameters actually returned. NaN only if `max_iterations == 0`, where nothing is ever
    // evaluated.
    let mut final_cost = f32::NAN;

    for _iter in 0..params.max_iterations {
        iters_done += 1;

        // Exact block update of the depth scales at the current geometry, before linearising the
        // rest. Alternating this closed form with the LM step is block-coordinate descent on the
        // joint objective: same stationary point as folding `s_i` in as a 7th camera DOF, without
        // widening every 6x6 block in the Schur reduction. A negative `depth_scale_prior` freezes
        // them at the seed — the fitted-then-frozen baseline this mechanism exists to beat.
        if log_depth && params.depth_scale_prior >= 0.0 {
            update_depth_scales(
                &mut dscales,
                &se3s,
                &xyz,
                observations,
                params.depth_scale_prior,
                &depth_weight,
            );
        }

        // ── Linearise: build A, C, B (per-obs), g_pose, g_point ──────────
        // A: n_free_poses × [36] (6×6 blocks).
        // C: n_free_points × [9]  (3×3 blocks).
        // We also keep observation-aligned B blocks (6×3) so we can iterate
        // by point during the Schur reduction.
        let mut a_blocks = vec![[0.0_f32; 36]; n_free_poses];
        let mut c_blocks = vec![[0.0_f32; 9]; n_free_points];
        let mut g_pose = vec![[0.0_f32; 6]; n_free_poses];
        let mut g_point = vec![[0.0_f32; 3]; n_free_points];

        // Per-observation B contributions, grouped by point (for the Schur
        // pass). We store (pose_local_idx, B_6x3) lists per free-point index.
        let mut b_by_point: Vec<Vec<(usize, [f32; 18])>> = vec![Vec::new(); n_free_points];

        // ── The shared-intrinsic border ─────────────────────────────────
        // A shared intrinsic couples EVERY camera to every other, so the reduced camera system
        // grows a DENSE BORDER: (6P + q) × (6P + q) with a full last block-column. Storing it that
        // way would destroy the covisibility sparsity the sparse path exists to exploit, so the
        // border lives in these four plain `Vec`s, OUTSIDE both the dense `Mat` and the block
        // accumulator, and is eliminated by its own Schur complement (see the bordered solve
        // below). `M`'s pattern, `dim`, and every line of `BlockAccum` are untouched.
        //
        //   D (q×q)      = Σ Jᵢᵀ Jᵢ                    — the border's own block
        //   g_intr (q)   = −Σ Jᵢᵀ r                    — its gradient
        //   E (6×q each) = Σ J_poseᵀ Jᵢ                — pose ↔ intrinsic coupling
        //   F (3×q each) = Σ J_pointᵀ Jᵢ               — point ↔ intrinsic coupling
        //
        // `F` is indexed BY POINT ONLY, not by (pose, point) as `b_by_point` is. That asymmetry is
        // exactly why the border is q columns and not 2P.
        let mut d_intr = [0.0_f32; Q_MAX * Q_MAX];
        let mut g_intr = [0.0_f32; Q_MAX];
        let mut e_blocks = vec![[0.0_f32; 6 * Q_MAX]; if q > 0 { n_free_poses } else { 0 }];
        let mut f_blocks = vec![[0.0_f32; 3 * Q_MAX]; if q > 0 { n_free_points } else { 0 }];
        // Largest r² seen this iteration, for the `1 + k1·r² > 0` clamp.
        let mut r2_max = 0.0_f32;

        // Also record observations that touch FIXED point but FREE pose —
        // contribute to A and g_pose only, no B.
        // (Symmetric case: free point + fixed pose contributes to C and
        //  g_point only. Both we handle below.)

        let t_lin = std::time::Instant::now();
        let mut cost = 0.0_f32;
        let mut n_depth_obs_iter = 0usize;
        let mut n_reproj_obs_iter = 0usize;

        for obs in observations {
            if obs.pose_idx >= p_total || obs.point_idx >= n_total {
                continue;
            }
            n_reproj_obs_iter += 1;
            let pose = &se3s[obs.pose_idx];
            let point = &xyz[obs.point_idx];
            // `j_intr` feeds the shared border, and only the selected columns of it.
            let (mut r, mut j_pose, mut j_point, j_intr) =
                residual_and_jacobians(pose, point, obs.pixel, &intr);
            let r_sq = r[0] * r[0] + r[1] * r[1];
            // 2×q border columns, zero-width and skipped when no intrinsic is free.
            let mut j_i = if q > 0 {
                ib.columns(&j_intr)
            } else {
                [0.0_f32; 2 * Q_MAX]
            };

            // IRLS weight; apply √w to r and J.
            let w = robust_weight(r_sq);
            if w != 1.0 {
                let sw = w.sqrt();
                r[0] *= sw;
                r[1] *= sw;
                for v in j_pose.iter_mut() {
                    *v *= sw;
                }
                for v in j_point.iter_mut() {
                    *v *= sw;
                }
                // The intrinsic columns take the SAME √w. Omitting this leaves an observation the
                // robust kernel down-weighted out of the geometry voting at full strength on the
                // intrinsic it was rejected from having an opinion about.
                for v in j_i.iter_mut() {
                    *v *= sw;
                }
            }
            cost += robust_cost(r_sq);

            let pli = pose_local[obs.pose_idx];
            let xli = point_local[obs.point_idx];

            if pli >= 0 {
                let pli = pli as usize;
                ata_6x6_into(&mut a_blocks[pli], &j_pose);
                atb_6x1_into(&mut g_pose[pli], &j_pose, &r);
            }
            if xli >= 0 {
                let xli = xli as usize;
                ata_3x3_into(&mut c_blocks[xli], &j_point);
                atb_3x1_into(&mut g_point[xli], &j_point, &r);
            }
            if pli >= 0 && xli >= 0 {
                let mut b_block = [0.0_f32; 18];
                atb_6x3_into(&mut b_block, &j_pose, &j_point);
                b_by_point[xli as usize].push((pli as usize, b_block));
            }

            if q > 0 {
                // `D` and `g_intr` take EVERY in-range observation, including those on fixed poses
                // and fixed points: a gauge-fixed camera is a perfectly good constraint on a shared
                // focal, and dropping it would throw away signal for no reason. Only `E` and `F`
                // are guarded, because only they have a free block to attach to.
                for a in 0..q {
                    for b in 0..q {
                        d_intr[a * Q_MAX + b] += j_i[a] * j_i[b] + j_i[Q_MAX + a] * j_i[Q_MAX + b];
                    }
                    g_intr[a] -= j_i[a] * r[0] + j_i[Q_MAX + a] * r[1];
                }
                if pli >= 0 {
                    let e = &mut e_blocks[pli as usize];
                    for i in 0..6 {
                        for a in 0..q {
                            e[i * Q_MAX + a] += j_pose[i] * j_i[a] + j_pose[6 + i] * j_i[Q_MAX + a];
                        }
                    }
                }
                if xli >= 0 {
                    let f = &mut f_blocks[xli as usize];
                    for i in 0..3 {
                        for a in 0..q {
                            f[i * Q_MAX + a] +=
                                j_point[i] * j_i[a] + j_point[3 + i] * j_i[Q_MAX + a];
                        }
                    }
                }
                if ib.refine_k1 {
                    r2_max = r2_max.max(normalised_r2(pose, point));
                }
            }

            // ── Depth residual (optional metric anchor) ─────────────────
            // r_z = (Z_pred − d_meas) / σ_depth
            // ∂Z/∂ρ  = e_z   (translation tangent contributes 1 to z)
            // ∂Z/∂ω  = row 2 of S = -R · skew(p_w)
            // ∂Z/∂Xw = row 2 of R
            // We treat the depth residual as a single extra row in the
            // stacked Jacobian, weighted by 1/σ. Its outer products are
            // added to A_p, C_p, B as for any other residual.
            if let Some(d_meas) = obs.depth_meas {
                let sigma = obs.depth_sigma.max(1e-6);

                // Recompute Z_pred + jacobian rows. We need the same z-clamp
                // semantics, and the geometry-only Jacobians (no projection
                // coefficients a0/b1/a2/b2).
                let pw = Vec3AF32::new(point.x as f32, point.y as f32, point.z as f32);
                let z_pred = clamped_z(pose, point);

                // Depth residual and `∂r/∂z`. In the absolute form the derivative is the constant
                // `1/σ` the Jacobian rows used to carry; in the log form it varies with `z`, which
                // is exactly what makes a fractional error cost the same near and far.
                let s_depth = dscales.get(obs.pose_idx).copied().unwrap_or(1.0);
                let (r_z, inv_sigma) = depth_residual(z_pred, d_meas, s_depth, sigma, log_depth);

                // J rows (1×6 pose, 1×3 point), all scaled by 1/σ.
                let rm = pose.r.matrix();
                let r20 = rm.col(0).z;
                let r21 = rm.col(1).z;
                let r22 = rm.col(2).z;
                let (px, py, pz) = (pw.x, pw.y, pw.z);
                // Row 2 of S = -R · skew(p_w):
                //   col0: -pz·r21 + py·r22
                //   col1:  pz·r20 - px·r22
                //   col2: -py·r20 + px·r21
                let s20 = -pz * r21 + py * r22;
                let s21 = pz * r20 - px * r22;
                let s22 = -py * r20 + px * r21;

                // J_pose_depth (1×6): [ρ(0,0,1) | ω(s20, s21, s22)] / σ
                let jpd = [
                    0.0_f32 * inv_sigma,
                    0.0_f32 * inv_sigma,
                    1.0_f32 * inv_sigma,
                    s20 * inv_sigma,
                    s21 * inv_sigma,
                    s22 * inv_sigma,
                ];
                // J_point_depth (1×3): [r20, r21, r22] / σ
                let jxd = [r20 * inv_sigma, r21 * inv_sigma, r22 * inv_sigma];

                // ── Apply IRLS robust weight to the depth residual ────────
                // The depth residual is a single scalar r_z (already scaled by
                // 1/σ_depth). Use the same Huber/Cauchy gate as the
                // reprojection path so outlier depth measurements (e.g.
                // boundary mis-samples) do not dominate the normal equations.
                // The gate uses ‖r_z‖² of the *whitened* residual, matching
                // the χ² interpretation (ORB-SLAM3 §IV.B uses χ²=7.815 for
                // 3-DoF RGB-D; `BaParams::depth_robust_scale_sq` selects the knee, defaulting to
                // the reprojection one for backwards compatibility).
                let r_sq_d = r_z * r_z;
                let w_d = depth_weight(r_sq_d);
                cost += depth_cost(r_sq_d);
                n_depth_obs_iter += 1;

                // Accumulate into A (6×6) — w · outer product jpd·jpdᵀ.
                if pli >= 0 {
                    let pli_u = pli as usize;
                    let ab = &mut a_blocks[pli_u];
                    for i in 0..6 {
                        for k in 0..6 {
                            ab[i * 6 + k] += w_d * jpd[i] * jpd[k];
                        }
                    }
                    // g_pose -= w · jpdᵀ · r_z
                    let gp = &mut g_pose[pli_u];
                    for i in 0..6 {
                        gp[i] -= w_d * jpd[i] * r_z;
                    }
                }
                // Accumulate into C (3×3) — w · outer product jxd·jxdᵀ.
                if xli >= 0 {
                    let xli_u = xli as usize;
                    let cb = &mut c_blocks[xli_u];
                    for i in 0..3 {
                        for k in 0..3 {
                            cb[i * 3 + k] += w_d * jxd[i] * jxd[k];
                        }
                    }
                    let gx = &mut g_point[xli_u];
                    for i in 0..3 {
                        gx[i] -= w_d * jxd[i] * r_z;
                    }
                }
                // Accumulate into B (6×3) — w · jpd·jxdᵀ.
                if pli >= 0 && xli >= 0 {
                    let mut b_block = [0.0_f32; 18];
                    for i in 0..6 {
                        for k in 0..3 {
                            b_block[i * 3 + k] = w_d * jpd[i] * jxd[k];
                        }
                    }
                    b_by_point[xli as usize].push((pli as usize, b_block));
                }
            }
        }
        // Recorded HERE, in the linearisation pass, not in the trial pass. The trial pass sits
        // downstream of the factorisation, so a solve whose Cholesky fails on every retry until
        // `λ > 1e10` would return `Err` having never written `BA_OBS` — leaving it holding the
        // count from a different, earlier problem, which is precisely the class of error it was
        // added to prevent. The linearisation pass runs on every iteration that runs at all.
        //
        // Depth rows are included: `n_depth_obs_iter` was already being counted and thrown away
        // one line below this, while `BA_OBS`'s doc apologised for being a lower bound.
        BA_OBS.store(
            (n_reproj_obs_iter + n_depth_obs_iter) as u64,
            Ordering::Relaxed,
        );

        // ── Per-pose translation prior (3-D position residual) ──────────────
        // For each pose i with a Some(prior), contribute a 3-row residual
        //
        //     r_pos = (C - C_prior) / σ
        //
        // with C = -R^T · t (camera centre in world frame). Jacobian wrt the
        // pose tangent ξ = [ρ; ω] is
        //
        //     ∂C/∂ρ = -I                 (3×3)
        //     ∂C/∂ω = [C]_×              (3×3, skew of C)
        //
        // derived from the right-perturbation retract `T·exp(ξ)` matching the
        // convention used by `residual_and_jacobians` above (see ReprojFactor
        // docs). With no coupling to point variables, this only augments the
        // pose-block A_ii and g_pose[i]; B and C in the Schur reduction are
        // untouched.
        if let Some(pp_slice) = pose_priors {
            for i_global in 0..p_total {
                let Some(prior) = pp_slice[i_global] else {
                    continue;
                };
                let pli = pose_local[i_global];
                if pli < 0 {
                    // Pose fixed — prior is moot.
                    continue;
                }
                let pli_u = pli as usize;
                let sigma = prior.sigma.max(1e-6);
                let inv_sigma = 1.0_f32 / sigma;

                // Camera centre C = -R^T · t.
                let pose = &se3s[i_global];
                let rm = pose.r.matrix();
                let t = pose.t;
                // R^T · t (i.e. R-transpose-times-t — apply R as world←cam to t).
                // rm.col(j) is column j of R (cam→world if you read it as R^T … but
                // our convention has R as world→cam). So R^T · t = sum over rows.
                // R^T_row0 = (r00, r10, r20) = R.col(0); so R^T · t = (R.col(0)·t,
                // R.col(1)·t, R.col(2)·t).
                let r_col0 = rm.col(0);
                let r_col1 = rm.col(1);
                let r_col2 = rm.col(2);
                let rt_t_x = r_col0.x * t.x + r_col0.y * t.y + r_col0.z * t.z;
                let rt_t_y = r_col1.x * t.x + r_col1.y * t.y + r_col1.z * t.z;
                let rt_t_z = r_col2.x * t.x + r_col2.y * t.y + r_col2.z * t.z;
                let c_pred = [-rt_t_x, -rt_t_y, -rt_t_z];

                // Residual r_pos = (C − C_prior) / σ  (3-vector).
                let r_pos = [
                    (c_pred[0] - prior.center_world[0]) * inv_sigma,
                    (c_pred[1] - prior.center_world[1]) * inv_sigma,
                    (c_pred[2] - prior.center_world[2]) * inv_sigma,
                ];

                // ── Apply IRLS robust weight to the pose-prior residual ───
                // The gate uses ‖r_pos‖² (sum of three whitened squared
                // components). This dampens single-pose VO glitches (a
                // mis-aligned chain step) so they cannot dominate the prior
                // term. We reuse `robust_scale_sq` for consistency with the
                // reprojection path; the residual is already whitened by 1/σ
                // so the gate is on the χ²-equivalent magnitude.
                let r_sq_p = r_pos[0] * r_pos[0] + r_pos[1] * r_pos[1] + r_pos[2] * r_pos[2];
                let w_p = robust_weight(r_sq_p);
                cost += robust_cost(r_sq_p);

                // Jacobian (3×6), all scaled by 1/σ:
                //   ∂C/∂ρ = -I
                //   ∂C/∂ω = [C]_× =  [ 0   -cz   cy ]
                //                    [ cz   0   -cx ]
                //                    [-cy   cx   0  ]
                let cx_ = c_pred[0];
                let cy_ = c_pred[1];
                let cz_ = c_pred[2];
                // Row-major 3×6 layout: [ρ(3) | ω(3)] per row.
                let j_pose_prior: [f32; 18] = [
                    // Row 0 (dCx)
                    -inv_sigma,
                    0.0,
                    0.0,
                    0.0,
                    -cz_ * inv_sigma,
                    cy_ * inv_sigma,
                    // Row 1 (dCy)
                    0.0,
                    -inv_sigma,
                    0.0,
                    cz_ * inv_sigma,
                    0.0,
                    -cx_ * inv_sigma,
                    // Row 2 (dCz)
                    0.0,
                    0.0,
                    -inv_sigma,
                    -cy_ * inv_sigma,
                    cx_ * inv_sigma,
                    0.0,
                ];

                // Accumulate into A_ii (6×6) — w · Σ_r J_r.T · J_r over 3 rows.
                let ab = &mut a_blocks[pli_u];
                for r_idx in 0..3 {
                    let row = &j_pose_prior[r_idx * 6..(r_idx + 1) * 6];
                    for ii in 0..6 {
                        for kk in 0..6 {
                            ab[ii * 6 + kk] += w_p * row[ii] * row[kk];
                        }
                    }
                }
                // RHS: g_pose -= w · Σ_r J_r.T · r_pos[r]
                let gp = &mut g_pose[pli_u];
                for r_idx in 0..3 {
                    let row = &j_pose_prior[r_idx * 6..(r_idx + 1) * 6];
                    for ii in 0..6 {
                        gp[ii] -= w_p * row[ii] * r_pos[r_idx];
                    }
                }

                // ── Optional orientation (up-vector) prior ────────────────
                // u_pred = Rᵀ · (0,−1,0), i.e. minus row 1 of R: where the image's up axis
                // currently points in the world. See `up_prior_residual`.
                //
                // For a FIXED camera-frame vector v this solver's right-perturbation convention
                // gives ∂(Rᵀv)/∂ω = +[Rᵀv]× and no ρ coupling — the same pattern the centre
                // prior's ω part follows (its [C]× IS [Rᵀ(−t)]×). Purely rotational, so like the
                // centre prior it augments only A_ii; B and C are untouched.
                if let Some(upw) = prior.up_world {
                    let inv_su = 1.0_f32 / prior.up_sigma.max(1e-6);
                    let (r_up, u_pred) =
                        up_prior_residual([r_col0.y, r_col1.y, r_col2.y], upw, prior.up_sigma);
                    let r_sq_u = r_up[0] * r_up[0] + r_up[1] * r_up[1] + r_up[2] * r_up[2];
                    // Reprojection knee, not the whitened one: the up residual is divided by
                    // `up_sigma` in unit-vector units, but it is gated for the same reason the
                    // centre prior is — to stop one badly-oriented view dominating — and it
                    // shares the centre prior's knee so a single pose prior is scored coherently.
                    let w_u = robust_weight(r_sq_u);
                    cost += robust_cost(r_sq_u);

                    let (ux, uy, uz) = (u_pred[0], u_pred[1], u_pred[2]);
                    // Row-major 3×6: rows of [u]× scaled by 1/σ in the ω half, ρ half zero.
                    let j_up: [f32; 18] = [
                        // Row 0 (dUx)
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -uz * inv_su,
                        uy * inv_su,
                        // Row 1 (dUy)
                        0.0,
                        0.0,
                        0.0,
                        uz * inv_su,
                        0.0,
                        -ux * inv_su,
                        // Row 2 (dUz)
                        0.0,
                        0.0,
                        0.0,
                        -uy * inv_su,
                        ux * inv_su,
                        0.0,
                    ];
                    let ab = &mut a_blocks[pli_u];
                    for r_idx in 0..3 {
                        let row = &j_up[r_idx * 6..(r_idx + 1) * 6];
                        for ii in 0..6 {
                            for kk in 0..6 {
                                ab[ii * 6 + kk] += w_u * row[ii] * row[kk];
                            }
                        }
                    }
                    let gp = &mut g_pose[pli_u];
                    for r_idx in 0..3 {
                        let row = &j_up[r_idx * 6..(r_idx + 1) * 6];
                        for ii in 0..6 {
                            gp[ii] -= w_u * row[ii] * r_up[r_idx];
                        }
                    }
                }
            }
        }

        // ── Motion priors (constant-velocity triplets) ──────────────────────
        // The ONLY residual family here that couples more than one pose, so the only one that
        // produces off-diagonal pose-pose blocks in the reduced camera system. They are collected
        // here and written into M after the A blocks are placed.
        //
        // Jacobians are finite differences over the solver's own `retract` (see
        // `bundle_adjust_schur_with_all_priors` for why). Residuals are σ-whitened, so they are
        // gated with the depth-family knee — see `BaParams::depth_robust_scale_sq` for why they
        // must not share the reprojection knee.
        let mut h_offdiag: std::collections::HashMap<(usize, usize), [f32; 36]> =
            std::collections::HashMap::new();
        if let Some(mps) = motion_priors {
            // Finite-difference step, RELATIVE for the translation block.
            //
            // The residual's translation term is built from camera centres (`-Rᵀt`, an f32 dot
            // product of world-scale quantities), so a fixed absolute step loses relative
            // precision as the map grows: measured, the error on dC/dρ is ~0.06% on a map a few
            // metres across but degrades linearly with extent, and a walkthrough map is tens of
            // metres. Scaling the step with the local coordinate magnitude holds the ratio
            // constant instead. Rotation parameters are dimensionless radians, so they keep the
            // absolute step.
            const FD_EPS: f32 = 1e-4;
            for mp in mps {
                if mp.i0 >= p_total || mp.i1 >= p_total || mp.i2 >= p_total {
                    continue;
                }
                let tri = [mp.i0, mp.i1, mp.i2];
                let locs = [pose_local[mp.i0], pose_local[mp.i1], pose_local[mp.i2]];
                if locs.iter().all(|&l| l < 0) {
                    // Fully fixed triplet constrains nothing.
                    continue;
                }
                let r0 = motion_prior_residual(&se3s[mp.i0], &se3s[mp.i1], &se3s[mp.i2], mp);
                let r_sq_m: f32 = r0.iter().map(|v| v * v).sum();
                let w_m = depth_weight(r_sq_m);
                cost += depth_cost(r_sq_m);

                // J: 6 residual rows × (3 poses × 6 params), FD one parameter at a time.
                let mut jac = [[0.0f32; 18]; 6];
                for (pi, &g) in tri.iter().enumerate() {
                    if locs[pi] < 0 {
                        continue;
                    }
                    // Magnitude of this camera's centre, the scale the translation columns are
                    // differentiated at. `1.0 +` keeps the step finite at the origin.
                    let c_mag = se3s[g].inverse().t.length();
                    for k in 0..6 {
                        let step = if k < 3 {
                            FD_EPS * (1.0 + c_mag)
                        } else {
                            FD_EPS
                        };
                        let mut delta = [0.0f32; 6];
                        delta[k] = step;
                        let pert = se3s[g].retract(&delta);
                        let refs = [
                            if pi == 0 { &pert } else { &se3s[mp.i0] },
                            if pi == 1 { &pert } else { &se3s[mp.i1] },
                            if pi == 2 { &pert } else { &se3s[mp.i2] },
                        ];
                        let rp = motion_prior_residual(refs[0], refs[1], refs[2], mp);
                        // Divided by the step ACTUALLY taken, not the nominal constant.
                        for row in 0..6 {
                            jac[row][pi * 6 + k] = (rp[row] - r0[row]) / step;
                        }
                    }
                }

                // Accumulate JᵀJ (per pose-PAIR block) and Jᵀr (per pose).
                for a in 0..3 {
                    let la = locs[a];
                    if la < 0 {
                        continue;
                    }
                    let la = la as usize;
                    let gp = &mut g_pose[la];
                    for i in 0..6 {
                        let mut acc = 0.0f32;
                        for row in 0..6 {
                            acc += jac[row][a * 6 + i] * r0[row];
                        }
                        gp[i] -= w_m * acc;
                    }
                    for b in 0..3 {
                        let lb = locs[b];
                        if lb < 0 {
                            continue;
                        }
                        let lb = lb as usize;
                        let mut blk = [0.0f32; 36];
                        for i in 0..6 {
                            for j in 0..6 {
                                let mut acc = 0.0f32;
                                for row in &jac {
                                    acc += row[a * 6 + i] * row[b * 6 + j];
                                }
                                blk[i * 6 + j] = w_m * acc;
                            }
                        }
                        if la == lb {
                            let ab = &mut a_blocks[la];
                            for x in 0..36 {
                                ab[x] += blk[x];
                            }
                        } else {
                            // Both orderings (a,b) and (b,a) are visited by this double loop, so
                            // BOTH keys get their own block — do NOT additionally write a
                            // transpose when draining this map or every off-diagonal entry is
                            // counted twice.
                            let e = h_offdiag.entry((la, lb)).or_insert([0.0f32; 36]);
                            for x in 0..36 {
                                e[x] += blk[x];
                            }
                        }
                    }
                }
            }
        }

        final_cost = cost;
        BA_LIN_NANOS.fetch_add(t_lin.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t_asm = std::time::Instant::now();

        // ── Apply LM damping: A[i] += λ·diag(A), C[j] += λ·diag(C) ─────
        //
        // Ellipsoidal, as Ceres' LevenbergMarquardtStrategy does it: damp with diag(JᵀJ) — here
        // already assembled as the block diagonals — clamped to [min_lm_diagonal,
        // max_lm_diagonal]. See `MAX_LM_DIAGONAL`: the ceiling here is 1e24, NOT Ceres' 1e32,
        // because this solver is f32 and 1e32 * a large λ overflows to +inf.
        //
        // A spherical λ·I damps every direction by the same ABSOLUTE amount, whatever its
        // curvature. That is not scale-free: one λ has to serve parameter blocks in different
        // units — rotation in radians, translation and points in metres — so it over-damps the
        // stiff directions and under-damps the soft ones, and λ itself is not dimensionless.
        // Scaling by the local curvature makes the damping relative.
        for ab in &mut a_blocks {
            for d in 0..6 {
                ab[d * 6 + d] += lambda * ab[d * 6 + d].clamp(MIN_LM_DIAGONAL, MAX_LM_DIAGONAL);
            }
        }
        for cb in &mut c_blocks {
            for d in 0..3 {
                cb[d * 3 + d] += lambda * cb[d * 3 + d].clamp(MIN_LM_DIAGONAL, MAX_LM_DIAGONAL);
            }
        }
        // Same ellipsoidal damping on the border, or the intrinsic direction alone runs at
        // undamped Gauss-Newton while every other block is damped: a wild focal jump on iteration
        // one, rejected, escalating λ for everybody.
        for d in 0..q {
            d_intr[d * Q_MAX + d] +=
                lambda * d_intr[d * Q_MAX + d].clamp(MIN_LM_DIAGONAL, MAX_LM_DIAGONAL);
        }

        // ── Build M (6Pf × 6Pf) + m (6Pf) ───────────────────────────────
        let dim = n_free_poses * 6;
        // 117 MB at P = 637, allocated and zeroed EVERY LM iteration. The sparse path never touches
        // it, so it is a 0×0 stub there; `Mat` has no null state, and the branches below never index
        // it while the accumulator is live.
        let mut m_mat = if params.sparse_reduced_system {
            Mat::<f64>::zeros(0, 0)
        } else {
            Mat::<f64>::zeros(dim, dim)
        };
        let mut m_vec = vec![0.0_f64; dim];

        // Build the sparse pattern before ANY write into the reduced system: the A-block loop just
        // below is the first writer and must not take the dense branch against the 0×0 stub.
        if params.sparse_reduced_system && accum.is_none() {
            // Motion priors couple cameras regardless of covisibility, so their pairs must be in
            // the pattern or the block they write has nowhere to go. Mapped into the SAME local
            // index space the accumulator uses; unregistered members drop out via `pose_local < 0`.
            let mut motion_pairs: Vec<(usize, usize)> = Vec::new();
            if let Some(mps) = motion_priors {
                for mp in mps {
                    if mp.i0 >= p_total || mp.i1 >= p_total || mp.i2 >= p_total {
                        continue;
                    }
                    let locs: Vec<usize> = [mp.i0, mp.i1, mp.i2]
                        .iter()
                        .filter_map(|&g| {
                            let l = pose_local[g];
                            (l >= 0).then_some(l as usize)
                        })
                        .collect();
                    for (x, &a) in locs.iter().enumerate() {
                        for &b in locs.iter().skip(x + 1) {
                            motion_pairs.push((a, b));
                        }
                    }
                }
            }
            accum = Some(BlockAccum::new(n_free_poses, &b_by_point, &motion_pairs));
        }
        if let Some(acc) = accum.as_mut() {
            acc.clear();
        }

        // ── Border, in f64 alongside `m_vec` ────────────────────────────
        // `S ← D`, `s ← g_intr`, `V ← E` (dim × q, row-major); the point elimination below
        // subtracts the `C⁻¹` corrections into all three. In f64 because the two border columns
        // are badly scaled against each other in pixel-space callers: the `k1` column is `fx·xn·r²`
        // and the `α` column is `fx₀·xd`, so `D`'s diagonals differ by ~fx² ≈ 2.5e5 at fx = 500.
        // (In a normalised-coordinate caller, fx₀ = 1, the spread is ~1, and this is moot.) A THIRD
        // intrinsic would want explicit Jacobi scaling of the border, not just f64.
        let mut s_mat = [0.0_f64; Q_MAX * Q_MAX];
        let mut s_vec = [0.0_f64; Q_MAX];
        for a in 0..q {
            s_vec[a] = g_intr[a] as f64;
            for b in 0..q {
                s_mat[a * Q_MAX + b] = d_intr[a * Q_MAX + b] as f64;
            }
        }
        let mut v_bord = vec![0.0_f64; dim * q];

        // Place A blocks on diagonal of M.
        for (k, ab) in a_blocks.iter().enumerate() {
            match accum.as_mut() {
                Some(acc) => {
                    let slot = acc
                        .slot(k, k)
                        .expect("BlockAccum::new inserts a diagonal slot for every free camera");
                    let blk = &mut acc.blocks[slot];
                    for i in 0..6 {
                        for j in 0..6 {
                            blk[i * 6 + j] = ab[i * 6 + j] as f64;
                        }
                    }
                }
                None => {
                    for i in 0..6 {
                        for j in 0..6 {
                            m_mat[(k * 6 + i, k * 6 + j)] = ab[i * 6 + j] as f64;
                        }
                    }
                }
            }
            for i in 0..6 {
                m_vec[k * 6 + i] = g_pose[k][i] as f64;
            }
            for i in 0..6 {
                for a in 0..q {
                    v_bord[(k * 6 + i) * q + a] = e_blocks[k][i * Q_MAX + a] as f64;
                }
            }
        }

        // Motion-prior off-diagonal pose-pose blocks (already IRLS-weighted).
        //
        // `HashMap` iteration order is randomised per process, which normally makes a
        // float accumulation order-dependent and therefore non-reproducible. It is safe here and
        // only here: the keys are unique, each writes a DISJOINT 6×6 destination block, and each
        // block is written with a single `+=` onto a cell no other key touches. Order cannot
        // change the result. Do not extend this loop to accumulate onto shared cells without
        // switching to a deterministic ordering.
        for ((la, lb), blk) in &h_offdiag {
            match accum.as_mut() {
                // `BlockAccum::new` was handed every motion-prior pair, so a live prior always has
                // a slot. If one somehow does not, fail loudly rather than silently dropping the
                // term: when two consecutive keyframes share no point, this prior is the ONLY thing
                // connecting them, and quietly discarding it severs the pose graph precisely where
                // the geometry is weakest.
                Some(acc) => match acc.slot(*la, *lb) {
                    Some(slot) => {
                        let dst = &mut acc.blocks[slot];
                        for i in 0..6 {
                            for j in 0..6 {
                                dst[i * 6 + j] += blk[i * 6 + j] as f64;
                            }
                        }
                    }
                    None => {
                        record_call_totals(&t_ba, iters_done);
                        return Err(SchurBaError::CholeskyFailed(format!(
                            "motion prior couples free cameras {la} and {lb}, which the sparse \
                             pattern does not cover"
                        )));
                    }
                },
                None => {
                    for i in 0..6 {
                        for j in 0..6 {
                            m_mat[(la * 6 + i, lb * 6 + j)] += blk[i * 6 + j] as f64;
                        }
                    }
                }
            }
        }

        // For each free point j: invert C_j, accumulate Schur correction
        //   M[i1, i2] -= B[i1, j] · C_j⁻¹ · B[i2, j].T
        //   m[i]     -= B[i, j]  · C_j⁻¹ · g_point[j]
        // Skip if C_j is singular (rare, but be safe).
        let mut c_inv_blocks: Vec<Option<[f32; 9]>> = Vec::with_capacity(n_free_points);
        for cb in &c_blocks {
            c_inv_blocks.push(invert_3x3(cb));
        }

        for (j, b_for_j) in b_by_point.iter().enumerate() {
            let Some(c_inv_j) = c_inv_blocks[j] else {
                continue;
            };
            // Pre-compute B_i · C⁻¹ for each i in this point's edge list.
            let bc: Vec<(usize, [f32; 18])> = b_for_j
                .iter()
                .map(|(i_loc, b)| (*i_loc, matmul_6x3_3x3(b, &c_inv_j)))
                .collect();

            // RHS: m[i] -= (B_i · C⁻¹) · g_point[j]
            let gp = g_point[j];
            for (i_loc, bc_block) in &bc {
                let bc_g = matvec_6x3_3(bc_block, &gp);
                let base = i_loc * 6;
                for r in 0..6 {
                    m_vec[base + r] -= bc_g[r] as f64;
                }
            }

            // Border's share of the SAME point elimination, off the SAME `c_inv_j` and `bc` — no
            // extra pass over the observation graph:
            //   fc = F_jᵀ C⁻¹        (q×3)
            //   s  -= fc · g_point[j]
            //   S  -= fc · F_j
            //   V[i] -= (B_i C⁻¹) · F_j     for every camera i on this point
            if q > 0 {
                let f_j = &f_blocks[j];
                let mut fc = [0.0_f32; Q_MAX * 3];
                for a in 0..q {
                    for k in 0..3 {
                        let mut acc = 0.0_f32;
                        for l in 0..3 {
                            acc += f_j[l * Q_MAX + a] * c_inv_j[l * 3 + k];
                        }
                        fc[a * 3 + k] = acc;
                    }
                }
                for a in 0..q {
                    let mut sg = 0.0_f32;
                    for k in 0..3 {
                        sg += fc[a * 3 + k] * gp[k];
                    }
                    s_vec[a] -= sg as f64;
                    for b in 0..q {
                        let mut sf = 0.0_f32;
                        for k in 0..3 {
                            sf += fc[a * 3 + k] * f_j[k * Q_MAX + b];
                        }
                        s_mat[a * Q_MAX + b] -= sf as f64;
                    }
                }
                for (i_loc, bc_block) in &bc {
                    let base = i_loc * 6;
                    for r in 0..6 {
                        for a in 0..q {
                            let mut sv = 0.0_f32;
                            for k in 0..3 {
                                sv += bc_block[r * 3 + k] * f_j[k * Q_MAX + a];
                            }
                            v_bord[(base + r) * q + a] -= sv as f64;
                        }
                    }
                }
            }

            // LHS: M[i1, i2] -= (B_i1 · C⁻¹) · B_i2.T   (6×6 block)
            for (idx1, (i1_loc, bc1)) in bc.iter().enumerate() {
                for (idx2, (i2_loc, _bc2_unused)) in bc.iter().enumerate() {
                    let b2 = &b_for_j[idx2].1;
                    // (6×3) @ (3×6) — bc1 (6×3) times b2.T (3×6).
                    // Compute element (r, c): sum_k bc1[r, k] · b2[c, k]
                    let row0 = i1_loc * 6;
                    let col0 = i2_loc * 6;
                    let _ = idx1;
                    let _ = idx2;
                    // Any pair reached here shares point j, so covisibility put it in the pattern.
                    // Resolved once per pair, outside the element loops: same subtraction, in the
                    // same order, into the same f64 destination type as the dense branch — only
                    // the address differs, which is what makes the two assemblies bit-identical.
                    let slot = accum.as_ref().map(|acc| {
                        acc.slot(*i1_loc, *i2_loc).expect(
                            "cameras sharing this point were covisible when the pattern was built",
                        )
                    });
                    match slot {
                        Some(slot) => {
                            let dst =
                                &mut accum.as_mut().expect("slot implies a live accum").blocks
                                    [slot];
                            for r in 0..6 {
                                for c in 0..6 {
                                    let mut s = 0.0_f32;
                                    for k in 0..3 {
                                        s += bc1[r * 3 + k] * b2[c * 3 + k];
                                    }
                                    dst[r * 6 + c] -= s as f64;
                                }
                            }
                        }
                        None => {
                            for r in 0..6 {
                                for c in 0..6 {
                                    let mut s = 0.0_f32;
                                    for k in 0..3 {
                                        s += bc1[r * 3 + k] * b2[c * 3 + k];
                                    }
                                    m_mat[(row0 + r, col0 + c)] -= s as f64;
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Solve M · δ_pose = m via Cholesky ────────────────────────────
        // Symmetrize numerically (the construction above should already be
        // symmetric to within roundoff; do an average to guarantee).
        //
        // Dense path only: this is an O(dim²) pass over the full matrix. The accumulator does the
        // same averaging inside `lower_triplets`, over the blocks that exist rather than over all
        // of them, and produces the same lower triangle bit for bit.
        if accum.is_none() {
            for i in 0..dim {
                for j in (i + 1)..dim {
                    let avg = 0.5 * (m_mat[(i, j)] + m_mat[(j, i)]);
                    m_mat[(i, j)] = avg;
                    m_mat[(j, i)] = avg;
                }
            }
        }
        BA_ASM_NANOS.fetch_add(t_asm.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t_fact = std::time::Instant::now();
        // RHS as faer columns. Identical for both paths — only the LHS storage differs.
        //
        // `1 + q` columns: `m` plus the border column block `V`, solved against the SAME
        // factorisation of the SAME `M`. That is the whole trick — freeing a shared intrinsic costs
        // `q` extra triangular solves against a factor that already exists, and touches neither the
        // sparsity pattern nor `dim`. faer sizes its solve scratch from `rhs.ncols()` on both the
        // dense and the sparse `Llt`, so no signature changes.
        let m_col = Mat::<f64>::from_fn(dim, 1 + q, |i, c| {
            if c == 0 {
                m_vec[i]
            } else {
                v_bord[i * q + (c - 1)]
            }
        });
        // A failed factorisation is not free, and it is not cheaper than a successful one — charge
        // it to the factor phase before unwinding, or the phase split silently attributes its
        // (already-banked) linearise and assemble time while crediting factor with nothing, biasing
        // exactly the comparison these counters exist to make.
        //
        // `BA_CALLS` was already incremented on entry; the give-up path records this call's work
        // too, or `BA_NANOS / BA_CALLS` under-reports by the failure rate and `BA_ITERS` loses
        // every iteration of every failed solve.
        let solved = match accum.as_ref() {
            Some(acc) => sparse_llt_solve(acc, dim, &m_col),
            None => m_mat
                .llt(faer::Side::Lower)
                .map(|chol| chol.solve(&m_col))
                .map_err(|e| FactorFailure::Numeric(format!("{e:?}"))),
        };
        let d_pose_col = match solved {
            Ok(d) => d,
            Err(FactorFailure::Structural(why)) => {
                BA_FACT_NANOS.fetch_add(t_fact.elapsed().as_nanos() as u64, Ordering::Relaxed);
                record_call_totals(&t_ba, iters_done);
                return Err(SchurBaError::CholeskyFailed(why));
            }
            Err(FactorFailure::Numeric(why)) => {
                BA_FACT_NANOS.fetch_add(t_fact.elapsed().as_nanos() as u64, Ordering::Relaxed);
                // Bump damping and retry next outer iteration.
                lambda *= 10.0;
                if lambda > 1e10 {
                    record_call_totals(&t_ba, iters_done);
                    return Err(SchurBaError::CholeskyFailed(why));
                }
                continue;
            }
        };

        // ── Eliminate the border: S_red δi = s_red, then δp = y₀ − Y δi ──
        // An unobservable border (pure rotation, a single plane) comes back as δi = 0, which keeps
        // the pose solve; it must NOT be turned into a `CholeskyFailed` by a property of the data.
        let mut d_intr_step = eliminate_border(&s_mat, &s_vec, &v_bord, &d_pose_col, dim, q);

        // ── Trial intrinsics, and the guards on them ────────────────────
        //
        // CLAMP, do not reject. A clamped step is just a shorter step and `new_cost < cost` still
        // decides it correctly; rejecting on the band instead would let a runaway focal escalate λ
        // and stall the pose solve that was doing nothing wrong.
        //
        // The clamp is folded back into `d_intr_step` BEFORE the pose and point back-substitutions
        // read it, so all three blocks move along one consistent step rather than the geometry
        // compensating for an intrinsic change that was then trimmed.
        let mut alpha_trial = alpha;
        let mut intr_trial = intr;
        if let Some(a) = ib.col_alpha() {
            d_intr_step[a] = clamped_step(alpha, d_intr_step[a], MIN_FOCAL_RATIO, MAX_FOCAL_RATIO);
            alpha_trial = alpha + d_intr_step[a] as f32;
            intr_trial.fx = alpha_trial * ib.fx0;
            intr_trial.fy = alpha_trial * ib.fy0;
        }
        if let Some(a) = ib.col_k1() {
            // `1 + k1·r²` must stay positive over every observed `r²`, or the radial map folds the
            // image over itself and the residual stops being a function of the geometry.
            let hi = k1_fold_bound(r2_max);
            d_intr_step[a] = clamped_step(intr.k1, d_intr_step[a], -hi, hi);
            intr_trial.k1 = intr.k1 + d_intr_step[a] as f32;
        }

        // ── Back-substitute for points: δ_x[j] = C⁻¹ (g_x - B.T · δ_p - F · δi) ──
        let d_pose = pose_step_from_border(&d_pose_col, &d_intr_step, dim, q);
        let mut d_point = vec![[0.0_f32; 3]; n_free_points];
        for (j, b_for_j) in b_by_point.iter().enumerate() {
            let Some(c_inv_j) = c_inv_blocks[j] else {
                continue;
            };
            // rhs = g_point[j] - sum_i B[i, j].T · δ_pose[i] - F_j · δi
            let mut rhs = g_point[j];
            for (i_loc, b_block) in b_for_j {
                let mut dp6 = [0.0_f32; 6];
                let base = i_loc * 6;
                for r in 0..6 {
                    dp6[r] = d_pose[base + r] as f32;
                }
                let contrib = matvec_6x3t_6(b_block, &dp6);
                for c in 0..3 {
                    rhs[c] -= contrib[c];
                }
            }
            // The `− F_j δi` term. Omitting it is silent: points then absorb the focal change as a
            // systematic radial depth bias, the cost still decreases, LM still accepts, and the
            // solve still looks convergent. `bordered_solve_matches_a_dense_reference` is what
            // catches it.
            if q > 0 {
                let f_j = &f_blocks[j];
                for (c, rhs_c) in rhs.iter_mut().enumerate() {
                    for a in 0..q {
                        *rhs_c -= f_j[c * Q_MAX + a] * d_intr_step[a] as f32;
                    }
                }
            }
            d_point[j] = matvec_3x3_3(&c_inv_j, &rhs);
        }

        BA_FACT_NANOS.fetch_add(t_fact.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t_trial = std::time::Instant::now();

        // ── Trial: retract poses, add to points, recompute cost ─────────
        let mut se3s_trial = se3s.clone();
        for i_global in 0..p_total {
            let pli = pose_local[i_global];
            if pli < 0 {
                continue;
            }
            let pli = pli as usize;
            let delta: [f32; 6] = [
                d_pose[pli * 6] as f32,
                d_pose[pli * 6 + 1] as f32,
                d_pose[pli * 6 + 2] as f32,
                d_pose[pli * 6 + 3] as f32,
                d_pose[pli * 6 + 4] as f32,
                d_pose[pli * 6 + 5] as f32,
            ];
            se3s_trial[i_global] = se3s[i_global].retract(&delta);
        }
        let mut xyz_trial = xyz.clone();
        for i_global in 0..n_total {
            let xli = point_local[i_global];
            if xli < 0 {
                continue;
            }
            let xli = xli as usize;
            let dp = d_point[xli];
            xyz_trial[i_global] = Vec3F64::new(
                xyz[i_global].x + dp[0] as f64,
                xyz[i_global].y + dp[1] as f64,
                xyz[i_global].z + dp[2] as f64,
            );
        }

        let mut new_cost = 0.0_f32;
        for obs in observations {
            if obs.pose_idx >= p_total || obs.point_idx >= n_total {
                continue;
            }
            let pose = &se3s_trial[obs.pose_idx];
            let point = &xyz_trial[obs.point_idx];
            // `intr_trial`, NOT `intr`. Once fx/k1 are free, evaluating the trial cost at the old
            // intrinsics compares the OLD model against a NEW step, and `new_cost < cost` accepts
            // garbage — a solver that looks convergent and is not. The type system is what makes
            // this safe: there is no `&PinholeCamera` in scope to reach for by accident.
            let (r, _, _, _) = residual_and_jacobians(pose, point, obs.pixel, &intr_trial);
            let r_sq = r[0] * r[0] + r[1] * r[1];
            new_cost += robust_cost(r_sq);

            // Depth residual contribution to trial cost, scored with the same robust loss as
            // the linearisation pass so accept/reject reflects one objective.
            if let Some(d_meas) = obs.depth_meas {
                let sigma = obs.depth_sigma.max(1e-6);
                let z_pred = clamped_z(pose, point);
                // The SAME `depth_residual` call as the linearisation pass, so accept/reject
                // scores one objective. Recomputing it by hand here is how the two drift apart —
                // even `x / s` versus `x * (1.0 / s)` differs by an ulp in f32 (up to 67% of
                // inputs at σ = 0.03, one-sided), which biases `new_cost < cost` toward accept.
                let s_depth = dscales.get(obs.pose_idx).copied().unwrap_or(1.0);
                let (r_z, _) = depth_residual(z_pred, d_meas, s_depth, sigma, log_depth);
                let r_sq_d = r_z * r_z;
                new_cost += depth_cost(r_sq_d);
            }
        }

        // Pose-prior contribution to trial cost.
        if let Some(pp_slice) = pose_priors {
            for i_global in 0..p_total {
                let Some(prior) = pp_slice[i_global] else {
                    continue;
                };
                if pose_local[i_global] < 0 {
                    continue;
                }
                let sigma = prior.sigma.max(1e-6);
                let inv_sigma = 1.0_f32 / sigma;
                let pose = &se3s_trial[i_global];
                let rm = pose.r.matrix();
                let t = pose.t;
                let r_col0 = rm.col(0);
                let r_col1 = rm.col(1);
                let r_col2 = rm.col(2);
                let rt_t_x = r_col0.x * t.x + r_col0.y * t.y + r_col0.z * t.z;
                let rt_t_y = r_col1.x * t.x + r_col1.y * t.y + r_col1.z * t.z;
                let rt_t_z = r_col2.x * t.x + r_col2.y * t.y + r_col2.z * t.z;
                let c_pred = [-rt_t_x, -rt_t_y, -rt_t_z];
                let r0 = (c_pred[0] - prior.center_world[0]) * inv_sigma;
                let r1 = (c_pred[1] - prior.center_world[1]) * inv_sigma;
                let r2 = (c_pred[2] - prior.center_world[2]) * inv_sigma;
                // Scored with the same robust loss as the linearisation pass so accept/reject
                // reflects one objective.
                let r_sq_p = r0 * r0 + r1 * r1 + r2 * r2;
                new_cost += robust_cost(r_sq_p);

                // Up-prior contribution. The accept test must see the SAME objective the
                // linearisation minimised, or LM rejects every step that trades a little
                // reprojection for uprightness — which is every step this prior exists to take.
                if let Some(upw) = prior.up_world {
                    let (r_up, _) =
                        up_prior_residual([r_col0.y, r_col1.y, r_col2.y], upw, prior.up_sigma);
                    let r_sq_u = r_up[0] * r_up[0] + r_up[1] * r_up[1] + r_up[2] * r_up[2];
                    new_cost += robust_cost(r_sq_u);
                }
            }
        }

        // Motion-prior contribution to the trial cost — same accept-test-consistency argument.
        if let Some(mps) = motion_priors {
            for mp in mps {
                if mp.i0 >= p_total || mp.i1 >= p_total || mp.i2 >= p_total {
                    continue;
                }
                if [mp.i0, mp.i1, mp.i2].iter().all(|&g| pose_local[g] < 0) {
                    continue;
                }
                let r = motion_prior_residual(
                    &se3s_trial[mp.i0],
                    &se3s_trial[mp.i1],
                    &se3s_trial[mp.i2],
                    mp,
                );
                let r_sq_m: f32 = r.iter().map(|v| v * v).sum();
                new_cost += depth_cost(r_sq_m);
            }
        }

        BA_TRIAL_NANOS.fetch_add(t_trial.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if new_cost < cost {
            // Accept step.
            let rel = if cost > 1e-12 {
                (cost - new_cost) / cost
            } else {
                0.0
            };
            se3s = se3s_trial;
            xyz = xyz_trial;
            // Committed with the geometry, in the same breath, so no exit path can return poses
            // fitted under one intrinsic model and a camera describing another.
            intr = intr_trial;
            alpha = alpha_trial;
            final_cost = new_cost;
            // Floor at 1e-7, not 1e-8. Damping is now RELATIVE (`λ·diag`), and in f32 any
            // `λ < 2⁻²⁵ ≈ 3e-8` makes `x + λ·x == x` bit-exactly for EVERY magnitude of `x` (at
            // 6e-8 it depends on where `x` sits in its binade: measured 0/2000 magnitudes) —
            // so a 1e-8 floor is not "almost Gauss-Newton", it is silently NO damping at all,
            // reached after ~11 accepted steps from the default `initial_lambda = 1e-3`.
            lambda = (lambda / 3.0).max(1e-7);
            if rel < params.cost_tolerance {
                converged = true;
                break;
            }
        } else {
            // Reject — bump damping and retry.
            lambda *= 10.0;
            if lambda > 1e10 {
                break;
            }
        }
    }

    // Pack results.
    let mut out_poses = Vec::with_capacity(p_total);
    for i in 0..p_total {
        if pose_is_free[i] {
            out_poses.push(se3_to_pose(&se3s[i]));
        } else {
            out_poses.push(poses[i]);
        }
    }
    let mut out_points = Vec::with_capacity(n_total);
    for i in 0..n_total {
        if point_is_free[i] {
            out_points.push(xyz[i]);
        } else {
            out_points.push(points[i]);
        }
    }

    record_call_totals(&t_ba, iters_done);

    Ok(BaResult {
        poses: out_poses,
        points: out_points,
        depth_scales: if log_depth { dscales } else { Vec::new() },
        camera: (q > 0).then(|| intr.into_camera(camera_seed)),
        iterations: iters_done,
        converged,
        final_cost,
    })
}

#[cfg(test)]
mod tests {

    /// The radial term must be INERT at `k1 = 0` — bit for bit, not to within a tolerance.
    ///
    /// A tolerance is not good enough here. A 1-ulp shift in one residual changes the trial cost,
    /// which changes an LM accept/reject, which changes the entire trajectory of the solve. So
    /// this transcribes the pre-distortion formulas verbatim and compares raw bit patterns.
    ///
    /// It is also what pins the two non-obvious writing choices in `residual_and_jacobians`: that
    /// `u` keeps the original `(fx·x)·inv_z` grouping (f32 multiply is not associative, so
    /// `fx·(x·inv_z)` is a *different float*), and that the new cross columns are added on the
    /// side that leaves the surviving sum untouched.
    #[test]
    fn radial_term_is_bit_inert_at_k1_zero() {
        use super::*;

        assert_eq!(BA_K1, 0.0, "this test only characterises the k1 = 0 solver");

        /// The projection exactly as it was before the radial term existed.
        fn pinhole_reference(
            pose: &SE3F32,
            point_w: &Vec3F64,
            pixel: [f32; 2],
            camera: &PinholeCamera,
        ) -> ([f32; 2], [f32; 12], [f32; 6]) {
            let fx = camera.fx as f32;
            let fy = camera.fy as f32;
            let cx = camera.cx as f32;
            let cy = camera.cy as f32;

            let pw = Vec3AF32::new(point_w.x as f32, point_w.y as f32, point_w.z as f32);
            let pc = *pose * pw;
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

            let u = fx * pc.x * inv_z + cx;
            let v = fy * pc.y * inv_z + cy;
            let r = [u - pixel[0], v - pixel[1]];

            let a0 = fx * inv_z;
            let a2 = -fx * pc.x * inv_z2;
            let b1 = fy * inv_z;
            let b2 = -fy * pc.y * inv_z2;

            let rm = pose.r.matrix();
            let (r00, r01, r02) = (rm.col(0).x, rm.col(1).x, rm.col(2).x);
            let (r10, r11, r12) = (rm.col(0).y, rm.col(1).y, rm.col(2).y);
            let (r20, r21, r22) = (rm.col(0).z, rm.col(1).z, rm.col(2).z);

            let (px, py, pz) = (pw.x, pw.y, pw.z);
            let s00 = -pz * r01 + py * r02;
            let s10 = -pz * r11 + py * r12;
            let s20 = -pz * r21 + py * r22;
            let s01 = pz * r00 - px * r02;
            let s11 = pz * r10 - px * r12;
            let s21 = pz * r20 - px * r22;
            let s02 = -py * r00 + px * r01;
            let s12 = -py * r10 + px * r11;
            let s22 = -py * r20 + px * r21;

            let jpt_00 = a0 * r00 + a2 * r20;
            let jpt_01 = a0 * r01 + a2 * r21;
            let jpt_02 = a0 * r02 + a2 * r22;
            let jpt_10 = b1 * r10 + b2 * r20;
            let jpt_11 = b1 * r11 + b2 * r21;
            let jpt_12 = b1 * r12 + b2 * r22;

            let jom_00 = a0 * s00 + a2 * s20;
            let jom_01 = a0 * s01 + a2 * s21;
            let jom_02 = a0 * s02 + a2 * s22;
            let jom_10 = b1 * s10 + b2 * s20;
            let jom_11 = b1 * s11 + b2 * s21;
            let jom_12 = b1 * s12 + b2 * s22;

            let j_pose: [f32; 12] = [
                jpt_00, jpt_01, jpt_02, jom_00, jom_01, jom_02, jpt_10, jpt_11, jpt_12, jom_10,
                jom_11, jom_12,
            ];
            let j_point: [f32; 6] = [jpt_00, jpt_01, jpt_02, jpt_10, jpt_11, jpt_12];
            (r, j_pose, j_point)
        }

        struct Rng(u64);
        impl Rng {
            fn next_f32(&mut self) -> f32 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            }
        }
        let mut rng = Rng(0x5eed_1234_abcd_ef01);

        // Generic geometry: nothing degenerate, so the bit patterns must match EXACTLY.
        let mut generic: Vec<(SE3F32, Vec3F64, [f32; 2], PinholeCamera)> = Vec::new();
        for _ in 0..2000 {
            let pose = SE3F32::IDENTITY.retract(&[
                2.0 * rng.next_f32(),
                2.0 * rng.next_f32(),
                2.0 * rng.next_f32(),
                0.8 * rng.next_f32(),
                0.8 * rng.next_f32(),
                0.8 * rng.next_f32(),
            ]);
            let point = Vec3F64::new(
                (5.0 * rng.next_f32()) as f64,
                (5.0 * rng.next_f32()) as f64,
                (5.0 * rng.next_f32()) as f64,
            );
            let pixel = [400.0 * rng.next_f32(), 300.0 * rng.next_f32()];
            let camera = PinholeCamera {
                fx: 500.0 + 200.0 * rng.next_f32() as f64,
                fy: 500.0 + 200.0 * rng.next_f32() as f64,
                cx: 320.0 + 50.0 * rng.next_f32() as f64,
                cy: 240.0 + 50.0 * rng.next_f32() as f64,
                k1: 0.0,
                k2: 0.0,
                p1: 0.0,
                p2: 0.0,
            };
            generic.push((pose, point, pixel, camera));
        }
        // Degenerate geometry the solver can still hit: points behind the camera and on the
        // z-clamp, and points sitting exactly on an optical axis (where a coordinate is a signed
        // zero — see `same_value` below).
        let cam = PinholeCamera {
            fx: 517.3,
            fy: 516.5,
            cx: 318.6,
            cy: 255.3,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let degenerate: Vec<(SE3F32, Vec3F64, [f32; 2], PinholeCamera)> = vec![
            (SE3F32::IDENTITY, Vec3F64::ZERO, [0.0, 0.0], cam.clone()),
            (
                SE3F32::IDENTITY,
                Vec3F64::new(0.0, 0.0, 1e-9),
                [1.0, -1.0],
                cam.clone(),
            ),
            (
                SE3F32::IDENTITY,
                Vec3F64::new(0.0, 2.0, -3.0),
                [12.0, 8.0],
                cam.clone(),
            ),
            (
                SE3F32::IDENTITY,
                Vec3F64::new(-0.0, 2.0, 4.0),
                [3.0, 3.0],
                cam.clone(),
            ),
            (
                SE3F32::IDENTITY,
                Vec3F64::new(-0.0, -2.0, 4.0),
                [3.0, 3.0],
                cam.clone(),
            ),
            (
                SE3F32::IDENTITY,
                Vec3F64::new(3.0, -0.0, 4.0),
                [3.0, 3.0],
                cam.clone(),
            ),
        ];

        /// Same value, allowing the two encodings of zero to count as equal.
        ///
        /// `to_bits` equality is the real assertion and holds for all generic geometry. The one
        /// admitted exception is zero SIGN: when a camera-frame coordinate is `-0.0`, the new
        /// (exactly zero) cross term can turn `-0.0 + 0.0` into `+0.0` where the old code kept
        /// `-0.0`. Same value, same behaviour in every sum and product downstream — and only
        /// reachable for a point lying exactly on an optical axis.
        fn same_value(a: f32, b: f32) -> bool {
            a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0)
        }

        for (i, (pose, point, pixel, camera)) in generic.iter().chain(degenerate.iter()).enumerate()
        {
            // `refine_k1: false` — the seed this test characterises, i.e. `BA_K1`.
            let intr = BaIntrinsics::seed(camera, false);
            let (r, j_pose, j_point, j_intr) = residual_and_jacobians(pose, point, *pixel, &intr);
            let (r_ref, j_pose_ref, j_point_ref) = pinhole_reference(pose, point, *pixel, camera);

            let strict = i < generic.len();
            for (k, (a, b)) in r.iter().zip(r_ref.iter()).enumerate() {
                assert!(
                    same_value(*a, *b),
                    "case {i}: residual[{k}] {a:?} ({:#010x}) != pinhole {b:?} ({:#010x})",
                    a.to_bits(),
                    b.to_bits()
                );
                if strict {
                    assert_eq!(a.to_bits(), b.to_bits(), "case {i}: residual[{k}] bits");
                }
            }
            for (k, (a, b)) in j_pose.iter().zip(j_pose_ref.iter()).enumerate() {
                assert!(
                    same_value(*a, *b),
                    "case {i}: j_pose[{k}] {a:?} ({:#010x}) != pinhole {b:?} ({:#010x})",
                    a.to_bits(),
                    b.to_bits()
                );
                if strict {
                    assert_eq!(a.to_bits(), b.to_bits(), "case {i}: j_pose[{k}] bits");
                }
            }
            for (k, (a, b)) in j_point.iter().zip(j_point_ref.iter()).enumerate() {
                assert!(
                    same_value(*a, *b),
                    "case {i}: j_point[{k}] {a:?} ({:#010x}) != pinhole {b:?} ({:#010x})",
                    a.to_bits(),
                    b.to_bits()
                );
                if strict {
                    assert_eq!(a.to_bits(), b.to_bits(), "case {i}: j_point[{k}] bits");
                }
            }

            // The intrinsic columns are not fed to the solver, but they must be the pinhole ones
            // at k1 = 0: ∂u/∂fx = x/z, ∂u/∂cx = 1, and the k1 column ∝ r², which is 0 only on the
            // optical axis. Everything else in the 2×5 is structurally zero.
            let fx = camera.fx as f32;
            let pw = Vec3AF32::new(point.x as f32, point.y as f32, point.z as f32);
            let pc = *pose * pw;
            let z_ref = if pc.z.abs() < MIN_Z {
                if pc.z >= 0.0 {
                    MIN_Z
                } else {
                    -MIN_Z
                }
            } else {
                pc.z
            };
            let inv_z = 1.0 / z_ref;
            let (xn, yn) = (pc.x * inv_z, pc.y * inv_z);
            let r2 = xn * xn + yn * yn;
            assert!(same_value(j_intr[0], xn), "case {i}: du/dfx must be x/z");
            assert_eq!(j_intr[1], 0.0);
            assert_eq!(j_intr[2], 1.0);
            assert_eq!(j_intr[3], 0.0);
            assert!(same_value(j_intr[4], fx * xn * r2), "case {i}: du/dk1");
            assert_eq!(j_intr[5], 0.0);
            assert!(same_value(j_intr[6], yn), "case {i}: dv/dfy must be y/z");
            assert_eq!(j_intr[7], 0.0);
            assert_eq!(j_intr[8], 1.0);
        }
    }

    /// The `k1 = 0` reduction above cannot see an error in the distortion derivative itself —
    /// every term it checks is multiplied by zero. This pins `∂(xd, yd)/∂(xn, yn)` against a
    /// numeric derivative at NONZERO `k1`, so the chain rule the next step depends on is checked
    /// before anything is freed.
    #[test]
    fn radial_jacobian_matches_the_numeric_derivative() {
        use super::*;

        // The distorted point, independently, in f64.
        let xd = |xn: f64, yn: f64, k1: f64| xn * (1.0 + k1 * (xn * xn + yn * yn));
        let yd = |xn: f64, yn: f64, k1: f64| yn * (1.0 + k1 * (xn * xn + yn * yn));

        let h = 1e-5;
        for &k1 in &[-0.35_f64, -0.05, 0.0, 0.05, 0.4] {
            for &(xn, yn) in &[
                (0.0_f64, 0.0_f64),
                (0.3, -0.2),
                (-0.7, 0.45),
                (0.9, 0.9),
                (0.05, -0.8),
            ] {
                let rad = radial_term(xn as f32, yn as f32, k1 as f32);

                let num_dxd_dxn = (xd(xn + h, yn, k1) - xd(xn - h, yn, k1)) / (2.0 * h);
                let num_dxd_dyn = (xd(xn, yn + h, k1) - xd(xn, yn - h, k1)) / (2.0 * h);
                let num_dyd_dxn = (yd(xn + h, yn, k1) - yd(xn - h, yn, k1)) / (2.0 * h);
                let num_dyd_dyn = (yd(xn, yn + h, k1) - yd(xn, yn - h, k1)) / (2.0 * h);
                // The ∂/∂k1 column of the projection is fx·xn·r², i.e. ∂xd/∂k1 scaled by fx.
                let num_dxd_dk1 = (xd(xn, yn, k1 + h) - xd(xn, yn, k1 - h)) / (2.0 * h);

                let tol = 2e-4;
                for (name, ours, numeric) in [
                    ("dxd_dxn", rad.dxd_dxn as f64, num_dxd_dxn),
                    ("dxd_dyn", rad.cross as f64, num_dxd_dyn),
                    ("dyd_dxn", rad.cross as f64, num_dyd_dxn),
                    ("dyd_dyn", rad.dyd_dyn as f64, num_dyd_dyn),
                    ("dxd_dk1", (xn as f32 * rad.r2) as f64, num_dxd_dk1),
                ] {
                    assert!(
                        (ours - numeric).abs() <= tol * numeric.abs().max(1.0),
                        "k1={k1} xn={xn} yn={yn}: {name} analytic {ours} vs numeric {numeric}"
                    );
                }
                assert!(
                    ((rad.d as f64) - (1.0 + k1 * (xn * xn + yn * yn))).abs() <= 1e-6,
                    "k1={k1} xn={xn} yn={yn}: d"
                );
            }
        }
    }

    /// Per-camera true depth scales for [`per_camera_scale_scene`]. Product is 1, so `Σ ln s = 0`
    /// and the regulariser's preferred gauge coincides with true world scale — otherwise the
    /// recovered map shrinks by `exp(−mean(ln s))` and the tests measure the gauge choice rather
    /// than the mechanism.
    const SCENE_S_TRUE: [f64; 5] = [1.0, 1.25, 0.8, 1.25, 0.8];

    /// Scene where the depth "network" carries a PER-CAMERA scale error (±25%) — the actual
    /// failure mode of monocular metric depth on video, where each frame's prediction is
    /// individually miscalibrated. Geometry mirrors `schur_ba_with_depth_recovers_scale`, and the
    /// init carries the same 2× scale drift.
    ///
    /// Returns `(camera, true_poses, true_points, observations, init_poses, init_points)`.
    #[allow(clippy::type_complexity)]
    fn per_camera_scale_scene() -> (
        PinholeCamera,
        Vec<Pose3d>,
        Vec<Vec3F64>,
        Vec<BaObservation>,
        Vec<Pose3d>,
        Vec<Vec3F64>,
    ) {
        struct Lcg {
            state: u64,
        }
        impl Lcg {
            fn new(seed: u64) -> Self {
                Self { state: seed }
            }
            fn next_u64(&mut self) -> u64 {
                self.state = self
                    .state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.state
            }
            fn normal(&mut self) -> f64 {
                let u1 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u2 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u1 = u1.max(1e-12);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            }
        }
        let mut rng = Lcg::new(0x5EED_1234_5678_9ABC_u64);

        let cam = PinholeCamera {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        let cam_positions = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(0.1, 0.0, 0.0),
            Vec3F64::new(0.2, 0.0, 0.0),
            Vec3F64::new(0.3, 0.0, 0.0),
            Vec3F64::new(0.4, 0.0, 0.0),
        ];
        let true_poses: Vec<Pose3d> = cam_positions.iter().map(|&p| translate_pose(p)).collect();

        // Per-camera monocular scale error. Product is 1, so Σ ln s = 0 and the regulariser does
        // not bias world scale — otherwise the recovered map shrinks by exp(−mean(ln s)) and the
        // test would be measuring the gauge choice rather than the mechanism.
        let s_true = [1.0_f64, 1.25, 0.8, 1.25, 0.8];

        let mut true_points: Vec<Vec3F64> = Vec::with_capacity(50);
        for k in 0..50 {
            let kf = k as f64;
            let x = (kf * 0.37).sin() * 1.2 + (kf * 0.13).cos() * 0.4;
            let y = (kf * 0.29).cos() * 0.9 + (kf * 0.11).sin() * 0.3;
            let z = 4.0 + (kf * 0.41).sin() * 1.5;
            true_points.push(Vec3F64::new(x, y, z));
        }

        const REL_SIGMA: f32 = 0.02;
        let mut observations: Vec<BaObservation> = Vec::new();
        for (pi, pose) in true_poses.iter().enumerate() {
            for (xi, pt) in true_points.iter().enumerate() {
                let pc = pose.transform_point(pt);
                if pc.z <= 0.2 {
                    continue;
                }
                let u = cam.fx * pc.x / pc.z + cam.cx + 0.3 * rng.normal();
                let v = cam.fy * pc.y / pc.z + cam.cy + 0.3 * rng.normal();
                // The network reports z / s_true — recovering s_true is the job.
                let d_meas = (pc.z / s_true[pi]) * (1.0 + 0.02 * rng.normal());
                observations.push(BaObservation {
                    pose_idx: pi,
                    point_idx: xi,
                    pixel: [u as f32, v as f32],
                    fixed_pose: pi == 0,
                    fixed_point: false,
                    depth_meas: Some(d_meas as f32),
                    depth_sigma: REL_SIGMA,
                });
            }
        }

        // Same 2× scale drift at init as the metric-depth test.
        let init_poses: Vec<Pose3d> = true_poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i == 0 {
                    *p
                } else {
                    let t = p.translation;
                    Pose3d::new(p.rotation, Vec3F64::new(t.x * 2.0, t.y * 2.0, t.z * 2.0))
                }
            })
            .collect();
        let init_points: Vec<Vec3F64> = true_points
            .iter()
            .map(|p| Vec3F64::new(p.x * 2.0, p.y * 2.0, p.z * 2.0))
            .collect();

        (
            cam,
            true_poses,
            true_points,
            observations,
            init_poses,
            init_points,
        )
    }

    /// The log depth residual is C¹ across the cheirality changeover at `z = s·m`.
    ///
    /// NOTE: this exercises the LOG branch only — `z = s·m` with `s`, `m` > 0 is positive, so the
    /// `z ≤ 0` arm is never taken. It pins that the residual vanishes and the slope is `1/(sm·σ)`
    /// at the expansion point. It does NOT show the two branches agree; see `depth_residual`, where
    /// Every Jacobian of the residual, checked against central differences at NONZERO `k1`.
    ///
    /// This is the test the `const BA_K1` made impossible, and its absence was not theoretical:
    /// with the coefficient fixed at 0 the composed chain rule is unreachable, and SEVEN
    /// deliberately wrong derivatives passed the entire suite — `u` missing its `* rad.d`, `a0`
    /// built from `dyd_dyn`, `b0` scaled by `fx`, the `a2` cross-term sign flipped, the `b2` cross
    /// using `pc.y`, `du/dfx = xn` instead of `xd`, and `dv/dk1` scaled by `fx`. Only mutations
    /// INSIDE `radial_term` were caught, because only those changed the `k1 = 0` value.
    ///
    /// `k1 = -0.3` with `r^2 ~ 1` deliberately: at a small coefficient near the optical axis the
    /// distortion terms are within the finite-difference tolerance of zero and two of those seven
    /// slip through. The regime has to be one where the term being tested actually matters.
    #[test]
    fn every_jacobian_matches_central_differences_at_nonzero_k1() {
        const K1: f32 = -0.3;
        let camera = PinholeCamera {
            fx: 600.0,
            fy: 590.0,
            ..test_camera()
        };
        // A general rotation, not identity: the cross terms of the distorted Jacobian vanish on an
        // axis-aligned pose and the mutants that flip them would survive.
        let (ca, sa) = (0.21f64.cos(), 0.21f64.sin());
        let (cb, sb) = ((-0.14f64).cos(), (-0.14f64).sin());
        let rot = Mat3F64::from_cols(
            Vec3F64::new(ca * cb, sa * cb, -sb),
            Vec3F64::new(-sa, ca, 0.0),
            Vec3F64::new(ca * sb, sa * sb, cb),
        );
        let pose = pose_to_se3(&Pose3d::new(rot, Vec3F64::new(0.4, -0.2, 0.15)));
        // r^2 ~ 1 in normalised coordinates: far enough off-axis that the radial term dominates.
        let point = Vec3F64::new(1.6, -1.1, 1.9);
        let pixel = [301.0f32, 262.0f32];

        let intr = BaIntrinsics {
            k1: K1,
            ..BaIntrinsics::seed(&camera, false)
        };
        let (_, j_pose, j_point, j_intr) = residual_and_jacobians(&pose, &point, pixel, &intr);

        // POINT: perturb the world point.
        for axis in 0..3 {
            let h = 1e-4_f64;
            let mut lo = point;
            let mut hi = point;
            match axis {
                0 => {
                    lo.x -= h;
                    hi.x += h;
                }
                1 => {
                    lo.y -= h;
                    hi.y += h;
                }
                _ => {
                    lo.z -= h;
                    hi.z += h;
                }
            }
            let (rl, ..) = residual_and_jacobians(&pose, &lo, pixel, &intr);
            let (rh, ..) = residual_and_jacobians(&pose, &hi, pixel, &intr);
            for row in 0..2 {
                let fd = (rh[row] - rl[row]) as f64 / (2.0 * h);
                let an = j_point[row * 3 + axis] as f64;
                assert!(
                    (fd - an).abs() <= 1e-2 * an.abs().max(1.0),
                    "j_point[{row}][{axis}]: analytic {an}, central difference {fd}"
                );
            }
        }

        // INTRINSICS: fx, fy, cx, cy, k1 — the columns step 2 will free.
        let probes: [(usize, f32); 5] = [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0), (4, K1)];
        for (col, _) in probes {
            let h = if col == 4 { 1e-3 } else { 1e-2 };
            let bump = |s: f32| {
                let mut c = intr;
                match col {
                    0 => c.fx += s,
                    1 => c.fy += s,
                    2 => c.cx += s,
                    3 => c.cy += s,
                    _ => c.k1 += s,
                }
                residual_and_jacobians(&pose, &point, pixel, &c).0
            };
            let rl = bump(-h);
            let rh = bump(h);
            for row in 0..2 {
                let fd = (rh[row] - rl[row]) / (2.0 * h);
                let an = j_intr[row * 5 + col];
                assert!(
                    (fd - an).abs() <= 1e-2 * an.abs().max(1.0),
                    "j_intr[{row}][{col}]: analytic {an}, central difference {fd}"
                );
            }
        }
        // `dv/dk1` is the last column of the second row and was previously unasserted, which is
        // exactly why an `fx`-instead-of-`fy` typo there survived.
        assert!(
            j_intr[9].abs() > 1.0,
            "dv/dk1 must be non-trivial in this regime"
        );

        // The `α` column the solver actually consumes is a FIXED linear combination of the `fx`
        // and `fy` columns the loop above just validated against central differences — the chain
        // rule through `fx = α·fx₀`, `fy = α·fy₀`. Asserting the packing here is cheaper, and
        // strictly stronger, than a second finite-difference pass over `α`.
        let ib = IntrinsicBlock::new(
            &BaParams {
                refine_focal: true,
                refine_k1: true,
                ..BaParams::default()
            },
            &intr,
        );
        assert_eq!(ib.q, 2);
        assert_eq!(ib.col_alpha(), Some(0));
        assert_eq!(ib.col_k1(), Some(1));
        let cols = ib.columns(&j_intr);
        assert_eq!(cols[0], intr.fx * j_intr[0], "du/dα = fx₀ · du/dfx");
        assert_eq!(cols[Q_MAX], intr.fy * j_intr[6], "dv/dα = fy₀ · dv/dfy");
        assert_eq!(cols[1], j_intr[4], "du/dk1 column");
        assert_eq!(cols[Q_MAX + 1], j_intr[9], "dv/dk1 column");

        // With only `k1` free, its column must COMPACT to index 0, not sit at index 1 behind a
        // structurally zero `α` column — a zero column makes the reduced border singular and sends
        // every step down the "unobservable" fallback.
        let k1_only = IntrinsicBlock::new(
            &BaParams {
                refine_k1: true,
                ..BaParams::default()
            },
            &intr,
        );
        assert_eq!(k1_only.q, 1);
        assert_eq!(k1_only.col_alpha(), None);
        assert_eq!(k1_only.col_k1(), Some(0));
        assert_eq!(k1_only.columns(&j_intr)[0], j_intr[4]);

        let _ = j_pose;
    }

    /// they demonstrably do not.
    #[test]
    fn depth_residual_log_branch_is_zero_and_correctly_sloped_at_z_eq_sm() {
        let (m, s, sigma) = (3.0_f32, 1.2_f32, 0.05_f32);
        let sm = s * m;

        // Value and slope are continuous at the changeover.
        let (r_at, d_at) = depth_residual(sm, m, s, sigma, true);
        assert!(
            r_at.abs() < 1e-6,
            "residual at z = s·m should vanish: {r_at}"
        );
        let expected_slope = 1.0 / (sm * sigma);
        assert!(
            (d_at - expected_slope).abs() / expected_slope < 1e-5,
            "slope {d_at} != {expected_slope} at changeover"
        );

        // Log branch: a relative error is what it reports, so the SAME fractional error at very
        // different depths yields the same residual. That equalisation is the entire point.
        let (r_near, _) = depth_residual(1.10 * 1.0 * s, 1.0, s, sigma, true);
        let (r_far, _) = depth_residual(1.10 * 20.0 * s, 20.0, s, sigma, true);
        assert!(
            (r_near - r_far).abs() < 1e-4,
            "log residual should be depth-invariant: near {r_near} vs far {r_far}"
        );

        // Legacy branch, same comparison: the far point dominates by 20×.
        let (l_near, _) = depth_residual(1.10, 1.0, s, sigma, false);
        let (l_far, _) = depth_residual(22.0, 20.0, s, sigma, false);
        assert!(
            l_far > 15.0 * l_near,
            "metric residual should scale with depth: near {l_near} vs far {l_far}"
        );
    }

    /// A FREE per-camera depth scale recovers geometry that a FROZEN one cannot.
    ///
    /// Arm A optimises the scales jointly; arm B freezes them at 1.0, which is what a
    /// fitted-then-frozen implementation does when the fit is stale. Arm B has to distort geometry
    /// to satisfy five mutually inconsistent depth priors; arm A explains the inconsistency with
    /// the scales and leaves geometry alone.
    #[test]
    fn schur_ba_free_depth_scale_beats_frozen() {
        let (cam, true_poses, true_points, observations, init_poses, init_points) =
            per_camera_scale_scene();
        let s_true = SCENE_S_TRUE;

        let max_err = |pts: &[Vec3F64]| -> f64 {
            pts.iter()
                .zip(&true_points)
                .map(|(a, b)| (*a - *b).length())
                .fold(0.0_f64, f64::max)
        };

        let arm = |prior: f32| BaParams {
            max_iterations: 100,
            cost_tolerance: 1e-8,
            depth_log_residual: true,
            depth_scale_prior: prior,
            ..BaParams::default()
        };
        let run = |prior: f32| {
            bundle_adjust_schur(&init_poses, &init_points, &observations, &cam, &arm(prior))
                .unwrap()
        };

        let free = run(1.0);
        let frozen = run(-1.0);
        let unregularised = run(0.0);
        let err_free = max_err(&free.points);
        let err_frozen = max_err(&frozen.points);

        // λ = 0: the global rescale direction is exactly flat in the objective, so the scales
        // absorb the whole 2× init drift (recovered ≈ 2·s_true) and geometry never corrects.
        // Locked down because it is the mechanism's failure mode, not a hypothetical.
        assert!(
            max_err(&unregularised.points) > 1.0,
            "λ=0 should let the scales absorb the drift — if this now passes, the gauge changed \
             and the regulariser's justification needs rechecking"
        );

        // At λ the per-camera deviation is shrunk by 1/(1+λ) BY CONSTRUCTION, so exact recovery is
        // not the contract. What must hold: every camera's scale moves the right way and captures
        // a real fraction of its true error.
        assert_eq!(free.depth_scales.len(), true_poses.len());
        for (i, (s, t)) in free.depth_scales.iter().zip(&s_true).enumerate() {
            let (got, want) = (f64::from(*s).ln(), t.ln());
            if want.abs() < 1e-6 {
                assert!(got.abs() < 0.02, "camera {i}: scale {s} should stay near 1");
                continue;
            }
            let captured = got / want;
            assert!(
                (0.35..=1.0).contains(&captured),
                "camera {i}: recovered {s:.3} captures {captured:.2} of true {t:.3} \
                 — expected the λ=1 shrinkage of ~0.5"
            );
        }

        assert!(
            err_free < 0.85 * err_frozen,
            "free scale ({err_free:.4} m) did not beat frozen ({err_frozen:.4} m) \
             — the joint scale estimate is inert"
        );
    }

    /// The scale prior anchors to `s = 1` ABSOLUTELY, not to whatever seed the caller passed.
    ///
    /// Regression test for a measured failure. The real pipeline re-fits `depth_scales_init` from
    /// the CURRENT geometry before every solve, so an anchor that shrank toward the seed tracked
    /// the drift instead of resisting it: on a 365-keyframe walk the map doubled (54.8 m → 108 m)
    /// while every scale-invariant metric improved. The original synthetic missed it because it
    /// left `depth_scales_init` empty, making the seed 1.0 and the two anchors indistinguishable.
    ///
    /// So: hand the solver a badly WRONG seed. The result must barely move — the seed is a starting
    /// point, not a gauge.
    #[test]
    fn schur_ba_scale_prior_anchors_absolutely_not_to_seed() {
        let (cam, _true_poses, true_points, observations, init_poses, init_points) =
            per_camera_scale_scene();
        let max_err = |pts: &[Vec3F64]| -> f64 {
            pts.iter()
                .zip(&true_points)
                .map(|(a, b)| (*a - *b).length())
                .fold(0.0_f64, f64::max)
        };
        let run = |seed: &[f32]| {
            let p = BaParams {
                max_iterations: 50,
                cost_tolerance: 1e-8,
                depth_log_residual: true,
                depth_scale_prior: 1.0,
                depth_scales_init: seed.to_vec(),
                ..BaParams::default()
            };
            let r =
                bundle_adjust_schur(&init_poses, &init_points, &observations, &cam, &p).unwrap();
            (max_err(&r.points), r.depth_scales)
        };

        let (err_unit, scales_unit) = run(&[1.0; 5]);
        // A seed inflated 2× is exactly what a re-fit against 2×-drifted geometry produces.
        let (err_bad, scales_bad) = run(&[2.0; 5]);

        assert!(
            err_bad < 2.0 * err_unit.max(0.02),
            "a 2× wrong seed moved the solution ({err_unit:.4} m → {err_bad:.4} m) — the prior is \
             anchoring to the seed, so a caller that re-fits the seed each solve has no gauge"
        );
        for (i, (a, b)) in scales_unit.iter().zip(&scales_bad).enumerate() {
            assert!(
                (a - b).abs() < 0.15 * a.max(1e-6),
                "camera {i}: seed changed the converged scale ({a:.3} vs {b:.3})"
            );
        }
    }

    /// The IRLS weight must be the derivative of the cost the accept test compares:
    /// `d/ds[½ρ(s)] = ½·w(s)`, for `s = ‖r‖²`.
    ///
    /// This is the identity that was broken. The solver used to accumulate the √w-scaled
    /// residual, i.e. `½ρ'(s)·s`, which for Huber past the knee moves at exactly half the rate
    /// of the true loss — so every step's measured reduction was halved on downweighted
    /// observations while the model's prediction was not. Anything that reintroduces the
    /// surrogate INSIDE THESE TWO FUNCTIONS fails here.
    ///
    /// It does NOT see the solver's call sites, and nothing else does either: restoring the
    /// surrogate at the six `cost +=` / `new_cost +=` sites in `bundle_adjust_schur_with_priors`
    /// leaves the whole suite green. Closing that needs the compared objective observable from
    /// outside, i.e. the final cost on `BaResult`.
    #[test]
    fn robust_weight_is_the_derivative_of_robust_cost() {
        let scale = 1.5_f32;
        for kind in [
            RobustKernelKind::Identity,
            RobustKernelKind::Huber,
            RobustKernelKind::Cauchy,
            // Tukey is aliased to Cauchy here; included so the alias is pinned rather than
            // merely assumed.
            RobustKernelKind::Tukey,
        ] {
            // Straddle the knee: inside, at, and well past it.
            for &r in &[0.1_f32, 0.9, 1.4, 1.5, 1.6, 3.0, 10.0] {
                let s = r * r;
                let h = 1e-3_f32 * s.max(1.0);
                let d_num =
                    (robust_cost(kind, scale, s + h) - robust_cost(kind, scale, s - h)) / (2.0 * h);
                let d_ana = 0.5 * robust_weight(kind, scale, s);
                assert!(
                    (d_num - d_ana).abs() <= 2e-3 * d_ana.abs().max(1e-3),
                    "{kind:?} at r={r}: d/ds cost = {d_num}, but w/2 = {d_ana}"
                );
            }
        }
    }

    /// Past the Huber knee the IRLS surrogate `½·w·s` is exactly HALF the true loss's slope.
    /// Pinning the factor keeps the regression legible if the above ever fails.
    #[test]
    fn huber_surrogate_moves_at_half_the_true_rate() {
        let scale = 1.0_f32;
        for &r in &[1.5_f32, 3.0, 10.0] {
            let s = r * r;
            let h = 1e-4_f32 * s;
            let surrogate = |s: f32| 0.5 * robust_weight(RobustKernelKind::Huber, scale, s) * s;
            let d_surr = (surrogate(s + h) - surrogate(s - h)) / (2.0 * h);
            let d_true = (robust_cost(RobustKernelKind::Huber, scale, s + h)
                - robust_cost(RobustKernelKind::Huber, scale, s - h))
                / (2.0 * h);
            let ratio = d_surr / d_true;
            assert!(
                (ratio - 0.5).abs() < 1e-2,
                "r={r}: surrogate/true slope = {ratio}, expected 0.5"
            );
        }
    }

    /// THE regression test for this change: the cost the solver reports must be the objective it
    /// claims to minimise, evaluated independently at the poses and points it returned.
    ///
    /// Every other test here exercises `robust_weight`/`robust_cost` in isolation, so all of them
    /// stay green if the SOLVER goes back to accumulating the IRLS surrogate `½ρ'(s)·s` at its
    /// `cost +=` / `new_cost +=` sites — which is exactly what the bug was. This one recomputes
    /// `Σ ½ρ(‖r‖²)` from the returned solution with no help from the solver and compares.
    ///
    /// It needs residuals PAST the Huber knee at the optimum to bite, because inside the knee the
    /// surrogate and the loss agree identically. Hence the deliberate gross outlier: it cannot be
    /// fitted, so it still sits far past the knee when the solve finishes, where the surrogate is
    /// exactly half the true loss.
    ///
    /// It runs a SECOND time with `refine_focal` and `refine_k1` on, where it becomes the strongest
    /// single assertion available about the stale-intrinsics hazard: the caller re-projects at the
    /// RETURNED camera, so a trial pass evaluated at stale intrinsics reports a `final_cost`
    /// computed under a model the returned camera is not, and the two disagree. It catches a wrong
    /// `BaResult::camera` in the same breath.
    #[test]
    fn reported_cost_is_the_objective_the_solver_minimises() {
        for (refine_focal, refine_k1) in [(false, false), (true, true)] {
            let (poses, points, mut observations, camera) = perturbed_two_view_problem();

            // A gross outlier, unfittable by construction, so it stays saturated at the solution.
            let mut outlier = observations[0];
            outlier.pixel = [
                observations[0].pixel[0] + 5.0,
                observations[0].pixel[1] - 4.0,
            ];
            observations.push(outlier);

            let scale_sq = 0.01_f32;
            let params = BaParams {
                max_iterations: 25,
                robust: RobustKernelKind::Huber,
                robust_scale_sq: scale_sq,
                refine_focal,
                refine_k1,
                ..Default::default()
            };
            let res =
                bundle_adjust_schur(&poses, &points, &observations, &camera, &params).unwrap();

            let free = refine_focal || refine_k1;
            assert_eq!(res.camera.is_some(), free, "BaResult::camera presence");
            // Re-projecting at the RETURNED camera is the point of the second case.
            let fitted = res.camera.clone().unwrap_or_else(|| camera.clone());
            let intr = BaIntrinsics::seed(&fitted, refine_k1);

            // Independent evaluation of Σ ½ρ(s) at the returned solution.
            let scale = scale_sq.sqrt();
            let se3s: Vec<SE3F32> = res.poses.iter().map(pose_to_se3).collect();
            let mut expected = 0.0_f32;
            let mut saturated = 0usize;
            for obs in &observations {
                let (r, _, _, _) = residual_and_jacobians(
                    &se3s[obs.pose_idx],
                    &res.points[obs.point_idx],
                    obs.pixel,
                    &intr,
                );
                let r_sq = r[0] * r[0] + r[1] * r[1];
                if r_sq > scale * scale {
                    saturated += 1;
                }
                expected += robust_cost(RobustKernelKind::Huber, scale, r_sq);
            }

            assert!(
                saturated > 0,
                "fixture no longer saturates the Huber knee, so it cannot distinguish the \
                 surrogate (refine_focal={refine_focal})"
            );
            assert!(
                (res.final_cost - expected).abs() <= 1e-4 * expected.max(1e-6),
                "refine_focal={refine_focal} refine_k1={refine_k1}: solver reported {} but the \
                 objective at its own returned poses, points AND camera is {expected}",
                res.final_cost
            );
        }
    }

    /// The counters must actually count.
    ///
    /// NOTE what this test can and cannot assert, because it is the clearest available evidence
    /// about the design. The counters are process-global with no reset, and the harness runs
    /// tests in parallel, so any other test calling a bundle adjustment concurrently perturbs
    /// every delta measured here. An exact `BA_ITERS` delta or an exact `BA_OBS` value passes in
    /// isolation and fails in the full suite — verified. So this asserts only monotonic advance,
    /// which is all that is well-defined for shared global counters under a parallel runner.
    ///
    /// If these become `BaResult` fields or move behind a feature gate, this test can assert the
    /// exact values instead, and `BA_OBS`'s last-writer-wins semantics stop being observable.
    #[test]
    fn counters_record_the_solve_they_ran() {
        let (poses, points, observations, camera) = perturbed_two_view_problem();
        assert!(!observations.is_empty());
        let params = BaParams {
            max_iterations: 5,
            ..Default::default()
        };

        let calls_before = BA_CALLS.load(Ordering::Relaxed);
        let iters_before = BA_ITERS.load(Ordering::Relaxed);
        let nanos_before = BA_NANOS.load(Ordering::Relaxed);
        let lin_before = BA_LIN_NANOS.load(Ordering::Relaxed);

        let res = bundle_adjust_schur(&poses, &points, &observations, &camera, &params).unwrap();

        assert!(
            BA_CALLS.load(Ordering::Relaxed) > calls_before,
            "BA_CALLS never advanced"
        );
        assert!(
            BA_ITERS.load(Ordering::Relaxed) - iters_before >= res.iterations,
            "BA_ITERS advanced by less than the iterations this call reports"
        );
        assert!(
            BA_NANOS.load(Ordering::Relaxed) > nanos_before,
            "BA_NANOS never advanced"
        );
        assert!(
            BA_LIN_NANOS.load(Ordering::Relaxed) > lin_before,
            "BA_LIN_NANOS never advanced — the linearisation phase is not being timed"
        );
        // Non-zero rather than exact: a concurrent test's solve may have stored over it.
        assert!(
            BA_OBS.load(Ordering::Relaxed) > 0,
            "BA_OBS was never written"
        );
    }

    /// The other half of the pair. `robust_cost_is_half_the_shared_kornia_algebra_rho` pins the
    /// COST against kornia-algebra; nothing pinned the WEIGHT, and weight-vs-cost drift is the
    /// exact failure this whole change exists to fix — so pinning only one of the two closes half
    /// the hole.
    ///
    /// It also catches a subtler divergence: this module tests Huber's knee as `‖r‖ <= scale`
    /// while `HuberLoss::weight` tests `s <= scale²`. In f32 `sqrt` and squaring round
    /// differently, so those two disagree for `s` within an ulp of `scale²` — the same
    /// observation weighted `1.0` by one solver and `scale/‖r‖` by the other.
    #[test]
    fn robust_weight_matches_the_shared_kornia_algebra_weight() {
        use kornia_algebra::optim::losses::{CauchyLoss, HuberLoss, IdentityLoss, RobustLoss};
        let scale = 1.5_f32;
        let huber = HuberLoss::new(scale).unwrap();
        let cauchy = CauchyLoss::new(scale).unwrap();
        // Straddle the knee tightly: consecutive f32 values either side of scale².
        let knee = scale * scale;
        let radii = [
            1e-4_f32,
            0.1,
            0.9,
            f32::from_bits(knee.to_bits() - 1).sqrt(),
            scale,
            f32::from_bits(knee.to_bits() + 1).sqrt(),
            3.0,
            10.0,
        ];
        for &r in &radii {
            let s = r * r;
            for (kind, theirs) in [
                (RobustKernelKind::Identity, IdentityLoss.weight(s)),
                (RobustKernelKind::Huber, huber.weight(s)),
                (RobustKernelKind::Cauchy, cauchy.weight(s)),
            ] {
                let ours = robust_weight(kind, scale, s);
                assert!(
                    (ours - theirs).abs() <= 1e-4 * theirs.abs().max(f32::MIN_POSITIVE),
                    "{kind:?} at r={r}: ba_schur weight {ours} != kornia-algebra weight {theirs}"
                );
            }
        }
    }

    /// `robust_cost` must stay `½·RobustLoss::rho` from `kornia-algebra` — the shared,
    /// already-tested loss that `ba::bundle_adjust` routes the SAME `BaParams` through.
    /// Duplicating the algebra here is what let weight and cost diverge in the first place; this
    /// pins the copy to the original so the two solvers cannot drift apart silently.
    #[test]
    fn robust_cost_is_half_the_shared_kornia_algebra_rho() {
        use kornia_algebra::optim::losses::{CauchyLoss, HuberLoss, IdentityLoss, RobustLoss};
        let scale = 1.5_f32;
        let huber = HuberLoss::new(scale).unwrap();
        let cauchy = CauchyLoss::new(scale).unwrap();
        // The small radii matter as much as the large ones. An earlier revision stopped at 0.1,
        // and that was the only reason this test and `cauchy_cost_is_accurate_for_small_residuals`
        // could coexist: `CauchyLoss::rho` used `(1.0 + x).ln()` and was 28% low at r=1e-3, so
        // pinning one to the other over that range would have failed. Both use `ln_1p` now, so the
        // pin holds where it actually matters.
        for &r in &[1e-4_f32, 1e-3, 1e-2, 0.1, 0.9, 1.4, 1.5, 1.6, 3.0, 10.0] {
            let s = r * r;
            for (kind, rho) in [
                (RobustKernelKind::Identity, IdentityLoss.rho(s)),
                (RobustKernelKind::Huber, huber.rho(s)),
                (RobustKernelKind::Cauchy, cauchy.rho(s)),
            ] {
                let ours = robust_cost(kind, scale, s);
                // Relative only — an absolute floor would let the small-r cases pass vacuously,
                // which is exactly how the earlier revision hid the divergence.
                assert!(
                    (ours - 0.5 * rho).abs() <= 1e-4 * (0.5 * rho).abs().max(f32::MIN_POSITIVE),
                    "{kind:?} at r={r}: ba_schur {ours} != 0.5 * kornia-algebra rho {rho}"
                );
            }
        }
    }

    /// The Cauchy loss must stay accurate for SMALL residuals — the converged regime, which is
    /// exactly where the accept test has to resolve a difference. `(1.0 + x).ln()` in f32
    /// quantises `x` to multiples of ~1.19e-7 before the log and returns identically 0.0 once
    /// `x < 6e-8`; `ln_1p` does not. Reverting to `(1.0 + r_sq / s2).ln()` fails here.
    #[test]
    fn cauchy_cost_is_accurate_for_small_residuals() {
        for &scale in &[1.0_f32, 2.45, 1000.0] {
            for &r in &[1e-4_f32, 1e-3, 1e-2, 0.1] {
                let s = r * r;
                // For s << scale², ½ρ(s) → ½s.
                let ours = robust_cost(RobustKernelKind::Cauchy, scale, s);
                let quadratic = 0.5 * s;
                assert!(
                    ours > 0.0,
                    "scale={scale} r={r}: Cauchy cost flushed to zero"
                );
                assert!(
                    (ours - quadratic).abs() <= 1e-2 * quadratic,
                    "scale={scale} r={r}: Cauchy cost {ours} deviates from the quadratic limit \
                     {quadratic} by more than 1%"
                );
            }
        }
    }

    /// Two cameras (the first fixed) seeing four points, with the points perturbed off ground
    /// truth so there is a real step to take. Shared by the L2-collapse test below.
    fn perturbed_two_view_problem() -> (
        Vec<Pose3d>,
        Vec<Vec3F64>,
        Vec<BaObservation>,
        crate::camera::PinholeCamera,
    ) {
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
            [
                (cam.fx * pc.x / pc.z + cam.cx) as f32,
                (cam.fy * pc.y / pc.z + cam.cy) as f32,
            ]
        };
        let mut obs = Vec::new();
        for (pi, pt) in true_points.iter().enumerate() {
            obs.push(BaObservation {
                pose_idx: 0,
                point_idx: pi,
                pixel: project(&pose0, pt),
                fixed_pose: true,
                fixed_point: false,
                ..BaObservation::default()
            });
            obs.push(BaObservation {
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
        (vec![pose0, pose1], perturbed, obs, cam)
    }

    /// `BaParams::robust_scale_sq` documents that its `f32::INFINITY` default "collapses to the
    /// L2 fast path even for non-Identity kernel choices", and `ba::build_robust_loss` enforces
    /// that for the non-Schur solver. Before the guard, `Cauchy` at the default scale made every
    /// cost NaN, so `new_cost < cost` was always false, every step was rejected, and the solver
    /// returned its INPUT poses with `Ok` and no error.
    #[test]
    fn infinite_robust_scale_collapses_to_l2_instead_of_nan() {
        // Unit-level: the kernels themselves must not produce NaN at the default scale.
        for kind in [RobustKernelKind::Cauchy, RobustKernelKind::Tukey] {
            let w = robust_weight(kind, f32::INFINITY, 4.0);
            let c = robust_cost(kind, f32::INFINITY, 4.0);
            assert!(w.is_nan() && c.is_nan(), "precondition for the guard below");
        }

        // Solver-level: the guard must make Cauchy-with-default-scale behave exactly like L2.
        let (poses, points, obs, cam) = perturbed_two_view_problem();
        let l2 = bundle_adjust_schur(
            &poses,
            &points,
            &obs,
            &cam,
            &BaParams {
                max_iterations: 20,
                ..Default::default()
            },
        )
        .unwrap();
        let cauchy_default_scale = bundle_adjust_schur(
            &poses,
            &points,
            &obs,
            &cam,
            &BaParams {
                max_iterations: 20,
                robust: RobustKernelKind::Cauchy,
                // robust_scale_sq left at its f32::INFINITY default.
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(
            cauchy_default_scale.iterations, l2.iterations,
            "Cauchy at the default (infinite) scale must run the identical L2 solve"
        );
        assert_eq!(cauchy_default_scale.converged, l2.converged);
        for (a, b) in cauchy_default_scale.poses.iter().zip(l2.poses.iter()) {
            assert!(
                (a.translation - b.translation).length() < 1e-9,
                "Cauchy at the default scale diverged from the L2 solve"
            );
        }
        // And it must actually have moved — a NaN cost would return the input untouched.
        let moved = cauchy_default_scale
            .poses
            .iter()
            .zip(poses.iter())
            .any(|(a, b)| (a.translation - b.translation).length() > 1e-6);
        assert!(moved, "solver returned its input poses unchanged");
    }

    use super::*;
    use crate::camera::PinholeCamera;
    use kornia_algebra::Mat3F64;

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

    #[test]
    fn schur_ba_recovers_perturbed_poses() {
        let cam = test_camera();
        // Two-camera, four-point setup like ba's existing test, but solve via Schur.
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
        let result = bundle_adjust_schur(
            &[pose0, pose1],
            &perturbed,
            &observations,
            &cam,
            &BaParams {
                max_iterations: 30,
                ..BaParams::default()
            },
        )
        .unwrap();
        // The 4-points / 2-poses (1 fixed) problem has 18 unknowns vs 16
        // residuals — gauge ambiguity gives a 2-dim cost-zero manifold.
        // BA reaches cost=0 (verified by tracing) but lands on a different
        // point in that manifold depending on the solver's null-space
        // navigation. We assert geometric closeness within 0.2 m (about the
        // expected radius of the gauge ambiguity for this configuration).
        for (i, refined) in result.points.iter().enumerate() {
            let err = (*refined - true_points[i]).length();
            assert!(err < 0.2, "point {i} error {err} too large");
        }
    }

    /// Depth-anchored BA recovers absolute metric scale.
    ///
    /// Setup:
    ///   * 5 poses on a half-circle at radius 4 m looking inward at origin.
    ///   * 50 known 3D points scattered in a box around the origin.
    ///   * Project to pixels with σ=0.3 px Gaussian noise.
    ///   * Synthetic depth measurement per observation, σ=2% of true depth.
    ///   * INIT the BA with points scaled 2× from ground truth — without depth
    ///     residuals, this drift would be unobservable (gauge ambiguity).
    ///   * With depth_meas set, the BA should recover GT scale.
    fn translate_pose(t: Vec3F64) -> Pose3d {
        // Camera at position `cam_pos = t` looking down +Z (identity rotation
        // in world frame). Then R_w_c = I, t_w_c = cam_pos, and the
        // world→camera pose stored in Pose3d is the *inverse*:
        //   R_cw = I, t_cw = -cam_pos.
        Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-t.x, -t.y, -t.z))
    }

    #[test]
    fn schur_ba_with_depth_recovers_scale() {
        // Reproducible PRNG via std (no rand crate dep here).
        // Simple LCG for noise sampling.
        struct Lcg {
            state: u64,
        }
        impl Lcg {
            fn new(seed: u64) -> Self {
                Self { state: seed }
            }
            fn next_u64(&mut self) -> u64 {
                self.state = self
                    .state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.state
            }
            // Box–Muller standard normal (uses two uniforms).
            fn normal(&mut self) -> f64 {
                let u1 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u2 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u1 = u1.max(1e-12);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            }
        }
        let mut rng = Lcg::new(0x00C0_FFEE_DEAD_BEEF_u64);

        let cam = PinholeCamera {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        // 5 cameras translating along +X by 0, 0.1, 0.2, 0.3, 0.4 m
        // (typical small forward/sideways baseline in SLAM). All cameras
        // look down +Z (identity rotation). Camera 0 sits at the origin →
        // its pose is identity in both world and inverse-world frames,
        // so a global similarity that fixes the origin leaves it
        // unchanged. This makes (poses, points) up to scale invisible
        // to reprojection cost when pose 0 is fixed.
        let cam_positions = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(0.1, 0.0, 0.0),
            Vec3F64::new(0.2, 0.0, 0.0),
            Vec3F64::new(0.3, 0.0, 0.0),
            Vec3F64::new(0.4, 0.0, 0.0),
        ];
        let true_poses: Vec<Pose3d> = cam_positions.iter().map(|&p| translate_pose(p)).collect();

        // 50 well-distributed 3D points in front of the cameras, 3-6 m deep.
        let mut true_points: Vec<Vec3F64> = Vec::with_capacity(50);
        for k in 0..50 {
            let kf = k as f64;
            let x = (kf * 0.37).sin() * 1.2 + (kf * 0.13).cos() * 0.4;
            let y = (kf * 0.29).cos() * 0.9 + (kf * 0.11).sin() * 0.3;
            let z = 4.0 + (kf * 0.41).sin() * 1.5; // 2.5..5.5m in front
            true_points.push(Vec3F64::new(x, y, z));
        }

        // Build observations: pixels + depth, both noisy. Skip points behind
        // the camera (negative z) — they can happen for the wider angles.
        let mut observations: Vec<BaObservation> = Vec::new();
        for (pi, pose) in true_poses.iter().enumerate() {
            for (xi, pt) in true_points.iter().enumerate() {
                let pc = pose.transform_point(pt);
                if pc.z <= 0.2 {
                    continue;
                }
                let u = cam.fx * pc.x / pc.z + cam.cx + 0.3 * rng.normal();
                let v = cam.fy * pc.y / pc.z + cam.cy + 0.3 * rng.normal();
                // Depth noise: σ = 2% of true depth.
                let depth_sigma_m = 0.02 * pc.z as f32;
                let d_meas = (pc.z + 0.02 * pc.z * rng.normal()) as f32;
                observations.push(BaObservation {
                    pose_idx: pi,
                    point_idx: xi,
                    pixel: [u as f32, v as f32],
                    fixed_pose: pi == 0, // anchor pose 0
                    fixed_point: false,
                    depth_meas: Some(d_meas),
                    depth_sigma: depth_sigma_m,
                });
            }
        }

        // Initial guess: simulate a 2× global scale drift. Pose 0 stays at
        // identity (origin is similarity-invariant). Poses 1..N get their
        // translation scaled by 2× (i.e. cam baselines are 2× too long).
        // Points are also scaled 2× → the reprojection residual at this
        // init is *exactly zero* because (s·R · s·X + s·t)/(s·Z) = same
        // pixel. Only the depth residual can break this gauge.
        let init_poses: Vec<Pose3d> = true_poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i == 0 {
                    *p
                } else {
                    // Scale translation by 2× (rotation is identity, scale-invariant).
                    Pose3d::new(
                        p.rotation,
                        Vec3F64::new(
                            p.translation.x * 2.0,
                            p.translation.y * 2.0,
                            p.translation.z * 2.0,
                        ),
                    )
                }
            })
            .collect();
        let init_points: Vec<Vec3F64> = true_points
            .iter()
            .map(|p| Vec3F64::new(p.x * 2.0, p.y * 2.0, p.z * 2.0))
            .collect();

        let params = BaParams {
            max_iterations: 100,
            cost_tolerance: 1e-8,
            ..BaParams::default()
        };
        let result =
            bundle_adjust_schur(&init_poses, &init_points, &observations, &cam, &params).unwrap();

        // Assert geometric recovery.
        let mut max_pt_err: f64 = 0.0;
        let mut mean_pt_err: f64 = 0.0;
        for (i, refined) in result.points.iter().enumerate() {
            let err = (*refined - true_points[i]).length();
            if err > max_pt_err {
                max_pt_err = err;
            }
            mean_pt_err += err;
        }
        mean_pt_err /= result.points.len() as f64;

        // Sanity baseline: run the same BA WITHOUT depth, confirm it drifts
        // (failure to converge to GT scale is the whole point of this test).
        let no_depth_obs: Vec<BaObservation> = observations
            .iter()
            .map(|o| BaObservation {
                pose_idx: o.pose_idx,
                point_idx: o.point_idx,
                pixel: o.pixel,
                fixed_pose: o.fixed_pose,
                fixed_point: o.fixed_point,
                depth_meas: None,
                depth_sigma: 1.0,
            })
            .collect();
        let no_depth_result =
            bundle_adjust_schur(&init_poses, &init_points, &no_depth_obs, &cam, &params).unwrap();
        let mut max_pt_err_no_depth: f64 = 0.0;
        for (i, refined) in no_depth_result.points.iter().enumerate() {
            let err = (*refined - true_points[i]).length();
            if err > max_pt_err_no_depth {
                max_pt_err_no_depth = err;
            }
        }

        // 5cm GT recovery target per spec. Allow some slack for noise.
        assert!(
            max_pt_err < 0.10,
            "max point error {max_pt_err:.4} m (mean {mean_pt_err:.4}) too large \
             — depth anchor not working. (Without depth: {max_pt_err_no_depth:.4} m.)"
        );
        // Sanity: depth should beat no-depth by a wide margin.
        assert!(
            max_pt_err < 0.5 * max_pt_err_no_depth,
            "depth BA ({max_pt_err:.4}) did not significantly beat no-depth \
             ({max_pt_err_no_depth:.4}) — anchor likely inert"
        );

        // Pose error: pose 0 is anchored, so we measure the other four.
        let mut max_t_err: f64 = 0.0;
        let mut max_rot_err: f64 = 0.0;
        for (i, refined) in result.poses.iter().enumerate() {
            if i == 0 {
                continue;
            }
            let dt = refined.translation - true_poses[i].translation;
            let t_err = dt.length();
            if t_err > max_t_err {
                max_t_err = t_err;
            }

            // Rotation error (Frobenius angle): R_err = R_ref.T · R_refined
            let r_err = true_poses[i].rotation.transpose() * refined.rotation;
            // angle = acos((trace - 1) / 2), clamped.
            let trace = r_err.col(0).x + r_err.col(1).y + r_err.col(2).z;
            let cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
            let angle_rad = cos_angle.acos();
            if angle_rad > max_rot_err {
                max_rot_err = angle_rad;
            }
        }
        let max_rot_deg = max_rot_err.to_degrees();
        assert!(
            max_t_err < 0.05,
            "max translation error {max_t_err:.4} m too large"
        );
        assert!(
            max_rot_deg < 1.0,
            "max rotation error {max_rot_deg:.4}° too large"
        );
    }

    /// Pose-prior BA recovers lateral translation that pose-only BA cannot.
    ///
    /// Setup mirrors `schur_ba_with_depth_recovers_scale` but the
    /// perturbation is *lateral* (along X / Y), which the depth residual
    /// cannot constrain — depth only sees Z in cam frame, and for cameras
    /// looking down +Z, lateral world-frame translation of the rig is
    /// orthogonal to the cam-frame depth axis. The pose prior is the right
    /// tool: it constrains all three world-frame axes of every pose
    /// translation directly.
    #[test]
    fn schur_ba_with_pose_prior_recovers_lateral() {
        // Same LCG as the depth test.
        struct Lcg {
            state: u64,
        }
        impl Lcg {
            fn new(seed: u64) -> Self {
                Self { state: seed }
            }
            fn next_u64(&mut self) -> u64 {
                self.state = self
                    .state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.state
            }
            fn normal(&mut self) -> f64 {
                let u1 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u2 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u1 = u1.max(1e-12);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            }
        }
        let mut rng = Lcg::new(0x0000_BADC_AFE1_2345_u64);

        let cam = PinholeCamera {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        // 5 cameras moving forward (along +Z is the look-axis; the rig moves
        // along +X in world). All cameras have identity rotation (looking
        // down +Z). Pose 0 at origin acts as the gauge.
        let cam_positions = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(0.1, 0.0, 0.0),
            Vec3F64::new(0.2, 0.0, 0.0),
            Vec3F64::new(0.3, 0.0, 0.0),
            Vec3F64::new(0.4, 0.0, 0.0),
        ];
        // Pose stores world→cam, so t_cw = -C_world for R=I.
        let true_poses: Vec<Pose3d> = cam_positions
            .iter()
            .map(|p| Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-p.x, -p.y, -p.z)))
            .collect();

        // 50 points in front of the cameras.
        let mut true_points: Vec<Vec3F64> = Vec::with_capacity(50);
        for k in 0..50 {
            let kf = k as f64;
            let x = (kf * 0.37).sin() * 1.2 + (kf * 0.13).cos() * 0.4;
            let y = (kf * 0.29).cos() * 0.9 + (kf * 0.11).sin() * 0.3;
            let z = 4.0 + (kf * 0.41).sin() * 1.5;
            true_points.push(Vec3F64::new(x, y, z));
        }

        // Project pixels with σ=0.3 px noise.
        // NB: no pose is `fixed_pose` here — we let the entire rig float so
        // the lateral world-frame translation of all-poses-plus-points is a
        // genuine gauge mode that pure reprojection BA cannot resolve.
        let mut observations: Vec<BaObservation> = Vec::new();
        for (pi, pose) in true_poses.iter().enumerate() {
            for (xi, pt) in true_points.iter().enumerate() {
                let pc = pose.transform_point(pt);
                if pc.z <= 0.2 {
                    continue;
                }
                let u = cam.fx * pc.x / pc.z + cam.cx + 0.3 * rng.normal();
                let v = cam.fy * pc.y / pc.z + cam.cy + 0.3 * rng.normal();
                observations.push(BaObservation {
                    pose_idx: pi,
                    point_idx: xi,
                    pixel: [u as f32, v as f32],
                    fixed_pose: false,
                    fixed_point: false,
                    depth_meas: None,
                    depth_sigma: 1.0,
                });
            }
        }

        // ── Initial guess: translate the ENTIRE rig (all poses + all
        // points) laterally by +0.5 m in world Y. This is a global SE(3)
        // gauge mode: it preserves every reprojection residual exactly to
        // pixel precision, so pure reprojection BA cannot pull the system
        // back. Only the pose prior breaks the gauge.
        let lateral_offset = Vec3F64::new(0.0, 0.5, 0.0);
        let init_poses: Vec<Pose3d> = true_poses
            .iter()
            .map(|p| {
                // C_new = C + offset. Since C = -R^T·t and R=I, that means
                // t_new = t - offset (R is identity in this setup).
                Pose3d::new(
                    p.rotation,
                    Vec3F64::new(
                        p.translation.x - lateral_offset.x,
                        p.translation.y - lateral_offset.y,
                        p.translation.z - lateral_offset.z,
                    ),
                )
            })
            .collect();
        let init_points: Vec<Vec3F64> = true_points
            .iter()
            .map(|pt| {
                Vec3F64::new(
                    pt.x + lateral_offset.x,
                    pt.y + lateral_offset.y,
                    pt.z + lateral_offset.z,
                )
            })
            .collect();

        // ── Pose priors: tight (σ=0.05 m) at GT camera centres for free
        // poses (pose 0 is fixed so its entry is moot but we still set it
        // for completeness).
        let priors: Vec<Option<BaPosePrior>> = true_poses
            .iter()
            .map(|p| {
                // GT camera centre.
                let r_t = p.rotation.transpose();
                let c = -(r_t * p.translation);
                Some(BaPosePrior::new([c.x as f32, c.y as f32, c.z as f32], 0.05))
            })
            .collect();

        let params = BaParams {
            max_iterations: 100,
            cost_tolerance: 1e-9,
            ..BaParams::default()
        };
        let result = bundle_adjust_schur_with_priors(
            &init_poses,
            &init_points,
            &observations,
            &cam,
            &params,
            Some(&priors),
        )
        .unwrap();

        // ── Without prior (control): pose-only reprojection BA at this
        // perturbation has nothing pulling the camera laterally back to GT
        // because shifting cameras + scaling points (or just dragging the
        // whole rig) gives almost-zero residual. We do *not* assert the
        // control fails — only that the priored result succeeds.
        let mut max_t_err: f64 = 0.0;
        let mut max_t_err_lateral: f64 = 0.0;
        for (i, refined) in result.poses.iter().enumerate() {
            let dt = refined.translation - true_poses[i].translation;
            let t_err = dt.length();
            if t_err > max_t_err {
                max_t_err = t_err;
            }
            let lat = (dt.x * dt.x + dt.y * dt.y).sqrt();
            if lat > max_t_err_lateral {
                max_t_err_lateral = lat;
            }
            let _ = i;
        }
        eprintln!(
            "pose-prior BA: max_t_err={:.4} m, max_lateral={:.4} m, converged={}",
            max_t_err, max_t_err_lateral, result.converged,
        );

        // Recovered pose centres within 2 cm of GT in ALL 3 axes.
        // Pose prior at σ=0.05 anchors strongly; this is well within the
        // posterior radius for 5 cameras + 50 well-spread points.
        for (i, refined) in result.poses.iter().enumerate() {
            let r_t = refined.rotation.transpose();
            let c_ref = -(r_t * refined.translation);
            let r_t_gt = true_poses[i].rotation.transpose();
            let c_gt = -(r_t_gt * true_poses[i].translation);
            let dc = c_ref - c_gt;
            assert!(
                dc.x.abs() < 0.02 && dc.y.abs() < 0.02 && dc.z.abs() < 0.02,
                "pose {i} centre off GT: dC=({:.4}, {:.4}, {:.4}) m",
                dc.x,
                dc.y,
                dc.z,
            );
        }

        // Sanity: passing `None` for priors at this init should fail the
        // lateral test (drift not pulled back). Run it and check we don't
        // get within 2cm in Y — proves the prior is doing the work.
        let no_prior =
            bundle_adjust_schur(&init_poses, &init_points, &observations, &cam, &params).unwrap();
        let mut max_dy_no_prior: f64 = 0.0;
        for (i, refined) in no_prior.poses.iter().enumerate() {
            let r_t = refined.rotation.transpose();
            let c_ref = -(r_t * refined.translation);
            let r_t_gt = true_poses[i].rotation.transpose();
            let c_gt = -(r_t_gt * true_poses[i].translation);
            let dy = (c_ref.y - c_gt.y).abs();
            if dy > max_dy_no_prior {
                max_dy_no_prior = dy;
            }
            let _ = i;
        }
        eprintln!("no-prior control: max |dy| = {:.4} m", max_dy_no_prior);
        // The prior must beat no-prior on the lateral axis decisively.
        assert!(
            max_dy_no_prior > 0.05,
            "no-prior control happened to recover (max |dy|={:.4}) — test is \
             not exercising the lateral-drift mode it's meant to",
            max_dy_no_prior,
        );
    }

    /// Huber on the depth residual rejects a single outlier depth measurement
    /// (e.g. an object-boundary mis-sample at 10× σ). Without Huber the
    /// outlier pulls the reconstruction off GT; with Huber it's downweighted
    /// and the reconstruction stays accurate.
    ///
    /// Setup mirrors `schur_ba_with_depth_recovers_scale` but with a single
    /// corrupted depth measurement: `d_meas = true_depth * 1.5` on one obs.
    #[test]
    fn schur_ba_huber_rejects_depth_outlier() {
        struct Lcg {
            state: u64,
        }
        impl Lcg {
            fn new(seed: u64) -> Self {
                Self { state: seed }
            }
            fn next_u64(&mut self) -> u64 {
                self.state = self
                    .state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.state
            }
            fn normal(&mut self) -> f64 {
                let u1 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u2 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u1 = u1.max(1e-12);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            }
        }
        let mut rng = Lcg::new(0x000F_ACEF_00DC_0DE0_u64);

        let cam = PinholeCamera {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        // 5 cameras along +X, looking down +Z (identity rotation).
        let cam_positions = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(0.1, 0.0, 0.0),
            Vec3F64::new(0.2, 0.0, 0.0),
            Vec3F64::new(0.3, 0.0, 0.0),
            Vec3F64::new(0.4, 0.0, 0.0),
        ];
        let true_poses: Vec<Pose3d> = cam_positions
            .iter()
            .map(|p| Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-p.x, -p.y, -p.z)))
            .collect();

        // 50 points in front of the cameras.
        let mut true_points: Vec<Vec3F64> = Vec::with_capacity(50);
        for k in 0..50 {
            let kf = k as f64;
            let x = (kf * 0.37).sin() * 1.2 + (kf * 0.13).cos() * 0.4;
            let y = (kf * 0.29).cos() * 0.9 + (kf * 0.11).sin() * 0.3;
            let z = 4.0 + (kf * 0.41).sin() * 1.5;
            true_points.push(Vec3F64::new(x, y, z));
        }

        // Build observations with σ=0.3 px reproj noise and σ=2% depth noise.
        let mut observations: Vec<BaObservation> = Vec::new();
        for (pi, pose) in true_poses.iter().enumerate() {
            for (xi, pt) in true_points.iter().enumerate() {
                let pc = pose.transform_point(pt);
                if pc.z <= 0.2 {
                    continue;
                }
                let u = cam.fx * pc.x / pc.z + cam.cx + 0.3 * rng.normal();
                let v = cam.fy * pc.y / pc.z + cam.cy + 0.3 * rng.normal();
                let depth_sigma_m = 0.02 * pc.z as f32;
                let d_meas = (pc.z + 0.02 * pc.z * rng.normal()) as f32;
                observations.push(BaObservation {
                    pose_idx: pi,
                    point_idx: xi,
                    pixel: [u as f32, v as f32],
                    fixed_pose: pi == 0, // anchor pose 0
                    fixed_point: false,
                    depth_meas: Some(d_meas),
                    depth_sigma: depth_sigma_m,
                });
            }
        }

        // Inject ONE bad depth measurement at 50% inflation
        // (= true_depth * 1.5, far beyond σ=2%; the gate threshold √χ²(3,99%)
        // ≈ 2.8 σ in whitened units → 1.5/0.02 = 25 σ is a clear outlier).
        // Pick an observation that does NOT touch the anchored pose 0 so the
        // outlier can actually pull free variables. Choose pose 2 and the
        // first point it sees.
        let outlier_obs_idx = observations
            .iter()
            .position(|o| o.pose_idx == 2 && o.depth_meas.is_some())
            .expect("expected at least one depth obs on pose 2");
        let outlier_pt_idx = observations[outlier_obs_idx].point_idx;
        let pose2 = &true_poses[2];
        let true_z = pose2.transform_point(&true_points[outlier_pt_idx]).z as f32;
        observations[outlier_obs_idx].depth_meas = Some(true_z * 1.5);

        // 2× scale-drift init (same gauge-breaking setup as the scale test).
        let init_poses: Vec<Pose3d> = true_poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i == 0 {
                    *p
                } else {
                    Pose3d::new(
                        p.rotation,
                        Vec3F64::new(
                            p.translation.x * 2.0,
                            p.translation.y * 2.0,
                            p.translation.z * 2.0,
                        ),
                    )
                }
            })
            .collect();
        let init_points: Vec<Vec3F64> = true_points
            .iter()
            .map(|p| Vec3F64::new(p.x * 2.0, p.y * 2.0, p.z * 2.0))
            .collect();

        // ── Run 1: BA without robust kernel. The outlier dominates and
        // drags the affected point off GT.
        let params_no_huber = BaParams {
            max_iterations: 100,
            cost_tolerance: 1e-8,
            robust: RobustKernelKind::Identity,
            robust_scale_sq: f32::INFINITY,
            ..BaParams::default()
        };
        let result_no_huber = bundle_adjust_schur(
            &init_poses,
            &init_points,
            &observations,
            &cam,
            &params_no_huber,
        )
        .unwrap();

        // ── Run 2: BA WITH Huber. The outlier is downweighted; reconstruction
        // remains accurate.
        // ORB-SLAM3 §IV.B uses χ²=5.99 for 2-DoF reproj; we use the same
        // robust_scale_sq for the depth residual (whitened scalar, so the
        // gate triggers above ~√5.99 ≈ 2.45 σ).
        let params_huber = BaParams {
            max_iterations: 100,
            cost_tolerance: 1e-8,
            robust: RobustKernelKind::Huber,
            robust_scale_sq: 5.99,
            ..BaParams::default()
        };
        let result_huber = bundle_adjust_schur(
            &init_poses,
            &init_points,
            &observations,
            &cam,
            &params_huber,
        )
        .unwrap();

        // Compute max point error for both runs.
        let mut max_err_no_huber: f64 = 0.0;
        let mut outlier_err_no_huber: f64 = 0.0;
        for (i, refined) in result_no_huber.points.iter().enumerate() {
            let err = (*refined - true_points[i]).length();
            if err > max_err_no_huber {
                max_err_no_huber = err;
            }
            if i == outlier_pt_idx {
                outlier_err_no_huber = err;
            }
        }
        let mut max_err_huber: f64 = 0.0;
        let mut outlier_err_huber: f64 = 0.0;
        for (i, refined) in result_huber.points.iter().enumerate() {
            let err = (*refined - true_points[i]).length();
            if err > max_err_huber {
                max_err_huber = err;
            }
            if i == outlier_pt_idx {
                outlier_err_huber = err;
            }
        }
        eprintln!(
            "depth-outlier test: max_err no_huber={:.4} m (outlier pt {:.4}), \
             with_huber={:.4} m (outlier pt {:.4})",
            max_err_no_huber, outlier_err_no_huber, max_err_huber, outlier_err_huber,
        );

        // Without Huber: the 1.5× outlier perturbs the affected point by
        // roughly (1.5 - 1) × depth = 0.5 × ~4 m = 2 m of mismatch, dampened
        // by other obs to a smaller value but well above 10 cm.
        assert!(
            outlier_err_no_huber > 0.10,
            "expected outlier-affected point to drift >10 cm without Huber, got {:.4} m",
            outlier_err_no_huber,
        );
        // With Huber: outlier is downweighted, reconstruction is much better.
        // Huber caps the GRADIENT contribution at the scale parameter (it does
        // not zero it out — that would require a redescending kernel like
        // Cauchy/Tukey). With robust_scale_sq=5.99 and a 25 σ outlier we still
        // get ~scale/r_abs ≈ 10% weight; combined with point uncertainty from
        // the 2× scale-drift init, the outlier-affected point converges to
        // O(few cm) of residual error, vs decimetres without Huber.
        assert!(
            outlier_err_huber < 0.5 * outlier_err_no_huber,
            "expected Huber to halve outlier-induced error at minimum: \
             with_huber={:.4} m, no_huber={:.4} m",
            outlier_err_huber,
            outlier_err_no_huber,
        );
        assert!(
            outlier_err_huber < 0.10,
            "expected Huber to keep outlier-affected point within 10 cm, got {:.4} m",
            outlier_err_huber,
        );
        // Sanity: overall max error with Huber stays small too.
        assert!(
            max_err_huber < 0.10,
            "Huber-BA max point error {:.4} m too large (regression in inliers?)",
            max_err_huber,
        );
    }

    // ── Orientation (up) prior ───────────────────────────────────────────

    /// An orientation prior removes the global TILT that reprojection BA is blind to.
    ///
    /// The perturbation applied here is an exact null direction of everything upstream scores:
    /// the whole reconstruction — every pose and every point — is rotated about the world Z
    /// axis, which is also the cameras' optical axis and the line their centres sit on. So
    ///
    ///   * every reprojection residual is unchanged (rigid rotation of the scene about the
    ///     cameras),
    ///   * every camera CENTRE is unchanged (they lie on the rotation axis), so the centre
    ///     priors — which upstream already has — are satisfied exactly before and after.
    ///
    /// That is not a contrived setup, it is the monocular reality in miniature: a forward pass
    /// with no revisits carries no observation of absolute orientation whatsoever, and the tilt
    /// it accumulates is invisible to the objective. Only the up prior gives that direction any
    /// curvature. The control branch below runs the identical problem with `up_world: None` —
    /// i.e. exactly the upstream code path — and confirms the tilt survives untouched.
    #[test]
    fn schur_ba_up_prior_corrects_global_tilt() {
        let cam = PinholeCamera {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        // Cameras ON the world Z axis, identity rotation (looking down +Z). Their centres are
        // therefore fixed points of any rotation about Z.
        let centres: Vec<Vec3F64> = (0..5)
            .map(|i| Vec3F64::new(0.0, 0.0, -0.2 * i as f64))
            .collect();
        let true_poses: Vec<Pose3d> = centres
            .iter()
            .map(|c| Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-c.x, -c.y, -c.z)))
            .collect();

        // 60 points spread in front of the rig.
        let mut true_points: Vec<Vec3F64> = Vec::with_capacity(60);
        for k in 0..60 {
            let kf = k as f64;
            let x = (kf * 0.37).sin() * 1.4 + (kf * 0.13).cos() * 0.5;
            let y = (kf * 0.29).cos() * 1.1 + (kf * 0.11).sin() * 0.4;
            let z = 4.0 + (kf * 0.41).sin() * 1.5;
            true_points.push(Vec3F64::new(x, y, z));
        }

        // Noise-free projections: the point of the test is a null direction, so any residual
        // floor would just muddy the reading.
        let mut observations: Vec<BaObservation> = Vec::new();
        for (pi, pose) in true_poses.iter().enumerate() {
            for (xi, pt) in true_points.iter().enumerate() {
                let pc = pose.transform_point(pt);
                if pc.z <= 0.2 {
                    continue;
                }
                observations.push(BaObservation {
                    pose_idx: pi,
                    point_idx: xi,
                    pixel: [
                        (cam.fx * pc.x / pc.z + cam.cx) as f32,
                        (cam.fy * pc.y / pc.z + cam.cy) as f32,
                    ],
                    ..Default::default()
                });
            }
        }

        // ── Perturb: roll the ENTIRE reconstruction about world Z by θ. ──
        // World map p' = G·p, cameras R' = R·Gᵀ, t' = t. Reprojections are preserved exactly
        // (R'·p' + t' = R·p + t) and centres C' = G·C = C because C is on the axis.
        let theta = 0.35_f64;
        let (s, c) = (theta.sin(), theta.cos());
        let g = Mat3F64::from_cols(
            Vec3F64::new(c, s, 0.0),
            Vec3F64::new(-s, c, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let g_t = g.transpose();
        let init_poses: Vec<Pose3d> = true_poses
            .iter()
            .map(|p| Pose3d::new(p.rotation * g_t, p.translation))
            .collect();
        let init_points: Vec<Vec3F64> = true_points.iter().map(|p| g * *p).collect();

        // World-frame up direction the cameras' image-up (0, −1, 0) should point along.
        const UP_WORLD: [f32; 3] = [0.0, -1.0, 0.0];
        // Predicted image-up of a pose in the world frame: Rᵀ · (0,−1,0).
        let up_of = |p: &Pose3d| -> [f64; 3] {
            let rt = p.rotation.transpose();
            let u = rt * Vec3F64::new(0.0, -1.0, 0.0);
            [u.x, u.y, u.z]
        };
        let up_err = |p: &Pose3d| -> f64 {
            let u = up_of(p);
            ((u[0] - 0.0).powi(2) + (u[1] + 1.0).powi(2) + u[2].powi(2)).sqrt()
        };

        let init_err = init_poses.iter().map(up_err).fold(0.0_f64, f64::max);
        assert!(
            init_err > 0.3,
            "test is vacuous: initial tilt {init_err:.4} is already small"
        );

        // Centre priors (an UPSTREAM feature) pin translation and scale but say nothing about
        // roll — they are satisfied identically before and after the perturbation.
        let centre_only: Vec<Option<BaPosePrior>> = centres
            .iter()
            .map(|c| Some(BaPosePrior::new([c.x as f32, c.y as f32, c.z as f32], 0.05)))
            .collect();
        let with_up: Vec<Option<BaPosePrior>> = centre_only
            .iter()
            .map(|p| p.map(|p| p.with_up(UP_WORLD, 0.05)))
            .collect();

        let params = BaParams {
            max_iterations: 200,
            cost_tolerance: 1e-10,
            ..BaParams::default()
        };

        // Control: centre priors only — this is byte-for-byte the upstream objective.
        let control = bundle_adjust_schur_with_priors(
            &init_poses,
            &init_points,
            &observations,
            &cam,
            &params,
            Some(&centre_only),
        )
        .unwrap();
        let control_err = control.poses.iter().map(up_err).fold(0.0_f64, f64::max);
        assert!(
            control_err > 0.3,
            "centre-prior-only BA unexpectedly fixed the tilt ({control_err:.4}); the \
             perturbation is not the intended null direction and the test proves nothing"
        );

        // With the orientation prior.
        let result = bundle_adjust_schur_with_priors(
            &init_poses,
            &init_points,
            &observations,
            &cam,
            &params,
            Some(&with_up),
        )
        .unwrap();
        let err = result.poses.iter().map(up_err).fold(0.0_f64, f64::max);
        assert!(
            err < 0.05,
            "up prior left {err:.4} of tilt (control {control_err:.4}, initial {init_err:.4})"
        );
    }

    // ── Constant-velocity motion prior ───────────────────────────────────

    /// The constant-velocity prior pulls a middle keyframe back onto its neighbours' trajectory
    /// when reprojection cannot see where it is.
    ///
    /// Middle camera 1 observes a PRIVATE set of landmarks — no co-visibility with 0 or 2. That
    /// is the near-zero-parallax keyframe in its purest form: camera 1 plus its own points are a
    /// free-floating piece, and translating or rotating the pair together changes no reprojection
    /// residual at all. Cameras 0 and 2 are pinned by centre priors (an upstream feature), so
    /// upstream's objective is completely indifferent to where camera 1 sits between them.
    ///
    /// The triplet residual is what supplies the missing constraint, and it does so WITHOUT
    /// asserting a scale: it constrains the norm RATIO ‖C1−C0‖/‖C2−C0‖ toward `alpha`, plus
    /// constant angular velocity on the rotations. The control branch (`motion_priors = None`)
    /// is the upstream path and leaves the jitter exactly where it started.
    #[test]
    fn schur_ba_motion_prior_pulls_jittered_middle_pose() {
        let cam = PinholeCamera {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        // True trajectory: C0 = origin, C1 = midpoint, C2 = 2 m back along −Z; identity rotations.
        let true_centres = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(0.0, 0.0, -1.0),
            Vec3F64::new(0.0, 0.0, -2.0),
        ];
        let true_poses: Vec<Pose3d> = true_centres
            .iter()
            .map(|c| Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-c.x, -c.y, -c.z)))
            .collect();

        // Landmarks: 0..40 shared by cameras 0 and 2; 40..80 seen ONLY by camera 1.
        let mut points: Vec<Vec3F64> = Vec::with_capacity(80);
        for k in 0..80 {
            let kf = k as f64;
            let x = (kf * 0.37).sin() * 1.5 + (kf * 0.13).cos() * 0.6;
            let y = (kf * 0.29).cos() * 1.2 + (kf * 0.11).sin() * 0.5;
            let z = 4.0 + (kf * 0.41).sin() * 1.5;
            points.push(Vec3F64::new(x, y, z));
        }
        let visibility = |pose_idx: usize, point_idx: usize| -> bool {
            if pose_idx == 1 {
                point_idx >= 40
            } else {
                point_idx < 40
            }
        };

        let project = |pose: &Pose3d, pt: &Vec3F64| -> Option<[f32; 2]> {
            let pc = pose.transform_point(pt);
            (pc.z > 0.2).then(|| {
                [
                    (cam.fx * pc.x / pc.z + cam.cx) as f32,
                    (cam.fy * pc.y / pc.z + cam.cy) as f32,
                ]
            })
        };

        let mut observations: Vec<BaObservation> = Vec::new();
        for (pi, pose) in true_poses.iter().enumerate() {
            for (xi, pt) in points.iter().enumerate() {
                if !visibility(pi, xi) {
                    continue;
                }
                let Some(pixel) = project(pose, pt) else {
                    continue;
                };
                observations.push(BaObservation {
                    pose_idx: pi,
                    point_idx: xi,
                    pixel,
                    ..Default::default()
                });
            }
        }

        // ── Jitter camera 1 AND its private points by the same rigid motion, so its
        // reprojection residuals stay exactly zero. Only the triplet residual can object.
        let phi = 0.25_f64;
        let (sp, cp) = (phi.sin(), phi.cos());
        // Ry(φ) as the jittered world→cam rotation of camera 1.
        let r1_jit = Mat3F64::from_cols(
            Vec3F64::new(cp, 0.0, -sp),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(sp, 0.0, cp),
        );
        let c1_jit = Vec3F64::new(0.0, 0.0, -0.4);
        let t1_jit = -(r1_jit * c1_jit);
        let pose1_jit = Pose3d::new(r1_jit, t1_jit);

        let init_poses = vec![true_poses[0], pose1_jit, true_poses[2]];
        let init_points: Vec<Vec3F64> = points
            .iter()
            .enumerate()
            .map(|(xi, pt)| {
                if xi < 40 {
                    *pt
                } else {
                    // p' = R1'ᵀ · (pc − t1'), where pc are the point's TRUE camera-1 coordinates.
                    let pc = true_poses[1].transform_point(pt);
                    r1_jit.transpose() * (pc - t1_jit)
                }
            })
            .collect();

        // Centre priors on the ENDPOINTS only (upstream feature). Camera 1 gets none — its
        // position is exactly what the motion prior is being asked to supply.
        let pose_priors: Vec<Option<BaPosePrior>> = vec![
            Some(BaPosePrior::new([0.0, 0.0, 0.0], 0.01)),
            None,
            Some(BaPosePrior::new([0.0, 0.0, -2.0], 0.01)),
        ];

        let motion = [BaMotionPrior {
            i0: 0,
            i1: 1,
            i2: 2,
            alpha: 0.5,
            position_sigma: 0.02,
            orientation_sigma: 0.02,
        }];

        let params = BaParams {
            max_iterations: 300,
            cost_tolerance: 1e-10,
            ..BaParams::default()
        };

        // Diagnostics on the returned camera 1.
        let ratio_of = |poses: &[Pose3d]| -> f64 {
            let centre = |p: &Pose3d| -(p.rotation.transpose() * p.translation);
            let (c0, c1, c2) = (centre(&poses[0]), centre(&poses[1]), centre(&poses[2]));
            (c1 - c0).length() / (c2 - c0).length()
        };
        let tilt_of = |poses: &[Pose3d]| -> f64 {
            let tr = poses[1].rotation.col(0).x
                + poses[1].rotation.col(1).y
                + poses[1].rotation.col(2).z;
            (((tr - 1.0) * 0.5).clamp(-1.0, 1.0)).acos()
        };

        assert!(
            (ratio_of(&init_poses) - 0.2).abs() < 1e-6,
            "setup error: initial ratio {:.4} should be 0.2",
            ratio_of(&init_poses)
        );

        // Control: no motion priors — the upstream code path.
        let control = bundle_adjust_schur_with_all_priors(
            &init_poses,
            &init_points,
            &observations,
            &cam,
            &params,
            Some(&pose_priors),
            None,
        )
        .unwrap();
        let control_ratio = ratio_of(&control.poses);
        let control_tilt = tilt_of(&control.poses);
        assert!(
            (control_ratio - 0.5).abs() > 0.2 && control_tilt > 0.15,
            "reprojection alone unexpectedly recovered camera 1 (ratio {control_ratio:.4}, \
             tilt {control_tilt:.4} rad); the jitter is not the intended null direction"
        );

        // With the constant-velocity prior.
        let result = bundle_adjust_schur_with_all_priors(
            &init_poses,
            &init_points,
            &observations,
            &cam,
            &params,
            Some(&pose_priors),
            Some(&motion),
        )
        .unwrap();
        let ratio = ratio_of(&result.poses);
        let tilt = tilt_of(&result.poses);
        assert!(
            (ratio - 0.5).abs() < 0.05,
            "motion prior left the position ratio at {ratio:.4} (want 0.5; control {control_ratio:.4})"
        );
        assert!(
            tilt < 0.05,
            "motion prior left {tilt:.4} rad of angular-velocity error (control {control_tilt:.4})"
        );
    }

    /// The motion-prior FD Jacobian must stay usable on a map far from the origin.
    ///
    /// Its translation columns are differenced over camera centres — an f32 `-Rᵀt` — so a fixed
    /// ABSOLUTE step loses relative precision as coordinates grow: at magnitude `m` the perturbed
    /// centre keeps only `~7 - log10(m)` significant digits of the step.
    ///
    /// The offset here is deliberately extreme, and the honest reason is that anything realistic
    /// does not discriminate. Measured on this test with the absolute step restored: 500 m passes,
    /// 5 km passes, and only at 50 km does the ratio fall to 0.3864 and the recovery fail. So this
    /// is a guard against a class of error, not a reproduction of one that was hurting a real map
    /// — the relative step is simply free, and unlike the absolute one it has no horizon past
    /// which it quietly stops working.
    #[test]
    fn motion_prior_survives_a_map_far_from_the_origin() {
        const OFFSET: f64 = 50_000.0;
        let cam = PinholeCamera {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let shift = Vec3F64::new(OFFSET, OFFSET, OFFSET);
        let true_centres = [
            shift,
            shift + Vec3F64::new(0.0, 0.0, -1.0),
            shift + Vec3F64::new(0.0, 0.0, -2.0),
        ];
        let true_poses: Vec<Pose3d> = true_centres
            .iter()
            .map(|c| Pose3d::new(Mat3F64::IDENTITY, -(Mat3F64::IDENTITY * *c)))
            .collect();

        let mut points: Vec<Vec3F64> = Vec::with_capacity(80);
        for k in 0..80 {
            let kf = k as f64;
            points.push(
                shift
                    + Vec3F64::new(
                        (kf * 0.37).sin() * 1.5 + (kf * 0.13).cos() * 0.6,
                        (kf * 0.29).cos() * 1.2 + (kf * 0.11).sin() * 0.5,
                        4.0 + (kf * 0.41).sin() * 1.5,
                    ),
            );
        }
        // Camera 1 sees a PRIVATE landmark set, so jittering it together with those points keeps
        // its reprojection residuals at zero and only the triplet residual can object.
        let visibility = |pose_idx: usize, point_idx: usize| -> bool {
            if pose_idx == 1 {
                point_idx >= 40
            } else {
                point_idx < 40
            }
        };
        let mut observations: Vec<BaObservation> = Vec::new();
        for (pi, pose) in true_poses.iter().enumerate() {
            for (xi, pt) in points.iter().enumerate() {
                if !visibility(pi, xi) {
                    continue;
                }
                let pc = pose.transform_point(pt);
                if pc.z <= 0.2 {
                    continue;
                }
                observations.push(BaObservation {
                    pose_idx: pi,
                    point_idx: xi,
                    pixel: [
                        (cam.fx * pc.x / pc.z + cam.cx) as f32,
                        (cam.fy * pc.y / pc.z + cam.cy) as f32,
                    ],
                    ..Default::default()
                });
            }
        }

        let c1_jit = shift + Vec3F64::new(0.0, 0.0, -0.4);
        let pose1_jit = Pose3d::new(Mat3F64::IDENTITY, -(Mat3F64::IDENTITY * c1_jit));
        let init_poses = vec![true_poses[0], pose1_jit, true_poses[2]];
        let init_points: Vec<Vec3F64> = points
            .iter()
            .enumerate()
            .map(|(xi, pt)| {
                if xi < 40 {
                    *pt
                } else {
                    let pc = true_poses[1].transform_point(pt);
                    pose1_jit.rotation.transpose() * (pc - pose1_jit.translation)
                }
            })
            .collect();

        let pose_priors: Vec<Option<BaPosePrior>> = vec![
            Some(BaPosePrior::new(
                [
                    true_centres[0].x as f32,
                    true_centres[0].y as f32,
                    true_centres[0].z as f32,
                ],
                0.01,
            )),
            None,
            Some(BaPosePrior::new(
                [
                    true_centres[2].x as f32,
                    true_centres[2].y as f32,
                    true_centres[2].z as f32,
                ],
                0.01,
            )),
        ];
        let motion = [BaMotionPrior {
            i0: 0,
            i1: 1,
            i2: 2,
            alpha: 0.5,
            position_sigma: 0.02,
            orientation_sigma: 0.02,
        }];
        let params = BaParams {
            max_iterations: 300,
            cost_tolerance: 1e-10,
            ..BaParams::default()
        };
        let ratio_of = |poses: &[Pose3d]| -> f64 {
            let centre = |p: &Pose3d| -(p.rotation.transpose() * p.translation);
            let (c0, c1, c2) = (centre(&poses[0]), centre(&poses[1]), centre(&poses[2]));
            (c1 - c0).length() / (c2 - c0).length()
        };
        assert!(
            (ratio_of(&init_poses) - 0.2).abs() < 1e-4,
            "setup error: initial ratio {:.4} should be 0.2",
            ratio_of(&init_poses)
        );

        let res = bundle_adjust_schur_with_all_priors(
            &init_poses,
            &init_points,
            &observations,
            &cam,
            &params,
            Some(&pose_priors),
            Some(&motion),
        )
        .unwrap();
        let ratio = ratio_of(&res.poses);
        assert!(
            (ratio - 0.5).abs() < 0.1,
            "motion prior left the position ratio at {ratio:.4} on a map {OFFSET} m from the \
             origin (want 0.5, started 0.2) — the FD step is not tracking coordinate magnitude"
        );
    }

    // ── Sparse reduced camera system ────────────────────────────────────────

    /// Deterministic, seedable, and NOT the `rand` crate — these tests assert bit-level equality,
    /// so the numbers must be identical on every machine and every run.
    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Top 24 bits → [-1, 1), so the products below stay in a range where f32 rounding is
            // ordinary rather than denormal.
            ((self.0 >> 40) as f32 / (1u32 << 23) as f32) - 1.0
        }
    }

    /// The dense assembly, written out independently of the solver, so the accumulator is compared
    /// against something other than itself. Mirrors `bundle_adjust_schur_impl`: A blocks assigned
    /// onto the diagonal, motion-prior blocks added, per-point Schur corrections subtracted, then
    /// the whole matrix symmetrised.
    fn dense_reference(
        n: usize,
        a_blocks: &[[f32; 36]],
        offdiag: &[((usize, usize), [f32; 36])],
        schur: &[((usize, usize), [f32; 36])],
    ) -> Mat<f64> {
        let dim = n * 6;
        let mut m = Mat::<f64>::zeros(dim, dim);
        for (k, ab) in a_blocks.iter().enumerate() {
            for i in 0..6 {
                for j in 0..6 {
                    m[(k * 6 + i, k * 6 + j)] = ab[i * 6 + j] as f64;
                }
            }
        }
        for ((la, lb), blk) in offdiag {
            for i in 0..6 {
                for j in 0..6 {
                    m[(la * 6 + i, lb * 6 + j)] += blk[i * 6 + j] as f64;
                }
            }
        }
        for ((i1, i2), blk) in schur {
            for r in 0..6 {
                for c in 0..6 {
                    m[(i1 * 6 + r, i2 * 6 + c)] -= blk[r * 6 + c] as f64;
                }
            }
        }
        for i in 0..dim {
            for j in (i + 1)..dim {
                let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
                m[(i, j)] = avg;
                m[(j, i)] = avg;
            }
        }
        m
    }

    /// The lower triangle the sparse path hands to the factorisation must be the SAME NUMBERS the
    /// dense path hands to its factorisation — not close, identical. Anything less and
    /// `sparse_reduced_system` would be a second solver with its own convergence behaviour rather
    /// than a storage choice, and every existing tolerance in this file would be re-tuned by it.
    ///
    /// Bit equality is achievable because `BlockAccum` accumulates in `f64`, exactly as `Mat<f64>`
    /// does, in the same order, and averages the two triangles with the same expression.
    #[test]
    fn sparse_lower_triangle_is_bit_identical_to_the_dense_one() {
        const N: usize = 6;
        let mut rng = Lcg(0x5eed_1234);
        let rand_block = |rng: &mut Lcg| {
            let mut b = [0.0_f32; 36];
            for v in b.iter_mut() {
                *v = rng.next_f32();
            }
            b
        };

        // Banded covisibility: point j is seen by cameras j and j+1 only. Cameras 0 and 5 therefore
        // share nothing, which is the whole point — a dense matrix would store that block anyway.
        let b_by_point: Vec<Vec<(usize, [f32; 18])>> = (0..N - 1)
            .map(|j| vec![(j, [0.0; 18]), (j + 1, [0.0; 18])])
            .collect();
        // A motion prior couples 0 and 5 with no shared structure.
        let motion_pairs = [(0usize, N - 1)];

        let a_blocks: Vec<[f32; 36]> = (0..N).map(|_| rand_block(&mut rng)).collect();
        let offdiag: Vec<((usize, usize), [f32; 36])> = vec![
            ((0, N - 1), rand_block(&mut rng)),
            ((N - 1, 0), rand_block(&mut rng)),
        ];
        let mut schur: Vec<((usize, usize), [f32; 36])> = Vec::new();
        for b_for_j in &b_by_point {
            for (i1, _) in b_for_j {
                for (i2, _) in b_for_j {
                    schur.push(((*i1, *i2), rand_block(&mut rng)));
                }
            }
        }

        let dense = dense_reference(N, &a_blocks, &offdiag, &schur);

        let mut acc = BlockAccum::new(N, &b_by_point, &motion_pairs);
        acc.clear();
        for (k, ab) in a_blocks.iter().enumerate() {
            let slot = acc.slot(k, k).expect("diagonal slot");
            for i in 0..6 {
                for j in 0..6 {
                    acc.blocks[slot][i * 6 + j] = ab[i * 6 + j] as f64;
                }
            }
        }
        for ((la, lb), blk) in &offdiag {
            let slot = acc.slot(*la, *lb).expect("motion-prior slot");
            for i in 0..6 {
                for j in 0..6 {
                    acc.blocks[slot][i * 6 + j] += blk[i * 6 + j] as f64;
                }
            }
        }
        for ((i1, i2), blk) in &schur {
            let slot = acc.slot(*i1, *i2).expect("covisible slot");
            for r in 0..6 {
                for c in 0..6 {
                    acc.blocks[slot][r * 6 + c] -= blk[r * 6 + c] as f64;
                }
            }
        }

        let dim = N * 6;
        let trips = acc.lower_triplets();
        let mut got = vec![f64::NAN; dim * dim];
        for t in &trips {
            assert!(
                t.row >= t.col,
                "upper-triangle triplet at ({}, {})",
                t.row,
                t.col
            );
            got[t.row * dim + t.col] = t.val;
        }

        // CONTROL 1: the pattern must actually be sparse, or bit-equality is trivially satisfied by
        // a dense emission and this test proves nothing about the sparse path.
        let full = dim * (dim + 1) / 2;
        assert!(
            trips.len() < full * 3 / 4,
            "emitted {} of {full} lower entries — pattern is not sparse, so this test is not \
             testing sparsity",
            trips.len()
        );
        // CONTROL 2: the dense matrix must have real content in the blocks we do emit, or "equal"
        // could mean "both all zero".
        let live: usize = (0..dim)
            .flat_map(|r| (0..=r).map(move |c| (r, c)))
            .filter(|&(r, c)| dense[(r, c)] != 0.0)
            .count();
        assert!(
            live > dim,
            "dense reference is nearly empty ({live} nonzeros)"
        );

        for r in 0..dim {
            for c in 0..=r {
                let want = dense[(r, c)];
                let have = got[r * dim + c];
                if have.is_nan() {
                    // Structurally absent: the dense value must be exactly zero.
                    assert_eq!(
                        want, 0.0,
                        "({r}, {c}) is outside the sparse pattern but dense holds {want}"
                    );
                } else {
                    assert_eq!(
                        want.to_bits(),
                        have.to_bits(),
                        "({r}, {c}): dense {want:e} vs sparse {have:e}"
                    );
                }
            }
        }
    }

    /// A constant-velocity prior couples a camera triplet whether or not those cameras share a
    /// point. If the pattern were built from covisibility alone, the prior's 6×6 block would have
    /// nowhere to go — and dropping it would sever the pose graph exactly where the geometry is
    /// weakest, because when two keyframes share no point the prior is the ONLY thing connecting
    /// them.
    ///
    /// The control is the same accumulator built WITHOUT the motion pair: it must lack the slot.
    #[test]
    fn motion_prior_pairs_enter_the_pattern_without_covisibility() {
        // Cameras 0 and 1 share point 0. Camera 2 shares nothing with either.
        let b_by_point: Vec<Vec<(usize, [f32; 18])>> = vec![vec![(0, [0.0; 18]), (1, [0.0; 18])]];

        let covis_only = BlockAccum::new(3, &b_by_point, &[]);
        assert!(
            covis_only.slot(1, 2).is_none() && covis_only.slot(2, 1).is_none(),
            "control failed: (1, 2) is covisible after all, so this test cannot show the \
             motion-prior pairs are what added it"
        );

        let with_prior = BlockAccum::new(3, &b_by_point, &[(1, 2)]);
        assert!(
            with_prior.slot(1, 2).is_some(),
            "motion pair (1, 2) missing"
        );
        assert!(
            with_prior.slot(2, 1).is_some(),
            "motion pair inserted in one order only; `lower_triplets` averages a block against \
             its transpose and would read the wrong slot"
        );
        // Still not a dense pattern: (0, 2) is neither covisible nor coupled by the prior.
        assert!(with_prior.slot(0, 2).is_none());
    }

    /// A sequential capture: cameras walking forward, each point visible only from a short window
    /// of consecutive views. This is the regime `sparse_reduced_system` exists for, and the one
    /// where a covisibility-only pattern would drop a motion prior.
    ///
    /// Returns `(poses_gt, poses_init, points_gt, points_init, observations, motion_priors)`.
    #[allow(clippy::type_complexity)]
    fn walkthrough(
        cam: &PinholeCamera,
        n_cams: usize,
        window: usize,
    ) -> (
        Vec<Pose3d>,
        Vec<Pose3d>,
        Vec<Vec3F64>,
        Vec<Vec3F64>,
        Vec<BaObservation>,
        Vec<BaMotionPrior>,
    ) {
        let mut rng = Lcg(0xabcd_ef01);
        // Camera k sits at x = 0.4k looking down +Z.
        let pose_at = |x: f64| Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-x, 0.0, 0.0));
        let poses_gt: Vec<Pose3d> = (0..n_cams).map(|k| pose_at(0.4 * k as f64)).collect();

        // Four points per camera window, laid out ahead of the camera that starts the window.
        let mut points_gt = Vec::new();
        let mut owner = Vec::new();
        for k in 0..n_cams.saturating_sub(window - 1) {
            for q in 0..4 {
                let x = 0.4 * k as f64 + 0.2 * q as f64 + 0.1 * rng.next_f32() as f64;
                let y = 0.8 * rng.next_f32() as f64;
                let z = 5.0 + 0.5 * rng.next_f32() as f64;
                points_gt.push(Vec3F64::new(x, y, z));
                owner.push(k);
            }
        }

        let project = |pose: &Pose3d, pw: &Vec3F64| -> [f32; 2] {
            let pc = pose.transform_point(pw);
            [
                (cam.fx * pc.x / pc.z + cam.cx) as f32,
                (cam.fy * pc.y / pc.z + cam.cy) as f32,
            ]
        };
        let mut observations = Vec::new();
        for (pi, pt) in points_gt.iter().enumerate() {
            let first = owner[pi];
            let last = (first + window).min(n_cams);
            for (k, pose) in poses_gt.iter().enumerate().take(last).skip(first) {
                observations.push(BaObservation {
                    pose_idx: k,
                    point_idx: pi,
                    pixel: project(pose, pt),
                    // Camera 0 anchors the gauge.
                    fixed_pose: k == 0,
                    fixed_point: false,
                    ..BaObservation::default()
                });
            }
        }

        // Constant-velocity priors over every consecutive triplet. With `window = 2`, cameras k and
        // k+2 share NO point, so these are exactly the couplings covisibility does not supply.
        let motion: Vec<BaMotionPrior> = (0..n_cams.saturating_sub(2))
            .map(|k| BaMotionPrior {
                i0: k,
                i1: k + 1,
                i2: k + 2,
                alpha: 0.5,
                position_sigma: 0.05,
                orientation_sigma: 0.05,
            })
            .collect();

        let poses_init: Vec<Pose3d> = poses_gt
            .iter()
            .enumerate()
            .map(|(k, p)| {
                if k == 0 {
                    *p
                } else {
                    Pose3d::new(
                        p.rotation,
                        p.translation + Vec3F64::new(0.03, -0.02, 0.015) * (k as f64),
                    )
                }
            })
            .collect();
        let points_init: Vec<Vec3F64> = points_gt
            .iter()
            .map(|p| *p + Vec3F64::new(0.04, -0.05, 0.06))
            .collect();

        (
            poses_gt,
            poses_init,
            points_gt,
            points_init,
            observations,
            motion,
        )
    }

    /// End-to-end A/B: the same problem solved with `sparse_reduced_system` off and on. This is the
    /// test the unit one cannot be — it exercises the accumulator through the real assembly, so a
    /// mis-addressed slot or a dropped motion-prior block shows up as a different step and a
    /// visibly different answer.
    ///
    /// The tolerance is loose relative to the bit-equality above, and deliberately so: the two
    /// paths hand the SAME lower triangle to DIFFERENT factorisations. A fill-reducing sparse
    /// Cholesky permutes the matrix and does not perform the dense one's operations in the dense
    /// one's order, so the steps differ at roundoff and the difference compounds over LM
    /// iterations. What must not differ is where the two land.
    #[test]
    fn sparse_reduced_system_matches_dense_on_a_sequential_capture() {
        let cam = test_camera();
        let (poses_gt, poses_init, _points_gt, points_init, obs, motion) = walkthrough(&cam, 10, 2);

        let solve = |sparse: bool| {
            bundle_adjust_schur_with_all_priors(
                &poses_init,
                &points_init,
                &obs,
                &cam,
                &BaParams {
                    max_iterations: 25,
                    sparse_reduced_system: sparse,
                    ..BaParams::default()
                },
                None,
                Some(&motion),
            )
            .expect("solve failed")
        };
        let dense = solve(false);
        let sparse = solve(true);

        // CONTROL: the solve has to have DONE something, or "the two agree" is a statement about
        // two no-ops. Measure how far the dense run moved from its initial guess; the agreement
        // tolerance below is only meaningful if it is far smaller than this.
        let moved = poses_init
            .iter()
            .zip(dense.poses.iter())
            .map(|(a, b)| (a.translation - b.translation).length())
            .fold(0.0_f64, f64::max);
        assert!(
            moved > 1e-2,
            "dense run moved poses by at most {moved:e} m — nothing was solved, so agreement \
             with the sparse run proves nothing"
        );
        assert!(
            dense.final_cost < 1e-4,
            "dense run did not converge (final cost {})",
            dense.final_cost
        );

        let pose_gap = dense
            .poses
            .iter()
            .zip(sparse.poses.iter())
            .map(|(a, b)| (a.translation - b.translation).length())
            .fold(0.0_f64, f64::max);
        let point_gap = dense
            .points
            .iter()
            .zip(sparse.points.iter())
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0_f64, f64::max);
        assert!(
            pose_gap < 1e-6 && point_gap < 1e-6,
            "sparse and dense disagree: pose gap {pose_gap:e} m, point gap {point_gap:e} m \
             (dense moved {moved:e} m from its initial guess)"
        );

        // And the answer they agree on must be the RIGHT one, or the whole A/B could be two runs
        // of an identically broken assembly. Stated relative to the initial guess rather than as
        // an absolute bound: only camera 0 is fixed and every point is free, so the problem keeps
        // a gauge freedom and exact ground-truth recovery is not on offer at any tolerance.
        let worst = |a: &[Pose3d], b: &[Pose3d]| {
            a.iter()
                .zip(b.iter())
                .map(|(p, g)| (p.translation - g.translation).length())
                .fold(0.0_f64, f64::max)
        };
        let before = worst(&poses_init, &poses_gt);
        let after = worst(&sparse.poses, &poses_gt);
        assert!(
            after * 3.0 < before,
            "sparse run left the worst pose {after:e} m from truth, having started {before:e} m \
             away — it agrees with the dense run about a bad answer"
        );
        // Points are NOT checked against ground truth, and that is not an oversight: the residual
        // gauge is a similarity, the scene sits ~5 m out while the cameras span 3.6 m, and a scale
        // slide far too small to move a camera walks every point several centimetres. Measured
        // here: poses end 7× closer to truth while the worst point ends FARTHER out than it
        // started. The gauge-free statement is the objective, so assert on that instead.
        assert!(
            (dense.final_cost - sparse.final_cost).abs()
                <= 1e-3 * dense.final_cost.abs().max(1e-12),
            "objective disagrees: dense {:e} vs sparse {:e}",
            dense.final_cost,
            sparse.final_cost
        );
    }

    /// Where the two paths stand on a problem the size the sparse one is FOR. Not an assertion:
    /// a wall-clock threshold on a shared machine is a flake, and the answer legitimately flips
    /// with density — which is why `sparse_reduced_system` defaults to off.
    ///
    /// Run it deliberately, in release, single-threaded:
    ///
    /// ```text
    /// cargo test --release -p kornia-3d --lib -- --ignored --nocapture reduced_system_phase_timings
    /// ```
    #[test]
    #[ignore = "measurement, not an assertion; run explicitly in release"]
    fn reduced_system_phase_timings() {
        const N_CAMS: usize = 250;
        const WINDOW: usize = 8;
        let cam = test_camera();
        let (_, poses_init, _, points_init, obs, motion) = walkthrough(&cam, N_CAMS, WINDOW);

        let pattern_density = {
            // Same pattern the solver builds, counted here so the timings can be read against it.
            let mut by_point: Vec<Vec<usize>> = vec![Vec::new(); points_init.len()];
            for o in &obs {
                by_point[o.point_idx].push(o.pose_idx);
            }
            let mut seen = std::collections::HashSet::new();
            for cams in &by_point {
                for a in cams {
                    for b in cams {
                        seen.insert((*a, *b));
                    }
                }
            }
            seen.len() as f64 / (N_CAMS * N_CAMS) as f64
        };
        println!(
            "{N_CAMS} cameras, {} points, {} observations, dim = {}, covisible pairs = {:.1}%",
            points_init.len(),
            obs.len(),
            N_CAMS * 6,
            100.0 * pattern_density
        );

        for (sparse, intr) in [(false, false), (true, false), (true, true)] {
            BA_ASM_NANOS.swap(0, Ordering::Relaxed);
            BA_FACT_NANOS.swap(0, Ordering::Relaxed);
            BA_LIN_NANOS.swap(0, Ordering::Relaxed);
            let t = std::time::Instant::now();
            let r = bundle_adjust_schur_with_all_priors(
                &poses_init,
                &points_init,
                &obs,
                &cam,
                &BaParams {
                    max_iterations: 5,
                    sparse_reduced_system: sparse,
                    refine_focal: intr,
                    refine_k1: intr,
                    ..BaParams::default()
                },
                None,
                Some(&motion),
            )
            .expect("solve failed");
            let wall = t.elapsed().as_secs_f64();
            let per = |n: u64| n as f64 / 1e9 / r.iterations.max(1) as f64;
            println!(
                "sparse={sparse:<5} intr={intr:<5} iters={:<3} wall={wall:7.3}s  \
                 linearise={:7.3}s/it  \
                 assemble={:7.3}s/it  factorise={:7.3}s/it  cost={:.6e}",
                r.iterations,
                per(BA_LIN_NANOS.load(Ordering::Relaxed)),
                per(BA_ASM_NANOS.load(Ordering::Relaxed)),
                per(BA_FACT_NANOS.load(Ordering::Relaxed)),
                r.final_cost,
            );
        }
    }

    // ── Shared free intrinsics: α (focal scale) and k1 ──────────────────

    /// Bit-exactness gate for "every existing caller is unchanged".
    ///
    /// `BaParams::refine_focal` / `refine_k1` default false, so a default solve must reproduce the
    /// solver as it stood at `45e0a846` — the commit before the intrinsics were freed — to the
    /// LAST BIT, not to a tolerance. That claim covers more than the new border being skipped: the
    /// `BaIntrinsics` refactor hoisted four `f64 → f32` casts per observation out of the residual's
    /// hot loop, and the rest of the suite (all of which runs at `BaParams::default()`) would only
    /// notice a change that exceeded its tolerances.
    ///
    /// Constants captured by running that commit's solver on these two fixtures and dumping
    /// `f64::to_bits` of every returned pose translation, every rotation element and every point,
    /// plus `f32::to_bits` of `final_cost`. The walkthrough's 229 values are compared through an
    /// FNV-1a-64 digest of the same byte stream — the two-view fixture is spelled out so a failure
    /// there says WHICH value moved.
    #[test]
    fn defaults_are_bit_identical_to_the_frozen_solver() {
        fn dump(r: &BaResult) -> Vec<u64> {
            let mut v: Vec<u64> = vec![r.final_cost.to_bits() as u64];
            for p in &r.poses {
                for c in [p.translation.x, p.translation.y, p.translation.z] {
                    v.push(c.to_bits());
                }
                for c in 0..3 {
                    let col = p.rotation.col(c);
                    for e in [col.x, col.y, col.z] {
                        v.push(e.to_bits());
                    }
                }
            }
            for p in &r.points {
                for c in [p.x, p.y, p.z] {
                    v.push(c.to_bits());
                }
            }
            v
        }
        fn fnv1a64(v: &[u64]) -> u64 {
            let mut h = 0xcbf2_9ce4_8422_2325_u64;
            for x in v {
                for b in x.to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100_0000_01b3);
                }
            }
            h
        }

        // Fixture 1: `perturbed_two_view_problem` under `BaParams::default()`.
        const TWO_VIEW: [u64; 37] = [
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x3ff0000000000000,
            0x3fe00b34e0000000,
            0x3f52609100000000,
            0x3ea55843e0000000,
            0x3ff0000000000000,
            0xbe701781a0000000,
            0x3eab7ffde0000000,
            0x3e700b6460000000,
            0x3ff0000000000000,
            0x3f2c319460000000,
            0xbeab801a20000000,
            0xbf2c319460000000,
            0x3ff0000000000000,
            0xbff00afe9d7f9b33,
            0xbff00afec2416c7b,
            0x40140dbe4f497614,
            0x3ff00bb31400accd,
            0xbff00bb31c24b47b,
            0x40140e9fd2944e14,
            0x3ff00aa382f90ccd,
            0x3ff00aa3a01c4b85,
            0x40140d4c83546e14,
            0xbff00b59aafd8b33,
            0x3ff00b597e3f2b85,
            0x40140e300b4cde14,
        ];
        let (poses, points, obs, cam) = perturbed_two_view_problem();
        let r = bundle_adjust_schur(&poses, &points, &obs, &cam, &BaParams::default()).unwrap();
        assert_eq!(r.iterations, 10);
        assert!(!r.converged);
        assert!(r.camera.is_none(), "no flag set, so no camera is invented");
        let got = dump(&r);
        assert_eq!(got.len(), TWO_VIEW.len());
        for (i, (a, b)) in got.iter().zip(TWO_VIEW.iter()).enumerate() {
            assert_eq!(
                a, b,
                "two-view value {i} moved: {a:#018x} != frozen {b:#018x}"
            );
        }

        // Fixture 2: `walkthrough(&cam, 10, 2)` with motion priors, both storage paths. The two
        // agreed bit-for-bit at the frozen commit and must still.
        const WALKTHROUGH_DIGEST: u64 = 0x456e_57e0_0176_46b6;
        let cam2 = test_camera();
        let (_gt, poses_init, _pgt, points_init, obs2, motion) = walkthrough(&cam2, 10, 2);
        for sparse in [false, true] {
            let r = bundle_adjust_schur_with_all_priors(
                &poses_init,
                &points_init,
                &obs2,
                &cam2,
                &BaParams {
                    max_iterations: 25,
                    sparse_reduced_system: sparse,
                    ..BaParams::default()
                },
                None,
                Some(&motion),
            )
            .unwrap();
            assert_eq!(r.iterations, 25);
            assert!(r.camera.is_none());
            let d = fnv1a64(&dump(&r));
            assert_eq!(
                d, WALKTHROUGH_DIGEST,
                "walkthrough (sparse={sparse}) digest {d:#018x} != frozen \
                 {WALKTHROUGH_DIGEST:#018x} — a default-parameter solve is no longer bit-identical \
                 to the pre-intrinsics solver"
            );
        }
    }

    /// A scene whose geometry is at ground truth and whose ONLY error is in the camera.
    ///
    /// Observations are projected through `gt_cam` and, when `k1_gt != 0`, through the radial
    /// term — written out here in f64 rather than reusing `residual_and_jacobians`, so a mutation
    /// to the solver's own model cannot move the data with it.
    ///
    /// NOT built on `walkthrough`, and the reason is the whole reason these tests exist. That scene
    /// translates along x with IDENTICAL rotations, and there the focal is EXACTLY unobservable
    /// however many poses are fixed: with cameras at `C_k = (x_k, 0, 0)`, the point `(X, Y, sZ)`
    /// reproduces every pixel of `(X, Y, Z)` at focal `s·f`, because
    /// `s·f·(X − x_k)/(sZ) = f·(X − x_k)/Z`. Measured: a 15%-wrong focal recovered only to 535 of
    /// 500 there — LM stopping somewhere along a genuinely flat valley, not a solver bug.
    ///
    /// Varying the rotation is what breaks it: the compensating point motion would have to be
    /// `R_k(p'_w − C_k) ∝ diag(1, 1, s)·R_k(p_w − C_k)` simultaneously for every `k`, which no
    /// single `p'_w` satisfies once the `R_k` differ. So the cameras here sit on an arc and yaw,
    /// and the point depths span 4–10 m rather than 5.0–5.5.
    ///
    /// `n_fixed` poses are gauge-fixed. At least one must stay free or the solver returns
    /// `NoFreeVariables`.
    fn intrinsics_only_scene(
        gt_cam: &PinholeCamera,
        k1_gt: f64,
        n_cams: usize,
        n_fixed: usize,
    ) -> (Vec<Pose3d>, Vec<Vec3F64>, Vec<BaObservation>) {
        let mut rng = Lcg(0x1357_9bdf);
        let poses: Vec<Pose3d> = (0..n_cams)
            .map(|k| {
                let f = k as f64 - 0.5 * (n_cams as f64 - 1.0);
                let centre = Vec3F64::new(0.7 * f, 0.25 * (0.9 * f).sin(), -0.4 * f * f * 0.05);
                // Yaw toward the cloud, so no two cameras share a rotation.
                let yaw = 0.11 * f;
                let (c, s) = (yaw.cos(), yaw.sin());
                let r = Mat3F64::from_cols(
                    Vec3F64::new(c, 0.0, -s),
                    Vec3F64::new(0.0, 1.0, 0.0),
                    Vec3F64::new(s, 0.0, c),
                );
                // world→cam: p_c = R·p_w + t with t = −R·C.
                let t = Vec3F64::new(
                    -(r.col(0).x * centre.x + r.col(1).x * centre.y + r.col(2).x * centre.z),
                    -(r.col(0).y * centre.x + r.col(1).y * centre.y + r.col(2).y * centre.z),
                    -(r.col(0).z * centre.x + r.col(1).z * centre.y + r.col(2).z * centre.z),
                );
                Pose3d::new(r, t)
            })
            .collect();
        // Depth spread 4–10 m: a shallow slab is a near-planar scene, where the focal trades
        // against a homography instead of against nothing.
        let points: Vec<Vec3F64> = (0..60)
            .map(|_| {
                Vec3F64::new(
                    2.0 * rng.next_f32() as f64,
                    1.6 * rng.next_f32() as f64,
                    7.0 + 3.0 * rng.next_f32() as f64,
                )
            })
            .collect();

        let project = |pose: &Pose3d, pw: &Vec3F64| -> [f32; 2] {
            let pc = pose.transform_point(pw);
            let (xn, yn) = (pc.x / pc.z, pc.y / pc.z);
            let d = 1.0 + k1_gt * (xn * xn + yn * yn);
            [
                (gt_cam.fx * xn * d + gt_cam.cx) as f32,
                (gt_cam.fy * yn * d + gt_cam.cy) as f32,
            ]
        };
        let mut obs = Vec::new();
        for (pi, pt) in points.iter().enumerate() {
            for (k, pose) in poses.iter().enumerate() {
                obs.push(BaObservation {
                    pose_idx: k,
                    point_idx: pi,
                    pixel: project(pose, pt),
                    fixed_pose: k < n_fixed,
                    fixed_point: false,
                    ..BaObservation::default()
                });
            }
        }
        (poses, points, obs)
    }

    /// T2: a 15%-wrong focal seed must be recovered, and must NOT move when the flag is off.
    ///
    /// Two gauges, because they exercise different halves of the border. With all but the last
    /// camera fixed, almost every observation takes the `pli < 0` path and contributes to `D`,
    /// `g_intr` and `F` but not `E` — the case a border that only accumulated on free poses would
    /// silently get wrong. With only the first two fixed, `E` carries the work.
    #[test]
    fn focal_scale_recovers_a_wrong_seed() {
        let gt = test_camera(); // fx = fy = 500
        const SEED_FX: f64 = 575.0; // 15% high

        for n_fixed in [9usize, 2] {
            let (poses_gt, points_gt, obs) = intrinsics_only_scene(&gt, 0.0, 10, n_fixed);
            let seed_cam = PinholeCamera {
                fx: SEED_FX,
                fy: SEED_FX,
                ..gt.clone()
            };
            let solve = |refine_focal: bool| {
                bundle_adjust_schur(
                    &poses_gt,
                    &points_gt,
                    &obs,
                    &seed_cam,
                    &BaParams {
                        max_iterations: 40,
                        refine_focal,
                        ..BaParams::default()
                    },
                )
                .expect("solve failed")
            };

            let free = solve(true);
            let fitted = free
                .camera
                .clone()
                .expect("refine_focal must report a camera");
            let err = (fitted.fx - gt.fx).abs() / gt.fx;
            assert!(
                err < 0.01,
                "n_fixed={n_fixed}: focal came back {} from a {SEED_FX} seed, want {} (±1%)",
                fitted.fx,
                gt.fx
            );
            // The aspect ratio is NOT a free parameter — one α scales both axes.
            assert!(
                (fitted.fy / fitted.fx - gt.fy / gt.fx).abs() < 1e-6,
                "α must scale fx and fy together, got fx={} fy={}",
                fitted.fx,
                fitted.fy
            );

            // CONTROL: with the flag off the focal must not move, and the fit must be visibly
            // worse — otherwise this test would pass on a solver that simply does nothing.
            let fixed = solve(false);
            assert!(fixed.camera.is_none());
            assert!(
                free.final_cost < 0.05 * fixed.final_cost,
                "n_fixed={n_fixed}: freeing the focal barely helped ({} vs {}), so the fixture \
                 does not constrain the focal",
                free.final_cost,
                fixed.final_cost
            );
        }
    }

    /// T3: `k1` must be recovered from a scene synthesised through a known radial distortion.
    ///
    /// This is the only end-to-end exercise of the `∂/∂k1` column AND its plumbing;
    /// `every_jacobian_matches_central_differences_at_nonzero_k1` checks the derivative, not the
    /// wiring. The convergence bound is also what stands in for a unit test of the `− F_j δi` term
    /// in the point back-substitution: without it the point step is not the Gauss-Newton step and
    /// the solve needs far more iterations to reach this cost, if it reaches it at all.
    #[test]
    fn k1_recovers_a_distorted_scene() {
        let gt = test_camera();
        const K1_GT: f64 = -0.15;
        let (poses_gt, points_gt, obs) = intrinsics_only_scene(&gt, K1_GT, 10, 2);
        // Seeded at k1 = 0 — the caller's camera carries no distortion estimate.
        let solve = |refine_k1: bool| {
            bundle_adjust_schur(
                &poses_gt,
                &points_gt,
                &obs,
                &gt,
                &BaParams {
                    max_iterations: 40,
                    refine_k1,
                    ..BaParams::default()
                },
            )
            .expect("solve failed")
        };

        let free = solve(true);
        let fitted = free.camera.clone().expect("refine_k1 must report a camera");
        assert!(
            (fitted.k1 - K1_GT).abs() < 0.05 * K1_GT.abs(),
            "k1 came back {} from a 0.0 seed, want {K1_GT} (±5%)",
            fitted.k1
        );

        let fixed = solve(false);
        assert!(
            free.final_cost < 0.1 * fixed.final_cost,
            "freeing k1 gave {} against {} with it pinned — under an order of magnitude, so the \
             fixture does not exercise the distortion",
            free.final_cost,
            fixed.final_cost
        );
    }

    /// T4's behavioural twin, and the sharpest test of the accept/reject hazard.
    ///
    /// The distorted scene with the geometry at GT and only `k1` free to move: the points can
    /// absorb almost nothing, so `k1` is the ONLY parameter that meaningfully reduces cost.
    ///
    /// With a trial pass evaluated at the OLD intrinsics, `new_cost` is scored at `k1 = 0` with
    /// only a negligible point step, so `new_cost ≈ cost`, LM rejects every step, λ escalates to
    /// 1e10, the loop breaks, and the solver returns its seed with `Ok` — `final_cost` stuck at the
    /// first iteration's value and `k1` still 0. Every assertion below then fails.
    ///
    /// Run at a wrong focal too: `α` enters the model linearly and `k1` does not, so only one of
    /// the two would survive a partial fix.
    #[test]
    fn trial_cost_is_evaluated_at_the_trial_intrinsics() {
        let gt = test_camera();
        for (k1_gt, seed_fx, refine_focal, refine_k1) in [
            (-0.15_f64, 500.0_f64, false, true),
            (0.0, 575.0, true, false),
            (-0.15, 575.0, true, true),
        ] {
            let (poses_gt, points_gt, obs) = intrinsics_only_scene(&gt, k1_gt, 10, 2);
            let seed_cam = PinholeCamera {
                fx: seed_fx,
                fy: seed_fx,
                ..gt.clone()
            };
            let params = BaParams {
                max_iterations: 40,
                refine_focal,
                refine_k1,
                ..BaParams::default()
            };
            // Cost at the seed, with nothing free — the value a stale trial pass would return.
            let seed_cost = bundle_adjust_schur(
                &poses_gt,
                &points_gt,
                &obs,
                &seed_cam,
                &BaParams {
                    max_iterations: 1,
                    ..BaParams::default()
                },
            )
            .unwrap()
            .final_cost;

            let res = bundle_adjust_schur(&poses_gt, &points_gt, &obs, &seed_cam, &params).unwrap();
            let fitted = res.camera.clone().unwrap();
            assert!(
                res.iterations >= 2,
                "k1_gt={k1_gt} fx={seed_fx}: solver stopped after {} iterations — every step was \
                 rejected, the signature of a trial cost scored at stale intrinsics",
                res.iterations
            );
            assert!(
                res.final_cost < 0.01 * seed_cost,
                "k1_gt={k1_gt} fx={seed_fx}: final cost {} against a seed cost of {seed_cost}",
                res.final_cost
            );
            if refine_k1 {
                assert!(
                    (fitted.k1 - k1_gt).abs() < 0.02,
                    "k1 did not move to {k1_gt}: got {}",
                    fitted.k1
                );
            }
            if refine_focal {
                assert!(
                    (fitted.fx - gt.fx).abs() < 0.01 * gt.fx,
                    "focal did not move to {}: got {}",
                    gt.fx,
                    fitted.fx
                );
            }
        }
    }

    /// T5: the bordered elimination, against a dense Cholesky on the whole `(6P + q)` system.
    ///
    /// This is the border's analogue of `sparse_lower_triangle_is_bit_identical_to_the_dense_one`,
    /// and it is what catches a transposed `V`, a sign slip on `s − Vᵀy₀` and a missing `− Y δi`.
    /// It also asserts the claim the design rests on: freeing a shared intrinsic does NOT densify
    /// the reduced camera system — `dim` and the triplet count are the same with the border on.
    #[test]
    fn bordered_solve_matches_a_dense_reference() {
        use faer::linalg::solvers::Solve;
        for q in 1..=Q_MAX {
            let n = 4usize;
            let dim = n * 6;
            let full = dim + q;
            let mut rng = Lcg(0x5eed_1234 + q as u64);
            // H = GᵀG + 4I: symmetric, positive definite, and dense in the border by construction.
            let g = Mat::<f64>::from_fn(full, full, |_, _| rng.next_f32() as f64);
            let mut h = Mat::<f64>::zeros(full, full);
            for i in 0..full {
                for j in 0..full {
                    let mut s = 0.0;
                    for k in 0..full {
                        s += g[(k, i)] * g[(k, j)];
                    }
                    h[(i, j)] = s + if i == j { 4.0 } else { 0.0 };
                }
            }
            let rhs = Mat::<f64>::from_fn(full, 1, |_, _| rng.next_f32() as f64);

            // (a) reference: one dense solve of the whole bordered system.
            let want = h.llt(faer::Side::Lower).unwrap().solve(&rhs);

            // (b) the implemented path: factor M alone, multi-RHS solve for [y₀ | Y], eliminate.
            let m = Mat::<f64>::from_fn(dim, dim, |i, j| h[(i, j)]);
            let mut v_bord = vec![0.0_f64; dim * q];
            for i in 0..dim {
                for a in 0..q {
                    v_bord[i * q + a] = h[(i, dim + a)];
                }
            }
            let mut s_mat = [0.0_f64; Q_MAX * Q_MAX];
            let mut s_vec = [0.0_f64; Q_MAX];
            for a in 0..q {
                s_vec[a] = rhs[(dim + a, 0)];
                for b in 0..q {
                    s_mat[a * Q_MAX + b] = h[(dim + a, dim + b)];
                }
            }
            let multi = Mat::<f64>::from_fn(dim, 1 + q, |i, c| {
                if c == 0 {
                    rhs[(i, 0)]
                } else {
                    v_bord[i * q + (c - 1)]
                }
            });
            let sol = m.llt(faer::Side::Lower).unwrap().solve(&multi);
            let di = eliminate_border(&s_mat, &s_vec, &v_bord, &sol, dim, q);

            for a in 0..q {
                let got = di[a];
                let exp = want[(dim + a, 0)];
                assert!(
                    (got - exp).abs() <= 1e-9 * exp.abs().max(1e-6),
                    "q={q}: δi[{a}] = {got}, dense reference {exp}"
                );
            }
            // Through the SOLVER's own reconstruction, not a restatement of it.
            let dp = pose_step_from_border(&sol, &di, dim, q);
            for (i, &got) in dp.iter().enumerate() {
                let exp = want[(i, 0)];
                assert!(
                    (got - exp).abs() <= 1e-9 * exp.abs().max(1e-6),
                    "q={q}: δp[{i}] = {got}, dense reference {exp}"
                );
            }
        }

        // NO DENSIFICATION. `#1105`'s sparse path measured 64× faster at 600 views; a step-2 that
        // pushed the border into the reduced system's pattern would have discarded that. The
        // border lives outside `BlockAccum` entirely, so the pattern must be untouched.
        let cam = test_camera();
        let (_gt, poses_init, _pgt, points_init, obs, motion) = walkthrough(&cam, 10, 2);
        let mut triplets: Vec<usize> = Vec::new();
        for (refine_focal, refine_k1) in [(false, false), (true, true)] {
            let mut b_by_point: Vec<Vec<(usize, [f32; 18])>> = vec![Vec::new(); points_init.len()];
            for o in &obs {
                if !o.fixed_pose {
                    b_by_point[o.point_idx].push((o.pose_idx - 1, [0.0; 18]));
                }
            }
            let pairs: Vec<(usize, usize)> = motion
                .iter()
                .flat_map(|m| [(m.i0, m.i1), (m.i0, m.i2), (m.i1, m.i2)])
                .filter(|(a, b)| *a > 0 && *b > 0)
                .map(|(a, b)| (a - 1, b - 1))
                .collect();
            let acc = BlockAccum::new(poses_init.len() - 1, &b_by_point, &pairs);
            let _ = (refine_focal, refine_k1);
            triplets.push(acc.lower_triplets().len());
        }
        assert_eq!(
            triplets[0], triplets[1],
            "the border must not enter the reduced system's sparsity pattern"
        );
    }

    /// T6: the sparse/dense A/B of `sparse_reduced_system_matches_dense_on_a_sequential_capture`,
    /// with both intrinsics free.
    ///
    /// `V`, `S`, `s`, `S_red` and `δi` come off ONE shared code path — the border never enters
    /// either storage — so what differs between the two runs is only what already differed: `y₀`
    /// and `Y`, because a fill-reducing sparse Cholesky and a dense one do not perform the same
    /// operations in the same order.
    #[test]
    fn sparse_and_dense_agree_with_intrinsics_free() {
        let gt = test_camera();
        let (poses_gt, points_gt, obs) = intrinsics_only_scene(&gt, -0.12, 12, 2);
        let seed_cam = PinholeCamera {
            fx: 560.0,
            fy: 560.0,
            ..gt.clone()
        };
        let solve = |sparse: bool| {
            bundle_adjust_schur(
                &poses_gt,
                &points_gt,
                &obs,
                &seed_cam,
                &BaParams {
                    max_iterations: 30,
                    sparse_reduced_system: sparse,
                    refine_focal: true,
                    refine_k1: true,
                    ..BaParams::default()
                },
            )
            .expect("solve failed")
        };
        let dense = solve(false);
        let sparse = solve(true);
        let (cd, cs) = (
            dense.camera.clone().unwrap(),
            sparse.camera.clone().unwrap(),
        );

        // CONTROL: the intrinsics have to have MOVED, or "the two agree" is about two no-ops.
        assert!(
            (cd.fx - seed_cam.fx).abs() > 1.0 && (cd.k1 - 0.0).abs() > 0.01,
            "dense run left the intrinsics at the seed (fx {} k1 {}) — agreement proves nothing",
            cd.fx,
            cd.k1
        );

        let pose_gap = dense
            .poses
            .iter()
            .zip(sparse.poses.iter())
            .map(|(a, b)| (a.translation - b.translation).length())
            .fold(0.0_f64, f64::max);
        let point_gap = dense
            .points
            .iter()
            .zip(sparse.points.iter())
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0_f64, f64::max);
        assert!(
            pose_gap < 1e-6 && point_gap < 1e-6,
            "sparse and dense disagree: pose gap {pose_gap:e} m, point gap {point_gap:e} m"
        );
        assert!(
            (cd.fx - cs.fx).abs() < 1e-6 && (cd.k1 - cs.k1).abs() < 1e-6,
            "sparse and dense disagree on the intrinsics: fx {} vs {}, k1 {} vs {}",
            cd.fx,
            cs.fx,
            cd.k1,
            cs.k1
        );
    }

    /// T7: on a degenerate segment the focal must not run away, and must not fail either.
    ///
    /// A pure-rotation pair: no baseline, so no parallax, so the focal is unobservable — either
    /// `S_red` is singular and the fallback fires, or the step is taken and the ratio band stops
    /// it. Both outcomes are acceptable; a NaN camera, a `CholeskyFailed`, or a focal outside
    /// `[0.1, 10]×` the seed are not.
    #[test]
    fn focal_ratio_band_clamps_a_runaway() {
        let gt = test_camera();
        // Two cameras at the SAME centre, differing only by a yaw. Every point projects with zero
        // parallax, so depth — and with it the focal — is completely undetermined.
        let yaw = 0.15_f64;
        let (c, s) = (yaw.cos(), yaw.sin());
        let poses = vec![
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO),
            Pose3d::new(
                Mat3F64::from_cols(
                    Vec3F64::new(c, 0.0, -s),
                    Vec3F64::new(0.0, 1.0, 0.0),
                    Vec3F64::new(s, 0.0, c),
                ),
                Vec3F64::ZERO,
            ),
        ];
        let mut rng = Lcg(0x0bad_f0ca);
        let points: Vec<Vec3F64> = (0..40)
            .map(|_| {
                Vec3F64::new(
                    1.5 * rng.next_f32() as f64,
                    1.5 * rng.next_f32() as f64,
                    6.0 + 3.0 * rng.next_f32() as f64,
                )
            })
            .collect();
        let mut obs = Vec::new();
        for (pi, pt) in points.iter().enumerate() {
            for (k, pose) in poses.iter().enumerate() {
                let pc = pose.transform_point(pt);
                obs.push(BaObservation {
                    pose_idx: k,
                    point_idx: pi,
                    pixel: [
                        (gt.fx * pc.x / pc.z + gt.cx) as f32,
                        (gt.fy * pc.y / pc.z + gt.cy) as f32,
                    ],
                    fixed_pose: k == 0,
                    fixed_point: false,
                    ..BaObservation::default()
                });
            }
        }
        let seed_cam = PinholeCamera {
            fx: 700.0,
            fy: 700.0,
            ..gt.clone()
        };
        let res = bundle_adjust_schur(
            &poses,
            &points,
            &obs,
            &seed_cam,
            &BaParams {
                max_iterations: 40,
                refine_focal: true,
                refine_k1: true,
                ..BaParams::default()
            },
        )
        .expect("a degenerate segment must not become a CholeskyFailed");
        let cam = res.camera.clone().unwrap();
        assert!(
            cam.fx.is_finite() && cam.fy.is_finite() && cam.k1.is_finite(),
            "non-finite intrinsics: {cam:?}"
        );
        assert!(res.final_cost.is_finite());
        let ratio = cam.fx / seed_cam.fx;
        assert!(
            (MIN_FOCAL_RATIO as f64..=MAX_FOCAL_RATIO as f64).contains(&ratio),
            "focal ratio {ratio} left COLMAP's [{MIN_FOCAL_RATIO}, {MAX_FOCAL_RATIO}] band"
        );
    }

    /// The band must actually BIND, not merely be present.
    ///
    /// A seed 100× below the truth asks for `α = 100`, well outside COLMAP's `[0.1, 10]`. The
    /// solver must pin `α` at the ceiling and CARRY ON — clamp, do not reject. A clamped step is
    /// just a shorter step and `new_cost < cost` still decides it correctly, whereas rejecting on
    /// the band would escalate λ and stall the pose solve that was doing nothing wrong. So the
    /// fitted focal must land ON the ceiling, and the run must still have improved its cost.
    #[test]
    fn a_focal_seed_outside_the_band_is_clamped_not_rejected() {
        let gt = test_camera();
        let (poses, points, obs) = intrinsics_only_scene(&gt, 0.0, 10, 2);
        let seed_fx = gt.fx / 100.0; // asks for α = 100
        let seed_cam = PinholeCamera {
            fx: seed_fx,
            fy: seed_fx,
            ..gt.clone()
        };
        let params = |refine_focal: bool| BaParams {
            max_iterations: 30,
            refine_focal,
            ..BaParams::default()
        };
        let res = bundle_adjust_schur(&poses, &points, &obs, &seed_cam, &params(true)).unwrap();
        let cam = res.camera.clone().unwrap();
        let alpha = cam.fx / seed_fx;
        assert!(
            (alpha - MAX_FOCAL_RATIO as f64).abs() < 1e-4,
            "α came back {alpha}: the ratio band should have pinned it at {MAX_FOCAL_RATIO}"
        );
        // Clamped, not rejected: the run still had to make progress. (That the trimmed `δα` is also
        // what the geometry steps against is structural — `clamped_step` returns the step, so there
        // is no untrimmed one left to read.)
        let pinned = bundle_adjust_schur(&poses, &points, &obs, &seed_cam, &params(false)).unwrap();
        assert!(
            res.final_cost < 0.5 * pinned.final_cost && res.final_cost.is_finite(),
            "a clamped step must still be taken, and taken consistently: cost {} vs {} with the \
             focal pinned",
            res.final_cost,
            pinned.final_cost
        );
    }

    /// The robust kernel must down-weight an observation's vote on the INTRINSIC, not only on the
    /// geometry.
    ///
    /// The linearisation pass scales `r`, `J_pose` and `J_point` by `√w`; the border columns must
    /// take the same `√w` in the same block. If they do not, `g_intr` accumulates `√w·Jᵢᵀr` while
    /// the correct IRLS term is `w·Jᵢᵀr`, so an outlier the kernel rejected from the geometry still
    /// gets a `1/√w` over-loud vote on the focal it was rejected from having an opinion about.
    ///
    /// The outliers here are RADIALLY BIASED — every outlier pixel is pushed 30% further from the
    /// principal point — so they consistently ask for a larger focal rather than cancelling out.
    /// That is what turns a weighting bug into a measurable focal bias instead of noise.
    #[test]
    fn the_robust_kernel_down_weights_the_intrinsic_vote_too() {
        let gt = test_camera();
        let (poses, points, mut obs) = intrinsics_only_scene(&gt, 0.0, 10, 2);
        let clean = obs.len();
        for i in 0..clean {
            if i % 3 != 0 {
                continue;
            }
            let o = obs[i];
            obs.push(BaObservation {
                pixel: [
                    ((o.pixel[0] as f64 - gt.cx) * 1.3 + gt.cx) as f32,
                    ((o.pixel[1] as f64 - gt.cy) * 1.3 + gt.cy) as f32,
                ],
                ..o
            });
        }
        let seed_cam = PinholeCamera {
            fx: 520.0,
            fy: 520.0,
            ..gt.clone()
        };
        let res = bundle_adjust_schur(
            &poses,
            &points,
            &obs,
            &seed_cam,
            &BaParams {
                max_iterations: 40,
                robust: RobustKernelKind::Huber,
                robust_scale_sq: 4.0,
                refine_focal: true,
                ..BaParams::default()
            },
        )
        .unwrap();
        let fx = res.camera.clone().unwrap().fx;
        assert!(
            (fx - gt.fx).abs() < 0.02 * gt.fx,
            "focal came back {fx} against a ground truth of {} — the radially-biased outliers \
             pulled it, so their border columns are not carrying the IRLS weight",
            gt.fx
        );
    }

    /// λ must damp the BORDER as well as `A` and `C`.
    ///
    /// Leave the intrinsic direction at undamped Gauss-Newton while every other block is damped and
    /// iteration one takes a wild focal jump — which then gets rejected, which then escalates λ for
    /// everybody. Here: one iteration at λ = 1e6, where a damped border can move the focal by
    /// essentially nothing, and an undamped one takes its full GN step toward the true 500.
    #[test]
    fn lambda_damps_the_intrinsic_border() {
        let gt = test_camera();
        let (poses, points, obs) = intrinsics_only_scene(&gt, 0.0, 10, 2);
        let seed_fx = 575.0_f64;
        let seed_cam = PinholeCamera {
            fx: seed_fx,
            fy: seed_fx,
            ..gt.clone()
        };
        let step = |lambda: f32| -> f64 {
            let r = bundle_adjust_schur(
                &poses,
                &points,
                &obs,
                &seed_cam,
                &BaParams {
                    max_iterations: 1,
                    initial_lambda: lambda,
                    refine_focal: true,
                    ..BaParams::default()
                },
            )
            .unwrap();
            (r.camera.unwrap().fx - seed_fx).abs()
        };
        let stiff = step(1e6);
        let loose = step(1e-6);
        assert!(
            loose > 40.0,
            "control: at λ = 1e-6 one step should take the focal most of the way from {seed_fx} \
             to {}, but it moved {loose}",
            gt.fx
        );
        assert!(
            stiff < 1e-2 * loose,
            "λ = 1e6 moved the focal by {stiff} against {loose} at λ = 1e-6 — the border is not \
             being damped"
        );
    }

    /// `k1` must stay on the near side of the fold, from any seed and for any step.
    ///
    /// `1 + k1·r²` going non-positive is not a large error, it is a different function: at
    /// `k1 = −1/r²` that radius collapses onto the principal point and every derivative there is
    /// meaningless.
    ///
    /// WHAT THIS DOES NOT SHOW, because it was measured and is worth writing down: on every fixture
    /// in this file the clamp is not distinguishable from its absence at the RESULT. Seeded at
    /// `k1 = −50` it does bind — the first raw step is +25.3 and it is stretched to +46.6 so the
    /// trial lands exactly on `−0.9/r²_max` — but the unclamped solver climbs out of the fold on
    /// its own within a few iterations and returns the same `k1 = −0.150` at the same cost. So the
    /// end-to-end half of this test only pins "an absurd seed is not an error"; the guard proper is
    /// pinned as a PROPERTY of `clamped_step ∘ k1_fold_bound` below, and the regime where it is
    /// load-bearing is not yet measured.
    #[test]
    fn k1_is_held_on_the_near_side_of_the_fold() {
        // The bound, and the property that composes it with the step clamp: whatever raw step the
        // border produces, the trial `k1` lands where the radial term is still a function.
        assert_eq!(k1_fold_bound(0.9), K1_FOLD_MARGIN / 0.9);
        assert_eq!(k1_fold_bound(0.0), f32::INFINITY);
        for &r2 in &[0.05_f32, 0.27, 1.0, 4.0] {
            let hi = k1_fold_bound(r2);
            for &k1 in &[0.0_f32, -0.4, 0.9, -50.0, 1e6] {
                for &raw in &[0.0_f64, 1e-9, 3.0, -3.0, 1e9, -1e9] {
                    let k1_trial = k1 + clamped_step(k1, raw, -hi, hi) as f32;
                    // Slack of a few ulps OF `k1`, not of the bound: see `clamped_step`.
                    let slack = 8.0 * f32::EPSILON * k1.abs().max(hi);
                    assert!(
                        k1_trial.abs() <= hi + slack,
                        "k1={k1} raw={raw} r2={r2}: trial {k1_trial} escaped ±{hi}"
                    );
                    assert!(
                        1.0 + k1_trial * r2 >= 1.0 - K1_FOLD_MARGIN - slack * r2 - 1e-6,
                        "k1={k1} raw={raw} r2={r2}: 1 + k1·r² = {} is past the fold",
                        1.0 + k1_trial * r2
                    );
                }
            }
        }
        // And the focal band composes the same way.
        for &raw in &[1e9_f64, -1e9, 0.3] {
            let a = 1.0 + clamped_step(1.0, raw, MIN_FOCAL_RATIO, MAX_FOCAL_RATIO) as f32;
            assert!(
                (MIN_FOCAL_RATIO..=MAX_FOCAL_RATIO).contains(&a),
                "α escaped: {a}"
            );
        }

        let gt = test_camera();
        let (poses, points, obs) = intrinsics_only_scene(&gt, -0.15, 10, 2);
        let r2_max = obs
            .iter()
            .map(|o| normalised_r2(&pose_to_se3(&poses[o.pose_idx]), &points[o.point_idx]))
            .fold(0.0_f32, f32::max);
        assert!(r2_max > 0.0);

        let seed_cam = PinholeCamera {
            k1: -50.0,
            ..gt.clone()
        };
        let res = bundle_adjust_schur(
            &poses,
            &points,
            &obs,
            &seed_cam,
            &BaParams {
                max_iterations: 30,
                refine_k1: true,
                ..BaParams::default()
            },
        )
        .expect("an absurd k1 seed must not become an error");
        let k1 = res.camera.clone().unwrap().k1 as f32;
        assert!(k1.is_finite() && res.final_cost.is_finite());
        assert!(
            1.0 + k1 * r2_max > 0.0,
            "k1 came back {k1} with r²_max {r2_max}: 1 + k1·r² = {} is past the fold",
            1.0 + k1 * r2_max
        );
    }

    /// The unobservable-border fallback is a value decision, not an error path: `solve_border`
    /// must return `None` — never a panic and never a wild step — on a singular or non-positive
    /// `S_red`, and the driver turns that into `δi = 0`.
    #[test]
    fn a_singular_border_falls_back_instead_of_failing() {
        // Rank-1 2×2: two intrinsics that the data cannot tell apart.
        let s = [1.0_f64, 2.0, 2.0, 4.0];
        assert!(solve_border(&s, &[1.0, 2.0], 2).is_none());
        // Non-positive diagonal — not a curvature, so not a direction to step in.
        assert!(solve_border(&[0.0, 0.0, 0.0, 1.0], &[1.0, 1.0], 2).is_none());
        assert!(solve_border(&[-1.0, 0.0, 0.0, 1.0], &[1.0, 1.0], 2).is_none());
        assert!(solve_border(&[f64::NAN, 0.0, 0.0, 1.0], &[1.0, 1.0], 1).is_none());
        // Well-conditioned: the plain 2×2 solve.
        let x = solve_border(&[2.0, 0.0, 0.0, 4.0], &[6.0, 8.0], 2).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12);
        // q = 1 reads only the top-left entry.
        let x = solve_border(&[5.0, 9.0, 9.0, 9.0], &[10.0, 0.0], 1).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12);
    }
}
