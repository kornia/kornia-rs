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
//! Currently supports: identity loss only, fixed-pose anchors, fixed-point
//! gauge (motion-only BA). Robust kernels and full LM-with-backtracking
//! are TODO.

use faer::prelude::Solve;
use faer::Mat;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec3AF32, Vec3F64, SE3F32, SO3F32};
use thiserror::Error;

use crate::ba::{BaError, BaMotionPrior, BaObservation, BaParams, BaPosePrior, BaResult};
use crate::camera::PinholeCamera;
use crate::pose::Pose3d;
use crate::ransac::RobustKernelKind;

/// Bundle adjustments run, LM iterations summed over them, wall time, and the same split out for
/// LARGE systems (reduced dimension >= 1000).
///
/// These exist because this solve's cost was modelled three times from recorded figures and the
/// model was wrong every time — "two terminal BAs at a 100-iteration cap dominate" turned out to be
/// 52 iterations at one scale and 100-without-converging at another, and the factorisation everyone
/// assumed was the bottleneck measured 5.5% of the large adjustments' time. `BaResult` had carried
/// `iterations` and `converged` all along and nothing read them.
pub static BA_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Total LM iterations across every adjustment. See [`BA_CALLS`].
pub static BA_ITERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Summed `dim^3 / 1e6` across every adjustment — the shape of a dense Cholesky's cost, so its
/// distribution says whether factorisation work sits in one big solve or many small ones.
pub static BA_DIM_CUBED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Total microseconds spent inside bundle adjustment. See [`BA_CALLS`].
pub static BA_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Microseconds spent in adjustments with a reduced system of dimension >= 1000. See [`BA_CALLS`].
pub static BA_BIG_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// LM iterations in those same large adjustments. See [`BA_CALLS`].
pub static BA_BIG_ITERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Ceres' `min_lm_diagonal` / `max_lm_diagonal`: the LM damping diagonal is clamped to this
/// range, because extremely small or large entries of diag(JᵀJ) make the regularisation fail.
/// Per-phase microseconds inside the LM iteration, so "where does an iteration go" is answered
/// by measurement rather than by a cache model. Linearise = residuals + Jacobians + A/B/C/g;
/// assemble = damping + building the reduced camera system; factor = Cholesky + back-substitution;
/// trial = evaluating the objective at the trial point.
pub static BA_LIN_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// See [`BA_LIN_MICROS`].
pub static BA_ASM_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// See [`BA_LIN_MICROS`].
pub static BA_FACT_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// See [`BA_LIN_MICROS`].
pub static BA_TRIAL_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Observations actually seen by the adjustment (NOT the map-wide count).
pub static BA_OBS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

const MIN_LM_DIAGONAL: f64 = 1e-6;
const MAX_LM_DIAGONAL: f64 = 1e32;
const MIN_Z: f32 = 1e-3;

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
/// paper's bare `z − sm`: matching the derivative at the changeover keeps the cost C¹ there, so a
/// point crossing the image plane mid-iteration doesn't hand LM a step discontinuity. Behind the
/// camera the log is undefined and this pushes `z` back toward `sm` linearly.
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

// ── f32 ↔ f64 conversion helpers (shared shape with ba.rs) ───────────────

/// Infer the planarity prior's sigma from the trajectory's own HIGH-FREQUENCY out-of-plane motion.
///
/// The whole point is what this must NOT be estimated from. The global out-of-plane spread is the
/// drift the prior exists to remove, so setting sigma from it would hand the prior a tolerance equal
/// to the error and render it inert — the identical circularity that makes the loop-closure gauge
/// absorb trajectory drift as scale. A prior's strength can never be estimated from the quantity it
/// is meant to correct.
///
/// The two components separate by FREQUENCY. A gait bob (or a robot's suspension) is local and
/// high-frequency; drift is global and low-frequency. Detrending each camera against the median of
/// its `w` neighbours IN CAPTURE ORDER removes the bend and leaves the bob, and a robust MAD scale
/// of that residual is the physical variation the capture actually exhibits.
///
/// Measured on two real maps. A 643-keyframe house walk carrying 0.89 m of drift returns 0.028 map
/// units — 3.9 cm at that map's gauge, which is exactly a human's bob — while its global spread is
/// 0.633, a factor of 22. A 40-keyframe walk with no drift returns 0.062 against a global spread of
/// 0.037: LARGER than the deviations it would penalise, so the prior goes slack and does nothing.
/// That self-disabling behaviour is what makes inference safe to enable by default — a good capture
/// infers a tolerance it never exceeds, and only a drifted one gets pulled.
///
/// Ordering assumption: pose index is capture order, which holds for video keyframes. If it does not,
/// the window averages unrelated cameras and the estimate degrades toward the global spread — slack,
/// therefore harmless, never over-tight.
fn infer_plane_sigma(
    se3s: &[SE3F32],
    pose_local: &[i64],
    nrm: &[f64; 3],
    ctr: &[f64; 3],
) -> Option<f64> {
    let d: Vec<f64> = se3s
        .iter()
        .enumerate()
        .filter(|(i, _)| pose_local.get(*i).copied().unwrap_or(-1) >= 0)
        .map(|(_, p)| {
            let rm = p.r.matrix();
            let t = p.t;
            let (c0, c1, c2) = (rm.col(0), rm.col(1), rm.col(2));
            let c = [
                -f64::from(c0.x * t.x + c0.y * t.y + c0.z * t.z),
                -f64::from(c1.x * t.x + c1.y * t.y + c1.z * t.z),
                -f64::from(c2.x * t.x + c2.y * t.y + c2.z * t.z),
            ];
            (0..3).map(|k| nrm[k] * (c[k] - ctr[k])).sum::<f64>()
        })
        .collect();
    let n = d.len();
    if n < 12 {
        return None;
    }
    // Window small relative to the trajectory: at n = 40 a +-15 window spans three quarters of the
    // walk and detrends away real signal. A quarter of the trajectory, capped, keeps it local.
    let w = (n / 8).clamp(2, 15);
    let median = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    };
    let mut resid: Vec<f64> = (0..n)
        .map(|i| {
            let lo = i.saturating_sub(w);
            let hi = (i + w + 1).min(n);
            let mut win = d[lo..hi].to_vec();
            d[i] - median(&mut win)
        })
        .collect();
    let med = median(&mut resid.clone());
    let mut dev: Vec<f64> = resid.iter().map(|x| (x - med).abs()).collect();
    let sigma = 1.4826 * median(&mut dev);
    resid.clear();
    (sigma.is_finite() && sigma > 1e-9).then_some(sigma)
}

/// Best-fit plane through the camera centres of the poses being optimised.
///
/// Returns `(unit normal, centroid)`, or `None` when fewer than four free poses exist or the
/// centres are too close to collinear for a normal to mean anything — in both cases there is no
/// plane to speak of and the prior must not invent one.
///
/// Re-fitted every LM iteration and treated as CONSTANT within it (the centroid is not
/// differentiated through). Differentiating the fit would be exact but couples every pose to every
/// other, destroying the block structure the Schur reduction exists to exploit; the alternating form
/// converges because the plane moves far less than the poses do.
fn fit_centre_plane(se3s: &[SE3F32], pose_local: &[i64]) -> Option<([f64; 3], [f64; 3])> {
    let mut cs: Vec<[f64; 3]> = Vec::new();
    for (i, p) in se3s.iter().enumerate() {
        if pose_local.get(i).copied().unwrap_or(-1) < 0 {
            continue;
        }
        let rm = p.r.matrix();
        let t = p.t;
        let (c0, c1, c2) = (rm.col(0), rm.col(1), rm.col(2));
        cs.push([
            -f64::from(c0.x * t.x + c0.y * t.y + c0.z * t.z),
            -f64::from(c1.x * t.x + c1.y * t.y + c1.z * t.z),
            -f64::from(c2.x * t.x + c2.y * t.y + c2.z * t.z),
        ]);
    }
    if cs.len() < 4 {
        return None;
    }
    let n = cs.len() as f64;
    let mut mean = [0.0f64; 3];
    for c in &cs {
        for k in 0..3 {
            mean[k] += c[k] / n;
        }
    }
    let mut cov = [[0.0f64; 3]; 3];
    for c in &cs {
        let d = [c[0] - mean[0], c[1] - mean[1], c[2] - mean[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j] / n;
            }
        }
    }
    // Jacobi rotation to diagonal. Three axes, so a fixed sweep budget is exact to machine
    // precision and keeps this dependency-free.
    let mut a = cov;
    let mut v = [[0.0f64; 3]; 3];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let (mut p, mut q) = (0usize, 1usize);
        if a[0][2].abs() > a[p][q].abs() {
            (p, q) = (0, 2);
        }
        if a[1][2].abs() > a[p][q].abs() {
            (p, q) = (1, 2);
        }
        if a[p][q].abs() < 1e-18 {
            break;
        }
        let th = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (c, sn) = (th.cos(), th.sin());
        for k in 0..3 {
            let (akp, akq) = (a[k][p], a[k][q]);
            a[k][p] = c * akp - sn * akq;
            a[k][q] = sn * akp + c * akq;
        }
        for k in 0..3 {
            let (apk, aqk) = (a[p][k], a[q][k]);
            a[p][k] = c * apk - sn * aqk;
            a[q][k] = sn * apk + c * aqk;
        }
        for row in v.iter_mut() {
            let (vp, vq) = (row[p], row[q]);
            row[p] = c * vp - sn * vq;
            row[q] = sn * vp + c * vq;
        }
    }
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| {
        a[i][i]
            .partial_cmp(&a[j][j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (lo, mid, hi) = (idx[0], idx[1], idx[2]);
    // Degeneracy guard on the SECOND spread, not the smallest. Collinear centres — a straight
    // corridor, or any short walk — have TWO near-zero eigenvalues, and every plane through the line
    // fits them equally well; the "normal" is then whichever direction numerical noise picks, and
    // penalising deviation from it injects a constraint the data never supported. Testing the
    // smallest eigenvalue instead accepts exactly that case, since a collinear set is maximally
    // "flat" by that measure. An isotropic blob (lo/hi near 1) is rejected too: there is no plane
    // to speak of.
    if a[hi][hi] <= 1e-18 || a[mid][mid] / a[hi][hi] < 1e-4 || a[lo][lo] / a[hi][hi] > 0.98 {
        return None;
    }
    let nvec = [v[0][lo], v[1][lo], v[2][lo]];
    let len = (nvec[0] * nvec[0] + nvec[1] * nvec[1] + nvec[2] * nvec[2]).sqrt();
    (len > 1e-12).then(|| ([nvec[0] / len, nvec[1] / len, nvec[2] / len], mean))
}

/// 6-vector residual of one motion prior on the CURRENT pose estimates.
///
/// Layout: `[t_ratio, 0, 0, w01 − α·w02] / σ` (ratio branch) or
/// `[α(C2−C0) − (C1−C0), w01 − α·w02] / σ` when `C0 ≈ C2`. `C` are camera CENTRES (−Rᵀt), the
/// physically meaningful quantity; `w` are SO(3) logs of the relative rotations.
fn motion_prior_residual(p0: &SE3F32, p1: &SE3F32, p2: &SE3F32, mp: &BaMotionPrior) -> [f32; 6] {
    let centre = |p: &SE3F32| -> [f32; 3] {
        let rm = p.r.matrix();
        let t = p.t;
        let (c0, c1, c2) = (rm.col(0), rm.col(1), rm.col(2));
        [
            -(c0.x * t.x + c0.y * t.y + c0.z * t.z),
            -(c1.x * t.x + c1.y * t.y + c1.z * t.z),
            -(c2.x * t.x + c2.y * t.y + c2.z * t.z),
        ]
    };
    // SO(3) log of Ra · Rb^T, via the axis-angle (Rodrigues) formula on the composed matrix.
    let rel_log = |a: &SE3F32, b: &SE3F32| -> [f32; 3] {
        let (ra, rb) = (a.r.matrix(), b.r.matrix());
        // m = Ra · Rb^T  — element (i,j) = sum_k Ra[i,k]·Rb[j,k]
        let get = |m: &kornia_algebra::Mat3AF32, i: usize, j: usize| -> f32 {
            let c = m.col(j);
            match i {
                0 => c.x,
                1 => c.y,
                _ => c.z,
            }
        };
        let mut m = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut sacc = 0.0;
                for k in 0..3 {
                    sacc += get(&ra, i, k) * get(&rb, j, k);
                }
                m[i][j] = sacc;
            }
        }
        let tr = (m[0][0] + m[1][1] + m[2][2]).clamp(-1.0, 3.0);
        let cos_t = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0);
        let theta = cos_t.acos();
        if theta < 1e-6 {
            return [
                0.5 * (m[2][1] - m[1][2]),
                0.5 * (m[0][2] - m[2][0]),
                0.5 * (m[1][0] - m[0][1]),
            ];
        }
        let k = 0.5 * theta / theta.sin().max(1e-9);
        [
            k * (m[2][1] - m[1][2]),
            k * (m[0][2] - m[2][0]),
            k * (m[1][0] - m[0][1]),
        ]
    };

    let (c0, c1, c2) = (centre(p0), centre(p1), centre(p2));
    let d01 = [c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2]];
    let d02 = [c2[0] - c0[0], c2[1] - c0[1], c2[2] - c0[2]];
    let n01 = (d01[0] * d01[0] + d01[1] * d01[1] + d01[2] * d01[2]).sqrt();
    let n02 = (d02[0] * d02[0] + d02[1] * d02[1] + d02[2] * d02[2]).sqrt();
    let inv_sp = 1.0 / mp.position_sigma.max(1e-6);
    let inv_so = 1.0 / mp.orientation_sigma.max(1e-6);

    let mut r = [0.0f32; 6];
    if n02 > 1e-6 {
        r[0] = (mp.alpha - n01 / n02) * inv_sp;
    } else {
        // Stationary endpoints: fall back to the position difference (no scale in play).
        r[0] = (mp.alpha * d02[0] - d01[0]) * inv_sp;
        r[1] = (mp.alpha * d02[1] - d01[1]) * inv_sp;
        r[2] = (mp.alpha * d02[2] - d01[2]) * inv_sp;
    }
    let w01 = rel_log(p1, p0);
    let w02 = rel_log(p2, p0);
    r[3] = (w01[0] - mp.alpha * w02[0]) * inv_so;
    r[4] = (w01[1] - mp.alpha * w02[1]) * inv_so;
    r[5] = (w01[2] - mp.alpha * w02[2]) * inv_so;
    r
}

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

/// Computes (residual, J_pose 2×6, J_point 2×3) at the current state.
/// Returns the camera-frame point and the clamped z too, for back-substitution
/// reasoning.
///
/// Jacobian layout (row-major flat):
///   J_pose[0..6]:  [du/dρ_x, du/dρ_y, du/dρ_z, du/dω_x, du/dω_y, du/dω_z]
///   J_pose[6..12]: [dv/dρ_x, dv/dρ_y, dv/dρ_z, dv/dω_x, dv/dω_y, dv/dω_z]
///   J_point[0..3]: [du/dx,   du/dy,   du/dz]
///   J_point[3..6]: [dv/dx,   dv/dy,   dv/dz]
fn residual_and_jacobians(
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

    // J_proj row coefficients (∂[u; v] / ∂[X_c]).
    let a0 = fx * inv_z;
    let a2 = -fx * pc.x * inv_z2;
    let b1 = fy * inv_z;
    let b2 = -fy * pc.y * inv_z2;

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

    // J_pt = J_proj · R (3 cols).
    let jpt_00 = a0 * r00 + a2 * r20;
    let jpt_01 = a0 * r01 + a2 * r21;
    let jpt_02 = a0 * r02 + a2 * r22;
    let jpt_10 = b1 * r10 + b2 * r20;
    let jpt_11 = b1 * r11 + b2 * r21;
    let jpt_12 = b1 * r12 + b2 * r22;

    // J_omega = J_proj · S (3 cols).
    let jom_00 = a0 * s00 + a2 * s20;
    let jom_01 = a0 * s01 + a2 * s21;
    let jom_02 = a0 * s02 + a2 * s22;
    let jom_10 = b1 * s10 + b2 * s20;
    let jom_11 = b1 * s11 + b2 * s21;
    let jom_12 = b1 * s12 + b2 * s22;

    // Layout J_pose 2×6 row-major: [ρ(3) | ω(3)] per row.
    let j_pose: [f32; 12] = [
        jpt_00, jpt_01, jpt_02, jom_00, jom_01, jom_02, jpt_10, jpt_11, jpt_12, jom_10, jom_11,
        jom_12,
    ];
    // J_point 2×3 row-major.
    let j_point: [f32; 6] = [jpt_00, jpt_01, jpt_02, jpt_10, jpt_11, jpt_12];

    (r, j_pose, j_point)
}

// ── Small block primitives (f32) ─────────────────────────────────────────

#[inline]
fn ata_6x6_into(acc: &mut [f64; 36], j: &[f32; 12]) {
    // acc += J.T @ J  where J is 2×6 row-major.
    let r0 = &j[0..6];
    let r1 = &j[6..12];
    for i in 0..6 {
        for k in 0..6 {
            acc[i * 6 + k] +=
                f64::from(r0[i]) * f64::from(r0[k]) + f64::from(r1[i]) * f64::from(r1[k]);
        }
    }
}

#[inline]
fn ata_3x3_into(acc: &mut [f64; 9], j: &[f32; 6]) {
    let r0 = &j[0..3];
    let r1 = &j[3..6];
    for i in 0..3 {
        for k in 0..3 {
            acc[i * 3 + k] +=
                f64::from(r0[i]) * f64::from(r0[k]) + f64::from(r1[i]) * f64::from(r1[k]);
        }
    }
}

#[inline]
fn atb_6x3_into(acc: &mut [f64; 18], jp: &[f32; 12], jx: &[f32; 6]) {
    // acc += J_pose.T @ J_point  →  6 × 3 row-major.
    let jp0 = &jp[0..6];
    let jp1 = &jp[6..12];
    let jx0 = &jx[0..3];
    let jx1 = &jx[3..6];
    for i in 0..6 {
        for k in 0..3 {
            acc[i * 3 + k] +=
                f64::from(jp0[i]) * f64::from(jx0[k]) + f64::from(jp1[i]) * f64::from(jx1[k]);
        }
    }
}

#[inline]
fn atb_6x1_into(acc: &mut [f64; 6], j: &[f32; 12], r: &[f32; 2]) {
    // acc -= J.T @ r  (note negative for gradient convention).
    for i in 0..6 {
        acc[i] -= f64::from(j[i]) * f64::from(r[0]) + f64::from(j[6 + i]) * f64::from(r[1]);
    }
}

#[inline]
fn atb_3x1_into(acc: &mut [f64; 3], j: &[f32; 6], r: &[f32; 2]) {
    for i in 0..3 {
        acc[i] -= f64::from(j[i]) * f64::from(r[0]) + f64::from(j[3 + i]) * f64::from(r[1]);
    }
}

/// Invert a 3×3 row-major matrix. Returns None if singular.
fn invert_3x3(m: &[f64; 9]) -> Option<[f64; 9]> {
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
fn matmul_6x3_3x3(a: &[f64; 18], b: &[f64; 9]) -> [f64; 18] {
    let mut out = [0.0_f64; 18];
    for i in 0..6 {
        for k in 0..3 {
            let mut s = 0.0_f64;
            for r in 0..3 {
                s += a[i * 3 + r] * b[r * 3 + k];
            }
            out[i * 3 + k] = s;
        }
    }
    out
}

#[inline]
fn matvec_6x3_3(a: &[f64; 18], b: &[f64; 3]) -> [f64; 6] {
    let mut out = [0.0_f64; 6];
    for i in 0..6 {
        out[i] = a[i * 3] * b[0] + a[i * 3 + 1] * b[1] + a[i * 3 + 2] * b[2];
    }
    out
}

#[inline]
fn matvec_3x3_3(a: &[f64; 9], b: &[f64; 3]) -> [f64; 3] {
    [
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
        a[3] * b[0] + a[4] * b[1] + a[5] * b[2],
        a[6] * b[0] + a[7] * b[1] + a[8] * b[2],
    ]
}

#[inline]
fn matvec_6x3t_6(a: &[f64; 18], b: &[f64; 6]) -> [f64; 3] {
    // returns a.T @ b  →  3-vector; a is stored row-major 6×3
    let mut out = [0.0_f64; 3];
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

// ── Driver ───────────────────────────────────────────────────────────────

/// Bundle adjustment via dense Schur-complement reduction. Same external
/// contract as [`crate::ba::bundle_adjust`] but uses Schur internally:
/// the reduced 6P×6P camera system is solved with `faer`'s dense Cholesky;
/// points are recovered by back-substitution.
///
/// Respects the `fixed_pose` and `fixed_point` flags on each observation, and honours
/// `BaParams::robust` (IRLS: residual and Jacobian rows scaled by √ρ'(s), while the accept/reject
/// test compares the true robust cost ½ρ(s)), plus `max_iterations`, `initial_lambda` and
/// `cost_tolerance`.
///
/// `BaParams::gradient_tolerance` is NOT read by this solver. The only termination test is the
/// relative cost decrease against `cost_tolerance`, and it is checked on accepted steps only.
/// Callers wanting COLMAP's gradient-based stopping rule do not get it here.
pub fn bundle_adjust_schur(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera: &PinholeCamera,
    params: &BaParams,
) -> Result<BaResult, SchurBaError> {
    bundle_adjust_schur_with_priors(poses, points, observations, camera, params, None)
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
pub fn bundle_adjust_schur_with_priors(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera: &PinholeCamera,
    params: &BaParams,
    pose_priors: Option<&[Option<BaPosePrior>]>,
) -> Result<BaResult, SchurBaError> {
    bundle_adjust_schur_with_all_priors(
        poses,
        points,
        observations,
        camera,
        params,
        pose_priors,
        None,
    )
}

/// [`bundle_adjust_schur_with_priors`] plus constant-velocity motion priors over shot triplets
/// (see [`BaMotionPrior`]).
///
/// Motion residuals couple THREE pose blocks, so unlike every other residual family here they
/// contribute off-diagonal pose-pose blocks to the reduced camera system. Their Jacobians are
/// obtained by finite differences over the solver's own retraction — the residual (a norm ratio
/// composed with an SO(3) log) has an unpleasant closed form, the FD cost is negligible
/// (~tens of triplets × 19 residual evaluations), and using `retract` itself guarantees the
/// perturbation convention can never drift out of sync with the analytic residuals.
#[allow(clippy::too_many_arguments)]

/// Block-sparse accumulator for the reduced camera system.
///
/// The reduced system is 6Px6P but only ~2% populated: two cameras couple only if they share a
/// point. Materialising it densely costs 117 MB at P=637 and, worse, makes every 6x6 block write
/// touch six columns `ld * 8` bytes apart — 36 cache lines on six pages, per block, with no reuse.
/// Measured on a 300-keyframe solve: ~770k blocks, ~28M line touches, against 0.02 s of arithmetic.
/// The assembly ran ~70x its flop cost and all of it was memory.
///
/// Here each 6x6 block is 288 contiguous bytes, found through a dense index rather than a hash, and
/// the dense matrix is never allocated at all on the sparse path.
///
/// NOTE the failure mode this is written to avoid: an earlier attempt accumulated into compact
/// blocks and then SCATTERED them into the dense matrix, which performed every one of the original
/// scattered writes plus the block pass on top — measured 28% slower. Compacting the accumulation
/// only pays if the dense matrix is never built.

/// Resolve the assembly thread count: `KORNIA_BA_THREADS` overrides `BaParams::assembly_threads`,
/// which in turn overrides "ask the machine". Always at least 1.
///
/// The env override exists because the right number is a property of the BOX, not the caller — on
/// a 6-core Jetson sharing the die with the CUDA front-end, 6 is not automatically the fastest
/// setting, and finding that out should not require rebuilding every downstream binary.
/// One camera's RHS contribution from a point: `(local pose index, 6-vector)`.
type RhsContrib = (usize, [f64; 6]);
/// One camera-pair's 6x6 contribution: `(flat pair key, block)`. The key is
/// `i1 * n_free_poses + i2`, so the reduction needs no index lookup to decode it.
type BlockContrib = (usize, [f64; 36]);
/// What one chunk of points contributes to the reduced system.
type ChunkContrib = (Vec<RhsContrib>, Vec<BlockContrib>);

fn assembly_threads(params: &BaParams) -> usize {
    if let Some(v) = std::env::var("KORNIA_BA_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return v.max(1);
    }
    if params.assembly_threads > 0 {
        return params.assembly_threads;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

struct BlockAccum {
    /// Row-major `(i1, i2) -> slot`, or `usize::MAX` for a pair that never couples. `n * n` entries
    /// of `usize` is 3.2 MB at P=637 against the 117 MB it replaces, and a direct index beats a hash
    /// lookup in a loop that runs ~770k times per iteration.
    index: Vec<usize>,
    /// Flat 6x6 blocks, one per coupled pair.
    blocks: Vec<[f64; 36]>,
    /// `(i1, i2)` for each slot, for emitting triplets without scanning the index.
    pairs: Vec<(usize, usize)>,
    n: usize,
}

impl BlockAccum {
    /// Build the coupling pattern once. It is a function of which cameras share a point — plus any
    /// pair coupled by a MOTION PRIOR, which no observation need connect.
    ///
    /// Covisibility alone is not the whole pattern. A constant-velocity prior couples a triplet
    /// whether or not those cameras see common structure, and when they do not, the block it writes
    /// has no slot and the factorisation fails outright:
    /// `CholeskyFailed("motion prior couples cameras with no shared point")`.
    ///
    /// That is not a corner case, it is what coarser sampling produces. Measured on a 3211-frame
    /// upload: at 6 Hz every consecutive keyframe pair shared points and the bug never fired; at
    /// 3 Hz — the rate that makes a 570-keyframe solve tractable at all — some consecutive pairs
    /// stopped overlapping and every candidate died here, at 296 of 322 cameras already registered.
    ///
    /// Dropping such priors instead would be worse than the failure: when two consecutive keyframes
    /// share NO point, the motion prior is the only thing connecting them, so discarding it severs
    /// the pose graph exactly where the geometry is weakest.
    fn new(
        n: usize,
        b_by_point: &[Vec<(usize, [f64; 18])>],
        motion_pairs: &[(usize, usize)],
    ) -> Self {
        let mut index = vec![usize::MAX; n * n];
        let mut pairs = Vec::new();
        for b_for_j in b_by_point {
            for (i1, _) in b_for_j.iter() {
                for (i2, _) in b_for_j.iter() {
                    let e = &mut index[i1 * n + i2];
                    if *e == usize::MAX {
                        *e = pairs.len();
                        pairs.push((*i1, *i2));
                    }
                }
            }
        }
        // Motion-prior couplings, added whether or not the cameras share structure. Both triangles:
        // the accumulator is addressed by (row, col) and the prior writes in one order while the
        // symmetric read may come in the other.
        for &(a, b) in motion_pairs {
            if a >= n || b >= n {
                continue;
            }
            for (i1, i2) in [(a, b), (b, a)] {
                let e = &mut index[i1 * n + i2];
                if *e == usize::MAX {
                    *e = pairs.len();
                    pairs.push((i1, i2));
                }
            }
        }
        // Diagonal blocks always exist: every free camera carries its own A block.
        for i in 0..n {
            let e = &mut index[i * n + i];
            if *e == usize::MAX {
                *e = pairs.len();
                pairs.push((i, i));
            }
        }
        let blocks = vec![[0.0; 36]; pairs.len()];
        Self {
            index,
            blocks,
            pairs,
            n,
        }
    }

    fn clear(&mut self) {
        for b in self.blocks.iter_mut() {
            *b = [0.0; 36];
        }
    }

    /// Slot for a pair, or `None` if these two cameras never share a point.
    #[inline]
    fn slot(&self, i1: usize, i2: usize) -> Option<usize> {
        let s = self.index[i1 * self.n + i2];
        (s != usize::MAX).then_some(s)
    }

    /// Lower-triangle triplets, summed on duplicates by faer exactly as the dense path summed into
    /// the matrix. Only the lower triangle is emitted, which is all `Side::Lower` reads.
    fn triplets(&self) -> Vec<faer::sparse::Triplet<usize, usize, f64>> {
        let mut t = Vec::with_capacity(self.blocks.len() * 21);
        for (slot, &(i1, i2)) in self.pairs.iter().enumerate() {
            let blk = &self.blocks[slot];
            let (row0, col0) = (i1 * 6, i2 * 6);
            for r in 0..6 {
                for c in 0..6 {
                    let v = blk[r * 6 + c];
                    if v != 0.0 && row0 + r >= col0 + c {
                        t.push(faer::sparse::Triplet::new(row0 + r, col0 + c, v));
                    }
                }
            }
        }
        t
    }
}

/// [`bundle_adjust_schur_with_priors`] plus constant-velocity motion priors over shot triplets
/// (see [`BaMotionPrior`]).
///
/// Motion residuals couple THREE pose blocks, so unlike every other residual family here they
/// contribute off-diagonal pose-pose blocks to the reduced camera system. Their Jacobians are
/// obtained by finite differences over the solver's own retraction — the residual (a norm ratio
/// composed with an SO(3) log) has an unpleasant closed form, the FD cost is negligible
/// (~tens of triplets × 19 residual evaluations), and using `retract` itself guarantees the
/// perturbation convention can never drift out of sync with the analytic residuals.
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
    let ba_t0 = std::time::Instant::now();
    let n_free_poses = pose_local.iter().filter(|&&x| x >= 0).count();
    let n_free_points = point_local.iter().filter(|&&x| x >= 0).count();

    if n_free_poses == 0 {
        return Err(SchurBaError::NoFreeVariables);
    }

    // Mutable state.
    let mut se3s: Vec<SE3F32> = poses.iter().map(pose_to_se3).collect();
    let mut xyz: Vec<Vec3F64> = points.to_vec();

    // Diagnostic trace (env-gated, no behaviour change). Read once: this is a hot loop.
    let trace_on = std::env::var("KORNIA_BA_TRACE").is_ok();

    let mut lambda = params.initial_lambda;
    // Nielsen's ν for the reject branch: consecutive rejections escalate the damping bump
    // (2, 4, 8, …) instead of a fixed ×10, so one bad linearization does not overshoot λ past
    // the useful range and cost the next several iterations undoing it.
    let mut nu = 2.0_f32;
    let mut prev_cost: Option<f64> = None;
    // Block-sparse reduced system, built on the first iteration when the sparse path is enabled.
    let mut accum: Option<BlockAccum> = None;
    let mut iters_done = 0usize;
    let mut converged = false;

    // ── Per-camera depth scales ─────────────────────────────────────────────
    // Seeded from `depth_scales_init` (a robust median fit is the intended seed) so the log
    // residual starts near its optimum. Only meaningful in log mode; the legacy residual ignores
    // them, and an unset/short init vector pads with 1.0 rather than failing.
    let log_depth = params.depth_log_residual;
    let mut dscales = vec![1.0_f32; p_total];
    for (i, s) in params.depth_scales_init.iter().take(p_total).enumerate() {
        if *s > 0.0 && s.is_finite() {
            dscales[i] = *s;
        }
    }
    // Robust weight for depth residuals, hoisted out of the loop so the scale update and the
    // linearisation gate observations identically.
    let depth_knee = if params.depth_robust_scale_sq > 0.0 {
        params.depth_robust_scale_sq.sqrt()
    } else {
        params.robust_scale_sq.sqrt().max(1e-6)
    };
    let depth_robust_w: Box<dyn Fn(f32) -> f32> = match params.robust {
        RobustKernelKind::Identity => Box::new(|_| 1.0),
        RobustKernelKind::Huber => Box::new(move |r_sq: f32| {
            let r = r_sq.sqrt();
            if r <= depth_knee {
                1.0
            } else {
                depth_knee / r
            }
        }),
        RobustKernelKind::Cauchy | RobustKernelKind::Tukey => Box::new(move |r_sq: f32| {
            let s2 = depth_knee * depth_knee;
            s2 / (s2 + r_sq)
        }),
    };

    for _iter in 0..params.max_iterations {
        iters_done += 1;

        // Exact block update of the depth scales at the current geometry, before linearising the
        // rest. A negative `depth_scale_prior` freezes them at the seed — the fitted-then-frozen
        // baseline this whole mechanism exists to beat.
        if log_depth && params.depth_scale_prior >= 0.0 {
            update_depth_scales(
                &mut dscales,
                &se3s,
                &xyz,
                observations,
                params.depth_scale_prior,
                &depth_robust_w,
            );
        }

        // ── Linearise: build A, C, B (per-obs), g_pose, g_point ──────────
        // A: n_free_poses × [36] (6×6 blocks).
        // C: n_free_points × [9]  (3×3 blocks).
        // We also keep observation-aligned B blocks (6×3) so we can iterate
        // by point during the Schur reduction.
        let mut a_blocks = vec![[0.0_f64; 36]; n_free_poses];
        let mut c_blocks = vec![[0.0_f64; 9]; n_free_points];
        let mut g_pose = vec![[0.0_f64; 6]; n_free_poses];
        let mut g_point = vec![[0.0_f64; 3]; n_free_points];

        // Per-observation B contributions, grouped by point (for the Schur
        // pass). We store (pose_local_idx, B_6x3) lists per free-point index.
        let mut b_by_point: Vec<Vec<(usize, [f64; 18])>> = vec![Vec::new(); n_free_points];

        // Also record observations that touch FIXED point but FREE pose —
        // contribute to A and g_pose only, no B.
        // (Symmetric case: free point + fixed pose contributes to C and
        //  g_point only. Both we handle below.)
        // Robust-loss IRLS weight per observation. weight w = min(1, scale/‖r‖)
        // for Huber, w = scale²/(scale²+‖r‖²) for Cauchy. Identity uses w=1.
        // Apply √w to both residual and Jacobian rows (equivalent to multiplying
        // the obs's contribution to the normal equations by w).
        // A non-finite or non-positive scale collapses to plain L2 rather than producing NaN.
        // `robust_scale_sq` DEFAULTS to `f32::INFINITY`, and with Cauchy that gave weight
        // inf/inf = NaN and cost 0.5·inf·ln(1) = NaN; `NaN < cost` is false forever, so every
        // step was rejected and the solver returned its input poses with `Ok`.
        // `ba::build_robust_loss` has always guarded this; this path did not.
        let robust = if params.robust_scale_sq.is_finite() && params.robust_scale_sq > 0.0 {
            params.robust
        } else {
            RobustKernelKind::Identity
        };
        let robust_scale = params.robust_scale_sq.sqrt().max(1e-6);
        let huber_w = |r_sq: f32| -> f32 {
            // ‖r‖ ≤ scale → w=1; else w = scale/‖r‖
            let r_norm = r_sq.sqrt();
            if r_norm <= robust_scale {
                1.0
            } else {
                robust_scale / r_norm
            }
        };
        let cauchy_w = |r_sq: f32| -> f32 {
            let s2 = robust_scale * robust_scale;
            s2 / (s2 + r_sq)
        };
        // Depth residuals are σ-whitened (1.0 == one standard deviation) while reprojection
        // residuals are in normalized-camera units — a shared knee treats a 1σ depth measurement
        // as a gross outlier. See `BaParams::depth_robust_scale_sq`.
        let depth_scale = if params.depth_robust_scale_sq > 0.0 {
            params.depth_robust_scale_sq.sqrt()
        } else {
            robust_scale
        };
        let huber_w_depth = |r_sq: f32| -> f32 {
            let r_norm = r_sq.sqrt();
            if r_norm <= depth_scale {
                1.0
            } else {
                depth_scale / r_norm
            }
        };
        let cauchy_w_depth = |r_sq: f32| -> f32 {
            let s2 = depth_scale * depth_scale;
            s2 / (s2 + r_sq)
        };
        // The TRUE robust cost ½·ρ(s) — the objective the weights above are the derivative of
        // (w = ρ'(s), which is exactly what makes IRLS solve the robust problem).
        //
        // The √w-scaled residual accumulates ½·ρ'(s)·s instead, which is NOT that objective. For
        // Huber past the knee the surrogate is ½k‖r‖ while the real loss is k‖r‖ − k²/2, so it
        // falls at HALF the true rate on every downweighted observation. The gain ratio
        // ρ = actual/predicted then divides a halved actual reduction by an unhalved model
        // prediction and comes out systematically deflated. Nielsen shrinks λ only when ρ > 0.5,
        // so once enough observations sit past the knee λ can never decay, LM never hands back
        // over to Gauss-Newton, and convergence degrades from quadratic to linear. Measured on a
        // 300-keyframe solve: healthy for 10 iterations (ρ≈0.85, λ 9.6e-4→5.9e-5), then ρ pinned
        // at 0.4157 for the remaining 90 while λ ratcheted back up to 1.4e-4. Ceres compares this
        // function across a trial step, not the surrogate.
        let robust_cost = |r_sq: f32, scale: f32| -> f32 {
            match robust {
                RobustKernelKind::Identity => 0.5 * r_sq,
                RobustKernelKind::Huber => {
                    // Knee as `s <= scale²`, matching `HuberLoss::weight`; `sqrt(s) <= scale`
                    // rounds differently in f32 within an ulp of the knee.
                    if r_sq <= scale * scale {
                        0.5 * r_sq
                    } else {
                        scale * r_sq.sqrt() - 0.5 * scale * scale
                    }
                }
                // Tukey shares Cauchy's weight here, so it shares Cauchy's loss too.
                RobustKernelKind::Cauchy | RobustKernelKind::Tukey => {
                    // `ln_1p`, NOT `(1.0 + x).ln()`. In f32 the sum quantises to steps of 1.19e-7
                    // BEFORE the log, so small residuals — the converged regime the accept test
                    // has to resolve — lose most or all of their value: 28.4% low at scale=2.45,
                    // ‖r‖=1e-3, and exactly 0.0 for any r_sq/s2 < 6e-8.
                    let s2 = scale * scale;
                    0.5 * s2 * (r_sq / s2).ln_1p()
                }
            }
        };

        let t_lin = std::time::Instant::now();
        // f64 ACCUMULATOR over f32 terms. Each residual is computed in f32 (that is what the
        // Jacobians and blocks are), but ~93k of them are summed here, and an f32 running total
        // loses precision as it grows: the per-iteration cost decrease this solve reports is
        // ~5e-4 relative, which is only ~4000 f32 ulps at this magnitude. The short-step
        // convergence probe visibly hit that noise floor at rel ~ 8e-6. Summing in f64 costs
        // nothing measurable (one accumulator, not a per-term widening) and removes it.
        let mut cost = 0.0_f64;
        let mut n_depth_obs_iter = 0usize;

        for obs in observations {
            if obs.pose_idx >= p_total || obs.point_idx >= n_total {
                continue;
            }
            let pose = &se3s[obs.pose_idx];
            let point = &xyz[obs.point_idx];
            let (mut r, mut j_pose, mut j_point) =
                residual_and_jacobians(pose, point, obs.pixel, camera);
            let r_sq = r[0] * r[0] + r[1] * r[1];

            // IRLS weight; apply √w to r and J.
            let w = match robust {
                RobustKernelKind::Identity => 1.0,
                RobustKernelKind::Huber => huber_w(r_sq),
                RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq),
            };
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
            }
            cost += f64::from(robust_cost(r_sq, robust_scale));

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
                let mut b_block = [0.0_f64; 18];
                atb_6x3_into(&mut b_block, &j_pose, &j_point);
                b_by_point[xli as usize].push((pli as usize, b_block));
            }

            // ── Depth residual (optional metric anchor) ─────────────────
            // r_z = (Z_pred − d_meas) / σ_depth
            // ∂Z/∂ρ  = row 2 of R  — NOT e_z. `retract` is rplus (se3.rs), so the
            //          update is t ← t + R·υ and ∂Z/∂υ = e_zᵀR, the *same* vector as
            //          ∂Z/∂Xw. Assuming e_z pulls every camera along world +Z instead
            //          of along its own optical axis; the two coincide only for a
            //          camera looking down world +Z, which is the anchor pose and
            //          every depth test scene (they all use Mat3F64::IDENTITY), so
            //          the suite could not see it.
            // ∂Z/∂ω  = row 2 of S = -R · skew(p_w)
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

                // Depth residual and ∂r/∂z. The geometric Jacobian rows below are ∂z/∂θ, so the
                // chain rule collapses to a single scale factor — the two residual forms differ
                // only in that factor (1/σ for the metric form, 1/(z·σ) for the log form).
                let (r_z, inv_sigma) =
                    depth_residual(z_pred, d_meas, dscales[obs.pose_idx], sigma, log_depth);

                // J rows (1×6 pose, 1×3 point), all scaled by ∂r/∂z.
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

                // J_pose_depth (1×6): [ρ(r20, r21, r22) | ω(s20, s21, s22)] / σ
                let jpd = [
                    r20 * inv_sigma,
                    r21 * inv_sigma,
                    r22 * inv_sigma,
                    s20 * inv_sigma,
                    s21 * inv_sigma,
                    s22 * inv_sigma,
                ];
                // J_point_depth (1×3) IS the ρ block: both are ∂z/∂(world displacement).
                let jxd = &jpd[..3];

                // ── Apply IRLS robust weight to the depth residual ────────
                // The depth residual is a single scalar r_z (already scaled by
                // 1/σ_depth). Use the same Huber/Cauchy gate as the
                // reprojection path so outlier depth measurements (e.g.
                // boundary mis-samples) do not dominate the normal equations.
                // The gate uses ‖r_z‖² of the *whitened* residual, matching
                // the χ² interpretation (ORB-SLAM3 §IV.B uses χ²=7.815 for
                // 3-DoF RGB-D; we reuse `robust_scale_sq` for simplicity).
                let r_sq_d = r_z * r_z;
                let w_d = match robust {
                    RobustKernelKind::Identity => 1.0,
                    RobustKernelKind::Huber => huber_w_depth(r_sq_d),
                    RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w_depth(r_sq_d),
                };
                cost += f64::from(robust_cost(r_sq_d, depth_scale));
                n_depth_obs_iter += 1;

                // Accumulate into A (6×6) — w · outer product jpd·jpdᵀ.
                if pli >= 0 {
                    let pli_u = pli as usize;
                    let ab = &mut a_blocks[pli_u];
                    for i in 0..6 {
                        for k in 0..6 {
                            ab[i * 6 + k] += f64::from(w_d * jpd[i] * jpd[k]);
                        }
                    }
                    // g_pose -= w · jpdᵀ · r_z
                    let gp = &mut g_pose[pli_u];
                    for i in 0..6 {
                        gp[i] -= f64::from(w_d * jpd[i] * r_z);
                    }
                }
                // Accumulate into C (3×3) — w · outer product jxd·jxdᵀ.
                if xli >= 0 {
                    let xli_u = xli as usize;
                    let cb = &mut c_blocks[xli_u];
                    for i in 0..3 {
                        for k in 0..3 {
                            cb[i * 3 + k] += f64::from(w_d * jxd[i] * jxd[k]);
                        }
                    }
                    let gx = &mut g_point[xli_u];
                    for i in 0..3 {
                        gx[i] -= f64::from(w_d * jxd[i] * r_z);
                    }
                }
                // Accumulate into B (6×3) — w · jpd·jxdᵀ.
                if pli >= 0 && xli >= 0 {
                    let mut b_block = [0.0_f64; 18];
                    for i in 0..6 {
                        for k in 0..3 {
                            b_block[i * 3 + k] = f64::from(w_d * jpd[i] * jxd[k]);
                        }
                    }
                    b_by_point[xli as usize].push((pli as usize, b_block));
                }
            }
        }
        let _ = n_depth_obs_iter; // currently unused; reserved for future telemetry

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
                let w_p = match robust {
                    RobustKernelKind::Identity => 1.0,
                    RobustKernelKind::Huber => huber_w(r_sq_p),
                    RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq_p),
                };
                cost += f64::from(robust_cost(r_sq_p, robust_scale));

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
                            ab[ii * 6 + kk] += f64::from(w_p * row[ii] * row[kk]);
                        }
                    }
                }
                // RHS: g_pose -= w · Σ_r J_r.T · r_pos[r]
                let gp = &mut g_pose[pli_u];
                for r_idx in 0..3 {
                    let row = &j_pose_prior[r_idx * 6..(r_idx + 1) * 6];
                    for ii in 0..6 {
                        gp[ii] -= f64::from(w_p * row[ii] * r_pos[r_idx]);
                    }
                }

                // ── Optional gravity (up-vector) prior ────────────────────
                // u_pred = R^T · (0,−1,0): the camera's image-up expressed in
                // the world. For a fixed camera-frame vector v, this solver's
                // convention gives ∂(R^T v)/∂ω = +[R^T v]× and no ρ coupling —
                // the same pattern the centre prior's ω-part follows (its
                // [C]× IS [R^T(−t)]×). Purely rotational, so it augments only
                // A_ii like the centre prior.
                if let Some(upw) = prior.up_world {
                    let inv_su = 1.0_f32 / prior.up_sigma.max(1e-6);
                    // u_pred = R^T · up_cam, i.e. the claimed camera-frame direction expressed in
                    // world. With the default up_cam = (0,−1,0) this reduces to −(row 1 of R), the
                    // old fixed "image-up is world-up" form.
                    let a = prior.up_cam;
                    let u_pred = [
                        r_col0.x * a[0] + r_col0.y * a[1] + r_col0.z * a[2],
                        r_col1.x * a[0] + r_col1.y * a[1] + r_col1.z * a[2],
                        r_col2.x * a[0] + r_col2.y * a[1] + r_col2.z * a[2],
                    ];
                    let r_up = [
                        (u_pred[0] - upw[0]) * inv_su,
                        (u_pred[1] - upw[1]) * inv_su,
                        (u_pred[2] - upw[2]) * inv_su,
                    ];
                    let r_sq_u = r_up[0] * r_up[0] + r_up[1] * r_up[1] + r_up[2] * r_up[2];
                    let w_u = match robust {
                        RobustKernelKind::Identity => 1.0,
                        RobustKernelKind::Huber => huber_w(r_sq_u),
                        RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq_u),
                    };
                    cost += f64::from(robust_cost(r_sq_u, robust_scale));

                    let (ux, uy, uz) = (u_pred[0], u_pred[1], u_pred[2]);
                    // Rows of [u]× scaled by 1/σ, ρ-part zero.
                    let j_up: [f32; 18] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -uz * inv_su,
                        uy * inv_su,
                        0.0,
                        0.0,
                        0.0,
                        uz * inv_su,
                        0.0,
                        -ux * inv_su,
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
                                ab[ii * 6 + kk] += f64::from(w_u * row[ii] * row[kk]);
                            }
                        }
                    }
                    for r_idx in 0..3 {
                        let row = &j_up[r_idx * 6..(r_idx + 1) * 6];
                        for ii in 0..6 {
                            gp[ii] -= f64::from(w_u * row[ii] * r_up[r_idx]);
                        }
                    }
                }
            }
        }

        // ── Trajectory planarity prior (1-D out-of-plane residual) ─────────
        // For each FREE pose, one scalar residual
        //
        //     r_plane = n · (C - C_bar) / σ
        //
        // with (n, C_bar) the best-fit plane through the free camera centres, refitted this
        // iteration and held constant within it. Only the component ALONG the normal is penalised,
        // so the walk moves freely in-plane; this is the anisotropy a `BaPosePrior` cannot express.
        //
        // Jacobian: contract the centre prior's own 3×6 block with nᵀ. With ∂C/∂ρ = -I and
        // ∂C/∂ω = [C]×, and using nᵀ[C]× = (n × C)ᵀ:
        //
        //     ∂r/∂ρ = -nᵀ / σ            ∂r/∂ω = (n × C)ᵀ / σ
        //
        // One row, no point coupling, so like the other pose priors it touches only A_ii and
        // g_pose and leaves the Schur reduction's B and C blocks alone.
        //
        // NOT robustified. The other priors down-weight a single bad pose so it cannot dominate;
        // here a large residual is exactly the signal — a camera that has drifted furthest out of
        // plane is the one most in need of correction, and a robust kernel would switch the prior
        // off precisely where it matters. The plane fit itself is the outlier defence: it is a
        // least-squares fit over every free centre, so one excursion cannot carry it.
        if params.plane_prior_sigma != 0.0 {
            if let Some((nrm, ctr)) = fit_centre_plane(&se3s, &pose_local) {
                // Negative means INFER from the trajectory's own high-frequency bob. Falling back to
                // "no prior" when inference declines is deliberate: the alternative is a guessed
                // constant, and over-trust is this family's measured failure mode.
                let sigma = match (params.plane_prior_sigma < 0.0)
                    .then(|| infer_plane_sigma(&se3s, &pose_local, &nrm, &ctr))
                {
                    Some(Some(sg)) => sg,
                    // Inference declined (too few poses). An infinite sigma zeroes the weight, so the
                    // term contributes nothing — the same outcome as skipping, without a second path.
                    Some(None) => f64::INFINITY,
                    None => f64::from(params.plane_prior_sigma),
                };
                let inv_sigma = 1.0_f64 / sigma.max(1e-6);
                for i_global in 0..p_total {
                    let pli = pose_local[i_global];
                    if pli < 0 {
                        continue;
                    }
                    let pli_u = pli as usize;
                    let pose = &se3s[i_global];
                    let rm = pose.r.matrix();
                    let t = pose.t;
                    let (c0, c1, c2) = (rm.col(0), rm.col(1), rm.col(2));
                    let c_pred = [
                        -f64::from(c0.x * t.x + c0.y * t.y + c0.z * t.z),
                        -f64::from(c1.x * t.x + c1.y * t.y + c1.z * t.z),
                        -f64::from(c2.x * t.x + c2.y * t.y + c2.z * t.z),
                    ];
                    let d = [c_pred[0] - ctr[0], c_pred[1] - ctr[1], c_pred[2] - ctr[2]];
                    let r_plane = (nrm[0] * d[0] + nrm[1] * d[1] + nrm[2] * d[2]) * inv_sigma;
                    cost += 0.5 * f64::from(r_plane) * f64::from(r_plane);

                    // n × C
                    let ncx = [
                        nrm[1] * c_pred[2] - nrm[2] * c_pred[1],
                        nrm[2] * c_pred[0] - nrm[0] * c_pred[2],
                        nrm[0] * c_pred[1] - nrm[1] * c_pred[0],
                    ];
                    let j: [f64; 6] = [
                        -nrm[0] * inv_sigma,
                        -nrm[1] * inv_sigma,
                        -nrm[2] * inv_sigma,
                        ncx[0] * inv_sigma,
                        ncx[1] * inv_sigma,
                        ncx[2] * inv_sigma,
                    ];
                    let ab = &mut a_blocks[pli_u];
                    for ii in 0..6 {
                        for kk in 0..6 {
                            ab[ii * 6 + kk] += j[ii] * j[kk];
                        }
                    }
                    let gp = &mut g_pose[pli_u];
                    for ii in 0..6 {
                        gp[ii] -= j[ii] * r_plane;
                    }
                }
            }
        }

        // ── Motion priors (constant-velocity triplets) ──────────────────
        // FD Jacobians over the solver's own retraction; residuals whitened by their sigmas, so
        // gate with the depth-family knee (same "whitened units" family — see
        // `BaParams::depth_robust_scale_sq` for why they must not share the reprojection knee).
        let mut h_offdiag: std::collections::HashMap<(usize, usize), [f64; 36]> =
            std::collections::HashMap::new();
        if let Some(mps) = motion_priors {
            const FD_EPS: f32 = 1e-4;
            for mp in mps {
                if mp.i0 >= p_total || mp.i1 >= p_total || mp.i2 >= p_total {
                    continue;
                }
                let tri = [mp.i0, mp.i1, mp.i2];
                let locs: Vec<i64> = tri.iter().map(|&g| pose_local[g]).collect();
                if locs.iter().all(|&l| l < 0) {
                    continue; // fully fixed triplet constrains nothing
                }
                let r0 = motion_prior_residual(&se3s[mp.i0], &se3s[mp.i1], &se3s[mp.i2], mp);
                let r_sq_m: f32 = r0.iter().map(|v| v * v).sum();
                let w_m = match robust {
                    RobustKernelKind::Identity => 1.0,
                    RobustKernelKind::Huber => huber_w_depth(r_sq_m),
                    RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w_depth(r_sq_m),
                };
                cost += f64::from(robust_cost(r_sq_m, depth_scale));

                // J: 6 residual rows × (3 poses × 6 params), FD one param at a time.
                let mut jac = [[0.0f32; 18]; 6];
                for (pi, &g) in tri.iter().enumerate() {
                    if locs[pi] < 0 {
                        continue;
                    }
                    for k in 0..6 {
                        let mut delta = [0.0f32; 6];
                        delta[k] = FD_EPS;
                        let pert = se3s[g].retract(&delta);
                        let refs = [
                            if pi == 0 { &pert } else { &se3s[mp.i0] },
                            if pi == 1 { &pert } else { &se3s[mp.i1] },
                            if pi == 2 { &pert } else { &se3s[mp.i2] },
                        ];
                        let rp = motion_prior_residual(refs[0], refs[1], refs[2], mp);
                        for row in 0..6 {
                            jac[row][pi * 6 + k] = (rp[row] - r0[row]) / FD_EPS;
                        }
                    }
                }

                // Accumulate JᵀJ (per pose-pair block) and Jᵀr (per pose).
                for (a, &ga) in tri.iter().enumerate() {
                    let la = locs[a];
                    if la < 0 {
                        continue;
                    }
                    let la = la as usize;
                    let _ = ga;
                    // Gradient.
                    let gp = &mut g_pose[la];
                    for i in 0..6 {
                        let mut acc = 0.0f32;
                        for row in 0..6 {
                            acc += jac[row][a * 6 + i] * r0[row];
                        }
                        gp[i] -= f64::from(w_m * acc);
                    }
                    for (b, &gb) in tri.iter().enumerate() {
                        let lb = locs[b];
                        if lb < 0 {
                            continue;
                        }
                        let lb = lb as usize;
                        let _ = gb;
                        // 6×6 block JaᵀJb.
                        let mut blk = [0.0f32; 36];
                        for i in 0..6 {
                            for j in 0..6 {
                                let mut acc = 0.0f32;
                                for row in 0..6 {
                                    acc += jac[row][a * 6 + i] * jac[row][b * 6 + j];
                                }
                                blk[i * 6 + j] = w_m * acc;
                            }
                        }
                        if la == lb {
                            let ab = &mut a_blocks[la];
                            for x in 0..36 {
                                ab[x] += f64::from(blk[x]);
                            }
                        } else {
                            let e = h_offdiag.entry((la, lb)).or_insert([0.0f64; 36]);
                            for x in 0..36 {
                                e[x] += f64::from(blk[x]);
                            }
                        }
                    }
                }
            }
        }

        // Cost convergence (post-step convergence will follow successful steps below).
        if let Some(pc) = prev_cost {
            // Only declare convergence here on a *successful* step path; we'll
            // do that after accepting a step. For now, just log.
            let _ = pc;
        }

        BA_LIN_MICROS.fetch_add(
            t_lin.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let t_asm = std::time::Instant::now();

        // ── Apply LM damping ────────────────────────────────────────────
        //
        //   A[i] += λ·diag(A),  C[j] += λ·diag(C)
        //
        // Ellipsoidal, as Ceres does it — `LevenbergMarquardtStrategy` damps with the squared
        // column norms of J (i.e. diag(JᵀJ)) clamped to [min_lm_diagonal, max_lm_diagonal] =
        // [1e-6, 1e32]. The distinction matters here because one λ has to serve parameter blocks
        // in different units: rotation in radians, translation and points in metres. A spherical
        // λ·I damps a direction by the same absolute amount regardless of how well that direction
        // is already constrained, so it over-damps the stiff directions and under-damps the soft
        // ones. Scaling by the local curvature makes the damping relative, which is what makes λ
        // dimensionless and comparable across blocks.
        //
        // Measured against the spherical form on a 300-keyframe solve: terminal adjustment 87 → 35
        // iterations, ba_big_secs 163.4 → 84.3, with the answer unchanged (ATE vs COLMAP 0.117 →
        // 0.116 m, flatness and jitter identical to 3 decimals). Pure conditioning, not a
        // different minimum.
        let damp_diag: Vec<[f64; 6]> = a_blocks
            .iter()
            .map(|ab| {
                let mut d = [1.0_f64; 6];
                for (k, dk) in d.iter_mut().enumerate() {
                    *dk = ab[k * 6 + k].clamp(MIN_LM_DIAGONAL, MAX_LM_DIAGONAL);
                }
                d
            })
            .collect();
        let damp_diag_pt: Vec<[f64; 3]> = c_blocks
            .iter()
            .map(|cb| {
                let mut d = [1.0_f64; 3];
                for (k, dk) in d.iter_mut().enumerate() {
                    *dk = cb[k * 3 + k].clamp(MIN_LM_DIAGONAL, MAX_LM_DIAGONAL);
                }
                d
            })
            .collect();
        for (ab, dd) in a_blocks.iter_mut().zip(&damp_diag) {
            for d in 0..6 {
                ab[d * 6 + d] += f64::from(lambda) * dd[d];
            }
        }
        for (cb, dd) in c_blocks.iter_mut().zip(&damp_diag_pt) {
            for d in 0..3 {
                cb[d * 3 + d] += f64::from(lambda) * dd[d];
            }
        }

        // ── Build M (dense 6Pf × 6Pf) + m (6Pf) ─────────────────────────
        let dim = n_free_poses * 6;
        // 117 MB at P=637, allocated and zeroed EVERY LM iteration. The sparse path never touches
        // it, so it is a 0x0 stub there; `Mat` has no null state and the branches below never index
        // it when the accumulator is live.
        let mut m_mat = if params.sparse_reduced_system {
            Mat::<f64>::zeros(0, 0)
        } else {
            Mat::<f64>::zeros(dim, dim)
        };
        let mut m_vec = vec![0.0_f64; dim];

        // Place A blocks on diagonal of M.
        // Built before ANY write into the reduced system, because the A-block loop below is the
        // first writer and must not take the dense branch against the 0x0 stub. The pattern is a
        // function of the observation graph, so it is built once and only cleared thereafter.
        if params.sparse_reduced_system && accum.is_none() {
            // Motion priors couple triplets regardless of covisibility, so their pairs must be
            // in the pattern or the block they write has nowhere to go. Mapped into the SAME local
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

        for (k, ab) in a_blocks.iter().enumerate() {
            match accum.as_mut() {
                Some(acc) => {
                    if let Some(slot) = acc.slot(k, k) {
                        let blk = &mut acc.blocks[slot];
                        for i in 0..6 {
                            for j in 0..6 {
                                blk[i * 6 + j] = ab[i * 6 + j] as f64;
                            }
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
        }

        // Motion-prior off-diagonal pose-pose blocks (already weighted).
        for ((la, lb), blk) in &h_offdiag {
            match accum.as_mut() {
                // A motion prior can couple cameras that share no point, so the pair may be absent
                // from the pattern. Fall back to the dense matrix in that case rather than dropping
                // the term — the pattern is a superset of point coupling, not of all coupling.
                Some(acc) => match acc.slot(*la, *lb) {
                    Some(slot) => {
                        let dst = &mut acc.blocks[slot];
                        for i in 0..6 {
                            for j in 0..6 {
                                dst[i * 6 + j] += blk[i * 6 + j] as f64;
                            }
                        }
                    }
                    None => return Err(SchurBaError::CholeskyFailed(
                        "motion prior couples cameras with no shared point; the sparse pattern \
                             does not cover it"
                            .into(),
                    )),
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
        let mut c_inv_blocks: Vec<Option<[f64; 9]>> = Vec::with_capacity(n_free_points);
        for cb in &c_blocks {
            c_inv_blocks.push(invert_3x3(cb));
        }

        // ── Schur correction, parallel over POINTS ──────────────────────
        //
        // Points are independent: each contributes to `m_vec` and to the 6x6 blocks of the camera
        // pairs that see it. The serial version was 72% of solve time and ran on one core.
        //
        // DETERMINISM. The points are cut into `n_threads` FIXED contiguous chunks, each chunk
        // accumulates into its own buffer, and the buffers are folded back IN CHUNK ORDER. Float
        // addition is not associative, so a shared accumulator written in thread-arrival order
        // would make the map depend on scheduling — the exact defect class behind the CUDA
        // matcher, track-order and descriptor fixes. Here the summation order is a function of the
        // partition alone, so every thread count produces bit-identical output.
        let n_threads = assembly_threads(params).min(b_by_point.len().max(1));
        let chunk = b_by_point.len().div_ceil(n_threads.max(1));

        // Per-point work, shared by both paths. Returns the RHS contribution and, for the sparse
        // path, the (slot, block) contributions.
        let point_contrib =
            |j: usize| -> Option<(Vec<(usize, [f64; 6])>, Vec<(usize, [f64; 36])>)> {
                let c_inv_j = c_inv_blocks[j]?;
                let b_for_j = &b_by_point[j];
                let bc: Vec<(usize, [f64; 18])> = b_for_j
                    .iter()
                    .map(|(i_loc, b)| (*i_loc, matmul_6x3_3x3(b, &c_inv_j)))
                    .collect();

                let gp = g_point[j];
                let mut rhs = Vec::with_capacity(bc.len());
                for (i_loc, bc_block) in &bc {
                    rhs.push((*i_loc, matvec_6x3_3(bc_block, &gp)));
                }

                let mut blocks = Vec::with_capacity(bc.len() * bc.len());
                for (i1_loc, bc1) in bc.iter() {
                    for (idx2, (i2_loc, _)) in bc.iter().enumerate() {
                        let b2 = &b_for_j[idx2].1;
                        let mut blk = [0.0_f64; 36];
                        for r in 0..6 {
                            for c in 0..6 {
                                let mut acc = 0.0_f64;
                                for k in 0..3 {
                                    acc += bc1[r * 3 + k] * b2[c * 3 + k];
                                }
                                blk[r * 6 + c] = acc;
                            }
                        }
                        blocks.push(((*i1_loc, *i2_loc), blk));
                    }
                }
                // Encode the pair as a flat key here so the reduction below needs no index lookup.
                let blocks = blocks
                    .into_iter()
                    .map(|((i1, i2), blk)| (i1 * n_free_poses + i2, blk))
                    .collect();
                Some((rhs, blocks))
            };

        let ranges: Vec<(usize, usize)> = (0..b_by_point.len())
            .step_by(chunk.max(1))
            .map(|st| (st, (st + chunk).min(b_by_point.len())))
            .collect();

        let per_chunk: Vec<ChunkContrib> = if n_threads <= 1 {
            ranges
                .iter()
                .map(|&(a, b)| {
                    let mut r = Vec::new();
                    let mut k = Vec::new();
                    for j in a..b {
                        if let Some((rr, kk)) = point_contrib(j) {
                            r.extend(rr);
                            k.extend(kk);
                        }
                    }
                    (r, k)
                })
                .collect()
        } else {
            use rayon::prelude::*;
            ranges
                .par_iter()
                .map(|&(a, b)| {
                    let mut r = Vec::new();
                    let mut k = Vec::new();
                    for j in a..b {
                        if let Some((rr, kk)) = point_contrib(j) {
                            r.extend(rr);
                            k.extend(kk);
                        }
                    }
                    (r, k)
                })
                .collect()
        };

        // Fold IN CHUNK ORDER. `par_iter().collect()` preserves input order, so this is the same
        // sequence of additions the serial loop performed, regardless of completion order.
        for (rhs, blocks) in &per_chunk {
            for (i_loc, bc_g) in rhs {
                let base = i_loc * 6;
                for r in 0..6 {
                    m_vec[base + r] -= bc_g[r];
                }
            }
            match accum.as_mut() {
                Some(acc) => {
                    for (key, blk) in blocks {
                        let (i1_loc, i2_loc) = (key / n_free_poses, key % n_free_poses);
                        let slot = acc.index[i1_loc * acc.n + i2_loc];
                        let dst = &mut acc.blocks[slot];
                        for t in 0..36 {
                            dst[t] -= blk[t];
                        }
                    }
                }
                None => {
                    for (key, blk) in blocks {
                        let (i1_loc, i2_loc) = (key / n_free_poses, key % n_free_poses);
                        let (row0, col0) = (i1_loc * 6, i2_loc * 6);
                        for r in 0..6 {
                            for c in 0..6 {
                                m_mat[(row0 + r, col0 + c)] -= blk[r * 6 + c];
                            }
                        }
                    }
                }
            }
        }

        // ── Solve M · δ_pose = m via Cholesky ────────────────────────────
        // Symmetrize numerically (the construction above should already be
        // symmetric to within roundoff; do an average to guarantee).
        // Dense path only: O(dim^2/2) with one index striding `ld * 8` bytes. The accumulator's
        // triplet emission takes the lower triangle directly, so there is nothing to average.
        if accum.is_none() {
            for i in 0..dim {
                for j in (i + 1)..dim {
                    let avg = 0.5 * (m_mat[(i, j)] + m_mat[(j, i)]);
                    m_mat[(i, j)] = avg;
                    m_mat[(j, i)] = avg;
                }
            }
        }
        // SPARSE PATH. Two cameras couple in this matrix only if they share a point, and on a real
        // walkthrough that is 2.2% of pairs — so a dense factorisation spends nearly all of its time
        // on structurally-zero entries. The assembly above already touched only the nonzero blocks;
        // this just stops materialising the rest.
        //
        // Emitted as triplets, which faer SUMS on duplicates, so the same three contributions (the
        // diagonal A blocks, the motion-prior off-diagonals, the Schur corrections) accumulate
        // exactly as they did into the dense matrix. Only the LOWER triangle is emitted, which is
        // all `Side::Lower` reads — and the symmetrisation above has already made the two triangles
        // agree, so no information is lost by dropping the upper one.
        BA_ASM_MICROS.fetch_add(
            t_asm.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let t_fact = std::time::Instant::now();
        let d_pose_col = if params.sparse_reduced_system {
            // Straight from the compact blocks. Scanning the dense matrix for nonzeros — what this
            // replaced — was an O(dim^2/2) pass over 117 MB to recover a structure the assembly
            // already knew, and it forced the matrix to exist at all.
            let trips = accum
                .as_ref()
                .expect("sparse path builds the accumulator before the first solve")
                .triplets();
            let a = match faer::sparse::SparseColMat::try_new_from_triplets(dim, dim, &trips) {
                Ok(a) => a,
                Err(e) => return Err(SchurBaError::CholeskyFailed(format!("{e:?}"))),
            };
            let sym = match faer::sparse::linalg::solvers::SymbolicLlt::try_new(
                a.symbolic(),
                faer::Side::Lower,
            ) {
                Ok(s) => s,
                Err(e) => return Err(SchurBaError::CholeskyFailed(format!("{e:?}"))),
            };
            match faer::sparse::linalg::solvers::Llt::try_new_with_symbolic(
                sym,
                a.as_ref(),
                faer::Side::Lower,
            ) {
                Ok(l) => {
                    use faer::linalg::solvers::Solve;
                    l.solve(&Mat::<f64>::from_fn(dim, 1, |i, _| m_vec[i]))
                }
                Err(_) => {
                    lambda *= 10.0;
                    if lambda > 1e10 {
                        return Err(SchurBaError::CholeskyFailed("sparse llt".into()));
                    }
                    continue;
                }
            }
        } else {
            let chol = match m_mat.llt(faer::Side::Lower) {
                Ok(c) => c,
                Err(e) => {
                    // Bump damping and retry next outer iteration.
                    lambda *= 10.0;
                    if lambda > 1e10 {
                        return Err(SchurBaError::CholeskyFailed(format!("{e:?}")));
                    }
                    continue;
                }
            };
            chol.solve(&Mat::<f64>::from_fn(dim, 1, |i, _| m_vec[i]))
        };

        // ── Back-substitute for points: δ_x[j] = C⁻¹ (g_x - B.T · δ_p) ──
        let mut d_pose = vec![0.0_f64; dim];
        for i in 0..dim {
            d_pose[i] = d_pose_col[(i, 0)];
        }
        let mut d_point = vec![[0.0_f64; 3]; n_free_points];
        for (j, b_for_j) in b_by_point.iter().enumerate() {
            let Some(c_inv_j) = c_inv_blocks[j] else {
                continue;
            };
            // rhs = g_point[j] - sum_i B[i, j].T · δ_pose[i]
            let mut rhs = g_point[j];
            for (i_loc, b_block) in b_for_j {
                let mut dp6 = [0.0_f64; 6];
                let base = i_loc * 6;
                for r in 0..6 {
                    dp6[r] = d_pose[base + r];
                }
                let contrib = matvec_6x3t_6(b_block, &dp6);
                for c in 0..3 {
                    rhs[c] -= contrib[c];
                }
            }
            d_point[j] = matvec_3x3_3(&c_inv_j, &rhs);
        }

        BA_FACT_MICROS.fetch_add(
            t_fact.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let t_trial = std::time::Instant::now();

        // ── Trial: retract poses, add to points, recompute cost ─────────
        //
        // Evaluates the objective at `x + t·δ`. Parameterised by `t` so the same code path serves
        // the full trial step (t = 1) and, under the trace, a SHORT step. The short step is what
        // decides whether a disappointing gain ratio means curvature or a broken model: for any
        // model that is consistent with the function to first order,
        //     [F(x) − F(x + tδ)] / (t · δᵀg)  →  1   as t → 0
        // no matter how strong the curvature, because the quadratic term dies as t². A ratio that
        // settles anywhere else says the linearisation is not a model of the cost being compared.
        let eval_at = |t: f32| -> (f64, f64, usize, usize, Vec<SE3F32>, Vec<Vec3F64>) {
            let mut se3s_trial = se3s.clone();
            for i_global in 0..p_total {
                let pli = pose_local[i_global];
                if pli < 0 {
                    continue;
                }
                let pli = pli as usize;
                let delta: [f32; 6] = [
                    d_pose[pli * 6] as f32 * t,
                    d_pose[pli * 6 + 1] as f32 * t,
                    d_pose[pli * 6 + 2] as f32 * t,
                    d_pose[pli * 6 + 3] as f32 * t,
                    d_pose[pli * 6 + 4] as f32 * t,
                    d_pose[pli * 6 + 5] as f32 * t,
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
                    xyz[i_global].x + dp[0] as f64 * f64::from(t),
                    xyz[i_global].y + dp[1] as f64 * f64::from(t),
                    xyz[i_global].z + dp[2] as f64 * f64::from(t),
                );
            }

            let mut new_cost = 0.0_f64;
            // Trace-only: the RAW (unrobustified, unweighted) reprojection error at the trial point,
            // so a slow tail can be read as real geometric improvement or as drift in the priors.
            let mut trace_reproj_sq = 0.0_f64;
            let mut trace_reproj_n = 0usize;
            let mut trace_robust_n = 0usize;
            for obs in observations {
                if obs.pose_idx >= p_total || obs.point_idx >= n_total {
                    continue;
                }
                let pose = &se3s_trial[obs.pose_idx];
                let point = &xyz_trial[obs.point_idx];
                let (r, _, _) = residual_and_jacobians(pose, point, obs.pixel, camera);
                let r_sq = r[0] * r[0] + r[1] * r[1];
                let w = match robust {
                    RobustKernelKind::Identity => 1.0,
                    RobustKernelKind::Huber => huber_w(r_sq),
                    RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq),
                };
                new_cost += f64::from(robust_cost(r_sq, robust_scale));
                // Counted unconditionally: `BA_OBS` reports what the adjustment actually sees, which
                // is NOT the map-wide observation count and was the source of a 7x error in a
                // per-observation timing comparison.
                trace_reproj_n += 1;
                if trace_on {
                    trace_reproj_sq += f64::from(r_sq);
                    if w < 1.0 {
                        trace_robust_n += 1;
                    }
                }

                // Depth residual contribution to the trial cost, through the same `robust_cost`
                // the linearisation pass uses, so accept/reject compares like with like.
                if let Some(d_meas) = obs.depth_meas {
                    let sigma = obs.depth_sigma.max(1e-6);
                    let z_pred = clamped_z(pose, point);
                    // Scales are held at their current value across the trial: the LM accept/reject
                    // must compare the SAME objective the step was computed against, so re-fitting
                    // `s` here would let a rejected step look good on a moved goalpost.
                    let (r_z, _) =
                        depth_residual(z_pred, d_meas, dscales[obs.pose_idx], sigma, log_depth);
                    let r_sq_d = r_z * r_z;
                    new_cost += f64::from(robust_cost(r_sq_d, depth_scale));
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
                    let r_sq_p = r0 * r0 + r1 * r1 + r2 * r2;
                    new_cost += f64::from(robust_cost(r_sq_p, robust_scale));

                    // Up-prior contribution — the accept/reject decision must see the
                    // same objective the linearisation minimised or the LM loop
                    // rejects every step that trades reprojection for uprightness.
                    if let Some(upw) = prior.up_world {
                        let inv_su = 1.0_f32 / prior.up_sigma.max(1e-6);
                        // Same generalisation as the cost pass — see there.
                        let a = prior.up_cam;
                        let u_pred = [
                            r_col0.x * a[0] + r_col0.y * a[1] + r_col0.z * a[2],
                            r_col1.x * a[0] + r_col1.y * a[1] + r_col1.z * a[2],
                            r_col2.x * a[0] + r_col2.y * a[1] + r_col2.z * a[2],
                        ];
                        let ru0 = (u_pred[0] - upw[0]) * inv_su;
                        let ru1 = (u_pred[1] - upw[1]) * inv_su;
                        let ru2 = (u_pred[2] - upw[2]) * inv_su;
                        let r_sq_u = ru0 * ru0 + ru1 * ru1 + ru2 * ru2;
                        new_cost += f64::from(robust_cost(r_sq_u, robust_scale));
                    }
                }
            }

            // Planarity contribution to the trial cost. The accept/reject test must see the SAME
            // objective the linearisation minimised, or LM rejects every step that trades a little
            // reprojection for flatness — which is every step this prior exists to take.
            //
            // The plane is re-fitted on the TRIAL poses rather than reused from the linearisation. Both
            // are defensible; refitting is the honest one, because it scores the trial trajectory by its
            // own best-fit plane instead of by a plane chosen to flatter the previous iterate. A step
            // that merely rotates the whole trajectory would otherwise look like an improvement.
            if params.plane_prior_sigma != 0.0 {
                if let Some((nrm, ctr)) = fit_centre_plane(&se3s_trial, &pose_local) {
                    // Re-inferred on the trial poses for the same reason the plane is re-fitted there:
                    // the trial trajectory must be scored by its own tolerance, not a stale one.
                    let sigma = match (params.plane_prior_sigma < 0.0)
                        .then(|| infer_plane_sigma(&se3s_trial, &pose_local, &nrm, &ctr))
                    {
                        Some(Some(sg)) => sg,
                        Some(None) => f64::INFINITY,
                        None => f64::from(params.plane_prior_sigma),
                    };
                    let inv_sigma = 1.0_f64 / sigma.max(1e-6);
                    for (i_global, pose) in se3s_trial.iter().enumerate() {
                        if pose_local[i_global] < 0 {
                            continue;
                        }
                        let rm = pose.r.matrix();
                        let t = pose.t;
                        let (c0, c1, c2) = (rm.col(0), rm.col(1), rm.col(2));
                        let c_pred = [
                            -f64::from(c0.x * t.x + c0.y * t.y + c0.z * t.z),
                            -f64::from(c1.x * t.x + c1.y * t.y + c1.z * t.z),
                            -f64::from(c2.x * t.x + c2.y * t.y + c2.z * t.z),
                        ];
                        let r_plane = (nrm[0] * (c_pred[0] - ctr[0])
                            + nrm[1] * (c_pred[1] - ctr[1])
                            + nrm[2] * (c_pred[2] - ctr[2]))
                            * inv_sigma;
                        new_cost += 0.5 * f64::from(r_plane) * f64::from(r_plane);
                    }
                }
            }

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
                    new_cost += f64::from(robust_cost(r_sq_m, depth_scale));
                }
            }
            (
                new_cost,
                trace_reproj_sq,
                trace_reproj_n,
                trace_robust_n,
                se3s_trial,
                xyz_trial,
            )
        };

        let (new_cost, trace_reproj_sq, trace_reproj_n, trace_robust_n, se3s_trial, xyz_trial) =
            eval_at(1.0);
        BA_TRIAL_MICROS.fetch_add(
            t_trial.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        BA_OBS.store(trace_reproj_n as u64, std::sync::atomic::Ordering::Relaxed);

        // Gain ratio ρ = actual reduction / model-predicted reduction (Ceres/Nielsen trust
        // region). The LM step δ solves (H+λI)δ = g, so the quadratic model predicts a decrease
        // of 0.5·δᵀ(g + λδ) — computed in the FULL pose+point space (the Schur elimination is
        // algebraic, not a model change), with the UNDAMPED gradients g_pose/g_point.
        let mut pred = 0.0_f64;
        // Max-norm of the UNDAMPED gradient — the quantity COLMAP terminates on
        // (`gradient_tolerance = 1e-4`, `function_tolerance = 0`). Traced only; nothing reads it.
        let mut gmax = 0.0_f64;
        // δᵀg — the FIRST-ORDER predicted decrease, i.e. the directional derivative along the
        // step. Unlike `pred` this carries no curvature and no damping, so it is the quantity a
        // short-step probe must reproduce.
        let mut gdotd = 0.0_f64;
        for k in 0..n_free_poses {
            for i in 0..6 {
                let d = d_pose[k * 6 + i];
                // Same damping metric the system was built with (see `damp_diag`), otherwise the
                // gain ratio compares against a model the solver never formed.
                pred += d * (g_pose[k][i] + f64::from(lambda) * damp_diag[k][i] * d);
                gdotd += d * g_pose[k][i];
                gmax = gmax.max(g_pose[k][i].abs());
            }
        }
        for (j, dp) in d_point.iter().enumerate() {
            for c in 0..3 {
                pred += dp[c] * (g_point[j][c] + f64::from(lambda) * damp_diag_pt[j][c] * dp[c]);
                gdotd += dp[c] * g_point[j][c];
                gmax = gmax.max(g_point[j][c].abs());
            }
        }
        // Short-step probes. `fo(t)` is the realised decrease over the first-order prediction at
        // step `tδ`; consistency forces fo(t) → 1 as t → 0.
        let (fo_10, fo_02) = if trace_on && gdotd > 0.0 {
            let probe = |t: f32| -> f64 {
                let (c_t, ..) = eval_at(t);
                (cost - c_t) / (f64::from(t) * gdotd)
            };
            (probe(0.1), probe(0.02))
        } else {
            (f64::NAN, f64::NAN)
        };
        pred *= 0.5;
        let rho = if pred > 1e-20 {
            (cost - new_cost) / pred
        } else {
            // Degenerate model prediction: fall back to plain cost comparison so a solved-but-
            // flat system still makes progress instead of spinning the damping loop.
            if new_cost < cost {
                1.0
            } else {
                -1.0
            }
        };

        if rho > 0.0 && new_cost < cost {
            // Accept step.
            let rel = if cost > 1e-12 {
                (cost - new_cost) / cost
            } else {
                0.0
            };
            se3s = se3s_trial;
            xyz = xyz_trial;
            prev_cost = Some(new_cost);
            // Nielsen's update: a step that matched the model (ρ→1) slashes λ toward the
            // Gauss-Newton regime; a barely-accepted step (ρ→0) leaves λ nearly unchanged.
            // Replaces the fixed ÷3, whose one-size decrement both over-trusted weak steps and
            // under-trusted strong ones.
            let f = 2.0 * rho - 1.0;
            let scale = (1.0 - f * f * f).max(1.0 / 3.0) as f32;
            lambda = (lambda * scale).max(1e-8);
            nu = 2.0;
            if trace_on {
                let rmse = if trace_reproj_n > 0 {
                    (trace_reproj_sq / trace_reproj_n as f64).sqrt()
                } else {
                    0.0
                };
                let robust_frac = if trace_reproj_n > 0 {
                    trace_robust_n as f64 / trace_reproj_n as f64
                } else {
                    0.0
                };
                eprintln!(
                    "BA_TRACE iter={iters_done} ACCEPT cost={new_cost:.6e} rel={rel:.3e} rho={rho:.3e} lambda={lambda:.3e} gmax={gmax:.4e} rmse={rmse:.5} rfrac={robust_frac:.4} fo10={fo_10:.4} fo02={fo_02:.4}"
                );
            }
            if rel < f64::from(params.cost_tolerance) {
                converged = true;
                break;
            }
        } else {
            // Reject — escalate damping and retry.
            //
            // NOTE: convergence is NOT tested here, only in the accept branch. A solve sitting at
            // its minimum, where every trial step is rejected because there is no improvement left
            // to find, therefore burns its whole iteration budget without ever declaring success.
            if trace_on {
                eprintln!(
                    "BA_TRACE iter={iters_done} REJECT rho={rho:.3e} lambda={lambda:.3e} nu={nu:.1} gmax={gmax:.4e} fo10={fo_10:.4} fo02={fo_02:.4}"
                );
            }
            lambda *= nu;
            nu *= 2.0;
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

    BA_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Per-call shape, gated by `KORNIA_BA_SIZES`. Mirrors the same probe on the upstream port so
    // the two can be compared call-for-call: aggregate totals cannot tell "many small solves" from
    // "few large ones", and inferring that from totals has gone wrong twice.
    if std::env::var_os("KORNIA_BA_SIZES").is_some() {
        eprintln!(
            "BA_SIZE poses={n_free_poses} points={n_free_points} obs={} iters={iters_done} ms={}",
            observations.len(),
            ba_t0.elapsed().as_millis()
        );
    }
    BA_ITERS.fetch_add(iters_done, std::sync::atomic::Ordering::Relaxed);
    BA_DIM_CUBED.fetch_add(
        (n_free_poses as u64 * 6).pow(3) / 1_000_000,
        std::sync::atomic::Ordering::Relaxed,
    );
    BA_MICROS.fetch_add(
        ba_t0.elapsed().as_micros() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    if n_free_poses * 6 >= 1000 {
        BA_BIG_MICROS.fetch_add(
            ba_t0.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        BA_BIG_ITERS.fetch_add(iters_done, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(BaResult {
        poses: out_poses,
        points: out_points,
        iterations: iters_done,
        converged,
        depth_scales: if log_depth { dscales } else { Vec::new() },
    })
}

#[cfg(test)]
mod tests {
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

    /// Assembly scaling, `cargo test -p kornia-3d --lib assembly_thread_scaling -- --ignored
    /// --nocapture`. Not an assertion — a measurement, printed for the reader.
    #[test]
    #[ignore]
    fn assembly_thread_scaling() {
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
        // 120 cameras over a 3000-point cloud with ~40% visibility: big enough that the assembly
        // dominates, small enough to run in a test.
        let n_cam = 120usize;
        let poses: Vec<Pose3d> = (0..n_cam)
            .map(|i| {
                let t = i as f64 * 0.05;
                Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-t, 0.01 * t, 0.0))
            })
            .collect();
        let pts: Vec<Vec3F64> = (0..3000)
            .map(|k| {
                let kf = k as f64;
                Vec3F64::new(
                    (kf * 0.37).sin() * 3.0,
                    (kf * 0.29).cos() * 2.0,
                    5.0 + (kf * 0.11).sin() * 2.0,
                )
            })
            .collect();
        let mut observations = Vec::new();
        for (ci, pose) in poses.iter().enumerate() {
            for (pi, pt) in pts.iter().enumerate() {
                if (pi + ci * 7) % 5 >= 2 {
                    continue;
                }
                let pc = pose.transform_point(pt);
                if pc.z <= 0.1 {
                    continue;
                }
                observations.push(BaObservation {
                    pose_idx: ci,
                    point_idx: pi,
                    pixel: [
                        (camera.fx * pc.x / pc.z + camera.cx) as f32,
                        (camera.fy * pc.y / pc.z + camera.cy) as f32,
                    ],
                    fixed_pose: ci == 0,
                    fixed_point: false,
                    ..BaObservation::default()
                });
            }
        }
        let start_pts: Vec<Vec3F64> = pts
            .iter()
            .map(|p| *p + Vec3F64::new(0.02, -0.015, 0.03))
            .collect();
        println!(
            "problem: {n_cam} cameras, {} points, {} observations",
            pts.len(),
            observations.len()
        );
        for threads in [1usize, 2, 4, 6] {
            let t0 = std::time::Instant::now();
            let r = bundle_adjust_schur(
                &poses,
                &start_pts,
                &observations,
                &camera,
                &BaParams {
                    max_iterations: 6,
                    sparse_reduced_system: true,
                    assembly_threads: threads,
                    ..Default::default()
                },
            )
            .unwrap();
            println!(
                "  threads={threads}  {:.2}s  iters={}",
                t0.elapsed().as_secs_f64(),
                r.iterations
            );
        }
    }

    /// The parallel assembly must give BIT-IDENTICAL results at every thread count.
    ///
    /// This is the whole safety argument for parallelising it. The points are cut into fixed
    /// contiguous chunks and the per-chunk accumulators are folded back in chunk order, so the
    /// sequence of float additions is a property of the partition and not of which thread
    /// finishes first. Had it been a shared accumulator under `par_iter`, the map would depend on
    /// scheduling — the same defect that made the CUDA matcher, the track ordering and the SIFT
    /// descriptors nondeterministic, each of which took a separate investigation to find.
    ///
    /// Bit-identical, not approximately equal: anything looser would pass while the ordering
    /// silently varied.
    #[test]
    fn assembly_is_deterministic_across_thread_counts() {
        // The env override wins over `BaParams`, so it must not be set while this runs.
        assert!(
            std::env::var_os("KORNIA_BA_THREADS").is_none(),
            "unset KORNIA_BA_THREADS: it overrides the thread counts this test sweeps"
        );
        // Same shape as `sparse_reduced_system_matches_dense`: five cameras on an arc over a
        // shared 4x4 cloud, so the reduced system has real off-diagonal blocks and the point
        // partition actually splits across chunks.
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
        let gt_poses: Vec<Pose3d> = (0..5)
            .map(|i| {
                let t = i as f64 * 0.2;
                Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-t, 0.02 * t, 0.0))
            })
            .collect();
        let mut gt_points = Vec::new();
        for gx in 0..4 {
            for gy in 0..4 {
                gt_points.push(Vec3F64::new(
                    -0.6 + 0.4 * gx as f64,
                    -0.6 + 0.4 * gy as f64,
                    4.0 + 0.1 * ((gx + gy) as f64),
                ));
            }
        }
        let mut observations = Vec::new();
        for (ci, pose) in gt_poses.iter().enumerate() {
            for (pi, pt) in gt_points.iter().enumerate() {
                let pc = pose.transform_point(pt);
                observations.push(BaObservation {
                    pose_idx: ci,
                    point_idx: pi,
                    pixel: [
                        (camera.fx * pc.x / pc.z + camera.cx) as f32,
                        (camera.fy * pc.y / pc.z + camera.cy) as f32,
                    ],
                    fixed_pose: ci == 0,
                    fixed_point: false,
                    ..BaObservation::default()
                });
            }
        }
        let points: Vec<Vec3F64> = gt_points
            .iter()
            .map(|p| *p + Vec3F64::new(0.03, -0.02, 0.04))
            .collect();
        let poses: Vec<Pose3d> = gt_poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i == 0 {
                    *p
                } else {
                    Pose3d::new(p.rotation, p.translation + Vec3F64::new(0.02, -0.01, 0.015))
                }
            })
            .collect();
        let mut reference: Option<(Vec<[f64; 3]>, Vec<[f64; 3]>)> = None;
        for threads in [1_usize, 2, 3, 6] {
            let r = bundle_adjust_schur(
                &poses,
                &points,
                &observations,
                &camera,
                &BaParams {
                    max_iterations: 15,
                    assembly_threads: threads,
                    ..Default::default()
                },
            )
            .unwrap();
            let got = (
                r.poses
                    .iter()
                    .map(|p| [p.translation.x, p.translation.y, p.translation.z])
                    .collect::<Vec<_>>(),
                r.points.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>(),
            );
            match &reference {
                None => reference = Some(got),
                Some(want) => {
                    assert_eq!(
                        want.0, got.0,
                        "pose translations differ at {threads} threads (bit-exact comparison)"
                    );
                    assert_eq!(
                        want.1, got.1,
                        "points differ at {threads} threads (bit-exact comparison)"
                    );
                }
            }
        }
    }

    /// The sparse reduced-camera solve must produce the same answer as the dense one.
    ///
    /// This is the whole safety argument for the flag. The sparse path changes HOW the reduced
    /// system is factorised, never WHAT is factorised: the assembly above is untouched, and only the
    /// lower triangle is emitted because that is all `Side::Lower` reads. If the two paths ever
    /// disagree beyond roundoff, the sparse assembly has dropped or double-counted an entry — which
    /// is exactly the failure a 2.2%-dense matrix makes easy to miss, since almost every entry it
    /// could drop is legitimately zero.
    ///
    /// Multi-camera on purpose: a two-camera problem has a reduced system of one free 6x6 block and
    /// no off-diagonal structure at all, so it would pass while telling us nothing about the
    /// coupling terms.
    #[test]
    fn sparse_reduced_system_matches_dense() {
        let cam = PinholeCamera {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        // Five cameras along an arc, all seeing a common cloud, so the reduced system has genuine
        // off-diagonal blocks for the Schur correction to fill.
        let poses: Vec<Pose3d> = (0..5)
            .map(|i| {
                let t = i as f64 * 0.2;
                Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-t, 0.02 * t, 0.0))
            })
            .collect();
        let mut points = Vec::new();
        for gx in 0..4 {
            for gy in 0..4 {
                points.push(Vec3F64::new(
                    -0.6 + 0.4 * gx as f64,
                    -0.6 + 0.4 * gy as f64,
                    4.0 + 0.1 * ((gx + gy) as f64),
                ));
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
        for (ci, pose) in poses.iter().enumerate() {
            for (pi, pt) in points.iter().enumerate() {
                observations.push(BaObservation {
                    pose_idx: ci,
                    point_idx: pi,
                    pixel: project(pose, pt),
                    fixed_pose: ci == 0,
                    fixed_point: false,
                    ..BaObservation::default()
                });
            }
        }
        // Perturb so the solver has real work to do; both arms get the identical start.
        let start_pts: Vec<Vec3F64> = points
            .iter()
            .map(|p| *p + Vec3F64::new(0.03, -0.02, 0.04))
            .collect();
        let start_poses: Vec<Pose3d> = poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i == 0 {
                    *p
                } else {
                    Pose3d::new(p.rotation, p.translation + Vec3F64::new(0.01, -0.01, 0.02))
                }
            })
            .collect();

        let run = |sparse: bool| {
            let params = BaParams {
                max_iterations: 12,
                sparse_reduced_system: sparse,
                ..Default::default()
            };
            bundle_adjust_schur(&start_poses, &start_pts, &observations, &cam, &params)
                .expect("ba converges")
        };
        let dense = run(false);
        let sparse = run(true);

        for (i, (d, s)) in dense.poses.iter().zip(sparse.poses.iter()).enumerate() {
            let dt = (d.translation - s.translation).length();
            assert!(dt < 1e-6, "camera {i} translation differs by {dt:e}");
            for r in 0..3 {
                for c in 0..3 {
                    let dr = (d.rotation.col(c)[r] - s.rotation.col(c)[r]).abs();
                    assert!(dr < 1e-6, "camera {i} rotation[{r}][{c}] differs by {dr:e}");
                }
            }
        }
        for (j, (d, s)) in dense.points.iter().zip(sparse.points.iter()).enumerate() {
            let dp = (*d - *s).length();
            assert!(dp < 1e-6, "point {j} differs by {dp:e}");
        }
        // And it must actually have been a non-trivial problem: 5 cameras, one fixed.
        assert_eq!(dense.poses.len(), 5);
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

    /// The depth prior's pose-translation Jacobian equals row 2 of R, verified against
    /// finite differences on `retract` itself.
    ///
    /// `retract` is rplus (`t <- t + R*upsilon`), so d z / d upsilon = e_z^T R = row 2 of R --
    /// the same vector as d z / d Xw. Writing `e_z` instead is correct only for a camera
    /// looking down world +Z, which is what every depth *scene* in this file builds
    /// (`translate_pose` uses `Mat3F64::IDENTITY`).
    ///
    /// Note a full-BA test does NOT catch this: measured, the wrong Jacobian still converges
    /// to ground truth at every yaw up to 179 degrees, because the depth term only has to
    /// break the 1-DOF scale gauge and any direction with a nonzero component along it does
    /// that, after which the (correct) reprojection Jacobian resolves the rest. The defect is
    /// therefore only observable in the derivative, which is what this test checks.
    #[test]
    fn depth_pose_translation_jacobian_matches_finite_differences() {
        // A camera yawed well away from world +Z, so e_z and row 2 of R differ sharply.
        let a = 55.0_f64.to_radians();
        let (s, c) = (a.sin(), a.cos());
        let r_cw = Mat3F64::from_cols(
            Vec3F64::new(c, 0.0, s),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(-s, 0.0, c),
        );
        let pose = Pose3d::new(r_cw, Vec3F64::new(0.13, -0.07, 0.31));
        let point = Vec3F64::new(0.42, -0.18, 3.7);

        let se3 = pose_to_se3(&pose);
        let z_of = |d: &[f32; 6]| -> f64 {
            let p = se3_to_pose(&se3.retract(d));
            p.transform_point(&point).z
        };

        // Central differences on the three translation tangent components.
        let h = 1e-4_f32;
        let mut numeric = [0.0_f64; 3];
        for (i, n) in numeric.iter_mut().enumerate() {
            let (mut dp, mut dm) = ([0.0_f32; 6], [0.0_f32; 6]);
            dp[i] = h;
            dm[i] = -h;
            *n = (z_of(&dp) - z_of(&dm)) / (2.0 * h as f64);
        }

        let rm = pose.rotation;
        let analytic = [rm.col(0).z, rm.col(1).z, rm.col(2).z];

        for i in 0..3 {
            assert!(
                (numeric[i] - analytic[i]).abs() < 1e-3,
                "d z/d upsilon[{i}]: finite difference {:.6}, row 2 of R {:.6}",
                numeric[i],
                analytic[i]
            );
        }

        // And confirm the old e_z form is genuinely wrong here, so this test cannot pass
        // vacuously if someone reintroduces it.
        let e_z = [0.0_f64, 0.0, 1.0];
        let max_dev = (0..3)
            .map(|i| (numeric[i] - e_z[i]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_dev > 0.1,
            "scene is too close to the degenerate e_z case to discriminate (dev {max_dev:.4})"
        );
    }

    /// The log depth residual is C¹ across the cheirality changeover at `z = s·m`.
    ///
    /// Both branches must agree in value AND slope there, or a point crossing the image plane
    /// mid-iteration hands LM a step discontinuity and the line search thrashes.
    #[test]
    fn depth_residual_log_branches_match_at_changeover() {
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

    /// A weak scale prior converges too slowly to be safe at realistic iteration counts.
    ///
    /// The regulariser removes the unsupported global-scale mode by geometric leak — each sweep
    /// multiplies it by `1/(1+λ)` — so a small λ *looks* best when the sweep is run to 100
    /// iterations and silently fails at the 10–30 a real pipeline budgets. Measured on the same
    /// scene: λ=0.1 leaves 1.76 m of error at 10 iterations and 0.06 m at 100; λ=1.0 is already at
    /// 0.058 m by iteration 10. This is why the default is 1.0 and not the sweep's argmin.
    #[test]
    fn schur_ba_weak_scale_prior_needs_more_iterations() {
        let (cam, true_poses, true_points, observations, init_poses, init_points) =
            per_camera_scale_scene();
        let max_err = |pts: &[Vec3F64]| -> f64 {
            pts.iter()
                .zip(&true_points)
                .map(|(a, b)| (*a - *b).length())
                .fold(0.0_f64, f64::max)
        };
        let run = |prior: f32, iters: usize| {
            let p = BaParams {
                max_iterations: iters,
                cost_tolerance: 1e-8,
                depth_log_residual: true,
                depth_scale_prior: prior,
                ..BaParams::default()
            };
            max_err(
                &bundle_adjust_schur(&init_poses, &init_points, &observations, &cam, &p)
                    .unwrap()
                    .points,
            )
        };
        let _ = &true_poses;

        let weak_short = run(0.1, 10);
        let weak_long = run(0.1, 100);
        let default_short = run(1.0, 10);

        assert!(
            weak_short > 10.0 * weak_long,
            "λ=0.1 should be far from converged at 10 iterations \
             (got {weak_short:.4} m vs {weak_long:.4} m at 100)"
        );
        assert!(
            default_short < 0.1,
            "λ=1.0 should be converged by 10 iterations, got {default_short:.4} m"
        );
    }

    /// A motion prior over cameras that share NO point must not kill the sparse factorisation.
    ///
    /// Regression test for a failure that only appears at coarse sampling. The sparse reduced
    /// camera system's pattern is built from COVISIBILITY, but a constant-velocity prior couples a
    /// triplet whether or not those cameras see common structure — so when they do not, the block
    /// it writes has no slot and Cholesky fails outright.
    ///
    /// Measured on a 3211-frame upload: at 6 Hz every consecutive keyframe pair shared points and
    /// this never fired; at 3 Hz some pairs stopped overlapping and every candidate died here, with
    /// 296 of 322 cameras already registered. Dropping such priors would be worse than failing —
    /// when consecutive keyframes share no point, the prior is the ONLY thing joining them.
    ///
    /// The scene below is built so cameras 0 and 2 share nothing: each sees its own disjoint set of
    /// points, and the prior spans all three.
    #[test]
    fn motion_prior_over_disjoint_cameras_survives_sparse_path() {
        let cam = PinholeCamera {
            fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0,
            k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
        };
        // Three cameras marching along +X.
        let poses: Vec<Pose3d> = (0..3)
            .map(|k| Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-(k as f64) * 0.5, 0.0, 0.0)))
            .collect();

        // Disjoint structure: camera k sees ONLY points [4k, 4k+4).
        let mut points = Vec::new();
        for k in 0..3 {
            for t in 0..4 {
                let f = t as f64;
                points.push(Vec3F64::new(
                    (k as f64) * 0.5 + f * 0.1 - 0.15,
                    f * 0.12 - 0.18,
                    3.0 + f * 0.05,
                ));
            }
        }
        let mut obs = Vec::new();
        for (k, pose) in poses.iter().enumerate() {
            for t in 0..4 {
                let pi = k * 4 + t;
                let pc = pose.transform_point(&points[pi]);
                obs.push(BaObservation {
                    pose_idx: k,
                    point_idx: pi,
                    pixel: [
                        (cam.fx * pc.x / pc.z + cam.cx) as f32,
                        (cam.fy * pc.y / pc.z + cam.cy) as f32,
                    ],
                    fixed_pose: k == 0,
                    fixed_point: false,
                    ..BaObservation::default()
                });
            }
        }
        // The prior spans 0,1,2 — cameras 0 and 2 share no observation whatsoever.
        let mps = [BaMotionPrior {
            i0: 0, i1: 1, i2: 2,
            alpha: 0.5,
            position_sigma: 0.1,
            orientation_sigma: 0.1,
        }];
        let params = BaParams {
            max_iterations: 5,
            sparse_reduced_system: true,
            ..BaParams::default()
        };
        let r = bundle_adjust_schur_with_all_priors(
            &poses, &points, &obs, &cam, &params, None, Some(&mps),
        );
        assert!(
            r.is_ok(),
            "sparse path rejected a motion prior over non-covisible cameras: {:?}",
            r.err()
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
                Some(BaPosePrior {
                    center_world: [c.x as f32, c.y as f32, c.z as f32],
                    sigma: 0.05,
                    up_world: None,
                    up_sigma: 0.0,
                    up_cam: [0.0, -1.0, 0.0],
                })
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

    /// Out-of-plane spread of a set of world→cam poses' camera centres, as a fraction of the
    /// largest spread. Gauge-free: a ratio of principal spreads is invariant to the similarity
    /// freedom a monocular reconstruction carries.
    fn flatness(poses: &[Pose3d]) -> f64 {
        let se3s: Vec<SE3F32> = poses.iter().map(pose_to_se3).collect();
        let local: Vec<i64> = (0..poses.len() as i64).collect();
        let Some((n, ctr)) = fit_centre_plane(&se3s, &local) else {
            return 0.0;
        };
        let cs: Vec<[f64; 3]> = poses
            .iter()
            .map(|p| {
                let c = p.inverse().translation;
                [c.x, c.y, c.z]
            })
            .collect();
        let out = cs
            .iter()
            .map(|c| (0..3).map(|k| n[k] * (c[k] - ctr[k])).sum::<f64>().abs())
            .fold(0.0f64, f64::max);
        let span = (0..3)
            .map(|k| {
                let v: Vec<f64> = cs.iter().map(|c| c[k]).collect();
                v.iter().copied().fold(f64::MIN, f64::max)
                    - v.iter().copied().fold(f64::MAX, f64::min)
            })
            .fold(0.0f64, f64::max);
        if span > 1e-12 {
            out / span
        } else {
            0.0
        }
    }

    /// The planarity residual's analytic Jacobian agrees with finite differences.
    ///
    /// This is the assertion the implementation actually needs. The end-to-end claim — "the prior
    /// recovers a bent walk" — CANNOT be demonstrated on a compact synthetic scene, and it is worth
    /// recording why rather than staging something that appears to show it:
    ///
    /// - With observations generated from the BENT poses, the bend is the exact global optimum of
    ///   reprojection. Flattening is not a similarity, so it strictly increases the residual and the
    ///   prior is correctly refused. Measured: 0.139 -> 0.137 at sigma 0.02.
    /// - With observations from the FLAT truth and the solve started bent, plain BA recovers
    ///   flatness 0.0000 unaided at every sigma tried. A 9-camera arc over a 25-point grid observes
    ///   its own vertical perfectly well; there is nothing for a prior to add.
    ///
    /// The prior earns its place only where out-of-plane motion is WEAKLY OBSERVED — hundreds of
    /// forward-walking keyframes with narrow co-visibility, which is the real clip and not something
    /// a unit test reconstructs honestly. That claim belongs to ground-truth validation.
    ///
    /// One thing the sweep did establish, and it matches the up-prior's history: an over-tight sigma
    /// HURTS. At 0.005 the prior dragged an otherwise-perfect solve to 0.0227 flatness. The failure
    /// mode of this family of priors is over-trust, not under-trust.
    #[test]
    fn plane_prior_jacobian_matches_finite_differences() {
        // Rodrigues from an axis-angle. Hand-written columns are not a rotation unless you check —
        // an earlier version of this test used three plausible-looking columns whose first and third
        // had a dot product of 0.039, and `pose_to_se3` silently produced a pose whose centre bore no
        // relation to the inputs, which reads exactly like a wrong Jacobian.
        let pose = {
            let axis = {
                let v = [0.42_f64, -0.31, 0.85];
                let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                [v[0] / l, v[1] / l, v[2] / l]
            };
            let th = 0.7_f64;
            let (s_, c_) = th.sin_cos();
            let k = Mat3F64::from_cols(
                Vec3F64::new(0.0, axis[2], -axis[1]),
                Vec3F64::new(-axis[2], 0.0, axis[0]),
                Vec3F64::new(axis[1], -axis[0], 0.0),
            );
            let r = Mat3F64::IDENTITY + k * s_ + (k * k) * (1.0 - c_);
            Pose3d::new(r, Vec3F64::new(0.35, -0.12, 1.7))
        };
        let se3 = pose_to_se3(&pose);
        let n = {
            let v = [0.31_f64, -0.62, 0.72];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let ctr = [0.11_f64, -0.04, 0.29];
        let sigma = 0.08_f64;

        let centre = |t: &SE3F32| -> [f64; 3] {
            let rm = t.r.matrix();
            let tt = t.t;
            let (c0, c1, c2) = (rm.col(0), rm.col(1), rm.col(2));
            [
                -f64::from(c0.x * tt.x + c0.y * tt.y + c0.z * tt.z),
                -f64::from(c1.x * tt.x + c1.y * tt.y + c1.z * tt.z),
                -f64::from(c2.x * tt.x + c2.y * tt.y + c2.z * tt.z),
            ]
        };
        let resid = |t: &SE3F32| -> f64 {
            let c = centre(t);
            (0..3).map(|k| n[k] * (c[k] - ctr[k])).sum::<f64>() / sigma
        };

        // Analytic: dr/drho = -n^T / sigma, dr/domega = (n x C)^T / sigma.
        let c = centre(&se3);
        let ncx = [
            n[1] * c[2] - n[2] * c[1],
            n[2] * c[0] - n[0] * c[2],
            n[0] * c[1] - n[1] * c[0],
        ];
        let analytic = [
            -n[0] / sigma,
            -n[1] / sigma,
            -n[2] / sigma,
            ncx[0] / sigma,
            ncx[1] / sigma,
            ncx[2] / sigma,
        ];

        // Central differences over the solver's OWN retraction, which is what the Jacobian must be
        // taken with respect to — a derivative correct for some other parameterisation would be
        // silently wrong here and would show up only as a slow LM.
        let eps = 1e-4_f32;
        for k in 0..6 {
            let mut dp = [0.0f32; 6];
            dp[k] = eps;
            let mut dm = [0.0f32; 6];
            dm[k] = -eps;
            let fd = (resid(&se3.retract(&dp)) - resid(&se3.retract(&dm))) / (2.0 * f64::from(eps));
            let tol = 1e-2 * analytic[k].abs().max(1.0);
            assert!(
                (fd - analytic[k]).abs() < tol,
                "component {k}: analytic {:.6} vs finite difference {:.6}",
                analytic[k],
                fd
            );
        }
    }

    /// A degenerate centre set must not be handed an invented plane.
    ///
    /// Collinear cameras — a straight corridor, or any short walk — admit no distinguishable
    /// normal: every plane through the line fits equally well. Penalising deviation from an
    /// arbitrary one of them would inject a constraint the data never supported, in a direction
    /// chosen by numerical noise.
    #[test]
    fn plane_fit_refuses_a_collinear_walk() {
        let along: Vec<SE3F32> = (0..9)
            .map(|i| {
                let x = i as f64 * 0.3;
                pose_to_se3(&Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(x, 0.0, 0.0)).inverse())
            })
            .collect();
        let local: Vec<i64> = (0..9).collect();
        assert!(
            fit_centre_plane(&along, &local).is_none(),
            "collinear centres must yield no plane"
        );

        let too_few: Vec<SE3F32> = along.iter().take(3).cloned().collect();
        assert!(
            fit_centre_plane(&too_few, &[0, 1, 2]).is_none(),
            "3 centres are always coplanar"
        );
    }

    /// Inferred sigma tracks the BOB, not the DRIFT — the property the whole scheme rests on.
    ///
    /// Two trajectories with the SAME high-frequency wobble, one of them additionally bent by a slow
    /// low-frequency drift 20x larger. The inferred sigma must be near-identical for both: if the
    /// drift leaked into it, the prior's tolerance would grow to match the error it exists to
    /// correct and the term would go inert exactly when it is needed.
    #[test]
    fn inferred_sigma_ignores_drift_and_tracks_the_bob() {
        let bob = 0.03_f64;
        let make = |drift: f64| -> (Vec<SE3F32>, Vec<i64>) {
            let n = 120usize;
            let poses: Vec<SE3F32> = (0..n)
                .map(|i| {
                    let u = i as f64 / (n - 1) as f64;
                    // A SMOOTH high-frequency bob (period 6 keyframes) plus a slow bend. Not a
                    // square wave: a two-valued signal has no central mass, so its MAD reports the
                    // peak-to-peak separation rather than the amplitude — an earlier version of this
                    // test alternated +-bob and the estimator dutifully returned 2.96x the amplitude.
                    // A real bob is continuous, and for a sine of amplitude A the robust scale is
                    // A/sqrt(2).
                    let z = bob * (std::f64::consts::TAU * i as f64 / 6.0).sin()
                        + drift * (std::f64::consts::PI * u).sin();
                    // The in-plane path must be functionally ORTHOGONAL to the drift or the plane
                    // fit simply tilts and absorbs it — a plane can mix the coordinates linearly, so
                    // any drift expressible as a linear combination of the in-plane shape is not
                    // out-of-plane at all. `0.7*sin(3u)` looked like a different axis but is a hump
                    // over u in [0,1], nearly collinear with the sin(pi*u) drift (measured: the
                    // "drifted" scene came out with a spread of 0.027 instead of 0.6). A fast
                    // zig-zag in y carries no low-frequency component for the drift to hide in, and
                    // sin(pi*u) is symmetric about u=0.5 so it is uncorrelated with x as well.
                    let centre =
                        Vec3F64::new(2.0 * u, 0.7 * (std::f64::consts::TAU * 2.5 * u).sin(), z);
                    pose_to_se3(&Pose3d::new(Mat3F64::IDENTITY, centre).inverse())
                })
                .collect();
            let local: Vec<i64> = (0..n as i64).collect();
            (poses, local)
        };

        let (flat, lf) = make(0.0);
        let (drifted, ld) = make(0.6);
        let (n0, c0) = fit_centre_plane(&flat, &lf).expect("plane");
        let (n1, c1) = fit_centre_plane(&drifted, &ld).expect("plane");
        let s_flat = infer_plane_sigma(&flat, &lf, &n0, &c0).expect("sigma");
        let s_drift = infer_plane_sigma(&drifted, &ld, &n1, &c1).expect("sigma");

        // A sine of amplitude A has robust scale ~A/sqrt(2); allow a factor of two either way,
        // because the exact constant is a property of the waveform and not of the claim being made.
        assert!(
            s_flat > 0.5 * bob / std::f64::consts::SQRT_2 && s_flat < 2.0 * bob,
            "sigma should be within a factor of 2 of the bob {bob:.4}, got {s_flat:.4}"
        );
        assert!(
            (s_drift / s_flat - 1.0).abs() < 0.35,
            "drift leaked into the inferred sigma: {s_flat:.4} (flat) vs {s_drift:.4} (drifted, \
             0.6 of low-frequency bend added)"
        );
        // And the global spread must be dominated by that drift, or the scene proves nothing.
        let spread = |se3s: &[SE3F32], n: &[f64; 3], c: &[f64; 3]| -> f64 {
            se3s.iter()
                .map(|p| {
                    let rm = p.r.matrix();
                    let t = p.t;
                    let (a, b, d) = (rm.col(0), rm.col(1), rm.col(2));
                    let cc = [
                        -f64::from(a.x * t.x + a.y * t.y + a.z * t.z),
                        -f64::from(b.x * t.x + b.y * t.y + b.z * t.z),
                        -f64::from(d.x * t.x + d.y * t.y + d.z * t.z),
                    ];
                    (0..3).map(|k| n[k] * (cc[k] - c[k])).sum::<f64>().abs()
                })
                .fold(0.0f64, f64::max)
        };
        let g = spread(&drifted, &n1, &c1);
        assert!(
            g > 5.0 * s_drift,
            "drifted scene is not actually drifted (spread {g:.3})"
        );
    }
}
