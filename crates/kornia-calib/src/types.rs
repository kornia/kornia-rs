//! Input/output types for multi-camera calibration.

use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec2F64;

use crate::error::CalibError;

/// One AprilTag observed by one or more cameras.
///
/// Corners are given in **aruco winding** `(TL, TR, BR, BL)` in the raw
/// (possibly distorted) image pixels of each camera. The same physical corner
/// index must be used across every camera that observes the tag.
pub struct TagObservation {
    /// Tag id (as decoded by the detector).
    pub tag_id: u16,
    /// `(camera_index, [corner0..3])` for every camera that saw this tag.
    pub per_camera: Vec<(usize, [Vec2F64; 4])>,
}

/// A multi-view feature track: one observation of the same physical scene point per camera that
/// sees it. Replaces the `C(k,2)` independent two-view points (one per camera pair) with a single
/// shared 3D point — removing statistical double-counting of each pixel and coupling the poses
/// through shared structure. Build these from pairwise matches with [`crate::build_tracks`].
pub struct FeatureTrack {
    /// `(camera_index, raw_pixel)`, one per camera observing this point (at least two).
    pub obs: Vec<(usize, Vec2F64)>,
}

/// A single feature correspondence: the same physical scene point seen by two
/// cameras, given as the source pixel in each. Feature matches are optional;
/// they add the rotation/translation constraints a single planar tag lacks.
pub struct FeatureMatch {
    /// First camera index.
    pub cam_a: usize,
    /// Second camera index.
    pub cam_b: usize,
    /// Pixel of the point in camera `cam_a`.
    pub uv_a: Vec2F64,
    /// Pixel of the point in camera `cam_b`.
    pub uv_b: Vec2F64,
}

/// Tunable parameters for [`crate::calibrate_apriltag`].
///
/// Note: `robust_scale_sq`, `min_parallax_deg`, and `max_reprojection_error`
/// live in **normalized** image coordinates because bundle adjustment runs on
/// an identity pinhole (per-camera intrinsics are folded into the observations).
pub struct CalibConfig {
    /// AprilTag side length in metres (sets absolute metric scale).
    pub tag_size_m: f64,
    /// Maximum bundle-adjustment LM iterations.
    pub max_iterations: usize,
    /// Squared Huber scale (normalized units). `(0.01)^2` ≈ 5 px at focal 500.
    pub robust_scale_sq: f32,
    /// Run a second registration pass against the bundle-adjusted map.
    ///
    /// Raises coverage substantially where the first pass was conservative (7-Scenes chess: 23/60
    /// -> 59/60 registered), but on scenes with repetitive structure the extra registrations are
    /// wrong and wreck the result (stairs: 1.68 -> 23.19 cm ATE). It therefore belongs in a
    /// caller's search rather than being always-on.
    pub second_pass: bool,
    /// Which bootstrap seed pair to use, `0` being the best-scoring one.
    ///
    /// The seed fixes the gauge, seeds the point cloud, and decides which cameras can register at
    /// all — so it dominates the result, and no scalar computed from the pair alone reliably picks
    /// the winner (parallax, RH ratio and support have each been tried as the ranking criterion and
    /// each traded one scene for another). Exposing the rank lets a caller reconstruct against the
    /// top few seeds and choose on the finished map instead. Ranks past the last viable seed
    /// saturate rather than fail.
    pub seed_rank: usize,
    /// Minimum parallax (degrees) for a triangulated free point.
    pub min_parallax_deg: f64,
    /// Maximum reprojection error (normalized units) when validating a
    /// triangulated point.
    pub max_reprojection_error: f64,
    /// X84 cutoff multiplier: after a Huber warm-start, free-point (feature) observations whose
    /// residual exceeds `median + x84_k·1.4826·MAD` are dropped before the final Cauchy pass. `2.5`
    /// is the standard X84 value; fixed board/tag corners (the gauge) are never dropped.
    pub x84_k: f64,
    /// Gravity-prior strength for video reconstruction: standard deviation (unit-vector units) of
    /// a per-pose prior pulling each camera's image-up toward the reference camera's image-up.
    ///
    /// `0` disables (the rig-calibration default — rig cameras are NOT all upright). For handheld
    /// walkthrough video this is the only absolute-orientation observation the capture carries:
    /// with no revisits, rotational drift accumulates freely along the chain (measured 30 degrees
    /// of pitch/roll drift across a 45 s clip), and no amount of reprojection cost can see it.
    /// `0.25` tolerates ~14 degrees of genuine hand tilt at 1 sigma.
    pub up_prior_sigma: f64,
    /// Relative sigma of metric depth priors (`σ = rel · d`); 0 disables. See
    /// `calibrate_features_with_depth`. `0.15` trusts a monocular depth network to ~15%.
    pub depth_prior_rel_sigma: f64,
    /// Constant-velocity motion-prior sigma over consecutive registered triplets; 0 disables.
    ///
    /// The value is the sigma of the translation norm-RATIO residual (unitless, see
    /// `BaMotionPrior`); the angular-velocity sigma is derived as half of it in radians. `0.1`
    /// tolerates ~10% deviation from constant-velocity at 1 sigma — loose enough for a human
    /// walking pace, tight enough to suppress the frame-to-frame scale/rotation jitter that
    /// accumulates into drift on GPS-less no-revisit video.
    pub motion_prior_sigma: f64,
    /// Refine global focal scale + leading radial distortion (k1) against the reconstruction,
    /// alternating with the bundle adjustment (see the closed-form fit in `sfm.rs`). For guessed
    /// phone intrinsics; leave OFF when intrinsics are calibrated.
    pub refine_intrinsics: bool,
    /// Absolute PnP inlier count required to register a view (COLMAP's
    /// `abs_pose_min_num_inliers`, default 30).
    ///
    /// 30 is calibrated for COLMAP-resolution imagery. On low-resolution video the correspondence
    /// pool at hard transitions (blurred doorway pans) runs 40-70, and the measured margin at the
    /// gate can be ZERO — one clip's coverage hinged on a boundary view passing with exactly 30
    /// inliers, and any perturbation (e.g. enabling depth priors, which trade a few percent of
    /// local fit) collapsed 190/200 to 88/200. The post-BA filter + de-registration hygiene is
    /// what polices bad registrations now; the gate only needs to reject garbage.
    pub min_registration_inliers: usize,
    /// Cameras left FREE in each periodic bundle adjustment during growth; 0 optimises all of them.
    ///
    /// The Schur reduction solves a DENSE `6P x 6P` reduced camera system, so a global BA costs
    /// `O(P^3)` per LM iteration. That is the right choice at rig scale — 8 cameras, or even the ~170
    /// this solver was designed around, give a 1020x1020 matrix a dense Cholesky eats in milliseconds.
    /// Per-frame video is a different regime: 1095 keyframes make it 6570x6570, which is 345 MB and
    /// ~9e10 flops PER ITERATION, times `max_iterations`, times every periodic BA.
    ///
    /// Windowing makes that cost constant rather than cubic: only the most co-visible
    /// `local_ba_window` cameras around the newly registered ones stay free, the rest are pinned via
    /// `BaObservation::fixed_pose`, and the reduced system shrinks to `6 x window`. Points seen only by
    /// pinned cameras are pinned too — optimising them would move the very structure the free cameras
    /// are being fitted against. This is the standard local-BA arrangement (ORB-SLAM's local window);
    /// correcting accumulated global error remains the terminal BA's job.
    ///
    /// Defaults to 0 so existing callers keep exactly the behaviour they were tuned against.
    pub local_ba_window: usize,
    /// Called after each camera is registered, as `(registered_so_far, total_cameras)`.
    ///
    /// Registration is the longest phase of a large solve and the only one whose duration is
    /// data-dependent: a caller driving a progress UI has nothing to report for tens of minutes
    /// without this, and cannot tell a solve that is growing from one that has stalled at a
    /// barrier. `None` by default, so existing callers are unaffected.
    pub progress: Option<std::sync::Arc<dyn Fn(usize, usize) + Send + Sync>>,
    /// Re-gauge each keyframe's monocular depth priors by its own fitted scale before every bundle
    /// adjustment, instead of trusting one global scale for the whole map.
    ///
    /// Learned depth is not gauge-stable frame to frame, and on forward motion a per-frame scale
    /// error is indistinguishable from along-axis translation — so a global scale hands every wander
    /// straight to the trajectory. See `fit_depth_scales` for why this is scale-only rather than the
    /// affine fit ViPE uses.
    pub depth_per_keyframe_scale: bool,
    /// Let bundle adjustment OPTIMISE the per-keyframe depth scales instead of freezing them at the
    /// fitted median, and switch the depth residual to log space. Requires
    /// [`Self::depth_per_keyframe_scale`]; ignored without it.
    ///
    /// The value is the strength of the prior holding each scale near its fitted seed, as a
    /// multiple of that keyframe's own depth weight — see `kornia_3d::ba::BaParams::depth_scale_prior`.
    /// `< 0` (the default) keeps the fitted-then-frozen behaviour.
    ///
    /// Why freezing is a trap: a scale fitted from the CURRENT geometry inherits that geometry's
    /// drift, so the prior then certifies the error it was supposed to correct. Letting BA move the
    /// scale breaks that circularity — but only while the prior is present. At `0` the scales
    /// absorb the whole depth residual and the term goes inert (measured: 5.7 m of point error
    /// against 0.08 m frozen, on the synthetic in `ba_schur`).
    ///
    /// `1.0` is the calibrated default: converged by ~10 LM iterations, where `0.1` still carries
    /// 1.8 m of error at 10 and needs 50+ to settle. Follows VidMap (ECCV 2026) eq. 3–4.
    pub depth_scale_prior: f64,
    /// MEASURED gravity direction per camera, in that camera's own frame (OpenCV axes, so roughly
    /// `+y` for an upright camera). `None` per entry where it could not be measured.
    ///
    /// Turns the "up" prior from an assumption into an observation. Without it the prior asserts a
    /// FIXED camera-frame up of `(0,−1,0)` for every view — "the phone was held perfectly upright,
    /// identically, all walk" — which is measurably false: a reference clip averages `(0.03, 0.93,
    /// 0.38)`, tilted 22 degrees. At a 3-degree sigma that is a 22-degree lie asserted on every
    /// frame.
    ///
    /// It matters far beyond correcting a bias. On a single forward pass with no revisits, global
    /// orientation is UNOBSERVABLE from correspondences alone, so accumulated tilt drift has nothing
    /// to bound it — measured consequence: 4.14 m of camera-height variation on a one-floor walk,
    /// with no axis identifiable as vertical. An external reference is the only fix, and with no IMU
    /// the scene's own verticals are the one available.
    pub gravity_cam: Option<std::sync::Arc<Vec<Option<[f64; 3]>>>>,
    /// Sigma for the up residual when it carries a MEASUREMENT rather than the upright assumption.
    ///
    /// Deliberately far looser than `up_prior_sigma`. A prior's sigma must state the ESTIMATOR's
    /// accuracy, not the precision one would like: vanishing-point gravity lands at 1-2 degrees at
    /// best, and a hand-rolled detector measured 7.6 degrees off OpenCV's LSD on the same frames.
    /// Asserting that at the 0.05 (~3 degree) sigma used for the assumption collapsed a
    /// reconstruction outright — extents 6.6 m -> 1.9 m, trajectory 62.5 m -> 12.7 m, 24 cameras
    /// lost — because bundle adjustment believes what it is told, and a confident wrong rotation
    /// drags the points and then the scale with it.
    ///
    /// 0.15 is ~8.6 degrees: enough for gravity to break the tilt-drift degeneracy that
    /// correspondences alone cannot see, not enough to overrule them.
    pub gravity_sigma: f64,
    /// Re-admit observations the reprojection filter removed, once the poses that judged them have
    /// improved (COLMAP's `CompleteTracks`; see `complete_tracks` in `sfm.rs`).
    ///
    /// OFF by default because it is measured to help on short sequences and HURT on long ones, and
    /// the failure is invisible to every aggregate metric.
    ///
    /// - 7-Scenes pumpkin (110 keyframes, one room): ATE 16.56 -> 4.24 cm, points 2374 -> 5192,
    ///   health FAIL -> WARN. chess and fire byte-identical.
    /// - One house, 643 keyframes, identical argv: residual and track length both IMPROVED —
    ///   rmse_honest 2.342 -> 1.944 px, observations +2.9%, tracks spanning >= 100 keyframes
    ///   601 -> 1459 — while the map SEVERED. `cut_min` 17 -> 0, largest step 17.9x -> 76.9x the
    ///   median, and a revisit that shared 399 points across kf150<->252 shared ZERO.
    ///
    /// The two are the same event. Points fell 3.9% while observations rose: tracks MERGED. A merge
    /// buys exactly what the aggregates reward — longer tracks, lower residual, since one fused
    /// track fits better than the two distinct points it swallowed — so on a clip with revisits,
    /// re-admitting at the creation threshold (tighter than the filter's 2x hysteresis) can fuse
    /// geometry the wider gate had deliberately kept apart. Long tracks starting in kf100-199 went
    /// 1280 -> 1; the revisit was not outvoted, it was dissolved.
    ///
    /// Enable for short clean sequences. On anything with a revisit, verify `cut_min` did not move.
    pub complete_tracks: bool,
    /// Penalise camera centres for leaving their own best-fit plane, in map units. 0 disables.
    ///
    /// See [`kornia_3d::ba::BaParams::plane_prior_sigma`]. Applies to the LOCAL and GLOBAL bundle
    /// adjustments alike: the constraint is a statement about the whole trajectory's shape, so a
    /// windowed BA that ignored it would undo between windows what the global one imposed.
    pub plane_prior_sigma: f64,
}

impl CalibConfig {
    /// Config with the default solver settings for the given tag size (metres).
    pub fn new(tag_size_m: f64) -> Self {
        Self {
            tag_size_m,
            max_iterations: 40,
            robust_scale_sq: (0.01f32).powi(2),
            min_parallax_deg: 0.2,
            max_reprojection_error: 0.01,
            x84_k: 2.5,
            up_prior_sigma: 0.0,
            depth_prior_rel_sigma: 0.0,
            motion_prior_sigma: 0.0,
            refine_intrinsics: false,
            min_registration_inliers: 30,
            second_pass: false,
            seed_rank: 0,
            local_ba_window: 0,
            progress: None,
            depth_per_keyframe_scale: true,
            depth_scale_prior: -1.0,
            gravity_cam: None,
            gravity_sigma: 0.15,
            complete_tracks: false,
            plane_prior_sigma: 0.0,
        }
    }
}

/// Per-camera pose covariance + observability, conditional on fixed intrinsics + fixed target
/// geometry. Computed from each camera's fixed-point (board/gauge) reprojections.
pub struct CameraStats {
    /// Camera index.
    pub camera: usize,
    /// Did this camera observe the target (get a pose)?
    pub registered: bool,
    /// Number of fixed-point observations used for the covariance.
    pub num_obs: usize,
    /// This camera's reprojection RMS in **pixels** over its fixed-point observations (`-1` if none).
    pub reproj_rmse_px: f64,
    /// Rotation standard deviation (**degrees**): √trace of the rotation covariance block. `∞` if
    /// unregistered / under-constrained.
    pub rot_sigma_deg: f64,
    /// Translation standard deviation (**metres**): √trace of the translation covariance block.
    pub trans_sigma_m: f64,
    /// Smallest eigenvalue of the non-dimensionalized pose Hessian. Near zero ⇒ an **unobservable**
    /// DOF (e.g. the in-plane tilt a single planar tag cannot fix). Larger ⇒ better constrained.
    /// `None` when there is no Hessian to analyse — an unregistered camera, or the empirical
    /// (multi-shot) path whose uncertainty is a measured sample spread, not a linearized covariance.
    pub min_eigenvalue: Option<f64>,
    /// Eigenvector of `min_eigenvalue` — the weakest direction `[ωx,ωy,ωz, νx,νy,νz]` in the camera
    /// frame. `None` in the same cases as [`CameraStats::min_eigenvalue`].
    pub weakest_dof: Option<[f64; 6]>,
}

impl CameraStats {
    /// An entry for a camera with no usable pose covariance: unregistered, or too few fixed-point
    /// observations to form the `6×6` Hessian. All uncertainty fields are the "unknown" sentinels.
    pub(crate) fn unconstrained(camera: usize, registered: bool, num_obs: usize) -> Self {
        Self {
            camera,
            registered,
            num_obs,
            reproj_rmse_px: -1.0,
            rot_sigma_deg: f64::INFINITY,
            trans_sigma_m: f64::INFINITY,
            min_eigenvalue: None,
            weakest_dof: None,
        }
    }
}

/// Result of a multi-camera extrinsic calibration.
pub struct RigCalibration {
    /// Per-camera pose `T_world_cam` (camera optical frame → world). `None`
    /// for a camera that did not observe the reference tag. World frame = the
    /// reference tag's frame.
    pub poses: Vec<Option<Pose3d>>,
    /// Id of the tag chosen as the world/gauge anchor.
    pub reference_tag_id: u16,
    /// Final reprojection RMS in **pixels** (each residual scaled by its own
    /// camera's focal). `-1.0` if no valid observation remained.
    pub reproj_rmse_px: f64,
    /// Per-camera covariance + observability (one entry per camera; see [`CameraStats`]).
    pub per_camera: Vec<CameraStats>,
    /// Intrinsics correction the solve FITTED, as `(gamma, k1, k2, p1, p2)`. `None` when refinement
    /// was disabled, found nothing, or produced a fit outside its sanity bounds.
    ///
    /// Load-bearing, and previously discarded. Refinement applies itself by rewriting the normalized
    /// observations in place — `n' = gamma * (n * radial + tangential)` — so the SOLVE gets the
    /// corrected camera while the caller's own camera model stays at whatever guess it started from.
    /// A caller that then stores its guess alongside these poses is recording a camera that did not
    /// produce them, and nothing downstream can detect the mismatch.
    ///
    /// `gamma` scales normalized coordinates, so it INVERTS into focal length:
    /// `fx_true = fx_assumed / gamma`. gamma < 1 means the true focal is LONGER than assumed.
    ///
    /// Why this matters more than a few percent suggests: an under-estimated focal inflates every
    /// recovered rotation by `f_true / f_assumed`, which integrates along a walk into a slow
    /// trajectory bend that no reprojection residual can see. Measured on a 3208-frame handheld clip
    /// whose container metadata gave 1440 px against a true focal near 1669 (+15.9%): 81.4% of the
    /// trajectory's out-of-plane residual fitted a paraboloid with 2.40 m of sagitta. Rebuilding at
    /// the corrected focal collapsed that to 7.9% and 0.171 m, took `cut_min` from 55 to 319 and
    /// reprojection rmse from 2.255 to 1.956 px. Every quality number the build printed looked
    /// healthy in BOTH cases, which is exactly why this has to be reported rather than inferred.
    pub camera_correction: Option<(f64, f64, f64, f64, f64)>,
}

impl RigCalibration {
    /// Re-express every camera pose so the `reference` camera's frame becomes
    /// the world origin, optionally pre-multiplied by a world `gauge` (e.g. a
    /// z-up re-gauging). Returns `T_world'_cam` per camera, `None` where the
    /// camera was not solved.
    ///
    /// `world' = gauge ∘ T_ref⁻¹`, so `poses[reference]` maps to `gauge`
    /// (identity when `gauge` is `None`).
    pub fn rebased(
        &self,
        reference: usize,
        gauge: Option<Pose3d>,
    ) -> Result<Vec<Option<Pose3d>>, CalibError> {
        let ref_pose = self
            .poses
            .get(reference)
            .copied()
            .flatten()
            .ok_or(CalibError::ReferenceNotSolved(reference))?;
        let g = gauge.unwrap_or(Pose3d::IDENTITY);
        let ref_inv = ref_pose.inverse();
        Ok(self
            .poses
            .iter()
            .map(|p| p.as_ref().map(|pc| g.compose(&ref_inv.compose(pc))))
            .collect())
    }
}
