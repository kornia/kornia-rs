//! Input/output types for multi-camera calibration.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Vec2F64, Vec3F64};

use crate::error::CalibError;

/// One AprilTag observed by one or more cameras.
///
/// Corners are given in **aruco winding** `(TL, TR, BR, BL)` in the raw
/// (possibly distorted) image pixels of each camera. The same physical corner
/// index must be used across every camera that observes the tag.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct FeatureTrack {
    /// `(camera_index, raw_pixel)`, one per camera observing this point (at least two).
    pub obs: Vec<(usize, Vec2F64)>,
}

/// A single feature correspondence: the same physical scene point seen by two
/// cameras, given as the source pixel in each. Feature matches are optional;
/// they add the rotation/translation constraints a single planar tag lacks.
#[derive(Debug, Clone)]
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
    /// Minimum parallax (degrees) for a triangulated free point.
    pub min_parallax_deg: f64,
    /// Maximum reprojection error (normalized units) when validating a
    /// triangulated point.
    pub max_reprojection_error: f64,
    /// X84 cutoff multiplier: after a Huber warm-start, free-point (feature) observations whose
    /// residual exceeds `median + x84_k·1.4826·MAD` are dropped before the final Cauchy pass. `2.5`
    /// is the standard X84 value; fixed board/tag corners (the gauge) are never dropped.
    pub x84_k: f64,
    /// Relative standard deviation of the per-observation metric depth priors passed to
    /// [`crate::reconstruct`], as `σ = depth_prior_rel_sigma · d`. **`0.0` (the default) disables
    /// depth entirely**; `0.15` trusts a monocular depth network to ~15%.
    ///
    /// Depth is what makes a monocular reconstruction metric without a fiducial and — pinning every
    /// segment of the chain rather than one global average — what stops scale drifting along a
    /// no-revisit walkthrough. Deflated into reprojection units; see `sfm.rs::depth_fields`.
    pub depth_prior_rel_sigma: f64,
    /// Re-gauge each view's depth priors by its OWN fitted scale before bundle adjustment, instead
    /// of trusting a single global scale. `true` by default; inert unless
    /// [`Self::depth_prior_rel_sigma`] is non-zero and depths are supplied.
    ///
    /// Learned depth is not gauge-stable frame to frame, and on forward motion a per-frame scale
    /// error is indistinguishable from along-axis translation — so one global scale hands every
    /// wander of the network to the trajectory. See `sfm.rs::fit_depth_scales`.
    pub depth_per_keyframe_scale: bool,
    /// Standard deviation (unit-vector units) of a per-view prior pulling each camera's image-up
    /// toward world up. **`0.0` (the default) disables it** — rig cameras are not all upright, so
    /// this must stay off for rig calibration.
    ///
    /// On handheld walkthrough video with no revisits it is the only observation of global
    /// orientation the capture carries. `0.25` tolerates ~14° of genuine hand tilt at 1σ.
    pub up_prior_sigma: f64,
    /// Constant-velocity motion-prior sigma over consecutive registered triplets. **`0.0` (the
    /// default) disables it.**
    ///
    /// The value is the sigma of the translation norm-RATIO residual (unitless — see
    /// [`kornia_3d::ba::BaMotionPrior`]); the angular sigma is half of it, in radians. `0.1`
    /// tolerates ~10% deviation from constant velocity at 1σ. Only meaningful when view indices are
    /// TIME-ORDERED, which is why it is off for a rig.
    pub motion_prior_sigma: f64,
    /// Absolute PnP inlier count required to register a view (COLMAP's
    /// `abs_pose_min_num_inliers`). Default `30`.
    ///
    /// PnP-RANSAC succeeds whenever it finds *a* consensus, and a thin registration is not a
    /// self-correcting mistake: new points are triangulated FROM the bad pose and then feed the
    /// next view's PnP. Set to `0` for the pre-gate behaviour of accepting any successful PnP.
    pub min_registration_inliers: usize,
    /// Refine a global focal scale plus radial/tangential distortion against the reconstruction,
    /// alternating with bundle adjustment. **`false` by default**; leave it off when the intrinsics
    /// are calibrated.
    ///
    /// For guessed intrinsics (a FOV sweep, `k1` assumed zero on an ultra-wide) the error bends the
    /// whole reconstruction in a way no reprojection residual reveals. The fit is reported on
    /// [`Reconstruction::camera_correction`] and a second BA re-settles the geometry against it.
    pub refine_intrinsics: bool,
    /// Called after each view is registered, as `(registered_so_far, total_views)`. `None` by
    /// default.
    pub progress: Option<std::sync::Arc<dyn Fn(usize, usize) + Send + Sync>>,
}

impl CalibConfig {
    /// Config with flux-derived defaults for the given tag size (metres).
    ///
    /// Every prior defaults to off, so none of them touches a solve unless a caller opts in. The
    /// one exception is [`Self::min_registration_inliers`], which defaults to COLMAP's `30` rather
    /// than `0`: views whose PnP found only a handful of inliers are now refused where they were
    /// previously admitted. Set it to `0` for the old behaviour.
    pub fn new(tag_size_m: f64) -> Self {
        Self {
            tag_size_m,
            max_iterations: 40,
            robust_scale_sq: (0.01f32).powi(2),
            min_parallax_deg: 0.2,
            max_reprojection_error: 0.01,
            x84_k: 2.5,
            depth_prior_rel_sigma: 0.0,
            depth_per_keyframe_scale: true,
            up_prior_sigma: 0.0,
            motion_prior_sigma: 0.0,
            min_registration_inliers: 30,
            refine_intrinsics: false,
            progress: None,
        }
    }
}

/// Per-camera pose covariance + observability, conditional on fixed intrinsics + fixed target
/// geometry. Computed from each camera's fixed-point (board/gauge) reprojections.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct RigCalibration {
    /// Per-camera pose `T_world_cam` (camera optical frame → world). `None`
    /// for a camera that did not observe the reference tag. World frame = the
    /// reference tag's frame.
    pub poses: Vec<Option<Pose3d>>,
    /// Id of the tag chosen as the world/gauge anchor, for the tag and board producers.
    ///
    /// On the feature/SfM path ([`crate::reconstruct`]) the world frame is the bootstrap view's
    /// and the tag only fixes scale, so this names the scale bar rather than a gauge anchor.
    /// `0` doubles as "no tag", which collides with a real tag id -- see
    /// [`ScaleSource`] and the `From<Reconstruction>` impl.
    pub reference_tag_id: u16,
    /// Final reprojection RMS in **pixels** (each residual scaled by its own
    /// camera's focal). `-1.0` if no valid observation remained.
    pub reproj_rmse_px: f64,
    /// Per-camera covariance + observability (one entry per camera; see [`CameraStats`]).
    pub per_camera: Vec<CameraStats>,
}

/// A reconstructed world point and the input track it came from.
///
/// One struct rather than parallel `points` / `point_track_id` vectors, so the pairing cannot come
/// apart: there is no length to keep in sync and no way to reorder one without the other. The
/// track id is what lets a caller carry its own per-track data -- descriptors, colours, semantic
/// labels -- onto the map without re-matching.
#[derive(Debug, Clone, Copy)]
pub struct Point {
    /// Position in the world frame, after bundle adjustment and after metric scaling when the
    /// reconstruction's [`ScaleSource`] is [`ScaleSource::Tag`].
    pub position: Vec3F64,
    /// Index of the input track this point was reconstructed from.
    pub track_id: usize,
}

/// One image measurement that survived into the final solve: view `view` saw point `point` at
/// pixel `pixel`.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    /// Index into [`Reconstruction::views`].
    pub view: usize,
    /// Index into [`Reconstruction::points`].
    pub point: usize,
    /// Raw image pixel, in the same convention the input tracks used.
    pub pixel: Vec2F64,
}

/// How the reconstruction's metric scale was fixed.
///
/// A feature-only reconstruction is determined up to a similarity, so the scale has to come from
/// somewhere outside the images. Reporting WHICH is not cosmetic: a caller that mistakes an
/// up-to-scale map for a metric one gets distances that are self-consistent and wrong.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleSource {
    /// A tag of known side length anchored the scale; [`Reconstruction::points`] are in metres.
    Tag {
        /// Id of the tag used as the metric anchor.
        id: u16,
        /// Its known side length, in metres.
        size_m: f64,
    },
    /// No metric anchor was available. Poses and points are consistent with each other but carry
    /// an arbitrary global scale.
    UpToScale,
}

/// A feature-based reconstruction: the map, not just the cameras.
///
/// The solver computes all of this; the earlier API threw most of it away to return a
/// [`RigCalibration`]. That is the right shape for rig calibration, where the cameras ARE the
/// answer, and the wrong one for mapping, where the points, the correspondence between tracks and
/// points, and the surviving observations are the answer. Downstream code needing the map had to
/// re-derive it or fork; [`reconstruct`](crate::reconstruct) returns it.
///
/// [`RigCalibration`] remains reachable via `From` for the rig-calibration use case.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Reconstruction {
    /// Per-view pose `T_world_cam` (camera optical frame → world). `None` for a view that could
    /// not be registered.
    pub views: Vec<Option<Pose3d>>,
    /// Reconstructed world points, each carrying the input track it came from. Published in
    /// ascending track order, so two runs over identical input agree.
    pub points: Vec<Point>,
    /// The observations that survived into the final solve. At most the input observation count,
    /// and usually fewer: tracks that never triangulated, and views that never registered,
    /// contribute none. Equal only when every track triangulates and every view registers.
    pub observations: Vec<Observation>,
    /// Final reprojection RMS in pixels, or `-1.0` if no valid observation remained.
    pub reproj_rmse_px: f64,
    /// Per-view covariance + observability, one entry per input view.
    pub per_view: Vec<CameraStats>,
    /// Where the metric scale came from -- see [`ScaleSource`].
    pub scale: ScaleSource,
    /// Intrinsics correction the solve FITTED, as `(gamma, k1, k2, p1, p2)`. `None` when
    /// [`CalibConfig::refine_intrinsics`] was off, the fit was singular, or it landed outside its
    /// sanity bounds.
    ///
    /// Load-bearing output, not telemetry: refinement applies itself by rewriting the NORMALIZED
    /// observations in place, leaving the caller's camera model at its original guess — a caller
    /// that stores that guess alongside these poses records a camera that did not produce them.
    ///
    /// `gamma` scales normalized coordinates, so it INVERTS into focal length:
    /// `fx_true = fx_assumed / gamma`. The distortion terms are OpenCV's, to be applied to the
    /// assumed camera's existing model.
    pub camera_correction: Option<(f64, f64, f64, f64, f64)>,
}

impl From<Reconstruction> for RigCalibration {
    /// Keep only what rig calibration needs. Lossy by construction, and in two ways worth stating.
    ///
    /// The obvious one: the points, the track correspondence and the observations are dropped.
    ///
    /// The consequential one: [`ScaleSource`] collapses into a bare `reference_tag_id`, using `0`
    /// to mean "no tag". `0` is a VALID tag id, so a `RigCalibration` consumer cannot distinguish
    /// "metric, anchored by tag 0" from "up to scale, no tag involved" -- the exact ambiguity
    /// `ScaleSource` exists to remove. Keep the [`Reconstruction`] if that distinction matters.
    ///
    /// Note also that on the feature/SfM path the world frame is the bootstrap view's, not the
    /// tag's: the tag is a scale bar, so `reference_tag_id` names a tag that fixed the metric
    /// rather than one that anchored the gauge.
    fn from(r: Reconstruction) -> Self {
        RigCalibration {
            poses: r.views,
            reference_tag_id: match r.scale {
                ScaleSource::Tag { id, .. } => id,
                ScaleSource::UpToScale => 0,
            },
            reproj_rmse_px: r.reproj_rmse_px,
            per_camera: r.per_view,
        }
    }
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
