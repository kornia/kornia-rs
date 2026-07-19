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
    /// Minimum parallax (degrees) for a triangulated free point.
    pub min_parallax_deg: f64,
    /// Maximum reprojection error (normalized units) when validating a
    /// triangulated point.
    pub max_reprojection_error: f64,
    /// X84 cutoff multiplier: after a Huber warm-start, free-point (feature) observations whose
    /// residual exceeds `median + x84_k·1.4826·MAD` are dropped before the final Cauchy pass. `2.5`
    /// is the standard X84 value; fixed board/tag corners (the gauge) are never dropped.
    pub x84_k: f64,
}

impl CalibConfig {
    /// Config with flux-derived defaults for the given tag size (metres).
    pub fn new(tag_size_m: f64) -> Self {
        Self {
            tag_size_m,
            max_iterations: 40,
            robust_scale_sq: (0.01f32).powi(2),
            min_parallax_deg: 0.2,
            max_reprojection_error: 0.01,
            x84_k: 2.5,
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
