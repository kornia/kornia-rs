//! Projective point-to-plane ICP on RGB-D pyramids.
//!
//! The method is the projective point-to-plane ICP of KinectFusion (Newcombe et al., "KinectFusion:
//! Real-time dense surface mapping and tracking", ISMAR 2011, section 3.5), with the
//! point-to-plane error of Chen and Medioni ("Object modelling by registration of multiple range
//! images", 1992). Each valid source pixel `p_s` is moved by the current estimate
//! `T_target_source = (R, t)`, `p = R p_s + t`, projected into the target image and matched to
//! the target vertex `v_t` and normal `n_t` of the pixel it lands on (no kd-tree). The residual
//! is `r = n_t . (p - v_t)` in metres.
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
//! `R <- exp(w) R`, `t <- exp(w) t + v`. A system whose Cholesky pivots collapse relative to
//! their block is rejected as singular rather than solved along its null space. The `J^T W J` of
//! a final pass at the returned pose is reported as [`IcpPlaneResult::information`].

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
/// largest diagonal of its block (rotation rows 0-2, translation rows 3-5) marks an unobserved
/// direction. The blocks are separate because rotation entries grow with depth squared while
/// translation entries are O(1) from unit normals.
const PIVOT_RTOL: f64 = 1e-8;

/// Fewest inliers that can constrain the six pose parameters.
const MIN_INLIERS: usize = 6;

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
    /// Depth noise model that sets each row's `sigma`. A frame's per-pixel sigma map
    /// ([`RgbdLevel::depth_sigmas`]) takes precedence over it point by point.
    pub depth_noise: DepthNoise,
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
            depth_noise: DepthNoise::Constant { sigma_m: 0.01 },
        }
    }
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
    /// weighting, not a sensor model. It can be singular when [`Self::observability`] is `0`.
    pub information: Mat6F64,
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
/// and information matrix of a final pass at that pose.
///
/// # Errors
///
/// * [`RgbdIcpError::PyramidLevelsMismatch`] if the pyramids have different level counts.
/// * [`RgbdIcpError::IterationsLevelsMismatch`] if `criteria.iters_per_level` does not have one
///   entry per level.
/// * [`RgbdIcpError::InvalidCriteria`] if a threshold of `criteria` or a parameter of its
///   `depth_noise` is not finite and positive, or `max_normal_angle_rad` exceeds pi.
/// * [`RgbdIcpError::TooFewCorrespondences`] if fewer than 6 correspondences pass the gates in
///   any pass, the final one included (low overlap or a bad initial guess).
/// * [`RgbdIcpError::SingularNormalEquations`] if the geometry does not constrain all six
///   degrees of freedom (for example a single plane), or the solve is not finite.
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
    let gates = check_criteria(criteria)?;
    run_icp(src, tgt, init_rotation, init_translation, criteria, &gates)
}

/// The gates of `criteria`, or the first threshold out of its range.
fn check_criteria(criteria: &IcpPlaneCriteria) -> Result<PassGates, RgbdIcpError> {
    let inf = f64::INFINITY;
    let noise = criteria.depth_noise.params().map(|(f, v)| (f, v, inf));
    for (field, value, max) in [
        ("max_dist_m", criteria.max_dist_m, inf),
        ("huber_delta", criteria.huber_delta, inf),
        ("update_tolerance", criteria.update_tolerance, inf),
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
    Ok(PassGates {
        max_dist_m: criteria.max_dist_m,
        cos_max_normal_angle: criteria.max_normal_angle_rad.cos(),
        huber_delta: criteria.huber_delta,
        depth_noise: criteria.depth_noise,
    })
}

/// Gates and weighting of one association pass.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PassGates {
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
        let Some(TargetSample {
            vertex: v_t,
            normal: n_t,
            depth_sigma: tgt_sigma,
        }) = sample_nearest(tgt, uv.x, uv.y)
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

/// Solves `A x = -b` by Cholesky, rejecting a collapsed pivot (see [`PIVOT_RTOL`]) or a
/// non-finite step.
fn solve_step(a: &Mat6F64, b: &Vec6F64) -> Result<Vec6F64, RgbdIcpError> {
    let chol = a.cholesky().ok_or(RgbdIcpError::SingularNormalEquations)?;
    // the pivots of the factorisation are the squared diagonal of L
    let l = chol.l_dirty();
    let block_ok = |rows: std::ops::Range<usize>| {
        let diag_finite = rows.clone().all(|i| a[(i, i)].is_finite());
        let max_diag = rows.clone().map(|i| a[(i, i)]).fold(0.0, f64::max);
        // `>=` so that a NaN pivot fails
        diag_finite
            && max_diag > 0.0
            && rows
                .into_iter()
                .all(|i| l[(i, i)] * l[(i, i)] >= PIVOT_RTOL * max_diag)
    };
    if !(block_ok(0..3) && block_ok(3..6)) {
        return Err(RgbdIcpError::SingularNormalEquations);
    }
    let x = chol.solve(&(-b));
    if !x.iter().all(|v| v.is_finite()) {
        return Err(RgbdIcpError::SingularNormalEquations);
    }
    Ok(x)
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

/// Coarse-to-fine Gauss-Newton from the initial pose; inputs already checked.
fn run_icp(
    src: &RgbdPyramid,
    tgt: &RgbdPyramid,
    init_rotation: &Mat3F64,
    init_translation: &Vec3F64,
    criteria: &IcpPlaneCriteria,
    gates: &PassGates,
) -> Result<IcpPlaneResult, RgbdIcpError> {
    let (mut rotation, mut translation) = (*init_rotation, *init_translation);
    let mut iterations = 0;

    // iters_per_level is coarsest first; levels[0] is the finest
    let levels = (0..src.levels.len()).rev();
    for (&iters, level) in criteria.iters_per_level.iter().zip(levels) {
        let (s, t) = (&src.levels[level], &tgt.levels[level]);
        for _ in 0..iters {
            let eqs = accumulate_level(s, t, &rotation, &translation, gates, None);
            check_inliers(&eqs)?;
            let x = solve_step(&eqs.a, &eqs.b)?;
            let (w, v) = (
                Vec3F64::new(x[0], x[1], x[2]),
                Vec3F64::new(x[3], x[4], x[5]),
            );
            let dr = SO3F64::exp(w).matrix();
            rotation = dr * rotation;
            translation = dr * translation + v;
            iterations += 1;

            // relative to the scene depth so that one tolerance fits any scale
            let mean_depth = eqs.sum_depth_m / eqs.num_inliers as f64;
            let tol = criteria.update_tolerance;
            if w.length() < tol && v.length() < tol * mean_depth {
                break;
            }
        }
    }

    // the reported figures come from the returned pose, not the last pre-update pass; the one
    // allocation of a call, sized by the image so that it never regrows
    let finest = &src.levels[0];
    let mut depths = Vec::with_capacity(finest.size().width * finest.size().height);
    let eqs = accumulate_level(
        finest,
        &tgt.levels[0],
        &rotation,
        &translation,
        gates,
        Some(&mut depths),
    );
    check_inliers(&eqs)?;
    let (rotation_conditioning, translation_conditioning) = block_conditioning(&eqs.a);
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
        median_depth_m: median_depth_m(&mut depths),
        information: eqs.a,
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

    /// One fronto-parallel plane leaves in-plane translation and rotation unobserved.
    #[test]
    fn test_single_plane_is_singular() -> Result<(), Box<dyn std::error::Error>> {
        let scene = Scene {
            planes: vec![Plane {
                normal: Vec3F64::new(0.0, 0.0, 1.0),
                d: 2.0,
            }],
            spheres: vec![],
        };
        let pyr = pyramid(&scene, &Mat3F64::IDENTITY, &Vec3F64::ZERO)?;
        let result = icp_from_identity(&pyr, &pyr, &IcpPlaneCriteria::default());
        assert!(
            matches!(result, Err(RgbdIcpError::SingularNormalEquations)),
            "{result:?}"
        );
        Ok(())
    }

    #[test]
    fn solve_step_rejects_nan_and_collapsed_pivots() {
        let b = Vec6F64::from_element(1.0);
        let x = solve_step(&Mat6F64::identity(), &b);
        assert!(matches!(x, Ok(x) if x == -b));
        let mut nan = Mat6F64::identity();
        nan[(4, 4)] = f64::NAN;
        assert!(solve_step(&nan, &b).is_err());
        // positive definite, but one translation pivot is 1e-10 of its block
        let mut weak = Mat6F64::identity();
        weak[(5, 5)] = 1e-10;
        assert!(solve_step(&weak, &b).is_err());
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

    /// The reported information must be the Hessian of the cost the solver minimises.
    ///
    /// Fixes the final pass's correspondences and their weights, perturbs the estimate in the
    /// solver's left tangent and takes central second differences of `0.5 * sum w r^2`. At a
    /// converged, noise-free solve the residuals are near zero, so the Gauss-Newton `J^T W J`
    /// and the true Hessian agree; a wrong Jacobian, sign, axis order or weight would not.
    #[test]
    fn information_matches_finite_difference_hessian() -> Result<(), Box<dyn std::error::Error>> {
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
            ..Default::default()
        };
        let res = icp_from_identity(&src, &tgt, &criteria)?;
        let info = &res.information;
        assert!((info - info.transpose()).amax() == 0.0);

        let gates = check_criteria(&criteria)?;
        let (rot, t) = (res.rotation, res.translation);
        let rows = gated_rows(&src.levels[0], &tgt.levels[0], &rot, &t, &gates);
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
        // nearest-pixel association is piecewise constant, so finite differences that
        // re-associate at each perturbed pose would only see snapping jumps: none here
        assert!(
            worst < 1e-3,
            "information disagrees with the fixed-target Hessian: {worst:.3e} over {} rows",
            rows.len()
        );
        Ok(())
    }
}
