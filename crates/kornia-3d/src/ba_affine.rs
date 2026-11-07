//! Simultaneous affine bundle adjustment using Cauchy IRLS and Schur LM.
//!
//! Uses the same block elimination as [`crate::ba_schur`], with eight additive camera
//! parameters instead of six SE(3) tangents. All assembly and solves are f64.
//! Point blocks use nalgebra Cholesky; the small reduced camera system is dense.
//! Four noncoplanar fixed points define the affine gauge; when any point is free, each
//! of those four must be seen in at least two views. Inputs must form one connected
//! camera/point graph. The solver normalizes world coordinates internally.
//! This solver has a separate API: no pinhole priors, shared
//! intrinsics, automatic gauge selection, or mixed camera types are implied.
//!
//! The Cauchy objective follows `kornia_algebra::optim::losses::CauchyLoss`,
//! evaluated here in f64 (that trait accepts f32). nalgebra provides the SVD
//! gauge check and Cholesky solves. The dense reduced system uses O(C²) storage
//! and O(C³) factorization work, so this API targets few-camera, many-point scenes.
//!
//! References: Triggs et al., "Bundle Adjustment: A Modern Synthesis" (2000), for the
//! Schur elimination and robust IRLS weights; Nielsen, "Damping Parameter in Marquardt's
//! Method" (IMM-REP-1999-05), for the gain-ratio damping update; Hartley and Zisserman,
//! "Multiple View Geometry", section 6.3, for the affine camera.
use crate::camera::AffineCamera;
use kornia_algebra::{Vec2F64, Vec3F64};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};

const CAMERA_DIM: usize = 8;
const POINT_DIM: usize = 3;
const DIAGONAL_FLOOR: f64 = 1e-12;
const GAUGE_SINGULAR_RATIO: f64 = 1e-6;
/// Scaled-gradient stop as a fraction of the largest pixel magnitude: about 4500 ulp of
/// that magnitude, just above where residual rounding dominates the gradient.
const GRADIENT_TOLERANCE: f64 = 1e-12;
const MIN_DAMPING: f64 = 1e-15;
/// Above this the damping dominates the Hessian diagonal, so a small cost decrease only
/// reflects a short step and says nothing about stationarity.
const MAX_CONVERGENCE_DAMPING: f64 = 1.;
/// Fraction of the cost below which a predicted decrease cannot be resolved by evaluating it.
const COST_RESOLUTION: f64 = f64::EPSILON;
/// Views an anchor needs to pin its 3D position when free points can absorb a gauge change.
const MIN_ANCHOR_VIEWS: usize = 2;
type CameraBlock = SMatrix<f64, CAMERA_DIM, CAMERA_DIM>;
type CrossBlock = SMatrix<f64, CAMERA_DIM, POINT_DIM>;
type PointBlock = SMatrix<f64, POINT_DIM, POINT_DIM>;
type CameraVector = SVector<f64, CAMERA_DIM>;
type PointVector = SVector<f64, POINT_DIM>;

/// A pixel measurement associated with an affine camera and world point.
#[derive(Clone, Copy, Debug)]
pub struct AffineObservation {
    /// Index into the cameras slice.
    pub camera_index: usize,
    /// Index into the points slice.
    pub point_index: usize,
    /// Measured image coordinates in pixels.
    pub pixel: Vec2F64,
}

/// Parameters for affine Schur Levenberg–Marquardt.
#[derive(Clone, Debug)]
pub struct AffineBaParams {
    /// Maximum number of linearizations (rejected damping trials do not count).
    pub max_iterations: usize,
    /// Cauchy scale applied to the norm of each 2D residual, in pixels.
    pub robust_scale: f64,
    /// Relative accepted cost decrease at which to stop, once damping is at most 1.
    pub cost_tolerance: f64,
    /// Initial multiplier for diagonal Hessian damping.
    pub initial_damping: f64,
    /// Maximum damping trials per linearization.
    pub max_trials: usize,
}
impl Default for AffineBaParams {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            robust_scale: 2.,
            cost_tolerance: 1e-7,
            initial_damping: 1e-3,
            max_trials: 30,
        }
    }
}

/// Result of an affine solve, in the original input coordinate frame.
#[derive(Debug)]
pub struct AffineBaResult {
    /// Optimized affine cameras.
    pub cameras: Vec<AffineCamera>,
    /// Optimized points; fixed points retain their exact input values.
    pub points: Vec<Vec3F64>,
    /// Initial `0.5 * scale² * sum(log1p(squared_residual / scale²))`.
    pub initial_cost: f64,
    /// Final cost using the same observations and convention as `initial_cost`.
    pub cost: f64,
    /// Number of linearizations performed.
    pub iterations: usize,
    /// Number of accepted updates.
    pub accepted_steps: usize,
    /// Whether the cost decrease, scaled gradient or numerical floor satisfied a stop test.
    pub converged: bool,
}

/// Input or numerical failure in affine BA.
#[derive(Debug, thiserror::Error)]
pub enum AffineBaError {
    /// Invalid sizes, indices, parameters, or nonfinite input.
    #[error("invalid affine BA input: {0}")]
    InvalidInput(String),
    /// Fixed points do not define a sufficiently noncoplanar affine frame.
    #[error(
        "affine BA requires four noncoplanar fixed points, each in two views if any point is free"
    )]
    InvalidGauge,
    /// No step was accepted before the damping trials ran out, away from a numerical floor.
    #[error("affine LM exhausted damping trials at iteration {iteration}")]
    NoProgress {
        /// One-based linearization index.
        iteration: usize,
    },
}

fn invalid(message: impl Into<String>) -> AffineBaError {
    AffineBaError::InvalidInput(message.into())
}

/// Centroid and root-mean-square distance from it.
fn centroid_and_rms(points: &[Vec3F64]) -> (Vec3F64, f64) {
    let center = points.iter().copied().fold(Vec3F64::ZERO, |a, b| a + b) / points.len() as f64;
    let squared = points
        .iter()
        .map(|x| (*x - center).dot(*x - center))
        .sum::<f64>();
    (center, (squared / points.len() as f64).sqrt())
}

/// Whether the fixed points define an affine frame, conditioned in their own coordinates so
/// free points cannot change the verdict.
///
/// With free points, an anchor seen once leaves its viewing ray open: a gauge change along
/// that ray keeps its projection while free points and other cameras absorb the rest. Only
/// anchors in at least [`MIN_ANCHOR_VIEWS`] views count then. Without free points there is
/// no gauge and each camera is fitted to fixed coordinates, so every anchor counts.
fn anchors_define_frame(
    points: &[Vec3F64],
    fixed_points: &[bool],
    by_point: &ObservationIndex,
) -> bool {
    let min_views = if fixed_points.iter().all(|f| *f) {
        1
    } else {
        MIN_ANCHOR_VIEWS
    };
    let anchors: Vec<_> = points
        .iter()
        .zip(fixed_points)
        .enumerate()
        .filter(|(p, (_, fixed))| **fixed && by_point.row(*p).len() >= min_views)
        .map(|(_, (x, _))| *x)
        .collect();
    if anchors.len() < 4 {
        return false;
    }
    let (center, scale) = centroid_and_rms(&anchors);
    if scale <= 0. {
        return false;
    }
    let frame = DMatrix::from_fn(anchors.len(), 4, |i, j| {
        if j == 3 {
            1.
        } else {
            ((anchors[i] - center) / scale).to_array()[j]
        }
    });
    let singular = frame.svd(false, false).singular_values;
    singular.min() >= GAUGE_SINGULAR_RATIO * singular.max()
}

/// Observation indices grouped by point or camera, in compressed sparse row layout. Each row
/// keeps observation order, so the Schur sums do not depend on how the index is stored.
struct ObservationIndex {
    offsets: Vec<usize>,
    entries: Vec<usize>,
}

impl ObservationIndex {
    /// Group `obs` into `rows` rows by `key`, which must be below `rows` for every observation.
    fn new(rows: usize, obs: &[AffineObservation], key: fn(&AffineObservation) -> usize) -> Self {
        let mut offsets = vec![0; rows + 1];
        for o in obs {
            offsets[key(o) + 1] += 1;
        }
        for r in 0..rows {
            offsets[r + 1] += offsets[r];
        }
        let mut next = offsets.clone();
        let mut entries = vec![0; obs.len()];
        for (i, o) in obs.iter().enumerate() {
            let slot = &mut next[key(o)];
            entries[*slot] = i;
            *slot += 1;
        }
        Self { offsets, entries }
    }

    fn row(&self, row: usize) -> &[usize] {
        &self.entries[self.offsets[row]..self.offsets[row + 1]]
    }
}

/// Check the inputs and return the observations grouped by point.
fn validate(
    cameras: &[AffineCamera],
    points: &[Vec3F64],
    obs: &[AffineObservation],
    fixed: &[bool],
    params: &AffineBaParams,
) -> Result<ObservationIndex, AffineBaError> {
    if cameras.is_empty() || points.is_empty() || obs.is_empty() || fixed.len() != points.len() {
        return Err(invalid(
            "nonempty cameras, points, observations and one fixed flag per point required",
        ));
    }
    if params.max_iterations == 0
        || params.max_trials == 0
        || !params.robust_scale.is_finite()
        || params.robust_scale <= 0.
        || !params.robust_scale.powi(2).is_finite()
        || params.robust_scale.powi(2) == 0.
        || !params.cost_tolerance.is_finite()
        || params.cost_tolerance <= 0.
        || !params.initial_damping.is_finite()
        || params.initial_damping <= 0.
    {
        return Err(invalid(
            "iteration limits and finite scales/tolerances must be positive",
        ));
    }
    if cameras
        .iter()
        .any(|c| c.matrix.iter().flatten().any(|v| !v.is_finite()))
        || points
            .iter()
            .any(|x| x.to_array().iter().any(|v| !v.is_finite()))
    {
        return Err(invalid("camera and point coordinates must be finite"));
    }
    for (i, o) in obs.iter().enumerate() {
        if o.camera_index >= cameras.len() {
            return Err(invalid(format!(
                "observation {i}: camera index {} out of bounds for {} cameras",
                o.camera_index,
                cameras.len()
            )));
        }
        if o.point_index >= points.len() {
            return Err(invalid(format!(
                "observation {i}: point index {} out of bounds for {} points",
                o.point_index,
                points.len()
            )));
        }
        if o.pixel.to_array().iter().any(|v| !v.is_finite()) {
            return Err(invalid(format!("observation {i}: pixel must be finite")));
        }
    }
    let by_point = ObservationIndex::new(points.len(), obs, |o| o.point_index);
    let by_camera = ObservationIndex::new(cameras.len(), obs, |o| o.camera_index);
    // Each camera remembers the last point it saw. Rows are in observation order, so the
    // smallest repeat is the first observation that duplicates an earlier one.
    let mut last_point = vec![points.len(); cameras.len()];
    let mut first_duplicate = None;
    for p in 0..points.len() {
        for &i in by_point.row(p) {
            let c = obs[i].camera_index;
            if last_point[c] == p && first_duplicate.is_none_or(|first| i < first) {
                first_duplicate = Some(i);
            }
            last_point[c] = p;
        }
    }
    if let Some(i) = first_duplicate {
        return Err(invalid(format!(
            "observation {i}: duplicate of camera {} and point {}",
            obs[i].camera_index, obs[i].point_index
        )));
    }
    for (p, is_fixed) in fixed.iter().enumerate() {
        let views = by_point.row(p).len();
        if views == 0 {
            return Err(invalid(format!("point {p} has no observations")));
        }
        if !is_fixed && views < 2 {
            return Err(invalid(format!(
                "free point {p} needs two views, has {views}"
            )));
        }
    }
    for c in 0..cameras.len() {
        let observed = by_camera.row(c).len();
        if observed < 4 {
            return Err(invalid(format!(
                "camera {c} needs four observed points, has {observed}"
            )));
        }
    }
    let mut seen = vec![false; cameras.len()];
    let mut todo = vec![0];
    seen[0] = true;
    while let Some(c) = todo.pop() {
        for &a in by_camera.row(c) {
            for &b in by_point.row(obs[a].point_index) {
                let v = obs[b].camera_index;
                if !seen[v] {
                    seen[v] = true;
                    todo.push(v);
                }
            }
        }
    }
    if seen.iter().any(|s| !s) {
        return Err(invalid("observation graph is disconnected"));
    }
    Ok(by_point)
}

/// Cauchy `0.5 * rho` of a squared residual norm.
fn cauchy_cost(squared_norm: f64, scale2: f64) -> f64 {
    0.5 * scale2 * (squared_norm / scale2).ln_1p()
}

/// IRLS weight: twice the derivative of [`cauchy_cost`] with respect to the squared norm.
fn cauchy_weight(squared_norm: f64, scale2: f64) -> f64 {
    1. / (1. + squared_norm / scale2)
}

fn cost(
    cameras: &[AffineCamera],
    points: &[Vec3F64],
    obs: &[AffineObservation],
    scale: f64,
) -> f64 {
    let scale2 = scale * scale;
    obs.iter()
        .map(|o| {
            let r = cameras[o.camera_index].project(&points[o.point_index]) - o.pixel;
            cauchy_cost(r.dot(r), scale2)
        })
        .sum()
}

struct NormalEquations {
    cc: Vec<CameraBlock>,
    cp: Vec<CrossBlock>,
    pp: Vec<PointBlock>,
    gc: Vec<CameraVector>,
    gp: Vec<PointVector>,
}
impl NormalEquations {
    fn new(nc: usize, np: usize, no: usize) -> Self {
        Self {
            cc: vec![CameraBlock::zeros(); nc],
            cp: vec![CrossBlock::zeros(); no],
            pp: vec![PointBlock::zeros(); np],
            gc: vec![CameraVector::zeros(); nc],
            gp: vec![PointVector::zeros(); np],
        }
    }
    fn assemble(
        &mut self,
        cameras: &[AffineCamera],
        points: &[Vec3F64],
        obs: &[AffineObservation],
        fixed: &[bool],
        scale: f64,
    ) {
        self.cc.fill(CameraBlock::zeros());
        self.pp.fill(PointBlock::zeros());
        self.gc.fill(CameraVector::zeros());
        self.gp.fill(PointVector::zeros());
        for (i, o) in obs.iter().enumerate() {
            let camera = &cameras[o.camera_index];
            let x = &points[o.point_index];
            let r = camera.project(x) - o.pixel;
            let w = cauchy_weight(r.dot(r), scale * scale);
            let (jc, jp) = camera.jacobians(x);
            let jc = SMatrix::<f64, 2, 8>::from_fn(|i, j| jc[i][j]);
            let jp = SMatrix::<f64, 2, 3>::from_fn(|i, j| jp[i][j]);
            let r = SVector::<f64, 2>::from_row_slice(&r.to_array());
            self.cc[o.camera_index] += jc.transpose() * jc * w;
            self.gc[o.camera_index] += jc.transpose() * r * w;
            if !fixed[o.point_index] {
                self.pp[o.point_index] += jp.transpose() * jp * w;
                self.gp[o.point_index] += jp.transpose() * r * w;
                self.cp[i] = jc.transpose() * jp * w;
            }
        }
    }

    /// Largest `|g_k| / sqrt(H_kk)`, in pixels. Rescaling a parameter leaves it unchanged, so
    /// unlike the raw gradient it does not grow with camera or world magnitude.
    fn scaled_gradient_max(&self) -> f64 {
        fn scaled(g: f64, h: f64) -> f64 {
            if h > 0. {
                g.abs() / h.sqrt()
            } else {
                0.
            }
        }
        let cameras = self.gc.iter().zip(&self.cc).flat_map(|(g, h)| {
            g.iter()
                .enumerate()
                .map(move |(k, g)| scaled(*g, h[(k, k)]))
        });
        let points = self.gp.iter().zip(&self.pp).flat_map(|(g, h)| {
            g.iter()
                .enumerate()
                .map(move |(k, g)| scaled(*g, h[(k, k)]))
        });
        cameras.chain(points).fold(0., f64::max)
    }

    /// Decrease of the damped quadratic model for a step: `0.5 * d' (damping * D * d - g)`.
    fn predicted_decrease(
        &self,
        camera_steps: &[CameraVector],
        point_steps: &[PointVector],
        fixed: &[bool],
        damping: f64,
    ) -> f64 {
        let mut twice = 0.;
        for (c, step) in camera_steps.iter().enumerate() {
            for (k, delta) in step.iter().enumerate() {
                twice += delta
                    * (damping * self.cc[c][(k, k)].max(DIAGONAL_FLOOR) * delta - self.gc[c][k]);
            }
        }
        for (p, step) in point_steps.iter().enumerate() {
            if !fixed[p] {
                for (k, delta) in step.iter().enumerate() {
                    twice += delta
                        * (damping * self.pp[p][(k, k)].max(DIAGONAL_FLOOR) * delta
                            - self.gp[p][k]);
                }
            }
        }
        0.5 * twice
    }
}

/// Optimize affine cameras and 3D points together with Schur LM.
///
/// Fixing four noncoplanar points selects a representative of the 12-dimensional
/// affine gauge. When any point is free, each of the four must be seen in at least two
/// views; those views also need non-parallel viewing directions, which cannot be checked
/// before the cameras are solved. Fixing additional points imposes additional geometric
/// constraints. Normalization and damping do not substitute for these anchors. Degenerate
/// local geometry is damped; no observations or points are silently discarded.
///
/// # Arguments
/// * `cameras` - Initial world-to-pixel affine matrices, all adjustable.
/// * `points` - Initial finite world coordinates.
/// * `observations` - Unique camera/point pairs in one connected component.
/// * `fixed_points` - One flag per point; true points are held exactly.
/// * `params` - Cauchy scale and LM stopping parameters.
/// # Returns
/// Optimized state and convergence statistics. Stops with `converged = true` when the
/// scaled gradient is negligible, the cost decrease falls below `cost_tolerance` at damping
/// of at most 1, or no step can resolve a further decrease. Exhausting the iteration cap,
/// or the damping trials after some progress, returns the last accepted state with
/// `converged = false`.
/// # Errors
/// Returns [`AffineBaError`] for invalid inputs, a degenerate fixed-point frame, or
/// damping trials exhausted before any accepted step. Planar anchor sets are rejected.
/// # Example
/// ```
/// use kornia_3d::{ba_affine::*, camera::AffineCamera};
/// use kornia_algebra::Vec3F64;
/// let points = vec![Vec3F64::ZERO, Vec3F64::new(1.,0.,0.), Vec3F64::new(0.,1.,0.), Vec3F64::new(0.,0.,1.)];
/// let cameras = vec![AffineCamera { matrix: [[1.,0.,0.,0.], [0.,1.,0.,0.]] }];
/// let observations: Vec<_> = points.iter().enumerate().map(|(p,x)| AffineObservation {
///     camera_index: 0, point_index: p, pixel: cameras[0].project(x)
/// }).collect();
/// let result = bundle_adjust_affine(&cameras, &points, &observations, &[true;4], &AffineBaParams::default())?;
/// assert!(result.cost < 1e-20);
/// # Ok::<(), AffineBaError>(())
/// ```
pub fn bundle_adjust_affine(
    cameras: &[AffineCamera],
    points: &[Vec3F64],
    observations: &[AffineObservation],
    fixed_points: &[bool],
    params: &AffineBaParams,
) -> Result<AffineBaResult, AffineBaError> {
    let by_point = validate(cameras, points, observations, fixed_points, params)?;
    let (center, scale) = centroid_and_rms(points);
    if !scale.is_finite() {
        return Err(invalid("world coordinates overflow normalization"));
    }
    if !anchors_define_frame(points, fixed_points, &by_point) {
        return Err(AffineBaError::InvalidGauge);
    }
    let mut x: Vec<_> = points.iter().map(|x| (*x - center) / scale).collect();
    let mut cams = cameras.to_vec();
    for c in &mut cams {
        let t = c.project(&center);
        for (i, row) in c.matrix.iter_mut().enumerate() {
            for value in &mut row[..3] {
                *value *= scale;
            }
            row[3] = t.to_array()[i];
        }
    }
    let nc = cams.len();
    let np = x.len();
    let mut normal = NormalEquations::new(nc, np, observations.len());
    let mut inv = vec![PointBlock::zeros(); np];
    let mut dc = vec![CameraVector::zeros(); nc];
    let mut dp = vec![PointVector::zeros(); np];
    let mut trial_cams = cams.clone();
    let mut trial_x = x.clone();
    let mut reduced = DMatrix::<f64>::zeros(nc * CAMERA_DIM, nc * CAMERA_DIM);
    let mut rhs = DVector::<f64>::zeros(nc * CAMERA_DIM);
    let initial_cost = cost(&cams, &x, observations, params.robust_scale);
    if !initial_cost.is_finite() {
        return Err(invalid("initial cost overflow"));
    }
    let mut current_cost = initial_cost;
    let mut damping = params.initial_damping;
    let mut nu = 2.;
    let mut accepted_steps = 0;
    let mut iterations = 0;
    let mut converged = false;
    let gradient_tolerance = GRADIENT_TOLERANCE
        * observations
            .iter()
            .flat_map(|o| o.pixel.to_array())
            .fold(1., |largest: f64, v| largest.max(v.abs()));
    for iteration in 1..=params.max_iterations {
        iterations = iteration;
        normal.assemble(&cams, &x, observations, fixed_points, params.robust_scale);
        if normal.scaled_gradient_max() < gradient_tolerance {
            converged = true;
            break;
        }
        let mut accepted = false;
        let mut smallest_predicted = f64::INFINITY;
        for _ in 0..params.max_trials {
            reduced.fill(0.);
            rhs.fill(0.);
            for c in 0..nc {
                for i in 0..CAMERA_DIM {
                    rhs[c * CAMERA_DIM + i] = -normal.gc[c][i];
                    for j in 0..CAMERA_DIM {
                        reduced[(c * CAMERA_DIM + i, c * CAMERA_DIM + j)] = normal.cc[c][(i, j)];
                    }
                    reduced[(c * CAMERA_DIM + i, c * CAMERA_DIM + i)] +=
                        damping * normal.cc[c][(i, i)].max(DIAGONAL_FLOOR);
                }
            }
            let mut valid = true;
            for p in 0..np {
                if fixed_points[p] {
                    continue;
                }
                let mut h = normal.pp[p];
                for k in 0..POINT_DIM {
                    h[(k, k)] += damping * h[(k, k)].max(DIAGONAL_FLOOR);
                }
                let Some(chol) = h.cholesky() else {
                    valid = false;
                    break;
                };
                inv[p] = chol.solve(&PointBlock::identity());
                for &a in by_point.row(p) {
                    let ca = observations[a].camera_index;
                    let t = normal.cp[a] * inv[p];
                    let b = t * normal.gp[p];
                    for i in 0..CAMERA_DIM {
                        rhs[ca * CAMERA_DIM + i] += b[i];
                    }
                    for &b in by_point.row(p) {
                        let cb = observations[b].camera_index;
                        let block = t * normal.cp[b].transpose();
                        for i in 0..CAMERA_DIM {
                            for j in 0..CAMERA_DIM {
                                reduced[(ca * CAMERA_DIM + i, cb * CAMERA_DIM + j)] -=
                                    block[(i, j)];
                            }
                        }
                    }
                }
            }
            let solution = if valid {
                reduced.clone().cholesky().map(|c| c.solve(&rhs))
            } else {
                None
            };
            if let Some(solution) = solution {
                for c in 0..nc {
                    for k in 0..CAMERA_DIM {
                        dc[c][k] = solution[c * CAMERA_DIM + k];
                        trial_cams[c].matrix[k / 4][k % 4] =
                            cams[c].matrix[k / 4][k % 4] + dc[c][k];
                    }
                }
                for p in 0..np {
                    if fixed_points[p] {
                        continue;
                    }
                    let mut b = normal.gp[p];
                    for &o in by_point.row(p) {
                        b += normal.cp[o].transpose() * dc[observations[o].camera_index];
                    }
                    dp[p] = -inv[p] * b;
                    trial_x[p] = x[p] + Vec3F64::new(dp[p][0], dp[p][1], dp[p][2]);
                }
                let next_cost = cost(&trial_cams, &trial_x, observations, params.robust_scale);
                let predicted = normal.predicted_decrease(&dc, &dp, fixed_points, damping);
                smallest_predicted = smallest_predicted.min(predicted);
                if next_cost.is_finite() && next_cost < current_cost {
                    let decrease = current_cost - next_cost;
                    converged = decrease < params.cost_tolerance * current_cost
                        && damping <= MAX_CONVERGENCE_DAMPING;
                    let rho = decrease / predicted.max(f64::MIN_POSITIVE);
                    damping *= (1.0_f64 / 3.).max(1. - (2. * rho - 1.).powi(3));
                    damping = damping.max(MIN_DAMPING);
                    nu = 2.;
                    current_cost = next_cost;
                    std::mem::swap(&mut cams, &mut trial_cams);
                    std::mem::swap(&mut x, &mut trial_x);
                    accepted_steps += 1;
                    accepted = true;
                    break;
                }
            }
            damping *= nu;
            nu *= 2.;
            if !damping.is_finite() {
                break;
            }
        }
        if !accepted {
            // The gradient is analytic, so a small enough damped step descends. Rejecting every
            // step until the predicted decrease is unresolvable means the cost is numerically
            // stationary; residual rounding can hold the scaled gradient above its tolerance.
            if smallest_predicted <= COST_RESOLUTION * current_cost {
                converged = true;
            } else if accepted_steps == 0 {
                return Err(AffineBaError::NoProgress { iteration });
            }
            break;
        }
        if converged {
            break;
        }
    }
    for c in &mut cams {
        for row in &mut c.matrix {
            for v in &mut row[..3] {
                *v /= scale;
            }
            row[3] -= Vec3F64::new(row[0], row[1], row[2]).dot(center);
        }
    }
    for (p, point) in x.iter_mut().enumerate() {
        *point = if fixed_points[p] {
            points[p]
        } else {
            *point * scale + center
        };
    }
    Ok(AffineBaResult {
        cameras: cams,
        points: x,
        initial_cost,
        cost: current_cost,
        iterations,
        accepted_steps,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (
        Vec<AffineCamera>,
        Vec<Vec3F64>,
        Vec<AffineObservation>,
        Vec<bool>,
    ) {
        let points: Vec<_> = (0..60)
            .map(|i| {
                let t = i as f64;
                Vec3F64::new(
                    (t * 1.7).sin() * 4.,
                    (t * 0.9).cos() * 3.,
                    (t * 0.4).sin() * 2.,
                )
            })
            .collect();
        let cameras: Vec<_> = (0..4)
            .map(|i| {
                let a = i as f64 * 1.5;
                AffineCamera {
                    matrix: [
                        [a.cos() * 3., a.sin() * 3., 0., 100.],
                        [-a.sin() * 2., a.cos() * 2., -2., 200.],
                    ],
                }
            })
            .collect();
        let observations = cameras
            .iter()
            .enumerate()
            .flat_map(|(c, cam)| {
                points
                    .iter()
                    .enumerate()
                    .map(move |(p, x)| AffineObservation {
                        camera_index: c,
                        point_index: p,
                        pixel: cam.project(x),
                    })
            })
            .collect();
        let fixed = (0..points.len()).map(|p| p < 4).collect();
        (cameras, points, observations, fixed)
    }
    #[test]
    fn cauchy_cost_matches_numpy_reference() {
        // NumPy: 0.5 * 4 * np.log1p(np.array([25., 0.]) / 4).sum().
        let camera = AffineCamera {
            matrix: [[0.; 4]; 2],
        };
        let observations = [
            AffineObservation {
                camera_index: 0,
                point_index: 0,
                pixel: Vec2F64::new(3., 4.),
            },
            AffineObservation {
                camera_index: 0,
                point_index: 0,
                pixel: Vec2F64::new(0., 0.),
            },
        ];
        let actual = cost(&[camera], &[Vec3F64::ZERO], &observations, 2.);
        assert!((actual - 3.9620029377331666).abs() < 1e-14);
    }

    #[test]
    fn camera_only_matches_numpy_lstsq() -> Result<(), AffineBaError> {
        // np.linalg.lstsq(np.c_[points, np.ones(5)], pixels, rcond=None)[0].T
        // gives [[2, -1, 3, 7], [-1, 4, 2, -3]] for these literal observations.
        let points = vec![
            Vec3F64::new(0., 0., 0.),
            Vec3F64::new(1., 0., 0.),
            Vec3F64::new(0., 1., 0.),
            Vec3F64::new(0., 0., 1.),
            Vec3F64::new(2., -1., 3.),
        ];
        let pixels = [[7., -3.], [9., -4.], [6., 1.], [10., -1.], [21., -3.]];
        let obs: Vec<_> = pixels
            .iter()
            .enumerate()
            .map(|(point_index, &pixel)| AffineObservation {
                camera_index: 0,
                point_index,
                pixel: Vec2F64::from_array(pixel),
            })
            .collect();
        let cameras = [AffineCamera {
            matrix: [[1., 0., 1., 6.], [0., 3., 1., -2.]],
        }];
        let result = bundle_adjust_affine(
            &cameras,
            &points,
            &obs,
            &[true; 5],
            &AffineBaParams::default(),
        )?;
        for (actual, expected) in result.cameras[0]
            .matrix
            .iter()
            .flatten()
            .zip([2., -1., 3., 7., -1., 4., 2., -3.])
        {
            assert!((actual - expected).abs() < 1e-7);
        }
        assert_eq!(result.points, points);
        Ok(())
    }

    #[test]
    fn rejects_invalid_scale_and_visibility() {
        let (cams, points, obs, fixed) = fixture();
        for scale in [0., -1., f64::INFINITY, 1e200, 1e-200] {
            let params = AffineBaParams {
                robust_scale: scale,
                ..Default::default()
            };
            assert!(matches!(
                bundle_adjust_affine(&cams, &points, &obs, &fixed, &params),
                Err(AffineBaError::InvalidInput(_))
            ));
        }
        let sparse: Vec<_> = obs
            .iter()
            .copied()
            .filter(|o| o.point_index != 5 || o.camera_index == 0)
            .collect();
        assert!(matches!(
            bundle_adjust_affine(&cams, &points, &sparse, &fixed, &AffineBaParams::default()),
            Err(AffineBaError::InvalidInput(_))
        ));
        let disconnected: Vec<_> = obs
            .iter()
            .copied()
            .filter(|o| (o.camera_index < 2) == (o.point_index < 30))
            .collect();
        assert!(matches!(
            bundle_adjust_affine(
                &cams,
                &points,
                &disconnected,
                &fixed,
                &AffineBaParams::default()
            ),
            Err(AffineBaError::InvalidInput(_))
        ));
    }

    #[test]
    fn recovers_cameras_and_points() -> Result<(), AffineBaError> {
        let (cameras, points, observations, fixed) = fixture();
        let mut initial_cameras = cameras.clone();
        for c in &mut initial_cameras {
            c.matrix[0][2] += 0.03;
            c.matrix[1][3] += 0.2;
        }
        let mut initial_points = points.clone();
        for (p, x) in initial_points.iter_mut().enumerate().skip(4) {
            x.x += 0.1 * (p as f64).sin();
            x.z -= 0.05;
        }
        let result = bundle_adjust_affine(
            &initial_cameras,
            &initial_points,
            &observations,
            &fixed,
            &AffineBaParams::default(),
        )?;
        assert!(result.cost < 1e-12, "{}", result.cost);
        assert!(result.accepted_steps < 30);
        for (p, (a, b)) in result.points.iter().zip(&points).enumerate() {
            assert!((*a - *b).length() < 1e-5);
            if fixed[p] {
                assert_eq!(a, b);
            }
        }
        Ok(())
    }
    #[test]
    fn schur_step_matches_full_damped_system() -> Result<(), AffineBaError> {
        let (mut cams, mut points, obs, fixed) = fixture();
        cams[0].matrix[0][3] += 0.3;
        points[10].x += 0.2;
        let (center, scale) = centroid_and_rms(&points);
        let x: Vec<_> = points.iter().map(|p| (*p - center) / scale).collect();
        let mut normalized = cams.clone();
        for c in &mut normalized {
            let t = c.project(&center);
            for (r, row) in c.matrix.iter_mut().enumerate() {
                for a in &mut row[..3] {
                    *a *= scale;
                }
                row[3] = t.to_array()[r];
            }
        }
        let mut normal = NormalEquations::new(cams.len(), points.len(), obs.len());
        normal.assemble(&normalized, &x, &obs, &fixed, 2.);
        let cam_dim = cams.len() * CAMERA_DIM;
        let n = cam_dim + points.len() * POINT_DIM;
        let mut h = DMatrix::<f64>::zeros(n, n);
        let mut g = DVector::<f64>::zeros(n);
        for c in 0..cams.len() {
            for i in 0..CAMERA_DIM {
                g[c * CAMERA_DIM + i] = normal.gc[c][i];
                for j in 0..CAMERA_DIM {
                    h[(c * CAMERA_DIM + i, c * CAMERA_DIM + j)] = normal.cc[c][(i, j)];
                }
            }
        }
        for p in 0..points.len() {
            for i in 0..POINT_DIM {
                let row = cam_dim + p * POINT_DIM + i;
                if fixed[p] {
                    h[(row, row)] = 1.;
                    continue;
                }
                g[row] = normal.gp[p][i];
                for j in 0..POINT_DIM {
                    h[(row, cam_dim + p * POINT_DIM + j)] = normal.pp[p][(i, j)];
                }
            }
        }
        for (o, b) in obs.iter().zip(&normal.cp) {
            if !fixed[o.point_index] {
                for i in 0..CAMERA_DIM {
                    for j in 0..POINT_DIM {
                        let row = o.camera_index * CAMERA_DIM + i;
                        let col = cam_dim + o.point_index * POINT_DIM + j;
                        h[(row, col)] += b[(i, j)];
                        h[(col, row)] += b[(i, j)];
                    }
                }
            }
        }
        for k in 0..n {
            h[(k, k)] += 1e-3 * h[(k, k)].max(DIAGONAL_FLOOR);
        }
        let step = h.cholesky().unwrap().solve(&(-g));
        let params = AffineBaParams {
            max_iterations: 1,
            ..Default::default()
        };
        let result = bundle_adjust_affine(&cams, &points, &obs, &fixed, &params)?;
        assert_eq!(result.accepted_steps, 1);
        for (p, point) in result.points.iter().enumerate() {
            for k in 0..POINT_DIM {
                let expected = (x[p].to_array()[k] + step[cam_dim + p * POINT_DIM + k]) * scale
                    + center.to_array()[k];
                assert!((point.to_array()[k] - expected).abs() < 1e-10);
            }
        }
        for (c, camera) in result.cameras.iter().enumerate() {
            for r in 0..2 {
                let mut row = normalized[c].matrix[r];
                for k in 0..4 {
                    row[k] += step[c * CAMERA_DIM + r * 4 + k];
                }
                for value in &mut row[..3] {
                    *value /= scale;
                }
                row[3] -= Vec3F64::new(row[0], row[1], row[2]).dot(center);
                for (k, expected) in row.iter().enumerate() {
                    assert!((camera.matrix[r][k] - expected).abs() < 1e-10);
                }
            }
        }
        Ok(())
    }
    #[test]
    fn rejects_planar_gauge_and_bad_inputs() {
        let (cams, mut points, obs, mut fixed) = fixture();
        let params = AffineBaParams::default();
        fixed.fill(false);
        assert!(matches!(
            bundle_adjust_affine(&cams, &points, &obs, &fixed, &params),
            Err(AffineBaError::InvalidGauge)
        ));
        fixed[..4].fill(true);
        for p in &mut points[..4] {
            p.z = 0.;
        }
        assert!(matches!(
            bundle_adjust_affine(&cams, &points, &obs, &fixed, &params),
            Err(AffineBaError::InvalidGauge)
        ));
        points[5].x = f64::NAN;
        assert!(matches!(
            bundle_adjust_affine(&cams, &points, &obs, &fixed, &params),
            Err(AffineBaError::InvalidInput(_))
        ));
    }
    #[test]
    fn rejects_duplicate_observations() {
        let (cams, points, mut obs, fixed) = fixture();
        obs.push(obs[0]);
        assert!(matches!(
            bundle_adjust_affine(&cams, &points, &obs, &fixed, &AffineBaParams::default()),
            Err(AffineBaError::InvalidInput(_))
        ));
    }
    #[test]
    fn robust_cost_decreases_with_outliers() -> Result<(), AffineBaError> {
        let (cams, points, mut obs, fixed) = fixture();
        for o in obs.iter_mut().filter(|o| o.point_index >= 4).step_by(13) {
            o.pixel.x += 25.;
            o.pixel.y -= 15.;
        }
        let result =
            bundle_adjust_affine(&cams, &points, &obs, &fixed, &AffineBaParams::default())?;
        assert!(result.cost < result.initial_cost);
        let mut errors: Vec<_> = result
            .points
            .iter()
            .zip(&points)
            .map(|(a, b)| (*a - *b).length())
            .collect();
        errors.sort_by(f64::total_cmp);
        assert!(
            errors[errors.len() / 2] < 0.1,
            "median point error: {}",
            errors[errors.len() / 2]
        );
        Ok(())
    }

    struct Scene {
        cameras: Vec<AffineCamera>,
        truth: Vec<Vec3F64>,
        start: Vec<Vec3F64>,
        observations: Vec<AffineObservation>,
        fixed: Vec<bool>,
    }

    /// Deterministic value in `[-1, 1)` (Knuth's MMIX LCG), so scenes need no RNG dependency.
    fn uniform(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64 / (1_u64 << 53) as f64) * 2. - 1.
    }

    /// Four cameras of coefficient size `focal` and translation `offset` viewing `count`
    /// points: a tetrahedron of fixed anchors, then free points started up to 0.05 off.
    fn scene(count: usize, noise: f64, focal: f64, offset: f64) -> Scene {
        let mut state = 0x1234_5678_u64;
        let mut truth: Vec<_> = (0..count)
            .map(|_| {
                Vec3F64::new(
                    uniform(&mut state) * 2.,
                    uniform(&mut state) * 2.,
                    uniform(&mut state) * 2.,
                )
            })
            .collect();
        truth[0] = Vec3F64::new(0., 0., 0.);
        truth[1] = Vec3F64::new(2., 0., 0.);
        truth[2] = Vec3F64::new(0., 2., 0.);
        truth[3] = Vec3F64::new(0., 0., 2.);
        let cameras: Vec<_> = (0..4)
            .map(|i| {
                let (s, c) = (i as f64 * 1.5).sin_cos();
                AffineCamera {
                    matrix: [
                        [c * focal, s * focal, 0.1 * focal, offset],
                        [-s * focal, c * focal, -0.7 * focal, offset],
                    ],
                }
            })
            .collect();
        let mut observations = Vec::new();
        for (camera_index, camera) in cameras.iter().enumerate() {
            for (point_index, x) in truth.iter().enumerate() {
                let mut pixel = camera.project(x);
                pixel.x += noise * uniform(&mut state);
                pixel.y += noise * uniform(&mut state);
                observations.push(AffineObservation {
                    camera_index,
                    point_index,
                    pixel,
                });
            }
        }
        let mut start = truth.clone();
        for x in start.iter_mut().skip(4) {
            x.x += 0.05 * uniform(&mut state);
            x.y += 0.05 * uniform(&mut state);
            x.z += 0.05 * uniform(&mut state);
        }
        let fixed = (0..count).map(|p| p < 4).collect();
        Scene {
            cameras,
            truth,
            start,
            observations,
            fixed,
        }
    }

    fn median_error(a: &[Vec3F64], b: &[Vec3F64]) -> f64 {
        let mut errors: Vec<_> = a.iter().zip(b).map(|(a, b)| (*a - *b).length()).collect();
        errors.sort_by(f64::total_cmp);
        errors[errors.len() / 2]
    }

    #[test]
    fn solved_scene_at_realistic_scale_converges() -> Result<(), AffineBaError> {
        for (noise, focal, offset) in [
            (0., 500., 1e3),
            (0., 5000., 1e3),
            (0., 5000., 1e5),
            (0., 50000., 1e6),
            (1e-9, 5000., 1e3),
        ] {
            let s = scene(200, noise, focal, offset);
            let result = bundle_adjust_affine(
                &s.cameras,
                &s.start,
                &s.observations,
                &s.fixed,
                &AffineBaParams::default(),
            )?;
            assert!(result.converged, "noise {noise} focal {focal}");
            assert!(median_error(&result.points, &s.truth) < 1e-9);
        }
        Ok(())
    }

    #[test]
    fn exact_start_is_not_an_error() -> Result<(), AffineBaError> {
        let s = scene(200, 0., 5000., 1e5);
        let result = bundle_adjust_affine(
            &s.cameras,
            &s.truth,
            &s.observations,
            &s.fixed,
            &AffineBaParams::default(),
        )?;
        assert!(result.converged);
        assert!(result.cost <= result.initial_cost);
        Ok(())
    }

    #[test]
    fn unreachable_cost_tolerance_keeps_the_solution() -> Result<(), AffineBaError> {
        let s = scene(200, 0.5, 3., 100.);
        let solve = |cost_tolerance| {
            bundle_adjust_affine(
                &s.cameras,
                &s.start,
                &s.observations,
                &s.fixed,
                &AffineBaParams {
                    cost_tolerance,
                    ..Default::default()
                },
            )
        };
        let reference = solve(1e-12)?;
        let result = solve(1e-30)?;
        assert!(result.converged);
        assert!((result.cost - reference.cost).abs() < 1e-9 * reference.cost);
        Ok(())
    }

    #[test]
    fn large_initial_damping_is_not_reported_as_convergence() -> Result<(), AffineBaError> {
        let s = scene(200, 0.5, 3., 100.);
        let solve = |initial_damping| {
            bundle_adjust_affine(
                &s.cameras,
                &s.start,
                &s.observations,
                &s.fixed,
                &AffineBaParams {
                    initial_damping,
                    ..Default::default()
                },
            )
        };
        let reference = solve(1e-3)?;
        for damping in [1e7, 1e10] {
            let result = solve(damping)?;
            assert!(result.converged, "damping {damping}");
            assert!(
                (result.cost - reference.cost).abs() < 1e-5 * reference.cost,
                "damping {damping}: {} vs {}",
                result.cost,
                reference.cost
            );
        }
        Ok(())
    }

    #[test]
    fn anchors_in_a_single_view_are_rejected() {
        let s = scene(60, 0., 3., 100.);
        let observations: Vec<_> = s
            .observations
            .iter()
            .copied()
            .filter(|o| o.point_index >= 4 || o.camera_index == 0)
            .collect();
        assert!(matches!(
            bundle_adjust_affine(
                &s.cameras,
                &s.start,
                &observations,
                &s.fixed,
                &AffineBaParams::default()
            ),
            Err(AffineBaError::InvalidGauge)
        ));
    }

    #[test]
    fn anchors_in_two_views_fix_the_gauge() -> Result<(), AffineBaError> {
        let s = scene(60, 0., 3., 100.);
        let observations: Vec<_> = s
            .observations
            .iter()
            .copied()
            .filter(|o| o.point_index >= 4 || o.camera_index < 2)
            .collect();
        let mut other_start = s.start.clone();
        for x in other_start.iter_mut().skip(4) {
            x.z += 0.3;
        }
        for start in [&s.start, &other_start] {
            let result = bundle_adjust_affine(
                &s.cameras,
                start,
                &observations,
                &s.fixed,
                &AffineBaParams::default(),
            )?;
            assert!(median_error(&result.points, &s.truth) < 1e-7);
        }
        Ok(())
    }

    #[test]
    fn far_free_point_does_not_invalidate_good_anchors() -> Result<(), AffineBaError> {
        let mut s = scene(60, 0., 3., 100.);
        s.start[10] = Vec3F64::new(1e8, 0., 0.);
        let result = bundle_adjust_affine(
            &s.cameras,
            &s.start,
            &s.observations,
            &s.fixed,
            &AffineBaParams::default(),
        )?;
        assert!(result.cost < result.initial_cost);
        Ok(())
    }

    #[test]
    fn uniformly_tiny_world_is_normalized() -> Result<(), AffineBaError> {
        let s = scene(60, 0., 3., 100.);
        for world_scale in [1e-16, 1e-20] {
            let points: Vec<_> = s.start.iter().map(|x| *x * world_scale).collect();
            let cameras: Vec<_> = s
                .cameras
                .iter()
                .map(|c| {
                    let mut matrix = c.matrix;
                    for row in &mut matrix {
                        for value in &mut row[..3] {
                            *value /= world_scale;
                        }
                    }
                    AffineCamera { matrix }
                })
                .collect();
            let result = bundle_adjust_affine(
                &cameras,
                &points,
                &s.observations,
                &s.fixed,
                &AffineBaParams::default(),
            )?;
            let recovered: Vec<_> = result.points.iter().map(|x| *x / world_scale).collect();
            assert!(median_error(&recovered, &s.truth) < 1e-7);
        }
        Ok(())
    }

    #[test]
    fn coordinates_overflowing_normalization_are_invalid_input() {
        let s = scene(60, 0., 3., 100.);
        let points: Vec<_> = s.start.iter().map(|x| *x * 1e200).collect();
        assert!(matches!(
            bundle_adjust_affine(
                &s.cameras,
                &points,
                &s.observations,
                &s.fixed,
                &AffineBaParams::default()
            ),
            Err(AffineBaError::InvalidInput(_))
        ));
    }

    #[test]
    fn validation_errors_name_the_failing_item() {
        let s = scene(60, 0., 3., 100.);
        let params = AffineBaParams::default();
        let message = |observations: &[AffineObservation]| {
            bundle_adjust_affine(&s.cameras, &s.start, observations, &s.fixed, &params)
                .map(|_| ())
                .unwrap_err()
                .to_string()
        };
        let mut bad_camera = s.observations.clone();
        bad_camera[3].camera_index = 9;
        assert!(message(&bad_camera).contains("observation 3"));
        assert!(message(&bad_camera).contains("camera index 9"));
        let mut bad_point = s.observations.clone();
        bad_point[4].point_index = 70;
        assert!(message(&bad_point).contains("point index 70"));
        let mut bad_pixel = s.observations.clone();
        bad_pixel[5].pixel.y = f64::NAN;
        assert!(message(&bad_pixel).contains("observation 5"));
        let unobserved: Vec<_> = s
            .observations
            .iter()
            .copied()
            .filter(|o| o.point_index != 2)
            .collect();
        assert!(message(&unobserved).contains("point 2 has no observations"));
        let single_view: Vec<_> = s
            .observations
            .iter()
            .copied()
            .filter(|o| o.point_index != 7 || o.camera_index == 0)
            .collect();
        assert!(message(&single_view).contains("free point 7"));
        let starved: Vec<_> = s
            .observations
            .iter()
            .copied()
            .filter(|o| o.camera_index != 2 || o.point_index < 3)
            .collect();
        assert!(message(&starved).contains("camera 2"));
        // Repeats appended in the opposite order to their points, and separated from the
        // original by the other cameras: the earliest observation is the one reported.
        let of_point = |p| {
            s.observations
                .iter()
                .copied()
                .find(|o| o.point_index == p)
                .unwrap()
        };
        let mut repeated = s.observations.clone();
        let first_repeat = repeated.len();
        repeated.push(of_point(50));
        repeated.push(of_point(2));
        assert!(message(&repeated).contains(&format!(
            "observation {first_repeat}: duplicate of camera {} and point 50",
            of_point(50).camera_index
        )));
    }

    #[test]
    fn observation_index_keeps_observation_order_and_empty_rows() {
        let observations: Vec<_> = [2, 0, 2, 0, 2]
            .into_iter()
            .map(|point_index| AffineObservation {
                camera_index: 0,
                point_index,
                pixel: Vec2F64::new(0., 0.),
            })
            .collect();
        let index = ObservationIndex::new(4, &observations, |o| o.point_index);
        assert_eq!(index.row(0), [1, 3]);
        assert_eq!(index.row(1), [] as [usize; 0]);
        assert_eq!(index.row(2), [0, 2, 4]);
        assert_eq!(index.row(3), [] as [usize; 0]);
    }

    /// Moves every free point and camera far enough that steps get rejected.
    fn far_start(s: &mut Scene, far: f64) {
        for (i, x) in s.start.iter_mut().enumerate().skip(4) {
            x.x += far * ((i * 37 % 17) as f64 / 8.5 - 1.);
            x.y -= far * ((i * 53 % 19) as f64 / 9.5 - 1.);
        }
        for c in &mut s.cameras {
            c.matrix[0][3] += far;
            c.matrix[1][0] -= far * 0.2;
        }
    }

    #[test]
    fn no_accepted_step_reports_no_progress() {
        let mut s = scene(60, 0., 3., 100.);
        far_start(&mut s, 1000.);
        let params = AffineBaParams {
            max_trials: 1,
            ..Default::default()
        };
        assert!(matches!(
            bundle_adjust_affine(&s.cameras, &s.start, &s.observations, &s.fixed, &params),
            Err(AffineBaError::NoProgress { iteration: 1 })
        ));
    }

    #[test]
    fn rejection_after_progress_returns_the_last_state() -> Result<(), AffineBaError> {
        let mut s = scene(60, 0., 3., 100.);
        far_start(&mut s, 3000.);
        let params = AffineBaParams {
            max_trials: 1,
            initial_damping: 1e-15,
            ..Default::default()
        };
        let result =
            bundle_adjust_affine(&s.cameras, &s.start, &s.observations, &s.fixed, &params)?;
        assert_eq!(result.accepted_steps, 1);
        assert!(!result.converged);
        assert!(result.cost < result.initial_cost);
        Ok(())
    }

    #[test]
    fn single_view_extra_anchor_is_allowed() -> Result<(), AffineBaError> {
        let mut s = scene(60, 0., 3., 100.);
        s.fixed[4] = true;
        s.start[4] = s.truth[4];
        let observations: Vec<_> = s
            .observations
            .iter()
            .copied()
            .filter(|o| o.point_index != 4 || o.camera_index == 0)
            .collect();
        let result = bundle_adjust_affine(
            &s.cameras,
            &s.start,
            &observations,
            &s.fixed,
            &AffineBaParams::default(),
        )?;
        assert!(median_error(&result.points, &s.truth) < 1e-7);
        Ok(())
    }

    #[test]
    fn cauchy_matches_the_shared_kornia_algebra_loss() {
        use kornia_algebra::optim::losses::{CauchyLoss, RobustLoss};
        let scale = 1.5_f32;
        let shared = CauchyLoss::new(scale).unwrap();
        let scale2 = f64::from(scale * scale);
        for radius in [1e-2_f32, 0.1, 1., 1.5, 10., 1e3] {
            let squared = radius * radius;
            let cost = cauchy_cost(f64::from(squared), scale2);
            let weight = cauchy_weight(f64::from(squared), scale2);
            let shared_cost = 0.5 * f64::from(shared.rho(squared));
            let shared_weight = f64::from(shared.weight(squared));
            assert!((cost - shared_cost).abs() <= 1e-5 * shared_cost, "{radius}");
            assert!(
                (weight - shared_weight).abs() <= 1e-5 * shared_weight,
                "{radius}"
            );
        }
    }

    #[test]
    fn cauchy_weight_is_the_derivative_of_the_cost() {
        let scale2 = 2.25;
        for squared in [1e-2, 0.5, 2.25, 40., 1e4] {
            let step = 1e-6 * squared;
            let derivative = (cauchy_cost(squared + step, scale2)
                - cauchy_cost(squared - step, scale2))
                / (2. * step);
            assert!((2. * derivative - cauchy_weight(squared, scale2)).abs() < 1e-7);
        }
    }
}
