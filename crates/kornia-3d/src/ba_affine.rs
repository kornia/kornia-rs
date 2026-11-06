//! Simultaneous affine bundle adjustment using Cauchy IRLS and Schur LM.
//!
//! Uses the same block elimination as [`crate::ba_schur`], with eight additive camera
//! parameters instead of six SE(3) tangents. All assembly and solves are f64.
//! Point blocks use nalgebra Cholesky; the small reduced camera system is dense.
//! Four noncoplanar fixed points define the affine gauge. Inputs must form one
//! connected camera/point graph. The solver normalizes world coordinates internally.
//! This solver has a separate API: no pinhole priors, shared
//! intrinsics, automatic gauge selection, or mixed camera types are implied.
//!
//! The Cauchy objective follows `kornia_algebra::optim::losses::CauchyLoss`,
//! evaluated here in f64 (that trait accepts f32). nalgebra provides the SVD
//! gauge check and Cholesky solves. The dense reduced system uses O(C²) storage
//! and O(C³) factorization work, so this API targets few-camera, many-point scenes.
use crate::camera::AffineCamera;
use kornia_algebra::Vec3F64;
use nalgebra::{DMatrix, DVector, SMatrix, SVector};

const CAMERA_DIM: usize = 8;
const POINT_DIM: usize = 3;
const DIAGONAL_FLOOR: f64 = 1e-12;
const GAUGE_SINGULAR_RATIO: f64 = 1e-6;
const GRADIENT_TOLERANCE: f64 = 1e-10;
const MIN_DAMPING: f64 = 1e-15;
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
    pub pixel: [f64; 2],
}

/// Parameters for affine Schur Levenberg–Marquardt.
#[derive(Clone, Debug)]
pub struct AffineBaParams {
    /// Maximum number of linearizations (rejected damping trials do not count).
    pub max_iterations: usize,
    /// Cauchy scale applied to the norm of each 2D residual, in pixels.
    pub robust_scale: f64,
    /// Relative accepted cost decrease at which to stop.
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
    /// Whether the accepted cost decrease or gradient satisfied the stopping criterion.
    pub converged: bool,
}

/// Input or numerical failure in affine BA.
#[derive(Debug, thiserror::Error)]
pub enum AffineBaError {
    /// Invalid sizes, indices, parameters, or nonfinite input.
    #[error("invalid affine BA input: {0}")]
    InvalidInput(String),
    /// Fixed points do not define a sufficiently noncoplanar affine frame.
    #[error("affine BA requires four noncoplanar fixed points")]
    InvalidGauge,
    /// No acceptable finite step after all damping trials.
    #[error("affine LM exhausted damping trials at iteration {iteration}")]
    NoProgress {
        /// One-based linearization index.
        iteration: usize,
    },
}

fn invalid(message: &str) -> AffineBaError {
    AffineBaError::InvalidInput(message.into())
}

fn validate(
    cameras: &[AffineCamera],
    points: &[Vec3F64],
    obs: &[AffineObservation],
    fixed: &[bool],
    params: &AffineBaParams,
) -> Result<(), AffineBaError> {
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
    let mut point_views = vec![Vec::new(); points.len()];
    let mut camera_points = vec![Vec::new(); cameras.len()];
    for o in obs {
        if o.camera_index >= cameras.len()
            || o.point_index >= points.len()
            || o.pixel.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid observation index or pixel"));
        }
        if point_views[o.point_index].contains(&o.camera_index) {
            return Err(invalid("duplicate camera/point observation"));
        }
        point_views[o.point_index].push(o.camera_index);
        camera_points[o.camera_index].push(o.point_index);
    }
    if point_views
        .iter()
        .zip(fixed)
        .any(|(v, f)| v.is_empty() || (!f && v.len() < 2))
        || camera_points.iter().any(|p| p.len() < 4)
    {
        return Err(invalid(
            "each camera needs four points; each free point needs two views",
        ));
    }
    let mut seen = vec![false; cameras.len()];
    let mut todo = vec![0];
    seen[0] = true;
    while let Some(c) = todo.pop() {
        for &p in &camera_points[c] {
            for &v in &point_views[p] {
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
    Ok(())
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
            let r = cameras[o.camera_index].project(&points[o.point_index])
                - kornia_algebra::Vec2F64::from_array(o.pixel);
            0.5 * scale2 * (r.dot(r) / scale2).ln_1p()
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
            let r = camera.project(x) - kornia_algebra::Vec2F64::from_array(o.pixel);
            let w = 1. / (1. + r.dot(r) / (scale * scale));
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
}

/// Optimize affine cameras and 3D points together with Schur LM.
///
/// Fixing four noncoplanar points selects a representative of the 12-dimensional
/// affine gauge. Fixing additional points imposes additional geometric constraints.
/// Normalization and damping do not substitute for these anchors. Degenerate local
/// geometry is damped; no observations or points are silently discarded.
///
/// # Arguments
/// * `cameras` - Initial world-to-pixel affine matrices, all adjustable.
/// * `points` - Initial finite world coordinates.
/// * `observations` - Unique camera/point pairs in one connected component.
/// * `fixed_points` - One flag per point; true points are held exactly.
/// * `params` - Cauchy scale and LM stopping parameters.
/// # Returns
/// Optimized state and convergence statistics. Exhausting the iteration cap without
/// satisfying a stopping criterion returns the last accepted state with `converged = false`.
/// # Errors
/// Returns [`AffineBaError`] for invalid inputs, a degenerate fixed-point frame,
/// or exhausted damping trials. Planar anchor sets are rejected explicitly.
/// # Example
/// ```
/// use kornia_3d::{ba_affine::*, camera::AffineCamera};
/// use kornia_algebra::Vec3F64;
/// let points = vec![Vec3F64::ZERO, Vec3F64::new(1.,0.,0.), Vec3F64::new(0.,1.,0.), Vec3F64::new(0.,0.,1.)];
/// let cameras = vec![AffineCamera { matrix: [[1.,0.,0.,0.], [0.,1.,0.,0.]] }];
/// let observations: Vec<_> = points.iter().enumerate().map(|(p,x)| AffineObservation {
///     camera_index: 0, point_index: p, pixel: cameras[0].project(x).to_array()
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
    validate(cameras, points, observations, fixed_points, params)?;
    let center = points.iter().copied().fold(Vec3F64::ZERO, |a, b| a + b) / points.len() as f64;
    let scale = (points
        .iter()
        .map(|x| (*x - center).dot(*x - center))
        .sum::<f64>()
        / points.len() as f64)
        .sqrt();
    if !scale.is_finite() || scale <= f64::EPSILON {
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
    let anchors: Vec<_> = x
        .iter()
        .zip(fixed_points)
        .filter_map(|(x, f)| f.then_some(*x))
        .collect();
    if anchors.len() < 4 {
        return Err(AffineBaError::InvalidGauge);
    }
    let anchor_matrix = DMatrix::from_fn(anchors.len(), 4, |i, j| {
        if j == 3 {
            1.
        } else {
            anchors[i].to_array()[j]
        }
    });
    let singular = anchor_matrix.svd(false, false).singular_values;
    if singular.min() < GAUGE_SINGULAR_RATIO * singular.max() {
        return Err(AffineBaError::InvalidGauge);
    }
    let mut by_point = vec![Vec::new(); x.len()];
    for (i, o) in observations.iter().enumerate() {
        by_point[o.point_index].push(i);
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
    for iteration in 1..=params.max_iterations {
        iterations = iteration;
        normal.assemble(&cams, &x, observations, fixed_points, params.robust_scale);
        let gradient_max = normal
            .gc
            .iter()
            .flat_map(|g| g.iter())
            .chain(normal.gp.iter().flat_map(|g| g.iter()))
            .fold(0.0_f64, |a, b| a.max(b.abs()));
        if gradient_max < GRADIENT_TOLERANCE {
            converged = true;
            break;
        }
        let mut accepted = false;
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
                for &a in &by_point[p] {
                    let ca = observations[a].camera_index;
                    let t = normal.cp[a] * inv[p];
                    let b = t * normal.gp[p];
                    for i in 0..CAMERA_DIM {
                        rhs[ca * CAMERA_DIM + i] += b[i];
                    }
                    for &b in &by_point[p] {
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
                    for &o in &by_point[p] {
                        b += normal.cp[o].transpose() * dc[observations[o].camera_index];
                    }
                    dp[p] = -inv[p] * b;
                    trial_x[p] = x[p] + Vec3F64::new(dp[p][0], dp[p][1], dp[p][2]);
                }
                let next_cost = cost(&trial_cams, &trial_x, observations, params.robust_scale);
                if next_cost.is_finite() && next_cost < current_cost {
                    let mut pred = 0.;
                    for (c, step) in dc.iter().enumerate() {
                        for (k, delta) in step.iter().enumerate() {
                            pred += delta
                                * (damping * normal.cc[c][(k, k)].max(DIAGONAL_FLOOR) * delta
                                    - normal.gc[c][k]);
                        }
                    }
                    for (p, step) in dp.iter().enumerate() {
                        if !fixed_points[p] {
                            for (k, delta) in step.iter().enumerate() {
                                pred += delta
                                    * (damping * normal.pp[p][(k, k)].max(DIAGONAL_FLOOR) * delta
                                        - normal.gp[p][k]);
                            }
                        }
                    }
                    let decrease = current_cost - next_cost;
                    let rho = decrease / (0.5 * pred).max(f64::MIN_POSITIVE);
                    damping *= (1.0_f64 / 3.).max(1. - (2. * rho - 1.).powi(3));
                    damping = damping.max(MIN_DAMPING);
                    nu = 2.;
                    converged = decrease < params.cost_tolerance * current_cost;
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
            return Err(AffineBaError::NoProgress { iteration });
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
                        pixel: cam.project(x).to_array(),
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
                pixel: [3., 4.],
            },
            AffineObservation {
                camera_index: 0,
                point_index: 0,
                pixel: [0., 0.],
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
                pixel,
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
        let center = points.iter().copied().fold(Vec3F64::ZERO, |a, b| a + b) / points.len() as f64;
        let scale = (points
            .iter()
            .map(|x| (*x - center).dot(*x - center))
            .sum::<f64>()
            / points.len() as f64)
            .sqrt();
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
            o.pixel[0] += 25.;
            o.pixel[1] -= 15.;
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
}
