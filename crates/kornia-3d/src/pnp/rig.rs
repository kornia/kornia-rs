//! Absolute pose of a multi-camera rig from 2D-3D matches given as rays.
//!
//! Each match is a ray of one rig camera (origin and unit direction in the rig
//! frame) and the world point it sees. Minimal samples come from one camera:
//! [`ap3p_lifted`](crate::pnp::rig::ap3p_lifted) solves AP3P (Ke and Roumeliotis, CVPR 2017, through
//! [`solve_ap3p_multi`](crate::pnp::ap3p::solve_ap3p_multi)) in a virtual pinhole at the shared
//! origin and lifts the poses to the rig. Three rays with different origins
//! return no pose; this is not a generalized P3P (gP3P, UPnP) solver. The
//! RANSAC driver draws hypotheses from a prefix of the matches (one camera's)
//! and scores them on every camera of the rig, in pixels on each camera's image
//! plane.

use super::ap3p::solve_ap3p_multi;
use super::PnPError;
use crate::camera::Ray;
use crate::pose::Pose3d;
use crate::ransac::{run, Estimator, RansacConfig, Sampler, ThresholdConsensus};
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec3AF32, Vec3F64};

/// A 2D-3D match as a ray of a multi-camera rig, everything in the rig frame.
#[derive(Debug, Clone, Copy)]
pub struct RayMatch {
    /// The camera ray of the observation: the camera centre (metres) and the
    /// unit bearing of the pixel.
    pub ray: Ray,
    /// The camera's unit optical axis.
    pub axis: Vec3F64,
    /// The matched point in the world frame.
    pub point_world: Vec3F64,
    /// Pixels per unit of image-plane error (the camera's fx): residuals come out in px².
    pub pixel_scale: f64,
}

/// The outcome of [`ransac_lifted_ap3p`].
#[derive(Debug, Clone)]
pub struct RigPnPResult {
    /// The world-to-rig pose.
    pub pose_world_to_rig: Pose3d,
    /// Per match, whether its residual is within the threshold.
    pub inliers: Vec<bool>,
}

/// Shortest sum of three unit bearings, or RMS radius of three world points,
/// that still defines a frame to solve in.
const MIN_SPREAD: f64 = 1e-9;

/// World-to-rig poses from three matches seen by one camera.
///
/// Runs AP3P in a virtual pinhole at the shared origin, looking along the mean
/// bearing, on world points centred and scaled for its `f32` arithmetic, and
/// lifts each solution to the rig frame.
///
/// # Arguments
///
/// * `matches` - Three matches whose ray origins are bit-equal.
///
/// # Returns
///
/// Up to 4 world-to-rig poses; empty when the origins differ, the bearings or
/// the points are degenerate, or AP3P finds no solution.
///
/// # Errors
///
/// This function is infallible; degenerate input returns no pose.
pub fn ap3p_lifted(matches: &[RayMatch; 3]) -> Vec<Pose3d> {
    let origin = matches[0].ray.origin;
    if matches[1..].iter().any(|m| m.ray.origin != origin) {
        return Vec::new();
    }
    let mean_bearing = matches
        .iter()
        .fold(Vec3F64::ZERO, |acc, m| acc + m.ray.direction);
    if !spread_enough(mean_bearing.length()) {
        return Vec::new();
    }
    // A virtual pinhole at the origin looking along the mean bearing keeps the
    // three bearings in front of it.
    let to_virtual = rotation_taking_to_z(mean_bearing);
    let mut image = [Vec2F32::ZERO; 3];
    for (pixel, m) in image.iter_mut().zip(matches) {
        let b = to_virtual * m.ray.direction;
        if b.z <= 0.0 {
            return Vec::new();
        }
        *pixel = Vec2F32::new((b.x / b.z) as f32, (b.y / b.z) as f32);
    }
    // kornia's AP3P is f32: centre and scale the points so its rounding is relative.
    let centre = matches
        .iter()
        .fold(Vec3F64::ZERO, |acc, m| acc + m.point_world)
        / 3.0;
    let scale = (matches
        .iter()
        .map(|m| (m.point_world - centre).dot(m.point_world - centre))
        .sum::<f64>()
        / 3.0)
        .sqrt();
    if !spread_enough(scale) {
        return Vec::new();
    }
    let world = (*matches).map(|m| {
        let p = (m.point_world - centre) / scale;
        Vec3AF32::new(p.x as f32, p.y as f32, p.z as f32)
    });
    let Ok(solutions) = solve_ap3p_multi(&world, &image, &Mat3AF32::IDENTITY) else {
        return Vec::new();
    };
    // X_virtual = R (X - c) + s t', and X_rig = R_sᵀ X_virtual + origin.
    let from_virtual = to_virtual.transpose();
    solutions
        .iter()
        .map(|solution| {
            let r = Mat3F64::from(solution.rotation.as_dmat3());
            let t = solution.translation;
            let t = Vec3F64::new(f64::from(t.x), f64::from(t.y), f64::from(t.z));
            Pose3d::new(
                from_virtual * r,
                from_virtual * (t * scale - r * centre) + origin,
            )
        })
        .collect()
}

fn spread_enough(spread: f64) -> bool {
    spread.is_finite() && spread > MIN_SPREAD
}

/// A rotation `R` with `R · axis / |axis| = +z`.
fn rotation_taking_to_z(axis: Vec3F64) -> Mat3F64 {
    let z = axis.normalize();
    let helper = if z.x.abs() < 0.9 {
        Vec3F64::new(1.0, 0.0, 0.0)
    } else {
        Vec3F64::new(0.0, 1.0, 0.0)
    };
    let x = (helper - z * helper.dot(z)).normalize();
    let y = z.cross(x);
    Mat3F64::from_cols(x, y, z).transpose()
}

/// Squared pixel error of a match on its camera's image plane.
///
/// `(pixel_scale · |p/(p·a) − d/(d·a)|)²` with `p = R X + t − origin`, `d` the
/// ray direction and `a` the optical axis, so an off-axis error reads in pixels
/// both radially and tangentially.
///
/// # Arguments
///
/// * `pose_world_to_rig` - The world-to-rig pose to score.
/// * `m` - The match.
///
/// # Returns
///
/// The squared error in px², or `f64::INFINITY` when the point or the ray is
/// not in front of the camera.
///
/// # Errors
///
/// This function is infallible.
pub fn ray_residual_sq(pose_world_to_rig: &Pose3d, m: &RayMatch) -> f64 {
    let p = pose_world_to_rig.transform_point(&m.point_world) - m.ray.origin;
    let (depth, along) = (p.dot(m.axis), m.ray.direction.dot(m.axis));
    if depth <= 0.0 || along <= 0.0 {
        return f64::INFINITY;
    }
    (m.pixel_scale * (p / depth - m.ray.direction / along).length()).powi(2)
}

/// [`ap3p_lifted`] as a RANSAC [`Estimator`], with [`ray_residual_sq`] as its residual.
///
/// It has no non-minimal solver: the driver's refit (the final one, and
/// LO-RANSAC's) falls back to the default, a minimal fit on the first three
/// inliers, which gives no pose when they come from different cameras. So
/// the refit does not polish the pose as it does for least-squares estimators.
pub struct LiftedAp3pEstimator;

impl Estimator for LiftedAp3pEstimator {
    type Model = Pose3d;
    type Sample = RayMatch;
    const SAMPLE_SIZE: usize = 3;

    fn fit(&self, samples: &[RayMatch], out: &mut Vec<Pose3d>) {
        if let Ok(triple) = <&[RayMatch; 3]>::try_from(samples) {
            out.extend(ap3p_lifted(triple));
        }
    }

    fn residual(&self, model: &Pose3d, sample: &RayMatch) -> f64 {
        ray_residual_sq(model, sample)
    }
}

/// A RANSAC sampler that draws only from the first `prefix` samples.
///
/// Uses its own splitmix64 state, so a seed gives the same draws on every
/// platform.
pub struct PrefixSampler {
    state: u64,
    prefix: usize,
}

impl PrefixSampler {
    /// A sampler over `[0, prefix)` seeded with `seed`.
    ///
    /// # Arguments
    ///
    /// * `seed` - Initial splitmix64 state.
    /// * `prefix` - Draws come from `[0, min(prefix, n))`.
    ///
    /// # Returns
    ///
    /// The sampler.
    ///
    /// # Errors
    ///
    /// This function is infallible.
    pub fn new(seed: u64, prefix: usize) -> Self {
        Self {
            state: seed,
            prefix,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, bound)` (Lemire's multiply-shift; the bias is below 2⁻⁵⁰).
    fn below(&mut self, bound: usize) -> usize {
        ((u128::from(self.next_u64()) * bound as u128) >> 64) as usize
    }
}

impl Sampler for PrefixSampler {
    /// Floyd's combination algorithm over `[0, min(prefix, n))`. When `out` is
    /// longer than that population, only its first `population` entries are drawn.
    fn sample(&mut self, n: usize, out: &mut [usize]) {
        let population = self.prefix.min(n);
        let k = out.len();
        for (filled, j) in (population.saturating_sub(k)..population).enumerate() {
            let t = self.below(j + 1);
            out[filled] = if out[..filled].contains(&t) { j } else { t };
        }
    }
}

/// RANSAC rig pose from one camera's hypotheses, scored on every camera.
///
/// Runs [`crate::ransac::run`] with [`LiftedAp3pEstimator`] and
/// [`ThresholdConsensus`]: hypotheses come from triples of
/// `matches[..hypothesis_count]` (the matches of one camera, placed first), and
/// each is scored on all matches by [`ray_residual_sq`]. All `max_iterations`
/// draws run: the driver's adaptive stop counts support over every match, but
/// only the prefix is drawn from, so a pose carried by the other cameras would
/// end the search early.
///
/// # Arguments
///
/// * `matches` - All matches, hypothesis camera first.
/// * `hypothesis_count` - Length of the prefix to draw from.
/// * `threshold_px` - Inlier threshold in pixels.
/// * `max_iterations` - Number of draws.
/// * `seed` - Sampler seed.
///
/// # Returns
///
/// The best pose with its inlier mask over `matches`, or `Ok(None)` when no
/// draw gives a pose.
///
/// # Errors
///
/// [`PnPError::InsufficientCorrespondences`] when the prefix has fewer than 3
/// matches; [`PnPError::InvalidInput`] when the prefix rays do not all share
/// one origin (the prefix spans cameras, so no triple would give a pose).
///
/// # Example
///
/// ```rust
/// use kornia_3d::camera::Ray;
/// use kornia_3d::pnp::rig::{ransac_lifted_ap3p, RayMatch};
/// use kornia_3d::pose::Pose3d;
/// use kornia_algebra::{Mat3F64, Vec3F64};
///
/// // The rig sits at world (0, 0, -1); its one camera is at the rig origin.
/// let truth = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 1.0));
/// let matches: Vec<RayMatch> = (0..12)
///     .map(|i| {
///         let p_rig = Vec3F64::new((i % 4) as f64 * 0.5 - 0.7, (i / 4) as f64 * 0.4 - 0.4, 4.0 + (i % 3) as f64);
///         RayMatch {
///             ray: Ray { origin: Vec3F64::ZERO, direction: p_rig.normalize() },
///             axis: Vec3F64::new(0.0, 0.0, 1.0),
///             point_world: truth.inverse().transform_point(&p_rig),
///             pixel_scale: 450.0,
///         }
///     })
///     .collect();
/// let result = ransac_lifted_ap3p(&matches, matches.len(), 2.0, 50, 7)?.unwrap();
/// assert!((result.pose_world_to_rig.translation - truth.translation).length() < 1e-3);
/// assert!(result.inliers.iter().all(|&inlier| inlier));
/// # Ok::<(), kornia_3d::pnp::PnPError>(())
/// ```
pub fn ransac_lifted_ap3p(
    matches: &[RayMatch],
    hypothesis_count: usize,
    threshold_px: f64,
    max_iterations: u32,
    seed: u64,
) -> Result<Option<RigPnPResult>, PnPError> {
    let prefix = hypothesis_count.min(matches.len());
    if prefix < LiftedAp3pEstimator::SAMPLE_SIZE {
        return Err(PnPError::InsufficientCorrespondences {
            required: LiftedAp3pEstimator::SAMPLE_SIZE,
            actual: prefix,
        });
    }
    // A prefix that spans cameras is a layout bug in the caller; without this
    // it would read as no consensus.
    let origin = matches[0].ray.origin;
    if let Some(i) = matches[..prefix]
        .iter()
        .position(|m| m.ray.origin != origin)
    {
        return Err(PnPError::InvalidInput(format!(
            "hypothesis match {i} has a different ray origin than match 0"
        )));
    }
    let threshold_sq = threshold_px * threshold_px;
    // No adaptive stop: see the rustdoc.
    let config = RansacConfig {
        max_iters: max_iterations,
        inlier_threshold: threshold_sq,
        confidence: 1.0,
        ..RansacConfig::default()
    };
    let result = run(
        &LiftedAp3pEstimator,
        &ThresholdConsensus {
            threshold: threshold_sq,
        },
        &mut PrefixSampler::new(seed, prefix),
        matches,
        &config,
    );
    Ok(result.model.map(|pose_world_to_rig| RigPnPResult {
        pose_world_to_rig,
        inliers: result.inliers,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::SO3F64;

    const FX: f64 = 240.0;

    fn yaw(deg: f64) -> Mat3F64 {
        SO3F64::exp(Vec3F64::new(0.0, deg.to_radians(), 0.0)).matrix()
    }

    /// `T_PC` of a rig camera turned 70 deg and offset 11 cm from the primary.
    fn side_camera() -> Pose3d {
        Pose3d::new(yaw(70.0), Vec3F64::new(0.11, 0.0, 0.0))
    }

    fn truth() -> Pose3d {
        Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.05, -0.3, 0.1)).matrix(),
            Vec3F64::new(0.4, -0.2, 1.3),
        )
    }

    /// The match of a point at `p_camera` in the camera with `T_PC`.
    fn ray_match(primary_from_camera: &Pose3d, p_camera: Vec3F64) -> RayMatch {
        let p_rig = primary_from_camera.transform_point(&p_camera);
        RayMatch {
            ray: Ray {
                origin: primary_from_camera.translation,
                direction: primary_from_camera.rotation * p_camera.normalize(),
            },
            axis: primary_from_camera.rotation * Vec3F64::new(0.0, 0.0, 1.0),
            point_world: truth().inverse().transform_point(&p_rig),
            pixel_scale: FX,
        }
    }

    fn rotation_error(a: &Pose3d, b: &Pose3d) -> f64 {
        SO3F64::from_matrix(&(a.rotation * b.rotation.transpose()))
            .log()
            .length()
    }

    #[test]
    fn three_side_camera_rays_give_the_rig_pose() {
        let camera = side_camera();
        let matches = [
            ray_match(&camera, Vec3F64::new(-0.8, 0.3, 4.0)),
            ray_match(&camera, Vec3F64::new(0.6, -0.5, 5.5)),
            ray_match(&camera, Vec3F64::new(0.2, 0.7, 3.2)),
        ];
        let poses = ap3p_lifted(&matches);
        assert!(!poses.is_empty() && poses.len() <= 4);
        let truth = truth();
        let found = poses.iter().any(|pose| {
            rotation_error(pose, &truth) < 2e-5
                && (pose.translation - truth.translation).length() < 1e-4
        });
        assert!(found, "{poses:?}");
        for m in &matches {
            let best = poses
                .iter()
                .map(|pose| ray_residual_sq(pose, m))
                .fold(f64::INFINITY, f64::min);
            assert!(best < 1e-4, "residual {best} px²");
        }
    }

    /// The residual is the pixel error on the camera's image plane. A 4 px error at 45 deg
    /// off the axis of the side camera reads 4 px radially and tangentially.
    #[test]
    fn an_off_axis_error_reads_in_pixels() {
        let camera = side_camera();
        let m = ray_match(&camera, Vec3F64::new(4.0, 0.0, 4.0));
        for offset in [
            Vec3F64::new(4.0 / FX, 0.0, 0.0),
            Vec3F64::new(0.0, 4.0 / FX, 0.0),
        ] {
            let observed = Vec3F64::new(1.0, 0.0, 1.0) + offset;
            let m = RayMatch {
                ray: Ray {
                    direction: camera.rotation * observed.normalize(),
                    ..m.ray
                },
                ..m
            };
            let px = ray_residual_sq(&truth(), &m).sqrt();
            assert!((px - 4.0).abs() < 1e-9, "{px} px for {offset:?}");
        }
    }

    #[test]
    fn rays_from_two_origins_are_not_lifted() {
        let matches = [
            ray_match(&side_camera(), Vec3F64::new(-0.8, 0.3, 4.0)),
            ray_match(&side_camera(), Vec3F64::new(0.6, -0.5, 5.5)),
            ray_match(&Pose3d::IDENTITY, Vec3F64::new(0.2, 0.7, 3.2)),
        ];
        assert!(ap3p_lifted(&matches).is_empty());
    }

    #[test]
    fn a_point_behind_the_ray_has_no_residual() {
        let m = ray_match(&Pose3d::IDENTITY, Vec3F64::new(0.1, 0.2, 3.0));
        assert!(ray_residual_sq(&truth(), &m) < 1e-18);
        let behind = RayMatch {
            ray: Ray {
                direction: -m.ray.direction,
                ..m.ray
            },
            ..m
        };
        assert_eq!(ray_residual_sq(&truth(), &behind), f64::INFINITY);
        let behind_camera = RayMatch {
            point_world: truth()
                .inverse()
                .transform_point(&Vec3F64::new(-0.1, -0.2, -3.0)),
            ..m
        };
        assert_eq!(ray_residual_sq(&truth(), &behind_camera), f64::INFINITY);
    }

    #[test]
    fn the_sampler_draws_distinct_indices_from_its_prefix() {
        let mut sampler = PrefixSampler::new(3, 7);
        let mut out = [0usize; 3];
        for _ in 0..1000 {
            sampler.sample(100, &mut out);
            assert!(out.iter().all(|&i| i < 7), "{out:?}");
            assert!(out[0] != out[1] && out[0] != out[2] && out[1] != out[2]);
        }
    }

    #[test]
    fn ransac_finds_the_pose_from_one_camera_and_scores_both() {
        let side = side_camera();
        let mut matches = Vec::new();
        for i in 0..40 {
            let (x, y) = ((i % 8) as f64 * 0.4 - 1.4, (i / 8) as f64 * 0.3 - 0.6);
            let mut m = ray_match(&side, Vec3F64::new(x, y, 4.0 + (i % 3) as f64));
            if i % 3 == 0 {
                m.point_world += Vec3F64::new(0.7, -0.4, 0.5);
            }
            matches.push(m);
        }
        for i in 0..30 {
            let (x, y) = ((i % 6) as f64 * 0.5 - 1.2, (i / 6) as f64 * 0.4 - 0.8);
            matches.push(ray_match(
                &Pose3d::IDENTITY,
                Vec3F64::new(x, y, 6.0 + (i % 2) as f64),
            ));
        }
        let RigPnPResult {
            pose_world_to_rig: pose,
            inliers,
        } = ransac_lifted_ap3p(&matches, 40, 4.0, 300, 0)
            .unwrap()
            .unwrap();
        let truth = truth();
        assert!(rotation_error(&pose, &truth) < 1e-4);
        assert!((pose.translation - truth.translation).length() < 1e-3);
        assert_eq!(inliers.iter().filter(|&&b| b).count(), 26 + 30);
        assert!(inliers[40..].iter().all(|&b| b));
    }

    /// A mediocre pose whose support is mostly the other camera must not end the search
    /// before a clean triple of the hypothesis camera is drawn. The prefix holds 8 true and 32
    /// matches of a pose 0.5 m off; 100 distant points fit both poses and 60 near ones only the
    /// truth. The off pose (132 of 200) capped the loop at 21 draws over the whole population,
    /// so the truth (8 of 40 in the prefix, 0.57 % per draw) came out in about 11 % of seeds;
    /// 300 draws find it in about 82 %.
    #[test]
    fn a_mediocre_pose_does_not_cut_the_prefix_search_short() {
        let side = side_camera();
        let truth = truth();
        let off = Pose3d::new(
            truth.rotation,
            truth.translation + Vec3F64::new(0.5, 0.0, 0.0),
        );
        let seen_by = |pose: &Pose3d, camera: &Pose3d, p_camera: Vec3F64| RayMatch {
            point_world: pose
                .inverse()
                .transform_point(&camera.transform_point(&p_camera)),
            ..ray_match(camera, p_camera)
        };
        let grid = |i: usize, z: f64| {
            Vec3F64::new(
                ((i % 7) as f64 * 0.37 - 1.1) * z / 4.0,
                ((i / 7 % 6) as f64 * 0.29 - 0.7) * z / 4.0,
                z + (i % 3) as f64 * 0.4,
            )
        };
        let mut matches = Vec::new();
        for i in 0..40 {
            let pose = if i % 5 == 0 { &truth } else { &off };
            matches.push(seen_by(pose, &side, grid(i, 4.0)));
        }
        for i in 0..100 {
            matches.push(seen_by(&truth, &Pose3d::IDENTITY, grid(i, 100.0)));
        }
        for i in 0..60 {
            matches.push(seen_by(&truth, &Pose3d::IDENTITY, grid(i, 5.0)));
        }
        let found = (0..40)
            .filter(|&seed| {
                ransac_lifted_ap3p(&matches, 40, 4.0, 300, seed)
                    .unwrap()
                    .is_some_and(|result| {
                        (result.pose_world_to_rig.translation - truth.translation).length() < 0.05
                    })
            })
            .count();
        assert!(found >= 24, "the truth in {found} of 40 seeds");
    }

    #[test]
    fn too_few_hypothesis_matches_are_refused() {
        let m = ray_match(&Pose3d::IDENTITY, Vec3F64::new(0.1, 0.2, 3.0));
        assert!(matches!(
            ransac_lifted_ap3p(&[m; 10], 2, 4.0, 300, 0),
            Err(PnPError::InsufficientCorrespondences {
                required: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn a_prefix_spanning_two_origins_is_refused() {
        let m = ray_match(&Pose3d::IDENTITY, Vec3F64::new(0.1, 0.2, 3.0));
        let mut other = m;
        other.ray.origin = Vec3F64::new(0.1, 0.0, 0.0);
        let matches = [m, m, m, other, m];
        assert!(matches!(
            ransac_lifted_ap3p(&matches, 5, 4.0, 300, 0),
            Err(PnPError::InvalidInput(reason)) if reason.contains("match 3")
        ));
        // Past the prefix, other cameras are only scored.
        assert!(ransac_lifted_ap3p(&matches, 3, 4.0, 300, 0).is_ok());
    }
}
