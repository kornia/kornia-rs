//! Helpers shared by the unit tests and the PnP benches of this crate: seeded random poses and
//! the pixel observations they generate. The library compiles this module only for tests; the
//! benches include the file with `#[path]`.

use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec3AF32, Vec3F64, SO3F64};
use rand::{rngs::StdRng, RngExt};

/// A standard normal sample (Box–Muller); `rand` alone has no normal distribution.
pub(crate) fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// A rotation about a uniformly random axis by an angle drawn uniformly from `[0, max_angle)`.
pub(crate) fn random_rotation(rng: &mut StdRng, max_angle: f64) -> Mat3F64 {
    let axis = Vec3F64::new(gauss(rng), gauss(rng), gauss(rng)).normalize();
    SO3F64::exp(axis * rng.random_range(0.0..max_angle)).matrix()
}

/// The pinhole camera matrix with focal lengths `(fx, fy)` and principal point `(cx, cy)`.
pub(crate) fn k_matrix(fx: f32, fy: f32, cx: f32, cy: f32) -> Mat3AF32 {
    Mat3AF32::from_cols_array(&[fx, 0.0, 0.0, 0.0, fy, 0.0, cx, cy, 1.0])
}

/// A camera, a ground-truth world → camera pose and the correspondences they generate.
pub(crate) struct Scene {
    pub k: Mat3AF32,
    pub rotation: Mat3F64,
    pub translation: Vec3F64,
    pub world: Vec<Vec3AF32>,
    pub image: Vec<Vec2F32>,
}

impl Scene {
    /// Projects `world` with the pose through the pinhole camera `k`, adding Gaussian noise of
    /// `noise_px` pixels to each image coordinate.
    pub(crate) fn project(
        rng: &mut StdRng,
        k: Mat3AF32,
        world: Vec<Vec3F64>,
        rotation: Mat3F64,
        translation: Vec3F64,
        noise_px: f64,
    ) -> Self {
        let k64 = Mat3F64::from(k);
        let image = world
            .iter()
            .map(|&m| {
                let h = k64 * (rotation * m + translation);
                Vec2F32::new(
                    (h.x / h.z + noise_px * gauss(rng)) as f32,
                    (h.y / h.z + noise_px * gauss(rng)) as f32,
                )
            })
            .collect();
        Self {
            k,
            rotation,
            translation,
            world: world.iter().map(|m| m.as_vec3af32()).collect(),
            image,
        }
    }

    /// `n` random points in a box of half-width `size` around the origin, seen by `k` with a
    /// random rotation from `depth` metres.
    pub(crate) fn random(
        rng: &mut StdRng,
        k: Mat3AF32,
        n: usize,
        size: f64,
        depth: f64,
        noise_px: f64,
    ) -> Self {
        let world = (0..n)
            .map(|_| {
                Vec3F64::new(
                    rng.random_range(-size..size),
                    rng.random_range(-size..size),
                    rng.random_range(-size..size),
                )
            })
            .collect();
        let rotation = random_rotation(rng, std::f64::consts::PI);
        let translation = Vec3F64::new(
            rng.random_range(-0.5..0.5),
            rng.random_range(-0.5..0.5),
            depth,
        );
        Self::project(rng, k, world, rotation, translation, noise_px)
    }

    /// `n` random points on the square `[-1, 1]² × {0}` (a marker or chessboard), seen by `k`
    /// from 5 m with a rotation of up to 1.2 rad.
    pub(crate) fn random_planar(rng: &mut StdRng, k: Mat3AF32, n: usize, noise_px: f64) -> Self {
        let world = (0..n)
            .map(|_| {
                Vec3F64::new(
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                    0.0,
                )
            })
            .collect();
        let rotation = random_rotation(rng, 1.2);
        let translation = Vec3F64::new(0.1, -0.2, 5.0);
        Self::project(rng, k, world, rotation, translation, noise_px)
    }
}

/// Angle in degrees between an estimated rotation and the truth. `2·asin(‖R₁ − R₂‖_F / 2√2)`
/// stays accurate for tiny angles, unlike `acos((tr(R₁ᵀR₂) − 1) / 2)` on `f32` matrices.
pub(crate) fn rotation_error_deg(estimate: &Mat3AF32, truth: &Mat3F64) -> f64 {
    let estimate = Mat3F64::from(*estimate);
    let frobenius = (0..3)
        .map(|i| (estimate.col(i) - truth.col(i)).length_squared())
        .sum::<f64>()
        .sqrt();
    let chord = frobenius / (2.0 * std::f64::consts::SQRT_2);
    (2.0 * chord.clamp(0.0, 1.0).asin()).to_degrees()
}

/// Distance between an estimated translation and the truth.
pub(crate) fn translation_error(estimate: Vec3AF32, truth: Vec3F64) -> f64 {
    (Vec3F64::from(estimate) - truth).length()
}
