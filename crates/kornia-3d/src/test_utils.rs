//! Seeded random poses and the pixel observations they generate, shared by the unit tests.

use crate::camera::PinholeCamera;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec2F64, Vec3AF32, Vec3F64, SO3F64};
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

/// The camera most tests use: `f = 800` px, principal point `(640, 480)`.
pub(crate) fn k_default() -> Mat3AF32 {
    k_matrix(800.0, 800.0, 640.0, 480.0)
}

/// The shape of a random scene.
pub(crate) struct SceneConfig {
    /// Number of correspondences.
    pub n: usize,
    /// Standard deviation of the Gaussian pixel noise.
    pub noise_px: f64,
    /// Half the side of the cube (or square, when `planar`) holding the points, in metres.
    pub half_size: f64,
    /// Distance of the camera from the points, in metres.
    pub depth: f64,
    /// Points on the plane `z = 0`, like a marker or a chessboard.
    pub planar: bool,
    /// Largest rotation angle of the camera, in radians.
    pub max_angle: f64,
}

impl SceneConfig {
    /// `n` points in a 2 m cube seen from `depth` metres at any rotation.
    pub(crate) fn cube(n: usize, depth: f64, noise_px: f64) -> Self {
        Self {
            n,
            noise_px,
            half_size: 1.0,
            depth,
            planar: false,
            max_angle: std::f64::consts::PI,
        }
    }

    /// The same shape with the points on the plane `z = 0`.
    pub(crate) fn planar(self) -> Self {
        Self {
            planar: true,
            ..self
        }
    }
}

/// A camera, a ground-truth world → camera pose and the correspondences they generate.
pub(crate) struct Scene {
    /// Camera matrix.
    pub k: Mat3AF32,
    /// Ground-truth world → camera rotation.
    pub rotation: Mat3F64,
    /// Ground-truth world → camera translation.
    pub translation: Vec3F64,
    /// World points.
    pub world: Vec<Vec3AF32>,
    /// Their pixels, with noise; any outliers come last.
    pub image: Vec<Vec2F32>,
}

impl Scene {
    /// A random scene of shape `config`, seen through `k`. Points less than a tenth of the depth in
    /// front of the camera are drawn again.
    pub(crate) fn generate(rng: &mut StdRng, k: Mat3AF32, config: &SceneConfig) -> Self {
        let camera = pinhole(k);
        let rotation = random_rotation(rng, config.max_angle);
        let translation = Vec3F64::new(
            rng.random_range(-0.1..0.1) * config.depth,
            rng.random_range(-0.1..0.1) * config.depth,
            config.depth,
        );
        let mut world = Vec::with_capacity(config.n);
        let mut image = Vec::with_capacity(config.n);
        while world.len() < config.n {
            let z = if config.planar {
                0.0
            } else {
                rng.random_range(-config.half_size..config.half_size)
            };
            let p = Vec3F64::new(
                rng.random_range(-config.half_size..config.half_size),
                rng.random_range(-config.half_size..config.half_size),
                z,
            );
            let Some(uv) =
                camera.project_to_pixel(&(rotation * p + translation), 0.1 * config.depth)
            else {
                continue;
            };
            world.push(p.as_vec3af32());
            image.push(with_noise(rng, uv, config.noise_px));
        }
        Self {
            k,
            rotation,
            translation,
            world,
            image,
        }
    }

    /// Appends `count` wrong correspondences: world points in `[-1, 1]³`, pixels anywhere in the
    /// image (taken as twice the principal point).
    pub(crate) fn add_outliers(&mut self, rng: &mut StdRng, count: usize) {
        let (width, height) = (2.0 * self.k.z_axis().x, 2.0 * self.k.z_axis().y);
        for _ in 0..count {
            self.world.push(Vec3AF32::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ));
            self.image.push(Vec2F32::new(
                rng.random_range(0.0..width),
                rng.random_range(0.0..height),
            ));
        }
    }
}

/// The distortion-free camera with the focal lengths and principal point of `k`.
fn pinhole(k: Mat3AF32) -> PinholeCamera {
    let k = Mat3F64::from(k);
    PinholeCamera {
        fx: k.x_axis().x,
        fy: k.y_axis().y,
        cx: k.z_axis().x,
        cy: k.z_axis().y,
        ..PinholeCamera::IDENTITY
    }
}

/// The pixel `uv` with Gaussian noise of `noise_px` pixels.
fn with_noise(rng: &mut StdRng, uv: Vec2F64, noise_px: f64) -> Vec2F32 {
    Vec2F32::new(
        (uv.x + noise_px * gauss(rng)) as f32,
        (uv.y + noise_px * gauss(rng)) as f32,
    )
}

/// Angle in degrees between an estimated rotation and the truth. `2·asin(‖R₁ − R₂‖_F / 2√2)`
/// stays accurate for tiny angles, unlike `acos((tr(R₁ᵀR₂) − 1) / 2)` on `f32` matrices.
pub(crate) fn rotation_error_deg(estimate: impl Into<Mat3F64>, truth: Mat3F64) -> f64 {
    let estimate = estimate.into();
    let frobenius = (0..3)
        .map(|i| (estimate.col(i) - truth.col(i)).length_squared())
        .sum::<f64>()
        .sqrt();
    let chord = frobenius / (2.0 * std::f64::consts::SQRT_2);
    (2.0 * chord.clamp(0.0, 1.0).asin()).to_degrees()
}

/// Distance between an estimated translation and the truth.
pub(crate) fn translation_error(estimate: impl Into<Vec3F64>, truth: Vec3F64) -> f64 {
    (estimate.into() - truth).length()
}
