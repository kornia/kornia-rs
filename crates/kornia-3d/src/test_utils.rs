//! Seeded synthetic scenes shared by unit tests and benchmarks.
//!
//! Available outside unit tests only with the `test-utils` feature.

use crate::camera::PinholeCamera;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec2F64, Vec3AF32, Vec3F64, SO3F64};
use rand::{rngs::StdRng, RngExt};

/// Maximum sampling attempts per requested correspondence.
const MAX_POINT_SAMPLING_ATTEMPTS: usize = 128;
/// Minimum accepted camera depth as a fraction of the configured distance.
const MIN_DEPTH_FRACTION: f64 = 0.1;

/// Errors from generating synthetic scene fixtures.
#[derive(Debug, thiserror::Error)]
pub enum SceneError {
    /// A fixture parameter is outside its supported range.
    #[error("Invalid scene parameter {name}: {reason}")]
    InvalidParameter {
        /// Parameter that failed validation.
        name: &'static str,
        /// Required range or constraint.
        reason: &'static str,
    },
    /// The sampling budget could not produce enough visible points.
    #[error("Generated {generated} of {requested} visible points in {attempts} attempts")]
    SamplingExhausted {
        /// Number of correspondences requested.
        requested: usize,
        /// Number of visible correspondences generated.
        generated: usize,
        /// Sampling attempts made.
        attempts: usize,
    },
    /// A generated observation cannot be represented as finite f32 coordinates.
    #[error("Generated coordinates are not finite in f32")]
    NonFiniteObservation,
    /// Storage for the requested correspondences could not be reserved.
    #[error("Cannot reserve scene storage: {0}")]
    Allocation(#[from] std::collections::TryReserveError),
}

fn validate_angle(max_angle: f64) -> Result<(), SceneError> {
    if !max_angle.is_finite() || !(0.0..=std::f64::consts::PI).contains(&max_angle) {
        return Err(SceneError::InvalidParameter {
            name: "max_angle",
            reason: "must be finite and in [0, pi] radians",
        });
    }
    Ok(())
}

/// A standard normal sample (Box–Muller); `rand` alone has no normal distribution.
fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generate a rotation about a uniformly random axis.
///
/// # Arguments
///
/// * `rng` - Seeded random number generator.
/// * `max_angle` - Maximum angle in `[0, π]` radians; zero returns the identity.
///
/// # Returns
///
/// A rotation whose angle is sampled uniformly from `[0, max_angle)`, or the identity at zero.
///
/// # Errors
///
/// Returns [`SceneError::InvalidParameter`] for a non-finite or out-of-range angle.
pub fn random_rotation(rng: &mut StdRng, max_angle: f64) -> Result<Mat3F64, SceneError> {
    validate_angle(max_angle)?;
    if max_angle == 0.0 {
        return Ok(Mat3F64::IDENTITY);
    }
    let axis = Vec3F64::new(gauss(rng), gauss(rng), gauss(rng)).normalize();
    Ok(SO3F64::exp(axis * rng.random_range(0.0..max_angle)).matrix())
}

/// Construct a pinhole camera matrix without skew.
///
/// # Arguments
///
/// * `fx` - Horizontal focal length in pixels.
/// * `fy` - Vertical focal length in pixels.
/// * `cx` - Horizontal principal point in pixels.
/// * `cy` - Vertical principal point in pixels.
///
/// # Returns
///
/// The camera matrix in column-major storage.
///
/// # Errors
///
/// This fixture does not validate its inputs or return errors.
pub fn k_matrix(fx: f32, fy: f32, cx: f32, cy: f32) -> Mat3AF32 {
    Mat3AF32::from_cols_array(&[fx, 0.0, 0.0, 0.0, fy, 0.0, cx, cy, 1.0])
}

/// Construct the fixture camera: `f = 800` px, principal point `(640, 480)`.
///
/// # Arguments
///
/// None.
///
/// # Returns
///
/// The camera matrix used by synthetic scenes.
///
/// # Errors
///
/// This fixture does not return errors.
pub fn k_default() -> Mat3AF32 {
    k_matrix(800.0, 800.0, 640.0, 480.0)
}

/// The shape of a random scene.
pub struct SceneConfig {
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
    fn validate(&self) -> Result<usize, SceneError> {
        for (name, valid, reason) in [
            (
                "half_size",
                self.half_size.is_finite()
                    && self.half_size > 0.0
                    && self.half_size <= f64::from(f32::MAX),
                "must be positive and at most f32::MAX",
            ),
            (
                "depth",
                self.depth.is_finite() && self.depth > 0.0 && self.depth <= f64::from(f32::MAX),
                "must be positive and at most f32::MAX",
            ),
            (
                "noise_px",
                self.noise_px.is_finite() && self.noise_px >= 0.0,
                "must be finite and non-negative",
            ),
        ] {
            if !valid {
                return Err(SceneError::InvalidParameter { name, reason });
            }
        }
        validate_angle(self.max_angle)?;
        self.n
            .checked_mul(MAX_POINT_SAMPLING_ATTEMPTS)
            .ok_or(SceneError::InvalidParameter {
                name: "n",
                reason: "sampling budget overflows usize",
            })
    }

    /// Configure points in a 2 m cube, viewed at a random rotation.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of correspondences.
    /// * `depth` - Positive camera distance in metres.
    /// * `noise_px` - Standard deviation of Gaussian pixel noise.
    ///
    /// # Returns
    ///
    /// A scene configuration with a 1 m half-size and rotation angles below π.
    ///
    /// # Errors
    ///
    /// This fixture does not validate its inputs or return errors.
    pub fn cube(n: usize, depth: f64, noise_px: f64) -> Self {
        Self {
            n,
            noise_px,
            half_size: 1.0,
            depth,
            planar: false,
            max_angle: std::f64::consts::PI,
        }
    }

    /// Restrict the configured scene to the plane `z = 0`.
    ///
    /// # Arguments
    ///
    /// * `self` - Configuration to modify.
    ///
    /// # Returns
    ///
    /// The configuration with planar sampling enabled.
    ///
    /// # Errors
    ///
    /// This fixture does not return errors.
    pub fn planar(self) -> Self {
        Self {
            planar: true,
            ..self
        }
    }
}

/// A camera, a ground-truth world → camera pose and the correspondences they generate.
pub struct Scene {
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
    /// Generate a random scene, rejecting points closer than a tenth of the camera distance.
    ///
    /// # Arguments
    ///
    /// * `rng` - Seeded random number generator.
    /// * `k` - Finite pinhole camera matrix with positive focal lengths.
    /// * `config` - Shape, distance, noise and correspondence count. The half-size, depth
    ///   and camera distance must be positive and at most f32::MAX; noise must be finite
    ///   and non-negative, and the maximum angle must be in `[0, π]` radians.
    ///
    /// # Returns
    ///
    /// Correspondences and their ground-truth world-to-camera pose.
    ///
    /// # Errors
    ///
    /// Returns [`SceneError::InvalidParameter`] for invalid configuration or camera parameters,
    /// [`SceneError::SamplingExhausted`] if the finite sampling budget is exhausted,
    /// [`SceneError::NonFiniteObservation`] for coordinates outside finite f32 range, or
    /// [`SceneError::Allocation`] if correspondence storage cannot be reserved.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_3d::test_utils::{k_default, Scene, SceneConfig};
    /// use rand::{rngs::StdRng, SeedableRng};
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let scene = Scene::generate(&mut rng, k_default(), &SceneConfig::cube(20, 6.0, 1.0))?;
    /// assert_eq!(scene.world.len(), scene.image.len());
    /// # Ok::<(), kornia_3d::test_utils::SceneError>(())
    /// ```
    pub fn generate(
        rng: &mut StdRng,
        k: Mat3AF32,
        config: &SceneConfig,
    ) -> Result<Self, SceneError> {
        let max_attempts = config.validate()?;
        Self::generate_with_budget(rng, k, config, max_attempts)
    }

    fn generate_with_budget(
        rng: &mut StdRng,
        k: Mat3AF32,
        config: &SceneConfig,
        max_attempts: usize,
    ) -> Result<Self, SceneError> {
        if !k.is_finite()
            || k.x_axis.x <= 0.0
            || k.y_axis.y <= 0.0
            || k.x_axis.y != 0.0
            || k.x_axis.z != 0.0
            || k.y_axis.x != 0.0
            || k.y_axis.z != 0.0
            || k.z_axis.z != 1.0
        {
            return Err(SceneError::InvalidParameter {
                name: "k",
                reason: "must be a finite pinhole camera matrix with positive focal lengths",
            });
        }
        let camera = pinhole(k);
        let rotation = random_rotation(rng, config.max_angle)?;
        let translation = Vec3F64::new(
            rng.random_range(-0.1..0.1) * config.depth,
            rng.random_range(-0.1..0.1) * config.depth,
            config.depth,
        );
        let mut world = Vec::new();
        let mut image = Vec::new();
        world.try_reserve_exact(config.n)?;
        image.try_reserve_exact(config.n)?;
        let mut attempts = 0;
        while world.len() < config.n {
            if attempts == max_attempts {
                return Err(SceneError::SamplingExhausted {
                    requested: config.n,
                    generated: world.len(),
                    attempts,
                });
            }
            attempts += 1;
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
            let Some(uv) = camera.project_to_pixel(
                &(rotation * p + translation),
                MIN_DEPTH_FRACTION * config.depth,
            ) else {
                continue;
            };
            let point = p.as_vec3af32();
            let pixel = with_noise(rng, uv, config.noise_px);
            if !point.to_array().into_iter().all(f32::is_finite)
                || !pixel.to_array().into_iter().all(f32::is_finite)
            {
                return Err(SceneError::NonFiniteObservation);
            }
            world.push(point);
            image.push(pixel);
        }
        Ok(Self {
            k,
            rotation,
            translation,
            world,
            image,
        })
    }

    /// Append wrong correspondences drawn independently in world and image coordinates.
    ///
    /// # Arguments
    ///
    /// * `self` - Scene with a finite, positive principal point.
    /// * `rng` - Seeded random number generator.
    /// * `count` - Number of outliers to append.
    ///
    /// # Returns
    ///
    /// Updates both correspondence arrays in place, with outliers last. World points lie
    /// in `[-1, 1]³`; image dimensions are twice the principal point.
    ///
    /// # Errors
    ///
    /// Returns [`SceneError::InvalidParameter`] if twice either principal point coordinate is
    /// non-positive or non-finite, or the correspondence arrays differ in length.
    /// Returns [`SceneError::Allocation`] if storage cannot be reserved.
    /// Validation completes before either correspondence array is modified.
    pub fn add_outliers(&mut self, rng: &mut StdRng, count: usize) -> Result<(), SceneError> {
        let (width, height) = (2.0 * self.k.z_axis().x, 2.0 * self.k.z_axis().y);
        if !width.is_finite() || width <= 0.0 || !height.is_finite() || height <= 0.0 {
            return Err(SceneError::InvalidParameter {
                name: "principal_point",
                reason: "twice each coordinate must be positive and finite",
            });
        }
        if self.world.len() != self.image.len() {
            return Err(SceneError::InvalidParameter {
                name: "correspondences",
                reason: "world and image arrays must have the same length",
            });
        }
        self.world.try_reserve(count)?;
        self.image.try_reserve(count)?;
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
        Ok(())
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
///
/// # Arguments
///
/// * `estimate` - Estimated rotation.
/// * `truth` - Ground-truth rotation.
///
/// # Returns
///
/// Angular error in degrees, in `[0, 180]` for finite rotations.
///
/// # Errors
///
/// This fixture does not return errors; non-finite inputs yield NaN.
pub fn rotation_error_deg(estimate: impl Into<Mat3F64>, truth: Mat3F64) -> f64 {
    let estimate = estimate.into();
    let frobenius = (0..3)
        .map(|i| (estimate.col(i) - truth.col(i)).length_squared())
        .sum::<f64>()
        .sqrt();
    let chord = frobenius / (2.0 * std::f64::consts::SQRT_2);
    (2.0 * chord.clamp(0.0, 1.0).asin()).to_degrees()
}

/// Compute the distance between an estimated translation and the truth.
///
/// # Arguments
///
/// * `estimate` - Estimated translation.
/// * `truth` - Ground-truth translation, in the same units.
///
/// # Returns
///
/// Euclidean distance in the input units.
///
/// # Errors
///
/// This fixture does not return errors; non-finite inputs may yield NaN or infinity.
pub fn translation_error(estimate: impl Into<Vec3F64>, truth: Vec3F64) -> f64 {
    (estimate.into() - truth).length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn invalid_configuration_does_not_advance_rng() {
        for (name, config) in [
            (
                "half_size",
                SceneConfig {
                    half_size: 0.0,
                    ..SceneConfig::cube(4, 6.0, 0.0)
                },
            ),
            (
                "half_size",
                SceneConfig {
                    half_size: f64::MAX,
                    ..SceneConfig::cube(4, 6.0, 0.0)
                },
            ),
            ("depth", SceneConfig::cube(4, f64::NAN, 0.0)),
            ("depth", SceneConfig::cube(4, 0.0, 0.0)),
            ("noise_px", SceneConfig::cube(4, 6.0, -1.0)),
            ("noise_px", SceneConfig::cube(4, 6.0, f64::INFINITY)),
            (
                "max_angle",
                SceneConfig {
                    max_angle: f64::INFINITY,
                    ..SceneConfig::cube(4, 6.0, 0.0)
                },
            ),
            ("n", SceneConfig::cube(usize::MAX, 6.0, 0.0)),
        ] {
            let mut rng = StdRng::seed_from_u64(42);
            let mut unchanged = StdRng::seed_from_u64(42);
            assert!(matches!(Scene::generate(&mut rng, k_default(), &config),
                Err(SceneError::InvalidParameter { name: actual, .. }) if actual == name));
            assert_eq!(rng.random::<u64>(), unchanged.random::<u64>());
        }
    }

    #[test]
    fn zero_angle_generates_an_identity_pose() -> Result<(), SceneError> {
        let mut rng = StdRng::seed_from_u64(42);
        assert_eq!(random_rotation(&mut rng, 0.0)?, Mat3F64::IDENTITY);
        let config = SceneConfig {
            max_angle: 0.0,
            ..SceneConfig::cube(4, 6.0, 0.0)
        };
        let scene = Scene::generate(&mut rng, k_default(), &config)?;
        assert_eq!(scene.rotation, Mat3F64::IDENTITY);
        assert_eq!(scene.world.len(), 4);
        assert_eq!(scene.image.len(), 4);
        for invalid in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(matches!(
                random_rotation(&mut rng, invalid),
                Err(SceneError::InvalidParameter {
                    name: "max_angle",
                    ..
                })
            ));
        }
        Ok(())
    }

    #[test]
    fn sampling_stops_at_the_attempt_budget() {
        let mut rng = StdRng::seed_from_u64(42);
        let config = SceneConfig::cube(1, 6.0, 0.0);
        assert!(matches!(
            Scene::generate_with_budget(&mut rng, k_default(), &config, 0),
            Err(SceneError::SamplingExhausted {
                requested: 1,
                generated: 0,
                attempts: 0
            })
        ));
    }

    #[test]
    fn generated_pixels_must_fit_f32() {
        let mut rng = StdRng::seed_from_u64(42);
        assert!(matches!(
            Scene::generate(&mut rng, k_default(), &SceneConfig::cube(1, 6.0, f64::MAX)),
            Err(SceneError::NonFiniteObservation)
        ));
    }

    #[test]
    fn outlier_validation_and_reservation_preserve_correspondences() -> Result<(), SceneError> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut scene = Scene::generate(&mut rng, k_default(), &SceneConfig::cube(4, 6.0, 0.0))?;
        let world = scene.world.clone();
        let image = scene.image.clone();
        for invalid in [0.0, -1.0, f32::NAN, f32::MAX] {
            scene.k.z_axis.x = invalid;
            assert!(matches!(
                scene.add_outliers(&mut rng, 1),
                Err(SceneError::InvalidParameter {
                    name: "principal_point",
                    ..
                })
            ));
            assert_eq!(scene.world, world);
            assert_eq!(scene.image, image);
        }
        scene.k = k_default();
        assert!(matches!(
            scene.add_outliers(&mut rng, usize::MAX),
            Err(SceneError::Allocation(_))
        ));
        assert_eq!(scene.world, world);
        assert_eq!(scene.image, image);
        scene.image.truncate(3);
        assert!(matches!(
            scene.add_outliers(&mut rng, 1),
            Err(SceneError::InvalidParameter {
                name: "correspondences",
                ..
            })
        ));
        assert_eq!(scene.world, world);
        assert_eq!(scene.image, image[..3]);
        Ok(())
    }
}
