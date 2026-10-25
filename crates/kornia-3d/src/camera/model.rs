//! The interface shared by all camera models.

use super::{FisheyeCamera, PinholeCamera};
use kornia_algebra::{Vec2F64, Vec3F64};

/// Maps between camera-frame points and pixels of the raw, distorted image.
///
/// The camera frame is x right, y down, z forward; pixels are `(column, row)`.
pub trait CameraModel {
    /// Projects a camera-frame point to a pixel, applying the model's distortion.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - Point in the camera frame.
    ///
    /// # Returns
    ///
    /// The pixel, or `None` when the model cannot image the point: for a
    /// pinhole, `z <= 1e-8`; for a fisheye, a point on the optical axis at or
    /// behind the camera.
    fn project(&self, p_cam: &Vec3F64) -> Option<Vec2F64>;

    /// Like [`CameraModel::project`], also returning the 2×3 Jacobian.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - Point in the camera frame.
    ///
    /// # Returns
    ///
    /// The pixel and the Jacobian rows `[∂u/∂p_cam, ∂v/∂p_cam]`, in pixels per
    /// camera-frame unit; `None` exactly when [`CameraModel::project`] is.
    fn project_with_jacobian(&self, p_cam: &Vec3F64) -> Option<(Vec2F64, [Vec3F64; 2])>;

    /// Unprojects a pixel to the ray it images.
    ///
    /// # Arguments
    ///
    /// * `pixel` - Pixel of the raw image.
    ///
    /// # Returns
    ///
    /// The unit bearing of the ray, in the camera frame, or `None` when no ray
    /// in the model's valid range images the pixel: it lies past the fold, where
    /// the distortion turns back on itself, or the inversion did not converge.
    fn unproject(&self, pixel: &Vec2F64) -> Option<Vec3F64>;
}

/// A camera whose model is chosen at runtime.
#[derive(Debug, Clone)]
pub enum Camera {
    /// Pinhole with Brown-Conrady distortion.
    Pinhole(PinholeCamera),
    /// Kannala-Brandt fisheye.
    Fisheye(FisheyeCamera),
}

impl From<PinholeCamera> for Camera {
    fn from(camera: PinholeCamera) -> Self {
        Self::Pinhole(camera)
    }
}

impl From<FisheyeCamera> for Camera {
    fn from(camera: FisheyeCamera) -> Self {
        Self::Fisheye(camera)
    }
}

impl CameraModel for Camera {
    fn project(&self, p_cam: &Vec3F64) -> Option<Vec2F64> {
        match self {
            Self::Pinhole(camera) => camera.project(p_cam),
            Self::Fisheye(camera) => camera.project(p_cam),
        }
    }

    fn project_with_jacobian(&self, p_cam: &Vec3F64) -> Option<(Vec2F64, [Vec3F64; 2])> {
        match self {
            Self::Pinhole(camera) => camera.project_with_jacobian(p_cam),
            Self::Fisheye(camera) => camera.project_with_jacobian(p_cam),
        }
    }

    fn unproject(&self, pixel: &Vec2F64) -> Option<Vec3F64> {
        match self {
            Self::Pinhole(camera) => camera.unproject(pixel),
            Self::Fisheye(camera) => camera.unproject(pixel),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EuRoC `cam0`.
    fn radtan() -> PinholeCamera {
        PinholeCamera {
            fx: 458.654,
            fy: 457.296,
            cx: 367.215,
            cy: 248.375,
            k1: -0.28340811,
            k2: 0.07395907,
            p1: 0.00019359,
            p2: 1.76187114e-05,
        }
    }

    /// Hilti 2026 `cam0`.
    fn kannala_brandt() -> FisheyeCamera {
        FisheyeCamera {
            fx: 461.64,
            fy: 459.72,
            cx: 732.95,
            cy: 720.54,
            k1: 0.0344,
            k2: -0.0216,
            k3: 0.0031,
            k4: -0.0005,
        }
    }

    fn assert_jacobian_matches_finite_differences(camera: &impl CameraModel, p: Vec3F64) {
        let (pixel, rows) = camera.project_with_jacobian(&p).unwrap();
        assert_eq!(pixel, camera.project(&p).unwrap());

        let h = 1e-6;
        let axes = [
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        ];
        for (col, axis) in axes.into_iter().enumerate() {
            let plus = camera.project(&(p + axis * h)).unwrap();
            let minus = camera.project(&(p - axis * h)).unwrap();
            let numeric = (plus - minus) / (2.0 * h);
            let analytic = Vec2F64::new(rows[0].to_array()[col], rows[1].to_array()[col]);
            let tolerance = 1e-5 * analytic.length().max(1.0);
            assert!(
                (numeric - analytic).length() < tolerance,
                "∂/∂p[{col}] at {p:?}: analytic {analytic:?}, numeric {numeric:?}"
            );
        }
    }

    fn assert_round_trips(camera: &impl CameraModel, p: Vec3F64, tolerance: f64) {
        let bearing = camera.unproject(&camera.project(&p).unwrap()).unwrap();
        assert!((bearing.length() - 1.0).abs() < 1e-12);
        assert!(
            (bearing - p.normalize()).length() < tolerance,
            "{p:?} came back as {bearing:?}"
        );
    }

    #[test]
    fn pinhole_jacobian_matches_finite_differences() {
        for p in [
            Vec3F64::new(0.0, 0.0, 2.0),
            Vec3F64::new(0.4, -0.3, 1.5),
            Vec3F64::new(-0.6, 0.35, 1.0),
        ] {
            assert_jacobian_matches_finite_differences(&radtan(), p);
        }
    }

    #[test]
    fn fisheye_jacobian_matches_finite_differences() {
        for p in [
            Vec3F64::new(0.0, 0.0, 2.0),
            Vec3F64::new(1e-9, -1e-9, 2.0),
            Vec3F64::new(0.4, -0.3, 1.5),
            Vec3F64::new(3.0, 1.0, 0.5),
            Vec3F64::new(1.0, 0.5, -0.2),
        ] {
            assert_jacobian_matches_finite_differences(&kannala_brandt(), p);
        }
    }

    #[test]
    fn pinhole_unproject_inverts_project() {
        for p in [
            Vec3F64::new(0.0, 0.0, 2.0),
            Vec3F64::new(0.2, -0.15, 1.5),
            Vec3F64::new(-0.3, 0.2, 1.0),
        ] {
            assert_round_trips(&radtan(), p, 1e-9);
        }
    }

    #[test]
    fn pinhole_unproject_round_trips_image_corners() {
        let camera = radtan();
        for (u, v) in [(0.0, 0.0), (751.0, 0.0), (0.0, 479.0), (751.0, 479.0)] {
            let pixel = Vec2F64::new(u, v);
            let back = camera.project(&camera.unproject(&pixel).unwrap()).unwrap();
            assert!(
                (back - pixel).length() < 1e-6,
                "{pixel:?} came back as {back:?}"
            );
        }
    }

    #[test]
    fn fisheye_unproject_inverts_project_beyond_ninety_degrees() {
        for p in [
            Vec3F64::new(0.4, -0.3, 1.5),
            Vec3F64::new(3.0, 1.0, 0.5),
            Vec3F64::new(1.0, 0.5, -0.2),
        ] {
            assert_round_trips(&kannala_brandt(), p, 1e-9);
        }
    }

    #[test]
    fn pinhole_without_distortion_projects_like_the_ideal_pinhole() {
        let camera = PinholeCamera {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            ..radtan()
        };
        let p = Vec3F64::new(0.4, -0.3, 1.5);
        let ideal = camera.project_to_pixel(&p, 0.0).unwrap();
        assert!((CameraModel::project(&camera, &p).unwrap() - ideal).length() < 1e-9);
    }

    #[test]
    fn pinhole_project_rejects_points_behind_the_camera() {
        assert!(CameraModel::project(&radtan(), &Vec3F64::new(0.1, 0.0, -1.0)).is_none());
        assert!(radtan()
            .project_with_jacobian(&Vec3F64::new(0.1, 0.0, 0.0))
            .is_none());
    }

    #[test]
    fn camera_dispatches_to_its_model() {
        let p = Vec3F64::new(0.4, -0.3, 1.5);
        let pixel = Vec2F64::new(500.0, 300.0);
        for (camera, model) in [
            (Camera::from(radtan()), &radtan() as &dyn CameraModel),
            (Camera::from(kannala_brandt()), &kannala_brandt()),
        ] {
            assert_eq!(camera.project(&p), model.project(&p));
            assert_eq!(camera.unproject(&pixel), model.unproject(&pixel));
            assert_eq!(
                camera.project_with_jacobian(&p),
                model.project_with_jacobian(&p)
            );
        }
    }
}
