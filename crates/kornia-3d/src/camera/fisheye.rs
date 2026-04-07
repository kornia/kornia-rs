//! Kannala-Brandt equidistant fisheye projection model.

use crate::camera::{CameraModel, ProjectionReject};
use crate::pose::Pose3d;
use kornia_algebra::{Vec2F64, Vec3F64};

/// Kannala-Brandt equidistant fisheye projection model.
///
/// Projects a 3D point (x, y, z) to pixel (u, v) using:
///   r = sqrt(x² + y²)
///   theta = atan2(r, z)
///   theta_d = theta + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
///   u = fx * theta_d * (x/r) + cx
///   v = fy * theta_d * (y/r) + cy
#[derive(Debug, Clone)]
pub struct FisheyeCamera {
    /// Focal length in x (pixels).
    pub fx: f64,
    /// Focal length in y (pixels).
    pub fy: f64,
    /// Principal point x (pixels).
    pub cx: f64,
    /// Principal point y (pixels).
    pub cy: f64,
    /// Distortion coefficient k1.
    pub k1: f64,
    /// Distortion coefficient k2.
    pub k2: f64,
    /// Distortion coefficient k3.
    pub k3: f64,
    /// Distortion coefficient k4.
    pub k4: f64,
}

/// Maximum number of Newton-Raphson iterations for unproject.
const NEWTON_MAX_ITER: usize = 10;

/// Convergence tolerance for Newton-Raphson in unproject.
const NEWTON_EPS: f64 = 1e-12;

/// Epsilon threshold for zero-checks to avoid singularities.
const EPSILON: f64 = 1e-12;

impl FisheyeCamera {
    /// Evaluates the distortion polynomial theta_d and its derivative f_prime with respect to theta.
    /// Returns `(theta_d, f_prime)`.
    fn distortion(&self, theta: f64) -> (f64, f64) {
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta4 * theta4;
        let theta_d = theta
            + self.k1 * theta2 * theta
            + self.k2 * theta4 * theta
            + self.k3 * theta6 * theta
            + self.k4 * theta8 * theta;
        let f_prime = 1.0
            + 3.0 * self.k1 * theta2
            + 5.0 * self.k2 * theta4
            + 7.0 * self.k3 * theta6
            + 9.0 * self.k4 * theta8;
        (theta_d, f_prime)
    }

    /// Project a 3D camera-frame point to pixel coordinates.
    ///
    /// Returns `Some((pixel, z))` where `z` is the depth in camera frame.
    /// Returns `None` if the point cannot be projected (e.g., exactly at optical center).
    ///
    /// The projection supports points behind the camera (`z < 0`) as long as
    /// the distortion model is valid (standard for fisheye lenses with >180° FoV).
    pub fn project_with_depth(&self, point: &Vec3F64) -> Option<(Vec2F64, f64)> {
        let x = point.x;
        let y = point.y;
        let z = point.z;

        let r = (x * x + y * y).sqrt();
        // Point at the optical center — undefined projection.
        if r < EPSILON && z.abs() < EPSILON {
            return None;
        }
        // Points on the negative optical axis have undefined azimuth.
        if r < EPSILON && z < 0.0 {
            return None;
        }

        let theta = r.atan2(z);
        let (theta_d, _) = self.distortion(theta);

        // On the optical axis: r ≈ 0, theta ≈ 0, theta_d ≈ 0 → principal point.
        if r < EPSILON {
            return Some((Vec2F64::new(self.cx, self.cy), z));
        }

        let scale = theta_d / r;
        let u = self.fx * scale * x + self.cx;
        let v = self.fy * scale * y + self.cy;

        Some((Vec2F64::new(u, v), z))
    }

    /// Projects a camera-frame 3D point and checks image bounds.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinates.
    /// * `image_size` - Image dimensions for bounds checking.
    ///
    /// # Returns
    ///
    /// `Ok(pixel)` on success.
    ///
    /// # Errors
    ///
    /// Returns [`ProjectionReject::InvalidProjection`] if the point cannot be projected.
    /// Returns [`ProjectionReject::OutOfImage`] if the pixel falls outside `image_size`.
    pub fn project_to_image_bounds(
        &self,
        p_cam: &Vec3F64,
        image_size: kornia_image::ImageSize,
    ) -> Result<Vec2F64, ProjectionReject> {
        let (pixel, _) = self
            .project_with_depth(p_cam)
            .ok_or(ProjectionReject::InvalidProjection)?;

        if pixel.x < 0.0
            || pixel.y < 0.0
            || pixel.x >= image_size.width as f64
            || pixel.y >= image_size.height as f64
        {
            return Err(ProjectionReject::OutOfImage);
        }
        Ok(pixel)
    }

    /// Computes squared reprojection error for a camera-frame 3D point.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinates.
    /// * `u_meas` - Measured pixel x coordinate.
    /// * `v_meas` - Measured pixel y coordinate.
    ///
    /// # Returns
    ///
    /// `Some(error_sq)` on success, or `None` when projection is invalid (optical center).
    pub fn reprojection_error_sq_cam(
        &self,
        p_cam: &Vec3F64,
        u_meas: f64,
        v_meas: f64,
    ) -> Option<f64> {
        let (projected, _) = self.project_with_depth(p_cam)?;
        let du = projected.x - u_meas;
        let dv = projected.y - v_meas;
        Some(du * du + dv * dv)
    }

    /// Computes squared reprojection error for a world-frame 3D point.
    ///
    /// Internally transforms `p_world` into camera frame using `pose_world_to_cam`
    /// and then projects using the fisheye model.
    ///
    /// # Arguments
    ///
    /// * `pose_world_to_cam` - World-to-camera rigid body transform.
    /// * `p_world` - 3D point in world coordinates.
    /// * `u_meas` - Measured pixel x coordinate.
    /// * `v_meas` - Measured pixel y coordinate.
    ///
    /// # Returns
    ///
    /// `Some(error_sq)` on success, or `None` when projection is invalid.
    pub fn reprojection_error_sq_world(
        &self,
        pose_world_to_cam: &Pose3d,
        p_world: &Vec3F64,
        u_meas: f64,
        v_meas: f64,
    ) -> Option<f64> {
        let p_cam = pose_world_to_cam.transform_point(p_world);
        self.reprojection_error_sq_cam(&p_cam, u_meas, v_meas)
    }

    /// Unproject a pixel to a unit bearing ray in the camera frame.
    ///
    /// Inverts the distortion model by solving
    /// `theta_d = theta + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹`
    /// for `theta` using Newton-Raphson iteration, then constructs the 3D
    /// unit ray `(sin(theta)·cos(phi), sin(theta)·sin(phi), cos(theta))`.
    pub fn unproject(&self, pixel: &Vec2F64) -> Vec3F64 {
        let mx = (pixel.x - self.cx) / self.fx;
        let my = (pixel.y - self.cy) / self.fy;
        let theta_d = (mx * mx + my * my).sqrt();

        // Pixel is at the principal point → on-axis ray.
        if theta_d < EPSILON {
            return Vec3F64::new(0.0, 0.0, 1.0);
        }

        // Newton-Raphson: solve f(theta) = theta + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹ - theta_d = 0
        let mut theta = theta_d;
        for _ in 0..NEWTON_MAX_ITER {
            let (theta_d_calc, f_prime) = self.distortion(theta);
            if f_prime.abs() < EPSILON {
                break;
            }
            let f = theta_d_calc - theta_d;
            let delta = f / f_prime;
            theta -= delta;
            if delta.abs() < NEWTON_EPS {
                break;
            }
        }

        // Recover the bearing direction from the azimuthal angle phi.
        let phi = my.atan2(mx);
        let sin_theta = theta.sin();

        Vec3F64::new(sin_theta * phi.cos(), sin_theta * phi.sin(), theta.cos())
    }
}

impl CameraModel for FisheyeCamera {
    fn intrinsics(&self) -> (f64, f64, f64, f64) {
        (self.fx, self.fy, self.cx, self.cy)
    }

    fn project(&self, p_cam: &Vec3F64) -> Option<Vec2F64> {
        self.project_with_depth(p_cam).map(|(pixel, _z)| pixel)
    }

    fn project_to_image(
        &self,
        p_cam: &Vec3F64,
        image_size: kornia_image::ImageSize,
    ) -> Result<Vec2F64, ProjectionReject> {
        self.project_to_image_bounds(p_cam, image_size)
    }

    fn unproject(&self, pixel: &Vec2F64) -> Option<Vec3F64> {
        Some(FisheyeCamera::unproject(self, pixel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::project_point;
    use kornia_algebra::Mat3F64;

    /// Fisheye camera with zero distortion (pure equidistant model).
    fn fisheye_camera_zero_distortion() -> FisheyeCamera {
        FisheyeCamera {
            fx: 200.0,
            fy: 200.0,
            cx: 736.0,
            cy: 720.0,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            k4: 0.0,
        }
    }

    /// Fisheye camera with typical distortion coefficients.
    fn fisheye_camera_with_distortion() -> FisheyeCamera {
        FisheyeCamera {
            fx: 200.0,
            fy: 200.0,
            cx: 736.0,
            cy: 720.0,
            k1: 0.05,
            k2: -0.01,
            k3: 0.002,
            k4: -0.0003,
        }
    }

    #[test]
    fn test_fisheye_project_on_axis() {
        let cam = fisheye_camera_zero_distortion();
        // Point on the optical axis → should project to principal point.
        let p = Vec3F64::new(0.0, 0.0, 5.0);
        let (uv, z) = cam.project_with_depth(&p).unwrap();
        assert!((uv.x - cam.cx).abs() < 1e-12);
        assert!((uv.y - cam.cy).abs() < 1e-12);
        assert!((z - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_fisheye_project_optical_center_returns_none() {
        let cam = fisheye_camera_zero_distortion();
        // Point at the optical center → undefined, returns None.
        let p = Vec3F64::new(0.0, 0.0, 0.0);
        assert!(cam.project_with_depth(&p).is_none());
    }

    #[test]
    fn test_fisheye_project_negative_z_axis_returns_none() {
        let cam = fisheye_camera_zero_distortion();
        // Point on the negative optical axis → undefined azimuth, returns None.
        let p = Vec3F64::new(0.0, 0.0, -5.0);
        assert!(cam.project_with_depth(&p).is_none());
    }

    #[test]
    fn test_fisheye_project_zero_distortion_45_deg() {
        let cam = fisheye_camera_zero_distortion();
        // Point at 45° in the x-z plane: theta = pi/4.
        // With zero distortion: theta_d = theta = pi/4.
        // r = 1.0, x = 1.0, y = 0.0 → scale = theta_d / r = pi/4.
        // u = fx * (pi/4) * 1.0 + cx
        let p = Vec3F64::new(1.0, 0.0, 1.0);
        let (uv, z) = cam.project_with_depth(&p).unwrap();
        let expected_u = cam.fx * std::f64::consts::FRAC_PI_4 + cam.cx;
        assert!((uv.x - expected_u).abs() < 1e-10);
        assert!((uv.y - cam.cy).abs() < 1e-10);
        assert!((z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_fisheye_unproject_principal_point() {
        let cam = fisheye_camera_zero_distortion();
        // Principal point → on-axis bearing ray (0, 0, 1).
        let ray = cam.unproject(&Vec2F64::new(cam.cx, cam.cy));
        assert!((ray.x).abs() < 1e-12);
        assert!((ray.y).abs() < 1e-12);
        assert!((ray.z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_fisheye_roundtrip_zero_distortion() {
        let cam = fisheye_camera_zero_distortion();
        // Test multiple 3D points at various angles.
        let points = [
            Vec3F64::new(1.0, 0.0, 1.0),  // 45° in x-z
            Vec3F64::new(0.0, 1.0, 1.0),  // 45° in y-z
            Vec3F64::new(1.0, 1.0, 1.0),  // off-axis
            Vec3F64::new(0.5, -0.3, 2.0), // moderate angle
            Vec3F64::new(0.0, 0.0, 3.0),  // on-axis
        ];
        for pt in &points {
            let (pixel, _) = cam.project_with_depth(pt).unwrap();
            let ray = cam.unproject(&pixel);
            // Recover the original direction (normalize the input point).
            let len = (pt.x * pt.x + pt.y * pt.y + pt.z * pt.z).sqrt();
            let dir = Vec3F64::new(pt.x / len, pt.y / len, pt.z / len);
            assert!(
                (ray.x - dir.x).abs() < 1e-6
                    && (ray.y - dir.y).abs() < 1e-6
                    && (ray.z - dir.z).abs() < 1e-6,
                "Round-trip failed for point ({}, {}, {}): got ({}, {}, {}), expected ({}, {}, {})",
                pt.x,
                pt.y,
                pt.z,
                ray.x,
                ray.y,
                ray.z,
                dir.x,
                dir.y,
                dir.z,
            );
        }
    }

    #[test]
    fn test_fisheye_roundtrip_with_distortion() {
        let cam = fisheye_camera_with_distortion();
        let points = [
            Vec3F64::new(1.0, 0.0, 1.0),
            Vec3F64::new(0.0, 1.0, 1.0),
            Vec3F64::new(1.0, 1.0, 2.0),
            Vec3F64::new(-0.5, 0.3, 1.5),
            Vec3F64::new(0.1, -0.1, 3.0),
        ];
        for pt in &points {
            let (pixel, _) = cam.project_with_depth(pt).unwrap();
            let ray = cam.unproject(&pixel);
            let len = (pt.x * pt.x + pt.y * pt.y + pt.z * pt.z).sqrt();
            let dir = Vec3F64::new(pt.x / len, pt.y / len, pt.z / len);
            assert!(
                (ray.x - dir.x).abs() < 1e-6
                    && (ray.y - dir.y).abs() < 1e-6
                    && (ray.z - dir.z).abs() < 1e-6,
                "Round-trip failed for point ({}, {}, {}): got ({}, {}, {}), expected ({}, {}, {})",
                pt.x,
                pt.y,
                pt.z,
                ray.x,
                ray.y,
                ray.z,
                dir.x,
                dir.y,
                dir.z,
            );
        }
    }

    #[test]
    fn test_fisheye_project_near_optical_axis() {
        let cam = fisheye_camera_with_distortion();
        // Very small r, positive z → should be near principal point and stable.
        let p = Vec3F64::new(1e-10, 1e-10, 5.0);
        let (uv, _) = cam.project_with_depth(&p).unwrap();
        assert!((uv.x - cam.cx).abs() < 1.0);
        assert!((uv.y - cam.cy).abs() < 1.0);
    }

    #[test]
    fn test_fisheye_roundtrip_wide_angle() {
        let cam = fisheye_camera_zero_distortion();
        // Point at ~80° from the optical axis.
        let p = Vec3F64::new(5.0, 0.0, 1.0);
        let (pixel, _) = cam.project_with_depth(&p).unwrap();
        let ray = cam.unproject(&pixel);
        let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
        let dir = Vec3F64::new(p.x / len, p.y / len, p.z / len);
        assert!((ray.x - dir.x).abs() < 1e-6);
        assert!((ray.y - dir.y).abs() < 1e-6);
        assert!((ray.z - dir.z).abs() < 1e-6);
    }

    #[test]
    fn test_fisheye_project_behind_camera() {
        let cam = fisheye_camera_zero_distortion();
        // Point behind camera: z = -1.0, r = 1.0. theta = 3*pi/4.
        let p = Vec3F64::new(1.0, 0.0, -1.0);
        let (uv, z) = cam.project_with_depth(&p).unwrap();
        let expected_u = cam.fx * (3.0 * std::f64::consts::FRAC_PI_4) + cam.cx;
        assert!((uv.x - expected_u).abs() < 1e-10);
        assert!((z + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_point_fisheye() {
        let cam = fisheye_camera_zero_distortion();
        let pose = crate::pose::Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);
        let p_world = Vec3F64::new(1.0, 0.0, 1.0);
        let (u, v, z) = project_point(&cam, &pose, &p_world).unwrap();
        let expected_u = cam.fx * std::f64::consts::FRAC_PI_4 + cam.cx;
        assert!((u - expected_u).abs() < 1e-10);
        assert!((v - cam.cy).abs() < 1e-10);
        assert!((z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_reprojection_error_sq_fisheye() {
        let cam = fisheye_camera_zero_distortion();
        let p = Vec3F64::new(1.0, 0.0, 1.0);
        let (uv, _) = cam.project_with_depth(&p).unwrap();
        let err = cam.reprojection_error_sq_cam(&p, uv.x, uv.y).unwrap();
        assert!(err < 1e-12);

        let err_offset = cam.reprojection_error_sq_cam(&p, uv.x + 1.0, uv.y).unwrap();
        assert!((err_offset - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_fisheye_project_to_image() {
        let cam = fisheye_camera_zero_distortion();
        let size = kornia_image::ImageSize {
            width: 800,
            height: 800,
        };

        let p_in = Vec3F64::new(0.0, 0.0, 5.0);
        assert!(cam.project_to_image(&p_in, size).is_ok());

        let p_out = Vec3F64::new(10.0, 10.0, 0.5); // Very wide, outside 640x480 bounds
        assert_eq!(
            cam.project_to_image(&p_out, size).unwrap_err(),
            ProjectionReject::OutOfImage
        );
    }
}
