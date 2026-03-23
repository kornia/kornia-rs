use kornia_algebra::Vec3F64;
use serde::{Deserialize, Serialize};

/// Fisheye camera using the Kannala-Brandt equidistant projection model.
///
/// Projects a 3D point (x, y, z) to pixel (u, v) using:
///   r = sqrt(x² + y²)
///   theta = atan2(r, z)
///   theta_d = theta + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
///   u = fx * theta_d * (x/r) + cx
///   v = fy * theta_d * (y/r) + cy
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FisheyeCamera {
    /// Focal length in x
    pub fx: f64,
    /// Focal length in y
    pub fy: f64,
    /// Principal point x
    pub cx: f64,
    /// Principal point y
    pub cy: f64,
    /// First radial distortion coefficient
    pub k1: f64,
    /// Second radial distortion coefficient
    pub k2: f64,
    /// Third radial distortion coefficient
    pub k3: f64,
    /// Fourth radial distortion coefficient
    pub k4: f64,
}

impl FisheyeCamera {
    /// Project a 3D point to pixel coordinates.
    ///
    /// # Arguments
    ///
    /// * `point` - 3D point in camera coordinates.
    ///
    /// # Returns
    ///
    /// Returns `Some([u, v])` if the point is projected successfully.
    /// Returns `None` if the point is at the optical center or directly behind the camera axis.
    pub fn project(&self, point: &Vec3F64) -> Option<[f64; 2]> {
        let r = (point.x * point.x + point.y * point.y).sqrt();

        // A fisheye camera can have a FOV > 180°, so we do not universally reject z < 0.
        // We only reject the optical center or points near directly behind the camera axis.
        if r < 1e-8 && point.z <= 0.0 {
            return None;
        }

        let theta = r.atan2(point.z);
        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let theta_d =
            theta + self.k1 * theta3 + self.k2 * theta5 + self.k3 * theta7 + self.k4 * theta9;

        let scale = if r > 1e-8 {
            theta_d / r
        } else {
            0.0 // If r is zero (and z > 0), x and y are zero, so multiplying by scale does not change u, v from cx, cy
        };

        let u = self.fx * point.x * scale + self.cx;
        let v = self.fy * point.y * scale + self.cy;

        Some([u, v])
    }

    /// Unproject a pixel to a unit bearing ray in camera frame.
    ///
    /// Requires solving theta_d → theta via Newton's method.
    ///
    /// # Arguments
    ///
    /// * `pixel` - Pixel coordinates `[u, v]`.
    ///
    /// # Returns
    ///
    /// Returns the unit 3D bearing ray `Vec3F64` corresponding to the pixel.
    pub fn unproject(&self, pixel: [f64; 2]) -> Vec3F64 {
        let nx = (pixel[0] - self.cx) / self.fx;
        let ny = (pixel[1] - self.cy) / self.fy;
        let theta_d = (nx * nx + ny * ny).sqrt();

        let mut theta = theta_d; // Initial guess
        if theta_d > 1e-8 {
            for _ in 0..10 {
                let theta2 = theta * theta;
                let theta4 = theta2 * theta2;
                let theta6 = theta4 * theta2;
                let theta8 = theta4 * theta4;

                let f = theta + self.k1 * theta * theta2 + self.k2 * theta * theta4
                    + self.k3 * theta * theta6
                    + self.k4 * theta * theta8
                    - theta_d;
                let df = 1.0 + 3.0 * self.k1 * theta2 + 5.0 * self.k2 * theta4
                    + 7.0 * self.k3 * theta6
                    + 9.0 * self.k4 * theta8;

                let step = f / df;
                theta -= step;

                if step.abs() < 1e-8 {
                    break;
                }
            }
        } else {
            theta = 0.0;
        }

        let scale = if theta_d > 1e-8 {
            theta.sin() / theta_d
        } else {
            1.0 // If theta_d approaches 0, sin(theta)/theta_d approaches 1
        };

        let x = nx * scale;
        let y = ny * scale;
        let z = theta.cos();

        Vec3F64::new(x, y, z)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fisheye_roundtrip() {
        let cam = FisheyeCamera {
            fx: 500.0,
            fy: 500.0,
            cx: 640.0,
            cy: 480.0,
            k1: 0.1,
            k2: 0.01,
            k3: 0.001,
            k4: 0.0001,
        };

        // Test with several points
        let points = vec![
            Vec3F64::new(1.0, 2.0, 5.0),
            Vec3F64::new(-1.0, 3.0, 2.0),
            Vec3F64::new(0.5, -0.5, 10.0),
            Vec3F64::new(0.0, 0.0, 5.0), // Optical axis
        ];

        for pt in points {
            let pixel = cam.project(&pt).expect("Project failed");
            let ray = cam.unproject(pixel);

            let unit_pt = pt.normalize();
            assert_relative_eq!(ray.x, unit_pt.x, epsilon = 1e-6);
            assert_relative_eq!(ray.y, unit_pt.y, epsilon = 1e-6);
            assert_relative_eq!(ray.z, unit_pt.z, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_fisheye_optical_center() {
        let cam = FisheyeCamera {
            fx: 500.0,
            fy: 500.0,
            cx: 640.0,
            cy: 480.0,
            k1: 0.1,
            k2: 0.01,
            k3: 0.001,
            k4: 0.0001,
        };

        let pt = Vec3F64::new(0.0, 0.0, 5.0);
        let pixel = cam.project(&pt).unwrap();
        assert_relative_eq!(pixel[0], cam.cx);
        assert_relative_eq!(pixel[1], cam.cy);
    }

    #[test]
    fn test_fisheye_behind_camera() {
        let cam = FisheyeCamera {
            fx: 500.0,
            fy: 500.0,
            cx: 640.0,
            cy: 480.0,
            k1: 0.1,
            k2: 0.01,
            k3: 0.001,
            k4: 0.0001,
        };

        // Fisheye can project points behind camera as long as r is not zero.
        let pt = Vec3F64::new(1.0, 1.0, -1.0);
        let pixel = cam.project(&pt);
        assert!(pixel.is_some());

        // Point directly at optical center should return None
        let pt_center = Vec3F64::new(0.0, 0.0, 0.0);
        assert!(cam.project(&pt_center).is_none());
    }
}
