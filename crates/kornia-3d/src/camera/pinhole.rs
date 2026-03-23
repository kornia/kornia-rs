use kornia_algebra::Vec3F64;
use serde::{Deserialize, Serialize};

/// Pinhole camera model for perspective projection.
///
/// Projects a 3D point (x, y, z) to pixel (u, v) using:
///   u = fx * (x / z) + cx
///   v = fy * (y / z) + cy
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PinholeCamera {
    /// Focal length in x
    pub fx: f64,
    /// Focal length in y
    pub fy: f64,
    /// Principal point x
    pub cx: f64,
    /// Principal point y
    pub cy: f64,
}

impl PinholeCamera {
    /// Project a 3D point to pixel coordinates.
    ///
    /// # Arguments
    ///
    /// * `point` - 3D point in camera coordinates.
    ///
    /// # Returns
    ///
    /// Returns `Some([u, v])` if the point is in front of the camera (z > 0).
    /// Returns `None` if the point is behind or at the optical center (z <= 0).
    pub fn project(&self, point: &Vec3F64) -> Option<[f64; 2]> {
        if point.z <= 0.0 {
            return None;
        }

        let u = self.fx * (point.x / point.z) + self.cx;
        let v = self.fy * (point.y / point.z) + self.cy;

        Some([u, v])
    }

    /// Unproject a pixel to a unit bearing ray in camera frame.
    ///
    /// # Arguments
    ///
    /// * `pixel` - Pixel coordinates `[u, v]`.
    ///
    /// # Returns
    ///
    /// Returns the unit 3D bearing ray `Vec3F64` corresponding to the pixel.
    pub fn unproject(&self, pixel: [f64; 2]) -> Vec3F64 {
        let x = (pixel[0] - self.cx) / self.fx;
        let y = (pixel[1] - self.cy) / self.fy;
        let z = 1.0;

        Vec3F64::new(x, y, z).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinhole_roundtrip() {
        let cam = PinholeCamera {
            fx: 500.0,
            fy: 500.0,
            cx: 640.0,
            cy: 480.0,
        };

        let pt = Vec3F64::new(1.0, 2.0, 5.0);
        let pixel = cam.project(&pt).unwrap();
        let ray = cam.unproject(pixel);

        let unit_pt = pt.normalize();
        assert!((ray.x - unit_pt.x).abs() < 1e-6);
        assert!((ray.y - unit_pt.y).abs() < 1e-6);
        assert!((ray.z - unit_pt.z).abs() < 1e-6);
    }
}
