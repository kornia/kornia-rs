//! Pinhole camera model with Brown-Conrady distortion.

use crate::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_image::ImageSize;

use crate::camera::{CameraModel, ProjectionReject};

/// Pinhole camera with Brown-Conrady radial-tangential distortion.
#[derive(Debug, Clone)]
pub struct PinholeCamera {
    /// Focal length in x (pixels).
    pub fx: f64,
    /// Focal length in y (pixels).
    pub fy: f64,
    /// Principal point x (pixels).
    pub cx: f64,
    /// Principal point y (pixels).
    pub cy: f64,
    /// Radial distortion coefficient k1.
    pub k1: f64,
    /// Radial distortion coefficient k2.
    pub k2: f64,
    /// Tangential distortion coefficient p1.
    pub p1: f64,
    /// Tangential distortion coefficient p2.
    pub p2: f64,
}

impl CameraModel for PinholeCamera {
    fn intrinsics(&self) -> (f64, f64, f64, f64) {
        (self.fx, self.fy, self.cx, self.cy)
    }

    fn project(&self, p_cam: &Vec3F64) -> Option<Vec2F64> {
        self.project_to_pixel(p_cam, 1e-8)
    }

    fn project_to_image(
        &self,
        p_cam: &Vec3F64,
        image_size: ImageSize,
    ) -> Result<Vec2F64, ProjectionReject> {
        self.project_to_image_with_depth(p_cam, 1e-8, image_size)
    }

    fn unproject(&self, pixel: &Vec2F64) -> Option<Vec3F64> {
        let undistorted = self.undistort(pixel.x, pixel.y);
        Some(Vec3F64::new(undistorted.x, undistorted.y, 1.0))
    }
}

impl PinholeCamera {
    /// Undistorts a pixel using iterative Brown-Conrady inversion.
    ///
    /// Returns the undistorted pixel `(u, v)` in the ideal pinhole image plane.
    /// Convergence is typically reached in 5 iterations for typical lens distortions.
    pub fn undistort(&self, col: f64, row: f64) -> Vec2F64 {
        let xd = (col - self.cx) / self.fx;
        let yd = (row - self.cy) / self.fy;
        let mut x = xd;
        let mut y = yd;
        for _ in 0..5 {
            let r2 = x * x + y * y;
            let rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
            let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
            let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
            x = (xd - dx) / rad;
            y = (yd - dy) / rad;
        }
        Vec2F64::new(self.fx * x + self.cx, self.fy * y + self.cy)
    }

    /// Projects a camera-frame 3D point to pixel coordinates.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinates.
    /// * `min_depth` - Points with `z <= min_depth` are rejected.
    ///
    /// # Returns
    ///
    /// `Some(pixel)` on success, or `None` when `p_cam.z <= min_depth`.
    pub fn project_to_pixel(&self, p_cam: &Vec3F64, min_depth: f64) -> Option<Vec2F64> {
        if p_cam.z <= min_depth {
            return None;
        }
        let u = self.fx * p_cam.x / p_cam.z + self.cx;
        let v = self.fy * p_cam.y / p_cam.z + self.cy;
        Some(Vec2F64::new(u, v))
    }

    /// Projects a camera-frame 3D point and checks image bounds.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinates.
    /// * `min_depth` - Points with `z <= min_depth` are rejected.
    /// * `image_size` - Image dimensions for bounds checking.
    ///
    /// # Returns
    ///
    /// `Ok(pixel)` on success.
    ///
    /// # Errors
    ///
    /// Returns [`ProjectionReject::BelowMinDepth`] if `p_cam.z <= min_depth`.
    /// Returns [`ProjectionReject::OutOfImage`] if the pixel falls outside `image_size`.
    pub fn project_to_image_with_depth(
        &self,
        p_cam: &Vec3F64,
        min_depth: f64,
        image_size: ImageSize,
    ) -> Result<Vec2F64, ProjectionReject> {
        let pixel = self
            .project_to_pixel(p_cam, min_depth)
            .ok_or(ProjectionReject::BelowMinDepth)?;
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
    /// Returns `None` when projection is invalid (for example behind camera).
    pub fn reprojection_error_sq_cam(
        &self,
        p_cam: &Vec3F64,
        u_meas: f64,
        v_meas: f64,
    ) -> Option<f64> {
        let projected = self.project_to_pixel(p_cam, 1e-8)?;
        let du = projected.x - u_meas;
        let dv = projected.y - v_meas;
        Some(du * du + dv * dv)
    }

    /// Computes squared reprojection error for a world-frame 3D point.
    ///
    /// Internally transforms `p_world` into camera frame using `pose_world_to_cam`
    /// and then projects using the current pinhole model.
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

    /// Builds the 3x3 intrinsic matrix K.
    pub fn intrinsic_matrix(&self) -> Mat3F64 {
        Mat3F64::from_cols(
            Vec3F64::new(self.fx, 0.0, 0.0),
            Vec3F64::new(0.0, self.fy, 0.0),
            Vec3F64::new(self.cx, self.cy, 1.0),
        )
    }

    /// Undistort matched keypoint pairs, skipping out-of-bounds indices.
    ///
    /// Given two sets of `[col, row]` keypoints and a list of index pairs
    /// `(idx_in_kps1, idx_in_kps2)`, returns the undistorted points for both views.
    pub fn undistort_matched_pairs(
        &self,
        kps1: &[[f32; 2]],
        kps2: &[[f32; 2]],
        matches: &[(usize, usize)],
    ) -> (Vec<Vec2F64>, Vec<Vec2F64>) {
        let mut pts1 = Vec::with_capacity(matches.len());
        let mut pts2 = Vec::with_capacity(matches.len());
        for &(i1, i2) in matches {
            if i1 >= kps1.len() || i2 >= kps2.len() {
                continue;
            }
            let p1 = kps1[i1];
            let p2 = kps2[i2];
            pts1.push(self.undistort(p1[0] as f64, p1[1] as f64));
            pts2.push(self.undistort(p2[0] as f64, p2[1] as f64));
        }
        (pts1, pts2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Mat3F64;

    fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 200.0,
            fy: 200.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    #[test]
    fn test_project_to_pixel() {
        let cam = camera();
        let p = Vec3F64::new(0.0, 0.0, 5.0);
        let uv = cam.project_to_pixel(&p, 1e-8).unwrap();
        assert!((uv.x - 320.0).abs() < 1e-12);
        assert!((uv.y - 240.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_to_image() {
        let cam = camera();
        let p = Vec3F64::new(0.0, 0.0, 5.0);
        let uv = cam
            .project_to_image(
                &p,
                ImageSize {
                    width: 640,
                    height: 480,
                },
            )
            .unwrap();
        assert!((uv.x - 320.0).abs() < 1e-12);
        assert!((uv.y - 240.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_to_image_out_of_image() {
        let cam = camera();
        let p = Vec3F64::new(10.0, 0.0, 1.0);
        let err = cam
            .project_to_image(
                &p,
                ImageSize {
                    width: 640,
                    height: 480,
                },
            )
            .unwrap_err();
        assert_eq!(err, ProjectionReject::OutOfImage);
    }

    #[test]
    fn test_project_to_image_below_min_depth() {
        let cam = camera();
        let p = Vec3F64::new(0.0, 0.0, 1e-9);
        let err = cam
            .project_to_image(
                &p,
                ImageSize {
                    width: 640,
                    height: 480,
                },
            )
            .unwrap_err();
        assert_eq!(err, ProjectionReject::BelowMinDepth);
    }

    #[test]
    fn test_reprojection_error_sq_cam_zero() {
        let cam = camera();
        let p = Vec3F64::new(0.0, 0.0, 5.0);
        let err = cam.reprojection_error_sq_cam(&p, 320.0, 240.0).unwrap();
        assert!(err < 1e-12);
    }

    #[test]
    fn test_reprojection_error_sq_world_zero_identity() {
        let cam = camera();
        let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);
        let p_world = Vec3F64::new(0.0, 0.0, 5.0);
        let err = cam
            .reprojection_error_sq_world(&pose, &p_world, 320.0, 240.0)
            .unwrap();
        assert!(err < 1e-12);
    }
}
