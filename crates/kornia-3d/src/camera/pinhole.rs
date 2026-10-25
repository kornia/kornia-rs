//! Pinhole camera model with Brown-Conrady distortion.

use super::CameraModel;
use crate::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_image::ImageSize;

/// Projection rejection reasons shared by all camera models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionReject {
    /// Point depth is below or equal to the requested minimum.
    BelowMinDepth,
    /// Point is invalid (e.g. at the optical center) or cannot be projected.
    InvalidProjection,
    /// Projection lies outside image bounds.
    OutOfImage,
}

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

/// Project a world point through a pose and pinhole camera.
///
/// Returns `(u, v, z_cam)` or `None` if the point is behind the camera.
pub fn project_point(
    camera: &PinholeCamera,
    pose: &Pose3d,
    point_world: &Vec3F64,
) -> Option<(f64, f64, f64)> {
    let p_cam = pose.transform_point(point_world);
    if p_cam.z <= 1e-8 {
        return None;
    }
    let u = camera.fx * p_cam.x / p_cam.z + camera.cx;
    let v = camera.fy * p_cam.y / p_cam.z + camera.cy;
    Some((u, v, p_cam.z))
}

impl PinholeCamera {
    /// The identity pinhole: `fx = fy = 1`, `cx = cy = 0`, no distortion.
    ///
    /// A point projected through this camera stays in normalized image coordinates (`x/z`, `y/z`).
    /// Useful as the camera model for optimizers/triangulators that work in normalized coordinates
    /// after each observation has been pre-normalized with [`PinholeCamera::normalize`].
    pub const IDENTITY: Self = Self {
        fx: 1.0,
        fy: 1.0,
        cx: 0.0,
        cy: 0.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    };

    /// Map a pixel to **normalized image coordinates** — the point `(x, y)` on the `z = 1` plane
    /// whose ideal-pinhole projection is this pixel (Hartley & Zisserman's `x̂ = K⁻¹·x`; the output
    /// of OpenCV's `undistortPoints`).
    ///
    /// Removes distortion via [`PinholeCamera::undistort`], then applies `K⁻¹`:
    /// `((u_ideal − cx) / fx, (v_ideal − cy) / fy)`; with zero distortion just
    /// `((u − cx) / fx, (v − cy) / fy)`. It is the inverse of the ideal-pinhole projection, so a
    /// camera-frame point `p` satisfies `normalize(project(p)) == (p.x/p.z, p.y/p.z)`.
    ///
    /// Note: this is the projective normalization onto the `z = 1` plane, **not** an L2 (unit-length)
    /// vector normalization — the result is a 2D image-plane point, not a unit bearing ray.
    pub fn normalize(&self, uv: Vec2F64) -> Vec2F64 {
        let up = self.undistort(uv.x, uv.y);
        Vec2F64::new((up.x - self.cx) / self.fx, (up.y - self.cy) / self.fy)
    }

    /// Undistorts a pixel by inverting Brown-Conrady distortion with Newton iteration.
    ///
    /// # Arguments
    ///
    /// * `col` - Column of the pixel in the raw image.
    /// * `row` - Row of the pixel in the raw image.
    ///
    /// # Returns
    ///
    /// The pixel `(u, v)` in the ideal pinhole image. For a pixel the distortion
    /// cannot produce (past the fold of the radial map) this is the inversion's
    /// best estimate, not a true preimage; [`CameraModel::unproject`] returns
    /// `None` there instead.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn undistort(&self, col: f64, row: f64) -> Vec2F64 {
        let (n, _) =
            self.undistort_normalized((col - self.cx) / self.fx, (row - self.cy) / self.fy);
        Vec2F64::new(self.fx * n.x + self.cx, self.fy * n.y + self.cy)
    }

    /// Inverts [`PinholeCamera::distort_normalized`] by damped Newton iteration.
    ///
    /// Returns the last iterate and whether it is the true preimage: converged,
    /// inside the fold radius, and with a positive Jacobian determinant. A root
    /// past the fold has as small a residual as the true one but lies on the
    /// wrong side of the optical axis, so convergence alone does not suffice.
    ///
    /// Without distortion the first residual is exactly zero, so the input is
    /// returned unchanged.
    fn undistort_normalized(&self, xd: f64, yd: f64) -> (Vec2F64, bool) {
        let target = Vec2F64::new(xd, yd);
        let mut p = target;
        let mut r = self.distort_normalized(p.x, p.y) - target;
        let mut cost = r.dot(r);
        for _ in 0..UNDISTORT_MAX_ITER {
            if cost < UNDISTORT_TOL_SQ {
                break;
            }
            let [[a, b], [c, d]] = self.distortion_jacobian(p.x, p.y);
            let det = a * d - b * c;
            if det.abs() < UNDISTORT_MIN_DET {
                return (p, false);
            }
            let step = Vec2F64::new(d * r.x - b * r.y, a * r.y - c * r.x) / det;
            // Halve the step until the residual shrinks: near a flat stretch of
            // the radial map a full Newton step overshoots the root.
            let mut scale = 1.0;
            loop {
                let candidate = p - step * scale;
                let r_candidate = self.distort_normalized(candidate.x, candidate.y) - target;
                let cost_candidate = r_candidate.dot(r_candidate);
                if cost_candidate < cost {
                    (p, r, cost) = (candidate, r_candidate, cost_candidate);
                    break;
                }
                scale *= 0.5;
                if scale < UNDISTORT_MIN_STEP {
                    return (p, false);
                }
            }
        }
        let [[a, b], [c, d]] = self.distortion_jacobian(p.x, p.y);
        let valid = cost < UNDISTORT_TOL_SQ
            && p.dot(p) < self.fold_radius_sq()
            && a * d - b * c > UNDISTORT_MIN_DET;
        (p, valid)
    }

    /// Squared normalized radius at which the radial map `r ↦ r·(1 + k1·r² + k2·r⁴)`
    /// stops increasing and folds back over the image: the first positive root
    /// of `1 + 3·k1·s + 5·k2·s²` in `s = r²`, or infinity when it has none.
    fn fold_radius_sq(&self) -> f64 {
        let (a, b) = (5.0 * self.k2, 3.0 * self.k1);
        if a == 0.0 {
            return if b < 0.0 { -1.0 / b } else { f64::INFINITY };
        }
        let disc = b * b - 4.0 * a;
        if disc < 0.0 {
            return f64::INFINITY;
        }
        // Roots of a·s² + b·s + 1, written to avoid cancellation; their product is 1/a.
        let q = -0.5 * (b + b.signum() * disc.sqrt());
        [q / a, 1.0 / q]
            .into_iter()
            .filter(|s| *s > 0.0)
            .fold(f64::INFINITY, f64::min)
    }

    /// Applies Brown-Conrady distortion to a point on the `z = 1` plane.
    fn distort_normalized(&self, x: f64, y: f64) -> Vec2F64 {
        let r2 = x * x + y * y;
        let rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
        Vec2F64::new(
            x * rad + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x),
            y * rad + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y,
        )
    }

    /// Rows of `∂ distort_normalized / ∂(x, y)`.
    fn distortion_jacobian(&self, x: f64, y: f64) -> [[f64; 2]; 2] {
        let r2 = x * x + y * y;
        let rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
        let drad_dr2 = self.k1 + 2.0 * self.k2 * r2;
        let cross = 2.0 * drad_dr2 * x * y + 2.0 * self.p1 * x + 2.0 * self.p2 * y;
        [
            [
                rad + 2.0 * drad_dr2 * x * x + 2.0 * self.p1 * y + 6.0 * self.p2 * x,
                cross,
            ],
            [
                cross,
                rad + 2.0 * drad_dr2 * y * y + 6.0 * self.p1 * y + 2.0 * self.p2 * x,
            ],
        ]
    }

    /// Projects a camera-frame 3D point through the ideal pinhole, ignoring
    /// distortion; [`CameraModel::project`] applies it.
    ///
    /// Returns `None` when `p_cam.z <= min_depth`.
    pub fn project_to_pixel(&self, p_cam: &Vec3F64, min_depth: f64) -> Option<Vec2F64> {
        if p_cam.z <= min_depth {
            return None;
        }
        let u = self.fx * p_cam.x / p_cam.z + self.cx;
        let v = self.fy * p_cam.y / p_cam.z + self.cy;
        Some(Vec2F64::new(u, v))
    }

    /// Projects a camera-frame 3D point and checks image bounds.
    pub fn project_to_image(
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

    /// Returns `(fx, fy, cx, cy)`.
    pub fn intrinsics(&self) -> (f64, f64, f64, f64) {
        (self.fx, self.fy, self.cx, self.cy)
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

/// Points closer than this to the `z = 0` plane are not projected.
const MIN_DEPTH: f64 = 1e-8;

/// Newton iteration cap for [`PinholeCamera::undistort_normalized`].
const UNDISTORT_MAX_ITER: usize = 20;
/// Squared normalized-plane residual at which undistortion has converged.
const UNDISTORT_TOL_SQ: f64 = 1e-24;
/// Below this Jacobian determinant the distortion is not invertible.
const UNDISTORT_MIN_DET: f64 = 1e-12;
/// Smallest fraction of a Newton step tried before undistortion gives up.
const UNDISTORT_MIN_STEP: f64 = 1e-6;

impl CameraModel for PinholeCamera {
    fn project(&self, p_cam: &Vec3F64) -> Option<Vec2F64> {
        if p_cam.z <= MIN_DEPTH {
            return None;
        }
        let d = self.distort_normalized(p_cam.x / p_cam.z, p_cam.y / p_cam.z);
        Some(Vec2F64::new(
            self.fx * d.x + self.cx,
            self.fy * d.y + self.cy,
        ))
    }

    fn project_with_jacobian(&self, p_cam: &Vec3F64) -> Option<(Vec2F64, [Vec3F64; 2])> {
        if p_cam.z <= MIN_DEPTH {
            return None;
        }
        let inv_z = 1.0 / p_cam.z;
        let (x, y) = (p_cam.x * inv_z, p_cam.y * inv_z);
        let d = self.distort_normalized(x, y);
        let [[dxd_dx, dxd_dy], [dyd_dx, dyd_dy]] = self.distortion_jacobian(x, y);
        // ∂(x, y)/∂p_cam = [1/z, 0, -x/z; 0, 1/z, -y/z].
        let row = |a: f64, b: f64, f: f64| {
            Vec3F64::new(f * a * inv_z, f * b * inv_z, -f * (a * x + b * y) * inv_z)
        };
        Some((
            Vec2F64::new(self.fx * d.x + self.cx, self.fy * d.y + self.cy),
            [row(dxd_dx, dxd_dy, self.fx), row(dyd_dx, dyd_dy, self.fy)],
        ))
    }

    fn unproject(&self, pixel: &Vec2F64) -> Option<Vec3F64> {
        let (n, valid) =
            self.undistort_normalized((pixel.x - self.cx) / self.fx, (pixel.y - self.cy) / self.fy);
        valid.then(|| Vec3F64::new(n.x, n.y, 1.0).normalize())
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
                1e-8,
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
    fn test_project_to_image_accepts_kornia_image_size() {
        let cam = camera();
        let p = Vec3F64::new(0.0, 0.0, 5.0);
        let uv = cam
            .project_to_image(
                &p,
                1e-8,
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
                1e-8,
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
        let p = Vec3F64::new(0.0, 0.0, 0.5);
        let err = cam
            .project_to_image(
                &p,
                1.0,
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

    #[test]
    fn test_identity_and_normalize_roundtrip() {
        // IDENTITY leaves normalized coords untouched.
        let id = PinholeCamera::IDENTITY;
        assert_eq!(id.fx, 1.0);
        let n = id.normalize(Vec2F64::new(0.3, -0.2));
        assert!((n.x - 0.3).abs() < 1e-12 && (n.y + 0.2).abs() < 1e-12);
        // normalize is the inverse of the (distortion-free) pinhole projection.
        let cam = PinholeCamera {
            fx: 500.0,
            fy: 480.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let (xn, yn) = (0.12, -0.05);
        let u = cam.fx * xn + cam.cx;
        let v = cam.fy * yn + cam.cy;
        let back = cam.normalize(Vec2F64::new(u, v));
        assert!((back.x - xn).abs() < 1e-9 && (back.y - yn).abs() < 1e-9);
    }

    #[test]
    fn test_unproject_converges_on_a_flat_radial_map() {
        // Monotonic but nearly flat around r = 0.8: plain Newton overshoots and
        // stalls far from the unique preimage r ≈ 1.30926.
        let cam = PinholeCamera {
            fx: 500.0,
            fy: 500.0,
            cx: 0.0,
            cy: 0.0,
            k1: -1.0,
            k2: 0.451,
            p1: 0.0,
            p2: 0.0,
        };
        let pixel = Vec2F64::new(400.0, 0.0);
        let bearing = cam.unproject(&pixel).unwrap();
        assert!((bearing.x / bearing.z - 1.30926).abs() < 1e-5);
        assert!((cam.project(&bearing).unwrap() - pixel).length() < 1e-6);
    }

    #[test]
    fn test_unproject_rejects_pixels_past_the_fold() {
        // With k1 = -0.28 the distorted radius peaks at about 364 px, so the
        // corners of a 752×480 image have no preimage. Undamped Newton finds a
        // root there anyway, on the far side of the optical axis.
        let cam = PinholeCamera {
            fx: 500.0,
            fy: 500.0,
            cx: 376.0,
            cy: 240.0,
            k1: -0.28,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        assert!(cam.unproject(&Vec2F64::new(0.0, 0.0)).is_none());
        assert!(cam.unproject(&Vec2F64::new(751.0, 479.0)).is_none());

        let inside = Vec2F64::new(376.0 + 360.0, 240.0);
        let bearing = cam.unproject(&inside).unwrap();
        assert!(bearing.x > 0.0);
        assert!((cam.project(&bearing).unwrap() - inside).length() < 1e-6);
    }

    #[test]
    fn test_unproject_rejects_a_converged_root_past_the_fold() {
        // Damped Newton converges here (residual below 1e-12) to the normalized
        // point (1.12, 0.89), on the far side of the axis from the pixel and
        // past the fold at r ≈ 0.62; the pixel itself has no preimage.
        let cam = PinholeCamera {
            fx: 500.0,
            fy: 500.0,
            cx: 0.0,
            cy: 0.0,
            k1: -0.8,
            k2: -0.1,
            p1: 0.0,
            p2: 0.0,
        };
        assert!(cam.unproject(&Vec2F64::new(-595.0, -475.0)).is_none());
    }

    #[test]
    fn test_fold_radius_sq() {
        let cam = |k1, k2| PinholeCamera { k1, k2, ..camera() };
        assert_eq!(cam(0.0, 0.0).fold_radius_sq(), f64::INFINITY);
        assert_eq!(cam(0.1, 0.0).fold_radius_sq(), f64::INFINITY);
        assert!((cam(-0.28, 0.0).fold_radius_sq() - 1.0 / 0.84).abs() < 1e-12);
        // EuRoC cam0: 1 − 0.85·s + 0.37·s² has no real root.
        assert_eq!(cam(-0.28340811, 0.07395907).fold_radius_sq(), f64::INFINITY);
        // 1 − 3·s + 2·s² = (1 − s)(1 − 2s): the fold is the smaller root.
        let s = cam(-1.0, 0.4).fold_radius_sq();
        assert!((s - 0.5).abs() < 1e-12, "{s}");
    }
}
