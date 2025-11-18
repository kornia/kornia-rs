use glam::{Mat3, Vec3};
use kornia_imgproc::calibration::{
    distortion::PolynomialDistortion,
    CameraExtrinsic, CameraIntrinsic,
};

/// Utility function to compute the Euclidean distance between two points.
///
/// # Arguments
///
/// * `a` - A point in 3D space.
/// * `b` - Another point in 3D space.
///
/// # Returns
///
/// The Euclidean distance between the two points.
///
/// Example:
/// ```
/// use kornia_3d::ops::euclidean_distance;
///
/// let a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// let dst = euclidean_distance(&a, &b);
/// ```
pub fn euclidean_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    assert_eq!(a.len(), b.len());
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Project 3D points to 2D image coordinates.
///
/// This function projects 3D world points onto a 2D image plane using camera
/// intrinsic and extrinsic parameters, similar to OpenCV's `projectPoints`.
/// It uses SIMD-accelerated operations via the `glam` crate for performance.
///
/// # Arguments
///
/// * `points_3d` - Array of 3D points in world coordinates
/// * `intrinsic` - Camera intrinsic parameters (focal length, principal point)
/// * `extrinsic` - Camera extrinsic parameters (rotation, translation)
/// * `distortion` - Optional distortion parameters. If `None`, no distortion is applied.
///
/// # Returns
///
/// A vector of 2D image coordinates `[u, v]` for each input 3D point.
/// Points behind the camera (z <= 0) will have NaN coordinates.
///
/// # Example
///
/// ```
/// use kornia_3d::ops::project_points;
/// use kornia_imgproc::calibration::{CameraIntrinsic, CameraExtrinsic};
///
/// let points_3d = vec![[1.0, 2.0, 5.0], [0.0, 0.0, 3.0]];
/// let intrinsic = CameraIntrinsic {
///     fx: 500.0,
///     fy: 500.0,
///     cx: 320.0,
///     cy: 240.0,
/// };
/// let extrinsic = CameraExtrinsic {
///     rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
///     translation: [0.0, 0.0, 0.0],
/// };
///
/// let points_2d = project_points(&points_3d, &intrinsic, &extrinsic, None);
/// assert_eq!(points_2d.len(), 2);
/// ```
pub fn project_points(
    points_3d: &[[f64; 3]],
    intrinsic: &CameraIntrinsic,
    extrinsic: &CameraExtrinsic,
    distortion: Option<&PolynomialDistortion>,
) -> Vec<[f64; 2]> {
    // Convert rotation matrix to glam Mat3
    // OpenCV uses row-major, glam uses column-major
    let r_mat = Mat3::from_cols(
        Vec3::new(
            extrinsic.rotation[0][0] as f32,
            extrinsic.rotation[1][0] as f32,
            extrinsic.rotation[2][0] as f32,
        ),
        Vec3::new(
            extrinsic.rotation[0][1] as f32,
            extrinsic.rotation[1][1] as f32,
            extrinsic.rotation[2][1] as f32,
        ),
        Vec3::new(
            extrinsic.rotation[0][2] as f32,
            extrinsic.rotation[1][2] as f32,
            extrinsic.rotation[2][2] as f32,
        ),
    );

    // Convert translation vector to glam Vec3
    let t_vec = Vec3::new(
        extrinsic.translation[0] as f32,
        extrinsic.translation[1] as f32,
        extrinsic.translation[2] as f32,
    );

    // Extract intrinsic parameters
    let fx = intrinsic.fx;
    let fy = intrinsic.fy;
    let cx = intrinsic.cx;
    let cy = intrinsic.cy;

    // Pre-compute intrinsic vectors for efficient projection
    let intr_x = Vec3::new(fx as f32, 0.0, cx as f32);
    let intr_y = Vec3::new(0.0, fy as f32, cy as f32);

    points_3d
        .iter()
        .map(|&point_3d| {
            // Convert 3D point to glam Vec3
            let pw = Vec3::new(
                point_3d[0] as f32,
                point_3d[1] as f32,
                point_3d[2] as f32,
            );

            // Transform to camera coordinates: pc = R * pw + t
            let pc = r_mat * pw + t_vec;

            // Check if point is behind camera
            if pc.z <= 0.0 {
                return [f64::NAN, f64::NAN];
            }

            // Project to image plane (undistorted)
            let inv_z = 1.0 / pc.z;
            let u_undist = (intr_x.dot(pc) * inv_z) as f64;
            let v_undist = (intr_y.dot(pc) * inv_z) as f64;

            // Apply distortion if provided
            if let Some(dist) = distortion {
                let (u_dist, v_dist) = kornia_imgproc::calibration::distortion::distort_point_polynomial(
                    u_undist,
                    v_undist,
                    intrinsic,
                    dist,
                );
                [u_dist, v_dist]
            } else {
                [u_undist, v_undist]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use kornia_imgproc::calibration::distortion::PolynomialDistortion;

    #[test]
    fn test_euclidean_distance() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_relative_eq!(euclidean_distance(&a, &b), 5.196152, epsilon = 1e-6);
    }

    #[test]
    fn test_project_points_no_distortion() {
        let points_3d = vec![[0.0, 0.0, 5.0], [1.0, 2.0, 10.0]];
        let intrinsic = CameraIntrinsic {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
        };
        let extrinsic = CameraExtrinsic {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        };

        let points_2d = project_points(&points_3d, &intrinsic, &extrinsic, None);
        assert_eq!(points_2d.len(), 2);

        // First point: [0, 0, 5] should project to [cx, cy] = [320, 240]
        assert_relative_eq!(points_2d[0][0], 320.0, epsilon = 1e-5);
        assert_relative_eq!(points_2d[0][1], 240.0, epsilon = 1e-5);

        // Second point: [1, 2, 10] should project to [fx*1/10 + cx, fy*2/10 + cy]
        assert_relative_eq!(points_2d[1][0], 500.0 * 1.0 / 10.0 + 320.0, epsilon = 1e-5);
        assert_relative_eq!(points_2d[1][1], 500.0 * 2.0 / 10.0 + 240.0, epsilon = 1e-5);
    }

    #[test]
    fn test_project_points_with_distortion() {
        let points_3d = vec![[0.0, 0.0, 5.0]];
        let intrinsic = CameraIntrinsic {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
        };
        let extrinsic = CameraExtrinsic {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        };
        let distortion = PolynomialDistortion {
            k1: 0.1,
            k2: 0.01,
            k3: 0.001,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: 0.0005,
            p2: 0.0005,
        };

        let points_2d = project_points(&points_3d, &intrinsic, &extrinsic, Some(&distortion));
        assert_eq!(points_2d.len(), 1);
        // With distortion, the point should be slightly different from [320, 240]
        assert!((points_2d[0][0] - 320.0).abs() < 1.0);
        assert!((points_2d[0][1] - 240.0).abs() < 1.0);
    }

    #[test]
    fn test_project_points_behind_camera() {
        let points_3d = vec![[0.0, 0.0, -5.0]]; // Behind camera
        let intrinsic = CameraIntrinsic {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
        };
        let extrinsic = CameraExtrinsic {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        };

        let points_2d = project_points(&points_3d, &intrinsic, &extrinsic, None);
        assert_eq!(points_2d.len(), 1);
        assert!(points_2d[0][0].is_nan());
        assert!(points_2d[0][1].is_nan());
    }

    #[test]
    fn test_project_points_with_rotation() {
        let points_3d = vec![[1.0, 0.0, 5.0]];
        let intrinsic = CameraIntrinsic {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
        };
        // 90 degree rotation around Z axis
        let extrinsic = CameraExtrinsic {
            rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        };

        let points_2d = project_points(&points_3d, &intrinsic, &extrinsic, None);
        assert_eq!(points_2d.len(), 1);
        // After rotation, [1, 0, 5] becomes [0, 1, 5] in camera frame
        // Should project to [cx, fy*1/5 + cy]
        assert_relative_eq!(points_2d[0][0], 320.0, epsilon = 1e-5);
        assert_relative_eq!(points_2d[0][1], 500.0 * 1.0 / 5.0 + 240.0, epsilon = 1e-5);
    }

    #[test]
    fn test_project_points_with_translation() {
        let points_3d = vec![[0.0, 0.0, 5.0]];
        let intrinsic = CameraIntrinsic {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
        };
        let extrinsic = CameraExtrinsic {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [1.0, 2.0, 0.0], // Translate camera
        };

        let points_2d = project_points(&points_3d, &intrinsic, &extrinsic, None);
        assert_eq!(points_2d.len(), 1);
        // After translation, point becomes [1, 2, 5] in camera frame
        assert_relative_eq!(points_2d[0][0], 500.0 * 1.0 / 5.0 + 320.0, epsilon = 1e-5);
        assert_relative_eq!(points_2d[0][1], 500.0 * 2.0 / 5.0 + 240.0, epsilon = 1e-5);
    }
}
