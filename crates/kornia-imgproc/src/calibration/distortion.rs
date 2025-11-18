use super::{CameraExtrinsic, CameraIntrinsic};
use crate::interpolation::grid::meshgrid_from_fn;
use kornia_image::ImageSize;
use kornia_tensor::{allocator::CpuAllocator, CpuTensor2, TensorError};
use std::error::Error;
use std::fmt;

/// Represents a 2D point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2<T> {
    /// The x coordinate
    pub x: T,
    /// The y coordinate
    pub y: T,
}

#[derive(Debug, Clone)]
pub enum DistortionError {
    InvalidInputShape(String),
    InvalidMatrixShape(String),
    InternalError(String),
}

impl fmt::Display for DistortionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DistortionError::InvalidInputShape(msg) => write!(f, "Invalid input shape: {}", msg),
            DistortionError::InvalidMatrixShape(msg) => write!(f, "Invalid matrix shape: {}", msg),
            DistortionError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl Error for DistortionError {}

/// Represents the polynomial distortion parameters of a camera using the Brown-Conrad model.
///
/// This struct encapsulates both radial (k1-k6) and tangential (p1-p2) distortion coefficients.
/// These parameters are used to model lens distortion in camera calibration and image correction.
#[derive(Clone, Debug)]
pub struct PolynomialDistortion {
    /// The first radial distortion coefficient
    pub k1: f64,
    /// The second radial distortion coefficient
    pub k2: f64,
    /// The third radial distortion coefficient
    pub k3: f64,
    /// The fourth radial distortion coefficient
    pub k4: f64,
    /// The fifth radial distortion coefficient
    pub k5: f64,
    /// The sixth radial distortion coefficient
    pub k6: f64,
    /// The first tangential distortion coefficient
    pub p1: f64,
    /// The second tangential distortion coefficient
    pub p2: f64,
}

/// Applies polynomial distortion to a point using the Brown-Conrady model
///
/// This function takes an undistorted point and applies both radial and tangential
/// distortion based on the provided camera intrinsics and distortion parameters.
///
/// # Arguments
///
/// * `point` - The undistorted 2D point.
/// * `intrinsic` - The intrinsic parameters of the camera.
/// * `distortion` - The distortion parameters of the camera.
///
/// # Returns
///
/// The distorted 2D point.
pub fn distort_point_polynomial(
    point: Point2<f64>,
    intrinsic: &CameraIntrinsic,
    distortion: &PolynomialDistortion,
) -> Point2<f64> {
    let (fx, fy, cx, cy) = (intrinsic.fx, intrinsic.fy, intrinsic.cx, intrinsic.cy);
    let (k1, k2, k3, k4, k5, k6, p1, p2) = (
        distortion.k1,
        distortion.k2,
        distortion.k3,
        distortion.k4,
        distortion.k5,
        distortion.k6,
        distortion.p1,
        distortion.p2,
    );

    // normalize the coordinates
    let xn = (point.x - cx) / fx;
    let yn = (point.y - cy) / fy;

    // calculate the radial distance
    let r2 = xn * xn + yn * yn;
    let r4 = r2 * r2;
    let r6 = r4 * r2;

    // radial distortion
    let kr = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6);

    // tangential distortion
    // This specific order of operations must be matched by the inverse function.
    let x_2 = 2.0 * xn;
    let y_2 = 2.0 * yn;
    let xy_2 = x_2 * yn; // This is (2.0 * xn) * yn
    let xd = xn * kr + xy_2 * p1 + p2 * (r2 + x_2 * xn); // (2*xn*yn)*p1 + p2*(r2 + 2*xn^2)
    let yd = yn * kr + p1 * (r2 + y_2 * yn) + xy_2 * p2; // p1*(r2 + 2*yn^2) + (2*xn*yn)*p2

    // denormalize the coordinates
    Point2 {
        x: fx * xd + cx,
        y: fy * yd + cy,
    }
}

/// Computes the ideal point coordinates from observed (distorted) point coordinates.
///
/// This function iteratively finds the inverse of the distortion model for a single point.
///
/// # Arguments
/// * `distorted_point` - The observed (distorted) 2D point.
/// * `intrinsic` - The intrinsic parameters of the camera.
/// * `distortion` - The distortion parameters of the camera.
///
/// # Returns
///
/// The undistorted 2D point in normalized camera coordinates.
fn undistort_single_point_iterative(
    distorted_point: Point2<f64>,
    intrinsic: &CameraIntrinsic,
    distortion: &PolynomialDistortion,
) -> Point2<f64> {
    let (fx, fy, cx, cy) = (intrinsic.fx, intrinsic.fy, intrinsic.cx, intrinsic.cy);
    let (k1, k2, k3, k4, k5, k6, p1, p2) = (
        distortion.k1,
        distortion.k2,
        distortion.k3,
        distortion.k4,
        distortion.k5,
        distortion.k6,
        distortion.p1,
        distortion.p2,
    );

    // Normalize the distorted point coordinates
    let x_distorted_norm = (distorted_point.x - cx) / fx;
    let y_distorted_norm = (distorted_point.y - cy) / fy;

    // Initial guess for the undistorted point is the distorted point itself
    let mut x_undistorted_norm = x_distorted_norm;
    let mut y_undistorted_norm = y_distorted_norm;

    // Iteratively refine the estimate of the undistorted point.
    // 20 iterations are sufficient now that the math is a true inverse.
    for _ in 0..20 {
        let r2 = x_undistorted_norm * x_undistorted_norm + y_undistorted_norm * y_undistorted_norm;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let radial_numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let radial_denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
        let radial_dist = radial_numerator / radial_denominator;

        // **FIX:** Match the exact FP order of operations from distort_point_polynomial
        let x_2_u = 2.0 * x_undistorted_norm;
        let y_2_u = 2.0 * y_undistorted_norm;
        let xy_2_u = x_2_u * y_undistorted_norm; // This is (2.0 * x_u) * y_u

        let d_tan_x =
            xy_2_u * p1 + p2 * (r2 + x_2_u * x_undistorted_norm);
        let d_tan_y =
            p1 * (r2 + y_2_u * y_undistorted_norm) + xy_2_u * p2;

        // Invert the distortion equation to solve for the undistorted point
        x_undistorted_norm = (x_distorted_norm - d_tan_x) / radial_dist;
        y_undistorted_norm = (y_distorted_norm - d_tan_y) / radial_dist;
    }

    Point2 {
        x: x_undistorted_norm,
        y: y_undistorted_norm,
    }
}

/// Computes the ideal point coordinates from observed (distorted) point coordinates.
///
/// This function is analogous to OpenCV's `undistortPoints`. It operates on a set of points.
///
/// # Arguments
/// * `src_points` - A tensor of observed point coordinates, with shape (N, 2).
/// * `intrinsic` - The camera matrix.
/// * `distortion` - The vector of distortion coefficients.
/// * `r_matrix` - Optional rectification transformation (3x3 matrix).
/// * `p_matrix` - Optional new camera matrix (3x3) or new projection matrix (3x4).
///
/// # Returns
///
/// A tensor of ideal point coordinates (N, 2) after undistortion and optional transformations.
pub fn undistort_points_polynomial(
    src_points: &CpuTensor2<f64>,
    intrinsic: &CameraIntrinsic,
    distortion: &PolynomialDistortion,
    r_matrix: Option<&CpuTensor2<f64>>,
    p_matrix: Option<&CpuTensor2<f64>>,
) -> Result<CpuTensor2<f64>, DistortionError> {
    if src_points.shape[1] != 2 {
        return Err(DistortionError::InvalidInputShape(
            "Input points must be an Nx2 tensor.".to_string(),
        ));
    }

    let num_points = src_points.shape[0];
    let src_data = src_points.as_slice();

    // --- 1. Undistort all points iteratively ---
    let undistorted_normalized_points: Vec<Point2<f64>> = (0..num_points)
        .map(|i| {
            let distorted_point = Point2 {
                x: src_data[i * 2],
                y: src_data[i * 2 + 1],
            };
            undistort_single_point_iterative(distorted_point, intrinsic, distortion)
        })
        .collect();

    // --- 2. Apply optional rectification rotation (R) ---
    const R_IDENTITY_DATA: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let r_data = match r_matrix {
        Some(r_mat) => {
            if r_mat.shape != [3, 3] {
                return Err(DistortionError::InvalidMatrixShape(
                    "R matrix must be 3x3.".to_string(),
                ));
            }
            r_mat.as_slice()
        }
        None => &R_IDENTITY_DATA,
    };

    let rotated_rays: Vec<[f64; 3]> = undistorted_normalized_points
        .iter()
        .map(|point_norm| {
            let x = point_norm.x;
            let y = point_norm.y;
            let z = 1.0;
            [
                r_data[0] * x + r_data[1] * y + r_data[2] * z, // rotated_x
                r_data[3] * x + r_data[4] * y + r_data[5] * z, // rotated_y
                r_data[6] * x + r_data[7] * y + r_data[8] * z, // rotated_z
            ]
        })
        .collect();

    // --- 3. Apply optional new projection matrix (P) or homogenize ---
    let mut dst_points_vec = Vec::with_capacity(num_points * 2);

    if let Some(p_mat) = p_matrix {
        let p_shape = p_mat.shape;
        if !(p_shape == [3, 3] || p_shape == [3, 4]) {
            return Err(DistortionError::InvalidMatrixShape(
                "New projection matrix P must be 3x3 or 3x4.".to_string(),
            ));
        }
        let p_data = p_mat.as_slice();
        let width = p_shape[1]; // 3 or 4
        let has_translation = width == 4;

        for ray in &rotated_rays {
            let rotated_x = ray[0];
            let rotated_y = ray[1];
            let rotated_z = ray[2];

            let t_x = if has_translation { p_data[3] } else { 0.0 };
            let t_y = if has_translation { p_data[width + 3] } else { 0.0 };
            let t_w = if has_translation { p_data[2 * width + 3] } else { 0.0 };

            let final_x =
                p_data[0] * rotated_x + p_data[1] * rotated_y + p_data[2] * rotated_z + t_x;
            let final_y = p_data[width] * rotated_x
                + p_data[width + 1] * rotated_y
                + p_data[width + 2] * rotated_z
                + t_y;
            let final_w = p_data[2 * width] * rotated_x
                + p_data[2 * width + 1] * rotated_y
                + p_data[2 * width + 2] * rotated_z
                + t_w;

            let w_inv = if final_w.abs() > 1e-9 { 1.0 / final_w } else { 0.0 };
            dst_points_vec.push(final_x * w_inv);
            dst_points_vec.push(final_y * w_inv);
        }
    } else {
        // No P matrix. Output is just the rectified normalized coordinates.
        for ray in &rotated_rays {
            let final_x = ray[0];
            let final_y = ray[1];
            let final_w = ray[2];

            let w_inv = if final_w.abs() > 1e-9 { 1.0 / final_w } else { 0.0 };
            dst_points_vec.push(final_x * w_inv);
            dst_points_vec.push(final_y * w_inv);
        }
    }

    CpuTensor2::from_shape_slice([num_points, 2], &dst_points_vec, CpuAllocator)
        .map_err(|e| DistortionError::InternalError(e.to_string()))
}

/// Generate the undistort and rectify map for a polynomial distortion model (Brown-Conrady)
///
/// This function creates a mapping that can be used to correct for lens distortion in an image.
/// It creates a map such that: `undistorted_img(x, y) = distorted_img(map_x(x, y), map_y(x, y))`.
/// It assumes the grid of `(x, y)` coordinates corresponds to the *ideal* (undistorted) image pixels.
///
/// # Arguments
///
/// * `intrinsic` - The intrinsic parameters of the camera.
/// * `extrinsic` - The extrinsic parameters of the camera - currently unused.
/// * `new_intrinsic` - The new intrinsic parameters for the output image - currently unused.
/// * `distortion` - The distortion parameters of the camera.
/// * `size` - The size of the image to be corrected (i.e., the destination image).
///
/// # Returns
///
/// A tuple containing two f32 tensors: `map_x` and `map_y`.
pub fn generate_correction_map_polynomial(
    intrinsic: &CameraIntrinsic,
    _extrinsic: &CameraExtrinsic,
    _new_intrinsic: &CameraIntrinsic,
    distortion: &PolynomialDistortion,
    size: &ImageSize,
) -> Result<(CpuTensor2<f32>, CpuTensor2<f32>), TensorError> {
    let (dst_rows, dst_cols) = (size.height, size.width);
    meshgrid_from_fn(dst_cols, dst_rows, |x, y| {
        let ideal_point = Point2 {
            x: x as f64,
            y: y as f64,
        };
        let distorted_point = distort_point_polynomial(ideal_point, intrinsic, distortion);
        Ok((distorted_point.x as f32, distorted_point.y as f32))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;
    use kornia_tensor::allocator::CpuAllocator;

    // Helper to create common test objects
    fn get_test_camera_params() -> (CameraIntrinsic, PolynomialDistortion) {
        let intrinsic = CameraIntrinsic {
            fx: 577.5,
            fy: 577.5,
            cx: 319.5,
            cy: 239.5,
        };
        let distortion = PolynomialDistortion {
            k1: 0.1,
            k2: -0.05,
            k3: 0.005,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: 0.001,
            p2: -0.002,
        };
        (intrinsic, distortion)
    }

    #[test]
    fn test_round_trip_distortion() -> Result<(), DistortionError> {
        let (intrinsic, distortion) = get_test_camera_params();
        let ideal_points_data = vec![100.0, 80.0, 400.0, 300.0, 250.0, 150.0];
        let ideal_points_tensor =
            CpuTensor2::from_shape_slice([3, 2], &ideal_points_data, CpuAllocator).unwrap();
        let ideal_data = ideal_points_tensor.as_slice();

        let mut distorted_points_data = Vec::new();
        for i in 0..3 {
            let point = Point2 {
                x: ideal_data[i * 2],
                y: ideal_data[i * 2 + 1],
            };
            let distorted = distort_point_polynomial(point, &intrinsic, &distortion);
            distorted_points_data.push(distorted.x);
            distorted_points_data.push(distorted.y);
        }
        let distorted_points_tensor =
            CpuTensor2::from_shape_slice([3, 2], &distorted_points_data, CpuAllocator).unwrap();

        let recovered_normalized = undistort_points_polynomial(
            &distorted_points_tensor,
            &intrinsic,
            &distortion,
            None,
            None,
        )?;

        let recovered_data = recovered_normalized.as_slice();
        for i in 0..3 {
            let recovered_x = recovered_data[i * 2] * intrinsic.fx + intrinsic.cx;
            let recovered_y = recovered_data[i * 2 + 1] * intrinsic.fy + intrinsic.cy;

            // This test always passed, but now it passes with much higher required precision.
            assert!((recovered_x - ideal_data[i * 2]).abs() < 1e-9);
            assert!((recovered_y - ideal_data[i * 2 + 1]).abs() < 1e-9);
        }
        Ok(())
    }

    #[test]
    fn test_generate_correction_map_polynomial() -> Result<(), TensorError> {
        let (intrinsic, distortion) = get_test_camera_params();
        let size = ImageSize {
            width: 8,
            height: 4,
        };
        let extrinsic = CameraExtrinsic {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        };
        let (map_x, map_y) = generate_correction_map_polynomial(
            &intrinsic,
            &extrinsic,
            &intrinsic,
            &distortion,
            &size,
        )?;

        assert_eq!(map_x.shape, [4, 8]);
        assert_eq!(map_y.shape, [4, 8]);
        Ok(())
    }
} 