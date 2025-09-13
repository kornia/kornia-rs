use super::{CameraExtrinsic, CameraIntrinsic};
use crate::interpolation::grid::meshgrid_from_fn;
use anyhow::{bail, Result};
use kornia_image::ImageSize;
use kornia_tensor::{CpuTensor2, TensorError};
use ndarray::{arr1, s, stack, Array1, Array2, Axis};

/// Represents the polynomial distortion parameters of a camera using the Brown-Conrady model.
///
/// This struct encapsulates both radial (k1-k6) and tangential (p1-p2) distortion coefficients.
/// These parameters are used to model lens distortion in camera calibration and image correction.
///
/// # Fields
///
/// * `k1`, `k2`, `k3` - First, second, and third radial distortion coefficients
/// * `k4`, `k5`, `k6` - Fourth, fifth, and sixth radial distortion coefficients
/// * `p1`, `p2` - First and second tangential distortion coefficients
///
/// # Note
///
/// Higher-order coefficients (k4-k6) are often set to zero for simpler models.
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
/// This function takes an undistorted point (x, y) and applies both radial and tangential
/// distortion based on the provided camera intrinsics and distortion parameters.
///
/// # Arguments
///
/// * `x` - The x coordinate of the undistorted point
/// * `y` - The y coordinate of the undistorted point
/// * `intrinsic` - The intrinsic parameters of the camera
/// * `distortion` - The distortion parameters of the camera
///
/// # Returns
///
/// A tuple `(x', y')` containing the coordinates of the distorted point
///
/// # Example
///
/// ```
/// use kornia_imgproc::calibration::{CameraIntrinsic, distortion::{PolynomialDistortion, distort_point_polynomial}};
///
/// let intrinsic = CameraIntrinsic { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 };
/// let distortion = PolynomialDistortion { k1: 0.1, k2: 0.01, k3: 0.001, k4: 0.0, k5: 0.0, k6: 0.0, p1: 0.0005, p2: 0.0005 };
///
/// let (x_distorted, y_distorted) = distort_point_polynomial(100.0, 100.0, &intrinsic, &distortion);
/// ```
pub fn distort_point_polynomial(
    x: f64,
    y: f64,
    intrinsic: &CameraIntrinsic,
    distortion: &PolynomialDistortion,
) -> (f64, f64) {
    // unpack the intrinsic and distortion parameters
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
    let x = (x - cx) / fx;
    let y = (y - cy) / fy;

    // calculate the radial distance
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let r6 = r4 * r2;

    // radial distortion
    let kr = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6);

    // tangential distortion
    let x_2 = 2.0 * x;
    let y_2 = 2.0 * y;
    let xy_2 = x_2 * y;
    let xd = x * kr + xy_2 * p1 + p2 * (r2 + x_2 * x);
    let yd = y * kr + p1 * (r2 + y_2 * y) + xy_2 * p2;

    // denormalize the coordinates
    (fx * xd + cx, fy * yd + cy)
}

/// Generate the undistort and rectify map for a polynomial distortion model (Brown-Conrady)
///
/// This function creates a mapping that can be used to correct for lens distortion in an image.
/// It generates two maps (map_x and map_y) that describe how each pixel in the distorted image
/// should be remapped to create an undistorted image.
///
/// # Arguments
///
/// * `intrinsic` - The intrinsic parameters of the camera (focal length, principal point)
/// * `extrinsic` - The extrinsic parameters of the camera (rotation, translation) - currently unused
/// * `new_intrinsic` - The new intrinsic parameters for the output image - currently unused
/// * `distortion` - The distortion parameters of the camera (radial and tangential coefficients)
/// * `size` - The size of the image to be corrected
///
/// # Returns
///
/// A tuple containing:
/// * `map_x` - A 2D tensor representing the x-coordinates for remapping
/// * `map_y` - A 2D tensor representing the y-coordinates for remapping
///
/// Both maps have the same dimensions as the input image.
///
/// # Errors
///
/// Returns a `TensorError` if there's an issue creating the meshgrid or performing calculations.
pub fn generate_correction_map_polynomial(
    intrinsic: &CameraIntrinsic,
    _extrinsic: &CameraExtrinsic,
    _new_intrinsic: &CameraIntrinsic,
    distortion: &PolynomialDistortion,
    size: &ImageSize,
) -> Result<(CpuTensor2<f32>, CpuTensor2<f32>), TensorError> {
    //// create a grid of x and y coordinates for the output image
    //// and interpolate the values from the input image.
    let (dst_rows, dst_cols) = (size.height, size.width);
    let (map_x, map_y) = meshgrid_from_fn(dst_cols, dst_rows, |x, y| {
        let (xdst, ydst) = distort_point_polynomial(x as f64, y as f64, intrinsic, distortion);
        Ok((xdst as f32, ydst as f32))
    })?;

    Ok((map_x, map_y))
}

/// Computes the ideal point coordinates from observed (distorted) point coordinates.
///
/// This function iteratively finds the undistorted point coordinates that, when distorted,
/// match the observed `src_points`. It operates in a vectorized manner for efficiency and
/// can optionally apply rectification and new projection matrices.
///
/// # Arguments
/// * `src_points` - An `Nx2` `ndarray` array of observed (distorted) point coordinates.
/// * `intrinsic` - The intrinsic parameters of the camera.
/// * `distortion` - The distortion parameters of the camera.
/// * `r_matrix` - Optional 3x3 rectification transformation. If `None`, an identity matrix is used.
/// * `p_matrix` - Optional new projection matrix (3x3 or 3x4). If `None`, the final transformation is determined by `r_matrix` alone.
///
/// # Returns
///
/// An `anyhow::Result` containing the `Nx2` `ndarray` array of corrected (undistorted and rectified) points.
pub fn undistort_points_polynomial(
    src_points: &Array2<f64>,
    intrinsic: &CameraIntrinsic,
    distortion: &PolynomialDistortion,
    r_matrix: &Option<Array2<f64>>,
    p_matrix: &Option<Array2<f64>>,
) -> Result<Array2<f64>> {
    // --- 1. Validation ---
    if src_points.ndim() != 2 || src_points.shape()[1] != 2 {
        bail!("Input points must be an Nx2 array.");
    }

    // --- 2. Extract Coefficients ---
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

    // --- 3. Normalize Distorted Points (Vectorized) ---
    let u = src_points.column(0);
    let v = src_points.column(1);
    let x_distorted = u.mapv(|u_i| (u_i - cx) / fx);
    let y_distorted = v.mapv(|v_i| (v_i - cy) / fy);

    // --- 4. Iteratively Find Undistorted Points (Vectorized) ---
    let mut x = x_distorted.clone();
    let mut y = y_distorted.clone();

    // Iterate to find the inverse of the distortion function.
    // A higher number of iterations are needed for convergence with significant distortion.
    for _ in 0..10 {
        let r2 = &x * &x + &y * &y;
        let r4 = &r2 * &r2;
        let r6 = &r4 * &r2;

        let radial_numerator = 1.0 + k1 * &r2 + k2 * &r4 + k3 * &r6;
        let radial_denominator = 1.0 + k4 * &r2 + k5 * &r4 + k6 * &r6;
        let radial_dist = &radial_numerator / &radial_denominator;

        let d_tan_x = 2.0 * p1 * &x * &y + p2 * (&r2 + 2.0 * &x * &x);
        let d_tan_y = p1 * (&r2 + 2.0 * &y * &y) + 2.0 * p2 * &x * &y;

        x = (&x_distorted - &d_tan_x) / &radial_dist;
        y = (&y_distorted - &d_tan_y) / &radial_dist;
    }

    // --- 5. Apply Rectification (R) and New Projection (P) if provided ---
    let ones = Array1::ones(src_points.nrows());
    let undistorted_homo = stack(Axis(1), &[x.view(), y.view(), ones.view()])?;

    let identity = Array2::eye(3);
    let r_mat = r_matrix.as_ref().unwrap_or(&identity);

    // Create intrinsic matrix K
    let k_matrix = Array2::from_shape_vec(
        (3, 3),
        vec![fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
    ).unwrap();

    // Compose the final transformation matrix (P * R), using K if P is None
    let final_transform_matrix = if let Some(p) = p_matrix {
        if p.shape() != [3, 4] && p.shape() != [3, 3] {
            bail!("P matrix must be a 3x3 or 3x4 array.");
        }
        let p_3x3 = p.slice(s![.., ..3]);
        p_3x3.dot(r_mat)
    } else {
        k_matrix.dot(r_mat)
    };

    let projected_homo = undistorted_homo.dot(&final_transform_matrix.t());

    // --- 6. Final Perspective Divide and Output Formatting ---
    let mut dst_points = Array2::zeros((src_points.nrows(), 2));
    let final_x = projected_homo.column(0);
    let final_y = projected_homo.column(1);
    let w = projected_homo.column(2);

    // Use azip for efficient, parallel-friendly row-wise operations
    ndarray::azip!((mut dst_row in dst_points.rows_mut(), &x_i in &final_x, &y_i in &final_y, &w_i in &w) {
        let w_inv = if w_i.abs() > 1e-6 { 1.0 / w_i } else { 0.0 };
        dst_row.assign(&arr1(&[x_i * w_inv, y_i * w_inv]));
    });

    Ok(dst_points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;
    use ndarray::arr2;

    #[test]
    fn test_distort_point_polynomial() {
        let intrinsic = CameraIntrinsic {
            fx: 577.48583984375,
            fy: 652.8748779296875,
            cx: 577.48583984375,
            cy: 386.1428833007813,
        };

        let distortion = PolynomialDistortion {
            k1: 1.7547749280929563,
            k2: 0.0097926277667284,
            k3: -0.027250492945313457,
            k4: 2.1092164516448975,
            k5: 0.462927520275116,
            k6: -0.08215277642011642,
            p1: -0.00005457743463921361,
            p2: 0.00003006766564794816,
        };

        let (x, y) = (100.0, 20.0);
        let (x, y) = distort_point_polynomial(x, y, &intrinsic, &distortion);

        assert_ne!(x, 194.24656721843076);
        assert_eq!(y, 98.83006704526377);
    }

    #[test]
    fn test_undistort_rectify_map_polynomial() -> Result<(), TensorError> {
        let intrinsic = CameraIntrinsic {
            fx: 577.48583984375,
            fy: 652.8748779296875,
            cx: 577.48583984375,
            cy: 386.1428833007813,
        };

        let distortion = PolynomialDistortion {
            k1: 1.7547749280929563,
            k2: 0.0097926277667284,
            k3: -0.027250492945313457,
            k4: 2.1092164516448975,
            k5: 0.462927520275116,
            k6: -0.08215277642011642,
            p1: -0.00005457743463921361,
            p2: 0.00003006766564794816,
        };

        let size = ImageSize {
            width: 8,
            height: 4,
        };

        let (map_x, map_y) = generate_correction_map_polynomial(
            &intrinsic,
            &CameraExtrinsic {
                rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                translation: [0.0, 0.0, 0.0],
            },
            &intrinsic,
            &distortion,
            &size,
        )?;

        assert_eq!(map_x.shape[0], 4);
        assert_eq!(map_x.shape[1], 8);
        assert_eq!(map_y.shape[0], 4);
        assert_eq!(map_y.shape[1], 8);

        Ok(())
    }

    #[test]
    fn test_undistort_points_polynomial() {
        let intrinsic = CameraIntrinsic {
            fx: 577.4858,
            fy: 577.4858,
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

        // 1. Define an original, undistorted point
        let x_undistorted = 200.0;
        let y_undistorted = 150.0;

        // 2. Distort this point using the forward function
        let (x_distorted, y_distorted) =
            distort_point_polynomial(x_undistorted, y_undistorted, &intrinsic, &distortion);

        // 3. Create an ndarray with the distorted point
        let distorted_points = arr2(&[[x_distorted, y_distorted]]);

        // 4. Undistort the point using the new function
        let undistorted_result =
            undistort_points_polynomial(&distorted_points, &intrinsic, &distortion, &None, &None)
                .unwrap();

        // 5. Check if the result is close to the original undistorted point
        let result_point = undistorted_result.row(0);
        let tolerance = 1e-6; // A small tolerance for floating point comparisons

        assert!((result_point[0] - x_undistorted).abs() < tolerance);
        assert!((result_point[1] - y_undistorted).abs() < tolerance);
    }
}