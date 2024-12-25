use super::{CameraExtrinsic, CameraIntrinsic};
use crate::interpolation::grid::meshgrid_from_fn;
use kornia_image::ImageSize;
use kornia_tensor::{CpuTensor2, TensorError};

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

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

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
}
