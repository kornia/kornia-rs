use super::{CameraExtrinsic, CameraIntrinsic};
use crate::interpolation::grid::meshgrid_from_fn;
use kornia_core::{Tensor2, TensorError};
use kornia_image::ImageSize;

/// Represents the polynomial distortion parameters of a camera
///
/// # Fields
///
/// * `k1` - The first radial distortion coefficient
/// * `k2` - The second radial distortion coefficient
/// * `k3` - The third radial distortion coefficient
/// * `k4` - The fourth radial distortion coefficient
/// * `k5` - The fifth radial distortion coefficient
/// * `k6` - The sixth radial distortion coefficient
/// * `p1` - The first tangential distortion coefficient
/// * `p2` - The second tangential distortion coefficient
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

/// Distort a point using polynomial distortion
///
/// # Arguments
///
/// * `x` - The x coordinate of the point
/// * `y` - The y coordinate of the point
/// * `intrinsic` - The intrinsic parameters of the camera
/// * `distortion` - The distortion parameters of the camera
///
/// # Returns
///
/// * `x` - The x coordinate of the distorted point
/// * `y` - The y coordinate of the distorted point
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

    // radial distortion
    let kr = (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        / (1.0 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2);

    // tangential distortion
    let xd = x * kr + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    let yd = y * kr + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    // denormalize the coordinates
    let xdst = fx * xd + cx;
    let ydst = fy * yd + cy;

    (xdst, ydst)
}

/// Generate the undistort and rectify map for a polynomial distortion model
///
/// # Arguments
///
/// * `intrinsic` - The intrinsic parameters of the camera
/// * `extrinsic` - The extrinsic parameters of the camera
/// * `new_intrinsic` - The new intrinsic parameters of the camera
/// * `distortion` - The distortion parameters of the camera
/// * `size` - The size of the image
///
/// # Returns
///
/// * `map_x` - The x map for undistorting and rectifying the image
/// * `map_y` - The y map for undistorting and rectifying the image
pub fn generate_correction_map_polynomial(
    intrinsic: &CameraIntrinsic,
    _extrinsic: &CameraExtrinsic,
    _new_intrinsic: &CameraIntrinsic,
    distortion: &PolynomialDistortion,
    size: &ImageSize,
) -> Result<(Tensor2<f32>, Tensor2<f32>), TensorError> {
    //// create a grid of x and y coordinates for the output image
    //// and interpolate the values from the input image.
    let (dst_rows, dst_cols) = (size.height, size.width);
    let (map_x, map_y) = meshgrid_from_fn(dst_cols, dst_rows, |x, y| {
        let (xdst, ydst) = distort_point_polynomial(x as f64, y as f64, intrinsic, distortion);
        Ok((xdst as f32, ydst as f32))
    })?;
    //let (mut map_x, mut map_y) = meshgrid_image(dst_rows, dst_rows, dst_cols, dst_cols)?;

    //// create a grid of x and y coordinates for the output image
    //map_x
    //    .as_slice_mut()
    //    .par_chunks_exact_mut(dst_cols)
    //    .zip(map_y.as_slice_mut().par_chunks_exact_mut(dst_cols))
    //    .for_each(|(xarr, yarr)| {
    //        xarr.iter_mut().zip(yarr.iter_mut()).for_each(|(x, y)| {
    //            let (xdst, ydst) =
    //                distort_point_polynomial(*x as f64, *y as f64, intrinsic, distortion);
    //            *x = xdst as f32;
    //            *y = ydst as f32;
    //        });
    //    });

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
