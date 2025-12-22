use super::{CameraExtrinsic, CameraIntrinsic};
use crate::interpolation::grid::meshgrid_from_fn;
use kornia_image::ImageSize;
use kornia_tensor::{CpuTensor2, Tensor, TensorAllocator, TensorError};

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

/// Termination criteria for iterative undistortion
#[derive(Clone, Copy, Debug)]
pub struct TermCriteria {
    /// Maximum number of iterations allowed.
    pub max_iter: usize,
    /// Minimum required improvement between iterations.
    pub eps: f64,
}

impl Default for TermCriteria {
    fn default() -> Self {
        Self {
            max_iter: 5,
            eps: 1e-2,
        }
    }
}

/// Iteratively solves for the undistorted **normalized** point using the
/// Brown–Conrady polynomial distortion model.
///
/// This function compenstates for distortion using iterative fixed-point method
///
/// # Arguments
///
/// * `x`, `y` — Initial guess for the **distorted** normalized coordinates.  
///   These are typically `(u - cx) / fx`, `(v - cy) / fy`.
///
/// * `u`, `v` — Original pixel coordinates.  
/// * `intr` — Camera intrinsic parameters (fx, fy, cx, cy).
/// * `dist` — Full Brown–Conrady distortion parameters:
///   radial (`k1..k6`) and tangential (`p1, p2`).
/// * `criteria` — Termination conditions:
///   - maximum number of iterations,
///   - epsilon threshold for stopping when update becomes sufficiently small.
///
/// # Returns
///
/// Returns a tuple `(x_u, y_u)` - recovered **undistorted normalized coordinates**.
/// These can be projected into pixel space via:
/// ```text
/// u_u = fx * x_u + cx
/// v_u = fy * y_u + cy
/// ```
///
/// # Notes
///
/// - Supported components:
///   - Radial distortion: `k1..k6`  
///   - Tangential distortion: `p1`, `p2`
///   - **Prism distortion is not supported.**
///
/// - This function inverts distortion only in **normalized** space.  
///   Any rectification or reprojection using `R` or `P` matrices must be
///   applied separately.
pub fn undistort_normalized_point_iter(
    mut x: f64,
    mut y: f64,
    u: f64,
    v: f64,
    intr: &CameraIntrinsic,
    dist: &PolynomialDistortion,
    criteria: TermCriteria,
) -> (f64, f64) {
    let fx = intr.fx;
    let fy = intr.fy;
    let cx = intr.cx;
    let cy = intr.cy;

    let k1 = dist.k1;
    let k2 = dist.k2;
    let p1 = dist.p1;
    let p2 = dist.p2;
    let k3 = dist.k3;
    let k4 = dist.k4;
    let k5 = dist.k5;
    let k6 = dist.k6;

    let mut prev_error = f64::INFINITY;
    // damping factor for fixed-points updates
    let mut alpha = 1.0;

    for _ in 0..criteria.max_iter {
        // compute r^2 for current estimate
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        // compute inverse combined radial distortion
        // icdist = (1 + k4*r^2 + k5*r^4 + k6*r^6) / (1 + k1*r^2 + k2*r^4 + k3*r^6)
        let denom = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        if denom.abs() < f64::EPSILON {
            // fallback to identity (should not happen in normal calibrations)
            break;
        }
        let numer = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
        let icdist = numer / denom;
        if !icdist.is_finite() {
            break;
        }

        // tangential + thin prism (we only have p1,p2 here; prism terms not included)
        let delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

        // damped fixed-point update
        // x0,y0 define current distorted-normalized coordinates before iteration (derived from pixel u,v)
        let x0 = (u - cx) / fx;
        let y0 = (v - cy) / fy;

        // but in this iteration we keep using x,y and correct toward (x0 - delta)*icdist
        // to be numerically stable we use new_x = (1-alpha)x + alpha(x0 - deltaX)*icdist
        let new_x = (1.0 - alpha) * x + alpha * (x0 - delta_x) * icdist;
        let new_y = (1.0 - alpha) * y + alpha * (y0 - delta_y) * icdist;

        // compute reprojection error (if eps used)
        let mut error = 0.0;
        if criteria.eps > 0.0 {
            // forward-distort new_x/new_y then project back to pixel coords to compute error
            let r2n = new_x * new_x + new_y * new_y;
            let r4n = r2n * r2n;
            let r6n = r4n * r2n;

            let cdist = 1.0 + k1 * r2n + k2 * r4n + k3 * r6n;
            let invcdist2 = 1.0 / (1.0 + k4 * r2n + k5 * r4n + k6 * r6n);

            let a1 = 2.0 * new_x * new_y;
            let a2 = r2n + 2.0 * new_x * new_x;
            let a3 = r2n + 2.0 * new_y * new_y;

            let xd0 = new_x * cdist * invcdist2 + p1 * a1 + p2 * a2;
            let yd0 = new_y * cdist * invcdist2 + p1 * a3 + p2 * a1;

            let x_proj = xd0 * fx + cx;
            let y_proj = yd0 * fy + cy;
            error = ((x_proj - u).powi(2) + (y_proj - v).powi(2)).sqrt();
        }

        // damping strategy
        // if reprojection error worsens reduce α (stronger damping)
        // otherwise accept the update
        if error > prev_error {
            alpha *= 0.5;
        } else {
            x = new_x;
            y = new_y;
        }

        prev_error = error;

        // convergence test (pixel RMS error)
        if criteria.eps > 0.0 && error < criteria.eps {
            break;
        }
    }

    (x, y)
}

/// Apply optional R (3x3) rectification and P (3x3 or 3x4) projection to normalized
/// undistorted coordinates.
///
/// # Arguments
/// * `x`, `y` — normalized undistorted coordinates.
/// * `r_opt` — optional 3×3 rectification matrix `R`.
/// * `p3_opt` — optional 3×3 projection matrix `P`.
/// * `p34_opt` — optional 3×4 projection matrix `P`.
///
/// - If r_opt is Some(&[[f64;3];3]), this function compute v = R * [x,y,1]^T and normalize v.
/// - If p3_opt (3x3 P) is Some, this function compute p = P * [x,y,1]^T and normalize p.
/// - Else if p34_opt (3x4 P) is Some, we compute p = P * [x,y,1,1]^T and normalize p.
/// - If none provided, we return the normalized (x,y) unchanged.
///
/// # Returns
/// `(x', y')` — the transformed coordinates after applying `R` and/or `P`.  
pub fn apply_r_and_p(
    mut x: f64,
    mut y: f64,
    r_opt: Option<&[[f64; 3]; 3]>,
    p3_opt: Option<&[[f64; 3]; 3]>,
    p34_opt: Option<&[[f64; 4]; 3]>,
) -> (f64, f64) {
    // apply R if present (v = R * [x,y,1]^T and normalize v)
    if let Some(r) = r_opt {
        let xx = r[0][0] * x + r[0][1] * y + r[0][2];
        let yy = r[1][0] * x + r[1][1] * y + r[1][2];
        let ww = r[2][0] * x + r[2][1] * y + r[2][2];
        if ww.abs() > f64::EPSILON {
            x = xx / ww;
            y = yy / ww;
        }
    }

    // apply P (3x3) (p = P * [x,y,1]^T and normalize p)
    if let Some(p3) = p3_opt {
        let xp = p3[0][0] * x + p3[0][1] * y + p3[0][2];
        let yp = p3[1][0] * x + p3[1][1] * y + p3[1][2];
        let wp = p3[2][0] * x + p3[2][1] * y + p3[2][2];
        if wp.abs() > f64::EPSILON {
            return (xp / wp, yp / wp);
        } else {
            return (xp, yp);
        }
    } else if let Some(p34) = p34_opt {
        // apply P (3x4) (p = P * [x,y,1,1]^T and normalize p)
        let xp = p34[0][0] * x + p34[0][1] * y + p34[0][2] * 1.0 + p34[0][3] * 1.0;
        let yp = p34[1][0] * x + p34[1][1] * y + p34[1][2] * 1.0 + p34[1][3] * 1.0;
        let wp = p34[2][0] * x + p34[2][1] * y + p34[2][2] * 1.0 + p34[2][3] * 1.0;
        if wp.abs() > f64::EPSILON {
            return (xp / wp, yp / wp);
        } else {
            return (xp, yp);
        }
    }

    // no projection, return normalized coordinates
    (x, y)
}

/// Validate that the source and destination tensors have correct and compatible shapes.
/// # Arguments
/// * `src` — Input tensor of shape `(N × 2)` containing source coordinates.
/// * `dst` — Output tensor of shape `(N × 2)` that will receive the transformed points.
///
/// # Returns
/// * `Ok(())` if shape constraints are satisfied.
/// * `Err(TensorError)` describing the mismatch otherwise.
fn check_shapes<A: TensorAllocator>(
    src: &Tensor<f64, 2, A>,
    dst: &mut Tensor<f64, 2, A>,
) -> Result<(), TensorError> {
    // src must be (N × 2)
    if src.shape[1] != 2 {
        return Err(TensorError::DimensionMismatch(format!(
            "src must have shape Nx2, got Nx{}",
            src.shape[1]
        )));
    }

    // dst must be (N × 2)
    if dst.shape[1] != 2 {
        return Err(TensorError::DimensionMismatch(format!(
            "dst must have shape Nx2, got Nx{}",
            dst.shape[1]
        )));
    }

    // src and dst must have same N
    if src.shape[0] != dst.shape[0] {
        return Err(TensorError::DimensionMismatch(format!(
            "src and dst must have same number of rows: src N={}, dst N={}",
            src.shape[0], dst.shape[0]
        )));
    }

    Ok(())
}

/// Computes the ideal point coordinates from the observed point coordinates.
/// - Solves for the undistorted point using the
///   Brown–Conrady polynomial distortion model (Iterative fixed-point method)
///   implemeted in `undistort_normalized_point_iter`.
/// - Optionally apply R(Rectification matrix) and/or P(Projection matrix)
///   implemented in `apply_r_and_p`.
///
/// # Arguments
/// * `src` - Input Tensor<f64, 2, A> with shape [N, 2] containing distorted pixel coords (u,v)
/// * `dst` - Output Tensor<f64, 2, A> with shape [N, 2] (will be overwritten)
/// * `intr` - camera intrinsic parameters in a struct CameraIntrinsic
/// * `dist` - polynomial distortion model parameters in a struct PolynomialDistortion
/// * `r_opt` - Option<&[[f64;3];3]> rectification matrix R (or None)
/// * `p3_opt` - Option<&[[f64;3];3]> projection matrix P (3x3) (or None)
/// * `p34_opt` - Option<&[[f64;4];3]> projection matrix P (3x4) (or None)
/// * `criteria` - termination criteria for the iterative undistortion
///
/// # Returns
/// * `Ok(())` if successfull computation of ideal(undistorted) coordinates.
/// * `Err(TensorError)` if dimesion mismatch is found for `src` and `dst`.
///
/// # Example:
/// ```rust
/// use kornia_tensor::{CpuAllocator, Tensor};
/// use kornia_imgproc::calibration::{CameraIntrinsic, distortion::{PolynomialDistortion, undistort_points, TermCriteria}};
/// // distorted points
/// let src = Tensor::<f64, 2, _>::from_shape_vec([2, 2], vec![320.0, 240.0, 100.0,  50.0], CpuAllocator).unwrap();
/// let mut dst = Tensor::<f64, 2, _>::from_shape_vec([2, 2], vec![0.0; 4], CpuAllocator).unwrap();
///
/// // intrinsics
/// let intr = CameraIntrinsic {
///     fx: 800.0, fy: 800.0,
///     cx: 320.0, cy: 240.0,
/// };
///
/// // simple distortion model
/// let dist = PolynomialDistortion {
///     k1: -0.1, k2: 0.01, k3: 0.0, k4:0.0, k5:0.0, k6:0.0,
///     p1: 0.0, p2: 0.0,
/// };
///
/// // no R and P so undistorted normalized coords
/// undistort_points(
///     &src,
///     &mut dst,
///     &intr,
///     &dist,
///     None,   // R
///     None,   // P (3x3)
///     None,   // P (3x4)
///     TermCriteria::default()
/// ).unwrap();
///
/// // dst[[i,0]], dst[[i,1]] now contain the undistorted normalized coordinates.
/// ```
#[allow(clippy::too_many_arguments)]
pub fn undistort_points<A: TensorAllocator>(
    src: &Tensor<f64, 2, A>,
    dst: &mut Tensor<f64, 2, A>,
    intr: &CameraIntrinsic,
    dist: &PolynomialDistortion,
    r_opt: Option<&[[f64; 3]; 3]>,
    p3_opt: Option<&[[f64; 3]; 3]>,
    p34_opt: Option<&[[f64; 4]; 3]>,
    criteria: TermCriteria,
) -> Result<(), TensorError> {
    check_shapes(src, dst)?;
    let n = src.shape[0];

    let src_slice = src.storage.as_slice();
    let dst_slice = dst.storage.as_mut_slice();

    // precompute strides-based indexing
    let s_stride0 = src.strides[0];
    let s_stride1 = src.strides[1];
    let d_stride0 = dst.strides[0];
    let d_stride1 = dst.strides[1];

    for row in 0..n {
        // read distorted pixel coordinates (u, v)
        let u = src_slice[row * s_stride0];
        let v = src_slice[row * s_stride0 + s_stride1];

        // normalized distorted coordinates
        let x_init = (u - intr.cx) / intr.fx;
        let y_init = (v - intr.cy) / intr.fy;

        // iterative undistortion (returns normalized undistorted coordinates)
        let (x_u, y_u) =
            undistort_normalized_point_iter(x_init, y_init, u, v, intr, dist, criteria);

        // apply R and P (if provided), result is transformed coordinates or normalized coordinates.
        let (out_x, out_y) = apply_r_and_p(x_u, y_u, r_opt, p3_opt, p34_opt);

        // store undistored coordiantes to dst tensor
        dst_slice[row * d_stride0] = out_x;
        dst_slice[row * d_stride0 + d_stride1] = out_y;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;
    use kornia_tensor::CpuAllocator;

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

    // helper function
    fn realistic_intr() -> CameraIntrinsic {
        CameraIntrinsic {
            fx: 612.3,
            fy: 610.8,
            cx: 320.1,
            cy: 241.7,
        }
    }

    // helper function
    fn realistic_dist() -> PolynomialDistortion {
        PolynomialDistortion {
            k1: -0.321,
            k2: 0.124,
            k3: -0.012,
            k4: 0.210,
            k5: -0.015,
            k6: 0.001,
            p1: 0.00042,
            p2: -0.00031,
        }
    }

    // helper function
    fn strong_dist() -> PolynomialDistortion {
        PolynomialDistortion {
            k1: -0.7,
            k2: 0.35,
            k3: -0.05,
            k4: 0.6,
            k5: 0.05,
            k6: -0.01,
            p1: 0.0008,
            p2: -0.0006,
        }
    }

    // helper function
    fn no_dist() -> PolynomialDistortion {
        PolynomialDistortion {
            k1: 0.,
            k2: 0.,
            k3: 0.,
            k4: 0.,
            k5: 0.,
            k6: 0.,
            p1: 0.,
            p2: 0.,
        }
    }

    // helper function
    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    // helper function
    fn forward_dist_norm(x: f64, y: f64, d: &PolynomialDistortion) -> (f64, f64) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let c = 1. + d.k1 * r2 + d.k2 * r4 + d.k3 * r6;
        let inv = 1. / (1. + d.k4 * r2 + d.k5 * r4 + d.k6 * r6);

        let a1 = 2. * x * y;
        let a2 = r2 + 2. * x * x;
        let a3 = r2 + 2. * y * y;

        let xd = x * c * inv + d.p1 * a1 + d.p2 * a2;
        let yd = y * c * inv + d.p1 * a3 + d.p2 * a1;

        (xd, yd)
    }

    // helper function
    fn norm_to_pix(k: &CameraIntrinsic, x: f64, y: f64) -> (f64, f64) {
        (x * k.fx + k.cx, y * k.fy + k.cy)
    }

    #[test]
    fn identity_no_distortion() {
        let intr = realistic_intr();
        let dist = no_dist();
        let c = TermCriteria {
            max_iter: 10,
            eps: 1e-12,
        };

        let u = 420.0;
        let v = 260.0;
        let x0 = (u - intr.cx) / intr.fx;
        let y0 = (v - intr.cy) / intr.fy;

        let (xu, yu) = undistort_normalized_point_iter(x0, y0, u, v, &intr, &dist, c);

        assert!(approx(xu, x0, 1e-12));
        assert!(approx(yu, y0, 1e-12));
    }

    #[test]
    fn stable_distortion() {
        let intr = realistic_intr();
        let dist = realistic_dist();
        let c = TermCriteria {
            max_iter: 20,
            eps: 1e-7,
        };

        let pts = [(0.0, 0.0), (0.15, -0.1)];

        for &(x, y) in &pts {
            let (xd, yd) = forward_dist_norm(x, y, &dist);
            let (u, v) = norm_to_pix(&intr, xd, yd);

            let x0 = (u - intr.cx) / intr.fx;
            let y0 = (v - intr.cy) / intr.fy;

            let (xu, yu) = undistort_normalized_point_iter(x0, y0, u, v, &intr, &dist, c);

            assert!(approx(xu, x, 1e-5));
            assert!(approx(yu, y, 1e-5));
        }
    }

    #[test]
    fn strong_distortion() {
        let intr = realistic_intr();
        let dist = strong_dist();
        let c = TermCriteria {
            max_iter: 50,
            eps: 1e-7,
        };

        let pts = [(0.0, 0.0), (0.08, -0.05)];

        for &(x, y) in &pts {
            let (xd, yd) = forward_dist_norm(x, y, &dist);
            let (u, v) = norm_to_pix(&intr, xd, yd);

            let x0 = (u - intr.cx) / intr.fx;
            let y0 = (v - intr.cy) / intr.fy;

            let (xu, yu) = undistort_normalized_point_iter(x0, y0, u, v, &intr, &dist, c);

            assert!(approx(xu, x, 1e-4));
            assert!(approx(yu, y, 1e-4));
        }
    }

    #[test]
    fn undistort_with_r_and_p() {
        let intr = realistic_intr();
        let dist = realistic_dist();
        let c = TermCriteria::default();

        let (x, y) = (0.1, -0.08);

        let (xd, yd) = forward_dist_norm(x, y, &dist);
        let (u, v) = norm_to_pix(&intr, xd, yd);

        let src = Tensor::<f64, 2, _>::from_shape_vec([1, 2], vec![u, v], CpuAllocator).unwrap();
        let mut dst =
            Tensor::<f64, 2, _>::from_shape_vec([1, 2], vec![0.0, 0.0], CpuAllocator).unwrap();

        let p = [
            [intr.fx, 0.0, intr.cx],
            [0.0, intr.fy, intr.cy],
            [0.0, 0.0, 1.0],
        ];

        undistort_points(&src, &mut dst, &intr, &dist, None, Some(&p), None, c).unwrap();

        let out = dst.storage.as_slice();
        let u_out = out[0];
        let v_out = out[1];

        let (ue, ve) = norm_to_pix(&intr, x, y);

        assert!(approx(u_out, ue, 1e-2));
        assert!(approx(v_out, ve, 1e-2));
    }

    #[test]
    fn shape_error() {
        let src =
            Tensor::<f64, 2, _>::from_shape_vec([1, 3], vec![1., 2., 3.], CpuAllocator).unwrap();
        let mut dst =
            Tensor::<f64, 2, _>::from_shape_vec([1, 2], vec![0., 0.], CpuAllocator).unwrap();

        let r = undistort_points(
            &src,
            &mut dst,
            &realistic_intr(),
            &no_dist(),
            None,
            None,
            None,
            TermCriteria::default(),
        );

        assert!(r.is_err());
    }
}
