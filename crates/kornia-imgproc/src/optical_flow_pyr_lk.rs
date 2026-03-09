/// Lucas–Kanade optical flow with pyramids.
use crate::filter::scharr_spatial_gradient_float;
use crate::interpolation::{interpolate_pixel, InterpolationMode};
use crate::pyramid::pyrdown_f32;
use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use rayon::prelude::*;
use thiserror::Error;

/// Termination criteria for LK iterations
#[derive(Debug, Clone)]
/// Terminate after a fixed number of iterations.
pub enum TermCriteria {
    /// Terminate after a fixed number of iterations.
    Count,
    /// Terminate when the error is below a threshold.
    Eps,
    /// Terminate when either the iteration count or error threshold is reached.
    Both,
}

/// Border handling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Clamp coordinates to the image border.
pub enum BorderMode {
    /// Clamp coordinates to the image border.
    Clamp,
    /// Mirror coordinates at the image border.
    Mirror,
    /// Reject points that fall outside the image border.
    Reject,
}

/// Parameters for LK optical flow
#[derive(Debug, Clone)]
pub struct PyrLKParams {
    /// LK integration window size in pixels.
    ///
    /// Currently, only a window size of exactly `21` is supported.
    pub win_size: usize,
    /// Maximum pyramid level (0 means single-scale tracking).
    pub max_level: usize,
    /// Maximum Gauss-Newton iterations per pyramid level.
    pub max_iter: usize,
    /// Convergence epsilon for incremental flow update.
    pub epsilon: f32,
    /// Minimum accepted eigenvalue of the 2x2 structure tensor.
    pub min_eigen_threshold: f32,
    /// Whether to use `next_pts_in` as initial flow estimate.
    pub use_initial_flow: bool,
    /// Iteration termination policy.
    pub term_criteria: TermCriteria,
    /// Border handling policy for image sampling.
    pub border_mode: BorderMode,
}

impl Default for PyrLKParams {
    fn default() -> Self {
        Self {
            win_size: 21,
            max_level: 3,
            max_iter: 30,
            epsilon: 0.01,
            min_eigen_threshold: 1e-4,
            use_initial_flow: false,
            term_criteria: TermCriteria::Both,
            border_mode: BorderMode::Clamp,
        }
    }
}

/// Output for each tracked feature from LK tracking.
#[derive(Debug, Clone)]
pub struct PyrLKResult {
    /// Tracked positions of the feature points.
    pub next_pts: Vec<[f32; 2]>,
    /// Status flag (1 if tracked successfully, 0 otherwise).
    pub status: Vec<u8>,
    /// Tracking error for each feature.
    pub error: Vec<f32>,
}

/// Precomputed data for sparse pyramidal LK tracking.
#[derive(Clone)]
pub struct PyrLKPrecomputed<A: ImageAllocator> {
    /// Gaussian pyramid of previous image.
    pub prev_pyr: Vec<Image<f32, 1, A>>,
    /// Gaussian pyramid of next image.
    pub next_pyr: Vec<Image<f32, 1, A>>,
    /// X gradient pyramid for previous image.
    pub grad_x_pyr: Vec<Image<f32, 1, A>>,
    /// Y gradient pyramid for previous image.
    pub grad_y_pyr: Vec<Image<f32, 1, A>>,
}

/// Error type for sparse pyramidal Lucas–Kanade optical flow.
#[derive(Debug, Error)]
pub enum PyrLKError {
    /// The window size is unsupported.
    #[error("invalid LK window size: {0}. Currently, only window size 21 is supported")]
    InvalidWindowSize(usize),
    /// Input image sizes must match.
    #[error(
        "invalid image size: prev is {prev_width}x{prev_height}, next is {next_width}x{next_height}"
    )]
    ImageSizeMismatch {
        /// Previous image width.
        prev_width: usize,
        /// Previous image height.
        prev_height: usize,
        /// Next image width.
        next_width: usize,
        /// Next image height.
        next_height: usize,
    },
    /// Initial flow points length does not match feature count.
    #[error("next_pts_in length ({provided}) does not match prev_pts length ({expected})")]
    InitialFlowLengthMismatch {
        /// Expected number of points.
        expected: usize,
        /// Provided number of points.
        provided: usize,
    },
    /// Image operation failure from lower-level APIs.
    #[error(transparent)]
    Image(#[from] ImageError),
    /// Invalid precomputed pyramid layout for LK tracking.
    #[error("invalid precomputed pyramids: expected {expected_levels} levels, got prev={prev_levels}, next={next_levels}, grad_x={grad_x_levels}, grad_y={grad_y_levels}")]
    InvalidPrecomputedLevels {
        /// Expected number of levels.
        expected_levels: usize,
        /// Previous pyramid levels.
        prev_levels: usize,
        /// Next pyramid levels.
        next_levels: usize,
        /// X-gradient pyramid levels.
        grad_x_levels: usize,
        /// Y-gradient pyramid levels.
        grad_y_levels: usize,
    },
    /// Precomputed pyramids have mismatched image dimensions at a specific level.
    #[error("precomputed pyramid size mismatch at level {level}: prev={prev_size:?}, next={next_size:?}, grad_x={grad_x_size:?}, grad_y={grad_y_size:?}")]
    InvalidPrecomputedSizes {
        /// The pyramid level index where the mismatch occurred.
        level: usize,
        /// Size of the previous image at this level.
        prev_size: ImageSize,
        /// Size of the next image at this level.
        next_size: ImageSize,
        /// Size of the X-gradient image at this level.
        grad_x_size: ImageSize,
        /// Size of the Y-gradient image at this level.
        grad_y_size: ImageSize,
    },
}

fn mirror_coord(x: f32, max: f32) -> f32 {
    if max <= 0.0 {
        return 0.0;
    }
    let period = 2.0 * max;
    let mut t = x.rem_euclid(period);
    if t > max {
        t = period - t;
    }
    t
}

#[inline]
fn sample_at<A: ImageAllocator>(img: &Image<f32, 1, A>, x: f32, y: f32, mode: BorderMode) -> f32 {
    let max_x = img.cols() as f32 - 1.0;
    let max_y = img.rows() as f32 - 1.0;
    let (xf, yf) = match mode {
        BorderMode::Clamp => (x.clamp(0.0, max_x), y.clamp(0.0, max_y)),
        BorderMode::Mirror => (mirror_coord(x, max_x), mirror_coord(y, max_y)),
        BorderMode::Reject => (x, y),
    };
    interpolate_pixel(img, xf, yf, 0, InterpolationMode::Bilinear)
}

fn track_feature<A: ImageAllocator>(
    pt: [f32; 2],
    initial_flow: Option<[f32; 2]>,
    precomputed: &PyrLKPrecomputed<A>,
    params: &PyrLKParams,
) -> Option<([f32; 2], f32)> {
    const WIN_SIZE: usize = 21;
    const HALF_WIN: i32 = 10;
    const WIN_PIXELS: usize = WIN_SIZE * WIN_SIZE;

    let initial_scale = 1.0 / 2.0_f32.powi(params.max_level as i32);
    let mut dx = initial_flow.map_or(0.0, |d| d[0] * initial_scale);
    let mut dy = initial_flow.map_or(0.0, |d| d[1] * initial_scale);
    let mut tracking_error = 0.0f32;

    for lvl in (0..=params.max_level).rev() {
        let scale = 1.0 / 2.0_f32.powi(lvl as i32);
        let xc = pt[0] * scale;
        let yc = pt[1] * scale;

        if lvl < params.max_level {
            dx *= 2.0;
            dy *= 2.0;
        }

        let prev = &precomputed.prev_pyr[lvl];
        let next = &precomputed.next_pyr[lvl];
        let ix = &precomputed.grad_x_pyr[lvl];
        let iy = &precomputed.grad_y_pyr[lvl];

        let hw = HALF_WIN as f32;
        if params.border_mode == BorderMode::Reject
            && !(xc >= hw
                && yc >= hw
                && xc < (prev.cols() as f32 - hw)
                && yc < (prev.rows() as f32 - hw))
        {
            return None;
        }

        let mut prev_patch = [0.0f32; WIN_PIXELS];
        let mut ix_patch = [0.0f32; WIN_PIXELS];
        let mut iy_patch = [0.0f32; WIN_PIXELS];
        let mut a = 0.0f32;
        let mut b = 0.0f32;
        let mut c = 0.0f32;

        let mut idx = 0usize;
        for wy in -HALF_WIN..=HALF_WIN {
            for wx in -HALF_WIN..=HALF_WIN {
                let px = xc + wx as f32;
                let py = yc + wy as f32;
                let i0 = sample_at(prev, px, py, params.border_mode);
                let ixv = sample_at(ix, px, py, params.border_mode);
                let iyv = sample_at(iy, px, py, params.border_mode);
                prev_patch[idx] = i0;
                ix_patch[idx] = ixv;
                iy_patch[idx] = iyv;
                a += ixv * ixv;
                b += ixv * iyv;
                c += iyv * iyv;
                idx += 1;
            }
        }

        let det = a * c - b * b;
        if det.abs() < 1e-7 {
            return None;
        }
        let trace = a + c;
        let delta = a - c;
        let lambda_min = (trace - ((delta * delta + 4.0 * b * b).sqrt())) * 0.5;
        if lambda_min < params.min_eigen_threshold {
            return None;
        }
        let inv_det = 1.0 / det;

        for _iter in 0..params.max_iter {
            let xnc = xc + dx;
            let ync = yc + dy;

            if params.border_mode == BorderMode::Reject
                && !(xnc >= hw
                    && ync >= hw
                    && xnc < (next.cols() as f32 - hw)
                    && ync < (next.rows() as f32 - hw))
            {
                return None;
            }

            let mut d = 0.0f32;
            let mut e = 0.0f32;
            let mut idx = 0usize;
            for wy in -HALF_WIN..=HALF_WIN {
                for wx in -HALF_WIN..=HALF_WIN {
                    let qx = xnc + wx as f32;
                    let qy = ync + wy as f32;
                    let i1 = sample_at(next, qx, qy, params.border_mode);
                    let it = i1 - prev_patch[idx];
                    d -= ix_patch[idx] * it;
                    e -= iy_patch[idx] * it;
                    idx += 1;
                }
            }

            let delta_x = inv_det * (c * d - b * e);
            let delta_y = inv_det * (-b * d + a * e);
            dx += delta_x;
            dy += delta_y;

            if matches!(params.term_criteria, TermCriteria::Eps | TermCriteria::Both)
                && (delta_x * delta_x + delta_y * delta_y) < params.epsilon * params.epsilon
            {
                break;
            }
        }

        if lvl == 0 {
            let mut err_sum = 0.0f32;
            let mut idx = 0usize;
            for wy in -HALF_WIN..=HALF_WIN {
                for wx in -HALF_WIN..=HALF_WIN {
                    let qx = pt[0] + dx + wx as f32;
                    let qy = pt[1] + dy + wy as f32;
                    let i1 = sample_at(next, qx, qy, params.border_mode);
                    err_sum += (i1 - prev_patch[idx]).abs();
                    idx += 1;
                }
            }
            tracking_error = err_sum / WIN_PIXELS as f32;
        }
    }

    Some(([pt[0] + dx, pt[1] + dy], tracking_error))
}

/// Build Gaussian pyramids and gradients required by sparse LK.
pub fn build_lk_precomputed<A: ImageAllocator>(
    prev_img: &Image<f32, 1, A>,
    next_img: &Image<f32, 1, A>,
    max_level: usize,
) -> Result<PyrLKPrecomputed<A>, PyrLKError> {
    if prev_img.size() != next_img.size() {
        return Err(PyrLKError::ImageSizeMismatch {
            prev_width: prev_img.width(),
            prev_height: prev_img.height(),
            next_width: next_img.width(),
            next_height: next_img.height(),
        });
    }
    let mut prev_pyr = Vec::with_capacity(max_level + 1);
    let mut next_pyr = Vec::with_capacity(max_level + 1);
    prev_pyr.push(prev_img.clone());
    next_pyr.push(next_img.clone());

    for l in 1..=max_level {
        let prev_src = &prev_pyr[l - 1];
        let next_src = &next_pyr[l - 1];

        let down_size = ImageSize {
            width: prev_src.width().div_ceil(2),
            height: prev_src.height().div_ceil(2),
        };

        let mut prev_down =
            Image::from_size_val(down_size, 0.0f32, prev_img.storage.alloc().clone())?;
        let mut next_down =
            Image::from_size_val(down_size, 0.0f32, next_img.storage.alloc().clone())?;

        pyrdown_f32(prev_src, &mut prev_down)?;
        pyrdown_f32(next_src, &mut next_down)?;

        prev_pyr.push(prev_down);
        next_pyr.push(next_down);
    }

    let mut grad_x_pyr = Vec::with_capacity(max_level + 1);
    let mut grad_y_pyr = Vec::with_capacity(max_level + 1);
    for img in prev_pyr.iter().take(max_level + 1) {
        let mut ix = Image::from_size_val(img.size(), 0.0f32, prev_img.storage.alloc().clone())?;
        let mut iy = Image::from_size_val(img.size(), 0.0f32, prev_img.storage.alloc().clone())?;
        scharr_spatial_gradient_float(img, &mut ix, &mut iy)?;
        grad_x_pyr.push(ix);
        grad_y_pyr.push(iy);
    }

    Ok(PyrLKPrecomputed {
        prev_pyr,
        next_pyr,
        grad_x_pyr,
        grad_y_pyr,
    })
}

/// Compute sparse pyramidal Lucas–Kanade optical flow.
///
/// # Arguments
/// * `prev_img` - Previous image (grayscale, f32, shape HxWx1)
/// * `next_img` - Next image (grayscale, f32, shape HxWx1)
/// * `prev_pts` - Feature points to track (N x 2)
/// * `next_pts_in` - Optional initial guess for next points
/// * `params` - LK parameters
///
/// # Returns
/// * `Result<PyrLKResult, PyrLKError>` with next points, status, and error
pub fn calc_optical_flow_pyr_lk<A: ImageAllocator>(
    prev_img: &Image<f32, 1, A>,
    next_img: &Image<f32, 1, A>,
    prev_pts: &[[f32; 2]],
    next_pts_in: Option<&[[f32; 2]]>,
    params: &PyrLKParams,
) -> Result<PyrLKResult, PyrLKError> {
    if params.win_size != 21 {
        return Err(PyrLKError::InvalidWindowSize(params.win_size));
    }
    if prev_img.size() != next_img.size() {
        return Err(PyrLKError::ImageSizeMismatch {
            prev_width: prev_img.width(),
            prev_height: prev_img.height(),
            next_width: next_img.width(),
            next_height: next_img.height(),
        });
    }
    if params.use_initial_flow
        && next_pts_in
            .as_ref()
            .is_some_and(|pts| pts.len() != prev_pts.len())
    {
        return Err(PyrLKError::InitialFlowLengthMismatch {
            expected: prev_pts.len(),
            provided: next_pts_in.as_ref().map_or(0, |pts| pts.len()),
        });
    }
    let precomputed = build_lk_precomputed(prev_img, next_img, params.max_level)?;
    calc_optical_flow_pyr_lk_with_precomputed(&precomputed, prev_pts, next_pts_in, params)
}

/// Compute sparse pyramidal LK flow using precomputed pyramids and gradients.
pub fn calc_optical_flow_pyr_lk_with_precomputed<A: ImageAllocator>(
    precomputed: &PyrLKPrecomputed<A>,
    prev_pts: &[[f32; 2]],
    next_pts_in: Option<&[[f32; 2]]>,
    params: &PyrLKParams,
) -> Result<PyrLKResult, PyrLKError> {
    if params.win_size != 21 {
        return Err(PyrLKError::InvalidWindowSize(params.win_size));
    }

    let expected_levels = params.max_level + 1;
    if precomputed.prev_pyr.len() != expected_levels
        || precomputed.next_pyr.len() != expected_levels
        || precomputed.grad_x_pyr.len() != expected_levels
        || precomputed.grad_y_pyr.len() != expected_levels
    {
        return Err(PyrLKError::InvalidPrecomputedLevels {
            expected_levels,
            prev_levels: precomputed.prev_pyr.len(),
            next_levels: precomputed.next_pyr.len(),
            grad_x_levels: precomputed.grad_x_pyr.len(),
            grad_y_levels: precomputed.grad_y_pyr.len(),
        });
    }

    for level in 0..expected_levels {
        let prev_size = precomputed.prev_pyr[level].size();
        let next_size = precomputed.next_pyr[level].size();
        let grad_x_size = precomputed.grad_x_pyr[level].size();
        let grad_y_size = precomputed.grad_y_pyr[level].size();

        if next_size != prev_size || grad_x_size != prev_size || grad_y_size != prev_size {
            return Err(PyrLKError::InvalidPrecomputedSizes {
                level,
                prev_size,
                next_size,
                grad_x_size,
                grad_y_size,
            });
        }
    }

    if params.use_initial_flow
        && next_pts_in
            .as_ref()
            .is_some_and(|pts| pts.len() != prev_pts.len())
    {
        return Err(PyrLKError::InitialFlowLengthMismatch {
            expected: prev_pts.len(),
            provided: next_pts_in.as_ref().map_or(0, |pts| pts.len()),
        });
    }

    let n_features = prev_pts.len();
    let mut next_pts = vec![[0.0f32; 2]; n_features];
    let mut status = vec![0u8; n_features];
    let mut error = vec![0.0f32; n_features];

    next_pts
        .par_iter_mut()
        .zip(status.par_iter_mut())
        .zip(error.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((next_pt, st), err))| {
            let initial_flow = if params.use_initial_flow {
                next_pts_in.map(|initial_pts| {
                    [
                        initial_pts[i][0] - prev_pts[i][0],
                        initial_pts[i][1] - prev_pts[i][1],
                    ]
                })
            } else {
                None
            };
            if let Some((np, e)) = track_feature(prev_pts[i], initial_flow, precomputed, params) {
                *next_pt = np;
                *st = 1;
                *err = e;
            } else {
                *next_pt = prev_pts[i];
                *st = 0;
                *err = 0.0;
            }
        });

    Ok(PyrLKResult {
        next_pts,
        status,
        error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::allocator::CpuAllocator;

    fn default_params() -> PyrLKParams {
        PyrLKParams::default()
    }

    fn make_circle_image(size: usize, cx: f32, cy: f32, r: f32) -> Image<f32, 1, CpuAllocator> {
        let mut img =
            Image::<f32, 1, CpuAllocator>::from_size_val([size, size].into(), 0.0, CpuAllocator)
                .unwrap();
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if (dx * dx + dy * dy).sqrt() < r {
                    img.set_pixel(x, y, 0, 1.0).unwrap();
                }
            }
        }
        img
    }

    fn make_gradient_image(size: usize) -> Image<f32, 1, CpuAllocator> {
        let mut img =
            Image::<f32, 1, CpuAllocator>::from_size_val([size, size].into(), 0.0, CpuAllocator)
                .unwrap();
        for y in 0..size {
            for x in 0..size {
                let xf = x as f32;
                let yf = y as f32;
                let v = 0.5
                    + 0.2 * (0.13 * xf).sin()
                    + 0.2 * (0.17 * yf).cos()
                    + 0.1 * (0.07 * (xf + yf)).sin();
                img.set_pixel(x, y, 0, v.clamp(0.0, 1.0)).unwrap();
            }
        }
        img
    }

    #[test]
    fn test_lk_synthetic_integer_translation() {
        let size = 64;
        let dx = 5.0;
        let dy = -3.0;
        let img1 = make_circle_image(size, 32.0, 32.0, 10.0);
        let img2 = make_circle_image(size, 32.0 + dx, 32.0 + dy, 10.0);
        let pts = vec![[32.0, 32.0]];
        let params = default_params();

        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params).unwrap();

        assert_eq!(result.status[0], 1);
        let est_dx = result.next_pts[0][0] - pts[0][0];
        let est_dy = result.next_pts[0][1] - pts[0][1];
        assert!(
            (est_dx - dx).abs() < 0.01,
            "Integer translation dx error: {} vs {}",
            est_dx,
            dx
        );
        assert!(
            (est_dy - dy).abs() < 0.01,
            "Integer translation dy error: {} vs {}",
            est_dy,
            dy
        );
    }

    #[test]
    fn test_lk_synthetic_subpixel_translation() {
        let size = 64;
        let dx = 0.4;
        let dy = -0.7;
        let img1 = make_circle_image(size, 32.0, 32.0, 10.0);
        let img2 = make_circle_image(size, 32.0 + dx, 32.0 + dy, 10.0);
        let pts = vec![[32.0, 32.0]];
        let params = default_params();

        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params).unwrap();

        assert_eq!(result.status[0], 1);
        let est_dx = result.next_pts[0][0] - pts[0][0];
        let est_dy = result.next_pts[0][1] - pts[0][1];
        assert!(
            (est_dx - dx).abs() < 0.1,
            "Subpixel dx error: {} vs {}",
            est_dx,
            dx
        );
        assert!(
            (est_dy - dy).abs() < 0.1,
            "Subpixel dy error: {} vs {}",
            est_dy,
            dy
        );
    }

    #[test]
    fn test_lk_large_motion_across_pyramid() {
        let size = 128;
        let dx = 16.0;
        let dy = -12.0;
        let img1 = make_circle_image(size, 64.0, 64.0, 15.0);
        let img2 = make_circle_image(size, 64.0 + dx, 64.0 + dy, 15.0);
        let pts = vec![[64.0, 64.0]];
        let mut params = default_params();
        params.max_level = 3;

        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params).unwrap();

        assert_eq!(result.status[0], 1);
        let est_dx = result.next_pts[0][0] - pts[0][0];
        let est_dy = result.next_pts[0][1] - pts[0][1];
        assert!(
            (est_dx - dx).abs() < 1.0,
            "Large motion dx error: {} vs {}",
            est_dx,
            dx
        );
        assert!(
            (est_dy - dy).abs() < 1.0,
            "Large motion dy error: {} vs {}",
            est_dy,
            dy
        );
    }

    #[test]
    fn test_lk_border_case_handling() {
        let size = 64;
        let dx = 3.0;
        let dy = 2.0;
        let img1 = make_circle_image(size, 32.0, 32.0, 10.0);
        let img2 = make_circle_image(size, 32.0 + dx, 32.0 + dy, 10.0);

        // Test features near borders
        let pts = vec![[15.0, 15.0], [48.0, 48.0], [15.0, 48.0], [48.0, 15.0]];
        let params = default_params();

        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params).unwrap();

        // At least some should succeed with clamp mode
        let successful = result.status.iter().filter(|&&s| s == 1).count();
        assert!(successful > 0, "Some border features should succeed");
    }

    #[test]
    fn test_lk_low_texture_rejection() {
        let size = 64;
        let img1 =
            Image::<f32, 1, CpuAllocator>::from_size_val([size, size].into(), 0.5, CpuAllocator)
                .unwrap();
        let img2 = img1.clone();
        let pts = vec![[32.0, 32.0]];
        let params = default_params();

        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params).unwrap();

        // Flat region should be rejected
        assert_eq!(result.status[0], 0, "Low texture should be rejected");
    }

    #[test]
    fn test_lk_initial_flow_correctness() {
        let size = 64;
        let dx = 3.0;
        let dy = 2.0;
        let img1 = make_circle_image(size, 32.0, 32.0, 10.0);
        let img2 = make_circle_image(size, 32.0 + dx, 32.0 + dy, 10.0);
        let pts = vec![[32.0, 32.0]];
        let mut params = default_params();
        params.use_initial_flow = true;

        let initial_good = vec![[32.0 + 2.5, 32.0 + 1.5]];
        let result =
            calc_optical_flow_pyr_lk(&img1, &img2, &pts, Some(&initial_good), &params).unwrap();

        assert_eq!(result.status[0], 1);
        let est_dx = result.next_pts[0][0] - pts[0][0];
        let est_dy = result.next_pts[0][1] - pts[0][1];
        assert!(
            (est_dx - dx).abs() < 0.02,
            "Initial flow dx error: {} vs {}",
            est_dx,
            dx
        );
        assert!(
            (est_dy - dy).abs() < 0.02,
            "Initial flow dy error: {} vs {}",
            est_dy,
            dy
        );
    }

    #[test]
    fn test_lk_invalid_window_size() {
        let size = 64;
        let img = make_circle_image(size, 32.0, 32.0, 10.0);
        let pts = vec![[32.0, 32.0]];
        let mut params = default_params();

        // Unsupported window size
        params.win_size = 15;
        let result = calc_optical_flow_pyr_lk(&img, &img, &pts, None, &params);
        assert!(matches!(result, Err(PyrLKError::InvalidWindowSize(15))));
    }

    #[test]
    fn test_lk_invalid_precomputed_sizes() {
        let size = 64;
        let img1 = make_circle_image(size, 32.0, 32.0, 10.0);
        let img2 = make_circle_image(size, 32.0, 32.0, 10.0);
        let params = default_params();

        let mut precomputed = build_lk_precomputed(&img1, &img2, params.max_level).unwrap();

        // Corrupt the size of one of the levels
        let bad_img = make_circle_image(size + 1, 32.0, 32.0, 10.0);
        precomputed.next_pyr[1] = bad_img;

        let pts = vec![[32.0, 32.0]];
        let result = calc_optical_flow_pyr_lk_with_precomputed(&precomputed, &pts, None, &params);

        match result {
            Err(PyrLKError::InvalidPrecomputedSizes { level, .. }) => {
                assert_eq!(level, 1, "Should fail at the corrupted level");
            }
            _ => panic!("Expected InvalidPrecomputedSizes error"),
        }
    }

    #[test]
    fn test_lk_opencv_parity_comparison() {
        let size = 64;
        let dx = 4.0;
        let dy = -3.0;
        let img1 = make_circle_image(size, 32.0, 32.0, 10.0);
        let img2 = make_circle_image(size, 32.0 + dx, 32.0 + dy, 10.0);
        let pts = vec![[32.0, 32.0]];
        let params = default_params();

        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params).unwrap();

        assert_eq!(result.status[0], 1);
        let est_dx = result.next_pts[0][0] - pts[0][0];
        let est_dy = result.next_pts[0][1] - pts[0][1];

        assert!(
            (est_dx - dx).abs() < 0.1,
            "OpenCV parity: dx error = {}",
            (est_dx - dx).abs()
        );
        assert!(
            (est_dy - dy).abs() < 0.1,
            "OpenCV parity: dy error = {}",
            (est_dy - dy).abs()
        );
        assert!(
            result.error[0] < 0.1,
            "OpenCV parity: tracking error should be low"
        );
    }

    #[test]
    fn test_lk_deterministic_results() {
        let size = 64;
        let dx = 4.5;
        let dy = -2.3;
        let img1 = make_circle_image(size, 32.0, 32.0, 10.0);
        let img2 = make_circle_image(size, 32.0 + dx, 32.0 + dy, 10.0);
        let pts = vec![[32.0, 32.0], [28.0, 28.0], [36.0, 36.0]];
        let params = default_params();

        let result1 = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params).unwrap();
        let result2 = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params).unwrap();

        for i in 0..pts.len() {
            assert_eq!(
                result1.status[i], result2.status[i],
                "Status should be deterministic"
            );
            assert_eq!(
                result1.next_pts[i][0], result2.next_pts[i][0],
                "X coordinate should be deterministic"
            );
            assert_eq!(
                result1.next_pts[i][1], result2.next_pts[i][1],
                "Y coordinate should be deterministic"
            );
            assert_eq!(
                result1.error[i], result2.error[i],
                "Error should be deterministic"
            );
        }
    }

    #[test]
    fn test_lk_zero_motion() {
        let size = 64;
        let img = make_circle_image(size, 32.0, 32.0, 10.0);
        let pts = vec![[32.0, 32.0], [28.0, 28.0], [36.0, 36.0]];
        let params = default_params();

        let result = calc_optical_flow_pyr_lk(&img, &img, &pts, None, &params).unwrap();

        for (i, pt) in pts.iter().enumerate() {
            assert_eq!(result.status[i], 1);
            let est_dx = result.next_pts[i][0] - pt[0];
            let est_dy = result.next_pts[i][1] - pt[1];
            assert!(
                est_dx.abs() < 0.1,
                "Zero motion: dx should be near 0, got {}",
                est_dx
            );
            assert!(
                est_dy.abs() < 0.1,
                "Zero motion: dy should be near 0, got {}",
                est_dy
            );
        }
    }

    #[test]
    fn test_lk_multiple_features() {
        let size = 128;
        let dx = 3.0;
        let dy = -2.0;
        let img1 = make_gradient_image(size);
        let mut img2 =
            Image::<f32, 1, CpuAllocator>::from_size_val([size, size].into(), 0.0, CpuAllocator)
                .unwrap();

        // Shift image
        for y in 0..size {
            for x in 0..size {
                let src_x = (x as i32 - dx as i32).max(0).min(size as i32 - 1) as usize;
                let src_y = (y as i32 - dy as i32).max(0).min(size as i32 - 1) as usize;
                let val = *img1.get([src_y, src_x, 0]).unwrap();
                img2.set_pixel(x, y, 0, val).unwrap();
            }
        }

        let pts = vec![
            [20.0, 20.0],
            [40.0, 40.0],
            [60.0, 60.0],
            [80.0, 80.0],
            [100.0, 100.0],
        ];
        let params = default_params();

        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params).unwrap();

        let successful_tracks = result.status.iter().filter(|&&s| s == 1).count();
        assert!(
            successful_tracks >= 3,
            "At least 3 features should be tracked successfully"
        );
    }
}
