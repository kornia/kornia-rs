use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use kornia_tensor::CpuAllocator;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use super::{fast_horizontal_filter, kernels, separable_filter};

/// Blur an image using a box blur filter
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with shape (H, W, C).
/// * `kernel_size` - The size of the kernel (kernel_x, kernel_y).
///
/// PRECONDITION: `src` and `dst` must have the same shape.
pub fn box_blur<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    kernel_size: (usize, usize),
) -> Result<(), ImageError> {
    let kernel_x = kernels::box_blur_kernel_1d(kernel_size.0);
    let kernel_y = kernels::box_blur_kernel_1d(kernel_size.1);
    separable_filter(src, dst, &kernel_x, &kernel_y)?;
    Ok(())
}

/// Box blur for u8 images (NEON-accelerated separable path).
///
/// Same semantics as [`box_blur`] but operates directly on u8 via a Q8 uniform
/// kernel summing to 256, reusing the same striped ring-buffer path as
/// [`gaussian_blur_u8`]. The k=5 case hits the 32-u8/iter vmull/vmlal V-pass
/// unroll.
pub fn box_blur_u8<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, C, A1>,
    dst: &mut Image<u8, C, A2>,
    kernel_size: (usize, usize),
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let (kx, ky) = kernel_size;
    if kx == 0 || ky == 0 || kx % 2 == 0 || ky % 2 == 0 {
        return Err(ImageError::InvalidSigmaValue(kx as f32, ky as f32));
    }

    let ikx = quantize_kernel_256(&kernels::box_blur_kernel_1d(kx));
    let iky = quantize_kernel_256(&kernels::box_blur_kernel_1d(ky));

    separable_blur_u8_striped(
        src.as_slice(),
        dst.as_slice_mut(),
        src.rows(),
        src.cols(),
        C,
        &ikx,
        kx / 2,
        &iky,
        ky / 2,
    );

    Ok(())
}

/// Blur an image using a gaussian blur filter
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with shape (H, W, C).
/// * `kernel_size` - The size of the kernel (kernel_x, kernel_y). They can differ,
///   but they both have to be positive and odd. Or, they can be zero
///   and they will be computed from sigma values based on:
///   kernel_size = 8*sigma + 1
/// * `sigma` - The sigma of the gaussian kernel (sigma_x, sigma_y). sigma_y can
///   be zero and it will take on the same value as sigma_x. Or, they
///   can both be zero and they will be computed based on:
///   sigma = (kernel_size - 1) / 8
///
/// PRECONDITION: `src` and `dst` must have the same shape.
/// NOTE: This function uses a constant border type.
pub fn gaussian_blur<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    kernel_size: (usize, usize),
    sigma: (f32, f32),
) -> Result<(), ImageError> {
    let (mut kernel_x, mut kernel_y) = kernel_size;
    let (mut sigma_x, mut sigma_y) = sigma;

    // Satisfy setting sigma_y = sigma_x if sigma_y is zero.
    if sigma_y <= 0.0 {
        sigma_y = sigma_x;
    }

    // Auto-compute the kernel sizes based on sigma if 0 or negative using SciPy convention.
    // NOTE: the `| 1` is to ensure that the number is always odd i.e. the 2^0
    //       bit is always ON.
    if kernel_x == 0 && sigma_x > 0.0 {
        kernel_x = (2.0 * (4.0 * sigma_x).round() + 1.0) as usize | 1;
    }
    if kernel_y == 0 && sigma_y > 0.0 {
        kernel_y = (2.0 * (4.0 * sigma_y).round() + 1.0) as usize | 1;
    }
    if !(kernel_x > 0 && kernel_x % 2 == 1 && kernel_y > 0 && kernel_y % 2 == 1) {
        return Err(ImageError::InvalidSigmaValue(sigma_x, sigma_y));
    }

    // Sigma should be always positive.
    sigma_x = sigma_x.max(0.0);
    sigma_y = sigma_y.max(0.0);

    // Auto-compute the sigma values using SciPy convention.
    if sigma_x == 0.0 {
        sigma_x = (kernel_x as f32 - 1.0) / 8.0;
    }
    if sigma_y == 0.0 {
        sigma_y = (kernel_y as f32 - 1.0) / 8.0;
    }

    let kernel_x = kernels::gaussian_kernel_1d(kernel_x, sigma_x);
    let kernel_y = kernels::gaussian_kernel_1d(kernel_y, sigma_y);
    separable_filter(src, dst, &kernel_x, &kernel_y)?;

    Ok(())
}

/// Compute image edges using the Sobel operator
///
/// Applies the Sobel operator to compute edge detection by computing first-order derivatives.
/// This is a fundamental algorithm for feature detection in computer vision used for:
///
/// - **Robotics**: Line following, obstacle detection, edge-based navigation
/// - **Medical Imaging**: Feature extraction, boundary detection in scans
/// - **Industrial**: Defect detection, quality control
/// - **Computer Vision**: Feature detection, edge-based segmentation
///
/// ## Algorithm
///
/// The Sobel operator computes gradients in two directions:
/// ```text
/// Gx (vertical edges)    Gy (horizontal edges)
/// -1  0  1              -1 -2 -1
/// -2  0  2               0  0  0
/// -1  0  1               1  2  1
///
/// Output magnitude = sqrt(Gx² + Gy²)
/// ```
///
/// ## Performance
///
/// Typical latencies on modern hardware:
/// - 1080p (1920x1080) @ kernel_size=3: ~10-12ms
/// - 720p (1280x720)   @ kernel_size=3: ~5-6ms
/// - 4K (3840x2160)    @ kernel_size=3: ~40-50ms
///
/// FPS equivalents (throughput):
/// - 1080p @ 30 FPS: achieves ~80-100 FPS with 3x3 kernel
/// - 720p @ 30 FPS: achieves ~150-166 FPS with 3x3 kernel
///
/// Parallelized using rayon for multi-core processing.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C), type f32
/// * `dst` - The destination image with shape (H, W, C), type f32
///   Output contains edge magnitude at each pixel
/// * `kernel_size` - The size of the Sobel kernel: 1, 3, 5, or 7
///   - Smaller (1, 3): Precise edges, faster, more noise sensitivity
///   - Larger (5, 7): Smooth edges, slower, more noise robust
///
/// # Example
///
/// ```ignore
/// use kornia::image::Image;
/// use kornia::imgproc::filter::sobel;
/// use kornia::io::functional as F;
/// use kornia_tensor::CpuAllocator;
///
/// // Load image
/// let gray = F::read_image_any_gray8("robotics_line.png")?;
/// let gray_f32 = gray.map(|&x| x as f32 / 255.0)?;
///
/// // Apply Sobel edge detection
/// let mut edges = Image::from_size_val(gray_f32.size(), 0.0, CpuAllocator)?;
/// sobel(&gray_f32, &mut edges, 3)?;
///
/// // Save output
/// let edges_u8 = edges.map(|&x| (x * 255.0) as u8)?;
/// F::write_image(&edges_u8, "edges.png")?;
/// ```
///
/// # Error Handling
///
/// Returns an `ImageError` if:
/// - Input and output images have different shapes
/// - Invalid kernel size provided (must be 1, 3, 5, or 7)
/// - Memory allocation fails
///
/// # Preconditions
///
/// - `src` and `dst` must have the same shape (H, W, C)
/// - Input should be normalized (0.0 to 1.0 or similar range)
/// - For best results, input should be grayscale or single-channel
///
/// # See Also
///
/// - For robotics applications: See `examples/edge-detect-cli`
/// - For thresholding output: Use `kornia::imgproc::threshold`
/// - For morphological operations: Use `kornia::imgproc::morphology`
pub fn sobel<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    kernel_size: usize,
) -> Result<(), ImageError> {
    // get the sobel kernels
    let (kernel_x, kernel_y) = kernels::sobel_kernel_1d(kernel_size)?;

    // apply the sobel filter using separable filter
    let mut gx = Image::<f32, C, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
    separable_filter(src, &mut gx, &kernel_x, &kernel_y)?;

    let mut gy = Image::<f32, C, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
    separable_filter(src, &mut gy, &kernel_y, &kernel_x)?;

    // compute the magnitude in parallel by rows
    dst.as_slice_mut()
        .iter_mut()
        .zip(gx.as_slice().iter())
        .zip(gy.as_slice().iter())
        .for_each(|((dst, &gx), &gy)| {
            *dst = (gx * gx + gy * gy).sqrt();
        });

    Ok(())
}

/// Blur an image using a box blur filter multiple times to achieve a near gaussian blur
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with shape (H, W, C).
/// * `kernel_size` - The size of the kernel (kernel_x, kernel_y).
/// * `sigma` - The sigma of the gaussian kernel, xy-ordered.
///
/// PRECONDITION: `src` and `dst` must have the same shape.
pub fn box_blur_fast<const C: usize, A: ImageAllocator>(
    src: &Image<f32, C, A>,
    dst: &mut Image<f32, C, A>,
    sigma: (f32, f32),
) -> Result<(), ImageError> {
    let half_kernel_x_sizes = kernels::box_blur_fast_kernels_1d(sigma.0, 3);
    let half_kernel_y_sizes = kernels::box_blur_fast_kernels_1d(sigma.1, 3);

    let transposed_size = ImageSize {
        width: src.size().height,
        height: src.size().width,
    };

    let mut input_img = src;
    let mut transposed = Image::<f32, C, _>::from_size_val(transposed_size, 0.0, CpuAllocator)?;

    for (half_kernel_x_size, half_kernel_y_size) in
        half_kernel_x_sizes.iter().zip(half_kernel_y_sizes.iter())
    {
        fast_horizontal_filter(input_img, &mut transposed, *half_kernel_x_size)?;
        fast_horizontal_filter(&transposed, dst, *half_kernel_y_size)?;

        input_img = dst;
    }

    Ok(())
}

/// Compute the first order image derivative in both x and y using a Sobel operator.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W).
/// * `dst` - The destination image with shape (H, W, 2).
pub fn spatial_gradient_float<
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
    A3: ImageAllocator,
>(
    src: &Image<f32, C, A1>,
    dx: &mut Image<f32, C, A2>,
    dy: &mut Image<f32, C, A3>,
) -> Result<(), ImageError> {
    if src.size() != dx.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dx.cols(),
            dx.rows(),
        ));
    }

    if src.size() != dy.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dy.cols(),
            dy.rows(),
        ));
    }

    let (sobel_x, sobel_y) = kernels::normalized_sobel_kernel3();
    let cols = src.cols();

    let src_data = src.as_slice();

    dx.as_slice_mut()
        .chunks_mut(cols * C)
        .zip(dy.as_slice_mut().chunks_mut(cols * C))
        .enumerate()
        .for_each(|(r, (dx_row, dy_row))| {
            dx_row
                .chunks_mut(C)
                .zip(dy_row.chunks_mut(C))
                .enumerate()
                .for_each(|(c, (dx_c, dy_c))| {
                    let mut sum_x = [0.0; C];
                    let mut sum_y = [0.0; C];
                    for dy in 0..3 {
                        for dx in 0..3 {
                            let row = (r + dy).min(src.rows()).max(1) - 1;
                            let col = (c + dx).min(src.cols()).max(1) - 1;
                            for ch in 0..C {
                                let src_pix_offset = (row * src.cols() + col) * C + ch;
                                let val = unsafe { src_data.get_unchecked(src_pix_offset) };
                                sum_x[ch] += val * sobel_x[dy][dx];
                                sum_y[ch] += val * sobel_y[dy][dx];
                            }
                        }
                    }
                    dx_c.copy_from_slice(&sum_x);
                    dy_c.copy_from_slice(&sum_y);
                });
        });

    Ok(())
}

/// Compute the first order image derivative in both x and y using a Sobel operator.
/// Parallel by row.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W).
/// * `dst` - The destination image with shape (H, W, 2).
pub fn spatial_gradient_float_parallel_row<
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
    A3: ImageAllocator,
>(
    src: &Image<f32, C, A1>,
    dx: &mut Image<f32, C, A2>,
    dy: &mut Image<f32, C, A3>,
) -> Result<(), ImageError> {
    if src.size() != dx.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dx.cols(),
            dx.rows(),
        ));
    }

    if src.size() != dy.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dy.cols(),
            dy.rows(),
        ));
    }

    let (sobel_x, sobel_y) = kernels::normalized_sobel_kernel3();
    let cols = src.cols();

    let src_data = src.as_slice();

    dx.as_slice_mut()
        .par_chunks_mut(cols * C)
        .zip(dy.as_slice_mut().par_chunks_mut(cols * C))
        .enumerate()
        .for_each(|(r, (dx_row, dy_row))| {
            dx_row
                .chunks_mut(C)
                .zip(dy_row.chunks_mut(C))
                .enumerate()
                .for_each(|(c, (dx_c, dy_c))| {
                    let mut sum_x = [0.0; C];
                    let mut sum_y = [0.0; C];
                    for dy in 0..3 {
                        for dx in 0..3 {
                            let row = (r + dy).min(src.rows()).max(1) - 1;
                            let col = (c + dx).min(src.cols()).max(1) - 1;
                            for ch in 0..C {
                                let src_pix_offset = (row * src.cols() + col) * C + ch;
                                let val = unsafe { src_data.get_unchecked(src_pix_offset) };
                                sum_x[ch] += val * sobel_x[dy][dx];
                                sum_y[ch] += val * sobel_y[dy][dx];
                            }
                        }
                    }
                    dx_c.copy_from_slice(&sum_x);
                    dy_c.copy_from_slice(&sum_y);
                });
        });

    Ok(())
}

/// Compute the first order image derivative in both x and y using a Sobel operator.
/// Parallel both by row and col.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W).
/// * `dst` - The destination image with shape (H, W, 2).
pub fn spatial_gradient_float_parallel<
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
    A3: ImageAllocator,
>(
    src: &Image<f32, C, A1>,
    dx: &mut Image<f32, C, A2>,
    dy: &mut Image<f32, C, A3>,
) -> Result<(), ImageError> {
    if src.size() != dx.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dx.cols(),
            dx.rows(),
        ));
    }

    if src.size() != dy.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dy.cols(),
            dy.rows(),
        ));
    }

    let (sobel_x, sobel_y) = kernels::normalized_sobel_kernel3();
    let cols = src.cols();

    let src_data = src.as_slice();

    dx.as_slice_mut()
        .par_chunks_mut(cols * C)
        .zip(dy.as_slice_mut().par_chunks_mut(cols * C))
        .enumerate()
        .for_each(|(r, (dx_row, dy_row))| {
            dx_row
                .par_chunks_mut(C)
                .zip(dy_row.par_chunks_mut(C))
                .enumerate()
                .for_each(|(c, (dx_c, dy_c))| {
                    let mut sum_x = [0.0; C];
                    let mut sum_y = [0.0; C];
                    for dy in 0..3 {
                        for dx in 0..3 {
                            let row = (r + dy).min(src.rows()).max(1) - 1;
                            let col = (c + dx).min(src.cols()).max(1) - 1;
                            for ch in 0..C {
                                let src_pix_offset = (row * src.cols() + col) * C + ch;
                                let val = unsafe { src_data.get_unchecked(src_pix_offset) };
                                sum_x[ch] += val * sobel_x[dy][dx];
                                sum_y[ch] += val * sobel_y[dy][dx];
                            }
                        }
                    }
                    dx_c.copy_from_slice(&sum_x);
                    dy_c.copy_from_slice(&sum_y);
                });
        });

    Ok(())
}

type GaussianParams = ((usize, usize), (f32, f32));

/// Resolve gaussian parameters to a validated (kernel_size, sigma) pair.
/// Mirrors the auto-compute logic in `gaussian_blur` (f32 path).
fn resolve_gaussian_params(
    kernel_size: (usize, usize),
    sigma: (f32, f32),
) -> Result<GaussianParams, ImageError> {
    let (mut kx, mut ky) = kernel_size;
    let (mut sx, mut sy) = sigma;

    if sy <= 0.0 {
        sy = sx;
    }
    if kx == 0 && sx > 0.0 {
        kx = (2.0 * (4.0 * sx).round() + 1.0) as usize | 1;
    }
    if ky == 0 && sy > 0.0 {
        ky = (2.0 * (4.0 * sy).round() + 1.0) as usize | 1;
    }
    if !(kx > 0 && kx % 2 == 1 && ky > 0 && ky % 2 == 1) {
        return Err(ImageError::InvalidSigmaValue(sx, sy));
    }

    sx = sx.max(0.0);
    sy = sy.max(0.0);
    if sx == 0.0 {
        sx = (kx as f32 - 1.0) / 8.0;
    }
    if sy == 0.0 {
        sy = (ky as f32 - 1.0) / 8.0;
    }

    Ok(((kx, ky), (sx, sy)))
}

/// Gaussian blur for u8 images (NEON-accelerated separable path).
///
/// Same parameter semantics as [`gaussian_blur`] but operates directly on u8,
/// avoiding u8→f32 round-trips. Uses Q8 quantized 1D kernels summing to 256
/// so the final right-shift by 16 produces the clamped u8 result.
///
/// * `src` - Source u8 image.
/// * `dst` - Destination u8 image (same size as src).
/// * `kernel_size` - Kernel size (width, height). Must be odd and positive.
///   If zero, it is computed from `sigma`.
/// * `sigma` - Gaussian sigma (sigma_x, sigma_y). If zero, it is computed
///   from `kernel_size`.
pub fn gaussian_blur_u8<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, C, A1>,
    dst: &mut Image<u8, C, A2>,
    kernel_size: (usize, usize),
    sigma: (f32, f32),
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let ((kx, ky), (sx, sy)) = resolve_gaussian_params(kernel_size, sigma)?;

    // Binomial fast path for the common 5x5, sigma≈1 case. Two separable
    // passes of [1,2,1]/4 convolve to [1,4,6,4,1]/16 — a close approximation
    // of a true Gaussian with sigma≈1.0 that stays in u8 throughout (halving
    // adds, no widen/pack). This is what OpenCV's cv2.GaussianBlur uses
    // internally when sigma=0.
    #[cfg(target_arch = "aarch64")]
    if kx == 5 && ky == 5 && (0.7..=1.3).contains(&sx) && (0.7..=1.3).contains(&sy) {
        gaussian_blur_5x5_binomial_u8::<C>(
            src.as_slice(),
            dst.as_slice_mut(),
            src.rows(),
            src.cols(),
        );
        return Ok(());
    }

    let ikx = quantize_kernel_256(&kernels::gaussian_kernel_1d(kx, sx));
    let iky = quantize_kernel_256(&kernels::gaussian_kernel_1d(ky, sy));

    // Unified path: per-thread u8 ring buffer (Q8+Q8 decimated).
    //
    // Note: an earlier k=5 register-rolling specialization was tried but
    // consistently ran ~2.8× slower than this general path on A78AE — the
    // column-major writes broke write-combining and the primed H-pass
    // per-column overhead dominated. Removed 2026-04-18.
    separable_blur_u8_striped(
        src.as_slice(),
        dst.as_slice_mut(),
        src.rows(),
        src.cols(),
        C,
        &ikx,
        kx / 2,
        &iky,
        ky / 2,
    );

    Ok(())
}
/// Quantize a float kernel (summing to 1.0) to u8 weights summing to 256.
fn quantize_kernel_256(kernel: &[f32]) -> Vec<u8> {
    let half = kernel.len() / 2;
    let mut ik: Vec<u8> = kernel.iter().map(|&k| (k * 256.0 + 0.5) as u8).collect();
    // Fix rounding: adjust center weight so weights sum to exactly 256
    let sum: u16 = ik.iter().map(|&w| w as u16).sum();
    if sum != 256 {
        let center = ik[half] as i16 + (256 - sum as i16);
        ik[half] = center.clamp(0, 255) as u8;
    }
    ik
}

/// Horizontal pass for a single source row (clamped to image bounds),
/// writing `stride` u8 results (post Q8 shift-right-8) into `dst_row`.
///
/// Uses Q8 split — instead of storing the full Q16 accumulator (u16),
/// we shift by 8 and store u8. This halves the ring-buffer footprint and
/// lets the V-pass reuse the cheap `vmull_u8 / vmlal_u8` path (32 u8 per
/// iter vs 16). The precision cost is ≤1 LSB per pass (≤2 total vs
/// u16-intermediate).
#[inline(always)]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn hpass_u8_row(
    src: &[u8],
    src_row_abs: isize,
    rows: usize,
    cols: usize,
    channels: usize,
    stride: usize,
    half_x: usize,
    kernel_x: &[u8],
    #[cfg(target_arch = "aarch64")] kvecs_x: &[std::arch::aarch64::uint8x8_t],
    padded: &mut [u8],
    dst_row: &mut [u8],
) {
    let ksize_x = kernel_x.len();
    let src_r = src_row_abs.max(0).min(rows as isize - 1) as usize;
    let row_src = &src[src_r * stride..(src_r + 1) * stride];
    let first_px = &row_src[0..channels];
    let last_px = &row_src[(cols - 1) * channels..cols * channels];
    for i in 0..half_x {
        padded[i * channels..(i + 1) * channels].copy_from_slice(first_px);
        let off = (half_x + cols + i) * channels;
        padded[off..off + channels].copy_from_slice(last_px);
    }
    padded[half_x * channels..(half_x + cols) * channels].copy_from_slice(row_src);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let pp = padded.as_ptr();
        let dp = dst_row.as_mut_ptr();
        let bulk16 = stride & !15;
        let mut j = 0usize;
        while j < bulk16 {
            let s0 = vld1q_u8(pp.add(j));
            let mut acc_lo = vmull_u8(vget_low_u8(s0), kvecs_x[0]);
            let mut acc_hi = vmull_u8(vget_high_u8(s0), kvecs_x[0]);
            for ki in 1..ksize_x {
                let sk = vld1q_u8(pp.add(j + ki * channels));
                acc_lo = vmlal_u8(acc_lo, vget_low_u8(sk), kvecs_x[ki]);
                acc_hi = vmlal_u8(acc_hi, vget_high_u8(sk), kvecs_x[ki]);
            }
            let packed = vcombine_u8(vshrn_n_u16(acc_lo, 8), vshrn_n_u16(acc_hi, 8));
            vst1q_u8(dp.add(j), packed);
            j += 16;
        }
        while j < stride {
            let mut a = 0u32;
            for ki in 0..ksize_x {
                a += *pp.add(j + ki * channels) as u32 * kernel_x[ki] as u32;
            }
            *dp.add(j) = (a >> 8) as u8;
            j += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for j in 0..stride {
            let mut acc = 0u32;
            for ki in 0..ksize_x {
                acc += padded[j + ki * channels] as u32 * kernel_x[ki] as u32;
            }
            dst_row[j] = (acc >> 8) as u8;
        }
    }
}

/// Fused separable blur with per-thread u8 ring buffer (Q8+Q8 decimated).
///
/// For each output row `r`, the V-pass needs H-pass results for source rows
/// `r - half_y ..= r + half_y`. Instead of pre-computing the entire H-pass
/// into a large u16 temp buffer (which busts L2 at 1080p — ~2 MB/thread),
/// we keep a rolling window of exactly `ksize_y` H-passed rows in a ring.
/// The H-pass stores `>>8` of its u16 accumulator, so the ring holds u8
/// (~29 KB at 1080p 5×5, L1-resident). The V-pass reuses the cheap
/// `vmull_u8 / vmlal_u8` path — 32 u8 per iter, half the memory traffic
/// of a u16 ring. The intermediate truncation costs ≤2 LSB of precision
/// vs the full Q16 path; acceptable for image filtering.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn separable_blur_u8_striped(
    src: &[u8],
    dst: &mut [u8],
    rows: usize,
    cols: usize,
    channels: usize,
    kernel_x: &[u8],
    half_x: usize,
    kernel_y: &[u8],
    half_y: usize,
) {
    use rayon::prelude::*;

    let stride = cols * channels;
    let ksize_y = kernel_y.len();
    let padded_stride = (cols + 2 * half_x) * channels;

    // One strip per rayon worker — pure parallel partition.
    let nthreads = rayon::current_num_threads().max(1);
    let strip_h = rows.div_ceil(nthreads).max(1).min(rows);

    let strips: Vec<(usize, usize)> = (0..rows)
        .step_by(strip_h)
        .map(|start| (start, (start + strip_h).min(rows)))
        .collect();

    let mut strip_slices: Vec<&mut [u8]> = dst.chunks_mut(strip_h * stride).collect();

    strip_slices.par_iter_mut().zip(strips.par_iter()).for_each(
        |(strip_dst, &(out_start, out_end))| {
            let out_rows = out_end - out_start;

            // u8 ring buffer: `ksize_y` H-passed-and-shifted rows. Source row
            // with absolute index `r` lives at slot `r.rem_euclid(ksize_y)`.
            let ring_len = ksize_y * stride;
            let mut ring: Vec<u8> = Vec::with_capacity(ring_len);
            // Safety: every slot is written by an H-pass before any V-pass read.
            #[allow(clippy::uninit_vec)]
            unsafe {
                ring.set_len(ring_len)
            };

            let mut padded = vec![0u8; padded_stride];

            #[cfg(target_arch = "aarch64")]
            let kvecs_x: Vec<_> = kernel_x
                .iter()
                .map(|&w| unsafe { std::arch::aarch64::vdup_n_u8(w) })
                .collect();
            #[cfg(target_arch = "aarch64")]
            let kvecs_y: Vec<_> = kernel_y
                .iter()
                .map(|&w| unsafe { std::arch::aarch64::vdup_n_u8(w) })
                .collect();

            // Prime: H-pass source rows `out_start - half_y .. out_start + half_y`
            // (inclusive) into ring. After this, all taps for output row 0 are resident.
            for k in 0..ksize_y {
                let r_abs = out_start as isize + k as isize - half_y as isize;
                let slot = r_abs.rem_euclid(ksize_y as isize) as usize;
                let dst_row = &mut ring[slot * stride..(slot + 1) * stride];
                hpass_u8_row(
                    src,
                    r_abs,
                    rows,
                    cols,
                    channels,
                    stride,
                    half_x,
                    kernel_x,
                    #[cfg(target_arch = "aarch64")]
                    &kvecs_x,
                    &mut padded,
                    dst_row,
                );
            }

            for oi in 0..out_rows {
                // Evict the oldest tap and install the new one (same ring slot).
                if oi > 0 {
                    let r_new = out_start as isize + oi as isize + half_y as isize;
                    let slot = r_new.rem_euclid(ksize_y as isize) as usize;
                    let dst_row = &mut ring[slot * stride..(slot + 1) * stride];
                    hpass_u8_row(
                        src,
                        r_new,
                        rows,
                        cols,
                        channels,
                        stride,
                        half_x,
                        kernel_x,
                        #[cfg(target_arch = "aarch64")]
                        &kvecs_x,
                        &mut padded,
                        dst_row,
                    );
                }

                // Tap pointers: tap k = src row (out_start + oi - half_y + k),
                // at ring slot `that.rem_euclid(ksize_y)`.
                let mut tap_ptrs: [*const u8; 32] = [std::ptr::null(); 32];
                let ring_ptr = ring.as_ptr();
                for k in 0..ksize_y {
                    let r_abs = out_start as isize + oi as isize - half_y as isize + k as isize;
                    let slot = r_abs.rem_euclid(ksize_y as isize) as usize;
                    tap_ptrs[k] = unsafe { ring_ptr.add(slot * stride) };
                }

                let out_row = &mut strip_dst[oi * stride..(oi + 1) * stride];

                #[cfg(target_arch = "aarch64")]
                unsafe {
                    use std::arch::aarch64::*;
                    let dp = out_row.as_mut_ptr();
                    let bulk16 = stride & !15;
                    let mut j = 0usize;

                    if ksize_y == 5 {
                        // 32-byte-per-iter V-pass for the common 5-tap case.
                        // Hoisting taps + kernel vectors out of the inner loop
                        // and emitting 10 loads + 10 vmull/vmlal lets the OoO
                        // core overlap 2 independent 16-byte lanes per iter.
                        let t0 = tap_ptrs[0];
                        let t1 = tap_ptrs[1];
                        let t2 = tap_ptrs[2];
                        let t3 = tap_ptrs[3];
                        let t4 = tap_ptrs[4];
                        let k0 = kvecs_y[0];
                        let k1 = kvecs_y[1];
                        let k2 = kvecs_y[2];
                        let k3 = kvecs_y[3];
                        let k4 = kvecs_y[4];
                        let bulk32 = stride & !31;
                        while j < bulk32 {
                            let a0 = vld1q_u8(t0.add(j));
                            let a1 = vld1q_u8(t1.add(j));
                            let a2 = vld1q_u8(t2.add(j));
                            let a3 = vld1q_u8(t3.add(j));
                            let a4 = vld1q_u8(t4.add(j));
                            let b0 = vld1q_u8(t0.add(j + 16));
                            let b1 = vld1q_u8(t1.add(j + 16));
                            let b2 = vld1q_u8(t2.add(j + 16));
                            let b3 = vld1q_u8(t3.add(j + 16));
                            let b4 = vld1q_u8(t4.add(j + 16));
                            let mut al = vmull_u8(vget_low_u8(a0), k0);
                            let mut ah = vmull_u8(vget_high_u8(a0), k0);
                            let mut bl = vmull_u8(vget_low_u8(b0), k0);
                            let mut bh = vmull_u8(vget_high_u8(b0), k0);
                            al = vmlal_u8(al, vget_low_u8(a1), k1);
                            ah = vmlal_u8(ah, vget_high_u8(a1), k1);
                            bl = vmlal_u8(bl, vget_low_u8(b1), k1);
                            bh = vmlal_u8(bh, vget_high_u8(b1), k1);
                            al = vmlal_u8(al, vget_low_u8(a2), k2);
                            ah = vmlal_u8(ah, vget_high_u8(a2), k2);
                            bl = vmlal_u8(bl, vget_low_u8(b2), k2);
                            bh = vmlal_u8(bh, vget_high_u8(b2), k2);
                            al = vmlal_u8(al, vget_low_u8(a3), k3);
                            ah = vmlal_u8(ah, vget_high_u8(a3), k3);
                            bl = vmlal_u8(bl, vget_low_u8(b3), k3);
                            bh = vmlal_u8(bh, vget_high_u8(b3), k3);
                            al = vmlal_u8(al, vget_low_u8(a4), k4);
                            ah = vmlal_u8(ah, vget_high_u8(a4), k4);
                            bl = vmlal_u8(bl, vget_low_u8(b4), k4);
                            bh = vmlal_u8(bh, vget_high_u8(b4), k4);
                            vst1q_u8(
                                dp.add(j),
                                vcombine_u8(vshrn_n_u16(al, 8), vshrn_n_u16(ah, 8)),
                            );
                            vst1q_u8(
                                dp.add(j + 16),
                                vcombine_u8(vshrn_n_u16(bl, 8), vshrn_n_u16(bh, 8)),
                            );
                            j += 32;
                        }
                        while j < bulk16 {
                            let s0 = vld1q_u8(t0.add(j));
                            let s1 = vld1q_u8(t1.add(j));
                            let s2 = vld1q_u8(t2.add(j));
                            let s3 = vld1q_u8(t3.add(j));
                            let s4 = vld1q_u8(t4.add(j));
                            let mut al = vmull_u8(vget_low_u8(s0), k0);
                            let mut ah = vmull_u8(vget_high_u8(s0), k0);
                            al = vmlal_u8(al, vget_low_u8(s1), k1);
                            ah = vmlal_u8(ah, vget_high_u8(s1), k1);
                            al = vmlal_u8(al, vget_low_u8(s2), k2);
                            ah = vmlal_u8(ah, vget_high_u8(s2), k2);
                            al = vmlal_u8(al, vget_low_u8(s3), k3);
                            ah = vmlal_u8(ah, vget_high_u8(s3), k3);
                            al = vmlal_u8(al, vget_low_u8(s4), k4);
                            ah = vmlal_u8(ah, vget_high_u8(s4), k4);
                            vst1q_u8(
                                dp.add(j),
                                vcombine_u8(vshrn_n_u16(al, 8), vshrn_n_u16(ah, 8)),
                            );
                            j += 16;
                        }
                    } else {
                        // 16 u8 per iter, using vmull/vmlal u8 — same pattern as H-pass.
                        while j < bulk16 {
                            let t0 = tap_ptrs[0];
                            let s0 = vld1q_u8(t0.add(j));
                            let mut acc_lo = vmull_u8(vget_low_u8(s0), kvecs_y[0]);
                            let mut acc_hi = vmull_u8(vget_high_u8(s0), kvecs_y[0]);
                            for ki in 1..ksize_y {
                                let tk = tap_ptrs[ki];
                                let sk = vld1q_u8(tk.add(j));
                                acc_lo = vmlal_u8(acc_lo, vget_low_u8(sk), kvecs_y[ki]);
                                acc_hi = vmlal_u8(acc_hi, vget_high_u8(sk), kvecs_y[ki]);
                            }
                            let packed =
                                vcombine_u8(vshrn_n_u16(acc_lo, 8), vshrn_n_u16(acc_hi, 8));
                            vst1q_u8(dp.add(j), packed);
                            j += 16;
                        }
                    }

                    while j < stride {
                        let mut acc = 0u32;
                        for ki in 0..ksize_y {
                            acc += *tap_ptrs[ki].add(j) as u32 * kernel_y[ki] as u32;
                        }
                        *dp.add(j) = (acc >> 8) as u8;
                        j += 1;
                    }
                }

                #[cfg(not(target_arch = "aarch64"))]
                {
                    for j in 0..stride {
                        let mut acc = 0u32;
                        for ki in 0..ksize_y {
                            acc += unsafe { *tap_ptrs[ki].add(j) } as u32 * kernel_y[ki] as u32;
                        }
                        out_row[j] = (acc >> 8) as u8;
                    }
                }
            }
        },
    );
}

/// 5x5 binomial Gaussian approximation for u8 using halving-add cascades.
///
/// Convolves the image with [1,4,6,4,1]/16 in both dimensions, implemented as
/// two separable passes of [1,2,1]/4. The 3-tap pass is `(a + 2b + c + 2) / 4`,
/// computed via `vrhaddq_u8(vrhaddq_u8(a,b), vrhaddq_u8(b,c))`. This yields
/// `(a + 2b + c + 3) / 4`, within ±1 LSB of the true binomial — the same trick
/// OpenCV uses in its `cv2.GaussianBlur(sigma=0)` fast path.
///
/// Parallelized by row chunks; each worker keeps a rolling 3-row ring of H-passed
/// rows in ~10-20 KB of L1 so the tmp buffer never hits DRAM.
#[cfg(target_arch = "aarch64")]
fn gaussian_blur_5x5_binomial_u8<const C: usize>(
    src: &[u8],
    dst: &mut [u8],
    rows: usize,
    cols: usize,
) {
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use rayon::slice::ParallelSliceMut;

    let stride = cols * C;

    // Small image: single-thread to skip rayon overhead.
    if rows * cols < 256 * 1024 {
        binomial_blur_chunk::<C>(src, dst, rows, cols, 0, rows);
        return;
    }

    // Row-chunked parallelism. Each chunk does its own H-pass with 2-row overlap
    // at boundaries (redundant work is O(rows/chunk), negligible).
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_rows = rows.div_ceil(n_threads).max(16);

    dst.par_chunks_mut(chunk_rows * stride)
        .enumerate()
        .for_each(|(chunk_idx, dst_chunk)| {
            let start = chunk_idx * chunk_rows;
            let end = (start + dst_chunk.len() / stride).min(rows);
            binomial_blur_chunk::<C>(src, dst_chunk, rows, cols, start, end);
        });
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn binomial_blur_chunk<const C: usize>(
    src: &[u8],
    dst_chunk: &mut [u8],
    rows: usize,
    cols: usize,
    start: usize,
    end: usize,
) {
    if start >= end {
        return;
    }
    let stride = cols * C;
    // 3-row ring. Fits in L1 for row up to ~10K pixels.
    let mut ring = vec![0u8; 3 * stride];

    let hprow = |y: usize, out: &mut [u8]| {
        let src_row = &src[y * stride..y * stride + stride];
        hpass_binomial_row::<C>(src_row, out, cols);
    };

    // Prefill ring[0] = H(start - 1 clamped), ring[1] = H(start).
    {
        let (r0, rest) = ring.split_at_mut(stride);
        let (r1, _r2) = rest.split_at_mut(stride);
        hprow(start.saturating_sub(1), r0);
        hprow(start, r1);
    }

    for (i, drow) in dst_chunk.chunks_exact_mut(stride).enumerate() {
        let y = start + i;
        let yp1 = (y + 1).min(rows - 1);
        let new_slot = (i + 2) % 3;
        // Write ring[new_slot] = H(yp1)
        let (new_lo, new_hi) = (new_slot * stride, new_slot * stride + stride);
        hprow(yp1, &mut ring[new_lo..new_hi]);

        // Read top=H(y-1), mid=H(y), bot=H(yp1) for V-pass
        let ring_view: &[u8] = &ring;
        let top_slot = i % 3;
        let mid_slot = (i + 1) % 3;
        let bot_slot = new_slot;
        let top = &ring_view[top_slot * stride..top_slot * stride + stride];
        let mid = &ring_view[mid_slot * stride..mid_slot * stride + stride];
        let bot = &ring_view[bot_slot * stride..bot_slot * stride + stride];
        vpass_binomial_row(top, mid, bot, drow);

        // If y is the last row, bot must already equal mid (yp1 clamps).
        // If i == 0 and start == 0, top must equal mid (y-1 clamps to 0).
        // Both are handled by our saturating_sub / min calls above.
        let _ = end;
    }
}

/// H-pass [1,2,1]/4 on one row. Edge pixels replicate.
/// For interleaved C-channel RGB storage, byte-strided by C gives per-channel neighbors.
#[cfg(target_arch = "aarch64")]
#[inline]
fn hpass_binomial_row<const C: usize>(src: &[u8], dst: &mut [u8], cols: usize) {
    use std::arch::aarch64::*;
    let stride = cols * C;

    // First pixel: left neighbor replicates to self.
    for c in 0..C {
        let a = src[c] as u16;
        let b = a;
        let d = src[C + c] as u16;
        dst[c] = ((a + 2 * b + d + 2) >> 2) as u8;
    }

    // Middle bytes: compute (src[i-C] + 2*src[i] + src[i+C] + 2) / 4 per byte.
    unsafe {
        let mut i = C;
        // Vector bulk: need i >= C and i + 16 + C <= stride.
        while i + 16 + C <= stride {
            let a = vld1q_u8(src.as_ptr().add(i - C));
            let b = vld1q_u8(src.as_ptr().add(i));
            let d = vld1q_u8(src.as_ptr().add(i + C));
            let ab = vrhaddq_u8(a, b);
            let bd = vrhaddq_u8(b, d);
            let out = vrhaddq_u8(ab, bd);
            vst1q_u8(dst.as_mut_ptr().add(i), out);
            i += 16;
        }
        // Scalar tail up to stride - C (last pixel handled below).
        while i + C < stride {
            let a = src[i - C] as u16;
            let b = src[i] as u16;
            let d = src[i + C] as u16;
            dst[i] = ((a + 2 * b + d + 2) >> 2) as u8;
            i += 1;
        }
    }

    // Last pixel: right neighbor replicates to self.
    for c in 0..C {
        let a = src[stride - 2 * C + c] as u16;
        let b = src[stride - C + c] as u16;
        let d = b;
        dst[stride - C + c] = ((a + 2 * b + d + 2) >> 2) as u8;
    }
}

/// V-pass [1,2,1]/4 between 3 rows. All rows same length; byte-wise halving adds.
#[cfg(target_arch = "aarch64")]
#[inline]
fn vpass_binomial_row(top: &[u8], mid: &[u8], bot: &[u8], dst: &mut [u8]) {
    use std::arch::aarch64::*;
    let stride = dst.len();
    unsafe {
        let mut i = 0;
        while i + 16 <= stride {
            let a = vld1q_u8(top.as_ptr().add(i));
            let b = vld1q_u8(mid.as_ptr().add(i));
            let d = vld1q_u8(bot.as_ptr().add(i));
            let ab = vrhaddq_u8(a, b);
            let bd = vrhaddq_u8(b, d);
            let out = vrhaddq_u8(ab, bd);
            vst1q_u8(dst.as_mut_ptr().add(i), out);
            i += 16;
        }
        while i < stride {
            let a = top[i] as u16;
            let b = mid[i] as u16;
            let d = bot[i] as u16;
            dst[i] = ((a + 2 * b + d + 2) >> 2) as u8;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_blur_fast() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        let img = Image::new(size, (0..25).map(|x| x as f32).collect(), CpuAllocator)?;
        let mut dst = Image::<_, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;

        box_blur_fast(&img, &mut dst, (0.5, 0.5))?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[
                4.444444, 4.9259257, 5.7037034, 6.4814816, 6.962963,
                6.851851, 7.3333335, 8.111111, 8.888889, 9.370372,
                10.740741, 11.222222, 12.0, 12.777779, 13.259262,
                14.629628, 15.111112, 15.888888, 16.666666, 17.14815,
                17.037035, 17.518518, 18.296295, 19.074074, 19.555555,
            ],
        );

        Ok(())
    }

    #[test]
    fn test_gaussian_blur() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        let img = Image::new(size, (0..25).map(|x| x as f32).collect(), CpuAllocator)?;

        let mut dst = Image::<_, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;

        gaussian_blur(&img, &mut dst, (3, 3), (0.5, 0.5))?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[
                0.57097936, 1.4260278, 2.3195207, 3.213014, 3.5739717,
                4.5739717, 5.999999, 7.0, 7.999999, 7.9349294,
                9.041435, 10.999999, 12.0, 12.999998, 12.402394,
                13.5089, 15.999998, 17.0, 17.999996, 16.86986,
                15.58594, 18.230816, 19.124311, 20.017801, 18.588936,
            ]
        );
        Ok(())
    }

    #[test]
    fn test_gaussian_blur_autocompute_ksize() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        let img = Image::new(size, (0..25).map(|x| x as f32).collect(), CpuAllocator)?;

        let mut dst = Image::<_, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;

        gaussian_blur(&img, &mut dst, (0, 0), (0.5, 0.5))?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[0.573374, 1.4282724, 2.3214629, 3.2134287, 3.5740836,
              4.5745554, 5.999999, 7.000791, 7.997888, 7.9328527,
              9.039831, 10.997623, 11.999999, 12.996041, 12.399015,
              13.500337, 15.989445, 16.992872, 17.987333, 16.858635,
              15.576923, 18.21976, 19.117384, 20.004917, 18.577633,
            ]
        );
        Ok(())
    }

    #[test]
    fn test_gaussian_blur_autocompute_sigmas() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        let img = Image::new(size, (0..25).map(|x| x as f32).collect(), CpuAllocator)?;

        let mut dst = Image::<_, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;

        gaussian_blur(&img, &mut dst, (3, 3), (0.0, 0.0))?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[0.002010752, 1.001341, 2.001006, 3.0006707, 3.9986594,
              4.998659, 6.0, 7.0000005, 8.0, 8.996648,
              9.996984, 11.0, 12.000002, 13.0, 13.994974,
              14.995307, 16.0, 17.0, 18.000002, 18.9933,
              19.985254, 20.991283, 21.990952, 22.990616, 23.981903,
            ]
        );
        Ok(())
    }

    #[test]
    fn test_spatial_gradient() -> Result<(), ImageError> {
        // First, define a type alias for the function signature
        type FilterFunction = fn(
            &Image<f32, 2, CpuAllocator>,
            &mut Image<f32, 2, CpuAllocator>,
            &mut Image<f32, 2, CpuAllocator>,
        ) -> Result<(), ImageError>;

        // Then, define a type for the test tuple
        type TestCase = (FilterFunction, &'static str);

        // Now use these types in the static array
        static TEST_FUNCTIONS: &[TestCase] = &[
            (spatial_gradient_float, "spatial_gradient_float"),
            (
                spatial_gradient_float_parallel_row,
                "spatial_gradient_float_parallel_row",
            ),
            (
                spatial_gradient_float_parallel,
                "spatial_gradient_float_parallel",
            ),
        ];

        let size = ImageSize {
            width: 5,
            height: 5,
        };

        let img = Image::<f32, 2, _>::new(
            size,
            (0..25).flat_map(|x| [x as f32, x as f32 + 25.0]).collect(),
            CpuAllocator,
        )?;
        for (test_fn, fn_name) in TEST_FUNCTIONS {
            let mut dx = Image::<_, 2, _>::from_size_val(size, 0.0, CpuAllocator)?;
            let mut dy = Image::<_, 2, _>::from_size_val(size, 0.0, CpuAllocator)?;

            test_fn(&img, &mut dx, &mut dy)?;

            #[rustfmt::skip]
            assert_eq!(
                dx.channel(0)?.as_slice(),
                &[
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000
                ],
                "{fn_name} dx channel(0)",
            );

            #[rustfmt::skip]
            assert_eq!(
                dx.channel(1)?.as_slice(),
                &[
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000
                ],
                "{fn_name} dx channel(1)",
            );

            #[rustfmt::skip]
            assert_eq!(
                dy.channel(0)?.as_slice(),
                &[
                    2.5000, 2.5000, 2.5000, 2.5000, 2.5000,
                    5.0000, 5.0000, 5.0000, 5.0000, 5.0000,
                    5.0000, 5.0000, 5.0000, 5.0000, 5.0000,
                    5.0000, 5.0000, 5.0000, 5.0000, 5.0000,
                    2.5000, 2.5000, 2.5000, 2.5000, 2.5000
                ],
                "{fn_name} dy channel(0)",
            );

            #[rustfmt::skip]
            assert_eq!(
                dy.channel(1)?.as_slice(),
                &[
                    2.5000, 2.5000, 2.5000, 2.5000, 2.5000,
                    5.0000, 5.0000, 5.0000, 5.0000, 5.0000,
                    5.0000, 5.0000, 5.0000, 5.0000, 5.0000,
                    5.0000, 5.0000, 5.0000, 5.0000, 5.0000,
                    2.5000, 2.5000, 2.5000, 2.5000, 2.5000
                ],
                "{fn_name} dy channel(1)",
            );
        }

        Ok(())
    }
}
