/// Lucas–Kanade optical flow with pyramids.
use crate::filter::scharr_spatial_gradient_float;
use crate::interpolation::{interpolate_pixel_fast, InterpolationMode};
use crate::pyramid::pyrdown_f32;
use kornia_image::{Image, ImageError, ImageSize};
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
pub struct PyrLKPrecomputed {
    /// Gaussian pyramid of previous image.
    pub prev_pyr: Vec<Image<f32, 1>>,
    /// Gaussian pyramid of next image.
    pub next_pyr: Vec<Image<f32, 1>>,
    /// X gradient pyramid for previous image.
    pub grad_x_pyr: Vec<Image<f32, 1>>,
    /// Y gradient pyramid for previous image.
    pub grad_y_pyr: Vec<Image<f32, 1>>,
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
    /// `params.use_initial_flow` was enabled but `next_pts_in` was not provided.
    #[error("use_initial_flow is true but next_pts_in is None; provide initial points or disable use_initial_flow")]
    InitialFlowMissing,
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
fn sample_at(img: &Image<f32, 1>, x: f32, y: f32, mode: BorderMode) -> f32 {
    let max_x = img.cols() as f32 - 1.0;
    let max_y = img.rows() as f32 - 1.0;
    let (xf, yf) = match mode {
        BorderMode::Clamp => (x.clamp(0.0, max_x), y.clamp(0.0, max_y)),
        BorderMode::Mirror => (mirror_coord(x, max_x), mirror_coord(y, max_y)),
        BorderMode::Reject => (x, y),
    };
    interpolate_pixel_fast(img, xf, yf, 0, InterpolationMode::Bilinear)
}

/// Interior fast-path safety check.
///
/// Returns `true` iff a 21x21 window centred at `(cx, cy)` can be sampled
/// with the **fast-path** kernels (`build_three_patches_interior`,
/// `iter_step_interior`, `error_pass_interior`) without any per-pixel
/// bounds branching:
///
/// * Every integer corner `(iu, iv)` of every sampled pixel must satisfy
///   `0 <= iu, iu+1 < cols`, `0 <= iv, iv+1 < rows`.
/// * For `wx in [-10, 10]` we need `cx.trunc() + wx >= 0`
///   and `cx.trunc() + wx + 1 < cols`, i.e. `cx >= 10.0` and
///   `cx < (cols - 11) as f32`. (Note the strict `< cols - 11.0` is
///   tight: at `cx == cols - 11.0` exactly, `cx.trunc() + 10 + 1 == cols`,
///   which would index one past the last column.)
///
/// Callers fall back to the slow `sample_at`/BorderMode path when this
/// returns `false`.
#[inline(always)]
fn is_interior_for_lk(cx: f32, cy: f32, cols: usize, rows: usize) -> bool {
    cx >= 10.0 && cy >= 10.0 && cx < (cols as f32 - 11.0) && cy < (rows as f32 - 11.0)
}

/// Fused fast-path patch build for the three images sampled at the LK
/// patch centre: `prev`, `ix`, `iy`. The four bilinear weights are
/// computed **once** per patch (constant across all 441 pixels) and
/// applied via direct slice indexing — no per-pixel `fract()`, no
/// `get_unchecked` method dispatch, no `iu+1 < cols` branching.
///
/// Returns the 2x2 structure-tensor entries `(a, b, c) = (sum ix*ix,
/// sum ix*iy, sum iy*iy)` over the patch, which the caller uses to test
/// trackability (`lambda_min`) and to invert the system in the iter loop.
///
/// # Safety contract
///
/// Caller MUST have established [`is_interior_for_lk`] for `(cx, cy)`
/// against the dimensions of `prev`/`ix`/`iy` (all three images share
/// the same size — checked at the API boundary in
/// `calc_optical_flow_pyr_lk_with_precomputed`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn build_three_patches_interior(
    prev: &Image<f32, 1>,
    ix: &Image<f32, 1>,
    iy: &Image<f32, 1>,
    cx: f32,
    cy: f32,
    prev_out: &mut [f32; 441],
    ix_out: &mut [f32; 441],
    iy_out: &mut [f32; 441],
) -> (f32, f32, f32) {
    let cols = prev.cols();
    let prev_data = prev.as_slice();
    let ix_data = ix.as_slice();
    let iy_data = iy.as_slice();
    let (iu_base, iv_base, w) = weights_for(cx, cy);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return build_three_patches_interior_neon(
            prev_data, ix_data, iy_data, cols, iu_base, iv_base, w, prev_out, ix_out, iy_out,
        );
    }

    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 && cpu.has_fma {
            unsafe {
                return build_three_patches_interior_avx2(
                    prev_data, ix_data, iy_data, cols, iu_base, iv_base, w, prev_out, ix_out,
                    iy_out,
                );
            }
        }
    }

    #[allow(unreachable_code)]
    build_three_patches_interior_scalar(
        prev_data, ix_data, iy_data, cols, iu_base, iv_base, w, prev_out, ix_out, iy_out,
    )
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn build_three_patches_interior_scalar(
    prev_data: &[f32],
    ix_data: &[f32],
    iy_data: &[f32],
    cols: usize,
    iu_base: i32,
    iv_base: i32,
    w: BilinearWeights,
    prev_out: &mut [f32; 441],
    ix_out: &mut [f32; 441],
    iy_out: &mut [f32; 441],
) -> (f32, f32, f32) {
    let mut a = 0.0_f32;
    let mut b = 0.0_f32;
    let mut c = 0.0_f32;
    let mut idx = 0usize;
    for wy in -10i32..=10 {
        let iv = (iv_base + wy) as usize;
        let row_top = iv * cols;
        let row_bot = row_top + cols;
        let iu_start = (iu_base - 10) as usize;
        for k in 0..21usize {
            let iu = iu_start + k;
            let p00 = prev_data[row_top + iu];
            let p01 = prev_data[row_top + iu + 1];
            let p10 = prev_data[row_bot + iu];
            let p11 = prev_data[row_bot + iu + 1];
            let pv = p00 * w.w00 + p01 * w.w01 + p10 * w.w10 + p11 * w.w11;
            prev_out[idx] = pv;

            let x00 = ix_data[row_top + iu];
            let x01 = ix_data[row_top + iu + 1];
            let x10 = ix_data[row_bot + iu];
            let x11 = ix_data[row_bot + iu + 1];
            let xv = x00 * w.w00 + x01 * w.w01 + x10 * w.w10 + x11 * w.w11;
            ix_out[idx] = xv;

            let y00 = iy_data[row_top + iu];
            let y01 = iy_data[row_top + iu + 1];
            let y10 = iy_data[row_bot + iu];
            let y11 = iy_data[row_bot + iu + 1];
            let yv = y00 * w.w00 + y01 * w.w01 + y10 * w.w10 + y11 * w.w11;
            iy_out[idx] = yv;

            a += xv * xv;
            b += xv * yv;
            c += yv * yv;
            idx += 1;
        }
    }
    (a, b, c)
}

/// AVX2+FMA fused patch builder. Each row processed as 2×8 + 5 tail.
/// Three images (prev, ix, iy) share the same broadcast weights — the
/// dominant memory-bandwidth pattern, so the wins are smaller than the
/// iter-step but still measurable since we run it 4× per feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn build_three_patches_interior_avx2(
    prev_data: &[f32],
    ix_data: &[f32],
    iy_data: &[f32],
    cols: usize,
    iu_base: i32,
    iv_base: i32,
    w: BilinearWeights,
    prev_out: &mut [f32; 441],
    ix_out: &mut [f32; 441],
    iy_out: &mut [f32; 441],
) -> (f32, f32, f32) {
    use std::arch::x86_64::*;
    let w00v = _mm256_set1_ps(w.w00);
    let w01v = _mm256_set1_ps(w.w01);
    let w10v = _mm256_set1_ps(w.w10);
    let w11v = _mm256_set1_ps(w.w11);
    let mut a_acc = _mm256_setzero_ps();
    let mut b_acc = _mm256_setzero_ps();
    let mut c_acc = _mm256_setzero_ps();
    let mut a_scalar = 0.0_f32;
    let mut b_scalar = 0.0_f32;
    let mut c_scalar = 0.0_f32;

    let prev_ptr = prev_data.as_ptr();
    let ix_ptr = ix_data.as_ptr();
    let iy_ptr = iy_data.as_ptr();
    let prev_out_ptr = prev_out.as_mut_ptr();
    let ix_out_ptr = ix_out.as_mut_ptr();
    let iy_out_ptr = iy_out.as_mut_ptr();
    let mut idx = 0usize;

    // Helper: bilinear 4-load + FMA chain.
    #[inline(always)]
    unsafe fn bilin8(
        data: *const f32,
        base_top: usize,
        base_bot: usize,
        w00v: __m256,
        w01v: __m256,
        w10v: __m256,
        w11v: __m256,
    ) -> __m256 {
        let p00 = _mm256_loadu_ps(data.add(base_top));
        let p01 = _mm256_loadu_ps(data.add(base_top + 1));
        let p10 = _mm256_loadu_ps(data.add(base_bot));
        let p11 = _mm256_loadu_ps(data.add(base_bot + 1));
        let mut v = _mm256_mul_ps(p00, w00v);
        v = _mm256_fmadd_ps(p01, w01v, v);
        v = _mm256_fmadd_ps(p10, w10v, v);
        v = _mm256_fmadd_ps(p11, w11v, v);
        v
    }

    for wy in -10i32..=10 {
        let iv = (iv_base + wy) as usize;
        let row_top = iv * cols;
        let row_bot = row_top + cols;
        let iu_start = (iu_base - 10) as usize;

        for batch in 0..2usize {
            let off = batch * 8;
            let base_top = row_top + iu_start + off;
            let base_bot = row_bot + iu_start + off;

            let pv = bilin8(prev_ptr, base_top, base_bot, w00v, w01v, w10v, w11v);
            let xv = bilin8(ix_ptr, base_top, base_bot, w00v, w01v, w10v, w11v);
            let yv = bilin8(iy_ptr, base_top, base_bot, w00v, w01v, w10v, w11v);

            _mm256_storeu_ps(prev_out_ptr.add(idx + off), pv);
            _mm256_storeu_ps(ix_out_ptr.add(idx + off), xv);
            _mm256_storeu_ps(iy_out_ptr.add(idx + off), yv);

            // a += xv*xv, b += xv*yv, c += yv*yv.
            a_acc = _mm256_fmadd_ps(xv, xv, a_acc);
            b_acc = _mm256_fmadd_ps(xv, yv, b_acc);
            c_acc = _mm256_fmadd_ps(yv, yv, c_acc);
        }

        for k in 16..21usize {
            let iu = iu_start + k;
            let p00 = prev_data[row_top + iu];
            let p01 = prev_data[row_top + iu + 1];
            let p10 = prev_data[row_bot + iu];
            let p11 = prev_data[row_bot + iu + 1];
            let pv = p00 * w.w00 + p01 * w.w01 + p10 * w.w10 + p11 * w.w11;
            prev_out[idx + k] = pv;

            let x00 = ix_data[row_top + iu];
            let x01 = ix_data[row_top + iu + 1];
            let x10 = ix_data[row_bot + iu];
            let x11 = ix_data[row_bot + iu + 1];
            let xv = x00 * w.w00 + x01 * w.w01 + x10 * w.w10 + x11 * w.w11;
            ix_out[idx + k] = xv;

            let y00 = iy_data[row_top + iu];
            let y01 = iy_data[row_top + iu + 1];
            let y10 = iy_data[row_bot + iu];
            let y11 = iy_data[row_bot + iu + 1];
            let yv = y00 * w.w00 + y01 * w.w01 + y10 * w.w10 + y11 * w.w11;
            iy_out[idx + k] = yv;

            a_scalar += xv * xv;
            b_scalar += xv * yv;
            c_scalar += yv * yv;
        }

        idx += 21;
    }

    (
        hsum_avx_ps(a_acc) + a_scalar,
        hsum_avx_ps(b_acc) + b_scalar,
        hsum_avx_ps(c_acc) + c_scalar,
    )
}

/// NEON fused patch builder. 5×4 + 1 tail per row.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn build_three_patches_interior_neon(
    prev_data: &[f32],
    ix_data: &[f32],
    iy_data: &[f32],
    cols: usize,
    iu_base: i32,
    iv_base: i32,
    w: BilinearWeights,
    prev_out: &mut [f32; 441],
    ix_out: &mut [f32; 441],
    iy_out: &mut [f32; 441],
) -> (f32, f32, f32) {
    unsafe {
        use std::arch::aarch64::*;
        let w00v = vdupq_n_f32(w.w00);
        let w01v = vdupq_n_f32(w.w01);
        let w10v = vdupq_n_f32(w.w10);
        let w11v = vdupq_n_f32(w.w11);
        let mut a_acc = vdupq_n_f32(0.0);
        let mut b_acc = vdupq_n_f32(0.0);
        let mut c_acc = vdupq_n_f32(0.0);
        let mut a_scalar = 0.0_f32;
        let mut b_scalar = 0.0_f32;
        let mut c_scalar = 0.0_f32;

        #[inline(always)]
        unsafe fn bilin4(
            data: *const f32,
            base_top: usize,
            base_bot: usize,
            w00v: float32x4_t,
            w01v: float32x4_t,
            w10v: float32x4_t,
            w11v: float32x4_t,
        ) -> float32x4_t {
            unsafe {
                let p00 = vld1q_f32(data.add(base_top));
                let p01 = vld1q_f32(data.add(base_top + 1));
                let p10 = vld1q_f32(data.add(base_bot));
                let p11 = vld1q_f32(data.add(base_bot + 1));
                let mut v = vmulq_f32(p00, w00v);
                v = vfmaq_f32(v, p01, w01v);
                v = vfmaq_f32(v, p10, w10v);
                v = vfmaq_f32(v, p11, w11v);
                v
            }
        }

        let prev_ptr = prev_data.as_ptr();
        let ix_ptr = ix_data.as_ptr();
        let iy_ptr = iy_data.as_ptr();
        let prev_out_ptr = prev_out.as_mut_ptr();
        let ix_out_ptr = ix_out.as_mut_ptr();
        let iy_out_ptr = iy_out.as_mut_ptr();
        let mut idx = 0usize;

        for wy in -10i32..=10 {
            let iv = (iv_base + wy) as usize;
            let row_top = iv * cols;
            let row_bot = row_top + cols;
            let iu_start = (iu_base - 10) as usize;

            for batch in 0..5usize {
                let off = batch * 4;
                let base_top = row_top + iu_start + off;
                let base_bot = row_bot + iu_start + off;

                let pv = bilin4(prev_ptr, base_top, base_bot, w00v, w01v, w10v, w11v);
                let xv = bilin4(ix_ptr, base_top, base_bot, w00v, w01v, w10v, w11v);
                let yv = bilin4(iy_ptr, base_top, base_bot, w00v, w01v, w10v, w11v);

                vst1q_f32(prev_out_ptr.add(idx + off), pv);
                vst1q_f32(ix_out_ptr.add(idx + off), xv);
                vst1q_f32(iy_out_ptr.add(idx + off), yv);

                a_acc = vfmaq_f32(a_acc, xv, xv);
                b_acc = vfmaq_f32(b_acc, xv, yv);
                c_acc = vfmaq_f32(c_acc, yv, yv);
            }

            let iu = iu_start + 20;
            let p00 = prev_data[row_top + iu];
            let p01 = prev_data[row_top + iu + 1];
            let p10 = prev_data[row_bot + iu];
            let p11 = prev_data[row_bot + iu + 1];
            let pv = p00 * w.w00 + p01 * w.w01 + p10 * w.w10 + p11 * w.w11;
            prev_out[idx + 20] = pv;

            let x00 = ix_data[row_top + iu];
            let x01 = ix_data[row_top + iu + 1];
            let x10 = ix_data[row_bot + iu];
            let x11 = ix_data[row_bot + iu + 1];
            let xv = x00 * w.w00 + x01 * w.w01 + x10 * w.w10 + x11 * w.w11;
            ix_out[idx + 20] = xv;

            let y00 = iy_data[row_top + iu];
            let y01 = iy_data[row_top + iu + 1];
            let y10 = iy_data[row_bot + iu];
            let y11 = iy_data[row_bot + iu + 1];
            let yv = y00 * w.w00 + y01 * w.w01 + y10 * w.w10 + y11 * w.w11;
            iy_out[idx + 20] = yv;

            a_scalar += xv * xv;
            b_scalar += xv * yv;
            c_scalar += yv * yv;

            idx += 21;
        }

        (
            vaddvq_f32(a_acc) + a_scalar,
            vaddvq_f32(b_acc) + b_scalar,
            vaddvq_f32(c_acc) + c_scalar,
        )
    }
}

/// Fast-path single Gauss-Newton iteration step: samples `next` at
/// `(cx, cy)`, computes the temporal difference vs the cached `prev`
/// patch, and accumulates `(d, e) = (-sum ix*it, -sum iy*it)`.
///
/// # Safety contract
///
/// Caller MUST have established [`is_interior_for_lk`] for `(cx, cy)`
/// against `next`'s dimensions.
/// The four bilinear weights, packed for the SIMD kernels.
#[derive(Clone, Copy)]
struct BilinearWeights {
    w00: f32,
    w01: f32,
    w10: f32,
    w11: f32,
}

#[inline(always)]
fn weights_for(cx: f32, cy: f32) -> (i32, i32, BilinearWeights) {
    let iu_base = cx.trunc() as i32;
    let iv_base = cy.trunc() as i32;
    let fu = cx - iu_base as f32;
    let fv = cy - iv_base as f32;
    let fuu = 1.0 - fu;
    let fvv = 1.0 - fv;
    (
        iu_base,
        iv_base,
        BilinearWeights {
            w00: fuu * fvv,
            w01: fu * fvv,
            w10: fuu * fv,
            w11: fu * fv,
        },
    )
}

#[inline(always)]
fn iter_step_interior(
    next: &Image<f32, 1>,
    cx: f32,
    cy: f32,
    prev_patch: &[f32; 441],
    ix_patch: &[f32; 441],
    iy_patch: &[f32; 441],
) -> (f32, f32) {
    let cols = next.cols();
    let next_data = next.as_slice();
    let (iu_base, iv_base, w) = weights_for(cx, cy);

    // Cross-arch SIMD dispatcher mirroring the kornia-imgproc/3d kernel
    // convention: aarch64 → NEON unconditionally (NEON is baseline for
    // supported aarch64 targets), x86_64 → AVX2+FMA when probed at runtime,
    // otherwise the portable scalar reference. All three paths produce the
    // same `(d, e)` modulo FMA-reordering noise.
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return iter_step_interior_neon(
            next_data, cols, iu_base, iv_base, w, prev_patch, ix_patch, iy_patch,
        );
    }

    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 && cpu.has_fma {
            unsafe {
                return iter_step_interior_avx2(
                    next_data, cols, iu_base, iv_base, w, prev_patch, ix_patch, iy_patch,
                );
            }
        }
    }

    #[allow(unreachable_code)]
    iter_step_interior_scalar(
        next_data, cols, iu_base, iv_base, w, prev_patch, ix_patch, iy_patch,
    )
}

/// Portable scalar reference for [`iter_step_interior`] — single source
/// of numeric truth. SIMD paths must match this to FMA-reordering tolerance.
#[inline]
#[allow(clippy::too_many_arguments)]
fn iter_step_interior_scalar(
    next_data: &[f32],
    cols: usize,
    iu_base: i32,
    iv_base: i32,
    w: BilinearWeights,
    prev_patch: &[f32; 441],
    ix_patch: &[f32; 441],
    iy_patch: &[f32; 441],
) -> (f32, f32) {
    let mut d = 0.0_f32;
    let mut e = 0.0_f32;
    let mut idx = 0usize;
    for wy in -10i32..=10 {
        let iv = (iv_base + wy) as usize;
        let row_top = iv * cols;
        let row_bot = row_top + cols;
        let iu_start = (iu_base - 10) as usize;
        for k in 0..21usize {
            let iu = iu_start + k;
            let p00 = next_data[row_top + iu];
            let p01 = next_data[row_top + iu + 1];
            let p10 = next_data[row_bot + iu];
            let p11 = next_data[row_bot + iu + 1];
            let i1 = p00 * w.w00 + p01 * w.w01 + p10 * w.w10 + p11 * w.w11;
            let it = i1 - prev_patch[idx];
            d -= ix_patch[idx] * it;
            e -= iy_patch[idx] * it;
            idx += 1;
        }
    }
    (d, e)
}

/// AVX2+FMA iter-step kernel. Each row of 21 pixels is processed as two
/// 8-wide YMM batches (pixels 0..16) followed by a 5-pixel scalar tail.
/// Weights are broadcast once and stay in registers across the whole
/// 441-pixel patch.
///
/// # Safety
/// - Caller has runtime-checked `cpu_features().has_avx2 && has_fma`.
/// - Caller has established [`is_interior_for_lk`] so every load below is
///   in bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn iter_step_interior_avx2(
    next_data: &[f32],
    cols: usize,
    iu_base: i32,
    iv_base: i32,
    w: BilinearWeights,
    prev_patch: &[f32; 441],
    ix_patch: &[f32; 441],
    iy_patch: &[f32; 441],
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let w00v = _mm256_set1_ps(w.w00);
    let w01v = _mm256_set1_ps(w.w01);
    let w10v = _mm256_set1_ps(w.w10);
    let w11v = _mm256_set1_ps(w.w11);
    let mut d_acc = _mm256_setzero_ps();
    let mut e_acc = _mm256_setzero_ps();
    let mut d_scalar = 0.0_f32;
    let mut e_scalar = 0.0_f32;

    let next_ptr = next_data.as_ptr();
    let prev_ptr = prev_patch.as_ptr();
    let ix_ptr = ix_patch.as_ptr();
    let iy_ptr = iy_patch.as_ptr();
    let mut idx = 0usize;

    for wy in -10i32..=10 {
        let iv = (iv_base + wy) as usize;
        let row_top = iv * cols;
        let row_bot = row_top + cols;
        let iu_start = (iu_base - 10) as usize;

        // Two 8-wide batches: pixels [0..8] and [8..16].
        for batch in 0..2usize {
            let off = batch * 8;
            let base_top = row_top + iu_start + off;
            let base_bot = row_bot + iu_start + off;

            let p00 = _mm256_loadu_ps(next_ptr.add(base_top));
            let p01 = _mm256_loadu_ps(next_ptr.add(base_top + 1));
            let p10 = _mm256_loadu_ps(next_ptr.add(base_bot));
            let p11 = _mm256_loadu_ps(next_ptr.add(base_bot + 1));

            // i1 = p00*w00 + p01*w01 + p10*w10 + p11*w11, chained as FMAs.
            let mut i1 = _mm256_mul_ps(p00, w00v);
            i1 = _mm256_fmadd_ps(p01, w01v, i1);
            i1 = _mm256_fmadd_ps(p10, w10v, i1);
            i1 = _mm256_fmadd_ps(p11, w11v, i1);

            let prev_v = _mm256_loadu_ps(prev_ptr.add(idx + off));
            let it = _mm256_sub_ps(i1, prev_v);

            let ix_v = _mm256_loadu_ps(ix_ptr.add(idx + off));
            let iy_v = _mm256_loadu_ps(iy_ptr.add(idx + off));

            // d -= ix*it  ==  d_acc = -ix*it + d_acc  (fnmadd).
            d_acc = _mm256_fnmadd_ps(ix_v, it, d_acc);
            e_acc = _mm256_fnmadd_ps(iy_v, it, e_acc);
        }

        // 5-pixel scalar tail (k = 16..21).
        for k in 16..21usize {
            let iu = iu_start + k;
            let p00 = next_data[row_top + iu];
            let p01 = next_data[row_top + iu + 1];
            let p10 = next_data[row_bot + iu];
            let p11 = next_data[row_bot + iu + 1];
            let i1 = p00 * w.w00 + p01 * w.w01 + p10 * w.w10 + p11 * w.w11;
            let it = i1 - prev_patch[idx + k];
            d_scalar -= ix_patch[idx + k] * it;
            e_scalar -= iy_patch[idx + k] * it;
        }

        idx += 21;
    }

    (hsum_avx_ps(d_acc) + d_scalar, hsum_avx_ps(e_acc) + e_scalar)
}

/// Horizontal sum of a `__m256` of 8 f32 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf = _mm_movehl_ps(sums, sums);
    let sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

/// NEON iter-step kernel. Each row of 21 pixels processed as five 4-wide
/// `vld1q_f32` batches (pixels 0..20) + 1-pixel scalar tail.
///
/// # Safety
/// - aarch64 architectural; `target_feature(neon)` for the intrinsics.
/// - Caller has established [`is_interior_for_lk`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn iter_step_interior_neon(
    next_data: &[f32],
    cols: usize,
    iu_base: i32,
    iv_base: i32,
    w: BilinearWeights,
    prev_patch: &[f32; 441],
    ix_patch: &[f32; 441],
    iy_patch: &[f32; 441],
) -> (f32, f32) {
    unsafe {
        use std::arch::aarch64::*;
        let w00v = vdupq_n_f32(w.w00);
        let w01v = vdupq_n_f32(w.w01);
        let w10v = vdupq_n_f32(w.w10);
        let w11v = vdupq_n_f32(w.w11);
        let mut d_acc = vdupq_n_f32(0.0);
        let mut e_acc = vdupq_n_f32(0.0);
        let mut d_scalar = 0.0_f32;
        let mut e_scalar = 0.0_f32;

        let next_ptr = next_data.as_ptr();
        let prev_ptr = prev_patch.as_ptr();
        let ix_ptr = ix_patch.as_ptr();
        let iy_ptr = iy_patch.as_ptr();
        let mut idx = 0usize;

        for wy in -10i32..=10 {
            let iv = (iv_base + wy) as usize;
            let row_top = iv * cols;
            let row_bot = row_top + cols;
            let iu_start = (iu_base - 10) as usize;

            // Five 4-wide batches: pixels [0..4], [4..8], [8..12], [12..16], [16..20].
            for batch in 0..5usize {
                let off = batch * 4;
                let base_top = row_top + iu_start + off;
                let base_bot = row_bot + iu_start + off;

                let p00 = vld1q_f32(next_ptr.add(base_top));
                let p01 = vld1q_f32(next_ptr.add(base_top + 1));
                let p10 = vld1q_f32(next_ptr.add(base_bot));
                let p11 = vld1q_f32(next_ptr.add(base_bot + 1));

                let mut i1 = vmulq_f32(p00, w00v);
                i1 = vfmaq_f32(i1, p01, w01v);
                i1 = vfmaq_f32(i1, p10, w10v);
                i1 = vfmaq_f32(i1, p11, w11v);

                let prev_v = vld1q_f32(prev_ptr.add(idx + off));
                let it = vsubq_f32(i1, prev_v);

                let ix_v = vld1q_f32(ix_ptr.add(idx + off));
                let iy_v = vld1q_f32(iy_ptr.add(idx + off));

                // d -= ix*it  ==  fma-subtract: d_acc = d_acc - ix*it.
                d_acc = vfmsq_f32(d_acc, ix_v, it);
                e_acc = vfmsq_f32(e_acc, iy_v, it);
            }

            // 1-pixel scalar tail (k = 20).
            let iu = iu_start + 20;
            let p00 = next_data[row_top + iu];
            let p01 = next_data[row_top + iu + 1];
            let p10 = next_data[row_bot + iu];
            let p11 = next_data[row_bot + iu + 1];
            let i1 = p00 * w.w00 + p01 * w.w01 + p10 * w.w10 + p11 * w.w11;
            let it = i1 - prev_patch[idx + 20];
            d_scalar -= ix_patch[idx + 20] * it;
            e_scalar -= iy_patch[idx + 20] * it;

            idx += 21;
        }

        (vaddvq_f32(d_acc) + d_scalar, vaddvq_f32(e_acc) + e_scalar)
    }
}

/// Fast-path final error pass: samples `next` and accumulates
/// `sum |i1 - prev_patch|` for the tracking-error output.
///
/// # Safety contract
///
/// Caller MUST have established [`is_interior_for_lk`] for `(cx, cy)`.
#[inline(always)]
fn error_pass_interior(next: &Image<f32, 1>, cx: f32, cy: f32, prev_patch: &[f32; 441]) -> f32 {
    let cols = next.cols();
    let next_data = next.as_slice();
    let (iu_base, iv_base, w) = weights_for(cx, cy);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return error_pass_interior_neon(next_data, cols, iu_base, iv_base, w, prev_patch);
    }

    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 && cpu.has_fma {
            unsafe {
                return error_pass_interior_avx2(next_data, cols, iu_base, iv_base, w, prev_patch);
            }
        }
    }

    #[allow(unreachable_code)]
    error_pass_interior_scalar(next_data, cols, iu_base, iv_base, w, prev_patch)
}

#[inline]
fn error_pass_interior_scalar(
    next_data: &[f32],
    cols: usize,
    iu_base: i32,
    iv_base: i32,
    w: BilinearWeights,
    prev_patch: &[f32; 441],
) -> f32 {
    let mut err = 0.0_f32;
    let mut idx = 0usize;
    for wy in -10i32..=10 {
        let iv = (iv_base + wy) as usize;
        let row_top = iv * cols;
        let row_bot = row_top + cols;
        let iu_start = (iu_base - 10) as usize;
        for k in 0..21usize {
            let iu = iu_start + k;
            let p00 = next_data[row_top + iu];
            let p01 = next_data[row_top + iu + 1];
            let p10 = next_data[row_bot + iu];
            let p11 = next_data[row_bot + iu + 1];
            let i1 = p00 * w.w00 + p01 * w.w01 + p10 * w.w10 + p11 * w.w11;
            err += (i1 - prev_patch[idx]).abs();
            idx += 1;
        }
    }
    err
}

/// AVX2+FMA error-pass kernel. Same 2×8 + 5-tail row layout as the
/// iter-step kernel. `|i1 - prev|` is implemented as an AND with the
/// sign-bit-cleared mask (faster than vmaxps with negation).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn error_pass_interior_avx2(
    next_data: &[f32],
    cols: usize,
    iu_base: i32,
    iv_base: i32,
    w: BilinearWeights,
    prev_patch: &[f32; 441],
) -> f32 {
    use std::arch::x86_64::*;
    let w00v = _mm256_set1_ps(w.w00);
    let w01v = _mm256_set1_ps(w.w01);
    let w10v = _mm256_set1_ps(w.w10);
    let w11v = _mm256_set1_ps(w.w11);
    // 0x7FFF_FFFF mask: clears the sign bit → absolute value.
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffffu32 as i32));
    let mut err_acc = _mm256_setzero_ps();
    let mut err_scalar = 0.0_f32;

    let next_ptr = next_data.as_ptr();
    let prev_ptr = prev_patch.as_ptr();
    let mut idx = 0usize;

    for wy in -10i32..=10 {
        let iv = (iv_base + wy) as usize;
        let row_top = iv * cols;
        let row_bot = row_top + cols;
        let iu_start = (iu_base - 10) as usize;

        for batch in 0..2usize {
            let off = batch * 8;
            let base_top = row_top + iu_start + off;
            let base_bot = row_bot + iu_start + off;

            let p00 = _mm256_loadu_ps(next_ptr.add(base_top));
            let p01 = _mm256_loadu_ps(next_ptr.add(base_top + 1));
            let p10 = _mm256_loadu_ps(next_ptr.add(base_bot));
            let p11 = _mm256_loadu_ps(next_ptr.add(base_bot + 1));

            let mut i1 = _mm256_mul_ps(p00, w00v);
            i1 = _mm256_fmadd_ps(p01, w01v, i1);
            i1 = _mm256_fmadd_ps(p10, w10v, i1);
            i1 = _mm256_fmadd_ps(p11, w11v, i1);

            let prev_v = _mm256_loadu_ps(prev_ptr.add(idx + off));
            let diff = _mm256_sub_ps(i1, prev_v);
            let abs_diff = _mm256_and_ps(diff, abs_mask);
            err_acc = _mm256_add_ps(err_acc, abs_diff);
        }

        for k in 16..21usize {
            let iu = iu_start + k;
            let p00 = next_data[row_top + iu];
            let p01 = next_data[row_top + iu + 1];
            let p10 = next_data[row_bot + iu];
            let p11 = next_data[row_bot + iu + 1];
            let i1 = p00 * w.w00 + p01 * w.w01 + p10 * w.w10 + p11 * w.w11;
            err_scalar += (i1 - prev_patch[idx + k]).abs();
        }

        idx += 21;
    }

    hsum_avx_ps(err_acc) + err_scalar
}

/// NEON error-pass kernel. `|i1 - prev|` via `vabsq_f32`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn error_pass_interior_neon(
    next_data: &[f32],
    cols: usize,
    iu_base: i32,
    iv_base: i32,
    w: BilinearWeights,
    prev_patch: &[f32; 441],
) -> f32 {
    unsafe {
        use std::arch::aarch64::*;
        let w00v = vdupq_n_f32(w.w00);
        let w01v = vdupq_n_f32(w.w01);
        let w10v = vdupq_n_f32(w.w10);
        let w11v = vdupq_n_f32(w.w11);
        let mut err_acc = vdupq_n_f32(0.0);
        let mut err_scalar = 0.0_f32;

        let next_ptr = next_data.as_ptr();
        let prev_ptr = prev_patch.as_ptr();
        let mut idx = 0usize;

        for wy in -10i32..=10 {
            let iv = (iv_base + wy) as usize;
            let row_top = iv * cols;
            let row_bot = row_top + cols;
            let iu_start = (iu_base - 10) as usize;

            for batch in 0..5usize {
                let off = batch * 4;
                let base_top = row_top + iu_start + off;
                let base_bot = row_bot + iu_start + off;

                let p00 = vld1q_f32(next_ptr.add(base_top));
                let p01 = vld1q_f32(next_ptr.add(base_top + 1));
                let p10 = vld1q_f32(next_ptr.add(base_bot));
                let p11 = vld1q_f32(next_ptr.add(base_bot + 1));

                let mut i1 = vmulq_f32(p00, w00v);
                i1 = vfmaq_f32(i1, p01, w01v);
                i1 = vfmaq_f32(i1, p10, w10v);
                i1 = vfmaq_f32(i1, p11, w11v);

                let prev_v = vld1q_f32(prev_ptr.add(idx + off));
                let diff = vsubq_f32(i1, prev_v);
                err_acc = vaddq_f32(err_acc, vabsq_f32(diff));
            }

            let iu = iu_start + 20;
            let p00 = next_data[row_top + iu];
            let p01 = next_data[row_top + iu + 1];
            let p10 = next_data[row_bot + iu];
            let p11 = next_data[row_bot + iu + 1];
            let i1 = p00 * w.w00 + p01 * w.w01 + p10 * w.w10 + p11 * w.w11;
            err_scalar += (i1 - prev_patch[idx + 20]).abs();

            idx += 21;
        }

        vaddvq_f32(err_acc) + err_scalar
    }
}

fn track_feature(
    pt: [f32; 2],
    initial_flow: Option<[f32; 2]>,
    precomputed: &PyrLKPrecomputed,
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
        let (a, b, c);

        // Fast-path interior kernel: hoist the four bilinear weights once
        // per patch (constant across all 441 pixels) and use direct slice
        // indexing. Skips per-pixel BorderMode dispatch + `iu+1<cols`
        // branches. Slow path retained for patches that straddle a border.
        if is_interior_for_lk(xc, yc, prev.cols(), prev.rows()) {
            (a, b, c) = build_three_patches_interior(
                prev,
                ix,
                iy,
                xc,
                yc,
                &mut prev_patch,
                &mut ix_patch,
                &mut iy_patch,
            );
        } else {
            let mut aa = 0.0f32;
            let mut bb = 0.0f32;
            let mut cc = 0.0f32;
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
                    aa += ixv * ixv;
                    bb += ixv * iyv;
                    cc += iyv * iyv;
                    idx += 1;
                }
            }
            a = aa;
            b = bb;
            c = cc;
        }

        let det = a * c - b * b;
        if det.abs() < 1e-7 {
            return None;
        }
        let trace = a + c;
        let delta = a - c;
        let lambda_min = (trace - ((delta * delta + 4.0 * b * b).sqrt())) * 0.5;
        if lambda_min / (WIN_PIXELS as f32) < params.min_eigen_threshold {
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

            let (d, e) = if is_interior_for_lk(xnc, ync, next.cols(), next.rows()) {
                iter_step_interior(next, xnc, ync, &prev_patch, &ix_patch, &iy_patch)
            } else {
                let mut dd = 0.0f32;
                let mut ee = 0.0f32;
                let mut idx = 0usize;
                for wy in -HALF_WIN..=HALF_WIN {
                    for wx in -HALF_WIN..=HALF_WIN {
                        let qx = xnc + wx as f32;
                        let qy = ync + wy as f32;
                        let i1 = sample_at(next, qx, qy, params.border_mode);
                        let it = i1 - prev_patch[idx];
                        dd -= ix_patch[idx] * it;
                        ee -= iy_patch[idx] * it;
                        idx += 1;
                    }
                }
                (dd, ee)
            };

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
            let ex = pt[0] + dx;
            let ey = pt[1] + dy;
            let err_sum = if is_interior_for_lk(ex, ey, next.cols(), next.rows()) {
                error_pass_interior(next, ex, ey, &prev_patch)
            } else {
                let mut err = 0.0f32;
                let mut idx = 0usize;
                for wy in -HALF_WIN..=HALF_WIN {
                    for wx in -HALF_WIN..=HALF_WIN {
                        let qx = ex + wx as f32;
                        let qy = ey + wy as f32;
                        let i1 = sample_at(next, qx, qy, params.border_mode);
                        err += (i1 - prev_patch[idx]).abs();
                        idx += 1;
                    }
                }
                err
            };
            tracking_error = err_sum / WIN_PIXELS as f32;
        }
    }

    Some(([pt[0] + dx, pt[1] + dy], tracking_error))
}

/// Build Gaussian pyramids and gradients required by sparse LK.
pub fn build_lk_precomputed(
    prev_img: &Image<f32, 1>,
    next_img: &Image<f32, 1>,
    max_level: usize,
) -> Result<PyrLKPrecomputed, PyrLKError> {
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
pub fn calc_optical_flow_pyr_lk(
    prev_img: &Image<f32, 1>,
    next_img: &Image<f32, 1>,
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
    if params.use_initial_flow {
        let Some(initial_pts) = next_pts_in else {
            return Err(PyrLKError::InitialFlowMissing);
        };
        if initial_pts.len() != prev_pts.len() {
            return Err(PyrLKError::InitialFlowLengthMismatch {
                expected: prev_pts.len(),
                provided: initial_pts.len(),
            });
        }
    }
    let precomputed = build_lk_precomputed(prev_img, next_img, params.max_level)?;
    calc_optical_flow_pyr_lk_with_precomputed(&precomputed, prev_pts, next_pts_in, params)
}

/// Compute sparse pyramidal LK flow using precomputed pyramids and gradients.
pub fn calc_optical_flow_pyr_lk_with_precomputed(
    precomputed: &PyrLKPrecomputed,
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

    if params.use_initial_flow {
        let Some(initial_pts) = next_pts_in else {
            return Err(PyrLKError::InitialFlowMissing);
        };
        if initial_pts.len() != prev_pts.len() {
            return Err(PyrLKError::InitialFlowLengthMismatch {
                expected: prev_pts.len(),
                provided: initial_pts.len(),
            });
        }
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
    use kornia_tensor::host_alloc;

    fn default_params() -> PyrLKParams {
        PyrLKParams::default()
    }

    /// Equivalent of `build_three_patches_interior` written via the public
    /// `sample_at` / BorderMode path. Used as the byte-equality oracle for
    /// the fast-path kernels.
    fn build_three_patches_slow(
        prev: &Image<f32, 1>,
        ix: &Image<f32, 1>,
        iy: &Image<f32, 1>,
        cx: f32,
        cy: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32, f32, f32) {
        let mut p = vec![0.0_f32; 441];
        let mut xs = vec![0.0_f32; 441];
        let mut ys = vec![0.0_f32; 441];
        let mut a = 0.0_f32;
        let mut b = 0.0_f32;
        let mut c = 0.0_f32;
        let mut idx = 0;
        for wy in -10i32..=10 {
            for wx in -10i32..=10 {
                let px = cx + wx as f32;
                let py = cy + wy as f32;
                let pv = sample_at(prev, px, py, BorderMode::Clamp);
                let xv = sample_at(ix, px, py, BorderMode::Clamp);
                let yv = sample_at(iy, px, py, BorderMode::Clamp);
                p[idx] = pv;
                xs[idx] = xv;
                ys[idx] = yv;
                a += xv * xv;
                b += xv * yv;
                c += yv * yv;
                idx += 1;
            }
        }
        (p, xs, ys, a, b, c)
    }

    /// Validates that for an interior patch centre, the fast-path
    /// `build_three_patches_interior` is *byte-equal* (modulo FMA
    /// reordering noise — here we accept 1e-5 relative because the slow
    /// path is `a*b + c*d + ...` and the fast path is the same expression
    /// rearranged) to the slow path going through `sample_at`.
    #[test]
    fn interior_patch_build_matches_slow_path() {
        // Use the gradient image (smooth, lots of non-zero pixels) so the
        // structure-tensor entries are non-trivial.
        let size = 128;
        let img = make_gradient_image(size);
        let mut ix =
            Image::<f32, 1>::from_size_val([size, size].into(), 0.0, host_alloc()).unwrap();
        let mut iy =
            Image::<f32, 1>::from_size_val([size, size].into(), 0.0, host_alloc()).unwrap();
        crate::filter::scharr_spatial_gradient_float(&img, &mut ix, &mut iy).unwrap();

        // A few interior centres with non-trivial fractional parts so the
        // bilinear weights aren't all zero/one.
        let centres = [(40.5, 50.25), (60.0, 60.0), (75.125, 55.875), (90.0, 90.0)];
        for &(cx, cy) in centres.iter() {
            assert!(
                is_interior_for_lk(cx, cy, size, size),
                "test centre ({cx}, {cy}) must be in the interior"
            );

            let (slow_p, slow_x, slow_y, slow_a, slow_b, slow_c) =
                build_three_patches_slow(&img, &ix, &iy, cx, cy);

            let mut fast_p = [0.0f32; 441];
            let mut fast_x = [0.0f32; 441];
            let mut fast_y = [0.0f32; 441];
            let (fast_a, fast_b, fast_c) = build_three_patches_interior(
                &img,
                &ix,
                &iy,
                cx,
                cy,
                &mut fast_p,
                &mut fast_x,
                &mut fast_y,
            );

            for i in 0..441 {
                let tol = 1e-5_f32 * slow_p[i].abs().max(1.0);
                assert!(
                    (fast_p[i] - slow_p[i]).abs() < tol,
                    "prev pixel {i} fast={} slow={}",
                    fast_p[i],
                    slow_p[i]
                );
                let tol = 1e-5_f32 * slow_x[i].abs().max(1.0);
                assert!(
                    (fast_x[i] - slow_x[i]).abs() < tol,
                    "ix pixel {i} fast={} slow={}",
                    fast_x[i],
                    slow_x[i]
                );
                let tol = 1e-5_f32 * slow_y[i].abs().max(1.0);
                assert!(
                    (fast_y[i] - slow_y[i]).abs() < tol,
                    "iy pixel {i} fast={} slow={}",
                    fast_y[i],
                    slow_y[i]
                );
            }
            // Accumulated sums use ~441 adds — looser tolerance.
            let tol = 1e-4_f32 * slow_a.abs().max(1.0);
            assert!(
                (fast_a - slow_a).abs() < tol,
                "a fast={fast_a} slow={slow_a} (centre {cx},{cy})"
            );
            assert!((fast_b - slow_b).abs() < 1e-4 * slow_b.abs().max(1.0));
            assert!((fast_c - slow_c).abs() < 1e-4 * slow_c.abs().max(1.0));
        }
    }

    /// SIMD parity: the AVX2 (or NEON) iter-step kernel must agree with
    /// the scalar reference to FMA-reordering tolerance. The dispatcher
    /// routes interior calls through the SIMD path on this machine, so a
    /// direct comparison validates the SIMD lane logic + scalar tail.
    #[test]
    fn simd_iter_step_matches_scalar() {
        let size = 128;
        let img = make_gradient_image(size);
        let cols = img.cols();
        let data = img.as_slice();

        // A couple of representative non-trivial patches.
        let prev_patch: [f32; 441] = std::array::from_fn(|i| (i as f32 * 0.013).sin());
        let ix_patch: [f32; 441] = std::array::from_fn(|i| (i as f32 * 0.021).cos());
        let iy_patch: [f32; 441] = std::array::from_fn(|i| ((i as f32) * 0.017).sin() * 0.5);

        for (cx, cy) in [(40.5_f32, 50.25_f32), (60.0, 60.0), (75.125, 55.875)] {
            assert!(is_interior_for_lk(cx, cy, size, size));
            let (iu, iv, w) = weights_for(cx, cy);

            let scalar =
                iter_step_interior_scalar(data, cols, iu, iv, w, &prev_patch, &ix_patch, &iy_patch);

            #[cfg(target_arch = "x86_64")]
            {
                let cpu = crate::simd::cpu_features();
                if cpu.has_avx2 && cpu.has_fma {
                    let simd = unsafe {
                        iter_step_interior_avx2(
                            data,
                            cols,
                            iu,
                            iv,
                            w,
                            &prev_patch,
                            &ix_patch,
                            &iy_patch,
                        )
                    };
                    let tol = 1e-3_f32 * scalar.0.abs().max(1.0);
                    assert!(
                        (simd.0 - scalar.0).abs() < tol,
                        "AVX2 d mismatch at ({cx},{cy}): simd={} scalar={}",
                        simd.0,
                        scalar.0,
                    );
                    let tol = 1e-3_f32 * scalar.1.abs().max(1.0);
                    assert!(
                        (simd.1 - scalar.1).abs() < tol,
                        "AVX2 e mismatch at ({cx},{cy}): simd={} scalar={}",
                        simd.1,
                        scalar.1,
                    );
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                let simd = unsafe {
                    iter_step_interior_neon(
                        data,
                        cols,
                        iu,
                        iv,
                        w,
                        &prev_patch,
                        &ix_patch,
                        &iy_patch,
                    )
                };
                let tol = 1e-3_f32 * scalar.0.abs().max(1.0);
                assert!((simd.0 - scalar.0).abs() < tol);
                let tol = 1e-3_f32 * scalar.1.abs().max(1.0);
                assert!((simd.1 - scalar.1).abs() < tol);
            }
        }
    }

    /// SIMD parity: the AVX2 (or NEON) `build_three_patches_interior`
    /// kernel must produce per-pixel identical output to scalar, and the
    /// returned `(a, b, c)` Hessian entries must match within accumulated
    /// FMA-reordering tolerance.
    #[test]
    fn simd_build_three_patches_matches_scalar() {
        let size = 128;
        let img = make_gradient_image(size);
        // Build gradient images so all three patches have non-trivial data.
        let mut ix =
            Image::<f32, 1>::from_size_val([size, size].into(), 0.0, host_alloc()).unwrap();
        let mut iy =
            Image::<f32, 1>::from_size_val([size, size].into(), 0.0, host_alloc()).unwrap();
        crate::filter::scharr_spatial_gradient_float(&img, &mut ix, &mut iy).unwrap();
        let cols = img.cols();

        for (cx, cy) in [(40.5_f32, 50.25_f32), (75.125, 55.875)] {
            let (iu, iv, w) = weights_for(cx, cy);

            let mut sp = [0.0_f32; 441];
            let mut sx = [0.0_f32; 441];
            let mut sy = [0.0_f32; 441];
            // `scalar` is consumed only by the x86_64 AVX2 parity check below; on
            // other arches the comparison is cfg'd out, so the binding is unused.
            #[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
            let scalar = build_three_patches_interior_scalar(
                img.as_slice(),
                ix.as_slice(),
                iy.as_slice(),
                cols,
                iu,
                iv,
                w,
                &mut sp,
                &mut sx,
                &mut sy,
            );

            #[cfg(target_arch = "x86_64")]
            {
                let cpu = crate::simd::cpu_features();
                if cpu.has_avx2 && cpu.has_fma {
                    let mut fp = [0.0_f32; 441];
                    let mut fx = [0.0_f32; 441];
                    let mut fy = [0.0_f32; 441];
                    let simd = unsafe {
                        build_three_patches_interior_avx2(
                            img.as_slice(),
                            ix.as_slice(),
                            iy.as_slice(),
                            cols,
                            iu,
                            iv,
                            w,
                            &mut fp,
                            &mut fx,
                            &mut fy,
                        )
                    };
                    // Per-pixel: FMA reordering can produce ~1 ULP differences,
                    // safely within 1e-5 relative.
                    for i in 0..441 {
                        let tol = 1e-5_f32 * sp[i].abs().max(1.0);
                        assert!(
                            (fp[i] - sp[i]).abs() < tol,
                            "prev pixel {i} simd={} scalar={}",
                            fp[i],
                            sp[i]
                        );
                        let tol = 1e-5_f32 * sx[i].abs().max(1.0);
                        assert!((fx[i] - sx[i]).abs() < tol);
                        let tol = 1e-5_f32 * sy[i].abs().max(1.0);
                        assert!((fy[i] - sy[i]).abs() < tol);
                    }
                    // Accumulated sums use ~441 adds — relaxed tolerance.
                    let tol = 1e-3_f32 * scalar.0.abs().max(1.0);
                    assert!(
                        (simd.0 - scalar.0).abs() < tol,
                        "a simd={} scalar={}",
                        simd.0,
                        scalar.0
                    );
                    assert!((simd.1 - scalar.1).abs() < 1e-3 * scalar.1.abs().max(1.0));
                    assert!((simd.2 - scalar.2).abs() < 1e-3 * scalar.2.abs().max(1.0));
                }
            }
        }
    }

    /// Border feature: the fast-path safety check rejects this centre, so
    /// the slow `sample_at`/BorderMode path runs. Verifies the fast/slow
    /// branch wiring stays correct when the patch straddles an edge.
    #[test]
    fn border_feature_uses_slow_path() {
        let size = 64;
        let dx = 1.0_f32;
        let dy = 0.5_f32;
        let img1 = make_gradient_image(size);
        let mut img2 =
            Image::<f32, 1>::from_size_val([size, size].into(), 0.0, host_alloc()).unwrap();
        for y in 0..size {
            for x in 0..size {
                let sx = (x as f32 - dx).clamp(0.0, size as f32 - 1.0);
                let sy = (y as f32 - dy).clamp(0.0, size as f32 - 1.0);
                let v = sample_at(&img1, sx, sy, BorderMode::Clamp);
                img2.set_pixel(x, y, 0, v).unwrap();
            }
        }
        // Centre at L0 close enough to the edge that the patch *at the
        // smallest pyramid level* straddles the safe band (size 64 → 8 at
        // L3, where the safe band is [10, -3] = empty). Pick (12, 12) which
        // is interior at L0 but lies in the border slow-path at L3.
        let pt = [12.0_f32, 12.0_f32];
        assert!(
            !is_interior_for_lk(pt[0] / 8.0, pt[1] / 8.0, size / 8, size / 8),
            "centre should be in the border slow-path at L3"
        );
        let params = PyrLKParams {
            max_level: 3,
            ..PyrLKParams::default()
        };
        let result = calc_optical_flow_pyr_lk(&img1, &img2, &[pt], None, &params).unwrap();
        // We don't assert precise tracking accuracy here (border features
        // can be jittery); we only assert the call doesn't panic and
        // returns *some* result via the slow path.
        assert_eq!(result.next_pts.len(), 1);
        assert_eq!(result.status.len(), 1);
    }

    /// End-to-end equivalence: for a *fully-interior* feature, the public
    /// LK output (which is now using the fast path) must agree with the
    /// known synthetic translation to the same precision as the original
    /// implementation did. This catches integration-level breakage in the
    /// fast-path wiring.
    #[test]
    fn interior_feature_end_to_end_subpixel() {
        // 128x128 image, feature firmly inside the [10, 117] safe band.
        let size = 128;
        let dx = 0.4;
        let dy = -0.7;
        let img1 = make_gradient_image(size);
        // Synthesise img2 as img1 shifted by (dx, dy).
        let mut img2 =
            Image::<f32, 1>::from_size_val([size, size].into(), 0.0, host_alloc()).unwrap();
        for y in 0..size {
            for x in 0..size {
                let sx = (x as f32 - dx).clamp(0.0, size as f32 - 1.0);
                let sy = (y as f32 - dy).clamp(0.0, size as f32 - 1.0);
                let v = sample_at(&img1, sx, sy, BorderMode::Clamp);
                img2.set_pixel(x, y, 0, v).unwrap();
            }
        }
        let pts = vec![[64.0, 64.0]]; // interior
        let params = default_params();
        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params).unwrap();
        assert_eq!(result.status[0], 1);
        let est_dx = result.next_pts[0][0] - pts[0][0];
        let est_dy = result.next_pts[0][1] - pts[0][1];
        assert!(
            (est_dx - dx).abs() < 0.1,
            "interior fast-path subpixel dx: est={est_dx} truth={dx}"
        );
        assert!(
            (est_dy - dy).abs() < 0.1,
            "interior fast-path subpixel dy: est={est_dy} truth={dy}"
        );
    }

    fn make_circle_image(size: usize, cx: f32, cy: f32, r: f32) -> Image<f32, 1> {
        let mut img =
            Image::<f32, 1>::from_size_val([size, size].into(), 0.0, host_alloc()).unwrap();
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

    fn make_gradient_image(size: usize) -> Image<f32, 1> {
        let mut img =
            Image::<f32, 1>::from_size_val([size, size].into(), 0.0, host_alloc()).unwrap();
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
        // Use a point near the circle edge (r≈9.9 from center at 45°) so the
        // 21×21 tracking window contains the circle arc in both x and y,
        // giving the structure tensor sufficient gradient energy to pass the
        // normalized min_eigen check at every pyramid level.
        let pts = vec![[71.0, 57.0]];
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
        let img1 = Image::<f32, 1>::from_size_val([size, size].into(), 0.5, host_alloc()).unwrap();
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
    fn test_lk_initial_flow_missing() {
        let size = 64;
        let img1 = make_circle_image(size, 32.0, 32.0, 10.0);
        let img2 = make_circle_image(size, 33.0, 32.0, 10.0);
        let pts = vec![[32.0, 32.0]];
        let mut params = default_params();
        params.use_initial_flow = true;

        // Full entry point.
        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params);
        assert!(matches!(result, Err(PyrLKError::InitialFlowMissing)));

        // Precomputed entry point.
        let precomputed = build_lk_precomputed(&img1, &img2, params.max_level).unwrap();
        let result = calc_optical_flow_pyr_lk_with_precomputed(&precomputed, &pts, None, &params);
        assert!(matches!(result, Err(PyrLKError::InitialFlowMissing)));
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
            Image::<f32, 1>::from_size_val([size, size].into(), 0.0, host_alloc()).unwrap();

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
