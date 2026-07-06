//! Fused `resize + normalize + layout-convert` kernels.
//!
//! Collapses the standard 3-buffer ML preprocess chain into a single pass:
//! every input byte touched once, every output byte written once, nothing
//! in between leaves L1.
//!
//! Numerically *more* accurate than the separate pipeline because the 2×2
//! average stays in f32 (the `/4` is folded into `scale`) instead of being
//! requantized to u8 between resize and normalize.

use rayon::prelude::*;

/// Per-channel ImageNet-style normalization parameters.
///
/// `scale[c] = 1.0 / (std[c] * 255.0)` and `bias[c] = -mean[c] / std[c]` so the
/// fused kernel can compute `out = src_u8 * scale + bias` with a single FMA
/// per lane instead of `(src/255 - mean) / std`.
#[derive(Copy, Clone, Debug)]
pub struct NormalizeParams<const C: usize> {
    /// Per-channel multiplicative factor applied to the raw u8 value.
    pub scale: [f32; C],
    /// Per-channel additive bias applied after `scale`.
    pub bias: [f32; C],
}

impl<const C: usize> NormalizeParams<C> {
    /// Build from per-channel mean+std in [0, 1] range (the PyTorch convention).
    /// `u8 x` → `(x/255 - mean) / std` → `x * (1/(std*255)) + (-mean/std)`.
    pub fn from_mean_std(mean: [f32; C], std: [f32; C]) -> Self {
        let mut scale = [0f32; C];
        let mut bias = [0f32; C];
        for c in 0..C {
            scale[c] = 1.0 / (std[c] * 255.0);
            bias[c] = -mean[c] / std[c];
        }
        Self { scale, bias }
    }
}

/// Fused 2× exact-downscale bilinear + per-channel normalize + `HWC→CHW` layout
/// conversion for RGB u8 → f32.
///
/// # Arguments
///
/// * `src`    — `[src_h × src_w × 3]` row-major u8 input
/// * `src_w`  — input width in pixels (must equal `2 * dst_w`)
/// * `src_h`  — input height in pixels (must equal `2 * dst_h`)
/// * `dst`    — `[3 × dst_h × dst_w]` f32 output (NCHW, channels-first)
/// * `dst_w`  — output width
/// * `dst_h`  — output height
/// * `params` — per-channel pre-combined scale/bias from [`NormalizeParams`]
///
/// # Errors
///
/// Returns [`kornia_image::ImageError`] if `src` or `dst` length does not match
/// the expected size, or if `src_w != 2 * dst_w` or `src_h != 2 * dst_h`.
pub fn resize_normalize_to_tensor_u8_to_f32(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
    params: &NormalizeParams<3>,
) -> Result<(), kornia_image::ImageError> {
    // Real runtime validation (release-safe)
    if src_w != 2 * dst_w || src_h != 2 * dst_h {
        return Err(kornia_image::ImageError::InvalidImageSize(
            src_w,
            src_h,
            dst_w * 2,
            dst_h * 2,
        ));
    }
    if src.len() != src_h * src_w * 3 {
        return Err(kornia_image::ImageError::InvalidChannelShape(
            src.len(),
            src_h * src_w * 3,
        ));
    }
    if dst.len() != 3 * dst_h * dst_w {
        return Err(kornia_image::ImageError::InvalidChannelShape(
            dst.len(),
            3 * dst_h * dst_w,
        ));
    }

    debug_assert_eq!(src_w, 2 * dst_w);
    debug_assert_eq!(src_h, 2 * dst_h);
    debug_assert_eq!(src.len(), src_h * src_w * 3);
    debug_assert_eq!(dst.len(), 3 * dst_h * dst_w);

    let src_stride = src_w * 3;
    let plane_size = dst_h * dst_w;

    // NCHW output: three contiguous planes. Split the mutable borrow once so
    // each task writes to all 3 planes independently.
    let (r_plane, rest) = dst.split_at_mut(plane_size);
    let (g_plane, b_plane) = rest.split_at_mut(plane_size);

    // Group 8 dst rows per rayon task. Matches `pyrdown_2x_rgb_u8`'s shape —
    // one-row-per-task costs ~5 μs of spawn overhead × 540 rows = ~2.7 ms of
    // unnecessary dispatch at 1080p→540p, which dominates the kernel itself.
    const ROWS_PER_TASK: usize = 8;
    let chunk = dst_w * ROWS_PER_TASK;

    r_plane
        .par_chunks_mut(chunk)
        .zip(g_plane.par_chunks_mut(chunk))
        .zip(b_plane.par_chunks_mut(chunk))
        .enumerate()
        .for_each(|(ti, ((r_chunk, g_chunk), b_chunk))| {
            let y_start = ti * ROWS_PER_TASK;
            let nrows = r_chunk.len() / dst_w;
            for dy in 0..nrows {
                let y = y_start + dy;
                let r0 = &src[(2 * y) * src_stride..(2 * y + 1) * src_stride];
                let r1 = &src[(2 * y + 1) * src_stride..(2 * y + 2) * src_stride];
                let r_row = &mut r_chunk[dy * dst_w..(dy + 1) * dst_w];
                let g_row = &mut g_chunk[dy * dst_w..(dy + 1) * dst_w];
                let b_row = &mut b_chunk[dy * dst_w..(dy + 1) * dst_w];
                fused_row_rgb_u8_to_nchw_f32(r0, r1, r_row, g_row, b_row, dst_w, params);
            }
        });

    Ok(())
}

/// Fused **general bilinear** resize + per-channel normalize + HWC→CHW for RGB
/// u8 → f32, for **arbitrary** src→dst sizes (up- or down-scale).
///
/// When `src == 2×dst` exactly it dispatches to the dedicated box-average NEON
/// path ([`resize_normalize_to_tensor_u8_to_f32`]); otherwise it bilinearly
/// samples with OpenCV's half-pixel mapping (`src = (dst+0.5)·scale − 0.5`),
/// interpolating in f32 so there is no intermediate u8 requantization — the
/// result is at least as accurate as `cv2.resize`-then-normalize and stays a
/// single pass (every output byte written once, directly in CHW layout).
///
/// * `src` — `[src_h × src_w × 3]` row-major u8 (HWC)
/// * `dst` — `[3 × dst_h × dst_w]` f32 output (CHW), `out = sample·scale + bias`
///
/// # Errors
///
/// Returns [`kornia_image::ImageError`] if `src` or `dst` length does not match
/// the expected size, or (for the 2× variant) if `src_w != 2 * dst_w` or
/// `src_h != 2 * dst_h`.
pub fn resize_normalize_to_tensor_u8_to_f32_bilinear(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
    params: &NormalizeParams<3>,
) -> Result<(), kornia_image::ImageError> {
    if src.len() != src_h * src_w * 3 {
        return Err(kornia_image::ImageError::InvalidChannelShape(
            src.len(),
            src_h * src_w * 3,
        ));
    }
    if dst.len() != 3 * dst_h * dst_w {
        return Err(kornia_image::ImageError::InvalidChannelShape(
            dst.len(),
            3 * dst_h * dst_w,
        ));
    }

    debug_assert_eq!(src.len(), src_h * src_w * 3);
    debug_assert_eq!(dst.len(), 3 * dst_h * dst_w);

    // Exact 2× downscale → dedicated fused box kernel (fully NEON-vectorized).
    if src_w == 2 * dst_w && src_h == 2 * dst_h {
        return resize_normalize_to_tensor_u8_to_f32(src, src_w, src_h, dst, dst_w, dst_h, params);
    }
    if dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 {
        return Ok(());
    }

    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;
    let src_stride = src_w * 3;

    // Per-output-column source byte offsets (already ×3 for the R lane) + weight,
    // computed once and reused across every row (OpenCV half-pixel convention).
    let mut x0b = vec![0usize; dst_w];
    let mut x1b = vec![0usize; dst_w];
    let mut wx = vec![0f32; dst_w];
    for dx in 0..dst_w {
        let fx = ((dx as f32 + 0.5) * scale_x - 0.5).max(0.0);
        let x0 = (fx as usize).min(src_w - 1);
        let x1 = (x0 + 1).min(src_w - 1);
        x0b[dx] = x0 * 3;
        x1b[dx] = x1 * 3;
        wx[dx] = fx - x0 as f32;
    }

    let plane = dst_h * dst_w;
    let (r_plane, rest) = dst.split_at_mut(plane);
    let (g_plane, b_plane) = rest.split_at_mut(plane);

    const ROWS_PER_TASK: usize = 8;
    let chunk = dst_w * ROWS_PER_TASK;
    r_plane
        .par_chunks_mut(chunk)
        .zip(g_plane.par_chunks_mut(chunk))
        .zip(b_plane.par_chunks_mut(chunk))
        .enumerate()
        .for_each(|(ti, ((rc, gc), bc))| {
            let y_start = ti * ROWS_PER_TASK;
            let nrows = rc.len() / dst_w;
            for dy in 0..nrows {
                let y = y_start + dy;
                let fy = ((y as f32 + 0.5) * scale_y - 0.5).max(0.0);
                let y0 = (fy as usize).min(src_h - 1);
                let y1 = (y0 + 1).min(src_h - 1);
                let wy = fy - y0 as f32;
                let row0 = &src[y0 * src_stride..y0 * src_stride + src_stride];
                let row1 = &src[y1 * src_stride..y1 * src_stride + src_stride];
                let ro = &mut rc[dy * dst_w..dy * dst_w + dst_w];
                let go = &mut gc[dy * dst_w..dy * dst_w + dst_w];
                let bo = &mut bc[dy * dst_w..dy * dst_w + dst_w];
                fused_bilinear_row(row0, row1, &x0b, &x1b, &wx, wy, ro, go, bo, dst_w, params);
            }
        });

    Ok(())
}

/// One output row of the general bilinear path: dispatch to NEON / AVX2 / scalar.
/// `x0b`/`x1b` are per-column source byte offsets (R lane), `wx` the x-weights.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn fused_bilinear_row(
    row0: &[u8],
    row1: &[u8],
    x0b: &[usize],
    x1b: &[usize],
    wx: &[f32],
    wy: f32,
    r_out: &mut [f32],
    g_out: &mut [f32],
    b_out: &mut [f32],
    dst_w: usize,
    params: &NormalizeParams<3>,
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is architectural on aarch64.
    unsafe {
        fused_bilinear_row_neon(
            row0, row1, x0b, x1b, wx, wy, r_out, g_out, b_out, dst_w, params,
        );
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if crate::simd::cpu_features().has_avx2 && crate::simd::cpu_features().has_fma {
        unsafe {
            fused_bilinear_row_avx2(
                row0, row1, x0b, x1b, wx, wy, r_out, g_out, b_out, dst_w, params,
            )
        };
        return;
    }
    #[allow(unreachable_code)]
    fused_bilinear_row_scalar(
        row0, row1, x0b, x1b, wx, wy, r_out, g_out, b_out, dst_w, params,
    );
}

/// Scalar reference / tail for the general bilinear row.
#[allow(clippy::too_many_arguments)]
#[inline]
fn fused_bilinear_row_scalar(
    row0: &[u8],
    row1: &[u8],
    x0b: &[usize],
    x1b: &[usize],
    wx: &[f32],
    wy: f32,
    r_out: &mut [f32],
    g_out: &mut [f32],
    b_out: &mut [f32],
    dst_w: usize,
    params: &NormalizeParams<3>,
) {
    let blerp = |a: f32, b: f32, c: f32, d: f32, w: f32| {
        let top = a + w * (b - a);
        let bot = c + w * (d - c);
        top + wy * (bot - top)
    };
    for dx in 0..dst_w {
        let (o0, o1, w) = (x0b[dx], x1b[dx], wx[dx]);
        let r = blerp(
            row0[o0] as f32,
            row0[o1] as f32,
            row1[o0] as f32,
            row1[o1] as f32,
            w,
        );
        let g = blerp(
            row0[o0 + 1] as f32,
            row0[o1 + 1] as f32,
            row1[o0 + 1] as f32,
            row1[o1 + 1] as f32,
            w,
        );
        let b = blerp(
            row0[o0 + 2] as f32,
            row0[o1 + 2] as f32,
            row1[o0 + 2] as f32,
            row1[o1 + 2] as f32,
            w,
        );
        r_out[dx] = r * params.scale[0] + params.bias[0];
        g_out[dx] = g * params.scale[1] + params.bias[1];
        b_out[dx] = b * params.scale[2] + params.bias[2];
    }
}

/// NEON general-bilinear row: 4 dst px/iter. Gathers the 4×4 corner samples per
/// channel into stack arrays (the only scalar part — bilinear has arbitrary src
/// offsets), then does the bilinear blend + scale/bias as f32x4 FMAs and writes
/// 4 contiguous outputs per plane.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn fused_bilinear_row_neon(
    row0: &[u8],
    row1: &[u8],
    x0b: &[usize],
    x1b: &[usize],
    wx: &[f32],
    wy: f32,
    r_out: &mut [f32],
    g_out: &mut [f32],
    b_out: &mut [f32],
    dst_w: usize,
    params: &NormalizeParams<3>,
) {
    unsafe {
        use std::arch::aarch64::*;
        let sc = [
            vdupq_n_f32(params.scale[0]),
            vdupq_n_f32(params.scale[1]),
            vdupq_n_f32(params.scale[2]),
        ];
        let bi = [
            vdupq_n_f32(params.bias[0]),
            vdupq_n_f32(params.bias[1]),
            vdupq_n_f32(params.bias[2]),
        ];
        let wy4 = vdupq_n_f32(wy);

        let bulk = dst_w & !3;
        let mut dx = 0;
        while dx < bulk {
            let wx4 = vld1q_f32(wx.as_ptr().add(dx));
            // Gather the 4 corner samples for the 4 output pixels, all channels.
            let (mut p00, mut p01, mut p10, mut p11) =
                ([0f32; 12], [0f32; 12], [0f32; 12], [0f32; 12]);
            for k in 0..4 {
                let o0 = *x0b.get_unchecked(dx + k);
                let o1 = *x1b.get_unchecked(dx + k);
                for c in 0..3 {
                    p00[k * 3 + c] = *row0.get_unchecked(o0 + c) as f32;
                    p01[k * 3 + c] = *row0.get_unchecked(o1 + c) as f32;
                    p10[k * 3 + c] = *row1.get_unchecked(o0 + c) as f32;
                    p11[k * 3 + c] = *row1.get_unchecked(o1 + c) as f32;
                }
            }
            // Per channel: deinterleave the 4 lanes via a strided load into f32x4,
            // bilinear blend, scale/bias, contiguous store.
            for (c, out) in [r_out.as_mut_ptr(), g_out.as_mut_ptr(), b_out.as_mut_ptr()]
                .into_iter()
                .enumerate()
            {
                let gather = |arr: &[f32; 12]| {
                    vld1q_f32([arr[c], arr[3 + c], arr[6 + c], arr[9 + c]].as_ptr())
                };
                let a = gather(&p00);
                let b = gather(&p01);
                let cc = gather(&p10);
                let d = gather(&p11);
                let top = vfmaq_f32(a, wx4, vsubq_f32(b, a));
                let bot = vfmaq_f32(cc, wx4, vsubq_f32(d, cc));
                let val = vfmaq_f32(top, wy4, vsubq_f32(bot, top));
                vst1q_f32(out.add(dx), vfmaq_f32(bi[c], val, sc[c]));
            }
            dx += 4;
        }
        if dx < dst_w {
            fused_bilinear_row_scalar(
                row0,
                row1,
                &x0b[dx..],
                &x1b[dx..],
                &wx[dx..],
                wy,
                &mut r_out[dx..],
                &mut g_out[dx..],
                &mut b_out[dx..],
                dst_w - dx,
                params,
            );
        }
    }
}

/// AVX2/FMA general-bilinear row: 8 dst px/iter. Same gather→blend→scale shape as
/// the NEON path, widened to 8-wide `__m256`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn fused_bilinear_row_avx2(
    row0: &[u8],
    row1: &[u8],
    x0b: &[usize],
    x1b: &[usize],
    wx: &[f32],
    wy: f32,
    r_out: &mut [f32],
    g_out: &mut [f32],
    b_out: &mut [f32],
    dst_w: usize,
    params: &NormalizeParams<3>,
) {
    use std::arch::x86_64::*;
    let sc = [
        _mm256_set1_ps(params.scale[0]),
        _mm256_set1_ps(params.scale[1]),
        _mm256_set1_ps(params.scale[2]),
    ];
    let bi = [
        _mm256_set1_ps(params.bias[0]),
        _mm256_set1_ps(params.bias[1]),
        _mm256_set1_ps(params.bias[2]),
    ];
    let wy8 = _mm256_set1_ps(wy);

    let bulk = dst_w & !7;
    let mut dx = 0;
    while dx < bulk {
        let wx8 = _mm256_loadu_ps(wx.as_ptr().add(dx));
        let (mut p00, mut p01, mut p10, mut p11) = ([0f32; 24], [0f32; 24], [0f32; 24], [0f32; 24]);
        for k in 0..8 {
            let o0 = *x0b.get_unchecked(dx + k);
            let o1 = *x1b.get_unchecked(dx + k);
            for c in 0..3 {
                p00[k * 3 + c] = *row0.get_unchecked(o0 + c) as f32;
                p01[k * 3 + c] = *row0.get_unchecked(o1 + c) as f32;
                p10[k * 3 + c] = *row1.get_unchecked(o0 + c) as f32;
                p11[k * 3 + c] = *row1.get_unchecked(o1 + c) as f32;
            }
        }
        for (c, out) in [r_out.as_mut_ptr(), g_out.as_mut_ptr(), b_out.as_mut_ptr()]
            .into_iter()
            .enumerate()
        {
            let gather = |arr: &[f32; 24]| {
                _mm256_setr_ps(
                    arr[c],
                    arr[3 + c],
                    arr[6 + c],
                    arr[9 + c],
                    arr[12 + c],
                    arr[15 + c],
                    arr[18 + c],
                    arr[21 + c],
                )
            };
            let a = gather(&p00);
            let b = gather(&p01);
            let cc = gather(&p10);
            let d = gather(&p11);
            let top = _mm256_fmadd_ps(_mm256_sub_ps(b, a), wx8, a);
            let bot = _mm256_fmadd_ps(_mm256_sub_ps(d, cc), wx8, cc);
            let val = _mm256_fmadd_ps(_mm256_sub_ps(bot, top), wy8, top);
            _mm256_storeu_ps(out.add(dx), _mm256_fmadd_ps(val, sc[c], bi[c]));
        }
        dx += 8;
    }
    if dx < dst_w {
        fused_bilinear_row_scalar(
            row0,
            row1,
            &x0b[dx..],
            &x1b[dx..],
            &wx[dx..],
            wy,
            &mut r_out[dx..],
            &mut g_out[dx..],
            &mut b_out[dx..],
            dst_w - dx,
            params,
        );
    }
}

/// Row-level fused kernel: one 2× downscale + normalize + HWC→CHW split.
#[inline(always)]
fn fused_row_rgb_u8_to_nchw_f32(
    r0: &[u8],
    r1: &[u8],
    r_out: &mut [f32],
    g_out: &mut [f32],
    b_out: &mut [f32],
    dst_w: usize,
    params: &NormalizeParams<3>,
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is architectural on aarch64.
    unsafe {
        fused_row_neon(r0, r1, r_out, g_out, b_out, dst_w, params);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if crate::simd::cpu_features().has_avx2 && crate::simd::cpu_features().has_fma {
        unsafe { fused_row_avx2(r0, r1, r_out, g_out, b_out, dst_w, params) };
        return;
    }
    #[allow(unreachable_code)]
    fused_row_scalar(r0, r1, r_out, g_out, b_out, dst_w, params);
}

/// Scalar reference: f32-precise (no u8 quantization between avg and normalize).
#[allow(dead_code)]
#[inline]
fn fused_row_scalar(
    r0: &[u8],
    r1: &[u8],
    r_out: &mut [f32],
    g_out: &mut [f32],
    b_out: &mut [f32],
    dst_w: usize,
    params: &NormalizeParams<3>,
) {
    let sr = params.scale[0] * 0.25;
    let sg = params.scale[1] * 0.25;
    let sb = params.scale[2] * 0.25;
    let br = params.bias[0];
    let bg = params.bias[1];
    let bb = params.bias[2];
    for x in 0..dst_w {
        let base0 = 2 * x * 3;
        let base1 = base0 + 3;
        let sum_r = r0[base0] as u32 + r0[base1] as u32 + r1[base0] as u32 + r1[base1] as u32;
        let sum_g = r0[base0 + 1] as u32
            + r0[base1 + 1] as u32
            + r1[base0 + 1] as u32
            + r1[base1 + 1] as u32;
        let sum_b = r0[base0 + 2] as u32
            + r0[base1 + 2] as u32
            + r1[base0 + 2] as u32
            + r1[base1 + 2] as u32;
        r_out[x] = (sum_r as f32) * sr + br;
        g_out[x] = (sum_g as f32) * sg + bg;
        b_out[x] = (sum_b as f32) * sb + bb;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fused_row_neon(
    r0: &[u8],
    r1: &[u8],
    r_out: &mut [f32],
    g_out: &mut [f32],
    b_out: &mut [f32],
    dst_w: usize,
    params: &NormalizeParams<3>,
) {
    unsafe {
        use std::arch::aarch64::*;

        // Fold the 2×2 average's /4 into scale so the hot loop is pure FMA —
        // no integer rounding, no u8 requantization.
        let sr = vdupq_n_f32(params.scale[0] * 0.25);
        let sg = vdupq_n_f32(params.scale[1] * 0.25);
        let sb = vdupq_n_f32(params.scale[2] * 0.25);
        let br = vdupq_n_f32(params.bias[0]);
        let bg = vdupq_n_f32(params.bias[1]);
        let bb = vdupq_n_f32(params.bias[2]);

        // Per-channel 16-output-pixel emitter. Takes the 4 deinterleaved u8x16
        // channel lanes (two adjacent src chunks per row, two rows), computes
        // the 2×2 sum in u16, then widens→cvts→FMAs into 4 f32x4 stores.
        //
        // Written as a macro (not a closure) because unsafe + captured `_v`
        // vectors inside `#[target_feature]` contexts are easier to reason about
        // when fully inlined, and Rust closures don't inherit target_feature.
        macro_rules! emit_channel {
            ($a:expr_2021, $b:expr_2021, $c:expr_2021, $d:expr_2021, $scale:expr_2021, $bias:expr_2021, $out:expr_2021) => {{
                let lo = vaddq_u16(vpaddlq_u8($a), vpaddlq_u8($c)); // out px 0..7
                let hi = vaddq_u16(vpaddlq_u8($b), vpaddlq_u8($d)); // out px 8..15
                let f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
                let f1 = vcvtq_f32_u32(vmovl_high_u16(lo));
                let f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
                let f3 = vcvtq_f32_u32(vmovl_high_u16(hi));
                vst1q_f32($out, vfmaq_f32($bias, f0, $scale));
                vst1q_f32($out.add(4), vfmaq_f32($bias, f1, $scale));
                vst1q_f32($out.add(8), vfmaq_f32($bias, f2, $scale));
                vst1q_f32($out.add(12), vfmaq_f32($bias, f3, $scale));
            }};
        }

        let bulk = dst_w & !15;
        let mut x = 0usize;
        while x < bulk {
            // 16 dst px → 32 src px per row → 2 × vld3q_u8 per row.
            let src0 = r0.as_ptr().add(x * 2 * 3);
            let src1 = r1.as_ptr().add(x * 2 * 3);
            let s0 = vld3q_u8(src0);
            let s1 = vld3q_u8(src0.add(48));
            let s2 = vld3q_u8(src1);
            let s3 = vld3q_u8(src1.add(48));

            emit_channel!(s0.0, s1.0, s2.0, s3.0, sr, br, r_out.as_mut_ptr().add(x));
            emit_channel!(s0.1, s1.1, s2.1, s3.1, sg, bg, g_out.as_mut_ptr().add(x));
            emit_channel!(s0.2, s1.2, s2.2, s3.2, sb, bb, b_out.as_mut_ptr().add(x));

            x += 16;
        }

        // Scalar tail for the last (dst_w % 16) pixels.
        let sr_s = params.scale[0] * 0.25;
        let sg_s = params.scale[1] * 0.25;
        let sb_s = params.scale[2] * 0.25;
        let br_s = params.bias[0];
        let bg_s = params.bias[1];
        let bb_s = params.bias[2];
        while x < dst_w {
            let base0 = 2 * x * 3;
            let base1 = base0 + 3;
            let sum_r = *r0.get_unchecked(base0) as u32
                + *r0.get_unchecked(base1) as u32
                + *r1.get_unchecked(base0) as u32
                + *r1.get_unchecked(base1) as u32;
            let sum_g = *r0.get_unchecked(base0 + 1) as u32
                + *r0.get_unchecked(base1 + 1) as u32
                + *r1.get_unchecked(base0 + 1) as u32
                + *r1.get_unchecked(base1 + 1) as u32;
            let sum_b = *r0.get_unchecked(base0 + 2) as u32
                + *r0.get_unchecked(base1 + 2) as u32
                + *r1.get_unchecked(base0 + 2) as u32
                + *r1.get_unchecked(base1 + 2) as u32;
            *r_out.get_unchecked_mut(x) = (sum_r as f32) * sr_s + br_s;
            *g_out.get_unchecked_mut(x) = (sum_g as f32) * sg_s + bg_s;
            *b_out.get_unchecked_mut(x) = (sum_b as f32) * sb_s + bb_s;
            x += 1;
        }
    }
}

// =============================================================================
// AVX2/FMA port of the fused 2×2-downscale + normalize + HWC→CHW row.
//
// AVX2 lacks a 3-way deinterleave (no `vld3q_u8` equivalent), so we replace it
// with 9× PSHUFB + 6× POR per 16-pixel chunk to extract R/G/B as separate u8x16
// vectors. The downstream 2×2 sum + widen-to-f32 + FMA chain then mirrors NEON
// one-for-one: `_mm_maddubs_epi16(x, set1_epi8(1))` replaces NEON's
// `vpaddlq_u8` (pairwise widening add) — both produce 8 u16 from 16 u8.
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn vld3_u8_avx2_x86(
    p: *const u8,
) -> (
    std::arch::x86_64::__m128i,
    std::arch::x86_64::__m128i,
    std::arch::x86_64::__m128i,
) {
    use std::arch::x86_64::*;
    let s0 = _mm_loadu_si128(p as *const __m128i);
    let s1 = _mm_loadu_si128(p.add(16) as *const __m128i);
    let s2 = _mm_loadu_si128(p.add(32) as *const __m128i);

    // R: s0 indices 0,3,6,9,12,15 → output 0..5; s1 indices 2,5,8,11,14 → 6..10;
    //    s2 indices 1,4,7,10,13 → 11..15. (-1 lanes are zero-fill via PSHUFB.)
    let m_r0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_r1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let m_r2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
    let m_g0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_g1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let m_g2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
    let m_b0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_b1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let m_b2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);

    let r = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(s0, m_r0), _mm_shuffle_epi8(s1, m_r1)),
        _mm_shuffle_epi8(s2, m_r2),
    );
    let g = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(s0, m_g0), _mm_shuffle_epi8(s1, m_g1)),
        _mm_shuffle_epi8(s2, m_g2),
    );
    let b = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(s0, m_b0), _mm_shuffle_epi8(s1, m_b1)),
        _mm_shuffle_epi8(s2, m_b2),
    );
    (r, g, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_row_avx2(
    r0: &[u8],
    r1: &[u8],
    r_out: &mut [f32],
    g_out: &mut [f32],
    b_out: &mut [f32],
    dst_w: usize,
    params: &NormalizeParams<3>,
) {
    use std::arch::x86_64::*;

    // Fold the 2×2 average's /4 into scale: hot loop is pure FMA, no requantize.
    let sr = _mm_set1_ps(params.scale[0] * 0.25);
    let sg = _mm_set1_ps(params.scale[1] * 0.25);
    let sb = _mm_set1_ps(params.scale[2] * 0.25);
    let br = _mm_set1_ps(params.bias[0]);
    let bg = _mm_set1_ps(params.bias[1]);
    let bb = _mm_set1_ps(params.bias[2]);
    let zero = _mm_setzero_si128();
    let one_epi8 = _mm_set1_epi8(1);

    // Per-channel emitter for 16 dst pixels at a time. The 4 args are
    // (a, b, c, d) = channel-deinterleaved u8x16 from row0-chunk0, row0-chunk1,
    // row1-chunk0, row1-chunk1 respectively.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn emit_channel(
        a: __m128i,
        b: __m128i,
        c: __m128i,
        d: __m128i,
        scale: __m128,
        bias: __m128,
        out: *mut f32,
        zero: __m128i,
        one_epi8: __m128i,
    ) {
        // Pairwise sum within each 16-byte chunk: maddubs(x, [1,1,1,...]) gives
        // 8 i16 = (x[0]+x[1], x[2]+x[3], ..., x[14]+x[15]). Sum of 2 such is
        // ≤ 1020 — fits easily in i16.
        let pa = _mm_maddubs_epi16(a, one_epi8);
        let pb = _mm_maddubs_epi16(b, one_epi8);
        let pc = _mm_maddubs_epi16(c, one_epi8);
        let pd = _mm_maddubs_epi16(d, one_epi8);
        let lo = _mm_add_epi16(pa, pc); // 8 i16 = output px 0..7 sums
        let hi = _mm_add_epi16(pb, pd); // 8 i16 = output px 8..15 sums

        // Widen i16 → i32 (values are non-negative, unpack-with-zero is correct).
        let i0 = _mm_unpacklo_epi16(lo, zero);
        let i1 = _mm_unpackhi_epi16(lo, zero);
        let i2 = _mm_unpacklo_epi16(hi, zero);
        let i3 = _mm_unpackhi_epi16(hi, zero);
        let f0 = _mm_cvtepi32_ps(i0);
        let f1 = _mm_cvtepi32_ps(i1);
        let f2 = _mm_cvtepi32_ps(i2);
        let f3 = _mm_cvtepi32_ps(i3);
        _mm_storeu_ps(out, _mm_fmadd_ps(f0, scale, bias));
        _mm_storeu_ps(out.add(4), _mm_fmadd_ps(f1, scale, bias));
        _mm_storeu_ps(out.add(8), _mm_fmadd_ps(f2, scale, bias));
        _mm_storeu_ps(out.add(12), _mm_fmadd_ps(f3, scale, bias));
    }

    let bulk = dst_w & !15;
    let mut x = 0usize;
    while x < bulk {
        // 16 dst px → 32 src px per row → 2× 48-byte chunks per row.
        let src0 = r0.as_ptr().add(x * 2 * 3);
        let src1 = r1.as_ptr().add(x * 2 * 3);
        let (r_a, g_a, b_a) = vld3_u8_avx2_x86(src0);
        let (r_b, g_b, b_b) = vld3_u8_avx2_x86(src0.add(48));
        let (r_c, g_c, b_c) = vld3_u8_avx2_x86(src1);
        let (r_d, g_d, b_d) = vld3_u8_avx2_x86(src1.add(48));

        emit_channel(
            r_a,
            r_b,
            r_c,
            r_d,
            sr,
            br,
            r_out.as_mut_ptr().add(x),
            zero,
            one_epi8,
        );
        emit_channel(
            g_a,
            g_b,
            g_c,
            g_d,
            sg,
            bg,
            g_out.as_mut_ptr().add(x),
            zero,
            one_epi8,
        );
        emit_channel(
            b_a,
            b_b,
            b_c,
            b_d,
            sb,
            bb,
            b_out.as_mut_ptr().add(x),
            zero,
            one_epi8,
        );

        x += 16;
    }

    // Scalar tail.
    let sr_s = params.scale[0] * 0.25;
    let sg_s = params.scale[1] * 0.25;
    let sb_s = params.scale[2] * 0.25;
    let br_s = params.bias[0];
    let bg_s = params.bias[1];
    let bb_s = params.bias[2];
    while x < dst_w {
        let base0 = 2 * x * 3;
        let base1 = base0 + 3;
        let sum_r = *r0.get_unchecked(base0) as u32
            + *r0.get_unchecked(base1) as u32
            + *r1.get_unchecked(base0) as u32
            + *r1.get_unchecked(base1) as u32;
        let sum_g = *r0.get_unchecked(base0 + 1) as u32
            + *r0.get_unchecked(base1 + 1) as u32
            + *r1.get_unchecked(base0 + 1) as u32
            + *r1.get_unchecked(base1 + 1) as u32;
        let sum_b = *r0.get_unchecked(base0 + 2) as u32
            + *r0.get_unchecked(base1 + 2) as u32
            + *r1.get_unchecked(base0 + 2) as u32
            + *r1.get_unchecked(base1 + 2) as u32;
        *r_out.get_unchecked_mut(x) = (sum_r as f32) * sr_s + br_s;
        *g_out.get_unchecked_mut(x) = (sum_g as f32) * sg_s + bg_s;
        *b_out.get_unchecked_mut(x) = (sum_b as f32) * sb_s + bb_s;
        x += 1;
    }
}

/// Split a `[3 * dst_h * dst_w]` CHW buffer into row bands of up to
/// `rows_per_band` destination rows, each owning disjoint mutable slices of
/// all three planes — the unit of rayon parallelism for the fused plane
/// writers below.
fn plane_bands(
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
    rows_per_band: usize,
) -> Vec<(usize, [&mut [f32]; 3])> {
    let pixels = dst_w * dst_h;
    let (p0, rest) = dst.split_at_mut(pixels);
    let (p1, p2) = rest.split_at_mut(pixels);
    let mut ps: [&mut [f32]; 3] = [p0, p1, p2];
    let mut out = Vec::with_capacity(dst_h.div_ceil(rows_per_band.max(1)));
    let mut y = 0usize;
    while y < dst_h {
        let rows = rows_per_band.min(dst_h - y);
        let mut band: Vec<&mut [f32]> = Vec::with_capacity(3);
        let mut rem: Vec<&mut [f32]> = Vec::with_capacity(3);
        for pl in ps {
            let (a, b) = pl.split_at_mut(rows * dst_w);
            band.push(a);
            rem.push(b);
        }
        out.push((y, band.try_into().unwrap()));
        ps = rem.try_into().unwrap();
        y += rows;
    }
    out
}

/// Fused **nearest** resize + per-channel normalize + `HWC→CHW` for RGB u8 →
/// f32, any target size, in a single pass (no intermediate u8 image).
///
/// Same centre-convention index maps as `resize_fast_u8` nearest, so the
/// sampled bytes are identical to the two-step path — only the u8→f32 FMA is
/// fused in.
pub fn resize_normalize_to_tensor_u8_to_f32_nearest(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
    params: &NormalizeParams<3>,
) -> Result<(), kornia_image::ImageError> {
    if src.len() != src_h * src_w * 3 {
        return Err(kornia_image::ImageError::InvalidChannelShape(
            src.len(),
            src_h * src_w * 3,
        ));
    }
    if dst.len() != 3 * dst_h * dst_w {
        return Err(kornia_image::ImageError::InvalidChannelShape(
            dst.len(),
            3 * dst_h * dst_w,
        ));
    }
    let sx = src_w as f64 / dst_w as f64;
    let sy = src_h as f64 / dst_h as f64;
    let xmap: Vec<usize> = (0..dst_w)
        .map(|x| ((((x as f64 + 0.5) * sx).floor() as i64).clamp(0, src_w as i64 - 1)) as usize)
        .collect();
    let (scale, bias) = (params.scale, params.bias);
    plane_bands(dst, dst_w, dst_h, 32)
        .into_par_iter()
        .for_each(|(y0, band)| {
            let rows = band[0].len() / dst_w;
            for r in 0..rows {
                let y = y0 + r;
                let yi =
                    ((((y as f64 + 0.5) * sy).floor() as i64).clamp(0, src_h as i64 - 1)) as usize;
                let srow = &src[yi * src_w * 3..(yi + 1) * src_w * 3];
                for (x, &xi) in xmap.iter().enumerate() {
                    let o = xi * 3;
                    band[0][r * dst_w + x] = srow[o] as f32 * scale[0] + bias[0];
                    band[1][r * dst_w + x] = srow[o + 1] as f32 * scale[1] + bias[1];
                    band[2][r * dst_w + x] = srow[o + 2] as f32 * scale[2] + bias[2];
                }
            }
        });
    Ok(())
}

/// Fused **separable (bicubic / lanczos, ± antialias)** resize + normalize +
/// `HWC→CHW` for RGB u8 → f32: the horizontal pass reuses the SIMD separable
/// machinery; the vertical pass accumulates in i32 and emits normalized f32
/// straight into the three planes — the final u8 quantization of the two-step
/// path is skipped entirely (one less pass AND more accurate).
#[allow(clippy::too_many_arguments)]
pub fn resize_normalize_to_tensor_u8_to_f32_separable(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
    params: &NormalizeParams<3>,
    filter: crate::interpolation::InterpolationMode,
    antialias: bool,
) -> Result<(), kornia_image::ImageError> {
    use super::common::{build_xsrc_lut, pack_xw_i16, precompute_contribs, FilterKind};
    use super::separable::horizontal_batch;

    if src.len() != src_h * src_w * 3 {
        return Err(kornia_image::ImageError::InvalidChannelShape(
            src.len(),
            src_h * src_w * 3,
        ));
    }
    if dst.len() != 3 * dst_h * dst_w {
        return Err(kornia_image::ImageError::InvalidChannelShape(
            dst.len(),
            3 * dst_h * dst_w,
        ));
    }
    let filt = match filter {
        crate::interpolation::InterpolationMode::Lanczos => FilterKind::Lanczos3,
        crate::interpolation::InterpolationMode::Bicubic => FilterKind::Cubic,
        other => {
            return Err(kornia_image::ImageError::UnsupportedInterpolation(other));
        }
    };
    const Q: i32 = 14;
    let (xofs, xw, kx) = precompute_contribs(src_w, dst_w, filt, antialias);
    let (yofs, yw, ky) = precompute_contribs(src_h, dst_h, filt, antialias);
    let xsrc = build_xsrc_lut(&xofs, dst_w, kx, src_w);
    let xw = pack_xw_i16(&xw);
    let last_sx_safe = src_w.saturating_sub(1);
    let src_stride = src_w * 3;
    let hbuf_row_len = dst_w * 3;
    let round1: i32 = 1 << (Q - 1);

    // Horizontal pass (SIMD row kernels) into the full i16 intermediate.
    let mut hbuf = vec![0i16; src_h * hbuf_row_len];
    const H_BATCH: usize = 16;
    hbuf.par_chunks_mut(hbuf_row_len * H_BATCH)
        .enumerate()
        .for_each(|(bi, batch_out)| {
            horizontal_batch::<3>(
                src,
                src_stride,
                batch_out,
                hbuf_row_len,
                bi * H_BATCH,
                dst_w,
                kx,
                &xsrc,
                &xw,
                last_sx_safe,
                round1,
            );
        });

    // Vertical pass: i32 accumulate over ky i16 rows, then normalize and emit
    // f32 planes directly — `acc` is the pixel value scaled by 2^14, so the
    // FMA constants fold the /2^14 in.
    let inv_q = 1.0f32 / (1 << Q) as f32;
    let s = [
        params.scale[0] * inv_q,
        params.scale[1] * inv_q,
        params.scale[2] * inv_q,
    ];
    let b = params.bias;

    plane_bands(dst, dst_w, dst_h, 16)
        .into_par_iter()
        .for_each(|(y0, band)| {
            let rows = band[0].len() / dst_w;
            for r in 0..rows {
                let yo = y0 + r;
                let y_base = yofs[yo];
                for x in 0..dst_w {
                    let mut acc = [0i32; 3];
                    for k in 0..ky {
                        let sy = (y_base + k as i32).clamp(0, src_h as i32 - 1) as usize;
                        let row = &hbuf[sy * hbuf_row_len..(sy + 1) * hbuf_row_len];
                        let w = yw[yo * ky + k];
                        acc[0] += row[x * 3] as i32 * w;
                        acc[1] += row[x * 3 + 1] as i32 * w;
                        acc[2] += row[x * 3 + 2] as i32 * w;
                    }
                    for c in 0..3 {
                        band[c][r * dst_w + x] = acc[c] as f32 * s[c] + b[c];
                    }
                }
            }
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare the fused kernel's output against an f64 reference on a
    /// synthetic image with a non-trivial remainder (dst_w not divisible
    /// by 16, so both the NEON bulk and the scalar tail are exercised).
    #[test]
    fn fused_2x_normalize_matches_f64_reference() {
        let dst_w = 37;
        let dst_h = 5;
        let src_w = 2 * dst_w;
        let src_h = 2 * dst_h;
        let src: Vec<u8> = (0..src_h * src_w * 3)
            .map(|i| ((i * 7 + 3) % 256) as u8)
            .collect();
        let mut dst = vec![0f32; 3 * dst_h * dst_w];

        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        let params = NormalizeParams::<3>::from_mean_std(mean, std);

        resize_normalize_to_tensor_u8_to_f32(&src, src_w, src_h, &mut dst, dst_w, dst_h, &params)
            .unwrap();

        // f64 reference: (avg(2x2)/255 - mean) / std, CHW layout.
        let plane = dst_h * dst_w;
        for y in 0..dst_h {
            for x in 0..dst_w {
                let b0 = (2 * y) * src_w * 3 + 2 * x * 3;
                let b1 = b0 + 3;
                let b2 = (2 * y + 1) * src_w * 3 + 2 * x * 3;
                let b3 = b2 + 3;
                for ch in 0..3 {
                    let sum = src[b0 + ch] as f64
                        + src[b1 + ch] as f64
                        + src[b2 + ch] as f64
                        + src[b3 + ch] as f64;
                    let expect = ((sum / 4.0) / 255.0 - mean[ch] as f64) / std[ch] as f64;
                    let got = dst[ch * plane + y * dst_w + x] as f64;
                    assert!(
                        (got - expect).abs() < 1e-4,
                        "ch={ch} y={y} x={x} got={got} expect={expect}"
                    );
                }
            }
        }
    }

    /// Corner test: a completely zero src image produces `-mean/std` on
    /// every output pixel (bias only).
    #[test]
    fn fused_2x_normalize_zero_input() {
        let dst_w = 16;
        let dst_h = 2;
        let src = vec![0u8; 2 * dst_w * 2 * dst_h * 3];
        let mut dst = vec![0f32; 3 * dst_h * dst_w];
        let mean = [0.5, 0.25, 0.75];
        let std = [0.5, 0.25, 0.75];
        let params = NormalizeParams::<3>::from_mean_std(mean, std);
        resize_normalize_to_tensor_u8_to_f32(
            &src,
            2 * dst_w,
            2 * dst_h,
            &mut dst,
            dst_w,
            dst_h,
            &params,
        )
        .unwrap();
        let plane = dst_h * dst_w;
        for ch in 0..3 {
            let expect = -mean[ch] / std[ch];
            for i in 0..plane {
                let got = dst[ch * plane + i];
                assert!(
                    (got - expect).abs() < 1e-6,
                    "ch={ch} i={i} got={got} expect={expect}"
                );
            }
        }
    }

    /// General bilinear path on a non-2× ratio matches an f64 bilinear reference
    /// (OpenCV half-pixel mapping) after normalize, in CHW layout.
    #[test]
    fn fused_bilinear_general_matches_f64_reference() {
        let (src_w, src_h) = (60, 40);
        let (dst_w, dst_h) = (37, 23); // arbitrary, non-integer ratio, with tail
        let src: Vec<u8> = (0..src_h * src_w * 3)
            .map(|i| ((i * 13 + 7) % 256) as u8)
            .collect();
        let mut dst = vec![0f32; 3 * dst_h * dst_w];
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        let params = NormalizeParams::<3>::from_mean_std(mean, std);

        resize_normalize_to_tensor_u8_to_f32_bilinear(
            &src, src_w, src_h, &mut dst, dst_w, dst_h, &params,
        )
        .unwrap();

        let sx = src_w as f64 / dst_w as f64;
        let sy = src_h as f64 / dst_h as f64;
        let at = |y: usize, x: usize, c: usize| src[(y * src_w + x) * 3 + c] as f64;
        let plane = dst_h * dst_w;
        for dy in 0..dst_h {
            let fy = ((dy as f64 + 0.5) * sy - 0.5).max(0.0);
            let y0 = (fy as usize).min(src_h - 1);
            let y1 = (y0 + 1).min(src_h - 1);
            let wy = fy - y0 as f64;
            for dx in 0..dst_w {
                let fx = ((dx as f64 + 0.5) * sx - 0.5).max(0.0);
                let x0 = (fx as usize).min(src_w - 1);
                let x1 = (x0 + 1).min(src_w - 1);
                let wx = fx - x0 as f64;
                for c in 0..3 {
                    let top = at(y0, x0, c) + wx * (at(y0, x1, c) - at(y0, x0, c));
                    let bot = at(y1, x0, c) + wx * (at(y1, x1, c) - at(y1, x0, c));
                    let val = top + wy * (bot - top);
                    let expect = (val / 255.0 - mean[c] as f64) / std[c] as f64;
                    let got = dst[c * plane + dy * dst_w + dx] as f64;
                    assert!(
                        (got - expect).abs() < 1e-3,
                        "c={c} dy={dy} dx={dx} got={got} expect={expect}"
                    );
                }
            }
        }
    }

    /// The 2× ratio dispatches into the box path; the bilinear entry must agree
    /// with the direct 2× call bit-for-bit (same buffer, same code).
    #[test]
    fn fused_bilinear_dispatches_2x() {
        let (dst_w, dst_h) = (20, 12);
        let (src_w, src_h) = (2 * dst_w, 2 * dst_h);
        let src: Vec<u8> = (0..src_h * src_w * 3).map(|i| (i % 251) as u8).collect();
        let params = NormalizeParams::<3>::from_mean_std([0.5, 0.4, 0.3], [0.25, 0.2, 0.3]);
        let mut a = vec![0f32; 3 * dst_h * dst_w];
        let mut b = vec![0f32; 3 * dst_h * dst_w];
        resize_normalize_to_tensor_u8_to_f32(&src, src_w, src_h, &mut a, dst_w, dst_h, &params)
            .unwrap();
        resize_normalize_to_tensor_u8_to_f32_bilinear(
            &src, src_w, src_h, &mut b, dst_w, dst_h, &params,
        )
        .unwrap();
        assert_eq!(a, b);
    }
}
