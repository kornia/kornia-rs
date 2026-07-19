//! Bilateral filter — byte-for-byte with `cv2.bilateralFilter` for u8
//! single-channel images.
//!
//! OpenCV's exact pipeline is mirrored end to end:
//!
//! * `radius = d/2` (or `round(1.5·sigma_space)` when `d <= 0`), floored at
//!   1; circular tap mask `sqrt(i²+j²) <= radius`, taps ordered row-major;
//!   `BORDER_REFLECT_101` borders; `sigma <= 1e-6` returns the input.
//! * the color table is built the way cv2 actually builds it: entries
//!   `0..256-nlanes` with OpenCV's OWN vectorized exp polynomial
//!   ([`v_exp_f32`] below — NOT libm `expf`), the last `nlanes = 4`
//!   entries (aarch64 NEON lanes) with scalar `expf` — reproducing the
//!   split exactly;
//! * per-pixel accumulation `w = space_w[k] · color_w[|val − val0|]`,
//!   `wsum += w`, `sum = fma(val, w, sum)` (cv2's `v_muladd` / contracted
//!   scalar), output `rint(sum / wsum)`.
//!
//! The CUDA kernel (`cuda/bilateral.rs`) receives the SAME host-built
//! tables and mirrors the accumulation loop textually (`fmaf`, IEEE
//! division, `rintf`), so device output is byte-identical to the CPU's.
//!
//! VPI's `BilateralFilter` uses a different formula (measured maxdiff ~32
//! vs cv2 on random inputs) — byte parity against cv2 and VPI
//! simultaneously is impossible; cv2 is the reference here, matching the
//! rest of the library.

use kornia_image::{Image, ImageError};
use rayon::prelude::*;

/// OpenCV's `v_exp_default_32f` polynomial (core/hal/intrin_math.hpp),
/// transcribed operation-for-operation with `f32::mul_add` standing in for
/// `v_fma`. This is what cv2 uses to fill the bilateral color table (all
/// entries except the last vector-width's worth) — its results differ from
/// libm `expf` by up to a few ULP, and those ULPs are visible in output
/// bytes, so parity REQUIRES this exact polynomial.
#[inline]
// The literals are transcribed from cv2 verbatim — clippy's rounding
// suggestions would change the f32 bit patterns parity depends on.
#[allow(clippy::excessive_precision)]
pub(crate) fn v_exp_f32(x: f32) -> f32 {
    const LO: f32 = -88.376_26_f32;
    const HI: f32 = 89.0;
    const LOG2EF: f32 = std::f32::consts::LOG2_E;
    const C1: f32 = -6.933_593_8E-1;
    const C2: f32 = 2.121_944_4E-4;
    const P0: f32 = 1.987_569_2E-4;
    const P1: f32 = 1.398_199_9E-3;
    const P2: f32 = 8.333_452E-3;
    const P3: f32 = 4.166_579_6E-2;
    const P4: f32 = 1.666_666_5E-1;
    const P5: f32 = 5.000_000_2E-1;

    let x = x.clamp(LO, HI);
    let t = x.mul_add(LOG2EF, 0.5);
    let mm = t.floor();
    let mi = mm as i32;
    let scale = f32::from_bits(((mi + 0x7f) << 23) as u32);

    let x = mm.mul_add(C1, x);
    let x = mm.mul_add(C2, x);
    let xx = x * x;

    let y = x.mul_add(P0, P1);
    let y = y.mul_add(x, P2);
    let y = y.mul_add(x, P3);
    let y = y.mul_add(x, P4);
    let y = y.mul_add(x, P5);

    let y = y.mul_add(xx, x);
    let y = y + 1.0;
    y * scale
}

/// Everything the two implementations (CPU + CUDA launcher) share: the
/// tables cv2 would have built and the tap offsets. Single-source so the
/// sides can never disagree.
pub struct BilateralTables {
    /// Window radius (`d/2`, or cv2's sigma-derived rule for `d <= 0`).
    pub radius: i32,
    /// Circular-mask taps as (dy, dx), row-major — cv2's `space_ofs` order.
    pub taps: Vec<(i32, i32)>,
    /// Per-tap spatial weight (f32, `exp` in f64 like cv2's).
    pub space_weight: Vec<f32>,
    /// 256-entry color weight table (cv2's v_exp/expf split).
    pub color_weight: Vec<f32>,
    /// Tap ACCUMULATION order inside cv2's SIMD region. cv2's unrolled
    /// `maxk == 13` (d = 5) block sums lines 1,5,2,4,3 — permutation
    /// `[0,12,1,2,3,9,10,11,4,5,6,7,8]` — while its scalar tail (the last
    /// `width mod 16` pixels of each row) sums sequentially. f32 addition
    /// is order-sensitive, so byte parity must reproduce BOTH orders and
    /// the position split. Identity for every other tap count (the
    /// `maxk == 5` unroll happens to be sequential).
    pub simd_order: Vec<usize>,
}

/// First x NOT covered by cv2's 16-pixel SIMD loop (`for j <= width-16;
/// j += 16`): pixels below this use [`BilateralTables::simd_order`],
/// pixels at/above it accumulate sequentially.
#[inline]
pub(crate) fn simd_region_end(width: usize) -> usize {
    if width >= 16 {
        ((width - 16) / 16) * 16 + 16
    } else {
        0
    }
}

/// Build cv2-identical bilateral tables (see module docs).
pub fn build_tables(d: i32, sigma_color: f64, sigma_space: f64) -> BilateralTables {
    let gauss_color_coeff = (-0.5 / (sigma_color * sigma_color)) as f32;
    let gauss_space_coeff = (-0.5 / (sigma_space * sigma_space)) as f32;

    let radius = if d <= 0 {
        (sigma_space * 1.5).round_ties_even() as i32
    } else {
        d / 2
    }
    .max(1);

    // Color table: cv2 fills 0..(256 - nlanes) with its SIMD exp polynomial
    // and the tail with scalar expf; NEON nlanes = 4.
    const NLANES: usize = 4;
    let mut color_weight = vec![0f32; 256];
    let mut i = 0;
    while i < 256 - NLANES {
        for k in 0..NLANES {
            let fi = (i + k) as f32;
            color_weight[i + k] = v_exp_f32(fi * fi * gauss_color_coeff);
        }
        i += NLANES;
    }
    for (j, cw) in color_weight.iter_mut().enumerate().skip(i) {
        *cw = (((j * j) as f32) * gauss_color_coeff).exp();
    }

    // Space taps: row-major, circular mask, center INCLUDED (the 8u path
    // keeps it, unlike cv2's 32f path).
    let mut taps = Vec::new();
    let mut space_weight = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let r = ((dy * dy + dx * dx) as f64).sqrt();
            if r > radius as f64 {
                continue;
            }
            space_weight.push(((r * r) * gauss_space_coeff as f64).exp() as f32);
            taps.push((dy, dx));
        }
    }

    let simd_order: Vec<usize> = if taps.len() == 13 {
        vec![0, 12, 1, 2, 3, 9, 10, 11, 4, 5, 6, 7, 8]
    } else {
        (0..taps.len()).collect()
    };

    BilateralTables {
        radius,
        taps,
        space_weight,
        color_weight,
        simd_order,
    }
}

#[inline]
fn reflect_101(p: i64, len: i64) -> i64 {
    if len == 1 {
        return 0;
    }
    let mut p = p;
    while p < 0 || p >= len {
        if p < 0 {
            p = -p;
        } else {
            p = 2 * (len - 1) - p;
        }
    }
    p
}

/// Bilateral filter for 8-bit single-channel images — byte-for-byte with
/// `cv2.bilateralFilter(src, d, sigma_color, sigma_space)` (default
/// reflect_101 border). `sigma <= 1e-6` copies the input through,
/// mirroring cv2. Device pairs run the CUDA kernel with the same
/// host-built tables — byte-identical to the CPU path.
pub fn bilateral_filter(
    src: &Image<u8, 1>,
    dst: &mut Image<u8, 1>,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }
    const EPS: f64 = 1e-6;
    if sigma_color <= EPS || sigma_space <= EPS {
        // cv2: degenerate sigmas copy the source through.
        #[cfg(feature = "cuda")]
        {
            use crate::try_device;
            try_device!(src, dst, |stream| cuda_adapters::copy_cuda(
                src, dst, stream
            ));
        }
        dst.as_slice_mut().copy_from_slice(src.as_slice());
        return Ok(());
    }

    let t = build_tables(d, sigma_color, sigma_space);

    #[cfg(feature = "cuda")]
    {
        use crate::try_device;
        try_device!(src, dst, |stream| cuda_adapters::bilateral_cuda(
            src, dst, &t, stream
        ));
    }

    let (w, h) = (src.cols(), src.rows());
    let s = src.as_slice();

    dst.as_slice_mut()
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, drow)| {
            let simd_end = simd_region_end(w);
            for (x, dpx) in drow.iter_mut().enumerate() {
                let val0 = s[y * w + x] as i32;
                let mut wsum = 0f32;
                let mut sum = 0f32;
                let in_simd = x < simd_end;
                for kk in 0..t.taps.len() {
                    // cv2's SIMD region accumulates in simd_order; its
                    // scalar tail accumulates sequentially (see
                    // BilateralTables::simd_order).
                    let k = if in_simd { t.simd_order[kk] } else { kk };
                    let (dy, dx) = t.taps[k];
                    let sy = reflect_101(y as i64 + dy as i64, h as i64) as usize;
                    let sx = reflect_101(x as i64 + dx as i64, w as i64) as usize;
                    let val = s[sy * w + sx] as i32;
                    // cv2's inner expression: w = space·color, wsum += w,
                    // sum = fma(val, w, sum). The CUDA kernel mirrors this.
                    let wgt =
                        t.space_weight[k] * t.color_weight[(val - val0).unsigned_abs() as usize];
                    wsum += wgt;
                    sum = (val as f32).mul_add(wgt, sum);
                }
                *dpx = (sum / wsum).round_ties_even() as u8;
            }
        });
    Ok(())
}

#[cfg(feature = "cuda")]
mod cuda_adapters {
    use super::*;
    use crate::cuda::bilateral::launch_bilateral_u8;
    use crate::cuda::dispatch::{device_slices, untyped_device_err};
    use cudarc::driver::CudaStream;
    use std::sync::Arc;

    fn err(e: impl std::fmt::Display) -> ImageError {
        ImageError::Cuda(e.to_string())
    }

    pub(super) fn copy_cuda(
        src: &Image<u8, 1>,
        dst: &mut Image<u8, 1>,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let (s, d) = device_slices!(src, dst);
        stream.memcpy_dtod(s, d).map_err(err)
    }

    pub(super) fn bilateral_cuda(
        src: &Image<u8, 1>,
        dst: &mut Image<u8, 1>,
        t: &BilateralTables,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let ctx = stream.context();
        let (s, d) = device_slices!(src, dst);
        launch_bilateral_u8(ctx, stream, s, d, src.cols(), src.rows(), t).map_err(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    fn sz(w: usize, h: usize) -> ImageSize {
        ImageSize {
            width: w,
            height: h,
        }
    }

    #[test]
    fn degenerate_sigma_copies_through() {
        let src = Image::<u8, 1>::new(sz(4, 3), (0u8..12).collect()).unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(sz(4, 3), 0).unwrap();
        bilateral_filter(&src, &mut dst, 5, 0.0, 50.0).unwrap();
        assert_eq!(src.as_slice(), dst.as_slice());
    }

    #[test]
    fn constant_image_unchanged() {
        let src = Image::<u8, 1>::from_size_val(sz(16, 12), 200).unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(sz(16, 12), 0).unwrap();
        bilateral_filter(&src, &mut dst, 5, 50.0, 50.0).unwrap();
        assert!(dst.as_slice().iter().all(|&v| v == 200));
    }

    #[test]
    fn radius_rule_matches_cv2() {
        // d>0: radius = d/2; d<=0: round(1.5*sigma_space), floor at 1.
        assert_eq!(build_tables(5, 50.0, 50.0).radius, 2);
        assert_eq!(build_tables(0, 50.0, 2.0).radius, 3);
        assert_eq!(build_tables(-1, 50.0, 0.1).radius, 1);
        // circular mask: d=5 -> 13 taps (not 25), d=3 -> 5 taps.
        assert_eq!(build_tables(5, 50.0, 50.0).taps.len(), 13);
        assert_eq!(build_tables(3, 50.0, 50.0).taps.len(), 5);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_u8};
    use kornia_image::ImageSize;

    fn sz(w: usize, h: usize) -> ImageSize {
        ImageSize {
            width: w,
            height: h,
        }
    }

    #[test]
    fn bilateral_device_equals_host_byte_exact() {
        let stream = default_stream();
        for (w, h) in [(64usize, 48usize), (67, 43), (9, 5)] {
            for (d, sc, ss) in [
                (5i32, 50.0, 50.0),
                (3, 25.0, 10.0),
                (9, 75.0, 75.0),
                (0, 30.0, 3.0),
            ] {
                let src = Image::<u8, 1>::new(sz(w, h), pattern_u8(w * h)).unwrap();
                let mut cpu = Image::<u8, 1>::from_size_val(sz(w, h), 0).unwrap();
                bilateral_filter(&src, &mut cpu, d, sc, ss).unwrap();

                let d_src = src.to_cuda(&stream).unwrap();
                let mut d_dst = Image::<u8, 1>::zeros_cuda(sz(w, h), &stream).unwrap();
                bilateral_filter(&d_src, &mut d_dst, d, sc, ss).unwrap();
                assert_eq!(
                    d_dst.to_host_owned().unwrap().as_slice(),
                    cpu.as_slice(),
                    "{w}x{h} d={d} sc={sc} ss={ss}"
                );
            }
        }
    }
}
