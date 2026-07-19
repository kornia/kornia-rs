//! Median blur — byte-for-byte with `cv2.medianBlur` AND VPI's
//! `MedianFilter` (verified empirically: the two agree bit-for-bit for 3×3
//! and 5×5 u8 with replicate borders), so one implementation matches both.
//!
//! The median is an exact order statistic — no rounding, no float — so
//! parity only requires the same border rule (`BORDER_REPLICATE`, which is
//! what `cv2.medianBlur` hardwires) and an exact median. Sorting networks
//! (min/max only) give that on CPU and GPU alike; the CUDA kernels in
//! `cuda/median.rs` are textual twins of the networks below.

use kornia_image::{Image, ImageError};
use rayon::prelude::*;

/// Compare-exchange: after the call `*a <= *b`.
#[inline(always)]
fn ce(v: &mut [u8], a: usize, b: usize) {
    let (x, y) = (v[a], v[b]);
    v[a] = x.min(y);
    v[b] = x.max(y);
}

/// Paeth's 19-exchange median-of-9 network. The CUDA kernel source is
/// GENERATED from this same list (`cuda/median.rs`), so the sides cannot
/// drift.
pub(crate) const NET9: [(usize, usize); 19] = [
    (1, 2),
    (4, 5),
    (7, 8),
    (0, 1),
    (3, 4),
    (6, 7),
    (1, 2),
    (4, 5),
    (7, 8),
    (0, 3),
    (5, 8),
    (4, 7),
    (3, 6),
    (1, 4),
    (2, 5),
    (4, 7),
    (4, 2),
    (6, 4),
    (4, 2),
];

/// Exact median of 9 via [`NET9`].
#[inline(always)]
pub(crate) fn median9(v: &mut [u8; 9]) -> u8 {
    for &(a, b) in NET9.iter() {
        ce(v, a, b);
    }
    v[4]
}

/// Smith's classic 99-exchange median-of-25 network — also the generator
/// for the CUDA 5×5 kernel.
pub(crate) const NET25: [(usize, usize); 99] = [
    (0, 1),
    (3, 4),
    (2, 4),
    (2, 3),
    (6, 7),
    (5, 7),
    (5, 6),
    (9, 10),
    (8, 10),
    (8, 9),
    (12, 13),
    (11, 13),
    (11, 12),
    (15, 16),
    (14, 16),
    (14, 15),
    (18, 19),
    (17, 19),
    (17, 18),
    (21, 22),
    (20, 22),
    (20, 21),
    (23, 24),
    (2, 5),
    (3, 6),
    (0, 6),
    (0, 3),
    (4, 7),
    (1, 7),
    (1, 4),
    (11, 14),
    (8, 14),
    (8, 11),
    (12, 15),
    (9, 15),
    (9, 12),
    (13, 16),
    (10, 16),
    (10, 13),
    (20, 23),
    (17, 23),
    (17, 20),
    (21, 24),
    (18, 24),
    (18, 21),
    (19, 22),
    (8, 17),
    (9, 18),
    (0, 18),
    (0, 9),
    (10, 19),
    (1, 19),
    (1, 10),
    (11, 20),
    (2, 20),
    (2, 11),
    (12, 21),
    (3, 21),
    (3, 12),
    (13, 22),
    (4, 22),
    (4, 13),
    (14, 23),
    (5, 23),
    (5, 14),
    (15, 24),
    (6, 24),
    (6, 15),
    (7, 16),
    (7, 19),
    (13, 21),
    (15, 23),
    (7, 13),
    (7, 15),
    (1, 9),
    (3, 11),
    (5, 17),
    (11, 17),
    (9, 17),
    (4, 10),
    (6, 12),
    (7, 14),
    (4, 6),
    (4, 7),
    (12, 14),
    (10, 14),
    (6, 7),
    (10, 12),
    (6, 10),
    (6, 17),
    (12, 17),
    (7, 17),
    (7, 10),
    (12, 18),
    (7, 12),
    (10, 18),
    (12, 20),
    (10, 20),
    (10, 12),
];

/// Exact median of 25 via [`NET25`].
#[inline(always)]
pub(crate) fn median25(v: &mut [u8; 25]) -> u8 {
    for &(a, b) in NET25.iter() {
        ce(v, a, b);
    }
    v[12]
}

/// Median blur for 8-bit images — byte-for-byte with
/// `cv2.medianBlur(src, ksize)` and VPI's CUDA `MedianFilter` (the two are
/// themselves bit-identical for these kernel sizes). `ksize` must be 3
/// or 5; borders replicate. Device pairs run the CUDA sorting-network
/// kernels, byte-identical to the CPU path.
pub fn median_blur<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    ksize: usize,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }
    if ksize != 3 && ksize != 5 {
        return Err(ImageError::InvalidKernelLength(ksize, ksize));
    }

    #[cfg(feature = "cuda")]
    {
        use crate::try_device;
        try_device!(src, dst, |stream| cuda_adapters::median_blur_cuda(
            src, dst, ksize, stream
        ));
    }

    let (w, h) = (src.cols(), src.rows());
    let s = src.as_slice();
    let r = (ksize / 2) as i64;

    dst.as_slice_mut()
        .par_chunks_mut(w * C)
        .enumerate()
        .for_each(|(y, drow)| {
            // NEON C1 fast path: run the same network on 16 lanes at once
            // (u8 min/max are exact — byte-identical to the scalar walk).
            #[cfg(target_arch = "aarch64")]
            if C == 1 {
                let done = if ksize == 3 {
                    neon::median_row::<9>(s, drow, w, h, y, 1, &NET9, 4)
                } else {
                    neon::median_row::<25>(s, drow, w, h, y, 2, &NET25, 12)
                };
                if done {
                    return;
                }
            }
            // Replicate-clamped source row pointers for the window.
            for x in 0..w {
                for c in 0..C {
                    if ksize == 3 {
                        let mut win = [0u8; 9];
                        let mut n = 0;
                        for dy in -1i64..=1 {
                            let sy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                            for dx in -1i64..=1 {
                                let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                                win[n] = s[(sy * w + sx) * C + c];
                                n += 1;
                            }
                        }
                        drow[x * C + c] = median9(&mut win);
                    } else {
                        let mut win = [0u8; 25];
                        let mut n = 0;
                        for dy in -r..=r {
                            let sy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                            for dx in -r..=r {
                                let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                                win[n] = s[(sy * w + sx) * C + c];
                                n += 1;
                            }
                        }
                        drow[x * C + c] = median25(&mut win);
                    }
                }
            }
        });
    Ok(())
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// One output row, C1: interior columns 16 at a time through the
    /// sorting network in NEON lanes; border columns (replicate clamp)
    /// and rows shorter than a vector fall back to the caller's scalar
    /// walk (return false). `TAPS` = ksize². Lane-wise min/max is the
    /// same exact network as the scalar path — byte-identical.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn median_row<const TAPS: usize>(
        s: &[u8],
        drow: &mut [u8],
        w: usize,
        h: usize,
        y: usize,
        r: usize,
        net: &[(usize, usize)],
        center: usize,
    ) -> bool {
        if w < 16 + 2 * r {
            return false;
        }
        let k = 2 * r + 1;
        // Interior x range where no horizontal clamping is needed.
        let x0 = r;
        let x1 = w - r;
        // SAFETY: all loads read x-r..x+r+16-1+r within [0, w) for
        // x in [x0, x1-16]; row indices are replicate-clamped.
        unsafe {
            let mut rows: [*const u8; 25] = [std::ptr::null(); 25];
            for (dy, row) in rows.iter_mut().enumerate().take(k) {
                let sy = (y + dy).saturating_sub(r).min(h - 1);
                *row = s.as_ptr().add(sy * w);
            }
            let mut x = x0;
            while x + 16 <= x1 {
                let mut v: [uint8x16_t; TAPS] = [vdupq_n_u8(0); TAPS];
                let mut n = 0;
                for row in rows.iter().take(k) {
                    for dx in 0..k {
                        v[n] = vld1q_u8(row.add(x + dx - r));
                        n += 1;
                    }
                }
                for &(a, b) in net.iter() {
                    let lo = vminq_u8(v[a], v[b]);
                    let hi = vmaxq_u8(v[a], v[b]);
                    v[a] = lo;
                    v[b] = hi;
                }
                vst1q_u8(drow.as_mut_ptr().add(x), v[center]);
                x += 16;
            }
            // Scalar remainder: interior tail + both borders.
            let scalar = |x: usize, drow: &mut [u8]| {
                let mut win = [0u8; TAPS];
                let mut n = 0;
                for dy in 0..k {
                    let sy = (y + dy).saturating_sub(r).min(h - 1);
                    for dx in 0..k {
                        let sx = (x + dx).saturating_sub(r).min(w - 1);
                        win[n] = s[sy * w + sx];
                        n += 1;
                    }
                }
                for &(a, b) in net.iter() {
                    let (lo, hi) = (win[a].min(win[b]), win[a].max(win[b]));
                    win[a] = lo;
                    win[b] = hi;
                }
                drow[x] = win[center];
            };
            for xx in 0..x0 {
                scalar(xx, drow);
            }
            for xx in x..w {
                scalar(xx, drow);
            }
        }
        true
    }
}

#[cfg(feature = "cuda")]
mod cuda_adapters {
    use super::*;
    use crate::cuda::dispatch::{device_slices, untyped_device_err};
    use crate::cuda::median::launch_median_u8;
    use cudarc::driver::CudaStream;
    use std::sync::Arc;

    fn err(e: impl std::fmt::Display) -> ImageError {
        ImageError::Cuda(e.to_string())
    }

    pub(super) fn median_blur_cuda<const C: usize>(
        src: &Image<u8, C>,
        dst: &mut Image<u8, C>,
        ksize: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let ctx = stream.context();
        let (s, d) = device_slices!(src, dst);
        launch_median_u8(ctx, stream, s, d, src.cols(), src.rows(), C, ksize).map_err(err)
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

    fn median_naive(win: &mut [u8]) -> u8 {
        win.sort_unstable();
        win[win.len() / 2]
    }

    #[test]
    fn networks_match_naive_median() {
        // Exhaustive-ish pseudo-random check of both networks.
        let mut state = 0x12345678u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u8
        };
        for _ in 0..2000 {
            let mut w9 = [0u8; 9];
            for v in w9.iter_mut() {
                *v = next();
            }
            let mut sorted = w9.to_vec();
            assert_eq!(median9(&mut w9.clone()), median_naive(&mut sorted));

            let mut w25 = [0u8; 25];
            for v in w25.iter_mut() {
                *v = next();
            }
            let mut sorted = w25.to_vec();
            assert_eq!(median25(&mut w25.clone()), median_naive(&mut sorted));
        }
    }

    #[test]
    fn rejects_bad_ksize_and_size_mismatch() {
        let src = Image::<u8, 1>::from_size_val(sz(8, 8), 0).unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(sz(8, 8), 0).unwrap();
        assert!(median_blur(&src, &mut dst, 4).is_err());
        assert!(median_blur(&src, &mut dst, 7).is_err());
        let mut small = Image::<u8, 1>::from_size_val(sz(4, 8), 0).unwrap();
        assert!(median_blur(&src, &mut small, 3).is_err());
    }

    #[test]
    fn constant_image_unchanged() {
        let src = Image::<u8, 3>::from_size_val(sz(16, 12), 99).unwrap();
        let mut dst = Image::<u8, 3>::from_size_val(sz(16, 12), 0).unwrap();
        median_blur(&src, &mut dst, 3).unwrap();
        assert!(dst.as_slice().iter().all(|&v| v == 99));
        median_blur(&src, &mut dst, 5).unwrap();
        assert!(dst.as_slice().iter().all(|&v| v == 99));
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
    fn median_device_equals_host_byte_exact() {
        let stream = default_stream();
        for (w, h) in [(64usize, 48usize), (67, 43), (5, 4), (1, 1), (2, 9)] {
            for ksize in [3usize, 5] {
                let src = Image::<u8, 1>::new(sz(w, h), pattern_u8(w * h)).unwrap();
                let mut cpu = Image::<u8, 1>::from_size_val(sz(w, h), 0).unwrap();
                median_blur(&src, &mut cpu, ksize).unwrap();

                let d_src = src.to_cuda(&stream).unwrap();
                let mut d_dst = Image::<u8, 1>::zeros_cuda(sz(w, h), &stream).unwrap();
                median_blur(&d_src, &mut d_dst, ksize).unwrap();
                assert_eq!(
                    d_dst.to_host_owned().unwrap().as_slice(),
                    cpu.as_slice(),
                    "{w}x{h} k={ksize}"
                );
            }
        }
        // C3
        let (w, h) = (33usize, 21usize);
        for ksize in [3usize, 5] {
            let src = Image::<u8, 3>::new(sz(w, h), pattern_u8(w * h * 3)).unwrap();
            let mut cpu = Image::<u8, 3>::from_size_val(sz(w, h), 0).unwrap();
            median_blur(&src, &mut cpu, ksize).unwrap();
            let d_src = src.to_cuda(&stream).unwrap();
            let mut d_dst = Image::<u8, 3>::zeros_cuda(sz(w, h), &stream).unwrap();
            median_blur(&d_src, &mut d_dst, ksize).unwrap();
            assert_eq!(
                d_dst.to_host_owned().unwrap().as_slice(),
                cpu.as_slice(),
                "C3 {w}x{h} k={ksize}"
            );
        }
    }
}
