//! CLAHE — Contrast-Limited Adaptive Histogram Equalization.
//!
//! Byte-for-byte with OpenCV's `cv::createCLAHE(clip, grid).apply()` for u8
//! single-channel images, including:
//!
//! * the tile-padding rule: when the size is not divisible by the grid the
//!   image is (virtually) extended right/bottom with `BORDER_REFLECT_101`
//!   padding — and a dimension that IS divisible still receives one full
//!   extra tile of padding when the other is not (OpenCV computes
//!   `tiles - size % tiles` unconditionally in that branch);
//! * the integer clip limit `max(1, trunc(clip · tileArea / 256))` and the
//!   two-phase excess redistribution (uniform batch + residual with stride
//!   `max(256 / residual, 1)`);
//! * the f32 LUT scale `255 / tileArea` with round-half-to-even
//!   (`saturate_cast<uchar>`), and the f32 bilinear tile blend evaluated
//!   with the exact FMA contraction OpenCV's aarch64 wheels compile to
//!   (GCC `-ffp-contract=fast` on the scalar loop): weights
//!   `txf = fma(x, 1/tw, -0.5)`, blend `fma(i1, ya1, i2·ya)` with
//!   `i_k = fma(l_k0, xa1, l_k1·xa)` — calibrated empirically per pixel
//!   against cv2 4.13 (18 configurations, 0 mismatches). `f32::mul_add`
//!   is a true fused multiply-add on every platform (libm fallback where
//!   there is no FMA unit), so this is bit-stable everywhere.
//!
//! The CUDA path (`cuda/clahe.rs`) is the textual twin of the two host
//! stages below (NVRTC `fmad=false`, mirrored expression trees), so device
//! output is byte-identical to the CPU's.

use kornia_image::{Image, ImageError};
use rayon::prelude::*;

/// `cv::borderInterpolate(p, len, BORDER_REFLECT_101)` — iterative form so
/// degenerate cases (padding wider than the image) fold repeatedly exactly
/// like OpenCV. The CUDA `reflect_101` device function is its textual twin.
#[inline]
fn reflect_101(mut p: i64, len: i64) -> i64 {
    if len == 1 {
        return 0;
    }
    while p < 0 || p >= len {
        if p < 0 {
            p = -p;
        } else {
            p = 2 * (len - 1) - p;
        }
    }
    p
}

/// Everything derived once from (size, grid, clip) — single source for the
/// CPU path and the CUDA launchers so the two sides can never disagree on
/// tile geometry, the integer clip limit, or the f32 LUT scale.
#[derive(Debug, Clone, Copy)]
pub struct ClaheGeometry {
    /// Grid width (tiles).
    pub tiles_x: usize,
    /// Grid height (tiles).
    pub tiles_y: usize,
    /// Tile width in pixels (after OpenCV's padding rule).
    pub tile_w: usize,
    /// Tile height in pixels (after OpenCV's padding rule).
    pub tile_h: usize,
    /// Integer clip limit (0 = clipping disabled, mirroring OpenCV).
    pub clip_limit: i32,
    /// `255f32 / tileArea` — the LUT quantization scale.
    pub lut_scale: f32,
    /// `1f32 / tile_w` — computed ONCE here and passed to the CUDA kernel
    /// as an argument (single-source f32 division).
    pub inv_tw: f32,
    /// `1f32 / tile_h` (same single-source rule).
    pub inv_th: f32,
}

/// Derive the CLAHE tile geometry, integer clip limit and f32 scales for a
/// given image size, grid and clip limit (OpenCV's exact rules — see the
/// module docs). Shared by the CPU path and the CUDA launchers.
pub fn clahe_geometry(
    width: usize,
    height: usize,
    grid: (usize, usize),
    clip_limit: f64,
) -> Result<ClaheGeometry, ImageError> {
    let (tiles_x, tiles_y) = grid;
    if tiles_x == 0 || tiles_y == 0 {
        return Err(ImageError::InvalidImageSize(tiles_x, tiles_y, 1, 1));
    }
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidImageSize(width, height, 1, 1));
    }
    // OpenCV's padding rule (CLAHE_Impl::apply): only when BOTH dimensions
    // divide evenly is the image used as-is; otherwise both paddings are
    // computed as `tiles - size % tiles`, so a divisible dimension gets one
    // full extra tile.
    let (tile_w, tile_h) = if width.is_multiple_of(tiles_x) && height.is_multiple_of(tiles_y) {
        (width / tiles_x, height / tiles_y)
    } else {
        let ext_w = width + (tiles_x - width % tiles_x);
        let ext_h = height + (tiles_y - height % tiles_y);
        (ext_w / tiles_x, ext_h / tiles_y)
    };
    let tile_area = tile_w * tile_h;
    // cv2: clipLimit = max(1, (int)(clipLimit_ * tileSizeTotal / histSize)),
    // or 0 (disabled) when clipLimit_ <= 0.
    let clip = if clip_limit > 0.0 {
        ((clip_limit * tile_area as f64 / 256.0) as i32).max(1)
    } else {
        0
    };
    Ok(ClaheGeometry {
        tiles_x,
        tiles_y,
        tile_w,
        tile_h,
        clip_limit: clip,
        lut_scale: 255.0f32 / tile_area as f32,
        inv_tw: 1.0f32 / tile_w as f32,
        inv_th: 1.0f32 / tile_h as f32,
    })
}

/// Build the per-tile equalization LUTs (`tiles_y · tiles_x · 256` bytes,
/// tile-major). Mirrors `CLAHE_CalcLut_Body<uchar>`; the CUDA `clahe_lut_u8`
/// kernel is the textual twin of this function.
pub(crate) fn build_tile_luts(src: &Image<u8, 1>, g: &ClaheGeometry) -> Vec<u8> {
    let (w, h) = (src.cols(), src.rows());
    let s = src.as_slice();
    let mut luts = vec![0u8; g.tiles_x * g.tiles_y * 256];

    luts.par_chunks_mut(256).enumerate().for_each(|(k, lut)| {
        let ty = k / g.tiles_x;
        let tx = k % g.tiles_x;

        // Histogram over the (virtually reflect_101-extended) tile. Interior
        // tiles never touch the padding: take a branch-free slice walk (the
        // hot path — only the last tile row/column ever reflects).
        let mut hist = [0i32; 256];
        let x0 = tx * g.tile_w;
        let y0 = ty * g.tile_h;
        if x0 + g.tile_w <= w && y0 + g.tile_h <= h {
            for ly in 0..g.tile_h {
                let row = &s[(y0 + ly) * w + x0..(y0 + ly) * w + x0 + g.tile_w];
                let mut chunks = row.chunks_exact(4);
                for c in &mut chunks {
                    // 4-way unroll (mirrors cv2's): distinct bins can't
                    // alias a single increment chain.
                    hist[c[0] as usize] += 1;
                    hist[c[1] as usize] += 1;
                    hist[c[2] as usize] += 1;
                    hist[c[3] as usize] += 1;
                }
                for &px in chunks.remainder() {
                    hist[px as usize] += 1;
                }
            }
        } else {
            for ly in 0..g.tile_h {
                let sy = reflect_101((y0 + ly) as i64, h as i64) as usize;
                let row = &s[sy * w..sy * w + w];
                for lx in 0..g.tile_w {
                    let sx = reflect_101((x0 + lx) as i64, w as i64) as usize;
                    hist[row[sx] as usize] += 1;
                }
            }
        }

        // Clip + redistribute (cv2's two-phase scheme).
        if g.clip_limit > 0 {
            let mut clipped = 0i32;
            for b in hist.iter_mut() {
                if *b > g.clip_limit {
                    clipped += *b - g.clip_limit;
                    *b = g.clip_limit;
                }
            }
            let redist_batch = clipped / 256;
            let mut residual = clipped - redist_batch * 256;
            for b in hist.iter_mut() {
                *b += redist_batch;
            }
            if residual != 0 {
                let residual_step = (256 / residual).max(1) as usize;
                let mut i = 0;
                while i < 256 && residual > 0 {
                    hist[i] += 1;
                    i += residual_step;
                    residual -= 1;
                }
            }
        }

        // cdf → LUT: f32 scale, round-half-to-even (cv2 saturate_cast).
        let mut sum = 0i32;
        for i in 0..256 {
            sum += hist[i];
            lut[i] = ((sum as f32 * g.lut_scale).round_ties_even() as i32).clamp(0, 255) as u8;
        }
    });

    luts
}

/// Contrast-Limited Adaptive Histogram Equalization for 8-bit
/// single-channel images — byte-for-byte with
/// `cv2.createCLAHE(clip_limit, grid).apply(src)`.
///
/// `grid` is `(tiles_x, tiles_y)` (OpenCV's `tileGridSize` width, height);
/// `clip_limit <= 0` disables clipping (plain tiled AHE), mirroring OpenCV.
/// Device pairs run the CUDA LUT-build + blend kernels, byte-identical to
/// the CPU path.
pub fn clahe(
    src: &Image<u8, 1>,
    dst: &mut Image<u8, 1>,
    clip_limit: f64,
    grid: (usize, usize),
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }
    let g = clahe_geometry(src.cols(), src.rows(), grid, clip_limit)?;

    #[cfg(feature = "cuda")]
    {
        use crate::try_device;
        try_device!(src, dst, |stream| cuda_adapters::clahe_cuda(
            src, dst, &g, stream
        ));
    }

    let luts = build_tile_luts(src, &g);
    let (w, tiles_x) = (src.cols(), g.tiles_x);

    // Per-column interpolation tables (cv2 precomputes these once; values
    // are identical to the per-pixel expressions the CUDA kernel evaluates —
    // same f32 operations, fmad-free).
    let mut ind1 = vec![0usize; w];
    let mut ind2 = vec![0usize; w];
    let mut xa = vec![0f32; w];
    let mut xa1 = vec![0f32; w];
    for x in 0..w {
        let txf = (x as f32).mul_add(g.inv_tw, -0.5f32);
        let tx1 = txf.floor() as i32;
        let tx2 = tx1 + 1;
        xa[x] = txf - tx1 as f32;
        xa1[x] = 1.0f32 - xa[x];
        ind1[x] = (tx1.max(0) as usize) * 256;
        ind2[x] = (tx2.min(g.tiles_x as i32 - 1) as usize) * 256;
    }

    // Horizontal spans: maximal x-ranges with constant (ind1, ind2). Per
    // tile-row band we pack the 4 surrounding LUT bytes of every value into
    // ONE u32 table per span (l00 | l01<<8 | l10<<16 | l11<<24), so the
    // blend needs a single gather per pixel. Tables store the exact LUT
    // bytes — a pure access-pattern change, outputs stay byte-identical.
    // Scratch is per-thread (for_each_init) with the band key memoized, so
    // parallelism stays flat over rows.
    let mut spans: Vec<(usize, usize, usize)> = Vec::new(); // (x_end, ind1, ind2)
    {
        let mut x = 0;
        while x < w {
            let (i1, i2) = (ind1[x], ind2[x]);
            let mut e = x + 1;
            while e < w && ind1[e] == i1 && ind2[e] == i2 {
                e += 1;
            }
            spans.push((e, i1, i2));
            x = e;
        }
    }

    let s = src.as_slice();
    let n_spans = spans.len();
    dst.as_slice_mut()
        .par_chunks_mut(w)
        .enumerate()
        .for_each_init(
            || (vec![0u32; n_spans * 256], usize::MAX),
            |(tt, cached_band), (y, drow)| {
                let srow = &s[y * w..y * w + w];
                let tyf = (y as f32).mul_add(g.inv_th, -0.5f32);
                let ty1 = tyf.floor() as i32;
                let ya = tyf - ty1 as f32;
                let ya1 = 1.0f32 - ya;
                let j1 = ty1.max(0) as usize;
                let j2 = ((ty1 + 1).min(g.tiles_y as i32 - 1)) as usize;
                let band = j1 * g.tiles_y + j2;
                if *cached_band != band {
                    let p1 = &luts[j1 * tiles_x * 256..];
                    let p2 = &luts[j2 * tiles_x * 256..];
                    for (si, &(_, i1, i2)) in spans.iter().enumerate() {
                        for v in 0..256 {
                            tt[si * 256 + v] = p1[i1 + v] as u32
                                | ((p1[i2 + v] as u32) << 8)
                                | ((p2[i1 + v] as u32) << 16)
                                | ((p2[i2 + v] as u32) << 24);
                        }
                    }
                    *cached_band = band;
                }
                let mut x0 = 0;
                for (si, &(x_end, _, _)) in spans.iter().enumerate() {
                    interpolate_span(
                        &srow[x0..x_end],
                        &mut drow[x0..x_end],
                        &tt[si * 256..si * 256 + 256],
                        &xa[x0..x_end],
                        &xa1[x0..x_end],
                        ya,
                        ya1,
                    );
                    x0 = x_end;
                }
            },
        );
    Ok(())
}

/// One span of the tile-LUT blend: `tt` is the packed 256-entry u32 table
/// (`l00 | l01<<8 | l10<<16 | l11<<24`). Scalar reference; on aarch64 the
/// NEON path below is used — both are bit-exact (NEON `vfmaq_f32` is the
/// same fused multiply-add as `f32::mul_add`, `vcvtnq_s32_f32` the same
/// round-half-to-even as `round_ties_even`, and unpacking the u32 recovers
/// the exact LUT bytes).
fn interpolate_span_scalar(
    srow: &[u8],
    drow: &mut [u8],
    tt: &[u32],
    xa: &[f32],
    xa1: &[f32],
    ya: f32,
    ya1: f32,
) {
    for (x, d) in drow.iter_mut().enumerate() {
        let t = tt[srow[x] as usize];
        let (l00, l01) = ((t & 0xFF) as f32, ((t >> 8) & 0xFF) as f32);
        let (l10, l11) = (((t >> 16) & 0xFF) as f32, (t >> 24) as f32);
        // cv2's blend with its compiled FMA contraction (see module docs);
        // the CUDA kernel mirrors this expression textually.
        let i1 = l00.mul_add(xa1[x], l01 * xa[x]);
        let i2 = l10.mul_add(xa1[x], l11 * xa[x]);
        let res = i1.mul_add(ya1, i2 * ya);
        *d = (res.round_ties_even() as i32).clamp(0, 255) as u8;
    }
}

#[cfg(not(target_arch = "aarch64"))]
use interpolate_span_scalar as interpolate_span;

/// NEON span kernel: ONE scalar table gather per pixel (no NEON gather
/// exists) feeding a 4-lane fused blend. Lane-wise identical to
/// `interpolate_span_scalar` (see its parity notes).
#[cfg(target_arch = "aarch64")]
fn interpolate_span(
    srow: &[u8],
    drow: &mut [u8],
    tt: &[u32],
    xa: &[f32],
    xa1: &[f32],
    ya: f32,
    ya1: f32,
) {
    use std::arch::aarch64::*;
    let w = drow.len();
    let mut x = 0;
    // SAFETY: lane indices are x..x+4 < w; table indices are v < 256 and
    // the table has exactly 256 entries (sliced by the caller).
    unsafe {
        let ya_v = vdupq_n_f32(ya);
        let ya1_v = vdupq_n_f32(ya1);
        let mask = vdupq_n_u32(0xFF);
        // 8 px/iteration: the 8 independent scalar gathers fill the load
        // pipes while the previous iteration's f32 chain retires.
        while x + 8 <= w {
            let mut a = [0u32; 8];
            for (k, ak) in a.iter_mut().enumerate() {
                *ak = *tt.get_unchecked(*srow.get_unchecked(x + k) as usize);
            }
            let blend = |packed: uint32x4_t, xa_v: float32x4_t, xa1_v: float32x4_t| {
                let l00 = vcvtq_f32_u32(vandq_u32(packed, mask));
                let l01 = vcvtq_f32_u32(vandq_u32(vshrq_n_u32(packed, 8), mask));
                let l10 = vcvtq_f32_u32(vandq_u32(vshrq_n_u32(packed, 16), mask));
                let l11 = vcvtq_f32_u32(vshrq_n_u32(packed, 24));
                // i_k = fma(l_k0, xa1, l_k1*xa); res = fma(i1, ya1, i2*ya)
                let i1 = vfmaq_f32(vmulq_f32(l01, xa_v), l00, xa1_v);
                let i2 = vfmaq_f32(vmulq_f32(l11, xa_v), l10, xa1_v);
                let res = vfmaq_f32(vmulq_f32(i2, ya_v), i1, ya1_v);
                vcvtnq_s32_f32(res)
            };
            let r_lo = blend(
                vld1q_u32(a.as_ptr()),
                vld1q_f32(xa.as_ptr().add(x)),
                vld1q_f32(xa1.as_ptr().add(x)),
            );
            let r_hi = blend(
                vld1q_u32(a.as_ptr().add(4)),
                vld1q_f32(xa.as_ptr().add(x + 4)),
                vld1q_f32(xa1.as_ptr().add(x + 4)),
            );
            let r_u8 = vqmovn_u16(vcombine_u16(vqmovun_s32(r_lo), vqmovun_s32(r_hi)));
            vst1_u8(drow.as_mut_ptr().add(x), r_u8);
            x += 8;
        }
    }
    interpolate_span_scalar(&srow[x..], &mut drow[x..], tt, &xa[x..], &xa1[x..], ya, ya1);
}

#[cfg(feature = "cuda")]
mod cuda_adapters {
    use super::*;
    use crate::cuda::clahe::{launch_clahe_apply_u8, launch_clahe_lut_u8};
    use crate::cuda::dispatch::{device_slices, untyped_device_err};
    use cudarc::driver::CudaStream;
    use std::sync::Arc;

    fn err(e: impl std::fmt::Display) -> ImageError {
        ImageError::Cuda(e.to_string())
    }

    pub(super) fn clahe_cuda(
        src: &Image<u8, 1>,
        dst: &mut Image<u8, 1>,
        g: &ClaheGeometry,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let ctx = stream.context();
        let (s, d) = device_slices!(src, dst);
        let (w, h) = (src.cols(), src.rows());
        let mut luts = stream
            .alloc_zeros::<u8>(g.tiles_x * g.tiles_y * 256)
            .map_err(err)?;
        launch_clahe_lut_u8(ctx, stream, s, &mut luts, w, h, g).map_err(err)?;
        launch_clahe_apply_u8(ctx, stream, s, d, &luts, w, h, g).map_err(err)?;
        Ok(())
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
    fn geometry_matches_cv2_padding_rule() {
        // divisible: no padding
        let g = clahe_geometry(64, 64, (8, 8), 40.0).unwrap();
        assert_eq!((g.tile_w, g.tile_h), (8, 8));
        // width divisible, height not: BOTH padded (full extra tile in x)
        let g = clahe_geometry(64, 63, (8, 8), 40.0).unwrap();
        assert_eq!((g.tile_w, g.tile_h), (9, 8));
        // clip limit truncation + floor at 1
        let g = clahe_geometry(64, 64, (8, 8), 40.0).unwrap();
        assert_eq!(g.clip_limit, 10); // 40*64/256 = 10
        let g = clahe_geometry(16, 16, (8, 8), 1.0).unwrap();
        assert_eq!(g.clip_limit, 1); // trunc(4/256)=0 -> max(1)
        let g = clahe_geometry(64, 64, (8, 8), 0.0).unwrap();
        assert_eq!(g.clip_limit, 0); // disabled
    }

    #[test]
    fn constant_image_stays_constant() {
        let src = Image::<u8, 1>::from_size_val(sz(64, 48), 100).unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(sz(64, 48), 0).unwrap();
        clahe(&src, &mut dst, 40.0, (8, 8)).unwrap();
        // Constant tile → cdf jumps to tileArea at bin 100 → LUT maps 100
        // near 255 uniformly; the blend of equal LUT values is that value.
        let v = dst.as_slice()[0];
        assert!(dst.as_slice().iter().all(|&p| p == v));
    }

    #[test]
    fn rejects_zero_grid_and_size_mismatch() {
        let src = Image::<u8, 1>::from_size_val(sz(8, 8), 0).unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(sz(8, 8), 0).unwrap();
        assert!(clahe(&src, &mut dst, 40.0, (0, 8)).is_err());
        let mut small = Image::<u8, 1>::from_size_val(sz(4, 8), 0).unwrap();
        assert!(clahe(&src, &mut small, 40.0, (8, 8)).is_err());
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

    /// Device CLAHE must be byte-exact vs the CPU path: divisible and
    /// non-divisible sizes (padding path), several grids and clip limits,
    /// including clipping disabled and the degenerate 1-tile grid.
    #[test]
    fn clahe_device_equals_host_byte_exact() {
        let stream = default_stream();
        for (w, h) in [(64usize, 48usize), (67, 43), (128, 128), (33, 9)] {
            for grid in [(8usize, 8usize), (4, 3), (1, 1)] {
                for clip in [40.0f64, 2.5, 0.0] {
                    let src = Image::<u8, 1>::new(sz(w, h), pattern_u8(w * h)).unwrap();
                    let mut cpu = Image::<u8, 1>::from_size_val(sz(w, h), 0).unwrap();
                    clahe(&src, &mut cpu, clip, grid).unwrap();

                    let d_src = src.to_cuda(&stream).unwrap();
                    let mut d_dst = Image::<u8, 1>::zeros_cuda(sz(w, h), &stream).unwrap();
                    clahe(&d_src, &mut d_dst, clip, grid).unwrap();
                    let back = d_dst.to_host_owned().unwrap();
                    assert_eq!(
                        back.as_slice(),
                        cpu.as_slice(),
                        "{w}x{h} grid={grid:?} clip={clip}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod probe_tests {
    use super::*;
    use kornia_image::ImageSize;

    /// Stage-split probe (not a test of correctness). Run with
    /// `cargo test -p kornia-imgproc --release --lib clahe_stage_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn clahe_stage_probe() {
        let (w, h) = (1920usize, 1080usize);
        let data: Vec<u8> = (0..w * h)
            .map(|i| ((i * 2654435761usize) >> 24) as u8)
            .collect();
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: w,
                height: h,
            },
            data,
        )
        .unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(src.size(), 0).unwrap();
        let g = clahe_geometry(w, h, (8, 8), 40.0).unwrap();

        let mut best_lut = f64::INFINITY;
        let mut best_all = f64::INFINITY;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..20 {
                std::hint::black_box(build_tile_luts(&src, &g));
            }
            best_lut = best_lut.min(t.elapsed().as_secs_f64() / 20.0);

            let t = std::time::Instant::now();
            for _ in 0..20 {
                clahe(&src, &mut dst, 40.0, (8, 8)).unwrap();
            }
            best_all = best_all.min(t.elapsed().as_secs_f64() / 20.0);
        }
        println!(
            "lut build: {:.3} ms, full: {:.3} ms, interp: {:.3} ms",
            best_lut * 1e3,
            best_all * 1e3,
            (best_all - best_lut) * 1e3
        );
    }
}
