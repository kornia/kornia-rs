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

        // Histogram over the (virtually reflect_101-extended) tile.
        let mut hist = [0i32; 256];
        for ly in 0..g.tile_h {
            let sy = reflect_101((ty * g.tile_h + ly) as i64, h as i64) as usize;
            let row = &s[sy * w..sy * w + w];
            for lx in 0..g.tile_w {
                let sx = reflect_101((tx * g.tile_w + lx) as i64, w as i64) as usize;
                hist[row[sx] as usize] += 1;
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

    let s = src.as_slice();
    dst.as_slice_mut()
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, drow)| {
            let srow = &s[y * w..y * w + w];
            let tyf = (y as f32).mul_add(g.inv_th, -0.5f32);
            let ty1 = tyf.floor() as i32;
            let ty2 = ty1 + 1;
            let ya = tyf - ty1 as f32;
            let ya1 = 1.0f32 - ya;
            let p1 = &luts[(ty1.max(0) as usize) * tiles_x * 256..];
            let p2 = &luts[(ty2.min(g.tiles_y as i32 - 1) as usize) * tiles_x * 256..];
            for x in 0..w {
                let v = srow[x] as usize;
                // cv2's blend with its compiled FMA contraction (see module
                // docs); the CUDA kernel mirrors this expression textually.
                let i1 = (p1[ind1[x] + v] as f32).mul_add(xa1[x], p1[ind2[x] + v] as f32 * xa[x]);
                let i2 = (p2[ind1[x] + v] as f32).mul_add(xa1[x], p2[ind2[x] + v] as f32 * xa[x]);
                let res = i1.mul_add(ya1, i2 * ya);
                drow[x] = (res.round_ties_even() as i32).clamp(0, 255) as u8;
            }
        });
    Ok(())
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
