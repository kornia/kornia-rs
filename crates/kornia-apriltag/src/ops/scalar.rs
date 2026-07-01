//! Portable scalar implementations of the dispatched kernels.
//!
//! The arch-specific variants ([`super::neon`], [`super::avx2`]) fall back to
//! [`tile_min_max`] for their scalar tails, so it stays un-gated; the full-scalar
//! kernels below are only compiled where no SIMD path applies.

// `Pixel` is only referenced by the full-scalar `classify_row` (compiled off aarch64).
#[cfg(not(target_arch = "aarch64"))]
use crate::utils::Pixel;

/// Scalar min/max over one `tile_size`×`tile_size` tile at column `tile_x`, row `tile_y`.
#[inline]
pub(crate) fn tile_min_max(
    img_data: &[u8],
    img_width: usize,
    tile_size: usize,
    tile_x: usize,
    tile_y: usize,
) -> (u8, u8) {
    let mut lo = 255u8;
    let mut hi = 0u8;
    for row in 0..tile_size {
        let row_start = (tile_y * tile_size + row) * img_width + tile_x * tile_size;
        for &px in &img_data[row_start..row_start + tile_size] {
            lo = lo.min(px);
            hi = hi.max(px);
        }
    }
    (lo, hi)
}

/// Classify a contiguous row: `pixel > thresh` → White, else Black.
#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn classify_row(src: &[u8], dst: &mut [Pixel], thresh: u8) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = if *s > thresh {
            Pixel::White
        } else {
            Pixel::Black
        };
    }
}

/// Fill per-tile min/max for every full tile in row-major order.
#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn fill_tile_stats(
    img_data: &[u8],
    img_width: usize,
    tile_size: usize,
    tiles_x: usize,
    tiles_y: usize,
    tile_min: &mut [u8],
    tile_max: &mut [u8],
) {
    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            let idx = tile_y * tiles_x + tile_x;
            let (lo, hi) = tile_min_max(img_data, img_width, tile_size, tile_x, tile_y);
            tile_min[idx] = lo;
            tile_max[idx] = hi;
        }
    }
}

/// Interior Gaussian smooth of `errors[half..len-half]` into `out`.
#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn smooth_interior(
    errors: &[f32],
    kernel: &[f32],
    out: &mut [f32],
    half: usize,
    len: usize,
) {
    for iy in half..len - half {
        let mut acc = 0.0f32;
        for (ki, &kv) in kernel.iter().enumerate() {
            acc += errors[iy + ki - half] * kv;
        }
        out[iy] = acc;
    }
}
