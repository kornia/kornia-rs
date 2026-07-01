//! Architecture-dispatched SIMD kernels for the detection pipeline.
//!
//! Each kernel has a portable scalar reference ([`scalar`]) plus NEON ([`neon`],
//! aarch64) and AVX2 ([`avx2`], x86_64) variants. The dispatchers below pick one
//! per call: NEON on aarch64 (baseline), AVX2 on x86_64 when the (cached) runtime
//! probe confirms it, scalar otherwise. Feature detection delegates to
//! `kornia_imgproc::simd::cpu_features()`, which caches the `cpuid` result.

pub(crate) mod scalar;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;

use crate::utils::Pixel;

/// Returns `true` if the running CPU supports AVX2 (always `false` off x86_64).
#[inline]
pub fn has_avx2() -> bool {
    kornia_imgproc::simd::cpu_features().has_avx2
}

/// Returns `true` if the running CPU supports both AVX2 and FMA.
#[inline]
pub fn has_avx2_fma() -> bool {
    let cpu = kornia_imgproc::simd::cpu_features();
    cpu.has_avx2 && cpu.has_fma
}

/// Classify a contiguous row: `pixel > thresh` → White, else Black.
#[inline]
pub(crate) fn classify_row(src: &[u8], dst: &mut [Pixel], thresh: u8) {
    // AArch64: NEON is baseline.
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 always has NEON; Pixel is #[repr(u8)].
    return unsafe { neon::classify_row(src, dst, thresh) };

    // x86_64: AVX2 when the runtime probe confirms it.
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 confirmed by runtime probe; Pixel is #[repr(u8)].
        return unsafe { avx2::classify_row(src, dst, thresh) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    scalar::classify_row(src, dst, thresh);
}

/// Fill per-tile min/max for every full tile (row-major).
///
/// # Safety
/// `tiles_y * tile_size * img_width + tiles_x * tile_size ≤ img_data.len()`.
#[inline]
pub(crate) fn fill_tile_stats(
    img_data: &[u8],
    img_width: usize,
    tile_size: usize,
    tiles_x: usize,
    tiles_y: usize,
    tile_min: &mut [u8],
    tile_max: &mut [u8],
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: bounds guaranteed by the caller (floor-division tile counts).
    return unsafe {
        neon::fill_tile_stats(img_data, img_width, tile_size, tiles_x, tiles_y, tile_min, tile_max)
    };

    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 confirmed by runtime probe; bounds as above.
        return unsafe {
            avx2::fill_tile_stats(
                img_data, img_width, tile_size, tiles_x, tiles_y, tile_min, tile_max,
            )
        };
    }

    #[cfg(not(target_arch = "aarch64"))]
    scalar::fill_tile_stats(img_data, img_width, tile_size, tiles_x, tiles_y, tile_min, tile_max);
}

/// Interior Gaussian smooth of `errors[half..len-half]` into `out`.
#[inline]
pub(crate) fn smooth_interior(errors: &[f32], kernel: &[f32], out: &mut [f32], half: usize, len: usize) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON mandatory on ARMv8-A.
    return unsafe { neon::smooth_interior(errors, kernel, out, half, len) };

    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2+FMA confirmed by the (cached) runtime probe.
        return unsafe { avx2::smooth_interior(errors, kernel, out, half, len) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    scalar::smooth_interior(errors, kernel, out, half, len);
}
