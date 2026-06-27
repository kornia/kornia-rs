//! NEON / AVX2 kernels for the bandwidth-bound channel & alpha conversions.
//!
//! These conversions do no arithmetic on the pixel values — they only reorder
//! channels (`bgr_from_rgb`), broadcast (`rgb_from_gray`), or add/strip an alpha
//! lane (`rgba_from_rgb`, `bgra_from_rgb`). They are memory-bandwidth-bound, so
//! the goal is a single clean streaming pass with structured de/interleave loads
//! and stores (`vld3q`/`vld4q`/`vst3q`/`vst4q`) rather than any clever math.
//!
//! Each kernel keeps a scalar tail and a portable scalar fallback that doubles as
//! the correctness oracle on non-NEON targets. AVX2 has no 3/4-way byte
//! de/interleave, so the x86_64 paths are scaffolds that defer to scalar for now.

use super::super::kernel_common::{par_strip_dispatch, par_strip_dispatch_nm};

// ===== BGR <-> RGB (3->3 channel reverse) ==========================================

/// RGB8 <-> BGR8: swap the R and B channels (symmetric, used both directions).
///
/// 16-pixel SIMD alignment keeps the `vld3q_u8`/`vst3q_u8` bulk loop intact at
/// rayon strip boundaries.
pub fn bgr_from_rgb_u8(src: &[u8], dst: &mut [u8], npixels: usize) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 3);
    par_strip_dispatch(src, dst, npixels, 3, 16, bgr_from_rgb_u8_kernel);
}

#[inline]
fn bgr_from_rgb_u8_kernel(src: &[u8], dst: &mut [u8], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        bgr_from_rgb_u8_neon(src, dst, npixels);
        return;
    }
    #[allow(unreachable_code)]
    bgr_from_rgb_u8_scalar(src, dst, npixels);
}

/// NEON RGB8 -> BGR8: 16 pixels per iteration. `vld3q_u8` deinterleaves into
/// R/G/B lanes; we store back with `.0` and `.2` swapped via `vst3q_u8`.
#[cfg(target_arch = "aarch64")]
fn bgr_from_rgb_u8_neon(src: &[u8], dst: &mut [u8], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let bulk = npixels & !15;
        let mut i = 0usize;
        while i < bulk {
            let v = vld3q_u8(sp.add(i * 3));
            vst3q_u8(dp.add(i * 3), uint8x16x3_t(v.2, v.1, v.0));
            i += 16;
        }
        while i < npixels {
            let s = i * 3;
            *dp.add(s) = *sp.add(s + 2);
            *dp.add(s + 1) = *sp.add(s + 1);
            *dp.add(s + 2) = *sp.add(s);
            i += 1;
        }
    }
}

/// Portable scalar fallback / oracle for RGB8 <-> BGR8.
#[allow(dead_code)]
fn bgr_from_rgb_u8_scalar(src: &[u8], dst: &mut [u8], npixels: usize) {
    for i in 0..npixels {
        let s = i * 3;
        dst[s] = src[s + 2];
        dst[s + 1] = src[s + 1];
        dst[s + 2] = src[s];
    }
}

/// RGB f32 <-> BGR f32: swap the R and B channels.
pub fn bgr_from_rgb_f32(src: &[f32], dst: &mut [f32], npixels: usize) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 3);
    par_strip_dispatch(src, dst, npixels, 3, 4, bgr_from_rgb_f32_kernel);
}

#[inline]
fn bgr_from_rgb_f32_kernel(src: &[f32], dst: &mut [f32], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        bgr_from_rgb_f32_neon(src, dst, npixels);
        return;
    }
    #[allow(unreachable_code)]
    bgr_from_rgb_f32_scalar(src, dst, npixels);
}

/// NEON RGB f32 -> BGR f32: 4 pixels per iteration (`vld3q_f32`/`vst3q_f32`).
#[cfg(target_arch = "aarch64")]
fn bgr_from_rgb_f32_neon(src: &[f32], dst: &mut [f32], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let bulk = npixels & !3;
        let mut i = 0usize;
        while i < bulk {
            let v = vld3q_f32(sp.add(i * 3));
            vst3q_f32(dp.add(i * 3), float32x4x3_t(v.2, v.1, v.0));
            i += 4;
        }
        while i < npixels {
            let s = i * 3;
            *dp.add(s) = *sp.add(s + 2);
            *dp.add(s + 1) = *sp.add(s + 1);
            *dp.add(s + 2) = *sp.add(s);
            i += 1;
        }
    }
}

/// Portable scalar fallback / oracle for RGB f32 <-> BGR f32.
#[allow(dead_code)]
fn bgr_from_rgb_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize) {
    for i in 0..npixels {
        let s = i * 3;
        dst[s] = src[s + 2];
        dst[s + 1] = src[s + 1];
        dst[s + 2] = src[s];
    }
}

// ===== RGB -> RGBA (3->4 add opaque alpha) =========================================

/// RGB8 -> RGBA8: copy R/G/B and append an opaque alpha (255).
pub fn rgba_from_rgb_u8(src: &[u8], dst: &mut [u8], npixels: usize) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 4);
    par_strip_dispatch_nm(src, dst, npixels, 3, 4, 16, rgba_from_rgb_u8_kernel);
}

#[inline]
fn rgba_from_rgb_u8_kernel(src: &[u8], dst: &mut [u8], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        rgba_from_rgb_u8_neon(src, dst, npixels);
        return;
    }
    #[allow(unreachable_code)]
    rgba_from_rgb_u8_scalar(src, dst, npixels);
}

/// NEON RGB8 -> RGBA8: 16 pixels per iteration. `vld3q_u8` deinterleaves R/G/B;
/// `vst4q_u8` reinterleaves them with a `vdupq_n_u8(255)` alpha lane.
#[cfg(target_arch = "aarch64")]
fn rgba_from_rgb_u8_neon(src: &[u8], dst: &mut [u8], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let alpha = vdupq_n_u8(255);
        let bulk = npixels & !15;
        let mut i = 0usize;
        while i < bulk {
            let v = vld3q_u8(sp.add(i * 3));
            vst4q_u8(dp.add(i * 4), uint8x16x4_t(v.0, v.1, v.2, alpha));
            i += 16;
        }
        while i < npixels {
            let s = i * 3;
            let d = i * 4;
            *dp.add(d) = *sp.add(s);
            *dp.add(d + 1) = *sp.add(s + 1);
            *dp.add(d + 2) = *sp.add(s + 2);
            *dp.add(d + 3) = 255;
            i += 1;
        }
    }
}

/// Portable scalar fallback / oracle for RGB8 -> RGBA8.
#[allow(dead_code)]
fn rgba_from_rgb_u8_scalar(src: &[u8], dst: &mut [u8], npixels: usize) {
    for i in 0..npixels {
        let s = i * 3;
        let d = i * 4;
        dst[d] = src[s];
        dst[d + 1] = src[s + 1];
        dst[d + 2] = src[s + 2];
        dst[d + 3] = 255;
    }
}

/// RGB f32 -> RGBA f32: copy R/G/B and append an opaque alpha (1.0).
pub fn rgba_from_rgb_f32(src: &[f32], dst: &mut [f32], npixels: usize) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 4);
    par_strip_dispatch_nm(src, dst, npixels, 3, 4, 4, rgba_from_rgb_f32_kernel);
}

#[inline]
fn rgba_from_rgb_f32_kernel(src: &[f32], dst: &mut [f32], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        rgba_from_rgb_f32_neon(src, dst, npixels);
        return;
    }
    #[allow(unreachable_code)]
    rgba_from_rgb_f32_scalar(src, dst, npixels);
}

/// NEON RGB f32 -> RGBA f32: 4 pixels per iteration (`vld3q_f32` + `vst4q_f32`,
/// alpha lane = `vdupq_n_f32(1.0)`).
#[cfg(target_arch = "aarch64")]
fn rgba_from_rgb_f32_neon(src: &[f32], dst: &mut [f32], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let alpha = vdupq_n_f32(1.0);
        let bulk = npixels & !3;
        let mut i = 0usize;
        while i < bulk {
            let v = vld3q_f32(sp.add(i * 3));
            vst4q_f32(dp.add(i * 4), float32x4x4_t(v.0, v.1, v.2, alpha));
            i += 4;
        }
        while i < npixels {
            let s = i * 3;
            let d = i * 4;
            *dp.add(d) = *sp.add(s);
            *dp.add(d + 1) = *sp.add(s + 1);
            *dp.add(d + 2) = *sp.add(s + 2);
            *dp.add(d + 3) = 1.0;
            i += 1;
        }
    }
}

/// Portable scalar fallback / oracle for RGB f32 -> RGBA f32.
#[allow(dead_code)]
fn rgba_from_rgb_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize) {
    for i in 0..npixels {
        let s = i * 3;
        let d = i * 4;
        dst[d] = src[s];
        dst[d + 1] = src[s + 1];
        dst[d + 2] = src[s + 2];
        dst[d + 3] = 1.0;
    }
}

// ===== RGB -> BGRA (3->4 swap R/B + add opaque alpha) ==============================

/// RGB8 -> BGRA8: swap R/B into B/G/R order and append an opaque alpha (255).
pub fn bgra_from_rgb_u8(src: &[u8], dst: &mut [u8], npixels: usize) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 4);
    par_strip_dispatch_nm(src, dst, npixels, 3, 4, 16, bgra_from_rgb_u8_kernel);
}

#[inline]
fn bgra_from_rgb_u8_kernel(src: &[u8], dst: &mut [u8], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        bgra_from_rgb_u8_neon(src, dst, npixels);
        return;
    }
    #[allow(unreachable_code)]
    bgra_from_rgb_u8_scalar(src, dst, npixels);
}

/// NEON RGB8 -> BGRA8: 16 pixels per iteration. `vld3q_u8` deinterleaves R/G/B;
/// `vst4q_u8` stores them as B/G/R/A with `.0`/`.2` swapped and alpha = 255.
#[cfg(target_arch = "aarch64")]
fn bgra_from_rgb_u8_neon(src: &[u8], dst: &mut [u8], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let alpha = vdupq_n_u8(255);
        let bulk = npixels & !15;
        let mut i = 0usize;
        while i < bulk {
            let v = vld3q_u8(sp.add(i * 3));
            vst4q_u8(dp.add(i * 4), uint8x16x4_t(v.2, v.1, v.0, alpha));
            i += 16;
        }
        while i < npixels {
            let s = i * 3;
            let d = i * 4;
            *dp.add(d) = *sp.add(s + 2);
            *dp.add(d + 1) = *sp.add(s + 1);
            *dp.add(d + 2) = *sp.add(s);
            *dp.add(d + 3) = 255;
            i += 1;
        }
    }
}

/// Portable scalar fallback / oracle for RGB8 -> BGRA8.
#[allow(dead_code)]
fn bgra_from_rgb_u8_scalar(src: &[u8], dst: &mut [u8], npixels: usize) {
    for i in 0..npixels {
        let s = i * 3;
        let d = i * 4;
        dst[d] = src[s + 2];
        dst[d + 1] = src[s + 1];
        dst[d + 2] = src[s];
        dst[d + 3] = 255;
    }
}

/// RGB f32 -> BGRA f32: swap R/B into B/G/R order and append an opaque alpha (1.0).
pub fn bgra_from_rgb_f32(src: &[f32], dst: &mut [f32], npixels: usize) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 4);
    par_strip_dispatch_nm(src, dst, npixels, 3, 4, 4, bgra_from_rgb_f32_kernel);
}

#[inline]
fn bgra_from_rgb_f32_kernel(src: &[f32], dst: &mut [f32], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        bgra_from_rgb_f32_neon(src, dst, npixels);
        return;
    }
    #[allow(unreachable_code)]
    bgra_from_rgb_f32_scalar(src, dst, npixels);
}

/// NEON RGB f32 -> BGRA f32: 4 pixels per iteration (`vld3q_f32` + `vst4q_f32`,
/// R/B swapped, alpha lane = `vdupq_n_f32(1.0)`).
#[cfg(target_arch = "aarch64")]
fn bgra_from_rgb_f32_neon(src: &[f32], dst: &mut [f32], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let alpha = vdupq_n_f32(1.0);
        let bulk = npixels & !3;
        let mut i = 0usize;
        while i < bulk {
            let v = vld3q_f32(sp.add(i * 3));
            vst4q_f32(dp.add(i * 4), float32x4x4_t(v.2, v.1, v.0, alpha));
            i += 4;
        }
        while i < npixels {
            let s = i * 3;
            let d = i * 4;
            *dp.add(d) = *sp.add(s + 2);
            *dp.add(d + 1) = *sp.add(s + 1);
            *dp.add(d + 2) = *sp.add(s);
            *dp.add(d + 3) = 1.0;
            i += 1;
        }
    }
}

/// Portable scalar fallback / oracle for RGB f32 -> BGRA f32.
#[allow(dead_code)]
fn bgra_from_rgb_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize) {
    for i in 0..npixels {
        let s = i * 3;
        let d = i * 4;
        dst[d] = src[s + 2];
        dst[d + 1] = src[s + 1];
        dst[d + 2] = src[s];
        dst[d + 3] = 1.0;
    }
}
