/// BT.601 f32 luma weights (Y = 0.299·R + 0.587·G + 0.114·B).
pub(super) const RW_F32: f32 = 0.299;
pub(super) const GW_F32: f32 = 0.587;
pub(super) const BW_F32: f32 = 0.114;

/// BT.601 luma weights in OpenCV's Q14 fixed point — `(R·4899 + G·9617 +
/// B·1868 + 8192) >> 14` is byte-for-byte what `cv2.cvtColor(RGB2GRAY)`
/// computes for u8, so kornia's u8 gray (CPU and CUDA, which share these
/// constants) matches OpenCV exactly.
pub(crate) const RW_Q14: u32 = 4899;
pub(crate) const GW_Q14: u32 = 9617;
pub(crate) const BW_Q14: u32 = 1868;
pub(crate) const Q14_GRAY_HALF: u32 = 1 << 13;

use super::super::kernel_common::par_strip_dispatch_nm;

// ===== RGB8 → Gray8 ================================================================

/// RGB u8 to grayscale u8: gray = (4899*R + 9617*G + 1868*B + 8192) >> 14
/// — OpenCV's exact Q14 formula (byte-parity with cv2).
///
/// Parallelized over row-strips for large images; single-threaded SIMD below
/// the threshold to avoid rayon dispatch overhead.
pub fn gray_from_rgb_u8(src: &[u8], dst: &mut [u8], npixels: usize) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels);
    // 32-pixel alignment keeps the SIMD bulk loop (2× vld3q_u8) intact at strip boundaries.
    par_strip_dispatch_nm(src, dst, npixels, 3, 1, 32, gray_from_rgb_u8_kernel);
}

/// Kernel dispatcher: NEON (aarch64), AVX2 (x86_64), or scalar fallback.
#[inline]
fn gray_from_rgb_u8_kernel(src: &[u8], dst: &mut [u8], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        gray_from_rgb_u8_neon(src, dst, npixels);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 {
            // SAFETY: AVX2 confirmed by the runtime probe.
            unsafe { gray_from_rgb_u8_avx2(src, dst, npixels) };
            return;
        }
    }
    #[allow(unreachable_code)]
    gray_from_rgb_u8_scalar(src, dst, npixels);
}

/// NEON RGB u8 → gray u8: 16 pixels per iteration (vld3q_u8) in OpenCV's
/// Q14 fixed point. The weights exceed 8 bits, so the chain widens
/// u8 → u16 → u32 (`vmull_n_u16`/`vmlal_n_u16`), adds the rounding half,
/// shifts down 14 and narrows back to u8. Bit-exact with the scalar oracle
/// and with cv2's u8 path.
#[cfg(target_arch = "aarch64")]
fn gray_from_rgb_u8_neon(src: &[u8], dst: &mut [u8], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let half = vdupq_n_u32(Q14_GRAY_HALF);

        #[inline(always)]
        unsafe fn q14_u16x8(
            r: uint16x8_t,
            g: uint16x8_t,
            b: uint16x8_t,
            half: uint32x4_t,
        ) -> uint16x8_t {
            let lo = vmlal_n_u16(
                vmlal_n_u16(
                    vmlal_n_u16(half, vget_low_u16(r), RW_Q14 as u16),
                    vget_low_u16(g),
                    GW_Q14 as u16,
                ),
                vget_low_u16(b),
                BW_Q14 as u16,
            );
            let hi = vmlal_n_u16(
                vmlal_n_u16(
                    vmlal_n_u16(half, vget_high_u16(r), RW_Q14 as u16),
                    vget_high_u16(g),
                    GW_Q14 as u16,
                ),
                vget_high_u16(b),
                BW_Q14 as u16,
            );
            vcombine_u16(vshrn_n_u32(lo, 14), vshrn_n_u32(hi, 14))
        }

        let bulk16 = npixels & !15;
        let mut i = 0usize;
        while i < bulk16 {
            let rgb = vld3q_u8(sp.add(i * 3));
            let r_lo = vmovl_u8(vget_low_u8(rgb.0));
            let r_hi = vmovl_u8(vget_high_u8(rgb.0));
            let g_lo = vmovl_u8(vget_low_u8(rgb.1));
            let g_hi = vmovl_u8(vget_high_u8(rgb.1));
            let b_lo = vmovl_u8(vget_low_u8(rgb.2));
            let b_hi = vmovl_u8(vget_high_u8(rgb.2));
            let y_lo = q14_u16x8(r_lo, g_lo, b_lo, half);
            let y_hi = q14_u16x8(r_hi, g_hi, b_hi, half);
            vst1q_u8(dp.add(i), vcombine_u8(vmovn_u16(y_lo), vmovn_u16(y_hi)));
            i += 16;
        }
        // Scalar tail
        while i < npixels {
            let si = i * 3;
            *dp.add(i) = ((RW_Q14 * *sp.add(si) as u32
                + GW_Q14 * *sp.add(si + 1) as u32
                + BW_Q14 * *sp.add(si + 2) as u32
                + Q14_GRAY_HALF)
                >> 14) as u8;
            i += 1;
        }
    }
}

/// AVX2 RGB→gray u8: 16 pixels per iteration.
///
/// AVX2 lacks a 3-way byte deinterleave, so we use SSSE3 PSHUFB with three
/// per-channel mask triplets to gather R, G, B from three 16-byte chunks of
/// the input stream. Then widen to u16, FMA-style accumulate the weighted
/// sum (77*R + 150*G + 29*B), shift right by 8, and pack back to u8.
///
/// # Safety
/// - Caller must ensure AVX2 is available.
/// - `src.len() >= npixels * 3`, `dst.len() >= npixels`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gray_from_rgb_u8_avx2(src: &[u8], dst: &mut [u8], npixels: usize) {
    use std::arch::x86_64::*;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    // Per-channel deinterleave masks for 16 pixels (48 input bytes split into
    // three 16-byte chunks `a`, `b`, `c`). `-1` byte zeros that lane via PSHUFB.
    // Output lanes 0..15 hold the channel values for pixels 0..15.
    let m_a_r = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_b_r = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let m_c_r = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
    let m_a_g = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_b_g = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let m_c_g = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
    let m_a_b = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_b_b = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let m_c_b = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);

    let half = _mm256_set1_epi32(Q14_GRAY_HALF as i32);
    let w_r = _mm256_set1_epi32(RW_Q14 as i32);
    let w_g = _mm256_set1_epi32(GW_Q14 as i32);
    let w_b = _mm256_set1_epi32(BW_Q14 as i32);

    let bulk = npixels & !15;
    let mut i = 0usize;
    while i < bulk {
        let a = _mm_loadu_si128(sp.add(i * 3) as *const __m128i);
        let b = _mm_loadu_si128(sp.add(i * 3 + 16) as *const __m128i);
        let c = _mm_loadu_si128(sp.add(i * 3 + 32) as *const __m128i);

        let r = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(a, m_a_r), _mm_shuffle_epi8(b, m_b_r)),
            _mm_shuffle_epi8(c, m_c_r),
        );
        let g = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(a, m_a_g), _mm_shuffle_epi8(b, m_b_g)),
            _mm_shuffle_epi8(c, m_c_g),
        );
        let bch = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(a, m_a_b), _mm_shuffle_epi8(b, m_b_b)),
            _mm_shuffle_epi8(c, m_c_b),
        );

        // Q14 weights exceed 16-bit products, so widen u8 → epi32 and use
        // 32-bit MACs; (acc + 8192) >> 14, then pack 32→16→8.
        #[inline(always)]
        unsafe fn q14_8px(
            px: __m128i,
            w_r: __m256i,
            w_g: __m256i,
            w_b: __m256i,
            half: __m256i,
            g: __m128i,
            bch: __m128i,
        ) -> __m128i {
            let r32 = _mm256_cvtepu8_epi32(px);
            let g32 = _mm256_cvtepu8_epi32(g);
            let b32 = _mm256_cvtepu8_epi32(bch);
            let acc = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(r32, w_r), _mm256_mullo_epi32(g32, w_g)),
                    _mm256_mullo_epi32(b32, w_b),
                ),
                half,
            );
            let sh = _mm256_srli_epi32(acc, 14);
            let packed16 = _mm256_packus_epi32(sh, sh); // lanes duplicated per 128-bit half
            let lo = _mm256_castsi256_si128(packed16);
            let hi = _mm256_extracti128_si256(packed16, 1);
            let u16x8 = _mm_unpacklo_epi64(lo, hi);
            _mm_packus_epi16(u16x8, u16x8)
        }
        let y0 = q14_8px(r, w_r, w_g, w_b, half, g, bch);
        let r_hi = _mm_srli_si128(r, 8);
        let g_hi = _mm_srli_si128(g, 8);
        let b_hi = _mm_srli_si128(bch, 8);
        let y1 = q14_8px(r_hi, w_r, w_g, w_b, half, g_hi, b_hi);
        _mm_storeu_si128(dp.add(i) as *mut __m128i, _mm_unpacklo_epi64(y0, y1));

        i += 16;
    }

    // Scalar tail.
    while i < npixels {
        let si = i * 3;
        *dp.add(i) = ((RW_Q14 * *sp.add(si) as u32
            + GW_Q14 * *sp.add(si + 1) as u32
            + BW_Q14 * *sp.add(si + 2) as u32
            + Q14_GRAY_HALF)
            >> 14) as u8;
        i += 1;
    }
}

/// Portable scalar fallback — referenced when neither NEON nor AVX2 is available.
#[allow(dead_code)]
fn gray_from_rgb_u8_scalar(src: &[u8], dst: &mut [u8], npixels: usize) {
    for (i, out) in dst.iter_mut().take(npixels).enumerate() {
        let si = i * 3;
        *out = ((RW_Q14 * src[si] as u32
            + GW_Q14 * src[si + 1] as u32
            + BW_Q14 * src[si + 2] as u32
            + Q14_GRAY_HALF)
            >> 14) as u8;
    }
}

// ===== RGB f32 → Gray f32 ==========================================================

/// Slice-level RGB f32 → grayscale f32.
///
/// Parallelized over row-strips for large images (> [`PAR_THRESHOLD`] px); single-threaded
/// SIMD below the threshold to avoid rayon dispatch overhead.
pub fn gray_from_rgb_f32(src: &[f32], dst: &mut [f32], npixels: usize) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels);
    // 8-pixel alignment keeps both NEON (8 px/iter) and AVX2 (8 px/iter) tails intact.
    par_strip_dispatch_nm(src, dst, npixels, 3, 1, 8, gray_from_rgb_f32_kernel);
}

/// Kernel dispatcher: NEON (aarch64), AVX2+FMA (x86_64), or scalar fallback.
#[inline]
fn gray_from_rgb_f32_kernel(src: &[f32], dst: &mut [f32], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        gray_from_rgb_f32_neon(src, dst, npixels);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 && cpu.has_fma {
            // SAFETY: AVX2+FMA confirmed by runtime probe.
            unsafe { gray_from_rgb_f32_avx2(src, dst, npixels) };
            return;
        }
    }
    #[allow(unreachable_code)]
    gray_from_rgb_f32_scalar(src, dst, npixels);
}

/// NEON RGB f32 → gray f32: 8 pixels per iteration (2× vld3q_f32).
///
/// `vld3q_f32` reads 4 RGB pixels (12 f32 values) and deinterleaves them into
/// separate R, G, B `float32x4_t` lanes in a single structured load — no shuffle
/// instructions required. Two independent loads + FMA chains let the OoO A78AE
/// overlap both sequences across both load pipes and both FMA pipes.
///
/// Loop structure:
/// - Bulk 8-pixel step (2× vld3q_f32 + 2× FMA chain + 2× vst1q_f32)
/// - 4-pixel remainder step (1× vld3q_f32 + FMA chain + vst1q_f32)
/// - Scalar tail (0–3 pixels)
#[cfg(target_arch = "aarch64")]
fn gray_from_rgb_f32_neon(src: &[f32], dst: &mut [f32], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let wr = vdupq_n_f32(RW_F32);
        let wg = vdupq_n_f32(GW_F32);
        let wb = vdupq_n_f32(BW_F32);
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();

        // 8 pixels per iteration (2× vld3q_f32).
        // Chain: y = r*RW + g*GW + b*BW
        //          = fmadd(r, wr, fmadd(g, wg, mul(b, wb)))
        let bulk8 = npixels & !7;
        let mut i = 0usize;
        while i < bulk8 {
            let a = vld3q_f32(sp.add(i * 3)); // pixels i..i+4: .0=R .1=G .2=B
            let b = vld3q_f32(sp.add((i + 4) * 3)); // pixels i+4..i+8

            let ya = vfmaq_f32(vfmaq_f32(vmulq_f32(a.2, wb), a.1, wg), a.0, wr);
            let yb = vfmaq_f32(vfmaq_f32(vmulq_f32(b.2, wb), b.1, wg), b.0, wr);

            vst1q_f32(dp.add(i), ya);
            vst1q_f32(dp.add(i + 4), yb);
            i += 8;
        }
        // 4-pixel remainder
        if i + 4 <= npixels {
            let a = vld3q_f32(sp.add(i * 3));
            let ya = vfmaq_f32(vfmaq_f32(vmulq_f32(a.2, wb), a.1, wg), a.0, wr);
            vst1q_f32(dp.add(i), ya);
            i += 4;
        }
        // Scalar tail (0–3 pixels)
        while i < npixels {
            let si = i * 3;
            *dp.add(i) = RW_F32 * *sp.add(si) + GW_F32 * *sp.add(si + 1) + BW_F32 * *sp.add(si + 2);
            i += 1;
        }
    }
}

/// AVX2+FMA RGB f32 → grayscale f32: 8 pixels per iteration.
///
/// Three sequential 256-bit loads cover 8 RGB pixels (24 f32 values).
/// `_mm256_permutevar8x32_ps` + `_mm256_blend_ps` deinterleave R, G, B lanes;
/// `_mm256_fmadd_ps` accumulates the weighted sum in a single pass.
/// No gather instructions are used.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available (`cpuid` check).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gray_from_rgb_f32_avx2(src: &[f32], dst: &mut [f32], npixels: usize) {
    use std::arch::x86_64::*;

    let rw = _mm256_set1_ps(RW_F32);
    let gw = _mm256_set1_ps(GW_F32);
    let bw = _mm256_set1_ps(BW_F32);

    // Deinterleave indices for 8 RGB pixels stored in three 256-bit vectors:
    //   v0: [R0,G0,B0, R1,G1,B1, R2,G2]   (lanes 0-7)
    //   v1: [B2,R3,G3, B3,R4,G4, B4,R5]   (lanes 8-15)
    //   v2: [G5,B5,R6, G6,B6,R7, G7,B7]   (lanes 16-23)
    //
    // _mm256_set_epi32(x7,x6,x5,x4,x3,x2,x1,x0) → lane i = xi.
    let pr0 = _mm256_set_epi32(0, 0, 0, 0, 0, 6, 3, 0);
    let pr1 = _mm256_set_epi32(0, 0, 7, 4, 1, 0, 0, 0);
    let pr2 = _mm256_set_epi32(5, 2, 0, 0, 0, 0, 0, 0);

    let pg0 = _mm256_set_epi32(0, 0, 0, 0, 0, 7, 4, 1);
    let pg1 = _mm256_set_epi32(0, 0, 0, 5, 2, 0, 0, 0);
    let pg2 = _mm256_set_epi32(6, 3, 0, 0, 0, 0, 0, 0);

    let pb0 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 5, 2);
    let pb1 = _mm256_set_epi32(0, 0, 0, 6, 3, 0, 0, 0);
    let pb2 = _mm256_set_epi32(7, 4, 1, 0, 0, 0, 0, 0);

    let mut i = 0usize;
    while i + 8 <= npixels {
        let base = src.as_ptr().add(i * 3);
        let v0 = _mm256_loadu_ps(base);
        let v1 = _mm256_loadu_ps(base.add(8));
        let v2 = _mm256_loadu_ps(base.add(16));

        let r = _mm256_blend_ps::<0xC0>(
            _mm256_blend_ps::<0x38>(
                _mm256_permutevar8x32_ps(v0, pr0),
                _mm256_permutevar8x32_ps(v1, pr1),
            ),
            _mm256_permutevar8x32_ps(v2, pr2),
        );
        let g = _mm256_blend_ps::<0xE0>(
            _mm256_blend_ps::<0x18>(
                _mm256_permutevar8x32_ps(v0, pg0),
                _mm256_permutevar8x32_ps(v1, pg1),
            ),
            _mm256_permutevar8x32_ps(v2, pg2),
        );
        let b = _mm256_blend_ps::<0xE0>(
            _mm256_blend_ps::<0x1C>(
                _mm256_permutevar8x32_ps(v0, pb0),
                _mm256_permutevar8x32_ps(v1, pb1),
            ),
            _mm256_permutevar8x32_ps(v2, pb2),
        );

        let out = _mm256_fmadd_ps(r, rw, _mm256_fmadd_ps(g, gw, _mm256_mul_ps(b, bw)));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), out);
        i += 8;
    }
    // Scalar tail for npixels % 8 != 0.
    while i < npixels {
        let b = i * 3;
        dst[i] = RW_F32 * src[b] + GW_F32 * src[b + 1] + BW_F32 * src[b + 2];
        i += 1;
    }
}

/// Portable scalar fallback — used when neither NEON nor AVX2+FMA is available.
fn gray_from_rgb_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize) {
    for (i, out) in dst.iter_mut().take(npixels).enumerate() {
        let b = i * 3;
        *out = RW_F32 * src[b] + GW_F32 * src[b + 1] + BW_F32 * src[b + 2];
    }
}

// ===== Gray -> RGB (1->3 broadcast) ================================================
//
// Bandwidth-bound: replicate the single gray channel across R/G/B. No arithmetic,
// just a structured load (`vld1q`) + interleaved store (`vst3q`).

/// Gray8 -> RGB8: replicate the single channel across all three output channels.
pub fn rgb_from_gray_u8(src: &[u8], dst: &mut [u8], npixels: usize) {
    debug_assert!(src.len() >= npixels);
    debug_assert!(dst.len() >= npixels * 3);
    par_strip_dispatch_nm(src, dst, npixels, 1, 3, 16, rgb_from_gray_u8_kernel);
}

#[inline]
fn rgb_from_gray_u8_kernel(src: &[u8], dst: &mut [u8], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        rgb_from_gray_u8_neon(src, dst, npixels);
        return;
    }
    #[allow(unreachable_code)]
    rgb_from_gray_u8_scalar(src, dst, npixels);
}

/// NEON Gray8 -> RGB8: 16 pixels per iteration. `vld1q_u8` loads 16 gray values;
/// `vst3q_u8` writes them to all three R/G/B lanes.
#[cfg(target_arch = "aarch64")]
fn rgb_from_gray_u8_neon(src: &[u8], dst: &mut [u8], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let bulk = npixels & !15;
        let mut i = 0usize;
        while i < bulk {
            let g = vld1q_u8(sp.add(i));
            vst3q_u8(dp.add(i * 3), uint8x16x3_t(g, g, g));
            i += 16;
        }
        while i < npixels {
            let g = *sp.add(i);
            let d = i * 3;
            *dp.add(d) = g;
            *dp.add(d + 1) = g;
            *dp.add(d + 2) = g;
            i += 1;
        }
    }
}

/// Portable scalar fallback / oracle for Gray8 -> RGB8.
#[allow(dead_code)]
fn rgb_from_gray_u8_scalar(src: &[u8], dst: &mut [u8], npixels: usize) {
    for (px, &g) in dst.chunks_exact_mut(3).zip(src.iter()).take(npixels) {
        px[0] = g;
        px[1] = g;
        px[2] = g;
    }
}

/// Gray f32 -> RGB f32: replicate the single channel across all three channels.
pub fn rgb_from_gray_f32(src: &[f32], dst: &mut [f32], npixels: usize) {
    debug_assert!(src.len() >= npixels);
    debug_assert!(dst.len() >= npixels * 3);
    par_strip_dispatch_nm(src, dst, npixels, 1, 3, 4, rgb_from_gray_f32_kernel);
}

#[inline]
fn rgb_from_gray_f32_kernel(src: &[f32], dst: &mut [f32], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        rgb_from_gray_f32_neon(src, dst, npixels);
        return;
    }
    #[allow(unreachable_code)]
    rgb_from_gray_f32_scalar(src, dst, npixels);
}

/// NEON Gray f32 -> RGB f32: 4 pixels per iteration (`vld1q_f32`/`vst3q_f32`).
#[cfg(target_arch = "aarch64")]
fn rgb_from_gray_f32_neon(src: &[f32], dst: &mut [f32], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let bulk = npixels & !3;
        let mut i = 0usize;
        while i < bulk {
            let g = vld1q_f32(sp.add(i));
            vst3q_f32(dp.add(i * 3), float32x4x3_t(g, g, g));
            i += 4;
        }
        while i < npixels {
            let g = *sp.add(i);
            let d = i * 3;
            *dp.add(d) = g;
            *dp.add(d + 1) = g;
            *dp.add(d + 2) = g;
            i += 1;
        }
    }
}

/// Portable scalar fallback / oracle for Gray f32 -> RGB f32.
#[allow(dead_code)]
fn rgb_from_gray_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize) {
    for (px, &g) in dst.chunks_exact_mut(3).zip(src.iter()).take(npixels) {
        px[0] = g;
        px[1] = g;
        px[2] = g;
    }
}
