//! x86_64-specific SIMD kernels for color conversion.
//!
//! These paths use `std::arch::x86_64` intrinsics directly for operations
//! where the `wide` crate leaves performance on the table — primarily those
//! requiring byte-level shuffles (`pshufb`) for de-interleaving RGB pixel
//! triplets into planar R/G/B lane vectors.
//!
//! The functions here are `unsafe` and annotated with `#[target_feature]`;
//! the public entry points in `super::color` dispatch to them at runtime via
//! [`std::arch::is_x86_feature_detected!`].

use std::arch::x86_64::*;

/// SSSE3 kernel: convert one row of 3-channel u8 RGB to 1-channel u8 gray
/// using the integer MAC `Y = (77·R + 150·G + 29·B) >> 8`.
///
/// Processes 16 pixels (48 input bytes → 16 output bytes) per iteration:
/// three unaligned 128-bit loads, three `pshufb`-based deinterleave stages
/// (9 shuffles + 6 ORs), then unsigned widen → 3× `pmullw` → `pmaddwd`-free
/// sum → `psrlw 8` → `packuswb` → store. Scalar tail for the last <16 pixels.
///
/// # Safety
/// Caller must guarantee the target supports SSSE3. `dst.len()` is the pixel
/// count; `src.len()` must be at least `3 * dst.len()`.
#[target_feature(enable = "ssse3")]
pub unsafe fn gray_row_u8_ssse3(src: &[u8], dst: &mut [u8]) {
    debug_assert!(src.len() >= 3 * dst.len());
    const LANES: usize = 16;

    // For each of the three 16-byte input chunks (covering 48 bytes = 16
    // pixels), shuffle masks to extract R/G/B bytes into their final
    // destination lanes. A mask byte with high bit set produces 0 in pshufb,
    // so three chunk-shuffles OR'd together assemble a full 16-byte plane.
    //
    // Input layout per chunk (0..16 | 16..32 | 32..48):
    //   chunk0: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5
    //   chunk1: G5 B5 R6 G6 B6 R7 G7 B7 R8 G8 B8 R9 G9 B9 R10 G10
    //   chunk2: B10 R11 G11 B11 R12 G12 B12 R13 G13 B13 R14 G14 B14 R15 G15 B15
    let m_r0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_r1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let m_r2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);

    let m_g0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_g1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let m_g2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);

    let m_b0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_b1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let m_b2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);

    let w_r = _mm_set1_epi16(77);
    let w_g = _mm_set1_epi16(150);
    let w_b = _mm_set1_epi16(29);
    let zero = _mm_setzero_si128();

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    let n = dst.len();
    let vec_end = n - (n % LANES);

    let mut i = 0;
    while i < vec_end {
        let base = i * 3;
        let c0 = _mm_loadu_si128(src_ptr.add(base) as *const __m128i);
        let c1 = _mm_loadu_si128(src_ptr.add(base + 16) as *const __m128i);
        let c2 = _mm_loadu_si128(src_ptr.add(base + 32) as *const __m128i);

        // Deinterleave: OR three partial-plane shuffles per channel.
        let r = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(c0, m_r0), _mm_shuffle_epi8(c1, m_r1)),
            _mm_shuffle_epi8(c2, m_r2),
        );
        let g = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(c0, m_g0), _mm_shuffle_epi8(c1, m_g1)),
            _mm_shuffle_epi8(c2, m_g2),
        );
        let b = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(c0, m_b0), _mm_shuffle_epi8(c1, m_b1)),
            _mm_shuffle_epi8(c2, m_b2),
        );

        // Widen each plane to two u16x8 halves.
        let r_lo = _mm_unpacklo_epi8(r, zero);
        let r_hi = _mm_unpackhi_epi8(r, zero);
        let g_lo = _mm_unpacklo_epi8(g, zero);
        let g_hi = _mm_unpackhi_epi8(g, zero);
        let b_lo = _mm_unpacklo_epi8(b, zero);
        let b_hi = _mm_unpackhi_epi8(b, zero);

        // Y = (77·R + 150·G + 29·B) >> 8, per 16-bit lane. 255·(77+150+29) =
        // 65280 fits in u16, so low-half mul+add is correct.
        let y_lo = _mm_srli_epi16(
            _mm_add_epi16(
                _mm_add_epi16(_mm_mullo_epi16(r_lo, w_r), _mm_mullo_epi16(g_lo, w_g)),
                _mm_mullo_epi16(b_lo, w_b),
            ),
            8,
        );
        let y_hi = _mm_srli_epi16(
            _mm_add_epi16(
                _mm_add_epi16(_mm_mullo_epi16(r_hi, w_r), _mm_mullo_epi16(g_hi, w_g)),
                _mm_mullo_epi16(b_hi, w_b),
            ),
            8,
        );
        // Pack back to u8 with saturation (values are already in [0,255]).
        let y = _mm_packus_epi16(y_lo, y_hi);
        _mm_storeu_si128(dst_ptr.add(i) as *mut __m128i, y);

        i += LANES;
    }

    // Scalar tail for the remaining <16 pixels.
    while i < n {
        let base = i * 3;
        let r = *src.get_unchecked(base) as u16;
        let g = *src.get_unchecked(base + 1) as u16;
        let b = *src.get_unchecked(base + 2) as u16;
        *dst.get_unchecked_mut(i) = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
        i += 1;
    }
}

/// AVX2 kernel: u8 RGB → u8 Gray, 32 pixels (96 input bytes → 32 output bytes)
/// per iteration. Matches the 256-bit width LLVM autovec produces from the
/// scalar `chunks_exact(3).zip` loop.
///
/// Strategy: three contiguous 256-bit loads, then `vperm2i128` to re-lane
/// them so each `ymm_chunk*` register's low/high 128-bit lanes both hold the
/// same "chunk type" (same RGB phase within 3-byte stride). That turns the
/// problem into two 16-pixel SSSE3-style deinterleaves running in parallel,
/// using the same 16-byte mask broadcast to both lanes. After `pshufb` + OR,
/// widen each half with `vpmovzxbw`, do u16 MAC, `packus`, then `vpermq
/// 0xD8` to fix the lane-local-pack byte order.
///
/// # Safety
/// Target must support AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn gray_row_u8_avx2(src: &[u8], dst: &mut [u8]) {
    debug_assert!(src.len() >= 3 * dst.len());
    const LANES: usize = 32;

    // 128-bit chunk-local masks (same as the SSSE3 version).
    let m_r0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_r1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let m_r2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
    let m_g0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_g1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let m_g2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
    let m_b0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_b1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let m_b2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);

    // Broadcast each 128-bit mask to both lanes of a 256-bit mask.
    let br = |m: __m128i| _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(m), m);
    let mr0 = br(m_r0);
    let mr1 = br(m_r1);
    let mr2 = br(m_r2);
    let mg0 = br(m_g0);
    let mg1 = br(m_g1);
    let mg2 = br(m_g2);
    let mb0 = br(m_b0);
    let mb1 = br(m_b1);
    let mb2 = br(m_b2);

    let w_r = _mm256_set1_epi16(77);
    let w_g = _mm256_set1_epi16(150);
    let w_b = _mm256_set1_epi16(29);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let n = dst.len();
    let vec_end = n - (n % LANES);

    let mut i = 0;
    while i < vec_end {
        let base = i * 3;
        // Three contiguous 256-bit loads cover 96 bytes = 32 RGB pixels.
        //  a = [bytes 0..32]  = low:chunk0  | high:chunk1
        //  b = [bytes 32..64] = low:chunk2  | high:chunk0_of_next16
        //  c = [bytes 64..96] = low:chunk1_of_next16 | high:chunk2_of_next16
        let a = _mm256_loadu_si256(src_ptr.add(base) as *const __m256i);
        let b = _mm256_loadu_si256(src_ptr.add(base + 32) as *const __m256i);
        let c = _mm256_loadu_si256(src_ptr.add(base + 64) as *const __m256i);

        // Re-lane so each ymm_chunkN register has the SAME chunk type in
        // both 128-bit lanes (block0 in low, block1 in high):
        //  ymm_c0 = [a.low | b.high]  -> both chunk0
        //  ymm_c1 = [a.high | c.low]  -> both chunk1
        //  ymm_c2 = [b.low | c.high]  -> both chunk2
        // permute2x128 imm8 encoding: [hi_src:3:2] [lo_src:1:0], 0/1 = lo/hi
        // of first arg, 2/3 = lo/hi of second arg.
        let y_c0 = _mm256_permute2x128_si256::<0x30>(a, b); // lo=a.lo(0), hi=b.hi(3)
        let y_c1 = _mm256_permute2x128_si256::<0x21>(a, c); // lo=a.hi(1), hi=c.lo(2)
        let y_c2 = _mm256_permute2x128_si256::<0x30>(b, c); // lo=b.lo(0), hi=c.hi(3)

        // Now each 128-bit lane can be deinterleaved independently with the
        // same mask, so _mm256_shuffle_epi8 (which is lane-local) works.
        let r = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_shuffle_epi8(y_c0, mr0),
                _mm256_shuffle_epi8(y_c1, mr1),
            ),
            _mm256_shuffle_epi8(y_c2, mr2),
        );
        let g = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_shuffle_epi8(y_c0, mg0),
                _mm256_shuffle_epi8(y_c1, mg1),
            ),
            _mm256_shuffle_epi8(y_c2, mg2),
        );
        let bb = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_shuffle_epi8(y_c0, mb0),
                _mm256_shuffle_epi8(y_c1, mb1),
            ),
            _mm256_shuffle_epi8(y_c2, mb2),
        );

        // Widen each 32-byte plane to 2× u16x16 via vpmovzxbw. Natural byte
        // order is preserved: r_lo = [R0..R15], r_hi = [R16..R31].
        let r_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r));
        let r_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r));
        let g_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g));
        let g_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(g));
        let b_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(bb));
        let b_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(bb));

        let y_lo = _mm256_srli_epi16::<8>(_mm256_add_epi16(
            _mm256_add_epi16(
                _mm256_mullo_epi16(r_lo, w_r),
                _mm256_mullo_epi16(g_lo, w_g),
            ),
            _mm256_mullo_epi16(b_lo, w_b),
        ));
        let y_hi = _mm256_srli_epi16::<8>(_mm256_add_epi16(
            _mm256_add_epi16(
                _mm256_mullo_epi16(r_hi, w_r),
                _mm256_mullo_epi16(g_hi, w_g),
            ),
            _mm256_mullo_epi16(b_hi, w_b),
        ));

        // packus interleaves y_lo/y_hi per-128-bit-lane, so the resulting
        // bytes are in order [Y0..Y7, Y16..Y23, Y8..Y15, Y24..Y31]. Fix with
        // vpermq imm8=0xD8 (quadword indices [0,2,1,3]).
        let y = _mm256_packus_epi16(y_lo, y_hi);
        let y = _mm256_permute4x64_epi64::<0xD8>(y);
        _mm256_storeu_si256(dst_ptr.add(i) as *mut __m256i, y);

        i += LANES;
    }

    // Scalar tail.
    while i < n {
        let base = i * 3;
        let r = *src.get_unchecked(base) as u16;
        let g = *src.get_unchecked(base + 1) as u16;
        let b = *src.get_unchecked(base + 2) as u16;
        *dst.get_unchecked_mut(i) = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
        i += 1;
    }
}

