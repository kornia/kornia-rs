//! x86_64 AVX2 kernels for threshold operations.

use std::arch::x86_64::*;

/// AVX2 binary threshold for u8: `out = src > threshold ? max_value : 0`.
///
/// Uses the `vpminub`/`vpcmpeqb`/`vpandn` pattern (same as LLVM's autovec
/// baseline) to avoid the need for a signed-bias XOR: `vpminub` is an
/// unsigned min, so `v == min(v, threshold)` is the "v ≤ threshold" mask —
/// whose bitwise complement is "v > threshold". `vpandn(mask_le, max_v)`
/// writes `max_v` where the mask is zero and `0` elsewhere — exactly the
/// thresholding semantics, in 3 µops per 32-byte block.
///
/// Unrolled 4× (128 bytes per loop iteration) to keep the AVX2 ports busy:
/// `vpminub`/`vpcmpeqb` live on p0/p1 and loads/stores on p23/p4 on Intel,
/// so four independent dependency chains per iter fill the pipeline.
///
/// # Safety
/// Target must support AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn threshold_row_u8_avx2(src: &[u8], dst: &mut [u8], threshold: u8, max_value: u8) {
    debug_assert_eq!(src.len(), dst.len());
    const LANES: usize = 128; // 4 × 32 bytes per unrolled iteration

    let thr = _mm256_set1_epi8(threshold as i8);
    let max_v = _mm256_set1_epi8(max_value as i8);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let n = src.len();
    let vec_end = n - (n % LANES);

    let mut i = 0;
    while i < vec_end {
        let v0 = _mm256_loadu_si256(src_ptr.add(i) as *const __m256i);
        let v1 = _mm256_loadu_si256(src_ptr.add(i + 32) as *const __m256i);
        let v2 = _mm256_loadu_si256(src_ptr.add(i + 64) as *const __m256i);
        let v3 = _mm256_loadu_si256(src_ptr.add(i + 96) as *const __m256i);

        // min(v, threshold) — independent per lane
        let m0 = _mm256_min_epu8(v0, thr);
        let m1 = _mm256_min_epu8(v1, thr);
        let m2 = _mm256_min_epu8(v2, thr);
        let m3 = _mm256_min_epu8(v3, thr);

        // le_mask = (v == min) → 0xFF where v ≤ threshold
        let le0 = _mm256_cmpeq_epi8(v0, m0);
        let le1 = _mm256_cmpeq_epi8(v1, m1);
        let le2 = _mm256_cmpeq_epi8(v2, m2);
        let le3 = _mm256_cmpeq_epi8(v3, m3);

        // andnot(le_mask, max_v) = max_v where le_mask==0 (i.e. v > threshold)
        let o0 = _mm256_andnot_si256(le0, max_v);
        let o1 = _mm256_andnot_si256(le1, max_v);
        let o2 = _mm256_andnot_si256(le2, max_v);
        let o3 = _mm256_andnot_si256(le3, max_v);

        _mm256_storeu_si256(dst_ptr.add(i) as *mut __m256i, o0);
        _mm256_storeu_si256(dst_ptr.add(i + 32) as *mut __m256i, o1);
        _mm256_storeu_si256(dst_ptr.add(i + 64) as *mut __m256i, o2);
        _mm256_storeu_si256(dst_ptr.add(i + 96) as *mut __m256i, o3);

        i += LANES;
    }

    // 32-byte tail
    while i + 32 <= n {
        let v = _mm256_loadu_si256(src_ptr.add(i) as *const __m256i);
        let m = _mm256_min_epu8(v, thr);
        let le = _mm256_cmpeq_epi8(v, m);
        let o = _mm256_andnot_si256(le, max_v);
        _mm256_storeu_si256(dst_ptr.add(i) as *mut __m256i, o);
        i += 32;
    }

    // Scalar tail for the last <32 bytes.
    while i < n {
        *dst.get_unchecked_mut(i) = if *src.get_unchecked(i) > threshold {
            max_value
        } else {
            0
        };
        i += 1;
    }
}
