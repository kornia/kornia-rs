//! Brute-force 128-D descriptor matching on CPU, with Lowe's ratio test and a
//! mutual nearest-neighbour check.
//!
//! Mirrors the CUDA matcher's semantics exactly — squared L2 throughout, the
//! same ratio convention, the same mutual check — so a caller gets the same pair
//! set from either backend.
//!
//! # Why squared distances
//!
//! The ratio test `d_best < ratio * d_second` on true distances is equivalent to
//! `d2_best < ratio^2 * d2_second` on squared ones, so no square root is ever
//! taken.
//!
//! # Why no score matrix
//!
//! Each query keeps its best and second-best while tiles of train descriptors
//! pass over it, and the mutual check reads the same distances. Materialising the
//! `n1 x n2` matrix would be tens of megabytes of traffic for a result that is
//! two small arrays.

use super::descriptor::DESCR_LEN;
#[cfg(target_arch = "x86_64")]
use crate::features::r#match::squared_l2_avx2;
use crate::features::r#match::{scan_float_rows, FloatMetric};

/// Squared L2 between two descriptors, four lanes at a time.
///
/// Four independent accumulators, not one: a single accumulator makes this a
/// serial `fma` chain 32 long, which at ~4-cycle latency is latency-bound rather
/// than throughput-bound. Same reason the blur uses four.
///
/// # Arguments
///
/// * `a`, `b` - Descriptors. The vector path runs when both are exactly
///   `DESCR_LEN` (128) long; any other lengths fall back to
///   [`l2_sq_scalar`], which sums over the common prefix — the same result
///   every target produces.
///
/// # Returns
///
/// The squared Euclidean distance.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    // Real (release-mode) length check: the loads below read exactly
    // `DESCR_LEN` floats from each slice.
    if a.len() != DESCR_LEN || b.len() != DESCR_LEN {
        return l2_sq_scalar(a, b);
    }
    // SAFETY: both slices are exactly `DESCR_LEN` long (checked above) and
    // `DESCR_LEN % 16 == 0`, so every 4-lane load is in bounds.
    unsafe {
        let (mut s0, mut s1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let (mut s2, mut s3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let (mut pa, mut pb) = (a.as_ptr(), b.as_ptr());
        for _ in 0..DESCR_LEN / 16 {
            let d0 = vsubq_f32(vld1q_f32(pa), vld1q_f32(pb));
            let d1 = vsubq_f32(vld1q_f32(pa.add(4)), vld1q_f32(pb.add(4)));
            let d2 = vsubq_f32(vld1q_f32(pa.add(8)), vld1q_f32(pb.add(8)));
            let d3 = vsubq_f32(vld1q_f32(pa.add(12)), vld1q_f32(pb.add(12)));
            s0 = vfmaq_f32(s0, d0, d0);
            s1 = vfmaq_f32(s1, d1, d1);
            s2 = vfmaq_f32(s2, d2, d2);
            s3 = vfmaq_f32(s3, d3, d3);
            pa = pa.add(16);
            pb = pb.add(16);
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)))
    }
}

/// Squared L2 between two descriptors.
///
/// On x86_64 with AVX2 and FMA this is their vector form; elsewhere off aarch64 it
/// is [`l2_sq_scalar`]. Mismatched lengths sum over the common prefix on every target.
///
/// # Returns
///
/// The squared Euclidean distance.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 && cpu.has_fma {
            // SAFETY: the shared selector confirms AVX2 and FMA.
            return unsafe { l2_sq_avx2(a, b) };
        }
    }
    l2_sq_scalar(a, b)
}

/// # Safety
///
/// The caller must ensure AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: this function's own target features are the kernel's requirement.
    unsafe { squared_l2_avx2(a, b) }
}

/// Squared L2, scalar. The portable fallback, and the reference the vector form
/// is checked against.
///
/// A plain multiply and add: without FMA in the build's baseline, `f32::mul_add`
/// is a software call for every dimension.
///
/// Kept public and always compiled, not `#[cfg]`-ed away: a fallback that only
/// builds on the platforms that do not use it is a fallback nobody tests.
#[inline]
pub fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        s += d * d;
    }
    s
}

/// Match `d1` against `d2`, returning the surviving `(query, train)` pairs.
///
/// `ratio` is Lowe's ratio on true distances (0.8 is the usual value); `>= 1.0`
/// disables the test. That threshold is not the same as leaving the arithmetic
/// comparison in place: with tied best and second distances — identical or
/// all-zero descriptors — `d1 < 1.0 * d2` is false, so a ratio of exactly 1
/// would reject every match instead of accepting every one.
///
/// `cross_check` additionally requires each pair to be a mutual nearest
/// neighbour.
pub fn match_descriptors(
    d1: &[f32],
    n1: usize,
    d2: &[f32],
    n2: usize,
    ratio: f32,
    cross_check: bool,
) -> Vec<[i32; 2]> {
    match_descriptors_impl(d1, n1, d2, n2, ratio, cross_check, false)
}

/// [`match_descriptors`] forced onto the scalar distance kernel.
///
/// Same results; exists so the fallback is reachable and testable on the
/// platforms that would otherwise never run it.
pub fn match_descriptors_scalar(
    d1: &[f32],
    n1: usize,
    d2: &[f32],
    n2: usize,
    ratio: f32,
    cross_check: bool,
) -> Vec<[i32; 2]> {
    match_descriptors_impl(d1, n1, d2, n2, ratio, cross_check, true)
}

fn match_descriptors_impl(
    d1: &[f32],
    n1: usize,
    d2: &[f32],
    n2: usize,
    ratio: f32,
    cross_check: bool,
    scalar: bool,
) -> Vec<[i32; 2]> {
    if n1 == 0 || n2 == 0 || d1.len() < n1 * DESCR_LEN || d2.len() < n2 * DESCR_LEN {
        return Vec::new();
    }
    let (fwd, rev) = scan_float_rows(
        &d1[..n1 * DESCR_LEN],
        &d2[..n2 * DESCR_LEN],
        DESCR_LEN,
        FloatMetric::SquaredL2,
        cross_check,
        scalar,
    );
    let ratio2 = ratio * ratio;

    let mut out = Vec::new();
    for (i, row) in fwd.iter().enumerate() {
        if row.index == usize::MAX {
            continue;
        }
        let (d_best, d_second) = (row.best, row.second);
        // A second-best of +inf means there was only one candidate; the
        // reference keeps such a match rather than dividing by infinity.
        // Deliberately the negated form, matching the CUDA kernel: with a NaN
        // distance `!(a < b)` rejects where `a >= b` would also reject but for a
        // different reason, and keeping the two spellings identical is what
        // makes the backends return the same pair set.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if ratio2 < 1.0 && d_second < f32::MAX && !(d_best < ratio2 * d_second) {
            continue;
        }
        if cross_check && rev.index[row.index] != i {
            continue;
        }
        out.push([i as i32, row.index as i32]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: on aarch64 `l2_sq` only `debug_assert`ed the lengths and
    /// then issued 128-float NEON loads, reading out of bounds in release for
    /// short slices. Every target must now agree with the scalar reference.
    #[test]
    fn l2_sq_short_or_mismatched_slices_match_scalar() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [0.0f32; 5];
        assert_eq!(l2_sq(&a, &b), l2_sq_scalar(&a, &b));
        assert_eq!(l2_sq(&a, &b), 14.0);
        assert_eq!(l2_sq(&[], &[]), 0.0);
        let long = vec![1.0f32; DESCR_LEN + 3];
        let zero = vec![0.0f32; DESCR_LEN + 3];
        assert_eq!(l2_sq(&long, &zero), (DESCR_LEN + 3) as f32);
    }

    fn rand_desc(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n * DESCR_LEN)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) % 256) as f32
            })
            .collect()
    }

    #[test]
    fn self_match_is_the_identity() {
        let n = 200;
        let d = rand_desc(n, 12345);
        let pairs = match_descriptors(&d, n, &d, n, 1.0, true);
        assert_eq!(pairs.len(), n);
        assert!(pairs.iter().all(|[a, b]| a == b));
    }

    #[test]
    fn l2_sq_vector_form_agrees_with_scalar_at_every_length() {
        for len in [0, 1, 7, 8, 9, 31, 32, 33, 64, 127, DESCR_LEN, 129, 300] {
            let a: Vec<f32> = rand_desc(3, 5)[..len].to_vec();
            let b: Vec<f32> = rand_desc(3, 6)[..len].to_vec();
            assert_eq!(l2_sq(&a, &b), l2_sq_scalar(&a, &b), "length {len}");
        }
    }

    /// Cross-checked pairs and forward pairs must match brute force on sets that cross
    /// tiles and worker tasks, in the CUDA matcher's integer domain.
    #[test]
    fn one_pass_matches_brute_force() {
        let (n1, n2) = (331usize, 217usize);
        let (a, b) = (rand_desc(n1, 31), rand_desc(n2, 32));
        fn row(set: &[f32], i: usize) -> &[f32] {
            &set[i * DESCR_LEN..(i + 1) * DESCR_LEN]
        }
        let table: Vec<Vec<f32>> = (0..n1)
            .map(|i| {
                (0..n2)
                    .map(|j| l2_sq_scalar(row(&a, i), row(&b, j)))
                    .collect()
            })
            .collect();
        let nearest = |values: Vec<f32>| {
            let mut order: Vec<usize> = (0..values.len()).collect();
            order.sort_by(|&p, &q| values[p].total_cmp(&values[q]).then(p.cmp(&q)));
            (order[0], values[order[0]], values[order[1]])
        };
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for (ratio, cross) in [(0.8f32, true), (0.8, false), (1.0, true), (0.95, true)] {
                let mut want = Vec::new();
                for (i, distances) in table.iter().enumerate() {
                    let (j, best, second) = nearest(distances.clone());
                    let back = nearest(table.iter().map(|r| r[j]).collect()).0;
                    if (ratio >= 1.0 || best < ratio * ratio * second) && (!cross || back == i) {
                        want.push([i as i32, j as i32]);
                    }
                }
                let got = pool.install(|| match_descriptors(&a, n1, &b, n2, ratio, cross));
                assert_eq!(got, want, "ratio {ratio} cross {cross} threads {threads}");
            }
        }
    }

    /// The vector and scalar kernels must agree on the *pair set*. They sum in
    /// different orders, so the distances can differ in the last bits; what has
    /// to hold is that the ordering those distances induce is the same.
    #[test]
    fn vector_and_scalar_agree() {
        let (n1, n2) = (257usize, 193usize);
        let a = rand_desc(n1, 999);
        let b = rand_desc(n2, 4242);
        for (ratio, cross) in [(0.8, true), (0.8, false), (1.0, true)] {
            let v = match_descriptors(&a, n1, &b, n2, ratio, cross);
            let s = match_descriptors_scalar(&a, n1, &b, n2, ratio, cross);
            assert_eq!(v, s, "ratio {ratio} cross {cross}");
        }
    }

    #[test]
    fn ratio_rejects_ambiguous_and_one_disables_it() {
        // Two identical train descriptors make every query ambiguous: best and
        // second-best are equal, so no ratio below 1 can accept them.
        let q = rand_desc(8, 7);
        let mut t = rand_desc(1, 7);
        t.extend_from_within(..);
        assert!(match_descriptors(&q, 8, &t, 2, 0.8, false).is_empty());
        // ...and a ratio of 1 must accept them rather than reject them all.
        assert_eq!(match_descriptors(&q, 8, &t, 2, 1.0, false).len(), 8);
    }

    #[test]
    fn single_train_survives_the_ratio_test() {
        let q = rand_desc(5, 3);
        let t = rand_desc(1, 88);
        let pairs = match_descriptors(&q, 5, &t, 1, 0.8, false);
        assert_eq!(pairs.len(), 5, "no second-best to divide by");
        assert!(pairs.iter().all(|[_, j]| *j == 0));
    }

    #[test]
    fn cross_check_rejects_non_mutual() {
        let mut q = rand_desc(1, 21);
        let mut near = q.clone();
        near[0] += 3.0;
        q.append(&mut near);
        let mut t = rand_desc(1, 21);
        t.extend_from_slice(&rand_desc(1, 555));
        let pairs = match_descriptors(&q, 2, &t, 2, 1.0, true);
        // Train 0 is nearest to both queries but can be mutual with only one.
        assert!(pairs.iter().filter(|[_, j]| *j == 0).count() <= 1);
    }

    #[test]
    fn degenerate_inputs() {
        let d = rand_desc(4, 1);
        assert!(match_descriptors(&d, 0, &d, 4, 0.8, true).is_empty());
        assert!(match_descriptors(&d, 4, &d, 0, 0.8, true).is_empty());
        // A short buffer is refused rather than read past.
        assert!(match_descriptors(&d[..DESCR_LEN], 4, &d, 4, 0.8, true).is_empty());
        let z = vec![0.0f32; 4 * DESCR_LEN];
        assert!(match_descriptors(&z, 4, &z, 4, 0.8, false).is_empty());
        assert_eq!(match_descriptors(&z, 4, &z, 4, 1.0, false).len(), 4);
    }
}
