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
//! Each query keeps its best and second-best in registers while the train set
//! streams past. Materialising the `n1 x n2` matrix would be tens of megabytes
//! of traffic for a result that is two small arrays.

use rayon::prelude::*;

use super::descriptor::DESCR_LEN;

/// Squared L2 between two descriptors, four lanes at a time.
///
/// Four independent accumulators, not one: a single accumulator makes this a
/// serial `fma` chain 32 long, which at ~4-cycle latency is latency-bound rather
/// than throughput-bound. Same reason the blur uses four.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(a.len(), DESCR_LEN);
    debug_assert_eq!(b.len(), DESCR_LEN);
    // SAFETY: both slices are `DESCR_LEN` long and `DESCR_LEN % 16 == 0`.
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
/// Off aarch64 this is [`l2_sq_scalar`]; the vector form is the aarch64 twin
/// above.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    l2_sq_scalar(a, b)
}

/// Squared L2, scalar. The portable fallback, and the reference the vector form
/// is checked against.
///
/// Kept public and always compiled, not `#[cfg]`-ed away: a fallback that only
/// builds on the platforms that do not use it is a fallback nobody tests.
#[inline]
pub fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        s = d.mul_add(d, s);
    }
    s
}

/// Nearest and second-nearest train descriptor for each query.
///
/// Returns `(index, best, second)` per query, as squared distances.
fn best2(q: &[f32], nq: usize, t: &[f32], nt: usize, scalar: bool) -> Vec<(i32, f32, f32)> {
    (0..nq)
        .into_par_iter()
        .map(|i| {
            let qd = &q[i * DESCR_LEN..(i + 1) * DESCR_LEN];
            let (mut best, mut second, mut best_j) = (f32::MAX, f32::MAX, -1i32);
            for j in 0..nt {
                let td = &t[j * DESCR_LEN..(j + 1) * DESCR_LEN];
                let d = if scalar {
                    l2_sq_scalar(qd, td)
                } else {
                    l2_sq(qd, td)
                };
                if d < best {
                    second = best;
                    best = d;
                    best_j = j as i32;
                } else if d < second {
                    second = d;
                }
            }
            (best_j, best, second)
        })
        .collect()
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
    let fwd = best2(d1, n1, d2, n2, scalar);
    // The reverse scan is only needed for the mutual check.
    let rev = if cross_check {
        best2(d2, n2, d1, n1, scalar)
    } else {
        Vec::new()
    };
    let ratio2 = ratio * ratio;

    let mut out = Vec::new();
    for (i, &(j, d_best, d_second)) in fwd.iter().enumerate() {
        if j < 0 {
            continue;
        }
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
        if cross_check && rev[j as usize].0 != i as i32 {
            continue;
        }
        out.push([i as i32, j]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
