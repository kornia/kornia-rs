/// Hamming distance between two fixed-size byte descriptors.
#[inline]
pub fn hamming_distance<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    // NEON specialization for ORB's 32-byte descriptor: two `veorq_u8` +
    // `vcntq_u8` + `vaddvq_u8` cover the entire descriptor in ~6 cycles —
    // the scalar byte-at-a-time XOR+popcount loop takes ~60+.
    #[cfg(target_arch = "aarch64")]
    if N == 32 {
        use std::arch::aarch64::*;
        unsafe {
            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let x0 = veorq_u8(vld1q_u8(ap), vld1q_u8(bp));
            let x1 = veorq_u8(vld1q_u8(ap.add(16)), vld1q_u8(bp.add(16)));
            let p0 = vcntq_u8(x0);
            let p1 = vcntq_u8(x1);
            return (vaddvq_u8(p0) as u32) + (vaddvq_u8(p1) as u32);
        }
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

/// Match binary descriptors using brute-force Hamming distance.
///
/// For each descriptor in `descriptors1`, finds the nearest neighbor in `descriptors2`.
/// Optionally filters matches by maximum distance, cross-check, and Lowe's ratio test.
///
/// # Arguments
///
/// * `descriptors1` - First set of N-byte binary descriptors.
/// * `descriptors2` - Second set of N-byte binary descriptors.
/// * `max_distance` - If set, discard matches with Hamming distance above this threshold.
/// * `cross_check` - If true, keep only mutual nearest neighbors.
/// * `max_ratio` - If set, apply Lowe's ratio test (best / second-best < ratio).
///
/// # Returns
///
/// Vector of `(i, j)` index pairs into `descriptors1` and `descriptors2`.
pub fn match_descriptors<const N: usize>(
    descriptors1: &[[u8; N]],
    descriptors2: &[[u8; N]],
    max_distance: Option<u32>,
    cross_check: bool,
    max_ratio: Option<f32>,
) -> Vec<(usize, usize)> {
    if descriptors1.is_empty() || descriptors2.is_empty() {
        return vec![];
    }

    // Forward pass: for each desc1[i], find best and second-best in desc2.
    // Each row is independent — parallelize across descriptors1.
    use rayon::prelude::*;
    let fwd: Vec<(usize, u32, u32)> = descriptors1
        .par_iter()
        .map(|d1| {
            let mut best_j = 0usize;
            let mut best_dist = u32::MAX;
            let mut second_dist = u32::MAX;
            for (j, d2) in descriptors2.iter().enumerate() {
                let dist = hamming_distance(d1, d2);
                if dist < best_dist {
                    second_dist = best_dist;
                    best_dist = dist;
                    best_j = j;
                } else if dist < second_dist {
                    second_dist = dist;
                }
            }
            (best_j, best_dist, second_dist)
        })
        .collect();

    // Reverse pass (only if cross-check): for each desc2[j], find best match in desc1.
    // Parallelize across descriptors2 to mirror the forward pass.
    let rev_best_i = if cross_check {
        let rev: Vec<usize> = descriptors2
            .par_iter()
            .map(|d2| {
                let mut best_i = 0usize;
                let mut best_dist = u32::MAX;
                for (i, d1) in descriptors1.iter().enumerate() {
                    let dist = hamming_distance(d1, d2);
                    if dist < best_dist {
                        best_dist = dist;
                        best_i = i;
                    }
                }
                best_i
            })
            .collect();
        Some(rev)
    } else {
        None
    };

    // Build matches applying all filters in one pass.
    let mut matches = Vec::new();
    for (i, &(j, best_dist, second_dist)) in fwd.iter().enumerate() {
        if let Some(max_dist) = max_distance {
            if best_dist > max_dist {
                continue;
            }
        }

        if let Some(ref rev) = rev_best_i {
            if rev[j] != i {
                continue;
            }
        }

        if let Some(ratio) = max_ratio {
            if ratio < 1.0 {
                let denom = if second_dist == 0 {
                    f32::EPSILON
                } else {
                    second_dist as f32
                };
                if best_dist as f32 / denom >= ratio {
                    continue;
                }
            }
        }

        matches.push((i, j));
    }

    matches
}
