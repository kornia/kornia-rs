/// Hamming distance between two fixed-size byte descriptors.
#[inline]
fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
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
    let m = descriptors1.len();
    let n = descriptors2.len();
    if m == 0 || n == 0 {
        return vec![];
    }

    // Forward pass: for each desc1[i], find best and second-best match in desc2.
    let mut fwd_best_j = vec![0usize; m];
    let mut fwd_best_dist = vec![u32::MAX; m];
    let mut fwd_second_dist = vec![u32::MAX; m];

    for (i, d1) in descriptors1.iter().enumerate() {
        for (j, d2) in descriptors2.iter().enumerate() {
            let dist = hamming_distance(d1, d2);
            if dist < fwd_best_dist[i] {
                fwd_second_dist[i] = fwd_best_dist[i];
                fwd_best_dist[i] = dist;
                fwd_best_j[i] = j;
            } else if dist < fwd_second_dist[i] {
                fwd_second_dist[i] = dist;
            }
        }
    }

    // Reverse pass (only if cross-check): for each desc2[j], find best match in desc1.
    let rev_best_i = if cross_check {
        let mut rev = vec![0usize; n];
        let mut rev_dist = vec![u32::MAX; n];
        for (i, d1) in descriptors1.iter().enumerate() {
            for (j, d2) in descriptors2.iter().enumerate() {
                let dist = hamming_distance(d1, d2);
                if dist < rev_dist[j] {
                    rev_dist[j] = dist;
                    rev[j] = i;
                }
            }
        }
        Some(rev)
    } else {
        None
    };

    // Build matches applying all filters in one pass.
    let mut matches = Vec::new();
    for i in 0..m {
        let j = fwd_best_j[i];
        let best_dist = fwd_best_dist[i];

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
                let second = fwd_second_dist[i];
                let denom = if second == 0 {
                    f32::EPSILON
                } else {
                    second as f32
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
