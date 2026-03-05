/// Configuration for ORB-style descriptor matching.
#[derive(Clone, Copy, Debug)]
pub struct OrbMatchConfig {
    /// Nearest-neighbor ratio for accepting matches.
    pub nn_ratio: f32,
    /// Maximum Hamming distance to accept a match.
    pub th_low: u32,
    /// Whether to apply orientation histogram consistency filtering.
    pub check_orientation: bool,
    /// Orientation histogram length.
    pub histo_length: usize,
}

impl Default for OrbMatchConfig {
    fn default() -> Self {
        Self {
            nn_ratio: 0.6,
            th_low: 50,
            check_orientation: true,
            histo_length: 30,
        }
    }
}

/// Match ORB descriptors using ORB-SLAM3-style logic (ratio test + optional orientation histogram).
/// Descriptors must be packed 32-byte arrays (256 bits).
pub fn match_orb_descriptors(
    angles1: &[f32],
    desc1: &[[u8; 32]],
    angles2: &[f32],
    desc2: &[[u8; 32]],
    config: OrbMatchConfig,
) -> Vec<(usize, usize)> {
    assert_eq!(angles1.len(), desc1.len(), "angles1/desc1 length mismatch");
    assert_eq!(angles2.len(), desc2.len(), "angles2/desc2 length mismatch");

    if desc1.is_empty() || desc2.is_empty() {
        return Vec::new();
    }

    let mut matches: Vec<Option<usize>> = vec![None; desc1.len()];
    let mut rot_hist: Vec<Vec<usize>> = vec![Vec::new(); config.histo_length];
    let factor = 1.0f32 / config.histo_length as f32;

    for i in 0..desc1.len() {
        let mut best = u32::MAX;
        let mut second = u32::MAX;
        let mut best_j = 0usize;

        for (j, d2) in desc2.iter().enumerate() {
            let d = hamming_distance(&desc1[i], d2);
            if d < best {
                second = best;
                best = d;
                best_j = j;
            } else if d < second {
                second = d;
            }
        }

        if best <= config.th_low && (best as f32) < config.nn_ratio * (second as f32) {
            let j = best_j;
            matches[i] = Some(j);

            if config.check_orientation {
                let mut rot = angles1[i].to_degrees() - angles2[j].to_degrees();
                if rot < 0.0 {
                    rot += 360.0;
                }
                let mut bin = (rot * factor).round() as usize;
                if bin == config.histo_length {
                    bin = 0;
                }
                rot_hist[bin].push(i);
            }
        }
    }

    if config.check_orientation {
        let (ind1, ind2, ind3) = three_maxima(&rot_hist);
        for (bin_idx, bin) in rot_hist.iter().enumerate() {
            if Some(bin_idx) == ind1 || Some(bin_idx) == ind2 || Some(bin_idx) == ind3 {
                continue;
            }
            for &i in bin {
                matches[i] = None;
            }
        }
    }

    let mut result = Vec::new();
    for (i, m) in matches.into_iter().enumerate() {
        if let Some(j) = m {
            result.push((i, j));
        }
    }
    result
}

/// Compute Hamming distance between two 32-byte packed descriptors.
/// Uses popcount on each byte and sums the results (max distance = 256).
#[inline]
fn hamming_distance(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

/// Return the indices of the three largest histogram bins, discarding any that
/// fall below 10% of the largest bin.
fn three_maxima(histo: &[Vec<usize>]) -> (Option<usize>, Option<usize>, Option<usize>) {
    let mut ind1 = None;
    let mut ind2 = None;
    let mut ind3 = None;
    let mut max1 = 0usize;
    let mut max2 = 0usize;
    let mut max3 = 0usize;

    for (i, bin) in histo.iter().enumerate() {
        let s = bin.len();
        if s > max1 {
            max3 = max2;
            ind3 = ind2;
            max2 = max1;
            ind2 = ind1;
            max1 = s;
            ind1 = Some(i);
        } else if s > max2 {
            max3 = max2;
            ind3 = ind2;
            max2 = s;
            ind2 = Some(i);
        } else if s > max3 {
            max3 = s;
            ind3 = Some(i);
        }
    }

    if max2 < (max1 as f32 * 0.1) as usize {
        ind2 = None;
        ind3 = None;
    } else if max3 < (max1 as f32 * 0.1) as usize {
        ind3 = None;
    }

    (ind1, ind2, ind3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orb_match_orientation_histogram() {
        // Create 21 distinct descriptors: 20 inliers + 1 outlier.
        // Need enough inliers so that 10% threshold (20 * 0.1 = 2) filters the single outlier.
        const N: usize = 21;
        let mut desc1 = vec![[0u8; 32]; N];
        let mut desc2 = vec![[0u8; 32]; N];

        // Set different bytes to make descriptors distinguishable.
        for i in 0..N {
            desc1[i][i % 32] ^= 1 << (i % 8);
            desc1[i][(i + 1) % 32] = (i * 17) as u8; // Make them unique
            desc2[i][i % 32] ^= 1 << (i % 8);
            desc2[i][(i + 1) % 32] = (i * 17) as u8;
        }

        // Angles: first 20 in same bin, last 1 in different bin.
        let angles1 = vec![0.0f32; N];
        let mut angles2 = vec![0.0f32; N];
        angles2[N - 1] = 180.0_f32.to_radians(); // Outlier

        let config = OrbMatchConfig {
            nn_ratio: 0.9,
            th_low: 50,
            check_orientation: true,
            histo_length: 30,
        };

        let matches = match_orb_descriptors(&angles1, &desc1, &angles2, &desc2, config);
        // The outlier bin (1 match) should be removed because 1 < 20 * 0.1 = 2.
        // Expect 20 matches.
        assert_eq!(matches.len(), N - 1);
    }

    #[test]
    fn test_hamming_distance() {
        let a = [0u8; 32];
        let b = [0u8; 32];
        assert_eq!(hamming_distance(&a, &b), 0);

        let mut c = [0u8; 32];
        c[0] = 0xFF; // 8 bits set
        assert_eq!(hamming_distance(&a, &c), 8);

        let d = [0xFFu8; 32]; // all 256 bits set
        assert_eq!(hamming_distance(&a, &d), 256);
    }
}
