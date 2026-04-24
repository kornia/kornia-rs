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

/// Borrowed view over ORB features — descriptors + keypoint positions +
/// per-keypoint octaves, all as parallel slices.
///
/// Used by [`match_orb_by_projection`] on both sides of the match (predicted
/// projections from a map or previous frame, and observed keypoints in the
/// current frame). Collapses what would otherwise be six positional slice
/// arguments on the matcher into two view structs.
///
/// For the "predicted" side, `keypoints_xy` are the projections of map points
/// into the current frame (not the originally-detected keypoints); octaves
/// are the octave the map point was originally detected at, which drives the
/// scale-aware search radius.
#[derive(Debug, Copy, Clone)]
pub struct OrbFeaturesView<'a, const N: usize> {
    /// Binary descriptors (one row per feature, `N` bytes each).
    pub descriptors: &'a [[u8; N]],
    /// Feature positions as `[col, row]` in image pixels.
    pub keypoints_xy: &'a [[f32; 2]],
    /// Pyramid octave per feature (0 = full resolution, higher = coarser).
    pub octaves: &'a [u8],
}

impl<const N: usize> OrbFeaturesView<'_, N> {
    /// Number of features in the view. All three slices must have this
    /// length; this is asserted by the matcher.
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// `true` when the view contains zero features.
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }
}

/// Configuration for [`match_orb_by_projection`].
///
/// Mirrors the knobs exposed by ORB-SLAM3's `ORBmatcher::SearchByProjection`.
#[derive(Debug, Clone)]
pub struct ByProjectionConfig {
    /// Base search radius in pixels at octave 0. The effective radius for
    /// a candidate predicted at octave `o` is `base_radius * scale_factors[o]`
    /// — higher octaves (coarser pyramid) project back to larger pixel
    /// uncertainty at full resolution.
    pub base_radius: f32,
    /// Per-octave scale multiplier, typically `downscale.powi(o)` for each
    /// octave `o`. Caller provides this precomputed; ORB-SLAM3 stores it as
    /// `mvScaleFactors` on the frame.
    pub scale_factors: Vec<f32>,
    /// Maximum allowed octave difference between predicted and candidate
    /// keypoint (default 1). BRIEF is scale-variant so cross-octave matches
    /// are unreliable.
    pub max_octave_diff: u8,
    /// Upper bound on Hamming distance for an accepted match (default 50 for
    /// 256-bit BRIEF, matching ORB-SLAM3's `TH_HIGH` / `TH_LOW` tuning).
    pub max_distance: u32,
    /// Lowe's ratio test threshold (default 0.75). A best/second-best ratio
    /// above this rejects the match. Set to >=1.0 to disable.
    pub max_ratio: f32,
}

impl Default for ByProjectionConfig {
    fn default() -> Self {
        Self {
            base_radius: 15.0,
            scale_factors: Vec::new(),
            max_octave_diff: 1,
            max_distance: 50,
            max_ratio: 0.75,
        }
    }
}

/// Scale-aware guided Hamming matcher — ORB-SLAM3's `SearchByProjection`.
///
/// For each predicted feature (from projecting a map point or tracked keypoint
/// into the current frame), searches observed keypoints inside a scale-
/// aware radius and with a compatible octave, and returns the best Hamming
/// match subject to Lowe's ratio test and a distance threshold.
///
/// This is the hot path for ORB-SLAM-family tracking — it replaces the O(M·N)
/// brute-force matcher with a spatially-gated one, while still letting the
/// descriptor distance break ties inside each gate.
///
/// # Arguments
///
/// * `predicted` - View over predicted features. `keypoints_xy` here are the
///   projections of map points / prior-frame features into the current
///   frame; `octaves` are the octaves at which those features were
///   originally detected (stored on the map point).
/// * `observed` - View over currently-detected features in the target frame.
/// * `cfg` - [`ByProjectionConfig`] — search radius, octave tolerance, ratio
///   and distance thresholds.
///
/// # Returns
///
/// Vector of `(i, j)` index pairs into `predicted` and `observed`
/// respectively. Order is deterministic (sorted by `i`).
pub fn match_orb_by_projection<const N: usize>(
    predicted: OrbFeaturesView<'_, N>,
    observed: OrbFeaturesView<'_, N>,
    cfg: &ByProjectionConfig,
) -> Vec<(usize, usize)> {
    use rayon::prelude::*;

    assert_eq!(predicted.descriptors.len(), predicted.keypoints_xy.len());
    assert_eq!(predicted.descriptors.len(), predicted.octaves.len());
    assert_eq!(observed.descriptors.len(), observed.keypoints_xy.len());
    assert_eq!(observed.descriptors.len(), observed.octaves.len());

    if predicted.is_empty() || observed.is_empty() {
        return Vec::new();
    }

    // `Default` leaves scale_factors empty on purpose — the caller must pass
    // the per-octave scale from its pyramid. Silently defaulting to 1.0 would
    // make the matcher non-scale-aware with no signal.
    assert!(
        !cfg.scale_factors.is_empty(),
        "ByProjectionConfig.scale_factors must be populated",
    );

    let base_r = cfg.base_radius;
    let max_oct_diff = cfg.max_octave_diff as i32;
    let max_dist = cfg.max_distance;
    let ratio = cfg.max_ratio;
    let scale_factors = &cfg.scale_factors;

    // Per-predicted-feature forward pass: find best/second-best observed kp
    // inside the scale-aware gate. Parallelize over predictions.
    let matches: Vec<Option<(usize, usize)>> = predicted
        .descriptors
        .par_iter()
        .enumerate()
        .map(|(i, d_pred)| {
            let pred_oct = predicted.octaves[i];
            let scale = scale_factors
                .get(pred_oct as usize)
                .copied()
                .unwrap_or(1.0);
            let radius = base_r * scale;
            let radius_sq = radius * radius;
            let [px, py] = predicted.keypoints_xy[i];

            let mut best_j = 0usize;
            let mut best_dist = u32::MAX;
            let mut second_dist = u32::MAX;

            for (j, d_curr) in observed.descriptors.iter().enumerate() {
                // Octave gate: BRIEF is scale-variant so reject cross-octave.
                let oct_diff = (observed.octaves[j] as i32 - pred_oct as i32).abs();
                if oct_diff > max_oct_diff {
                    continue;
                }
                // Spatial gate: radius grows with predicted octave.
                let [cx, cy] = observed.keypoints_xy[j];
                let dx = cx - px;
                let dy = cy - py;
                if dx * dx + dy * dy > radius_sq {
                    continue;
                }

                let dist = hamming_distance(d_pred, d_curr);
                if dist < best_dist {
                    second_dist = best_dist;
                    best_dist = dist;
                    best_j = j;
                } else if dist < second_dist {
                    second_dist = dist;
                }
            }

            if best_dist > max_dist {
                return None;
            }
            if ratio < 1.0 && second_dist != u32::MAX {
                let denom = if second_dist == 0 {
                    f32::EPSILON
                } else {
                    second_dist as f32
                };
                if best_dist as f32 / denom >= ratio {
                    return None;
                }
            }
            Some((i, best_j))
        })
        .collect();

    matches.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc_from_bit(bit: u8) -> [u8; 32] {
        let mut d = [0u8; 32];
        d[0] = bit;
        d
    }

    #[test]
    fn by_projection_respects_radius_and_octave() {
        // Three predicted features at different octaves.
        let d_pred = vec![desc_from_bit(0x01), desc_from_bit(0x02), desc_from_bit(0x04)];
        let pred_xy = vec![[100.0, 100.0], [50.0, 50.0], [200.0, 200.0]];
        let pred_oct = vec![0u8, 1, 2];

        // Current-frame candidates.
        // j=0: identical to pred[0], at (101,100), oct 0 — should match pred[0].
        // j=1: matches pred[1] descriptor at (55,51) oct 1 — within scaled radius.
        // j=2: matches pred[2] descriptor but at oct 0 (cross-octave) — must be rejected.
        // j=3: matches pred[2] descriptor at oct 2 but 100px away — radius reject.
        // j=4: matches pred[2] descriptor at (198,201) oct 2 — should match pred[2].
        let d_curr = vec![
            desc_from_bit(0x01),
            desc_from_bit(0x02),
            desc_from_bit(0x04),
            desc_from_bit(0x04),
            desc_from_bit(0x04),
        ];
        let curr_xy = vec![
            [101.0, 100.0],
            [55.0, 51.0],
            [200.0, 200.0],
            [100.0, 100.0],
            [198.0, 201.0],
        ];
        let curr_oct = vec![0u8, 1, 0, 2, 2];

        let cfg = ByProjectionConfig {
            base_radius: 10.0,
            scale_factors: vec![1.0, 1.2, 1.44],
            max_octave_diff: 1,
            max_distance: 256,
            max_ratio: 1.0, // disable ratio test — we want the spatial/octave gates only
        };

        let predicted = OrbFeaturesView {
            descriptors: &d_pred,
            keypoints_xy: &pred_xy,
            octaves: &pred_oct,
        };
        let observed = OrbFeaturesView {
            descriptors: &d_curr,
            keypoints_xy: &curr_xy,
            octaves: &curr_oct,
        };

        let mut matches = match_orb_by_projection(predicted, observed, &cfg);
        matches.sort();

        assert_eq!(matches, vec![(0, 0), (1, 1), (2, 4)]);
    }

    #[test]
    fn by_projection_empty_inputs() {
        let empty_desc: Vec<[u8; 32]> = vec![];
        let empty_xy: Vec<[f32; 2]> = vec![];
        let empty_oct: Vec<u8> = vec![];
        let cfg = ByProjectionConfig::default();

        let view = OrbFeaturesView {
            descriptors: &empty_desc,
            keypoints_xy: &empty_xy,
            octaves: &empty_oct,
        };
        let m = match_orb_by_projection(view, view, &cfg);
        assert!(m.is_empty());
    }
}
