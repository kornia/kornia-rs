//! Pairwise descriptor matching between frames.
//!
//! Matches feature descriptors between frame pairs within a sliding window and
//! emits the correspondences as [`TrackEdge`]s, ready for `kornia_calib`'s
//! `build_tracks` to chain into multi-view tracks.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use kornia_3d::pose::{ransac_fundamental, RansacParams};
use kornia_algebra::Vec2F64;
use kornia_calib::TrackEdge;
use kornia_imgproc::features::{match_orb_descriptors, sift_match_descriptors, OrbMatchConfig};
use rayon::prelude::*;

use crate::features::FrameFeatures;

/// Match every frame `i` against frames `i+1 ..= i+window` and collect the
/// resulting [`TrackEdge`]s.
///
/// The `kpt_a` / `kpt_b` fields are the indices into each frame's keypoint
/// list, which is what lets `build_tracks` chain matches transitively.
///
/// # Arguments
///
/// * `features` - Per-frame features, indexed by frame number.
/// * `window` - How many following frames each frame is matched against.
///   Frames `i` and `j` are only matched when `0 < j - i <= window`.
/// * `ratio` - Lowe's ratio-test threshold (e.g. `0.8`). Matches whose best
///   distance is not comfortably below the second-best are rejected.
/// * `orb_check_orientation` - Apply ORB-SLAM3's orientation-histogram
///   consistency filtering. Disable for orbit/turntable captures where the
///   camera rotates systematically (e.g. `swiss_knife.mp4`).
pub fn match_sequential_pairs(
    features: &[FrameFeatures],
    window: usize,
    ratio: f32,
    orb_check_orientation: bool,
) -> Vec<TrackEdge> {
    if features.len() < 2 {
        return Vec::new();
    }

    let mut edges = Vec::new();
    for i in 0..features.len() {
        let end = (i + 1 + window).min(features.len());
        for j in (i + 1)..end {
            let matches = match_pair(&features[i], &features[j], ratio, orb_check_orientation);
            for (kpt_a, kpt_b) in matches {
                edges.push(TrackEdge {
                    cam_a: i,
                    kpt_a: kpt_a as u32,
                    uv_a: keypoint_to_uv(features[i].keypoints[kpt_a]),
                    cam_b: j,
                    kpt_b: kpt_b as u32,
                    uv_b: keypoint_to_uv(features[j].keypoints[kpt_b]),
                });
            }
        }
    }
    edges
}

fn keypoint_to_uv(kp: [f32; 2]) -> Vec2F64 {
    // keypoints are stored as [col, row]; TrackEdge expects (x, y) = (col, row).
    Vec2F64::new(kp[0] as f64, kp[1] as f64)
}

/// Like [`match_sequential_pairs`], but matches the frame pairs concurrently
/// with rayon.
///
/// The set of `(i, j)` pairs is identical to the sequential version, so the
/// resulting edges are the same set (in a different order). Edges are grouped
/// per pair, so `build_tracks` still chains them identically.
///
/// # Arguments
///
/// * `features` - Per-frame features, indexed by frame number.
/// * `window` - How many following frames each frame is matched against.
/// * `ratio` - Lowe's ratio-test threshold.
/// * `orb_check_orientation` - Apply ORB-SLAM3 orientation-histogram
///   consistency filtering (see [`match_sequential_pairs`]).
pub fn match_pairs_parallel(
    features: &[FrameFeatures],
    window: usize,
    ratio: f32,
    orb_check_orientation: bool,
) -> Vec<TrackEdge> {
    if features.len() < 2 {
        return Vec::new();
    }

    // Enumerate the same (i, j) pairs the sequential version visits.
    let pairs: Vec<(usize, usize)> = (0..features.len())
        .flat_map(|i| {
            let end = (i + 1 + window).min(features.len());
            (i + 1..end).map(move |j| (i, j))
        })
        .collect();
    let total_pairs = pairs.len();
    eprintln!("  matching: {total_pairs} pairs in parallel");
    let done = AtomicUsize::new(0);

    pairs
        .par_iter()
        .flat_map(|&(i, j)| {
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(100) || n == total_pairs {
                eprintln!("  matching: {n}/{total_pairs} pairs");
            }
            let matches = match_pair(&features[i], &features[j], ratio, orb_check_orientation);
            let pair_edges: Vec<TrackEdge> = matches
                .into_iter()
                .map(|(kpt_a, kpt_b)| TrackEdge {
                    cam_a: i,
                    kpt_a: kpt_a as u32,
                    uv_a: keypoint_to_uv(features[i].keypoints[kpt_a]),
                    cam_b: j,
                    kpt_b: kpt_b as u32,
                    uv_b: keypoint_to_uv(features[j].keypoints[kpt_b]),
                })
                .collect();
            pair_edges.into_par_iter()
        })
        .collect()
}

/// Filter [`TrackEdge`]s per camera pair using epipolar-geometry RANSAC.
///
/// For each pair of cameras, fits a fundamental matrix to the raw descriptor
/// matches and keeps only the geometrically consistent (inlier) ones. This
/// rejects false matches that a descriptor ratio test alone cannot catch —
/// which is the dominant source of bad tracks for binary (ORB) descriptors.
///
/// # Arguments
///
/// * `edges` - Raw descriptor matches from a matcher.
/// * `threshold` - RANSAC inlier threshold in pixels.
/// * `min_inliers` - Minimum inlier count for a pair's fundamental matrix to
///   be trusted. Pairs with fewer raw matches than this (or than 8, the
///   minimum for the 8-point algorithm) are dropped entirely.
///
/// # Returns
///
/// The subset of `edges` that are inliers of a verified fundamental matrix.
pub fn verify_matches_geometrically(
    edges: &[TrackEdge],
    threshold: f64,
    min_inliers: usize,
) -> Vec<TrackEdge> {
    // Group edge indices by camera pair.
    let mut pairs: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        pairs.entry((e.cam_a, e.cam_b)).or_default().push(i);
    }

    let min_inliers = min_inliers.max(8);
    let mut verified = Vec::new();
    for indices in pairs.into_values() {
        if indices.len() < min_inliers {
            continue;
        }
        let x1: Vec<Vec2F64> = indices.iter().map(|&i| edges[i].uv_a).collect();
        let x2: Vec<Vec2F64> = indices.iter().map(|&i| edges[i].uv_b).collect();
        let params = RansacParams {
            max_iterations: 2000,
            threshold,
            min_inliers,
            random_seed: Some(0),
            refit: true,
        };
        let Ok(result) = ransac_fundamental(&x1, &x2, &params) else {
            continue;
        };
        for (&idx, &is_inlier) in indices.iter().zip(result.inliers.iter()) {
            if is_inlier {
                verified.push(edges[idx].clone());
            }
        }
    }
    verified
}

/// Append wide-baseline matches to `edges`: for every frame `i`, also match it
/// against `i+K, i+2K, ...` where `K` is the stride, skipping pairs already
/// covered by the sliding window (`j - i <= window`).
///
/// Wide-baseline pairs give large-parallax triangulation and long tracks,
/// which let more cameras register via PnP. They are harder to match (more
/// outliers), so run `--geo-verify` after enabling this.
///
/// Returns the number of edges appended.
pub fn append_wide_baseline_edges(
    features: &[FrameFeatures],
    window: usize,
    stride: usize,
    ratio: f32,
    orb_check_orientation: bool,
    edges: &mut Vec<TrackEdge>,
) -> usize {
    if stride == 0 || features.len() <= window + 1 {
        return 0;
    }
    let mut added = 0;
    for i in 0..features.len() {
        // First j > i + window with (j - i) % stride == 0.
        let start = i + window + 1;
        let first = start + ((stride - (start - i) % stride) % stride);
        for j in (first..features.len()).step_by(stride) {
            let matches = match_pair(&features[i], &features[j], ratio, orb_check_orientation);
            for (kpt_a, kpt_b) in matches {
                edges.push(TrackEdge {
                    cam_a: i,
                    kpt_a: kpt_a as u32,
                    uv_a: keypoint_to_uv(features[i].keypoints[kpt_a]),
                    cam_b: j,
                    kpt_b: kpt_b as u32,
                    uv_b: keypoint_to_uv(features[j].keypoints[kpt_b]),
                });
                added += 1;
            }
        }
    }
    added
}

/// Match a single frame pair; returns `(idx_in_a, idx_in_b)` pairs.
fn match_pair(
    a: &FrameFeatures,
    b: &FrameFeatures,
    ratio: f32,
    orb_check_orientation: bool,
) -> Vec<(usize, usize)> {
    // ORB path: ORB-SLAM3 style matcher with optional orientation-histogram
    // filtering. Filtering assumes the camera rotates (features rotate
    // together); disable it for orbit/turntable captures where that breaks.
    if let (Some(d1), Some(o1), Some(d2), Some(o2)) = (
        &a.descriptors_orb,
        &a.orientations_orb,
        &b.descriptors_orb,
        &b.orientations_orb,
    ) {
        let config = OrbMatchConfig {
            nn_ratio: ratio,
            check_orientation: orb_check_orientation,
            ..OrbMatchConfig::default()
        };
        return match_orb_descriptors(o1, d1, o2, d2, config);
    }
    // SIFT path: L2 matching with Lowe's ratio test.
    if let (Some(d1), Some(d2)) = (&a.descriptors_sift, &b.descriptors_sift) {
        return sift_match_descriptors(d1, a.n_keypoints(), d2, b.n_keypoints(), ratio, true)
            .into_iter()
            .map(|[q, t]| (q as usize, t as usize))
            .collect();
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build ORB-style features whose descriptors carry a unique id in the
    /// first byte (rest zero). Descriptors with the same id match exactly.
    fn orb_features(keypoints: Vec<[f32; 2]>, ids: &[u8]) -> FrameFeatures {
        let descriptors = ids
            .iter()
            .map(|&id| {
                let mut d = [0u8; 32];
                d[0] = id;
                d
            })
            .collect();
        FrameFeatures {
            keypoints,
            descriptors_orb: Some(descriptors),
            orientations_orb: Some(vec![0.0; ids.len()]),
            descriptors_sift: None,
        }
    }

    /// Build SIFT-style features: descriptor `i` is 128 floats all equal to `ids[i]`.
    fn sift_features(keypoints: Vec<[f32; 2]>, ids: &[u8]) -> FrameFeatures {
        let mut descriptors = Vec::new();
        for &id in ids {
            descriptors.extend(std::iter::repeat_n(id as f32, 128));
        }
        FrameFeatures {
            keypoints,
            descriptors_orb: None,
            orientations_orb: None,
            descriptors_sift: Some(descriptors),
        }
    }

    fn keypoints(n: usize) -> Vec<[f32; 2]> {
        (0..n).map(|i| [10.0 + i as f32 * 20.0, 30.0]).collect()
    }

    fn pair_set(edges: &[TrackEdge]) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> = edges.iter().map(|e| (e.cam_a, e.cam_b)).collect();
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    #[test]
    fn matches_identical_orb_descriptors_between_consecutive_frames() {
        let a = orb_features(keypoints(3), &[1, 2, 4]);
        let b = orb_features(keypoints(3), &[1, 2, 4]);
        let edges = match_sequential_pairs(&[a, b], 1, 0.8, true);

        assert_eq!(edges.len(), 3);
        let mut kpts: Vec<(u32, u32)> = edges.iter().map(|e| (e.kpt_a, e.kpt_b)).collect();
        kpts.sort_unstable();
        assert_eq!(kpts, vec![(0, 0), (1, 1), (2, 2)]);
        assert!(edges.iter().all(|e| e.cam_a == 0 && e.cam_b == 1));
        // uv_* should be the keypoint coordinates of the matching indices.
        assert_eq!(edges[0].uv_a.x, 10.0);
        assert_eq!(edges[0].uv_b.x, 10.0);
    }

    #[test]
    fn matches_identical_sift_descriptors_between_consecutive_frames() {
        let a = sift_features(keypoints(2), &[1, 2]);
        let b = sift_features(keypoints(2), &[1, 2]);
        let edges = match_sequential_pairs(&[a, b], 1, 0.8, true);

        assert_eq!(edges.len(), 2);
        let mut kpts: Vec<(u32, u32)> = edges.iter().map(|e| (e.kpt_a, e.kpt_b)).collect();
        kpts.sort_unstable();
        assert_eq!(kpts, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn window_size_controls_which_pairs_are_matched() {
        let f0 = orb_features(keypoints(3), &[1, 2, 4]);
        let f1 = orb_features(keypoints(3), &[1, 2, 4]);
        let f2 = orb_features(keypoints(3), &[1, 2, 4]);

        let edges_w1 = match_sequential_pairs(&[f0.clone(), f1.clone(), f2.clone()], 1, 0.8, true);
        assert_eq!(pair_set(&edges_w1), vec![(0, 1), (1, 2)]);

        let edges_w2 = match_sequential_pairs(&[f0, f1, f2], 2, 0.8, true);
        assert_eq!(pair_set(&edges_w2), vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn returns_empty_when_frames_have_no_features() {
        let empty = FrameFeatures {
            keypoints: Vec::new(),
            descriptors_orb: Some(Vec::new()),
            orientations_orb: Some(Vec::new()),
            descriptors_sift: None,
        };
        let edges = match_sequential_pairs(&[empty.clone(), empty], 1, 0.8, true);
        assert!(edges.is_empty());
    }

    #[test]
    fn returns_empty_for_single_frame() {
        let f = orb_features(keypoints(3), &[1, 2, 4]);
        let edges = match_sequential_pairs(&[f], 5, 0.8, true);
        assert!(edges.is_empty());
    }

    #[test]
    fn ratio_test_filters_ambiguous_matches() {
        // Keypoint A0 is equidistant from B0 and B1 (both 2 bits away), so the
        // best/second-best ratio is 1.0 — rejected by a strict ratio but kept
        // by a loose one.
        let a = orb_features(vec![[0.0, 0.0]], &[1]);
        let b = orb_features(vec![[0.0, 0.0], [0.0, 0.0]], &[2, 4]);

        let strict = match_sequential_pairs(&[a.clone(), b.clone()], 1, 0.8, true);
        assert!(strict.is_empty(), "ambiguous match must be rejected");

        let loose = match_sequential_pairs(&[a, b], 1, 1.5, true);
        assert_eq!(loose.len(), 1);
        assert_eq!((loose[0].kpt_a, loose[0].kpt_b), (0, 0));
    }

    #[test]
    fn geometric_verification_keeps_inliers_drops_outliers() {
        use crate::reconstruction::make_camera;
        use crate::test_util;
        use kornia_3d::pose::Pose3d;
        use kornia_algebra::Vec3F64;

        // Two cameras with a proper baseline and converging rotation, viewing a
        // plane of 3D points — a well-conditioned epipolar geometry.
        let k = make_camera(500.0, 500.0, 320.0, 240.0);
        let pose_a = Pose3d::IDENTITY;
        let pose_b = Pose3d::new(test_util::rot(0.4, 0.05), Vec3F64::new(-0.6, 0.0, 0.1));

        let mut edges = Vec::new();
        // 40 inliers: an 8x5 grid of points with depth variation (NOT planar,
        // so the fundamental matrix is well-constrained), projected into both views.
        let mut proj_a = Vec::new();
        let mut proj_b = Vec::new();
        for i in 0..8 {
            for j in 0..5 {
                let z = 1.4 + 0.3 * ((i * 5 + j) % 3) as f64;
                let p = Vec3F64::new(-0.4 + 0.1 * i as f64, -0.3 + 0.15 * j as f64, z);
                proj_a.push(test_util::project(p, &pose_a, &k));
                proj_b.push(test_util::project(p, &pose_b, &k));
                let idx = (i * 5 + j) as u32;
                edges.push(TrackEdge {
                    cam_a: 0,
                    kpt_a: idx,
                    uv_a: proj_a[idx as usize],
                    cam_b: 1,
                    kpt_b: idx,
                    uv_b: proj_b[idx as usize],
                });
            }
        }
        // 10 outliers: the correct correspondence, corrupted by a large pixel
        // offset — far outside any epipolar constraint.
        for i in 0..10 {
            edges.push(TrackEdge {
                cam_a: 0,
                kpt_a: (40 + i) as u32,
                uv_a: proj_a[i],
                cam_b: 1,
                kpt_b: (40 + i) as u32,
                uv_b: Vec2F64::new(proj_b[i].x + 400.0, proj_b[i].y + 300.0),
            });
        }

        let verified = verify_matches_geometrically(&edges, 3.0, 8);
        assert_eq!(verified.len(), 40, "all inliers kept, all outliers dropped");
        assert!(verified.iter().all(|e| e.kpt_a < 40));
    }

    #[test]
    fn geometric_verification_drops_pairs_with_few_matches() {
        // Fewer than 8 raw matches: not enough for the 8-point algorithm.
        let mut edges = Vec::new();
        for i in 0..5 {
            edges.push(TrackEdge {
                cam_a: 0,
                kpt_a: i as u32,
                uv_a: Vec2F64::new(i as f64, 0.0),
                cam_b: 1,
                kpt_b: i as u32,
                uv_b: Vec2F64::new(i as f64, 0.0),
            });
        }
        assert!(verify_matches_geometrically(&edges, 3.0, 8).is_empty());
    }

    #[test]
    fn wide_baseline_skips_window_pairs_and_steps_by_stride() {
        // 5 frames, all sharing the same 2 descriptors, so every pair matches.
        let f = || orb_features(keypoints(2), &[1, 2]);
        let frames: Vec<FrameFeatures> = (0..5).map(|_| f()).collect();

        let mut edges = Vec::new();
        let added = append_wide_baseline_edges(&frames, 1, 2, 0.8, true, &mut edges);
        assert_eq!(added, 4 * 2, "pairs (0,2),(0,4),(1,3),(2,4) x 2 matches");
        let pairs: Vec<(usize, usize)> = edges.iter().map(|e| (e.cam_a, e.cam_b)).collect();
        for (a, b) in &pairs {
            assert!(
                b - a > 1 && (b - a) % 2 == 0,
                "pair ({a},{b}) must be wide-baseline (dist > window, multiple of stride)"
            );
        }
    }

    #[test]
    fn wide_baseline_zero_stride_is_noop() {
        let f = orb_features(keypoints(2), &[1, 2]);
        let frames = vec![f.clone(), f];
        let mut edges = Vec::new();
        assert_eq!(
            append_wide_baseline_edges(&frames, 1, 0, 0.8, true, &mut edges),
            0
        );
        assert!(edges.is_empty());
    }
}
