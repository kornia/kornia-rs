//! Pairwise descriptor matching between frames.
//!
//! Matches feature descriptors between frame pairs within a sliding window and
//! emits the correspondences as [`TrackEdge`]s, ready for `kornia_calib`'s
//! `build_tracks` to chain into multi-view tracks.

use kornia_algebra::Vec2F64;
use kornia_calib::TrackEdge;
use kornia_imgproc::features::{match_descriptors, sift_match_descriptors};

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
pub fn match_sequential_pairs(
    features: &[FrameFeatures],
    window: usize,
    ratio: f32,
) -> Vec<TrackEdge> {
    if features.len() < 2 {
        return Vec::new();
    }

    let mut edges = Vec::new();
    for i in 0..features.len() {
        let end = (i + 1 + window).min(features.len());
        for j in (i + 1)..end {
            let matches = match_pair(&features[i], &features[j], ratio);
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

/// Match a single frame pair; returns `(idx_in_a, idx_in_b)` pairs.
fn match_pair(a: &FrameFeatures, b: &FrameFeatures, ratio: f32) -> Vec<(usize, usize)> {
    match (&a.descriptors_orb, &b.descriptors_orb) {
        (Some(d1), Some(d2)) => match_descriptors::<32>(d1, d2, None, true, Some(ratio)),
        _ => match (&a.descriptors_sift, &b.descriptors_sift) {
            (Some(d1), Some(d2)) => {
                sift_match_descriptors(d1, a.n_keypoints(), d2, b.n_keypoints(), ratio, true)
                    .into_iter()
                    .map(|[q, t]| (q as usize, t as usize))
                    .collect()
            }
            _ => Vec::new(),
        },
    }
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
        let edges = match_sequential_pairs(&[a, b], 1, 0.8);

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
        let edges = match_sequential_pairs(&[a, b], 1, 0.8);

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

        let edges_w1 = match_sequential_pairs(&[f0.clone(), f1.clone(), f2.clone()], 1, 0.8);
        assert_eq!(pair_set(&edges_w1), vec![(0, 1), (1, 2)]);

        let edges_w2 = match_sequential_pairs(&[f0, f1, f2], 2, 0.8);
        assert_eq!(pair_set(&edges_w2), vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn returns_empty_when_frames_have_no_features() {
        let empty = FrameFeatures {
            keypoints: Vec::new(),
            descriptors_orb: Some(Vec::new()),
            descriptors_sift: None,
        };
        let edges = match_sequential_pairs(&[empty.clone(), empty], 1, 0.8);
        assert!(edges.is_empty());
    }

    #[test]
    fn returns_empty_for_single_frame() {
        let f = orb_features(keypoints(3), &[1, 2, 4]);
        let edges = match_sequential_pairs(&[f], 5, 0.8);
        assert!(edges.is_empty());
    }

    #[test]
    fn ratio_test_filters_ambiguous_matches() {
        // Keypoint A0 is equidistant from B0 and B1 (both 2 bits away), so the
        // best/second-best ratio is 1.0 — rejected by a strict ratio but kept
        // by a loose one.
        let a = orb_features(vec![[0.0, 0.0]], &[1]);
        let b = orb_features(vec![[0.0, 0.0], [0.0, 0.0]], &[2, 4]);

        let strict = match_sequential_pairs(&[a.clone(), b.clone()], 1, 0.8);
        assert!(strict.is_empty(), "ambiguous match must be rejected");

        let loose = match_sequential_pairs(&[a, b], 1, 1.5);
        assert_eq!(loose.len(), 1);
        assert_eq!((loose[0].kpt_a, loose[0].kpt_b), (0, 0));
    }
}
