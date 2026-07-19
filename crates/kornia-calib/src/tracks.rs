//! Chain pairwise feature matches into multi-view [`FeatureTrack`]s via union-find.
//!
//! The front-end (e.g. a detector + matcher) produces pairwise correspondences between cameras. Each
//! correspondence carries a stable per-camera keypoint id, so the same physical point matched across
//! several pairs can be merged into one track. Feeding one shared 3D point per track to bundle
//! adjustment (instead of an independent point per camera pair) removes double-counting and couples
//! the poses through common structure.

use std::collections::HashMap;

use kornia_algebra::Vec2F64;

use crate::types::FeatureTrack;

/// A pairwise match between two cameras' keypoints. `kpt_*` are stable per-camera keypoint ids (e.g.
/// indices into that camera's detected-keypoint list) so matches chain into tracks; `uv_*` are the
/// raw pixels.
pub struct TrackEdge {
    /// First camera index.
    pub cam_a: usize,
    /// First camera keypoint id.
    pub kpt_a: u32,
    /// First camera pixel.
    pub uv_a: Vec2F64,
    /// Second camera index.
    pub cam_b: usize,
    /// Second camera keypoint id.
    pub kpt_b: u32,
    /// Second camera pixel.
    pub uv_b: Vec2F64,
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // path halving
        x = parent[x];
    }
    x
}

/// Chain pairwise matches into multi-view [`FeatureTrack`]s via union-find over `(camera, keypoint)`
/// nodes.
///
/// A track is emitted only if it spans **≥ 2 distinct cameras**. A component that contains two
/// keypoints from the **same** camera is **dropped**: a physical point projects at most once per
/// view, so a same-camera collision signals a bad merge (the COLMAP/OpenMVG track-consistency
/// invariant).
pub fn build_tracks(edges: &[TrackEdge]) -> Vec<FeatureTrack> {
    // Dense node ids for each unique (camera, keypoint), remembering its pixel.
    let mut id: HashMap<(usize, u32), usize> = HashMap::new();
    let mut nodes: Vec<(usize, Vec2F64)> = Vec::new();
    for e in edges {
        for (cam, kpt, uv) in [(e.cam_a, e.kpt_a, e.uv_a), (e.cam_b, e.kpt_b, e.uv_b)] {
            id.entry((cam, kpt)).or_insert_with(|| {
                let i = nodes.len();
                nodes.push((cam, uv));
                i
            });
        }
    }

    // Union the two endpoints of every edge.
    let mut parent: Vec<usize> = (0..nodes.len()).collect();
    for e in edges {
        let a = id[&(e.cam_a, e.kpt_a)];
        let b = id[&(e.cam_b, e.kpt_b)];
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[ra] = rb;
        }
    }

    // Group nodes by root component.
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for n in 0..nodes.len() {
        let r = find(&mut parent, n);
        groups.entry(r).or_default().push(n);
    }

    // Emit one track per component: one observation per camera; drop on a same-camera collision.
    let mut tracks = Vec::new();
    for members in groups.into_values() {
        let mut per_cam: HashMap<usize, Vec2F64> = HashMap::new();
        let mut conflict = false;
        for &n in &members {
            let (cam, uv) = nodes[n];
            if per_cam.insert(cam, uv).is_some() {
                conflict = true;
                break;
            }
        }
        if conflict || per_cam.len() < 2 {
            continue;
        }
        tracks.push(FeatureTrack {
            obs: per_cam.into_iter().collect(),
        });
    }
    tracks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(ca: usize, ka: u32, cb: usize, kb: u32) -> TrackEdge {
        TrackEdge {
            cam_a: ca,
            kpt_a: ka,
            uv_a: Vec2F64::new(ka as f64, 0.0),
            cam_b: cb,
            kpt_b: kb,
            uv_b: Vec2F64::new(kb as f64, 0.0),
        }
    }

    #[test]
    fn chains_transitive_matches_into_one_track() {
        // (0,5)-(1,7) and (1,7)-(2,9) → one 3-view track {0,1,2}.
        let tracks = build_tracks(&[edge(0, 5, 1, 7), edge(1, 7, 2, 9)]);
        assert_eq!(tracks.len(), 1);
        let mut cams: Vec<usize> = tracks[0].obs.iter().map(|(c, _)| *c).collect();
        cams.sort_unstable();
        assert_eq!(cams, vec![0, 1, 2]);
    }

    #[test]
    fn drops_same_camera_collision() {
        // (0,5)-(1,7) and (1,8)-(0,5): node (0,5) merges cams 1's kpt 7 and 8 → two cam-1 obs → drop.
        let tracks = build_tracks(&[edge(0, 5, 1, 7), edge(1, 8, 0, 5)]);
        assert!(
            tracks.is_empty(),
            "same-camera collision must drop the track"
        );
    }

    #[test]
    fn keeps_independent_two_view_tracks() {
        let tracks = build_tracks(&[edge(0, 1, 1, 1), edge(0, 2, 2, 2)]);
        assert_eq!(tracks.len(), 2);
        assert!(tracks.iter().all(|t| t.obs.len() == 2));
    }
}
