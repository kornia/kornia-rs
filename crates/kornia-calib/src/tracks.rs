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
#[derive(Debug, Clone)]
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

/// Two observations of one camera closer than this are the same feature, not a contradiction. SIFT
/// orientation siblings sit at a bit-identical (x, y), so any positive tolerance catches them; half a
/// pixel also absorbs sub-pixel refinement differences without admitting two genuinely distinct
/// image locations.
const SAME_PIXEL_PX: f64 = 0.5;

/// Chain pairwise matches into multi-view [`FeatureTrack`]s via union-find over `(camera, keypoint)`
/// nodes.
///
/// A track is emitted only if it spans **≥ 2 distinct cameras**. A component that reaches the same
/// camera at two DIFFERENT image locations is dropped: a physical point projects at most once per
/// view, so that signals a bad merge (the COLMAP/OpenMVG track-consistency invariant). Reaching it
/// twice at the SAME location is a SIFT orientation sibling and is merged instead — see the body for
/// why the distinction matters more than it sounds.
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

    // Emit one track per component: one observation per camera.
    //
    // A component can reach the same camera twice, and the two cases are NOT the same thing:
    //
    //  - ORIENTATION SIBLINGS. SIFT emits one keypoint per orientation peak at a bit-identical
    //    (x, y). Measured on a 643-keyframe clip's own feature store: 28.1% of keypoints share an
    //    exact pixel with a sibling and 15.9% of positions carry more than one. Two siblings reaching
    //    one component say the SAME THING about geometry -- same pixel, same ray -- so there is no
    //    contradiction to resolve, and dropping the track discards every genuine correspondence in it
    //    over a disagreement that carries zero information.
    //  - A GENUINE AMBIGUITY. Two distinct image locations in one camera merged into one component
    //    means the matching really is inconsistent, and the track should not be trusted.
    //
    // Dropping the whole component in BOTH cases was the previous behaviour, and it is biased by
    // construction against LONG tracks. Cross-check makes matching one-to-one inside a pair, so a
    // 2-view track can never collide and is structurally immune; a k-view track gets one exposure per
    // edge, and survival falls as (1-q)^E. That is a rule which deletes tracks in proportion to how
    // valuable they are, and it reproduces the observed profile exactly -- median track length 2 with
    // 66% two-view points, against COLMAP's median 5 with 2.6%, on identical frames. The severity
    // scales with feature density: it is pronounced at 8192 keypoints per frame and mild at the
    // ~2000 COLMAP detects by default.
    //
    // So: siblings are merged (keep the first -- they are the same pixel), a real disagreement still
    // drops the track.
    let mut tracks = Vec::new();
    for members in groups.into_values() {
        let mut per_cam: HashMap<usize, Vec2F64> = HashMap::new();
        let mut conflict = false;
        for &n in &members {
            let (cam, uv) = nodes[n];
            if let Some(prev) = per_cam.insert(cam, uv) {
                // Same pixel to within a fraction of a pixel: an orientation sibling, not a
                // contradiction. Keep the observation already recorded and carry on.
                if (prev - uv).length() <= SAME_PIXEL_PX {
                    per_cam.insert(cam, prev);
                } else {
                    conflict = true;
                    break;
                }
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

    /// An orientation sibling must not destroy the track it appears in.
    ///
    /// This is the shape of the real defect: SIFT fires two keypoints at ONE pixel (different
    /// orientation peaks), different pairs match different siblings, union-find merges them, and the
    /// old rule deleted the whole component — including every correspondence in it that was perfectly
    /// good. The bias is against LONG tracks specifically: cross-check makes matching one-to-one
    /// inside a pair, so a 2-view track can never collide, while a k-view track gets one exposure per
    /// edge.
    #[test]
    fn orientation_siblings_do_not_destroy_a_track() {
        // Camera 1 detects the same corner twice at an identical pixel: keypoints 10 and 11.
        // Camera 0 matches its keypoint 0 to sibling 10; camera 2 matches its keypoint 0 to
        // sibling 11. Union-find puts 0, 10, 11 and 2's keypoint into one component.
        let p0 = Vec2F64::new(100.0, 200.0);
        let p1 = Vec2F64::new(150.0, 250.0); // the shared pixel in camera 1
        let p2 = Vec2F64::new(200.0, 300.0);
        let edges = vec![
            TrackEdge {
                cam_a: 0,
                kpt_a: 0,
                uv_a: p0,
                cam_b: 1,
                kpt_b: 10,
                uv_b: p1,
            },
            TrackEdge {
                cam_a: 1,
                kpt_a: 11,
                uv_a: p1,
                cam_b: 2,
                kpt_b: 0,
                uv_b: p2,
            },
            TrackEdge {
                cam_a: 0,
                kpt_a: 0,
                uv_a: p0,
                cam_b: 2,
                kpt_b: 0,
                uv_b: p2,
            },
        ];
        let tracks = build_tracks(&edges);
        assert_eq!(
            tracks.len(),
            1,
            "the sibling collision must not delete the track"
        );
        assert_eq!(tracks[0].obs.len(), 3, "all three cameras must survive");
        let mut cams: Vec<usize> = tracks[0].obs.iter().map(|(c, _)| *c).collect();
        cams.sort_unstable();
        assert_eq!(cams, vec![0, 1, 2]);
    }

    /// A GENUINE same-camera ambiguity is still rejected.
    ///
    /// The pair to the test above. Merging on distance would be worthless if it also merged two
    /// distinct image locations — that is a real bad merge and the track must still die.
    #[test]
    fn distinct_same_camera_observations_still_drop_the_track() {
        let p0 = Vec2F64::new(100.0, 200.0);
        let a = Vec2F64::new(150.0, 250.0);
        let b = Vec2F64::new(600.0, 700.0); // far from `a`: a real contradiction
        let p2 = Vec2F64::new(200.0, 300.0);
        let edges = vec![
            TrackEdge {
                cam_a: 0,
                kpt_a: 0,
                uv_a: p0,
                cam_b: 1,
                kpt_b: 10,
                uv_b: a,
            },
            TrackEdge {
                cam_a: 1,
                kpt_a: 11,
                uv_a: b,
                cam_b: 2,
                kpt_b: 0,
                uv_b: p2,
            },
            TrackEdge {
                cam_a: 0,
                kpt_a: 0,
                uv_a: p0,
                cam_b: 2,
                kpt_b: 0,
                uv_b: p2,
            },
        ];
        assert!(
            build_tracks(&edges).is_empty(),
            "a real same-camera ambiguity must still drop"
        );
    }
}
