//! Tags-free (feature-driven) multi-camera calibration via incremental structure-from-motion.
//!
//! Natural-feature tracks — not a tag — drive the geometry. A best-connected camera pair bootstraps
//! the reconstruction from the two-view essential matrix, remaining cameras register by PnP against
//! the growing point cloud — each newly registered camera refined against its co-visible
//! neighbours before it is allowed to triangulate, and the whole map re-solved every time the
//! registered set grows 10% — and a terminal bundle adjustment polishes everything. The
//! reconstruction is recovered **up to scale** (the fundamental monocular ambiguity); a tag fixes
//! that one scalar — the tag is a *scale bar*, nothing else. Output poses are `T_world_cam` in the
//! reference camera's frame, metric ONLY when a tag actually anchored the scale: supplying a tag is
//! not sufficient, since it must also be seen by two registered views and triangulate. See
//! [`crate::ScaleSource`], which reports which case a given result is.
//!
//! Everything except the incremental orchestration is reused: `ransac_essential_5pt` +
//! `decompose_essential` (bootstrap relative pose), [`kornia_3d::pose::triangulate_matched_points`],
//! [`kornia_3d::pnp::solve_pnp_ransac`] (register a new camera into the cloud), and
//! [`kornia_3d::ba_schur::bundle_adjust_schur`].

use std::collections::{HashMap, HashSet};

use kornia_3d::ba::{BaMotionPrior, BaObservation, BaParams, BaPosePrior, BaResult};
use kornia_3d::ba_schur::bundle_adjust_schur_with_all_priors;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pnp::{solve_pnp_ransac, PnPMethod, RansacParams as PnpRansacParams};
use kornia_3d::pose::{
    decompose_essential, ransac_essential_5pt, ransac_fundamental, ransac_homography,
    triangulate_matched_points, Pose3d, RansacParams as TvRp, TriangulationConfig,
};
use kornia_3d::ransac::RobustKernelKind;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec2F64, Vec3AF32, Vec3F64};

use crate::error::CalibError;
use crate::types::{
    CameraStats, FeatureTrack, Observation, Point, Reconstruction, ReconstructionConfig,
    ScaleSource, TagObservation,
};

/// Convert an f32 PnP rotation/translation (world→cam) into an f64 [`Pose3d`].
fn pose_from_pnp(r: Mat3AF32, t: Vec3AF32) -> Pose3d {
    let a = r.to_cols_array(); // column-major [f32; 9]
    let rot = Mat3F64::from_cols_array(&[
        a[0] as f64,
        a[1] as f64,
        a[2] as f64,
        a[3] as f64,
        a[4] as f64,
        a[5] as f64,
        a[6] as f64,
        a[7] as f64,
        a[8] as f64,
    ]);
    Pose3d::new(rot, Vec3F64::new(t.x as f64, t.y as f64, t.z as f64))
}

/// Reproject a world point into a camera and return the normalized-coordinate residual, or `None`
/// behind the camera.
fn norm_residual(pose: &Pose3d, p_world: Vec3F64, n: Vec2F64) -> Option<f64> {
    let pc = pose.transform_point(&p_world);
    if pc.z <= 1e-6 {
        return None;
    }
    Some(((pc.x / pc.z - n.x).powi(2) + (pc.y / pc.z - n.y).powi(2)).sqrt())
}

/// Run a global bundle adjustment each time the registered set grows by this factor.
///
/// COLMAP's `Mapper.ba_global_images_ratio`, same value. A ratio rather than a fixed interval so
/// the total cost stays proportional to the problem: many cheap solves while the map is small, few
/// expensive ones once it is large.
const BA_IMAGES_RATIO: f64 = 1.1;

/// Registered cameras kept free, besides the newly registered one, in the local BA that follows
/// each registration. ORB-SLAM's local window is the same order of magnitude.
const LOCAL_BA_NEIGHBOURS: usize = 6;

/// LM iterations for that local BA. Fewer than the global `ReconstructionConfig::max_iterations`: this runs
/// once per registered camera and only has to pull a fresh PnP pose onto the existing structure,
/// not converge the whole map.
const LOCAL_BA_ITERATIONS: usize = 25;

/// How many of the best-supported view pairs are solved and ranked as bootstrap candidates.
///
/// Each rejected candidate costs one two-view solve, so this is the budget for buying a
/// well-conditioned seed. Twelve is enough to get past a run of adjacent, near-zero-baseline pairs
/// on video without making the bootstrap a measurable share of the reconstruction.
const BOOTSTRAP_CANDIDATES: usize = 12;

/// A seed pair whose median triangulation angle is below this is too degenerate to anchor a map.
///
/// This ORDERS seeds rather than rejecting them: `select_bootstrap_pair` sorts floor-passing pairs
/// first and falls back to cheirality support when none qualify, so raising it can change which
/// pair wins but never whether a reconstruction happens at all. Deliberately far below COLMAP's
/// `Mapper.init_min_tri_angle = 16`, which assumes a candidate set that can actually reach a
/// double-digit angle; at keyframe spacing typical of a handheld walkthrough, the widest-parallax
/// candidate also carries the thinnest support, and a seed fitted to thin support loses to a
/// shorter-baseline one fitted to plenty.
const MIN_SEED_PARALLAX_DEG: f64 = 1.5;

/// COLMAP's `Mapper.init_max_forward_motion`, same value: a pair whose unit translation points
/// nearly along the optical axis triangulates everything near the epipole, where depth is barely
/// observable. A walkthrough is forward-motion dominated, so this DEMOTES rather than rejects —
/// rejecting outright would empty the candidate list on exactly the clips that need a seed most.
const MAX_SEED_FORWARD: f64 = 0.95;

/// Reporting threshold on the homography-vs-fundamental ratio (ORB-SLAM2's value), above which the
/// pair is planar or rotation-dominated enough that the essential decomposition is ambiguous.
/// Flagged through `KORNIA_CALIB_DEBUG`, never enforced; see `homography_vs_fundamental_ratio`.
const MAX_SEED_H_RATIO: f64 = 0.45;

/// Filter threshold as a multiple of [`ReconstructionConfig::max_reprojection_error`].
///
/// COLMAP creates points at `tri_max_reproj_error` and filters them at the looser
/// `filter_max_reproj_error`. Filtering at the creation threshold would make a point sitting on the
/// boundary flip state every round — dropped, retriangulated from the same evidence, dropped again.
const FILTER_REPROJ_SLACK: f64 = 2.0;

/// Drop points bundle adjustment could not fix, and shed the individual sightings that failed on
/// the points that survive. Returns how many points were dropped.
///
/// Filtering is at the OBSERVATION level, not the point level: a view with one blurred sighting of
/// an otherwise good point should lose that sighting, not the point — and the point should stop
/// paying that view's residual in every later solve. Without this a bad sighting is permanent, and
/// since `Reconstruction::reproj_rmse_px` is a mean over observations, a tail of them inflates it
/// several-fold while the geometry underneath is fine. Measured on 7-Scenes `pumpkin` at 60
/// keyframes: 13.49 px reported against a robust 1.40 px, at an ATE of 3.78 cm versus the filtered
/// arm's 3.76 cm. A caller that gates on reported rmse then rejects a good reconstruction.
///
/// Only observations of REGISTERED cameras are judged. An unregistered camera's observations are
/// the 2D-3D correspondences its own PnP will need, and testing them against a pose that does not
/// exist yet is meaningless.
fn filter_points(
    point3d: &mut HashMap<usize, Vec3F64>,
    norm: &mut [Vec<(usize, Vec2F64)>],
    norm_depth: &mut [Vec<Option<f32>>],
    poses: &[Option<Pose3d>],
    max_reproj_norm: f64,
) -> usize {
    let before = point3d.len();
    // ONE predicate, shared by the survival test and the pruning below. Written twice they diverged
    // on `Some(NaN)`: `is_some_and(|e| e <= max)` rejects it while `Some(e) => e > max` does not, so
    // a NaN-poisoned sighting counted against its point's survival AND stayed in the map.
    let is_good = |pose: &Pose3d, p: Vec3F64, uv: Vec2F64| {
        norm_residual(pose, p, uv).is_some_and(|e| e <= max_reproj_norm)
    };
    // Which surviving tracks to prune observations from. Collected rather than pruned in place
    // because `retain`'s closure holds `point3d` borrowed; the order is irrelevant, as each track's
    // pruning depends only on itself.
    let mut prune: Vec<usize> = Vec::new();
    point3d.retain(|ti, p| {
        let (mut good, mut seen) = (0usize, 0usize);
        for (c, uv) in &norm[*ti] {
            let Some(pose) = &poses[*c] else { continue };
            seen += 1;
            if is_good(pose, *p, *uv) {
                good += 1;
            }
        }
        // Two surviving sightings is the minimum that still constrains a 3D point, and a track that
        // lost more than half of what saw it is not a point with a bad view — it is a bad point.
        if good < 2 || 2 * good < seen {
            return false;
        }
        // NO triangulation-angle gate here. A low-parallax point is badly conditioned, but culling
        // one is a separate decision from removing a wrong measurement, it is unmeasured by the
        // work this filter is justified on, and the only threshold available to it runs backwards:
        // `sequential()` lowers `min_parallax_deg` to 0.05 precisely because video BOOTSTRAP needs
        // ill-conditioned seed points, so using it as an output cull would give video callers — the
        // ones whose clouds are actually full of low-parallax points — the loosest cull of all.
        prune.push(*ti);
        true
    });
    for ti in prune {
        let p = point3d[&ti];
        // Mask first, then retain both arrays through it: `norm` and `norm_depth` must stay index
        // paired, and the order must SURVIVE. `tracks.rs` publishes ascending-camera-index order as
        // contract, and observations are emitted in this order — `swap_remove` would be cheaper but
        // permutes the track, breaking that for every track that loses a sighting.
        //
        // A DROPPED point keeps every observation, so `triangulate_new` can rebuild that track from
        // better poses later; only surviving points shed sightings. Observations of UNREGISTERED
        // cameras are kept regardless — they are that camera's future PnP correspondences, and
        // judging them against a pose that does not exist yet is meaningless.
        let keep: Vec<bool> = norm[ti]
            .iter()
            .map(|(c, uv)| match &poses[*c] {
                None => true,
                Some(pose) => is_good(pose, p, *uv),
            })
            .collect();
        let mut k = 0usize;
        norm[ti].retain(|_| {
            k += 1;
            keep[k - 1]
        });
        let mut k = 0usize;
        norm_depth[ti].retain(|_| {
            k += 1;
            keep[k - 1]
        });
    }
    before - point3d.len()
}

/// COLMAP's `CompleteTracks`: re-admit observations the filter removed, once the poses that
/// condemned them have improved. Returns how many were recovered.
///
/// Filtering happens against the geometry of the moment, and the first filter fires when three
/// cameras are registered — so without this a sighting rejected against an early, raw PnP pose is
/// lost for the rest of the run, even after the solve that would have vindicated it. Track length
/// then erodes monotonically over a long capture: every global BA judges every sighting again, and
/// the decision only ever goes one way.
///
/// Only tracks that actually lost something are considered, and only cameras that now hold a pose.
/// Observations are matched on CAMERA index, not pixel: a track holds at most one observation per
/// camera by construction, so the index is the identity, and a float comparison would be slower and
/// wrong at the boundary.
fn complete_tracks(
    point3d: &HashMap<usize, Vec3F64>,
    norm: &mut [Vec<(usize, Vec2F64)>],
    norm_depth: &mut [Vec<Option<f32>>],
    norm0: &[Vec<(usize, Vec2F64)>],
    norm_depth0: &[Vec<Option<f32>>],
    poses: &[Option<Pose3d>],
    max_reproj_norm: f64,
) -> usize {
    let mut added = 0usize;
    for (ti, p) in point3d.iter() {
        let (Some(orig), Some(orig_d)) = (norm0.get(*ti), norm_depth0.get(*ti)) else {
            continue;
        };
        if orig.len() == norm[*ti].len() {
            continue; // nothing was ever removed from this track
        }
        let before = norm[*ti].len();
        for (k, (c, uv)) in orig.iter().enumerate() {
            if norm[*ti].iter().any(|(c2, _)| c2 == c) {
                continue;
            }
            let Some(Some(pose)) = poses.get(*c) else {
                continue;
            };
            if norm_residual(pose, *p, *uv).is_some_and(|e| e <= max_reproj_norm) {
                norm[*ti].push((*c, *uv));
                norm_depth[*ti].push(orig_d.get(k).copied().flatten());
                added += 1;
            }
        }
        // Re-admission APPENDS, so restore ascending camera order — `tracks.rs` publishes that as
        // contract and `Reconstruction::observations` is emitted in this order. Sorted in lockstep
        // with `norm_depth` by permuting an index array, since the two are index-paired.
        if norm[*ti].len() > before {
            let mut idx: Vec<usize> = (0..norm[*ti].len()).collect();
            idx.sort_unstable_by_key(|i| norm[*ti][*i].0);
            norm[*ti] = idx.iter().map(|i| norm[*ti][*i]).collect();
            norm_depth[*ti] = idx.iter().map(|i| norm_depth[*ti][*i]).collect();
        }
    }
    added
}

/// COLMAP's `MergeTracks`: fuse tracks that resolved to the same physical point, so the solve
/// learns they are one. Returns how many tracks were absorbed.
///
/// Two views of the same landmark reached through different match chains stay separate tracks
/// forever otherwise, each carrying half the evidence. That is what makes a revisit inert: loop
/// closure moves the CAMERAS while every landmark stays duplicated, so the reprojection cost never
/// learns the loop exists and the next bundle adjustment pulls it back open. Merging is what turns
/// a revisit into a constraint the solve can feel.
///
/// Two conditions, both necessary:
///
/// * **No shared camera.** A track holds at most one observation per camera by construction, and a
///   merged track must too — two sightings of "the same" point in one image mean it is not one
///   point, whatever the distance says.
/// * **Every observation of BOTH tracks must reproject within tolerance against the MERGED
///   position.** Distance alone would fuse a genuinely close pair and hand the solve a point that
///   fits neither, manufacturing exactly the drift this exists to remove.
///
/// Candidates come from a voxel grid at `radius`, so this is linear in the cloud rather than
/// quadratic. Iteration is over SORTED track ids: `HashMap` order is randomised per process, and
/// merging is order-dependent (a absorbs b, not the reverse), so unsorted iteration would make the
/// reconstruction irreproducible.
fn merge_tracks(
    point3d: &mut HashMap<usize, Vec3F64>,
    norm: &mut [Vec<(usize, Vec2F64)>],
    norm_depth: &mut [Vec<Option<f32>>],
    poses: &[Option<Pose3d>],
    max_reproj_norm: f64,
    radius: f64,
) -> usize {
    if radius <= 0.0 {
        return 0;
    }
    let cell = |v: Vec3F64| {
        (
            (v.x / radius).floor() as i64,
            (v.y / radius).floor() as i64,
            (v.z / radius).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut ids: Vec<usize> = point3d.keys().copied().collect();
    ids.sort_unstable();
    for ti in &ids {
        grid.entry(cell(point3d[ti])).or_default().push(*ti);
    }

    let mut merged = 0usize;
    let mut absorbed: HashSet<usize> = HashSet::new();
    for ta in &ids {
        if absorbed.contains(ta) {
            continue;
        }
        let pa = point3d[ta];
        let (cx, cy, cz) = cell(pa);
        let mut cands: Vec<usize> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        cands.extend(v.iter().copied());
                    }
                }
            }
        }
        cands.sort_unstable();
        for tb in cands {
            if tb <= *ta || absorbed.contains(&tb) || !point3d.contains_key(&tb) {
                continue;
            }
            let pb = point3d[&tb];
            let d = pa - pb;
            if (d.x * d.x + d.y * d.y + d.z * d.z).sqrt() > radius {
                continue;
            }
            if norm[*ta]
                .iter()
                .any(|(c, _)| norm[tb].iter().any(|(c2, _)| c2 == c))
            {
                continue;
            }
            // Weighted by observation count: the merged point should sit where the combined
            // evidence puts it, not halfway between a 30-view track and a 2-view one.
            let (na, nb) = (norm[*ta].len() as f64, norm[tb].len() as f64);
            let pm = (pa * na + pb * nb) / (na + nb);
            let fits = |ti: usize| {
                norm[ti].iter().all(|(c, uv)| match poses.get(*c) {
                    Some(Some(pose)) => {
                        norm_residual(pose, pm, *uv).is_some_and(|e| e <= max_reproj_norm)
                    }
                    // Unregistered cameras cannot judge, and must not veto.
                    _ => true,
                })
            };
            if !fits(*ta) || !fits(tb) {
                continue;
            }
            // `tb`'s observations MOVE, they are not copied: leaving them behind would let
            // `triangulate_new` rebuild `tb` on the next round and silently undo the merge.
            let taken: Vec<(usize, Vec2F64)> = std::mem::take(&mut norm[tb]);
            let taken_d: Vec<Option<f32>> = std::mem::take(&mut norm_depth[tb]);
            norm[*ta].extend(taken);
            norm_depth[*ta].extend(taken_d);
            let mut idx: Vec<usize> = (0..norm[*ta].len()).collect();
            idx.sort_unstable_by_key(|i| norm[*ta][*i].0);
            norm[*ta] = idx.iter().map(|i| norm[*ta][*i]).collect();
            norm_depth[*ta] = idx.iter().map(|i| norm_depth[*ta][*i]).collect();
            point3d.insert(*ta, pm);
            point3d.remove(&tb);
            absorbed.insert(tb);
            merged += 1;
        }
    }
    merged
}

/// How many cameras currently hold a pose.
fn registered_now(poses: &[Option<Pose3d>]) -> usize {
    poses.iter().filter(|p| p.is_some()).count()
}

/// `focus` plus the `neighbours` registered cameras most co-visible with it.
///
/// Co-visibility rather than index distance, because registration order is not trajectory order: a
/// camera registered late can sit anywhere along the walk, and its error is shared with whatever
/// observes the same points, not with whatever has an adjacent index. Ranked by how many
/// triangulated tracks each candidate shares with `focus`, which is the criterion ORB-SLAM's local
/// window uses.
fn covisible_window(
    focus: usize,
    a0: usize,
    point3d: &HashMap<usize, Vec3F64>,
    norm: &[Vec<(usize, Vec2F64)>],
    poses: &[Option<Pose3d>],
    neighbours: usize,
) -> HashSet<usize> {
    let mut shared: HashMap<usize, usize> = HashMap::new();
    for ti in point3d.keys() {
        let obs = &norm[*ti];
        if !obs.iter().any(|(c, _)| *c == focus) {
            continue;
        }
        for (c, _) in obs {
            // `a0` is excluded: the global gauge anchor is pinned in every solve, so admitting it
            // would spend one of the `neighbours` slots on a pose the local BA cannot move — a
            // 7-camera window that frees only 6. It is most co-visible early in growth, which is
            // exactly when the window is tightest.
            if *c != focus && *c != a0 && poses[*c].is_some() {
                *shared.entry(*c).or_insert(0) += 1;
            }
        }
    }
    let mut ranked: Vec<(usize, usize)> = shared.into_iter().collect();
    // Most-shared first, camera index breaking ties: `HashMap` iteration order is randomised per
    // process, so an unstable ranking here would make the whole reconstruction irreproducible.
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut out: HashSet<usize> = HashSet::from([focus]);
    out.extend(ranked.into_iter().take(neighbours).map(|(c, _)| c));
    out
}

/// Bundle-adjust and write the result back into the growing state.
///
/// `free` restricts which cameras may move: `None` frees every registered camera (a global BA),
/// `Some(set)` pins the rest through `BaObservation::fixed_pose`. Only free cameras are written
/// back — a pinned camera's entry in the result is its input pose, and the gauge anchor `a0` must
/// never move at all.
#[allow(clippy::too_many_arguments)]
fn refine_in_place(
    poses: &mut [Option<Pose3d>],
    point3d: &mut HashMap<usize, Vec3F64>,
    norm: &[Vec<(usize, Vec2F64)>],
    norm_depth: &[Vec<Option<f32>>],
    idcam: &PinholeCamera,
    a0: usize,
    config: &ReconstructionConfig,
    free: Option<&HashSet<usize>>,
    max_iterations: usize,
) -> Result<(), CalibError> {
    // Sorted, not `HashMap` order: the solve accumulates in this order and float addition is not
    // associative, so an unsorted pass would make the map differ run to run. Same reason the
    // terminal solve sorts.
    let mut track_ids: Vec<usize> = point3d.keys().copied().collect();
    track_ids.sort_unstable();
    let res = global_ba(
        poses,
        point3d,
        &track_ids,
        norm,
        norm_depth,
        idcam,
        a0,
        config,
        free,
        max_iterations,
    )?;
    for (c, p) in poses.iter_mut().enumerate() {
        if p.is_some() && c != a0 && free.is_none_or(|f| f.contains(&c)) {
            *p = Some(res.poses[c]);
        }
    }
    for (pidx, ti) in track_ids.iter().enumerate() {
        if let Some(v) = res.points.get(pidx) {
            point3d.insert(*ti, *v);
        }
    }
    Ok(())
}

/// Reconstruct from feature tracks, returning the MAP as well as the camera poses.
///
/// Bootstrap from the best-supported view pair, register the rest by PnP against the growing
/// cloud, bundle-adjust, and optionally anchor the metric scale from a tag.
///
/// Returns the map, not just the cameras. If only the rig geometry is wanted, convert with
/// `RigCalibration::from(..)` -- see that `From` impl for exactly what the conversion drops.
///
/// # Arguments
///
/// * `cameras` - intrinsics per view, indexed the same way as `FeatureTrack::obs`. Distortion is
///   applied when normalising observations.
/// * `tags_for_scale` - optional metric anchor. Empty means the result is up to scale; the first
///   tag's frame otherwise fixes the world gauge and its known side length fixes the metric.
/// * `tracks` - multi-view feature tracks. A track needs at least two views to triangulate, and
///   the reconstruction needs enough shared tracks between views to register them.
/// * `config` - solver settings; see [`ReconstructionConfig`].
/// * `obs_depth` - optional metric depth per observation, shaped exactly like `tracks`:
///   `obs_depth[i][j]` is the depth of `tracks[i].obs[j]`. `None` for the classic depth-free solve;
///   ignored unless [`ReconstructionConfig::depth_prior_rel_sigma`] is positive.
///
/// # Depth
///
/// Monocular reprojection is exactly scale-invariant, so a fiducial-free walkthrough has no metric
/// scale and no defence against drift along the chain — measured, rooms late in a clip reconstruct
/// several times larger than early ones. Depth residuals observe absolute scale directly and pin
/// EVERY segment, not just a global average. They are a soft prior: robustified, sigma-weighted by
/// [`ReconstructionConfig::depth_prior_rel_sigma`], re-gauged per
/// [`ReconstructionConfig::depth_per_keyframe_scale`].
///
/// # Returns
///
/// A [`Reconstruction`]: the registered view poses (`T_world_cam`, `None` where a view could not
/// be registered), the triangulated points, the track each point came from, the observations that
/// survived the solve, per-view statistics, and where the metric scale came from.
///
/// # Errors
///
/// [`CalibError`] if the inputs cannot produce a reconstruction — too few views or tracks, no
/// viable bootstrap pair, or a bundle-adjustment failure.
///
/// # Example
///
/// ```no_run
/// use kornia_calib::{reconstruct, ReconstructionConfig, FeatureTrack, ScaleSource};
/// use kornia_3d::camera::PinholeCamera;
/// use kornia_algebra::Vec2F64;
///
/// # fn main() -> Result<(), kornia_calib::CalibError> {
/// let cam = || PinholeCamera { fx: 600.0, fy: 600.0, cx: 320.0, cy: 240.0, ..PinholeCamera::IDENTITY };
/// let cameras = vec![cam(), cam(), cam()];
///
/// // One scene point seen by three views. Real input comes from a matcher.
/// let tracks = vec![FeatureTrack {
///     obs: vec![
///         (0, Vec2F64::new(320.0, 240.0)),
///         (1, Vec2F64::new(300.0, 241.0)),
///         (2, Vec2F64::new(340.0, 239.0)),
///     ],
/// }];
///
/// // `None` for the depth argument: no metric depth measurements, so the map is up to scale.
/// let recon = reconstruct(&cameras, &[], &tracks, &ReconstructionConfig::new(0.1), None)?;
///
/// // No tag was supplied, so the map is honestly up to scale rather than silently "metric".
/// assert_eq!(recon.scale, ScaleSource::UpToScale);
///
/// // Carry per-track data onto the map without re-matching -- each point names its own track.
/// for point in &recon.points {
///     let _world_position = point.position;
///     let _source_track = &tracks[point.track_id];
/// }
/// # Ok(())
/// # }
/// ```
pub fn reconstruct(
    cameras: &[PinholeCamera],
    tags_for_scale: &[TagObservation],
    tracks: &[FeatureTrack],
    config: &ReconstructionConfig,
    obs_depth: Option<&[Vec<Option<f64>>]>,
) -> Result<Reconstruction, CalibError> {
    reconstruct_inner(
        cameras,
        tags_for_scale,
        tracks,
        config,
        obs_depth,
        true,
        true,
    )
}

/// [`reconstruct`], with its two behaviour-defining choices switchable off.
///
/// `growth_ba: false` leaves only the terminal solve — the arrangement this file had before the
/// local and periodic solves were added. `ranked_seed: false` restricts bootstrap selection to the
/// single most-shared-tracks pair — the arrangement this file had before the seed was chosen on
/// geometry. Both exist so
/// `growth_ba_keeps_the_long_tracks_a_terminal_solve_alone_loses` and
/// `ranked_bootstrap_registers_more_views_than_the_most_shared_tracks_seed` can measure the two
/// arms of the same pipeline against each other on one scene, instead of asserting that the current
/// arm is merely "good enough" — an assertion that would pass on a no-op change. Deliberately NOT
/// public: a caller has no reason to ask for the worse arm.
fn reconstruct_inner(
    cameras: &[PinholeCamera],
    tags_for_scale: &[TagObservation],
    tracks: &[FeatureTrack],
    config: &ReconstructionConfig,
    obs_depth: Option<&[Vec<Option<f64>>]>,
    growth_ba: bool,
    ranked_seed: bool,
) -> Result<Reconstruction, CalibError> {
    let n_cams = cameras.len();
    let idcam = PinholeCamera::IDENTITY;
    let tcfg = TriangulationConfig {
        min_parallax_deg: config.min_parallax_deg,
        max_reprojection_error: config.max_reprojection_error,
        ..Default::default()
    };

    // Per track: normalized observation per camera (undistort + K⁻¹). Raw pixels stay in `tracks`.
    // MUTABLE because `refine_intrinsics` rewrites these in place once it has fitted a correction.
    let mut norm: Vec<Vec<(usize, Vec2F64)>> = tracks
        .iter()
        .map(|t| {
            t.obs
                .iter()
                .map(|(c, uv)| (*c, cameras[*c].normalize(*uv)))
                .collect()
        })
        .collect();
    // Parallel to `norm`: metric depth per observation, or `None`. All-`None` when the feature is
    // off, so downstream sites index it unconditionally; a wrong-shaped caller array yields `None`.
    //
    // Built by walking `norm`, NOT `obs_depth`, so that promise is structural rather than a hope
    // about the caller. Reading `obs_depth` directly made the shape whatever the caller passed:
    // every reader here goes through `.get().and_then(.get())` and so merely saw `None` on a short
    // row — but a row of the WRONG length silently pairs each depth with the wrong observation, and
    // `filter_points` now mutates the two in lockstep, where a length mismatch would desynchronise
    // them outright.
    let mut norm_depth: Vec<Vec<Option<f32>>> = match obs_depth {
        Some(d) if config.depth_prior_rel_sigma > 0.0 => norm
            .iter()
            .enumerate()
            .map(|(ti, obs)| {
                let row = d.get(ti);
                (0..obs.len())
                    .map(|j| {
                        row.and_then(|r| r.get(j))
                            .copied()
                            .flatten()
                            .map(|v| v as f32)
                    })
                    .collect()
            })
            .collect(),
        _ => norm.iter().map(|t| vec![None; t.len()]).collect(),
    };

    // Pristine copy of every observation, kept so `filter_points`' deletions are RECOVERABLE.
    //
    // The filter judges a sighting against the poses of the moment, and the first filter fires when
    // three cameras are registered — so a sighting rejected against an early, bad pose was gone for
    // good, even after later solves corrected the geometry that condemned it. COLMAP does not do
    // that: its `DeleteObservation` cuts the Point3D<->Point2D link while the Point2D survives in
    // the correspondence graph, precisely so `CompleteTracks` can re-admit it. This is that graph.
    let norm0 = norm.clone();
    let norm_depth0 = norm_depth.clone();

    // --- Bootstrap: pick a seed pair on GEOMETRY, then two-view essential (world = cam a0, s = 1). ---
    let (a0, b0, seed_pose) = select_bootstrap_pair(tracks, &norm, cameras, &idcam, ranked_seed)?;

    let mut poses: Vec<Option<Pose3d>> = vec![None; n_cams];
    poses[a0] = Some(Pose3d::IDENTITY);
    poses[b0] = Some(seed_pose); // T_b0_a0, unit translation

    // Triangulate every track visible in the bootstrap pair → seed the point cloud (world = a0 frame).
    let mut point3d: HashMap<usize, Vec3F64> = HashMap::new();
    triangulate_new(&mut point3d, &norm, &poses, &idcam, &tcfg);

    // --- Incremental grow: register the unplaced camera with the most 2D↔3D links via PnP. ---
    // PnP (nondeterministic EPnP-RANSAC) can transiently fail for one camera while others remain
    // solvable, so a failure marks just that camera unregisterable and the loop keeps growing —
    // NOT aborting every remaining camera.
    let mut pnp_failed: HashSet<usize> = HashSet::new();
    // Next registered-camera count at which a global bundle adjustment is due. Seeded off the
    // bootstrap pair, floored at 3 so the first registration does not immediately trigger one.
    let mut next_ba = (registered_now(&poses) as f64 * BA_IMAGES_RATIO).max(3.0);
    loop {
        // For each unplaced camera, gather (world_point, normalized_pixel) from tracks with a 3D point.
        let mut best: Option<(usize, Vec<Vec3F64>, Vec<Vec2F64>)> = None;
        for c in 0..n_cams {
            if poses[c].is_some() || pnp_failed.contains(&c) {
                continue;
            }
            let (mut wp, mut ip) = (Vec::new(), Vec::new());
            for (ti, obs) in norm.iter().enumerate() {
                if let (Some(p), Some((_, uv))) =
                    (point3d.get(&ti), obs.iter().find(|(cc, _)| *cc == c))
                {
                    wp.push(*p);
                    ip.push(*uv);
                }
            }
            if wp.len() >= 4 && best.as_ref().is_none_or(|(_, w, _)| wp.len() > w.len()) {
                best = Some((c, wp, ip));
            }
        }
        let Some((c, wp, ip)) = best else { break };

        let world: Vec<Vec3AF32> = wp
            .iter()
            .map(|p| Vec3AF32::new(p.x as f32, p.y as f32, p.z as f32))
            .collect();
        let image: Vec<Vec2F32> = ip
            .iter()
            .map(|p| Vec2F32::new(p.x as f32, p.y as f32))
            .collect();
        let pnp = solve_pnp_ransac(
            &world,
            &image,
            &Mat3AF32::IDENTITY, // normalized coords ⇒ identity intrinsics
            None,
            PnPMethod::EPnPDefault,
            &PnpRansacParams {
                // Normalized units: the pipeline's own outlier bound, defaulting to the 0.01 this
                // was hardcoded to.
                reproj_threshold_px: config.max_reprojection_error as f32,
                // Seed the sampler; `RansacParams::default()` leaves `random_seed: None`, which
                // made registration NONDETERMINISTIC — 12 / 30 / 39 cameras over three runs of the
                // same command on EuRoC MH01. A transient PnP failure marks a camera
                // unregisterable, so that randomness changes the map, not merely its timing.
                random_seed: Some(0x00C0_FFEE ^ c as u64),
                ..Default::default()
            },
        );
        // PnP returning `Ok` is not acceptance: `solve_pnp_ransac` succeeds on *any* consensus, and
        // `triangulate_new` builds points FROM the pose which feed the next camera's PnP. Measured
        // on EuRoC MH01, global RMSE rose WITH the registered count: 34 cams 10.4 px, 38 at 23.2.
        let accepted = pnp.as_ref().ok().and_then(|r| {
            let pose = pose_from_pnp(r.pose.rotation, r.pose.translation);
            // Score BOTH the refit pose and the RANSAC consensus, keep the larger: `inliers` is
            // classified against the PRE-refit minimal-sample model, so it under-counts — a
            // 107-inlier consensus has been seen reclassifying to under 30 against its own refit.
            // Same threshold for both.
            let n_refit = wp
                .iter()
                .zip(ip.iter())
                .filter(|(w, i)| {
                    norm_residual(&pose, **w, **i)
                        .is_some_and(|e| e <= config.max_reprojection_error)
                })
                .count();
            (n_refit.max(r.inliers.len()) >= config.min_registration_inliers).then_some(pose)
        });
        match accepted {
            Some(pose) => {
                poses[c] = Some(pose);
                if let Some(cb) = config.progress.as_ref() {
                    cb(poses.iter().filter(|p| p.is_some()).count(), n_cams);
                }
                // Refine the new pose against the existing map BEFORE triangulating from it. A
                // point built off a raw PnP pose inherits that pose's error and then becomes the
                // 2D↔3D evidence the NEXT camera registers against, so the error compounds down
                // the chain rather than staying local. Failure here (a window with nothing free in
                // it) is an ordinary no-op: keep the PnP pose and carry on.
                if growth_ba {
                    let window =
                        covisible_window(c, a0, &point3d, &norm, &poses, LOCAL_BA_NEIGHBOURS);
                    let _ = refine_in_place(
                        &mut poses,
                        &mut point3d,
                        &norm,
                        &norm_depth,
                        &idcam,
                        a0,
                        config,
                        Some(&window),
                        LOCAL_BA_ITERATIONS,
                    );
                }
                let before = point3d.len();
                triangulate_new(&mut point3d, &norm, &poses, &idcam, &tcfg);
                // A camera that failed earlier may register comfortably once the cloud has grown;
                // without this retry `min_registration_inliers` would make every transient
                // rejection permanent and strictly REDUCE the registered count versus no gate at
                // all. Gating on the cloud ACTUALLY growing also makes it terminate: between clears
                // each iteration either registers a camera or drops one from consideration.
                if point3d.len() > before {
                    pnp_failed.clear();
                }
                // Periodic global bundle adjustment, on COLMAP's `ba_global_images_ratio`: run one
                // every time the registered set has grown by 10%. A ratio rather than a fixed
                // interval keeps the total cost proportional — frequent while the map is small and
                // cheap, rare once it is large and expensive.
                //
                // The local BA above only ever moves a seven-camera window, so error accumulated
                // ALONG the chain is nobody's residual until a solve sees the whole thing. Without
                // this the reconstruction drifts until PnP inlier counts sink under
                // `min_registration_inliers` and growth stalls with most views unregistered.
                if growth_ba && registered_now(&poses) as f64 >= next_ba {
                    // The solve is ATTEMPTED, and the schedule advances either way. Gating the
                    // advance on success meant a global BA that failed — `CholeskyFailed` once LM
                    // damping escalates past 1e10, which is exactly what an ill-conditioned map
                    // produces — left `next_ba` behind a monotonically growing `registered_now`.
                    // The guard was then true on every following registration, turning ~23
                    // scheduled global solves into one per registration, each a full assembly over
                    // every point and observation, each failing the same way, and each discarded by
                    // `.is_ok()` so nothing reported it. A failed refinement is a no-op on the map;
                    // retrying it immediately cannot fix the conditioning that caused it.
                    let _ = refine_in_place(
                        &mut poses,
                        &mut point3d,
                        &norm,
                        &norm_depth,
                        &idcam,
                        a0,
                        config,
                        None,
                        config.max_iterations,
                    );
                    // COLMAP's iterate step: BA -> filter -> retriangulate. Filtering after the
                    // solve drops what BA could not fix and sheds the sightings it could not
                    // reconcile; retriangulating rebuilds those tracks from the refined poses. So
                    // the cloud the NEXT registration is judged against is clean, rather than
                    // accreting every early mistake for the rest of the walk.
                    filter_points(
                        &mut point3d,
                        &mut norm,
                        &mut norm_depth,
                        &poses,
                        FILTER_REPROJ_SLACK * config.max_reprojection_error,
                    );
                    // Re-admit before retriangulating: a track starved only because the filter took
                    // its sightings should get them back before `triangulate_new` judges whether it
                    // can be rebuilt.
                    if config.complete_tracks {
                        complete_tracks(
                            &point3d,
                            &mut norm,
                            &mut norm_depth,
                            &norm0,
                            &norm_depth0,
                            &poses,
                            FILTER_REPROJ_SLACK * config.max_reprojection_error,
                        );
                    }
                    // Merge radius from the reprojection tolerance at TYPICAL depth: a point may
                    // sit `tol * z` off-axis and still land inside the pixel tolerance at range z.
                    // Median depth rather than per-point so one radius serves the whole grid.
                    if config.merge_radius_rel > 0.0 {
                        let mut zs: Vec<f64> = point3d
                            .values()
                            .filter_map(|p| poses[a0].as_ref().map(|q| q.transform_point(p).z))
                            .filter(|z| *z > 0.0)
                            .collect();
                        if !zs.is_empty() {
                            zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let zmed = zs[zs.len() / 2];
                            merge_tracks(
                                &mut point3d,
                                &mut norm,
                                &mut norm_depth,
                                &poses,
                                FILTER_REPROJ_SLACK * config.max_reprojection_error,
                                config.merge_radius_rel * config.max_reprojection_error * zmed,
                            );
                        }
                    }
                    triangulate_new(&mut point3d, &norm, &poses, &idcam, &tcfg);
                    // `max(next_ba + 1.0)` so a set that grew by less than one whole camera per
                    // trigger (small maps) still advances the threshold and cannot re-fire on
                    // every registration.
                    next_ba = (registered_now(&poses) as f64 * BA_IMAGES_RATIO).max(next_ba + 1.0);
                }
            }
            // Failed outright, or its consensus was too thin: retry once the cloud has grown.
            None => {
                pnp_failed.insert(c);
            }
        }
    }

    // One last filter before the terminal solve, so it is not asked to reconcile sightings that
    // every earlier solve already failed to — and so the published `observations` carry only
    // evidence the final geometry actually supports.
    filter_points(
        &mut point3d,
        &mut norm,
        &mut norm_depth,
        &poses,
        FILTER_REPROJ_SLACK * config.max_reprojection_error,
    );
    // Every camera now holds its final pose, so this is where the most previously-untestable
    // evidence becomes testable: a sighting dropped against an early pose is judged one last time
    // against the pose the map actually ships.
    if config.complete_tracks {
        complete_tracks(
            &point3d,
            &mut norm,
            &mut norm_depth,
            &norm0,
            &norm_depth0,
            &poses,
            FILTER_REPROJ_SLACK * config.max_reprojection_error,
        );
    }

    // --- Bundle adjustment: all track points free, the reference camera (a0) fixed to anchor gauge. ---
    // Iterate the triangulated points in TRACK ORDER, not `HashMap` order. Rust's default hasher
    // is randomised per process, so `for (ti, p) in &point3d` yields a different order every run.
    // That was harmless while these only fed bundle adjustment and the RMS statistics, all
    // permutation-invariant -- but `Reconstruction` publishes `points`, `point_track_id` and
    // `observations`, so the order becomes part of the contract. A caller diffing two runs,
    // snapshotting a map, or holding point indices across a rebuild would otherwise see churn
    // with no cause. This file already guards the same hazard for the bootstrap pair.
    let mut track_ids: Vec<usize> = point3d.keys().copied().collect();
    track_ids.sort_unstable();

    // `pt_index[ti]` is `ti`'s position in `track_ids`: stable across BA rounds, so the published
    // `points` order matches the BA input order in every one of them.
    let pt_index: HashMap<usize, usize> =
        track_ids.iter().enumerate().map(|(i, t)| (*t, i)).collect();
    let point_track_id: Vec<usize> = track_ids.clone();
    let mut kept_obs: Vec<Observation> = Vec::new();
    for ti in &track_ids {
        for (c, _) in norm[*ti].iter() {
            if poses[*c].is_none() {
                continue;
            }
            // BY CAMERA ID, not by position. `norm[ti]` began as a positional map of
            // `tracks[ti].obs` and `pixel: tracks[ti].obs[j].1` relied on that — but
            // `filter_points` removes sightings from `norm` and nothing removes them from `tracks`,
            // so after any pruning index `j` addresses a different observation and the published
            // pixel belongs to another view (often the outlier just rejected). A track holds at
            // most one observation per camera, so the id is unambiguous.
            let Some((_, pixel)) = tracks[*ti].obs.iter().find(|(cc, _)| cc == c) else {
                continue;
            };
            kept_obs.push(Observation {
                view: *c,
                point: pt_index[ti],
                pixel: *pixel,
            });
        }
    }

    let mut res = global_ba(
        &poses,
        &point3d,
        &track_ids,
        &norm,
        &norm_depth,
        &idcam,
        a0,
        config,
        None,
        config.max_iterations,
    )?;

    // --- Alternating intrinsics refinement (COLMAP does it INSIDE its BA; this equivalent needs no
    // solver surgery). The solver sees NORMALIZED coordinates, so a focal error is one global scale
    // `gamma` and distortion a radial/tangential polynomial — linear in the stacked unknowns.
    let camera_correction = if config.refine_intrinsics {
        let fit = fit_camera_correction(&res, &pt_index, &norm, &poses);
        if let Some((gamma, k1, k2, p1, p2)) = fit {
            for track in norm.iter_mut() {
                for (_, n) in track.iter_mut() {
                    let (x, y) = (n.x, n.y);
                    let r2 = x * x + y * y;
                    let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
                    n.x = gamma * (x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x));
                    n.y = gamma * (y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y);
                }
            }
            // Feed the first solve's state back in and RE-SOLVE against the corrected observations:
            // without this the refinement would be reported but never reach the returned poses.
            for (c, p) in poses.iter_mut().enumerate() {
                if p.is_some() {
                    *p = Some(res.poses[c]);
                }
            }
            for (ti, pidx) in &pt_index {
                if let Some(v) = res.points.get(*pidx) {
                    point3d.insert(*ti, *v);
                }
            }
            res = global_ba(
                &poses,
                &point3d,
                &track_ids,
                &norm,
                &norm_depth,
                &idcam,
                a0,
                config,
                None,
                config.max_iterations,
            )?;
        }
        fit
    } else {
        None
    };

    let registered: Vec<bool> = poses.iter().map(|p| p.is_some()).collect();

    // --- Metric scale from the tag: triangulate its corners, compare to the known side length. ---
    // Scaling the world by `s` (points ×s AND world→cam translation ×s) leaves reprojection unchanged,
    // so we compute per-camera stats on the UNSCALED BA result and only scale the output translation.
    // `None` when the tag could NOT anchor the scale -- unset size, seen by fewer than two
    // REGISTERED cameras, or its corners failed to triangulate. Those are ordinary outcomes, not
    // corner cases: a camera that cannot register via PnP is skipped by design, so "a tag was
    // supplied but the views that saw it never registered" happens in normal use. Reporting
    // `ScaleSource::Tag` off `tags_for_scale.first()` would then claim metric for a map still at
    // the identity scale -- exactly the ambiguity this type exists to remove.
    let anchored = tag_scale(
        tags_for_scale,
        cameras,
        &res.poses,
        &registered,
        &idcam,
        &tcfg,
        config.tag_size_m,
    );
    let scale = anchored.unwrap_or(1.0);

    // Pixels-per-normalized-unit correction: `refine_intrinsics` rewrote `norm` into the true
    // camera's frame (fx_true = fx_assumed / gamma) but the RMS below uses the ASSUMED fx.
    let focal_scale = camera_correction.map_or(1.0, |(gamma, ..)| 1.0 / gamma.max(1e-9));

    // Per-camera reprojection RMS (pixels); analytical covariance is tag-oriented so stays `None`.
    // Use the BA-optimized points (`res.points`), NOT the pre-BA cloud: BA moves points as free
    // variables, so evaluating the pre-BA cloud under post-BA poses would report a stale residual.
    // `pt_index` indexes both identically (same order as the BA input).
    let per_camera = (0..n_cams)
        .map(|c| {
            feature_stats(
                focal_scale,
                c,
                &res.poses,
                &registered,
                &res.points,
                &pt_index,
                &norm,
                &cameras[c],
            )
        })
        .collect();
    let reproj_rmse_px = global_reproj_rmse(
        focal_scale,
        &res.poses,
        &registered,
        &res.points,
        &pt_index,
        &norm,
        cameras,
    );

    // Output `T_world_cam` (camera→world): invert the metric world→cam (translation scaled).
    let out_poses: Vec<Option<Pose3d>> = (0..n_cams)
        .map(|c| {
            registered[c].then(|| {
                Pose3d::new(res.poses[c].rotation, res.poses[c].translation * scale).inverse()
            })
        })
        .collect();
    // Points share the world frame with the output poses, so they take the same metric scale.
    // Paired with their track id here rather than returned as parallel vectors, so the two cannot
    // come apart in a caller's hands.
    let out_points: Vec<Point> = res
        .points
        .iter()
        .zip(&point_track_id)
        .map(|(p, &track_id)| Point {
            position: *p * scale,
            track_id,
        })
        .collect();

    Ok(Reconstruction {
        views: out_poses,
        points: out_points,
        observations: kept_obs,
        reproj_rmse_px,
        per_view: per_camera,
        scale: match (anchored, tags_for_scale.first()) {
            (Some(_), Some(t)) => ScaleSource::Tag {
                id: t.tag_id,
                size_m: config.tag_size_m,
            },
            _ => ScaleSource::UpToScale,
        },
        camera_correction,
    })
}
/// The reference reprojection sigma every prior family is deflated by, in NORMALISED units.
///
/// Bundle adjustment here runs against `PinholeCamera::IDENTITY`, so its residuals are in
/// normalised units while caller prior sigmas are physical; dividing by this puts every family on
/// one scale, which is what lets a single Huber knee gate them. `max_reprojection_error / 2`
/// because that threshold reads as 2σ. Derived ONCE because four call sites must agree exactly —
/// depth, up, motion, and the knee gating all three — and drift mis-weights them silently.
fn reproj_sigma(config: &ReconstructionConfig) -> f64 {
    (config.max_reprojection_error / 2.0).max(1e-6)
}

/// Depth measurement + already-deflated sigma for observation `j` of track `ti`, or `(None, 0.0)`.
///
/// DEFLATED by [`reproj_sigma`], not the raw metric sigma: reprojection residuals are unwhitened
/// normalized values (implicit σ = a whole focal length!) while depth residuals divide by theirs,
/// so the metric sigma goes depth-dominated — measured, `rel_sigma` 5.0 still halved registration.
fn depth_fields(
    norm_depth: &[Vec<Option<f32>>],
    ti: usize,
    j: usize,
    config: &ReconstructionConfig,
) -> (Option<f32>, f32) {
    let rel_sigma = config.depth_prior_rel_sigma;
    match norm_depth.get(ti).and_then(|t| t.get(j)).copied().flatten() {
        Some(d) if rel_sigma > 0.0 && d > 0.0 => {
            let sigma_r = reproj_sigma(config) as f32;
            (Some(d), (rel_sigma as f32) * d / sigma_r)
        }
        _ => (None, 0.0),
    }
}

/// Robust per-view depth gauge: the median of `z_map / d_pred` over that view's depth
/// observations, re-gauged against the median view and clamped. `1.0` where a view lacks enough
/// pairs to fit.
///
/// Per view rather than one global scale because learned depth is not gauge-stable frame to frame:
/// on forward motion a per-frame scale error is indistinguishable from along-axis translation, so a
/// global scale hands every wander of the network to the trajectory. Scale only, NOT the affine
/// `s·d + t`: a free intercept absorbs bas-relief compression instead of correcting it (measured —
/// sparse depth spanned 1.13 across a view where the network saw 2.13) and makes ABSOLUTE scale
/// unobservable from depth. Normalising by the median re-gauges views without moving the map.
fn fit_depth_scales(
    poses: &[Option<Pose3d>],
    point3d: &HashMap<usize, Vec3F64>,
    track_ids: &[usize],
    norm: &[Vec<(usize, Vec2F64)>],
    norm_depth: &[Vec<Option<f32>>],
    n_cams: usize,
) -> Vec<f64> {
    /// Below this many pairs a median is noise, and a wrong gauge is worse than the global one.
    const MIN_PAIRS: usize = 12;

    let mut per_cam: Vec<Vec<f64>> = vec![Vec::new(); n_cams];
    for ti in track_ids {
        let Some(p) = point3d.get(ti) else { continue };
        for (j, (c, _)) in norm[*ti].iter().enumerate() {
            // `.get(ti)`, not `[ti]`: a wrong-shaped `obs_depth` must yield `None` per lookup, not
            // panic — indexing here panicked on depth supplied for only the first N tracks.
            let (Some(pose), Some(d)) = (
                &poses[*c],
                norm_depth
                    .get(*ti)
                    .and_then(|t| t.get(j))
                    .copied()
                    .flatten(),
            ) else {
                continue;
            };
            let z = pose.transform_point(p).z;
            if z > 1e-9 && d > 0.0 {
                per_cam[*c].push(z / d as f64); // map units per network unit
            }
        }
    }
    // Median, not mean: depth networks hallucinate at occlusions and on mirrors — a fat tail.
    let median = |v: &mut Vec<f64>| -> Option<f64> {
        if v.len() < MIN_PAIRS {
            return None;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = v[v.len() / 2];
        (m.is_finite() && m > 1e-9).then_some(m)
    };
    let mut scales: Vec<Option<f64>> = per_cam.iter_mut().map(median).collect();

    let mut fitted: Vec<f64> = scales.iter().flatten().copied().collect();
    if fitted.len() < 2 {
        return vec![1.0; n_cams];
    }
    fitted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let anchor = fitted[fitted.len() / 2];
    for s in scales.iter_mut().flatten() {
        *s /= anchor;
    }
    // A wildly-off scale means a broken pose or depth map, not a gauge offset: fall back to neutral.
    scales
        .into_iter()
        .map(|s| match s {
            Some(v) if (0.5..2.0).contains(&v) => v,
            _ => 1.0,
        })
        .collect()
}

/// Per-view up priors, or `None` when [`ReconstructionConfig::up_prior_sigma`] is off.
///
/// The camera-frame direction asserted to be up is image-up `(0, −1, 0)` — "held roughly upright".
/// World up is a GAUGE choice and has to agree with the anchor camera `a0`, whose pose is held
/// fixed: `up_world = R_a0ᵀ · (0,−1,0)`. Any other choice fights the one pose the solve cannot move.
fn up_priors(
    poses: &[Option<Pose3d>],
    a0: usize,
    config: &ReconstructionConfig,
) -> Option<Vec<Option<BaPosePrior>>> {
    if config.up_prior_sigma <= 0.0 {
        return None;
    }
    const UP_CAM: [f64; 3] = [0.0, -1.0, 0.0];
    let up_world: [f32; 3] = match poses.get(a0).and_then(|p| *p) {
        Some(pa) => {
            let w = pa.rotation.transpose() * Vec3F64::new(UP_CAM[0], UP_CAM[1], UP_CAM[2]);
            [w.x as f32, w.y as f32, w.z as f32]
        }
        None => [0.0, -1.0, 0.0],
    };
    // Deflated into reprojection units, as in `depth_fields`. Raw, the prior was `1/σ_r` times
    // stiffer than every other term in the solve — 360× at fx 1440 with an 8 px threshold.
    let sigma_r = reproj_sigma(config);
    Some(
        poses
            .iter()
            .map(|p| {
                p.as_ref().map(|_| {
                    BaPosePrior::orientation_only(
                        up_world,
                        (config.up_prior_sigma / sigma_r) as f32,
                    )
                })
            })
            .collect(),
    )
}

/// Constant-velocity motion priors over consecutive REGISTERED triplets, or `None` when off.
///
/// Consecutive in VIEW-INDEX order (video keyframes are time-ordered), with triplets spanning more
/// than [`MAX_TRIPLET_SPAN`] indices skipped: a bridge across a long unregistered stretch is not a
/// constant-velocity hypothesis worth asserting. Sigmas are deflated like the other priors.
fn motion_priors_for(
    poses: &[Option<Pose3d>],
    config: &ReconstructionConfig,
) -> Option<Vec<BaMotionPrior>> {
    /// Widest index gap a triplet may span before it stops being a plausible motion hypothesis.
    const MAX_TRIPLET_SPAN: usize = 12;

    if config.motion_prior_sigma <= 0.0 {
        return None;
    }
    let sigma_r = reproj_sigma(config);
    let sp = (config.motion_prior_sigma / sigma_r) as f32;
    let so = (0.5 * config.motion_prior_sigma / sigma_r) as f32;
    let reg: Vec<usize> = poses
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|_| i))
        .collect();
    let out: Vec<BaMotionPrior> = reg
        .windows(3)
        .filter(|w| w[2] - w[0] <= MAX_TRIPLET_SPAN)
        .map(|w| BaMotionPrior {
            i0: w[0],
            i1: w[1],
            i2: w[2],
            alpha: (w[1] - w[0]) as f32 / (w[2] - w[0]) as f32,
            position_sigma: sp,
            orientation_sigma: so,
        })
        .collect();
    (!out.is_empty()).then_some(out)
}

/// One bundle adjustment over the triangulated points and the registered views, with the anchor
/// camera `a0` fixed to hold the gauge. Callable more than once on the same problem, which is what
/// `refine_intrinsics` needs after it rewrites `norm` in place, and what the periodic refinement
/// during growth needs by construction.
///
/// `free` is `None` for a global solve (every registered camera optimised) or `Some(set)` to pin
/// everything outside a local window. `max_iterations` is a parameter rather than
/// [`ReconstructionConfig::max_iterations`] because the mid-growth local solves want far fewer.
#[allow(clippy::too_many_arguments)]
fn global_ba(
    poses: &[Option<Pose3d>],
    point3d: &HashMap<usize, Vec3F64>,
    track_ids: &[usize],
    norm: &[Vec<(usize, Vec2F64)>],
    norm_depth: &[Vec<Option<f32>>],
    idcam: &PinholeCamera,
    a0: usize,
    config: &ReconstructionConfig,
    free: Option<&HashSet<usize>>,
    max_iterations: usize,
) -> Result<BaResult, CalibError> {
    // Re-fit the per-view depth gauge against the CURRENT geometry before every solve. Alternating
    // rather than joint: the scales are closed-form medians, so each pass refines the other.
    let depth_scale = if config.depth_per_keyframe_scale {
        fit_depth_scales(poses, point3d, track_ids, norm, norm_depth, poses.len())
    } else {
        vec![1.0; poses.len()]
    };
    let mut points: Vec<Vec3F64> = Vec::with_capacity(track_ids.len());
    let mut obs: Vec<BaObservation> = Vec::new();
    for (pidx, ti) in track_ids.iter().enumerate() {
        // Fail loudly rather than skip: a skip shifts every later `pidx` off its point.
        let p = point3d.get(ti).ok_or_else(|| {
            CalibError::BundleAdjust(format!("track {ti} has no triangulated point"))
        })?;
        points.push(*p);
        // In a windowed solve, a point no free camera observes has every one of its observations
        // pinned on both sides: the whole residual block is a constant that costs assembly and
        // moves nothing. Skipping it is what keeps a local BA's size proportional to the window
        // rather than to the whole clip. The entry stays in `points` so `pidx` still lines up with
        // `track_ids` for the caller's write-back — the solver simply reports it back unchanged.
        if free.is_some_and(|f| {
            !norm[*ti]
                .iter()
                .any(|(c, _)| poses[*c].is_some() && f.contains(c))
        }) {
            continue;
        }
        for (j, (c, nrm)) in norm[*ti].iter().enumerate() {
            if poses[*c].is_none() {
                continue;
            }
            let (depth_meas, depth_sigma) = depth_fields(norm_depth, *ti, j, config);
            obs.push(BaObservation {
                pose_idx: *c,
                point_idx: pidx,
                // Reference camera fixed → gauge anchor; outside a local window, pinned so the
                // window is fitted TO the surrounding structure instead of dragging it along.
                fixed_pose: *c == a0 || free.is_some_and(|f| !f.contains(c)),
                pixel: [nrm.x as f32, nrm.y as f32],
                fixed_point: false,
                // This view's own gauge baked in: the residual measures shape, not scale.
                depth_meas: depth_meas.map(|d| d * depth_scale[*c] as f32),
                depth_sigma,
            });
        }
    }
    let poses_ba: Vec<Pose3d> = poses
        .iter()
        .map(|p| p.unwrap_or(Pose3d::IDENTITY))
        .collect();
    bundle_adjust_schur_with_all_priors(
        &poses_ba,
        &points,
        &obs,
        idcam,
        &BaParams {
            max_iterations,
            robust: RobustKernelKind::Huber,
            robust_scale_sq: config.robust_scale_sq,
            // Depth AND motion residuals are deflated by `reproj_sigma`, so their Huber knee is
            // 1.345 × that scale, squared. Gated on EITHER family: `ba_schur` uses one knee for
            // both, so motion priors without depth would silently gate at ~2σ instead of 1.345σ.
            depth_robust_scale_sq: if config.depth_prior_rel_sigma > 0.0
                || config.motion_prior_sigma > 0.0
            {
                let sr = (1.345 * reproj_sigma(config)) as f32;
                sr * sr
            } else {
                0.0
            },
            sparse_reduced_system: config.sparse_reduced_system,
            ..Default::default()
        },
        up_priors(poses, a0, config).as_deref(),
        motion_priors_for(poses, config).as_deref(),
    )
    .map_err(|e| CalibError::BundleAdjust(format!("{e:?}")))
}

/// Closed-form OpenCV-model intrinsics correction `(gamma, k1, k2, p1, p2)` fitted against the
/// current reconstruction, or `None` when singular or outside its sanity bounds. Linear in the
/// stacked unknowns `beta = (gamma, gamma·k1, gamma·k2, gamma·p1, gamma·p2)`:
///
/// ```text
///   u_x = b0·x + b1·x·r² + b2·x·r⁴ + b3·(2xy)     + b4·(r²+2x²)
///   u_y = b0·y + b1·y·r² + b2·y·r⁴ + b3·(r²+2y²)  + b4·(2xy)
/// ```
///
/// so it costs one least squares. The tangential terms are what a focal-only fit can never see;
/// forcing a decentred lens into `k1` leaves a residue that looks like scene curvature.
fn fit_camera_correction(
    res: &BaResult,
    pt_index: &HashMap<usize, usize>,
    norm: &[Vec<(usize, Vec2F64)>],
    poses: &[Option<Pose3d>],
) -> Option<(f64, f64, f64, f64, f64)> {
    let mut ata = [[0.0f64; 5]; 5];
    let mut atb = [0.0f64; 5];
    // DETERMINISM: float addition is not associative, so accumulation ORDER changes the fitted
    // correction, and Rust randomises `HashMap` iteration per process. Sort, or the map varies.
    let mut ordered: Vec<(&usize, &usize)> = pt_index.iter().collect();
    ordered.sort_unstable_by_key(|(ti, _)| **ti);
    for (ti, pidx) in ordered {
        let Some(p) = res.points.get(*pidx) else {
            continue;
        };
        for (c, n) in &norm[*ti] {
            if poses[*c].is_none() {
                continue;
            }
            let pc = res.poses[*c].transform_point(p);
            if pc.z <= 1e-9 {
                continue;
            }
            let (u, v) = (pc.x / pc.z, pc.y / pc.z);
            let (x, y) = (n.x, n.y);
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let rows = [
                ([x, x * r2, x * r4, 2.0 * x * y, r2 + 2.0 * x * x], u),
                ([y, y * r2, y * r4, r2 + 2.0 * y * y, 2.0 * x * y], v),
            ];
            for (basis, target) in rows {
                for i in 0..5 {
                    for j in 0..5 {
                        ata[i][j] += basis[i] * basis[j];
                    }
                    atb[i] += basis[i] * target;
                }
            }
        }
    }
    // Full 5-parameter fit first, falling back to the (gamma, k1) subproblem: thin tracks make the
    // r⁴ and tangential columns nearly collinear on a narrow-FOV rig.
    let full = solve_sym5(&ata, &atb).and_then(|b| {
        let gamma = b[0];
        if gamma.abs() < 1e-9 {
            return None;
        }
        let (k1, k2, p1, p2) = (b[1] / gamma, b[2] / gamma, b[3] / gamma, b[4] / gamma);
        ((0.7..1.3).contains(&gamma)
            && (-0.3..0.3).contains(&k1)
            && (-0.1..0.1).contains(&k2)
            && (-0.05..0.05).contains(&p1)
            && (-0.05..0.05).contains(&p2))
        .then_some((gamma, k1, k2, p1, p2))
    });
    full.or_else(|| {
        let det = ata[0][0] * ata[1][1] - ata[0][1] * ata[0][1];
        if det.abs() <= 1e-12 {
            return None;
        }
        let gamma = (atb[0] * ata[1][1] - atb[1] * ata[0][1]) / det;
        let gk1 = (atb[1] * ata[0][0] - atb[0] * ata[0][1]) / det;
        let k1 = if gamma.abs() > 1e-9 { gk1 / gamma } else { 0.0 };
        // Sanity bounds: a fit outside them means the MAP is wrong, not the camera, and applying
        // it would let geometry errors masquerade as optics.
        ((0.7..1.3).contains(&gamma) && (-0.3..0.3).contains(&k1))
            .then_some((gamma, k1, 0.0, 0.0, 0.0))
    })
}

/// Solve the symmetric PSD `5×5` system `A x = b` by Gaussian elimination with partial pivoting.
/// `None` when a pivot collapses — collinear distortion columns, so the caller falls back to two.
fn solve_sym5(a: &[[f64; 5]; 5], b: &[f64; 5]) -> Option<[f64; 5]> {
    let mut m = [[0.0f64; 6]; 5];
    for i in 0..5 {
        m[i][..5].copy_from_slice(&a[i]);
        m[i][5] = b[i];
    }
    for col in 0..5 {
        let piv = (col..5).max_by(|&i, &j| {
            m[i][col]
                .abs()
                .partial_cmp(&m[j][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        for row in (col + 1)..5 {
            let f = m[row][col] / m[col][col];
            for k in col..6 {
                m[row][k] -= f * m[col][k];
            }
        }
    }
    let mut x = [0.0f64; 5];
    for i in (0..5).rev() {
        let mut s = m[i][5];
        for j in (i + 1)..5 {
            s -= m[i][j] * x[j];
        }
        x[i] = s / m[i][i];
    }
    Some(x)
}

/// Choose the bootstrap pair and recover its two-view pose, as `(a0, b0, T_b0_a0)`.
///
/// Picking the pair with the MOST shared tracks is the natural choice for a rig and the wrong one
/// for a video: consecutive views overlap most precisely because they are closest together, so
/// "most matches" systematically selects the SMALLEST baseline in the sequence — the worst
/// conditioning an essential matrix can be given. The seed is not a recoverable mistake either,
/// because it fixes the world gauge and every later view registers by PnP against the cloud
/// triangulated from it: a seed with no parallax triangulates nothing, nothing clears the growth
/// loop's admission bar, and the reconstruction stops at the two seed views.
///
/// So: take the best-supported candidates, actually SOLVE each one, and choose on geometry —
/// median triangulation angle first (a floor that rejects the degenerate low-parallax pairs), then
/// forward-motion dominance, then cheirality support. The whole ordering is a total order with a
/// deterministic `(a, b)` tie-break, because `HashMap` iteration order must not decide the seed.
///
/// `ranked: false` restricts the candidate list to the single best-supported pair, which is
/// exactly the argmax-shared-tracks selection this replaced. It is private and exists so
/// `ranked_bootstrap_registers_more_views_than_the_most_shared_tracks_seed` can measure the two
/// selections against each other with everything else held fixed.
fn select_bootstrap_pair(
    tracks: &[FeatureTrack],
    norm: &[Vec<(usize, Vec2F64)>],
    cameras: &[PinholeCamera],
    idcam: &PinholeCamera,
    ranked: bool,
) -> Result<(usize, usize, Pose3d), CalibError> {
    // Count shared tracks per camera pair.
    let mut pair_count: HashMap<(usize, usize), usize> = HashMap::new();
    for obs in norm {
        for i in 0..obs.len() {
            for j in (i + 1)..obs.len() {
                let (a, b) = (obs[i].0.min(obs[j].0), obs[i].0.max(obs[j].0));
                *pair_count.entry((a, b)).or_insert(0) += 1;
            }
        }
    }
    if pair_count.is_empty() {
        return Err(CalibError::NoReferenceTagView);
    }
    // Best-supported first, deterministic tie-break on the smaller `(a, b)`.
    let mut by_count: Vec<((usize, usize), usize)> = pair_count.into_iter().collect();
    by_count.sort_by(|x, y| y.1.cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
    let n_candidates = if ranked {
        BOOTSTRAP_CANDIDATES.min(by_count.len())
    } else {
        1
    };

    // `(a, b, T_b_a, cheirality-valid count, median parallax in degrees, RH ratio)`.
    let mut seeds: Vec<(usize, usize, Pose3d, usize, f64, Option<f64>)> = Vec::new();
    for ((ca, cb), n_shared) in &by_count[..n_candidates] {
        // The 5-point RANSAC needs 8 correspondences; a pair below that cannot be solved at all.
        if *n_shared < 8 {
            continue;
        }
        if let Some((pose, cnt, par, rh)) = try_bootstrap_pair(*ca, *cb, tracks, cameras, idcam) {
            seeds.push((*ca, *cb, pose, cnt, par, rh));
        }
    }
    if seeds.is_empty() {
        return Err(CalibError::BundleAdjust(
            "no bootstrap pair produced a valid two-view pose".into(),
        ));
    }

    // Prefer a pair that clears the parallax floor; among those, the least forward-dominated; among
    // those, the most cheirality-valid points.
    //
    // Each criterion DEMOTES rather than rejects, and that is deliberate: a handheld walkthrough is
    // forward-motion dominated throughout and can be near-degenerate throughout, so a hard gate can
    // empty the candidate list on exactly the clips that need a seed most. A weak seed still beats
    // no map, provided the choice is the best one available.
    //
    // The `(a, b)` tie-break is not cosmetic — without a total order, equal-scoring seeds would
    // reintroduce the run-to-run variability the count sort above exists to remove.
    seeds.sort_by(|a, b| {
        let (a_par, b_par) = (a.4 >= MIN_SEED_PARALLAX_DEG, b.4 >= MIN_SEED_PARALLAX_DEG);
        let (a_fwd, b_fwd) = (
            a.2.translation.z.abs() <= MAX_SEED_FORWARD,
            b.2.translation.z.abs() <= MAX_SEED_FORWARD,
        );
        b_par
            .cmp(&a_par)
            .then_with(|| b_fwd.cmp(&a_fwd))
            .then_with(|| b.3.cmp(&a.3))
            .then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
    });

    let (a0, b0, pose, cnt, par, rh) = seeds[0];
    if let Some(rh) = rh {
        eprintln!(
            "[calib] seed pair ({a0},{b0}) of {} candidates: cheirality={cnt} \
             median_parallax={par:.2} deg RH={rh:.3}{}",
            seeds.len(),
            if rh > MAX_SEED_H_RATIO {
                "  (DEGENERATE — no clean pair available)"
            } else {
                ""
            }
        );
    }
    Ok((a0, b0, pose))
}

/// Solve one candidate bootstrap pair.
///
/// Returns `(T_a_to_b with unit translation, cheirality-valid count, median triangulation angle in
/// degrees, homography-vs-fundamental ratio)`, or `None` when the pair has too few
/// correspondences, no decomposition survives the cheirality vote, or two decompositions survive it
/// about equally well.
///
/// The median angle is what distinguishes a usable seed from a degenerate one: two nearly
/// coincident views still produce a confident-looking essential matrix, whose triangulated depths
/// are meaningless.
///
/// The RH ratio is `None` unless `KORNIA_CALIB_DEBUG` is set — see the note at its computation.
fn try_bootstrap_pair(
    a0: usize,
    b0: usize,
    tracks: &[FeatureTrack],
    cameras: &[PinholeCamera],
    idcam: &PinholeCamera,
) -> Option<(Pose3d, usize, f64, Option<f64>)> {
    let (mut x1, mut x2) = (Vec::new(), Vec::new());
    for t in tracks {
        let pa = t.obs.iter().find(|(c, _)| *c == a0);
        let pb = t.obs.iter().find(|(c, _)| *c == b0);
        if let (Some((_, ua)), Some((_, ub))) = (pa, pb) {
            x1.push(*ua);
            x2.push(*ub);
        }
    }
    if x1.len() < 8 {
        return None;
    }

    // Calibrated 5-point Nistér essential + explicit cheirality vote. We drive the essential arm
    // directly (not the full `TwoViewEstimator`, whose F/H model selection can pick a degenerate
    // homography on a converging rig): a calibrated rig always wants the essential.
    let n1: Vec<Vec2F64> = x1.iter().map(|p| cameras[a0].normalize(*p)).collect();
    let n2: Vec<Vec2F64> = x2.iter().map(|p| cameras[b0].normalize(*p)).collect();
    // `ransac_essential_5pt` only normalizes with K⁻¹ — it does NOT undistort. Feed UNDISTORTED
    // pixels so the essential arm matches the undistorted-normalized coords every other stage uses
    // (cheirality vote, triangulation, BA); K⁻¹·undistort(px) == normalize(px). The 2.0 px threshold
    // stays valid in undistorted pixel space. (Raw pixels here would bias R/t on a distorted lens.)
    let x1u: Vec<Vec2F64> = x1.iter().map(|p| cameras[a0].undistort(p.x, p.y)).collect();
    let x2u: Vec<Vec2F64> = x2.iter().map(|p| cameras[b0].undistort(p.x, p.y)).collect();

    // Model-selection score: DIAGNOSTIC ONLY, and computed only when someone is looking.
    //
    // It is not a ranking criterion — among pairs that are all somewhat planar, "least planar" is
    // not "best conditioned", and enforcing it as a hard reject fails outright on handheld
    // sequences whose every pair is near-degenerate. What it does do is identify that regime, which
    // is the single most useful thing to know when a map comes out mirrored, so it is worth
    // reporting. Since it costs two extra RANSAC fits per candidate and cannot change the outcome,
    // it is skipped entirely outside a debug run. The per-pair RNG seed keeps it reproducible
    // without making every pair share one sampler sequence.
    let rh = std::env::var_os("KORNIA_CALIB_DEBUG").and_then(|_| {
        let seed_rng = 0xB007 ^ ((a0 as u64) << 16) ^ b0 as u64;
        homography_vs_fundamental_ratio(&x1u, &x2u, seed_rng)
    });

    let rp = TvRp {
        max_iterations: 2000,
        threshold: 2.0,
        min_inliers: 8,
        random_seed: Some(0),
        refit: true,
    };
    let ess = ransac_essential_5pt(
        &x1u,
        &x2u,
        &cameras[a0].intrinsic_matrix(),
        &cameras[b0].intrinsic_matrix(),
        &rp,
    )
    .ok()?;
    let cands = decompose_essential(&ess.model)?;

    // Lenient triangulation for the cheirality vote (count points in front of BOTH cameras).
    let tvote = TriangulationConfig {
        min_parallax_deg: 0.0,
        max_reprojection_error: 1e9,
        min_cheirality_count: 0,
        ..Default::default()
    };
    let mut best = (0usize, Pose3d::IDENTITY);
    let mut runner_up = 0usize;
    for (r, t) in cands {
        let pb = Pose3d::new(r, t); // world(=a0) → b, unit translation
        let mut cnt = 0usize;
        for k in 0..n1.len() {
            if let Ok(pts) = triangulate_matched_points(
                &[n1[k]],
                &[n2[k]],
                &Pose3d::IDENTITY,
                &pb,
                idcam,
                &tvote,
            ) {
                if let Some(p) = pts.first() {
                    if p.position.z > 0.0 && pb.transform_point(&p.position).z > 0.0 {
                        cnt += 1;
                    }
                }
            }
        }
        if cnt > best.0 {
            runner_up = best.0;
            best = (cnt, pb);
        } else if cnt > runner_up {
            runner_up = cnt;
        }
    }
    if best.0 == 0 {
        return None;
    }

    // Twofold-ambiguity guard.
    //
    // The cheirality vote picks whichever of the four essential decompositions puts the most points
    // in front of both cameras. When the scene is planar or the motion is near-pure rotation, two
    // decompositions score almost equally — the classic twofold planar ambiguity — and the winner
    // is then decided by noise. Committing to it yields a MIRRORED reconstruction that is
    // internally self-consistent: bundle adjustment happily drives its reprojection error down, so
    // the map looks healthy by every internal metric while being structurally wrong.
    //
    // `TriangulationConfig::cheirality_ambiguity_max` encodes the threshold, but the manual vote
    // above never consults it, so it is applied explicitly here. Rejecting the pair is cheap: the
    // caller simply ranks the next candidate.
    if (runner_up as f64) >= tvote.cheirality_ambiguity_max * best.0 as f64 {
        if std::env::var_os("KORNIA_CALIB_DEBUG").is_some() {
            eprintln!(
                "[calib] seed ({a0},{b0}) REJECTED: ambiguous decomposition \
                 (best={} runner_up={} ratio={:.2})",
                best.0,
                runner_up,
                runner_up as f64 / best.0 as f64
            );
        }
        return None;
    }

    // Median triangulation angle for the chosen pose: the angle between the two bearing rays, with
    // the second rotated into the first camera's frame.
    let r_t = best.1.rotation.transpose();
    let mut angles: Vec<f64> = Vec::with_capacity(n1.len());
    for k in 0..n1.len() {
        let d1 = Vec3F64::new(n1[k].x, n1[k].y, 1.0).normalize();
        let d2 = r_t * Vec3F64::new(n2[k].x, n2[k].y, 1.0).normalize();
        let c = d1.dot(d2).clamp(-1.0, 1.0);
        angles.push(c.acos().to_degrees());
    }
    angles.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
    let median = angles.get(angles.len() / 2).copied().unwrap_or(0.0);

    Some((best.1, best.0, median, rh))
}

/// ORB-SLAM2's homography-vs-fundamental model-selection score, `RH = SH / (SH + SF)`.
///
/// Both models are scored with a symmetric transfer error, accumulating `threshold - chi2` per
/// inlier observation so a model is rewarded for explaining points *tightly*, not merely for
/// counting them. The chi-square thresholds and the score cap are ORB-SLAM2's
/// (`Initializer::CheckHomography` / `CheckFundamental`): 5.991 for the homography (2 degrees of
/// freedom, point-to-point) and 3.841 for the fundamental (1 DOF, point-to-line), with the
/// fundamental's per-inlier score capped against 5.991 so the two totals live on the same scale.
///
/// A high ratio means a homography explains the correspondences about as well as the epipolar
/// geometry does, which happens exactly when the scene is planar or the motion is near-pure
/// rotation. In that regime the essential matrix is not merely imprecise but ambiguous, and
/// committing to one of its decompositions can produce a mirrored reconstruction that bundle
/// adjustment then makes internally self-consistent — undetectable from reprojection error alone.
///
/// Returns `None` when either model cannot be estimated, or when neither explains anything.
fn homography_vs_fundamental_ratio(x1: &[Vec2F64], x2: &[Vec2F64], seed: u64) -> Option<f64> {
    /// Chi-square 95% bound at 2 DOF: the homography's point-to-point transfer error.
    const TH_H: f64 = 5.991;
    /// Chi-square 95% bound at 1 DOF: the fundamental's point-to-epipolar-line distance.
    const TH_F: f64 = 3.841;
    /// The fundamental's per-inlier score is capped against the HOMOGRAPHY's bound, so a 1-DOF
    /// inlier and a 2-DOF inlier contribute comparably and `RH` is not biased by the model's
    /// dimension. Same asymmetry as the reference implementation.
    const SCORE_CAP: f64 = 5.991;
    /// Assumed keypoint localization sigma, in pixels.
    const INV_SIGMA_SQ: f64 = 1.0;

    if x1.len() < 8 {
        return None;
    }
    let rp = TvRp {
        max_iterations: 2000,
        threshold: 2.0,
        min_inliers: 8,
        random_seed: Some(seed),
        refit: true,
    };

    let h = ransac_homography(x1, x2, &rp).ok()?;
    let f = ransac_fundamental(x1, x2, &rp).ok()?;

    // --- Homography score: symmetric transfer error, both directions. ---
    let hm = h.model;
    let hinv = hm.inverse();
    let mut s_h = 0.0;
    for (p1, p2) in x1.iter().zip(x2.iter()) {
        // 1 -> 2
        let d = hm * Vec3F64::new(p1.x, p1.y, 1.0);
        if d.z.abs() > 1e-12 {
            let (u, v) = (d.x / d.z, d.y / d.z);
            let chi = ((p2.x - u).powi(2) + (p2.y - v).powi(2)) * INV_SIGMA_SQ;
            if chi < TH_H {
                s_h += TH_H - chi;
            }
        }
        // 2 -> 1
        let d = hinv * Vec3F64::new(p2.x, p2.y, 1.0);
        if d.z.abs() > 1e-12 {
            let (u, v) = (d.x / d.z, d.y / d.z);
            let chi = ((p1.x - u).powi(2) + (p1.y - v).powi(2)) * INV_SIGMA_SQ;
            if chi < TH_H {
                s_h += TH_H - chi;
            }
        }
    }

    // --- Fundamental score: symmetric point-to-epipolar-line distance. ---
    let fm = f.model;
    let ft = fm.transpose();
    let mut s_f = 0.0;
    for (p1, p2) in x1.iter().zip(x2.iter()) {
        // Line in image 2 induced by the point in image 1.
        let l = fm * Vec3F64::new(p1.x, p1.y, 1.0);
        let den = l.x * l.x + l.y * l.y;
        if den > 1e-12 {
            let num = (l.x * p2.x + l.y * p2.y + l.z).powi(2);
            let chi = (num / den) * INV_SIGMA_SQ;
            if chi < TH_F {
                s_f += SCORE_CAP - chi;
            }
        }
        // Line in image 1 induced by the point in image 2.
        let l = ft * Vec3F64::new(p2.x, p2.y, 1.0);
        let den = l.x * l.x + l.y * l.y;
        if den > 1e-12 {
            let num = (l.x * p1.x + l.y * p1.y + l.z).powi(2);
            let chi = (num / den) * INV_SIGMA_SQ;
            if chi < TH_F {
                s_f += SCORE_CAP - chi;
            }
        }
    }

    let total = s_h + s_f;
    if total <= 0.0 {
        return None;
    }
    Some(s_h / total)
}
/// Triangulate every not-yet-reconstructed track that has ≥2 placed cameras, adding it to `point3d`.
fn triangulate_new(
    point3d: &mut HashMap<usize, Vec3F64>,
    norm: &[Vec<(usize, Vec2F64)>],
    poses: &[Option<Pose3d>],
    idcam: &PinholeCamera,
    tcfg: &TriangulationConfig,
) {
    for (ti, obs) in norm.iter().enumerate() {
        if point3d.contains_key(&ti) {
            continue;
        }
        let placed: Vec<(usize, Vec2F64)> = obs
            .iter()
            .copied()
            .filter(|(c, _)| poses[*c].is_some())
            .collect();
        if placed.len() < 2 {
            continue;
        }
        // Widest-baseline pair among placed cameras.
        let centers: Vec<Vec3F64> = placed
            .iter()
            .map(|(c, _)| poses[*c].unwrap().inverse().translation)
            .collect();
        let mut best = (0usize, 1usize, -1.0f64);
        for i in 0..placed.len() {
            for j in (i + 1)..placed.len() {
                let d = (centers[i] - centers[j]).length();
                if d > best.2 {
                    best = (i, j, d);
                }
            }
        }
        let (ca, ua) = placed[best.0];
        let (cb, ub) = placed[best.1];
        if let Ok(pts) = triangulate_matched_points(
            &[ua],
            &[ub],
            &poses[ca].unwrap(),
            &poses[cb].unwrap(),
            idcam,
            tcfg,
        ) {
            if pts.len() == 1 {
                point3d.insert(ti, pts[0].position);
            }
        }
    }
}

/// Metric scale factor `tag_size / reconstructed_side` from the reference tag's triangulated corners.
/// Returns `1.0` (leave up-to-scale) when the tag can't be triangulated or the tag size is unset.
#[allow(clippy::too_many_arguments)]
fn tag_scale(
    tags: &[TagObservation],
    cameras: &[PinholeCamera],
    poses_w2c: &[Pose3d],
    registered: &[bool],
    idcam: &PinholeCamera,
    tcfg: &TriangulationConfig,
    tag_size_m: f64,
) -> Option<f64> {
    if tag_size_m <= 0.0 {
        return None;
    }
    let tag = tags.first()?;
    let seers: Vec<usize> = tag
        .per_camera
        .iter()
        .map(|(c, _)| *c)
        .filter(|c| registered[*c])
        .collect();
    if seers.len() < 2 {
        return None;
    }
    // Widest-baseline placed pair seeing the tag (same pair for all 4 corners).
    let centers: Vec<Vec3F64> = seers
        .iter()
        .map(|c| poses_w2c[*c].inverse().translation)
        .collect();
    let mut best = (0usize, 1usize, -1.0f64);
    for i in 0..seers.len() {
        for j in (i + 1)..seers.len() {
            let d = (centers[i] - centers[j]).length();
            if d > best.2 {
                best = (i, j, d);
            }
        }
    }
    let (ca, cb) = (seers[best.0], seers[best.1]);
    let cca = tag.per_camera.iter().find(|(c, _)| *c == ca).unwrap().1;
    let ccb = tag.per_camera.iter().find(|(c, _)| *c == cb).unwrap().1;
    let mut world: [Option<Vec3F64>; 4] = [None; 4];
    for k in 0..4 {
        let na = cameras[ca].normalize(cca[k]);
        let nb = cameras[cb].normalize(ccb[k]);
        if let Ok(pts) =
            triangulate_matched_points(&[na], &[nb], &poses_w2c[ca], &poses_w2c[cb], idcam, tcfg)
        {
            if pts.len() == 1 {
                world[k] = Some(pts[0].position);
            }
        }
    }
    // Aruco winding (TL,TR,BR,BL): all four edges are one tag side. Average the available edges.
    let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
    let (mut sum, mut cnt) = (0.0f64, 0usize);
    for (a, b) in edges {
        if let (Some(pa), Some(pb)) = (world[a], world[b]) {
            sum += (pa - pb).length();
            cnt += 1;
        }
    }
    let recon_side = sum / cnt as f64;
    if cnt == 0 || recon_side < 1e-9 {
        return None;
    }
    Some(tag_size_m / recon_side)
}

/// Per-camera reprojection RMS (pixels) for the feature path; analytical covariance fields left `None`.
/// Reprojection is scale-invariant, so the UNSCALED world→cam BA poses are used directly.
#[allow(clippy::too_many_arguments)]
fn feature_stats(
    focal_scale: f64,
    camera: usize,
    poses_w2c: &[Pose3d],
    registered: &[bool],
    points: &[Vec3F64],
    pt_index: &HashMap<usize, usize>,
    norm: &[Vec<(usize, Vec2F64)>],
    cam: &PinholeCamera,
) -> CameraStats {
    if !registered[camera] {
        return CameraStats::unconstrained(camera, false, 0);
    }
    let pose = poses_w2c[camera]; // world→cam
    let (mut se, mut num) = (0.0f64, 0usize);
    for (ti, obs) in norm.iter().enumerate() {
        let Some(&pidx) = pt_index.get(&ti) else {
            continue;
        };
        let Some((_, n)) = obs.iter().find(|(c, _)| *c == camera) else {
            continue;
        };
        if let Some(r) = norm_residual(&pose, points[pidx], *n) {
            // `focal_scale` converts into the CORRECTED camera's pixels; 1.0 when refinement is off.
            se += (r * cam.fx * focal_scale).powi(2); // r Euclidean in normalized units; fx≈fy
            num += 1;
        }
    }
    let mut s = CameraStats::unconstrained(camera, true, num);
    if num > 0 {
        s.reproj_rmse_px = (se / num as f64).sqrt();
    }
    s
}

fn global_reproj_rmse(
    focal_scale: f64,
    poses_w2c: &[Pose3d],
    registered: &[bool],
    points: &[Vec3F64],
    pt_index: &HashMap<usize, usize>,
    norm: &[Vec<(usize, Vec2F64)>],
    cameras: &[PinholeCamera],
) -> f64 {
    let (mut se, mut num) = (0.0f64, 0usize);
    for (ti, obs) in norm.iter().enumerate() {
        let Some(&pidx) = pt_index.get(&ti) else {
            continue;
        };
        for (c, n) in obs {
            if !registered[*c] {
                continue;
            }
            if let Some(r) = norm_residual(&poses_w2c[*c], points[pidx], *n) {
                se += (r * cameras[*c].fx * focal_scale).powi(2);
                num += 1;
            }
        }
    }
    if num > 0 {
        (se / num as f64).sqrt()
    } else {
        -1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RigCalibration;

    fn pinhole(f: f64) -> PinholeCamera {
        PinholeCamera {
            fx: f,
            fy: f,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }
    fn rot(yaw: f64, pitch: f64) -> Mat3F64 {
        let (cy, sy) = (yaw.cos(), yaw.sin());
        let (cp, sp) = (pitch.cos(), pitch.sin());
        Mat3F64::from_cols(
            Vec3F64::new(cy, 0.0, -sy),
            Vec3F64::new(sy * sp, cp, cy * sp),
            Vec3F64::new(sy * cp, -sp, cy * cp),
        )
    }
    fn project(pw: Vec3F64, pose_w2c: &Pose3d, k: &PinholeCamera) -> Vec2F64 {
        let pc = pose_w2c.transform_point(&pw);
        Vec2F64::new(k.fx * pc.x / pc.z + k.cx, k.fy * pc.y / pc.z + k.cy)
    }

    #[test]
    fn recovers_metric_extrinsics_features_drive_geometry_tag_scales() {
        // 3 cameras (world→cam) viewing a textured cloud ~2 m away from oblique angles; a 10 cm tag is
        // the ONLY metric reference. Features drive the geometry; the tag fixes scale.
        let cams = [pinhole(500.0), pinhole(500.0), pinhole(500.0)];
        // Cameras CONVERGE on the cloud (~(0,0,1.4)): the left cam yaws right, the right cam yaws left,
        // so all three actually see the shared region (as a real overlapping-FOV rig does).
        let gt = [
            Pose3d::new(rot(0.0, 0.05), Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.40, 0.05), Vec3F64::new(-0.6, 0.0, 0.10)),
            Pose3d::new(rot(-0.40, 0.05), Vec3F64::new(0.6, 0.0, 0.15)),
        ];
        // Only add an observation when the point is IN FRONT and inside the 640x480 image — real
        // feature tracks only exist where a camera actually sees the point.
        let (w, hgt) = (640.0, 480.0);
        let visible = |p: Vec3F64, c: usize| -> Option<Vec2F64> {
            let pc = gt[c].transform_point(&p);
            if pc.z <= 0.1 {
                return None;
            }
            let uv = project(p, &gt[c], &cams[c]);
            (uv.x >= 0.0 && uv.x < w && uv.y >= 0.0 && uv.y < hgt).then_some(uv)
        };
        let mut tracks: Vec<FeatureTrack> = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                let x = -0.5 + 0.111 * i as f64;
                let y = -0.5 + 0.111 * j as f64;
                let z = 1.4 + 0.5 * ((i * 5 + j) as f64 * 0.7).sin() + 0.05 * (i as f64 - j as f64);
                let p = Vec3F64::new(x, y, z);
                let obs: Vec<(usize, Vec2F64)> = (0..3)
                    .filter_map(|c| visible(p, c).map(|uv| (c, uv)))
                    .collect();
                if obs.len() >= 2 {
                    tracks.push(FeatureTrack { obs });
                }
            }
        }
        // 10 cm tag (aruco winding TL,TR,BR,BL) centred at (0,0,2).
        let s = 0.10;
        let corners = [
            Vec3F64::new(-s / 2.0, s / 2.0, 1.4),
            Vec3F64::new(s / 2.0, s / 2.0, 1.4),
            Vec3F64::new(s / 2.0, -s / 2.0, 1.4),
            Vec3F64::new(-s / 2.0, -s / 2.0, 1.4),
        ];
        let tag = TagObservation {
            tag_id: 0,
            per_camera: (0..3)
                .map(|c| {
                    (
                        c,
                        [
                            project(corners[0], &gt[c], &cams[c]),
                            project(corners[1], &gt[c], &cams[c]),
                            project(corners[2], &gt[c], &cams[c]),
                            project(corners[3], &gt[c], &cams[c]),
                        ],
                    )
                })
                .collect(),
        };

        let cfg = ReconstructionConfig::new(s);
        let cal = match reconstruct(&cams, std::slice::from_ref(&tag), &tracks, &cfg, None)
            .map(RigCalibration::from)
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("ERR: {e:?}");
                panic!("{e:?}");
            }
        };
        assert!(
            cal.poses.iter().all(|p| p.is_some()),
            "all cameras registered"
        );

        // Camera-to-camera baselines (gauge-invariant, metric) must match ground truth. Output poses
        // are T_world_cam (camera→world) → translation is the camera centre in world.
        let rc: Vec<Vec3F64> = cal.poses.iter().map(|p| p.unwrap().translation).collect();
        let gc: Vec<Vec3F64> = gt.iter().map(|p| p.inverse().translation).collect();
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let r = (rc[i] - rc[j]).length();
            let g = (gc[i] - gc[j]).length();
            assert!(
                (r - g).abs() < 0.02,
                "baseline {i}-{j}: recovered {r:.4} vs gt {g:.4}"
            );
        }
        assert!(
            cal.reproj_rmse_px >= 0.0 && cal.reproj_rmse_px < 1.0,
            "reproj {}",
            cal.reproj_rmse_px
        );

        // The MAP must be metric too, not just the poses. `reconstruct` runs the same solve, so
        // the points it returns have to share the poses' scale -- this is the assertion the
        // no-tag consistency test cannot make, because there `scale == 1.0` and dropping the
        // scaling changes nothing.
        let recon = reconstruct(&cams, std::slice::from_ref(&tag), &tracks, &cfg, None)
            .expect("same solve must succeed");
        assert_eq!(
            recon.scale,
            ScaleSource::Tag {
                id: 0,
                size_m: 0.10
            }
        );
        assert!(!recon.points.is_empty());

        // Reproject every observation through its own view and point: metric points against
        // metric poses land on the measured pixel, unscaled points against scaled poses do not.
        let mut worst: f64 = 0.0;
        for o in &recon.observations {
            let w2c = recon.views[o.view].expect("registered").inverse();
            worst = worst.max(
                (project(recon.points[o.point].position, &w2c, &cams[o.view]) - o.pixel).length(),
            );
        }
        assert!(
            worst < 2.0,
            "map is not in the same metric frame as the poses: worst reprojection {worst:.3} px"
        );

        // And the cloud spans a physically sensible extent -- the synthetic points live within
        // ~1.2 m of each other, which an unscaled or mis-scaled map would not reproduce.
        let (lo, hi) = recon
            .points
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                (lo.min(p.position.z), hi.max(p.position.z))
            });
        assert!(
            (0.5..4.0).contains(&lo) && (0.5..4.0).contains(&hi),
            "point depths {lo:.2}..{hi:.2} m are not the metric scene"
        );
    }

    /// `reconstruct` must return a map that is CONSISTENT with the poses it returns.
    ///
    /// The pre-M1 API computed all of this and discarded it, so this test could not be written
    /// against it at all -- there was no map to check. Asserting the points are
    /// merely non-empty would be weak, so this reprojects every returned observation through its
    /// own view and its own point and requires it to land on the measured pixel.
    #[test]
    fn reconstruct_returns_a_map_consistent_with_its_poses() {
        // Same synthetic rig as the calibration test above: three views of a point cloud.
        let cams = vec![pinhole(600.0), pinhole(600.0), pinhole(600.0)];
        let poses_gt = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.25, 0.0), Vec3F64::new(-0.30, 0.0, 0.02)),
            Pose3d::new(rot(-0.22, 0.05), Vec3F64::new(0.28, 0.01, 0.03)),
        ];
        let world: Vec<Vec3F64> = (0..24)
            .map(|i| {
                let (a, b) = ((i % 6) as f64, (i / 6) as f64);
                Vec3F64::new(
                    -0.25 + 0.1 * a,
                    -0.15 + 0.1 * b,
                    1.4 + 0.05 * ((i % 5) as f64),
                )
            })
            .collect();

        let tracks: Vec<FeatureTrack> = world
            .iter()
            .map(|pw| FeatureTrack {
                obs: (0..3)
                    .map(|c| (c, project(*pw, &poses_gt[c], &cams[c])))
                    .collect(),
            })
            .collect();

        let cfg = ReconstructionConfig::new(0.1);
        let recon =
            reconstruct(&cams, &[], &tracks, &cfg, None).expect("synthetic scene must solve");

        // The map came back at all -- the whole point of M1.
        assert!(!recon.points.is_empty(), "no points returned");
        assert!(!recon.observations.is_empty(), "no observations returned");
        assert!(
            recon.points.iter().all(|p| p.track_id < tracks.len()),
            "every point must name a real input track"
        );
        // No tag was supplied, so the map is honestly up to scale rather than silently "metric".
        assert_eq!(recon.scale, ScaleSource::UpToScale);

        // Every track id must index the input, and every observation must index the map.
        for o in &recon.observations {
            assert!(o.view < recon.views.len(), "view {} out of range", o.view);
            assert!(
                o.point < recon.points.len(),
                "point {} out of range",
                o.point
            );
            assert!(
                recon.views[o.view].is_some(),
                "an observation names view {} which was never registered",
                o.view
            );
        }

        // The real assertion: reprojecting each returned point through its own returned pose must
        // reproduce the measured pixel. This fails if points, poses and observations are not
        // mutually consistent -- e.g. if the points were left in the pre-scaling frame, or if
        // point_track_id were misaligned with points.
        let mut worst: f64 = 0.0;
        for o in &recon.observations {
            let t_world_cam = recon.views[o.view].expect("checked above");
            let w2c = t_world_cam.inverse();
            let uv = project(recon.points[o.point].position, &w2c, &cams[o.view]);
            worst = worst.max((uv - o.pixel).length());
        }
        assert!(
            worst < 1.0,
            "returned map is not consistent with its poses: worst reprojection {worst:.3} px"
        );
    }

    /// A tag that could not anchor the scale must NOT be reported as the scale source.
    ///
    /// `tag_scale` falls back to the identity in ordinary situations -- an unset `tag_size_m`,
    /// a tag seen by fewer than two REGISTERED views, corners that fail to triangulate. Deciding
    /// `ScaleSource` from `tags_for_scale.first()` reported `Tag { .. }` in all of them, so the
    /// caller was told the map is metric while the points sat at the identity scale. That is the
    /// exact ambiguity `ScaleSource` exists to remove, so it is worth a test: the old
    /// `reference_tag_id` carried the same lie and nothing caught it.
    #[test]
    fn a_tag_that_cannot_anchor_is_reported_as_up_to_scale() {
        let cams = vec![pinhole(600.0), pinhole(600.0), pinhole(600.0)];
        let poses_gt = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.25, 0.0), Vec3F64::new(-0.30, 0.0, 0.02)),
            Pose3d::new(rot(-0.22, 0.05), Vec3F64::new(0.28, 0.01, 0.03)),
        ];
        let world: Vec<Vec3F64> = (0..24)
            .map(|i| {
                let (a, b) = ((i % 6) as f64, (i / 6) as f64);
                Vec3F64::new(
                    -0.25 + 0.1 * a,
                    -0.15 + 0.1 * b,
                    1.4 + 0.05 * ((i % 5) as f64),
                )
            })
            .collect();
        let tracks: Vec<FeatureTrack> = world
            .iter()
            .map(|pw| FeatureTrack {
                obs: (0..3)
                    .map(|c| (c, project(*pw, &poses_gt[c], &cams[c])))
                    .collect(),
            })
            .collect();

        // A tag seen by ONE view only: `tag_scale` needs two registered seers, so it cannot
        // anchor. The tag is supplied, so the old rule would have claimed `Tag { id: 3, .. }`.
        let s = 0.05;
        let corners = [
            Vec3F64::new(-s, s, 1.5),
            Vec3F64::new(s, s, 1.5),
            Vec3F64::new(s, -s, 1.5),
            Vec3F64::new(-s, -s, 1.5),
        ];
        let tag = TagObservation {
            tag_id: 3,
            per_camera: vec![(
                0,
                [
                    project(corners[0], &poses_gt[0], &cams[0]),
                    project(corners[1], &poses_gt[0], &cams[0]),
                    project(corners[2], &poses_gt[0], &cams[0]),
                    project(corners[3], &poses_gt[0], &cams[0]),
                ],
            )],
        };

        let cfg = ReconstructionConfig::new(2.0 * s);
        let recon = reconstruct(&cams, std::slice::from_ref(&tag), &tracks, &cfg, None)
            .expect("the feature tracks alone must still solve");

        assert_eq!(
            recon.scale,
            ScaleSource::UpToScale,
            "a tag seen by one view cannot anchor scale, so the map must not claim to be metric"
        );

        // `tag_scale` has four fallback paths, and reporting `Tag { .. }` for ANY of them is the
        // bug. The case above is "seen by fewer than two registered views"; cover the other three.

        // (a) Size unset: nothing to convert reconstruction units into.
        let mut two_view = tag.clone();
        two_view.per_camera.push((
            1,
            [
                project(corners[0], &poses_gt[1], &cams[1]),
                project(corners[1], &poses_gt[1], &cams[1]),
                project(corners[2], &poses_gt[1], &cams[1]),
                project(corners[3], &poses_gt[1], &cams[1]),
            ],
        ));
        let unset = reconstruct(
            &cams,
            std::slice::from_ref(&two_view),
            &tracks,
            &ReconstructionConfig::new(0.0),
            None,
        )
        .expect("solves");
        assert_eq!(
            unset.scale,
            ScaleSource::UpToScale,
            "tag_size_m = 0 cannot anchor scale"
        );

        // (b) Degenerate corners: all four coincide, so the reconstructed side is ~0 and the
        // implied scale would be a division by nothing.
        let degenerate = TagObservation {
            tag_id: 3,
            per_camera: (0..2)
                .map(|c| {
                    let uv = project(corners[0], &poses_gt[c], &cams[c]);
                    (c, [uv, uv, uv, uv])
                })
                .collect(),
        };
        let deg = reconstruct(
            &cams,
            std::slice::from_ref(&degenerate),
            &tracks,
            &cfg,
            None,
        )
        .expect("solves");
        assert_eq!(
            deg.scale,
            ScaleSource::UpToScale,
            "a tag whose corners coincide has no measurable side, so it cannot anchor scale"
        );

        // (c) Corners that cannot triangulate: put them behind the cameras, so cheirality rejects
        // every one and no side is reconstructed at all.
        let behind: [Vec3F64; 4] = [
            Vec3F64::new(-s, s, -1.0),
            Vec3F64::new(s, s, -1.0),
            Vec3F64::new(s, -s, -1.0),
            Vec3F64::new(-s, -s, -1.0),
        ];
        let untriangulable = TagObservation {
            tag_id: 3,
            per_camera: (0..2)
                .map(|c| {
                    (
                        c,
                        [
                            project(behind[0], &poses_gt[c], &cams[c]),
                            project(behind[1], &poses_gt[c], &cams[c]),
                            project(behind[2], &poses_gt[c], &cams[c]),
                            project(behind[3], &poses_gt[c], &cams[c]),
                        ],
                    )
                })
                .collect(),
        };
        let untri = reconstruct(
            &cams,
            std::slice::from_ref(&untriangulable),
            &tracks,
            &cfg,
            None,
        )
        .expect("solves");
        assert_eq!(
            untri.scale,
            ScaleSource::UpToScale,
            "corners that fail to triangulate cannot anchor scale"
        );
    }

    /// `RigCalibration::from` inherits the scale honesty, and that CHANGES its behaviour.
    ///
    /// Before this, `reference_tag_id` was `tags_for_scale.first()`'s id whenever a tag was
    /// supplied — even when `tag_scale` had fallen back to the identity, so a caller could read a
    /// real tag id off a map that was still up to scale. It is now `0` unless the tag actually
    /// anchored.
    ///
    /// That is a deliberate fix, not a regression, but it is a behaviour change on the EXISTING
    /// entry point rather than only on the new one, so it is asserted here at the
    /// `RigCalibration` level and not merely via `ScaleSource`.
    #[test]
    fn rig_calibration_reports_no_reference_tag_when_the_tag_did_not_anchor() {
        let cams = vec![pinhole(600.0), pinhole(600.0), pinhole(600.0)];
        let poses_gt = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.25, 0.0), Vec3F64::new(-0.30, 0.0, 0.02)),
            Pose3d::new(rot(-0.22, 0.05), Vec3F64::new(0.28, 0.01, 0.03)),
        ];
        let tracks: Vec<FeatureTrack> = (0..24)
            .map(|i| {
                let (a, b) = ((i % 6) as f64, (i / 6) as f64);
                let p = Vec3F64::new(
                    -0.25 + 0.1 * a,
                    -0.15 + 0.1 * b,
                    1.4 + 0.05 * ((i % 5) as f64),
                );
                FeatureTrack {
                    obs: (0..3)
                        .map(|c| (c, project(p, &poses_gt[c], &cams[c])))
                        .collect(),
                }
            })
            .collect();

        // Tag id 9, seen by ONE view: `tag_scale` needs two registered seers, so it cannot anchor.
        let s = 0.05;
        let corners = [
            Vec3F64::new(-s, s, 1.5),
            Vec3F64::new(s, s, 1.5),
            Vec3F64::new(s, -s, 1.5),
            Vec3F64::new(-s, -s, 1.5),
        ];
        let tag = TagObservation {
            tag_id: 9,
            per_camera: vec![(
                0,
                [
                    project(corners[0], &poses_gt[0], &cams[0]),
                    project(corners[1], &poses_gt[0], &cams[0]),
                    project(corners[2], &poses_gt[0], &cams[0]),
                    project(corners[3], &poses_gt[0], &cams[0]),
                ],
            )],
        };

        let cal = reconstruct(
            &cams,
            std::slice::from_ref(&tag),
            &tracks,
            &ReconstructionConfig::new(2.0 * s),
            None,
        )
        .map(RigCalibration::from)
        .expect("the feature tracks alone must still solve");

        assert_eq!(
            cal.reference_tag_id, 0,
            "tag 9 was supplied but could not anchor scale, so it must not be reported as the \
             reference — that would tell the caller the map is metric when it is not"
        );
    }

    /// The published map order must not depend on `HashMap` iteration order.
    #[test]
    fn map_order_is_deterministic_and_sorted_by_track() {
        let cams = vec![pinhole(600.0), pinhole(600.0)];
        let poses_gt = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.25, 0.0), Vec3F64::new(-0.30, 0.0, 0.02)),
        ];
        let tracks: Vec<FeatureTrack> = (0..30)
            .map(|i| {
                let p = Vec3F64::new(
                    -0.3 + 0.02 * i as f64,
                    -0.1 + 0.01 * (i % 7) as f64,
                    1.3 + 0.03 * (i % 5) as f64,
                );
                FeatureTrack {
                    obs: (0..2)
                        .map(|c| (c, project(p, &poses_gt[c], &cams[c])))
                        .collect(),
                }
            })
            .collect();
        let cfg = ReconstructionConfig::new(0.1);

        let a = reconstruct(&cams, &[], &tracks, &cfg, None).expect("solves");
        let b = reconstruct(&cams, &[], &tracks, &cfg, None).expect("solves");

        let ids = |r: &Reconstruction| r.points.iter().map(|p| p.track_id).collect::<Vec<_>>();
        let (ia, ib) = (ids(&a), ids(&b));
        assert!(!ia.is_empty(), "need points to order");
        assert!(
            ia.windows(2).all(|w| w[0] < w[1]),
            "points must be published in ascending track order, got {:?}",
            &ia[..ia.len().min(8)]
        );
        assert_eq!(
            ia, ib,
            "two runs over identical input must publish the same map order"
        );
    }

    /// Three converging views of a textured cloud at a SPECIFIED metric scale (the shape is
    /// scale-free, so `scale` only moves the metric relative to the bootstrap's unit baseline).
    /// Returns `(cameras, world→cam ground truth, tracks)`.
    fn converging_rig(f: f64, scale: f64) -> (Vec<PinholeCamera>, Vec<Pose3d>, Vec<FeatureTrack>) {
        let cams = vec![pinhole(f), pinhole(f), pinhole(f)];
        let gt = vec![
            Pose3d::new(rot(0.0, 0.05), Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.40, 0.05), Vec3F64::new(-0.6, 0.0, 0.10) * scale),
            Pose3d::new(rot(-0.40, 0.05), Vec3F64::new(0.6, 0.0, 0.15) * scale),
        ];
        let (w, h) = (640.0, 480.0);
        let mut tracks: Vec<FeatureTrack> = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                let x = -0.5 + 0.111 * i as f64;
                let y = -0.5 + 0.111 * j as f64;
                let z = 1.4 + 0.5 * ((i * 5 + j) as f64 * 0.7).sin() + 0.05 * (i as f64 - j as f64);
                let p = Vec3F64::new(x, y, z) * scale;
                let obs: Vec<(usize, Vec2F64)> = (0..3)
                    .filter_map(|c| {
                        let pc = gt[c].transform_point(&p);
                        if pc.z <= 0.1 * scale {
                            return None;
                        }
                        let uv = project(p, &gt[c], &cams[c]);
                        (uv.x >= 0.0 && uv.x < w && uv.y >= 0.0 && uv.y < h).then_some((c, uv))
                    })
                    .collect();
                if obs.len() >= 2 {
                    tracks.push(FeatureTrack { obs });
                }
            }
        }
        (cams, gt, tracks)
    }

    /// Ground-truth metric depth for every observation, shaped exactly like `tracks`.
    fn gt_depths(
        gt: &[Pose3d],
        tracks: &[FeatureTrack],
        cams: &[PinholeCamera],
    ) -> Vec<Vec<Option<f64>>> {
        // Intersect each track's first two rays under the GT poses — exact for noise-free data.
        tracks
            .iter()
            .map(|t| {
                let (c0, uv0) = t.obs[0];
                let (c1, uv1) = t.obs[1];
                let pts = triangulate_matched_points(
                    &[cams[c0].normalize(uv0)],
                    &[cams[c1].normalize(uv1)],
                    &gt[c0],
                    &gt[c1],
                    &PinholeCamera::IDENTITY,
                    &TriangulationConfig {
                        min_parallax_deg: 0.0,
                        max_reprojection_error: 1e9,
                        min_cheirality_count: 0,
                        ..Default::default()
                    },
                )
                .expect("synthetic rays must intersect");
                let p = pts[0].position;
                t.obs
                    .iter()
                    .map(|(c, _)| Some(gt[*c].transform_point(&p).z))
                    .collect()
            })
            .collect()
    }

    /// Median of `z_reconstructed / d_measured` over every surviving observation: `1.0` exactly
    /// when the map is at the metric scale the depths assert.
    fn median_depth_ratio(
        recon: &Reconstruction,
        tracks: &[FeatureTrack],
        depths: &[Vec<Option<f64>>],
    ) -> f64 {
        let mut r: Vec<f64> = Vec::new();
        for o in &recon.observations {
            let ti = recon.points[o.point].track_id;
            let Some(j) = tracks[ti].obs.iter().position(|(c, _)| *c == o.view) else {
                continue;
            };
            let Some(d) = depths[ti][j] else { continue };
            let w2c = recon.views[o.view].expect("registered").inverse();
            let z = w2c.transform_point(&recon.points[o.point].position).z;
            if d > 1e-9 {
                r.push(z / d);
            }
        }
        assert!(!r.is_empty(), "no observation carried a depth to compare");
        r.sort_by(|a, b| a.partial_cmp(b).unwrap());
        r[r.len() / 2]
    }

    /// **Depth measurements fix the scale of an otherwise scale-free reconstruction.**
    ///
    /// The bootstrap fixes the gauge with a UNIT seed baseline; here the true seed baseline is
    /// deliberately far from 1, so the depth-free control lands several times too large. That
    /// control is load-bearing: without it the assertions would pass on a build that ignored depth.
    #[test]
    fn depth_priors_fix_the_scale_of_an_otherwise_scale_free_reconstruction() {
        // Scale 0.4: widest camera-centre baseline ~0.5 m, so the unit-baseline convention
        // inflates the depth-free map ~2x.
        let (cams, gt, tracks) = converging_rig(500.0, 0.4);
        let depths = gt_depths(&gt, &tracks, &cams);

        // Control: no depth. Same tracks, same config -- only the argument differs.
        let free = reconstruct(&cams, &[], &tracks, &ReconstructionConfig::new(0.0), None)
            .expect("solves");
        let free_ratio = median_depth_ratio(&free, &tracks, &depths);
        assert!(
            free_ratio > 1.5,
            "the depth-free control must be visibly off-scale for this test to mean anything, \
             got ratio {free_ratio:.4}"
        );

        // With depth. `depth_prior_rel_sigma` is what ARMS the feature.
        let cfg = ReconstructionConfig {
            depth_prior_rel_sigma: 0.15,
            ..ReconstructionConfig::new(0.0)
        };
        let metric = reconstruct(&cams, &[], &tracks, &cfg, Some(&depths)).expect("solves");
        let ratio = median_depth_ratio(&metric, &tracks, &depths);
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "depth priors must bring the map to the metric scale they assert: ratio {ratio:.4} \
             (depth-free control {free_ratio:.4})"
        );

        // Metric SCALE, not merely metric depths: camera centres must sit at their true separations.
        let centre = |p: &Option<Pose3d>| p.expect("registered").translation;
        let gt_centre = |p: &Pose3d| p.inverse().translation;
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let got = (centre(&metric.views[i]) - centre(&metric.views[j])).length();
            let want = (gt_centre(&gt[i]) - gt_centre(&gt[j])).length();
            assert!(
                (got - want).abs() < 0.05 * want,
                "baseline {i}-{j} is {got:.4} m, ground truth {want:.4} m"
            );
        }

        // Still HONEST about the source: depth is a soft prior, and `ScaleSource` names fiducials.
        assert_eq!(metric.scale, ScaleSource::UpToScale);
    }

    /// Passing depths while `depth_prior_rel_sigma` is 0 must change NOTHING: the knob, not the
    /// argument, arms the feature. Bit-identical, not merely close.
    #[test]
    fn depths_are_inert_until_the_sigma_arms_them() {
        let (cams, gt, tracks) = converging_rig(500.0, 0.4);
        let depths = gt_depths(&gt, &tracks, &cams);
        let cfg = ReconstructionConfig::new(0.0);

        let without = reconstruct(&cams, &[], &tracks, &cfg, None).expect("solves");
        let with = reconstruct(&cams, &[], &tracks, &cfg, Some(&depths)).expect("solves");

        assert_eq!(without.points.len(), with.points.len());
        for (a, b) in without.points.iter().zip(&with.points) {
            assert_eq!(a.position, b.position, "depth leaked into a disarmed solve");
        }
    }

    /// `min_registration_inliers` decides whether a thinly-supported view is admitted. Views 0 and
    /// 1 share 80 tracks and bootstrap the map; view 2 sees only 20 of them, so its PnP consensus
    /// cannot exceed 20. The default gate of 30 must refuse it; lowering the gate must admit it.
    #[test]
    fn min_registration_inliers_gates_a_thinly_supported_view() {
        let cams = vec![pinhole(600.0), pinhole(600.0), pinhole(600.0)];
        let gt = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.30, 0.0), Vec3F64::new(-0.35, 0.0, 0.02)),
            Pose3d::new(rot(-0.28, 0.04), Vec3F64::new(0.33, 0.01, 0.03)),
        ];
        let tracks: Vec<FeatureTrack> = (0..100)
            .map(|i| {
                let p = Vec3F64::new(
                    -0.30 + 0.006 * i as f64,
                    -0.20 + 0.004 * (i % 11) as f64,
                    1.30 + 0.05 * (i % 7) as f64,
                );
                // Only the last 20 tracks are visible to view 2.
                let seen: &[usize] = if i < 80 { &[0, 1] } else { &[0, 1, 2] };
                FeatureTrack {
                    obs: seen
                        .iter()
                        .map(|c| (*c, project(p, &gt[*c], &cams[*c])))
                        .collect(),
                }
            })
            .collect();

        let gated = reconstruct(&cams, &[], &tracks, &ReconstructionConfig::new(0.0), None)
            .expect("solves");
        assert_eq!(
            ReconstructionConfig::new(0.0).min_registration_inliers,
            30,
            "this test is calibrated against the documented default"
        );
        assert!(
            gated.views[2].is_none(),
            "20 correspondences cannot clear a 30-inlier gate, so view 2 must stay unregistered"
        );
        assert!(gated.views[0].is_some() && gated.views[1].is_some());

        let open = reconstruct(
            &cams,
            &[],
            &tracks,
            &ReconstructionConfig {
                min_registration_inliers: 15,
                ..ReconstructionConfig::new(0.0)
            },
            None,
        )
        .expect("solves");
        assert!(
            open.views[2].is_some(),
            "lowering the gate below the available support must admit the view"
        );
    }

    /// `refine_intrinsics` must REPORT its fit and then ACT on it.
    ///
    /// 1. It reaches the caller on [`Reconstruction::camera_correction`] — refinement applies
    ///    itself by rewriting normalized observations in place, so a caller that never sees the fit
    ///    keeps storing a camera that did not produce these poses.
    /// 2. It CHANGES THE POSES: without a second solve, the corrected observations are never read.
    ///
    /// It deliberately does NOT claim recovery of a known FOCAL error: the fit only sees what
    /// bundle adjustment failed to absorb, and a focal error is very nearly absorbable — MEASURED,
    /// handing the solver 320 for a true focal of 400 left `gamma = 0.9994`. Radial DISTORTION is a
    /// nonlinear warp a rigid reconstruction cannot absorb, so that is what the scene injects.
    #[test]
    fn refine_intrinsics_reports_its_fit_and_re_solves_against_it() {
        const K1_TRUE: f64 = 0.25;
        let (cams, _, clean) = converging_rig(500.0, 1.0);
        // Barrel-distort every pixel; the solver's cameras keep `k1 = 0`, so it survives
        // `normalize`.
        let tracks: Vec<FeatureTrack> = clean
            .iter()
            .map(|t| FeatureTrack {
                obs: t
                    .obs
                    .iter()
                    .map(|(c, uv)| {
                        let (x, y) = ((uv.x - 320.0) / 500.0, (uv.y - 240.0) / 500.0);
                        let s = 1.0 + K1_TRUE * (x * x + y * y);
                        (
                            *c,
                            Vec2F64::new(320.0 + 500.0 * x * s, 240.0 + 500.0 * y * s),
                        )
                    })
                    .collect(),
            })
            .collect();

        let plain = reconstruct(&cams, &[], &tracks, &ReconstructionConfig::new(0.0), None)
            .expect("solves");
        assert_eq!(
            plain.camera_correction, None,
            "refinement is off by default, so there is no fit to report"
        );

        let refined = reconstruct(
            &cams,
            &[],
            &tracks,
            &ReconstructionConfig {
                refine_intrinsics: true,
                ..ReconstructionConfig::new(0.0)
            },
            None,
        )
        .expect("solves");

        let (gamma, k1, ..) = refined
            .camera_correction
            .expect("injected distortion must produce a reportable fit");
        assert!(
            k1 < -0.005,
            "the fit must UNDO the injected barrel distortion (k1_true = +{K1_TRUE}), so its own \
             k1 has to be negative; got {k1:.5}"
        );
        // The focal was correct here, so the fit must not invent one while fitting distortion.
        assert!(
            (0.95..1.05).contains(&gamma),
            "a correctly-assumed focal must survive the fit intact; gamma={gamma:.4} implies \
             fx_true = {:.1} against the true 500",
            500.0 / gamma
        );

        // The second solve is the load-bearing half: without it, poses BIT-IDENTICAL to `plain`.
        let moved = plain
            .views
            .iter()
            .zip(&refined.views)
            .filter_map(|(a, b)| Some((a.as_ref()?.translation - b.as_ref()?.translation).length()))
            .fold(0.0f64, f64::max);
        assert!(
            moved > 1e-6,
            "the refinement was reported but never re-solved against: poses are unchanged"
        );
        assert!(
            refined.views.iter().all(|v| v.is_some()),
            "the re-solve must not cost a registration"
        );
    }

    /// The up and motion priors reach the solver at a strength that does not overrule the images.
    ///
    /// Both sigmas are physical and then DEFLATED into normalized reprojection units. Skipping that
    /// makes each prior row `1/sigma_r` times stiffer than every reprojection row (360x at a phone
    /// focal) and the solve fits the assumption instead of the scene. So: both priors ON at
    /// production values must leave a good reconstruction good.
    #[test]
    fn up_and_motion_priors_do_not_overrule_the_image_evidence() {
        // Views are index-ordered along the rig, which is what makes a motion prior meaningful.
        let (cams, gt, tracks) = converging_rig(500.0, 1.0);
        let baseline = reconstruct(&cams, &[], &tracks, &ReconstructionConfig::new(0.0), None)
            .expect("solves");
        let with_priors = reconstruct(
            &cams,
            &[],
            &tracks,
            &ReconstructionConfig {
                up_prior_sigma: 0.25,
                motion_prior_sigma: 0.1,
                ..ReconstructionConfig::new(0.0)
            },
            None,
        )
        .expect("solves");

        assert!(
            with_priors.views.iter().all(|v| v.is_some()),
            "the priors must not cost a registration"
        );
        // Baselines are gauge-invariant only up to the bootstrap's global scale: compare RATIOS.
        let c = |r: &Reconstruction, i: usize| r.views[i].expect("registered").translation;
        let g = |i: usize| gt[i].inverse().translation;
        let want = (g(0) - g(2)).length() / (g(0) - g(1)).length();
        for (name, r) in [("baseline", &baseline), ("with priors", &with_priors)] {
            let got = (c(r, 0) - c(r, 2)).length() / (c(r, 0) - c(r, 1)).length();
            assert!(
                (got - want).abs() < 0.05 * want,
                "{name}: baseline ratio {got:.4} vs ground truth {want:.4}"
            );
        }
        assert!(
            with_priors.reproj_rmse_px < 1.0,
            "priors drove the fit away from the pixels: rmse {:.3} px",
            with_priors.reproj_rmse_px
        );
    }

    /// The reported pixel RMS must be in the corrected camera's pixels, not the assumed one's:
    /// `refine_intrinsics` rewrites observations into the TRUE camera's frame, where
    /// `fx_true = fx_assumed / gamma`, so converting with the assumed `fx` scales every reported
    /// number by gamma. Worth pinning because `reproj_rmse_px` is callers' acceptance metric.
    #[test]
    fn reported_rmse_is_in_corrected_pixels() {
        const K1_TRUE: f64 = 0.25;
        let (cams, _, clean) = converging_rig(500.0, 1.0);
        let tracks: Vec<FeatureTrack> = clean
            .iter()
            .map(|t| FeatureTrack {
                obs: t
                    .obs
                    .iter()
                    .map(|(c, uv)| {
                        let (x, y) = ((uv.x - 320.0) / 500.0, (uv.y - 240.0) / 500.0);
                        let s = 1.0 + K1_TRUE * (x * x + y * y);
                        (
                            *c,
                            Vec2F64::new(320.0 + 500.0 * x * s, 240.0 + 500.0 * y * s),
                        )
                    })
                    .collect(),
            })
            .collect();

        let refined = reconstruct(
            &cams,
            &[],
            &tracks,
            &ReconstructionConfig {
                refine_intrinsics: true,
                ..ReconstructionConfig::new(0.0)
            },
            None,
        )
        .expect("solves");

        let (gamma, ..) = refined
            .camera_correction
            .expect("injected distortion must produce a fit");
        assert!(
            (gamma - 1.0).abs() > 1e-9,
            "test is vacuous unless the fit actually moved gamma (got {gamma})"
        );

        // Global vs per-camera under the CORRECTED focal: either left in assumed-fx pixels and this
        // ratio would sit at gamma rather than 1.
        let n: usize = refined.per_view.iter().map(|s| s.num_obs).sum();
        assert!(n > 0, "no observations to check");
        let pooled: f64 = refined
            .per_view
            .iter()
            .filter(|s| s.num_obs > 0)
            .map(|s| s.reproj_rmse_px.powi(2) * s.num_obs as f64)
            .sum::<f64>()
            / n as f64;
        let pooled = pooled.sqrt();
        assert!(
            (pooled - refined.reproj_rmse_px).abs() <= 1e-6 * refined.reproj_rmse_px.max(1.0),
            "global RMS {:.6} disagrees with the pooled per-camera RMS {pooled:.6} — the two \
             conversions are not using the same focal",
            refined.reproj_rmse_px
        );
    }

    /// A forward-walking sequence is exactly the case the general defaults mishandle.
    ///
    /// Cameras march along -Z looking forward, so consecutive views share the most tracks AND have
    /// the smallest baseline — the bootstrap picks the worst-conditioned pair available. At the
    /// default `min_parallax_deg` every triangulation from it is rejected, growth finds nothing to
    /// register against. Downstream on real 120- and 459-keyframe clips that ends in
    /// `NoFreeVariables`; in THIS scene it ends more gently — the default returns `Ok` having placed
    /// only the two bootstrap views — but the mechanism is the same and the assertion below pins the
    /// difference rather than the failure mode. `sequential()` is the difference between a
    /// reconstruction and a two-view stub, not a quality tweak.
    #[test]
    fn sequential_preset_rescues_a_forward_walk() {
        let n_cams = 6;
        let cams: Vec<PinholeCamera> = (0..n_cams).map(|_| pinhole(500.0)).collect();
        // Walk forward along -Z in 12 cm steps: a phone walkthrough, not a rig.
        let gt: Vec<Pose3d> = (0..n_cams)
            .map(|i| {
                let c = Vec3F64::new(0.0, 0.0, -0.12 * i as f64);
                Pose3d::new(Mat3F64::IDENTITY, -(Mat3F64::IDENTITY * c))
            })
            .collect();
        // Landmarks well ahead of the walk, so every view sees them at a shallow angle.
        let mut pts = Vec::new();
        for i in 0..90 {
            let a = i as f64 * 0.7;
            pts.push(Vec3F64::new(
                a.cos() * 1.2,
                a.sin() * 0.9,
                6.0 + (i % 7) as f64 * 0.3,
            ));
        }
        let tracks: Vec<FeatureTrack> = pts
            .iter()
            .map(|p| FeatureTrack {
                obs: gt
                    .iter()
                    .enumerate()
                    .filter_map(|(c, pose)| {
                        let pc = pose.transform_point(p);
                        (pc.z > 0.1).then(|| {
                            (
                                c,
                                Vec2F64::new(
                                    500.0 * pc.x / pc.z + 320.0,
                                    500.0 * pc.y / pc.z + 240.0,
                                ),
                            )
                        })
                    })
                    .collect(),
            })
            .collect();

        // The strict inequality below rests on the registration gate blocking growth from the
        // tiny seed cloud the default admits, so pin that dependency: if this default ever moves to
        // 0 the test could silently stop discriminating.
        assert_eq!(
            ReconstructionConfig::new(0.0).min_registration_inliers,
            30,
            "this test assumes the documented registration gate"
        );
        let got = reconstruct(&cams, &[], &tracks, &ReconstructionConfig::new(0.0), None);
        let seq = reconstruct(
            &cams,
            &[],
            &tracks,
            &ReconstructionConfig::new(0.0).sequential(),
            None,
        )
        .expect("sequential preset must reconstruct a forward walk");

        let placed = seq.views.iter().filter(|v| v.is_some()).count();
        assert!(
            placed >= n_cams - 1,
            "sequential preset placed only {placed} of {n_cams} views"
        );
        // The control is allowed to fail OR to place fewer views; what it must not do is match the
        // preset, because then this test proves nothing about the preset.
        let placed_base = got
            .map(|r| r.views.iter().filter(|v| v.is_some()).count())
            .unwrap_or(0);
        assert!(
            placed_base < placed,
            "default config placed {placed_base} views and the preset {placed} — the scene does not \
             exercise the parallax gate, so this test would pass on a no-op preset"
        );
    }

    /// Why `sequential()` does NOT arm the constant-velocity prior.
    ///
    /// `sequential_preset_rescues_a_forward_walk` cannot test this: it walks in exactly uniform
    /// steps, so `alpha = 0.5` holds to machine precision and the motion residual is identically
    /// zero. This one accelerates and then PAUSES — two steps of 0.05, two of 0.25, then a repeated
    /// centre — which is where a norm-ratio residual is at its worst: the paused triplet's
    /// denominator nearly vanishes and its stiffness grows as 1/n02.
    ///
    /// Measured: disarmed recovers the segment ratios EXACTLY; armed at 0.1 overshoots segment 1 by
    /// 85% (0.1230 against 0.0667). The prior overrules the image evidence on a capture whose speed
    /// genuinely varies, which a preset cannot rule out. This test guards the preset against someone
    /// re-arming it without measuring — and it is also the test the uniform-speed scene could not
    /// be: there, `alpha = 0.5` holds exactly and the residual is identically zero.
    #[test]
    fn sequential_preset_does_not_bend_a_variable_speed_walk() {
        let steps = [0.05, 0.05, 0.25, 0.25, 0.0, 0.15];
        let mut centres = vec![Vec3F64::new(0.0, 0.0, 0.0)];
        for s in steps {
            let last = *centres.last().unwrap();
            centres.push(last + Vec3F64::new(0.0, 0.0, -s));
        }
        let n_cams = centres.len();
        let cams: Vec<PinholeCamera> = (0..n_cams).map(|_| pinhole(500.0)).collect();
        let gt: Vec<Pose3d> = centres
            .iter()
            .map(|c| Pose3d::new(Mat3F64::IDENTITY, -(Mat3F64::IDENTITY * *c)))
            .collect();

        let mut pts = Vec::new();
        for i in 0..120 {
            let a = i as f64 * 0.53;
            pts.push(Vec3F64::new(
                a.cos() * 1.3,
                a.sin() * 1.0,
                5.0 + (i % 9) as f64 * 0.25,
            ));
        }
        let tracks: Vec<FeatureTrack> = pts
            .iter()
            .map(|p| FeatureTrack {
                obs: gt
                    .iter()
                    .enumerate()
                    .filter_map(|(c, pose)| {
                        let pc = pose.transform_point(p);
                        (pc.z > 0.1).then(|| {
                            (
                                c,
                                Vec2F64::new(
                                    500.0 * pc.x / pc.z + 320.0,
                                    500.0 * pc.y / pc.z + 240.0,
                                ),
                            )
                        })
                    })
                    .collect(),
            })
            .collect();

        // CONTROL: same preset with the motion prior disarmed. If this deviates too, the cause is
        // the pause geometry, not the prior.
        let mut ctrl_cfg = ReconstructionConfig::new(0.0).sequential();
        ctrl_cfg.motion_prior_sigma = 0.0;
        let ctrl = reconstruct(&cams, &[], &tracks, &ctrl_cfg, None).expect("control solves");
        let r = reconstruct(
            &cams,
            &[],
            &tracks,
            &ReconstructionConfig::new(0.0).sequential(),
            None,
        )
        .expect("variable-speed walk must still reconstruct");
        let placed: Vec<usize> = (0..n_cams).filter(|&i| r.views[i].is_some()).collect();
        assert!(
            placed.len() >= n_cams - 1,
            "placed only {} of {n_cams}",
            placed.len()
        );

        // Gauge-invariant check: the recovered inter-view distances, normalised by their own total,
        // must match ground truth's. A prior that bent the trajectory toward constant velocity would
        // even these out — which is exactly the failure this asserts against.
        let centre = |p: &Pose3d| -(p.rotation.transpose() * p.translation);
        let rec: Vec<Vec3F64> = placed
            .iter()
            .map(|&i| centre(&r.views[i].unwrap()))
            .collect();
        let gtc: Vec<Vec3F64> = placed.iter().map(|&i| centres[i]).collect();
        let seg = |v: &[Vec3F64]| -> Vec<f64> {
            let d: Vec<f64> = v.windows(2).map(|w| (w[1] - w[0]).length()).collect();
            let t: f64 = d.iter().sum();
            d.iter().map(|x| x / t.max(1e-12)).collect()
        };
        let (a, b) = (seg(&rec), seg(&gtc));
        let cr: Vec<Vec3F64> = placed
            .iter()
            .map(|&i| centre(&ctrl.views[i].unwrap()))
            .collect();
        let c = seg(&cr);
        for (i, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
            let armed = (ra - rb).abs();
            let disarmed = (c[i] - rb).abs();
            assert!(
                armed < 0.05 || armed <= disarmed + 1e-6,
                "segment {i}: armed {ra:.4} (err {armed:.4}) vs disarmed {:.4} (err {disarmed:.4}), \
                 ground truth {rb:.4} — the motion prior made this segment WORSE",
                c[i]
            );
        }
    }

    /// Deterministic uniform noise. `rand` is not a dependency of this crate, and a fixed stream
    /// keeps the two arms of the comparison below reading the SAME scene.
    struct Lcg(u64);
    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn signed(&mut self) -> f64 {
            self.unit() * 2.0 - 1.0
        }
    }

    /// A sideways walkthrough: `n_views` cameras stepping 12 cm along +X past a slab of scene
    /// points, with a little wobble in the pose and `noise_px` of observation noise.
    ///
    /// Long enough that no view sees more than a fraction of the walk, which is the regime the
    /// growth-time bundle adjustment exists for: consecutive views overlap, distant ones do not, so
    /// registration error has somewhere to compound.
    fn walkthrough(
        n_views: usize,
        n_points: usize,
        noise_px: f64,
    ) -> (Vec<PinholeCamera>, Vec<Pose3d>, Vec<FeatureTrack>) {
        let k = pinhole(600.0);
        let poses: Vec<Pose3d> = (0..n_views)
            .map(|i| {
                let t = i as f64;
                let c = Vec3F64::new(0.12 * t, 0.03 * (0.7 * t).sin(), 0.02 * (0.5 * t).cos());
                let r = rot(0.03 * (0.23 * t).sin(), 0.02 * (0.31 * t).cos());
                Pose3d::new(r, -(r * c))
            })
            .collect();

        let mut rng = Lcg(0x5EED_1234_ABCD_0001);
        let pts: Vec<Vec3F64> = (0..n_points)
            .map(|_| {
                Vec3F64::new(
                    -1.5 + 11.5 * rng.unit(),
                    -1.6 + 3.2 * rng.unit(),
                    3.0 + 3.0 * rng.unit(),
                )
            })
            .collect();

        let mut noise = Lcg(0x5EED_1234_ABCD_0002);
        let tracks: Vec<FeatureTrack> = pts
            .iter()
            .filter_map(|p| {
                let obs: Vec<(usize, Vec2F64)> = (0..n_views)
                    .filter_map(|c| {
                        let pc = poses[c].transform_point(p);
                        if pc.z <= 0.5 {
                            return None;
                        }
                        let uv = project(*p, &poses[c], &k);
                        (uv.x >= 0.0 && uv.x < 640.0 && uv.y >= 0.0 && uv.y < 480.0).then_some((
                            c,
                            Vec2F64::new(
                                uv.x + noise_px * noise.signed(),
                                uv.y + noise_px * noise.signed(),
                            ),
                        ))
                    })
                    .collect();
                (obs.len() >= 3).then_some(FeatureTrack { obs })
            })
            .collect();
        (vec![k; n_views], poses, tracks)
    }

    /// Median angular error, in degrees, of the RELATIVE rotation between consecutive registered
    /// views against ground truth.
    ///
    /// Relative rather than absolute because the comparison must be gauge-free: a reconstruction's
    /// world frame and its scale are both arbitrary, so absolute poses are not comparable without
    /// fitting a Sim(3) first. Views with either end unregistered are skipped.
    fn median_relative_rotation_error_deg(views: &[Option<Pose3d>], gt: &[Pose3d]) -> f64 {
        let mut err_deg: Vec<f64> = Vec::new();
        for i in 1..gt.len() {
            let (Some(a), Some(b)) = (views[i - 1], views[i]) else {
                continue;
            };
            // `views` are T_world_cam, so cam(i-1) -> cam(i) is `a.rotation^T * b.rotation`.
            let r_est = a.rotation.transpose() * b.rotation;
            let r_gt = gt[i - 1].rotation * gt[i].rotation.transpose();
            let d = r_gt.transpose() * r_est;
            let trace = d.col(0).x + d.col(1).y + d.col(2).z;
            err_deg.push(((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees());
        }
        err_deg.sort_by(|a, b| a.partial_cmp(b).unwrap());
        err_deg[err_deg.len() / 2]
    }

    /// How many registered views observe each reconstructed point.
    fn map_spans(recon: &Reconstruction) -> Vec<usize> {
        let mut per_point = vec![0usize; recon.points.len()];
        for o in &recon.observations {
            per_point[o.point] += 1;
        }
        per_point.sort_unstable();
        per_point
    }

    /// Bundle adjustment DURING growth is what keeps a long clip growing and its long tracks
    /// triangulable.
    ///
    /// Every registration is fitted by PnP against points that were themselves triangulated from
    /// poses nothing has refined, so with a terminal-only solve the error compounds ALONG the
    /// chain. Two things then fail together: `triangulate_new` builds each point from the
    /// WIDEST-baseline pair of placed cameras and rejects it over `max_reprojection_error` — and
    /// widest-baseline is also furthest-apart-in-the-walk, so the first structure drift destroys is
    /// exactly the long tracks — and the views left unregistered lose the 2D-3D links they needed.
    /// Neither shows up as an error: both arms below return `Ok`.
    ///
    /// Asserted against the SAME pipeline with the growth-time solves switched off
    /// (`reconstruct_inner(.., growth_ba: false)`) rather than against a fixed threshold, so a
    /// change that silently neuters the local BA fails here instead of passing on a no-op. The
    /// headline metric is track SPAN in the map, not the point count: a map can hold points and
    /// still have nothing tying its two ends together.
    ///
    /// Invisible at rig scale, which is where every other test in this file lives — at 8 or 20
    /// views a single terminal BA cleans up whatever drifted. It takes a walk this long to see it.
    #[test]
    fn growth_ba_keeps_the_long_tracks_a_terminal_solve_alone_loses() {
        let (cams, gt, tracks) = walkthrough(72, 300, 0.4);
        // `max_iterations` below the default 40 only to keep the test quick; both arms get it, and
        // the terminal solve the control depends on is the one it makes cheaper.
        let config = ReconstructionConfig {
            max_iterations: 20,
            ..ReconstructionConfig::new(0.0)
        };

        let with =
            reconstruct_inner(&cams, &[], &tracks, &config, None, true, true).expect("with BA");
        let without =
            reconstruct_inner(&cams, &[], &tracks, &config, None, false, true).expect("without BA");

        let reg = |r: &Reconstruction| r.views.iter().filter(|v| v.is_some()).count();
        let span_with = map_spans(&with);
        let span_without = map_spans(&without);
        let med = |s: &[usize]| if s.is_empty() { 0 } else { s[s.len() / 2] };
        let long = |s: &[usize]| s.iter().filter(|&&n| n >= 20).count();

        println!(
            "with growth BA:   {} / {} views, {} points, median span {}, >=20-view {}, rmse {:.3} px",
            reg(&with), cams.len(), with.points.len(), med(&span_with), long(&span_with),
            with.reproj_rmse_px,
        );
        println!(
            "terminal BA only: {} / {} views, {} points, median span {}, >=20-view {}, rmse {:.3} px",
            reg(&without), cams.len(), without.points.len(), med(&span_without), long(&span_without),
            without.reproj_rmse_px,
        );

        // Ground truth on the arm we ship, so "beats the control" cannot be met by both arms being
        // wrong.
        let med_rot = median_relative_rotation_error_deg(&with.views, &gt);
        println!("with growth BA:   median relative-rotation error {med_rot:.4} deg");
        assert_eq!(
            reg(&with),
            cams.len(),
            "the shipped arm should register every view of a clean synthetic walk"
        );
        assert!(
            med_rot < 0.1,
            "the shipped arm should also be RIGHT, not just better: median relative-rotation \
             error {med_rot:.4} deg"
        );

        // The comparison proper. Measured on this scene, in this order:
        //   >=20-view tracks 233 vs 149
        //   points           292 vs 181
        //
        // MEDIAN span is deliberately NOT compared between the arms. `filter_points` drops points
        // with fewer than two REGISTERED sightings, and in the control 17 views are unregistered —
        // so the filter removes precisely that arm's short-span points and RAISES its median (30
        // before observation filtering existed, 33 after) above the shipped arm's 32, while the
        // shipped arm loses nothing and stays ahead on every absolute count. A statistic computed
        // over survivors cannot compare two populations that were culled at different rates. The
        // absolute floor below still guards chaining in the arm being shipped.
        assert!(
            med(&span_with) >= 30,
            "growth BA median map span {} fell below the absolute floor",
            med(&span_with)
        );
        assert!(
            long(&span_with) > long(&span_without),
            "growth BA must keep more >=20-view tracks: {} vs {}",
            long(&span_with),
            long(&span_without)
        );
        assert!(
            with.points.len() > without.points.len(),
            "growth BA must triangulate more of the scene: {} vs {} points",
            with.points.len(),
            without.points.len()
        );
    }

    /// A handful of grossly wrong sightings must not survive into the published map.
    ///
    /// Mismatches are not rare in practice, and bundle adjustment cannot fix one: the point is
    /// pulled between an observation that agrees with the geometry and one that never will, and the
    /// loser is whichever side has fewer votes. Left in, they are permanent — and because
    /// [`Reconstruction::reproj_rmse_px`] is a MEAN over observations, a tail of them inflates the
    /// reported error several-fold over a map whose geometry is fine. See [`filter_points`] for the
    /// measurement on real data.
    ///
    /// Asserted on the corrupted arm's ABSOLUTE rmse rather than against a clean control, because
    /// the failure mode is exactly that the number looks plausible while the map is not: a
    /// regression that stops filtering leaves registration and geometry untouched and moves only
    /// this statistic. Registration and rotation accuracy are asserted alongside it so the filter
    /// cannot pass by discarding the good evidence with the bad.
    #[test]
    fn gross_outlier_observations_do_not_reach_the_published_map() {
        let (cams, gt, mut tracks) = walkthrough(40, 300, 0.4);
        // Every 7th track has one sighting displaced by 40 px: far outside any noise the solver
        // should tolerate, and the scale a real feature mismatch produces.
        let mut corrupted = 0usize;
        for (i, t) in tracks.iter_mut().enumerate() {
            if i % 7 == 0 && t.obs.len() >= 3 {
                t.obs[1].1 += Vec2F64::new(40.0, -40.0);
                corrupted += 1;
            }
        }
        assert!(corrupted > 20, "test needs a meaningful number of outliers");

        let config = ReconstructionConfig {
            max_iterations: 20,
            ..ReconstructionConfig::new(0.0)
        };
        let r = reconstruct_inner(&cams, &[], &tracks, &config, None, true, true).expect("recon");
        let registered = r.views.iter().filter(|v| v.is_some()).count();

        let med_rot = median_relative_rotation_error_deg(&r.views, &gt);
        println!(
            "{corrupted} corrupted tracks: {registered} / {} views, {} points, rmse {:.4} px, \
             median relative-rotation error {med_rot:.4} deg",
            cams.len(),
            r.points.len(),
            r.reproj_rmse_px,
        );

        // 0.31 px here; the same scene through this file WITHOUT the filter reports 4.27 px at an
        // identical 207 points / 40 registered views and a comparable rotation error — the
        // separation is entirely in the statistic, which is the point. `2.0` sits between the two
        // with a wide margin either way.
        assert!(
            r.reproj_rmse_px < 2.0,
            "outlier sightings reached the published map: rmse {:.3} px",
            r.reproj_rmse_px
        );
        assert_eq!(
            registered,
            cams.len(),
            "the filter must not cost registrations: {registered} / {} views",
            cams.len()
        );
        // EXACT, and load-bearing. Every other assertion here gets EASIER as the filter destroys
        // more of the map: rmse falls, registration is settled in the growth loop before the
        // terminal filter runs, and rotation error is per-view. Only a cloud floor can fail on
        // over-filtering. Each corrupted track keeps a large majority of good sightings, so none of
        // them should lose its point — see the companion test for the drop path.
        assert_eq!(
            r.points.len(),
            tracks.len(),
            "one bad sighting must cost the SIGHTING, not the point: {} of {} points survived",
            r.points.len(),
            tracks.len()
        );
        assert!(
            med_rot < 0.1,
            "the surviving map must also be RIGHT: median relative-rotation error {med_rot:.4} deg"
        );

        // Every published observation must carry the pixel ITS OWN view saw. Nothing else in this
        // test can see that: `reproj_rmse_px` is computed over `norm`, so a wrong `pixel` moves it
        // by exactly zero. `filter_points` prunes `norm` and nothing prunes `tracks`, so a
        // positional lookup emits another view's pixel for every track that lost a sighting —
        // frequently the outlier that was just rejected.
        for o in &r.observations {
            let track = &tracks[r.points[o.point].track_id];
            assert!(
                track
                    .obs
                    .iter()
                    .any(|(c, uv)| *c == o.view && *uv == o.pixel),
                "published observation for view {} carries a pixel that view never saw",
                o.view
            );
        }
    }

    /// The point-drop rule, exercised directly: below two reconcilable sightings a point goes, and
    /// it takes none of its observations with it.
    ///
    /// Driven through `filter_points` rather than the full pipeline deliberately. Ruining tracks in
    /// a reconstruction does not isolate this rule — the bad points corrupt the poses, which fails
    /// other points, which changes what the filter sees; measured on a 40-view walk with 19 ruined
    /// tracks, 44 CLEAN points died with them and 8 ruined ones survived. That feedback loop is
    /// worth knowing about but it is not this rule, and tuning such a scene until the count looks
    /// right would be fitting the test to the answer.
    ///
    /// Also pins the two invariants a caller cannot see but the rest of the file depends on:
    /// surviving tracks keep their observations in ascending camera order, and `norm_depth` is
    /// pruned in lockstep with `norm`.
    #[test]
    fn a_point_whose_evidence_is_mostly_wrong_is_dropped_whole() {
        // Four cameras abreast, identity rotation, all looking at one point ahead of them.
        let poses: Vec<Option<Pose3d>> = (0..4)
            .map(|i| {
                Some(Pose3d::new(
                    rot(0.0, 0.0),
                    Vec3F64::new(-(i as f64), 0.0, 0.0),
                ))
            })
            .collect();
        let p = Vec3F64::new(1.5, 0.0, 5.0);
        let clean: Vec<(usize, Vec2F64)> = poses
            .iter()
            .enumerate()
            .map(|(c, q)| {
                let pc = q.as_ref().expect("all posed").transform_point(&p);
                (c, Vec2F64::new(pc.x / pc.z, pc.y / pc.z))
            })
            .collect();
        // Normalized units: the clean residuals are 0, and every displacement below is >= 15x this.
        const MAX: f64 = 0.02;

        // ONE bad sighting: the point stays, that sighting goes, the rest keep their order.
        let mut point3d: HashMap<usize, Vec3F64> = HashMap::from([(0, p)]);
        let mut norm = vec![clean.clone()];
        let mut depths = vec![vec![Some(5.0f32); 4]];
        norm[0][2].1 += Vec2F64::new(0.5, 0.0);
        let dropped = filter_points(&mut point3d, &mut norm, &mut depths, &poses, MAX);
        assert_eq!(dropped, 0, "one bad sighting must not cost the point");
        assert_eq!(
            norm[0].iter().map(|(c, _)| *c).collect::<Vec<_>>(),
            vec![0, 1, 3],
            "the bad sighting must go and the survivors stay in ascending camera order"
        );
        assert_eq!(depths[0].len(), 3, "norm_depth must be pruned in lockstep");

        // THREE of four bad, each in its own direction so no pose reconciles them: one sighting is
        // left standing, which constrains nothing, so the point goes whole.
        let mut point3d: HashMap<usize, Vec3F64> = HashMap::from([(0, p)]);
        let mut norm = vec![clean.clone()];
        let mut depths = vec![vec![Some(5.0f32); 4]];
        for (j, o) in norm[0].iter_mut().enumerate().skip(1) {
            o.1 += Vec2F64::new(0.3 * j as f64, 0.2);
        }
        let dropped = filter_points(&mut point3d, &mut norm, &mut depths, &poses, MAX);
        assert_eq!(
            dropped, 1,
            "a point with one reconcilable sighting is unconstrained"
        );
        assert!(point3d.is_empty());
        assert_eq!(
            norm[0].len(),
            4,
            "a DROPPED point keeps every observation, so triangulate_new can rebuild it later"
        );
        assert_eq!(depths[0].len(), 4);
    }

    /// The sparse reduced system must change the COST of the solve, never its answer.
    ///
    /// It is a different storage and factorisation of the same matrix, so anything that makes the
    /// two disagree is a bug in the sparse assembly rather than a tuning choice — which is why this
    /// asserts equality of the reconstruction and not merely similar quality. `kornia-3d` pins the
    /// lower triangle as bit-identical at the matrix level; this pins the end-to-end result a
    /// caller of [`reconstruct`] actually receives.
    #[test]
    fn sparse_reduced_system_does_not_change_the_reconstruction() {
        let (cams, _gt, tracks) = walkthrough(40, 300, 0.4);
        let solve = |sparse: bool| {
            let config = ReconstructionConfig {
                max_iterations: 20,
                sparse_reduced_system: sparse,
                ..ReconstructionConfig::new(0.0)
            };
            reconstruct_inner(&cams, &[], &tracks, &config, None, true, true).expect("recon")
        };
        let (sp, de) = (solve(true), solve(false));

        assert_eq!(sp.points.len(), de.points.len(), "point count diverged");
        assert_eq!(
            sp.views.iter().filter(|v| v.is_some()).count(),
            de.views.iter().filter(|v| v.is_some()).count(),
            "registration diverged"
        );
        assert!(
            (sp.reproj_rmse_px - de.reproj_rmse_px).abs() < 1e-6,
            "rmse diverged: sparse {} vs dense {}",
            sp.reproj_rmse_px,
            de.reproj_rmse_px
        );
        let worst = sp
            .points
            .iter()
            .zip(&de.points)
            .map(|(a, b)| {
                let d = a.position - b.position;
                (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-9, "point positions diverged by {worst}");
    }

    /// A sighting the filter dropped under an early pose comes back once the pose is good — and the
    /// track it returns to is still in ascending camera order.
    ///
    /// This is the half of COLMAP's filter/complete pair that keeps track length from eroding
    /// monotonically: filtering only ever removes, and it judges against whatever geometry exists
    /// at the time, so without re-admission a long capture loses evidence permanently to poses it
    /// later corrects. The ordering assertion is not incidental — re-admission APPENDS, and
    /// `tracks.rs` publishes ascending camera order as contract.
    #[test]
    fn complete_tracks_readmits_what_the_filter_dropped() {
        let poses: Vec<Option<Pose3d>> = (0..4)
            .map(|i| {
                Some(Pose3d::new(
                    rot(0.0, 0.0),
                    Vec3F64::new(-(i as f64), 0.0, 0.0),
                ))
            })
            .collect();
        let p = Vec3F64::new(1.5, 0.0, 5.0);
        let full: Vec<(usize, Vec2F64)> = poses
            .iter()
            .enumerate()
            .map(|(c, q)| {
                let pc = q.as_ref().expect("posed").transform_point(&p);
                (c, Vec2F64::new(pc.x / pc.z, pc.y / pc.z))
            })
            .collect();

        // The map as the filter left it: cameras 1 and 2 were dropped against poses since improved.
        let norm0 = vec![full.clone()];
        let norm_depth0 = vec![vec![Some(5.0f32); 4]];
        let mut norm = vec![vec![full[0], full[3]]];
        let mut norm_depth = vec![vec![Some(5.0f32); 2]];
        let point3d: HashMap<usize, Vec3F64> = HashMap::from([(0, p)]);

        let added = complete_tracks(
            &point3d,
            &mut norm,
            &mut norm_depth,
            &norm0,
            &norm_depth0,
            &poses,
            0.02,
        );
        assert_eq!(added, 2, "both dropped sightings should be re-admitted");
        assert_eq!(
            norm[0].iter().map(|(c, _)| *c).collect::<Vec<_>>(),
            vec![0, 1, 2, 3],
            "re-admission must restore ascending camera order, not append"
        );
        assert_eq!(norm_depth[0].len(), 4, "norm_depth must stay index-paired");

        // A sighting that does NOT fit the current pose stays out.
        let mut norm = vec![vec![full[0], full[3]]];
        let mut norm_depth = vec![vec![Some(5.0f32); 2]];
        let mut bad0 = norm0.clone();
        bad0[0][1].1 += Vec2F64::new(0.5, 0.0);
        let added = complete_tracks(
            &point3d,
            &mut norm,
            &mut norm_depth,
            &bad0,
            &norm_depth0,
            &poses,
            0.02,
        );
        assert_eq!(added, 1, "only the sighting that fits should return");
    }

    /// Duplicate tracks fuse; tracks that only LOOK close do not.
    ///
    /// The negative cases carry the weight. Merging on distance alone would fuse a genuinely close
    /// pair and hand the solve a point that fits neither observation set, manufacturing exactly the
    /// drift merging exists to remove — and fusing two sightings that appear in the SAME image
    /// asserts one landmark was in two places at once.
    #[test]
    fn merge_tracks_fuses_duplicates_but_not_distinct_points() {
        let poses: Vec<Option<Pose3d>> = (0..4)
            .map(|i| {
                Some(Pose3d::new(
                    rot(0.0, 0.0),
                    Vec3F64::new(-(i as f64), 0.0, 0.0),
                ))
            })
            .collect();
        let proj = |p: Vec3F64, c: usize| {
            let pc = poses[c].as_ref().expect("posed").transform_point(&p);
            (c, Vec2F64::new(pc.x / pc.z, pc.y / pc.z))
        };
        let p = Vec3F64::new(1.5, 0.0, 5.0);

        // ONE landmark, split across two tracks by different match chains: cameras 0,1 and 2,3.
        let mut point3d: HashMap<usize, Vec3F64> = HashMap::from([(0, p), (1, p)]);
        let mut norm = vec![vec![proj(p, 0), proj(p, 1)], vec![proj(p, 2), proj(p, 3)]];
        let mut norm_depth = vec![vec![None; 2], vec![None; 2]];
        let n = merge_tracks(&mut point3d, &mut norm, &mut norm_depth, &poses, 0.02, 0.05);
        assert_eq!(n, 1, "a split landmark must fuse");
        assert_eq!(point3d.len(), 1);
        assert_eq!(
            norm[0].iter().map(|(c, _)| *c).collect::<Vec<_>>(),
            vec![0, 1, 2, 3],
            "merged track must hold both halves in ascending camera order"
        );
        assert!(
            norm[1].is_empty(),
            "the absorbed track's observations must MOVE, or triangulate_new rebuilds it"
        );

        // Two DISTINCT points inside the radius: close, but no merged position fits both.
        let q = Vec3F64::new(1.54, 0.0, 5.0);
        let mut point3d: HashMap<usize, Vec3F64> = HashMap::from([(0, p), (1, q)]);
        let mut norm = vec![vec![proj(p, 0), proj(p, 1)], vec![proj(q, 2), proj(q, 3)]];
        let mut norm_depth = vec![vec![None; 2], vec![None; 2]];
        let n = merge_tracks(
            &mut point3d,
            &mut norm,
            &mut norm_depth,
            &poses,
            0.001,
            0.05,
        );
        assert_eq!(n, 0, "distinct points must not fuse on proximity alone");
        assert_eq!(point3d.len(), 2);

        // Same camera in both tracks: never one landmark, whatever the distance says.
        let mut point3d: HashMap<usize, Vec3F64> = HashMap::from([(0, p), (1, p)]);
        let mut norm = vec![vec![proj(p, 0), proj(p, 1)], vec![proj(p, 1), proj(p, 2)]];
        let mut norm_depth = vec![vec![None; 2], vec![None; 2]];
        let n = merge_tracks(&mut point3d, &mut norm, &mut norm_depth, &poses, 0.02, 0.05);
        assert_eq!(n, 0, "tracks sharing a camera must not fuse");
    }

    /// A forward walk: `n_views` cameras marching along their own optical axis toward a shell of
    /// landmarks ahead, every landmark visible in every view. Returns `(cameras, tracks)`.
    ///
    /// This is the geometry a handheld video capture actually has, and the geometry that breaks a
    /// most-shared-tracks bootstrap. Because every landmark is seen everywhere, EVERY view pair
    /// shares the same number of tracks, so the shared-track criterion cannot distinguish them at
    /// all and falls through to its `(a, b)` tie-break — landing on the ADJACENT pair `(0, 1)`,
    /// whose 6 cm baseline against 7-8 m landmarks is the smallest in the sequence. That is not an
    /// artefact of the tie: on a real forward walk, adjacency is what MAXIMISES overlap, so the
    /// argmax lands on an adjacent pair for the same reason with counts that genuinely differ.
    ///
    /// The numbers are chosen so the two regimes are unambiguous at the DEFAULT
    /// `min_parallax_deg` of 0.2. MEASURED on this scene: the adjacent pair's median triangulation
    /// angle is 0.16 deg and it triangulates 3 of the 240 tracks — below the growth loop's 4-point
    /// admission bar, so the map ends at its two seed views. `(0, 9)` is the first pair to clear
    /// the 1.5 deg seed floor, at 1.54 deg, and it triangulates all 240.
    fn forward_walk(n_views: usize) -> (Vec<PinholeCamera>, Vec<FeatureTrack>) {
        const F: f64 = 520.0;
        const STEP: f64 = 0.06;
        let (w, h) = (640.0, 480.0);

        // xorshift64: deterministic, self-contained, and — unlike a trig lattice — it does not put
        // the landmarks on a surface the homography arm could fit.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 100_000) as f64 / 100_000.0
        };

        let cams: Vec<PinholeCamera> = (0..n_views).map(|_| pinhole(F)).collect();
        // Camera i sits at (0, 0, i * STEP) looking down +Z, i.e. world→cam is a pure -z translation.
        let gt: Vec<Pose3d> = (0..n_views)
            .map(|i| {
                Pose3d::new(
                    Mat3F64::IDENTITY,
                    Vec3F64::new(0.0, 0.0, -(i as f64) * STEP),
                )
            })
            .collect();

        // A shell of landmarks 7.0-8.5 m ahead at 0.45-0.55 of their depth off-axis. A shell rather
        // than a filled volume so the parallax spread stays tight: with the depth and the off-axis
        // fraction both wandering, the low-parallax tail of a wide pair would overlap the
        // high-parallax tail of the adjacent one and neither regime would be clean.
        let mut tracks: Vec<FeatureTrack> = Vec::new();
        for _ in 0..240 {
            let z = 7.0 + 1.5 * rnd();
            let theta = std::f64::consts::TAU * rnd();
            let rho = 0.45 + 0.10 * rnd();
            let p = Vec3F64::new(rho * z * theta.cos(), 0.70 * rho * z * theta.sin(), z);
            let obs: Vec<(usize, Vec2F64)> = (0..n_views)
                .filter_map(|c| {
                    let pc = gt[c].transform_point(&p);
                    if pc.z <= 0.5 {
                        return None;
                    }
                    let uv = project(p, &gt[c], &cams[c]);
                    (uv.x >= 0.0 && uv.x < w && uv.y >= 0.0 && uv.y < h).then_some((c, uv))
                })
                .collect();
            if obs.len() >= 2 {
                tracks.push(FeatureTrack { obs });
            }
        }
        (cams, tracks)
    }

    /// **Choosing the bootstrap pair on shared-track count strands the reconstruction at the seed.**
    ///
    /// Asserted on the number of views the SfM registers BY ITSELF, because that is the quantity
    /// the seed actually decides and the only one a caller's own repair layer cannot inflate: an
    /// optical-flow or dead-reckoning fallback downstream will happily report a full trajectory on
    /// top of a two-view map. Point count would not do either — the argmax arm's failure is that
    /// its seed cloud is too thin to register anything, so counting points measures the symptom
    /// through a second lens rather than the outcome.
    ///
    /// Both arms run the identical pipeline on the identical scene; `ranked_seed` is the single
    /// variable. The comparison is against the OLD behaviour rather than a fixed threshold, so this
    /// cannot pass on a change that does nothing.
    #[test]
    fn ranked_bootstrap_registers_more_views_than_the_most_shared_tracks_seed() {
        let n_views = 14;
        let (cams, tracks) = forward_walk(n_views);
        assert_eq!(
            tracks.len(),
            240,
            "every landmark should be seen by every view"
        );
        let config = ReconstructionConfig::new(0.0);

        let registered = |r: &Reconstruction| r.views.iter().filter(|p| p.is_some()).count();

        let ranked = reconstruct_inner(&cams, &[], &tracks, &config, None, true, true)
            .expect("ranked seed selection must reconstruct");
        // The argmax arm is allowed to fail outright — a seed that triangulates nothing is exactly
        // the failure under test — so an error counts as zero views rather than aborting the test.
        let argmax = reconstruct_inner(&cams, &[], &tracks, &config, None, true, false);
        let n_ranked = registered(&ranked);
        let n_argmax = argmax.as_ref().map(registered).unwrap_or(0);

        assert!(
            n_ranked > n_argmax,
            "ranked seed registered {n_ranked} views, most-shared-tracks seed registered \
             {n_argmax} of {n_views} — the ranked selection must register STRICTLY more, or this \
             scene does not exercise the change"
        );
        assert_eq!(
            n_ranked, n_views,
            "the ranked seed should register the whole walk, not merely beat the argmax"
        );
        // Pin the failure mode itself: the argmax arm must stall at (or below) the seed pair, not
        // merely register somewhat fewer views. Without this the test would keep passing if the
        // argmax arm degraded gracefully instead of collapsing, which is a different claim.
        assert!(
            n_argmax <= 2,
            "expected the most-shared-tracks seed to strand the map at its two seed views, got \
             {n_argmax}"
        );
    }
}
