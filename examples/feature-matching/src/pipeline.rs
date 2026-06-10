//! Classical "find a known object in a scene" pipeline, mirroring OpenCV's
//! [Features2D + Homography] tutorial but with XFeat learned features instead
//! of SIFT/ORB.
//!
//! ## What we mirror from the OpenCV tutorial
//!
//! The OpenCV C++/Python tutorial *"Features2D + Homography to find a known
//! object"* (a.k.a. `find_obj`) runs this exact pipeline:
//!
//! 1. Detect keypoints + compute descriptors on both the reference (object)
//!    and the scene image.
//! 2. Match descriptors with a k-NN matcher (k = 2).
//! 3. **Lowe ratio test** at `0.75`: keep a match only if the best neighbour
//!    is clearly closer than the second-best (`d1 < 0.75 * d2`).
//! 4. `findHomography(srcPoints, dstPoints, RANSAC, 3.0)` — robustly fit a
//!    planar homography with a 3-pixel reprojection threshold.
//! 5. `perspectiveTransform` the reference image's four corners through `H`
//!    and draw the projected quad on the scene to outline the found object.
//!
//! We keep the same parameter choices:
//! - ratio test = `0.75` (Lowe),
//! - RANSAC reprojection threshold = `3.0` px,
//! - up to `2000` RANSAC iterations with confidence-based early exit
//!   (`0.995`), matching `findHomography`'s default `maxIters`/`confidence`,
//! - optional least-squares refit (DLT) on the inliers, equivalent to the
//!   final inlier-only re-estimation OpenCV performs internally.
//!
//! XFeat descriptors are 64-dim L2-normalised floats, so "distance" is in
//! cosine space; we apply Lowe's ratio in cosine space exactly as
//! [`kornia_imgproc::features::match_descriptors_f32`] does, plus a mutual
//! nearest-neighbour (cross-check) gate which is the modern strengthening of
//! the bare ratio test.
//!
//! Everything here is webcam-agnostic and headless-testable: it takes two f32
//! gray buffers and returns named structs.
//!
//! [Features2D + Homography]: https://docs.opencv.org/4.x/d7/dff/tutorial_feature_homography.html

use kornia_3d::ransac::{
    estimators::HomographyEstimator, run, ConsensusKind, Match2d2d, RansacConfig,
    ThresholdConsensus, UniformSampler,
};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_xfeat::{KeyPoint, PackedWeights, XFeat, XFeatConfig};
use rand::{rngs::StdRng, SeedableRng};

/// XFeat descriptor dimensionality (fixed by the model).
pub const DESC_DIM: usize = 64;

/// Minimum cosine similarity for a match, following the upstream XFeat
/// convention (`XFeat.match_mkpts` uses `min_cossim = 0.82` with mutual
/// nearest-neighbour filtering). We deliberately depart from the OpenCV
/// SIFT tutorial's Lowe ratio test here: learned descriptors like XFeat's
/// have much flatter nearest-neighbour distance distributions than SIFT,
/// so a 0.75 cosine-space ratio rejects nearly everything (measured: 1
/// match on the fixture pair vs hundreds with the cosine gate).
pub const MIN_COSSIM: f32 = 0.82;

/// RANSAC reprojection threshold in pixels (OpenCV `findHomography` default for
/// this tutorial is `3.0`).
pub const RANSAC_REPROJ_PX: f64 = 3.0;

/// RANSAC iteration cap (OpenCV `findHomography` default `maxIters = 2000`).
pub const RANSAC_MAX_ITERS: u32 = 2000;

/// RANSAC confidence (OpenCV `findHomography` default `confidence = 0.995`).
pub const RANSAC_CONFIDENCE: f64 = 0.995;

/// Fixed RNG seed for reproducible RANSAC. No wall-clock seeding — the example
/// must be deterministic so the headless offline mode is testable.
pub const RANSAC_SEED: u64 = 0x5EED_F00D_CAFE_1234;

/// Extracted features for one image: keypoints (in the image's pixel coords)
/// and their parallel descriptor rows.
pub struct Features {
    /// Keypoints, sorted by descending score (as XFeat returns them).
    pub keypoints: Vec<KeyPoint>,
    /// Per-keypoint 64-dim L2-normalised descriptors.
    pub descriptors: Vec<[f32; DESC_DIM]>,
}

impl Features {
    /// Number of detected keypoints.
    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    /// Whether no keypoints were detected.
    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }
}

/// Run XFeat on a single pre-aligned f32 gray buffer (`h * w`, dims multiples
/// of 32). Returns owned keypoints + descriptors so the borrowed
/// [`kornia_xfeat::XFeatOutput`] view doesn't escape.
///
/// A fresh [`XFeat`] is built per call: the model's zero-alloc arena is sized
/// to one resolution, and ref / scene generally differ in size.
pub fn extract_features(
    gray: &[f32],
    h: usize,
    w: usize,
) -> Result<Features, Box<dyn std::error::Error>> {
    let weights = PackedWeights::from_safetensors_bytes(kornia_xfeat::weights::embedded_bytes())?;
    let cfg = XFeatConfig {
        height: h,
        width: w,
        ..XFeatConfig::default()
    };
    let mut model = XFeat::new(cfg, weights)?;
    let out = model.extract(gray)?;

    let keypoints = out.keypoints.to_vec();
    let mut descriptors = Vec::with_capacity(keypoints.len());
    for row in out.descriptors.chunks_exact(DESC_DIM) {
        let mut d = [0.0f32; DESC_DIM];
        d.copy_from_slice(row);
        descriptors.push(d);
    }

    Ok(Features {
        keypoints,
        descriptors,
    })
}

/// A surviving descriptor correspondence: indices into the reference and scene
/// feature sets respectively.
#[derive(Debug, Clone, Copy)]
pub struct FeatureMatch {
    /// Index into the reference [`Features`].
    pub ref_idx: usize,
    /// Index into the scene [`Features`].
    pub scene_idx: usize,
}

/// Match reference descriptors against scene descriptors with mutual
/// nearest-neighbour cross-check plus an absolute cosine-similarity gate —
/// the upstream XFeat matching convention (see [`MIN_COSSIM`] for why this
/// replaces the OpenCV tutorial's Lowe ratio test).
///
/// Delegates to [`kornia_imgproc::features::match_descriptors_f32`], the
/// workspace's XFeat-convention cosine matcher.
pub fn match_features(reference: &Features, scene: &Features) -> Vec<FeatureMatch> {
    use kornia::imgproc::features::match_descriptors_f32;

    let pairs = match_descriptors_f32::<DESC_DIM>(
        &reference.descriptors,
        &scene.descriptors,
        Some(MIN_COSSIM), // absolute cosine gate (XFeat convention)
        true,             // mutual nearest neighbour (cross-check)
        None,             // no Lowe ratio — hostile to learned descriptors
    );

    pairs
        .into_iter()
        .map(|(ref_idx, scene_idx)| FeatureMatch { ref_idx, scene_idx })
        .collect()
}

/// One image corner / its projection, as `(x, y)` pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Corner {
    /// X (column) in pixels.
    pub x: f64,
    /// Y (row) in pixels.
    pub y: f64,
}

/// Result of the homography stage: the estimated `3x3` H (reference → scene),
/// the inlier mask over the input matches, the inlier count, and the reference
/// rectangle's four corners projected into the scene (OpenCV
/// `perspectiveTransform` step).
pub struct HomographyResult {
    /// Estimated homography mapping reference pixels to scene pixels, row-major.
    pub h: [[f64; 3]; 3],
    /// Inlier mask aligned to the `matches` slice passed in.
    pub inliers: Vec<bool>,
    /// Number of inliers.
    pub inlier_count: usize,
    /// Number of RANSAC iterations actually consumed (≤ [`RANSAC_MAX_ITERS`]).
    pub num_iters: u32,
    /// Reference rectangle corners projected through `h`, in scene pixels.
    /// Order: top-left, top-right, bottom-right, bottom-left.
    pub projected_corners: [Corner; 4],
}

/// Estimate a homography from reference→scene matches with RANSAC, then project
/// the reference image's four corners through it.
///
/// `ref_w` / `ref_h` are the reference image dimensions (in the same pixel
/// coordinate frame as `reference.keypoints`); they define the rectangle whose
/// corners get projected. Returns `None` if fewer than 4 matches are available
/// or RANSAC fails to find any consensus model.
///
/// Mirrors `findHomography(..., RANSAC, 3.0)` + `perspectiveTransform`.
pub fn estimate_homography(
    reference: &Features,
    scene: &Features,
    matches: &[FeatureMatch],
    ref_w: usize,
    ref_h: usize,
) -> Option<HomographyResult> {
    if matches.len() < 4 {
        return None;
    }

    // Build 2D-2D correspondences in pixel coordinates: ref → scene.
    let samples: Vec<Match2d2d> = matches
        .iter()
        .map(|m| {
            let r = &reference.keypoints[m.ref_idx];
            let s = &scene.keypoints[m.scene_idx];
            Match2d2d::new(
                Vec2F64::new(r.x as f64, r.y as f64),
                Vec2F64::new(s.x as f64, s.y as f64),
            )
        })
        .collect();

    // Classic threshold RANSAC. The homography estimator's residual is the
    // forward transfer error *squared* in pixels, so the OpenCV 3px reprojection
    // threshold becomes 3² = 9 in residual units.
    let estimator = HomographyEstimator;
    let consensus = ThresholdConsensus {
        threshold: RANSAC_REPROJ_PX * RANSAC_REPROJ_PX,
    };
    let mut sampler = UniformSampler::new(StdRng::seed_from_u64(RANSAC_SEED));
    let cfg = RansacConfig {
        max_iters: RANSAC_MAX_ITERS,
        confidence: RANSAC_CONFIDENCE,
        inlier_threshold: RANSAC_REPROJ_PX * RANSAC_REPROJ_PX,
        // Local optimisation refit every 10 accepted hypotheses: this is the
        // least-squares (DLT) re-fit on the inlier set, equivalent to OpenCV's
        // final inlier-only re-estimation.
        lo_every: 10,
        parallel: false,
        consensus: ConsensusKind::Threshold,
    };

    let result = run(&estimator, &consensus, &mut sampler, &samples, &cfg);
    let model: Mat3F64 = result.model?;

    let h = model.to_cols_array(); // column-major [m00,m10,m20, m01,m11,m21, m02,m12,m22]
    let h_rows = [[h[0], h[3], h[6]], [h[1], h[4], h[7]], [h[2], h[5], h[8]]];

    // perspectiveTransform of the reference rectangle's corners (TL, TR, BR, BL).
    let w = ref_w as f64;
    let ht = ref_h as f64;
    let rect = [[0.0, 0.0], [w, 0.0], [w, ht], [0.0, ht]];
    let mut projected = [Corner { x: 0.0, y: 0.0 }; 4];
    for (i, &[px, py]) in rect.iter().enumerate() {
        let v = model * Vec3F64::new(px, py, 1.0);
        let z = if v.z.abs() < 1e-12 { 1e-12 } else { v.z };
        projected[i] = Corner {
            x: v.x / z,
            y: v.y / z,
        };
    }

    let inlier_count = result.inlier_count();
    Some(HomographyResult {
        h: h_rows,
        inliers: result.inliers,
        inlier_count,
        num_iters: result.num_iters,
        projected_corners: projected,
    })
}
