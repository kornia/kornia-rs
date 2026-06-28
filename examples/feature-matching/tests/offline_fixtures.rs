//! Headless integration test: run the full feature-matching + homography
//! pipeline on the two committed XFeat fixtures (two views of the same planar
//! scene) and assert a sensible homography with many inliers.
//!
//! This is the testable form of the example's offline mode. It deliberately
//! depends only on the example's own modules, included here via `#[path]`, so
//! the pipeline stays a single source of truth.

// The test exercises a subset of the pipeline API; items used only by the
// example binary (e.g. `Features::is_empty`, `HomographyResult::num_iters`)
// are dead code in THIS compilation unit only.
#[allow(dead_code)]
#[path = "../src/pipeline.rs"]
mod pipeline;
#[allow(dead_code)]
#[path = "../src/preprocess.rs"]
mod preprocess;

use std::path::PathBuf;

use kornia::io::functional::read_image_any_rgb8;

use pipeline::{estimate_homography, extract_features, match_features};
use preprocess::rgb8_to_aligned_gray;

fn fixture(kind: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/kornia-xfeat/tests/fixtures/v1")
        .join(kind)
        .join("input.png")
}

fn load_aligned(kind: &str) -> preprocess::AlignedGray {
    let rgb = read_image_any_rgb8(fixture(kind)).expect("read fixture png");
    let (h, w) = (rgb.height(), rgb.width());
    rgb8_to_aligned_gray(rgb.as_slice(), h, w).expect("align fixture to 32")
}

#[test]
fn fixture_pair_yields_sensible_homography() {
    let ref_gray = load_aligned("ref");
    let scene_gray = load_aligned("tgt");

    let ref_feats =
        extract_features(&ref_gray.data, ref_gray.height, ref_gray.width).expect("extract ref");
    let scene_feats = extract_features(&scene_gray.data, scene_gray.height, scene_gray.width)
        .expect("extract scene");

    // XFeat should find a healthy number of keypoints on these textured images.
    assert!(
        ref_feats.len() > 100,
        "too few ref keypoints: {}",
        ref_feats.len()
    );
    assert!(
        scene_feats.len() > 100,
        "too few scene keypoints: {}",
        scene_feats.len()
    );

    let matches = match_features(&ref_feats, &scene_feats);
    assert!(
        matches.len() >= 4,
        "need >= 4 matches for a homography, got {}",
        matches.len()
    );

    let h = estimate_homography(
        &ref_feats,
        &scene_feats,
        &matches,
        ref_gray.width,
        ref_gray.height,
    )
    .expect("RANSAC should find a homography on two views of the same scene");

    // The fixtures are two views of the same planar scene, so we expect a
    // substantial inlier set, not a lucky 4-point fit.
    assert!(
        h.inlier_count >= 15,
        "expected many inliers, got {} / {} matches",
        h.inlier_count,
        matches.len()
    );

    // Inlier mask must agree with the cached count and be aligned to `matches`.
    assert_eq!(h.inliers.len(), matches.len());
    assert_eq!(h.inliers.iter().filter(|&&b| b).count(), h.inlier_count);

    // H must be finite and the projected corners must land at finite locations.
    for row in &h.h {
        for &v in row {
            assert!(v.is_finite(), "H has non-finite entry");
        }
    }
    for c in &h.projected_corners {
        assert!(c.x.is_finite() && c.y.is_finite(), "non-finite corner");
    }

    // Determinant of the top-left 2x2 should be non-degenerate (the object
    // didn't collapse to a line / point under H).
    let det2 = h.h[0][0] * h.h[1][1] - h.h[0][1] * h.h[1][0];
    assert!(
        det2.abs() > 1e-6,
        "near-degenerate homography (det2={det2})"
    );
}
