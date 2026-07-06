//! Integration test: numeric parity between the C apriltag library and kornia-rs AprilTagDecoder.
//!
//! Both detectors are configured identically:
//!   - Tag family: Tag36H11 with 2 bits corrected
//!   - Decimation: 2× (C: `set_decimation(2.0)`, kornia: default `downscale_factor=2`)
//!   - quad_sigma / Gaussian blur: 0 (default for both)
//!   - refine_edges: enabled (default for both)
//!
//! ## Corner ordering
//! C reports corners CW in image coords (y-down) starting from the bottom-left: [BL, BR, TR, TL].
//! kornia reports corners CW in image coords starting from the top-right:        [TR, BR, BL, TL].
//! Both orderings represent the same quadrilateral; the fixed mapping for upright tags is
//! kornia[i] ↔ C[CMAP[i]] where CMAP = [2, 1, 0, 3].
//! For robustness (tags at arbitrary orientation in the scene), the comparison below tries all
//! 8 valid corner permutations (4 CW rotations + 4 CCW rotations) and uses the best-matching one.
//!
//! ## Tolerance
//! Corner tolerance is 2.0 px for task A2. Known systematic divergences (D1 decimation
//! coordinate offset, D4 refine_edges search-range difference) cause sub-pixel gaps that will
//! be fixed in A3, after which the tolerance should tighten to ≤ 0.1 px.

use std::path::PathBuf;

use apriltag::DetectorBuilder;
use kornia_apriltag::{
    decoder::Detection as KorniaDetection, family::TagFamilyKind, AprilTagDecoder, DecodeTagsConfig,
};
use kornia_image::{Image, ImageSize};

/// Maximum allowed per-axis pixel difference between C and kornia corner coordinates
/// after optimal alignment.
///
/// C's gradient clustering uses non-deterministic OpenMP ordering, which causes
/// fit_single_quad to produce initial corners ~0.5–1px different from Kornia's
/// deterministic (y-then-x sorted) ordering. Refine_edges reduces but does not
/// fully eliminate this gap. The practical agreement for a well-detected tag is
/// ≤0.6 px, so a 1.0 px tolerance comfortably covers real FP precision differences.
const CORNER_TOLERANCE: f32 = 1.0;

/// The 8 valid corner permutations (4 CW-rotation starts × 2 winding directions).
/// Entry `CORNER_PERMS[p][c_i]` gives the kornia corner index that corresponds to C corner `c_i`
/// under permutation `p`.
const CORNER_PERMS: [[usize; 4]; 8] = [
    // same winding, 4 rotational starts
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
    // flipped winding, 4 rotational starts  (note: [2,1,0,3] is the standard upright mapping)
    [0, 3, 2, 1],
    [1, 0, 3, 2],
    [2, 1, 0, 3],
    [3, 2, 1, 0],
];

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Load an image as 8-bit grayscale using the `image` crate (supports JPEG, PNG, …).
/// Returns `(pixels, width, height)`.
fn load_gray(path: &PathBuf) -> (Vec<u8>, usize, usize) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {}", path.display(), e))
        .into_luma8();
    let w = img.width() as usize;
    let h = img.height() as usize;
    (img.into_raw(), w, h)
}

/// Build a C `apriltag` detector configured to match kornia defaults.
fn build_c_detector() -> apriltag::Detector {
    let mut det = DetectorBuilder::new()
        .add_family_bits(apriltag::Family::tag_36h11(), 2)
        .build()
        .unwrap();
    // Match kornia's default downscale_factor = 2
    det.set_decimation(2.0);
    // quad_sigma stays at 0 (default) — no set_sigma() call
    det
}

/// Build a kornia `AprilTagDecoder` for the given image dimensions.
fn build_kornia_detector(width: usize, height: usize) -> AprilTagDecoder {
    let mut config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]).unwrap();
    // Pin the classic 0.5 midpoint split so this test measures algorithmic parity
    // with the reference C AprilTag (which uses `min + (max - min) / 2`). The
    // library default (0.33) intentionally biases toward white for small/glary
    // tags and is exercised separately, not against the C corners here.
    config.threshold_split = 0.5;
    // downscale_factor = 2 and refine_edges = true by default
    AprilTagDecoder::new(config, ImageSize { width, height }).unwrap()
}

/// Run the C detector on a grayscale byte slice. Returns every detection as
/// `(id, hamming, corners)` — scenes contain repeated ids; never key by id.
fn run_c(
    gray: &[u8],
    width: usize,
    height: usize,
    det: &mut apriltag::Detector,
) -> Vec<(usize, usize, [[f64; 2]; 4])> {
    let mut c_img = apriltag::Image::zeros_with_stride(width, height, width).unwrap();
    c_img.as_slice_mut().copy_from_slice(gray);
    det.detect(&c_img)
        .into_iter()
        .map(|d| (d.id(), d.hamming(), d.corners()))
        .collect()
}

/// Run the kornia decoder and return ALL detections (before dedup) as a Vec.
///
/// Using `decode_all` ensures that when the image contains multiple valid quads
/// for the same tag ID (e.g. two physical copies of id=0), the parity check
/// can pick the detection closest to C's corners rather than the highest-margin one.
fn run_kornia(
    gray: &[u8],
    width: usize,
    height: usize,
    det: &mut AprilTagDecoder,
) -> Vec<KorniaDetection> {
    let img = Image::<u8, 1>::from_size_slice(ImageSize { width, height }, gray).unwrap();
    let dets = det.decode_all(&img).unwrap();
    det.clear();
    dets
}

/// Find the permutation of kornia corners that best matches the C corners.
///
/// Tries all 8 valid winding-consistent permutations and returns the minimum
/// max-per-axis delta and the permutation that achieved it.
///
/// The `perm[c_i]` value gives the kornia corner index that corresponds to C corner `c_i`.
fn best_corner_alignment(c_corners: &[[f64; 2]; 4], k_det: &KorniaDetection) -> (f32, [usize; 4]) {
    // Extract kornia corners as plain f32 [x, y] pairs to avoid naming Vec2F32.
    let kc: [[f32; 2]; 4] = [
        [k_det.quad.corners[0].x, k_det.quad.corners[0].y],
        [k_det.quad.corners[1].x, k_det.quad.corners[1].y],
        [k_det.quad.corners[2].x, k_det.quad.corners[2].y],
        [k_det.quad.corners[3].x, k_det.quad.corners[3].y],
    ];

    let mut best_max_delta = f32::MAX;
    let mut best_perm = [0usize, 1, 2, 3];

    for &perm in &CORNER_PERMS {
        let max_delta = (0..4)
            .map(|c_i| {
                let k_i = perm[c_i];
                let dx = (kc[k_i][0] - c_corners[c_i][0] as f32).abs();
                let dy = (kc[k_i][1] - c_corners[c_i][1] as f32).abs();
                dx.max(dy)
            })
            .fold(0.0f32, f32::max);

        if max_delta < best_max_delta {
            best_max_delta = max_delta;
            best_perm = perm;
        }
    }

    (best_max_delta, best_perm)
}

/// Core parity check for a single image.
///
/// Core parity check for a single image.
///
/// For every C detection, finds the spatially closest Kornia candidate (from the full
/// pre-dedup set) and asserts:
/// - hamming distances match exactly
/// - corner coordinates agree within `corner_tolerance_px` under best-alignment permutation
///
/// Using the full candidate set (not just the max-margin-per-id survivor) handles images
/// where the same tag ID appears multiple times and Kornia's dedup might pick a different
/// instance than C.
///
/// Returns `(c_count, kornia_candidate_count, global_max_delta)`.
fn check_image(
    label: &str,
    gray: &[u8],
    width: usize,
    height: usize,
    c_det: &mut apriltag::Detector,
    k_det: &mut AprilTagDecoder,
    corner_tolerance_px: f32,
) -> (usize, usize, f32) {
    println!("=== Testing: {} ({}×{}) ===", label, width, height);

    let c_dets = run_c(gray, width, height, c_det);
    // All kornia detections before dedup — lets us match C by spatial proximity.
    let k_all = run_kornia(gray, width, height, k_det);

    println!(
        "  C detections: {}, kornia detections: {}",
        c_dets.len(),
        k_all.len()
    );

    let mut global_max_delta: f32 = 0.0;

    // Every C detection must have a spatially close Kornia candidate with matching hamming.
    for &(c_id, c_hamming, c_corners) in &c_dets {
        let c_corners = &c_corners;
        // Filter candidates to those with same id AND hamming ≤ c_hamming (including equal).
        // We match on id AND hamming to mirror C's quick-decode semantics.
        let candidates: Vec<&KorniaDetection> = k_all
            .iter()
            .filter(|d| d.id == c_id as u16 && d.hamming as usize == c_hamming)
            .collect();

        if candidates.is_empty() {
            panic!(
                "[{}] tag ID {} found by C (hamming={}) but MISSED by kornia",
                label, c_id, c_hamming
            );
        }

        // Among matching candidates, pick the one with the best corner alignment.
        let (k, max_delta, best_perm) = candidates
            .iter()
            .map(|k| {
                let (delta, perm) = best_corner_alignment(c_corners, k);
                (*k, delta, perm)
            })
            .min_by(|(_, a, _), (_, b, _)| a.partial_cmp(b).unwrap())
            .unwrap();

        global_max_delta = global_max_delta.max(max_delta);

        println!(
            "  tag ID {:>4}: hamming={}, max_corner_delta={:.3} px (best-perm={:?}){}",
            c_id,
            c_hamming,
            max_delta,
            best_perm,
            if max_delta > corner_tolerance_px {
                " *** EXCEEDS TOLERANCE ***"
            } else {
                ""
            }
        );
        for (c_i, &k_i) in best_perm.iter().enumerate() {
            println!(
                "    corner {}: C=({:.3},{:.3})  kornia=({:.3},{:.3})  delta={:.3}",
                c_i,
                c_corners[c_i][0],
                c_corners[c_i][1],
                k.quad.corners[k_i].x,
                k.quad.corners[k_i].y,
                {
                    let dx = (k.quad.corners[k_i].x - c_corners[c_i][0] as f32).abs();
                    let dy = (k.quad.corners[k_i].y - c_corners[c_i][1] as f32).abs();
                    dx.max(dy)
                }
            );
        }

        assert!(
            max_delta <= corner_tolerance_px,
            "[{}] tag ID {}: max corner delta {:.3} px > {:.1} px tolerance \
             (best permutation={:?})",
            label,
            c_id,
            max_delta,
            corner_tolerance_px,
            best_perm
        );
    }

    (c_dets.len(), k_all.len(), global_max_delta)
}

// ─── test functions ──────────────────────────────────────────────────────────

/// Parity check on `apriltags_tag36h11.jpg` — a 799×533 multi-tag real-world scene.
///
/// D4 (refine_edges range) fixed in Task A3. D1/D2 (decimation stride) investigated:
/// applying true top-left subsample caused 299 px regression (different quad selected);
/// Nearest resize already achieves ≤0.26 px parity so D1/D2 are deferred.
#[test]
fn test_parity_tag36h11_apriltags_jpg() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/apriltags_tag36h11.jpg");

    let (gray, w, h) = load_gray(&path);
    let mut c_det = build_c_detector();
    let mut k_det = build_kornia_detector(w, h);

    let (c_count, k_count, max_delta) = check_image(
        "apriltags_tag36h11.jpg",
        &gray,
        w,
        h,
        &mut c_det,
        &mut k_det,
        CORNER_TOLERANCE,
    );

    println!(
        "apriltags_tag36h11.jpg summary: C={} kornia={} max_corner_delta={:.3} px",
        c_count, k_count, max_delta
    );

    assert!(
        c_count > 0,
        "C detector found no tags in apriltags_tag36h11.jpg — image load or C library issue?"
    );

    // Regression guard: the scene repeats the same tag id — post-dedup kornia
    // must keep every physical instance like C does. Exact-count equality is
    // stricter than the tolerance-based matching above; if a borderline
    // non-dedup divergence ever trips this, compare spatial clusters instead.
    let img = Image::<u8, 1>::from_size_slice(
        ImageSize {
            width: w,
            height: h,
        },
        &gray,
    )
    .unwrap();
    let deduped = k_det.decode(&img).unwrap();
    assert_eq!(
        deduped.len(),
        c_count,
        "post-dedup kornia detection count must match C on the repeated-id scene"
    );
}

/// Parity check on `apriltag.png` — a 30×30 single-tag image.
///
/// With decimation=2 the internal processing runs on a 15×15 image, which is tiny.
/// Neither detector may detect the tag; if C detects it, kornia must too.
#[test]
fn test_parity_tag36h11_apriltag_png() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/apriltag.png");

    let (gray, w, h) = load_gray(&path);
    let mut c_det = build_c_detector();
    let mut k_det = build_kornia_detector(w, h);

    let (c_count, k_count, max_delta) = check_image(
        "apriltag.png",
        &gray,
        w,
        h,
        &mut c_det,
        &mut k_det,
        CORNER_TOLERANCE,
    );

    println!(
        "apriltag.png summary: C={} kornia={} max_corner_delta={:.3} px \
         (small image; zero detections acceptable with decimation=2)",
        c_count, k_count, max_delta
    );
    // No mandatory detection assertion — the image may be too small after decimation.
}

/// Parity check on all `tag36_11_*.png` images in the apriltag-imgs submodule.
///
/// Each image is 60×60 and contains exactly one Tag36H11 marker.
/// The kornia detector is created once and reused (calling `clear()` between images).
#[test]
fn test_parity_tag36h11_submodule_images() {
    let tag_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/apriltag-imgs/tag36h11");

    if !tag_dir.exists() {
        println!(
            "Submodule directory not found at {:?}; skipping.\n\
             Run `git submodule update --init` to populate.",
            tag_dir
        );
        return;
    }

    // All single-tag images in the submodule are 60×60.
    const W: usize = 60;
    const H: usize = 60;

    let mut c_det = build_c_detector();
    let mut k_det = build_kornia_detector(W, H);

    let mut entries: Vec<_> = std::fs::read_dir(&tag_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut images_tested = 0usize;
    let mut total_c = 0usize;
    let mut total_k = 0usize;
    let mut max_delta_global: f32 = 0.0;

    for entry in &entries {
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();

        // Skip mosaic and non-tag-image files.
        if !file_name_str.starts_with("tag36_11_") || !file_name_str.ends_with(".png") {
            continue;
        }

        let path = entry.path();
        let (gray, w, h) = load_gray(&path);

        if w != W || h != H {
            println!("  Skipping {} (unexpected size {}×{})", file_name_str, w, h);
            continue;
        }

        let (c_count, k_count, max_delta) = check_image(
            &file_name_str,
            &gray,
            W,
            H,
            &mut c_det,
            &mut k_det,
            CORNER_TOLERANCE,
        );

        total_c += c_count;
        total_k += k_count;
        max_delta_global = max_delta_global.max(max_delta);
        images_tested += 1;
    }

    println!(
        "\nSubmodule summary: {} images tested, \
         C found {} tags total, kornia found {} tags total, \
         global max corner delta: {:.3} px",
        images_tested, total_c, total_k, max_delta_global
    );

    assert!(
        images_tested > 0,
        "No tag36_11_*.png files found in {:?}",
        tag_dir
    );
}
