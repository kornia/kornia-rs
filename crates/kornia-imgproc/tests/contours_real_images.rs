//! Integration tests for `find_contours` against the cv2 tutorial images
//! (`pic1`-`pic4`). These tests exercise topology that synthetic shapes
//! don't reach — many small disjoint outers, dense holes, multi-row
//! components. We check the contour count against
//! `cv2.findContours(img, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`.
//!
//! The PNG files live under `examples/data/` and are downloaded on demand
//! via `examples/fetch_fixtures.sh`. Tests gracefully skip if a fixture
//! is missing — synthetic-shape regression coverage runs unconditionally
//! via the `synth_ext_count` example.
//!
//! cv2 ground-truth values were captured from `cv2 4.13.0` on Linux,
//! using kornia's gray formula `(77*R + 150*G + 29*B) >> 8` to keep
//! binarisation identical between the two paths:
//!
//! ```text
//! pic1 (400x300):     1 contour  (large white background)
//! pic2 (400x300):     1 contour  (5722 dark holes inside)
//! pic3 (400x300):     1 contour
//! pic4 (400x300):   844 contours (many disjoint shapes)
//! ```

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::contours::{find_contours, ContourApproximationMode, RetrievalMode};
use std::path::PathBuf;

/// Loads a fixture PNG, binarises it at threshold 127, returns the image
/// or None if the file is missing.
fn load_binary(name: &str) -> Option<Image<u8, 1, CpuAllocator>> {
    let path: PathBuf = ["crates", "kornia-imgproc", "examples", "data", name]
        .iter()
        .collect();
    if !path.exists() {
        eprintln!("  skipping (fixture not present at {path:?})");
        return None;
    }
    let rgb = kornia_io::png::read_image_png_rgb8(&path).ok()?;
    let (w, h) = (rgb.width(), rgb.height());
    let mut gray = Image::<u8, 1, _>::from_size_val(
        ImageSize { width: w, height: h }, 0, CpuAllocator,
    ).ok()?;
    kornia_imgproc::color::gray_from_rgb_u8(&rgb, &mut gray).ok()?;
    let mut bw = Image::<u8, 1, _>::from_size_val(
        ImageSize { width: w, height: h }, 0, CpuAllocator,
    ).ok()?;
    kornia_imgproc::threshold::threshold_binary(&gray, &mut bw, 127, 1).ok()?;
    Some(bw)
}

fn ext_count(img: &Image<u8, 1, CpuAllocator>) -> usize {
    let r = find_contours(img, RetrievalMode::External, ContourApproximationMode::Simple)
        .expect("find_contours should succeed");
    r.contours.len()
}

#[test]
fn pic1_external_matches_cv2() {
    let Some(img) = load_binary("pic1.png") else { return; };
    // cv2.findContours(pic1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE) = 1
    assert_eq!(ext_count(&img), 1, "pic1 EXT count differs from cv2");
}

#[test]
fn pic2_external_matches_cv2() {
    let Some(img) = load_binary("pic2.png") else { return; };
    // pic2 has 1 outer (white background) + 5722 dark holes inside.
    // RETR_EXTERNAL discards the holes. cv2 returns 1.
    assert_eq!(ext_count(&img), 1, "pic2 EXT count differs from cv2");
}

#[test]
fn pic3_external_matches_cv2() {
    let Some(img) = load_binary("pic3.png") else { return; };
    assert_eq!(ext_count(&img), 1, "pic3 EXT count differs from cv2");
}

#[test]
fn pic4_external_matches_cv2() {
    let Some(img) = load_binary("pic4.png") else { return; };
    // Bit-exact parity vs cv2 when both consume the same binarisation
    // (kornia's `(77*R + 150*G + 29*B) >> 8`). The 881-contour result
    // sometimes quoted for pic4 uses cv2.IMREAD_GRAYSCALE's slightly
    // different gray formula — that's an input difference, not an
    // algorithm difference. See `examples/check_correctness.py`.
    assert_eq!(ext_count(&img), 844, "pic4 EXT count differs from cv2");
}

/// LIST mode (all contours, no hierarchy filtering) on pic1 should
/// return cv2 LIST count exactly.
#[test]
fn pic1_list_matches_cv2() {
    let Some(img) = load_binary("pic1.png") else { return; };
    let r = find_contours(&img, RetrievalMode::List, ContourApproximationMode::Simple)
        .expect("find_contours");
    assert_eq!(r.contours.len(), 17, "pic1 LIST count differs from cv2");
}
