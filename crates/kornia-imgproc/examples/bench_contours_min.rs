//! Standalone bench for kornia find_contours — same fixtures as bench_contours.rs,
//! emits CSV-friendly numbers for cross-comparison with cv2.findContours (Python).
//!
//! Usage: cargo run --release --example bench_contours_min -p kornia-imgproc

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::contours::{
    find_contours, ContourApproximationMode, FindContoursExecutor, RetrievalMode,
};
use std::time::Instant;

const SIZES: &[(usize, usize)] = &[(128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)];
const REPS: usize = 20;
const WARMUP: usize = 5;

/// Filled square with margin = size/8.
fn make_filled_square(w: usize, h: usize) -> Vec<u8> {
    let mw = w / 8;
    let mh = h / 8;
    let mut d = vec![0u8; w * h];
    for r in mh..(h - mh) {
        for c in mw..(w - mw) {
            d[r * w + c] = 1;
        }
    }
    d
}

/// Hollow square ring.
fn make_hollow_square(w: usize, h: usize) -> Vec<u8> {
    let ow = w / 8;
    let oh = h / 8;
    let iw = w / 4;
    let ih = h / 4;
    let mut d = vec![0u8; w * h];
    for r in oh..(h - oh) {
        for c in ow..(w - ow) {
            if r < ih || r >= (h - ih) || c < iw || c >= (w - iw) {
                d[r * w + c] = 1;
            }
        }
    }
    d
}

/// Sparse noise via the same LCG as bench_contours.rs (deterministic, comparable).
fn make_noise(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..w * h)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) & 1) as u8
        })
        .collect()
}

fn stats(mut s: Vec<f64>) -> (f64, f64, f64) {
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[0], s[s.len() / 2], s.iter().sum::<f64>() / s.len() as f64)
}

fn run_one(
    label: &str,
    w: usize,
    h: usize,
    data: Vec<u8>,
    retrieval: RetrievalMode,
    approx: ContourApproximationMode,
) {
    let img = Image::<u8, 1, _>::new(ImageSize { width: w, height: h }, data, CpuAllocator)
        .expect("kornia image");
    let mut exec = FindContoursExecutor::new();

    // --- find_contours (current API: per-contour Vec allocation) ---
    for _ in 0..WARMUP {
        let _ = exec.find_contours(&img, retrieval, approx);
    }
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        let _ = exec.find_contours(&img, retrieval, approx);
        samples.push(t.elapsed().as_secs_f64());
    }
    let (mn, md, mu) = stats(samples);
    let pix_per_s = (w * h) as f64 / md / 1e6;
    println!(
        "kornia,{label},{w}x{h},{:.1},{:.1},{:.1},{:.1}",
        mn * 1e6, md * 1e6, mu * 1e6, pix_per_s
    );

    // --- find_contours_view (zero-copy view into the executor's arena) ---
    for _ in 0..WARMUP {
        let _ = exec.find_contours_view(&img, retrieval, approx);
    }
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        let _ = exec.find_contours_view(&img, retrieval, approx);
        samples.push(t.elapsed().as_secs_f64());
    }
    let (mn, md, mu) = stats(samples);
    let pix_per_s = (w * h) as f64 / md / 1e6;
    println!(
        "kornia_view,{label},{w}x{h},{:.1},{:.1},{:.1},{:.1}",
        mn * 1e6, md * 1e6, mu * 1e6, pix_per_s
    );
}

/// Load + binarize an OpenCV-tutorial test image (pic1/pic3/pic4).
/// Threshold at 127 to match what OpenCV docs/tutorial do.
fn load_test_image(path: &str) -> Option<(usize, usize, Vec<u8>)> {
    let img = image::open(path).ok()?.to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut buf = vec![0u8; w * h];
    for (i, p) in img.into_raw().iter().enumerate() {
        buf[i] = (*p > 127) as u8;
    }
    Some((w, h, buf))
}

fn main() {
    println!("# CSV: impl,fixture,size,min_us,med_us,mean_us,Mpix_per_s_median");
    println!("impl,fixture,size,min_us,med_us,mean_us,Mpix_s");

    // === Real-world OpenCV tutorial images ===
    for (label, path) in [
        ("pic1_external_simple", "crates/kornia-imgproc/examples/data/pic1.png"),
        ("pic3_external_simple", "crates/kornia-imgproc/examples/data/pic3.png"),
        ("pic4_external_simple", "crates/kornia-imgproc/examples/data/pic4.png"),
    ] {
        match load_test_image(path) {
            Some((w, h, data)) => run_one(
                label, w, h, data,
                RetrievalMode::External,
                ContourApproximationMode::Simple,
            ),
            None => eprintln!("skipping {label}: failed to load {path}"),
        }
    }
    // pic2_list_simple: 5723 contours — stress test
    if let Some((w, h, data)) = load_test_image("crates/kornia-imgproc/examples/data/pic1.png") {
        run_one("pic1_list_simple", w, h, data, RetrievalMode::List, ContourApproximationMode::Simple);
    }
    if let Some((w, h, data)) = load_test_image("crates/kornia-imgproc/examples/data/pic4.png") {
        run_one("pic4_list_simple", w, h, data, RetrievalMode::List, ContourApproximationMode::Simple);
    }

    // === Synthetic fixtures (kept for stress testing) ===
    for &(w, h) in SIZES {
        run_one(
            "filled_square_external_simple",
            w, h, make_filled_square(w, h),
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        );
    }
    for &(w, h) in SIZES {
        run_one(
            "hollow_square_external_simple",
            w, h, make_hollow_square(w, h),
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        );
    }
    for &(w, h) in SIZES {
        run_one(
            "sparse_noise_external_simple",
            w, h, make_noise(w, h, 0xC0FFEE),
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        );
    }
    // None variants — the heavier path
    for &(w, h) in SIZES {
        run_one(
            "filled_square_external_none",
            w, h, make_filled_square(w, h),
            RetrievalMode::External,
            ContourApproximationMode::None,
        );
    }
}
