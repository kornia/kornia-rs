//! Bench: LSL run-based path vs Suzuki/Abe per-pixel path on the same fixtures.
//!
//! NOTE: LSL output is NOT yet bit-exact with Suzuki/Abe (different traversal
//! order; CCW post-process not yet applied). This bench measures RAW
//! algorithmic speed; correctness validation is a separate concern (will be
//! locked in by check_correctness.py once direction-reversal post-pass added).

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::contours::{find_contours, ContourApproximationMode, RetrievalMode};
use kornia_imgproc::contours_lsl::{find_external_contours_lsl, LslExecutor};
use std::time::Instant;

const REPS: usize = 20;
const WARMUP: usize = 5;

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

fn run_one(label: &str, w: usize, h: usize, data: Vec<u8>) {
    // LSL path with reusable executor (the realistic hot-loop case)
    let mut exec = LslExecutor::new();
    for _ in 0..WARMUP {
        let _ = exec.find_external_contours(&data, w, h);
    }
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        let _ = exec.find_external_contours(&data, w, h);
        samples.push(t.elapsed().as_secs_f64());
    }
    let (_, md_lsl, _) = stats(samples);
    let n_contours = exec.contour_count();

    // Suzuki/Abe path (current kornia, also reuses buffers via FindContoursExecutor)
    let img = Image::<u8, 1, _>::new(ImageSize { width: w, height: h }, data, CpuAllocator).unwrap();
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..WARMUP {
        let _ = find_contours(&img, RetrievalMode::External, ContourApproximationMode::None);
    }
    for _ in 0..REPS {
        let t = Instant::now();
        let _ = find_contours(&img, RetrievalMode::External, ContourApproximationMode::None);
        samples.push(t.elapsed().as_secs_f64());
    }
    let (_, md_sa, _) = stats(samples);

    let speedup = md_sa / md_lsl;
    println!(
        "{label:18} {w}x{h:4}  Suzuki/Abe={:>8.1}μs  LSL={:>8.1}μs  speedup={speedup:>5.2}×  ({n_contours} contours)",
        md_sa * 1e6,
        md_lsl * 1e6,
    );
}

fn main() {
    println!("# LSL run-based vs Suzuki/Abe per-pixel — Jetson Orin Nano");
    println!("# REPS={REPS}, WARMUP={WARMUP}, External mode\n");

    for (w, h) in [(128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)] {
        run_one("filled_square", w, h, make_filled_square(w, h));
    }
    println!();
    for (w, h) in [(128, 128), (256, 256), (512, 512), (1024, 1024)] {
        run_one("sparse_noise", w, h, make_noise(w, h, 0xC0FFEE));
    }
}
