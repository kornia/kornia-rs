//! Deterministic instruction counts for the SIFT CPU kernels.
//!
//! Criterion measures wall time, which on this host is contended: several of
//! this module's optimisations are 1-3% effects, and measuring them needed arms
//! interleaved back to back to survive concurrent builds. Callgrind counts
//! instructions instead — the same number every run, no warm-up, unaffected by
//! load — so "did this issue less work" is answered directly.
//!
//! `DescriptorScratch` is crate-internal, so the descriptor is exercised
//! through `pipeline` rather than in isolation.
//!
//! It does **not** replace the criterion benches. Callgrind does not model the
//! store-forwarding stalls that dominate the descriptor's histogram scatter, nor
//! cache behaviour, so a change can cut instructions and still be slower. Use
//! this to confirm the mechanism and `bench_features` to confirm the win.
//!
//! ```bash
//! cargo bench -p kornia-imgproc --bench bench_sift_instr
//! ```
//!
//! Needs `valgrind` and `cargo install iai-callgrind-runner --version 0.14.0`.

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use kornia_imgproc::features::{
    l2_sq, l2_sq_scalar, sift_detect_and_compute, FirstOctave, SiftConfig,
};

/// Deterministic pseudo-image; a fixed seed keeps the counts reproducible.
fn image(w: usize, h: usize) -> Vec<f32> {
    let mut s = 0x9E3779B9u32;
    (0..w * h)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 16) & 255) as f32
        })
        .collect()
}

// The NEON distance kernel against its scalar twin.
//
// A pure instruction-count comparison: the two compute the same value, so the
// ratio is the vectorisation factor with no timing noise in it.
#[library_benchmark]
fn l2_neon() {
    let a = image(128, 1);
    let b = image(128, 1);
    let mut acc = 0.0f32;
    for _ in 0..64 {
        acc += l2_sq(black_box(&a), black_box(&b));
    }
    black_box(acc);
}

#[library_benchmark]
fn l2_scalar() {
    let a = image(128, 1);
    let b = image(128, 1);
    let mut acc = 0.0f32;
    for _ in 0..64 {
        acc += l2_sq_scalar(black_box(&a), black_box(&b));
    }
    black_box(acc);
}

// A whole small frame, so a change can be checked against the total rather
// than one stage in isolation.
#[library_benchmark]
fn pipeline() {
    let (w, h) = (192usize, 144usize);
    let img = image(w, h);
    let cfg = SiftConfig::default();
    let r = sift_detect_and_compute(&img, w, h, &cfg, FirstOctave::Native, 3, false).unwrap();
    black_box(r.keypoints.len());
}

library_benchmark_group!(
    name = sift_instr;
    benchmarks = l2_neon, l2_scalar, pipeline
);

main!(library_benchmark_groups = sift_instr);
