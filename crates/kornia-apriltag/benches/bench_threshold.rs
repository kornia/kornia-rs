//! Benchmark: adaptive_threshold – serial vs parallel vs OpenCV
//!
//! Run with:
//!   cargo bench -p kornia-apriltag --bench bench_threshold
//!
//! To include the OpenCV comparison enable the `opencv` feature:
//!   cargo bench -p kornia-apriltag --bench bench_threshold --features opencv
//!
//! Results are written to `target/criterion/`.  The cross-over point where
//! parallel becomes faster than serial can be read from the per-size groups.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kornia_apriltag::threshold::{
    adaptive_threshold_parallel, adaptive_threshold_serial, PARALLEL_PIXEL_THRESHOLD,
};
use kornia_apriltag::threshold::TileMinMax;
use kornia_apriltag::utils::Pixel;
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};

// ─── image sizes to sweep ─────────────────────────────────────────────────────

/// (label, width, height)
const SIZES: &[(&str, usize, usize)] = &[
    ("32x32", 32, 32),
    ("64x64", 64, 64),
    ("128x128", 128, 128),
    ("160x120", 160, 120),
    ("240x180", 240, 180),
    ("320x240", 320, 240),    // ← default PARALLEL_PIXEL_THRESHOLD
    ("480x360", 480, 360),
    ("640x480", 640, 480),
    ("1280x720", 1280, 720),
    ("1920x1080", 1920, 1080),
];

const TILE_SIZE: usize = 4;
const MIN_DIFF: u8 = 20;

// ─── helpers ─────────────────────────────────────────────────────────────────

fn make_src(width: usize, height: usize) -> Image<u8, 1, CpuAllocator> {
    // Checkerboard gradient so no tile is uniform (exercises the threshold path).
    let data: Vec<u8> = (0..width * height)
        .map(|i| (((i / width) ^ (i % width)) & 0xFF) as u8)
        .collect();
    Image::new(ImageSize { width, height }, data, CpuAllocator).unwrap()
}

fn make_dst(width: usize, height: usize) -> Image<Pixel, 1, CpuAllocator> {
    Image::from_size_val(ImageSize { width, height }, Pixel::Skip, CpuAllocator).unwrap()
}

// ─── benchmark groups ────────────────────────────────────────────────────────

fn bench_serial(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_threshold/serial");

    for &(label, w, h) in SIZES {
        let pixels = w * h;
        group.throughput(Throughput::Elements(pixels as u64));

        let src = make_src(w, h);

        group.bench_with_input(BenchmarkId::new("serial", label), &(w, h), |b, &(w, h)| {
            let mut dst = make_dst(w, h);
            let mut tile_mm = TileMinMax::new(src.size(), TILE_SIZE);
            b.iter(|| {
                adaptive_threshold_serial(
                    &src,
                    &mut dst,
                    &mut tile_mm,
                    MIN_DIFF,
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_threshold/parallel");

    for &(label, w, h) in SIZES {
        let pixels = w * h;
        group.throughput(Throughput::Elements(pixels as u64));

        let src = make_src(w, h);

        group.bench_with_input(BenchmarkId::new("parallel", label), &(w, h), |b, &(w, h)| {
            let mut dst = make_dst(w, h);
            let mut tile_mm = TileMinMax::new(src.size(), TILE_SIZE);
            b.iter(|| {
                adaptive_threshold_parallel(
                    &src,
                    &mut dst,
                    &mut tile_mm,
                    MIN_DIFF,
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

/// Side-by-side comparison per image size.
///
/// Look at each size group to find where parallel overtakes serial.
fn bench_crossover(c: &mut Criterion) {
    for &(label, w, h) in SIZES {
        let mut group = c.benchmark_group(format!("adaptive_threshold/crossover/{label}"));
        let pixels = w * h;
        group.throughput(Throughput::Elements(pixels as u64));

        let src = make_src(w, h);
        let crossover_note = if pixels < PARALLEL_PIXEL_THRESHOLD {
            "below-threshold"
        } else {
            "above-threshold"
        };

        // serial
        group.bench_function(format!("serial ({crossover_note})"), |b| {
            let mut dst = make_dst(w, h);
            let mut tile_mm = TileMinMax::new(src.size(), TILE_SIZE);
            b.iter(|| {
                adaptive_threshold_serial(&src, &mut dst, &mut tile_mm, MIN_DIFF).unwrap()
            });
        });

        // parallel
        group.bench_function(format!("parallel ({crossover_note})"), |b| {
            let mut dst = make_dst(w, h);
            let mut tile_mm = TileMinMax::new(src.size(), TILE_SIZE);
            b.iter(|| {
                adaptive_threshold_parallel(&src, &mut dst, &mut tile_mm, MIN_DIFF).unwrap()
            });
        });

        group.finish();
    }
}

// ─── optional OpenCV comparison ───────────────────────────────────────────────
//
// Enable with:  cargo bench ... --features opencv
//
// OpenCV's `adaptive_threshold` uses a Gaussian or mean blur over a block
// neighbourhood (ADAPTIVE_THRESH_MEAN_C / ADAPTIVE_THRESH_GAUSSIAN_C) which
// differs slightly from kornia's tile-min-max approach, but the throughput
// numbers are still a meaningful reference point.
//
// If the feature is absent these functions compile to nothing.

#[cfg(feature = "opencv")]
mod opencv_bench {
    use super::*;
    use opencv::{
        core::{Mat, Size},
        imgproc,
        prelude::*,
    };

    /// Convert our Image to an OpenCV Mat (grayscale u8, single channel).
    fn to_cv_mat(img: &Image<u8, 1, CpuAllocator>) -> Mat {
        let (h, w) = (img.height() as i32, img.width() as i32);
        let mut mat =
            Mat::new_rows_cols_with_default(h, w, opencv::core::CV_8UC1, 0.into()).unwrap();
        let flat: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(mat.data_mut().cast(), (h * w) as usize) };
        flat.copy_from_slice(img.as_slice());
        mat
    }

    pub fn bench_opencv(c: &mut Criterion) {
        let mut group = c.benchmark_group("adaptive_threshold/opencv");

        // OpenCV block_size must be odd; use the same neighbourhood size as our tile.
        let block_size = (TILE_SIZE * 2 + 1) as i32; // e.g. tile_size=4 → block_size=9
        let c_constant = 2.0f64;

        for &(label, w, h) in super::SIZES {
            let pixels = w * h;
            group.throughput(Throughput::Elements(pixels as u64));

            let src = super::make_src(w, h);
            let cv_src = to_cv_mat(&src);

            group.bench_with_input(
                criterion::BenchmarkId::new("opencv", label),
                &(w, h),
                |b, _| {
                    let mut cv_dst = Mat::default();
                    b.iter(|| {
                        imgproc::adaptive_threshold(
                            &cv_src,
                            &mut cv_dst,
                            255.0,
                            imgproc::ADAPTIVE_THRESH_MEAN_C,
                            imgproc::THRESH_BINARY,
                            block_size,
                            c_constant,
                        )
                        .unwrap()
                    });
                },
            );
        }

        group.finish();
    }
}

// ─── criterion wiring ─────────────────────────────────────────────────────────

#[cfg(not(feature = "opencv"))]
criterion_group!(benches, bench_serial, bench_parallel, bench_crossover);

#[cfg(feature = "opencv")]
criterion_group!(
    benches,
    bench_serial,
    bench_parallel,
    bench_crossover,
    opencv_bench::bench_opencv
);

criterion_main!(benches);
