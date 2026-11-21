//! `median_blur` for `u8`, `u16` and `f32`, ksize 3 and 5, CPU.
//!
//! `bench_median_baselines.py` times `cv2.medianBlur` on the same images.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::filter::{median_blur, MedianBlur};

const SIZES: [(usize, usize); 2] = [(640, 480), (1280, 720)];

/// Multiplicative hash of the pixel index: same data as the baseline
/// script. The networks are branch-free, so timing does not depend on it.
fn hash(i: usize) -> u64 {
    (i as u64).wrapping_mul(2_654_435_761)
}

fn bench_one<T: MedianBlur + Default>(
    c: &mut Criterion,
    name: &str,
    size: ImageSize,
    data: Vec<T>,
) {
    let mut group = c.benchmark_group("Median");
    let src = Image::<T, 1>::new(size, data).unwrap();
    let mut dst = Image::<T, 1>::from_size_val(size, T::default()).unwrap();
    let id = format!("{}x{}", size.width, size.height);
    for ksize in [3usize, 5] {
        group.bench_function(BenchmarkId::new(format!("{name}_k{ksize}"), &id), |b| {
            b.iter(|| {
                std::hint::black_box(median_blur(
                    std::hint::black_box(&src),
                    std::hint::black_box(&mut dst),
                    std::hint::black_box(ksize),
                ))
            })
        });
    }
    group.finish();
}

fn bench_median(c: &mut Criterion) {
    for (w, h) in SIZES {
        let size = ImageSize {
            width: w,
            height: h,
        };
        let n = w * h;
        bench_one(
            c,
            "u8",
            size,
            (0..n).map(|i| (hash(i) >> 24) as u8).collect(),
        );
        bench_one(
            c,
            "u16",
            size,
            (0..n).map(|i| (hash(i) >> 16) as u16).collect(),
        );
        bench_one(
            c,
            "f32",
            size,
            (0..n)
                .map(|i| f32::from((hash(i) >> 16) as u16) * 1e-3)
                .collect(),
        );
    }
}

criterion_group!(benches, bench_median);
criterion_main!(benches);
