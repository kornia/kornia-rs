use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::histogram::{compute_histogram, equalize_hist};
use std::hint::black_box;

fn make_gray(width: usize, height: usize) -> Image<u8, 1> {
    let mut state = 0x9E37_79B9u32;
    let data: Vec<u8> = (0..width * height)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect();
    Image::new(ImageSize { width, height }, data).unwrap()
}

fn bench_histogram(c: &mut Criterion) {
    let mut hist_group = c.benchmark_group("compute_histogram");
    for &(width, height) in &[(640, 480), (1920, 1080)] {
        let img = make_gray(width, height);
        hist_group.throughput(Throughput::Elements((width * height) as u64));
        for &num_bins in &[64usize, 256] {
            let mut hist = vec![0usize; num_bins];
            hist_group.bench_with_input(
                BenchmarkId::new(format!("{width}x{height}"), num_bins),
                &num_bins,
                |b, &bins| {
                    b.iter(|| {
                        compute_histogram(black_box(&img), black_box(&mut hist), bins).unwrap();
                    });
                },
            );
        }
    }
    hist_group.finish();

    let mut eq_group = c.benchmark_group("equalize_hist");
    for &(width, height) in &[(640, 480), (1920, 1080)] {
        let img = make_gray(width, height);
        let mut dst = Image::<u8, 1>::from_size_val(img.size(), 0).unwrap();
        eq_group.throughput(Throughput::Elements((width * height) as u64));
        eq_group.bench_function(
            BenchmarkId::from_parameter(format!("{width}x{height}")),
            |b| {
                b.iter(|| {
                    equalize_hist(black_box(&img), black_box(&mut dst)).unwrap();
                });
            },
        );
    }
    eq_group.finish();
}

criterion_group!(benches, bench_histogram);
criterion_main!(benches);
