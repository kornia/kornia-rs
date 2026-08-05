use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::Image;
use kornia_imgproc::metrics;

/// Deterministic pseudo-random f32 in [0, 1).
///
/// The metrics take *two* images. Benchmarking them with one all-zero buffer
/// compared against itself makes every difference exactly zero, which pins
/// `huber` to one side of its `|diff| <= delta` branch and never exercises the
/// other. Two independently seeded buffers keep both arms live.
fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32
        })
        .collect()
}

fn bench_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics");

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        group.throughput(criterion::Throughput::Elements(
            (*width * *height * 3) as u64,
        ));

        let parameter_string = format!("{width}x{height}");
        let image_size = [*width, *height].into();
        let len = width * height * 3;

        let image1 = Image::<f32, 3>::new(image_size, pseudo_random(len, 0x1234_5678)).unwrap();
        let image2 = Image::<f32, 3>::new(image_size, pseudo_random(len, 0x9E37_79B9)).unwrap();
        let pair = (image1, image2);

        group.bench_with_input(
            BenchmarkId::new("mse", &parameter_string),
            &pair,
            |b, (x, y)| b.iter(|| metrics::mse(std::hint::black_box(x), std::hint::black_box(y))),
        );

        group.bench_with_input(
            BenchmarkId::new("l1_loss", &parameter_string),
            &pair,
            |b, (x, y)| {
                b.iter(|| metrics::l1_loss(std::hint::black_box(x), std::hint::black_box(y)))
            },
        );

        // delta = 0.25 straddles the branch for uniformly distributed inputs:
        // most |diff| land above it, a meaningful minority below.
        group.bench_with_input(
            BenchmarkId::new("huber", &parameter_string),
            &pair,
            |b, (x, y)| {
                b.iter(|| {
                    metrics::huber(
                        std::hint::black_box(x),
                        std::hint::black_box(y),
                        std::hint::black_box(0.25),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_metrics);
criterion_main!(benches);
