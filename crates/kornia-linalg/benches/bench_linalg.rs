use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::mat;
use glam::{Mat3, Vec3};
use kornia_algebra::Mat3F32;
use kornia_linalg::svd;

fn bench_svd3(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd3");

    // 1. Setup Data (glam type)
    let a1_glam = Mat3 {
        x_axis: Vec3::new(1.0, 0.0, 0.0),
        y_axis: Vec3::new(0.0, 2.0, 0.0),
        z_axis: Vec3::new(0.0, 0.0, 3.0),
    };

    // 2. Convert to kornia type for svd3 (do this once, outside the loop)
    let a1_kornia = Mat3F32::from(a1_glam);

    let a2 = mat![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];

    group.bench_function(BenchmarkId::new("svd3", ""), |b| {
        b.iter(|| {
            // Pass the correct kornia wrapper type
            svd::svd3(&a1_kornia);
            std::hint::black_box(());
        })
    });

    group.bench_function(BenchmarkId::new("svd3_faer", ""), |b| {
        b.iter(|| {
            a2.svd();
            std::hint::black_box(());
        })
    });
}

criterion_group!(benches, bench_svd3);
criterion_main!(benches);
