use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::mat;
use kornia_algebra::{linalg::svd, Mat3F32, Mat3F64, Vec3F32, Vec3F64};

fn bench_svd3(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd3");
    let a1 = Mat3F32::from_cols(
        Vec3F32::new(1.0, 0.0, 0.0),
        Vec3F32::new(0.0, 2.0, 0.0),
        Vec3F32::new(0.0, 0.0, 3.0),
    );

    let a2 = mat![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];

    group.bench_function(BenchmarkId::new("svd3", ""), |b| {
        b.iter(|| {
            svd::svd3_f32(&a1);
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

fn bench_svd3_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd3_f64");
    let a1 = Mat3F64::from_cols(
        Vec3F64::new(1.0, 0.0, 0.0),
        Vec3F64::new(0.0, 2.0, 0.0),
        Vec3F64::new(0.0, 0.0, 3.0),
    );

    let a2 = mat![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];

    group.bench_function(BenchmarkId::new("svd3", ""), |b| {
        b.iter(|| {
            svd::svd3_f64(&a1);
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

criterion_group!(benches, bench_svd3, bench_svd3_f64);
criterion_main!(benches);
