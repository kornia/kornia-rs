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
            let _ = std::hint::black_box(a2.svd());
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
            let _ = std::hint::black_box(a2.svd());
        })
    });
}

fn bench_nearest_rotation(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_rotation");
    // A rotation with a little noise, as SQPnP and `Pose3d::orthonormalized` pass in.
    let m = Mat3F64::from_cols(
        Vec3F64::new(0.36, 0.48, -0.8),
        Vec3F64::new(-0.8, 0.61, 0.0),
        Vec3F64::new(0.48, 0.64, 0.6),
    );

    group.bench_function(BenchmarkId::new("nearest_rotation", ""), |b| {
        b.iter(|| std::hint::black_box(std::hint::black_box(m).nearest_rotation()))
    });

    group.bench_function(BenchmarkId::new("svd3_f64", ""), |b| {
        b.iter(|| std::hint::black_box(svd::svd3_f64(std::hint::black_box(&m))))
    });
}

criterion_group!(benches, bench_svd3, bench_svd3_f64, bench_nearest_rotation);
criterion_main!(benches);
