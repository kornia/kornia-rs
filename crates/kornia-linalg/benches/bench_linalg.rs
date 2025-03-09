use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::mat;
use glam::{Mat3, Vec3};
use kornia_linalg::linalg;

fn bench_svd3(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd3");
    let a1 = Mat3 {
        x_axis: Vec3::new(1.0, 0.0, 0.0),
        y_axis: Vec3::new(0.0, 2.0, 0.0),
        z_axis: Vec3::new(0.0, 0.0, 3.0),
    };

    let a2 = mat![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];

    group.bench_function(BenchmarkId::new("svd3", ""), |b| {
        b.iter(|| {
            linalg::svd3(&a1);
            black_box(());
        })
    });

    group.bench_function(BenchmarkId::new("svd3_faer", ""), |b| {
        b.iter(|| {
            a2.svd();
            black_box(());
        })
    });
}

criterion_group!(benches, bench_svd3);
criterion_main!(benches);
