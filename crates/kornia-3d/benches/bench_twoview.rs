use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::pose::{
    decompose_essential, enforce_essential_constraints, essential_from_fundamental,
    fundamental_8point, ransac_fundamental, ransac_homography, sampson_distance, RansacParams,
};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

/// Generate synthetic point correspondences satisfying a fundamental matrix.
fn generate_fundamental_data(n: usize) -> (Vec<Vec2F64>, Vec<Vec2F64>, Mat3F64) {
    let f_true = Mat3F64::from_cols(
        Vec3F64::new(0.0, -0.001, 0.01),
        Vec3F64::new(0.0015, 0.0, -0.02),
        Vec3F64::new(-0.01, 0.02, 1.0),
    );
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    for i in 0..n {
        let xi = i as f64 * 1.2 - (n as f64 / 2.0);
        let yi = i as f64 * -0.8 + (n as f64 / 4.0);
        let x = Vec3F64::new(xi, yi, 1.0);
        let l = f_true * x;
        let xp = if l.x.abs() > 1e-12 { -l.z / l.x } else { 0.0 };
        x1.push(Vec2F64::new(xi, yi));
        x2.push(Vec2F64::new(xp, 0.0));
    }
    (x1, x2, f_true)
}

/// Generate synthetic point correspondences satisfying a homography.
fn generate_homography_data(n: usize) -> (Vec<Vec2F64>, Vec<Vec2F64>) {
    let h_true = Mat3F64::from_cols(
        Vec3F64::new(1.2, 0.0, 0.001),
        Vec3F64::new(0.1, 0.9, 0.002),
        Vec3F64::new(5.0, -3.0, 1.0),
    );
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let side = (n as f64).sqrt().ceil() as usize;
    for i in 0..n {
        let xi = (i % side) as f64 * 2.0 - (side as f64);
        let yi = (i / side) as f64 * 1.5 - (side as f64 / 2.0);
        let p = Vec3F64::new(xi, yi, 1.0);
        let hp = h_true * p;
        x1.push(Vec2F64::new(xi, yi));
        x2.push(Vec2F64::new(hp.x / hp.z, hp.y / hp.z));
    }
    (x1, x2)
}

fn bench_fundamental_8point(c: &mut Criterion) {
    let mut group = c.benchmark_group("fundamental_8point");
    for &n in &[8, 50, 200] {
        let (x1, x2, _) = generate_fundamental_data(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let _ = std::hint::black_box(fundamental_8point(&x1, &x2));
            });
        });
    }
    group.finish();
}

fn bench_sampson_distance(c: &mut Criterion) {
    let (x1, x2, f_true) = generate_fundamental_data(1);
    c.bench_function("sampson_distance", |b| {
        b.iter(|| {
            std::hint::black_box(sampson_distance(&f_true, &x1[0], &x2[0]));
        });
    });
}

fn bench_decompose_essential(c: &mut Criterion) {
    let e = Mat3F64::from_cols(
        Vec3F64::new(0.0, -1.0, 0.0),
        Vec3F64::new(1.0, 0.0, 0.0),
        Vec3F64::new(0.0, 0.0, 0.0),
    );
    let e = enforce_essential_constraints(&e);
    c.bench_function("decompose_essential", |b| {
        b.iter(|| {
            std::hint::black_box(decompose_essential(&e));
        });
    });
}

fn bench_essential_pipeline(c: &mut Criterion) {
    let (x1, x2, _) = generate_fundamental_data(50);
    let f = fundamental_8point(&x1, &x2).unwrap();
    let k = Mat3F64::from_cols(
        Vec3F64::new(500.0, 0.0, 0.0),
        Vec3F64::new(0.0, 500.0, 0.0),
        Vec3F64::new(320.0, 240.0, 1.0),
    );
    c.bench_function("essential_pipeline", |b| {
        b.iter(|| {
            let e = essential_from_fundamental(&f, &k, &k);
            let e = enforce_essential_constraints(&e);
            std::hint::black_box(decompose_essential(&e));
        });
    });
}

fn bench_ransac_fundamental(c: &mut Criterion) {
    let mut group = c.benchmark_group("ransac_fundamental");
    for &n in &[50, 200, 500] {
        let (x1, x2, _) = generate_fundamental_data(n);
        let params = RansacParams {
            max_iterations: 500,
            threshold: 1.0,
            min_inliers: 10,
            random_seed: Some(42),
        };
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let _ = std::hint::black_box(ransac_fundamental(&x1, &x2, &params));
            });
        });
    }
    group.finish();
}

fn bench_ransac_homography(c: &mut Criterion) {
    let mut group = c.benchmark_group("ransac_homography");
    for &n in &[25, 100, 500] {
        let (x1, x2) = generate_homography_data(n);
        let params = RansacParams {
            max_iterations: 500,
            threshold: 1e-6,
            min_inliers: 10,
            random_seed: Some(42),
        };
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let _ = std::hint::black_box(ransac_homography(&x1, &x2, &params));
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_fundamental_8point,
    bench_sampson_distance,
    bench_decompose_essential,
    bench_essential_pipeline,
    bench_ransac_fundamental,
    bench_ransac_homography,
);
criterion_main!(benches);
