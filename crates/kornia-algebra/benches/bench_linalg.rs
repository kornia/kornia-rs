use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::mat;
use kornia_algebra::{linalg::svd, Mat3F32, Mat3F64, Mat4F32, Vec3F32, Vec3F64};
use nalgebra::{Matrix3, Matrix4};
use rand::Rng;
use std::hint::black_box;


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

fn bench_mat3_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat3_mul");
    let mut rng = rand::rng();

    let k_m1 = Mat3F32::from_cols_array(&[rng.random(); 9]);
    let k_m2 = Mat3F32::from_cols_array(&[rng.random(); 9]);

    let n_m1 = Matrix3::new(
        k_m1.x_axis.x,
        k_m1.y_axis.x,
        k_m1.z_axis.x,
        k_m1.x_axis.y,
        k_m1.y_axis.y,
        k_m1.z_axis.y,
        k_m1.x_axis.z,
        k_m1.y_axis.z,
        k_m1.z_axis.z,
    );

    let n_m2 = Matrix3::new(
        k_m2.x_axis.x,
        k_m2.y_axis.x,
        k_m2.z_axis.x,
        k_m2.x_axis.y,
        k_m2.y_axis.y,
        k_m2.z_axis.y,
        k_m2.x_axis.z,
        k_m2.y_axis.z,
        k_m2.z_axis.z,
    );

    let f_data1 = k_m1.to_cols_array();
    let f_data2 = k_m2.to_cols_array();
    let f_m1 = faer::mat::from_column_major_slice(&f_data1, 3, 3);
    let f_m2 = faer::mat::from_column_major_slice(&f_data2, 3, 3);

    group.bench_function(BenchmarkId::new("kornia", ""), |b| {
        b.iter(|| black_box(k_m1) * black_box(k_m2))
    });

    group.bench_function(BenchmarkId::new("nalgebra", ""), |b| {
        b.iter(|| black_box(n_m1) * black_box(n_m2))
    });

    group.bench_function(BenchmarkId::new("faer", ""), |b| {
        b.iter(|| black_box(f_m1 * f_m2))
    });

    group.finish();
}

fn bench_mat4_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat4_mul");
    let mut rng = rand::rng();

    let k_m1 = Mat4F32::from_cols_array(&[rng.random(); 16]);
    let k_m2 = Mat4F32::from_cols_array(&[rng.random(); 16]);

    let n_m1 = Matrix4::from_iterator(k_m1.to_cols_array());
    let n_m2 = Matrix4::from_iterator(k_m2.to_cols_array());

    let f_data1 = k_m1.to_cols_array();
    let f_data2 = k_m2.to_cols_array();
    let f_m1 = faer::mat::from_column_major_slice(&f_data1, 4, 4);
    let f_m2 = faer::mat::from_column_major_slice(&f_data2, 4, 4);

    group.bench_function(BenchmarkId::new("kornia", ""), |b| {
        b.iter(|| black_box(k_m1) * black_box(k_m2))
    });

    group.bench_function(BenchmarkId::new("nalgebra", ""), |b| {
        b.iter(|| black_box(n_m1) * black_box(n_m2))
    });

    group.bench_function(BenchmarkId::new("faer", ""), |b| {
        b.iter(|| black_box(f_m1 * f_m2))
    });

    group.finish();
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

criterion_group!(benches, bench_svd3, bench_svd3_f64, bench_mat3_mul, bench_mat4_mul);
criterion_main!(benches);