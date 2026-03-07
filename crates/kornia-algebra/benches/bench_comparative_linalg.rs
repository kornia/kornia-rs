//! Comparative linear algebra benchmarks: kornia-algebra vs nalgebra.
//!
//! This benchmark suite compares the performance of kornia-algebra's linear algebra
//! operations against the industry-standard nalgebra library.
//!
//! Libraries compared:
//! - kornia-algebra: Built-in types (Mat3F32, Mat4F32, Vec3F32, etc.)
//! - nalgebra: Industry-standard Rust linear algebra library
//!
//! Use cases:
//! - Matrix multiplication (core operation)
//! - Matrix inversion (pose estimation, camera calibration)
//! - Vector operations (point transformations)

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use kornia_algebra::{Mat3F32, Mat4F32, Vec3F32, Vec4F32};
use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};


// Matrix Multiplication Benchmarks (3x3 and 4x4)

fn bench_kornia_mat3_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/mat3_mul");
    group.bench_function("kornia", |b| {
        let m1 = black_box(Mat3F32::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        ]));
        let m2 = black_box(Mat3F32::from_cols_array(&[
            9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ]));
        b.iter(|| m1 * m2)
    });
    group.finish();
}

fn bench_nalgebra_mat3_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/mat3_mul");
    group.bench_function("nalgebra", |b| {
        let m1 = black_box(Matrix3::new(
            1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0,
        ));
        let m2 = black_box(Matrix3::new(
            9.0, 6.0, 3.0, 8.0, 5.0, 2.0, 7.0, 4.0, 1.0,
        ));
        b.iter(|| m1 * m2)
    });
    group.finish();
}

fn bench_kornia_mat4_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/mat4_mul");
    group.bench_function("kornia", |b| {
        let m1 = black_box(Mat4F32::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ]));
        let m2 = black_box(Mat4F32::from_cols_array(&[
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
            1.0,
        ]));
        b.iter(|| m1 * m2)
    });
    group.finish();
}

fn bench_nalgebra_mat4_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/mat4_mul");
    group.bench_function("nalgebra", |b| {
        let m1 = black_box(Matrix4::new(
            1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0,
            16.0,
        ));
        let m2 = black_box(Matrix4::new(
            16.0, 12.0, 8.0, 4.0, 15.0, 11.0, 7.0, 3.0, 14.0, 10.0, 6.0, 2.0, 13.0, 9.0, 5.0,
            1.0,
        ));
        b.iter(|| m1 * m2)
    });
    group.finish();
}

// Matrix Inversion Benchmarks

fn bench_kornia_mat3_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/mat3_inverse");
    group.bench_function("kornia", |b| {
        let m = black_box(Mat3F32::from_cols_array(&[
            2.0, 0.5, 0.0, 1.0, 3.0, 0.2, 0.1, 0.0, 2.5,
        ]));
        b.iter(|| m.inverse())
    });
    group.finish();
}

fn bench_nalgebra_mat3_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/mat3_inverse");
    group.bench_function("nalgebra", |b| {
        let m = black_box(Matrix3::new(
            2.0, 1.0, 0.1, 0.5, 3.0, 0.0, 0.0, 0.2, 2.5,
        ));
        b.iter(|| m.try_inverse())
    });
    group.finish();
}

fn bench_kornia_mat4_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/mat4_inverse");
    group.bench_function("kornia", |b| {
        let m = black_box(Mat4F32::from_cols_array(&[
            2.0, 0.5, 0.0, 0.1, 1.0, 3.0, 0.2, 0.0, 0.1, 0.0, 2.5, 0.5, 0.0, 0.0, 0.0, 1.0,
        ]));
        b.iter(|| m.inverse())
    });
    group.finish();
}

fn bench_nalgebra_mat4_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/mat4_inverse");
    group.bench_function("nalgebra", |b| {
        let m = black_box(Matrix4::new(
            2.0, 1.0, 0.1, 0.0, 0.5, 3.0, 0.0, 0.0, 0.0, 0.2, 2.5, 0.0, 0.1, 0.0, 0.5, 1.0,
        ));
        b.iter(|| m.try_inverse())
    });
    group.finish();
}

// Vector Transformation Benchmarks (Matrix * Vector)

fn bench_kornia_vec3_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/vec3_transform");
    group.bench_function("kornia", |b| {
        let m = black_box(Mat3F32::from_cols_array(&[
            1.5, 0.2, 0.1, 0.3, 2.0, 0.0, 0.0, 0.5, 1.8,
        ]));
        let v = black_box(Vec3F32::new(1.0, 2.0, 3.0));
        b.iter(|| m * v)
    });
    group.finish();
}

fn bench_nalgebra_vec3_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/vec3_transform");
    group.bench_function("nalgebra", |b| {
        let m = black_box(Matrix3::new(
            1.5, 0.3, 0.0, 0.2, 2.0, 0.5, 0.1, 0.0, 1.8,
        ));
        let v = black_box(Vector3::new(1.0, 2.0, 3.0));
        b.iter(|| m * v)
    });
    group.finish();
}

fn bench_kornia_vec4_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/vec4_transform");
    group.bench_function("kornia", |b| {
        let m = black_box(Mat4F32::from_cols_array(&[
            1.5, 0.2, 0.0, 0.1, 0.3, 2.0, 0.5, 0.0, 0.0, 0.1, 1.8, 0.2, 0.0, 0.0, 0.0, 1.0,
        ]));
        let v = black_box(Vec4F32::new(1.0, 2.0, 3.0, 1.0));
        b.iter(|| m * v)
    });
    group.finish();
}

fn bench_nalgebra_vec4_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/vec4_transform");
    group.bench_function("nalgebra", |b| {
        let m = black_box(Matrix4::new(
            1.5, 0.3, 0.0, 0.0, 0.2, 2.0, 0.1, 0.0, 0.0, 0.5, 1.8, 0.0, 0.1, 0.0, 0.2, 1.0,
        ));
        let v = black_box(Vector4::new(1.0, 2.0, 3.0, 1.0));
        b.iter(|| m * v)
    });
    group.finish();
}


// Benchmark groups and main

criterion_group!(
    mat3_benchmarks,
    bench_kornia_mat3_multiplication,
    bench_nalgebra_mat3_multiplication,
    bench_kornia_mat3_inverse,
    bench_nalgebra_mat3_inverse,
    bench_kornia_vec3_transform,
    bench_nalgebra_vec3_transform,
);

criterion_group!(
    mat4_benchmarks,
    bench_kornia_mat4_multiplication,
    bench_nalgebra_mat4_multiplication,
    bench_kornia_mat4_inverse,
    bench_nalgebra_mat4_inverse,
    bench_kornia_vec4_transform,
    bench_nalgebra_vec4_transform,
);

criterion_main!(mat3_benchmarks, mat4_benchmarks);