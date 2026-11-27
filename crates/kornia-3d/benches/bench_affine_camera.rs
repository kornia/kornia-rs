use criterion::{criterion_group, criterion_main, Criterion};
use kornia_3d::camera::AffineCamera;
use kornia_algebra::Vec3F64;
use std::hint::black_box;

fn bench_affine_camera(c: &mut Criterion) {
    let camera = AffineCamera {
        matrix: [[1.2, -0.3, 0.4, 12.], [0.2, 2., -0.7, -3.]],
    };
    let point = Vec3F64::new(2., -4., 1.);
    c.bench_function("affine_camera/project", |b| {
        b.iter(|| black_box(camera).project(black_box(&point)))
    });
    let matrix =
        nalgebra::SMatrix::<f64, 2, 4>::from_row_slice(&[1.2, -0.3, 0.4, 12., 0.2, 2., -0.7, -3.]);
    let homogeneous = nalgebra::SVector::<f64, 4>::new(2., -4., 1., 1.);
    c.bench_function("affine_camera/nalgebra_projection_reference", |b| {
        b.iter(|| black_box(matrix) * black_box(homogeneous))
    });
    c.bench_function("affine_camera/jacobians", |b| {
        b.iter(|| black_box(camera).jacobians(black_box(&point)))
    });
}

criterion_group!(benches, bench_affine_camera);
criterion_main!(benches);
