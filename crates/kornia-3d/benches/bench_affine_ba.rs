use criterion::{criterion_group, criterion_main, Criterion};
use kornia_3d::{
    ba_affine::{bundle_adjust_affine, AffineBaParams, AffineObservation},
    camera::AffineCamera,
};
use kornia_algebra::Vec3F64;
use std::hint::black_box;

fn bench(c: &mut Criterion) {
    let truth: Vec<_> = (0..10_000)
        .map(|i| {
            let t = i as f64;
            Vec3F64::new(
                (t * 1.7).sin() * 4.,
                (t * 0.9).cos() * 3.,
                (t * 0.4).sin() * 2.,
            )
        })
        .collect();
    let cameras: Vec<_> = (0..4)
        .map(|i| {
            let a = i as f64 * 1.5;
            AffineCamera {
                matrix: [
                    [a.cos() * 3., a.sin() * 3., 0., 100.],
                    [-a.sin() * 2., a.cos() * 2., -2., 200.],
                ],
            }
        })
        .collect();
    let obs: Vec<_> = cameras
        .iter()
        .enumerate()
        .flat_map(|(c, cam)| {
            truth
                .iter()
                .enumerate()
                .map(move |(p, x)| AffineObservation {
                    camera_index: c,
                    point_index: p,
                    pixel: cam.project(x),
                })
        })
        .collect();
    let fixed: Vec<_> = (0..truth.len()).map(|p| p < 4).collect();
    let mut points = truth.clone();
    for (i, x) in points.iter_mut().enumerate().skip(4) {
        x.x += 0.1 * (i as f64).sin();
        x.z -= 0.05;
    }
    let params = AffineBaParams {
        max_iterations: 20,
        ..Default::default()
    };
    c.bench_function("affine_schur_4_cameras_10000_points", |b| {
        b.iter(|| {
            black_box(
                bundle_adjust_affine(
                    black_box(&cameras),
                    black_box(&points),
                    &obs,
                    &fixed,
                    &params,
                )
                .unwrap(),
            )
        })
    });
}
criterion_group!(benches, bench);
criterion_main!(benches);
