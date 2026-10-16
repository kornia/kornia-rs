//! Runtime of the PnP solvers on synthetic correspondences.
//!
//! Random points in a 2 m box (or, for `planar`, a 2 m square such as a marker or a chessboard)
//! are seen from 6 m with a fixed rotation, and 1 px Gaussian noise is added to the image points.
//! EPnP is timed without and with LM refinement (the configuration that gives an accurate pose),
//! SQPnP with its default parameters. The `pnp_ransac` group times `solve_pnp_ransac` with each
//! kernel on 100 such points plus 100 random outliers.
//!
//! Each `pnp` iteration solves 20 random scenes and each `pnp_ransac` iteration 5, so a reported
//! time is for the whole batch (divide by 20 or 5 for one solve) and does not depend on one lucky
//! scene. Every scene is solved once before timing; a case where the solver fails on any scene is
//! reported and skipped, so no error path is ever timed.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::pnp::{
    solve_pnp, solve_pnp_ransac, EPnPParams, LMRefineParams, PnPMethod, RansacParams,
};
use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32, Vec3F64, SO3F64};
use rand::{rngs::StdRng, RngExt, SeedableRng};
use std::hint::black_box;

/// A standard normal sample (Box–Muller); `rand` alone has no normal distribution.
fn gauss(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// One scene's correspondences.
type Scene = (Vec<Vec3AF32>, Vec<Vec2F32>);

/// `count` scenes of `n` points seen with 1 px noise, each followed by `outliers` random
/// correspondences. The seed is fixed, so every run times the same scenes.
fn scenes(count: usize, n: usize, planar: bool, outliers: usize) -> Vec<Scene> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..count)
        .map(|_| scene(&mut rng, n, planar, outliers))
        .collect()
}

fn scene(rng: &mut StdRng, n: usize, planar: bool, outliers: usize) -> Scene {
    // Rotation of 0.6 rad about a fixed oblique (unit) axis, translation 6 m forward.
    let rotation = SO3F64::exp(Vec3F64::new(0.48, -0.6, 0.64) * 0.6).matrix();
    let translation = Vec3F64::new(0.2, -0.1, 6.0);

    let mut world = Vec::with_capacity(n + outliers);
    let mut image = Vec::with_capacity(n + outliers);
    for _ in 0..n {
        let z = if planar {
            0.0
        } else {
            rng.random_range(-1.0..1.0)
        };
        let p = Vec3F64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), z);
        let pc = rotation * p + translation;
        world.push(Vec3AF32::new(p.x as f32, p.y as f32, p.z as f32));
        image.push(Vec2F32::new(
            (800.0 * pc.x / pc.z + 640.0 + gauss(rng)) as f32,
            (800.0 * pc.y / pc.z + 480.0 + gauss(rng)) as f32,
        ));
    }
    for _ in 0..outliers {
        world.push(Vec3AF32::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        ));
        image.push(Vec2F32::new(
            rng.random_range(0.0..1280.0),
            rng.random_range(0.0..960.0),
        ));
    }
    (world, image)
}

fn k_matrix() -> Mat3AF32 {
    Mat3AF32::from_cols_array(&[800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 640.0, 480.0, 1.0])
}

fn bench_pnp(c: &mut Criterion) {
    let k = k_matrix();
    let mut group = c.benchmark_group("pnp");
    for (n, planar) in [6usize, 20, 100, 500]
        .into_iter()
        .flat_map(|n| [(n, false), (n, true)])
    {
        let batch = scenes(20, n, planar, 0);
        let epnp_lm = PnPMethod::EPnP(EPnPParams {
            refine_lm: Some(LMRefineParams::default()),
            ..Default::default()
        });
        for (name, method) in [
            ("epnp", PnPMethod::EPnPDefault),
            ("epnp_lm", epnp_lm),
            ("sqpnp", PnPMethod::SQPnPDefault),
        ] {
            let name = if planar {
                format!("{name}_planar")
            } else {
                name.to_string()
            };
            if let Some(e) = batch
                .iter()
                .find_map(|(w, i)| solve_pnp(w, i, &k, None, method.clone()).err())
            {
                eprintln!("skipping pnp/{name}/{n}: the solver fails on a scene ({e})");
                continue;
            }
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
                b.iter(|| {
                    for (world, image) in &batch {
                        let _ = black_box(solve_pnp(
                            black_box(world),
                            black_box(image),
                            &k,
                            None,
                            method.clone(),
                        ));
                    }
                })
            });
        }
    }
    group.finish();
}

fn bench_pnp_ransac(c: &mut Criterion) {
    let k = k_matrix();
    let batch = scenes(5, 100, false, 100);
    let params = RansacParams {
        reproj_threshold_px: 3.0,
        random_seed: Some(1),
        ..Default::default()
    };
    let mut group = c.benchmark_group("pnp_ransac");
    for (name, kernel) in [
        ("epnp", PnPMethod::EPnPDefault),
        ("ap3p", PnPMethod::AP3PDefault),
        ("sqpnp", PnPMethod::SQPnPDefault),
    ] {
        if let Some(e) = batch
            .iter()
            .find_map(|(w, i)| solve_pnp_ransac(w, i, &k, None, kernel.clone(), &params).err())
        {
            eprintln!("skipping pnp_ransac/{name}: RANSAC fails on a scene ({e})");
            continue;
        }
        group.bench_function(name, |b| {
            b.iter(|| {
                for (world, image) in &batch {
                    let _ = black_box(solve_pnp_ransac(
                        black_box(world),
                        black_box(image),
                        &k,
                        None,
                        kernel.clone(),
                        &params,
                    ));
                }
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pnp, bench_pnp_ransac);
criterion_main!(benches);
