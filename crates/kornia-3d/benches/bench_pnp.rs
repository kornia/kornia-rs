//! PnP solver benchmarks. Each `pnp` iteration solves 20 scenes and each `pnp_ransac` iteration 5;
//! a case where the solver fails on any scene is skipped, so no error path is timed.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::pnp::{
    solve_pnp, solve_pnp_ransac, EPnPParams, LMRefineParams, PnPMethod, RansacParams,
};
use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32, SO3F32};
use rand::{rngs::StdRng, RngExt, SeedableRng};

const FOCAL: f32 = 800.0;
const CX: f32 = 640.0;
const CY: f32 = 480.0;

/// The 2D-3D correspondences of one scene.
struct Correspondences {
    world: Vec<Vec3AF32>,
    image: Vec<Vec2F32>,
}

/// `count` scenes of `n` points with up to 1 px noise, each followed by `outliers` wrong
/// correspondences. The seed is fixed, so every run times the same scenes.
fn generate_scenes(count: usize, n: usize, planar: bool, outliers: usize) -> Vec<Correspondences> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..count)
        .map(|_| generate_scene(&mut rng, n, planar, outliers))
        .collect()
}

/// Points in a 2 m cube (or, when `planar`, a 2 m square) seen from 6 m.
fn generate_scene(rng: &mut StdRng, n: usize, planar: bool, outliers: usize) -> Correspondences {
    let rotation = SO3F32::exp(Vec3AF32::new(0.48, -0.6, 0.64) * 0.6).matrix();
    let translation = Vec3AF32::new(0.2, -0.1, 6.0);

    let mut world = Vec::with_capacity(n + outliers);
    let mut image = Vec::with_capacity(n + outliers);
    for _ in 0..n {
        let z = if planar {
            0.0
        } else {
            rng.random_range(-1.0..1.0)
        };
        let p = Vec3AF32::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), z);
        let pc = rotation * p + translation;
        world.push(p);
        image.push(Vec2F32::new(
            FOCAL * pc.x / pc.z + CX + rng.random_range(-1.0..1.0),
            FOCAL * pc.y / pc.z + CY + rng.random_range(-1.0..1.0),
        ));
    }
    for _ in 0..outliers {
        world.push(Vec3AF32::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        ));
        image.push(Vec2F32::new(
            rng.random_range(0.0..2.0 * CX),
            rng.random_range(0.0..2.0 * CY),
        ));
    }
    Correspondences { world, image }
}

fn camera_matrix() -> Mat3AF32 {
    Mat3AF32::from_cols_array(&[FOCAL, 0.0, 0.0, 0.0, FOCAL, 0.0, CX, CY, 1.0])
}

fn bench_pnp(c: &mut Criterion) {
    let k = camera_matrix();
    let mut group = c.benchmark_group("pnp");
    for (n, planar) in [6usize, 20, 100, 500]
        .into_iter()
        .flat_map(|n| [(n, false), (n, true)])
    {
        let scenes = generate_scenes(20, n, planar, 0);
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
            if let Some(e) = scenes
                .iter()
                .find_map(|s| solve_pnp(&s.world, &s.image, &k, None, method.clone()).err())
            {
                eprintln!("skipping pnp/{name}/{n}: the solver fails on a scene ({e})");
                continue;
            }
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
                b.iter(|| {
                    for s in &scenes {
                        let _ = std::hint::black_box(solve_pnp(
                            std::hint::black_box(&s.world),
                            std::hint::black_box(&s.image),
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
    let k = camera_matrix();
    let scenes = generate_scenes(5, 100, false, 100);
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
        if let Some(e) = scenes.iter().find_map(|s| {
            solve_pnp_ransac(&s.world, &s.image, &k, None, kernel.clone(), &params).err()
        }) {
            eprintln!("skipping pnp_ransac/{name}: RANSAC fails on a scene ({e})");
            continue;
        }
        group.bench_function(name, |b| {
            b.iter(|| {
                for s in &scenes {
                    let _ = std::hint::black_box(solve_pnp_ransac(
                        std::hint::black_box(&s.world),
                        std::hint::black_box(&s.image),
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
