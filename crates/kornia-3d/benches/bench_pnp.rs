//! PnP solver benchmarks. Each iteration solves `PNP_SCENES` (`pnp`) or `RANSAC_SCENES`
//! (`pnp_ransac`) scenes; a case where the solver fails on any scene is skipped, so no error path
//! is timed.
//!
//! `cargo bench -p kornia-3d --features test-utils --bench bench_pnp`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::pnp::{
    solve_pnp, solve_pnp_ransac, EPnPParams, LMRefineParams, PnPMethod, RansacParams,
};
use kornia_3d::test_utils::{k_default, Scene, SceneConfig, SceneError};
use rand::{rngs::StdRng, SeedableRng};

/// Scenes solved per `pnp` iteration.
const PNP_SCENES: usize = 20;
/// Scenes solved per `pnp_ransac` iteration.
const RANSAC_SCENES: usize = 5;

/// Generate scenes once, outside the timed loop, with the same fixtures as the accuracy tests.
fn generate_scenes(count: usize, config: &SceneConfig, outliers: usize) -> Vec<Scene> {
    let mut rng = StdRng::seed_from_u64(42);
    let scenes: Result<Vec<_>, SceneError> = (0..count)
        .map(|_| {
            let mut scene = Scene::generate(&mut rng, k_default(), config)?;
            scene.add_outliers(&mut rng, outliers)?;
            Ok(scene)
        })
        .collect();
    match scenes {
        Ok(scenes) => scenes,
        Err(error) => {
            eprintln!("Cannot generate benchmark scenes: {error}");
            std::process::exit(1);
        }
    }
}

fn bench_pnp(c: &mut Criterion) {
    let k = k_default();
    let epnp_lm = PnPMethod::EPnP(EPnPParams {
        refine_lm: Some(LMRefineParams::default()),
        ..Default::default()
    });
    let mut group = c.benchmark_group("pnp");
    for (n, planar) in [6usize, 20, 100, 500]
        .into_iter()
        .flat_map(|n| [(n, false), (n, true)])
    {
        let config = SceneConfig {
            planar,
            ..SceneConfig::cube(n, 6.0, 1.0)
        };
        let scenes = generate_scenes(PNP_SCENES, &config, 0);
        for (name, method) in [
            ("epnp", PnPMethod::EPnPDefault),
            ("epnp_lm", epnp_lm.clone()),
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
    let k = k_default();
    let scenes = generate_scenes(RANSAC_SCENES, &SceneConfig::cube(100, 6.0, 1.0), 100);
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
