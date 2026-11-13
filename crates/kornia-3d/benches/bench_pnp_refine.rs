use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::pnp::{refine_pose_lm, refine_pose_lm_6dof, LMRefineParams};
use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};

fn k_default() -> Mat3AF32 {
    Mat3AF32::from_cols(
        Vec3AF32::new(800.0, 0.0, 0.0),
        Vec3AF32::new(0.0, 800.0, 0.0),
        Vec3AF32::new(640.0, 480.0, 1.0),
    )
}

/// Points in front of an identity camera, with deterministic pixel noise of up to 1.5 px.
fn synthetic_problem(n: usize) -> (Vec<Vec3AF32>, Vec<Vec2F32>) {
    let mut world = Vec::with_capacity(n);
    let mut image = Vec::with_capacity(n);
    for i in 0..n {
        let x = ((i % 10) as f32 - 4.5) / 4.5;
        let y = (((i / 10) % 10) as f32 - 4.5) / 4.5;
        let z = 2.0 + 0.25 * (i % 7) as f32;
        let noise = ((i * 37 % 11) as f32 - 5.0) * 0.3;
        world.push(Vec3AF32::new(x, y, z));
        image.push(Vec2F32::new(
            800.0 * x / z + 640.0 + noise,
            800.0 * y / z + 480.0 - noise,
        ));
    }
    (world, image)
}

fn bench_pnp_refine(c: &mut Criterion) {
    let k = k_default();
    let params = LMRefineParams::default();
    let rotation = Mat3AF32::IDENTITY;
    let translation = Vec3AF32::new(0.05, -0.03, 0.1);

    let mut group = c.benchmark_group("pnp_refine");
    for &n in &[10, 100, 500] {
        let (world, image) = synthetic_problem(n);
        group.bench_with_input(BenchmarkId::new("refine_pose_lm", n), &n, |b, _| {
            b.iter(|| {
                std::hint::black_box(refine_pose_lm(
                    &world,
                    &image,
                    &k,
                    &rotation,
                    &translation,
                    None,
                    &params,
                ))
            });
        });
        group.bench_with_input(BenchmarkId::new("refine_pose_lm_6dof", n), &n, |b, _| {
            b.iter(|| {
                std::hint::black_box(refine_pose_lm_6dof(
                    &world,
                    &image,
                    &k,
                    &rotation,
                    &translation,
                    None,
                    &params,
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pnp_refine);
criterion_main!(benches);
