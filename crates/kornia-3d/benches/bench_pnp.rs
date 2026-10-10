//! Runtime of the PnP solvers on synthetic correspondences.
//!
//! Random non-coplanar points in a 2 m box are seen from 6 m with a random rotation, and 1 px
//! Gaussian noise is added to the image points. EPnP is timed without and with LM refinement
//! (the configuration that gives an accurate pose), SQPnP with its default parameters.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::pnp::{solve_pnp, EPnPParams, LMRefineParams, PnPMethod};
use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
use std::hint::black_box;

/// Deterministic uniform/normal generator, so the benchmark needs no extra dependency.
struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.uniform().max(1e-12), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn scene(n: usize) -> (Vec<Vec3AF32>, Vec<Vec2F32>) {
    let mut rng = Lcg(42);
    // Rotation of 0.6 rad about a fixed oblique axis, translation 6 m forward.
    let (ax, ay, az) = (0.48f64, -0.6, 0.64);
    let angle = 0.6f64;
    let (s, c) = angle.sin_cos();
    let r = [
        [
            c + ax * ax * (1.0 - c),
            ax * ay * (1.0 - c) - az * s,
            ax * az * (1.0 - c) + ay * s,
        ],
        [
            ay * ax * (1.0 - c) + az * s,
            c + ay * ay * (1.0 - c),
            ay * az * (1.0 - c) - ax * s,
        ],
        [
            az * ax * (1.0 - c) - ay * s,
            az * ay * (1.0 - c) + ax * s,
            c + az * az * (1.0 - c),
        ],
    ];
    let t = [0.2, -0.1, 6.0];

    let mut world = Vec::with_capacity(n);
    let mut image = Vec::with_capacity(n);
    for _ in 0..n {
        let p = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        let pc: Vec<f64> = (0..3)
            .map(|i| r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + t[i])
            .collect();
        world.push(Vec3AF32::new(p[0] as f32, p[1] as f32, p[2] as f32));
        image.push(Vec2F32::new(
            (800.0 * pc[0] / pc[2] + 640.0 + rng.gauss()) as f32,
            (800.0 * pc[1] / pc[2] + 480.0 + rng.gauss()) as f32,
        ));
    }
    (world, image)
}

fn bench_pnp(c: &mut Criterion) {
    let k = Mat3AF32::from_cols_array(&[800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 640.0, 480.0, 1.0]);
    let mut group = c.benchmark_group("pnp");
    for n in [6usize, 20, 100, 500] {
        let (world, image) = scene(n);
        let epnp_lm = PnPMethod::EPnP(EPnPParams {
            refine_lm: Some(LMRefineParams::default()),
            ..Default::default()
        });
        for (name, method) in [
            ("epnp", PnPMethod::EPnPDefault),
            ("epnp_lm", epnp_lm),
            ("sqpnp", PnPMethod::SQPnPDefault),
        ] {
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
                b.iter(|| {
                    let pose = solve_pnp(
                        black_box(&world),
                        black_box(&image),
                        &k,
                        None,
                        method.clone(),
                    );
                    black_box(pose).ok()
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_pnp);
criterion_main!(benches);
