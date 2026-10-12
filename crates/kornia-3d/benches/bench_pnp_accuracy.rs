//! Accuracy and robustness of the PnP solvers on synthetic correspondences.
//!
//! Prints Markdown tables of the median rotation error, the median relative translation error
//! and the failure rate (rotation error above 5° or no pose returned) for EPnP, EPnP + LM and
//! SQPnP: across correspondence counts, across pixel-noise levels, and in harder configurations
//! (planar points, a small object far away, the 4-point minimum). Each cell is 200 random scenes
//! with a fixed seed, so the output is reproducible.
//!
//! Run with `cargo bench -p kornia-3d --bench bench_pnp_accuracy`.

use kornia_3d::pnp::{solve_pnp, EPnPParams, LMRefineParams, PnPMethod};
use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};

const TRIALS: usize = 200;
const FAILURE_DEG: f64 = 5.0;
const FOCAL: f64 = 800.0;
const CENTER: (f64, f64) = (640.0, 480.0);

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

type Mat3 = [[f64; 3]; 3];

/// Rotation of a uniformly random angle in [0, π) about a random axis (Rodrigues).
fn random_rotation(rng: &mut Lcg) -> Mat3 {
    let (x, y, z) = (rng.gauss(), rng.gauss(), rng.gauss());
    let norm = (x * x + y * y + z * z).sqrt().max(1e-12);
    let (ax, ay, az) = (x / norm, y / norm, z / norm);
    let (s, c) = rng.range(0.0, std::f64::consts::PI).sin_cos();
    let t = 1.0 - c;
    [
        [c + ax * ax * t, ax * ay * t - az * s, ax * az * t + ay * s],
        [ay * ax * t + az * s, c + ay * ay * t, ay * az * t - ax * s],
        [az * ax * t - ay * s, az * ay * t + ax * s, c + az * az * t],
    ]
}

fn apply(r: &Mat3, p: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2])
}

/// The shape of one batch of random scenes.
struct SceneKind {
    n: usize,
    noise_px: f64,
    /// Half-size of the cube (or square, when planar) holding the points, in metres.
    size: f64,
    depth: f64,
    planar: bool,
}

struct Scene {
    world: Vec<Vec3AF32>,
    image: Vec<Vec2F32>,
    rotation: Mat3,
    translation: [f64; 3],
}

fn make_scene(rng: &mut Lcg, kind: &SceneKind) -> Scene {
    let rotation = random_rotation(rng);
    let translation = [
        rng.range(-0.1, 0.1) * kind.depth,
        rng.range(-0.1, 0.1) * kind.depth,
        kind.depth,
    ];
    let mut world = Vec::with_capacity(kind.n);
    let mut image = Vec::with_capacity(kind.n);
    while world.len() < kind.n {
        let z = if kind.planar {
            0.0
        } else {
            rng.range(-kind.size, kind.size)
        };
        let p = [
            rng.range(-kind.size, kind.size),
            rng.range(-kind.size, kind.size),
            z,
        ];
        let rp = apply(&rotation, p);
        let pc = [
            rp[0] + translation[0],
            rp[1] + translation[1],
            rp[2] + translation[2],
        ];
        // Keep only points well in front of the camera, as a real detector would.
        if pc[2] < 0.1 * kind.depth {
            continue;
        }
        world.push(Vec3AF32::new(p[0] as f32, p[1] as f32, p[2] as f32));
        image.push(Vec2F32::new(
            (FOCAL * pc[0] / pc[2] + CENTER.0 + kind.noise_px * rng.gauss()) as f32,
            (FOCAL * pc[1] / pc[2] + CENTER.1 + kind.noise_px * rng.gauss()) as f32,
        ));
    }
    Scene {
        world,
        image,
        rotation,
        translation,
    }
}

/// Rotation error in degrees, `2·asin(‖R₁ − R₂‖_F / 2√2)` (accurate for small angles).
fn rotation_error_deg(estimate: &Mat3AF32, truth: &Mat3) -> f64 {
    let c = estimate.0.to_cols_array();
    let mut sq = 0.0;
    for (col, chunk) in c.chunks(3).enumerate() {
        for (row, v) in chunk.iter().enumerate() {
            sq += (*v as f64 - truth[row][col]).powi(2);
        }
    }
    let chord = sq.sqrt() / (2.0 * std::f64::consts::SQRT_2);
    (2.0 * chord.clamp(0.0, 1.0).asin()).to_degrees()
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

/// Median rotation error (°), median relative translation error (%) and failure rate (%).
fn evaluate(method: &PnPMethod, kind: &SceneKind, seed: u64) -> (f64, f64, f64) {
    let k = Mat3AF32::from_cols_array(&[
        FOCAL as f32,
        0.0,
        0.0,
        0.0,
        FOCAL as f32,
        0.0,
        CENTER.0 as f32,
        CENTER.1 as f32,
        1.0,
    ]);
    // Same seed for every solver, so they all see the same scenes.
    let mut rng = Lcg(seed);
    let (mut rot, mut trans, mut failures) = (Vec::new(), Vec::new(), 0);
    for _ in 0..TRIALS {
        let scene = make_scene(&mut rng, kind);
        let Ok(pose) = solve_pnp(&scene.world, &scene.image, &k, None, method.clone()) else {
            failures += 1;
            continue;
        };
        let r = rotation_error_deg(&pose.rotation, &scene.rotation);
        let t = scene.translation;
        let dt = [
            pose.translation.x as f64 - t[0],
            pose.translation.y as f64 - t[1],
            pose.translation.z as f64 - t[2],
        ];
        let rel =
            (dt.iter().map(|v| v * v).sum::<f64>() / t.iter().map(|v| v * v).sum::<f64>()).sqrt();
        if !(r.is_finite() && r <= FAILURE_DEG) {
            failures += 1;
        }
        rot.push(r);
        trans.push(100.0 * rel);
    }
    (
        median(rot),
        median(trans),
        100.0 * failures as f64 / TRIALS as f64,
    )
}

fn methods() -> Vec<(&'static str, PnPMethod)> {
    vec![
        ("EPnP", PnPMethod::EPnPDefault),
        (
            "EPnP+LM",
            PnPMethod::EPnP(EPnPParams {
                refine_lm: Some(LMRefineParams::default()),
                ..Default::default()
            }),
        ),
        ("SQPnP", PnPMethod::SQPnPDefault),
    ]
}

fn table(title: &str, column: &str, rows: Vec<(String, SceneKind)>) {
    let methods = methods();
    println!("\n### {title}\n");
    print!("| {column} |");
    for (name, _) in &methods {
        print!(" {name} rot° | {name} trans % | {name} fail % |");
    }
    println!();
    print!("|---|");
    for _ in &methods {
        print!("---|---|---|");
    }
    println!();
    for (i, (label, kind)) in rows.iter().enumerate() {
        print!("| {label} |");
        for (_, method) in &methods {
            let (r, t, f) = evaluate(method, kind, 1000 + i as u64);
            print!(" {r:.3} | {t:.3} | {f:.1} |");
        }
        println!();
    }
}

fn main() {
    println!("PnP accuracy, {TRIALS} random scenes per cell, fixed seeds; failure = rotation error > {FAILURE_DEG}° or no pose.");

    table(
        "Correspondence count (1 px noise, points in a 2 m cube 6 m away)",
        "n",
        [4, 6, 10, 20, 50, 100, 500]
            .into_iter()
            .map(|n| {
                (
                    n.to_string(),
                    SceneKind {
                        n,
                        noise_px: 1.0,
                        size: 1.0,
                        depth: 6.0,
                        planar: false,
                    },
                )
            })
            .collect(),
    );

    table(
        "Pixel noise (20 points in a 2 m cube 6 m away)",
        "noise px",
        [0.0, 0.5, 1.0, 2.0, 5.0]
            .into_iter()
            .map(|noise| {
                (
                    format!("{noise}"),
                    SceneKind {
                        n: 20,
                        noise_px: noise,
                        size: 1.0,
                        depth: 6.0,
                        planar: false,
                    },
                )
            })
            .collect(),
    );

    table(
        "Harder configurations (1 px noise)",
        "case",
        vec![
            (
                "planar, n=20".into(),
                SceneKind {
                    n: 20,
                    noise_px: 1.0,
                    size: 1.0,
                    depth: 6.0,
                    planar: true,
                },
            ),
            (
                "small object far (0.5 m cube at 30 m), n=20".into(),
                SceneKind {
                    n: 20,
                    noise_px: 1.0,
                    size: 0.25,
                    depth: 30.0,
                    planar: false,
                },
            ),
            (
                "close range (2 m cube at 2.5 m), n=20".into(),
                SceneKind {
                    n: 20,
                    noise_px: 1.0,
                    size: 1.0,
                    depth: 2.5,
                    planar: false,
                },
            ),
            (
                "minimum, n=4".into(),
                SceneKind {
                    n: 4,
                    noise_px: 0.5,
                    size: 1.0,
                    depth: 6.0,
                    planar: false,
                },
            ),
        ],
    );
}
