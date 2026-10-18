//! Accuracy and robustness of the PnP solvers on synthetic correspondences.
//!
//! Prints Markdown tables of the median rotation error, the median relative translation error
//! and the failure rate (rotation error above 5° or no pose returned) for EPnP, EPnP + LM and
//! SQPnP: across correspondence counts, across pixel-noise levels, and in harder configurations
//! (planar points, a small object far away, close range, the 4-point minimum). A last table runs
//! `solve_pnp_ransac` with each kernel on scenes with outliers. Each cell is 200 random scenes with
//! a fixed seed, so the output is reproducible.
//!
//! Run with `cargo bench -p kornia-3d --bench bench_pnp_accuracy`.

use kornia_3d::pnp::{
    solve_pnp, solve_pnp_ransac, EPnPParams, LMRefineParams, PnPMethod, PnPResult, RansacParams,
};
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec3AF32, Vec3F64};
use rand::{rngs::StdRng, RngExt, SeedableRng};

const TRIALS: usize = 200;
const FAILURE_DEG: f64 = 5.0;
const FOCAL: f64 = 800.0;
const CENTER: (f64, f64) = (640.0, 480.0);

// The crate's unit-test helpers (`src/test_utils.rs`), shared with the benches; a bench uses
// only some of them.
#[path = "../src/test_utils.rs"]
#[allow(dead_code)]
mod test_utils;
use test_utils::{gauss, random_rotation, rotation_error_deg};

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
    rotation: Mat3F64,
    translation: Vec3F64,
}

fn make_scene(rng: &mut StdRng, kind: &SceneKind) -> Scene {
    let rotation = random_rotation(rng, std::f64::consts::PI);
    let translation = Vec3F64::new(
        rng.random_range(-0.1..0.1) * kind.depth,
        rng.random_range(-0.1..0.1) * kind.depth,
        kind.depth,
    );
    let mut world = Vec::with_capacity(kind.n);
    let mut image = Vec::with_capacity(kind.n);
    while world.len() < kind.n {
        let z = if kind.planar {
            0.0
        } else {
            rng.random_range(-kind.size..kind.size)
        };
        let p = Vec3F64::new(
            rng.random_range(-kind.size..kind.size),
            rng.random_range(-kind.size..kind.size),
            z,
        );
        let pc = rotation * p + translation;
        // Keep only points well in front of the camera, as a real detector would.
        if pc.z < 0.1 * kind.depth {
            continue;
        }
        world.push(p.as_vec3af32());
        image.push(Vec2F32::new(
            (FOCAL * pc.x / pc.z + CENTER.0 + kind.noise_px * gauss(rng)) as f32,
            (FOCAL * pc.y / pc.z + CENTER.1 + kind.noise_px * gauss(rng)) as f32,
        ));
    }
    Scene {
        world,
        image,
        rotation,
        translation,
    }
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

/// Median rotation error (°), median relative translation error (%) and failure rate (%) of
/// `solve` over `TRIALS` scenes of `kind`, each with `outliers` random correspondences added.
fn evaluate(
    kind: &SceneKind,
    outliers: usize,
    seed: u64,
    solve: impl Fn(&[Vec3AF32], &[Vec2F32], &Mat3AF32) -> Option<PnPResult>,
) -> (f64, f64, f64) {
    let k = test_utils::k_matrix(FOCAL as f32, FOCAL as f32, CENTER.0 as f32, CENTER.1 as f32);
    // Same seed for every solver, so they all see the same scenes.
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut rot, mut trans, mut failures) = (Vec::new(), Vec::new(), 0);
    for _ in 0..TRIALS {
        let scene = make_scene(&mut rng, kind);
        let (mut world, mut image) = (scene.world.clone(), scene.image.clone());
        for _ in 0..outliers {
            world.push(Vec3AF32::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ));
            image.push(Vec2F32::new(
                rng.random_range(0.0..2.0 * CENTER.0 as f32),
                rng.random_range(0.0..2.0 * CENTER.1 as f32),
            ));
        }
        let Some(pose) = solve(&world, &image, &k) else {
            failures += 1;
            continue;
        };
        let r = rotation_error_deg(&pose.rotation, &scene.rotation);
        let t = scene.translation;
        let rel = (Vec3F64::from(pose.translation) - t).length() / t.length();
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
            let (r, t, f) = evaluate(kind, 0, 1000 + i as u64, |w, img, k| {
                solve_pnp(w, img, k, None, method.clone()).ok()
            });
            print!(" {r:.3} | {t:.3} | {f:.1} |");
        }
        println!();
    }
}

/// The same statistics for `solve_pnp_ransac` with each kernel (3 px threshold, fixed seed, other
/// parameters default), on scenes with `outliers` random correspondences added.
fn ransac_table(title: &str, rows: Vec<(String, SceneKind, usize)>) {
    let kernels = [
        ("EPnP", PnPMethod::EPnPDefault),
        ("AP3P", PnPMethod::AP3PDefault),
        ("SQPnP", PnPMethod::SQPnPDefault),
    ];
    let params = RansacParams {
        reproj_threshold_px: 3.0,
        random_seed: Some(1),
        ..Default::default()
    };
    println!("\n### {title}\n");
    print!("| case |");
    for (name, _) in &kernels {
        print!(" {name} rot° | {name} trans % | {name} fail % |");
    }
    println!();
    print!("|---|");
    for _ in &kernels {
        print!("---|---|---|");
    }
    println!();
    for (i, (label, kind, outliers)) in rows.iter().enumerate() {
        print!("| {label} |");
        for (_, kernel) in &kernels {
            let (r, t, f) = evaluate(kind, *outliers, 2000 + i as u64, |w, img, k| {
                solve_pnp_ransac(w, img, k, None, kernel.clone(), &params)
                    .ok()
                    .map(|result| result.pose)
            });
            print!(" {r:.3} | {t:.3} | {f:.1} |");
        }
        println!();
    }
}

fn main() {
    println!(
        "PnP accuracy, {TRIALS} random scenes per cell, fixed seeds; \
         failure = rotation error > {FAILURE_DEG}° or no pose."
    );

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

    let inliers = |planar| SceneKind {
        n: 50,
        noise_px: 1.0,
        size: 1.0,
        depth: 6.0,
        planar,
    };
    ransac_table(
        "Inside solve_pnp_ransac (50 inliers with 1 px noise, plus random outliers)",
        vec![
            ("no outliers".into(), inliers(false), 0),
            ("25% outliers".into(), inliers(false), 17),
            ("50% outliers".into(), inliers(false), 50),
            ("planar, 25% outliers".into(), inliers(true), 17),
        ],
    );
}
