//! Accuracy and robustness of the PnP solvers on synthetic correspondences.

use super::{
    solve_pnp, solve_pnp_ransac, EPnPParams, LMRefineParams, PnPMethod, PnPResult, RansacParams,
};
use crate::test_utils::{k_default, rotation_error_deg, translation_error, Scene, SceneConfig};
use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
use rand::{rngs::StdRng, SeedableRng};

const TRIALS: usize = 200;
const FAILURE_DEG: f64 = 5.0;

/// One table row: `TRIALS` scenes of shape `config`, each with `outliers` wrong correspondences.
struct Row {
    label: String,
    config: SceneConfig,
    outliers: usize,
}

impl Row {
    fn new(label: impl Into<String>, config: SceneConfig) -> Self {
        Self {
            label: label.into(),
            config,
            outliers: 0,
        }
    }

    fn with_outliers(self, outliers: usize) -> Self {
        Self { outliers, ..self }
    }
}

/// How well a solver did on one row.
struct Accuracy {
    /// Median rotation error, in degrees.
    rotation_deg: f64,
    /// Median translation error, in percent of the camera distance.
    translation_pct: f64,
    /// Scenes with no pose or a rotation error above `FAILURE_DEG`, in percent.
    failure_pct: f64,
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

/// The [`Accuracy`] of `solve` on `row`. The scenes depend only on `seed`, so every solver sees
/// the same ones.
fn evaluate(
    row: &Row,
    seed: u64,
    solve: impl Fn(&[Vec3AF32], &[Vec2F32], &Mat3AF32) -> Option<PnPResult>,
) -> Accuracy {
    let k = k_default();
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut rot, mut trans, mut failures) = (Vec::new(), Vec::new(), 0);
    for _ in 0..TRIALS {
        let mut scene = Scene::generate(&mut rng, k, &row.config);
        scene.add_outliers(&mut rng, row.outliers);
        let Some(pose) = solve(&scene.world, &scene.image, &k) else {
            failures += 1;
            continue;
        };
        let r = rotation_error_deg(pose.rotation, scene.rotation);
        if !(r.is_finite() && r <= FAILURE_DEG) {
            failures += 1;
        }
        rot.push(r);
        trans.push(
            100.0 * translation_error(pose.translation, scene.translation)
                / scene.translation.length(),
        );
    }
    Accuracy {
        rotation_deg: median(rot),
        translation_pct: median(trans),
        failure_pct: 100.0 * failures as f64 / TRIALS as f64,
    }
}

/// Prints one Markdown table: a row per `rows` entry, three columns per method. With `ransac`
/// set, each method is the kernel of `solve_pnp_ransac`; otherwise it runs through `solve_pnp`.
fn print_table(
    title: &str,
    column: &str,
    methods: &[(&str, PnPMethod)],
    ransac: Option<&RansacParams>,
    rows: &[Row],
    seed: u64,
) {
    println!("\n### {title}\n");
    print!("| {column} |");
    for (name, _) in methods {
        print!(" {name} rot° | {name} trans % | {name} fail % |");
    }
    println!();
    print!("|---|");
    for _ in methods {
        print!("---|---|---|");
    }
    println!();
    for (i, row) in rows.iter().enumerate() {
        print!("| {} |", row.label);
        for (_, method) in methods {
            let a = evaluate(row, seed + i as u64, |w, img, k| match ransac {
                None => solve_pnp(w, img, k, None, method.clone()).ok(),
                Some(params) => solve_pnp_ransac(w, img, k, None, method.clone(), params)
                    .ok()
                    .map(|result| result.pose),
            });
            print!(
                " {:.3} | {:.3} | {:.1} |",
                a.rotation_deg, a.translation_pct, a.failure_pct
            );
        }
        println!();
    }
}

/// Prints the accuracy tables in `crates/kornia-3d/benchmarks.md`. Run it in release:
///
/// ```text
/// cargo test --release -p kornia-3d --lib -- --ignored --nocapture pnp_accuracy_report
/// ```
#[test]
#[ignore = "measurement, not an assertion; run explicitly in release"]
fn pnp_accuracy_report() {
    let solvers = [
        ("EPnP", PnPMethod::EPnPDefault),
        (
            "EPnP+LM",
            PnPMethod::EPnP(EPnPParams {
                refine_lm: Some(LMRefineParams::default()),
                ..Default::default()
            }),
        ),
        ("SQPnP", PnPMethod::SQPnPDefault),
    ];
    let kernels = [
        ("EPnP", PnPMethod::EPnPDefault),
        ("AP3P", PnPMethod::AP3PDefault),
        ("SQPnP", PnPMethod::SQPnPDefault),
    ];
    let ransac = RansacParams {
        reproj_threshold_px: 3.0,
        random_seed: Some(1),
        ..Default::default()
    };
    let planar = |config: SceneConfig| SceneConfig {
        planar: true,
        ..config
    };

    println!(
        "PnP accuracy, {TRIALS} random scenes per cell, fixed seeds; \
         failure = rotation error > {FAILURE_DEG}° or no pose."
    );
    print_table(
        "Correspondence count (1 px noise, points in a 2 m cube 6 m away)",
        "n",
        &solvers,
        None,
        &[4, 6, 10, 20, 50, 100, 500]
            .map(|n| Row::new(n.to_string(), SceneConfig::cube(n, 6.0, 1.0))),
        1000,
    );
    print_table(
        "Pixel noise (20 points in a 2 m cube 6 m away)",
        "noise px",
        &solvers,
        None,
        &[0.0, 0.5, 1.0, 2.0, 5.0]
            .map(|noise| Row::new(format!("{noise}"), SceneConfig::cube(20, 6.0, noise))),
        1000,
    );
    print_table(
        "Harder configurations (1 px noise)",
        "case",
        &solvers,
        None,
        &[
            Row::new("planar, n=20", planar(SceneConfig::cube(20, 6.0, 1.0))),
            Row::new(
                "small object far (0.5 m cube at 30 m), n=20",
                SceneConfig {
                    size: 0.25,
                    ..SceneConfig::cube(20, 30.0, 1.0)
                },
            ),
            Row::new(
                "close range (2 m cube at 2.5 m), n=20",
                SceneConfig::cube(20, 2.5, 1.0),
            ),
            Row::new(
                "minimum, n=4 (0.5 px noise)",
                SceneConfig::cube(4, 6.0, 0.5),
            ),
        ],
        1000,
    );
    print_table(
        "Inside solve_pnp_ransac (50 inliers with 1 px noise, plus random outliers)",
        "case",
        &kernels,
        Some(&ransac),
        &[
            Row::new("no outliers", SceneConfig::cube(50, 6.0, 1.0)),
            Row::new("25% outliers", SceneConfig::cube(50, 6.0, 1.0)).with_outliers(17),
            Row::new("50% outliers", SceneConfig::cube(50, 6.0, 1.0)).with_outliers(50),
            Row::new(
                "planar, 25% outliers",
                planar(SceneConfig::cube(50, 6.0, 1.0)),
            )
            .with_outliers(17),
        ],
        2000,
    );
}
