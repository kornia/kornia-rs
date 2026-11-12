//! One projective ICP call per frame, as a real-time loop runs it: the in-place pyramid rebuild
//! of the new frame, then every level of the solve against a fixed keyframe, with each
//! [`IcpKernel`] and each [`Association`].
//!
//! Besides criterion, prints per variant the per-call distribution (wall median / p95, on-CPU
//! mean from `/proc/thread-self/schedstat`, which stays meaningful on a loaded box), the pyramid
//! and solve split, the iteration count and the error against the rendered motion.
//! `ICP_BENCH_DISTRIBUTION_ONLY=1` skips criterion.
//!
//! The scene renderer is a copy of the one in `tests/icp_alloc.rs` (the crate's own synthetic
//! scenes are test-only), with the camera rotated as well as moved.

use criterion::{criterion_group, BenchmarkId, Criterion};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::registration::{
    icp_projective_plane_with, Association, IcpKernel, IcpPlaneCriteria, IcpPlaneResult,
    IcpWorkspace, PyramidScratch, RgbdPyramid,
};
use kornia_algebra::{Mat3F64, Vec3F64, SO3F64};
use kornia_image::{Image, ImageSize};
use std::time::Instant;

const LEVELS: usize = 3;

/// Translation of the frame's camera from the keyframe's, metres.
fn true_translation() -> Vec3F64 {
    Vec3F64::new(0.02, 0.0, 0.01)
}

/// Yaw of the frame's camera from the keyframe's, degrees.
const TRUE_YAW_DEG: f64 = 1.0;

fn true_rotation() -> Mat3F64 {
    SO3F64::exp(Vec3F64::new(0.0, TRUE_YAW_DEG.to_radians(), 0.0)).matrix()
}

fn camera(size: ImageSize) -> PinholeCamera {
    let f = 0.625 * size.width as f64;
    PinholeCamera {
        fx: f,
        fy: f,
        cx: size.width as f64 / 2.0 - 0.5,
        cy: size.height as f64 / 2.0 - 0.5,
        ..PinholeCamera::IDENTITY
    }
}

/// A wall corner and a sphere seen from a camera at `T_world_cam = (rot_wc, t_wc)`, z-depth in
/// millimetres.
fn render(size: ImageSize, rot_wc: &Mat3F64, t_wc: &Vec3F64) -> Image<u16, 1> {
    let planes = [
        (Vec3F64::new(0.0, 0.0, 1.0), 2.0),
        (Vec3F64::new(1.0, 0.0, 0.0), 0.8),
        (Vec3F64::new(0.0, 1.0, 0.0), 0.6),
    ];
    let (centre, radius) = (Vec3F64::new(-0.2, -0.15, 1.2), 0.3);
    let cam = camera(size);
    let mut out = Image::<u16, 1>::from_size_val(size, 0).expect("depth image");
    for (idx, px) in out.as_slice_mut().iter_mut().enumerate() {
        let (u, v) = (idx % size.width, idx / size.width);
        let dc = Vec3F64::new(
            (u as f64 - cam.cx) / cam.fx,
            (v as f64 - cam.cy) / cam.fy,
            1.0,
        );
        // ray in the world, scaled so that its camera-frame z is 1: `t` is the z-depth
        let d = *rot_wc * dc;
        let mut best = f64::INFINITY;
        for (n, off) in &planes {
            let den = n.dot(d);
            if den.abs() > 1e-12 {
                let t = (off - n.dot(*t_wc)) / den;
                if t > 0.05 && t < best {
                    best = t;
                }
            }
        }
        let oc = *t_wc - centre;
        let (a, b, c) = (d.dot(d), oc.dot(d), oc.dot(oc) - radius * radius);
        let disc = b * b - a * c;
        if disc >= 0.0 {
            let t = (-b - disc.sqrt()) / a;
            if t > 0.05 && t < best {
                best = t;
            }
        }
        if best.is_finite() {
            *px = (best * 1000.0).round().min(65535.0) as u16;
        }
    }
    out
}

struct Case {
    name: &'static str,
    size: ImageSize,
    keyframe: Image<u16, 1>,
    frame: Image<u16, 1>,
}

fn cases() -> Vec<Case> {
    [(160, 100, "160x100"), (320, 200, "320x200")]
        .into_iter()
        .map(|(width, height, name)| {
            let size = ImageSize { width, height };
            Case {
                name,
                size,
                keyframe: render(size, &Mat3F64::IDENTITY, &Vec3F64::ZERO),
                frame: render(size, &true_rotation(), &true_translation()),
            }
        })
        .collect()
}

/// On-CPU nanoseconds of this thread so far, or `None` off Linux.
fn thread_cpu_ns() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
    s.split_whitespace().next()?.parse().ok()
}

const KERNELS: [IcpKernel; 2] = [IcpKernel::Scalar, IcpKernel::Simd];
const ASSOCIATIONS: [Association; 3] = [
    Association::Nearest,
    Association::Bilinear,
    Association::Hybrid,
];

fn variant_name(kernel: IcpKernel, association: Association) -> String {
    let k = match kernel {
        IcpKernel::Scalar => "scalar",
        _ => "simd",
    };
    let a = match association {
        Association::Nearest => "nearest",
        Association::Bilinear => "bilinear",
        _ => "hybrid",
    };
    format!("{k}-{a}")
}

/// Per-variant state reused across calls, as a real-time loop holds it.
struct Runner {
    criteria: IcpPlaneCriteria,
    ws: IcpWorkspace,
    scratch: PyramidScratch,
    pyr: RgbdPyramid,
    keyframe: RgbdPyramid,
}

impl Runner {
    fn new(kernel: IcpKernel, association: Association, case: &Case) -> Self {
        let cam = camera(case.size);
        let keyframe = RgbdPyramid::from_depth_mm(&case.keyframe, &cam, LEVELS).expect("keyframe");
        let pyr = RgbdPyramid::from_depth_mm(&case.frame, &cam, LEVELS).expect("frame");
        let mut criteria = IcpPlaneCriteria::default();
        criteria.association = association;
        Self {
            criteria,
            ws: IcpWorkspace::new(kernel),
            scratch: PyramidScratch::default(),
            pyr,
            keyframe,
        }
    }

    /// One call; returns the (pyramid, solve) on-CPU ms and the result.
    fn call(&mut self, case: &Case) -> (f64, f64, IcpPlaneResult) {
        let cam = camera(case.size);
        let c0 = thread_cpu_ns().unwrap_or(0);
        self.pyr
            .rebuild_from_depth_mm(&case.frame, &cam, LEVELS, &mut self.scratch)
            .expect("pyramid");
        let c1 = thread_cpu_ns().unwrap_or(0);
        let (r, t) = (Mat3F64::IDENTITY, Vec3F64::ZERO);
        let result = icp_projective_plane_with(
            &mut self.ws,
            &self.pyr,
            &self.keyframe,
            &r,
            &t,
            &self.criteria,
        )
        .expect("icp");
        let c2 = thread_cpu_ns().unwrap_or(0);
        let result = std::hint::black_box(result);
        ((c1 - c0) as f64 * 1e-6, (c2 - c1) as f64 * 1e-6, result)
    }
}

/// Error of `r` against the rendered motion: (rotation deg, translation mm).
fn error(r: &IcpPlaneResult) -> (f64, f64) {
    let d = SO3F64::from_matrix(&(true_rotation().transpose() * r.rotation));
    let rot = d.log().length().to_degrees();
    (rot, (r.translation - true_translation()).length() * 1e3)
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// Per-call wall and on-CPU distribution, printed before criterion runs.
fn distribution() {
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!("loadavg: {}", load.trim());
    if let Ok(exe) = std::env::current_exe() {
        println!("binary: {}", exe.display());
    }
    for case in cases() {
        // A/A guard: the two kernels must not be the same code under two names
        let mut kernels_differ = false;
        for association in ASSOCIATIONS {
            let mut translations = Vec::new();
            for kernel in KERNELS {
                let mut runner = Runner::new(kernel, association, &case);
                for _ in 0..20 {
                    runner.call(&case);
                }
                let (mut wall, mut pyr_t, mut icp_t) = (Vec::new(), Vec::new(), Vec::new());
                let n = 200;
                let total0 = thread_cpu_ns().unwrap_or(0);
                let mut last = None;
                for _ in 0..n {
                    let t0 = Instant::now();
                    let (p, i, r) = runner.call(&case);
                    wall.push(t0.elapsed().as_secs_f64() * 1e3);
                    pyr_t.push(p);
                    icp_t.push(i);
                    last = Some(r);
                }
                let Some(r) = last else { return };
                // schedstat advances at scheduler ticks, so one call's CPU time is quantised;
                // the mean over all calls is exact
                let cpu_mean = (thread_cpu_ns().unwrap_or(0) - total0) as f64 * 1e-6 / n as f64;
                let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
                let (rot_err, t_err) = error(&r);
                println!(
                    "{} {:<16} kernel {:?}: wall median {:6.2} p95 {:6.2} ms | on-CPU mean \
                     {:6.2} ms | pyramid {:.2} + solve {:.2} ms mean | {} iterations, {} \
                     inliers | error {:.4} deg {:.3} mm",
                    case.name,
                    variant_name(kernel, association),
                    runner.ws.kernel(),
                    pct(&mut wall, 0.5),
                    pct(&mut wall, 0.95),
                    cpu_mean,
                    mean(&pyr_t),
                    mean(&icp_t),
                    r.iterations,
                    r.num_inliers,
                    rot_err,
                    t_err,
                );
                translations.push(r.translation.to_array().map(f64::to_bits));
            }
            kernels_differ |= translations[0] != translations[1];
        }
        assert!(
            kernels_differ,
            "{}: scalar and SIMD gave the same bits in every mode; is one arm mislabelled?",
            case.name
        );
    }
}

fn bench_icp(c: &mut Criterion) {
    let mut group = c.benchmark_group("icp_call");
    for case in cases() {
        for association in ASSOCIATIONS {
            for kernel in KERNELS {
                let mut runner = Runner::new(kernel, association, &case);
                group.bench_with_input(
                    BenchmarkId::new(variant_name(kernel, association), case.name),
                    &case,
                    |b, case| b.iter(|| runner.call(case)),
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_icp);

fn main() {
    distribution();
    if std::env::var("ICP_BENCH_DISTRIBUTION_ONLY").is_err() {
        benches();
        Criterion::default().configure_from_args().final_summary();
    }
}
