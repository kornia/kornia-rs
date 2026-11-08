//! Real-time budget of one projective ICP call: pyramid build of the new frame plus every level.
//!
//! Budget: 20 ms on one Jetson Orin core at 320x200 (15 Hz depth); 8 ms single-threaded on an
//! x86 laptop. Besides criterion, prints the per-call distribution (median / p95) of wall time and
//! of on-CPU time from `/proc/thread-self/schedstat`, which stays meaningful on a loaded box.

use criterion::{criterion_group, BenchmarkId, Criterion};
use kornia_3d::registration::{
    icp_projective_plane, icp_projective_plane_with, Association, DepthIntrinsics, IcpKernel,
    IcpPlaneCriteria, IcpWorkspace, PyramidScratch, RgbdPyramid,
};
use std::time::Instant;

const LEVELS: usize = 3;

/// Corner of three walls plus a sphere, as in the crate's synthetic tests.
fn render(intr: &DepthIntrinsics, rot_wc: &[[f64; 3]; 3], t_wc: &[f64; 3]) -> Vec<u16> {
    let planes = [
        ([0.0, 0.0, 1.0], 2.0),
        ([1.0, 0.0, 0.0], 0.8),
        ([0.0, 1.0, 0.0], 0.6),
    ];
    let (c, rad) = ([-0.2, -0.15, 1.2], 0.3);
    let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut out = vec![0u16; intr.width * intr.height];
    for v in 0..intr.height {
        for u in 0..intr.width {
            let dc = [
                (u as f64 - intr.cx) / intr.fx,
                (v as f64 - intr.cy) / intr.fy,
                1.0,
            ];
            let d = [
                dot(&rot_wc[0], &dc),
                dot(&rot_wc[1], &dc),
                dot(&rot_wc[2], &dc),
            ];
            let mut best = f64::INFINITY;
            for (n, off) in &planes {
                let den = dot(n, &d);
                if den.abs() > 1e-12 {
                    let t = (off - dot(n, t_wc)) / den;
                    if t > 0.05 && t < best {
                        best = t;
                    }
                }
            }
            let oc = [t_wc[0] - c[0], t_wc[1] - c[1], t_wc[2] - c[2]];
            let (a, b, cc) = (dot(&d, &d), dot(&oc, &d), dot(&oc, &oc) - rad * rad);
            let disc = b * b - a * cc;
            if disc >= 0.0 {
                let t = (-b - disc.sqrt()) / a;
                if t > 0.05 && t < best {
                    best = t;
                }
            }
            if best.is_finite() {
                out[v * intr.width + u] = (best * 1000.0).round().min(65535.0) as u16;
            }
        }
    }
    out
}

fn intrinsics(width: usize, height: usize) -> DepthIntrinsics {
    let f = 0.625 * width as f64;
    DepthIntrinsics {
        fx: f,
        fy: f,
        cx: width as f64 / 2.0 - 0.5,
        cy: height as f64 / 2.0 - 0.5,
        width,
        height,
    }
}

fn yaw(deg: f64) -> [[f64; 3]; 3] {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

struct Case {
    name: &'static str,
    intr: DepthIntrinsics,
    keyframe: RgbdPyramid,
    frame: Vec<u16>,
}

fn cases() -> Vec<Case> {
    [(160, 100, "160x100"), (320, 200, "320x200")]
        .into_iter()
        .map(|(w, h, name)| {
            let intr = intrinsics(w, h);
            let kf = render(&intr, &IDENTITY, &[0.0; 3]);
            let keyframe = RgbdPyramid::from_depth_mm(&kf, &intr, LEVELS).expect("keyframe");
            let frame = render(&intr, &yaw(1.0), &[0.02, 0.0, 0.01]);
            Case {
                name,
                intr,
                keyframe,
                frame,
            }
        })
        .collect()
}

/// On-CPU nanoseconds of this thread so far, or `None` off Linux.
fn thread_cpu_ns() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
    s.split_whitespace().next()?.parse().ok()
}

/// What one timed call runs.
#[derive(Clone, Copy)]
enum Variant {
    /// `from_depth_mm` + `icp_projective_plane`: allocating API, the pre-workspace path.
    Legacy,
    /// In-place pyramid + workspace, scalar kernels, nearest association.
    Scalar,
    /// In-place pyramid + workspace, SIMD kernels, nearest association.
    Simd,
    /// As `Simd`, hybrid association.
    SimdHybrid,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::Legacy => "legacy",
            Variant::Scalar => "scalar",
            Variant::Simd => "simd",
            Variant::SimdHybrid => "simd-hybrid",
        }
    }
}

const VARIANTS: [Variant; 4] = [
    Variant::Legacy,
    Variant::Scalar,
    Variant::Simd,
    Variant::SimdHybrid,
];

/// Per-variant state reused across calls, as a real-time loop would hold it.
struct Runner {
    variant: Variant,
    criteria: IcpPlaneCriteria,
    ws: IcpWorkspace,
    scratch: PyramidScratch,
    pyr: RgbdPyramid,
}

impl Runner {
    fn new(variant: Variant) -> Self {
        let kernel = match variant {
            Variant::Scalar => IcpKernel::Scalar,
            _ => IcpKernel::Simd,
        };
        let association = match variant {
            Variant::SimdHybrid => Association::Hybrid,
            _ => Association::Nearest,
        };
        Self {
            variant,
            criteria: IcpPlaneCriteria {
                association,
                ..Default::default()
            },
            ws: IcpWorkspace::new(kernel),
            scratch: PyramidScratch::default(),
            pyr: RgbdPyramid { levels: Vec::new() },
        }
    }

    /// One call; returns (pyramid, icp) on-CPU ms and the iteration count.
    fn call(&mut self, case: &Case) -> (f64, f64, usize) {
        let c0 = thread_cpu_ns().unwrap_or(0);
        let (c1, r) = if let Variant::Legacy = self.variant {
            let pyr = RgbdPyramid::from_depth_mm(&case.frame, &case.intr, LEVELS).expect("pyramid");
            let c1 = thread_cpu_ns().unwrap_or(0);
            let r = icp_projective_plane(
                &pyr,
                &case.keyframe,
                IDENTITY,
                [0.0; 3],
                self.criteria.clone(),
            );
            (c1, r)
        } else {
            self.pyr
                .rebuild_from_depth_mm(
                    &case.frame,
                    &case.intr,
                    LEVELS,
                    self.ws.kernel,
                    &mut self.scratch,
                )
                .expect("pyramid");
            let c1 = thread_cpu_ns().unwrap_or(0);
            let r = icp_projective_plane_with(
                &mut self.ws,
                &self.pyr,
                &case.keyframe,
                IDENTITY,
                [0.0; 3],
                &self.criteria,
            );
            (c1, r)
        };
        let c2 = thread_cpu_ns().unwrap_or(0);
        let iterations = std::hint::black_box(r.expect("icp")).iterations;
        ((c1 - c0) as f64 * 1e-6, (c2 - c1) as f64 * 1e-6, iterations)
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// Per-call wall and on-CPU distribution, printed before criterion runs.
fn distribution() {
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!("loadavg: {}", load.trim());
    for case in cases() {
        for variant in VARIANTS {
            let mut runner = Runner::new(variant);
            for _ in 0..20 {
                runner.call(&case);
            }
            let (mut wall, mut cpu, mut pyr_t, mut icp_t) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut iters = 0;
            let n = 200;
            let total0 = thread_cpu_ns().unwrap_or(0);
            for _ in 0..n {
                let t0 = Instant::now();
                let (p, i, it) = runner.call(&case);
                wall.push(t0.elapsed().as_secs_f64() * 1e3);
                cpu.push(p + i);
                pyr_t.push(p);
                icp_t.push(i);
                iters = it;
            }
            // schedstat only advances at scheduler ticks, so per-call CPU is quantised to ~1 ms;
            // the mean over all calls is exact
            let cpu_mean = (thread_cpu_ns().unwrap_or(0) - total0) as f64 * 1e-6 / n as f64;
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            println!(
                "{} {:<11}: wall median {:6.2} p95 {:6.2} ms | on-CPU mean {:6.2} ms, p95 {:6.2} \
                 (tick-quantised) | pyramid {:.2} + icp {:.2} ms mean, {iters} iterations",
                case.name,
                variant.name(),
                pct(&mut wall, 0.5),
                pct(&mut wall, 0.95),
                cpu_mean,
                pct(&mut cpu, 0.95),
                mean(&pyr_t),
                mean(&icp_t),
            );
        }
    }
}

fn bench_icp(c: &mut Criterion) {
    let mut group = c.benchmark_group("icp_call");
    for case in cases() {
        for variant in VARIANTS {
            let mut runner = Runner::new(variant);
            group.bench_with_input(
                BenchmarkId::new(variant.name(), case.name),
                &case,
                |b, case| b.iter(|| runner.call(case)),
            );
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
