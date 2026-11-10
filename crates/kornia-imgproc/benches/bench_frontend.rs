//! Visual-frontend stages on a real EuRoC stereo frame pair (752x480).
//!
//! `KORNIA_BENCH_EUROC` points at a sequence's `mav0` directory; without it
//! the bench is skipped. `KORNIA_BENCH_FRAME` picks the left frame index.
//!
//! Times are process CPU time, not wall time: on a loaded machine a reniced
//! bench is mostly waiting for a core, and the CPU budget is what a
//! fixed core count has to fit.
use criterion::measurement::{Measurement, ValueFormatter};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::features::OrbDetector;
use kornia_imgproc::filter::scharr_spatial_gradient_float;
use kornia_imgproc::optical_flow_pyr_lk::{
    build_lk_precomputed, calc_optical_flow_pyr_lk_with_precomputed, PyrLKParams,
};
use kornia_imgproc::pyramid::pyrdown_f32;
use std::hint::black_box;
use std::path::{Path, PathBuf};

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

extern "C" {
    fn clock_gettime(clock_id: i32, tp: *mut Timespec) -> i32;
}

/// Linux `CLOCK_PROCESS_CPUTIME_ID`, the same value on x86_64 and aarch64.
const CLOCK_PROCESS_CPUTIME_ID: i32 = 2;

fn process_cpu_ns() -> f64 {
    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable timespec with the C layout
    // (`#[repr(C)]`, two 64-bit fields on 64-bit Linux), and the clock id is a
    // valid Linux clock.
    unsafe { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 * 1e9 + ts.tv_nsec as f64
}

struct ProcessCpu;
struct NanosFormatter;

impl ValueFormatter for NanosFormatter {
    fn scale_values(&self, _typical: f64, values: &mut [f64]) -> &'static str {
        values.iter_mut().for_each(|v| *v /= 1e6);
        "ms cpu"
    }
    fn scale_throughputs(&self, _t: f64, _th: &Throughput, _v: &mut [f64]) -> &'static str {
        "n/a"
    }
    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "ns"
    }
}

impl Measurement for ProcessCpu {
    type Intermediate = f64;
    type Value = f64;
    fn start(&self) -> f64 {
        process_cpu_ns()
    }
    fn end(&self, start: f64) -> f64 {
        process_cpu_ns() - start
    }
    fn add(&self, a: &f64, b: &f64) -> f64 {
        a + b
    }
    fn zero(&self) -> f64 {
        0.0
    }
    fn to_f64(&self, v: &f64) -> f64 {
        *v
    }
    fn formatter(&self) -> &dyn ValueFormatter {
        &NanosFormatter
    }
}

fn load_gray(path: &Path) -> Image<u8, 1> {
    let img = image::open(path).expect("readable PNG").to_luma8();
    let size = ImageSize {
        width: img.width() as usize,
        height: img.height() as usize,
    };
    Image::new(size, img.into_raw()).expect("valid image")
}

fn frame_paths(mav0: &Path, cam: &str) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(mav0.join(cam).join("data"))
        .expect("cam data dir")
        .map(|e| e.expect("dir entry").path())
        .collect();
    paths.sort();
    paths
}

struct Fixture {
    left0: Image<u8, 1>,
    left1: Image<u8, 1>,
    right0: Image<u8, 1>,
}

fn fixture() -> Option<Fixture> {
    let mav0 = PathBuf::from(std::env::var("KORNIA_BENCH_EUROC").ok()?);
    let idx: usize = std::env::var("KORNIA_BENCH_FRAME")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let left = frame_paths(&mav0, "cam0");
    let right = frame_paths(&mav0, "cam1");
    Some(Fixture {
        left0: load_gray(&left[idx]),
        left1: load_gray(&left[idx + 1]),
        right0: load_gray(&right[idx]),
    })
}

fn lk_pyramid(img: &Image<f32, 1>, max_level: usize) -> Vec<Image<f32, 1>> {
    let mut pyr = vec![img.clone()];
    for _ in 0..max_level {
        let src = pyr.last().expect("level 0");
        let size = ImageSize {
            width: src.width().div_ceil(2),
            height: src.height().div_ceil(2),
        };
        let mut dst = Image::from_size_val(size, 0.0f32).expect("alloc");
        pyrdown_f32(src, &mut dst).expect("pyrdown");
        pyr.push(dst);
    }
    pyr
}

fn is_border_at(p: &[f32; 2], size: ImageSize, lvl: usize) -> bool {
    let s = 1.0 / (1u32 << lvl) as f32;
    let cols = size.width.div_ceil(1 << lvl) as f32;
    let rows = size.height.div_ceil(1 << lvl) as f32;
    let (x, y) = (p[0] * s, p[1] * s);
    !(x >= 10.0 && y >= 10.0 && x < cols - 11.0 && y < rows - 11.0)
}

/// Share of LK start windows per level that miss the interior fast path.
fn report_border_share(pts: &[[f32; 2]], size: ImageSize, max_level: usize) {
    let mut line = String::from("lk border share by level:");
    for lvl in 0..=max_level {
        let border = pts.iter().filter(|p| is_border_at(p, size, lvl)).count();
        line += &format!(" L{lvl}={:.2}", border as f32 / pts.len().max(1) as f32);
    }
    eprintln!("{line} (n={})", pts.len());
}

fn bench_frontend(c: &mut Criterion<ProcessCpu>) {
    let Some(fx) = fixture() else {
        eprintln!("KORNIA_BENCH_EUROC unset; skipping bench_frontend");
        return;
    };
    let size = fx.left0.size();
    let params = PyrLKParams::default();
    let detector = OrbDetector {
        n_keypoints: 1000,
        ..OrbDetector::default()
    };
    let prev_f32 = fx.left0.cast::<f32>().expect("cast");
    let next_f32 = fx.left1.cast::<f32>().expect("cast");
    let features = detector.detect_and_extract_u8(&fx.left0).expect("orb");
    let pts = features.keypoints_xy.clone();
    report_border_share(&pts, size, params.max_level);

    let mut g = c.benchmark_group("frontend");
    g.sample_size(60);

    g.bench_function("cast_u8_f32", |b| {
        b.iter(|| black_box(fx.left0.cast::<f32>().expect("cast")))
    });

    let half = ImageSize {
        width: size.width.div_ceil(2),
        height: size.height.div_ceil(2),
    };
    let mut half_dst = Image::from_size_val(half, 0.0f32).expect("alloc");
    g.bench_function("pyrdown_f32_l0_l1", |b| {
        b.iter(|| pyrdown_f32(black_box(&prev_f32), &mut half_dst).expect("pyrdown"))
    });

    g.bench_function("lk_pyramid_one_image", |b| {
        b.iter(|| black_box(lk_pyramid(&prev_f32, params.max_level)))
    });

    let pyr = lk_pyramid(&prev_f32, params.max_level);
    g.bench_function("scharr_gradient_pyramid", |b| {
        b.iter(|| {
            for img in &pyr {
                let mut ix = Image::from_size_val(img.size(), 0.0f32).expect("alloc");
                let mut iy = Image::from_size_val(img.size(), 0.0f32).expect("alloc");
                scharr_spatial_gradient_float(img, &mut ix, &mut iy).expect("scharr");
                black_box((ix, iy));
            }
        })
    });

    g.bench_function("build_lk_precomputed", |b| {
        b.iter(|| {
            black_box(build_lk_precomputed(&prev_f32, &next_f32, params.max_level).expect("lk"))
        })
    });

    let pre = build_lk_precomputed(&prev_f32, &next_f32, params.max_level).expect("lk");
    g.bench_function(format!("lk_track_{}_pts", pts.len()), |b| {
        b.iter(|| {
            black_box(
                calc_optical_flow_pyr_lk_with_precomputed(&pre, &pts, None, &params).expect("lk"),
            )
        })
    });

    let top = params.max_level;
    let (border_pts, interior_pts): (Vec<[f32; 2]>, Vec<[f32; 2]>) =
        pts.iter().partition(|p| is_border_at(p, size, top));
    for (name, subset) in [("interior", &interior_pts), ("border", &border_pts)] {
        g.bench_function(format!("lk_track_{name}_{}_pts", subset.len()), |b| {
            b.iter(|| {
                black_box(
                    calc_optical_flow_pyr_lk_with_precomputed(&pre, subset, None, &params)
                        .expect("lk"),
                )
            })
        });
    }

    g.bench_function("orb_detect_and_extract_u8_1000", |b| {
        b.iter(|| black_box(detector.detect_and_extract_u8(&fx.left0).expect("orb")))
    });
    g.bench_function("orb_build_pyramid_u8", |b| {
        b.iter(|| black_box(detector.build_pyramid_u8(&fx.left0).expect("pyramid")))
    });
    let orb_pyr = detector.build_pyramid_u8(&fx.left0).expect("pyramid");
    g.bench_function("orb_detect_and_extract_u8_1000_from_pyramid", |b| {
        b.iter(|| {
            black_box(
                detector
                    .detect_and_extract_u8_from_pyramid(&orb_pyr)
                    .expect("orb"),
            )
        })
    });
    g.bench_function("orb_detect_and_extract_u8_1000_right", |b| {
        b.iter(|| black_box(detector.detect_and_extract_u8(&fx.right0).expect("orb")))
    });
    // Cost vs keypoint budget: ORB at N, LK at the about 0.4 N tracks a
    // SLAM frontend keeps (an even spatial subsample of the ORB points).
    for n in [500usize, 700, 850, 1000, 1200] {
        let orb = OrbDetector {
            n_keypoints: n,
            ..OrbDetector::default()
        };
        g.bench_function(format!("orb_n{n}"), |b| {
            b.iter(|| black_box(orb.detect_and_extract_u8(&fx.left0).expect("orb")))
        });
        let k = n * 2 / 5;
        let step = (pts.len() / k).max(1);
        let subset: Vec<[f32; 2]> = pts.iter().step_by(step).take(k).copied().collect();
        g.bench_function(format!("lk_tracks_{}", subset.len()), |b| {
            b.iter(|| {
                black_box(
                    calc_optical_flow_pyr_lk_with_precomputed(&pre, &subset, None, &params)
                        .expect("lk"),
                )
            })
        });
    }
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_measurement(ProcessCpu);
    targets = bench_frontend
}
criterion_main!(benches);
