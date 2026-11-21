//! Visual-frontend stages on a real EuRoC stereo frame pair (752x480).
//!
//! `KORNIA_BENCH_EUROC` points at a sequence's `mav0` directory; without it
//! the bench is skipped. `KORNIA_BENCH_FRAME` picks the left frame index.
//!
//! Times are process CPU time, not wall time: on a loaded machine a reniced
//! bench is mostly waiting for a core, and the CPU budget is what a
//! fixed core count has to fit. Before timing, each stage's heap
//! allocations per call (first call, then a later one) are printed.
use criterion::measurement::{Measurement, ValueFormatter};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::features::OrbDetector;
use kornia_imgproc::filter::scharr_spatial_gradient_separable_f32;
use kornia_imgproc::optical_flow_pyr_lk::{calc_optical_flow_pyr_lk_u8, LkPyramidU8, PyrLKParams};
use kornia_imgproc::pyramid::{pyrdown_separable_f32, pyrdown_u8};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Counts heap allocations and bytes; the stages run on the bench thread.
struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards every call to `System` unchanged; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Prints the allocations and bytes of the first and of a later call of `f`.
fn report_allocs(name: &str, mut f: impl FnMut()) {
    let mut once = || {
        let (a, b) = (
            ALLOCS.load(Ordering::Relaxed),
            BYTES.load(Ordering::Relaxed),
        );
        f();
        (
            ALLOCS.load(Ordering::Relaxed) - a,
            BYTES.load(Ordering::Relaxed) - b,
        )
    };
    let (cold, _) = (once(), once());
    let steady = once();
    eprintln!(
        "allocs {name}: first call {} ({} B), steady {} ({} B)",
        cold.0, cold.1, steady.0, steady.1
    );
}

#[cfg(unix)]
fn process_cpu_ns() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable `timespec`, and `CLOCK_PROCESS_CPUTIME_ID`
    // is a valid clock id on Linux and macOS.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_PROCESS_CPUTIME_ID) failed");
    ts.tv_sec as f64 * 1e9 + ts.tv_nsec as f64
}

#[cfg(not(unix))]
fn process_cpu_ns() -> f64 {
    panic!("bench_frontend measures process CPU time and needs a unix target");
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

fn bench_frontend(c: &mut Criterion<ProcessCpu>) {
    let Some(fx) = fixture() else {
        eprintln!("KORNIA_BENCH_EUROC unset; skipping bench_frontend");
        return;
    };
    let size = fx.left0.size();
    let params = PyrLKParams::default();
    let (win, max_level) = (params.win_size, params.max_level);
    let detector = OrbDetector {
        n_keypoints: 1000,
        ..OrbDetector::default()
    };
    let features = detector.detect_and_extract_u8(&fx.left0).expect("orb");
    let pts = features.keypoints_xy.clone();
    let prev_f32 = fx.left0.cast::<f32>().expect("cast");

    let half = ImageSize {
        width: size.width.div_ceil(2),
        height: size.height.div_ceil(2),
    };
    let mut half_u8 = Image::from_size_val(half, 0u8).expect("alloc");
    let mut half_f32 = Image::from_size_val(half, 0.0f32).expect("alloc");
    let mut ix = Image::from_size_val(size, 0.0f32).expect("alloc");
    let mut iy = Image::from_size_val(size, 0.0f32).expect("alloc");
    let mut prev = LkPyramidU8::new(&fx.left0, win, max_level, true).expect("lk");
    let mut next = LkPyramidU8::new(&fx.left1, win, max_level, false).expect("lk");
    let mut orb_levels = Vec::new();

    report_allocs("pyrdown_u8", || {
        pyrdown_u8(&fx.left0, &mut half_u8).expect("pyrdown")
    });
    report_allocs("pyrdown_separable_f32", || {
        pyrdown_separable_f32(&prev_f32, &mut half_f32).expect("pyrdown")
    });
    report_allocs("scharr_separable_f32", || {
        scharr_spatial_gradient_separable_f32(&prev_f32, &mut ix, &mut iy).expect("scharr")
    });
    report_allocs("lk_pyramids_new", || {
        black_box(LkPyramidU8::new(&fx.left0, win, max_level, true).expect("lk"));
        black_box(LkPyramidU8::new(&fx.left1, win, max_level, false).expect("lk"));
    });
    report_allocs("lk_pyramids_rebuild", || {
        prev.rebuild(&fx.left0, win, max_level, true).expect("lk");
        next.rebuild(&fx.left1, win, max_level, false).expect("lk");
    });
    report_allocs("lk_track", || {
        black_box(calc_optical_flow_pyr_lk_u8(&prev, &next, &pts, None, &params).expect("lk"));
    });
    report_allocs("orb_detect_and_extract_u8", || {
        black_box(detector.detect_and_extract_u8(&fx.left0).expect("orb"));
    });
    report_allocs("orb_build_pyramid_u8_into", || {
        detector
            .build_pyramid_u8_into(&fx.left0, &mut orb_levels)
            .expect("pyramid")
    });

    let mut g = c.benchmark_group("frontend");
    g.sample_size(60);

    g.bench_function("pyrdown_u8_l0_l1", |b| {
        b.iter(|| pyrdown_u8(black_box(&fx.left0), &mut half_u8).expect("pyrdown"))
    });
    g.bench_function("pyrdown_separable_f32_l0_l1", |b| {
        b.iter(|| pyrdown_separable_f32(black_box(&prev_f32), &mut half_f32).expect("pyrdown"))
    });
    g.bench_function("scharr_separable_f32_l0", |b| {
        b.iter(|| {
            scharr_spatial_gradient_separable_f32(black_box(&prev_f32), &mut ix, &mut iy)
                .expect("scharr")
        })
    });
    g.bench_function("lk_pyramids_new", |b| {
        b.iter(|| {
            black_box(LkPyramidU8::new(&fx.left0, win, max_level, true).expect("lk"));
            black_box(LkPyramidU8::new(&fx.left1, win, max_level, false).expect("lk"));
        })
    });
    g.bench_function("lk_pyramids_rebuild", |b| {
        b.iter(|| {
            prev.rebuild(black_box(&fx.left0), win, max_level, true)
                .expect("lk");
            next.rebuild(black_box(&fx.left1), win, max_level, false)
                .expect("lk");
        })
    });
    g.bench_function(format!("lk_track_{}_pts", pts.len()), |b| {
        b.iter(|| {
            black_box(calc_optical_flow_pyr_lk_u8(&prev, &next, &pts, None, &params).expect("lk"))
        })
    });

    g.bench_function("orb_detect_and_extract_u8_1000", |b| {
        b.iter(|| black_box(detector.detect_and_extract_u8(&fx.left0).expect("orb")))
    });
    g.bench_function("orb_build_pyramid_u8_into", |b| {
        b.iter(|| {
            detector
                .build_pyramid_u8_into(&fx.left0, &mut orb_levels)
                .expect("pyramid")
        })
    });
    detector
        .build_pyramid_u8_into(&fx.left0, &mut orb_levels)
        .expect("pyramid");
    let orb_pyr: Vec<&Image<u8, 1>> = std::iter::once(&fx.left0).chain(&orb_levels).collect();
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
                    calc_optical_flow_pyr_lk_u8(&prev, &next, &subset, None, &params).expect("lk"),
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
