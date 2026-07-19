//! GPU image-processing benchmark with H2D / kernel / D2H transfer breakdown.
//!
//! Covers resize (bilinear, nearest, bicubic, lanczos) and warp (affine,
//! perspective) at 1920×1080 and 3840×2160, comparing GPU round-trip timing
//! against the kornia CPU baseline.
//!
//! # Timing methodology
//!
//! Three CUDA events surround `memcpy_htod`, the kernel launch, and
//! `memcpy_dtoh` on the same stream; `event.elapsed_ms(end)` gives each
//! segment's hardware time after `stream.synchronize()`.  Events are created
//! once per case and reused across ITERS to avoid allocation overhead.
//!
//! # Output
//!
//! Prints a GitHub-flavoured Markdown table to stdout.  Redirect or append:
//!
//! ```text
//! cargo run --example bench_cuda_imgproc --features cuda --release \
//!     >> benchmarks.md
//! ```
//!
//! # Build
//!
//! ```text
//! cargo build --example bench_cuda_imgproc --features cuda --release
//! ./target/release/examples/bench_cuda_imgproc
//! ```

use std::{sync::Arc, time::Instant};

use cudarc::driver::{sys::CUevent_flags, CudaContext, CudaSlice, CudaStream};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::{
    cuda::{
        resize::{
            launch_resize_bicubic_cuda, launch_resize_bilinear_downscale_cuda,
            launch_resize_lanczos_cuda, launch_resize_nearest_downscale_cuda, PixelMapping,
        },
        warp_affine::launch_warp_affine_bilinear_cuda,
        warp_perspective::launch_warp_perspective_bilinear_cuda,
    },
    interpolation::InterpolationMode,
    resize::resize,
    warp::{get_rotation_matrix2d, warp_affine, warp_perspective},
};

const WARMUP: u32 = 30;
const ITERS: u32 = 100;
const NC: usize = 3; // RGB f32

// ── result type ───────────────────────────────────────────────────────────────

struct SegmentTimes {
    h2d_ms: f64,
    kernel_ms: f64,
    d2h_ms: f64,
}

impl SegmentTimes {
    fn total_ms(&self) -> f64 {
        self.h2d_ms + self.kernel_ms + self.d2h_ms
    }
}

// ── core benchmark helper ─────────────────────────────────────────────────────

/// Benchmark one operation with CUDA-event-based H2D / kernel / D2H breakdown.
///
/// `launch(src, dst)` receives pre-allocated device slices and must enqueue
/// the kernel onto `stream`.  `bench_segments` owns the H2D and D2H copies so
/// the closure does not need to capture `src_dev`.
fn bench_segments(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src_host: &[f32],
    dst_host: &mut [f32],
    src_dev: &mut CudaSlice<f32>,
    dst_dev: &mut CudaSlice<f32>,
    mut launch: impl FnMut(&CudaSlice<f32>, &mut CudaSlice<f32>),
) -> SegmentTimes {
    let make_ev = || {
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("create timing event")
    };
    let ev0 = make_ev(); // before H2D
    let ev1 = make_ev(); // after H2D / before kernel
    let ev2 = make_ev(); // after kernel / before D2H
    let ev3 = make_ev(); // after D2H

    // Warmup — no timing.
    for _ in 0..WARMUP {
        stream.memcpy_htod(src_host, src_dev).expect("warmup H→D");
        launch(src_dev, dst_dev);
        stream.synchronize().expect("warmup sync");
    }

    let mut h2d_sum = 0.0_f64;
    let mut k_sum = 0.0_f64;
    let mut d2h_sum = 0.0_f64;

    for _ in 0..ITERS {
        ev0.record(stream).expect("record ev0");
        stream.memcpy_htod(src_host, src_dev).expect("H→D");
        ev1.record(stream).expect("record ev1");
        launch(src_dev, dst_dev);
        ev2.record(stream).expect("record ev2");
        stream
            .memcpy_dtoh(dst_dev as &CudaSlice<f32>, dst_host)
            .expect("D→H");
        ev3.record(stream).expect("record ev3");
        stream.synchronize().expect("sync");

        h2d_sum += ev0.elapsed_ms(&ev1).expect("h2d elapsed") as f64;
        k_sum += ev1.elapsed_ms(&ev2).expect("kernel elapsed") as f64;
        d2h_sum += ev2.elapsed_ms(&ev3).expect("d2h elapsed") as f64;
    }

    SegmentTimes {
        h2d_ms: h2d_sum / ITERS as f64,
        kernel_ms: k_sum / ITERS as f64,
        d2h_ms: d2h_sum / ITERS as f64,
    }
}

// ── CPU baselines ─────────────────────────────────────────────────────────────

fn cpu_resize_ms(sw: u32, sh: u32, dw: u32, dh: u32, mode: InterpolationMode) -> f64 {
    let n = sw as usize * sh as usize * NC;
    let src = Image::<f32, 3>::new(
        ImageSize {
            width: sw as usize,
            height: sh as usize,
        },
        (0..n).map(|i| i as f32 / (n - 1) as f32).collect(),
    )
    .expect("src");
    let mut dst = Image::<f32, 3>::from_size_val(
        ImageSize {
            width: dw as usize,
            height: dh as usize,
        },
        0.0,
    )
    .expect("dst");
    for _ in 0..5 {
        resize(&src, &mut dst, mode).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        resize(&src, &mut dst, mode).expect("resize");
        std::hint::black_box(dst.as_slice());
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn cpu_warp_affine_ms(w: u32, h: u32, m: &[f32; 6]) -> f64 {
    let n = w as usize * h as usize * NC;
    let src = Image::<f32, 3>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        (0..n).map(|i| i as f32 / (n - 1) as f32).collect(),
    )
    .expect("src");
    let mut dst = Image::<f32, 3>::from_size_val(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        0.0,
    )
    .expect("dst");
    for _ in 0..5 {
        warp_affine(&src, &mut dst, m, InterpolationMode::Bilinear).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        warp_affine(&src, &mut dst, m, InterpolationMode::Bilinear).expect("warp_affine");
        std::hint::black_box(dst.as_slice());
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn cpu_warp_perspective_ms(w: u32, h: u32, hmat: &[f32; 9]) -> f64 {
    let n = w as usize * h as usize * NC;
    let src = Image::<f32, 3>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        (0..n).map(|i| i as f32 / (n - 1) as f32).collect(),
    )
    .expect("src");
    let mut dst = Image::<f32, 3>::from_size_val(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        0.0,
    )
    .expect("dst");
    for _ in 0..5 {
        warp_perspective(&src, &mut dst, hmat, InterpolationMode::Bilinear).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        warp_perspective(&src, &mut dst, hmat, InterpolationMode::Bilinear)
            .expect("warp_perspective");
        std::hint::black_box(dst.as_slice());
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

// ── geometry helpers ──────────────────────────────────────────────────────────

fn rotation_2x3(w: u32, h: u32, deg: f32) -> [f32; 6] {
    get_rotation_matrix2d((w as f32 / 2.0, h as f32 / 2.0), deg, 1.0)
}

fn rotation_3x3(w: u32, h: u32, deg: f32) -> [f32; 9] {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let (sin_a, cos_a) = deg.to_radians().sin_cos();
    [
        cos_a,
        sin_a,
        (1.0 - cos_a) * cx - sin_a * cy,
        -sin_a,
        cos_a,
        sin_a * cx + (1.0 - cos_a) * cy,
        0.0,
        0.0,
        1.0,
    ]
}

// ── table output ──────────────────────────────────────────────────────────────

fn print_header() {
    println!(
        "| {:<31} | {:<8} | {:<13} | {:>8} | {:>7} | {:>9} | {:>7} | {:>13} | {:>16} | {:>19} |",
        "Operation",
        "Interp",
        "Resolution",
        "CPU (ms)",
        "H2D (ms)",
        "Kernel (ms)",
        "D2H (ms)",
        "Total GPU (ms)",
        "Speedup (kernel)",
        "Speedup (roundtrip)"
    );
    println!(
        "| {:-<31} | {:-<8} | {:-<13} | {:-<8} | {:-<7} | {:-<9} | {:-<7} | {:-<13} | {:-<16} | {:-<19} |",
        "", "", "", "", "", "", "", "", "", ""
    );
}

fn print_row(op: &str, interp: &str, res: &str, cpu_ms: f64, seg: &SegmentTimes) {
    let total = seg.total_ms();
    println!(
        "| {:<31} | {:<8} | {:<13} | {:>8.2} | {:>7.2} | {:>9.2} | {:>7.2} | {:>13.2} | {:>15.1}x | {:>18.1}x |",
        op, interp, res, cpu_ms,
        seg.h2d_ms, seg.kernel_ms, seg.d2h_ms, total,
        cpu_ms / seg.kernel_ms, cpu_ms / total,
    );
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let ctx = Arc::new(CudaContext::new(0).expect("CUDA device 0"));
    let stream = ctx.default_stream();

    let gpu_name = ctx.name().unwrap_or_else(|_| "unknown GPU".into());
    println!("<!-- bench_cuda_imgproc  GPU: {gpu_name}  warmup={WARMUP}  iters={ITERS} -->");
    println!();
    print_header();

    // ── resize ────────────────────────────────────────────────────────────────

    struct ResizeCase {
        interp: &'static str,
        sw: u32,
        sh: u32,
        dw: u32,
        dh: u32,
        cpu_mode: InterpolationMode,
    }

    let resize_cases = [
        ResizeCase {
            interp: "bilinear",
            sw: 1920,
            sh: 1080,
            dw: 960,
            dh: 540,
            cpu_mode: InterpolationMode::Bilinear,
        },
        ResizeCase {
            interp: "bilinear",
            sw: 3840,
            sh: 2160,
            dw: 1920,
            dh: 1080,
            cpu_mode: InterpolationMode::Bilinear,
        },
        ResizeCase {
            interp: "nearest",
            sw: 1920,
            sh: 1080,
            dw: 960,
            dh: 540,
            cpu_mode: InterpolationMode::Nearest,
        },
        ResizeCase {
            interp: "nearest",
            sw: 3840,
            sh: 2160,
            dw: 1920,
            dh: 1080,
            cpu_mode: InterpolationMode::Nearest,
        },
        ResizeCase {
            interp: "bicubic",
            sw: 1920,
            sh: 1080,
            dw: 960,
            dh: 540,
            cpu_mode: InterpolationMode::Bicubic,
        },
        ResizeCase {
            interp: "bicubic",
            sw: 3840,
            sh: 2160,
            dw: 1920,
            dh: 1080,
            cpu_mode: InterpolationMode::Bicubic,
        },
        ResizeCase {
            interp: "lanczos",
            sw: 1920,
            sh: 1080,
            dw: 960,
            dh: 540,
            cpu_mode: InterpolationMode::Bilinear,
        },
        ResizeCase {
            interp: "lanczos",
            sw: 3840,
            sh: 2160,
            dw: 1920,
            dh: 1080,
            cpu_mode: InterpolationMode::Bilinear,
        },
    ];

    for c in &resize_cases {
        let n_src = c.sw as usize * c.sh as usize * NC;
        let n_dst = c.dw as usize * c.dh as usize * NC;
        let src_host: Vec<f32> = (0..n_src).map(|i| i as f32 / (n_src - 1) as f32).collect();
        let mut dst_host = vec![0.0f32; n_dst];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<f32>(n_dst).expect("alloc dst");

        let (sw, sh, dw, dh, interp) = (c.sw, c.sh, c.dw, c.dh, c.interp);
        let ctx2 = ctx.clone();
        let stream2 = stream.clone();

        let cpu_ms = cpu_resize_ms(sw, sh, dw, dh, c.cpu_mode);
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| match interp {
                "nearest" => launch_resize_nearest_downscale_cuda(
                    &ctx2,
                    &stream2,
                    src,
                    dst,
                    sw,
                    sh,
                    dw,
                    dh,
                    PixelMapping::HalfPixel,
                    None,
                )
                .expect("nearest"),
                "bicubic" => launch_resize_bicubic_cuda(
                    &ctx2,
                    &stream2,
                    src,
                    dst,
                    sw,
                    sh,
                    dw,
                    dh,
                    PixelMapping::HalfPixel,
                    None,
                )
                .expect("bicubic"),
                "lanczos" => launch_resize_lanczos_cuda(
                    &ctx2,
                    &stream2,
                    src,
                    dst,
                    sw,
                    sh,
                    dw,
                    dh,
                    PixelMapping::HalfPixel,
                    None,
                )
                .expect("lanczos"),
                _ => launch_resize_bilinear_downscale_cuda(
                    &ctx2,
                    &stream2,
                    src,
                    dst,
                    sw,
                    sh,
                    dw,
                    dh,
                    PixelMapping::HalfPixel,
                    None,
                )
                .expect("bilinear"),
            },
        );

        let res = format!("{sw}×{sh}→{dw}×{dh}");
        print_row("resize", interp, &res, cpu_ms, &seg);
    }

    // ── warp affine ───────────────────────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n = sw as usize * sh as usize * NC;
        let src_host: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let mut dst_host = vec![0.0f32; n];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<f32>(n).expect("alloc dst");

        let m = rotation_2x3(sw, sh, 30.0);
        let ctx2 = ctx.clone();
        let stream2 = stream.clone();

        let cpu_ms = cpu_warp_affine_ms(sw, sh, &m);
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_warp_affine_bilinear_cuda(
                    &ctx2, &stream2, src, dst, sw, sh, sw, sh, &m, None,
                )
                .expect("warp_affine");
            },
        );

        print_row(
            "warp_affine (30° rot)",
            "bilinear",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }

    // ── warp perspective ──────────────────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n = sw as usize * sh as usize * NC;
        let src_host: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let mut dst_host = vec![0.0f32; n];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<f32>(n).expect("alloc dst");

        let hmat = rotation_3x3(sw, sh, 30.0);
        let ctx2 = ctx.clone();
        let stream2 = stream.clone();

        let cpu_ms = cpu_warp_perspective_ms(sw, sh, &hmat);
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_warp_perspective_bilinear_cuda(
                    &ctx2, &stream2, src, dst, sw, sh, sw, sh, &hmat, None,
                )
                .expect("warp_perspective");
            },
        );

        print_row(
            "warp_perspective (30° rot)",
            "bilinear",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }
}
