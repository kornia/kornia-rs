//! GPU image-processing benchmark with H2D / kernel / D2H transfer breakdown.
//!
//! Covers resize (f32, u8), warp (affine, perspective, u8), remap (f32, u8),
//! filters (Gaussian blur, Sobel), morphology (erode, dilate), and color
//! conversion at 1920×1080 and 3840×2160, comparing GPU round-trip timing
//! against the kornia CPU baseline.
//!
//! # Timing methodology
//!
//! Three CUDA events surround `memcpy_htod`, the kernel launch, and
//! `memcpy_dtoh` on the same stream; `event.elapsed_ms(end)` gives each
//! segment's hardware time after `stream.synchronize()`. Events are created
//! once per case and reused across ITERS to avoid allocation overhead.
//!
//! # Output
//!
//! Prints a GitHub-flavoured Markdown table to stdout. Redirect or append:
//!
//! ```text
//! cargo run --example bench_cuda_imgproc --features cuda --release \
//!     >> benchmarks.md
//! ```

use std::{any::Any, sync::Arc, time::Instant};

use cudarc::driver::{sys::CUevent_flags, CudaContext, CudaSlice, CudaStream, DeviceRepr};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::{
    color::{
        bgr_from_rgb, gray_from_rgb, gray_from_rgb_u8 as cpu_gray_from_rgb_u8, hls_from_rgb,
        hsv_from_rgb, rgb_from_gray, ycbcr_from_rgb,
    },
    cuda::{
        color::{
            gray::{launch_gray_from_rgb_f32, launch_gray_from_rgb_u8, launch_rgb_from_gray_u8},
            hsv_hls::{launch_hls_from_rgb_f32, launch_hsv_from_rgb_f32},
            swizzle::launch_bgr_from_rgb_u8,
            yuv::{launch_ycc_from_rgb_f32, launch_ycc_from_rgb_u8, ChromaOrder},
        },
        filter::{
            launch_binomial3_u8, launch_gradient_magnitude_f32, launch_separable_blur_u8q8,
            launch_separable_filter_f32,
        },
        morphology::{launch_morphology_u8_cuda, MorphBorder, MorphOp},
        remap::{
            launch_remap_bilinear_cuda, launch_remap_bilinear_u8_cuda, launch_remap_nearest_u8_cuda,
        },
        resize::{
            launch_resize_bicubic_cuda, launch_resize_bilinear_downscale_cuda,
            launch_resize_lanczos_cuda, launch_resize_nearest_downscale_cuda, PixelMapping,
        },
        resize_u8::{launch_resize_u8_bilinear_cuda, launch_resize_u8_nearest_cuda},
        warp_affine::launch_warp_affine_bilinear_cuda,
        warp_affine_u8::launch_warp_affine_u8_bilinear_cuda,
        warp_perspective::launch_warp_perspective_bilinear_cuda,
        warp_perspective_u8::launch_warp_perspective_u8_bilinear_cuda,
    },
    filter::{box_blur_u8, gaussian_blur, gaussian_blur_u8, kernels::gaussian_kernel_1d, sobel},
    interpolation::{remap, remap_u8, InterpolationMode},
    morphology::{
        dilate, erode,
        kernels::{Kernel, KernelShape},
    },
    padding::PaddingMode,
    resize::{bilinear_axis_lut, nearest_axis_lut, resize},
    warp::{
        get_rotation_matrix2d, warp_affine, warp_affine_u8, warp_perspective, warp_perspective_u8,
    },
};

use kornia_imgproc::cuda::filter::{launch_integral_image_cuda, launch_laplacian_u8_cuda};
use kornia_imgproc::filter::{integral_image_u8, laplacian_u8};

const WARMUP: u32 = 30;
const ITERS: u32 = 100;
const NC: usize = 3; // RGB

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
fn bench_segments<T: DeviceRepr + Any + Copy>(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src_host: &[T],
    dst_host: &mut [T],
    src_dev: &mut CudaSlice<T>,
    dst_dev: &mut CudaSlice<T>,
    mut launch: impl FnMut(&CudaSlice<T>, &mut CudaSlice<T>),
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
            .memcpy_dtoh(dst_dev as &CudaSlice<T>, dst_host)
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

fn cpu_warp_affine_u8_ms(w: u32, h: u32, m: &[f32; 6]) -> f64 {
    let n = w as usize * h as usize * NC;
    let src = Image::<u8, 3>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        (0..n).map(|i| (i % 256) as u8).collect(),
    )
    .expect("src");
    let mut dst = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        0,
    )
    .expect("dst");
    for _ in 0..5 {
        warp_affine_u8(&src, &mut dst, m).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        warp_affine_u8(&src, &mut dst, m).expect("warp_affine_u8");
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

fn cpu_warp_perspective_u8_ms(w: u32, h: u32, hmat: &[f32; 9]) -> f64 {
    let n = w as usize * h as usize * NC;
    let src = Image::<u8, 3>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        (0..n).map(|i| (i % 256) as u8).collect(),
    )
    .expect("src");
    let mut dst = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        0,
    )
    .expect("dst");
    for _ in 0..5 {
        warp_perspective_u8(&src, &mut dst, hmat).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        warp_perspective_u8(&src, &mut dst, hmat).expect("warp_perspective_u8");
        std::hint::black_box(dst.as_slice());
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn cpu_remap_ms(w: u32, h: u32, mode: InterpolationMode) -> f64 {
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
    let mx_data: Vec<f32> = (0..h).flat_map(|_| (0..w).map(|x| x as f32)).collect();
    let my_data: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |_| y as f32)).collect();
    let map_x = Image::<f32, 1>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        mx_data,
    )
    .expect("map_x");
    let map_y = Image::<f32, 1>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        my_data,
    )
    .expect("map_y");

    for _ in 0..5 {
        remap(&src, &mut dst, &map_x, &map_y, mode).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        remap(&src, &mut dst, &map_x, &map_y, mode).expect("remap");
        std::hint::black_box(dst.as_slice());
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn cpu_gaussian_blur_ms(w: u32, h: u32) -> f64 {
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
        gaussian_blur(&src, &mut dst, (5, 5), (1.5, 1.5)).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        gaussian_blur(&src, &mut dst, (5, 5), (1.5, 1.5)).expect("gaussian_blur");
        std::hint::black_box(dst.as_slice());
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn cpu_gaussian_blur_u8_ms(w: u32, h: u32) -> f64 {
    let n = w as usize * h as usize * NC;
    let src = Image::<u8, 3>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        (0..n).map(|i| (i % 256) as u8).collect(),
    )
    .expect("src");
    let mut dst = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        0,
    )
    .expect("dst");
    for _ in 0..5 {
        gaussian_blur_u8(&src, &mut dst, (3, 3), (1.0, 1.0)).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        gaussian_blur_u8(&src, &mut dst, (3, 3), (1.0, 1.0)).expect("gaussian_blur_u8");
        std::hint::black_box(dst.as_slice());
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn cpu_box_blur_u8_ms(w: u32, h: u32) -> f64 {
    let n = w as usize * h as usize * NC;
    let src = Image::<u8, 3>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        (0..n).map(|i| (i % 256) as u8).collect(),
    )
    .expect("src");
    let mut dst = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        0,
    )
    .expect("dst");
    for _ in 0..5 {
        box_blur_u8(&src, &mut dst, (3, 3)).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        box_blur_u8(&src, &mut dst, (3, 3)).expect("box_blur_u8");
        std::hint::black_box(dst.as_slice());
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn cpu_sobel_ms(w: u32, h: u32) -> f64 {
    let n = w as usize * h as usize * NC;
    let src = Image::<f32, 3>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        (0..n).map(|i| i as f32 / (n - 1) as f32).collect(),
    )
    .expect("src");
    let mut dst_x = Image::<f32, 3>::from_size_val(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        0.0,
    )
    .expect("dst_x");
    for _ in 0..5 {
        sobel(&src, &mut dst_x, 3).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        sobel(&src, &mut dst_x, 3).expect("sobel");
        std::hint::black_box(dst_x.as_slice());
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn cpu_erode_ms(w: u32, h: u32) -> f64 {
    let n = w as usize * h as usize * NC;
    let src = Image::<u8, 3>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        (0..n).map(|i| (i % 256) as u8).collect(),
    )
    .expect("src");
    let mut dst = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        0,
    )
    .expect("dst");
    let kernel = Kernel::new(KernelShape::Box { size: 3 });
    for _ in 0..5 {
        erode(&src, &mut dst, &kernel, PaddingMode::Constant, [0, 0, 0]).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        erode(&src, &mut dst, &kernel, PaddingMode::Constant, [0, 0, 0]).expect("erode");
        std::hint::black_box(dst.as_slice());
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn cpu_dilate_ms(w: u32, h: u32) -> f64 {
    let n = w as usize * h as usize * NC;
    let src = Image::<u8, 3>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        (0..n).map(|i| (i % 256) as u8).collect(),
    )
    .expect("src");
    let mut dst = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        0,
    )
    .expect("dst");
    let kernel = Kernel::new(KernelShape::Box { size: 3 });
    for _ in 0..5 {
        dilate(&src, &mut dst, &kernel, PaddingMode::Constant, [0, 0, 0]).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        dilate(&src, &mut dst, &kernel, PaddingMode::Constant, [0, 0, 0]).expect("dilate");
        std::hint::black_box(dst.as_slice());
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn cpu_laplacian_ms(w: u32, h: u32) -> f64 {
    let n = w as usize * h as usize;
    let src = Image::<u8, 1>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        (0..n).map(|i| (i % 256) as u8).collect(),
    )
    .expect("src");
    let mut dst = Image::<i16, 1>::from_size_val(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        0,
    )
    .expect("dst");
    for _ in 0..5 {
        laplacian_u8(&src, &mut dst).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        laplacian_u8(&src, &mut dst).expect("laplacian");
        std::hint::black_box(dst.as_slice());
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn cpu_integral_ms(w: u32, h: u32) -> f64 {
    let n = w as usize * h as usize;
    let src = Image::<u8, 1>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        (0..n).map(|i| (i % 256) as u8).collect(),
    )
    .expect("src");
    let mut dst = Image::<f32, 1>::from_size_val(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        0.0,
    )
    .expect("dst");
    for _ in 0..5 {
        integral_image_u8(&src, &mut dst).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        integral_image_u8(&src, &mut dst).expect("integral");
        std::hint::black_box(dst.as_slice());
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn cpu_gray_from_rgb_ms(w: u32, h: u32) -> f64 {
    let n = w as usize * h as usize * NC;
    let src = Image::<f32, 3>::new(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        (0..n).map(|i| i as f32 / (n - 1) as f32).collect(),
    )
    .expect("src");
    let mut dst = Image::<f32, 1>::from_size_val(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        0.0,
    )
    .expect("dst");
    for _ in 0..5 {
        gray_from_rgb(&src, &mut dst).expect("warmup");
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        gray_from_rgb(&src, &mut dst).expect("gray_from_rgb");
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

    // ── resize f32 ────────────────────────────────────────────────────────────

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
        print_row("resize (f32)", interp, &res, cpu_ms, &seg);
    }

    // ── resize u8 ─────────────────────────────────────────────────────────────

    for c in &[
        (1920u32, 1080u32, 960u32, 540u32, "bilinear"),
        (3840, 2160, 1920, 1080, "bilinear"),
        (1920, 1080, 960, 540, "nearest"),
        (3840, 2160, 1920, 1080, "nearest"),
    ] {
        let (sw, sh, dw, dh, interp) = *c;
        let n_src = sw as usize * sh as usize * NC;
        let n_dst = dw as usize * dh as usize * NC;
        let src_host: Vec<u8> = (0..n_src).map(|i| (i % 256) as u8).collect();
        let mut dst_host = vec![0u8; n_dst];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<u8>(n_dst).expect("alloc dst");

        let ctx2 = ctx.clone();
        let stream2 = stream.clone();
        let mode = if interp == "nearest" {
            InterpolationMode::Nearest
        } else {
            InterpolationMode::Bilinear
        };
        let cpu_ms = cpu_resize_ms(sw, sh, dw, dh, mode);

        let xmap = nearest_axis_lut(sw as usize, dw as usize);
        let ymap = nearest_axis_lut(sh as usize, dh as usize);
        let xmap_dev = stream.clone_htod(&xmap).expect("H→D xmap");
        let ymap_dev = stream.clone_htod(&ymap).expect("H→D ymap");

        let (xofs, xfx, _) = bilinear_axis_lut(sw as usize, dw as usize);
        let (yofs, yfy, _) = bilinear_axis_lut(sh as usize, dh as usize);
        let xofs_dev = stream.clone_htod(&xofs).expect("H→D xofs");
        let xfx_dev = stream.clone_htod(&xfx).expect("H→D xfx");
        let yofs_dev = stream.clone_htod(&yofs).expect("H→D yofs");
        let yfy_dev = stream.clone_htod(&yfy).expect("H→D yfy");

        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| match interp {
                "nearest" => launch_resize_u8_nearest_cuda(
                    &ctx2, &stream2, src, dst, sw, sh, dw, dh, NC as u32, &xmap_dev, &ymap_dev,
                    None,
                )
                .expect("resize_nearest_u8"),
                _ => launch_resize_u8_bilinear_cuda(
                    &ctx2, &stream2, src, dst, sw, sh, dw, dh, NC as u32, &xofs_dev, &xfx_dev,
                    &yofs_dev, &yfy_dev, None,
                )
                .expect("resize_bilinear_u8"),
            },
        );

        let res = format!("{sw}×{sh}→{dw}×{dh}");
        print_row("resize (u8)", interp, &res, cpu_ms, &seg);
    }

    // ── warp affine f32 ───────────────────────────────────────────────────────

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
            "warp_affine (30° rot, f32)",
            "bilinear",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }

    // ── warp affine u8 ────────────────────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n = sw as usize * sh as usize * NC;
        let src_host: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let mut dst_host = vec![0u8; n];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<u8>(n).expect("alloc dst");

        let m = rotation_2x3(sw, sh, 30.0);
        let ctx2 = ctx.clone();
        let stream2 = stream.clone();

        let cpu_ms = cpu_warp_affine_u8_ms(sw, sh, &m);
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_warp_affine_u8_bilinear_cuda(
                    &ctx2, &stream2, src, dst, &m, sw, sh, sw, sh, NC as u32, None,
                )
                .expect("warp_affine_u8");
            },
        );

        print_row(
            "warp_affine (30° rot, u8)",
            "bilinear",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }

    // ── warp perspective f32 ──────────────────────────────────────────────────

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
            "warp_perspective (30° rot, f32)",
            "bilinear",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }

    // ── warp perspective u8 ───────────────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n = sw as usize * sh as usize * NC;
        let src_host: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let mut dst_host = vec![0u8; n];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<u8>(n).expect("alloc dst");

        let hmat = rotation_3x3(sw, sh, 30.0);
        let ctx2 = ctx.clone();
        let stream2 = stream.clone();

        let cpu_ms = cpu_warp_perspective_u8_ms(sw, sh, &hmat);
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_warp_perspective_u8_bilinear_cuda(
                    &ctx2, &stream2, src, dst, &hmat, sw, sh, sw, sh, NC as u32, None,
                )
                .expect("warp_perspective_u8");
            },
        );

        print_row(
            "warp_perspective (30° rot, u8)",
            "bilinear",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }

    // ── remap f32 ─────────────────────────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n = sw as usize * sh as usize * NC;
        let src_host: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let mut dst_host = vec![0.0f32; n];
        let mx_host: Vec<f32> = (0..sh).flat_map(|_| (0..sw).map(|x| x as f32)).collect();
        let my_host: Vec<f32> = (0..sh)
            .flat_map(|y| (0..sw).map(move |_| y as f32))
            .collect();

        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<f32>(n).expect("alloc dst");
        let map_x_dev = stream.clone_htod(&mx_host).expect("H→D map_x");
        let map_y_dev = stream.clone_htod(&my_host).expect("H→D map_y");

        let ctx2 = ctx.clone();
        let stream2 = stream.clone();

        let cpu_ms = cpu_remap_ms(sw, sh, InterpolationMode::Bilinear);
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_remap_bilinear_cuda(
                    &ctx2, &stream2, src, &map_x_dev, &map_y_dev, dst, sw, sh, sw, sh, None,
                )
                .expect("remap_bilinear");
            },
        );

        print_row(
            "remap (f32)",
            "bilinear",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }

    // ── filters (gaussian_blur, sobel) ────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n = sw as usize * sh as usize * NC;
        let src_host: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let mut dst_host = vec![0.0f32; n];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<f32>(n).expect("alloc dst");

        let kx = gaussian_kernel_1d(5, 1.5);
        let ky = kx.clone();
        let kx_dev = stream.clone_htod(&kx).expect("H→D kx");
        let ky_dev = stream.clone_htod(&ky).expect("H→D ky");
        let mut scratch = stream.alloc_zeros::<f32>(n).expect("alloc scratch");

        let ctx2 = ctx.clone();
        let stream2 = stream.clone();

        let cpu_ms = cpu_gaussian_blur_ms(sw, sh);
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_separable_filter_f32(
                    &ctx2,
                    &stream2,
                    src,
                    dst,
                    &mut scratch,
                    &kx_dev,
                    5,
                    &ky_dev,
                    5,
                    sw,
                    sh,
                    NC as u32,
                )
                .expect("gaussian_blur");
            },
        );
        print_row(
            "gaussian_blur (5x5, f32)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );

        // Gaussian blur u8 (3x3 binomial fast path)
        let src_host_u8: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let mut dst_host_u8 = vec![0u8; n];
        let mut src_dev_u8 = stream.clone_htod(&src_host_u8).expect("H→D src");
        let mut dst_dev_u8 = stream.alloc_zeros::<u8>(n).expect("alloc dst");
        let mut scratch_u8 = stream.alloc_zeros::<u8>(n).expect("alloc scratch");

        let cpu_ms_u8 = cpu_gaussian_blur_u8_ms(sw, sh);
        let seg_u8 = bench_segments(
            &ctx,
            &stream,
            &src_host_u8,
            &mut dst_host_u8,
            &mut src_dev_u8,
            &mut dst_dev_u8,
            |src, dst| {
                launch_binomial3_u8(
                    &ctx2,
                    &stream2,
                    src,
                    dst,
                    &mut scratch_u8,
                    sw,
                    sh,
                    NC as u32,
                )
                .expect("gaussian_blur_u8 (binomial)");
            },
        );
        print_row(
            "gaussian_blur (3x3, u8)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_ms_u8,
            &seg_u8,
        );

        // Box blur u8 (3x3 general Q8 path)
        let k_box = vec![85u8, 86, 85]; // 256/3 quantized
        let k_box_dev = stream.clone_htod(&k_box).expect("H→D box kernel");
        let cpu_box_ms = cpu_box_blur_u8_ms(sw, sh);
        let seg_box = bench_segments(
            &ctx,
            &stream,
            &src_host_u8,
            &mut dst_host_u8,
            &mut src_dev_u8,
            &mut dst_dev_u8,
            |src, dst| {
                launch_separable_blur_u8q8(
                    &ctx2,
                    &stream2,
                    src,
                    dst,
                    &mut scratch_u8,
                    &k_box_dev,
                    3,
                    &k_box_dev,
                    3,
                    sw,
                    sh,
                    NC as u32,
                )
                .expect("box_blur_u8");
            },
        );
        print_row(
            "box_blur (3x3, u8)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_box_ms,
            &seg_box,
        );

        let (sobel_x, sobel_y) =
            kornia_imgproc::filter::kernels::sobel_kernel_1d(3).expect("sobel_kernel_1d");
        let sx_dev = stream.clone_htod(&sobel_x).expect("H→D sx");
        let sy_dev = stream.clone_htod(&sobel_y).expect("H→D sy");
        let mut gx_dev = stream.alloc_zeros::<f32>(n).expect("alloc gx");
        let mut gy_dev = stream.alloc_zeros::<f32>(n).expect("alloc gy");
        let cpu_sobel = cpu_sobel_ms(sw, sh);
        let seg_sobel = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_separable_filter_f32(
                    &ctx2,
                    &stream2,
                    src,
                    &mut gx_dev,
                    &mut scratch,
                    &sx_dev,
                    3,
                    &sy_dev,
                    3,
                    sw,
                    sh,
                    NC as u32,
                )
                .expect("sobel_gx");
                launch_separable_filter_f32(
                    &ctx2,
                    &stream2,
                    src,
                    &mut gy_dev,
                    &mut scratch,
                    &sy_dev,
                    3,
                    &sx_dev,
                    3,
                    sw,
                    sh,
                    NC as u32,
                )
                .expect("sobel_gy");
                launch_gradient_magnitude_f32(&ctx2, &stream2, &gx_dev, &gy_dev, dst, n)
                    .expect("gradient_magnitude");
            },
        );
        print_row(
            "sobel (3x3, f32)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_sobel,
            &seg_sobel,
        );

        let cpu_laplacian = cpu_laplacian_ms(sw, sh);
        let src_u8 = Image::<u8, 1>::new(
            ImageSize {
                width: sw as usize,
                height: sh as usize,
            },
            vec![0u8; (sw * sh) as usize],
        )
        .unwrap();
        let mut dst_i16 = Image::<i16, 1>::from_size_val(
            ImageSize {
                width: sw as usize,
                height: sh as usize,
            },
            0,
        )
        .unwrap();

        let make_ev = || {
            ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap()
        };
        let ev0 = make_ev();
        let ev1 = make_ev();
        let ev2 = make_ev();
        let ev3 = make_ev();

        let mut src_dev_u8 = src_u8.to_cuda(&stream).unwrap();
        let mut dst_dev_i16 = dst_i16.to_cuda(&stream).unwrap();

        for _ in 0..WARMUP {
            launch_laplacian_u8_cuda(&src_dev_u8, &mut dst_dev_i16, &stream).unwrap();
            stream.synchronize().unwrap();
        }

        let mut h2d_sum = 0.0_f64;
        let mut k_sum = 0.0_f64;
        let mut d2h_sum = 0.0_f64;
        for _ in 0..ITERS {
            ev0.record(&stream).unwrap();
            stream
                .memcpy_htod(src_u8.as_slice(), src_dev_u8.0.as_cudaslice_mut().unwrap())
                .unwrap();
            ev1.record(&stream).unwrap();
            launch_laplacian_u8_cuda(&src_dev_u8, &mut dst_dev_i16, &stream).unwrap();
            ev2.record(&stream).unwrap();
            stream
                .memcpy_dtoh(
                    dst_dev_i16.0.as_cudaslice().unwrap(),
                    dst_i16.as_slice_mut(),
                )
                .unwrap();
            ev3.record(&stream).unwrap();
            stream.synchronize().unwrap();
            h2d_sum += ev0.elapsed_ms(&ev1).unwrap() as f64;
            k_sum += ev1.elapsed_ms(&ev2).unwrap() as f64;
            d2h_sum += ev2.elapsed_ms(&ev3).unwrap() as f64;
        }

        let seg_laplacian = SegmentTimes {
            h2d_ms: h2d_sum / ITERS as f64,
            kernel_ms: k_sum / ITERS as f64,
            d2h_ms: d2h_sum / ITERS as f64,
        };

        print_row(
            "laplacian (3x3, u8)",
            "-",
            &format!("{sw}×{sh}"),
            cpu_laplacian,
            &seg_laplacian,
        );

        let cpu_integral = cpu_integral_ms(sw, sh);
        let mut dst_f32 = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: sw as usize,
                height: sh as usize,
            },
            0.0,
        )
        .unwrap();
        let mut dst_dev_f32 = dst_f32.to_cuda(&stream).unwrap();

        for _ in 0..WARMUP {
            launch_integral_image_cuda(&src_dev_u8, &mut dst_dev_f32, &stream).unwrap();
            stream.synchronize().unwrap();
        }

        let mut h2d_sum = 0.0_f64;
        let mut k_sum = 0.0_f64;
        let mut d2h_sum = 0.0_f64;
        for _ in 0..ITERS {
            ev0.record(&stream).unwrap();
            stream
                .memcpy_htod(src_u8.as_slice(), src_dev_u8.0.as_cudaslice_mut().unwrap())
                .unwrap();
            ev1.record(&stream).unwrap();
            launch_integral_image_cuda(&src_dev_u8, &mut dst_dev_f32, &stream).unwrap();
            ev2.record(&stream).unwrap();
            stream
                .memcpy_dtoh(
                    dst_dev_f32.0.as_cudaslice().unwrap(),
                    dst_f32.as_slice_mut(),
                )
                .unwrap();
            ev3.record(&stream).unwrap();
            stream.synchronize().unwrap();
            h2d_sum += ev0.elapsed_ms(&ev1).unwrap() as f64;
            k_sum += ev1.elapsed_ms(&ev2).unwrap() as f64;
            d2h_sum += ev2.elapsed_ms(&ev3).unwrap() as f64;
        }

        let seg_integral = SegmentTimes {
            h2d_ms: h2d_sum / ITERS as f64,
            kernel_ms: k_sum / ITERS as f64,
            d2h_ms: d2h_sum / ITERS as f64,
        };

        print_row(
            "integral (u8)",
            "-",
            &format!("{sw}×{sh}"),
            cpu_integral,
            &seg_integral,
        );
    }

    // ── morphology (erode, dilate u8) ─────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n = sw as usize * sh as usize * NC;
        let src_host: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let mut dst_host = vec![0u8; n];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<u8>(n).expect("alloc dst");

        let ctx2 = ctx.clone();
        let stream2 = stream.clone();

        let taps = vec![
            -1i32, -1, -1, 0, -1, 1, 0, -1, 0, 0, 0, 1, 1, -1, 1, 0, 1, 1,
        ];
        let cval_dev = stream.clone_htod(&vec![0u8; NC]).expect("H→D cval");

        let cpu_ms = cpu_erode_ms(sw, sh);
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_morphology_u8_cuda(
                    &ctx2,
                    &stream2,
                    src,
                    dst,
                    sw,
                    sh,
                    NC as u32,
                    &taps,
                    &cval_dev,
                    MorphOp::Erode,
                    MorphBorder::Replicate,
                    None,
                )
                .expect("erode");
            },
        );
        print_row(
            "erode (3x3, u8)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );

        let cpu_dilate = cpu_dilate_ms(sw, sh);
        let seg_dilate = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_morphology_u8_cuda(
                    &ctx2,
                    &stream2,
                    src,
                    dst,
                    sw,
                    sh,
                    NC as u32,
                    &taps,
                    &cval_dev,
                    MorphOp::Dilate,
                    MorphBorder::Replicate,
                    None,
                )
                .expect("dilate");
            },
        );
        print_row(
            "dilate (3x3, u8)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_dilate,
            &seg_dilate,
        );
    }

    // ── color (gray_from_rgb) ─────────────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n_src = sw as usize * sh as usize * NC;
        let n_dst = sw as usize * sh as usize;
        let src_host: Vec<f32> = (0..n_src).map(|i| i as f32 / (n_src - 1) as f32).collect();
        let mut dst_host = vec![0.0f32; n_dst];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<f32>(n_dst).expect("alloc dst");

        let stream2 = stream.clone();

        let cpu_ms = cpu_gray_from_rgb_ms(sw, sh);
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_gray_from_rgb_f32(&stream2, src, dst, sw as usize * sh as usize)
                    .expect("gray_from_rgb");
            },
        );
        print_row(
            "gray_from_rgb (f32)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }

    // ── remap u8 (bilinear + nearest) ────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n = sw as usize * sh as usize * NC;
        let src_host: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let mut dst_host = vec![0u8; n];
        let mx_host: Vec<f32> = (0..sh).flat_map(|_| (0..sw).map(|x| x as f32)).collect();
        let my_host: Vec<f32> = (0..sh)
            .flat_map(|y| (0..sw).map(move |_| y as f32))
            .collect();

        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<u8>(n).expect("alloc dst");
        let map_x_dev = stream.clone_htod(&mx_host).expect("H→D map_x");
        let map_y_dev = stream.clone_htod(&my_host).expect("H→D map_y");

        let ctx2 = ctx.clone();
        let stream2 = stream.clone();

        // CPU baseline: remap_u8
        let cpu_ms = {
            let src_img = Image::<u8, 3>::new(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                src_host.clone(),
            )
            .expect("src");
            let mut dst_img = Image::<u8, 3>::from_size_val(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                0,
            )
            .expect("dst");
            let map_x_img = Image::<f32, 1>::new(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                mx_host.clone(),
            )
            .expect("map_x");
            let map_y_img = Image::<f32, 1>::new(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                my_host.clone(),
            )
            .expect("map_y");
            for _ in 0..5 {
                remap_u8(
                    &src_img,
                    &mut dst_img,
                    &map_x_img,
                    &map_y_img,
                    InterpolationMode::Bilinear,
                )
                .expect("warmup");
            }
            let t = std::time::Instant::now();
            for _ in 0..ITERS {
                remap_u8(
                    &src_img,
                    &mut dst_img,
                    &map_x_img,
                    &map_y_img,
                    InterpolationMode::Bilinear,
                )
                .expect("remap_u8");
                std::hint::black_box(dst_img.as_slice());
            }
            t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
        };

        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_remap_bilinear_u8_cuda(
                    &ctx2, &stream2, src, &map_x_dev, &map_y_dev, dst, sw, sh, sw, sh, NC as u32,
                    None,
                )
                .expect("remap_bilinear_u8");
            },
        );
        print_row(
            "remap (u8)",
            "bilinear",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );

        let seg_nn = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_remap_nearest_u8_cuda(
                    &ctx2, &stream2, src, &map_x_dev, &map_y_dev, dst, sw, sh, sw, sh, NC as u32,
                    None,
                )
                .expect("remap_nearest_u8");
            },
        );
        print_row(
            "remap (u8)",
            "nearest",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg_nn,
        );
    }

    // ── color: gray_from_rgb u8 ───────────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n_src = sw as usize * sh as usize * NC;
        let n_dst = sw as usize * sh as usize;
        let src_host: Vec<u8> = (0..n_src).map(|i| (i % 256) as u8).collect();
        let mut dst_host = vec![0u8; n_dst];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<u8>(n_dst).expect("alloc dst");
        let stream2 = stream.clone();

        let cpu_ms = {
            let src_img = Image::<u8, 3>::new(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                src_host.clone(),
            )
            .expect("src");
            let mut dst_img = Image::<u8, 1>::from_size_val(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                0,
            )
            .expect("dst");
            for _ in 0..5 {
                cpu_gray_from_rgb_u8(&src_img, &mut dst_img).expect("warmup");
            }
            let t = std::time::Instant::now();
            for _ in 0..ITERS {
                cpu_gray_from_rgb_u8(&src_img, &mut dst_img).expect("cpu_gray_u8");
                std::hint::black_box(dst_img.as_slice());
            }
            t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
        };
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_gray_from_rgb_u8(&stream2, src, dst, n_dst).expect("gray_from_rgb_u8");
            },
        );
        print_row(
            "gray_from_rgb (u8)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }

    // ── color: rgb_from_gray u8 ───────────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n_src = sw as usize * sh as usize; // 1-ch gray
        let n_dst = sw as usize * sh as usize * NC; // 3-ch RGB
        let src_host: Vec<u8> = (0..n_src).map(|i| (i % 256) as u8).collect();
        let mut dst_host = vec![0u8; n_dst];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<u8>(n_dst).expect("alloc dst");
        let stream2 = stream.clone();

        // No dedicated CPU function for rgb_from_gray; measure a trivial broadcast
        // to give a relative speedup figure.
        let cpu_ms = {
            let src_img = Image::<u8, 1>::new(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                src_host.clone(),
            )
            .expect("src");
            let mut dst_img = Image::<u8, 3>::from_size_val(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                0,
            )
            .expect("dst");
            for _ in 0..5 {
                rgb_from_gray(&src_img, &mut dst_img).expect("warmup");
            }
            let t = std::time::Instant::now();
            for _ in 0..ITERS {
                rgb_from_gray(&src_img, &mut dst_img).expect("cpu_rgb_from_gray");
                std::hint::black_box(dst_img.as_slice());
            }
            t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
        };
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_rgb_from_gray_u8(&stream2, src, dst, n_src).expect("rgb_from_gray_u8");
            },
        );
        print_row(
            "rgb_from_gray (u8)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }

    // ── color: HSV ↔ RGB (f32) ────────────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n = sw as usize * sh as usize;
        let n_elems = n * NC;
        let src_host: Vec<f32> = (0..n_elems).map(|i| (i % 256) as f32).collect();
        let mut dst_host = vec![0.0f32; n_elems];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<f32>(n_elems).expect("alloc dst");
        let stream2 = stream.clone();

        let cpu_ms = {
            let src_img = Image::<f32, 3>::new(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                src_host.clone(),
            )
            .expect("src");
            let mut dst_img = Image::<f32, 3>::from_size_val(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                0.0,
            )
            .expect("dst");
            for _ in 0..5 {
                hsv_from_rgb(&src_img, &mut dst_img).expect("warmup");
            }
            let t = std::time::Instant::now();
            for _ in 0..ITERS {
                hsv_from_rgb(&src_img, &mut dst_img).expect("cpu_hsv");
                std::hint::black_box(dst_img.as_slice());
            }
            t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
        };
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_hsv_from_rgb_f32(&stream2, src, dst, n).expect("hsv_from_rgb_f32");
            },
        );
        print_row(
            "hsv_from_rgb (f32)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }

    // ── color: HLS ↔ RGB (f32) ────────────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n = sw as usize * sh as usize;
        let n_elems = n * NC;
        let src_host: Vec<f32> = (0..n_elems).map(|i| (i % 256) as f32).collect();
        let mut dst_host = vec![0.0f32; n_elems];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<f32>(n_elems).expect("alloc dst");
        let stream2 = stream.clone();

        let cpu_ms = {
            let src_img = Image::<f32, 3>::new(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                src_host.clone(),
            )
            .expect("src");
            let mut dst_img = Image::<f32, 3>::from_size_val(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                0.0,
            )
            .expect("dst");
            for _ in 0..5 {
                hls_from_rgb(&src_img, &mut dst_img).expect("warmup");
            }
            let t = std::time::Instant::now();
            for _ in 0..ITERS {
                hls_from_rgb(&src_img, &mut dst_img).expect("cpu_hls");
                std::hint::black_box(dst_img.as_slice());
            }
            t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
        };
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_hls_from_rgb_f32(&stream2, src, dst, n).expect("hls_from_rgb_f32");
            },
        );
        print_row(
            "hls_from_rgb (f32)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }

    // ── color: ycc_from_rgb (u8) ──────────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n = sw as usize * sh as usize;
        let n_elems = n * NC;
        let src_host: Vec<u8> = (0..n_elems).map(|i| (i % 256) as u8).collect();
        let mut dst_host = vec![0u8; n_elems];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<u8>(n_elems).expect("alloc dst");
        let stream2 = stream.clone();

        let cpu_ms = {
            let src_img = Image::<u8, 3>::new(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                src_host.clone(),
            )
            .expect("src");
            let mut dst_img = Image::<u8, 3>::from_size_val(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                0,
            )
            .expect("dst");
            for _ in 0..5 {
                ycbcr_from_rgb(&src_img, &mut dst_img).expect("warmup");
            }
            let t = std::time::Instant::now();
            for _ in 0..ITERS {
                ycbcr_from_rgb(&src_img, &mut dst_img).expect("cpu_ycbcr");
                std::hint::black_box(dst_img.as_slice());
            }
            t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
        };
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_ycc_from_rgb_u8(&stream2, src, dst, n, ChromaOrder::YCrCb)
                    .expect("ycc_from_rgb_u8");
            },
        );
        print_row(
            "ycc_from_rgb (u8)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }

    // ── color: ycc_from_rgb (f32) ─────────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n = sw as usize * sh as usize;
        let n_elems = n * NC;
        let src_host: Vec<f32> = (0..n_elems).map(|i| (i % 256) as f32).collect();
        let mut dst_host = vec![0.0f32; n_elems];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<f32>(n_elems).expect("alloc dst");
        let stream2 = stream.clone();

        let cpu_ms = {
            let src_img = Image::<f32, 3>::new(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                src_host.clone(),
            )
            .expect("src");
            let mut dst_img = Image::<f32, 3>::from_size_val(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                0.0,
            )
            .expect("dst");
            for _ in 0..5 {
                ycbcr_from_rgb(&src_img, &mut dst_img).expect("warmup");
            }
            let t = std::time::Instant::now();
            for _ in 0..ITERS {
                ycbcr_from_rgb(&src_img, &mut dst_img).expect("cpu_ycbcr_f32");
                std::hint::black_box(dst_img.as_slice());
            }
            t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
        };
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_ycc_from_rgb_f32(&stream2, src, dst, n, ChromaOrder::YCrCb)
                    .expect("ycc_from_rgb_f32");
            },
        );
        print_row(
            "ycc_from_rgb (f32)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }

    // ── color: bgr_from_rgb (u8) ──────────────────────────────────────────────

    for &(sw, sh) in &[(1920u32, 1080u32), (3840, 2160)] {
        let n = sw as usize * sh as usize;
        let n_elems = n * NC;
        let src_host: Vec<u8> = (0..n_elems).map(|i| (i % 256) as u8).collect();
        let mut dst_host = vec![0u8; n_elems];
        let mut src_dev = stream.clone_htod(&src_host).expect("H→D src");
        let mut dst_dev = stream.alloc_zeros::<u8>(n_elems).expect("alloc dst");
        let stream2 = stream.clone();

        let cpu_ms = {
            let src_img = Image::<u8, 3>::new(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                src_host.clone(),
            )
            .expect("src");
            let mut dst_img = Image::<u8, 3>::from_size_val(
                ImageSize {
                    width: sw as usize,
                    height: sh as usize,
                },
                0,
            )
            .expect("dst");
            for _ in 0..5 {
                bgr_from_rgb(&src_img, &mut dst_img).expect("warmup");
            }
            let t = std::time::Instant::now();
            for _ in 0..ITERS {
                bgr_from_rgb(&src_img, &mut dst_img).expect("cpu_bgr");
                std::hint::black_box(dst_img.as_slice());
            }
            t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
        };
        let seg = bench_segments(
            &ctx,
            &stream,
            &src_host,
            &mut dst_host,
            &mut src_dev,
            &mut dst_dev,
            |src, dst| {
                launch_bgr_from_rgb_u8(&stream2, src, dst, n).expect("bgr_from_rgb_u8");
            },
        );
        print_row(
            "bgr_from_rgb (u8)",
            "n/a",
            &format!("{sw}×{sh}"),
            cpu_ms,
            &seg,
        );
    }
}
