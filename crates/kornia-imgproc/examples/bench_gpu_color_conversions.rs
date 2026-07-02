//! Benchmark: kornia CPU (NEON) vs kornia CUDA color conversions.
//!
//! Three numbers per (op, size):
//! - **cpu**: the production CPU path (NEON on aarch64), wall clock per call.
//! - **cuda-kernel**: device-resident src/dst, warmed up, batches of launches
//!   bounded by a stream sync, divided by the batch size. Source rotates over
//!   `N_BUFFS` buffers to defeat L2 so big sizes reflect DRAM bandwidth.
//! - **cuda-e2e**: upload (H2D) + kernel + download (D2H) + sync per call —
//!   what a host-resident caller actually pays. On Jetson the copies are real
//!   (`clone_htod` copies even on physically-unified memory).
//!
//! Output: human table, or `--json` for one JSON object per row (consumed by
//! kornia-py/benchmarks/report_color_bench.py).
//!
//! ```text
//! cargo run --example bench_gpu_color_conversions --features gpu-cuda --release [-- --json]
//! ```
#![cfg(feature = "gpu-cuda")]

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_image::color_spaces::BayerPattern;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::color::{self, ColormapType};
use kornia_imgproc::gpu::color_cuda::{bayer, gray, hsv_hls, misc, swizzle, video, yuv};

const SIZES: &[(usize, usize)] = &[(640, 480), (1280, 720), (1920, 1080), (3840, 2160)];
const N_BUFFS: usize = 8;
const KERNEL_WARMUP: usize = 30;
const KERNEL_BATCHES: usize = 10;
const KERNEL_PER_BATCH: usize = 20;
const E2E_ITERS: usize = 15;

/// Timing summary for one (op, size, variant) cell.
struct BenchStats {
    min_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    gbps: f64,
}

fn stats_from(mut samples_ms: Vec<f64>, bytes_moved: usize) -> BenchStats {
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_ms = samples_ms[0];
    let p50_ms = samples_ms[samples_ms.len() / 2];
    let p95_ms = samples_ms[((samples_ms.len() as f64 * 0.95) as usize).min(samples_ms.len() - 1)];
    // Bandwidth from the fastest sample (closest to hardware capability).
    let gbps = bytes_moved as f64 / (min_ms * 1e-3) / 1e9;
    BenchStats {
        min_ms,
        p50_ms,
        p95_ms,
        gbps,
    }
}

/// Deterministic byte pattern (LCG, seed fixed) shared by all variants.
fn pattern_u8(len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    let mut state = 0x1234_5678u32;
    while v.len() < len {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        v.push((state >> 24) as u8);
    }
    v
}

fn pattern_f32(len: usize) -> Vec<f32> {
    pattern_u8(len).into_iter().map(|b| b as f32).collect()
}

fn time_each<F: FnMut()>(iters: usize, mut f: F) -> Vec<f64> {
    let mut out = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        out.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    out
}

/// One benchmarked conversion: closures own their buffers.
struct OpBench<'a> {
    name: &'static str,
    /// src + dst bytes for one image (for GB/s).
    bytes_moved: usize,
    cpu: Box<dyn FnMut() + 'a>,
    /// Enqueue ONE kernel launch using rotating source buffer `i`.
    gpu_launch: Box<dyn FnMut(usize) + 'a>,
    /// One full upload + kernel + download + sync.
    gpu_e2e: Box<dyn FnMut() + 'a>,
}

fn run_op(op: &mut OpBench, stream: &Arc<CudaStream>, width: usize, height: usize, json: bool) {
    let cpu_iters = if width >= 3840 { 10 } else { 25 };
    let cpu = stats_from(time_each(cpu_iters, &mut op.cpu), op.bytes_moved);

    // Kernel-only: warmup, then batches bounded by a sync.
    for i in 0..KERNEL_WARMUP {
        (op.gpu_launch)(i % N_BUFFS);
    }
    stream.synchronize().unwrap();
    let mut batch_avgs = Vec::with_capacity(KERNEL_BATCHES);
    let mut launch_idx = 0usize;
    for _ in 0..KERNEL_BATCHES {
        let t0 = Instant::now();
        for _ in 0..KERNEL_PER_BATCH {
            (op.gpu_launch)(launch_idx % N_BUFFS);
            launch_idx += 1;
        }
        stream.synchronize().unwrap();
        batch_avgs.push(t0.elapsed().as_secs_f64() * 1e3 / KERNEL_PER_BATCH as f64);
    }
    let kernel = stats_from(batch_avgs, op.bytes_moved);

    // End-to-end.
    (op.gpu_e2e)(); // warm (first NVRTC compile already amortized by kernel pass)
    let e2e = stats_from(time_each(E2E_ITERS, &mut op.gpu_e2e), op.bytes_moved);

    for (variant, s) in [
        ("kornia-cpu", &cpu),
        ("kornia-cuda-kernel", &kernel),
        ("kornia-cuda-e2e", &e2e),
    ] {
        if json {
            println!(
                "{{\"op\":\"{}\",\"width\":{},\"height\":{},\"variant\":\"{}\",\"min_ms\":{:.6},\"p50_ms\":{:.6},\"p95_ms\":{:.6},\"gbps\":{:.3}}}",
                op.name, width, height, variant, s.min_ms, s.p50_ms, s.p95_ms, s.gbps
            );
        } else {
            println!(
                "{:>26} {:>10} {:>20} min {:>9.4} ms  p50 {:>9.4} ms  p95 {:>9.4} ms  {:>7.1} GB/s",
                op.name,
                format!("{width}x{height}"),
                variant,
                s.min_ms,
                s.p50_ms,
                s.p95_ms,
                s.gbps
            );
        }
    }
}

/// Build an elementwise-u8 OpBench: `cin` src channels → `cout` dst channels.
#[allow(clippy::too_many_arguments)]
fn elementwise_u8<'a>(
    name: &'static str,
    stream: &'a Arc<CudaStream>,
    npixels: usize,
    cin: usize,
    cout: usize,
    cpu: Box<dyn FnMut() + 'a>,
    launch: impl Fn(&Arc<CudaStream>, &CudaSlice<u8>, &mut CudaSlice<u8>, usize) + 'a,
) -> OpBench<'a> {
    let src = pattern_u8(npixels * cin);
    let d_srcs: Vec<CudaSlice<u8>> = (0..N_BUFFS)
        .map(|_| stream.clone_htod(&src).unwrap())
        .collect();
    let d_dst = std::cell::RefCell::new(stream.alloc_zeros::<u8>(npixels * cout).unwrap());
    let launch = std::rc::Rc::new(launch);

    let l1 = launch.clone();
    let s1 = stream.clone();
    let gpu_launch = Box::new(move |i: usize| {
        l1(&s1, &d_srcs[i], &mut d_dst.borrow_mut(), npixels);
    });

    let s2 = stream.clone();
    let src2 = src.clone();
    let gpu_e2e = Box::new(move || {
        let d_src = s2.clone_htod(&src2).unwrap();
        let mut d_out = s2.alloc_zeros::<u8>(npixels * cout).unwrap();
        launch(&s2, &d_src, &mut d_out, npixels);
        let _host: Vec<u8> = s2.clone_dtoh(&d_out).unwrap();
        s2.synchronize().unwrap();
    });

    OpBench {
        name,
        bytes_moved: npixels * (cin + cout),
        cpu,
        gpu_launch,
        gpu_e2e,
    }
}

fn main() {
    let json = std::env::args().any(|a| a == "--json");
    let ctx = CudaContext::new(0).expect("CUDA device 0 required");
    let stream = ctx.default_stream();

    if !json {
        println!("kornia color conversions: CPU (NEON) vs CUDA — Jetson Orin");
        println!(
            "kernel-only: {KERNEL_BATCHES}x{KERNEL_PER_BATCH} launches, {N_BUFFS} rotating src buffers"
        );
    }

    for &(width, height) in SIZES {
        let n = width * height;
        let size = ImageSize { width, height };

        // Host images reused by the CPU closures.
        let rgb_u8 = Image::<u8, 3>::new(size, pattern_u8(n * 3)).unwrap();
        let rgba_u8 = Image::<u8, 4>::new(size, pattern_u8(n * 4)).unwrap();
        let gray_u8 = Image::<u8, 1>::new(size, pattern_u8(n)).unwrap();
        let rgb_f32 = Image::<f32, 3>::new(size, pattern_f32(n * 3)).unwrap();

        // Build, run, and DROP each op immediately — keeping every op's rotating
        // device buffers alive at once OOMs at 4K (the f32 ops hold 8x100 MB).
        let run = |mut op: OpBench| run_op(&mut op, &stream, width, height, json);

        // ---- gray / swizzle / yuv / sepia (elementwise u8) ----
        {
            let src = rgb_u8.clone();
            let mut dst = Image::<u8, 1>::from_size_val(size, 0).unwrap();
            run(elementwise_u8(
                "gray_from_rgb_u8",
                &stream,
                n,
                3,
                1,
                Box::new(move || color::gray_from_rgb_u8(&src, &mut dst).unwrap()),
                |s, a, b, np| gray::launch_gray_from_rgb_u8(s, a, b, np).unwrap(),
            ));
        }
        {
            let src = rgb_u8.clone();
            let mut dst = Image::<u8, 3>::from_size_val(size, 0).unwrap();
            run(elementwise_u8(
                "bgr_from_rgb_u8",
                &stream,
                n,
                3,
                3,
                Box::new(move || color::bgr_from_rgb(&src, &mut dst).unwrap()),
                |s, a, b, np| swizzle::launch_bgr_from_rgb_u8(s, a, b, np).unwrap(),
            ));
        }
        {
            let src = rgb_u8.clone();
            let mut dst = Image::<u8, 4>::from_size_val(size, 0).unwrap();
            run(elementwise_u8(
                "rgba_from_rgb_u8",
                &stream,
                n,
                3,
                4,
                Box::new(move || color::rgba_from_rgb(&src, &mut dst).unwrap()),
                |s, a, b, np| swizzle::launch_rgba_from_rgb_u8(s, a, b, np).unwrap(),
            ));
        }
        {
            let src = rgba_u8.clone();
            let mut dst = Image::<u8, 3>::from_size_val(size, 0).unwrap();
            run(elementwise_u8(
                "rgb_from_rgba_u8",
                &stream,
                n,
                4,
                3,
                Box::new(move || color::rgb_from_rgba(&src, &mut dst, None).unwrap()),
                |s, a, b, np| swizzle::launch_rgb_from_rgba_u8(s, a, b, np, false, None).unwrap(),
            ));
        }
        {
            let src = rgb_u8.clone();
            let mut dst = Image::<u8, 3>::from_size_val(size, 0).unwrap();
            run(elementwise_u8(
                "ycbcr_from_rgb_u8",
                &stream,
                n,
                3,
                3,
                Box::new(move || color::ycbcr_from_rgb(&src, &mut dst).unwrap()),
                |s, a, b, np| {
                    yuv::launch_ycc_from_rgb_u8(s, a, b, np, yuv::ChromaOrder::YCrCb).unwrap()
                },
            ));
        }
        {
            let src = rgb_u8.clone();
            let mut dst = Image::<u8, 3>::from_size_val(size, 0).unwrap();
            run(elementwise_u8(
                "rgb_from_ycbcr_u8",
                &stream,
                n,
                3,
                3,
                Box::new(move || color::rgb_from_ycbcr(&src, &mut dst).unwrap()),
                |s, a, b, np| {
                    yuv::launch_rgb_from_ycc_u8(s, a, b, np, yuv::ChromaOrder::YCrCb).unwrap()
                },
            ));
        }
        {
            let src = rgb_u8.clone();
            let mut dst = Image::<u8, 3>::from_size_val(size, 0).unwrap();
            run(elementwise_u8(
                "sepia_from_rgb_u8",
                &stream,
                n,
                3,
                3,
                Box::new(move || color::sepia_from_rgb_u8(&src, &mut dst).unwrap()),
                |s, a, b, np| misc::launch_sepia_from_rgb_u8(s, a, b, np).unwrap(),
            ));
        }
        {
            let src = gray_u8.clone();
            let mut dst = Image::<u8, 3>::from_size_val(size, 0).unwrap();
            run(elementwise_u8(
                "apply_colormap_jet_u8",
                &stream,
                n,
                1,
                3,
                Box::new(move || color::apply_colormap(&src, &mut dst, ColormapType::Jet).unwrap()),
                |s, a, b, np| {
                    misc::launch_apply_colormap_u8(s, a, b, np, ColormapType::Jet).unwrap()
                },
            ));
        }
        {
            let src = gray_u8.clone();
            let mut dst = Image::<u8, 3>::from_size_val(size, 0).unwrap();
            run(elementwise_u8(
                "rgb_from_bayer_rggb_u8",
                &stream,
                n,
                1,
                3,
                Box::new(move || {
                    color::rgb_from_bayer(&src, BayerPattern::Rggb, &mut dst).unwrap()
                }),
                move |s, a, b, _np| {
                    bayer::launch_rgb_from_bayer_u8(s, a, b, height, width, BayerPattern::Rggb)
                        .unwrap()
                },
            ));
        }
        // ---- video decode/encode ----
        {
            let src = pattern_u8(n * 2);
            let mut dst = Image::<u8, 3>::from_size_val(size, 0).unwrap();
            let src_cpu = src.clone();
            run(elementwise_u8(
                "rgb_from_yuyv_u8",
                &stream,
                n,
                2,
                3,
                Box::new(move || color::rgb_from_yuyv(&src_cpu, &mut dst).unwrap()),
                move |s, a, b, _np| {
                    video::launch_rgb_from_packed422_u8(
                        s,
                        a,
                        b,
                        width,
                        height,
                        video::Packed422::Yuyv,
                    )
                    .unwrap()
                },
            ));
        }
        {
            let src = rgb_u8.clone();
            let mut dst = vec![0u8; n * 2];
            run(elementwise_u8(
                "yuyv_from_rgb_u8",
                &stream,
                n,
                3,
                2,
                Box::new(move || {
                    kornia_imgproc::color::yuyv_from_rgb(&src, &mut dst).unwrap();
                }),
                move |s, a, b, _np| video::launch_yuyv_from_rgb_u8(s, a, b, width, height).unwrap(),
            ));
        }
        // ---- f32 ops ----
        type CpuF32Fn =
            fn(&Image<f32, 3>, &mut Image<f32, 3>) -> Result<(), kornia_image::ImageError>;
        type LaunchF32Fn = fn(
            &Arc<CudaStream>,
            &CudaSlice<f32>,
            &mut CudaSlice<f32>,
            usize,
        )
            -> Result<(), kornia_imgproc::gpu::color_cuda::CudaColorError>;
        let f32_cases: [(&'static str, CpuF32Fn, LaunchF32Fn); 2] = [
            (
                "hsv_from_rgb_f32",
                color::hsv_from_rgb,
                hsv_hls::launch_hsv_from_rgb_f32,
            ),
            (
                "lab_from_rgb_f32",
                color::lab_from_rgb,
                kornia_imgproc::gpu::color_cuda::cie::launch_lab_from_rgb_f32,
            ),
        ];
        for (name, cpu_fn, launch_fn) in f32_cases {
            let src_img = rgb_f32.clone();
            let mut dst_img = Image::<f32, 3>::from_size_val(size, 0.0).unwrap();
            let host = rgb_f32.as_slice().to_vec();
            let d_srcs: Vec<CudaSlice<f32>> = (0..N_BUFFS)
                .map(|_| stream.clone_htod(&host).unwrap())
                .collect();
            let d_dst = std::cell::RefCell::new(stream.alloc_zeros::<f32>(n * 3).unwrap());
            let s1 = stream.clone();
            let s2 = stream.clone();
            let host2 = host.clone();
            run(OpBench {
                name,
                bytes_moved: n * 6 * 4,
                cpu: Box::new(move || cpu_fn(&src_img, &mut dst_img).unwrap()),
                gpu_launch: Box::new(move |i| {
                    launch_fn(&s1, &d_srcs[i], &mut d_dst.borrow_mut(), n).unwrap()
                }),
                gpu_e2e: Box::new(move || {
                    let d_src = s2.clone_htod(&host2).unwrap();
                    let mut d_out = s2.alloc_zeros::<f32>(n * 3).unwrap();
                    launch_fn(&s2, &d_src, &mut d_out, n).unwrap();
                    let _h: Vec<f32> = s2.clone_dtoh(&d_out).unwrap();
                    s2.synchronize().unwrap();
                }),
            });
        }
        // ---- NV12 encode (1.5 bytes/px output) ----
        {
            let src_img = rgb_u8.clone();
            let mut dst_host = vec![0u8; n * 3 / 2];
            let host = rgb_u8.as_slice().to_vec();
            let d_srcs: Vec<CudaSlice<u8>> = (0..N_BUFFS)
                .map(|_| stream.clone_htod(&host).unwrap())
                .collect();
            let d_dst = std::cell::RefCell::new(stream.alloc_zeros::<u8>(n * 3 / 2).unwrap());
            let s1 = stream.clone();
            let s2 = stream.clone();
            let host2 = host.clone();
            run(OpBench {
                name: "nv12_from_rgb_u8",
                bytes_moved: n * 3 + n * 3 / 2,
                cpu: Box::new(move || color::nv12_from_rgb(&src_img, &mut dst_host).unwrap()),
                gpu_launch: Box::new(move |i| {
                    video::launch_nv12_from_rgb_u8(
                        &s1,
                        &d_srcs[i],
                        &mut d_dst.borrow_mut(),
                        width,
                        height,
                    )
                    .unwrap()
                }),
                gpu_e2e: Box::new(move || {
                    let d_src = s2.clone_htod(&host2).unwrap();
                    let mut d_out = s2.alloc_zeros::<u8>(n * 3 / 2).unwrap();
                    video::launch_nv12_from_rgb_u8(&s2, &d_src, &mut d_out, width, height).unwrap();
                    let _h: Vec<u8> = s2.clone_dtoh(&d_out).unwrap();
                    s2.synchronize().unwrap();
                }),
            });
        }
        // ---- NV12 decode (1.5 bytes/px frame) ----
        {
            let frame = pattern_u8(n * 3 / 2);
            let mut dst_img = Image::<u8, 3>::from_size_val(size, 0).unwrap();
            let frame_cpu = frame.clone();
            let d_srcs: Vec<CudaSlice<u8>> = (0..N_BUFFS)
                .map(|_| stream.clone_htod(&frame).unwrap())
                .collect();
            let d_dst = std::cell::RefCell::new(stream.alloc_zeros::<u8>(n * 3).unwrap());
            let s1 = stream.clone();
            let s2 = stream.clone();
            let frame2 = frame.clone();
            run(OpBench {
                name: "rgb_from_nv12_u8",
                bytes_moved: n * 3 / 2 + n * 3,
                cpu: Box::new(move || color::rgb_from_nv12(&frame_cpu, &mut dst_img).unwrap()),
                gpu_launch: Box::new(move |i| {
                    video::launch_rgb_from_planar420_u8(
                        &s1,
                        &d_srcs[i],
                        &mut d_dst.borrow_mut(),
                        width,
                        height,
                        video::Planar420::Nv12,
                    )
                    .unwrap()
                }),
                gpu_e2e: Box::new(move || {
                    let d_src = s2.clone_htod(&frame2).unwrap();
                    let mut d_out = s2.alloc_zeros::<u8>(n * 3).unwrap();
                    video::launch_rgb_from_planar420_u8(
                        &s2,
                        &d_src,
                        &mut d_out,
                        width,
                        height,
                        video::Planar420::Nv12,
                    )
                    .unwrap();
                    let _h: Vec<u8> = s2.clone_dtoh(&d_out).unwrap();
                    s2.synchronize().unwrap();
                }),
            });
        }

        if !json {
            println!();
        }
    }
}
