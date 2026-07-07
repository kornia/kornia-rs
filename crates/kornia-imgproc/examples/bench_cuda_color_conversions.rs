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
//! cargo run --example bench_cuda_color_conversions --features cuda --release [-- --json]
//! ```
#![cfg(feature = "cuda")]

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_image::color_spaces::BayerPattern;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::color::{self, ColormapType};
use kornia_imgproc::cuda::color_cuda::{bayer, gray, hsv_hls, misc, swizzle, video, yuv};

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

/// Single source of truth for the output row — the JSON schema must stay
/// stable for report_color_bench.py.
fn print_row(op: &str, w: usize, h: usize, variant: &str, s: &BenchStats, json: bool) {
    if json {
        println!(
            "{{\"op\":\"{op}\",\"width\":{w},\"height\":{h},\"variant\":\"{variant}\",\"min_ms\":{:.6},\"p50_ms\":{:.6},\"p95_ms\":{:.6},\"gbps\":{:.3}}}",
            s.min_ms, s.p50_ms, s.p95_ms, s.gbps
        );
    } else {
        println!(
            "{op:>26} {:>10} {variant:>20} min {:>9.4} ms  p50 {:>9.4} ms  p95 {:>9.4} ms  {:>7.1} GB/s",
            format!("{w}x{h}"),
            s.min_ms,
            s.p50_ms,
            s.p95_ms,
            s.gbps
        );
    }
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
        print_row(op.name, width, height, variant, s, json);
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

/// End-to-end through **pinned** host memory: src and dst live in reusable
/// page-locked staging buffers, so the driver DMAs both copies instead of
/// staging them through an internal bounce buffer.
#[allow(clippy::too_many_arguments)]
fn run_pinned_e2e_u8(
    name: &'static str,
    stream: &Arc<CudaStream>,
    npixels: usize,
    cin: usize,
    cout: usize,
    width: usize,
    height: usize,
    json: bool,
    launch: impl Fn(&Arc<CudaStream>, &CudaSlice<u8>, &mut CudaSlice<u8>, usize),
) {
    let ctx = stream.context().clone();
    // Pinned staging buffers are allocated ONCE (cuMemHostAlloc page-locks —
    // it is far too expensive to sit inside a frame loop) and reused.
    let mut src_pin = kornia_tensor::zeros_pinned::<u8, 3>([height, width, cin], &ctx).unwrap();
    src_pin
        .as_slice_mut()
        .copy_from_slice(&pattern_u8(npixels * cin));
    let mut dst_pin = kornia_tensor::zeros_pinned::<u8, 1>([npixels * cout], &ctx).unwrap();

    let mut iter = || {
        let d_src = src_pin.to_cuda(stream).unwrap();
        let mut d_out = stream.alloc_zeros::<u8>(npixels * cout).unwrap();
        launch(stream, d_src.as_cudaslice().unwrap(), &mut d_out, npixels);
        stream.memcpy_dtoh(&d_out, dst_pin.as_slice_mut()).unwrap();
        stream.synchronize().unwrap();
    };
    iter(); // warm
    let s = stats_from(time_each(E2E_ITERS, &mut iter), npixels * (cin + cout));
    print_row(name, width, height, "kornia-cuda-e2e-pinned", &s, json);
}

/// Fused color+resize+normalize (`Preprocessor` with a `SourceFormat`) vs the
/// decode-then-preprocess chain. Device-resident src, batched launches + sync,
/// 1080p-style frame -> 640x640 letterbox tensor.
#[cfg(feature = "cuda")]
fn run_fused_preprocess(stream: &Arc<CudaStream>, width: usize, height: usize, json: bool) {
    use kornia_imgproc::preprocess::{Preprocessor, ResizeMode, SourceFormat};
    const OUT: usize = 640;
    const BATCH: usize = 20;
    const ROUNDS: usize = 8;
    let n = width * height;

    type Decode = fn(&Arc<CudaStream>, &CudaSlice<u8>, &mut CudaSlice<u8>, usize, usize);
    let cases: [(&str, SourceFormat, usize, Decode); 3] = [
        (
            "preprocess_nv12_640",
            SourceFormat::Nv12,
            n * 3 / 2,
            |s, a, b, w, h| {
                video::launch_rgb_from_planar420_u8(s, a, b, w, h, video::Planar420::Nv12).unwrap()
            },
        ),
        (
            "preprocess_yuyv_640",
            SourceFormat::Yuyv,
            n * 2,
            |s, a, b, w, h| {
                video::launch_rgb_from_packed422_u8(s, a, b, w, h, video::Packed422::Yuyv).unwrap()
            },
        ),
        (
            "preprocess_bgr_640",
            SourceFormat::Bgr8,
            n * 3,
            |s, a, b, w, h| swizzle::launch_bgr_from_rgb_u8(s, a, b, w * h).unwrap(),
        ),
    ];

    for (name, format, src_len, decode) in cases {
        let src = pattern_u8(src_len);
        let d_src = stream.clone_htod(&src).unwrap();
        let pre_fused = Preprocessor::builder()
            .mode(ResizeMode::Letterbox)
            .source_format(format)
            .build_cuda(stream.clone())
            .unwrap();
        let pre_rgb = Preprocessor::letterbox(stream.clone()).unwrap();
        let mut dst = kornia_tensor::zeros_cuda::<f32, 4>([1, 3, OUT, OUT], stream).unwrap();
        // src bytes read + CHW f32 written (ignores the chained variant's extra
        // intermediate traffic on purpose: same logical work for both).
        let bytes = src_len + OUT * OUT * 3 * 4;

        // Fused: one kernel per frame.
        for _ in 0..BATCH {
            pre_fused.run_raw(&d_src, width, height, &mut dst).unwrap();
        }
        stream.synchronize().unwrap();
        let mut fused_samples = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let t0 = Instant::now();
            for _ in 0..BATCH {
                pre_fused.run_raw(&d_src, width, height, &mut dst).unwrap();
            }
            stream.synchronize().unwrap();
            fused_samples.push(t0.elapsed().as_secs_f64() * 1e3 / BATCH as f64);
        }
        let s_fused = stats_from(fused_samples, bytes);

        // Chained: decode to a persistent device RGB image, then RGB preprocess.
        let d_rgb = stream.alloc_zeros::<u8>(n * 3).unwrap();
        let mut rgb_img = Image::<u8, 3>(kornia_tensor::Tensor::from_cudaslice(
            d_rgb,
            [height, width, 3],
            stream.clone(),
        ));
        for _ in 0..BATCH {
            decode(
                stream,
                &d_src,
                rgb_img.0.as_cudaslice_mut().unwrap(),
                width,
                height,
            );
            pre_rgb.run(&rgb_img, &mut dst).unwrap();
        }
        stream.synchronize().unwrap();
        let mut chained_samples = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let t0 = Instant::now();
            for _ in 0..BATCH {
                decode(
                    stream,
                    &d_src,
                    rgb_img.0.as_cudaslice_mut().unwrap(),
                    width,
                    height,
                );
                pre_rgb.run(&rgb_img, &mut dst).unwrap();
            }
            stream.synchronize().unwrap();
            chained_samples.push(t0.elapsed().as_secs_f64() * 1e3 / BATCH as f64);
        }
        let s_chained = stats_from(chained_samples, bytes);

        for (variant, s) in [("fused", &s_fused), ("chained", &s_chained)] {
            print_row(name, width, height, variant, s, json);
        }
    }
}

/// One-shot experiments (run at the smallest size): CUDA Graph replay for the
/// launch-overhead-bound small-image case, and an 8 px/thread gray variant
/// probing the bandwidth headroom above the shipped 4 px/thread kernel.
fn run_experiments(json: bool) {
    use kornia_tensor::CudaKernel;

    let ctx = CudaContext::new(0).expect("CUDA device 0 required");
    // Stream capture is illegal on the default (null) stream.
    let stream = ctx.new_stream().unwrap();

    // ── Experiment A: CUDA Graph replay for the launch-overhead-bound small
    // images (the rgba_from_rgb @ VGA red cell) — MEASURED NO-GO with cudarc
    // 0.19: its safe launch path makes every kernel argument wait on the
    // slice's tracking event, and any such event recorded outside capture
    // makes the captured launch fail with CUDA_ERROR_STREAM_CAPTURE_ISOLATION
    // (verified in both THREAD_LOCAL and RELAXED capture modes). Stream
    // capture would need either raw-pointer launches or an upstream cudarc
    // capture-aware mode. Until then, small-image launch overhead (~15 µs)
    // stands; batch small conversions into fewer, larger launches instead.

    // ── Experiment B: 8 px/thread gray (two word-triples per thread) ──
    {
        const GRAY8_SRC: &str = r#"
extern "C" __global__ void gray8px(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ngroups = (npixels + 7u) / 8u;
    if (g >= ngroups) return;
    unsigned int p = g * 8u;
    if (p + 8u <= npixels) {
        const unsigned int* s32 = (const unsigned int*)src;
        unsigned int y[8];
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            unsigned int w0 = __ldg(&s32[g * 6u + q * 3u]);
            unsigned int w1 = __ldg(&s32[g * 6u + q * 3u + 1u]);
            unsigned int w2 = __ldg(&s32[g * 6u + q * 3u + 2u]);
            y[q*4+0] = (77u*(w0 & 0xFFu) + 150u*((w0 >> 8) & 0xFFu) + 29u*((w0 >> 16) & 0xFFu)) >> 8;
            y[q*4+1] = (77u*(w0 >> 24) + 150u*(w1 & 0xFFu) + 29u*((w1 >> 8) & 0xFFu)) >> 8;
            y[q*4+2] = (77u*((w1 >> 16) & 0xFFu) + 150u*(w1 >> 24) + 29u*(w2 & 0xFFu)) >> 8;
            y[q*4+3] = (77u*((w2 >> 8) & 0xFFu) + 150u*((w2 >> 16) & 0xFFu) + 29u*(w2 >> 24)) >> 8;
        }
        uint2 out;
        out.x = y[0] | (y[1] << 8) | (y[2] << 16) | (y[3] << 24);
        out.y = y[4] | (y[5] << 8) | (y[6] << 16) | (y[7] << 24);
        ((uint2*)dst)[g] = out;
    } else {
        for (unsigned int i = p; i < npixels; ++i) {
            unsigned int b = i * 3u;
            dst[i] = (unsigned char)((77u*src[b] + 150u*src[b+1u] + 29u*src[b+2u]) >> 8);
        }
    }
}
"#;
        let (w, h) = (1920usize, 1080usize);
        let n = w * h;
        let kernel = CudaKernel::compile(stream.context(), GRAY8_SRC, "gray8px").unwrap();
        let src = pattern_u8(n * 3);
        let d_srcs: Vec<CudaSlice<u8>> = (0..N_BUFFS)
            .map(|_| stream.clone_htod(&src).unwrap())
            .collect();
        let mut d_dst = stream.alloc_zeros::<u8>(n).unwrap();
        let n32 = n as u32;
        let mut go = |i: usize| {
            kernel
                .launch_builder(&stream)
                .arg(&d_srcs[i % N_BUFFS])
                .arg(&mut d_dst)
                .arg(&n32)
                .launch_1d(n32.div_ceil(8))
                .unwrap();
        };
        for i in 0..30 {
            go(i);
        }
        stream.synchronize().unwrap();
        let mut samples = Vec::new();
        let mut li = 0usize;
        for _ in 0..10 {
            let t0 = Instant::now();
            for _ in 0..20 {
                go(li);
                li += 1;
            }
            stream.synchronize().unwrap();
            samples.push(t0.elapsed().as_secs_f64() * 1e3 / 20.0);
        }
        let st = stats_from(samples, n * 4);
        if json {
            println!(
                "{{\"op\":\"exp_gray_8px_thread\",\"width\":{w},\"height\":{h},\"variant\":\"kernel\",\"min_ms\":{:.6},\"p50_ms\":{:.6},\"p95_ms\":{:.6},\"gbps\":{:.3}}}",
                st.min_ms, st.p50_ms, st.p95_ms, st.gbps
            );
        } else {
            println!(
                "{:>26} {:>10} {:>20} min {:>9.4} ms  {:>7.1} GB/s",
                "exp_gray_8px_thread",
                format!("{w}x{h}"),
                "kernel",
                st.min_ms,
                st.gbps
            );
        }
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

    run_experiments(json);

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
            -> Result<(), kornia_imgproc::cuda::color_cuda::CudaColorError>;
        let f32_cases: [(&'static str, CpuF32Fn, LaunchF32Fn); 2] = [
            (
                "hsv_from_rgb_f32",
                color::hsv_from_rgb,
                hsv_hls::launch_hsv_from_rgb_f32,
            ),
            (
                "lab_from_rgb_f32",
                color::lab_from_rgb,
                kornia_imgproc::cuda::color_cuda::cie::launch_lab_from_rgb_f32,
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

        // ---- fused color+preprocess vs chained ----
        #[cfg(feature = "cuda")]
        run_fused_preprocess(&stream, width, height, json);

        // ---- pinned-memory end-to-end (representative u8 ops) ----
        run_pinned_e2e_u8(
            "gray_from_rgb_u8",
            &stream,
            n,
            3,
            1,
            width,
            height,
            json,
            |s, a, b, np| gray::launch_gray_from_rgb_u8(s, a, b, np).unwrap(),
        );
        run_pinned_e2e_u8(
            "bgr_from_rgb_u8",
            &stream,
            n,
            3,
            3,
            width,
            height,
            json,
            |s, a, b, np| swizzle::launch_bgr_from_rgb_u8(s, a, b, np).unwrap(),
        );
        run_pinned_e2e_u8(
            "ycbcr_from_rgb_u8",
            &stream,
            n,
            3,
            3,
            width,
            height,
            json,
            |s, a, b, np| {
                yuv::launch_ycc_from_rgb_u8(s, a, b, np, yuv::ChromaOrder::YCrCb).unwrap()
            },
        );

        if !json {
            println!();
        }
    }
}
