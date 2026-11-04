//! CUDA FAST (`cuda::fast`) vs the CPU `fast_detect_rows_u8` it is bit-exact with.
//!
//! Per case: the GPU score map is first checked against the CPU corners, then
//! - CPU: `fast_detect_rows_u8` (rayon) and `fast_detect_rows_u8_serial`,
//!   wall clock, mean over `CPU_ITERS` calls after `WARMUP` calls;
//! - GPU kernel: `launch_fast_score_u8` only, `ITERS` back-to-back launches
//!   between CUDA events, fastest of `ROUNDS` (no transfers);
//! - GPU end-to-end: H2D of the image + kernel + D2H of the u16 score map +
//!   synchronize, buffers preallocated, wall clock, mean over `ITERS`
//!   (excludes the host-side compaction into a corner list).
//!
//! ```text
//! cargo bench -p kornia-imgproc --features cuda --bench bench_cuda_fast
//! ```

use std::{sync::Arc, time::Instant};

use cudarc::driver::{sys::CUevent_flags, CudaContext, CudaStream};
use kornia_image::Image;
use kornia_imgproc::{
    cuda::fast::launch_fast_score_u8,
    features::{fast_detect_rows_u8, fast_detect_rows_u8_serial},
};

const WARMUP: usize = 20;
const ROUNDS: usize = 5;
const ITERS: usize = 100;
const CPU_ITERS: usize = 30;
const THRESHOLD: f32 = 0.15;
const ARC: usize = 9;
const BORDER: usize = 3;

fn synthetic(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let check = ((x / 16 + y / 16) % 2) as u8 * 120;
            let noise = ((x.wrapping_mul(37) + y.wrapping_mul(13)) % 100) as u8;
            data[y * width + x] = check.wrapping_add(noise);
        }
    }
    data
}

fn mean_ms(iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn bench(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, name: &str, img: &Image<u8, 1>) {
    let (w, h) = (img.width(), img.height());
    let npx = w * h;
    let t_u8 = (THRESHOLD * 255.0).round().clamp(1.0, 255.0) as u32;
    let margin = BORDER.max(3) as u32;
    let mut d_src = stream.clone_htod(img.as_slice()).expect("htod");
    let mut d_dst = stream.alloc_zeros::<u16>(npx).expect("alloc");
    let mut h_dst = vec![0u16; npx];
    let launch = |src: &cudarc::driver::CudaSlice<u8>, dst: &mut cudarc::driver::CudaSlice<u16>| {
        launch_fast_score_u8(
            stream, src, dst, w as u32, h as u32, t_u8, ARC as u32, margin,
        )
        .expect("launch")
    };

    // Correctness first: every CPU corner on the GPU with the same score
    // (the CPU's SIMD in-block filter drops some FAST-9 corners at width >= 800).
    launch(&d_src, &mut d_dst);
    stream.memcpy_dtoh(&d_dst, &mut h_dst).expect("dtoh");
    stream.synchronize().expect("sync");
    let cpu = fast_detect_rows_u8(img, THRESHOLD, ARC, BORDER, 0..h);
    for &([y, x], s) in &cpu {
        assert_eq!(h_dst[y * w + x], (s * 255.0).round() as u16, "({y}, {x})");
    }
    let gpu_n = h_dst.iter().filter(|&&s| s != 0).count();
    if w < 800 {
        assert_eq!(gpu_n, cpu.len(), "corner count");
    }

    let cpu_par = mean_ms(CPU_ITERS, || {
        std::hint::black_box(fast_detect_rows_u8(img, THRESHOLD, ARC, BORDER, 0..h));
    });
    let cpu_ser = mean_ms(CPU_ITERS, || {
        std::hint::black_box(fast_detect_rows_u8_serial(
            img,
            THRESHOLD,
            ARC,
            BORDER,
            0..h,
        ));
    });

    let ev = || {
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event")
    };
    let (start, end) = (ev(), ev());
    for _ in 0..WARMUP {
        launch(&d_src, &mut d_dst);
    }
    let mut kernel = f64::INFINITY;
    for _ in 0..ROUNDS {
        start.record(stream).expect("record");
        for _ in 0..ITERS {
            launch(&d_src, &mut d_dst);
        }
        end.record(stream).expect("record");
        stream.synchronize().expect("sync");
        kernel = kernel.min(start.elapsed_ms(&end).expect("elapsed") as f64 / ITERS as f64);
    }

    let e2e = mean_ms(ITERS, || {
        stream
            .memcpy_htod(img.as_slice(), &mut d_src)
            .expect("htod");
        launch(&d_src, &mut d_dst);
        stream.memcpy_dtoh(&d_dst, &mut h_dst).expect("dtoh");
        stream.synchronize().expect("sync");
    });

    println!(
        "| {name} {w}x{h} | {} / {gpu_n} | {cpu_ser:.3} | {cpu_par:.3} | {kernel:.4} | {e2e:.3} | {:.1}x | {:.1}x |",
        cpu.len(),
        cpu_par / kernel,
        cpu_par / e2e
    );
}

fn main() {
    let ctx = CudaContext::new(0).expect("device 0");
    let stream = ctx.new_stream().expect("stream");
    let (major, minor) = ctx.compute_capability().expect("cc");
    let name = ctx.name().unwrap_or_else(|_| "unknown GPU".into());
    let threads = rayon::current_num_threads();

    println!("GPU: {name} (sm_{major}{minor}); CPU rayon threads: {threads}");
    println!("FAST-{ARC}, threshold {THRESHOLD}, border {BORDER}; warmup {WARMUP}, GPU {ROUNDS}x{ITERS} (kernel: fastest round), CPU {CPU_ITERS} iters");
    println!();
    println!("| image | corners CPU / GPU | CPU 1T ms | CPU {threads}T ms | GPU kernel ms | GPU H2D+kernel+D2H ms | kernel vs CPU {threads}T | e2e vs CPU {threads}T |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");

    let mh01 = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/mh01_frame1.png");
    if let Ok(img) = kornia_io::png::read_image_png_mono8(mh01) {
        bench(&ctx, &stream, "mh01_frame1", &img.into_inner());
    }
    for (w, h) in [(752, 480), (1920, 1080), (3840, 2160)] {
        let img = Image::new([w, h].into(), synthetic(w, h)).expect("image");
        bench(&ctx, &stream, "synthetic", &img);
    }
}
