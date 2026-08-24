use std::time::Instant;

use argh::FromArgs;
use cudarc::driver::CudaContext;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::cuda::sift::{FirstOctave, SiftCuda, SiftCudaConfig};

/// Benchmark unified memory against explicit device copies for SIFT.
#[derive(FromArgs)]
struct Args {
    /// CUDA device ordinal (default 0)
    #[argh(option, default = "0")]
    device: usize,

    /// timed iterations per case (default 10)
    #[argh(option, default = "10")]
    iters: usize,

    /// untimed warmup iterations per case (default 3)
    #[argh(option, default = "3")]
    warmup: usize,

    /// accepted and ignored: `cargo bench` passes `--bench` to the harness
    #[argh(switch)]
    #[allow(dead_code)]
    bench: bool,
}

const SIZES: &[(&str, usize, usize)] = &[
    ("VGA    640x480", 640, 480),
    ("HD    1280x720", 1280, 720),
    ("FHD  1920x1080", 1920, 1080),
];

fn timed<F>(warmup: usize, iters: usize, mut f: F) -> Result<f64, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    for _ in 0..warmup {
        f()?;
    }
    let start = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(start.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

fn fill_ramp(buf: &mut [f32]) {
    let inv = 255.0 / buf.len() as f32;
    for (i, px) in buf.iter_mut().enumerate() {
        *px = i as f32 * inv;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    let ctx = CudaContext::new(args.device)?;
    let stream = ctx.default_stream();

    let integrated =
        ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)?;
    let cc_major = ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    )?;
    let cc_minor = ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    )?;

    println!("device {}: sm_{cc_major}{cc_minor}", args.device);
    println!(
        "integrated: {} ({})",
        integrated,
        if integrated == 1 {
            "shared physical memory — copies are redundant"
        } else {
            "discrete — copies cross PCIe"
        }
    );
    println!("iters: {} (warmup {})\n", args.iters, args.warmup);

    println!("── per frame, end-to-end SIFT detect_and_compute() ───────────");
    println!(
        "{:<16} {:>11} {:>11} {:>9}",
        "size", "explicit ms", "unified ms", "speedup"
    );
    println!("{}", "-".repeat(52));

    for &(label, w, h) in SIZES {
        let size = ImageSize {
            width: w,
            height: h,
        };

        let cfg = SiftCudaConfig::default();
        let mut plan_explicit = SiftCuda::new(
            &ctx,
            &stream,
            w,
            h,
            cfg,
            FirstOctave::Native, // fo=0
            4,                   // 4 octaves
        )?;
        plan_explicit.set_fast_descriptor(true);

        let mut plan_unified = SiftCuda::new(&ctx, &stream, w, h, cfg, FirstOctave::Native, 4)?;
        plan_unified.set_fast_descriptor(true);

        // Explicit: host frame, uploaded, compute. (D2H for descriptors is avoided in python API, so we just measure compute + H2D).
        let explicit_ms = timed(args.warmup, args.iters, || {
            let mut host = Image::<f32, 1>::from_size_val(size, 0.0)?;
            fill_ramp(host.as_slice_mut());

            let device = host.to_cuda(&stream)?;
            let feats = plan_explicit.detect_and_compute(&ctx, &stream, &device)?;
            std::hint::black_box(&feats);
            Ok(())
        })?;

        // Unified: write into unified memory, compute directly.
        let unified_ms = timed(args.warmup, args.iters, || {
            let mut u_src = Image::<f32, 1>::zeros_cuda_unified(size, &stream)?;
            fill_ramp(u_src.as_slice_mut());

            let feats = plan_unified.detect_and_compute(&ctx, &stream, &u_src)?;
            std::hint::black_box(&feats);
            Ok(())
        })?;

        let speedup = explicit_ms / unified_ms;
        println!("{label:<16} {explicit_ms:>11.3} {unified_ms:>11.3} {speedup:>8.2}x");
    }

    Ok(())
}
