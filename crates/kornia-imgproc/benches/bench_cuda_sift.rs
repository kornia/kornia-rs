use std::hint::black_box;
use std::time::Instant;

use argh::FromArgs;
use cudarc::driver::CudaContext;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::cuda::sift::{FirstOctave, SiftCuda, SiftCudaConfig, SiftMatcher, DESCR_LEN};

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

/// Best per-iteration milliseconds of each mode. Each iteration starts from the
/// next mode, so no mode always runs first.
fn best_of_rotated<const MODES: usize>(
    warmup: usize,
    iters: usize,
    mut run: impl FnMut(usize) -> Result<(), Box<dyn std::error::Error>>,
) -> Result<[f64; MODES], Box<dyn std::error::Error>> {
    for mode in 0..MODES {
        for _ in 0..warmup {
            run(mode)?;
        }
    }
    let mut best = [f64::INFINITY; MODES];
    for iter in 0..iters {
        for step in 0..MODES {
            let mode = (iter + step) % MODES;
            let start = Instant::now();
            run(mode)?;
            best[mode] = best[mode].min(start.elapsed().as_secs_f64() * 1e3);
        }
    }
    Ok(best)
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

    println!("\nmatching, best ms per pair of descriptor blocks");
    println!(
        "{:<8} {:>11} {:>11} {:>11}",
        "rows", "forward", "cross-check", "match_smnn"
    );
    for rows in [512usize, 1024, 2048, 4096, 8192] {
        let (a, b) = (match_descriptors(rows, 1), match_descriptors(rows, 2));
        let (da, db) = (stream.clone_htod(&a)?, stream.clone_htod(&b)?);
        let mut matcher = SiftMatcher::new(&stream, rows)?;
        let [forward_ms, cross_check_ms, smnn_ms] =
            best_of_rotated(args.warmup, args.iters, |mode| {
                let (d1, d2) = (black_box(da.as_view()), black_box(db.as_view()));
                if mode == 2 {
                    black_box(matcher.match_smnn(&ctx, &stream, &d1, rows, &d2, rows, 0.95)?);
                } else {
                    black_box(matcher.match_descriptors(
                        &ctx,
                        &stream,
                        &d1,
                        rows,
                        &d2,
                        rows,
                        0.95,
                        mode == 1,
                    )?);
                }
                Ok(())
            })?;
        println!("{rows:<8} {forward_ms:>11.3} {cross_check_ms:>11.3} {smnn_ms:>11.3}");
    }

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

fn match_descriptors(rows: usize, mut seed: u64) -> Vec<f32> {
    (0..rows * DESCR_LEN)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) % 256) as f32
        })
        .collect()
}
