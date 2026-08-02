//! Unified (managed) memory vs. explicit H2D/D2H copies.
//!
//! The case for [`kornia_tensor::zeros_unified`] / `Image::zeros_unified` is
//! that on an **integrated** GPU the CPU and GPU share one physical memory, so
//! an explicit copy moves bytes from RAM back to the same RAM. This benchmark
//! measures what that actually costs, so the claim can be checked on the target
//! platform instead of assumed.
//!
//! The two paths are not a single number each — they trade a *recurring* cost
//! for a *one-time* one, so each component is timed separately:
//!
//! * `alloc` — one-time, per buffer: `zeros_cuda` vs `zeros_unified`.
//! * `fill` — per frame, whenever the CPU produces the data: writing into a
//!   pageable host buffer vs writing straight into unified memory. Unified
//!   memory can be slower to write from the CPU, which is the hidden cost.
//! * `transfer` — per frame: H2D + D2H for the explicit path; **zero** for
//!   unified, which is the whole point.
//!
//! Unified wins when the extra per-frame *fill* cost is smaller than the
//! per-frame *transfer* it removes.
//!
//! ```text
//! cargo run --release -p kornia-imgproc --features cuda --example bench_unified_memory
//! ```

use std::sync::Arc;
use std::time::Instant;

use argh::FromArgs;
use cudarc::driver::CudaContext;
use kornia_image::{Image, ImageError, ImageSize};

/// Benchmark unified memory against explicit device copies.
#[derive(FromArgs)]
struct Args {
    /// CUDA device ordinal (default 0)
    #[argh(option, default = "0")]
    device: usize,

    /// timed iterations per case (default 20)
    #[argh(option, default = "20")]
    iters: usize,

    /// untimed warmup iterations per case (default 5)
    #[argh(option, default = "5")]
    warmup: usize,
}

/// Resolutions to sweep, as (label, width, height).
const SIZES: &[(&str, usize, usize)] = &[
    ("VGA    640x480", 640, 480),
    ("HD    1280x720", 1280, 720),
    ("FHD  1920x1080", 1920, 1080),
    ("4K   3840x2160", 3840, 2160),
];

/// Mean milliseconds per iteration of `f`, after `warmup` untimed runs.
fn timed<F>(warmup: usize, iters: usize, mut f: F) -> Result<f64, ImageError>
where
    F: FnMut() -> Result<(), ImageError>,
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

/// Write a ramp into `buf`. The same work whether `buf` is pageable host
/// memory or unified memory — only the destination differs, which is exactly
/// the cost being compared.
fn fill_ramp(buf: &mut [f32]) {
    let inv = 1.0 / buf.len() as f32;
    for (i, px) in buf.iter_mut().enumerate() {
        *px = i as f32 * inv;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    let ctx = CudaContext::new(args.device)?;
    let stream = ctx.default_stream();

    // cudaDevAttrIntegrated: 1 on Jetson (shared physical memory), 0 on a
    // discrete card. The whole premise of unified memory hinges on this bit.
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
    println!("iters: {} (warmup {})", args.iters, args.warmup);

    println!("\n── one-time, per buffer ─────────────────────────────────────");
    println!(
        "{:<16} {:>9} {:>14} {:>14}",
        "size", "MiB", "zeros_cuda ms", "unified ms"
    );
    for &(label, w, h) in SIZES {
        let size = ImageSize {
            width: w,
            height: h,
        };
        let mib = (w * h * 3 * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0);

        let dev = timed(args.warmup, args.iters, || {
            let d = Image::<f32, 3>::zeros_cuda(size, &stream)?;
            std::hint::black_box(&d);
            Ok(())
        })?;
        let uni = timed(args.warmup, args.iters, || {
            let u = Image::<f32, 3>::zeros_unified(size, &ctx)?;
            std::hint::black_box(&u);
            Ok(())
        })?;

        println!("{label:<16} {mib:>9.1} {dev:>14.3} {uni:>14.3}");
    }

    println!("\n── per frame ────────────────────────────────────────────────");
    println!(
        "{:<16} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "size", "fill", "fill_uni", "H2D", "D2H", "explicit", "unified"
    );
    println!("{}", "-".repeat(76));

    for &(label, w, h) in SIZES {
        let size = ImageSize {
            width: w,
            height: h,
        };

        // Host-side production of one frame, into ordinary pageable memory.
        let mut host = Image::<f32, 3>::from_size_val(size, 0.0)?;
        let fill = timed(args.warmup, args.iters, || {
            fill_ramp(host.as_slice_mut());
            Ok(())
        })?;

        // The same production, writing directly into unified memory. Allocated
        // once, outside the timer — allocation is the one-time cost above.
        let mut uni = Image::<f32, 3>::zeros_unified(size, &ctx)?;
        let fill_uni = timed(args.warmup, args.iters, || {
            fill_ramp(uni.as_slice_mut());
            Ok(())
        })?;

        // Explicit path: upload the produced frame, run, download the result.
        let h2d = timed(args.warmup, args.iters, || {
            let d = host.to_cuda(&stream)?;
            std::hint::black_box(&d);
            Ok(())
        })?;
        let device = host.to_cuda(&stream)?;
        let d2h = timed(args.warmup, args.iters, || {
            let back = device.to_host_image(&stream)?;
            std::hint::black_box(&back);
            Ok(())
        })?;

        // Per-frame totals. Unified pays no transfer at all.
        let explicit = fill + h2d + d2h;
        let unified = fill_uni;

        println!(
            "{label:<16} {fill:>8.3} {fill_uni:>9.3} {h2d:>9.3} {d2h:>9.3} \
             {explicit:>9.3} {unified:>9.3}"
        );
    }

    println!(
        "\n'explicit' = fill + H2D + D2H. 'unified' = fill_uni, with no transfer.\n\
         Unified wins where (fill_uni - fill) < (H2D + D2H).\n\
         \n\
         Caveat: the unified column is CPU-side only — no kernel reads the\n\
         buffer, so it excludes whatever the GPU pays to access it. On an\n\
         integrated part that should be nothing; on a discrete one the driver\n\
         demand-pages over PCIe. The probe below is why it cannot be measured\n\
         end-to-end yet."
    );

    device_dispatch_probe(&ctx)?;

    Ok(())
}

/// Check whether a unified image can actually reach a kornia CUDA kernel
/// through the normal public API. Allocation working is not the same as the
/// operation dispatching, and that difference decides whether the feature is
/// usable today.
fn device_dispatch_probe(ctx: &Arc<CudaContext>) -> Result<(), Box<dyn std::error::Error>> {
    use kornia_imgproc::interpolation::InterpolationMode;
    use kornia_imgproc::resize::resize;

    let src = Image::<f32, 3>::zeros_unified(
        ImageSize {
            width: 64,
            height: 48,
        },
        ctx,
    )?;
    let mut dst = Image::<f32, 3>::zeros_unified(
        ImageSize {
            width: 32,
            height: 24,
        },
        ctx,
    )?;

    print!("\nresize() on unified images: ");
    match resize(&src, &mut dst, InterpolationMode::Bilinear) {
        Ok(()) => println!("dispatched"),
        Err(e) => println!("FAILED — {e}"),
    }

    Ok(())
}
