//! Unified (managed) memory vs. explicit H2D/D2H copies.
//!
//! The case for [`kornia_tensor::zeros_unified`] / `Image::zeros_unified` is
//! that on an **integrated** GPU the CPU and GPU share one physical memory, so
//! an explicit copy moves bytes from RAM to the same RAM. This benchmark
//! measures what that copy actually costs, so the claim can be checked on the
//! target platform instead of assumed.
//!
//! Run on a Jetson (integrated) and on a discrete GPU; the two should disagree.
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

/// Milliseconds elapsed while running `f` `iters` times, after `warmup` untimed
/// runs. Returns the mean per-iteration duration.
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
    println!("iters: {} (warmup {})\n", args.iters, args.warmup);

    println!(
        "{:<16} {:>9} {:>10} {:>10} {:>10} {:>10}",
        "size", "MiB", "H2D ms", "D2H ms", "copies ms", "unified ms"
    );
    println!("{}", "-".repeat(70));

    for &(label, w, h) in SIZES {
        let size = ImageSize {
            width: w,
            height: h,
        };
        let n = w * h * 3;
        let mib = (n * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0);

        // Ramp data, so the pages are genuinely dirty on the host side.
        let data: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let host = Image::<f32, 3>::new(size, data)?;

        // ── Explicit path: host image → device → back ────────────────────────
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

        // ── Unified path: allocate once, write from the host, no copy ────────
        let unified = timed(args.warmup, args.iters, || {
            let mut u = Image::<f32, 3>::zeros_unified(size, &ctx)?;
            fill_unified(&mut u, n);
            std::hint::black_box(&u);
            Ok(())
        })?;

        println!(
            "{label:<16} {mib:>9.1} {h2d:>10.3} {d2h:>10.3} {:>10.3} {unified:>10.3}",
            h2d + d2h
        );
    }

    println!(
        "\n'copies ms' is the round trip the unified path removes; \
         'unified ms' is alloc + host fill of the same buffer."
    );

    device_dispatch_probe(&ctx)?;

    Ok(())
}

/// Write a ramp straight into unified memory from the CPU — no H2D copy. This
/// is the access pattern unified memory is meant to enable.
fn fill_unified(img: &mut Image<f32, 3>, n: usize) {
    for (i, px) in img.as_slice_mut().iter_mut().enumerate() {
        *px = i as f32 / n as f32;
    }
}

/// Check whether a unified image can actually reach a kornia CUDA kernel
/// through the normal public API. Allocation working is not the same as the
/// operation dispatching, and that difference is what decides whether the
/// feature is usable today.
fn device_dispatch_probe(ctx: &Arc<CudaContext>) -> Result<(), Box<dyn std::error::Error>> {
    use kornia_imgproc::interpolation::InterpolationMode;
    use kornia_imgproc::resize::resize;

    let src_size = ImageSize {
        width: 64,
        height: 48,
    };
    let dst_size = ImageSize {
        width: 32,
        height: 24,
    };

    let src = Image::<f32, 3>::zeros_unified(src_size, ctx)?;
    let mut dst = Image::<f32, 3>::zeros_unified(dst_size, ctx)?;

    print!("\nresize() on unified images: ");
    match resize(&src, &mut dst, InterpolationMode::Bilinear) {
        Ok(()) => println!("dispatched"),
        Err(e) => println!("FAILED — {e}"),
    }

    Ok(())
}
