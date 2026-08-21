//! Unified (managed) memory vs. explicit H2D/D2H copies.
//!
//! The case for [`kornia_tensor::zeros_cuda_unified`] / `Image::zeros_cuda_unified` is
//! that on an **integrated** GPU the CPU and GPU share one physical memory, so
//! an explicit copy moves bytes from RAM back to the same RAM. This benchmark
//! measures what that actually costs, so the claim can be checked on the target
//! platform instead of assumed.
//!
//! The two paths are not a single number each — they trade a *recurring* cost
//! for a *one-time* one, so each component is timed separately:
//!
//! * `alloc` — one-time, per buffer: `zeros_cuda` vs `zeros_cuda_unified`.
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
use cudarc::driver::{CudaContext, CudaStream};
use kornia_image::{Image, ImageError, ImageSize};
use kornia_tensor::cuda::zeros_pinned_wc;
use rayon::prelude::*;

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
/// Uses Rayon to saturate CPU memory bandwidth.
fn fill_ramp(buf: &mut [f32]) {
    let inv = 1.0 / buf.len() as f32;
    // Chunking to balance Rayon overhead vs memory write speed
    let chunk_size = 4096;
    buf.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let offset = chunk_idx * chunk_size;
            for (i, px) in chunk.iter_mut().enumerate() {
                *px = (offset + i) as f32 * inv;
            }
        });
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
    // Whether the CPU may touch managed memory while *any* kernel is running.
    // Where this is 0, host access during device work faults outright, and every
    // read has to be ordered after all device work — not just the stream that
    // produced it.
    let concurrent_managed = ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
    )?;
    println!(
        "concurrent managed access: {} ({})",
        concurrent_managed,
        if concurrent_managed == 1 {
            "CPU may touch managed memory while kernels run"
        } else {
            "CPU must wait for all device work before touching managed memory"
        }
    );
    println!("iters: {} (warmup {})", args.iters, args.warmup);

    println!(
        "\n── one-time, per buffer ─────────────────────────────────────────────────────────────"
    );
    println!(
        "{:<16} {:>9} {:>14} {:>14} {:>14}",
        "size", "MiB", "zeros_cuda ms", "unified ms", "pinned_wc ms"
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
            let u = Image::<f32, 3>::zeros_cuda_unified(size, &stream)?;
            std::hint::black_box(&u);
            Ok(())
        })?;
        let pinned = timed(args.warmup, args.iters, || {
            let p = Image::<f32, 3>(
                zeros_pinned_wc([size.height, size.width, 3], stream.context())
                    .map_err(|e| ImageError::Cuda(e.to_string()))?,
            );
            std::hint::black_box(&p);
            Ok(())
        })?;

        println!("{label:<16} {mib:>9.1} {dev:>14.3} {uni:>14.3} {pinned:>14.3}");
    }

    println!("\n── per frame ─────────────────────────────────────────────────────────────────────────────────");
    println!(
        "{:<16} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "size", "fill", "fill_uni", "fill_wc", "H2D", "D2H", "explicit", "unified"
    );
    println!("{}", "-".repeat(95));

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
        let mut uni = Image::<f32, 3>::zeros_cuda_unified(size, &stream)?;
        let fill_uni = timed(args.warmup, args.iters, || {
            fill_ramp(uni.as_slice_mut());
            Ok(())
        })?;

        let mut pinned_wc = Image::<f32, 3>(
            zeros_pinned_wc([size.height, size.width, 3], stream.context())
                .map_err(|e| ImageError::Cuda(e.to_string()))?,
        );
        let fill_wc = timed(args.warmup, args.iters, || {
            fill_ramp(pinned_wc.as_slice_mut());
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
            "{label:<16} {fill:>8.3} {fill_uni:>9.3} {fill_wc:>9.3} {h2d:>9.3} {d2h:>9.3} \
             {explicit:>9.3} {unified:>9.3}"
        );
    }

    println!(
        "\n'explicit' = fill + H2D + D2H. 'unified' = fill_uni, with no transfer.\n\
         Unified wins where (fill_uni - fill) < (H2D + D2H)."
    );

    end_to_end_resize(&stream, args.warmup, args.iters)?;

    Ok(())
}

/// The measurement that decides the feature: a real `resize()` per frame, both
/// ways, through the ordinary public API.
///
/// * explicit — fill a host image, `to_cuda`, resize, `to_host_image`.
/// * unified — fill a unified image, resize into a unified destination. The CPU
///   reads the result directly; there is no transfer on either side.
///
/// The two are also checked against each other, because a faster path that
/// computes something different is worthless.
fn end_to_end_resize(
    stream: &Arc<CudaStream>,
    warmup: usize,
    iters: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use kornia_imgproc::interpolation::InterpolationMode;
    use kornia_imgproc::resize::resize;

    const MODE: InterpolationMode = InterpolationMode::Bilinear;

    println!("\n── per frame, end-to-end resize() ───────────────────────────────────────────────────────────");
    println!(
        "{:<16} {:>11} {:>11} {:>11} {:>9}",
        "size", "explicit ms", "unified ms", "pinned_wc ms", "speedup"
    );
    println!("{}", "-".repeat(80));

    for &(label, w, h) in SIZES {
        let src_size = ImageSize {
            width: w,
            height: h,
        };
        let dst_size = ImageSize {
            width: w / 2,
            height: h / 2,
        };

        // Explicit: host frame, uploaded and downloaded around the kernel.
        let mut host = Image::<f32, 3>::from_size_val(src_size, 0.0)?;
        let explicit = timed(warmup, iters, || {
            fill_ramp(host.as_slice_mut());
            let d_src = host.to_cuda(stream)?;
            let mut d_dst = Image::<f32, 3>::zeros_cuda(dst_size, stream)?;
            resize(&d_src, &mut d_dst, MODE)?;
            let back = d_dst.to_host_image(stream)?;
            std::hint::black_box(&back);
            Ok(())
        })?;

        // Unified: the same frame produced in place, resized, read in place.
        let mut u_src = Image::<f32, 3>::zeros_cuda_unified(src_size, stream)?;
        let mut u_dst = Image::<f32, 3>::zeros_cuda_unified(dst_size, stream)?;
        let unified = timed(warmup, iters, || {
            fill_ramp(u_src.as_slice_mut());
            resize(&u_src, &mut u_dst, MODE)?;
            stream
                .synchronize()
                .map_err(|e| ImageError::Cuda(e.to_string()))?;
            std::hint::black_box(u_dst.as_slice());
            Ok(())
        })?;

        let mut wc_src = Image::<f32, 3>(
            zeros_pinned_wc([src_size.height, src_size.width, 3], stream.context())
                .map_err(|e| ImageError::Cuda(e.to_string()))?,
        );
        let mut wc_dst = Image::<f32, 3>(
            zeros_pinned_wc([dst_size.height, dst_size.width, 3], stream.context())
                .map_err(|e| ImageError::Cuda(e.to_string()))?,
        );
        let pinned = timed(warmup, iters, || {
            fill_ramp(wc_src.as_slice_mut());
            // Since pinned is host memory, we must do DMA transfers to device.
            let d_src = wc_src.to_cuda(stream)?;
            let mut d_dst = Image::<f32, 3>::zeros_cuda(dst_size, stream)?;
            resize(&d_src, &mut d_dst, MODE)?;
            // DMA transfer back
            d_dst.to_host_into(wc_dst.as_slice_mut())?;
            // Wait for transfers
            stream
                .synchronize()
                .map_err(|e| ImageError::Cuda(e.to_string()))?;
            std::hint::black_box(wc_dst.as_slice());
            Ok(())
        })?;

        let best_opt = unified.min(pinned);
        println!(
            "{label:<16} {explicit:>11.3} {unified:>11.3} {pinned:>11.3} {:>8.2}x",
            explicit / best_opt
        );
    }

    // Correctness: the unified result must equal the explicit one bit for bit.
    let src_size = ImageSize {
        width: 64,
        height: 48,
    };
    let dst_size = ImageSize {
        width: 32,
        height: 24,
    };

    let mut host = Image::<f32, 3>::from_size_val(src_size, 0.0)?;
    fill_ramp(host.as_slice_mut());
    let d_src = host.to_cuda(stream)?;
    let mut d_dst = Image::<f32, 3>::zeros_cuda(dst_size, stream)?;
    resize(&d_src, &mut d_dst, MODE)?;
    let device_out = d_dst.to_host_image(stream)?;

    let mut u_src = Image::<f32, 3>::zeros_cuda_unified(src_size, stream)?;
    fill_ramp(u_src.as_slice_mut());
    let mut u_dst = Image::<f32, 3>::zeros_cuda_unified(dst_size, stream)?;
    resize(&u_src, &mut u_dst, MODE)?;
    stream.synchronize()?;

    let mismatches = device_out
        .as_slice()
        .iter()
        .zip(u_dst.as_slice())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!(
        "\nunified vs device output: {}",
        if mismatches == 0 {
            "bit-identical".to_string()
        } else {
            format!("{mismatches} mismatched elements")
        }
    );

    Ok(())
}
