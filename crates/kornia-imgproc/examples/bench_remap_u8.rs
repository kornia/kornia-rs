//! u8 remap vs the f32 remap kernel.
//!
//! The u8 kernel exists so a u8 image can be remapped without widening to f32
//! first. Its case is bandwidth: remap is memory-bound, and u8 moves a quarter
//! of the bytes per channel, so the kernel should approach a 4x win on the
//! image traffic — diluted by the two f32 coordinate maps, which are the same
//! size either way and quickly dominate.
//!
//! Both paths are timed through the public API on device-resident images, so
//! this measures the kernels as callers actually reach them.
//!
//! There is deliberately no "convert to f32, remap, convert back" column: the
//! only `cast_and_scale` in the tree is CPU-side, so that alternative cannot be
//! measured on-device at all. Which is itself the argument for this kernel — a
//! u8 pipeline would otherwise have to round-trip through the host to use the
//! f32 remap.
//!
//! ```text
//! cargo run --release -p kornia-imgproc --features cuda --example bench_remap_u8
//! ```

use std::sync::Arc;
use std::time::Instant;

use argh::FromArgs;
use cudarc::driver::{CudaContext, CudaStream};
use kornia_image::{Image, ImageError, ImageSize};
use kornia_imgproc::interpolation::{remap, remap_u8, InterpolationMode};

/// Benchmark the u8 remap kernel against the f32 one.
#[derive(FromArgs)]
struct Args {
    /// CUDA device ordinal (default 0)
    #[argh(option, default = "0")]
    device: usize,

    /// timed iterations per case (default 100)
    #[argh(option, default = "100")]
    iters: usize,

    /// untimed warmup iterations per case (default 20)
    #[argh(option, default = "20")]
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
///
/// The launches are asynchronous, so the stream is synchronized inside the
/// timed region — the number covers the kernels, not just the launch calls.
fn timed<F>(
    stream: &Arc<CudaStream>,
    warmup: usize,
    iters: usize,
    mut f: F,
) -> Result<f64, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), ImageError>,
{
    for _ in 0..warmup {
        f()?;
    }
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    stream.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

/// A rotation map with fractional coordinates, so the sampler does real
/// bilinear work rather than an aligned copy.
fn rotation_maps(w: usize, h: usize) -> Result<(Image<f32, 1>, Image<f32, 1>), ImageError> {
    let size = ImageSize {
        width: w,
        height: h,
    };
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let (sin, cos) = (0.25f32).sin_cos();

    let mut mx = Vec::with_capacity(w * h);
    let mut my = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            mx.push(cx + dx * cos - dy * sin);
            my.push(cy + dx * sin + dy * cos);
        }
    }
    Ok((
        Image::<f32, 1>::new(size, mx)?,
        Image::<f32, 1>::new(size, my)?,
    ))
}

/// Bytes the kernel must move for one call: source reads (4 taps for bilinear,
/// 1 for nearest), the destination write, and the two f32 maps.
fn traffic_mib(w: usize, h: usize, px: usize, taps: usize) -> f64 {
    let pixels = w * h;
    let image = pixels * 3 * px * (taps + 1);
    let maps = pixels * 2 * 4;
    (image + maps) as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    let ctx = CudaContext::new(args.device)?;
    let stream = ctx.default_stream();

    let cc_major = ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    )?;
    let cc_minor = ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    )?;
    println!("device {}: sm_{cc_major}{cc_minor}", args.device);
    println!("3 channels, device-resident, via the public remap/remap_u8 API");
    println!("iters: {} (warmup {})", args.iters, args.warmup);

    for (mode, taps) in [
        (InterpolationMode::Bilinear, 4usize),
        (InterpolationMode::Nearest, 1usize),
    ] {
        println!("\n── {mode:?} ──────────────────────────────────────────────");
        println!(
            "{:<16} {:>10} {:>10} {:>9} {:>11} {:>11}",
            "size", "u8 ms", "f32 ms", "speedup", "u8 GiB/s", "f32 GiB/s"
        );
        println!("{}", "-".repeat(72));

        for &(label, w, h) in SIZES {
            let size = ImageSize {
                width: w,
                height: h,
            };
            let (map_x, map_y) = rotation_maps(w, h)?;
            let d_mx = map_x.to_cuda(&stream)?;
            let d_my = map_y.to_cuda(&stream)?;

            let u8_data: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
            let u8_src = Image::<u8, 3>::new(size, u8_data)?.to_cuda(&stream)?;
            let mut u8_dst = Image::<u8, 3>::zeros_cuda(size, &stream)?;
            let t_u8 = timed(&stream, args.warmup, args.iters, || {
                remap_u8(&u8_src, &mut u8_dst, &d_mx, &d_my, mode)
            })?;

            let f32_data: Vec<f32> = (0..w * h * 3).map(|i| (i % 251) as f32 / 251.0).collect();
            let f32_src = Image::<f32, 3>::new(size, f32_data)?.to_cuda(&stream)?;
            let mut f32_dst = Image::<f32, 3>::zeros_cuda(size, &stream)?;
            let t_f32 = timed(&stream, args.warmup, args.iters, || {
                remap(&f32_src, &mut f32_dst, &d_mx, &d_my, mode)
            })?;

            let gib = |mib: f64, ms: f64| mib / 1024.0 / (ms / 1e3);
            println!(
                "{label:<16} {t_u8:>10.4} {t_f32:>10.4} {:>8.2}x {:>11.1} {:>11.1}",
                t_f32 / t_u8,
                gib(traffic_mib(w, h, 1, taps), t_u8),
                gib(traffic_mib(w, h, 4, taps), t_f32),
            );
        }
    }

    println!(
        "\nGiB/s counts source taps + destination + both f32 maps. The maps are\n\
         f32 in either case, so they cap how much the narrower pixel type can win."
    );

    Ok(())
}
