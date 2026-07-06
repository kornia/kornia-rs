//! Micro-benchmark: GPU warp-affine vs CPU reference.
//!
//! Tests a 45° centre rotation across three image sizes, measuring throughput
//! for bilinear and nearest-neighbor interpolation on both GPU and CPU paths.
//!
//! ```text
//! cargo run --example bench_gpu_warp_affine --features gpu-cuda --release
//! ```

use kornia_image::{Image, ImageSize};
use kornia_imgproc::{
    interpolation::InterpolationMode,
    warp::{get_rotation_matrix2d, warp_affine},
};
use std::time::Instant;

const WARMUP: u32 = 50;
const ITERS: u32 = 200;

/// (width, height) test cases — same as bench_warp.rs.
const CASES: &[(u32, u32)] = &[(256, 224), (512, 448), (1024, 896), (1920, 1080)];

const NC: u32 = 3;

fn gb_per_sec(npix: usize, ms: f64) -> f64 {
    // 1 src read + 1 dst write per output pixel.
    (npix as f64 * NC as f64 * 8.0) / (ms * 1e-3) / 1e9
}

fn main() {
    #[cfg(feature = "gpu-cuda")]
    run_gpu_cuda();

    #[cfg(not(feature = "gpu-cuda"))]
    println!("(native CUDA section skipped — build with --features gpu-cuda to enable)");

    run_cpu();
}

// --------------------------------------------------------------------------
// GPU section
// --------------------------------------------------------------------------

#[cfg(feature = "gpu-cuda")]
fn run_gpu_cuda() {
    use cudarc::driver::CudaContext;
    use kornia_imgproc::gpu::warp_affine_cuda::{
        launch_warp_affine_bicubic_cuda, launch_warp_affine_bilinear_cuda,
        launch_warp_affine_nearest_cuda,
    };

    let ctx = std::sync::Arc::new(CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();

    for method in ["nearest", "bilinear", "bicubic"] {
        println!(
            "\n=== GPU warp-affine {method} (45° rotation, __ldg, 32×8 grid, {ITERS} iters) ==="
        );
        println!("  {:<20}  {:>10}  {:>10}", "case", "ms/iter", "GB/s");
        println!("  {}", "-".repeat(46));

        for &(w, h) in CASES {
            let npix = (w * h) as usize;
            let nc = NC as usize;

            let src_data: Vec<f32> = (0..npix * nc).map(|i| (i % 256) as f32 / 255.0).collect();

            let src_dev = stream.clone_htod(&src_data).expect("H→D copy");
            let mut dst_dev = stream.alloc_zeros::<f32>(npix * nc).expect("alloc dst");

            let m = get_rotation_matrix2d((w as f32 / 2.0, h as f32 / 2.0), 45.0, 1.0);

            let launch = |dst: &mut cudarc::driver::CudaSlice<f32>| match method {
                "nearest" => {
                    launch_warp_affine_nearest_cuda(&ctx, &stream, &src_dev, dst, w, h, w, h, &m, None)
                        .expect("nearest launch")
                }
                "bicubic" => {
                    launch_warp_affine_bicubic_cuda(&ctx, &stream, &src_dev, dst, w, h, w, h, &m)
                        .expect("bicubic launch")
                }
                _ => launch_warp_affine_bilinear_cuda(&ctx, &stream, &src_dev, dst, w, h, w, h, &m, None)
                    .expect("bilinear launch"),
            };

            for _ in 0..WARMUP {
                launch(&mut dst_dev);
            }
            stream.synchronize().expect("sync");

            let t = Instant::now();
            for _ in 0..ITERS {
                launch(&mut dst_dev);
            }
            stream.synchronize().expect("sync");
            let ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

            println!(
                "  {:<20}  {:>10.3}  {:>10.2}",
                format!("{w}×{h}"),
                ms,
                gb_per_sec(npix, ms),
            );
        }
    }
}

// --------------------------------------------------------------------------
// CPU section
// --------------------------------------------------------------------------

fn run_cpu() {
    for method in [InterpolationMode::Nearest, InterpolationMode::Bilinear] {
        let label = match method {
            InterpolationMode::Nearest => "nearest",
            _ => "bilinear",
        };
        println!("\n=== CPU warp-affine {label} (45° rotation, f32, kornia) ===");
        println!("  {:<20}  {:>10}  {:>10}", "case", "ms/iter", "GB/s");
        println!("  {}", "-".repeat(46));

        for &(w, h) in CASES {
            let npix = (w * h) as usize;
            let nc = NC as usize;

            let src_data: Vec<f32> = (0..npix * nc).map(|i| (i % 256) as f32 / 255.0).collect();
            let src = Image::<f32, 3>::new(
                ImageSize {
                    width: w as usize,
                    height: h as usize,
                },
                src_data,
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

            let m = get_rotation_matrix2d((w as f32 / 2.0, h as f32 / 2.0), 45.0, 1.0);

            for _ in 0..WARMUP {
                warp_affine(&src, &mut dst, &m, method).expect("warp");
                std::hint::black_box(dst.as_slice());
            }
            let t = Instant::now();
            for _ in 0..ITERS {
                warp_affine(&src, &mut dst, &m, method).expect("warp");
                std::hint::black_box(dst.as_slice());
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

            println!(
                "  {:<20}  {:>10.3}  {:>10.2}",
                format!("{w}×{h}"),
                ms,
                gb_per_sec(npix, ms),
            );
        }
    }
}
