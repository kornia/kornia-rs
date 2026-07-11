//! Micro-benchmark: CUDA warp-perspective (bilinear, nearest, bicubic, Lanczos-3).
//!
//! **Purpose:** Measure throughput of all four warp-perspective kernels across
//! typical image sizes, and compare each against the equivalent warp-affine mode.
//!
//! **What is measured:**
//! 1. `warp_perspective_bilinear`  — texture-backed bilinear, 3-ch f32.
//! 2. `warp_perspective_nearest`   — texture-backed nearest-neighbor, 3-ch f32.
//! 3. `warp_perspective_bicubic`   — Keys a=−0.5 via `__ldg`, 3-ch f32.
//! 4. `warp_perspective_lanczos`   — Lanczos-3 (6×6 taps) via `__ldg`, 3-ch f32.
//! All use a perspective-tilt homography so the homogeneous w-divide is exercised
//! on every pixel.
//!
//! ```text
//! cargo run --example bench_cuda_warp_perspective --features cuda --release
//! ```

use std::time::Instant;

const WARMUP: u32 = 50;
const ITERS: u32 = 200;
const NC: u32 = 3;

const CASES: &[(u32, u32)] = &[(512, 512), (1280, 720), (1920, 1080), (3840, 2160)];

fn main() {
    #[cfg(feature = "cuda")]
    run_benchmark();

    #[cfg(not(feature = "cuda"))]
    println!("(build with --features cuda to run the benchmark)");
}

#[cfg(feature = "cuda")]
fn run_benchmark() {
    use std::sync::Arc;

    use cudarc::driver::CudaContext;
    use kornia_imgproc::cuda::warp_perspective::{
        launch_warp_perspective_bicubic_cuda, launch_warp_perspective_bilinear_cuda,
        launch_warp_perspective_lanczos_cuda, launch_warp_perspective_nearest_cuda,
    };

    let ctx = Arc::new(CudaContext::new(0).expect("CUDA device 0"));
    let stream = ctx.default_stream();

    // Projective tilt: w varies from 1.0 to ~1.2 across the image.
    fn tilt_homography(w: u32, h: u32) -> [f32; 9] {
        let fw = w as f32;
        let fh = h as f32;
        let s = 0.2_f32;
        [1.0, s / fh, 0.0, 0.0, 1.0, 0.0, 0.0, s / (fw * fh), 1.0]
    }

    println!("\n=== CUDA warp-perspective benchmark ({ITERS} iters, perspective-tilt H) ===\n");

    for method in ["nearest", "bilinear", "bicubic", "lanczos"] {
        println!("  [{method}]");
        println!("  {:<20}  {:>12}  {:>12}", "case (W×H)", "ms/frame", "GB/s");
        println!("  {}", "-".repeat(48));

        for &(w, h) in CASES {
            let npix = (w * h) as usize;
            let src_data: Vec<f32> = (0..npix * NC as usize)
                .map(|i| (i % 256) as f32 / 255.0)
                .collect();

            let homo = tilt_homography(w, h);
            let src_dev = stream.clone_htod(&src_data).expect("H→D src");
            let mut dst_dev = stream
                .alloc_zeros::<f32>(npix * NC as usize)
                .expect("alloc dst");

            macro_rules! launch {
                () => {
                    match method {
                        "bilinear" => launch_warp_perspective_bilinear_cuda(
                            &ctx,
                            &stream,
                            &src_dev,
                            &mut dst_dev,
                            w,
                            h,
                            w,
                            h,
                            &homo,
                            None,
                        )
                        .expect("bilinear"),
                        "nearest" => launch_warp_perspective_nearest_cuda(
                            &ctx,
                            &stream,
                            &src_dev,
                            &mut dst_dev,
                            w,
                            h,
                            w,
                            h,
                            &homo,
                            None,
                        )
                        .expect("nearest"),
                        "bicubic" => launch_warp_perspective_bicubic_cuda(
                            &ctx,
                            &stream,
                            &src_dev,
                            &mut dst_dev,
                            w,
                            h,
                            w,
                            h,
                            &homo,
                            None,
                        )
                        .expect("bicubic"),
                        "lanczos" => launch_warp_perspective_lanczos_cuda(
                            &ctx,
                            &stream,
                            &src_dev,
                            &mut dst_dev,
                            w,
                            h,
                            w,
                            h,
                            &homo,
                            None,
                        )
                        .expect("lanczos"),
                        _ => unreachable!(),
                    }
                };
            }

            for _ in 0..WARMUP {
                launch!();
            }
            stream.synchronize().expect("sync");

            let t = Instant::now();
            for _ in 0..ITERS {
                launch!();
            }
            stream.synchronize().expect("sync");
            let ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

            // 1 src read + 1 dst write per output pixel, 3 channels, 4 bytes each.
            let gb_s = (npix as f64 * NC as f64 * 8.0) / (ms * 1e-3) / 1e9;

            println!("  {:<20}  {:>12.3}  {:>12.2}", format!("{w}×{h}"), ms, gb_s,);
        }
        println!();
    }

    println!("  Note: nearest/bilinear use texture objects; bicubic/lanczos use __ldg on raw src.");
    println!("  Bicubic/Lanczos OOB taps clamped (BORDER_REPLICATE); OOB centre → 0.");
}
