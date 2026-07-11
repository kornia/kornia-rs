//! Micro-benchmark: CUDA warp-perspective (bilinear, nearest, bicubic).
//!
//! **Purpose:** Measure the throughput of the three warp-perspective kernels
//! across typical image sizes, and compare bicubic quality vs speed vs
//! warp-affine bilinear.
//!
//! **What is measured:**
//! 1. `warp_perspective_bilinear`  — texture-backed bilinear, 3-ch f32.
//! 2. `warp_perspective_nearest`   — texture-backed nearest-neighbor, 3-ch f32.
//! 3. `warp_perspective_bicubic`   — Keys a=-0.5 via `__ldg`, 3-ch f32.
//! 4. `warp_affine_bilinear`       — reference: same rotation, no perspective.
//! All use a perspective-tilt homography (top edge scaled to 80% of bottom)
//! so the homogeneous w-divide is exercised on every pixel.
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
    use kornia_imgproc::cuda::warp_affine::launch_warp_affine_bilinear_cuda;
    use kornia_imgproc::cuda::warp_perspective::{
        launch_warp_perspective_bicubic_cuda, launch_warp_perspective_bilinear_cuda,
        launch_warp_perspective_nearest_cuda,
    };
    use kornia_imgproc::warp::get_rotation_matrix2d;

    let ctx = Arc::new(CudaContext::new(0).expect("CUDA device 0"));
    let stream = ctx.default_stream();

    // Perspective-tilt homography: top edge at 80% width, bottom at full width.
    // h maps destination → source (forward homography; inverted internally).
    fn tilt_homography(w: u32, h: u32) -> [f32; 9] {
        let fw = w as f32;
        let fh = h as f32;
        // Projective tilt: H = [[1, s/fh, 0], [0, 1, 0], [0, s/(fw*fh), 1]], s=0.2.
        // Exercises the w-divide on every pixel (w varies from 1.0 to ~1.2 across the image).
        let s = 0.2_f32;
        [1.0, s / fh, 0.0, 0.0, 1.0, 0.0, 0.0, s / (fw * fh), 1.0]
    }

    println!("\n=== CUDA warp-perspective benchmark ({ITERS} iters, perspective-tilt H) ===\n");
    println!(
        "  {:<20}  {:>10}  {:>10}  {:>10}  {:>10}",
        "case (W×H)", "bilinear", "nearest", "bicubic", "affine-BL"
    );
    println!(
        "  {:<20}  {:>10}  {:>10}  {:>10}  {:>10}",
        "", "ms/frame", "ms/frame", "ms/frame", "ms/frame"
    );
    println!("  {}", "-".repeat(68));

    for &(w, h) in CASES {
        let npix = (w * h) as usize;
        let src_data: Vec<f32> = (0..npix * NC as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();

        let homo = tilt_homography(w, h);
        let center = (w as f32 / 2.0, h as f32 / 2.0);
        let affine_m = get_rotation_matrix2d(center, 0.0_f32, 1.0); // identity-ish for comparison

        let src_dev = stream.clone_htod(&src_data).expect("H→D src");
        let mut dst_bl = stream
            .alloc_zeros::<f32>(npix * NC as usize)
            .expect("alloc");
        let mut dst_nn = stream
            .alloc_zeros::<f32>(npix * NC as usize)
            .expect("alloc");
        let mut dst_bc = stream
            .alloc_zeros::<f32>(npix * NC as usize)
            .expect("alloc");
        let mut dst_aff = stream
            .alloc_zeros::<f32>(npix * NC as usize)
            .expect("alloc");

        // Warmup — compile kernels and warm caches.
        for _ in 0..WARMUP {
            launch_warp_perspective_bilinear_cuda(
                &ctx,
                &stream,
                &src_dev,
                &mut dst_bl,
                w,
                h,
                w,
                h,
                &homo,
                None,
            )
            .expect("persp bilinear");
            launch_warp_perspective_nearest_cuda(
                &ctx,
                &stream,
                &src_dev,
                &mut dst_nn,
                w,
                h,
                w,
                h,
                &homo,
                None,
            )
            .expect("persp nearest");
            launch_warp_perspective_bicubic_cuda(
                &ctx,
                &stream,
                &src_dev,
                &mut dst_bc,
                w,
                h,
                w,
                h,
                &homo,
                None,
            )
            .expect("persp bicubic");
            launch_warp_affine_bilinear_cuda(
                &ctx,
                &stream,
                &src_dev,
                &mut dst_aff,
                w,
                h,
                w,
                h,
                &affine_m,
                None,
            )
            .expect("affine bilinear");
        }
        stream.synchronize().expect("sync");

        // Time bilinear perspective.
        let t = Instant::now();
        for _ in 0..ITERS {
            launch_warp_perspective_bilinear_cuda(
                &ctx,
                &stream,
                &src_dev,
                &mut dst_bl,
                w,
                h,
                w,
                h,
                &homo,
                None,
            )
            .expect("persp bilinear");
        }
        stream.synchronize().expect("sync");
        let bl_ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        // Time nearest perspective.
        let t = Instant::now();
        for _ in 0..ITERS {
            launch_warp_perspective_nearest_cuda(
                &ctx,
                &stream,
                &src_dev,
                &mut dst_nn,
                w,
                h,
                w,
                h,
                &homo,
                None,
            )
            .expect("persp nearest");
        }
        stream.synchronize().expect("sync");
        let nn_ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        // Time bicubic perspective.
        let t = Instant::now();
        for _ in 0..ITERS {
            launch_warp_perspective_bicubic_cuda(
                &ctx,
                &stream,
                &src_dev,
                &mut dst_bc,
                w,
                h,
                w,
                h,
                &homo,
                None,
            )
            .expect("persp bicubic");
        }
        stream.synchronize().expect("sync");
        let bc_ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        // Time affine bilinear (reference).
        let t = Instant::now();
        for _ in 0..ITERS {
            launch_warp_affine_bilinear_cuda(
                &ctx,
                &stream,
                &src_dev,
                &mut dst_aff,
                w,
                h,
                w,
                h,
                &affine_m,
                None,
            )
            .expect("affine bilinear");
        }
        stream.synchronize().expect("sync");
        let aff_ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        println!(
            "  {:<20}  {:>10.3}  {:>10.3}  {:>10.3}  {:>10.3}",
            format!("{w}×{h}"),
            bl_ms,
            nn_ms,
            bc_ms,
            aff_ms,
        );
    }

    println!("\n  Note: bicubic uses __ldg on raw src (no texture), so OOB taps are clamped");
    println!("  (BORDER_REPLICATE on the 4×4 neighbourhood) while the OOB centre → 0.");
}
