//! Micro-benchmark: CUDA remap vs fused warp-affine.
//!
//! **Purpose:** Determine whether remap is fast enough to serve as the base
//! primitive for warp-perspective (and other warps), or whether a fused
//! inline-homography kernel is needed.
//!
//! **Decision rule (from mentor):**
//! * Remap within ~10% of fused → use remap as base (less code duplication).
//! * Remap significantly slower (extra map-read overhead) → separate fused kernels.
//!
//! **What is measured:**
//! 1. `remap_bilinear`   — reads map_x/map_y from device, then bilinear sample.
//! 2. `warp_affine_bilinear` — computes source coord from matrix (no map reads).
//! Both perform the same 45° rotation so the sampling work is identical.
//!
//! **Timing methodology:**
//! Queue all ITERS launches then block with a device-to-host scalar read.
//!
//! ```text
//! cargo run --example bench_cuda_remap --features cuda --release
//! ```

use std::time::Instant;

fn print_comparison_row(w: u32, h: u32, remap_ms: f64, affine_ms: f64) {
    let overhead_pct = (remap_ms - affine_ms) / affine_ms * 100.0;
    let decision = if overhead_pct.abs() <= 10.0 {
        "USE REMAP"
    } else {
        "FUSED FASTER"
    };
    println!(
        "  {:<20}  {:>12.3}  {:>12.3}  {:>11.1}%  {:>10}",
        format!("{w}×{h}"),
        remap_ms,
        affine_ms,
        overhead_pct,
        decision,
    );
}

const WARMUP: u32 = 50;
const ITERS: u32 = 200;
const NC: u32 = 3;

/// Test cases: (width, height) — same image used as both src and dst.
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
    use kornia_imgproc::cuda::remap::{launch_remap_bilinear_cuda, launch_remap_nearest_cuda};
    use kornia_imgproc::cuda::warp_affine::{
        launch_warp_affine_bilinear_cuda, launch_warp_affine_nearest_cuda,
    };
    use kornia_imgproc::warp::{get_rotation_matrix2d, invert_affine_transform};

    // ── Local map-generation helpers ──────────────────────────────────────────
    // These helpers serve the architecture-decision benchmark only.  The library
    // surface is the pure remap launchers; callers that need affine/homography
    // maps for production use should generate them from the fused warp kernels.

    fn maps_from_affine(m: &[f32; 6], dst_w: u32, dst_h: u32) -> (Vec<f32>, Vec<f32>) {
        let mi = invert_affine_transform(m);
        let npix = (dst_w * dst_h) as usize;
        let mut mx = Vec::with_capacity(npix);
        let mut my = Vec::with_capacity(npix);
        for dy in 0..dst_h {
            let base_sx = mi[1] * dy as f32 + mi[2];
            let base_sy = mi[4] * dy as f32 + mi[5];
            for dx in 0..dst_w {
                mx.push(mi[0] * dx as f32 + base_sx);
                my.push(mi[3] * dx as f32 + base_sy);
            }
        }
        (mx, my)
    }

    fn maps_from_homography(h: &[f32; 9], dst_w: u32, dst_h: u32) -> (Vec<f32>, Vec<f32>) {
        // Standard 3×3 cofactor inverse; scale-invariant singularity guard.
        let [h00, h01, h02, h10, h11, h12, h20, h21, h22] = *h;
        let c00 = h11 * h22 - h12 * h21;
        let c01 = h12 * h20 - h10 * h22;
        let c02 = h10 * h21 - h11 * h20;
        let c10 = h02 * h21 - h01 * h22;
        let c11 = h00 * h22 - h02 * h20;
        let c12 = h01 * h20 - h00 * h21;
        let c20 = h01 * h12 - h02 * h11;
        let c21 = h02 * h10 - h00 * h12;
        let c22 = h00 * h11 - h01 * h10;
        let det = h00 * c00 + h01 * c01 + h02 * c02;
        let scale = h22.abs().powi(3).max(1e-30_f32);
        if det.abs() < 1e-7 * scale {
            return (
                vec![-1.0; (dst_w * dst_h) as usize],
                vec![-1.0; (dst_w * dst_h) as usize],
            );
        }
        let inv_det = 1.0 / det;
        let inv = [
            c00 * inv_det,
            c10 * inv_det,
            c20 * inv_det,
            c01 * inv_det,
            c11 * inv_det,
            c21 * inv_det,
            c02 * inv_det,
            c12 * inv_det,
            c22 * inv_det,
        ];
        let npix = (dst_w * dst_h) as usize;
        let mut mx = Vec::with_capacity(npix);
        let mut my = Vec::with_capacity(npix);
        for dy in 0..dst_h {
            let y = dy as f32;
            let row_sx = inv[1] * y + inv[2];
            let row_sy = inv[4] * y + inv[5];
            let row_w = inv[7] * y + inv[8];
            for dx in 0..dst_w {
                let x = dx as f32;
                let w = inv[6] * x + row_w;
                if w < 1e-10 {
                    mx.push(-1.0);
                    my.push(-1.0);
                } else {
                    let inv_w = 1.0 / w;
                    mx.push((inv[0] * x + row_sx) * inv_w);
                    my.push((inv[3] * x + row_sy) * inv_w);
                }
            }
        }
        (mx, my)
    }

    let ctx = Arc::new(CudaContext::new(0).expect("CUDA device 0"));
    let stream = ctx.default_stream();

    println!("\n=== CUDA remap vs warp-affine — architecture decision benchmark ===");
    println!("  (45° rotation, bilinear, {ITERS} iters)");
    println!(
        "\n  {:<20}  {:>12}  {:>12}  {:>12}  {:>10}",
        "case (W×H)", "remap-BL ms", "affine-BL ms", "overhead", "decision"
    );
    println!("  {}", "-".repeat(74));

    for &(w, h) in CASES {
        let npix = (w * h) as usize;
        let src_data: Vec<f32> = (0..npix * NC as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();

        // 45° rotation around image centre.
        let center = (w as f32 / 2.0, h as f32 / 2.0);
        let m = get_rotation_matrix2d(center, 45.0_f32.to_radians(), 1.0);

        // Precompute maps (CPU side — done once, not timed).
        let (host_mx, host_my) = maps_from_affine(&m, w, h);

        let src_dev = stream.clone_htod(&src_data).expect("H→D src");
        let map_x_dev = stream.clone_htod(&host_mx).expect("H→D map_x");
        let map_y_dev = stream.clone_htod(&host_my).expect("H→D map_y");
        let mut dst_remap = stream
            .alloc_zeros::<f32>(npix * NC as usize)
            .expect("alloc remap dst");
        let mut dst_affine = stream
            .alloc_zeros::<f32>(npix * NC as usize)
            .expect("alloc affine dst");

        // ── Warmup ──────────────────────────────────────────────────────────
        for _ in 0..WARMUP {
            launch_remap_bilinear_cuda(
                &ctx,
                &stream,
                &src_dev,
                &map_x_dev,
                &map_y_dev,
                &mut dst_remap,
                w,
                h,
                w,
                h,
                None,
            )
            .expect("remap bilinear");
            launch_warp_affine_bilinear_cuda(
                &ctx,
                &stream,
                &src_dev,
                &mut dst_affine,
                w,
                h,
                w,
                h,
                &m,
                None,
            )
            .expect("affine bilinear");
        }
        stream.synchronize().expect("sync");

        // ── Time remap ──────────────────────────────────────────────────────
        let t = Instant::now();
        for _ in 0..ITERS {
            launch_remap_bilinear_cuda(
                &ctx,
                &stream,
                &src_dev,
                &map_x_dev,
                &map_y_dev,
                &mut dst_remap,
                w,
                h,
                w,
                h,
                None,
            )
            .expect("remap bilinear");
        }
        stream.synchronize().expect("sync");
        let remap_ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        // ── Time warp-affine ────────────────────────────────────────────────
        let t = Instant::now();
        for _ in 0..ITERS {
            launch_warp_affine_bilinear_cuda(
                &ctx,
                &stream,
                &src_dev,
                &mut dst_affine,
                w,
                h,
                w,
                h,
                &m,
                None,
            )
            .expect("affine bilinear");
        }
        stream.synchronize().expect("sync");
        let affine_ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        print_comparison_row(w, h, remap_ms, affine_ms);
    }

    // ── Nearest-neighbor comparison ──────────────────────────────────────────
    println!(
        "\n  {:<20}  {:>12}  {:>12}  {:>12}  {:>10}",
        "case (W×H)", "remap-NN ms", "affine-NN ms", "overhead", "decision"
    );
    println!("  {}", "-".repeat(74));

    for &(w, h) in CASES {
        let npix = (w * h) as usize;
        let src_data: Vec<f32> = (0..npix * NC as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let center = (w as f32 / 2.0, h as f32 / 2.0);
        let m = get_rotation_matrix2d(center, 45.0_f32.to_radians(), 1.0);
        let (host_mx, host_my) = maps_from_affine(&m, w, h);

        let src_dev = stream.clone_htod(&src_data).expect("H→D src");
        let map_x_dev = stream.clone_htod(&host_mx).expect("H→D map_x");
        let map_y_dev = stream.clone_htod(&host_my).expect("H→D map_y");
        let mut dst_remap = stream
            .alloc_zeros::<f32>(npix * NC as usize)
            .expect("alloc");
        let mut dst_affine = stream
            .alloc_zeros::<f32>(npix * NC as usize)
            .expect("alloc");

        for _ in 0..WARMUP {
            launch_remap_nearest_cuda(
                &ctx,
                &stream,
                &src_dev,
                &map_x_dev,
                &map_y_dev,
                &mut dst_remap,
                w,
                h,
                w,
                h,
                None,
            )
            .expect("remap nearest");
            launch_warp_affine_nearest_cuda(
                &ctx,
                &stream,
                &src_dev,
                &mut dst_affine,
                w,
                h,
                w,
                h,
                &m,
                None,
            )
            .expect("affine nearest");
        }
        stream.synchronize().expect("sync");

        let t = Instant::now();
        for _ in 0..ITERS {
            launch_remap_nearest_cuda(
                &ctx,
                &stream,
                &src_dev,
                &map_x_dev,
                &map_y_dev,
                &mut dst_remap,
                w,
                h,
                w,
                h,
                None,
            )
            .expect("remap nearest");
        }
        stream.synchronize().expect("sync");
        let remap_ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        let t = Instant::now();
        for _ in 0..ITERS {
            launch_warp_affine_nearest_cuda(
                &ctx,
                &stream,
                &src_dev,
                &mut dst_affine,
                w,
                h,
                w,
                h,
                &m,
                None,
            )
            .expect("affine nearest");
        }
        stream.synchronize().expect("sync");
        let affine_ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        print_comparison_row(w, h, remap_ms, affine_ms);
    }

    // ── Homography (perspective) remap ───────────────────────────────────────
    println!("\n=== Remap with homography map (perspective tilt) — {ITERS} iters ===");
    println!("  {:<20}  {:>12}", "case (W×H)", "remap-BL ms");
    println!("  {}", "-".repeat(36));

    for &(w, h) in CASES {
        let npix = (w * h) as usize;
        let src_data: Vec<f32> = (0..npix * NC as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();

        // Simple perspective tilt: top edge compressed by 20%.
        let s = 0.2_f32;
        let hw = w as f32;
        let hh = h as f32;
        #[rustfmt::skip]
        let homo: [f32; 9] = [
            1.0,  s / hh,  0.0,
            0.0,  1.0,     0.0,
            0.0,  s / (hw * hh),  1.0,
        ];

        let (host_mx, host_my) = maps_from_homography(&homo, w, h);
        let src_dev = stream.clone_htod(&src_data).expect("H→D src");
        let map_x_dev = stream.clone_htod(&host_mx).expect("H→D map_x");
        let map_y_dev = stream.clone_htod(&host_my).expect("H→D map_y");
        let mut dst_dev = stream
            .alloc_zeros::<f32>(npix * NC as usize)
            .expect("alloc");

        for _ in 0..WARMUP {
            launch_remap_bilinear_cuda(
                &ctx,
                &stream,
                &src_dev,
                &map_x_dev,
                &map_y_dev,
                &mut dst_dev,
                w,
                h,
                w,
                h,
                None,
            )
            .expect("remap bilinear");
        }
        stream.synchronize().expect("sync");

        let t = Instant::now();
        for _ in 0..ITERS {
            launch_remap_bilinear_cuda(
                &ctx,
                &stream,
                &src_dev,
                &map_x_dev,
                &map_y_dev,
                &mut dst_dev,
                w,
                h,
                w,
                h,
                None,
            )
            .expect("remap bilinear");
        }
        stream.synchronize().expect("sync");
        let ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        println!("  {:<20}  {:>12.3}", format!("{w}×{h}"), ms);
    }

    println!("\nDecision guide: overhead ≤ 10% → use remap as the warp-perspective base.");
}
