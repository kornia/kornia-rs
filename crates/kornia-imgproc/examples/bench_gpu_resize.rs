//! Micro-benchmark: GPU nearest-neighbor and bilinear resize vs CPU reference.
//!
//! **Timing methodology:**
//! - GPU: queue all ITERS launches across N_BUFFS rotating source buffers,
//!   then block with `read_one_unchecked`.  The rotating buffers defeat GPU L2
//!   read cache across iterations.
//! - CPU: hand-rolled f32 bilinear loop (same algorithm, no SIMD).
//!
//! ```text
//! # CPU only (no CUDA needed)
//! cargo run --example bench_gpu_resize --release
//!
//! # CPU + GPU comparison
//! cargo run --example bench_gpu_resize --features gpu-cubecl --release
//! ```

use std::time::Instant;

const WARMUP: u32 = 50;
const ITERS: u32 = 200;
#[cfg(feature = "gpu-cubecl")]
const N_BUFFS: usize = 8;

/// (src_w, src_h, dst_w, dst_h) test cases.
const CASES: &[(u32, u32, u32, u32)] = &[
    (1024, 1024, 512, 512),   // 2× downscale
    (512, 512, 1024, 1024),   // 2× upscale
    (1920, 1080, 960, 540),   // 2× downscale 1080p→540p
    (1920, 1080, 3840, 2160), // 2× upscale 1080p→4K
    (3840, 2160, 1920, 1080), // 2× downscale 4K→1080p
];

const NC: u32 = 3; // RGB

// --------------------------------------------------------------------------
// CPU reference
// --------------------------------------------------------------------------

fn cpu_bilinear_3c(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
) {
    let scale_x = src_w as f64 / dst_w as f64;
    let scale_y = src_h as f64 / dst_h as f64;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = ((dx as f64 + 0.5) * scale_x - 0.5)
                .max(0.0)
                .min(src_w as f64 - 1.0);
            let sy = ((dy as f64 + 0.5) * scale_y - 0.5)
                .max(0.0)
                .min(src_h as f64 - 1.0);

            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);

            let fx = (sx - x0 as f64) as f32;
            let fy = (sy - y0 as f64) as f32;
            let w00 = (1.0 - fy) * (1.0 - fx);
            let w10 = (1.0 - fy) * fx;
            let w01 = fy * (1.0 - fx);
            let w11 = fy * fx;

            let dst_base = (dy * dst_w + dx) * 3;
            for c in 0..3_usize {
                dst[dst_base + c] = w00 * src[(y0 * src_w + x0) * 3 + c]
                    + w10 * src[(y0 * src_w + x1) * 3 + c]
                    + w01 * src[(y1 * src_w + x0) * 3 + c]
                    + w11 * src[(y1 * src_w + x1) * 3 + c];
            }
        }
    }
}

// --------------------------------------------------------------------------
// Bandwidth helper
// --------------------------------------------------------------------------

/// Effective memory bandwidth: 1 src read + 1 dst write per output pixel.
/// (NC channels × 4 B/f32 × 2 = NC × 8 B per pixel.)
/// Note: bilinear reads up to 4 src pixels per output pixel; cache hits for
/// upscale mean actual DRAM traffic is lower than this formula suggests.
fn gb_per_sec(npix_dst: usize, ms_per_iter: f64) -> f64 {
    (npix_dst as f64 * NC as f64 * 8.0) / (ms_per_iter * 1e-3) / 1e9
}

// --------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------

fn main() {
    #[cfg(feature = "gpu-cubecl")]
    run_gpu();

    #[cfg(not(feature = "gpu-cubecl"))]
    println!("(GPU section skipped — build with --features gpu-cubecl to enable)");

    run_cpu();
}

// --------------------------------------------------------------------------
// CPU section
// --------------------------------------------------------------------------

fn run_cpu() {
    println!("\n=== CPU bilinear (f32, 3-channel) ===");
    println!(
        "  {:<24}  {:>10}  {:>10}",
        "case (src→dst)", "ms/iter", "GB/s"
    );
    println!("  {}", "-".repeat(50));

    for &(sw, sh, dw, dh) in CASES {
        let npix_src = (sw * sh) as usize;
        let npix_dst = (dw * dh) as usize;
        let src: Vec<f32> = (0..npix_src * 3)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let mut dst = vec![0f32; npix_dst * 3];

        for _ in 0..WARMUP {
            cpu_bilinear_3c(
                &src,
                sw as usize,
                sh as usize,
                &mut dst,
                dw as usize,
                dh as usize,
            );
            std::hint::black_box(&dst);
        }
        let t = Instant::now();
        for _ in 0..ITERS {
            cpu_bilinear_3c(
                &src,
                sw as usize,
                sh as usize,
                &mut dst,
                dw as usize,
                dh as usize,
            );
            std::hint::black_box(&dst);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
        println!(
            "  {:<24}  {:>10.3}  {:>10.2}",
            format!("{sw}×{sh}→{dw}×{dh}"),
            ms,
            gb_per_sec(npix_dst, ms),
        );
    }
}

// --------------------------------------------------------------------------
// GPU section
// --------------------------------------------------------------------------

#[cfg(feature = "gpu-cubecl")]
fn run_gpu() {
    use cubecl::prelude::*;
    use cubecl_cuda::CudaRuntime;
    use kornia_imgproc::gpu::resize::{launch_resize_bilinear_f32, launch_resize_nearest_f32};

    fn f32_as_bytes(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) }
    }

    let device = <CudaRuntime as Runtime>::Device::default();
    let client = CudaRuntime::client(&device);

    for method in ["nearest", "bilinear"] {
        println!(
            "\n=== GPU {method} (CubeCL/CUDA, 3ch f32, {ITERS} iters, {N_BUFFS} rotating bufs) ==="
        );
        println!(
            "  {:<24}  {:>10}  {:>10}  {:>10}",
            "case (src→dst)", "ms/iter", "GB/s", "CPU BL ms"
        );
        println!("  {}", "-".repeat(60));

        for &(sw, sh, dw, dh) in CASES {
            let npix_src = (sw * sh) as usize;
            let npix_dst = (dw * dh) as usize;
            let nc = NC as usize;

            let src_bufs: Vec<_> = (0..N_BUFFS)
                .map(|b| {
                    let data: Vec<f32> = (0..npix_src * nc)
                        .map(|i| ((i + b * 17) % 256) as f32 / 255.0)
                        .collect();
                    client.create_from_slice(f32_as_bytes(&data))
                })
                .collect();

            let time_kernel = |launch: &dyn Fn(cubecl::server::Handle, cubecl::server::Handle)| {
                let dst = client.empty(npix_dst * nc * 4);
                for i in 0..WARMUP {
                    launch(src_bufs[i as usize % N_BUFFS].clone(), dst.clone());
                }
                let _ = client.read_one_unchecked(dst.clone());
                let dst2 = client.empty(npix_dst * nc * 4);
                let t = Instant::now();
                for i in 0..ITERS {
                    launch(src_bufs[i as usize % N_BUFFS].clone(), dst2.clone());
                }
                let _ = client.read_one_unchecked(dst2);
                t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
            };

            let gpu_ms = time_kernel(&|s, d| match method {
                "nearest" => {
                    launch_resize_nearest_f32::<CudaRuntime>(&client, s, d, sw, sh, dw, dh, NC)
                }
                _ => launch_resize_bilinear_f32::<CudaRuntime>(&client, s, d, sw, sh, dw, dh, NC),
            });

            let src_f32: Vec<f32> = (0..npix_src * nc)
                .map(|i| (i % 256) as f32 / 255.0)
                .collect();
            let mut dst_cpu = vec![0f32; npix_dst * nc];
            for _ in 0..WARMUP {
                cpu_bilinear_3c(
                    &src_f32,
                    sw as usize,
                    sh as usize,
                    &mut dst_cpu,
                    dw as usize,
                    dh as usize,
                );
                std::hint::black_box(&dst_cpu);
            }
            let t1 = Instant::now();
            for _ in 0..ITERS {
                cpu_bilinear_3c(
                    &src_f32,
                    sw as usize,
                    sh as usize,
                    &mut dst_cpu,
                    dw as usize,
                    dh as usize,
                );
                std::hint::black_box(&dst_cpu);
            }
            let cpu_ms = t1.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

            println!(
                "  {:<24}  {:>10.3}  {:>10.2}  {:>10.3}",
                format!("{sw}×{sh}→{dw}×{dh}"),
                gpu_ms,
                gb_per_sec(npix_dst, gpu_ms),
                cpu_ms,
            );
        }
    }
}
