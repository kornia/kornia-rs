//! Micro-benchmark: GPU `gray_from_rgb_f32` kernel vs CPU scalar baseline.
//!
//! Measures pure kernel time (no host↔device transfers) by pre-uploading all
//! buffers, queueing ITERS launches, then reading back once to force a sync.
//!
//! Run with:
//!
//! ```text
//! cargo run --example bench_gpu_color --features gpu-cubecl --release
//! ```

#[cfg(feature = "gpu-cubecl")]
mod inner {
    use std::time::Instant;

    use cubecl::prelude::*;
    use cubecl_cuda::CudaRuntime;
    use kornia_imgproc::gpu::color::launch_gray_from_rgb_f32;

    const WARMUP: u32 = 20;
    const ITERS: u32 = 200;

    const SIZES: &[(u32, u32)] = &[(512, 512), (1024, 1024), (1920, 1080), (3840, 2160)];

    fn f32_as_bytes(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) }
    }

    fn cpu_gray_from_rgb(src: &[f32], dst: &mut [f32]) {
        const RW: f32 = 0.299;
        const GW: f32 = 0.587;
        const BW: f32 = 0.114;
        for (i, d) in dst.iter_mut().enumerate() {
            let b = i * 3;
            *d = RW * src[b] + GW * src[b + 1] + BW * src[b + 2];
        }
    }

    pub fn run() {
        let device = <CudaRuntime as Runtime>::Device::default();
        let client = CudaRuntime::client(&device);

        println!(
            "  {:<16}  {:>9}  {:>11}  {:>12}  {:>11}  {:>11}",
            "size", "GPU ms", "GPU ns/px", "GPU GB/s", "CPU ms", "speedup"
        );
        println!("  {}", "-".repeat(77));

        for &(w, h) in SIZES {
            let npix = (w * h) as usize;

            // --- host data ---
            let src_f32: Vec<f32> =
                (0..npix * 3).map(|i| (i % 256) as f32 / 255.0).collect();
            let mut dst_cpu = vec![0f32; npix];

            // --- GPU buffers (uploaded once) ---
            let src_gpu = client.create_from_slice(f32_as_bytes(&src_f32));
            let dst_gpu = client.empty(npix * 4);

            // warm up GPU
            for _ in 0..WARMUP {
                launch_gray_from_rgb_f32(&client, src_gpu.clone(), dst_gpu.clone(), w, h);
            }
            // sync after warmup
            let _ = client.read_one_unchecked(dst_gpu.clone());

            // --- timed GPU run ---
            let t0 = Instant::now();
            for _ in 0..ITERS {
                launch_gray_from_rgb_f32(&client, src_gpu.clone(), dst_gpu.clone(), w, h);
            }
            let _ = client.read_one_unchecked(dst_gpu); // forces sync
            let gpu_elapsed = t0.elapsed();

            let gpu_ms = gpu_elapsed.as_secs_f64() * 1e3 / ITERS as f64;
            let gpu_ns_px = gpu_elapsed.as_nanos() as f64 / (ITERS as f64 * npix as f64);
            // 3 reads + 1 write, each 4 bytes
            let bytes_per_iter = npix as f64 * 4.0 * 4.0;
            let gpu_gb_s = bytes_per_iter * ITERS as f64 / gpu_elapsed.as_secs_f64() / 1e9;

            // --- timed CPU run (same iteration count) ---
            let t1 = Instant::now();
            for _ in 0..ITERS {
                cpu_gray_from_rgb(&src_f32, &mut dst_cpu);
                std::hint::black_box(&dst_cpu);
            }
            let cpu_elapsed = t1.elapsed();
            let cpu_ms = cpu_elapsed.as_secs_f64() * 1e3 / ITERS as f64;

            let speedup = cpu_ms / gpu_ms;

            println!(
                "  {:<16}  {:>9.3}  {:>11.3}  {:>12.2}  {:>11.3}  {:>10.2}×",
                format!("{w}×{h}"),
                gpu_ms,
                gpu_ns_px,
                gpu_gb_s,
                cpu_ms,
                speedup,
            );
        }
    }
}

fn main() {
    #[cfg(feature = "gpu-cubecl")]
    inner::run();

    #[cfg(not(feature = "gpu-cubecl"))]
    {
        eprintln!("error: re-run with --features gpu-cubecl");
        std::process::exit(1);
    }
}
