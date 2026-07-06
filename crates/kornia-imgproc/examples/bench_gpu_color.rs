//! Micro-benchmark: GPU `gray_from_rgb_f32` vs CPU scalar + AVX2+FMA.
//!
//! **Timing methodology:**
//! - GPU: queue all ITERS launches across N_BUFFS rotating source buffers,
//!   then block with `read_one_unchecked`.  Rotating buffers ensures the GPU's
//!   read-only L2 cache is fully defeated so the reported GB/s reflects DRAM
//!   bandwidth, not cache throughput.
//! - CPU scalar: plain loop, auto-vectorised by LLVM at `-O3`.
//! - CPU AVX2: 8-pixel-wide sequential loads + shuffle deinterleave + FMA.
//!   Three sequential 256-bit loads per 8 pixels; no gather instructions.
//!
//! ```text
//! # CPU only (no CUDA needed)
//! cargo run --example bench_gpu_color --release
//!
//! # CPU + GPU comparison
//! cargo run --example bench_gpu_color --features gpu-cubecl --release
//! ```

use std::time::Instant;

const WARMUP: u32 = 50;
const ITERS: u32 = 200;
#[cfg(feature = "gpu-cubecl")]
const N_BUFFS: usize = 8;
const SIZES: &[(u32, u32)] = &[(512, 512), (1024, 1024), (1920, 1080), (3840, 2160)];

const RW: f32 = 0.299;
const GW: f32 = 0.587;
const BW: f32 = 0.114;

// --------------------------------------------------------------------------
// CPU kernels
// --------------------------------------------------------------------------

fn cpu_gray_scalar(src: &[f32], dst: &mut [f32]) {
    for (i, d) in dst.iter_mut().enumerate() {
        let b = i * 3;
        *d = RW * src[b] + GW * src[b + 1] + BW * src[b + 2];
    }
}

/// AVX2+FMA: 8 pixels per iteration.
///
/// Loads 3 consecutive 256-bit chunks (24 f32 = 8 RGB pixels), deinterleaves
/// R/G/B via `_mm256_permutevar8x32_ps` + `_mm256_blend_ps`, then applies
/// the grayscale weights with fused multiply-add.  No gather instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cpu_gray_avx2(src: &[f32], dst: &mut [f32]) {
    use std::arch::x86_64::*;

    let rw = _mm256_set1_ps(RW);
    let gw = _mm256_set1_ps(GW);
    let bw = _mm256_set1_ps(BW);

    // ── Deinterleave pattern ──────────────────────────────────────────────
    // For 8 pixels the three 256-bit loads cover:
    //   v0: [R0,G0,B0, R1,G1,B1, R2,G2]   indices 0-7
    //   v1: [B2,R3,G3, B3,R4,G4, B4,R5]   indices 8-15
    //   v2: [G5,B5,R6, G6,B6,R7, G7,B7]   indices 16-23
    //
    // _mm256_set_epi32(x7,x6,x5,x4,x3,x2,x1,x0) → lane i = xi.
    //
    // R channel: R0=v0[0],R1=v0[3],R2=v0[6] | R3=v1[1],R4=v1[4],R5=v1[7] | R6=v2[2],R7=v2[5]
    let pr0 = _mm256_set_epi32(0, 0, 0, 0, 0, 6, 3, 0); // lanes 0,1,2 ← v0
    let pr1 = _mm256_set_epi32(0, 0, 7, 4, 1, 0, 0, 0); // lanes 3,4,5 ← v1
    let pr2 = _mm256_set_epi32(5, 2, 0, 0, 0, 0, 0, 0); // lanes 6,7   ← v2

    // G channel: G0=v0[1],G1=v0[4],G2=v0[7] | G3=v1[2],G4=v1[5] | G5=v2[0],G6=v2[3],G7=v2[6]
    let pg0 = _mm256_set_epi32(0, 0, 0, 0, 0, 7, 4, 1); // lanes 0,1,2 ← v0
    let pg1 = _mm256_set_epi32(0, 0, 0, 5, 2, 0, 0, 0); // lanes 3,4   ← v1
    let pg2 = _mm256_set_epi32(6, 3, 0, 0, 0, 0, 0, 0); // lanes 5,6,7 ← v2

    // B channel: B0=v0[2],B1=v0[5] | B2=v1[0],B3=v1[3],B4=v1[6] | B5=v2[1],B6=v2[4],B7=v2[7]
    let pb0 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 5, 2); // lanes 0,1   ← v0
    let pb1 = _mm256_set_epi32(0, 0, 0, 6, 3, 0, 0, 0); // lanes 2,3,4 ← v1
    let pb2 = _mm256_set_epi32(7, 4, 1, 0, 0, 0, 0, 0); // lanes 5,6,7 ← v2

    let n = dst.len();
    let mut i = 0;

    while i + 8 <= n {
        let base = src.as_ptr().add(i * 3);
        let v0 = _mm256_loadu_ps(base);
        let v1 = _mm256_loadu_ps(base.add(8));
        let v2 = _mm256_loadu_ps(base.add(16));

        // R: merge v0 (lanes 0-2) | v1 (lanes 3-5) | v2 (lanes 6-7)
        let r = _mm256_blend_ps::<0xC0>(
            _mm256_blend_ps::<0x38>(
                _mm256_permutevar8x32_ps(v0, pr0),
                _mm256_permutevar8x32_ps(v1, pr1),
            ),
            _mm256_permutevar8x32_ps(v2, pr2),
        );
        // G: merge v0 (lanes 0-2) | v1 (lanes 3-4) | v2 (lanes 5-7)
        let g = _mm256_blend_ps::<0xE0>(
            _mm256_blend_ps::<0x18>(
                _mm256_permutevar8x32_ps(v0, pg0),
                _mm256_permutevar8x32_ps(v1, pg1),
            ),
            _mm256_permutevar8x32_ps(v2, pg2),
        );
        // B: merge v0 (lanes 0-1) | v1 (lanes 2-4) | v2 (lanes 5-7)
        let b = _mm256_blend_ps::<0xE0>(
            _mm256_blend_ps::<0x1C>(
                _mm256_permutevar8x32_ps(v0, pb0),
                _mm256_permutevar8x32_ps(v1, pb1),
            ),
            _mm256_permutevar8x32_ps(v2, pb2),
        );

        let out = _mm256_fmadd_ps(r, rw, _mm256_fmadd_ps(g, gw, _mm256_mul_ps(b, bw)));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), out);
        i += 8;
    }

    // scalar tail for n % 8 != 0
    while i < n {
        let b = i * 3;
        dst[i] = RW * src[b] + GW * src[b + 1] + BW * src[b + 2];
        i += 1;
    }
}

fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Warms up and times the AVX2 path. Returns `None` if AVX2/FMA is absent.
// `src`/`dst` are only referenced inside `#[cfg(target_arch = "x86_64")]`; on
// other hosts the early `return None` makes them unreachable.
#[allow(unused_variables)]
fn time_avx2(src: &[f32], dst: &mut [f32]) -> Option<f64> {
    if !avx2_available() {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        for _ in 0..WARMUP {
            unsafe { cpu_gray_avx2(src, dst) };
            std::hint::black_box(dst as &[f32]);
        }
        let t = Instant::now();
        for _ in 0..ITERS {
            unsafe { cpu_gray_avx2(src, dst) };
            std::hint::black_box(dst as &[f32]);
        }
        Some(t.elapsed().as_secs_f64() * 1e3 / ITERS as f64)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}

/// Effective memory bandwidth: 3R + 1W = 16 B/pixel.
fn gb_per_sec(npix: usize, ms_per_iter: f64) -> f64 {
    (npix as f64 * 16.0) / (ms_per_iter * 1e-3) / 1e9
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
    println!("\n=== CPU baseline ===");
    println!(
        "  {:<16}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}",
        "size", "scalar ms", "scalar GB/s", "avx2 ms", "avx2 GB/s", "speedup"
    );
    println!("  {}", "-".repeat(72));

    for &(w, h) in SIZES {
        let npix = (w * h) as usize;
        let src: Vec<f32> = (0..npix * 3).map(|i| (i % 256) as f32 / 255.0).collect();
        let mut dst = vec![0f32; npix];

        for _ in 0..WARMUP {
            cpu_gray_scalar(&src, &mut dst);
            std::hint::black_box(&dst);
        }
        let t = Instant::now();
        for _ in 0..ITERS {
            cpu_gray_scalar(&src, &mut dst);
            std::hint::black_box(&dst);
        }
        let scalar_ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        match time_avx2(&src, &mut dst) {
            Some(avx2_ms) => println!(
                "  {:<16}  {:>10.3}  {:>10.2}  {:>10.3}  {:>10.2}  {:>7.2}×",
                format!("{w}×{h}"),
                scalar_ms,
                gb_per_sec(npix, scalar_ms),
                avx2_ms,
                gb_per_sec(npix, avx2_ms),
                scalar_ms / avx2_ms,
            ),
            None => println!(
                "  {:<16}  {:>10.3}  {:>10.2}  {:>10}  {:>10}  {:>8}",
                format!("{w}×{h}"),
                scalar_ms,
                gb_per_sec(npix, scalar_ms),
                "N/A",
                "N/A",
                "N/A",
            ),
        }
    }
}

// --------------------------------------------------------------------------
// GPU section
// --------------------------------------------------------------------------

#[cfg(feature = "gpu-cubecl")]
fn run_gpu() {
    use cubecl::prelude::*;
    use cubecl_cuda::CudaRuntime;
    use kornia_imgproc::gpu::color::launch_gray_from_rgb_f32;

    fn f32_as_bytes(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) }
    }

    println!("=== GPU (CubeCL/CUDA) — batch {ITERS} iters, {N_BUFFS} rotating src buffers ===");
    println!(
        "  {:<16}  {:>9}  {:>11}  {:>12}  {:>12}  {:>10}  {:>9}",
        "size", "GPU ms", "GPU ns/px", "GPU GB/s", "scalar ms", "AVX2 ms", "speedup"
    );
    println!("  {}", "-".repeat(90));

    let device = <CudaRuntime as Runtime>::Device::default();
    let client = CudaRuntime::client(&device);

    for &(w, h) in SIZES {
        let npix = (w * h) as usize;

        // N_BUFFS distinct source buffers defeat GPU L2 read cache across iterations.
        let src_bufs: Vec<_> = (0..N_BUFFS)
            .map(|b| {
                let data: Vec<f32> = (0..npix * 3)
                    .map(|i| ((i + b * 17) % 256) as f32 / 255.0)
                    .collect();
                client.create_from_slice(f32_as_bytes(&data))
            })
            .collect();
        let dst_gpu = client.empty(npix * 4);

        for i in 0..WARMUP {
            launch_gray_from_rgb_f32(
                &client,
                src_bufs[i as usize % N_BUFFS].clone(),
                dst_gpu.clone(),
                w,
                h,
            );
        }
        let _ = client.read_one_unchecked(dst_gpu.clone());
        let dst_gpu = client.empty(npix * 4);

        // All ITERS are queued then synced once → measures sustained throughput.
        let t0 = Instant::now();
        for i in 0..ITERS {
            launch_gray_from_rgb_f32(
                &client,
                src_bufs[i as usize % N_BUFFS].clone(),
                dst_gpu.clone(),
                w,
                h,
            );
        }
        let _ = client.read_one_unchecked(dst_gpu);
        let gpu_elapsed = t0.elapsed();
        let gpu_ms = gpu_elapsed.as_secs_f64() * 1e3 / ITERS as f64;
        let gpu_ns_px = gpu_elapsed.as_nanos() as f64 / (ITERS as f64 * npix as f64);

        // CPU comparison baselines — cache-warm (src_f32 was just allocated and
        // touched by GPU buffer creation above, so it likely sits in L3).
        let src_f32: Vec<f32> = (0..npix * 3).map(|i| (i % 256) as f32 / 255.0).collect();
        let mut dst_cpu = vec![0f32; npix];

        for _ in 0..WARMUP {
            cpu_gray_scalar(&src_f32, &mut dst_cpu);
            std::hint::black_box(&dst_cpu);
        }
        let t1 = Instant::now();
        for _ in 0..ITERS {
            cpu_gray_scalar(&src_f32, &mut dst_cpu);
            std::hint::black_box(&dst_cpu);
        }
        let scalar_ms = t1.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        let avx2_ms = time_avx2(&src_f32, &mut dst_cpu).unwrap_or(scalar_ms);

        println!(
            "  {:<16}  {:>9.3}  {:>11.3}  {:>12.2}  {:>12.3}  {:>10.3}  {:>8.2}×",
            format!("{w}×{h}"),
            gpu_ms,
            gpu_ns_px,
            gb_per_sec(npix, gpu_ms),
            scalar_ms,
            avx2_ms,
            avx2_ms / gpu_ms,
        );
    }
}
