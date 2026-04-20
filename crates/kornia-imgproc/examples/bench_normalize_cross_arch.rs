//! Cross-arch benchmark for `normalize_rgb_u8` — measures SIMD vs scalar
//! speedup on whichever target we're compiled for.
//!
//! Runs two kernels over a 1920×1080 RGB u8 image:
//!   1. The scalar reference (reproduced inline — private in normalize.rs).
//!   2. The arch-dispatched fast path (`normalize_rgb_u8`), which picks
//!      NEON on aarch64 or AVX2+FMA on x86_64.
//!
//! Across arches, the absolute times differ (qemu translation overhead for
//! x86 on aarch64 hosts ≈ 5–20× slowdown vs native silicon), so only the
//! SIMD/scalar *ratio* is cross-arch comparable. That ratio tells us how
//! well each ISA's kernel stacks up against its own baseline — a meaningful
//! proxy for how the ports will compare on real hardware.

use std::time::Instant;

use kornia_imgproc::normalize::normalize_rgb_u8;
use kornia_imgproc::simd::cpu_features;

fn scalar_ref(src: &[u8], dst: &mut [f32], npixels: usize, scale: &[f32; 3], offset: &[f32; 3]) {
    for i in 0..npixels {
        let base = i * 3;
        dst[base] = src[base] as f32 * scale[0] + offset[0];
        dst[base + 1] = src[base + 1] as f32 * scale[1] + offset[1];
        dst[base + 2] = src[base + 2] as f32 * scale[2] + offset[2];
    }
}

fn bench<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    // Warmup
    for _ in 0..5 {
        f();
    }
    // Three rounds, take best median.
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        if ms < best {
            best = ms;
        }
    }
    best
}

fn main() {
    let (w, h) = (1920usize, 1080usize);
    let npix = w * h;

    // Deterministic PRNG-seeded input so runs are comparable.
    let mut src = vec![0u8; npix * 3];
    let mut s: u32 = 0xdeadbeef;
    for b in src.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (s >> 24) as u8;
    }

    // ImageNet normalization constants.
    let scale = [1.0 / (0.229 * 255.0), 1.0 / (0.224 * 255.0), 1.0 / (0.225 * 255.0)];
    let offset = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225];

    let mut dst_simd = vec![0.0f32; npix * 3];
    let mut dst_scalar = vec![0.0f32; npix * 3];

    let cpu = cpu_features();
    let arch_tag = if cfg!(target_arch = "aarch64") {
        "aarch64 NEON"
    } else if cfg!(target_arch = "x86_64") && cpu.has_avx2 && cpu.has_fma {
        if cpu.has_avx512f {
            "x86_64 AVX2+FMA (AVX-512F also available)"
        } else {
            "x86_64 AVX2+FMA"
        }
    } else {
        "portable scalar"
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ normalize_rgb_u8  —  1920×1080 RGB u8  (6.22 MB src / 24.9 MB dst) ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("Arch: {}", arch_tag);
    println!();

    let iters = 50;
    let t_scalar = bench(
        || scalar_ref(&src, &mut dst_scalar, npix, &scale, &offset),
        iters,
    );
    let t_simd = bench(
        || normalize_rgb_u8(&src, &mut dst_simd, npix, &scale, &offset),
        iters,
    );

    // Throughput in MPix/s (million pixels per second)
    let mpix_scalar = (npix as f64) / (t_scalar * 1e-3) / 1e6;
    let mpix_simd = (npix as f64) / (t_simd * 1e-3) / 1e6;

    println!("Path     |   time/call |   throughput");
    println!("---------+-------------+-------------");
    println!("scalar   | {:8.3} ms | {:7.1} MPix/s", t_scalar, mpix_scalar);
    println!("SIMD     | {:8.3} ms | {:7.1} MPix/s", t_simd, mpix_simd);
    println!();
    println!("SIMD speedup over scalar on this arch: {:.2}×", t_scalar / t_simd);

    // Correctness cross-check.
    let mut max_diff = 0.0f32;
    for (a, b) in dst_simd.iter().zip(dst_scalar.iter()) {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    println!("Max |SIMD − scalar|: {:.3e} (FMA rounding noise expected ~1e-7)", max_diff);
}
