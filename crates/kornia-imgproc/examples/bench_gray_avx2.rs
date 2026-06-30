//! Verify the AVX2 RGB→gray u8 path is active and measure its speedup over scalar.
//!
//! Run: `cargo run --release -p kornia-imgproc --example bench_gray_avx2`

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::color::gray_from_rgb_u8;
use std::time::Instant;

/// Reference scalar implementation (same formula as the kornia kernel).
fn scalar(src: &[u8], dst: &mut [u8], npixels: usize) {
    for (i, out) in dst.iter_mut().take(npixels).enumerate() {
        let si = i * 3;
        *out = ((77 * src[si] as u32 + 150 * src[si + 1] as u32 + 29 * src[si + 2] as u32) >> 8)
            as u8;
    }
}

fn time<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    // warmup
    for _ in 0..3 {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() / iters as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let feats = kornia_imgproc::simd::cpu_features();
    println!("CPU AVX2 detected: {}", feats.has_avx2);
    #[cfg(target_arch = "x86_64")]
    println!("compiled for x86_64 → kornia uses rgb_to_gray_u8_avx2 when AVX2 present");

    for (w, h) in [(640usize, 480usize), (1280, 720), (1920, 1080)] {
        let n = w * h;
        // Deterministic pseudo-random RGB input.
        let mut src = vec![0u8; n * 3];
        let mut s: u32 = 0x1234_5678;
        for b in src.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (s >> 24) as u8;
        }
        let img = Image::<u8, 3, CpuAllocator>::new(
            ImageSize { width: w, height: h },
            src.clone(),
            CpuAllocator,
        )?;
        let mut dst = Image::<u8, 1, CpuAllocator>::from_size_val(img.size(), 0u8, CpuAllocator)?;

        // Correctness: AVX2/kornia path vs scalar must be bit-identical.
        gray_from_rgb_u8(&img, &mut dst)?;
        let mut ref_dst = vec![0u8; n];
        scalar(&src, &mut ref_dst, n);
        let mismatches = dst
            .as_slice()
            .iter()
            .zip(&ref_dst)
            .filter(|(a, b)| a != b)
            .count();

        let iters = (200_000_000 / n).max(20);
        let t_kornia = time(iters, || {
            gray_from_rgb_u8(&img, &mut dst).unwrap();
        });
        let mut scratch = vec![0u8; n];
        let t_scalar = time(iters, || {
            scalar(&src, &mut scratch, n);
        });

        let mpix = n as f64 / 1e6;
        println!(
            "\n{w}x{h} ({mpix:.2} MP)  mismatches={mismatches}\n  \
             kornia(AVX2): {:.3} ms  ({:.0} MP/s)\n  \
             scalar ref  : {:.3} ms  ({:.0} MP/s)\n  \
             speedup     : {:.2}x",
            t_kornia * 1e3,
            mpix / t_kornia,
            t_scalar * 1e3,
            mpix / t_scalar,
            t_scalar / t_kornia,
        );
    }
    Ok(())
}
