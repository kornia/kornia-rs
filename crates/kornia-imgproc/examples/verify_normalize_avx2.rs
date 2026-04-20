//! Cross-arch sanity runner for `normalize_rgb_u8`.
//!
//! Built as an example (not a test) to stay outside the dev-dep graph —
//! that keeps the workspace build from pulling in `criterion`/`alloca`
//! and their C-toolchain requirement under `qemu-x86_64` emulation.
//!
//! Run natively on aarch64 or cross under qemu:
//!
//! ```sh
//! cargo run --release -p kornia-imgproc --example verify_normalize_avx2 \
//!     --target x86_64-unknown-linux-gnu
//! ```
//!
//! Expected output reports which SIMD path the dispatcher selected and
//! max-abs-diff of the vectorized result vs scalar reference over a
//! deterministic PRNG-seeded 1000-pixel input.
//!
//! qemu caveat: AVX2 TCG emulation landed in qemu 7.2. On Ubuntu 22.04
//! jammy (`qemu-user-static` is 6.2) `-cpu max` advertises AVX2 to the
//! feature probe so dispatch selects the AVX2 path, but the actual
//! `vfmadd` traps with SIGILL. Cross-run under jammy therefore only
//! verifies compile + link + dispatch-selection; execution correctness
//! requires qemu ≥ 7.2 or native x86 hardware.

use kornia_imgproc::normalize::normalize_rgb_u8;
use kornia_imgproc::simd::cpu_features;

fn main() {
    let cpu = cpu_features();
    println!("CPU features detected by runtime probe:");
    println!("  has_avx2    = {}", cpu.has_avx2);
    println!("  has_fma     = {}", cpu.has_fma);
    println!("  has_avx512f = {}", cpu.has_avx512f);
    println!("  has_neon    = {}", cpu.has_neon);
    println!("  has_fp16    = {}", cpu.has_fp16);
    println!("  has_dotprod = {}", cpu.has_dotprod);

    let active = if cfg!(target_arch = "aarch64") && cpu.has_neon {
        "aarch64 NEON"
    } else if cfg!(target_arch = "x86_64") && cpu.has_avx2 && cpu.has_fma {
        "x86_64 AVX2+FMA"
    } else {
        "portable scalar"
    };
    println!("Dispatcher will select: {}\n", active);

    let npix = 1000;
    let mut src = vec![0u8; npix * 3];
    let mut s: u32 = 0xdeadbeef;
    for b in src.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (s >> 24) as u8;
    }
    // ImageNet normalization constants folded into scale/offset form.
    let scale = [1.0 / (0.229 * 255.0), 1.0 / (0.224 * 255.0), 1.0 / (0.225 * 255.0)];
    let offset = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225];

    let mut dst_vec = vec![0.0f32; npix * 3];
    normalize_rgb_u8(&src, &mut dst_vec, npix, &scale, &offset);

    // Hand-rolled scalar reference inline (can't call the private
    // `normalize_rgb_u8_scalar` directly from outside the crate; this is a
    // literal transcription of that function's body).
    let mut dst_ref = vec![0.0f32; npix * 3];
    for i in 0..npix {
        let base = i * 3;
        dst_ref[base] = src[base] as f32 * scale[0] + offset[0];
        dst_ref[base + 1] = src[base + 1] as f32 * scale[1] + offset[1];
        dst_ref[base + 2] = src[base + 2] as f32 * scale[2] + offset[2];
    }

    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (a, b)) in dst_vec.iter().zip(dst_ref.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
            max_idx = i;
        }
    }

    println!("Vectorized vs scalar: max |diff| = {:.3e} at index {}", max_abs, max_idx);
    if max_abs < 1e-4 {
        println!("PASS (within FMA rounding tolerance)");
    } else {
        eprintln!("FAIL — vectorized path disagrees with scalar reference!");
        std::process::exit(1);
    }
}
