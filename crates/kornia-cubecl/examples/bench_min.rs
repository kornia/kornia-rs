//! Minimal bench: NEON vs cubecl-cpu × {kernel-only, end-to-end} × 4 sizes.
//! No criterion — std::time only, 10 reps, reports min/median/mean.

use kornia_cubecl::resize::{resize_bilinear_u8_rgb, resize_bilinear_u8_rgb_x16, resize_bilinear_u8_rgb_x4};
use kornia_cubecl::runtime;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize};
use kornia_tensor::CpuAllocator;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::time::Instant;

// (src_w, src_h, dst_w, dst_h) — power-of-two 2x sweep + ML-realistic 1080p→540p.
const SIZES: &[(usize, usize, usize, usize)] = &[
    (512, 256, 256, 128),
    (1024, 512, 512, 256),
    (2048, 1024, 1024, 512),
    (4096, 2048, 2048, 1024),
    (8192, 4096, 4096, 2048),
    (1920, 1080, 960, 540),
];
const REPS: usize = 10;
const WARMUP: usize = 3;

fn make_image(w: usize, h: usize) -> Image<u8, 3, CpuAllocator> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut buf = vec![0u8; w * h * 3];
    rng.fill_bytes(&mut buf);
    Image::new(ImageSize { width: w, height: h }, buf, CpuAllocator).unwrap()
}

fn stats(mut samples: Vec<f64>) -> (f64, f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = samples[0];
    let med = samples[samples.len() / 2];
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    (min, med, mean)
}

fn fmt_us(s: f64) -> String { format!("{:>9.1}", s * 1e6) }

fn fmt_mpix(dst_pix: usize, s: f64) -> String {
    format!("{:>7.1}", (dst_pix as f64) / s / 1e6)
}

fn main() {
    let cpu_client = runtime::init_cpu();

    println!("\n# Bilinear u8 RGB 2x downscale — NEON (fast_image_resize) vs cubecl-cpu");
    println!("# Reps = {REPS} (with {WARMUP} warmup), reporting min/median/mean μs and Mpix/s (median).\n");
    println!("{:<14}{:<24}{:>11}{:>11}{:>11}{:>10}",
        "src→dst", "arm", "min(μs)", "med(μs)", "mean(μs)", "Mpix/s");
    println!("{}", "-".repeat(80));

    for &(src_w, src_h, dst_w, dst_h) in SIZES {
        let dst_pix = dst_w * dst_h;
        let id = format!("{src_w}x{src_h}→{dst_w}x{dst_h}");
        let src = make_image(src_w, src_h);
        let mut dst_neon = Image::<u8, 3, _>::from_size_val(
            ImageSize { width: dst_w, height: dst_h }, 0, CpuAllocator,
        ).unwrap();

        // --- NEON baseline ---
        for _ in 0..WARMUP {
            resize::resize_fast_rgb(&src, &mut dst_neon, InterpolationMode::Bilinear).unwrap();
        }
        let mut s = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            resize::resize_fast_rgb(&src, &mut dst_neon, InterpolationMode::Bilinear).unwrap();
            s.push(t.elapsed().as_secs_f64());
        }
        let (mn, md, mu) = stats(s);
        println!("{:<14}{:<24}{}{}{}{}", id, "neon", fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md));

        // --- cubecl-cpu kernel-only (warm device buffers) ---
        let src_h_cpu = cpu_client.create_from_slice(src.as_slice());
        let dst_h_cpu = cpu_client.empty(dst_w * dst_h * 3);
        for _ in 0..WARMUP {
            resize_bilinear_u8_rgb::<runtime::CpuRuntime>(
                &cpu_client, &src_h_cpu,
                ImageSize { width: src_w, height: src_h }, &dst_h_cpu,
                ImageSize { width: dst_w, height: dst_h },
            ).unwrap();
            let _ = cubecl::future::block_on(cpu_client.sync());
        }
        let mut s = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            resize_bilinear_u8_rgb::<runtime::CpuRuntime>(
                &cpu_client, &src_h_cpu,
                ImageSize { width: src_w, height: src_h }, &dst_h_cpu,
                ImageSize { width: dst_w, height: dst_h },
            ).unwrap();
            let _ = cubecl::future::block_on(cpu_client.sync());
            s.push(t.elapsed().as_secs_f64());
        }
        let (mn, md, mu) = stats(s);
        println!("{:<14}{:<24}{}{}{}{}", "", "cubecl_cpu_kernel", fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md));

        // --- cubecl-cpu kernel-only x4 (4 dst pixels per thread) ---
        if dst_w % 4 == 0 {
            for _ in 0..WARMUP {
                resize_bilinear_u8_rgb_x4::<runtime::CpuRuntime>(
                    &cpu_client, &src_h_cpu,
                    ImageSize { width: src_w, height: src_h }, &dst_h_cpu,
                    ImageSize { width: dst_w, height: dst_h },
                ).unwrap();
                let _ = cubecl::future::block_on(cpu_client.sync());
            }
            let mut s4 = Vec::with_capacity(REPS);
            for _ in 0..REPS {
                let t = Instant::now();
                resize_bilinear_u8_rgb_x4::<runtime::CpuRuntime>(
                    &cpu_client, &src_h_cpu,
                    ImageSize { width: src_w, height: src_h }, &dst_h_cpu,
                    ImageSize { width: dst_w, height: dst_h },
                ).unwrap();
                let _ = cubecl::future::block_on(cpu_client.sync());
                s4.push(t.elapsed().as_secs_f64());
            }
            let (mn, md, mu) = stats(s4);
            println!("{:<14}{:<24}{}{}{}{}", "", "cubecl_cpu_kernel_x4", fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md));
        }

        // --- cubecl-cpu kernel-only x16 (16 dst pixels per thread) ---
        if dst_w % 16 == 0 {
            for _ in 0..WARMUP {
                resize_bilinear_u8_rgb_x16::<runtime::CpuRuntime>(
                    &cpu_client, &src_h_cpu,
                    ImageSize { width: src_w, height: src_h }, &dst_h_cpu,
                    ImageSize { width: dst_w, height: dst_h },
                ).unwrap();
                let _ = cubecl::future::block_on(cpu_client.sync());
            }
            let mut s16 = Vec::with_capacity(REPS);
            for _ in 0..REPS {
                let t = Instant::now();
                resize_bilinear_u8_rgb_x16::<runtime::CpuRuntime>(
                    &cpu_client, &src_h_cpu,
                    ImageSize { width: src_w, height: src_h }, &dst_h_cpu,
                    ImageSize { width: dst_w, height: dst_h },
                ).unwrap();
                let _ = cubecl::future::block_on(cpu_client.sync());
                s16.push(t.elapsed().as_secs_f64());
            }
            let (mn, md, mu) = stats(s16);
            println!("{:<14}{:<24}{}{}{}{}", "", "cubecl_cpu_kernel_x16", fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md));
        }

        // --- cubecl-cpu end-to-end (alloc + upload + kernel + read) ---
        for _ in 0..WARMUP {
            let s = cpu_client.create_from_slice(src.as_slice());
            let d = cpu_client.empty(dst_w * dst_h * 3);
            resize_bilinear_u8_rgb::<runtime::CpuRuntime>(
                &cpu_client, &s,
                ImageSize { width: src_w, height: src_h }, &d,
                ImageSize { width: dst_w, height: dst_h },
            ).unwrap();
            let _ = cpu_client.read_one(d).unwrap();
        }
        let mut samples = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            let s = cpu_client.create_from_slice(src.as_slice());
            let d = cpu_client.empty(dst_w * dst_h * 3);
            resize_bilinear_u8_rgb::<runtime::CpuRuntime>(
                &cpu_client, &s,
                ImageSize { width: src_w, height: src_h }, &d,
                ImageSize { width: dst_w, height: dst_h },
            ).unwrap();
            let _ = cpu_client.read_one(d).unwrap();
            samples.push(t.elapsed().as_secs_f64());
        }
        let (mn, md, mu) = stats(samples);
        println!("{:<14}{:<24}{}{}{}{}", "", "cubecl_cpu_e2e", fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md));
        println!();
    }
}
