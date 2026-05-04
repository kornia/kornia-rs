//! Bench: sequential 3-kernel pipeline vs single fused kernel.
//!
//! Pipeline: u8 RGB src --(bilinear resize)--> u8 RGB tmp --(rgb_to_gray)--> u8 gray --(normalize)--> f32 dst
//!
//! "Sequential" launches three kernels with intermediate device buffers (DRAM round-trips).
//! "Fused" calls a single kernel that does all three ops inline.

use kornia_cubecl::resize::{
    normalize_u8_to_f32, resize_bilinear_u8_rgb_with_weights, resize_to_gray_normalize_with_weights,
    rgb_to_gray_u8, WeightHandles,
};
use kornia_cubecl::runtime;
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::time::Instant;

const PIPELINES: &[(usize, usize, usize, usize)] = &[
    (1024, 512, 512, 256),
    (2048, 1024, 1024, 512),
    (4096, 2048, 2048, 1024),
    (8192, 4096, 4096, 2048),
    (1920, 1080, 960, 540),
];
const REPS: usize = 10;
const WARMUP: usize = 3;
const MEAN: f32 = 127.5;
const INV_STD: f32 = 1.0 / 64.0;

fn make_image(w: usize, h: usize) -> Image<u8, 3, CpuAllocator> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut buf = vec![0u8; w * h * 3];
    rng.fill_bytes(&mut buf);
    Image::new(ImageSize { width: w, height: h }, buf, CpuAllocator).unwrap()
}

fn stats(mut s: Vec<f64>) -> (f64, f64, f64) {
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[0], s[s.len() / 2], s.iter().sum::<f64>() / s.len() as f64)
}

fn fmt_us(s: f64) -> String { format!("{:>9.1}", s * 1e6) }
fn fmt_mpix(p: usize, s: f64) -> String { format!("{:>7.1}", (p as f64) / s / 1e6) }

fn main() {
    let cuda = runtime::init_cuda().expect("CUDA required");
    println!("# Pipeline: bilinear resize → RGB→gray → normalize_to_f32");
    println!("# Sequential = 3 kernel launches w/ intermediate DRAM buffers");
    println!("# Fused      = 1 kernel launch, no intermediate buffers");
    println!("# Reps={REPS}, warmup={WARMUP}\n");
    println!("{:<22}{:<14}{:>11}{:>11}{:>11}{:>10}",
        "src→dst", "arm", "min(μs)", "med(μs)", "mean(μs)", "Mpix/s");
    println!("{}", "-".repeat(80));

    for &(src_w, src_h, dst_w, dst_h) in PIPELINES {
        let dst_pix = dst_w * dst_h;
        let label = format!("{src_w}x{src_h}→{dst_w}x{dst_h}");
        let src = make_image(src_w, src_h);

        // Persistent device buffers
        let src_h_cu = cuda.create_from_slice(src.as_slice());
        let tmp_rgb = cuda.empty(dst_w * dst_h * 3);    // resize output (u8 RGB)
        let tmp_gray = cuda.empty(dst_w * dst_h);       // gray output (u8)
        let dst_f32 = cuda.empty(dst_w * dst_h * 4);    // f32 output (4 bytes/elem)
        let weights = WeightHandles::new::<runtime::CudaRuntime>(
            &cuda,
            ImageSize { width: src_w, height: src_h },
            ImageSize { width: dst_w, height: dst_h },
        );

        // === SEQUENTIAL: 3 kernel launches ===
        for _ in 0..WARMUP {
            resize_bilinear_u8_rgb_with_weights::<runtime::CudaRuntime>(
                &cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
                &tmp_rgb, ImageSize { width: dst_w, height: dst_h }, &weights,
            ).unwrap();
            rgb_to_gray_u8::<runtime::CudaRuntime>(&cuda, &tmp_rgb, &tmp_gray, dst_w, dst_h).unwrap();
            normalize_u8_to_f32::<runtime::CudaRuntime>(&cuda, &tmp_gray, &dst_f32, dst_w * dst_h, MEAN, INV_STD).unwrap();
            let _ = cubecl::future::block_on(cuda.sync());
        }
        let mut samples = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            resize_bilinear_u8_rgb_with_weights::<runtime::CudaRuntime>(
                &cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
                &tmp_rgb, ImageSize { width: dst_w, height: dst_h }, &weights,
            ).unwrap();
            rgb_to_gray_u8::<runtime::CudaRuntime>(&cuda, &tmp_rgb, &tmp_gray, dst_w, dst_h).unwrap();
            normalize_u8_to_f32::<runtime::CudaRuntime>(&cuda, &tmp_gray, &dst_f32, dst_w * dst_h, MEAN, INV_STD).unwrap();
            let _ = cubecl::future::block_on(cuda.sync());
            samples.push(t.elapsed().as_secs_f64());
        }
        let (mn, md, mu) = stats(samples);
        println!("{:<22}{:<14}{}{}{}{}", label, "sequential_3k", fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md));

        // === FUSED: 1 kernel launch ===
        for _ in 0..WARMUP {
            resize_to_gray_normalize_with_weights::<runtime::CudaRuntime>(
                &cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
                &dst_f32, ImageSize { width: dst_w, height: dst_h }, &weights, MEAN, INV_STD,
            ).unwrap();
            let _ = cubecl::future::block_on(cuda.sync());
        }
        let mut samples = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            resize_to_gray_normalize_with_weights::<runtime::CudaRuntime>(
                &cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
                &dst_f32, ImageSize { width: dst_w, height: dst_h }, &weights, MEAN, INV_STD,
            ).unwrap();
            let _ = cubecl::future::block_on(cuda.sync());
            samples.push(t.elapsed().as_secs_f64());
        }
        let (mn, md, mu) = stats(samples);
        println!("{:<22}{:<14}{}{}{}{}", "", "fused_1k", fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md));
        println!();
    }
}
