//! Bench: sequential 3-kernel pipeline vs single fused kernel.
//!
//! Pipeline: u8 RGB src --(bilinear resize)--> u8 RGB tmp --(rgb_to_gray)--> u8 gray --(normalize)--> f32 dst
//!
//! "Sequential" launches three kernels with intermediate device buffers (DRAM round-trips).
//! "Fused" calls a single kernel that does all three ops inline.

use kornia_cubecl::resize::{
    hwc_u8_to_chw_f32_normalize, normalize_u8_to_f32, resize_bilinear_u8_rgb_with_weights,
    resize_to_chw_normalize_with_weights, resize_to_gray_normalize_with_weights, rgb_to_gray_u8,
    WeightHandles,
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
// ImageNet-style per-channel mean/std (in u8 scale, 0-255)
const MEAN_RGB: [f32; 3] = [123.675, 116.28, 103.53];
const INV_STD_RGB: [f32; 3] = [1.0 / 58.395, 1.0 / 57.12, 1.0 / 57.375];

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

fn run_resize_only(cuda: &cubecl::client::ComputeClient<runtime::CudaRuntime>) {
    println!("\n## Pipeline 0: bilinear resize ONLY (HWC u8 RGB out)");
    println!("# baseline = our best resize-only kernel (with pre-uploaded weights)\n");
    println!("{:<22}{:<14}{:>11}{:>11}{:>11}{:>10}",
        "src→dst", "arm", "min(μs)", "med(μs)", "mean(μs)", "Mpix/s");
    println!("{}", "-".repeat(80));

    for &(src_w, src_h, dst_w, dst_h) in PIPELINES {
        let dst_pix = dst_w * dst_h;
        let label = format!("{src_w}x{src_h}→{dst_w}x{dst_h}");
        let src = make_image(src_w, src_h);
        let src_h_cu = cuda.create_from_slice(src.as_slice());
        let dst_h_cu = cuda.empty(dst_w * dst_h * 3);
        let weights = WeightHandles::new::<runtime::CudaRuntime>(
            cuda,
            ImageSize { width: src_w, height: src_h },
            ImageSize { width: dst_w, height: dst_h },
        );

        for _ in 0..WARMUP {
            resize_bilinear_u8_rgb_with_weights::<runtime::CudaRuntime>(
                cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
                &dst_h_cu, ImageSize { width: dst_w, height: dst_h }, &weights,
            ).unwrap();
            let _ = cubecl::future::block_on(cuda.sync());
        }
        let mut samples = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            resize_bilinear_u8_rgb_with_weights::<runtime::CudaRuntime>(
                cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
                &dst_h_cu, ImageSize { width: dst_w, height: dst_h }, &weights,
            ).unwrap();
            let _ = cubecl::future::block_on(cuda.sync());
            samples.push(t.elapsed().as_secs_f64());
        }
        let (mn, md, mu) = stats(samples);
        println!("{:<22}{:<14}{}{}{}{}", label, "resize_only", fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md));
    }
}

fn run_gray_pipeline(cuda: &cubecl::client::ComputeClient<runtime::CudaRuntime>) {
    println!("\n## Pipeline 1: bilinear resize → RGB→gray → normalize_to_f32 (HWC out)");
    println!("# sequential = 3 kernel launches with DRAM intermediates");
    println!("# fused      = 1 kernel launch\n");
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

fn run_chw_pipeline(cuda: &cubecl::client::ComputeClient<runtime::CudaRuntime>) {
    println!("\n## Pipeline 2: bilinear resize → per-channel normalize → CHW f32 (ML preprocessing)");
    println!("# sequential = 2 kernel launches (resize HWC u8 + transpose+normalize to CHW f32)");
    println!("# fused      = 1 kernel launch (resize+normalize+CHW write all inline)\n");
    println!("{:<22}{:<14}{:>11}{:>11}{:>11}{:>10}",
        "src→dst", "arm", "min(μs)", "med(μs)", "mean(μs)", "Mpix/s");
    println!("{}", "-".repeat(80));

    for &(src_w, src_h, dst_w, dst_h) in PIPELINES {
        let dst_pix = dst_w * dst_h;
        let label = format!("{src_w}x{src_h}→{dst_w}x{dst_h}");
        let src = make_image(src_w, src_h);

        let src_h_cu = cuda.create_from_slice(src.as_slice());
        let tmp_rgb = cuda.empty(dst_w * dst_h * 3);
        let dst_chw = cuda.empty(3 * dst_w * dst_h * 4); // 3 planes * pixels * 4 bytes/f32
        let weights = WeightHandles::new::<runtime::CudaRuntime>(
            cuda,
            ImageSize { width: src_w, height: src_h },
            ImageSize { width: dst_w, height: dst_h },
        );

        // === SEQUENTIAL: 2 kernels (resize + transpose-normalize) ===
        for _ in 0..WARMUP {
            resize_bilinear_u8_rgb_with_weights::<runtime::CudaRuntime>(
                cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
                &tmp_rgb, ImageSize { width: dst_w, height: dst_h }, &weights,
            ).unwrap();
            hwc_u8_to_chw_f32_normalize::<runtime::CudaRuntime>(
                cuda, &tmp_rgb, &dst_chw, dst_w, dst_h, MEAN_RGB, INV_STD_RGB,
            ).unwrap();
            let _ = cubecl::future::block_on(cuda.sync());
        }
        let mut samples = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            resize_bilinear_u8_rgb_with_weights::<runtime::CudaRuntime>(
                cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
                &tmp_rgb, ImageSize { width: dst_w, height: dst_h }, &weights,
            ).unwrap();
            hwc_u8_to_chw_f32_normalize::<runtime::CudaRuntime>(
                cuda, &tmp_rgb, &dst_chw, dst_w, dst_h, MEAN_RGB, INV_STD_RGB,
            ).unwrap();
            let _ = cubecl::future::block_on(cuda.sync());
            samples.push(t.elapsed().as_secs_f64());
        }
        let (mn, md, mu) = stats(samples);
        println!("{:<22}{:<14}{}{}{}{}", label, "sequential_2k", fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md));

        // === FUSED: 1 kernel ===
        for _ in 0..WARMUP {
            resize_to_chw_normalize_with_weights::<runtime::CudaRuntime>(
                cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
                &dst_chw, ImageSize { width: dst_w, height: dst_h }, &weights, MEAN_RGB, INV_STD_RGB,
            ).unwrap();
            let _ = cubecl::future::block_on(cuda.sync());
        }
        let mut samples = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            resize_to_chw_normalize_with_weights::<runtime::CudaRuntime>(
                cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
                &dst_chw, ImageSize { width: dst_w, height: dst_h }, &weights, MEAN_RGB, INV_STD_RGB,
            ).unwrap();
            let _ = cubecl::future::block_on(cuda.sync());
            samples.push(t.elapsed().as_secs_f64());
        }
        let (mn, md, mu) = stats(samples);
        println!("{:<22}{:<14}{}{}{}{}", "", "fused_1k", fmt_us(mn), fmt_us(md), fmt_us(mu), fmt_mpix(dst_pix, md));
        println!();
    }
}

fn run_summary_table(cuda: &cubecl::client::ComputeClient<runtime::CudaRuntime>) {
    println!("\n## Summary: side-by-side at 1080p → 540p (typical ML preprocessing)\n");

    let (src_w, src_h, dst_w, dst_h) = (1920, 1080, 960, 540);
    let dst_pix = dst_w * dst_h;
    let src = make_image(src_w, src_h);
    let src_h_cu = cuda.create_from_slice(src.as_slice());
    let weights = WeightHandles::new::<runtime::CudaRuntime>(
        cuda,
        ImageSize { width: src_w, height: src_h },
        ImageSize { width: dst_w, height: dst_h },
    );

    fn time_n<F: FnMut()>(mut f: F) -> f64 {
        for _ in 0..WARMUP { f(); }
        let mut s = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            f();
            s.push(t.elapsed().as_secs_f64());
        }
        let (_, md, _) = stats(s);
        md
    }

    let dst_rgb = cuda.empty(dst_w * dst_h * 3);
    let dst_gray = cuda.empty(dst_w * dst_h);
    let dst_f32 = cuda.empty(dst_w * dst_h * 4);
    let dst_chw = cuda.empty(3 * dst_w * dst_h * 4);
    let tmp_rgb = cuda.empty(dst_w * dst_h * 3);
    let tmp_gray = cuda.empty(dst_w * dst_h);

    let p0 = time_n(|| {
        resize_bilinear_u8_rgb_with_weights::<runtime::CudaRuntime>(
            cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
            &dst_rgb, ImageSize { width: dst_w, height: dst_h }, &weights,
        ).unwrap();
        let _ = cubecl::future::block_on(cuda.sync());
    });

    let p1_seq = time_n(|| {
        resize_bilinear_u8_rgb_with_weights::<runtime::CudaRuntime>(
            cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
            &tmp_rgb, ImageSize { width: dst_w, height: dst_h }, &weights,
        ).unwrap();
        rgb_to_gray_u8::<runtime::CudaRuntime>(cuda, &tmp_rgb, &tmp_gray, dst_w, dst_h).unwrap();
        normalize_u8_to_f32::<runtime::CudaRuntime>(cuda, &tmp_gray, &dst_f32, dst_w*dst_h, MEAN, INV_STD).unwrap();
        let _ = cubecl::future::block_on(cuda.sync());
    });
    let p1_fused = time_n(|| {
        resize_to_gray_normalize_with_weights::<runtime::CudaRuntime>(
            cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
            &dst_f32, ImageSize { width: dst_w, height: dst_h }, &weights, MEAN, INV_STD,
        ).unwrap();
        let _ = cubecl::future::block_on(cuda.sync());
    });

    let p2_seq = time_n(|| {
        resize_bilinear_u8_rgb_with_weights::<runtime::CudaRuntime>(
            cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
            &tmp_rgb, ImageSize { width: dst_w, height: dst_h }, &weights,
        ).unwrap();
        hwc_u8_to_chw_f32_normalize::<runtime::CudaRuntime>(
            cuda, &tmp_rgb, &dst_chw, dst_w, dst_h, MEAN_RGB, INV_STD_RGB,
        ).unwrap();
        let _ = cubecl::future::block_on(cuda.sync());
    });
    let p2_fused = time_n(|| {
        resize_to_chw_normalize_with_weights::<runtime::CudaRuntime>(
            cuda, &src_h_cu, ImageSize { width: src_w, height: src_h },
            &dst_chw, ImageSize { width: dst_w, height: dst_h }, &weights, MEAN_RGB, INV_STD_RGB,
        ).unwrap();
        let _ = cubecl::future::block_on(cuda.sync());
    });

    println!("{:<48}{:>12}{:>12}{:>12}", "Pipeline", "median(μs)", "Mpix/s", "vs P0");
    println!("{}", "-".repeat(84));
    println!("{:<48}{:>12.1}{:>12.1}{:>12}", "P0: resize only (HWC u8 RGB)",        p0*1e6,      dst_pix as f64 / p0 / 1e6, "1.00×");
    println!("{:<48}{:>12.1}{:>12.1}{:>11.2}×", "P1 sequential: resize → gray → norm (3k)", p1_seq*1e6,  dst_pix as f64 / p1_seq / 1e6, p0/p1_seq);
    println!("{:<48}{:>12.1}{:>12.1}{:>11.2}×", "P1 fused:      resize+gray+norm (1k)",     p1_fused*1e6,dst_pix as f64 / p1_fused / 1e6, p0/p1_fused);
    println!("{:<48}{:>12.1}{:>12.1}{:>11.2}×", "P2 sequential: resize → norm+CHW (2k)",    p2_seq*1e6,  dst_pix as f64 / p2_seq / 1e6, p0/p2_seq);
    println!("{:<48}{:>12.1}{:>12.1}{:>11.2}×", "P2 fused:      resize+norm+CHW (1k)",      p2_fused*1e6,dst_pix as f64 / p2_fused / 1e6, p0/p2_fused);
    println!();
    println!("Fusion savings:");
    println!("  Pipeline 1 (gray HWC):  sequential {:.1} μs → fused {:.1} μs  =  {:.2}× faster",
        p1_seq*1e6, p1_fused*1e6, p1_seq/p1_fused);
    println!("  Pipeline 2 (CHW f32):   sequential {:.1} μs → fused {:.1} μs  =  {:.2}× faster",
        p2_seq*1e6, p2_fused*1e6, p2_seq/p2_fused);
}

fn main() {
    let cuda = runtime::init_cuda().expect("CUDA required");
    println!("# Cubecl pipeline fusion benches on Jetson Orin Nano");
    println!("# Reps={REPS}, warmup={WARMUP}");
    run_resize_only(&cuda);
    run_gray_pipeline(&cuda);
    run_chw_pipeline(&cuda);
    run_summary_table(&cuda);
}
