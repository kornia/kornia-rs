//! Micro-benchmark for the fused `resize + normalize + layout-convert` kernel.
//!
//! Compares three pipelines on a 1080p → 540p 3-channel u8 → f32 NCHW run:
//!
//! 1. `pyrdown_2x_rgb_u8` alone (u8 → u8, no normalize, no layout) — floor.
//! 2. Separate pipeline: pyrdown + scalar normalize/layout-convert pass.
//! 3. Fused `resize_normalize_to_tensor_u8_to_f32` — the new kernel.
//!
//! The fused number should land close to (1) since the normalize+layout math
//! piggybacks on the same 2×2 average and happens entirely in registers.

use std::time::Instant;

use kornia_imgproc::resize::{resize_fast_rgb, resize_normalize_to_tensor_u8_to_f32, NormalizeParams};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;

fn main() {
    let src_w = 1920;
    let src_h = 1080;
    let dst_w = src_w / 2;
    let dst_h = src_h / 2;

    let src_bytes: Vec<u8> = (0..src_w * src_h * 3).map(|i| (i % 251) as u8).collect();
    let src_img = Image::<u8, 3, _>::new(
        ImageSize {
            width: src_w,
            height: src_h,
        },
        src_bytes.clone(),
        CpuAllocator,
    )
    .unwrap();

    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let params = NormalizeParams::<3>::from_mean_std(mean, std);

    let mut dst_u8 = Image::<u8, 3, _>::from_size_val(
        ImageSize {
            width: dst_w,
            height: dst_h,
        },
        0u8,
        CpuAllocator,
    )
    .unwrap();
    let mut dst_chw_f32 = vec![0f32; 3 * dst_h * dst_w];

    let n = 200;

    // Warmup.
    for _ in 0..20 {
        resize_fast_rgb(&src_img, &mut dst_u8, InterpolationMode::Bilinear).unwrap();
        resize_normalize_to_tensor_u8_to_f32(
            &src_bytes, src_w, src_h, &mut dst_chw_f32, dst_w, dst_h, &params,
        );
    }

    // 1. pyrdown alone (floor: pure resize cost).
    let t0 = Instant::now();
    for _ in 0..n {
        resize_fast_rgb(&src_img, &mut dst_u8, InterpolationMode::Bilinear).unwrap();
    }
    let pyrdown_ms = t0.elapsed().as_secs_f64() / n as f64 * 1000.0;

    // 2. Separate: pyrdown + scalar normalize/layout pass.
    let t0 = Instant::now();
    for _ in 0..n {
        resize_fast_rgb(&src_img, &mut dst_u8, InterpolationMode::Bilinear).unwrap();
        separate_normalize_hwc_to_chw(
            dst_u8.as_slice(),
            &mut dst_chw_f32,
            dst_w,
            dst_h,
            mean,
            std,
        );
    }
    let separate_ms = t0.elapsed().as_secs_f64() / n as f64 * 1000.0;

    // 3. Fused kernel.
    let t0 = Instant::now();
    for _ in 0..n {
        resize_normalize_to_tensor_u8_to_f32(
            &src_bytes, src_w, src_h, &mut dst_chw_f32, dst_w, dst_h, &params,
        );
    }
    let fused_ms = t0.elapsed().as_secs_f64() / n as f64 * 1000.0;

    println!("1080p → 540p, RGB u8 → f32 NCHW, ImageNet mean/std:");
    println!("  pyrdown_2x (u8→u8, no norm):  {:>6.3} ms  (floor)", pyrdown_ms);
    println!(
        "  separate: pyrdown + norm+layout: {:>6.3} ms  ({:.2}× pyrdown)",
        separate_ms,
        separate_ms / pyrdown_ms
    );
    println!(
        "  fused (this PR):                 {:>6.3} ms  ({:.2}× pyrdown, {:.2}× vs separate)",
        fused_ms,
        fused_ms / pyrdown_ms,
        separate_ms / fused_ms
    );
}

/// Reference non-fused normalize + HWC→CHW pass (scalar, three passes).
fn separate_normalize_hwc_to_chw(
    src: &[u8],
    dst: &mut [f32],
    w: usize,
    h: usize,
    mean: [f32; 3],
    std: [f32; 3],
) {
    let plane = h * w;
    let (r_plane, rest) = dst.split_at_mut(plane);
    let (g_plane, b_plane) = rest.split_at_mut(plane);
    for i in 0..plane {
        let r = src[i * 3] as f32 / 255.0;
        let g = src[i * 3 + 1] as f32 / 255.0;
        let b = src[i * 3 + 2] as f32 / 255.0;
        r_plane[i] = (r - mean[0]) / std[0];
        g_plane[i] = (g - mean[1]) / std[1];
        b_plane[i] = (b - mean[2]) / std[2];
    }
}
