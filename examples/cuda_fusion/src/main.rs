//! FKL-style kernel fusion: compose per-op stages into ONE generated CUDA
//! kernel where intermediates stay in registers.
//!
//! ```bash
//! cargo run -p cuda-fusion --release
//! ```
//!
//! Demonstrates:
//! 1. the standard DNN-preprocess chain (resize → normalize → CHW write);
//! 2. a NOVEL chain (resize → normalize → gray → single-plane write) that
//!    exists nowhere as hand-written kernel code — the engine generates it
//!    from the same stage library;
//! 3. printing the generated CUDA source;
//! 4. a sustained benchmark of the fused kernel.

use cudarc::driver::CudaContext;
use kornia_imgproc::cuda::fusion::{
    FusedPipeline, Normalize, ReadU8RgbBilinear, RgbToGray, WriteC1F32, WriteChwF32,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;

    let (sw, sh) = (1920u32, 1080u32);
    let (dw, dh) = (640u32, 640u32);

    // Fake camera frame (u8 HWC RGB).
    let host: Vec<u8> = (0..(sw * sh * 3) as usize)
        .map(|i| ((i * 2654435761usize) >> 24) as u8)
        .collect();
    let d_src = stream.clone_htod(&host)?;

    // ── 1. DNN preprocess: resize → normalize(ImageNet) → planar CHW ────
    let read = ReadU8RgbBilinear {
        src_w: sw,
        src_h: sh,
        dst_w: dw,
        dst_h: dh,
    };
    let norm = Normalize {
        scale: [
            1.0 / 255.0 / 0.229,
            1.0 / 255.0 / 0.224,
            1.0 / 255.0 / 0.225,
        ],
        bias: [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    };
    let pipe = FusedPipeline::build(&ctx, &[&read, &norm, &WriteChwF32], dw, dh)?;
    let mut d_out = stream.alloc_zeros::<f32>((3 * dw * dh) as usize)?;
    pipe.launch(&stream, &d_src, &mut d_out)?;
    stream.synchronize()?;
    println!("preprocess chain: OK ({}x{} -> CHW {}x{})", sw, sh, dw, dh);

    // ── 2. Novel composition: same stages, different chain ──────────────
    let gray_pipe = FusedPipeline::build(
        &ctx,
        &[
            &read,
            &Normalize {
                scale: [1.0 / 255.0; 3],
                bias: [0.0; 3],
            },
            &RgbToGray,
            &WriteC1F32,
        ],
        dw,
        dh,
    )?;
    let mut d_gray = stream.alloc_zeros::<f32>((dw * dh) as usize)?;
    gray_pipe.launch(&stream, &d_src, &mut d_gray)?;
    stream.synchronize()?;
    println!("novel gray chain: OK (no hand-written kernel exists for this)");

    // ── 3. The generated CUDA source ────────────────────────────────────
    println!("\n──── generated kernel (gray chain) ────");
    println!("{}", gray_pipe.generated_source());

    // ── 4. Sustained benchmark ──────────────────────────────────────────
    for _ in 0..50 {
        pipe.launch(&stream, &d_src, &mut d_out)?;
    }
    stream.synchronize()?;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        for _ in 0..200 {
            pipe.launch(&stream, &d_src, &mut d_out)?;
        }
        stream.synchronize()?;
        best = best.min(t.elapsed().as_secs_f64() / 200.0);
    }
    println!(
        "fused preprocess sustained: {:.3} ms/frame (1080p -> 640x640 CHW, one kernel)",
        best * 1e3
    );
    Ok(())
}
