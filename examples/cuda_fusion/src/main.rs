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

    // ── 4. The FKL paper pipeline, for the head-to-head comparison ─────
    // FusedKernelLibrary's flagship example (README/paper): bilinear
    // resize → Mul(1/255) → SplitWrite to planar f32. Ours: same chain
    // via Normalize{scale: 1/255, bias: 0} + WriteChwF32. Single and
    // batch-4, matching the FKL benchmark harness shape.
    let fkl_norm = Normalize {
        scale: [1.0 / 255.0; 3],
        bias: [0.0; 3],
    };
    let fkl_pipe = FusedPipeline::build(&ctx, &[&read, &fkl_norm, &WriteChwF32], dw, dh)?;
    let mut d_fkl = stream.alloc_zeros::<f32>((3 * dw * dh) as usize)?;
    for _ in 0..50 {
        fkl_pipe.launch(&stream, &d_src, &mut d_fkl)?;
    }
    stream.synchronize()?;
    let mut best_fkl = f64::INFINITY;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        for _ in 0..100 {
            fkl_pipe.launch(&stream, &d_src, &mut d_fkl)?;
        }
        stream.synchronize()?;
        best_fkl = best_fkl.min(t.elapsed().as_secs_f64() / 100.0);
    }

    let batch = 4usize;
    let d_srcs: Vec<_> = (0..batch)
        .map(|_| stream.clone_htod(&host))
        .collect::<Result<_, _>>()?;
    let src_refs: Vec<&cudarc::driver::CudaSlice<u8>> = d_srcs.iter().collect();
    let bpipe = FusedPipeline::build_batched(
        &ctx,
        &[&read, &fkl_norm, &WriteChwF32],
        dw,
        dh,
        batch as u32,
        (3 * dw * dh) as usize,
    )?;
    let mut d_bout = stream.alloc_zeros::<f32>(batch * (3 * dw * dh) as usize)?;
    for _ in 0..50 {
        bpipe.launch_batched(&stream, &src_refs, &mut d_bout)?;
    }
    stream.synchronize()?;
    let mut best_b = f64::INFINITY;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        for _ in 0..100 {
            bpipe.launch_batched(&stream, &src_refs, &mut d_bout)?;
        }
        stream.synchronize()?;
        best_b = best_b.min(t.elapsed().as_secs_f64() / 100.0);
    }
    println!(
        "FKL-paper pipeline (resize -> mul 1/255 -> planar split):
  single: {:.3} ms   batch-4: {:.3} ms   (compare with the FKL C++ binary — see README)",
        best_fkl * 1e3,
        best_b * 1e3
    );

    // ── 5. Sustained benchmark ──────────────────────────────────────────
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
