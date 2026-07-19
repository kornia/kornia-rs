//! Dump GPU resize output as JSON for OpenCV correctness comparison.
//!
//! ```text
//! cargo build --example dump_cuda_resize --features cuda --release
//! ./target/release/examples/dump_cuda_resize bilinear 64 48 32 24
//! ```
//!
//! Prints one JSON object to stdout:
//! `{"mode":"bilinear","src_w":64,"src_h":48,"dst_w":32,"dst_h":24,"pixels":[...]}`

use std::sync::Arc;

use cudarc::driver::CudaContext;
use kornia_imgproc::cuda::resize::{
    launch_resize_bicubic_cuda, launch_resize_bilinear_downscale_cuda,
    launch_resize_lanczos_cuda, launch_resize_nearest_downscale_cuda,
};

fn usage() -> ! {
    eprintln!("Usage: dump_cuda_resize bilinear|nearest|bicubic|lanczos SW SH DW DH");
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        usage();
    }
    let mode = args[0].as_str();
    let sw: u32 = args[1].parse().unwrap_or_else(|_| usage());
    let sh: u32 = args[2].parse().unwrap_or_else(|_| usage());
    let dw: u32 = args[3].parse().unwrap_or_else(|_| usage());
    let dh: u32 = args[4].parse().unwrap_or_else(|_| usage());

    let total_src = sw as usize * sh as usize * 3;
    let total_dst = dw as usize * dh as usize * 3;

    // Ramp image: pixel[i] = i / (total-1). Identical to numpy.arange(N)/（N-1).
    let src: Vec<f32> = (0..total_src)
        .map(|i| i as f32 / (total_src - 1).max(1) as f32)
        .collect();

    let ctx = Arc::new(CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();

    let src_dev = stream.clone_htod(&src).expect("H→D src");
    let mut dst_dev = stream.alloc_zeros::<f32>(total_dst).expect("alloc dst");

    match mode {
        "bilinear" => launch_resize_bilinear_downscale_cuda(
            &ctx, &stream, &src_dev, &mut dst_dev, sw, sh, dw, dh, None,
        )
        .expect("launch bilinear"),
        "nearest" => launch_resize_nearest_downscale_cuda(
            &ctx, &stream, &src_dev, &mut dst_dev, sw, sh, dw, dh, None,
        )
        .expect("launch nearest"),
        "bicubic" => {
            launch_resize_bicubic_cuda(&ctx, &stream, &src_dev, &mut dst_dev, sw, sh, dw, dh, None)
                .expect("launch bicubic")
        }
        "lanczos" => {
            launch_resize_lanczos_cuda(&ctx, &stream, &src_dev, &mut dst_dev, sw, sh, dw, dh, None)
                .expect("launch lanczos")
        }
        _ => usage(),
    }

    let dst = stream.clone_dtoh(&dst_dev).expect("D→H dst");
    stream.synchronize().expect("sync");

    print!(
        r#"{{"mode":"{mode}","src_w":{sw},"src_h":{sh},"dst_w":{dw},"dst_h":{dh},"pixels":["#
    );
    for (i, v) in dst.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{v:.8}");
    }
    println!("]}}");
}
