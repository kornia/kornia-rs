//! Dump GPU warp-perspective output as JSON for VPI/OpenCV correctness comparison.
//!
//! ```text
//! cargo build --example dump_cuda_warp_perspective --features cuda --release
//! ./target/release/examples/dump_cuda_warp_perspective bilinear 64 64 45.0
//! ```
//!
//! Prints one JSON object to stdout:
//! `{"mode":"bilinear","w":64,"h":64,"angle":45.0,"h3x3":[...],"pixels":[...]}`
//!
//! `h3x3` is the 9-element row-major forward homography (affine rotation embedded
//! in a 3×3 matrix, bottom row = [0, 0, 1]), matching what cv2.getPerspectiveTransform
//! and vpi.perspwarp expect.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use kornia_imgproc::cuda::warp_perspective::{
    launch_warp_perspective_bicubic_cuda, launch_warp_perspective_bilinear_cuda,
    launch_warp_perspective_lanczos_cuda, launch_warp_perspective_nearest_cuda,
};

fn usage() -> ! {
    eprintln!("Usage: dump_cuda_warp_perspective bilinear|nearest|bicubic|lanczos W H ANGLE_DEG");
    std::process::exit(1);
}

/// Rotation around image centre embedded in a 3×3 homography.
/// Bottom row is [0, 0, 1] — this is an affine-subcase perspective transform,
/// so warp_perspective and warp_affine should give identical output for this matrix.
fn homography_rotation(w: u32, h: u32, angle_deg: f32) -> [f32; 9] {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let angle_rad = angle_deg * std::f32::consts::PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    [
        cos_a,
        sin_a,
        (1.0 - cos_a) * cx - sin_a * cy,
        -sin_a,
        cos_a,
        sin_a * cx + (1.0 - cos_a) * cy,
        0.0,
        0.0,
        1.0,
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        usage();
    }
    let mode = args[0].as_str();
    let w: u32 = args[1].parse().unwrap_or_else(|_| usage());
    let h: u32 = args[2].parse().unwrap_or_else(|_| usage());
    let angle: f32 = args[3].parse().unwrap_or_else(|_| usage());

    let total = w as usize * h as usize * 3;
    let src: Vec<f32> = (0..total)
        .map(|i| i as f32 / (total - 1).max(1) as f32)
        .collect();

    let hmat = homography_rotation(w, h, angle);

    let ctx = Arc::new(CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();

    let src_dev = stream.clone_htod(&src).expect("H→D src");
    let mut dst_dev = stream.alloc_zeros::<f32>(total).expect("alloc dst");

    match mode {
        "bilinear" => launch_warp_perspective_bilinear_cuda(
            &ctx,
            &stream,
            &src_dev,
            &mut dst_dev,
            w,
            h,
            w,
            h,
            &hmat,
            None,
        )
        .expect("launch bilinear"),
        "nearest" => launch_warp_perspective_nearest_cuda(
            &ctx,
            &stream,
            &src_dev,
            &mut dst_dev,
            w,
            h,
            w,
            h,
            &hmat,
            None,
        )
        .expect("launch nearest"),
        "bicubic" => launch_warp_perspective_bicubic_cuda(
            &ctx,
            &stream,
            &src_dev,
            &mut dst_dev,
            w,
            h,
            w,
            h,
            &hmat,
            None,
        )
        .expect("launch bicubic"),
        "lanczos" => launch_warp_perspective_lanczos_cuda(
            &ctx,
            &stream,
            &src_dev,
            &mut dst_dev,
            w,
            h,
            w,
            h,
            &hmat,
            None,
        )
        .expect("launch lanczos"),
        _ => usage(),
    }

    let dst = stream.clone_dtoh(&dst_dev).expect("D→H dst");
    stream.synchronize().expect("sync");

    let hmat_str: Vec<String> = hmat.iter().map(|v| format!("{v:.8}")).collect();
    let hmat_json = hmat_str.join(",");
    print!(r#"{{"mode":"{mode}","w":{w},"h":{h},"angle":{angle},"h3x3":[{hmat_json}],"pixels":["#);
    for (i, v) in dst.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{v:.8}");
    }
    println!("]}}");
}
