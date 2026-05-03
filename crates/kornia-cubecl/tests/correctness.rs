//! Correctness: cubecl kernels must match `fast_image_resize` (NEON path)
//! to within ±1 LSB per channel, ≤ 0.1% mismatched channels.

use kornia_cubecl::resize::resize_bilinear_u8_rgb;
use kornia_cubecl::runtime;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize};
use kornia_tensor::CpuAllocator;
use rand::{rngs::StdRng, RngCore, SeedableRng};

const SIZES: &[(usize, usize)] = &[(512, 256), (1024, 512), (2048, 1024), (4096, 2048)];
const TOLERANCE_LSB: u8 = 1;
const MAX_MISMATCH_FRAC: f64 = 0.001;

fn make_image(w: usize, h: usize) -> Image<u8, 3, CpuAllocator> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut buf = vec![0u8; w * h * 3];
    rng.fill_bytes(&mut buf);
    Image::new(ImageSize { width: w, height: h }, buf, CpuAllocator).unwrap()
}

fn neon_reference(src: &Image<u8, 3, CpuAllocator>, dst_w: usize, dst_h: usize) -> Vec<u8> {
    let mut dst = Image::<u8, 3, _>::from_size_val(
        ImageSize { width: dst_w, height: dst_h },
        0,
        CpuAllocator,
    ).unwrap();
    resize::resize_fast_rgb(src, &mut dst, InterpolationMode::Bilinear).unwrap();
    dst.as_slice().to_vec()
}

fn compare(reference: &[u8], actual: &[u8], dst_w: usize, dst_h: usize, label: &str) {
    assert_eq!(reference.len(), actual.len(), "{label}: buffer length mismatch");
    let total_channels = (dst_w * dst_h * 3) as f64;
    let max_mismatch = (total_channels * MAX_MISMATCH_FRAC).ceil() as usize;
    let mut bad = 0usize;
    let mut max_diff: i32 = 0;
    for (r, a) in reference.iter().zip(actual.iter()) {
        let d = (*r as i32 - *a as i32).abs();
        if d > TOLERANCE_LSB as i32 {
            bad += 1;
        }
        if d > max_diff { max_diff = d; }
    }
    assert!(
        bad <= max_mismatch,
        "{label}: {bad} channels differ by > {TOLERANCE_LSB} LSB (max allowed {max_mismatch} of {total_channels}); max_diff={max_diff}"
    );
    eprintln!("[{label}] {dst_w}x{dst_h}: bad={bad}/{}, max_diff={max_diff} (within tol)", total_channels as usize);
}

#[cfg(feature = "cpu")]
#[test]
fn cubecl_cpu_matches_neon_within_tolerance() {
    let client = runtime::init_cpu();
    for &(src_w, src_h) in SIZES {
        let (dst_w, dst_h) = (src_w / 2, src_h / 2);
        let src = make_image(src_w, src_h);
        let reference = neon_reference(&src, dst_w, dst_h);

        let src_handle = client.create_from_slice(src.as_slice());
        let dst_handle = client.empty(dst_w * dst_h * 3);

        resize_bilinear_u8_rgb::<runtime::CpuRuntime>(
            &client,
            &src_handle,
            ImageSize { width: src_w, height: src_h },
            &dst_handle,
            ImageSize { width: dst_w, height: dst_h },
        ).unwrap();

        let actual = client.read_one(dst_handle.clone()).unwrap();
        compare(&reference, actual.as_ref(), dst_w, dst_h, "cpu");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cubecl_cuda_matches_neon_within_tolerance() {
    let client = match runtime::init_cuda() {
        Ok(c) => c,
        Err(msg) => { eprintln!("skipping cuda test: {msg}"); return; }
    };
    for &(src_w, src_h) in SIZES {
        let (dst_w, dst_h) = (src_w / 2, src_h / 2);
        let src = make_image(src_w, src_h);
        let reference = neon_reference(&src, dst_w, dst_h);

        let src_handle = client.create_from_slice(src.as_slice());
        let dst_handle = client.empty(dst_w * dst_h * 3);

        resize_bilinear_u8_rgb::<runtime::CudaRuntime>(
            &client,
            &src_handle,
            ImageSize { width: src_w, height: src_h },
            &dst_handle,
            ImageSize { width: dst_w, height: dst_h },
        ).unwrap();

        let actual = client.read_one(dst_handle.clone()).unwrap();
        compare(&reference, actual.as_ref(), dst_w, dst_h, "cuda");
    }
}
