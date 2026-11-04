//! CUDA FAST corner detector launcher; bit-exact with `fast_detect_rows_u8`
//! in `crates/kornia-imgproc/src/features/fast.rs` (see [`fast_detect_u8`] for
//! the one CPU path it does not reproduce).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig};
use kornia_image::Image;
use kornia_tensor::{CudaError, CudaKernel};

use kornia_tensor::cuda::oxide;

super::define_cuda_error!(
    /// Error type for the CUDA FAST detector launchers.
    CudaFastError,
    "CUDA FAST error: {0}"
);

impl From<kornia_tensor::cuda::oxide::OxideError> for CudaFastError {
    fn from(e: kornia_tensor::cuda::oxide::OxideError) -> Self {
        CudaFastError::Cuda(e.to_string())
    }
}

impl From<CudaError> for CudaFastError {
    fn from(e: CudaError) -> Self {
        CudaFastError::Cuda(e.to_string())
    }
}

/// Launch raw FAST-N score map evaluation on GPU.
///
/// `dst[y * width + x]` receives the raw score (u16), 0 if not a corner or
/// within `margin` of the image edge. `margin` is raised to at least 3 (the
/// ring radius), as the CPU's `border.max(3)`, so the kernel never reads
/// outside `src`. For `n == 9` the score is the FAST-9
/// `cornerScore` (as the CPU's `corner_score_9_scalar`), otherwise
/// `Σ|ring - center|` (as the CPU's `fast_score_scalar`).
///
/// Zero pixels is a no-op. Errors if `threshold_u8` is outside `1..=255`,
/// if `width * height` does not fit the kernel's `i32` indexing, or if a
/// slice holds fewer than `width * height` elements.
#[allow(clippy::too_many_arguments)]
pub fn launch_fast_score_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u16>,
    width: u32,
    height: u32,
    threshold_u8: u32,
    n: u32,
    margin: u32,
) -> Result<(), CudaFastError> {
    let npixels = (width as usize)
        .checked_mul(height as usize)
        .filter(|&p| p <= i32::MAX as usize)
        .ok_or_else(|| {
            CudaFastError::Cuda(format!(
                "{width}x{height} exceeds the kernel's i32 pixel index"
            ))
        })?;
    if !(1..=255).contains(&threshold_u8) {
        return Err(CudaFastError::Cuda(format!(
            "threshold_u8 {threshold_u8} outside 1..=255"
        )));
    }
    if npixels == 0 {
        return Ok(());
    }

    CudaFastError::check_slice("src", src.len(), npixels)?;
    CudaFastError::check_slice("dst", dst.len(), npixels)?;

    // The kernel reads a radius-3 ring; a smaller margin would read out of bounds.
    let margin = margin.max(3);
    let kernel = oxide::kernel(stream, oxide::entry::FAST_SCORE_U8)?;

    let config = LaunchConfig {
        block_dim: (32, 8, 1),
        grid_dim: (width.div_ceil(32), height.div_ceil(8), 1),
        shared_mem_bytes: 0,
    };

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&width)
        .arg(&height)
        .arg(&threshold_u8)
        .arg(&n)
        .arg(&margin)
        .launch_cfg(config)?;

    Ok(())
}

/// Detect FAST corners on device and return `([y, x], score)` in row-major
/// order, the same shape and order as `fast_detect_rows_u8`.
///
/// Bit-exact with `fast_detect_rows_u8(src, threshold, arc_length, border,
/// 0..height)` except in one case: for `arc_length == 9`, `width >= 800`,
/// and a CPU with the AVX2/NEON path, the CPU additionally drops corners that
/// are not a 1-D local maximum inside each 16-pixel SIMD block (its
/// `use_inblock_filter`). The GPU does not apply that filter, so there it
/// returns a superset of the CPU corners, with identical scores on the
/// corners both return.
pub fn fast_detect_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    width: usize,
    height: usize,
    threshold: f32,
    arc_length: usize,
    border: usize,
) -> Result<Vec<([usize; 2], f32)>, CudaFastError> {
    let npixels = width
        .checked_mul(height)
        .ok_or_else(|| CudaFastError::Cuda("width * height overflow".into()))?;
    if npixels == 0 {
        return Ok(Vec::new());
    }
    let to_u32 = |v: usize, what: &str| {
        u32::try_from(v).map_err(|_| CudaFastError::Cuda(format!("{what} {v} exceeds u32")))
    };

    // SAFETY: `fast_score_u8` writes all `width * height` scores (0 for
    // border and non-corner pixels; `fast_score_u8_degenerate` checks it), and
    // nothing reads `d_scores` before that launch on the same stream.
    let mut d_scores = unsafe { stream.alloc::<u16>(npixels) }?;
    // Same conversion as `fast_detect_rows_u8_impl`.
    let threshold_u8 = (threshold * 255.0).round().clamp(1.0, 255.0) as u32;
    // Same as the CPU `border.max(3)`; a margin past u32 excludes every pixel either way.
    let margin = border.max(3).min(u32::MAX as usize) as u32;

    launch_fast_score_u8(
        stream,
        src,
        &mut d_scores,
        to_u32(width, "width")?,
        to_u32(height, "height")?,
        threshold_u8,
        to_u32(arc_length, "arc_length")?,
        margin,
    )?;

    let scores = stream.clone_dtoh(&d_scores)?;
    stream.synchronize()?;

    let mut corners = Vec::new();
    // Same scale as the CPU (`scratch[i] as f32 * scale` / `fast_score_scalar`).
    let scale = 1.0f32 / 255.0;
    for (y, row) in scores.chunks_exact(width).enumerate() {
        for (x, &s) in row.iter().enumerate() {
            if s != 0 {
                corners.push(([y, x], s as f32 * scale));
            }
        }
    }

    Ok(corners)
}

/// Convenience wrapper for [`fast_detect_u8`] taking an [`Image<u8, 1>`]
/// (includes the host-to-device copy).
pub fn fast_detect_u8_image(
    stream: &Arc<CudaStream>,
    src: &Image<u8, 1>,
    threshold: f32,
    arc_length: usize,
    border: usize,
) -> Result<Vec<([usize; 2], f32)>, CudaFastError> {
    let d_src = stream.clone_htod(src.as_slice())?;
    fast_detect_u8(
        stream,
        &d_src,
        src.width(),
        src.height(),
        threshold,
        arc_length,
        border,
    )
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::features::fast_detect_rows_u8;

    const SENTINEL: u16 = 0xABCD;
    const PAD: usize = 9;

    fn default_stream() -> Arc<CudaStream> {
        crate::cuda::color::test_utils::default_stream()
    }

    fn checkerboard_u8(width: usize, height: usize, block_size: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let check = ((x / block_size) + (y / block_size)).is_multiple_of(2);
                data[y * width + x] = if check { 240 } else { 20 };
            }
        }
        data
    }

    fn noise_u8(width: usize, height: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height];
        let mut state = 0xDEAD_BEEFu32;
        for b in &mut data {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (state >> 24) as u8;
        }
        data
    }

    /// `mh01_frame1.png` (752x480 EuRoC frame) and `dog.jpeg` converted to gray.
    fn real_images() -> Vec<(&'static str, Image<u8, 1>)> {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data");
        let mh01 = kornia_io::png::read_image_png_mono8(dir.join("mh01_frame1.png"))
            .expect("mh01_frame1.png")
            .into_inner();
        let rgb = kornia_io::jpeg::read_image_jpeg_rgb8(dir.join("dog.jpeg")).expect("dog.jpeg");
        let mut dog = Image::from_size_val(rgb.size(), 0u8).unwrap();
        crate::color::gray_from_rgb_u8(&rgb, &mut dog).unwrap();
        vec![("mh01_frame1", mh01), ("dog", dog)]
    }

    /// Dense u16 map from the CPU `([y, x], score)` list (score = raw / 255).
    fn cpu_score_map(img: &Image<u8, 1>, threshold: f32, n: usize, border: usize) -> Vec<u16> {
        let (w, h) = (img.width(), img.height());
        let mut map = vec![0u16; w * h];
        for ([y, x], s) in fast_detect_rows_u8(img, threshold, n, border, 0..h) {
            map[y * w + x] = (s * 255.0).round() as u16;
        }
        map
    }

    fn gpu_score_map(
        stream: &Arc<CudaStream>,
        img: &Image<u8, 1>,
        threshold: f32,
        n: usize,
        border: usize,
    ) -> Vec<u16> {
        let (w, h) = (img.width(), img.height());
        let npx = w * h;
        let d_src = stream.clone_htod(img.as_slice()).unwrap();
        let mut d_dst = stream.clone_htod(&vec![SENTINEL; npx + PAD]).unwrap();
        let t = (threshold * 255.0).round().clamp(1.0, 255.0) as u32;
        launch_fast_score_u8(
            stream,
            &d_src,
            &mut d_dst,
            w as u32,
            h as u32,
            t,
            n as u32,
            // Raw border: borders 0..3 exercise the launcher's margin clamp.
            border as u32,
        )
        .unwrap();
        let out: Vec<u16> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        assert!(
            out[npx..].iter().all(|&v| v == SENTINEL),
            "wrote past width*height ({w}x{h})"
        );
        out[..npx].to_vec()
    }

    /// GPU must equal the CPU, except FAST-9 at width >= 800 where the CPU's
    /// SIMD in-block local-max filter may drop corners: there every CPU corner
    /// must be on the GPU with the same score.
    fn assert_parity(
        stream: &Arc<CudaStream>,
        name: &str,
        img: &Image<u8, 1>,
        threshold: f32,
        n: usize,
        border: usize,
    ) {
        let (w, h) = (img.width(), img.height());
        let cpu = cpu_score_map(img, threshold, n, border);
        let gpu = gpu_score_map(stream, img, threshold, n, border);
        let ctx = format!("{name} {w}x{h} n={n} t={threshold} border={border}");
        if n == 9 && w >= 800 {
            for (i, (&c, &g)) in cpu.iter().zip(&gpu).enumerate() {
                assert!(
                    c == 0 || c == g,
                    "{ctx}: CPU corner at ({}, {}) score {c}, GPU {g}",
                    i / w,
                    i % w
                );
            }
        } else if let Some(i) = (0..w * h).find(|&i| cpu[i] != gpu[i]) {
            panic!(
                "{ctx}: first mismatch at ({}, {}): CPU {} GPU {}",
                i / w,
                i % w,
                cpu[i],
                gpu[i]
            );
        }
        // The list API must match the CPU list exactly where the maps match.
        if !(n == 9 && w >= 800) {
            let gpu_list = fast_detect_u8_image(stream, img, threshold, n, border).unwrap();
            let cpu_list = fast_detect_rows_u8(img, threshold, n, border, 0..h);
            assert_eq!(gpu_list, cpu_list, "{ctx}: list API");
        }
    }

    #[test]
    fn fast_score_u8_bit_exact() {
        let stream = default_stream();
        let sizes = [
            (1, 1),
            (1, 37),
            (37, 1),
            (6, 6),
            (7, 7),
            (8, 7),
            (17, 9),
            (31, 33),
            (333, 217),
            (640, 480),
            (752, 480),
        ];
        for (w, h) in sizes {
            for (name, data) in [
                ("noise", noise_u8(w, h)),
                ("checker", checkerboard_u8(w, h, 5)),
            ] {
                let img = Image::new([w, h].into(), data).unwrap();
                for n in [9, 10, 12] {
                    // 0.0 clamps to threshold_u8 = 1, 1.0 to 255.
                    for threshold in [0.0f32, 0.05, 0.2, 1.0] {
                        for border in [0, 3, 4, 10] {
                            assert_parity(&stream, name, &img, threshold, n, border);
                        }
                    }
                }
            }
        }
        for (name, img) in real_images() {
            for n in [9, 10, 12] {
                for threshold in [0.02f32, 0.05, 0.1, 0.15, 0.3] {
                    for border in [0, 7, 31] {
                        assert_parity(&stream, name, &img, threshold, n, border);
                    }
                }
            }
        }
    }

    #[test]
    fn fast_score_u8_bit_exact_hd() {
        let stream = default_stream();
        for (w, h) in [(1920, 1080), (1921, 1079)] {
            for (name, data) in [
                ("noise", noise_u8(w, h)),
                ("checker", checkerboard_u8(w, h, 16)),
            ] {
                let img = Image::new([w, h].into(), data).unwrap();
                for n in [9, 10, 12] {
                    for threshold in [0.05f32, 0.2] {
                        assert_parity(&stream, name, &img, threshold, n, 3);
                    }
                }
                // FAST-9 at HD in full: the CPU dense response (same scores, no
                // in-block filter, margin 3) must equal the GPU map everywhere.
                for threshold in [0.05f32, 0.2] {
                    let mut det =
                        crate::features::FastDetector::new(img.size(), threshold, 9, 1).unwrap();
                    let cpu: Vec<u16> = det
                        .compute_corner_response_u8(&img)
                        .as_slice()
                        .iter()
                        .map(|&s| (s * 255.0).round() as u16)
                        .collect();
                    let gpu = gpu_score_map(&stream, &img, threshold, 9, 3);
                    assert!(
                        cpu == gpu,
                        "{name} {w}x{h} t={threshold}: dense FAST-9 response"
                    );
                }
            }
        }
    }

    #[test]
    fn fast_score_u8_degenerate() {
        let stream = default_stream();
        let d_src = stream.clone_htod(&[0u8; 16]).unwrap();
        let mut d_dst = stream.clone_htod(&[SENTINEL; 16]).unwrap();

        // Zero pixels: no-op, dst untouched.
        for (w, h) in [(0, 0), (0, 5), (5, 0)] {
            launch_fast_score_u8(&stream, &d_src, &mut d_dst, w, h, 20, 9, 3).unwrap();
        }
        let out: Vec<u16> = stream.clone_dtoh(&d_dst).unwrap();
        assert!(out.iter().all(|&v| v == SENTINEL));

        // Overflowing / out-of-i32 size, bad threshold, short slices -> Err.
        assert!(
            launch_fast_score_u8(&stream, &d_src, &mut d_dst, u32::MAX, u32::MAX, 20, 9, 3)
                .is_err()
        );
        assert!(launch_fast_score_u8(&stream, &d_src, &mut d_dst, 65536, 32768, 20, 9, 3).is_err());
        assert!(launch_fast_score_u8(&stream, &d_src, &mut d_dst, 4, 4, 0, 9, 3).is_err());
        assert!(launch_fast_score_u8(&stream, &d_src, &mut d_dst, 4, 4, 256, 9, 3).is_err());
        assert!(matches!(
            launch_fast_score_u8(&stream, &d_src, &mut d_dst, 5, 5, 20, 9, 3),
            Err(CudaFastError::SliceTooSmall { what: "src", .. })
        ));

        // Zero-size list API.
        assert!(fast_detect_u8(&stream, &d_src, 0, 7, 0.1, 9, 3)
            .unwrap()
            .is_empty());
    }
}
