//! CUDA kernels for RGB ↔ grayscale conversion.
//!
//! u8 mirrors the CPU Q14 fixed-point math bit-for-bit
//! (`color/gray/kernels.rs`); f32 uses the same BT.601 weights. RGB8 → Gray8
//! is a cuda-oxide kernel (`kernels-oxide/src/color.rs`); the rest are NVRTC.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::{get_kernel, launch_map, CudaColorError, KernelCell, PxPerThread};
use crate::cuda::oxide;

static GRAY_FROM_RGB_F32_SRC: &str = r#"
extern "C" __global__ void gray_from_rgb_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    dst[i] = GRAY_WR_F * __ldg(&src[b])
           + GRAY_WG_F * __ldg(&src[b + 1u])
           + GRAY_WB_F * __ldg(&src[b + 2u]);
}
"#;

static GRAY_F64_SRC: &str = r#"
extern "C" __global__ void gray_from_rgb_f64(
    const double* __restrict__ src,
    double* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    dst[i] = 0.299 * __ldg(&src[b]) + 0.587 * __ldg(&src[b + 1u]) + 0.114 * __ldg(&src[b + 2u]);
}

extern "C" __global__ void rgb_from_gray_f64(
    const double* __restrict__ src,
    double* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    double g = __ldg(&src[i]);
    unsigned int d = i * 3u;
    dst[d] = g; dst[d + 1u] = g; dst[d + 2u] = g;
}
"#;
const GRAY_F64_FNS: &[&str] = &["gray_from_rgb_f64", "rgb_from_gray_f64"];

static RGB_FROM_GRAY_U8_SRC: &str = r#"
extern "C" __global__ void rgb_from_gray_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned char g = __ldg(&src[i]);
    unsigned int d = i * 3u;
    dst[d] = g; dst[d + 1u] = g; dst[d + 2u] = g;
}
"#;

static RGB_FROM_GRAY_F32_SRC: &str = r#"
extern "C" __global__ void rgb_from_gray_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    float g = __ldg(&src[i]);
    unsigned int d = i * 3u;
    dst[d] = g; dst[d + 1u] = g; dst[d + 2u] = g;
}
"#;

static GRAY_F64: super::KernelSuiteCell = super::KernelSuiteCell::new();
static GRAY_FROM_RGB_F32: KernelCell = KernelCell::new();
static RGB_FROM_GRAY_U8: KernelCell = KernelCell::new();
static RGB_FROM_GRAY_F32: KernelCell = KernelCell::new();

/// Launch RGB8 → Gray8: `gray = (4899·R + 9617·G + 1868·B + 8192) >> 14`
/// (OpenCV Q14, bit-exact vs CPU).
///
/// Word-vectorized 4 px/thread; the kernel is written in Rust
/// (`kernels-oxide/src/color.rs`) and needs an sm_75+ GPU.
///
/// # Arguments
///
/// * `stream` - Stream to launch on.
/// * `src` - Interleaved RGB bytes, at least `3 * npixels` long.
/// * `dst` - Gray output, at least `npixels` long.
/// * `npixels` - Pixel count; zero is a no-op.
///
/// # Returns
///
/// `Ok(())` once the kernel is enqueued.
///
/// # Errors
///
/// Returns [`CudaColorError::SliceTooSmall`] if a buffer is short, or
/// [`CudaColorError::Cuda`] if the kernel module fails to load, `3 * npixels`
/// exceeds `u32::MAX`, or the launch fails.
pub fn launch_gray_from_rgb_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = oxide::kernel(stream, oxide::entry::GRAY_FROM_RGB_U8)?;
    launch_map(kernel, stream, src, dst, npixels, 3, 1, PxPerThread::Four)
}

/// Launch RGB f32 → Gray f32 with BT.601 weights (0.299, 0.587, 0.114).
pub fn launch_gray_from_rgb_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = get_kernel(
        &GRAY_FROM_RGB_F32,
        stream,
        GRAY_FROM_RGB_F32_SRC,
        "gray_from_rgb_f32",
    )?;
    launch_map(kernel, stream, src, dst, npixels, 3, 1, PxPerThread::One)
}

/// Launch Gray8 → RGB8 broadcast (replicate the channel).
pub fn launch_rgb_from_gray_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = get_kernel(
        &RGB_FROM_GRAY_U8,
        stream,
        RGB_FROM_GRAY_U8_SRC,
        "rgb_from_gray_u8",
    )?;
    launch_map(kernel, stream, src, dst, npixels, 1, 3, PxPerThread::One)
}

/// Launch Gray f32 → RGB f32 broadcast (replicate the channel).
pub fn launch_rgb_from_gray_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = get_kernel(
        &RGB_FROM_GRAY_F32,
        stream,
        RGB_FROM_GRAY_F32_SRC,
        "rgb_from_gray_f32",
    )?;
    launch_map(kernel, stream, src, dst, npixels, 1, 3, PxPerThread::One)
}

/// Launch RGB f64 → Gray f64 (BT.601 weights, matches the CPU f64 oracle).
pub fn launch_gray_from_rgb_f64(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f64>,
    dst: &mut CudaSlice<f64>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = super::get_kernel_suite(&GRAY_F64, stream, GRAY_F64_SRC, GRAY_F64_FNS, 0)?;
    launch_map(kernel, stream, src, dst, npixels, 3, 1, PxPerThread::One)
}

/// Launch Gray f64 → RGB f64 broadcast.
pub fn launch_rgb_from_gray_f64(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f64>,
    dst: &mut CudaSlice<f64>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = super::get_kernel_suite(&GRAY_F64, stream, GRAY_F64_SRC, GRAY_F64_FNS, 1)?;
    launch_map(kernel, stream, src, dst, npixels, 1, 3, PxPerThread::One)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_f32, pattern_u8};

    /// Public launcher vs the CPU path, byte for byte: every `n mod 4` tail, HD
    /// sizes, no write past `npixels` into a sentinel-filled oversized `dst`;
    /// then image shapes (1xN, Nx1, odd, 4K) through device `ConvertColor`.
    #[test]
    fn gray_from_rgb_u8_bit_exact() {
        use crate::color::ConvertColor;
        use kornia_image::color_spaces::{Gray8, Rgb8};
        use kornia_image::ImageSize;

        const PAD: usize = 9;
        const SENTINEL: u8 = 0xAB;
        let stream = default_stream();

        for n in [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            1023,
            37 * 23,
            640 * 480,
            1920 * 1080 + 3,
        ] {
            let src = pattern_u8(n * 3);
            let mut cpu = vec![0u8; n];
            crate::color::gray::kernels::gray_from_rgb_u8(&src, &mut cpu, n);

            let d_src = stream.clone_htod(&src).unwrap();
            let mut d_dst = stream.clone_htod(&vec![SENTINEL; n + PAD]).unwrap();
            launch_gray_from_rgb_u8(&stream, &d_src, &mut d_dst, n).unwrap();
            let out: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
            stream.synchronize().unwrap();
            assert!(
                out[n..].iter().all(|&b| b == SENTINEL),
                "wrote past npixels (n={n})"
            );
            assert_eq!(out[..n], cpu[..], "GPU vs CPU (n={n})");
        }

        for (width, height) in [
            (1, 1),
            (1, 7),
            (7, 1),
            (1, 4099),
            (4099, 1),
            (33, 17),
            (255, 3),
            (1025, 1),
            (1919, 1081),
            (3840, 2160),
        ] {
            let size = ImageSize { width, height };
            let rgb = Rgb8::from_size_vec(size, pattern_u8(width * height * 3)).unwrap();
            let mut cpu = Gray8::from_size_val(size, 0).unwrap();
            rgb.convert(&mut cpu).unwrap();

            let mut gpu = Gray8::zeros_cuda(size, &stream).unwrap();
            rgb.to_cuda(&stream).unwrap().convert(&mut gpu).unwrap();
            let gpu = gpu.to_host(&stream).unwrap();
            assert_eq!(gpu.as_slice(), cpu.as_slice(), "{width}x{height}");
        }
    }

    /// Zero pixels is a no-op; a pixel count whose byte count wraps `usize` or
    /// passes `u32::MAX` is an error, not a launch.
    #[test]
    fn gray_from_rgb_u8_degenerate_npixels() {
        const SENTINEL: u8 = 0xAB;
        let stream = default_stream();
        let d_src = stream.clone_htod(&[7u8; 12]).unwrap();
        let mut d_dst = stream.clone_htod(&[SENTINEL; 4]).unwrap();

        launch_gray_from_rgb_u8(&stream, &d_src, &mut d_dst, 0).unwrap();
        let out: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(out, [SENTINEL; 4], "zero pixels must not write");

        // The first wraps 3 * n to 2.
        for n in [usize::MAX / 3 + 1, u32::MAX as usize / 3 + 1] {
            assert!(
                matches!(
                    launch_gray_from_rgb_u8(&stream, &d_src, &mut d_dst, n),
                    Err(CudaColorError::Cuda(_))
                ),
                "n={n} must be rejected"
            );
        }
    }

    #[test]
    fn gray_from_rgb_f32_close_to_cpu() {
        let stream = default_stream();
        let n = 640 * 480;
        let src = pattern_f32(n * 3);
        let mut cpu = vec![0f32; n];
        crate::color::gray::kernels::gray_from_rgb_f32(&src, &mut cpu, n);

        let d_src = stream.clone_htod(&src).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(n).unwrap();
        launch_gray_from_rgb_f32(&stream, &d_src, &mut d_dst, n).unwrap();
        let gpu: Vec<f32> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();

        let max_diff = crate::cuda::color::test_utils::max_abs_diff_f32(&gpu, &cpu);
        assert!(max_diff <= 1e-3, "f32 gray max diff {max_diff} > 1e-3");
    }

    #[test]
    fn rgb_from_gray_broadcast_exact() {
        let stream = default_stream();
        let n = 37 * 23;
        let src = pattern_u8(n);
        let d_src = stream.clone_htod(&src).unwrap();
        let mut d_dst = stream.alloc_zeros::<u8>(n * 3).unwrap();
        launch_rgb_from_gray_u8(&stream, &d_src, &mut d_dst, n).unwrap();
        let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        for (i, &g) in src.iter().enumerate() {
            assert_eq!(&gpu[i * 3..i * 3 + 3], &[g, g, g]);
        }
    }
}
