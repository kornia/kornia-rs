//! CUDA kernels for RGB ↔ grayscale conversion.
//!
//! u8 mirrors the CPU Q8 fixed-point math bit-for-bit
//! (`color/gray/kernels.rs`); f32 uses the same BT.601 weights.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::{get_kernel, launch_map, CudaColorError, KernelCell, PxPerThread};

// Vectorized: each thread handles 4 pixels = 12 src bytes (three 32-bit word
// loads, always 4-byte aligned since 12 ≡ 0 mod 4 and cudarc allocations are
// 256-byte aligned) and stores the 4 grays as one 32-bit word. The last
// (partial) quad falls back to byte addressing. Math is the same truncating
// Q8 shift as the CPU NEON path — bit-exact.
static GRAY_FROM_RGB_U8_SRC: &str = r#"
__device__ __forceinline__ unsigned int gray_q8(
    unsigned int r, unsigned int g, unsigned int b)
{
    return (GRAY_WR_Q8 * r + GRAY_WG_Q8 * g + GRAY_WB_Q8 * b) >> 8;
}

extern "C" __global__ void gray_from_rgb_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int q = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int nquads = (npixels + 3u) / 4u;
    if (q >= nquads) return;

    unsigned int p = q * 4u;
    if (p + 4u <= npixels) {
        const unsigned int* s32 = (const unsigned int*)src;
        unsigned int w0 = __ldg(&s32[q * 3u]);
        unsigned int w1 = __ldg(&s32[q * 3u + 1u]);
        unsigned int w2 = __ldg(&s32[q * 3u + 2u]);
        // Byte layout: w0 = r0 g0 b0 r1 | w1 = g1 b1 r2 g2 | w2 = b2 r3 g3 b3
        unsigned int g0 = gray_q8(w0 & 0xFFu, (w0 >> 8) & 0xFFu, (w0 >> 16) & 0xFFu);
        unsigned int g1 = gray_q8(w0 >> 24, w1 & 0xFFu, (w1 >> 8) & 0xFFu);
        unsigned int g2 = gray_q8((w1 >> 16) & 0xFFu, w1 >> 24, w2 & 0xFFu);
        unsigned int g3 = gray_q8((w2 >> 8) & 0xFFu, (w2 >> 16) & 0xFFu, w2 >> 24);
        ((unsigned int*)dst)[q] = g0 | (g1 << 8) | (g2 << 16) | (g3 << 24);
    } else {
        for (unsigned int i = p; i < npixels; ++i) {
            unsigned int b = i * 3u;
            dst[i] = (unsigned char)gray_q8(
                __ldg(&src[b]), __ldg(&src[b + 1u]), __ldg(&src[b + 2u]));
        }
    }
}
"#;

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
static GRAY_FROM_RGB_U8: KernelCell = KernelCell::new();
static GRAY_FROM_RGB_F32: KernelCell = KernelCell::new();
static RGB_FROM_GRAY_U8: KernelCell = KernelCell::new();
static RGB_FROM_GRAY_F32: KernelCell = KernelCell::new();

/// Launch RGB8 → Gray8: `gray = (77·R + 150·G + 29·B) >> 8` (bit-exact vs CPU).
///
/// Word-vectorized 4 px/thread (see kernel source).
pub fn launch_gray_from_rgb_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = get_kernel(
        &GRAY_FROM_RGB_U8,
        stream,
        GRAY_FROM_RGB_U8_SRC,
        "gray_from_rgb_u8",
    )?;
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
    use crate::cuda::color_cuda::test_utils::{default_stream, pattern_f32, pattern_u8};

    // Odd size (tail coverage) + a realistic size.
    const SIZES: &[usize] = &[1, 37 * 23, 640 * 480];

    #[test]
    fn gray_from_rgb_u8_bit_exact_vs_cpu() {
        let stream = default_stream();
        for &n in SIZES {
            let src = pattern_u8(n * 3);
            let mut cpu = vec![0u8; n];
            crate::color::gray::kernels::gray_from_rgb_u8(&src, &mut cpu, n);

            let d_src = stream.clone_htod(&src).unwrap();
            let mut d_dst = stream.alloc_zeros::<u8>(n).unwrap();
            launch_gray_from_rgb_u8(&stream, &d_src, &mut d_dst, n).unwrap();
            let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
            stream.synchronize().unwrap();

            assert_eq!(gpu, cpu, "u8 gray must be bit-exact vs CPU (n={n})");
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

        let max_diff = crate::cuda::color_cuda::test_utils::max_abs_diff_f32(&gpu, &cpu);
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
