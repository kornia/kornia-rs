//! CUDA kernels for RGB ↔ YCbCr / YUV (planar 3-channel, full-range OpenCV).
//!
//! Mirrors Family A of `color/yuv/kernels.rs`: u8 runs the Q14 fixed-point
//! path bit-for-bit (arithmetic `>>` on negative `int` in CUDA C equals Rust
//! `i32 >>`), f32 runs in `[0, 1]`. `YCrCb` stores `[Y, Cr, Cb]` and `YuvCbCr`
//! stores `[Y, U=Cb, V=Cr]` — same math, different chroma permutation, so each
//! order is its own `extern "C"` entry selected on the Rust side.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

pub use crate::color::yuv::kernels::ChromaOrder;

use super::{check_len, get_kernel_suite, CudaColorError, KernelSuiteCell};

static YCC_U8_SRC: &str = r#"
// Forward: RGB u8 -> (Y, Cr, Cb) u8, full-range Q14 (matches ycc_from_rgb_u8_px).
__device__ __forceinline__ void ycc_px_u8(
    int r, int g, int b, unsigned char* y_o, unsigned char* cr_o, unsigned char* cb_o)
{
    int y  = (C_YR * r + C_YG * g + C_YB * b + Q14_HALF) >> Q14_SHIFT;
    int cr = ((r - y) * C_YCRI + (128 << Q14_SHIFT) + Q14_HALF) >> Q14_SHIFT;
    int cb = ((b - y) * C_YCBI + (128 << Q14_SHIFT) + Q14_HALF) >> Q14_SHIFT;
    *y_o  = sat_u8(y);
    *cr_o = sat_u8(cr);
    *cb_o = sat_u8(cb);
}

// Inverse: (Y, Cr, Cb) u8 -> RGB u8, full-range Q14 (matches rgb_from_ycc_u8_px).
__device__ __forceinline__ void rgb_px_u8(
    int y, int cr, int cb, unsigned char* r_o, unsigned char* g_o, unsigned char* b_o)
{
    cr -= 128;
    cb -= 128;
    int r = y + ((C_CR2R * cr + Q14_HALF) >> Q14_SHIFT);
    int g = y + ((C_CR2G * cr + C_CB2G * cb + Q14_HALF) >> Q14_SHIFT);
    int b = y + ((C_CB2B * cb + Q14_HALF) >> Q14_SHIFT);
    *r_o = sat_u8(r);
    *g_o = sat_u8(g);
    *b_o = sat_u8(b);
}

extern "C" __global__ void ycc_from_rgb_u8_ycrcb(
    const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    unsigned char y, cr, cb;
    ycc_px_u8(__ldg(&src[b]), __ldg(&src[b + 1u]), __ldg(&src[b + 2u]), &y, &cr, &cb);
    dst[b] = y; dst[b + 1u] = cr; dst[b + 2u] = cb;
}

extern "C" __global__ void ycc_from_rgb_u8_yuv(
    const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    unsigned char y, cr, cb;
    ycc_px_u8(__ldg(&src[b]), __ldg(&src[b + 1u]), __ldg(&src[b + 2u]), &y, &cr, &cb);
    dst[b] = y; dst[b + 1u] = cb; dst[b + 2u] = cr;
}

extern "C" __global__ void rgb_from_ycc_u8_ycrcb(
    const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    unsigned char r, g, bl;
    rgb_px_u8(__ldg(&src[b]), __ldg(&src[b + 1u]), __ldg(&src[b + 2u]), &r, &g, &bl);
    dst[b] = r; dst[b + 1u] = g; dst[b + 2u] = bl;
}

extern "C" __global__ void rgb_from_ycc_u8_yuv(
    const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    unsigned char r, g, bl;
    rgb_px_u8(__ldg(&src[b]), __ldg(&src[b + 2u]), __ldg(&src[b + 1u]), &r, &g, &bl);
    dst[b] = r; dst[b + 1u] = g; dst[b + 2u] = bl;
}
"#;
const YCC_U8_FNS: &[&str] = &[
    "ycc_from_rgb_u8_ycrcb",
    "ycc_from_rgb_u8_yuv",
    "rgb_from_ycc_u8_ycrcb",
    "rgb_from_ycc_u8_yuv",
];

static YCC_F32_SRC: &str = r#"
// Forward (matches ycc_from_rgb_f32_px): y in [0,1], chroma centered at 0.5.
__device__ __forceinline__ void ycc_px_f32(
    float r, float g, float b, float* y_o, float* cr_o, float* cb_o)
{
    float y = F_YR * r + F_YG * g + F_YB * b;
    *y_o  = y;
    *cr_o = (r - y) * F_CR + 0.5f;
    *cb_o = (b - y) * F_CB + 0.5f;
}

// Inverse (matches rgb_from_ycc_f32_px).
__device__ __forceinline__ void rgb_px_f32(
    float y, float cr, float cb, float* r_o, float* g_o, float* b_o)
{
    float r = y + (cr - 0.5f) / F_CR;
    float b = y + (cb - 0.5f) / F_CB;
    float g = (y - F_YR * r - F_YB * b) / F_YG;
    *r_o = r; *g_o = g; *b_o = b;
}

extern "C" __global__ void ycc_from_rgb_f32_ycrcb(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    float y, cr, cb;
    ycc_px_f32(__ldg(&src[b]), __ldg(&src[b + 1u]), __ldg(&src[b + 2u]), &y, &cr, &cb);
    dst[b] = y; dst[b + 1u] = cr; dst[b + 2u] = cb;
}

extern "C" __global__ void ycc_from_rgb_f32_yuv(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    float y, cr, cb;
    ycc_px_f32(__ldg(&src[b]), __ldg(&src[b + 1u]), __ldg(&src[b + 2u]), &y, &cr, &cb);
    dst[b] = y; dst[b + 1u] = cb; dst[b + 2u] = cr;
}

extern "C" __global__ void rgb_from_ycc_f32_ycrcb(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    float r, g, bl;
    rgb_px_f32(__ldg(&src[b]), __ldg(&src[b + 1u]), __ldg(&src[b + 2u]), &r, &g, &bl);
    dst[b] = r; dst[b + 1u] = g; dst[b + 2u] = bl;
}

extern "C" __global__ void rgb_from_ycc_f32_yuv(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    float r, g, bl;
    rgb_px_f32(__ldg(&src[b]), __ldg(&src[b + 2u]), __ldg(&src[b + 1u]), &r, &g, &bl);
    dst[b] = r; dst[b + 1u] = g; dst[b + 2u] = bl;
}
"#;
const YCC_F32_FNS: &[&str] = &[
    "ycc_from_rgb_f32_ycrcb",
    "ycc_from_rgb_f32_yuv",
    "rgb_from_ycc_f32_ycrcb",
    "rgb_from_ycc_f32_yuv",
];

static YCC_U8: KernelSuiteCell = KernelSuiteCell::new();
static YCC_F32: KernelSuiteCell = KernelSuiteCell::new();

/// Suite entry index: forward/inverse × chroma order.
fn entry_index(forward: bool, order: ChromaOrder) -> usize {
    match (forward, order) {
        (true, ChromaOrder::YCrCb) => 0,
        (true, ChromaOrder::YuvCbCr) => 1,
        (false, ChromaOrder::YCrCb) => 2,
        (false, ChromaOrder::YuvCbCr) => 3,
    }
}

fn launch_u8(
    forward: bool,
    order: ChromaOrder,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    check_len("src", src.len(), npixels * 3)?;
    check_len("dst", dst.len(), npixels * 3)?;
    let kernel = get_kernel_suite(
        &YCC_U8,
        stream,
        YCC_U8_SRC,
        YCC_U8_FNS,
        entry_index(forward, order),
    )?;
    let n = npixels as u32;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&n)
        .launch_1d(n)?;
    Ok(())
}

fn launch_f32(
    forward: bool,
    order: ChromaOrder,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    check_len("src", src.len(), npixels * 3)?;
    check_len("dst", dst.len(), npixels * 3)?;
    let kernel = get_kernel_suite(
        &YCC_F32,
        stream,
        YCC_F32_SRC,
        YCC_F32_FNS,
        entry_index(forward, order),
    )?;
    let n = npixels as u32;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&n)
        .launch_1d(n)?;
    Ok(())
}

/// Launch RGB8 → YCbCr/YUV u8 (full-range Q14, bit-exact vs CPU).
pub fn launch_ycc_from_rgb_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
    order: ChromaOrder,
) -> Result<(), CudaColorError> {
    launch_u8(true, order, stream, src, dst, npixels)
}

/// Launch YCbCr/YUV u8 → RGB8 (full-range Q14, bit-exact vs CPU).
pub fn launch_rgb_from_ycc_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
    order: ChromaOrder,
) -> Result<(), CudaColorError> {
    launch_u8(false, order, stream, src, dst, npixels)
}

/// Launch RGB f32 → YCbCr/YUV f32 (full-range, values in `[0, 1]`).
pub fn launch_ycc_from_rgb_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
    order: ChromaOrder,
) -> Result<(), CudaColorError> {
    launch_f32(true, order, stream, src, dst, npixels)
}

/// Launch YCbCr/YUV f32 → RGB f32 (full-range, values in `[0, 1]`).
pub fn launch_rgb_from_ycc_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
    order: ChromaOrder,
) -> Result<(), CudaColorError> {
    launch_f32(false, order, stream, src, dst, npixels)
}

#[cfg(all(test, feature = "gpu-cuda"))]
mod tests {
    use super::*;
    use crate::gpu::color_cuda::test_utils::{default_stream, pattern_f32, pattern_u8};

    #[test]
    fn ycc_u8_roundtrip_entries_bit_exact_vs_cpu() {
        let stream = default_stream();
        let n = 640 * 480;
        let rgb = pattern_u8(n * 3);

        for order in [ChromaOrder::YCrCb, ChromaOrder::YuvCbCr] {
            // Forward.
            let mut cpu = vec![0u8; n * 3];
            crate::color::yuv::kernels::ycc_from_rgb_u8_scalar(&rgb, &mut cpu, n, order);
            let d_src = stream.clone_htod(&rgb).unwrap();
            let mut d_dst = stream.alloc_zeros::<u8>(n * 3).unwrap();
            launch_ycc_from_rgb_u8(&stream, &d_src, &mut d_dst, n, order).unwrap();
            let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(gpu, cpu, "forward u8 must be bit-exact");

            // Inverse (feed the forward output back).
            let mut cpu_rgb = vec![0u8; n * 3];
            crate::color::yuv::kernels::rgb_from_ycc_u8_scalar(&cpu, &mut cpu_rgb, n, order);
            let d_src2 = stream.clone_htod(&cpu).unwrap();
            let mut d_dst2 = stream.alloc_zeros::<u8>(n * 3).unwrap();
            launch_rgb_from_ycc_u8(&stream, &d_src2, &mut d_dst2, n, order).unwrap();
            let gpu_rgb: Vec<u8> = stream.clone_dtoh(&d_dst2).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(gpu_rgb, cpu_rgb, "inverse u8 must be bit-exact");
        }
    }

    #[test]
    fn ycc_f32_close_to_cpu() {
        let stream = default_stream();
        let n = 37 * 23;
        let rgb = pattern_f32(n * 3);

        let mut cpu = vec![0f32; n * 3];
        crate::color::yuv::kernels::ycc_from_rgb_f32_scalar(&rgb, &mut cpu, n, ChromaOrder::YCrCb);
        let d_src = stream.clone_htod(&rgb).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(n * 3).unwrap();
        launch_ycc_from_rgb_f32(&stream, &d_src, &mut d_dst, n, ChromaOrder::YCrCb).unwrap();
        let gpu: Vec<f32> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();

        let max_diff = gpu
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff <= 1e-3, "f32 ycc max diff {max_diff}");
    }
}
