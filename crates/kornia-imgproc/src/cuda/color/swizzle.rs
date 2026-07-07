//! CUDA kernels for the bandwidth-bound channel & alpha conversions
//! (BGR↔RGB swap, RGBA/BGRA add/strip, optional alpha blend over a background).
//!
//! Mirrors `color/rgb/kernels.rs` (u8/f32 swizzles) and the `alpha_blend`
//! helper in `color/rgb/mod.rs` (f32 blend with `roundf`, matching Rust
//! `f32::round` half-away-from-zero semantics).

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::{
    check_len, get_kernel, get_kernel_suite, launch_map, CudaColorError, KernelCell,
    KernelSuiteCell, PxPerThread,
};

// Word-vectorized: 4 px/thread via the shared c3 quad helpers (bit-exact —
// pure byte permutation). Partial tail falls back to byte addressing.
static BGR_FROM_RGB_U8_SRC: &str = r#"
extern "C" __global__ void bgr_from_rgb_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int q = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int nquads = (npixels + 3u) / 4u;
    if (q >= nquads) return;
    unsigned int p = q * 4u;
    if (p + 4u <= npixels) {
        unsigned int r[4], g[4], b[4];
        load_c3_quad((const unsigned int*)src, q, r, g, b);
        store_c3_quad((unsigned int*)dst, q, b, g, r);
    } else {
        for (unsigned int i = p; i < npixels; ++i) {
            unsigned int s = i * 3u;
            unsigned char r = __ldg(&src[s]);
            unsigned char g = __ldg(&src[s + 1u]);
            unsigned char bl = __ldg(&src[s + 2u]);
            dst[s] = bl; dst[s + 1u] = g; dst[s + 2u] = r;
        }
    }
}
"#;

static BGR_FROM_RGB_F32_SRC: &str = r#"
extern "C" __global__ void bgr_from_rgb_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    float r = __ldg(&src[b]);
    float g = __ldg(&src[b + 1u]);
    float bl = __ldg(&src[b + 2u]);
    dst[b] = bl; dst[b + 1u] = g; dst[b + 2u] = r;
}
"#;

// 3→4 expansions: copy (or swap) RGB and append an opaque alpha.
static EXPAND_U8_SRC: &str = r#"
extern "C" __global__ void rgba_from_rgb_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int s = i * 3u, d = i * 4u;
    dst[d] = __ldg(&src[s]);
    dst[d + 1u] = __ldg(&src[s + 1u]);
    dst[d + 2u] = __ldg(&src[s + 2u]);
    dst[d + 3u] = 255;
}

extern "C" __global__ void bgra_from_rgb_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int s = i * 3u, d = i * 4u;
    dst[d] = __ldg(&src[s + 2u]);
    dst[d + 1u] = __ldg(&src[s + 1u]);
    dst[d + 2u] = __ldg(&src[s]);
    dst[d + 3u] = 255;
}
"#;
const EXPAND_U8_FNS: &[&str] = &["rgba_from_rgb_u8", "bgra_from_rgb_u8"];

static EXPAND_F32_SRC: &str = r#"
extern "C" __global__ void rgba_from_rgb_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int s = i * 3u, d = i * 4u;
    dst[d] = __ldg(&src[s]);
    dst[d + 1u] = __ldg(&src[s + 1u]);
    dst[d + 2u] = __ldg(&src[s + 2u]);
    dst[d + 3u] = 1.0f;
}

extern "C" __global__ void bgra_from_rgb_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int s = i * 3u, d = i * 4u;
    dst[d] = __ldg(&src[s + 2u]);
    dst[d + 1u] = __ldg(&src[s + 1u]);
    dst[d + 2u] = __ldg(&src[s]);
    dst[d + 3u] = 1.0f;
}
"#;
const EXPAND_F32_FNS: &[&str] = &["rgba_from_rgb_f32", "bgra_from_rgb_f32"];

// 4→3 strips. `swap` variants read BGRA; `_bg` variants alpha-blend over a
// uniform background — separate entries, so no per-pixel branch on has_bg.
// Blend math matches color/rgb/mod.rs::alpha_blend (f32, roundf = Rust round).
static STRIP_U8_SRC: &str = r#"
__device__ __forceinline__ unsigned char blend1(
    unsigned char c, unsigned char bg, float alpha)
{
    return (unsigned char)roundf((float)c * alpha + (float)bg * (1.0f - alpha));
}

extern "C" __global__ void rgb_from_rgba_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int s = i * 4u, d = i * 3u;
    dst[d] = __ldg(&src[s]);
    dst[d + 1u] = __ldg(&src[s + 1u]);
    dst[d + 2u] = __ldg(&src[s + 2u]);
}

extern "C" __global__ void rgb_from_bgra_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int s = i * 4u, d = i * 3u;
    dst[d] = __ldg(&src[s + 2u]);
    dst[d + 1u] = __ldg(&src[s + 1u]);
    dst[d + 2u] = __ldg(&src[s]);
}

extern "C" __global__ void rgb_from_rgba_bg_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int npixels,
    unsigned int bg_r, unsigned int bg_g, unsigned int bg_b)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int s = i * 4u, d = i * 3u;
    float alpha = (float)__ldg(&src[s + 3u]) / 255.0f;
    dst[d]      = blend1(__ldg(&src[s]),      (unsigned char)bg_r, alpha);
    dst[d + 1u] = blend1(__ldg(&src[s + 1u]), (unsigned char)bg_g, alpha);
    dst[d + 2u] = blend1(__ldg(&src[s + 2u]), (unsigned char)bg_b, alpha);
}

extern "C" __global__ void rgb_from_bgra_bg_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int npixels,
    unsigned int bg_r, unsigned int bg_g, unsigned int bg_b)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int s = i * 4u, d = i * 3u;
    float alpha = (float)__ldg(&src[s + 3u]) / 255.0f;
    dst[d]      = blend1(__ldg(&src[s + 2u]), (unsigned char)bg_r, alpha);
    dst[d + 1u] = blend1(__ldg(&src[s + 1u]), (unsigned char)bg_g, alpha);
    dst[d + 2u] = blend1(__ldg(&src[s]),      (unsigned char)bg_b, alpha);
}
"#;
const STRIP_U8_FNS: &[&str] = &[
    "rgb_from_rgba_u8",
    "rgb_from_bgra_u8",
    "rgb_from_rgba_bg_u8",
    "rgb_from_bgra_bg_u8",
];

static BGR_FROM_RGB_U8: KernelCell = KernelCell::new();
static BGR_FROM_RGB_F32: KernelCell = KernelCell::new();
static EXPAND_U8: KernelSuiteCell = KernelSuiteCell::new();
static EXPAND_F32: KernelSuiteCell = KernelSuiteCell::new();
static STRIP_U8: KernelSuiteCell = KernelSuiteCell::new();

/// Launch RGB8 ↔ BGR8 channel swap (symmetric — also BGR → RGB;
/// word-vectorized 4 px/thread).
pub fn launch_bgr_from_rgb_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = get_kernel(
        &BGR_FROM_RGB_U8,
        stream,
        BGR_FROM_RGB_U8_SRC,
        "bgr_from_rgb_u8",
    )?;
    launch_map(kernel, stream, src, dst, npixels, 3, 3, PxPerThread::Four)
}

/// Launch RGB f32 ↔ BGR f32 channel swap (symmetric).
pub fn launch_bgr_from_rgb_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = get_kernel(
        &BGR_FROM_RGB_F32,
        stream,
        BGR_FROM_RGB_F32_SRC,
        "bgr_from_rgb_f32",
    )?;
    launch_map(kernel, stream, src, dst, npixels, 3, 3, PxPerThread::One)
}

/// Which 3→4 expansion to run.
#[derive(Clone, Copy)]
enum Expand {
    Rgba = 0,
    Bgra = 1,
}

fn launch_expand_u8(
    which: Expand,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = get_kernel_suite(
        &EXPAND_U8,
        stream,
        EXPAND_U8_SRC,
        EXPAND_U8_FNS,
        which as usize,
    )?;
    launch_map(kernel, stream, src, dst, npixels, 3, 4, PxPerThread::One)
}

fn launch_expand_f32(
    which: Expand,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = get_kernel_suite(
        &EXPAND_F32,
        stream,
        EXPAND_F32_SRC,
        EXPAND_F32_FNS,
        which as usize,
    )?;
    launch_map(kernel, stream, src, dst, npixels, 3, 4, PxPerThread::One)
}

/// Launch RGB8 → RGBA8 (append opaque alpha 255).
pub fn launch_rgba_from_rgb_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    launch_expand_u8(Expand::Rgba, stream, src, dst, npixels)
}

/// Launch RGB8 → BGRA8 (swap R/B, append opaque alpha 255).
pub fn launch_bgra_from_rgb_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    launch_expand_u8(Expand::Bgra, stream, src, dst, npixels)
}

/// Launch RGB f32 → RGBA f32 (append opaque alpha 1.0).
pub fn launch_rgba_from_rgb_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    launch_expand_f32(Expand::Rgba, stream, src, dst, npixels)
}

/// Launch RGB f32 → BGRA f32 (swap R/B, append opaque alpha 1.0).
pub fn launch_bgra_from_rgb_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    launch_expand_f32(Expand::Bgra, stream, src, dst, npixels)
}

/// Launch RGBA8/BGRA8 → RGB8 with an optional uniform background blend.
///
/// `swapped = false` reads RGBA order, `true` reads BGRA. With
/// `background = Some(bg)` each channel is alpha-blended over `bg` exactly
/// like the CPU `alpha_blend` (f32 math, round-half-away-from-zero); with
/// `None` the alpha channel is dropped.
pub fn launch_rgb_from_rgba_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
    swapped: bool,
    background: Option<[u8; 3]>,
) -> Result<(), CudaColorError> {
    check_len("src", src.len(), npixels * 4)?;
    check_len("dst", dst.len(), npixels * 3)?;
    let index = match (swapped, background.is_some()) {
        (false, false) => 0, // rgb_from_rgba_u8
        (true, false) => 1,  // rgb_from_bgra_u8
        (false, true) => 2,  // rgb_from_rgba_bg_u8
        (true, true) => 3,   // rgb_from_bgra_bg_u8
    };
    let kernel = get_kernel_suite(&STRIP_U8, stream, STRIP_U8_SRC, STRIP_U8_FNS, index)?;
    let n = npixels as u32;
    // Hoisted so the borrows live as long as the launch builder.
    let bg = background.unwrap_or_default();
    let (bg_r, bg_g, bg_b) = (bg[0] as u32, bg[1] as u32, bg[2] as u32);
    let builder = kernel.launch_builder(stream).arg(src).arg(dst).arg(&n);
    if background.is_some() {
        builder.arg(&bg_r).arg(&bg_g).arg(&bg_b).launch_1d(n)?;
    } else {
        builder.launch_1d(n)?;
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_u8};

    #[test]
    fn bgr_swap_u8_bit_exact_vs_cpu() {
        let stream = default_stream();
        let n = 37 * 23;
        let src = pattern_u8(n * 3);
        let mut cpu = vec![0u8; n * 3];
        crate::color::rgb::kernels::bgr_from_rgb_u8(&src, &mut cpu, n);

        let d_src = stream.clone_htod(&src).unwrap();
        let mut d_dst = stream.alloc_zeros::<u8>(n * 3).unwrap();
        launch_bgr_from_rgb_u8(&stream, &d_src, &mut d_dst, n).unwrap();
        let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn expand_and_strip_u8_bit_exact_vs_cpu() {
        let stream = default_stream();
        let n = 640 * 480;
        let rgb = pattern_u8(n * 3);

        // RGB → RGBA / BGRA
        type Launch = fn(
            &Arc<CudaStream>,
            &CudaSlice<u8>,
            &mut CudaSlice<u8>,
            usize,
        ) -> Result<(), CudaColorError>;
        type CpuFn = fn(&[u8], &mut [u8], usize);
        for (launch, cpu_fn) in [
            (
                launch_rgba_from_rgb_u8 as Launch,
                crate::color::rgb::kernels::rgba_from_rgb_u8 as CpuFn,
            ),
            (
                launch_bgra_from_rgb_u8 as Launch,
                crate::color::rgb::kernels::bgra_from_rgb_u8 as CpuFn,
            ),
        ] {
            let mut cpu = vec![0u8; n * 4];
            cpu_fn(&rgb, &mut cpu, n);
            let d_src = stream.clone_htod(&rgb).unwrap();
            let mut d_dst = stream.alloc_zeros::<u8>(n * 4).unwrap();
            launch(&stream, &d_src, &mut d_dst, n).unwrap();
            let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(gpu, cpu);
        }

        // RGBA → RGB (drop alpha) — plain 4→3 copy.
        let rgba = pattern_u8(n * 4);
        let d_src = stream.clone_htod(&rgba).unwrap();
        let mut d_dst = stream.alloc_zeros::<u8>(n * 3).unwrap();
        launch_rgb_from_rgba_u8(&stream, &d_src, &mut d_dst, n, false, None).unwrap();
        let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        for i in 0..n {
            assert_eq!(&gpu[i * 3..i * 3 + 3], &rgba[i * 4..i * 4 + 3]);
        }
    }

    #[test]
    fn strip_with_background_close_to_cpu_blend() {
        let stream = default_stream();
        let n = 64 * 64;
        let rgba = pattern_u8(n * 4);
        let bg = [10u8, 200, 45];

        // Oracle = the production CPU path, so the GPU is always validated
        // against whatever blend semantics ship.
        use kornia_image::{Image, ImageSize};
        let size = ImageSize {
            width: 64,
            height: 64,
        };
        let src_img = Image::<u8, 4>::new(size, rgba.clone()).unwrap();
        let mut cpu_img = Image::<u8, 3>::from_size_val(size, 0).unwrap();
        crate::color::rgb_from_rgba(&src_img, &mut cpu_img, Some(bg)).unwrap();
        let cpu = cpu_img.as_slice().to_vec();

        let d_src = stream.clone_htod(&rgba).unwrap();
        let mut d_dst = stream.alloc_zeros::<u8>(n * 3).unwrap();
        launch_rgb_from_rgba_u8(&stream, &d_src, &mut d_dst, n, false, Some(bg)).unwrap();
        let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();

        // FMA contraction on the device may shift a round-to-nearest tie by 1.
        let max_diff = gpu
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (*a as i16 - *b as i16).unsigned_abs())
            .max()
            .unwrap();
        assert!(max_diff <= 1, "background blend differs by {max_diff} > 1");
    }
}
