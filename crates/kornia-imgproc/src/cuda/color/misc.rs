//! CUDA kernels for miscellaneous color ops: sepia and colormap application.
//!
//! Sepia u8 mirrors the CPU Q8 fixed-point path bit-for-bit
//! (`color/sepia.rs::sepia_u8_scalar`); sepia f32 mirrors the shared
//! `matrix3_affine_f32` MAC (`b + m0·c0 + m1·c1 + m2·c2` evaluation order).
//!
//! Colormap passes the 256-entry LUT as a device buffer of `u32` words packed
//! `0x00BBGGRR` (the CPU AVX2 gather layout, `color/colormap.rs`). Uploaded
//! LUTs are cached per `(ColormapType, device ordinal)` — 21 maps × 1 KiB max.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaSlice, CudaStream};

pub use crate::color::ColormapType;

use super::{check_len, get_kernel, launch_map, CudaColorError, KernelCell, PxPerThread};

// Word-vectorized 4 px/thread; Q8 coefficients round(coeff*256) — matches
// color/sepia.rs::Q bit-for-bit.
static SEPIA_U8_SRC: &str = r#"
__device__ __forceinline__ void sepia_px(
    unsigned int r, unsigned int g, unsigned int b,
    unsigned int* rr, unsigned int* gg, unsigned int* bb)
{
    *rr = min((101u * r + 197u * g + 48u * b + 128u) >> 8, 255u);
    *gg = min(( 89u * r + 176u * g + 43u * b + 128u) >> 8, 255u);
    *bb = min(( 70u * r + 137u * g + 34u * b + 128u) >> 8, 255u);
}

extern "C" __global__ void sepia_from_rgb_u8(
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
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            sepia_px(r[k], g[k], b[k], &r[k], &g[k], &b[k]);
        }
        store_c3_quad((unsigned int*)dst, q, r, g, b);
    } else {
        for (unsigned int i = p; i < npixels; ++i) {
            unsigned int s = i * 3u;
            unsigned int rr, gg, bb;
            sepia_px(__ldg(&src[s]), __ldg(&src[s + 1u]), __ldg(&src[s + 2u]),
                     &rr, &gg, &bb);
            dst[s] = (unsigned char)rr;
            dst[s + 1u] = (unsigned char)gg;
            dst[s + 2u] = (unsigned char)bb;
        }
    }
}
"#;

static SEPIA_F32_SRC: &str = r#"
extern "C" __global__ void sepia_from_rgb_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    float r  = __ldg(&src[b]);
    float g  = __ldg(&src[b + 1u]);
    float bl = __ldg(&src[b + 2u]);
    // Same MAC order as matrix3_affine_f32 with SEPIA_M and zero offset.
    dst[b]      = 0.393f * r + 0.769f * g + 0.189f * bl;
    dst[b + 1u] = 0.349f * r + 0.686f * g + 0.168f * bl;
    dst[b + 2u] = 0.272f * r + 0.534f * g + 0.131f * bl;
}
"#;

static SEPIA_U8: KernelCell = KernelCell::new();
static SEPIA_F32: KernelCell = KernelCell::new();

/// Launch sepia tone on RGB8 (Q8 fixed point, bit-exact vs CPU;
/// word-vectorized 4 px/thread).
pub fn launch_sepia_from_rgb_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = get_kernel(&SEPIA_U8, stream, SEPIA_U8_SRC, "sepia_from_rgb_u8")?;
    launch_map(kernel, stream, src, dst, npixels, 3, 3, PxPerThread::Four)
}

/// Launch sepia tone on RGB f32 (linear matrix, no clamp — like the CPU path).
pub fn launch_sepia_from_rgb_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = get_kernel(&SEPIA_F32, stream, SEPIA_F32_SRC, "sepia_from_rgb_f32")?;
    launch_map(kernel, stream, src, dst, npixels, 3, 3, PxPerThread::One)
}

static COLORMAP_SRC: &str = r#"
extern "C" __global__ void apply_colormap_u8(
    const unsigned char* __restrict__ src,
    const unsigned int* __restrict__ lut,   // 256 x 0x00BBGGRR words
    unsigned char* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int w = __ldg(&lut[__ldg(&src[i])]);
    unsigned int d = i * 3u;
    dst[d]      = (unsigned char)(w & 0xFFu);
    dst[d + 1u] = (unsigned char)((w >> 8) & 0xFFu);
    dst[d + 2u] = (unsigned char)((w >> 16) & 0xFFu);
}
"#;

static COLORMAP_KERNEL: KernelCell = KernelCell::new();

/// Device-resident packed LUTs, keyed by (colormap, device ordinal).
type LutCache = Mutex<HashMap<(ColormapType, usize), Arc<CudaSlice<u32>>>>;
static LUT_CACHE: OnceLock<LutCache> = OnceLock::new();

fn device_lut(
    stream: &Arc<CudaStream>,
    colormap: ColormapType,
) -> Result<Arc<CudaSlice<u32>>, CudaColorError> {
    let key = (colormap, stream.context().ordinal());
    let mut cache = LUT_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("colormap LUT cache mutex poisoned");
    if let Some(lut) = cache.get(&key) {
        return Ok(lut.clone());
    }
    // Pack the three channel LUTs into 0x00BBGGRR words (CPU AVX2 layout).
    let lut = colormap.lut();
    let mut packed = [0u32; 256];
    for (i, w) in packed.iter_mut().enumerate() {
        *w = (lut.r[i] as u32) | ((lut.g[i] as u32) << 8) | ((lut.b[i] as u32) << 16);
    }
    let dev = Arc::new(
        stream
            .clone_htod(&packed)
            .map_err(|e| CudaColorError::Cuda(e.to_string()))?,
    );
    cache.insert(key, dev.clone());
    Ok(dev)
}

/// Apply one of the 21 OpenCV colormaps to a Gray8 device buffer, producing
/// RGB8. Bit-exact vs the CPU LUT path (a table lookup has no rounding).
pub fn launch_apply_colormap_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
    colormap: ColormapType,
) -> Result<(), CudaColorError> {
    check_len("src", src.len(), npixels)?;
    check_len("dst", dst.len(), npixels * 3)?;
    let lut = device_lut(stream, colormap)?;
    let kernel = get_kernel(&COLORMAP_KERNEL, stream, COLORMAP_SRC, "apply_colormap_u8")?;
    let n = npixels as u32;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(lut.as_ref())
        .arg(dst)
        .arg(&n)
        .launch_1d(n)?;
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_u8};

    #[test]
    fn colormap_bit_exact_vs_cpu() {
        use kornia_image::{Image, ImageSize};
        let stream = default_stream();
        let (w, h) = (64usize, 48usize);
        let gray = pattern_u8(w * h);

        for colormap in [
            ColormapType::Jet,
            ColormapType::Viridis,
            ColormapType::Turbo,
        ] {
            let src = Image::<u8, 1>::new(
                ImageSize {
                    width: w,
                    height: h,
                },
                gray.clone(),
            )
            .unwrap();
            let mut cpu = Image::<u8, 3>::from_size_val(src.size(), 0).unwrap();
            crate::color::apply_colormap(&src, &mut cpu, colormap).unwrap();

            let d_src = stream.clone_htod(&gray).unwrap();
            let mut d_dst = stream.alloc_zeros::<u8>(w * h * 3).unwrap();
            launch_apply_colormap_u8(&stream, &d_src, &mut d_dst, w * h, colormap).unwrap();
            let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(gpu, cpu.as_slice(), "{colormap:?} must be bit-exact");
        }
    }

    #[test]
    fn sepia_u8_bit_exact_vs_cpu() {
        let stream = default_stream();
        let n = 640 * 480;
        let rgb = pattern_u8(n * 3);
        let mut cpu = vec![0u8; n * 3];
        crate::color::sepia::sepia_u8_scalar(&rgb, &mut cpu, n);

        let d_src = stream.clone_htod(&rgb).unwrap();
        let mut d_dst = stream.alloc_zeros::<u8>(n * 3).unwrap();
        launch_sepia_from_rgb_u8(&stream, &d_src, &mut d_dst, n).unwrap();
        let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(gpu, cpu);
    }
}
