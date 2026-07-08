//! CUDA kernels for video pixel-format decode/encode (u8 only).
//!
//! Mirrors Family B of `color/yuv/kernels.rs` bit-for-bit:
//!
//! - **Decode** (BT.601 *limited* range, `ITUR_BT_601` Q20): packed 4:2:2
//!   (YUYV/UYVY/YVYU → RGB) and planar 4:2:0 (NV12/NV21/I420/YV12 → RGB).
//!   Byte order / plane layout are uniform scalar or pointer arguments — one
//!   kernel body serves every variant with no per-pixel branching.
//! - **Mode decode**: `convert_yuyv_to_rgb_u8` Bt601Full / Bt709Full /
//!   Bt601Limited as three `extern "C"` entries (Q10 constants from
//!   `color/yuv/mod.rs`).
//! - **Encode** (BT.601 limited, Q8 66/129/25 luma, ±38/74/112 chroma):
//!   RGB → YUYV (chroma = rounded pair average) and RGB → NV12 (chroma =
//!   rounded 2×2 block average).
//!
//! These launchers are **low-level only**: the host-side `Yuyv8`/`Nv12`/…
//! wrappers carry raw `Vec<u8>` buffers, so there is no device-resident typed
//! wrapper for `ConvertColor` dispatch — callers manage `CudaSlice<u8>`
//! buffers directly. Layout matches the CPU functions: packed 4:2:2 is
//! 2 bytes/px; planar 4:2:0 is the full Y plane followed by the chroma
//! plane(s); RGB is 3 bytes/px.

use std::sync::{Arc, LazyLock};

use cudarc::driver::{CudaSlice, CudaStream};

pub use crate::color::yuv::kernels::{Packed422, Planar420};
pub use crate::color::YuvToRgbMode;

use super::{check_len, get_kernel, get_kernel_suite, CudaColorError, KernelCell, KernelSuiteCell};

use super::config_2d;

// Q20 BT.601-limited decode helpers — constants match color/yuv/kernels.rs:620-626.
static DECODE_COMMON: &str = r#"
#define ITUR_SHIFT 20
#define ITUR_HALF  (1 << 19)
#define D_CY  1220542
#define D_CUB 2116026
#define D_CUG (-409993)
#define D_CVG (-852492)
#define D_CVR 1673527

// Pre-scaled luma term max(0, Y-16) * CY — matches yy_term().
__device__ __forceinline__ int yy_term(int y) {
    return max(y - 16, 0) * D_CY;
}

// One luma sample + shared (U,V) -> RGB triple — matches decode_px().
__device__ __forceinline__ void decode_px(
    int yy, int u, int v, unsigned char* r, unsigned char* g, unsigned char* b)
{
    u -= 128;
    v -= 128;
    *b = sat_u8((yy + D_CUB * u + ITUR_HALF) >> ITUR_SHIFT);
    *g = sat_u8((yy + D_CUG * u + D_CVG * v + ITUR_HALF) >> ITUR_SHIFT);
    *r = sat_u8((yy + D_CVR * v + ITUR_HALF) >> ITUR_SHIFT);
}
"#;

// ── Packed 4:2:2 decode ──────────────────────────────────────────────────────

// One thread per 2-pixel group; byte offsets within the 4-byte group are
// uniform scalar args (fmt.offsets() on the Rust side), so YUYV/UYVY/YVYU
// share one branch-free body.
static PACKED422_SRC_TAIL: &str = r#"
extern "C" __global__ void rgb_from_packed422_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int ngroups,
    unsigned int o_y0, unsigned int o_u, unsigned int o_y1, unsigned int o_v)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    unsigned int base = g * 4u;
    int y0 = __ldg(&src[base + o_y0]);
    int u  = __ldg(&src[base + o_u]);
    int y1 = __ldg(&src[base + o_y1]);
    int v  = __ldg(&src[base + o_v]);
    unsigned int d = g * 6u;
    decode_px(yy_term(y0), u, v, &dst[d],      &dst[d + 1u], &dst[d + 2u]);
    decode_px(yy_term(y1), u, v, &dst[d + 3u], &dst[d + 4u], &dst[d + 5u]);
}
"#;

// ── Planar 4:2:0 decode ──────────────────────────────────────────────────────

// One thread per 2×2 luma block (chroma coordinate). The U/V pointers and the
// chroma index stride are computed on the Rust side (NV12/NV21 = interleaved
// plane with step 2 and swapped base offsets; I420/YV12 = separate planes with
// step 1), so all four formats share this body.
static PLANAR420_SRC_TAIL: &str = r#"
extern "C" __global__ void rgb_from_planar420_u8(
    const unsigned char* __restrict__ y_plane,
    const unsigned char* __restrict__ u_plane,
    const unsigned char* __restrict__ v_plane,
    unsigned char* __restrict__ dst,
    unsigned int width,
    unsigned int cw,
    unsigned int ch,
    unsigned int c_step)
{
    unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= cw || cy >= ch) return;

    unsigned int cidx = (cy * cw + cx) * c_step;
    int u = __ldg(&u_plane[cidx]);
    int v = __ldg(&v_plane[cidx]);

    unsigned int x = cx * 2u;
    unsigned int y_top = cy * 2u;
    #pragma unroll
    for (unsigned int dy = 0u; dy < 2u; ++dy) {
        unsigned int row = y_top + dy;
        unsigned int ybase = row * width + x;
        unsigned int dbase = (row * width + x) * 3u;
        #pragma unroll
        for (unsigned int dx = 0u; dx < 2u; ++dx) {
            int yy = yy_term(__ldg(&y_plane[ybase + dx]));
            decode_px(yy, u, v,
                &dst[dbase + dx * 3u],
                &dst[dbase + dx * 3u + 1u],
                &dst[dbase + dx * 3u + 2u]);
        }
    }
}
"#;

// ── YUYV mode decode (Bt601Full / Bt709Full / Bt601Limited) ─────────────────

// Q10 constants from color/yuv/mod.rs::yuv_to_rgb_u8_* — one entry per mode.
static YUYV_MODES_SRC: &str = r#"
__device__ __forceinline__ void px_bt601_full(
    int y, int u, int v, unsigned char* r, unsigned char* g, unsigned char* b)
{
    u -= 128; v -= 128;
    *r = sat_u8(y + ((1436 * v + 512) >> 10));
    *g = sat_u8(y - ((352 * u + 731 * v + 512) >> 10));
    *b = sat_u8(y + ((1815 * u + 512) >> 10));
}

__device__ __forceinline__ void px_bt709_full(
    int y, int u, int v, unsigned char* r, unsigned char* g, unsigned char* b)
{
    u -= 128; v -= 128;
    *r = sat_u8(y + ((1612 * v + 512) >> 10));
    *g = sat_u8(y - ((192 * u + 479 * v + 512) >> 10));
    *b = sat_u8(y + ((1900 * u + 512) >> 10));
}

__device__ __forceinline__ void px_bt601_limited(
    int y, int u, int v, unsigned char* r, unsigned char* g, unsigned char* b)
{
    int yv = ((y - 16) * 1192 + 512) >> 10;
    u -= 128; v -= 128;
    *r = sat_u8(yv + ((1634 * v + 512) >> 10));
    *g = sat_u8(yv - ((401 * u + 832 * v + 512) >> 10));
    *b = sat_u8(yv + ((2066 * u + 512) >> 10));
}

#define YUYV_MODE_KERNEL(NAME, PX)                                            \
extern "C" __global__ void NAME(                                              \
    const unsigned char* __restrict__ src,                                    \
    unsigned char* __restrict__ dst,                                          \
    unsigned int ngroups)                                                     \
{                                                                             \
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;                   \
    if (g >= ngroups) return;                                                 \
    unsigned int base = g * 4u;                                               \
    int y0 = __ldg(&src[base]);                                               \
    int u  = __ldg(&src[base + 1u]);                                          \
    int y1 = __ldg(&src[base + 2u]);                                          \
    int v  = __ldg(&src[base + 3u]);                                          \
    unsigned int d = g * 6u;                                                  \
    PX(y0, u, v, &dst[d],      &dst[d + 1u], &dst[d + 2u]);                   \
    PX(y1, u, v, &dst[d + 3u], &dst[d + 4u], &dst[d + 5u]);                   \
}

YUYV_MODE_KERNEL(yuyv_to_rgb_bt601_full_u8,    px_bt601_full)
YUYV_MODE_KERNEL(yuyv_to_rgb_bt709_full_u8,    px_bt709_full)
YUYV_MODE_KERNEL(yuyv_to_rgb_bt601_limited_u8, px_bt601_limited)
"#;
const YUYV_MODES_FNS: &[&str] = &[
    "yuyv_to_rgb_bt601_full_u8",
    "yuyv_to_rgb_bt709_full_u8",
    "yuyv_to_rgb_bt601_limited_u8",
];

// ── Encode (BT.601 limited, Q8) ──────────────────────────────────────────────

// Constants match color/yuv/kernels.rs:1136-1146; encode_y / encode_uv exactly.
static ENCODE_SRC: &str = r#"
#define ENC_SHIFT 8
#define ENC_HALF  (1 << 7)
#define E_YR 66
#define E_YG 129
#define E_YB 25
#define E_UR (-38)
#define E_UG (-74)
#define E_UB 112
#define E_VR 112
#define E_VG (-94)
#define E_VB (-18)

__device__ __forceinline__ unsigned char encode_y(int r, int g, int b) {
    return sat_u8(((E_YR * r + E_YG * g + E_YB * b + ENC_HALF) >> ENC_SHIFT) + 16);
}

__device__ __forceinline__ void encode_uv(
    int r, int g, int b, unsigned char* u, unsigned char* v)
{
    *u = sat_u8(((E_UR * r + E_UG * g + E_UB * b + ENC_HALF) >> ENC_SHIFT) + 128);
    *v = sat_u8(((E_VR * r + E_VG * g + E_VB * b + ENC_HALF) >> ENC_SHIFT) + 128);
}

// One thread per horizontal pixel pair -> Y0 U Y1 V (chroma = rounded average).
extern "C" __global__ void yuyv_from_rgb_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int ngroups)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    unsigned int s = g * 6u;
    int r0 = __ldg(&src[s]),      g0 = __ldg(&src[s + 1u]), b0 = __ldg(&src[s + 2u]);
    int r1 = __ldg(&src[s + 3u]), g1 = __ldg(&src[s + 4u]), b1 = __ldg(&src[s + 5u]);
    unsigned char u, v;
    encode_uv((r0 + r1 + 1) >> 1, (g0 + g1 + 1) >> 1, (b0 + b1 + 1) >> 1, &u, &v);
    unsigned int d = g * 4u;
    dst[d]      = encode_y(r0, g0, b0);
    dst[d + 1u] = u;
    dst[d + 2u] = encode_y(r1, g1, b1);
    dst[d + 3u] = v;
}

// One thread per 2×2 block -> 4 Y samples + 1 interleaved UV pair.
// dst layout: Y plane (w*h bytes) followed by the UV plane (w*h/2 bytes);
// y_len = w*h is passed so the kernel addresses the UV plane.
extern "C" __global__ void nv12_from_rgb_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int width,
    unsigned int cw,
    unsigned int ch,
    unsigned int y_len)
{
    unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= cw || cy >= ch) return;

    unsigned int x = cx * 2u;
    int rs = 0, gs = 0, bs = 0;
    #pragma unroll
    for (unsigned int dy = 0u; dy < 2u; ++dy) {
        unsigned int row = cy * 2u + dy;
        unsigned int s = (row * width + x) * 3u;
        #pragma unroll
        for (unsigned int dx = 0u; dx < 2u; ++dx) {
            int r = __ldg(&src[s + dx * 3u]);
            int g = __ldg(&src[s + dx * 3u + 1u]);
            int b = __ldg(&src[s + dx * 3u + 2u]);
            dst[row * width + x + dx] = encode_y(r, g, b);
            rs += r; gs += g; bs += b;
        }
    }
    unsigned char u, v;
    encode_uv((rs + 2) >> 2, (gs + 2) >> 2, (bs + 2) >> 2, &u, &v);
    unsigned int uvbase = y_len + cy * width + x;
    dst[uvbase]      = u;
    dst[uvbase + 1u] = v;
}
"#;
const ENCODE_FNS: &[&str] = &["yuyv_from_rgb_u8", "nv12_from_rgb_u8"];

// Joined once — get_kernel only reads the source on first compile, so a
// per-call format! would allocate ~2 KB per frame for nothing.
static PACKED422_SRC: LazyLock<String> =
    LazyLock::new(|| format!("{DECODE_COMMON}\n{PACKED422_SRC_TAIL}"));
static PLANAR420_SRC: LazyLock<String> =
    LazyLock::new(|| format!("{DECODE_COMMON}\n{PLANAR420_SRC_TAIL}"));

static PACKED422: KernelCell = KernelCell::new();
static PLANAR420: KernelCell = KernelCell::new();
static YUYV_MODES: KernelSuiteCell = KernelSuiteCell::new();
static ENCODE: KernelSuiteCell = KernelSuiteCell::new();

/// Byte offsets `(y0, u, y1, v)` within a 4-byte packed group — delegates to
/// the CPU path's format-defining table so the two can never drift.
fn packed_offsets(fmt: Packed422) -> (u32, u32, u32, u32) {
    let (y0, u, y1, v) = fmt.offsets();
    (y0 as u32, u as u32, y1 as u32, v as u32)
}

/// Decode a packed 4:2:2 device buffer (2 bytes/px) to RGB (3 bytes/px),
/// BT.601 limited range. `width` must be even. Bit-exact vs the CPU path.
pub fn launch_rgb_from_packed422_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    width: usize,
    height: usize,
    fmt: Packed422,
) -> Result<(), CudaColorError> {
    assert!(width.is_multiple_of(2), "packed 4:2:2 width must be even");
    let npixels = width * height;
    check_len("src", src.len(), npixels * 2)?;
    check_len("dst", dst.len(), npixels * 3)?;
    let kernel = get_kernel(&PACKED422, stream, &PACKED422_SRC, "rgb_from_packed422_u8")?;
    let ngroups = (npixels / 2) as u32;
    let (o_y0, o_u, o_y1, o_v) = packed_offsets(fmt);
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&ngroups)
        .arg(&o_y0)
        .arg(&o_u)
        .arg(&o_y1)
        .arg(&o_v)
        .launch_1d(ngroups)?;
    Ok(())
}

/// Decode a planar 4:2:0 device buffer (`Y` plane followed by chroma) to RGB,
/// BT.601 limited range. `src` holds the full frame (`w*h*3/2` bytes);
/// `width`/`height` must be even. Bit-exact vs the CPU path.
pub fn launch_rgb_from_planar420_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    width: usize,
    height: usize,
    fmt: Planar420,
) -> Result<(), CudaColorError> {
    assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "planar 4:2:0 dimensions must be even"
    );
    let y_len = width * height;
    let c_len = y_len / 4;
    check_len("src", src.len(), y_len + 2 * c_len)?;
    check_len("dst", dst.len(), y_len * 3)?;

    let kernel = get_kernel(&PLANAR420, stream, &PLANAR420_SRC, "rgb_from_planar420_u8")?;

    let y_plane = src.slice(0..y_len);
    // (u_plane, v_plane, chroma index step) per format — mirrors chroma_at().
    let (u_plane, v_plane, c_step) = match fmt {
        Planar420::Nv12 => (src.slice(y_len..), src.slice(y_len + 1..), 2u32),
        Planar420::Nv21 => (src.slice(y_len + 1..), src.slice(y_len..), 2u32),
        Planar420::I420 => (
            src.slice(y_len..y_len + c_len),
            src.slice(y_len + c_len..),
            1u32,
        ),
        Planar420::Yv12 => (
            src.slice(y_len + c_len..),
            src.slice(y_len..y_len + c_len),
            1u32,
        ),
    };

    let (w, cw, ch) = (width as u32, (width / 2) as u32, (height / 2) as u32);
    kernel
        .launch_builder(stream)
        .arg(&y_plane)
        .arg(&u_plane)
        .arg(&v_plane)
        .arg(dst)
        .arg(&w)
        .arg(&cw)
        .arg(&ch)
        .arg(&c_step)
        .launch_cfg(config_2d(cw, ch))?;
    Ok(())
}

/// Decode a YUYV device buffer to RGB with an explicit [`YuvToRgbMode`]
/// (Bt601Full / Bt709Full / Bt601Limited). Bit-exact vs
/// `convert_yuyv_to_rgb_u8`.
pub fn launch_convert_yuyv_to_rgb_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    width: usize,
    height: usize,
    mode: YuvToRgbMode,
) -> Result<(), CudaColorError> {
    assert!(width.is_multiple_of(2), "YUYV width must be even");
    let npixels = width * height;
    check_len("src", src.len(), npixels * 2)?;
    check_len("dst", dst.len(), npixels * 3)?;
    let index = match mode {
        YuvToRgbMode::Bt601Full => 0,
        YuvToRgbMode::Bt709Full => 1,
        YuvToRgbMode::Bt601Limited => 2,
    };
    let kernel = get_kernel_suite(&YUYV_MODES, stream, YUYV_MODES_SRC, YUYV_MODES_FNS, index)?;
    let ngroups = (npixels / 2) as u32;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&ngroups)
        .launch_1d(ngroups)?;
    Ok(())
}

/// Encode an RGB device buffer to packed YUYV (BT.601 limited; shared chroma =
/// rounded pair average). Bit-exact vs the CPU path.
pub fn launch_yuyv_from_rgb_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    width: usize,
    height: usize,
) -> Result<(), CudaColorError> {
    assert!(width.is_multiple_of(2), "YUYV width must be even");
    let npixels = width * height;
    check_len("src", src.len(), npixels * 3)?;
    check_len("dst", dst.len(), npixels * 2)?;
    let kernel = get_kernel_suite(&ENCODE, stream, ENCODE_SRC, ENCODE_FNS, 0)?;
    let ngroups = (npixels / 2) as u32;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&ngroups)
        .launch_1d(ngroups)?;
    Ok(())
}

/// Encode an RGB device buffer to NV12 (`dst` = Y plane then interleaved UV,
/// `w*h*3/2` bytes; chroma = rounded 2×2 block average). Bit-exact vs the CPU
/// path.
pub fn launch_nv12_from_rgb_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    width: usize,
    height: usize,
) -> Result<(), CudaColorError> {
    assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "NV12 dimensions must be even"
    );
    let y_len = width * height;
    check_len("src", src.len(), y_len * 3)?;
    check_len("dst", dst.len(), y_len + y_len / 2)?;
    let kernel = get_kernel_suite(&ENCODE, stream, ENCODE_SRC, ENCODE_FNS, 1)?;
    let (w, cw, ch, ylen32) = (
        width as u32,
        (width / 2) as u32,
        (height / 2) as u32,
        y_len as u32,
    );
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&w)
        .arg(&cw)
        .arg(&ch)
        .arg(&ylen32)
        .launch_cfg(config_2d(cw, ch))?;
    Ok(())
}

// ── DeviceVideoFrame: typed device-resident video buffer ────────────────────

/// The pixel layout of a [`DeviceVideoFrame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoFormat {
    /// Packed 4:2:2 (`Yuyv`/`Uyvy`/`Yvyu`), 2 bytes/px.
    Packed422(Packed422),
    /// Planar 4:2:0 (`Nv12`/`Nv21`/`I420`/`Yv12`), 1.5 bytes/px.
    Planar420(Planar420),
}

impl VideoFormat {
    /// Required buffer length in bytes for a `w × h` frame.
    /// (Kept consistent with `preprocess::SourceFormat::buffer_len` for the
    /// overlapping NV12/YUYV layouts — the enums live on different feature
    /// gates, so they cannot share code.)
    pub fn buffer_len(self, w: usize, h: usize) -> usize {
        match self {
            VideoFormat::Packed422(_) => w * h * 2,
            VideoFormat::Planar420(_) => w * h * 3 / 2,
        }
    }
}

/// A device-resident raw video frame with its format and geometry — the typed
/// counterpart of the host `Yuyv8`/`Nv12`/… wrappers, closing the gap that
/// video formats had no `ConvertColor` dispatch on the GPU.
///
/// The backing tensor carries its stream (see `Tensor::cuda_stream`), so
/// [`to_rgb`](Self::to_rgb) and the [`ConvertColor<Rgb8>`] impl need no
/// stream parameter; cross-stream destinations are event-fenced like every
/// other device conversion.
pub struct DeviceVideoFrame {
    tensor: kornia_tensor::Tensor<u8, 1>,
    width: usize,
    height: usize,
    format: VideoFormat,
}

impl DeviceVideoFrame {
    /// Upload raw host bytes in `format` layout (H2D copy).
    ///
    /// # Errors
    ///
    /// [`CudaColorError::SliceTooSmall`] if `data` is shorter than the format
    /// requires, or a CUDA error on upload failure.
    pub fn from_host(
        data: &[u8],
        width: usize,
        height: usize,
        format: VideoFormat,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, CudaColorError> {
        let need = format.buffer_len(width, height);
        check_len("src", data.len(), need)?;
        let slice = stream
            .clone_htod(&data[..need])
            .map_err(|e| CudaColorError::Cuda(e.to_string()))?;
        Ok(Self::from_cudaslice(
            slice,
            width,
            height,
            format,
            stream.clone(),
        ))
    }

    /// Zero-copy wrap an existing device buffer (e.g. straight from a capture
    /// pipeline). `slice.len()` must be at least the format's buffer length.
    pub fn from_cudaslice(
        slice: CudaSlice<u8>,
        width: usize,
        height: usize,
        format: VideoFormat,
        stream: Arc<CudaStream>,
    ) -> Self {
        let len = slice.len();
        Self {
            tensor: kornia_tensor::Tensor::from_cudaslice(slice, [len], stream),
            width,
            height,
            format,
        }
    }

    /// Frame width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Frame height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Pixel layout of this frame.
    pub fn format(&self) -> VideoFormat {
        self.format
    }

    /// Decode to a device-resident RGB image (BT.601 limited range,
    /// bit-exact vs the CPU path). `dst` must be device-resident with the
    /// frame's dimensions; a different stream is event-fenced.
    ///
    /// # Errors
    ///
    /// [`kornia_image::ImageError`] on size/residency mismatch or CUDA failure.
    pub fn to_rgb(
        &self,
        dst: &mut kornia_image::Image<u8, 3>,
    ) -> Result<(), kornia_image::ImageError> {
        use kornia_image::ImageError;
        if dst.width() != self.width || dst.height() != self.height {
            return Err(ImageError::InvalidImageSize(
                self.width,
                self.height,
                dst.cols(),
                dst.rows(),
            ));
        }
        let src_stream = self
            .tensor
            .cuda_stream()
            .ok_or_else(|| ImageError::Cuda("video frame is not device-backed".into()))?;
        let exec = crate::color::cuda_dispatch::device_exec_for(src_stream, &dst.0)?;
        let src_slice = self
            .tensor
            .as_cudaslice()
            .ok_or_else(|| ImageError::Cuda("video frame is not device-backed".into()))?;
        let dst_slice = dst
            .0
            .as_cudaslice_mut()
            .ok_or(ImageError::UnsupportedDevice)?;
        exec.run(|stream| {
            match self.format {
                VideoFormat::Packed422(fmt) => launch_rgb_from_packed422_u8(
                    stream,
                    src_slice,
                    dst_slice,
                    self.width,
                    self.height,
                    fmt,
                ),
                VideoFormat::Planar420(fmt) => launch_rgb_from_planar420_u8(
                    stream,
                    src_slice,
                    dst_slice,
                    self.width,
                    self.height,
                    fmt,
                ),
            }
            .map_err(ImageError::from)
        })
    }
}

impl crate::color::ConvertColor<kornia_image::color_spaces::Rgb8> for DeviceVideoFrame {
    fn convert(
        &self,
        dst: &mut kornia_image::color_spaces::Rgb8,
    ) -> Result<(), kornia_image::ImageError> {
        self.to_rgb(&mut dst.0)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use kornia_image::{Image, ImageSize};

    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_u8};

    const W: usize = 64;
    const H: usize = 48;

    #[test]
    fn device_video_frame_convert_matches_cpu() {
        use crate::color::ConvertColor;
        use kornia_image::color_spaces::Rgb8;
        use kornia_image::ImageSize;

        let stream = default_stream();
        let size = ImageSize {
            width: W,
            height: H,
        };

        for (fmt, len) in [
            (VideoFormat::Packed422(Packed422::Yuyv), W * H * 2),
            (VideoFormat::Planar420(Planar420::Nv12), W * H * 3 / 2),
        ] {
            let bytes = pattern_u8(len);
            let frame = DeviceVideoFrame::from_host(&bytes, W, H, fmt, &stream).unwrap();
            let mut rgb_d = Rgb8::zeros_cuda(size, &stream).unwrap();
            frame.convert(&mut rgb_d).unwrap();
            let rgb = rgb_d.to_host_owned().unwrap();

            let mut cpu = vec![0u8; W * H * 3];
            match fmt {
                VideoFormat::Packed422(f) => {
                    crate::color::yuv::kernels::rgb_from_packed422(&bytes, &mut cpu, W, H, f)
                }
                VideoFormat::Planar420(f) => crate::color::yuv::kernels::rgb_from_planar420(
                    &bytes[..W * H],
                    &bytes[W * H..],
                    &[],
                    &mut cpu,
                    W,
                    H,
                    f,
                ),
            }
            assert_eq!(
                rgb.as_slice(),
                &cpu[..],
                "{fmt:?} device frame must match CPU"
            );
        }
    }

    #[test]
    fn packed422_decode_bit_exact_vs_cpu() {
        let stream = default_stream();
        let yuv = pattern_u8(W * H * 2);
        for fmt in [Packed422::Yuyv, Packed422::Uyvy, Packed422::Yvyu] {
            let mut cpu = vec![0u8; W * H * 3];
            crate::color::yuv::kernels::rgb_from_packed422(&yuv, &mut cpu, W, H, fmt);

            let d_src = stream.clone_htod(&yuv).unwrap();
            let mut d_dst = stream.alloc_zeros::<u8>(W * H * 3).unwrap();
            launch_rgb_from_packed422_u8(&stream, &d_src, &mut d_dst, W, H, fmt).unwrap();
            let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(gpu, cpu, "packed 4:2:2 decode must be bit-exact");
        }
    }

    #[test]
    fn planar420_decode_bit_exact_vs_cpu() {
        let stream = default_stream();
        let frame = pattern_u8(W * H * 3 / 2);
        let y_len = W * H;
        let c_len = y_len / 4;

        for fmt in [
            Planar420::Nv12,
            Planar420::Nv21,
            Planar420::I420,
            Planar420::Yv12,
        ] {
            // CPU path takes pre-split planes (c1 empty for NV formats).
            let (c0, c1): (&[u8], &[u8]) = match fmt {
                Planar420::Nv12 | Planar420::Nv21 => (&frame[y_len..], &[]),
                Planar420::I420 | Planar420::Yv12 => {
                    (&frame[y_len..y_len + c_len], &frame[y_len + c_len..])
                }
            };
            let mut cpu = vec![0u8; y_len * 3];
            crate::color::yuv::kernels::rgb_from_planar420(
                &frame[..y_len],
                c0,
                c1,
                &mut cpu,
                W,
                H,
                fmt,
            );

            let d_src = stream.clone_htod(&frame).unwrap();
            let mut d_dst = stream.alloc_zeros::<u8>(y_len * 3).unwrap();
            launch_rgb_from_planar420_u8(&stream, &d_src, &mut d_dst, W, H, fmt).unwrap();
            let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(gpu, cpu, "planar 4:2:0 decode must be bit-exact");
        }
    }

    #[test]
    fn yuyv_mode_decode_bit_exact_vs_cpu() {
        let stream = default_stream();
        let yuv = pattern_u8(W * H * 2);
        for mode in [
            YuvToRgbMode::Bt601Full,
            YuvToRgbMode::Bt709Full,
            YuvToRgbMode::Bt601Limited,
        ] {
            let mut cpu = Image::<u8, 3>::from_size_val(
                ImageSize {
                    width: W,
                    height: H,
                },
                0,
            )
            .unwrap();
            crate::color::convert_yuyv_to_rgb_u8(&yuv, &mut cpu, mode).unwrap();

            let d_src = stream.clone_htod(&yuv).unwrap();
            let mut d_dst = stream.alloc_zeros::<u8>(W * H * 3).unwrap();
            launch_convert_yuyv_to_rgb_u8(&stream, &d_src, &mut d_dst, W, H, mode).unwrap();
            let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(gpu, cpu.as_slice(), "mode decode must be bit-exact");
        }
    }

    #[test]
    fn encode_bit_exact_vs_cpu() {
        let stream = default_stream();
        let rgb = pattern_u8(W * H * 3);

        // YUYV encode.
        let mut cpu_yuyv = vec![0u8; W * H * 2];
        crate::color::yuv::kernels::yuyv_from_rgb(&rgb, &mut cpu_yuyv, W, H);
        let d_src = stream.clone_htod(&rgb).unwrap();
        let mut d_dst = stream.alloc_zeros::<u8>(W * H * 2).unwrap();
        launch_yuyv_from_rgb_u8(&stream, &d_src, &mut d_dst, W, H).unwrap();
        let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(gpu, cpu_yuyv, "YUYV encode must be bit-exact");

        // NV12 encode.
        let mut cpu_y = vec![0u8; W * H];
        let mut cpu_uv = vec![0u8; W * H / 2];
        crate::color::yuv::kernels::nv12_from_rgb(&rgb, &mut cpu_y, &mut cpu_uv, W, H);
        let mut d_nv12 = stream.alloc_zeros::<u8>(W * H * 3 / 2).unwrap();
        launch_nv12_from_rgb_u8(&stream, &d_src, &mut d_nv12, W, H).unwrap();
        let gpu_nv12: Vec<u8> = stream.clone_dtoh(&d_nv12).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            &gpu_nv12[..W * H],
            &cpu_y[..],
            "NV12 Y plane must be bit-exact"
        );
        assert_eq!(
            &gpu_nv12[W * H..],
            &cpu_uv[..],
            "NV12 UV plane must be bit-exact"
        );
    }
}
