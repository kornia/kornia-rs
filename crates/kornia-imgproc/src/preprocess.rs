//! GPU image → model-input preprocessing: resize (+ optional pad) and normalize a
//! device-resident [`Image`] straight into a CHW `f32` [`Tensor`] in one CUDA kernel.
//!
//! This is the step every CNN / transformer vision model needs before inference:
//! take a camera frame of arbitrary size and produce the fixed `[1, 3, H, W]`
//! CHW `f32` tensor the network expects, in `[0, 1]` RGB. It runs entirely on the
//! GPU (kornia's [`CudaKernel`]) with no host round-trip — the source image
//! already lives on the device (`image.to_cuda(&stream)`), and the output tensor
//! is written in place so it can be reused frame to frame.
//!
//! Enabled by the `cudarc` feature.
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use cudarc::driver::CudaContext;
//! use kornia_image::Image;
//! use kornia_tensor::zeros_cuda;
//! use kornia_imgproc::preprocess::Preprocessor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let stream = CudaContext::new(0)?.default_stream();
//!
//! // Build once (compiles the kernel); reuse across frames.
//! let pre = Preprocessor::letterbox(stream.clone())?;
//!
//! // Model input buffer, allocated once and reused.
//! let mut input = zeros_cuda::<f32, 4>([1, 3, 640, 640], &stream)?;
//!
//! // Per frame: upload the camera image and preprocess into `input`.
//! # let host: Image<u8, 3> = Image::from_size_val([1280, 720].into(), 0)?;
//! let frame = Image(host.0.to_cuda(&stream)?);   // device-resident Image<u8, 3>
//! pre.run(&frame, &mut input)?;                  // input is now [1,3,640,640] RGB [0,1]
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use kornia_image::Image;
use kornia_tensor::{CudaError, CudaKernel, Tensor};

/// How the source image is fit into the model's output rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    /// Aspect-preserving scale + grey (114) pad — the YOLO / XFeat convention.
    /// The whole image is kept; the unused border is filled with grey.
    Letterbox,
    /// Anisotropic stretch to the full target, no padding — the RT-DETR / RF-DETR
    /// convention. The aspect ratio is not preserved.
    Stretch,
}

/// Pixel format of the source buffer — color decode is **fused** into the
/// resize kernel, so a raw camera frame (NV12/YUYV straight from a capture
/// pipeline) becomes a normalized CHW tensor in a single launch with no
/// intermediate RGB image in device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SourceFormat {
    /// Interleaved RGB, 3 bytes/px (the default).
    #[default]
    Rgb8,
    /// Interleaved BGR, 3 bytes/px (OpenCV convention).
    Bgr8,
    /// Interleaved RGBA, 4 bytes/px (alpha ignored).
    Rgba8,
    /// Interleaved BGRA, 4 bytes/px (alpha ignored).
    Bgra8,
    /// Single-channel grayscale, 1 byte/px (replicated to RGB).
    Gray8,
    /// Planar 4:2:0: full-res Y plane then interleaved half-res UV
    /// (`w*h*3/2` bytes). BT.601 limited range, byte-identical to
    /// `gpu::color_cuda::video`. Width and height must be even.
    Nv12,
    /// Packed 4:2:2 `Y0 U Y1 V`, 2 bytes/px. BT.601 limited range.
    /// Width must be even.
    Yuyv,
}

/// Interpolation used by the fused resize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sampling {
    /// 4-tap bilinear, clamp-to-edge (the default).
    #[default]
    Bilinear,
    /// Nearest texel (round-half-up), clamp-to-edge.
    Nearest,
}

/// Configuration for [`Preprocessor::with_options`].
#[derive(Debug, Clone, Copy)]
pub struct PreprocessorOptions {
    /// How the source is fit into the output rectangle.
    pub mode: ResizeMode,
    /// Source pixel format (decode is fused into the resize).
    pub format: SourceFormat,
    /// Resize interpolation.
    pub sampling: Sampling,
    /// Write `f16` output instead of `f32` (use [`Preprocessor::run_raw_f16`]).
    pub f16_output: bool,
}

impl Default for PreprocessorOptions {
    fn default() -> Self {
        Self {
            mode: ResizeMode::Letterbox,
            format: SourceFormat::Rgb8,
            sampling: Sampling::Bilinear,
            f16_output: false,
        }
    }
}

impl SourceFormat {
    /// Kernel entry point name for a (format, sampling, dtype) combination.
    fn entry(self, sampling: Sampling, f16: bool) -> String {
        let base = match self {
            SourceFormat::Rgb8 => "rgb8",
            SourceFormat::Bgr8 => "bgr8",
            SourceFormat::Rgba8 => "rgba8",
            SourceFormat::Bgra8 => "bgra8",
            SourceFormat::Gray8 => "gray8",
            SourceFormat::Nv12 => "nv12",
            SourceFormat::Yuyv => "yuyv",
        };
        let mut name = format!("resize_pad_to_chw_{base}");
        if matches!(sampling, Sampling::Nearest) {
            name.push_str("_nearest");
        }
        if f16 {
            name.push_str("_f16");
        }
        name
    }

    /// Required source buffer length in bytes for a `w × h` frame.
    fn buffer_len(self, w: usize, h: usize) -> usize {
        match self {
            SourceFormat::Rgb8 | SourceFormat::Bgr8 => w * h * 3,
            SourceFormat::Rgba8 | SourceFormat::Bgra8 => w * h * 4,
            SourceFormat::Gray8 => w * h,
            SourceFormat::Nv12 => w * h * 3 / 2,
            SourceFormat::Yuyv => w * h * 2,
        }
    }

    /// Byte stride of the primary plane row (passed to the kernel).
    fn pitch(self, w: usize) -> usize {
        match self {
            SourceFormat::Rgb8 | SourceFormat::Bgr8 => w * 3,
            SourceFormat::Rgba8 | SourceFormat::Bgra8 => w * 4,
            SourceFormat::Gray8 | SourceFormat::Nv12 => w,
            SourceFormat::Yuyv => w * 2,
        }
    }
}

/// Errors from GPU preprocessing.
#[derive(Debug, thiserror::Error)]
pub enum PreprocessError {
    /// A CUDA error from kernel compilation or launch.
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
    /// The source image is host-resident; call `image.to_cuda(&stream)` first.
    #[error("source image is not device-resident (call `.to_cuda(&stream)` first)")]
    NotDeviceImage,
    /// The destination tensor is host-resident; allocate it with `zeros_cuda`.
    #[error("destination tensor is not device-resident")]
    NotDeviceTensor,
    /// The source buffer is smaller than the format requires for the given size.
    #[error("source buffer holds {got} bytes; {format:?} at {width}x{height} needs {need}")]
    SourceBufferTooSmall {
        /// Source pixel format.
        format: SourceFormat,
        /// Frame width in pixels.
        width: usize,
        /// Frame height in pixels.
        height: usize,
        /// Bytes provided.
        got: usize,
        /// Bytes required.
        need: usize,
    },
    /// `run` on a typed `Image<u8, 3>` requires a 3-byte-per-pixel format.
    #[error("run() requires SourceFormat::Rgb8 or Bgr8 (got {0:?}); use run_raw()")]
    FormatNeedsRawBuffer(SourceFormat),
    /// The destination dtype does not match how the preprocessor was built
    /// (`f16_output` selects between [`Preprocessor::run_raw`] and
    /// [`Preprocessor::run_raw_f16`]).
    #[error("preprocessor was built with f16_output={built_f16}; use the matching run method")]
    OutputDtypeMismatch {
        /// Whether the compiled kernel writes f16.
        built_f16: bool,
    },
    /// A normalization std component was zero.
    #[error("normalization std must be non-zero in every channel")]
    ZeroStd,
    /// Subsampled formats require even frame dimensions.
    #[error("{format:?} requires even dimensions (got {width}x{height})")]
    OddDimensions {
        /// Source pixel format.
        format: SourceFormat,
        /// Frame width in pixels.
        width: usize,
        /// Frame height in pixels.
        height: usize,
    },
}

// 1-D grid over the `dst_w * dst_h` output pixels. The (ox, oy) pair is recovered
// from the flat index — consecutive threads map to consecutive `ox`, so global
// writes stay coalesced. Each source format gets its own `extern "C"` entry
// (generated by `PREPROCESS_KERNEL`, selected on the Rust side — no per-pixel
// format branching): the four bilinear taps call a per-format `fetch_*` that
// decodes ONE RGB pixel in place, so color conversion is **fused** into the
// resize instead of costing a full-image round-trip through device memory.
//
// The u8 decode math is byte-identical to `gpu/color_cuda` (Q20 BT.601-limited
// for NV12/YUYV), so fused output equals the decode-then-preprocess chain.
const KERNEL_SRC: &str = r#"
// f32 -> f16 bits via the hardware cvt instruction — avoids depending on
// cuda_fp16.h, which NVRTC cannot resolve without an include search path.
// The u16 bit pattern is exactly IEEE binary16 (what half::f16 stores).
__device__ __forceinline__ unsigned short pp_f32_to_f16(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

__device__ __forceinline__ unsigned char pp_sat_u8(int v) {
    return (unsigned char)min(max(v, 0), 255);
}

// Q20 BT.601-limited decode — constants match gpu/color_cuda/video.rs.
__device__ __forceinline__ void pp_decode_yuv(
    int y, int u, int v, unsigned char* r, unsigned char* g, unsigned char* b)
{
    int yy = max(y - 16, 0) * 1220542;
    u -= 128;
    v -= 128;
    *b = pp_sat_u8((yy + 2116026 * u + (1 << 19)) >> 20);
    *g = pp_sat_u8((yy + (-409993) * u + (-852492) * v + (1 << 19)) >> 20);
    *r = pp_sat_u8((yy + 1673527 * v + (1 << 19)) >> 20);
}

// Per-format single-pixel fetch: decode the RGB bytes at integer (x, y).
// `pitch` is the byte stride of the primary plane row.

__device__ __forceinline__ void fetch_rgb8(
    const unsigned char* __restrict__ src, int x, int y, int w, int h, int pitch,
    unsigned char* r, unsigned char* g, unsigned char* b)
{
    const unsigned char* p = src + (long)y * pitch + x * 3;
    *r = p[0]; *g = p[1]; *b = p[2];
}

__device__ __forceinline__ void fetch_bgr8(
    const unsigned char* __restrict__ src, int x, int y, int w, int h, int pitch,
    unsigned char* r, unsigned char* g, unsigned char* b)
{
    const unsigned char* p = src + (long)y * pitch + x * 3;
    *r = p[2]; *g = p[1]; *b = p[0];
}

__device__ __forceinline__ void fetch_rgba8(
    const unsigned char* __restrict__ src, int x, int y, int w, int h, int pitch,
    unsigned char* r, unsigned char* g, unsigned char* b)
{
    const unsigned char* p = src + (long)y * pitch + x * 4;
    *r = p[0]; *g = p[1]; *b = p[2];
}

__device__ __forceinline__ void fetch_bgra8(
    const unsigned char* __restrict__ src, int x, int y, int w, int h, int pitch,
    unsigned char* r, unsigned char* g, unsigned char* b)
{
    const unsigned char* p = src + (long)y * pitch + x * 4;
    *r = p[2]; *g = p[1]; *b = p[0];
}

__device__ __forceinline__ void fetch_gray8(
    const unsigned char* __restrict__ src, int x, int y, int w, int h, int pitch,
    unsigned char* r, unsigned char* g, unsigned char* b)
{
    unsigned char v = src[(long)y * pitch + x];
    *r = v; *g = v; *b = v;
}

// NV12: full-res Y plane (w*h) followed by interleaved half-res UV.
__device__ __forceinline__ void fetch_nv12(
    const unsigned char* __restrict__ src, int x, int y, int w, int h, int pitch,
    unsigned char* r, unsigned char* g, unsigned char* b)
{
    int yv = src[(long)y * w + x];
    const unsigned char* uv = src + (long)w * h + (long)(y >> 1) * w + (x >> 1) * 2;
    pp_decode_yuv(yv, uv[0], uv[1], r, g, b);
}

// YUYV: packed 4:2:2, 2 bytes/px, group layout Y0 U Y1 V.
__device__ __forceinline__ void fetch_yuyv(
    const unsigned char* __restrict__ src, int x, int y, int w, int h, int pitch,
    unsigned char* r, unsigned char* g, unsigned char* b)
{
    const unsigned char* grp = src + (long)y * w * 2 + (x >> 1) * 4;
    int yv = grp[(x & 1) ? 2 : 0];
    pp_decode_yuv(yv, grp[1], grp[3], r, g, b);
}

// Bilinear: 4 fetched taps, clamp-to-edge, texel-center convention.
#define SAMPLE_BILINEAR(FETCH, OUTV)                                            \
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);                             \
    float ax = sx - (float)x0, ay = sy - (float)y0;                             \
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);               \
    x0 = max(x0, 0); y0 = max(y0, 0);                                           \
    unsigned char t00[3], t10[3], t01[3], t11[3];                               \
    FETCH(src, x0, y0, src_w, src_h, src_pitch, &t00[0], &t00[1], &t00[2]);     \
    FETCH(src, x1, y0, src_w, src_h, src_pitch, &t10[0], &t10[1], &t10[2]);     \
    FETCH(src, x0, y1, src_w, src_h, src_pitch, &t01[0], &t01[1], &t01[2]);     \
    FETCH(src, x1, y1, src_w, src_h, src_pitch, &t11[0], &t11[1], &t11[2]);     \
    float OUTV[3];                                                              \
    _Pragma("unroll")                                                           \
    for (int c = 0; c < 3; ++c) {                                               \
        float v00 = (float)t00[c], v10 = (float)t10[c];                         \
        float v01 = (float)t01[c], v11 = (float)t11[c];                         \
        float top = v00 + (v10 - v00) * ax;                                     \
        float bot = v01 + (v11 - v01) * ax;                                     \
        OUTV[c] = top + (bot - top) * ay;                                       \
    }

// Nearest: round-half-up to the closest texel, clamp-to-edge.
#define SAMPLE_NEAREST(FETCH, OUTV)                                             \
    int xi = min(max((int)floorf(sx + 0.5f), 0), src_w - 1);                    \
    int yi = min(max((int)floorf(sy + 0.5f), 0), src_h - 1);                    \
    unsigned char t[3];                                                          \
    FETCH(src, xi, yi, src_w, src_h, src_pitch, &t[0], &t[1], &t[2]);           \
    float OUTV[3] = { (float)t[0], (float)t[1], (float)t[2] };

#define CVT_F32(x) (x)
#define CVT_F16(x) pp_f32_to_f16(x)

// One entry per (format, sampling, output dtype). Normalization is fused:
// out = (v/255 - mean_c) * inv_std_c; the grey (114) pad goes through the
// same normalization so padded regions match a CPU pad-then-normalize.
#define PP_KERNEL(NAME, FETCH, SAMPLE, OUTT, CVT)                                \
extern "C" __global__ void NAME(                                                 \
    const unsigned char* __restrict__ src,                                       \
    OUTT* __restrict__ dst,                                                      \
    float scale_x, float scale_y, float pad_x, float pad_y,                      \
    int src_w, int src_h, int src_pitch,                                         \
    int dst_w, int dst_h,                                                        \
    float mean0, float mean1, float mean2,                                       \
    float is0, float is1, float is2                                              \
) {                                                                              \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                               \
    int pixels = dst_w * dst_h;                                                  \
    if (i >= pixels) return;                                                     \
    int ox = i % dst_w;                                                          \
    int oy = i / dst_w;                                                          \
    float mean[3] = { mean0, mean1, mean2 };                                     \
    float is[3] = { is0, is1, is2 };                                             \
                                                                                 \
    float sx = ((float)ox - pad_x) / scale_x;                                    \
    float sy = ((float)oy - pad_y) / scale_y;                                    \
                                                                                 \
    if (sx < 0.0f || sy < 0.0f || sx >= (float)src_w || sy >= (float)src_h) {    \
        _Pragma("unroll")                                                        \
        for (int c = 0; c < 3; ++c) {                                            \
            dst[c * pixels + i] = CVT(((114.0f / 255.0f) - mean[c]) * is[c]);    \
        }                                                                        \
        return;                                                                  \
    }                                                                            \
                                                                                 \
    SAMPLE(FETCH, v)                                                             \
    _Pragma("unroll")                                                            \
    for (int c = 0; c < 3; ++c) {                                                \
        dst[c * pixels + i] = CVT((v[c] / 255.0f - mean[c]) * is[c]);            \
    }                                                                            \
}

PP_KERNEL(resize_pad_to_chw_rgb8,             fetch_rgb8, SAMPLE_BILINEAR, float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_rgb8_nearest,     fetch_rgb8, SAMPLE_NEAREST,  float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_rgb8_f16,         fetch_rgb8, SAMPLE_BILINEAR, unsigned short, CVT_F16)
PP_KERNEL(resize_pad_to_chw_rgb8_nearest_f16, fetch_rgb8, SAMPLE_NEAREST,  unsigned short, CVT_F16)
PP_KERNEL(resize_pad_to_chw_bgr8,             fetch_bgr8, SAMPLE_BILINEAR, float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_bgr8_nearest,     fetch_bgr8, SAMPLE_NEAREST,  float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_bgr8_f16,         fetch_bgr8, SAMPLE_BILINEAR, unsigned short, CVT_F16)
PP_KERNEL(resize_pad_to_chw_bgr8_nearest_f16, fetch_bgr8, SAMPLE_NEAREST,  unsigned short, CVT_F16)
PP_KERNEL(resize_pad_to_chw_rgba8,             fetch_rgba8, SAMPLE_BILINEAR, float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_rgba8_nearest,     fetch_rgba8, SAMPLE_NEAREST,  float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_rgba8_f16,         fetch_rgba8, SAMPLE_BILINEAR, unsigned short, CVT_F16)
PP_KERNEL(resize_pad_to_chw_rgba8_nearest_f16, fetch_rgba8, SAMPLE_NEAREST,  unsigned short, CVT_F16)
PP_KERNEL(resize_pad_to_chw_bgra8,             fetch_bgra8, SAMPLE_BILINEAR, float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_bgra8_nearest,     fetch_bgra8, SAMPLE_NEAREST,  float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_bgra8_f16,         fetch_bgra8, SAMPLE_BILINEAR, unsigned short, CVT_F16)
PP_KERNEL(resize_pad_to_chw_bgra8_nearest_f16, fetch_bgra8, SAMPLE_NEAREST,  unsigned short, CVT_F16)
PP_KERNEL(resize_pad_to_chw_gray8,             fetch_gray8, SAMPLE_BILINEAR, float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_gray8_nearest,     fetch_gray8, SAMPLE_NEAREST,  float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_gray8_f16,         fetch_gray8, SAMPLE_BILINEAR, unsigned short, CVT_F16)
PP_KERNEL(resize_pad_to_chw_gray8_nearest_f16, fetch_gray8, SAMPLE_NEAREST,  unsigned short, CVT_F16)
PP_KERNEL(resize_pad_to_chw_nv12,             fetch_nv12, SAMPLE_BILINEAR, float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_nv12_nearest,     fetch_nv12, SAMPLE_NEAREST,  float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_nv12_f16,         fetch_nv12, SAMPLE_BILINEAR, unsigned short, CVT_F16)
PP_KERNEL(resize_pad_to_chw_nv12_nearest_f16, fetch_nv12, SAMPLE_NEAREST,  unsigned short, CVT_F16)
PP_KERNEL(resize_pad_to_chw_yuyv,             fetch_yuyv, SAMPLE_BILINEAR, float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_yuyv_nearest,     fetch_yuyv, SAMPLE_NEAREST,  float,  CVT_F32)
PP_KERNEL(resize_pad_to_chw_yuyv_f16,         fetch_yuyv, SAMPLE_BILINEAR, unsigned short, CVT_F16)
PP_KERNEL(resize_pad_to_chw_yuyv_nearest_f16, fetch_yuyv, SAMPLE_NEAREST,  unsigned short, CVT_F16)
"#;

/// GPU image-to-tensor preprocessor: resize (+ optional pad) + normalize.
///
/// Built once for a [`ResizeMode`] (compiles the CUDA kernel for the device
/// behind its stream), then applied to any number of frames of any resolution via
/// [`run`](Self::run). It owns only the JIT-compiled kernel and its stream — no
/// per-frame buffers — so a single instance handles every input size and any
/// model size (the target H/W is read from the destination tensor each call).
///
/// The output is RGB in `[0, 1]`, CHW (`[1, 3, H, W]`). For models that expect a
/// further mean/std normalization (e.g. ImageNet), apply it as a second step on
/// the resulting tensor.
pub struct Preprocessor {
    kernel: CudaKernel,
    stream: Arc<CudaStream>,
    mode: ResizeMode,
    format: SourceFormat,
    sampling: Sampling,
    f16_output: bool,
    mean: [f32; 3],
    inv_std: [f32; 3],
}

impl Preprocessor {
    /// Build a preprocessor that **letterboxes** (aspect-preserving resize + grey
    /// pad) on `stream`. Compiles the kernel once (nvrtc JIT; CUDA caches the PTX).
    pub fn letterbox(stream: Arc<CudaStream>) -> Result<Self, PreprocessError> {
        Self::with_mode(stream, ResizeMode::Letterbox)
    }

    /// Build a preprocessor that **stretches** (anisotropic resize, no pad) on
    /// `stream` — for models trained on a square stretch (e.g. RF-DETR).
    pub fn stretch(stream: Arc<CudaStream>) -> Result<Self, PreprocessError> {
        Self::with_mode(stream, ResizeMode::Stretch)
    }

    /// Build a preprocessor for an explicit [`ResizeMode`] on `stream`
    /// (RGB source, like [`letterbox`](Self::letterbox)/[`stretch`](Self::stretch)).
    pub fn with_mode(stream: Arc<CudaStream>, mode: ResizeMode) -> Result<Self, PreprocessError> {
        Self::with_format(stream, mode, SourceFormat::Rgb8)
    }

    /// Build a preprocessor whose kernel **fuses color decode** for `format`
    /// into the resize — e.g. `SourceFormat::Nv12` turns a raw capture frame
    /// into the model tensor in one launch. Only the entry for the requested
    /// format is compiled.
    pub fn with_format(
        stream: Arc<CudaStream>,
        mode: ResizeMode,
        format: SourceFormat,
    ) -> Result<Self, PreprocessError> {
        Self::with_options(
            stream,
            PreprocessorOptions {
                mode,
                format,
                ..Default::default()
            },
        )
    }

    /// Build a preprocessor from full [`PreprocessorOptions`] — source format,
    /// sampling, and output dtype each select a dedicated kernel entry; only
    /// that one is compiled.
    pub fn with_options(
        stream: Arc<CudaStream>,
        opts: PreprocessorOptions,
    ) -> Result<Self, PreprocessError> {
        let ctx = stream.context();
        let entry = opts.format.entry(opts.sampling, opts.f16_output);
        let kernel = CudaKernel::compile(ctx, KERNEL_SRC, &entry)?;
        Ok(Self {
            kernel,
            stream,
            mode: opts.mode,
            format: opts.format,
            sampling: opts.sampling,
            f16_output: opts.f16_output,
            mean: [0.0; 3],
            inv_std: [1.0; 3],
        })
    }

    /// Fuse a per-channel mean/std normalization into the kernel:
    /// `out = (v/255 − mean_c) / std_c` (grey padding passes through the same
    /// normalization). Chainable; defaults to mean 0 / std 1 (plain `[0,1]`).
    ///
    /// # Errors
    ///
    /// [`PreprocessError::ZeroStd`] if any std component is zero.
    pub fn normalize(mut self, mean: [f32; 3], std: [f32; 3]) -> Result<Self, PreprocessError> {
        if std.contains(&0.0) {
            return Err(PreprocessError::ZeroStd);
        }
        self.mean = mean;
        self.inv_std = [1.0 / std[0], 1.0 / std[1], 1.0 / std[2]];
        Ok(self)
    }

    /// The CUDA stream this preprocessor launches on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The resize mode this preprocessor applies.
    pub fn mode(&self) -> ResizeMode {
        self.mode
    }

    /// The source pixel format this preprocessor decodes.
    pub fn format(&self) -> SourceFormat {
        self.format
    }

    /// The resize interpolation this preprocessor applies.
    pub fn sampling(&self) -> Sampling {
        self.sampling
    }

    /// Resize `src` into `dst` (`[1, 3, H, W]` CHW `f32`, RGB in `[0, 1]`).
    ///
    /// The target H/W is read from `dst`'s shape, so one preprocessor handles any
    /// model size. Both `src` and `dst` must be device-resident (the image via
    /// `to_cuda`, the tensor via `zeros_cuda`).
    ///
    /// # Errors
    ///
    /// [`PreprocessError::NotDeviceImage`] / [`PreprocessError::NotDeviceTensor`]
    /// if either operand is host-resident, or [`PreprocessError::Cuda`] on a CUDA
    /// launch failure.
    pub fn run(&self, src: &Image<u8, 3>, dst: &mut Tensor<f32, 4>) -> Result<(), PreprocessError> {
        if self.f16_output {
            return Err(PreprocessError::OutputDtypeMismatch { built_f16: true });
        }
        if !matches!(self.format, SourceFormat::Rgb8 | SourceFormat::Bgr8) {
            return Err(PreprocessError::FormatNeedsRawBuffer(self.format));
        }
        let (src_w, src_h) = (src.width(), src.height());
        let src_slice = src.as_cudaslice().ok_or(PreprocessError::NotDeviceImage)?;
        self.launch(src_slice, src_w, src_h, dst)
    }

    /// Preprocess a raw device buffer in this preprocessor's [`SourceFormat`]
    /// into `dst` (`[1, 3, H, W]` CHW `f32`, RGB in `[0, 1]`).
    ///
    /// This is the fused camera path: an NV12/YUYV frame straight from a
    /// capture pipeline (already device-resident) becomes the model input in
    /// one kernel — no intermediate RGB image is ever materialized.
    ///
    /// # Errors
    ///
    /// [`PreprocessError::SourceBufferTooSmall`] if `src` is shorter than the
    /// format requires for `src_width × src_height`,
    /// [`PreprocessError::OddDimensions`] for subsampled formats with odd
    /// dimensions, or the same errors as [`run`](Self::run).
    pub fn run_raw(
        &self,
        src: &cudarc::driver::CudaSlice<u8>,
        src_width: usize,
        src_height: usize,
        dst: &mut Tensor<f32, 4>,
    ) -> Result<(), PreprocessError> {
        if self.f16_output {
            return Err(PreprocessError::OutputDtypeMismatch { built_f16: true });
        }
        self.run_raw_checked(src, src_width, src_height, dst)
    }

    /// [`run_raw`](Self::run_raw) for a preprocessor built with
    /// `f16_output: true` — writes half-precision CHW output (half the
    /// tensor bandwidth, the dtype most TensorRT engines consume).
    pub fn run_raw_f16(
        &self,
        src: &cudarc::driver::CudaSlice<u8>,
        src_width: usize,
        src_height: usize,
        dst: &mut Tensor<half::f16, 4>,
    ) -> Result<(), PreprocessError> {
        if !self.f16_output {
            return Err(PreprocessError::OutputDtypeMismatch { built_f16: false });
        }
        self.run_raw_checked(src, src_width, src_height, dst)
    }

    fn run_raw_checked<O>(
        &self,
        src: &cudarc::driver::CudaSlice<u8>,
        src_width: usize,
        src_height: usize,
        dst: &mut Tensor<O, 4>,
    ) -> Result<(), PreprocessError>
    where
        O: DeviceRepr + ValidAsZeroBits + 'static,
    {
        let need = self.format.buffer_len(src_width, src_height);
        if src.len() < need {
            return Err(PreprocessError::SourceBufferTooSmall {
                format: self.format,
                width: src_width,
                height: src_height,
                got: src.len(),
                need,
            });
        }
        let even_ok = match self.format {
            SourceFormat::Nv12 => src_width.is_multiple_of(2) && src_height.is_multiple_of(2),
            SourceFormat::Yuyv => src_width.is_multiple_of(2),
            _ => true,
        };
        if !even_ok {
            return Err(PreprocessError::OddDimensions {
                format: self.format,
                width: src_width,
                height: src_height,
            });
        }
        self.launch(src, src_width, src_height, dst)
    }

    fn launch<O>(
        &self,
        src_slice: &cudarc::driver::CudaSlice<u8>,
        src_w: usize,
        src_h: usize,
        dst: &mut Tensor<O, 4>,
    ) -> Result<(), PreprocessError>
    where
        O: DeviceRepr + ValidAsZeroBits + 'static,
    {
        let (dst_h, dst_w) = (dst.shape[2], dst.shape[3]);

        let (scale_x, scale_y, pad_x, pad_y) = match self.mode {
            ResizeMode::Letterbox => {
                let s = f32::min(dst_w as f32 / src_w as f32, dst_h as f32 / src_h as f32);
                (
                    s,
                    s,
                    (dst_w as f32 - src_w as f32 * s) * 0.5,
                    (dst_h as f32 - src_h as f32 * s) * 0.5,
                )
            }
            ResizeMode::Stretch => (
                dst_w as f32 / src_w as f32,
                dst_h as f32 / src_h as f32,
                0.0,
                0.0,
            ),
        };

        let dst_slice = dst
            .as_cudaslice_mut()
            .ok_or(PreprocessError::NotDeviceTensor)?;

        let (sw, sh, sp) = (src_w as i32, src_h as i32, self.format.pitch(src_w) as i32);
        let (dw, dh) = (dst_w as i32, dst_h as i32);
        let total = (dst_w * dst_h) as u32;

        self.kernel
            .launch_builder(&self.stream)
            .arg(src_slice)
            .arg(dst_slice)
            .arg(&scale_x)
            .arg(&scale_y)
            .arg(&pad_x)
            .arg(&pad_y)
            .arg(&sw)
            .arg(&sh)
            .arg(&sp)
            .arg(&dw)
            .arg(&dh)
            .arg(&self.mean[0])
            .arg(&self.mean[1])
            .arg(&self.mean[2])
            .arg(&self.inv_std[0])
            .arg(&self.inv_std[1])
            .arg(&self.inv_std[2])
            .launch_1d(total)?;
        Ok(())
    }
}

#[cfg(all(test, feature = "cudarc", feature = "gpu-cuda"))]
mod tests {
    use super::*;
    use crate::gpu::color_cuda::{gray, swizzle, video};
    use cudarc::driver::{CudaContext, CudaSlice};
    use kornia_tensor::zeros_cuda;

    const W: usize = 100;
    const H: usize = 62;
    const OUT: usize = 64;

    fn pattern(len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let mut state = 0x9e37_79b9u32;
        while v.len() < len {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push((state >> 24) as u8);
        }
        v
    }

    /// Fused decode+resize must equal decode-then-resize: identical u8 taps
    /// through identical interpolation code.
    fn assert_fused_matches_chained(
        format: SourceFormat,
        decode: impl Fn(&Arc<CudaStream>, &CudaSlice<u8>, &mut CudaSlice<u8>),
    ) {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let src_bytes = pattern(format.buffer_len(W, H));
        let d_src = stream.clone_htod(&src_bytes).unwrap();

        for mode in [ResizeMode::Letterbox, ResizeMode::Stretch] {
            // Fused: raw buffer -> CHW tensor in one kernel.
            let pre = Preprocessor::with_format(stream.clone(), mode, format).unwrap();
            let mut dst_fused = zeros_cuda::<f32, 4>([1, 3, OUT, OUT], &stream).unwrap();
            pre.run_raw(&d_src, W, H, &mut dst_fused).unwrap();

            // Chained: decode to a full RGB image, then the RGB preprocessor.
            let mut d_rgb = stream.alloc_zeros::<u8>(W * H * 3).unwrap();
            decode(&stream, &d_src, &mut d_rgb);
            let rgb_img: Image<u8, 3> = Image(kornia_tensor::Tensor::from_cudaslice(
                d_rgb,
                [H, W, 3],
                stream.clone(),
            ));
            let pre_rgb = Preprocessor::with_mode(stream.clone(), mode).unwrap();
            let mut dst_ref = zeros_cuda::<f32, 4>([1, 3, OUT, OUT], &stream).unwrap();
            pre_rgb.run(&rgb_img, &mut dst_ref).unwrap();

            let fused = dst_fused.to_host(&stream).unwrap();
            let chained = dst_ref.to_host(&stream).unwrap();
            let max_diff = fused
                .as_slice()
                .iter()
                .zip(chained.as_slice())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_diff <= 1e-6,
                "{format:?}/{mode:?}: fused vs chained max diff {max_diff}"
            );
        }
    }

    #[test]
    fn fused_bgr_matches_chained() {
        // bgr swap is symmetric: applying it to BGR bytes yields RGB.
        assert_fused_matches_chained(SourceFormat::Bgr8, |s, a, b| {
            swizzle::launch_bgr_from_rgb_u8(s, a, b, W * H).unwrap()
        });
    }

    #[test]
    fn fused_rgba_matches_chained() {
        assert_fused_matches_chained(SourceFormat::Rgba8, |s, a, b| {
            swizzle::launch_rgb_from_rgba_u8(s, a, b, W * H, false, None).unwrap()
        });
    }

    #[test]
    fn fused_bgra_matches_chained() {
        assert_fused_matches_chained(SourceFormat::Bgra8, |s, a, b| {
            swizzle::launch_rgb_from_rgba_u8(s, a, b, W * H, true, None).unwrap()
        });
    }

    #[test]
    fn fused_gray_matches_chained() {
        assert_fused_matches_chained(SourceFormat::Gray8, |s, a, b| {
            gray::launch_rgb_from_gray_u8(s, a, b, W * H).unwrap()
        });
    }

    #[test]
    fn fused_nv12_matches_chained() {
        assert_fused_matches_chained(SourceFormat::Nv12, |s, a, b| {
            video::launch_rgb_from_planar420_u8(s, a, b, W, H, video::Planar420::Nv12).unwrap()
        });
    }

    #[test]
    fn fused_yuyv_matches_chained() {
        assert_fused_matches_chained(SourceFormat::Yuyv, |s, a, b| {
            video::launch_rgb_from_packed422_u8(s, a, b, W, H, video::Packed422::Yuyv).unwrap()
        });
    }

    #[test]
    fn nearest_fused_matches_chained() {
        // Same fused-vs-chained equivalence, nearest sampling.
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let src_bytes = pattern(SourceFormat::Nv12.buffer_len(W, H));
        let d_src = stream.clone_htod(&src_bytes).unwrap();

        let opts = |format| PreprocessorOptions {
            mode: ResizeMode::Letterbox,
            format,
            sampling: Sampling::Nearest,
            f16_output: false,
        };
        let pre = Preprocessor::with_options(stream.clone(), opts(SourceFormat::Nv12)).unwrap();
        let mut dst_fused = zeros_cuda::<f32, 4>([1, 3, OUT, OUT], &stream).unwrap();
        pre.run_raw(&d_src, W, H, &mut dst_fused).unwrap();

        let mut d_rgb = stream.alloc_zeros::<u8>(W * H * 3).unwrap();
        crate::gpu::color_cuda::video::launch_rgb_from_planar420_u8(
            &stream,
            &d_src,
            &mut d_rgb,
            W,
            H,
            crate::gpu::color_cuda::video::Planar420::Nv12,
        )
        .unwrap();
        let rgb_img: Image<u8, 3> = Image(kornia_tensor::Tensor::from_cudaslice(
            d_rgb,
            [H, W, 3],
            stream.clone(),
        ));
        let pre_rgb = Preprocessor::with_options(stream.clone(), opts(SourceFormat::Rgb8)).unwrap();
        let mut dst_ref = zeros_cuda::<f32, 4>([1, 3, OUT, OUT], &stream).unwrap();
        pre_rgb.run(&rgb_img, &mut dst_ref).unwrap();

        let fused = dst_fused.to_host(&stream).unwrap();
        let chained = dst_ref.to_host(&stream).unwrap();
        let max_diff = fused
            .as_slice()
            .iter()
            .zip(chained.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff <= 1e-6,
            "nearest fused vs chained max diff {max_diff}"
        );
    }

    #[test]
    fn f16_output_close_to_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let src_bytes = pattern(SourceFormat::Rgb8.buffer_len(W, H));
        let d_src = stream.clone_htod(&src_bytes).unwrap();

        let pre32 =
            Preprocessor::with_format(stream.clone(), ResizeMode::Stretch, SourceFormat::Rgb8)
                .unwrap();
        let mut d32 = zeros_cuda::<f32, 4>([1, 3, OUT, OUT], &stream).unwrap();
        pre32.run_raw(&d_src, W, H, &mut d32).unwrap();

        let pre16 = Preprocessor::with_options(
            stream.clone(),
            PreprocessorOptions {
                mode: ResizeMode::Stretch,
                format: SourceFormat::Rgb8,
                sampling: Sampling::Bilinear,
                f16_output: true,
            },
        )
        .unwrap();
        let mut d16 = zeros_cuda::<half::f16, 4>([1, 3, OUT, OUT], &stream).unwrap();
        pre16.run_raw_f16(&d_src, W, H, &mut d16).unwrap();

        // Dtype guards.
        assert!(matches!(
            pre16.run_raw(&d_src, W, H, &mut d32),
            Err(PreprocessError::OutputDtypeMismatch { built_f16: true })
        ));
        assert!(matches!(
            pre32.run_raw_f16(&d_src, W, H, &mut d16),
            Err(PreprocessError::OutputDtypeMismatch { built_f16: false })
        ));

        let h32 = d32.to_host(&stream).unwrap();
        let h16 = d16.to_host(&stream).unwrap();
        // Values live in [0,1]: half quantization step ≤ 2^-11 there.
        let max_diff = h32
            .as_slice()
            .iter()
            .zip(h16.as_slice())
            .map(|(a, b)| (a - b.to_f32()).abs())
            .fold(0f32, f32::max);
        assert!(max_diff <= 5e-4, "f16 vs f32 max diff {max_diff}");
    }

    #[test]
    fn fused_normalize_matches_post_normalize() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let src_bytes = pattern(SourceFormat::Rgb8.buffer_len(W, H));
        let d_src = stream.clone_htod(&src_bytes).unwrap();
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];

        let plain =
            Preprocessor::with_format(stream.clone(), ResizeMode::Stretch, SourceFormat::Rgb8)
                .unwrap();
        let mut d_plain = zeros_cuda::<f32, 4>([1, 3, OUT, OUT], &stream).unwrap();
        plain.run_raw(&d_src, W, H, &mut d_plain).unwrap();

        let fused =
            Preprocessor::with_format(stream.clone(), ResizeMode::Stretch, SourceFormat::Rgb8)
                .unwrap()
                .normalize(mean, std)
                .unwrap();
        let mut d_norm = zeros_cuda::<f32, 4>([1, 3, OUT, OUT], &stream).unwrap();
        fused.run_raw(&d_src, W, H, &mut d_norm).unwrap();

        let h_plain = d_plain.to_host(&stream).unwrap();
        let h_norm = d_norm.to_host(&stream).unwrap();
        let pixels = OUT * OUT;
        let max_diff = h_norm
            .as_slice()
            .iter()
            .enumerate()
            .map(|(i, &got)| {
                let c = i / pixels;
                let want = (h_plain.as_slice()[i] - mean[c]) / std[c];
                (got - want).abs()
            })
            .fold(0f32, f32::max);
        assert!(max_diff <= 1e-5, "fused normalize max diff {max_diff}");

        assert!(matches!(
            Preprocessor::with_format(stream, ResizeMode::Stretch, SourceFormat::Rgb8)
                .unwrap()
                .normalize([0.0; 3], [0.0, 1.0, 1.0]),
            Err(PreprocessError::ZeroStd)
        ));
    }

    #[test]
    fn raw_buffer_validation() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let pre =
            Preprocessor::with_format(stream.clone(), ResizeMode::Stretch, SourceFormat::Nv12)
                .unwrap();
        let mut dst = zeros_cuda::<f32, 4>([1, 3, OUT, OUT], &stream).unwrap();

        // Too-small buffer.
        let d_small = stream.alloc_zeros::<u8>(16).unwrap();
        assert!(matches!(
            pre.run_raw(&d_small, W, H, &mut dst),
            Err(PreprocessError::SourceBufferTooSmall { .. })
        ));

        // Odd dimensions for a subsampled format.
        let d_ok = stream.alloc_zeros::<u8>(101 * 63 * 2).unwrap();
        assert!(matches!(
            pre.run_raw(&d_ok, 101, 63, &mut dst),
            Err(PreprocessError::OddDimensions { .. })
        ));

        // run() with a non-3ch format must point at run_raw.
        let img = Image::<u8, 3>::from_size_val([4, 4].into(), 0)
            .unwrap()
            .to_cuda_image(&stream)
            .unwrap();
        assert!(matches!(
            pre.run(&img, &mut dst),
            Err(PreprocessError::FormatNeedsRawBuffer(SourceFormat::Nv12))
        ));
    }
}
