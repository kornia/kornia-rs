//! Image → model-input preprocessing: resize (+ optional pad) and normalize an
//! [`Image`] straight into a CHW `f32` [`Tensor`], on the **CPU** or, with the
//! `cudarc` feature, in **one CUDA kernel** on a device-resident image.
//!
//! This is the step every CNN / transformer vision model needs before inference:
//! take a camera frame of arbitrary size and produce the fixed `[1, 3, H, W]`
//! CHW `f32` tensor the network expects. The resize fit, normalization, and pad
//! value are configurable via [`PreprocessorBuilder`]:
//! - **resize** — [`ResizeMode::Letterbox`] (aspect-preserving + pad) or
//!   [`ResizeMode::Stretch`] (anisotropic, no pad).
//! - **normalize** — [`Normalize::UnitScale`] (RGB `[0,1]`) or [`Normalize::MeanStd`]
//!   (fold an ImageNet-style mean/std into the same pass).
//! - **sampling** — `Nearest`, `Bilinear` (default), or `Lanczos` (3-lobe windowed
//!   sinc), via the crate's shared [`InterpolationMode`] — the same filters
//!   `resize`/`warp` use.
//! - **channels** — [`run`](Preprocessor::run) is generic over the source channel
//!   count: `Image<u8, 3>` (RGB) or `Image<u8, 4>` (RGBA — the alpha byte is
//!   skipped), so a repacked RGBA camera surface reaches a CHW model tensor with no
//!   separate RGBA→RGB pass.
//!
//! [`build`](PreprocessorBuilder::build) makes a **CPU** preprocessor (host image
//! → host tensor, always available). [`build_cuda`](PreprocessorBuilder::build_cuda)
//! (feature `cudarc`) makes a **GPU** one that runs a fused kernel with no host
//! round-trip — the source already lives on the device (`image.to_cuda(&stream)`).
//!
//! # Example (CPU)
//!
//! ```no_run
//! use kornia_image::Image;
//! use kornia_tensor::Tensor;
//! use kornia_imgproc::preprocess::{Preprocessor, Normalize, ResizeMode};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let pre = Preprocessor::builder()
//!     .mode(ResizeMode::Stretch)
//!     .normalize(Normalize::imagenet())
//!     .build()?;                                    // CPU preprocessor
//!
//! let frame = Image::<u8, 3>::from_size_val([1280, 720].into(), 0)?;
//! let mut input = Tensor::<f32, 4>::from_shape_vec([1, 3, 640, 640], vec![0.0; 3 * 640 * 640])?;
//! pre.run(&frame, &mut input)?;                     // input is [1,3,640,640], ImageNet-normalized
//! # Ok(())
//! # }
//! ```
//!
//! For the GPU path build with [`build_cuda`](PreprocessorBuilder::build_cuda) and
//! pass device-resident operands (`image.to_cuda`, `zeros_cuda`).

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaStream;
use kornia_image::{Image, ImageError, ImageSize, InterpolationMode};
use kornia_tensor::Tensor;

use crate::resize::{
    resize_fast_u8_aa, resize_normalize_to_tensor_u8_to_f32_bilinear,
    resize_normalize_to_tensor_u8_to_f32_nearest, resize_normalize_to_tensor_u8_to_f32_separable,
    NormalizeParams,
};
#[cfg(feature = "cuda")]
use kornia_tensor::{CudaError, CudaKernel};

/// How the source image is fit into the model's output rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    /// Aspect-preserving scale + pad (grey 114 by default) — the YOLO / XFeat
    /// convention. The whole image is kept; the unused border is filled.
    Letterbox,
    /// Anisotropic stretch to the full target, no padding — the RT-DETR / RF-DETR
    /// convention. The aspect ratio is not preserved.
    Stretch,
}

/// How pixel values are normalized into the output `f32` tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Normalize {
    /// Scale `[0, 255]` → `[0, 1]` (divide by 255). The default.
    UnitScale,
    /// Per-channel `(v/255 - mean) / std` — e.g. ImageNet (see [`Normalize::imagenet`]).
    /// Means/stds are in the `[0, 1]` domain, matching the torchvision convention.
    MeanStd {
        /// Per-channel mean subtracted after scaling to `[0, 1]`.
        mean: [f32; 3],
        /// Per-channel standard deviation divided out.
        std: [f32; 3],
    },
}

/// The standard ImageNet per-channel mean (RGB order), matching torchvision —
/// the single source for [`Normalize::imagenet`] and language bindings.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// The standard ImageNet per-channel std (RGB order), matching torchvision.
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

impl Normalize {
    /// The standard ImageNet mean/std (RGB), matching torchvision.
    pub fn imagenet() -> Self {
        Normalize::MeanStd {
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
        }
    }

    /// Resolve to `(mean, inv_std)` in the `[0, 1]` domain. `UnitScale` is the
    /// identity affine (mean 0, inv_std 1) so one code path handles both. A
    /// non-finite or non-positive std would silently produce `inf`/`NaN` outputs,
    /// so it is rejected here (surfaced by `build`/`build_cuda`).
    fn mean_inv_std(self) -> Result<([f32; 3], [f32; 3]), PreprocessError> {
        match self {
            Normalize::UnitScale => Ok(([0.0; 3], [1.0; 3])),
            Normalize::MeanStd { mean, std } => {
                if std.iter().any(|s| !s.is_finite() || *s <= 0.0)
                    || mean.iter().any(|m| !m.is_finite())
                {
                    return Err(PreprocessError::InvalidNormalize { mean, std });
                }
                Ok((mean, [1.0 / std[0], 1.0 / std[1], 1.0 / std[2]]))
            }
        }
    }
}

/// Pixel format of the source buffer. For non-interleaved camera formats
/// (NV12/YUYV) and Gray, color decode is **fused** into the resize kernel —
/// a raw capture frame becomes the CHW tensor in one launch, no intermediate
/// RGB image in device memory. Selected per-launch as a warp-uniform kernel
/// argument, preserving the single-JIT-compile design.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SourceFormat {
    /// Interleaved RGB, 3 bytes/px (the default; also what `run::<3>` uses).
    #[default]
    Rgb8,
    /// Interleaved BGR, 3 bytes/px (OpenCV convention; swapped in-kernel).
    Bgr8,
    /// Interleaved RGBA, 4 bytes/px (alpha skipped; what `run::<4>` uses).
    Rgba8,
    /// Interleaved BGRA, 4 bytes/px (alpha skipped, swapped in-kernel).
    Bgra8,
    /// Single-channel grayscale, 1 byte/px (broadcast to RGB).
    Gray8,
    /// Planar 4:2:0: full-res Y plane then interleaved half-res UV
    /// (`w*h*3/2` bytes, BT.601 limited — byte-identical to
    /// `gpu::color_cuda::video`). Even dimensions required.
    Nv12,
    /// Packed 4:2:2 `Y0 U Y1 V`, 2 bytes/px (BT.601 limited). Even width.
    Yuyv,
}

#[cfg(feature = "cuda")]
impl SourceFormat {
    /// Kernel `fmt` launch-arg code (see the fetch_px table in KERNEL_SRC).
    fn fmt_code(self) -> i32 {
        match self {
            SourceFormat::Rgb8 | SourceFormat::Rgba8 => 0,
            SourceFormat::Bgr8 | SourceFormat::Bgra8 => 1,
            SourceFormat::Gray8 => 2,
            SourceFormat::Nv12 => 3,
            SourceFormat::Yuyv => 4,
        }
    }

    /// Interleaved bytes/px passed as `src_bpp` (unused by planar formats).
    fn bpp(self) -> usize {
        match self {
            SourceFormat::Rgb8 | SourceFormat::Bgr8 => 3,
            SourceFormat::Rgba8 | SourceFormat::Bgra8 => 4,
            SourceFormat::Gray8 | SourceFormat::Nv12 => 1,
            SourceFormat::Yuyv => 2,
        }
    }

    /// Byte stride of the primary plane row.
    fn pitch(self, w: usize) -> usize {
        w * self.bpp()
    }

    /// Required buffer length in bytes for a `w × h` frame — what
    /// [`Preprocessor::run_raw`] validates against (longer buffers are fine).
    pub fn buffer_len(self, w: usize, h: usize) -> usize {
        let chroma = if matches!(self, SourceFormat::Nv12) {
            w * h / 2
        } else {
            0
        };
        self.pitch(w) * h + chroma
    }

    /// Even-dimension requirement for subsampled formats.
    fn dims_ok(self, w: usize, h: usize) -> bool {
        match self {
            SourceFormat::Nv12 => w.is_multiple_of(2) && h.is_multiple_of(2),
            SourceFormat::Yuyv => w.is_multiple_of(2),
            _ => true,
        }
    }

    /// Source-geometry launch args for a tightly-packed `w × h` frame.
    fn geom(self, w: usize, h: usize) -> SrcGeom {
        SrcGeom {
            w,
            h,
            pitch: self.pitch(w),
            bpp: self.bpp(),
            fmt: self.fmt_code(),
        }
    }
}

impl SourceFormat {
    /// True for the interleaved RGB-order formats — the ones a typed image or
    /// pitched surface can carry. Camera formats (Gray/NV12/YUYV) need the
    /// raw-buffer entry points.
    fn interleaved(self) -> bool {
        matches!(
            self,
            SourceFormat::Rgb8 | SourceFormat::Bgr8 | SourceFormat::Rgba8 | SourceFormat::Bgra8
        )
    }

    /// Parse a case-insensitive format name (`"rgb"`/`"rgb8"`, `"bgr"`,
    /// `"rgba"`, `"bgra"`, `"gray"`, `"nv12"`, `"yuyv"`) — the single naming
    /// authority for language bindings.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "rgb" | "rgb8" => Self::Rgb8,
            "bgr" | "bgr8" => Self::Bgr8,
            "rgba" | "rgba8" => Self::Rgba8,
            "bgra" | "bgra8" => Self::Bgra8,
            "gray" | "gray8" => Self::Gray8,
            "nv12" => Self::Nv12,
            "yuyv" => Self::Yuyv,
            _ => return None,
        })
    }
}

/// Source-buffer geometry as the kernel sees it: dimensions, primary-plane
/// byte pitch, interleaved bytes/px, and the `fmt` decode selector.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct SrcGeom {
    w: usize,
    h: usize,
    pitch: usize,
    bpp: usize,
    fmt: i32,
}

/// Errors from preprocessing.
#[derive(Debug, thiserror::Error)]
pub enum PreprocessError {
    /// A CUDA error from kernel compilation or launch.
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
    /// A CUDA preprocessor was given a host-resident source image; call
    /// `image.to_cuda(&stream)` first (or build a CPU preprocessor).
    #[error("CUDA preprocessor requires a device-resident source image")]
    NotDeviceImage,
    /// A CUDA preprocessor was given a host-resident destination tensor; allocate
    /// it with `zeros_cuda` (or build a CPU preprocessor).
    #[error("CUDA preprocessor requires a device-resident destination tensor")]
    NotDeviceTensor,
    /// A CPU preprocessor was given a device-resident operand; use `build_cuda`.
    #[error("CPU preprocessor requires host-resident operands (use build_cuda for device data)")]
    NotHostData,
    /// The source channel count is not 3 (RGB) or 4 (RGBA).
    #[error("unsupported source channel count {0} (expected 3 or 4)")]
    UnsupportedChannels(usize),
    /// The destination tensor is not `[1, 3, H, W]` — the preprocessor writes
    /// three channel planes, so any other leading dims would index out of bounds.
    #[error("destination tensor must be [1, 3, H, W], got {0:?}")]
    BadOutputShape([usize; 4]),
    /// `Normalize::MeanStd` with a non-finite mean or a non-finite / non-positive
    /// std, which would silently produce `inf`/`NaN` outputs.
    #[error("invalid normalize: mean {mean:?} must be finite, std {std:?} must be finite and > 0")]
    InvalidNormalize {
        /// The rejected per-channel mean.
        mean: [f32; 3],
        /// The rejected per-channel std.
        std: [f32; 3],
    },
    /// An error from the underlying resize kernels or scratch-image allocation.
    #[error(transparent)]
    Image(#[from] ImageError),
    /// The sampling mode is not supported (only Nearest / Bilinear / Lanczos).
    #[error("unsupported sampling mode {0:?} (expected Nearest, Bilinear, or Lanczos)")]
    UnsupportedSampling(InterpolationMode),
    /// A [`PitchedSurface`] whose pitch/len don't cover `width`×`height`, or
    /// with a channel count other than 3/4.
    #[cfg(feature = "cuda")]
    #[error("invalid pitched surface (need pitch >= width*channels and len >= pitch*height)")]
    InvalidSurface,
    /// The typed `run`/`run_f16` entry requires an interleaved format matching
    /// the image's channel count; camera formats use `run_raw`.
    #[error("source format {0:?} needs run_raw (raw device buffer), not the typed run()")]
    FormatNeedsRawBuffer(SourceFormat),
    /// The raw source buffer is smaller than the format requires, or the
    /// dimensions violate the format's subsampling constraints.
    #[cfg(feature = "cuda")]
    #[error(
        "invalid raw source for {format:?} at {width}x{height} (got {got} bytes, need {need})"
    )]
    InvalidRawSource {
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
    /// The destination batch dim does not match the number of frames.
    #[cfg(feature = "cuda")]
    #[error("destination batch dim {dst_n} != frame count {frames}")]
    BatchMismatch {
        /// Destination tensor N.
        dst_n: usize,
        /// Provided frame count.
        frames: usize,
    },
    /// Source or output dimensions exceed the kernel's 32-bit indexing (the CUDA
    /// path indexes pixels as `int`). Unreachable on real hardware, guarded anyway.
    #[cfg(feature = "cuda")]
    #[error("dimensions exceed the 32-bit CUDA kernel index limit")]
    DimensionsTooLarge,
}

// The source→dst mapping for one call: a per-axis scale and pad offset, so
// `src = (dst - pad) / scale`. Computed once and shared by the CPU and CUDA paths
// so both produce identical geometry.
#[derive(Clone, Copy)]
struct Affine {
    scale_x: f32,
    scale_y: f32,
    // The fractional pad offsets are consumed by the CUDA kernel's per-sample
    // affine; the CPU path derives an integer content box from the scales.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pad_x: f32,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pad_y: f32,
}

impl Affine {
    fn new(mode: ResizeMode, sw: usize, sh: usize, dw: usize, dh: usize) -> Self {
        let (scale_x, scale_y, pad_x, pad_y) = match mode {
            ResizeMode::Letterbox => {
                let s = f32::min(dw as f32 / sw as f32, dh as f32 / sh as f32);
                (
                    s,
                    s,
                    (dw as f32 - sw as f32 * s) * 0.5,
                    (dh as f32 - sh as f32 * s) * 0.5,
                )
            }
            ResizeMode::Stretch => (dw as f32 / sw as f32, dh as f32 / sh as f32, 0.0, 0.0),
        };
        Self {
            scale_x,
            scale_y,
            pad_x,
            pad_y,
        }
    }
}

/// A device-resident **pitched** interleaved-u8 surface — the shape camera /
/// NVMM buffers arrive in (rows padded to a hardware pitch), which a tight
/// kornia [`Image`] cannot represent. Lets the fused GPU kernel consume the
/// producer's pixels directly, with no repack pass.
///
/// `data` must hold at least `row_pitch * height` bytes; `row_pitch >=
/// width * channels`. `channels` is 3 (RGB) or 4 (RGBA — alpha skipped).
#[cfg(feature = "cuda")]
pub struct PitchedSurface<'a> {
    /// Device buffer holding the pitched rows.
    pub data: &'a cudarc::driver::CudaSlice<u8>,
    /// Surface width in pixels.
    pub width: usize,
    /// Surface height in pixels.
    pub height: usize,
    /// Bytes per row (>= `width * channels`).
    pub row_pitch: usize,
    /// Interleaved bytes per pixel: 3 (RGB) or 4 (RGBA, alpha skipped).
    pub channels: usize,
}

#[cfg(feature = "cuda")]
struct CudaBackend {
    kernel: CudaKernel,
    kernel_f16: CudaKernel,
    stream: Arc<CudaStream>,
}

/// Output element types the CUDA path can write; each selects its
/// pre-compiled kernel variant (`f32` or round-to-nearest-even `f16`).
#[cfg(feature = "cuda")]
trait OutElem: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + 'static {
    fn kernel(cuda: &CudaBackend) -> &CudaKernel;
}

#[cfg(feature = "cuda")]
impl OutElem for f32 {
    fn kernel(cuda: &CudaBackend) -> &CudaKernel {
        &cuda.kernel
    }
}

#[cfg(feature = "cuda")]
impl OutElem for half::f16 {
    fn kernel(cuda: &CudaBackend) -> &CudaKernel {
        &cuda.kernel_f16
    }
}

// 1-D grid over the `dst_w * dst_h` output pixels. Reads interleaved `src_bpp`-byte
// pixels (3 = RGB, 4 = RGBA with the alpha byte skipped) with a hand-rolled
// bilinear sample (CUDA textures only support 1/2/4-channel elements, not 3),
// applies `(v/255 - mean) * inv_std` per channel, and writes
// channel-planar (CHW) `f32`. One parameterized kernel covers every mode / format /
// normalization — config is passed as launch args (single JIT compile). `src_pitch`
// is a parameter for generality, but kornia `Image`s are tight, so the launcher
// always passes `src_w * C`.
#[cfg(feature = "cuda")]
const KERNEL_SRC: &str = r#"
// One thin extern-C entry per (sampling mode × output dtype): `build_cuda`
// compiles only the selected sampling variant (per-variant register footprint
// — a single branching kernel measurably slowed the common bilinear case),
// for both f32 and f16 outputs. All sampling/normalize logic lives in shared
// __device__ helpers.

__device__ __forceinline__ bool plan_pixel(
    int i, int dst_w, int dst_h,
    float scale_x, float scale_y, float pad_x, float pad_y,
    int src_w, int src_h,
    float* sx, float* sy
) {
    int ox = i % dst_w;
    int oy = i / dst_w;
    *sx = ((float)ox - pad_x) / scale_x;
    *sy = ((float)oy - pad_y) / scale_y;
    return !(*sx < 0.0f || *sy < 0.0f || *sx >= (float)src_w || *sy >= (float)src_h);
}

// f32 -> IEEE binary16 bits, round-to-nearest-even. Manual conversion so the
// kernel has no cuda_fp16.h dependency (NVRTC-safe everywhere).
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned int x = __float_as_uint(f);
    unsigned int sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xFFu) - 127 + 15;
    unsigned int man = x & 0x7FFFFFu;
    if (exp >= 31) {
        // Inf stays Inf; NaN keeps a nonzero (quiet) mantissa instead of
        // collapsing to Inf.
        unsigned int nan_bit = (man != 0u) ? 0x0200u : 0u;
        return (unsigned short)(sign | 0x7C00u | nan_bit);
    }
    if (exp <= 0) {
        if (exp < -10) return (unsigned short)sign;
        man |= 0x800000u;
        unsigned int shift = (unsigned int)(14 - exp);
        unsigned short h = (unsigned short)(sign | (man >> shift));
        unsigned int rem = man & ((1u << shift) - 1u);
        unsigned int mid = 1u << (shift - 1u);
        if (rem > mid || (rem == mid && (h & 1u))) h++;
        return h;
    }
    unsigned short h = (unsigned short)(sign | ((unsigned int)exp << 10) | (man >> 13));
    unsigned int rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return h;
}

// 1-D Lanczos-3 weight — same form as the CPU side, so the two backends stay
// numerically comparable.
__device__ __forceinline__ float lanczos_w(float d) {
    float ad = fabsf(d);
    if (ad < 1e-6f) return 1.0f;
    if (ad >= 3.0f) return 0.0f;
    float pd = 3.14159265358979f * d;
    return 3.0f * sinf(pd) * sinf(pd / 3.0f) / (pd * pd);
}

// ── per-format pixel fetch + samplers ──
//
// `fmt` selects how one (x, y) texel decodes to RGB. It is a warp-uniform
// launch arg (same for every thread), so the branches predict perfectly and
// the single-JIT-compile design is preserved. Q20 BT.601-limited constants
// match gpu/color_cuda/video.rs bit-for-bit.
//   0 = interleaved RGB-order (bpp = src_bpp: 3 or 4, alpha skipped)
//   1 = interleaved BGR-order (bpp = src_bpp: 3 or 4)
//   2 = gray, 1 byte/px (broadcast)
//   3 = NV12: full-res Y plane then interleaved half-res UV (pitch = width)
//   4 = YUYV: packed 4:2:2 `Y0 U Y1 V`, 2 bytes/px

__device__ __forceinline__ void yuv_to_rgbf(int yv, int u, int v, float px[3]) {
    int yy = max(yv - 16, 0) * 1220542;
    u -= 128;
    v -= 128;
    px[2] = (float)min(max((yy + 2116026 * u + (1 << 19)) >> 20, 0), 255);
    px[1] = (float)min(max((yy + (-409993) * u + (-852492) * v + (1 << 19)) >> 20, 0), 255);
    px[0] = (float)min(max((yy + 1673527 * v + (1 << 19)) >> 20, 0), 255);
}

__device__ __forceinline__ void fetch_px(
    const unsigned char* __restrict__ src, int x, int y,
    int src_w, int src_h, int src_pitch, int src_bpp, int fmt, float px[3]
) {
    if (fmt <= 1) {
        const unsigned char* p = src + (long)y * src_pitch + x * src_bpp;
        if (fmt == 0) { px[0] = (float)p[0]; px[1] = (float)p[1]; px[2] = (float)p[2]; }
        else          { px[0] = (float)p[2]; px[1] = (float)p[1]; px[2] = (float)p[0]; }
    } else if (fmt == 2) {
        float v = (float)src[(long)y * src_pitch + x];
        px[0] = v; px[1] = v; px[2] = v;
    } else if (fmt == 3) {
        int yv = src[(long)y * src_w + x];
        const unsigned char* uv = src + (long)src_w * src_h + (long)(y >> 1) * src_w + (x >> 1) * 2;
        yuv_to_rgbf(yv, uv[0], uv[1], px);
    } else {
        const unsigned char* grp = src + (long)y * src_pitch + (x >> 1) * 4;
        int yv = grp[(x & 1) ? 2 : 0];
        yuv_to_rgbf(yv, grp[1], grp[3], px);
    }
}

// ── samplers: raw (0..255-scale) rgb for one destination pixel ──

__device__ __forceinline__ void sample_bilinear(
    const unsigned char* __restrict__ src, float sx, float sy,
    int src_w, int src_h, int src_pitch, int src_bpp, int fmt, float px[3]
) {
    // Bilinear: clamp-to-edge taps (mirrors interpolation::bilinear).
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    float ax = sx - (float)x0, ay = sy - (float)y0;
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    x0 = max(x0, 0); y0 = max(y0, 0);
    float t00[3], t10[3], t01[3], t11[3];
    fetch_px(src, x0, y0, src_w, src_h, src_pitch, src_bpp, fmt, t00);
    fetch_px(src, x1, y0, src_w, src_h, src_pitch, src_bpp, fmt, t10);
    fetch_px(src, x0, y1, src_w, src_h, src_pitch, src_bpp, fmt, t01);
    fetch_px(src, x1, y1, src_w, src_h, src_pitch, src_bpp, fmt, t11);
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float top = t00[c] + (t10[c] - t00[c]) * ax;
        float bot = t01[c] + (t11[c] - t01[c]) * ax;
        px[c] = top + (bot - top) * ay;
    }
}

__device__ __forceinline__ void sample_nearest(
    const unsigned char* __restrict__ src, float sx, float sy,
    int src_w, int src_h, int src_pitch, int src_bpp, int fmt, float px[3]
) {
    // Nearest: round + clamp-to-edge (mirrors interpolation::nearest).
    int xn = min(max((int)roundf(sx), 0), src_w - 1);
    int yn = min(max((int)roundf(sy), 0), src_h - 1);
    fetch_px(src, xn, yn, src_w, src_h, src_pitch, src_bpp, fmt, px);
}

__device__ __forceinline__ void sample_lanczos(
    const unsigned char* __restrict__ src, float sx, float sy,
    int src_w, int src_h, int src_pitch, int src_bpp, int fmt, float px[3]
) {
    // Lanczos-3: 6x6 windowed sinc, clamp-to-edge taps, weights renormalized
    // by their sum (mirrors interpolation::lanczos).
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    float acc[3] = { 0.0f, 0.0f, 0.0f };
    float wsum = 0.0f;
    for (int j = -2; j <= 3; ++j) {
        int yj = y0 + j;
        float wy = lanczos_w(sy - (float)yj);
        int yc = min(max(yj, 0), src_h - 1);
        for (int ii = -2; ii <= 3; ++ii) {
            int xi = x0 + ii;
            float w = wy * lanczos_w(sx - (float)xi);
            int xc = min(max(xi, 0), src_w - 1);
            float t[3];
            fetch_px(src, xc, yc, src_w, src_h, src_pitch, src_bpp, fmt, t);
            #pragma unroll
            for (int c = 0; c < 3; ++c) acc[c] += w * t[c];
            wsum += w;
        }
    }
    px[0] = acc[0] / wsum; px[1] = acc[1] / wsum; px[2] = acc[2] / wsum;
}

// ── entries: KERNEL(sampler, suffix) × output writer ──

#define ARGS \
    float scale_x, float scale_y, float pad_x, float pad_y, \
    int src_w, int src_h, int src_pitch, int src_bpp, int fmt, \
    int dst_w, int dst_h, \
    float m0, float m1, float m2, \
    float is0, float is1, float is2, \
    float pad_value

#define BODY(SAMPLER, STORE) \
    int i = blockIdx.x * blockDim.x + threadIdx.x; \
    int pixels = dst_w * dst_h; \
    if (i >= pixels) return; \
    float sx, sy; \
    float px[3]; \
    if (plan_pixel(i, dst_w, dst_h, scale_x, scale_y, pad_x, pad_y, src_w, src_h, &sx, &sy)) { \
        SAMPLER(src, sx, sy, src_w, src_h, src_pitch, src_bpp, fmt, px); \
    } else { \
        px[0] = pad_value; px[1] = pad_value; px[2] = pad_value; \
    } \
    float o0 = (px[0] / 255.0f - m0) * is0; \
    float o1 = (px[1] / 255.0f - m1) * is1; \
    float o2 = (px[2] / 255.0f - m2) * is2; \
    STORE(o0, o1, o2)

#define STORE_F32(o0, o1, o2) \
    dst[i] = o0; dst[pixels + i] = o1; dst[2 * pixels + i] = o2;
#define STORE_F16(o0, o1, o2) \
    dst[i] = f2h(o0); dst[pixels + i] = f2h(o1); dst[2 * pixels + i] = f2h(o2);

extern "C" __global__ void resize_normalize_to_chw_bilinear(
    const unsigned char* __restrict__ src, float* __restrict__ dst, ARGS
) { BODY(sample_bilinear, STORE_F32) }

extern "C" __global__ void resize_normalize_to_chw_nearest(
    const unsigned char* __restrict__ src, float* __restrict__ dst, ARGS
) { BODY(sample_nearest, STORE_F32) }

extern "C" __global__ void resize_normalize_to_chw_lanczos(
    const unsigned char* __restrict__ src, float* __restrict__ dst, ARGS
) { BODY(sample_lanczos, STORE_F32) }

extern "C" __global__ void resize_normalize_to_chw_bilinear_f16(
    const unsigned char* __restrict__ src, unsigned short* __restrict__ dst, ARGS
) { BODY(sample_bilinear, STORE_F16) }

extern "C" __global__ void resize_normalize_to_chw_nearest_f16(
    const unsigned char* __restrict__ src, unsigned short* __restrict__ dst, ARGS
) { BODY(sample_nearest, STORE_F16) }

extern "C" __global__ void resize_normalize_to_chw_lanczos_f16(
    const unsigned char* __restrict__ src, unsigned short* __restrict__ dst, ARGS
) { BODY(sample_lanczos, STORE_F16) }
"#;

/// Builder for a [`Preprocessor`]: pick the resize fit, normalization, and pad
/// value, then [`build`](Self::build) (CPU) or [`build_cuda`](Self::build_cuda).
///
/// Defaults: [`ResizeMode::Letterbox`], [`Normalize::UnitScale`], pad `114`.
#[derive(Debug, Clone, Copy)]
pub struct PreprocessorBuilder {
    mode: ResizeMode,
    normalize: Normalize,
    pad_value: u8,
    sampling: InterpolationMode,
    source_format: SourceFormat,
}

impl Default for PreprocessorBuilder {
    fn default() -> Self {
        Self {
            mode: ResizeMode::Letterbox,
            normalize: Normalize::UnitScale,
            pad_value: 114,
            sampling: InterpolationMode::Bilinear,
            source_format: SourceFormat::Rgb8,
        }
    }
}

impl PreprocessorBuilder {
    /// A builder with the defaults (letterbox, unit-scale, pad 114).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the source pixel format ([`SourceFormat`]) — camera formats
    /// (NV12/YUYV/Gray) fuse their decode into the resize kernel and are fed
    /// through [`Preprocessor::run_raw`].
    pub fn source_format(mut self, format: SourceFormat) -> Self {
        self.source_format = format;
        self
    }

    /// Set the resize fit ([`ResizeMode`]).
    pub fn mode(mut self, mode: ResizeMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the normalization ([`Normalize`]).
    pub fn normalize(mut self, normalize: Normalize) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the letterbox pad value (`[0, 255]`, applied before normalization).
    /// Ignored by [`ResizeMode::Stretch`], which never pads.
    pub fn pad_value(mut self, pad_value: u8) -> Self {
        self.pad_value = pad_value;
        self
    }

    /// Set the resampling filter ([`InterpolationMode`]): `Nearest`, `Bilinear`
    /// (the default), or `Lanczos` (3-lobe windowed sinc) — the three filters in
    /// common CV/ML use. `Bicubic` is not implemented and is rejected at build.
    pub fn sampling(mut self, sampling: InterpolationMode) -> Self {
        self.sampling = sampling;
        self
    }

    /// Build a **CPU** preprocessor (host image → host tensor). Always available.
    pub fn build(self) -> Result<Preprocessor, PreprocessError> {
        if !matches!(
            self.sampling,
            InterpolationMode::Nearest | InterpolationMode::Bilinear | InterpolationMode::Lanczos
        ) {
            return Err(PreprocessError::UnsupportedSampling(self.sampling));
        }
        let (mean, inv_std) = self.normalize.mean_inv_std()?;
        Ok(Preprocessor {
            mode: self.mode,
            sampling: self.sampling,
            source_format: self.source_format,
            mean,
            inv_std,
            pad_value: self.pad_value as f32,
            #[cfg(feature = "cuda")]
            cuda: None,
        })
    }

    /// Build a **CUDA** preprocessor on `stream` (compiles the kernel once). The
    /// [`run`](Preprocessor::run) operands must then be device-resident.
    #[cfg(feature = "cuda")]
    pub fn build_cuda(self, stream: Arc<CudaStream>) -> Result<Preprocessor, PreprocessError> {
        if !matches!(
            self.sampling,
            InterpolationMode::Nearest | InterpolationMode::Bilinear | InterpolationMode::Lanczos
        ) {
            return Err(PreprocessError::UnsupportedSampling(self.sampling));
        }
        let (entry, entry_f16) = match self.sampling {
            InterpolationMode::Nearest => (
                "resize_normalize_to_chw_nearest",
                "resize_normalize_to_chw_nearest_f16",
            ),
            InterpolationMode::Bilinear => (
                "resize_normalize_to_chw_bilinear",
                "resize_normalize_to_chw_bilinear_f16",
            ),
            InterpolationMode::Lanczos => (
                "resize_normalize_to_chw_lanczos",
                "resize_normalize_to_chw_lanczos_f16",
            ),
            // Rejected by the sampling validation above.
            other => return Err(PreprocessError::UnsupportedSampling(other)),
        };
        let kernel = CudaKernel::compile(stream.context(), KERNEL_SRC, entry)?;
        let kernel_f16 = CudaKernel::compile(stream.context(), KERNEL_SRC, entry_f16)?;
        let (mean, inv_std) = self.normalize.mean_inv_std()?;
        Ok(Preprocessor {
            mode: self.mode,
            sampling: self.sampling,
            source_format: self.source_format,
            mean,
            inv_std,
            pad_value: self.pad_value as f32,
            cuda: Some(CudaBackend {
                kernel,
                kernel_f16,
                stream,
            }),
        })
    }
}

/// Image-to-tensor preprocessor: resize (+ optional pad) + normalize, on the CPU
/// or (feature `cudarc`) the GPU.
///
/// Built once via [`Preprocessor::builder`], then applied to any number of frames
/// of any resolution via [`run`](Self::run) — the target H/W is read from the
/// destination tensor each call, so one instance handles every model size.
pub struct Preprocessor {
    mode: ResizeMode,
    sampling: InterpolationMode,
    source_format: SourceFormat,
    mean: [f32; 3],
    inv_std: [f32; 3],
    pad_value: f32,
    #[cfg(feature = "cuda")]
    cuda: Option<CudaBackend>,
}

fn validate_channels<const C: usize>() -> Result<(), PreprocessError> {
    if C != 3 && C != 4 {
        return Err(PreprocessError::UnsupportedChannels(C));
    }
    Ok(())
}

fn validate_dst_shape(shape: [usize; 4], expected_n: usize) -> Result<(), PreprocessError> {
    if shape[1] != 3 || (expected_n == 1 && shape[0] != 1) {
        return Err(PreprocessError::BadOutputShape(shape));
    }
    #[cfg(feature = "cuda")]
    if shape[0] != expected_n {
        return Err(PreprocessError::BatchMismatch {
            dst_n: shape[0],
            frames: expected_n,
        });
    }
    Ok(())
}

#[cfg(feature = "cuda")]
impl PitchedSurface<'_> {
    fn validate(&self) -> Result<(), PreprocessError> {
        if self.channels != 3 && self.channels != 4 {
            return Err(PreprocessError::UnsupportedChannels(self.channels));
        }
        if self.row_pitch < self.width * self.channels
            || self.data.len() < self.row_pitch * self.height
            || self.width == 0
            || self.height == 0
        {
            return Err(PreprocessError::InvalidSurface);
        }
        Ok(())
    }
}

impl Preprocessor {
    /// Start a [`PreprocessorBuilder`].
    pub fn builder() -> PreprocessorBuilder {
        PreprocessorBuilder::new()
    }

    /// The resize mode this preprocessor applies.
    pub fn mode(&self) -> ResizeMode {
        self.mode
    }

    /// The CUDA stream this preprocessor launches on, or `None` for a CPU one.
    #[cfg(feature = "cuda")]
    pub fn stream(&self) -> Option<&Arc<CudaStream>> {
        self.cuda.as_ref().map(|c| &c.stream)
    }

    /// Build a **CUDA** letterbox preprocessor (unit-scale, pad 114) on `stream`.
    #[cfg(feature = "cuda")]
    pub fn letterbox(stream: Arc<CudaStream>) -> Result<Self, PreprocessError> {
        PreprocessorBuilder::new()
            .mode(ResizeMode::Letterbox)
            .build_cuda(stream)
    }

    /// Build a **CUDA** stretch preprocessor (unit-scale) on `stream`.
    #[cfg(feature = "cuda")]
    pub fn stretch(stream: Arc<CudaStream>) -> Result<Self, PreprocessError> {
        PreprocessorBuilder::new()
            .mode(ResizeMode::Stretch)
            .build_cuda(stream)
    }

    /// Build a **CUDA** preprocessor for an explicit [`ResizeMode`] (unit-scale, pad 114).
    #[cfg(feature = "cuda")]
    pub fn with_mode(stream: Arc<CudaStream>, mode: ResizeMode) -> Result<Self, PreprocessError> {
        PreprocessorBuilder::new().mode(mode).build_cuda(stream)
    }

    /// Resize + normalize `src` into `dst` (`[1, 3, H, W]` CHW `f32`).
    ///
    /// Generic over the source channel count: `C = 3` (RGB) or `C = 4` (RGBA — the
    /// alpha byte is skipped). A CUDA preprocessor requires device-resident operands
    /// and runs one fused kernel; a CPU preprocessor requires host-resident operands.
    ///
    /// # Errors
    ///
    /// [`PreprocessError::UnsupportedChannels`] if `C` is not 3 or 4;
    /// [`PreprocessError::NotDeviceImage`] / [`PreprocessError::NotDeviceTensor`] if a
    /// CUDA preprocessor gets host operands; [`PreprocessError::NotHostData`] if a CPU
    /// preprocessor gets device operands; [`PreprocessError::Cuda`] on a launch failure.
    pub fn run<const C: usize>(
        &self,
        src: &Image<u8, C>,
        dst: &mut Tensor<f32, 4>,
    ) -> Result<(), PreprocessError> {
        validate_channels::<C>()?;
        // Both paths write three channel planes of dst_h*dst_w — any other
        // leading dims would run past the buffer (an OOB device write on CUDA).
        validate_dst_shape(dst.shape, 1)?;
        let (dst_h, dst_w) = (dst.shape[2], dst.shape[3]);
        let a = Affine::new(self.mode, src.width(), src.height(), dst_w, dst_h);

        self.validate_typed_format::<C>()?;
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            return self.run_typed_cuda::<f32, C>(cuda, src, dst, &a);
        }
        self.run_cpu::<C>(src, dst, &a)
    }

    /// The typed `run` entries feed interleaved pixels; the source format must
    /// agree with the image's `C`. The byte stride comes from `C` itself, so
    /// `Rgb`/`Bgr` also accept `C = 4` (that's exactly the RGBA/BGRA
    /// alpha-skipped layout — keeps `run::<4>` working on a default builder).
    /// Camera formats have no typed image and go through
    /// [`run_raw`](Self::run_raw).
    fn validate_typed_format<const C: usize>(&self) -> Result<(), PreprocessError> {
        let ok = match self.source_format {
            SourceFormat::Rgba8 | SourceFormat::Bgra8 => C == 4,
            f => f.interleaved(),
        };
        if ok {
            Ok(())
        } else {
            Err(PreprocessError::FormatNeedsRawBuffer(self.source_format))
        }
    }

    // Host path built on the crate's OPTIMIZED resize kernels, FUSED FIRST:
    // every RGB+bilinear case runs the crate's fused resize+normalize+CHW
    // kernel (`resize_normalize_to_tensor_u8_to_f32_bilinear`) in one f32
    // pass; the remaining cases (Nearest / AA-Lanczos, RGBA) ride
    // `resize_fast_u8_aa` (NEON-accelerated) plus a fused scale+bias
    // normalize/CHW placement. Note the CPU resampler is antialiased on
    // downscale, so CPU and CUDA outputs may differ at the sub-pixel level
    // (CPU is the higher-quality reference).
    fn run_cpu<const C: usize>(
        &self,
        src: &Image<u8, C>,
        dst: &mut Tensor<f32, 4>,
        a: &Affine,
    ) -> Result<(), PreprocessError> {
        // CPU preprocessor requires host-resident operands.
        #[cfg(feature = "cuda")]
        if src.0.as_cudaslice().is_some() || dst.as_cudaslice().is_some() {
            return Err(PreprocessError::NotHostData);
        }

        let (dst_h, dst_w) = (dst.shape[2], dst.shape[3]);
        let pixels = dst_h * dst_w;
        // Fused normalize as one FMA: `out = raw_u8 * scale + bias`.
        let params = NormalizeParams::<3> {
            scale: [
                self.inv_std[0] / 255.0,
                self.inv_std[1] / 255.0,
                self.inv_std[2] / 255.0,
            ],
            bias: [
                -self.mean[0] * self.inv_std[0],
                -self.mean[1] * self.inv_std[1],
                -self.mean[2] * self.inv_std[2],
            ],
        };

        // Content box: the letterboxed region the image scales into (whole
        // output for Stretch). Integer placement, centred like the CUDA affine.
        // When the pad is fractional the box is rounded to whole pixels, so the
        // content/pad boundary may sit one pixel from where the GPU kernel
        // (which pads per-sample on the fractional affine) puts it.
        let (cw, ch) = match self.mode {
            ResizeMode::Stretch => (dst_w, dst_h),
            ResizeMode::Letterbox => (
                ((src.width() as f32 * a.scale_x).round() as usize).clamp(1, dst_w),
                ((src.height() as f32 * a.scale_y).round() as usize).clamp(1, dst_h),
            ),
        };
        let (px0, py0) = ((dst_w - cw) / 2, (dst_h - ch) / 2);
        let padded = px0 != 0 || py0 != 0 || cw != dst_w || ch != dst_h;
        let pad = core::array::from_fn::<f32, 3, _>(|c| {
            self.pad_value * params.scale[c] + params.bias[c]
        });

        // FUSED FIRST: every RGB case — bilinear, nearest, AA lanczos — goes
        // through a fused resize+normalize+CHW kernel: one f32 pass, no
        // intermediate u8 requantization. Stretch writes dst directly;
        // Letterbox fuses into the content box, then places it (row memcpy).
        if C == 3 {
            let (sw, sh) = (src.width(), src.height());
            let sbuf = src.0.as_slice();
            let fused = |out: &mut [f32], w: usize, h: usize| match self.sampling {
                InterpolationMode::Bilinear => {
                    resize_normalize_to_tensor_u8_to_f32_bilinear(sbuf, sw, sh, out, w, h, &params)
                }
                InterpolationMode::Nearest => {
                    resize_normalize_to_tensor_u8_to_f32_nearest(sbuf, sw, sh, out, w, h, &params)
                }
                // AA lanczos: torchvision-parity chain, fused through the
                // separable engine's SIMD horizontal pass.
                InterpolationMode::Lanczos => resize_normalize_to_tensor_u8_to_f32_separable(
                    sbuf,
                    sw,
                    sh,
                    out,
                    w,
                    h,
                    &params,
                    InterpolationMode::Lanczos,
                    true,
                ),
                // Rejected at build.
                other => Err(ImageError::UnsupportedInterpolation(other)),
            };
            if !padded {
                fused(dst.as_slice_mut(), dst_w, dst_h)?;
                return Ok(());
            }
            let mut content = vec![0.0f32; 3 * ch * cw];
            fused(&mut content, cw, ch)?;
            let dst_buf = dst.as_slice_mut();
            for (c, &pad_c) in pad.iter().enumerate() {
                let plane = &mut dst_buf[c * pixels..(c + 1) * pixels];
                plane.fill(pad_c);
                for y in 0..ch {
                    let row = &content[c * ch * cw + y * cw..c * ch * cw + (y + 1) * cw];
                    plane[(py0 + y) * dst_w + px0..(py0 + y) * dst_w + px0 + cw]
                        .copy_from_slice(row);
                }
            }
            return Ok(());
        }

        // General path (RGBA):
        // 1) SIMD resample to the content size (NEON-accelerated fast resizer).
        let mut scratch = Image::<u8, C>::from_size_val(
            ImageSize {
                width: cw,
                height: ch,
            },
            0,
        )?;
        resize_fast_u8_aa(src, &mut scratch, self.sampling, true)?;

        // 2) Pad fill + fused normalize/CHW placement of the content box.
        let sbuf = scratch.0.as_slice();
        let dst_buf = dst.as_slice_mut();
        for (c, &pad_c) in pad.iter().enumerate() {
            let plane = &mut dst_buf[c * pixels..(c + 1) * pixels];
            if padded {
                plane.fill(pad_c);
            }
            for y in 0..ch {
                let row = &sbuf[y * cw * C..];
                let out = &mut plane[(py0 + y) * dst_w + px0..];
                for x in 0..cw {
                    out[x] = row[x * C + c] as f32 * params.scale[c] + params.bias[c];
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn run_typed_cuda<T: OutElem, const C: usize>(
        &self,
        cuda: &CudaBackend,
        src: &Image<u8, C>,
        dst: &mut Tensor<T, 4>,
        a: &Affine,
    ) -> Result<(), PreprocessError> {
        let (src_w, src_h) = (src.width(), src.height());
        let src_slice = src
            .0
            .as_cudaslice()
            .ok_or(PreprocessError::NotDeviceImage)?;
        let g = SrcGeom {
            w: src_w,
            h: src_h,
            pitch: src_w * C,
            bpp: C,
            fmt: self.source_format.fmt_code(),
        };
        self.launch_cuda(cuda, src_slice, g, dst, a)
    }

    #[cfg(feature = "cuda")]
    /// [`run`](Self::run), but writing a **half-precision** (`f16`) CHW tensor —
    /// for fp16 TensorRT engines: halves output traffic on a memory-bound op and
    /// removes the cast pass before inference. CUDA preprocessors only;
    /// round-to-nearest-even conversion in-kernel.
    pub fn run_f16<const C: usize>(
        &self,
        src: &Image<u8, C>,
        dst: &mut Tensor<half::f16, 4>,
    ) -> Result<(), PreprocessError> {
        validate_channels::<C>()?;
        validate_dst_shape(dst.shape, 1)?;
        self.validate_typed_format::<C>()?;
        let cuda = self.cuda.as_ref().ok_or(PreprocessError::NotDeviceImage)?;
        let a = Affine::new(
            self.mode,
            src.width(),
            src.height(),
            dst.shape[3],
            dst.shape[2],
        );
        self.run_typed_cuda::<half::f16, C>(cuda, src, dst, &a)
    }

    #[cfg(feature = "cuda")]
    /// Preprocess a device-resident **pitched** surface (camera / NVMM buffer)
    /// straight into the CHW tensor — resize + normalize + (RGBA→)RGB in the
    /// same single fused kernel, no repack pass. CUDA preprocessors only.
    ///
    /// The resize geometry is computed from the surface's `width`×`height`
    /// exactly as [`run`](Self::run) does from an image's.
    pub fn run_surface(
        &self,
        src: &PitchedSurface<'_>,
        dst: &mut Tensor<f32, 4>,
    ) -> Result<(), PreprocessError> {
        self.run_surface_impl(src, dst)
    }

    #[cfg(feature = "cuda")]
    /// [`run_surface`](Self::run_surface) writing a half-precision tensor — the
    /// full camera-to-fp16-engine path (pitched NVMM in, fp16 CHW out) in one
    /// fused kernel.
    pub fn run_surface_f16(
        &self,
        src: &PitchedSurface<'_>,
        dst: &mut Tensor<half::f16, 4>,
    ) -> Result<(), PreprocessError> {
        self.run_surface_impl(src, dst)
    }

    #[cfg(feature = "cuda")]
    fn run_surface_impl<T: OutElem>(
        &self,
        src: &PitchedSurface<'_>,
        dst: &mut Tensor<T, 4>,
    ) -> Result<(), PreprocessError> {
        src.validate()?;
        validate_dst_shape(dst.shape, 1)?;
        let cuda = self.cuda.as_ref().ok_or(PreprocessError::NotDeviceImage)?;
        let a = Affine::new(self.mode, src.width, src.height, dst.shape[3], dst.shape[2]);
        let g = SrcGeom {
            w: src.width,
            h: src.height,
            pitch: src.row_pitch,
            bpp: src.channels,
            fmt: self.surface_fmt_code()?,
        };
        self.launch_cuda(cuda, src.data, g, dst, &a)
    }

    /// Pitched surfaces are interleaved by construction (pitch covers
    /// `width*channels`); the subsampled camera formats address chroma from the
    /// frame *width*, so a pitched NV12/YUYV surface would decode garbage —
    /// those go through the tightly-packed [`run_raw`](Self::run_raw) instead.
    /// The format only selects the in-kernel swizzle; the byte stride comes
    /// from the surface's own `channels`.
    #[cfg(feature = "cuda")]
    fn surface_fmt_code(&self) -> Result<i32, PreprocessError> {
        if self.source_format.interleaved() {
            Ok(self.source_format.fmt_code())
        } else {
            Err(PreprocessError::FormatNeedsRawBuffer(self.source_format))
        }
    }

    /// Preprocess a **raw device frame buffer** (`CudaSlice<u8>`) in the
    /// builder's [`SourceFormat`] — the entry point for camera formats
    /// (NV12 / YUYV / Gray, and tightly-packed interleaved buffers): color
    /// decode is fused into the resize taps, so a capture frame becomes the
    /// normalized CHW tensor in **one kernel launch** with no intermediate
    /// RGB image in device memory.
    ///
    /// The buffer must be tightly packed (`SourceFormat::buffer_len` bytes or
    /// more; extra tail bytes — e.g. V4L2 buffer padding — are ignored).
    ///
    /// # Errors
    ///
    /// [`PreprocessError::InvalidRawSource`] if the buffer is too small or the
    /// dimensions violate the format's subsampling constraints (even dims for
    /// NV12, even width for YUYV); [`PreprocessError::NotDeviceImage`] on a
    /// CPU preprocessor.
    #[cfg(feature = "cuda")]
    pub fn run_raw(
        &self,
        src: &cudarc::driver::CudaSlice<u8>,
        src_w: usize,
        src_h: usize,
        dst: &mut Tensor<f32, 4>,
    ) -> Result<(), PreprocessError> {
        self.run_raw_impl(src, src_w, src_h, dst)
    }

    /// [`run_raw`](Self::run_raw) writing a half-precision tensor — raw camera
    /// frame to fp16 CHW engine input in one fused kernel.
    #[cfg(feature = "cuda")]
    pub fn run_raw_f16(
        &self,
        src: &cudarc::driver::CudaSlice<u8>,
        src_w: usize,
        src_h: usize,
        dst: &mut Tensor<half::f16, 4>,
    ) -> Result<(), PreprocessError> {
        self.run_raw_impl(src, src_w, src_h, dst)
    }

    #[cfg(feature = "cuda")]
    fn run_raw_impl<T: OutElem>(
        &self,
        src: &cudarc::driver::CudaSlice<u8>,
        src_w: usize,
        src_h: usize,
        dst: &mut Tensor<T, 4>,
    ) -> Result<(), PreprocessError> {
        validate_dst_shape(dst.shape, 1)?;
        let cuda = self.cuda.as_ref().ok_or(PreprocessError::NotDeviceImage)?;
        self.validate_raw(src.len(), src_w, src_h)?;
        let a = Affine::new(self.mode, src_w, src_h, dst.shape[3], dst.shape[2]);
        let g = self.source_format.geom(src_w, src_h);
        self.launch_cuda(cuda, src, g, dst, &a)
    }

    /// Preprocess `N` same-sized raw frames into a **batched** `[N, 3, H, W]`
    /// tensor — one [`run_raw`](Self::run_raw) launch per frame, each writing
    /// its own CHW plane of `dst`. All launches queue on the preprocessor's
    /// stream, so a single sync covers the whole batch (multi-camera rigs,
    /// batched TensorRT engines).
    ///
    /// # Errors
    ///
    /// [`PreprocessError::BatchMismatch`] if `dst.shape[0] != frames.len()`;
    /// otherwise as [`run_raw`](Self::run_raw), checked per frame.
    #[cfg(feature = "cuda")]
    pub fn run_raw_batch(
        &self,
        frames: &[&cudarc::driver::CudaSlice<u8>],
        src_w: usize,
        src_h: usize,
        dst: &mut Tensor<f32, 4>,
    ) -> Result<(), PreprocessError> {
        self.run_raw_batch_impl(frames, src_w, src_h, dst)
    }

    /// [`run_raw_batch`](Self::run_raw_batch) writing a half-precision tensor —
    /// batched raw frames straight into an fp16 engine input.
    #[cfg(feature = "cuda")]
    pub fn run_raw_batch_f16(
        &self,
        frames: &[&cudarc::driver::CudaSlice<u8>],
        src_w: usize,
        src_h: usize,
        dst: &mut Tensor<half::f16, 4>,
    ) -> Result<(), PreprocessError> {
        self.run_raw_batch_impl(frames, src_w, src_h, dst)
    }

    #[cfg(feature = "cuda")]
    fn run_raw_batch_impl<T: OutElem>(
        &self,
        frames: &[&cudarc::driver::CudaSlice<u8>],
        src_w: usize,
        src_h: usize,
        dst: &mut Tensor<T, 4>,
    ) -> Result<(), PreprocessError> {
        let cuda = self.cuda.as_ref().ok_or(PreprocessError::NotDeviceImage)?;
        validate_dst_shape(dst.shape, frames.len())?;
        let (dst_h, dst_w) = (dst.shape[2], dst.shape[3]);
        let a = Affine::new(self.mode, src_w, src_h, dst_w, dst_h);
        let g = self.source_format.geom(src_w, src_h);
        for src in frames {
            self.validate_raw(src.len(), src_w, src_h)?;
        }
        let dst_slice = dst
            .as_cudaslice_mut()
            .ok_or(PreprocessError::NotDeviceTensor)?;
        let plane = 3 * dst_h * dst_w;
        for (i, src) in frames.iter().enumerate() {
            let mut view = dst_slice.slice_mut(i * plane..(i + 1) * plane);
            self.launch_view(cuda, src, g, &mut view, dst_w, dst_h, &a)?;
        }
        Ok(())
    }

    /// A raw frame must satisfy the format's subsampling constraints and cover
    /// at least `buffer_len` bytes (longer is fine — capture buffers often pad).
    #[cfg(feature = "cuda")]
    fn validate_raw(&self, got: usize, w: usize, h: usize) -> Result<(), PreprocessError> {
        let f = self.source_format;
        let need = f.buffer_len(w, h);
        if !f.dims_ok(w, h) || got < need {
            return Err(PreprocessError::InvalidRawSource {
                format: f,
                width: w,
                height: h,
                got,
                need,
            });
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn launch_cuda<T: OutElem>(
        &self,
        cuda: &CudaBackend,
        src_slice: &cudarc::driver::CudaSlice<u8>,
        g: SrcGeom,
        dst: &mut Tensor<T, 4>,
        a: &Affine,
    ) -> Result<(), PreprocessError> {
        let (dst_h, dst_w) = (dst.shape[2], dst.shape[3]);
        let dst_slice = dst
            .as_cudaslice_mut()
            .ok_or(PreprocessError::NotDeviceTensor)?;
        let mut dst_view = dst_slice.slice_mut(..);
        self.launch_view(cuda, src_slice, g, &mut dst_view, dst_w, dst_h, a)
    }

    /// The single launch seam: every entry point ends here. `dst_view` is one
    /// image's `3*dst_h*dst_w` CHW plane — for batches, a sub-slice of the
    /// `[N, 3, H, W]` tensor.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn launch_view<T: OutElem>(
        &self,
        cuda: &CudaBackend,
        src_slice: &cudarc::driver::CudaSlice<u8>,
        g: SrcGeom,
        dst_view: &mut cudarc::driver::CudaViewMut<'_, T>,
        dst_w: usize,
        dst_h: usize,
        a: &Affine,
    ) -> Result<(), PreprocessError> {
        // The kernel indexes pixels as `int`; keep every dim + the pixel count
        // within i32 so the `as i32` / `as u32` launch args never truncate.
        let lim = i32::MAX as usize;
        if g.w > lim || g.h > lim || g.pitch > lim || dst_w.saturating_mul(dst_h) > lim {
            return Err(PreprocessError::DimensionsTooLarge);
        }

        let (sw, sh, sp, bpp) = (g.w as i32, g.h as i32, g.pitch as i32, g.bpp as i32);
        let fmt = g.fmt;
        let (dw, dh) = (dst_w as i32, dst_h as i32);
        let total = (dst_w * dst_h) as u32;
        let ([m0, m1, m2], [is0, is1, is2]) = (self.mean, self.inv_std);
        let pv = self.pad_value;

        T::kernel(cuda)
            .launch_builder(&cuda.stream)
            .arg(src_slice)
            .arg(dst_view)
            .arg(&a.scale_x)
            .arg(&a.scale_y)
            .arg(&a.pad_x)
            .arg(&a.pad_y)
            .arg(&sw)
            .arg(&sh)
            .arg(&sp)
            .arg(&bpp)
            .arg(&fmt)
            .arg(&dw)
            .arg(&dh)
            .arg(&m0)
            .arg(&m1)
            .arg(&m2)
            .arg(&is0)
            .arg(&is1)
            .arg(&is2)
            .arg(&pv)
            .launch_1d(total)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};

    fn host_solid<const C: usize>(w: usize, h: usize, px: [u8; C]) -> Image<u8, C> {
        let data: Vec<u8> = (0..w * h).flat_map(|_| px).collect();
        Image::<u8, C>::new(
            ImageSize {
                width: w,
                height: h,
            },
            data,
        )
        .unwrap()
    }

    fn host_dst(h: usize, w: usize) -> Tensor<f32, 4> {
        Tensor::<f32, 4>::from_shape_vec([1, 3, h, w], vec![0.0; 3 * h * w]).unwrap()
    }

    // A deterministic non-solid image (per-pixel gradient) for resampling tests.
    #[cfg(feature = "cuda")]
    fn host_gradient<const C: usize>(w: usize, h: usize) -> Image<u8, C> {
        let data: Vec<u8> = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    // Smooth ramp (no wraparound) so band comparisons between
                    // differently-aligned resamplers stay meaningful.
                    (0..C).map(move |c| {
                        ((x * 127 / (w - 1) + y * 127 / (h - 1)) as u8).saturating_add(c as u8 * 20)
                    })
                })
            })
            .collect();
        Image::<u8, C>::new(
            ImageSize {
                width: w,
                height: h,
            },
            data,
        )
        .unwrap()
    }

    const ALL_SAMPLING: [InterpolationMode; 3] = [
        InterpolationMode::Nearest,
        InterpolationMode::Bilinear,
        InterpolationMode::Lanczos,
    ];

    // A solid source stays exactly solid through every resampler (unit DC
    // response) under unit-scale: out == v/255 everywhere.
    #[test]
    fn cpu_stretch_solid_all_sampling() {
        for sampling in ALL_SAMPLING {
            let pre = Preprocessor::builder()
                .mode(ResizeMode::Stretch)
                .sampling(sampling)
                .build()
                .unwrap();
            let src = host_solid(5, 3, [10u8, 20, 30]);
            let mut dst = host_dst(4, 4);
            pre.run(&src, &mut dst).unwrap();
            let out = dst.as_slice();
            let px = 4 * 4;
            for (c, &v) in [10.0f32, 20.0, 30.0].iter().enumerate() {
                for i in 0..px {
                    assert!(
                        (out[c * px + i] - v / 255.0).abs() < 1e-4,
                        "{sampling:?} chan {c}"
                    );
                }
            }
        }
    }

    // Letterbox: content box exactly the constant, pad region exactly pad_value.
    #[test]
    fn cpu_letterbox_pad_geometry() {
        let pre = Preprocessor::builder().pad_value(32).build().unwrap();
        // Square 4×4 into 8×4 → scale 1, content columns 2..6, pad elsewhere.
        let src = host_solid(4, 4, [100u8, 100, 100]);
        let mut dst = host_dst(4, 8);
        pre.run(&src, &mut dst).unwrap();
        let out = dst.as_slice();
        let px = 4 * 8;
        for c in 0..3 {
            for y in 0..4 {
                for x in 0..8 {
                    let got = out[c * px + y * 8 + x];
                    let want = if (2..6).contains(&x) { 100.0 } else { 32.0 } / 255.0;
                    assert!((got - want).abs() < 1e-4, "chan {c} ({x},{y}): {got}");
                }
            }
        }
    }

    // RGBA (alpha ignored) must match tight RGB for the same colour.
    #[test]
    fn cpu_rgba_matches_rgb() {
        let pre = Preprocessor::builder()
            .mode(ResizeMode::Stretch)
            .build()
            .unwrap();
        let mut d3 = host_dst(8, 8);
        let mut d4 = host_dst(8, 8);
        pre.run(&host_solid(6, 4, [40u8, 80, 120]), &mut d3)
            .unwrap();
        pre.run(&host_solid(6, 4, [40u8, 80, 120, 200]), &mut d4)
            .unwrap();
        for (x, y) in d3.as_slice().iter().zip(d4.as_slice()) {
            assert!((x - y).abs() < 1e-4);
        }
    }

    // MeanStd folds the ImageNet normalize into the same pass.
    #[test]
    fn cpu_imagenet_normalize() {
        let pre = Preprocessor::builder()
            .mode(ResizeMode::Stretch)
            .normalize(Normalize::imagenet())
            .build()
            .unwrap();
        let mut dst = host_dst(4, 4);
        pre.run(&host_solid(4, 4, [128u8, 128, 128]), &mut dst)
            .unwrap();
        let out = dst.as_slice();
        let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
        for c in 0..3 {
            let want = (128.0 / 255.0 - mean[c]) / std[c];
            assert!((out[c * 16] - want).abs() < 1e-4, "chan {c}");
        }
    }

    #[test]
    fn rejects_bad_channels() {
        let pre = Preprocessor::builder().build().unwrap();
        let mut dst = host_dst(2, 2);
        assert!(matches!(
            pre.run(&host_solid(2, 2, [5u8]), &mut dst),
            Err(PreprocessError::UnsupportedChannels(1))
        ));
    }

    // std = 0 (or non-finite mean/std) must fail at build, not emit inf/NaN.
    #[test]
    fn rejects_invalid_normalize() {
        let bad = Preprocessor::builder()
            .normalize(Normalize::MeanStd {
                mean: [0.5; 3],
                std: [0.0, 0.2, 0.2],
            })
            .build();
        assert!(matches!(bad, Err(PreprocessError::InvalidNormalize { .. })));
    }

    // A non-[1,3,H,W] destination must be rejected before any write.
    #[test]
    fn rejects_bad_output_shape() {
        let pre = Preprocessor::builder().build().unwrap();
        let mut dst = Tensor::<f32, 4>::from_shape_vec([1, 1, 4, 4], vec![0.0; 16]).unwrap();
        assert!(matches!(
            pre.run(&host_solid(2, 2, [5u8, 5, 5]), &mut dst),
            Err(PreprocessError::BadOutputShape([1, 1, 4, 4]))
        ));
    }

    // Bicubic is declared in InterpolationMode but not wired here — reject at build.
    #[test]
    fn rejects_bicubic_sampling() {
        let bad = Preprocessor::builder()
            .sampling(InterpolationMode::Bicubic)
            .build();
        assert!(matches!(
            bad,
            Err(PreprocessError::UnsupportedSampling(
                InterpolationMode::Bicubic
            ))
        ));
    }

    // GPU: a solid source stays exactly solid through every kernel sampling
    // branch (nearest / bilinear / lanczos), including letterbox padding.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_solid_all_sampling() {
        use cudarc::driver::CudaContext;
        use kornia_tensor::zeros_cuda;

        let stream = CudaContext::new(0).unwrap().default_stream();
        let src = host_solid(5, 3, [10u8, 20, 30]);
        for sampling in ALL_SAMPLING {
            let pre = Preprocessor::builder()
                .mode(ResizeMode::Stretch)
                .sampling(sampling)
                .build_cuda(stream.clone())
                .unwrap();
            let dev: Image<u8, 3> = Image(src.0.to_cuda(&stream).unwrap());
            let mut dst = zeros_cuda::<f32, 4>([1, 3, 4, 4], &stream).unwrap();
            pre.run(&dev, &mut dst).unwrap();
            let out = stream.clone_dtoh(dst.as_cudaslice().unwrap()).unwrap();
            let px = 4 * 4;
            for (c, &v) in [10.0f32, 20.0, 30.0].iter().enumerate() {
                for i in 0..px {
                    assert!(
                        (out[c * px + i] - v / 255.0).abs() < 1e-4,
                        "{sampling:?} chan {c}"
                    );
                }
            }
        }
    }

    // A pitched RGBA surface (rows padded with garbage) must produce exactly
    // the same output as the equivalent tight RGBA image — the kernel must
    // step by pitch and never read the padding.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn pitched_surface_matches_tight() {
        use cudarc::driver::CudaContext;
        use kornia_tensor::zeros_cuda;

        let stream = CudaContext::new(0).unwrap().default_stream();
        let (w, h, pitch) = (23usize, 17usize, 23 * 4 + 13); // odd pitch + garbage pad
        let tight = host_gradient::<4>(w, h);

        let mut pitched = vec![0xAAu8; pitch * h]; // garbage everywhere
        for y in 0..h {
            pitched[y * pitch..y * pitch + w * 4]
                .copy_from_slice(&tight.0.as_slice()[y * w * 4..(y + 1) * w * 4]);
        }
        let dev_pitched = stream.clone_htod(&pitched).unwrap();

        for mode in [ResizeMode::Letterbox, ResizeMode::Stretch] {
            let pre = Preprocessor::builder()
                .mode(mode)
                .normalize(Normalize::imagenet())
                .build_cuda(stream.clone())
                .unwrap();

            let dev_img: Image<u8, 4> = Image(tight.0.to_cuda(&stream).unwrap());
            let mut d_img = zeros_cuda::<f32, 4>([1, 3, 6, 8], &stream).unwrap();
            pre.run(&dev_img, &mut d_img).unwrap();

            let surf = PitchedSurface {
                data: &dev_pitched,
                width: w,
                height: h,
                row_pitch: pitch,
                channels: 4,
            };
            let mut d_surf = zeros_cuda::<f32, 4>([1, 3, 6, 8], &stream).unwrap();
            pre.run_surface(&surf, &mut d_surf).unwrap();

            let a = stream.clone_dtoh(d_img.as_cudaslice().unwrap()).unwrap();
            let b = stream.clone_dtoh(d_surf.as_cudaslice().unwrap()).unwrap();
            assert_eq!(
                a, b,
                "pitched surface diverged from tight image in {mode:?}"
            );
        }
    }

    // fp16 output: kernel-side round-to-nearest-even conversion must equal the
    // half crate's from_f32 of the f32 kernel result, bit for bit.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn f16_matches_f32_rounded() {
        use cudarc::driver::CudaContext;
        use kornia_tensor::zeros_cuda;

        let stream = CudaContext::new(0).unwrap().default_stream();
        let src = host_gradient::<3>(23, 17);
        let dev: Image<u8, 3> = Image(src.0.to_cuda(&stream).unwrap());
        for sampling in ALL_SAMPLING {
            let pre = Preprocessor::builder()
                .mode(ResizeMode::Letterbox)
                .sampling(sampling)
                .normalize(Normalize::imagenet())
                .build_cuda(stream.clone())
                .unwrap();
            let mut d32 = zeros_cuda::<f32, 4>([1, 3, 6, 8], &stream).unwrap();
            pre.run(&dev, &mut d32).unwrap();
            let mut d16 = zeros_cuda::<half::f16, 4>([1, 3, 6, 8], &stream).unwrap();
            pre.run_f16(&dev, &mut d16).unwrap();

            let h32 = stream.clone_dtoh(d32.as_cudaslice().unwrap()).unwrap();
            let h16 = stream.clone_dtoh(d16.as_cudaslice().unwrap()).unwrap();
            for (i, (a, b)) in h32.iter().zip(&h16).enumerate() {
                assert_eq!(
                    half::f16::from_f32(*a).to_bits(),
                    b.to_bits(),
                    "{sampling:?} elem {i}: f32 {a} vs f16 {b}"
                );
            }
        }
    }

    // CPU vs CUDA on a real gradient: the CPU resampler is antialiased /
    // centre-aligned (PIL-grade) while the CUDA kernel samples directly, so
    // outputs are NOT bit-identical — but they must describe the same image.
    // A loose band still catches the real failure modes (channel swaps, CHW
    // transposes, normalize or pad errors), which show up as O(0.5) diffs.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cpu_close_to_cuda() {
        use cudarc::driver::CudaContext;
        use kornia_tensor::zeros_cuda;

        let stream = CudaContext::new(0).unwrap().default_stream();
        // Letterbox geometry with an INTEGER pad (20×10 → 8×6: scale 0.4, one
        // whole pad row top+bottom) so both backends put the content/pad
        // boundary on the same row and the band measures resampling only; the
        // fractional-pad case differs by design (see run_cpu).
        let src_lb = host_gradient::<3>(20, 10);
        let src_st = host_gradient::<3>(23, 17);
        for mode in [ResizeMode::Letterbox, ResizeMode::Stretch] {
            let src = match mode {
                ResizeMode::Letterbox => &src_lb,
                ResizeMode::Stretch => &src_st,
            };
            for sampling in ALL_SAMPLING {
                let cpu = Preprocessor::builder()
                    .mode(mode)
                    .sampling(sampling)
                    .build()
                    .unwrap();
                let gpu = Preprocessor::builder()
                    .mode(mode)
                    .sampling(sampling)
                    .build_cuda(stream.clone())
                    .unwrap();

                let mut d_cpu = host_dst(6, 8);
                cpu.run(src, &mut d_cpu).unwrap();

                let dev: Image<u8, 3> = Image(src.0.to_cuda(&stream).unwrap());
                let mut d_gpu = zeros_cuda::<f32, 4>([1, 3, 6, 8], &stream).unwrap();
                gpu.run(&dev, &mut d_gpu).unwrap();
                let gpu_host = stream.clone_dtoh(d_gpu.as_cudaslice().unwrap()).unwrap();

                let n = gpu_host.len() as f32;
                let mut mean_abs = 0.0f32;
                let mut max_abs = 0.0f32;
                for (a, b) in d_cpu.as_slice().iter().zip(&gpu_host) {
                    let d = (a - b).abs();
                    mean_abs += d;
                    max_abs = max_abs.max(d);
                }
                mean_abs /= n;
                assert!(
                    mean_abs < 0.05 && max_abs < 0.25,
                    "CPU/CUDA diverge in {mode:?}/{sampling:?}: mean {mean_abs} max {max_abs}"
                );
            }
        }
    }

    // The typed run() entries only accept interleaved formats matching C;
    // camera formats must be routed to run_raw. No GPU needed.
    #[test]
    fn typed_run_rejects_camera_formats() {
        let pre = Preprocessor::builder()
            .source_format(SourceFormat::Nv12)
            .build()
            .unwrap();
        let src = host_solid(4, 4, [1u8, 2, 3]);
        let mut dst = host_dst(4, 4);
        assert!(matches!(
            pre.run(&src, &mut dst),
            Err(PreprocessError::FormatNeedsRawBuffer(SourceFormat::Nv12))
        ));
        // Channel-count / format mismatch on interleaved formats too.
        let pre = Preprocessor::builder()
            .source_format(SourceFormat::Rgba8)
            .build()
            .unwrap();
        assert!(matches!(
            pre.run(&src, &mut dst),
            Err(PreprocessError::FormatNeedsRawBuffer(SourceFormat::Rgba8))
        ));
    }

    // Deterministic raw bytes for camera-format buffers.
    #[cfg(feature = "cuda")]
    fn raw_bytes(len: usize) -> Vec<u8> {
        (0..len).map(|i| ((i * 7 + 13) % 251) as u8).collect()
    }

    // Fused decode-in-the-taps must equal the chained path (CPU decode to RGB
    // — bit-exact to the GPU Q20 kernels — then the typed RGB run) for every
    // camera format and sampling mode. The kernel quantizes each decoded tap
    // to integer 0..255 exactly like the standalone decoders, so the resample
    // arithmetic sees identical inputs and the outputs match bit-for-bit.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_fused_formats_match_chained() {
        use crate::color::{bgr_from_rgb, rgb_from_gray, rgb_from_nv12, rgb_from_yuyv};
        use cudarc::driver::CudaContext;
        use kornia_tensor::zeros_cuda;

        let stream = CudaContext::new(0).unwrap().default_stream();
        let (w, h) = (8usize, 6usize);
        let size = ImageSize {
            width: w,
            height: h,
        };

        // (format, raw buffer, CPU-decoded RGB reference)
        let cases: Vec<(SourceFormat, Vec<u8>, Image<u8, 3>)> = {
            let mut v = Vec::new();

            let raw = raw_bytes(w * h * 3 / 2);
            let mut rgb = Image::<u8, 3>::new(size, vec![0; w * h * 3]).unwrap();
            rgb_from_nv12(&raw, &mut rgb).unwrap();
            v.push((SourceFormat::Nv12, raw, rgb));

            let raw = raw_bytes(w * h * 2);
            let mut rgb = Image::<u8, 3>::new(size, vec![0; w * h * 3]).unwrap();
            rgb_from_yuyv(&raw, &mut rgb).unwrap();
            v.push((SourceFormat::Yuyv, raw, rgb));

            let raw = raw_bytes(w * h);
            let gray = Image::<u8, 1>::new(size, raw.clone()).unwrap();
            let mut rgb = Image::<u8, 3>::new(size, vec![0; w * h * 3]).unwrap();
            rgb_from_gray(&gray, &mut rgb).unwrap();
            v.push((SourceFormat::Gray8, raw, rgb));

            // BGR raw buffer; the swap is involutive so bgr_from_rgb decodes it.
            let raw = raw_bytes(w * h * 3);
            let bgr = Image::<u8, 3>::new(size, raw.clone()).unwrap();
            let mut rgb = Image::<u8, 3>::new(size, vec![0; w * h * 3]).unwrap();
            bgr_from_rgb(&bgr, &mut rgb).unwrap();
            v.push((SourceFormat::Bgr8, raw, rgb));

            v
        };

        for (fmt, raw, rgb_ref) in &cases {
            let d_raw = stream.clone_htod(raw).unwrap();
            let d_rgb: Image<u8, 3> = Image(rgb_ref.0.to_cuda(&stream).unwrap());
            for sampling in ALL_SAMPLING {
                let fused = Preprocessor::builder()
                    .sampling(sampling)
                    .source_format(*fmt)
                    .build_cuda(stream.clone())
                    .unwrap();
                let chained = Preprocessor::builder()
                    .sampling(sampling)
                    .build_cuda(stream.clone())
                    .unwrap();

                let mut d_fused = zeros_cuda::<f32, 4>([1, 3, 5, 7], &stream).unwrap();
                fused.run_raw(&d_raw, w, h, &mut d_fused).unwrap();
                let mut d_chain = zeros_cuda::<f32, 4>([1, 3, 5, 7], &stream).unwrap();
                chained.run(&d_rgb, &mut d_chain).unwrap();

                let a = stream.clone_dtoh(d_fused.as_cudaslice().unwrap()).unwrap();
                let b = stream.clone_dtoh(d_chain.as_cudaslice().unwrap()).unwrap();
                for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                    assert!(
                        (x - y).abs() <= 1e-6,
                        "{fmt:?}/{sampling:?} idx {i}: fused {x} chained {y}"
                    );
                }
            }
        }
    }

    // run_raw_batch writes each frame's CHW plane exactly as a run_raw into a
    // single-image tensor would; batch-dim mismatch errors out.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_run_raw_batch_matches_single() {
        use cudarc::driver::CudaContext;
        use kornia_tensor::zeros_cuda;

        let stream = CudaContext::new(0).unwrap().default_stream();
        let (w, h) = (8usize, 6usize);
        let pre = Preprocessor::builder()
            .source_format(SourceFormat::Nv12)
            .build_cuda(stream.clone())
            .unwrap();

        let raws: Vec<_> = (0..3)
            .map(|k| {
                let mut b = raw_bytes(w * h * 3 / 2);
                b.iter_mut().for_each(|v| *v = v.wrapping_add(k * 31));
                stream.clone_htod(&b).unwrap()
            })
            .collect();
        let frames: Vec<_> = raws.iter().collect();

        let mut d_batch = zeros_cuda::<f32, 4>([3, 3, 5, 7], &stream).unwrap();
        pre.run_raw_batch(&frames, w, h, &mut d_batch).unwrap();
        let batch = stream.clone_dtoh(d_batch.as_cudaslice().unwrap()).unwrap();

        let plane = 3 * 5 * 7;
        for (i, raw) in raws.iter().enumerate() {
            let mut d_one = zeros_cuda::<f32, 4>([1, 3, 5, 7], &stream).unwrap();
            pre.run_raw(raw, w, h, &mut d_one).unwrap();
            let one = stream.clone_dtoh(d_one.as_cudaslice().unwrap()).unwrap();
            assert_eq!(&batch[i * plane..(i + 1) * plane], &one[..], "frame {i}");
        }

        // Batch-dim mismatch.
        let mut d_bad = zeros_cuda::<f32, 4>([2, 3, 5, 7], &stream).unwrap();
        assert!(matches!(
            pre.run_raw_batch(&frames, w, h, &mut d_bad),
            Err(PreprocessError::BatchMismatch {
                dst_n: 2,
                frames: 3
            })
        ));
    }

    // run_raw validates buffer length and subsampling dims before launching.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_run_raw_validates_source() {
        use cudarc::driver::CudaContext;
        use kornia_tensor::zeros_cuda;

        let stream = CudaContext::new(0).unwrap().default_stream();
        let pre = Preprocessor::builder()
            .source_format(SourceFormat::Nv12)
            .build_cuda(stream.clone())
            .unwrap();
        let mut dst = zeros_cuda::<f32, 4>([1, 3, 4, 4], &stream).unwrap();

        // Too small for 8x6 NV12 (needs 72 bytes).
        let short = stream.clone_htod(&raw_bytes(60)).unwrap();
        assert!(matches!(
            pre.run_raw(&short, 8, 6, &mut dst),
            Err(PreprocessError::InvalidRawSource { need: 72, .. })
        ));
        // Odd height violates NV12 subsampling.
        let raw = stream.clone_htod(&raw_bytes(8 * 5 * 2)).unwrap();
        assert!(matches!(
            pre.run_raw(&raw, 8, 5, &mut dst),
            Err(PreprocessError::InvalidRawSource { .. })
        ));
        // A pitched surface with a camera format is rejected.
        let surf_buf = stream.clone_htod(&raw_bytes(8 * 6 * 4)).unwrap();
        let surf = PitchedSurface {
            data: &surf_buf,
            width: 8,
            height: 6,
            row_pitch: 32,
            channels: 4,
        };
        assert!(matches!(
            pre.run_surface(&surf, &mut dst),
            Err(PreprocessError::FormatNeedsRawBuffer(SourceFormat::Nv12))
        ));
    }
}
