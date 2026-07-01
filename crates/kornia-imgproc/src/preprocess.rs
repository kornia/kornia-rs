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

#[cfg(feature = "cudarc")]
use std::sync::Arc;

#[cfg(feature = "cudarc")]
use cudarc::driver::CudaStream;
use kornia_image::Image;
use kornia_tensor::Tensor;
#[cfg(feature = "cudarc")]
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

impl Normalize {
    /// The standard ImageNet mean/std (RGB), matching torchvision.
    pub fn imagenet() -> Self {
        Normalize::MeanStd {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
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

/// Errors from preprocessing.
#[derive(Debug, thiserror::Error)]
pub enum PreprocessError {
    /// A CUDA error from kernel compilation or launch.
    #[cfg(feature = "cudarc")]
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
    /// Source or output dimensions exceed the kernel's 32-bit indexing (the CUDA
    /// path indexes pixels as `int`). Unreachable on real hardware, guarded anyway.
    #[cfg(feature = "cudarc")]
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
    pad_x: f32,
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

#[cfg(feature = "cudarc")]
struct CudaBackend {
    kernel: CudaKernel,
    stream: Arc<CudaStream>,
}

// 1-D grid over the `dst_w * dst_h` output pixels. Reads interleaved `src_bpp`-byte
// pixels (3 = RGB, 4 = RGBA with the alpha byte skipped) with a hand-rolled
// bilinear sample (CUDA textures only support 1/2/4-channel elements, not 3),
// applies `(v/255 - mean) * inv_std` per channel, and writes
// channel-planar (CHW) `f32`. One parameterized kernel covers every mode / format /
// normalization — config is passed as launch args (single JIT compile). `src_pitch`
// is a parameter for generality, but kornia `Image`s are tight, so the launcher
// always passes `src_w * C`.
#[cfg(feature = "cudarc")]
const KERNEL_SRC: &str = r#"
extern "C" __global__ void resize_normalize_to_chw(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    float scale_x, float scale_y, float pad_x, float pad_y,
    int src_w, int src_h, int src_pitch, int src_bpp,
    int dst_w, int dst_h,
    float m0, float m1, float m2,
    float is0, float is1, float is2,
    float pad_value
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = dst_w * dst_h;
    if (i >= pixels) return;
    int ox = i % dst_w;
    int oy = i / dst_w;

    float mean[3] = { m0, m1, m2 };
    float inv_std[3] = { is0, is1, is2 };

    float sx = ((float)ox - pad_x) / scale_x;
    float sy = ((float)oy - pad_y) / scale_y;

    if (sx < 0.0f || sy < 0.0f || sx >= (float)src_w || sy >= (float)src_h) {
        float pv = pad_value / 255.0f;
        #pragma unroll
        for (int c = 0; c < 3; ++c) dst[c * pixels + i] = (pv - mean[c]) * inv_std[c];
        return;
    }

    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    float ax = sx - (float)x0, ay = sy - (float)y0;
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    x0 = max(x0, 0); y0 = max(y0, 0);

    const unsigned char* r0 = src + (long)y0 * src_pitch;
    const unsigned char* r1 = src + (long)y1 * src_pitch;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float v00 = (float)r0[x0 * src_bpp + c], v10 = (float)r0[x1 * src_bpp + c];
        float v01 = (float)r1[x0 * src_bpp + c], v11 = (float)r1[x1 * src_bpp + c];
        float top = v00 + (v10 - v00) * ax;
        float bot = v01 + (v11 - v01) * ax;
        float v = (top + (bot - top) * ay) / 255.0f;
        dst[c * pixels + i] = (v - mean[c]) * inv_std[c];
    }
}
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
}

impl Default for PreprocessorBuilder {
    fn default() -> Self {
        Self {
            mode: ResizeMode::Letterbox,
            normalize: Normalize::UnitScale,
            pad_value: 114,
        }
    }
}

impl PreprocessorBuilder {
    /// A builder with the defaults (letterbox, unit-scale, pad 114).
    pub fn new() -> Self {
        Self::default()
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

    /// Build a **CPU** preprocessor (host image → host tensor). Always available.
    pub fn build(self) -> Result<Preprocessor, PreprocessError> {
        let (mean, inv_std) = self.normalize.mean_inv_std()?;
        Ok(Preprocessor {
            mode: self.mode,
            mean,
            inv_std,
            pad_value: self.pad_value as f32,
            #[cfg(feature = "cudarc")]
            cuda: None,
        })
    }

    /// Build a **CUDA** preprocessor on `stream` (compiles the kernel once). The
    /// [`run`](Preprocessor::run) operands must then be device-resident.
    #[cfg(feature = "cudarc")]
    pub fn build_cuda(self, stream: Arc<CudaStream>) -> Result<Preprocessor, PreprocessError> {
        let kernel = CudaKernel::compile(stream.context(), KERNEL_SRC, "resize_normalize_to_chw")?;
        let (mean, inv_std) = self.normalize.mean_inv_std()?;
        Ok(Preprocessor {
            mode: self.mode,
            mean,
            inv_std,
            pad_value: self.pad_value as f32,
            cuda: Some(CudaBackend { kernel, stream }),
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
    mean: [f32; 3],
    inv_std: [f32; 3],
    pad_value: f32,
    #[cfg(feature = "cudarc")]
    cuda: Option<CudaBackend>,
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
    #[cfg(feature = "cudarc")]
    pub fn stream(&self) -> Option<&Arc<CudaStream>> {
        self.cuda.as_ref().map(|c| &c.stream)
    }

    /// Build a **CUDA** letterbox preprocessor (unit-scale, pad 114) on `stream`.
    #[cfg(feature = "cudarc")]
    pub fn letterbox(stream: Arc<CudaStream>) -> Result<Self, PreprocessError> {
        PreprocessorBuilder::new()
            .mode(ResizeMode::Letterbox)
            .build_cuda(stream)
    }

    /// Build a **CUDA** stretch preprocessor (unit-scale) on `stream`.
    #[cfg(feature = "cudarc")]
    pub fn stretch(stream: Arc<CudaStream>) -> Result<Self, PreprocessError> {
        PreprocessorBuilder::new()
            .mode(ResizeMode::Stretch)
            .build_cuda(stream)
    }

    /// Build a **CUDA** preprocessor for an explicit [`ResizeMode`] (unit-scale, pad 114).
    #[cfg(feature = "cudarc")]
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
        if C != 3 && C != 4 {
            return Err(PreprocessError::UnsupportedChannels(C));
        }
        // Both paths write three channel planes of dst_h*dst_w — any other
        // leading dims would run past the buffer (an OOB device write on CUDA).
        if dst.shape[0] != 1 || dst.shape[1] != 3 {
            return Err(PreprocessError::BadOutputShape(dst.shape));
        }
        let (dst_h, dst_w) = (dst.shape[2], dst.shape[3]);
        let a = Affine::new(self.mode, src.width(), src.height(), dst_w, dst_h);

        #[cfg(feature = "cudarc")]
        if let Some(cuda) = &self.cuda {
            return self.run_cuda::<C>(cuda, src, dst, &a);
        }
        self.run_cpu::<C>(src, dst, &a)
    }

    // Host path: bilinear sample (at `src_bpp = C` stride) + pad + `(v/255-mean)*inv_std`
    // + HWC→CHW. Kept numerically identical to the CUDA kernel (`cpu_matches_cuda` test).
    fn run_cpu<const C: usize>(
        &self,
        src: &Image<u8, C>,
        dst: &mut Tensor<f32, 4>,
        a: &Affine,
    ) -> Result<(), PreprocessError> {
        // CPU preprocessor requires host-resident operands.
        #[cfg(feature = "cudarc")]
        if src.0.as_cudaslice().is_some() || dst.as_cudaslice().is_some() {
            return Err(PreprocessError::NotHostData);
        }

        let (dst_h, dst_w) = (dst.shape[2], dst.shape[3]);
        let (src_w, src_h) = (src.width(), src.height());
        let (mean, inv_std, pv) = (self.mean, self.inv_std, self.pad_value);
        let pixels = dst_h * dst_w;
        let src_pitch = src_w * C;
        let src_buf = src.0.as_slice();
        let dst_buf = dst.as_slice_mut();

        for oy in 0..dst_h {
            for ox in 0..dst_w {
                let i = oy * dst_w + ox;
                let sx = (ox as f32 - a.pad_x) / a.scale_x;
                let sy = (oy as f32 - a.pad_y) / a.scale_y;

                if sx < 0.0 || sy < 0.0 || sx >= src_w as f32 || sy >= src_h as f32 {
                    let pv01 = pv / 255.0;
                    for c in 0..3 {
                        dst_buf[c * pixels + i] = (pv01 - mean[c]) * inv_std[c];
                    }
                    continue;
                }

                let (x0f, y0f) = (sx.floor(), sy.floor());
                let (ax, ay) = (sx - x0f, sy - y0f);
                let (x0, y0) = (x0f as usize, y0f as usize);
                let x1 = (x0 + 1).min(src_w - 1);
                let y1 = (y0 + 1).min(src_h - 1);
                let (r0, r1) = (y0 * src_pitch, y1 * src_pitch);
                for c in 0..3 {
                    let v00 = src_buf[r0 + x0 * C + c] as f32;
                    let v10 = src_buf[r0 + x1 * C + c] as f32;
                    let v01 = src_buf[r1 + x0 * C + c] as f32;
                    let v11 = src_buf[r1 + x1 * C + c] as f32;
                    let top = v00 + (v10 - v00) * ax;
                    let bot = v01 + (v11 - v01) * ax;
                    let v = (top + (bot - top) * ay) / 255.0;
                    dst_buf[c * pixels + i] = (v - mean[c]) * inv_std[c];
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "cudarc")]
    fn run_cuda<const C: usize>(
        &self,
        cuda: &CudaBackend,
        src: &Image<u8, C>,
        dst: &mut Tensor<f32, 4>,
        a: &Affine,
    ) -> Result<(), PreprocessError> {
        let (dst_h, dst_w) = (dst.shape[2], dst.shape[3]);
        let (src_w, src_h) = (src.width(), src.height());
        // The kernel indexes pixels as `int`; keep every dim + the pixel count
        // within i32 so the `as i32` / `as u32` launch args never truncate.
        let lim = i32::MAX as usize;
        if src_w > lim || src_h > lim || dst_w.saturating_mul(dst_h) > lim {
            return Err(PreprocessError::DimensionsTooLarge);
        }
        let src_slice = src
            .0
            .as_cudaslice()
            .ok_or(PreprocessError::NotDeviceImage)?;
        let dst_slice = dst
            .as_cudaslice_mut()
            .ok_or(PreprocessError::NotDeviceTensor)?;

        let (sw, sh, sp, bpp) = (src_w as i32, src_h as i32, (src_w * C) as i32, C as i32);
        let (dw, dh) = (dst_w as i32, dst_h as i32);
        let total = (dst_w * dst_h) as u32;
        let ([m0, m1, m2], [is0, is1, is2]) = (self.mean, self.inv_std);
        let pv = self.pad_value;

        cuda.kernel
            .launch_builder(&cuda.stream)
            .arg(src_slice)
            .arg(dst_slice)
            .arg(&a.scale_x)
            .arg(&a.scale_y)
            .arg(&a.pad_x)
            .arg(&a.pad_y)
            .arg(&sw)
            .arg(&sh)
            .arg(&sp)
            .arg(&bpp)
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

    // CPU: a solid source stays solid; unit-scale gives v/255 everywhere.
    #[test]
    fn cpu_stretch_unit_scale_solid() {
        let pre = Preprocessor::builder()
            .mode(ResizeMode::Stretch)
            .build()
            .unwrap();
        let src = host_solid(5, 3, [10u8, 20, 30]);
        let mut dst = host_dst(4, 4);
        pre.run(&src, &mut dst).unwrap();
        let out = dst.as_slice();
        let px = 4 * 4;
        for (c, &v) in [10.0, 20.0, 30.0].iter().enumerate() {
            for i in 0..px {
                assert!((out[c * px + i] - v / 255.0).abs() < 1e-4, "chan {c}");
            }
        }
    }

    // CPU: RGBA (alpha ignored) matches tight RGB.
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
            assert!((x - y).abs() < 1e-6);
        }
    }

    // CPU: MeanStd folds the ImageNet normalize in.
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

    #[test]
    fn rejects_bad_channels() {
        let pre = Preprocessor::builder().build().unwrap();
        let mut dst = host_dst(2, 2);
        assert!(matches!(
            pre.run(&host_solid(2, 2, [5u8]), &mut dst),
            Err(PreprocessError::UnsupportedChannels(1))
        ));
    }

    // CPU and CUDA must agree (letterbox pad included). CUDA is the oracle check.
    #[cfg(feature = "cudarc")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cpu_matches_cuda() {
        use cudarc::driver::CudaContext;
        use kornia_tensor::zeros_cuda;

        let stream = CudaContext::new(0).unwrap().default_stream();
        let src = host_solid(7, 5, [30u8, 90, 200]);
        for mode in [ResizeMode::Letterbox, ResizeMode::Stretch] {
            let cpu = Preprocessor::builder()
                .mode(mode)
                .normalize(Normalize::imagenet())
                .build()
                .unwrap();
            let gpu = Preprocessor::builder()
                .mode(mode)
                .normalize(Normalize::imagenet())
                .build_cuda(stream.clone())
                .unwrap();

            let mut d_cpu = host_dst(6, 8);
            cpu.run(&src, &mut d_cpu).unwrap();

            let dev_src: Image<u8, 3> = Image(src.0.to_cuda(&stream).unwrap());
            let mut d_gpu = zeros_cuda::<f32, 4>([1, 3, 6, 8], &stream).unwrap();
            gpu.run(&dev_src, &mut d_gpu).unwrap();
            let gpu_host = stream.clone_dtoh(d_gpu.as_cudaslice().unwrap()).unwrap();

            for (a, b) in d_cpu.as_slice().iter().zip(&gpu_host) {
                assert!((a - b).abs() < 1e-4, "CPU/CUDA mismatch in {mode:?}");
            }
        }
    }
}
