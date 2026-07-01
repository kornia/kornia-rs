//! GPU image → model-input preprocessing: resize (+ optional pad) and normalize a
//! device-resident [`Image`] straight into a CHW `f32` [`Tensor`] in one CUDA kernel.
//!
//! This is the step every CNN / transformer vision model needs before inference:
//! take a camera frame of arbitrary size and produce the fixed `[1, 3, H, W]`
//! CHW `f32` tensor the network expects. It runs entirely on the GPU (kornia's
//! [`CudaKernel`]) with no host round-trip — the source image already lives on the
//! device (`image.to_cuda(&stream)`), and the output tensor is written in place so
//! it can be reused frame to frame.
//!
//! The output layout, resize fit, and normalization are configurable via
//! [`PreprocessorBuilder`]:
//! - **resize** — [`ResizeMode::Letterbox`] (aspect-preserving + pad) or
//!   [`ResizeMode::Stretch`] (anisotropic, no pad).
//! - **normalize** — [`Normalize::UnitScale`] (RGB `[0,1]`) or
//!   [`Normalize::MeanStd`] (fold an ImageNet-style mean/std into the same pass).
//! - **channels** — [`run`](Preprocessor::run) is generic over the source channel
//!   count: `Image<u8, 3>` (RGB) or `Image<u8, 4>` (RGBA — the alpha byte is
//!   skipped), so a pitched RGBA camera surface repacked into a tight `Image<u8, 4>`
//!   goes to a CHW model tensor in a single fused kernel (no separate RGBA→RGB pass).
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
//! use kornia_imgproc::preprocess::{Preprocessor, Normalize, ResizeMode};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let stream = CudaContext::new(0)?.default_stream();
//!
//! // Build once (compiles the kernel); reuse across frames.
//! let pre = Preprocessor::builder()
//!     .mode(ResizeMode::Stretch)
//!     .normalize(Normalize::imagenet())
//!     .build(stream.clone())?;
//!
//! // Model input buffer, allocated once and reused.
//! let mut input = zeros_cuda::<f32, 4>([1, 3, 640, 640], &stream)?;
//!
//! // Per frame: upload the camera image and preprocess into `input`.
//! # let host: Image<u8, 3> = Image::from_size_val([1280, 720].into(), 0)?;
//! let frame = Image(host.0.to_cuda(&stream)?);   // device-resident Image<u8, 3>
//! pre.run(&frame, &mut input)?;                  // input is now [1,3,640,640], ImageNet-normalized
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use cudarc::driver::CudaStream;
use kornia_image::Image;
use kornia_tensor::{CudaError, CudaKernel, Tensor};

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
    /// identity affine (mean 0, inv_std 1) so a single kernel handles both.
    fn mean_inv_std(self) -> ([f32; 3], [f32; 3]) {
        match self {
            Normalize::UnitScale => ([0.0; 3], [1.0; 3]),
            Normalize::MeanStd { mean, std } => (mean, [1.0 / std[0], 1.0 / std[1], 1.0 / std[2]]),
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
    /// The source channel count is not 3 (RGB) or 4 (RGBA).
    #[error("unsupported source channel count {0} (expected 3 or 4)")]
    UnsupportedChannels(usize),
}

// 1-D grid over the `dst_w * dst_h` output pixels. The (ox, oy) pair is recovered
// from the flat index — consecutive threads map to consecutive `ox`, so global
// writes stay coalesced. Reads interleaved `src_bpp`-byte pixels (3 = RGB, 4 =
// RGBA with the alpha byte skipped) from linear device memory at an arbitrary row
// pitch with a hand-rolled bilinear sample (CUDA textures only support 1/2/4-channel
// elements, not 3), applies `(v/255 - mean) * inv_std` per channel, and writes
// channel-planar (CHW) `f32`. One parameterized kernel covers every mode / format /
// normalization — the config is passed as launch args, so there is a single JIT
// compile and no branch divergence beyond the shared out-of-bounds pad test.
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

    // Bilinear with clamp-to-edge (texel-center convention).
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
/// value, then [`build`](PreprocessorBuilder::build) on a stream.
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

    /// Compile the kernel for `stream`'s device and build the [`Preprocessor`].
    pub fn build(self, stream: Arc<CudaStream>) -> Result<Preprocessor, PreprocessError> {
        let kernel = CudaKernel::compile(stream.context(), KERNEL_SRC, "resize_normalize_to_chw")?;
        let (mean, inv_std) = self.normalize.mean_inv_std();
        Ok(Preprocessor {
            kernel,
            stream,
            mode: self.mode,
            mean,
            inv_std,
            pad_value: self.pad_value as f32,
        })
    }
}

/// GPU image-to-tensor preprocessor: resize (+ optional pad) + normalize.
///
/// Built once (compiles the CUDA kernel for the device behind its stream), then
/// applied to any number of frames of any resolution via [`run`](Self::run). It
/// owns only the JIT-compiled kernel, its stream, and the config — no per-frame
/// buffers — so a single instance handles every input size and any model size (the
/// target H/W is read from the destination tensor each call).
///
/// Configure via [`Preprocessor::builder`]; [`letterbox`](Self::letterbox) /
/// [`stretch`](Self::stretch) are convenience constructors (unit-scale, pad 114).
pub struct Preprocessor {
    kernel: CudaKernel,
    stream: Arc<CudaStream>,
    mode: ResizeMode,
    mean: [f32; 3],
    inv_std: [f32; 3],
    pad_value: f32,
}

impl Preprocessor {
    /// Start a [`PreprocessorBuilder`].
    pub fn builder() -> PreprocessorBuilder {
        PreprocessorBuilder::new()
    }

    /// Build a preprocessor that **letterboxes** (aspect-preserving resize + grey
    /// 114 pad, unit-scale) on `stream`.
    pub fn letterbox(stream: Arc<CudaStream>) -> Result<Self, PreprocessError> {
        Self::with_mode(stream, ResizeMode::Letterbox)
    }

    /// Build a preprocessor that **stretches** (anisotropic resize, no pad,
    /// unit-scale) on `stream` — for models trained on a square stretch (e.g. RF-DETR).
    pub fn stretch(stream: Arc<CudaStream>) -> Result<Self, PreprocessError> {
        Self::with_mode(stream, ResizeMode::Stretch)
    }

    /// Build a preprocessor for an explicit [`ResizeMode`] (unit-scale, pad 114).
    /// For a mean/std normalization or a custom pad, use [`Preprocessor::builder`].
    pub fn with_mode(stream: Arc<CudaStream>, mode: ResizeMode) -> Result<Self, PreprocessError> {
        PreprocessorBuilder::new().mode(mode).build(stream)
    }

    /// The CUDA stream this preprocessor launches on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The resize mode this preprocessor applies.
    pub fn mode(&self) -> ResizeMode {
        self.mode
    }

    /// Resize + normalize `src` into `dst` (`[1, 3, H, W]` CHW `f32`).
    ///
    /// Generic over the source channel count: `C = 3` (RGB) or `C = 4` (RGBA — the
    /// alpha byte is skipped). The target H/W is read from `dst`'s shape, so one
    /// preprocessor handles any model size. Both operands must be device-resident
    /// (the image via `to_cuda`, the tensor via `zeros_cuda`).
    ///
    /// # Errors
    ///
    /// [`PreprocessError::UnsupportedChannels`] if `C` is not 3 or 4;
    /// [`PreprocessError::NotDeviceImage`] / [`PreprocessError::NotDeviceTensor`] if
    /// either operand is host-resident; [`PreprocessError::Cuda`] on a launch failure.
    pub fn run<const C: usize>(
        &self,
        src: &Image<u8, C>,
        dst: &mut Tensor<f32, 4>,
    ) -> Result<(), PreprocessError> {
        if C != 3 && C != 4 {
            return Err(PreprocessError::UnsupportedChannels(C));
        }
        let (dst_h, dst_w) = (dst.shape[2], dst.shape[3]);
        let (src_w, src_h) = (src.width(), src.height());

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

        let src_slice = src.as_cudaslice().ok_or(PreprocessError::NotDeviceImage)?;
        let dst_slice = dst
            .as_cudaslice_mut()
            .ok_or(PreprocessError::NotDeviceTensor)?;

        // Tight interleaved source: row pitch = width * channels, bytes-per-pixel = C.
        let (sw, sh, sp, bpp) = (src_w as i32, src_h as i32, (src_w * C) as i32, C as i32);
        let (dw, dh) = (dst_w as i32, dst_h as i32);
        let total = (dst_w * dst_h) as u32;
        let ([m0, m1, m2], [is0, is1, is2]) = (self.mean, self.inv_std);
        let pv = self.pad_value;

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

#[cfg(all(test, feature = "cudarc"))]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::zeros_cuda;

    // A solid `w×h` device image where every pixel is `px` (per-channel bytes).
    fn solid<const C: usize>(
        stream: &Arc<CudaStream>,
        w: usize,
        h: usize,
        px: [u8; C],
    ) -> Image<u8, C> {
        let data: Vec<u8> = (0..w * h).flat_map(|_| px).collect();
        let host = Image::<u8, C>::new(
            ImageSize {
                width: w,
                height: h,
            },
            data,
        )
        .unwrap();
        Image(host.0.to_cuda(stream).unwrap())
    }

    fn read(stream: &Arc<CudaStream>, dst: &Tensor<f32, 4>) -> Vec<f32> {
        stream.clone_dtoh(dst.as_cudaslice().unwrap()).unwrap()
    }

    // A solid source stays solid through resize; every output value equals the
    // per-channel constant under unit-scale (v/255). Also exercises Stretch.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn stretch_unit_scale_solid() {
        let stream = CudaContext::new(0).unwrap().default_stream();
        let pre = Preprocessor::stretch(stream.clone()).unwrap();
        let src = solid(&stream, 5, 3, [10u8, 20, 30]);
        let mut dst = zeros_cuda::<f32, 4>([1, 3, 4, 4], &stream).unwrap();
        pre.run(&src, &mut dst).unwrap();
        let out = read(&stream, &dst);
        let px = 4 * 4;
        for (c, &v) in [10.0, 20.0, 30.0].iter().enumerate() {
            for i in 0..px {
                assert!((out[c * px + i] - v / 255.0).abs() < 1e-4, "chan {c}");
            }
        }
    }

    // RGBA (alpha ignored) must match tight RGB with the same colour.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn rgba_matches_rgb() {
        let stream = CudaContext::new(0).unwrap().default_stream();
        let pre = Preprocessor::stretch(stream.clone()).unwrap();
        let rgb = solid(&stream, 6, 4, [40u8, 80, 120]);
        let rgba = solid(&stream, 6, 4, [40u8, 80, 120, 200]);
        let mut d3 = zeros_cuda::<f32, 4>([1, 3, 8, 8], &stream).unwrap();
        let mut d4 = zeros_cuda::<f32, 4>([1, 3, 8, 8], &stream).unwrap();
        pre.run(&rgb, &mut d3).unwrap();
        pre.run(&rgba, &mut d4).unwrap();
        let (a, b) = (read(&stream, &d3), read(&stream, &d4));
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "RGBA must equal RGB (alpha ignored)");
        }
    }

    // MeanStd folds the ImageNet normalize into the same pass.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn imagenet_normalize() {
        let stream = CudaContext::new(0).unwrap().default_stream();
        let pre = Preprocessor::builder()
            .mode(ResizeMode::Stretch)
            .normalize(Normalize::imagenet())
            .build(stream.clone())
            .unwrap();
        let src = solid(&stream, 4, 4, [128u8, 128, 128]);
        let mut dst = zeros_cuda::<f32, 4>([1, 3, 4, 4], &stream).unwrap();
        pre.run(&src, &mut dst).unwrap();
        let out = read(&stream, &dst);
        let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
        let px = 4 * 4;
        for c in 0..3 {
            let want = (128.0 / 255.0 - mean[c]) / std[c];
            assert!((out[c * px] - want).abs() < 1e-4, "chan {c}");
        }
    }

    // Channel counts other than 3/4 are rejected before any launch.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn rejects_bad_channels() {
        let stream = CudaContext::new(0).unwrap().default_stream();
        let pre = Preprocessor::stretch(stream.clone()).unwrap();
        let src = solid(&stream, 2, 2, [5u8]); // C = 1
        let mut dst = zeros_cuda::<f32, 4>([1, 3, 2, 2], &stream).unwrap();
        assert!(matches!(
            pre.run(&src, &mut dst),
            Err(PreprocessError::UnsupportedChannels(1))
        ));
    }
}
