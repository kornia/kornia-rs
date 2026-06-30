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

use cudarc::driver::CudaStream;
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
}

// 1-D grid over the `dst_w * dst_h` output pixels. The (ox, oy) pair is recovered
// from the flat index — consecutive threads map to consecutive `ox`, so global
// writes stay coalesced. Reads interleaved 3-byte RGB from linear device memory
// with a hand-rolled bilinear sample (CUDA textures only support 1/2/4-channel
// elements, not 3) and writes channel-planar (CHW) `f32` in `[0, 1]`.
const KERNEL_SRC: &str = r#"
extern "C" __global__ void resize_pad_to_chw_rgb8(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    float scale_x, float scale_y, float pad_x, float pad_y,
    int src_w, int src_h, int src_pitch,
    int dst_w, int dst_h
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = dst_w * dst_h;
    if (i >= pixels) return;
    int ox = i % dst_w;
    int oy = i / dst_w;

    float sx = ((float)ox - pad_x) / scale_x;
    float sy = ((float)oy - pad_y) / scale_y;

    if (sx < 0.0f || sy < 0.0f || sx >= (float)src_w || sy >= (float)src_h) {
        float g = 114.0f / 255.0f;
        dst[i] = g; dst[pixels + i] = g; dst[2*pixels + i] = g;
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
        float v00 = (float)r0[x0 * 3 + c], v10 = (float)r0[x1 * 3 + c];
        float v01 = (float)r1[x0 * 3 + c], v11 = (float)r1[x1 * 3 + c];
        float top = v00 + (v10 - v00) * ax;
        float bot = v01 + (v11 - v01) * ax;
        dst[c * pixels + i] = (top + (bot - top) * ay) / 255.0f;
    }
}
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

    /// Build a preprocessor for an explicit [`ResizeMode`] on `stream`.
    pub fn with_mode(stream: Arc<CudaStream>, mode: ResizeMode) -> Result<Self, PreprocessError> {
        let ctx = stream.context();
        let kernel = CudaKernel::compile(ctx, KERNEL_SRC, "resize_pad_to_chw_rgb8")?;
        Ok(Self {
            kernel,
            stream,
            mode,
        })
    }

    /// The CUDA stream this preprocessor launches on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The resize mode this preprocessor applies.
    pub fn mode(&self) -> ResizeMode {
        self.mode
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

        let (sw, sh, sp) = (src_w as i32, src_h as i32, (src_w * 3) as i32);
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
            .launch_1d(total)?;
        Ok(())
    }
}
