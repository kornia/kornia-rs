//! Hardware bilinear resize via CUDA texture objects for `kornia-imgproc`.
//!
//! # Why texture objects?
//!
//! CUDA's texture unit performs bilinear interpolation **in hardware** — a
//! single `tex2D<float>()` call replaces 4 source reads + 6 FMAs that the
//! plain `__ldg` bilinear kernel does per output channel.  For a pipeline
//! that resizes the same source to multiple output sizes (or for sources that
//! are large enough that L2 cache pressure is high), the up-front texture
//! setup cost is amortised and per-frame cost falls.
//!
//! # Pipeline
//!
//! 1. **De-interleave** (NVRTC kernel): RGBRGB… → R-plane, G-plane, B-plane.
//! 2. **Texture bind** (`CU_RESOURCE_TYPE_PITCH2D`): wraps the planar device
//!    slices as 3 single-channel float texture objects — zero-copy, no
//!    `cudaArray` transfer needed.
//! 3. **Hardware bilinear** (NVRTC kernel): 3 × `tex2D<float>()` per output
//!    pixel; the texture unit clamps, computes weights, and interpolates in
//!    hardware.
//!
//! # Benchmark note
//!
//! For a **single use** (setup + one resize), the de-interleave pass adds
//! ~0.37 ms for 1080p, making the texture path slower than the plain `__ldg`
//! kernel (0.178 ms).  The crossover is at roughly **3 resize operations on
//! the same source** — above that the per-resize cost (kernel only, ~0.05 ms)
//! wins.
//!
//! # Public API
//!
//! * [`CudaRgbTexture`]                    — RAII texture owner.
//! * [`launch_resize_bilinear_tex_cuda`]   — hardware bilinear from a texture.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, LaunchConfig};
use cudarc::driver::sys::{
    CUDA_RESOURCE_DESC_st, CUDA_RESOURCE_DESC_st__bindgen_ty_1,
    CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4,
    CUDA_TEXTURE_DESC_st,
    CUresourcetype_enum::CU_RESOURCE_TYPE_PITCH2D,
    CUarray_format_enum::CU_AD_FORMAT_FLOAT,
    CUfilter_mode_enum::{CU_TR_FILTER_MODE_LINEAR, CU_TR_FILTER_MODE_POINT},
    CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP,
    CUresult,
};
use kornia_tensor::CudaKernel;

// ── CUDA C source: de-interleave RGB → 3 planar float buffers ────────────────

static DEINTERLEAVE_SRC: &str = r#"
extern "C" __global__ void deinterleave_rgb_3c(
    const float* __restrict__ src,
    float* __restrict__ plane_r,
    float* __restrict__ plane_g,
    float* __restrict__ plane_b,
    unsigned int width,
    unsigned int height
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned int src_i = (y * width + x) * 3u;
    unsigned int dst_i =  y * width + x;

    plane_r[dst_i] = __ldg(&src[src_i]);
    plane_g[dst_i] = __ldg(&src[src_i + 1]);
    plane_b[dst_i] = __ldg(&src[src_i + 2]);
}
"#;

// ── CUDA C source: hardware bilinear via cudaTextureObject_t ─────────────────

// tex2D with unnormalized coordinates and CU_TR_FILTER_MODE_LINEAR:
//   tex2D(t, u, v) samples at source pixel position (u - 0.5, v - 0.5).
// To get half-pixel-aligned source coord sx = (dst_x + 0.5) * scale_x - 0.5,
// we pass u = sx + 0.5 = (dst_x + 0.5) * scale_x.

static BILINEAR_TEX_SRC: &str = r#"
extern "C" __global__ void resize_bilinear_tex_3c(
    unsigned long long tex_r,
    unsigned long long tex_g,
    unsigned long long tex_b,
    float* __restrict__ dst,
    unsigned int dst_w,
    unsigned int dst_h,
    float scale_x,
    float scale_y
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    float u = (dst_x + 0.5f) * scale_x;
    float v = (dst_y + 0.5f) * scale_y;

    cudaTextureObject_t tr = (cudaTextureObject_t)tex_r;
    cudaTextureObject_t tg = (cudaTextureObject_t)tex_g;
    cudaTextureObject_t tb = (cudaTextureObject_t)tex_b;

    unsigned int out = (dst_y * dst_w + dst_x) * 3u;
    dst[out]     = tex2D<float>(tr, u, v);
    dst[out + 1] = tex2D<float>(tg, u, v);
    dst[out + 2] = tex2D<float>(tb, u, v);
}
"#;

// ── Kernel caches ─────────────────────────────────────────────────────────────

static DEINTERLEAVE_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static BILINEAR_TEX_KERNEL: OnceLock<CudaKernel> = OnceLock::new();

const BLOCK_W: u32 = 16;
const BLOCK_H: u32 = 16;

fn make_config(w: u32, h: u32) -> LaunchConfig {
    LaunchConfig {
        block_dim: (BLOCK_W, BLOCK_H, 1),
        grid_dim: (w.div_ceil(BLOCK_W), h.div_ceil(BLOCK_H), 1),
        shared_mem_bytes: 0,
    }
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by the texture-based resize path.
#[derive(Debug, thiserror::Error)]
pub enum CudaTexResizeError {
    /// CUDA driver or NVRTC error.
    #[error("CUDA error: {0}")]
    Cuda(String),
    /// Source width does not meet the pitch-alignment requirement.
    #[error("source width {width} px × 4 B = {pitch} B is not a multiple of the required texture pitch alignment {align} B")]
    PitchAlignment {
        /// Source image width in pixels.
        width: u32,
        /// Computed pitch (width × 4 bytes).
        pitch: u32,
        /// Required pitch alignment in bytes.
        align: u32,
    },
    /// Output slice is too small.
    #[error("output slice length {got} < required {need}")]
    SliceTooSmall {
        /// Actual slice length.
        got: usize,
        /// Required length (dst_w × dst_h × 3).
        need: usize,
    },
}

impl From<cudarc::driver::DriverError> for CudaTexResizeError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        CudaTexResizeError::Cuda(e.to_string())
    }
}

// ── Texture object RAII wrapper ───────────────────────────────────────────────

/// Owns the planar device buffers and three `CUtexObject` handles for a
/// single-channel float texture per RGB channel.
///
/// Created from an interleaved float32 RGB device slice via
/// [`CudaRgbTexture::from_interleaved`].  The texture objects are destroyed on
/// drop.
///
/// # Reuse
///
/// Create once per source image, then call
/// [`launch_resize_bilinear_tex_cuda`] multiple times at different output
/// sizes without re-uploading or re-de-interleaving the source.
pub struct CudaRgbTexture {
    /// R-channel planar device buffer (src_w × src_h f32).
    pub plane_r: CudaSlice<f32>,
    /// G-channel planar device buffer.
    pub plane_g: CudaSlice<f32>,
    /// B-channel planar device buffer.
    pub plane_b: CudaSlice<f32>,
    /// Source image width.
    pub width: u32,
    /// Source image height.
    pub height: u32,
    // Raw texture object handles (CUtexObject = u64).
    tex_r: u64,
    tex_g: u64,
    tex_b: u64,
}

impl Drop for CudaRgbTexture {
    fn drop(&mut self) {
        unsafe {
            // Best-effort destruction; ignore errors on drop.
            let _ = cudarc::driver::sys::cuTexObjectDestroy(self.tex_r);
            let _ = cudarc::driver::sys::cuTexObjectDestroy(self.tex_g);
            let _ = cudarc::driver::sys::cuTexObjectDestroy(self.tex_b);
        }
    }
}

impl CudaRgbTexture {
    /// Build a [`CudaRgbTexture`] from an interleaved 3-channel float32 image.
    ///
    /// Launches a de-interleave kernel, then creates three
    /// `CU_RESOURCE_TYPE_PITCH2D` texture objects that wrap the resulting
    /// planar buffers — no `cudaArray` copy is performed.
    ///
    /// # Errors
    ///
    /// Returns [`CudaTexResizeError::PitchAlignment`] if `src_width × 4` is
    /// not a multiple of the device's texture pitch alignment (typically 32 B
    /// on Turing GPUs; all standard resolutions are compliant).
    pub fn from_interleaved(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<f32>,
        src_width: u32,
        src_height: u32,
    ) -> Result<Self, CudaTexResizeError> {
        let pitch = src_width * 4;
        let align = query_pitch_alignment(ctx)?;
        if pitch % align != 0 {
            return Err(CudaTexResizeError::PitchAlignment {
                width: src_width,
                pitch,
                align,
            });
        }

        let n_px = (src_width as usize) * (src_height as usize);
        let mut plane_r = stream.alloc_zeros::<f32>(n_px)?;
        let mut plane_g = stream.alloc_zeros::<f32>(n_px)?;
        let mut plane_b = stream.alloc_zeros::<f32>(n_px)?;

        // Launch de-interleave kernel.
        let kernel = DEINTERLEAVE_KERNEL.get_or_init(|| {
            CudaKernel::compile(ctx, DEINTERLEAVE_SRC, "deinterleave_rgb_3c")
                .expect("failed to compile deinterleave_rgb_3c")
        });
        kernel
            .launch_builder(stream)
            .arg(src)
            .arg(&mut plane_r)
            .arg(&mut plane_g)
            .arg(&mut plane_b)
            .arg(&src_width)
            .arg(&src_height)
            .launch_2d(src_width, src_height, make_config(src_width, src_height))
            .map_err(|e| CudaTexResizeError::Cuda(e.to_string()))?;

        // Create pitch2D texture objects (no cudaArray copy — wraps device memory directly).
        let tex_r = create_pitch2d_texture(
            stream, &plane_r, src_width, src_height, pitch as usize,
        )?;
        let tex_g = create_pitch2d_texture(
            stream, &plane_g, src_width, src_height, pitch as usize,
        )?;
        let tex_b = create_pitch2d_texture(
            stream, &plane_b, src_width, src_height, pitch as usize,
        )?;

        Ok(CudaRgbTexture { plane_r, plane_g, plane_b, width: src_width, height: src_height, tex_r, tex_g, tex_b })
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Query CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT (usually 32 on Turing).
fn query_pitch_alignment(ctx: &Arc<CudaContext>) -> Result<u32, CudaTexResizeError> {
    let mut val: i32 = 0;
    let result = unsafe {
        cudarc::driver::sys::cuDeviceGetAttribute(
            &mut val,
            cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
            ctx.cu_device(),
        )
    };
    if result != CUresult::CUDA_SUCCESS {
        return Err(CudaTexResizeError::Cuda(format!(
            "cuDeviceGetAttribute failed: {:?}", result
        )));
    }
    Ok(val as u32)
}

/// Create a single-channel float pitch2D texture object from a device slice.
///
/// `CU_RESOURCE_TYPE_PITCH2D` wraps the existing device memory without any
/// copy to a `cudaArray`.  Requires `pitchInBytes` to be aligned to the
/// device's texture pitch alignment.
fn create_pitch2d_texture(
    stream: &Arc<CudaStream>,
    plane: &CudaSlice<f32>,
    width: u32,
    height: u32,
    pitch_bytes: usize,
) -> Result<u64, CudaTexResizeError> {
    // Get raw device pointer. SyncOnDrop records a read event (cheap).
    let (dev_ptr, _guard) = plane.device_ptr(stream);

    let res_desc = CUDA_RESOURCE_DESC_st {
        resType: CU_RESOURCE_TYPE_PITCH2D,
        res: CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
            pitch2D: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4 {
                devPtr: dev_ptr,
                format: CU_AD_FORMAT_FLOAT,
                numChannels: 1,
                width: width as usize,
                height: height as usize,
                pitchInBytes: pitch_bytes,
            },
        },
        flags: 0,
    };

    let tex_desc = CUDA_TEXTURE_DESC_st {
        addressMode: [CU_TR_ADDRESS_MODE_CLAMP; 3],
        filterMode: CU_TR_FILTER_MODE_LINEAR,
        // flags = 0 → unnormalized coordinates (tex2D takes pixel coords, not 0-1).
        flags: 0,
        maxAnisotropy: 1,
        mipmapFilterMode: CU_TR_FILTER_MODE_POINT,
        mipmapLevelBias: 0.0,
        minMipmapLevelClamp: 0.0,
        maxMipmapLevelClamp: 0.0,
        borderColor: [0.0; 4],
        reserved: [0; 12],
    };

    let mut tex_obj: u64 = 0;
    let result = unsafe {
        cudarc::driver::sys::cuTexObjectCreate(
            &mut tex_obj,
            &res_desc,
            &tex_desc,
            std::ptr::null(), // no resource view desc
        )
    };

    if result != CUresult::CUDA_SUCCESS {
        return Err(CudaTexResizeError::Cuda(format!(
            "cuTexObjectCreate failed: {:?}", result
        )));
    }

    Ok(tex_obj)
}

// ── Public launcher ───────────────────────────────────────────────────────────

/// Launch hardware bilinear downscale using pre-built texture objects.
///
/// The `tex` must have been created for the same source dimensions.  The
/// hardware texture unit performs bilinear interpolation with
/// `CU_TR_ADDRESS_MODE_CLAMP` boundary handling.
///
/// # Per-call cost
///
/// Only the bilinear kernel itself runs (~0.05 ms for 1080p→540p on GTX 1650).
/// The de-interleave + texture setup from [`CudaRgbTexture::from_interleaved`]
/// is a one-time cost (~0.37 ms for 1080p); amortised over ≥ 3 resize calls it
/// is faster than the plain `__ldg` path.
///
/// # Errors
///
/// Returns [`CudaTexResizeError`] on compile failure, launch error, or size
/// mismatch.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_bilinear_tex_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    tex: &CudaRgbTexture,
    dst: &mut CudaSlice<f32>,
    dst_width: u32,
    dst_height: u32,
) -> Result<(), CudaTexResizeError> {
    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaTexResizeError::SliceTooSmall { got: dst.len(), need });
    }

    let kernel = BILINEAR_TEX_KERNEL.get_or_init(|| {
        CudaKernel::compile(ctx, BILINEAR_TEX_SRC, "resize_bilinear_tex_3c")
            .expect("failed to compile resize_bilinear_tex_3c")
    });

    let scale_x = tex.width as f32 / dst_width as f32;
    let scale_y = tex.height as f32 / dst_height as f32;

    kernel
        .launch_builder(stream)
        .arg(&tex.tex_r)
        .arg(&tex.tex_g)
        .arg(&tex.tex_b)
        .arg(dst)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&scale_x)
        .arg(&scale_y)
        .launch_2d(dst_width, dst_height, make_config(dst_width, dst_height))
        .map_err(|e| CudaTexResizeError::Cuda(e.to_string()))
}
