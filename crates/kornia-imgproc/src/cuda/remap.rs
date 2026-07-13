//! Native CUDA remap kernel for `kornia-imgproc`.
//!
//! # Algorithm
//!
//! Remap is a **generic warp** primitive: the caller precomputes a pair of
//! float maps `(map_x, map_y)` — one f32 per output pixel — that give the
//! floating-point source coordinate for each destination pixel.  The kernel
//! reads those maps and samples the source image at the indicated location.
//!
//! This decouples coordinate generation from sampling:
//! * Affine warp   → map_x/y computed by `remap_maps_from_affine`
//! * Perspective   → map_x/y computed by `remap_maps_from_homography`
//! * Lens-undist.  → any arbitrary non-linear mapping
//!
//! Whether remap is fast enough to serve as the base for warp-perspective (vs
//! a fused inline-homography kernel) is determined by the benchmark in
//! `examples/bench_cuda_remap.rs`.
//!
//! # Optimisations
//!
//! * **CUDA texture objects** — source reads route through the dedicated
//!   2D-spatial texture cache; `CU_TR_ADDRESS_MODE_BORDER` returns 0 for any
//!   OOB fetch, eliminating divergent bounds-check branches at image edges.
//! * **`__ldg` for maps** — `map_x` / `map_y` are read through the L1 cache
//!   hint; consecutive threads in a warp access consecutive map entries
//!   (perfect coalescing).
//! * **`CU_FUNC_CACHE_PREFER_L1`** — enlarges L1 to 64 KB on Turing since
//!   neither kernel uses shared memory.
//! * **32×8 thread block (default)** — full warp per output row for coalesced
//!   destination writes; same rationale as resize and warp-affine.
//!
//! # Public API
//!
//! * [`launch_remap_bilinear_cuda`]    — bilinear remap, 3-ch f32.
//! * [`launch_remap_nearest_cuda`]     — nearest-neighbor remap, 3-ch f32.
//! * [`remap_maps_from_affine`]        — build map_x/map_y from a 2×3 affine matrix.
//! * [`remap_maps_from_homography`]    — build map_x/map_y from a 3×3 homography.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, LaunchConfig};
use kornia_tensor::CudaKernel;

use super::texture::CudaTexObject;

// ── CUDA C source: bilinear remap via texture object ─────────────────────────
//
// Structurally identical to warp_affine_bilinear_tex_3c, except the source
// coordinate (sx, sy) is read from device arrays rather than computed from a
// matrix.  The extra two __ldg reads per output pixel are the only overhead
// vs a fused affine kernel — measured in bench_cuda_remap.rs.

static BILINEAR_SRC: &str = r#"
extern "C" __global__ void remap_bilinear_3c(
    unsigned long long        tex,    // source as pitch-2D texture (width = src_w * 3)
    const float* __restrict__ map_x,  // dst_w * dst_h source-x coordinates
    const float* __restrict__ map_y,  // dst_w * dst_h source-y coordinates
    float* __restrict__       dst,
    unsigned int dst_w,
    unsigned int dst_h
) {
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= dst_w || gy >= dst_h) return;

    unsigned int idx = gy * dst_w + gx;
    float sx = __ldg(&map_x[idx]);
    float sy = __ldg(&map_y[idx]);

    // Bilinear weights — matches warp_affine_bilinear_tex_3c.
    float x0f = floorf(sx);
    float y0f = floorf(sy);
    float fx   = sx - x0f;
    float fy   = sy - y0f;
    float w00  = (1.0f - fy) * (1.0f - fx);
    float w10  = (1.0f - fy) * fx;
    float w01  = fy * (1.0f - fx);
    float w11  = fy * fx;

    float tx0 = x0f * 3.0f;
    float ty1 = y0f + 1.0f;

    // Border mode: OOB fetches return 0.0 — no explicit bounds check needed.
    cudaTextureObject_t t = (cudaTextureObject_t)tex;
    unsigned int out = idx * 3u;

    dst[out]   = fmaf(w00, tex2D<float>(t, tx0,       y0f),
                fmaf(w10, tex2D<float>(t, tx0+3.0f,  y0f),
                fmaf(w01, tex2D<float>(t, tx0,       ty1),
                     w11 * tex2D<float>(t, tx0+3.0f,  ty1))));
    dst[out+1] = fmaf(w00, tex2D<float>(t, tx0+1.0f,  y0f),
                fmaf(w10, tex2D<float>(t, tx0+4.0f,  y0f),
                fmaf(w01, tex2D<float>(t, tx0+1.0f,  ty1),
                     w11 * tex2D<float>(t, tx0+4.0f,  ty1))));
    dst[out+2] = fmaf(w00, tex2D<float>(t, tx0+2.0f,  y0f),
                fmaf(w10, tex2D<float>(t, tx0+5.0f,  y0f),
                fmaf(w01, tex2D<float>(t, tx0+2.0f,  ty1),
                     w11 * tex2D<float>(t, tx0+5.0f,  ty1))));
}
"#;

// ── CUDA C source: nearest-neighbor remap via texture object ─────────────────

static NEAREST_SRC: &str = r#"
extern "C" __global__ void remap_nearest_3c(
    unsigned long long        tex,
    const float* __restrict__ map_x,
    const float* __restrict__ map_y,
    float* __restrict__       dst,
    unsigned int dst_w,
    unsigned int dst_h
) {
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= dst_w || gy >= dst_h) return;

    unsigned int idx = gy * dst_w + gx;
    float sx = __ldg(&map_x[idx]);
    float sy = __ldg(&map_y[idx]);

    // Round to nearest pixel; border mode handles OOB → 0.
    float tx = roundf(sx) * 3.0f;
    float ty = roundf(sy);

    cudaTextureObject_t t = (cudaTextureObject_t)tex;
    unsigned int out = idx * 3u;
    dst[out]   = tex2D<float>(t, tx,       ty);
    dst[out+1] = tex2D<float>(t, tx+1.0f,  ty);
    dst[out+2] = tex2D<float>(t, tx+2.0f,  ty);
}
"#;

// ── Kernel cache ──────────────────────────────────────────────────────────────

static BILINEAR_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static NEAREST_KERNEL: OnceLock<CudaKernel> = OnceLock::new();

const BLOCK_W: u32 = 32;
const BLOCK_H: u32 = 8;

// ── Internal helpers ──────────────────────────────────────────────────────────

fn make_config(dst_width: u32, dst_height: u32, block_dim: Option<(u32, u32)>) -> LaunchConfig {
    let (bw, bh) = block_dim.unwrap_or_else(|| (BLOCK_W.min(dst_width), BLOCK_H.min(dst_height)));
    LaunchConfig {
        block_dim: (bw, bh, 1),
        grid_dim: (dst_width.div_ceil(bw), dst_height.div_ceil(bh), 1),
        shared_mem_bytes: 0,
    }
}

fn compile_with_l1(ctx: &Arc<CudaContext>, src: &str, fn_name: &str) -> CudaKernel {
    let k = CudaKernel::compile(ctx, src, fn_name)
        .unwrap_or_else(|e| panic!("failed to compile {fn_name}: {e}"));
    let _ = k.prefer_l1_cache();
    k
}

fn make_tex(
    src: &CudaSlice<f32>,
    stream: &Arc<CudaStream>,
    src_width: u32,
    src_height: u32,
) -> Result<(CudaTexObject, u64), CudaRemapError> {
    // Safety: DevicePtr::device_ptr returns the raw CUdeviceptr; the _guard
    // records a stream event on drop (after the texture is destroyed) which is
    // the correct ordering.
    let (dev_ptr, _guard) = src.device_ptr(stream);
    let tex =
        CudaTexObject::new_pitch2d_border(dev_ptr, src_width as usize * 3, src_height as usize)
            .map_err(CudaRemapError::Cuda)?;
    let handle = tex.handle();
    Ok((tex, handle))
}

fn check_image_slice(
    slice: &CudaSlice<f32>,
    what: &'static str,
    width: u32,
    height: u32,
) -> Result<(), CudaRemapError> {
    let need = (width as usize) * (height as usize) * 3;
    if slice.len() < need {
        return Err(CudaRemapError::SliceTooSmall {
            what,
            got: slice.len(),
            need,
        });
    }
    Ok(())
}

fn check_map(
    map: &CudaSlice<f32>,
    what: &'static str,
    dst_width: u32,
    dst_height: u32,
) -> Result<(), CudaRemapError> {
    let need = (dst_width as usize) * (dst_height as usize);
    if map.len() < need {
        return Err(CudaRemapError::SliceTooSmall {
            what,
            got: map.len(),
            need,
        });
    }
    Ok(())
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by the CUDA remap launchers.
#[derive(Debug, thiserror::Error)]
pub enum CudaRemapError {
    /// CUDA driver / launch / texture error.
    #[error("CUDA remap error: {0}")]
    Cuda(String),
    /// A device slice is smaller than required.
    #[error("device slice '{what}' length {got} < required {need}")]
    SliceTooSmall {
        /// Which operand was too small (e.g. `"src"`, `"dst"`, `"map_x"`, `"map_y"`).
        what: &'static str,
        /// Actual slice length (in elements).
        got: usize,
        /// Minimum required length.
        need: usize,
    },
}

// ── Map-generation helpers ────────────────────────────────────────────────────

/// Build a `(map_x, map_y)` coordinate map on the CPU from a 2×3 affine matrix.
///
/// The map encodes the **inverse** transform (output → source), matching
/// OpenCV's `cv::remap` convention.  Pass this map to [`launch_remap_bilinear_cuda`]
/// to reproduce the same result as [`launch_warp_affine_bilinear_cuda`].
///
/// `m` is the **forward** 2×3 affine matrix (source → destination).  It is
/// inverted internally so callers can pass the same matrix they would pass to
/// `launch_warp_affine_*`.
///
/// Out-of-bounds coordinates are left as-is; the texture sampler returns 0
/// for them (`BORDER_CONSTANT = 0`).
///
/// The returned vecs have `dst_w * dst_h` elements each, in row-major order.
pub fn remap_maps_from_affine(
    m: &[f32; 6],
    dst_width: u32,
    dst_height: u32,
) -> (Vec<f32>, Vec<f32>) {
    use crate::warp::invert_affine_transform;
    let mi = invert_affine_transform(m);
    let npix = (dst_width * dst_height) as usize;
    let mut mx = Vec::with_capacity(npix);
    let mut my = Vec::with_capacity(npix);
    for dy in 0..dst_height {
        let base_sx = mi[1] * dy as f32 + mi[2];
        let base_sy = mi[4] * dy as f32 + mi[5];
        for dx in 0..dst_width {
            mx.push(mi[0] * dx as f32 + base_sx);
            my.push(mi[3] * dx as f32 + base_sy);
        }
    }
    (mx, my)
}

/// Build a `(map_x, map_y)` coordinate map on the CPU from a 3×3 homography matrix.
///
/// `h` is a row-major 3×3 homography (forward: source → destination).
/// The map encodes the **inverse** perspective transform (destination → source),
/// so that remap produces the same result as a fused warp-perspective kernel.
///
/// The returned vecs have `dst_w * dst_h` elements each, in row-major order.
/// Pixels whose homogeneous w-coordinate is zero or negative (i.e. `w < 1e-10`)
/// are mapped to `(-1.0, -1.0)`, which falls outside the source and returns 0
/// via border mode.  This culls back-projected points (negative w) as well as
/// near-degenerate projections.
pub fn remap_maps_from_homography(
    h: &[f32; 9],
    dst_width: u32,
    dst_height: u32,
) -> (Vec<f32>, Vec<f32>) {
    // Invert the 3×3 homography using the adjugate / determinant formula.
    let (h0, h1, h2) = (h[0], h[1], h[2]);
    let (h3, h4, h5) = (h[3], h[4], h[5]);
    let (h6, h7, h8) = (h[6], h[7], h[8]);

    let det = h0 * (h4 * h8 - h5 * h7) - h1 * (h3 * h8 - h5 * h6) + h2 * (h3 * h7 - h4 * h6);

    let inv_det = if det.abs() < 1e-10 { 0.0 } else { 1.0 / det };

    // Row-major inverse H.
    let i = [
        (h4 * h8 - h5 * h7) * inv_det,
        (h2 * h7 - h1 * h8) * inv_det,
        (h1 * h5 - h2 * h4) * inv_det,
        (h5 * h6 - h3 * h8) * inv_det,
        (h0 * h8 - h2 * h6) * inv_det,
        (h2 * h3 - h0 * h5) * inv_det,
        (h3 * h7 - h4 * h6) * inv_det,
        (h1 * h6 - h0 * h7) * inv_det,
        (h0 * h4 - h1 * h3) * inv_det,
    ];

    let npix = (dst_width * dst_height) as usize;
    let mut mx = Vec::with_capacity(npix);
    let mut my = Vec::with_capacity(npix);
    for dy in 0..dst_height {
        let y = dy as f32;
        let row_sx = i[1] * y + i[2];
        let row_sy = i[4] * y + i[5];
        let row_w = i[7] * y + i[8];
        for dx in 0..dst_width {
            let x = dx as f32;
            let w = i[6] * x + row_w;
            if w < 1e-10 {
                mx.push(-1.0);
                my.push(-1.0);
            } else {
                let inv_w = 1.0 / w;
                mx.push((i[0] * x + row_sx) * inv_w);
                my.push((i[3] * x + row_sy) * inv_w);
            }
        }
    }
    (mx, my)
}

// ── Private launcher core ─────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn launch_remap(
    kernel_cell: &OnceLock<CudaKernel>,
    kernel_src: &'static str,
    fn_name: &'static str,
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    map_x: &CudaSlice<f32>,
    map_y: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaRemapError> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(CudaRemapError::Cuda(
            "image dimensions must be non-zero".into(),
        ));
    }
    check_image_slice(src, "src", src_width, src_height)?;
    check_image_slice(dst, "dst", dst_width, dst_height)?;
    check_map(map_x, "map_x", dst_width, dst_height)?;
    check_map(map_y, "map_y", dst_width, dst_height)?;

    let kernel = kernel_cell.get_or_init(|| compile_with_l1(ctx, kernel_src, fn_name));
    let (_tex, tex_handle) = make_tex(src, stream, src_width, src_height)?;

    kernel
        .launch_builder(stream)
        .arg(&tex_handle)
        .arg(map_x)
        .arg(map_y)
        .arg(dst)
        .arg(&dst_width)
        .arg(&dst_height)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaRemapError::Cuda(e.to_string()))
}

// ── Public launchers ──────────────────────────────────────────────────────────

/// Launch the bilinear remap kernel for a 3-channel f32 image.
///
/// Each output pixel at `(gx, gy)` samples `src` at
/// `(map_x[gy*dst_w+gx], map_y[gy*dst_w+gx])` using bilinear interpolation.
/// Source coordinates outside `[0, src_w) × [0, src_h)` produce 0 output
/// (`BORDER_CONSTANT = 0`), matching the OpenCV default.
///
/// # Arguments
///
/// * `ctx`       — CUDA context (used for kernel compilation on first call).
/// * `stream`    — CUDA stream for launch and memory operations.
/// * `src`       — Source image, `src_w * src_h * 3` f32 elements, row-major.
/// * `map_x`     — Source x-coordinate per output pixel, `dst_w * dst_h` f32.
/// * `map_y`     — Source y-coordinate per output pixel, `dst_w * dst_h` f32.
/// * `dst`       — Destination buffer, at least `dst_w * dst_h * 3` f32.
/// * `src_width` / `src_height` — Source image dimensions.
/// * `dst_width` / `dst_height` — Destination image dimensions.
/// * `block_dim` — Optional `(bw, bh)` thread-block override; `None` → 32×8.
///
/// # Errors
///
/// Returns [`CudaRemapError`] on compile failure, texture-creation error,
/// launch error, or if any slice is too small.
#[allow(clippy::too_many_arguments)]
pub fn launch_remap_bilinear_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    map_x: &CudaSlice<f32>,
    map_y: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaRemapError> {
    launch_remap(
        &BILINEAR_KERNEL,
        BILINEAR_SRC,
        "remap_bilinear_3c",
        ctx,
        stream,
        src,
        map_x,
        map_y,
        dst,
        src_width,
        src_height,
        dst_width,
        dst_height,
        block_dim,
    )
}

/// Launch the nearest-neighbor remap kernel for a 3-channel f32 image.
///
/// Same as [`launch_remap_bilinear_cuda`] but uses round-to-nearest source
/// sampling.  Faster; suitable when the map was computed at integer precision
/// or when speed matters more than visual quality.
///
/// # Arguments
///
/// See [`launch_remap_bilinear_cuda`] — arguments are identical.
///
/// # Errors
///
/// Returns [`CudaRemapError`] on compile failure, texture-creation error,
/// launch error, or if any slice is too small.
#[allow(clippy::too_many_arguments)]
pub fn launch_remap_nearest_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    map_x: &CudaSlice<f32>,
    map_y: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaRemapError> {
    launch_remap(
        &NEAREST_KERNEL,
        NEAREST_SRC,
        "remap_nearest_3c",
        ctx,
        stream,
        src,
        map_x,
        map_y,
        dst,
        src_width,
        src_height,
        dst_width,
        dst_height,
        block_dim,
    )
}
