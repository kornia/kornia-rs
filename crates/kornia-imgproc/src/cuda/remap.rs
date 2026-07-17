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
//! * **`__ldg` source reads** — all kernels read the source through the L1
//!   read-only cache path via `__ldg`.  Unlike pitch-2D texture objects,
//!   `__ldg` works at any image width (no pitch-alignment constraint) and
//!   avoids the per-call `cuTexObjectCreate` overhead.
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

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::{check_geometry, make_config, try_compile_with_l1};

// ── CUDA C source: bilinear remap via __ldg ───────────────────────────────────
//
// Reads source through __ldg on a raw pointer (no texture object, no
// pitch-alignment constraint). OOB source coordinates (from the map) produce
// BORDER_CONSTANT = 0 via an explicit bounds check before sampling.

static BILINEAR_SRC: &str = r#"
extern "C" __global__ void remap_bilinear_3c(
    const float* __restrict__ src,
    const float* __restrict__ map_x,
    const float* __restrict__ map_y,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h
) {
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= dst_w || gy >= dst_h) return;

    unsigned long long idx = (unsigned long long)gy * dst_w + gx;
    float sx = __ldg(&map_x[idx]);
    float sy = __ldg(&map_y[idx]);
    unsigned long long out = idx * 3ull;

    // BORDER_CONSTANT = 0 for any OOB source coordinate.
    if (sx < 0.0f || sx >= (float)src_w || sy < 0.0f || sy >= (float)src_h) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }

    float sxc = fmaxf(fminf(sx, (float)(src_w - 1u)), 0.0f);
    float syc = fmaxf(fminf(sy, (float)(src_h - 1u)), 0.0f);

    unsigned int x0 = (unsigned int)sxc;
    unsigned int y0 = (unsigned int)syc;
    unsigned int x1 = min(x0 + 1u, src_w - 1u);
    unsigned int y1 = min(y0 + 1u, src_h - 1u);
    float fx = sxc - (float)x0;
    float fy = syc - (float)y0;

    float fxx = 1.0f - fx;
    float fyy = 1.0f - fy;
    float w00 = fyy * fxx;
    float w10 = fyy * fx;
    float w01 = fy  * fxx;
    float w11 = fy  * fx;

    unsigned long long r0 = (unsigned long long)y0 * src_w;
    unsigned long long r1 = (unsigned long long)y1 * src_w;
    unsigned long long b00 = (r0 + x0) * 3ull;
    unsigned long long b10 = (r0 + x1) * 3ull;
    unsigned long long b01 = (r1 + x0) * 3ull;
    unsigned long long b11 = (r1 + x1) * 3ull;

    #pragma unroll
    for (unsigned int c = 0u; c < 3u; ++c) {
        float v00 = __ldg(&src[b00 + c]);
        float v10 = __ldg(&src[b10 + c]);
        float v01 = __ldg(&src[b01 + c]);
        float v11 = __ldg(&src[b11 + c]);
        dst[out + c] = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
    }
}
"#;

// ── CUDA C source: nearest-neighbor remap via __ldg ───────────────────────────

static NEAREST_SRC: &str = r#"
extern "C" __global__ void remap_nearest_3c(
    const float* __restrict__ src,
    const float* __restrict__ map_x,
    const float* __restrict__ map_y,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h
) {
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= dst_w || gy >= dst_h) return;

    unsigned long long idx = (unsigned long long)gy * dst_w + gx;
    float sx = __ldg(&map_x[idx]);
    float sy = __ldg(&map_y[idx]);
    unsigned long long out = idx * 3ull;

    // BORDER_CONSTANT = 0 for any OOB source coordinate.
    if (sx < 0.0f || sx >= (float)src_w || sy < 0.0f || sy >= (float)src_h) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }

    unsigned int xi = min((unsigned int)roundf(sx), src_w - 1u);
    unsigned int yi = min((unsigned int)roundf(sy), src_h - 1u);

    unsigned long long b = ((unsigned long long)yi * src_w + xi) * 3ull;
    dst[out]   = __ldg(&src[b]);
    dst[out+1] = __ldg(&src[b+1]);
    dst[out+2] = __ldg(&src[b+2]);
}
"#;

// ── Kernel cache ──────────────────────────────────────────────────────────────

static BILINEAR_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static NEAREST_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

// ── Internal helpers ──────────────────────────────────────────────────────────

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
    /// CUDA driver / launch error.
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
/// Out-of-bounds coordinates are left as-is; the kernel returns 0 for them
/// (`BORDER_CONSTANT = 0`).
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
    let (h0, h1, h2) = (h[0], h[1], h[2]);
    let (h3, h4, h5) = (h[3], h[4], h[5]);
    let (h6, h7, h8) = (h[6], h[7], h[8]);

    let det = h0 * (h4 * h8 - h5 * h7) - h1 * (h3 * h8 - h5 * h6) + h2 * (h3 * h7 - h4 * h6);
    let inv_det = if det.abs() < 1e-10 { 0.0 } else { 1.0 / det };

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
    kernel_cell: &OnceLock<Result<CudaKernel, String>>,
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
    check_geometry(src_width, src_height, dst_width, dst_height, block_dim)
        .map_err(CudaRemapError::Cuda)?;
    check_image_slice(src, "src", src_width, src_height)?;
    check_image_slice(dst, "dst", dst_width, dst_height)?;
    check_map(map_x, "map_x", dst_width, dst_height)?;
    check_map(map_y, "map_y", dst_width, dst_height)?;

    let kernel = kernel_cell
        .get_or_init(|| try_compile_with_l1(ctx, kernel_src, fn_name))
        .as_ref()
        .map_err(|e| CudaRemapError::Cuda(e.clone()))?;

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(map_x)
        .arg(map_y)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
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
/// Source reads go through `__ldg` (L1 read-only cache); works at any image
/// width with no pitch-alignment constraint.
///
/// # Arguments
///
/// * `ctx`       — CUDA context (used for kernel compilation on first call).
/// * `stream`    — CUDA stream for the kernel launch.
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
/// Returns [`CudaRemapError`] on compile failure, launch error, or if any
/// slice is too small.
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
/// Source reads go through `__ldg`. Out-of-bounds coords yield 0.
///
/// # Arguments
///
/// See [`launch_remap_bilinear_cuda`] — arguments are identical.
///
/// # Errors
///
/// Returns [`CudaRemapError`] on compile failure, launch error, or if any
/// slice is too small.
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
