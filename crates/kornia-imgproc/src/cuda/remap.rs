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
//!
//! Map-generation helpers (`remap_maps_from_affine`, `remap_maps_from_homography`) live
//! in `examples/bench_cuda_remap.rs` — they serve the architecture-decision benchmark
//! only; the library API is the pure remap launchers.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::{check_geometry, make_config, try_compile_with_l1};

// Map-generation helpers (remap_maps_from_affine, remap_maps_from_homography)
// have been moved into examples/bench_cuda_remap.rs.  They served only the
// architecture-decision benchmark; the library surface is the pure launchers.

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

// ── Error type ────────────────────────────────────────────────────────────────

super::define_cuda_error!(
    /// Error returned by the CUDA remap launchers.
    CudaRemapError,
    "CUDA remap error: {0}"
);

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
    CudaRemapError::check_slice(
        "src",
        src.len(),
        (src_width as usize) * (src_height as usize) * 3,
    )?;
    CudaRemapError::check_slice(
        "dst",
        dst.len(),
        (dst_width as usize) * (dst_height as usize) * 3,
    )?;
    CudaRemapError::check_slice(
        "map_x",
        map_x.len(),
        (dst_width as usize) * (dst_height as usize),
    )?;
    CudaRemapError::check_slice(
        "map_y",
        map_y.len(),
        (dst_width as usize) * (dst_height as usize),
    )?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_f32};
    use crate::interpolation::InterpolationMode;
    use kornia_image::{Image, ImageSize};

    #[cfg(feature = "cuda")]
    fn cpu_and_gpu(
        w: usize,
        h: usize,
        mx: Vec<f32>,
        my: Vec<f32>,
        interpolation: InterpolationMode,
    ) -> (Vec<f32>, Vec<f32>) {
        let npix = w * h;
        let data = pattern_f32(npix * 3);

        let src_img = Image::<f32, 3>::new(
            ImageSize {
                width: w,
                height: h,
            },
            data.clone(),
        )
        .unwrap();
        let mut dst_img = Image::<f32, 3>::from_size_val(
            ImageSize {
                width: w,
                height: h,
            },
            0.0,
        )
        .unwrap();
        let map_x_t = Image::<f32, 1>::new(
            ImageSize {
                width: w,
                height: h,
            },
            mx.clone(),
        )
        .unwrap();
        let map_y_t = Image::<f32, 1>::new(
            ImageSize {
                width: w,
                height: h,
            },
            my.clone(),
        )
        .unwrap();
        crate::interpolation::remap(&src_img, &mut dst_img, &map_x_t, &map_y_t, interpolation)
            .unwrap();
        let cpu = dst_img.as_slice().to_vec();

        let stream = default_stream();
        let ctx = stream.context();
        let d_src = stream.clone_htod(&data).unwrap();
        let d_mx = stream.clone_htod(&mx).unwrap();
        let d_my = stream.clone_htod(&my).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(npix * 3).unwrap();
        let (wu, hu) = (w as u32, h as u32);
        match interpolation {
            InterpolationMode::Bilinear => launch_remap_bilinear_cuda(
                ctx, &stream, &d_src, &d_mx, &d_my, &mut d_dst, wu, hu, wu, hu, None,
            ),
            InterpolationMode::Nearest => launch_remap_nearest_cuda(
                ctx, &stream, &d_src, &d_mx, &d_my, &mut d_dst, wu, hu, wu, hu, None,
            ),
            other => panic!("unsupported mode in test: {other:?}"),
        }
        .unwrap();
        let gpu = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        (cpu, gpu)
    }

    fn assert_bit_exact(w: usize, h: usize, mx: Vec<f32>, my: Vec<f32>, mode: InterpolationMode) {
        let (cpu, gpu) = cpu_and_gpu(w, h, mx, my, mode);
        for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
            assert!(
                c.to_bits() == g.to_bits(),
                "{w}x{h} {mode:?}: element {i}: cpu {c} ({:#010x}) gpu {g} ({:#010x})",
                c.to_bits(),
                g.to_bits()
            );
        }
    }

    /// Identity map: GPU output must reproduce the source exactly and match CPU.
    #[test]
    #[ignore = "requires CUDA"]
    fn remap_identity_bilinear() {
        let (w, h) = (65, 33);
        let mx: Vec<f32> = (0..h).flat_map(|_| (0..w).map(|x| x as f32)).collect();
        let my: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |_| y as f32)).collect();
        assert_bit_exact(w, h, mx, my, InterpolationMode::Bilinear);
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn remap_identity_nearest() {
        let (w, h) = (65, 33);
        let mx: Vec<f32> = (0..h).flat_map(|_| (0..w).map(|x| x as f32)).collect();
        let my: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |_| y as f32)).collect();
        assert_bit_exact(w, h, mx, my, InterpolationMode::Nearest);
    }

    /// Sub-pixel bilinear: non-integer source coordinates in the image interior —
    /// exercises the weight expression against bilinear_interpolation's shape.
    #[test]
    #[ignore = "requires CUDA"]
    fn remap_subpixel_bilinear() {
        let (w, h) = (97, 65);
        // Each output pixel maps to (x+0.3, y+0.4); clamped to keep both taps
        // strictly inside [0, w-2] × [0, h-2] — avoids the replication edge path
        // which the CPU and GPU handle via different arithmetic.
        let mx: Vec<f32> = (0..h)
            .flat_map(|_| (0..w).map(|x| (x as f32 + 0.3).min((w - 2) as f32)))
            .collect();
        let my: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |_| (y as f32 + 0.4).min((h - 2) as f32)))
            .collect();
        assert_bit_exact(w, h, mx, my, InterpolationMode::Bilinear);
    }

    /// OOB coordinates must produce 0 on GPU; verifies the border guard.
    #[test]
    #[ignore = "requires CUDA"]
    fn remap_oob_writes_zero() {
        let (w, h) = (32, 32);
        let mx = vec![-5.0f32; w * h];
        let my = vec![-5.0f32; w * h];
        let stream = default_stream();
        let ctx = stream.context();
        let data = pattern_f32(w * h * 3);
        let d_src = stream.clone_htod(&data).unwrap();
        let d_mx = stream.clone_htod(&mx).unwrap();
        let d_my = stream.clone_htod(&my).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(w * h * 3).unwrap();
        launch_remap_bilinear_cuda(
            ctx, &stream, &d_src, &d_mx, &d_my, &mut d_dst, w as u32, h as u32, w as u32, h as u32,
            None,
        )
        .unwrap();
        let gpu = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        assert!(gpu.iter().all(|&v| v == 0.0), "OOB pixels must be 0");
    }
}
