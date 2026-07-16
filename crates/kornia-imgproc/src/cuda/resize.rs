//! Native CUDA downscale kernels for `kornia-imgproc`.
//!
//! # Why CubeCL bilinear downscale is slow
//!
//! CubeCL exposes no texture or shared-memory API, so every source read goes
//! through the plain L2 cache with no read-only hint.  Bilinear downscale
//! makes 4 source reads per output pixel at scattered addresses, leaving DRAM
//! bandwidth at ~55 % of peak.
//!
//! # Optimisations applied
//!
//! ## 32×8 thread block (default)
//!
//! A 16×16 block has each CUDA warp (32 threads) spanning **two output rows**
//! (threads 0–15 on row Y, threads 16–31 on row Y+1).  Every store and source
//! load instruction then generates L2 transactions from two different memory
//! row regions — a 25 % penalty in write transactions and source-read
//! locality.
//!
//! A **32×8 block** aligns an entire warp to a single output row and its
//! corresponding source row pair.  Writes are in one contiguous 384-byte
//! region (3 cache lines); bilinear reads are confined to two source rows
//! instead of four.
//!
//! ## Block size — fixed tap count, adaptive thread layout
//!
//! The interpolation tap count is **fixed at compile time** per mode
//! (bilinear = 2×2, bicubic = 4×4), which lets the compiler fully unroll the
//! weight loops and pipeline the `__ldg` reads.  This is the correct design
//! for GPU — a runtime-variable tap count would prevent unrolling and add
//! branch overhead per tap.
//!
//! The *thread block layout* is separate from the tap count and is tunable:
//! all launchers accept `block_dim: Option<(u32, u32)>`.  `None` selects
//! **auto mode**: 32×8 for images wider than 32 px and taller than 8 px,
//! clamped to `(min(32, dst_width), min(8, dst_height))` for smaller images
//! so thread blocks don't launch mostly-idle warps.  Pass `Some((bw, bh))`
//! to override manually (e.g. `Some((16, 4))` for a narrow portrait crop).
//!
//! ## L1 cache preference (`CU_FUNC_CACHE_PREFER_L1`)
//!
//! Neither kernel uses shared memory, so the SM's combined L1/smem budget is
//! fully allocated to the L1 data cache via `cuFuncSetCacheConfig`.  On Turing
//! (GTX 1650) this enlarges L1 from the default 32 KB to 64 KB, directly
//! improving hit rates.
//!
//! # Nearest-neighbor
//!
//! Reads exactly one source pixel per output pixel (no bilinear reuse), so
//! `__ldg` alone reaches ~91 % of DRAM peak.  Same 32×8 block and `float2`
//! stores applied for consistency.
//!
//! # Fused bilinear + normalise
//!
//! [`launch_resize_bilinear_normalize_cuda`] performs bilinear downscale and
//! per-channel normalisation `(pixel − mean) / std` in a single kernel.
//!
//! Separately, bilinear resize writes ~6.2 MB and a normalise pass reads and
//! rewrites the same 6.2 MB — a total of ~24.8 MB DRAM.  The fused kernel
//! eliminates the normalise pass: the same 12.4 MB total at the same bandwidth.
//! For a 1080p → 540p pipeline the combined time drops from ~0.32 ms to
//! ~0.18 ms (~1.7× faster).
//!
//! # Measured throughput (GTX 1650, 2× downscale)
//!
//! | Kernel              | 1080p→540p | GB/s  |
//! |---------------------|-----------|-------|
//! | Nearest             | 0.107 ms  | ~116  |
//! | Bilinear            | 0.178 ms  | ~70   |
//! | Bilinear+normalise  | ~0.178 ms | ~70   |
//! | CPU (BL)            | 5.18 ms   | ~2.4  |
//!
//! # Public API
//!
//! * [`launch_resize_bilinear_downscale_cuda`]  — bilinear downscale, 3-ch f32.
//! * [`launch_resize_nearest_downscale_cuda`]   — nearest-neighbor, 3-ch f32.
//! * [`launch_resize_bilinear_normalize_cuda`]  — bilinear downscale + normalise, 3-ch f32.
//! * [`launch_resize_bicubic_cuda`]             — bicubic resize (up or down), 3-ch f32.
//! * [`launch_resize_lanczos_cuda`]             — Lanczos-3 separable 2-pass resize (up or down), 3-ch f32.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig};
use kornia_tensor::CudaKernel;

// ── CUDA C source: bilinear, 32×8 block, __ldg reads ─────────────────────────
//
// For regular downscale, consecutive output pixels in a row sample predictable
// source locations. The unified L1 cache (`__ldg`) coalesces these reads well
// and already achieves ~84–88% DRAM bandwidth. Texture objects would route
// reads through the texture cache but the non-unit stride of the 1-channel
// pitch-2D trick (width = src_w * 3) hurts cache-line efficiency — benchmarks
// show a 10% regression vs the __ldg path for this pattern.

static BILINEAR_SRC: &str = r#"
extern "C" __global__ void resize_bilinear_downscale_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float ax, float bx,
    float ay, float by
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    // Plain multiply-add, NOT fmaf: with --fmad=false this rounds twice,
    // exactly like the CPU LUT's `a * x + b` — the coordinate is byte-exact
    // across CPU and GPU, which the parity tests assert. Clamp order (min
    // then max) is equivalent to Rust's f32::clamp for lo <= hi.
    float sx = fmaxf(fminf(ax * (float)dst_x + bx, (float)(src_w - 1u)), 0.0f);
    float sy = fmaxf(fminf(ay * (float)dst_y + by, (float)(src_h - 1u)), 0.0f);

    unsigned int x0 = (unsigned int)sx;
    unsigned int y0 = (unsigned int)sy;
    unsigned int x1 = min(x0 + 1u, src_w - 1u);
    unsigned int y1 = min(y0 + 1u, src_h - 1u);

    float fx  = sx - (float)x0;
    float fy  = sy - (float)y0;
    float w00 = (1.0f - fy) * (1.0f - fx);
    float w10 = (1.0f - fy) * fx;
    float w01 = fy * (1.0f - fx);
    float w11 = fy * fx;

    unsigned int b00 = (y0 * src_w + x0) * 3u;
    unsigned int b10 = (y0 * src_w + x1) * 3u;
    unsigned int b01 = (y1 * src_w + x0) * 3u;
    unsigned int b11 = (y1 * src_w + x1) * 3u;

    unsigned int out = (dst_y * dst_w + dst_x) * 3u;
    dst[out]     = w00*__ldg(&src[b00])   + w10*__ldg(&src[b10])   + w01*__ldg(&src[b01])   + w11*__ldg(&src[b11]);
    dst[out + 1] = w00*__ldg(&src[b00+1]) + w10*__ldg(&src[b10+1]) + w01*__ldg(&src[b01+1]) + w11*__ldg(&src[b11+1]);
    dst[out + 2] = w00*__ldg(&src[b00+2]) + w10*__ldg(&src[b10+2]) + w01*__ldg(&src[b01+2]) + w11*__ldg(&src[b11+2]);
}
"#;

// ── CUDA C source: nearest-neighbor downscale, __ldg reads ───────────────────

static NEAREST_SRC: &str = r#"
extern "C" __global__ void resize_nearest_downscale_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float ax, float bx,
    float ay, float by
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    // Half-pixel center alignment.
    // Plain multiply-add (not fmaf) so the coordinate is bit-identical to the
    // CPU LUT under --fmad=false. +0.5-then-truncate is round-half-up, which
    // equals the CPU's half-away-from-zero `round()` for the non-negative
    // coordinates produced here — including exact .5 ties, since both sides
    // compute the identical f32 coordinate.
    unsigned int src_xi = min((unsigned int)((ax * (float)dst_x + bx) + 0.5f), src_w - 1u);
    unsigned int src_yi = min((unsigned int)((ay * (float)dst_y + by) + 0.5f), src_h - 1u);

    unsigned int src_base = (src_yi * src_w + src_xi) * 3u;
    unsigned int out = (dst_y * dst_w + dst_x) * 3u;
    dst[out]     = __ldg(&src[src_base]);
    dst[out + 1] = __ldg(&src[src_base + 1]);
    dst[out + 2] = __ldg(&src[src_base + 2]);
}
"#;

// ── CUDA C source: fused bilinear downscale + per-channel normalise ───────────
//
// Kept as a plain __ldg kernel (no texture): normalise is a single-pass
// compute-bound operation and the fused path already eliminates the second
// DRAM round-trip.

static BILINEAR_NORMALIZE_SRC: &str = r#"
extern "C" __global__ void resize_bilinear_normalize_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float ax, float bx,
    float ay, float by,
    float mean0,    float mean1,    float mean2,
    float inv_std0, float inv_std1, float inv_std2
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    // Plain multiply-add, NOT fmaf: with --fmad=false this rounds twice,
    // exactly like the CPU LUT's `a * x + b` — the coordinate is byte-exact
    // across CPU and GPU, which the parity tests assert. Clamp order (min
    // then max) is equivalent to Rust's f32::clamp for lo <= hi.
    float sx = fmaxf(fminf(ax * (float)dst_x + bx, (float)(src_w - 1u)), 0.0f);
    float sy = fmaxf(fminf(ay * (float)dst_y + by, (float)(src_h - 1u)), 0.0f);

    unsigned int x0 = (unsigned int)sx;
    unsigned int y0 = (unsigned int)sy;
    unsigned int x1 = min(x0 + 1u, src_w - 1u);
    unsigned int y1 = min(y0 + 1u, src_h - 1u);

    float fx  = sx - (float)x0;
    float fy  = sy - (float)y0;
    float w00 = (1.0f - fy) * (1.0f - fx);
    float w10 = (1.0f - fy) * fx;
    float w01 = fy * (1.0f - fx);
    float w11 = fy * fx;

    unsigned int b00 = (y0 * src_w + x0) * 3u;
    unsigned int b10 = (y0 * src_w + x1) * 3u;
    unsigned int b01 = (y1 * src_w + x0) * 3u;
    unsigned int b11 = (y1 * src_w + x1) * 3u;

    // Bilinear interpolation via __ldg (read-only L1 cache).
    float ch0 = w00*__ldg(&src[b00])   + w10*__ldg(&src[b10])   + w01*__ldg(&src[b01])   + w11*__ldg(&src[b11]);
    float ch1 = w00*__ldg(&src[b00+1]) + w10*__ldg(&src[b10+1]) + w01*__ldg(&src[b01+1]) + w11*__ldg(&src[b11+1]);
    float ch2 = w00*__ldg(&src[b00+2]) + w10*__ldg(&src[b10+2]) + w01*__ldg(&src[b01+2]) + w11*__ldg(&src[b11+2]);

    // Fused normalise: (pixel - mean) * inv_std avoids a separate DRAM pass.
    unsigned int out = (dst_y * dst_w + dst_x) * 3u;
    dst[out]     = (ch0 - mean0) * inv_std0;
    dst[out + 1] = (ch1 - mean1) * inv_std1;
    dst[out + 2] = (ch2 - mean2) * inv_std2;
}
"#;

// ── CUDA C source: bicubic resize ─────────────────────────────────────────────
//
// Same Keys cubic (a = -0.5) as the warp-affine bicubic kernel. Applies the
// same Horner-form weight precomputation, row-base hoisting, #pragma unroll,
// and fmaf accumulation. Half-pixel centre alignment (matches OpenCV/PIL).
// Handles both upscale and downscale.

static BICUBIC_SRC: &str = r#"
extern "C" __global__ void resize_bicubic_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float ax, float bx,
    float ay, float by
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    // Plain multiply-add (not fmaf): bit-identical to the CPU LUT's `a*x + b`
    // under --fmad=false — the byte-exact contract, same as bilinear/nearest.
    float sx = fmaxf(fminf(ax * (float)dst_x + bx, (float)(src_w - 1u)), 0.0f);
    float sy = fmaxf(fminf(ay * (float)dst_y + by, (float)(src_h - 1u)), 0.0f);

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float frac_x = sx - (float)x0;
    float frac_y = sy - (float)y0;

    // Horner-form cubic weights — see warp_affine_bicubic_3c for derivation.
    float wx[4], wy[4];
    {
        float t;
        t = 1.0f + frac_x; wx[0] = fmaf(fmaf(fmaf(-0.5f, t, 2.5f), t, -4.0f), t, 2.0f);
        t =         frac_x; wx[1] = fmaf(fmaf( 1.5f, t, -2.5f) * t,       t, 1.0f);
        t = 1.0f - frac_x; wx[2] = fmaf(fmaf( 1.5f, t, -2.5f) * t,       t, 1.0f);
        t = 2.0f - frac_x; wx[3] = fmaf(fmaf(fmaf(-0.5f, t, 2.5f), t, -4.0f), t, 2.0f);
        t = 1.0f + frac_y; wy[0] = fmaf(fmaf(fmaf(-0.5f, t, 2.5f), t, -4.0f), t, 2.0f);
        t =         frac_y; wy[1] = fmaf(fmaf( 1.5f, t, -2.5f) * t,       t, 1.0f);
        t = 1.0f - frac_y; wy[2] = fmaf(fmaf( 1.5f, t, -2.5f) * t,       t, 1.0f);
        t = 2.0f - frac_y; wy[3] = fmaf(fmaf(fmaf(-0.5f, t, 2.5f), t, -4.0f), t, 2.0f);
    }

    unsigned int row[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int yi = max(0, min(y0 + i - 1, (int)src_h - 1));
        row[i] = (unsigned int)yi * src_w * 3u;
    }

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    #pragma unroll
    for (int dy = 0; dy < 4; ++dy) {
        #pragma unroll
        for (int dx = 0; dx < 4; ++dx) {
            int xi = max(0, min(x0 + dx - 1, (int)src_w - 1));
            unsigned int b = row[dy] + (unsigned int)xi * 3u;
            float w = wx[dx] * wy[dy];
            acc0 = fmaf(w, __ldg(&src[b]),   acc0);
            acc1 = fmaf(w, __ldg(&src[b+1]), acc1);
            acc2 = fmaf(w, __ldg(&src[b+2]), acc2);
        }
    }
    unsigned int out = (dst_y * dst_w + dst_x) * 3u;
    dst[out]     = acc0;
    dst[out + 1] = acc1;
    dst[out + 2] = acc2;
}
"#;

// ── CUDA C source: Lanczos-3 resize — separable 2-pass ───────────────────────
//
// Lanczos is separable: L(x,y) = L(x)·L(y). Instead of a 6×6=36 tap 2D
// kernel, two 6-tap 1D passes (horizontal then vertical) are used with an
// intermediate buffer of shape (dst_w × src_h × 3).
//
// Read reduction vs 2D:  2D = 36·dst_w·dst_h
//                        sep = 6·dst_w·(src_h + dst_h)
//   2× downscale: ~2× fewer reads   (e.g. 1080p→540p: 18.7M → 9.3M)
//   2× upscale:   ~4× fewer reads   (e.g. 1080p→4K:   299M  → 75M)
//
// Both kernels use precise `sinf` / IEEE division rather than the fast
// `__sinf` / `__fdividef` intrinsics: the CPU lanczos weight computation must
// reproduce the same bits, and the approximate intrinsics have no host
// equivalent. Weight computation is a per-output-pixel prologue; the precise
// forms measure in the noise on Orin.

// Each of the three Lanczos NVRTC modules (H, V, and warp-affine) defines the
// same `lanczos3` device helper. They are compiled as separate translation units,
// so sharing the name is fine — no ODR violation across NVRTC compilations.
static LANCZOS_H_SRC: &str = r#"
extern "C" __global__ void resize_lanczos_h_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    const int*   __restrict__ x0s,
    const float* __restrict__ wtab,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int src_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || src_y >= src_h) return;

    // Per-column tap base and normalized weights are host-precomputed by the
    // same Rust code the CPU path uses (`lanczos_axis`), so the two paths are
    // byte-exact by construction and the kernel is pure gather + FMA.
    int x0 = __ldg(&x0s[dst_x]);
    const float* w = &wtab[dst_x * 6u];

    unsigned int row = src_y * src_w * 3u;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    #pragma unroll
    for (int dx = 0; dx < 6; dx++) {
        int xi = max(0, min(x0 + dx - 2, (int)src_w - 1));
        unsigned int b = row + (unsigned int)xi * 3u;
        float wd = __ldg(&w[dx]);
        acc0 = fmaf(wd, __ldg(&src[b]),   acc0);
        acc1 = fmaf(wd, __ldg(&src[b+1]), acc1);
        acc2 = fmaf(wd, __ldg(&src[b+2]), acc2);
    }

    unsigned int out = (src_y * dst_w + dst_x) * 3u;
    dst[out]     = acc0;
    dst[out + 1] = acc1;
    dst[out + 2] = acc2;
}
"#;

static LANCZOS_V_SRC: &str = r#"
extern "C" __global__ void resize_lanczos_v_3c(
    const float* __restrict__ inter,
    float* __restrict__       dst,
    const int*   __restrict__ y0s,
    const float* __restrict__ wtab,
    unsigned int dst_w,
    unsigned int inter_h,
    unsigned int dst_h
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    int y0 = __ldg(&y0s[dst_y]);
    const float* w = &wtab[dst_y * 6u];

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    #pragma unroll
    for (int dy = 0; dy < 6; dy++) {
        int yi = max(0, min(y0 + dy - 2, (int)inter_h - 1));
        unsigned int b = ((unsigned int)yi * dst_w + dst_x) * 3u;
        float wd = __ldg(&w[dy]);
        acc0 = fmaf(wd, __ldg(&inter[b]),   acc0);
        acc1 = fmaf(wd, __ldg(&inter[b+1]), acc1);
        acc2 = fmaf(wd, __ldg(&inter[b+2]), acc2);
    }

    unsigned int out = (dst_y * dst_w + dst_x) * 3u;
    dst[out]     = acc0;
    dst[out + 1] = acc1;
    dst[out + 2] = acc2;
}
"#;

// ── Kernel caches ─────────────────────────────────────────────────────────────

static BILINEAR_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static NEAREST_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static BILINEAR_NORMALIZE_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static BICUBIC_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static LANCZOS_H_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static LANCZOS_V_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

// 32 threads wide → full warp maps to one output row (better write coalescing).
// 8 threads tall → 256 threads total, same occupancy as 16×16.
const BLOCK_W: u32 = 32;
const BLOCK_H: u32 = 8;

// ── Error type ────────────────────────────────────────────────────────────────

/// Error type returned by the CUDA downscale launchers.
#[derive(Debug, thiserror::Error)]
pub enum CudaResizeError {
    /// CUDA kernel compilation or launch failure.
    #[error("CUDA kernel compile/launch error: {0}")]
    Cuda(String),
    /// Output device slice is smaller than the required pixel count.
    #[error("output slice length {got} < required {need}")]
    SliceTooSmall {
        /// Actual slice length (in elements).
        got: usize,
        /// Minimum required length (dst_w × dst_h × 3).
        need: usize,
    },
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn make_config(dst_width: u32, dst_height: u32, block_dim: Option<(u32, u32)>) -> LaunchConfig {
    let (bw, bh) = block_dim.unwrap_or_else(|| {
        // Auto mode: clamp to image size so small images don't launch
        // blocks that are mostly idle (e.g. a 4×4 image with 32×8 = 256
        // threads has 93% idle threads).
        (BLOCK_W.min(dst_width), BLOCK_H.min(dst_height))
    });
    LaunchConfig {
        block_dim: (bw, bh, 1),
        grid_dim: (dst_width.div_ceil(bw), dst_height.div_ceil(bh), 1),
        shared_mem_bytes: 0,
    }
}

fn compile_with_l1(ctx: &Arc<CudaContext>, src: &str, fn_name: &str) -> CudaKernel {
    let k = CudaKernel::compile(ctx, src, fn_name)
        .unwrap_or_else(|e| panic!("failed to compile {fn_name}: {e}"));
    // Prefer L1 over shared memory (kernel uses no smem).
    // Ignoring errors: unsupported on some platforms but never fatal.
    let _ = k.prefer_l1_cache();
    k
}

fn try_compile_with_l1(
    ctx: &Arc<CudaContext>,
    src: &str,
    fn_name: &str,
) -> Result<CudaKernel, String> {
    let k = CudaKernel::compile(ctx, src, fn_name)
        .map_err(|e| format!("failed to compile {fn_name}: {e}"))?;
    let _ = k.prefer_l1_cache();
    Ok(k)
}

// ── Pixel mapping ─────────────────────────────────────────────────────────────

/// How destination pixel coordinates map to source coordinates.
///
/// Every mapping is affine in the destination index, `src = a*dst + b`; the
/// variants only differ in the coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelMapping {
    /// Pixel-center sampling, `src = (dst + 0.5) * (src_len/dst_len) - 0.5`.
    ///
    /// The convention of OpenCV, Pillow, ONNX `Resize`
    /// (`coordinate_transformation_mode = half_pixel`), PyTorch
    /// (`align_corners=False`), and NVIDIA VPI — and of the CPU
    /// [`resize`](crate::resize::resize). Use this unless you
    /// specifically need to reproduce align-corners output.
    HalfPixel,
    /// Corner-aligned sampling, `src = dst * (src_len-1)/(dst_len-1)`.
    ///
    /// Reproduces the align-corners output kornia's CPU resize produced
    /// before the half-pixel switch, and matches frameworks running with
    /// `align_corners=True`. A single-pixel destination axis maps to source
    /// coordinate 0.
    AlignCorners,
}

impl PixelMapping {
    /// Per-axis affine coefficients `(a, b)` of `src = a*dst + b`.
    fn coeffs(self, src_len: u32, dst_len: u32) -> (f32, f32) {
        match self {
            PixelMapping::HalfPixel => {
                let a = src_len as f32 / dst_len as f32;
                (a, 0.5 * a - 0.5)
            }
            PixelMapping::AlignCorners => {
                if dst_len > 1 {
                    ((src_len - 1) as f32 / (dst_len - 1) as f32, 0.0)
                } else {
                    (0.0, 0.0)
                }
            }
        }
    }
}

// ── Public launchers ──────────────────────────────────────────────────────────

/// Launch the bilinear downscale kernel for a 3-channel f32 image.
///
/// Uses `__ldg` (read-only L1 cache) for source reads and a 32×8 thread block
/// for write coalescing.  For 2× downscale at 1080p→540p on GTX 1650,
/// measured at ~70 GB/s effective write bandwidth (84–88% DRAM utilisation).
///
/// Texture objects were evaluated but regressed ~10% for downscale: the
/// `tex2D` instruction latency (~100 cycles vs ~30 for `__ldg`) outweighs any
/// 2D-cache benefit because the sequential downscale pattern is already
/// well-served by the unified L1 via `__ldg`.
///
/// # Arguments
///
/// * `ctx`       – CUDA context used for one-time kernel compilation.
/// * `stream`    – Stream for kernel execution.
/// * `src`       – Device slice: `src_h × src_w × 3` f32 values.
/// * `dst`       – Device slice: `dst_h × dst_w × 3` f32 values (written).
/// * `src_width`, `src_height` – Source image dimensions.
/// * `dst_width`, `dst_height` – Output image dimensions (must be ≤ source).
/// * `block_dim` – Optional thread-block override `(width, height)`.
///   `None` uses 32×8 (optimal for large images).
///
/// # Errors
///
/// Returns [`CudaResizeError`] on compile failure, launch error, or size mismatch.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_bilinear_downscale_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mapping: PixelMapping,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(CudaResizeError::Cuda(
            "image dimensions must be non-zero".into(),
        ));
    }
    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaResizeError::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }

    let kernel = BILINEAR_KERNEL
        .get_or_init(|| compile_with_l1(ctx, BILINEAR_SRC, "resize_bilinear_downscale_3c"));

    let (ax, bx) = mapping.coeffs(src_width, dst_width);
    let (ay, by) = mapping.coeffs(src_height, dst_height);

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&ax)
        .arg(&bx)
        .arg(&ay)
        .arg(&by)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Launch a fused bilinear downscale + per-channel normalise kernel.
///
/// Performs bilinear interpolation and `(pixel − mean[c]) / std[c]` in a
/// single pass, eliminating the separate normalise kernel that would otherwise
/// read and rewrite the full output buffer.  For a 1080p → 540p pipeline this
/// reduces combined resize+normalise time from ~0.32 ms to ~0.18 ms.
///
/// # Arguments
///
/// * `ctx`    – CUDA context used for one-time kernel compilation.
/// * `stream` – Stream for kernel execution.
/// * `src`    – Device slice: `src_h × src_w × 3` f32 values (channel-last RGB).
/// * `dst`    – Device slice: `dst_h × dst_w × 3` f32 values (written normalised).
/// * `src_width`, `src_height` – Source image dimensions.
/// * `dst_width`, `dst_height` – Output image dimensions (must be ≤ source).
/// * `mean`   – Per-channel mean `[ch0, ch1, ch2]` (same channel order as image).
/// * `std`    – Per-channel standard deviation (must be non-zero).
/// * `block_dim` – Optional thread-block override `(width, height)`.
///
/// # Errors
///
/// Returns [`CudaResizeError`] on compile failure, launch error, size mismatch,
/// or if any element of `std` is zero.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_bilinear_normalize_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mean: [f32; 3],
    std: [f32; 3],
    mapping: PixelMapping,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(CudaResizeError::Cuda(
            "image dimensions must be non-zero".into(),
        ));
    }
    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaResizeError::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }
    if std[0] == 0.0 || std[1] == 0.0 || std[2] == 0.0 {
        return Err(CudaResizeError::Cuda(
            "std must be non-zero for all channels".into(),
        ));
    }

    let kernel = BILINEAR_NORMALIZE_KERNEL.get_or_init(|| {
        compile_with_l1(ctx, BILINEAR_NORMALIZE_SRC, "resize_bilinear_normalize_3c")
    });

    let (ax, bx) = mapping.coeffs(src_width, dst_width);
    let (ay, by) = mapping.coeffs(src_height, dst_height);

    let inv_std0 = 1.0_f32 / std[0];
    let inv_std1 = 1.0_f32 / std[1];
    let inv_std2 = 1.0_f32 / std[2];

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&ax)
        .arg(&bx)
        .arg(&ay)
        .arg(&by)
        .arg(&mean[0])
        .arg(&mean[1])
        .arg(&mean[2])
        .arg(&inv_std0)
        .arg(&inv_std1)
        .arg(&inv_std2)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Launch the nearest-neighbor downscale kernel for a 3-channel f32 image.
///
/// Uses `__ldg` and a 32×8 thread block.  Reaches ~91 % of theoretical DRAM
/// bandwidth on GTX 1650 (single source read per output pixel).
///
/// # Arguments
///
/// See [`launch_resize_bilinear_downscale_cuda`] — arguments are identical.
///
/// # Errors
///
/// Returns [`CudaResizeError`] on compile failure, launch error, or size mismatch.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_nearest_downscale_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mapping: PixelMapping,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(CudaResizeError::Cuda(
            "image dimensions must be non-zero".into(),
        ));
    }
    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaResizeError::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }

    let kernel = NEAREST_KERNEL
        .get_or_init(|| compile_with_l1(ctx, NEAREST_SRC, "resize_nearest_downscale_3c"));

    let (ax, bx) = mapping.coeffs(src_width, dst_width);
    let (ay, by) = mapping.coeffs(src_height, dst_height);

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&ax)
        .arg(&bx)
        .arg(&ay)
        .arg(&by)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Launch the bicubic resize kernel for a 3-channel f32 image.
///
/// Uses Keys cubic interpolation (`a = -0.5`, matches OpenCV `INTER_CUBIC`)
/// with a 4×4 tap neighborhood and half-pixel centre alignment.  Supports
/// both upscale and downscale.  Out-of-range taps are clamped to the image
/// border (BORDER_REPLICATE).
///
/// Bicubic reads 16 source values per output pixel — more bandwidth-intensive
/// than bilinear (4 reads) but produces sharper results without ringing at the
/// default `a = -0.5` coefficient.  Hardware texture bilinear cannot accelerate
/// a 4×4 stencil, so this kernel continues to use `__ldg`.
///
/// # Arguments
///
/// * `ctx`       – CUDA context for one-time kernel compilation.
/// * `stream`    – Stream for kernel execution.
/// * `src`       – Device slice: `src_h × src_w × 3` f32 values.
/// * `dst`       – Device slice: `dst_h × dst_w × 3` f32 values (written).
/// * `src_width`, `src_height` – Source image dimensions (must be non-zero).
/// * `dst_width`, `dst_height` – Output dimensions (must be non-zero).
///
/// # Errors
///
/// Returns [`CudaResizeError`] on compile failure, launch error, or size mismatch.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_bicubic_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mapping: PixelMapping,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(CudaResizeError::Cuda(
            "src and dst dimensions must be non-zero".into(),
        ));
    }

    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaResizeError::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }

    let kernel = BICUBIC_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, BICUBIC_SRC, "resize_bicubic_3c"))
        .as_ref()
        .map_err(|e| CudaResizeError::Cuda(e.clone()))?;

    let (ax, bx) = mapping.coeffs(src_width, dst_width);
    let (ay, by) = mapping.coeffs(src_height, dst_height);

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&ax)
        .arg(&bx)
        .arg(&ay)
        .arg(&by)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Launch the Lanczos-3 resize kernel for a 3-channel f32 image.
///
/// Implements a separable 2-pass filter: a horizontal 6-tap pass writes an
/// intermediate buffer of shape `(dst_w × src_h × 3)`, then a vertical 6-tap
/// pass produces the final `(dst_w × dst_h × 3)` output.  This reduces source
/// reads from 36 (2D kernel) to 6+6=12, roughly 2× faster on 2× downscale and
/// ~4× faster on 2× upscale.
///
/// The intermediate buffer is allocated on the stream per call.  For tight
/// loops that resize the same dimensions repeatedly, the kernel compilations
/// are cached in `OnceLock`s so the NVRTC cost is paid only once.
///
/// # Arguments
///
/// * `ctx`       – CUDA context for one-time kernel compilation.
/// * `stream`    – Stream for kernel execution and scratch allocation.
/// * `src`       – Device slice: `src_h × src_w × 3` f32 values.
/// * `dst`       – Device slice: `dst_h × dst_w × 3` f32 values (written).
/// * `src_width`, `src_height` – Source image dimensions (must be non-zero).
/// * `dst_width`, `dst_height` – Output dimensions (must be non-zero).
///
/// # Errors
///
/// Returns [`CudaResizeError`] on compile failure, allocation failure, launch
/// error, or size mismatch.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_lanczos_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mapping: PixelMapping,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(CudaResizeError::Cuda(
            "src and dst dimensions must be non-zero".into(),
        ));
    }

    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaResizeError::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }

    let kernel_h = LANCZOS_H_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, LANCZOS_H_SRC, "resize_lanczos_h_3c"))
        .as_ref()
        .map_err(|e| CudaResizeError::Cuda(e.clone()))?;

    let kernel_v = LANCZOS_V_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, LANCZOS_V_SRC, "resize_lanczos_v_3c"))
        .as_ref()
        .map_err(|e| CudaResizeError::Cuda(e.clone()))?;

    // Per-axis tap/weight tables, host-built by the SAME `lanczos_axis` the
    // CPU separable path uses — byte-exact across backends by construction,
    // and the kernels stay pure gather + FMA (the pre-table kernels spent
    // ~40% of their time recomputing identical per-column weights per row).
    // `mapping` is fixed to the half-pixel contract the tables encode.
    if mapping != PixelMapping::HalfPixel {
        return Err(CudaResizeError::Cuda(
            "lanczos resize supports PixelMapping::HalfPixel only".into(),
        ));
    }
    let (x0s_h, wtab_h) =
        crate::interpolation::lanczos::lanczos_axis(src_width as usize, dst_width as usize);
    let (y0s_v, wtab_v) =
        crate::interpolation::lanczos::lanczos_axis(src_height as usize, dst_height as usize);
    let to_cuda_err = |e: cudarc::driver::result::DriverError| CudaResizeError::Cuda(e.to_string());
    let d_x0s = stream.clone_htod(&x0s_h).map_err(to_cuda_err)?;
    let d_wtab_h = stream.clone_htod(&wtab_h).map_err(to_cuda_err)?;
    let d_y0s = stream.clone_htod(&y0s_v).map_err(to_cuda_err)?;
    let d_wtab_v = stream.clone_htod(&wtab_v).map_err(to_cuda_err)?;

    // Intermediate: dst_w columns, src_h rows — horizontal pass output.
    // Pass 1 writes every element before pass 2 reads; no zero-fill needed.
    let inter_len = (dst_width as usize) * (src_height as usize) * 3;
    let mut intermediate = unsafe { stream.alloc::<f32>(inter_len) }
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))?;

    // Pass 1 — horizontal: (src_w, src_h) → (dst_w, src_h).
    kernel_h
        .launch_builder(stream)
        .arg(src)
        .arg(&mut intermediate)
        .arg(&d_x0s)
        .arg(&d_wtab_h)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .launch_2d(
            dst_width,
            src_height,
            make_config(dst_width, src_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))?;

    // Pass 2 — vertical: (dst_w, src_h) → (dst_w, dst_h).
    kernel_v
        .launch_builder(stream)
        .arg(&intermediate)
        .arg(dst)
        .arg(&d_y0s)
        .arg(&d_wtab_v)
        .arg(&dst_width)
        .arg(&src_height)
        .arg(&dst_height)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_f32};
    use crate::interpolation::InterpolationMode;
    use crate::resize::resize;
    use kornia_image::{Image, ImageSize};

    #[test]
    fn pixel_mapping_coeffs() {
        // HalfPixel: a = src/dst, b = a/2 - 1/2.
        assert_eq!(PixelMapping::HalfPixel.coeffs(640, 320), (2.0, 0.5));
        assert_eq!(PixelMapping::HalfPixel.coeffs(320, 640), (0.5, -0.25));
        // A 1-wide destination samples the source centre.
        assert_eq!(PixelMapping::HalfPixel.coeffs(9, 1), (9.0, 4.0));
        // AlignCorners: a = (src-1)/(dst-1), b = 0; 1-wide dst pins to 0.
        assert_eq!(PixelMapping::AlignCorners.coeffs(641, 321), (2.0, 0.0));
        assert_eq!(PixelMapping::AlignCorners.coeffs(9, 1), (0.0, 0.0));
    }

    /// Run CPU `resize` (half-pixel) and the GPU launcher on the same
    /// deterministic input; return both outputs.
    fn cpu_and_gpu(
        (sw, sh): (usize, usize),
        (dw, dh): (usize, usize),
        interpolation: InterpolationMode,
    ) -> (Vec<f32>, Vec<f32>) {
        let data = pattern_f32(sw * sh * 3);
        let src = Image::<f32, 3>::new(
            ImageSize {
                width: sw,
                height: sh,
            },
            data.clone(),
        )
        .unwrap();
        let mut cpu = Image::<f32, 3>::from_size_val(
            ImageSize {
                width: dw,
                height: dh,
            },
            0.0,
        )
        .unwrap();
        resize(&src, &mut cpu, interpolation).unwrap();

        let stream = default_stream();
        let ctx = &stream.context();
        let d_src = stream.clone_htod(&data).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(dw * dh * 3).unwrap();
        let (swu, shu, dwu, dhu) = (sw as u32, sh as u32, dw as u32, dh as u32);
        match interpolation {
            InterpolationMode::Bilinear => launch_resize_bilinear_downscale_cuda(
                ctx,
                &stream,
                &d_src,
                &mut d_dst,
                swu,
                shu,
                dwu,
                dhu,
                PixelMapping::HalfPixel,
                None,
            ),
            InterpolationMode::Nearest => launch_resize_nearest_downscale_cuda(
                ctx,
                &stream,
                &d_src,
                &mut d_dst,
                swu,
                shu,
                dwu,
                dhu,
                PixelMapping::HalfPixel,
                None,
            ),
            other => panic!("unsupported mode in test: {other:?}"),
        }
        .unwrap_or_else(|e| panic!("launch failed {sw}x{sh}->{dw}x{dh} ({interpolation:?}): {e}"));
        let gpu: Vec<f32> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        (cpu.as_slice().to_vec(), gpu)
    }

    /// Byte-exact comparison: the CPU LUT and the kernels compute the identical
    /// uncontracted `a*x + b` coordinate (see the contract note in
    /// `resize`), and the bilinear weight/sum expression shapes match,
    /// so CPU and GPU must agree bit-for-bit — no tolerance, no exclusions.
    fn assert_bit_exact(
        src: (usize, usize),
        dst: (usize, usize),
        interpolation: InterpolationMode,
    ) {
        let (cpu, gpu) = cpu_and_gpu(src, dst, interpolation);
        let bad = cpu
            .iter()
            .zip(&gpu)
            .enumerate()
            .find(|(_, (c, g))| c.to_bits() != g.to_bits());
        if let Some((i, (c, g))) = bad {
            panic!(
                "{src:?}->{dst:?} {interpolation:?}: first mismatch at element {i}: cpu {c} ({:#010x}) gpu {g} ({:#010x})",
                c.to_bits(),
                g.to_bits()
            );
        }
    }

    /// Dyadic 2× downscale — historically the easy case; now just one instance
    /// of the blanket byte-exact contract.
    #[test]
    fn resize_2x_downscale_matches_cpu() {
        assert_bit_exact((640, 480), (320, 240), InterpolationMode::Nearest);
        assert_bit_exact((640, 480), (320, 240), InterpolationMode::Bilinear);
    }

    /// Upscale goes through the same kernels (the "downscale" in the launcher
    /// names is historical) — parity must hold in both directions.
    #[test]
    fn resize_2x_upscale_matches_cpu() {
        assert_bit_exact((320, 240), (640, 480), InterpolationMode::Bilinear);
        assert_bit_exact((320, 240), (640, 480), InterpolationMode::Nearest);
    }

    /// Odd, prime-ish, and unaligned sizes with non-dyadic scales: exactly the
    /// cases where the pre-byte-exact code needed tolerances and rounding-tie
    /// exclusions. Bit-equality here is the point of the fmad=false +
    /// mirrored-expression contract.
    #[test]
    fn resize_odd_sizes_match_cpu() {
        for &(src, dst) in &[
            ((127, 63), (65, 33)),
            ((129, 97), (64, 48)),
            ((255, 130), (133, 67)),
            ((63, 129), (127, 255)), // upscale
        ] {
            assert_bit_exact(src, dst, InterpolationMode::Bilinear);
            assert_bit_exact(src, dst, InterpolationMode::Nearest);
        }
    }
}
