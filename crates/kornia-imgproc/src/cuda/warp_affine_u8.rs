//! Native CUDA u8 warp-affine (bilinear), byte-exact with `warp_affine_u8`.
//!
//! # Exactness strategy
//!
//! The CPU u8 path (`warp/affine.rs`) is integer fixed-point per row: a
//! per-row f32 anchor `sx_q_lo = ((sx0 + dsx*x_lo) * 65536) as i32` at the
//! valid span's left edge, then Q16 integer increments per column and a Q10
//! bilinear sampler. The span `[x_lo, x_hi)` is PART of the arithmetic (the
//! anchor is evaluated at `x_lo`), so a per-pixel f32 validity predicate
//! cannot reproduce the CPU bytes. Instead the kernel mirrors the CPU row
//! prologue exactly — one thread per block-row runs the `warp::span`
//! constraint math (same f32 expressions, `--fmad=false`) into shared
//! memory, and every thread derives its coordinate as
//! `sx_q = sx_q_lo + (x - x_lo) * dsx_q` in wrapping 32-bit arithmetic,
//! identical to the CPU's repeated `wrapping_add`.
//!
//! The Q10 sampler transcribes `bilinear_sample_u8_valid`'s scalar form:
//! `fx = (sx_q & 0xFFFF) >> 6` (truncation), `xi+1` clamped to the last
//! column/row (BORDER_REPLICATE at the far edge), u32 blend
//! `(top*fy1 + bot*fy + (1<<19)) >> 20` (round-half-up). Outside the span
//! the CPU memsets zero; the kernel writes zeros per pixel.
//!
//! The launcher takes the ALREADY-INVERTED matrix: the adapter inverts with
//! the same `invert_affine_transform` the CPU path calls, so the f32
//! inversion arithmetic is shared, not mirrored.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::resize::CudaResizeError as CudaWarpU8Error;
use super::{make_config, try_compile_with_l1};

static AFFINE_U8_SRC: &str = r#"
// Mirror of warp::span::constrain_span (f32; compiled with --fmad=false so
// -b/a, ceilf, floorf round exactly like the CPU expressions).
__device__ __forceinline__ void constrain_span(
    float a, float b, bool ge, float eps, long long* lo, long long* hi
) {
    if (fabsf(a) < eps || a == 0.0f) {
        bool feasible = ge ? (b >= 0.0f) : (b < 0.0f);
        if (!feasible) { *hi = *lo; }
        return;
    }
    float k = -b / a;
    if (ge) {
        if (a > 0.0f) { long long v = (long long)ceilf(k);      if (v > *lo) *lo = v; }
        else          { long long v = (long long)floorf(k) + 1; if (v < *hi) *hi = v; }
    } else {
        if (a > 0.0f) { long long v = (long long)ceilf(k);      if (v < *hi) *hi = v; }
        else          { long long v = (long long)floorf(k) + 1; if (v > *lo) *lo = v; }
    }
}

extern "C" __global__ void warp_affine_u8_bilinear(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    float m0, float m1, float m2, float m3, float m4, float m5, // inverse map
    int   dsx_q, int dsy_q,       // host-quantized Q16 increments
    int   src_w, int src_h,
    unsigned int dst_w, unsigned int dst_h,
    unsigned int channels
) {
    // Row prologue: one thread per block-row mirrors the CPU span + Q16
    // anchor computation; blockDim.y <= 8 (see make_config).
    __shared__ int s_xlo[8];
    __shared__ int s_xhi[8];
    __shared__ int s_sxq[8];
    __shared__ int s_syq[8];

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadIdx.x == 0 && y < dst_h) {
        float y_f = (float)y;
        float sx0 = m1 * y_f + m2;
        float sy0 = m4 * y_f + m5;

        // affine_valid_span(axes, dst_w, 1e-12): 0 <= d*x + s0 < upper.
        long long lo = 0, hi = (long long)dst_w;
        constrain_span(m0, sx0, true, 1e-12f, &lo, &hi);
        constrain_span(m0, sx0 - (float)src_w, false, 1e-12f, &lo, &hi);
        if (lo < hi) {
            constrain_span(m3, sy0, true, 1e-12f, &lo, &hi);
            constrain_span(m3, sy0 - (float)src_h, false, 1e-12f, &lo, &hi);
        }
        long long lo_c = min(max(lo, 0ll), (long long)dst_w);
        long long hi_c = min(max(hi, 0ll), (long long)dst_w);
        int xlo = (lo >= hi || lo_c >= hi_c) ? 0 : (int)lo_c;
        int xhi = (lo >= hi || lo_c >= hi_c) ? 0 : (int)hi_c;
        s_xlo[threadIdx.y] = xlo;
        s_xhi[threadIdx.y] = xhi;
        // Q16 anchors at x_lo — same f32 expression + truncating cast as CPU.
        s_sxq[threadIdx.y] = (int)((sx0 + m0 * (float)xlo) * 65536.0f);
        s_syq[threadIdx.y] = (int)((sy0 + m3 * (float)xlo) * 65536.0f);
    }
    __syncthreads();

    if (x >= dst_w || y >= dst_h) return;

    size_t d = ((size_t)y * dst_w + x) * channels;
    int xlo = s_xlo[threadIdx.y];
    int xhi = s_xhi[threadIdx.y];
    if ((int)x < xlo || (int)x >= xhi) {
        for (unsigned int ch = 0; ch < channels; ++ch) dst[d + ch] = 0;
        return;
    }

    // Wrapping i32 coordinate, identical to the CPU's repeated wrapping_add.
    unsigned int rel = x - (unsigned int)xlo;
    int sx_q = (int)((unsigned int)s_sxq[threadIdx.y] + rel * (unsigned int)dsx_q);
    int sy_q = (int)((unsigned int)s_syq[threadIdx.y] + rel * (unsigned int)dsy_q);

    int xi = sx_q >> 16;                                  // arithmetic shift
    int yi = sy_q >> 16;
    unsigned int fx = ((unsigned int)(sx_q & 0xFFFF)) >> 6; // Q10, truncation
    unsigned int fy = ((unsigned int)(sy_q & 0xFFFF)) >> 6;
    unsigned int fx1 = 1024u - fx;
    unsigned int fy1 = 1024u - fy;

    int xi1 = (xi + 1 < src_w) ? xi + 1 : xi;             // replicate far edge
    int yi1 = (yi + 1 < src_h) ? yi + 1 : yi;

    size_t row0 = (size_t)yi * src_w * channels;
    size_t row1 = (size_t)yi1 * src_w * channels;
    size_t off0 = (size_t)xi * channels;
    size_t off1 = (size_t)xi1 * channels;

    for (unsigned int ch = 0; ch < channels; ++ch) {
        unsigned int p00 = (unsigned int)__ldg(&src[row0 + off0 + ch]);
        unsigned int p01 = (unsigned int)__ldg(&src[row0 + off1 + ch]);
        unsigned int p10 = (unsigned int)__ldg(&src[row1 + off0 + ch]);
        unsigned int p11 = (unsigned int)__ldg(&src[row1 + off1 + ch]);
        unsigned int top = p00 * fx1 + p01 * fx;
        unsigned int bot = p10 * fx1 + p11 * fx;
        dst[d + ch] = (unsigned char)((top * fy1 + bot * fy + (1u << 19)) >> 20);
    }
}
"#;

static AFFINE_U8_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

/// Launch the u8 bilinear warp-affine kernel.
///
/// `m_inv` is the INVERSE (dst→src) 2×3 matrix — invert the forward matrix
/// with `warp::invert_affine_transform` on the host so the f32 inversion is
/// the same code the CPU path runs. `block_dim.1` must be ≤ 8 (row-prologue
/// shared-memory arrays); the default config satisfies this.
#[allow(clippy::too_many_arguments)]
pub fn launch_warp_affine_u8_bilinear_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    m_inv: &[f32; 6],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaWarpU8Error> {
    super::check_geometry(src_width, src_height, dst_width, dst_height, block_dim)
        .map_err(CudaWarpU8Error::Cuda)?;
    if block_dim.is_some_and(|(_, bh)| bh > 8) {
        return Err(CudaWarpU8Error::Cuda(
            "warp_affine_u8: block_dim.1 must be <= 8 (row-prologue smem)".into(),
        ));
    }
    let need = (dst_width as usize) * (dst_height as usize) * (channels as usize);
    if dst.len() < need {
        return Err(CudaWarpU8Error::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }

    let kernel = AFFINE_U8_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, AFFINE_U8_SRC, "warp_affine_u8_bilinear"))
        .as_ref()
        .map_err(|e| CudaWarpU8Error::Cuda(e.clone()))?;

    // Same Q16 quantization (f32 -> i32 truncating cast) the CPU applies.
    let dsx_q = (m_inv[0] * 65536.0) as i32;
    let dsy_q = (m_inv[3] * 65536.0) as i32;
    let (src_w_i, src_h_i) = (src_width as i32, src_height as i32);

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&m_inv[0])
        .arg(&m_inv[1])
        .arg(&m_inv[2])
        .arg(&m_inv[3])
        .arg(&m_inv[4])
        .arg(&m_inv[5])
        .arg(&dsx_q)
        .arg(&dsy_q)
        .arg(&src_w_i)
        .arg(&src_h_i)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&channels)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaWarpU8Error::Cuda(e.to_string()))
}
