//! Native CUDA u8 warp-perspective (bilinear), byte-exact with
//! `warp_perspective_u8`.
//!
//! # Exactness strategy
//!
//! The CPU u8 perspective evaluates the source coordinate DIRECTLY per
//! column (`nd = nd0 + dnd*x`, `inv_nd = 1.0/nd`, `xf = nx*inv_nd` — see
//! `warp::kernels::perspective_coord_at`), identically on scalar, NEON, and
//! AVX2. This kernel reproduces the same expression tree under
//! `--fmad=false` (plain mul + add, exact IEEE division), the same
//! truncating Q10 quantization, and the same Q10 integer blend, so device
//! output is bit-identical to every CPU backend.
//!
//! Per row the CPU classifies: uniform-sign `nd` rows get an analytic valid
//! span (zeros outside), with numerators negated when `nd < 0`; rows where
//! `nd` changes sign fall back to per-pixel bounds checks on the raw
//! parameters. One thread per block-row mirrors that classification and the
//! `warp::span` constraint math (eps = 0) into shared memory. Every sampled
//! pixel then goes through the bounds-checked sampler — for in-bounds
//! coordinates it computes the same bytes as the CPU's valid-path sampler,
//! and at the span's 1-pixel safety margins and on fallback rows it matches
//! the CPU's `bilinear_sample_u8` (zero outside, replicate at the far edge).
//!
//! The launcher takes the ALREADY-INVERTED homography: the adapter inverts
//! with the same `inverse_perspective_matrix` the CPU path calls.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::resize::CudaResizeError as CudaWarpU8Error;
use super::{make_config, try_compile_with_l1};

fn perspective_u8_src(channels: usize) -> String {
    format!(
        r#"
#define C {channels}
// Mirror of warp::span::constrain_span (see cuda/warp_affine_u8.rs).
__device__ __forceinline__ void constrain_span(
    float a, float b, bool ge, float eps, long long* lo, long long* hi
) {{
    if (fabsf(a) < eps || a == 0.0f) {{
        bool feasible = ge ? (b >= 0.0f) : (b < 0.0f);
        if (!feasible) {{ *hi = *lo; }}
        return;
    }}
    float k = -b / a;
    if (ge) {{
        if (a > 0.0f) {{ long long v = (long long)ceilf(k);      if (v > *lo) *lo = v; }}
        else          {{ long long v = (long long)floorf(k) + 1; if (v < *hi) *hi = v; }}
    }} else {{
        if (a > 0.0f) {{ long long v = (long long)ceilf(k);      if (v < *hi) *hi = v; }}
        else          {{ long long v = (long long)floorf(k) + 1; if (v > *lo) *lo = v; }}
    }}
}}

extern "C" __global__ void warp_perspective_u8_bilinear_c{channels}(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    float m0, float m1, float m2,   // inverse homography, row-major
    float m3, float m4, float m5,
    float m6, float m7, float m8,
    int   src_w, int src_h,
    unsigned int dst_w, unsigned int dst_h
) {{
    // Row prologue: mode 0 = fallback (nd changes sign; bounds-check every
    // pixel on raw params), 1 = uniform positive nd, 2 = uniform negative
    // nd (numerators negated, exactly like the CPU).
    __shared__ int s_xlo[8];
    __shared__ int s_xhi[8];
    __shared__ int s_mode[8];

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadIdx.x == 0 && y < dst_h) {{
        float y_f = (float)y;
        float nx0 = m1 * y_f + m2;
        float ny0 = m4 * y_f + m5;
        float nd0 = m7 * y_f + m8;
        float nd_end = nd0 + m6 * ((float)dst_w - 1.0f);
        bool uniform_pos = nd0 > 1e-6f && nd_end > 1e-6f;
        bool uniform_neg = nd0 < -1e-6f && nd_end < -1e-6f;

        if (!(uniform_pos || uniform_neg)) {{
            s_mode[threadIdx.y] = 0;
            s_xlo[threadIdx.y] = 0;
            s_xhi[threadIdx.y] = (int)dst_w;
        }} else {{
            float sgn = uniform_pos ? 1.0f : -1.0f;
            float NX0 = sgn * nx0, NY0 = sgn * ny0, ND0 = sgn * nd0;
            float DNX = sgn * m0, DNY = sgn * m3, DND = sgn * m6;
            float sw = (float)src_w, sh = (float)src_h;

            long long lo = 0, hi = (long long)dst_w;
            constrain_span(DNX, NX0, true, 0.0f, &lo, &hi);
            constrain_span(DNX - sw * DND, NX0 - sw * ND0, false, 0.0f, &lo, &hi);
            constrain_span(DNY, NY0, true, 0.0f, &lo, &hi);
            constrain_span(DNY - sh * DND, NY0 - sh * ND0, false, 0.0f, &lo, &hi);

            long long lo_c = min(max(lo, 0ll), (long long)dst_w);
            long long hi_c = min(max(hi, 0ll), (long long)dst_w);
            bool empty = lo_c >= hi_c;
            s_mode[threadIdx.y] = uniform_pos ? 1 : 2;
            s_xlo[threadIdx.y] = empty ? 0 : (int)lo_c;
            s_xhi[threadIdx.y] = empty ? 0 : (int)hi_c;
        }}
    }}
    __syncthreads();

    if (x >= dst_w || y >= dst_h) return;

    size_t d = ((size_t)y * dst_w + x) * C;
    int mode = s_mode[threadIdx.y];
    if (mode != 0 && ((int)x < s_xlo[threadIdx.y] || (int)x >= s_xhi[threadIdx.y])) {{
        #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) dst[d + ch] = 0;
        return;
    }}

    // Direct per-pixel coordinate — the perspective_coord_at expression
    // tree, on the (possibly negated) row parameters.
    float y_f = (float)y;
    float sgn = (mode == 2) ? -1.0f : 1.0f;
    float nx0 = sgn * (m1 * y_f + m2);
    float ny0 = sgn * (m4 * y_f + m5);
    float nd0 = sgn * (m7 * y_f + m8);
    float dnx = sgn * m0, dny = sgn * m3, dnd = sgn * m6;

    float x_f = (float)x;
    float nx = nx0 + dnx * x_f;
    float ny = ny0 + dny * x_f;
    float nd = nd0 + dnd * x_f;
    float inv_nd = 1.0f / nd;
    float xf = nx * inv_nd;
    float yf = ny * inv_nd;

    // Bounds-checked Q10 sampler — mirror of warp::common::bilinear_sample_u8.
    int xi = (int)floorf(xf);
    int yi = (int)floorf(yf);
    if (xi < 0 || xi >= src_w || yi < 0 || yi >= src_h) {{
        #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) dst[d + ch] = 0;
        return;
    }}
    int xi1 = (xi + 1 < src_w) ? xi + 1 : xi;
    int yi1 = (yi + 1 < src_h) ? yi + 1 : yi;
    unsigned int fx = (unsigned int)((xf - (float)xi) * 1024.0f);
    unsigned int fy = (unsigned int)((yf - (float)yi) * 1024.0f);
    unsigned int fx1 = 1024u - fx;
    unsigned int fy1 = 1024u - fy;

    size_t row0 = (size_t)yi * src_w * C;
    size_t row1 = (size_t)yi1 * src_w * C;
    size_t off0 = (size_t)xi * C;
    size_t off1 = (size_t)xi1 * C;

    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) {{
        unsigned int p00 = (unsigned int)__ldg(&src[row0 + off0 + ch]);
        unsigned int p01 = (unsigned int)__ldg(&src[row0 + off1 + ch]);
        unsigned int p10 = (unsigned int)__ldg(&src[row1 + off0 + ch]);
        unsigned int p11 = (unsigned int)__ldg(&src[row1 + off1 + ch]);
        unsigned int top = p00 * fx1 + p01 * fx;
        unsigned int bot = p10 * fx1 + p11 * fx;
        dst[d + ch] = (unsigned char)((top * fy1 + bot * fy + (1u << 19)) >> 20);
    }}
}}
"#
    )
}

type PerCKernelCache = Mutex<HashMap<u32, Arc<CudaKernel>>>;
static PERSPECTIVE_U8_KERNELS: OnceLock<PerCKernelCache> = OnceLock::new();

/// Launch the u8 bilinear warp-perspective kernel.
///
/// `m_inv` is the INVERSE (dst→src) homography — invert the forward matrix
/// with `warp::inverse_perspective_matrix` on the host so the f32 inversion
/// is the same code the CPU path runs. `block_dim.1` must be ≤ 8.
#[allow(clippy::too_many_arguments)]
pub fn launch_warp_perspective_u8_bilinear_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    m_inv: &[f32; 9],
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
            "warp_perspective_u8: block_dim.1 must be <= 8 (row-prologue smem)".into(),
        ));
    }
    let need = (dst_width as usize) * (dst_height as usize) * (channels as usize);
    if dst.len() < need {
        return Err(CudaWarpU8Error::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }

    // Per-C specialized kernel (scoped-guard cache lookup).
    let cache = PERSPECTIVE_U8_KERNELS.get_or_init(Default::default);
    let cached = cache
        .lock()
        .expect("warp kernel cache poisoned")
        .get(&channels)
        .cloned();
    let kernel = if let Some(hit) = cached {
        hit
    } else {
        let src_code = perspective_u8_src(channels as usize);
        let name = format!("warp_perspective_u8_bilinear_c{channels}");
        let built =
            Arc::new(try_compile_with_l1(ctx, &src_code, &name).map_err(CudaWarpU8Error::Cuda)?);
        cache
            .lock()
            .expect("warp kernel cache poisoned")
            .entry(channels)
            .or_insert(built)
            .clone()
    };

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
        .arg(&m_inv[6])
        .arg(&m_inv[7])
        .arg(&m_inv[8])
        .arg(&src_w_i)
        .arg(&src_h_i)
        .arg(&dst_width)
        .arg(&dst_height)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaWarpU8Error::Cuda(e.to_string()))
}
