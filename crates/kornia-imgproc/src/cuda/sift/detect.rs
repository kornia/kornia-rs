//! Scale-space extremum detection with sub-pixel refinement.
//!
//! One thread per interior DoG pixel of layers `1..=n_octave_layers`. A thread
//! that survives the 26-neighbour test runs the refinement inline and appends
//! its keypoint with an `atomicAdd` — there is no extrema mask and no separate
//! stream-compaction pass, because survivor density is ~0.1% and the atomic
//! append *is* the compaction.
//!
//! # Numerics
//!
//! The refinement solves a 3x3 system. The reference requests an LU
//! decomposition but its 3x3-by-1 specialisation **ignores the method and uses
//! Cramer's rule**, so that is what is reproduced here — including the
//! cofactor expression order of its determinant helper.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaView, CudaViewMut};

use super::kernels::get_or_compile;
use super::{SiftCudaConfig, SiftCudaError, SIFT_IMG_BORDER, SIFT_MAX_INTERP_STEPS};
use crate::cuda::make_config;

/// Number of `f32` slots each detected keypoint occupies in the output buffer.
pub const KP_STRIDE: usize = 9;

/// A refined keypoint as produced by the detector, before orientation
/// assignment. Field meanings mirror the reference's `KeyPoint`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiftRawKeypoint {
    /// Column in the octave's coordinate frame, scaled by `1 << octave`.
    pub x: f32,
    /// Row, scaled the same way.
    pub y: f32,
    /// Diameter of the meaningful neighbourhood.
    pub size: f32,
    /// `|contrast|` at the refined extremum.
    pub response: f32,
    /// Packed octave field: `octv | (layer << 8) | (round((xi + 0.5) * 255) << 16)`.
    pub octave: i32,
    /// Layer the extremum converged to.
    pub layer: i32,
    /// Sub-scale offset; `size` depends only on this, so it is the sensitive
    /// probe of the refinement solve.
    pub xi: f32,
    /// Integer column the refinement converged to, in the octave's frame.
    /// Orientation samples its patch around this pixel, not the sub-pixel one.
    pub cc: i32,
    /// Integer row the refinement converged to.
    pub rr: i32,
}

/// Extremum-rejection threshold: an *integer*, because the reference floors it
/// before comparing, and its images are f32 in `0..=255` rather than `0..=1`.
pub fn extrema_threshold(cfg: &SiftCudaConfig) -> i32 {
    (0.5 * cfg.contrast_threshold / cfg.n_octave_layers as f64 * 255.0).floor() as i32
}

/// glibc's `exp2f` table, derived as `asuint64(2^(i/32)) - (i << 47)`.
///
/// The reference computes `kpt.size` with host `powf(2.f, y)`, which glibc
/// evaluates as exactly its own `exp2f`. Neither CUDA's `powf` nor CUDA's
/// `exp2f` reproduces it, and the correctly-rounded value does not either —
/// verified against glibc over 200k samples in SIFT's actual exponent range.
fn exp2f_table() -> [u64; 32] {
    let mut t = [0u64; 32];
    for (i, slot) in t.iter_mut().enumerate() {
        let v = (i as f64 / 32.0).exp2();
        *slot = v.to_bits() - ((i as u64) << 47);
    }
    t
}

/// Device source for a bit-exact port of glibc's `exp2f`.
pub(crate) fn exp2f_device_src() -> String {
    let tab = exp2f_table()
        .iter()
        .map(|v| format!("0x{v:016x}ULL"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"
__device__ __forceinline__ float sift_exp2f(float x) {{
    const unsigned long long T[32] = {{ {tab} }};
    const double C0 = 0x1.c6af84b912394p-5;
    const double C1 = 0x1.ebfce50fac4f3p-3;
    const double C2 = 0x1.62e42ff0c52d6p-1;
    const double Shift = 0x1.8p+52 / 32.0;

    const double xd = (double)x;
    double kd = xd + Shift;
    const unsigned long long ki = __double_as_longlong(kd);
    kd -= Shift;
    const double r = xd - kd;
    const unsigned long long t = T[ki % 32ULL] + (ki << 47);
    const double s = __longlong_as_double(t);
    // glibc's aarch64 build contracts these into FMAs; NVRTC runs fmad=false,
    // so they must be requested explicitly.
    const double z = __fma_rn(C0, r, C1);
    const double r2 = r * r;
    double y = __fma_rn(C2, r, 1.0);
    y = __fma_rn(z, r2, y);
    y = y * s;
    return (float)y;
}}
"#
    )
}

fn detect_src(n_octave_layers: usize) -> String {
    let max_steps = SIFT_MAX_INTERP_STEPS;
    let border = SIFT_IMG_BORDER;
    let exp2f_src = exp2f_device_src();
    let inner_v: u32 = std::env::var("KORNIA_SIFT_INNER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let outer_v: u32 = std::env::var("KORNIA_SIFT_OUTER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let det_v: u32 = std::env::var("KORNIA_SIFT_DET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    format!(
        r#"{exp2f_src}
#define DET_V {det_v}
#define INNER_V {inner_v}
#define OUTER_V {outer_v}
#define IMG_BORDER {border}
#define MAX_INTERP_STEPS {max_steps}
#define N_OCTAVE_LAYERS {n_octave_layers}

// The reference's integer rounding is round-to-nearest with ties to EVEN, not
// ties away from zero. `__float2int_rn` has exactly those semantics. Ties are
// vanishingly rare on small images but show up as soon as there are thousands
// of keypoints.
__device__ __forceinline__ int cv_round_f(float v) {{
    return __float2int_rn(v);
}}

// Cofactor expansion in the reference's operand order, with the FMA
// contractions its build performs. For `A*B - C*D` the compiler rounds `C*D`
// then fuses; for `p - q + r` it chains two fused ops left to right.
__device__ __forceinline__ float mul_sub(float a, float b, float c, float d) {{
#if INNER_V == 0
    return __fmaf_rn(a, b, -(c * d));
#elif INNER_V == 1
    return -__fmaf_rn(c, d, -(a * b));
#else
    return a * b - c * d;
#endif
}}

__device__ __forceinline__ float outer3(float p0, float p1, float q0, float q1,
                                        float r0, float r1) {{
#if OUTER_V == 0
    return __fmaf_rn(r0, r1, __fmaf_rn(p0, p1, -(q0 * q1)));
#elif OUTER_V == 1
    return __fmaf_rn(r0, r1, p0 * p1 - q0 * q1);
#elif OUTER_V == 2
    return __fmaf_rn(p0, p1, -(q0 * q1)) + r0 * r1;
#else
    return p0 * p1 - q0 * q1 + r0 * r1;
#endif
}}
__device__ __forceinline__ float det3(
    float a00, float a01, float a02,
    float a10, float a11, float a12,
    float a20, float a21, float a22)
{{
    const float m0 = mul_sub(a11, a22, a21, a12);
    const float m1 = mul_sub(a10, a22, a20, a12);
    const float m2 = mul_sub(a10, a21, a20, a11);
#if DET_V == 0
    return outer3(a00, m0, a01, m1, a02, m2);
#elif DET_V == 1
    return __fmaf_rn(a02, m2, a00 * m0 - a01 * m1);
#elif DET_V == 2
    return __fmaf_rn(a00, m0, -(a01 * m1)) + a02 * m2;
#else
    return a00 * m0 - a01 * m1 + a02 * m2;
#endif
}}

extern "C" __global__ void sift_find_extrema(
    const float* __restrict__ dog,   // base of this octave's DoG stack
    int w, int h, int n_dog,
    int layer0, int octv,
    int threshold,
    float contrast_threshold, float edge_threshold, float sigma,
    float* __restrict__ out_kp,
    int* __restrict__ counter,
    int max_kp)
{{
    const int c = blockIdx.x * blockDim.x + threadIdx.x + IMG_BORDER;
    const int r = blockIdx.y * blockDim.y + threadIdx.y + IMG_BORDER;
    if (c >= w - IMG_BORDER || r >= h - IMG_BORDER) return;

    const long plane = (long)w * h;
    const float* curr = dog + (long)layer0 * plane;
    const float* prev = curr - plane;
    const float* next = curr + plane;

    const float val = curr[r * w + c];
    if (fabsf(val) <= (float)threshold) return;

#define AT(p, rr, cc) ((p)[(rr) * w + (cc)])
    // 8-neighbourhood extrema of the current layer, then of the two adjacent
    // layers including their centre pixels.
    bool ok = false;
    if (val > 0.0f) {{
        float m = AT(curr, r-1, c-1);
        m = fmaxf(m, AT(curr, r-1, c)); m = fmaxf(m, AT(curr, r-1, c+1));
        m = fmaxf(m, AT(curr, r,   c-1)); m = fmaxf(m, AT(curr, r,   c+1));
        m = fmaxf(m, AT(curr, r+1, c-1)); m = fmaxf(m, AT(curr, r+1, c));
        m = fmaxf(m, AT(curr, r+1, c+1));
        if (val >= m) {{
            float mp = AT(prev, r-1, c-1);
            mp = fmaxf(mp, AT(prev, r-1, c)); mp = fmaxf(mp, AT(prev, r-1, c+1));
            mp = fmaxf(mp, AT(prev, r,   c-1)); mp = fmaxf(mp, AT(prev, r, c));
            mp = fmaxf(mp, AT(prev, r,   c+1));
            mp = fmaxf(mp, AT(prev, r+1, c-1)); mp = fmaxf(mp, AT(prev, r+1, c));
            mp = fmaxf(mp, AT(prev, r+1, c+1));
            float mn = AT(next, r-1, c-1);
            mn = fmaxf(mn, AT(next, r-1, c)); mn = fmaxf(mn, AT(next, r-1, c+1));
            mn = fmaxf(mn, AT(next, r,   c-1)); mn = fmaxf(mn, AT(next, r, c));
            mn = fmaxf(mn, AT(next, r,   c+1));
            mn = fmaxf(mn, AT(next, r+1, c-1)); mn = fmaxf(mn, AT(next, r+1, c));
            mn = fmaxf(mn, AT(next, r+1, c+1));
            ok = (val >= mp) && (val >= mn);
        }}
    }} else {{
        float m = AT(curr, r-1, c-1);
        m = fminf(m, AT(curr, r-1, c)); m = fminf(m, AT(curr, r-1, c+1));
        m = fminf(m, AT(curr, r,   c-1)); m = fminf(m, AT(curr, r,   c+1));
        m = fminf(m, AT(curr, r+1, c-1)); m = fminf(m, AT(curr, r+1, c));
        m = fminf(m, AT(curr, r+1, c+1));
        if (val <= m) {{
            float mp = AT(prev, r-1, c-1);
            mp = fminf(mp, AT(prev, r-1, c)); mp = fminf(mp, AT(prev, r-1, c+1));
            mp = fminf(mp, AT(prev, r,   c-1)); mp = fminf(mp, AT(prev, r, c));
            mp = fminf(mp, AT(prev, r,   c+1));
            mp = fminf(mp, AT(prev, r+1, c-1)); mp = fminf(mp, AT(prev, r+1, c));
            mp = fminf(mp, AT(prev, r+1, c+1));
            float mn = AT(next, r-1, c-1);
            mn = fminf(mn, AT(next, r-1, c)); mn = fminf(mn, AT(next, r-1, c+1));
            mn = fminf(mn, AT(next, r,   c-1)); mn = fminf(mn, AT(next, r, c));
            mn = fminf(mn, AT(next, r,   c+1));
            mn = fminf(mn, AT(next, r+1, c-1)); mn = fminf(mn, AT(next, r+1, c));
            mn = fminf(mn, AT(next, r+1, c+1));
            ok = (val <= mp) && (val <= mn);
        }}
    }}
    if (!ok) return;

    // ── Sub-pixel refinement ────────────────────────────────────────────────
    const float img_scale = 1.0f / 255.0f;
    const float deriv_scale = img_scale * 0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale * 0.25f;

    int rr = r, cc = c, layer = layer0;
    float xi = 0.0f, xr = 0.0f, xc = 0.0f, contr = 0.0f;
    int step = 0;

    for (; step < MAX_INTERP_STEPS; step++) {{
        const float* im = dog + (long)layer * plane;
        const float* pv = im - plane;
        const float* nx = im + plane;

        float dDx = (AT(im, rr, cc+1) - AT(im, rr, cc-1)) * deriv_scale;
        float dDy = (AT(im, rr+1, cc) - AT(im, rr-1, cc)) * deriv_scale;
        float dDs = (AT(nx, rr, cc)   - AT(pv, rr, cc))   * deriv_scale;

        float v2 = AT(im, rr, cc) * 2.0f;
        float dxx = (AT(im, rr, cc+1) + AT(im, rr, cc-1) - v2) * second_deriv_scale;
        float dyy = (AT(im, rr+1, cc) + AT(im, rr-1, cc) - v2) * second_deriv_scale;
        float dss = (AT(nx, rr, cc)   + AT(pv, rr, cc)   - v2) * second_deriv_scale;
        float dxy = (AT(im, rr+1, cc+1) - AT(im, rr+1, cc-1)
                   - AT(im, rr-1, cc+1) + AT(im, rr-1, cc-1)) * cross_deriv_scale;
        float dxs = (AT(nx, rr, cc+1) - AT(nx, rr, cc-1)
                   - AT(pv, rr, cc+1) + AT(pv, rr, cc-1)) * cross_deriv_scale;
        float dys = (AT(nx, rr+1, cc) - AT(nx, rr-1, cc)
                   - AT(pv, rr+1, cc) + AT(pv, rr-1, cc)) * cross_deriv_scale;

        // Cramer's rule on H x = dD, in the reference's operand order.
        float d = det3(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);
        if (d == 0.0f) return;
        d = __fdiv_rn(1.0f, d);

        float x0 = d * outer3(dDx, mul_sub(dyy, dss, dys, dys),
                              dxy, mul_sub(dDy, dss, dys, dDs),
                              dxs, mul_sub(dDy, dys, dyy, dDs));
        float x1 = d * outer3(dxx, mul_sub(dDy, dss, dys, dDs),
                              dDx, mul_sub(dxy, dss, dys, dxs),
                              dxs, mul_sub(dxy, dDs, dDy, dxs));
        // x2's first cofactor is written `a11*b2 - b1*a21` in the reference, but
        // x0's third term is its exact negation `b1*a21 - a11*b2`. CSE evaluates
        // the pair ONCE and negates, and under the reference build's contraction
        // the two spellings are not bitwise negations: one rounds `a11*b2` and
        // fuses `b1*a21`, the other does the reverse. Mirror x0's spelling and
        // negate, or x2 lands 1-3 ULP out on ~30% of keypoints.
        float x2 = d * outer3(dxx, -mul_sub(dDy, dys, dyy, dDs),
                              dxy, mul_sub(dxy, dDs, dDy, dxs),
                              dDx, mul_sub(dxy, dys, dyy, dxs));

        xi = -x2; xr = -x1; xc = -x0;

        if (fabsf(xi) < 0.5f && fabsf(xr) < 0.5f && fabsf(xc) < 0.5f) break;

        if (fabsf(xi) > 715827882.0f /* INT_MAX/3 */ ||
            fabsf(xr) > 715827882.0f /* INT_MAX/3 */ ||
            fabsf(xc) > 715827882.0f /* INT_MAX/3 */) return;

        cc += cv_round_f(xc);
        rr += cv_round_f(xr);
        layer += cv_round_f(xi);

        if (layer < 1 || layer > N_OCTAVE_LAYERS ||
            cc < IMG_BORDER || cc >= w - IMG_BORDER ||
            rr < IMG_BORDER || rr >= h - IMG_BORDER) return;
    }}
    if (step >= MAX_INTERP_STEPS) return;

    {{
        const float* im = dog + (long)layer * plane;
        const float* pv = im - plane;
        const float* nx = im + plane;

        float dDx = (AT(im, rr, cc+1) - AT(im, rr, cc-1)) * deriv_scale;
        float dDy = (AT(im, rr+1, cc) - AT(im, rr-1, cc)) * deriv_scale;
        float dDs = (AT(nx, rr, cc)   - AT(pv, rr, cc))   * deriv_scale;
        // Matx::dot accumulates `s += a[i]*b[i]`, which contracts to an FMA chain.
        float t = dDx * xc;
        t = __fmaf_rn(dDy, xr, t);
        t = __fmaf_rn(dDs, xi, t);

        contr = __fmaf_rn(AT(im, rr, cc), img_scale, t * 0.5f);
        if (fabsf(contr) * N_OCTAVE_LAYERS < contrast_threshold) return;

        float v2 = AT(im, rr, cc) * 2.0f;
        float dxx = (AT(im, rr, cc+1) + AT(im, rr, cc-1) - v2) * second_deriv_scale;
        float dyy = (AT(im, rr+1, cc) + AT(im, rr-1, cc) - v2) * second_deriv_scale;
        float dxy = (AT(im, rr+1, cc+1) - AT(im, rr+1, cc-1)
                   - AT(im, rr-1, cc+1) + AT(im, rr-1, cc-1)) * cross_deriv_scale;
        float tr = dxx + dyy;
        float det = mul_sub(dxx, dyy, dxy, dxy);

        if (det <= 0.0f ||
            tr * tr * edge_threshold >= (edge_threshold + 1.0f) * (edge_threshold + 1.0f) * det)
            return;
    }}
#undef AT

    int slot = atomicAdd(counter, 1);
    if (slot >= max_kp) return;

    float* o = out_kp + (long)slot * {KP_STRIDE};
    o[0] = (cc + xc) * (float)(1 << octv);
    o[1] = (rr + xr) * (float)(1 << octv);
    // The reference calls host `powf(2.f, y)`, which glibc evaluates as exactly
    // its own `exp2f(y)` (verified bit-identical over 200k samples). Neither
    // CUDA's `powf` nor CUDA's `exp2f` reproduces it, and the correctly-rounded
    // value (double `exp2`, used here) is also not it — see the plan file: the
    // remaining fix is to port glibc's table-based `exp2f`. This form is the
    // closest available: exact on dog.jpeg, ~1 in 1800 inputs off by 1 ULP.
    const float pw = sift_exp2f(((float)layer + xi) / (float)N_OCTAVE_LAYERS);
    o[2] = sigma * pw * (float)(1 << octv) * 2.0f;
    o[3] = fabsf(contr);
    int packed = octv + (layer << 8) + (cv_round_f((xi + 0.5f) * 255.0f) << 16);
    o[4] = __int_as_float(packed);
    o[5] = __int_as_float(layer);
    o[6] = xi;
    o[7] = __int_as_float(cc);
    o[8] = __int_as_float(rr);
}}
"#,
        KP_STRIDE = KP_STRIDE
    )
}

/// Detect and refine extrema in one octave of the DoG stack.
///
/// `dog` is the octave's contiguous DoG stack (`n_dog` planes of `w * h`).
/// Keypoints are appended to `out_kp` (stride [`KP_STRIDE`]) and `counter` is
/// incremented atomically. The counter may exceed `max_kp`; the caller must
/// treat that as overflow rather than assuming the buffer holds every hit.
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_find_extrema_cuda_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    cfg: &SiftCudaConfig,
    dog: &CudaView<'_, f32>,
    out_kp: &mut CudaViewMut<'_, f32>,
    counter: &mut CudaViewMut<'_, i32>,
    width: u32,
    height: u32,
    n_dog: u32,
    layer: u32,
    octave: u32,
) -> Result<(), SiftCudaError> {
    if width == 0 || height == 0 {
        return Err(SiftCudaError::Geometry(
            "image dimensions must be non-zero".into(),
        ));
    }
    if layer == 0 || layer + 1 >= n_dog {
        return Err(SiftCudaError::Geometry(format!(
            "layer {layer} has no neighbour in a {n_dog}-plane DoG stack"
        )));
    }
    let plane = (width as usize) * (height as usize);
    if dog.len() < plane * n_dog as usize {
        return Err(SiftCudaError::SliceTooSmall {
            got: dog.len(),
            need: plane * n_dog as usize,
        });
    }
    if counter.is_empty() {
        return Err(SiftCudaError::SliceTooSmall {
            got: counter.len(),
            need: 1,
        });
    }
    let max_kp = (out_kp.len() / KP_STRIDE) as i32;

    let border = SIFT_IMG_BORDER as u32;
    if width <= 2 * border || height <= 2 * border {
        return Ok(()); // no interior pixels; nothing to detect
    }
    let (gw, gh) = (width - 2 * border, height - 2 * border);

    let key = format!(
        "sift_find_extrema:{}:{}:{}",
        cfg.n_octave_layers,
        std::env::var("KORNIA_SIFT_INNER").unwrap_or_default(),
        std::env::var("KORNIA_SIFT_OUTER").unwrap_or_default(),
    ) + &std::env::var("KORNIA_SIFT_DET").unwrap_or_default();
    let kernel = get_or_compile(
        ctx,
        &key,
        || detect_src(cfg.n_octave_layers),
        "sift_find_extrema",
    )?;

    let threshold = extrema_threshold(cfg);
    let (w_i, h_i, n_dog_i, layer_i, octv_i) = (
        width as i32,
        height as i32,
        n_dog as i32,
        layer as i32,
        octave as i32,
    );
    let (contrast, edge, sigma) = (
        cfg.contrast_threshold as f32,
        cfg.edge_threshold as f32,
        cfg.sigma as f32,
    );

    kernel
        .launch_builder(stream)
        .arg(dog)
        .arg(&w_i)
        .arg(&h_i)
        .arg(&n_dog_i)
        .arg(&layer_i)
        .arg(&octv_i)
        .arg(&threshold)
        .arg(&contrast)
        .arg(&edge)
        .arg(&sigma)
        .arg(out_kp)
        .arg(counter)
        .arg(&max_kp)
        .launch_2d(gw, gh, make_config(gw, gh, None))
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// Decode the flat keypoint buffer written by the detector.
pub fn decode_keypoints(raw: &[f32], count: usize) -> Vec<SiftRawKeypoint> {
    raw.chunks_exact(KP_STRIDE)
        .take(count)
        .map(|c| SiftRawKeypoint {
            x: c[0],
            y: c[1],
            size: c[2],
            response: c[3],
            octave: c[4].to_bits() as i32,
            layer: c[5].to_bits() as i32,
            xi: c[6],
            cc: c[7].to_bits() as i32,
            rr: c[8].to_bits() as i32,
        })
        .collect()
}

/// Convenience wrapper over [`launch_sift_find_extrema_cuda_view`] for whole buffers.
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_find_extrema_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    cfg: &SiftCudaConfig,
    dog: &CudaSlice<f32>,
    out_kp: &mut CudaSlice<f32>,
    counter: &mut CudaSlice<i32>,
    width: u32,
    height: u32,
    n_dog: u32,
    layer: u32,
    octave: u32,
) -> Result<(), SiftCudaError> {
    launch_sift_find_extrema_cuda_view(
        ctx,
        stream,
        cfg,
        &dog.as_view(),
        &mut out_kp.as_view_mut(),
        &mut counter.as_view_mut(),
        width,
        height,
        n_dog,
        layer,
        octave,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::default_stream;

    fn load_dump(path: &str) -> Option<(usize, usize, Vec<f32>)> {
        let bytes = std::fs::read(path).ok()?;
        let rows = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let cols = i32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let data: Vec<f32> = bytes[8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .take(rows * cols)
            .collect();
        Some((rows, cols, data))
    }

    /// Oracle keypoints: `[i32 n]` then `n` records of
    /// `{f32 x, y, size, angle, response; i32 octave}`.
    fn load_oracle_keypoints(path: &str) -> Vec<(f32, f32, f32, f32, i32)> {
        let b = std::fs::read(path).expect("keypoints dump");
        let n = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        (0..n)
            .map(|i| {
                let o = 4 + i * 24;
                let f =
                    |k: usize| f32::from_le_bytes(b[o + k * 4..o + k * 4 + 4].try_into().unwrap());
                let oct = i32::from_le_bytes(b[o + 20..o + 24].try_into().unwrap());
                (f(0), f(1), f(2), f(4), oct) // x, y, size, response, octave
            })
            .collect()
    }

    /// All oracle directories, colon-separated in `KORNIA_SIFT_ORACLE`.
    fn oracle_dirs() -> Vec<String> {
        std::env::var("KORNIA_SIFT_ORACLE")
            .ok()
            .map(|v| {
                v.split(':')
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// `(x, y, size, response, packed octave)` as compared against the oracle.
    type KpTuple = (f32, f32, f32, f32, i32);

    fn sort_key(k: &(f32, f32, f32, f32, i32)) -> (u32, u32, u32, u32, i32) {
        (
            k.0.to_bits(),
            k.1.to_bits(),
            k.2.to_bits(),
            k.3.to_bits(),
            k.4,
        )
    }

    #[test]
    fn extrema_detection_matches_reference_bitwise() {
        let dirs = oracle_dirs();
        if dirs.is_empty() {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        }
        for dir in dirs {
            eprintln!("  verifying against {dir}");
            extrema_detection_for_oracle(&dir);
        }
    }

    fn extrema_detection_for_oracle(dir: &str) {
        let cfg = SiftCudaConfig::default();
        let stream = default_stream();
        let ctx = &stream.context();

        let (bh, bw, _) = load_dump(&format!("{dir}/base.f32")).expect("base dump");
        let n_octaves = cfg.n_octaves(bh.min(bw));
        let n_dog = cfg.n_octave_layers + 2;

        let mut got: Vec<(f32, f32, f32, f32, i32)> = Vec::new();

        for o in 0..n_octaves {
            // Pack this octave's DoG planes contiguously.
            let mut stack: Vec<f32> = Vec::new();
            let (mut h, mut w) = (0usize, 0usize);
            for i in 0..n_dog {
                let (hh, ww, plane) =
                    load_dump(&format!("{dir}/dog_o{o}_l{i}.f32")).expect("dog dump");
                h = hh;
                w = ww;
                stack.extend_from_slice(&plane);
            }
            if w <= 2 * SIFT_IMG_BORDER as usize || h <= 2 * SIFT_IMG_BORDER as usize {
                continue;
            }

            let d_dog = stream.clone_htod(&stack).unwrap();
            let mut d_kp = stream
                .alloc_zeros::<f32>(cfg.max_keypoints * KP_STRIDE)
                .unwrap();
            let mut d_cnt = stream.clone_htod(&vec![0i32]).unwrap();

            for layer in 1..=cfg.n_octave_layers {
                launch_sift_find_extrema_cuda(
                    ctx,
                    &stream,
                    &cfg,
                    &d_dog,
                    &mut d_kp,
                    &mut d_cnt,
                    w as u32,
                    h as u32,
                    n_dog as u32,
                    layer as u32,
                    o as u32,
                )
                .unwrap();
            }

            let cnt = stream.clone_dtoh(&d_cnt).unwrap()[0] as usize;
            assert!(cnt <= cfg.max_keypoints, "octave {o} overflowed: {cnt}");
            let raw = stream.clone_dtoh(&d_kp).unwrap();
            for kp in decode_keypoints(&raw, cnt) {
                if let Ok(path) = std::env::var("KORNIA_SIFT_XIDUMP") {
                    use std::io::Write;
                    let mut f = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&path)
                        .unwrap();
                    let (px, py, psz) = (kp.x * 0.5, kp.y * 0.5, kp.size * 0.5);
                    // Same columns as verify_adjust.cpp's XIDUMP, so the two can
                    // be joined on (x, y): carrying the converged (layer, r, c)
                    // separates a different refinement PATH from a different
                    // result along the same path.
                    writeln!(
                        f,
                        "{:08x} {:08x} {:08x} {:08x} {} {} {}",
                        px.to_bits(),
                        py.to_bits(),
                        kp.xi.to_bits(),
                        psz.to_bits(),
                        kp.layer,
                        kp.rr,
                        kp.cc
                    )
                    .unwrap();
                }
                // first_octave = -1 post-processing: halve pt and size, rewrite octave.
                let scale = 0.5f32;
                let octave = (kp.octave & !255) | ((kp.octave - 1) & 255);
                got.push((
                    kp.x * scale,
                    kp.y * scale,
                    kp.size * scale,
                    kp.response,
                    octave,
                ));
            }
        }

        let mut want = load_oracle_keypoints(&format!("{dir}/keypoints.bin"));
        // The oracle emits one entry per orientation; the detector emits one per
        // refined extremum, so compare the deduplicated sets.
        want.sort_by_key(sort_key);
        want.dedup_by_key(|k| sort_key(k));
        got.sort_by_key(sort_key);
        got.dedup_by_key(|k| sort_key(k));

        if std::env::var("KORNIA_SIFT_FIELDS").is_ok() {
            // Per-field bit comparison. Match on (x, y) — proven exact — then
            // report which of the remaining fields disagree, with octave/layer.
            use std::collections::HashMap;
            let idx: HashMap<(u32, u32), &KpTuple> = got
                .iter()
                .map(|k| ((k.0.to_bits(), k.1.to_bits()), k))
                .collect();
            let (mut nsz, mut nr, mut noct, mut nmiss) = (0, 0, 0, 0);
            let mut per_oct: HashMap<i32, usize> = HashMap::new();
            for w in &want {
                match idx.get(&(w.0.to_bits(), w.1.to_bits())) {
                    None => nmiss += 1,
                    Some(g) => {
                        let mut bad = false;
                        if g.2.to_bits() != w.2.to_bits() {
                            nsz += 1;
                            bad = true;
                        }
                        if g.3.to_bits() != w.3.to_bits() {
                            nr += 1;
                            bad = true;
                        }
                        if g.4 != w.4 {
                            noct += 1;
                            bad = true;
                        }
                        if bad {
                            let o = (w.4 & 255) as i8 as i32;
                            let l = (w.4 >> 8) & 255;
                            *per_oct.entry(o * 100 + l).or_insert(0) += 1;
                        }
                    }
                }
            }
            eprintln!(
                "  FIELDS: no-xy-match={nmiss} size={nsz} response={nr} octave={noct} (of {})",
                want.len()
            );
            let mut v: Vec<_> = per_oct.into_iter().collect();
            v.sort();
            for (k, c) in v {
                eprintln!("    octave={} layer={}: {c} mismatching", k / 100, k % 100);
            }
        }
        if std::env::var("KORNIA_SIFT_DEBUG").is_ok() {
            let mut out = String::new();
            for w in &want {
                // nearest neighbour in `got` by (x, y)
                let best = got
                    .iter()
                    .min_by(|a, b| {
                        let d = |k: &(f32, f32, f32, f32, i32)| {
                            (k.0 - w.0).powi(2) + (k.1 - w.1).powi(2)
                        };
                        d(a).partial_cmp(&d(b)).unwrap()
                    })
                    .unwrap();
                out.push_str(&format!(
                    "ref x={:.6} y={:.6} sz={:.6} r={:.7} oct={:#x} | got x={:.6} y={:.6} sz={:.6} r={:.7} oct={:#x} | dx={:+.6} dy={:+.6}\n",
                    w.0, w.1, w.2, w.3, w.4, best.0, best.1, best.2, best.3, best.4,
                    best.0 - w.0, best.1 - w.1
                ));
            }
            eprintln!("{out}");
        }

        let missing: Vec<_> = want.iter().filter(|k| !got.contains(k)).collect();
        let extra: Vec<_> = got.iter().filter(|k| !want.contains(k)).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "detector mismatch: {} of {} reference keypoints missing, {} extra.\n  missing: {:?}\n  extra: {:?}",
            missing.len(), want.len(), extra.len(),
            &missing[..missing.len().min(3)], &extra[..extra.len().min(3)]
        );
    }
}
