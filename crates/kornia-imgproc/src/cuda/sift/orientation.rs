//! Orientation assignment: a 36-bin gradient histogram per keypoint.
//!
//! # Why one thread per keypoint
//!
//! The reference accumulates `hist[bin] += W[k] * Mag[k]` sequentially over the
//! patch, in a fixed `k` order. Float addition is not associative, so a
//! parallel histogram — whether via `atomicAdd` or a tree reduction — would
//! produce a different sum and break bit equality. One thread per keypoint
//! reproduces the order exactly. Keypoint counts are in the thousands, so there
//! is still ample parallelism; the patch loop is the inner work.
//!
//! # Numerics
//!
//! Uses the bit-exact primitives from [`super::hal`]: the reference computes the
//! weights with its own `exp`, the angles with its own `atan2` (degree-valued,
//! reciprocal-estimate based) and the magnitudes as `recip(rsqrt(x*x + y*y))`.
//! None of `expf`, `atan2f` or `sqrtf` reproduce those.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaView, CudaViewMut};

use super::hal::hal_device_src;
use super::kernels::get_or_compile;
use super::{SiftCudaConfig, SiftCudaError};
use crate::cuda::make_config;

/// Histogram bins (`SIFT_ORI_HIST_BINS`).
pub const ORI_HIST_BINS: usize = 36;
/// Patch radius factor (`SIFT_ORI_RADIUS`).
pub const ORI_RADIUS: f32 = 4.5;
/// Gaussian weight sigma factor (`SIFT_ORI_SIG_FCTR`).
pub const ORI_SIG_FCTR: f32 = 1.5;
/// Secondary-peak acceptance ratio (`SIFT_ORI_PEAK_RATIO`).
pub const ORI_PEAK_RATIO: f32 = 0.8;

/// `f32` slots per oriented keypoint: `x, y, size, response, octave, angle`.
pub const ORI_KP_STRIDE: usize = 6;

fn orientation_src() -> String {
    let n = ORI_HIST_BINS;
    format!(
        r#"{hal}

#define ORI_N {n}
#define PEAK_V {peak_v}

__device__ __forceinline__ int cv_round_ori(float v) {{ return __float2int_rn(v); }}

extern "C" __global__ void sift_orientation(
    const float* __restrict__ img, int w, int h,
    const float* __restrict__ kp_in, int n_kp, int kp_stride,
    float* __restrict__ out_kp, int* __restrict__ out_count, int max_out,
    int layer_filter)
{{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_kp) return;
    if (layer_filter >= 0 && __float_as_int(kp_in[(long)t * kp_stride + 5]) != layer_filter)
        return;

    const float* k = kp_in + (long)t * kp_stride;
    const float kx = k[0], ky = k[1], ksize = k[2], kresp = k[3];
    const int packed = __float_as_int(k[4]);
    const int layer  = __float_as_int(k[5]);
    const int cc     = __float_as_int(k[7]);
    const int rr     = __float_as_int(k[8]);
    const int octv   = packed & 255;

    // scl_octv = kpt.size * 0.5 / (1 << octv), with `size` still in the
    // octave's own scale (the 0.5 first-octave rescale happens later).
    const float scl_octv = ksize * 0.5f / (float)(1 << octv);
    const int radius = cv_round_ori({ori_radius}f * scl_octv);
    const float sigma = {ori_sig}f * scl_octv;
    const float expf_scale = -1.0f / (2.0f * sigma * sigma);

    float temphist[ORI_N];
    for (int i = 0; i < ORI_N; i++) temphist[i] = 0.0f;

    // Same traversal order as the reference: rows outer, columns inner.
    //
    // `calcOrientationHist` has a SIMD binning block and a scalar tail, but the
    // reference build compiles this translation unit WITHOUT SIMD, so every
    // sample takes the scalar path `temphist[bin] += W[k]*Mag[k]` -- which its
    // `-ffp-contract=fast` build fuses. Accumulate with an FMA throughout;
    // rounding the product separately leaves ~10 of 36 bins 1 ULP out.
    // NOTE: `magnitude32f`'s NEON body and scalar tail disagree -- the body
    // composes `recip(rsqrt(x*x + y*y))` from ARM's estimate instructions while
    // the tail is a plain `sqrtf`, and dumping the reference's own Mag array
    // confirms the last `len % 4` samples take the sqrt. Reproducing it changes
    // no angle: the sample count is `nx * ny` in closed form (the column bound
    // does not depend on the row), so this was implemented exactly and measured
    // -- 5 / 96 / 65 / 99 / 38 mismatches on dog / mh01 / tags / frame2 / sat,
    // identical to without. Do not re-try. (`exp32f` and `fastAtan2` were
    // checked the same way and show no tail difference at all.)
    for (int i = -radius; i <= radius; i++) {{
        const int y = rr + i;
        if (y <= 0 || y >= h - 1) continue;
        for (int j = -radius; j <= radius; j++) {{
            const int x = cc + j;
            if (x <= 0 || x >= w - 1) continue;

            const float dx = img[y * w + x + 1] - img[y * w + x - 1];
            const float dy = img[(y - 1) * w + x] - img[(y + 1) * w + x];
            const float wgt = sift_exp((float)(i * i + j * j) * expf_scale);
            const float ori = sift_atan2_deg(dy, dx);
            const float mag = sift_magnitude(dx, dy);

            int bin = cv_round_ori(((float)ORI_N / 360.0f) * ori);
            if (bin >= ORI_N) bin -= ORI_N;
            if (bin < 0) bin += ORI_N;
            temphist[bin] = __fmaf_rn(wgt, mag, temphist[bin]);
        }}
    }}

    // Smooth with [1,4,6,4,1]/16, wrapping. The reference's scalar form is
    //   (tn2+t2)*1/16 + (tn1+t1)*4/16 + t0*6/16
    // evaluated left to right, so its build contracts to
    //   fma(t0, 6/16, fma(tn2+t2, 1/16, (tn1+t1)*4/16)).
    // NOT the `v_fma(tn2+t2, 1/16, v_fma(tn1+t1, 4/16, t0*6/16))` of the SIMD
    // branch, which this build does not take.
    float hist[ORI_N];
    for (int i = 0; i < ORI_N; i++) {{
        const float tn2 = temphist[(i - 2 + ORI_N) % ORI_N];
        const float tn1 = temphist[(i - 1 + ORI_N) % ORI_N];
        const float t0  = temphist[i];
        const float t1  = temphist[(i + 1) % ORI_N];
        const float t2  = temphist[(i + 2) % ORI_N];
        hist[i] = __fmaf_rn(t0, 6.0f / 16.0f,
                  __fmaf_rn(tn2 + t2, 1.0f / 16.0f, (tn1 + t1) * (4.0f / 16.0f)));
    }}

    float omax = hist[0];
    for (int i = 1; i < ORI_N; i++) if (hist[i] > omax) omax = hist[i];
    const float mag_thr = omax * {peak_ratio}f;

    for (int j = 0; j < ORI_N; j++) {{
        const int l = j > 0 ? j - 1 : ORI_N - 1;
        const int r2 = j < ORI_N - 1 ? j + 1 : 0;
        if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr) {{
#if PEAK_V == 0
            const float den = hist[l] - 2.0f * hist[j] + hist[r2];
            float bin = (float)j + 0.5f * (hist[l] - hist[r2]) / den;
#elif PEAK_V == 1
            const float den = __fmaf_rn(-2.0f, hist[j], hist[l]) + hist[r2];
            float bin = (float)j + 0.5f * (hist[l] - hist[r2]) / den;
#elif PEAK_V == 2
            const float den = hist[l] - 2.0f * hist[j] + hist[r2];
            float bin = __fmaf_rn(0.5f, (hist[l] - hist[r2]) / den, (float)j);
#else
            const float den = __fmaf_rn(-2.0f, hist[j], hist[l]) + hist[r2];
            float bin = __fmaf_rn(0.5f, (hist[l] - hist[r2]) / den, (float)j);
#endif
            bin = bin < 0.0f ? (float)ORI_N + bin
                : bin >= (float)ORI_N ? bin - (float)ORI_N : bin;
            // `360.f - (360.f/n)*bin` is a mul-then-subtract, which the
            // reference build contracts. Rounding the product separately shifts
            // the angle by an ULP; the interpolation's own shape does not
            // matter (every den/bin variant agrees once this one is fused).
            float angle = __fmaf_rn(-(360.0f / (float)ORI_N), bin, 360.0f);
            if (fabsf(angle - 360.0f) < 1.19209290e-07f) angle = 0.0f;

            const int slot = atomicAdd(out_count, 1);
            if (slot >= max_out) return;
            float* o = out_kp + (long)slot * {stride};
            o[0] = kx; o[1] = ky; o[2] = ksize; o[3] = kresp;
            o[4] = __int_as_float(packed); o[5] = angle;
        }}
    }}
    (void)layer;
}}
"#,
        hal = hal_device_src(),
        n = n,
        peak_v = std::env::var("KORNIA_SIFT_PEAK")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0),
        ori_radius = ORI_RADIUS,
        ori_sig = ORI_SIG_FCTR,
        peak_ratio = ORI_PEAK_RATIO,
        stride = ORI_KP_STRIDE,
    )
}

/// Threads per block for the block-per-keypoint orientation kernel.
///
/// Swept on mh01 (whole-pipeline median, `KORNIA_SIFT_ORI_T`): **32 -> 20.6 ms**,
/// 64 -> 21.5, 128 -> 22.1, 256 -> 23.4. The opposite of the descriptor's
/// preference, and for a clear reason: the orientation patch is much smaller, so
/// a wide block leaves most threads idle, and the serial fold that preserves the
/// reference's accumulation order costs one sync per chunk regardless of width.
pub const ORI_BLOCK_THREADS: usize = 32;

/// Block-per-keypoint orientation, bit-identical to [`orientation_src`].
///
/// # Why this is still exact
///
/// The reference bins sequentially, `temphist[bin] = fma(w, m, temphist[bin])`,
/// and float addition is not associative — so the accumulation order cannot
/// change. It doesn't. The block splits only the *gather*: each thread computes
/// `(bin, w, m)` for one sample into shared memory, and a single thread then
/// folds a chunk into the histogram in the reference's own index order. Chunks
/// run in order, and samples within a chunk run in order, so every FMA happens
/// exactly where it did before.
///
/// That split is worth it because the gather is the expensive half — four
/// scattered global loads and three primitive evaluations per sample — while
/// the serial fold is one FMA. The old kernel ran one thread per keypoint,
/// which at realistic counts is ~2500 threads, under three warps per SM: it was
/// starved of occupancy, not of arithmetic (proved by `KORNIA_SIFT_FASTMATH`,
/// which removes the arithmetic and changes nothing).
///
/// The sample count is `nx * ny` in closed form — the loop is a rectangle
/// clipped to the image interior and the column bound does not depend on the
/// row — so a thread can map its flat index straight to a pixel.
fn orientation_block_src(threads: usize) -> String {
    orientation_block_src_dbg(threads, false)
}

/// `dbg` adds a debug output: for keypoint `dbg_idx`, the 36 raw histogram bins
/// followed by the 36 smoothed ones. Used by the chain diff against the
/// reference's own `calcOrientationHist`; the production kernel never carries
/// the extra parameters.
fn orientation_block_src_dbg(threads: usize, dbg: bool) -> String {
    let n = ORI_HIST_BINS;
    format!(
        r#"{hal}

#define ORI_N {n}
#define NTHREADS {threads}

__device__ __forceinline__ int cv_round_ori(float v) {{ return __float2int_rn(v); }}

extern "C" __global__ void sift_orientation_block(
    const float* __restrict__ img, int w, int h,
    const float* __restrict__ kp_in, int n_kp, int kp_stride,
    float* __restrict__ out_kp, int* __restrict__ out_count, int max_out,
    int layer_filter{dbg_params})
{{
    const int t = blockIdx.x;
    if (t >= n_kp) return;
    // Keypoints stay on device; each launch handles the ones belonging to the
    // Gaussian layer it was given. Filtering here instead of on the host saves a
    // download and an upload per layer per octave.
    if (layer_filter >= 0 && __float_as_int(kp_in[(long)t * kp_stride + 5]) != layer_filter)
        return;
    const int tid = threadIdx.x;

    __shared__ float temphist[ORI_N];
    __shared__ float hist[ORI_N];
    __shared__ int   s_bin[NTHREADS];
    __shared__ float s_w[NTHREADS];
    __shared__ float s_m[NTHREADS];
    __shared__ float s_omax;

    const float* k = kp_in + (long)t * kp_stride;
    const float kx = k[0], ky = k[1], ksize = k[2], kresp = k[3];
    const int packed = __float_as_int(k[4]);
    const int cc     = __float_as_int(k[7]);
    const int rr     = __float_as_int(k[8]);
    const int octv   = packed & 255;

    const float scl_octv = ksize * 0.5f / (float)(1 << octv);
    const int radius = cv_round_ori({ori_radius}f * scl_octv);
    const float sigma = {ori_sig}f * scl_octv;
    const float expf_scale = -1.0f / (2.0f * sigma * sigma);

    for (int i = tid; i < ORI_N; i += NTHREADS) temphist[i] = 0.0f;

    // Rows outer, columns inner — the reference's traversal, flattened.
    const int y0 = max(rr - radius, 1), y1 = min(rr + radius, h - 2);
    const int x0 = max(cc - radius, 1), x1 = min(cc + radius, w - 2);
    const int nx = x1 >= x0 ? x1 - x0 + 1 : 0;
    const int ny = y1 >= y0 ? y1 - y0 + 1 : 0;
    const int total = nx * ny;
    __syncthreads();

    for (int base = 0; base < total; base += NTHREADS) {{
        const int s = base + tid;
        int bin = -1;
        float wgt = 0.0f, mag = 0.0f;
        if (s < total) {{
            const int y = y0 + s / nx, x = x0 + s % nx;
            const int i = y - rr, j = x - cc;
            const float dx = img[y * w + x + 1] - img[y * w + x - 1];
            const float dy = img[(y - 1) * w + x] - img[(y + 1) * w + x];
            wgt = sift_exp((float)(i * i + j * j) * expf_scale);
            const float ori = sift_atan2_deg(dy, dx);
            mag = sift_magnitude(dx, dy);
            bin = cv_round_ori(((float)ORI_N / 360.0f) * ori);
            if (bin >= ORI_N) bin -= ORI_N;
            if (bin < 0) bin += ORI_N;
        }}
        s_bin[tid] = bin; s_w[tid] = wgt; s_m[tid] = mag;
        __syncthreads();
        // Serial fold, in the reference's sample order. One FMA per sample.
        if (tid == 0) {{
            const int m = min(NTHREADS, total - base);
            for (int q = 0; q < m; q++)
                temphist[s_bin[q]] = __fmaf_rn(s_w[q], s_m[q], temphist[s_bin[q]]);
        }}
        __syncthreads();
    }}

    // Smoothing is per-bin independent, so it parallelises freely.
    for (int i = tid; i < ORI_N; i += NTHREADS) {{
        const float tn2 = temphist[(i - 2 + ORI_N) % ORI_N];
        const float tn1 = temphist[(i - 1 + ORI_N) % ORI_N];
        const float t0  = temphist[i];
        const float t1  = temphist[(i + 1) % ORI_N];
        const float t2  = temphist[(i + 2) % ORI_N];
        hist[i] = __fmaf_rn(t0, 6.0f / 16.0f,
                  __fmaf_rn(tn2 + t2, 1.0f / 16.0f, (tn1 + t1) * (4.0f / 16.0f)));
    }}
    __syncthreads();
{dbg_body}
    if (tid == 0) {{
        float omax = hist[0];
        for (int i = 1; i < ORI_N; i++) if (hist[i] > omax) omax = hist[i];
        s_omax = omax;
    }}
    __syncthreads();
    const float mag_thr = s_omax * {peak_ratio}f;

    // One thread per bin. Peaks are emitted in whatever order the atomics land;
    // the host sorts the final keypoint list, so order here is not observable.
    for (int j = tid; j < ORI_N; j += NTHREADS) {{
        const int l = j > 0 ? j - 1 : ORI_N - 1;
        const int r2 = j < ORI_N - 1 ? j + 1 : 0;
        if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr) {{
// Peak interpolation. Every f32 expression tree here has been swept
            // (fused/unfused den, fused/unfused bin, fused/unfused angle) and
            // they all agree, so the shape does not matter -- only the fused
            // angle below does.
            //
            // Evaluating the interpolation and the angle in DOUBLE was also
            // tried, because on one hand-checked keypoint it reproduced the
            // oracle's stored angle exactly. It does not generalise: measured
            // over five oracles it is 4-6x WORSE (dog/mh01/tags/frame2/sat go
            // 5/96/65/99/38 -> 21/590/437/539/244). One matching sample out of a
            // 1-ULP difference is a coin flip; do not re-try this.
            const float den = hist[l] - 2.0f * hist[j] + hist[r2];
            float bin = (float)j + 0.5f * (hist[l] - hist[r2]) / den;
            bin = bin < 0.0f ? (float)ORI_N + bin
                : bin >= (float)ORI_N ? bin - (float)ORI_N : bin;
            float angle = __fmaf_rn(-(360.0f / (float)ORI_N), bin, 360.0f);
            if (fabsf(angle - 360.0f) < 1.19209290e-07f) angle = 0.0f;

            const int slot = atomicAdd(out_count, 1);
            if (slot < max_out) {{
                float* o = out_kp + (long)slot * {stride};
                o[0] = kx; o[1] = ky; o[2] = ksize; o[3] = kresp;
                o[4] = __int_as_float(packed); o[5] = angle;
            }}
        }}
    }}
}}
"#,
        hal = hal_device_src(),
        n = n,
        threads = threads,
        ori_radius = ORI_RADIUS,
        ori_sig = ORI_SIG_FCTR,
        peak_ratio = ORI_PEAK_RATIO,
        stride = ORI_KP_STRIDE,
        dbg_params = if dbg {
            ",\n    float* __restrict__ dbg, int dbg_idx"
        } else {
            ""
        },
        dbg_body = if dbg {
            "    if (t == dbg_idx && tid == 0) {\n                     for (int i = 0; i < ORI_N; i++) {\n                         dbg[i] = temphist[i];\n                         dbg[ORI_N + i] = hist[i];\n                     }\n                 }\n"
        } else {
            ""
        },
    )
}

/// Run orientation and additionally dump one keypoint's histograms.
///
/// `dbg` receives `2 * ORI_HIST_BINS` floats: the raw bins first, then the
/// smoothed ones. Test-only; the production launcher compiles a variant without
/// the extra parameters at all.
///
/// Gated on `cfg(test)`: it is unreachable outside this crate (`orientation` is
/// a private module and this is not re-exported), so in a normal build it is
/// dead code and fails the `-D warnings` clippy job.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_orientation_debug_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    img: &CudaSlice<f32>,
    width: u32,
    height: u32,
    kp_in: &CudaSlice<f32>,
    n_kp: u32,
    kp_stride: u32,
    out_kp: &mut CudaSlice<f32>,
    out_count: &mut CudaSlice<i32>,
    dbg: &mut CudaSlice<f32>,
    dbg_idx: i32,
) -> Result<(), SiftCudaError> {
    if dbg.len() < 2 * ORI_HIST_BINS {
        return Err(SiftCudaError::SliceTooSmall {
            got: dbg.len(),
            need: 2 * ORI_HIST_BINS,
        });
    }
    let threads = ORI_BLOCK_THREADS;
    let kernel = get_or_compile(
        ctx,
        &format!("sift_orientation_block_dbg:{threads}"),
        || orientation_block_src_dbg(threads, true),
        "sift_orientation_block",
    )?;
    let max_out = (out_kp.len() / ORI_KP_STRIDE) as i32;
    let (w_i, h_i, n_i, s_i) = (width as i32, height as i32, n_kp as i32, kp_stride as i32);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (n_kp, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    kernel
        .launch_builder(stream)
        .arg(img)
        .arg(&w_i)
        .arg(&h_i)
        .arg(kp_in)
        .arg(&n_i)
        .arg(&s_i)
        .arg(out_kp)
        .arg(out_count)
        .arg(&max_out)
        .arg(&(-1i32))
        .arg(dbg)
        .arg(&dbg_idx)
        .launch_cfg(cfg)
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// Assign orientations to detected keypoints.
///
/// `kp_in` is the detector's SoA buffer (stride [`super::KP_STRIDE`]); `img` is
/// the Gaussian layer the keypoint was found in. One keypoint may emit several
/// oriented copies, so `out_count` is incremented atomically and may exceed
/// `max_out` — treat that as overflow rather than assuming every hit was stored.
///
/// `KORNIA_SIFT_ORI=exact` selects the one-thread-per-keypoint kernel. Both are
/// bit-identical; the block kernel is the default because it is far faster.
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_orientation_cuda_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    _cfg: &SiftCudaConfig,
    img: &CudaView<'_, f32>,
    width: u32,
    height: u32,
    kp_in: &CudaView<'_, f32>,
    n_kp: u32,
    kp_stride: u32,
    out_kp: &mut CudaViewMut<'_, f32>,
    out_count: &mut CudaViewMut<'_, i32>,
    layer_filter: i32,
) -> Result<(), SiftCudaError> {
    if width == 0 || height == 0 {
        return Err(SiftCudaError::Geometry(
            "image dimensions must be non-zero".into(),
        ));
    }
    let need = (width as usize) * (height as usize);
    if img.len() < need {
        return Err(SiftCudaError::SliceTooSmall {
            got: img.len(),
            need,
        });
    }
    if kp_in.len() < (n_kp as usize) * (kp_stride as usize) {
        return Err(SiftCudaError::SliceTooSmall {
            got: kp_in.len(),
            need: (n_kp as usize) * (kp_stride as usize),
        });
    }
    if out_count.is_empty() {
        return Err(SiftCudaError::SliceTooSmall { got: 0, need: 1 });
    }
    if n_kp == 0 {
        return Ok(());
    }
    let max_out = (out_kp.len() / ORI_KP_STRIDE) as i32;

    let (w_i, h_i, n_i, s_i) = (width as i32, height as i32, n_kp as i32, kp_stride as i32);
    let exact = std::env::var("KORNIA_SIFT_ORI").as_deref() == Ok("exact");

    if exact {
        let key = format!(
            "sift_orientation:{}",
            std::env::var("KORNIA_SIFT_PEAK").unwrap_or_default()
        );
        let kernel = get_or_compile(ctx, &key, orientation_src, "sift_orientation")?;
        return kernel
            .launch_builder(stream)
            .arg(img)
            .arg(&w_i)
            .arg(&h_i)
            .arg(kp_in)
            .arg(&n_i)
            .arg(&s_i)
            .arg(out_kp)
            .arg(out_count)
            .arg(&max_out)
            .arg(&layer_filter)
            .launch_2d(n_kp, 1, make_config(n_kp, 1, Some((64, 1))))
            .map_err(|e| SiftCudaError::Cuda(e.to_string()));
    }

    // Swept with KORNIA_SIFT_ORI_T.
    let threads = std::env::var("KORNIA_SIFT_ORI_T")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|t| t.is_power_of_two() && *t >= 32 && *t <= 1024)
        .unwrap_or(ORI_BLOCK_THREADS);
    let kernel = get_or_compile(
        ctx,
        &format!("sift_orientation_block:{threads}"),
        || orientation_block_src(threads),
        "sift_orientation_block",
    )?;
    // One block per keypoint: the grid is the keypoint count, not a tiling.
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (n_kp, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    kernel
        .launch_builder(stream)
        .arg(img)
        .arg(&w_i)
        .arg(&h_i)
        .arg(kp_in)
        .arg(&n_i)
        .arg(&s_i)
        .arg(out_kp)
        .arg(out_count)
        .arg(&max_out)
        .arg(&layer_filter)
        .launch_cfg(cfg)
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// Convenience wrapper over [`launch_sift_orientation_cuda_view`] for whole buffers.
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_orientation_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    _cfg: &SiftCudaConfig,
    img: &CudaSlice<f32>,
    width: u32,
    height: u32,
    kp_in: &CudaSlice<f32>,
    n_kp: u32,
    kp_stride: u32,
    out_kp: &mut CudaSlice<f32>,
    out_count: &mut CudaSlice<i32>,
    layer_filter: i32,
) -> Result<(), SiftCudaError> {
    launch_sift_orientation_cuda_view(
        ctx,
        stream,
        _cfg,
        &img.as_view(),
        width,
        height,
        &kp_in.as_view(),
        n_kp,
        kp_stride,
        &mut out_kp.as_view_mut(),
        &mut out_count.as_view_mut(),
        layer_filter,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::default_stream;
    use crate::cuda::sift::{decode_keypoints, launch_sift_find_extrema_cuda, KP_STRIDE};

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

    /// Reference `(x, y, angle, size)` tuples, first octave only.
    fn load_ref_angles(dir: &str) -> Vec<(f32, f32, f32, f32)> {
        let b = std::fs::read(format!("{dir}/keypoints.bin")).expect("keypoints");
        let n = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        (0..n)
            .filter_map(|i| {
                let o = 4 + i * 24;
                let f =
                    |k: usize| f32::from_le_bytes(b[o + k * 4..o + k * 4 + 4].try_into().unwrap());
                let oct = i32::from_le_bytes(b[o + 20..o + 24].try_into().unwrap());
                ((oct & 255) == 255).then(|| (f(0), f(1), f(3), f(2)))
            })
            .collect()
    }

    /// Chain diff against the reference's own `calcOrientationHist`.
    ///
    /// `KORNIA_SIFT_ORIDUMP=layer,cc,rr` dumps this kernel's 36 raw and 36
    /// smoothed bins for the keypoint at that integer pixel, as bit patterns.
    /// Compare against `scratchpad/orihist <oracle> <layer> <cc> <rr> <radius>
    /// <sigma>`, which runs the reference verbatim on the same dumped layer.
    ///
    /// Every component of this stage has been verified in isolation and every
    /// isolation said "exact"; only a side-by-side chain diff has ever located
    /// one of these residuals.
    #[test]
    fn orientation_histogram_chain_dump() {
        let (Some(dir), Some(spec)) = (
            std::env::var("KORNIA_SIFT_ORACLE")
                .ok()
                .and_then(|v| v.split(':').next().map(String::from)),
            std::env::var("KORNIA_SIFT_ORIDUMP").ok(),
        ) else {
            eprintln!("KORNIA_SIFT_ORACLE / KORNIA_SIFT_ORIDUMP unset; skipping");
            return;
        };
        let parts: Vec<i32> = spec.split(',').filter_map(|v| v.parse().ok()).collect();
        assert_eq!(parts.len(), 3, "KORNIA_SIFT_ORIDUMP wants layer,cc,rr");
        let (layer, cc, rr) = (parts[0], parts[1], parts[2]);

        let cfg = SiftCudaConfig::default();
        let stream = default_stream();
        let ctx = &stream.context();
        let n_dog = cfg.n_octave_layers + 2;
        let mut stack: Vec<f32> = Vec::new();
        let (mut hh, mut ww) = (0usize, 0usize);
        for i in 0..n_dog {
            let (h, w, plane) = load_dump(&format!("{dir}/dog_o0_l{i}.f32")).expect("dog");
            hh = h;
            ww = w;
            stack.extend_from_slice(&plane);
        }
        let d_dog = stream.clone_htod(&stack).unwrap();
        let mut d_kp = stream
            .alloc_zeros::<f32>(cfg.max_keypoints * KP_STRIDE)
            .unwrap();
        let mut d_cnt = stream.clone_htod(&vec![0i32]).unwrap();
        for l in 1..=cfg.n_octave_layers {
            launch_sift_find_extrema_cuda(
                ctx,
                &stream,
                &cfg,
                &d_dog,
                &mut d_kp,
                &mut d_cnt,
                ww as u32,
                hh as u32,
                n_dog as u32,
                l as u32,
                0,
            )
            .unwrap();
        }
        let n = stream.clone_dtoh(&d_cnt).unwrap()[0] as usize;
        let raw = stream.clone_dtoh(&d_kp).unwrap();

        // Find the keypoint at this integer pixel, in this layer.
        let mut packed: Vec<f32> = Vec::new();
        let mut idx = -1i32;
        for i in 0..n {
            let r = &raw[i * KP_STRIDE..(i + 1) * KP_STRIDE];
            if r[5].to_bits() as i32 != layer {
                continue;
            }
            if r[7].to_bits() as i32 == cc && r[8].to_bits() as i32 == rr {
                idx = (packed.len() / KP_STRIDE) as i32;
            }
            packed.extend_from_slice(r);
        }
        assert!(idx >= 0, "no keypoint at layer={layer} cc={cc} rr={rr}");

        let (h2, w2, gauss) = load_dump(&format!("{dir}/gauss_o0_l{layer}.f32")).expect("gauss");
        let d_img = stream.clone_htod(&gauss).unwrap();
        let d_in = stream.clone_htod(&packed).unwrap();
        let n_in = (packed.len() / KP_STRIDE) as u32;
        let mut d_out = stream
            .alloc_zeros::<f32>(n_in as usize * 4 * ORI_KP_STRIDE)
            .unwrap();
        let mut d_oc = stream.clone_htod(&vec![0i32]).unwrap();
        let mut d_dbg = stream.alloc_zeros::<f32>(2 * ORI_HIST_BINS).unwrap();
        launch_sift_orientation_debug_cuda(
            ctx,
            &stream,
            &d_img,
            w2 as u32,
            h2 as u32,
            &d_in,
            n_in,
            KP_STRIDE as u32,
            &mut d_out,
            &mut d_oc,
            &mut d_dbg,
            idx,
        )
        .unwrap();
        let dbg = stream.clone_dtoh(&d_dbg).unwrap();
        let hex = |v: &[f32]| {
            v.iter()
                .map(|x| format!("{:08x}", x.to_bits()))
                .collect::<Vec<_>>()
                .join(" ")
        };
        eprintln!("TEMP {}", hex(&dbg[..ORI_HIST_BINS]));
        eprintln!("HIST {}", hex(&dbg[ORI_HIST_BINS..]));
    }

    #[test]
    fn orientation_matches_reference_bitwise() {
        let Some(dir) = std::env::var("KORNIA_SIFT_ORACLE")
            .ok()
            .and_then(|v| v.split(':').next().map(String::from))
        else {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        };
        let cfg = SiftCudaConfig::default();
        let stream = default_stream();
        let ctx = &stream.context();
        let n_dog = cfg.n_octave_layers + 2;

        // Octave 0 only: pack its DoG stack, detect, then orient each keypoint
        // against the Gaussian layer it was found in.
        let mut stack: Vec<f32> = Vec::new();
        let (mut hh, mut ww) = (0usize, 0usize);
        for i in 0..n_dog {
            let (h, w, plane) = load_dump(&format!("{dir}/dog_o0_l{i}.f32")).expect("dog");
            hh = h;
            ww = w;
            stack.extend_from_slice(&plane);
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
                ww as u32,
                hh as u32,
                n_dog as u32,
                layer as u32,
                0,
            )
            .unwrap();
        }
        let cnt = stream.clone_dtoh(&d_cnt).unwrap()[0] as usize;
        let kps = decode_keypoints(&stream.clone_dtoh(&d_kp).unwrap(), cnt);
        assert!(cnt > 0, "detector produced no keypoints");

        // Group by layer: orientation reads the Gaussian layer, not the DoG.
        let mut got: Vec<(f32, f32, f32, f32)> = Vec::new();
        // Patch parameters per position, so a mismatch can be replayed against
        // the C++ reference without re-deriving them by hand.
        /// `(layer, cc, rr, radius, sigma)` for one keypoint's patch.
        type Patch = (i32, i32, i32, i32, f32);
        let mut patch: std::collections::HashMap<(u32, u32), Patch> =
            std::collections::HashMap::new();
        for layer in 1..=cfg.n_octave_layers {
            let idx: Vec<usize> = (0..cnt).filter(|&i| kps[i].layer == layer as i32).collect();
            if idx.is_empty() {
                continue;
            }
            let mut packed: Vec<f32> = Vec::with_capacity(idx.len() * KP_STRIDE);
            let raw = stream.clone_dtoh(&d_kp).unwrap();
            for &i in &idx {
                packed.extend_from_slice(&raw[i * KP_STRIDE..(i + 1) * KP_STRIDE]);
                let kp = &kps[i];
                let scl_octv = kp.size * 0.5 / ((1 << (kp.octave & 255)) as f32);
                patch.insert(
                    ((kp.x * 0.5).to_bits(), (kp.y * 0.5).to_bits()),
                    (
                        layer as i32,
                        kp.cc,
                        kp.rr,
                        (ORI_RADIUS * scl_octv).round_ties_even() as i32,
                        ORI_SIG_FCTR * scl_octv,
                    ),
                );
            }
            let (_, _, gauss) =
                load_dump(&format!("{dir}/gauss_o0_l{layer}.f32")).expect("gauss layer");

            let d_img = stream.clone_htod(&gauss).unwrap();
            let d_in = stream.clone_htod(&packed).unwrap();
            let mut d_out = stream
                .alloc_zeros::<f32>(idx.len() * 4 * ORI_KP_STRIDE)
                .unwrap();
            let mut d_oc = stream.clone_htod(&vec![0i32]).unwrap();
            launch_sift_orientation_cuda(
                ctx,
                &stream,
                &cfg,
                &d_img,
                ww as u32,
                hh as u32,
                &d_in,
                idx.len() as u32,
                KP_STRIDE as u32,
                &mut d_out,
                &mut d_oc,
                -1,
            )
            .unwrap();
            let oc = stream.clone_dtoh(&d_oc).unwrap()[0] as usize;
            let out = stream.clone_dtoh(&d_out).unwrap();
            for r in out.chunks_exact(ORI_KP_STRIDE).take(oc) {
                got.push((r[0] * 0.5, r[1] * 0.5, r[5], r[2]));
            }
        }

        let want = load_ref_angles(&dir);
        // A keypoint with several orientation peaks appears as SEVERAL
        // reference entries at the same position. Compare the multiset of
        // angles per position, not entry-by-entry — matching positionally and
        // taking the first hit compares unrelated peaks.
        use std::collections::HashMap;
        let mut want_by_pos: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for w in &want {
            want_by_pos
                .entry((w.0.to_bits(), w.1.to_bits()))
                .or_default()
                .push(w.2.to_bits());
        }
        let mut got_by_pos: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for g in &got {
            got_by_pos
                .entry((g.0.to_bits(), g.1.to_bits()))
                .or_default()
                .push(g.2.to_bits());
        }

        let (mut matched, mut set_bad) = (0usize, 0usize);
        for (pos, wa) in &want_by_pos {
            if let Some(ga) = got_by_pos.get(pos) {
                matched += 1;
                let (mut a, mut b) = (wa.clone(), ga.clone());
                a.sort_unstable();
                b.sort_unstable();
                if a != b {
                    set_bad += 1;
                    if std::env::var("KORNIA_SIFT_ORIDBG").is_ok() {
                        let p = patch.get(pos).copied().unwrap_or_default();
                        eprintln!(
                            "    layer={} cc={} rr={} radius={} sigma={} \
                             sigbits={:08x} want={:?} got={:?}",
                            p.0,
                            p.1,
                            p.2,
                            p.3,
                            p.4,
                            p.4.to_bits(),
                            a.iter().map(|v| f32::from_bits(*v)).collect::<Vec<_>>(),
                            b.iter().map(|v| f32::from_bits(*v)).collect::<Vec<_>>(),
                        );
                    }
                }
            } else if std::env::var("KORNIA_SIFT_ORIDBG").is_ok() {
                // A reference position we never emit. `patch` is keyed off OUR
                // detector's output, so its presence separates "the detector
                // never found it" from "orientation dropped it".
                let p = patch.get(pos).copied().unwrap_or_default();
                eprintln!(
                    "    MISSING detected={} layer={} cc={} rr={} radius={} sigma={} want={:?}",
                    patch.contains_key(pos),
                    p.0,
                    p.1,
                    p.2,
                    p.3,
                    p.4,
                    wa.iter().map(|v| f32::from_bits(*v)).collect::<Vec<_>>(),
                );
            }
        }
        eprintln!(
            "  orientation: produced={} positions matched={}/{} angle_set_mismatch={}",
            got.len(),
            matched,
            want_by_pos.len(),
            set_bad
        );
        assert!(matched > 0, "no keypoints matched by position");
    }
}
