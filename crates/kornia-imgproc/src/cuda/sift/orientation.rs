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

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

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
#define TAIL_FMA {tail_fma}

__device__ __forceinline__ int cv_round_ori(float v) {{ return __float2int_rn(v); }}

extern "C" __global__ void sift_orientation(
    const float* __restrict__ img, int w, int h,
    const float* __restrict__ kp_in, int n_kp, int kp_stride,
    float* __restrict__ out_kp, int* __restrict__ out_count, int max_out)
{{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_kp) return;

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

    // The reference accumulates in TWO loops: a SIMD block over
    // `k <= len - vecsize` and a scalar tail for the last `len % vecsize`
    // samples. The SIMD block rounds `w * mag` into a buffer and then adds it;
    // the scalar tail writes `temphist[bin] += W[k]*Mag[k]`, which the backend's
    // `-ffp-contract=fast` build fuses. So the last `len % 4` samples use an FMA
    // and the rest do not. `len` is only known after the gather, so count it
    // first — the prepass is pure index arithmetic, no loads.
    int len = 0;
    for (int i = -radius; i <= radius; i++) {{
        const int y = rr + i;
        if (y <= 0 || y >= h - 1) continue;
        for (int j = -radius; j <= radius; j++) {{
            const int x = cc + j;
            if (x <= 0 || x >= w - 1) continue;
            len++;
        }}
    }}
    const int tail_from = TAIL_FMA ? (len & ~3) : len;

    // Same traversal order as the reference: rows outer, columns inner.
    int kk = 0;
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
            if (kk >= tail_from) temphist[bin] = __fmaf_rn(wgt, mag, temphist[bin]);
            else                 temphist[bin] += wgt * mag;
            kk++;
        }}
    }}

    // Smooth with [1,4,6,4,1]/16, wrapping. The reference evaluates this as
    // fma(tn2+t2, 1/16, fma(tn1+t1, 4/16, t0*6/16)) — keep that shape.
    float hist[ORI_N];
    for (int i = 0; i < ORI_N; i++) {{
        const float tn2 = temphist[(i - 2 + ORI_N) % ORI_N];
        const float tn1 = temphist[(i - 1 + ORI_N) % ORI_N];
        const float t0  = temphist[i];
        const float t1  = temphist[(i + 1) % ORI_N];
        const float t2  = temphist[(i + 2) % ORI_N];
        hist[i] = __fmaf_rn(tn2 + t2, 1.0f / 16.0f,
                  __fmaf_rn(tn1 + t1, 4.0f / 16.0f, t0 * (6.0f / 16.0f)));
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
            float angle = 360.0f - (360.0f / (float)ORI_N) * bin;
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
        tail_fma = std::env::var("KORNIA_SIFT_TAIL")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(1),
        ori_radius = ORI_RADIUS,
        ori_sig = ORI_SIG_FCTR,
        peak_ratio = ORI_PEAK_RATIO,
        stride = ORI_KP_STRIDE,
    )
}

/// Assign orientations to detected keypoints.
///
/// `kp_in` is the detector's SoA buffer (stride [`super::KP_STRIDE`]); `img` is
/// the Gaussian layer the keypoint was found in. One keypoint may emit several
/// oriented copies, so `out_count` is incremented atomically and may exceed
/// `max_out` — treat that as overflow rather than assuming every hit was stored.
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

    let key = format!(
        "sift_orientation:{}:{}",
        std::env::var("KORNIA_SIFT_PEAK").unwrap_or_default(),
        std::env::var("KORNIA_SIFT_TAIL").unwrap_or_default()
    );
    let kernel = get_or_compile(ctx, &key, orientation_src, "sift_orientation")?;
    let (w_i, h_i, n_i, s_i) = (width as i32, height as i32, n_kp as i32, kp_stride as i32);
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
        .launch_2d(n_kp, 1, make_config(n_kp, 1, Some((64, 1))))
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
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
        for layer in 1..=cfg.n_octave_layers {
            let idx: Vec<usize> = (0..cnt).filter(|&i| kps[i].layer == layer as i32).collect();
            if idx.is_empty() {
                continue;
            }
            let mut packed: Vec<f32> = Vec::with_capacity(idx.len() * KP_STRIDE);
            let raw = stream.clone_dtoh(&d_kp).unwrap();
            for &i in &idx {
                packed.extend_from_slice(&raw[i * KP_STRIDE..(i + 1) * KP_STRIDE]);
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
                }
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
