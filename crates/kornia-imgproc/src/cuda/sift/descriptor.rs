//! 128-dimensional SIFT descriptor: a 4x4 grid of 8-bin gradient histograms,
//! accumulated with trilinear interpolation.
//!
//! # Why one thread per keypoint
//!
//! Same reason as orientation: the reference scatters each sample into eight
//! histogram cells sequentially, and float addition is not associative. A
//! parallel scatter would change the sums. The per-keypoint work here is large
//! enough (a rotated patch of radius ~`3*scl*sqrt(2)*2.5`) that one thread each
//! still saturates the device at realistic keypoint counts.
//!
//! # Numerics
//!
//! Uses the bit-exact primitives from [`super::hal`] for angle, magnitude and
//! the Gaussian weight. The normalisation tail is the subtle part: the
//! reference computes `nrm2` with a **4-lane FMA accumulation followed by a
//! pairwise reduction**, not a scalar sum — see `sift_descr_nrm2` below.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaView, CudaViewMut};

use super::hal::hal_device_src;
use super::kernels::get_or_compile;
use super::SiftCudaError;
use crate::cuda::make_config;

// The reference's descriptor geometry has one definition, in
// `features::sift::descriptor`, and both backends read it from there. These are
// baked into the kernel source as literals, so a divergence would not be a
// compile error — it would be a silently different descriptor.
/// Grid width (`SIFT_DESCR_WIDTH`).
pub const DESCR_WIDTH: usize = crate::features::sift::descriptor::DESCR_WIDTH;
/// Orientation bins per cell (`SIFT_DESCR_HIST_BINS`).
pub const DESCR_HIST_BINS: usize = crate::features::sift::descriptor::DESCR_HIST_BINS;
/// Descriptor length in floats.
pub const DESCR_LEN: usize = crate::features::DESCR_LEN;
/// Threads per block for the shared-memory descriptor kernel. Must be a power
/// of two: the L2-norm reduction halves the active range each step.
///
/// Swept on mh01 (whole-pipeline median, `KORNIA_SIFT_DESC_T`): 128 -> 25.6 ms,
/// 256 -> 26.3, **512 -> 20.6**, 1024 -> 25.8. The patch is large enough that
/// 512 threads still each get real work, and the wider block cuts the number of
/// shared-memory atomic collisions per histogram bin.
pub const DESC_BLOCK_THREADS: usize = 512;

// ── Falsified descriptor optimisations (do not re-try) ───────────────────────
//
// This stage is ~8.3 ms of a ~20 ms pipeline, and the shared-memory atomics are
// 5.2 ms of that: replacing the eight `atomicAdd`s with plain (racy) adds
// measures 3.1 ms. So the atomics are the cost. Three ways of attacking them
// were implemented and measured, and all three lost:
//
// * **Transposed patch walk** — give consecutive lanes consecutive *rows*, so
//   their rotated bins differ and the atomics stop colliding. 8.3 -> 9.3 ms.
//   The lost coalescing on the image reads outweighs the conflict win.
// * **Replicated histograms** — one copy per warp (`REPL` of 2/4/8), reduced at
//   the end, so atomics contend only within a warp. Worse at every width and
//   block size: best case 9.6 ms, worst 24.7. The extra shared memory costs more
//   occupancy than the contention costs time — the same result this module has
//   now seen seven times.
// * **Run-merged atomics** — each thread takes `P` consecutive samples and sums
//   those landing in the same cell in registers, paying one set of eight atomics
//   per run. The bookkeeping alone regressed the P=1 case to 12.3 ms, and P=4
//   only recovered to 8.7. Net a wash against the plain scatter.
//
// NOT evidence about approximate math: a `KORNIA_SIFT_FASTMATH` knob once
// swapped the HAL for `__expf`/`atan2f`/`sqrtf` here and measured no gain.
// SASS counts later showed why — CUDA's `atan2f` is IEEE-accurate without
// `--use_fast_math` and compiles to 213 instructions against the exact HAL's
// 44, so that path was ~2x SLOWER than the one it replaced. The knob is gone.
// The exact HAL is ~0.73 ms of a 15.9 ms frame: little to win here either way.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DescMode {
    /// The default block kernel.
    Block,
    /// One thread per keypoint, the reference's summation order.
    Exact,
    /// Rotated-frame sampling approximation.
    Fast,
}

/// Read once: the descriptor launcher runs once per (octave, layer) — tens of
/// times per frame — and `env::var` allocates and scans the whole environment.
fn desc_mode() -> DescMode {
    static MODE: std::sync::OnceLock<DescMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("KORNIA_SIFT_DESC").ok().as_deref() {
        Some("exact") => DescMode::Exact,
        Some("fast") => DescMode::Fast,
        _ => DescMode::Block,
    })
}

/// Block size for the block kernel, swept with `KORNIA_SIFT_DESC_T`. The
/// reduction halves the active range each step, so it must stay a power of two.
/// Read once, for the same reason as [`desc_mode`].
fn desc_block_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("KORNIA_SIFT_DESC_T")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|t| t.is_power_of_two() && *t >= 32 && *t <= 1024)
            .unwrap_or(DESC_BLOCK_THREADS)
    })
}

/// Patch scale factor (`SIFT_DESCR_SCL_FCTR`).
pub const DESCR_SCL_FCTR: f32 = crate::features::sift::descriptor::DESCR_SCL_FCTR;
/// Post-normalisation clamp (`SIFT_DESCR_MAG_THR`).
pub const DESCR_MAG_THR: f32 = crate::features::sift::descriptor::DESCR_MAG_THR;
/// Quantisation factor (`SIFT_INT_DESCR_FCTR`).
pub const INT_DESCR_FCTR: f32 = crate::features::sift::descriptor::INT_DESCR_FCTR;

fn descriptor_src() -> String {
    let d = DESCR_WIDTH;
    let n = DESCR_HIST_BINS;
    let histlen = (d + 2) * (d + 2) * (n + 2);
    format!(
        r#"{hal}

#define DD {d}
#define NN {n}
#define HISTLEN {histlen}
#define DLEN {dlen}

__device__ __forceinline__ int cv_round_d(float v) {{ return __float2int_rn(v); }}
__device__ __forceinline__ int cv_floor_d(float v) {{
    const int i = (int)v;
    return i - (v < (float)i);
}}

// The reference accumulates `nrm2` over 4 SIMD lanes with FMA, then reduces
// pairwise: ((l0+l1) + (l2+l3)). A plain scalar sum gives a different value.
__device__ __forceinline__ float sift_descr_nrm2(const float* v, int len) {{
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int k = 0;
    for (; k <= len - 4; k += 4) {{
        a0 = __fmaf_rn(v[k + 0], v[k + 0], a0);
        a1 = __fmaf_rn(v[k + 1], v[k + 1], a1);
        a2 = __fmaf_rn(v[k + 2], v[k + 2], a2);
        a3 = __fmaf_rn(v[k + 3], v[k + 3], a3);
    }}
    float s = (a0 + a1) + (a2 + a3);
    for (; k < len; k++) s += v[k] * v[k];
    return s;
}}

extern "C" __global__ void sift_descriptor(
    const float* __restrict__ img, int w, int h,
    const float* __restrict__ kp_in, int n_kp, int kp_stride,
    float* __restrict__ out_desc,
    const int* __restrict__ range_start, const int* __restrict__ live_count,
    int cap)
{{
    // Same row range as the block kernel: the assembled pipeline calls this per
    // (octave, layer) and every layer owns its own slice of the output, so
    // starting at 0 would make each layer clobber the previous one's rows.
    const int t = range_start[0] + blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= min(min(range_start[0] + n_kp, *live_count), cap)) return;

    const float* k = kp_in + (long)t * kp_stride;
    // Position and scale are supplied in the OCTAVE's coordinate frame.
    const float ptx = k[0], pty = k[1], scl = k[2], ori = k[3];

    const int px = cv_round_d(ptx), py = cv_round_d(pty);
    float cos_t = cosf(ori * (float)(3.14159265358979323846 / 180.0));
    float sin_t = sinf(ori * (float)(3.14159265358979323846 / 180.0));
    const float bins_per_rad = (float)NN / 360.0f;
    const float exp_scale = -1.0f / ((float)DD * (float)DD * 0.5f);
    const float hist_width = {scl_fctr:?}f * scl;
    int radius = cv_round_d(hist_width * 1.4142135623730951f * ((float)DD + 1.0f) * 0.5f);
    const int diag = (int)sqrt((double)w * (double)w + (double)h * (double)h);
    if (radius > diag) radius = diag;
    cos_t /= hist_width;
    sin_t /= hist_width;

    float hist[HISTLEN];
    for (int i = 0; i < HISTLEN; i++) hist[i] = 0.0f;

    for (int i = -radius; i <= radius; i++) {{
        for (int j = -radius; j <= radius; j++) {{
            const float c_rot = j * cos_t - i * sin_t;
            const float r_rot = j * sin_t + i * cos_t;
            const float rbin = r_rot + (float)DD / 2.0f - 0.5f;
            const float cbin = c_rot + (float)DD / 2.0f - 0.5f;
            const int r = py + i, c = px + j;

            if (rbin > -1.0f && rbin < (float)DD && cbin > -1.0f && cbin < (float)DD &&
                r > 0 && r < h - 1 && c > 0 && c < w - 1) {{
                const float dx = img[r * w + c + 1] - img[r * w + c - 1];
                const float dy = img[(r - 1) * w + c] - img[(r + 1) * w + c];
                const float wgt = sift_exp((c_rot * c_rot + r_rot * r_rot) * exp_scale);
                const float ang = sift_atan2_deg(dy, dx);
                const float mag = sift_magnitude(dx, dy);

                float obin = (ang - ori) * bins_per_rad;
                float rb = rbin, cb = cbin;
                int r0 = cv_floor_d(rb), c0 = cv_floor_d(cb), o0 = cv_floor_d(obin);
                rb -= r0; cb -= c0; obin -= o0;
                if (o0 < 0) o0 += NN;
                if (o0 >= NN) o0 -= NN;

                const float m = mag * wgt;
                const float v_r1 = m * rb,      v_r0 = m - v_r1;
                const float v_rc11 = v_r1 * cb, v_rc10 = v_r1 - v_rc11;
                const float v_rc01 = v_r0 * cb, v_rc00 = v_r0 - v_rc01;
                const float v_rco111 = v_rc11 * obin, v_rco110 = v_rc11 - v_rco111;
                const float v_rco101 = v_rc10 * obin, v_rco100 = v_rc10 - v_rco101;
                const float v_rco011 = v_rc01 * obin, v_rco010 = v_rc01 - v_rco011;
                const float v_rco001 = v_rc00 * obin, v_rco000 = v_rc00 - v_rco001;

                // The fold below reads only cell rows/cols 1..DD; the border
                // ring of the (DD+2)x(DD+2) grid is write-only. With r0,c0 in
                // -1..DD-1, 36% of these adds land there — skip them. The
                // guarded adds touch exactly the same live cells in the same
                // order, so this is bitwise identical.
                const int rlo = r0 >= 0, rhi = r0 <= DD - 2;
                const int clo = c0 >= 0, chi = c0 <= DD - 2;
                const int idx = ((r0 + 1) * (DD + 2) + (c0 + 1)) * (NN + 2) + o0;
                if (rlo && clo) {{
                    hist[idx] += v_rco000;
                    hist[idx + 1] += v_rco001;
                }}
                if (rlo && chi) {{
                    hist[idx + (NN + 2)] += v_rco010;
                    hist[idx + (NN + 3)] += v_rco011;
                }}
                if (rhi && clo) {{
                    hist[idx + (DD + 2) * (NN + 2)] += v_rco100;
                    hist[idx + (DD + 2) * (NN + 2) + 1] += v_rco101;
                }}
                if (rhi && chi) {{
                    hist[idx + (DD + 3) * (NN + 2)] += v_rco110;
                    hist[idx + (DD + 3) * (NN + 2) + 1] += v_rco111;
                }}
            }}
        }}
    }}

    // Fold the circular orientation bins back into the d*d*n array.
    float raw[DLEN];
    for (int i = 0; i < DD; i++) {{
        for (int j = 0; j < DD; j++) {{
            const int idx = ((i + 1) * (DD + 2) + (j + 1)) * (NN + 2);
            hist[idx] += hist[idx + NN];
            // The reference also folds o-bin NN+1 into bin 1, but no scatter
            // can write it: o0 wraps to 0..NN-1, so writes reach NN at most.
            // Adding that guaranteed +0.0 is a no-op (all accumulands are
            // non-negative, so no -0.0 exists to be normalised by it).
            for (int kk = 0; kk < NN; kk++)
                raw[(i * DD + j) * NN + kk] = hist[idx + kk];
        }}
    }}

    // Normalise, clamp at MAG_THR, renormalise, scale and saturate to uchar.
    float nrm2 = sift_descr_nrm2(raw, DLEN);
    const float thr = sqrtf(nrm2) * {mag_thr:?}f;

    nrm2 = 0.0f;
    for (int i = 0; i < DLEN; i++) {{
        const float val = fminf(raw[i], thr);
        raw[i] = val;
        nrm2 += val * val;
    }}
    nrm2 = {int_fctr:?}f / fmaxf(sqrtf(nrm2), 1.1920929e-07f);

    float* o = out_desc + (long)t * DLEN;
    for (int i = 0; i < DLEN; i++) {{
        float v = raw[i] * nrm2;
        v = (float)cv_round_d(v);
        o[i] = fminf(fmaxf(v, 0.0f), 255.0f);
    }}
}}
"#,
        hal = hal_device_src(),
        d = d,
        n = n,
        histlen = histlen,
        dlen = DESCR_LEN,
        scl_fctr = DESCR_SCL_FCTR,
        mag_thr = DESCR_MAG_THR,
        int_fctr = INT_DESCR_FCTR,
    )
}

/// Block-per-keypoint descriptor: the 360-float histogram lives in shared
/// memory and the patch loop is split across the block.
///
/// The one-thread-per-keypoint kernel keeps the reference's sequential
/// accumulation order and is therefore bit-exact, but 360 floats per thread is
/// 1440 bytes -- far past any register budget, so it spills to local memory and
/// every histogram update becomes a local round-trip. Measured at 55% of the
/// whole pipeline, ~0.9 ms per keypoint.
///
/// Splitting the patch across a block and accumulating with shared-memory
/// atomics changes the summation order, so this is NOT bit-exact against the
/// reference (float addition is not associative). Orientation keeps the
/// sequential form because its histogram is 36 bins and stays in registers.
/// Floats per histogram cell in the shared-memory kernels.
///
/// The natural stride `NN + 2 = 10` maps the 36 cells onto only 16 of the 32
/// shared-memory banks, so on paper every spatial scatter takes a 2-way bank
/// conflict and an odd stride should fix it. Measured, padding is far worse —
/// descriptor stage, mh01, `KORNIA_SIFT_DESC_OSTRIDE`:
///
/// ```text
/// 10 (packed)  8.7 ms      11  18.4 ms      13  21.1 ms      17  18.2 ms
/// ```
///
/// So bank conflicts are not what these atomics are waiting on; spreading the
/// eight targets over a wider address range costs more than the conflicts did.
/// The knob stays for re-measurement, but do not pad by default.
fn ostride() -> usize {
    // Read once — this is called per fast-descriptor launch.
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("KORNIA_SIFT_DESC_OSTRIDE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v >= DESCR_HIST_BINS + 2)
            .unwrap_or(DESCR_HIST_BINS + 2)
    })
}

fn pack_desc_src() -> String {
    format!(
        r#"
// Oriented-keypoint rows -> descriptor launch input, on device.
//
// `rows[i]` names the ori_kp row backing output slot `i` (the host decides the
// order — retain_best and the (octave, layer) grouping are host-side sorts).
// Packing here means the 6-float rows the device already owns are never
// downloaded, repacked and re-uploaded; only the 4-byte row indices cross the
// bus. Expressions match the previous host pack exactly: 1/2^octv is exact in
// f32, and each product rounds once in both versions.
extern "C" __global__ void __launch_bounds__(256) sift_pack_desc(
    const float* __restrict__ ori_kp, const int* __restrict__ rows, int n,
    float* __restrict__ out)
{{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* k = ori_kp + (long)rows[i] * {ori_stride};
    const int packed = __float_as_int(k[4]);
    const float scale = 1.0f / (float)(1u << (packed & 255));
    float ang = 360.0f - k[5];
    if (fabsf(ang - 360.0f) < 1.1920929e-07f) ang = 0.0f;
    float* o = out + (long)i * {in_stride};
    o[0] = k[0] * scale;
    o[1] = k[1] * scale;
    o[2] = (k[2] * scale) * 0.5f;
    o[3] = ang;
}}
"#,
        ori_stride = 6usize,
        in_stride = DESC_IN_STRIDE,
    )
}

/// Launch [`pack_desc_src`]'s kernel: `n` threads, one output slot each.
pub fn launch_sift_pack_desc_cuda_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    ori_kp: &CudaView<'_, f32>,
    rows: &CudaView<'_, i32>,
    n: u32,
    out: &mut CudaViewMut<'_, f32>,
) -> Result<(), SiftCudaError> {
    if n == 0 {
        return Ok(());
    }
    let kernel = get_or_compile(ctx, "sift_pack_desc", pack_desc_src, "sift_pack_desc")?;
    let n_i = n as i32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (n.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    kernel
        .launch_builder(stream)
        .arg(ori_kp)
        .arg(rows)
        .arg(&n_i)
        .arg(out)
        .launch_cfg(cfg)
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

fn descriptor_block_src(threads: usize) -> String {
    let d = DESCR_WIDTH;
    let n = DESCR_HIST_BINS;
    format!(
        r#"{hal}

#define DD {d}
#define NN {n}
#define HISTLEN ((DD + 2) * (DD + 2) * OSTRIDE)
#define DLEN {dlen}
#define NTHREADS {threads}
#define OSTRIDE {ostride}
// Reduction scratch length: the L2-norm tree reduces DLEN partials, so sizing
// the scratch to the BLOCK inflated static shared memory by NTHREADS/DLEN
// (2048 B at 512 threads) and ran log2(NTHREADS/DLEN) tree levels that only
// added exact zeros. min(NTHREADS, DLEN); both are powers of two.
#define RLEN (NTHREADS < DLEN ? NTHREADS : DLEN)
#define HIST_SCALE 1048576.0f

__device__ __forceinline__ int cv_round_d(float v) {{ return __float2int_rn(v); }}
__device__ __forceinline__ int cv_floor_d(float v) {{ return (int)floorf(v); }}

// Warp-aggregated shared-memory atomic. The trilinear splat below has every
// lane hammering a small set of histogram bins, so lanes of a warp routinely
// target the SAME address and serialize on the atomic unit. Here the lanes
// sharing an address combine their contributions through shuffles and the
// leader issues one atomicAdd, turning up to 32 serialized atomics into one.
//
// Bit equality is preserved: the bins are fixed point (see HIST_SCALE), and
// unsigned integer addition is associative, so summing the peers in lane
// order gives exactly the value the per-lane atomics would have produced in
// any order. This is precisely why the histogram is fixed point and not f32 —
// the same reordering on floats would NOT be reproducible.
//
// Addresses are keyed by their low 32 bits: `histq` is __shared__, the shared
// window is far below 4 GiB, so two distinct bins cannot collide in the key.
//
// __match_any_sync is sm_70+; older devices take the plain per-lane atomic,
// which is slower but produces the identical result.
__device__ __forceinline__ void warp_atomicAdd(unsigned long long* address, unsigned long long val) {{
#if __CUDA_ARCH__ >= 700
    const unsigned int active = __activemask();
    const unsigned int match = __match_any_sync(active, (unsigned int)(size_t)address);
    const int lane = threadIdx.x % 32;
    const int leader = __ffs(match) - 1;

    unsigned long long sum = 0;
    unsigned int peers = match;
    while (peers) {{
        const int peer = __ffs(peers) - 1;
        peers &= ~(1u << peer);
        sum += __shfl_sync(match, val, peer);
    }}

    if (lane == leader) {{
        atomicAdd(address, sum);
    }}
#else
    atomicAdd(address, val);
#endif
}}

extern "C" __global__ void __launch_bounds__(NTHREADS) sift_descriptor_block(
    const float* __restrict__ img, int w, int h,
    const float* __restrict__ kp_in, int n_kp, int kp_stride,
    float* __restrict__ out_desc,
    const int* __restrict__ range_start, const int* __restrict__ live_count,
    int cap)
{{
    // Grid is an upper bound; blocks past the live count retire immediately.
    // `cap` is the launcher's row capacity: the orientation stage's counter is a
    // bare `atomicAdd`, so `*live_count` (and therefore `range_start[0]`) can
    // run past the buffer when a frame overflows. Without this clamp that is an
    // out-of-bounds device write, not just a dropped keypoint.
    const int t = range_start[0] + blockIdx.x;
    if (t >= min(min(range_start[0] + n_kp, *live_count), cap)) return;
    const int tid = threadIdx.x;

    // FIXED-POINT HISTOGRAM, for determinism.
    //
    // These bins were accumulated with float `atomicAdd`, whose completion order is warp
    // scheduling. Float addition is not associative, so the same patch produced different
    // descriptors on every run. Measured: three identical builds of one clip gave three different
    // descriptor digests, and the 0..255 output quantisation does NOT absorb it -- the emitted
    // descriptors themselves differed, which flipped ratio-test outcomes and changed the
    // reconstruction on roughly one run in three.
    //
    // Integer atomicAdd is exact and order-independent, so the sum is now a function of the input
    // alone. It is also MORE accurate than what it replaces: every contribution is a product of
    // weights in [0,1] and a gradient magnitude <= 255*sqrt(2) ~= 361, so a per-bin sum over even a
    // 128x128 patch stays under ~5.9e6. At scale 256 that is ~1.5e9 against `unsigned int`'s 4.29e9
    // (contributions are all non-negative, so the unsigned range is usable), while a float
    // accumulator at that magnitude has an ULP of 0.5 -- 128x coarser than this fixed point's
    // 1/256.
    __shared__ unsigned long long histq[HISTLEN];
    __shared__ float raw[DLEN];
    __shared__ float red[RLEN];

    const float* k = kp_in + (long)t * kp_stride;
    const float ptx = k[0], pty = k[1], scl = k[2], ori = k[3];

    const int px = cv_round_d(ptx), py = cv_round_d(pty);
    float cos_t = cosf(ori * (float)(3.14159265358979323846 / 180.0));
    float sin_t = sinf(ori * (float)(3.14159265358979323846 / 180.0));
    const float bins_per_rad = (float)NN / 360.0f;
    const float exp_scale = -1.0f / ((float)DD * (float)DD * 0.5f);
    const float hist_width = {scl_fctr:?}f * scl;
    int radius = cv_round_d(hist_width * 1.4142135623730951f * ((float)DD + 1.0f) * 0.5f);
    // `diag` bounds the patch to the image; computing it per thread in double
    // would be an FP64 op at 1/32 rate, and the bound needs no more than f32.
    const int diag = (int)sqrtf((float)w * (float)w + (float)h * (float)h);
    if (radius > diag) radius = diag;
    cos_t /= hist_width;
    sin_t /= hist_width;

    for (int i = tid; i < HISTLEN; i += NTHREADS) histq[i] = 0ull;
    __syncthreads();

    // Flatten the patch so the block strides over it; each thread accumulates
    // into shared memory with atomics.
    //
    // (i, j) are carried incrementally instead of recomputed as s/side, s%side:
    // `side` is a runtime value, so those were a ~20-instruction division
    // sequence per SAMPLE (~4.6M/frame). The stride per step is the
    // compile-time NTHREADS, so the carry needs one divide per KEYPOINT to
    // split it, then an add and a wrap test per step. The sample↔thread map is
    // unchanged, and it is not contractual anyway — the atomics already admit
    // any arrival order.
    const int side = 2 * radius + 1;
    const int total = side * side;
    const int step_i = NTHREADS / side, step_j = NTHREADS % side;
    int ii = tid / side, jj = tid % side;
    for (int s = tid; s < total; s += NTHREADS,
         jj += step_j, ii += step_i + (jj >= side), jj -= (jj >= side) ? side : 0) {{
        const int i = ii - radius;
        const int j = jj - radius;
        const float c_rot = j * cos_t - i * sin_t;
        const float r_rot = j * sin_t + i * cos_t;
        const float rbin = r_rot + (float)DD / 2.0f - 0.5f;
        const float cbin = c_rot + (float)DD / 2.0f - 0.5f;
        const int r = py + i, c = px + j;

        if (rbin > -1.0f && rbin < (float)DD && cbin > -1.0f && cbin < (float)DD &&
            r > 0 && r < h - 1 && c > 0 && c < w - 1) {{
            const float dx = img[r * w + c + 1] - img[r * w + c - 1];
            const float dy = img[(r - 1) * w + c] - img[(r + 1) * w + c];
            const float wgt = sift_exp((c_rot * c_rot + r_rot * r_rot) * exp_scale);
            const float ang = sift_atan2_deg(dy, dx);
            const float mag = sift_magnitude(dx, dy);

            float obin = (ang - ori) * bins_per_rad;
            float rb = rbin, cb = cbin;
            int r0 = cv_floor_d(rb), c0 = cv_floor_d(cb), o0 = cv_floor_d(obin);
            rb -= r0; cb -= c0; obin -= o0;
            if (o0 < 0) o0 += NN;
            if (o0 >= NN) o0 -= NN;

            const float m = mag * wgt;
            const float v_r1 = m * rb,      v_r0 = m - v_r1;
            const float v_rc11 = v_r1 * cb, v_rc10 = v_r1 - v_rc11;
            const float v_rc01 = v_r0 * cb, v_rc00 = v_r0 - v_rc01;
            const float v_rco111 = v_rc11 * obin, v_rco110 = v_rc11 - v_rco111;
            const float v_rco101 = v_rc10 * obin, v_rco100 = v_rc10 - v_rco101;
            const float v_rco011 = v_rc01 * obin, v_rco010 = v_rc01 - v_rco011;
            const float v_rco001 = v_rc00 * obin, v_rco000 = v_rc00 - v_rco001;

            // Skip the 36% of adds that land in the write-only border ring —
            // the fold reads cell rows/cols 1..DD only. Fewer shared atomics,
            // and the survivors are the already-contended interior addresses.
            const int rlo = r0 >= 0, rhi = r0 <= DD - 2;
            const int clo = c0 >= 0, chi = c0 <= DD - 2;
            const int idx = ((r0 + 1) * (DD + 2) + (c0 + 1)) * OSTRIDE + o0;
            if (rlo && clo) {{
                warp_atomicAdd(&histq[idx], __float2ull_rn(v_rco000 * HIST_SCALE));
                warp_atomicAdd(&histq[idx + 1], __float2ull_rn(v_rco001 * HIST_SCALE));
            }}
            if (rlo && chi) {{
                warp_atomicAdd(&histq[idx + OSTRIDE], __float2ull_rn(v_rco010 * HIST_SCALE));
                warp_atomicAdd(&histq[idx + OSTRIDE + 1], __float2ull_rn(v_rco011 * HIST_SCALE));
            }}
            if (rhi && clo) {{
                warp_atomicAdd(&histq[idx + (DD + 2) * OSTRIDE], __float2ull_rn(v_rco100 * HIST_SCALE));
                warp_atomicAdd(&histq[idx + (DD + 2) * OSTRIDE + 1], __float2ull_rn(v_rco101 * HIST_SCALE));
            }}
            if (rhi && chi) {{
                warp_atomicAdd(&histq[idx + (DD + 3) * OSTRIDE], __float2ull_rn(v_rco110 * HIST_SCALE));
                warp_atomicAdd(&histq[idx + (DD + 3) * OSTRIDE + 1], __float2ull_rn(v_rco111 * HIST_SCALE));
            }}
        }}
    }}
    __syncthreads();

    // Fold the circular orientation bins back into the d*d*n array.
    // One thread per OUTPUT element (128 active) instead of one per cell (16
    // active, 496 waiting at the barrier). Pure reads of `hist` producing the
    // identical expression per element; the o-bin NN wrap folds into kk == 0
    // only, and o-bin NN+1 is never written (o0 wraps to 0..NN-1) so its fold
    // was a guaranteed +0.0 no-op.
    for (int e = tid; e < DLEN; e += NTHREADS) {{
        const int cell = e / NN, kk = e % NN;
        const int i = cell / DD, j = cell % DD;
        const int idx = ((i + 1) * (DD + 2) + (j + 1)) * OSTRIDE;
        // Back to float exactly once, after the summation is complete, so the ordering of the
        // adds can no longer influence the result.
        raw[e] = kk == 0 ? (float)(histq[idx] + histq[idx + NN]) * (1.0f / HIST_SCALE)
                         : (float)histq[idx + kk] * (1.0f / HIST_SCALE);
    }}
    __syncthreads();

    // Block-reduced L2 norm, clamp at MAG_THR, renormalise.
    float part = 0.0f;
    for (int i = tid; i < DLEN; i += NTHREADS) part = __fmaf_rn(raw[i], raw[i], part);
    if (tid < RLEN) red[tid] = part;
    __syncthreads();
    for (int off = RLEN / 2; off > 0; off >>= 1) {{
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }}
    const float thr = sqrtf(red[0]) * {mag_thr:?}f;
    __syncthreads();

    part = 0.0f;
    for (int i = tid; i < DLEN; i += NTHREADS) {{
        const float val = fminf(raw[i], thr);
        raw[i] = val;
        part = __fmaf_rn(val, val, part);
    }}
    if (tid < RLEN) red[tid] = part;
    __syncthreads();
    for (int off = RLEN / 2; off > 0; off >>= 1) {{
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }}
    const float scale = {int_fctr:?}f / fmaxf(sqrtf(red[0]), 1.1920929e-07f);

    float* o = out_desc + (long)t * DLEN;
    for (int i = tid; i < DLEN; i += NTHREADS) {{
        float v = (float)cv_round_d(raw[i] * scale);
        o[i] = fminf(fmaxf(v, 0.0f), 255.0f);
    }}
}}
"#,
        hal = hal_device_src(),
        d = d,
        n = n,
        dlen = DESCR_LEN,
        threads = threads,
        ostride = ostride(),
        scl_fctr = DESCR_SCL_FCTR,
        mag_thr = DESCR_MAG_THR,
        int_fctr = INT_DESCR_FCTR,
    )
}

/// Samples per descriptor bin per axis in the rotated-frame kernel.
///
/// The sampling grid covers `[-1, DD]` in bin units, so the total is
/// `((DD + 1) * DESC_FAST_SAMP)^2` samples — 400 at the default, and constant
/// regardless of the keypoint's scale.
pub const DESC_FAST_SAMP: usize = 4;

/// Rotated-frame descriptor: **not** bit-exact, and much faster at large scales.
///
/// # What changes
///
/// The reference walks every pixel of an axis-aligned bounding square around the
/// keypoint and rotates each one into descriptor space. The square has side
/// `2 * round(3 * scl * sqrt(2) * 2.5) + 1`, so the sample count grows as
/// `scl^2` — roughly 1850 samples at a typical scale and far more at coarse
/// octaves, and about 28% of them are then thrown away for landing outside the
/// rotated patch.
///
/// This kernel inverts that. It walks a fixed grid *in the rotated frame* and
/// samples the image bilinearly at each point, the way CudaSift and VLFeat do.
/// Two consequences:
///
/// * the sample count is constant — `(5 * DESC_FAST_SAMP)^2` — so cost no longer
///   scales with the keypoint's size, which is where the reference's shape hurts
///   most;
/// * no sample is wasted, because the grid *is* the patch.
///
/// Trilinear interpolation, the Gaussian weighting, the two-pass normalisation
/// and the exact HAL primitives are all unchanged, so this is a sampling
/// approximation rather than a different descriptor.
///
/// # Why it is not the default
///
/// Bilinear resampling and a coarser grid give different bins, so descriptors
/// differ from `cv::SIFT` in more than the last bits. The default kernel stays
/// the one that reproduces the reference.
fn descriptor_fast_src(threads: usize, samp: usize) -> String {
    let d = DESCR_WIDTH;
    let n = DESCR_HIST_BINS;
    format!(
        r#"{hal}

#define DD {d}
#define NN {n}
#define HISTLEN ((DD + 2) * (DD + 2) * OSTRIDE)
#define DLEN {dlen}
#define NTHREADS {threads}
#define SAMP {samp}
// Reduction scratch length: the L2-norm tree reduces DLEN partials, so sizing
// the scratch to the BLOCK inflated static shared memory by NTHREADS/DLEN
// (2048 B at 512 threads) and ran log2(NTHREADS/DLEN) tree levels that only
// added exact zeros. min(NTHREADS, DLEN); both are powers of two.
#define RLEN (NTHREADS < DLEN ? NTHREADS : DLEN)
// The grid spans the bin range the reject below actually admits, `[-1, DD)`,
// which is DD+1 bins wide — NOT DD+2. It was DD+2, so `cb` ran to 4.875 while
// `floor(cb) >= DD` is discarded, and 4 of every 24 steps per axis were
// computed and thrown away: 1 - (5/6)^2 = 30.6% of the samples. The accepted
// set is unchanged, so this costs nothing in quality.
#define NSAMP ((DD + 1) * SAMP)
#define OSTRIDE {ostride}

__device__ __forceinline__ int cv_round_d(float v) {{ return __float2int_rn(v); }}
__device__ __forceinline__ int cv_floor_d(float v) {{ return (int)floorf(v); }}

// Bilinear image sample, clamped at the border.
__device__ __forceinline__ float sift_tex(
    const float* __restrict__ img, int w, int h, float x, float y)
{{
    x = fminf(fmaxf(x, 0.0f), (float)(w - 1));
    y = fminf(fmaxf(y, 0.0f), (float)(h - 1));
    const int x0 = (int)x, y0 = (int)y;
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = x - (float)x0, fy = y - (float)y0;
    const float a = img[y0 * w + x0], b = img[y0 * w + x1];
    const float c = img[y1 * w + x0], e = img[y1 * w + x1];
    const float top = __fmaf_rn(b - a, fx, a);
    const float bot = __fmaf_rn(e - c, fx, c);
    return __fmaf_rn(bot - top, fy, top);
}}

extern "C" __global__ void __launch_bounds__(NTHREADS) sift_descriptor_fast(
    const float* __restrict__ img, int w, int h,
    const float* __restrict__ kp_in, int n_kp, int kp_stride,
    float* __restrict__ out_desc,
    const int* __restrict__ range_start, const int* __restrict__ live_count,
    int cap)
{{
    // Grid is an upper bound; blocks past the live count retire immediately.
    // `cap` is the launcher's row capacity: the orientation stage's counter is a
    // bare `atomicAdd`, so `*live_count` (and therefore `range_start[0]`) can
    // run past the buffer when a frame overflows. Without this clamp that is an
    // out-of-bounds device write, not just a dropped keypoint.
    const int t = range_start[0] + blockIdx.x;
    if (t >= min(min(range_start[0] + n_kp, *live_count), cap)) return;
    const int tid = threadIdx.x;

    __shared__ float hist[HISTLEN];
    __shared__ float raw[DLEN];
    __shared__ float red[RLEN];

    const float* k = kp_in + (long)t * kp_stride;
    const float ptx = k[0], pty = k[1], scl = k[2], ori = k[3];

    const float cos_o = cosf(ori * (float)(3.14159265358979323846 / 180.0));
    const float sin_o = sinf(ori * (float)(3.14159265358979323846 / 180.0));
    const float bins_per_rad = (float)NN / 360.0f;
    const float exp_scale = -1.0f / ((float)DD * (float)DD * 0.5f);
    const float hist_width = {scl_fctr:?}f * scl;

    for (int i = tid; i < HISTLEN; i += NTHREADS) hist[i] = 0.0f;
    __syncthreads();

    // The grid walks descriptor space directly: `cb`/`rb` are bin coordinates,
    // so the cell a sample lands in is decided by the loop, not by the data.
    const float step = 1.0f / (float)SAMP;
    const int total = NSAMP * NSAMP;
    for (int s = tid; s < total; s += NTHREADS) {{
        const float cb = -1.0f + ((float)(s % NSAMP) + 0.5f) * step;
        const float rb = -1.0f + ((float)(s / NSAMP) + 0.5f) * step;
        // Bin coords -> the reference's rotated coords, in units of hist_width.
        const float c_rot = cb - ((float)DD / 2.0f - 0.5f);
        const float r_rot = rb - ((float)DD / 2.0f - 0.5f);
        // ...and back to the image by the inverse rotation.
        const float jx = hist_width * (c_rot * cos_o + r_rot * sin_o);
        const float iy = hist_width * (r_rot * cos_o - c_rot * sin_o);
        const float x = ptx + jx, y = pty + iy;
        if (x < 1.0f || x > (float)(w - 2) || y < 1.0f || y > (float)(h - 2)) continue;

        const float dx = sift_tex(img, w, h, x + 1.0f, y) - sift_tex(img, w, h, x - 1.0f, y);
        const float dy = sift_tex(img, w, h, x, y - 1.0f) - sift_tex(img, w, h, x, y + 1.0f);
        const float wgt = sift_exp((c_rot * c_rot + r_rot * r_rot) * exp_scale);
        const float ang = sift_atan2_deg(dy, dx);
        const float mag = sift_magnitude(dx, dy);

        float obin = (ang - ori) * bins_per_rad;
        float rf = rb, cf = cb;
        int r0 = cv_floor_d(rf), c0 = cv_floor_d(cf), o0 = cv_floor_d(obin);
        rf -= r0; cf -= c0; obin -= o0;
        if (o0 < 0) o0 += NN;
        if (o0 >= NN) o0 -= NN;
        // Retained as a guard although the grid above no longer produces an
        // out-of-range bin: it is the definition of the admitted region, and a
        // future change to NSAMP or SAMP should fail safe rather than scatter
        // outside `hist`.
        if (r0 < -1 || r0 >= DD || c0 < -1 || c0 >= DD) continue;

        // The grid is denser than the reference's pixel walk at small scales and
        // sparser at large ones; rescale so the total weight is comparable.
        const float m = mag * wgt * (step * step);
        const float v_r1 = m * rf,      v_r0 = m - v_r1;
        const float v_rc11 = v_r1 * cf, v_rc10 = v_r1 - v_rc11;
        const float v_rc01 = v_r0 * cf, v_rc00 = v_r0 - v_rc01;
        const float v_rco111 = v_rc11 * obin, v_rco110 = v_rc11 - v_rco111;
        const float v_rco101 = v_rc10 * obin, v_rco100 = v_rc10 - v_rco101;
        const float v_rco011 = v_rc01 * obin, v_rco010 = v_rc01 - v_rco011;
        const float v_rco001 = v_rc00 * obin, v_rco000 = v_rc00 - v_rco001;

        // Same border-ring elision as the block kernel: only cell rows/cols
        // 1..DD are ever read back.
        const int rlo = r0 >= 0, rhi = r0 <= DD - 2;
        const int clo = c0 >= 0, chi = c0 <= DD - 2;
        const int idx = ((r0 + 1) * (DD + 2) + (c0 + 1)) * OSTRIDE + o0;
        if (rlo && clo) {{
            atomicAdd(&hist[idx], v_rco000);
            atomicAdd(&hist[idx + 1], v_rco001);
        }}
        if (rlo && chi) {{
            atomicAdd(&hist[idx + OSTRIDE], v_rco010);
            atomicAdd(&hist[idx + OSTRIDE + 1], v_rco011);
        }}
        if (rhi && clo) {{
            atomicAdd(&hist[idx + (DD + 2) * OSTRIDE], v_rco100);
            atomicAdd(&hist[idx + (DD + 2) * OSTRIDE + 1], v_rco101);
        }}
        if (rhi && chi) {{
            atomicAdd(&hist[idx + (DD + 3) * OSTRIDE], v_rco110);
            atomicAdd(&hist[idx + (DD + 3) * OSTRIDE + 1], v_rco111);
        }}
    }}
    __syncthreads();

    // One thread per OUTPUT element (128 active) instead of one per cell (16
    // active, 496 waiting at the barrier). Pure reads of `hist` producing the
    // identical expression per element; the o-bin NN wrap folds into kk == 0
    // only, and o-bin NN+1 is never written (o0 wraps to 0..NN-1) so its fold
    // was a guaranteed +0.0 no-op.
    for (int e = tid; e < DLEN; e += NTHREADS) {{
        const int cell = e / NN, kk = e % NN;
        const int i = cell / DD, j = cell % DD;
        const int idx = ((i + 1) * (DD + 2) + (j + 1)) * OSTRIDE;
        raw[e] = kk == 0 ? hist[idx] + hist[idx + NN] : hist[idx + kk];
    }}
    __syncthreads();

    float part = 0.0f;
    for (int i = tid; i < DLEN; i += NTHREADS) part = __fmaf_rn(raw[i], raw[i], part);
    if (tid < RLEN) red[tid] = part;
    __syncthreads();
    for (int off = RLEN / 2; off > 0; off >>= 1) {{
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }}
    const float thr = sqrtf(red[0]) * {mag_thr:?}f;
    __syncthreads();

    part = 0.0f;
    for (int i = tid; i < DLEN; i += NTHREADS) {{
        const float val = fminf(raw[i], thr);
        raw[i] = val;
        part = __fmaf_rn(val, val, part);
    }}
    if (tid < RLEN) red[tid] = part;
    __syncthreads();
    for (int off = RLEN / 2; off > 0; off >>= 1) {{
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }}
    const float scale = {int_fctr:?}f / fmaxf(sqrtf(red[0]), 1.1920929e-07f);

    float* o = out_desc + (long)t * DLEN;
    for (int i = tid; i < DLEN; i += NTHREADS) {{
        float v = (float)cv_round_d(raw[i] * scale);
        o[i] = fminf(fmaxf(v, 0.0f), 255.0f);
    }}
}}
"#,
        hal = hal_device_src(),
        d = d,
        n = n,
        dlen = DESCR_LEN,
        threads = threads,
        samp = samp,
        ostride = ostride(),
        scl_fctr = DESCR_SCL_FCTR,
        mag_thr = DESCR_MAG_THR,
        int_fctr = INT_DESCR_FCTR,
    )
}

/// Compute 128-D descriptors for oriented keypoints.
///
/// `kp_in` supplies `(x, y, scl, ori)` per keypoint **in the octave's own
/// coordinate frame** — the caller is responsible for the octave rescale, as
/// the reference does in `calcDescriptors`. `out_desc` must hold
/// `n_kp * DESCR_LEN` floats.
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_descriptor_cuda_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    img: &CudaView<'_, f32>,
    width: u32,
    height: u32,
    kp_in: &CudaView<'_, f32>,
    n_kp: u32,
    kp_stride: u32,
    out_desc: &mut CudaViewMut<'_, f32>,
    fast_descriptor: bool,
    range_start: &CudaView<'_, i32>,
    live_count: &CudaView<'_, i32>,
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
    if n_kp == 0 {
        return Ok(());
    }
    // Every kernel below reads `x, y, scl, ori` from columns 0..=3 of each row.
    // With a shorter stride the last row's read runs past the buffer, so the
    // precondition is enforced here rather than assumed of the caller.
    if (kp_stride as usize) < DESC_IN_STRIDE {
        return Err(SiftCudaError::Geometry(format!(
            "keypoint stride {kp_stride} is smaller than the {DESC_IN_STRIDE} columns the kernel reads"
        )));
    }
    let need_kp = (n_kp as usize) * (kp_stride as usize);
    if kp_in.len() < need_kp {
        return Err(SiftCudaError::SliceTooSmall {
            got: kp_in.len(),
            need: need_kp,
        });
    }
    let need_out = (n_kp as usize) * DESCR_LEN;
    if out_desc.len() < need_out {
        return Err(SiftCudaError::SliceTooSmall {
            got: out_desc.len(),
            need: need_out,
        });
    }

    // Rows the kernel is allowed to touch. `n_kp` only bounds the launch WIDTH;
    // the first row comes from `range_start` on device, so this is the only
    // place that can stop an overflowing count from writing past either buffer.
    let cap = (out_desc.len() / DESCR_LEN).min(kp_in.len() / kp_stride as usize) as i32;

    // `KORNIA_SIFT_DESC=exact` selects the one-thread-per-keypoint kernel, which
    // reproduces the reference's sequential accumulation bit for bit but spills
    // its 360-float histogram to local memory. The default block kernel keeps
    // that histogram in shared memory and is ~an order of magnitude faster, at
    // the cost of a different (still correct) summation order.
    let mode = desc_mode();
    let exact = mode == DescMode::Exact;
    let fast = fast_descriptor || mode == DescMode::Fast;
    let (w_i, h_i, n_i, s_i) = (width as i32, height as i32, n_kp as i32, kp_stride as i32);

    if exact {
        let kernel = get_or_compile(ctx, "sift_descriptor", descriptor_src, "sift_descriptor")?;
        return kernel
            .launch_builder(stream)
            .arg(img)
            .arg(&w_i)
            .arg(&h_i)
            .arg(kp_in)
            .arg(&n_i)
            .arg(&s_i)
            .arg(out_desc)
            .arg(range_start)
            .arg(live_count)
            .arg(&cap)
            .launch_2d(n_kp, 1, make_config(n_kp, 1, Some((64, 1))))
            .map_err(|e| SiftCudaError::Cuda(e.to_string()));
    }

    if fast {
        let kernel = get_or_compile(
            ctx,
            &format!(
                "sift_descriptor_fast:{DESC_BLOCK_THREADS}:{DESC_FAST_SAMP}:{}",
                ostride()
            ),
            || descriptor_fast_src(DESC_BLOCK_THREADS, DESC_FAST_SAMP),
            "sift_descriptor_fast",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_kp, 1, 1),
            block_dim: (DESC_BLOCK_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        return kernel
            .launch_builder(stream)
            .arg(img)
            .arg(&w_i)
            .arg(&h_i)
            .arg(kp_in)
            .arg(&n_i)
            .arg(&s_i)
            .arg(out_desc)
            .arg(range_start)
            .arg(live_count)
            .arg(&cap)
            .launch_cfg(cfg)
            .map_err(|e| SiftCudaError::Cuda(e.to_string()));
    }

    let threads = desc_block_threads();
    let kernel = get_or_compile(
        ctx,
        &format!("sift_descriptor_block:{threads}"),
        || descriptor_block_src(threads),
        "sift_descriptor_block",
    )?;
    // One block per keypoint: the grid is the keypoint count, not a tiling of
    // it, so this cannot go through the 2-D helper.
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
        .arg(out_desc)
        .arg(range_start)
        .arg(live_count)
        .arg(&cap)
        .launch_cfg(cfg)
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// Floats per packed descriptor input row: `x, y, scl, ori`.
pub const DESC_IN_STRIDE: usize = 4;

/// Reorder descriptors on device by a host-computed permutation.
///
/// The final keypoint order — the reference's `removeDuplicatedSorted`, then
/// `retainBest` — is decided on the host, because it is a comparison sort over a
/// few thousand records and the comparator has six tie-break levels. Applying it
/// to the descriptors is the part worth keeping on device: `out[i]` is
/// `src[perm[i]]`, one coalesced 128-float row per keypoint, so the descriptors
/// never have to make a round trip just to be shuffled.
pub fn launch_sift_gather_descriptors_cuda_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaView<'_, f32>,
    perm: &CudaView<'_, i32>,
    n: u32,
    out: &mut CudaViewMut<'_, f32>,
) -> Result<(), SiftCudaError> {
    if n == 0 {
        return Ok(());
    }
    let need = (n as usize) * DESCR_LEN;
    if out.len() < need {
        return Err(SiftCudaError::SliceTooSmall {
            got: out.len(),
            need,
        });
    }
    if perm.len() < n as usize {
        return Err(SiftCudaError::SliceTooSmall {
            got: perm.len(),
            need: n as usize,
        });
    }

    let kernel = get_or_compile(
        ctx,
        "sift_gather_desc",
        || {
            format!(
                r#"
extern "C" __global__ void sift_gather_desc(
    const float* __restrict__ src, const int* __restrict__ perm, int n,
    int n_src, float* __restrict__ out)
{{
    const int i = blockIdx.x;
    if (i >= n) return;
    const int s = perm[i];
    if (s < 0 || s >= n_src) return;
    const float* a = src + (long)s * {dlen};
    float* b = out + (long)i * {dlen};
    for (int c = threadIdx.x; c < {dlen}; c += blockDim.x) b[c] = a[c];
}}
"#,
                dlen = DESCR_LEN
            )
        },
        "sift_gather_desc",
    )?;
    let (n_i, n_src) = (n as i32, (src.len() / DESCR_LEN) as i32);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (n, 1, 1),
        block_dim: (DESCR_LEN as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(perm)
        .arg(&n_i)
        .arg(&n_src)
        .arg(out)
        .launch_cfg(cfg)
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// Convenience wrapper over [`launch_sift_descriptor_cuda_view`] for whole buffers.
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_descriptor_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    img: &CudaSlice<f32>,
    width: u32,
    height: u32,
    kp_in: &CudaSlice<f32>,
    n_kp: u32,
    kp_stride: u32,
    out_desc: &mut CudaSlice<f32>,
    fast_descriptor: bool,
    range_start: &CudaView<'_, i32>,
    live_count: &CudaView<'_, i32>,
) -> Result<(), SiftCudaError> {
    launch_sift_descriptor_cuda_view(
        ctx,
        stream,
        &img.as_view(),
        width,
        height,
        &kp_in.as_view(),
        n_kp,
        kp_stride,
        &mut out_desc.as_view_mut(),
        fast_descriptor,
        range_start,
        live_count,
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

    /// The fast kernel is a sampling approximation, so it is checked by
    /// agreement with the exact one rather than against the oracle's bits: same
    /// normalisation, and descriptors that still point the same way.
    #[test]
    fn fast_descriptor_agrees_with_exact() {
        let Some(dir) = std::env::var("KORNIA_SIFT_ORACLE")
            .ok()
            .and_then(|v| v.split(':').next().map(String::from))
        else {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        };
        let b = std::fs::read(format!("{dir}/keypoints.bin")).expect("keypoints");
        let n = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let stream = default_stream();
        let ctx = &stream.context();

        // Octave -1, layer 1 only: enough keypoints to be meaningful, one layer
        // to load.
        let mut flat: Vec<f32> = Vec::new();
        for i in 0..n {
            let o = 4 + i * 24;
            let f = |k: usize| f32::from_le_bytes(b[o + k * 4..o + k * 4 + 4].try_into().unwrap());
            let packed = i32::from_le_bytes(b[o + 20..o + 24].try_into().unwrap());
            if (packed & 255) != 255 || ((packed >> 8) & 255) != 1 {
                continue;
            }
            let mut angle = 360.0f32 - f(3);
            if (angle - 360.0).abs() < f32::EPSILON {
                angle = 0.0;
            }
            flat.extend_from_slice(&[f(0) * 2.0, f(1) * 2.0, f(2) * 2.0 * 0.5, angle]);
        }
        let n_kp = flat.len() / 4;
        assert!(n_kp > 0, "no octave -1 layer 1 keypoints");
        let (h, w, img) = load_dump(&format!("{dir}/gauss_o0_l1.f32")).expect("gauss");

        let d_img = stream.clone_htod(&img).unwrap();
        let d_kp = stream.clone_htod(&flat).unwrap();
        let d_zero = stream.clone_htod(&vec![0i32]).unwrap();
        let d_n = stream.clone_htod(&vec![n_kp as i32]).unwrap();
        let run = |fast: bool| {
            let mut out = stream.alloc_zeros::<f32>(n_kp * DESCR_LEN).unwrap();
            launch_sift_descriptor_cuda(
                ctx,
                &stream,
                &d_img,
                w as u32,
                h as u32,
                &d_kp,
                n_kp as u32,
                4,
                &mut out,
                fast,
                &d_zero.as_view(),
                &d_n.as_view(),
            )
            .unwrap();
            stream.clone_dtoh(&out).unwrap()
        };
        let slow = run(false);
        let fast = run(true);

        let mut cos: Vec<f32> = Vec::with_capacity(n_kp);
        for i in 0..n_kp {
            let (a, c) = (
                &slow[i * DESCR_LEN..(i + 1) * DESCR_LEN],
                &fast[i * DESCR_LEN..(i + 1) * DESCR_LEN],
            );
            let dot: f32 = a.iter().zip(c).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nc: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            // Every descriptor is L2-normalised to 512 before quantisation.
            assert!(nc > 1.0, "fast descriptor {i} is all zero");
            cos.push(dot / (na * nc));
        }
        cos.sort_by(f32::total_cmp);
        let median = cos[cos.len() / 2];
        eprintln!(
            "  fast vs exact descriptor: n={n_kp} median cos={median:.4} min={:.4}",
            cos[0]
        );
        assert!(
            median > 0.85,
            "fast descriptor diverges from the exact one: median cosine {median}"
        );
    }

    /// Drive the descriptor from the REFERENCE keypoints, so this test isolates
    /// the descriptor from any upstream detector/orientation residual.
    #[test]
    fn descriptor_matches_reference_bitwise() {
        let Some(dir) = std::env::var("KORNIA_SIFT_ORACLE")
            .ok()
            .and_then(|v| v.split(':').next().map(String::from))
        else {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        };
        let b = std::fs::read(format!("{dir}/keypoints.bin")).expect("keypoints");
        let n = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let (_, dcols, desc_ref) =
            load_dump(&format!("{dir}/descriptors.f32")).expect("descriptors");
        assert_eq!(dcols, DESCR_LEN);

        let stream = default_stream();
        let ctx = &stream.context();

        // Reference keypoints carry the FINAL (first-octave-rescaled) values;
        // undo that to get the octave frame the descriptor works in.
        let mut rows: Vec<(usize, i32, [f32; 4])> = Vec::new();
        for i in 0..n {
            let o = 4 + i * 24;
            let f = |k: usize| f32::from_le_bytes(b[o + k * 4..o + k * 4 + 4].try_into().unwrap());
            let packed = i32::from_le_bytes(b[o + 20..o + 24].try_into().unwrap());
            let mut octv = packed & 255;
            if octv >= 128 {
                octv |= -128;
            }
            let layer = (packed >> 8) & 255;
            let scale = if octv >= 0 {
                1.0 / ((1 << octv) as f32)
            } else {
                (1 << -octv) as f32
            };
            // The reference passes `360 - kpt.angle` to the descriptor, with
            // near-360 collapsed to 0 — not the stored angle.
            let mut angle = 360.0f32 - f(3);
            if (angle - 360.0).abs() < f32::EPSILON {
                angle = 0.0;
            }
            rows.push((
                i,
                layer,
                [f(0) * scale, f(1) * scale, f(2) * scale * 0.5, angle],
            ));
            let _ = octv;
        }

        // Only octave -1 (the doubled base) is covered here: that is the layer
        // set the oracle dumps as gauss_o0_l*.
        let sel: Vec<&(usize, i32, [f32; 4])> = rows
            .iter()
            .filter(|(i, _, _)| {
                let o = 4 + i * 24;
                let packed = i32::from_le_bytes(b[o + 20..o + 24].try_into().unwrap());
                (packed & 255) == 255
            })
            .collect();
        assert!(!sel.is_empty(), "no octave -1 keypoints");

        let mut bad = 0usize;
        let mut total = 0usize;
        for layer in 1..=3i32 {
            let group: Vec<&&(usize, i32, [f32; 4])> =
                sel.iter().filter(|(_, l, _)| *l == layer).collect();
            if group.is_empty() {
                continue;
            }
            let (h, w, img) =
                load_dump(&format!("{dir}/gauss_o0_l{layer}.f32")).expect("gauss layer");
            let flat: Vec<f32> = group
                .iter()
                .flat_map(|(_, _, k)| k.iter().copied())
                .collect();

            let d_img = stream.clone_htod(&img).unwrap();
            let d_kp = stream.clone_htod(&flat).unwrap();
            let mut d_out = stream.alloc_zeros::<f32>(group.len() * DESCR_LEN).unwrap();
            let d_zero = stream.clone_htod(&vec![0i32]).unwrap();
            let d_n = stream.clone_htod(&vec![group.len() as i32]).unwrap();
            launch_sift_descriptor_cuda(
                ctx,
                &stream,
                &d_img,
                w as u32,
                h as u32,
                &d_kp,
                group.len() as u32,
                4,
                &mut d_out,
                false,
                &d_zero.as_view(),
                &d_n.as_view(),
            )
            .unwrap();
            let got = stream.clone_dtoh(&d_out).unwrap();

            for (gi, (ri, _, _)) in group.iter().enumerate() {
                let a = &got[gi * DESCR_LEN..(gi + 1) * DESCR_LEN];
                let e = &desc_ref[ri * DESCR_LEN..(ri + 1) * DESCR_LEN];
                total += 1;
                if a.iter().zip(e).any(|(x, y)| x.to_bits() != y.to_bits()) {
                    bad += 1;
                }
            }
        }
        let close = total - bad;
        eprintln!("  descriptor: {close}/{total} exact (octave -1, from reference keypoints)");
        assert!(total > 0);
        // The `exact` kernel reproduces the reference's summation order and must
        // be bit-identical. The default block kernel accumulates in shared
        // memory, so a bin can round differently; require it to still land on
        // the same bits for the overwhelming majority.
        if std::env::var("KORNIA_SIFT_DESC").as_deref() == Ok("exact") {
            assert_eq!(bad, 0, "exact descriptor kernel is not bit-exact");
        } else {
            assert!(
                bad * 100 <= total,
                "block descriptor kernel differs on {bad} of {total} descriptors"
            );
        }
    }
}
