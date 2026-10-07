//! CUDA twin of [`StereoMatcher`]: device-resident keypoints and descriptors in,
//! device-resident [`CudaStereoMatches`] out, no host sync per frame.
//!
//! Output is IDENTICAL to the CPU reference (asserted by test): the integer stages (row
//! band, Hamming, centred SAD, histogram median) are exact, and every float expression is
//! a textual mirror of its `matcher.rs` helper compiled with `fmad=false`, so the dot
//! product, parabola and depth round the same way. Ties resolve to the lowest right index
//! and the lowest SAD offset on both sides.
//!
//! Layout: right keypoints are first grouped into row buckets on the device (count, scan,
//! fill — the CPU's row table), then one warp per left keypoint strides over its row's
//! bucket: lanes test candidates in parallel; binary rows (short) are scored one per lane
//! and warp-reduced, float rows by the whole warp with coalesced loads; lanes
//! `0..=2L` then evaluate one SAD offset each. The median reject is a histogram over the
//! (bounded, integer) SAD values plus a one-block select — no sort.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig};
use kornia_image::{Image, ImageError};
use kornia_tensor::CudaKernel;

use super::matcher::{
    StereoDescriptors, StereoMatchError, StereoMatcher, StereoMatches, SubPixelFit, MAX_LEVELS,
};

/// How many keypoints a device side holds.
#[derive(Debug, Clone, Copy)]
pub enum KeypointCount<'a> {
    /// Known on the host.
    Host(usize),
    /// Written by an earlier kernel (e.g. a detector's atomic counter) — read on the
    /// device and clamped to `capacity`, so no host sync is needed between detection
    /// and matching. Buffers must hold `capacity` keypoints.
    Device {
        /// One `i32`.
        count: &'a CudaSlice<i32>,
        /// Upper bound; the launch grid is sized for it.
        capacity: usize,
    },
}

impl KeypointCount<'_> {
    fn capacity(&self) -> usize {
        match *self {
            Self::Host(n) => n,
            Self::Device { capacity, .. } => capacity,
        }
    }
}

/// Device descriptor rows; see [`StereoDescriptors`].
#[derive(Debug, Clone, Copy)]
pub enum CudaStereoDescriptors<'a> {
    /// Binary, `bytes` per keypoint, Hamming distance.
    Binary {
        /// Row-major `capacity × bytes`.
        data: &'a CudaSlice<u8>,
        /// Bytes per descriptor.
        bytes: usize,
    },
    /// Float, `dim` per keypoint, dot product.
    Float {
        /// Row-major `capacity × dim`.
        data: &'a CudaSlice<f32>,
        /// Floats per descriptor.
        dim: usize,
    },
}

/// Device keypoints of one view; see [`StereoKeypoints`].
#[derive(Debug, Clone, Copy)]
pub struct CudaStereoKeypoints<'a> {
    /// Interleaved `[x0, y0, x1, y1, …]`, level-0 pixels.
    pub xy: &'a CudaSlice<f32>,
    /// Octave per keypoint; `None` means all at octave 0.
    pub octaves: Option<&'a CudaSlice<u8>>,
    /// One descriptor per keypoint.
    pub descriptors: CudaStereoDescriptors<'a>,
    /// Number of keypoints.
    pub count: KeypointCount<'a>,
}

/// Device-resident [`StereoMatches`], caller-owned and reusable across frames. Every
/// slot up to capacity is rewritten per call: slots past the left count hold `-1`.
pub struct CudaStereoMatches {
    /// Matched right `u`, `-1` if none.
    pub u_right: CudaSlice<f32>,
    /// Metric depth, `-1` if none.
    pub depth: CudaSlice<f32>,
    /// Matched right keypoint index, `-1` if none.
    pub right_idx: CudaSlice<i32>,
    /// Best SAD per left keypoint (median reject input).
    sad: CudaSlice<i32>,
    capacity: usize,
    stream: Arc<CudaStream>,
}

impl CudaStereoMatches {
    /// Left-keypoint capacity.
    ///
    /// # Returns
    ///
    /// The number of left keypoints this buffer holds results for.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Copies the first `n` results to the host. Synchronizes the stream.
    ///
    /// # Arguments
    ///
    /// * `n` - Results to copy, clamped to [`capacity`](Self::capacity).
    ///
    /// # Returns
    ///
    /// The host [`StereoMatches`] for left keypoints `0..n`.
    ///
    /// # Errors
    ///
    /// Returns [`StereoMatchError::Image`]`(`[`ImageError::Cuda`]`)` on a copy failure.
    pub fn to_host(&self, n: usize) -> Result<StereoMatches, StereoMatchError> {
        let n = n.min(self.capacity);
        let s = &self.stream;
        Ok(StereoMatches {
            u_right: s.clone_dtoh(&self.u_right.slice(0..n)).map_err(cuda_err)?,
            depth: s.clone_dtoh(&self.depth.slice(0..n)).map_err(cuda_err)?,
            right_idx: s
                .clone_dtoh(&self.right_idx.slice(0..n))
                .map_err(cuda_err)?,
        })
    }
}

// ── Kernel ABI ───────────────────────────────────────────────────────────────────────
// `Params` / `Pyramid` are mirrored field-for-field by the structs in KERNELS.

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Pyramid {
    left: [u64; MAX_LEVELS],
    right: [u64; MAX_LEVELS],
    width: [i32; MAX_LEVELS],
    height: [i32; MAX_LEVELS],
}
// SAFETY: plain-old-data, repr(C), mirrored by `struct Pyramid` in KERNELS.
unsafe impl DeviceRepr for Pyramid {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Params {
    scale: [f32; MAX_LEVELS],
    inv_scale: [f32; MAX_LEVELS],
    row_band: f32,
    min_d: f32,
    max_d: f32,
    bf: f32,
    min_sim: f32,
    max_hamming: u32,
    n_levels: i32,
    rows: i32,
    gate_oct: i32,
    sad_on: i32,
    sad_w: i32,
    sad_l: i32,
    desc_width: i32,
    median_on: i32,
    fit_parabola: i32,
    bucketed: i32,
    hist_bins: i32,
    hist_shift: i32,
}
// SAFETY: plain-old-data, repr(C), mirrored by `struct Params` in KERNELS.
unsafe impl DeviceRepr for Params {}

const KERNELS: &str = r#"
#define MAX_LEVELS 8
struct Pyramid {
    unsigned long long left[MAX_LEVELS];
    unsigned long long right[MAX_LEVELS];
    int width[MAX_LEVELS];
    int height[MAX_LEVELS];
};
struct Params {
    float scale[MAX_LEVELS];
    float inv_scale[MAX_LEVELS];
    float row_band, min_d, max_d, bf, min_sim;
    unsigned int max_hamming;
    int n_levels, rows, gate_oct, sad_on, sad_w, sad_l, desc_width, median_on,
        fit_parabola, bucketed, hist_bins, hist_shift;
};

__device__ __forceinline__ int count_of(const int* c, int cap) {
    if (c == 0) return cap;
    int n = *c;
    return n < 0 ? 0 : (n > cap ? cap : n);
}

/* ---- mirrors of matcher.rs helpers; keep expression trees identical ---- */

/* row_span */
__device__ __forceinline__ void row_span(float y, float r, int rows, int* lo, int* hi) {
    *lo = max((int)floorf(y - r), 0);
    *hi = min((int)ceilf(y + r), rows - 1);
}
/* left_row */
__device__ __forceinline__ int left_row(float v, int rows) {
    return min(max((int)v, 0), rows - 1);
}
/* sad_fits */
__device__ __forceinline__ bool sad_fits(long long su_l, long long sv, long long su_r0, long long w, long long l, long long iw, long long ih) {
    return su_l - w >= 0 && su_l + w < iw && sv - w >= 0 && sv + w < ih
        && su_r0 - l - w >= 0 && su_r0 + l + w < iw;
}
/* Centred SAD of every search offset at once (matcher.rs::centred_sad, summed in a
   different but integer-exact order). The right window [su_r0-l-w, su_r0+l+w] has
   2(l+w)+1 <= 32 columns: lane c holds column c of the current patch row, lanes
   0..2w hold the left patch row, and lane t (offset t-l) gathers its 2w+1 terms by
   shuffle. Lanes past 2l return garbage the caller masks. All lanes must call. */
__device__ int warp_centred_sad(const unsigned char* L, const unsigned char* R, int stride,
                                int su_l, int su_r0, int sv, int w, int l, int lane) {
    const unsigned FULL = 0xffffffffu;
    int ncol = 2 * (l + w) + 1;
    int x0 = su_r0 - l - w;
    int lc = L[sv * stride + su_l];
    int r0 = lane < ncol ? (int)R[sv * stride + x0 + lane] : 0;
    int rc = __shfl_sync(FULL, r0, min(lane + w, 31));   /* R(su_r0 + offset, sv) */
    int sad = 0;
    for (int dy = -w; dy <= w; ++dy) {
        int row = (sv + dy) * stride;
        int rv = lane < ncol ? (int)R[row + x0 + lane] : 0;
        int lv = lane <= 2 * w ? (int)L[row + su_l - w + lane] : 0;
        for (int dx = 0; dx <= 2 * w; ++dx) {
            int a = __shfl_sync(FULL, lv, dx) - lc;
            int b = __shfl_sync(FULL, rv, min(lane + dx, 31)) - rc;
            sad += abs(a - b);
        }
    }
    return sad;
}

/* Hamming of one pair by one lane, as u32 words when rows are word-sized (rows start at
   j * bytes in a cudaMalloc'd buffer, so they are 4-aligned); integer, any order is exact. */
__device__ __forceinline__ unsigned int lane_dist_bin(const unsigned char* a, const unsigned char* b,
                                                      int bytes) {
    unsigned int d = 0;
    if ((bytes & 3) == 0) {
        const unsigned int* a4 = (const unsigned int*)a;
        const unsigned int* b4 = (const unsigned int*)b;
        for (int k = 0; k < (bytes >> 2); ++k) d += __popc(a4[k] ^ b4[k]);
    } else {
        for (int k = 0; k < bytes; ++k) d += __popc((unsigned int)(a[k] ^ b[k]));
    }
    return d;
}

/* Warp-cooperative dot in matcher.rs::dot's order (fmad=false); every lane returns it. */
__device__ __forceinline__ float warp_dist_flt(const float* a, const float* b, int dim, int lane) {
    float s = 0.0f;
    for (int k = lane; k < dim; k += 32) s = s + a[k] * b[k];
    for (int off = 16; off > 0; off >>= 1) s = s + __shfl_down_sync(0xffffffffu, s, off);
    return __shfl_sync(0xffffffffu, s, 0);
}

/* Is (cand, cj) better than (cur, j)? Lower Hamming, ties to the lowest index;
   j < 0 is "none". */
__device__ __forceinline__ bool better_bin(unsigned int cand, int cj, unsigned int cur, int j) {
    if (cj < 0) return false;
    if (j < 0) return true;
    return cand < cur || (cand == cur && cj < j);
}
/* Mirrors Best::offer: `s > sim` from sim = -inf, ties to the lowest index; NaN / -inf
   never win. */
__device__ __forceinline__ bool better_flt(float cand, int cj, float cur, int j) {
    return cj >= 0 && (cand > cur || (cand == cur && j >= 0 && cj < j));
}

template <bool BIN>
__device__ void stereo_match(
    Pyramid pyr, Params p,
    const float* __restrict__ lxy, const unsigned char* __restrict__ loct,
    const void* __restrict__ ldesc, const int* __restrict__ lcount, int lcap,
    const float* __restrict__ rxy, const unsigned char* __restrict__ roct,
    const void* __restrict__ rdesc, const int* __restrict__ rcount, int rcap,
    float* __restrict__ u_right, float* __restrict__ depth, int* __restrict__ ridx,
    int* __restrict__ sad_out, int* __restrict__ hist, int* __restrict__ n_acc,
    const int* __restrict__ row_count, const int* __restrict__ row_start,
    const int* __restrict__ bucket)
{
    const unsigned FULL = 0xffffffffu;
    int lane = threadIdx.x & 31;
    int il = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (il >= lcap) return;                     /* warp-uniform; the grid rounds lcap up */
    /* Sentinels up to capacity, so slots past a device count never keep an old match. */
    if (lane == 0) { u_right[il] = -1.0f; depth[il] = -1.0f; ridx[il] = -1; sad_out[il] = -1; }
    if (il >= count_of(lcount, lcap) || p.rows <= 0) return;   /* matcher.rs: rows <= 0 */
    int nr = count_of(rcount, rcap);

    float u_l = lxy[2 * il], v_l = lxy[2 * il + 1];
    int o_l = loct ? (int)loct[il] : 0;
    if (o_l >= p.n_levels) return;
    /* disparity_window */
    float min_u = u_l - p.max_d;
    float max_u = u_l - p.min_d;
    if (!(max_u >= 0.0f)) return;
    int row = left_row(v_l, p.rows);

    unsigned int bd = 0xffffffffu; float bs = -__int_as_float(0x7f800000); int bj = -1;  /* warp-uniform */
    /* Bucketed: only the rights whose row span holds `row` (built by stereo_bucket_*
       with the same row_span); else all of them. The predicate below is re-applied
       either way, so both modes select the identical candidate set. */
    int kbeg = 0, kend = nr;
    if (p.bucketed) { kbeg = row_start[row]; kend = kbeg + row_count[row]; }
    /* Lanes test 32 candidates at once; the survivors are then scored one at a time by
       the whole warp (coalesced descriptor rows), so best/bj stay warp-uniform. */
    const unsigned char* la8 = (const unsigned char*)ldesc + (size_t)il * p.desc_width;
    const float* laf = (const float*)ldesc + (size_t)il * p.desc_width;
    for (int k0 = kbeg; k0 < kend; k0 += 32) {
        int k = k0 + lane;
        int j = -1;
        bool ok = false;
        if (k < kend) {
            j = p.bucketed ? bucket[k] : k;
            int o_r = roct ? (int)roct[j] : 0;
            if (o_r < p.n_levels) {
                int lo, hi;
                row_span(rxy[2 * j + 1], p.row_band * p.scale[o_r], p.rows, &lo, &hi);
                float u_r = rxy[2 * j];
                ok = row >= lo && row <= hi
                    && !(p.gate_oct && abs(o_l - o_r) > 1)          /* octave_gate */
                    && (u_r >= min_u && u_r <= max_u);
            }
        }
        if (BIN) {
            /* Short rows: one candidate per lane; reduced across the warp below. */
            if (ok) {
                unsigned int d = lane_dist_bin(la8, (const unsigned char*)rdesc + (size_t)j * p.desc_width,
                                               p.desc_width);
                if (better_bin(d, j, bd, bj)) { bd = d; bj = j; }
            }
        } else {
            unsigned m = __ballot_sync(FULL, ok);
            while (m) {
                int src = __ffs(m) - 1;
                m &= m - 1;
                int jj = __shfl_sync(FULL, j, src);
                float sc = warp_dist_flt(laf, (const float*)rdesc + (size_t)jj * p.desc_width,
                                         p.desc_width, lane);
                if (better_flt(sc, jj, bs, bj)) { bs = sc; bj = jj; }
            }
        }
    }
    if (BIN) {
        for (int off = 16; off > 0; off >>= 1) {
            int oj = __shfl_down_sync(FULL, bj, off);
            unsigned int od = __shfl_down_sync(FULL, bd, off);
            if (better_bin(od, oj, bd, bj)) { bd = od; bj = oj; }
        }
        bj = __shfl_sync(FULL, bj, 0);
        bd = __shfl_sync(FULL, bd, 0);
    }
    if (bj < 0) return;
    if (BIN ? !(bd < p.max_hamming) : !(bs > p.min_sim)) return;   /* Best::accept */

    float u_r0 = rxy[2 * bj];
    float u_r;
    int best_sad = 0;
    if (!p.sad_on) {
        u_r = u_r0;
    } else {
        int w = p.sad_w, l = p.sad_l;
        float inv = p.inv_scale[o_l], sc = p.scale[o_l];
        int su_l = (int)roundf(u_l * inv);
        int sv = (int)roundf(v_l * inv);
        int su_r0 = (int)roundf(u_r0 * inv);
        int iw = pyr.width[o_l], ih = pyr.height[o_l];
        if (!sad_fits(su_l, sv, su_r0, w, l, iw, ih)) return;
        const unsigned char* L = (const unsigned char*)pyr.left[o_l];
        const unsigned char* R = (const unsigned char*)pyr.right[o_l];
        int inc = lane - l;
        int sad = warp_centred_sad(L, R, iw, su_l, su_r0, sv, w, l, lane);
        if (lane > 2 * l) sad = 0x7fffffff;
        /* argmin, ties to the smallest offset (= lowest lane) */
        int bsad = sad, binc = lane <= 2 * l ? inc : 0x7fffffff;
        for (int off = 16; off > 0; off >>= 1) {
            int os = __shfl_down_sync(FULL, bsad, off);
            int oi = __shfl_down_sync(FULL, binc, off);
            if (os < bsad || (os == bsad && oi < binc)) { bsad = os; binc = oi; }
        }
        bsad = __shfl_sync(FULL, bsad, 0);
        binc = __shfl_sync(FULL, binc, 0);
        int i = binc + l;
        int s1 = __shfl_sync(FULL, sad, i > 0 ? i - 1 : 0);
        int s2 = __shfl_sync(FULL, sad, i);
        int s3 = __shfl_sync(FULL, sad, i < 31 ? i + 1 : 31);
        if (binc == -l || binc == l) return;
        /* sub_pixel_offset */
        float d1 = (float)s1, d2 = (float)s2, d3 = (float)s3;
        float denom = p.fit_parabola ? 2.0f * (d1 + d3 - 2.0f * d2)
                                     : 2.0f * fmaxf(d1 - d2, d3 - d2);
        if (denom == 0.0f) return;
        float delta = (d1 - d3) / denom;
        if (!(delta >= -1.0f && delta <= 1.0f)) return;
        /* sub_pixel_u */
        u_r = sc * ((float)su_r0 + (float)binc + delta);
        best_sad = bsad;
    }
    if (lane != 0) return;
    /* finish */
    float disparity = u_l - u_r;
    if (!(disparity >= p.min_d && disparity < p.max_d)) return;
    if (disparity <= 0.0f) { disparity = 0.01f; u_r = u_l - 0.01f; }
    u_right[il] = u_r;
    depth[il] = p.bf / disparity;
    ridx[il] = bj;
    sad_out[il] = best_sad;
    if (p.median_on) {
        atomicAdd(&hist[best_sad], 1);
        atomicAdd(&hist[p.hist_bins + (best_sad >> p.hist_shift)], 1);   /* coarse bucket */
        atomicAdd(n_acc, 1);
    }
}

#define STEREO_ARGS \
    Pyramid pyr, Params p, \
    const float* lxy, const unsigned char* loct, const void* ldesc, const int* lcount, int lcap, \
    const float* rxy, const unsigned char* roct, const void* rdesc, const int* rcount, int rcap, \
    float* u_right, float* depth, int* ridx, int* sad_out, int* hist, int* n_acc, \
    const int* row_count, const int* row_start, const int* bucket
#define STEREO_PASS pyr, p, lxy, loct, ldesc, lcount, lcap, rxy, roct, rdesc, rcount, rcap, \
    u_right, depth, ridx, sad_out, hist, n_acc, row_count, row_start, bucket

extern "C" __global__ void stereo_match_bin(STEREO_ARGS) { stereo_match<true>(STEREO_PASS); }
extern "C" __global__ void stereo_match_flt(STEREO_ARGS) { stereo_match<false>(STEREO_PASS); }

/* Exclusive scan of part[0 .. 32*per) in place by warp 0 (`per` sequential + 5 shuffle
   steps); call with the whole block, between __syncthreads. */
__device__ void block_exclusive_scan(int* part, int per) {
    if (threadIdx.x >= 32) return;
    int lane = threadIdx.x;
    int base = lane * per, s = 0;
    for (int i = 0; i < per; ++i) { int v = part[base + i]; part[base + i] = s; s += v; }
    int incl = s;
    for (int off = 1; off < 32; off <<= 1) {
        int o = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) incl += o;
    }
    int excl = incl - s;
    for (int i = 0; i < per; ++i) part[base + i] += excl;
}

/* Row buckets for the right keypoints: count per row span, exclusive scan, fill.
   Order inside a bucket is atomic (arbitrary); the match reduction compares
   (score, index) explicitly, so it does not depend on it. */
extern "C" __global__ void stereo_bucket_count(Params p, const float* __restrict__ rxy,
                                               const unsigned char* __restrict__ roct,
                                               const int* __restrict__ rcount, int rcap,
                                               int* __restrict__ row_count) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= count_of(rcount, rcap)) return;
    int o = roct ? (int)roct[j] : 0;
    if (o >= p.n_levels) return;
    int lo, hi;
    row_span(rxy[2 * j + 1], p.row_band * p.scale[o], p.rows, &lo, &hi);
    for (int r = lo; r <= hi; ++r) atomicAdd(&row_count[r], 1);
}
extern "C" __global__ void stereo_bucket_scan(const int* __restrict__ row_count, int rows,
                                              int* __restrict__ row_start, int* __restrict__ cursor) {
    __shared__ int part[1024];
    int t = threadIdx.x;
    int chunk = (rows + 1023) / 1024;
    int b0 = t * chunk, b1 = min(rows, b0 + chunk);
    int s = 0;
    for (int b = b0; b < b1; ++b) s += row_count[b];
    part[t] = s;
    __syncthreads();
    block_exclusive_scan(part, 32);
    __syncthreads();
    int acc = part[t];
    for (int b = b0; b < b1; ++b) { row_start[b] = acc; cursor[b] = acc; acc += row_count[b]; }
}
extern "C" __global__ void stereo_bucket_fill(Params p, const float* __restrict__ rxy,
                                              const unsigned char* __restrict__ roct,
                                              const int* __restrict__ rcount, int rcap,
                                              int* __restrict__ cursor, int* __restrict__ bucket) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= count_of(rcount, rcap)) return;
    int o = roct ? (int)roct[j] : 0;
    if (o >= p.n_levels) return;
    int lo, hi;
    row_span(rxy[2 * j + 1], p.row_band * p.scale[o], p.rows, &lo, &hi);
    for (int r = lo; r <= hi; ++r) bucket[atomicAdd(&cursor[r], 1)] = j;
}

/* k-th smallest SAD (k = n/2): the value matcher.rs's select_nth_unstable returns.
   One 256-thread block: scan the coarse histogram (<= 256 buckets of 2^shift values)
   to find the bucket holding the k-th, then scan that bucket's fine bins, `per` per
   thread; the owning thread walks its run. */
extern "C" __global__ void stereo_median(const int* __restrict__ hist, int bins, int shift,
                                         const int* __restrict__ n_acc, int* __restrict__ median) {
    __shared__ int part[256];
    __shared__ int found_bucket, base_cum;
    int t = threadIdx.x;
    int n = *n_acc;
    if (n == 0) { if (t == 0) *median = -1; return; }
    int k = n / 2;
    int n_coarse = (bins + (1 << shift) - 1) >> shift;
    int c = t < n_coarse ? hist[bins + t] : 0;
    part[t] = c;
    __syncthreads();
    block_exclusive_scan(part, 8);
    __syncthreads();
    if (part[t] <= k && k < part[t] + c) { found_bucket = t; base_cum = part[t]; }
    __syncthreads();
    int per = max((1 << shift) >> 8, 1);
    int b0 = (found_bucket << shift) + t * per;
    int b1 = min(min(b0 + per, (found_bucket + 1) << shift), bins);
    int f = 0;
    for (int b = b0; b < b1; ++b) f += hist[b];
    __syncthreads();
    part[t] = f;
    __syncthreads();
    block_exclusive_scan(part, 8);
    __syncthreads();
    int cum = base_cum + part[t];
    if (!(cum <= k && k < cum + f)) return;   /* exactly one owner */
    for (int b = b0; b < b1; ++b) {
        cum += hist[b];
        if (cum > k) { *median = b; return; }
    }
}

/* median_threshold + reject */
extern "C" __global__ void stereo_reject(const int* __restrict__ lcount, int lcap, float factor,
                                         const int* __restrict__ median,
                                         const int* __restrict__ sad_out,
                                         float* u_right, float* depth, int* ridx) {
    int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= count_of(lcount, lcap)) return;
    int m = *median;
    if (m < 0 || ridx[il] < 0) return;
    float th = factor * (float)m;
    if (sad_out[il] > 0 && (float)sad_out[il] >= th) {   /* matcher.rs: median_rejects */ u_right[il] = -1.0f; depth[il] = -1.0f; ridx[il] = -1; }
}
"#;

/// Warps (left keypoints) per block of the match kernel.
const WARPS_PER_BLOCK: u32 = 4;

/// [`StereoMatcher`] with compiled kernels and its scratch; see the module docs.
pub struct CudaStereoMatcher {
    host: StereoMatcher,
    stream: Arc<CudaStream>,
    k_bin: CudaKernel,
    k_flt: CudaKernel,
    k_median: CudaKernel,
    k_reject: CudaKernel,
    k_bcount: CudaKernel,
    k_bscan: CudaKernel,
    k_bfill: CudaKernel,
    /// `[fine hist (bins) | coarse hist | n_acc | median]`, zeroed per call.
    scratch: CudaSlice<i32>,
    bins: usize,
    /// `[row_count | row_start | cursor]`, `3 × rows`; grown on demand.
    rows_buf: CudaSlice<i32>,
    /// Right indices grouped by row; grown on demand.
    bucket: CudaSlice<i32>,
}

/// log2 of the SAD values per coarse bucket: the smallest (>= 8) giving at most 256
/// buckets, the median kernel's block size. The default 11×11 window needs 8; the
/// widest legal one (w = 14, 428 911 bins) needs 11.
fn coarse_shift(bins: usize) -> u32 {
    let mut shift = 8;
    while bins.div_ceil(1 << shift) > 256 {
        shift += 1;
    }
    shift
}

fn coarse_bins(bins: usize) -> usize {
    bins.div_ceil(1 << coarse_shift(bins))
}

fn cuda_err(e: impl std::fmt::Display) -> StereoMatchError {
    StereoMatchError::Image(ImageError::Cuda(e.to_string()))
}

impl StereoMatcher {
    /// Compiles the kernels and allocates the scratch on `stream`'s device, returning a
    /// matcher for device-resident inputs. Synchronizes once, so nvrtc failures surface
    /// here — where a caller's CPU fallback can catch them — not on the first frame.
    ///
    /// # Arguments
    ///
    /// * `stream` - Stream all matching work is enqueued on.
    ///
    /// # Returns
    ///
    /// A [`CudaStereoMatcher`] with this matcher's config.
    ///
    /// # Errors
    ///
    /// Returns [`StereoMatchError::Image`]`(`[`ImageError::Cuda`]`)` on compile or
    /// allocation failure.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cudarc::driver::CudaContext;
    /// use kornia_3d::stereo::{
    ///     CudaStereoDescriptors, CudaStereoKeypoints, KeypointCount, StereoMatchConfig,
    ///     StereoMatcher,
    /// };
    /// use kornia_image::{Image, ImageSize};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.default_stream();
    /// let mut gpu = StereoMatcher::new(StereoMatchConfig::new(435.0, 0.11))?.to_cuda(&stream)?;
    ///
    /// // A detector's device outputs: interleaved xy, 64-D f32 descriptors, and its
    /// // atomic keypoint counter (read on the device, so no host sync in between).
    /// let cap = 1024;
    /// let xy = stream.alloc_zeros::<f32>(2 * cap)?;
    /// let desc = stream.alloc_zeros::<f32>(64 * cap)?;
    /// let count = stream.alloc_zeros::<i32>(1)?;
    /// let kp = CudaStereoKeypoints {
    ///     xy: &xy,
    ///     octaves: None,
    ///     descriptors: CudaStereoDescriptors::Float { data: &desc, dim: 64 },
    ///     count: KeypointCount::Device { count: &count, capacity: cap },
    /// };
    ///
    /// let size = ImageSize { width: 752, height: 480 };
    /// let left = Image::<u8, 1>::zeros_cuda(size, &stream)?;
    /// let right = Image::<u8, 1>::zeros_cuda(size, &stream)?;
    /// let mut out = gpu.alloc_matches(cap)?;
    /// gpu.match_device(&[left], &[right], &kp, &kp, &mut out)?;
    /// let host = out.to_host(cap)?; // synchronizes
    /// # let _ = host;
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_cuda(&self, stream: &Arc<CudaStream>) -> Result<CudaStereoMatcher, StereoMatchError> {
        let names = [
            "stereo_match_bin",
            "stereo_match_flt",
            "stereo_median",
            "stereo_reject",
            "stereo_bucket_count",
            "stereo_bucket_scan",
            "stereo_bucket_fill",
        ];
        let [k_bin, k_flt, k_median, k_reject, k_bcount, k_bscan, k_bfill]: [CudaKernel; 7] =
            CudaKernel::compile_many(stream.context(), KERNELS, &names)
                .map_err(cuda_err)?
                .try_into()
                .unwrap_or_else(|_| unreachable!("compile_many returns one kernel per name"));
        // Largest centred SAD: every term |a - b| <= 510.
        let bins = self
            .cfg
            .sad
            .map_or(1, |s| (2 * s.half_window + 1).pow(2) * 510 + 1);
        let scratch = stream
            .alloc_zeros::<i32>(bins + coarse_bins(bins) + 2)
            .map_err(cuda_err)?;
        let rows_buf = stream.alloc_zeros::<i32>(1).map_err(cuda_err)?;
        let bucket = stream.alloc_zeros::<i32>(1).map_err(cuda_err)?;
        stream.synchronize().map_err(cuda_err)?;
        Ok(CudaStereoMatcher {
            host: self.clone(),
            stream: stream.clone(),
            k_bin,
            k_flt,
            k_median,
            k_reject,
            k_bcount,
            k_bscan,
            k_bfill,
            scratch,
            bins,
            rows_buf,
            bucket,
        })
    }
}

impl CudaStereoMatcher {
    /// The CPU matcher this was built from (same config).
    ///
    /// # Returns
    ///
    /// The host [`StereoMatcher`], e.g. for a CPU fallback.
    pub fn host(&self) -> &StereoMatcher {
        &self.host
    }

    /// Allocates a result buffer for up to `capacity` left keypoints.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum left keypoints per call.
    ///
    /// # Returns
    ///
    /// A [`CudaStereoMatches`] on this matcher's stream, reusable across frames.
    ///
    /// # Errors
    ///
    /// Returns [`StereoMatchError::Image`]`(`[`ImageError::Cuda`]`)` on allocation failure.
    pub fn alloc_matches(&self, capacity: usize) -> Result<CudaStereoMatches, StereoMatchError> {
        let s = &self.stream;
        let n = capacity.max(1);
        Ok(CudaStereoMatches {
            u_right: s.alloc_zeros(n).map_err(cuda_err)?,
            depth: s.alloc_zeros(n).map_err(cuda_err)?,
            right_idx: s.alloc_zeros(n).map_err(cuda_err)?,
            sad: s.alloc_zeros(n).map_err(cuda_err)?,
            capacity,
            stream: s.clone(),
        })
    }

    /// Device twin of [`StereoMatcher::match_into`]. Enqueued on this matcher's stream
    /// with no host sync; inputs must be ready in that stream's order. See
    /// [`StereoMatcher::to_cuda`] for an example.
    ///
    /// # Arguments
    ///
    /// * `left_pyramid` / `right_pyramid` - Device-resident rectified images, one per
    ///   octave; needed only with SAD refinement.
    /// * `left` / `right` - Device keypoints, descriptors and counts.
    /// * `out` - Result buffer from [`alloc_matches`](Self::alloc_matches), with at
    ///   least the left capacity.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the work is enqueued; read `out` after syncing the stream.
    ///
    /// # Errors
    /// [`StereoMatchError`] on inconsistent inputs (checked before any launch),
    /// [`ImageError::HostResident`] for a host pyramid level,
    /// [`ImageError::DeviceMismatch`] for a pyramid level, keypoint buffer or output on
    /// another device, [`StereoMatchError::Stream`] for an output allocated on another
    /// stream, [`ImageError::Cuda`] on launch failure.
    pub fn match_device(
        &mut self,
        left_pyramid: &[Image<u8, 1>],
        right_pyramid: &[Image<u8, 1>],
        left: &CudaStereoKeypoints,
        right: &CudaStereoKeypoints,
        out: &mut CudaStereoMatches,
    ) -> Result<(), StereoMatchError> {
        let ordinal = self.stream.context().ordinal();
        check_device_keypoints(left, "left", ordinal)?;
        check_device_keypoints(right, "right", ordinal)?;
        super::matcher::check_descriptor_pair(
            &host_view(&left.descriptors),
            &host_view(&right.descriptors),
        )?;
        let lcap = left.count.capacity();
        if out.capacity < lcap {
            return Err(StereoMatchError::Length {
                side: "left",
                what: "match output capacity",
                got: out.capacity,
                expected: lcap,
            });
        }
        if out.u_right.ordinal() != ordinal {
            return Err(ImageError::DeviceMismatch.into());
        }
        if !Arc::ptr_eq(&out.stream, &self.stream) {
            return Err(StereoMatchError::Stream("match output"));
        }
        let cfg = &self.host.cfg;
        let st = self.stream.clone();

        // Pyramid: same validation as the CPU path, plus residency.
        let rows = self.host.check_pyramids(left_pyramid, right_pyramid)?;
        let mut pyr = Pyramid::default();
        let mut guards = Vec::new();
        if cfg.sad.is_some() {
            for (o, (l, r)) in left_pyramid.iter().zip(right_pyramid).enumerate() {
                let (lp, lg) = device_ptr(&st, l)?;
                let (rp, rg) = device_ptr(&st, r)?;
                guards.push(lg);
                guards.push(rg);
                pyr.left[o] = lp;
                pyr.right[o] = rp;
                pyr.width[o] = l.width() as i32;
                pyr.height[o] = l.height() as i32;
            }
        }

        let (median_on, median_factor) = match cfg.sad.and_then(|s| s.median_factor) {
            Some(f) => (1, f),
            None => (0, 0.0),
        };
        let (desc_width, binary) = match left.descriptors {
            CudaStereoDescriptors::Binary { bytes, .. } => (bytes, true),
            CudaStereoDescriptors::Float { dim, .. } => (dim, false),
        };
        let params = Params {
            scale: self.host.scale,
            inv_scale: self.host.inv_scale,
            row_band: cfg.row_band,
            min_d: cfg.min_disparity,
            max_d: cfg.max_disparity,
            bf: cfg.bf,
            min_sim: cfg.min_similarity,
            max_hamming: cfg.max_hamming,
            n_levels: cfg.n_levels as i32,
            rows,
            gate_oct: (left.octaves.is_some() || right.octaves.is_some()) as i32,
            sad_on: cfg.sad.is_some() as i32,
            sad_w: cfg.sad.map_or(0, |s| s.half_window as i32),
            sad_l: cfg.sad.map_or(0, |s| s.search_range as i32),
            desc_width: desc_width as i32,
            median_on,
            fit_parabola: cfg.sad.is_some_and(|s| s.fit == SubPixelFit::Parabola) as i32,
            // Row buckets need a bounded row count, i.e. an image; without one the
            // match kernel scans every right keypoint.
            bucketed: (rows != i32::MAX) as i32,
            hist_bins: self.bins as i32,
            hist_shift: coarse_shift(self.bins) as i32,
        };

        if lcap == 0 {
            return Ok(());
        }
        let rcap = right.count.capacity();
        let (lcap_i, rcap_i) = (lcap as i32, rcap as i32);
        let null = 0u64;
        let (row_count, row_start) = if params.bucketed == 1 {
            let rows_u = rows as usize;
            // A span covers at most min(2r + 3, rows) rows; r may be inf.
            let r_max = cfg.row_band * self.host.scale[cfg.n_levels - 1];
            let span = ((2.0 * r_max).floor() as usize)
                .saturating_add(3)
                .min(rows_u);
            let entries = rcap.saturating_mul(span).max(1);
            if self.rows_buf.len() < 3 * rows_u {
                self.rows_buf = st.alloc_zeros(3 * rows_u).map_err(cuda_err)?;
            }
            if self.bucket.len() < entries {
                self.bucket = st.alloc_zeros(entries).map_err(cuda_err)?;
            }
            st.memset_zeros(&mut self.rows_buf.slice_mut(0..rows_u))
                .map_err(cuda_err)?;
            let row_count = self.rows_buf.slice(0..rows_u);
            let row_start = self.rows_buf.slice(rows_u..2 * rows_u);
            let cursor = self.rows_buf.slice(2 * rows_u..3 * rows_u);
            if rcap > 0 {
                push_bucket_args(self.k_bcount.launch_builder(&st), &params, right, &null)
                    .arg(&rcap_i)
                    .arg(&row_count)
                    .launch_1d(rcap as u32)
                    .map_err(cuda_err)?;
                self.k_bscan
                    .launch_builder(&st)
                    .arg(&row_count)
                    .arg(&rows)
                    .arg(&row_start)
                    .arg(&cursor)
                    .launch_cfg(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1024, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map_err(cuda_err)?;
                push_bucket_args(self.k_bfill.launch_builder(&st), &params, right, &null)
                    .arg(&rcap_i)
                    .arg(&cursor)
                    .arg(&self.bucket)
                    .launch_1d(rcap as u32)
                    .map_err(cuda_err)?;
            }
            (row_count, row_start)
        } else {
            // Unused when not bucketed; any valid pointer will do.
            (self.rows_buf.slice(0..1), self.rows_buf.slice(0..1))
        };
        st.memset_zeros(&mut self.scratch).map_err(cuda_err)?;
        let kernel = if binary { &self.k_bin } else { &self.k_flt };
        let h = self.bins + coarse_bins(self.bins);
        let hist = self.scratch.slice(0..h);
        let n_acc = self.scratch.slice(h..h + 1);
        let median = self.scratch.slice(h + 1..h + 2);

        let mut b = kernel.launch_builder(&st).arg(&pyr).arg(&params);
        b = push_side(b, left, &null).arg(&lcap_i);
        b = push_side(b, right, &null).arg(&rcap_i);
        // Outputs pass as &mut so cudarc records their write event for other streams.
        b.arg(&mut out.u_right)
            .arg(&mut out.depth)
            .arg(&mut out.right_idx)
            .arg(&mut out.sad)
            .arg(&hist)
            .arg(&n_acc)
            .arg(&row_count)
            .arg(&row_start)
            .arg(&self.bucket)
            .launch_cfg(LaunchConfig {
                grid_dim: ((lcap as u32).div_ceil(WARPS_PER_BLOCK), 1, 1),
                block_dim: (32 * WARPS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(cuda_err)?;

        if median_on == 1 {
            self.k_median
                .launch_builder(&st)
                .arg(&hist)
                .arg(&params.hist_bins)
                .arg(&params.hist_shift)
                .arg(&n_acc)
                .arg(&median)
                .launch_cfg(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(cuda_err)?;
            push_count(self.k_reject.launch_builder(&st), left.count, &null)
                .arg(&lcap_i)
                .arg(&median_factor)
                .arg(&median)
                .arg(&mut out.sad)
                .arg(&mut out.u_right)
                .arg(&mut out.depth)
                .arg(&mut out.right_idx)
                .launch_1d(lcap as u32)
                .map_err(cuda_err)?;
        }
        drop(guards);
        Ok(())
    }
}

/// Raw device pointer of a pyramid level, after the residency/device checks the
/// rectifier applies. The guard must outlive the launch that reads the pointer.
fn device_ptr<'a>(
    stream: &'a Arc<CudaStream>,
    img: &'a Image<u8, 1>,
) -> Result<(u64, cudarc::driver::SyncOnDrop<'a>), StereoMatchError> {
    let Some(img_stream) = img.cuda_stream() else {
        return Err(ImageError::HostResident.into());
    };
    if img_stream.context().ordinal() != stream.context().ordinal() {
        return Err(ImageError::DeviceMismatch.into());
    }
    let slice = img.as_cudaslice().ok_or(ImageError::HostResident)?;
    Ok(slice.device_ptr(stream))
}

/// `(params, rxy, roct, rcount)` — the leading arguments of the bucket kernels.
fn push_bucket_args<'a>(
    b: kornia_tensor::CudaLaunchBuilder<'a>,
    params: &'a Params,
    k: &'a CudaStereoKeypoints<'a>,
    null: &'a u64,
) -> kornia_tensor::CudaLaunchBuilder<'a> {
    let b = push_octaves(b.arg(params).arg(k.xy), k.octaves, null);
    push_count(b, k.count, null)
}

fn push_side<'a>(
    b: kornia_tensor::CudaLaunchBuilder<'a>,
    k: &'a CudaStereoKeypoints<'a>,
    null: &'a u64,
) -> kornia_tensor::CudaLaunchBuilder<'a> {
    let b = push_octaves(b.arg(k.xy), k.octaves, null);
    let b = match k.descriptors {
        CudaStereoDescriptors::Binary { data, .. } => b.arg(data),
        CudaStereoDescriptors::Float { data, .. } => b.arg(data),
    };
    push_count(b, k.count, null)
}

/// Pushes the device count, or a null pointer when the count is host-known.
fn push_count<'a>(
    b: kornia_tensor::CudaLaunchBuilder<'a>,
    c: KeypointCount<'a>,
    null: &'a u64,
) -> kornia_tensor::CudaLaunchBuilder<'a> {
    match c {
        KeypointCount::Device { count, .. } => b.arg(count),
        KeypointCount::Host(_) => b.arg(null),
    }
}

/// Pushes the octave buffer, or a null pointer for all-octave-0.
fn push_octaves<'a>(
    b: kornia_tensor::CudaLaunchBuilder<'a>,
    o: Option<&'a CudaSlice<u8>>,
    null: &'a u64,
) -> kornia_tensor::CudaLaunchBuilder<'a> {
    match o {
        Some(o) => b.arg(o),
        None => b.arg(null),
    }
}

/// Zero-length host view carrying only kind and width, for the shared pair check.
fn host_view(d: &CudaStereoDescriptors) -> StereoDescriptors<'static> {
    match *d {
        CudaStereoDescriptors::Binary { bytes, .. } => {
            StereoDescriptors::Binary { data: &[], bytes }
        }
        CudaStereoDescriptors::Float { dim, .. } => StereoDescriptors::Float { data: &[], dim },
    }
}

/// Launcher hygiene: every buffer must hold `capacity` keypoints, since the kernel
/// indexes up to the (possibly device-side) count.
fn check_device_keypoints(
    k: &CudaStereoKeypoints,
    side: &'static str,
    ordinal: usize,
) -> Result<(), StereoMatchError> {
    let on = |o: usize| {
        if o != ordinal {
            Err(StereoMatchError::from(ImageError::DeviceMismatch))
        } else {
            Ok(())
        }
    };
    on(k.xy.ordinal())?;
    if let Some(o) = k.octaves {
        on(o.ordinal())?;
    }
    match k.descriptors {
        CudaStereoDescriptors::Binary { data, .. } => on(data.ordinal())?,
        CudaStereoDescriptors::Float { data, .. } => on(data.ordinal())?,
    }
    if let KeypointCount::Device { count, .. } = k.count {
        on(count.ordinal())?;
    }
    let cap = k.count.capacity();
    let len = |what: &'static str, got: usize, need: usize| {
        if got < need {
            Err(StereoMatchError::Length {
                side,
                what,
                got,
                expected: need,
            })
        } else {
            Ok(())
        }
    };
    len("xy floats", k.xy.len(), 2 * cap)?;
    if let Some(o) = k.octaves {
        len("octaves", o.len(), cap)?;
    }
    match k.descriptors {
        CudaStereoDescriptors::Binary { data, bytes } => {
            if bytes == 0 {
                return Err(StereoMatchError::Descriptors("zero-width descriptors"));
            }
            len("descriptor bytes", data.len(), cap * bytes)
        }
        CudaStereoDescriptors::Float { data, dim } => {
            if dim == 0 {
                return Err(StereoMatchError::Descriptors("zero-width descriptors"));
            }
            len("descriptor floats", data.len(), cap * dim)
        }
    }?;
    if let KeypointCount::Device { count, .. } = k.count {
        len("device count", count.len(), 1)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::matcher::tests::{kps, pair, pair_noisy, scene};
    use super::super::matcher::StereoKeypoints;
    use super::super::matcher::{SadRefine, StereoMatchConfig};
    use super::*;
    use cudarc::driver::CudaContext;

    /// Uploads host keypoints; real pipelines keep them device-resident end to end.
    struct UploadedKeypoints {
        xy: CudaSlice<f32>,
        octaves: Option<CudaSlice<u8>>,
        bin: Option<CudaSlice<u8>>,
        flt: Option<CudaSlice<f32>>,
        width: usize,
        n: usize,
    }

    impl UploadedKeypoints {
        fn new(stream: &Arc<CudaStream>, k: &StereoKeypoints) -> Result<Self, StereoMatchError> {
            // cudarc rejects zero-length allocations.
            fn pad<T: Default + Clone>(v: &[T]) -> Vec<T> {
                if v.is_empty() {
                    vec![T::default()]
                } else {
                    v.to_vec()
                }
            }
            let flat: Vec<f32> = k.xy.iter().flatten().copied().collect();
            let n = k.xy.len();
            let xy = stream.clone_htod(&pad(&flat)).map_err(cuda_err)?;
            let octaves = k
                .octaves
                .map(|o| stream.clone_htod(o))
                .transpose()
                .map_err(cuda_err)?;
            let (bin, flt, width) = match k.descriptors {
                StereoDescriptors::Binary { data, bytes } => (
                    Some(stream.clone_htod(&pad(data)).map_err(cuda_err)?),
                    None,
                    bytes,
                ),
                StereoDescriptors::Float { data, dim } => (
                    None,
                    Some(stream.clone_htod(&pad(data)).map_err(cuda_err)?),
                    dim,
                ),
            };
            Ok(Self {
                xy,
                octaves,
                bin,
                flt,
                width,
                n,
            })
        }

        fn view(&self) -> CudaStereoKeypoints<'_> {
            CudaStereoKeypoints {
                xy: &self.xy,
                octaves: self.octaves.as_ref(),
                descriptors: match (&self.bin, &self.flt) {
                    (Some(data), _) => CudaStereoDescriptors::Binary {
                        data,
                        bytes: self.width,
                    },
                    (_, Some(data)) => CudaStereoDescriptors::Float {
                        data,
                        dim: self.width,
                    },
                    _ => unreachable!("one descriptor buffer is always set"),
                },
                count: KeypointCount::Host(self.n),
            }
        }
    }

    /// Padded rows appended past a device count.
    const PAD: usize = 8;

    /// Host keypoints owning their arrays, for padding.
    struct OwnedKeypoints {
        xy: Vec<[f32; 2]>,
        octaves: Option<Vec<u8>>,
        bin: Vec<u8>,
        flt: Vec<f32>,
        binary: bool,
        width: usize,
    }

    impl OwnedKeypoints {
        /// `base` followed by up to [`PAD`] copies of `src`'s first keypoints (descriptors
        /// and octaves too), shifted left by `shift` px.
        fn padded(base: &StereoKeypoints, src: &StereoKeypoints, shift: f32) -> Self {
            let k = PAD.min(src.xy.len());
            let mut xy = base.xy.to_vec();
            xy.extend(src.xy[..k].iter().map(|&[x, y]| [x - shift, y]));
            let octaves = base.octaves.map(|o| {
                let mut o = o.to_vec();
                o.extend((0..k).map(|i| src.octaves.map_or(0, |s| s[i])));
                o
            });
            let (mut bin, mut flt) = (Vec::new(), Vec::new());
            let (binary, width) = match (base.descriptors, src.descriptors) {
                (
                    StereoDescriptors::Binary { data, bytes },
                    StereoDescriptors::Binary { data: s, .. },
                ) => {
                    bin = [data, &s[..k * bytes]].concat();
                    (true, bytes)
                }
                (
                    StereoDescriptors::Float { data, dim },
                    StereoDescriptors::Float { data: s, .. },
                ) => {
                    flt = [data, &s[..k * dim]].concat();
                    (false, dim)
                }
                _ => unreachable!("same descriptor kind"),
            };
            Self {
                xy,
                octaves,
                bin,
                flt,
                binary,
                width,
            }
        }

        fn view(&self) -> StereoKeypoints<'_> {
            StereoKeypoints {
                xy: &self.xy,
                octaves: self.octaves.as_deref(),
                descriptors: if self.binary {
                    StereoDescriptors::Binary {
                        data: &self.bin,
                        bytes: self.width,
                    }
                } else {
                    StereoDescriptors::Float {
                        data: &self.flt,
                        dim: self.width,
                    }
                },
            }
        }
    }

    fn run_both(
        cfg: StereoMatchConfig,
        lpyr: &[Image<u8, 1>],
        rpyr: &[Image<u8, 1>],
        l: &StereoKeypoints,
        r: &StereoKeypoints,
        device_count: bool,
    ) -> Result<(StereoMatches, StereoMatches), Box<dyn std::error::Error>> {
        let m = StereoMatcher::new(cfg)?;
        let mut cpu = StereoMatches::default();
        m.match_into(lpyr, rpyr, l, r, &mut cpu)?;

        let ctx = CudaContext::new(0)?;
        let stream = ctx.new_stream()?;
        let mut dm = m.to_cuda(&stream)?;
        let dl: Vec<Image<u8, 1>> = lpyr
            .iter()
            .map(|i| i.to_cuda(&stream))
            .collect::<Result<_, _>>()?;
        let dr: Vec<Image<u8, 1>> = rpyr
            .iter()
            .map(|i| i.to_cuda(&stream))
            .collect::<Result<_, _>>()?;
        let (nl, nr) = (l.xy.len(), r.xy.len());
        if !device_count {
            let (ul, ur) = (
                UploadedKeypoints::new(&stream, l)?,
                UploadedKeypoints::new(&stream, r)?,
            );
            let mut out = dm.alloc_matches(nl)?;
            dm.match_device(&dl, &dr, &ul.view(), &ur.view(), &mut out)?;
            return Ok((cpu, out.to_host(nl)?));
        }
        // Device counts below capacity on both sides; padded rows are matchable (left
        // copies, and right copies of the left at Hamming 0), so any read past the count
        // breaks parity.
        let pl = OwnedKeypoints::padded(l, l, 0.0);
        let pr = OwnedKeypoints::padded(r, l, 1.0);
        let (ul, ur) = (
            UploadedKeypoints::new(&stream, &pl.view())?,
            UploadedKeypoints::new(&stream, &pr.view())?,
        );
        let lc = stream.clone_htod(&[nl as i32])?;
        let rc = stream.clone_htod(&[nr as i32])?;
        let (mut lv, mut rv) = (ul.view(), ur.view());
        lv.count = KeypointCount::Device {
            count: &lc,
            capacity: pl.xy.len(),
        };
        rv.count = KeypointCount::Device {
            count: &rc,
            capacity: pr.xy.len(),
        };
        let mut out = dm.alloc_matches(pl.xy.len())?;
        // Twice on the same output: the tail must be reset, not left as is.
        for _ in 0..2 {
            dm.match_device(&dl, &dr, &lv, &rv, &mut out)?;
        }
        let all = out.to_host(pl.xy.len())?;
        assert!(
            all.right_idx[nl..].iter().all(|&i| i == -1),
            "tail not reset"
        );
        assert!(
            all.u_right[nl..].iter().all(|&u| u == -1.0),
            "tail not reset"
        );
        Ok((cpu, out.to_host(nl)?))
    }

    fn bits(m: &StereoMatches) -> (Vec<u32>, Vec<u32>, Vec<i32>) {
        (
            m.u_right.iter().map(|v| v.to_bits()).collect(),
            m.depth.iter().map(|v| v.to_bits()).collect(),
            m.right_idx.clone(),
        )
    }

    /// The CUDA matcher must return bit-identical results to the CPU reference, for both
    /// descriptor kinds, with and without SAD, and with a device-side count.
    #[test]
    fn cuda_matches_cpu_bit_exact() -> Result<(), Box<dyn std::error::Error>> {
        let d = 7.3;
        let (li, ri) = pair_noisy(d);
        let s = scene(d, 11);
        for binary in [true, false] {
            for sad in [
                Some(SadRefine::default()),
                // Widest window the warp holds: 2 * (7 + 8) + 1 = 31 columns.
                Some(SadRefine {
                    half_window: 7,
                    search_range: 8,
                    fit: SubPixelFit::Parabola,
                    ..SadRefine::default()
                }),
                // Widest patch: 29x29 → 428 911 SAD bins, coarse buckets of 2^11.
                Some(SadRefine {
                    half_window: 14,
                    search_range: 1,
                    ..SadRefine::default()
                }),
                None,
            ] {
                for device_count in [false, true] {
                    let mut c = StereoMatchConfig::new(300.0, 0.1);
                    c.sad = sad;
                    let (l, r) = (
                        kps(&s.lxy, &s.lbin, &s.lf, binary),
                        kps(&s.rxy, &s.rbin, &s.rf, binary),
                    );
                    // SAD off also runs image-less: unbounded rows, the brute-force scan.
                    let images = if sad.is_none() && device_count { 0 } else { 1 };
                    let (lp, rp) = (vec![li.clone(); images], vec![ri.clone(); images]);
                    let (cpu, gpu) = run_both(c, &lp, &rp, &l, &r, device_count)?;
                    assert!(cpu.num_matched() > s.lxy.len() / 2, "test scene must match");
                    if let Some(sr) = sad.filter(|sr| sr.median_factor.is_some()) {
                        // Prove the median reject fired, so its GPU twin was compared.
                        let mut c2 = StereoMatchConfig::new(300.0, 0.1);
                        c2.sad = Some(SadRefine {
                            median_factor: None,
                            ..sr
                        });
                        let mut all = StereoMatches::default();
                        StereoMatcher::new(c2)?.match_into(&lp, &rp, &l, &r, &mut all)?;
                        assert!(
                            all.num_matched() > cpu.num_matched(),
                            "median reject never fired"
                        );
                    }
                    assert_eq!(
                        bits(&cpu),
                        bits(&gpu),
                        "binary={binary} sad={} dev={device_count}",
                        sad.is_some()
                    );
                }
            }
        }
        Ok(())
    }

    /// Octave path: two-level pyramid, per-octave row band and SAD level, ±1 gate.
    #[test]
    fn cuda_matches_cpu_bit_exact_with_octaves() -> Result<(), Box<dyn std::error::Error>> {
        use super::super::matcher::tests::{texture, H, W};
        use kornia_image::ImageSize;
        let d = 9.4f32;
        let (l0, r0) = pair(d);
        // Level 1 = the same scene sampled at half resolution.
        let size1 = ImageSize {
            width: W / 2,
            height: H / 2,
        };
        let lvl = |shift: f32| -> Result<Image<u8, 1>, ImageError> {
            let mut v = vec![0u8; size1.width * size1.height];
            for y in 0..size1.height {
                for x in 0..size1.width {
                    v[y * size1.width + x] = texture(2.0 * x as f32 + shift, 2.0 * y as f32);
                }
            }
            Image::new(size1, v)
        };
        let (l1, r1) = (lvl(0.0)?, lvl(d)?);
        let s = scene(d, 5);
        let mut octs_l = vec![0u8; s.lxy.len()];
        let mut octs_r = vec![0u8; s.rxy.len()];
        for (i, o) in octs_l.iter_mut().enumerate() {
            *o = (i % 3 == 0) as u8;
        }
        for (j, o) in octs_r.iter_mut().enumerate() {
            *o = (j % 4 == 0) as u8 + (j % 7 == 0) as u8; // some at octave 2: gated / invalid
        }
        let mut c = StereoMatchConfig::new(300.0, 0.1);
        c.scale_factor = 2.0;
        c.n_levels = 2;
        for binary in [true, false] {
            let mut l = kps(&s.lxy, &s.lbin, &s.lf, binary);
            let mut r = kps(&s.rxy, &s.rbin, &s.rf, binary);
            l.octaves = Some(&octs_l);
            r.octaves = Some(&octs_r);
            let pyr_l = [l0.clone(), l1.clone()];
            let pyr_r = [r0.clone(), r1.clone()];
            let (cpu, gpu) = run_both(c.clone(), &pyr_l, &pyr_r, &l, &r, false)?;
            assert!(cpu.num_matched() > s.lxy.len() / 3, "test scene must match");
            assert_eq!(bits(&cpu), bits(&gpu), "binary={binary}");
        }
        Ok(())
    }

    /// Ties resolve to the lowest right index on both backends, even though a GPU row
    /// bucket is filled in atomic (arbitrary) order.
    #[test]
    fn cuda_ties_go_to_the_lowest_right_index() -> Result<(), Box<dyn std::error::Error>> {
        let (li, ri) = pair(4.0);
        let lxy = [[200.0f32, 120.0]];
        // 0: a worse decoy; 1..=40: identical to the left descriptor, same row, in range.
        let n = 41;
        let rxy: Vec<[f32; 2]> = (0..n).map(|k| [196.0 - k as f32, 120.0]).collect();
        let lbin = [0u8; 32];
        let mut rbin = vec![0u8; 32 * n];
        rbin[..32].fill(1);
        let lf = [0.125f32; 64];
        let mut rf = vec![0.125f32; 64 * n];
        rf[..64].fill(0.0);
        let sad_nomedian = SadRefine {
            median_factor: None,
            ..SadRefine::default()
        };
        for binary in [true, false] {
            for sad in [None, Some(sad_nomedian)] {
                // SAD off also runs image-less (unbucketed brute force).
                for images in if sad.is_none() { vec![0, 1] } else { vec![1] } {
                    let mut c = StereoMatchConfig::new(300.0, 0.1);
                    c.sad = sad;
                    c.min_similarity = 0.5;
                    let (l, r) = (kps(&lxy, &lbin, &lf, binary), kps(&rxy, &rbin, &rf, binary));
                    let (lp, rp) = (vec![li.clone(); images], vec![ri.clone(); images]);
                    let (cpu, gpu) = run_both(c, &lp, &rp, &l, &r, false)?;
                    let tag = format!("binary={binary} sad={} images={images}", sad.is_some());
                    assert_eq!(cpu.right_idx, vec![1], "cpu {tag}");
                    assert_eq!(bits(&cpu), bits(&gpu), "{tag}");
                }
            }
        }
        Ok(())
    }

    /// A NaN similarity never wins (CPU `s > sim` from -inf), even when it is the first
    /// candidate the GPU visits.
    #[test]
    fn cuda_nan_descriptor_never_wins() -> Result<(), Box<dyn std::error::Error>> {
        let (li, _) = pair(5.0);
        let lxy = [[200.0f32, 120.0]];
        let rxy = [[190.0f32, 120.0], [195.0, 120.0]];
        let lf = [0.125f32; 64];
        let mut rf = vec![0.125f32; 128];
        rf[0] = f32::NAN;
        let mut c = StereoMatchConfig::new(300.0, 0.1);
        c.sad = None;
        c.min_similarity = 0.5;
        for images in [0, 1] {
            for _ in 0..3 {
                let (l, r) = (kps(&lxy, &[], &lf, false), kps(&rxy, &[], &rf, false));
                let lp = vec![li.clone(); images];
                let (cpu, gpu) = run_both(c.clone(), &lp, &lp, &l, &r, false)?;
                assert_eq!(cpu.right_idx, vec![1], "images={images}");
                assert_eq!(bits(&cpu), bits(&gpu), "images={images}");
            }
        }
        Ok(())
    }

    /// An infinite row band must size the bucket buffer by the image, not overflow.
    #[test]
    fn cuda_infinite_row_band_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (li, ri) = pair(7.3);
        let s = scene(7.3, 13);
        let mut c = StereoMatchConfig::new(300.0, 0.1);
        c.sad = None;
        c.row_band = f32::INFINITY;
        for binary in [true, false] {
            let (l, r) = (
                kps(&s.lxy, &s.lbin, &s.lf, binary),
                kps(&s.rxy, &s.rbin, &s.rf, binary),
            );
            let (cpu, gpu) = run_both(
                c.clone(),
                std::slice::from_ref(&li),
                std::slice::from_ref(&ri),
                &l,
                &r,
                false,
            )?;
            assert!(cpu.num_matched() > 0, "binary={binary}");
            assert_eq!(bits(&cpu), bits(&gpu), "binary={binary}");
        }
        Ok(())
    }

    /// A zero-height level 0 leaves everything unmatched on both backends (SAD off, so
    /// the host image only bounds the rows).
    #[test]
    fn cuda_zero_height_image_matches_nothing() -> Result<(), Box<dyn std::error::Error>> {
        use kornia_image::ImageSize;
        let s = scene(7.0, 4);
        let empty = Image::<u8, 1>::new(
            ImageSize {
                width: 0,
                height: 0,
            },
            vec![],
        )?;
        let mut c = StereoMatchConfig::new(300.0, 0.1);
        c.sad = None;
        let m = StereoMatcher::new(c)?;
        let (l, r) = (
            kps(&s.lxy, &s.lbin, &s.lf, true),
            kps(&s.rxy, &s.rbin, &s.rf, true),
        );
        let lp = [empty];
        let mut cpu = StereoMatches::default();
        m.match_into(&lp, &lp, &l, &r, &mut cpu)?;
        let stream = CudaContext::new(0)?.new_stream()?;
        let mut dm = m.to_cuda(&stream)?;
        let (ul, ur) = (
            UploadedKeypoints::new(&stream, &l)?,
            UploadedKeypoints::new(&stream, &r)?,
        );
        let mut out = dm.alloc_matches(s.lxy.len())?;
        dm.match_device(&lp, &lp, &ul.view(), &ur.view(), &mut out)?;
        let gpu = out.to_host(s.lxy.len())?;
        assert_eq!(cpu.num_matched(), 0);
        assert_eq!(bits(&cpu), bits(&gpu));
        Ok(())
    }

    /// Median 0 (byte-identical patches) and right-only octaves, CPU == GPU.
    #[test]
    fn cuda_matches_cpu_zero_median_and_right_only_octaves(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (li, ri) = pair(6.0);
        let s = scene(6.0, 9);
        let octs_r: Vec<u8> = (0..s.rxy.len()).map(|j| (j % 3) as u8).collect();
        let mut c = StereoMatchConfig::new(300.0, 0.1);
        c.scale_factor = 2.0;
        c.n_levels = 3;
        let half = |i: &Image<u8, 1>, k: usize| -> Result<Image<u8, 1>, ImageError> {
            let (w, h) = (i.width() >> k, i.height() >> k);
            let src = i.as_slice();
            let v = (0..w * h)
                .map(|p| src[((p / w) << k) * i.width() + ((p % w) << k)])
                .collect();
            Image::new(
                kornia_image::ImageSize {
                    width: w,
                    height: h,
                },
                v,
            )
        };
        let lp = [li.clone(), half(&li, 1)?, half(&li, 2)?];
        let rp = [ri.clone(), half(&ri, 1)?, half(&ri, 2)?];
        for binary in [true, false] {
            let l = kps(&s.lxy, &s.lbin, &s.lf, binary);
            let mut r = kps(&s.rxy, &s.rbin, &s.rf, binary);
            r.octaves = Some(&octs_r);
            let (cpu, gpu) = run_both(c.clone(), &lp, &rp, &l, &r, false)?;
            assert!(cpu.num_matched() > s.lxy.len() / 4, "scene must match");
            assert_eq!(bits(&cpu), bits(&gpu), "binary={binary}");
        }
        Ok(())
    }

    #[test]
    fn host_pyramid_is_a_typed_error() -> Result<(), Box<dyn std::error::Error>> {
        let (li, ri) = pair(5.0);
        let s = scene(5.0, 2);
        let m = StereoMatcher::new(StereoMatchConfig::new(300.0, 0.1))?;
        let ctx = CudaContext::new(0)?;
        let stream = ctx.new_stream()?;
        let mut dm = m.to_cuda(&stream)?;
        let k = UploadedKeypoints::new(&stream, &kps(&s.lxy, &s.lbin, &s.lf, true))?;
        let mut out = dm.alloc_matches(s.lxy.len())?;
        let r = dm.match_device(&[li], &[ri], &k.view(), &k.view(), &mut out);
        assert!(matches!(
            r,
            Err(StereoMatchError::Image(ImageError::HostResident))
        ));
        Ok(())
    }
}
