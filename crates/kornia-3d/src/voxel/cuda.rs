//! CUDA twins of [`VoxelDownsampler::downsample_into`] and
//! [`VoxelDownsampler::neighbour_counts_into`]: device-resident points in,
//! device-resident results out, no host sync per call.
//!
//! The pipeline is sort-by-key, mirroring the host:
//!
//! 1. one thread per point writes the 128-bit sort word `(x ^ 2³¹,
//!    y ^ 2³¹, z ^ 2³¹, index)` as a `uint4`, padding to a power of two
//!    with all-ones words that sort last;
//! 2. a bitonic sort of the words: each 2048-word tile is sorted in shared
//!    memory, then the merge stages wider than a tile run as global
//!    compare-exchange passes and the narrower ones in shared memory again
//!    (`1 + Σ (log₂ k − 10)` launches for `k = 4096 … len`). The order is
//!    a total order on unique words, so the result equals the host sort;
//! 3. run heads (key differs from the previous word) are scanned per
//!    1024-word tile, the tile totals by one block, and each head writes
//!    its run start (and, for the neighbour count, its key);
//! 4. one thread per voxel reduces its run sequentially in ascending point
//!    index, with the host expressions (`fmad=false`), or counts its
//!    occupied neighbours by binary search over the sorted unique keys.
//!
//! An invalid point (non-finite or key outside `i32`) cannot fail the
//! enqueue without a sync: the smallest offending index is recorded on the
//! device (`atomicMax` of `u32::MAX − index`) and reported as
//! [`VoxelError::PointOutOfRange`] by the readback methods, the same error
//! the host returns eagerly.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, LaunchConfig, ValidAsZeroBits};
use kornia_tensor::CudaKernel;

use super::downsample::DownsampleMode;
use super::{VoxelConnectivity, VoxelError, VoxelLayout};
use crate::pointcloud::PointCloud;

#[cfg(doc)]
use super::VoxelDownsampler;

/// Largest cloud of the device path: the padded sort length must fit `u32`.
pub const MAX_CUDA_VOXEL_POINTS: usize = 1 << 31;

/// Words per shared-memory sort tile (`SORT_TILE` in KERNELS).
const SORT_TILE: usize = 2048;
/// Threads of a sort tile block (`SORT_THREADS` in KERNELS).
const SORT_THREADS: u32 = 1024;
/// Words per run-head scan tile, one per thread (`SCAN_TILE` in KERNELS).
const SCAN_TILE: usize = 1024;

/// `mode` values of `voxel_reduce`.
const MODE_CENTROID: i32 = 0;
const MODE_FIRST: i32 = 1;
const MODE_NEAREST: i32 = 2;

/// Voxel layout as passed to the kernels.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
// The fields are read on the device only.
#[allow(dead_code)]
struct LayoutParams {
    size: f64,
    ox: f64,
    oy: f64,
    oz: f64,
}

// SAFETY: plain-old-data, repr(C), four f64 fields (no padding); mirrored
// field-for-field by `struct Lay` in KERNELS.
unsafe impl DeviceRepr for LayoutParams {}

impl From<&VoxelLayout> for LayoutParams {
    fn from(l: &VoxelLayout) -> Self {
        let o = l.origin();
        Self {
            size: l.voxel_size(),
            ox: o[0],
            oy: o[1],
            oz: o[2],
        }
    }
}

const KERNELS: &str = r#"
#define SORT_TILE 2048
#define SORT_THREADS 1024
#define SCAN_TILE 1024
#define FULL 0xffffffffu
#define BIAS 0x80000000u
#define MODE_CENTROID 0
#define MODE_NEAREST 2

struct Lay { double size, ox, oy, oz; };

/* ---- mirrors of voxel/mod.rs and voxel/downsample.rs; keep the expression trees identical ---- */

/* axis_key */
__device__ __forceinline__ bool axis_key(double v, double o, double s, int* k) {
    double f = floor((v - o) / s);
    if (!(f >= -2147483648.0 && f <= 2147483647.0)) return false;
    *k = (int)f;
    return true;
}
/* axis_centre */
__device__ __forceinline__ double axis_centre(int k, double o, double s) {
    return o + ((double)k + 0.5) * s;
}
/* the `dist` closure of VoxelDownsampler::reduce */
__device__ __forceinline__ double dist2(const double* p, double cx, double cy, double cz) {
    double dx = p[0] - cx, dy = p[1] - cy, dz = p[2] - cz;
    return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ bool word_gt(uint4 a, uint4 b) {
    if (a.x != b.x) return a.x > b.x;
    if (a.y != b.y) return a.y > b.y;
    if (a.z != b.z) return a.z > b.z;
    return a.w > b.w;
}
__device__ __forceinline__ bool key_ne(uint4 a, uint4 b) {
    return a.x != b.x || a.y != b.y || a.z != b.z;
}

/* ---- 1. sort words ---- */

extern "C" __global__ void voxel_keys(const double* __restrict__ xyz, unsigned int n,
                                      unsigned int len, Lay L, uint4* __restrict__ words,
                                      unsigned int* __restrict__ err) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    uint4 w = make_uint4(FULL, FULL, FULL, FULL);
    if (i < n) {
        const double* p = xyz + 3 * (size_t)i;
        int kx, ky, kz;
        if (axis_key(p[0], L.ox, L.size, &kx) && axis_key(p[1], L.oy, L.size, &ky) &&
            axis_key(p[2], L.oz, L.size, &kz)) {
            w = make_uint4((unsigned int)kx ^ BIAS, (unsigned int)ky ^ BIAS,
                           (unsigned int)kz ^ BIAS, i);
        } else {
            w.w = i;
            atomicMax(err, FULL - i);
        }
    }
    words[i] = w;
}

/* ---- 2. bitonic sort ---- */

/* Lower element of compare-exchange pair `t` at distance `j`. */
__device__ __forceinline__ unsigned int pair_lo(unsigned int t, unsigned int j) {
    return 2u * j * (t / j) + (t % j);
}

__device__ __forceinline__ void tile_load(const uint4* words, unsigned int base, uint4* s) {
    s[threadIdx.x] = words[base + threadIdx.x];
    s[threadIdx.x + SORT_THREADS] = words[base + threadIdx.x + SORT_THREADS];
}
__device__ __forceinline__ void tile_store(uint4* words, unsigned int base, const uint4* s) {
    words[base + threadIdx.x] = s[threadIdx.x];
    words[base + threadIdx.x + SORT_THREADS] = s[threadIdx.x + SORT_THREADS];
}
/* Merge stages j = j0 .. 1 of stage k, in shared memory. */
__device__ __forceinline__ void tile_merge(uint4* s, unsigned int base, unsigned int k,
                                           unsigned int j0) {
    for (unsigned int j = j0; j > 0; j >>= 1) {
        unsigned int i = pair_lo(threadIdx.x, j);
        bool asc = ((base + i) & k) == 0;
        uint4 a = s[i], b = s[i + j];
        if (word_gt(a, b) == asc) {
            s[i] = b;
            s[i + j] = a;
        }
        __syncthreads();
    }
}

/* Every stage k <= SORT_TILE: tiles end sorted, alternating direction. */
extern "C" __global__ void sort_tile(uint4* __restrict__ words) {
    __shared__ uint4 s[SORT_TILE];
    unsigned int base = blockIdx.x * SORT_TILE;
    tile_load(words, base, s);
    __syncthreads();
    for (unsigned int k = 2; k <= SORT_TILE; k <<= 1) tile_merge(s, base, k, k >> 1);
    tile_store(words, base, s);
}

/* One global merge pass (j >= SORT_TILE) of stage k. */
extern "C" __global__ void sort_step(uint4* __restrict__ words, unsigned int j, unsigned int k,
                                     unsigned int pairs) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= pairs) return;
    unsigned int i = pair_lo(t, j);
    bool asc = (i & k) == 0;
    uint4 a = words[i], b = words[i + j];
    if (word_gt(a, b) == asc) {
        words[i] = b;
        words[i + j] = a;
    }
}

/* The merge passes j < SORT_TILE of stage k. */
extern "C" __global__ void sort_merge_tile(uint4* __restrict__ words, unsigned int k) {
    __shared__ uint4 s[SORT_TILE];
    unsigned int base = blockIdx.x * SORT_TILE;
    tile_load(words, base, s);
    __syncthreads();
    tile_merge(s, base, k, SORT_TILE >> 1);
    tile_store(words, base, s);
}

/* ---- 3. runs ---- */

/* Exclusive scan of one value per thread over a 1024-thread block; `*sum`
   gets the block total. Every thread must call it. */
__device__ __forceinline__ unsigned int block_scan(unsigned int v, unsigned int* sum) {
    __shared__ unsigned int warp_sum[32];
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int incl = v;
    for (unsigned int off = 1; off < 32; off <<= 1) {
        unsigned int o = __shfl_up_sync(FULL, incl, off);
        if (lane >= off) incl += o;
    }
    if (lane == 31) warp_sum[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        unsigned int w = warp_sum[lane];
        unsigned int wi = w;
        for (unsigned int off = 1; off < 32; off <<= 1) {
            unsigned int o = __shfl_up_sync(FULL, wi, off);
            if (lane >= off) wi += o;
        }
        warp_sum[lane] = wi - w;
        if (lane == 31) *sum = wi;
    }
    __syncthreads();
    return warp_sum[warp] + incl - v;
}

__device__ __forceinline__ unsigned int run_head(const uint4* words, unsigned int i) {
    return (i == 0 || key_ne(words[i], words[i - 1])) ? 1u : 0u;
}

extern "C" __global__ void run_tile_scan(const uint4* __restrict__ words, unsigned int n,
                                         unsigned int* __restrict__ excl,
                                         unsigned int* __restrict__ tile_sum) {
    __shared__ unsigned int total;
    unsigned int i = blockIdx.x * SCAN_TILE + threadIdx.x;
    unsigned int v = i < n ? run_head(words, i) : 0u;
    unsigned int ex = block_scan(v, &total);
    if (i < n) excl[i] = ex;
    if (threadIdx.x == 0) tile_sum[blockIdx.x] = total;
}

/* One block: exclusive scan of the tile totals, each thread a contiguous
   chunk. `*voxels` gets the number of runs. */
extern "C" __global__ void run_scan_tiles(const unsigned int* __restrict__ tile_sum,
                                          unsigned int tiles,
                                          unsigned int* __restrict__ tile_off,
                                          unsigned int* __restrict__ voxels) {
    __shared__ unsigned int total;
    unsigned int chunk = (tiles + blockDim.x - 1) / blockDim.x;
    unsigned int b0 = min(tiles, threadIdx.x * chunk);
    unsigned int b1 = min(tiles, b0 + chunk);
    unsigned int v = 0;
    for (unsigned int b = b0; b < b1; ++b) v += tile_sum[b];
    unsigned int a = block_scan(v, &total);
    for (unsigned int b = b0; b < b1; ++b) {
        tile_off[b] = a;
        a += tile_sum[b];
    }
    if (threadIdx.x == 0) *voxels = total;
}

extern "C" __global__ void run_finalize(const uint4* __restrict__ words, unsigned int n,
                                        const unsigned int* __restrict__ excl,
                                        const unsigned int* __restrict__ tile_off,
                                        unsigned int* __restrict__ run_start,
                                        unsigned int* __restrict__ run_of,
                                        uint4* __restrict__ unique) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int h = run_head(words, i);
    /* word 0 is always a head, so the inclusive count is >= 1 */
    unsigned int r = tile_off[i / SCAN_TILE] + excl[i] + h - 1u;
    if (h) {
        run_start[r] = i;
        if (unique) unique[r] = words[i];
    }
    if (run_of) run_of[i] = r;
}

/* ---- 4. per-voxel work ---- */

/* VoxelDownsampler::reduce */
extern "C" __global__ void voxel_reduce(const double* __restrict__ xyz,
                                        const uint4* __restrict__ words, unsigned int n,
                                        const unsigned int* __restrict__ run_start,
                                        const unsigned int* __restrict__ voxels, Lay L, int mode,
                                        double* __restrict__ out_xyz,
                                        unsigned int* __restrict__ out_index,
                                        unsigned int* __restrict__ out_count) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int m = *voxels;
    if (v == 0) *out_count = m;
    if (v >= m) return;
    unsigned int b = run_start[v];
    unsigned int e = v + 1 < m ? run_start[v + 1] : n;
    uint4 f = words[b];
    unsigned int best = f.w;
    double rx, ry, rz;
    if (mode == MODE_CENTROID) {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (unsigned int r = b; r < e; ++r) {
            const double* p = xyz + 3 * (size_t)words[r].w;
            sx += p[0];
            sy += p[1];
            sz += p[2];
        }
        double c = (double)(e - b);
        rx = sx / c;
        ry = sy / c;
        rz = sz / c;
    } else {
        if (mode == MODE_NEAREST) {
            double cx = axis_centre((int)(f.x ^ BIAS), L.ox, L.size);
            double cy = axis_centre((int)(f.y ^ BIAS), L.oy, L.size);
            double cz = axis_centre((int)(f.z ^ BIAS), L.oz, L.size);
            double bd = dist2(xyz + 3 * (size_t)best, cx, cy, cz);
            for (unsigned int r = b + 1; r < e; ++r) {
                unsigned int idx = words[r].w;
                double d = dist2(xyz + 3 * (size_t)idx, cx, cy, cz);
                if (d < bd) {
                    bd = d;
                    best = idx;
                }
            }
        }
        const double* p = xyz + 3 * (size_t)best;
        rx = p[0];
        ry = p[1];
        rz = p[2];
    }
    out_xyz[3 * (size_t)v] = rx;
    out_xyz[3 * (size_t)v + 1] = ry;
    out_xyz[3 * (size_t)v + 2] = rz;
    if (out_index) out_index[v] = best;
}

__device__ __forceinline__ bool key_lt(uint4 k, unsigned int x, unsigned int y, unsigned int z) {
    return k.x < x || (k.x == x && (k.y < y || (k.y == y && k.z < z)));
}
__device__ __forceinline__ bool key_le(uint4 k, unsigned int x, unsigned int y, unsigned int z) {
    return k.x < x || (k.x == x && (k.y < y || (k.y == y && k.z <= z)));
}

/* adjacent_occupied in downsample.rs */
extern "C" __global__ void voxel_count(const uint4* __restrict__ unique,
                                       const unsigned int* __restrict__ voxels, int max_l1,
                                       unsigned char* __restrict__ vcount) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int m = *voxels;
    if (v >= m) return;
    uint4 key = unique[v];
    const long long kmax = 0xffffffffLL;
    long long x = key.x, y = key.y, z = key.z;
    unsigned int count = 0;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int lxy = abs(dx) + abs(dy);
            if (lxy > max_l1) continue;
            long long tx = x + dx, ty = y + dy;
            if (tx < 0 || tx > kmax || ty < 0 || ty > kmax) continue;
            unsigned int ux = (unsigned int)tx, uy = (unsigned int)ty;
            unsigned int lo = (unsigned int)(z - 1 < 0 ? 0 : z - 1);
            unsigned int hi = (unsigned int)(z + 1 > kmax ? kmax : z + 1);
            unsigned int a = 0, c = m;
            while (a < c) {
                unsigned int mid = a + ((c - a) >> 1);
                if (key_lt(unique[mid], ux, uy, lo)) a = mid + 1; else c = mid;
            }
            for (unsigned int j = a; j < m && key_le(unique[j], ux, uy, hi); ++j) {
                long long dz = (long long)unique[j].z - z;
                int l1 = lxy + (int)(dz < 0 ? -dz : dz);
                if (l1 > 0 && l1 <= max_l1) ++count;
            }
        }
    }
    vcount[v] = (unsigned char)count;
}

extern "C" __global__ void scatter_counts(const uint4* __restrict__ words,
                                          const unsigned int* __restrict__ run_of,
                                          unsigned int n,
                                          const unsigned char* __restrict__ vcount,
                                          unsigned char* __restrict__ counts) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    counts[words[i].w] = vcount[run_of[i]];
}
"#;

/// Kernel names, in the order [`CudaVoxelFilter::new`] unpacks them.
const NAMES: [&str; 10] = [
    "voxel_keys",
    "sort_tile",
    "sort_step",
    "sort_merge_tile",
    "run_tile_scan",
    "run_scan_tiles",
    "run_finalize",
    "voxel_reduce",
    "voxel_count",
    "scatter_counts",
];

fn cuda_err(e: impl std::fmt::Display) -> VoxelError {
    VoxelError::Cuda(e.to_string())
}

/// Grows `buf` to at least `n` elements (contents unspecified).
fn ensure_len<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    buf: &mut CudaSlice<T>,
    n: usize,
) -> Result<(), VoxelError> {
    if buf.len() < n {
        *buf = stream.alloc_zeros::<T>(n).map_err(cuda_err)?;
    }
    Ok(())
}

/// Decodes the device error word: `0` = no error, else `u32::MAX − index`.
fn check_error_word(stream: &Arc<CudaStream>, err: &CudaSlice<u32>) -> Result<(), VoxelError> {
    let w = stream.clone_dtoh(err).map_err(cuda_err)?;
    match w.first().copied() {
        Some(v) if v != 0 => Err(VoxelError::PointOutOfRange {
            index: (u32::MAX - v) as usize,
        }),
        _ => Ok(()),
    }
}

/// Device-resident downsampling output, caller-owned and reusable across
/// calls: `count` voxels, each with its point (`f64` `x, y, z`) and the
/// input index of the point it came from.
pub struct CudaVoxelCloud {
    xyz: CudaSlice<f64>,
    index: CudaSlice<u32>,
    count: CudaSlice<u32>,
    err: CudaSlice<u32>,
    capacity: usize,
    stream: Arc<CudaStream>,
}

impl CudaVoxelCloud {
    /// Voxel capacity.
    ///
    /// # Returns
    ///
    /// The number of voxels the buffers hold; a call needs one slot per
    /// input point.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Device points.
    ///
    /// # Returns
    ///
    /// The `3 × capacity` `f64` buffer; the first `3 × count` hold one point
    /// per voxel in ascending key order.
    pub fn xyz(&self) -> &CudaSlice<f64> {
        &self.xyz
    }

    /// Device source indices.
    ///
    /// # Returns
    ///
    /// The `capacity` `u32` buffer: per voxel, the input index of its point
    /// (the lowest index of the voxel for [`DownsampleMode::Centroid`]).
    pub fn index(&self) -> &CudaSlice<u32> {
        &self.index
    }

    /// Device voxel count.
    ///
    /// # Returns
    ///
    /// One `u32` written by the last call, for device consumers that must
    /// not sync. Meaningless when the call recorded an invalid point.
    pub fn count(&self) -> &CudaSlice<u32> {
        &self.count
    }

    /// Reads the voxel count. Synchronizes the stream.
    ///
    /// # Returns
    ///
    /// The number of voxels of the last call.
    ///
    /// # Errors
    ///
    /// [`VoxelError::PointOutOfRange`] if the last call met an invalid
    /// point; [`VoxelError::Cuda`] on a copy failure.
    pub fn len_to_host(&self) -> Result<usize, VoxelError> {
        check_error_word(&self.stream, &self.err)?;
        let c = self.stream.clone_dtoh(&self.count).map_err(cuda_err)?;
        Ok((c.first().copied().unwrap_or(0) as usize).min(self.capacity))
    }

    /// Copies the points to a host [`PointCloud`]. Synchronizes the stream.
    ///
    /// # Returns
    ///
    /// The same points, in the same order, as the host
    /// [`VoxelDownsampler::downsample_into`] of the same input.
    ///
    /// # Errors
    ///
    /// As [`len_to_host`](Self::len_to_host).
    pub fn to_host(&self) -> Result<PointCloud, VoxelError> {
        let m = self.len_to_host()?;
        if m == 0 {
            return Ok(PointCloud::new(Vec::new(), None, None));
        }
        let xyz = self
            .stream
            .clone_dtoh(&self.xyz.slice(0..3 * m))
            .map_err(cuda_err)?;
        let points = xyz.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        Ok(PointCloud::new(points, None, None))
    }

    /// Copies the source indices to the host. Synchronizes the stream.
    ///
    /// # Returns
    ///
    /// One input index per voxel.
    ///
    /// # Errors
    ///
    /// As [`len_to_host`](Self::len_to_host).
    pub fn index_to_host(&self) -> Result<Vec<u32>, VoxelError> {
        let m = self.len_to_host()?;
        if m == 0 {
            return Ok(Vec::new());
        }
        self.stream
            .clone_dtoh(&self.index.slice(0..m))
            .map_err(cuda_err)
    }
}

/// Device-resident per-point neighbour counts, caller-owned and reusable.
pub struct CudaNeighbourCounts {
    counts: CudaSlice<u8>,
    err: CudaSlice<u32>,
    len: usize,
    capacity: usize,
    stream: Arc<CudaStream>,
}

impl CudaNeighbourCounts {
    /// Point capacity.
    ///
    /// # Returns
    ///
    /// The number of points the buffer holds.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Points of the last call.
    ///
    /// # Returns
    ///
    /// The `n` of the last [`CudaVoxelFilter::neighbour_counts`].
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the last call had no point.
    ///
    /// # Returns
    ///
    /// `true` if [`len`](Self::len) is `0`.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Device counts.
    ///
    /// # Returns
    ///
    /// The `capacity` `u8` buffer; the first [`len`](Self::len) hold one
    /// count per input point, in input order.
    pub fn counts(&self) -> &CudaSlice<u8> {
        &self.counts
    }

    /// Copies the counts to the host. Synchronizes the stream.
    ///
    /// # Returns
    ///
    /// The same counts as the host [`VoxelDownsampler::neighbour_counts_into`].
    ///
    /// # Errors
    ///
    /// [`VoxelError::PointOutOfRange`] if the last call met an invalid
    /// point; [`VoxelError::Cuda`] on a copy failure.
    pub fn to_host(&self) -> Result<Vec<u8>, VoxelError> {
        check_error_word(&self.stream, &self.err)?;
        if self.len == 0 {
            return Ok(Vec::new());
        }
        self.stream
            .clone_dtoh(&self.counts.slice(0..self.len))
            .map_err(cuda_err)
    }
}

/// CUDA voxel downsampling and neighbour counting on one stream: the
/// kernels are compiled once at construction and the scratch buffers are
/// reused across calls. Work is enqueued without host syncs; read results
/// after syncing the stream.
///
/// Input points are a device `f64` buffer of interleaved `x, y, z`
/// (`stream.clone_htod(points.as_flattened())` uploads a host slice).
pub struct CudaVoxelFilter {
    stream: Arc<CudaStream>,
    k_keys: CudaKernel,
    k_sort_tile: CudaKernel,
    k_sort_step: CudaKernel,
    k_sort_merge: CudaKernel,
    k_tile_scan: CudaKernel,
    k_scan_tiles: CudaKernel,
    k_finalize: CudaKernel,
    k_reduce: CudaKernel,
    k_count: CudaKernel,
    k_scatter: CudaKernel,
    /// Sort words, 4 `u32` (one `uint4`) each.
    words: CudaSlice<u32>,
    excl: CudaSlice<u32>,
    tile_sum: CudaSlice<u32>,
    tile_off: CudaSlice<u32>,
    run_start: CudaSlice<u32>,
    run_of: CudaSlice<u32>,
    voxels: CudaSlice<u32>,
    /// Unique keys, 4 `u32` each.
    unique: CudaSlice<u32>,
    vcount: CudaSlice<u8>,
}

impl CudaVoxelFilter {
    /// Compiles the kernels (nvrtc) for `stream`'s device and allocates the
    /// scratch buffers. Synchronizes `stream` once, so compile and
    /// allocation failures surface here rather than on the first call.
    ///
    /// # Arguments
    ///
    /// * `stream` - Stream every call of this filter enqueues on.
    ///
    /// # Returns
    ///
    /// A filter ready for [`downsample`](Self::downsample) and
    /// [`neighbour_counts`](Self::neighbour_counts).
    ///
    /// # Errors
    ///
    /// [`VoxelError::Cuda`] on a compile or allocation failure.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cudarc::driver::CudaContext;
    /// use kornia_3d::voxel::{CudaVoxelFilter, DownsampleMode, VoxelLayout};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let stream = CudaContext::new(0)?.default_stream();
    /// let mut filter = CudaVoxelFilter::new(&stream)?;
    /// let points = vec![[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.4, 0.1, 0.1]];
    /// let device = stream.clone_htod(points.as_flattened())?;
    /// let mut out = filter.alloc_voxel_cloud(points.len())?;
    /// let layout = VoxelLayout::new(1.0, [0.0; 3])?;
    /// filter.downsample(&device, points.len(), &layout, DownsampleMode::First, &mut out)?;
    /// assert_eq!(out.to_host()?.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(stream: &Arc<CudaStream>) -> Result<Self, VoxelError> {
        let kernels =
            CudaKernel::compile_many(stream.context(), KERNELS, &NAMES).map_err(cuda_err)?;
        let mut it = kernels.into_iter();
        let mut next = || {
            it.next()
                .ok_or_else(|| cuda_err("compile_many returned fewer kernels than requested"))
        };
        let k_keys = next()?;
        let k_sort_tile = next()?;
        let k_sort_step = next()?;
        let k_sort_merge = next()?;
        let k_tile_scan = next()?;
        let k_scan_tiles = next()?;
        let k_finalize = next()?;
        let k_reduce = next()?;
        let k_count = next()?;
        let k_scatter = next()?;
        let one = |s: &Arc<CudaStream>| s.alloc_zeros::<u32>(1).map_err(cuda_err);
        let filter = Self {
            words: one(stream)?,
            excl: one(stream)?,
            tile_sum: one(stream)?,
            tile_off: one(stream)?,
            run_start: one(stream)?,
            run_of: one(stream)?,
            voxels: one(stream)?,
            unique: one(stream)?,
            vcount: stream.alloc_zeros::<u8>(1).map_err(cuda_err)?,
            stream: stream.clone(),
            k_keys,
            k_sort_tile,
            k_sort_step,
            k_sort_merge,
            k_tile_scan,
            k_scan_tiles,
            k_finalize,
            k_reduce,
            k_count,
            k_scatter,
        };
        stream.synchronize().map_err(cuda_err)?;
        Ok(filter)
    }

    /// The stream this filter enqueues on.
    ///
    /// # Returns
    ///
    /// The stream given to [`new`](Self::new).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Allocates a reusable device downsampling output on this filter's
    /// stream.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum voxels, at least the point count of the
    ///   clouds it will receive.
    ///
    /// # Returns
    ///
    /// An empty [`CudaVoxelCloud`].
    ///
    /// # Errors
    ///
    /// [`VoxelError::Cuda`] on allocation failure.
    pub fn alloc_voxel_cloud(&self, capacity: usize) -> Result<CudaVoxelCloud, VoxelError> {
        let s = &self.stream;
        let n = capacity.max(1);
        let xyz_len = n
            .checked_mul(3)
            .ok_or_else(|| cuda_err("voxel capacity overflows usize"))?;
        Ok(CudaVoxelCloud {
            xyz: s.alloc_zeros::<f64>(xyz_len).map_err(cuda_err)?,
            index: s.alloc_zeros::<u32>(n).map_err(cuda_err)?,
            count: s.alloc_zeros::<u32>(1).map_err(cuda_err)?,
            err: s.alloc_zeros::<u32>(1).map_err(cuda_err)?,
            capacity,
            stream: s.clone(),
        })
    }

    /// Allocates a reusable device neighbour-count output on this filter's
    /// stream.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum points of the clouds it will receive.
    ///
    /// # Returns
    ///
    /// An empty [`CudaNeighbourCounts`].
    ///
    /// # Errors
    ///
    /// [`VoxelError::Cuda`] on allocation failure.
    pub fn alloc_neighbour_counts(
        &self,
        capacity: usize,
    ) -> Result<CudaNeighbourCounts, VoxelError> {
        let s = &self.stream;
        Ok(CudaNeighbourCounts {
            counts: s.alloc_zeros::<u8>(capacity.max(1)).map_err(cuda_err)?,
            err: s.alloc_zeros::<u32>(1).map_err(cuda_err)?,
            len: 0,
            capacity,
            stream: s.clone(),
        })
    }

    /// Validates a call's point buffer and output capacity.
    fn check_call(
        &self,
        points: &CudaSlice<f64>,
        n: usize,
        capacity: usize,
        out_stream: &Arc<CudaStream>,
    ) -> Result<(), VoxelError> {
        if n > MAX_CUDA_VOXEL_POINTS {
            return Err(VoxelError::TooManyPoints {
                len: n,
                max: MAX_CUDA_VOXEL_POINTS,
            });
        }
        let needed = 3 * n;
        if points.len() < needed {
            return Err(VoxelError::PointBufferTooSmall {
                len: points.len(),
                needed,
            });
        }
        if capacity < n {
            return Err(VoxelError::CapacityTooSmall {
                capacity,
                needed: n,
            });
        }
        if !Arc::ptr_eq(out_stream, &self.stream) {
            return Err(VoxelError::StreamMismatch);
        }
        Ok(())
    }

    /// Steps 1-3 for `n > 0` points: sort words, run starts, the voxel
    /// count in `self.voxels`, and with `for_counts` the run of every
    /// sorted word and the unique keys.
    fn sort_and_segment(
        &mut self,
        points: &CudaSlice<f64>,
        n: usize,
        layout: &LayoutParams,
        err: &mut CudaSlice<u32>,
        for_counts: bool,
    ) -> Result<(), VoxelError> {
        let st = self.stream.clone();
        let len = n.next_power_of_two().max(SORT_TILE);
        let tiles = n.div_ceil(SCAN_TILE);
        ensure_len(&st, &mut self.words, 4 * len)?;
        ensure_len(&st, &mut self.excl, n)?;
        ensure_len(&st, &mut self.tile_sum, tiles)?;
        ensure_len(&st, &mut self.tile_off, tiles)?;
        ensure_len(&st, &mut self.run_start, n)?;
        if for_counts {
            ensure_len(&st, &mut self.run_of, n)?;
            ensure_len(&st, &mut self.unique, 4 * n)?;
        }
        // `len <= MAX_CUDA_VOXEL_POINTS = 2³¹` and `n <= len`, so every
        // count below fits `u32`.
        let (n32, len32, tiles32) = (n as u32, len as u32, tiles as u32);
        let null = 0u64;

        st.memset_zeros(&mut *err).map_err(cuda_err)?;
        self.k_keys
            .launch_builder(&st)
            .arg(points)
            .arg(&n32)
            .arg(&len32)
            .arg(layout)
            .arg(&mut self.words)
            .arg(&mut *err)
            .launch_1d(len32)
            .map_err(cuda_err)?;

        let tile_cfg = LaunchConfig {
            grid_dim: ((len / SORT_TILE) as u32, 1, 1),
            block_dim: (SORT_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        self.k_sort_tile
            .launch_builder(&st)
            .arg(&mut self.words)
            .launch_cfg(tile_cfg)
            .map_err(cuda_err)?;
        let pairs = len32 / 2;
        let mut k = 2 * SORT_TILE;
        while k <= len {
            let k32 = k as u32;
            let mut j = k / 2;
            while j >= SORT_TILE {
                let j32 = j as u32;
                self.k_sort_step
                    .launch_builder(&st)
                    .arg(&mut self.words)
                    .arg(&j32)
                    .arg(&k32)
                    .arg(&pairs)
                    .launch_1d(pairs)
                    .map_err(cuda_err)?;
                j /= 2;
            }
            self.k_sort_merge
                .launch_builder(&st)
                .arg(&mut self.words)
                .arg(&k32)
                .launch_cfg(tile_cfg)
                .map_err(cuda_err)?;
            k *= 2;
        }

        self.k_tile_scan
            .launch_builder(&st)
            .arg(&self.words)
            .arg(&n32)
            .arg(&mut self.excl)
            .arg(&mut self.tile_sum)
            .launch_cfg(LaunchConfig {
                grid_dim: (tiles32, 1, 1),
                block_dim: (SCAN_TILE as u32, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(cuda_err)?;
        self.k_scan_tiles
            .launch_builder(&st)
            .arg(&self.tile_sum)
            .arg(&tiles32)
            .arg(&mut self.tile_off)
            .arg(&mut self.voxels)
            .launch_cfg(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (SCAN_TILE as u32, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(cuda_err)?;
        let b = self
            .k_finalize
            .launch_builder(&st)
            .arg(&self.words)
            .arg(&n32)
            .arg(&self.excl)
            .arg(&self.tile_off)
            .arg(&mut self.run_start);
        let b = if for_counts {
            b.arg(&mut self.run_of).arg(&mut self.unique)
        } else {
            b.arg(&null).arg(&null)
        };
        b.launch_1d(n32).map_err(cuda_err)?;
        Ok(())
    }

    /// Device twin of [`VoxelDownsampler::downsample_into`]: same points,
    /// same order, same source indices.
    ///
    /// # Arguments
    ///
    /// * `points` - Device `f64` buffer of at least `3 · n` interleaved
    ///   `x, y, z`.
    /// * `n` - Points to read.
    /// * `layout` - Voxel size and origin.
    /// * `mode` - Which point represents a voxel.
    /// * `out` - Output from [`alloc_voxel_cloud`](Self::alloc_voxel_cloud)
    ///   with at least `n` slots.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the work is enqueued; read `out` after syncing. An
    /// invalid point is reported by `out`'s readback methods.
    ///
    /// # Errors
    ///
    /// * [`VoxelError::TooManyPoints`] above [`MAX_CUDA_VOXEL_POINTS`].
    /// * [`VoxelError::PointBufferTooSmall`] if `points` holds fewer than
    ///   `3 · n` values; [`VoxelError::CapacityTooSmall`] if `out` holds
    ///   fewer than `n` voxels; [`VoxelError::StreamMismatch`] if `out` is
    ///   from another filter stream.
    /// * [`VoxelError::Cuda`] on an allocation or launch failure.
    pub fn downsample(
        &mut self,
        points: &CudaSlice<f64>,
        n: usize,
        layout: &VoxelLayout,
        mode: DownsampleMode,
        out: &mut CudaVoxelCloud,
    ) -> Result<(), VoxelError> {
        self.check_call(points, n, out.capacity, &out.stream)?;
        let st = self.stream.clone();
        if n == 0 {
            st.memset_zeros(&mut out.count).map_err(cuda_err)?;
            st.memset_zeros(&mut out.err).map_err(cuda_err)?;
            return Ok(());
        }
        let lay = LayoutParams::from(layout);
        self.sort_and_segment(points, n, &lay, &mut out.err, false)?;
        let mode_id = match mode {
            DownsampleMode::Centroid => MODE_CENTROID,
            DownsampleMode::First => MODE_FIRST,
            DownsampleMode::NearestToCentre => MODE_NEAREST,
        };
        let n32 = n as u32;
        self.k_reduce
            .launch_builder(&st)
            .arg(points)
            .arg(&self.words)
            .arg(&n32)
            .arg(&self.run_start)
            .arg(&self.voxels)
            .arg(&lay)
            .arg(&mode_id)
            .arg(&mut out.xyz)
            .arg(&mut out.index)
            .arg(&mut out.count)
            .launch_1d(n32)
            .map_err(cuda_err)?;
        Ok(())
    }

    /// Device twin of [`VoxelDownsampler::neighbour_counts_into`]: same
    /// counts, in input order.
    ///
    /// # Arguments
    ///
    /// * `points` - Device `f64` buffer of at least `3 · n` interleaved
    ///   `x, y, z`.
    /// * `n` - Points to read.
    /// * `layout` - Voxel size and origin.
    /// * `connectivity` - Which neighbours count.
    /// * `out` - Output from
    ///   [`alloc_neighbour_counts`](Self::alloc_neighbour_counts) with at
    ///   least `n` slots.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the work is enqueued; read `out` after syncing. An
    /// invalid point is reported by [`CudaNeighbourCounts::to_host`].
    ///
    /// # Errors
    ///
    /// As [`downsample`](Self::downsample).
    pub fn neighbour_counts(
        &mut self,
        points: &CudaSlice<f64>,
        n: usize,
        layout: &VoxelLayout,
        connectivity: VoxelConnectivity,
        out: &mut CudaNeighbourCounts,
    ) -> Result<(), VoxelError> {
        self.check_call(points, n, out.capacity, &out.stream)?;
        let st = self.stream.clone();
        out.len = n;
        if n == 0 {
            st.memset_zeros(&mut out.err).map_err(cuda_err)?;
            return Ok(());
        }
        let lay = LayoutParams::from(layout);
        self.sort_and_segment(points, n, &lay, &mut out.err, true)?;
        ensure_len(&st, &mut self.vcount, n)?;
        let n32 = n as u32;
        let max_l1 = connectivity.max_l1();
        self.k_count
            .launch_builder(&st)
            .arg(&self.unique)
            .arg(&self.voxels)
            .arg(&max_l1)
            .arg(&mut self.vcount)
            .launch_1d(n32)
            .map_err(cuda_err)?;
        self.k_scatter
            .launch_builder(&st)
            .arg(&self.words)
            .arg(&self.run_of)
            .arg(&n32)
            .arg(&self.vcount)
            .arg(&mut out.counts)
            .launch_1d(n32)
            .map_err(cuda_err)?;
        Ok(())
    }
}
