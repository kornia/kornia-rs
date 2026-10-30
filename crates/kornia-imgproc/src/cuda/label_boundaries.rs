//! CUDA kernels for label-boundary erosion — the device twin of
//! `morphology/label_boundaries/kernels.rs` (same seeds, same capped
//! separable distance, same select), so device and host results are
//! identical.
//!
//! Thresholds up to [`FUSED_MAX_THRESHOLD`] (radius <= 9) run as one
//! launch with no scratch buffer: `label_boundary_fused_*` loads a
//! `32 x 64` output tile plus a `t + 1` label halo into shared memory and
//! turns the seeds of the tile plus a `t` halo into one 64-bit mask per row
//! (warp ballots). The selected set is the seed set dilated by the diamond
//! (Manhattan) or square (Chebyshev) of radius `t`: per output row, the OR
//! over `|dy| <= t` of the seed row `dy` away shifted by up to `t - |dy|`
//! (or `t`) columns, a few word operations per 32 pixels. Every global read
//! is a coalesced tile row; the label map is read once (plus the halo) and
//! the output written once.
//!
//! Larger thresholds use two launches, no host round trip:
//!
//! 1. `label_boundary_v_*`: one thread per (column, row segment). The seed
//!    test runs on the fly from the label map (replicate-clamped
//!    neighbours, read through `__ldg`); a forward and a backward sweep give
//!    the vertical distance, capped at `t + 1`. A segment starts its sweeps
//!    `t` rows outside itself, which is exact because farther seeds are
//!    farther than `t`. Warps cover consecutive columns, so every row access
//!    is coalesced.
//! 2. `label_boundary_h_*`: one block per row. The row is scanned in tiles
//!    of `BLOCK * ITEMS` elements, forward then backward, each tile with a
//!    thread-local scan, a warp-shuffle scan of the per-thread aggregates
//!    and a cross-warp combine; the backward sweep writes `fill` or the
//!    label. Work per pixel is constant, independent of the radius.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, ValidAsZeroBits,
};
use kornia_tensor::CudaKernel;

use crate::connected_components::Connectivity;

super::define_cuda_error!(
    /// Error type for the CUDA label-boundary launchers.
    CudaLabelBoundaryError,
    "CUDA label-boundary error: {0}"
);

/// Threads per block of the vertical kernel (consecutive columns).
const V_BLOCK: u32 = 128;
/// Minimum rows per vertical segment; a segment re-reads `t` rows on each
/// side, so segments are at least `t` rows tall to bound that overhead.
const V_SEGMENT_MIN_ROWS: i32 = 16;
/// Threads per block of the horizontal kernel (one block per row).
const H_BLOCK: u32 = 128;
/// Consecutive elements per thread of the horizontal kernel; a tile is
/// `H_BLOCK * H_ITEMS = 640` elements.
const H_ITEMS: u32 = 5;
/// Largest distance threshold handled by the fused single-launch kernel;
/// its shared tiles are sized for this halo.
pub(crate) const FUSED_MAX_THRESHOLD: u16 = 8;
/// Output tile width of the fused kernel (one warp wide).
const F_TILE: u32 = 32;
/// Output tile height of the fused kernel; a tall tile amortizes the halo
/// and the per-block setup over more rows.
const F_TILE_H: u32 = 64;
/// Thread rows of the fused kernel's `F_TILE x F_ROWS` block.
const F_ROWS: u32 = 8;

/// Device-side spelling of a label dtype.
pub(crate) trait CudaLabelPixel: DeviceRepr + ValidAsZeroBits + Copy + 'static {
    /// Short tag used in kernel names and the kernel cache key.
    const TAG: &'static str;
    /// CUDA C element type.
    const CTYPE: &'static str;
    /// The value as a kernel `int` argument.
    fn as_i32(self) -> i32;
}

impl CudaLabelPixel for u8 {
    const TAG: &'static str = "u8";
    const CTYPE: &'static str = "unsigned char";
    fn as_i32(self) -> i32 {
        i32::from(self)
    }
}

impl CudaLabelPixel for u16 {
    const TAG: &'static str = "u16";
    const CTYPE: &'static str = "unsigned short";
    fn as_i32(self) -> i32 {
        i32::from(self)
    }
}

/// Kernel names and CUDA source for one (dtype, connectivity).
struct KernelSrc {
    vname: String,
    hname: String,
    fname: String,
    src: String,
}

fn kernel_src<T: CudaLabelPixel>(connectivity: Connectivity) -> KernelSrc {
    let tag = T::TAG;
    let ctype = T::CTYPE;
    let (conn, corners, threshold_step, tile_corners, reach) = match connectivity {
        Connectivity::Four => (4, "", "", "", "t - abs(dy)"),
        Connectivity::Eight => (
            8,
            "d |= (__ldg(ru + xl) != c) | (__ldg(ru + xr) != c) \
               | (__ldg(rd + xl) != c) | (__ldg(rd + xr) != c);",
            "v = v <= thr ? 0 : cap;",
            "d |= (lab[i][j] != c) | (lab[i][j + 2] != c) \
               | (lab[i + 2][j] != c) | (lab[i + 2][j + 2] != c);",
            "t",
        ),
    };
    let vname = format!("label_boundary_v_{tag}_c{conn}");
    let hname = format!("label_boundary_h_{tag}_c{conn}");
    let fname = format!("label_boundary_fused_{tag}_c{conn}");
    let src = format!(
        r#"
typedef {ctype} L;
#define BLOCK {H_BLOCK}
#define ITEMS {H_ITEMS}
#define TILE (BLOCK * ITEMS)
#define NWARPS (BLOCK / 32)
#define BIG 0x10000000

// Pixel (x, y) has a differently labelled neighbour; out-of-image
// neighbours are clamped, which never adds a difference.
__device__ __forceinline__ bool is_seed(const L* __restrict__ s, int x, int y, int w, int h) {{
    int xl = max(x - 1, 0);
    int xr = min(x + 1, w - 1);
    const L* rc = s + (size_t)y * w;
    const L* ru = s + (size_t)max(y - 1, 0) * w;
    const L* rd = s + (size_t)min(y + 1, h - 1) * w;
    L c = __ldg(rc + x);
    bool d = (__ldg(rc + xl) != c) | (__ldg(rc + xr) != c)
           | (__ldg(ru + x) != c) | (__ldg(rd + x) != c);
    {corners}
    return d;
}}

extern "C" __global__ void {vname}(
    const L* __restrict__ src, unsigned short* __restrict__ vdist,
    int w, int h, int thr, int seg
) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = blockIdx.y * seg;
    if (x >= w || y0 >= h) return;
    int y1 = min(y0 + seg, h);
    int cap = thr + 1;
    int run = cap;
    for (int y = max(y0 - thr, 0); y < y1; ++y) {{
        int f = is_seed(src, x, y, w, h) ? 0 : cap;
        run = min(f, run + 1);
        if (y >= y0) vdist[(size_t)y * w + x] = (unsigned short)run;
    }}
    run = cap;
    for (int y = min(y1 - 1 + thr, h - 1); y >= y0; --y) {{
        int f = is_seed(src, x, y, w, h) ? 0 : cap;
        run = min(f, run + 1);
        if (y < y1) {{
            size_t i = (size_t)y * w + x;
            vdist[i] = (unsigned short)min((int)vdist[i], run);
        }}
    }}
}}

// Min-plus (slope 1) inclusive scan of one tile. `a` holds this thread's
// ITEMS consecutive elements in scan order, `carry` the scanned value of
// the element just before the tile. Returns the scanned value of the
// tile's last element (block-uniform).
__device__ __forceinline__ int scan_tile(int* a, int carry, int* warp_tot, int* tile_last) {{
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    #pragma unroll
    for (int i = 1; i < ITEMS; ++i) a[i] = min(a[i], a[i - 1] + 1);
    int agg = a[ITEMS - 1];
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {{
        int o = __shfl_up_sync(0xffffffffu, agg, d);
        if (lane >= d) agg = min(agg, o + d * ITEMS);
    }}
    if (lane == 31) warp_tot[warp] = agg;
    __syncthreads();
    // Value at the element before this thread's first item.
    int ex = carry + t * ITEMS;
    for (int wp = 0; wp < warp; ++wp) {{
        ex = min(ex, warp_tot[wp] + (t - (wp + 1) * 32) * ITEMS);
    }}
    int prev = __shfl_up_sync(0xffffffffu, agg, 1);
    if (lane > 0) ex = min(ex, prev);
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) a[i] = min(a[i], ex + i + 1);
    if (t == BLOCK - 1) *tile_last = a[ITEMS - 1];
    __syncthreads();
    return *tile_last;
}}

extern "C" __global__ void {hname}(
    const L* __restrict__ src, unsigned short* vdist, L* __restrict__ dst,
    int w, int h, int thr, int fill
) {{
    __shared__ int warp_tot[NWARPS];
    __shared__ int tile_last;
    int y = blockIdx.x;
    if (y >= h) return;
    int cap = thr + 1;
    int t = threadIdx.x;
    // Written and re-read by this block: plain loads, not __ldg.
    unsigned short* row = vdist + (size_t)y * w;
    const L* lrow = src + (size_t)y * w;
    L* drow = dst + (size_t)y * w;
    int a[ITEMS];

    int carry = BIG;
    for (int base = 0; base < w; base += TILE) {{
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {{
            int x = base + t * ITEMS + i;
            int v = BIG;
            if (x < w) {{
                v = row[x];
                {threshold_step}
            }}
            a[i] = v;
        }}
        carry = scan_tile(a, carry, warp_tot, &tile_last);
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {{
            int x = base + t * ITEMS + i;
            if (x < w) row[x] = (unsigned short)min(a[i], cap);
        }}
    }}
    __syncthreads();

    carry = BIG;
    for (int end = w - 1; end >= 0; end -= TILE) {{
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {{
            int x = end - (t * ITEMS + i);
            a[i] = x >= 0 ? (int)row[x] : BIG;
        }}
        carry = scan_tile(a, carry, warp_tot, &tile_last);
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {{
            int x = end - (t * ITEMS + i);
            if (x >= 0) drow[x] = a[i] <= thr ? (L)fill : __ldg(lrow + x);
        }}
    }}
}}

#define FT {F_TILE}
#define FROWS {F_ROWS}
#define FMAXT {FUSED_MAX_THRESHOLD}
#define FTH {F_TILE_H}
#define FLW (FT + 2 * FMAXT + 2)
#define FLH (FTH + 2 * FMAXT + 2)
#define FSH (FTH + 2 * FMAXT)

// Seed test at tile seed coordinates (i, j); labels were loaded with
// clamped coordinates, so this matches `is_seed`.
__device__ __forceinline__ bool tile_seed(const L (*lab)[FLW], int i, int j) {{
    L c = lab[i + 1][j + 1];
    bool d = (lab[i + 1][j] != c) | (lab[i + 1][j + 2] != c)
           | (lab[i][j + 1] != c) | (lab[i + 2][j + 1] != c);
    {tile_corners}
    return d;
}}

// One FT x FTH output tile per block of FT x FROWS threads, thr <= FMAXT.
// Tile coordinates: output (r, c) is seed (r + t, c + t) and label
// (r + t + 1, c + t + 1). Each seed row (FT + 2t <= 64 columns) is a
// 64-bit mask built with two warp ballots; out-of-image seeds are 0.
// Output row r is the OR over |dy| <= t of seed row r + t + dy dilated
// horizontally by `reach` columns (t - |dy| for Manhattan, t for
// Chebyshev), i.e. the diamond / square of radius t.
extern "C" __global__ void {fname}(
    const L* __restrict__ src, L* __restrict__ dst,
    int w, int h, int thr, int fill
) {{
    __shared__ L lab[FLH][FLW];
    __shared__ unsigned long long smask[FSH];
    __shared__ unsigned long long omask[FTH];
    int t = thr;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x0 = blockIdx.x * FT;
    int y0 = blockIdx.y * FTH;
    int lw = FT + 2 * t + 2;
    int lh = FTH + 2 * t + 2;
    int sw = FT + 2 * t;
    int sh = FTH + 2 * t;

    // Interior tiles (the common case) skip the coordinate clamps: the
    // kernel is instruction bound, not memory bound.
    bool interior = x0 - t - 1 >= 0 && y0 - t - 1 >= 0
                 && x0 + FT + t + 1 <= w && y0 + FTH + t + 1 <= h;
    if (interior) {{
        const L* base = src + (size_t)(y0 - t - 1) * w + (x0 - t - 1);
        for (int i = ty; i < lh; i += FROWS) {{
            const L* row = base + (size_t)i * w;
            lab[i][tx] = __ldg(row + tx);
            if (tx < lw - FT) lab[i][tx + FT] = __ldg(row + tx + FT);
        }}
    }} else {{
        for (int i = ty; i < lh; i += FROWS) {{
            int gy = min(max(y0 - t - 1 + i, 0), h - 1);
            const L* row = src + (size_t)gy * w;
            for (int j = tx; j < lw; j += FT) {{
                int gx = min(max(x0 - t - 1 + j, 0), w - 1);
                lab[i][j] = __ldg(row + gx);
            }}
        }}
    }}
    for (int r = ty * FT + tx; r < FTH; r += FT * FROWS) omask[r] = 0ull;
    __syncthreads();

    // Warp-uniform loop: every lane reaches both ballots.
    for (int i = ty; i < sh; i += FROWS) {{
        int gy = y0 - t + i;
        bool row_in = gy >= 0 && gy < h;
        int gx = x0 - t + tx;
        bool lo = row_in && gx >= 0 && gx < w && tile_seed(lab, i, tx);
        unsigned int blo = __ballot_sync(0xffffffffu, lo);
        unsigned int bhi = 0u;
        if (t > 0) {{
            // Block-uniform: only a halo needs the high word.
            int j2 = tx + FT;
            bool hi = row_in && j2 < sw && gx + FT < w && tile_seed(lab, i, j2);
            bhi = __ballot_sync(0xffffffffu, hi);
        }}
        if (tx == 0) smask[i] = (unsigned long long)blo | ((unsigned long long)bhi << 32);
    }}
    __syncthreads();

    for (int r = tx; r < FTH; r += FT) {{
        unsigned long long acc = 0ull;
        for (int dy = -t + ty; dy <= t; dy += FROWS) {{
            unsigned long long m = smask[r + t + dy];
            int reach = {reach};
            unsigned long long d = m;
            for (int s = 1; s <= reach; ++s) d |= (m << s) | (m >> s);
            acc |= d;
        }}
        if (acc) atomicOr(&omask[r], acc);
    }}
    __syncthreads();

    int gx = x0 + tx;
    if (gx >= w) return;
    #pragma unroll
    for (int r = ty; r < FTH; r += FROWS) {{
        int gy = y0 + r;
        if (gy >= h) break;
        bool hit = ((unsigned int)(omask[r] >> t) >> tx) & 1u;
        dst[(size_t)gy * w + gx] = hit ? (L)fill : lab[r + t + 1][tx + t + 1];
    }}
}}
"#
    );
    KernelSrc {
        vname,
        hname,
        fname,
        src,
    }
}

/// The vertical, horizontal and fused kernels of one (dtype, connectivity).
#[derive(Clone)]
struct Kernels {
    vertical: Arc<CudaKernel>,
    horizontal: Arc<CudaKernel>,
    fused: Arc<CudaKernel>,
}
/// Keyed by device ordinal too: a module is loaded into one context and
/// cannot be launched on another device's stream.
type KernelMap = Mutex<HashMap<(usize, &'static str, bool), Kernels>>;
static KERNELS: OnceLock<KernelMap> = OnceLock::new();

fn get_kernels<T: CudaLabelPixel>(
    ctx: &Arc<CudaContext>,
    connectivity: Connectivity,
) -> Result<Kernels, CudaLabelBoundaryError> {
    let map = KERNELS.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (ctx.ordinal(), T::TAG, connectivity == Connectivity::Eight);
    let lock_err = |_| CudaLabelBoundaryError::Cuda("kernel cache lock poisoned".into());
    if let Some(k) = map.lock().map_err(lock_err)?.get(&key).cloned() {
        return Ok(k);
    }
    // Build outside the lock (scoped-guard idiom, like the sibling caches).
    let ks = kernel_src::<T>(connectivity);
    let names = [ks.vname.as_str(), ks.hname.as_str(), ks.fname.as_str()];
    let compiled = CudaKernel::compile_many(ctx, &ks.src, &names)
        .map_err(|e| CudaLabelBoundaryError::Cuda(e.to_string()))?;
    let mut compiled = compiled.into_iter();
    let (Some(kv), Some(kh), Some(kf)) = (compiled.next(), compiled.next(), compiled.next()) else {
        return Err(CudaLabelBoundaryError::Cuda(
            "label-boundary kernels missing from the compiled module".into(),
        ));
    };
    // The vertical kernel uses no shared memory; give the space to L1 for
    // the `__ldg` label reads.
    let _ = kv.prefer_l1_cache();
    let kernels = Kernels {
        vertical: Arc::new(kv),
        horizontal: Arc::new(kh),
        fused: Arc::new(kf),
    };
    Ok(map
        .lock()
        .map_err(lock_err)?
        .entry(key)
        .or_insert(kernels)
        .clone())
}

/// Checked `i32` dimensions; rejects zero sizes.
fn dims_i32(width: usize, height: usize) -> Result<(i32, i32), CudaLabelBoundaryError> {
    if width == 0 || height == 0 {
        return Err(CudaLabelBoundaryError::Cuda(
            "image dimensions must be non-zero".into(),
        ));
    }
    let w = i32::try_from(width)
        .map_err(|_| CudaLabelBoundaryError::Cuda("width exceeds i32".into()))?;
    let h = i32::try_from(height)
        .map_err(|_| CudaLabelBoundaryError::Cuda("height exceeds i32".into()))?;
    Ok((w, h))
}

/// Label-boundary erosion on device; same contract as
/// [`crate::morphology::erode_label_boundaries`]. `threshold` is the
/// distance threshold `radius - 1` from
/// `morphology::label_boundaries::distance_threshold`; `None` copies `src`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_erode_label_boundaries<T: CudaLabelPixel>(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<T>,
    dst: &mut CudaSlice<T>,
    width: usize,
    height: usize,
    threshold: Option<u16>,
    connectivity: Connectivity,
    fill: T,
) -> Result<(), CudaLabelBoundaryError> {
    let n = width * height;
    CudaLabelBoundaryError::check_slice("src", src.len(), n)?;
    CudaLabelBoundaryError::check_slice("dst", dst.len(), n)?;
    let Some(threshold) = threshold else {
        if n > 0 {
            stream.memcpy_dtod(&src.slice(..n), &mut dst.slice_mut(..n))?;
        }
        return Ok(());
    };
    let (w, h) = dims_i32(width, height)?;
    let thr = i32::from(threshold);
    let fill = fill.as_i32();
    let kernels = get_kernels::<T>(ctx, connectivity)?;

    if threshold <= FUSED_MAX_THRESHOLD {
        let cfg = LaunchConfig {
            block_dim: (F_TILE, F_ROWS, 1),
            grid_dim: (
                (w as u32).div_ceil(F_TILE),
                (h as u32).div_ceil(F_TILE_H),
                1,
            ),
            shared_mem_bytes: 0,
        };
        return kernels
            .fused
            .launch_builder(stream)
            .arg(src)
            .arg(dst)
            .arg(&w)
            .arg(&h)
            .arg(&thr)
            .arg(&fill)
            .launch_cfg(cfg)
            .map_err(|e| CudaLabelBoundaryError::Cuda(e.to_string()));
    }

    let seg = V_SEGMENT_MIN_ROWS.max(thr);
    let (kv, kh) = (&kernels.vertical, &kernels.horizontal);

    // SAFETY: every element of `vdist` is written by the vertical kernel
    // (each (column, segment) thread writes all rows of its segment in the
    // forward sweep) before the horizontal kernel, ordered after it on the
    // same stream, reads it.
    let mut vdist = unsafe { stream.alloc::<u16>(n) }?;

    let cfg_v = LaunchConfig {
        block_dim: (V_BLOCK, 1, 1),
        grid_dim: (
            (w as u32).div_ceil(V_BLOCK),
            (h as u32).div_ceil(seg as u32),
            1,
        ),
        shared_mem_bytes: 0,
    };
    kv.launch_builder(stream)
        .arg(src)
        .arg(&mut vdist)
        .arg(&w)
        .arg(&h)
        .arg(&thr)
        .arg(&seg)
        .launch_cfg(cfg_v)
        .map_err(|e| CudaLabelBoundaryError::Cuda(e.to_string()))?;

    let cfg_h = LaunchConfig {
        block_dim: (H_BLOCK, 1, 1),
        grid_dim: (h as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    kh.launch_builder(stream)
        .arg(src)
        .arg(&mut vdist)
        .arg(dst)
        .arg(&w)
        .arg(&h)
        .arg(&thr)
        .arg(&fill)
        .launch_cfg(cfg_h)
        .map_err(|e| CudaLabelBoundaryError::Cuda(e.to_string()))
}
