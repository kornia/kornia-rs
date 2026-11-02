//! CUDA kernels for the flying-pixel mask and flagged-pixel removal — twins
//! of `depth/kernels.rs`.
//!
//! One launch per op. The mask kernel's block owns a `tile_w(ksize) ×
//! TILE_H` output tile: its threads first sort the `ksize`-sample columns of
//! the tile plus a `ksize / 2` apron into shared memory (each column once,
//! shared by the `ksize` windows that contain it), synchronize, then run
//! the rank-selection network on `ksize` adjacent sorted columns. A warp
//! whose windows are all fully valid takes the plain median of 9 instead (a
//! warp-uniform branch, so no divergence). The networks are code-generated
//! from the const arrays the CPU uses (`COL3`, `RANK9`), keys and the
//! flying-pixel test are the same expressions, and every selection is
//! exact, so device and host results are bit-identical.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, ValidAsZeroBits,
};
use kornia_tensor::CudaKernel;

use super::try_compile_with_l1;
use crate::depth::kernels::{COL3, FLYING_KSIZE, MASK_FLAGGED, RANK9};

/// Threads per tile row: one warp (the warp-uniform fast path relies on
/// it). Each thread sorts exactly one column of the tile plus apron, so a
/// tile is `WARP - 2 * (ksize / 2)` outputs wide.
const WARP: u32 = 32;

/// Output tile height.
const TILE_H: u32 = 8;

super::define_cuda_error!(
    /// Error type for the CUDA depth launchers.
    CudaDepthError,
    "CUDA depth error: {0}"
);

/// Device-side spelling of a depth dtype's semantics (mirror of
/// `depth::kernels::DepthRepr`).
pub(crate) trait CudaDepthPixel: DeviceRepr + ValidAsZeroBits + Copy + 'static {
    /// Short tag used in kernel names and the kernel cache key.
    const TAG: &'static str;
    /// Device helper functions: `T`, `is_valid`, `to_key`, `from_key`,
    /// `invalid_value`, `abs_diff_f32`, `CE`. `is_valid` and `to_key` are
    /// the host rule ([`crate::depth::is_valid_depth_u16`] /
    /// [`crate::depth::is_valid_depth_f32`]) and `DepthRepr::key`,
    /// expression for expression.
    const PRELUDE: &'static str;
}

impl CudaDepthPixel for u16 {
    const TAG: &'static str = "u16";
    const PRELUDE: &'static str = r#"
typedef unsigned short T;
__device__ __forceinline__ bool is_valid(T v) { return v != 0; }
__device__ __forceinline__ T to_key(T v) { return (T)(v - 1); }
__device__ __forceinline__ T from_key(T k) { return (T)(k + 1); }
__device__ __forceinline__ T invalid_value() { return 0; }
__device__ __forceinline__ float abs_diff_f32(T a, T b) { return (float)abs((int)a - (int)b); }
#define CE(a, b) { T lo = min(a, b), hi = max(a, b); a = lo; b = hi; }
"#;
}

impl CudaDepthPixel for f32 {
    const TAG: &'static str = "f32";
    const PRELUDE: &'static str = r#"
typedef float T;
__device__ __forceinline__ bool is_valid(T v) { return isfinite(v) && v > 0.0f; }
__device__ __forceinline__ T to_key(T v) { return is_valid(v) ? v : __int_as_float(0x7f800000); }
__device__ __forceinline__ T from_key(T k) { return k; }
__device__ __forceinline__ T invalid_value() { return __int_as_float(0x7fc00000); }
__device__ __forceinline__ float abs_diff_f32(T a, T b) { return fabsf(a - b); }
#define CE(a, b) { T lo = fminf(a, b), hi = fmaxf(a, b); a = lo; b = hi; }
"#;
}

/// Output tile width for a `ksize` window: the warp minus the apron.
fn tile_w(ksize: usize) -> u32 {
    WARP - 2 * (ksize as u32 / 2)
}

/// Straight-line compare-exchanges on array `v`.
fn net_src(net: &[(usize, usize)], v: &str) -> String {
    net.iter()
        .map(|&(a, b)| format!("    CE({v}[{a}], {v}[{b}]);\n"))
        .collect()
}

/// `median_tile(src, w, h, c, &n)`: the shared-memory two-pass median of
/// the calling thread's pixel `(blockIdx.x * tile_w + threadIdx.x,
/// blockIdx.y * TILE_H + threadIdx.y)`, channel `c` of `channels`, for the
/// 3×3 window (`ksize` is [`FLYING_KSIZE`]); returns
/// the rank-`n / 2` key, `n` = valid tap count. Every thread of the block
/// must call it (it synchronizes), including those outside the image and
/// the apron threads `threadIdx.x >= tile_w` (whose result is discarded).
fn median_fn_src(ksize: usize, channels: usize) -> String {
    let taps = ksize * ksize;
    let r = ksize / 2;
    let half = taps / 2;
    let span = WARP as usize;
    let tile_w = tile_w(ksize);
    let last = tile_w - 1;
    let col_net = net_src(COL3, "v");
    let rank_net = net_src(RANK9, "v");
    // med3(max(lows), med3(mids), min(highs)) on sorted columns.
    let median = "        m = med3(max(max(v[0], v[3]), v[6]), med3(v[1], v[4], v[7]), min(min(v[2], v[5]), v[8]));\n";
    format!(
        r#"
__device__ __forceinline__ T med3(T a, T b, T c) {{
    return max(min(a, b), min(max(a, b), c));
}}

__device__ __forceinline__ T median_tile(
    const T* __restrict__ src, int w, int h, int c, int* count
) {{
    __shared__ T sk[{tile_h}][{ksize}][{span}];
    __shared__ unsigned char sn[{tile_h}][{span}];
    int ty = threadIdx.y;
    int x0 = blockIdx.x * {tile_w};
    int y = min((int)(blockIdx.y * {tile_h}) + ty, h - 1);
    {{
        int cx = threadIdx.x;
        int sx = min(max(x0 + cx - {r}, 0), w - 1);
        T v[{ksize}];
        int n = 0;
        #pragma unroll
        for (int i = 0; i < {ksize}; ++i) {{
            int sy = min(max(y + i - {r}, 0), h - 1);
            T s = __ldg(&src[(sy * w + sx) * {channels} + c]);
            n += is_valid(s) ? 1 : 0;
            v[i] = to_key(s);
        }}
{col_net}
        #pragma unroll
        for (int i = 0; i < {ksize}; ++i) sk[ty][i][cx] = v[i];
        sn[ty][cx] = (unsigned char)n;
    }}
    __syncthreads();
    T v[{taps}];
    int n = 0;
    // Apron threads read the last output's window (discarded, in range).
    int tx = min((int)threadIdx.x, {last});
    #pragma unroll
    for (int dc = 0; dc < {ksize}; ++dc) {{
        n += sn[ty][tx + dc];
        #pragma unroll
        for (int i = 0; i < {ksize}; ++i) v[dc * {ksize} + i] = sk[ty][i][tx + dc];
    }}
    __syncthreads();
    T m;
    if (__all_sync(0xffffffffu, n == {taps})) {{
{median}    }} else {{
{rank_net}
        int k = n / 2;
        m = v[0];
        #pragma unroll
        for (int i = 1; i <= {half}; ++i) {{
            if (i == k) m = v[i];
        }}
    }}
    *count = n;
    return from_key(m);
}}
"#,
        tile_h = TILE_H,
    )
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum Kind {
    FlyingMask,
    RemoveFlagged,
}

fn kernel_src<T: CudaDepthPixel>(kind: Kind) -> (String, String) {
    let tag = T::TAG;
    let prelude = T::PRELUDE;
    match kind {
        Kind::FlyingMask => {
            let name = format!("flying_pixel_mask_{tag}");
            let helper = median_fn_src(FLYING_KSIZE, 1);
            let tile_w = tile_w(FLYING_KSIZE);
            let src = format!(
                r#"{prelude}{helper}
extern "C" __global__ void {name}(
    const T* __restrict__ src, unsigned char* __restrict__ mask, int w, int h,
    float abs_threshold, float rel_threshold, int min_valid
) {{
    int x = blockIdx.x * {tile_w} + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // No early return: every thread takes part in the tile's barriers.
    int n;
    T m = median_tile(src, w, h, 0, &n);
    if (threadIdx.x >= {tile_w} || x >= w || y >= h) return;
    T d = src[y * w + x];
    unsigned char out = 0;
    if (is_valid(d)) {{
        if (n < min_valid) {{
            out = {MASK_FLAGGED};
        }} else {{
            float threshold = fmaxf(abs_threshold, rel_threshold * (float)d);
            if (abs_diff_f32(d, m) > threshold) out = {MASK_FLAGGED};
        }}
    }}
    mask[y * w + x] = out;
}}
"#
            );
            (name, src)
        }
        Kind::RemoveFlagged => {
            let name = format!("remove_flagged_{tag}");
            let src = format!(
                r#"{prelude}
extern "C" __global__ void {name}(
    const T* __restrict__ src, const unsigned char* __restrict__ mask,
    T* __restrict__ dst, int w, int h
) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    dst[i] = mask[i] != 0 ? invalid_value() : src[i];
}}
"#
            );
            (name, src)
        }
    }
}

type KernelMap = Mutex<HashMap<(&'static str, Kind), Arc<CudaKernel>>>;
static KERNELS: OnceLock<KernelMap> = OnceLock::new();

fn get_kernel<T: CudaDepthPixel>(
    ctx: &Arc<CudaContext>,
    kind: Kind,
) -> Result<Arc<CudaKernel>, CudaDepthError> {
    let map = KERNELS.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (T::TAG, kind);
    let lock_err = |_| CudaDepthError::Cuda("kernel cache lock poisoned".into());
    if let Some(k) = map.lock().map_err(lock_err)?.get(&key).cloned() {
        return Ok(k);
    }
    // Build outside the lock (scoped-guard idiom, like the sibling caches).
    let (name, src) = kernel_src::<T>(kind);
    // The tile kernels use shared memory: leave the L1 / shared carveout to
    // the driver. The elementwise kernel prefers L1.
    let kernel = match kind {
        Kind::RemoveFlagged => try_compile_with_l1(ctx, &src, &name),
        Kind::FlyingMask => CudaKernel::compile(ctx, &src, &name)
            .map_err(|e| format!("failed to compile {name}: {e}")),
    }
    .map_err(CudaDepthError::Cuda)?;
    let kernel = Arc::new(kernel);
    Ok(map
        .lock()
        .map_err(lock_err)?
        .entry(key)
        .or_insert(kernel)
        .clone())
}

/// Launch geometry of the tile kernels: `WARP × TILE_H` threads per
/// `tile_w(ksize) × TILE_H` output tile.
fn tile_config(width: u32, height: u32, ksize: usize) -> LaunchConfig {
    LaunchConfig {
        block_dim: (WARP, TILE_H, 1),
        grid_dim: (width.div_ceil(tile_w(ksize)), height.div_ceil(TILE_H), 1),
        shared_mem_bytes: 0,
    }
}

/// True for an image with no pixel: every launcher returns `Ok(())`
/// without launching, as the CPU path does.
fn is_empty(width: usize, height: usize) -> bool {
    width == 0 || height == 0
}

/// Checked `i32` dimensions of a non-empty image.
fn dims_i32(width: usize, height: usize) -> Result<(i32, i32), CudaDepthError> {
    let w = i32::try_from(width).map_err(|_| CudaDepthError::Cuda("width exceeds i32".into()))?;
    let h = i32::try_from(height).map_err(|_| CudaDepthError::Cuda("height exceeds i32".into()))?;
    Ok((w, h))
}

/// Flying-pixel mask on device; same contract as
/// [`crate::depth::flying_pixel_mask`], which validates the thresholds
/// before dispatching here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_flying_pixel_mask<T: CudaDepthPixel>(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<T>,
    mask: &mut CudaSlice<u8>,
    width: usize,
    height: usize,
    abs_threshold: f32,
    rel_threshold: f32,
    min_valid: usize,
) -> Result<(), CudaDepthError> {
    if is_empty(width, height) {
        return Ok(());
    }
    let (w, h) = dims_i32(width, height)?;
    CudaDepthError::check_slice("src", src.len(), width * height)?;
    CudaDepthError::check_slice("mask", mask.len(), width * height)?;
    // Any value above the tap count flags every valid pixel, as on the CPU.
    let min_valid = i32::try_from(min_valid).unwrap_or(i32::MAX);
    let kernel = get_kernel::<T>(ctx, Kind::FlyingMask)?;
    let cfg = tile_config(w as u32, h as u32, FLYING_KSIZE);
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(mask)
        .arg(&w)
        .arg(&h)
        .arg(&abs_threshold)
        .arg(&rel_threshold)
        .arg(&min_valid)
        .launch_cfg(cfg)
        .map_err(|e| CudaDepthError::Cuda(e.to_string()))
}

/// Flagged-pixel removal on device; same contract as
/// [`crate::depth::invalidate_masked_depth`].
pub(crate) fn launch_remove_flagged<T: CudaDepthPixel>(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<T>,
    mask: &CudaSlice<u8>,
    dst: &mut CudaSlice<T>,
    width: usize,
    height: usize,
) -> Result<(), CudaDepthError> {
    if is_empty(width, height) {
        return Ok(());
    }
    let (w, h) = dims_i32(width, height)?;
    CudaDepthError::check_slice("src", src.len(), width * height)?;
    CudaDepthError::check_slice("mask", mask.len(), width * height)?;
    CudaDepthError::check_slice("dst", dst.len(), width * height)?;
    let kernel = get_kernel::<T>(ctx, Kind::RemoveFlagged)?;
    let cfg = super::make_config(w as u32, h as u32, None);
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(mask)
        .arg(dst)
        .arg(&w)
        .arg(&h)
        .launch_cfg(cfg)
        .map_err(|e| CudaDepthError::Cuda(e.to_string()))
}
