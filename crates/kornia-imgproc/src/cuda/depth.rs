//! CUDA kernels for the valid-aware depth median, the flying-pixel mask and
//! flagged-pixel removal — twins of `depth/kernels.rs`.
//!
//! One launch per op. A block owns a `TILE_W × TILE_H` output tile: its
//! threads first sort the `ksize`-sample columns of the tile plus a
//! `ksize / 2` apron into shared memory (each column once, shared by the
//! `ksize` windows that contain it), synchronize, then run the rank-selection
//! network on `ksize` adjacent sorted columns. A warp whose windows are all
//! fully valid takes the plain-median network instead (a warp-uniform
//! branch, so no divergence). The networks are code-generated from the
//! const arrays the CPU uses (`COL3` / `COL5`, `RANK9` / `RANK25`,
//! `MED25`), keys and the flying-pixel test are the same expressions, and
//! every selection is exact, so device and host results are bit-identical.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use kornia_tensor::CudaKernel;

use super::try_compile_with_l1;
use crate::depth::kernels::{COL3, COL5, FLYING_KSIZE, MASK_FLAGGED, MED25, RANK25, RANK9};

/// Output tile width; one warp per tile row (the warp-uniform fast path
/// relies on it).
const TILE_W: u32 = 32;

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
    /// `invalid_value`, `abs_diff_f32`, `CE`.
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
__device__ __forceinline__ bool is_valid(T v) { return !isnan(v); }
__device__ __forceinline__ T to_key(T v) { return isnan(v) ? __int_as_float(0x7f800000) : v + 0.0f; }
__device__ __forceinline__ T from_key(T k) { return k; }
__device__ __forceinline__ T invalid_value() { return __int_as_float(0x7fc00000); }
__device__ __forceinline__ float abs_diff_f32(T a, T b) { return fabsf(a - b); }
#define CE(a, b) { T lo = fminf(a, b), hi = fmaxf(a, b); a = lo; b = hi; }
"#;
}

/// Straight-line compare-exchanges on array `v`.
fn net_src(net: &[(usize, usize)], v: &str) -> String {
    net.iter()
        .map(|&(a, b)| format!("    CE({v}[{a}], {v}[{b}]);\n"))
        .collect()
}

/// `median_tile(src, w, h, c, &n)`: the shared-memory two-pass median of
/// the calling thread's pixel `(blockIdx.x * TILE_W + threadIdx.x,
/// blockIdx.y * TILE_H + threadIdx.y)`, channel `c` of `channels`; returns
/// the rank-`n / 2` key, `n` = valid tap count. Every thread of the block
/// must call it (it synchronizes), including those outside the image.
fn median_fn_src(ksize: usize, channels: usize) -> String {
    let taps = ksize * ksize;
    let r = ksize / 2;
    let half = taps / 2;
    let span = TILE_W as usize + 2 * r;
    let col_net = net_src(if ksize == 3 { COL3 } else { COL5 }, "v");
    let rank_net = net_src(if ksize == 3 { RANK9 } else { RANK25 }, "v");
    let median = if ksize == 3 {
        // med3(max(lows), med3(mids), min(highs)) on sorted columns.
        "        m = med3(max(max(v[0], v[3]), v[6]), med3(v[1], v[4], v[7]), min(min(v[2], v[5]), v[8]));\n".to_string()
    } else {
        format!("{}        m = v[{half}];\n", net_src(MED25, "v"))
    };
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
    for (int cx = threadIdx.x; cx < {span}; cx += {tile_w}) {{
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
    #pragma unroll
    for (int dc = 0; dc < {ksize}; ++dc) {{
        n += sn[ty][threadIdx.x + dc];
        #pragma unroll
        for (int i = 0; i < {ksize}; ++i) v[dc * {ksize} + i] = sk[ty][i][threadIdx.x + dc];
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
        tile_w = TILE_W,
        tile_h = TILE_H,
    )
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum Kind {
    Median { ksize: usize, channels: usize },
    FlyingMask,
    RemoveFlagged,
}

fn kernel_src<T: CudaDepthPixel>(kind: Kind) -> (String, String) {
    let tag = T::TAG;
    let prelude = T::PRELUDE;
    match kind {
        Kind::Median { ksize, channels } => {
            let name = format!("median_depth_{tag}_{ksize}x{ksize}_c{channels}");
            let helper = median_fn_src(ksize, channels);
            let src = format!(
                r#"{prelude}{helper}
extern "C" __global__ void {name}(
    const T* __restrict__ src, T* __restrict__ dst, int w, int h
) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // No early return: every thread takes part in the tile's barriers.
    for (int c = 0; c < {channels}; ++c) {{
        int n;
        T m = median_tile(src, w, h, c, &n);
        if (x < w && y < h) dst[(y * w + x) * {channels} + c] = n == 0 ? invalid_value() : m;
    }}
}}
"#
            );
            (name, src)
        }
        Kind::FlyingMask => {
            let name = format!("flying_pixel_mask_{tag}");
            let helper = median_fn_src(FLYING_KSIZE, 1);
            let src = format!(
                r#"{prelude}{helper}
extern "C" __global__ void {name}(
    const T* __restrict__ src, unsigned char* __restrict__ mask, int w, int h,
    float abs_threshold, float rel_threshold, int min_valid
) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // No early return: every thread takes part in the tile's barriers.
    int n;
    T m = median_tile(src, w, h, 0, &n);
    if (x >= w || y >= h) return;
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
    let kernel = Arc::new(try_compile_with_l1(ctx, &src, &name).map_err(CudaDepthError::Cuda)?);
    Ok(map
        .lock()
        .map_err(lock_err)?
        .entry(key)
        .or_insert(kernel)
        .clone())
}

/// Checked `i32` dimensions; rejects zero sizes.
fn dims_i32(width: usize, height: usize) -> Result<(i32, i32), CudaDepthError> {
    if width == 0 || height == 0 {
        return Err(CudaDepthError::Cuda(
            "image dimensions must be non-zero".into(),
        ));
    }
    let w = i32::try_from(width).map_err(|_| CudaDepthError::Cuda("width exceeds i32".into()))?;
    let h = i32::try_from(height).map_err(|_| CudaDepthError::Cuda("height exceeds i32".into()))?;
    Ok((w, h))
}

/// Valid-aware median blur on device; same contract as
/// [`crate::depth::median_blur_depth`]. `ksize` must be 3 or 5.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_median_blur_depth<T: CudaDepthPixel>(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<T>,
    dst: &mut CudaSlice<T>,
    width: usize,
    height: usize,
    channels: usize,
    ksize: usize,
) -> Result<(), CudaDepthError> {
    if ksize != 3 && ksize != 5 {
        return Err(CudaDepthError::Cuda("ksize must be 3 or 5".into()));
    }
    if !(1..=4).contains(&channels) {
        return Err(CudaDepthError::Cuda("channels must be in 1..=4".into()));
    }
    let (w, h) = dims_i32(width, height)?;
    CudaDepthError::check_slice("src", src.len(), width * height * channels)?;
    CudaDepthError::check_slice("dst", dst.len(), width * height * channels)?;
    let kernel = get_kernel::<T>(ctx, Kind::Median { ksize, channels })?;
    let cfg = super::make_config(w as u32, h as u32, Some((TILE_W, TILE_H)));
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&w)
        .arg(&h)
        .launch_cfg(cfg)
        .map_err(|e| CudaDepthError::Cuda(e.to_string()))
}

/// Flying-pixel mask on device; same contract as
/// [`crate::depth::flying_pixel_mask`].
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
    let (w, h) = dims_i32(width, height)?;
    CudaDepthError::check_slice("src", src.len(), width * height)?;
    CudaDepthError::check_slice("mask", mask.len(), width * height)?;
    // Any value above the tap count flags every valid pixel, as on the CPU.
    let min_valid = i32::try_from(min_valid).unwrap_or(i32::MAX);
    let kernel = get_kernel::<T>(ctx, Kind::FlyingMask)?;
    let cfg = super::make_config(w as u32, h as u32, Some((TILE_W, TILE_H)));
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
