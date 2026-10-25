//! CUDA kernels for the valid-aware depth median, the flying-pixel mask and
//! flagged-pixel removal — textual twins of `depth/kernels.rs`.
//!
//! The sorting networks are code-generated from the same const arrays the
//! CPU uses (`SORT9` / `SORT25`), the window gather uses the same replicate
//! border and tap order, and the flying-pixel test is the same expression,
//! so device and host results are bit-identical by construction.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use kornia_tensor::CudaKernel;

use super::try_compile_with_l1;
use crate::depth::kernels::{FLYING_KSIZE, MASK_FLAGGED, SORT25, SORT9};

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
__device__ __forceinline__ T to_key(T v) { return isnan(v) ? __int_as_float(0x7f800000) : v; }
__device__ __forceinline__ T from_key(T k) { return k; }
__device__ __forceinline__ T invalid_value() { return __int_as_float(0x7fc00000); }
__device__ __forceinline__ float abs_diff_f32(T a, T b) { return fabsf(a - b); }
#define CE(a, b) { T lo = fminf(a, b), hi = fmaxf(a, b); a = lo; b = hi; }
"#;
}

/// `median_valid(src, x, y, w, h, c, &n)`: valid-sample median key of the
/// `ksize × ksize` replicate-border window, `n` = valid tap count.
fn median_fn_src(ksize: usize, channels: usize) -> String {
    let taps = ksize * ksize;
    let r = ksize / 2;
    let net: String = if ksize == 3 { &SORT9[..] } else { &SORT25[..] }
        .iter()
        .map(|&(a, b)| format!("    CE(v[{a}], v[{b}]);\n"))
        .collect();
    format!(
        r#"
__device__ __forceinline__ T median_valid(
    const T* __restrict__ src, int x, int y, int w, int h, int c, int* count
) {{
    T v[{taps}];
    int n = 0;
    int t = 0;
    #pragma unroll
    for (int dy = -{r}; dy <= {r}; ++dy) {{
        int sy = min(max(y + dy, 0), h - 1);
        #pragma unroll
        for (int dx = -{r}; dx <= {r}; ++dx) {{
            int sx = min(max(x + dx, 0), w - 1);
            T s = __ldg(&src[(sy * w + sx) * {channels} + c]);
            n += is_valid(s) ? 1 : 0;
            v[t++] = to_key(s);
        }}
    }}
{net}
    int k = n / 2;
    T m = v[0];
    #pragma unroll
    for (int i = 1; i < {taps}; ++i) {{
        if (i == k) m = v[i];
    }}
    *count = n;
    return from_key(m);
}}
"#
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
    if (x >= w || y >= h) return;
    #pragma unroll
    for (int c = 0; c < {channels}; ++c) {{
        int n;
        T m = median_valid(src, x, y, w, h, c, &n);
        dst[(y * w + x) * {channels} + c] = n == 0 ? invalid_value() : m;
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
    if (x >= w || y >= h) return;
    T d = src[y * w + x];
    unsigned char out = 0;
    if (is_valid(d)) {{
        int n;
        T m = median_valid(src, x, y, w, h, 0, &n);
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
    let cfg = super::make_config(w as u32, h as u32, None);
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
    let cfg = super::make_config(w as u32, h as u32, None);
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
