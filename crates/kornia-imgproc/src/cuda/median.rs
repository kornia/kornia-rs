//! CUDA median-blur kernels — textual twins of `filter/median.rs`.
//!
//! The sorting networks are CODE-GENERATED from the same const exchange
//! lists the CPU uses (single source: `median.rs::NET9` / `NET25` order is
//! formatted straight into the NVRTC source), so the two sides cannot
//! drift. Replicate borders, exact medians — byte parity by construction.
//!
//! One template serves `u8`, `u16` and `f32` ([`CudaMedianPixel`]): `f32`
//! samples are loaded as their bit patterns and compared through the same
//! IEEE 754 totalOrder key as the CPU kernels, so the median of every
//! window is the same sample, bit for bit, on both sides.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use kornia_tensor::CudaKernel;

use super::try_compile_with_l1;

super::define_cuda_error!(
    /// Error type for the CUDA median launcher.
    CudaMedianError,
    "CUDA median error: {0}"
);

use crate::filter::median::{NET25, NET9};

fn network_src(net: &[(usize, usize)]) -> String {
    net.iter()
        .map(|&(a, b)| format!("    CE(v[{a}], v[{b}]);\n"))
        .collect()
}

/// Device spelling of a [`crate::filter::median::MedianBlur`] pixel type.
pub(crate) trait CudaMedianPixel: DeviceRepr + ValidAsZeroBits + Copy + 'static {
    /// Kernel-name prefix and cache tag (`u8` keeps the original names).
    const TAG: &'static str;
    /// `E`: the element type the kernel loads and stores; `KEY(x)`: the
    /// order-preserving integer key of an element, its own inverse; `CE`:
    /// compare-exchange on keys.
    const PRELUDE: &'static str;
}

impl CudaMedianPixel for u8 {
    const TAG: &'static str = "";
    const PRELUDE: &'static str = r#"
typedef unsigned char E;
#define KEY(x) (x)
#define CE(a, b) { E lo = min(a, b), hi = max(a, b); a = lo; b = hi; }
"#;
}

impl CudaMedianPixel for u16 {
    const TAG: &'static str = "u16_";
    const PRELUDE: &'static str = r#"
typedef unsigned short E;
#define KEY(x) (x)
#define CE(a, b) { E lo = min(a, b), hi = max(a, b); a = lo; b = hi; }
"#;
}

impl CudaMedianPixel for f32 {
    const TAG: &'static str = "f32_";
    // `int` view of the f32 bits; KEY is the CPU `f32_total_order_key`
    // (`bits ^ ((bits >> 31) as u32 >> 1)`), expression for expression.
    const PRELUDE: &'static str = r#"
typedef int E;
#define KEY(x) ((x) ^ (int)(((unsigned int)((x) >> 31)) >> 1))
#define CE(a, b) { E lo = min(a, b), hi = max(a, b); a = lo; b = hi; }
"#;
}

fn median_src<T: CudaMedianPixel>(ksize: usize, channels: usize) -> (String, String) {
    let taps = ksize * ksize;
    let center = taps / 2;
    let r = ksize / 2;
    let tag = T::TAG;
    let prelude = T::PRELUDE;
    let name = format!("median_{tag}{ksize}x{ksize}_c{channels}");
    let net = if ksize == 3 {
        network_src(&NET9)
    } else {
        network_src(&NET25)
    };
    let src = format!(
        r#"
{prelude}
extern "C" __global__ void {name}(
    const E* __restrict__ src,
    E* __restrict__       dst,
    int w, int h
) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    #pragma unroll
    for (int c = 0; c < {channels}; ++c) {{
        E v[{taps}];
        int n = 0;
        #pragma unroll
        for (int dy = -{r}; dy <= {r}; ++dy) {{
            int sy = min(max(y + dy, 0), h - 1);
            #pragma unroll
            for (int dx = -{r}; dx <= {r}; ++dx) {{
                int sx = min(max(x + dx, 0), w - 1);
                E s = __ldg(&src[(sy * w + sx) * {channels} + c]);
                v[n++] = KEY(s);
            }}
        }}
{net}
        dst[(y * w + x) * {channels} + c] = KEY(v[{center}]);
    }}
}}
"#
    );
    (name, src)
}

type KernelMap = Mutex<HashMap<(&'static str, usize, usize), Arc<CudaKernel>>>;
static KERNELS: OnceLock<KernelMap> = OnceLock::new();

fn get_kernel<T: CudaMedianPixel>(
    ctx: &Arc<CudaContext>,
    ksize: usize,
    channels: usize,
) -> Result<Arc<CudaKernel>, CudaMedianError> {
    let map = KERNELS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(k) = map.lock().unwrap().get(&(T::TAG, ksize, channels)).cloned() {
        return Ok(k);
    }
    // Build outside the lock (scoped-guard idiom, like the sibling caches).
    let (name, src) = median_src::<T>(ksize, channels);
    let kernel = Arc::new(try_compile_with_l1(ctx, &src, &name).map_err(CudaMedianError::Cuda)?);
    Ok(map
        .lock()
        .unwrap()
        .entry((T::TAG, ksize, channels))
        .or_insert(kernel)
        .clone())
}

/// Median blur on device. `ksize` must be 3 or 5; borders replicate;
/// output is byte-identical to the CPU `median_blur` (same networks).
#[allow(clippy::too_many_arguments)]
pub fn launch_median_u8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    width: usize,
    height: usize,
    channels: usize,
    ksize: usize,
) -> Result<(), CudaMedianError> {
    launch_median(ctx, stream, src, dst, width, height, channels, ksize)
}

/// [`launch_median_u8`] for any [`CudaMedianPixel`] type; identical to the
/// CPU `median_blur` for that type.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_median<T: CudaMedianPixel>(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<T>,
    dst: &mut CudaSlice<T>,
    width: usize,
    height: usize,
    channels: usize,
    ksize: usize,
) -> Result<(), CudaMedianError> {
    if ksize != 3 && ksize != 5 {
        return Err(CudaMedianError::Cuda("ksize must be 3 or 5".into()));
    }
    if !(1..=4).contains(&channels) {
        return Err(CudaMedianError::Cuda("channels must be in 1..=4".into()));
    }
    if width == 0 || height == 0 {
        return Err(CudaMedianError::Cuda(
            "image dimensions must be non-zero".into(),
        ));
    }
    CudaMedianError::check_slice("src", src.len(), width * height * channels)?;
    CudaMedianError::check_slice("dst", dst.len(), width * height * channels)?;
    let w = i32::try_from(width).map_err(|_| CudaMedianError::Cuda("width exceeds i32".into()))?;
    let h =
        i32::try_from(height).map_err(|_| CudaMedianError::Cuda("height exceeds i32".into()))?;

    let kernel = get_kernel::<T>(ctx, ksize, channels)?;
    let cfg = super::make_config(w as u32, h as u32, None);
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&w)
        .arg(&h)
        .launch_cfg(cfg)
        .map_err(|e| CudaMedianError::Cuda(e.to_string()))
}
