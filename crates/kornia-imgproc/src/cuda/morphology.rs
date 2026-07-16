//! Native CUDA u8 morphology (dilate / erode), byte-exact with the CPU ops.
//!
//! Morphology is a pure integer min/max over the structuring element's
//! active taps — commutative and association-free — so byte-exactness only
//! requires sampling the SAME value multiset as the CPU:
//!
//! - active taps: the `k_data[i] == 1` offsets of the
//!   [`Kernel`](crate::morphology::Kernel), relative to the output pixel;
//! - borders: the CPU pads with `spatial_padding`; the kernel instead maps
//!   out-of-bounds coordinates per pixel with the same index functions
//!   (`PaddingMode::map_index` — replicate / reflect / reflect101 / wrap)
//!   or substitutes the per-channel constant;
//! - empty-kernel corner: the CPU dilate starts its max at `T::default()`
//!   and erode's `unwrap_or_default()` yields 0 when no tap is active — the
//!   kernel writes 0 for zero taps, and dilate folds 0 into its max.
//!
//! # Per-structuring-element kernel specialization
//!
//! The tap offsets are BAKED INTO THE NVRTC SOURCE as literals and the
//! compiled kernel is cached per (element hash, op). Three designs were
//! measured on Orin (3×3 box, 1080p C1): a runtime tap loop over a global
//! LUT (1.8 ms — serialized dependent loads), the same with shared-memory
//! staging (2.2 ms — the 8 KB static allocation cut occupancy and fought
//! the PREFER_L1 carveout), and baked literals with a fully unrolled loop
//! (fastest — the pixel loads issue independently). Structuring elements
//! are static per pipeline, so the one-time NVRTC compile (~1 s) amortizes
//! to zero; interior threads (window in bounds by the tap extents) skip all
//! border logic.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::{make_config, try_compile_with_l1};

/// Error type for the CUDA morphology launchers.
#[derive(Debug, thiserror::Error)]
pub enum CudaMorphologyError {
    /// CUDA kernel compilation or launch failure.
    #[error("CUDA kernel compile/launch error: {0}")]
    Cuda(String),
    /// Output device slice is smaller than the required element count.
    #[error("output slice length {got} < required {need}")]
    SliceTooSmall {
        /// Actual slice length (in elements).
        got: usize,
        /// Minimum required length (width × height × channels).
        need: usize,
    },
}

/// Which morphological reduction the kernel applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MorphOp {
    /// Neighborhood maximum.
    Dilate,
    /// Neighborhood minimum.
    Erode,
}

/// Border handling, mirroring `padding::PaddingMode` (same index math).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphBorder {
    /// Fill with the per-channel constant value.
    Constant = 0,
    /// Clamp to the edge pixel.
    Replicate = 1,
    /// Mirror excluding the edge pixel (101 flavor).
    Reflect101 = 2,
    /// Mirror including the edge pixel.
    Reflect = 3,
    /// Circular wrap.
    Wrap = 4,
}

/// Maximum active taps a specialized kernel will bake (a 45×45 box).
pub const MORPH_MAX_TAPS: usize = 1024;

/// Generate the specialized source for one structuring element + op.
fn morph_source(taps: &[i32], op: MorphOp) -> String {
    let ntaps = taps.len() / 2;
    let (mut min_dy, mut max_dy, mut min_dx, mut max_dx) = (0i32, 0i32, 0i32, 0i32);
    let mut tap_lits = String::new();
    for (i, pair) in taps.chunks_exact(2).enumerate() {
        let (dy, dx) = (pair[0], pair[1]);
        if i == 0 {
            (min_dy, max_dy, min_dx, max_dx) = (dy, dy, dx, dx);
        } else {
            min_dy = min_dy.min(dy);
            max_dy = max_dy.max(dy);
            min_dx = min_dx.min(dx);
            max_dx = max_dx.max(dx);
        }
        let _ = write!(tap_lits, "{{{dy},{dx}}},");
    }
    let (acc_init, fold) = match op {
        MorphOp::Dilate => ("0u", "max"),
        MorphOp::Erode => ("255u", "min"),
    };
    // Zero-tap corner: both CPU ops default to 0 — emit a memset-style body.
    if ntaps == 0 {
        return r#"
extern "C" __global__ void morphology_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    const unsigned char* __restrict__ constant_value,
    int width, int height, unsigned int channels, int border
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= (unsigned int)width || y >= (unsigned int)height) return;
    size_t d = ((size_t)y * width + x) * channels;
    for (unsigned int ch = 0; ch < channels; ++ch) dst[d + ch] = 0;
}
"#
        .to_string();
    }

    // Straight-line per-tap stanzas with literal offsets: the compiler folds
    // `dy*width + dx` into per-tap address math and every load issues
    // independently. (A `__constant__ TAPS[][2]` array variant measured 3×
    // slower — per-block constant-cache cold misses.)
    let mut interior_stanzas = String::new();
    let mut border_stanzas = String::new();
    for pair in taps.chunks_exact(2) {
        let (dy, dx) = (pair[0], pair[1]);
        let _ = writeln!(
            interior_stanzas,
            "        acc = {fold}(acc, (unsigned int)__ldg(base + \
             ((ptrdiff_t)({dy}) * width + ({dx})) * (ptrdiff_t)channels));"
        );
        let _ = writeln!(border_stanzas, "        BORDER_TAP({dy}, {dx});");
    }
    let _ = tap_lits; // literals are emitted per-stanza above

    format!(
        r#"
// Mirrors padding::PaddingMode::map_index (iterative reflect loops match
// the CPU exactly; pads are tiny).
__device__ __forceinline__ int map_index(int i, int len, int mode) {{
    if (mode == 1) {{ return min(max(i, 0), len - 1); }}
    if (mode == 2) {{
        if (len == 1) return 0;
        while (i < 0 || i >= len) {{ if (i < 0) i = -i; else i = 2 * len - i - 2; }}
        return i;
    }}
    if (mode == 3) {{
        if (len == 1) return 0;
        while (i < 0 || i >= len) {{ if (i < 0) i = -i - 1; else i = 2 * len - i - 1; }}
        return i;
    }}
    if (mode == 4) {{ int m = i % len; return m < 0 ? m + len : m; }}
    return 0;
}}

#define BORDER_TAP(dy, dx) do {{ \
    int sy = (int)y + (dy); \
    int sx = (int)x + (dx); \
    unsigned int v; \
    if (sx >= 0 && sx < width && sy >= 0 && sy < height) {{ \
        v = (unsigned int)__ldg(&src[((size_t)sy * width + sx) * channels + ch]); \
    }} else if (border == 0) {{ \
        v = (unsigned int)__ldg(&constant_value[ch]); \
    }} else {{ \
        int mx = map_index(sx, width, border); \
        int my = map_index(sy, height, border); \
        v = (unsigned int)__ldg(&src[((size_t)my * width + mx) * channels + ch]); \
    }} \
    acc = {fold}(acc, v); \
}} while (0)

extern "C" __global__ void morphology_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    const unsigned char* __restrict__ constant_value,
    int width, int height, unsigned int channels, int border
) {{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= (unsigned int)width || y >= (unsigned int)height) return;

    bool interior = (int)x >= {neg_min_dx} && (int)x < width - ({max_dx})
                 && (int)y >= {neg_min_dy} && (int)y < height - ({max_dy});

    size_t d = ((size_t)y * width + x) * channels;
    for (unsigned int ch = 0; ch < channels; ++ch) {{
        unsigned int acc = {acc_init};
        if (interior) {{
            const unsigned char* base = src + d + ch;
{interior_stanzas}        }} else {{
{border_stanzas}        }}
        dst[d + ch] = (unsigned char)acc;
    }}
}}
"#,
        neg_min_dx = -min_dx,
        neg_min_dy = -min_dy,
    )
}

/// Compiled specialized kernels, keyed by (taps content, op). Structuring
/// elements are static per pipeline, so this stays tiny; on overflow the
/// cache is cleared wholesale.
type MorphKernelCache = Mutex<HashMap<(Vec<i32>, MorphOp), Arc<CudaKernel>>>;
static MORPH_KERNELS: OnceLock<MorphKernelCache> = OnceLock::new();
const MORPH_KERNEL_CACHE_CAP: usize = 64;

/// Launch the u8 morphology kernel (dilate or erode), specialized for the
/// given structuring element.
///
/// `taps` are the active element offsets as `(dy, dx)` pairs relative to
/// the output pixel, at most [`MORPH_MAX_TAPS`]; they are baked into the
/// kernel source (compiled once per element + op, then cached).
/// `constant_value` is a device upload of the per-channel constant (read
/// only for [`MorphBorder::Constant`], but must hold `channels` bytes).
#[allow(clippy::too_many_arguments)]
pub fn launch_morphology_u8_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    width: u32,
    height: u32,
    channels: u32,
    taps: &[i32],
    constant_value: &CudaSlice<u8>,
    op: MorphOp,
    border: MorphBorder,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaMorphologyError> {
    super::check_geometry(width, height, width, height, block_dim)
        .map_err(CudaMorphologyError::Cuda)?;
    let need = (width as usize) * (height as usize) * (channels as usize);
    if dst.len() < need {
        return Err(CudaMorphologyError::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }
    if !taps.len().is_multiple_of(2) {
        return Err(CudaMorphologyError::Cuda(
            "taps must be (dy, dx) pairs".into(),
        ));
    }
    if taps.len() / 2 > MORPH_MAX_TAPS {
        return Err(CudaMorphologyError::Cuda(format!(
            "structuring element has {} active taps; the GPU kernel supports \
             at most {MORPH_MAX_TAPS} (use the CPU path for larger elements)",
            taps.len() / 2
        )));
    }
    if constant_value.len() < channels as usize {
        return Err(CudaMorphologyError::Cuda(
            "constant_value must hold at least `channels` bytes".into(),
        ));
    }

    let cache = MORPH_KERNELS.get_or_init(Default::default);
    let key = (taps.to_vec(), op);
    let cached = cache
        .lock()
        .expect("morph kernel cache poisoned")
        .get(&key)
        .cloned();
    let kernel = if let Some(hit) = cached {
        hit
    } else {
        let src_code = morph_source(taps, op);
        let built = Arc::new(
            try_compile_with_l1(ctx, &src_code, "morphology_u8")
                .map_err(CudaMorphologyError::Cuda)?,
        );
        let mut map = cache.lock().expect("morph kernel cache poisoned");
        if map.len() >= MORPH_KERNEL_CACHE_CAP {
            map.clear();
        }
        map.entry(key).or_insert(built).clone()
    };

    let border_i = border as i32;
    let (w_i, h_i) = (width as i32, height as i32);

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(constant_value)
        .arg(&w_i)
        .arg(&h_i)
        .arg(&channels)
        .arg(&border_i)
        .launch_2d(width, height, make_config(width, height, block_dim))
        .map_err(|e| CudaMorphologyError::Cuda(e.to_string()))
}
