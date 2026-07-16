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

    let scalar_kernel = format!(
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
    );

    // ── C=1 vector kernel: 4×4 output tile per thread ─────────────────────
    //
    // Each thread produces a 4-wide × 4-tall output tile. Per INPUT row
    // (indices `min_dy ..= max_dy + 3`) the kernel loads the aligned u32
    // words covering the tile plus the dx span once, normalizes them to a
    // byte stream anchored at `x0 + min_dx` with one __byte_perm per word,
    // and computes one horizontal fold per distinct dx-set of the element
    // (box: one; cross: two). Each output row then folds the horizontal
    // results of its tap rows vertically — for a 3×3 box that is 6 row
    // loads + 6 horizontal folds serving 16 outputs, instead of 36 loads.
    // All folds are __vmaxu4 / __vminu4: per-byte-lane u8 min/max over
    // exactly the CPU's tap multiset, so the result stays byte-exact.
    // Tiles touching a border or the buffer ends fall back to the scalar
    // bounds-checked taps per lane, which compute identical values.
    use std::collections::BTreeMap;
    let mut rows: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for pair in taps.chunks_exact(2) {
        let mut dxs = rows.remove(&pair[0]).unwrap_or_default();
        dxs.push(pair[1]);
        rows.insert(pair[0], dxs);
    }
    // Distinct dx-sets → horizontal-fold classes.
    let mut classes: Vec<Vec<i32>> = Vec::new();
    let mut row_class: BTreeMap<i32, usize> = BTreeMap::new();
    for (dy, dxs) in &rows {
        let mut key = dxs.clone();
        key.sort_unstable();
        let cid = classes.iter().position(|c| *c == key).unwrap_or_else(|| {
            classes.push(key.clone());
            classes.len() - 1
        });
        row_class.insert(*dy, cid);
    }

    let span = (max_dx - min_dx) as usize;
    // Normalized stream words R[0..rn]: highest byte read is span + 3.
    let rn = (span + 3) / 4 + 1;
    let wn = rn + 1;
    let vfold = match op {
        MorphOp::Dilate => "__vmaxu4",
        MorphOp::Erode => "__vminu4",
    };
    let vinit = match op {
        MorphOp::Dilate => "0u",
        MorphOp::Erode => "0xffffffffu",
    };

    // Which classes are needed at each input row rr = dy + j (j in 0..4).
    let rr_lo = min_dy;
    let rr_hi = max_dy + 3;
    let mut needed: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (dy, _) in &rows {
        let cid = row_class[dy];
        for j in 0..4 {
            let e = needed.entry(dy + j).or_default();
            if !e.contains(&cid) {
                e.push(cid);
            }
        }
    }

    // Horizontal-fold expression for class `cid` over stream words R{k}_{i}.
    let hfold_expr = |cid: usize, k: usize| -> String {
        let mut expr = String::new();
        for (n, dx) in classes[cid].iter().enumerate() {
            let o = (dx - min_dx) as usize;
            let (wi, rem) = (o / 4, o % 4);
            let window = if rem == 0 {
                format!("R{k}_{wi}")
            } else {
                let sel = 0x3210u32 + 0x1111 * rem as u32;
                format!("__byte_perm(R{k}_{wi}, R{k}_{}, 0x{sel:04x}u)", wi + 1)
            };
            expr = if n == 0 {
                window
            } else {
                format!("{vfold}({expr}, {window})")
            };
        }
        expr
    };

    // Per-input-row blocks: declare H vars, then load + normalize + fold.
    let mut h_decls = String::new();
    let mut row_blocks = String::new();
    for (rr, cids) in &needed {
        let k = (rr - rr_lo) as usize;
        for cid in cids {
            let _ = writeln!(h_decls, "    unsigned int H{cid}_{k};");
        }
        let mut w_loads = String::new();
        for i in 0..wn {
            let _ = writeln!(w_loads, "        unsigned int W{i} = __ldg(wp + {i});");
        }
        let mut r_norms = String::new();
        for i in 0..rn {
            let _ = writeln!(
                r_norms,
                "        unsigned int R{k}_{i} = __byte_perm(W{i}, W{}, sel_n);",
                i + 1
            );
        }
        let mut h_folds = String::new();
        for cid in cids {
            let _ = writeln!(h_folds, "        H{cid}_{k} = {};", hfold_expr(*cid, k));
        }
        let _ = write!(
            row_blocks,
            "    {{ // input row y0 + ({rr})\n\
             \x20       const unsigned char* p = src + (ptrdiff_t)(y0 + ({rr})) * width + x0 + ({min_dx});\n\
             \x20       unsigned int s = (unsigned int)((size_t)p & 3u);\n\
             \x20       const unsigned int* wp = (const unsigned int*)(p - s);\n\
             \x20       unsigned int sel_n = 0x3210u + 0x1111u * s;\n\
             {w_loads}{r_norms}{h_folds}    }}\n"
        );
    }

    // Vertical folds + stores per output row.
    let mut out_rows = String::new();
    for j in 0..4 {
        let mut expr = String::from(vinit);
        for (dy, _) in &rows {
            let cid = row_class[dy];
            let k = (dy + j - rr_lo) as usize;
            expr = format!("{vfold}({expr}, H{cid}_{k})");
        }
        let _ = write!(
            out_rows,
            "    {{\n\
             \x20       unsigned int acc4 = {expr};\n\
             \x20       size_t d = (size_t)(y0 + {j}) * width + x0;\n\
             \x20       dst[d + 0] = (unsigned char)(acc4 & 0xffu);\n\
             \x20       dst[d + 1] = (unsigned char)((acc4 >> 8) & 0xffu);\n\
             \x20       dst[d + 2] = (unsigned char)((acc4 >> 16) & 0xffu);\n\
             \x20       dst[d + 3] = (unsigned char)((acc4 >> 24) & 0xffu);\n\
             \x20   }}\n"
        );
    }

    let vector_kernel = format!(
        r#"
extern "C" __global__ void morphology_u8_c1v4(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    const unsigned char* __restrict__ constant_value,
    int width, int height, unsigned int channels, int border
) {{
    int x0 = 4 * (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y0 = 4 * (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x0 >= width || y0 >= height) return;

    bool vec_ok = x0 + 3 < width && y0 + 3 < height
               && x0 >= {neg_min_dx} && x0 + 3 < width - ({max_dx})
               && y0 >= {neg_min_dy} && y0 + 3 < height - ({max_dy});
    // The aligned word loads may reach up to 3 bytes before / after the
    // window; keep them inside the buffer (corner tiles take the scalar
    // fallback, whose values are identical).
    long long idx_lo = (long long)(y0 + ({rr_lo})) * width + x0 + ({min_dx});
    long long idx_hi = (long long)(y0 + ({rr_hi})) * width + x0 + ({min_dx});
    vec_ok = vec_ok && idx_lo >= 3
          && idx_hi + {load_span} <= (long long)width * height;

    if (vec_ok) {{
{h_decls}
{row_blocks}
{out_rows}
        return;
    }}

    unsigned int ch = 0u; // C == 1
    for (int jj = 0; jj < 4; ++jj) {{
        int yy = y0 + jj;
        if (yy >= height) break;
        unsigned int y = (unsigned int)yy;
        for (int lane = 0; lane < 4; ++lane) {{
            int x = x0 + lane;
            if (x >= width) break;
            unsigned int acc = {acc_init};
{border_stanzas_lane}            dst[(size_t)y * width + x] = (unsigned char)acc;
        }}
    }}
}}
"#,
        neg_min_dx = -min_dx,
        neg_min_dy = -min_dy,
        load_span = 4 * wn as i64,
        border_stanzas_lane = border_stanzas,
    );

    scalar_kernel + &vector_kernel
}

/// Compiled specialized kernels, keyed by (taps content, op). Structuring
/// elements are static per pipeline, so this stays tiny; on overflow the
/// cache is cleared wholesale.
type MorphKernelCache = Mutex<HashMap<(Vec<i32>, MorphOp, bool), Arc<CudaKernel>>>;
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

    // Single-channel images take the 4-pixel-per-thread vector kernel
    // (same tap multiset per byte lane via __vmaxu4/__vminu4 — byte-exact);
    // multi-channel and zero-tap elements use the scalar kernel.
    let vectorized = channels == 1 && !taps.is_empty();
    let fn_name = if vectorized {
        "morphology_u8_c1v4"
    } else {
        "morphology_u8"
    };

    let cache = MORPH_KERNELS.get_or_init(Default::default);
    let key = (taps.to_vec(), op, vectorized);
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
            try_compile_with_l1(ctx, &src_code, fn_name).map_err(CudaMorphologyError::Cuda)?,
        );
        let mut map = cache.lock().expect("morph kernel cache poisoned");
        if map.len() >= MORPH_KERNEL_CACHE_CAP {
            map.clear();
        }
        map.entry(key).or_insert(built).clone()
    };

    let border_i = border as i32;
    let (w_i, h_i) = (width as i32, height as i32);
    let (grid_w, grid_h) = if vectorized {
        (width.div_ceil(4), height.div_ceil(4))
    } else {
        (width, height)
    };

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(constant_value)
        .arg(&w_i)
        .arg(&h_i)
        .arg(&channels)
        .arg(&border_i)
        .launch_2d(grid_w, grid_h, make_config(grid_w, grid_h, block_dim))
        .map_err(|e| CudaMorphologyError::Cuda(e.to_string()))
}
