//! NVRTC source generation and the per-device kernel cache for CUDA SIFT.
//!
//! Shapes are code, parameters are data: tap counts and Gaussian coefficients
//! are baked into the source as literals (a `__constant__` array measured ~3x
//! worse on Orin — per-block cold misses), and the compiled module is cached
//! per `(device ordinal, shape key)`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::CudaContext;
use kornia_tensor::CudaKernel;

use super::SiftCudaError;

/// Gaussian kernel generation matching the reference implementation exactly.
///
/// The reference evaluates this in software floating point for cross-platform
/// determinism. Hardware `f64` is used here instead: the result is narrowed to
/// `f32` at the end, which absorbs any sub-ULP disagreement, and the output was
/// verified bit-identical to the reference for every sigma the default
/// configuration uses. [`tests::gaussian_kernel_matches_reference_bitwise`]
/// pins the exact bit patterns.
///
/// The structure matters as much as the formula:
/// * `x` steps over *odd integers* `1-n, 3-n, ...` (doubled coordinates) with
///   `scale2x = -0.125 / sigma^2` instead of `-0.5 / sigma^2`, so `x*x` stays an
///   exact integer and only `exp` rounds.
/// * Only the half kernel is evaluated; the centre tap is exactly `1.0` before
///   normalisation and the other half is mirrored, so symmetry is exact rather
///   than approximate.
pub fn gaussian_kernel_f32(n: usize, sigma: f64) -> Vec<f32> {
    crate::features::sift::params::gaussian_kernel_f32(n, sigma)
}

/// Emit an `f32` literal that cannot be perturbed by decimal round-tripping.
pub(crate) fn f32_lit(v: f32) -> String {
    format!("__int_as_float(0x{:08x})", v.to_bits())
}

const BORDER_HELPERS: &str = r#"
// Reflect-101 border: ... 2 1 | 0 1 2 ... n-1 | n-2 n-3 ...  (edge not repeated)
//
// The one-step closed form covers every overshoot up to n-1 past either edge,
// which holds for all SIFT blur call sites (taps reach at most n2 <= 13 past
// an edge and the pipeline floors octaves at 16 px). The while-loop is kept as
// the general fallback — NVRTC cannot prove the loop terminates in one step,
// so with it on the hot path every border TAP carried a loop, ~5x the cost of
// the fast-path tap and a divergence amplifier (audit finding D4).
__device__ __forceinline__ int refl101(int i, int n) {
    if (n == 1) return 0;
    const int r = (i < 0) ? -i : ((i >= n) ? 2 * n - i - 2 : i);
    if (r >= 0 && r < n) return r;
    while (i < 0 || i >= n) { i = (i < 0) ? -i : (2 * n - i - 2); }
    return i;
}
// Reflect border: ... 1 0 | 0 1 2 ... n-1 | n-1 n-2 ...  (edge repeated)
__device__ __forceinline__ int refl(int i, int n) {
    while (i < 0 || i >= n) { i = (i < 0) ? (-i - 1) : (2 * n - i - 1); }
    return i;
}
"#;

/// 2x upsample of the base image.
///
/// The reference's precise-upscale option defaults to **off**, so the base
/// image is built with an ordinary separable bilinear resize — *not* the affine
/// warp the precise path uses. The two differ by a quarter pixel, which shifts
/// every downstream keypoint, so this distinction is load-bearing.
///
/// Source coordinates follow `fx = (dx + 0.5) * scale - 0.5`, `sx = floor(fx)`,
/// `fx -= sx`, with out-of-range `sx` clamped and its fraction forced to zero.
/// For a 2x upscale that yields alternating `{0.75, 0.25}` and `{0.25, 0.75}`
/// weights, with the first and last output columns/rows degenerating to a copy.
///
/// Horizontal and vertical passes are fused: the reference materialises an f32
/// intermediate row buffer, and an f32 register holds exactly the same value,
/// so fusing changes no rounding.
pub fn upsample2x_src() -> String {
    r#"
// Source tap and fraction for one output coordinate of a 2x bilinear upscale.
//
// The reference computes `(float)(((double)d + 0.5) * scale - 0.5)` then floors
// it. Evaluating that in double per output pixel is what it says, but it is not
// what it MEANS: with `scale = 0.5` the expression is `d/2 - 0.25`, and for
// integer `d < 2^23` every step is exactly representable in f32 -- `d + 0.5` is
// exact, scaling by a power of two is exact, and subtracting 0.5 is exact -- so
// no rounding can occur and the double and f32 evaluations agree bit for bit.
//
// It then reduces to a parity test, with no floating-point work at all:
//   d = 2m      -> d/2 - 0.25 = m - 0.25  ->  s = m - 1, f = 0.75
//   d = 2m + 1  -> d/2 - 0.25 = m + 0.25  ->  s = m,     f = 0.25
//
// This matters because FP64 runs at 1/32 rate on this part, and the upsample
// evaluated it twice per thread: it cost 5.9 ms against a 0.7 ms traffic floor.
__device__ __forceinline__ void up_tab(int d, int n_src, int* sx, float* fx) {
    const int odd = d & 1;
    int s = (d >> 1) - (odd ^ 1);
    float f = odd ? 0.25f : 0.75f;
    if (s < 0) { f = 0.0f; s = 0; }
    if (s >= n_src - 1) { f = 0.0f; s = n_src - 1; }
    *sx = s; *fx = f;
}

extern "C" __global__ void __launch_bounds__(256) sift_upsample2x(
    const float* __restrict__ src, float* __restrict__ dst, int sw, int sh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int dw = sw * 2, dh = sh * 2;
    if (x >= dw || y >= dh) return;

    int sx, sy; float fx, fy;
    up_tab(x, sw, &sx, &fx);
    up_tab(y, sh, &sy, &fy);
    const int sx1 = min(sx + 1, sw - 1);
    const int sy1 = min(sy + 1, sh - 1);

    // Horizontal pass on both contributing rows, then the vertical blend.
    const float r0 = src[sy  * sw + sx] * (1.0f - fx) + src[sy  * sw + sx1] * fx;
    const float r1 = src[sy1 * sw + sx] * (1.0f - fx) + src[sy1 * sw + sx1] * fx;

    dst[y * dw + x] = r0 * (1.0f - fy) + r1 * fy;
}
"#
    .to_string()
}

/// Horizontal pass of the separable Gaussian.
///
/// Mirrors the reference's row vector filter: seed with a plain multiply, then
/// accumulate every remaining tap with a single-rounding FMA.
/// `acc = s[0]*k[0]` then `acc = fma(s[k], k[k], acc)` for `k >= 1`.
pub fn blur_h_src(kernel: &[f32]) -> String {
    let n = kernel.len();
    let n2 = n / 2;
    let mut taps = String::new();
    let mut fast = String::new();
    taps.push_str(&format!(
        "        float acc = src[row + refl101(x - {n2}, w)] * {};\n",
        f32_lit(kernel[0])
    ));
    fast.push_str(&format!(
        "        float acc = __ldg(&s0[0]) * {};\n",
        f32_lit(kernel[0])
    ));
    for (k, &c) in kernel.iter().enumerate().skip(1) {
        taps.push_str(&format!(
            "        acc = __fmaf_rn(src[row + refl101(x - {n2} + {k}, w)], {}, acc);\n",
            f32_lit(c)
        ));
        fast.push_str(&format!(
            "        acc = __fmaf_rn(__ldg(&s0[{k}]), {}, acc);\n",
            f32_lit(c)
        ));
    }
    format!(
        r#"{BORDER_HELPERS}
extern "C" __global__ void __launch_bounds__(256) sift_blur_h(
    const float* __restrict__ src, float* __restrict__ dst, int w, int h)
{{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int row = y * w;
    // Interior columns need no reflection; the border path is identical maths
    // with the index clamped, so the two produce bit-identical results.
    if (x >= {n2} && x < w - {n2}) {{
        const float* __restrict__ s0 = src + row + x - {n2};
{fast}        dst[row + x] = acc;
    }} else {{
{taps}        dst[row + x] = acc;
    }}
}}
"#
    )
}

/// Horizontal Gaussian pass computing `P` consecutive outputs per thread.
///
/// A `K`-tap filter over `P` neighbouring outputs touches only `K + P - 1`
/// distinct inputs, so tiling along the filter axis cuts loads from `K*P` to
/// `K+P-1` (for K=27, P=4: 108 -> 30). The taps live in registers and each is
/// reused by up to `P` accumulators.
///
/// The arithmetic per output is unchanged and in the same order, so results are
/// bit-identical to [`blur_h_src`].
///
/// Note the workspace has measured regressions from multi-pixel-per-thread on
/// *elementwise/gather* kernels; this one has much higher arithmetic intensity,
/// so it is measured rather than assumed either way.
pub(crate) fn blur_h_tiled_src(kernel: &[f32], p: usize) -> String {
    let n = kernel.len();
    let n2 = n / 2;
    let ntaps = n + p - 1;

    let mut loads = String::new();
    for t in 0..ntaps {
        loads.push_str(&format!("        const float t{t} = __ldg(&s0[{t}]);\n"));
    }
    let mut accs = String::new();
    for j in 0..p {
        accs.push_str(&format!(
            "        float acc{j} = t{j} * {};\n",
            f32_lit(kernel[0])
        ));
        for (k, &c) in kernel.iter().enumerate().skip(1) {
            accs.push_str(&format!(
                "        acc{j} = __fmaf_rn(t{}, {}, acc{j});\n",
                j + k,
                f32_lit(c)
            ));
        }
    }
    // Vectorised stores (float2/float4 for the whole tile, guarded on the width
    // being a multiple of the tile so the row stays 16-byte aligned) were
    // implemented and measured: neutral at 752x480 and at 1080p, within noise
    // either way. Reverted -- this kernel is not store-issue bound.
    let mut stores = String::new();
    for j in 0..p {
        stores.push_str(&format!("        dst[row + xb + {j}] = acc{j};\n"));
    }

    format!(
        r#"{BORDER_HELPERS}
extern "C" __global__ void __launch_bounds__(256) sift_blur_h_tiled(
    const float* __restrict__ src, float* __restrict__ dst, int w, int h)
{{
    const int xb = (blockIdx.x * blockDim.x + threadIdx.x) * {p};
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (xb >= w || y >= h) return;
    const int row = y * w;

    // Border test at WARP granularity: the per-lane test made warp 0 of every
    // row execute both the fast and the border path (threads are consecutive
    // in x, so the test diverges inside the warp). Testing the warp's whole
    // x-span keeps every warp on one path; edge warps take the border path for
    // all lanes, which is identical arithmetic with reflected indices, so the
    // result is bit-identical and only the (already slow) edge warps widen.
    const int wx0 = (blockIdx.x * blockDim.x + (threadIdx.x & ~31)) * {p};
    if (wx0 >= {n2} && wx0 + 32 * {p} - 1 < w - {n2}) {{
        const float* __restrict__ s0 = src + row + xb - {n2};
{loads}{accs}{stores}    }} else {{
        for (int j = 0; j < {p}; ++j) {{
            const int x = xb + j;
            if (x >= w) break;
            float acc = src[row + refl101(x - {n2}, w)] * {k0};
{tail}            dst[row + x] = acc;
        }}
    }}
}}
"#,
        k0 = f32_lit(kernel[0]),
        tail = kernel
            .iter()
            .enumerate()
            .skip(1)
            .map(|(k, &c)| format!(
                "            acc = __fmaf_rn(src[row + refl101(x - {n2} + {k}, w)], {}, acc);\n",
                f32_lit(c)
            ))
            .collect::<String>(),
    )
}

/// Vertical pass of the separable Gaussian with `q` outputs per thread.
///
/// Mirrors the reference's symmetric-column vector filter: the kernel is folded
/// about its centre, and each symmetric pair is **summed before** the FMA.
/// `acc = fma(s[0], ky[0], 0)` then `acc = fma(s[k] + s[-k], ky[k], acc)`.
/// Replacing the pair-sum with two separate FMAs changes the result and breaks
/// bit equality.
pub(crate) fn blur_v_tiled_src(kernel: &[f32], q: usize, dog: bool) -> String {
    let n = kernel.len();
    let n2 = n / 2;
    let ky = &kernel[n2..];
    let ntaps = n + q - 1;

    let mut fast = String::new();
    for t in 0..ntaps {
        fast.push_str(&format!(
            "        const float t{t} = __ldg(&c0[{t} * w]);\n"
        ));
    }
    for j in 0..q {
        fast.push_str(&format!(
            "        float acc{j} = __fmaf_rn(t{}, {}, 0.0f);\n",
            n2 + j,
            f32_lit(ky[0])
        ));
        for (k, &c) in ky.iter().enumerate().skip(1) {
            fast.push_str(&format!(
                "        acc{j} = __fmaf_rn(t{} + t{}, {}, acc{j});\n",
                n2 + j - k,
                n2 + j + k,
                f32_lit(c)
            ));
        }
    }
    for j in 0..q {
        fast.push_str(&format!("        dst[(y_base + {j}) * w + x] = acc{j};\n"));
        if dog {
            fast.push_str(&format!(
                "        dog[(y_base + {j}) * w + x] = acc{j} - lower[(y_base + {j}) * w + x];\n"
            ));
        }
    }

    let mut taps = String::new();
    for j in 0..q {
        taps.push_str("    {\n");
        taps.push_str(&format!("        const int y = y_base + {j};\n"));
        taps.push_str("        if (y < h) {\n");
        taps.push_str(&format!(
            "            float acc = __fmaf_rn(src[refl101(y, h) * w + x], {}, 0.0f);\n",
            f32_lit(ky[0])
        ));
        for (k, &c) in ky.iter().enumerate().skip(1) {
            taps.push_str(&format!(
                "            acc = __fmaf_rn(src[refl101(y + {k}, h) * w + x] + src[refl101(y - {k}, h) * w + x], {}, acc);\n",
                f32_lit(c)
            ));
        }
        taps.push_str("            dst[y * w + x] = acc;\n");
        if dog {
            taps.push_str("            dog[y * w + x] = acc - lower[y * w + x];\n");
        }
        taps.push_str("        }\n");
        taps.push_str("    }\n");
    }

    let (fn_name, dog_args) = if dog {
        (
            "sift_blur_v_dog",
            "\n    const float* __restrict__ lower, float* __restrict__ dog,",
        )
    } else {
        ("sift_blur_v", "")
    };

    format!(
        r#"{BORDER_HELPERS}
extern "C" __global__ void __launch_bounds__(256) {fn_name}(
    const float* __restrict__ src, float* __restrict__ dst,{dog_args} int w, int h)
{{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;
    const int y_base = (blockIdx.y * blockDim.y + threadIdx.y) * {q};

    // Border test at warp granularity: y_base is constant for all lanes if
    // blockDim.x == 32 (threads in a warp have the same threadIdx.y).
    if (y_base >= {n2} && y_base + {q} - 1 < h - {n2}) {{
        const float* __restrict__ c0 = src + (y_base - {n2}) * w + x;
{fast}    }} else {{
{taps}    }}
}}
"#
    )
}

/// Fused horizontal+vertical Gaussian in one launch, via a shared-memory band.
///
/// Halves DRAM traffic (one read of the source, one write of the result, instead
/// of read+write+read+write). **Only worth it when the pass is DRAM-bound.**
/// Measured on this part (`bench_roofline_taps`): the blur saturates at
/// ~58.8 GB/s and is flat from ksize 5..11, then becomes issue-bound from
/// ksize >= 17. So fusion pays for small kernels and costs occupancy for large
/// ones — which is why uniform smem fusion regressed here five times before.
///
/// The block computes `BLOCK_H + ksize - 1` horizontally-blurred rows into
/// shared memory, then each thread accumulates the vertical pass from it.
/// Numerically identical to the two-pass path: the intermediate is the same f32
/// value the unfused kernel would have written to global memory.
pub(crate) fn blur_hv_fused_src(kernel: &[f32], block_w: usize, block_h: usize) -> String {
    let n = kernel.len();
    let n2 = n / 2;
    let band = block_h + n - 1;
    let ky = &kernel[n2..];

    let mut htaps = String::new();
    htaps.push_str(&format!(
        "            float acc = src[srow + refl101(gx - {n2}, w)] * {};\n",
        f32_lit(kernel[0])
    ));
    for (k, &c) in kernel.iter().enumerate().skip(1) {
        htaps.push_str(&format!(
            "            acc = __fmaf_rn(src[srow + refl101(gx - {n2} + {k}, w)], {}, acc);\n",
            f32_lit(c)
        ));
    }
    let mut vtaps = String::new();
    vtaps.push_str(&format!(
        "    float out = __fmaf_rn(tile[(ty + {n2}) * {block_w} + tx], {}, 0.0f);\n",
        f32_lit(ky[0])
    ));
    for (k, &c) in ky.iter().enumerate().skip(1) {
        vtaps.push_str(&format!(
            "    out = __fmaf_rn(tile[(ty + {n2} + {k}) * {block_w} + tx] + tile[(ty + {n2} - {k}) * {block_w} + tx], {}, out);\n",
            f32_lit(c)
        ));
    }

    format!(
        r#"{BORDER_HELPERS}
extern "C" __global__ void __launch_bounds__(256) sift_blur_hv(
    const float* __restrict__ src, float* __restrict__ dst, int w, int h)
{{
    __shared__ float tile[{band} * {block_w}];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int gx = blockIdx.x * {block_w} + tx;
    const int gy = blockIdx.y * {block_h} + ty;
    const int y0 = blockIdx.y * {block_h} - {n2};

    // Fill the band: {band} rows x {block_w} cols, strided over the whole block.
    for (int idx = ty * {block_w} + tx; idx < {band} * {block_w}; idx += {block_w} * {block_h}) {{
        const int r = idx / {block_w};
        const int c = idx % {block_w};
        const int gxx = blockIdx.x * {block_w} + c;
        if (gxx < w) {{
            const int srow = refl101(y0 + r, h) * w;
            const int gx = gxx;
{htaps}            tile[idx] = acc;
        }} else {{
            tile[idx] = 0.0f;
        }}
    }}
    __syncthreads();

    if (gx >= w || gy >= h) return;
{vtaps}    dst[gy * w + gx] = out;
}}
"#
    )
}

/// Base image of each new octave: nearest-neighbour subsample to half size.
///
/// The reference computes the source index as `floor(x * src_w / dst_w)` in
/// `double`, clamped to the last column; the same expression is evaluated per
/// thread here. This is deliberately *not* a blur-and-decimate pyramid step —
/// the blur is already baked into the source layer, and adding another would
/// break bit equality.
pub fn downsample_nearest_src() -> String {
    r#"
extern "C" __global__ void __launch_bounds__(256) sift_downsample_nearest(
    const float* __restrict__ src, float* __restrict__ dst,
    int sw, int sh, int dw, int dh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;

    // The reference computes `floor(x * (double)sw / dw)`. That equals the exact
    // rational floor, and therefore integer division, whenever rounding cannot
    // push the product across an integer boundary -- which it cannot here:
    // `x*sw/dw` is a rational with denominator `dw`, so when it is not an
    // integer it misses one by at least `1/dw >= 1/2160`, astronomically more
    // than the ~1e-16 relative error of the double evaluation. So this is
    // bit-identical, and avoids two FP64 divisions per output pixel (FP64 runs
    // at 1/32 rate on this part). `x*sw` peaks at ~7.4M, well inside int range.
    int sx = (x * sw) / dw; if (sx > sw - 1) sx = sw - 1;
    int sy = (y * sh) / dh; if (sy > sh - 1) sy = sh - 1;

    dst[y * dw + x] = src[sy * sw + sx];
}
"#
    .to_string()
}

/// Difference-of-Gaussians between two adjacent scale-space layers.
///
/// The reference computes `dst = src2 - src1` as a plain elementwise f32
/// subtract, which is exactly representable — no contraction or accumulation
/// order to mirror here.
pub fn dog_src() -> String {
    r#"
extern "C" __global__ void __launch_bounds__(256) sift_dog(
    const float* __restrict__ lower, const float* __restrict__ upper,
    float* __restrict__ dst, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int i = y * w + x;
    dst[i] = upper[i] - lower[i];
}
"#
    .to_string()
}

// ── Kernel cache ──────────────────────────────────────────────────────────────

type CacheMap = HashMap<(usize, String), Arc<CudaKernel>>;

/// Bounded cache; evict a single entry at capacity rather than clearing the map
/// (wholesale clearing makes working sets larger than the cap stampede).
const CACHE_CAP: usize = 64;

fn cache() -> &'static Mutex<CacheMap> {
    static CACHE: OnceLock<Mutex<CacheMap>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Compile-and-cache a kernel keyed on `(device ordinal, key)`.
///
/// The lookup deliberately clones out of the guard and drops it before
/// compiling: holding the lock across the miss path is a measured deadlock in
/// this workspace.
pub(crate) fn get_or_compile(
    ctx: &Arc<CudaContext>,
    key: &str,
    build_src: impl FnOnce() -> String,
    fn_name: &str,
) -> Result<Arc<CudaKernel>, SiftCudaError> {
    // Look up BEFORE generating any source. Every caller's key is already
    // content-derived — the blur keys embed the tap count and a digest of the
    // coefficients, the orientation and descriptor keys embed their variant
    // selectors — so two different sources cannot share a key.
    //
    // An earlier version hashed the generated source into the key to guard
    // against a stale-kernel hypothesis. That hypothesis was later disproved
    // (the cache is an in-process `OnceLock`, and every test run is a fresh
    // process), but the guard remained and cost a full source regeneration plus
    // a hash of it on EVERY launch — ~50 times per image, and the single
    // largest cost in the assembled pipeline.
    let cache_key = (ctx.ordinal(), key.to_string());

    {
        let guard = cache().lock().expect("SIFT kernel cache mutex poisoned");
        if let Some(k) = guard.get(&cache_key).cloned() {
            return Ok(k);
        }
    }

    let src = build_src();

    let kernel = CudaKernel::compile(ctx, &src, fn_name)
        .map_err(|e| SiftCudaError::Cuda(format!("failed to compile {fn_name}: {e}")))?;
    let _ = kernel.prefer_l1_cache();
    let kernel = Arc::new(kernel);

    let mut guard = cache().lock().expect("SIFT kernel cache mutex poisoned");
    if guard.len() >= CACHE_CAP {
        if let Some(victim) = guard.keys().next().cloned() {
            guard.remove(&victim);
        }
    }
    Ok(guard.entry(cache_key).or_insert(kernel).clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_kernel_is_normalised_and_symmetric() {
        for &(n, sigma) in &[(11, 1.2489995956420898f64), (27, 3.090015587289591)] {
            let k = gaussian_kernel_f32(n, sigma);
            assert_eq!(k.len(), n);
            for i in 0..n / 2 {
                assert_eq!(k[i], k[n - 1 - i], "kernel not symmetric at {i}");
            }
            let sum: f64 = k.iter().map(|&v| v as f64).sum();
            assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
        }
    }

    /// Raw bit patterns captured from the reference's kernel generator for the
    /// narrowest and widest sigmas the default configuration uses. Refresh from
    /// the oracle dumper if the reference version ever changes.
    #[test]
    fn gaussian_kernel_matches_reference_bitwise() {
        let k11 = gaussian_kernel_f32(11, 1.2489995956420898);
        let want11: [u32; 11] = [
            0x38ddd93d, 0x3af825aa, 0x3c9234f8, 0x3db581b7, 0x3e6d6298, 0x3ea389e7, 0x3e6d6298,
            0x3db581b7, 0x3c9234f8, 0x3af825aa, 0x38ddd93d,
        ];
        assert_eq!(
            k11.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want11.to_vec()
        );

        let k27 = gaussian_kernel_f32(27, 3.090015587289591);
        let want27: [u32; 27] = [
            0x379b5026, 0x388fc81f, 0x396fbe10, 0x3a33ffe9, 0x3af369c2, 0x3b9437e2, 0x3c228e8c,
            0x3ca08e1d, 0x3d0ecf5d, 0x3d64ca77, 0x3da50bb5, 0x3dd671db, 0x3dfaec87, 0x3e0434fb,
            0x3dfaec87, 0x3dd671db, 0x3da50bb5, 0x3d64ca77, 0x3d0ecf5d, 0x3ca08e1d, 0x3c228e8c,
            0x3b9437e2, 0x3af369c2, 0x3a33ffe9, 0x396fbe10, 0x388fc81f, 0x379b5026,
        ];
        assert_eq!(
            k27.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want27.to_vec()
        );
    }

    #[test]
    fn blur_v_dog_generates_fused_kernel() {
        let k = gaussian_kernel_f32(11, 1.2489995956420898);
        let src = blur_v_tiled_src(&k, 2, true);
        assert!(src.contains("sift_blur_v_dog("), "kernel not renamed");
        assert!(
            src.contains("const float* __restrict__ lower"),
            "lower arg missing"
        );
        assert_eq!(
            src.matches("dog[y * w + x] = acc - lower[y * w + x];")
                .count(),
            2,
            "DoG store must be emitted on BOTH the interior and border paths"
        );
        assert_eq!(
            src.matches("dog[(y_base + 0) * w + x] = acc0 - lower[(y_base + 0) * w + x];")
                .count(),
            1,
            "DoG store must be emitted on fast path"
        );
    }

    #[test]
    fn f32_literals_round_trip_exactly() {
        for v in [1.0f32, 0.1, 3.0900156, f32::MIN_POSITIVE] {
            let lit = f32_lit(v);
            let hex = lit
                .trim_start_matches("__int_as_float(0x")
                .trim_end_matches(')');
            assert_eq!(u32::from_str_radix(hex, 16).unwrap(), v.to_bits());
        }
    }
}
