//! CUDA CLAHE kernels — textual twins of `clahe.rs`'s two host stages.
//!
//! * `clahe_lut_u8`: one block per tile. Shared-memory histogram over the
//!   (virtually reflect_101-extended) tile via `atomicAdd` (commutative —
//!   counts exact), then thread 0 runs cv2's clip + two-phase
//!   redistribution + cdf scan sequentially (deterministic), and all 256
//!   threads quantize their LUT entry with the f32 scale and `rintf`
//!   (round-half-to-even) — mirroring `build_tile_luts` line for line.
//! * `clahe_apply_u8`: per-pixel f32 bilinear blend of the 4 surrounding
//!   tile LUTs with cv2's compiled FMA contraction (explicit `fmaf` — the
//!   global NVRTC `fmad=false` only disables IMPLICIT contraction),
//!   mirroring the interpolation loop in `clahe.rs`.
//!
//! `inv_tw` / `inv_th` / `lut_scale` are computed ONCE on the host
//! (`ClaheGeometry`) and passed as arguments so both sides share the same
//! f32 divisions.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use crate::clahe::ClaheGeometry;

super::define_cuda_error!(
    /// Error type for the CUDA CLAHE launchers.
    CudaClaheError,
    "CUDA CLAHE error: {0}"
);

fn dim_u32(what: &'static str, v: usize) -> Result<u32, CudaClaheError> {
    u32::try_from(v).map_err(|_| CudaClaheError::Cuda(format!("{what} exceeds u32")))
}

static LUT_SRC: &str = r#"
extern "C" __global__ void clahe_lut_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       luts,
    int w, int h,
    int tiles_x,
    int tile_w, int tile_h,
    int clip_limit,
    float lut_scale
) {
    __shared__ int hist[256];
    const int tid = threadIdx.x;
    const int tile = blockIdx.x;
    const int ty = tile / tiles_x;
    const int tx = tile % tiles_x;

    hist[tid] = 0;
    __syncthreads();

    // Histogram over the (virtually reflect_101-extended) tile.
    const int area = tile_w * tile_h;
    for (int i = tid; i < area; i += 256) {
        int ly = i / tile_w;
        int lx = i - ly * tile_w;
        int sy = reflect_101(ty * tile_h + ly, h);
        int sx = reflect_101(tx * tile_w + lx, w);
        atomicAdd(&hist[__ldg(&src[sy * w + sx])], 1);
    }
    __syncthreads();

    // Clip + redistribute + cdf scan: thread 0, sequential — mirrors the
    // CPU loop in build_tile_luts exactly (integer, deterministic).
    if (tid == 0) {
        if (clip_limit > 0) {
            int clipped = 0;
            for (int i = 0; i < 256; ++i) {
                if (hist[i] > clip_limit) {
                    clipped += hist[i] - clip_limit;
                    hist[i] = clip_limit;
                }
            }
            int redist_batch = clipped / 256;
            int residual = clipped - redist_batch * 256;
            for (int i = 0; i < 256; ++i) {
                hist[i] += redist_batch;
            }
            if (residual != 0) {
                int residual_step = max(256 / residual, 1);
                for (int i = 0; i < 256 && residual > 0; i += residual_step, --residual) {
                    hist[i] += 1;
                }
            }
        }
        int sum = 0;
        for (int i = 0; i < 256; ++i) {
            sum += hist[i];
            hist[i] = sum;  // reuse as cdf
        }
    }
    __syncthreads();

    // f32 quantization, round-half-to-even — twin of the CPU's
    // (sum as f32 * lut_scale).round_ties_even().clamp(0, 255).
    float v = (float)hist[tid] * lut_scale;
    luts[tile * 256 + tid] = (unsigned char)min(max((int)rintf(v), 0), 255);
}
"#;

static APPLY_SRC: &str = r#"
extern "C" __global__ void clahe_apply_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    const unsigned char* __restrict__ luts,
    int w, int h,
    int tiles_x, int tiles_y,
    float inv_tw, float inv_th
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // Twin of the CPU per-column/per-row precompute in clahe.rs (same f32
    // expressions, fmad-free).
    float txf = fmaf((float)x, inv_tw, -0.5f);
    int tx1 = (int)floorf(txf);
    int tx2 = tx1 + 1;
    float xa = txf - (float)tx1;
    float xa1 = 1.0f - xa;
    int ind1 = max(tx1, 0) * 256;
    int ind2 = min(tx2, tiles_x - 1) * 256;

    float tyf = fmaf((float)y, inv_th, -0.5f);
    int ty1 = (int)floorf(tyf);
    int ty2 = ty1 + 1;
    float ya = tyf - (float)ty1;
    float ya1 = 1.0f - ya;
    const unsigned char* p1 = luts + max(ty1, 0) * tiles_x * 256;
    const unsigned char* p2 = luts + min(ty2, tiles_y - 1) * tiles_x * 256;

    int v = __ldg(&src[y * w + x]);
    // cv2's blend with its compiled FMA contraction (see clahe.rs).
    float i1 = fmaf((float)__ldg(&p1[ind1 + v]), xa1, (float)__ldg(&p1[ind2 + v]) * xa);
    float i2 = fmaf((float)__ldg(&p2[ind1 + v]), xa1, (float)__ldg(&p2[ind2 + v]) * xa);
    float res = fmaf(i1, ya1, i2 * ya);
    dst[y * w + x] = (unsigned char)min(max((int)rintf(res), 0), 255);
}
"#;

static LUT_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static APPLY_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

/// Build the per-tile CLAHE LUTs on device (`tiles_y · tiles_x · 256`
/// bytes, tile-major — same layout as `build_tile_luts`).
pub fn launch_clahe_lut_u8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    luts: &mut CudaSlice<u8>,
    width: usize,
    height: usize,
    g: &ClaheGeometry,
) -> Result<(), CudaClaheError> {
    let tiles = g.tiles_x * g.tiles_y;
    if tiles == 0 || g.tile_w == 0 || g.tile_h == 0 {
        return Err(CudaClaheError::Cuda("empty tile geometry".into()));
    }
    CudaClaheError::check_slice("src", src.len(), width * height)?;
    CudaClaheError::check_slice("luts", luts.len(), tiles * 256)?;
    let w = dim_u32("width", width)? as i32;
    let h = dim_u32("height", height)? as i32;
    let tiles_u32 = dim_u32("tiles", tiles)?;
    let (tw, th) = (
        dim_u32("tile_w", g.tile_w)? as i32,
        dim_u32("tile_h", g.tile_h)? as i32,
    );
    // Tile area must fit i32 (histogram counts are i32, like cv2's).
    i32::try_from(g.tile_w * g.tile_h)
        .map_err(|_| CudaClaheError::Cuda("tile area exceeds i32".into()))?;
    let kernel = {
        static COMPOSED: OnceLock<String> = OnceLock::new();
        let src = COMPOSED.get_or_init(|| format!("{}{}", super::pyramid::REFLECT_101, LUT_SRC));
        CudaClaheError::get_kernel(&LUT_KERNEL, ctx, src, "clahe_lut_u8")
    }?;
    let cfg = cudarc::driver::LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (tiles_u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let tiles_x = g.tiles_x as i32;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(luts)
        .arg(&w)
        .arg(&h)
        .arg(&tiles_x)
        .arg(&tw)
        .arg(&th)
        .arg(&g.clip_limit)
        .arg(&g.lut_scale)
        .launch_cfg(cfg)
        .map_err(|e| CudaClaheError::Cuda(e.to_string()))
}

/// Blend the 4 surrounding tile LUTs per pixel (cv2's f32 bilinear scheme).
#[allow(clippy::too_many_arguments)]
pub fn launch_clahe_apply_u8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    luts: &CudaSlice<u8>,
    width: usize,
    height: usize,
    g: &ClaheGeometry,
) -> Result<(), CudaClaheError> {
    if g.tiles_x == 0 || g.tiles_y == 0 {
        return Err(CudaClaheError::Cuda("empty tile geometry".into()));
    }
    CudaClaheError::check_slice("src", src.len(), width * height)?;
    CudaClaheError::check_slice("dst", dst.len(), width * height)?;
    CudaClaheError::check_slice("luts", luts.len(), g.tiles_x * g.tiles_y * 256)?;
    let w32 = dim_u32("width", width)?;
    let h32 = dim_u32("height", height)?;
    let (w, h) = (w32 as i32, h32 as i32);
    let kernel = CudaClaheError::get_kernel(&APPLY_KERNEL, ctx, APPLY_SRC, "clahe_apply_u8")?;
    let cfg = super::make_config(w32, h32, None);
    let (tiles_x, tiles_y) = (g.tiles_x as i32, g.tiles_y as i32);
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(luts)
        .arg(&w)
        .arg(&h)
        .arg(&tiles_x)
        .arg(&tiles_y)
        .arg(&g.inv_tw)
        .arg(&g.inv_th)
        .launch_cfg(cfg)
        .map_err(|e| CudaClaheError::Cuda(e.to_string()))
}
