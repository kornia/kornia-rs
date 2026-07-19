//! CUDA Canny kernels — textual twins of `canny.rs`'s integer pipeline.
//!
//! * `canny_sobel_mag`: fused Sobel 3×3 (`CV_16S`, replicate borders) +
//!   magnitude (L1 or L2 baked per variant) writing `dx`, `dy` and the
//!   ring-padded magnitude (`mstep = w + 2`, halo pre-zeroed) — exact
//!   integer transcription of `sobel3_i16` + the magnitude loop.
//! * `canny_nms`: OpenCV's fixed-point sector test with the exact
//!   tie-break asymmetries, writing the 0/1/2 map (ring pre-set to 1) —
//!   strong pixels also bump a global counter so the host knows whether to
//!   run hysteresis at all.
//! * `canny_hysteresis`: block-local shared-memory fixpoint (edges
//!   propagate across a whole 32×8 tile per launch) + a global `changed`
//!   flag; the host relaunches until stable. The fixpoint computes
//!   REACHABILITY — the same set the CPU stack flood computes — so the
//!   result is byte-identical regardless of sweep count.
//! * `canny_finalize`: `dst = (map == 2) ? 255 : 0`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::try_compile_with_l1;

/// Error type for the CUDA Canny launchers.
#[derive(Debug, thiserror::Error)]
pub enum CudaCannyError {
    /// CUDA driver / compile / launch error.
    #[error("CUDA canny error: {0}")]
    Cuda(String),
    /// A slice is smaller than required.
    #[error("device slice '{what}' length {got} < required {need}")]
    SliceTooSmall {
        /// Which operand was too small.
        what: &'static str,
        /// Actual length (elements).
        got: usize,
        /// Required length (elements).
        need: usize,
    },
}

fn check_slice(what: &'static str, got: usize, need: usize) -> Result<(), CudaCannyError> {
    if got < need {
        return Err(CudaCannyError::SliceTooSmall { what, got, need });
    }
    Ok(())
}

static SOBEL_MAG_SRC: &str = r#"
extern "C" __global__ void canny_sobel_mag(
    const unsigned char* __restrict__ src,
    short* __restrict__               dx,
    short* __restrict__               dy,
    int* __restrict__                 mag,
    int w, int h, int l2
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // Twin of canny.rs::sobel3_i16 (replicate borders, exact integer).
    int xm = max(x - 1, 0), xp = min(x + 1, w - 1);
    int ym = max(y - 1, 0), yp = min(y + 1, h - 1);
    const unsigned char* r0 = src + ym * w;
    const unsigned char* r1 = src + y * w;
    const unsigned char* r2 = src + yp * w;
    int a = __ldg(&r0[xm]), b = __ldg(&r0[x]), c = __ldg(&r0[xp]);
    int d = __ldg(&r1[xm]),                    f = __ldg(&r1[xp]);
    int g = __ldg(&r2[xm]), hh = __ldg(&r2[x]), i = __ldg(&r2[xp]);
    int gx = (c + 2 * f + i) - (a + 2 * d + g);
    int gy = (g + 2 * hh + i) - (a + 2 * b + c);
    dx[y * w + x] = (short)gx;
    dy[y * w + x] = (short)gy;
    // Ring-padded magnitude (mstep = w + 2), halo pre-zeroed by the host.
    mag[(y + 1) * (w + 2) + (x + 1)] = l2 ? (gx * gx + gy * gy) : (abs(gx) + abs(gy));
}
"#;

static NMS_SRC: &str = r#"
extern "C" __global__ void canny_nms(
    const short* __restrict__ dx,
    const short* __restrict__ dy,
    const int* __restrict__   mag,
    unsigned char* __restrict__ map,
    unsigned int* __restrict__  strong_count,
    int w, int h, int low, int high
) {
    const int TG22 = 13573;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int mstep = w + 2;
    int k = x + 1;
    const int* mag_p = mag + y * mstep;
    const int* mag_a = mag + (y + 1) * mstep;
    const int* mag_n = mag + (y + 2) * mstep;
    int m = __ldg(&mag_a[k]);
    unsigned char label = 1; // NON_EDGE
    if (m > low) {
        int xs = __ldg(&dx[y * w + x]);
        int ys = __ldg(&dy[y * w + x]);
        int ax = abs(xs);
        int ay15 = abs(ys) << 15;
        int tg22x = ax * TG22;
        // cv2's exact sector conditions and tie-breaks (see canny.rs).
        bool is_max;
        if (ay15 < tg22x) {
            is_max = m > __ldg(&mag_a[k - 1]) && m >= __ldg(&mag_a[k + 1]);
        } else {
            int tg67x = tg22x + (ax << 16);
            if (ay15 > tg67x) {
                is_max = m > __ldg(&mag_p[k]) && m >= __ldg(&mag_n[k]);
            } else {
                int s = ((xs ^ ys) < 0) ? -1 : 1;
                is_max = m > __ldg(&mag_p[k - s]) && m > __ldg(&mag_n[k + s]);
            }
        }
        if (is_max) {
            if (m > high) {
                label = 2; // EDGE (strong seed)
                atomicAdd(strong_count, 1u);
            } else {
                label = 0; // CANDIDATE
            }
        }
    }
    map[(y + 1) * mstep + k] = label;
}
"#;

static HYSTERESIS_SRC: &str = r#"
// Block-local fixpoint over a 64x16 macro-tile (each 32x8 thread owns a
// 2x2 cell quad in shared memory) with an ACTIVE-TILE worklist (128-wide
// tiles were tried and regressed ~25%: more wasted work per active tile): a tile
// runs only when flagged, and a tile that changes wakes itself and its 8
// neighbors for the next sweep — converged regions cost one byte read per
// sweep. Computes REACHABILITY, identical to the CPU stack flood.
#define TW 64
#define TH 16
extern "C" __global__ void canny_hysteresis(
    unsigned char* __restrict__ map,
    const unsigned char* __restrict__ active_in,
    unsigned char* __restrict__       active_out,
    unsigned int* __restrict__  changed,
    int w, int h, int tiles_x, int tiles_y
) {
    int bx = blockIdx.x, by = blockIdx.y;
    if (!active_in[by * tiles_x + bx]) return;

    __shared__ unsigned char tile[TH + 2][TW + 2];
    __shared__ int block_state; // bit0: any candidate, bit1: changed
    int mstep = w + 2;
    int x0 = bx * TW;
    int y0 = by * TH;
    int tx = threadIdx.x, ty = threadIdx.y;

    if (tx == 0 && ty == 0) block_state = 0;
    __syncthreads();

    bool saw_cand = false;
    for (int yy = ty; yy < TH + 2; yy += 8) {
        for (int xx = tx; xx < TW + 2; xx += 32) {
            int gy = y0 + yy;
            int gx = x0 + xx;
            unsigned char v = (gy <= h + 1 && gx <= w + 1) ? map[gy * mstep + gx] : 1;
            tile[yy][xx] = v;
            saw_cand |= (v == 0);
        }
    }
    if (saw_cand) atomicOr(&block_state, 1);
    __syncthreads();
    if (!(block_state & 1)) return; // no candidates: nothing can change

    for (;;) {
        __shared__ int iter_changed;
        if (tx == 0 && ty == 0) iter_changed = 0;
        __syncthreads();
        bool any = false;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            int yy = ty * 2 + (q >> 1) + 1;
            int xx = tx * 2 + (q & 1) + 1;
            if ((y0 + yy) <= h && (x0 + xx) <= w && tile[yy][xx] == 0) {
                if (tile[yy - 1][xx - 1] == 2 || tile[yy - 1][xx] == 2 || tile[yy - 1][xx + 1] == 2 ||
                    tile[yy][xx - 1] == 2     || tile[yy][xx + 1] == 2 ||
                    tile[yy + 1][xx - 1] == 2 || tile[yy + 1][xx] == 2 || tile[yy + 1][xx + 1] == 2) {
                    tile[yy][xx] = 2;
                    any = true;
                }
            }
        }
        if (any) iter_changed = 1;
        __syncthreads();
        if (!iter_changed) break;
        if (tx == 0 && ty == 0) block_state |= 2;
        __syncthreads();
    }

    if (!(block_state & 2)) return; // stable: no writeback, no wakeups

    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        int yy = ty * 2 + (q >> 1) + 1;
        int xx = tx * 2 + (q & 1) + 1;
        if ((y0 + yy) <= h && (x0 + xx) <= w) {
            map[(y0 + yy) * mstep + (x0 + xx)] = tile[yy][xx];
        }
    }
    if (tx == 0 && ty == 0) {
        atomicAdd(changed, 1u);
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nby = by + dy, nbx = bx + dx;
                if (nby >= 0 && nby < tiles_y && nbx >= 0 && nbx < tiles_x) {
                    active_out[nby * tiles_x + nbx] = 1;
                }
            }
        }
    }
}
"#;

static FILL_SRC: &str = r#"
// Initialize ONLY the one-pixel rings: map ring = 1 (NON_EDGE), mag ring
// = 0. The interiors are fully overwritten by canny_sobel_mag / canny_nms,
// so zero-initializing the whole buffers would be pure wasted bandwidth.
extern "C" __global__ void canny_fill_rings(
    unsigned char* __restrict__ map,
    int* __restrict__           mag,
    int w, int h
) {
    int mstep = w + 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int per = 2 * (w + 2) + 2 * h; // top + bottom rows, left + right cols
    if (i >= per) return;
    int idx;
    if (i < w + 2) {
        idx = i;                             // top row
    } else if (i < 2 * (w + 2)) {
        idx = (h + 1) * mstep + (i - (w + 2)); // bottom row
    } else {
        int j = i - 2 * (w + 2);
        int row = 1 + (j >> 1);
        idx = row * mstep + ((j & 1) ? (w + 1) : 0); // side columns
    }
    map[idx] = 1;
    mag[idx] = 0;
}
"#;

static FINALIZE_SRC: &str = r#"
extern "C" __global__ void canny_finalize(
    const unsigned char* __restrict__ map,
    unsigned char* __restrict__       dst,
    int w, int h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    dst[y * w + x] = (map[(y + 1) * (w + 2) + (x + 1)] == 2) ? 255 : 0;
}
"#;

static SOBEL_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static NMS_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static HYST_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static FINAL_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static FILL_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

fn get_kernel(
    cell: &'static OnceLock<Result<CudaKernel, String>>,
    ctx: &Arc<CudaContext>,
    src: &str,
    name: &str,
) -> Result<&'static CudaKernel, CudaCannyError> {
    cell.get_or_init(|| try_compile_with_l1(ctx, src, name))
        .as_ref()
        .map_err(|e| CudaCannyError::Cuda(e.clone()))
}

/// Full device Canny: gradients + NMS + hysteresis relaunch loop +
/// finalize. Byte-identical to the CPU `canny` (integer pipeline +
/// reachability fixpoint). Synchronizes the stream between hysteresis
/// sweeps to read the `changed` flag.
#[allow(clippy::too_many_arguments)]
pub fn launch_canny_u8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    width: usize,
    height: usize,
    low: i32,
    high: i32,
    l2_gradient: bool,
) -> Result<(), CudaCannyError> {
    if width == 0 || height == 0 {
        return Err(CudaCannyError::Cuda(
            "image dimensions must be non-zero".into(),
        ));
    }
    check_slice("src", src.len(), width * height)?;
    check_slice("dst", dst.len(), width * height)?;
    let w = i32::try_from(width).map_err(|_| CudaCannyError::Cuda("width exceeds i32".into()))?;
    let h = i32::try_from(height).map_err(|_| CudaCannyError::Cuda("height exceeds i32".into()))?;
    // Magnitude fits i32 for both L1 (<= 2040) and L2 (<= 2*1020^2).
    let err = |e: cudarc::driver::DriverError| CudaCannyError::Cuda(e.to_string());
    let mstep = width + 2;
    let ring = mstep * (height + 2);

    // SAFETY: dx/dy/mag interiors and the map interior are fully written
    // by canny_sobel_mag / canny_nms before any read; the rings are set by
    // canny_fill_rings below. No uninitialized element is ever read.
    let mut d_dx = unsafe { stream.alloc::<i16>(width * height) }.map_err(err)?;
    let mut d_dy = unsafe { stream.alloc::<i16>(width * height) }.map_err(err)?;
    let mut d_mag = unsafe { stream.alloc::<i32>(ring) }.map_err(err)?;
    let mut d_map = unsafe { stream.alloc::<u8>(ring) }.map_err(err)?;
    {
        let per = 2 * (width + 2) + 2 * height;
        let k = get_kernel(&FILL_KERNEL, ctx, FILL_SRC, "canny_fill_rings")?;
        let cfg1 = cudarc::driver::LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((per as u32).div_ceil(256), 1, 1),
            shared_mem_bytes: 0,
        };
        k.launch_builder(stream)
            .arg(&mut d_map)
            .arg(&mut d_mag)
            .arg(&w)
            .arg(&h)
            .launch_cfg(cfg1)
            .map_err(|e| CudaCannyError::Cuda(e.to_string()))?;
    }
    let mut d_flag = stream.alloc_zeros::<u32>(2).map_err(err)?; // [strong, changed]

    let cfg = super::make_config(w as u32, h as u32, None);
    let l2 = i32::from(l2_gradient);

    let k = get_kernel(&SOBEL_KERNEL, ctx, SOBEL_MAG_SRC, "canny_sobel_mag")?;
    k.launch_builder(stream)
        .arg(src)
        .arg(&mut d_dx)
        .arg(&mut d_dy)
        .arg(&mut d_mag)
        .arg(&w)
        .arg(&h)
        .arg(&l2)
        .launch_cfg(cfg)
        .map_err(|e| CudaCannyError::Cuda(e.to_string()))?;

    let k = get_kernel(&NMS_KERNEL, ctx, NMS_SRC, "canny_nms")?;
    k.launch_builder(stream)
        .arg(&d_dx)
        .arg(&d_dy)
        .arg(&d_mag)
        .arg(&mut d_map)
        .arg(&mut d_flag)
        .arg(&w)
        .arg(&h)
        .arg(&low)
        .arg(&high)
        .launch_cfg(cfg)
        .map_err(|e| CudaCannyError::Cuda(e.to_string()))?;

    // Hysteresis: relaunch the block-fixpoint sweep until no block changes.
    // (No early strong-count check: the sync it needs costs more than the
    // worklist sweeps it could skip.)
    {
        let hk = get_kernel(&HYST_KERNEL, ctx, HYSTERESIS_SRC, "canny_hysteresis")?;
        let tiles_x = width.div_ceil(64);
        let tiles_y = height.div_ceil(16);
        let ntiles = tiles_x * tiles_y;
        let hcfg = cudarc::driver::LaunchConfig {
            block_dim: (32, 8, 1),
            grid_dim: (tiles_x as u32, tiles_y as u32, 1),
            shared_mem_bytes: 0,
        };
        let (tx_i, ty_i) = (tiles_x as i32, tiles_y as i32);
        // Active-tile ping-pong: everything active for sweep 1; afterwards
        // only tiles woken by a changed neighbor run.
        let all_active = vec![1u8; ntiles];
        let mut d_active_a = stream.clone_htod(&all_active).map_err(err)?;
        let mut d_active_b = stream.alloc_zeros::<u8>(ntiles).map_err(err)?;
        // Termination: every sweep that reports `changed` converted at
        // least one candidate, and candidates are finite — so w·h is an
        // absolute upper bound (typical images converge in a handful).
        // Batch 3 sweeps per host sync (a sync stall costs more than a
        // worklist-empty sweep).
        let max_rounds = width * height;
        for _ in 0..max_rounds {
            stream.memset_zeros(&mut d_flag).map_err(err)?;
            for _ in 0..3 {
                stream.memset_zeros(&mut d_active_b).map_err(err)?;
                hk.launch_builder(stream)
                    .arg(&mut d_map)
                    .arg(&d_active_a)
                    .arg(&mut d_active_b)
                    .arg(&mut d_flag)
                    .arg(&w)
                    .arg(&h)
                    .arg(&tx_i)
                    .arg(&ty_i)
                    .launch_cfg(hcfg)
                    .map_err(|e| CudaCannyError::Cuda(e.to_string()))?;
                std::mem::swap(&mut d_active_a, &mut d_active_b);
            }
            let f: Vec<u32> = stream.clone_dtoh(&d_flag).map_err(err)?;
            stream.synchronize().map_err(err)?;
            if f[0] == 0 {
                break;
            }
        }
    }

    let k = get_kernel(&FINAL_KERNEL, ctx, FINALIZE_SRC, "canny_finalize")?;
    k.launch_builder(stream)
        .arg(&d_map)
        .arg(dst)
        .arg(&w)
        .arg(&h)
        .launch_cfg(cfg)
        .map_err(|e| CudaCannyError::Cuda(e.to_string()))
}
