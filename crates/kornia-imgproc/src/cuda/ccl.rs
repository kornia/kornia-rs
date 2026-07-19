//! CUDA connected-component labeling — label-identical to the CPU
//! union-find in `connected_components.rs`.
//!
//! Playne–Stephenson-style label equivalence: labels start as each
//! foreground pixel's own linear index and `atomicMin` merging + pointer
//! jumping iterate to a fixpoint where every pixel holds its component's
//! minimum linear index — the same canonical labeling the CPU's min-index
//! union-find produces. A device compaction then renumbers roots in index
//! order (== raster order of each component's first pixel, cv2-SAUF
//! numbering) and returns the label count.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::try_compile_with_l1;

/// Error type for the CUDA CCL launcher.
#[derive(Debug, thiserror::Error)]
pub enum CudaCclError {
    /// CUDA driver / compile / launch error.
    #[error("CUDA ccl error: {0}")]
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

fn check_slice(what: &'static str, got: usize, need: usize) -> Result<(), CudaCclError> {
    if got < need {
        return Err(CudaCclError::SliceTooSmall { what, got, need });
    }
    Ok(())
}

const BG: &str = "0x7FFFFFFF"; // background sentinel (i32 max)

static CCL_SRC_TMPL: &str = r#"
#define BG __BG__

// Row-run initialization: every foreground pixel starts with the index of
// its horizontal run's FIRST pixel (one thread per row, sequential scan).
// Horizontal merging is thereby already done — the fixpoint only has to
// merge vertically/diagonally, which cuts iterations dramatically. The
// run start is each run's min index, so the fixpoint target (component
// min index) is unchanged.
extern "C" __global__ void ccl_init(
    const unsigned char* __restrict__ src,
    int* __restrict__                 label,
    int w, int h
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= h) return;
    int run = -1;
    for (int x = 0; x < w; ++x) {
        int i = y * w + x;
        if (__ldg(&src[i])) {
            if (run < 0) run = i;
            label[i] = run;
        } else {
            label[i] = BG;
            run = -1;
        }
    }
}

__device__ __forceinline__ int find_root(const int* label, int i) {
    int l = label[i];
    while (l != label[l]) l = label[l];
    return l;
}

// One merge step: every foreground pixel pulls the minimum root among its
// connected neighbors onto its own root via atomicMin. Iterated with
// compression until stable; the fixpoint is the min-index labeling.
extern "C" __global__ void ccl_merge(
    const unsigned char* __restrict__ src,
    int* __restrict__                 label,
    unsigned int* __restrict__        changed,
    int w, int h, int eight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    if (!__ldg(&src[i])) return;

    int best = label[i];
    // All 4/8 neighbors (symmetric — atomicMin makes direction irrelevant).
    if (x > 0 && __ldg(&src[i - 1])) best = min(best, label[i - 1]);
    if (x + 1 < w && __ldg(&src[i + 1])) best = min(best, label[i + 1]);
    if (y > 0 && __ldg(&src[i - w])) best = min(best, label[i - w]);
    if (y + 1 < h && __ldg(&src[i + w])) best = min(best, label[i + w]);
    if (eight) {
        if (x > 0 && y > 0 && __ldg(&src[i - w - 1])) best = min(best, label[i - w - 1]);
        if (x + 1 < w && y > 0 && __ldg(&src[i - w + 1])) best = min(best, label[i - w + 1]);
        if (x > 0 && y + 1 < h && __ldg(&src[i + w - 1])) best = min(best, label[i + w - 1]);
        if (x + 1 < w && y + 1 < h && __ldg(&src[i + w + 1])) best = min(best, label[i + w + 1]);
    }
    int r = find_root(label, i);
    if (best < label[r]) {
        atomicMin(&label[r], best);
        atomicAdd(changed, 1u);
    }
}

// Pointer jumping: label[i] = root(i).
extern "C" __global__ void ccl_compress(
    const unsigned char* __restrict__ src,
    int* __restrict__                 label,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!__ldg(&src[i])) return;
    label[i] = find_root(label, i);
}

// Compaction phase 1: per-block (512 elems, 256 threads) exclusive scan
// of root flags in shared memory (Hillis–Steele).
extern "C" __global__ void ccl_scan_partial(
    const int* __restrict__ label,
    int* __restrict__       pos,
    int* __restrict__       block_sums,
    int n
) {
    __shared__ int buf[512];
    int b = blockIdx.x;
    int start = b * 512;
    int t = threadIdx.x;
    for (int k = 0; k < 2; ++k) {
        int j = start + t + k * 256;
        buf[t + k * 256] = (j < n && label[j] == j) ? 1 : 0;
    }
    __syncthreads();
    for (int off = 1; off < 512; off <<= 1) {
        int v0 = (t >= off) ? buf[t - off] : 0;
        int v1 = (t + 256 >= off) ? buf[t + 256 - off] : 0;
        __syncthreads();
        buf[t] += v0;
        buf[t + 256] += v1;
        __syncthreads();
    }
    // Inclusive -> exclusive on write-out.
    for (int k = 0; k < 2; ++k) {
        int idx = t + k * 256;
        int j = start + idx;
        if (j < n) {
            pos[j] = buf[idx] - ((label[j] == j) ? 1 : 0);
        }
    }
    if (t == 0) block_sums[b] = buf[511];
}

// Compaction phase 2: exclusive scan of the (small) block sums + total.
extern "C" __global__ void ccl_scan_offsets(
    int* __restrict__ block_sums,
    int* __restrict__ total,
    int nblocks
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int acc = 0;
    for (int b = 0; b < nblocks; ++b) {
        int v = block_sums[b];
        block_sums[b] = acc;
        acc += v;
    }
    *total = acc;
}

// Compaction phase 3: final labels — 0 for background, compact root rank
// + 1 for foreground (rank of the component's min-index pixel).
extern "C" __global__ void ccl_relabel(
    const int* __restrict__ label,
    const int* __restrict__ pos,
    const int* __restrict__ block_sums,
    int* __restrict__       out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int l = label[i];
    if (l == BG) {
        out[i] = 0;
    } else {
        out[i] = block_sums[l / 512] + pos[l] + 1;
    }
}
"#;

struct Kernels {
    init: CudaKernel,
    merge: CudaKernel,
    compress: CudaKernel,
    scan_partial: CudaKernel,
    scan_offsets: CudaKernel,
    relabel: CudaKernel,
}

static KERNELS: OnceLock<Result<Kernels, String>> = OnceLock::new();

fn get_kernels(ctx: &Arc<CudaContext>) -> Result<&'static Kernels, CudaCclError> {
    KERNELS
        .get_or_init(|| {
            let src = CCL_SRC_TMPL.replace("__BG__", BG);
            Ok(Kernels {
                init: try_compile_with_l1(ctx, &src, "ccl_init")?,
                merge: try_compile_with_l1(ctx, &src, "ccl_merge")?,
                compress: try_compile_with_l1(ctx, &src, "ccl_compress")?,
                scan_partial: try_compile_with_l1(ctx, &src, "ccl_scan_partial")?,
                scan_offsets: try_compile_with_l1(ctx, &src, "ccl_scan_offsets")?,
                relabel: try_compile_with_l1(ctx, &src, "ccl_relabel")?,
            })
        })
        .as_ref()
        .map_err(|e| CudaCclError::Cuda(e.clone()))
}

/// Full device CCL: init → (merge + compress) fixpoint → compaction.
/// Returns the label count (components + background), matching the CPU
/// `connected_components` return; output labels are identical to the
/// CPU's. Synchronizes the stream to read the convergence flag and the
/// final count.
pub fn launch_connected_components(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    out: &mut CudaSlice<i32>,
    width: usize,
    height: usize,
    eight: bool,
) -> Result<i32, CudaCclError> {
    super::check_geometry(
        width as u32,
        height as u32,
        width as u32,
        height as u32,
        None,
    )
    .map_err(CudaCclError::Cuda)?;
    let n = width * height;
    check_slice("src", src.len(), n)?;
    check_slice("out", out.len(), n)?;
    let w = i32::try_from(width).map_err(|_| CudaCclError::Cuda("width exceeds i32".into()))?;
    let h = i32::try_from(height).map_err(|_| CudaCclError::Cuda("height exceeds i32".into()))?;
    let n_i32 = i32::try_from(n).map_err(|_| CudaCclError::Cuda("size exceeds i32".into()))?;
    let err = |e: cudarc::driver::DriverError| CudaCclError::Cuda(e.to_string());
    let k = get_kernels(ctx)?;

    // SAFETY: label/pos interiors are fully written by ccl_init /
    // ccl_scan_partial before any read.
    let mut label = unsafe { stream.alloc::<i32>(n) }.map_err(err)?;
    let mut pos = unsafe { stream.alloc::<i32>(n) }.map_err(err)?;
    let nblocks = n.div_ceil(512);
    let mut block_sums = unsafe { stream.alloc::<i32>(nblocks) }.map_err(err)?;
    let mut d_total = stream.alloc_zeros::<i32>(1).map_err(err)?;
    let mut d_changed = stream.alloc_zeros::<u32>(1).map_err(err)?;

    let cfg1 = super::make_config(n as u32, 1, Some((256, 1)));
    let cfg2 = super::make_config(w as u32, h as u32, None);
    fn launch_err(e: impl std::fmt::Display) -> CudaCclError {
        CudaCclError::Cuda(e.to_string())
    }

    k.init
        .launch_builder(stream)
        .arg(src)
        .arg(&mut label)
        .arg(&w)
        .arg(&h)
        .launch_cfg(super::make_config(h as u32, 1, Some((128, 1))))
        .map_err(launch_err)?;

    let eight_i = i32::from(eight);
    // Merge + compress to fixpoint, two sweeps per host sync. (A
    // geometrically growing batch was tried and regressed — post-
    // convergence merge sweeps are expensive on dense content.)
    for _ in 0..n.max(1) {
        stream.memset_zeros(&mut d_changed).map_err(err)?;
        for _ in 0..2 {
            k.merge
                .launch_builder(stream)
                .arg(src)
                .arg(&mut label)
                .arg(&mut d_changed)
                .arg(&w)
                .arg(&h)
                .arg(&eight_i)
                .launch_cfg(cfg2)
                .map_err(launch_err)?;
            k.compress
                .launch_builder(stream)
                .arg(src)
                .arg(&mut label)
                .arg(&n_i32)
                .launch_cfg(cfg1)
                .map_err(launch_err)?;
        }
        let f: Vec<u32> = stream.clone_dtoh(&d_changed).map_err(err)?;
        stream.synchronize().map_err(err)?;
        if f[0] == 0 {
            break;
        }
    }

    // Compaction: rank roots in index order (== cv2-SAUF numbering).
    let nblocks_i32 =
        i32::try_from(nblocks).map_err(|_| CudaCclError::Cuda("nblocks exceeds i32".into()))?;
    k.scan_partial
        .launch_builder(stream)
        .arg(&label)
        .arg(&mut pos)
        .arg(&mut block_sums)
        .arg(&n_i32)
        .launch_cfg(cudarc::driver::LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (nblocks as u32, 1, 1),
            shared_mem_bytes: 0,
        })
        .map_err(launch_err)?;
    k.scan_offsets
        .launch_builder(stream)
        .arg(&mut block_sums)
        .arg(&mut d_total)
        .arg(&nblocks_i32)
        .launch_cfg(cudarc::driver::LaunchConfig {
            block_dim: (1, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        })
        .map_err(launch_err)?;
    k.relabel
        .launch_builder(stream)
        .arg(&label)
        .arg(&pos)
        .arg(&block_sums)
        .arg(out)
        .arg(&n_i32)
        .launch_cfg(cfg1)
        .map_err(launch_err)?;

    let total: Vec<i32> = stream.clone_dtoh(&d_total).map_err(err)?;
    stream.synchronize().map_err(err)?;
    Ok(total[0] + 1)
}
