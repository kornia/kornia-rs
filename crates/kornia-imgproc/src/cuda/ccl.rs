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

super::define_cuda_error!(
    /// Error type for the CUDA CCL launcher.
    CudaCclError,
    "CUDA ccl error: {0}"
);

const BG: &str = "0x7FFFFFFF"; // background sentinel (i32 max)

static CCL_SRC_TMPL: &str = r#"
#define BG __BG__

// Row-run initialization, block-parallel: every foreground pixel starts
// with the index of its horizontal run's first pixel WITHIN ITS 256-WIDE
// BLOCK. Per-warp ballots give 32-pixel foreground masks; a shared-mem
// max-scan over the 8 warp summaries extends run starts across warp
// boundaries, so runs only break every 256 pixels (the previous
// 1-thread-per-row serial form was fully uncoalesced and cost 1.4ms of
// the 2.9ms total at 1080p; a warp-only version broke runs every 32px
// and pushed 0.27ms of stitch unions into ccl_union). Block-boundary
// runs are stitched by one horizontal union per boundary in ccl_union.
extern "C" __global__ void ccl_init(
    const unsigned char* __restrict__ src,
    int* __restrict__                 label,
    int w, int h
) {
    __shared__ unsigned warp_mask[8];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    int i = y * w + x;
    // All 32 lanes participate in the ballot (grid x is warp-padded);
    // out-of-range lanes report background.
    bool fg = (x < w) && (__ldg(&src[i]) != 0);
    unsigned mask = __ballot_sync(0xFFFFFFFFu, fg);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_mask[warp] = mask;
    __syncthreads();
    if (x >= w) return;
    if (!fg) { label[i] = BG; return; }
    unsigned zeros_below = ~mask & ((1u << lane) - 1u);
    if (zeros_below) {
        // Run starts inside this warp: after the highest zero below us.
        label[i] = i - lane + (32 - __clz(zeros_below));
        return;
    }
    // Warp prefix is all-foreground: walk earlier warps' masks (max 7
    // shared-mem reads) to the nearest warp containing a zero. Its
    // highest zero bounds our run; 32 - clz(~m) lands on the pixel after
    // it (== the next warp's base when the zero is at lane 31). No
    // earlier warp with a zero -> the run start is the block's first
    // pixel, and ccl_union stitches across the block boundary.
    int block_base = y * w + blockIdx.x * blockDim.x;
    int start = block_base;
    for (int p = warp - 1; p >= 0; --p) {
        unsigned m = warp_mask[p];
        if (m != 0xFFFFFFFFu) {
            start = block_base + (p << 5) + (32 - __clz(~m));
            break;
        }
    }
    label[i] = start;
}

// Find with path halving. The shortcut write must be atomicMin: a plain
// store could overwrite (raise) a link a concurrent union just installed
// with atomicMin, losing that union. atomicMin keeps every entry
// monotonically decreasing, so any interleaving converges to the same
// min-index fixpoint.
__device__ __forceinline__ int find_root(int* label, int i) {
    int l = label[i];
    while (l != label[l]) {
        int p = label[l];
        int pp = label[p];
        if (pp < p) atomicMin(&label[l], pp);
        l = pp;
    }
    return l;
}

// Lock-free union (Komura): link the larger root under the smaller with
// atomicMin, retrying with the displaced value on contention. After ONE
// pass over all edges the parent forest is complete — no global
// iteration needed; a single compress pass then resolves every pixel to
// its component's min index.
__device__ void union_labels(int* label, int a, int b) {
    a = find_root(label, a);
    b = find_root(label, b);
    while (a != b) {
        int lo = min(a, b);
        int hi = max(a, b);
        int old = atomicMin(&label[hi], lo);
        if (old == hi) return;   // linked
        a = lo;                  // retry: displaced value keeps the chain
        b = old;
    }
}

// One pass: every foreground pixel unions with its already-scanned
// neighbors (up row; left is free via the row-run init).
extern "C" __global__ void ccl_union(
    const unsigned char* __restrict__ src,
    int* __restrict__                 label,
    int w, int h, int eight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    if (!__ldg(&src[i])) return;
    // Stitch runs split at 256-pixel block boundaries by ccl_init (every
    // row, including y == 0): one union per boundary inside a run.
    if ((x & 255) == 0 && x > 0 && __ldg(&src[i - 1])) {
        union_labels(label, i, i - 1);
    }
    if (y == 0) return;
    int up = i - w;
    // Skip unions already implied by horizontal run chains: (i,up) is
    // implied when (i-1) and (up-1) are both foreground. Cuts unions from
    // O(area) to O(run overlaps) on blob content.
    bool left = x > 0 && __ldg(&src[i - 1]);
    bool upl = x > 0 && __ldg(&src[up - 1]);
    bool upc = __ldg(&src[up]) != 0;
    if (upc && !(left && upl)) union_labels(label, i, up);
    if (eight) {
        // Diagonals: (i,up-1) implied by (i-1,up-1)+run or (i,up)+run.
        if (upl && !left && !upc) union_labels(label, i, up - 1);
        // (i,up+1) implied by (i,up)+run of up row.
        if (x + 1 < w && __ldg(&src[up + 1]) && !upc) union_labels(label, i, up + 1);
    }
}

// Compaction phase 1: per-block (512 elems, 256 threads) exclusive scan
// of root flags. Each thread owns 2 adjacent elements (vector int2
// load); warp-shuffle inclusive scan of the pair sums + one tiny shared
// pass over the 8 warp totals replaces the 9-step Hillis–Steele
// ping-pong (18 __syncthreads -> 2). Block-local ranks fit u16 (<= 512),
// halving the pos-array traffic here and in ccl_relabel.
extern "C" __global__ void ccl_scan_partial(
    const int* __restrict__       label,
    unsigned short* __restrict__  pos,
    int* __restrict__             block_sums,
    int n
) {
    __shared__ int warp_tot[8];
    int t = threadIdx.x;
    int j0 = blockIdx.x * 512 + t * 2;
    int f0 = 0, f1 = 0;
    if (j0 + 1 < n) {
        int2 l = *(const int2*)(label + j0);
        f0 = (l.x == j0) ? 1 : 0;
        f1 = (l.y == j0 + 1) ? 1 : 0;
    } else if (j0 < n) {
        f0 = (label[j0] == j0) ? 1 : 0;
    }
    int s = f0 + f1;
    int lane = t & 31;
    int warp = t >> 5;
    int inc = s;
    for (int off = 1; off < 32; off <<= 1) {
        int v = __shfl_up_sync(0xFFFFFFFFu, inc, off);
        if (lane >= off) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int v = (lane < 8) ? warp_tot[lane] : 0;
        for (int off = 1; off < 8; off <<= 1) {
            int u = __shfl_up_sync(0xFFFFFFFFu, v, off);
            if (lane >= off) v += u;
        }
        if (lane < 8) warp_tot[lane] = v;
    }
    __syncthreads();
    // Exclusive prefix of this thread's pair within the block.
    int pre = (inc - s) + (warp ? warp_tot[warp - 1] : 0);
    if (j0 < n)     pos[j0]     = (unsigned short)pre;
    if (j0 + 1 < n) pos[j0 + 1] = (unsigned short)(pre + f0);
    if (t == 255) block_sums[blockIdx.x] = pre + s;
}

// Compaction phase 2: exclusive scan of the block sums + total. Single
// block, 512 threads, Hillis–Steele over 512-wide chunks with a running
// carry (the previous single-THREAD loop cost 0.2ms at 1080p's ~4k
// block sums; this is ~20us).
extern "C" __global__ void ccl_scan_offsets(
    int* __restrict__ block_sums,
    int* __restrict__ total,
    int nblocks
) {
    __shared__ int buf[512];
    __shared__ int carry;
    int t = threadIdx.x;
    if (t == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < nblocks; base += 512) {
        int j = base + t;
        int v = (j < nblocks) ? block_sums[j] : 0;
        buf[t] = v;
        __syncthreads();
        for (int off = 1; off < 512; off <<= 1) {
            int u = (t >= off) ? buf[t - off] : 0;
            __syncthreads();
            buf[t] += u;
            __syncthreads();
        }
        if (j < nblocks) block_sums[j] = carry + buf[t] - v; // exclusive
        __syncthreads();
        if (t == 0) carry += buf[511];
        __syncthreads();
    }
    if (t == 0) *total = carry;
}

// Compaction phase 3: final labels — 0 for background, compact root rank
// + 1 for foreground (rank of the component's min-index pixel).
extern "C" __global__ void ccl_relabel(
    int* __restrict__                  label,
    const unsigned short* __restrict__ pos,
    const int* __restrict__            block_sums,
    int* __restrict__                  out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int l = label[i];
    if (l == BG) {
        out[i] = 0;
    } else {
        l = find_root(label, i);
        out[i] = block_sums[l / 512] + pos[l] + 1;
    }
}
"#;

struct Kernels {
    init: CudaKernel,
    union_k: CudaKernel,
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
                union_k: try_compile_with_l1(ctx, &src, "ccl_union")?,
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
    CudaCclError::check_slice("src", src.len(), n)?;
    CudaCclError::check_slice("out", out.len(), n)?;
    let w = i32::try_from(width).map_err(|_| CudaCclError::Cuda("width exceeds i32".into()))?;
    let h = i32::try_from(height).map_err(|_| CudaCclError::Cuda("height exceeds i32".into()))?;
    let n_i32 = i32::try_from(n).map_err(|_| CudaCclError::Cuda("size exceeds i32".into()))?;
    let k = get_kernels(ctx)?;

    // SAFETY: label/pos interiors are fully written by ccl_init /
    // ccl_scan_partial before any read.
    let mut label = unsafe { stream.alloc::<i32>(n) }?;
    let mut pos = unsafe { stream.alloc::<u16>(n) }?;
    let nblocks = n.div_ceil(512);
    let mut block_sums = unsafe { stream.alloc::<i32>(nblocks) }?;
    let mut d_total = stream.alloc_zeros::<i32>(1)?;

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
        .launch_cfg(cudarc::driver::LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((w as u32).div_ceil(256), h as u32, 1),
            shared_mem_bytes: 0,
        })
        .map_err(launch_err)?;

    let eight_i = i32::from(eight);
    // One union pass (lock-free) + one compress pass — no host iteration.
    k.union_k
        .launch_builder(stream)
        .arg(src)
        .arg(&mut label)
        .arg(&w)
        .arg(&h)
        .arg(&eight_i)
        .launch_cfg(cfg2)
        .map_err(launch_err)?;
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
            block_dim: (512, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        })
        .map_err(launch_err)?;
    k.relabel
        .launch_builder(stream)
        .arg(&mut label)
        .arg(&pos)
        .arg(&block_sums)
        .arg(out)
        .arg(&n_i32)
        .launch_cfg(cfg1)
        .map_err(launch_err)?;

    let total: Vec<i32> = stream.clone_dtoh(&d_total)?;
    stream.synchronize()?;
    Ok(total[0] + 1)
}
