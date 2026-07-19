//! Experimental GPU-accelerated image processing kernels.
//!
//! Backed by native CUDA kernels compiled at runtime via NVRTC, using `cudarc`
//! for device memory and launches. The whole module is gated on the `cuda`
//! feature at its declaration in `lib.rs`.

/// Native CUDA downscale kernels using `__ldg` read-only cache.
pub mod resize;

/// Native CUDA u8 resize kernels (integer LUT-driven, byte-exact with the
/// CPU u8 fast paths).
pub mod filter;
pub mod resize_u8;

/// Native CUDA warp-affine kernels (bilinear and nearest-neighbor).
pub mod warp_affine;

/// Native CUDA u8 warp-affine kernel (Q16/Q10 fixed point, byte-exact with
/// the CPU `warp_affine_u8`).
pub mod warp_affine_u8;

/// Native CUDA warp-perspective kernels (homography, bilinear / nearest / bicubic / Lanczos-3).
pub mod warp_perspective;

/// Native CUDA u8 warp-perspective kernel (direct rational coords + Q10
/// fixed point, byte-exact with the CPU `warp_perspective_u8`).
pub mod warp_perspective_u8;

/// Native CUDA color-space conversion kernels.
pub mod color;

/// Bilateral-filter kernel (byte-exact vs the CPU path and cv2).
pub mod bilateral;

/// Canny edge-detection kernels (byte-exact vs the CPU path and cv2).
pub mod canny;

/// Connected-component labeling kernels (label-identical to the CPU
/// union-find and cv2's SAUF numbering).
pub mod ccl;

/// CLAHE LUT-build + blend kernels (byte-exact vs the CPU path and cv2).
pub mod clahe;
pub mod histogram;

/// Median-blur sorting-network kernels (byte-exact vs the CPU path, cv2
/// and VPI).
pub mod median;
/// Native CUDA u8 morphology kernels (dilate / erode, byte-exact with the
/// CPU ops).
pub mod morphology;
pub mod pyramid;

/// Residency-aware dispatch machinery shared by all device-capable ops.
pub(crate) mod dispatch;

/// FKL-style kernel fusion engine (register-flow stage composition).
pub mod fusion;

use std::sync::Arc;

use cudarc::driver::{CudaContext, LaunchConfig};
use kornia_tensor::CudaKernel;

/// Default thread block for the 2D geometry kernels: a full warp per output
/// row for coalesced writes, 8 rows per block.
pub(crate) const BLOCK_W: u32 = 32;
pub(crate) const BLOCK_H: u32 = 8;

/// Build the 2D launch configuration, clamping the default block to the image
/// so tiny outputs don't waste threads. `block_dim` components are assumed
/// non-zero — enforced by [`check_geometry`], which every launcher calls first.
pub(crate) fn make_config(
    dst_width: u32,
    dst_height: u32,
    block_dim: Option<(u32, u32)>,
) -> LaunchConfig {
    let (bw, bh) = block_dim.unwrap_or_else(|| (BLOCK_W.min(dst_width), BLOCK_H.min(dst_height)));
    LaunchConfig {
        block_dim: (bw, bh, 1),
        grid_dim: (dst_width.div_ceil(bw), dst_height.div_ceil(bh), 1),
        shared_mem_bytes: 0,
    }
}

/// Compile a kernel and set `CU_FUNC_CACHE_PREFER_L1` (none of the geometry
/// kernels use shared memory, so give the space to L1 for `__ldg` hit rate).
pub(crate) fn try_compile_with_l1(
    ctx: &Arc<CudaContext>,
    src: &str,
    fn_name: &str,
) -> Result<CudaKernel, String> {
    let k = CudaKernel::compile(ctx, src, fn_name)
        .map_err(|e| format!("failed to compile {fn_name}: {e}"))?;
    let _ = k.prefer_l1_cache();
    Ok(k)
}

/// Validate the launch geometry shared by every warp/resize launcher: all
/// image dimensions non-zero (zero dims would underflow the kernels' `-1u`
/// clamps) and, when a `block_dim` override is given, both components non-zero
/// (a zero component would divide-by-zero in [`make_config`]).
pub(crate) fn check_geometry(
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    block_dim: Option<(u32, u32)>,
) -> Result<(), String> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err("image dimensions must be non-zero".into());
    }
    if block_dim.is_some_and(|(bw, bh)| bw == 0 || bh == 0) {
        return Err("block_dim components must be non-zero".into());
    }
    Ok(())
}
