//! Resize kernels: bilinear and nearest-neighbour interpolation.
//!
//! Each is byte-exact with its NVRTC twin in `crates/kornia-imgproc/src/cuda/resize.rs`.

use cuda_device::{kernel, thread};

// ── Bilinear — raw pointers ──────────────────────────────────────────────

/// Byte-exact twin of `resize_bilinear_downscale_3c` (NVRTC).
///
/// Parameter order and types match the CUDA C kernel one for one, so the same
/// cudarc launch builder drives either arm.
///
/// # Safety
///
/// `src` must point to at least `src_h * src_w * 3` initialised `f32` values.
/// `dst` must point to at least `dst_h * dst_w * 3` writable `f32` slots.
/// All dimensions must be non-zero. The launcher validates both before launching.
#[kernel]
pub unsafe fn resize_bilinear_downscale_3c(
    src: *const f32,
    dst: *mut f32,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    ax: f32,
    bx: f32,
    ay: f32,
    by: f32,
) {
    let dst_x = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    let dst_y = thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y();
    if dst_x >= dst_w || dst_y >= dst_h {
        return;
    }

    // Two separate multiplications + additions — never fused — so the coordinate
    // rounds twice just like the CPU LUT's `a * x + b`. `f32::min`/`max` lower
    // to `minimumnum`/`maximumnum`, which the cuda-oxide backend cannot verify;
    // explicit comparisons produce the identical value for the non-NaN
    // coordinates this kernel sees.
    let hix = (src_w - 1) as f32;
    let hiy = (src_h - 1) as f32;
    let cx = ax * dst_x as f32 + bx;
    let cy = ay * dst_y as f32 + by;
    let cx = if cx < hix { cx } else { hix };
    let cy = if cy < hiy { cy } else { hiy };
    let sx = if cx > 0.0 { cx } else { 0.0 };
    let sy = if cy > 0.0 { cy } else { 0.0 };

    let sw1 = src_w - 1;
    let sh1 = src_h - 1;
    let x0 = (sx as u32).min(sw1);
    let y0 = (sy as u32).min(sh1);
    let x1 = if x0 + 1 < sw1 { x0 + 1 } else { sw1 };
    let y1 = if y0 + 1 < sh1 { y0 + 1 } else { sh1 };

    let fx = sx - x0 as f32;
    let fy = sy - y0 as f32;
    let w00 = (1.0 - fy) * (1.0 - fx);
    let w10 = (1.0 - fy) * fx;
    let w01 = fy * (1.0 - fx);
    let w11 = fy * fx;

    let b00 = ((y0 * src_w + x0) * 3) as usize;
    let b10 = ((y0 * src_w + x1) * 3) as usize;
    let b01 = ((y1 * src_w + x0) * 3) as usize;
    let b11 = ((y1 * src_w + x1) * 3) as usize;
    let out = ((dst_y * dst_w + dst_x) * 3) as usize;

    // All twelve source loads are issued before the first store. Raw pointers
    // carry no `noalias`, so with stores interleaved the compiler cannot hoist a
    // later channel's loads above an earlier channel's store — three serialised
    // memory-latency round trips instead of one. Load hoisting restores parity
    // with the CUDA twin, which gets the batching for free from `__restrict__`.
    // SAFETY: src and dst are valid pointers to device memory with at least the
    // required sizes (validated by launcher). Indices (b00, b10, etc. and out) are
    // in bounds: dst coordinates are checked at function entry, source coordinates
    // are clamped to [0, src_w-1] and [0, src_h-1], and the 3-channel stride keeps
    // all accesses within the allocated buffer.
    unsafe {
        let s00 = *src.add(b00);
        let s10 = *src.add(b10);
        let s01 = *src.add(b01);
        let s11 = *src.add(b11);
        let t00 = *src.add(b00 + 1);
        let t10 = *src.add(b10 + 1);
        let t01 = *src.add(b01 + 1);
        let t11 = *src.add(b11 + 1);
        let u00 = *src.add(b00 + 2);
        let u10 = *src.add(b10 + 2);
        let u01 = *src.add(b01 + 2);
        let u11 = *src.add(b11 + 2);

        *dst.add(out) = w00 * s00 + w10 * s10 + w01 * s01 + w11 * s11;
        *dst.add(out + 1) = w00 * t00 + w10 * t10 + w01 * t01 + w11 * t11;
        *dst.add(out + 2) = w00 * u00 + w10 * u10 + w01 * u01 + w11 * u11;
    }
}

// ── Nearest-neighbour — raw pointers ──────────────────────────────────────

/// Byte-exact twin of `resize_nearest_downscale_3c` (NVRTC).
///
/// Coordinate computation: `(ax * dst_x + bx) + 0.5`, truncated to `u32`,
/// clamped to `[0, src_w - 1]`. The `+ 0.5` then truncation is round-half-up,
/// which equals the CPU's `round()` for the non-negative coordinates produced
/// here.
///
/// # Safety
///
/// Same invariants as [`resize_bilinear_downscale_3c`].
#[kernel]
pub unsafe fn resize_nearest_downscale_3c(
    src: *const f32,
    dst: *mut f32,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    ax: f32,
    bx: f32,
    ay: f32,
    by: f32,
) {
    let dst_x = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    let dst_y = thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y();
    if dst_x >= dst_w || dst_y >= dst_h {
        return;
    }

    // `+ 0.5` before truncation = round-half-up, matching the CPU path's
    // `round()` on non-negative coordinates.
    let sx = ax * dst_x as f32 + bx + 0.5;
    let sy = ay * dst_y as f32 + by + 0.5;

    let sw1 = src_w - 1;
    let sh1 = src_h - 1;
    let x = (sx as u32).min(sw1);
    let y = (sy as u32).min(sh1);

    let src_idx = ((y * src_w + x) * 3) as usize;
    let dst_idx = ((dst_y * dst_w + dst_x) * 3) as usize;

    // SAFETY: src and dst are valid pointers to device memory with at least the
    // required sizes (validated by launcher). Indices are in bounds: dst coordinates
    // are checked at function entry, source coordinates are clamped to [0, src_w-1]
    // and [0, src_h-1], and the 3-channel stride keeps all accesses within bounds.
    unsafe {
        *dst.add(dst_idx) = *src.add(src_idx);
        *dst.add(dst_idx + 1) = *src.add(src_idx + 1);
        *dst.add(dst_idx + 2) = *src.add(src_idx + 2);
    }
}
