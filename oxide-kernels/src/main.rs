//! Rust-authored resize kernels for kornia-rs, compiled to PTX by
//! NVlabs' cuda-oxide.
//!
//! Three kernels are provided, each a twin of the corresponding NVRTC kernel
//! in `crates/kornia-imgproc/src/cuda/resize.rs`.  The arithmetic is
//! transcribed expression for expression so that kornia's parity tests can
//! assert CPU == GPU with `assert_eq!` on raw bytes, not a tolerance.
//!
//! # Kernels
//!
//! * [`resize_bilinear_oxide_3c`] — bilinear interpolation, raw-pointer
//!   variant.  Parameter list identical to `resize_bilinear_downscale_3c` so
//!   the same cudarc launch builder drives either.
//! * [`resize_bilinear_oxide_slice_3c`] — same arithmetic through slice
//!   parameters (`&[f32]` / `DisjointSlice<f32>`).  Kept as a measured negative
//!   result: slice parameters lose `ld.global.nc` and add bounds-check branches,
//!   costing ~12% on upscale.
//! * [`resize_nearest_oxide_3c`] — nearest-neighbour interpolation, raw-pointer
//!   variant.  Parameter list identical to `resize_nearest_downscale_3c`.
//!
//! # Build
//!
//! ```sh
//! export PATH=/usr/local/cuda/bin:$PATH
//! cargo oxide build resize_oxide --arch sm_75 --no-fmad
//! # copy emitted PTX to the kornia crate:
//! cp resize_bilinear_oxide.ptx \
//!    ../crates/kornia-imgproc/src/cuda/resize_bilinear_oxide_sm75.ptx
//! cp resize_nearest_oxide.ptx \
//!    ../crates/kornia-imgproc/src/cuda/resize_nearest_oxide_sm75.ptx
//! ```
//!
//! Gate checks after copy (both must print 0):
//! ```sh
//! grep -cE 'fma\.rn\.f32|mad\.f32' resize_bilinear_oxide.ptx
//! grep -c '__nv_' resize_bilinear_oxide.ptx
//! ```
//!
//! # Float-contraction contract
//!
//! kornia sets NVRTC `fmad: Some(false)` so `ax * x + bx` rounds twice, like
//! the CPU LUT.  Rust never contracts float operations implicitly, so the kernels
//! below already emit `mul.f32` + `add.f32` — the byte-exact property is the
//! Rust default, not a flag.  When building with cuda-oxide main (≥ v0.2),
//! pass `--no-fmad` explicitly to keep that guarantee: v0.2 defaults to FMA
//! contraction to match nvcc `--fmad=true`.

use cuda_device::{DisjointSlice, kernel, thread};

// ── Bilinear — raw pointers ───────────────────────────────────────────────────

/// Byte-exact twin of `resize_bilinear_downscale_3c` (NVRTC).
///
/// Parameter order and types match the CUDA C kernel one for one, so the same
/// cudarc launch builder drives either arm in the benchmark.
///
/// # Safety
///
/// `src` must point to at least `src_h * src_w * 3` initialised `f32` values.
/// `dst` must point to at least `dst_h * dst_w * 3` writable `f32` slots.
/// All dimensions must be non-zero.  kornia's launcher validates both before
/// launching.
#[kernel]
pub unsafe fn resize_bilinear_oxide_3c(
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
    // rounds twice just like the CPU LUT's `a * x + b`.  `f32::min`/`max` lower
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

    // `as u32` is saturating on Rust; the clamps above keep both values in
    // range so the value matches the CUDA twin's `(unsigned int)sx`.
    // `Ord::min` on integers routes through drop glue the April snapshot cannot
    // emit, so explicit comparisons are used throughout.
    let x0 = sx as u32;
    let y0 = sy as u32;
    let sw1 = src_w - 1;
    let sh1 = src_h - 1;
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

    // All twelve source loads are issued before the first store.  Raw pointers
    // carry no `noalias`, so with stores interleaved the compiler cannot hoist a
    // later channel's loads above an earlier channel's store — three serialised
    // memory-latency round trips instead of one.  Load hoisting restores parity
    // with the CUDA twin, which gets the batching for free from `__restrict__`.
    // Measured on Orin at 2x upscale: 0.384 ms interleaved vs 0.197 ms hoisted.
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

// ── Bilinear — slice parameters ───────────────────────────────────────────────

/// Same bilinear arithmetic through slice parameters.
///
/// Kept as a measured negative result: `&[f32]` / `DisjointSlice<f32>` carry
/// `readonly`/`noalias` but the NVPTX backend does not infer `ld.global.nc`
/// from those attributes, address-space inference is lost, and bounds-check
/// branches are added.  Net cost: ~12% on upscale cases.
#[kernel]
pub fn resize_bilinear_oxide_slice_3c(
    src: &[f32],
    mut dst: DisjointSlice<f32>,
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

    let hix = (src_w - 1) as f32;
    let hiy = (src_h - 1) as f32;
    let cx = ax * dst_x as f32 + bx;
    let cy = ay * dst_y as f32 + by;
    let cx = if cx < hix { cx } else { hix };
    let cy = if cy < hiy { cy } else { hiy };
    let sx = if cx > 0.0 { cx } else { 0.0 };
    let sy = if cy > 0.0 { cy } else { 0.0 };

    let x0 = sx as u32;
    let y0 = sy as u32;
    let sw1 = src_w - 1;
    let sh1 = src_h - 1;
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

    let v0 = w00 * src[b00] + w10 * src[b10] + w01 * src[b01] + w11 * src[b11];
    let v1 = w00 * src[b00 + 1] + w10 * src[b10 + 1] + w01 * src[b01 + 1] + w11 * src[b11 + 1];
    let v2 = w00 * src[b00 + 2] + w10 * src[b10 + 2] + w01 * src[b01 + 2] + w11 * src[b11 + 2];
    unsafe {
        *dst.get_unchecked_mut(out) = v0;
        *dst.get_unchecked_mut(out + 1) = v1;
        *dst.get_unchecked_mut(out + 2) = v2;
    }
}

// ── Nearest-neighbour — raw pointers ─────────────────────────────────────────

/// Byte-exact twin of `resize_nearest_downscale_3c` (NVRTC).
///
/// Coordinate computation: `(ax * dst_x + bx) + 0.5`, truncated to `u32`,
/// clamped to `[0, src_w - 1]`.  The `+ 0.5` then truncation is round-half-up,
/// which equals the CPU's `round()` for the non-negative coordinates produced
/// here — including exact .5 ties, because both sides compute the identical
/// f32 value.
///
/// # Safety
///
/// Same invariants as [`resize_bilinear_oxide_3c`].
#[kernel]
pub unsafe fn resize_nearest_oxide_3c(
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
    // `round()`.  Clamped to `[0, src_dim - 1]` with explicit comparisons for
    // the same reason as the bilinear kernel — `Ord::min` on integers routes
    // through drop glue the backend cannot emit.
    let raw_x = ax * dst_x as f32 + bx + 0.5;
    let raw_y = ay * dst_y as f32 + by + 0.5;
    let xi = {
        let v = raw_x as u32;
        if v < src_w { v } else { src_w - 1 }
    };
    let yi = {
        let v = raw_y as u32;
        if v < src_h { v } else { src_h - 1 }
    };

    let src_base = ((yi * src_w + xi) * 3) as usize;
    let out = ((dst_y * dst_w + dst_x) * 3) as usize;

    // Three contiguous reads — hoist all before the stores (same noalias
    // reasoning as bilinear, though the penalty is smaller since there are only
    // three loads total).
    unsafe {
        let p0 = *src.add(src_base);
        let p1 = *src.add(src_base + 1);
        let p2 = *src.add(src_base + 2);
        *dst.add(out) = p0;
        *dst.add(out + 1) = p1;
        *dst.add(out + 2) = p2;
    }
}

fn main() {
    println!("resize_oxide: PTX-only crate; see the emitted .ptx files");
}
