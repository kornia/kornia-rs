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
//! * [`resize_bicubic_oxide_3c`] — Keys cubic (a = -0.5) interpolation,
//!   raw-pointer variant.  Uses plain `*` and `+` throughout (no `f32::mul_add`;
//!   backend gap #4: FMA intrinsics trigger NVVM IR mode, no PTX emitted).
//!   Results differ from the NVRTC kernel by ~1 ULP; parity tests compare against
//!   a no-FMA CPU reference.
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

// ── Bicubic — raw pointers ────────────────────────────────────────────────────

/// Keys cubic (a = -0.5) resize, raw-pointer variant.
///
/// **Note on FMA**: `f32::mul_add` lowers to `llvm.fma.f32`, which causes the
/// c8b3103 cuda-oxide backend to switch to NVVM IR mode and skip PTX emission
/// (it requires `libnvvm` at the consumer side, which kornia does not wire up).
/// This kernel uses plain `*` and `+` throughout.  The Horner weight values and
/// accumulated outputs will differ in the last bit from the NVRTC kernel, which
/// uses `fmaf` explicitly.  Parity tests compare against CPU (not NVRTC).
///
/// This is backend gap #4: any `f32::mul_add` / `llvm.fma.f32` causes the
/// backend to emit NVVM IR instead of PTX.
///
/// Row-by-row load hoist: 12 values per row are loaded into registers before any
/// accumulation for that row.  This caps register pressure at ~35 while still
/// preventing the no-`noalias` serialisation that costs ~2× on upscale.
///
/// # Safety
///
/// Same invariants as [`resize_bilinear_oxide_3c`].
#[kernel]
pub unsafe fn resize_bicubic_oxide_3c(
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

    // Continuous source coordinate, clamped to [0, src_dim-1].
    // Explicit comparisons replace fmaxf/fminf (same reason as bilinear).
    let hix = (src_w - 1) as f32;
    let hiy = (src_h - 1) as f32;
    let cx = ax * dst_x as f32 + bx;
    let cy = ay * dst_y as f32 + by;
    let cx = if cx < hix { cx } else { hix };
    let cy = if cy < hiy { cy } else { hiy };
    let sx = if cx > 0.0 { cx } else { 0.0_f32 };
    let sy = if cy > 0.0 { cy } else { 0.0_f32 };

    // sx/sy ≥ 0 after clamping, so truncation == floor.  `floorf` is not
    // used: it calls libnvvm, which would add an `__nv_floorf` symbol and
    // break the self-contained PTX contract.
    let x0 = sx as i32;
    let y0 = sy as i32;
    let frac_x = sx - x0 as f32;
    let frac_y = sy - y0 as f32;

    // Horner-form Keys cubic weights (a = -0.5), plain mul+add.
    // wx[0] / wx[3]: ((-0.5*t + 2.5)*t - 4.0)*t + 2.0
    // wx[1] / wx[2]: ((1.5*t - 2.5)*t)*t + 1.0
    // f32::mul_add is NOT used: it triggers the backend's NVVM IR mode (gap #4).
    // Results differ in the last bit from the NVRTC fmaf path; parity is
    // verified against CPU, not NVRTC.
    let wx0 = {
        let t = 1.0_f32 + frac_x;
        ((-0.5_f32 * t + 2.5_f32) * t - 4.0_f32) * t + 2.0_f32
    };
    let wx1 = {
        let t = frac_x;
        ((1.5_f32 * t - 2.5_f32) * t) * t + 1.0_f32
    };
    let wx2 = {
        let t = 1.0_f32 - frac_x;
        ((1.5_f32 * t - 2.5_f32) * t) * t + 1.0_f32
    };
    let wx3 = {
        let t = 2.0_f32 - frac_x;
        ((-0.5_f32 * t + 2.5_f32) * t - 4.0_f32) * t + 2.0_f32
    };
    let wy0 = {
        let t = 1.0_f32 + frac_y;
        ((-0.5_f32 * t + 2.5_f32) * t - 4.0_f32) * t + 2.0_f32
    };
    let wy1 = {
        let t = frac_y;
        ((1.5_f32 * t - 2.5_f32) * t) * t + 1.0_f32
    };
    let wy2 = {
        let t = 1.0_f32 - frac_y;
        ((1.5_f32 * t - 2.5_f32) * t) * t + 1.0_f32
    };
    let wy3 = {
        let t = 2.0_f32 - frac_y;
        ((-0.5_f32 * t + 2.5_f32) * t - 4.0_f32) * t + 2.0_f32
    };

    // Clamped row bases.  y0 ≥ 0 (sx was clamped); y0-1 is the only case that
    // can reach -1.  `i32::max`/`min` share the Ord::min drop-glue issue —
    // explicit comparisons used throughout.
    let sw3 = src_w * 3;
    let hi_y = (src_h as i32) - 1;
    let hi_x = (src_w as i32) - 1;

    let ry0 = (if y0 - 1 < 0 { 0_i32 } else { y0 - 1 }) as u32 * sw3;
    let ry1 = y0 as u32 * sw3;
    let ry2 = (if y0 + 1 > hi_y { hi_y } else { y0 + 1 }) as u32 * sw3;
    let ry3 = (if y0 + 2 > hi_y { hi_y } else { y0 + 2 }) as u32 * sw3;

    let cx0 = (if x0 - 1 < 0 { 0_i32 } else { x0 - 1 }) as u32 * 3;
    let cx1 = x0 as u32 * 3;
    let cx2 = (if x0 + 1 > hi_x { hi_x } else { x0 + 1 }) as u32 * 3;
    let cx3 = (if x0 + 2 > hi_x { hi_x } else { x0 + 2 }) as u32 * 3;

    let out = ((dst_y * dst_w + dst_x) * 3) as usize;

    // Row-by-row load hoist.  12 src values (4 cols × 3 channels) are issued
    // per row before that row's fma accumulation begins.  All stores follow the
    // final row — no src read is ordered after a dst write.
    unsafe {
        let mut acc0 = 0.0_f32;
        let mut acc1 = 0.0_f32;
        let mut acc2 = 0.0_f32;

        // ── row 0, wy = wy0 ────────────────────────────────────────────────
        let b0 = (ry0 + cx0) as usize; let b1 = (ry0 + cx1) as usize;
        let b2 = (ry0 + cx2) as usize; let b3 = (ry0 + cx3) as usize;
        let v00 = *src.add(b0);   let v01 = *src.add(b0+1); let v02 = *src.add(b0+2);
        let v10 = *src.add(b1);   let v11 = *src.add(b1+1); let v12 = *src.add(b1+2);
        let v20 = *src.add(b2);   let v21 = *src.add(b2+1); let v22 = *src.add(b2+2);
        let v30 = *src.add(b3);   let v31 = *src.add(b3+1); let v32 = *src.add(b3+2);
        let w0 = wx0*wy0; let w1 = wx1*wy0; let w2 = wx2*wy0; let w3 = wx3*wy0;
        acc0 += w0*v00; acc1 += w0*v01; acc2 += w0*v02;
        acc0 += w1*v10; acc1 += w1*v11; acc2 += w1*v12;
        acc0 += w2*v20; acc1 += w2*v21; acc2 += w2*v22;
        acc0 += w3*v30; acc1 += w3*v31; acc2 += w3*v32;

        // ── row 1, wy = wy1 ────────────────────────────────────────────────
        let b0 = (ry1 + cx0) as usize; let b1 = (ry1 + cx1) as usize;
        let b2 = (ry1 + cx2) as usize; let b3 = (ry1 + cx3) as usize;
        let v00 = *src.add(b0);   let v01 = *src.add(b0+1); let v02 = *src.add(b0+2);
        let v10 = *src.add(b1);   let v11 = *src.add(b1+1); let v12 = *src.add(b1+2);
        let v20 = *src.add(b2);   let v21 = *src.add(b2+1); let v22 = *src.add(b2+2);
        let v30 = *src.add(b3);   let v31 = *src.add(b3+1); let v32 = *src.add(b3+2);
        let w0 = wx0*wy1; let w1 = wx1*wy1; let w2 = wx2*wy1; let w3 = wx3*wy1;
        acc0 += w0*v00; acc1 += w0*v01; acc2 += w0*v02;
        acc0 += w1*v10; acc1 += w1*v11; acc2 += w1*v12;
        acc0 += w2*v20; acc1 += w2*v21; acc2 += w2*v22;
        acc0 += w3*v30; acc1 += w3*v31; acc2 += w3*v32;

        // ── row 2, wy = wy2 ────────────────────────────────────────────────
        let b0 = (ry2 + cx0) as usize; let b1 = (ry2 + cx1) as usize;
        let b2 = (ry2 + cx2) as usize; let b3 = (ry2 + cx3) as usize;
        let v00 = *src.add(b0);   let v01 = *src.add(b0+1); let v02 = *src.add(b0+2);
        let v10 = *src.add(b1);   let v11 = *src.add(b1+1); let v12 = *src.add(b1+2);
        let v20 = *src.add(b2);   let v21 = *src.add(b2+1); let v22 = *src.add(b2+2);
        let v30 = *src.add(b3);   let v31 = *src.add(b3+1); let v32 = *src.add(b3+2);
        let w0 = wx0*wy2; let w1 = wx1*wy2; let w2 = wx2*wy2; let w3 = wx3*wy2;
        acc0 += w0*v00; acc1 += w0*v01; acc2 += w0*v02;
        acc0 += w1*v10; acc1 += w1*v11; acc2 += w1*v12;
        acc0 += w2*v20; acc1 += w2*v21; acc2 += w2*v22;
        acc0 += w3*v30; acc1 += w3*v31; acc2 += w3*v32;

        // ── row 3, wy = wy3 ────────────────────────────────────────────────
        let b0 = (ry3 + cx0) as usize; let b1 = (ry3 + cx1) as usize;
        let b2 = (ry3 + cx2) as usize; let b3 = (ry3 + cx3) as usize;
        let v00 = *src.add(b0);   let v01 = *src.add(b0+1); let v02 = *src.add(b0+2);
        let v10 = *src.add(b1);   let v11 = *src.add(b1+1); let v12 = *src.add(b1+2);
        let v20 = *src.add(b2);   let v21 = *src.add(b2+1); let v22 = *src.add(b2+2);
        let v30 = *src.add(b3);   let v31 = *src.add(b3+1); let v32 = *src.add(b3+2);
        let w0 = wx0*wy3; let w1 = wx1*wy3; let w2 = wx2*wy3; let w3 = wx3*wy3;
        acc0 += w0*v00; acc1 += w0*v01; acc2 += w0*v02;
        acc0 += w1*v10; acc1 += w1*v11; acc2 += w1*v12;
        acc0 += w2*v20; acc1 += w2*v21; acc2 += w2*v22;
        acc0 += w3*v30; acc1 += w3*v31; acc2 += w3*v32;

        *dst.add(out)   = acc0;
        *dst.add(out+1) = acc1;
        *dst.add(out+2) = acc2;
    }
}

fn main() {
    println!("resize_oxide: PTX-only crate; see the emitted .ptx files");
}
