//! Rust-authored bilinear resize kernels for kornia-rs, compiled to PTX by
//! cuda-oxide.
//!
//! These are twins of `BILINEAR_SRC` in
//! `crates/kornia-imgproc/src/cuda/resize.rs:97` (entry
//! `resize_bilinear_downscale_3c`). The arithmetic is transcribed expression
//! for expression, because kornia's parity tests assert CPU == GPU with
//! `assert_eq!` on raw bytes, not a tolerance.
//!
//! Two variants exist to separate two questions that a single kernel would
//! conflate:
//!
//! * `resize_bilinear_oxide_3c` takes raw pointers, so its PTX parameter list
//!   matches the CUDA C kernel exactly and it carries no bounds checks. This is
//!   the byte-exactness arm.
//! * `resize_bilinear_oxide_slice_3c` takes `&[f32]` / `DisjointSlice<f32>`,
//!   which carry `readonly`/`noalias`. nvcc turns a bare
//!   `const float* __restrict__` into `ld.global.nc.f32` (the `__ldg` load) on
//!   its own; this variant tests whether the NVPTX backend does the same. The
//!   April cuda-oxide snapshot exposes no `__ldg` intrinsic and no `ptx_asm!`,
//!   so this is the only route to a non-coherent load.
//!
//! Build: `cargo oxide build resize_bilinear_oxide --arch sm_87`

use cuda_device::{DisjointSlice, kernel, thread};

/// Byte-exact twin of `resize_bilinear_downscale_3c`.
///
/// Parameter order matches the CUDA C kernel one for one, so the same cudarc
/// launch builder drives either.
///
/// # Safety
///
/// `src` must hold `src_h * src_w * 3` floats and `dst` at least
/// `dst_h * dst_w * 3`; all dimensions must be non-zero. kornia's launcher
/// validates both before launching.
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

    // Plain multiply then add, NOT a fused multiply-add: the two roundings are
    // what make the coordinate byte-identical to the CPU LUT's `a * x + b`.
    // The CUDA twin gets this from NVRTC `--fmad=false`; here it comes from
    // Rust never contracting float ops implicitly. Clamp order is min then max,
    // mirroring `fmaxf(fminf(..))`.
    let hix = (src_w - 1) as f32;
    let hiy = (src_h - 1) as f32;
    let cx = ax * dst_x as f32 + bx;
    let cy = ay * dst_y as f32 + by;
    // `f32::min`/`max` lower to LLVM `minimumnum`/`maximumnum`, which the April
    // cuda-oxide backend fails to verify. These comparisons are what
    // `fminf`/`fmaxf` reduce to for the non-NaN coordinates this kernel sees,
    // and they keep the min-then-max order of the CUDA twin.
    let cx = if cx < hix { cx } else { hix };
    let cy = if cy < hiy { cy } else { hiy };
    let sx = if cx > 0.0 { cx } else { 0.0 };
    let sy = if cy > 0.0 { cy } else { 0.0 };

    // Rust's `as u32` is saturating, so it emits a compare/select that the CUDA
    // twin's `(unsigned int)sx` does not. `to_int_unchecked` would give the bare
    // `cvt.rzi.u32.f32`, but the April backend cannot lower that intrinsic. The
    // clamps above keep both values in range, so the *value* is unaffected --
    // only the instruction count is.
    let x0 = sx as u32;
    let y0 = sy as u32;
    // Written as explicit comparisons, not `Ord::min`: the April cuda-oxide
    // snapshot routes integer `Ord::min` through drop glue it cannot emit.
    // Same value as the CUDA twin's `min(x0 + 1u, src_w - 1u)`.
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

    // Index math stays in u32 and widens only at the load, as in the CUDA twin.
    let b00 = ((y0 * src_w + x0) * 3) as usize;
    let b10 = ((y0 * src_w + x1) * 3) as usize;
    let b01 = ((y1 * src_w + x0) * 3) as usize;
    let b11 = ((y1 * src_w + x1) * 3) as usize;
    let out = ((dst_y * dst_w + dst_x) * 3) as usize;

    // All twelve loads are issued before the first store, and the stores come
    // last. This is not cosmetic: raw pointers carry no `noalias`, so with the
    // stores interleaved the compiler cannot hoist a later channel's loads
    // above an earlier channel's store, and the kernel pays three serialized
    // memory-latency round trips instead of one. The CUDA twin gets the same
    // batching for free from `__restrict__`. Measured on Orin at 2x upscale:
    // 0.384 ms interleaved vs 0.197 ms batched.
    //
    // Accumulation order still matches the twin's left-to-right sum; float
    // addition does not associate, so that ordering is part of the contract.
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

/// Same arithmetic through slice parameters, to see what `readonly`/`noalias`
/// buys in the emitted loads. Slices are fat pointers in PTX, so this kernel
/// takes two extra parameters and needs its own launch path.
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
    // `f32::min`/`max` lower to LLVM `minimumnum`/`maximumnum`, which the April
    // cuda-oxide backend fails to verify. These comparisons are what
    // `fminf`/`fmaxf` reduce to for the non-NaN coordinates this kernel sees,
    // and they keep the min-then-max order of the CUDA twin.
    let cx = if cx < hix { cx } else { hix };
    let cy = if cy < hiy { cy } else { hiy };
    let sx = if cx > 0.0 { cx } else { 0.0 };
    let sy = if cy > 0.0 { cy } else { 0.0 };

    let x0 = sx as u32;
    let y0 = sy as u32;
    // Written as explicit comparisons, not `Ord::min`: the April cuda-oxide
    // snapshot routes integer `Ord::min` through drop glue it cannot emit.
    // Same value as the CUDA twin's `min(x0 + 1u, src_w - 1u)`.
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

/// The device code is consumed as committed PTX by kornia's stable crates, so
/// this binary exists only to give the codegen backend an entry point to
/// compile.
fn main() {
    println!("resize_bilinear_oxide: PTX-only crate; see resize_bilinear_oxide.ptx");
}
