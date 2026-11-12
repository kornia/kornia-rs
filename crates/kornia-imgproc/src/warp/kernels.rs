//! Per-architecture kernels for warp inner loops.
//!
//! This module centralizes the arch-specific fast paths used by the warp
//! primitives (`warp_affine_u8`, `warp_perspective_u8`). High-level code
//! calls the dispatch functions here; the dispatch functions select the
//! best available implementation at compile time via `cfg`.
//!
//! # Kernels exposed
//!
//! - [`process_affine_span`]  — fills one valid-range span of an affine
//!   warp row. Scalar reference + aarch64 NEON via
//!   `bilinear_sample_u8_valid` (which has its own C=3 NEON path).
//! - [`process_perspective_span`] — fills one valid-range span of a
//!   perspective warp row. Scalar reference + aarch64 NEON that vectors
//!   the 4-wide reciprocal for `1/nd`.
//!
//! # Adding a new backend
//!
//! To add (e.g.) an AVX2 or SVE path:
//!
//! 1. Add a `#[cfg(target_arch = "x86_64")]` block below the existing
//!    aarch64 block in the relevant dispatch function.
//! 2. Implement the kernel as an `unsafe fn` with
//!    `#[target_feature(enable = "avx2")]`.
//! 3. Use the scalar fallback (`*_scalar`) as the reference for both
//!    semantics and numerical behavior. Existing tests in
//!    `crates/kornia-imgproc/src/warp/*.rs` pin the scalar invariants;
//!    add correctness tests for any new backend that compare lane-
//!    equivalent output against the scalar kernel.
//!
//! Each dispatch function is designed to have a stable signature so
//! backends can be added without touching the callers.

use super::common::{
    bilinear_sample_u8, bilinear_sample_u8_valid, bilinear_sample_u8_valid_unchecked, is_packed,
};

/// Process a run of `[x_lo, x_hi)` destination columns for one row of a
/// perspective warp, filling `dst_row` in-place.
///
/// Inputs describe the row-constant perspective state at column `x = 0`:
///
/// - `nx0`, `ny0`, `nd0`: numerator/denominator row anchors.
/// - `dnx`, `dny`, `dnd`: per-column increments.
///
/// Every backend evaluates the coordinate DIRECTLY per column —
/// `nd = nd0 + dnd * x`, `inv_nd = 1.0 / nd` (exact IEEE division),
/// `xf = nx * inv_nd` — with the identical expression tree, so scalar,
/// NEON, and AVX2 are **bit-exact** with each other and with the CUDA u8
/// perspective kernel (which mirrors the same expressions under
/// `--fmad=false`). The previous incremental form (`nx += dnx` per column,
/// SIMD reciprocal estimate + refine) drifted a few ulps and made the
/// backends agree only "up to sub-ULP noise"; direct evaluation removes
/// both the drift and the approximation.
///
/// Expectations (performance, not safety): for every `x ∈ [x_lo, x_hi)` the
/// source coordinate `(nx0+dnx*x)/(nd0+dnd*x)` is in `[0, src_w)` (and the y
/// analog in `[0, src_h)`), and `nd` does not change sign on the span. The
/// span is derived in f32 and can disagree with the direct per-column
/// evaluation at the edges, so every backend still bounds-checks each pixel
/// and writes zeros for out-of-range coordinates.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn process_perspective_span<const C: usize>(
    src: &[u8],
    src_w: i32,
    src_h: i32,
    src_stride: usize,
    dst_row: &mut [u8],
    x_lo: usize,
    x_hi: usize,
    nx0: f32,
    ny0: f32,
    nd0: f32,
    dnx: f32,
    dny: f32,
    dnd: f32,
) {
    // The SIMD helpers sample without re-checking the buffer layout; anything
    // but a packed `src_h x src_w x C` buffer (never the case for an `Image`)
    // takes the fully checked scalar path.
    let packed = is_packed::<C>(src, src_w, src_h, src_stride) && x_hi <= dst_row.len() / C;
    #[cfg(target_arch = "aarch64")]
    if packed {
        // SAFETY: NEON is baseline on aarch64-unknown-linux-gnu; `packed`
        // establishes the helper's buffer-layout precondition, and each
        // lane's tap is range-checked before sampling.
        unsafe {
            process_perspective_span_neon::<C>(
                src, src_w, src_h, src_stride, dst_row, x_lo, x_hi, nx0, ny0, nd0, dnx, dny, dnd,
            );
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if packed && crate::simd::cpu_features().has_avx2 {
        // SAFETY: AVX2 confirmed by the runtime probe; `packed` establishes
        // the helper's buffer-layout precondition, and each lane's tap is
        // range-checked before sampling.
        unsafe {
            process_perspective_span_avx2::<C>(
                src, src_w, src_h, src_stride, dst_row, x_lo, x_hi, nx0, ny0, nd0, dnx, dny, dnd,
            );
        }
        return;
    }
    #[allow(unreachable_code)]
    process_perspective_span_scalar::<C>(
        src, src_w, src_h, src_stride, dst_row, x_lo, x_hi, nx0, ny0, nd0, dnx, dny, dnd,
    );
}

/// Direct per-column perspective sample coordinate. Single source of the
/// expression tree every backend (and the CUDA u8 kernel) reproduces:
/// plain mul + add (no FMA), exact division, truncating Q10 quantization.
#[inline(always)]
pub(super) fn perspective_coord_at(
    x: usize,
    nx0: f32,
    ny0: f32,
    nd0: f32,
    dnx: f32,
    dny: f32,
    dnd: f32,
) -> (f32, f32) {
    let x_f = x as f32;
    let nx = nx0 + dnx * x_f;
    let ny = ny0 + dny * x_f;
    let nd = nd0 + dnd * x_f;
    let inv_nd = 1.0 / nd;
    (nx * inv_nd, ny * inv_nd)
}

/// Portable scalar implementation — reference for all backends.
///
/// Byte-identical to the SIMD lanes: the same direct coordinate, the same Q10
/// quantization, and the same per-lane integer range check + unchecked sample
/// ([`sample_lane_checked`]), behind the same once-per-span layout check.
#[inline]
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) fn process_perspective_span_scalar<const C: usize>(
    src: &[u8],
    src_w: i32,
    src_h: i32,
    src_stride: usize,
    dst_row: &mut [u8],
    x_lo: usize,
    x_hi: usize,
    nx0: f32,
    ny0: f32,
    nd0: f32,
    dnx: f32,
    dny: f32,
    dnd: f32,
) {
    if !is_packed::<C>(src, src_w, src_h, src_stride) {
        perspective_span_checked::<C>(
            src, src_w, src_h, src_stride, dst_row, x_lo, x_hi, nx0, ny0, nd0, dnx, dny, dnd,
        );
        return;
    }
    let (src_w_f, src_h_f) = (src_w as f32, src_h as f32);
    for x in x_lo..x_hi {
        let (xf, yf) = perspective_coord_at(x, nx0, ny0, nd0, dnx, dny, dnd);
        let dst_pixel = &mut dst_row[x * C..x * C + C];
        // Zeros when the f32 span analysis admitted a column whose direct
        // coordinate lands outside the source (or is NaN — the negated test
        // rejects it, as the AVX2 lanes' `cvtt` does). In range, `floor` is
        // plain truncation, so the taps and Q10 fractions match the SIMD
        // lanes exactly.
        if !(xf >= 0.0 && xf < src_w_f && yf >= 0.0 && yf < src_h_f) {
            dst_pixel.fill(0);
            continue;
        }
        let xi = xf as i32;
        let yi = yf as i32;
        let fx_q10 = ((xf - xi as f32) * 1024.0) as u32;
        let fy_q10 = ((yf - yi as f32) * 1024.0) as u32;
        // SAFETY: `is_packed` holds (checked above) and `dst_pixel` is a
        // `C`-byte slice; the helper range-checks the integer tap itself.
        unsafe {
            sample_lane_checked::<C>(
                src, src_w, src_h, src_stride, xi, yi, fx_q10, fy_q10, dst_pixel,
            );
        }
    }
}

/// Fully checked fallback for a `src` that is not a packed
/// `src_h x src_w x C` buffer (never the case for an `Image`).
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn perspective_span_checked<const C: usize>(
    src: &[u8],
    src_w: i32,
    src_h: i32,
    src_stride: usize,
    dst_row: &mut [u8],
    x_lo: usize,
    x_hi: usize,
    nx0: f32,
    ny0: f32,
    nd0: f32,
    dnx: f32,
    dny: f32,
    dnd: f32,
) {
    for x in x_lo..x_hi {
        let (xf, yf) = perspective_coord_at(x, nx0, ny0, nd0, dnx, dny, dnd);
        let dst_pixel = &mut dst_row[x * C..x * C + C];
        bilinear_sample_u8::<C>(src, src_w, src_h, src_stride, xf, yf, dst_pixel);
    }
}

/// Per-lane tail of the perspective span kernels: zero-fill if the lane's
/// integer source tap is outside the image (the span analysis is f32 and can
/// admit an edge column whose direct coordinate falls just outside), else
/// sample. Mirrors [`bilinear_sample_u8`]'s bounds check.
///
/// # Safety
///
/// `src` must be a packed `src_h x src_w x C` buffer (`is_packed`) and
/// `dst_pixel.len() >= C`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn sample_lane_checked<const C: usize>(
    src: &[u8],
    src_w: i32,
    src_h: i32,
    src_stride: usize,
    xi: i32,
    yi: i32,
    fx_q10: u32,
    fy_q10: u32,
    dst_pixel: &mut [u8],
) {
    if xi < 0 || xi >= src_w || yi < 0 || yi >= src_h {
        dst_pixel.fill(0);
        return;
    }
    // SAFETY: the tap is in range (checked above); packed buffer and a
    // `C`-byte `dst_pixel` per this function's contract. The fractions only
    // affect the arithmetic, not memory access.
    unsafe {
        bilinear_sample_u8_valid_unchecked::<C>(
            src, src_w, src_h, src_stride, xi, yi, fx_q10, fy_q10, dst_pixel,
        );
    }
}

/// aarch64 NEON implementation: 4-wide reciprocal + scalar bilinear per lane.
///
/// # Safety
/// - NEON is baseline on aarch64-unknown-linux-gnu (no runtime feature check
///   needed), so we mark the function `#[target_feature(enable = "neon")]`
///   to unlock the intrinsics but skip dynamic dispatch.
/// - `src` must be a packed `src_h x src_w x C` buffer (`is_packed`); every
///   lane's tap is range-checked before the unchecked sample.
/// - `dst_row.len() >= x_hi * C`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn process_perspective_span_neon<const C: usize>(
    src: &[u8],
    src_w: i32,
    src_h: i32,
    src_stride: usize,
    dst_row: &mut [u8],
    x_lo: usize,
    x_hi: usize,
    nx0: f32,
    ny0: f32,
    nd0: f32,
    dnx: f32,
    dny: f32,
    dnd: f32,
) {
    unsafe {
        use std::arch::aarch64::*;

        let n4 = (x_hi - x_lo) & !3;
        if n4 >= 4 {
            let dnx_v = vdupq_n_f32(dnx);
            let dny_v = vdupq_n_f32(dny);
            let dnd_v = vdupq_n_f32(dnd);
            let nx0_v = vdupq_n_f32(nx0);
            let ny0_v = vdupq_n_f32(ny0);
            let nd0_v = vdupq_n_f32(nd0);
            let lane_offsets: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
            let lo_v = vld1q_f32(lane_offsets.as_ptr());
            let one_v = vdupq_n_f32(1.0);
            let q10_v = vdupq_n_f32(1024.0);

            let mut xs = [0i32; 4];
            let mut ys = [0i32; 4];
            let mut fxs = [0u32; 4];
            let mut fys = [0u32; 4];

            let end = x_lo + n4;
            let mut xi = x_lo;
            while xi < end {
                // Direct per-lane coordinate: x as f32 is exact (columns are
                // far below 2^24), so `n0 + dn * x` is lane-for-lane the same
                // rounding as the scalar reference, and `vdivq_f32` is exact
                // IEEE division — bit-identical to `1.0 / nd` scalar. This
                // replaced the reciprocal estimate + Newton–Raphson refine
                // (sub-ULP noise) and the `+= dn` accumulation (drift).
                let x_f = vaddq_f32(vdupq_n_f32(xi as f32), lo_v);
                let nx_v = vaddq_f32(nx0_v, vmulq_f32(dnx_v, x_f));
                let ny_v = vaddq_f32(ny0_v, vmulq_f32(dny_v, x_f));
                let nd_v = vaddq_f32(nd0_v, vmulq_f32(dnd_v, x_f));
                let inv_nd = vdivq_f32(one_v, nd_v);
                let xf = vmulq_f32(nx_v, inv_nd);
                let yf = vmulq_f32(ny_v, inv_nd);
                let xi_v = vcvtq_s32_f32(vrndmq_f32(xf));
                let yi_v = vcvtq_s32_f32(vrndmq_f32(yf));
                let fx_v = vcvtq_u32_f32(vmulq_f32(vsubq_f32(xf, vcvtq_f32_s32(xi_v)), q10_v));
                let fy_v = vcvtq_u32_f32(vmulq_f32(vsubq_f32(yf, vcvtq_f32_s32(yi_v)), q10_v));
                vst1q_s32(xs.as_mut_ptr(), xi_v);
                vst1q_s32(ys.as_mut_ptr(), yi_v);
                vst1q_u32(fxs.as_mut_ptr(), fx_v);
                vst1q_u32(fys.as_mut_ptr(), fy_v);

                for lane in 0..4 {
                    let dst_pixel = &mut dst_row[(xi + lane) * C..(xi + lane) * C + C];
                    // SAFETY: packed buffer per this function's contract; `dst_pixel`
                    // is a `C`-byte slice.
                    sample_lane_checked::<C>(
                        src, src_w, src_h, src_stride, xs[lane], ys[lane], fxs[lane], fys[lane],
                        dst_pixel,
                    );
                }

                xi += 4;
            }
        }

        // 1-3 trailing columns: scalar tail (same direct evaluation).
        let tail_start = x_lo + n4;
        process_perspective_span_scalar::<C>(
            src, src_w, src_h, src_stride, dst_row, tail_start, x_hi, nx0, ny0, nd0, dnx, dny, dnd,
        );
    }
}

/// AVX2 implementation: 4-wide reciprocal (rcp + Newton–Raphson refine) +
/// scalar bilinear per lane. Mirrors the NEON path one-for-one; the per-lane
/// bilinear call dispatches to the AVX2 C=3 sampler in `common::bilinear_sample_u8_valid`.
///
/// # Safety
/// - AVX2 must be available (caller checks `cpu_features().has_avx2`).
/// - `src` must be a packed `src_h x src_w x C` buffer (`is_packed`); every
///   lane's tap is range-checked before the unchecked sample.
/// - `dst_row.len() >= x_hi * C`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn process_perspective_span_avx2<const C: usize>(
    src: &[u8],
    src_w: i32,
    src_h: i32,
    src_stride: usize,
    dst_row: &mut [u8],
    x_lo: usize,
    x_hi: usize,
    nx0: f32,
    ny0: f32,
    nd0: f32,
    dnx: f32,
    dny: f32,
    dnd: f32,
) {
    use std::arch::x86_64::*;

    let n4 = (x_hi - x_lo) & !3;
    if n4 >= 4 {
        let dnx_v = _mm_set1_ps(dnx);
        let dny_v = _mm_set1_ps(dny);
        let dnd_v = _mm_set1_ps(dnd);
        let nx0_v = _mm_set1_ps(nx0);
        let ny0_v = _mm_set1_ps(ny0);
        let nd0_v = _mm_set1_ps(nd0);
        let lane_offsets: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
        let lo_v = _mm_loadu_ps(lane_offsets.as_ptr());
        let one_v = _mm_set1_ps(1.0);
        let q10_v = _mm_set1_ps(1024.0);

        let mut xs = [0i32; 4];
        let mut ys = [0i32; 4];
        let mut fxs = [0u32; 4];
        let mut fys = [0u32; 4];

        let end = x_lo + n4;
        let mut xi = x_lo;
        while xi < end {
            // Direct per-lane coordinate with exact IEEE division — see the
            // NEON path for the bit-exactness rationale (this replaced the
            // rcp + Newton–Raphson estimate and the `+= dn` accumulation).
            let x_f = _mm_add_ps(_mm_set1_ps(xi as f32), lo_v);
            let nx_v = _mm_add_ps(nx0_v, _mm_mul_ps(dnx_v, x_f));
            let ny_v = _mm_add_ps(ny0_v, _mm_mul_ps(dny_v, x_f));
            let nd_v = _mm_add_ps(nd0_v, _mm_mul_ps(dnd_v, x_f));
            let inv_nd = _mm_div_ps(one_v, nd_v);
            let xf = _mm_mul_ps(nx_v, inv_nd);
            let yf = _mm_mul_ps(ny_v, inv_nd);
            // Floor → i32 (truncating cvt after floor matches NEON's
            // vrndm + vcvtq_s32_f32 chain).
            let xi_v = _mm_cvttps_epi32(_mm_floor_ps(xf));
            let yi_v = _mm_cvttps_epi32(_mm_floor_ps(yf));
            let fx_v = _mm_cvttps_epi32(_mm_mul_ps(_mm_sub_ps(xf, _mm_cvtepi32_ps(xi_v)), q10_v));
            let fy_v = _mm_cvttps_epi32(_mm_mul_ps(_mm_sub_ps(yf, _mm_cvtepi32_ps(yi_v)), q10_v));
            _mm_storeu_si128(xs.as_mut_ptr() as *mut __m128i, xi_v);
            _mm_storeu_si128(ys.as_mut_ptr() as *mut __m128i, yi_v);
            _mm_storeu_si128(fxs.as_mut_ptr() as *mut __m128i, fx_v);
            _mm_storeu_si128(fys.as_mut_ptr() as *mut __m128i, fy_v);

            for lane in 0..4 {
                let dst_pixel = &mut dst_row[(xi + lane) * C..(xi + lane) * C + C];
                // SAFETY: packed buffer per this function's contract; `dst_pixel`
                // is a `C`-byte slice.
                sample_lane_checked::<C>(
                    src, src_w, src_h, src_stride, xs[lane], ys[lane], fxs[lane], fys[lane],
                    dst_pixel,
                );
            }

            xi += 4;
        }
    }

    // 1-3 trailing columns: scalar tail (same direct evaluation).
    let tail_start = x_lo + n4;
    process_perspective_span_scalar::<C>(
        src, src_w, src_h, src_stride, dst_row, tail_start, x_hi, nx0, ny0, nd0, dnx, dny, dnd,
    );
}

/// Process a run of `[x_lo, x_hi)` destination columns for one row of an
/// affine warp using Q16 fixed-point source coordinates.
///
/// Affine warps are linear, so the caller pre-computes the initial
/// Q16 source coordinates `sx_q_lo = round(sx0 * 2^16)` at `x = x_lo`,
/// plus the per-column increments `dsx_q = round(dsx * 2^16)`, and so on
/// for y. The Q16 trick eliminates per-pixel `floor`+`cvt` on the
/// float coordinate — one arithmetic shift recovers the integer part
/// and the low 16 bits become the fractional weight (right-shifted 6
/// more to reach Q10).
///
/// Expected (not trusted): `0 ≤ (sx_q >> 16) < src_w` and `0 ≤ (sy_q >> 16) <
/// src_h` for every `x` in `[x_lo, x_hi)`. Coordinates that stray outside
/// (the f32 span and the Q16 walk can disagree at the edges) are clamped to
/// the border by the sampler; `x_hi` is clamped to `dst_row.len() / C`.
///
/// Numerical behavior: identical across backends — the per-pixel call
/// is `bilinear_sample_u8_valid::<C>`, which itself has an internal
/// NEON C=3 fast path. Unlike `process_perspective_span`, no 4-wide
/// outer unroll is emitted: the Q16 coord add is already a single-cycle
/// dependency that the OoO core overlaps across iterations, and an
/// explicit unroll measured ~8% *slower* (register pressure without
/// new ILP — the divide-chain hazard that made unrolling pay in the
/// perspective kernel simply does not exist here).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn process_affine_span<const C: usize>(
    src: &[u8],
    src_w: i32,
    src_h: i32,
    src_stride: usize,
    dst_row: &mut [u8],
    x_lo: usize,
    x_hi: usize,
    sx_q_lo: i64,
    sy_q_lo: i64,
    dsx_q: i64,
    dsy_q: i64,
) {
    process_affine_span_scalar::<C>(
        src, src_w, src_h, src_stride, dst_row, x_lo, x_hi, sx_q_lo, sy_q_lo, dsx_q, dsy_q,
    );
}

/// Portable scalar implementation. Dispatched to on every target today;
/// a NEON 4× unroll was tried and regressed ~8% (see dispatch comment).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn process_affine_span_scalar<const C: usize>(
    src: &[u8],
    src_w: i32,
    src_h: i32,
    src_stride: usize,
    dst_row: &mut [u8],
    x_lo: usize,
    x_hi: usize,
    sx_q_lo: i64,
    sy_q_lo: i64,
    dsx_q: i64,
    dsy_q: i64,
) {
    const Q: i64 = 16;
    // Q16 coordinates are carried in i64: in i32 they overflow once a
    // coordinate reaches 2^15 = 32768 pixels, which used to wrap `xi`/`yi`
    // to arbitrary values.
    let x_hi = x_hi.min(dst_row.len() / C);
    if x_lo >= x_hi {
        return;
    }
    let span = &mut dst_row[x_lo * C..x_hi * C];
    let last = (x_hi - x_lo - 1) as i64;

    // The Q16 walk is exact integer arithmetic and linear in the column, so
    // every tap in the span is in range iff both end taps are. The caller's
    // span comes from f32 algebra that can disagree with this walk by a
    // pixel, so check it here (O(1) per row) and fall back to the clamping
    // sampler when it does not hold.
    let axis_ok = |q0: i64, dq: i64, len: i32| -> bool {
        let Some(q1) = last.checked_mul(dq).and_then(|t| q0.checked_add(t)) else {
            return false;
        };
        let (a, b) = (q0 >> Q, q1 >> Q);
        let len = len as i64;
        a >= 0 && a < len && b >= 0 && b < len
    };
    let fast = is_packed::<C>(src, src_w, src_h, src_stride)
        && axis_ok(sx_q_lo, dsx_q, src_w)
        && axis_ok(sy_q_lo, dsy_q, src_h);

    let mut sx_q = sx_q_lo;
    let mut sy_q = sy_q_lo;
    if fast {
        for dst_pixel in span.chunks_exact_mut(C) {
            let xi = (sx_q >> Q) as i32;
            let yi = (sy_q >> Q) as i32;
            let fx_q10 = ((sx_q & 0xFFFF) as u32) >> 6;
            let fy_q10 = ((sy_q & 0xFFFF) as u32) >> 6;
            // SAFETY: both end taps of this linear walk are inside the image
            // (checked above), so every intermediate tap is too; `is_packed`
            // certifies the buffer layout; `dst_pixel` is a `C`-byte chunk.
            unsafe {
                bilinear_sample_u8_valid_unchecked::<C>(
                    src, src_w, src_h, src_stride, xi, yi, fx_q10, fy_q10, dst_pixel,
                );
            }
            sx_q += dsx_q;
            sy_q += dsy_q;
        }
    } else {
        affine_span_clamped::<C>(
            src, src_w, src_h, src_stride, span, sx_q, sy_q, dsx_q, dsy_q,
        );
    }
}

/// Clamping fallback for [`process_affine_span_scalar`]: used for rows whose
/// Q16 walk leaves the image (the f32 span disagreed with the fixed-point
/// arithmetic). Kept out of line so the proven-in-range loop stays compact.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn affine_span_clamped<const C: usize>(
    src: &[u8],
    src_w: i32,
    src_h: i32,
    src_stride: usize,
    span: &mut [u8],
    mut sx_q: i64,
    mut sy_q: i64,
    dsx_q: i64,
    dsy_q: i64,
) {
    const Q: i64 = 16;
    let (w_max, h_max) = (src_w as i64, src_h as i64);
    for dst_pixel in span.chunks_exact_mut(C) {
        // Clamp to `[-1, src]` before narrowing; the sampler then clamps
        // into the image (edge replicate).
        let xi = (sx_q >> Q).clamp(-1, w_max) as i32;
        let yi = (sy_q >> Q).clamp(-1, h_max) as i32;
        let fx_q10 = ((sx_q & 0xFFFF) as u32) >> 6;
        let fy_q10 = ((sy_q & 0xFFFF) as u32) >> 6;
        bilinear_sample_u8_valid::<C>(
            src, src_w, src_h, src_stride, xi, yi, fx_q10, fy_q10, dst_pixel,
        );
        sx_q = sx_q.wrapping_add(dsx_q);
        sy_q = sy_q.wrapping_add(dsy_q);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn f(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * (self.next() % 10_000) as f32 / 10_000.0
        }
    }

    type SpanFn = fn(&[u8], i32, i32, usize, &mut [u8], usize, usize, f32, f32, f32, f32, f32, f32);

    fn check<const C: usize>(rng: &mut Lcg) {
        let (sw, sh) = (1 + rng.next() as usize % 12, 1 + rng.next() as usize % 12);
        let src: Vec<u8> = (0..sw * sh * C).map(|_| rng.next() as u8).collect();
        let dst_w = 24;
        let x_lo = rng.next() as usize % dst_w;
        let x_hi = x_lo + rng.next() as usize % (dst_w - x_lo + 1);
        let degenerate = rng.next() & 7 == 0;
        let (nd0, dnd) = if degenerate {
            (0.0, 0.0)
        } else {
            (rng.f(0.5, 2.0), rng.f(-0.01, 0.01))
        };
        let (nx0, ny0) = (rng.f(-3.0, 14.0), rng.f(-3.0, 14.0));
        let (dnx, dny) = (rng.f(-1.0, 1.0), rng.f(-1.0, 1.0));
        let args = (sw as i32, sh as i32, sw * C);
        let run = |f: SpanFn| {
            let mut dst = vec![0xAAu8; dst_w * C];
            f(
                &src, args.0, args.1, args.2, &mut dst, x_lo, x_hi, nx0, ny0, nd0, dnx, dny, dnd,
            );
            dst
        };
        let scalar = run(process_perspective_span_scalar::<C>);
        let reference = run(perspective_span_checked::<C>);
        assert_eq!(scalar, reference, "scalar vs checked reference");
        if !degenerate {
            let dispatched = run(process_perspective_span::<C>);
            assert_eq!(scalar, dispatched, "scalar vs SIMD dispatch");
        }
    }

    /// The scalar perspective span (range check + unchecked sampler) must be
    /// byte-identical to the fully checked float sampler it replaced and to
    /// the SIMD backends, including columns whose coordinate leaves the
    /// source (and NaN / infinite coordinates for the reference).
    #[test]
    fn perspective_scalar_span_matches_reference_and_simd() {
        let mut rng = Lcg(0x5EED);
        for _ in 0..3000 {
            check::<1>(&mut rng);
            check::<3>(&mut rng);
            check::<4>(&mut rng);
        }
    }
}
