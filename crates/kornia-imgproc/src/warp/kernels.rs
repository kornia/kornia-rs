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

use super::common::bilinear_sample_u8_valid;

/// Process a run of `[x_lo, x_hi)` destination columns for one row of a
/// perspective warp, filling `dst_row` in-place.
///
/// Inputs describe the row-constant perspective state at column `x_lo`:
///
/// - `nx`, `ny`, `nd`: numerator/denominator at `x = x_lo`.
/// - `dnx`, `dny`, `dnd`: per-column increments.
///
/// Preconditions (the caller is responsible for these):
///
/// - For every `x ∈ [x_lo, x_hi)` the source coordinate
///   `(nx+dnx*x)/(nd+dnd*x)` is in `[0, src_w)`, and the y coordinate
///   analog is in `[0, src_h)`.
/// - `nd` does not change sign on `[x_lo, x_hi)` (i.e. the row stays on
///   one side of the vanishing line).
///
/// The scalar fallback is always available and fully portable. The
/// aarch64/NEON path processes 4 columns per iteration using a 4-lane
/// reciprocal (estimate + one Newton–Raphson refine); for the 1-3
/// trailing columns it falls back to the scalar inner loop.
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
    nx: f32,
    ny: f32,
    nd: f32,
    dnx: f32,
    dny: f32,
    dnd: f32,
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64-unknown-linux-gnu. The helper
    // requires `x_lo ≤ x_hi ≤ dst_row.len() / C` and the perspective-sample-
    // in-bounds invariants documented above.
    unsafe {
        process_perspective_span_neon::<C>(
            src, src_w, src_h, src_stride, dst_row, x_lo, x_hi, nx, ny, nd, dnx, dny, dnd,
        );
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if crate::simd::cpu_features().has_avx2 {
        unsafe {
            process_perspective_span_avx2::<C>(
                src, src_w, src_h, src_stride, dst_row, x_lo, x_hi, nx, ny, nd, dnx, dny, dnd,
            );
        }
        return;
    }
    #[allow(unreachable_code)]
    process_perspective_span_scalar::<C>(
        src, src_w, src_h, src_stride, dst_row, x_lo, x_hi, nx, ny, nd, dnx, dny, dnd,
    );
}

/// Portable scalar implementation — reference for all backends.
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
    mut nx: f32,
    mut ny: f32,
    mut nd: f32,
    dnx: f32,
    dny: f32,
    dnd: f32,
) {
    for x in x_lo..x_hi {
        let inv_nd = 1.0 / nd;
        let xf = nx * inv_nd;
        let yf = ny * inv_nd;
        let xi = xf.floor() as i32;
        let yi = yf.floor() as i32;
        let fx_q10 = ((xf - xi as f32) * 1024.0) as u32;
        let fy_q10 = ((yf - yi as f32) * 1024.0) as u32;
        let dst_pixel = &mut dst_row[x * C..x * C + C];
        bilinear_sample_u8_valid::<C>(
            src, src_w, src_h, src_stride, xi, yi, fx_q10, fy_q10, dst_pixel,
        );
        nx += dnx;
        ny += dny;
        nd += dnd;
    }
}

/// aarch64 NEON implementation: 4-wide reciprocal + scalar bilinear per lane.
///
/// # Safety
/// - NEON is baseline on aarch64-unknown-linux-gnu (no runtime feature check
///   needed), so we mark the function `#[target_feature(enable = "neon")]`
///   to unlock the intrinsics but skip dynamic dispatch.
/// - Caller must satisfy the perspective-span preconditions documented on
///   [`process_perspective_span`].
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
    mut nx: f32,
    mut ny: f32,
    mut nd: f32,
    dnx: f32,
    dny: f32,
    dnd: f32,
) {
    use std::arch::aarch64::*;

    let n4 = (x_hi - x_lo) & !3;
    if n4 >= 4 {
        let dnx_v = vdupq_n_f32(dnx);
        let dny_v = vdupq_n_f32(dny);
        let dnd_v = vdupq_n_f32(dnd);
        let lane_offsets: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
        let lo_v = vld1q_f32(lane_offsets.as_ptr());
        let mut nx_v = vaddq_f32(vdupq_n_f32(nx), vmulq_f32(dnx_v, lo_v));
        let mut ny_v = vaddq_f32(vdupq_n_f32(ny), vmulq_f32(dny_v, lo_v));
        let mut nd_v = vaddq_f32(vdupq_n_f32(nd), vmulq_f32(dnd_v, lo_v));
        let step_nx = vmulq_n_f32(dnx_v, 4.0);
        let step_ny = vmulq_n_f32(dny_v, 4.0);
        let step_nd = vmulq_n_f32(dnd_v, 4.0);
        let q10_v = vdupq_n_f32(1024.0);

        let mut xs = [0i32; 4];
        let mut ys = [0i32; 4];
        let mut fxs = [0u32; 4];
        let mut fys = [0u32; 4];

        let end = x_lo + n4;
        let mut xi = x_lo;
        while xi < end {
            // Reciprocal: 1 NR step → ~17-bit precision, sufficient for
            // image-domain coords (worst-case fractional weight error
            // below 1 Q10 ULP).
            let r0 = vrecpeq_f32(nd_v);
            let inv_nd = vmulq_f32(vrecpsq_f32(nd_v, r0), r0);
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
                bilinear_sample_u8_valid::<C>(
                    src, src_w, src_h, src_stride, xs[lane], ys[lane], fxs[lane], fys[lane],
                    dst_pixel,
                );
            }

            nx_v = vaddq_f32(nx_v, step_nx);
            ny_v = vaddq_f32(ny_v, step_ny);
            nd_v = vaddq_f32(nd_v, step_nd);
            xi += 4;
        }
        // Pull scalar state from vector lane 0 for the tail.
        nx = vgetq_lane_f32::<0>(nx_v);
        ny = vgetq_lane_f32::<0>(ny_v);
        nd = vgetq_lane_f32::<0>(nd_v);
    }

    // 1-3 trailing columns: scalar tail.
    let tail_start = x_lo + n4;
    process_perspective_span_scalar::<C>(
        src, src_w, src_h, src_stride, dst_row, tail_start, x_hi, nx, ny, nd, dnx, dny, dnd,
    );
}

/// AVX2 implementation: 4-wide reciprocal (rcp + Newton–Raphson refine) +
/// scalar bilinear per lane. Mirrors the NEON path one-for-one; the per-lane
/// bilinear call dispatches to the AVX2 C=3 sampler in `common::bilinear_sample_u8_valid`.
///
/// # Safety
/// - AVX2 must be available (caller checks `cpu_features().has_avx2`).
/// - Caller must satisfy the perspective-span preconditions documented on
///   [`process_perspective_span`].
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
    mut nx: f32,
    mut ny: f32,
    mut nd: f32,
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
        let lane_offsets: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
        let lo_v = _mm_loadu_ps(lane_offsets.as_ptr());
        let mut nx_v = _mm_add_ps(_mm_set1_ps(nx), _mm_mul_ps(dnx_v, lo_v));
        let mut ny_v = _mm_add_ps(_mm_set1_ps(ny), _mm_mul_ps(dny_v, lo_v));
        let mut nd_v = _mm_add_ps(_mm_set1_ps(nd), _mm_mul_ps(dnd_v, lo_v));
        let step_nx = _mm_mul_ps(dnx_v, _mm_set1_ps(4.0));
        let step_ny = _mm_mul_ps(dny_v, _mm_set1_ps(4.0));
        let step_nd = _mm_mul_ps(dnd_v, _mm_set1_ps(4.0));
        let q10_v = _mm_set1_ps(1024.0);
        let two = _mm_set1_ps(2.0);

        let mut xs = [0i32; 4];
        let mut ys = [0i32; 4];
        let mut fxs = [0u32; 4];
        let mut fys = [0u32; 4];

        let end = x_lo + n4;
        let mut xi = x_lo;
        while xi < end {
            // Reciprocal: rcp gives ~12-bit estimate; one NR step
            // r1 = r0 * (2 - nd * r0) lifts to ~24-bit precision.
            let r0 = _mm_rcp_ps(nd_v);
            let inv_nd = _mm_mul_ps(r0, _mm_sub_ps(two, _mm_mul_ps(nd_v, r0)));
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
                bilinear_sample_u8_valid::<C>(
                    src, src_w, src_h, src_stride, xs[lane], ys[lane], fxs[lane], fys[lane],
                    dst_pixel,
                );
            }

            nx_v = _mm_add_ps(nx_v, step_nx);
            ny_v = _mm_add_ps(ny_v, step_ny);
            nd_v = _mm_add_ps(nd_v, step_nd);
            xi += 4;
        }
        // Pull scalar state from vector lane 0 for the tail.
        let mut tmp = [0f32; 4];
        _mm_storeu_ps(tmp.as_mut_ptr(), nx_v);
        nx = tmp[0];
        _mm_storeu_ps(tmp.as_mut_ptr(), ny_v);
        ny = tmp[0];
        _mm_storeu_ps(tmp.as_mut_ptr(), nd_v);
        nd = tmp[0];
    }

    // 1-3 trailing columns: scalar tail.
    let tail_start = x_lo + n4;
    process_perspective_span_scalar::<C>(
        src, src_w, src_h, src_stride, dst_row, tail_start, x_hi, nx, ny, nd, dnx, dny, dnd,
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
/// Preconditions: `0 ≤ (sx_q >> 16) < src_w` and `0 ≤ (sy_q >> 16) <
/// src_h` for every `x` in `[x_lo, x_hi)`. `dst_row.len() ≥ x_hi * C`.
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
    sx_q_lo: i32,
    sy_q_lo: i32,
    dsx_q: i32,
    dsy_q: i32,
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
    sx_q_lo: i32,
    sy_q_lo: i32,
    dsx_q: i32,
    dsy_q: i32,
) {
    const Q: i32 = 16;
    let mut sx_q = sx_q_lo;
    let mut sy_q = sy_q_lo;
    for x in x_lo..x_hi {
        let dst_pixel_ptr = unsafe { dst_row.as_mut_ptr().add(x * C) };
        let xi = sx_q >> Q;
        let yi = sy_q >> Q;
        let fx_q10 = ((sx_q & 0xFFFF) as u32) >> 6;
        let fy_q10 = ((sy_q & 0xFFFF) as u32) >> 6;
        let dst_pixel = unsafe { std::slice::from_raw_parts_mut(dst_pixel_ptr, C) };
        bilinear_sample_u8_valid::<C>(
            src, src_w, src_h, src_stride, xi, yi, fx_q10, fy_q10, dst_pixel,
        );
        sx_q = sx_q.wrapping_add(dsx_q);
        sy_q = sy_q.wrapping_add(dsy_q);
    }
}
