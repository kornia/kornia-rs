//! Stable-Rust fp16 GEMM micro-kernel using inline asm (aarch64 ARMv8.2 fp16).
//!
//! # Design
//!
//! This module provides an MR=4, NR=8 fp16 GEMM micro-kernel that avoids all
//! nightly-only intrinsics. The strategy:
//!
//! * Use `uint16x8_t` (stable NEON type) as a register proxy for `float16x8_t`.
//! * Emit FMLA.8H, FCVTN, FCVTN2, FCVTL, FCVTL2, and FMAX.8H via raw inline
//!   asm blocks so the compiler never needs to understand fp16 semantics.
//! * The `vreg_low16` register class (V0–V15) is required for the indexed-element
//!   operand of `FMLA Vd.8H, Vn.8H, Vm.H[idx]` — the ISA restricts Vm to
//!   V0–V15 for half-precision lane indexing.
//!
//! # Layout contracts (micro-kernel)
//!
//! ```text
//! A panel : [K, MR=4] f16-as-u16, column-major  → a_ptr[k*4 + m]
//! B panel : [K, NR=8] f16-as-u16, row-major     → b_ptr[k*8 + n]
//! C tile  : MR=4 row pointers, each pointing to NR=8 f16-as-u16 values
//! ```
//!
//! # Public surface
//!
//! * [`f32_to_f16_slice`] — batch f32→f16 NEON conversion.
//! * [`f16_to_f32_slice`] — batch f16→f32 NEON conversion.
//! * [`gemm_4x8_f16`]    — MR=4 × NR=8 micro-kernel.
//! * [`gemm_f16_mnk`]    — full M×K×N GEMM driver (tiles into micro-kernels).

#![allow(dead_code)]

use core::arch::aarch64::*;
use core::arch::asm;

// ─── f32 ↔ f16 conversion ────────────────────────────────────────────────────

/// Convert a slice of f32 to f16 (stored as u16), NEON-accelerated.
///
/// Writes into the caller-provided `dst` slice (must have `len >= src.len()`).
/// Uses `FCVTN` (4×f32 → 4×f16 low half) and `FCVTN2` (4×f32 → high half)
/// for groups of 8, then scalar `half::f16` for the tail.
///
/// # Safety
/// Caller must ensure `src` and `dst` are valid, aligned slices with `dst.len() >= src.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub unsafe fn f32_to_f16_buf(src: &[f32], dst: &mut [u16]) {
    debug_assert!(dst.len() >= src.len(), "dst too small: {} < {}", dst.len(), src.len());
    let n8 = src.len() / 8;
    let mut sp = src.as_ptr();
    let mut dp = dst.as_mut_ptr();
    for _ in 0..n8 {
        let lo = vld1q_f32(sp);
        let hi = vld1q_f32(sp.add(4));
        let mut result: uint16x8_t;
        // FCVTN  converts 4×f32→4×f16 into the lower half of the result register.
        // FCVTN2 converts another 4×f32→4×f16 into the upper half.
        // The `r` operand is marked `out` because FCVTN2 reads the low half of the
        // destination, so we must use two separate asm! blocks to avoid an
        // undefined-read: first build the low half (FCVTN), capture `r`, then
        // fill the high half (FCVTN2, inlateout so low half is preserved).
        asm!(
            "fcvtn  {r:v}.4h,  {lo:v}.4s",
            r  = out(vreg) result,
            lo = in(vreg) lo,
            options(nostack, pure, nomem),
        );
        // FCVTN2 needs to read+modify: inlateout preserves the low 4 lanes.
        asm!(
            "fcvtn2 {r:v}.8h, {hi:v}.4s",
            r  = inlateout(vreg) result,
            hi = in(vreg) hi,
            options(nostack),
        );
        vst1q_u16(dp, result);
        sp = sp.add(8);
        dp = dp.add(8);
    }
    // Scalar tail for lengths not divisible by 8.
    for i in (n8 * 8)..src.len() {
        let f = *src.get_unchecked(i);
        *dst.get_unchecked_mut(i) = half::f16::from_f32(f).to_bits();
    }
}

/// Convert a slice of f32 to f16 (stored as u16), NEON-accelerated.
/// Vec-resizing variant kept for backward compatibility and one-off conversions.
///
/// # Safety
/// Caller must ensure `src` is a valid, aligned slice.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub unsafe fn f32_to_f16_slice(src: &[f32], dst: &mut Vec<u16>) {
    dst.resize(src.len(), 0u16);
    f32_to_f16_buf(src, dst.as_mut_slice());
}

/// Convert a slice of f16 (stored as u16) to f32, NEON-accelerated.
///
/// Writes into the caller-provided `dst` slice (must have `len >= src.len()`).
/// Uses `FCVTL` (low 4 lanes f16→f32) and `FCVTL2` (high 4 lanes f16→f32)
/// for groups of 8, then scalar `half::f16` for the tail.
///
/// # Safety
/// Caller must ensure `src` and `dst` are valid, aligned slices with `dst.len() >= src.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub unsafe fn f16_to_f32_buf(src: &[u16], dst: &mut [f32]) {
    debug_assert!(dst.len() >= src.len(), "dst too small: {} < {}", dst.len(), src.len());
    let n8 = src.len() / 8;
    let mut sp = src.as_ptr();
    let mut dp = dst.as_mut_ptr();
    for _ in 0..n8 {
        let h8 = vld1q_u16(sp);
        let mut lo: float32x4_t;
        let mut hi: float32x4_t;
        asm!(
            "fcvtl  {lo:v}.4s, {h:v}.4h",
            "fcvtl2 {hi:v}.4s, {h:v}.8h",
            lo = out(vreg) lo,
            hi = out(vreg) hi,
            h  = in(vreg) h8,
            options(nostack, pure, nomem),
        );
        vst1q_f32(dp, lo);
        vst1q_f32(dp.add(4), hi);
        sp = sp.add(8);
        dp = dp.add(8);
    }
    for i in (n8 * 8)..src.len() {
        let bits = *src.get_unchecked(i);
        *dst.get_unchecked_mut(i) = half::f16::from_bits(bits).to_f32();
    }
}

/// Convert a slice of f16 (stored as u16) to f32, NEON-accelerated.
/// Vec-resizing variant kept for backward compatibility and one-off conversions.
///
/// # Safety
/// Caller must ensure `src` is a valid, aligned slice.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub unsafe fn f16_to_f32_slice(src: &[u16], dst: &mut Vec<f32>) {
    dst.resize(src.len(), 0.0f32);
    f16_to_f32_buf(src, dst.as_mut_slice());
}

/// Fused f16→f32 + bias + optional relu for a conv1×1 output strip.
///
/// Replaces the two-pass approach (f16→f32 into buffer, then bias+relu over buffer)
/// with a single pass: read f16 from `src`, convert, add `bias`, optionally relu, write f32.
/// Saves one full read+write of the M×c_out output buffer per strip (~300 KB per strip at
/// strip_m=800, c_out=64).
///
/// `src`: f16 input  `[strip_m * c_out]`
/// `dst`: f32 output `[strip_m * c_out]`  (must be disjoint from `src`)
/// `bias`: f32 bias  `[c_out]`
/// `c_out`: number of output channels (inner stride)
/// `use_relu`: apply `max(0, v)` after bias
///
/// # Safety
/// `src.len() >= strip_m * c_out`, `dst.len() >= strip_m * c_out`, `bias.len() == c_out`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub unsafe fn f16_to_f32_bias_relu_strip(
    src:     &[u16],
    dst:     &mut [f32],
    bias:    &[f32],
    c_out:   usize,
    use_relu: bool,
) {
    use std::arch::aarch64::*;
    let strip_m = src.len() / c_out;
    let n8      = c_out / 8;
    let tail    = c_out % 8;
    let zero    = vdupq_n_f32(0.0f32);

    for px in 0..strip_m {
        let sp = src .as_ptr()     .add(px * c_out);
        let dp = dst .as_mut_ptr() .add(px * c_out);
        let bp = bias.as_ptr();

        for g in 0..n8 {
            let h8 = vld1q_u16(sp.add(g * 8));
            let mut lo: float32x4_t;
            let mut hi: float32x4_t;
            asm!(
                "fcvtl  {lo:v}.4s, {h:v}.4h",
                "fcvtl2 {hi:v}.4s, {h:v}.8h",
                lo = out(vreg) lo,
                hi = out(vreg) hi,
                h  = in(vreg) h8,
                options(nostack, pure, nomem),
            );
            let blo = vld1q_f32(bp.add(g * 8));
            let bhi = vld1q_f32(bp.add(g * 8 + 4));
            let rlo = vaddq_f32(lo, blo);
            let rhi = vaddq_f32(hi, bhi);
            let rlo = if use_relu { vmaxq_f32(rlo, zero) } else { rlo };
            let rhi = if use_relu { vmaxq_f32(rhi, zero) } else { rhi };
            vst1q_f32(dp.add(g * 8),     rlo);
            vst1q_f32(dp.add(g * 8 + 4), rhi);
        }
        for co in (n8 * 8)..(n8 * 8 + tail) {
            let bits = *src.get_unchecked(px * c_out + co);
            let v = half::f16::from_bits(bits).to_f32() + *bias.get_unchecked(co);
            *dst.get_unchecked_mut(px * c_out + co) = if use_relu { v.max(0.0) } else { v };
        }
    }
}

// ─── FMLA asm wrappers ────────────────────────────────────────────────────────
//
// Each wrapper performs: acc += b × a.h[lane]  (all 8 output lanes simultaneously).
//
// The `a` operand uses `vreg_low16` (V0–V15) because the ARMv8.2 ISA restricts
// `Vm` in `FMLA Vd.8H, Vn.8H, Vm.H[idx]` to registers V0–V15.
// `b` and `acc` use the full `vreg` class (V0–V31).

/// acc += b × a.h[0]  (fp16, 8 lanes)
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,fp16")]
unsafe fn fmla_f16x8_lane0(mut acc: uint16x8_t, b: uint16x8_t, a: uint16x4_t) -> uint16x8_t {
    asm!(
        "fmla {acc:v}.8h, {b:v}.8h, {a:v}.h[0]",
        acc = inlateout(vreg) acc,
        b   = in(vreg) b,
        a   = in(vreg_low16) a,
        options(nostack),
    );
    acc
}

/// acc += b × a.h[1]  (fp16, 8 lanes)
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,fp16")]
unsafe fn fmla_f16x8_lane1(mut acc: uint16x8_t, b: uint16x8_t, a: uint16x4_t) -> uint16x8_t {
    asm!(
        "fmla {acc:v}.8h, {b:v}.8h, {a:v}.h[1]",
        acc = inlateout(vreg) acc,
        b   = in(vreg) b,
        a   = in(vreg_low16) a,
        options(nostack),
    );
    acc
}

/// acc += b × a.h[2]  (fp16, 8 lanes)
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,fp16")]
unsafe fn fmla_f16x8_lane2(mut acc: uint16x8_t, b: uint16x8_t, a: uint16x4_t) -> uint16x8_t {
    asm!(
        "fmla {acc:v}.8h, {b:v}.8h, {a:v}.h[2]",
        acc = inlateout(vreg) acc,
        b   = in(vreg) b,
        a   = in(vreg_low16) a,
        options(nostack),
    );
    acc
}

/// acc += b × a.h[3]  (fp16, 8 lanes)
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,fp16")]
unsafe fn fmla_f16x8_lane3(mut acc: uint16x8_t, b: uint16x8_t, a: uint16x4_t) -> uint16x8_t {
    asm!(
        "fmla {acc:v}.8h, {b:v}.8h, {a:v}.h[3]",
        acc = inlateout(vreg) acc,
        b   = in(vreg) b,
        a   = in(vreg_low16) a,
        options(nostack),
    );
    acc
}

// ─── fp16 direct-conv tap accumulator ────────────────────────────────────────

/// Inline FCVTN: 4 f32 values → 4 fp16 values packed into a `uint16x4_t`.
///
/// FCVTN fills the lower 4 lanes of the output V register. `vget_low_u16` is
/// a zero-cost register view, giving the `uint16x4_t` expected by the indexed
/// `FMLA Vd.8H, Vn.8H, Vm.H[i]` instruction.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,fp16")]
unsafe fn fcvtn_f32x4_to_f16x4(v: float32x4_t) -> uint16x4_t {
    let mut out: uint16x8_t;
    asm!(
        "fcvtn {out:v}.4h, {v:v}.4s",
        out = out(vreg) out,
        v   = in(vreg) v,
        options(nostack, pure, nomem),
    );
    vget_low_u16(out)
}

/// Accumulate one 3×3 tap into TWO fp16 accumulators (8 output channels each).
///
/// f32 input is converted to fp16 inline via FCVTN — no intermediate buffer.
/// Each FMLA.8H processes 8 output channels in one instruction, giving 2×
/// arithmetic density vs the f32 `accum_tap_2px` counterpart.
///
/// Uses two independent partial-sum pairs (A=ci..ci+3, B=ci+4..ci+7) so the
/// A78AE OoO scheduler fills the 4-cycle FMLA latency stalls of chain A with
/// chain B ops.  Measured speedup: ~1.25× for c_in=24, ~1.20× for c_in=64.
/// Single fold (`vaddq_u16`) at end of call — negligible overhead.
///
/// Weight layout: `[c_in, 8]` fp16-as-u16 from `repack_weights_co8_3x3_f16`.
/// `c_in` must be a multiple of 4.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
#[inline]
pub unsafe fn accum_tap_2px_f16(
    acc0: &mut uint16x8_t,
    acc1: &mut uint16x8_t,
    ip0:  *const f32,
    ip1:  *const f32,
    w_tap: *const u16,
    c_in: usize,
) {
    let mut acc0b = vdupq_n_u16(0u16);
    let mut acc1b = vdupq_n_u16(0u16);
    let mut ci = 0;
    while ci + 8 <= c_in {
        let iv0a = fcvtn_f32x4_to_f16x4(vld1q_f32(ip0.add(ci)));
        let iv1a = fcvtn_f32x4_to_f16x4(vld1q_f32(ip1.add(ci)));
        let iv0b = fcvtn_f32x4_to_f16x4(vld1q_f32(ip0.add(ci + 4)));
        let iv1b = fcvtn_f32x4_to_f16x4(vld1q_f32(ip1.add(ci + 4)));
        let wv0 = vld1q_u16(w_tap.add((ci + 0) * 8));
        let wv1 = vld1q_u16(w_tap.add((ci + 1) * 8));
        let wv2 = vld1q_u16(w_tap.add((ci + 2) * 8));
        let wv3 = vld1q_u16(w_tap.add((ci + 3) * 8));
        let wv4 = vld1q_u16(w_tap.add((ci + 4) * 8));
        let wv5 = vld1q_u16(w_tap.add((ci + 5) * 8));
        let wv6 = vld1q_u16(w_tap.add((ci + 6) * 8));
        let wv7 = vld1q_u16(w_tap.add((ci + 7) * 8));
        *acc0 = fmla_f16x8_lane0(*acc0, wv0, iv0a);
        *acc1 = fmla_f16x8_lane0(*acc1, wv0, iv1a);
        *acc0 = fmla_f16x8_lane1(*acc0, wv1, iv0a);
        *acc1 = fmla_f16x8_lane1(*acc1, wv1, iv1a);
        *acc0 = fmla_f16x8_lane2(*acc0, wv2, iv0a);
        *acc1 = fmla_f16x8_lane2(*acc1, wv2, iv1a);
        *acc0 = fmla_f16x8_lane3(*acc0, wv3, iv0a);
        *acc1 = fmla_f16x8_lane3(*acc1, wv3, iv1a);
        acc0b = fmla_f16x8_lane0(acc0b, wv4, iv0b);
        acc1b = fmla_f16x8_lane0(acc1b, wv4, iv1b);
        acc0b = fmla_f16x8_lane1(acc0b, wv5, iv0b);
        acc1b = fmla_f16x8_lane1(acc1b, wv5, iv1b);
        acc0b = fmla_f16x8_lane2(acc0b, wv6, iv0b);
        acc1b = fmla_f16x8_lane2(acc1b, wv6, iv1b);
        acc0b = fmla_f16x8_lane3(acc0b, wv7, iv0b);
        acc1b = fmla_f16x8_lane3(acc1b, wv7, iv1b);
        ci += 8;
    }
    while ci < c_in {
        let iv0 = fcvtn_f32x4_to_f16x4(vld1q_f32(ip0.add(ci)));
        let iv1 = fcvtn_f32x4_to_f16x4(vld1q_f32(ip1.add(ci)));
        let wv0 = vld1q_u16(w_tap.add((ci + 0) * 8));
        let wv1 = vld1q_u16(w_tap.add((ci + 1) * 8));
        let wv2 = vld1q_u16(w_tap.add((ci + 2) * 8));
        let wv3 = vld1q_u16(w_tap.add((ci + 3) * 8));
        *acc0 = fmla_f16x8_lane0(*acc0, wv0, iv0);
        *acc0 = fmla_f16x8_lane1(*acc0, wv1, iv0);
        *acc0 = fmla_f16x8_lane2(*acc0, wv2, iv0);
        *acc0 = fmla_f16x8_lane3(*acc0, wv3, iv0);
        *acc1 = fmla_f16x8_lane0(*acc1, wv0, iv1);
        *acc1 = fmla_f16x8_lane1(*acc1, wv1, iv1);
        *acc1 = fmla_f16x8_lane2(*acc1, wv2, iv1);
        *acc1 = fmla_f16x8_lane3(*acc1, wv3, iv1);
        ci += 4;
    }
    *acc0 = vaddq_u16(*acc0, acc0b);
    *acc1 = vaddq_u16(*acc1, acc1b);
}

/// Dual-chain variant of [`accum_tap_2px_f16`] for A/B benchmarking.
///
/// Uses two independent partial-sum pairs (A=ci..ci+3, B=ci+4..ci+7) so the
/// OoO scheduler can overlap the two 4-FMLA chains, filling the 4-cycle FMLA
/// latency stalls of chain A with chain B ops. Folded once at end of call.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
#[inline]
pub unsafe fn accum_tap_2px_f16_dual(
    acc0: &mut uint16x8_t,
    acc1: &mut uint16x8_t,
    ip0:  *const f32,
    ip1:  *const f32,
    w_tap: *const u16,
    c_in: usize,
) {
    let mut acc0b = vdupq_n_u16(0u16);
    let mut acc1b = vdupq_n_u16(0u16);
    let mut ci = 0;
    while ci + 8 <= c_in {
        let iv0a = fcvtn_f32x4_to_f16x4(vld1q_f32(ip0.add(ci)));
        let iv1a = fcvtn_f32x4_to_f16x4(vld1q_f32(ip1.add(ci)));
        let iv0b = fcvtn_f32x4_to_f16x4(vld1q_f32(ip0.add(ci + 4)));
        let iv1b = fcvtn_f32x4_to_f16x4(vld1q_f32(ip1.add(ci + 4)));
        let wv0 = vld1q_u16(w_tap.add((ci + 0) * 8));
        let wv1 = vld1q_u16(w_tap.add((ci + 1) * 8));
        let wv2 = vld1q_u16(w_tap.add((ci + 2) * 8));
        let wv3 = vld1q_u16(w_tap.add((ci + 3) * 8));
        let wv4 = vld1q_u16(w_tap.add((ci + 4) * 8));
        let wv5 = vld1q_u16(w_tap.add((ci + 5) * 8));
        let wv6 = vld1q_u16(w_tap.add((ci + 6) * 8));
        let wv7 = vld1q_u16(w_tap.add((ci + 7) * 8));
        *acc0 = fmla_f16x8_lane0(*acc0, wv0, iv0a);
        *acc1 = fmla_f16x8_lane0(*acc1, wv0, iv1a);
        *acc0 = fmla_f16x8_lane1(*acc0, wv1, iv0a);
        *acc1 = fmla_f16x8_lane1(*acc1, wv1, iv1a);
        *acc0 = fmla_f16x8_lane2(*acc0, wv2, iv0a);
        *acc1 = fmla_f16x8_lane2(*acc1, wv2, iv1a);
        *acc0 = fmla_f16x8_lane3(*acc0, wv3, iv0a);
        *acc1 = fmla_f16x8_lane3(*acc1, wv3, iv1a);
        acc0b = fmla_f16x8_lane0(acc0b, wv4, iv0b);
        acc1b = fmla_f16x8_lane0(acc1b, wv4, iv1b);
        acc0b = fmla_f16x8_lane1(acc0b, wv5, iv0b);
        acc1b = fmla_f16x8_lane1(acc1b, wv5, iv1b);
        acc0b = fmla_f16x8_lane2(acc0b, wv6, iv0b);
        acc1b = fmla_f16x8_lane2(acc1b, wv6, iv1b);
        acc0b = fmla_f16x8_lane3(acc0b, wv7, iv0b);
        acc1b = fmla_f16x8_lane3(acc1b, wv7, iv1b);
        ci += 8;
    }
    while ci < c_in {
        let iv0 = fcvtn_f32x4_to_f16x4(vld1q_f32(ip0.add(ci)));
        let iv1 = fcvtn_f32x4_to_f16x4(vld1q_f32(ip1.add(ci)));
        let wv0 = vld1q_u16(w_tap.add((ci + 0) * 8));
        let wv1 = vld1q_u16(w_tap.add((ci + 1) * 8));
        let wv2 = vld1q_u16(w_tap.add((ci + 2) * 8));
        let wv3 = vld1q_u16(w_tap.add((ci + 3) * 8));
        *acc0 = fmla_f16x8_lane0(*acc0, wv0, iv0);
        *acc0 = fmla_f16x8_lane1(*acc0, wv1, iv0);
        *acc0 = fmla_f16x8_lane2(*acc0, wv2, iv0);
        *acc0 = fmla_f16x8_lane3(*acc0, wv3, iv0);
        *acc1 = fmla_f16x8_lane0(*acc1, wv0, iv1);
        *acc1 = fmla_f16x8_lane1(*acc1, wv1, iv1);
        *acc1 = fmla_f16x8_lane2(*acc1, wv2, iv1);
        *acc1 = fmla_f16x8_lane3(*acc1, wv3, iv1);
        ci += 4;
    }
    *acc0 = vaddq_u16(*acc0, acc0b);
    *acc1 = vaddq_u16(*acc1, acc1b);
}

/// Single-pixel variant of [`accum_tap_2px_f16`].
///
/// Uses the same dual-chain pattern: `acc` handles channels ci..ci+3,
/// `accb` handles ci+4..ci+7 independently, folded once at end.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
#[inline]
pub(crate) unsafe fn accum_tap_1px_f16(
    acc: &mut uint16x8_t,
    ip:  *const f32,
    w_tap: *const u16,
    c_in: usize,
) {
    let mut accb = vdupq_n_u16(0u16);
    let mut ci = 0;
    while ci + 8 <= c_in {
        let iva = fcvtn_f32x4_to_f16x4(vld1q_f32(ip.add(ci)));
        let ivb = fcvtn_f32x4_to_f16x4(vld1q_f32(ip.add(ci + 4)));
        let wv0 = vld1q_u16(w_tap.add((ci + 0) * 8));
        let wv1 = vld1q_u16(w_tap.add((ci + 1) * 8));
        let wv2 = vld1q_u16(w_tap.add((ci + 2) * 8));
        let wv3 = vld1q_u16(w_tap.add((ci + 3) * 8));
        let wv4 = vld1q_u16(w_tap.add((ci + 4) * 8));
        let wv5 = vld1q_u16(w_tap.add((ci + 5) * 8));
        let wv6 = vld1q_u16(w_tap.add((ci + 6) * 8));
        let wv7 = vld1q_u16(w_tap.add((ci + 7) * 8));
        *acc = fmla_f16x8_lane0(*acc, wv0, iva);
        *acc = fmla_f16x8_lane1(*acc, wv1, iva);
        *acc = fmla_f16x8_lane2(*acc, wv2, iva);
        *acc = fmla_f16x8_lane3(*acc, wv3, iva);
        accb = fmla_f16x8_lane0(accb, wv4, ivb);
        accb = fmla_f16x8_lane1(accb, wv5, ivb);
        accb = fmla_f16x8_lane2(accb, wv6, ivb);
        accb = fmla_f16x8_lane3(accb, wv7, ivb);
        ci += 8;
    }
    while ci < c_in {
        let iv = fcvtn_f32x4_to_f16x4(vld1q_f32(ip.add(ci)));
        let wv0 = vld1q_u16(w_tap.add((ci + 0) * 8));
        let wv1 = vld1q_u16(w_tap.add((ci + 1) * 8));
        let wv2 = vld1q_u16(w_tap.add((ci + 2) * 8));
        let wv3 = vld1q_u16(w_tap.add((ci + 3) * 8));
        *acc = fmla_f16x8_lane0(*acc, wv0, iv);
        *acc = fmla_f16x8_lane1(*acc, wv1, iv);
        *acc = fmla_f16x8_lane2(*acc, wv2, iv);
        *acc = fmla_f16x8_lane3(*acc, wv3, iv);
        ci += 4;
    }
    *acc = vaddq_u16(*acc, accb);
}

/// ReLU in fp16: max(v, 0) element-wise over 8 half-precision lanes.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,fp16")]
unsafe fn relu_f16x8(v: uint16x8_t) -> uint16x8_t {
    let mut zero: uint16x8_t;
    asm!(
        "movi {0:v}.8h, #0",
        out(vreg) zero,
        options(nostack, pure, nomem),
    );
    let mut out: uint16x8_t;
    asm!(
        "fmax {0:v}.8h, {1:v}.8h, {2:v}.8h",
        out(vreg) out,
        in(vreg) v,
        in(vreg) zero,
        options(nostack, pure, nomem),
    );
    out
}

// ─── fp16 element-wise arithmetic helpers (for Winograd output transform) ────

/// Load an 8-channel f32 bias block into two `float32x4_t` halves.
///
/// When `bias` is `None`, both halves are zero. `co_base + 8 <= bias.len()`.
///
/// # Safety
/// `neon` must be available (always true on aarch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn load_bias_f16x8(
    bias: Option<&[f32]>,
    co_base: usize,
) -> (float32x4_t, float32x4_t) {
    match bias {
        Some(bs) => (
            vld1q_f32(bs[co_base..].as_ptr()),
            vld1q_f32(bs[co_base + 4..].as_ptr()),
        ),
        None => (vdupq_n_f32(0.0), vdupq_n_f32(0.0)),
    }
}

/// Load 8 contiguous f16-as-u16 values into a `uint16x8_t`.
///
/// Wraps `vld1q_u16` in a `target_feature`-gated fn so callers outside a
/// `#[target_feature(enable = "neon")]` context can use it on stable Rust.
///
/// # Safety
/// `ptr` must point to at least 8 valid `u16` values.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn vld1q_u16_wrap(ptr: *const u16) -> uint16x8_t {
    vld1q_u16(ptr)
}

/// FADD.8H: add two f16x8 vectors.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
#[inline]
pub(crate) unsafe fn fadd_f16x8(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    let mut out: uint16x8_t;
    asm!(
        "fadd {0:v}.8h, {1:v}.8h, {2:v}.8h",
        out(vreg) out,
        in(vreg) a,
        in(vreg) b,
        options(nostack, pure, nomem),
    );
    out
}

/// FSUB.8H: subtract two f16x8 vectors.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
#[inline]
pub(crate) unsafe fn fsub_f16x8(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    let mut out: uint16x8_t;
    asm!(
        "fsub {0:v}.8h, {1:v}.8h, {2:v}.8h",
        out(vreg) out,
        in(vreg) a,
        in(vreg) b,
        options(nostack, pure, nomem),
    );
    out
}

/// FCVTL: widen low 4 lanes of a f16x8 to f32x4.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
#[inline]
pub(crate) unsafe fn fcvtl_lo(v: uint16x8_t) -> float32x4_t {
    let mut out: float32x4_t;
    asm!(
        "fcvtl {0:v}.4s, {1:v}.4h",
        out(vreg) out,
        in(vreg) v,
        options(nostack, pure, nomem),
    );
    out
}

/// FCVTL2: widen high 4 lanes of a f16x8 to f32x4.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
#[inline]
pub(crate) unsafe fn fcvtl_hi(v: uint16x8_t) -> float32x4_t {
    let mut out: float32x4_t;
    asm!(
        "fcvtl2 {0:v}.4s, {1:v}.8h",
        out(vreg) out,
        in(vreg) v,
        options(nostack, pure, nomem),
    );
    out
}

/// Apply Winograd F(4,3) output transform Aᵀ·M·A in f16 for 8 channels at once,
/// then convert to f32, add bias, apply ReLU/Identity, and write to output.
///
/// `m[p]` = loaded f16x8 from `m_acc_f16[tile_ow*36*c_out + p*c_out + co_base]`.
/// Aᵀ (4×6): `[[1,1,1,1,1,0],[0,1,-1,2,-2,0],[0,1,1,4,4,0],[0,1,-1,8,-8,1]]`
/// A   (6×4): transpose of Aᵀ.
///
/// `bias_lo`/`bias_hi` hold the 8 f32 bias values for channels
/// `co_base..co_base+8`. `relu` selects ReLU (true) or Identity (false). Sigmoid
/// is never applied inside Winograd layers so it is not handled here.
///
/// Writes 4 rows × 4 cols × 8 ch = 128 f32 values to `out_row_base` at:
///   `out_row_base[ry * out_row_stride + (ow0 + cx) * c_out + co_base .. +8]`
///
/// # Safety
/// * `out_row_base` must be valid for writes at every computed offset.
/// * `c_out` must be ≥ `co_base + 8`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn winograd_f43_output_transform_write_f16x8(
    m: &[uint16x8_t; 36], // m[r*6+c] for r,c in 0..6
    bias_lo: float32x4_t, // bias for co_base..co_base+4
    bias_hi: float32x4_t, // bias for co_base+4..co_base+8
    relu: bool,
    out_row_base: *mut f32,
    out_row_stride: usize, // floats: w_out * c_out
    ow0: usize,
    c_out: usize,
    co_base: usize,
    valid_rows: usize, // rows to write (1..=4; 4 for full tiles)
    valid_cols: usize, // cols to write (1..=4; 4 for full tiles)
) {
    // Convenience macro: m[r][c]
    macro_rules! m {
        ($r:expr, $c:expr) => {
            m[$r * 6 + $c]
        };
    }

    // ── Step 1: s = Aᵀ @ M  (4×6 result) ─────────────────────────────────
    // Aᵀ rows:
    //   [0]: [1, 1, 1, 1, 1, 0]  → s0[c] = m[0][c]+m[1][c]+m[2][c]+m[3][c]+m[4][c]
    //   [1]: [0, 1,-1, 2,-2, 0]  → s1[c] = m[1][c]-m[2][c]+2*m[3][c]-2*m[4][c]
    //   [2]: [0, 1, 1, 4, 4, 0]  → s2[c] = m[1][c]+m[2][c]+4*m[3][c]+4*m[4][c]
    //   [3]: [0, 1,-1, 8,-8, 1]  → s3[c] = m[1][c]-m[2][c]+8*m[3][c]-8*m[4][c]+m[5][c]
    //
    // ×2 via self-add (exact for fp16). ×4 = ×2 of ×2. ×8 = ×2 of ×4.
    let mut s: [[uint16x8_t; 6]; 4] = core::mem::zeroed();
    for c in 0..6usize {
        let m3x2 = fadd_f16x8(m!(3, c), m!(3, c));
        let m4x2 = fadd_f16x8(m!(4, c), m!(4, c));
        let m3x4 = fadd_f16x8(m3x2, m3x2);
        let m4x4 = fadd_f16x8(m4x2, m4x2);
        let m3x8 = fadd_f16x8(m3x4, m3x4);
        let m4x8 = fadd_f16x8(m4x4, m4x4);

        s[0][c] = fadd_f16x8(
            fadd_f16x8(fadd_f16x8(fadd_f16x8(m!(0, c), m!(1, c)), m!(2, c)), m!(3, c)),
            m!(4, c),
        );
        s[1][c] = fsub_f16x8(fadd_f16x8(fsub_f16x8(m!(1, c), m!(2, c)), m3x2), m4x2);
        s[2][c] = fadd_f16x8(fadd_f16x8(fadd_f16x8(m!(1, c), m!(2, c)), m3x4), m4x4);
        s[3][c] = fsub_f16x8(
            fadd_f16x8(fadd_f16x8(fsub_f16x8(m!(1, c), m!(2, c)), m3x8), m!(5, c)),
            m4x8,
        );
    }

    // ── Step 2: y = s @ A  (4×4 result) ──────────────────────────────────
    // A cols (= Aᵀ rows transposed):
    //   [0]: [1, 1, 1, 1, 1, 0]  → y[i][0] = s[i][0]+s[i][1]+s[i][2]+s[i][3]+s[i][4]
    //   [1]: [0, 1,-1, 2,-2, 0]  → y[i][1] = s[i][1]-s[i][2]+2*s[i][3]-2*s[i][4]
    //   [2]: [0, 1, 1, 4, 4, 0]  → y[i][2] = s[i][1]+s[i][2]+4*s[i][3]+4*s[i][4]
    //   [3]: [0, 1,-1, 8,-8, 1]  → y[i][3] = s[i][1]-s[i][2]+8*s[i][3]-8*s[i][4]+s[i][5]
    let mut y: [[uint16x8_t; 4]; 4] = core::mem::zeroed();
    for i in 0..4usize {
        let s3x2 = fadd_f16x8(s[i][3], s[i][3]);
        let s4x2 = fadd_f16x8(s[i][4], s[i][4]);
        let s3x4 = fadd_f16x8(s3x2, s3x2);
        let s4x4 = fadd_f16x8(s4x2, s4x2);
        let s3x8 = fadd_f16x8(s3x4, s3x4);
        let s4x8 = fadd_f16x8(s4x4, s4x4);

        y[i][0] = fadd_f16x8(
            fadd_f16x8(fadd_f16x8(fadd_f16x8(s[i][0], s[i][1]), s[i][2]), s[i][3]),
            s[i][4],
        );
        y[i][1] = fsub_f16x8(fadd_f16x8(fsub_f16x8(s[i][1], s[i][2]), s3x2), s4x2);
        y[i][2] = fadd_f16x8(fadd_f16x8(fadd_f16x8(s[i][1], s[i][2]), s3x4), s4x4);
        y[i][3] = fsub_f16x8(
            fadd_f16x8(fadd_f16x8(fsub_f16x8(s[i][1], s[i][2]), s3x8), s[i][5]),
            s4x8,
        );
    }

    // ── Step 3: convert, add bias, activate, store ───────────────────────
    let zero = vdupq_n_f32(0.0);
    for ry in 0..valid_rows {
        for cx in 0..valid_cols {
            // Convert f16x8 → f32x4 (lo) + f32x4 (hi)
            let lo = vaddq_f32(fcvtl_lo(y[ry][cx]), bias_lo);
            let hi = vaddq_f32(fcvtl_hi(y[ry][cx]), bias_hi);
            let (lo, hi) = if relu {
                (vmaxq_f32(lo, zero), vmaxq_f32(hi, zero))
            } else {
                (lo, hi)
            };
            let dst = out_row_base.add(ry * out_row_stride + (ow0 + cx) * c_out + co_base);
            vst1q_f32(dst, lo);
            vst1q_f32(dst.add(4), hi);
        }
    }
}

// ─── MR=4, NR=8 fp16 GEMM micro-kernel ──────────────────────────────────────

/// MR=4, NR=8 fp16 GEMM micro-kernel (stable-Rust inline asm, aarch64 fp16).
///
/// Computes `C[0..4, 0..8] += A[0..K, 0..4]ᵀ × B[0..K, 0..8]` in fp16.
///
/// # Layout
/// ```text
/// A panel : [K, MR=4] packed f16-as-u16, column-major — a[k*4 + m]
/// B panel : [K, NR=8] packed f16-as-u16, row-major    — b[k*8 + n]
/// C tile  : four row pointers c0..c3, each NR=8 f16-as-u16 values
/// ```
///
/// When `accumulate` is `false`, the C tile is zeroed before accumulation.
/// When `true`, the existing C values are loaded and accumulated into.
///
/// # Safety
/// * `a` must point to at least `k * 4` valid u16 values.
/// * `b` must point to at least `k * 8` valid u16 values.
/// * `c0..c3` must each point to at least `8` valid u16 values (writable).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub unsafe fn gemm_4x8_f16(
    k: usize,
    a: *const u16, // A panel: [K, MR=4] column-major f16-as-u16
    b: *const u16, // B panel: [K, NR=8] row-major f16-as-u16
    c0: *mut u16,  // C row 0 output (NR=8 values)
    c1: *mut u16,  // C row 1 output
    c2: *mut u16,  // C row 2 output
    c3: *mut u16,  // C row 3 output
    accumulate: bool, // false → zero C tile first; true → add into existing C
) {
    let zero_bits: u16 = half::f16::ZERO.to_bits(); // 0x0000

    let mut acc0: uint16x8_t = if accumulate { vld1q_u16(c0) } else { vdupq_n_u16(zero_bits) };
    let mut acc1: uint16x8_t = if accumulate { vld1q_u16(c1) } else { vdupq_n_u16(zero_bits) };
    let mut acc2: uint16x8_t = if accumulate { vld1q_u16(c2) } else { vdupq_n_u16(zero_bits) };
    let mut acc3: uint16x8_t = if accumulate { vld1q_u16(c3) } else { vdupq_n_u16(zero_bits) };

    let mut ap = a;
    let mut bp = b;

    // ── K loop, unrolled 4× for better throughput ─────────────────────────────
    // Each iteration consumes 4 consecutive K slices:
    //   A: 4 × MR=4 u16 values (4 groups of 4 half-precision scalars)
    //   B: 4 × NR=8 u16 values (4 groups of 8 half-precision values)
    // Produces: 4 × MR × FMLA.8H = 16 fused multiply-accumulate ops.
    let k4 = k / 4;
    for _ in 0..k4 {
        // K+0: a0 = A[k+0, 0..4] as uint16x4_t (broadcast source lanes)
        //      b0 = B[k+0, 0..8] as uint16x8_t
        let a0 = vld1_u16(ap);
        let b0 = vld1q_u16(bp);
        acc0 = fmla_f16x8_lane0(acc0, b0, a0);
        acc1 = fmla_f16x8_lane1(acc1, b0, a0);
        acc2 = fmla_f16x8_lane2(acc2, b0, a0);
        acc3 = fmla_f16x8_lane3(acc3, b0, a0);

        // K+1
        let a1 = vld1_u16(ap.add(4));
        let b1 = vld1q_u16(bp.add(8));
        acc0 = fmla_f16x8_lane0(acc0, b1, a1);
        acc1 = fmla_f16x8_lane1(acc1, b1, a1);
        acc2 = fmla_f16x8_lane2(acc2, b1, a1);
        acc3 = fmla_f16x8_lane3(acc3, b1, a1);

        // K+2
        let a2 = vld1_u16(ap.add(8));
        let b2 = vld1q_u16(bp.add(16));
        acc0 = fmla_f16x8_lane0(acc0, b2, a2);
        acc1 = fmla_f16x8_lane1(acc1, b2, a2);
        acc2 = fmla_f16x8_lane2(acc2, b2, a2);
        acc3 = fmla_f16x8_lane3(acc3, b2, a2);

        // K+3
        let a3 = vld1_u16(ap.add(12));
        let b3 = vld1q_u16(bp.add(24));
        acc0 = fmla_f16x8_lane0(acc0, b3, a3);
        acc1 = fmla_f16x8_lane1(acc1, b3, a3);
        acc2 = fmla_f16x8_lane2(acc2, b3, a3);
        acc3 = fmla_f16x8_lane3(acc3, b3, a3);

        ap = ap.add(16); // 4 K steps × MR=4 u16 values per step
        bp = bp.add(32); // 4 K steps × NR=8 u16 values per step
    }

    // ── K tail (k not divisible by 4) — one step at a time ───────────────────
    for _ in (k4 * 4)..k {
        let a_tail = vld1_u16(ap);
        let b_tail = vld1q_u16(bp);
        acc0 = fmla_f16x8_lane0(acc0, b_tail, a_tail);
        acc1 = fmla_f16x8_lane1(acc1, b_tail, a_tail);
        acc2 = fmla_f16x8_lane2(acc2, b_tail, a_tail);
        acc3 = fmla_f16x8_lane3(acc3, b_tail, a_tail);
        ap = ap.add(4);
        bp = bp.add(8);
    }

    // ── Store results ─────────────────────────────────────────────────────────
    vst1q_u16(c0, acc0);
    vst1q_u16(c1, acc1);
    vst1q_u16(c2, acc2);
    vst1q_u16(c3, acc3);
}

/// MR=8 × NR=8 fp16 microkernel — eliminates FMA pipeline stalls.
///
/// With MR=4 and two dual-issue FMA pipes, 4 FMAs finish in 2 cycles but
/// FMA latency is 4 cycles → 2-cycle stall per K-step (50 % efficiency).
/// With MR=8, 8 FMAs use both pipes for exactly 4 cycles = FMA latency,
/// so acc0 is always ready before it is needed again → zero stalls → 2×
/// throughput vs `gemm_4x8_f16`.
///
/// A panel layout: `[K, 8]` — for each K index, 8 consecutive u16 values.
/// B panel layout: `[K, 8]` — same as before.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub unsafe fn gemm_8x8_f16(
    k: usize,
    a: *const u16, // A panel: [K, MR=8] column-major f16-as-u16
    b: *const u16, // B panel: [K, NR=8] row-major f16-as-u16
    c0: *mut u16,
    c1: *mut u16,
    c2: *mut u16,
    c3: *mut u16,
    c4: *mut u16,
    c5: *mut u16,
    c6: *mut u16,
    c7: *mut u16,
    accumulate: bool,
) {
    let zero_bits: u16 = half::f16::ZERO.to_bits();

    let mut acc0: uint16x8_t = if accumulate { vld1q_u16(c0) } else { vdupq_n_u16(zero_bits) };
    let mut acc1: uint16x8_t = if accumulate { vld1q_u16(c1) } else { vdupq_n_u16(zero_bits) };
    let mut acc2: uint16x8_t = if accumulate { vld1q_u16(c2) } else { vdupq_n_u16(zero_bits) };
    let mut acc3: uint16x8_t = if accumulate { vld1q_u16(c3) } else { vdupq_n_u16(zero_bits) };
    let mut acc4: uint16x8_t = if accumulate { vld1q_u16(c4) } else { vdupq_n_u16(zero_bits) };
    let mut acc5: uint16x8_t = if accumulate { vld1q_u16(c5) } else { vdupq_n_u16(zero_bits) };
    let mut acc6: uint16x8_t = if accumulate { vld1q_u16(c6) } else { vdupq_n_u16(zero_bits) };
    let mut acc7: uint16x8_t = if accumulate { vld1q_u16(c7) } else { vdupq_n_u16(zero_bits) };

    let mut ap = a;
    let mut bp = b;

    // K-unrolled 4×.  A panel stride per K step = MR=8 u16; B panel stride = NR=8 u16.
    // Rows 0–3 use a_lo (low 4 half-precision values); rows 4–7 use a_hi (high 4).
    // Each K step issues 8 FMLA.8H operations across the two FMA pipes (4 cycles),
    // which exactly matches the 4-cycle FMA latency → zero accumulator stalls.
    let k4 = k / 4;
    for _ in 0..k4 {
        // K+0
        let a_lo0 = vld1_u16(ap);
        let a_hi0 = vld1_u16(ap.add(4));
        let b0    = vld1q_u16(bp);
        acc0 = fmla_f16x8_lane0(acc0, b0, a_lo0);
        acc1 = fmla_f16x8_lane1(acc1, b0, a_lo0);
        acc2 = fmla_f16x8_lane2(acc2, b0, a_lo0);
        acc3 = fmla_f16x8_lane3(acc3, b0, a_lo0);
        acc4 = fmla_f16x8_lane0(acc4, b0, a_hi0);
        acc5 = fmla_f16x8_lane1(acc5, b0, a_hi0);
        acc6 = fmla_f16x8_lane2(acc6, b0, a_hi0);
        acc7 = fmla_f16x8_lane3(acc7, b0, a_hi0);

        // K+1
        let a_lo1 = vld1_u16(ap.add(8));
        let a_hi1 = vld1_u16(ap.add(12));
        let b1    = vld1q_u16(bp.add(8));
        acc0 = fmla_f16x8_lane0(acc0, b1, a_lo1);
        acc1 = fmla_f16x8_lane1(acc1, b1, a_lo1);
        acc2 = fmla_f16x8_lane2(acc2, b1, a_lo1);
        acc3 = fmla_f16x8_lane3(acc3, b1, a_lo1);
        acc4 = fmla_f16x8_lane0(acc4, b1, a_hi1);
        acc5 = fmla_f16x8_lane1(acc5, b1, a_hi1);
        acc6 = fmla_f16x8_lane2(acc6, b1, a_hi1);
        acc7 = fmla_f16x8_lane3(acc7, b1, a_hi1);

        // K+2
        let a_lo2 = vld1_u16(ap.add(16));
        let a_hi2 = vld1_u16(ap.add(20));
        let b2    = vld1q_u16(bp.add(16));
        acc0 = fmla_f16x8_lane0(acc0, b2, a_lo2);
        acc1 = fmla_f16x8_lane1(acc1, b2, a_lo2);
        acc2 = fmla_f16x8_lane2(acc2, b2, a_lo2);
        acc3 = fmla_f16x8_lane3(acc3, b2, a_lo2);
        acc4 = fmla_f16x8_lane0(acc4, b2, a_hi2);
        acc5 = fmla_f16x8_lane1(acc5, b2, a_hi2);
        acc6 = fmla_f16x8_lane2(acc6, b2, a_hi2);
        acc7 = fmla_f16x8_lane3(acc7, b2, a_hi2);

        // K+3
        let a_lo3 = vld1_u16(ap.add(24));
        let a_hi3 = vld1_u16(ap.add(28));
        let b3    = vld1q_u16(bp.add(24));
        acc0 = fmla_f16x8_lane0(acc0, b3, a_lo3);
        acc1 = fmla_f16x8_lane1(acc1, b3, a_lo3);
        acc2 = fmla_f16x8_lane2(acc2, b3, a_lo3);
        acc3 = fmla_f16x8_lane3(acc3, b3, a_lo3);
        acc4 = fmla_f16x8_lane0(acc4, b3, a_hi3);
        acc5 = fmla_f16x8_lane1(acc5, b3, a_hi3);
        acc6 = fmla_f16x8_lane2(acc6, b3, a_hi3);
        acc7 = fmla_f16x8_lane3(acc7, b3, a_hi3);

        ap = ap.add(32); // 4 K steps × MR=8 u16 per step
        bp = bp.add(32); // 4 K steps × NR=8 u16 per step
    }

    // K tail
    for _ in (k4 * 4)..k {
        let a_lo = vld1_u16(ap);
        let a_hi = vld1_u16(ap.add(4));
        let b_tail = vld1q_u16(bp);
        acc0 = fmla_f16x8_lane0(acc0, b_tail, a_lo);
        acc1 = fmla_f16x8_lane1(acc1, b_tail, a_lo);
        acc2 = fmla_f16x8_lane2(acc2, b_tail, a_lo);
        acc3 = fmla_f16x8_lane3(acc3, b_tail, a_lo);
        acc4 = fmla_f16x8_lane0(acc4, b_tail, a_hi);
        acc5 = fmla_f16x8_lane1(acc5, b_tail, a_hi);
        acc6 = fmla_f16x8_lane2(acc6, b_tail, a_hi);
        acc7 = fmla_f16x8_lane3(acc7, b_tail, a_hi);
        ap = ap.add(8);
        bp = bp.add(8);
    }

    vst1q_u16(c0, acc0);
    vst1q_u16(c1, acc1);
    vst1q_u16(c2, acc2);
    vst1q_u16(c3, acc3);
    vst1q_u16(c4, acc4);
    vst1q_u16(c5, acc5);
    vst1q_u16(c6, acc6);
    vst1q_u16(c7, acc7);
}

// ─── Full GEMM driver ─────────────────────────────────────────────────────────

/// Full M×K×N GEMM: `C[M, N] += A[M, K] × B[K, N]`, all in fp16 (u16 proxy).
///
/// Uses a two-phase panel-packing strategy (BLAS-style):
/// 1. Pre-pack the full B matrix once into `pack_b` as `[n_blocks, K, NR]` panels.
/// 2. For each M-block: pack the A panel once, then iterate all N-blocks using
///    the pre-packed B — eliminating redundant B packing (was done once per tile pair).
///
/// # Arguments
/// * `a`         — `[M, K]` row-major f16-as-u16.
/// * `b`         — `[K, N]` row-major f16-as-u16 (weights, already transposed if needed).
/// * `c`         — `[M, N]` row-major f16-as-u16 output; zeroed on entry, then written.
/// * `pack_a`    — caller-provided scratch, size `>= k * MR = k * 4`.
/// * `pack_b`    — caller-provided scratch, size `>= (n/8)*k*8`. For N divisible by NR=8
///                 this equals `k * n`; allocate `k * n` to be safe.
/// * `relu`      — if `true`, apply ReLU to each element of the output tile after GEMM.
///
/// # Panics
/// Debug-asserts slice lengths against `m`, `k`, `n`.
pub fn gemm_f16_mnk_with_pack(
    a: &[u16],        // [M, K] row-major f16-as-u16
    b: &[u16],        // [K, N] row-major f16-as-u16
    c: &mut [u16],    // [M, N] row-major f16-as-u16 — caller pre-zeros or pre-fills with bias
    m: usize,
    k: usize,
    n: usize,
    pack_a: &mut [u16], // scratch >= k * 4 (one A panel)
    pack_b: &mut [u16], // scratch >= (n/8)*k*8 (full B pre-pack)
    relu: bool,
) {
    const MR: usize = 4;
    const NR: usize = 8;

    let m_blocks = m / MR;
    let m_rem = m % MR;
    let n_blocks = n / NR;
    let n_rem = n % NR;

    debug_assert_eq!(a.len(), m * k, "A length mismatch: expected {m}×{k}={}", m * k);
    debug_assert_eq!(b.len(), k * n, "B length mismatch: expected {k}×{n}={}", k * n);
    debug_assert_eq!(c.len(), m * n, "C length mismatch: expected {m}×{n}={}", m * n);
    debug_assert!(pack_a.len() >= k * MR, "pack_a too small: {} < {}", pack_a.len(), k * MR);
    debug_assert!(
        pack_b.len() >= n_blocks * k * NR,
        "pack_b too small: {} < {}",
        pack_b.len(), n_blocks * k * NR
    );

    // ── Phase 1: Pre-pack entire B into [n_blocks, K, NR] layout (done once) ─
    // Eliminates the 1-per-(mb,nb)-pair redundancy: for M=4800, K=64, N=64
    // this drops B-pack calls from 9,600 to 8.
    for nb in 0..n_blocks {
        let nr_start = nb * NR;
        for ki in 0..k {
            for ni in 0..NR {
                pack_b[nb * k * NR + ki * NR + ni] = b[ki * n + nr_start + ni];
            }
        }
    }

    // ── Phase 2: Pack A once per M-block; iterate all N-blocks ───────────────
    for mb in 0..m_blocks {
        let mr_start = mb * MR;

        // Pack A panel [MR=4 rows, K cols] → [K, MR=4] column-major. Done once per M-block.
        let a_packed = &mut pack_a[..k * MR];
        for ki in 0..k {
            for mi in 0..MR {
                a_packed[ki * MR + mi] = a[(mr_start + mi) * k + ki];
            }
        }

        for nb in 0..n_blocks {
            let nr_start = nb * NR;

            #[cfg(target_arch = "aarch64")]
            unsafe {
                let b_packed = &pack_b[nb * k * NR..(nb + 1) * k * NR];

                let c0 = c.as_mut_ptr().add(mr_start * n + nr_start);
                let c1 = c.as_mut_ptr().add((mr_start + 1) * n + nr_start);
                let c2 = c.as_mut_ptr().add((mr_start + 2) * n + nr_start);
                let c3 = c.as_mut_ptr().add((mr_start + 3) * n + nr_start);

                gemm_4x8_f16(
                    k,
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    c0, c1, c2, c3,
                    true, // accumulate into existing C (pre-filled with bias by caller)
                );

                if relu {
                    for row_ptr in [c0, c1, c2, c3] {
                        let v = vld1q_u16(row_ptr);
                        vst1q_u16(row_ptr, relu_f16x8(v));
                    }
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                // Scalar fallback for non-aarch64 hosts (cross-compile/test).
                gemm_tile_scalar_f16(a, b, c, mr_start, nr_start, k, n, MR, NR, relu);
            }
        }

        // ── N remainder: columns n_blocks*NR .. n ─────────────────────────────
        if n_rem > 0 {
            let nr_start = n_blocks * NR;
            gemm_tile_scalar_f16(a, b, c, mr_start, nr_start, k, n, MR, n_rem, relu);
        }
    }

    // ── M remainder: rows m_blocks*MR .. m ───────────────────────────────────
    if m_rem > 0 {
        let mr_start = m_blocks * MR;
        gemm_tile_scalar_f16(a, b, c, mr_start, 0, k, n, m_rem, n, relu);
    }
}

/// Full M×K×N GEMM — allocating variant (kept for tests and one-off uses).
/// Prefer [`gemm_f16_mnk_with_pack`] on the hot path.
pub fn gemm_f16_mnk(
    a: &[u16],     // [M, K] row-major f16-as-u16
    b: &[u16],     // [K, N] row-major f16-as-u16
    c: &mut [u16], // [M, N] row-major f16-as-u16 — caller pre-zeros or pre-fills with bias
    m: usize,
    k: usize,
    n: usize,
    relu: bool,
) {
    let mut pack_a = vec![0u16; k * 4];
    let mut pack_b = vec![0u16; k * n]; // full B pre-pack: (n/8)*k*8 ≤ k*n
    gemm_f16_mnk_with_pack(a, b, c, m, k, n, &mut pack_a, &mut pack_b, relu);
}

/// Scalar fp16 tile: C[mr_start..mr_start+mr_rows, nr_start..nr_start+nr_cols]
/// += A[mr_start.., :] × B[:, nr_start..]. Remainder handler and non-aarch64 fallback.
fn gemm_tile_scalar_f16(
    a: &[u16],
    b: &[u16],
    c: &mut [u16],
    mr_start: usize,
    nr_start: usize,
    k: usize,
    n: usize,
    mr_rows: usize,
    nr_cols: usize,
    relu: bool,
) {
    for mi in 0..mr_rows {
        for ni in 0..nr_cols {
            let mut acc = half::f16::from_bits(c[(mr_start + mi) * n + nr_start + ni]).to_f32();
            for ki in 0..k {
                let av = half::f16::from_bits(a[(mr_start + mi) * k + ki]).to_f32();
                let bv = half::f16::from_bits(b[ki * n + nr_start + ni]).to_f32();
                acc += av * bv;
            }
            if relu && acc < 0.0 {
                acc = 0.0;
            }
            c[(mr_start + mi) * n + nr_start + ni] = half::f16::from_f32(acc).to_bits();
        }
    }
}

// ─── conv1×1 fp16 hot-path (scratch-buffer variant) ─────────────────────────

/// 1×1 NHWC convolution using the fp16 GEMM micro-kernel — zero-alloc hot path.
///
/// Treats the conv1×1 as a flat matrix multiply:
///   C [H×W, c_out] = A [H×W, c_in] × B [c_in, c_out]
///
/// All intermediate buffers are provided by the caller (pre-allocated on the
/// XFeat struct) so this function performs zero heap allocations per call.
///
/// # Scratch sizes (caller must guarantee)
/// * `scratch_a` — length >= `h * w * c_in`  (f16 input)
/// * `scratch_b` — length >= `c_in * c_out`   (f16 transposed weights)
/// * `scratch_c` — length >= `h * w * c_out`  (f16 GEMM output)
/// * `pack_a`    — length >= `c_in * 4`          (GEMM A-panel pack)
/// * `pack_b`    — length >= `c_in * c_out`      (full B pre-pack buffer)
///
/// Activation::Sigmoid falls back to the NEON f32 conv1x1 v2 path since
/// sigmoid in fp16 is impractical.
#[cfg(target_arch = "aarch64")]
pub fn conv1x1_nhwc_f16_scratch(
    args: &super::Conv1x1Args<'_>,
    output: &mut [f32],
    scratch_a: &mut [u16], // f16 input:   >= h*w*c_in
    scratch_b: &mut [u16], // f16 weights: >= c_in*c_out (transposed)
    scratch_c: &mut [u16], // f16 output:  >= h*w*c_out
    pack_a:    &mut [u16], // GEMM A-pack: >= c_in*4
    pack_b:    &mut [u16], // GEMM B-pack: >= c_in*c_out (full B pre-pack)
) {
    use super::Activation;

    let &super::Conv1x1Args {
        input,
        weights,
        bias,
        h,
        w,
        c_in,
        c_out,
        activation,
    } = args;

    // Sigmoid is not suitable for fp16 (loss of precision in the exponent);
    // fall back to the NEON f32 path.
    if activation == Activation::Sigmoid {
        super::neon::conv1x1_nhwc_v2(args, output);
        return;
    }

    let m = h * w; // number of pixels

    // Convert input to f16 — write into caller's scratch_a[0..m*c_in].
    let a_f16 = &mut scratch_a[..m * c_in];
    unsafe { f32_to_f16_buf(input, a_f16) };

    // Transpose weights [c_out, c_in] → [c_in, c_out] in f16 into scratch_b.
    // gemm_f16_mnk expects B in [K=c_in, N=c_out] row-major.
    let b_f16 = &mut scratch_b[..c_in * c_out];
    for co in 0..c_out {
        for ci in 0..c_in {
            b_f16[ci * c_out + co] =
                half::f16::from_f32(weights[co * c_in + ci]).to_bits();
        }
    }

    // Run fp16 GEMM — write result into scratch_c[0..m*c_out], no relu.
    let c_f16 = &mut scratch_c[..m * c_out];
    c_f16.fill(half::f16::ZERO.to_bits());
    gemm_f16_mnk_with_pack(a_f16, b_f16, c_f16, m, c_in, c_out, pack_a, pack_b, false);

    // Convert C back to f32 (reuse output slice directly), add bias, apply activation.
    // We convert into output[..] in-place: first convert, then add bias in-place.
    #[cfg(target_arch = "aarch64")]
    unsafe {
        f16_to_f32_buf(c_f16, &mut output[..m * c_out]);
    }

    let use_relu = activation == Activation::Relu;
    for px in 0..m {
        for co in 0..c_out {
            let val = output[px * c_out + co] + bias[co];
            output[px * c_out + co] = if use_relu { val.max(0.0) } else { val };
        }
    }
}

// ─── conv1×1 fp16 vtable entry ────────────────────────────────────────────────

/// 1×1 NHWC convolution using the fp16 GEMM micro-kernel — vtable-compatible entry.
///
/// This is the vtable entry (`fn(&Conv1x1Args, &mut [f32])`); it allocates its own
/// scratch buffers.  For the zero-alloc hot path used by `XFeat::extract`, see
/// [`conv1x1_nhwc_f16_scratch`].
///
/// On the Jetson Orin NX (Cortex-A78AE), FMLA.8H has 2× throughput over
/// FMLA.4S, so this path is roughly 2× faster for the head layers
/// (heatmap_head, keypoint_head) that dominate the non-Winograd time.
#[cfg(target_arch = "aarch64")]
pub fn conv1x1_nhwc_f16(args: &super::Conv1x1Args<'_>, output: &mut [f32]) {
    use super::Activation;

    // Sigmoid early-out — fall back to the NEON f32 path.
    if args.activation == Activation::Sigmoid {
        super::neon::conv1x1_nhwc_v2(args, output);
        return;
    }

    let m = args.h * args.w;
    let c_in = args.c_in;
    let c_out = args.c_out;

    // Allocate scratch buffers for this one-off / vtable call.
    let mut scratch_a = vec![0u16; m * c_in];
    let mut scratch_b = vec![0u16; c_in * c_out];
    let mut scratch_c = vec![half::f16::ZERO.to_bits(); m * c_out];
    let mut pack_a    = vec![0u16; c_in * 4];
    let mut pack_b    = vec![0u16; c_in * 8];

    conv1x1_nhwc_f16_scratch(
        args,
        output,
        &mut scratch_a,
        &mut scratch_b,
        &mut scratch_c,
        &mut pack_a,
        &mut pack_b,
    );
    // All work (bias + activation) is done inside conv1x1_nhwc_f16_scratch.
}

// ─── Parallel conv1×1 ─────────────────────────────────────────────────────────

use std::cell::RefCell;

/// Pixels per Rayon strip for parallel conv1×1: 800 × (c_in + c_out) × 2 ≈ 200 KB,
/// fitting inside each core's 512 KB L2. With M=4800 (60×80) this gives exactly 6 strips,
/// one per core on the Jetson Orin's 6-core A78AE cluster — no idle cores.
pub const CONV1X1_STRIP_SIZE: usize = 800;

thread_local! {
    static CONV1X1_STRIP_A:  RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
    static CONV1X1_STRIP_C:  RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
    static CONV1X1_STRIP_PA: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
}

/// Like [`gemm_f16_mnk_with_pack`] but B is already in `[n_blocks, K, NR]` packed layout.
/// Phase 1 (B packing) is skipped — caller must have pre-packed `b` before calling.
/// Restriction: `n` must be a multiple of NR=8 (asserted in debug builds).
#[cfg(target_arch = "aarch64")]
pub fn gemm_f16_mnk_packed_b(
    a: &[u16],          // [M, K] row-major f16-as-u16
    packed_b: &[u16],   // [n_blocks, K, NR] pre-packed f16-as-u16
    b_rem: &[u16],      // [K, n_rem] row-major f16 — the n%NR tail columns; may be empty
    c: &mut [u16],      // [M, N] row-major f16-as-u16 — caller pre-zeros
    m: usize,
    k: usize,
    n: usize,           // total N (may not be a multiple of NR=8)
    pack_a: &mut [u16], // scratch >= k * 8
) {
    const MR: usize = 8;
    const NR: usize = 8;

    let m_blocks = m / MR;
    let m_rem    = m % MR;
    let n_blocks = n / NR;
    let n_rem    = n % NR;
    debug_assert!(b_rem.len() >= k * n_rem);

    // Phase 2: Pack A once per M-block; call microkernel for every N-block.
    for mb in 0..m_blocks {
        let mr_start = mb * MR;

        let a_packed = &mut pack_a[..k * MR];
        for ki in 0..k {
            for mi in 0..MR {
                a_packed[ki * MR + mi] = a[(mr_start + mi) * k + ki];
            }
        }

        for nb in 0..n_blocks {
            let nr_start = nb * NR;
            unsafe {
                let b_packed = &packed_b[nb * k * NR..(nb + 1) * k * NR];
                let c0 = c.as_mut_ptr().add(mr_start * n + nr_start);
                let c1 = c.as_mut_ptr().add((mr_start + 1) * n + nr_start);
                let c2 = c.as_mut_ptr().add((mr_start + 2) * n + nr_start);
                let c3 = c.as_mut_ptr().add((mr_start + 3) * n + nr_start);
                let c4 = c.as_mut_ptr().add((mr_start + 4) * n + nr_start);
                let c5 = c.as_mut_ptr().add((mr_start + 5) * n + nr_start);
                let c6 = c.as_mut_ptr().add((mr_start + 6) * n + nr_start);
                let c7 = c.as_mut_ptr().add((mr_start + 7) * n + nr_start);
                gemm_8x8_f16(
                    k, a_packed.as_ptr(), b_packed.as_ptr(),
                    c0, c1, c2, c3, c4, c5, c6, c7, true,
                );
            }
        }

        // N-remainder: NEON dot product for n_rem=1 (contiguous b column),
        // scalar fallback for n_rem=2..7.
        if n_rem > 0 {
            let nr_start = n_blocks * NR;
            if n_rem == 1 {
                // b_rem[ki * 1 + 0] = b_rem[ki]: a single contiguous column of length k.
                // Compute MR independent NEON dot products, 2 rows interleaved to fill both FMA pipes.
                unsafe {
                    use std::arch::aarch64::*;
                    let bp = b_rem.as_ptr();
                    let k8 = k / 8;
                    for pair in 0..(MR / 2) {
                        let mi0 = pair * 2;
                        let mi1 = pair * 2 + 1;
                        let ap0 = a.as_ptr().add((mr_start + mi0) * k);
                        let ap1 = a.as_ptr().add((mr_start + mi1) * k);
                        let mut acc0_lo = vdupq_n_f32(0.0f32);
                        let mut acc0_hi = vdupq_n_f32(0.0f32);
                        let mut acc1_lo = vdupq_n_f32(0.0f32);
                        let mut acc1_hi = vdupq_n_f32(0.0f32);
                        for ki8 in 0..k8 {
                            let ah0 = vld1q_u16(ap0.add(ki8 * 8));
                            let ah1 = vld1q_u16(ap1.add(ki8 * 8));
                            let bh  = vld1q_u16(bp .add(ki8 * 8));
                            let mut a0lo: float32x4_t; let mut a0hi: float32x4_t;
                            let mut a1lo: float32x4_t; let mut a1hi: float32x4_t;
                            let mut blo:  float32x4_t; let mut bhi:  float32x4_t;
                            asm!("fcvtl  {lo:v}.4s, {h:v}.4h",
                                 "fcvtl2 {hi:v}.4s, {h:v}.8h",
                                 lo = out(vreg) a0lo, hi = out(vreg) a0hi, h = in(vreg) ah0,
                                 options(nostack, pure, nomem));
                            asm!("fcvtl  {lo:v}.4s, {h:v}.4h",
                                 "fcvtl2 {hi:v}.4s, {h:v}.8h",
                                 lo = out(vreg) a1lo, hi = out(vreg) a1hi, h = in(vreg) ah1,
                                 options(nostack, pure, nomem));
                            asm!("fcvtl  {lo:v}.4s, {h:v}.4h",
                                 "fcvtl2 {hi:v}.4s, {h:v}.8h",
                                 lo = out(vreg) blo, hi = out(vreg) bhi, h = in(vreg) bh,
                                 options(nostack, pure, nomem));
                            acc0_lo = vfmaq_f32(acc0_lo, a0lo, blo);
                            acc0_hi = vfmaq_f32(acc0_hi, a0hi, bhi);
                            acc1_lo = vfmaq_f32(acc1_lo, a1lo, blo);
                            acc1_hi = vfmaq_f32(acc1_hi, a1hi, bhi);
                        }
                        let s0 = vaddvq_f32(vaddq_f32(acc0_lo, acc0_hi));
                        let s1 = vaddvq_f32(vaddq_f32(acc1_lo, acc1_hi));
                        let mut acc0 = s0;
                        let mut acc1 = s1;
                        for ki in (k8 * 8)..k {
                            let av0 = half::f16::from_bits(*ap0.add(ki)).to_f32();
                            let av1 = half::f16::from_bits(*ap1.add(ki)).to_f32();
                            let bv  = half::f16::from_bits(*bp .add(ki)).to_f32();
                            acc0 += av0 * bv;
                            acc1 += av1 * bv;
                        }
                        c[(mr_start + mi0) * n + nr_start] = half::f16::from_f32(acc0).to_bits();
                        c[(mr_start + mi1) * n + nr_start] = half::f16::from_f32(acc1).to_bits();
                    }
                }
            } else {
                for mi in 0..MR {
                    for ni in 0..n_rem {
                        let mut acc = 0.0f32;
                        for ki in 0..k {
                            let a_val = half::f16::from_bits(a[(mr_start + mi) * k + ki]).to_f32();
                            let b_val = half::f16::from_bits(b_rem[ki * n_rem + ni]).to_f32();
                            acc += a_val * b_val;
                        }
                        c[(mr_start + mi) * n + nr_start + ni] = half::f16::from_f32(acc).to_bits();
                    }
                }
            }
        }
    }

    // M remainder: scalar path using packed_b directly (b_packed[nb][ki][ni]).
    if m_rem > 0 {
        let mr_start = m_blocks * MR;
        for mi in 0..m_rem {
            for nb in 0..n_blocks {
                let nr_start = nb * NR;
                let b_packed_slice = &packed_b[nb * k * NR..(nb + 1) * k * NR];
                for ni in 0..NR {
                    let mut acc = half::f16::from_bits(c[(mr_start + mi) * n + nr_start + ni]).to_f32();
                    for ki in 0..k {
                        let a_val = half::f16::from_bits(a[(mr_start + mi) * k + ki]).to_f32();
                        acc = a_val.mul_add(half::f16::from_bits(b_packed_slice[ki * NR + ni]).to_f32(), acc);
                    }
                    c[(mr_start + mi) * n + nr_start + ni] = half::f16::from_f32(acc).to_bits();
                }
            }
            // N-remainder for M-remainder rows too.
            if n_rem > 0 {
                let nr_start = n_blocks * NR;
                for ni in 0..n_rem {
                    let mut acc = 0.0f32;
                    for ki in 0..k {
                        let a_val = half::f16::from_bits(a[(mr_start + mi) * k + ki]).to_f32();
                        let b_val = half::f16::from_bits(b_rem[ki * n_rem + ni]).to_f32();
                        acc += a_val * b_val;
                    }
                    c[(mr_start + mi) * n + nr_start + ni] = half::f16::from_f32(acc).to_bits();
                }
            }
        }
    }
}

/// 1×1 NHWC convolution using a pre-packed B matrix and Rayon parallelism over
/// the spatial (M = h×w) dimension.
///
/// Effective when M is large (≥ 2 × CONV1X1_STRIP_SIZE). For any c_out, the
/// NR-aligned columns use the NEON microkernel and the N-remainder (c_out%8)
/// uses a scalar epilogue — no alignment restriction on c_out.
///
/// scratch_b: must hold at least c_in*c_out u16 (for B transposition).
/// pack_b: must hold at least c_in*c_out u16 (aligned blocks + rem tail).
#[cfg(target_arch = "aarch64")]
pub fn conv1x1_nhwc_f16_parallel(
    args: &super::Conv1x1Args<'_>,
    output: &mut [f32],
    scratch_b: &mut [u16], // >= c_in * c_out
    pack_b: &mut [u16],    // >= c_in * c_out
) {
    use rayon::prelude::*;
    use super::Activation;

    // Sigmoid needs f32 precision — fall back.
    if args.activation == Activation::Sigmoid {
        super::neon::conv1x1_nhwc_v2(args, output);
        return;
    }

    let m    = args.h * args.w;
    let c_in  = args.c_in;
    let c_out = args.c_out;

    // Only parallel when M is large enough to amortise Rayon overhead.
    if m < 2 * CONV1X1_STRIP_SIZE {
        let mut sa = vec![0u16; m * c_in];
        let mut sc = vec![half::f16::ZERO.to_bits(); m * c_out];
        let mut pa = vec![0u16; c_in * 4];
        conv1x1_nhwc_f16_scratch(args, output, &mut sa, scratch_b, &mut sc, &mut pa, pack_b);
        return;
    }

    // ── Phase 1: transpose + f32→f16 B (single-threaded, done once) ────────
    let b_f16 = &mut scratch_b[..c_in * c_out];
    for co in 0..c_out {
        for ci in 0..c_in {
            b_f16[ci * c_out + co] =
                half::f16::from_f32(args.weights[co * c_in + ci]).to_bits();
        }
    }

    // ── Phase 2: pre-pack B — aligned blocks then N-remainder ───────────────
    const NR: usize = 8;
    let n_blocks      = c_out / NR;
    let n_rem         = c_out % NR;
    let c_out_aligned = n_blocks * NR;
    let b_rem_offset  = n_blocks * c_in * NR; // pack_b index where b_rem starts

    for nb in 0..n_blocks {
        let nr_start = nb * NR;
        for ki in 0..c_in {
            for ni in 0..NR {
                pack_b[nb * c_in * NR + ki * NR + ni] = b_f16[ki * c_out + nr_start + ni];
            }
        }
    }
    // Pack the n_rem tail columns in [K, n_rem] row-major layout.
    for ki in 0..c_in {
        for ni in 0..n_rem {
            pack_b[b_rem_offset + ki * n_rem + ni] = b_f16[ki * c_out + c_out_aligned + ni];
        }
    }

    // ── Phase 3: parallel strips over M ─────────────────────────────────────
    // Pass pack_b as a raw pointer+len so the closure is 'Send (no mutable borrow).
    let pb_ptr      = pack_b.as_ptr() as usize;
    let pb_total    = b_rem_offset + c_in * n_rem; // total used pack_b elements
    let bias_ptr    = args.bias.as_ptr() as usize;
    let use_relu    = args.activation == Activation::Relu;

    args.input
        .par_chunks(CONV1X1_STRIP_SIZE * c_in)
        .zip(output.par_chunks_mut(CONV1X1_STRIP_SIZE * c_out))
        .for_each(|(in_strip, out_strip)| {
            let strip_m = in_strip.len() / c_in;

            // Reconstruct read-only slices from raw pointers (safe: outlives scope).
            let packed_b_all: &[u16] = unsafe {
                std::slice::from_raw_parts(pb_ptr as *const u16, pb_total)
            };
            let packed_b_ro = &packed_b_all[..b_rem_offset];
            let b_rem_ro    = &packed_b_all[b_rem_offset..];
            let bias: &[f32] = unsafe {
                std::slice::from_raw_parts(bias_ptr as *const f32, c_out)
            };

            CONV1X1_STRIP_A.with(|ca| {
            CONV1X1_STRIP_C.with(|cc| {
            CONV1X1_STRIP_PA.with(|cp| {
                let mut a_buf = ca.borrow_mut();
                let mut c_buf = cc.borrow_mut();
                let mut pa    = cp.borrow_mut();

                if a_buf.len() < strip_m * c_in  { a_buf.resize(strip_m * c_in, 0); }
                if c_buf.len() < strip_m * c_out { c_buf.resize(strip_m * c_out, 0); }
                if pa.len()    < c_in * 8        { pa.resize(c_in * 8, 0); }

                // f32 → f16 input strip.
                unsafe { f32_to_f16_buf(in_strip, &mut a_buf[..strip_m * c_in]) };

                // Clear C (no bias pre-fill; bias is added after f16→f32 below).
                c_buf[..strip_m * c_out].fill(0);

                // GEMM with pre-packed B (NEON for aligned columns, scalar for n_rem).
                gemm_f16_mnk_packed_b(
                    &a_buf[..strip_m * c_in],
                    packed_b_ro,
                    b_rem_ro,
                    &mut c_buf[..strip_m * c_out],
                    strip_m, c_in, c_out,
                    &mut pa[..c_in * 8],
                );

                // Fused f16→f32 + bias + relu in a single pass over the output buffer
                // (avoids one full read+write of M×c_out f32 vs the old two-pass approach).
                unsafe {
                    f16_to_f32_bias_relu_strip(
                        &c_buf[..strip_m * c_out],
                        out_strip,
                        bias,
                        c_out,
                        use_relu,
                    )
                };
            })})});
        });
}

// ─── Vectorized Winograd F(4,3) input transform ──────────────────────────────

/// B^T applied simultaneously to 4 channels (float32x4_t).
///
/// d[i] are 4-channel values for spatial position i in a row or column of 6.
/// Returns the 6 transformed 4-channel values.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn bt_f32x4(
    d0: std::arch::aarch64::float32x4_t,
    d1: std::arch::aarch64::float32x4_t,
    d2: std::arch::aarch64::float32x4_t,
    d3: std::arch::aarch64::float32x4_t,
    d4: std::arch::aarch64::float32x4_t,
    d5: std::arch::aarch64::float32x4_t,
) -> [std::arch::aarch64::float32x4_t; 6] {
    use std::arch::aarch64::*;
    let four = vdupq_n_f32(4.0);
    let five = vdupq_n_f32(5.0);
    let two  = vdupq_n_f32(2.0);
    // t0 = 4*d0 - 5*d2 + d4
    let t0 = vaddq_f32(vsubq_f32(vmulq_f32(four, d0), vmulq_f32(five, d2)), d4);
    // t1 = -4*(d1+d2) + d3 + d4
    let t1 = vsubq_f32(vaddq_f32(d3, d4), vmulq_f32(four, vaddq_f32(d1, d2)));
    // t2 = 4*(d1-d2) - d3 + d4
    let t2 = vaddq_f32(vsubq_f32(vmulq_f32(four, vsubq_f32(d1, d2)), d3), d4);
    // t3 = 2*(d3-d1) + (d4-d2)
    let t3 = vaddq_f32(vmulq_f32(two, vsubq_f32(d3, d1)), vsubq_f32(d4, d2));
    // t4 = 2*(d1-d3) + (d4-d2)
    let t4 = vaddq_f32(vmulq_f32(two, vsubq_f32(d1, d3)), vsubq_f32(d4, d2));
    // t5 = 4*d1 - 5*d3 + d5
    let t5 = vaddq_f32(vsubq_f32(vmulq_f32(four, d1), vmulq_f32(five, d3)), d5);
    [t0, t1, t2, t3, t4, t5]
}

/// Vectorized input transform for Winograd F(4,3): processes all c_in channels for
/// one tile column simultaneously, exploiting NHWC's contiguous channel dimension.
///
/// Replaces the per-channel scalar loop:
/// ```ignore
/// for ci in 0..c_in { extract_patch_f32_6x6(.., ci); transform(..) → a_all }
/// ```
/// with a single pass that loads all channels per spatial position, applies the
/// B^T transform using float32x4_t (4 channels at a time), and scatters to a_all.
///
/// Requires `c_in` to be a multiple of 4; a scalar tail handles any remainder.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub unsafe fn transform_input_tile_allch_f43(
    input: &[f32],
    h_in: usize,
    w_in: usize,
    c_in: usize,
    ih_start: isize,
    iw_start: isize,
    n_tile_w_gemm: usize,
    tile_ow: usize,
    a_all: &mut [u16],
) {
    use std::arch::aarch64::*;

    let zero = vdupq_n_f32(0.0);
    let cb_blocks = c_in / 4;

    for cb in 0..cb_blocks {
        let ci = cb * 4;

        // ── Load [6, 6, 4] neighborhood for this 4-channel block ──────────
        let mut nbhd = [[zero; 6]; 6]; // [row][col]
        for r in 0..6isize {
            let ih = ih_start + r;
            if ih < 0 || ih >= h_in as isize {
                continue;
            }
            let row_base = ih as usize * w_in;
            for c in 0..6isize {
                let iw = iw_start + c;
                if iw >= 0 && iw < w_in as isize {
                    nbhd[r as usize][c as usize] = vld1q_f32(
                        input.as_ptr().add((row_base + iw as usize) * c_in + ci),
                    );
                }
            }
        }

        // ── Row-wise B^T ────────────────────────────────────────────────────
        let mut t = [[zero; 6]; 6]; // [row][col]
        for r in 0..6usize {
            let n = nbhd[r];
            let row = bt_f32x4(n[0], n[1], n[2], n[3], n[4], n[5]);
            t[r] = row;
        }

        // ── Col-wise B^T → scatter to a_all ─────────────────────────────────
        // FCVTN: float32x4_t → float16x4 (lower 64 bits) in one instruction,
        // replacing 4 scalar half::f16::from_f32 calls per element (~15-30× speedup).
        for c in 0..6usize {
            let col = bt_f32x4(t[0][c], t[1][c], t[2][c], t[3][c], t[4][c], t[5][c]);
            for r in 0..6usize {
                let p = r * 6 + c;
                let out_base = p * n_tile_w_gemm * c_in + tile_ow * c_in + ci;
                // FCVTN fills the lower 4 half-words of a 128-bit register.
                let mut f16x8: uint16x8_t;
                asm!(
                    "fcvtn {r:v}.4h, {v:v}.4s",
                    r = out(vreg) f16x8,
                    v = in(vreg) col[r],
                    options(nostack, pure, nomem),
                );
                vst1_u16(a_all.as_mut_ptr().add(out_base), vget_low_u16(f16x8));
            }
        }
    }

    // ── Scalar tail for c_in % 4 != 0 (only hits for c_in=1 in XFeat) ──────
    let c_rem_start = cb_blocks * 4;
    for ci in c_rem_start..c_in {
        let patch =
            crate::ops::winograd::extract_patch_f32_6x6_pub(input, h_in, w_in, c_in, ih_start, iw_start, ci);
        let v = crate::ops::winograd::winograd_transform_input_tile_f32_f43(&patch);
        let idx = tile_ow * c_in + ci;
        for p in 0..36usize {
            a_all[p * n_tile_w_gemm * c_in + idx] = half::f16::from_f32(v[p]).to_bits();
        }
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference GEMM in f32 for tolerance comparison.
    fn matmul_ref_f32(a_f32: &[f32], b_f32: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    c[i * n + j] += a_f32[i * k + p] * b_f32[p * n + j];
                }
            }
        }
        c
    }

    fn to_f16(v: &[f32]) -> Vec<u16> {
        v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect()
    }
    fn to_f32(v: &[u16]) -> Vec<f32> {
        v.iter().map(|&x| half::f16::from_bits(x).to_f32()).collect()
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_f32_to_f16_roundtrip() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let mut f16_buf: Vec<u16> = Vec::new();
        unsafe {
            f32_to_f16_slice(&src, &mut f16_buf);
        }
        let mut back: Vec<f32> = Vec::new();
        unsafe {
            f16_to_f32_slice(&f16_buf, &mut back);
        }
        for (i, (&a, &b)) in src.iter().zip(back.iter()).enumerate() {
            let err = (a - b).abs();
            assert!(err < 0.01, "element {i}: f32={a} roundtrip={b} err={err}");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_4x8_identity() {
        // A = identity-like 4×4 (K=4, MR=4), B = ones 4×8.
        // Expected: each C row = sum of corresponding A row repeated across 8 cols.
        let k = 4usize;
        let a_f32: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let b_f32: Vec<f32> = vec![1.0f32; k * 8];

        // Pack A into [K, MR=4] column-major.
        let mut a_packed = vec![0u16; k * 4];
        for ki in 0..k {
            for mi in 0..4 {
                a_packed[ki * 4 + mi] = half::f16::from_f32(a_f32[mi * k + ki]).to_bits();
            }
        }
        let b_packed = to_f16(&b_f32);

        let mut c_buf = vec![0u16; 4 * 8];
        let cp = c_buf.as_mut_ptr();
        unsafe {
            gemm_4x8_f16(
                k,
                a_packed.as_ptr(),
                b_packed.as_ptr(),
                cp,
                cp.add(8),
                cp.add(16),
                cp.add(24),
                false,
            );
        }

        let c_f32 = to_f32(&c_buf);
        // Row 0: A[0,:] · ones = 1.0 (only first column is 1)
        // Row 1: A[1,:] · ones = 1.0 etc.
        for row in 0..4 {
            for col in 0..8 {
                let got = c_f32[row * 8 + col];
                let exp = 1.0f32;
                let err = (got - exp).abs();
                assert!(err < 0.01, "row={row} col={col}: got={got} expected={exp}");
            }
        }
    }

    #[test]
    fn test_gemm_f16_mnk_correctness() {
        // M=8, K=8, N=16 — exercises 2 MR-blocks × 2 NR-blocks with no remainder.
        let m = 8usize;
        let k = 8usize;
        let n = 16usize;

        let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.05).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();
        let ref_c = matmul_ref_f32(&a_f32, &b_f32, m, k, n);

        let a_f16 = to_f16(&a_f32);
        let b_f16 = to_f16(&b_f32);
        let mut c_f16 = vec![0u16; m * n];

        gemm_f16_mnk(&a_f16, &b_f16, &mut c_f16, m, k, n, false);

        let c_f32 = to_f32(&c_f16);
        for (i, (&got, &exp)) in c_f32.iter().zip(ref_c.iter()).enumerate() {
            // fp16 has ~0.1% relative error; use 0.5% tolerance to account for
            // accumulated rounding across K=8 steps.
            let rel_err = if exp.abs() > 1e-3 {
                (got - exp).abs() / exp.abs()
            } else {
                (got - exp).abs()
            };
            assert!(
                rel_err < 0.005,
                "element {i}: got={got:.4} expected={exp:.4} rel_err={rel_err:.6}"
            );
        }
    }

    #[test]
    fn test_gemm_f16_mnk_remainder() {
        // M=5, K=5, N=10 — tests both M and N remainders.
        let m = 5usize;
        let k = 5usize;
        let n = 10usize;

        let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let ref_c = matmul_ref_f32(&a_f32, &b_f32, m, k, n);

        let a_f16 = to_f16(&a_f32);
        let b_f16 = to_f16(&b_f32);
        let mut c_f16 = vec![0u16; m * n];

        gemm_f16_mnk(&a_f16, &b_f16, &mut c_f16, m, k, n, false);

        let c_f32 = to_f32(&c_f16);
        for (i, (&got, &exp)) in c_f32.iter().zip(ref_c.iter()).enumerate() {
            let rel_err = if exp.abs() > 1e-3 {
                (got - exp).abs() / exp.abs()
            } else {
                (got - exp).abs()
            };
            assert!(
                rel_err < 0.01,
                "element {i}: got={got:.4} expected={exp:.4} rel_err={rel_err:.6}"
            );
        }
    }

    #[test]
    fn test_gemm_f16_mnk_relu() {
        // 4×4×8: ensure ReLU zeroes negative outputs.
        let m = 4usize;
        let k = 4usize;
        let n = 8usize;
        // Mix of positive and negative values.
        let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32 - 16.0) * 0.25).collect();

        let a_f16 = to_f16(&a_f32);
        let b_f16 = to_f16(&b_f32);
        let mut c_f16 = vec![0u16; m * n];

        gemm_f16_mnk(&a_f16, &b_f16, &mut c_f16, m, k, n, true);

        let c_f32 = to_f32(&c_f16);
        for (i, &v) in c_f32.iter().enumerate() {
            assert!(v >= 0.0, "ReLU violated at element {i}: got {v}");
        }
    }

    /// Exercise `gemm_f16_mnk_packed_b` with N not a multiple of NR=8 (regression
    /// guard for the kp_head c_out=65 case).
    #[test]
    fn test_gemm_f16_mnk_packed_b_n_remainder() {
        const NR: usize = 8;
        // M=8 (2 MR-blocks), K=16, N=65 (8 NR-blocks + 1 rem).
        let m = 8usize;
        let k = 16usize;
        let n = 65usize;
        let n_blocks = n / NR;
        let n_rem    = n % NR;

        let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let ref_c = matmul_ref_f32(&a_f32, &b_f32, m, k, n);

        let a_f16 = to_f16(&a_f32);

        // Build packed_b: [n_blocks, K, NR] then [K, n_rem] tail.
        // b_f32 is [K, N] row-major.
        let b_rem_offset = n_blocks * k * NR;
        let mut packed_b = vec![0u16; b_rem_offset + k * n_rem];
        for nb in 0..n_blocks {
            let nr_start = nb * NR;
            for ki in 0..k {
                for ni in 0..NR {
                    packed_b[nb * k * NR + ki * NR + ni] =
                        half::f16::from_f32(b_f32[ki * n + nr_start + ni]).to_bits();
                }
            }
        }
        for ki in 0..k {
            for ni in 0..n_rem {
                packed_b[b_rem_offset + ki * n_rem + ni] =
                    half::f16::from_f32(b_f32[ki * n + n_blocks * NR + ni]).to_bits();
            }
        }

        let mut c_f16 = vec![0u16; m * n];
        let mut pa = vec![0u16; k * 8];
        gemm_f16_mnk_packed_b(
            &a_f16,
            &packed_b[..b_rem_offset],
            &packed_b[b_rem_offset..],
            &mut c_f16,
            m, k, n,
            &mut pa,
        );

        let c_f32 = to_f32(&c_f16);
        for (i, (&got, &exp)) in c_f32.iter().zip(ref_c.iter()).enumerate() {
            let rel_err = if exp.abs() > 1e-3 {
                (got - exp).abs() / exp.abs()
            } else {
                (got - exp).abs()
            };
            assert!(
                rel_err < 0.01,
                "element {i} (m={}, n={}): got={got:.4} expected={exp:.4} rel_err={rel_err:.6}",
                i / n, i % n,
            );
        }
    }

    /// Validate that `conv1x1_nhwc_f16_parallel` is correct for c_out=65 (kp_head shape).
    /// Regression guard: previously this would fall to serial because 65 % 8 != 0.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_conv1x1_f16_parallel_c_out_65() {
        use crate::ops::{Activation, Conv1x1Args};

        let h = 60usize;
        let w = 80usize;
        let c_in  = 64usize;
        let c_out = 65usize;
        let m = h * w;

        // Keep values small so f16 accumulation doesn't overflow (f16 max ≈ 65504).
        // Each output = sum_{k=0}^{63} a[k] * b[k]. With a ∈ [0, 0.02] and b ∈ [0, 0.01],
        // worst-case accumulation ≈ 64 * 0.02 * 0.01 = 0.0128 — well within f16 range.
        let weights: Vec<f32> = (0..c_out * c_in).map(|i| ((i % 20) as f32 + 1.0) * 0.0005).collect();
        let bias: Vec<f32>    = vec![0.0f32; c_out];
        let input: Vec<f32>   = (0..m * c_in).map(|i| ((i % 40) as f32 + 1.0) * 0.001).collect();

        // Reference: scalar path via the standard serial function.
        let mut ref_out = vec![0.0f32; m * c_out];
        crate::ops::scalar::conv1x1_nhwc(
            &Conv1x1Args { input: &input, weights: &weights, bias: &bias,
                           h, w, c_in, c_out, activation: Activation::Relu },
            &mut ref_out,
        );

        // Parallel path.
        let mut par_out = vec![0.0f32; m * c_out];
        let mut scratch_b = vec![0u16; c_in * c_out];
        let mut pack_b    = vec![0u16; c_in * c_out];
        conv1x1_nhwc_f16_parallel(
            &Conv1x1Args { input: &input, weights: &weights, bias: &bias,
                           h, w, c_in, c_out, activation: Activation::Relu },
            &mut par_out,
            &mut scratch_b,
            &mut pack_b,
        );

        for (i, (&got, &exp)) in par_out.iter().zip(ref_out.iter()).enumerate() {
            let err = (got - exp).abs();
            let tol = 0.01 * exp.abs().max(0.1);
            assert!(
                err < tol,
                "pixel {} ch {}: parallel={got:.4} scalar={exp:.4} abs_err={err:.6}",
                i / c_out, i % c_out,
            );
        }
    }
}
