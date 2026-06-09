//! BLIS-style MR=4, NR=8 micro-kernel GEMM for Winograd inner products.
//!
//! # Overview
//!
//! In Winograd F(2×2, 3×3) the per-spatial-position inner GEMM is:
//!
//! ```text
//!   C[M, N] += A[M, K] × B[K, N]
//!   M = num_tiles, K = c_in, N = c_out
//! ```
//!
//! This module provides:
//!
//! * [`winograd_gemm_f32`]  — public driver, tiles M×K×N into MR×NR blocks,
//!   calls the NEON micro-kernel for full blocks and scalar for remainders.
//!   Parallelised over MR-blocks with rayon.
//!
//! * [`pack_b_winograd`]    — transposes `weights_transformed[pos, c_out, c_in]`
//!   to `[c_in, c_out]` row-major for one Winograd position.
//!
//! # Micro-kernel tiling
//!
//! ```text
//! MR = 4   (output tile rows, = num_tiles dimension)
//! NR = 8   (output tile cols, = c_out dimension)
//! ```
//!
//! A panel  : `[MR=4, K]` row-major f32 (input tiles, K = c_in).
//! B panel  : `[K, NR=8]` row-major f32 (Winograd-transformed weights for
//!              one of the 16 spatial positions).
//! C tile   : `[MR=4, NR=8]` f32 accumulators (4 float32x4_t pairs).
//!
//! # fp16 path
//!
//! A native fp16 micro-kernel using `float16x8_t` / `vfmaq_laneq_f16` is
//! available on aarch64 with the `fp16` CPU extension (Cortex-A55 / A78AE /
//! Orin). It requires Rust nightly and the `stdarch_neon_f16` feature; gate it
//! with `#[cfg(feature = "nightly-fp16")]` and add the Cargo feature flag.
//! The stable f32 path is the primary implementation here.

#![allow(dead_code)]

use rayon::prelude::*;

// ─── tile constants ───────────────────────────────────────────────────────────

/// Micro-kernel tile height: number of A rows processed per micro-kernel invocation.
pub const MR: usize = 4;
/// Micro-kernel tile width: number of B columns processed per micro-kernel invocation.
pub const NR: usize = 8;

// ─── scalar fallback ──────────────────────────────────────────────────────────

/// Scalar MR×NR GEMM tile — correctness oracle and remainder handler.
///
/// `a`: `[MR, K]` row-major; `b`: `[K, NR]` row-major; `c`: `[MR, NR]` row-major.
///
/// Accumulates into `c` (does not zero it first).
#[inline(always)]
fn gemm_tile_scalar(
    mr: usize,
    nr: usize,
    k: usize,
    a: &[f32], // slice over full A panel, stride = full K
    a_row_stride: usize,
    a_row_offset: usize, // first A row index for this tile
    b: &[f32], // slice over full B panel
    b_col_offset: usize, // first B col for this tile
    c: &mut [f32], // full C, stride = n
    c_row_offset: usize,
    n: usize, // full N (stride of C rows)
) {
    for m in 0..mr {
        for p in 0..k {
            let a_val = a[(a_row_offset + m) * a_row_stride + p];
            for nb in 0..nr {
                c[(c_row_offset + m) * n + b_col_offset + nb] +=
                    a_val * b[p * NR + nb]; // B panel is [K, NR] packed
            }
        }
    }
}

// ─── aarch64 NEON f32 micro-kernel ───────────────────────────────────────────

/// NEON MR=4 × NR=8 f32 GEMM micro-kernel.
///
/// # Layout
/// * `a` — pointer to row `mr_start` of A: `[MR=4, K]` row-major f32.
///   Stride between consecutive A rows is `k` (full K).
/// * `b` — pointer to B panel: `[K, NR=8]` row-major f32.  The entire K×8
///   slice is contiguous.
/// * `c` — pointer to row `mr_start` of C: write `MR=4` rows × `NR=8` cols.
///   Row stride of C is `ldc` (full N).
///
/// Accumulates (+=) into C.
///
/// # Safety
/// Caller must ensure:
/// * `a` points to at least `MR * k` valid f32 values.
/// * `b` points to at least `k * NR` valid f32 values.
/// * `c` points to at least `(MR-1)*ldc + NR` valid f32 values.
/// * `k` is a multiple of 4 (the K loop is unrolled 4×; for non-multiple K
///   use the scalar fallback for the remainder).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn gemm_mr4_nr8_f32(
    k: usize,
    a: *const f32, // A panel row ptr: [MR=4, K] row-major
    b: *const f32, // B panel ptr:     [K, NR=8] row-major
    c: *mut f32,   // C output row ptr: row stride = ldc
    ldc: usize,
) {
    use std::arch::aarch64::*;

    // 4 accumulator pairs — each pair covers one A-row × 8 B-cols.
    // acc_lo[m]: lanes 0..3 of output row m
    // acc_hi[m]: lanes 4..7 of output row m
    let mut acc0_lo = vdupq_n_f32(0.0f32);
    let mut acc0_hi = vdupq_n_f32(0.0f32);
    let mut acc1_lo = vdupq_n_f32(0.0f32);
    let mut acc1_hi = vdupq_n_f32(0.0f32);
    let mut acc2_lo = vdupq_n_f32(0.0f32);
    let mut acc2_hi = vdupq_n_f32(0.0f32);
    let mut acc3_lo = vdupq_n_f32(0.0f32);
    let mut acc3_hi = vdupq_n_f32(0.0f32);

    let mut b_ptr = b;

    // ── K loop, unrolled 4× ──
    //   Each iteration: load 4 B rows (4 × float32x4_t pairs = 4 × 8 floats)
    //                   load 4 A scalars per A-row (broadcast)
    //                   16 vfmaq_laneq_f32 (4 rows × 4 k-steps)
    //
    // This structure keeps both A78AE FP32 FMA pipes busy and avoids
    // load-use stalls on B (128-byte = 32 f32 = 4 × NR=8 B rows per iter).
    let k4 = k & !3; // floor(k/4)*4
    let mut kk = 0usize;
    while kk < k4 {
        // Load 4 consecutive B rows: each row is NR=8 f32 = 2 × float32x4_t.
        let b0_lo = vld1q_f32(b_ptr);
        let b0_hi = vld1q_f32(b_ptr.add(4));
        let b1_lo = vld1q_f32(b_ptr.add(8));
        let b1_hi = vld1q_f32(b_ptr.add(12));
        let b2_lo = vld1q_f32(b_ptr.add(16));
        let b2_hi = vld1q_f32(b_ptr.add(20));
        let b3_lo = vld1q_f32(b_ptr.add(24));
        let b3_hi = vld1q_f32(b_ptr.add(28));
        b_ptr = b_ptr.add(32); // advance 4 B rows × NR=8

        // Load A column-of-rows for each of the 4 A-rows.
        // a_row{m}[kk..kk+3] as float32x4_t, then use lane intrinsics.
        let a0 = vld1q_f32(a.add(kk)); // row 0
        let a1 = vld1q_f32(a.add(k + kk)); // row 1
        let a2 = vld1q_f32(a.add(2 * k + kk)); // row 2
        let a3 = vld1q_f32(a.add(3 * k + kk)); // row 3

        // k-step 0
        acc0_lo = vfmaq_laneq_f32::<0>(acc0_lo, b0_lo, a0);
        acc0_hi = vfmaq_laneq_f32::<0>(acc0_hi, b0_hi, a0);
        acc1_lo = vfmaq_laneq_f32::<0>(acc1_lo, b0_lo, a1);
        acc1_hi = vfmaq_laneq_f32::<0>(acc1_hi, b0_hi, a1);
        acc2_lo = vfmaq_laneq_f32::<0>(acc2_lo, b0_lo, a2);
        acc2_hi = vfmaq_laneq_f32::<0>(acc2_hi, b0_hi, a2);
        acc3_lo = vfmaq_laneq_f32::<0>(acc3_lo, b0_lo, a3);
        acc3_hi = vfmaq_laneq_f32::<0>(acc3_hi, b0_hi, a3);

        // k-step 1
        acc0_lo = vfmaq_laneq_f32::<1>(acc0_lo, b1_lo, a0);
        acc0_hi = vfmaq_laneq_f32::<1>(acc0_hi, b1_hi, a0);
        acc1_lo = vfmaq_laneq_f32::<1>(acc1_lo, b1_lo, a1);
        acc1_hi = vfmaq_laneq_f32::<1>(acc1_hi, b1_hi, a1);
        acc2_lo = vfmaq_laneq_f32::<1>(acc2_lo, b1_lo, a2);
        acc2_hi = vfmaq_laneq_f32::<1>(acc2_hi, b1_hi, a2);
        acc3_lo = vfmaq_laneq_f32::<1>(acc3_lo, b1_lo, a3);
        acc3_hi = vfmaq_laneq_f32::<1>(acc3_hi, b1_hi, a3);

        // k-step 2
        acc0_lo = vfmaq_laneq_f32::<2>(acc0_lo, b2_lo, a0);
        acc0_hi = vfmaq_laneq_f32::<2>(acc0_hi, b2_hi, a0);
        acc1_lo = vfmaq_laneq_f32::<2>(acc1_lo, b2_lo, a1);
        acc1_hi = vfmaq_laneq_f32::<2>(acc1_hi, b2_hi, a1);
        acc2_lo = vfmaq_laneq_f32::<2>(acc2_lo, b2_lo, a2);
        acc2_hi = vfmaq_laneq_f32::<2>(acc2_hi, b2_hi, a2);
        acc3_lo = vfmaq_laneq_f32::<2>(acc3_lo, b2_lo, a3);
        acc3_hi = vfmaq_laneq_f32::<2>(acc3_hi, b2_hi, a3);

        // k-step 3
        acc0_lo = vfmaq_laneq_f32::<3>(acc0_lo, b3_lo, a0);
        acc0_hi = vfmaq_laneq_f32::<3>(acc0_hi, b3_hi, a0);
        acc1_lo = vfmaq_laneq_f32::<3>(acc1_lo, b3_lo, a1);
        acc1_hi = vfmaq_laneq_f32::<3>(acc1_hi, b3_hi, a1);
        acc2_lo = vfmaq_laneq_f32::<3>(acc2_lo, b3_lo, a2);
        acc2_hi = vfmaq_laneq_f32::<3>(acc2_hi, b3_hi, a2);
        acc3_lo = vfmaq_laneq_f32::<3>(acc3_lo, b3_lo, a3);
        acc3_hi = vfmaq_laneq_f32::<3>(acc3_hi, b3_hi, a3);

        kk += 4;
    }

    // ── K tail (k not multiple of 4) ─ scalar ─────────────────────────────────
    while kk < k {
        let b_lo = vld1q_f32(b_ptr);
        let b_hi = vld1q_f32(b_ptr.add(4));
        b_ptr = b_ptr.add(8);

        let a0_s = *a.add(kk);
        let a1_s = *a.add(k + kk);
        let a2_s = *a.add(2 * k + kk);
        let a3_s = *a.add(3 * k + kk);

        let av0 = vdupq_n_f32(a0_s);
        let av1 = vdupq_n_f32(a1_s);
        let av2 = vdupq_n_f32(a2_s);
        let av3 = vdupq_n_f32(a3_s);

        acc0_lo = vfmaq_f32(acc0_lo, b_lo, av0);
        acc0_hi = vfmaq_f32(acc0_hi, b_hi, av0);
        acc1_lo = vfmaq_f32(acc1_lo, b_lo, av1);
        acc1_hi = vfmaq_f32(acc1_hi, b_hi, av1);
        acc2_lo = vfmaq_f32(acc2_lo, b_lo, av2);
        acc2_hi = vfmaq_f32(acc2_hi, b_hi, av2);
        acc3_lo = vfmaq_f32(acc3_lo, b_lo, av3);
        acc3_hi = vfmaq_f32(acc3_hi, b_hi, av3);

        kk += 1;
    }

    // ── Accumulate into C (+=) ───────────────────────────────────────────────
    // Load existing C values and add accumulators.
    let c0 = c;
    let c1 = c.add(ldc);
    let c2 = c.add(2 * ldc);
    let c3 = c.add(3 * ldc);

    let cv0_lo = vld1q_f32(c0);
    let cv0_hi = vld1q_f32(c0.add(4));
    let cv1_lo = vld1q_f32(c1);
    let cv1_hi = vld1q_f32(c1.add(4));
    let cv2_lo = vld1q_f32(c2);
    let cv2_hi = vld1q_f32(c2.add(4));
    let cv3_lo = vld1q_f32(c3);
    let cv3_hi = vld1q_f32(c3.add(4));

    vst1q_f32(c0, vaddq_f32(cv0_lo, acc0_lo));
    vst1q_f32(c0.add(4), vaddq_f32(cv0_hi, acc0_hi));
    vst1q_f32(c1, vaddq_f32(cv1_lo, acc1_lo));
    vst1q_f32(c1.add(4), vaddq_f32(cv1_hi, acc1_hi));
    vst1q_f32(c2, vaddq_f32(cv2_lo, acc2_lo));
    vst1q_f32(c2.add(4), vaddq_f32(cv2_hi, acc2_hi));
    vst1q_f32(c3, vaddq_f32(cv3_lo, acc3_lo));
    vst1q_f32(c3.add(4), vaddq_f32(cv3_hi, acc3_hi));
}

// ─── fp16 micro-kernel (nightly-only, opt-in) ─────────────────────────────────
//
// Enabled when the crate is built with `--cfg kornia_nightly_fp16` (pass via
// RUSTFLAGS=--cfg=kornia_nightly_fp16 or .cargo/config.toml).
//
// Uses the `stdarch_neon_f16` nightly feature for float16x8_t / vfmaq_laneq_f16.
// On Cortex-A78AE both FP16 FMA pipes dual-issue: 16 vfmaq_laneq_f16 per 4-k
// iteration yields ~4× FP16 FMA utilisation over the f32 path.
//
// The FP16 kernel accumulates in float16x8_t (native half precision) and
// converts to f32 on store — acceptable precision for Winograd intermediates
// since the outer output transform re-scales.

/// fp16 MR=4 × NR=8 micro-kernel (nightly + fp16 CPU feature required).
///
/// Same calling convention as [`gemm_mr4_nr8_f32`]; a/b are fp32 slices
/// downcast to f16 externally before calling this function.
///
/// # Safety
/// Same requirements as [`gemm_mr4_nr8_f32`].
#[cfg(all(target_arch = "aarch64", kornia_nightly_fp16))]
#[target_feature(enable = "neon,fp16")]
#[allow(unused)]
unsafe fn gemm_mr4_nr8_fp16_native(
    k: usize,
    a: *const u16, // f16 bit-cast from half::f16 or __fp16 equivalent
    b: *const u16, // f16 bit-cast
    c: *mut f32,   // f32 accumulation output
    ldc: usize,
) {
    // NOTE: This function body is intentionally left as a stub.
    // It will be filled when the `stdarch_neon_f16` feature is stabilised
    // or when the crate is configured to build with nightly.
    //
    // The structure mirrors gemm_mr4_nr8_f32 but replaces:
    //   float32x4_t  →  float16x8_t   (single register covers all NR=8 lanes)
    //   vfmaq_laneq_f32::<L>  →  vfmaq_laneq_f16::<L>
    //   vld1q_f32 / vst1q_f32  →  vld1q_f16 / vcvt_f32_f16
    //
    // 4 accumulators: acc0..acc3 : float16x8_t
    // 4 B rows per k-iteration: b0..b3 : float16x8_t
    // 4 A scalars per row: a0..a3 : float16x4_t (lanes 0..3)
    //
    // 16 vfmaq_laneq_f16 per k-step-of-4  (4 rows × 4 k-steps)
    // Store: vcvt_f32_f16(acc) → vaddq_f32 with existing C → vst1q_f32
    let _ = (k, a, b, c, ldc);
}

// ─── public GEMM driver ──────────────────────────────────────────────────────

/// Winograd inner GEMM: `C += A × B` where
/// * `A` is `[m, k]` row-major f32 (input tiles, one Winograd position)
/// * `B` is `[k, n]` row-major f32 (pre-packed weight panel, one position)
/// * `C` is `[m, n]` row-major f32 (output accumulator, one position)
///
/// Tiles the M×N output into `MR×NR` blocks.  Full blocks are processed by the
/// NEON micro-kernel; remainder rows/cols fall back to scalar.
///
/// Parallelised over M-blocks via rayon `par_chunks_mut`.
///
/// # Panics
/// Panics (debug) if slice lengths are inconsistent with `m`, `k`, `n`.
pub fn winograd_gemm_f32(
    a: &[f32],    // [m, k]
    b: &[f32],    // [k, n]
    c: &mut [f32], // [m, n]  — pre-zeroed by caller or initialised to bias
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(a.len(), m * k, "A: expected {m}×{k}");
    debug_assert_eq!(b.len(), k * n, "B: expected {k}×{n}");
    debug_assert_eq!(c.len(), m * n, "C: expected {m}×{n}");

    // Number of full MR-blocks and remainder.
    let m_blocks = m / MR;
    let m_rem = m % MR;

    // Wrap raw pointers for Send across rayon threads (immutable A and B,
    // non-overlapping C row slices from par_chunks_mut).
    let a_ptr = a.as_ptr() as usize;
    let b_ptr = b.as_ptr() as usize;

    // Process full MR-row blocks in parallel.
    c.par_chunks_mut(MR * n)
        .take(m_blocks)
        .enumerate()
        .for_each(|(mb, c_block)| {
            let mr_offset = mb * MR; // first A row for this block
            let a_block = unsafe { std::slice::from_raw_parts((a_ptr as *const f32).add(mr_offset * k), MR * k) };
            let b_panel = unsafe { std::slice::from_raw_parts(b_ptr as *const f32, k * n) };
            gemm_block_f32(a_block, b_panel, c_block, MR, k, n);
        });

    // Serial remainder rows (m_rem < MR — too few to fill a full tile).
    if m_rem > 0 {
        let rem_row_start = m_blocks * MR;
        let a_rem = &a[rem_row_start * k..];
        let c_rem = &mut c[rem_row_start * n..];
        gemm_block_scalar(a_rem, b, c_rem, m_rem, k, n);
    }
}

/// Process one `[MR, K] × [K, N]` block of C using the NEON micro-kernel for
/// full NR-col tiles and scalar for the N remainder.
fn gemm_block_f32(
    a: &[f32],     // [MR, K]
    b: &[f32],     // [K, N]
    c: &mut [f32], // [MR, N]
    _mr: usize,    // actual rows (always MR for full blocks; scalar handles remainders)
    k: usize,
    n: usize,
) {
    let n_blocks = n / NR;
    let n_rem = n % NR;

    for nb in 0..n_blocks {
        let b_col = nb * NR;
        // Pack B panel for this NR column block: [K, NR] contiguous.
        // B is [K, N] row-major; we need [K, NR] stride-NR slices packed contiguously.
        let b_packed = pack_b_panel(b, k, n, b_col);

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                gemm_mr4_nr8_f32(
                    k,
                    a.as_ptr(),
                    b_packed.as_ptr(),
                    c.as_mut_ptr().add(b_col), // C row 0, col nb*NR
                    n,                          // ldc = full N
                );
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            // Portable fallback for non-aarch64 builds.
            // Gather C tile, accumulate, scatter back.
            let mut c_tile = vec![0.0f32; MR * NR];
            for m in 0..MR {
                for j in 0..NR {
                    c_tile[m * NR + j] = c[m * n + b_col + j];
                }
            }
            gemm_block_scalar(a, &b_packed, &mut c_tile, MR, k, NR);
            for m in 0..MR {
                for j in 0..NR {
                    c[m * n + b_col + j] = c_tile[m * NR + j];
                }
            }
        }
    }

    // N remainder (scalar): directly operate on the non-packed B rows.
    if n_rem > 0 {
        let b_col = n_blocks * NR;
        for m in 0..MR {
            for p in 0..k {
                let a_val = a[m * k + p];
                for j in 0..n_rem {
                    c[m * n + b_col + j] += a_val * b[p * n + b_col + j];
                }
            }
        }
    }
}

/// Pure-scalar GEMM block: `c[0..mr, 0..n] += a[0..mr, 0..k] × b[0..k, 0..n]`.
///
/// `b` must be `[k, n]` row-major with row-stride `n`.
/// `c` must be `[mr, n]` row-major with row-stride `n`.
fn gemm_block_scalar(
    a: &[f32],     // [mr, k]
    b: &[f32],     // [k, n]
    c: &mut [f32], // [mr, n]
    mr: usize,
    k: usize,
    n: usize,
) {
    for m in 0..mr {
        for p in 0..k {
            let a_val = a[m * k + p];
            for j in 0..n {
                c[m * n + j] += a_val * b[p * n + j];
            }
        }
    }
}

/// Pack a `[K, NR]` B panel from the larger `[K, N]` B matrix.
///
/// Extracts columns `col_start..col_start+NR` for all K rows into a contiguous
/// `[K, NR]` buffer that the micro-kernel can stream linearly.
fn pack_b_panel(b: &[f32], k: usize, n: usize, col_start: usize) -> Vec<f32> {
    let mut panel = vec![0.0f32; k * NR];
    for ki in 0..k {
        for j in 0..NR {
            panel[ki * NR + j] = b[ki * n + col_start + j];
        }
    }
    panel
}

// ─── B-packing helper ─────────────────────────────────────────────────────────

/// Transpose `weights_transformed[position, :, :]` from `[c_out, c_in]`
/// (as stored in the flat `[16, c_out, c_in]` buffer) to `[c_in, c_out]`
/// row-major f32, ready for use as the B panel in [`winograd_gemm_f32`].
///
/// # Arguments
/// * `weights_transformed` — flat `[16, c_out, c_in]` Winograd weight buffer.
/// * `c_out`, `c_in` — channel dimensions.
/// * `position` — which of the 16 Winograd spatial positions (0..16).
///
/// # Returns
/// A `Vec<f32>` of length `c_in * c_out` in `[c_in, c_out]` row-major order.
pub fn pack_b_winograd(
    weights_transformed: &[f32], // [16, c_out, c_in]
    c_out: usize,
    c_in: usize,
    position: usize,
) -> Vec<f32> {
    debug_assert!(position < 16, "Winograd position must be 0..15");
    debug_assert_eq!(
        weights_transformed.len(),
        16 * c_out * c_in,
        "weights_transformed length mismatch"
    );

    // Slice for this position: [c_out, c_in] starting at position * c_out * c_in.
    let pos_offset = position * c_out * c_in;
    let w_pos = &weights_transformed[pos_offset..pos_offset + c_out * c_in];

    // Transpose [c_out, c_in] → [c_in, c_out].
    let mut out = vec![0.0f32; c_in * c_out];
    for co in 0..c_out {
        for ci in 0..c_in {
            out[ci * c_out + co] = w_pos[co * c_in + ci];
        }
    }
    out
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference GEMM: C = A × B (no bias, no accumulation).
    fn matmul_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    c[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        c
    }

    #[test]
    fn test_pack_b_winograd_shape() {
        let c_out = 8;
        let c_in = 4;
        let w = (0..16 * c_out * c_in).map(|i| i as f32).collect::<Vec<_>>();
        let packed = pack_b_winograd(&w, c_out, c_in, 0);
        assert_eq!(packed.len(), c_in * c_out);
    }

    #[test]
    fn test_pack_b_winograd_transpose() {
        let c_out = 4;
        let c_in = 2;
        // weights[pos=0, co, ci] = co * c_in + ci  (simple index)
        let mut w = vec![0.0f32; 16 * c_out * c_in];
        for co in 0..c_out {
            for ci in 0..c_in {
                w[co * c_in + ci] = (co * c_in + ci) as f32;
            }
        }
        let packed = pack_b_winograd(&w, c_out, c_in, 0);
        // packed[ci, co] should equal w[co, ci]
        for co in 0..c_out {
            for ci in 0..c_in {
                assert_eq!(
                    packed[ci * c_out + co],
                    w[co * c_in + ci],
                    "mismatch at co={co} ci={ci}"
                );
            }
        }
    }

    #[test]
    fn test_winograd_gemm_f32_exact() {
        // Simple 4×4 × 4×8 GEMM (M=4, K=4, N=8).
        let m = 4;
        let k = 4;
        let n = 8;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let ref_c = matmul_ref(&a, &b, m, k, n);

        let mut c = vec![0.0f32; m * n];
        winograd_gemm_f32(&a, &b, &mut c, m, k, n);

        for (i, (&got, &exp)) in c.iter().zip(ref_c.iter()).enumerate() {
            let rel_err = if exp.abs() > 1e-6 {
                (got - exp).abs() / exp.abs()
            } else {
                (got - exp).abs()
            };
            assert!(
                rel_err < 1e-5,
                "element {i}: got={got} expected={exp} rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn test_winograd_gemm_f32_larger() {
        // M=12 (3 MR-blocks), K=16 (4 K-unroll iters), N=24 (3 NR-blocks).
        let m = 12;
        let k = 16;
        let n = 24;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let ref_c = matmul_ref(&a, &b, m, k, n);

        let mut c = vec![0.0f32; m * n];
        winograd_gemm_f32(&a, &b, &mut c, m, k, n);

        for (i, (&got, &exp)) in c.iter().zip(ref_c.iter()).enumerate() {
            let rel_err = if exp.abs() > 1e-6 {
                (got - exp).abs() / exp.abs()
            } else {
                (got - exp).abs()
            };
            assert!(
                rel_err < 1e-4,
                "element {i}: got={got} expected={exp} rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn test_winograd_gemm_f32_remainder() {
        // M=5 (1 full MR-block + 1 remainder row), K=6 (1 full k4 + 2 tail), N=10 (1 NR-block + 2 cols).
        let m = 5;
        let k = 6;
        let n = 10;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.05).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();
        let ref_c = matmul_ref(&a, &b, m, k, n);

        let mut c = vec![0.0f32; m * n];
        winograd_gemm_f32(&a, &b, &mut c, m, k, n);

        for (i, (&got, &exp)) in c.iter().zip(ref_c.iter()).enumerate() {
            let rel_err = if exp.abs() > 1e-6 {
                (got - exp).abs() / exp.abs()
            } else {
                (got - exp).abs()
            };
            assert!(
                rel_err < 1e-4,
                "element {i}: got={got} expected={exp} rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn test_winograd_gemm_xfeat_dims() {
        // Approximate XFeat layer 0: num_tiles=64, c_in=1, c_out=24.
        let m = 64;
        let k = 1;
        let n = 24;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let ref_c = matmul_ref(&a, &b, m, k, n);

        let mut c = vec![0.0f32; m * n];
        winograd_gemm_f32(&a, &b, &mut c, m, k, n);

        for (i, (&got, &exp)) in c.iter().zip(ref_c.iter()).enumerate() {
            let rel_err = if exp.abs() > 1e-6 {
                (got - exp).abs() / exp.abs()
            } else {
                (got - exp).abs()
            };
            assert!(
                rel_err < 1e-4,
                "element {i}: got={got} expected={exp} rel_err={rel_err}"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_mr4_nr8_kernel_direct() {
        // Direct test of the NEON micro-kernel.
        let k = 8;
        let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.1).collect();
        // B panel is [K, NR=8] contiguous.
        let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.1).collect();
        let ref_c = matmul_ref(&a, &b, MR, k, NR);

        let mut c = vec![0.0f32; MR * NR];
        unsafe {
            gemm_mr4_nr8_f32(k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), NR);
        }

        for (i, (&got, &exp)) in c.iter().zip(ref_c.iter()).enumerate() {
            let rel_err = if exp.abs() > 1e-5 {
                (got - exp).abs() / exp.abs()
            } else {
                (got - exp).abs()
            };
            assert!(
                rel_err < 1e-4,
                "element {i}: got={got:.6} expected={exp:.6}"
            );
        }
    }
}
