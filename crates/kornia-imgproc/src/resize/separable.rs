//! Q14 separable bicubic / lanczos resize.
//!
//! Two passes over the image with an `i16` intermediate buffer:
//!   1. **Horizontal** — for each source row, convolve with the per-dst-col
//!      coefficient LUT to produce `dst_w` i16 intermediates per row.
//!   2. **Vertical**   — for each destination row, reduce `ky` i16 rows with
//!      the per-dst-row coefficient LUT and narrow/saturate to u8.
//!
//! The module picks one of two orchestrations based on whether the per-thread
//! horizontal buffer fits L2:
//!   - **Global two-pass** (`resize_global_two_pass`): cheap for small dst_w
//!     where the whole intermediate fits L2 per thread.
//!   - **Strip-fused H→V** (the `else` branch of `resize_separable_u8`): for
//!     larger outputs, process a vertical strip at a time so `hbuf` stays
//!     L2-hot across H and V.

use rayon::prelude::*;

use super::common::{build_xsrc_lut, pack_xw_i16, precompute_contribs, FilterKind};
use super::kernels::{horizontal_row_rgb_u8, vertical_row};

/// Run the horizontal pass over a contiguous run of source rows, writing
/// i16 intermediate rows into `out`. Uses the 4-row NEON kernel for C=3 to
/// amortize coefficient loads across four src rows; everything else falls
/// through to the per-row kernel.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn horizontal_batch<const C: usize>(
    src: &[u8],
    src_stride: usize,
    out: &mut [i16],
    hbuf_row_len: usize,
    sy_start: usize,
    dst_w: usize,
    kx: usize,
    xsrc: &[u16],
    xw: &[i16],
    last_sx_safe: usize,
    round1: i32,
) {
    let n_rows = out.len() / hbuf_row_len;
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    {
        if C == 3 {
            while i + 4 <= n_rows {
                let sy0 = sy_start + i;
                let s0 = &src[sy0 * src_stride..(sy0 + 1) * src_stride];
                let s1 = &src[(sy0 + 1) * src_stride..(sy0 + 2) * src_stride];
                let s2 = &src[(sy0 + 2) * src_stride..(sy0 + 3) * src_stride];
                let s3 = &src[(sy0 + 3) * src_stride..(sy0 + 4) * src_stride];
                let (out_head, out_rest) = out[i * hbuf_row_len..].split_at_mut(hbuf_row_len);
                let (out_1, out_rest) = out_rest.split_at_mut(hbuf_row_len);
                let (out_2, out_rest) = out_rest.split_at_mut(hbuf_row_len);
                let (out_3, _) = out_rest.split_at_mut(hbuf_row_len);
                unsafe {
                    super::kernels::horizontal_rows_rgb_u8_x4(
                        [s0, s1, s2, s3],
                        [out_head, out_1, out_2, out_3],
                        dst_w,
                        kx,
                        xsrc,
                        xw,
                        last_sx_safe,
                        round1,
                    );
                }
                i += 4;
            }
        }
    }
    while i < n_rows {
        let sy = sy_start + i;
        let src_row = &src[sy * src_stride..(sy + 1) * src_stride];
        let out_row = &mut out[i * hbuf_row_len..(i + 1) * hbuf_row_len];
        horizontal_row_rgb_u8::<C>(
            src_row,
            out_row,
            dst_w,
            kx,
            xsrc,
            xw,
            last_sx_safe,
            round1,
        );
        i += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn resize_global_two_pass<const C: usize>(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
    xsrc: &[u16],
    xw: &[i16],
    kx: usize,
    yofs: &[i32],
    yw: &[i32],
    ky: usize,
    last_sx_safe: usize,
    round1: i32,
    round2: i32,
) {
    let _ = dst_h;
    let src_stride = src_w * C;
    let dst_stride = dst_w * C;
    let hbuf_row_len = dst_w * C;
    let mut hbuf = vec![0i16; src_h * hbuf_row_len];
    const H_BATCH: usize = 16;
    hbuf.par_chunks_mut(hbuf_row_len * H_BATCH)
        .enumerate()
        .for_each(|(bi, batch_out)| {
            let sy0 = bi * H_BATCH;
            horizontal_batch::<C>(
                src,
                src_stride,
                batch_out,
                hbuf_row_len,
                sy0,
                dst_w,
                kx,
                xsrc,
                xw,
                last_sx_safe,
                round1,
            );
        });

    const V_BATCH: usize = 8;
    dst.par_chunks_mut(dst_stride * V_BATCH)
        .enumerate()
        .for_each(|(bi, batch_dst)| {
            let yo0 = bi * V_BATCH;
            let batch_rows = batch_dst.len() / dst_stride;
            let n = dst_w * C;
            let mut rows: Vec<&[i16]> = Vec::with_capacity(ky);
            let mut w: Vec<i16> = Vec::with_capacity(ky);
            for r in 0..batch_rows {
                let yo = yo0 + r;
                let y0 = yofs[yo];
                rows.clear();
                w.clear();
                for k in 0..ky {
                    let sy = (y0 + k as i32).clamp(0, src_h as i32 - 1) as usize;
                    rows.push(&hbuf[sy * hbuf_row_len..(sy + 1) * hbuf_row_len]);
                    w.push(yw[yo * ky + k] as i16);
                }
                let dst_row = &mut batch_dst[r * dst_stride..(r + 1) * dst_stride];
                vertical_row(&rows, &w, dst_row, n, round2);
            }
        });
}

/// Q14 separable bicubic / lanczos resize. Top-level entry point for the
/// separable pipeline; picks between the global two-pass and strip-fused
/// orchestrations based on L2 capacity.
#[allow(clippy::too_many_arguments)]
pub(super) fn resize_separable_u8<const C: usize>(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
    filt: FilterKind,
    antialias: bool,
) {
    const Q: i32 = 14;

    let (xofs, xw, kx) = precompute_contribs(src_w, dst_w, filt, antialias);
    let (yofs, yw, ky) = precompute_contribs(src_h, dst_h, filt, antialias);
    let src_stride = src_w * C;
    let dst_stride = dst_w * C;

    let xsrc = build_xsrc_lut(&xofs, dst_w, kx, src_w);
    // Pack weights i32→i16 once so the horizontal inner loop can issue
    // `vmlal_n_s16` directly without per-iter casts. Q14 coeffs have ≤ 16384
    // peak magnitude so they fit i16 with headroom.
    let xw = pack_xw_i16(&xw);
    let last_sx_safe = src_w.saturating_sub(1);

    let hbuf_row_len = dst_w * C;
    let round1: i32 = 1 << (Q - 1);
    let round2: i32 = 1 << (Q - 1);
    let nthreads = rayon::current_num_threads().max(1);
    let per_row = hbuf_row_len * 2;

    // Per-thread hbuf slice (global plan) = (src_h / nthreads) * per_row.
    // If that fits private L2 (~256KB), the global two-pass plan wins because
    // strip fusion adds (dst_h / strip_h * ky) rows of overlap overhead.
    //
    // If the per-thread slice overflows L2, switch to strip-fused execution
    // with tile sizes chosen so the local hbuf stays in L2.
    let per_thread_global = src_h.div_ceil(nthreads) * per_row;
    let l2_target = 192 * 1024;
    let use_strips = per_thread_global > l2_target;

    if !use_strips {
        // Global two-pass: cheap for small dst_w where per-thread hbuf fits L2.
        resize_global_two_pass::<C>(
            src,
            src_w,
            src_h,
            dst,
            dst_w,
            dst_h,
            &xsrc,
            &xw,
            kx,
            &yofs,
            &yw,
            ky,
            last_sx_safe,
            round1,
            round2,
        );
        return;
    }

    // Strip-fused H→V: bound strip_h so (strip_h*scale_y + ky) * per_row ≤ L2.
    let scale_y_q8 = ((src_h as u64) << 8) / (dst_h.max(1) as u64);
    let band_cap = (l2_target / per_row.max(1)).saturating_sub(ky);
    let strip_h_mem = ((band_cap as u64) << 8) / scale_y_q8.max(1);
    let strip_h_par = dst_h.div_ceil(nthreads);
    let strip_h = (strip_h_mem as usize)
        .min(strip_h_par.max(8))
        .min(dst_h)
        .max(1);

    let mut strip_slices: Vec<&mut [u8]> = dst.chunks_mut(strip_h * dst_stride).collect();
    let strips: Vec<(usize, usize)> = (0..dst_h)
        .step_by(strip_h)
        .map(|s| (s, (s + strip_h).min(dst_h)))
        .collect();

    strip_slices.par_iter_mut().zip(strips.par_iter()).for_each(
        |(strip_dst, &(yo_start, yo_end))| {
            let n = dst_w * C;

            // Source-row span needed for this strip: [sy_min, sy_max).
            let mut sy_min = i32::MAX;
            let mut sy_max = i32::MIN;
            for &y0 in &yofs[yo_start..yo_end] {
                sy_min = sy_min.min(y0);
                sy_max = sy_max.max(y0 + ky as i32);
            }
            // Clamp to image bounds for allocation; border replication is
            // handled inside the per-row clamp below.
            let sy_span_start = sy_min.max(0) as usize;
            let sy_span_end = sy_max.min(src_h as i32) as usize;
            let band_rows = sy_span_end.saturating_sub(sy_span_start);
            if band_rows == 0 {
                return;
            }

            // Local hbuf: exactly the rows this strip consumes. Unlike the
            // global version, this stays in L1/L2 across H and V.
            let mut temp: Vec<i16> = Vec::with_capacity(band_rows * hbuf_row_len);
            // SAFETY: horizontal pass writes every element before any read.
            #[allow(clippy::uninit_vec)]
            unsafe {
                temp.set_len(band_rows * hbuf_row_len)
            };

            // Horizontal pass over the needed src rows only. Processes rows
            // in groups of 4 to share coefficient (xsrc/xw) loads and fill
            // the MAC pipelines.
            horizontal_batch::<C>(
                src,
                src_stride,
                &mut temp,
                hbuf_row_len,
                sy_span_start,
                dst_w,
                kx,
                &xsrc,
                &xw,
                last_sx_safe,
                round1,
            );

            // Vertical pass: consume local hbuf (L1/L2-hot) straight to output.
            let mut rows: Vec<&[i16]> = Vec::with_capacity(ky);
            let mut w: Vec<i16> = Vec::with_capacity(ky);
            for (oi, yo) in (yo_start..yo_end).enumerate() {
                let y0 = yofs[yo];
                rows.clear();
                w.clear();
                for k in 0..ky {
                    // Map global src index to band-local. Rows outside the
                    // band (edge replication) clamp to the nearest kept row.
                    let sy_global = (y0 + k as i32).clamp(0, src_h as i32 - 1) as usize;
                    let sy_local = sy_global.min(sy_span_end - 1).saturating_sub(sy_span_start);
                    rows.push(&temp[sy_local * hbuf_row_len..(sy_local + 1) * hbuf_row_len]);
                    w.push(yw[yo * ky + k] as i16);
                }
                let dst_row = &mut strip_dst[oi * dst_stride..(oi + 1) * dst_stride];
                vertical_row(&rows, &w, dst_row, n, round2);
            }
        },
    );
}
