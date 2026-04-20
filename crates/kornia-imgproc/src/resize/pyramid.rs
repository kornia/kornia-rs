//! Exact-2× integer-ratio fast paths for bilinear resize.
//!
//! These kernels exploit the fact that a 2× scale with pixel-center sampling
//! collapses the bilinear weights to fixed `{0.25, 0.75}` (upscale) or
//! `{0.25, 0.25, 0.25, 0.25}` (downscale) patterns — no LUT, no fractional
//! arithmetic, no gather. Each outer driver parallelizes over output row
//! groups and dispatches to the row-level kernels in [`super::kernels`].

use rayon::prelude::*;

use super::kernels::{blend_75_25_row, hinterp_row_rgb_u8, pyrdown_row_rgb_u8};

/// 2× box-averaging downsample for RGB u8.
///
/// Equivalent to bilinear at exact 2:1 downscale and ~4× faster on aarch64
/// thanks to the dedicated NEON row kernel. Groups 8 output rows per rayon
/// task so small strides (e.g. 540p = 2.8 KB/row) amortize spawn overhead.
pub(super) fn pyrdown_2x_rgb_u8(src: &[u8], dst: &mut [u8], src_w: usize, _src_h: usize) {
    let dst_w = src_w / 2;
    let src_stride = src_w * 3;
    let dst_stride = dst_w * 3;

    const ROWS_PER_TASK: usize = 8;
    let chunk_bytes = dst_stride * ROWS_PER_TASK;

    dst.par_chunks_mut(chunk_bytes)
        .enumerate()
        .for_each(|(ti, dst_chunk)| {
            let y_base = ti * ROWS_PER_TASK;
            let nrows = dst_chunk.len() / dst_stride;
            for dy in 0..nrows {
                let y = y_base + dy;
                let r0 = &src[(2 * y) * src_stride..(2 * y + 1) * src_stride];
                let r1 = &src[(2 * y + 1) * src_stride..(2 * y + 2) * src_stride];
                let dst_row = &mut dst_chunk[dy * dst_stride..(dy + 1) * dst_stride];
                pyrdown_row_rgb_u8(r0, r1, dst_row, dst_w);
            }
        });
}

/// 2× exact bilinear upscale for RGB u8.
///
/// With pixel-center sampling (`sx = (x + 0.5) * 0.5 - 0.5`) the per-pixel
/// weights collapse to a fixed `{0.25, 0.75}` pattern: each dst row is either
/// a horizontally-interpolated edge row (first/last) or a `0.75 / 0.25` blend
/// of two horizontally-interpolated neighbour rows. The 75/25 blend is
/// computed via a `vrhaddq_u8(a, vrhaddq_u8(a, b))` pair — no fractional
/// arithmetic, no LUT, no gather. One source row produces two output rows
/// sharing the same horizontal-interpolation cost.
pub(super) fn pyrup_2x_rgb_u8(src: &[u8], dst: &mut [u8], src_w: usize, src_h: usize) {
    let dst_w = src_w * 2;
    let src_stride = src_w * 3;
    let dst_stride = dst_w * 3;

    // Layout: [edge_top (1 row) | inner (2·(src_h-1) rows) | edge_bot (1 row)].
    let (edge_top, rest) = dst.split_at_mut(dst_stride);
    let (inner, edge_bot) = rest.split_at_mut(2 * (src_h - 1) * dst_stride);

    // Edge rows come from a single source row (clamped f=0/1 in the vertical
    // direction), so they only need the horizontal pass.
    hinterp_row_rgb_u8(&src[..src_stride], edge_top, src_w);
    hinterp_row_rgb_u8(
        &src[(src_h - 1) * src_stride..src_h * src_stride],
        edge_bot,
        src_w,
    );

    // Inner blocks: block I consumes src rows (I, I+1) and writes dst rows
    // (2I+1, 2I+2). Group 64 blocks (128 dst rows) per rayon task. At
    // 1080p→2160p that's ~2.9 MB per task — well above the ~10 KB rayon
    // dispatch threshold and below L2 (3 MB on A78AE).
    const BLOCKS_PER_TASK: usize = 64;
    let chunk_bytes = BLOCKS_PER_TASK * 2 * dst_stride;

    inner
        .par_chunks_mut(chunk_bytes)
        .enumerate()
        .for_each(|(ti, inner_chunk)| {
            let block_start = ti * BLOCKS_PER_TASK;
            let blocks = inner_chunk.len() / (2 * dst_stride);

            // Rolling 2-row scratch: horizontally-upscaled "prev" (h_a) and
            // "next" (h_b) source rows. Both sit in L1 (~23 KB each at 1080p).
            let mut h_a = vec![0u8; dst_stride];
            let mut h_b = vec![0u8; dst_stride];

            hinterp_row_rgb_u8(
                &src[block_start * src_stride..(block_start + 1) * src_stride],
                &mut h_a,
                src_w,
            );

            for i in 0..blocks {
                let src_next = block_start + i + 1;
                hinterp_row_rgb_u8(
                    &src[src_next * src_stride..(src_next + 1) * src_stride],
                    &mut h_b,
                    src_w,
                );

                let (dst_top, dst_bot) = inner_chunk
                    [i * 2 * dst_stride..(i + 1) * 2 * dst_stride]
                    .split_at_mut(dst_stride);

                // dst row 2I+1 = 0.75·h_a + 0.25·h_b
                // dst row 2I+2 = 0.25·h_a + 0.75·h_b  (same blend, args swapped)
                blend_75_25_row(&h_a, &h_b, dst_top);
                blend_75_25_row(&h_b, &h_a, dst_bot);

                std::mem::swap(&mut h_a, &mut h_b);
            }
        });
}
