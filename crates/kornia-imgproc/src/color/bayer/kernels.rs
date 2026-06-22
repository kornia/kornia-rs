//! Bayer demosaic kernels (u8, bilinear) — scalar oracle + NEON.
//!
//! The mosaic is interpreted via the per-cell color phase derived from the
//! [`BayerPattern`]. Given the color at `(row%2, col%2)`, every interior pixel
//! reconstructs its two missing channels with rounded integer bilinear
//! interpolation (OpenCV-compatible):
//!
//! - at an R location → G = avg4(N,S,E,W), B = avg4(diagonal corners)
//! - at a B location → G = avg4(N,S,E,W), R = avg4(diagonal corners)
//! - at a G location on an R row → R = avg2(W,E), B = avg2(N,S)
//! - at a G location on a B row → B = avg2(W,E), R = avg2(N,S)
//!
//! Rounded averages: `avg2 = (a+b+1)>>1`, `avg4 = (a+b+c+d+2)>>2`.
//! Borders use replicate (clamp-to-edge) addressing.

use kornia_image::color_spaces::BayerPattern;

/// The color a sensel carries, resolved from the pattern + pixel parity.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cell {
    R,
    GonRRow,
    GonBRow,
    B,
}

/// Resolve the per-cell color layout into a 2×2 phase table indexed by
/// `[row & 1][col & 1]`.
#[inline]
fn phase_table(pattern: BayerPattern) -> [[Cell; 2]; 2] {
    use Cell::*;
    match pattern {
        // R G / G B  → row0 has R, row1 has B
        BayerPattern::Rggb => [[R, GonRRow], [GonBRow, B]],
        // B G / G R  → row0 has B, row1 has R
        BayerPattern::Bggr => [[B, GonBRow], [GonRRow, R]],
        // G R / B G  → row0 has R, row1 has B
        BayerPattern::Grbg => [[GonRRow, R], [B, GonBRow]],
        // G B / R G  → row0 has B, row1 has R
        BayerPattern::Gbrg => [[GonBRow, B], [R, GonRRow]],
    }
}

#[inline(always)]
fn avg2(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

#[inline(always)]
fn avg4(a: u8, b: u8, c: u8, d: u8) -> u8 {
    ((a as u16 + b as u16 + c as u16 + d as u16 + 2) >> 2) as u8
}

/// Demosaic a single output pixel `(r, c)` with replicate-border addressing.
/// Used for the scalar oracle and for the borders/remainder of the NEON path.
#[inline]
fn demosaic_px(
    src: &[u8],
    dst: &mut [u8],
    r: usize,
    c: usize,
    rows: usize,
    cols: usize,
    table: &[[Cell; 2]; 2],
) {
    let at = |r: isize, c: isize| -> u8 {
        let rr = r.clamp(0, rows as isize - 1) as usize;
        let cc = c.clamp(0, cols as isize - 1) as usize;
        src[rr * cols + cc]
    };
    let ri = r as isize;
    let ci = c as isize;
    let center = src[r * cols + c];
    let cell = table[r & 1][c & 1];
    let (red, green, blue) = match cell {
        Cell::R => {
            let g = avg4(
                at(ri - 1, ci),
                at(ri + 1, ci),
                at(ri, ci - 1),
                at(ri, ci + 1),
            );
            let b = avg4(
                at(ri - 1, ci - 1),
                at(ri - 1, ci + 1),
                at(ri + 1, ci - 1),
                at(ri + 1, ci + 1),
            );
            (center, g, b)
        }
        Cell::B => {
            let g = avg4(
                at(ri - 1, ci),
                at(ri + 1, ci),
                at(ri, ci - 1),
                at(ri, ci + 1),
            );
            let rr = avg4(
                at(ri - 1, ci - 1),
                at(ri - 1, ci + 1),
                at(ri + 1, ci - 1),
                at(ri + 1, ci + 1),
            );
            (rr, g, center)
        }
        Cell::GonRRow => {
            let rr = avg2(at(ri, ci - 1), at(ri, ci + 1));
            let b = avg2(at(ri - 1, ci), at(ri + 1, ci));
            (rr, center, b)
        }
        Cell::GonBRow => {
            let b = avg2(at(ri, ci - 1), at(ri, ci + 1));
            let rr = avg2(at(ri - 1, ci), at(ri + 1, ci));
            (rr, center, b)
        }
    };
    let o = (r * cols + c) * 3;
    dst[o] = red;
    dst[o + 1] = green;
    dst[o + 2] = blue;
}

/// Scalar Bayer demosaic — the reference/oracle. Replicate-border addressing.
///
/// `src` is `rows*cols` bytes; `dst` is `rows*cols*3` interleaved RGB bytes.
pub fn rgb_from_bayer_scalar(
    src: &[u8],
    dst: &mut [u8],
    rows: usize,
    cols: usize,
    pattern: BayerPattern,
) {
    debug_assert!(src.len() >= rows * cols);
    debug_assert!(dst.len() >= rows * cols * 3);
    if rows == 0 || cols == 0 {
        return;
    }
    let table = phase_table(pattern);
    for r in 0..rows {
        for c in 0..cols {
            demosaic_px(src, dst, r, c, rows, cols, &table);
        }
    }
}

/// Public entry: dispatch to NEON on aarch64, scalar elsewhere.
///
/// Interior rows/cols use the SIMD kernel on aarch64; the 1-px replicate border
/// is always handled by the scalar path so both produce bit-identical output.
pub fn rgb_from_bayer_dispatch(
    src: &[u8],
    dst: &mut [u8],
    rows: usize,
    cols: usize,
    pattern: BayerPattern,
) {
    #[cfg(target_arch = "aarch64")]
    {
        rgb_from_bayer_neon(src, dst, rows, cols, pattern);
        return;
    }
    #[allow(unreachable_code)]
    rgb_from_bayer_scalar(src, dst, rows, cols, pattern);
}

/// NEON Bayer demosaic.
///
/// Strategy: the four border lines (top/bottom row, left/right column) are done
/// by the scalar oracle so replicate-edge handling is shared and exact. The
/// interior (`1..rows-1` × `1..cols-1`) is processed two rows at a time. Each
/// interior row pair is split into even/odd columns with `vld2q_u8` so the two
/// mosaic phases land in separate registers; the 2-neighbor cases use
/// `vrhaddq_u8` (rounded halving add) and the 4-neighbor cases use widening
/// `vaddl_u8` + `vrshrn_n_u16(_, 2)`. Results re-interleave with `vst3q_u8`.
///
/// For images too small to have an interior (rows < 3 or cols < 3) we fall back
/// to the scalar path entirely.
#[cfg(target_arch = "aarch64")]
pub fn rgb_from_bayer_neon(
    src: &[u8],
    dst: &mut [u8],
    rows: usize,
    cols: usize,
    pattern: BayerPattern,
) {
    // Tiny images: no interior to vectorize.
    if rows < 3 || cols < 3 {
        rgb_from_bayer_scalar(src, dst, rows, cols, pattern);
        return;
    }

    use std::arch::aarch64::*;
    let table = phase_table(pattern);

    // Scalar only where NEON can't reach: the top/bottom border rows. Interior rows
    // get NEON for the bulk and a scalar fill for the left/right edges + remainder.
    // (The old code scalar-filled the *whole* image first — O(N) and ~16× slower.)
    for cc in 0..cols {
        demosaic_px(src, dst, 0, cc, rows, cols, &table);
        demosaic_px(src, dst, rows - 1, cc, rows, cols, &table);
    }

    // SIMD interior columns: process 32-wide blocks of the even/odd split. Each
    // vld2q_u8 covers 32 source columns (16 even + 16 odd). We require a full
    // interior block, so the per-row column span is [1, cols-1).
    unsafe {
        for r in 1..rows - 1 {
            let row_off = r * cols;
            // Left border (col 0) — scalar (replicate edge).
            demosaic_px(src, dst, r, 0, rows, cols, &table);
            // Phase of the colored sensel on this interior row at column 0.
            // We iterate per-pixel-pair via the even/odd vld2 split, but the
            // simplest correct-and-fast structure here is a column loop with
            // 16-pixel NEON blocks aligned so we never touch the borders.
            //
            // We compute for columns c in [1, cols-1). Start at the first even
            // multiple that keeps a 16-wide window inside the interior.
            // One 16-pixel NEON block at interior column `cc`. The 4-neighbor cases
            // need true (a+b+c+d+2)>>2 (nested vrhadd is not bit-exact) via the
            // widening avg4_u8; the 2-neighbor cases are exact with vrhaddq_u8.
            let mut blk = |cc: usize| {
                let center = vld1q_u8(src.as_ptr().add(row_off + cc));
                let west = vld1q_u8(src.as_ptr().add(row_off + cc - 1));
                let east = vld1q_u8(src.as_ptr().add(row_off + cc + 1));
                let north = vld1q_u8(src.as_ptr().add(row_off - cols + cc));
                let south = vld1q_u8(src.as_ptr().add(row_off + cols + cc));
                let nw = vld1q_u8(src.as_ptr().add(row_off - cols + cc - 1));
                let ne = vld1q_u8(src.as_ptr().add(row_off - cols + cc + 1));
                let sw = vld1q_u8(src.as_ptr().add(row_off + cols + cc - 1));
                let se = vld1q_u8(src.as_ptr().add(row_off + cols + cc + 1));
                let g4 = avg4_u8(north, south, west, east);
                let diag4 = avg4_u8(nw, ne, sw, se);
                let h2 = vrhaddq_u8(west, east);
                let v2 = vrhaddq_u8(north, south);
                let cell_even = table[r & 1][cc & 1];
                let cell_odd = table[r & 1][(cc + 1) & 1];
                let (r_v, g_v, b_v) = assemble_rgb(center, g4, diag4, h2, v2, cell_even, cell_odd);
                vst3q_u8(
                    dst.as_mut_ptr().add((row_off + cc) * 3),
                    uint8x16x3_t(r_v, g_v, b_v),
                );
            };
            let mut c = 1usize;
            // 2× unrolled: two independent 16-px blocks per iter so the OoO core
            // overlaps their load/compute chains.
            while c + 32 <= cols - 1 {
                blk(c);
                blk(c + 16);
                c += 32;
            }
            while c + 16 <= cols - 1 {
                blk(c);
                c += 16;
            }
            // Remainder of this interior row (NEON tail) + right border (col cols-1),
            // all via scalar with replicate-edge addressing.
            for cc in c..cols {
                demosaic_px(src, dst, r, cc, rows, cols, &table);
            }
        }
    }
}

/// Widening rounded average of four u8 vectors: `(a+b+c+d+2)>>2` per lane.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn avg4_u8(
    a: std::arch::aarch64::uint8x16_t,
    b: std::arch::aarch64::uint8x16_t,
    c: std::arch::aarch64::uint8x16_t,
    d: std::arch::aarch64::uint8x16_t,
) -> std::arch::aarch64::uint8x16_t {
    use std::arch::aarch64::*;
    // low/high 8-lane halves widened to u16, summed, then vrshrn (rounding
    // narrowing shift right) by 2 == (sum + 2) >> 2.
    let lo = vaddq_u16(
        vaddl_u8(vget_low_u8(a), vget_low_u8(b)),
        vaddl_u8(vget_low_u8(c), vget_low_u8(d)),
    );
    let hi = vaddq_u16(
        vaddl_u8(vget_high_u8(a), vget_high_u8(b)),
        vaddl_u8(vget_high_u8(c), vget_high_u8(d)),
    );
    vcombine_u8(vrshrn_n_u16::<2>(lo), vrshrn_n_u16::<2>(hi))
}

/// Assemble per-lane R/G/B vectors given the precomputed interpolations and the
/// even/odd column cell kinds for this row block.
///
/// Lanes alternate `cell_even, cell_odd, cell_even, ...`. We compute the result
/// for both phases over the whole vector then blend with an even/odd lane mask.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn assemble_rgb(
    center: std::arch::aarch64::uint8x16_t,
    g4: std::arch::aarch64::uint8x16_t,
    diag4: std::arch::aarch64::uint8x16_t,
    h2: std::arch::aarch64::uint8x16_t,
    v2: std::arch::aarch64::uint8x16_t,
    cell_even: Cell,
    cell_odd: Cell,
) -> (
    std::arch::aarch64::uint8x16_t,
    std::arch::aarch64::uint8x16_t,
    std::arch::aarch64::uint8x16_t,
) {
    use std::arch::aarch64::*;

    // Per-cell (R,G,B) selection as vectors:
    //   R-cell:      R=center, G=g4,     B=diag4
    //   B-cell:      R=diag4,  G=g4,     B=center
    //   G-on-R-row:  R=h2,     G=center, B=v2
    //   G-on-B-row:  R=v2,     G=center, B=h2
    let pick = |cell: Cell| -> (uint8x16_t, uint8x16_t, uint8x16_t) {
        match cell {
            Cell::R => (center, g4, diag4),
            Cell::B => (diag4, g4, center),
            Cell::GonRRow => (h2, center, v2),
            Cell::GonBRow => (v2, center, h2),
        }
    };
    let (re, ge, be) = pick(cell_even);
    let (ro, go, bo) = pick(cell_odd);

    // Even-lane mask: 0xFF on even byte lanes, 0x00 on odd.
    let lane_idx = vld1q_u8([0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].as_ptr());
    let one = vdupq_n_u8(1);
    let is_odd = vtstq_u8(lane_idx, one); // 0xFF where lane index is odd
    let blend = |ev: uint8x16_t, od: uint8x16_t| vbslq_u8(is_odd, od, ev);

    (blend(re, ro), blend(ge, go), blend(be, bo))
}
