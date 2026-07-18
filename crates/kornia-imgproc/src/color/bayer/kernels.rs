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
pub(crate) enum Cell {
    R,
    GonRRow,
    GonBRow,
    B,
}

/// Resolve the per-cell color layout into a 2×2 phase table indexed by
/// `[row & 1][col & 1]`.
#[inline]
pub(crate) fn phase_table(pattern: BayerPattern) -> [[Cell; 2]; 2] {
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
        bayer_border_replicate(dst, rows, cols);
        return;
    }
    #[allow(unreachable_code)]
    {
        rgb_from_bayer_scalar(src, dst, rows, cols, pattern);
        bayer_border_replicate(dst, rows, cols);
    }
}

/// cv2's demosaic border semantics: after interpolation, the 1-pixel frame is
/// replaced by its interior neighbour (rows first, then columns, so corners
/// end up equal to the (1,1)-interior pixel) — byte-parity with
/// `cv2.cvtColor(COLOR_Bayer*2RGB)`. Images narrower than 3 pixels keep the
/// replicate-interpolated values (no interior row/column to copy from).
pub(crate) fn bayer_border_replicate(dst: &mut [u8], rows: usize, cols: usize) {
    let w = cols * 3;
    if rows >= 3 {
        let (first, rest) = dst.split_at_mut(w);
        first.copy_from_slice(&rest[..w]);
        let (head, last) = dst.split_at_mut((rows - 1) * w);
        last.copy_from_slice(&head[(rows - 2) * w..]);
    }
    if cols >= 3 {
        for r in 0..rows {
            let row = &mut dst[r * w..(r + 1) * w];
            let (c0, c1): (usize, usize) = (0, 3);
            row.copy_within(c1..c1 + 3, c0);
            let (cl, cp) = ((cols - 1) * 3, (cols - 2) * 3);
            row.copy_within(cp..cp + 3, cl);
        }
    }
}

/// NEON Bayer demosaic.
///
/// Borders (top/bottom rows, left/right columns) and the per-row tail use the
/// scalar oracle so replicate-edge handling is shared and exact. The interior is
/// processed in even-aligned 32-column blocks: each of the 3 source rows is loaded
/// with `vld2q_u8` (deinterleaving even/odd columns), so each pixel's phase is
/// uniform per register — no per-lane masks. 2-neighbour cases use `vrhaddq_u8`,
/// 4-neighbour cases use widening `vaddl_u8` + `vrshrn_n_u16(_, 2)`; results
/// re-interleave with `vzipq_u8` + `vst3q_u8`. Bit-exact vs the scalar oracle.
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
            let pr = row_off - cols; // prev row
            let nr = row_off + cols; // next row
                                     // Phase is uniform per even/odd column for this row (no per-lane masks).
            let cell_e = table[r & 1][0]; // even columns (col & 1 == 0)
            let cell_o = table[r & 1][1]; // odd columns

            // Scalar prefix: col 0 (border) + col 1 (odd, before the even-aligned bulk).
            demosaic_px(src, dst, r, 0, rows, cols, &table);
            demosaic_px(src, dst, r, 1, rows, cols, &table);

            let sp = src.as_ptr();
            let dp = dst.as_mut_ptr();
            let mut base = 2usize;
            // Carry-based: one vld2q per row per 32-col block. The west-shifted odds come
            // from the carried previous block (vext<15>), the east-shifted evens from the
            // looked-ahead next block (vext<1>) — 3 loads/block in steady state instead of
            // 9. Needs the lookahead [base+32,base+63] in-row, so a small guard + scalar tail.
            if cols >= 66 {
                // Process one 32-col block at `b` from its 3 row registers + the 6 shifted
                // neighbour registers. Captures raw pointers by copy (no dst borrow).
                let process = |cp: uint8x16x2_t,
                               cc: uint8x16x2_t,
                               cn: uint8x16x2_t,
                               po_w: uint8x16_t,
                               co_w: uint8x16_t,
                               no_w: uint8x16_t,
                               pe_e: uint8x16_t,
                               ce_e: uint8x16_t,
                               ne_e: uint8x16_t,
                               b: usize| {
                    // Even-column pixels (center = cc.0).
                    let g4e = avg4_u8(cp.0, cn.0, co_w, cc.1);
                    let dge = avg4_u8(po_w, cp.1, no_w, cn.1);
                    let h2e = vrhaddq_u8(co_w, cc.1);
                    let v2e = vrhaddq_u8(cp.0, cn.0);
                    let (re, ge, be) = assemble_cell(cell_e, cc.0, g4e, dge, h2e, v2e);
                    // Odd-column pixels (center = cc.1).
                    let g4o = avg4_u8(cp.1, cn.1, cc.0, ce_e);
                    let dgo = avg4_u8(cp.0, pe_e, cn.0, ne_e);
                    let h2o = vrhaddq_u8(cc.0, ce_e);
                    let v2o = vrhaddq_u8(cp.1, cn.1);
                    let (ro, go, bo) = assemble_cell(cell_o, cc.1, g4o, dgo, h2o, v2o);
                    let zr = vzipq_u8(re, ro);
                    let zg = vzipq_u8(ge, go);
                    let zb = vzipq_u8(be, bo);
                    vst3q_u8(dp.add((row_off + b) * 3), uint8x16x3_t(zr.0, zg.0, zb.0));
                    vst3q_u8(
                        dp.add((row_off + b + 16) * 3),
                        uint8x16x3_t(zr.1, zg.1, zb.1),
                    );
                };

                // Block 0 (base=2): no previous block, so its west/east shifts load directly
                // from base-2 / base+2 (carry-free seed). This also seeds the carry registers.
                let p_m = vld2q_u8(sp.add(pr + base - 2));
                let c_m = vld2q_u8(sp.add(row_off + base - 2));
                let n_m = vld2q_u8(sp.add(nr + base - 2));
                let mut cp = vld2q_u8(sp.add(pr + base));
                let mut cc = vld2q_u8(sp.add(row_off + base));
                let mut cn = vld2q_u8(sp.add(nr + base));
                let p_p = vld2q_u8(sp.add(pr + base + 2));
                let c_p = vld2q_u8(sp.add(row_off + base + 2));
                let n_p = vld2q_u8(sp.add(nr + base + 2));
                process(cp, cc, cn, p_m.1, c_m.1, n_m.1, p_p.0, c_p.0, n_p.0, base);
                let mut prev_po = cp.1;
                let mut prev_co = cc.1;
                let mut prev_no = cn.1;
                base += 32;

                // Blocks 1..: 3 vld2q for `cur`, lookahead next block's evens for east shift.
                while base + 31 < cols {
                    cp = vld2q_u8(sp.add(pr + base));
                    cc = vld2q_u8(sp.add(row_off + base));
                    cn = vld2q_u8(sp.add(nr + base));
                    if base + 63 >= cols {
                        break; // no room to look ahead — scalar tail handles this block
                    }
                    let np = vld2q_u8(sp.add(pr + base + 32));
                    let nc = vld2q_u8(sp.add(row_off + base + 32));
                    let nn = vld2q_u8(sp.add(nr + base + 32));
                    let po_w = vextq_u8::<15>(prev_po, cp.1);
                    let co_w = vextq_u8::<15>(prev_co, cc.1);
                    let no_w = vextq_u8::<15>(prev_no, cn.1);
                    let pe_e = vextq_u8::<1>(cp.0, np.0);
                    let ce_e = vextq_u8::<1>(cc.0, nc.0);
                    let ne_e = vextq_u8::<1>(cn.0, nn.0);
                    process(cp, cc, cn, po_w, co_w, no_w, pe_e, ce_e, ne_e, base);
                    prev_po = cp.1;
                    prev_co = cc.1;
                    prev_no = cn.1;
                    base += 32;
                }
            }
            // Scalar remainder (tail + right border) with replicate-edge addressing.
            for cc in base..cols {
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
    unsafe {
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
}

/// Assemble (R,G,B) for a register of same-phase pixels (uniform cell).
///   R-cell:     R=center, G=g4,     B=diag    | B-cell:    R=diag, G=g4, B=center
///   G-on-R-row: R=h2,     G=center, B=v2      | G-on-B-row: R=v2,  G=center, B=h2
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn assemble_cell(
    cell: Cell,
    center: std::arch::aarch64::uint8x16_t,
    g4: std::arch::aarch64::uint8x16_t,
    diag: std::arch::aarch64::uint8x16_t,
    h2: std::arch::aarch64::uint8x16_t,
    v2: std::arch::aarch64::uint8x16_t,
) -> (
    std::arch::aarch64::uint8x16_t,
    std::arch::aarch64::uint8x16_t,
    std::arch::aarch64::uint8x16_t,
) {
    match cell {
        Cell::R => (center, g4, diag),
        Cell::B => (diag, g4, center),
        Cell::GonRRow => (h2, center, v2),
        Cell::GonBRow => (v2, center, h2),
    }
}
