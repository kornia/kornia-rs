//! FAST corner kernels; bit-exact with `fast_detect_rows_u8`
//! in `crates/kornia-imgproc/src/features/fast.rs`.

use cuda_device::{kernel, thread};

#[inline(always)]
fn min_u8(a: u8, b: u8) -> u8 {
    if a > b {
        b
    } else {
        a
    }
}

#[inline(always)]
fn max_u8(a: u8, b: u8) -> u8 {
    if a > b {
        a
    } else {
        b
    }
}

#[inline(always)]
fn has_n_consecutive_ones(mask: u32, n: u32) -> bool {
    let m2 = mask & (mask >> 1);
    let m4 = m2 & (m2 >> 2);
    let m8 = m4 & (m4 >> 4);
    if n == 9 {
        (m8 & (mask >> 8)) != 0
    } else if n == 12 {
        (m8 & (m4 >> 8)) != 0
    } else {
        let mut acc = mask;
        let mut i = 1;
        while i < n {
            acc &= mask >> i;
            i += 1;
        }
        acc != 0
    }
}

/// FAST-N score per pixel. dst[y * width + x] = raw score (u16), 0 if no corner.
///
/// # Safety
/// `src` and `dst` hold `width * height` elements; `margin >= 3`.
#[kernel]
pub unsafe fn fast_score_u8(
    src: *const u8,
    dst: *mut u16,
    width: u32,
    height: u32,
    threshold_u8: u32,
    n: u32,
    margin: u32,
) {
    let x = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    let y = thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y();
    if x >= width || y >= height {
        return;
    }

    unsafe {
        let idx = (y * width + x) as usize;
        if y < margin
            || y >= height.saturating_sub(margin)
            || x < margin
            || x >= width.saturating_sub(margin)
        {
            *dst.add(idx) = 0;
            return;
        }

        let c = *src.add(idx);
        let t = threshold_u8 as u8;
        let lower = c.saturating_sub(t);
        let upper = c.saturating_add(t);

        let stride = width as i32;
        let idx_i = idx as i32;

        // Cardinals (ring 0, 4, 8, 12) tested first for early rejection.
        let p0 = *src.add((idx_i + 3) as usize);
        let p4 = *src.add((idx_i + 3 * stride) as usize);
        let p8 = *src.add((idx_i - 3) as usize);
        let p12 = *src.add((idx_i - 3 * stride) as usize);

        let b0 = (p0 > upper) as u32;
        let b4 = (p4 > upper) as u32;
        let b8 = (p8 > upper) as u32;
        let b12 = (p12 > upper) as u32;

        let d0 = (p0 < lower) as u32;
        let d4 = (p4 < lower) as u32;
        let d8 = (p8 < lower) as u32;
        let d12 = (p12 < lower) as u32;

        let b_count = b0 + b4 + b8 + b12;
        let d_count = d0 + d4 + d8 + d12;
        let min_card = n / 4;
        if b_count < min_card && d_count < min_card {
            *dst.add(idx) = 0;
            return;
        }

        // Remaining 12 ring pixels: constant offsets avoid runtime array indexing
        // and eliminate all .local stack spills in the PTX.
        let p1 = *src.add((idx_i + stride + 3) as usize);
        let p2 = *src.add((idx_i + 2 * stride + 2) as usize);
        let p3 = *src.add((idx_i + 3 * stride + 1) as usize);
        let p5 = *src.add((idx_i + 3 * stride - 1) as usize);
        let p6 = *src.add((idx_i + 2 * stride - 2) as usize);
        let p7 = *src.add((idx_i + stride - 3) as usize);
        let p9 = *src.add((idx_i - stride - 3) as usize);
        let p10 = *src.add((idx_i - 2 * stride - 2) as usize);
        let p11 = *src.add((idx_i - 3 * stride - 1) as usize);
        let p13 = *src.add((idx_i - 3 * stride + 1) as usize);
        let p14 = *src.add((idx_i - 2 * stride + 2) as usize);
        let p15 = *src.add((idx_i - stride + 3) as usize);

        // Branchless bitmask generation avoids warp divergence across 32 threads.
        let mut bright_mask: u32 = b0 | (b4 << 4) | (b8 << 8) | (b12 << 12);
        let mut dark_mask: u32 = d0 | (d4 << 4) | (d8 << 8) | (d12 << 12);

        macro_rules! check_p {
            ($p:expr, $bit:expr) => {
                bright_mask |= (($p > upper) as u32) << $bit;
                dark_mask |= (($p < lower) as u32) << $bit;
            };
        }

        check_p!(p1, 1);
        check_p!(p2, 2);
        check_p!(p3, 3);
        check_p!(p5, 5);
        check_p!(p6, 6);
        check_p!(p7, 7);
        check_p!(p9, 9);
        check_p!(p10, 10);
        check_p!(p11, 11);
        check_p!(p13, 13);
        check_p!(p14, 14);
        check_p!(p15, 15);

        let bright_dup = bright_mask | (bright_mask << 16);
        let dark_dup = dark_mask | (dark_mask << 16);

        let has_bright = has_n_consecutive_ones(bright_dup, n);
        let has_dark = has_n_consecutive_ones(dark_dup, n);

        if !has_bright && !has_dark {
            *dst.add(idx) = 0;
            return;
        }

        let score = if n == 9 {
            // For n=9, a 16-ring cannot simultaneously hold both a 9-arc of bright
            // pixels and a 9-arc of dark pixels (9 + 9 = 18 > 16). Evaluating only
            // the active side cuts 88 min/max operations and halves register pressure.
            let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) =
                if has_bright {
                    (
                        p0.saturating_sub(c),
                        p1.saturating_sub(c),
                        p2.saturating_sub(c),
                        p3.saturating_sub(c),
                        p4.saturating_sub(c),
                        p5.saturating_sub(c),
                        p6.saturating_sub(c),
                        p7.saturating_sub(c),
                        p8.saturating_sub(c),
                        p9.saturating_sub(c),
                        p10.saturating_sub(c),
                        p11.saturating_sub(c),
                        p12.saturating_sub(c),
                        p13.saturating_sub(c),
                        p14.saturating_sub(c),
                        p15.saturating_sub(c),
                    )
                } else {
                    (
                        c.saturating_sub(p0),
                        c.saturating_sub(p1),
                        c.saturating_sub(p2),
                        c.saturating_sub(p3),
                        c.saturating_sub(p4),
                        c.saturating_sub(p5),
                        c.saturating_sub(p6),
                        c.saturating_sub(p7),
                        c.saturating_sub(p8),
                        c.saturating_sub(p9),
                        c.saturating_sub(p10),
                        c.saturating_sub(p11),
                        c.saturating_sub(p12),
                        c.saturating_sub(p13),
                        c.saturating_sub(p14),
                        c.saturating_sub(p15),
                    )
                };

            let mut side_score: u8 = 0;

            macro_rules! step_pair {
                (
                    $v_start:expr, $v_end:expr,
                    $v1:expr, $v2:expr, $v3:expr, $v4:expr, $v5:expr, $v6:expr, $v7:expr, $v8:expr
                ) => {{
                    let core = min_u8(
                        $v1,
                        min_u8(
                            $v2,
                            min_u8($v3, min_u8($v4, min_u8($v5, min_u8($v6, min_u8($v7, $v8))))),
                        ),
                    );
                    let arc0 = min_u8(core, $v_start);
                    let arc1 = min_u8(core, $v_end);
                    side_score = max_u8(side_score, max_u8(arc0, arc1));
                }};
            }

            // Canonical FAST-9 step-by-2 sharing 8-element cores between start k and k+1.
            step_pair!(v0, v9, v1, v2, v3, v4, v5, v6, v7, v8);
            step_pair!(v2, v11, v3, v4, v5, v6, v7, v8, v9, v10);
            step_pair!(v4, v13, v5, v6, v7, v8, v9, v10, v11, v12);
            step_pair!(v6, v15, v7, v8, v9, v10, v11, v12, v13, v14);
            step_pair!(v8, v1, v9, v10, v11, v12, v13, v14, v15, v0);
            step_pair!(v10, v3, v11, v12, v13, v14, v15, v0, v1, v2);
            step_pair!(v12, v5, v13, v14, v15, v0, v1, v2, v3, v4);
            step_pair!(v14, v7, v15, v0, v1, v2, v3, v4, v5, v6);

            side_score as u32
        } else {
            macro_rules! abs_diff {
                ($p:expr) => {
                    if $p > c {
                        ($p - c) as u32
                    } else {
                        (c - $p) as u32
                    }
                };
            }
            abs_diff!(p0)
                + abs_diff!(p1)
                + abs_diff!(p2)
                + abs_diff!(p3)
                + abs_diff!(p4)
                + abs_diff!(p5)
                + abs_diff!(p6)
                + abs_diff!(p7)
                + abs_diff!(p8)
                + abs_diff!(p9)
                + abs_diff!(p10)
                + abs_diff!(p11)
                + abs_diff!(p12)
                + abs_diff!(p13)
                + abs_diff!(p14)
                + abs_diff!(p15)
        };

        *dst.add(idx) = score as u16;
    }
}
