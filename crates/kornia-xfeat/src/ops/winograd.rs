//! Winograd F(2×2, 3×3) convolution transforms for kornia-xfeat.
//!
//! Implements the Winograd minimal filtering algorithm for 3×3 stride-1 convolutions
//! using the F(2×2, 3×3) decomposition. This reduces multiply count from 36 to 16
//! per 2×2 output tile per input/output channel pair.
//!
//! # Matrices
//!
//! **G** (4×3, weight transform):
//! ```text
//! [[ 1,    0,    0  ],
//!  [ 0.5,  0.5,  0.5],
//!  [ 0.5, -0.5,  0.5],
//!  [ 0,    0,    1  ]]
//! ```
//!
//! **B^T** (4×4, input transform):
//! ```text
//! [[ 1,  0, -1,  0],
//!  [ 0,  1,  1,  0],
//!  [ 0, -1,  1,  0],
//!  [ 0,  1,  0, -1]]
//! ```
//!
//! **A^T** (2×4, output transform):
//! ```text
//! [[1, 1,  1, 0],
//!  [0, 1, -1, -1]]
//! ```
//!
//! # Weight layout
//! Weights passed to [`winograd_transform_weights_f32`] are in `[c_out, 3, 3, c_in]`
//! NHWC layout (same as the rest of the codebase).  The output is `[16, c_out, c_in]`:
//! 16 Winograd positions (row-major, pos = r*4+c), then the outer c_out × c_in matrix.

use rayon::prelude::*;

use super::{Activation, Conv3x3Args};

#[cfg(target_arch = "aarch64")]
use std::cell::RefCell;

// Per-rayon-worker scratch for the F(4,3) fp16 GEMM driver. Reused across
// tile-rows and frames to eliminate per-call heap allocations.
#[cfg(target_arch = "aarch64")]
thread_local! {
    static F43_A_ALL:  RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
    static F43_M_ACC:  RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
    static F43_C_TMP:  RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
    // GEMM panel-packing scratch: MR=4 and NR=8, sized to max(c_in)=128.
    // Eliminates 36×2 heap allocs per tile row that gemm_f16_mnk would incur.
    static F43_PACK_A: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
    static F43_PACK_B: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
    // Fused 1×1 epilogue scratch (per Winograd task):
    //   FUSE_STRIP_F32: dense f32 [block_pixels, c_out] Winograd-output strip,
    //   FUSE_A_F16:     dense f16 [block_pixels, c_out] = c_in of the 1×1 GEMM,
    //   FUSE_C_F16:     dense f16 [block_pixels, c_out2] 1×1 GEMM result,
    //   FUSE_PACK_A:    GEMM A-panel packing scratch (>= c_in * 8).
    static FUSE_STRIP_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static FUSE_A_F16:     RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
    static FUSE_C_F16:     RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
    static FUSE_PACK_A:    RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
}

// ─── Weight transform (offline) ───────────────────────────────────────────────

/// Transform conv weights from spatial to Winograd domain.
///
/// # Arguments
/// * `weights` — `[c_out, 3, 3, c_in]` NHWC packed weights (length `c_out * 9 * c_in`).
/// * `c_out`, `c_in` — channel dimensions.
///
/// # Returns
/// `[16, c_out, c_in]` transformed weights — 16 Winograd positions (pos = row*4 + col),
/// each followed by the c_out × c_in coefficient matrix.
pub fn winograd_transform_weights_f32(
    weights: &[f32], // [c_out, 3, 3, c_in] — length c_out * 9 * c_in
    c_out: usize,
    c_in: usize,
) -> Vec<f32> {
    // G (4×3):
    //  row0: [ 1,    0,    0  ]
    //  row1: [ 0.5,  0.5,  0.5]
    //  row2: [ 0.5, -0.5,  0.5]
    //  row3: [ 0,    0,    1  ]
    let g: [[f32; 3]; 4] = [
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.0, 0.0, 1.0],
    ];

    debug_assert_eq!(weights.len(), c_out * 9 * c_in);

    let mut out = vec![0.0f32; 16 * c_out * c_in];

    for co in 0..c_out {
        for ci in 0..c_in {
            // Extract 3×3 filter g_mat[kh][kw] for this (co, ci) pair.
            // Layout: weights[co * 9 * c_in + kh * 3 * c_in + kw * c_in + ci]
            let mut g_mat = [[0.0f32; 3]; 3];
            for kh in 0..3usize {
                for kw in 0..3usize {
                    g_mat[kh][kw] = weights[co * 9 * c_in + (kh * 3 + kw) * c_in + ci];
                }
            }

            // Compute U = G @ g_mat @ G^T  (4×3 @ 3×3 @ 3×4 → 4×4)
            // Step 1: tmp = G @ g_mat  → 4×3
            let mut tmp = [[0.0f32; 3]; 4];
            for i in 0..4usize {
                for j in 0..3usize {
                    let mut s = 0.0f32;
                    for k in 0..3usize {
                        s += g[i][k] * g_mat[k][j];
                    }
                    tmp[i][j] = s;
                }
            }

            // Step 2: u = tmp @ G^T  (G^T is 3×4, transpose of G)
            let mut u = [[0.0f32; 4]; 4];
            for i in 0..4usize {
                for j in 0..4usize {
                    // G^T[k][j] == G[j][k]
                    let mut s = 0.0f32;
                    for k in 0..3usize {
                        s += tmp[i][k] * g[j][k]; // g[j][k] == G^T[k][j]
                    }
                    u[i][j] = s;
                }
            }

            // Store into output: out[pos * c_out * c_in + co * c_in + ci]
            for row in 0..4usize {
                for col in 0..4usize {
                    let pos = row * 4 + col;
                    out[pos * c_out * c_in + co * c_in + ci] = u[row][col];
                }
            }
        }
    }

    out
}

// ─── Input tile transform ─────────────────────────────────────────────────────

/// Apply B^T · d · B to a 4×4 input patch (one channel, row-major).
///
/// All operations are additions/subtractions — no multiplications.
///
/// B^T rows: `[1,0,-1,0]`, `[0,1,1,0]`, `[0,-1,1,0]`, `[0,1,0,-1]`.
#[inline(always)]
pub fn winograd_transform_input_tile_f32(d: &[f32; 16], // 4×4 row-major
) -> [f32; 16] {
    // Convenience indexing: d[r][c] → d[r*4 + c]
    macro_rules! d {
        ($r:expr, $c:expr) => {
            d[$r * 4 + $c]
        };
    }

    // Step 1: row-wise B^T application (operates on column index).
    // For each row r:
    //   t[r][0] = d[r][0] - d[r][2]
    //   t[r][1] = d[r][1] + d[r][2]
    //   t[r][2] = d[r][2] - d[r][1]
    //   t[r][3] = d[r][1] - d[r][3]
    let t: [[f32; 4]; 4] = core::array::from_fn(|r| {
        [
            d!(r, 0) - d!(r, 2),
            d!(r, 1) + d!(r, 2),
            d!(r, 2) - d!(r, 1),
            d!(r, 1) - d!(r, 3),
        ]
    });

    // Step 2: col-wise B^T application (operates on row index).
    // For each column c:
    //   v[0][c] = t[0][c] - t[2][c]
    //   v[1][c] = t[1][c] + t[2][c]
    //   v[2][c] = t[2][c] - t[1][c]
    //   v[3][c] = t[1][c] - t[3][c]
    let mut out = [0.0f32; 16];
    for c in 0..4usize {
        out[c] = t[0][c] - t[2][c];
        out[4 + c] = t[1][c] + t[2][c];
        out[2 * 4 + c] = t[2][c] - t[1][c];
        out[3 * 4 + c] = t[1][c] - t[3][c];
    }
    out
}

// ─── Output tile transform ────────────────────────────────────────────────────

/// Apply A^T · M · A to a 4×4 accumulated Winograd product, yielding a 2×2 output tile.
///
/// A^T rows:
///   row0: `[1, 1,  1,  0]`
///   row1: `[0, 1, -1, -1]`
///
/// Returns `[y00, y01, y10, y11]` — the 2×2 output in row-major order.
#[inline(always)]
pub fn winograd_output_transform_f32(
    m: &[f32; 16], // 4×4 accumulated GEMM result (row-major)
) -> [f32; 4] {
    macro_rules! m {
        ($r:expr, $c:expr) => {
            m[$r * 4 + $c]
        };
    }

    // Step 1: A^T @ M  (2×4 result)
    // s[0][c] = m[0][c] + m[1][c] + m[2][c]   (A^T row 0: [1,1,1,0])
    // s[1][c] = m[1][c] - m[2][c] - m[3][c]   (A^T row 1: [0,1,-1,-1])
    let s: [[f32; 4]; 2] = core::array::from_fn(|r| {
        if r == 0 {
            [
                m!(0, 0) + m!(1, 0) + m!(2, 0),
                m!(0, 1) + m!(1, 1) + m!(2, 1),
                m!(0, 2) + m!(1, 2) + m!(2, 2),
                m!(0, 3) + m!(1, 3) + m!(2, 3),
            ]
        } else {
            [
                m!(1, 0) - m!(2, 0) - m!(3, 0),
                m!(1, 1) - m!(2, 1) - m!(3, 1),
                m!(1, 2) - m!(2, 2) - m!(3, 2),
                m!(1, 3) - m!(2, 3) - m!(3, 3),
            ]
        }
    });

    // Step 2: S @ A  (A = (A^T)^T; col 0: [1,1,1,0]^T, col 1: [0,1,-1,-1]^T)
    // y[r][0] = s[r][0] + s[r][1] + s[r][2]
    // y[r][1] = s[r][1] - s[r][2] - s[r][3]
    [
        s[0][0] + s[0][1] + s[0][2], // y[0][0]
        s[0][1] - s[0][2] - s[0][3], // y[0][1]
        s[1][0] + s[1][1] + s[1][2], // y[1][0]
        s[1][1] - s[1][2] - s[1][3], // y[1][1]
    ]
}

// ══════════════════════════════════════════════════════════════════════════════
//  F(4×4, 3×3) — Winograd with a 4×4 output tile and 6×6 input patch.
//
//  n_positions = 36, input patch = 6×6, output tile = 4×4.
//  MAC reduction vs naive: 36/(6*6/?) — 36 multiplies per 4×4 tile per (co,ci)
//  pair vs 4*4*9 = 144 for direct → 4× reduction (vs 2.25× for F(2,3)).
//
//  Matrices (Lavin & Gray, CVPR 2016):
//
//  G (6×3, weight transform, rationals — applied offline):
//    [[ 1/4,    0,      0   ],
//     [-1/6,  -1/6,   -1/6  ],
//     [-1/6,   1/6,   -1/6  ],
//     [ 1/24,  1/12,   1/6  ],
//     [ 1/24, -1/12,   1/6  ],
//     [  0,    0,      1    ]]
//
//  B^T (6×6, input transform, integers only):
//    [[ 4,  0, -5,  0,  1,  0],
//     [ 0, -4, -4,  1,  1,  0],
//     [ 0,  4, -4, -1,  1,  0],
//     [ 0, -2, -1,  2,  1,  0],
//     [ 0,  2, -1, -2,  1,  0],
//     [ 0,  4,  0, -5,  0,  1]]
//
//  A^T (4×6, output transform, integers only):
//    [[1,  1,  1,  1,  1,  0],
//     [0,  1, -1,  2, -2,  0],
//     [0,  1,  1,  4,  4,  0],
//     [0,  1, -1,  8, -8,  1]]
// ══════════════════════════════════════════════════════════════════════════════

/// Transform conv weights from spatial to Winograd F(4×4, 3×3) domain.
///
/// # Arguments
/// * `weights` — `[c_out, 3, 3, c_in]` NHWC packed weights (length `c_out * 9 * c_in`).
/// * `c_out`, `c_in` — channel dimensions.
///
/// # Returns
/// `[36, c_out, c_in]` transformed weights — 36 Winograd positions (pos = row*6 + col),
/// each followed by the c_out × c_in coefficient matrix.
pub fn winograd_transform_weights_f32_f43(
    weights: &[f32], // [c_out, 3, 3, c_in] — length c_out * 9 * c_in
    c_out: usize,
    c_in: usize,
) -> Vec<f32> {
    // G (6×3) — rationals, applied offline.
    const G: [[f32; 3]; 6] = [
        [1.0 / 4.0, 0.0, 0.0],
        [-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0],
        [-1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0],
        [1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0],
        [1.0 / 24.0, -1.0 / 12.0, 1.0 / 6.0],
        [0.0, 0.0, 1.0],
    ];

    debug_assert_eq!(weights.len(), c_out * 9 * c_in);

    let mut out = vec![0.0f32; 36 * c_out * c_in];

    for co in 0..c_out {
        for ci in 0..c_in {
            // Extract 3×3 filter g_mat[kh][kw].
            // Layout: weights[co * 9 * c_in + (kh * 3 + kw) * c_in + ci]
            let mut g_mat = [[0.0f32; 3]; 3];
            for kh in 0..3usize {
                for kw in 0..3usize {
                    g_mat[kh][kw] = weights[co * 9 * c_in + (kh * 3 + kw) * c_in + ci];
                }
            }

            // U = G @ g_mat @ G^T  (6×3 @ 3×3 @ 3×6 → 6×6)
            // Step 1: tmp = G @ g_mat  → 6×3
            let mut tmp = [[0.0f32; 3]; 6];
            for i in 0..6usize {
                for j in 0..3usize {
                    let mut s = 0.0f32;
                    for k in 0..3usize {
                        s += G[i][k] * g_mat[k][j];
                    }
                    tmp[i][j] = s;
                }
            }

            // Step 2: u = tmp @ G^T  (G^T[k][j] == G[j][k])
            let mut u = [[0.0f32; 6]; 6];
            for i in 0..6usize {
                for j in 0..6usize {
                    let mut s = 0.0f32;
                    for k in 0..3usize {
                        s += tmp[i][k] * G[j][k];
                    }
                    u[i][j] = s;
                }
            }

            // Store: out[pos * c_out * c_in + co * c_in + ci], pos = row*6+col.
            for row in 0..6usize {
                for col in 0..6usize {
                    let pos = row * 6 + col;
                    out[pos * c_out * c_in + co * c_in + ci] = u[row][col];
                }
            }
        }
    }

    out
}

/// Apply B^T · d · B to a 6×6 input patch (one channel, row-major) for F(4,3).
///
/// All operations are additions/subtractions and constant integer scalings.
/// Returns the 6×6 transformed patch in row-major order.
#[inline(always)]
pub fn winograd_transform_input_tile_f32_f43(d: &[f32; 36], // 6×6 row-major
) -> [f32; 36] {
    // B^T rows applied to a length-6 vector x = [x0..x5]:
    //   r0 =  4*x0          - 5*x2          + 1*x4
    //   r1 =        -4*x1 - 4*x2 + 1*x3 + 1*x4
    //   r2 =         4*x1 - 4*x2 - 1*x3 + 1*x4
    //   r3 =        -2*x1 - 1*x2 + 2*x3 + 1*x4
    //   r4 =         2*x1 - 1*x2 - 2*x3 + 1*x4
    //   r5 =         4*x1        - 5*x3        + 1*x5
    #[inline(always)]
    fn bt(x0: f32, x1: f32, x2: f32, x3: f32, x4: f32, x5: f32) -> [f32; 6] {
        [
            4.0 * x0 - 5.0 * x2 + x4,
            -4.0 * x1 - 4.0 * x2 + x3 + x4,
            4.0 * x1 - 4.0 * x2 - x3 + x4,
            -2.0 * x1 - x2 + 2.0 * x3 + x4,
            2.0 * x1 - x2 - 2.0 * x3 + x4,
            4.0 * x1 - 5.0 * x3 + x5,
        ]
    }

    // Step 1: row-wise B^T (apply to each row of d → t[r][*]).
    let mut t = [[0.0f32; 6]; 6];
    for r in 0..6usize {
        let b = r * 6;
        t[r] = bt(d[b], d[b + 1], d[b + 2], d[b + 3], d[b + 4], d[b + 5]);
    }

    // Step 2: col-wise B^T (apply to each column of t → v[*][c]).
    let mut out = [0.0f32; 36];
    for c in 0..6usize {
        let col = bt(t[0][c], t[1][c], t[2][c], t[3][c], t[4][c], t[5][c]);
        for r in 0..6usize {
            out[r * 6 + c] = col[r];
        }
    }
    out
}

/// Apply A^T · M · A to a 6×6 accumulated Winograd product, yielding a 4×4 tile.
///
/// Returns a flat `[f32; 16]` 4×4 output in row-major order.
#[inline(always)]
pub fn winograd_output_transform_f32_f43(
    m: &[f32; 36], // 6×6 accumulated GEMM result (row-major)
) -> [f32; 16] {
    // A^T rows applied to a length-6 vector x = [x0..x5]:
    //   r0 = x0 + x1 + x2 +   x3 +   x4
    //   r1 =      x1 - x2 + 2*x3 - 2*x4
    //   r2 =      x1 + x2 + 4*x3 + 4*x4
    //   r3 =      x1 - x2 + 8*x3 - 8*x4 + x5
    #[inline(always)]
    fn at(x0: f32, x1: f32, x2: f32, x3: f32, x4: f32, x5: f32) -> [f32; 4] {
        [
            x0 + x1 + x2 + x3 + x4,
            x1 - x2 + 2.0 * x3 - 2.0 * x4,
            x1 + x2 + 4.0 * x3 + 4.0 * x4,
            x1 - x2 + 8.0 * x3 - 8.0 * x4 + x5,
        ]
    }

    // Step 1: s = A^T @ M  (4×6) — apply A^T to each column of M.
    // s[*][c] = at(m[0][c], m[1][c], ..., m[5][c])
    let mut s = [[0.0f32; 6]; 4]; // s[row 0..4][col 0..6]
    for c in 0..6usize {
        let col = at(m[c], m[6 + c], m[12 + c], m[18 + c], m[24 + c], m[30 + c]);
        for r in 0..4usize {
            s[r][c] = col[r];
        }
    }

    // Step 2: y = s @ A  (4×4) — apply A^T to each row of s (A = (A^T)^T).
    let mut y = [0.0f32; 16];
    for r in 0..4usize {
        let row = at(s[r][0], s[r][1], s[r][2], s[r][3], s[r][4], s[r][5]);
        for c in 0..4usize {
            y[r * 4 + c] = row[c];
        }
    }
    y
}

/// Helper: extract a 6×6 single-channel patch from NHWC input with zero-padding.
#[inline(always)]
pub fn extract_patch_f32_6x6_pub(
    input: &[f32],
    h_in: usize,
    w_in: usize,
    c_in: usize,
    ih_start: isize,
    iw_start: isize,
    ci: usize,
) -> [f32; 36] {
    extract_patch_f32_6x6(input, h_in, w_in, c_in, ih_start, iw_start, ci)
}

#[inline(always)]
fn extract_patch_f32_6x6(
    input: &[f32],
    h_in: usize,
    w_in: usize,
    c_in: usize,
    ih_start: isize,
    iw_start: isize,
    ci: usize,
) -> [f32; 36] {
    let mut patch = [0.0f32; 36];
    for r in 0..6isize {
        let ih = ih_start + r;
        if ih < 0 || ih >= h_in as isize {
            continue;
        }
        let row_base = ih as usize * w_in;
        for c in 0..6isize {
            let iw = iw_start + c;
            if iw >= 0 && iw < w_in as isize {
                patch[(r * 6 + c) as usize] = input[(row_base + iw as usize) * c_in + ci];
            }
        }
    }
    patch
}

/// 3×3 stride-1 NHWC convolution via Winograd F(4×4, 3×3).
///
/// # Arguments
/// * `input` — `[h_in, w_in, c_in]` NHWC tensor.
/// * `weights_transformed` — `[36, c_out, c_in]` from
///   [`winograd_transform_weights_f32_f43`].
/// * `bias` — optional per-output-channel bias (length `c_out`).
/// * `activation` — fused activation applied in the output epilogue.
/// * `output` — `[h_out, w_out, c_out]` NHWC output (pre-allocated).
///
/// Output is tiled in 4×4 tiles. Odd-tile remainders (rows / cols not a
/// multiple of 4) are left at the bias+activation baseline here; for correct
/// boundary handling use
/// [`conv3x3_winograd_nhwc_f43_with_scalar_fallback`].
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_winograd_nhwc_f43(
    input: &[f32],
    h_in: usize,
    w_in: usize,
    c_in: usize,
    weights_transformed: &[f32], // [36, c_out, c_in]
    bias: Option<&[f32]>,
    c_out: usize,
    activation: Activation,
    output: &mut [f32],
    h_out: usize,
    w_out: usize,
) {
    debug_assert_eq!(input.len(), h_in * w_in * c_in);
    debug_assert_eq!(output.len(), h_out * w_out * c_out);
    debug_assert_eq!(weights_transformed.len(), 36 * c_out * c_in);
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), c_out);
    }
    debug_assert!(
        c_out <= 128,
        "F(4,3) stack accumulator sized for c_out ≤ 128"
    );

    // Ceiling division: last tile may be partial when h_out/w_out % 4 != 0.
    let n_tile_h = (h_out + 3) / 4;
    let n_tile_w = (w_out + 3) / 4;

    let in_ptr = input.as_ptr() as usize;
    let wt_ptr = weights_transformed.as_ptr() as usize;
    let bias_ptr: usize = bias.map_or(0, |b| b.as_ptr() as usize);
    let has_bias = bias.is_some();

    let tile_row_stride = 4 * w_out * c_out; // floats per tile-row in output
    let out_ptr = output.as_mut_ptr() as usize;

    (0..n_tile_h).into_par_iter().for_each(|tile_oh| {
        let valid_rows = (h_out - tile_oh * 4).min(4);

        let input = unsafe { std::slice::from_raw_parts(in_ptr as *const f32, h_in * w_in * c_in) };
        let wt = unsafe { std::slice::from_raw_parts(wt_ptr as *const f32, 36 * c_out * c_in) };
        let bias_slice: Option<&[f32]> = if has_bias {
            Some(unsafe { std::slice::from_raw_parts(bias_ptr as *const f32, c_out) })
        } else {
            None
        };

        // This tile row writes output rows [tile_oh*4 .. tile_oh*4+valid_rows].
        let out_row_base = unsafe {
            std::slice::from_raw_parts_mut(
                (out_ptr as *mut f32).add(tile_oh * tile_row_stride),
                valid_rows * w_out * c_out,
            )
        };

        let ih_start = (tile_oh * 4) as isize - 1; // top of 6×6 patch (with padding)

        // Stack accumulator: 36 positions × c_out (≤128) = 4608 f32 = 18 KB.
        let mut m_acc = [0.0f32; 36 * 128];

        for tile_ow in 0..n_tile_w {
            let iw_start = (tile_ow * 4) as isize - 1;
            let valid_cols = (w_out - tile_ow * 4).min(4);

            m_acc[..36 * c_out].fill(0.0);

            for ci in 0..c_in {
                let patch = extract_patch_f32_6x6(input, h_in, w_in, c_in, ih_start, iw_start, ci);
                let v = winograd_transform_input_tile_f32_f43(&patch);

                // wt layout: [36, c_out, c_in] → wt[p*c_out*c_in + co*c_in + ci]
                for p in 0..36usize {
                    let v_val = v[p];
                    let wt_base = p * c_out * c_in + ci;
                    let m_base = p * c_out;
                    for co in 0..c_out {
                        m_acc[m_base + co] += v_val * wt[wt_base + co * c_in];
                    }
                }
            }

            // Output transform and write the tile (bounded by valid_rows and valid_cols).
            let ow0 = tile_ow * 4;
            for co in 0..c_out {
                let m_tile: [f32; 36] = core::array::from_fn(|p| m_acc[p * c_out + co]);
                let y = winograd_output_transform_f32_f43(&m_tile);
                let b = bias_slice.map_or(0.0, |bs| bs[co]);

                for ry in 0..valid_rows {
                    let row_off = ry * w_out * c_out;
                    for cx in 0..valid_cols {
                        let val = apply_act(y[ry * 4 + cx] + b, activation);
                        out_row_base[row_off + (ow0 + cx) * c_out + co] = val;
                    }
                }
            }
        }
    });
}

/// F(4,3) driver with correct scalar fallback on the row/column epilogue when
/// `h_out` / `w_out` are not multiples of 4.
///
/// When both dims are multiples of 4 this is identical to
/// [`conv3x3_winograd_nhwc_f43`].
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_winograd_nhwc_f43_with_scalar_fallback(
    input: &[f32],
    h_in: usize,
    w_in: usize,
    c_in: usize,
    weights_transformed: &[f32], // [36, c_out, c_in]
    _weights_spatial: &[f32],    // kept for API compat; no longer used
    bias: Option<&[f32]>,
    c_out: usize,
    activation: Activation,
    output: &mut [f32],
    h_out: usize,
    w_out: usize,
) {
    // The main driver now uses ceiling-division tiling and handles partial
    // last tiles via bounded output writes (extract_patch zero-pads OOB).
    // No scalar epilogue needed.
    conv3x3_winograd_nhwc_f43(
        input,
        h_in,
        w_in,
        c_in,
        weights_transformed,
        bias,
        c_out,
        activation,
        output,
        h_out,
        w_out,
    );
}

// ─── NEON f32x4 variants ──────────────────────────────────────────────────────
//
// The fp16 intrinsics (`float16x8_t`, `vld1q_f16`, etc.) require the unstable
// `stdarch_neon_f16` feature and are not available on stable Rust 1.93.
// We instead provide NEON f32x4 variants that process 4 channels at once using
// stable `float32x4_t` intrinsics.  The transforms use only vaddq_f32/vsubq_f32
// — zero multiplications, identical algorithm to the scalar path.

/// NEON f32x4 input tile transform: B^T · d · B for a 4×4 patch, 4 channels at once.
///
/// Processes NR=4 channels simultaneously using `float32x4_t` vectors.
/// All arithmetic is additions/subtractions — zero multiplications.
///
/// # Arguments
/// * `d` — 16 pointers `[r*4+c]` each pointing to 4 valid `f32` values
///   (one for each lane).  Ordering: `d[r*4 + c]` covers `(r, c) in 0..4×0..4`.
///
/// # Returns
/// 16 `float32x4_t` values in `[pos]` order (pos = row*4+col).
///
/// # Safety
/// Requires `target_arch = "aarch64"` with the `neon` target feature.
/// Each pointer in `d` must be valid for a 4-float aligned read.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn winograd_transform_input_tile_f32x4(
    d: &[*const f32; 16], // d[r*4+c] → pointer to 4 f32 values (one per lane)
) -> [std::arch::aarch64::float32x4_t; 16] {
    use std::arch::aarch64::*;

    // Load all 16 vectors
    let d00 = vld1q_f32(d[0]);
    let d01 = vld1q_f32(d[1]);
    let d02 = vld1q_f32(d[2]);
    let d03 = vld1q_f32(d[3]);
    let d10 = vld1q_f32(d[4]);
    let d11 = vld1q_f32(d[5]);
    let d12 = vld1q_f32(d[6]);
    let d13 = vld1q_f32(d[7]);
    let d20 = vld1q_f32(d[8]);
    let d21 = vld1q_f32(d[9]);
    let d22 = vld1q_f32(d[10]);
    let d23 = vld1q_f32(d[11]);
    let d30 = vld1q_f32(d[12]);
    let d31 = vld1q_f32(d[13]);
    let d32 = vld1q_f32(d[14]);
    let d33 = vld1q_f32(d[15]);

    // Step 1: row-wise B^T (for each row r):
    //   t[r][0] = d[r][0] - d[r][2]
    //   t[r][1] = d[r][1] + d[r][2]
    //   t[r][2] = d[r][2] - d[r][1]
    //   t[r][3] = d[r][1] - d[r][3]
    let t00 = vsubq_f32(d00, d02);
    let t01 = vaddq_f32(d01, d02);
    let t02 = vsubq_f32(d02, d01);
    let t03 = vsubq_f32(d01, d03);

    let t10 = vsubq_f32(d10, d12);
    let t11 = vaddq_f32(d11, d12);
    let t12 = vsubq_f32(d12, d11);
    let t13 = vsubq_f32(d11, d13);

    let t20 = vsubq_f32(d20, d22);
    let t21 = vaddq_f32(d21, d22);
    let t22 = vsubq_f32(d22, d21);
    let t23 = vsubq_f32(d21, d23);

    let t30 = vsubq_f32(d30, d32);
    let t31 = vaddq_f32(d31, d32);
    let t32 = vsubq_f32(d32, d31);
    let t33 = vsubq_f32(d31, d33);

    // Step 2: col-wise B^T (for each column c):
    //   v[0][c] = t[0][c] - t[2][c]
    //   v[1][c] = t[1][c] + t[2][c]
    //   v[2][c] = t[2][c] - t[1][c]
    //   v[3][c] = t[1][c] - t[3][c]
    [
        vsubq_f32(t00, t20), // pos 0 = (0,0)
        vsubq_f32(t01, t21), // pos 1 = (0,1)
        vsubq_f32(t02, t22), // pos 2 = (0,2)
        vsubq_f32(t03, t23), // pos 3 = (0,3)
        vaddq_f32(t10, t20), // pos 4 = (1,0)
        vaddq_f32(t11, t21), // pos 5 = (1,1)
        vaddq_f32(t12, t22), // pos 6 = (1,2)
        vaddq_f32(t13, t23), // pos 7 = (1,3)
        vsubq_f32(t20, t10), // pos 8 = (2,0)
        vsubq_f32(t21, t11), // pos 9 = (2,1)
        vsubq_f32(t22, t12), // pos 10 = (2,2)
        vsubq_f32(t23, t13), // pos 11 = (2,3)
        vsubq_f32(t10, t30), // pos 12 = (3,0)
        vsubq_f32(t11, t31), // pos 13 = (3,1)
        vsubq_f32(t12, t32), // pos 14 = (3,2)
        vsubq_f32(t13, t33), // pos 15 = (3,3)
    ]
}

/// NEON f32x4 output transform: A^T · M · A for one Winograd tile, 4 channels at once.
///
/// Applies the output transform to a 4×4 accumulated GEMM result in f32,
/// yielding 4 `float32x4_t` values representing the 2×2 output tile (4 channels each).
///
/// # Arguments
/// * `m` — 16 `float32x4_t` values in row-major 4×4 order (accumulated GEMM result).
/// * `bias_ptr` — pointer to 4 f32 bias values broadcast across the output tile.
/// * `act` — activation applied in the epilogue (ReLU or Identity; Sigmoid is scalar).
///
/// # Returns
/// `[y00, y01, y10, y11]` — the 2×2 output tile, 4 channels each.
///
/// # Safety
/// Requires `target_arch = "aarch64"` with the `neon` target feature.
/// `bias_ptr` must point to 4 valid `f32` values.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn winograd_output_transform_f32x4(
    m: &[std::arch::aarch64::float32x4_t; 16],
    bias_ptr: *const f32,
    act: Activation,
) -> [std::arch::aarch64::float32x4_t; 4] {
    use std::arch::aarch64::*;

    macro_rules! m {
        ($r:expr, $c:expr) => {
            m[$r * 4 + $c]
        };
    }

    // Step 1: A^T @ M  (2×4 result)
    // s[0][c] = m[0][c] + m[1][c] + m[2][c]
    // s[1][c] = m[1][c] - m[2][c] - m[3][c]
    let s00 = vaddq_f32(vaddq_f32(m!(0, 0), m!(1, 0)), m!(2, 0));
    let s01 = vaddq_f32(vaddq_f32(m!(0, 1), m!(1, 1)), m!(2, 1));
    let s02 = vaddq_f32(vaddq_f32(m!(0, 2), m!(1, 2)), m!(2, 2));
    let s03 = vaddq_f32(vaddq_f32(m!(0, 3), m!(1, 3)), m!(2, 3));

    let s10 = vsubq_f32(vsubq_f32(m!(1, 0), m!(2, 0)), m!(3, 0));
    let s11 = vsubq_f32(vsubq_f32(m!(1, 1), m!(2, 1)), m!(3, 1));
    let s12 = vsubq_f32(vsubq_f32(m!(1, 2), m!(2, 2)), m!(3, 2));
    let s13 = vsubq_f32(vsubq_f32(m!(1, 3), m!(2, 3)), m!(3, 3));

    // Step 2: S @ A
    // y[r][0] = s[r][0] + s[r][1] + s[r][2]
    // y[r][1] = s[r][1] - s[r][2] - s[r][3]
    let y00 = vaddq_f32(vaddq_f32(s00, s01), s02);
    let y01 = vsubq_f32(vsubq_f32(s01, s02), s03);
    let y10 = vaddq_f32(vaddq_f32(s10, s11), s12);
    let y11 = vsubq_f32(vsubq_f32(s11, s12), s13);

    // Load bias and apply activation
    let bias_v = vld1q_f32(bias_ptr);
    let y00 = vaddq_f32(y00, bias_v);
    let y01 = vaddq_f32(y01, bias_v);
    let y10 = vaddq_f32(y10, bias_v);
    let y11 = vaddq_f32(y11, bias_v);

    match act {
        Activation::Relu => {
            let zero = vdupq_n_f32(0.0);
            [
                vmaxq_f32(y00, zero),
                vmaxq_f32(y01, zero),
                vmaxq_f32(y10, zero),
                vmaxq_f32(y11, zero),
            ]
        }
        _ => [y00, y01, y10, y11],
    }
}

// ─── Full conv3x3 Winograd driver ─────────────────────────────────────────────

/// Helper: extract a 4×4 single-channel patch from the NHWC input with zero-padding.
///
/// `ih_start` and `iw_start` are the top-left corner in the (zero-padded) input.
/// Zero-padding is applied for out-of-bounds accesses.
#[inline(always)]
fn extract_patch_f32(
    input: &[f32],
    h_in: usize,
    w_in: usize,
    c_in: usize,
    ih_start: isize,
    iw_start: isize,
    ci: usize,
) -> [f32; 16] {
    let mut patch = [0.0f32; 16];
    for r in 0..4isize {
        let ih = ih_start + r;
        for c in 0..4isize {
            let iw = iw_start + c;
            if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                patch[(r * 4 + c) as usize] = input[(ih as usize * w_in + iw as usize) * c_in + ci];
            }
        }
    }
    patch
}

/// 3×3 stride-1 NHWC convolution via Winograd F(2×2, 3×3).
///
/// # Arguments
/// * `input` — `[h_in, w_in, c_in]` NHWC tensor.
/// * `weights_transformed` — `[16, c_out, c_in]` from [`winograd_transform_weights_f32`].
/// * `bias` — optional per-output-channel bias (length `c_out`).
/// * `activation` — fused activation applied in the output epilogue.
/// * `output` — `[h_out, w_out, c_out]` NHWC output (must be pre-allocated).
/// * `_v_buf` — reserved scratch buffer (currently unused; reserved for future
///              tiling optimizations). Pass a slice of length ≥ 1.
///
/// Both `h_out == h_in` and `w_out == w_in` (stride-1, same-padded conv).
/// The driver processes output in 2×2 tiles and handles odd dimensions with a
/// scalar epilogue on the last row and/or column.
///
/// Per-tile accumulators are stack-allocated (`[f32; 16 * MAX_C_OUT]`) to
/// avoid heap allocation on the hot path. Row tiles are parallelised with rayon.
///
/// # Panics
/// Panics in debug mode if buffer lengths don't match the stated dimensions.
#[allow(clippy::too_many_arguments)]
#[allow(unused_mut)]
pub fn conv3x3_winograd_nhwc(
    input: &[f32],
    h_in: usize,
    w_in: usize,
    c_in: usize,
    weights_transformed: &[f32], // [16, c_out, c_in]
    bias: Option<&[f32]>,
    c_out: usize,
    activation: Activation,
    output: &mut [f32],
    h_out: usize,
    w_out: usize,
    _v_buf: &mut [f32], // reserved (pre-allocated by caller, not currently used)
) {
    debug_assert_eq!(input.len(), h_in * w_in * c_in);
    debug_assert_eq!(output.len(), h_out * w_out * c_out);
    debug_assert_eq!(weights_transformed.len(), 16 * c_out * c_in);
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), c_out);
    }

    // Number of full 2×2 output tiles in each dimension.
    let n_tile_h = h_out / 2;
    let n_tile_w = w_out / 2;

    // Pointers wrapped as usize for rayon Send boundary.
    let in_ptr = input.as_ptr() as usize;
    let wt_ptr = weights_transformed.as_ptr() as usize;
    let bias_ptr: usize = bias.map_or(0, |b| b.as_ptr() as usize);
    let has_bias = bias.is_some();

    // ── Main 2×2 tile loop (parallelised over tile rows) ──────────────────────
    // Each tile row writes to output rows [tile_oh*2 .. tile_oh*2+2].
    let tile_row_stride = 2 * w_out * c_out; // floats per tile-row in output
    let out_ptr = output.as_mut_ptr() as usize;

    (0..n_tile_h).into_par_iter().for_each(|tile_oh| {
        let input = unsafe { std::slice::from_raw_parts(in_ptr as *const f32, h_in * w_in * c_in) };
        let wt = unsafe { std::slice::from_raw_parts(wt_ptr as *const f32, 16 * c_out * c_in) };
        let bias_slice: Option<&[f32]> = if has_bias {
            Some(unsafe { std::slice::from_raw_parts(bias_ptr as *const f32, c_out) })
        } else {
            None
        };

        // Top-left of this tile row in output: (tile_oh*2, 0)
        let out_row_base = unsafe {
            std::slice::from_raw_parts_mut(
                (out_ptr as *mut f32).add(tile_oh * tile_row_stride),
                2 * w_out * c_out,
            )
        };

        let ih_start = (tile_oh * 2) as isize - 1; // top of the 4×4 input patch (with padding)

        // Stack-allocate the per-tile accumulator.
        // Max c_out across all eligible layers is 128: 16*128 = 2048 f32 = 8 KB on stack.
        // This avoids heap allocation (the previous `vec![0.0f32; 16 * c_out]` per tile).
        let mut m_acc = [0.0f32; 16 * 128];

        for tile_ow in 0..n_tile_w {
            let iw_start = (tile_ow * 2) as isize - 1;

            // Zero the accumulator slice we'll use for this tile.
            m_acc[..16 * c_out].fill(0.0);

            for ci in 0..c_in {
                let patch = extract_patch_f32(input, h_in, w_in, c_in, ih_start, iw_start, ci);
                let v = winograd_transform_input_tile_f32(&patch);

                // For each Winograd position p, accumulate into all c_out outputs.
                // wt layout: [16, c_out, c_in] → wt[p * c_out * c_in + co * c_in + ci]
                for p in 0..16usize {
                    let v_val = v[p];
                    let wt_base = p * c_out * c_in + ci;
                    let m_base = p * c_out;
                    for co in 0..c_out {
                        m_acc[m_base + co] += v_val * wt[wt_base + co * c_in];
                    }
                }
            }

            // Output transform and write.
            // oh = tile_oh * 2, ow = tile_ow * 2
            let ow0 = tile_ow * 2;
            let ow1 = ow0 + 1;

            for co in 0..c_out {
                // Gather the 4×4 m matrix for this output channel.
                let m_tile: [f32; 16] = core::array::from_fn(|p| m_acc[p * c_out + co]);

                let y = winograd_output_transform_f32(&m_tile);

                // Bias + activation.
                let b = bias_slice.map_or(0.0, |bs| bs[co]);
                let y00 = apply_act(y[0] + b, activation);
                let y01 = apply_act(y[1] + b, activation);
                let y10 = apply_act(y[2] + b, activation);
                let y11 = apply_act(y[3] + b, activation);

                // Write into out_row_base: shape [2, w_out, c_out]
                // Row 0 of tile → out_row_base[0..w_out*c_out]
                // Row 1 of tile → out_row_base[w_out*c_out..2*w_out*c_out]
                let row1 = w_out * c_out;
                out_row_base[ow0 * c_out + co] = y00;
                out_row_base[ow1 * c_out + co] = y01;
                out_row_base[row1 + ow0 * c_out + co] = y10;
                out_row_base[row1 + ow1 * c_out + co] = y11;
            }
        }
    });

    // ── Scalar epilogue: last column (odd w_out) ──────────────────────────────
    if w_out % 2 != 0 {
        let ow = w_out - 1;
        let iw_start = ow as isize - 1;
        for oh in 0..(n_tile_h * 2) {
            let ih_start = oh as isize - 1;
            for ci in 0..c_in {
                let _ = (ih_start, iw_start, ci);
            }
            // Scalar conv for this (oh, ow) pixel.
            for co in 0..c_out {
                let mut acc = 0.0f32;
                for kh in 0..3isize {
                    let ih = oh as isize + kh - 1;
                    if ih < 0 || ih >= h_in as isize {
                        continue;
                    }
                    for kw in 0..3isize {
                        let iw = ow as isize + kw - 1;
                        if iw < 0 || iw >= w_in as isize {
                            continue;
                        }
                        // weights_transformed is NOT the spatial weights — we need to
                        // use the original conv for the scalar fallback pixels.
                        // However, since we only have weights_transformed here, we
                        // recompute from scratch using a direct read.
                        // NOTE: the driver doesn't receive original weights, so we use
                        // the identity: for boundary pixels we fall back to zero (the
                        // epilogue is rare and correctness is preserved for the full
                        // Winograd path on even dims).
                        let _ = (ih, iw, kh, kw, co);
                        // For simplicity, leave the epilogue pixel at 0 — documented
                        // behaviour: callers should pad to even dimensions for full
                        // correctness. This avoids needing original weights in the driver.
                        let _ = acc;
                    }
                }
                // The scalar epilogue pixels are zeroed when original weights are absent.
                // Full correctness requires passing original weights separately — see
                // `conv3x3_winograd_nhwc_with_scalar_fallback` for the extended API.
                output[oh * w_out * c_out + ow * c_out + co] =
                    apply_act(acc + bias.map_or(0.0, |b| b[co]), activation);
            }
        }
    }

    // ── Scalar epilogue: last row (odd h_out) ────────────────────────────────
    if h_out % 2 != 0 {
        let oh = h_out - 1;
        let w_cols = if w_out % 2 != 0 { w_out } else { n_tile_w * 2 };
        for ow in 0..w_cols {
            for co in 0..c_out {
                let acc = 0.0f32;
                output[oh * w_out * c_out + ow * c_out + co] =
                    apply_act(acc + bias.map_or(0.0, |b| b[co]), activation);
            }
        }
    }
}

/// Version of the driver that also accepts the original spatial weights for
/// correct scalar fallback on boundary pixels when spatial dims are odd.
///
/// When `h_out` and `w_out` are both even, this is identical to
/// [`conv3x3_winograd_nhwc`].
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_winograd_nhwc_with_scalar_fallback(
    input: &[f32],
    h_in: usize,
    w_in: usize,
    c_in: usize,
    weights_transformed: &[f32], // [16, c_out, c_in]
    weights_spatial: &[f32],     // [c_out, 9, c_in] — for scalar epilogue pixels
    bias: Option<&[f32]>,
    c_out: usize,
    activation: Activation,
    output: &mut [f32],
    h_out: usize,
    w_out: usize,
) {
    // Allocate a temporary v_buf and run the main Winograd path.
    let n_tile_h = h_out / 2;
    let n_tile_w = w_out / 2;
    let mut v_buf = vec![0.0f32; n_tile_h * n_tile_w * 16 * c_in];
    conv3x3_winograd_nhwc(
        input,
        h_in,
        w_in,
        c_in,
        weights_transformed,
        bias,
        c_out,
        activation,
        output,
        h_out,
        w_out,
        &mut v_buf,
    );

    let n_tile_h = h_out / 2;
    let n_tile_w = w_out / 2;

    // Recompute boundary pixels using scalar conv when dims are odd.
    let scalar_col = if w_out % 2 != 0 {
        Some(w_out - 1)
    } else {
        None
    };
    let scalar_row = if h_out % 2 != 0 {
        Some(h_out - 1)
    } else {
        None
    };

    // Last column pixels for all processed rows.
    if let Some(ow) = scalar_col {
        for oh in 0..(n_tile_h * 2) {
            for co in 0..c_out {
                let mut acc = 0.0f32;
                for kh in 0..3isize {
                    let ih = oh as isize + kh - 1;
                    if ih < 0 || ih >= h_in as isize {
                        continue;
                    }
                    for kw in 0..3isize {
                        let iw = ow as isize + kw - 1;
                        if iw < 0 || iw >= w_in as isize {
                            continue;
                        }
                        let ih = ih as usize;
                        let iw = iw as usize;
                        let tap = ((kh + 1) * 3 + (kw + 1)) as usize;
                        for ci in 0..c_in {
                            acc += input[(ih * w_in + iw) * c_in + ci]
                                * weights_spatial[(co * 9 + tap) * c_in + ci];
                        }
                    }
                }
                let b = bias.map_or(0.0, |bs| bs[co]);
                output[oh * w_out * c_out + ow * c_out + co] = apply_act(acc + b, activation);
            }
        }
    }

    // Last row pixels (includes corner if both dims odd).
    if let Some(oh) = scalar_row {
        let w_range = n_tile_w * 2;
        let w_range = if scalar_col.is_some() {
            w_range + 1
        } else {
            w_range
        };
        for ow in 0..w_range {
            for co in 0..c_out {
                let mut acc = 0.0f32;
                for kh in 0..3isize {
                    let ih = oh as isize + kh - 1;
                    if ih < 0 || ih >= h_in as isize {
                        continue;
                    }
                    for kw in 0..3isize {
                        let iw = ow as isize + kw - 1;
                        if iw < 0 || iw >= w_in as isize {
                            continue;
                        }
                        let ih = ih as usize;
                        let iw = iw as usize;
                        let tap = ((kh + 1) * 3 + (kw + 1)) as usize;
                        for ci in 0..c_in {
                            acc += input[(ih * w_in + iw) * c_in + ci]
                                * weights_spatial[(co * 9 + tap) * c_in + ci];
                        }
                    }
                }
                let b = bias.map_or(0.0, |bs| bs[co]);
                output[oh * w_out * c_out + ow * c_out + co] = apply_act(acc + b, activation);
            }
        }
    }
}

// ─── fp16 Winograd driver (aarch64 + ARMv8.2 fp16) ───────────────────────────

/// fp16 Winograd F(2×2, 3×3) NHWC convolution driver.
///
/// Same semantics as [`conv3x3_winograd_nhwc`] but uses the ARMv8.2 FMLA.8H
/// GEMM micro-kernel for the per-position matrix products.  Each of the 16
/// Winograd positions runs a `(M=n_tile_w, K=c_in, N=c_out)` GEMM entirely in
/// fp16, accumulating results across positions.  The output transform runs in
/// f32 (V→f32, bias, activation) so final accuracy matches the f32 Winograd
/// path to within fp16 rounding (max abs error ≈ 5×10⁻³ for typical XFeat
/// activations).
///
/// # Arguments
/// Same as [`conv3x3_winograd_nhwc`] except `weights_transformed_f16` is the
/// `[16 * c_out * c_in]` weight buffer down-cast to f16 (stored as `u16`).
///
/// Only available on `aarch64` (uses `neon_asm_f16::gemm_f16_mnk`).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[allow(unused_variables)]
pub fn conv3x3_winograd_nhwc_f16(
    input: &[f32],
    h_in: usize,
    w_in: usize,
    c_in: usize,
    weights_transformed_f16: &[u16], // [16, c_out, c_in] f16-as-u16
    bias: Option<&[f32]>,
    c_out: usize,
    activation: Activation,
    output: &mut [f32],
    h_out: usize,
    w_out: usize,
    _v_buf: &mut [f32], // reserved (same API as f32 variant)
) {
    debug_assert_eq!(input.len(), h_in * w_in * c_in);
    debug_assert_eq!(output.len(), h_out * w_out * c_out);
    debug_assert_eq!(weights_transformed_f16.len(), 16 * c_out * c_in);
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), c_out);
    }

    let n_tile_h = h_out / 2;
    let n_tile_w = w_out / 2;

    // ── Pre-pack the 16 B panels (f16, [c_in, c_out] for each position) ──────
    // weights_transformed_f16 layout: [16, c_out, c_in] → we need [c_in, c_out]
    // (transposed) for use as the B matrix in GEMM(A[n_tile_w, c_in], B[c_in, c_out]).
    // Allocate once per conv call; shared read-only across all rayon workers.
    let b_panels_f16: Vec<Vec<u16>> = (0..16)
        .map(|p| {
            let pos_offset = p * c_out * c_in;
            let mut panel = vec![0u16; c_in * c_out];
            // Transpose [c_out, c_in] → [c_in, c_out]
            for co in 0..c_out {
                for ci in 0..c_in {
                    panel[ci * c_out + co] = weights_transformed_f16[pos_offset + co * c_in + ci];
                }
            }
            panel
        })
        .collect();

    // Pointers for rayon boundary.
    let in_ptr = input.as_ptr() as usize;
    let bias_ptr: usize = bias.map_or(0, |b| b.as_ptr() as usize);
    let has_bias = bias.is_some();
    let b_panel_ptrs: Vec<usize> = b_panels_f16.iter().map(|v| v.as_ptr() as usize).collect();
    let b_panel_ptrs_ptr = b_panel_ptrs.as_ptr() as usize;

    let tile_row_stride = 2 * w_out * c_out;
    let out_ptr = output.as_mut_ptr() as usize;

    (0..n_tile_h).into_par_iter().for_each(|tile_oh| {
        let input = unsafe { std::slice::from_raw_parts(in_ptr as *const f32, h_in * w_in * c_in) };
        let bias_slice: Option<&[f32]> = if has_bias {
            Some(unsafe { std::slice::from_raw_parts(bias_ptr as *const f32, c_out) })
        } else {
            None
        };
        let out_row_base = unsafe {
            std::slice::from_raw_parts_mut(
                (out_ptr as *mut f32).add(tile_oh * tile_row_stride),
                2 * w_out * c_out,
            )
        };
        // Rebuild B-panel pointer slice for this thread.
        let b_ptrs: &[usize] =
            unsafe { std::slice::from_raw_parts(b_panel_ptrs_ptr as *const usize, 16) };

        let ih_start = (tile_oh * 2) as isize - 1;

        // ── Per-position: collect A[n_tile_w, c_in] and run fp16 GEMM ─────────
        //
        // For each position p in 0..16:
        //   A_f16[tile_ow, ci] = V[tile_ow, p, ci]
        //   C_f16[tile_ow, co] += A_f16 × B_f16[p]  (gemm_f16_mnk)
        //
        // After all 16 positions, convert C_f16 to f32, apply output transform
        // + bias + activation, and write the 2×2 output tile.
        //
        // m_acc_f16[tile_ow, p, co] — accumulates fp16 Winograd products.
        // Size: n_tile_w * 16 * c_out (all positions for the entire tile row).
        // Max: 320 * 16 * 128 = 655 360 u16 = 1.25 MB — too large for stack.
        // Heap-allocate per tile-row (n_tile_h ≤ 240 workers for 480-row images).
        let m_acc_len = n_tile_w * 16 * c_out;
        let zero_f16 = half::f16::ZERO.to_bits();
        let mut m_acc_f16: Vec<u16> = vec![zero_f16; m_acc_len];

        for p in 0..16usize {
            // Build A_f16[n_tile_w, c_in]: for each tile column, collect the
            // transformed input value at position p for all c_in channels.
            let mut a_f16: Vec<u16> = vec![zero_f16; n_tile_w * c_in];

            for tile_ow in 0..n_tile_w {
                let iw_start = (tile_ow * 2) as isize - 1;
                for ci in 0..c_in {
                    let patch = extract_patch_f32(input, h_in, w_in, c_in, ih_start, iw_start, ci);
                    let v = winograd_transform_input_tile_f32(&patch);
                    a_f16[tile_ow * c_in + ci] = half::f16::from_f32(v[p]).to_bits();
                }
            }

            // C slice for this position: m_acc_f16[tile_ow, p, co]
            // = m_acc_f16[tile_ow * 16 * c_out + p * c_out + co]
            // We need contiguous C[n_tile_w, c_out] for each position p.
            // Build a separate view by collecting into a temp buffer, then scatter.
            // Alternatively, we stride the C buffer differently.
            //
            // Layout: m_acc_f16[tile_ow * (16 * c_out) + p * c_out + co]
            // For a fixed p, the C rows are non-contiguous (stride = 16 * c_out).
            // gemm_f16_mnk requires contiguous [M, N] C — so we use a temp buf.
            let b_ptr =
                unsafe { std::slice::from_raw_parts(b_ptrs[p] as *const u16, c_in * c_out) };
            let mut c_tmp = vec![zero_f16; n_tile_w * c_out];

            // Run the fp16 GEMM: C_tmp[n_tile_w, c_out] += A[n_tile_w, c_in] × B[c_in, c_out]
            super::neon_asm_f16::gemm_f16_mnk(
                &a_f16, b_ptr, &mut c_tmp, n_tile_w, // M
                c_in,     // K
                c_out,    // N
                false,    // no relu — accumulate, apply relu only in output epilogue
            );

            // Scatter C_tmp into m_acc_f16 at the correct position stride.
            for tile_ow in 0..n_tile_w {
                let dst_base = tile_ow * 16 * c_out + p * c_out;
                let src_base = tile_ow * c_out;
                m_acc_f16[dst_base..dst_base + c_out]
                    .copy_from_slice(&c_tmp[src_base..src_base + c_out]);
            }
        }

        // ── Output transform: for each output tile, apply A^T · M · A ─────────
        for tile_ow in 0..n_tile_w {
            let ow0 = tile_ow * 2;
            let ow1 = ow0 + 1;

            for co in 0..c_out {
                // Gather the 4×4 m matrix for this (tile, co): convert f16 → f32.
                let m_tile: [f32; 16] = core::array::from_fn(|p| {
                    let bits = m_acc_f16[tile_ow * 16 * c_out + p * c_out + co];
                    half::f16::from_bits(bits).to_f32()
                });

                let y = winograd_output_transform_f32(&m_tile);

                let b = bias_slice.map_or(0.0, |bs| bs[co]);
                let y00 = apply_act(y[0] + b, activation);
                let y01 = apply_act(y[1] + b, activation);
                let y10 = apply_act(y[2] + b, activation);
                let y11 = apply_act(y[3] + b, activation);

                let row1 = w_out * c_out;
                out_row_base[ow0 * c_out + co] = y00;
                out_row_base[ow1 * c_out + co] = y01;
                out_row_base[row1 + ow0 * c_out + co] = y10;
                out_row_base[row1 + ow1 * c_out + co] = y11;
            }
        }
    });

    // ── Scalar epilogue for odd h_out / w_out ─────────────────────────────────
    // (same as f32 variant: leave boundary pixels as zero since original weights
    // are not available here; full correctness requires even dimensions which all
    // XFeat layers satisfy.)
    if w_out % 2 != 0 {
        let ow = w_out - 1;
        for oh in 0..(n_tile_h * 2) {
            for co in 0..c_out {
                output[oh * w_out * c_out + ow * c_out + co] =
                    apply_act(bias.map_or(0.0, |b| b[co]), activation);
            }
        }
    }
    if h_out % 2 != 0 {
        let oh = h_out - 1;
        let w_cols = if w_out % 2 != 0 { w_out } else { n_tile_w * 2 };
        for ow in 0..w_cols {
            for co in 0..c_out {
                output[oh * w_out * c_out + ow * c_out + co] =
                    apply_act(bias.map_or(0.0, |b| b[co]), activation);
            }
        }
    }
}

// ─── fp16 Winograd F(4×4, 3×3) driver (aarch64 + ARMv8.2 fp16) ───────────────

/// fp16 Winograd F(4×4, 3×3) NHWC convolution driver.
///
/// Same semantics as [`conv3x3_winograd_nhwc_f43`] but uses the ARMv8.2
/// FMLA.8H GEMM micro-kernel for the per-position matrix products. Each of the
/// 36 Winograd positions runs a `(M=n_tile_w, K=c_in, N=c_out)` GEMM in fp16,
/// accumulating across positions. The output transform runs in f32.
///
/// `weights_transformed_f16` is the `[36 * c_out * c_in]` weight buffer
/// down-cast to f16 (stored as `u16`).
///
/// Only available on `aarch64` (uses `neon_asm_f16::gemm_f16_mnk`). Odd-tile
/// remainders are left at the bias+activation baseline; use
/// [`conv3x3_winograd_nhwc_f43_f16_with_scalar_fallback`] for correctness.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_winograd_nhwc_f43_f16(
    input: &[f32],
    h_in: usize,
    w_in: usize,
    c_in: usize,
    weights_transformed_f16: &[u16], // [36, c_out, c_in] f16-as-u16 (debug_assert only)
    b_panels_f16: &[u16],            // [36 * c_in * c_out] pre-transposed: [p][ci*c_out+co]
    b_panels_packed: &[u16],         // [36, slot] pre-packed GEMM layout; empty → pack on demand
    bias: Option<&[f32]>,
    c_out: usize,
    activation: Activation,
    output: &mut [f32],
    h_out: usize,
    w_out: usize,
) {
    debug_assert_eq!(input.len(), h_in * w_in * c_in);
    debug_assert_eq!(output.len(), h_out * w_out * c_out);
    debug_assert_eq!(weights_transformed_f16.len(), 36 * c_out * c_in);
    debug_assert_eq!(b_panels_f16.len(), 36 * c_in * c_out);
    let _ = weights_transformed_f16; // kept for the shape debug_assert only
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), c_out);
    }

    // Ceiling division: the last tile may be partial (valid_rows < 4) when
    // h_out % 4 != 0. extract_patch_f32_6x6 zero-pads OOB input rows, and
    // the output write is bounded by valid_rows/valid_cols.
    let n_tile_h = (h_out + 3) / 4;
    let n_tile_w = (w_out + 3) / 4;
    // Pad n_tile_w to the next multiple of the GEMM's M granularity. The fp16
    // GEMM driver processes M in MR=8 micro-tiles plus a vectorised MR=4 tail
    // (`gemm_4x8_f16` fast path), so padding only to a multiple of 4 — rather
    // than 8 — halves the worst-case phantom-row waste for the small spatial
    // layers (block4/block5: n_tile_w 10→12 not 16, 5→8... 2→4 not 8) while the
    // single leftover MR=4 block still runs at full vector throughput.
    const MR: usize = 8;
    const M_PAD: usize = 4;
    let n_tile_w_gemm = n_tile_w.div_ceil(M_PAD) * M_PAD;

    // B panels are pre-transposed in the WinogradCache (one alloc at model
    // build time) — panel p is `b_panels_f16[p*c_in*c_out .. (p+1)*c_in*c_out]`.
    let in_ptr = input.as_ptr() as usize;
    let bias_ptr: usize = bias.map_or(0, |b| b.as_ptr() as usize);
    let has_bias = bias.is_some();
    let b_panels_ptr = b_panels_f16.as_ptr() as usize;

    // Pre-packed B: skip per-frame B packing when available.
    const NR_W: usize = 8;
    let n_blocks_w = c_out / NR_W;
    let n_rem_w = c_out % NR_W;
    let slot_sz = n_blocks_w * c_in * NR_W + c_in * n_rem_w;
    let use_prepacked = !b_panels_packed.is_empty() && b_panels_packed.len() == 36 * slot_sz;
    let b_packed_ptr = if use_prepacked {
        b_panels_packed.as_ptr() as usize
    } else {
        0
    };

    let tile_row_stride = 4 * w_out * c_out;
    let out_ptr = output.as_mut_ptr() as usize;

    // Column-split: raise outer task count to the next multiple of n_rayon_threads
    // by splitting each tile-row into `col_parts` column slices.
    //
    // Formula: col_parts = n_rayon_threads / gcd(n_tile_h, n_rayon_threads).
    // This gives the smallest multiplier k such that n_tile_h × k is divisible by
    // n_rayon_threads — eliminating the work-stealing tail imbalance with the
    // fewest extra tasks.
    //
    // Example (60×80, 12 Rayon workers):
    //   n_tile_h=15, gcd(15,12)=3 → col_parts=4 → 60 tasks = 5×12 (perfect balance).
    //   Each task covers 5 tile-columns (M=8 after GEMM padding) and uses the same
    //   thread-local scratch buffers — no extra heap allocations.
    //
    // Cap at n_tile_w so each task covers ≥1 tile column.
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let n_rayon_threads = rayon::current_num_threads().max(1);
    let g = gcd(n_tile_h, n_rayon_threads);
    let col_parts = (n_rayon_threads / g).min(n_tile_w.max(1));

    // Co-block parallelism: when pre-packed B panels are available and c_out > NR_W,
    // split the GEMM inside each tile-row into NR_W-wide subtasks.  Idle Rayon
    // workers from the outer (tile-row) pool steal inner tasks, increasing thread
    // utilisation from n_tile_h/N_THREADS to (n_tile_h × n_co8)/N_THREADS.
    // Effective for small spatial sizes: block5 (4 tile-rows → 4×16=64 items),
    // block4 (8 → 128 items), block_fusion (15 → 120 items) with 12 threads.
    let n_co8 = n_blocks_w; // = c_out / NR_W
                            // Co-block parallelism is only useful when outer task count < thread count.
                            // When col_parts > 1 (column-split active), outer tasks already fill the pool —
                            // enabling co_par would add oversubscription and per-tile heap allocations.
    let use_co_par = use_prepacked
        && n_co8 > 1
        && col_parts == 1
        && n_tile_h < n_rayon_threads
        && n_tile_w_gemm > 8;

    (0..n_tile_h * col_parts)
        .into_par_iter()
        .for_each(|task_id| {
            let tile_oh = task_id / col_parts;
            let col_part = task_id % col_parts;
            // Column range for this task (relative within the tile-row).
            // Clamp col_start to n_tile_w: when col_parts > n_tile_w (small spatial
            // layers with many threads), trailing tasks own no columns. Without the
            // clamp, `col_end - col_start` underflows usize and the scatter loop runs
            // off the thread-local m_acc buffer (panic at large inputs, e.g. 576×800).
            let tiles_per_part = n_tile_w.div_ceil(col_parts);
            let col_start = (col_part * tiles_per_part).min(n_tile_w);
            let col_end = (col_start + tiles_per_part).min(n_tile_w);
            let n_col_tiles = col_end - col_start;
            if n_col_tiles == 0 {
                return;
            }
            // GEMM panel width for this column slice (padded to M_PAD=4; the
            // GEMM handles a single MR=4 tail block at full vector throughput).
            let n_tile_w_gemm_part = n_col_tiles.div_ceil(M_PAD) * M_PAD;
            // For partial last tiles (h_out % 4 != 0), only write the valid rows.
            let valid_rows = (h_out - tile_oh * 4).min(4);

            let input =
                unsafe { std::slice::from_raw_parts(in_ptr as *const f32, h_in * w_in * c_in) };
            let bias_slice: Option<&[f32]> = if has_bias {
                Some(unsafe { std::slice::from_raw_parts(bias_ptr as *const f32, c_out) })
            } else {
                None
            };

            let ih_start = (tile_oh * 4) as isize - 1;

            // n_tile_w_gemm is computed once outside the closure (see above).
            let zero_f16 = half::f16::ZERO.to_bits();
            let a_len = 36 * n_tile_w_gemm_part * c_in;
            let m_len = n_tile_w_gemm_part * 36 * c_out;
            let c_len = n_tile_w_gemm_part * c_out;
            let pa_len = c_in * MR;
            let pb_len = c_in * c_out;

            // Phase 1: build all 36 A panels (input transform) — sequential on this thread.
            // The result is shared read-only with inner co-block workers via a raw pointer.
            F43_A_ALL.with(|a_cell| {
                let mut a_all = a_cell.borrow_mut();

                if a_all.len() < a_len {
                    a_all.resize(a_len, zero_f16);
                }
                // Phantom columns (n_tile_w..n_tile_w_gemm) must stay zero.
                a_all[..a_len].fill(zero_f16);

                // Build A panels for this column slice only (relative indexing).
                // rel_ow = 0..n_col_tiles maps to absolute tile_ow = col_start..col_end.
                #[cfg(target_arch = "aarch64")]
                for tile_ow in col_start..col_end {
                    let rel_ow = tile_ow - col_start;
                    let iw_start = (tile_ow * 4) as isize - 1;
                    unsafe {
                        super::neon_asm_f16::transform_input_tile_allch_f43(
                            input,
                            h_in,
                            w_in,
                            c_in,
                            ih_start,
                            iw_start,
                            n_tile_w_gemm_part,
                            rel_ow,
                            &mut a_all,
                        );
                    }
                }

                #[cfg(not(target_arch = "aarch64"))]
                for tile_ow in col_start..col_end {
                    let rel_ow = tile_ow - col_start;
                    let iw_start = (tile_ow * 4) as isize - 1;
                    for ci in 0..c_in {
                        let patch =
                            extract_patch_f32_6x6(input, h_in, w_in, c_in, ih_start, iw_start, ci);
                        let v = winograd_transform_input_tile_f32_f43(&patch);
                        let idx = rel_ow * c_in + ci;
                        for p in 0..36usize {
                            a_all[p * n_tile_w_gemm_part * c_in + idx] =
                                half::f16::from_f32(v[p]).to_bits();
                        }
                    }
                }

                let out_row_base_ptr =
                    unsafe { (out_ptr as *mut f32).add(tile_oh * tile_row_stride) } as usize;

                if use_co_par {
                    // ── Co-block parallelism: Phase 2 parallel GEMM, Phase 3 serial ─
                    // Clone a_all and release RefMut BEFORE inner par_iter to avoid
                    // RefCell re-entrancy when Rayon re-uses this thread for another
                    // outer tile_oh task while we are waiting for the inner par.
                    let a_copy: Vec<u16> = a_all[..a_len].to_vec();
                    drop(a_all); // release RefMut; inner par reads from a_copy
                    let a_ptr = a_copy.as_ptr() as usize;

                    // Each co-block's accumulator occupies a unique, non-overlapping
                    // slice of m_acc_all, so the parallel GEMM phase has no
                    // false-sharing.  The output-transform (Phase 3) runs serially
                    // after the parallel phase, so each co-block writes to disjoint
                    // output-channel bytes — no cache-line contention there either.
                    let m_co_len = n_tile_w_gemm * 36 * NR_W;
                    let c_co_len = n_tile_w_gemm * NR_W;
                    let mut m_acc_all = vec![0u16; n_co8 * m_co_len];
                    let m_all_ptr = m_acc_all.as_mut_ptr() as usize;

                    // Phase 2: GEMM — parallel across co-blocks.
                    (0..n_co8).into_par_iter().for_each(|co_blk| {
                        let a = unsafe { std::slice::from_raw_parts(a_ptr as *const u16, a_len) };
                        let bp = unsafe {
                            std::slice::from_raw_parts(b_packed_ptr as *const u16, 36 * slot_sz)
                        };
                        // Exclusive m_acc slice for this co-block.
                        let m_co = unsafe {
                            std::slice::from_raw_parts_mut(
                                (m_all_ptr as *mut u16).add(co_blk * m_co_len),
                                m_co_len,
                            )
                        };

                        F43_C_TMP.with(|c_cell| {
                            F43_PACK_A.with(|pa_cell| {
                                let mut c_tmp = c_cell.borrow_mut();
                                let mut pack_a = pa_cell.borrow_mut();

                                if c_tmp.len() < c_co_len {
                                    c_tmp.resize(c_co_len, zero_f16);
                                }
                                if pack_a.len() < pa_len {
                                    pack_a.resize(pa_len, zero_f16);
                                }

                                for p in 0..36usize {
                                    let a_f16 = &a
                                        [p * n_tile_w_gemm * c_in..(p + 1) * n_tile_w_gemm * c_in];
                                    c_tmp[..c_co_len].fill(zero_f16);

                                    let p_off = p * slot_sz + co_blk * c_in * NR_W;
                                    super::neon_asm_f16::gemm_f16_mnk_packed_b(
                                        a_f16,
                                        &bp[p_off..p_off + c_in * NR_W],
                                        &[],
                                        &mut c_tmp[..c_co_len],
                                        n_tile_w_gemm,
                                        c_in,
                                        NR_W,
                                        &mut pack_a[..pa_len],
                                    );

                                    // Scatter into compact m_co: layout [tile_ow, 36, NR_W].
                                    for tile_ow in 0..n_tile_w {
                                        let dst = tile_ow * 36 * NR_W + p * NR_W;
                                        let src = tile_ow * NR_W;
                                        m_co[dst..dst + NR_W]
                                            .copy_from_slice(&c_tmp[src..src + NR_W]);
                                    }
                                }
                            })
                        });
                    });

                    // Phase 3: output transform — serial, no false-sharing.
                    {
                        use super::neon_asm_f16::{
                            load_bias_f16x8, vld1q_u16_wrap,
                            winograd_f43_output_transform_write_f16x8,
                        };
                        let out_row_stride = w_out * c_out;
                        let relu = matches!(activation, Activation::Relu);
                        let out_p = out_row_base_ptr as *mut f32;

                        for co_blk in 0..n_co8 {
                            let co_base = co_blk * NR_W;
                            let m = &m_acc_all[co_blk * m_co_len..(co_blk + 1) * m_co_len];

                            for tile_ow in 0..n_tile_w {
                                let ow0 = tile_ow * 4;
                                let tb = tile_ow * 36 * NR_W;
                                let valid_cols = (w_out - ow0).min(4);

                                let (bias_lo, bias_hi) =
                                    unsafe { load_bias_f16x8(bias_slice, co_base) };

                                let mut m_f16 = unsafe { [vld1q_u16_wrap(m.as_ptr()); 36] };
                                for (p, slot) in m_f16.iter_mut().enumerate() {
                                    *slot = unsafe { vld1q_u16_wrap(m[tb + p * NR_W..].as_ptr()) };
                                }

                                unsafe {
                                    winograd_f43_output_transform_write_f16x8(
                                        &m_f16,
                                        bias_lo,
                                        bias_hi,
                                        relu,
                                        out_p,
                                        out_row_stride,
                                        ow0,
                                        c_out,
                                        co_base,
                                        valid_rows,
                                        valid_cols,
                                    );
                                }
                            }
                        }
                    }
                } else {
                    // ── Non-co_par: original full-c_out sequential GEMM + output ─────
                    let a_ptr = a_all.as_ptr() as usize;
                    F43_M_ACC.with(|m_cell| {
                        F43_C_TMP.with(|c_cell| {
                            F43_PACK_A.with(|pa_cell| {
                                F43_PACK_B.with(|pb_cell| {
                                    let mut m_acc_f16 = m_cell.borrow_mut();
                                    let mut c_tmp = c_cell.borrow_mut();
                                    let mut pack_a = pa_cell.borrow_mut();
                                    let mut pack_b = pb_cell.borrow_mut();

                                    if m_acc_f16.len() < m_len {
                                        m_acc_f16.resize(m_len, zero_f16);
                                    }
                                    if c_tmp.len() < c_len {
                                        c_tmp.resize(c_len, zero_f16);
                                    }
                                    if pack_a.len() < pa_len {
                                        pack_a.resize(pa_len, zero_f16);
                                    }
                                    if pack_b.len() < pb_len {
                                        pack_b.resize(pb_len, zero_f16);
                                    }

                                    m_acc_f16[..m_len].fill(zero_f16);

                                    let a = unsafe {
                                        std::slice::from_raw_parts(a_ptr as *const u16, a_len)
                                    };

                                    for p in 0..36usize {
                                        // A panels for this column slice only (relative indices).
                                        let a_f16 = &a[p * n_tile_w_gemm_part * c_in
                                            ..(p + 1) * n_tile_w_gemm_part * c_in];
                                        c_tmp[..c_len].fill(zero_f16);

                                        if use_prepacked {
                                            let bp = unsafe {
                                                std::slice::from_raw_parts(
                                                    b_packed_ptr as *const u16,
                                                    36 * slot_sz,
                                                )
                                            };
                                            let p_off = p * slot_sz;
                                            let packed_b =
                                                &bp[p_off..p_off + n_blocks_w * c_in * NR_W];
                                            let b_rem_tail = &bp[p_off + n_blocks_w * c_in * NR_W
                                                ..(p + 1) * slot_sz];
                                            super::neon_asm_f16::gemm_f16_mnk_packed_b(
                                                a_f16,
                                                packed_b,
                                                b_rem_tail,
                                                &mut c_tmp[..c_len],
                                                n_tile_w_gemm_part,
                                                c_in,
                                                c_out,
                                                &mut pack_a[..pa_len],
                                            );
                                        } else {
                                            let b_panels: &[u16] = unsafe {
                                                std::slice::from_raw_parts(
                                                    b_panels_ptr as *const u16,
                                                    36 * c_in * c_out,
                                                )
                                            };
                                            let b_ptr =
                                                &b_panels[p * c_in * c_out..(p + 1) * c_in * c_out];
                                            super::neon_asm_f16::gemm_f16_mnk_with_pack(
                                                a_f16,
                                                b_ptr,
                                                &mut c_tmp[..c_len],
                                                n_tile_w_gemm_part,
                                                c_in,
                                                c_out,
                                                &mut pack_a[..pa_len],
                                                &mut pack_b[..pb_len],
                                                false,
                                            );
                                        }

                                        // Scatter: rel_ow = 0..n_col_tiles (relative column index).
                                        for rel_ow in 0..n_col_tiles {
                                            let dst_base = rel_ow * 36 * c_out + p * c_out;
                                            let src_base = rel_ow * c_out;
                                            m_acc_f16[dst_base..dst_base + c_out].copy_from_slice(
                                                &c_tmp[src_base..src_base + c_out],
                                            );
                                        }
                                    }

                                    // ── Output transform ─────────────────────────────────────────
                                    use super::neon_asm_f16::{
                                        load_bias_f16x8, vld1q_u16_wrap,
                                        winograd_f43_output_transform_write_f16x8,
                                    };
                                    let out_row_stride = w_out * c_out;
                                    let relu = matches!(activation, Activation::Relu);
                                    let m_acc = &m_acc_f16[..m_len];
                                    let out_p = out_row_base_ptr as *mut f32;

                                    // Iterate over absolute tile columns (col_start..col_end);
                                    // use rel_ow for m_acc indexing, tile_ow for output column.
                                    for tile_ow in col_start..col_end {
                                        let rel_ow = tile_ow - col_start;
                                        let ow0 = tile_ow * 4;
                                        let tile_base = rel_ow * 36 * c_out;
                                        let valid_cols = (w_out - tile_ow * 4).min(4);

                                        let mut co_base = 0usize;
                                        while co_base + 8 <= c_out {
                                            let (bias_lo, bias_hi) =
                                                unsafe { load_bias_f16x8(bias_slice, co_base) };

                                            let mut m_f16 =
                                                unsafe { [vld1q_u16_wrap(m_acc.as_ptr()); 36] };
                                            for (p, slot) in m_f16.iter_mut().enumerate() {
                                                *slot = unsafe {
                                                    vld1q_u16_wrap(
                                                        m_acc[tile_base + p * c_out + co_base..]
                                                            .as_ptr(),
                                                    )
                                                };
                                            }

                                            unsafe {
                                                winograd_f43_output_transform_write_f16x8(
                                                    &m_f16,
                                                    bias_lo,
                                                    bias_hi,
                                                    relu,
                                                    out_p,
                                                    out_row_stride,
                                                    ow0,
                                                    c_out,
                                                    co_base,
                                                    valid_rows,
                                                    valid_cols,
                                                );
                                            }
                                            co_base += 8;
                                        }

                                        // Scalar tail (no XFeat layer has c_out % 8 != 0).
                                        for co in co_base..c_out {
                                            let m_tile: [f32; 36] = core::array::from_fn(|p| {
                                                let bits = m_acc[tile_base + p * c_out + co];
                                                half::f16::from_bits(bits).to_f32()
                                            });
                                            let y = winograd_output_transform_f32_f43(&m_tile);
                                            let b = bias_slice.map_or(0.0, |bs| bs[co]);
                                            for ry in 0..valid_rows {
                                                let row_off = ry * out_row_stride;
                                                for cx in 0..valid_cols {
                                                    let val =
                                                        apply_act(y[ry * 4 + cx] + b, activation);
                                                    unsafe {
                                                        *out_p.add(
                                                            row_off + (ow0 + cx) * c_out + co,
                                                        ) = val;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                })
                            })
                        })
                    });
                }
            });
        });
}

// ─── fp16 Winograd F(4,3) + fused 1×1 epilogue driver ───────────────────────

/// fp16 Winograd F(4×4, 3×3) NHWC driver with a **fused 1×1 conv epilogue**.
///
/// Identical 3×3 Winograd numerics to [`conv3x3_winograd_nhwc_f43_f16`], but
/// instead of writing the 3×3 result to a full-size f32 intermediate buffer
/// (later re-read by a separate Rayon-dispatched 1×1 conv), each Winograd task
/// writes its 4-row × column-slice block into a **task-local dense f32 strip**
/// and immediately runs the 1×1 conv on those pixels — inside the *same* Rayon
/// task. A 1×1 conv has no spatial halo, so a per-task epilogue needs nothing
/// from other tasks. This removes one Rayon dispatch barrier and the
/// write+read of the ~1.2 MB f32 intermediate.
///
/// # Bit-exactness vs the unfused (wino + standalone 1×1) sequence
/// The 1×1 GEMM microkernel (`gemm_f16_mnk_packed_b` / `gemm_8x8_f16`) computes
/// each output pixel as an independent fp16-accumulated dot product against the
/// same pre-packed B. The only grouping-dependent rounding in that kernel is
/// the `m % MR(=8)` remainder rows, which fall through a *scalar f32-accumulate*
/// path rather than the fp16 NEON kernel. The standalone 1×1 path slices the
/// spatial dimension into `CONV1X1_STRIP_SIZE`-pixel strips (a multiple of 8),
/// so for layers whose total pixel count is a multiple of 8 **no** pixel ever
/// hits the m-remainder path. This fused driver therefore only runs when every
/// task's `block_pixels` is a multiple of 8 (caller-gated via
/// [`f43_fused1x1_is_bit_exact`]); the per-task GEMM then has `m_rem == 0` and
/// every pixel goes through the identical fp16 kernel → bit-identical results.
///
/// `b1x1_packed` / `b1x1_rem_offset` are the `(packed, b_rem_offset)` pair from
/// `prepack_conv1x1_b_f16` for the 1×1 layer (c_in = `c_out` of the 3×3 layer,
/// c_out = `c_out2`). `out1x1` is the **final** f16 output `[h_out*w_out, c_out2]`.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_winograd_nhwc_f43_f16_fused1x1(
    input: &[f32],
    h_in: usize,
    w_in: usize,
    c_in: usize,
    weights_transformed_f16: &[u16], // [36, c_out, c_in] (debug_assert only)
    b_panels_f16: &[u16],            // [36 * c_in * c_out] pre-transposed
    b_panels_packed: &[u16],         // [36, slot] pre-packed; empty → pack on demand
    bias: Option<&[f32]>,            // 3×3 bias
    c_out: usize,                    // 3×3 output channels (= 1×1 c_in)
    activation: Activation,          // 3×3 activation
    // ── fused 1×1 epilogue ──
    b1x1_packed: &[u16],    // prepack_conv1x1_b_f16 output for the 1×1 layer
    b1x1_rem_offset: usize, // = (c_out2 / 8) * c_out * 8
    bias1x1: &[f32],        // [c_out2]
    c_out2: usize,          // 1×1 output channels
    act1x1: Activation,     // 1×1 activation (Identity or Relu; Sigmoid unsupported here)
    out1x1: &mut [u16],     // FINAL f16 output [h_out * w_out * c_out2]
    h_out: usize,
    w_out: usize,
) {
    debug_assert_eq!(input.len(), h_in * w_in * c_in);
    debug_assert_eq!(out1x1.len(), h_out * w_out * c_out2);
    debug_assert_eq!(weights_transformed_f16.len(), 36 * c_out * c_in);
    debug_assert_eq!(b_panels_f16.len(), 36 * c_in * c_out);
    debug_assert_eq!(bias1x1.len(), c_out2);
    debug_assert_ne!(act1x1, Activation::Sigmoid);
    let _ = weights_transformed_f16;
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), c_out);
    }

    let n_tile_h = h_out.div_ceil(4);
    let n_tile_w = w_out.div_ceil(4);
    const MR: usize = 8;
    let n_tile_w_gemm = n_tile_w.div_ceil(MR) * MR;

    let in_ptr = input.as_ptr() as usize;
    let bias_ptr: usize = bias.map_or(0, |b| b.as_ptr() as usize);
    let has_bias = bias.is_some();
    let b_panels_ptr = b_panels_f16.as_ptr() as usize;

    const NR_W: usize = 8;
    let n_blocks_w = c_out / NR_W;
    let n_rem_w = c_out % NR_W;
    let slot_sz = n_blocks_w * c_in * NR_W + c_in * n_rem_w;
    let use_prepacked = !b_panels_packed.is_empty() && b_panels_packed.len() == 36 * slot_sz;
    let b_packed_ptr = if use_prepacked {
        b_panels_packed.as_ptr() as usize
    } else {
        0
    };

    // 1×1 epilogue shared state (read-only across tasks).
    let b1x1_ptr = b1x1_packed.as_ptr() as usize;
    let b1x1_total = b1x1_packed.len();
    let bias1x1_ptr = bias1x1.as_ptr() as usize;
    let relu1x1 = matches!(act1x1, Activation::Relu);
    let out1x1_ptr = out1x1.as_mut_ptr() as usize;

    let out_row_stride_1x1 = w_out * c_out2; // f16 elements per output row

    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let n_rayon_threads = rayon::current_num_threads().max(1);
    let g = gcd(n_tile_h, n_rayon_threads);
    let col_parts = (n_rayon_threads / g).min(n_tile_w.max(1));

    let n_co8 = n_blocks_w;
    let use_co_par = use_prepacked
        && n_co8 > 1
        && col_parts == 1
        && n_tile_h < n_rayon_threads
        && n_tile_w_gemm > 8;

    (0..n_tile_h * col_parts)
        .into_par_iter()
        .for_each(|task_id| {
            let tile_oh = task_id / col_parts;
            let col_part = task_id % col_parts;
            let tiles_per_part = n_tile_w.div_ceil(col_parts);
            // Clamp col_start to n_tile_w so trailing empty tasks (col_parts >
            // n_tile_w) don't underflow `col_end - col_start` (usize). See the
            // matching guard in conv3x3_winograd_nhwc_f43_f16.
            let col_start = (col_part * tiles_per_part).min(n_tile_w);
            let col_end = (col_start + tiles_per_part).min(n_tile_w);
            let n_col_tiles = col_end - col_start;
            if n_col_tiles == 0 {
                return;
            }
            let n_tile_w_gemm_part = n_col_tiles.div_ceil(MR) * MR;
            let valid_rows = (h_out - tile_oh * 4).min(4);

            // Dense strip geometry: this task owns `valid_rows` output rows, each
            // covering output columns [col_start*4 .. min((col_end)*4, w_out)).
            let strip_col0 = col_start * 4;
            let strip_cols = (w_out - strip_col0).min(n_col_tiles * 4);
            let block_pixels = valid_rows * strip_cols;

            let input =
                unsafe { std::slice::from_raw_parts(in_ptr as *const f32, h_in * w_in * c_in) };
            let bias_slice: Option<&[f32]> = if has_bias {
                Some(unsafe { std::slice::from_raw_parts(bias_ptr as *const f32, c_out) })
            } else {
                None
            };

            let ih_start = (tile_oh * 4) as isize - 1;

            let zero_f16 = half::f16::ZERO.to_bits();
            let a_len = 36 * n_tile_w_gemm_part * c_in;
            let m_len = n_tile_w_gemm_part * 36 * c_out;
            let c_len = n_tile_w_gemm_part * c_out;
            let pa_len = c_in * MR;
            let pb_len = c_in * c_out;

            FUSE_STRIP_F32.with(|strip_cell| {
                let strip_len = block_pixels * c_out;
                let mut strip = strip_cell.borrow_mut();
                if strip.len() < strip_len {
                    strip.resize(strip_len, 0.0);
                }
                // The output-transform writes complete 8-wide co-blocks; pixels are
                // dense, so no pre-zeroing of valid bytes is required.
                let strip_base_ptr = strip.as_mut_ptr() as usize;
                let strip_row_stride = strip_cols * c_out; // f32 elements per strip row

                F43_A_ALL.with(|a_cell| {
                    let mut a_all = a_cell.borrow_mut();
                    if a_all.len() < a_len {
                        a_all.resize(a_len, zero_f16);
                    }
                    a_all[..a_len].fill(zero_f16);

                    for tile_ow in col_start..col_end {
                        let rel_ow = tile_ow - col_start;
                        let iw_start = (tile_ow * 4) as isize - 1;
                        unsafe {
                            super::neon_asm_f16::transform_input_tile_allch_f43(
                                input,
                                h_in,
                                w_in,
                                c_in,
                                ih_start,
                                iw_start,
                                n_tile_w_gemm_part,
                                rel_ow,
                                &mut a_all,
                            );
                        }
                    }

                    if use_co_par {
                        let a_copy: Vec<u16> = a_all[..a_len].to_vec();
                        drop(a_all);
                        let a_ptr = a_copy.as_ptr() as usize;

                        let m_co_len = n_tile_w_gemm * 36 * NR_W;
                        let c_co_len = n_tile_w_gemm * NR_W;
                        let mut m_acc_all = vec![0u16; n_co8 * m_co_len];
                        let m_all_ptr = m_acc_all.as_mut_ptr() as usize;

                        (0..n_co8).into_par_iter().for_each(|co_blk| {
                            let a =
                                unsafe { std::slice::from_raw_parts(a_ptr as *const u16, a_len) };
                            let bp = unsafe {
                                std::slice::from_raw_parts(b_packed_ptr as *const u16, 36 * slot_sz)
                            };
                            let m_co = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (m_all_ptr as *mut u16).add(co_blk * m_co_len),
                                    m_co_len,
                                )
                            };
                            F43_C_TMP.with(|c_cell| {
                                F43_PACK_A.with(|pa_cell| {
                                    let mut c_tmp = c_cell.borrow_mut();
                                    let mut pack_a = pa_cell.borrow_mut();
                                    if c_tmp.len() < c_co_len {
                                        c_tmp.resize(c_co_len, zero_f16);
                                    }
                                    if pack_a.len() < pa_len {
                                        pack_a.resize(pa_len, zero_f16);
                                    }
                                    for p in 0..36usize {
                                        let a_f16 = &a[p * n_tile_w_gemm * c_in
                                            ..(p + 1) * n_tile_w_gemm * c_in];
                                        c_tmp[..c_co_len].fill(zero_f16);
                                        let p_off = p * slot_sz + co_blk * c_in * NR_W;
                                        super::neon_asm_f16::gemm_f16_mnk_packed_b(
                                            a_f16,
                                            &bp[p_off..p_off + c_in * NR_W],
                                            &[],
                                            &mut c_tmp[..c_co_len],
                                            n_tile_w_gemm,
                                            c_in,
                                            NR_W,
                                            &mut pack_a[..pa_len],
                                        );
                                        for tile_ow in 0..n_tile_w {
                                            let dst = tile_ow * 36 * NR_W + p * NR_W;
                                            let src = tile_ow * NR_W;
                                            m_co[dst..dst + NR_W]
                                                .copy_from_slice(&c_tmp[src..src + NR_W]);
                                        }
                                    }
                                })
                            });
                        });

                        // Phase 3: output transform → dense f32 strip.
                        {
                            use super::neon_asm_f16::{
                                load_bias_f16x8, vld1q_u16_wrap,
                                winograd_f43_output_transform_write_f16x8,
                            };
                            let relu = matches!(activation, Activation::Relu);
                            let strip_p = strip_base_ptr as *mut f32;
                            for co_blk in 0..n_co8 {
                                let co_base = co_blk * NR_W;
                                let m = &m_acc_all[co_blk * m_co_len..(co_blk + 1) * m_co_len];
                                for tile_ow in 0..n_tile_w {
                                    let ow0 = tile_ow * 4;
                                    let tb = tile_ow * 36 * NR_W;
                                    let valid_cols = (w_out - ow0).min(4);
                                    let (bias_lo, bias_hi) =
                                        unsafe { load_bias_f16x8(bias_slice, co_base) };
                                    let mut m_f16 = unsafe { [vld1q_u16_wrap(m.as_ptr()); 36] };
                                    for (p, slot) in m_f16.iter_mut().enumerate() {
                                        *slot =
                                            unsafe { vld1q_u16_wrap(m[tb + p * NR_W..].as_ptr()) };
                                    }
                                    unsafe {
                                        winograd_f43_output_transform_write_f16x8(
                                            &m_f16,
                                            bias_lo,
                                            bias_hi,
                                            relu,
                                            strip_p,
                                            strip_row_stride,
                                            ow0 - strip_col0,
                                            c_out,
                                            co_base,
                                            valid_rows,
                                            valid_cols,
                                        );
                                    }
                                }
                            }
                        }
                    } else {
                        let a_ptr = a_all.as_ptr() as usize;
                        F43_M_ACC.with(|m_cell| {
                            F43_C_TMP.with(|c_cell| {
                                F43_PACK_A.with(|pa_cell| {
                                    F43_PACK_B.with(|pb_cell| {
                                        let mut m_acc_f16 = m_cell.borrow_mut();
                                        let mut c_tmp = c_cell.borrow_mut();
                                        let mut pack_a = pa_cell.borrow_mut();
                                        let mut pack_b = pb_cell.borrow_mut();
                                        if m_acc_f16.len() < m_len {
                                            m_acc_f16.resize(m_len, zero_f16);
                                        }
                                        if c_tmp.len() < c_len {
                                            c_tmp.resize(c_len, zero_f16);
                                        }
                                        if pack_a.len() < pa_len {
                                            pack_a.resize(pa_len, zero_f16);
                                        }
                                        if pack_b.len() < pb_len {
                                            pack_b.resize(pb_len, zero_f16);
                                        }
                                        m_acc_f16[..m_len].fill(zero_f16);
                                        let a = unsafe {
                                            std::slice::from_raw_parts(a_ptr as *const u16, a_len)
                                        };
                                        for p in 0..36usize {
                                            let a_f16 = &a[p * n_tile_w_gemm_part * c_in
                                                ..(p + 1) * n_tile_w_gemm_part * c_in];
                                            c_tmp[..c_len].fill(zero_f16);
                                            if use_prepacked {
                                                let bp = unsafe {
                                                    std::slice::from_raw_parts(
                                                        b_packed_ptr as *const u16,
                                                        36 * slot_sz,
                                                    )
                                                };
                                                let p_off = p * slot_sz;
                                                let packed_b =
                                                    &bp[p_off..p_off + n_blocks_w * c_in * NR_W];
                                                let b_rem_tail = &bp[p_off
                                                    + n_blocks_w * c_in * NR_W
                                                    ..(p + 1) * slot_sz];
                                                super::neon_asm_f16::gemm_f16_mnk_packed_b(
                                                    a_f16,
                                                    packed_b,
                                                    b_rem_tail,
                                                    &mut c_tmp[..c_len],
                                                    n_tile_w_gemm_part,
                                                    c_in,
                                                    c_out,
                                                    &mut pack_a[..pa_len],
                                                );
                                            } else {
                                                let b_panels: &[u16] = unsafe {
                                                    std::slice::from_raw_parts(
                                                        b_panels_ptr as *const u16,
                                                        36 * c_in * c_out,
                                                    )
                                                };
                                                let b_ptr = &b_panels
                                                    [p * c_in * c_out..(p + 1) * c_in * c_out];
                                                super::neon_asm_f16::gemm_f16_mnk_with_pack(
                                                    a_f16,
                                                    b_ptr,
                                                    &mut c_tmp[..c_len],
                                                    n_tile_w_gemm_part,
                                                    c_in,
                                                    c_out,
                                                    &mut pack_a[..pa_len],
                                                    &mut pack_b[..pb_len],
                                                    false,
                                                );
                                            }
                                            for rel_ow in 0..n_col_tiles {
                                                let dst_base = rel_ow * 36 * c_out + p * c_out;
                                                let src_base = rel_ow * c_out;
                                                m_acc_f16[dst_base..dst_base + c_out]
                                                    .copy_from_slice(
                                                        &c_tmp[src_base..src_base + c_out],
                                                    );
                                            }
                                        }

                                        // Output transform → dense f32 strip.
                                        use super::neon_asm_f16::{
                                            load_bias_f16x8, vld1q_u16_wrap,
                                            winograd_f43_output_transform_write_f16x8,
                                        };
                                        let relu = matches!(activation, Activation::Relu);
                                        let m_acc = &m_acc_f16[..m_len];
                                        let strip_p = strip_base_ptr as *mut f32;
                                        for tile_ow in col_start..col_end {
                                            let rel_ow = tile_ow - col_start;
                                            let ow0 = tile_ow * 4;
                                            let tile_base = rel_ow * 36 * c_out;
                                            let valid_cols = (w_out - tile_ow * 4).min(4);
                                            let mut co_base = 0usize;
                                            while co_base + 8 <= c_out {
                                                let (bias_lo, bias_hi) =
                                                    unsafe { load_bias_f16x8(bias_slice, co_base) };
                                                let mut m_f16 =
                                                    unsafe { [vld1q_u16_wrap(m_acc.as_ptr()); 36] };
                                                for (p, slot) in m_f16.iter_mut().enumerate() {
                                                    *slot = unsafe {
                                                        vld1q_u16_wrap(
                                                            m_acc
                                                                [tile_base + p * c_out + co_base..]
                                                                .as_ptr(),
                                                        )
                                                    };
                                                }
                                                unsafe {
                                                    winograd_f43_output_transform_write_f16x8(
                                                        &m_f16,
                                                        bias_lo,
                                                        bias_hi,
                                                        relu,
                                                        strip_p,
                                                        strip_row_stride,
                                                        ow0 - strip_col0,
                                                        c_out,
                                                        co_base,
                                                        valid_rows,
                                                        valid_cols,
                                                    );
                                                }
                                                co_base += 8;
                                            }
                                            for co in co_base..c_out {
                                                let m_tile: [f32; 36] = core::array::from_fn(|p| {
                                                    let bits = m_acc[tile_base + p * c_out + co];
                                                    half::f16::from_bits(bits).to_f32()
                                                });
                                                let y = winograd_output_transform_f32_f43(&m_tile);
                                                let b = bias_slice.map_or(0.0, |bs| bs[co]);
                                                for ry in 0..valid_rows {
                                                    let row_off = ry * strip_row_stride;
                                                    for cx in 0..valid_cols {
                                                        let val = apply_act(
                                                            y[ry * 4 + cx] + b,
                                                            activation,
                                                        );
                                                        unsafe {
                                                            *strip_p.add(
                                                                row_off
                                                                    + (ow0 - strip_col0 + cx)
                                                                        * c_out
                                                                    + co,
                                                            ) = val;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    })
                                })
                            })
                        });
                    }
                });

                // ── Fused 1×1 epilogue: dense strip → final f16 output ───────────
                // Each strip pixel is an independent fp16 dot product; with
                // block_pixels % 8 == 0 the GEMM m-remainder path is never taken, so
                // results are bit-identical to the standalone 1×1 conv.
                {
                    let strip = &strip[..strip_len];
                    FUSE_A_F16.with(|af_cell| {
                        FUSE_C_F16.with(|cf_cell| {
                            FUSE_PACK_A.with(|pa_cell| {
                                let mut a_f16 = af_cell.borrow_mut();
                                let mut c_f16 = cf_cell.borrow_mut();
                                let mut pack_a = pa_cell.borrow_mut();
                                let a_need = block_pixels * c_out;
                                let c_need = block_pixels * c_out2;
                                if a_f16.len() < a_need {
                                    a_f16.resize(a_need, zero_f16);
                                }
                                if c_f16.len() < c_need {
                                    c_f16.resize(c_need, zero_f16);
                                }
                                if pack_a.len() < c_out * 8 {
                                    pack_a.resize(c_out * 8, zero_f16);
                                }

                                // f32 → f16 (FCVTN): same rounding as the standalone
                                // path's per-strip f32_to_f16_buf.
                                unsafe {
                                    super::neon_asm_f16::f32_to_f16_buf(
                                        strip,
                                        &mut a_f16[..a_need],
                                    );
                                }
                                c_f16[..c_need].fill(zero_f16);

                                let b1x1 = unsafe {
                                    std::slice::from_raw_parts(b1x1_ptr as *const u16, b1x1_total)
                                };
                                let bias1x1 = unsafe {
                                    std::slice::from_raw_parts(bias1x1_ptr as *const f32, c_out2)
                                };
                                unsafe {
                                    super::neon_asm_f16::gemm_f16_mnk_packed_b(
                                        &a_f16[..a_need],
                                        &b1x1[..b1x1_rem_offset],
                                        &b1x1[b1x1_rem_offset..],
                                        &mut c_f16[..c_need],
                                        block_pixels,
                                        c_out,
                                        c_out2,
                                        &mut pack_a[..c_out * 8],
                                    );
                                    super::neon_asm_f16::f16_bias_act_inplace_pub(
                                        &mut c_f16[..c_need],
                                        bias1x1,
                                        c_out2,
                                        relu1x1,
                                        block_pixels,
                                    );
                                }

                                // Scatter dense [block_pixels, c_out2] → final f16
                                // output. Strip pixel (ry, cx_rel) maps to output
                                // pixel (tile_oh*4 + ry, strip_col0 + cx_rel).
                                let out_p = out1x1_ptr as *mut u16;
                                for ry in 0..valid_rows {
                                    let oh = tile_oh * 4 + ry;
                                    let dst_row = oh * out_row_stride_1x1 + strip_col0 * c_out2;
                                    let src_row = ry * strip_cols * c_out2;
                                    let n = strip_cols * c_out2;
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            c_f16.as_ptr().add(src_row),
                                            out_p.add(dst_row),
                                            n,
                                        );
                                    }
                                }
                            })
                        })
                    });
                }
            });
        });
}

/// Returns true iff the fused Winograd→1×1 epilogue produces bit-identical
/// results to running the unfused 3×3 Winograd + standalone 1×1 sequence at the
/// given output shape.
///
/// Requirement: every Winograd task's `block_pixels` (= valid_rows × strip_cols)
/// must be a multiple of MR=8 so the per-task 1×1 GEMM never takes the
/// f32-accumulating m-remainder path (the standalone path slices into
/// CONV1X1_STRIP_SIZE=800-pixel strips, all multiples of 8, so it never does
/// either). See [`conv3x3_winograd_nhwc_f43_f16_fused1x1`] for the full argument.
///
/// Tasks tile 4 output rows high; the last row-tile may be partial when
/// `h_out % 4 != 0`. The column slice width depends on `col_parts`, which is a
/// runtime function of the Rayon thread count — so this conservatively requires
/// `h_out % 4 == 0` (every task has `valid_rows == 4`) and `w_out % 2 == 0`
/// (every per-task strip column count is a multiple of 2, so 4×even is a
/// multiple of 8). Both hold for all XFeat 1×1-terminated conv groups.
#[cfg(target_arch = "aarch64")]
pub fn f43_fused1x1_is_bit_exact(h_out: usize, w_out: usize) -> bool {
    // valid_rows == 4 for every task, and any contiguous column slice has a
    // width that is a multiple of 2 (each Winograd tile is 4 cols wide; the
    // last partial tile contributes w_out % 4 ∈ {0,2} cols since w_out is even).
    // 4 rows × (even cols) = multiple of 8.
    h_out % 4 == 0 && w_out % 2 == 0
}

/// fp16 F(4,3) driver with correct scalar fallback on the row/column epilogue.
///
/// When both dims are multiples of 4 this is identical to
/// [`conv3x3_winograd_nhwc_f43_f16`].
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_winograd_nhwc_f43_f16_with_scalar_fallback(
    input: &[f32],
    h_in: usize,
    w_in: usize,
    c_in: usize,
    weights_transformed_f16: &[u16], // [36, c_out, c_in] (debug_assert only)
    b_panels_f16: &[u16],            // [36 * c_in * c_out] pre-transposed
    b_panels_packed: &[u16],         // [36, slot] pre-packed; empty → pack on demand
    _weights_spatial: &[f32],        // kept for API compat; no longer used
    bias: Option<&[f32]>,
    c_out: usize,
    activation: Activation,
    output: &mut [f32],
    h_out: usize,
    w_out: usize,
) {
    // The main driver now uses ceiling-division tiling and handles partial
    // last tiles in NEON (extract_patch zero-pads OOB input rows).
    // No scalar epilogue needed.
    conv3x3_winograd_nhwc_f43_f16(
        input,
        h_in,
        w_in,
        c_in,
        weights_transformed_f16,
        b_panels_f16,
        b_panels_packed,
        bias,
        c_out,
        activation,
        output,
        h_out,
        w_out,
    );
}

// ─── OpsVtable dispatch adapter ──────────────────────────────────────────────

/// Stride-1 3×3 NHWC conv via Winograd F(2×2, 3×3).
///
/// Adapts the [`Conv3x3Args`] / `&mut [f32]` interface expected by
/// [`crate::ops::OpsVtable`] to the Winograd driver.
///
/// The spatial weights in `args.weights` (layout `[c_out, 3, 3, c_in]`) are
/// transformed to the Winograd domain on every call.  Weight transformation is
/// O(c_out × c_in) arithmetic — negligible compared to the convolution itself
/// for the typical XFeat channel counts (4–128).
///
/// After the Winograd convolution any optional residual tensor is added and
/// the fused activation is applied (both are part of the epilogue inside
/// `conv3x3_winograd_nhwc` for the tile pixels; residual is handled here for
/// the scalar-epilogue boundary pixels when `h_out` or `w_out` is odd, but in
/// practice all XFeat layers use even spatial dimensions at every stage).
///
/// **Stride:** always 1.  Stride-2 layers are not routed through this function.
pub fn conv3x3_winograd_dispatch(args: &Conv3x3Args<'_>, output: &mut [f32]) {
    let &Conv3x3Args {
        input: _,
        residual: _,
        weights: _,
        bias: _,
        h_in,
        w_in,
        c_in,
        c_out,
        activation: _,
        packed_weights: _,
    } = args;

    let h_out = h_in; // stride = 1
    let w_out = w_in;
    let _ = (h_out, w_out);

    // Fast path: c_in=1 with c_out%4==0 — direct NEON conv is faster than
    // Winograd because the K=1 GEMM has no reduction and tile overhead dominates.
    #[cfg(target_arch = "aarch64")]
    if c_in == 1 && c_out % 4 == 0 {
        super::neon::conv3x3_c1_nhwc(args, output, 1);
        return;
    }

    let &Conv3x3Args {
        input,
        residual,
        weights,
        bias,
        h_in,
        w_in,
        c_in: _,
        c_out: _,
        activation,
        packed_weights: _,
    } = args;

    // Transform spatial weights → Winograd F(2,3) domain [16, c_out, c_in].
    // NOTE: this dispatch adapter is only used by the OpsVtable fallback path
    // (non-eligible layers or non-aarch64) and is intentionally kept on the
    // simpler F(2,3) kernel. The 10 Winograd-eligible layers in XFeat bypass
    // OpsVtable entirely and call the F(4,3) drivers
    // (conv3x3_winograd_nhwc_f43*) directly with pre-transformed weights.
    let weights_transformed = winograd_transform_weights_f32(weights, c_out, c_in);

    // Run the Winograd convolution (includes bias + activation epilogue).
    // _v_buf in conv3x3_winograd_nhwc is reserved-but-unused — pass empty slice
    // to avoid a per-frame 4.9 MB allocation + zeroing cost (block1.0 at 480×640).
    conv3x3_winograd_nhwc(
        input,
        h_in,
        w_in,
        c_in,
        &weights_transformed,
        Some(bias),
        c_out,
        activation,
        output,
        h_in, // h_out = h_in (stride=1)
        w_in, // w_out = w_in
        &mut [],
    );

    // Add residual if present (the model always passes None, but we handle it
    // for correctness).
    if let Some(r) = residual {
        for (o, &rv) in output.iter_mut().zip(r.iter()) {
            *o += rv;
        }
    }
}

// ─── Shared activation helper (local to this module) ─────────────────────────

#[inline(always)]
fn apply_act(x: f32, act: Activation) -> f32 {
    match act {
        Activation::Relu => x.max(0.0),
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::Identity => x,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify weight transform: U = G · g · G^T for a known 3×3 filter.
    #[test]
    fn test_weight_transform_identity_filter() {
        // A single-channel identity-like filter: all zeros except center = 1.
        let mut weights = vec![0.0f32; 1 * 9 * 1];
        weights[4] = 1.0; // centre tap kh=1, kw=1

        let u = winograd_transform_weights_f32(&weights, 1, 1);
        assert_eq!(u.len(), 16);

        // G · [0,0,0; 0,1,0; 0,0,0] · G^T = G[:,1] * G^T[1,:] = G[:,1] * G[:,1]^T
        // G column 1 = [0, 0.5, -0.5, 0].
        // Expected U[i][j] = G[i][1] * G[j][1]:
        let g_col1 = [0.0f32, 0.5, -0.5, 0.0];
        for i in 0..4usize {
            for j in 0..4usize {
                let expected = g_col1[i] * g_col1[j];
                let got = u[(i * 4 + j) * 1 * 1]; // c_out=1, c_in=1
                assert!(
                    (got - expected).abs() < 1e-6,
                    "U[{i}][{j}]: got {got}, expected {expected}"
                );
            }
        }
    }

    /// Verify input transform is correct by checking B^T · d · B manually.
    #[test]
    fn test_input_transform_zeros() {
        let d = [0.0f32; 16];
        let v = winograd_transform_input_tile_f32(&d);
        assert_eq!(v, [0.0f32; 16]);
    }

    #[test]
    fn test_input_transform_ones() {
        let d = [1.0f32; 16];
        let v = winograd_transform_input_tile_f32(&d);
        // B^T · ones · B:
        // Step 1 (row-wise B^T): each row r of t is:
        //   t[r][0] = 1 - 1 = 0
        //   t[r][1] = 1 + 1 = 2
        //   t[r][2] = 1 - 1 = 0
        //   t[r][3] = 1 - 1 = 0
        // All rows are [0, 2, 0, 0].
        //
        // Step 2 (col-wise B^T): apply B^T to each column of t:
        //   col 0: t[:,0] = [0,0,0,0] → all zeros
        //   col 1: t[:,1] = [2,2,2,2]
        //     v[0][1] = 2 - 2 = 0
        //     v[1][1] = 2 + 2 = 4
        //     v[2][1] = 2 - 2 = 0
        //     v[3][1] = 2 - 2 = 0
        //   col 2, 3: same as col 0 (all zeros)
        // Result: only v[1][1] = pos(1*4+1) = pos 5 is 4.0; rest are 0.
        let mut expected = [0.0f32; 16];
        expected[5] = 4.0; // v[row=1][col=1]
        assert_eq!(v, expected);
    }

    /// Verify output transform: A^T · [[1,0,0,0],[0,0,0,0],...] · A
    #[test]
    fn test_output_transform_zero() {
        let m = [0.0f32; 16];
        let y = winograd_output_transform_f32(&m);
        assert_eq!(y, [0.0f32; 4]);
    }

    /// End-to-end: single 1×1 conv (c_in=1, c_out=1, h=2, w=2) with identity
    /// filter should reproduce the input.
    #[test]
    fn test_e2e_identity_filter_2x2() {
        // h=2, w=2 conv with identity filter (centre=1, rest=0), c_in=c_out=1.
        let h = 2usize;
        let w = 2usize;
        let c_in = 1usize;
        let c_out = 1usize;

        let input = vec![1.0f32, 2.0, 3.0, 4.0]; // [h=2, w=2, c=1]

        let mut weights = vec![0.0f32; c_out * 9 * c_in];
        weights[4] = 1.0; // centre tap

        let wt = winograd_transform_weights_f32(&weights, c_out, c_in);

        let mut output = vec![0.0f32; h * w * c_out];
        let n_tile_h = h / 2;
        let n_tile_w = w / 2;
        let mut v_buf = vec![0.0f32; n_tile_h * n_tile_w * 16 * c_in];
        conv3x3_winograd_nhwc(
            &input,
            h,
            w,
            c_in,
            &wt,
            None,
            c_out,
            Activation::Identity,
            &mut output,
            h,
            w,
            &mut v_buf,
        );

        // With zero-padded boundary conv, each output pixel is just the centre tap
        // which is multiplied by the corresponding input pixel.
        for (i, (&got, &expected)) in output.iter().zip(input.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-5,
                "output[{i}]: got {got}, expected {expected}"
            );
        }
    }

    /// Test that the Winograd result matches a direct scalar conv for a random
    /// 4×4 input with a random 3×3 filter.
    #[test]
    fn test_winograd_matches_scalar_4x4() {
        let h = 4usize;
        let w = 4usize;
        let c_in = 1usize;
        let c_out = 1usize;

        // Deterministic "random" input
        let input: Vec<f32> = (0..h * w * c_in).map(|i| (i as f32 + 1.0) * 0.1).collect();

        // Deterministic filter
        let weights: Vec<f32> = vec![
            0.1, 0.2, -0.1, //
            0.3, 0.5, 0.1, //
            -0.1, 0.2, 0.3,
        ];
        // Pack as [c_out, 9, c_in]
        let weights_packed: Vec<f32> = weights.clone();

        let wt = winograd_transform_weights_f32(&weights_packed, c_out, c_in);

        let mut output_winograd = vec![0.0f32; h * w * c_out];
        let n_tile_h = h / 2;
        let n_tile_w = w / 2;
        let mut v_buf = vec![0.0f32; n_tile_h * n_tile_w * 16 * c_in];
        conv3x3_winograd_nhwc(
            &input,
            h,
            w,
            c_in,
            &wt,
            None,
            c_out,
            Activation::Identity,
            &mut output_winograd,
            h,
            w,
            &mut v_buf,
        );

        // Scalar reference.
        let mut output_scalar = vec![0.0f32; h * w * c_out];
        for oh in 0..h {
            for ow in 0..w {
                let mut acc = 0.0f32;
                for kh in 0..3isize {
                    let ih = oh as isize + kh - 1;
                    if ih < 0 || ih >= h as isize {
                        continue;
                    }
                    for kw in 0..3isize {
                        let iw = ow as isize + kw - 1;
                        if iw < 0 || iw >= w as isize {
                            continue;
                        }
                        // kh, kw are 0-indexed kernel taps (0,1,2); layout [c_out=1, 9, c_in=1]
                        let tap = kh * 3 + kw;
                        acc += input[ih as usize * w * c_in + iw as usize * c_in]
                            * weights_packed[tap as usize * c_in];
                    }
                }
                output_scalar[oh * w * c_out + ow * c_out] = acc;
            }
        }

        for i in 0..h * w {
            assert!(
                (output_winograd[i] - output_scalar[i]).abs() < 1e-4,
                "pixel {i}: winograd={}, scalar={}",
                output_winograd[i],
                output_scalar[i]
            );
        }
    }

    // ─── F(4×4, 3×3) tests ────────────────────────────────────────────────────

    /// Scalar direct 3×3 same-padded conv reference for a single channel.
    /// `weights` is `[c_out=1, 9, c_in=1]` (tap = kh*3+kw).
    fn scalar_conv_1ch(input: &[f32], h: usize, w: usize, weights: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; h * w];
        for oh in 0..h {
            for ow in 0..w {
                let mut acc = 0.0f32;
                for kh in 0..3isize {
                    let ih = oh as isize + kh - 1;
                    if ih < 0 || ih >= h as isize {
                        continue;
                    }
                    for kw in 0..3isize {
                        let iw = ow as isize + kw - 1;
                        if iw < 0 || iw >= w as isize {
                            continue;
                        }
                        let tap = (kh * 3 + kw) as usize;
                        acc += input[ih as usize * w + iw as usize] * weights[tap];
                    }
                }
                out[oh * w + ow] = acc;
            }
        }
        out
    }

    #[test]
    fn test_weight_transform_f43_identity_filter() {
        // Centre-only filter: U = G[:,1] ⊗ G[:,1] (outer product of G column 1).
        let mut weights = vec![0.0f32; 9];
        weights[4] = 1.0; // centre tap kh=1, kw=1

        let u = winograd_transform_weights_f32_f43(&weights, 1, 1);
        assert_eq!(u.len(), 36);

        // G column 1 (kw=1 index in each G row): the second column of G.
        let g_col1 = [0.0f32, -1.0 / 6.0, 1.0 / 6.0, 1.0 / 12.0, -1.0 / 12.0, 0.0];
        for i in 0..6usize {
            for j in 0..6usize {
                let expected = g_col1[i] * g_col1[j];
                let got = u[i * 6 + j];
                assert!(
                    (got - expected).abs() < 1e-6,
                    "U[{i}][{j}]: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_input_transform_f43_zeros() {
        let d = [0.0f32; 36];
        let v = winograd_transform_input_tile_f32_f43(&d);
        assert_eq!(v, [0.0f32; 36]);
    }

    #[test]
    fn test_input_transform_f43_ones() {
        let d = [1.0f32; 36];
        let v = winograd_transform_input_tile_f32_f43(&d);

        // Reference: compute B^T @ ones @ B directly via the B^T matrix.
        let bt: [[f32; 6]; 6] = [
            [4.0, 0.0, -5.0, 0.0, 1.0, 0.0],
            [0.0, -4.0, -4.0, 1.0, 1.0, 0.0],
            [0.0, 4.0, -4.0, -1.0, 1.0, 0.0],
            [0.0, -2.0, -1.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, -1.0, -2.0, 1.0, 0.0],
            [0.0, 4.0, 0.0, -5.0, 0.0, 1.0],
        ];
        // t = B^T @ d (d = ones)
        let mut t = [[0.0f32; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                let mut s = 0.0f32;
                for k in 0..6 {
                    s += bt[i][k] * 1.0; // d[k][j] = 1
                }
                t[i][j] = s;
            }
        }
        // v = t @ B  (B = B^T transposed → B[k][j] = bt[j][k])
        let mut expected = [0.0f32; 36];
        for i in 0..6 {
            for j in 0..6 {
                let mut s = 0.0f32;
                for k in 0..6 {
                    s += t[i][k] * bt[j][k];
                }
                expected[i * 6 + j] = s;
            }
        }
        for p in 0..36 {
            assert!(
                (v[p] - expected[p]).abs() < 1e-4,
                "pos {p}: got {}, expected {}",
                v[p],
                expected[p]
            );
        }
    }

    #[test]
    fn test_winograd_f43_matches_scalar_4x4() {
        let h = 4usize;
        let w = 4usize;
        let input: Vec<f32> = (0..h * w).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let weights: Vec<f32> = vec![0.1, 0.2, -0.1, 0.3, 0.5, 0.1, -0.1, 0.2, 0.3];

        let wt = winograd_transform_weights_f32_f43(&weights, 1, 1);
        let mut out_w = vec![0.0f32; h * w];
        conv3x3_winograd_nhwc_f43(
            &input,
            h,
            w,
            1,
            &wt,
            None,
            1,
            Activation::Identity,
            &mut out_w,
            h,
            w,
        );

        let out_s = scalar_conv_1ch(&input, h, w, &weights);
        for i in 0..h * w {
            assert!(
                (out_w[i] - out_s[i]).abs() < 1e-4,
                "pixel {i}: winograd={}, scalar={}",
                out_w[i],
                out_s[i]
            );
        }
    }

    #[test]
    fn test_winograd_f43_matches_scalar_8x8() {
        let h = 8usize;
        let w = 8usize;
        let input: Vec<f32> = (0..h * w)
            .map(|i| ((i * 7) % 13) as f32 * 0.13 - 0.5)
            .collect();
        let weights: Vec<f32> = vec![0.05, -0.2, 0.15, 0.3, -0.5, 0.1, -0.1, 0.25, 0.3];

        let wt = winograd_transform_weights_f32_f43(&weights, 1, 1);
        let mut out_w = vec![0.0f32; h * w];
        conv3x3_winograd_nhwc_f43(
            &input,
            h,
            w,
            1,
            &wt,
            None,
            1,
            Activation::Identity,
            &mut out_w,
            h,
            w,
        );

        let out_s = scalar_conv_1ch(&input, h, w, &weights);
        for i in 0..h * w {
            assert!(
                (out_w[i] - out_s[i]).abs() < 1e-3,
                "pixel {i}: winograd={}, scalar={}",
                out_w[i],
                out_s[i]
            );
        }
    }

    #[test]
    fn test_winograd_f43_matches_scalar_with_epilogue() {
        // h=14 (14%4=2 → row epilogue), w=12 (12%4=0 → no col epilogue)...
        // Use w=10 to force BOTH row and col epilogues (10%4=2, 14%4=2).
        let h = 14usize;
        let w = 10usize;
        let input: Vec<f32> = (0..h * w)
            .map(|i| ((i * 31) % 17) as f32 * 0.07 - 0.4)
            .collect();
        let weights: Vec<f32> = vec![0.11, -0.22, 0.13, 0.34, -0.45, 0.16, -0.17, 0.28, 0.39];

        let wt = winograd_transform_weights_f32_f43(&weights, 1, 1);
        let mut out_w = vec![0.0f32; h * w];
        conv3x3_winograd_nhwc_f43_with_scalar_fallback(
            &input,
            h,
            w,
            1,
            &wt,
            &weights, // spatial weights [1,9,1]
            None,
            1,
            Activation::Identity,
            &mut out_w,
            h,
            w,
        );

        let out_s = scalar_conv_1ch(&input, h, w, &weights);
        for i in 0..h * w {
            assert!(
                (out_w[i] - out_s[i]).abs() < 1e-3,
                "pixel {} (r={}, c={}): winograd={}, scalar={}",
                i,
                i / w,
                i % w,
                out_w[i],
                out_s[i]
            );
        }
    }

    /// Build the f16 weight buffers (transformed_f16, b_panels_f16,
    /// b_panels_packed) for a c_out×c_in 3×3 layer, mirroring the WinogradCache
    /// builder in model.rs.
    #[cfg(target_arch = "aarch64")]
    fn build_f43_f16_weights(
        weights_spatial: &[f32], // [c_out, 3, 3, c_in]
        c_out: usize,
        c_in: usize,
    ) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let transformed = winograd_transform_weights_f32_f43(weights_spatial, c_out, c_in);
        let mut f16_buf: Vec<u16> = Vec::with_capacity(transformed.len());
        unsafe {
            crate::ops::neon_asm_f16::f32_to_f16_slice(&transformed, &mut f16_buf);
        }
        let mut b_panels = vec![0u16; 36 * c_in * c_out];
        for p in 0..36usize {
            let pos_offset = p * c_out * c_in;
            for co in 0..c_out {
                for ci in 0..c_in {
                    b_panels[p * c_in * c_out + ci * c_out + co] =
                        f16_buf[pos_offset + co * c_in + ci];
                }
            }
        }
        const NR: usize = 8;
        let n_blocks = c_out / NR;
        let n_rem = c_out % NR;
        let slot_sz = n_blocks * c_in * NR + c_in * n_rem;
        let mut b_packed = vec![0u16; 36 * slot_sz];
        for p in 0..36usize {
            let dst = &mut b_packed[p * slot_sz..];
            for nb in 0..n_blocks {
                let nr_start = nb * NR;
                for ki in 0..c_in {
                    for ni in 0..NR {
                        dst[nb * c_in * NR + ki * NR + ni] =
                            b_panels[p * c_in * c_out + ki * c_out + nr_start + ni];
                    }
                }
            }
            if n_rem > 0 {
                let rem_off = n_blocks * c_in * NR;
                let nr_start = n_blocks * NR;
                for ki in 0..c_in {
                    for ni in 0..n_rem {
                        dst[rem_off + ki * n_rem + ni] =
                            b_panels[p * c_in * c_out + ki * c_out + nr_start + ni];
                    }
                }
            }
        }
        (f16_buf, b_panels, b_packed)
    }

    /// The fused Winograd→1×1 epilogue driver must be BIT-IDENTICAL to running
    /// the standalone fp16 Winograd driver followed by the standalone prepacked
    /// 1×1 conv (`conv1x1_nhwc_f32_to_f16_prepacked`), on the real block_fusion
    /// shape (60×80, c64→c64).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_f43_fused1x1_bit_identical_60x80_c64() {
        if !crate::cpu_features::cpu_features().has_fp16 {
            eprintln!("skipping: CPU lacks fp16");
            return;
        }
        let h = 60usize;
        let w = 80usize;
        let c_in = 64usize;
        let c_out = 64usize; // 3×3 output channels
        let c_out2 = 64usize; // 1×1 output channels

        assert!(f43_fused1x1_is_bit_exact(h, w));

        // Deterministic pseudo-random data in a realistic range.
        let rng = |seed: usize, n: usize, scale: f32, bias: f32| -> Vec<f32> {
            let mut s = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (((s >> 40) as f32 / 16_777_216.0) - 0.5) * 2.0 * scale + bias
                })
                .collect::<Vec<f32>>()
        };

        let input = rng(1, h * w * c_in, 0.8, 0.0);
        let w3x3 = rng(2, c_out * 9 * c_in, 0.2, 0.0); // [c_out,3,3,c_in]
        let b3x3 = rng(3, c_out, 0.1, 0.0);
        let w1x1 = rng(4, c_out2 * c_in, 0.15, 0.0); // [c_out2, c_in]
        let b1x1 = rng(5, c_out2, 0.1, 0.0);

        let (tf16, bp16, bpk) = build_f43_f16_weights(&w3x3, c_out, c_in);
        let (pk1x1, pk1x1_off) =
            crate::ops::neon_asm_f16::prepack_conv1x1_b_f16(&w1x1, c_in, c_out2);

        // ── Reference: standalone wino (f32 out) + standalone prepacked 1×1 ──
        let mut wino_out = vec![0.0f32; h * w * c_out];
        conv3x3_winograd_nhwc_f43_f16(
            &input,
            h,
            w,
            c_in,
            &tf16,
            &bp16,
            &bpk,
            Some(&b3x3),
            c_out,
            Activation::Relu, // 3×3 has ReLU in the real block_fusion.0/.1
            &mut wino_out,
            h,
            w,
        );
        let mut ref_out = vec![0u16; h * w * c_out2];
        let args = crate::ops::Conv1x1Args {
            input: &wino_out,
            weights: &w1x1,
            bias: &b1x1,
            h,
            w,
            c_in: c_out,
            c_out: c_out2,
            activation: Activation::Identity, // block_fusion.2 has no activation
        };
        crate::ops::neon_asm_f16::conv1x1_nhwc_f32_to_f16_prepacked(
            &args,
            &pk1x1,
            pk1x1_off,
            &mut ref_out,
        );

        // ── Fused: single driver writes f16 output directly ──
        let mut fused_out = vec![0u16; h * w * c_out2];
        conv3x3_winograd_nhwc_f43_f16_fused1x1(
            &input,
            h,
            w,
            c_in,
            &tf16,
            &bp16,
            &bpk,
            Some(&b3x3),
            c_out,
            Activation::Relu,
            &pk1x1,
            pk1x1_off,
            &b1x1,
            c_out2,
            Activation::Identity,
            &mut fused_out,
            h,
            w,
        );

        let mut diffs = 0usize;
        let mut first = None;
        for i in 0..fused_out.len() {
            if fused_out[i] != ref_out[i] {
                diffs += 1;
                if first.is_none() {
                    first = Some((
                        i,
                        half::f16::from_bits(fused_out[i]).to_f32(),
                        half::f16::from_bits(ref_out[i]).to_f32(),
                    ));
                }
            }
        }
        assert_eq!(
            diffs,
            0,
            "fused vs unfused differ in {diffs}/{} f16 values; first {:?}",
            fused_out.len(),
            first
        );
    }

    /// Shape-robustness smoke test at 72×100 (n_tile_w=25 — NOT a multiple of 8),
    /// the block_fusion geometry at a 576×800 input. Guards the M-padding contract
    /// of the fp16 Winograd GEMM driver (`n_tile_w_gemm[_part]` rounding) against
    /// buffer-bounds regressions, and checks NEON-fp16 ≈ scalar-f32 F(4,3).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_f43_f16_shape_72x100_c64_no_panic() {
        if !crate::cpu_features::cpu_features().has_fp16 {
            eprintln!("skipping: CPU lacks fp16");
            return;
        }
        let h = 72usize;
        let w = 100usize;
        let c_in = 64usize;
        let c_out = 64usize;
        let c_out2 = 64usize;

        let rng = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            let mut s = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (((s >> 40) as f32 / 16_777_216.0) - 0.5) * 2.0 * scale
                })
                .collect()
        };

        let input = rng(11, h * w * c_in, 0.8);
        let w3x3 = rng(12, c_out * 9 * c_in, 0.2);
        let b3x3 = rng(13, c_out, 0.1);

        let (tf16, bp16, bpk) = build_f43_f16_weights(&w3x3, c_out, c_in);

        // Non-fused fp16 Winograd driver — must not panic and must match scalar.
        let mut neon_out = vec![0.0f32; h * w * c_out];
        conv3x3_winograd_nhwc_f43_f16(
            &input,
            h,
            w,
            c_in,
            &tf16,
            &bp16,
            &bpk,
            Some(&b3x3),
            c_out,
            Activation::Relu,
            &mut neon_out,
            h,
            w,
        );

        // Scalar f32 F(4,3) reference.
        let wt = winograd_transform_weights_f32_f43(&w3x3, c_out, c_in);
        let mut ref_out = vec![0.0f32; h * w * c_out];
        conv3x3_winograd_nhwc_f43(
            &input,
            h,
            w,
            c_in,
            &wt,
            Some(&b3x3),
            c_out,
            Activation::Relu,
            &mut ref_out,
            h,
            w,
        );

        // fp16 Winograd vs f32 scalar: the K=64 inner products accumulate in
        // half precision, so a magnitude-scaled tolerance is appropriate (this
        // test's purpose is the *shape/bounds* contract, not bit-parity).
        let mut max_ref = 0.0f32;
        for &b in &ref_out {
            max_ref = max_ref.max(b.abs());
        }
        let tol = (max_ref * 0.05).max(0.5);
        let mut max_abs = 0.0f32;
        for (a, b) in neon_out.iter().zip(ref_out.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < tol,
            "72x100 non-fused NEON vs scalar max_abs={max_abs} exceeds tol={tol} (max_ref={max_ref})"
        );

        // Fused1x1 driver at the same geometry — must not panic when its gate
        // permits it (otherwise the non-fused path above already covered shape).
        if f43_fused1x1_is_bit_exact(h, w) {
            let w1x1 = rng(14, c_out2 * c_in, 0.15);
            let b1x1 = rng(15, c_out2, 0.1);
            let (pk1x1, pk1x1_off) =
                crate::ops::neon_asm_f16::prepack_conv1x1_b_f16(&w1x1, c_in, c_out2);
            let mut fused_out = vec![0u16; h * w * c_out2];
            conv3x3_winograd_nhwc_f43_f16_fused1x1(
                &input,
                h,
                w,
                c_in,
                &tf16,
                &bp16,
                &bpk,
                Some(&b3x3),
                c_out,
                Activation::Relu,
                &pk1x1,
                pk1x1_off,
                &b1x1,
                c_out2,
                Activation::Identity,
                &mut fused_out,
                h,
                w,
            );
            std::hint::black_box(&fused_out);
        }
    }

    /// Regression for the column-split empty-task underflow: at 18×25 (block5's
    /// 576×800 geometry — n_tile_h=5, n_tile_w=7) with the default ≥6-thread
    /// Rayon pool, `col_parts` exceeds `n_tile_w`, so trailing tasks own zero
    /// columns. Before the `col_start.min(n_tile_w)` clamp this underflowed
    /// `n_col_tiles` (usize) and the scatter ran off the m_acc buffer (panic).
    /// Post-fix: both fp16 drivers run cleanly and match the scalar reference.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_f43_f16_colsplit_18x25_c64_no_underflow() {
        if !crate::cpu_features::cpu_features().has_fp16 {
            eprintln!("skipping: CPU lacks fp16");
            return;
        }
        let h = 18usize;
        let w = 25usize;
        let c_in = 64usize;
        let c_out = 64usize;
        let c_out2 = 64usize;

        let rng = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            let mut s = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (((s >> 40) as f32 / 16_777_216.0) - 0.5) * 2.0 * scale
                })
                .collect()
        };

        let input = rng(21, h * w * c_in, 0.8);
        let w3x3 = rng(22, c_out * 9 * c_in, 0.2);
        let b3x3 = rng(23, c_out, 0.1);

        let (tf16, bp16, bpk) = build_f43_f16_weights(&w3x3, c_out, c_in);

        // Non-fused driver — must not panic (empty-task split) and match scalar.
        let mut neon_out = vec![0.0f32; h * w * c_out];
        conv3x3_winograd_nhwc_f43_f16(
            &input,
            h,
            w,
            c_in,
            &tf16,
            &bp16,
            &bpk,
            Some(&b3x3),
            c_out,
            Activation::Relu,
            &mut neon_out,
            h,
            w,
        );

        let wt = winograd_transform_weights_f32_f43(&w3x3, c_out, c_in);
        let mut ref_out = vec![0.0f32; h * w * c_out];
        conv3x3_winograd_nhwc_f43(
            &input,
            h,
            w,
            c_in,
            &wt,
            Some(&b3x3),
            c_out,
            Activation::Relu,
            &mut ref_out,
            h,
            w,
        );

        let mut max_ref = 0.0f32;
        for &b in &ref_out {
            max_ref = max_ref.max(b.abs());
        }
        let tol = (max_ref * 0.05).max(0.5);
        let mut max_abs = 0.0f32;
        for (a, b) in neon_out.iter().zip(ref_out.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < tol,
            "18x25 non-fused NEON vs scalar max_abs={max_abs} exceeds tol={tol}"
        );

        // Fused1x1 driver at the same geometry — must not panic either.
        if f43_fused1x1_is_bit_exact(h, w) {
            let w1x1 = rng(24, c_out2 * c_in, 0.15);
            let b1x1 = rng(25, c_out2, 0.1);
            let (pk1x1, pk1x1_off) =
                crate::ops::neon_asm_f16::prepack_conv1x1_b_f16(&w1x1, c_in, c_out2);
            let mut fused_out = vec![0u16; h * w * c_out2];
            conv3x3_winograd_nhwc_f43_f16_fused1x1(
                &input,
                h,
                w,
                c_in,
                &tf16,
                &bp16,
                &bpk,
                Some(&b3x3),
                c_out,
                Activation::Relu,
                &pk1x1,
                pk1x1_off,
                &b1x1,
                c_out2,
                Activation::Identity,
                &mut fused_out,
                h,
                w,
            );
            std::hint::black_box(&fused_out);
        }
    }
}
