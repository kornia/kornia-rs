use numpy::{PyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Decode COCO column-major RLE into a column-major flat u8 buffer.
/// flat[x * mh + y] == 1 means pixel (row=y, col=x) is foreground.
fn decode_rle_flat(rle: &[u32], mh: usize, mw: usize) -> Vec<u8> {
    let mask_size = mh * mw;
    let mut flat = vec![0u8; mask_size];
    let mut pos = 0usize;
    let mut is_fg = false;
    for &count in rle {
        let count = count as usize;
        if is_fg && pos < mask_size {
            let end = (pos + count).min(mask_size);
            flat[pos..end].fill(1);
        }
        pos = pos.saturating_add(count);
        if pos >= mask_size {
            break;
        }
        is_fg = !is_fg;
    }
    flat
}

// ── NEON 8×8 byte block transpose (aarch64) ───────────────────────────────────
//
// Reads 8 contiguous bytes from each of 8 columns in column-major flat storage,
// applies three rounds of vtrn at 1-/2-/4-byte granularity, then writes 8 rows
// of 8 bytes to row-major output.  Both read and write sequences are contiguous
// within each block, which keeps the access pattern cache-friendly.

#[cfg(target_arch = "aarch64")]
unsafe fn transpose_8x8_neon(
    src: *const u8, // &flat[bx * mh + by]; columns spaced mh apart
    dst: *mut u8,   // &out[by * mw + bx]; rows spaced mw apart
    mh: usize,
    mw: usize,
) {
    use std::arch::aarch64::*;

    // Load 8 columns × 8 rows from column-major flat
    let a0 = vld1_u8(src);
    let a1 = vld1_u8(src.add(mh));
    let a2 = vld1_u8(src.add(2 * mh));
    let a3 = vld1_u8(src.add(3 * mh));
    let a4 = vld1_u8(src.add(4 * mh));
    let a5 = vld1_u8(src.add(5 * mh));
    let a6 = vld1_u8(src.add(6 * mh));
    let a7 = vld1_u8(src.add(7 * mh));

    // Step 1 — interleave adjacent bytes (stride 1)
    let tr01 = vtrn_u8(a0, a1);
    let tr23 = vtrn_u8(a2, a3);
    let tr45 = vtrn_u8(a4, a5);
    let tr67 = vtrn_u8(a6, a7);

    // Step 2 — interleave adjacent 2-byte words (stride 2)
    let t0246_a = vtrn_u16(vreinterpret_u16_u8(tr01.0), vreinterpret_u16_u8(tr23.0));
    let t0246_b = vtrn_u16(vreinterpret_u16_u8(tr01.1), vreinterpret_u16_u8(tr23.1));
    let t4567_a = vtrn_u16(vreinterpret_u16_u8(tr45.0), vreinterpret_u16_u8(tr67.0));
    let t4567_b = vtrn_u16(vreinterpret_u16_u8(tr45.1), vreinterpret_u16_u8(tr67.1));

    // Step 3 — interleave adjacent 4-byte words (stride 4) → complete rows
    let r04 = vtrn_u32(
        vreinterpret_u32_u8(vreinterpret_u8_u16(t0246_a.0)),
        vreinterpret_u32_u8(vreinterpret_u8_u16(t4567_a.0)),
    );
    let r15 = vtrn_u32(
        vreinterpret_u32_u8(vreinterpret_u8_u16(t0246_b.0)),
        vreinterpret_u32_u8(vreinterpret_u8_u16(t4567_b.0)),
    );
    let r26 = vtrn_u32(
        vreinterpret_u32_u8(vreinterpret_u8_u16(t0246_a.1)),
        vreinterpret_u32_u8(vreinterpret_u8_u16(t4567_a.1)),
    );
    let r37 = vtrn_u32(
        vreinterpret_u32_u8(vreinterpret_u8_u16(t0246_b.1)),
        vreinterpret_u32_u8(vreinterpret_u8_u16(t4567_b.1)),
    );

    // Store 8 complete rows (each 8 bytes) to row-major output
    vst1_u8(dst,             vreinterpret_u8_u32(r04.0)); // row 0
    vst1_u8(dst.add(mw),     vreinterpret_u8_u32(r15.0)); // row 1
    vst1_u8(dst.add(2 * mw), vreinterpret_u8_u32(r26.0)); // row 2
    vst1_u8(dst.add(3 * mw), vreinterpret_u8_u32(r37.0)); // row 3
    vst1_u8(dst.add(4 * mw), vreinterpret_u8_u32(r04.1)); // row 4
    vst1_u8(dst.add(5 * mw), vreinterpret_u8_u32(r15.1)); // row 5
    vst1_u8(dst.add(6 * mw), vreinterpret_u8_u32(r26.1)); // row 6
    vst1_u8(dst.add(7 * mw), vreinterpret_u8_u32(r37.1)); // row 7
}

/// Transpose a column-major flat u8 array to a row-major 2-D array.
/// Uses NEON 8×8 block transpose on aarch64; scalar on other architectures.
fn col_major_to_row_major(flat: &[u8], out: &mut [u8], mh: usize, mw: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        let bh = mh / 8; // full 8-row blocks
        let bw = mw / 8; // full 8-column blocks

        for bx in 0..bw {
            for by in 0..bh {
                unsafe {
                    transpose_8x8_neon(
                        flat.as_ptr().add(bx * 8 * mh + by * 8),
                        out.as_mut_ptr().add(by * 8 * mw + bx * 8),
                        mh,
                        mw,
                    );
                }
            }
        }

        // Scalar cleanup: remaining rows in fully-covered columns
        for bx in 0..bw * 8 {
            for by in bh * 8..mh {
                out[by * mw + bx] = flat[bx * mh + by];
            }
        }
        // Scalar cleanup: remaining columns (all rows)
        for bx in bw * 8..mw {
            for by in 0..mh {
                out[by * mw + bx] = flat[bx * mh + by];
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for x in 0..mw {
            for y in 0..mh {
                out[y * mw + x] = flat[x * mh + y];
            }
        }
    }
}

// ── Public Python functions ───────────────────────────────────────────────────

/// Decode COCO column-major RLE → (H, W) uint8 mask (1 = foreground).
///
/// `rle`   — alternating background/foreground run counts; first run is always
///           the background count (may be 0 if the top-left pixel is foreground).
/// `shape` — `(H, W)` of the original mask in pixels.
///
/// The transpose from column-major RLE to row-major numpy array uses NEON 8×8
/// block transpose on aarch64 for maximum throughput.
#[pyfunction]
pub fn rle_to_mask<'py>(
    py: Python<'py>,
    rle: Vec<u32>,
    shape: (usize, usize),
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let (mh, mw) = shape;
    if mh == 0 || mw == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "shape dimensions must be > 0",
        ));
    }
    let flat = decode_rle_flat(&rle, mh, mw);
    let arr = unsafe { PyArray::<u8, _>::new(py, [mh, mw], false) };
    let out = unsafe { std::slice::from_raw_parts_mut(arr.data(), mh * mw) };
    col_major_to_row_major(&flat, out, mh, mw);
    Ok(arr)
}

/// Encode a (H, W) uint8 mask → COCO column-major RLE.
///
/// Nonzero pixels are treated as foreground.  The first element of the returned
/// list is always the background run count (may be 0).
#[pyfunction]
pub fn mask_to_rle(
    _py: Python<'_>,
    mask: &Bound<'_, PyArray2<u8>>,
) -> PyResult<Vec<u32>> {
    if !mask.is_c_contiguous() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "mask must be C-contiguous",
        ));
    }
    let shape = mask.shape();
    let mh = shape[0];
    let mw = shape[1];
    let src = unsafe { std::slice::from_raw_parts(mask.data(), mh * mw) };

    // Iterate in column-major order: x in 0..mw, y in 0..mh.
    // Starting with is_fg = false guarantees the first push is the bg count.
    let mut rle: Vec<u32> = Vec::new();
    let mut is_fg = false;
    let mut run: u32 = 0;

    for x in 0..mw {
        for y in 0..mh {
            let pixel_fg = src[y * mw + x] != 0;
            if pixel_fg == is_fg {
                run += 1;
            } else {
                rle.push(run);
                run = 1;
                is_fg = pixel_fg;
            }
        }
    }
    rle.push(run);
    Ok(rle)
}
