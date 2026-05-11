use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

// ── Helpers ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum SampleMethod {
    Mean,
    Median,
}

#[inline]
fn is_valid(m: u8, d: u16) -> bool {
    m != 0 && d > 0
}

#[inline]
fn ensure_c_contiguous(ok: bool, name: &str) -> PyResult<()> {
    if ok {
        Ok(())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{name} must be C-contiguous"
        )))
    }
}

fn extract_mask_info(item: &Bound<'_, PyAny>) -> PyResult<(usize, usize, usize)> {
    let mask = item.cast::<PyArray2<u8>>().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "each mask must be a 2D uint8 numpy array (use segmentation.rle_to_mask)",
        )
    })?;
    ensure_c_contiguous(mask.is_c_contiguous(), "each mask")?;
    let mshape = mask.shape();
    Ok((mask.data() as usize, mshape[0], mshape[1]))
}

// ── NEON: mean (fully vectorised, same resolution) ────────────────────────────
//
// Processes 8 pixels/iteration.  For each chunk:
//   - combined_valid  = (mask != 0) AND (depth > 0)   — u16x8 boolean (0xFFFF/0)
//   - masked_depth    = depth AND combined_valid        — zero-out invalid lanes
//   - sum            += horizontal sum of masked_depth
//   - count          += number of 0xFFFF lanes          (via negate trick)
// No collection step, no branch per pixel.

#[cfg(target_arch = "aarch64")]
unsafe fn mean_neon_same_res(depth_slice: &[u16], mask_slice: &[u8], n: usize) -> (u32, bool) {
    use std::arch::aarch64::*;

    let mut sum: u64 = 0;
    let mut count: u32 = 0;
    let mut i = 0;

    while i + 8 <= n {
        let m8 = vld1_u8(mask_slice.as_ptr().add(i));
        let d8 = vld1q_u16(depth_slice.as_ptr().add(i));

        // mask_fg_u16: 0xFFFF where mask byte != 0, else 0
        let mask_fg_u8 = vmvn_u8(vceq_u8(m8, vdup_n_u8(0)));
        let mask_fg_u16 = vmvnq_u16(vceqq_u16(vmovl_u8(mask_fg_u8), vdupq_n_u16(0)));

        // depth_valid: 0xFFFF where depth > 0, else 0
        let depth_valid = vcgtq_u16(d8, vdupq_n_u16(0));

        // Combined validity and masked depth
        let valid = vandq_u16(mask_fg_u16, depth_valid);
        let masked = vandq_u16(d8, valid);

        // Accumulate: vaddlvq_u16 widens u16x8 → u64 horizontal sum
        sum += vaddlvq_u16(masked) as u64;
        // Count valid lanes: 0xFFFF → -1 as i16 → negate → 1
        let ones = vreinterpretq_u16_s16(vnegq_s16(vreinterpretq_s16_u16(valid)));
        count += vaddvq_u16(ones) as u32;

        i += 8;
    }

    // Scalar tail
    while i < n {
        if is_valid(mask_slice[i], depth_slice[i]) {
            sum += depth_slice[i] as u64;
            count += 1;
        }
        i += 1;
    }

    if count == 0 {
        return (0, false);
    }
    ((sum / count as u64) as u32, true)
}

// ── NEON: median collection (same resolution) ─────────────────────────────────
//
// Uses vmaxv_u8 on 8-pixel mask chunks as a cheap gate: if the whole chunk is
// all-background (max == 0) we skip 8 pixels with a single NEON instruction.
// Typical segmentation masks are >80% background, so the majority of chunks
// never touch the depth buffer at all.  Foreground chunks fall through to a
// tight scalar inner loop over just 8 elements.

#[cfg(target_arch = "aarch64")]
unsafe fn collect_neon_same_res(depth_slice: &[u16], mask_slice: &[u8], n: usize) -> Vec<u16> {
    use std::arch::aarch64::*;

    let mut values: Vec<u16> = Vec::with_capacity(n / 8);
    let mut i = 0;

    while i + 8 <= n {
        let m8 = vld1_u8(mask_slice.as_ptr().add(i));
        // Skip the whole chunk if all mask values are zero (background)
        if vmaxv_u8(m8) == 0 {
            i += 8;
            continue;
        }
        // At least one foreground pixel — scalar collect over the 8 elements
        for j in 0..8usize {
            let m = *mask_slice.as_ptr().add(i + j);
            let v = *depth_slice.as_ptr().add(i + j);
            if is_valid(m, v) {
                values.push(v);
            }
        }
        i += 8;
    }
    // Scalar tail
    while i < n {
        if is_valid(mask_slice[i], depth_slice[i]) {
            values.push(depth_slice[i]);
        }
        i += 1;
    }
    values
}

// ── Scalar fallback: resize-aware sampling ────────────────────────────────────
//
// Used when mask and depth have different dimensions (nearest-neighbour resize)
// and on non-aarch64 targets.

fn sample_one_scalar(
    depth_slice: &[u16],
    dh: usize,
    dw: usize,
    mask_slice: &[u8],
    mh: usize,
    mw: usize,
    method: SampleMethod,
) -> (u32, bool) {
    let mut values: Vec<u16> = Vec::with_capacity((dh * dw) / 8);

    if dh == mh && dw == mw {
        for dy in 0..dh {
            let row = dy * dw;
            for dx in 0..dw {
                let v = depth_slice[row + dx];
                if is_valid(mask_slice[row + dx], v) {
                    values.push(v);
                }
            }
        }
    } else {
        for dy in 0..dh {
            let my = (dy * mh / dh).min(mh - 1);
            let depth_row = dy * dw;
            let mask_row = my * mw;
            for dx in 0..dw {
                let mx = (dx * mw / dw).min(mw - 1);
                let v = depth_slice[depth_row + dx];
                if is_valid(mask_slice[mask_row + mx], v) {
                    values.push(v);
                }
            }
        }
    }

    if values.is_empty() {
        return (0, false);
    }
    match method {
        SampleMethod::Mean => {
            let sum: u64 = values.iter().map(|&v| v as u64).sum();
            ((sum / values.len() as u64) as u32, true)
        }
        SampleMethod::Median => {
            values.sort_unstable();
            (values[values.len() / 2] as u32, true)
        }
    }
}

// ── Dispatcher ────────────────────────────────────────────────────────────────

fn sample_one(
    depth_slice: &[u16],
    dh: usize,
    dw: usize,
    mask_slice: &[u8],
    mh: usize,
    mw: usize,
    method: SampleMethod,
) -> (u32, bool) {
    #[cfg(target_arch = "aarch64")]
    if dh == mh && dw == mw {
        let n = dh * dw;
        return match method {
            SampleMethod::Mean => unsafe { mean_neon_same_res(depth_slice, mask_slice, n) },
            SampleMethod::Median => {
                let mut vals = unsafe { collect_neon_same_res(depth_slice, mask_slice, n) };
                if vals.is_empty() {
                    return (0, false);
                }
                vals.sort_unstable();
                (vals[vals.len() / 2] as u32, true)
            }
        };
    }
    sample_one_scalar(depth_slice, dh, dw, mask_slice, mh, mw, method)
}

// ── Public Python function ────────────────────────────────────────────────────

/// Sample depth under a list of masks and return `(value, is_valid)` per mask.
///
/// `depth`  — (H_d, W_d) uint16 numpy array, C-contiguous.
/// `masks`  — Python list of (H_m, W_m) **uint8** numpy arrays.  Use
///            `kornia_rs.segmentation.rle_to_mask` to decode from COCO RLE.
///            Masks with different resolution from `depth` are rescaled via
///            nearest-neighbour.
/// `method` — "median" (default) or "mean".
///
/// On aarch64 with same-resolution masks, sampling is NEON-vectorised.
/// Across masks, sampling always runs in parallel via rayon (GIL released).
#[pyfunction]
#[pyo3(signature = (depth, masks, method = "median"))]
pub fn sample_depth(
    py: Python<'_>,
    depth: &Bound<'_, PyArray2<u16>>,
    masks: &Bound<'_, PyList>,
    method: &str,
) -> PyResult<Vec<(u32, bool)>> {
    ensure_c_contiguous(depth.is_c_contiguous(), "depth")?;
    let method = match method {
        "mean" => SampleMethod::Mean,
        "median" => SampleMethod::Median,
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "method must be 'median' or 'mean', got '{other}'"
            )))
        }
    };

    let dshape = depth.shape();
    let dh = dshape[0];
    let dw = dshape[1];
    let depth_ptr = depth.data() as usize;

    let mask_infos = masks
        .iter()
        .map(|item| extract_mask_info(&item))
        .collect::<PyResult<Vec<_>>>()?;

    let results = py.detach(|| {
        mask_infos
            .par_iter()
            .map(|&(mask_ptr, mh, mw)| {
                let depth_slice =
                    unsafe { std::slice::from_raw_parts(depth_ptr as *const u16, dh * dw) };
                let mask_slice =
                    unsafe { std::slice::from_raw_parts(mask_ptr as *const u8, mh * mw) };
                sample_one(depth_slice, dh, dw, mask_slice, mh, mw, method)
            })
            .collect::<Vec<_>>()
    });

    Ok(results)
}
