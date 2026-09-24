//! Numpy ↔ kornia-algebra small-matrix/vector conversions shared across the
//! geometry bindings (homography, two-view pose, …), plus the memory-safety
//! validators every raw numpy buffer access in this crate goes through.
//!
//! Numpy arrays handed in from Python are untrusted: they can be strided
//! (`a[::-1]`), zero-stride broadcasts (`np.broadcast_to`), misaligned
//! (`np.frombuffer(buf, offset=1, dtype=np.float32)`) or read-only. Reading
//! `len` contiguous elements from `arr.data()` is only sound once the array
//! is known to be C-contiguous and aligned — use [`c_slice`] (or
//! [`require_c_contig_aligned`]) instead of `slice::from_raw_parts` directly.

use numpy::ndarray::Dimension;
use numpy::{Element, PyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kornia_algebra::{Mat3F64, Vec2F64};

/// Map any displayable error to a Python `ValueError`.
pub(crate) fn to_value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Require `arr` to be C-contiguous with a data pointer aligned for `T`.
///
/// Returns a `ValueError` naming `what` otherwise. Non-contiguous arrays
/// (negative / zero / non-unit strides) would make a flat `len`-element read
/// run past the real allocation; a misaligned pointer makes any typed `&[T]`
/// over it undefined behaviour.
pub(crate) fn require_c_contig_aligned<T: Element, D: Dimension>(
    arr: &Bound<'_, PyArray<T, D>>,
    what: &str,
) -> PyResult<()> {
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{what} must be a C-contiguous array (got a strided/broadcast view; \
             pass np.ascontiguousarray(...))"
        )));
    }
    if !(arr.data() as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return Err(PyValueError::new_err(format!(
            "{what} data pointer is not aligned for its dtype; pass a copy \
             (np.ascontiguousarray(...) / .copy())"
        )));
    }
    Ok(())
}

/// Require the data pointer of `arr` to be aligned for `T`.
///
/// Needed before `as_array()` (ndarray views, which honour arbitrary strides
/// but assume aligned elements) on arrays that may be misaligned views.
pub(crate) fn require_aligned<T: Element, D: Dimension>(
    arr: &Bound<'_, PyArray<T, D>>,
    what: &str,
) -> PyResult<()> {
    if !(arr.data() as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return Err(PyValueError::new_err(format!(
            "{what} data pointer is not aligned for its dtype; pass a copy \
             (np.ascontiguousarray(...) / .copy())"
        )));
    }
    Ok(())
}

/// Borrow the elements of a numpy array as a flat row-major slice.
///
/// Validates C-contiguity and alignment first (see
/// [`require_c_contig_aligned`]); the slice length is the array's real
/// element count, never a caller-computed product.
pub(crate) fn c_slice<'a, T: Element, D: Dimension>(
    arr: &'a Bound<'_, PyArray<T, D>>,
    what: &str,
) -> PyResult<&'a [T]> {
    require_c_contig_aligned(arr, what)?;
    let len = arr.len();
    if len == 0 {
        return Ok(&[]);
    }
    // SAFETY: the array is C-contiguous and aligned (checked above), so its
    // `len` elements are laid out back to back starting at `data()`, which is
    // non-null for a non-empty array. The returned slice borrows `arr`, which
    // holds a strong reference, so the buffer outlives the slice. The GIL is
    // held for the lifetime of the `Bound`, so Python code cannot resize or
    // free the buffer while the slice is in use on this thread.
    Ok(unsafe { std::slice::from_raw_parts(arr.data() as *const T, len) })
}

/// Require that `arr` has numpy's `WRITEABLE` flag set.
///
/// `out=` destinations are written through a raw pointer, which would silently
/// bypass numpy's read-only protection (e.g. an `np.frombuffer(bytes)` view).
pub(crate) fn require_writeable<T: Element, D: Dimension>(
    arr: &Bound<'_, PyArray<T, D>>,
    what: &str,
) -> PyResult<()> {
    // SAFETY: `as_array_ptr` returns the live `PyArrayObject` behind `arr`
    // (kept alive by the `Bound`); reading its `flags` field is a plain load.
    let flags = unsafe { (*arr.as_array_ptr()).flags };
    if flags & numpy::npyffi::NPY_ARRAY_WRITEABLE == 0 {
        return Err(PyValueError::new_err(format!(
            "{what} is read-only; pass a writeable array"
        )));
    }
    Ok(())
}

/// `true` if the byte ranges `[a, a + a_len)` and `[b, b + b_len)` intersect.
pub(crate) fn ranges_overlap(a: *const u8, a_len: usize, b: *const u8, b_len: usize) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    let (a0, b0) = (a as usize, b as usize);
    a0 < b0.saturating_add(b_len) && b0 < a0.saturating_add(a_len)
}

/// Checked element count of an array with `dims`, whose total byte size with
/// `itemsize`-byte elements must also fit in `isize` (Rust's allocation limit).
///
/// Returns a `ValueError` on overflow instead of silently wrapping.
pub(crate) fn checked_numel(dims: &[usize], itemsize: usize) -> PyResult<usize> {
    let n = dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| PyValueError::new_err(format!("array dimensions {dims:?} overflow")))?;
    match n.checked_mul(itemsize) {
        Some(bytes) if bytes <= isize::MAX as usize => Ok(n),
        _ => Err(PyValueError::new_err(format!(
            "array dimensions {dims:?} are too large"
        ))),
    }
}

/// Reshape a flat row-major block into `(len / cols, cols)`.
///
/// `PyArray2::from_vec2` derives the column count from the first row, so an
/// empty result would come back as `(0, 0)` instead of `(0, cols)` and break
/// any caller that slices a column or stacks the array. Going through a flat
/// vector also skips the per-row `Vec` allocation and copy.
pub(crate) fn rows_to_numpy<T: numpy::Element>(
    py: Python<'_>,
    flat: Vec<T>,
    cols: usize,
) -> PyResult<Bound<'_, PyArray2<T>>> {
    let rows = flat.len() / cols;
    numpy::PyArray1::from_vec(py, flat)
        .reshape([rows, cols])
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Copy a `(N, 2)` C-contiguous float64 numpy array into a `Vec<Vec2F64>`.
///
/// Returns `ValueError` if the array is not `(N, 2)`, not C-contiguous, or
/// misaligned.
pub(crate) fn unpack_pts(arr: &Bound<'_, PyArray2<f64>>) -> PyResult<Vec<Vec2F64>> {
    let s = arr.shape();
    if s[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "expected (N, 2) float64 points, got ({}, {})",
            s[0], s[1]
        )));
    }
    let raw = c_slice(arr, "point array")?;
    Ok(raw
        .chunks_exact(2)
        .map(|p| Vec2F64::new(p[0], p[1]))
        .collect())
}

/// Validate and copy a `(3, 3)` C-contiguous float64 numpy array into a
/// column-major `Mat3F64`. Returns `PyValueError` on mismatched shape,
/// non-contiguous layout or a misaligned buffer.
pub(crate) fn unpack_mat3(arr: &Bound<'_, PyArray2<f64>>) -> PyResult<Mat3F64> {
    let s = arr.shape();
    if s[0] != 3 || s[1] != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "expected (3, 3) float64 matrix, got ({}, {})",
            s[0], s[1]
        )));
    }
    let raw = c_slice(arr, "matrix")?;
    // Numpy is row-major, Mat3F64 is column-major — transpose on load.
    Ok(Mat3F64::from_cols_array(&[
        raw[0], raw[3], raw[6], raw[1], raw[4], raw[7], raw[2], raw[5], raw[8],
    ]))
}

/// Pack a `Mat3F64` into a row-major `(3, 3)` numpy array (column-major →
/// row-major transpose on write).
pub(crate) fn mat3_to_py<'py>(py: Python<'py>, m: &Mat3F64) -> Bound<'py, PyArray2<f64>> {
    let cols = m.to_cols_array();
    let arr = unsafe { PyArray::<f64, _>::new(py, [3, 3], false) };
    let slice = unsafe { std::slice::from_raw_parts_mut(arr.data(), 9) };
    for r in 0..3 {
        for c in 0..3 {
            slice[r * 3 + c] = cols[c * 3 + r];
        }
    }
    arr
}

/// Pack a `&[bool]` inlier mask into a `(N,)` uint8 numpy array (1 = inlier).
pub(crate) fn mask_to_py<'py>(py: Python<'py>, inliers: &[bool]) -> Bound<'py, PyArray1<u8>> {
    let n = inliers.len();
    let arr = unsafe { PyArray::<u8, _>::new(py, [n], false) };
    let slice = unsafe { std::slice::from_raw_parts_mut(arr.data(), n) };
    for (dst, src) in slice.iter_mut().zip(inliers.iter()) {
        *dst = *src as u8;
    }
    arr
}
