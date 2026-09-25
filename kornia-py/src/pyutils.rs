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
use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyValueError};
use pyo3::prelude::*;

use kornia_algebra::{Mat3F64, Vec2F64};

/// Map any displayable error (or message) to a Python `ValueError`.
pub(crate) fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `true` if `ptr` is a multiple of `align` (which must be non-zero).
#[inline]
pub(crate) fn is_ptr_aligned(ptr: *const u8, align: usize) -> bool {
    (ptr as usize).is_multiple_of(align)
}

/// Require `ptr` to be aligned to `align` bytes.
///
/// Typed (`u16`/`f32`/...) access through a misaligned pointer is undefined
/// behaviour, so every raw buffer that is reinterpreted as `&[T]` goes through
/// this. Returns a `ValueError` naming `what` otherwise.
pub(crate) fn require_ptr_aligned(
    ptr: *const u8,
    align: usize,
    what: impl std::fmt::Display,
) -> PyResult<()> {
    if is_ptr_aligned(ptr, align) {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{what} data pointer is not aligned to {align} bytes for its dtype; pass a copy \
         (np.ascontiguousarray(...) / .copy())"
    )))
}

/// Require the data pointer of `arr` to be aligned for `T`.
///
/// Needed before `as_array()` (ndarray views, which honour arbitrary strides
/// but assume aligned elements) on arrays that may be misaligned views.
pub(crate) fn require_aligned<T: Element, D: Dimension>(
    arr: &Bound<'_, PyArray<T, D>>,
    what: impl std::fmt::Display,
) -> PyResult<()> {
    require_ptr_aligned(arr.data() as *const u8, std::mem::align_of::<T>(), what)
}

/// Require `arr` to be C-contiguous with a data pointer aligned for `T`.
///
/// Returns a `ValueError` naming `what` otherwise. Non-contiguous arrays
/// (negative / zero / non-unit strides) would make a flat `len`-element read
/// run past the real allocation; a misaligned pointer makes any typed `&[T]`
/// over it undefined behaviour.
pub(crate) fn require_c_contig_aligned<T: Element, D: Dimension>(
    arr: &Bound<'_, PyArray<T, D>>,
    what: impl std::fmt::Display,
) -> PyResult<()> {
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{what} must be a C-contiguous array (got a strided/broadcast view; \
             pass np.ascontiguousarray(...))"
        )));
    }
    require_aligned(arr, what)
}

/// Borrow the elements of a numpy array as a flat row-major slice.
///
/// Validates C-contiguity and alignment first (see
/// [`require_c_contig_aligned`]); the slice length is the array's real
/// element count, never a caller-computed product.
pub(crate) fn c_slice<'a, T: Element, D: Dimension>(
    arr: &'a Bound<'_, PyArray<T, D>>,
    what: impl std::fmt::Display,
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
    what: impl std::fmt::Display,
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
/// Returns an `OverflowError` instead of silently wrapping. This is the single
/// size check behind every user-controlled shape in the bindings (see also
/// [`checked_bytes`] and `backing::byte_len`), so the same overflow raises the
/// same exception everywhere.
pub(crate) fn checked_numel(dims: &[usize], itemsize: usize) -> PyResult<usize> {
    let n = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
    match n {
        Some(n)
            if n.checked_mul(itemsize)
                .is_some_and(|b| b <= isize::MAX as usize) =>
        {
            Ok(n)
        }
        _ => Err(PyOverflowError::new_err(format!(
            "array dimensions {dims:?} are too large"
        ))),
    }
}

/// Checked total byte size of an array with `dims` and `itemsize`-byte
/// elements; see [`checked_numel`].
pub(crate) fn checked_bytes(dims: &[usize], itemsize: usize) -> PyResult<usize> {
    // Cannot overflow: `checked_numel` bounds `n * itemsize` by `isize::MAX`.
    Ok(checked_numel(dims, itemsize)? * itemsize)
}

/// Validate a caller-provided `out=` array before writing through its pointer.
///
/// Checks, in order: the shape is exactly `expected`; the array is writeable
/// (a raw-pointer write would otherwise silently mutate e.g. an immutable
/// `bytes` buffer); it is C-contiguous and aligned for `T`; and its bytes do
/// not overlap `[src_ptr, src_ptr + src_bytes)` (the kernels read the source
/// while writing `out`, so aliasing them is a data race / Rust aliasing
/// violation). Error messages are prefixed with `"{op}: out"`, which is only
/// formatted on the error path.
pub(crate) fn validate_out<T: Element, D: Dimension>(
    arr: &Bound<'_, PyArray<T, D>>,
    expected: &[usize],
    src_ptr: *const u8,
    src_bytes: usize,
    op: &str,
) -> PyResult<()> {
    if arr.shape() != expected {
        return Err(PyValueError::new_err(format!(
            "{op}: out shape {:?} must be {expected:?}",
            arr.shape()
        )));
    }
    require_writeable(arr, format_args!("{op}: out"))?;
    require_c_contig_aligned(arr, format_args!("{op}: out"))?;
    let out_bytes = arr.len() * std::mem::size_of::<T>();
    if ranges_overlap(arr.data() as *const u8, out_bytes, src_ptr, src_bytes) {
        return Err(PyValueError::new_err(format!(
            "{op}: out must not share memory with the input image"
        )));
    }
    Ok(())
}

/// Allocate a zero-filled `Vec<T>` of `n` elements, reporting allocation
/// failure as a Python `MemoryError` instead of aborting the interpreter
/// (`n` is often user-controlled).
pub(crate) fn try_zeroed_vec<T: Clone + Default>(n: usize) -> PyResult<Vec<T>> {
    let mut v: Vec<T> = Vec::new();
    v.try_reserve_exact(n).map_err(|_| {
        PyMemoryError::new_err(format!(
            "cannot allocate a buffer of {n} x {}-byte elements",
            std::mem::size_of::<T>()
        ))
    })?;
    v.resize(n, T::default());
    Ok(v)
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
        .map_err(value_err)
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
