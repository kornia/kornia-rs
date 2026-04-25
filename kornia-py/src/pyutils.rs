//! Numpy ↔ kornia-algebra small-matrix/vector conversions shared across the
//! geometry bindings (homography, two-view pose, …).
//!
//! All functions here assume the caller has already validated shape/dtype/
//! contiguity — they are thin reinterpret-then-copy helpers, not validators.

use numpy::{PyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use kornia_algebra::{Mat3F64, Vec2F64};

/// Copy a `(N, 2)` C-contiguous float64 numpy array into a `Vec<Vec2F64>`.
pub(crate) fn unpack_pts(arr: &Bound<'_, PyArray2<f64>>) -> Vec<Vec2F64> {
    let n = arr.shape()[0];
    unsafe {
        let raw = std::slice::from_raw_parts(arr.data(), n * 2);
        (0..n)
            .map(|i| Vec2F64::new(raw[i * 2], raw[i * 2 + 1]))
            .collect()
    }
}

/// Validate and copy a `(3, 3)` C-contiguous float64 numpy array into a
/// column-major `Mat3F64`. Returns `PyValueError` on mismatched shape or
/// non-contiguous layout.
pub(crate) fn unpack_mat3(arr: &Bound<'_, PyArray2<f64>>) -> PyResult<Mat3F64> {
    if !arr.is_c_contiguous() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "matrix must be C-contiguous",
        ));
    }
    let s = arr.shape();
    if s[0] != 3 || s[1] != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "expected (3, 3) float64 matrix, got ({}, {})",
            s[0], s[1]
        )));
    }
    // Numpy is row-major, Mat3F64 is column-major — transpose on load.
    let raw = unsafe { std::slice::from_raw_parts(arr.data(), 9) };
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
