//! Build numpy array views over a raw, externally-owned data pointer.
//!
//! These helpers wrap a `*mut u8` (the data pointer of an `Image`'s
//! [`crate::backing::Backing`]) in a C-contiguous numpy `ndarray` whose `base`
//! is a Python keep-alive object. As long as the keep-alive outlives the view,
//! the buffer stays valid — this is how an owned `AlignedBytes` buffer is
//! exposed to numpy/torch without a copy.

use numpy::npyffi::{self, npy_intp, PY_ARRAY_API};
use numpy::{Element, PyArray3, PyArrayDescrMethods};
use pyo3::prelude::*;
use std::os::raw::c_int;

/// Ranks used anywhere in this crate: 3 (`Image`, HWC) and 4 (`Tensor`, NCHW).
const MAX_RANK: usize = 4;

/// Create a C-contiguous numpy view of rank `dims.len()` and element type `T`
/// over `data_ptr`, tying its lifetime to `base`.
///
/// `base` is installed as the numpy array's base object; it must keep the
/// memory at `data_ptr` alive for at least as long as the returned array (and
/// any view derived from it). When `readonly` is true the `WRITEABLE` flag is
/// cleared so numpy refuses in-place writes.
///
/// # Safety
///
/// - `data_ptr` must point to at least `dims.iter().product()` valid elements
///   of `T`, laid out C-contiguously.
/// - `base` must own / keep alive that memory.
/// - `dims.len() <= MAX_RANK`.
pub unsafe fn view<'py, T: Element>(
    py: Python<'py>,
    data_ptr: *mut u8,
    dims: &[usize],
    base: Py<PyAny>,
    readonly: bool,
) -> PyResult<Bound<'py, PyAny>> {
    // Real (release-safe) bound check, not a debug_assert: `dims.len()` is
    // passed to numpy as the array rank against the fixed `dims_buf` below, so a
    // rank past MAX_RANK would make numpy read uninitialized stack past the
    // buffer. No current caller exceeds MAX_RANK, but this keeps the `pub unsafe`
    // contract enforced in release builds too.
    if dims.len() > MAX_RANK {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "numpy_view::view: rank {} exceeds MAX_RANK {MAX_RANK}",
            dims.len()
        )));
    }
    let mut dims_buf = [0 as npy_intp; MAX_RANK];
    for (slot, &d) in dims_buf.iter_mut().zip(dims) {
        *slot = d as npy_intp;
    }
    let flags = if readonly {
        0
    } else {
        npyffi::NPY_ARRAY_WRITEABLE
    };
    // SAFETY: numpy C-API call with a valid dtype, dims and data pointer.
    let ptr = unsafe {
        PY_ARRAY_API.PyArray_NewFromDescr(
            py,
            PY_ARRAY_API.get_type_object(py, npyffi::NpyTypes::PyArray_Type),
            T::get_dtype(py).into_dtype_ptr(),
            dims.len() as c_int,
            dims_buf.as_mut_ptr(),
            std::ptr::null_mut(), // strides: NULL => C-contiguous
            data_ptr as *mut std::ffi::c_void,
            flags as c_int,
            std::ptr::null_mut(),
        )
    };
    if ptr.is_null() {
        return Err(PyErr::fetch(py));
    }
    // Install the keep-alive as base so the buffer outlives the view.
    let base_ptr = base.into_ptr();
    // SAFETY: `ptr` is a fresh array object; `base_ptr` is a new strong ref that
    // PyArray_SetBaseObject steals on success.
    let rc = unsafe {
        PY_ARRAY_API.PyArray_SetBaseObject(py, ptr as *mut npyffi::PyArrayObject, base_ptr)
    };
    if rc != 0 {
        // SetBaseObject failed and already decref'd base_ptr.
        // Decref the freshly-allocated array to avoid leaking it.
        unsafe { pyo3::ffi::Py_DECREF(ptr) };
        return Err(PyErr::fetch(py));
    }
    // SAFETY: ptr is an owned reference to a PyAny wrapping the fresh ndarray.
    Ok(unsafe { Bound::from_owned_ptr(py, ptr) })
}

/// Create a C-contiguous `PyArray3<T>` view over `data_ptr` with shape
/// `(h, w, c)`. See [`view`] for the safety contract.
///
/// # Safety
/// Same contract as [`view`], with `dims = [h, w, c]`.
pub unsafe fn view3<T: Element>(
    py: Python<'_>,
    data_ptr: *mut u8,
    h: usize,
    w: usize,
    c: usize,
    base: Py<PyAny>,
    readonly: bool,
) -> PyResult<Py<PyArray3<T>>> {
    // SAFETY: forwarded from the caller's contract on `view`.
    let arr = unsafe { view::<T>(py, data_ptr, &[h, w, c], base, readonly)? };
    Ok(arr.cast_into_unchecked::<PyArray3<T>>().unbind())
}
