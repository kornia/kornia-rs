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
pub unsafe fn view<T: Element>(
    py: Python<'_>,
    data_ptr: *mut u8,
    dims: &[usize],
    base: Py<PyAny>,
    readonly: bool,
) -> PyResult<Py<PyAny>> {
    let mut dims: Vec<npy_intp> = dims.iter().map(|&d| d as npy_intp).collect();
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
            dims.as_mut_ptr(),
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
    Ok(unsafe { Bound::from_owned_ptr(py, ptr) }.unbind())
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
    Ok(arr.bind(py).clone().cast_into_unchecked::<PyArray3<T>>().unbind())
}
