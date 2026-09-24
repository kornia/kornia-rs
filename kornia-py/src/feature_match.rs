use numpy::{PyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::pyutils::{c_slice, to_value_err};

use kornia_imgproc::features::match_descriptors;

/// Brute-force matcher for 32-byte binary descriptors (ORB, BRIEF).
///
/// Args:
///     descriptors1: `(M, 32)` uint8 query descriptors.
///     descriptors2: `(N, 32)` uint8 train descriptors.
///     max_distance: optional Hamming-distance cap.
///     cross_check: if True, keep only mutual nearest neighbors (OpenCV BFMatcher `crossCheck=True`).
///     max_ratio: optional Lowe's ratio test threshold (`best / second_best < ratio`).
///
/// Returns:
///     `(K, 2)` int64 array of `(query_idx, train_idx)` pairs.
#[pyfunction(name = "match_descriptors")]
#[pyo3(signature = (descriptors1, descriptors2, max_distance=None, cross_check=false, max_ratio=None))]
pub fn match_descriptors_py(
    py: Python<'_>,
    descriptors1: Bound<'_, PyArray2<u8>>,
    descriptors2: Bound<'_, PyArray2<u8>>,
    max_distance: Option<u32>,
    cross_check: bool,
    max_ratio: Option<f32>,
) -> PyResult<Py<PyArray2<i64>>> {
    if !descriptors1.is_c_contiguous() || !descriptors2.is_c_contiguous() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "descriptor arrays must be C-contiguous",
        ));
    }
    let s1 = descriptors1.shape();
    let s2 = descriptors2.shape();
    if s1[1] != 32 || s2[1] != 32 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "expected (N, 32) uint8 descriptors, got ({}, {}) and ({}, {})",
            s1[0], s1[1], s2[0], s2[1]
        )));
    }

    // Zero-copy reinterpret as 32-byte rows. `c_slice` validates contiguity
    // and uses the array's real element count, so `as_chunks` can never read
    // past the numpy buffer.
    let (d1, _) = c_slice(&descriptors1, "descriptors1")?.as_chunks::<32>();
    let (d2, _) = c_slice(&descriptors2, "descriptors2")?.as_chunks::<32>();

    let matches =
        py.detach(|| match_descriptors::<32>(d1, d2, max_distance, cross_check, max_ratio));

    let k = matches.len();
    let out = PyArray::<i64, _>::zeros(py, [k, 2], false);
    {
        // SAFETY: `out` is a freshly allocated, zero-initialised, C-contiguous
        // (k, 2) i64 array that no other code references yet.
        let slice = unsafe { out.as_slice_mut() }.map_err(to_value_err)?;
        for (dst, (q, t)) in slice.chunks_exact_mut(2).zip(matches.iter()) {
            dst[0] = *q as i64;
            dst[1] = *t as i64;
        }
    }
    Ok(out.unbind())
}
