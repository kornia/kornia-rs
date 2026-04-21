use numpy::{PyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

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

    let (m, n) = (s1[0], s2[0]);

    // Reinterpret (N, 32) as &[[u8; 32]] in place. PyArray guarantees the buffer
    // is live for the duration of this call — no copy needed.
    let d1: &[[u8; 32]] = unsafe {
        std::slice::from_raw_parts(descriptors1.data() as *const [u8; 32], m)
    };
    let d2: &[[u8; 32]] = unsafe {
        std::slice::from_raw_parts(descriptors2.data() as *const [u8; 32], n)
    };

    let matches = py.detach(|| {
        match_descriptors::<32>(d1, d2, max_distance, cross_check, max_ratio)
    });

    let k = matches.len();
    let out = unsafe {
        let arr = PyArray::<i64, _>::new(py, [k, 2], false);
        let slice = std::slice::from_raw_parts_mut(arr.data(), k * 2);
        for (i, (q, t)) in matches.iter().enumerate() {
            slice[i * 2] = *q as i64;
            slice[i * 2 + 1] = *t as i64;
        }
        arr
    };
    Ok(out.unbind())
}
