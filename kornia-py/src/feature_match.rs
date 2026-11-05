use numpy::{PyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::pyutils::{c_slice, value_err};

use kornia_imgproc::features::{match_descriptors, match_smnn, match_smnn_f32};

type SmnnArrays = (Py<PyArray2<f32>>, Py<PyArray2<i64>>);
type DescriptorRows<'a> = (&'a [[u8; 32]], &'a [[u8; 32]]);

fn descriptor_rows<'a>(
    descriptors1: &'a Bound<'_, PyArray2<u8>>,
    descriptors2: &'a Bound<'_, PyArray2<u8>>,
) -> PyResult<DescriptorRows<'a>> {
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
    let (d1, _) = c_slice(descriptors1, "descriptors1")?.as_chunks::<32>();
    let (d2, _) = c_slice(descriptors2, "descriptors2")?.as_chunks::<32>();
    Ok((d1, d2))
}

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
    let (d1, d2) = descriptor_rows(&descriptors1, &descriptors2)?;

    let matches =
        py.detach(|| match_descriptors::<32>(d1, d2, max_distance, cross_check, max_ratio));

    let k = matches.len();
    let out = PyArray::<i64, _>::zeros(py, [k, 2], false);
    {
        // SAFETY: `out` is a freshly allocated, zero-initialised, C-contiguous
        // (k, 2) i64 array that no other code references yet.
        let slice = unsafe { out.as_slice_mut() }.map_err(value_err)?;
        for (dst, (q, t)) in slice.chunks_exact_mut(2).zip(matches.iter()) {
            dst[0] = *q as i64;
            dst[1] = *t as i64;
        }
    }
    Ok(out.unbind())
}

/// Match mutual nearest neighbors that pass Lowe's ratio test in both directions.
///
/// The contract of `kornia.feature.match_smnn`: keeps pairs with
/// `nearest / second_nearest <= th` both ways. `(N, 32)` uint8 descriptors (ORB, BRIEF)
/// use Hamming distances; `(N, D)` float32 descriptors (SIFT, RootSIFT) use L2.
///
/// Args:
///     descriptors1: `(M, 32)` uint8 or `(M, D)` float32 descriptors.
///     descriptors2: `(N, 32)` uint8 or `(N, D)` float32 descriptors.
///     th: distance ratio threshold.
///
/// Returns:
///     `(dists, idxs)`: `(K, 1)` float32 larger ratio of each match, and `(K, 2)`
///     int64 index pairs, in the order of the rows of `descriptors1`.
#[pyfunction(name = "match_smnn")]
#[pyo3(signature = (descriptors1, descriptors2, th=0.95))]
pub fn match_smnn_py(
    py: Python<'_>,
    descriptors1: Bound<'_, PyAny>,
    descriptors2: Bound<'_, PyAny>,
    th: f32,
) -> PyResult<SmnnArrays> {
    let matches = if let (Ok(a), Ok(b)) = (
        descriptors1.cast::<PyArray2<u8>>(),
        descriptors2.cast::<PyArray2<u8>>(),
    ) {
        let (d1, d2) = descriptor_rows(a, b)?;
        py.detach(|| match_smnn::<32>(d1, d2, th))
    } else if let (Ok(a), Ok(b)) = (
        descriptors1.cast::<PyArray2<f32>>(),
        descriptors2.cast::<PyArray2<f32>>(),
    ) {
        let (dim, other) = (a.shape()[1], b.shape()[1]);
        if dim != other {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "descriptors must have one dimension, got {dim} and {other}"
            )));
        }
        let (d1, d2) = (c_slice(a, "descriptors1")?, c_slice(b, "descriptors2")?);
        py.detach(|| match_smnn_f32(d1, d2, dim, th))
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "expected two (N, 32) uint8 arrays or two (N, D) float32 arrays",
        ));
    };

    let k = matches.len();
    let dists = PyArray::<f32, _>::zeros(py, [k, 1], false);
    let idxs = PyArray::<i64, _>::zeros(py, [k, 2], false);
    {
        // SAFETY: both arrays are freshly allocated, zero-initialised and
        // C-contiguous, and no other code references them yet.
        let (dist_rows, idx_rows) = unsafe { (dists.as_slice_mut(), idxs.as_slice_mut()) };
        let (dist_rows, idx_rows) = (dist_rows.map_err(value_err)?, idx_rows.map_err(value_err)?);
        for (m, (dist, idx)) in matches
            .iter()
            .zip(dist_rows.iter_mut().zip(idx_rows.chunks_exact_mut(2)))
        {
            *dist = m.ratio;
            idx[0] = m.query as i64;
            idx[1] = m.train as i64;
        }
    }
    Ok((dists.unbind(), idxs.unbind()))
}
