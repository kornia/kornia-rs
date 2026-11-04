//! Python bindings for camera projection APIs in `kornia_3d::camera`.

use crate::pyutils::{c_slice, rows_to_numpy};
use kornia_3d::camera::{CameraModel, PinholeCamera};
use kornia_algebra::Vec3F64;
use numpy::{PyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Python wrapper around [`PinholeCamera`].
#[pyclass(
    name = "PinholeCamera",
    module = "kornia_rs.k3d",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyPinholeCamera {
    inner: PinholeCamera,
}

#[pymethods]
impl PyPinholeCamera {
    /// Create a Brown-Conrady pinhole camera.
    #[new]
    #[pyo3(signature = (fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(fx: f64, fy: f64, cx: f64, cy: f64, k1: f64, k2: f64, p1: f64, p2: f64) -> Self {
        Self {
            inner: PinholeCamera {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                p1,
                p2,
            },
        }
    }

    /// Project camera-frame 3D points to distorted image pixels.
    ///
    /// Args:
    ///     points_cam: `(N, 3)` float64 array of camera-frame points.
    ///
    /// Returns:
    ///     `(N, 2)` float64 array of distorted pixels.
    ///     Rows whose points cannot be projected are returned as `[nan, nan]`.
    ///
    /// Raises:
    ///     ValueError: if `points_cam` is not C-contiguous, aligned, or shape `(N, 3)`.
    fn project<'py>(
        &self,
        py: Python<'py>,
        points_cam: Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = points_cam.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected (N, 3) float64 points, got shape {:?}",
                shape
            )));
        }
        let raw = c_slice(&points_cam, "points_cam")?;
        let mut out = Vec::with_capacity(shape[0] * 2);
        for point in raw.chunks_exact(3) {
            let p_cam = Vec3F64::new(point[0], point[1], point[2]);
            if let Some(pixel) = self.inner.project(&p_cam) {
                out.push(pixel.x);
                out.push(pixel.y);
            } else {
                out.push(f64::NAN);
                out.push(f64::NAN);
            }
        }
        rows_to_numpy(py, out, 2)
    }

    fn __repr__(&self) -> String {
        format!(
            "PinholeCamera(fx={}, fy={}, cx={}, cy={}, k1={}, k2={}, p1={}, p2={})",
            self.inner.fx,
            self.inner.fy,
            self.inner.cx,
            self.inner.cy,
            self.inner.k1,
            self.inner.k2,
            self.inner.p1,
            self.inner.p2,
        )
    }
}
