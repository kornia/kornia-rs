//! Python bindings for camera projection APIs in `kornia_3d::camera`.

use crate::pyutils::{c_slice, rows_to_numpy};
use kornia_3d::camera::{CameraModel, FisheyeCamera, PinholeCamera};
use kornia_algebra::{Vec2F64, Vec3F64};
use numpy::{PyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

fn project_rows<'py, C: CameraModel>(
    camera: &C,
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
        if let Some(pixel) = camera.project(&p_cam) {
            out.push(pixel.x);
            out.push(pixel.y);
        } else {
            out.push(f64::NAN);
            out.push(f64::NAN);
        }
    }
    rows_to_numpy(py, out, 2)
}

fn unproject_rows<'py, C: CameraModel>(
    camera: &C,
    py: Python<'py>,
    pixels: Bound<'py, PyArray2<f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = pixels.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected (N, 2) float64 pixels, got shape {:?}",
            shape
        )));
    }
    let raw = c_slice(&pixels, "pixels")?;
    let mut out = Vec::with_capacity(shape[0] * 3);
    for pixel in raw.chunks_exact(2) {
        let pixel = Vec2F64::new(pixel[0], pixel[1]);
        if let Some(bearing) = camera.unproject(&pixel) {
            out.push(bearing.x);
            out.push(bearing.y);
            out.push(bearing.z);
        } else {
            out.push(f64::NAN);
            out.push(f64::NAN);
            out.push(f64::NAN);
        }
    }
    rows_to_numpy(py, out, 3)
}

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
        project_rows(&self.inner, py, points_cam)
    }

    /// Unproject distorted image pixels to unit camera-frame bearing rays.
    ///
    /// Args:
    ///     pixels: `(N, 2)` float64 array of distorted image pixels.
    ///
    /// Returns:
    ///     `(N, 3)` float64 array of unit bearing vectors.
    ///     Rows whose pixels cannot be unprojected are returned as `[nan, nan, nan]`.
    ///
    /// Raises:
    ///     ValueError: if `pixels` is not C-contiguous, aligned, or shape `(N, 2)`.
    fn unproject<'py>(
        &self,
        py: Python<'py>,
        pixels: Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        unproject_rows(&self.inner, py, pixels)
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

/// Python wrapper around [`FisheyeCamera`].
#[pyclass(
    name = "FisheyeCamera",
    module = "kornia_rs.k3d",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFisheyeCamera {
    inner: FisheyeCamera,
}

#[pymethods]
impl PyFisheyeCamera {
    /// Create a Kannala-Brandt fisheye camera.
    #[new]
    #[pyo3(signature = (fx, fy, cx, cy, k1=0.0, k2=0.0, k3=0.0, k4=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(fx: f64, fy: f64, cx: f64, cy: f64, k1: f64, k2: f64, k3: f64, k4: f64) -> Self {
        Self {
            inner: FisheyeCamera {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
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
        project_rows(&self.inner, py, points_cam)
    }

    /// Unproject distorted image pixels to unit camera-frame bearing rays.
    ///
    /// Args:
    ///     pixels: `(N, 2)` float64 array of distorted image pixels.
    ///
    /// Returns:
    ///     `(N, 3)` float64 array of unit bearing vectors.
    ///     Rows whose pixels cannot be unprojected are returned as `[nan, nan, nan]`.
    ///
    /// Raises:
    ///     ValueError: if `pixels` is not C-contiguous, aligned, or shape `(N, 2)`.
    fn unproject<'py>(
        &self,
        py: Python<'py>,
        pixels: Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        unproject_rows(&self.inner, py, pixels)
    }

    fn __repr__(&self) -> String {
        format!(
            "FisheyeCamera(fx={}, fy={}, cx={}, cy={}, k1={}, k2={}, k3={}, k4={})",
            self.inner.fx,
            self.inner.fy,
            self.inner.cx,
            self.inner.cy,
            self.inner.k1,
            self.inner.k2,
            self.inner.k3,
            self.inner.k4,
        )
    }
}
