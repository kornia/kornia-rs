//! Python bindings for camera projection APIs in `kornia_3d::camera`.

use crate::pyutils::{c_slice, rows_to_numpy, value_err};
use kornia_3d::camera::{Camera as RuntimeCamera, CameraModel, FisheyeCamera, PinholeCamera};
use kornia_algebra::{Vec2F64, Vec3F64};
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

type PyPixels<'py> = Bound<'py, PyArray2<f64>>;
type PyBearings<'py> = Bound<'py, PyArray2<f64>>;
type PyJacobians<'py> = Bound<'py, PyArray3<f64>>;
type PyProjectWithJacobianResult<'py> = PyResult<(PyPixels<'py>, PyJacobians<'py>)>;

fn project_rows<'py, C: CameraModel>(
    camera: &C,
    py: Python<'py>,
    points_cam: Bound<'py, PyArray2<f64>>,
) -> PyResult<PyPixels<'py>> {
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
) -> PyResult<PyBearings<'py>> {
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

fn project_with_jacobian_rows<'py, C: CameraModel>(
    camera: &C,
    py: Python<'py>,
    points_cam: Bound<'py, PyArray2<f64>>,
) -> PyProjectWithJacobianResult<'py> {
    let shape = points_cam.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected (N, 3) float64 points, got shape {:?}",
            shape
        )));
    }
    let raw = c_slice(&points_cam, "points_cam")?;
    let mut pixels = Vec::with_capacity(shape[0] * 2);
    let mut jacobians = Vec::with_capacity(shape[0] * 6);
    for point in raw.chunks_exact(3) {
        let p_cam = Vec3F64::new(point[0], point[1], point[2]);
        if let Some((pixel, rows)) = camera.project_with_jacobian(&p_cam) {
            pixels.push(pixel.x);
            pixels.push(pixel.y);
            for row in rows {
                jacobians.push(row.x);
                jacobians.push(row.y);
                jacobians.push(row.z);
            }
        } else {
            pixels.push(f64::NAN);
            pixels.push(f64::NAN);
            jacobians.extend_from_slice(&[f64::NAN; 6]);
        }
    }
    let pixels = rows_to_numpy(py, pixels, 2)?;
    let jacobians = PyArray1::from_vec(py, jacobians)
        .reshape([shape[0], 2, 3])
        .map_err(value_err)?;
    Ok((pixels, jacobians))
}

/// Python wrapper around [`PinholeCamera`].
#[pyclass(name = "PinholeCamera", module = "kornia_rs.k3d", frozen)]
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
    ) -> PyResult<PyPixels<'py>> {
        project_rows(&self.inner, py, points_cam)
    }

    /// Project camera-frame 3D points to distorted image pixels and Jacobians.
    ///
    /// Args:
    ///     points_cam: `(N, 3)` float64 array of camera-frame points.
    ///
    /// Returns:
    ///     Tuple `(pixels, jacobians)` where:
    ///     - `pixels` is `(N, 2)` float64 distorted image pixels.
    ///     - `jacobians` is `(N, 2, 3)` float64 Jacobians `[du/dx, du/dy, du/dz; dv/dx, dv/dy, dv/dz]`.
    ///     Rows whose points cannot be projected are returned as NaNs in both outputs.
    ///
    /// Raises:
    ///     ValueError: if `points_cam` is not C-contiguous, aligned, or shape `(N, 3)`.
    fn project_with_jacobian<'py>(
        &self,
        py: Python<'py>,
        points_cam: Bound<'py, PyArray2<f64>>,
    ) -> PyProjectWithJacobianResult<'py> {
        project_with_jacobian_rows(&self.inner, py, points_cam)
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
    ) -> PyResult<PyBearings<'py>> {
        unproject_rows(&self.inner, py, pixels)
    }

    /// Undistort raw image pixels to ideal pinhole pixels.
    ///
    /// Args:
    ///     pixels: `(N, 2)` float64 array of distorted image pixels.
    ///
    /// Returns:
    ///     `(N, 2)` float64 array of ideal pinhole pixels.
    ///     Rows whose pixels cannot be undistorted are returned as `[nan, nan]`.
    ///
    /// Raises:
    ///     ValueError: if `pixels` is not C-contiguous, aligned, or shape `(N, 2)`.
    fn undistort<'py>(
        &self,
        py: Python<'py>,
        pixels: Bound<'py, PyArray2<f64>>,
    ) -> PyResult<PyPixels<'py>> {
        let shape = pixels.shape();
        if shape.len() != 2 || shape[1] != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected (N, 2) float64 pixels, got shape {:?}",
                shape
            )));
        }
        let raw = c_slice(&pixels, "pixels")?;
        let mut out = Vec::with_capacity(shape[0] * 2);
        for pixel in raw.chunks_exact(2) {
            if let Some(undistorted) = self.inner.undistort(pixel[0], pixel[1]) {
                out.push(undistorted.x);
                out.push(undistorted.y);
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

/// Python wrapper around [`FisheyeCamera`].
#[pyclass(name = "FisheyeCamera", module = "kornia_rs.k3d", frozen)]
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
    ) -> PyResult<PyPixels<'py>> {
        project_rows(&self.inner, py, points_cam)
    }

    /// Project camera-frame 3D points to distorted image pixels and Jacobians.
    ///
    /// Args:
    ///     points_cam: `(N, 3)` float64 array of camera-frame points.
    ///
    /// Returns:
    ///     Tuple `(pixels, jacobians)` where:
    ///     - `pixels` is `(N, 2)` float64 distorted image pixels.
    ///     - `jacobians` is `(N, 2, 3)` float64 Jacobians `[du/dx, du/dy, du/dz; dv/dx, dv/dy, dv/dz]`.
    ///     Rows whose points cannot be projected are returned as NaNs in both outputs.
    ///
    /// Raises:
    ///     ValueError: if `points_cam` is not C-contiguous, aligned, or shape `(N, 3)`.
    fn project_with_jacobian<'py>(
        &self,
        py: Python<'py>,
        points_cam: Bound<'py, PyArray2<f64>>,
    ) -> PyProjectWithJacobianResult<'py> {
        project_with_jacobian_rows(&self.inner, py, points_cam)
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
    ) -> PyResult<PyBearings<'py>> {
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

/// Python wrapper around the runtime-dispatched [`RuntimeCamera`] enum.
#[pyclass(name = "Camera", module = "kornia_rs.k3d", frozen)]
#[derive(Clone)]
pub struct PyCamera {
    inner: RuntimeCamera,
}

#[pymethods]
impl PyCamera {
    /// Create a runtime-dispatched camera holding a Brown-Conrady pinhole model.
    #[staticmethod]
    #[pyo3(signature = (fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn pinhole(fx: f64, fy: f64, cx: f64, cy: f64, k1: f64, k2: f64, p1: f64, p2: f64) -> Self {
        Self {
            inner: RuntimeCamera::from(PinholeCamera {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                p1,
                p2,
            }),
        }
    }

    /// Create a runtime-dispatched camera holding a Kannala-Brandt fisheye model.
    #[staticmethod]
    #[pyo3(signature = (fx, fy, cx, cy, k1=0.0, k2=0.0, k3=0.0, k4=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn fisheye(fx: f64, fy: f64, cx: f64, cy: f64, k1: f64, k2: f64, k3: f64, k4: f64) -> Self {
        Self {
            inner: RuntimeCamera::from(FisheyeCamera {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
            }),
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
    ) -> PyResult<PyPixels<'py>> {
        project_rows(&self.inner, py, points_cam)
    }

    /// Project camera-frame 3D points to distorted image pixels and Jacobians.
    ///
    /// Args:
    ///     points_cam: `(N, 3)` float64 array of camera-frame points.
    ///
    /// Returns:
    ///     Tuple `(pixels, jacobians)` where:
    ///     - `pixels` is `(N, 2)` float64 distorted image pixels.
    ///     - `jacobians` is `(N, 2, 3)` float64 Jacobians `[du/dx, du/dy, du/dz; dv/dx, dv/dy, dv/dz]`.
    ///     Rows whose points cannot be projected are returned as NaNs in both outputs.
    ///
    /// Raises:
    ///     ValueError: if `points_cam` is not C-contiguous, aligned, or shape `(N, 3)`.
    fn project_with_jacobian<'py>(
        &self,
        py: Python<'py>,
        points_cam: Bound<'py, PyArray2<f64>>,
    ) -> PyProjectWithJacobianResult<'py> {
        project_with_jacobian_rows(&self.inner, py, points_cam)
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
    ) -> PyResult<PyBearings<'py>> {
        unproject_rows(&self.inner, py, pixels)
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            RuntimeCamera::Pinhole(camera) => format!(
                "Camera.pinhole(fx={}, fy={}, cx={}, cy={}, k1={}, k2={}, p1={}, p2={})",
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                camera.k1,
                camera.k2,
                camera.p1,
                camera.p2,
            ),
            RuntimeCamera::Fisheye(camera) => format!(
                "Camera.fisheye(fx={}, fy={}, cx={}, cy={}, k1={}, k2={}, k3={}, k4={})",
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                camera.k1,
                camera.k2,
                camera.k3,
                camera.k4,
            ),
        }
    }
}
