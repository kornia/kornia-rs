//! Python `ColorSpace` enum mirroring `kornia_image::ColorSpace`.

use kornia_image::ColorSpace;
use pyo3::prelude::*;

/// Per-pixel color space tag used by `Image.cvt_color`.
#[pyclass(
    name = "ColorSpace",
    eq,
    eq_int,
    from_py_object,
    module = "kornia_rs.image"
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyColorSpace {
    Rgb,
    Bgr,
    Gray,
    Rgba,
    Bgra,
    Hsv,
    Hls,
    Lab,
    Luv,
    Xyz,
    LinearRgb,
    YCbCr,
    Yuv,
}

#[pymethods]
impl PyColorSpace {
    /// Map variant to its stable integer discriminant (0-12).
    #[allow(clippy::wrong_self_convention)]
    fn to_int(&self) -> u8 {
        match self {
            PyColorSpace::Rgb => 0,
            PyColorSpace::Bgr => 1,
            PyColorSpace::Gray => 2,
            PyColorSpace::Rgba => 3,
            PyColorSpace::Bgra => 4,
            PyColorSpace::Hsv => 5,
            PyColorSpace::Hls => 6,
            PyColorSpace::Lab => 7,
            PyColorSpace::Luv => 8,
            PyColorSpace::Xyz => 9,
            PyColorSpace::LinearRgb => 10,
            PyColorSpace::YCbCr => 11,
            PyColorSpace::Yuv => 12,
        }
    }

    /// Reconstruct a `ColorSpace` from its integer discriminant. Used by pickle.
    #[staticmethod]
    fn _from_int(v: u8) -> PyResult<PyColorSpace> {
        match v {
            0 => Ok(PyColorSpace::Rgb),
            1 => Ok(PyColorSpace::Bgr),
            2 => Ok(PyColorSpace::Gray),
            3 => Ok(PyColorSpace::Rgba),
            4 => Ok(PyColorSpace::Bgra),
            5 => Ok(PyColorSpace::Hsv),
            6 => Ok(PyColorSpace::Hls),
            7 => Ok(PyColorSpace::Lab),
            8 => Ok(PyColorSpace::Luv),
            9 => Ok(PyColorSpace::Xyz),
            10 => Ok(PyColorSpace::LinearRgb),
            11 => Ok(PyColorSpace::YCbCr),
            12 => Ok(PyColorSpace::Yuv),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ColorSpace._from_int: unknown discriminant {v}"
            ))),
        }
    }

    /// Support `pickle.dumps` / `pickle.loads` round-trips.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (u8,))> {
        let from_int = py.get_type::<PyColorSpace>().getattr("_from_int")?.unbind();
        Ok((from_int, (self.to_int(),)))
    }
}

impl From<PyColorSpace> for ColorSpace {
    fn from(v: PyColorSpace) -> Self {
        match v {
            PyColorSpace::Rgb => ColorSpace::Rgb,
            PyColorSpace::Bgr => ColorSpace::Bgr,
            PyColorSpace::Gray => ColorSpace::Gray,
            PyColorSpace::Rgba => ColorSpace::Rgba,
            PyColorSpace::Bgra => ColorSpace::Bgra,
            PyColorSpace::Hsv => ColorSpace::Hsv,
            PyColorSpace::Hls => ColorSpace::Hls,
            PyColorSpace::Lab => ColorSpace::Lab,
            PyColorSpace::Luv => ColorSpace::Luv,
            PyColorSpace::Xyz => ColorSpace::Xyz,
            PyColorSpace::LinearRgb => ColorSpace::LinearRgb,
            PyColorSpace::YCbCr => ColorSpace::YCbCr,
            PyColorSpace::Yuv => ColorSpace::Yuv,
        }
    }
}

impl From<ColorSpace> for PyColorSpace {
    fn from(v: ColorSpace) -> Self {
        match v {
            ColorSpace::Rgb => PyColorSpace::Rgb,
            ColorSpace::Bgr => PyColorSpace::Bgr,
            ColorSpace::Gray => PyColorSpace::Gray,
            ColorSpace::Rgba => PyColorSpace::Rgba,
            ColorSpace::Bgra => PyColorSpace::Bgra,
            ColorSpace::Hsv => PyColorSpace::Hsv,
            ColorSpace::Hls => PyColorSpace::Hls,
            ColorSpace::Lab => PyColorSpace::Lab,
            ColorSpace::Luv => PyColorSpace::Luv,
            ColorSpace::Xyz => PyColorSpace::Xyz,
            ColorSpace::LinearRgb => PyColorSpace::LinearRgb,
            ColorSpace::YCbCr => PyColorSpace::YCbCr,
            ColorSpace::Yuv => PyColorSpace::Yuv,
        }
    }
}
