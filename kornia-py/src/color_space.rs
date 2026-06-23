//! Python `ColorSpace` enum mirroring `kornia_image::ColorSpace`.

use kornia_image::ColorSpace;
use pyo3::prelude::*;

/// Per-pixel color space tag used by `Image.cvt_color`.
#[pyclass(name = "ColorSpace", eq, eq_int, from_py_object, module = "kornia_rs.image")]
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
