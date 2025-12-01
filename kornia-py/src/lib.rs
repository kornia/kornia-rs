mod color;
mod enhance;
mod histogram;
mod icp;
mod image;
mod io;
mod pointcloud;
mod resize;
mod warp;

use crate::icp::{PyICPConvergenceCriteria, PyICPResult};
use crate::image::{PyImageLayout, PyImageSize, PyPixelFormat};
use crate::io::jpegturbo::{PyImageDecoder, PyImageEncoder};
use pyo3::prelude::*;

pub fn get_version() -> String {
    let version = env!("CARGO_PKG_VERSION").to_string();
    // cargo uses "1.0-alpha1" etc. while python uses "1.0.0a1", this is not full compatibility,
    // but it's good enough for now
    // see https://docs.rs/semver/1.0.9/semver/struct.Version.html#method.parse for rust spec
    // see https://peps.python.org/pep-0440/ for python spec
    // it seems the dot after "alpha/beta" e.g. "-alpha.1" is not necessary, hence why this works
    version.replace("-alpha", "a").replace("-beta", "b")
}

#[pymodule(gil_used = false)]
pub fn kornia_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", get_version())?;
    m.add_function(wrap_pyfunction!(color::rgb_from_gray, m)?)?;
    m.add_function(wrap_pyfunction!(color::rgb_from_rgba, m)?)?;
    m.add_function(wrap_pyfunction!(color::rgb_from_bgra, m)?)?;
    m.add_function(wrap_pyfunction!(color::bgr_from_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(color::gray_from_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(enhance::add_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(histogram::compute_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(icp::icp_vanilla, m)?)?;
    m.add_function(wrap_pyfunction!(io::functional::read_image_any, m)?)?;
    m.add_function(wrap_pyfunction!(io::functional::read_image, m)?)?;
    m.add_function(wrap_pyfunction!(io::png::decode_image_png_u8, m)?)?;
    m.add_function(wrap_pyfunction!(io::png::decode_image_png_u16, m)?)?;
    m.add_function(wrap_pyfunction!(io::png::read_image_png_u8, m)?)?;
    m.add_function(wrap_pyfunction!(io::png::read_image_png_u16, m)?)?;
    m.add_function(wrap_pyfunction!(io::png::write_image_png_u8, m)?)?;
    m.add_function(wrap_pyfunction!(io::png::write_image_png_u16, m)?)?;
    m.add_function(wrap_pyfunction!(io::jpeg::decode_image_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(io::jpeg::read_image_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(io::jpeg::write_image_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_f32, m)?)?;
    m.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_u8, m)?)?;
    m.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_u16, m)?)?;
    m.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_f32, m)?)?;
    m.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_u8, m)?)?;
    m.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_u16, m)?)?;
    m.add_function(wrap_pyfunction!(io::jpegturbo::decode_image_jpegturbo, m)?)?;
    m.add_function(wrap_pyfunction!(io::jpegturbo::read_image_jpegturbo, m)?)?;
    m.add_function(wrap_pyfunction!(io::jpegturbo::write_image_jpegturbo, m)?)?;
    m.add_function(wrap_pyfunction!(resize::resize, m)?)?;
    m.add_function(wrap_pyfunction!(warp::warp_affine, m)?)?;
    m.add_function(wrap_pyfunction!(warp::warp_perspective, m)?)?;
    m.add_class::<PyImageSize>()?;
    m.add_class::<PyPixelFormat>()?;
    m.add_class::<PyImageLayout>()?;
    m.add_class::<PyImageDecoder>()?;
    m.add_class::<PyImageEncoder>()?;
    m.add_class::<PyICPConvergenceCriteria>()?;
    m.add_class::<PyICPResult>()?;
    Ok(())
}
