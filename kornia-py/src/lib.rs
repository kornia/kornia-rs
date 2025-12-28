mod apriltag;
mod color;
mod enhance;
mod histogram;
mod icp;
mod image;
mod io;
mod pointcloud;
mod resize;
mod warp;

use crate::image::{PyImageLayout, PyImageSize, PyPixelFormat};
use pyo3::prelude::*;

pub fn get_version() -> String {
    let version = env!("CARGO_PKG_VERSION").to_string();
    version.replace("-alpha", "a").replace("-beta", "b")
}

/// Helper to emit a deprecation warning in Python
fn warn_deprecation(py: Python<'_>, message: &str) -> PyResult<()> {
    let category = py.get_type::<pyo3::exceptions::PyDeprecationWarning>();
    pyo3::PyErr::warn(py, &category, message, 0)
}

// --- Root Level Deprecated Wrappers ---
// These functions keep the old API alive while warning the user to migrate.

#[pyfunction]
pub fn rgb_from_gray(py: Python<'_>, image: image::PyImage) -> PyResult<image::PyImage> {
    warn_deprecation(py, "kornia_rs.rgb_from_gray is deprecated, use kornia_rs.imgproc.rgb_from_gray instead")?;
    color::rgb_from_gray(image)
}

#[pyfunction]
pub fn resize(py: Python<'_>, image: image::PyImage, new_size: (usize, usize), interpolation: &str) -> PyResult<image::PyImage> {
    warn_deprecation(py, "kornia_rs.resize is deprecated, use kornia_rs.imgproc.resize instead")?;
    resize::resize(image, new_size, interpolation)
}

#[pyfunction]
pub fn read_image(py: Python<'_>, file_path: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    warn_deprecation(py, "kornia_rs.read_image is deprecated, use kornia_rs.io.read_image instead")?;
    io::functional::read_image(file_path)
}

#[pymodule(gil_used = false)]
pub fn kornia_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", get_version())?;

    // --- Root level (Deprecated) functions ---
    // We add the most common ones back to the root with warnings.
    m.add_function(wrap_pyfunction!(rgb_from_gray, m)?)?;
    m.add_function(wrap_pyfunction!(resize, m)?)?;
    m.add_function(wrap_pyfunction!(read_image, m)?)?;

    // --- 1. IMAGE MODULE ---
    let image_mod = PyModule::new(py, "image")?;
    image_mod.add_class::<image::PyImageSize>()?;
    image_mod.add_class::<image::PyPixelFormat>()?;
    image_mod.add_class::<image::PyImageLayout>()?;
    m.add_submodule(&image_mod)?;

    // --- 2. IO MODULE ---
    let io_mod = PyModule::new(py, "io")?;
    io_mod.add_function(wrap_pyfunction!(io::functional::read_image, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::functional::read_image_any, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::jpeg::read_image_jpeg, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::jpeg::write_image_jpeg, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::png::read_image_png_u8, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::png::write_image_png_u8, &io_mod)?)?;
    io_mod.add_class::<io::jpegturbo::PyImageDecoder>()?;
    io_mod.add_class::<io::jpegturbo::PyImageEncoder>()?;
    m.add_submodule(&io_mod)?;

    // --- 3. IMGPROC MODULE (Color, Enhance, Histogram, Resize, Warp) ---
    let imgproc_mod = PyModule::new(py, "imgproc")?;
    imgproc_mod.add_function(wrap_pyfunction!(color::rgb_from_gray, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(color::rgb_from_rgba, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(color::rgb_from_bgra, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(color::bgr_from_rgb, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(color::gray_from_rgb, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(enhance::add_weighted, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(histogram::compute_histogram, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(resize::resize, &imgproc_mod)?)?;
    // Warp moved here from geometry
    imgproc_mod.add_function(wrap_pyfunction!(warp::warp_affine, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(warp::warp_perspective, &imgproc_mod)?)?;
    m.add_submodule(&imgproc_mod)?;

    // --- 4. 3D MODULE (ICP) ---
    let three_d_mod = PyModule::new(py, "three_d")?;
    three_d_mod.add_function(wrap_pyfunction!(icp::icp_vanilla, &three_d_mod)?)?;
    three_d_mod.add_class::<icp::PyICPConvergenceCriteria>()?;
    three_d_mod.add_class::<icp::PyICPResult>()?;
    m.add_submodule(&three_d_mod)?;

    // --- 5. APRILTAG MODULE ---
    let apriltag_mod = PyModule::new(py, "apriltag")?;
    apriltag_mod.add_class::<apriltag::PyDecodeTagsConfig>()?;
    apriltag_mod.add_class::<apriltag::PyFitQuadConfig>()?;
    apriltag_mod.add_class::<apriltag::PyAprilTagDecoder>()?;
    apriltag_mod.add_class::<apriltag::PyApriltagDetection>()?;
    apriltag_mod.add_class::<apriltag::PyQuad>()?;

    let apriltag_family_mod = PyModule::new(py, "family")?;
    apriltag_family_mod.add_class::<apriltag::family::PyTagFamily>()?;
    apriltag_family_mod.add_class::<apriltag::family::PyTagFamilyKind>()?;
    apriltag_family_mod.add_class::<apriltag::family::PyQuickDecode>()?;
    apriltag_family_mod.add_class::<apriltag::family::PySharpeningBuffer>()?;
    apriltag_mod.add_submodule(&apriltag_family_mod)?;

    m.add_submodule(&apriltag_mod)?;

    Ok(())
}