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

use crate::icp::{PyICPConvergenceCriteria, PyICPResult};
use crate::image::{PyImageLayout, PyImageSize, PyPixelFormat};
use crate::io::jpegturbo::{PyImageDecoder, PyImageEncoder};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyDeprecationWarning;
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

/// Helper to emit a deprecation warning in Python
fn warn_deprecation(py: Python<'_>, message: &str) -> PyResult<()> {
    use std::ffi::CString;
    let message_cstr = CString::new(message).map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Deprecation warning message contained a null byte",
        )
    })?;

    pyo3::PyErr::warn(
        py,
        &py.get_type::<PyDeprecationWarning>(),
        message_cstr.as_c_str(),
        2,
    )
}

// Root Level Deprecated Wrappers

// Color
#[pyfunction(name = "rgb_from_gray")]
pub fn rgb_from_gray_deprecated(py: Python<'_>, image: image::PyImage) -> PyResult<image::PyImage> {
    warn_deprecation(
        py,
        "kornia_rs.rgb_from_gray is deprecated. Use kornia_rs.imgproc.rgb_from_gray.",
    )?;
    color::rgb_from_gray(image)
}

#[pyfunction(name = "rgb_from_rgba")]
#[pyo3(signature = (image, background=None))]
pub fn rgb_from_rgba_deprecated(
    py: Python<'_>,
    image: image::PyImage,
    background: Option<[u8; 3]>,
) -> PyResult<image::PyImage> {
    warn_deprecation(
        py,
        "kornia_rs.rgb_from_rgba is deprecated. Use kornia_rs.imgproc.rgb_from_rgba.",
    )?;
    color::rgb_from_rgba(image, background)
}

#[pyfunction(name = "rgb_from_bgra")]
#[pyo3(signature = (image, background=None))]
pub fn rgb_from_bgra_deprecated(
    py: Python<'_>,
    image: image::PyImage,
    background: Option<[u8; 3]>,
) -> PyResult<image::PyImage> {
    warn_deprecation(
        py,
        "kornia_rs.rgb_from_bgra is deprecated. Use kornia_rs.imgproc.rgb_from_bgra.",
    )?;
    color::rgb_from_bgra(image, background)
}

#[pyfunction(name = "bgr_from_rgb")]
pub fn bgr_from_rgb_deprecated(py: Python<'_>, image: image::PyImage) -> PyResult<image::PyImage> {
    warn_deprecation(
        py,
        "kornia_rs.bgr_from_rgb is deprecated. Use kornia_rs.imgproc.bgr_from_rgb.",
    )?;
    color::bgr_from_rgb(image)
}

#[pyfunction(name = "gray_from_rgb")]
pub fn gray_from_rgb_deprecated(py: Python<'_>, image: image::PyImage) -> PyResult<image::PyImage> {
    warn_deprecation(
        py,
        "kornia_rs.gray_from_rgb is deprecated. Use kornia_rs.imgproc.gray_from_rgb.",
    )?;
    color::gray_from_rgb(image)
}

// Enhance / Histogram
#[pyfunction(name = "add_weighted")]
pub fn add_weighted_deprecated(
    py: Python<'_>,
    src1: image::PyImage,
    alpha: f32,
    src2: image::PyImage,
    beta: f32,
    gamma: f32,
) -> PyResult<image::PyImage> {
    warn_deprecation(
        py,
        "kornia_rs.add_weighted is deprecated. Use kornia_rs.imgproc.add_weighted.",
    )?;
    enhance::add_weighted(src1, alpha, src2, beta, gamma)
}

#[pyfunction(name = "compute_histogram")]
pub fn compute_histogram_deprecated(
    py: Python<'_>,
    image: image::PyImage,
    nbins: usize,
) -> PyResult<Vec<usize>> {
    warn_deprecation(
        py,
        "kornia_rs.compute_histogram is deprecated. Use kornia_rs.imgproc.compute_histogram.",
    )?;
    histogram::compute_histogram(image, nbins)
}

// ICP
#[pyfunction(name = "icp_vanilla")]
pub fn icp_vanilla_deprecated(
    py: Python<'_>,
    source: pointcloud::PyPointCloud,
    target: pointcloud::PyPointCloud,
    initial_rot: Py<PyArray2<f64>>,
    initial_trans: Py<PyArray1<f64>>,
    criteria: PyICPConvergenceCriteria,
) -> PyResult<PyICPResult> {
    warn_deprecation(
        py,
        "kornia_rs.icp_vanilla is deprecated. Use kornia_rs.k3d.icp_vanilla.",
    )?;

    icp::icp_vanilla(source, target, initial_rot, initial_trans, criteria)
}

// IO
#[pyfunction(name = "read_image")]
pub fn read_image_deprecated(py: Python<'_>, file_path: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    warn_deprecation(
        py,
        "kornia_rs.read_image is deprecated. Use kornia_rs.io.read_image.",
    )?;
    io::functional::read_image(file_path)
}

// JPEG
#[pyfunction(name = "read_image_jpeg")]
pub fn read_image_jpeg_deprecated(
    py: Python<'_>,
    file_path: &str,
    mode: &str,
) -> PyResult<image::PyImage> {
    warn_deprecation(
        py,
        "kornia_rs.read_image_jpeg is deprecated. Use kornia_rs.io.read_image_jpeg.",
    )?;
    io::jpeg::read_image_jpeg(file_path, mode)
}

#[pyfunction(name = "write_image_jpeg")]
pub fn write_image_jpeg_deprecated(
    py: Python<'_>,
    file_path: &str,
    image: image::PyImage,
    mode: &str,
    quality: u8,
) -> PyResult<()> {
    warn_deprecation(
        py,
        "kornia_rs.write_image_jpeg is deprecated. Use kornia_rs.io.write_image_jpeg.",
    )?;
    io::jpeg::write_image_jpeg(file_path, image, mode, quality)
}

// PNG
#[pyfunction(name = "read_image_png_u8")]
pub fn read_image_png_u8_deprecated(
    py: Python<'_>,
    file_path: &str,
    mode: &str,
) -> PyResult<image::PyImage> {
    warn_deprecation(
        py,
        "kornia_rs.read_image_png_u8 is deprecated. Use kornia_rs.io.read_image_png_u8.",
    )?;
    io::png::read_image_png_u8(file_path, mode)
}

#[pyfunction(name = "write_image_png_u8")]
pub fn write_image_png_u8_deprecated(
    py: Python<'_>,
    file_path: &str,
    image: image::PyImage,
    mode: &str,
) -> PyResult<()> {
    warn_deprecation(
        py,
        "kornia_rs.write_image_png_u8 is deprecated. Use kornia_rs.io.write_image_png_u8.",
    )?;
    io::png::write_image_png_u8(file_path, image, mode)
}

// Resize
#[pyfunction(name = "resize")]
pub fn resize_deprecated(
    py: Python<'_>,
    image: image::PyImage,
    new_size: (usize, usize),
    interpolation: &str,
) -> PyResult<image::PyImage> {
    warn_deprecation(
        py,
        "kornia_rs.resize is deprecated. Use kornia_rs.imgproc.resize.",
    )?;
    resize::resize(image, new_size, interpolation)
}

// Warp
#[pyfunction(name = "warp_affine")]
pub fn warp_affine_deprecated(
    py: Python<'_>,
    image: image::PyImage,
    m: [f32; 6],
    new_size: (usize, usize),
    interpolation: &str,
) -> PyResult<image::PyImage> {
    warn_deprecation(
        py,
        "kornia_rs.warp_affine is deprecated. Use kornia_rs.imgproc.warp_affine.",
    )?;
    warp::warp_affine(image, m, new_size, interpolation)
}

#[pyfunction(name = "warp_perspective")]
pub fn warp_perspective_deprecated(
    py: Python<'_>,
    image: image::PyImage,
    m: [f32; 9],
    new_size: (usize, usize),
    interpolation: &str,
) -> PyResult<image::PyImage> {
    warn_deprecation(
        py,
        "kornia_rs.warp_perspective is deprecated. Use kornia_rs.imgproc.warp_perspective.",
    )?;
    warp::warp_perspective(image, m, new_size, interpolation)
}

// Main Python Module Definition

#[pymodule(gil_used = false)]
pub fn kornia_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", get_version())?;

    // Deprecated root-level functions
    m.add_function(wrap_pyfunction!(rgb_from_gray_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(rgb_from_rgba_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(rgb_from_bgra_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(bgr_from_rgb_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(gray_from_rgb_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(add_weighted_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(compute_histogram_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(icp_vanilla_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(read_image_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(
        io::functional::read_image_any_deprecated,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(read_image_jpeg_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(write_image_jpeg_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(read_image_png_u8_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(write_image_png_u8_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(resize_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(warp_affine_deprecated, m)?)?;
    m.add_function(wrap_pyfunction!(warp_perspective_deprecated, m)?)?;

    // Deprecated root-level classes
    // PyO3 does not support emitting a deprecation warning when a class is imported.
    // These top-level classes are retained for backward compatibility purposes only.
    // Users should import them from `kornia_rs::k3d`
    m.add_class::<PyICPConvergenceCriteria>()?;
    m.add_class::<PyICPResult>()?;

    // Image submodule
    let image_mod = PyModule::new(py, "image")?;
    image_mod.add_class::<PyImageSize>()?;
    image_mod.add_class::<PyPixelFormat>()?;
    image_mod.add_class::<PyImageLayout>()?;
    m.add_submodule(&image_mod)?;

    // IO submodule
    let io_mod = PyModule::new(py, "io")?;

    // NOTE:
    // Some IO helpers (e.g. decode/enc helpers and TIFF internals)
    // are now available only under `kornia_rs.io`.
    // There are no deprecated root-scoped aliases among these functions.
    io_mod.add_function(wrap_pyfunction!(io::functional::read_image, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::png::decode_image_png_u8, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::png::decode_image_png_u16, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::png::read_image_png_u8, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::png::read_image_png_u16, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::png::write_image_png_u8, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::png::write_image_png_u16, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::jpeg::decode_image_jpeg, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::jpeg::read_image_jpeg, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::jpeg::write_image_jpeg, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::jpeg::encode_image_jpeg, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_f32, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_u8, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_u16, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_f32, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_u8, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_u16, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(
        io::jpegturbo::decode_image_jpegturbo,
        &io_mod
    )?)?;
    io_mod.add_function(wrap_pyfunction!(
        io::jpegturbo::read_image_jpegturbo,
        &io_mod
    )?)?;
    io_mod.add_function(wrap_pyfunction!(
        io::jpegturbo::write_image_jpegturbo,
        &io_mod
    )?)?;
    io_mod.add_class::<PyImageDecoder>()?;
    io_mod.add_class::<PyImageEncoder>()?;
    m.add_submodule(&io_mod)?;

    // Imgproc submodule
    let imgproc_mod = PyModule::new(py, "imgproc")?;
    imgproc_mod.add_function(wrap_pyfunction!(color::rgb_from_gray, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(color::rgb_from_rgba, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(color::rgb_from_bgra, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(color::bgr_from_rgb, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(color::gray_from_rgb, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(enhance::add_weighted, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(
        histogram::compute_histogram,
        &imgproc_mod
    )?)?;
    imgproc_mod.add_function(wrap_pyfunction!(resize::resize, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(warp::warp_affine, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(warp::warp_perspective, &imgproc_mod)?)?;
    m.add_submodule(&imgproc_mod)?;

    // K3D submodule
    let k3d_mod = PyModule::new(py, "k3d")?;
    k3d_mod.add_function(wrap_pyfunction!(icp::icp_vanilla, &k3d_mod)?)?;
    k3d_mod.add_class::<PyICPConvergenceCriteria>()?;
    k3d_mod.add_class::<PyICPResult>()?;
    m.add_submodule(&k3d_mod)?;

    // Apriltag submodule
    let apriltag_mod = PyModule::new(py, "apriltag")?;
    apriltag_mod.add_class::<apriltag::PyDecodeTagsConfig>()?;
    apriltag_mod.add_class::<apriltag::PyFitQuadConfig>()?;
    apriltag_mod.add_class::<apriltag::PyAprilTagDecoder>()?;
    apriltag_mod.add_class::<apriltag::PyApriltagDetection>()?;
    apriltag_mod.add_class::<apriltag::PyQuad>()?;
    apriltag_mod.add_class::<apriltag::family::PyTagFamilyKind>()?;

    let apriltag_family_mod = PyModule::new(py, "family")?;
    apriltag_family_mod.add_class::<apriltag::family::PyTagFamily>()?;
    apriltag_family_mod.add_class::<apriltag::family::PyTagFamilyKind>()?;
    apriltag_family_mod.add_class::<apriltag::family::PyQuickDecode>()?;
    apriltag_family_mod.add_class::<apriltag::family::PySharpeningBuffer>()?;
    apriltag_mod.add_submodule(&apriltag_family_mod)?;

    m.add_submodule(&apriltag_mod)?;

    Ok(())
}
