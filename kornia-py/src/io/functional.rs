use crate::image::{
    alloc_output_pyarray, alloc_output_pyarray_f32, alloc_output_pyarray_u16, to_pyerr,
};
use kornia_image::{
    color_spaces::{Gray16, Gray8, Grayf32, Rgb16, Rgb8, Rgba16, Rgba8, Rgbf32},
    PixelFormat,
};
use kornia_io::jpeg as jpeg_io;
use kornia_io::png as png_io;
use kornia_io::tiff as tiff_io;
use pyo3::prelude::*;
use std::fs;
use std::path::Path;

/// Deprecated: Reads an image from a file path.
///
/// .. warning::
///    `read_image_any` is deprecated. Please use `kornia_rs.io.read_image` instead.
#[pyfunction(name = "read_image_any")]
pub fn read_image_any_deprecated(
    py: Python<'_>,
    file_path: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    crate::warn_deprecation(
        py,
        "kornia_rs.read_image_any is deprecated. Use kornia_rs.io.read_image instead.",
    )?;

    read_image(py, file_path)
}

/// Reads an image from a file path and returns it as a Python tensor.
///
/// Supported formats and bit depths:
/// - **PNG**: 8-bit, 16-bit (Mono, RGB, RGBA)
/// - **TIFF**: 8-bit, 16-bit, 32-bit float (Mono, RGB)
/// - **JPEG**: 8-bit (Mono, RGB)
#[pyfunction]
pub fn read_image(py: Python<'_>, file_path: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let path_obj = file_path
        .call_method0("__fspath__")
        .unwrap_or_else(|_| file_path.clone());

    let path_os: std::ffi::OsString = path_obj.extract().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "file_path must implement __fspath__ and return a valid path",
        )
    })?;

    let path = Path::new(&path_os);
    let path_display = path_os.to_str().unwrap_or("<non-utf8 path>").to_string();

    if !path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            format!("File does not exist: {}", path_display),
        ));
    }

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase())
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Could not determine file extension for: {}",
                path_display
            ))
        })?;

    match extension.as_str() {
        "png" => read_image_png_dispatcher(py, path),
        "tiff" | "tif" => read_image_tiff_dispatcher(py, path),
        "jpg" | "jpeg" => read_image_jpeg_dispatcher(py, path),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported file format: {}. Supported formats: png, tiff, jpeg",
            extension
        ))),
    }
}

fn read_file_bytes(path: &Path) -> PyResult<Vec<u8>> {
    fs::read(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

/// Internal dispatcher for decoding PNG images.
fn read_image_png_dispatcher(py: Python<'_>, file_path: &Path) -> PyResult<Py<PyAny>> {
    let png_data = read_file_bytes(file_path)?;
    let layout = png_io::decode_image_png_layout(&png_data).map_err(to_pyerr)?;

    match (layout.channels, layout.pixel_format) {
        (1, PixelFormat::U8) => {
            let (dst, out) = unsafe { alloc_output_pyarray::<1>(py, layout.image_size)? };
            let mut wrapped = Gray8(dst);
            png_io::decode_image_png_mono8(&png_data, &mut wrapped).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (1, PixelFormat::U16) => {
            let (dst, out) = unsafe { alloc_output_pyarray_u16::<1>(py, layout.image_size)? };
            let mut wrapped = Gray16(dst);
            png_io::decode_image_png_mono16(&png_data, &mut wrapped).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (3, PixelFormat::U8) => {
            let (dst, out) = unsafe { alloc_output_pyarray::<3>(py, layout.image_size)? };
            let mut wrapped = Rgb8(dst);
            png_io::decode_image_png_rgb8(&png_data, &mut wrapped).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (3, PixelFormat::U16) => {
            let (dst, out) = unsafe { alloc_output_pyarray_u16::<3>(py, layout.image_size)? };
            let mut wrapped = Rgb16(dst);
            png_io::decode_image_png_rgb16(&png_data, &mut wrapped).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (4, PixelFormat::U8) => {
            let (dst, out) = unsafe { alloc_output_pyarray::<4>(py, layout.image_size)? };
            let mut wrapped = Rgba8(dst);
            png_io::decode_image_png_rgba8(&png_data, &mut wrapped).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (4, PixelFormat::U16) => {
            let (dst, out) = unsafe { alloc_output_pyarray_u16::<4>(py, layout.image_size)? };
            let mut wrapped = Rgba16(dst);
            png_io::decode_image_png_rgba16(&png_data, &mut wrapped).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (channels, pixel_format) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported PNG format: {} channels, {:?} pixel format",
            channels, pixel_format
        ))),
    }
}

/// Internal dispatcher for decoding TIFF images.
fn read_image_tiff_dispatcher(py: Python<'_>, file_path: &Path) -> PyResult<Py<PyAny>> {
    let tiff_data = read_file_bytes(file_path)?;
    let layout = tiff_io::decode_image_tiff_layout(&tiff_data).map_err(to_pyerr)?;

    match (layout.channels, layout.pixel_format) {
        (1, PixelFormat::U8) => {
            let (dst, out) = unsafe { alloc_output_pyarray::<1>(py, layout.image_size)? };
            let mut wrapped = Gray8(dst);
            tiff_io::decode_image_tiff_mono8(&tiff_data, &mut wrapped).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (1, PixelFormat::U16) => {
            let (dst, out) = unsafe { alloc_output_pyarray_u16::<1>(py, layout.image_size)? };
            let mut wrapped = Gray16(dst);
            tiff_io::decode_image_tiff_mono16(&tiff_data, &mut wrapped).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (3, PixelFormat::U8) => {
            let (dst, out) = unsafe { alloc_output_pyarray::<3>(py, layout.image_size)? };
            let mut wrapped = Rgb8(dst);
            tiff_io::decode_image_tiff_rgb8(&tiff_data, &mut wrapped).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (3, PixelFormat::U16) => {
            let (dst, out) = unsafe { alloc_output_pyarray_u16::<3>(py, layout.image_size)? };
            let mut wrapped = Rgb16(dst);
            tiff_io::decode_image_tiff_rgb16(&tiff_data, &mut wrapped).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (1, PixelFormat::F32) => {
            let (dst, out) = unsafe { alloc_output_pyarray_f32::<1>(py, layout.image_size)? };
            let mut wrapped = Grayf32(dst);
            tiff_io::decode_image_tiff_mono32f(&tiff_data, &mut wrapped).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (3, PixelFormat::F32) => {
            let (dst, out) = unsafe { alloc_output_pyarray_f32::<3>(py, layout.image_size)? };
            let mut wrapped = Rgbf32(dst);
            tiff_io::decode_image_tiff_rgb32f(&tiff_data, &mut wrapped).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (channels, pixel_format) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported TIFF format: {} channels with pixel format {:?}",
            channels, pixel_format
        ))),
    }
}

/// Internal dispatcher for decoding JPEG images.
fn read_image_jpeg_dispatcher(py: Python<'_>, file_path: &Path) -> PyResult<Py<PyAny>> {
    let jpeg_data = read_file_bytes(file_path)?;
    let layout = jpeg_io::decode_image_jpeg_layout(&jpeg_data).map_err(to_pyerr)?;

    match (layout.channels, layout.pixel_format) {
        (1, PixelFormat::U8) => {
            let (mut dst, out) = unsafe { alloc_output_pyarray::<1>(py, layout.image_size)? };
            jpeg_io::decode_image_jpeg_mono8(&jpeg_data, &mut dst).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (3, PixelFormat::U8) => {
            let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, layout.image_size)? };
            jpeg_io::decode_image_jpeg_rgb8(&jpeg_data, &mut dst).map_err(to_pyerr)?;
            Ok(out.into())
        }
        (channels, pixel_format) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported JPEG format: {} channels with pixel format {:?}",
            channels, pixel_format
        ))),
    }
}
