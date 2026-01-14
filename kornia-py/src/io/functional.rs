use crate::image::{
    FromPyImage, FromPyImageF32, FromPyImageU16, PyImage, PyImageF32, PyImageU16, ToPyImage,
    ToPyImageF32, ToPyImageU16,
};
use kornia_image::{
    allocator::CpuAllocator,
    color_spaces::{Gray16, Gray8, Grayf32, Rgb16, Rgb8, Rgba16, Rgba8, Rgbf32},
    Image, PixelFormat,
};
use kornia_io::jpeg as jpeg_io;
use kornia_io::png as png_io;
use kornia_io::tiff as tiff_io;
use pyo3::prelude::*;
use std::fs;
use std::path::Path;

fn img_err(e: kornia_image::ImageError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

fn io_err(e: kornia_io::error::IoError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
}

#[pyfunction(name = "read_image_any")]
pub fn read_image_any_deprecated(
    py: Python<'_>,
    file_path: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    crate::warn_deprecation(
        py,
        "kornia_rs.read_image_any is deprecated. Use kornia_rs.io.read_image instead.",
    )?;

    read_image(file_path)
}

#[pyfunction(name = "write_image_any")]
pub fn write_image_any_deprecated(
    py: Python<'_>,
    file_path: Bound<'_, PyAny>,
    image: Bound<'_, PyAny>,
    mode: Option<&str>,
    quality: Option<u8>,
) -> PyResult<()> {
    crate::warn_deprecation(
        py,
        "kornia_rs.write_image_any is deprecated. Use kornia_rs.io.write_image instead.",
    )?;

    // If mode is None or "auto", detect from image
    let mode = match mode {
        Some(m) if m != "auto" => m,
        _ => infer_image_mode(&image)?,
    };

    write_image(file_path, image, mode, quality)
}

/// Helper to infer mode automatically
fn infer_image_mode(image: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    // First, try to infer the mode from the image shape if available.
    if let Ok(shape_obj) = image.getattr("shape") {
        if let Ok(shape) = shape_obj.extract::<Vec<usize>>() {
            let channels = match shape.len() {
                // (H, W) -> single-channel image
                2 => 1,
                // (H, W, C) -> channel last
                3 => shape[2],
                _ => 0,
            };

            match channels {
                1 => return Ok("mono"),
                3 => return Ok("rgb"),
                4 => return Ok("rgba"),
                _ => {}
            }
        }
    }

    // Fallback: preserve previous behavior based on known wrapper types.
    if image.extract::<PyImage>().is_ok()
        || image.extract::<PyImageU16>().is_ok()
        || image.extract::<PyImageF32>().is_ok()
    {
        Ok("rgb")
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot infer image mode. Please specify mode explicitly.",
        ))
    }
}

#[pyfunction(name = "decode_image_any")]
#[allow(unused_variables)]
pub fn decode_image_any_deprecated(
    py: Python<'_>,
    file_path: Bound<'_, PyAny>,
    mode: Option<&str>,
) -> PyResult<Py<PyAny>> {
    crate::warn_deprecation(
        py,
        "kornia_rs.decode_image_any is deprecated. Use kornia_rs.io.decode_image instead.",
    )?;

    decode_image(file_path)
}

#[pyfunction]
pub fn read_image(file_path: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Attempt to obtain a path-like object via PEP 519 (`__fspath__`)
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
        "png" => read_image_png_dispatcher(path),
        "tiff" | "tif" => read_image_tiff_dispatcher(path),
        "jpg" | "jpeg" => read_image_jpeg_dispatcher(path),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported file format: {}. Supported formats: png, tiff, jpeg",
            extension
        ))),
    }
}

/// Write an image to a file.
///
/// # Arguments
/// * `file_path` - Path to save the image.
/// * `image` - PyImage / PyImageU16 / PyImageF32 instance.
/// * `mode` - Color mode: "rgb", "rgba", "mono". Default is "rgb".
/// * `quality` - Optional JPEG quality (0-100), only used for JPEGs.
#[pyfunction(signature = (file_path, image, mode="rgb", quality=None))]
pub fn write_image(
    file_path: Bound<'_, PyAny>,
    image: Bound<'_, PyAny>,
    mode: &str,
    quality: Option<u8>,
) -> PyResult<()> {
    // PEP 519 support
    let path_obj = file_path
        .call_method0("__fspath__")
        .unwrap_or_else(|_| file_path.clone());

    let path_os: std::ffi::OsString = path_obj.extract().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "file_path must implement __fspath__ and return a valid path",
        )
    })?;

    let path = Path::new(&path_os);
    let path_display = path_os.to_str().unwrap_or("<non-utf8 path>");

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
        "png" => write_image_png_dispatcher(path, image, mode),
        "tiff" | "tif" => write_image_tiff_dispatcher(path, image, mode),
        "jpg" | "jpeg" => write_image_jpeg_dispatcher(path, image, mode, quality),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported file format: {}. Supported formats: png, tiff, jpeg",
            extension
        ))),
    }
}

/// Decode an image file from disk into a Python object.
///
/// Decoding behavior is determined by the file extension (e.g. `png`, `jpg`, `jpeg`).
/// Despite the name, this function reads the image from a file path and does not
/// decode from in-memory bytes.
///
/// # Parameters
/// * `file_path` - A path-like object implementing `__fspath__` (for example a `str` or
///   `pathlib.Path`) that points to the image file on disk.
///
/// # Returns
/// A Python object containing the decoded image data. In typical usage this will be a
/// Python image representation (for example a NumPy `ndarray`) that can be passed to
/// Kornia or other image-processing functions.
///
/// # Errors
/// * `TypeError` if `file_path` does not implement `__fspath__` or cannot be converted to
///   a valid path.
/// * `FileNotFoundError` if the file does not exist.
/// * `ValueError` if the file extension cannot be determined or is not supported
///   (`png`, `jpg`, `jpeg`, `tiff` are currently supported).
///
/// # Examples
/// ```python
/// # Decode an image from disk
/// img = decode_image("input.png")
///
/// # `img` now contains the decoded image data and can be passed to Kornia APIs.
/// ```
///
/// This function is a convenience alias for [`read_image`] and exists for API
/// compatibility with earlier versions and other Kornia bindings. It forwards
/// all arguments directly to [`read_image`].
#[pyfunction]
pub fn decode_image(file_path: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    read_image(file_path)
}

fn read_image_png_dispatcher(file_path: &Path) -> PyResult<Py<PyAny>> {
    // Read file once
    let png_data = fs::read(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    // Get layout from bytes
    let layout = png_io::decode_image_png_layout(&png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let channels = layout.channels;
    let pixel_format = layout.pixel_format;
    let image_size = layout.image_size;

    // Decode from bytes (not file) to avoid reading twice
    match (channels, pixel_format) {
        (1, PixelFormat::U8) => {
            let mut img = Gray8::from_size_val(image_size, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            png_io::decode_image_png_mono8(&png_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img
                .to_pyimage()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                        "failed to convert image: {}",
                        e
                    ))
                })?
                .into())
        }
        (1, PixelFormat::U16) => {
            let mut img = Gray16::from_size_val(image_size, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            png_io::decode_image_png_mono16(&png_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img
                .to_pyimage_u16()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                        "failed to convert image: {}",
                        e
                    ))
                })?
                .into())
        }
        (3, PixelFormat::U8) => {
            let mut img = Rgb8::from_size_val(image_size, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            png_io::decode_image_png_rgb8(&png_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img
                .to_pyimage()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                        "failed to convert image: {}",
                        e
                    ))
                })?
                .into())
        }
        (3, PixelFormat::U16) => {
            let mut img = Rgb16::from_size_val(image_size, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            png_io::decode_image_png_rgb16(&png_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img
                .to_pyimage_u16()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                        "failed to convert image: {}",
                        e
                    ))
                })?
                .into())
        }
        (4, PixelFormat::U8) => {
            let mut img = Rgba8::from_size_val(image_size, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            png_io::decode_image_png_rgba8(&png_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img
                .to_pyimage()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                        "failed to convert image: {}",
                        e
                    ))
                })?
                .into())
        }
        (4, PixelFormat::U16) => {
            let mut img = Rgba16::from_size_val(image_size, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            png_io::decode_image_png_rgba16(&png_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img
                .to_pyimage_u16()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                        "failed to convert image: {}",
                        e
                    ))
                })?
                .into())
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported PNG format: {} channels, {:?} pixel format",
            channels, pixel_format
        ))),
    }
}

fn write_image_png_dispatcher(
    file_path: &Path,
    image: Bound<'_, PyAny>,
    mode: &str,
) -> PyResult<()> {
    let path = file_path
        .to_str()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid path"))?;

    if let Ok(img) = image.extract::<PyImage>() {
        match mode {
            "rgb" => {
                let img = Image::<u8, 3, _>::from_pyimage(img).map_err(img_err)?;
                png_io::write_image_png_rgb8(path, &img).map_err(io_err)?;
            }
            "rgba" => {
                let img = Image::<u8, 4, _>::from_pyimage(img).map_err(img_err)?;
                png_io::write_image_png_rgba8(path, &img).map_err(io_err)?;
            }
            "mono" => {
                let img = Image::<u8, 1, _>::from_pyimage(img).map_err(img_err)?;
                png_io::write_image_png_gray8(path, &img).map_err(io_err)?;
            }
            _ => return invalid_png_mode_u8(),
        }
        return Ok(());
    }

    if let Ok(img) = image.extract::<PyImageU16>() {
        match mode {
            "rgb" => {
                let img = Image::<u16, 3, _>::from_pyimage_u16(img).map_err(img_err)?;
                png_io::write_image_png_rgb16(path, &img).map_err(io_err)?;
            }
            "rgba" => {
                let img = Image::<u16, 4, _>::from_pyimage_u16(img).map_err(img_err)?;
                png_io::write_image_png_rgba16(path, &img).map_err(io_err)?;
            }
            "mono" => {
                let img = Image::<u16, 1, _>::from_pyimage_u16(img).map_err(img_err)?;
                png_io::write_image_png_gray16(path, &img).map_err(io_err)?;
            }
            _ => return invalid_png_mode_u16(),
        }
        return Ok(());
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "PNG supports PyImage (u8) and PyImageU16 only",
    ))
}

fn read_image_tiff_dispatcher(file_path: &Path) -> PyResult<Py<PyAny>> {
    let tiff_data = fs::read(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let layout = tiff_io::decode_image_tiff_layout(&tiff_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let channels = layout.channels;
    let pixel_format = layout.pixel_format;
    let image_size = layout.image_size;

    match (channels, pixel_format) {
        (1, PixelFormat::U8) => {
            let mut img = Gray8::from_size_val(image_size, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            tiff_io::decode_image_tiff_mono8(&tiff_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img
                .to_pyimage()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                        "failed to convert image: {}",
                        e
                    ))
                })?
                .into())
        }
        (1, PixelFormat::U16) => {
            let mut img = Gray16::from_size_val(image_size, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            tiff_io::decode_image_tiff_mono16(&tiff_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img
                .to_pyimage_u16()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                        "failed to convert image: {}",
                        e
                    ))
                })?
                .into())
        }
        (3, PixelFormat::U8) => {
            let mut img = Rgb8::from_size_val(image_size, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            tiff_io::decode_image_tiff_rgb8(&tiff_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img
                .to_pyimage()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                        "failed to convert image: {}",
                        e
                    ))
                })?
                .into())
        }
        (3, PixelFormat::U16) => {
            let mut img = Rgb16::from_size_val(image_size, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            tiff_io::decode_image_tiff_rgb16(&tiff_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img
                .to_pyimage_u16()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                        "failed to convert image: {}",
                        e
                    ))
                })?
                .into())
        }
        (1, PixelFormat::F32) => {
            let mut img = Grayf32::from_size_val(image_size, 0.0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            tiff_io::decode_image_tiff_mono32f(&tiff_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img
                .to_pyimage_f32()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                        "failed to convert image: {}",
                        e
                    ))
                })?
                .into())
        }
        (3, PixelFormat::F32) => {
            let mut img = Rgbf32::from_size_val(image_size, 0.0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            tiff_io::decode_image_tiff_rgb32f(&tiff_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img
                .to_pyimage_f32()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                        "failed to convert image: {}",
                        e
                    ))
                })?
                .into())
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported TIFF format: {} channels with pixel format {:?}",
            channels, pixel_format
        ))),
    }
}

fn write_image_tiff_dispatcher(
    file_path: &Path,
    image: Bound<'_, PyAny>,
    mode: &str,
) -> PyResult<()> {
    let path = file_path
        .to_str()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid path"))?;

    if let Ok(img) = image.extract::<PyImage>() {
        match mode {
            "rgb" => {
                let img = Image::<u8, 3, _>::from_pyimage(img).map_err(img_err)?;
                tiff_io::write_image_tiff_rgb8(path, &img).map_err(io_err)?;
            }
            "mono" => {
                let img = Image::<u8, 1, _>::from_pyimage(img).map_err(img_err)?;
                tiff_io::write_image_tiff_mono8(path, &img).map_err(io_err)?;
            }
            _ => return invalid_tiff_mode_u8(),
        }
        return Ok(());
    }

    if let Ok(img) = image.extract::<PyImageU16>() {
        match mode {
            "rgb" => {
                let img = Image::<u16, 3, _>::from_pyimage_u16(img).map_err(img_err)?;
                tiff_io::write_image_tiff_rgb16(path, &img).map_err(io_err)?;
            }
            "mono" => {
                let img = Image::<u16, 1, _>::from_pyimage_u16(img).map_err(img_err)?;
                tiff_io::write_image_tiff_mono16(path, &img).map_err(io_err)?;
            }
            _ => return invalid_tiff_mode_u16(),
        }
        return Ok(());
    }

    if let Ok(img) = image.extract::<PyImageF32>() {
        match mode {
            "mono" => {
                let img = Image::<f32, 1, _>::from_pyimage_f32(img).map_err(img_err)?;
                tiff_io::write_image_tiff_mono32f(path, &img).map_err(io_err)?;
            }
            "rgb" => {
                let img = Image::<f32, 3, _>::from_pyimage_f32(img).map_err(img_err)?;
                tiff_io::write_image_tiff_rgb32f(path, &img).map_err(io_err)?;
            }
            _ => return invalid_tiff_mode_f32(),
        }
        return Ok(());
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "TIFF supports PyImage, PyImageU16, PyImageF32",
    ))
}

fn read_image_jpeg_dispatcher(file_path: &Path) -> PyResult<Py<PyAny>> {
    // Read file once
    let jpeg_data = fs::read(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    // Get layout from bytes
    let layout = jpeg_io::decode_image_jpeg_layout(&jpeg_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Decode from bytes (not file) to avoid reading twice
    let py_image = match (layout.channels, layout.pixel_format) {
        (1, PixelFormat::U8) => {
            let mut img = Gray8::from_size_val(layout.image_size, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            jpeg_io::decode_image_jpeg_mono8(&jpeg_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            img.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?
        }
        (3, PixelFormat::U8) => {
            let mut img = Rgb8::from_size_val(layout.image_size, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            jpeg_io::decode_image_jpeg_rgb8(&jpeg_data, &mut img)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            img.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported JPEG format: {} channels with pixel format {:?}",
                layout.channels, layout.pixel_format
            )))
        }
    };

    Ok(py_image.into())
}

fn write_image_jpeg_dispatcher(
    file_path: &Path,
    image: Bound<'_, PyAny>,
    mode: &str,
    quality: Option<u8>,
) -> PyResult<()> {
    let path = file_path
        .to_str()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid path"))?;

    let quality = quality.unwrap_or(95);
    let img = image.extract::<PyImage>()?;

    match mode {
        "rgb" => {
            let img = Image::<u8, 3, _>::from_pyimage(img).map_err(img_err)?;
            jpeg_io::write_image_jpeg_rgb8(path, &img, quality).map_err(io_err)?;
        }
        "mono" => {
            let img = Image::<u8, 1, _>::from_pyimage(img).map_err(img_err)?;
            jpeg_io::write_image_jpeg_gray8(path, &img, quality).map_err(io_err)?;
        }
        _ => return invalid_jpeg_mode(),
    }

    Ok(())
}

fn invalid_png_mode_u8() -> PyResult<()> {
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        r#"Supported PNG u8 modes: "rgb", "rgba", "mono""#,
    ))
}

fn invalid_png_mode_u16() -> PyResult<()> {
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        r#"Supported PNG u16 modes: "rgb", "rgba", "mono""#,
    ))
}

fn invalid_tiff_mode_u8() -> PyResult<()> {
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        r#"Supported TIFF u8 modes: "rgb", "mono""#,
    ))
}

fn invalid_tiff_mode_u16() -> PyResult<()> {
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        r#"Supported TIFF u16 modes: "rgb", "mono""#,
    ))
}

fn invalid_tiff_mode_f32() -> PyResult<()> {
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        r#"Supported TIFF f32 modes: "mono", "rgb""#,
    ))
}

fn invalid_jpeg_mode() -> PyResult<()> {
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        r#"Supported JPEG modes: "rgb", "mono""#,
    ))
}
