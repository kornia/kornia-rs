use crate::image::{ToPyImage, ToPyImageF32, ToPyImageU16};
use kornia_image::{
    allocator::CpuAllocator,
    color_spaces::{Gray16, Gray8, Grayf32, Rgb16, Rgb8, Rgba16, Rgba8, Rgbf32},
    PixelFormat,
};
use kornia_io::jpeg as jpeg_io;
use kornia_io::png as png_io;
use kornia_io::tiff as tiff_io;
use pyo3::prelude::*;
use std::fs;
use std::path::Path;

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
