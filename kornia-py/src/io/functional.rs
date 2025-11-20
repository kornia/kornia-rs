use crate::image::{PyImage, ToPyImage, ToPyImageF32, ToPyImageU16};
use crate::io::jpeg as J_py;
use kornia_io::functional as F;
use kornia_io::png as P_IO;
use kornia_io::tiff as T_IO;
use kornia_io::jpeg as J_IO;
use kornia_image::{allocator::CpuAllocator, color_spaces::{Gray8, Gray16, Grayf32, Rgb8, Rgb16, Rgbf32}, ImagePixelFormat};
use pyo3::prelude::*;
use std::path::Path;
use std::fs;

#[pyfunction]
#[pyo3(warn(message = "read_image_any is deprecated, use read_image instead", category = pyo3::exceptions::PyDeprecationWarning))]
pub fn read_image_any(file_path: &str) -> PyResult<PyImage> {
    let image = F::read_image_any_rgb8(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileExistsError, _>(e.to_string()))?;
    let pyimage = image.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;
    Ok(pyimage)
}
#[pyfunction]
pub fn read_image(file_path: Bound<'_, PyAny>) -> PyResult<PyObject> {
    let file_path_str = if let Ok(s) = file_path.extract::<&str>() {
        s.to_string()
    } else if let Ok(s) = file_path.extract::<String>() {
        s
    } else if let Ok(fspath) = file_path.call_method0("__fspath__") {
        fspath.extract::<String>().or_else(|_| {
            fspath.extract::<&str>().map(|s| s.to_string())
        })?
    } else if let Ok(str_repr) = file_path.call_method0("__str__") {
        str_repr.extract::<String>().or_else(|_| {
            str_repr.extract::<&str>().map(|s| s.to_string())
        })?
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "file_path must be a string or pathlib.Path object"
        ));
    };

    let path = Path::new(&file_path_str);
    
    if !path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            format!("File does not exist: {}", file_path_str)
        ));
    }

    let extension = path.extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase())
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Could not determine file extension for: {}", file_path_str)
        ))?;

    match extension.as_str() {
        "png" => read_image_png_dispatcher(&path),
        "tiff" | "tif" => read_image_tiff_dispatcher(&path),
        "jpg" | "jpeg" => read_image_jpeg_dispatcher(&path),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unsupported file format: {}. Supported formats: png, tiff, jpeg", extension)
        )),
    }
}

fn read_image_png_dispatcher(file_path: &Path) -> PyResult<PyObject> {
    let png_data = fs::read(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let layout = P_IO::decode_image_png_info(&png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let channels = layout.channels;
    let pixel_format = layout.pixel_format;

    match (channels, pixel_format) {
        (1, ImagePixelFormat::U8) => {
            let img = P_IO::read_image_png_mono8(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img.to_pyimage()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(
                    format!("failed to convert image: {}", e)
                ))?
                .into())
        }
        (1, ImagePixelFormat::U16) => {
            let img = P_IO::read_image_png_mono16(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img.to_pyimage_u16()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(
                    format!("failed to convert image: {}", e)
                ))?
                .into())
        }
        (3, ImagePixelFormat::U8) => {
            let img = P_IO::read_image_png_rgb8(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img.to_pyimage()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(
                    format!("failed to convert image: {}", e)
                ))?
                .into())
        }
        (3, ImagePixelFormat::U16) => {
            let img = P_IO::read_image_png_rgb16(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img.to_pyimage_u16()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(
                    format!("failed to convert image: {}", e)
                ))?
                .into())
        }
        (4, ImagePixelFormat::U8) => {
            let img = P_IO::read_image_png_rgba8(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img.to_pyimage()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(
                    format!("failed to convert image: {}", e)
                ))?
                .into())
        }
        (4, ImagePixelFormat::U16) => {
            let img = P_IO::read_image_png_rgba16(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img.to_pyimage_u16()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(
                    format!("failed to convert image: {}", e)
                ))?
                .into())
        }
        (2, _) => {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "PNG GrayscaleAlpha color type is not supported"
            ))
        }
        _ => {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Unsupported PNG format: {} channels, {:?} pixel format",
                    channels, pixel_format
                )
            ))
        }
    }
}

fn read_image_tiff_dispatcher(file_path: &Path) -> PyResult<PyObject> {
    use tiff::decoder::DecodingResult;

    let (result, layout) = T_IO::read_image_tiff_with_metadata(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let image_size = layout.image_size;
    let channels = layout.channels;
    let pixel_format = layout.pixel_format;

    match (result, pixel_format, channels) {
        (DecodingResult::U8(data), ImagePixelFormat::U8, 1) => {
            let img = Gray8::from_size_vec(image_size, data, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img.to_pyimage()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(
                    format!("failed to convert image: {}", e)
                ))?
                .into())
        }
        (DecodingResult::U8(data), ImagePixelFormat::U8, 3) => {
            let img = Rgb8::from_size_vec(image_size, data, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img.to_pyimage()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(
                    format!("failed to convert image: {}", e)
                ))?
                .into())
        }
        (DecodingResult::U16(data), ImagePixelFormat::U16, 1) => {
            let img = Gray16::from_size_vec(image_size, data, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img.to_pyimage_u16()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(
                    format!("failed to convert image: {}", e)
                ))?
                .into())
        }
        (DecodingResult::U16(data), ImagePixelFormat::U16, 3) => {
            let img = Rgb16::from_size_vec(image_size, data, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img.to_pyimage_u16()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(
                    format!("failed to convert image: {}", e)
                ))?
                .into())
        }
        (DecodingResult::F32(data), ImagePixelFormat::F32, 1) => {
            let img = Grayf32::from_size_vec(image_size, data, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img.to_pyimage_f32()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(
                    format!("failed to convert image: {}", e)
                ))?
                .into())
        }
        (DecodingResult::F32(data), ImagePixelFormat::F32, 3) => {
            let img = Rgbf32::from_size_vec(image_size, data, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(img.to_pyimage_f32()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(
                    format!("failed to convert image: {}", e)
                ))?
                .into())
        }
        (_, _, channels) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!(
                "Unsupported TIFF format: {} channels with pixel format {:?}",
                channels, pixel_format
            )
        )),
    }
}

fn read_image_jpeg_dispatcher(file_path: &Path) -> PyResult<PyObject> {
    use std::fs;

    let jpeg_data = fs::read(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let layout = J_IO::decode_image_jpeg_info(&jpeg_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let file_path_str = file_path.to_string_lossy();
    let img = match layout.channels {
        1 => J_py::read_image_jpeg(&file_path_str, "mono")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
        3 => J_py::read_image_jpeg(&file_path_str, "rgb")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported JPEG channel count: {}", layout.channels)
                ));
            }
        };
        Ok(img.into())
}
