use crate::image::{PyImage, ToPyImage};
use crate::io::{jpeg as J, png as P, tiff as T};
use kornia_io::functional as F;
use pyo3::prelude::*;
use std::path::Path;
use std::fs::File;

#[pyfunction]
pub fn read_image_any(file_path: &str) -> PyResult<PyImage> {
    let image = F::read_image_any_rgb8(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileExistsError, _>(format!("{}", e)))?;
    let pyimage = image.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;
    Ok(pyimage)
}

/// Generic image reader that automatically detects format, bit depth, and channels.
///
/// This function attempts to read an image file and automatically determine:
/// - File format (PNG, TIFF, JPEG)
/// - Bit depth (u8, u16, f32)
/// - Channel count (mono, rgb, rgba)
///
/// Returns a numpy array with the appropriate dtype (uint8, uint16, or float32).
///
/// # Arguments
///
/// * `file_path` - The path to the image file.
///
/// # Returns
///
/// A numpy array (PyObject) that can be uint8, uint16, or float32 depending on the image.
///
/// # Example
///
/// ```python
/// import kornia_rs as K
/// import numpy as np
///
/// # Works with u8, u16, f32 images
/// img = K.read_image("path/to/image.png")
/// assert isinstance(img, np.ndarray)
/// ```
#[pyfunction]
pub fn read_image(file_path: &str) -> PyResult<PyObject> {
    let path = Path::new(file_path);
    
    // Verify file exists
    if !path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            format!("File does not exist: {}", file_path)
        ));
    }

    // Get file extension
    let extension = path.extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase())
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Could not determine file extension for: {}", file_path)
        ))?;

    // Route to appropriate reader based on format
    match extension.as_str() {
        "png" => read_image_png_auto(file_path),
        "tiff" | "tif" => read_image_tiff_auto(file_path),
        "jpg" | "jpeg" => read_image_jpeg_auto(file_path),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unsupported file format: {}. Supported formats: png, tiff, jpeg", extension)
        )),
    }
}

fn read_image_png_auto(file_path: &str) -> PyResult<PyObject> {
    use png::{Decoder, BitDepth, ColorType};

    // Read PNG header to detect properties
    let file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
    
    let decoder = Decoder::new(file);
    let reader = decoder.read_info()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to read PNG header: {}", e)
        ))?;

    let info = reader.info();
    let bit_depth = info.bit_depth;
    let color_type = info.color_type;

    // Determine mode based on color type
    let mode = match color_type {
        ColorType::Grayscale => "mono",
        ColorType::Rgb => "rgb",
        ColorType::Rgba => "rgba",
        ColorType::GrayscaleAlpha => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "PNG GrayscaleAlpha color type is not supported"
            ));
        }
        ColorType::Indexed => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "PNG Indexed color type is not supported"
            ));
        }
    };

    // Route to appropriate reader based on bit depth
    match bit_depth {
        BitDepth::Eight => {
            let img = match mode {
                "mono" => P::read_image_png_u8(file_path, "mono")
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?,
                "rgb" => P::read_image_png_u8(file_path, "rgb")
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?,
                "rgba" => P::read_image_png_u8(file_path, "rgba")
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?,
                _ => unreachable!(),
            };
            Ok(img.into())
        }
        BitDepth::Sixteen => {
            let img = match mode {
                "mono" => P::read_image_png_u16(file_path, "mono")
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?,
                "rgb" => P::read_image_png_u16(file_path, "rgb")
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?,
                "rgba" => P::read_image_png_u16(file_path, "rgba")
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?,
                _ => unreachable!(),
            };
            Ok(img.into())
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unsupported PNG bit depth: {:?}", bit_depth)
        )),
    }
}

fn read_image_tiff_auto(file_path: &str) -> PyResult<PyObject> {
    use tiff::decoder::{Decoder, DecodingResult};

    // Read TIFF to detect type
    let file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
    
    let mut decoder = Decoder::new(file)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to read TIFF: {}", e)
        ))?;

    // Get dimensions
    let _ = decoder.dimensions()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to get TIFF dimensions: {}", e)
        ))?;

    // Read the image to detect type
    let result = decoder.read_image()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to read TIFF image: {}", e)
        ))?;

    // Determine mode and type from decoding result
    match result {
        DecodingResult::U8(_) => {
            // Try mono first (1 channel), then rgb (3 channels)
            match T::read_image_tiff_u8(file_path, "mono") {
                Ok(img) => Ok(img.into()),
                Err(_) => {
                    // Try RGB
                    let img = T::read_image_tiff_u8(file_path, "rgb")
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
                    Ok(img.into())
                }
            }
        }
        DecodingResult::U16(_) => {
            // Try mono first, then rgb
            match T::read_image_tiff_u16(file_path, "mono") {
                Ok(img) => Ok(img.into()),
                Err(_) => {
                    // Try RGB
                    let img = T::read_image_tiff_u16(file_path, "rgb")
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
                    Ok(img.into())
                }
            }
        }
        DecodingResult::F32(_) => {
            // Try mono first, then rgb
            match T::read_image_tiff_f32(file_path, "mono") {
                Ok(img) => Ok(img.into()),
                Err(_) => {
                    // Try RGB
                    let img = T::read_image_tiff_f32(file_path, "rgb")
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
                    Ok(img.into())
                }
            }
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unsupported TIFF decoding result type"
        )),
    }
}

fn read_image_jpeg_auto(file_path: &str) -> PyResult<PyObject> {
    use std::fs;

    // Read JPEG file and detect channels
    let jpeg_data = fs::read(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

    let (_, num_channels) = J::decode_image_jpeg_info(&jpeg_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

    // JPEG only supports u8, determine mode from channels
    let img = match num_channels {
        1 => J::read_image_jpeg(file_path, "mono")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?,
        3 => J::read_image_jpeg(file_path, "rgb")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported JPEG channel count: {}", num_channels)
            ));
        }
    };
    Ok(img.into())
}
