use crate::image::{
    FromPyImage, FromPyImageF32, FromPyImageU16, PyImage, PyImageF32, PyImageU16, ToPyImage,
    ToPyImageF32, ToPyImageU16,
};
use kornia_image::Image;
use kornia_io::tiff as k_tiff;
use pyo3::prelude::*;

/// Reads a TIFF image from a file path into an 8-bit tensor.
///
/// # Arguments
/// * `file_path` (str): The path to the TIFF file to read.
/// * `mode` (str): The color mode to decode the image into.
///   Must be strictly lowercase: `"rgb"` or `"mono"`.
///
/// # Returns
/// * `numpy.ndarray`: The decoded 8-bit image tensor with dtype `uint8` and shape `(H, W, 3)` for `"rgb"` or `(H, W)` for `"mono"`.
///
/// # Exceptions
/// * `ValueError`: If the mode is unsupported (case-sensitive) or the file fails to read.
/// * `Exception`: If the image fails to convert to a Python tensor.
#[pyfunction]
pub fn read_image_tiff_u8(file_path: &str, mode: &str) -> PyResult<PyImage> {
    let result = match mode {
        "rgb" => {
            let img = k_tiff::read_image_tiff_rgb8(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimg = img.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimg
        }
        "mono" => {
            let img = k_tiff::read_image_tiff_mono8(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimg = img.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimg
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                String::from(
                    r#"\
        The following are the supported values of mode:
        1) "rgb" -> 8-bit RGB
        2) "mono" -> 8-bit Monochrome
        "#,
                ),
            ))
        }
    };

    Ok(result)
}

/// Reads a TIFF image from a file path into a 16-bit tensor.
///
/// # Arguments
/// * `file_path` (str): The path to the TIFF file to read.
/// * `mode` (str): The color mode to decode the image into.
///   Must be strictly lowercase: `"rgb"` or `"mono"`.
///
/// # Returns
/// * `numpy.ndarray`: The decoded 16-bit image tensor with dtype `uint16`.
///
/// # Exceptions
/// * `ValueError`: If the mode is unsupported (case-sensitive) or the file fails to read.
/// * `Exception`: If the image fails to convert to a Python tensor.
#[pyfunction]
pub fn read_image_tiff_u16(file_path: &str, mode: &str) -> PyResult<PyImageU16> {
    let result = match mode {
        "rgb" => {
            let img = k_tiff::read_image_tiff_rgb16(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimg = img.to_pyimage_u16().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimg
        }
        "mono" => {
            let img = k_tiff::read_image_tiff_mono16(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimg = img.to_pyimage_u16().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimg
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                String::from(
                    r#"\
        The following are the supported values of mode:
        1) "rgb" -> 16-bit RGB
        2) "mono" -> 16-bit Monochrome
        "#,
                ),
            ))
        }
    };
    Ok(result)
}

/// Reads a TIFF image from a file path into a 32-bit float tensor.
///
/// # Arguments
/// * `file_path` (str): The path to the TIFF file to read.
/// * `mode` (str): The color mode to decode the image into.
///   Must be strictly lowercase: `"rgb"` or `"mono"`.
///
/// # Returns
/// * `numpy.ndarray`: The decoded 32-bit float image tensor with dtype `float32`.
///
/// # Exceptions
/// * `ValueError`: If the mode is unsupported (case-sensitive) or the file fails to read.
/// * `Exception`: If the image fails to convert to a Python tensor.
#[pyfunction]
pub fn read_image_tiff_f32(file_path: &str, mode: &str) -> PyResult<PyImageF32> {
    let result = match mode {
        "mono" => {
            let img = k_tiff::read_image_tiff_mono32f(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimg = img.to_pyimage_f32().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimg
        }
        "rgb" => {
            let img = k_tiff::read_image_tiff_rgb32f(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimg = img.to_pyimage_f32().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimg
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                String::from(
                    r#"\
        The following are the supported values of mode:
        1) "mono" -> 32-bit Floating Point Monochrome
        2) "rgb" -> 32-bit Floating Point RGB
        "#,
                ),
            ))
        }
    };
    Ok(result)
}

/// Writes an 8-bit image tensor to a TIFF file.
///
/// # Arguments
/// * `file_path` (str): The path where the TIFF file will be saved.
/// * `image` (numpy.ndarray): The 8-bit image tensor to write with dtype `uint8`.
/// * `mode` (str): The color mode of the image.
///   Must be strictly lowercase: `"rgb"` or `"mono"`.
///
/// # Exceptions
/// * `ValueError`: If the mode is unsupported (case-sensitive).
/// * `Exception`: If the image format is incompatible or writing fails.
#[pyfunction]
pub fn write_image_tiff_u8(file_path: &str, image: PyImage, mode: &str) -> PyResult<()> {
    match mode {
        "rgb" => {
            let image = Image::<u8, 3, _>::from_pyimage(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
            k_tiff::write_image_tiff_rgb8(file_path, &image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        }
        "mono" => {
            let image = Image::<u8, 1, _>::from_pyimage(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
            k_tiff::write_image_tiff_mono8(file_path, &image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                String::from(
                    r#"\
        The following are the supported values of mode:
        1) "rgb" -> 8-bit RGB
        2) "mono" -> 8-bit Monochrome
        "#,
                ),
            ))
        }
    };

    Ok(())
}

/// Writes a 16-bit image tensor to a TIFF file.
///
/// # Arguments
/// * `file_path` (str): The path where the TIFF file will be saved.
/// * `image` (numpy.ndarray): The 16-bit image tensor to write, with dtype `uint16`.
/// * `mode` (str): The color mode of the image.
///   Must be strictly lowercase: `"rgb"` or `"mono"`.
///
/// # Exceptions
/// * `ValueError`: If the mode is unsupported (case-sensitive).
/// * `Exception`: If the image format is incompatible or writing fails.
#[pyfunction]
pub fn write_image_tiff_u16(file_path: &str, image: PyImageU16, mode: &str) -> PyResult<()> {
    match mode {
        "rgb" => {
            let image = Image::<u16, 3, _>::from_pyimage_u16(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
            k_tiff::write_image_tiff_rgb16(file_path, &image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        }
        "mono" => {
            let image = Image::<u16, 1, _>::from_pyimage_u16(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
            k_tiff::write_image_tiff_mono16(file_path, &image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                String::from(
                    r#"\
        The following are the supported values of mode:
        1) "rgb" -> 16-bit RGB
        2) "mono" -> 16-bit Monochrome
        "#,
                ),
            ))
        }
    };
    Ok(())
}

/// Writes a 32-bit float image tensor to a TIFF file.
///
/// # Arguments
/// * `file_path` (str): The path where the TIFF file will be saved.
/// * `image` (numpy.ndarray): The 32-bit float image tensor to write, with dtype `float32`.
///   For `"mono"` mode, the expected shape is (H, W, 1); for `"rgb"` mode, the expected shape is (H, W, 3).
/// * `mode` (str): The color mode of the image. Must be strictly lowercase: `"rgb"` or `"mono"`.
///
/// # Exceptions
/// * `ValueError`: If the mode is unsupported (case-sensitive).
/// * `Exception`: If the image format is incompatible or writing fails.
#[pyfunction]
pub fn write_image_tiff_f32(file_path: &str, image: PyImageF32, mode: &str) -> PyResult<()> {
    match mode {
        "mono" => {
            let image = Image::<f32, 1, _>::from_pyimage_f32(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
            k_tiff::write_image_tiff_mono32f(file_path, &image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        }
        "rgb" => {
            let image = Image::<f32, 3, _>::from_pyimage_f32(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
            k_tiff::write_image_tiff_rgb32f(file_path, &image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                String::from(
                    r#"\
        The following are the supported values of mode:
        1) "mono" -> 32-bit Floating Point Monochrome
        2) "rgb" -> 32-bit Floating Point RGB
        "#,
                ),
            ))
        }
    }
    Ok(())
}
