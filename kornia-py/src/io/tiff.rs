use crate::image::{
    alloc_output_pyarray, alloc_output_pyarray_f32, alloc_output_pyarray_u16, numpy_as_image,
    numpy_as_image_f32, numpy_as_image_u16, to_pyerr, PyImage, PyImageF32, PyImageU16,
};
use kornia_image::color_spaces::{Gray16, Gray8, Grayf32, Rgb16, Rgb8, Rgbf32};
use kornia_io::tiff as k_tiff;
use pyo3::prelude::*;

fn read_file_bytes(path: &str) -> PyResult<Vec<u8>> {
    std::fs::read(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

fn unsupported_mode_err(modes: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "The following are the supported values of mode:\n{}",
        modes
    ))
}

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
#[pyfunction]
pub fn read_image_tiff_u8(py: Python<'_>, file_path: &str, mode: &str) -> PyResult<PyImage> {
    let bytes = read_file_bytes(file_path)?;
    let layout = k_tiff::decode_image_tiff_layout(&bytes).map_err(to_pyerr)?;
    match mode {
        "rgb" => {
            let (dst, out) = unsafe { alloc_output_pyarray::<3>(py, layout.image_size)? };
            let mut wrapped = Rgb8(dst);
            k_tiff::decode_image_tiff_rgb8(&bytes, &mut wrapped).map_err(to_pyerr)?;
            Ok(out)
        }
        "mono" => {
            let (dst, out) = unsafe { alloc_output_pyarray::<1>(py, layout.image_size)? };
            let mut wrapped = Gray8(dst);
            k_tiff::decode_image_tiff_mono8(&bytes, &mut wrapped).map_err(to_pyerr)?;
            Ok(out)
        }
        _ => Err(unsupported_mode_err(
            "  1) \"rgb\"  -> 8-bit RGB\n  2) \"mono\" -> 8-bit Monochrome",
        )),
    }
}

/// Reads a TIFF image from a file path into a 16-bit tensor.
#[pyfunction]
pub fn read_image_tiff_u16(py: Python<'_>, file_path: &str, mode: &str) -> PyResult<PyImageU16> {
    let bytes = read_file_bytes(file_path)?;
    let layout = k_tiff::decode_image_tiff_layout(&bytes).map_err(to_pyerr)?;
    match mode {
        "rgb" => {
            let (dst, out) = unsafe { alloc_output_pyarray_u16::<3>(py, layout.image_size)? };
            let mut wrapped = Rgb16(dst);
            k_tiff::decode_image_tiff_rgb16(&bytes, &mut wrapped).map_err(to_pyerr)?;
            Ok(out)
        }
        "mono" => {
            let (dst, out) = unsafe { alloc_output_pyarray_u16::<1>(py, layout.image_size)? };
            let mut wrapped = Gray16(dst);
            k_tiff::decode_image_tiff_mono16(&bytes, &mut wrapped).map_err(to_pyerr)?;
            Ok(out)
        }
        _ => Err(unsupported_mode_err(
            "  1) \"rgb\"  -> 16-bit RGB\n  2) \"mono\" -> 16-bit Monochrome",
        )),
    }
}

/// Reads a TIFF image from a file path into a 32-bit float tensor.
#[pyfunction]
pub fn read_image_tiff_f32(py: Python<'_>, file_path: &str, mode: &str) -> PyResult<PyImageF32> {
    let bytes = read_file_bytes(file_path)?;
    let layout = k_tiff::decode_image_tiff_layout(&bytes).map_err(to_pyerr)?;
    match mode {
        "mono" => {
            let (dst, out) = unsafe { alloc_output_pyarray_f32::<1>(py, layout.image_size)? };
            let mut wrapped = Grayf32(dst);
            k_tiff::decode_image_tiff_mono32f(&bytes, &mut wrapped).map_err(to_pyerr)?;
            Ok(out)
        }
        "rgb" => {
            let (dst, out) = unsafe { alloc_output_pyarray_f32::<3>(py, layout.image_size)? };
            let mut wrapped = Rgbf32(dst);
            k_tiff::decode_image_tiff_rgb32f(&bytes, &mut wrapped).map_err(to_pyerr)?;
            Ok(out)
        }
        _ => Err(unsupported_mode_err(
            "  1) \"mono\" -> 32-bit Floating Point Monochrome\n  2) \"rgb\"  -> 32-bit Floating Point RGB",
        )),
    }
}

/// Writes an 8-bit image tensor to a TIFF file.
#[pyfunction]
pub fn write_image_tiff_u8(
    py: Python<'_>,
    file_path: &str,
    image: PyImage,
    mode: &str,
) -> PyResult<()> {
    match mode {
        "rgb" => {
            let image = unsafe { numpy_as_image::<3>(py, &image)? };
            k_tiff::write_image_tiff_rgb8(file_path, &image).map_err(to_pyerr)?;
        }
        "mono" => {
            let image = unsafe { numpy_as_image::<1>(py, &image)? };
            k_tiff::write_image_tiff_mono8(file_path, &image).map_err(to_pyerr)?;
        }
        _ => {
            return Err(unsupported_mode_err(
                "  1) \"rgb\"  -> 8-bit RGB\n  2) \"mono\" -> 8-bit Monochrome",
            ))
        }
    }
    Ok(())
}

/// Writes a 16-bit image tensor to a TIFF file.
#[pyfunction]
pub fn write_image_tiff_u16(
    py: Python<'_>,
    file_path: &str,
    image: PyImageU16,
    mode: &str,
) -> PyResult<()> {
    match mode {
        "rgb" => {
            let image = unsafe { numpy_as_image_u16::<3>(py, &image)? };
            k_tiff::write_image_tiff_rgb16(file_path, &image).map_err(to_pyerr)?;
        }
        "mono" => {
            let image = unsafe { numpy_as_image_u16::<1>(py, &image)? };
            k_tiff::write_image_tiff_mono16(file_path, &image).map_err(to_pyerr)?;
        }
        _ => {
            return Err(unsupported_mode_err(
                "  1) \"rgb\"  -> 16-bit RGB\n  2) \"mono\" -> 16-bit Monochrome",
            ))
        }
    }
    Ok(())
}

/// Writes a 32-bit float image tensor to a TIFF file.
#[pyfunction]
pub fn write_image_tiff_f32(
    py: Python<'_>,
    file_path: &str,
    image: PyImageF32,
    mode: &str,
) -> PyResult<()> {
    match mode {
        "mono" => {
            let image = unsafe { numpy_as_image_f32::<1>(py, &image)? };
            k_tiff::write_image_tiff_mono32f(file_path, &image).map_err(to_pyerr)?;
        }
        "rgb" => {
            let image = unsafe { numpy_as_image_f32::<3>(py, &image)? };
            k_tiff::write_image_tiff_rgb32f(file_path, &image).map_err(to_pyerr)?;
        }
        _ => {
            return Err(unsupported_mode_err(
                "  1) \"mono\" -> 32-bit Floating Point Monochrome\n  2) \"rgb\"  -> 32-bit Floating Point RGB",
            ))
        }
    }
    Ok(())
}
