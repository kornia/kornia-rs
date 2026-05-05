use kornia_image::{
    color_spaces::{Gray16, Gray8, Rgb16, Rgb8, Rgba16, Rgba8},
    ImageSize,
};
use pyo3::prelude::*;

use crate::image::{
    alloc_output_pyarray, alloc_output_pyarray_u16, numpy_as_image, numpy_as_image_u16, to_pyerr,
    PyImage, PyImageU16,
};
use kornia_io::png as P;

fn read_file_bytes(path: &str) -> PyResult<Vec<u8>> {
    std::fs::read(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

fn unsupported_mode_err(modes: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "The following are the supported values of mode:\n{}",
        modes
    ))
}

/// Reads a PNG image from a file path into an 8-bit tensor.
#[pyfunction]
pub fn read_image_png_u8(py: Python<'_>, file_path: &str, mode: &str) -> PyResult<PyImage> {
    let bytes = read_file_bytes(file_path)?;
    let layout = P::decode_image_png_layout(&bytes).map_err(to_pyerr)?;
    decode_image_png_u8_inner(py, &bytes, layout.image_size, mode)
}

/// Reads a PNG image from a file path into a 16-bit tensor.
#[pyfunction]
pub fn read_image_png_u16(py: Python<'_>, file_path: &str, mode: &str) -> PyResult<PyImageU16> {
    let bytes = read_file_bytes(file_path)?;
    let layout = P::decode_image_png_layout(&bytes).map_err(to_pyerr)?;
    decode_image_png_u16_inner(py, &bytes, layout.image_size, mode)
}

/// Writes an 8-bit image tensor to a PNG file.
#[pyfunction]
pub fn write_image_png_u8(
    py: Python<'_>,
    file_path: &str,
    image: PyImage,
    mode: &str,
) -> PyResult<()> {
    match mode {
        "rgb" => {
            let image = unsafe { numpy_as_image::<3>(py, &image)? };
            P::write_image_png_rgb8(file_path, &image).map_err(to_pyerr)?;
        }
        "rgba" => {
            let image = unsafe { numpy_as_image::<4>(py, &image)? };
            P::write_image_png_rgba8(file_path, &image).map_err(to_pyerr)?;
        }
        "mono" => {
            let image = unsafe { numpy_as_image::<1>(py, &image)? };
            P::write_image_png_gray8(file_path, &image).map_err(to_pyerr)?;
        }
        _ => {
            return Err(unsupported_mode_err(
                "  1) \"rgb\"  -> 8-bit RGB\n  2) \"rgba\" -> 8-bit RGBA\n  3) \"mono\" -> 8-bit Monochrome",
            ))
        }
    }
    Ok(())
}

/// Writes a 16-bit image tensor to a PNG file.
#[pyfunction]
pub fn write_image_png_u16(
    py: Python<'_>,
    file_path: &str,
    image: PyImageU16,
    mode: &str,
) -> PyResult<()> {
    match mode {
        "rgb" => {
            let image = unsafe { numpy_as_image_u16::<3>(py, &image)? };
            P::write_image_png_rgb16(file_path, &image).map_err(to_pyerr)?;
        }
        "rgba" => {
            let image = unsafe { numpy_as_image_u16::<4>(py, &image)? };
            P::write_image_png_rgba16(file_path, &image).map_err(to_pyerr)?;
        }
        "mono" => {
            let image = unsafe { numpy_as_image_u16::<1>(py, &image)? };
            P::write_image_png_gray16(file_path, &image).map_err(to_pyerr)?;
        }
        _ => {
            return Err(unsupported_mode_err(
                "  1) \"rgb\"  -> 16-bit RGB\n  2) \"rgba\" -> 16-bit RGBA\n  3) \"mono\" -> 16-bit Monochrome",
            ))
        }
    }
    Ok(())
}

fn decode_image_png_u8_inner(
    py: Python<'_>,
    src: &[u8],
    size: ImageSize,
    mode: &str,
) -> PyResult<PyImage> {
    match mode {
        "rgb" => {
            let (dst, out) = unsafe { alloc_output_pyarray::<3>(py, size)? };
            let mut wrapped = Rgb8(dst);
            P::decode_image_png_rgb8(src, &mut wrapped).map_err(to_pyerr)?;
            Ok(out)
        }
        "rgba" => {
            let (dst, out) = unsafe { alloc_output_pyarray::<4>(py, size)? };
            let mut wrapped = Rgba8(dst);
            P::decode_image_png_rgba8(src, &mut wrapped).map_err(to_pyerr)?;
            Ok(out)
        }
        "mono" => {
            let (dst, out) = unsafe { alloc_output_pyarray::<1>(py, size)? };
            let mut wrapped = Gray8(dst);
            P::decode_image_png_mono8(src, &mut wrapped).map_err(to_pyerr)?;
            Ok(out)
        }
        _ => Err(unsupported_mode_err(
            "  1) \"rgb\"  -> 8-bit RGB\n  2) \"rgba\" -> 8-bit RGBA\n  3) \"mono\" -> 8-bit Monochrome",
        )),
    }
}

fn decode_image_png_u16_inner(
    py: Python<'_>,
    src: &[u8],
    size: ImageSize,
    mode: &str,
) -> PyResult<PyImageU16> {
    match mode {
        "rgb" => {
            let (dst, out) = unsafe { alloc_output_pyarray_u16::<3>(py, size)? };
            let mut wrapped = Rgb16(dst);
            P::decode_image_png_rgb16(src, &mut wrapped).map_err(to_pyerr)?;
            Ok(out)
        }
        "rgba" => {
            let (dst, out) = unsafe { alloc_output_pyarray_u16::<4>(py, size)? };
            let mut wrapped = Rgba16(dst);
            P::decode_image_png_rgba16(src, &mut wrapped).map_err(to_pyerr)?;
            Ok(out)
        }
        "mono" => {
            let (dst, out) = unsafe { alloc_output_pyarray_u16::<1>(py, size)? };
            let mut wrapped = Gray16(dst);
            P::decode_image_png_mono16(src, &mut wrapped).map_err(to_pyerr)?;
            Ok(out)
        }
        _ => Err(unsupported_mode_err(
            "  1) \"rgb\"  -> 16-bit RGB\n  2) \"rgba\" -> 16-bit RGBA\n  3) \"mono\" -> 16-bit Monochrome",
        )),
    }
}

/// Decodes an 8-bit PNG image from raw bytes (caller-supplied dimensions).
#[pyfunction]
pub fn decode_image_png_u8(
    py: Python<'_>,
    src: &[u8],
    image_shape: (usize, usize),
    mode: &str,
) -> PyResult<PyImage> {
    let size = ImageSize {
        width: image_shape.1,
        height: image_shape.0,
    };
    decode_image_png_u8_inner(py, src, size, mode)
}

/// Decodes a 16-bit PNG image from raw bytes (caller-supplied dimensions).
#[pyfunction]
pub fn decode_image_png_u16(
    py: Python<'_>,
    src: &[u8],
    image_shape: (usize, usize),
    mode: &str,
) -> PyResult<PyImageU16> {
    let size = ImageSize {
        width: image_shape.1,
        height: image_shape.0,
    };
    decode_image_png_u16_inner(py, src, size, mode)
}
