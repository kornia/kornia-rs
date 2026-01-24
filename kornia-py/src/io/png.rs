use kornia_image::{
    allocator::CpuAllocator,
    color_spaces::{Gray16, Gray8, Rgb16, Rgb8, Rgba16, Rgba8},
    Image, ImageSize,
};
use pyo3::prelude::*;

use crate::image::{FromPyImage, FromPyImageU16, PyImage, PyImageU16, ToPyImage, ToPyImageU16};
use kornia_io::png as P;

#[pyfunction]
pub fn read_image_png_u8(file_path: &str, mode: &str) -> PyResult<PyImage> {
    let result = match mode {
        "rgb" => {
            let img = P::read_image_png_rgb8(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimg = img.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimg
        }
        "rgba" => {
            let img = P::read_image_png_rgba8(file_path)
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
            let img = P::read_image_png_mono8(file_path)
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
        2) "rgba" -> 8-bit RGBA
        3) "mono" -> 8-bit Monochrome
        "#,
                ),
            ))
        }
    };

    Ok(result)
}

#[allow(dead_code)]
#[pyfunction]
pub fn read_image_png_u16(file_path: &str, mode: &str) -> PyResult<PyImageU16> {
    let result = match mode {
        "rgb" => {
            let img = P::read_image_png_rgb16(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimg = img.to_pyimage_u16().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimg
        }
        "rgba" => {
            let img = P::read_image_png_rgba16(file_path)
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
            let img = P::read_image_png_mono16(file_path)
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
        2) "rgba" -> 16-bit RGBA
        3) "mono" -> 16-bit Monochrome
        "#,
                ),
            ))
        }
    };

    Ok(result)
}

#[pyfunction]
pub fn write_image_png_u8(file_path: &str, image: PyImage, mode: &str) -> PyResult<()> {
    match mode {
        "rgb" => {
            let image = Image::<u8, 3, _>::from_pyimage(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
            P::write_image_png_rgb8(file_path, &image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        }
        "rgba" => {
            let image = Image::<u8, 4, _>::from_pyimage(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
            P::write_image_png_rgba8(file_path, &image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        }
        "mono" => {
            let image = Image::<u8, 1, _>::from_pyimage(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
            P::write_image_png_gray8(file_path, &image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                String::from(
                    r#"\
        The following are the supported values of mode:
        1) "rgb" -> 8-bit RGB
        2) "rgba" -> 8-bit RGBA
        3) "mono" -> 8-bit Monochrome
        "#,
                ),
            ))
        }
    };

    Ok(())
}

#[allow(dead_code)]
#[pyfunction]
pub fn write_image_png_u16(file_path: &str, image: PyImageU16, mode: &str) -> PyResult<()> {
    match mode {
        "rgb" => {
            let image = Image::<u16, 3, _>::from_pyimage_u16(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
            P::write_image_png_rgb16(file_path, &image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        }
        "rgba" => {
            let image = Image::<u16, 4, _>::from_pyimage_u16(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
            P::write_image_png_rgba16(file_path, &image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        }
        "mono" => {
            let image = Image::<u16, 1, _>::from_pyimage_u16(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
            P::write_image_png_gray16(file_path, &image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                String::from(
                    r#"\
        The following are the supported values of mode:
        1) "rgb" -> 16-bit RGB
        2) "rgba" -> 16-bit RGBA
        3) "mono" -> 16-bit Monochrome
        "#,
                ),
            ))
        }
    };

    Ok(())
}

/// Decodes the PNG Image from raw bytes.
///
/// The following modes are supported:
/// 1. "rgb8" -> 8-bit RGB
/// 2. "rgba8" -> 8-bit RGBA
/// 3. "mono8" -> 8-bit Monochrome
/// 4. "rgb16" -> 16-bit RGB
/// 5. "rgba16" -> 16-bit RGBA
/// 6. "mono16" -> 16-bit Monochrome
///
/// ```py
/// import kornia_rs as K
///
/// img = K.decode_image_png(bytes(img_data), (32, 32), "rgb8")
/// ```
#[allow(dead_code)]
#[pyfunction]
pub fn decode_image_png_u8(
    src: &[u8],
    image_shape: (usize, usize),
    mode: &str,
) -> PyResult<PyImage> {
    let image_shape = ImageSize {
        width: image_shape.1,
        height: image_shape.0,
    };

    let result = match mode {
        "rgb" => {
            let mut image = Rgb8::from_size_val(image_shape, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            P::decode_image_png_rgb8(src, &mut image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimage = image.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimage
        }
        "rgba" => {
            let mut image = Rgba8::from_size_val(image_shape, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            P::decode_image_png_rgba8(src, &mut image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimage = image.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimage
        }
        "mono" => {
            let mut image = Gray8::from_size_val(image_shape, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            P::decode_image_png_mono8(src, &mut image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimage = image.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimage
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                String::from(
                    r#"\
        The following are the supported values of mode:
        1) "rgb" -> 8-bit RGB
        2) "rgba" -> 8-bit RGBA
        3) "mono" -> 8-bit Monochrome
        "#,
                ),
            ))
        }
    };

    Ok(result)
}

#[allow(dead_code)]
#[pyfunction]
pub fn decode_image_png_u16(
    src: &[u8],
    image_shape: (usize, usize),
    mode: &str,
) -> PyResult<PyImageU16> {
    let image_shape = ImageSize {
        width: image_shape.1,
        height: image_shape.0,
    };

    let result = match mode {
        "rgb" => {
            let mut image = Rgb16::from_size_val(image_shape, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            P::decode_image_png_rgb16(src, &mut image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimage = image.to_pyimage_u16().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimage
        }
        "rgba" => {
            let mut image = Rgba16::from_size_val(image_shape, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            P::decode_image_png_rgba16(src, &mut image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimage = image.to_pyimage_u16().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimage
        }
        "mono" => {
            let mut image = Gray16::from_size_val(image_shape, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            P::decode_image_png_mono16(src, &mut image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimage = image.to_pyimage_u16().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimage
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                String::from(
                    r#"\
        The following are the supported values of mode:
        1) "rgb" -> 16-bit RGB
        2) "rgba" -> 16-bit RGBA
        3) "mono" -> 16-bit Monochrome
        "#,
                ),
            ))
        }
    };

    Ok(result)
}
