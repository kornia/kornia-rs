use crate::image::{
    FromPyImage, FromPyImageF32, FromPyImageU16, PyImage, PyImageF32, PyImageU16, ToPyImage,
    ToPyImageF32, ToPyImageU16,
};
use kornia_image::Image;
use kornia_io::tiff as k_tiff;
use pyo3::prelude::*;

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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
