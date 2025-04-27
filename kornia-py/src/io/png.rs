use kornia_image::{Image, ImageSize};
use pyo3::prelude::*;

use crate::image::{PyImage, ToPyImage};
use kornia_io::png as P;

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
/// img = K.decode_image_png(bytes(img_data), [32, 32], "rgb8")
/// ```
#[pyfunction]
pub fn decode_image_png(src: &[u8], image_shape: Vec<usize>, mode: &str) -> PyResult<PyImage> {
    if image_shape.len() != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            String::from("Missing width and height of image, pass [width, height]"),
        ));
    }

    let image_shape = ImageSize {
        width: image_shape[1],
        height: image_shape[0],
    };

    let result = match mode {
        "rgb8" => {
            let mut image: Image<u8, 3> = Image::from_size_val(image_shape, 0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            P::decode_image_png_rgb8(src, &mut image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            image.to_pyimage()
        }
        "rgba8" => {
            let mut image: Image<u8, 4> = Image::from_size_val(image_shape, 0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            P::decode_image_png_rgba8(src, &mut image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            image.to_pyimage()
        }
        "mono8" => {
            let mut image: Image<u8, 1> = Image::from_size_val(image_shape, 0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            P::decode_image_png_mono8(src, &mut image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            image.to_pyimage()
        }
        "rgb16" => {
            let image_data = vec![0; image_shape.width * image_shape.height * 3];
            let mut image: Image<u16, 3> = Image::new(image_shape, image_data)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            P::decode_image_png_rgb16(src, &mut image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            image.to_pyimage()
        }
        "rgba16" => {
            let mut image: Image<u16, 4> = Image::from_size_val(image_shape, 0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            P::decode_image_png_rgba16(src, &mut image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            image.to_pyimage()
        }
        "mono16" => {
            let mut image: Image<u16, 1> = Image::from_size_val(image_shape, 0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            P::decode_image_png_mono16(src, &mut image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            image.to_pyimage()
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                String::from(
                    r#"\
        The following are the supported values of mode:
        1) "rgb8" -> 8-bit RGB
        2) "rgba8" -> 8-bit RGBA
        3) "mono8" -> 8-bit Monochrome
        4) "rgb16" -> 16-bit RGB
        5) "rgba16" -> 16-bit RGBA
        6) "mono16" -> 16-bit Monochrome
        "#,
                ),
            ))
        }
    };

    Ok(result)
}
