use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::{allocator::CpuAllocator, Image};
use kornia_io::jpeg as J;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(
    text_signature = "(file_path, mode, /)",
    doc = "Read a JPEG image from disk.\n\n\
Parameters\n----------\n\
file_path : str\n\
    Path to the JPEG image file.\n\
mode : str\n\
    Image mode. Supported values are:\n\
    - 'rgb'  : 8-bit RGB image (H, W, 3)\n\
    - 'mono' : 8-bit grayscale image (H, W)\n\n\
Returns\n-------\n\
PyImage\n\
    Image loaded as a NumPy-backed PyImage.\n\n\
Raises\n------\n\
ValueError\n\
    If an unsupported mode is provided or the image cannot be read.\n\n\
Examples\n--------\n\
>>> import kornia_rs as K\n\
>>> img = K.read_image_jpeg('dog.jpg', 'rgb')\n\
>>> img.shape\n\
(195, 258, 3)\n"
)]
pub fn read_image_jpeg(file_path: &str, mode: &str) -> PyResult<PyImage> {
    let result = match mode {
        "rgb" => {
            let img = J::read_image_jpeg_rgb8(file_path)
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
            let img = J::read_image_jpeg_mono8(file_path)
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

#[pyfunction]
#[pyo3(
    text_signature = "(file_path, image, mode, quality, /)",
    doc = "Write an image to disk in JPEG format.\n\n\
Parameters\n----------\n\
file_path : str\n\
    Output file path (e.g. 'output.jpg').\n\
image : PyImage\n\
    Image to write. Expected dtype is uint8.\n\
mode : str\n\
    Image mode. Supported values are:\n\
    - 'rgb'  : RGB image with shape (H, W, 3)\n\
    - 'mono' : Grayscale image with shape (H, W)\n\
quality : int\n\
    JPEG quality in range [0, 100]. Higher means better quality.\n\n\
Returns\n-------\n\
None\n\n\
Raises\n------\n\
ValueError\n\
    If an unsupported mode is provided or encoding fails.\n\n\
Examples\n--------\n\
>>> import kornia_rs as K\n\
>>> img = K.read_image_jpeg('dog.jpg', 'rgb')\n\
>>> K.write_image_jpeg('out.jpg', img, 'rgb', quality=90)\n"
)]
pub fn write_image_jpeg(file_path: &str, image: PyImage, mode: &str, quality: u8) -> PyResult<()> {
    match mode {
        "rgb" => {
            let image = Image::<u8, 3, _>::from_pyimage(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            J::write_image_jpeg_rgb8(file_path, &image, quality)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        }
        "mono" => {
            let image = Image::<u8, 1, _>::from_pyimage(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            J::write_image_jpeg_gray8(file_path, &image, quality)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
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
    }
    Ok(())
}

#[pyfunction]
/// Decodes the JPEG Image from raw bytes.
///
/// ```py
/// import kornia_rs as K
///
/// img = K.decode_image_jpeg(bytes(img_data))
/// ```
pub fn decode_image_jpeg(src: &[u8]) -> PyResult<PyImage> {
    let layout = J::decode_image_jpeg_layout(src)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let result = match layout.channels {
        3 => {
            let mut output_image =
                kornia_image::color_spaces::Rgb8::from_size_val(layout.image_size, 0, CpuAllocator)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            J::decode_image_jpeg_rgb8(src, &mut output_image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let output_pyimage = output_image.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            output_pyimage
        }
        1 => {
            let mut output_image = kornia_image::color_spaces::Gray8::from_size_val(
                layout.image_size,
                0,
                CpuAllocator,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            J::decode_image_jpeg_mono8(src, &mut output_image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let output_pyimage = output_image.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            output_pyimage
        }
        ch => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported number of channels: {}",
                ch
            )))
        }
    };

    Ok(result)
}

#[pyfunction]
/// Encodes an RGB u8 image to JPEG bytes.
///
/// # Arguments
///
/// * `image` - RGB image as numpy array (H, W, 3) with dtype uint8
/// * `quality` - JPEG quality (0-100, where 100 is highest quality)
///
/// # Returns
///
/// bytes containing JPEG-encoded image
///
/// # Example
///
/// ```py
/// import kornia_rs as K
/// import numpy as np
///
/// img = K.read_image_jpeg("dog.jpg", "rgb")
/// jpeg_bytes = K.encode_image_jpeg(img, quality=95)
/// with open("output.jpg", "wb") as f:
///     f.write(jpeg_bytes)
/// ```
pub fn encode_image_jpeg(image: PyImage, quality: u8) -> PyResult<Vec<u8>> {
    let image = Image::<u8, 3, _>::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let mut buffer = Vec::new();
    J::encode_image_jpeg_rgb8(&image, quality, &mut buffer)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(buffer)
}
