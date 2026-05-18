use kornia_image::ImageSize;
use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray_u16, numpy_as_image_u16, to_pyerr, PyImageU16};
use kornia_io::rvl as R;


/// Encodes a single-channel 16-bit depth image to RVL-compressed bytes.
///
/// Args:
///     image: numpy array of shape ``(H, W, 1)`` and dtype ``uint16``.
///
/// Returns:
///     ``bytes`` — compressed data with a 12-byte header (magic + width + height).
///
/// Example::
///
///     import numpy as np
///     import kornia_rs.kornia_rs as kr
///
///     depth = np.random.randint(0, 5000, (720, 1280, 1), dtype=np.uint16)
///     compressed = kr.io.encode_image_rvl(depth)
///     recovered  = kr.io.decode_image_rvl(compressed)
///     assert (recovered == depth).all()
#[pyfunction]
pub fn encode_image_rvl(py: Python<'_>, image: PyImageU16) -> PyResult<Vec<u8>> {
    let img = unsafe { numpy_as_image_u16::<1>(py, &image)? };
    R::encode_image_rvl(&img).map_err(to_pyerr)
}

/// Decodes RVL-compressed bytes back to a single-channel 16-bit depth image.
///
/// Args:
///     src: compressed bytes produced by :func:`encode_image_rvl` or
///          :func:`read_image_rvl`.
///
/// Returns:
///     numpy array of shape ``(H, W, 1)`` and dtype ``uint16``.
#[pyfunction]
pub fn decode_image_rvl(py: Python<'_>, src: &[u8]) -> PyResult<PyImageU16> {
    let img = R::decode_image_rvl(src).map_err(to_pyerr)?;
    let size = ImageSize {
        width: img.width(),
        height: img.height(),
    };
    let (mut dst, out) = unsafe { alloc_output_pyarray_u16::<1>(py, size)? };
    dst.as_slice_mut().copy_from_slice(img.as_slice());
    Ok(out)
}

/// Writes a single-channel 16-bit depth image to an RVL file.
///
/// Args:
///     file_path: destination file path (conventionally ``.rvl``).
///     image:     numpy array of shape ``(H, W, 1)`` and dtype ``uint16``.
#[pyfunction]
pub fn write_image_rvl(py: Python<'_>, file_path: &str, image: PyImageU16) -> PyResult<()> {
    let img = unsafe { numpy_as_image_u16::<1>(py, &image)? };
    R::write_image_rvl(file_path, &img).map_err(to_pyerr)
}

/// Reads an RVL file into a single-channel 16-bit depth image.
///
/// Args:
///     file_path: path to a ``.rvl`` file written by :func:`write_image_rvl`.
///
/// Returns:
///     numpy array of shape ``(H, W, 1)`` and dtype ``uint16``.
#[pyfunction]
pub fn read_image_rvl(py: Python<'_>, file_path: &str) -> PyResult<PyImageU16> {
    let img = R::read_image_rvl(file_path).map_err(to_pyerr)?;
    let size = ImageSize {
        width: img.width(),
        height: img.height(),
    };
    let (mut dst, out) = unsafe { alloc_output_pyarray_u16::<1>(py, size)? };
    dst.as_slice_mut().copy_from_slice(img.as_slice());
    Ok(out)
}
