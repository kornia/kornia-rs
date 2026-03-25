use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray, numpy_to_f32_image, to_pyerr, PyImage};
use kornia_image::{allocator::CpuAllocator, Image, ImageError};
use kornia_imgproc::enhance;

#[pyfunction]
pub fn add_weighted(
    py: Python<'_>,
    src1: PyImage,
    alpha: f32,
    src2: PyImage,
    beta: f32,
    gamma: f32,
) -> PyResult<PyImage> {
    let image1 = numpy_to_f32_image::<3>(py, &src1)?;
    let image2 = numpy_to_f32_image::<3>(py, &src2)?;
    let size = image1.size();
    let (mut dst_u8, out) = unsafe { alloc_output_pyarray::<3>(py, size)? };

    py.detach(|| -> Result<(), ImageError> {
        let mut dst_f32 = Image::from_size_val(size, 0.0f32, CpuAllocator)?;
        enhance::add_weighted(&image1, alpha, &image2, beta, gamma, &mut dst_f32)?;
        dst_u8.as_slice_mut().iter_mut()
            .zip(dst_f32.as_slice().iter())
            .for_each(|(d, &s)| *d = s as u8);
        Ok(())
    }).map_err(to_pyerr)?;

    Ok(out)
}
