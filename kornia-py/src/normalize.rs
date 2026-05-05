use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray_f32, numpy_to_f32_image, to_pyerr, PyImage, PyImageF32};
use kornia_image::{allocator::CpuAllocator, Image, ImageError};
use kornia_imgproc::normalize;

#[pyfunction]
pub fn normalize_mean_std(
    py: Python<'_>,
    image: PyImage,
    mean: [f32; 3],
    std: [f32; 3],
) -> PyResult<PyImageF32> {
    let src_f32 = numpy_to_f32_image::<3>(py, &image)?;
    let size = src_f32.size();
    let (mut out_img, out) = unsafe { alloc_output_pyarray_f32::<3>(py, size)? };

    py.detach(|| -> Result<(), ImageError> {
        let mut dst_f32 = Image::from_size_val(size, 0.0f32, CpuAllocator)?;
        normalize::normalize_mean_std(&src_f32, &mut dst_f32, &mean, &std)?;
        out_img
            .as_slice_mut()
            .iter_mut()
            .zip(dst_f32.as_slice().iter())
            .for_each(|(d, &s)| *d = s);
        Ok(())
    })
    .map_err(to_pyerr)?;

    Ok(out)
}
