use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray, numpy_to_f32_image, to_pyerr, PyImage};
use kornia_image::{allocator::CpuAllocator, Image, ImageError};
use kornia_imgproc::enhance;

#[pyfunction]
#[pyo3(name = "adjust_brightness")]
pub fn adjust_brightness_py(py: Python<'_>, image: PyImage, factor: f32) -> PyResult<PyImage> {
    let src_f32 = numpy_to_f32_image::<3>(py, &image)?;
    let size = src_f32.size();
    let (mut dst_u8, out) = unsafe { alloc_output_pyarray::<3>(py, size)? };

    py.detach(|| -> Result<(), ImageError> {
        let mut normalized: Image<f32, 3, _> = Image::from_size_val(size, 0.0f32, CpuAllocator)?;
        normalized
            .as_slice_mut()
            .iter_mut()
            .zip(src_f32.as_slice().iter())
            .for_each(|(d, &s)| *d = s / 255.0);

        let mut dst_f32 = Image::from_size_val(size, 0.0f32, CpuAllocator)?;
        enhance::adjust_brightness(&normalized, &mut dst_f32, factor, true)?;

        dst_u8
            .as_slice_mut()
            .iter_mut()
            .zip(dst_f32.as_slice().iter())
            .for_each(|(d, &s)| *d = (s * 255.0) as u8);
        Ok(())
    })
    .map_err(to_pyerr)?;

    Ok(out)
}
