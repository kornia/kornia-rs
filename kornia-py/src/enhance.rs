use pyo3::prelude::*;
use rayon::prelude::*;

use crate::image::{alloc_output_pyarray, numpy_as_image, to_pyerr, PyImage};
use kornia_image::ImageError;

const CHUNK: usize = 1 << 16;

#[pyfunction]
pub fn add_weighted(
    py: Python<'_>,
    src1: PyImage,
    alpha: f32,
    src2: PyImage,
    beta: f32,
    gamma: f32,
) -> PyResult<PyImage> {
    // SAFETY: the views borrow the numpy arrays, which the caller keeps alive and does not mutate
    // for the duration of this call.
    let (image1, image2) = unsafe {
        (
            numpy_as_image::<3>(py, &src1)?,
            numpy_as_image::<3>(py, &src2)?,
        )
    };
    if image1.size() != image2.size() {
        return Err(to_pyerr(ImageError::InvalidImageSize(
            image1.cols(),
            image1.rows(),
            image2.cols(),
            image2.rows(),
        )));
    }
    // SAFETY: `dst` aliases the fresh array `out`, which stays alive and is only handed to Python
    // after the kernel has written every element through `dst`.
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, image1.size())? };

    // Bit-identical to `kornia_imgproc::enhance::add_weighted` on f32 copies of the inputs (same
    // expression order) followed by a saturating `as u8`.
    py.detach(|| {
        dst.as_slice_mut()
            .par_chunks_mut(CHUNK)
            .zip(image1.as_slice().par_chunks(CHUNK))
            .zip(image2.as_slice().par_chunks(CHUNK))
            .for_each(|((d, a), b)| {
                for ((d, &a), &b) in d.iter_mut().zip(a).zip(b) {
                    *d = ((a as f32 * alpha) + (b as f32 * beta) + gamma) as u8;
                }
            });
    });

    Ok(out)
}
