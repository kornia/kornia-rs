use pyo3::prelude::*;
use rayon::prelude::*;

use crate::image::{alloc_output_pyarray_t, numpy_as_image, PyImage, PyImageF32, UNINIT};

const CHUNK_PIXELS: usize = 1 << 14;

#[pyfunction]
pub fn normalize_mean_std(
    py: Python<'_>,
    image: PyImage,
    mean: [f32; 3],
    std: [f32; 3],
) -> PyResult<PyImageF32> {
    // SAFETY: the view borrows the numpy array, which the caller keeps alive and does not mutate
    // for the duration of this call.
    let src = unsafe { numpy_as_image::<3>(py, &image)? };
    // SAFETY: `dst` aliases the fresh array `out`, which stays alive and is only handed to Python
    // after the kernel has written every element through `dst`.
    let (mut dst, out) = unsafe { alloc_output_pyarray_t::<f32, 3, UNINIT>(py, src.size())? };

    // Bit-identical to `kornia_imgproc::normalize::normalize_mean_std` on an f32 copy of the
    // input: `(x - mean) / std` per element, with no 1/255 scaling.
    py.detach(|| {
        dst.as_slice_mut()
            .par_chunks_mut(3 * CHUNK_PIXELS)
            .zip(src.as_slice().par_chunks(3 * CHUNK_PIXELS))
            .for_each(|(d, s)| {
                for (dp, sp) in d.chunks_exact_mut(3).zip(s.chunks_exact(3)) {
                    for c in 0..3 {
                        dp[c] = (sp[c] as f32 - mean[c]) / std[c];
                    }
                }
            });
    });

    Ok(out)
}
