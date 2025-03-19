use num_traits::{AsPrimitive, NumCast};
use kornia_image::{Image, ImageError};

use rayon::prelude::*;
use crate::filter::{gaussian_blur, kernels, separable_filter};


/// Method to calculate gradient for feature response
#[derive(Default)]
pub enum GradsMode {
    /// Sobel operators
    #[default]
    Sobel,
    /// Finite difference
    Diff,
}


/// Compute the Hessian response of an image.
///
/// The Hessian response is computed as the absolute value of the determinant of the Hessian matrix.
///
/// Args:
///     src: The source image with shape (H, W).
///     dst: The destination image with shape (H, W).
pub fn hessian_response(src: &Image<f32, 1>, dst: &mut Image<f32, 1>) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let src_data = src.as_slice();

    dst.as_slice_mut()
        .par_chunks_exact_mut(src.cols())
        .enumerate()
        .for_each(|(row_idx, row_chunk)| {
            if row_idx == 0 || row_idx == src.rows() - 1 {
                // skip the first and last row
                return;
            }

            let row_offset = row_idx * src.cols();

            row_chunk
                .iter_mut()
                .enumerate()
                .for_each(|(col_idx, dst_pixel)| {
                    if col_idx == 0 || col_idx == src.cols() - 1 {
                        // skip the first and last column
                        return;
                    }

                    let current_idx = row_offset + col_idx;
                    let prev_row_idx = current_idx - src.cols();
                    let next_row_idx = current_idx + src.cols();

                    let v11 = src_data[prev_row_idx - 1];
                    let v12 = src_data[prev_row_idx];
                    let v13 = src_data[prev_row_idx + 1];
                    let v21 = src_data[current_idx - 1];
                    let v22 = src_data[current_idx];
                    let v23 = src_data[current_idx + 1];
                    let v31 = src_data[next_row_idx - 1];
                    let v32 = src_data[next_row_idx];
                    let v33 = src_data[next_row_idx + 1];

                    let dxx = v21 - 2.0 * v22 + v23;
                    let dyy = v12 - 2.0 * v22 + v32;
                    let dxy = 0.25 * (v31 - v11 - v33 + v13);

                    let det = dxx * dyy - dxy * dxy;

                    *dst_pixel = det;
                });
        });

    Ok(())
}


/// A builder object to initialize Harris response
pub struct HarrisResponse {
    k: f32,
    grads_mode: GradsMode,
    sigmas: f32
}

impl HarrisResponse {
    /// Creates a Harris response object with default values
    pub fn new() -> Self {
        Self {
            k: 0.04,
            grads_mode: GradsMode::default(),
            sigmas: 0.0,
        }
    }

    /// Sets the `k` value (usually between 0.04 to 0.06)
    pub fn with_k(self, k: f32 ) -> Self {
        Self { k, ..self }
    }

    /// Sets the gradient mode
    pub fn with_grads_mode(self, grads_mode: GradsMode) -> Self {
        Self { grads_mode, ..self }
    }

    /// Sets the sigma
    pub fn with_sigmas(self, sigmas: f32) -> Self {
        Self { sigmas, ..self }
    }

    /// Computes the harris response of an image.
    ///
    /// The Harris response is computed by the determinant minus the trace squared.
    ///
    /// Args:
    ///     src: The source image with shape (H, W).
    ///     dst: The destination image with shape (H, W).
    pub fn compute(&self, src: &Image<f32, 1>, dst: &mut Image<f32, 1>) -> Result<(), ImageError> {
        if src.size() != dst.size() {
            return Err(ImageError::InvalidImageSize(
                src.cols(),
                src.rows(),
                dst.cols(),
                dst.rows(),
            ));
        }

        let src_data = src.as_slice();
        let col_slice = src.cols()..src_data.len()-src.cols();
        let row_slice = 1..src.rows()-1;
        let mut dx2_data = vec![0.0; src_data.len()];
        let mut dy2_data = vec![0.0; src_data.len()];
        let mut dxy_data = vec![0.0; src_data.len()];
        // let mut dx2_blurred: Image<f32, 1> = Image::from_size_val(src.size(), 0.0f32)?;
        // let mut dy2_blurred: Image<f32, 1> = Image::from_size_val(src.size(), 0.0f32)?;
        // let mut dxy_blurred: Image<f32, 1> = Image::from_size_val(src.size(), 0.0f32)?;

        // let (kernel_x, kernel_y) = kernels::sobel_kernel_1d(3);
        //
        // let mut gx = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
        // separable_filter(src, &mut gx, &kernel_x, &kernel_y)?;
        // let gx_data = gx.as_slice();
        //
        // let mut gy = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
        // separable_filter(src, &mut gy, &kernel_y, &kernel_x)?;
        // let gy_data = gy.as_slice();

        dx2_data.as_mut_slice()[col_slice.clone()].par_chunks_exact_mut(src.cols())//.skip(1).take(src.rows()-2)
            .zip(dy2_data.as_mut_slice()[col_slice.clone()].par_chunks_exact_mut(src.cols()))//.skip(1).take(src.rows()-2))
            .zip(dxy_data.as_mut_slice()[col_slice.clone()].par_chunks_exact_mut(src.cols()))//.skip(1).take(src.rows()-2))
            .enumerate()
            .for_each(|(row_idx, ((dx2_chunk, dy2_chunk), dxy_chunk))| {
                let row_offset = (row_idx+1) * src.cols();

                dx2_chunk[row_slice.clone()].iter_mut()
                    .zip(dy2_chunk[row_slice.clone()].iter_mut())
                    .zip(dxy_chunk[row_slice.clone()].iter_mut())
                    .enumerate()
                    .for_each(|(col_idx, ((dx2_pixel, dy2_pixel), dxy_pixel))| {
                        let current_idx = row_offset + col_idx+1;
                        let prev_row_idx = current_idx - src.cols();
                        let next_row_idx = current_idx + src.cols();

                        let v11 = unsafe { src_data.get_unchecked(prev_row_idx - 1) };
                        let v12 = unsafe { src_data.get_unchecked(prev_row_idx) };
                        let v13 = unsafe { src_data.get_unchecked(prev_row_idx + 1) };
                        let v21 = unsafe { src_data.get_unchecked(current_idx - 1) };
                        let v23 = unsafe { src_data.get_unchecked(current_idx + 1) };
                        let v31 = unsafe { src_data.get_unchecked(next_row_idx - 1) };
                        let v32 = unsafe { src_data.get_unchecked(next_row_idx) };
                        let v33 = unsafe { src_data.get_unchecked(next_row_idx + 1) };

                        // I_x,I_y via 3x3 sobel operator and convolved
                        let dx = (-v33+v31 -2.0*v23+2.0*v21 - v13+v11)*0.125;
                        let dy = (-v33-2.0*v32-v31 + v13+2.0*v12+v11)*0.125;

                        // filter normalization
                        *dx2_pixel = dx*dx;
                        *dy2_pixel = dy*dy;
                        *dxy_pixel = dx*dy;
                    });
            });

        // gaussian_blur(&Image::from_size_slice(src.size(), &dx2_data)?, &mut dx2_blurred,
        //               (7,7), (1.0,1.0))?;
        // gaussian_blur(&Image::from_size_slice(src.size(), &dy2_data)?, &mut dy2_blurred,
        //               (7,7), (1.0,1.0))?;
        // gaussian_blur(&Image::from_size_slice(src.size(), &dxy_data)?, &mut dxy_blurred,
        //               (7,7), (1.0,1.0))?;

        dst.as_slice_mut()[col_slice.clone()].par_chunks_exact_mut(src.cols())
            .enumerate()
            .for_each(|(row_idx, dst_chunk)| {
                let row_offset = (row_idx+1) * src.cols();

                dst_chunk[row_slice.clone()].iter_mut()
                    .enumerate()
                    .for_each(|(col_idx, dst_pixel)| {
                        let current_idx = row_offset + col_idx+1;
                        let prev_row_idx = current_idx - src.cols();
                        let next_row_idx = current_idx + src.cols();

                        let mut m11 = 0.0;
                        let mut m22 = 0.0;
                        let mut m12 = 0.0;

                        let idxs = [
                            prev_row_idx - 1, prev_row_idx, prev_row_idx + 1,
                            current_idx - 1, current_idx, current_idx + 1,
                            next_row_idx - 1, next_row_idx, next_row_idx + 1,
                        ];
                        for idx in idxs {
                            m11 += unsafe { dx2_data.get_unchecked(idx) };
                            m22 += unsafe { dy2_data.get_unchecked(idx) };
                            m12 += unsafe { dxy_data.get_unchecked(idx) };
                        }

                        let det = m11 * m22 - m12 * m12;
                        let trace = m11 + m22;
                        let response = det - self.k * trace * trace;

                        *dst_pixel = f32::max(0.0, response);
                    });
            });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hessian_response() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let src = Image::from_size_slice(
            [5, 5].into(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )?;

        let mut dst = Image::from_size_val([5, 5].into(), 0.0)?;
        hessian_response(&src, &mut dst)?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 4.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );
        Ok(())
    }

    #[test]
    fn test_harris_response() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let src = Image::from_size_slice(
            [9, 9].into(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        )?;

        let mut dst = Image::from_size_val([9, 9].into(), 0.0)?;
        HarrisResponse::new().compute(&src, &mut dst)?;

        // pulled from kornia's example
        #[rustfmt::skip]
        assert_eq!(dst.as_slice(), &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.01953125, 0.14078125, 0.08238281, 0.0, 0.08238281, 0.14078125, 0.01953125, 0.0,
            0.0, 0.14078125, 0.49203125, 0.37144533, 0.0, 0.37144533, 0.49203125, 0.14078125, 0.0,
            0.0, 0.08238281, 0.37144533, 0.32496095, 0.0, 0.32496095, 0.37144533, 0.08238281, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.08238281, 0.37144533, 0.32496095, 0.0, 0.32496095, 0.37144533, 0.08238281, 0.0,
            0.0, 0.14078125, 0.49203125, 0.37144533, 0.0, 0.37144533, 0.49203125, 0.14078125, 0.0,
            0.0, 0.01953125, 0.14078125, 0.08238281, 0.0, 0.08238281, 0.14078125, 0.01953125, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]);

        Ok(())
    }
}
