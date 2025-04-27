use crate::filter::gaussian_blur;
use kornia_image::{Image, ImageError, ImageSize};
use rayon::prelude::*;

/// Method to calculate gradient for feature response
#[derive(Default)]
pub enum GradsMode {
    /// Sobel operators
    #[default]
    Sobel,
    /// Finite difference
    Diff,
}

fn _get_kernel_size(sigma: f32) -> usize {
    let mut ksize = (2.0 * 4.0 * sigma + 1.0) as usize;

    // matches OpenCV, but may cause padding problem for small images
    // PyTorch does not allow to pad more than original size.
    // Therefore there is a hack in forward function
    if ksize % 2 == 0 {
        ksize += 1;
    }

    ksize
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
    image_size: ImageSize,
    k: f32,
    dx2_data: Vec<f32>,
    dy2_data: Vec<f32>,
    dxy_data: Vec<f32>,
}

impl HarrisResponse {
    /// Creates a Harris response object with default values
    pub fn new(image_size: ImageSize) -> Self {
        Self {
            image_size,
            k: 0.04,
            dx2_data: vec![0.0; image_size.width * image_size.height],
            dy2_data: vec![0.0; image_size.width * image_size.height],
            dxy_data: vec![0.0; image_size.width * image_size.height],
        }
    }

    /// Sets the `k` value (usually between 0.04 to 0.06)
    pub fn with_k(self, k: f32) -> Self {
        Self { k, ..self }
    }

    /// Computes the harris response of an image.
    ///
    /// The Harris response is computed by the determinant minus the trace squared.
    ///
    /// Args:
    ///     src: The source image with shape (H, W).
    ///     dst: The destination image with shape (H, W).
    pub fn compute(
        &mut self,
        src: &Image<f32, 1>,
        dst: &mut Image<f32, 1>,
    ) -> Result<(), ImageError> {
        if src.size() != self.image_size {
            return Err(ImageError::InvalidImageSize(
                src.size().width,
                src.size().height,
                self.image_size.width,
                self.image_size.height,
            ));
        }
        if dst.size() != self.image_size {
            return Err(ImageError::InvalidImageSize(
                dst.size().width,
                dst.size().height,
                self.image_size.width,
                self.image_size.height,
            ));
        }

        let src_data = src.as_slice();
        let col_slice = src.cols()..src_data.len() - src.cols();
        let row_slice = 1..src.cols() - 1;

        self.dx2_data
            .as_mut_slice()
            .get_mut(col_slice.clone())
            // SAFETY: we ranges is valid
            .unwrap()
            .par_chunks_exact_mut(src.cols())
            .zip(
                self.dy2_data
                    .as_mut_slice()
                    .get_mut(col_slice.clone())
                    // SAFETY: we ranges is valid
                    .unwrap()
                    .par_chunks_exact_mut(src.cols()),
            )
            .zip(
                self.dxy_data
                    .as_mut_slice()
                    .get_mut(col_slice.clone())
                    // SAFETY: we ranges is valid
                    .unwrap()
                    .par_chunks_exact_mut(src.cols()),
            )
            .enumerate()
            .for_each(|(row_idx, ((dx2_chunk, dy2_chunk), dxy_chunk))| {
                let row_offset = (row_idx + 1) * src.cols();

                dx2_chunk
                    .get_mut(row_slice.clone())
                    // SAFETY: we ranges is valid
                    .unwrap()
                    .iter_mut()
                    .zip(
                        dy2_chunk
                            .get_mut(row_slice.clone())
                            // SAFETY: we ranges is valid
                            .unwrap()
                            .iter_mut(),
                    )
                    .zip(dxy_chunk.get_mut(row_slice.clone()).unwrap().iter_mut())
                    .enumerate()
                    .for_each(|(col_idx, ((dx2_pixel, dy2_pixel), dxy_pixel))| {
                        let current_idx = row_offset + col_idx + 1;
                        let prev_row_idx = current_idx - src.cols();
                        let next_row_idx = current_idx + src.cols();

                        let (v11, v12, v13, v21, v23, v31, v32, v33) = unsafe {
                            // SAFETY: we ranges is valid
                            (
                                src_data.get_unchecked(prev_row_idx - 1),
                                src_data.get_unchecked(prev_row_idx),
                                src_data.get_unchecked(prev_row_idx + 1),
                                src_data.get_unchecked(current_idx - 1),
                                src_data.get_unchecked(current_idx + 1),
                                src_data.get_unchecked(next_row_idx - 1),
                                src_data.get_unchecked(next_row_idx),
                                src_data.get_unchecked(next_row_idx + 1),
                            )
                        };

                        // I_x,I_y via 3x3 sobel operator and convolved
                        let dx = (-v33 + v31 - 2.0 * v23 + 2.0 * v21 - v13 + v11) * 0.125;
                        let dy = (-v33 - 2.0 * v32 - v31 + v13 + 2.0 * v12 + v11) * 0.125;

                        // filter normalization
                        *dx2_pixel = dx * dx;
                        *dy2_pixel = dy * dy;
                        *dxy_pixel = dx * dy;
                    });
            });

        dst.as_slice_mut()
            .get_mut(col_slice.clone())
            // SAFETY: we ranges is valid
            .unwrap()
            .par_chunks_exact_mut(src.cols())
            .enumerate()
            .for_each(|(row_idx, dst_chunk)| {
                let row_offset = (row_idx + 1) * src.cols();

                dst_chunk
                    .get_mut(row_slice.clone())
                    // SAFETY: we ranges is valid
                    .unwrap()
                    .iter_mut()
                    .enumerate()
                    .for_each(|(col_idx, dst_pixel)| {
                        let current_idx = row_offset + col_idx + 1;
                        let prev_row_idx = current_idx - src.cols();
                        let next_row_idx = current_idx + src.cols();

                        let mut m11 = 0.0;
                        let mut m22 = 0.0;
                        let mut m12 = 0.0;

                        let idxs = [
                            prev_row_idx - 1,
                            prev_row_idx,
                            prev_row_idx + 1,
                            current_idx - 1,
                            current_idx,
                            current_idx + 1,
                            next_row_idx - 1,
                            next_row_idx,
                            next_row_idx + 1,
                        ];
                        for idx in idxs {
                            // SAFETY: we ranges is valid
                            unsafe {
                                m11 += self.dx2_data.get_unchecked(idx);
                                m22 += self.dy2_data.get_unchecked(idx);
                                m12 += self.dxy_data.get_unchecked(idx);
                            }
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

/// Compute the DoG response of an image.
///
/// The DoG response is computed as the difference of the Gaussian responses of two images.
///
/// Args:
///     src: The source image with shape (H, W).
///     dst: The destination image with shape (H, W).
///     sigma1: The sigma of the first Gaussian kernel.
///     sigma2: The sigma of the second Gaussian kernel.
pub fn dog_response(
    src: &Image<f32, 1>,
    dst: &mut Image<f32, 1>,
    sigma1: f32,
    sigma2: f32,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let mut gauss1 = Image::from_size_val(src.size(), 0.0)?;
    let mut gauss2 = Image::from_size_val(src.size(), 0.0)?;
    let ks1 = _get_kernel_size(sigma1);
    let ks2 = _get_kernel_size(sigma2);

    gaussian_blur(src, &mut gauss1, (ks1, ks1), (sigma1, sigma1))?;
    gaussian_blur(src, &mut gauss2, (ks2, ks2), (sigma2, sigma2))?;

    let gauss1_data = gauss1.as_slice();
    let gauss2_data = gauss2.as_slice();
    let dst_data = dst.as_slice_mut();

    dst_data
        .iter_mut()
        .zip(gauss2_data.iter().zip(gauss1_data.iter()))
        .for_each(|(dst_pixel, (gauss2_pixel, gauss1_pixel))| {
            *dst_pixel = gauss2_pixel - gauss1_pixel;
        });

    Ok(())
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
        HarrisResponse::new(dst.size()).compute(&src, &mut dst)?;

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

    #[test]
    fn test_harris_rectangle() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let src = Image::from_size_slice(
            ImageSize {width: 9, height: 12},
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        )?;

        let mut dst = Image::from_size_val(src.size(), 0.0)?;
        HarrisResponse::new(src.size())
            .with_k(0.01)
            .compute(&src, &mut dst)?;

        #[rustfmt::skip]
        assert_eq!(dst.as_slice(), &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.03125, 0.17875001, 0.144375, 0.0, 0.144375, 0.17875001, 0.03125, 0.0,
            0.0, 0.17875001, 0.57125, 0.456875, 0.0, 0.456875, 0.57125, 0.17875001, 0.0,
            0.0, 0.144375, 0.456875, 0.374209, 0.0, 0.374209, 0.456875, 0.144375, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.144375, 0.456875, 0.374209, 0.0, 0.374209, 0.456875, 0.144375, 0.0,
            0.0, 0.17875001, 0.57125, 0.456875, 0.0, 0.456875, 0.57125, 0.17875001, 0.0,
            0.0, 0.03125, 0.17875001, 0.144375, 0.0, 0.144375, 0.17875001, 0.03125, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]);

        Ok(())
    }

    #[test]
    fn test_harris_builder_pattern() -> Result<(), ImageError> {
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
        HarrisResponse::new(dst.size())
            .with_k(0.01)
            .compute(&src, &mut dst)?;

        #[rustfmt::skip]
        assert_eq!(dst.as_slice(), &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.03125, 0.17875001, 0.144375, 0.0, 0.144375, 0.17875001, 0.03125, 0.0,
            0.0, 0.17875001, 0.57125, 0.456875, 0.0, 0.456875, 0.57125, 0.17875001, 0.0,
            0.0, 0.144375, 0.456875, 0.374209, 0.0, 0.374209, 0.456875, 0.144375, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.144375, 0.456875, 0.374209, 0.0, 0.374209, 0.456875, 0.144375, 0.0,
            0.0, 0.17875001, 0.57125, 0.456875, 0.0, 0.456875, 0.57125, 0.17875001, 0.0,
            0.0, 0.03125, 0.17875001, 0.144375, 0.0, 0.144375, 0.17875001, 0.03125, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]);

        Ok(())
    }

    #[test]
    fn test_dog_response() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let src = Image::from_size_slice(
            [5, 5].into(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )?;

        let mut dst = Image::from_size_val([5, 5].into(), 0.0)?;

        let sigma1 = 0.5;
        let sigma2 = 1.0;

        dog_response(&src, &mut dst, sigma1, sigma2)?;

        let center_value = dst.as_slice()[2 * 5 + 2];
        let expected_center_value = -0.2195;
        assert!(
            (center_value - expected_center_value).abs() < 1e-4,
            "Center value should be close to expected value"
        );

        let sum: f32 = dst.as_slice().iter().sum();
        let expected_sum = -0.7399;
        assert!(
            (sum - expected_sum).abs() < 1e-4,
            "Sum of DoG response should be close to expected value"
        );

        Ok(())
    }
}
