use kornia_image::{Image, ImageError};

use rayon::prelude::*;
use crate::filter::gaussian_blur;

/// Method to calculate gradient for feature response
pub enum GradsMode {
    /// Sobel operators
    SOBEL,
    /// Finite difference
    DIFF,
}

impl Default for GradsMode {
    fn default() -> Self {
        GradsMode::SOBEL
    }
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

/// Computes the harris response
pub fn harris_response(src: &Image<f32, 1>,
                       dst: &mut Image<f32, 1>,
                       k: Option<f32>,
                       _grads_mode: GradsMode,
                       _sigmas: Option<f32>) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let src_data = src.as_slice();
    let mut dx2_data = vec![0.0; src_data.len()];
    let mut dy2_data = vec![0.0; src_data.len()];
    let mut dxy_data = vec![0.0; src_data.len()];
    let mut dx2_blurred: Image<f32, 1> = Image::from_size_val(src.size(), 0.0f32)?;
    let mut dy2_blurred: Image<f32, 1> = Image::from_size_val(src.size(), 0.0f32)?;
    let mut dxy_blurred: Image<f32, 1> = Image::from_size_val(src.size(), 0.0f32)?;

    dx2_data.as_mut_slice().par_chunks_exact_mut(src.cols())
        .zip(dy2_data.as_mut_slice().par_chunks_exact_mut(src.cols()))
        .zip(dxy_data.as_mut_slice().par_chunks_exact_mut(src.cols()))
        .enumerate()
        .for_each(|(row_idx, ((dx2_chunk, dy2_chunk), dxy_chunk))| {
            if row_idx == 0 || row_idx == src.rows() - 1 {
                // skip the first and last row
                return;
            }

            let row_offset = row_idx * src.cols();

            dx2_chunk.iter_mut()
                .zip(dy2_chunk.iter_mut())
                .zip(dxy_chunk.iter_mut())
                .enumerate()
                .for_each(|(col_idx, ((dx2_pixel,dy2_pixel),dxy_pixel))| {
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
                    // let v22 = src_data[current_idx];
                    let v23 = src_data[current_idx + 1];
                    let v31 = src_data[next_row_idx - 1];
                    let v32 = src_data[next_row_idx];
                    let v33 = src_data[next_row_idx + 1];

                    // I_x,I_y via sobel operator and convolved
                    let dx = -v33+v31 -2.0*v23+2.0*v21 - v13+v11;
                    let dy = -v33-2.0*v32-v31 + v13+2.0*v12+v11;

                    // filter normalization
                    *dx2_pixel = dx*dx*0.125*0.125;
                    *dy2_pixel = dy*dy*0.125*0.125;
                    *dxy_pixel = dx*dy*0.125*0.125;
                });
        });

    gaussian_blur(&Image::from_size_slice(src.size(), &dx2_data)?, &mut dx2_blurred,
                  (7,7), (1.0,1.0))?;
    gaussian_blur(&Image::from_size_slice(src.size(), &dy2_data)?, &mut dy2_blurred,
                  (7,7), (1.0,1.0))?;
    gaussian_blur(&Image::from_size_slice(src.size(), &dxy_data)?, &mut dxy_blurred,
                  (7,7), (1.0,1.0))?;

    dst.as_slice_mut().par_chunks_exact_mut(src.cols())
        .zip(dx2_blurred.as_slice().par_chunks_exact(src.cols()))
        .zip(dy2_blurred.as_slice().par_chunks_exact(src.cols()))
        .zip(dxy_blurred.as_slice().par_chunks_exact(src.cols()))
        .enumerate()
        .for_each(|(row_idx, (((dst_chunk, dx2_chunk), dy2_chunk), dxy_chunk))| {
            if row_idx == 0 || row_idx == src.rows() - 1 {
                // skip the first and last row
                return;
            }

            // let row_offset = row_idx * src.cols();

            dst_chunk.iter_mut()
                .zip(dx2_chunk.iter())
                .zip(dy2_chunk.iter())
                .zip(dxy_chunk.iter())
                .enumerate()
                .for_each(|(col_idx, (((dst_pixel, dx2_pixel), dy2_pixel), dxy_pixel))| {
                    if col_idx == 0 || col_idx == src.cols() - 1 {
                        // skip the first and last column
                        return;
                    }

                    // let current_idx = row_offset + col_idx;
                    // let prev_row_idx = current_idx - src.cols();
                    // let next_row_idx = current_idx + src.cols();

                    // println!("{:?}", (dx2_pixel, dy2_pixel, dxy_pixel));

                    let det = dx2_pixel*dy2_pixel - dxy_pixel*dxy_pixel;
                    let trace = dx2_pixel + dy2_pixel;
                    let response = det-k.unwrap_or(0.04)*trace*trace;

                    *dst_pixel = response;
                });
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
        // 7x7 actual matrix with 6x6 padding (manually added)
        // this test uses the python version, which includes padded, reflected
        //      gaussian blurring
        #[rustfmt::skip]
        let src = Image::from_size_slice(
            [13, 13].into(),
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ]
        )?;

        let mut dst = Image::from_size_val([13, 13].into(), 0.0)?;
        harris_response(&src, &mut dst, None, GradsMode::SOBEL, None)?;

        // pulled from kornia's example
        #[rustfmt::skip]
        assert_eq!(dst.as_slice(), &[
            0.0,    0.0,    0.0,       0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
            0.0,    0.0,    0.0,       0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
            0.0,    0.0,    0.0,       0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
            0.0,    0.0,    0.0,    0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012,    0.0,    0.0,    0.0,
            0.0,    0.0,    0.0,    0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039,    0.0,    0.0,    0.0,
            0.0,    0.0,    0.0,    0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020,    0.0,    0.0,    0.0,
            0.0,    0.0,    0.0,    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,    0.0,    0.0,    0.0,
            0.0,    0.0,    0.0,    0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020,    0.0,    0.0,    0.0,
            0.0,    0.0,    0.0,    0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039,    0.0,    0.0,    0.0,
            0.0,    0.0,    0.0,    0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012,    0.0,    0.0,    0.0,
            0.0,    0.0,    0.0,       0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
            0.0,    0.0,    0.0,       0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
            0.0,    0.0,    0.0,       0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        ]);
        /*
        [0.0,          0.0,           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 4.4416565e-6,  4.4130673e-5, 9.646139e-5, 2.0181862e-5, 0.0,       0.0,           0.0, 2.0181862e-5, 9.646139e-5, 4.4130666e-5, 4.4416565e-6, 0.0,
         0.0,  4.413066e-5, 0.00042495583, 0.0009689632, 0.0006931168, 0.0,      0.0,           0.0, 0.0006931168, 0.0009689632, 0.00042495577, 4.413066e-5, 0.0,
         0.0,  9.646137e-5, 0.00096896343, 0.0022852654, 0.0023444288, 0.0010550992, 0.000115120725, 0.0010550992, 0.0023444288, 0.0022852654, 0.0009689633, 9.646137e-5, 0.0,
         0.0, 2.0181848e-5, 0.00069311686,  0.002344429, 0.0035938283, 0.0025961832,   0.0009943076, 0.0025961834, 0.0035938283,  0.002344429, 0.00069311686, 2.0181848e-5, 0.0,
         0.0,          0.0, 0.0,           0.0010550992, 0.0025961827, 0.0023042131,   0.0010774175, 0.0023042134, 0.0025961827, 0.0010550992, 0.0, 0.0, 0.0,
         0.0,          0.0, 0.0,           0.000115120725, 0.0009943072, 0.0010774173, 0.0005471616, 0.0010774175, 0.0009943072, 0.000115120725, 0.0, 0.0, 0.0,
         0.0,          0.0, 0.0,           0.0010550992, 0.0025961823, 0.0023042131,   0.0010774175, 0.0023042134, 0.0025961823, 0.0010550992, 0.0, 0.0, 0.0,
         0.0, 2.0181862e-5,  0.0006931169, 0.0023444288, 0.0035938285, 0.0025961832,   0.0009943073, 0.0025961834, 0.0035938285, 0.0023444288, 0.0006931169, 2.0181862e-5, 0.0,
         0.0,  9.646137e-5, 0.00096896343, 0.0022852654, 0.0023444288, 0.0010550992, 0.000115120725, 0.0010550992, 0.0023444288, 0.0022852654, 0.0009689633, 9.646137e-5, 0.0,
         0.0, 4.4130662e-5, 0.00042495588, 0.0009689633, 0.0006931168, 0.0,          0.0,            0.0,          0.0006931168, 0.0009689633, 0.00042495588, 4.4130662e-5, 0.0,
         0.0, 4.4416565e-6,  4.4130673e-5,  9.646137e-5, 2.0181848e-5, 0.0,          0.0,          0.0,         2.0181848e-5, 9.646137e-5, 4.4130666e-5, 4.4416565e-6, 0.0,
         0.0,          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
         */
        /* (9x9) simulating reflection padding
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0002841423, 0.0007295691, 0.0005588511, 0.00019384903, 0.0005588511, 0.0007295691, 0.0002841423, 0.0,
         0.0, 0.00072956935, 0.0023982283, 0.0021957904, 0.000919978, 0.0021957906, 0.0023982283, 0.00072956935, 0.0,
         0.0, 0.00055885117, 0.00219579, 0.002209182, 0.001059561, 0.0022091821, 0.00219579, 0.00055885117, 0.0,
         0.0, 0.00019384903, 0.000919978, 0.001059561, 0.0005471616, 0.001059561, 0.000919978, 0.00019384903, 0.0,
         0.0, 0.00055885117, 0.00219579, 0.0022091821, 0.001059561, 0.0022091824, 0.00219579, 0.00055885117, 0.0,
         0.0, 0.00072956935, 0.0023982283, 0.0021957904, 0.000919978, 0.0021957906, 0.0023982283, 0.00072956935, 0.0,
         0.0, 0.0002841423, 0.0007295691, 0.0005588511, 0.00019384903, 0.0005588511, 0.0007295691, 0.0002841423, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
         */
        /* (7x7)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0018022883, 0.0019787555, 0.00086526806, 0.0019787555, 0.0018022883, 0.0,
         0.0, 0.001978755, 0.0021538243, 0.0010490229, 0.0021538243, 0.001978755, 0.0,
         0.0, 0.00086526806, 0.0010490227, 0.0005456688, 0.0010490227, 0.00086526806, 0.0,
         0.0, 0.0019787552, 0.0021538243, 0.0010490227, 0.0021538245, 0.0019787552, 0.0,
         0.0, 0.0018022883, 0.001978755, 0.00086526806, 0.0019787552, 0.0018022883, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
         */

        Ok(())
    }
}
