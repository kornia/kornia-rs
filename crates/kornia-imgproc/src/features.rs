use num_traits::{AsPrimitive, NumCast};
use kornia_image::{Image, ImageError};

use rayon::prelude::*;
use crate::filter::{gaussian_blur, kernels, separable_filter};
// use std::arch::x86_64::*;


/// Method to calculate gradient for feature response
pub enum GradsMode {
    /// Sobel operators
    Sobel,
    /// Finite difference
    Diff,
}

impl Default for GradsMode {
    fn default() -> Self {
        GradsMode::Sobel
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

/// Computes the harris response of an image.
///
/// The Harris response is computed by the determinant minus the trace squared.
///
/// Args:
///     src: The source image with shape (H, W).
///     dst: The destination image with shape (H, W).
///     k: Harris detector free parameter (usually between 0.04 to 0.06)
// #[target_feature(enable = "avx2")]  // tested on x86_64
pub fn harris_response(
    src: &Image<f32, 1>,
    dst: &mut Image<f32, 1>,
    k: Option<f32>,
    _grads_mode: GradsMode,
    _sigmas: Option<f32>
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // let src_data = src.as_slice();
    // let mut dx2_data = vec![0.0; src_data.len()];
    // let mut dy2_data = vec![0.0; src_data.len()];
    // let mut dxy_data = vec![0.0; src_data.len()];
    // let mut dx2_blurred: Image<f32, 1> = Image::from_size_val(src.size(), 0.0f32)?;
    // let mut dy2_blurred: Image<f32, 1> = Image::from_size_val(src.size(), 0.0f32)?;
    // let mut dxy_blurred: Image<f32, 1> = Image::from_size_val(src.size(), 0.0f32)?;

    let (kernel_x, kernel_y) = kernels::sobel_kernel_1d(3);

    let mut gx = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
    separable_filter(src, &mut gx, &kernel_x, &kernel_y)?;
    let gx_data = gx.as_slice();

    let mut gy = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
    separable_filter(src, &mut gy, &kernel_y, &kernel_x)?;
    let gy_data = gy.as_slice();


    // // SIMD/AVS2 optimization
    // for i in 0..gx_data.len() {
    //     dx2_data[i] = gx_data[i]*gx_data[i]*0.125*0.125;
    //     dy2_data[i] = gy_data[i]*gy_data[i]*0.125*0.125;
    //     dxy_data[i] = gx_data[i]*gy_data[i]*0.125*0.125;
    // }


    // dx2_data.as_mut_slice().par_chunks_exact_mut(src.cols())
    //     .zip(dy2_data.as_mut_slice().par_chunks_exact_mut(src.cols()))
    //     .zip(dxy_data.as_mut_slice().par_chunks_exact_mut(src.cols()))
    //     .enumerate()
    //     .for_each(|(row_idx, ((dx2_chunk, dy2_chunk), dxy_chunk))| {
    //         if row_idx == 0 || row_idx == src.rows() - 1 {
    //             // skip the first and last row
    //             return;
    //         }
    //
    //         let row_offset = row_idx * src.cols();
    //
    //         dx2_chunk.iter_mut()
    //             .zip(dy2_chunk.iter_mut())
    //             .zip(dxy_chunk.iter_mut())
    //             .enumerate()
    //             .for_each(|(col_idx, ((dx2_pixel,dy2_pixel),dxy_pixel))| {
    //                 if col_idx == 0 || col_idx == src.cols() - 1 {
    //                     // skip the first and last column
    //                     return;
    //                 }
    //
    //                 let current_idx = row_offset + col_idx;
    //                 let prev_row_idx = current_idx - src.cols();
    //                 let next_row_idx = current_idx + src.cols();
    //
    //                 let v11 = src_data[prev_row_idx - 1];
    //                 let v12 = src_data[prev_row_idx];
    //                 let v13 = src_data[prev_row_idx + 1];
    //                 let v21 = src_data[current_idx - 1];
    //                 // let v22 = src_data[current_idx];
    //                 let v23 = src_data[current_idx + 1];
    //                 let v31 = src_data[next_row_idx - 1];
    //                 let v32 = src_data[next_row_idx];
    //                 let v33 = src_data[next_row_idx + 1];
    //
    //                 // I_x,I_y via 3x3 sobel operator and convolved
    //                 let dx = -v33+v31 -2.0*v23+2.0*v21 - v13+v11;
    //                 let dy = -v33-2.0*v32-v31 + v13+2.0*v12+v11;
    //
    //                 // filter normalization
    //                 *dx2_pixel = dx*dx*0.125*0.125;
    //                 *dy2_pixel = dy*dy*0.125*0.125;
    //                 *dxy_pixel = dx*dy*0.125*0.125;
    //             });
    //     });

    // gaussian_blur(&Image::from_size_slice(src.size(), &dx2_data)?, &mut dx2_blurred,
    //               (7,7), (1.0,1.0))?;
    // gaussian_blur(&Image::from_size_slice(src.size(), &dy2_data)?, &mut dy2_blurred,
    //               (7,7), (1.0,1.0))?;
    // gaussian_blur(&Image::from_size_slice(src.size(), &dxy_data)?, &mut dxy_blurred,
    //               (7,7), (1.0,1.0))?;

    dst.as_slice_mut().par_chunks_exact_mut(src.cols())
        // .zip(dx2_data.as_slice().par_chunks_exact(src.cols()))
        // .zip(dx2_data.as_slice().par_chunks_exact(src.cols()))
        // .zip(dx2_data.as_slice().par_chunks_exact(src.cols()))
        // .zip(gx.as_slice().par_chunks_exact(src.cols()))
        // .zip(gy.as_slice().par_chunks_exact(src.cols()))
        .enumerate()
        .for_each(|(row_idx, dst_chunk)| {
            if row_idx == 0 || row_idx == src.rows() - 1 {
                // skip the first and last row
                return;
            }

            let row_offset = row_idx * src.cols();

            dst_chunk.iter_mut()
                // .zip(gx_chunk.iter())
                // .zip(gy_chunk.iter())
                .enumerate()
                .for_each(|(col_idx, dst_pixel)| {
                    if col_idx == 0 || col_idx == src.cols() - 1 {
                        // skip the first and last column
                        return;
                    }

                    let current_idx = row_offset + col_idx;
                    let prev_row_idx = current_idx - src.cols();
                    let next_row_idx = current_idx + src.cols();

                    let mut m11 = 0.0;
                    let mut m22 = 0.0;
                    let mut m12 = 0.0;  // same as m21

                    let idxs = [
                        prev_row_idx - 1, prev_row_idx, prev_row_idx + 1,
                        current_idx - 1, current_idx, current_idx + 1,
                        next_row_idx - 1, next_row_idx, next_row_idx + 1,
                    ];
                    for idx in idxs {
                        m11 += gx_data[idx]*gx_data[idx]*0.125*0.125;
                        m22 += gy_data[idx]*gy_data[idx]*0.125*0.125;
                        m12 += gx_data[idx]*gy_data[idx]*0.125*0.125;
                    }

                    let det = m11*m22 - m12*m12;
                    let trace = m11 + m22;
                    let response = det-k.unwrap_or(0.04)*trace*trace;

                    *dst_pixel = f32::max(0.0, response);
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
        println!("{:?}", std::env::consts::ARCH);

        // 7x7 actual matrix with 6x6 padding (manually added)
        // this test uses the python version, which includes padded, reflected
        //      gaussian blurring
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
        harris_response(&src, &mut dst, None, GradsMode::Sobel, None)?;

        // pulled from kornia's example
        #[rustfmt::skip]
        assert_eq!(dst.as_slice(), &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012, 0.0,
            0.0, 0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039, 0.0,
            0.0, 0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020, 0.0,
            0.0, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0,
            0.0, 0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020, 0.0,
            0.0, 0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039, 0.0,
            0.0, 0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        /*
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.01953125, 0.14078125, 0.08238281, 0.0, 0.08238281, 0.14078125, 0.01953125, 0.0,
         0.0, 0.14078125, 0.49203125, 0.37144533, 0.0, 0.37144533, 0.49203125, 0.14078125, 0.0,
         0.0, 0.08238281, 0.37144533, 0.32496095, 0.0, 0.32496095, 0.37144533, 0.08238281, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.08238281, 0.37144533, 0.32496095, 0.0, 0.32496095, 0.37144533, 0.08238281, 0.0,
         0.0, 0.14078125, 0.49203125, 0.37144533, 0.0, 0.37144533, 0.49203125, 0.14078125, 0.0,
         0.0, 0.01953125, 0.14078125, 0.08238281, 0.0, 0.08238281, 0.14078125, 0.01953125, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        */
        /*
        [0.0,         0.0,           0.0,          0.0,          0.0,          0.0, 0.0,
         0.0,   0.10839844,   0.18792969, -0.022499999,   0.18792969,   0.10839844, 0.0,
         0.0,   0.18792969,   0.32496095, -0.022499999,   0.32496095,   0.18792969, 0.0,
         0.0, -0.022499999, -0.022499999,          0.0, -0.022499999, -0.022499999, 0.0,
         0.0,   0.18792969,   0.32496095, -0.022499999,   0.32496095,   0.18792969, 0.0,
         0.0,   0.10839844,   0.18792969, -0.022499999,   0.18792969,   0.10839844, 0.0,
         0.0,          0.0,          0.0,          0.0,          0.0,          0.0, 0.0]
        */
        /*
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
         */

        Ok(())
    }
}
