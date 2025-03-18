use num_traits::{AsPrimitive, NumCast};
use kornia_image::{Image, ImageError};

use rayon::prelude::*;
use wide::{f32x4, f32x8};
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
#[inline(never)]
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

    let src_data = src.as_slice();
    let col_slice = src.cols()..src_data.len()-src.cols();
    let row_slice = 1..src.rows()-1;
    let mut dx_data = vec![0.0; src_data.len()];
    let mut dy_data = vec![0.0; src_data.len()];
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

    // dx2_data.as_mut_slice()[col_slice.clone()].par_chunks_exact_mut(src.cols())//.skip(1).take(src.rows()-2)
    //     .zip(dy2_data.as_mut_slice()[col_slice.clone()].par_chunks_exact_mut(src.cols()))//.skip(1).take(src.rows()-2))
    //     .zip(dxy_data.as_mut_slice()[col_slice.clone()].par_chunks_exact_mut(src.cols()))//.skip(1).take(src.rows()-2))
    dx_data.as_mut_slice()[col_slice.clone()].par_chunks_exact_mut(src.cols())
        .zip(dy_data.as_mut_slice()[col_slice.clone()].par_chunks_exact_mut(src.cols()))
        .enumerate()
        .for_each(|(row_idx, (dx_chunk, dy_chunk))| {
            // if row_idx == 0 || row_idx == src.rows() - 1 {
            //     // skip the first and last row
            //     return;
            // }

            let row_offset = (row_idx+1) * src.cols();

            // dx2_chunk[row_slice.clone()].iter_mut()//.skip(1).take(src.cols()-2)
            //     .zip(dy2_chunk[row_slice.clone()].iter_mut())//.skip(1).take(src.cols()-2))
            //     .zip(dxy_chunk[row_slice.clone()].iter_mut())//.skip(1).take(src.cols()-2))
            dx_chunk[row_slice.clone()].iter_mut()
                .zip(dy_chunk[row_slice.clone()].iter_mut())
                .enumerate()
                .for_each(|(col_idx, (dx_pixel, dy_pixel))| {
                    // if col_idx == 0 || col_idx == src.cols() - 1 {
                    //     // skip the first and last column
                    //     return;
                    // }

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
                    *dx_pixel /* let dx */ = (-v33+v31 -2.0*v23+2.0*v21 - v13+v11)*0.125;
                    *dy_pixel /* let dy */ = (-v33-2.0*v32-v31 + v13+2.0*v12+v11)*0.125;

                    // filter normalization
                    // *dx2_pixel = dx*dx*0.125*0.125;
                    // *dy2_pixel = dy*dy*0.125*0.125;
                    // *dxy_pixel = dx*dy*0.125*0.125;
                });
        });

    // partial sum
    unsafe {
        *dx2_data.get_unchecked_mut(0) = dx_data.get_unchecked(0)*dx_data.get_unchecked(0);
        *dy2_data.get_unchecked_mut(0) = dy_data.get_unchecked(0)*dy_data.get_unchecked(0);
        *dxy_data.get_unchecked_mut(0) = dx_data.get_unchecked(0)*dy_data.get_unchecked(0);
        for i in 1..src.rows() {
            let cur = i*src.cols();
            let prev = (i-1)*src.cols();
            *dx2_data.get_unchecked_mut(cur) = dx2_data.get_unchecked(prev)+dx_data.get_unchecked(cur)*dx_data.get_unchecked(cur);
            *dy2_data.get_unchecked_mut(cur) = dy2_data.get_unchecked(prev)+dy_data.get_unchecked(cur)*dy_data.get_unchecked(cur);
            *dxy_data.get_unchecked_mut(cur) = dxy_data.get_unchecked(prev)+dx_data.get_unchecked(cur)*dy_data.get_unchecked(cur);
        }
        for j in 1..src.cols() {
            *dx2_data.get_unchecked_mut(j) = dx2_data.get_unchecked(j-1)+dx_data.get_unchecked(j)*dx_data.get_unchecked(j);
            *dy2_data.get_unchecked_mut(j) = dy2_data.get_unchecked(j-1)+dy_data.get_unchecked(j)*dy_data.get_unchecked(j);
            *dxy_data.get_unchecked_mut(j) = dxy_data.get_unchecked(j-1)+dx_data.get_unchecked(j)*dy_data.get_unchecked(j);
        }
        for i in 1..src.rows() {
            for j in 1..src.cols() {
                let cur = i*src.cols()+j;
                let l = i*src.cols()+j-1;
                let u = (i-1)*src.cols()+j;
                let ul = (i-1)*src.cols()+j-1;
                *dx2_data.get_unchecked_mut(i*src.cols()+j) = dx2_data.get_unchecked(u)+dx2_data.get_unchecked(l)-
                    dx2_data.get_unchecked(ul)+dx_data.get_unchecked(cur)*dx_data.get_unchecked(cur);
                *dy2_data.get_unchecked_mut(i*src.cols()+j) = dy2_data.get_unchecked(u)+dy2_data.get_unchecked(l)-
                    dy2_data.get_unchecked(ul)+dy_data.get_unchecked(cur)*dy_data.get_unchecked(cur);
                *dxy_data.get_unchecked_mut(i*src.cols()+j) = dxy_data.get_unchecked(u)+dxy_data.get_unchecked(l)-
                    dxy_data.get_unchecked(ul)+dx_data.get_unchecked(cur)*dy_data.get_unchecked(cur);
            }
        }
    }

    // if std::is_x86_feature_detected!("avx2") {
    //     println!("AVX2 is supported and should be used");
    // } else {
    //     println!("AVX2 is NOT supported. Falling back to scalar operations");
    // }

    // let chunk_size = 4096;
    // let lane_numbers = 4;

    // dx2_data.as_mut_slice().par_chunks_exact_mut(chunk_size)
    //     .zip(dy2_data.as_mut_slice().par_chunks_exact_mut(chunk_size))
    //     .zip(dxy_data.as_mut_slice().par_chunks_exact_mut(chunk_size))
    //     .zip(dx_data.as_slice().par_chunks_exact(chunk_size))
    //     .zip(dy_data.as_slice().par_chunks_exact(chunk_size))
    //     // .enumerate()
    //     .for_each(|(((((dx2_chunk), dy2_chunk), dxy_chunk), dx_chunk), dy_chunk)| {
    //         let mut i = 0;
    //         while i+lane_numbers <= dx2_chunk.len() {
    //             let dx_simd = f32x4::from(&dx_chunk[i..i+lane_numbers]);
    //             let dy_simd = f32x4::from(&dy_chunk[i..i+lane_numbers]);
    //             let dx2_simd = dx_simd*dx_simd*0.125*0.125;
    //             let dy2_simd = dy_simd*dy_simd*0.125*0.125;
    //             let dxy_simd = dx_simd*dy_simd*0.125*0.125;
    //             dx2_chunk[i..i+lane_numbers].copy_from_slice(&dx2_simd.to_array());
    //             dy2_chunk[i..i+lane_numbers].copy_from_slice(&dy2_simd.to_array());
    //             dxy_chunk[i..i+lane_numbers].copy_from_slice(&dxy_simd.to_array());
    //             i += lane_numbers
    //         }
    //     });
    // let mut i = (src_data.len()/chunk_size)*chunk_size;
    // // simd optimization
    // while i+lane_numbers <= dx_data.len() {
    //     let dx_simd = f32x4::from(&dx_data[i..i+lane_numbers]);
    //     let dy_simd = f32x4::from(&dy_data[i..i+lane_numbers]);
    //     let dx2_simd = dx_simd*dx_simd*0.125*0.125;
    //     let dy2_simd = dy_simd*dy_simd*0.125*0.125;
    //     let dxy_simd = dx_simd*dy_simd*0.125*0.125;
    //     dx2_data[i..i+lane_numbers].copy_from_slice(&dx2_simd.to_array());
    //     dy2_data[i..i+lane_numbers].copy_from_slice(&dy2_simd.to_array());
    //     dxy_data[i..i+lane_numbers].copy_from_slice(&dxy_simd.to_array());
    //     i += lane_numbers
    // }
    // // finish remaining parts
    // for j in i..dx_data.len() {
    //     let dx = dx_data[j];
    //     let dy = dy_data[j];
    //     let dx2 = dx*dx*0.125*0.125;
    //     let dy2 = dy*dy*0.125*0.125;
    //     let dxy = dx*dy*0.125*0.125;
    //     dx2_data[j] = dx2;
    //     dy2_data[j] = dy2;
    //     dxy_data[j] = dxy;
    // }
    // println!("dx2_data: {:?}", dx2_data);

    // gaussian_blur(&Image::from_size_slice(src.size(), &dx2_data)?, &mut dx2_blurred,
    //               (7,7), (1.0,1.0))?;
    // gaussian_blur(&Image::from_size_slice(src.size(), &dy2_data)?, &mut dy2_blurred,
    //               (7,7), (1.0,1.0))?;
    // gaussian_blur(&Image::from_size_slice(src.size(), &dxy_data)?, &mut dxy_blurred,
    //               (7,7), (1.0,1.0))?;

    // let mut m11_data = vec![0.0; src_data.len()];
    // let mut m22_data = vec![0.0; src_data.len()];
    // let mut m12_data = vec![0.0; src_data.len()];

    dst.as_slice_mut()[col_slice.clone()].par_chunks_exact_mut(src.cols())//.skip(1).take(src.rows()-2)
    // m11_data.as_mut_slice()[col_slice.clone()].par_chunks_exact_mut(src.cols())
    //     .zip(m22_data.as_mut_slice()[col_slice.clone()].par_chunks_exact_mut(src.cols()))
    //     .zip(m12_data.as_mut_slice()[col_slice.clone()].par_chunks_exact_mut(src.cols()))
        // .zip(dx2_data.as_slice().par_chunks_exact(src.cols()))
        // .zip(dx2_data.as_slice().par_chunks_exact(src.cols()))
        // .zip(dx2_data.as_slice().par_chunks_exact(src.cols()))
        // .zip(gx.as_slice().par_chunks_exact(src.cols()))
        // .zip(gy.as_slice().par_chunks_exact(src.cols()))
        .enumerate()
        .for_each(|(row_idx, dst_chunk)| {
            // if row_idx == 0 || row_idx == src.rows() - 1 {
            //     // skip the first and last row
            //     return;
            // }

            let row_offset = (row_idx+1) * src.cols();

            dst_chunk[row_slice.clone()].iter_mut()//.skip(1).take(src.cols()-2)
            // m11_chunk[row_slice.clone()].iter_mut()
            //     .zip(m22_chunk[row_slice.clone()].iter_mut())
            //     .zip(m12_chunk[row_slice.clone()].iter_mut())
                // .zip(gx_chunk.iter())
                // .zip(gy_chunk.iter())
                .enumerate()
                .for_each(|(col_idx, dst_pixel)| {
                    // if col_idx == 0 || col_idx == src.cols() - 1 {
                    //     // skip the first and last column
                    //     return;
                    // }

                    let current_idx = (row_offset + col_idx+1) as isize;
                    // let prev_row_idx = current_idx - src.cols();
                    // let next_row_idx = current_idx + src.cols();
                    let ul_idx = current_idx - (src.cols() as isize)*2 - 2;  // upper left
                    let lr_idx = current_idx + (src.cols() as isize) + 1;  // lower right
                    let ll_idx = current_idx + (src.cols() as isize) - 2;  // lower left
                    let ur_idx = current_idx - (src.cols() as isize)*2 + 1;  // upper right

                    // let mut m11 = 0.0;
                    // let mut m22 = 0.0;
                    // let mut m12 = 0.0;  // same as m21
                    //
                    // let idxs = [
                    //     prev_row_idx - 1, prev_row_idx, prev_row_idx + 1,
                    //     current_idx - 1, current_idx, current_idx + 1,
                    //     next_row_idx - 1, next_row_idx, next_row_idx + 1,
                    // ];
                    // for idx in idxs {
                    //     m11 += unsafe { dx2_data.get_unchecked(idx) };// gx_data[idx]*gx_data[idx]*0.125*0.125;
                    //     m22 += unsafe { dy2_data.get_unchecked(idx) };// gy_data[idx]*gy_data[idx]*0.125*0.125;
                    //     m12 += unsafe { dxy_data.get_unchecked(idx) };// gx_data[idx]*gy_data[idx]*0.125*0.125;
                    // }

                    unsafe { // requesting 3x3 partial sum
                        let mut m11 = *dx2_data.get_unchecked(lr_idx as usize);
                        let mut m22 = *dy2_data.get_unchecked(lr_idx as usize);
                        let mut m12 = *dxy_data.get_unchecked(lr_idx as usize);
                        if col_idx > 1 {
                            m11 += dx2_data.get_unchecked(ll_idx as usize);
                            m22 += dx2_data.get_unchecked(ll_idx as usize);
                            m12 += dx2_data.get_unchecked(ll_idx as usize);
                        }
                        if row_idx > 1 {
                            m11 += dx2_data.get_unchecked(ur_idx as usize);
                            m22 += dx2_data.get_unchecked(ur_idx as usize);
                            m12 += dx2_data.get_unchecked(ur_idx as usize);
                        }
                        if col_idx > 1 && row_idx > 1 {
                            m11 += dx2_data.get_unchecked(ul_idx as usize);
                            m22 += dx2_data.get_unchecked(ul_idx as usize);
                            m12 += dx2_data.get_unchecked(ul_idx as usize);
                        }

                        // let idxs = [
                        //     prev_row_idx - 1, prev_row_idx, prev_row_idx + 1,
                        //     current_idx - 1, current_idx, current_idx + 1,
                        //     next_row_idx - 1, next_row_idx, next_row_idx + 1,
                        // ];
                        // for idx in idxs {
                        //     m11 += unsafe { dx2_data.get_unchecked(idx) };// gx_data[idx]*gx_data[idx]*0.125*0.125;
                        //     m22 += unsafe { dy2_data.get_unchecked(idx) };// gy_data[idx]*gy_data[idx]*0.125*0.125;
                        //     m12 += unsafe { dxy_data.get_unchecked(idx) };// gx_data[idx]*gy_data[idx]*0.125*0.125;
                        // }

                        // *m11_pixel = m11;
                        // *m22_pixel = m22;
                        // *m12_pixel = m12;

                        let det = m11 * m22 - m12 * m12;
                        let trace = m11 + m22;
                        let response = det - k.unwrap_or(0.04) * trace * trace;

                        *dst_pixel = f32::max(0.0, response);
                    }
                });
        });


    // finish remaining parts
    // println!("m11_data: {:?}", m11_data);
    // println!("m22_data: {:?}", m22_data);
    // println!("m12_data: {:?}", m12_data);

    // let mut dst_data = dst.as_slice_mut();
    // dst_data.par_chunks_exact_mut(chunk_size)
    //     .zip(m11_data.as_slice().par_chunks_exact(chunk_size))
    //     .zip(m22_data.as_slice().par_chunks_exact(chunk_size))
    //     .zip(m12_data.as_slice().par_chunks_exact(chunk_size))
    //     // .enumerate()
    //     .for_each(|((((dst_chunk), m11_chunk), m22_chunk), m12_chunk)| {
    //         let mut i = 0;
    //         while i+lane_numbers <= dst_chunk.len() {
    //             let m11_simd = f32x4::from(&m11_chunk[i..i+lane_numbers]);
    //             let m22_simd = f32x4::from(&m22_chunk[i..i+lane_numbers]);
    //             let m12_simd = f32x4::from(&m12_chunk[i..i+lane_numbers]);
    //             let pre_simd = m11_simd*m22_simd-m12_simd*m12_simd-k.unwrap_or(0.04)*(m11_simd+m22_simd)*(m11_simd+m22_simd);
    //             let dst_simd = f32x4::max(f32x4::splat(0.0), pre_simd);
    //             dst_chunk[i..i+lane_numbers].copy_from_slice(&dst_simd.to_array());
    //             i += lane_numbers
    //         }
    //     });
    // let mut i = (src_data.len()/chunk_size)*chunk_size;
    // while i+lane_numbers <= dx_data.len() {
    //     let m11_simd = f32x4::from(&m11_data[i..i+lane_numbers]);
    //     let m22_simd = f32x4::from(&m22_data[i..i+lane_numbers]);
    //     let m12_simd = f32x4::from(&m12_data[i..i+lane_numbers]);
    //     let pre_simd = m11_simd*m22_simd-m12_simd*m12_simd-k.unwrap_or(0.04)*(m11_simd+m22_simd)*(m11_simd+m22_simd);
    //     let dst_simd = f32x4::max(f32x4::splat(0.0), pre_simd);
    //     dst_data[i..i+lane_numbers].copy_from_slice(&dst_simd.to_array());
    //     i += lane_numbers
    // }
    // for j in i..dx_data.len() {
    //     let m11 = m11_data[j];
    //     let m22 = m22_data[j];
    //     let m12 = m12_data[j];
    //     let dst_pixel = m11*m22 - m12*m12 - k.unwrap_or(0.04)*(m11+m22)*(m11+m22);
    //     dst_data[j] = f32::max(0.0, dst_pixel);
    // }
    // println!("dst_data: {:?}", dst_data);

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
        /* skip/take iter version (the output is likely wrong)
          [0.0, 0.0,         0.0, 0.0,            0.0,           0.0, 0.0, 0.0, 0.0,
           0.0, 0.0,         0.0, 0.168125, 0.29878905, 0.124648444, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.043789063, 0.519375, 0.86566406, 0.46402344, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.039541014, 0.3868262, 0.6730566, 0.40311524, 0.0, 0.0, 0.0,
           0.0, 0.0,         0.0,       0.0,       0.0,        0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.039541014, 0.3868262, 0.6730566, 0.40311524, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.043789063, 0.519375, 0.86566406, 0.46402344, 0.0, 0.0, 0.0,
           0.0, 0.0,         0.0, 0.168125, 0.29878905, 0.124648444, 0.0, 0.0, 0.0,
           0.0, 0.0,         0.0,       0.0,       0.0,         0.0, 0.0, 0.0, 0.0]

        */
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
