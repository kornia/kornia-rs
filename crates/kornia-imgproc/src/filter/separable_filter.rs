use kornia_image::{Image, ImageError};

/// Apply a separable filter to an image.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with shape (H, W, C).
/// * `kernel_x` - The horizontal kernel.
/// * `kernel_y` - The vertical kernel.
pub fn separable_filter<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    kernel_x: &[f32],
    kernel_y: &[f32],
) -> Result<(), ImageError> {
    if kernel_x.is_empty() || kernel_y.is_empty() {
        return Err(ImageError::InvalidKernelLength(
            kernel_x.len(),
            kernel_y.len(),
        ));
    }

    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let half_kernel_x = kernel_x.len() / 2;
    let half_kernel_y = kernel_y.len() / 2;

    let src_data = src.as_slice();
    let dst_data = dst.as_slice_mut();

    // preallocate the temporary buffer for intermediate results
    // TODO: use a better buffer allocation strategy
    let mut temp = vec![0.0; src_data.len()];

    // Row-wise filtering
    for r in 0..src.rows() {
        let row_offset = r * src.cols();
        for c in 0..src.cols() {
            let col_offset = (row_offset + c) * C;
            for ch in 0..C {
                let pix_offset = col_offset + ch;
                let mut row_acc = 0.0;
                for (k_idx, k_val) in kernel_x.iter().enumerate() {
                    let x_pos = c as isize + k_idx as isize - half_kernel_x as isize;
                    if x_pos >= 0 && x_pos < src.cols() as isize {
                        let neighbor_idx = (row_offset + x_pos as usize) * C + ch;
                        row_acc += unsafe { src_data.get_unchecked(neighbor_idx) } * k_val;
                    }
                }

                unsafe {
                    *temp.get_unchecked_mut(pix_offset) = row_acc;
                }
            }
        }
    }

    // Column-wise filtering
    for r in 0..src.rows() {
        let row_offset = r * src.cols();
        for c in 0..src.cols() {
            let col_offset = (row_offset + c) * C;
            for ch in 0..C {
                let pix_offset = col_offset + ch;
                let mut col_acc = 0.0;
                for (k_idx, k_val) in kernel_y.iter().enumerate() {
                    let y_pos = r as isize + k_idx as isize - half_kernel_y as isize;
                    if y_pos >= 0 && y_pos < src.rows() as isize {
                        let neighbor_idx = (y_pos as usize * src.cols() + c) * C + ch;
                        col_acc += unsafe { temp.get_unchecked(neighbor_idx) } * k_val;
                    }
                }
                unsafe {
                    *dst_data.get_unchecked_mut(pix_offset) = col_acc;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    fn test_separable_filter() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        #[rustfmt::skip]
        let img = Image::new(
            size,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )?;

        let mut dst = Image::<_, 1>::from_size_val(img.size(), 0f32)?;
        let kernel_x = vec![1.0, 1.0, 1.0];
        let kernel_y = vec![1.0, 1.0, 1.0];
        separable_filter(&img, &mut dst, &kernel_x, &kernel_y)?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );

        let xsum = dst.as_slice().iter().sum::<f32>();
        assert_eq!(xsum, 9.0);

        Ok(())
    }
}
