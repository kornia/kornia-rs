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
                    let k_offset = k_idx - half_kernel_x;
                    let x_pos = c + k_offset;
                    if x_pos > 0 && x_pos < src.cols() {
                        let neighbor_idx = (row_offset + x_pos) * C + ch;
                        row_acc += src_data[neighbor_idx] * k_val;
                    }
                }
                temp[pix_offset] = row_acc;
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
                    let k_offset = k_idx - half_kernel_y;
                    let y_pos = r + k_offset;
                    if y_pos > 0 && y_pos < src.rows() {
                        let neighbor_idx = (y_pos * src.cols() + c) * C + ch;
                        col_acc += temp[neighbor_idx] * k_val;
                    }
                }
                dst_data[pix_offset] = col_acc;
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
        let img = Image::new(
            size,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            ],
        )?;

        let mut dst = Image::<_, 1>::from_size_val(img.size(), 0f32)?;
        let kernel_x = vec![1.0];
        let kernel_y = vec![1.0];
        separable_filter(&img, &mut dst, &kernel_x, &kernel_y)?;
        assert_eq!(
            dst.as_slice(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0, 10.0, 0.0, 12.0, 13.0, 14.0, 15.0,
                0.0, 17.0, 18.0, 19.0, 20.0, 0.0, 22.0, 23.0, 24.0, 25.0
            ]
        );

        Ok(())
    }
}
