use kornia_image::{Image, ImageError};

/// Apply a separable filter to an image.
pub fn separable_filter(
    src: &Image<f32, 3>,
    dst: &mut Image<f32, 3>,
    kernel_x: &[f32],
    kernel_y: &[f32],
) -> Result<(), ImageError> {
    if kernel_x.len() != kernel_y.len() {
        return Err(ImageError::MismatchedKernelLength(
            kernel_x.len(),
            kernel_y.len(),
        ));
    }

    let norm_x = kernel_x.iter().sum::<f32>();
    let norm_y = kernel_y.iter().sum::<f32>();

    let half_kernel_x = kernel_x.len() / 2;
    let half_kernel_y = kernel_y.len() / 2;

    let src_data = src.as_slice();
    let dst_data = dst.as_slice_mut();

    // preallocate the temporary buffer for intermediate results
    let mut temp = vec![0.0; src_data.len()];

    // apply the horizontal filter
    for y in 0..src.rows() {
        let row_offset = y * src.cols();
        for x in 0..src.cols() {
            let col_offset = row_offset + x;
            for c in 0..src.num_channels() {
                let temp_idx = col_offset * src.num_channels() + c;
                let mut sum = 0.0;
                for (kx_idx, kx) in kernel_x.iter().enumerate() {
                    let x_pos = x + kx_idx - half_kernel_x;
                    if x_pos > 0 && x_pos < src.cols() {
                        let src_idx = (row_offset + x_pos) * src.num_channels() + c;
                        sum += src_data[src_idx] * kx;
                    }
                }
                temp[temp_idx] = sum / norm_x;
            }
        }
    }

    // apply the vertical filter
    for x in 0..src.cols() {
        for y in 0..src.rows() {
            let row_offset = y * src.cols();
            for c in 0..src.num_channels() {
                let dst_idx = (row_offset + x) * src.num_channels() + c;
                let mut sum = 0.0;
                for (ky_idx, ky) in kernel_y.iter().enumerate() {
                    let y_pos = y + ky_idx - half_kernel_y;
                    if y_pos > 0 && y_pos < src.rows() {
                        let temp_idx = (y_pos * src.cols() + x) * src.num_channels() + c;
                        sum += temp[temp_idx] * ky;
                    }
                }
                dst_data[dst_idx] = sum / norm_y;
            }
        }
    }

    Ok(())
}
