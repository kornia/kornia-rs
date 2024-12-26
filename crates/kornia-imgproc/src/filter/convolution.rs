use kornia_image::{Image, ImageError};

/// Apply a separable filter to an image.
pub fn separable_filter(
    src: &Image<f32, 3>,
    dst: &mut Image<f32, 3>,
    kernel_x: &[f32],
    kernel_y: &[f32],
) -> Result<(), ImageError> {
    let mut temp = vec![0.0; src.num_channels() * src.width() * src.height()];

    let norm_x = kernel_x.iter().sum::<f32>();
    let norm_y = kernel_y.iter().sum::<f32>();

    // apply the horizontal filter
    for y in 0..src.height() {
        for x in 0..src.width() {
            for c in 0..src.num_channels() {
                let temp_idx = (y * src.width() + x) * src.num_channels() + c;
                let mut sum = 0.0;
                let half_kernel = kernel_x.len() / 2;
                for k in 0..kernel_x.len() {
                    let x_pos = x + k - half_kernel;
                    if x_pos > 0 && x_pos < src.width() {
                        let src_idx = (y * src.width() + x_pos) * src.num_channels() + c;
                        sum += src.as_slice()[src_idx] * kernel_x[k];
                    }
                }
                temp[temp_idx] = sum / norm_x;
            }
        }
    }

    // apply the vertical filter
    for x in 0..src.width() {
        for y in 0..src.height() {
            for c in 0..src.num_channels() {
                let dst_idx = (y * src.width() + x) * src.num_channels() + c;
                let mut sum = 0.0;
                let half_kernel = kernel_y.len() / 2;
                for k in 0..kernel_y.len() {
                    let y_pos = y + k - half_kernel;
                    if y_pos > 0 && y_pos < src.height() {
                        let temp_idx = (y_pos * src.width() + x) * src.num_channels() + c;
                        sum += temp[temp_idx] * kernel_y[k];
                    }
                }
                dst.as_slice_mut()[dst_idx] = sum / norm_y;
            }
        }
    }
    Ok(())
}
