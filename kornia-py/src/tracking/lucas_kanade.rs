use image::{GrayImage, Luma};
use nalgebra::{DMatrix, Vector2};

/// Compute optical flow using Lucas-Kanade method.
pub fn lucas_kanade_optical_flow(
    img1: &GrayImage, img2: &GrayImage, window_size: usize,
) -> Vec<(Vector2<f32>, Vector2<f32>)> {
    let mut flow_vectors = Vec::new();
    let half_window = (window_size / 2) as i32;

    for y in half_window..(img1.height() as i32 - half_window) {
        for x in half_window..(img1.width() as i32 - half_window) {
            let mut i_xx = 0.0;
            let mut i_xy = 0.0;
            let mut i_yy = 0.0;
            let mut i_xt = 0.0;
            let mut i_yt = 0.0;

            for wy in -half_window..=half_window {
                for wx in -half_window..=half_window {
                    let x_pos = (x + wx) as u32;
                    let y_pos = (y + wy) as u32;

                    let i_x = (img2.get_pixel(x_pos + 1, y_pos)[0] as f32 - img1.get_pixel(x_pos - 1, y_pos)[0] as f32) / 2.0;
                    let i_y = (img2.get_pixel(x_pos, y_pos + 1)[0] as f32 - img1.get_pixel(x_pos, y_pos - 1)[0] as f32) / 2.0;
                    let i_t = img2.get_pixel(x_pos, y_pos)[0] as f32 - img1.get_pixel(x_pos, y_pos)[0] as f32;

                    i_xx += i_x * i_x;
                    i_xy += i_x * i_y;
                    i_yy += i_y * i_y;
                    i_xt += i_x * i_t;
                    i_yt += i_y * i_t;
                }
            }

            let A = DMatrix::from_row_slice(2, 2, &[i_xx, i_xy, i_xy, i_yy]);
            let b = Vector2::new(-i_xt, -i_yt);

            if let Some(v) = A.try_inverse() {
                let flow = v * b;
                flow_vectors.push((Vector2::new(x as f32, y as f32), flow));
            }
        }
    }

    flow_vectors
}
