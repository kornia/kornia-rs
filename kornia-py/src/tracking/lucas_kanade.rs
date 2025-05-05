use glam::{Vec2, Mat2};
use kornia::filters::sobel;
use kornia::tensor::Tensor;
use tch::{Device, Kind};

/// Implements Lucas-Kanade Optical Flow using glam-rs
pub fn lucas_kanade(
    prev_img: &Tensor,
    next_img: &Tensor,
    points: &[(f32, f32)],
    win_size: usize,
) -> Vec<(f32, f32)> {
    let mut flow = Vec::new();

    for &(x, y) in points {
        let mut sum_dx = Vec2::ZERO;
        let mut sum_dy = Vec2::ZERO;

        for i in 0..win_size {
            for j in 0..win_size {
                // Calculate gradients using Kornia's Sobel filter
                let dx = sobel(prev_img, "x", true);
                let dy = sobel(prev_img, "y", true);

                let vel = Vec2::new(dx.double_value(&[i as i64, j as i64]) as f32, dy.double_value(&[i as i64, j as i64]) as f32);
                sum_dx += vel;
                sum_dy += vel;
            }
        }

        // Average the flow over the window
        let avg_flow = Vec2::new(sum_dx.x / win_size as f32, sum_dy.y / win_size as f32);
        flow.push((x + avg_flow.x, y + avg_flow.y));
    }

    flow
}
