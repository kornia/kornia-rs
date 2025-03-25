use kornia::tensor::Tensor;
use glam::Vec2;
use kornia::filters::sobel_filter;  // Use Kornia’s sobel filter

/// Implements Lucas-Kanade Optical Flow using Kornia's Sobel filter
pub fn lucas_kanade(prev_img: &Tensor, next_img: &Tensor, win_size: usize) -> Vec<Vec2> {
    let mut flow = Vec::new();

    // Use Kornia’s sobel_filter for gradient calculation
    let grad_x = sobel_filter(prev_img, true);   // X-gradient
    let grad_y = sobel_filter(prev_img, false);  // Y-gradient

    for i in 0..win_size {
        for j in 0..win_size {
            let dx = grad_x.get(i).unwrap();
            let dy = grad_y.get(j).unwrap();

            let vx = dx.sum(0).item::<f32>();
            let vy = dy.sum(0).item::<f32>();

            flow.push(Vec2::new(vx, vy));
        }
    }

    flow
}
