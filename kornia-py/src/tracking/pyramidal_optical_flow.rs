use crate::pyramid::build_gaussian_pyramid;
use crate::lucas_kanade::lucas_kanade_optical_flow;
use image::GrayImage;

/// Compute Pyramidal Lucas-Kanade Optical Flow.
pub fn pyramidal_lucas_kanade(
    img1: &GrayImage, img2: &GrayImage, levels: usize, window_size: usize,
) -> Vec<(f32, f32, f32, f32)> {
    let pyr1 = build_gaussian_pyramid(img1, levels, 1.5);
    let pyr2 = build_gaussian_pyramid(img2, levels, 1.5);
    
    let mut flow = vec![];

    for (lvl, (img1_lvl, img2_lvl)) in pyr1.iter().zip(pyr2.iter()).enumerate().rev() {
        let scale = 2.0f32.powi(lvl as i32);
        let lk_flow = lucas_kanade_optical_flow(img1_lvl, img2_lvl, window_size);

        for (pos, vel) in lk_flow {
            let x = pos.x * scale;
            let y = pos.y * scale;
            let u = vel.x * scale;
            let v = vel.y * scale;
            flow.push((x, y, u, v));
        }
    }

    flow
}
