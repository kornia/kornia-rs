use kornia_image::{Image, ImageError};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

/// Fast feature detector
pub fn fast_feature_detector(
    src: &Image<u8, 1>,
    threshold: u8,
) -> Result<Vec<[usize; 2]>, ImageError> {
    let mut keypoints = Vec::new();

    let (cols, rows) = (src.cols(), src.rows());
    let src_data = src.as_slice();

    for y in 3..(rows - 3) {
        for x in 3..(cols - 3) {
            if is_fast_corner(src_data, x, y, cols, threshold) {
                keypoints.push([x, y]);
            }
        }
    }

    Ok(keypoints)
}

fn is_fast_corner(src: &[u8], x: usize, y: usize, cols: usize, threshold: u8) -> bool {
    let current_idx = y * cols + x;
    let center_pixel = src[current_idx];

    let mut darker_count = 0;
    let mut brighter_count = 0;

    let offsets = [
        (-3, 0),
        (-3, 1),
        (-2, 2),
        (-1, 3),
        (0, 3),
        (1, 3),
        (2, 2),
        (3, 1),
        (3, 0),
        (3, -1),
        (2, -2),
        (1, -3),
        (0, -3),
        (-1, -3),
        (-2, -2),
        (-3, -1),
    ];

    for (dx, dy) in offsets.iter() {
        let nx = x as isize + dx;
        let ny = y as isize + dy;

        let neighbor_idx = ny * cols as isize + nx;
        let neighbor_pixel = src[neighbor_idx as usize];

        if neighbor_pixel <= center_pixel.wrapping_sub(threshold) {
            darker_count += 1;
        }

        if neighbor_pixel >= center_pixel.wrapping_add(threshold) {
            brighter_count += 1;
        }

        if darker_count >= threshold || brighter_count >= threshold {
            return true;
        }
    }

    false
}
