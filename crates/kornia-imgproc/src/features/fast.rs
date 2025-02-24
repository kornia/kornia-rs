use kornia_image::{Image, ImageError};
use rayon::prelude::*;

// Define all 16 points in the Bresenham circle (x, y)
const CIRCLE_OFFSETS: [(i32, i32); 16] = [
    (0, -3),  // 1
    (1, -3),  // 2
    (2, -2),  // 3
    (3, -1),  // 4
    (3, 0),   // 5
    (3, 1),   // 6
    (2, 2),   // 7
    (1, 3),   // 8
    (0, 3),   // 9
    (-1, 3),  // 10
    (-2, 2),  // 11
    (-3, 1),  // 12
    (-3, 0),  // 13
    (-3, -1), // 14
    (-2, -2), // 15
    (-1, -3), // 16
];

/// Fast feature detector
pub fn fast_feature_detector(
    src: &Image<u8, 1>,
    threshold: u8,
    arc_length: u8,
) -> Result<Vec<[i32; 2]>, ImageError> {
    let (cols, rows) = (src.cols() as i32, src.rows() as i32);
    let src_data = src.as_slice();

    // Process rows in parallel
    let keypoints = (3..(rows - 3) as i32)
        .into_par_iter()
        .flat_map(|y| {
            let row_start_idx = y * cols;
            let mut row_keypoints = Vec::new();

            for x in 3..(cols - 3) as i32 {
                if is_fast_corner(src_data, x, y, cols, row_start_idx, threshold, arc_length) {
                    row_keypoints.push([x, y]);
                }
            }

            row_keypoints
        })
        .collect();

    Ok(keypoints)
}

fn is_fast_corner(
    src: &[u8],
    x: i32,
    y: i32,
    cols: i32,
    row_start_idx: i32,
    threshold: u8,
    arc_length: u8,
) -> bool {
    let current_idx = row_start_idx + x;
    let center_pixel = unsafe { *src.get_unchecked(current_idx as usize) };
    let lower_threshold = &center_pixel.saturating_sub(threshold);
    let upper_threshold = &center_pixel.saturating_add(threshold);

    // Helper to get pixel value efficiently with unchecked access
    let get_pixel_from_offset = |off_idx: usize| unsafe {
        let (off_y, off_x) = CIRCLE_OFFSETS[off_idx];
        src.get_unchecked(((y + off_y) * cols + (x + off_x)) as usize)
    };

    // Fast rejection test - check if at least 3 of the 4 high-speed test points are different enough
    let p1 = get_pixel_from_offset(0);
    let p5 = get_pixel_from_offset(4);
    let p9 = get_pixel_from_offset(8);
    let p13 = get_pixel_from_offset(12);

    let m0 = (p1 < lower_threshold && p5 < lower_threshold)
        || (p5 < lower_threshold && p9 < lower_threshold)
        || (p9 < lower_threshold && p13 < lower_threshold)
        || (p13 < lower_threshold && p1 < lower_threshold);

    let m1 = (p1 > upper_threshold || p5 > upper_threshold)
        || (p9 > upper_threshold || p13 > upper_threshold)
        || (p13 > upper_threshold || p1 > upper_threshold)
        || (p1 > upper_threshold || p5 > upper_threshold);

    if !m0 && !m1 {
        return false;
    }

    // check the remaining pixels
    let p2 = get_pixel_from_offset(1);
    let p3 = get_pixel_from_offset(2);
    let p4 = get_pixel_from_offset(3);
    let p6 = get_pixel_from_offset(5);
    let p7 = get_pixel_from_offset(6);
    let p8 = get_pixel_from_offset(7);
    let p10 = get_pixel_from_offset(9);
    let p11 = get_pixel_from_offset(10);
    let p12 = get_pixel_from_offset(11);
    let p14 = get_pixel_from_offset(13);
    let p15 = get_pixel_from_offset(14);
    let p16 = get_pixel_from_offset(15);
    let pixels = [
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16,
    ];

    let mut consecutive_brighter = 0u8;
    let mut consecutive_darker = 0u8;

    for pixel in pixels {
        if pixel > upper_threshold {
            consecutive_brighter += 1;
            consecutive_darker = 0;
        } else if pixel < lower_threshold {
            consecutive_darker += 1;
            consecutive_brighter = 0;
        } else {
            consecutive_brighter = 0;
            consecutive_darker = 0;
        }

        if consecutive_brighter >= arc_length || consecutive_darker >= arc_length {
            return true;
        }
    }

    false
}

/// Fast feature detector with non-maximum suppression
pub fn fast_feature_detector_nms(
    src: &Image<u8, 1>,
    threshold: u8,
    arc_length: u8,
    nms_radius: usize,
) -> Result<Vec<[i32; 2]>, ImageError> {
    // First detect all keypoints
    let keypoints = fast_feature_detector(src, threshold, arc_length)?;

    if keypoints.is_empty() {
        return Ok(Vec::new());
    }

    // Calculate scores for each keypoint
    let mut keypoints_with_scores: Vec<([i32; 2], u32)> = keypoints
        .par_iter() // Use parallel iterator for score calculation
        .map(|&[x, y]| {
            let score = calculate_fast_score(src, x as usize, y as usize, threshold);
            ([x, y], score)
        })
        .collect();

    // Sort by score in descending order
    keypoints_with_scores.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    // Apply non-maximum suppression
    let mut result = Vec::with_capacity(keypoints_with_scores.len() / 4); // Estimate capacity
    let mut suppressed = vec![false; keypoints_with_scores.len()];
    let nms_radius_squared = nms_radius * nms_radius;

    for i in 0..keypoints_with_scores.len() {
        if suppressed[i] {
            continue;
        }

        let [x_i, y_i] = keypoints_with_scores[i].0;
        result.push([x_i, y_i]);

        // Use rayon to parallelize the suppression check for large keypoint sets
        if keypoints_with_scores.len() > 1000 {
            (i + 1..keypoints_with_scores.len())
                .into_iter()
                .for_each(|j| {
                    if !suppressed[j] {
                        let [x_j, y_j] = keypoints_with_scores[j].0;
                        let dx = (x_i as isize - x_j as isize).abs() as usize;
                        let dy = (y_i as isize - y_j as isize).abs() as usize;

                        // Use squared distance for faster comparison
                        if dx * dx + dy * dy <= nms_radius_squared {
                            suppressed[j] = true;
                        }
                    }
                });
        } else {
            // For smaller sets, sequential processing is faster due to less overhead
            for j in (i + 1)..keypoints_with_scores.len() {
                if !suppressed[j] {
                    let [x_j, y_j] = keypoints_with_scores[j].0;
                    let dx = (x_i as isize - x_j as isize).abs() as usize;
                    let dy = (y_i as isize - y_j as isize).abs() as usize;

                    // Use squared distance for faster comparison
                    if dx * dx + dy * dy <= nms_radius_squared {
                        suppressed[j] = true;
                    }
                }
            }
        }
    }

    Ok(result)
}

fn calculate_fast_score(src: &Image<u8, 1>, x: usize, y: usize, threshold: u8) -> u32 {
    // Define all 16 points in the Bresenham circle (y, x)
    const CIRCLE_OFFSETS: [(isize, isize); 16] = [
        (0, -3),  // 1 (12 o'clock)
        (1, -3),  // 2
        (2, -2),  // 3
        (3, -1),  // 4
        (3, 0),   // 5
        (3, 1),   // 6
        (2, 2),   // 7
        (1, 3),   // 8
        (0, 3),   // 9 (6 o'clock)
        (-1, 3),  // 10
        (-2, 2),  // 11
        (-3, 1),  // 12
        (-3, 0),  // 13
        (-3, -1), // 14
        (-2, -2), // 15
        (-1, -3), // 16
    ];

    let cols = src.cols();
    let center_pixel = src.get_pixel(x, y, 0).unwrap();

    // Helper to get pixel value safely
    let get_pixel = |off_y: isize, off_x: isize| {
        let new_y = y as isize + off_y;
        let new_x = x as isize + off_x;

        if new_y >= 0 && new_y < src.rows() as isize && new_x >= 0 && new_x < cols as isize {
            src.get_pixel(new_x as usize, new_y as usize, 0).unwrap()
        } else {
            center_pixel
        }
    };

    // Calculate score as sum of absolute differences
    let mut score: u32 = 0;

    for &(dy, dx) in &CIRCLE_OFFSETS {
        let pixel = get_pixel(dy, dx);
        let diff = if pixel > center_pixel {
            pixel - center_pixel
        } else {
            center_pixel - pixel
        };

        // Only count differences that are greater than the threshold
        if diff > threshold {
            score += diff as u32;
        }
    }

    score
}
