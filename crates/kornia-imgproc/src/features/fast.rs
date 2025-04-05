use kornia_image::{Image, ImageError};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

// Structure to represent a feature point with its score and coordinates. Useful for NMS.
#[derive(Copy, Clone, Eq, PartialEq)]
struct FeaturePoint {
    score: i32,
    x: usize,
    y: usize,
}

impl Ord for FeaturePoint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score)
    }
}

impl PartialOrd for FeaturePoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Fast feature detector
///
/// # Arguments
///
/// * `src` - The source image as Gray8 image.
/// * `threshold` - The threshold for the fast feature detector.
/// * `arc_length` - The total number of consecutive pixels in the Bresenham circle that must be brighter or darker than the center pixel.
///
/// # Returns
///
/// A vector containing the coordinates of the detected keypoints.
pub fn fast_feature_detector(
    src: &Image<u8, 1>,
    threshold: u8,
    arc_length: u8,
) -> Result<Vec<[i32; 2]>, ImageError> {
    let (cols, rows) = (src.cols() as i32, src.rows() as i32);

    // Precompute the offsets for the Bresenham circle
    let offsets = [
        -3 * cols,     // 1
        -3 * cols + 1, // 2
        -2 * cols + 2, // 3
        -cols + 3,     // 4
        cols + 3,      // 5
        cols + 3,      // 6
        2 * cols + 2,  // 7
        3 * cols + 1,  // 8
        3 * cols,      // 9
        3 * cols - 1,  // 10
        2 * cols - 2,  // 11
        cols - 3,      // 12
        -cols - 3,     // 13
        -2 * cols - 2, // 14
        -3 * cols - 1, // 15
        -3 * cols,     // 16
    ];

    // Process rows in parallel
    let keypoints = (3..rows - 3)
        .into_par_iter()
        .flat_map(|y| {
            let row_start_idx = y * cols;
            let mut row_keypoints = Vec::new();

            for x in 3..cols - 3 {
                if is_fast_corner(
                    src.as_slice(),
                    row_start_idx + x,
                    offsets,
                    threshold,
                    arc_length,
                ) {
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
    pixel_idx: i32,
    offsets: [i32; 16],
    threshold: u8,
    arc_length: u8,
) -> bool {
    let center_pixel = unsafe { *src.get_unchecked(pixel_idx as usize) };
    let lower_threshold = &center_pixel.saturating_sub(threshold);
    let upper_threshold = &center_pixel.saturating_add(threshold);

    // Helper to get pixel value efficiently with unchecked access
    let get_pixel_from_offset =
        |off_idx: usize| unsafe { src.get_unchecked((pixel_idx + offsets[off_idx]) as usize) };

    // Fast rejection test - check if at least 3 of the 4 high-speed test points are different enough
    let p1 = get_pixel_from_offset(0);
    let p5 = get_pixel_from_offset(4);
    let p9 = get_pixel_from_offset(8);
    let p13 = get_pixel_from_offset(12);

    let m0 = !(p5 >= lower_threshold || p1 >= lower_threshold && p9 >= lower_threshold);
    let m1 = !(p5 <= upper_threshold || p1 <= upper_threshold && p9 <= upper_threshold);

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

/// Calculate the FAST corner score for a pixel using Sum of Absolute Differences (SAD).
/// Based on https://www.edwardrosten.com/work/rosten_2006_machine.pdf.
/// Returns a tuple of (is_corner: bool, score: i32)
#[inline(always)]
fn get_fast_corner_score_inner(
    src: &[u8],
    pixel_idx: i32,
    offsets: [i32; 16],
    threshold: u8,
    arc_length: u8,
) -> (bool, i32) {
    let center_pixel = unsafe { *src.get_unchecked(pixel_idx as usize) };
    let lower_threshold = &center_pixel.saturating_sub(threshold);
    let upper_threshold = &center_pixel.saturating_add(threshold);

    // Helper to get pixel value efficiently with unchecked access
    let get_pixel_from_offset =
        |off_idx: usize| unsafe { src.get_unchecked((pixel_idx + offsets[off_idx]) as usize) };

    // Fast rejection test - check if at least 3 of the 4 high-speed test points are different enough
    let p1 = get_pixel_from_offset(0);
    let p5 = get_pixel_from_offset(4);
    let p9 = get_pixel_from_offset(8);
    let p13 = get_pixel_from_offset(12);

    let m0 = (p1 >= lower_threshold) as u8 + 
                (p5 >= lower_threshold) as u8 + 
                (p9 >= lower_threshold) as u8 + 
                (p13 >= lower_threshold) as u8;

    let m1 = (p1 <= upper_threshold) as u8 + 
                (p5 <= upper_threshold) as u8 + 
                (p9 <= upper_threshold) as u8 + 
                (p13 <= upper_threshold) as u8;

    if m0 < 3 && m1 < 3 {
        return (false, 0);
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
    ]; // Values are repeated to handle circular indexing
    
    // Use a bitmask of size N, and shift it.
    let mut bright_bitmask = 0u16;
    let mut dark_bitmask = 0u16;
    for (i, &val) in pixels.iter().enumerate() {
        if val > upper_threshold {
            bright_bitmask |= 1 << i;
        }

        if val < lower_threshold {
            dark_bitmask |= 1 << i;
        }
    }

    let window_mask = (1u16 << arc_length) - 1; // Create a bitmask of length arc_length

    // Now we can use the bitmask to determine valid segment
    let mut is_corner = false;
    let mut shift_amount: usize = 0;
    for shift in 0..16 {
        let curr_window_mask = window_mask.rotate_left(shift);
        let bright_window_bits = bright_bitmask & curr_window_mask;
        let dark_window_bits = dark_bitmask & curr_window_mask;

        if bright_window_bits.count_ones() >= arc_length as u32 || dark_window_bits.count_ones() >= arc_length as u32{
            is_corner = true;
            shift_amount = shift as usize;
            break;
        }
    }

    if !is_corner {
        return (false, 0); // Not a corner found
    }

    // Sum of absolute differences for the corner score. 
    let mut score = 0i32;
    for offset in shift_amount..shift_amount + arc_length as usize {
        let curr_idx = offset.rem_euclid(16); 
        score += (center_pixel.abs_diff(*pixels[curr_idx]) - threshold) as i32;
    }

    (true, score)
}

#[target_feature(enable = "avx2")]
unsafe fn get_fast_corner_score_avx2(
    src: &[u8],
    pixel_idx: i32,
    offsets: [i32; 16],
    threshold: u8,
    arc_length: u8,
) -> (bool, i32) {
    get_fast_corner_score_inner(src, pixel_idx, offsets, threshold, arc_length)
}

/// Wrapper function for FAST corner score calculation.
pub fn get_fast_corner_score(
    src: &[u8],
    pixel_idx: i32,
    offsets: [i32; 16],
    threshold: u8,
    arc_length: u8,
) -> (bool, i32) {
    if is_x86_feature_detected!("avx2") {
        unsafe { get_fast_corner_score_avx2(src, pixel_idx, offsets, threshold, arc_length) }
    } else {
        get_fast_corner_score_inner(src, pixel_idx, offsets, threshold, arc_length)
    }
}

/// Fast feature detector with Non-Maximum Suppression (NMS)
/// Goal is to replace the previous `fast_feature_detector` implementation.
///
/// # Arguments
///
/// * `src` - The source image as Gray8 image.
/// * `threshold` - The threshold for the fast feature detector.
/// * `arc_length` - The total number of consecutive pixels in the Bresenham circle that must be brighter or darker than the center pixel.
/// * `nms` - A boolean flag to enable or disable Non-Maximum Suppression.
/// # Returns
///
/// A vector containing the coordinates of the detected keypoints.
#[inline(always)]
pub fn fast_feature_nms_detector_inner(
    src: &Image<u8, 1>,
    threshold: u8,
    arc_length: u8,
    nms: bool
) -> Vec<[i32; 2]> {
    let (cols, rows) = (src.cols() as i32, src.rows() as i32);

    // Precompute the offsets for the Bresenham circle
    let offsets = [
        -3 * cols,     // 1
        -3 * cols + 1, // 2
        -2 * cols + 2, // 3
        -cols + 3,     // 4
        3,             // 5
        cols + 3,      // 6
        2 * cols + 2,  // 7
        3 * cols + 1,  // 8
        3 * cols,      // 9
        3 * cols - 1,  // 10
        2 * cols - 2,  // 11
        cols - 3,      // 12
        -3,            // 13
        -cols - 3,     // 14
        -2 * cols - 2, // 15
        -3 * cols - 1,     // 16
    ];

    // Process rows in parallel
    let (kp1, scores): (Vec<[i32; 2]>, Vec<i32>) = (3..rows - 3)
        .into_par_iter()
        .flat_map(|y| {
            let row_start_idx = y * cols;
            let mut row_keypoints = Vec::new();
            let mut kp_scores = Vec::new();

            for x in 3..cols - 3 {
                let (is_corner, score) = get_fast_corner_score(
                    src.as_slice(),
                    row_start_idx + x,
                    offsets,
                    threshold,
                    arc_length,
                );
                if is_corner {
                    // score_img[(y * cols + x) as usize] = score; 
                    row_keypoints.push([x, y]);
                    kp_scores.push(score);
                }
            }
            
            (row_keypoints, kp_scores)
        })
        .unzip();

    // Exit early if NMS disabled
    if !nms {
        return kp1;
    }
    
    let mut heap = BinaryHeap::with_capacity(kp1.len());
    for (point, score) in kp1.into_iter().zip(scores) {
        heap.push(FeaturePoint{
            score,
            x: point[0] as usize,
            y: point[1] as usize,
        });
    }

    let mut ignore_map = vec![false; (rows * cols) as usize];
    let mut kp2 = Vec::new();
    while let Some(point) = heap.pop()
    {
        let idx = point.y * cols as usize + point.x;
        if ignore_map[idx] {
            continue; // This point has been suppressed
        }

        // Keep point
        kp2.push([point.x as i32, point.y as i32]);

        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue; // skip the center pixel
                }
                let nx = point.x as i32 + dx;
                let ny = point.y as i32 + dy;
                let ignore_idx = (ny * cols + nx) as usize;
                ignore_map[ignore_idx] = true; // Mark this point as suppressed
            }
        }
    }
    
    kp2
}

#[target_feature(enable = "avx2")]
unsafe fn fast_feature_nms_detector_avx2(
    src: &Image<u8, 1>,
    threshold: u8,
    arc_length: u8,
    nms: bool
) -> Vec<[i32; 2]> {
    fast_feature_nms_detector_inner(src, threshold, arc_length, nms)
}

/// Wrapper for FAST feature detector with Non-Maximum Suppression (NMS). 
pub fn fast_feature_nms_detector(
    src: &Image<u8, 1>,
    threshold: u8,
    arc_length: u8,
    nms: bool
) -> Vec<[i32; 2]> {
    if is_x86_feature_detected!("avx2") {
        unsafe { fast_feature_nms_detector_avx2(src, threshold, arc_length, nms) }
    } else {
        fast_feature_nms_detector_inner(src, threshold, arc_length, nms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::Image;

    #[test]
    fn test_fast_feature_detector() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let img = Image::new(
            [7, 7].into(),
            vec![
                50,  50,  50,  50,  50,  50,  50,
                50,  50,  50,  50,  50,  50,  50,
                50,  50,  50, 200,  50,  50,  50,
                50,  50, 200, 200, 200,  50,  50,
                50,  50,  50, 200,  50,  50,  50,
                50,  50,  50,  50,  50,  50,  50,
                50,  50,  50,  50,  50,  50,  50,
            ],
        )?;
        let expected_keypoints = vec![[3, 3]];
        let keypoints = fast_feature_detector(&img, 100, 9)?;
        assert_eq!(keypoints.len(), expected_keypoints.len());
        assert_eq!(keypoints, expected_keypoints);
        Ok(())
    }

    #[test]
    fn test_fast_feature_nms_detector_up() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let img = Image::new(
            [7, 7].into(),
            vec![
                50,  50,  50,  50,  50,  50,  50,
                50,  50,  50,  50,  50,  50,  50,
                50,  50,  50,  50,  50,  50,  50,
                50,  50,  50, 200,  50,  50,  50,
               200, 200, 200, 200, 200, 200, 200,
               200, 200, 200, 200, 200, 200, 200,
               200, 200, 200, 200, 200, 200, 200,
            ],
        )?;
        let expected_keypoints = vec![[3, 3]];
        let keypoints = fast_feature_nms_detector(&img, 100, 9, true);
        assert_eq!(keypoints.len(), expected_keypoints.len());
        assert_eq!(keypoints, expected_keypoints);
        
        Ok(())
    }

    #[test]
    fn test_fast_feature_nms_detector_left() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let img = Image::new(
            [7, 7].into(),
            vec![
               200, 200, 200,  50,  50,  50,  50,
               200, 200, 200,  50,  50,  50,  50,
               200, 200, 200,  50,  50,  50,  50,
               200, 200, 200, 200,  50,  50,  50,
               200, 200, 200,  50,  50,  50,  50,
               200, 200, 200,  50,  50,  50,  50,
               200, 200, 200,  50,  50,  50,  50,
            ],
        )?;
        let expected_keypoints = vec![[3, 3]];
        let keypoints = fast_feature_nms_detector(&img, 100, 9, true);
        assert_eq!(keypoints.len(), expected_keypoints.len());
        assert_eq!(keypoints, expected_keypoints);
        
        Ok(())
    }

    #[test]
    fn test_fast_feature_nms_detector_right() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let img = Image::new(
            [7, 7].into(),
            vec![
               50,  50,  50,  50, 200, 200, 200,
               50,  50,  50,  50, 200, 200, 200,
               50,  50,  50,  50, 200, 200, 200,
               50,  50,  50, 200, 200, 200, 200,
               50,  50,  50,  50, 200, 200, 200,
               50,  50,  50,  50, 200, 200, 200,
               50,  50,  50,  50, 200, 200, 200,
            ],
        )?;
        let expected_keypoints = vec![[3, 3]];
        let keypoints = fast_feature_nms_detector(&img, 100, 9, true);
        assert_eq!(keypoints.len(), expected_keypoints.len());
        assert_eq!(keypoints, expected_keypoints);
        
        Ok(())
    }

    #[test]
    fn test_fast_feature_nms_detector_down() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let img = Image::new(
            [7, 7].into(),
            vec![
              200, 200, 200, 200, 200, 200, 200,
              200, 200, 200, 200, 200, 200, 200,
              200, 200, 200, 200, 200, 200, 200,
               50,  50,  50, 200,  50,  50,  50,
               50,  50,  50,  50,  50,  50,  50,
               50,  50,  50,  50,  50,  50,  50,
               50,  50,  50,  50,  50,  50,  50,
            ],
        )?;
        let expected_keypoints = vec![[3, 3]];
        let keypoints = fast_feature_nms_detector(&img, 100, 9, true);
        assert_eq!(keypoints.len(), expected_keypoints.len());
        assert_eq!(keypoints, expected_keypoints);
        
        Ok(())
    }
}
