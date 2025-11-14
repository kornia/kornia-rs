use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use rayon::prelude::*;

/// Detects corners using the FAST (Features from Accelerated Segment Test) algorithm.
///
/// FAST is a high-speed corner detection method that identifies keypoints by examining
/// a circular ring of pixels (Bresenham circle of radius 3) around each candidate point.
/// A pixel is considered a corner if a contiguous sequence of pixels in the circle are all
/// either brighter or darker than the center pixel by at least the specified threshold.
///
/// # Arguments
///
/// * `src` - The source grayscale image (single-channel u8).
/// * `threshold` - Intensity difference threshold. Pixels differing from the center by at
///                 least this value are considered for the arc test.
/// * `arc_length` - The minimum number of consecutive pixels in the 16-pixel Bresenham
///                  circle that must all be brighter or all darker than the center pixel.
///                  Common values: 9 for FAST-9, 12 for FAST-12.
///
/// # Returns
///
/// A vector of `[x, y]` coordinates representing detected corner keypoints.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::features::fast_feature_detector;
///
/// let image = Image::<u8, 1>::from_size_val(
///     ImageSize { width: 100, height: 100 },
///     128,
/// ).unwrap();
///
/// let keypoints = fast_feature_detector(&image, 20, 9).unwrap();
/// // keypoints contains detected corner locations
/// ```
///
/// # See also
///
/// * E. Rosten and T. Drummond, "Machine learning for high-speed corner detection,"
///   ECCV 2006.
pub fn fast_feature_detector<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    threshold: u8,
    arc_length: u8,
) -> Result<Vec<[i32; 2]>, ImageError> {
    let (cols, rows) = (src.cols() as i32, src.rows() as i32);

    // Precompute the offsets for the Bresenham circle
    let offsets = [
        -3 * cols,     // 1: (0, -3)
        -3 * cols + 1, // 2: (1, -3)
        -2 * cols + 2, // 3: (2, -2)
        -cols + 3,     // 4: (3, -1)
        3,             // 5: (3, 0)
        cols + 3,      // 6: (3, 1)
        2 * cols + 2,  // 7: (2, 2)
        3 * cols + 1,  // 8: (1, 3)
        3 * cols,      // 9: (0, 3)
        3 * cols - 1,  // 10: (-1, 3)
        2 * cols - 2,  // 11: (-2, 2)
        cols - 3,      // 12: (-3, 1)
        -3,            // 13: (-3, 0)
        -cols - 3,     // 14: (-3, -1)
        -2 * cols - 2, // 15: (-2, -2)
        -3 * cols - 1, // 16: (-1, -3)
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

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::Image;
    use kornia_tensor::CpuAllocator;

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
            CpuAllocator
        )?;
        let expected_keypoints = vec![[3, 3]];
        let keypoints = fast_feature_detector(&img, 100, 9)?;
        assert_eq!(keypoints.len(), expected_keypoints.len());
        assert_eq!(keypoints, expected_keypoints);
        Ok(())
    }
}
