use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use kornia_tensor::CpuAllocator;
use rayon::prelude::*;

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
pub fn fast_feature_detector<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
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

/// TODO
#[derive(Clone, Copy, PartialEq)]
pub enum PixelType {
    /// TODO
    Brighter,
    /// TODO
    Darker,
    /// TODO
    Similar,
}

/// TODO
pub fn corner_fast<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    threshold: u8,
    arc_length: usize,
) -> Result<Image<u8, 1, CpuAllocator>, ImageError> {
    let src_slice = src.as_slice();

    let mut speed_sum_b: i32;
    let mut speed_sum_d: i32;
    let mut curr_pixel: u8;
    let mut ring_pixel: u8;
    let mut lower_threshold: u8;
    let mut upper_threshold: u8;

    let mut corner_response = vec![0; src_slice.len()];

    let rp = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
    let cp = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];
    let mut bins = [PixelType::Similar; 16];
    let mut circle_intensities = [0u8; 16];

    let mut curr_response: u8;

    for y in 3..src.height() - 3 {
        let iy = y * src.width();

        for x in 3..src.width() - 3 {
            let ix = iy + x;

            curr_pixel = src_slice[ix];
            lower_threshold = curr_pixel.saturating_sub(threshold);
            upper_threshold = curr_pixel.saturating_add(threshold);

            if arc_length >= 12 {
                speed_sum_b = 0;
                speed_sum_d = 0;

                for k in [0, 4, 8, 12] {
                    let ik = ((y as isize + rp[k]) * src.width() as isize + (x as isize + cp[k]))
                        as usize;

                    ring_pixel = src_slice[ik];
                    if ring_pixel > upper_threshold {
                        speed_sum_b += 1;
                    } else if ring_pixel < lower_threshold {
                        speed_sum_d += 1;
                    }
                }

                if speed_sum_d < 3 && speed_sum_b < 3 {
                    continue;
                }
            }

            for k in 0..16 {
                let ik =
                    ((y as isize + rp[k]) * src.width() as isize + (x as isize + cp[k])) as usize;

                circle_intensities[k] = src_slice[ik];
                if circle_intensities[k] > upper_threshold {
                    bins[k] = PixelType::Brighter;
                } else if circle_intensities[k] < lower_threshold {
                    bins[k] = PixelType::Darker;
                } else {
                    bins[k] = PixelType::Similar;
                }
            }

            // Test for bright pixels
            curr_response = corner_fast_response(
                curr_pixel,
                &circle_intensities,
                &bins,
                PixelType::Brighter,
                arc_length,
            );

            // Test for dark pixels
            if curr_pixel == 0 {
                curr_response = corner_fast_response(
                    curr_pixel,
                    &circle_intensities,
                    &bins,
                    PixelType::Darker,
                    arc_length,
                );
            }

            corner_response[ix] = curr_response;
        }
    }

    return Ok(Image::new(src.size(), corner_response, CpuAllocator)?);
}

/// TODO
fn corner_fast_response(
    curr_pixel: u8,
    circle_intensities: &[u8; 16],
    bins: &[PixelType; 16],
    state: PixelType,
    n: usize,
) -> u8 {
    let mut consecutive_count = 0;
    let mut curr_response: u8;

    for l in 0..15 + n {
        if bins[l % 16] == state {
            consecutive_count += 1;
            if consecutive_count == n {
                curr_response = 0;
                for m in 0..16 {
                    curr_response = curr_response
                        .saturating_add(circle_intensities[m].saturating_sub(curr_pixel));
                }

                return curr_response;
            }
        } else {
            consecutive_count = 0;
        }
    }

    return 0;
}

/// TODO
pub fn peak_local_max<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    min_distance: usize,
    threshold: u8,
) -> Result<Vec<[usize; 2]>, ImageError> {
    let border_width = min_distance;

    let mut mask = get_peak_mask(src, threshold)?;
    exclude_border(&mut mask, border_width);

    let coordinates = get_high_intensity_peaks(src, &mask, min_distance);
    Ok(coordinates)
}

fn get_peak_mask<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    threshold: u8,
) -> Result<Image<bool, 1, CpuAllocator>, ImageError> {
    let src_slice = src.as_slice();

    let mask = src_slice.iter().map(|px| *px > threshold).collect();
    Ok(Image::new(src.size(), mask, CpuAllocator)?)
}

fn exclude_border<A: ImageAllocator>(label: &mut Image<bool, 1, A>, border_width: usize) {
    let label_size = label.size();
    let label_slice = label.as_slice_mut();

    for y in 0..label_size.height {
        for x in 0..label_size.width {
            if x < border_width
                || x >= label_size.width.saturating_sub(border_width)
                || y < border_width
                || y >= label_size.height.saturating_sub(border_width)
            {
                label_slice[y * label_size.width + x] = false;
            }
        }
    }
}

fn get_high_intensity_peaks<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 1, A1>,
    mask: &Image<bool, 1, A2>,
    min_distance: usize,
) -> Vec<[usize; 2]> {
    let src_size = src.size();
    let width = src_size.width;
    let height = src_size.height;

    let coords_intensity: Vec<[usize; 2]> = mask
        .as_slice()
        .iter()
        .enumerate()
        .filter(|&(_, &value)| value)
        .map(|(i, _)| {
            let y = i / width;
            let x = i % width;
            [y, x]
        })
        .collect();

    let mut taken = vec![vec![false; width]; height];
    let mut result = Vec::new();

    for coord in coords_intensity {
        let y = coord[0];
        let x = coord[1];

        // If this location is already suppressed, skip
        if taken[y][x] {
            continue;
        }

        // Accept this peak
        result.push([y, x]);

        // Suppress all within min_distance
        let y0 = y - min_distance;
        let y1 = (y + min_distance + 1).min(height);
        let x0 = x - min_distance;
        let x1 = (x + min_distance + 1).min(width);

        for yy in y0..y1 {
            for xx in x0..x1 {
                // Chebyshev distance (max norm)
                if (yy as isize - y as isize)
                    .abs()
                    .max((xx as isize - x as isize).abs())
                    < min_distance as isize
                {
                    taken[yy][xx] = true;
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use crate::color::gray_from_rgb_u8;

    use super::*;
    use kornia_image::Image;
    use kornia_io::jpeg::read_image_jpeg_rgb8;
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_fast_feature_detector() -> Result<(), Box<dyn std::error::Error>> {
        #[rustfmt::skip]
        let img = read_image_jpeg_rgb8("../../tests/data/dog.jpeg")?;
        let mut gray_img = Image::from_size_val(img.size(), 0, CpuAllocator)?;
        gray_from_rgb_u8(&img, &mut gray_img)?;

        let expected_keypoints = vec![
            [32, 86],
            [44, 80],
            [45, 175],
            [55, 76],
            [63, 183],
            [70, 88],
            [71, 169],
            [108, 125],
            [120, 64],
            [125, 164],
            [128, 92],
            [133, 177],
            [135, 161],
            [138, 96],
            [153, 104],
            [161, 148],
        ];

        const THRESHOLD: u8 = 30;
        let fast_response = corner_fast(&gray_img, THRESHOLD, 12)?;
        let keypoints = peak_local_max(&fast_response, 10, THRESHOLD)?;

        assert_eq!(keypoints.len(), expected_keypoints.len());
        assert_eq!(keypoints, expected_keypoints);
        Ok(())
    }
}
