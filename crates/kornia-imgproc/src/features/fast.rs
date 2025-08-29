use kiddo::{ImmutableKdTree, SquaredEuclidean};
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
#[derive(Debug, Clone, Copy)]
pub struct PixelIndex {
    /// TODO
    pub x: usize,
    /// TODO
    pub y: usize,
    /// TODO
    pub index: usize,
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
            lower_threshold = curr_pixel - threshold;
            upper_threshold = curr_pixel + threshold;

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
                    curr_response += circle_intensities[m] - curr_pixel;
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
) -> Result<Vec<[f32; 2]>, ImageError> {
    let border_width = min_distance;

    // TODO: Avoid this step by taking this value as parameter
    // let threshold = src
    //     .as_slice()
    //     .iter()
    //     .copied()
    //     .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater))
    //     .unwrap_or_default();

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
) -> Vec<[f32; 2]> {
    let src_size = src.size();

    let coords: Vec<_> = mask
        .as_slice()
        .iter()
        .enumerate()
        .filter(|&(_, &value)| value)
        .map(|(i, _)| {
            let y = (i / src_size.width) as f32;
            let x = (i % src_size.width) as f32;

            [y, x]
        })
        .collect();

    if min_distance > 1 {
        unimplemented!()
    }

    coords
}

/// TODO
pub fn corner_peaks<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    min_distance: usize,
    threshold: u8,
) -> Result<Vec<[f32; 2]>, ImageError> {
    let coords = peak_local_max(src, min_distance, threshold)?;

    let tree = ImmutableKdTree::new_from_slice(&coords);

    let min_distance = min_distance as f32;
    let mut rejected = std::collections::HashSet::new();
    let mut result = Vec::new();

    for (i, pixel) in coords.iter().enumerate() {
        let i = i as u64;

        if rejected.contains(&i) {
            continue;
        }

        let neighbors = tree.within::<SquaredEuclidean>(pixel, min_distance);
        for neighbor_idx in neighbors {
            let neighbor_idx = neighbor_idx.item;
            if neighbor_idx != i {
                rejected.insert(neighbor_idx);
            }
        }

        result.push(*pixel);
    }

    Ok(result)
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
