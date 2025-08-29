use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use kornia_tensor::CpuAllocator;
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq)]
enum PixelType {
    Brighter,
    Darker,
    Similar,
}

/// TODO
#[derive(Clone)]
pub struct FastDetector {
    /// TODO
    pub threshold: u8,
    /// TODO
    pub min_distance: usize,
    /// TODO
    pub arc_length: usize,
    corner_response: Image<u8, 1, CpuAllocator>,
    mask: Image<bool, 1, CpuAllocator>,
    bins: [PixelType; 16],
    circle_intensities: [u8; 16],
}

impl FastDetector {
    /// TODO
    pub fn new(
        image_size: ImageSize,
        threshold: u8,
        arc_length: usize,
        min_distance: usize,
    ) -> Result<Self, ImageError> {
        Ok(Self {
            threshold,
            min_distance,
            arc_length,
            corner_response: Image::from_size_val(image_size, 0, CpuAllocator)?,
            mask: Image::from_size_val(image_size, false, CpuAllocator)?,
            bins: [PixelType::Similar; 16],
            circle_intensities: [0; 16],
        })
    }

    /// TODO
    pub fn corner_fast<A: ImageAllocator>(
        &mut self,
        src: &Image<u8, 1, A>,
    ) -> &Image<u8, 1, CpuAllocator> {
        let src_slice = src.as_slice();

        let mut speed_sum_b: i32;
        let mut speed_sum_d: i32;
        let mut curr_pixel: u8;
        let mut ring_pixel: u8;
        let mut lower_threshold: u8;
        let mut upper_threshold: u8;

        let corner_response = self.corner_response.as_slice_mut();

        const RP: [isize; 16] = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
        const CP: [isize; 16] = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];

        let mut curr_response: u8;

        for y in 3..src.height() - 3 {
            let iy = y * src.width();

            for x in 3..src.width() - 3 {
                let ix = iy + x;

                curr_pixel = src_slice[ix];
                lower_threshold = curr_pixel.saturating_sub(self.threshold);
                upper_threshold = curr_pixel.saturating_add(self.threshold);

                if self.arc_length >= 12 {
                    speed_sum_b = 0;
                    speed_sum_d = 0;

                    for k in [0, 4, 8, 12] {
                        let ik = ((y as isize + RP[k]) * src.width() as isize
                            + (x as isize + CP[k])) as usize;

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
                    let ik = ((y as isize + RP[k]) * src.width() as isize + (x as isize + CP[k]))
                        as usize;

                    self.circle_intensities[k] = src_slice[ik];
                    if self.circle_intensities[k] > upper_threshold {
                        self.bins[k] = PixelType::Brighter;
                    } else if self.circle_intensities[k] < lower_threshold {
                        self.bins[k] = PixelType::Darker;
                    } else {
                        self.bins[k] = PixelType::Similar;
                    }
                }

                // Test for bright pixels
                curr_response = corner_fast_response(
                    curr_pixel,
                    &self.circle_intensities,
                    &self.bins,
                    PixelType::Brighter,
                    self.arc_length,
                );

                // Test for dark pixels
                if curr_pixel == 0 {
                    curr_response = corner_fast_response(
                        curr_pixel,
                        &self.circle_intensities,
                        &self.bins,
                        PixelType::Darker,
                        self.arc_length,
                    );
                }

                corner_response[ix] = curr_response;
            }
        }

        &self.corner_response
    }

    /// TODO
    pub fn get_keypoints(&mut self) -> Result<Vec<[usize; 2]>, ImageError> {
        get_peak_mask(&self.corner_response, &mut self.mask, self.threshold);
        exclude_border(&mut self.mask, self.min_distance);

        let coordinates =
            get_high_intensity_peaks(&self.corner_response, &self.mask, self.min_distance);
        Ok(coordinates)
    }
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

fn get_peak_mask<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    mask: &mut Image<bool, 1, A>,
    threshold: u8,
) {
    let src_slice = src.as_slice();
    let mask_slice = mask.as_slice_mut();

    src_slice
        .iter()
        .zip(mask_slice)
        .for_each(|(src, mask)| *mask = *src > threshold);
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

        let mut fast_detector = FastDetector::new(gray_img.size(), THRESHOLD, 12, 10)?;
        fast_detector.corner_fast(&gray_img);
        let keypoints = fast_detector.get_keypoints()?;

        assert_eq!(keypoints.len(), expected_keypoints.len());
        assert_eq!(keypoints, expected_keypoints);
        Ok(())
    }
}
