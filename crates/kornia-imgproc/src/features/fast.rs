use std::ops::ControlFlow;

use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use kornia_tensor::CpuAllocator;
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq)]
enum PixelType {
    Brighter,
    Darker,
    Similar,
}

/// A FAST (Features from Accelerated Segment Test) feature detector for corner detection in images.
#[derive(Clone)]
pub struct FastDetector {
    /// The intensity threshold for detecting corners.
    pub threshold: f32,
    /// The minimum distance between detected keypoints.
    pub min_distance: usize,
    /// The minimum arc length for a sequence of contiguous pixels to be considered a corner.
    pub arc_length: usize,
    corner_response: Image<f32, 1, CpuAllocator>,
    mask: Image<bool, 1, CpuAllocator>,
    taken: Vec<bool>,
}

impl FastDetector {
    /// Creates a new `FastDetector` with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `image_size` - The size of the image to process.
    /// * `threshold` - The intensity threshold for detecting corners.
    /// * `arc_length` - The minimum arc length for a sequence of contiguous pixels to be considered a corner.
    /// * `min_distance` - The minimum distance between detected keypoints.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `FastDetector` or an `ImageError`.
    pub fn new(
        image_size: ImageSize,
        threshold: f32,
        arc_length: usize,
        min_distance: usize,
    ) -> Result<Self, ImageError> {
        Ok(Self {
            threshold,
            min_distance,
            arc_length,
            corner_response: Image::from_size_val(image_size, 0.0, CpuAllocator)?,
            mask: Image::from_size_val(image_size, false, CpuAllocator)?,
            taken: vec![false; image_size.height * image_size.width],
        })
    }

    /// Clears the internal state of the detector, marking it ready to detect again.
    pub fn clear(&mut self) {
        self.taken.par_iter_mut().for_each(|px| *px = false);
    }

    /// Computes the corner response for the input image.
    ///
    /// # Arguments
    ///
    /// * `src` - The source grayscale image.
    ///
    /// # Returns
    ///
    /// Returns a reference to the image containing the corner response.
    pub fn compute_corner_response<A: ImageAllocator>(
        &mut self,
        src: &Image<f32, 1, A>,
    ) -> Result<&Image<f32, 1, CpuAllocator>, ImageError> {
        if self.corner_response.size() != src.size() {
            return Err(ImageError::InvalidImageSize(
                src.width(),
                src.height(),
                self.corner_response.width(),
                self.corner_response.height(),
            ));
        }

        let src_slice = src.as_slice();

        let width = src.width();
        let height = src.height();

        let corner_response = self.corner_response.as_slice_mut();

        const RP: [isize; 16] = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
        const CP: [isize; 16] = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];

        corner_response[3 * width..(height - 3) * width]
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let y = row_idx + 3;
                let mut bins = [PixelType::Similar; 16];
                let mut circle_intensities = [0f32; 16];

                for x in 3..width - 3 {
                    let ix = x;
                    let src_ix = y * width + x;
                    let curr_pixel = src_slice[src_ix];
                    let lower_threshold = curr_pixel - self.threshold;
                    let upper_threshold = curr_pixel + self.threshold;

                    let mut speed_sum_b = 0;
                    let mut speed_sum_d = 0;

                    for &k in &[0, 4, 8, 12] {
                        let ik =
                            ((y as isize + RP[k]) * width as isize + (x as isize + CP[k])) as usize;
                        let ring_pixel = src_slice[ik];
                        if ring_pixel > upper_threshold {
                            speed_sum_b += 1;
                        } else if ring_pixel < lower_threshold {
                            speed_sum_d += 1;
                        }
                    }
                    if speed_sum_d < 3 && speed_sum_b < 3 {
                        row[ix] = 0.0;
                        continue;
                    }

                    for k in 0..16 {
                        let ik =
                            ((y as isize + RP[k]) * width as isize + (x as isize + CP[k])) as usize;
                        circle_intensities[k] = src_slice[ik];
                        bins[k] = if circle_intensities[k] > upper_threshold {
                            PixelType::Brighter
                        } else if circle_intensities[k] < lower_threshold {
                            PixelType::Darker
                        } else {
                            PixelType::Similar
                        };
                    }

                    // Test for bright pixels
                    let mut curr_response = corner_fast_response(
                        curr_pixel,
                        &circle_intensities,
                        &bins,
                        PixelType::Brighter,
                        self.arc_length,
                    );

                    // Test for dark pixels
                    if curr_response == 0.0 {
                        curr_response = corner_fast_response(
                            curr_pixel,
                            &circle_intensities,
                            &bins,
                            PixelType::Darker,
                            self.arc_length,
                        );
                    }

                    row[ix] = curr_response;
                }
            });

        Ok(&self.corner_response)
    }

    /// Extracts keypoints from the computed corner response.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of keypoint coordinates or an `ImageError`.
    pub fn extract_keypoints(&mut self) -> Result<Vec<[usize; 2]>, ImageError> {
        get_peak_mask(
            &self.corner_response,
            &mut self.mask,
            self.threshold,
            self.min_distance,
        );

        let coordinates = get_high_intensity_peaks(
            &self.corner_response,
            &self.mask,
            self.min_distance,
            &mut self.taken,
        );
        Ok(coordinates)
    }
}

fn corner_fast_response(
    curr_pixel: f32,
    circle_intensities: &[f32; 16],
    bins: &[PixelType; 16],
    state: PixelType,
    n: usize,
) -> f32 {
    let mut consecutive_count = 0;
    let mut curr_response = 0.0;

    if let ControlFlow::Break(_) = (0..15 + n).try_for_each(|l| {
        if bins[l % 16] == state {
            consecutive_count += 1;
            if consecutive_count == n {
                curr_response = 0.0;
                circle_intensities.iter().for_each(|m| {
                    curr_response += (m - curr_pixel).abs();
                });

                return ControlFlow::Break(());
            }
        } else {
            consecutive_count = 0;
        }

        ControlFlow::Continue(())
    }) {
        return curr_response;
    }

    0.0
}

fn get_peak_mask<A: ImageAllocator>(
    src: &Image<f32, 1, A>,
    mask: &mut Image<bool, 1, A>,
    threshold: f32,
    border_width: usize,
) {
    let mask_size = mask.size();

    let src_slice = src.as_slice();
    let mask_slice = mask.as_slice_mut();

    src_slice
        .par_iter()
        .zip(mask_slice)
        .enumerate()
        .for_each(|(i, (src, mask))| {
            let y = i / mask_size.width;
            let x = i % mask_size.width;

            if x < border_width
                || x >= mask_size.width.saturating_sub(border_width)
                || y < border_width
                || y >= mask_size.height.saturating_sub(border_width)
            {
                *mask = false;
            } else {
                *mask = *src > threshold;
            }
        });
}

fn get_high_intensity_peaks<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, 1, A1>,
    mask: &Image<bool, 1, A2>,
    min_distance: usize,
    taken: &mut [bool],
) -> Vec<[usize; 2]> {
    let src_size = src.size();
    let width = src_size.width;
    let height = src_size.height;

    let coords_intensity: Vec<[usize; 2]> = mask
        .as_slice()
        .par_iter()
        .enumerate()
        .filter(|&(_, &value)| value)
        .map(|(i, _)| {
            let y = i / width;
            let x = i % width;
            [y, x]
        })
        .collect();

    let mut result = Vec::new();

    for coord in coords_intensity {
        let y = coord[0];
        let x = coord[1];
        let idx = y * width + x;

        // If this location is already suppressed, skip
        if taken[idx] {
            continue;
        }

        // Accept this peak
        result.push([y, x]);

        // Suppress all within min_distance
        let y0 = y.saturating_sub(min_distance);
        let y1 = (y + min_distance + 1).min(height);
        let x0 = x.saturating_sub(min_distance);
        let x1 = (x + min_distance + 1).min(width);

        (y0..y1)
            .flat_map(|yy| (x0..x1).map(move |xx| (yy, xx)))
            .filter(|&(yy, xx)| {
                (yy as isize - y as isize)
                    .abs()
                    .max((xx as isize - x as isize).abs())
                    < min_distance as isize
            })
            .for_each(|(yy, xx)| {
                taken[yy * width + xx] = true;
            });
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

        let mut gray_imgf32 = Image::from_size_val(img.size(), 0.0, CpuAllocator)?;
        gray_img
            .as_slice()
            .iter()
            .zip(gray_imgf32.as_slice_mut())
            .for_each(|(&p, m)| {
                *m = p as f32 / 255.0;
            });

        let expected_keypoints = vec![
            [32, 86],
            [60, 75],
            [63, 183],
            [71, 84],
            [72, 169],
            [106, 69],
            [109, 125],
            [120, 64],
            [125, 165],
            [132, 94],
            [135, 161],
            [141, 121],
            [143, 99],
            [153, 104],
            [161, 148],
        ];

        const THRESHOLD: f32 = 0.15;

        let mut fast_detector = FastDetector::new(gray_img.size(), THRESHOLD, 12, 10)?;
        fast_detector.compute_corner_response(&gray_imgf32)?;
        let keypoints = fast_detector.extract_keypoints()?;

        assert_eq!(keypoints.len(), expected_keypoints.len());
        assert_eq!(keypoints, expected_keypoints);
        Ok(())
    }
}
