use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use kornia_tensor::CpuAllocator;
use rayon::prelude::*;

// Using u8 constants can be slightly faster in tight loops than an enum.
const PIXEL_SIMILAR: u8 = 0;
const PIXEL_BRIGHTER: u8 = 1;
const PIXEL_DARKER: u8 = 2;

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
    taken: Vec<bool>,
}

impl FastDetector {
    /// Creates a new `FastDetector` with the specified parameters.
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
            taken: vec![false; image_size.height * image_size.width],
        })
    }

    /// Clears the internal state of the detector, marking it ready to detect again.
    pub fn clear(&mut self) {
        self.taken.iter_mut().for_each(|px| *px = false);
    }

    /// Computes the corner response for the input image.
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

        const ROW_OFFSETS: [isize; 16] = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
        const COLUMN_OFFSETS: [isize; 16] = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];

        corner_response[3 * width..(height - 3) * width]
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let y = row_idx + 3;
                let mut bins = [PIXEL_SIMILAR; 16];
                let mut circle_intensities = [0f32; 16];

                for x in 3..width - 3 {
                    let ix = x;
                    let src_ix = y * width + x;
                    let curr_pixel = src_slice[src_ix];
                    let lower_threshold = curr_pixel - self.threshold;
                    let upper_threshold = curr_pixel + self.threshold;

                    // Speed test: check 4 equidistant pixels first
                    let mut speed_sum_b = 0;
                    let mut speed_sum_d = 0;
                    for &k in &[0, 4, 8, 12] {
                        let ik = ((y as isize + ROW_OFFSETS[k]) * width as isize
                            + (x as isize + COLUMN_OFFSETS[k]))
                            as usize;
                        let ring_pixel = src_slice[ik];
                        if ring_pixel > upper_threshold {
                            speed_sum_b += 1;
                        } else if ring_pixel < lower_threshold {
                            speed_sum_d += 1;
                        }
                    }

                    if speed_sum_b < 3 && speed_sum_d < 3 {
                        row[ix] = 0.0;
                        continue;
                    }

                    // Full test for all 16 pixels
                    for k in 0..16 {
                        let ik = ((y as isize + ROW_OFFSETS[k]) * width as isize
                            + (x as isize + COLUMN_OFFSETS[k]))
                            as usize;
                        let intensity = src_slice[ik];
                        circle_intensities[k] = intensity;
                        bins[k] = if intensity > upper_threshold {
                            PIXEL_BRIGHTER
                        } else if intensity < lower_threshold {
                            PIXEL_DARKER
                        } else {
                            PIXEL_SIMILAR
                        };
                    }

                    // Test for a contiguous arc of bright pixels
                    let mut curr_response = corner_fast_response(
                        curr_pixel,
                        &circle_intensities,
                        &bins,
                        PIXEL_BRIGHTER,
                        self.arc_length,
                    );

                    // If not found, test for a contiguous arc of dark pixels
                    if curr_response == 0.0 {
                        curr_response = corner_fast_response(
                            curr_pixel,
                            &circle_intensities,
                            &bins,
                            PIXEL_DARKER,
                            self.arc_length,
                        );
                    }
                    row[ix] = curr_response;
                }
            });

        Ok(&self.corner_response)
    }

    /// Extracts keypoints using Non-Maximum Suppression on the corner response.
    pub fn extract_keypoints(&mut self) -> Vec<[usize; 2]> {
        get_high_intensity_peaks(
            &self.corner_response,
            self.threshold,
            self.min_distance,
            &mut self.taken,
        )
    }
}

fn corner_fast_response(
    curr_pixel: f32,
    circle_intensities: &[f32; 16],
    bins: &[u8; 16],
    state: u8,
    n: usize,
) -> f32 {
    let mut consecutive_count = 0;

    for l in 0..(16 + n - 1) {
        if bins[l % 16] == state {
            consecutive_count += 1;

            if consecutive_count >= n {
                // calculate score
                return circle_intensities
                    .iter()
                    .map(|&m| (m - curr_pixel).abs())
                    .sum();
            }
        } else {
            consecutive_count = 0;
        }
    }
    0.0
}

fn get_high_intensity_peaks(
    src: &Image<f32, 1, CpuAllocator>,
    threshold: f32,
    min_distance: usize,
    taken: &mut [bool],
) -> Vec<[usize; 2]> {
    let src_size = src.size();
    let width = src_size.width;
    let height = src_size.height;
    let src_slice = src.as_slice();

    // 1. Collect all pixels with a response score above the threshold.
    let mut candidates: Vec<(f32, usize, [usize; 2])> = src_slice
        .par_iter()
        .enumerate()
        .filter_map(|(i, &score)| {
            if score > threshold {
                let y = i / width;
                let x = i % width;

                if x >= min_distance
                    && x < width - min_distance
                    && y >= min_distance
                    && y < height - min_distance
                {
                    Some((score, i, [y, x]))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    // 2. Sort candidates in descending order of their response score.
    candidates.par_sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut result = Vec::new();
    let min_dist_isize = min_distance as isize;

    // 3. Iterate through sorted candidates and perform suppression.
    for (_, idx, [y, x]) in candidates {
        // location already suppressed, skip
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

        for yy in y0..y1 {
            let iyy = yy * width;

            for xx in x0..x1 {
                // Using Chebyshev distance (max of coord differences) for suppression square
                if (yy as isize - y as isize)
                    .abs()
                    .max((xx as isize - x as isize).abs())
                    <= min_dist_isize
                {
                    taken[iyy + xx] = true;
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

        let mut gray_imgf32 = Image::from_size_val(img.size(), 0.0, CpuAllocator)?;
        gray_img
            .as_slice()
            .iter()
            .zip(gray_imgf32.as_slice_mut())
            .for_each(|(&p, m)| {
                *m = p as f32 / 255.0;
            });

        let expected_keypoints = vec![
            [134, 95],
            [129, 162],
            [71, 84],
            [60, 75],
            [120, 64],
            [162, 148],
            [109, 69],
            [109, 125],
            [69, 184],
            [153, 104],
            [72, 169],
            [141, 121],
            [32, 86],
        ];

        const THRESHOLD: f32 = 0.15;

        let mut fast_detector = FastDetector::new(gray_img.size(), THRESHOLD, 12, 10)?;
        fast_detector.compute_corner_response(&gray_imgf32)?;
        let keypoints = fast_detector.extract_keypoints();

        assert_eq!(keypoints.len(), expected_keypoints.len());
        assert_eq!(keypoints, expected_keypoints);
        Ok(())
    }
}
