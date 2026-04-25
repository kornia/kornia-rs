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
    ) -> &Image<f32, 1, CpuAllocator> {
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

                    let bright_response = corner_fast_response(
                        curr_pixel,
                        &circle_intensities,
                        &bins,
                        PixelType::Brighter,
                        self.arc_length,
                    );

                    let dark_response = corner_fast_response(
                        curr_pixel,
                        &circle_intensities,
                        &bins,
                        PixelType::Darker,
                        self.arc_length,
                    );

                    row[ix] = bright_response.max(dark_response);
                }
            });

        &self.corner_response
    }

    /// Extracts keypoints from the computed corner response.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of keypoint coordinates or an `ImageError`.
    pub fn extract_keypoints(&mut self) -> Result<Vec<[usize; 2]>, ImageError> {
        get_peak_mask(&self.corner_response, &mut self.mask, self.threshold);
        exclude_border(&mut self.mask, self.min_distance);

        let coordinates = get_high_intensity_peaks(
            &self.corner_response,
            &self.mask,
            self.min_distance,
            &mut self.taken,
        );
        Ok(coordinates)
    }

    /// Extracts raw FAST candidates before nonmax suppression.
    pub fn extract_raw_keypoints(&self) -> Vec<[usize; 2]> {
        let size = self.corner_response.size();
        let width = size.width;
        self.corner_response
            .as_slice()
            .iter()
            .enumerate()
            .filter(|&(_, &response)| response > self.threshold)
            .map(|(i, _)| [i / width, i % width])
            .collect()
    }

    /// Returns the last computed FAST corner response image.
    pub fn corner_response(&self) -> &Image<f32, 1, CpuAllocator> {
        &self.corner_response
    }
}

fn corner_fast_response(
    curr_pixel: f32,
    circle_intensities: &[f32; 16],
    bins: &[PixelType; 16],
    state: PixelType,
    n: usize,
) -> f32 {
    let mut best_score = 0.0f32;
    for start in 0..16 {
        let mut arc_min = f32::INFINITY;
        let mut valid_arc = true;
        for offset in 0..n {
            let idx = (start + offset) % 16;
            if bins[idx] != state {
                valid_arc = false;
                break;
            }
            let diff = match state {
                PixelType::Brighter => circle_intensities[idx] - curr_pixel,
                PixelType::Darker => curr_pixel - circle_intensities[idx],
                PixelType::Similar => 0.0,
            };
            arc_min = arc_min.min(diff);
        }
        if valid_arc {
            best_score = best_score.max(arc_min);
        }
    }
    best_score
}

fn get_peak_mask<A: ImageAllocator>(
    src: &Image<f32, 1, A>,
    mask: &mut Image<bool, 1, A>,
    threshold: f32,
) {
    let width = src.width();
    let height = src.height();
    let src_slice = src.as_slice();
    let mask_slice = mask.as_slice_mut();

    mask_slice.fill(false);

    if width < 3 || height < 3 {
        return;
    }

    mask_slice[width..(height - 1) * width]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row_idx, row_mask)| {
            let y = row_idx + 1;
            for x in 1..width - 1 {
                let center = src_slice[y * width + x];
                if center <= threshold {
                    row_mask[x] = false;
                    continue;
                }

                let mut is_strict_max = true;
                'neighbors: for yy in (y - 1)..=(y + 1) {
                    for xx in (x - 1)..=(x + 1) {
                        if yy == y && xx == x {
                            continue;
                        }
                        if src_slice[yy * width + xx] >= center {
                            is_strict_max = false;
                            break 'neighbors;
                        }
                    }
                }
                row_mask[x] = is_strict_max;
            }
        });
}

fn exclude_border<A: ImageAllocator>(label: &mut Image<bool, 1, A>, border_width: usize) {
    let label_size = label.size();
    let label_slice = label.as_slice_mut();

    (0..label_size.height).for_each(|y| {
        let iy = y * label_size.width;

        (0..label_size.width).for_each(|x| {
            if x < border_width
                || x >= label_size.width.saturating_sub(border_width)
                || y < border_width
                || y >= label_size.height.saturating_sub(border_width)
            {
                label_slice[iy + x] = false;
            }
        });
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
    let src_slice = src.as_slice();

    let mut coords_with_response: Vec<([usize; 2], f32)> = mask
        .as_slice()
        .iter()
        .enumerate()
        .filter(|&(_, &value)| value)
        .map(|(i, _)| {
            let y = i / width;
            let x = i % width;
            ([y, x], src_slice[i])
        })
        .collect();

    coords_with_response
        .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = Vec::new();

    for (coord, _) in coords_with_response {
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
    use std::collections::HashSet;

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
            [69, 184],
            [72, 84],
            [72, 169],
            [109, 125],
            [122, 63],
            [125, 165],
            [134, 95],
            [135, 161],
            [141, 121],
            [153, 104],
            [162, 148],
        ];

        let legacy_keypoints = vec![[71, 84], [109, 69], [109, 125], [120, 64], [129, 162]];

        const THRESHOLD: f32 = 0.15;

        let mut fast_detector = FastDetector::new(gray_img.size(), THRESHOLD, 12, 10)?;
        fast_detector.compute_corner_response(&gray_imgf32);
        let keypoints = fast_detector.extract_keypoints()?;
        assert_eq!(keypoints.len(), expected_keypoints.len());
        let expected: HashSet<[usize; 2]> = expected_keypoints.into_iter().collect();
        let actual: HashSet<[usize; 2]> = keypoints.into_iter().collect();
        assert_eq!(actual, expected);
        let legacy: HashSet<[usize; 2]> = legacy_keypoints.into_iter().collect();
        assert_ne!(actual, legacy);
        Ok(())
    }

    #[test]
    fn test_fast_9_accepts_arc_with_only_two_cardinals() -> Result<(), Box<dyn std::error::Error>> {
        let mut img = Image::from_size_val(
            ImageSize {
                width: 7,
                height: 7,
            },
            0.5f32,
            CpuAllocator,
        )?;

        let center_r = 3usize;
        let center_c = 3usize;
        let width = img.width();
        let ring_offsets = [
            (0isize, 3isize),
            (1, 3),
            (2, 2),
            (3, 1),
            (3, 0),
            (3, -1),
            (2, -2),
            (1, -3),
            (0, -3),
        ];

        for (dr, dc) in ring_offsets {
            let r = (center_r as isize + dr) as usize;
            let c = (center_c as isize + dc) as usize;
            img.as_slice_mut()[r * width + c] = 1.0;
        }

        let mut fast_detector = FastDetector::new(img.size(), 0.1, 9, 1)?;
        fast_detector.compute_corner_response(&img);

        let raw: HashSet<[usize; 2]> = fast_detector.extract_raw_keypoints().into_iter().collect();
        assert!(raw.contains(&[center_r, center_c]));

        let nms: HashSet<[usize; 2]> = fast_detector.extract_keypoints()?.into_iter().collect();
        assert!(nms.contains(&[center_r, center_c]));
        Ok(())
    }
}
