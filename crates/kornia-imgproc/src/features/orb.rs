// TODO: Remove Missing docs
#![allow(missing_docs)]

use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use kornia_tensor::CpuAllocator;

use crate::{
    features::{
        other::{corner_fast, peak_local_max},
        HarrisResponse,
    },
    filter::gaussian_blur,
    resize::resize_native,
};

pub struct OrbDectector {
    pub n_keypoints: usize,
    pub fast_n: usize,
    pub fast_threshold: f32,
    pub harris_k: f32,
    pub downscale: f32,
    pub n_scales: usize,
}

impl Default for OrbDectector {
    fn default() -> Self {
        Self {
            downscale: 1.2,
            n_scales: 8,
            n_keypoints: 500,
            fast_n: 9,
            fast_threshold: 0.08,
            harris_k: 0.04,
        }
    }
}

impl OrbDectector {
    pub fn new() -> Self {
        Self::default()
    }

    fn build_pyramid<A: ImageAllocator>(
        &self,
        img: &Image<f32, 1, A>,
    ) -> Result<Vec<Image<f32, 1, CpuAllocator>>, ImageError> {
        let img = Image::from_size_slice(img.size(), img.as_slice(), CpuAllocator)?;

        let mut pyramid = Vec::with_capacity(self.n_scales);
        let mut current = img.clone();
        pyramid.push(current.clone());

        for _ in 1..self.n_scales {
            let next = pyramid_reduce(&current, self.downscale)?;
            if next.size() == current.size() {
                break;
            }

            pyramid.push(next.clone());
            current = next;
        }

        Ok(pyramid)
    }

    fn detect_octave<A: ImageAllocator>(
        &self,
        octave_image: &Image<f32, 1, A>,
    ) -> Result<(Vec<(usize, usize)>, Vec<f32>, Vec<f32>), ImageError> {
        let fast_response = corner_fast(&octave_image, 0.15, 12)?;
        let keypoints = peak_local_max(&fast_response, 1)?;

        if keypoints.is_empty() {
            return Ok((vec![], vec![], vec![]));
        }

        let mask = mask_border_keypoints(octave_image.size(), &keypoints, 16);
        let filtered_keypoints: Vec<_> = keypoints
            .iter()
            .zip(mask.iter())
            .filter_map(|(kp, &m)| if m { Some((kp.0, kp.1)) } else { None })
            .collect();

        if filtered_keypoints.is_empty() {
            return Ok((vec![], vec![], vec![]));
        }

        let orientations = corner_orientations(octave_image, &keypoints);

        let mut response = Image::from_size_val(octave_image.size(), 0f32, CpuAllocator)?;
        let mut harris_response = HarrisResponse::new(octave_image.size()).with_k(self.harris_k);
        harris_response.compute(octave_image, &mut response)?;

        let filtered_responses: Vec<_> = filtered_keypoints
            .iter()
            .map(|&(r, c)| response.as_slice()[response.size().index(r, c)])
            .collect();

        Ok((filtered_keypoints, orientations, filtered_responses))
    }

    pub fn detect<A: ImageAllocator>(
        &self,
        src: &Image<f32, 1, A>,
    ) -> Result<(Vec<(f32, f32)>, Vec<f32>, Vec<f32>, Vec<f32>), ImageError> {
        let pyramid = self.build_pyramid(src)?;

        let mut keypoints_list = vec![];
        let mut orientations_list = vec![];
        let mut scales_list = vec![];
        let mut responses_list = vec![];

        for (octave, octave_image) in pyramid.iter().enumerate() {
            let (keypoints, orientations, responses) = self.detect_octave(&octave_image)?;

            let scale = self.downscale.powi(octave as i32);

            for i in 0..keypoints.len() {
                keypoints_list.push((keypoints[i].0 as f32 * scale, keypoints[i].1 as f32 * scale));
                orientations_list.push(orientations[i]);
                scales_list.push(scale);
                responses_list.push(responses[i]);
            }
        }

        let n_keypoints = keypoints_list.len();
        if n_keypoints < self.n_keypoints {
            // Not enough keypoints, return all
            Ok((
                keypoints_list,
                scales_list,
                orientations_list,
                responses_list,
            ))
        } else {
            let mut indices: Vec<usize> = (0..n_keypoints).collect();
            indices.sort_unstable_by(|&i, &j| {
                responses_list[j].partial_cmp(&responses_list[i]).unwrap()
            });

            let mut best_keypoints = Vec::with_capacity(self.n_keypoints);
            let mut best_scales = Vec::with_capacity(self.n_keypoints);
            let mut best_orientations = Vec::with_capacity(self.n_keypoints);
            let mut best_responses = Vec::with_capacity(self.n_keypoints);

            for &idx in indices.iter().take(self.n_keypoints) {
                best_keypoints.push(keypoints_list[idx]);
                best_scales.push(scales_list[idx]);
                best_orientations.push(orientations_list[idx]);
                best_responses.push(responses_list[idx]);
            }

            Ok((
                best_keypoints,
                best_scales,
                best_orientations,
                best_responses,
            ))
        }
    }

    fn extract_octave<A: ImageAllocator>(
        &self,
        octave_image: &Image<f32, 1, A>,
        keypoints: &[(i32, i32)],
        orientations: &[f32],
    ) -> (Vec<Vec<u8>>, Vec<bool>) {
        let mask = mask_border_keypoints_i32(octave_image.size(), keypoints, 20);

        // Filter keypoints and orientations by mask
        let filtered_keypoints: Vec<_> = keypoints
            .iter()
            .zip(mask.iter())
            .filter_map(|(kp, &m)| if m { Some(*kp) } else { None })
            .collect();

        let filtered_orientations: Vec<f32> = orientations
            .iter()
            .zip(mask.iter())
            .filter_map(|(&o, &m)| if m { Some(o) } else { None })
            .collect();

        let descriptors = orb_loop(octave_image, &filtered_keypoints, &filtered_orientations);

        (descriptors, mask)
    }

    pub fn extract<A: ImageAllocator>(
        &self,
        src: &Image<f32, 1, A>,
        keypoints: &[(f32, f32)],
        scales: &[f32],
        orientations: &[f32],
    ) -> Result<(Vec<Vec<u8>>, Vec<bool>), ImageError> {
        let pyramid = self.build_pyramid(src)?;

        let mut descriptors_list: Vec<Vec<u8>> = Vec::new();
        let mut mask_list: Vec<bool> = Vec::new();

        let octaves: Vec<usize> = scales
            .iter()
            .map(|&s| (s.ln() / self.downscale.ln()).round() as usize)
            .collect();

        for (octave, octave_image) in pyramid.iter().enumerate() {
            let octave_mask: Vec<bool> = octaves.iter().map(|&o| o == octave).collect();

            let n_in_octave = octave_mask.iter().filter(|&&b| b).count();
            if n_in_octave == 0 {
                continue;
            }

            let mut octave_keypoints = Vec::with_capacity(n_in_octave);
            let mut octave_orientations = Vec::with_capacity(n_in_octave);

            for ((&(y, x), &ori), &is_in_octave) in
                keypoints.iter().zip(orientations).zip(&octave_mask)
            {
                if is_in_octave {
                    let scale = self.downscale.powi(octave as i32);
                    octave_keypoints.push(((y / scale).round() as i32, (x / scale).round() as i32));
                    octave_orientations.push(ori);
                }
            }

            let (descriptors, mask) =
                self.extract_octave(octave_image, &octave_keypoints, &octave_orientations);

            descriptors_list.extend(descriptors);
            mask_list.extend(mask);
        }

        Ok((descriptors_list, mask_list))
    }
}

fn pyramid_reduce<A: ImageAllocator>(
    img: &Image<f32, 1, A>,
    downscale: f32,
) -> Result<Image<f32, 1, CpuAllocator>, ImageError> {
    let sigma = 2.0 * downscale / 6.0;

    let mut smoothed = Image::from_size_val(img.size(), 0.0, CpuAllocator)?;
    gaussian_blur(img, &mut smoothed, (0, 0), (sigma, 0.0))?;

    let new_h = (smoothed.height() as f32 / downscale).ceil() as usize;
    let new_w = (smoothed.width() as f32 / downscale).ceil() as usize;

    let mut resized = Image::from_size_val(
        ImageSize {
            width: new_w,
            height: new_h,
        },
        0.0,
        CpuAllocator,
    )?;
    resize_native(
        &smoothed,
        &mut resized,
        crate::interpolation::InterpolationMode::Nearest,
    )?;

    Ok(resized)
}

fn mask_border_keypoints(
    size: ImageSize,
    keypoints: &[(usize, usize, f32)],
    distance: i32,
) -> Vec<bool> {
    let rows = size.height;
    let cols = size.width;

    keypoints
        .iter()
        .map(|(r, c, _)| {
            let min = distance.saturating_sub(1);
            let max_row = rows as isize - distance as isize + 1;
            let max_col = cols as isize - distance as isize + 1;
            let r = *r as isize;
            let c = *c as isize;

            (min as isize) < r && r < max_row && (min as isize) < c && c < max_col
        })
        .collect()
}

// TODO: Make a single function by keeping keypoints type same
fn mask_border_keypoints_i32(
    size: ImageSize,
    keypoints: &[(i32, i32)],
    distance: i32,
) -> Vec<bool> {
    let rows = size.height;
    let cols = size.width;

    keypoints
        .iter()
        .map(|(r, c)| {
            let min = distance.saturating_sub(1);
            let max_row = rows as isize - distance as isize + 1;
            let max_col = cols as isize - distance as isize + 1;
            let r = *r as isize;
            let c = *c as isize;

            (min as isize) < r && r < max_row && (min as isize) < c && c < max_col
        })
        .collect()
}

fn corner_orientations<A: ImageAllocator>(
    src: &Image<f32, 1, A>,
    corners: &[(usize, usize, f32)],
) -> Vec<f32> {
    const M_SIZE: usize = 31; // NOTE: This must be uneven
    let mask = vec![false; M_SIZE * M_SIZE];
    let src_slice = src.as_slice();

    let mrows2 = (M_SIZE as i32 - 1) / 2;
    let mcols2 = (M_SIZE as i32 - 1) / 2;

    let height = src.height() as i32;
    let width = src.width() as i32;

    let mut orientations = Vec::with_capacity(corners.len());

    for &(r0, c0, _) in corners {
        let mut m01 = 0f32;
        let mut m10 = 0f32;

        for r in 0..M_SIZE as i32 {
            let mut m01_tmp = 0f32;

            for c in 0..M_SIZE as i32 {
                let mask_idx = (r as usize) * M_SIZE + (c as usize);
                if mask[mask_idx] != false {
                    let rr = r0 as i32 + r - mrows2;
                    let cc = c0 as i32 + c - mcols2;

                    if rr >= 0 && rr < height && cc >= 0 && cc < width {
                        let curr_pixel = src_slice[src.size().index(rr as usize, cc as usize)];
                        m10 += curr_pixel * (c - mcols2) as f32;
                        m01_tmp += curr_pixel;
                    }
                }
            }

            m01 += m01_tmp * (r - mrows2) as f32;
        }
        orientations.push(m01.atan2(m10));
    }

    orientations
}

fn orb_loop<A: ImageAllocator>(
    src: &Image<f32, 1, A>,
    keypoints: &[(i32, i32)],
    orientation: &[f32],
) -> Vec<Vec<u8>> {
    let n_keypoints = keypoints.len();
    let descriptor_len = POS0.len();
    let height = src.height() as i32;
    let width = src.width() as i32;

    let mut descriptors = vec![vec![0; descriptor_len]; n_keypoints];

    for i in 0..n_keypoints {
        let angle = orientation[i];
        let sin_a = angle.sin();
        let cos_a = angle.cos();

        let kr = keypoints[i].0;
        let kc = keypoints[i].1;

        for j in 0..descriptor_len {
            let pr0 = POS0[j][0] as f32;
            let pc0 = POS0[j][1] as f32;
            let pr1 = POS1[j][0] as f32;
            let pc1 = POS1[j][1] as f32;

            let spr0 = (sin_a * pr0 + cos_a * pc0).round() as i32;
            let spc0 = (cos_a * pr0 - sin_a * pc0).round() as i32;
            let spr1 = (sin_a * pr1 + cos_a * pc1).round() as i32;
            let spc1 = (cos_a * pr1 - sin_a * pc1).round() as i32;

            let r0 = kr + spr0;
            let c0 = kc + spc0;
            let r1 = kr + spr1;
            let c1 = kc + spc1;

            if r0 >= 0
                && r0 < height
                && c0 >= 0
                && c0 < width
                && r1 >= 0
                && r1 < height
                && c1 >= 0
                && c1 < width
            {
                let v0 = src.as_slice()[src.size().index(r0 as usize, c0 as usize)];
                let v1 = src.as_slice()[src.size().index(r1 as usize, c1 as usize)];
                descriptors[i][j] = if v0 < v1 { 1 } else { 0 };
            } else {
                descriptors[i][j] = 0;
            }
        }
    }

    descriptors
}

const POS0: [[i8; 2]; 256] = [
    [8, -3],
    [4, 2],
    [-11, 9],
    [7, -12],
    [2, -13],
    [1, -7],
    [-2, -10],
    [-13, -13],
    [-13, -3],
    [10, 4],
    [-13, -8],
    [-11, 7],
    [7, 7],
    [-4, -5],
    [-13, 2],
    [-9, 0],
    [12, -6],
    [-3, 6],
    [-6, -13],
    [11, -13],
    [4, 7],
    [5, -3],
    [3, -7],
    [-8, -7],
    [-2, 11],
    [-13, 12],
    [-7, 3],
    [-4, 2],
    [-10, -12],
    [5, -12],
    [5, -6],
    [1, 0],
    [9, 11],
    [4, 7],
    [2, -1],
    [-4, -12],
    [-8, -5],
    [4, 11],
    [0, -8],
    [-13, -2],
    [-3, -2],
    [-6, 9],
    [8, 12],
    [0, 9],
    [7, -5],
    [-13, -6],
    [10, 7],
    [-6, -3],
    [10, -9],
    [-13, 8],
    [-13, 0],
    [3, 3],
    [5, 7],
    [-1, 7],
    [3, -10],
    [2, -4],
    [-13, 0],
    [-13, -7],
    [-13, 3],
    [-7, 12],
    [6, -10],
    [-9, -1],
    [-2, -5],
    [-12, 5],
    [3, -10],
    [-7, -7],
    [-3, -2],
    [2, 9],
    [-11, -13],
    [-1, 6],
    [5, -3],
    [-4, -13],
    [-9, -6],
    [-12, -10],
    [10, 2],
    [7, 12],
    [-7, -13],
    [-4, 9],
    [7, -1],
    [-7, 6],
    [-13, 11],
    [-3, 7],
    [7, -8],
    [-13, -7],
    [1, -3],
    [2, -6],
    [-4, 3],
    [-1, -13],
    [7, 1],
    [1, -1],
    [9, 1],
    [-1, -9],
    [-13, -13],
    [7, 7],
    [12, -5],
    [6, 3],
    [5, -13],
    [2, -12],
    [3, 8],
    [2, 6],
    [9, -12],
    [-8, 4],
    [-11, 12],
    [1, 12],
    [6, -9],
    [2, 3],
    [6, 3],
    [3, -3],
    [7, 8],
    [-11, -5],
    [-10, 11],
    [-5, -8],
    [-10, 5],
    [8, -1],
    [4, -6],
    [-10, 12],
    [4, -2],
    [-2, 0],
    [-5, -8],
    [7, -6],
    [-9, -13],
    [-5, -13],
    [8, -8],
    [-9, -11],
    [1, -8],
    [7, -4],
    [-2, 1],
    [11, -6],
    [-12, -9],
    [3, 7],
    [5, 5],
    [0, -4],
    [-9, 12],
    [0, 7],
    [-1, 2],
    [5, 11],
    [3, 5],
    [-13, -4],
    [-5, 9],
    [-4, -7],
    [6, 5],
    [-7, 6],
    [-13, 6],
    [1, -10],
    [4, 1],
    [-2, -2],
    [2, -12],
    [-2, -13],
    [4, 1],
    [-6, -10],
    [-3, -13],
    [7, 5],
    [4, -2],
    [-13, 9],
    [7, 1],
    [7, -8],
    [-7, -4],
    [-8, 11],
    [-13, 6],
    [2, 4],
    [10, -5],
    [-6, -5],
    [8, -3],
    [2, -12],
    [-11, -2],
    [-12, -13],
    [-11, 0],
    [5, -3],
    [-2, -13],
    [-1, -8],
    [-13, -11],
    [-10, -2],
    [-3, 9],
    [2, -3],
    [-9, -13],
    [-4, 6],
    [-4, 12],
    [-6, -11],
    [6, -3],
    [-13, 11],
    [11, 11],
    [7, -5],
    [-1, 12],
    [-4, -8],
    [-7, 1],
    [-13, -12],
    [-7, -2],
    [-8, 5],
    [-5, -1],
    [-13, 7],
    [1, 5],
    [1, 0],
    [9, 12],
    [5, -8],
    [-1, 11],
    [-9, -3],
    [-1, -10],
    [-13, 1],
    [8, -11],
    [2, -13],
    [7, -13],
    [-10, -10],
    [-10, -8],
    [4, -6],
    [3, 12],
    [-4, 2],
    [5, -13],
    [4, -13],
    [-9, 9],
    [0, 3],
    [-12, 1],
    [3, 2],
    [-10, -10],
    [8, -13],
    [-8, -12],
    [2, 2],
    [10, 6],
    [6, 8],
    [-7, 10],
    [-3, -9],
    [-1, -13],
    [-3, -7],
    [-8, -2],
    [4, 2],
    [2, -5],
    [6, -9],
    [3, -1],
    [11, -1],
    [-3, 0],
    [4, -11],
    [2, -4],
    [-10, -6],
    [-13, 7],
    [-13, 12],
    [6, 0],
    [0, -1],
    [-13, 3],
    [-9, 8],
    [-13, -6],
    [5, -9],
    [2, 7],
    [-1, -6],
    [9, 5],
    [11, -3],
    [3, 0],
    [-1, 4],
    [3, -6],
    [-13, 0],
    [5, 8],
    [8, 9],
    [7, -4],
    [-10, 4],
    [7, 3],
    [9, -7],
    [7, 0],
    [-1, -6],
];

const POS1: [[i8; 2]; 256] = [
    [9, 5],
    [7, -12],
    [-8, 2],
    [12, -13],
    [2, 12],
    [1, 6],
    [-2, -4],
    [-11, -8],
    [-12, -9],
    [11, 9],
    [-8, -9],
    [-9, 12],
    [12, 6],
    [-3, 0],
    [-12, -3],
    [-7, 5],
    [12, -1],
    [-2, 12],
    [-4, -8],
    [12, -8],
    [5, 1],
    [10, -3],
    [6, 12],
    [-6, -2],
    [-1, -10],
    [-8, 10],
    [-5, -3],
    [-3, 7],
    [-6, 11],
    [6, -7],
    [7, -1],
    [4, -5],
    [11, -13],
    [4, 12],
    [4, 4],
    [-2, 7],
    [-7, -10],
    [9, 12],
    [1, -13],
    [-8, 2],
    [-2, 3],
    [-4, -9],
    [10, 7],
    [1, 3],
    [11, -10],
    [-11, 0],
    [12, 1],
    [-6, 12],
    [12, -4],
    [-8, -12],
    [-8, -4],
    [7, 8],
    [10, -7],
    [1, -12],
    [5, 6],
    [3, -10],
    [-13, 5],
    [-12, 12],
    [-11, 8],
    [-4, 7],
    [12, 8],
    [-7, -6],
    [0, 12],
    [-7, 5],
    [8, -13],
    [-4, 5],
    [-1, -7],
    [5, -11],
    [-5, -13],
    [0, -1],
    [5, 2],
    [-4, 12],
    [-9, 6],
    [-8, -4],
    [12, -3],
    [12, 12],
    [-6, 5],
    [-3, 4],
    [12, 2],
    [-5, 1],
    [-12, 5],
    [-2, -6],
    [12, -7],
    [-11, -12],
    [12, 12],
    [3, 0],
    [-2, -13],
    [1, 9],
    [8, -6],
    [3, 12],
    [12, 6],
    [-1, 3],
    [-10, 5],
    [10, 12],
    [12, 9],
    [7, 11],
    [6, 10],
    [2, 3],
    [4, -6],
    [12, -13],
    [10, 3],
    [-7, 9],
    [-4, -6],
    [2, -8],
    [7, -4],
    [3, -2],
    [11, 0],
    [8, -8],
    [9, 3],
    [-6, -4],
    [-5, 10],
    [-3, 12],
    [-9, 0],
    [12, -6],
    [6, -11],
    [-8, 7],
    [6, 7],
    [-2, 12],
    [-5, 2],
    [10, 12],
    [-8, -8],
    [-5, -2],
    [9, -13],
    [-9, 0],
    [1, -2],
    [9, 1],
    [-1, -4],
    [12, -11],
    [-6, 4],
    [7, 12],
    [10, 8],
    [2, 8],
    [-5, -13],
    [2, 12],
    [1, 7],
    [7, -9],
    [6, -8],
    [-8, 9],
    [-3, -3],
    [-3, -12],
    [8, 0],
    [-6, 12],
    [-5, -2],
    [3, 10],
    [8, -4],
    [2, -13],
    [12, 12],
    [0, -6],
    [9, 3],
    [-3, -5],
    [-1, 1],
    [12, -11],
    [5, -7],
    [-9, -5],
    [8, 6],
    [7, 6],
    [-7, 1],
    [-7, -8],
    [-12, -8],
    [3, 9],
    [12, 3],
    [-6, 7],
    [9, -8],
    [2, 8],
    [-10, 3],
    [-7, -9],
    [-10, -5],
    [11, 8],
    [-1, 12],
    [0, 9],
    [-12, -5],
    [-10, 11],
    [-2, -13],
    [3, 2],
    [-4, 0],
    [-3, -10],
    [-2, -7],
    [-4, 9],
    [6, 11],
    [-5, 5],
    [12, 6],
    [12, -2],
    [0, 7],
    [-3, -2],
    [-6, 7],
    [-8, -13],
    [-6, -8],
    [-6, -9],
    [-4, 5],
    [-8, 10],
    [5, -13],
    [10, -13],
    [10, -1],
    [10, -9],
    [1, -13],
    [-6, 2],
    [1, 12],
    [-8, -10],
    [10, -6],
    [3, -6],
    [12, -9],
    [-5, -7],
    [-8, -13],
    [8, 5],
    [8, -13],
    [-3, -3],
    [10, -12],
    [5, -1],
    [-4, 3],
    [3, -9],
    [-6, 1],
    [4, -8],
    [-10, 9],
    [12, 12],
    [-6, -5],
    [3, 7],
    [11, -8],
    [8, -12],
    [-6, 5],
    [-3, 9],
    [-1, 5],
    [-3, 4],
    [-8, 3],
    [12, 12],
    [3, 11],
    [11, -13],
    [7, 12],
    [12, 4],
    [-3, 6],
    [4, 12],
    [2, 1],
    [-8, 1],
    [-11, 1],
    [-11, -13],
    [11, -13],
    [1, 4],
    [-9, -2],
    [-6, -3],
    [-8, -2],
    [8, 10],
    [3, -9],
    [-1, -1],
    [11, -2],
    [12, -8],
    [3, 5],
    [0, 10],
    [4, 5],
    [-10, 5],
    [12, 11],
    [9, -6],
    [8, -12],
    [-10, 9],
    [12, 4],
    [10, -2],
    [12, -2],
    [0, -11],
];
