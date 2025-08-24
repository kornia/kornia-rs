// TODO: Remove Missing docs
#![allow(missing_docs)]

use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use kornia_tensor::CpuAllocator;

use crate::{filter::gaussian_blur, resize::resize_native};

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
        octave_image: Image<f32, 1, A>,
    ) -> Result<Option<(Vec<(usize, usize)>, Vec<f32>, Vec<f32>)>, ImageError> {
        todo!()
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

fn corner_peaks<A: ImageAllocator>(
    img: &[Image<f32, 1, A>],
    min_distance: usize,
    threshold_abs: Option<f32>,
    exclude_border: bool,
    num_peaks: Option<usize>,
) -> Vec<[i32; 2]> {
    todo!()
}

fn mask_border_keypoints(size: ImageSize, keypoints: &[[i32; 2]], distance: i32) -> Vec<bool> {
    let rows = size.height;
    let cols = size.width;

    keypoints
        .iter()
        .map(|[r, c]| {
            let min = distance.saturating_sub(1);
            let max_row = rows as isize - distance as isize + 1;
            let max_col = cols as isize - distance as isize + 1;
            let r = *r as isize;
            let c = *c as isize;

            (min as isize) < r && r < max_row && (min as isize) < c && c < max_col
        })
        .collect()
}
