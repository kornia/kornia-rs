use std::f32::consts::SQRT_2;

use kornia_image::{allocator::ImageAllocator, Image, ImageSize};
use kornia_tensor::CpuAllocator;

/// TODO
pub enum OrbError {
    /// Patch Size smaller than 2
    SmallPatchSize,
}

/// TODO
#[derive(Debug, PartialEq)]
pub enum ScoreType {
    /// TODO
    HarrisScore,
    /// TODO
    FastScore,
}

/// TODO
pub struct Point2d<T = f32> {
    /// TODO
    pub x: T,
    /// TODO
    pub y: T,
}

/// TODO
pub struct KeyPoint {
    /// TODO
    pub pt: Point2d,
    /// TODO
    pub size: f32,
    /// TODO
    pub angle: f32,
    /// TODO
    pub response: f32,
    /// TODO
    pub octave: usize,
    /// TODO
    pub class_id: usize,
}

/// TODO
#[derive(Default, Clone, Copy)]
pub struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// TODO
#[derive(Debug, PartialEq)]
pub struct OrbDetector {
    /// TODO
    pub nfeatures: usize,
    /// TODO
    pub scale_factor: f32,
    /// TODO
    pub nlevels: usize,
    /// TODO
    pub edge_threshold: usize,
    /// TODO
    pub first_level: i32,
    /// TODO
    pub wta_k: usize,
    /// TODO
    pub score_type: ScoreType,
    /// TODO
    pub patch_size: usize,
    /// TODO
    pub fast_threshold: usize,
}

impl Default for OrbDetector {
    fn default() -> Self {
        Self {
            nfeatures: 500,
            scale_factor: 1.2,
            nlevels: 8,
            edge_threshold: 31,
            first_level: 0,
            wta_k: 2,
            score_type: ScoreType::HarrisScore,
            patch_size: 31,
            fast_threshold: 20,
        }
    }
}

impl OrbDetector {
    /// TODO
    pub fn detect_and_compute<A1: ImageAllocator, A2: ImageAllocator, A3: ImageAllocator>(
        &self,
        src: &Image<u8, 1, A1>,
        mask: Option<&Image<u8, 1, A2>>,
        keypoints: Option<Vec<KeyPoint>>,
        descriptors: Option<&mut Image<u8, 1, A3>>,
        use_provided_keypoints: bool,
    ) -> Result<(), OrbError> {
        if self.patch_size < 2 {
            return Err(OrbError::SmallPatchSize);
        }

        let do_keypoints = !use_provided_keypoints;
        let do_descriptors = descriptors.is_some();

        if !do_keypoints && !do_descriptors {
            // TODO: Return meaningful error
            return Ok(());
        }

        const HARRIS_BLOCK_SIZE: usize = 9;
        let half_patch_size = self.patch_size / 2;
        let desc_patch_size = (half_patch_size as f32 * SQRT_2).ceil() as usize;
        let border = self
            .edge_threshold
            .max(desc_patch_size.max(HARRIS_BLOCK_SIZE / 2))
            + 1;

        let mut n_levels = self.nlevels;
        let mut level = 0;
        let mut n_keypoints = match keypoints {
            Some(ref k) => k.len(),
            None => 0,
        };
        let mut sorted_by_level = true;

        if !do_keypoints {
            // TODO: Don't unwrap it
            let keypoints = keypoints.as_ref().unwrap();
            n_levels = 0;
            (0..n_keypoints).for_each(|i| {
                level = keypoints[i].octave;
                debug_assert!(level >= 0);

                if i > 0 && level < keypoints[i - 1].octave {
                    sorted_by_level = false;
                }
                n_levels = n_levels.max(level);
            });

            n_levels += 1;
        }

        let mut layer_info = Vec::with_capacity(n_levels);
        let mut layer_ofs = Vec::with_capacity(n_levels);
        let mut layer_scale = vec![0.0; n_levels];

        let level0_inv_scale = 1.0 / get_scale(0, self.first_level, self.scale_factor);
        let level0_width = (src.cols() as f32 * level0_inv_scale).round() as usize;
        let level0_height = (src.rows() as f32 * level0_inv_scale).round() as usize;
        let mut buf_size = Point2d {
            x: ((level0_width + border * 2) - 15) as isize & -16,
            y: 0,
        };

        let mut level_dy = level0_height + border * 2;
        let mut level_ofs = Point2d { x: 0, y: 0 };

        (0..n_levels).for_each(|level| {
            let scale = get_scale(level as i32, self.first_level, self.scale_factor);
            layer_scale[level] = scale;

            let inv_scale = 1.0 / scale;
            let sz = Point2d {
                x: (src.cols() as f32 * inv_scale).round() as usize,
                y: (src.rows() as f32 * inv_scale).round() as usize,
            };
            let whole_size = Point2d {
                x: sz.x + border * 2,
                y: sz.y + border * 2,
            };

            if (level_ofs.x + whole_size.x) as isize > buf_size.x {
                level_ofs = Point2d {
                    x: 0,
                    y: level_ofs.y + level_dy,
                };
                level_dy = whole_size.y;
            }

            let l_info = Rect {
                x: (level_ofs.x + border) as u32,
                y: (level_ofs.y + border) as u32,
                width: (sz.x) as u32,
                height: (sz.y) as u32,
            };

            layer_info[level] = l_info;
            layer_ofs[level] = l_info.y as isize * buf_size.x + l_info.x as isize;
            level_ofs.x += whole_size.x;
        });
        buf_size.y = (level_ofs.y + level_dy) as isize;

        let img_pyramid: Image<u8, 1, _> = Image::from_size_val(
            ImageSize {
                width: buf_size.x as usize,
                height: buf_size.y as usize,
            },
            0,
            CpuAllocator,
        )
        .unwrap();

        let mut img_mask_pyramid = None;
        if mask.is_some() {
            img_mask_pyramid = Some(
                Image::<u8, 1, _>::from_size_val(
                    ImageSize {
                        width: buf_size.x as usize,
                        height: buf_size.y as usize,
                    },
                    0u8,
                    CpuAllocator,
                )
                .unwrap(),
            );
        }

        let prev_img = src;
        let prev_mask = mask;

        (0..n_levels).for_each(|level| {
            let l_info = layer_info[level];
            let sz = Point2d {
                x: l_info.width,
                y: l_info.height,
            };
            let whole_size = Point2d {
                x: sz.x + border as u32 * 2,
                y: sz.y + border as u32 * 2,
            };
            let whole_l_info = Rect {
                x: l_info.x - border as u32,
                y: l_info.y - border as u32,
                width: whole_size.x,
                height: whole_size.y,
            };

            let ext_img = ImageView::from_rect(&img_pyramid, &whole_l_info);
            let curr_img = ImageView::from_view_rect(
                &ext_img,
                &Rect {
                    x: border as u32,
                    y: border as u32,
                    width: sz.x,
                    height: sz.y,
                },
            );

            let (ext_mask, curr_mask) = match mask {
                Some(mask) => {
                    let ext_mask = ImageView::from_rect(mask, &whole_l_info);
                    let curr_mask = ImageView::from_view_rect(
                        &ext_mask,
                        &Rect {
                            x: border as u32,
                            y: border as u32,
                            width: sz.x,
                            height: sz.y,
                        },
                    );

                    (Some(ext_mask), Some(curr_mask))
                }
                None => (None, None),
            };

            if level != self.first_level as usize {}
        });

        Ok(())
    }
}

fn get_scale(level: i32, first_level: i32, scale_factor: f32) -> f32 {
    scale_factor.powi(level - first_level)
}

#[derive(Clone, Copy)]
struct ImageView<'a, T, A: ImageAllocator> {
    img: &'a Image<T, 1, A>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

impl<'a, T, A: ImageAllocator> ImageView<'a, T, A> {
    fn from_rect(img: &'a Image<T, 1, A>, rect: &Rect) -> Self {
        Self {
            img,
            x: rect.x as usize,
            y: rect.y as usize,
            width: rect.width as usize,
            height: rect.height as usize,
        }
    }

    fn from_view_rect(view: &'_ Self, rect: &Rect) -> Self {
        Self {
            img: view.img,
            x: view.x + rect.x as usize,
            y: view.y + rect.y as usize,
            width: rect.width as usize,
            height: rect.height as usize,
        }
    }
}

struct ImageViewIterator<'a, T, A: ImageAllocator> {
    view: ImageView<'a, T, A>,
    current_height: usize,
    current_x: usize,
}

impl<'a, T, A: ImageAllocator> IntoIterator for ImageView<'a, T, A> {
    type Item = &'a T;

    type IntoIter = ImageViewIterator<'a, T, A>;

    fn into_iter(self) -> Self::IntoIter {
        ImageViewIterator {
            view: self,
            current_height: 0,
            current_x: 0,
        }
    }
}

impl<'a, T, A: ImageAllocator> Iterator for ImageViewIterator<'a, T, A> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_height >= self.view.height {
            return None;
        }

        let idx = (self.current_height + self.view.y) * self.view.img.width()
            + (self.current_x + self.view.x);
        let result = Some(&self.view.img.as_slice()[idx]);

        self.current_x += 1;
        if self.current_x >= self.view.width {
            self.current_x = 0;
            self.current_height += 1;
        }

        result
    }
}
