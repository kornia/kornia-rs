use crate::smolvlm::utils::SmolVlmError;
use candle_core::{Device, Shape, Tensor};
use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    Image, ImageSize,
};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast};
use std::borrow::Cow;

// ImageNet mean and std for normalization
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Image preprocessor for SmolVLM model
pub struct SmolVlmImagePreprocessor {
    max_size: u32,
    outer_patch_size: u32,

    // buffers for resizing images
    buf_resize: Option<Image<u8, 3, CpuAllocator>>,
    buf_global_resize: Option<Image<u8, 3, CpuAllocator>>,

    // buffers for padding images
    buf_padded_img: Option<Image<u8, 3, CpuAllocator>>,
    buf_padded_mask: Option<Image<u8, 1, CpuAllocator>>,
    buf_global_padded_img: Option<Image<u8, 3, CpuAllocator>>,
    buf_global_padded_mask: Option<Image<u8, 1, CpuAllocator>>,
}

impl SmolVlmImagePreprocessor {
    /// Create a new SmolVLM image preprocessor
    pub fn new(max_size: u32, outer_patch_size: u32) -> Self {
        Self {
            max_size,
            outer_patch_size,
            buf_resize: None,
            buf_global_resize: None,
            buf_padded_img: None,
            buf_padded_mask: None,
            buf_global_padded_img: None,
            buf_global_padded_mask: None,
        }
    }

    /// Preprocess an image for SmolVLM model inference
    pub fn preprocess<A: ImageAllocator>(
        &mut self,
        img: &Image<u8, 3, A>,
        device: &Device,
    ) -> Result<(Tensor, Tensor, ImageSize), SmolVlmError> {
        // Process main image: resize then pad in-place
        {
            let img_resized =
                Self::resize_image_with_buffer(img, &mut self.buf_resize, self.max_size)?;
            match img_resized {
                Cow::Borrowed(img_ref) => {
                    Self::pad_image_in_place(
                        img_ref,
                        &mut self.buf_padded_img,
                        &mut self.buf_padded_mask,
                        self.outer_patch_size,
                    )?;
                }
                Cow::Owned(img_owned) => {
                    Self::pad_image_in_place(
                        &img_owned,
                        &mut self.buf_padded_img,
                        &mut self.buf_padded_mask,
                        self.outer_patch_size,
                    )?;
                }
            }
        }

        // Process global image: resize then pad in-place
        {
            let global_resized = Self::resize_image_with_buffer(
                img,
                &mut self.buf_global_resize,
                self.outer_patch_size,
            )?;
            match global_resized {
                Cow::Borrowed(img_ref) => {
                    Self::pad_image_in_place(
                        img_ref,
                        &mut self.buf_global_padded_img,
                        &mut self.buf_global_padded_mask,
                        self.outer_patch_size,
                    )?;
                }
                Cow::Owned(img_owned) => {
                    Self::pad_image_in_place(
                        &img_owned,
                        &mut self.buf_global_padded_img,
                        &mut self.buf_global_padded_mask,
                        self.outer_patch_size,
                    )?;
                }
            }
        }

        // Now we can safely access the buffers
        let img_padded = self.buf_padded_img.take().unwrap();
        let mask = self.buf_padded_mask.take().unwrap(); // Take ownership
        let global_img = self.buf_global_padded_img.take().unwrap();
        let global_mask = self.buf_global_padded_mask.take().unwrap(); // Take ownership

        // convert to tensors and normalize
        // Process masks first (they need mutable self)
        let mask_tensor = Self::mask_to_tensor(mask, device)?;
        let global_mask_tensor = Self::mask_to_tensor(global_mask, device)?;

        // Then process images (they need immutable self)
        let img_tensor = Self::image_to_normalized_tensor(img_padded, device)?;
        let global_img_tensor = Self::image_to_normalized_tensor(global_img, device)?;

        // create patches and concatenate with global image
        let (img_patches, mask_patches, size) = self.create_patches_with_global(
            &img_tensor,
            &mask_tensor,
            &global_img_tensor,
            &global_mask_tensor,
        )?;

        Ok((img_patches, mask_patches, size))
    }

    /// Single resize function that takes buffer by value and returns it (lazy init)
    fn resize_image_with_buffer<'a, A: ImageAllocator>(
        img: &'a Image<u8, 3, A>,
        buffer: &'a mut Option<Image<u8, 3, CpuAllocator>>,
        target_size: u32,
    ) -> Result<Cow<'a, Image<u8, 3, CpuAllocator>>, SmolVlmError> {
        let (width, height) = (img.width() as u32, img.height() as u32);
        let longest_edge = width.max(height);

        if longest_edge <= target_size {
            // Return borrowed reference - no copy needed
            Ok(Cow::Borrowed(unsafe {
                std::mem::transmute::<&Image<u8, 3, A>, &Image<u8, 3, CpuAllocator>>(img)
            }))
        } else {
            // Resize case - use and return the buffer directly
            let scale_factor = target_size as f32 / longest_edge as f32;
            let new_width = (width as f32 * scale_factor) as usize;
            let new_height = (height as f32 * scale_factor) as usize;

            // Lazy init buffer only if size changed
            let needs_resize = buffer.as_ref().map_or(true, |buf| {
                buf.width() != new_width || buf.height() != new_height
            });

            if needs_resize {
                *buffer = Some(Image::<u8, 3, CpuAllocator>::from_size_val(
                    ImageSize {
                        width: new_width,
                        height: new_height,
                    },
                    0,
                    CpuAllocator,
                )?);
            }

            let buf = buffer.as_mut().unwrap();
            resize_fast(img, buf, InterpolationMode::Bilinear)?;

            Ok(Cow::Borrowed(buf))
        }
    }

    /// Pad image to be multiples of outer_patch_size and create corresponding mask
    /// Returns references to the internal buffers to avoid copying
    fn pad_image_in_place<A: ImageAllocator>(
        img: &Image<u8, 3, A>,
        img_buffer: &mut Option<Image<u8, 3, CpuAllocator>>,
        mask_buffer: &mut Option<Image<u8, 1, CpuAllocator>>,
        outer_patch_size: u32,
    ) -> Result<(), SmolVlmError> {
        let (width, height) = (img.width(), img.height());
        let new_width = (width as u32).div_ceil(outer_patch_size) * outer_patch_size;
        let new_height = (height as u32).div_ceil(outer_patch_size) * outer_patch_size;

        // Lazy init buffers only if size changed
        let needs_resize = img_buffer.as_ref().map_or(true, |buf| {
            buf.width() != new_width as usize || buf.height() != new_height as usize
        });

        if needs_resize {
            *img_buffer = Some(Image::<u8, 3, CpuAllocator>::from_size_val(
                ImageSize {
                    width: new_width as usize,
                    height: new_height as usize,
                },
                0,
                CpuAllocator,
            )?);
        }

        let needs_mask_resize = mask_buffer.as_ref().map_or(true, |buf| {
            buf.width() != new_width as usize || buf.height() != new_height as usize
        });

        if needs_mask_resize {
            *mask_buffer = Some(Image::<u8, 1, CpuAllocator>::from_size_val(
                ImageSize {
                    width: new_width as usize,
                    height: new_height as usize,
                },
                0,
                CpuAllocator,
            )?);
        }

        let padded_img = img_buffer.as_mut().unwrap();
        let padded_mask = mask_buffer.as_mut().unwrap();

        // Zero out buffers (padding area)
        padded_img.as_slice_mut().fill(0);
        padded_mask.as_slice_mut().fill(0);

        // Fast row-by-row copying using slices
        let img_slice = img.as_slice();
        let padded_img_slice = padded_img.as_slice_mut();
        let padded_mask_slice = padded_mask.as_slice_mut();

        // Copy image data row by row
        for y in 0..height {
            let src_offset = y * width * 3;
            let dst_offset = y * new_width as usize * 3;
            let row_bytes = width * 3;

            padded_img_slice[dst_offset..dst_offset + row_bytes]
                .copy_from_slice(&img_slice[src_offset..src_offset + row_bytes]);
        }

        // Set mask for valid region
        for y in 0..height {
            let mask_offset = y * new_width as usize;
            padded_mask_slice[mask_offset..mask_offset + width].fill(255);
        }

        Ok(())
    }

    /// Convert image to normalized tensor
    fn image_to_normalized_tensor(
        img: Image<u8, 3, CpuAllocator>, // Take ownership
        device: &Device,
    ) -> Result<Tensor, SmolVlmError> {
        let (width, height) = (img.width(), img.height());
        let img_data = img.into_vec(); // No clone, take ownership

        let mut tensor = Tensor::from_vec(img_data, Shape::from_dims(&[height, width, 3]), device)?
            .permute(vec![2, 0, 1])?
            .to_dtype(candle_core::DType::F32)?;

        // Normalize: scale to [0,1] then apply ImageNet normalization
        tensor = (tensor / 255.0)?;
        let mean = Tensor::from_slice(&MEAN, (3, 1, 1), device)?;
        let std = Tensor::from_slice(&STD, (3, 1, 1), device)?;

        tensor = tensor.broadcast_sub(&mean)?;
        tensor = tensor.broadcast_div(&std)?;

        Ok(tensor)
    }

    /// Convert mask to tensor
    fn mask_to_tensor(
        mask: Image<u8, 1, CpuAllocator>, // Take ownership of the mask
        device: &Device,
    ) -> Result<Tensor, SmolVlmError> {
        let (width, height) = (mask.width(), mask.height());
        let mask_data = mask.into_vec(); // Take ownership and convert to vec

        Ok(
            Tensor::from_vec(mask_data, Shape::from_dims(&[height, width]), device)?
                .to_dtype(candle_core::DType::F32)?,
        )
    }

    /// Create patches from image and mask, then concatenate with global image
    fn create_patches_with_global(
        &self,
        img: &Tensor,
        mask: &Tensor,
        global_img: &Tensor,
        global_mask: &Tensor,
    ) -> Result<(Tensor, Tensor, ImageSize), SmolVlmError> {
        let (c, h, w) = img.dims3()?;
        let cols = w / self.outer_patch_size as usize;
        let rows = h / self.outer_patch_size as usize;

        // Split image into patches
        let img_patches = img
            .unsqueeze(2)?
            .unsqueeze(4)?
            .reshape(&[
                c,
                rows,
                self.outer_patch_size as usize,
                cols,
                self.outer_patch_size as usize,
            ])?
            .permute([1, 3, 0, 2, 4])?
            .reshape(&[
                rows * cols,
                c,
                self.outer_patch_size as usize,
                self.outer_patch_size as usize,
            ])?;

        // Split mask into patches
        let mask_patches = mask
            .unsqueeze(1)?
            .unsqueeze(3)?
            .reshape(&[
                rows,
                self.outer_patch_size as usize,
                cols,
                self.outer_patch_size as usize,
            ])?
            .permute([0, 2, 1, 3])?
            .reshape(&[
                rows * cols,
                self.outer_patch_size as usize,
                self.outer_patch_size as usize,
            ])?;

        // Concatenate global image with patches
        let img_patches = Tensor::cat(&[&img_patches, &global_img.unsqueeze(0)?], 0)?;
        let mask_patches = Tensor::cat(&[&mask_patches, &global_mask.unsqueeze(0)?], 0)?;

        Ok((
            img_patches,
            mask_patches,
            ImageSize {
                width: cols,
                height: rows,
            },
        ))
    }
}

pub fn get_prompt_split_image(img_seq_len: usize, size: ImageSize) -> String {
    let mut s = String::new();

    for h in 0..size.height {
        for w in 0..size.width {
            s += &format!(
                "<fake_token_around_image><row_{}_col_{}>{}",
                h + 1,
                w + 1,
                "<image>".repeat(img_seq_len)
            );
        }
        s += "\n";
    }

    s += &format!(
        "\n<fake_token_around_image><global-img>{}<fake_token_around_image>",
        "<image>".repeat(img_seq_len)
    );

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use kornia_image::allocator::CpuAllocator;

    #[test]
    fn test_smolvlm_preprocessor_basic() -> Result<(), SmolVlmError> {
        // Create a simple test image (64x64, RGB)
        let img = Image::<u8, 3, CpuAllocator>::from_size_val(
            ImageSize {
                width: 64,
                height: 64,
            },
            128, // gray value
            CpuAllocator,
        )?;

        let mut preprocessor = SmolVlmImagePreprocessor::new(512, 32);
        let device = Device::Cpu;

        let (img_patches, mask_patches, size) = preprocessor.preprocess(&img, &device)?;

        // Check that we got the expected dimensions
        assert_eq!(img_patches.dims().len(), 4); // [patches, channels, height, width]
        assert_eq!(mask_patches.dims().len(), 3); // [patches, height, width]

        // Should have patch dimensions + 1 global image
        let expected_patches = (size.width * size.height) + 1;
        assert_eq!(img_patches.dims()[0], expected_patches);
        assert_eq!(mask_patches.dims()[0], expected_patches);

        // Check patch size
        assert_eq!(img_patches.dims()[2], 32); // patch height
        assert_eq!(img_patches.dims()[3], 32); // patch width
        assert_eq!(mask_patches.dims()[1], 32); // patch height
        assert_eq!(mask_patches.dims()[2], 32); // patch width

        // Check channels
        assert_eq!(img_patches.dims()[1], 3); // RGB

        Ok(())
    }

    #[test]
    fn test_resize_image_to_fit() -> Result<(), SmolVlmError> {
        let img = Image::<u8, 3, CpuAllocator>::from_size_val(
            ImageSize {
                width: 100,
                height: 200,
            },
            0,
            CpuAllocator,
        )?;

        // Test no resize needed
        let mut buffer = None;
        let result = SmolVlmImagePreprocessor::resize_image_with_buffer(&img, &mut buffer, 300)?;
        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 200);

        // Test resize needed
        let mut buffer = None;
        let result = SmolVlmImagePreprocessor::resize_image_with_buffer(&img, &mut buffer, 100)?;
        assert_eq!(result.width(), 50); // scaled down proportionally
        assert_eq!(result.height(), 100); // longest edge = 100

        Ok(())
    }

    #[test]
    fn test_small_image_explicit_values() -> Result<(), SmolVlmError> {
        // Create a small 6x4 image with explicit RGB values
        let mut img_data = vec![0u8; 6 * 4 * 3]; // 6 width x 4 height x 3 channels

        // Set explicit pixel values in a pattern (R, G, B)
        // Top row: Red gradient
        for x in 0..6 {
            let offset = x * 3;
            img_data[offset] = (x * 40) as u8; // R: 0, 40, 80, 120, 160, 200
            img_data[offset + 1] = 0; // G: 0
            img_data[offset + 2] = 0; // B: 0
        }

        // Second row: Green gradient
        for x in 0..6 {
            let offset = (6 + x) * 3;
            img_data[offset] = 0; // R: 0
            img_data[offset + 1] = (x * 40) as u8; // G: 0, 40, 80, 120, 160, 200
            img_data[offset + 2] = 0; // B: 0
        }

        // Third row: Blue gradient
        for x in 0..6 {
            let offset = (12 + x) * 3;
            img_data[offset] = 0; // R: 0
            img_data[offset + 1] = 0; // G: 0
            img_data[offset + 2] = (x * 40) as u8; // B: 0, 40, 80, 120, 160, 200
        }

        // Fourth row: Mixed colors
        for x in 0..6 {
            let offset = (18 + x) * 3;
            img_data[offset] = (x * 20) as u8; // R: 0, 20, 40, 60, 80, 100
            img_data[offset + 1] = (x * 30) as u8; // G: 0, 30, 60, 90, 120, 150
            img_data[offset + 2] = (x * 10) as u8; // B: 0, 10, 20, 30, 40, 50
        }

        let img = Image::<u8, 3, CpuAllocator>::new(
            ImageSize {
                width: 6,
                height: 4,
            },
            img_data.clone(),
            CpuAllocator,
        )?;

        let mut preprocessor = SmolVlmImagePreprocessor::new(8, 4); // small sizes for testing

        // First, test just the padding to verify it works correctly
        let mut img_buffer = None;
        let mut mask_buffer = None;
        SmolVlmImagePreprocessor::pad_image_in_place(&img, &mut img_buffer, &mut mask_buffer, 4)?;

        let padded_img_data = img_buffer.as_ref().unwrap().as_slice();
        let original_img_data = img.as_slice();

        // Check each row
        for y in 0..4 {
            let original_row_start = y * 6 * 3;
            let padded_row_start = y * 8 * 3;

            // Verify original data is intact
            assert_eq!(
                &padded_img_data[padded_row_start..padded_row_start + 6 * 3],
                &original_img_data[original_row_start..original_row_start + 6 * 3]
            );

            // Verify padding columns are zero
            assert_eq!(
                &padded_img_data[padded_row_start + 6 * 3..padded_row_start + 8 * 3],
                &[0u8; 2 * 3]
            );
        }

        // Now test the full preprocessing pipeline
        let device = Device::Cpu;
        let (img_patches, mask_patches, size) = preprocessor.preprocess(&img, &device)?;

        // Verify basic structure
        assert_eq!(img_patches.dims().len(), 4); // [patches, channels, height, width]
        assert_eq!(mask_patches.dims().len(), 3); // [patches, height, width]

        // Should have 2x1 patches (8x4 padded image / 4x4 patches) + 1 global = 3 total
        let expected_patches = (size.width * size.height) + 1;
        assert_eq!(img_patches.dims()[0], expected_patches);
        assert_eq!(mask_patches.dims()[0], expected_patches);
        assert_eq!(size.width, 2); // 8 / 4 = 2 patches wide
        assert_eq!(size.height, 1); // 4 / 4 = 1 patch tall

        // Check patch dimensions
        assert_eq!(img_patches.dims()[2], 4); // patch height
        assert_eq!(img_patches.dims()[3], 4); // patch width
        assert_eq!(mask_patches.dims()[1], 4); // patch height
        assert_eq!(mask_patches.dims()[2], 4); // patch width

        // Check channels
        assert_eq!(img_patches.dims()[1], 3); // RGB

        Ok(())
    }
}
