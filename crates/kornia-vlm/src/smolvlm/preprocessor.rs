use candle_core::{Device, Shape, Tensor};
use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    Image, ImageSize,
};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast};

use crate::smolvlm::utils::SmolVlmError;

// ImageNet mean and std for normalization
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

type Image3<A> = Image<u8, 3, A>;
type Image1<A> = Image<u8, 1, A>;

/// Image preprocessor for SmolVLM model
#[derive(Clone)]
pub struct SmolVlmImagePreprocessor {
    max_size: u32,
    outer_patch_size: u32,
}

impl SmolVlmImagePreprocessor {
    /// Create a new SmolVLM image preprocessor
    pub fn new(max_size: u32, outer_patch_size: u32) -> Self {
        Self {
            max_size,
            outer_patch_size,
        }
    }

    /// Preprocess an image for SmolVLM model inference
    pub fn preprocess<A: ImageAllocator>(
        &self,
        img: &Image<u8, 3, A>,
        device: &Device,
    ) -> Result<(Tensor, Tensor, ImageSize), SmolVlmError> {
        // resizing image to match the max_size (on the longest edge)
        let img_resized = self.resize_image_to_fit(img, self.max_size)?;

        // create global image resized to outer_patch_size
        let global_img = self.resize_image_to_fit(img, self.outer_patch_size)?;

        // padding images for all dimensions to be multiples of the outer_patch_size
        let (img_padded, mask) = self.pad_image_with_mask(&img_resized)?;
        let (global_img, global_mask) = self.pad_image_with_mask(&global_img)?;

        // convert to tensors and normalize
        let img_tensor = self.image_to_normalized_tensor(&img_padded, device)?;
        let mask_tensor = self.mask_to_tensor(mask, device)?;
        let global_img_tensor = self.image_to_normalized_tensor(&global_img, device)?;
        let global_mask_tensor = self.mask_to_tensor(global_mask, device)?;

        // create patches and concatenate with global image
        let (img_patches, mask_patches, size) = self.create_patches_with_global(
            &img_tensor,
            &mask_tensor,
            &global_img_tensor,
            &global_mask_tensor,
        )?;

        Ok((img_patches, mask_patches, size))
    }

    /// Generic resize method that resizes image to fit within target_size while maintaining aspect ratio
    fn resize_image_to_fit<A: ImageAllocator>(
        &self,
        img: &Image<u8, 3, A>,
        target_size: u32,
    ) -> Result<Image<u8, 3, A>, SmolVlmError> {
        let (width, height) = (img.width() as u32, img.height() as u32);
        let longest_edge = width.max(height);

        if longest_edge <= target_size {
            // TODO: can we avoid cloning here? and return a reference?
            Ok(img.clone())
        } else {
            let scale_factor = target_size as f32 / longest_edge as f32;
            let new_width = (width as f32 * scale_factor) as usize;
            let new_height = (height as f32 * scale_factor) as usize;

            // TODO: lazily allocate the image
            let mut resized = Image::<u8, 3, A>::from_size_val(
                ImageSize {
                    width: new_width,
                    height: new_height,
                },
                0,
                img.0.storage.alloc().clone(),
            )?;

            resize_fast(img, &mut resized, InterpolationMode::Bilinear)?;
            Ok(resized)
        }
    }

    /// Pad image to be multiples of outer_patch_size and create corresponding mask
    fn pad_image_with_mask<A: ImageAllocator>(
        &self,
        img: &Image<u8, 3, A>,
    ) -> Result<(Image3<CpuAllocator>, Image1<CpuAllocator>), SmolVlmError> {
        let (width, height) = (img.width(), img.height());
        let new_width = (width as u32).div_ceil(self.outer_patch_size) * self.outer_patch_size;
        let new_height = (height as u32).div_ceil(self.outer_patch_size) * self.outer_patch_size;

        let mut padded_img = Image::<u8, 3, _>::from_size_val(
            ImageSize {
                width: new_width as usize,
                height: new_height as usize,
            },
            0,
            CpuAllocator,
        )?;

        let mut padded_mask = Image::<u8, 1, _>::from_size_val(
            ImageSize {
                width: new_width as usize,
                height: new_height as usize,
            },
            0,
            CpuAllocator,
        )?;

        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    // TODO: this method are slow, we should use a more efficient way to set the pixel maybe with iterators
                    padded_img.set_pixel(x, y, c, *img.get_pixel(x, y, c)?)?;
                }
                padded_mask.set_pixel(x, y, 0, 255)?;
            }
        }

        Ok((padded_img, padded_mask))
    }

    /// Convert image to normalized tensor
    fn image_to_normalized_tensor(
        &self,
        img: &Image<u8, 3, CpuAllocator>,
        device: &Device,
    ) -> Result<Tensor, SmolVlmError> {
        let (width, height) = (img.width(), img.height());
        let img_data = img.as_slice();

        // TODO: check if make a copy of the image
        let mut tensor =
            Tensor::from_slice(img_data, Shape::from_dims(&[height, width, 3]), device)?
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
        &self,
        mask: Image<u8, 1, CpuAllocator>,
        device: &Device,
    ) -> Result<Tensor, SmolVlmError> {
        let (width, height) = (mask.width(), mask.height());
        let mask_data = mask.as_slice();

        Ok(
            Tensor::from_slice(mask_data, Shape::from_dims(&[height, width]), device)?
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
        let img = Image3::<CpuAllocator>::from_size_val(
            ImageSize {
                width: 64,
                height: 64,
            },
            128, // gray value
            CpuAllocator,
        )?;

        let preprocessor = SmolVlmImagePreprocessor::new(512, 32);
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
        let img = Image3::<CpuAllocator>::from_size_val(
            ImageSize {
                width: 100,
                height: 200,
            },
            0,
            CpuAllocator,
        )?;

        let preprocessor = SmolVlmImagePreprocessor::new(512, 32);

        // Test no resize needed
        let result = preprocessor.resize_image_to_fit(&img, 300)?;
        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 200);

        // Test resize needed
        let result = preprocessor.resize_image_to_fit(&img, 100)?;
        assert_eq!(result.width(), 50); // scaled down proportionally
        assert_eq!(result.height(), 100); // longest edge = 100

        Ok(())
    }
}
