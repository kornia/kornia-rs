//! Image processing for SmolVLM

use std::path::Path;

use image::{imageops, io::Reader as ImageReader, DynamicImage, GenericImageView};

use crate::smolvlm::common::{ProcessedImage, SmolVLMConfig, SmolVLMError};

/// ImageProcessor handles all image preprocessing for SmolVLM
#[derive(Debug)]
pub struct ImageProcessor {
    config: SmolVLMConfig,
}

impl ImageProcessor {
    /// Create a new image processor from the given configuration
    pub fn new(config: &SmolVLMConfig) -> Result<Self, SmolVLMError> {
        // Validate configuration
        config.validate()?;

        Ok(Self {
            config: config.clone(),
        })
    }

    /// Process an image from a file path
    pub fn process_image_from_path(
        &self,
        image_path: &str,
    ) -> Result<ProcessedImage, SmolVLMError> {
        // Check if file exists
        if !Path::new(image_path).exists() {
            return Err(SmolVLMError::ImageProcessingError(format!(
                "Image file not found: {}",
                image_path
            )));
        }

        // Load the image
        let img = ImageReader::open(image_path)
            .map_err(|e| {
                SmolVLMError::ImageProcessingError(format!("Failed to open image: {}", e))
            })?
            .decode()
            .map_err(|e| {
                SmolVLMError::ImageProcessingError(format!("Failed to decode image: {}", e))
            })?;

        // Process the image
        self.process_image(img)
    }

    /// Process an image from bytes
    pub fn process_image_from_bytes(
        &self,
        image_bytes: &[u8],
    ) -> Result<ProcessedImage, SmolVLMError> {
        // Load the image from bytes
        let img = image::load_from_memory(image_bytes).map_err(|e| {
            SmolVLMError::ImageProcessingError(format!("Failed to load image from bytes: {}", e))
        })?;

        // Process the image
        self.process_image(img)
    }

    /// Process a loaded image
    fn process_image(&self, img: DynamicImage) -> Result<ProcessedImage, SmolVLMError> {
        // Resize the image to the expected dimensions
        let (target_width, target_height) = self.config.image_size;
        let resized = img.resize_exact(
            target_width as u32,
            target_height as u32,
            imageops::FilterType::Lanczos3,
        );

        // Convert to RGB if not already
        let rgb = resized.to_rgb8();

        // Normalize pixel values to [0, 1] and then apply standard normalization
        // Mean and std values commonly used for ImageNet models
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        // Allocate memory for normalized data
        let num_channels = 3; // RGB
        let mut normalized_data = Vec::with_capacity(target_height * target_width * num_channels);

        // Normalize pixel values
        for pixel in rgb.pixels() {
            for c in 0..num_channels {
                let pixel_value = pixel[c] as f32 / 255.0;
                let normalized_value = (pixel_value - mean[c]) / std[c];
                normalized_data.push(normalized_value);
            }
        }

        // Create the processed image
        // Shape is (batch_size=1, channels=3, height, width)
        let processed_image = ProcessedImage::new(
            normalized_data,
            (1, num_channels, target_height, target_width),
        );

        Ok(processed_image)
    }
}
