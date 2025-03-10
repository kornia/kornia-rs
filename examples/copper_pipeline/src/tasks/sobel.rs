use cu29::prelude::*;
use kornia::{image::Image, imgproc};

use super::cu_image::{ImageGrayU8Msg, ImageRGBU8Msg};

pub struct Sobel;

impl Freezable for Sobel {}

impl<'cl> CuTask<'cl> for Sobel {
    type Input = input_msg!('cl, ImageRGBU8Msg);
    type Output = output_msg!('cl, ImageGrayU8Msg);

    fn new(_config: Option<&ComponentConfig>) -> Result<Self, CuError>
    where
        Self: Sized,
    {
        Ok(Self {})
    }

    fn start(&mut self, _clock: &RobotClock) -> Result<(), CuError> {
        Ok(())
    }

    fn stop(&mut self, _clock: &RobotClock) -> Result<(), CuError> {
        Ok(())
    }

    fn process(
        &mut self,
        _clock: &RobotClock,
        input: Self::Input,
        output: Self::Output,
    ) -> Result<(), CuError> {
        let Some(src) = input.payload() else {
            return Ok(());
        };

        println!("src: {:?}", src.image.size());

        let img = src
            .image
            .channel(0)
            .map_err(|e| CuError::new_with_cause("Failed to get channel", e))?;

        let img_f32 = img
            .cast_and_scale(1.0f32 / 255.0f32)
            .map_err(|e| CuError::new_with_cause("Failed to cast image", e))?;

        let mut img_sobel = Image::from_size_val(img_f32.size(), 0.0f32)
            .map_err(|e| CuError::new_with_cause("Failed to create image", e))?;

        imgproc::filter::sobel(&img_f32, &mut img_sobel, 3)
            .map_err(|e| CuError::new_with_cause("Failed to apply sobel", e))?;

        let dst = img_sobel
            .scale_and_cast::<u8>(255.0f32)
            .map_err(|e| CuError::new_with_cause("Failed to cast image", e))?;

        output.set_payload(ImageGrayU8Msg { image: dst });

        Ok(())
    }
}
