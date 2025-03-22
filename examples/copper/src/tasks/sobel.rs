use cu29::prelude::*;
use kornia::{image::Image, imgproc};

use super::cu_image::{ImageGray8Msg, ImageRgb8Msg};

pub struct Sobel;

impl Freezable for Sobel {}

impl<'cl> CuTask<'cl> for Sobel {
    type Input = input_msg!('cl, ImageRgb8Msg);
    type Output = output_msg!('cl, ImageGray8Msg);

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

        let img = src
            .channel(0)
            .map_err(|e| CuError::new_with_cause("Failed to get channel", e))?
            .map(|&x| x as f32)
            .map_err(|e| CuError::new_with_cause("Failed to cast image to f32", e))?;

        let mut img_sobel = Image::from_size_val(img.size(), 0.0f32)
            .map_err(|e| CuError::new_with_cause("Failed to create image", e))?;

        imgproc::filter::sobel(&img, &mut img_sobel, 3)
            .map_err(|e| CuError::new_with_cause("Failed to apply sobel", e))?;

        let dst = img_sobel
            .map(|&x| x as u8)
            .map_err(|e| CuError::new_with_cause("Failed to cast image to u8", e))?;

        output.set_payload(ImageGray8Msg(dst));

        Ok(())
    }
}
