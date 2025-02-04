use cu29::prelude::*;

use super::cu_image::{ImageGrayU8Msg, ImageRGBU8Msg};

pub struct RerunViz {
    rec: rerun::RecordingStream,
}

impl Freezable for RerunViz {}

impl<'cl> CuSinkTask<'cl> for RerunViz {
    type Input = input_msg!('cl, ImageRGBU8Msg, ImageRGBU8Msg, ImageGrayU8Msg);

    fn new(_config: Option<&ComponentConfig>) -> Result<Self, CuError>
    where
        Self: Sized,
    {
        Ok(Self {
            rec: rerun::RecordingStreamBuilder::new("kornia_app")
                .spawn()
                .map_err(|e| CuError::new_with_cause("Failed to spawn rerun stream", e))?,
        })
    }

    fn process(&mut self, _clock: &RobotClock, input: Self::Input) -> Result<(), CuError> {
        let (img1, img2, img3) = input;

        if let Some(img) = img1.payload() {
            log_image_rgb(&self.rec, "webcam", img)?;
        }

        if let Some(img) = img2.payload() {
            log_image_gray(&self.rec, "sobel", img)?;
        }

        if let Some(img) = img3.payload() {
            log_image_rgb(&self.rec, "rtsp", img)?;
        }

        Ok(())
    }
}

fn log_image_rgb(
    rec: &rerun::RecordingStream,
    name: &str,
    img: &ImageRGBU8Msg,
) -> Result<(), CuError> {
    rec.log(
        name,
        &rerun::Image::from_elements(
            img.image.as_slice(),
            img.image.size().into(),
            rerun::ColorModel::RGB,
        ),
    )
    .map_err(|e| CuError::new_with_cause("Failed to log image", e))?;
    Ok(())
}

fn log_image_gray(
    rec: &rerun::RecordingStream,
    name: &str,
    img: &ImageGrayU8Msg,
) -> Result<(), CuError> {
    rec.log(
        name,
        &rerun::Image::from_elements(
            img.image.as_slice(),
            img.image.size().into(),
            rerun::ColorModel::L,
        ),
    )
    .map_err(|e| CuError::new_with_cause("Failed to log image", e))?;
    Ok(())
}
