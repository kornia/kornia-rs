use cu29::prelude::*;

use crate::tasks::ImageRGBU8Msg;

pub struct RerunViz {
    rec: rerun::RecordingStream,
}

impl Freezable for RerunViz {}

impl<'cl> CuSinkTask<'cl> for RerunViz {
    type Input = input_msg!('cl, ImageRGBU8Msg);

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
        let Some(img) = input.payload() else {
            return Ok(());
        };

        // println!("{:?}", img.image.as_slice());
        println!("RerunViz::process: logging image: {:?}", img.image.size());

        self.rec
            .log(
                "image",
                &rerun::Image::from_elements(
                    img.image.as_slice(),
                    img.image.size().into(),
                    rerun::ColorModel::RGB,
                ),
            )
            .map_err(|e| CuError::new_with_cause("Failed to log image", e))?;

        println!("RerunViz::process: logged image");

        Ok(())
    }
}
