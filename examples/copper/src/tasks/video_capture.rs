use cu29::prelude::*;
use kornia::io::stream::{CameraCapture, RTSPCameraConfig, V4L2CameraConfig};

use super::cu_image::ImageRgb8Msg;

pub struct VideoCapture(pub CameraCapture);

impl Freezable for VideoCapture {}

impl<'cl> CuSrcTask<'cl> for VideoCapture {
    type Output = output_msg!('cl, ImageRgb8Msg);

    fn new(config: Option<&ComponentConfig>) -> Result<Self, CuError>
    where
        Self: Sized,
    {
        let Some(config) = config else {
            return Err(CuError::from("No config provided"));
        };

        let source_type = config
            .get::<String>("source_type")
            .ok_or(CuError::from("No source type provided"))?;

        let source_uri = config
            .get::<String>("source_uri")
            .ok_or(CuError::from("No source uri provided"))?;

        let cam = match source_type.as_str() {
            "rtsp" => RTSPCameraConfig::new()
                .with_url(&source_uri)
                .build()
                .map_err(|e| CuError::new_with_cause("Failed to build camera", e))?,
            "v4l2" => {
                // parse the needed parameters from the config
                let image_cols = config
                    .get::<u32>("image_cols")
                    .ok_or(CuError::from("No image cols provided"))?;
                let image_rows = config
                    .get::<u32>("image_rows")
                    .ok_or(CuError::from("No image rows provided"))?;
                let source_fps = config
                    .get::<u32>("source_fps")
                    .ok_or(CuError::from("No source fps provided"))?;

                V4L2CameraConfig::new()
                    .with_device(&source_uri)
                    .with_fps(source_fps)
                    .with_size([image_cols as usize, image_rows as usize].into())
                    .build()
                    .map_err(|e| CuError::new_with_cause("Failed to build camera", e))?
            }
            _ => return Err(CuError::from("Invalid source type")),
        };

        Ok(Self(cam))
    }

    fn start(&mut self, _clock: &RobotClock) -> Result<(), CuError> {
        self.0
            .start()
            .map_err(|e| CuError::new_with_cause("Failed to start camera", e))
    }

    fn stop(&mut self, _clock: &RobotClock) -> Result<(), CuError> {
        self.0
            .close()
            .map_err(|e| CuError::new_with_cause("Failed to stop camera", e))
    }

    fn process(&mut self, _clock: &RobotClock, output: Self::Output) -> Result<(), CuError> {
        let Some(img) = self
            .0
            .grab()
            .map_err(|e| CuError::new_with_cause("Failed to grab image", e))?
        else {
            return Ok(());
        };

        output.set_payload(ImageRgb8Msg(img));

        Ok(())
    }
}
