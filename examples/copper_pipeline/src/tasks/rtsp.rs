use cu29::prelude::*;
use kornia::io::stream::{CameraCapture, RTSPCameraConfig};

use super::cu_image::ImageRGBU8Msg;

// default config for the rtsp camera
const DEFAULT_URL: &str = "rtsp://admin:admin@192.168.1.100:554/Streaming/Channels/1";

pub struct RtspCamera {
    cam: CameraCapture,
}

impl Freezable for RtspCamera {}

impl<'cl> CuSrcTask<'cl> for RtspCamera {
    type Output = output_msg!('cl, ImageRGBU8Msg);

    fn new(config: Option<&ComponentConfig>) -> Result<Self, CuError>
    where
        Self: Sized,
    {
        let url = if let Some(config) = config {
            config
                .get::<String>("url")
                .unwrap_or(DEFAULT_URL.to_string())
        } else {
            DEFAULT_URL.to_string()
        };

        let cam = RTSPCameraConfig::new()
            .with_url(&url)
            .build()
            .map_err(|e| CuError::new_with_cause("Failed to build camera", e))?;

        Ok(Self { cam })
    }

    fn start(&mut self, _clock: &RobotClock) -> Result<(), CuError> {
        self.cam
            .start()
            .map_err(|e| CuError::new_with_cause("Failed to start camera", e))
    }

    fn stop(&mut self, _clock: &RobotClock) -> Result<(), CuError> {
        self.cam
            .close()
            .map_err(|e| CuError::new_with_cause("Failed to stop camera", e))
    }

    fn process(&mut self, _clock: &RobotClock, output: Self::Output) -> Result<(), CuError> {
        let Some(img) = self
            .cam
            .grab()
            .map_err(|e| CuError::new_with_cause("Failed to grab image", e))?
        else {
            return Ok(());
        };

        output.set_payload(ImageRGBU8Msg { image: img });

        Ok(())
    }
}
