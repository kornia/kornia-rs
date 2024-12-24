use cu29::prelude::*;
use kornia::io::stream::{CameraCapture, V4L2CameraConfig};

use crate::tasks::ImageRGBU8Msg;

// default config for the webcam
const DEFAULT_CAMERA_ID: u32 = 0;
const DEFAULT_RES_ROWS: u32 = 480;
const DEFAULT_RES_COLS: u32 = 640;
const DEFAULT_FPS: u32 = 30;

pub struct Webcam {
    cam: CameraCapture,
}

impl Freezable for Webcam {}

impl<'cl> CuSrcTask<'cl> for Webcam {
    type Output = output_msg!('cl, ImageRGBU8Msg);

    fn new(config: Option<&ComponentConfig>) -> Result<Self, CuError>
    where
        Self: Sized,
    {
        let (camera_id, res_rows, res_cols, fps) = if let Some(config) = config {
            let camera_id = config.get::<u32>("camera_id").unwrap_or(DEFAULT_CAMERA_ID);
            let res_rows = config.get::<u32>("res_rows").unwrap_or(DEFAULT_RES_ROWS);
            let res_cols = config.get::<u32>("res_cols").unwrap_or(DEFAULT_RES_COLS);
            let fps = config.get::<u32>("fps").unwrap_or(DEFAULT_FPS);
            (camera_id, res_rows, res_cols, fps)
        } else {
            (
                DEFAULT_CAMERA_ID,
                DEFAULT_RES_ROWS,
                DEFAULT_RES_COLS,
                DEFAULT_FPS,
            )
        };

        let cam = V4L2CameraConfig::new()
            .with_camera_id(camera_id)
            .with_fps(fps)
            .with_size([res_cols as usize, res_rows as usize].into())
            .build()
            .map_err(|e| CuError::new_with_cause("Failed to build camera", e))?;

        Ok(Self { cam })
    }

    fn start(&mut self, _clock: &RobotClock) -> Result<(), CuError> {
        println!("Webcam::start");
        self.cam
            .start()
            .map_err(|e| CuError::new_with_cause("Failed to start camera", e))
    }

    fn stop(&mut self, _clock: &RobotClock) -> Result<(), CuError> {
        println!("Webcam::stop");
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
