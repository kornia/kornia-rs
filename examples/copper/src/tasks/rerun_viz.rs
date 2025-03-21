use super::cu_image::{ImageGray8Msg, ImageRgb8Msg};
use cu29::prelude::*;
use std::str::FromStr;

const RERUN_HOST: &str = "127.0.0.1";
const RERUN_PORT: u32 = 9876;

pub struct RerunViz(rerun::RecordingStream);

impl std::ops::Deref for RerunViz {
    type Target = rerun::RecordingStream;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Freezable for RerunViz {}

impl<'cl> CuSinkTask<'cl> for RerunViz {
    type Input = input_msg!('cl, ImageRgb8Msg, ImageRgb8Msg, ImageGray8Msg);

    fn new(config: Option<&ComponentConfig>) -> Result<Self, CuError>
    where
        Self: Sized,
    {
        let (host, port) = if let Some(config) = config {
            (
                config
                    .get::<String>("host")
                    .ok_or(CuError::from("Host is required"))?,
                config
                    .get::<u32>("port")
                    .ok_or(CuError::from("Port is required"))?,
            )
        } else {
            (RERUN_HOST.to_string(), RERUN_PORT)
        };

        let rec = rerun::RecordingStreamBuilder::new("kornia_app")
            .connect_tcp_opts(
                std::net::SocketAddr::from_str(&format!("{}:{}", host, port))
                    .map_err(|e| CuError::new_with_cause("Failed to parse host and port", e))?,
                None,
            )
            .map_err(|e| CuError::new_with_cause("Failed to spawn rerun stream", e))?;

        Ok(Self(rec))
    }

    fn process(&mut self, clock: &RobotClock, input: Self::Input) -> Result<(), CuError> {
        let (img1, img2, img3) = input;
        let timestamp_ns = clock.now().as_nanos();

        if let Some(img) = img1.payload() {
            log_image_rgb8(self, "webcam", timestamp_ns, img)?;
        }

        if let Some(img) = img2.payload() {
            log_image_rgb8(self, "sobel", timestamp_ns, img)?;
        }

        if let Some(img) = img3.payload() {
            log_image_gray8(self, "rtsp", timestamp_ns, img)?;
        }

        Ok(())
    }
}

fn log_image_rgb8(
    rec: &rerun::RecordingStream,
    name: &str,
    timestamp_ns: u64,
    img: &ImageRgb8Msg,
) -> Result<(), CuError> {
    rec.set_time_nanos(name, timestamp_ns as i64);
    rec.log(
        name,
        &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::RGB),
    )
    .map_err(|e| CuError::new_with_cause("Failed to log image", e))?;
    Ok(())
}

fn log_image_gray8(
    rec: &rerun::RecordingStream,
    name: &str,
    timestamp_ns: u64,
    img: &ImageGray8Msg,
) -> Result<(), CuError> {
    rec.set_time_nanos(name, timestamp_ns as i64);
    rec.log(
        name,
        &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::L),
    )
    .map_err(|e| CuError::new_with_cause("Failed to log image", e))?;
    Ok(())
}
