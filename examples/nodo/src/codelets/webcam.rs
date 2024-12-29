use eyre::eyre;
use kornia::{
    image::{ops, Image, ImageSize},
    imgproc,
    io::{
        fps_counter::FpsCounter,
        stream::{CameraCapture, V4L2CameraConfig},
    },
};
use nodo::prelude::*;

type ImageRGB8 = Image<u8, 3>;

#[derive(Default)]
pub struct Webcam {
    capture: Option<CameraCapture>,
}

impl Webcam {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let capture = V4L2CameraConfig::new()
            .with_camera_id(0)
            .with_fps(30)
            .with_size(ImageSize {
                width: 640,
                height: 480,
            })
            .build()?;
        Ok(Self {
            capture: Some(capture),
        })
    }
}

impl Codelet for Webcam {
    type Status = DefaultStatus;
    type Config = ();
    type Rx = ();
    type Tx = DoubleBufferTx<Message<ImageRGB8>>;

    fn build_bundles(_cfg: &Self::Config) -> (Self::Rx, Self::Tx) {
        ((), DoubleBufferTx::new(1))
    }

    fn start(&mut self, _ctx: &Context<Self>, _rx: &mut Self::Rx, _tx: &mut Self::Tx) -> Outcome {
        if let Some(capture) = self.capture.as_mut() {
            capture.start()?;
        }
        SUCCESS
    }

    fn step(&mut self, ctx: &Context<Self>, _rx: &mut Self::Rx, tx: &mut Self::Tx) -> Outcome {
        let Ok(Some(image)) = self
            .capture
            .as_mut()
            .ok_or(eyre!("Camera not initialized"))?
            .grab()
        else {
            return SKIPPED;
        };

        println!("image: {:?}", image.size());
        let now = ctx.clocks.app_mono.now();
        let now_acqtime = ctx.clocks.sys_mono.now();

        tx.push(Message {
            seq: 0,
            stamp: Stamp {
                acqtime: now_acqtime,
                pubtime: now,
            },
            value: image,
        })?;

        SUCCESS
    }

    fn stop(&mut self, _ctx: &Context<Self>, _rx: &mut Self::Rx, _tx: &mut Self::Tx) -> Outcome {
        if let Some(capture) = self.capture.as_mut() {
            capture.close()?;
        }
        SUCCESS
    }
}
