use crate::protos::{CompressedImage, Header};
use kornia_image::ImageSize;
use kornia_io::v4l::{PixelFormat, V4LCameraConfig, V4lVideoCapture};
use nix::sys::time::TimeValLike;
use ros_z::{
    context::ZContext, msg::ProtobufSerdes, node::ZNode, pubsub::ZPub, Builder, Result as ZResult,
};
use std::sync::Arc;

/// ROS2-style camera publisher node
/// Encapsulates the node, publisher, and camera in a class-like structure
pub struct CameraNode {
    #[allow(dead_code)]
    node: ZNode,
    publisher: ZPub<CompressedImage, ProtobufSerdes<CompressedImage>>,
    camera: V4lVideoCapture,
    frame_id: String,
    pixel_format: String,
}

impl CameraNode {
    /// Create a new camera publisher node
    pub fn new(ctx: Arc<ZContext>, camera_id: u32, fps: u32) -> ZResult<Self> {
        // create ROS-Z node
        let node = ctx.create_node("camera_node").build()?;

        // create publisher with protobuf serialization
        let publisher = node
            .create_pub::<CompressedImage>(format!("/camera/{camera_id}/compressed").as_str())
            .with_serdes::<ProtobufSerdes<CompressedImage>>()
            .build()?;

        // initialize camera
        let pixel_format = PixelFormat::MJPG;
        let camera = V4lVideoCapture::new(V4LCameraConfig {
            device_path: format!("/dev/video{}", camera_id),
            size: ImageSize {
                width: 640,
                height: 480,
            },
            fps,
            format: pixel_format,
            buffer_size: 4,
        })?;

        log::info!("Camera initialized: /dev/video{}", camera_id);
        log::info!("Image size: 640x480, FPS: {}", fps);

        Ok(Self {
            node,
            publisher,
            camera,
            frame_id: format!("camera_{}", camera_id),
            pixel_format: pixel_format.as_str().to_lowercase(),
        })
    }

    /// Run the main publishing loop until the cancellation token is set
    pub async fn run(&mut self, shutdown_tx: tokio::sync::watch::Sender<()>) -> ZResult<()> {
        let mut shutdown_rx = shutdown_tx.subscribe();

        log::info!("Camera node started, publishing");

        let mut interval = tokio::time::interval(std::time::Duration::from_millis(33));

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    break;
                }
                _ = interval.tick() => {
                    if let Some(frame) = self.camera.grab_frame()? {
                        let pub_time = nix::time::clock_gettime(nix::time::ClockId::CLOCK_REALTIME)?.num_nanoseconds() as u64;

                        let msg = CompressedImage {
                            header: Some(Header {
                                acq_time: stamp_from_sec_nanos(
                                    frame.timestamp.sec as u64,
                                    frame.timestamp.usec as u32,
                                ),
                                pub_time,
                                sequence: frame.sequence,
                                frame_id: self.frame_id.clone(),
                            }),
                            format: self.pixel_format.clone(),
                            data: frame.buffer.into_vec(),
                        };

                        self.publisher.publish(&msg)?;
                    }
                }
            }
        }

        log::info!("Shutting down camera node...");

        Ok(())
    }
}

/// Convert seconds and nanoseconds to a single nanosecond timestamp
fn stamp_from_sec_nanos(sec: u64, nanos: u32) -> u64 {
    sec * 1_000_000_000 + (nanos as u64)
}
