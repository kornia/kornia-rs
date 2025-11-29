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
    frame_rx: flume::Receiver<CompressedImage>,
    handle: Option<std::thread::JoinHandle<()>>,
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
        let mut camera = V4lVideoCapture::new(V4LCameraConfig {
            device_path: format!("/dev/video{}", camera_id),
            size: ImageSize {
                width: 640,
                height: 480,
            },
            fps,
            format: pixel_format,
            buffer_size: 4,
        })?;

        let (frame_tx, frame_rx) = flume::unbounded();

        let handle = std::thread::spawn(move || {
            while let Ok(Some(frame)) = camera.grab_frame() {
                let pub_time = nix::time::clock_gettime(nix::time::ClockId::CLOCK_REALTIME)
                    .expect("Failed to get clock time")
                    .num_nanoseconds() as u64;
                let msg = CompressedImage {
                    header: Some(Header {
                        acq_time: stamp_from_sec_nanos(
                            frame.timestamp.sec as u64,
                            frame.timestamp.usec as u32,
                        ),
                        pub_time,
                        sequence: frame.sequence,
                        frame_id: "camera".to_string(),
                    }),
                    format: "mjpg".to_string(),
                    data: frame.buffer.into_vec(),
                };
                if let Err(e) = frame_tx.send(msg) {
                    log::error!("Error sending frame to channel: {}", e);
                }
            }
        });

        log::info!("Camera initialized: /dev/video{}", camera_id);
        log::info!("Image size: 640x480, FPS: {}", fps);

        Ok(Self {
            node,
            publisher,
            frame_rx,
            handle: Some(handle),
        })
    }

    /// Run the main publishing loop until the cancellation token is set
    pub async fn run(&mut self, shutdown_tx: tokio::sync::watch::Sender<()>) -> ZResult<()> {
        let mut shutdown_rx = shutdown_tx.subscribe();

        log::info!("Camera node started, publishing");

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    break;
                }
                Ok(frame) = self.frame_rx.recv_async() => {
                    if let Err(e) = self.publisher.async_publish(&frame).await {
                        log::error!("Error publishing frame: {}", e);
                    }
                }
            }
        }

        log::info!("Shutting down camera node...");

        Ok(())
    }
}

impl Drop for CameraNode {
    fn drop(&mut self) {
        log::info!("Shutting down camera node...");
        if let Some(handle) = self.handle.take() {
            if let Err(e) = handle.join() {
                log::error!("Error joining camera thread: {:?}", e);
            }
        }
    }
}

/// Convert seconds and nanoseconds to a single nanosecond timestamp
fn stamp_from_sec_nanos(sec: u64, nanos: u32) -> u64 {
    sec * 1_000_000_000 + (nanos as u64)
}
