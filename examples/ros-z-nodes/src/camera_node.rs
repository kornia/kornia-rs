use crate::protos::{CompressedImage, Header};
use kornia_image::ImageSize;
use kornia_io::v4l::{PixelFormat, V4LCameraConfig, V4lVideoCapture};
use nix::sys::time::TimeValLike;
use ros_z::{
    context::ZContext, msg::ProtobufSerdes, node::ZNode, pubsub::ZPub, Builder, Result as ZResult,
};
use std::sync::Arc;

/// ROS2-style camera publisher node
pub struct V4lCameraNode {
    #[allow(dead_code)]
    node: ZNode,
    publisher: ZPub<CompressedImage, ProtobufSerdes<CompressedImage>>,
    handle: Option<std::thread::JoinHandle<()>>,
    camera_id: u32,
    fps: u32,
}

impl V4lCameraNode {
    /// Create a new camera publisher node
    pub fn new(ctx: Arc<ZContext>, camera_id: u32, fps: u32) -> ZResult<Self> {
        // create ROS-Z node
        let node = ctx.create_node("camera_node").build()?;

        // create publisher with protobuf serialization
        let publisher = node
            .create_pub::<CompressedImage>(format!("/camera/{camera_id}/compressed").as_str())
            .with_serdes::<ProtobufSerdes<CompressedImage>>()
            .build()?;

        Ok(Self {
            node,
            publisher,
            handle: None,
            camera_id,
            fps,
        })
    }

    /// Run the main publishing loop until the cancellation token is set
    pub async fn run(mut self, shutdown_tx: tokio::sync::watch::Sender<()>) -> ZResult<()> {
        let mut shutdown_rx = shutdown_tx.subscribe();

        // initialize camera
        let camera = V4lVideoCapture::new(V4LCameraConfig {
            device_path: format!("/dev/video{}", self.camera_id),
            size: ImageSize {
                width: 640,
                height: 480,
            },
            fps: self.fps,
            format: PixelFormat::MJPG,
            buffer_size: 4,
        })?;

        let (frame_tx, frame_rx) = flume::unbounded();

        self.handle = Some(start_camera_thread(camera, frame_tx, shutdown_rx.clone()));

        log::info!("Camera node started, publishing");

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    break;
                }
                Ok(frame) = frame_rx.recv_async() => {
                    if let Err(e) = self.publisher.async_publish(&frame).await {
                        log::error!("Error publishing frame: {}", e);
                    }
                }
            }
        }

        log::info!("Shutting down camera node...");

        // Wait for camera thread to complete before returning
        if let Some(handle) = self.handle.take() {
            if let Err(e) = handle.join() {
                log::error!("Error joining camera thread during shutdown: {:?}", e);
            }
        }

        Ok(())
    }
}

impl Drop for V4lCameraNode {
    fn drop(&mut self) {
        // Safety net: if node was dropped without calling run() or run() didn't complete cleanup
        if let Some(handle) = self.handle.take() {
            log::warn!("Camera node dropped with active thread, joining...");
            if let Err(e) = handle.join() {
                log::error!("Error joining camera thread in drop: {:?}", e);
            }
        }
    }
}

/// Convert seconds and nanoseconds to a single nanosecond timestamp
fn stamp_from_sec_nanos(sec: u64, nanos: u32) -> u64 {
    sec * 1_000_000_000 + (nanos as u64)
}

/// Get current publication timestamp in nanoseconds
fn get_pub_time() -> Option<u64> {
    nix::time::clock_gettime(nix::time::ClockId::CLOCK_REALTIME)
        .ok()
        .map(|time| time.num_nanoseconds() as u64)
}

/// Convert a camera frame to a CompressedImage message
fn frame_to_compressed_image(frame: kornia_io::v4l::EncodedFrame) -> Option<CompressedImage> {
    let pub_time = get_pub_time()?;

    Some(CompressedImage {
        header: Some(Header {
            acq_time: stamp_from_sec_nanos(frame.timestamp.sec as u64, frame.timestamp.usec as u32),
            pub_time,
            sequence: frame.sequence,
            frame_id: "camera".to_string(),
        }),
        format: "jpeg".to_string(),
        data: frame.buffer.into_vec(),
    })
}

/// Start the camera capture thread
///
/// This thread runs synchronously, capturing frames from the camera and sending them
/// through a channel to the async publishing loop. The thread checks for shutdown
/// signals and exits cleanly when the channel is closed or shutdown is requested.
fn start_camera_thread(
    mut camera: V4lVideoCapture,
    frame_tx: flume::Sender<CompressedImage>,
    shutdown_rx: tokio::sync::watch::Receiver<()>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        loop {
            // Check shutdown at start of each iteration
            if shutdown_rx.has_changed().unwrap_or(false) {
                break;
            }

            // Grab frame, skip if failed
            let Ok(Some(frame)) = camera.grab_frame() else {
                continue;
            };

            // Convert frame to message
            let Some(msg) = frame_to_compressed_image(frame) else {
                log::warn!("Failed to get timestamp, skipping frame");
                continue;
            };

            // Send frame, exit if channel closed
            if let Err(e) = frame_tx.send(msg) {
                log::error!("Error sending frame to channel: {}", e);
            }
        }
    })
}
