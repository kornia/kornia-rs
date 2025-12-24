use crate::protos::{camera::v1::CompressedImage, header::v1::Header};
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
    camera_name: String,
    device_id: u32,
    fps: u32,
}

impl V4lCameraNode {
    /// Create a new camera publisher node
    ///
    /// # Arguments
    /// * `ctx` - ROS-Z context
    /// * `camera_name` - Name for topics (e.g., "front", "back")
    /// * `device_id` - V4L device ID (e.g., 0 for /dev/video0)
    /// * `fps` - Frames per second
    pub fn new(ctx: Arc<ZContext>, camera_name: &str, device_id: u32, fps: u32) -> ZResult<Self> {
        let node = ctx.create_node("camera_node").build()?;

        let topic = format!("camera/{camera_name}/compressed");
        let publisher = node
            .create_pub::<CompressedImage>(topic.as_str())
            .with_serdes::<ProtobufSerdes<CompressedImage>>()
            .build()?;

        log::info!("Camera '{}' publishing to '{}'", camera_name, topic);

        Ok(Self {
            node,
            publisher,
            handle: None,
            camera_name: camera_name.to_string(),
            device_id,
            fps,
        })
    }

    /// Run the main publishing loop until the cancellation token is set
    pub async fn run(mut self, shutdown_tx: tokio::sync::watch::Sender<()>) -> ZResult<()> {
        let mut shutdown_rx = shutdown_tx.subscribe();

        let camera = V4lVideoCapture::new(V4LCameraConfig {
            device_path: format!("/dev/video{}", self.device_id),
            size: ImageSize {
                width: 640,
                height: 480,
            },
            fps: self.fps,
            format: PixelFormat::MJPG,
            buffer_size: 4,
        })?;

        let (frame_tx, frame_rx) = flume::unbounded();
        let camera_name = self.camera_name.clone();

        self.handle = Some(start_camera_thread(
            camera,
            frame_tx,
            shutdown_rx.clone(),
            camera_name.clone(),
        ));

        log::info!("Camera '{}' started, publishing", self.camera_name);

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

        log::info!("Shutting down camera '{}'...", self.camera_name);

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
        if let Some(handle) = self.handle.take() {
            log::warn!("Camera node dropped with active thread, joining...");
            if let Err(e) = handle.join() {
                log::error!("Error joining camera thread in drop: {:?}", e);
            }
        }
    }
}

fn stamp_from_sec_nanos(sec: u64, nanos: u32) -> u64 {
    sec * 1_000_000_000 + (nanos as u64)
}

fn get_pub_time() -> Option<u64> {
    nix::time::clock_gettime(nix::time::ClockId::CLOCK_REALTIME)
        .ok()
        .map(|time| time.num_nanoseconds() as u64)
}

fn frame_to_compressed_image(
    frame: kornia_io::v4l::EncodedFrame,
    camera_name: &str,
) -> Option<CompressedImage> {
    let pub_time = get_pub_time()?;

    Some(CompressedImage {
        header: Some(Header {
            acq_time: stamp_from_sec_nanos(frame.timestamp.sec as u64, frame.timestamp.usec as u32),
            pub_time,
            sequence: frame.sequence,
            frame_id: camera_name.to_string(),
        }),
        format: "jpeg".to_string(),
        data: frame.buffer.into_vec(),
    })
}

fn start_camera_thread(
    mut camera: V4lVideoCapture,
    frame_tx: flume::Sender<CompressedImage>,
    shutdown_rx: tokio::sync::watch::Receiver<()>,
    camera_name: String,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || loop {
        if shutdown_rx.has_changed().unwrap_or(false) {
            break;
        }

        let Ok(Some(frame)) = camera.grab_frame() else {
            continue;
        };

        let Some(msg) = frame_to_compressed_image(frame, &camera_name) else {
            log::warn!("Failed to get timestamp, skipping frame");
            continue;
        };

        if let Err(e) = frame_tx.send(msg) {
            log::error!("Error sending frame to channel: {}", e);
        }
    })
}
