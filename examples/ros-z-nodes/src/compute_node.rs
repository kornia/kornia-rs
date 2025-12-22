//! Compute Node - subscribes to raw images via SHM, computes statistics, publishes stats

use crate::protos::{Header, ImageStats, RawImage};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use prost::Message;
use ros_z::{
    context::ZContext, msg::ProtobufSerdes, node::ZNode, pubsub::ZPub, Builder, Result as ZResult,
};
use std::sync::Arc;
use zenoh::Wait;

fn get_pub_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

pub struct ComputeNode {
    #[allow(dead_code)]
    node: ZNode,
    camera_name: String,
    stats_publisher: ZPub<ImageStats, ProtobufSerdes<ImageStats>>,
}

impl ComputeNode {
    pub fn new(ctx: Arc<ZContext>, camera_name: &str) -> ZResult<Self> {
        let node = ctx.create_node("compute_node").build()?;

        // ros-z publisher for stats
        let stats_topic = format!("camera/{camera_name}/stats");
        let stats_publisher = node
            .create_pub::<ImageStats>(stats_topic.as_str())
            .with_serdes::<ProtobufSerdes<ImageStats>>()
            .build()?;

        log::info!("Compute node publishing to '{}'", stats_topic);

        Ok(Self {
            node,
            camera_name: camera_name.to_string(),
            stats_publisher,
        })
    }

    pub async fn run(self, shutdown_tx: tokio::sync::watch::Sender<()>) -> ZResult<()> {
        let mut shutdown_rx = shutdown_tx.subscribe();

        // Create Zenoh session with SHM enabled for receiving raw images
        let mut zenoh_config = zenoh::Config::default();
        zenoh_config.insert_json5("transport/shared_memory/enabled", "true")?;

        let session = zenoh::open(zenoh_config).wait()?;

        // Subscribe to raw images via Zenoh SHM
        let camera_topic = format!("camera/{}/raw_shm", self.camera_name);
        let subscriber = session.declare_subscriber(&camera_topic).wait()?;

        log::info!(
            "Compute node started for camera '{}', subscribing to '{}' (Zenoh SHM)",
            self.camera_name,
            camera_topic
        );

        loop {
            tokio::select! {
                biased;

                _ = shutdown_rx.changed() => break,

                Ok(sample) = subscriber.recv_async() => {
                    // Get payload bytes (works for both SHM and regular data)
                    let bytes = sample.payload().to_bytes();

                    // Decode protobuf
                    let msg = RawImage::decode(bytes.as_ref())?;

                    // convert to image kornia
                    let img = Image::<u8, 3, CpuAllocator>::new(ImageSize {
                        width: msg.width as usize,
                        height: msg.height as usize,
                    }, msg.data, CpuAllocator)?;

                    // compute mean and std
                    let (std, mean) = kornia_imgproc::core::std_mean(&img);

                    let stats = ImageStats {
                        header: msg.header.map(|h| Header {
                            acq_time: h.acq_time,
                            pub_time: get_pub_time(),
                            sequence: h.sequence,
                            frame_id: h.frame_id,
                        }),
                        mean: mean.into_iter().map(|x| x as f32).collect(),
                        std: std.into_iter().map(|x| x as f32).collect(),
                    };

                    if let Err(e) = self.stats_publisher.async_publish(&stats).await {
                        log::error!("Failed to publish stats: {}", e);
                    }
                }
            }
        }

        log::info!("Compute node stopped");
        Ok(())
    }
}
