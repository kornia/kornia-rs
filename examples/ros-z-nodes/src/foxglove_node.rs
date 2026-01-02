use crate::protos::{CompressedImage, ImageStats};
use foxglove::{
    schemas::{CompressedImage as FoxgloveCompressedImage, Timestamp},
    Channel, WebSocketServer,
};
use ros_z::{
    context::ZContext, msg::ProtobufSerdes, node::ZNode, pubsub::ZSub, Builder, Result as ZResult,
};
use std::sync::Arc;
use zenoh::sample::Sample;

pub struct FoxgloveNode {
    #[allow(dead_code)]
    node: ZNode,
    camera_subscriber: ZSub<CompressedImage, Sample, ProtobufSerdes<CompressedImage>>,
    stats_subscriber: ZSub<ImageStats, Sample, ProtobufSerdes<ImageStats>>,
    camera_channel: Channel<FoxgloveCompressedImage>,
    stats_channel: Channel<ImageStats>,
}

impl FoxgloveNode {
    pub fn new(ctx: Arc<ZContext>, camera_name: &str) -> ZResult<Self> {
        let node = ctx.create_node("foxglove_node").build()?;

        let camera_topic = format!("camera/{camera_name}/compressed");
        let camera_subscriber = node
            .create_sub::<CompressedImage>(camera_topic.as_str())
            .with_serdes::<ProtobufSerdes<CompressedImage>>()
            .build()?;

        let stats_topic = format!("camera/{camera_name}/stats");
        let stats_subscriber = node
            .create_sub::<ImageStats>(stats_topic.as_str())
            .with_serdes::<ProtobufSerdes<ImageStats>>()
            .build()?;

        let camera_channel = Channel::<FoxgloveCompressedImage>::new(camera_topic.as_str());
        let stats_channel = Channel::<ImageStats>::new(stats_topic.as_str());

        log::info!(
            "Foxglove node subscribing to '{}' and '{}'",
            camera_topic,
            stats_topic
        );

        Ok(Self {
            node,
            camera_subscriber,
            stats_subscriber,
            camera_channel,
            stats_channel,
        })
    }

    pub async fn run(self, shutdown_tx: tokio::sync::watch::Sender<()>) -> ZResult<()> {
        let mut shutdown_rx = shutdown_tx.subscribe();

        let server = WebSocketServer::new().start().await?;

        loop {
            tokio::select! {
                biased;

                _ = shutdown_rx.changed() => {
                    break;
                }
                Ok(msg) = self.camera_subscriber.async_recv() => {
                    // convert to foxglove message to leverage the schema and visualization tools
                    let foxglove_msg = FoxgloveCompressedImage {
                        timestamp: msg.header.as_ref().map(|h| {
                            Timestamp::new(
                                (h.pub_time / 1_000_000_000) as u32,
                                (h.pub_time % 1_000_000_000) as u32,
                            )
                        }),
                        frame_id: msg.header.as_ref().unwrap().frame_id.clone(),
                        format: msg.format.clone(),
                        data: msg.data.into(),
                    };
                    self.camera_channel.log(&foxglove_msg);
                }
                Ok(msg) = self.stats_subscriber.async_recv() => {
                    // Publish the ImageStats proto directly with its schema
                    self.stats_channel.log(&msg);
                }
            }
        }

        log::info!("Shutting down foxglove node...");

        server.stop().wait().await;

        Ok(())
    }
}
