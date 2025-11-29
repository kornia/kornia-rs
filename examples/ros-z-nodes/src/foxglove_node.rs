use foxglove::{
    schemas::{CompressedImage as FoxgloveCompressedImage, Timestamp},
    WebSocketServer,
};
use ros_z::{
    context::ZContext, msg::ProtobufSerdes, node::ZNode, pubsub::ZSub, Builder, Result as ZResult,
};
use std::sync::Arc;
use zenoh::sample::Sample;

use crate::protos::CompressedImage;

pub struct FoxgloveNode {
    #[allow(dead_code)]
    node: ZNode,
    subscriber: ZSub<CompressedImage, Sample, ProtobufSerdes<CompressedImage>>,
}

impl FoxgloveNode {
    pub fn new(ctx: Arc<ZContext>, camera_id: u32) -> ZResult<Self> {
        // create ROS-Z node
        let node = ctx.create_node("foxglove_node").build()?;

        let topic = format!("/camera/{camera_id}/compressed");

        // create subscriber with protobuf serialization
        let subscriber = node
            .create_sub::<CompressedImage>(topic.as_str())
            .with_serdes::<ProtobufSerdes<CompressedImage>>()
            .build()?;

        Ok(Self { node, subscriber })
    }

    pub async fn run(&mut self, shutdown_tx: tokio::sync::watch::Sender<()>) -> ZResult<()> {
        let mut shutdown_rx = shutdown_tx.subscribe();

        log::info!("Foxglove node started, subscribing");

        let server = WebSocketServer::new().start().await?;

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    break;
                }
                Ok(msg) = self.subscriber.async_recv() => {

                    let foxglove_msg = FoxgloveCompressedImage {
                        timestamp: msg.header.as_ref().map(|h| {
                            Timestamp::new(
                                (h.pub_time / 1_000_000_000) as u32,
                                (h.pub_time % 1_000_000_000) as u32,
                            )
                        }),
                        frame_id: msg.header.as_ref()
                            .map(|h| h.frame_id.clone())
                            .unwrap_or_else(|| "camera".to_string()),
                        format: msg.format.clone(),
                        data: msg.data.into(),
                    };

                    foxglove::log!("/camera/0/compressed", foxglove_msg);
                }
            }
        }

        log::info!("Shutting down foxglove node...");

        server.stop().wait().await;

        Ok(())
    }
}
