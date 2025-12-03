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
    camera_id: u32,
}

impl FoxgloveNode {
    pub fn new(ctx: Arc<ZContext>, camera_id: u32) -> ZResult<Self> {
        // create ROS-Z node
        let node = ctx.create_node("foxglove_node").build()?;

        // create subscriber with protobuf serialization
        let subscriber = node
            .create_sub::<CompressedImage>(format!("/camera/{camera_id}/compressed").as_str())
            .with_serdes::<ProtobufSerdes<CompressedImage>>()
            .build()?;

        Ok(Self {
            node,
            subscriber,
            camera_id,
        })
    }

    pub async fn run(self, shutdown_tx: tokio::sync::watch::Sender<()>) -> ZResult<()> {
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

                    match self.camera_id {
                        0 => foxglove::log!("/camera/0/compressed", foxglove_msg),
                        1 => foxglove::log!("/camera/1/compressed", foxglove_msg),
                        e => panic!("Invalid camera id: {}", e),
                    }
                }
            }
        }

        log::info!("Shutting down foxglove node...");

        server.stop().wait().await;

        Ok(())
    }
}
