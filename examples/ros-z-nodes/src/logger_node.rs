use ros_z::{
    context::ZContext, msg::ProtobufSerdes, node::ZNode, pubsub::ZSub, Builder, Result as ZResult,
};
use std::sync::Arc;
use zenoh::sample::Sample;

use crate::protos::CompressedImage;

pub struct LoggerNode {
    #[allow(dead_code)]
    node: ZNode,
    subscriber: ZSub<CompressedImage, Sample, ProtobufSerdes<CompressedImage>>,
}

impl LoggerNode {
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

    pub async fn run(self, shutdown_tx: tokio::sync::watch::Sender<()>) -> ZResult<()> {
        let mut shutdown_rx = shutdown_tx.subscribe();

        log::info!("Logger node started, subscribing");

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    break;
                }
                Ok(msg) = self.subscriber.async_recv() => {
                    log::info!("Received message: {:?}", msg);
                }
            }
        }

        log::info!("Shutting down logger node...");

        Ok(())
    }
}
