use crate::protos::ImageStats;
use ros_z::{
    context::ZContext, msg::ProtobufSerdes, node::ZNode, pubsub::ZSub, Builder, Result as ZResult,
};
use std::sync::Arc;
use zenoh::sample::Sample;

pub struct LoggerNode {
    #[allow(dead_code)]
    node: ZNode,
    camera_name: String,
    subscriber: ZSub<ImageStats, Sample, ProtobufSerdes<ImageStats>>,
}

impl LoggerNode {
    pub fn new(ctx: Arc<ZContext>, camera_name: &str) -> ZResult<Self> {
        let node = ctx.create_node("logger_node").build()?;

        let topic = format!("camera/{camera_name}/stats");
        let subscriber = node
            .create_sub::<ImageStats>(topic.as_str())
            .with_serdes::<ProtobufSerdes<ImageStats>>()
            .build()?;

        log::info!("Logger node subscribing to '{}'", topic);

        Ok(Self {
            node,
            camera_name: camera_name.to_string(),
            subscriber,
        })
    }

    pub async fn run(self, shutdown_tx: tokio::sync::watch::Sender<()>) -> ZResult<()> {
        let mut shutdown_rx = shutdown_tx.subscribe();

        log::info!("Logger node started for camera '{}'", self.camera_name);

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    break;
                }
                Ok(msg) = self.subscriber.async_recv() => {
                    let mean_avg = msg.mean.iter().sum::<f32>() / msg.mean.len() as f32;
                    let std_avg = msg.std.iter().sum::<f32>() / msg.std.len() as f32;
                    log::info!(
                        "[stats] Frame {}: mean={:.2}, std={:.2}",
                        msg.header.as_ref().unwrap().sequence,
                        mean_avg,
                        std_avg
                    );
                }
            }
        }

        log::info!("Shutting down logger node...");

        Ok(())
    }
}
