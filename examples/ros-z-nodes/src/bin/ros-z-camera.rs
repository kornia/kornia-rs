use argh::FromArgs;
use ros_z::{context::ZContextBuilder, Builder, Result as ZResult};
use std::sync::Arc;

use ros_z_nodes::{
    camera_node::V4lCameraNode, compute_node::ComputeNode, decoder_node::DecoderNode,
    foxglove_node::FoxgloveNode, logger_node::LoggerNode,
};

#[derive(FromArgs)]
/// ROS2-style camera publisher using ros-z
struct Args {
    /// camera name for topics (e.g., "front", "back")
    #[argh(option, short = 'n', default = "String::from(\"front\")")]
    camera_name: String,

    /// V4L device ID (e.g., 0 for /dev/video0)
    #[argh(option, short = 'd', default = "0")]
    device_id: u32,

    /// frames per second
    #[argh(option, short = 'f', default = "30")]
    fps: u32,
}

#[cfg(target_os = "linux")]
#[tokio::main]
async fn main() -> ZResult<()> {
    let env = env_logger::Env::default().default_filter_or("info");
    env_logger::init_from_env(env);

    let args: Args = argh::from_env();

    log::info!(
        "Starting camera '{}' (device: /dev/video{}, fps: {})",
        args.camera_name,
        args.device_id,
        args.fps
    );

    let shutdown_tx = tokio::sync::watch::Sender::new(());

    ctrlc::set_handler({
        let shutdown_tx = shutdown_tx.clone();
        move || {
            log::info!("Received Ctrl+C, shutting down gracefully...");
            shutdown_tx.send(()).ok();
        }
    })?;

    let ctx = Arc::new(ZContextBuilder::default().build()?);

    let camera_node = V4lCameraNode::new(ctx.clone(), &args.camera_name, args.device_id, args.fps)?;
    let compute_node = ComputeNode::new(ctx.clone(), &args.camera_name)?;
    let decoder_node = DecoderNode::new(ctx.clone(), &args.camera_name)?;
    let foxglove_node = FoxgloveNode::new(ctx.clone(), &args.camera_name)?;
    let logger_node = LoggerNode::new(ctx.clone(), &args.camera_name)?;

    let nodes = vec![
        tokio::spawn(camera_node.run(shutdown_tx.clone())),
        tokio::spawn(compute_node.run(shutdown_tx.clone())),
        tokio::spawn(decoder_node.run(shutdown_tx.clone())),
        tokio::spawn(logger_node.run(shutdown_tx.clone())),
        tokio::spawn(foxglove_node.run(shutdown_tx.clone())),
    ];

    for node in nodes {
        node.await??;
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    let env = env_logger::Env::default().default_filter_or("info");
    env_logger::init_from_env(env);

    log::error!("This example is only supported on Linux due to V4L dependency.");
    std::process::exit(1);
}
