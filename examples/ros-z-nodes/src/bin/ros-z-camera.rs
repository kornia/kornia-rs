use argh::FromArgs;
use ros_z::{context::ZContextBuilder, Builder, Result as ZResult};
use std::sync::Arc;

use ros_z_nodes::{
    camera_node::V4lCameraNode, foxglove_node::FoxgloveNode, logger_node::LoggerNode,
};

#[derive(FromArgs)]
/// ROS2-style camera publisher using ros-z
struct Args {
    /// the camera id to use
    #[argh(option, short = 'c', default = "0")]
    camera_id: u32,

    /// the frames per second to record
    #[argh(option, short = 'f', default = "30")]
    fps: u32,
}

#[cfg(target_os = "linux")]
#[tokio::main]
async fn main() -> ZResult<()> {
    let env = env_logger::Env::default().default_filter_or("info");
    env_logger::init_from_env(env);

    let args: Args = argh::from_env();

    // create the cancellation token
    let shutdown_tx = tokio::sync::watch::Sender::new(());

    ctrlc::set_handler({
        let shutdown_tx = shutdown_tx.clone();
        move || {
            log::info!("Received Ctrl+C, shutting down gracefully...");
            shutdown_tx.send(()).ok();
        }
    })?;

    // initialize ROS-Z context
    let ctx = Arc::new(ZContextBuilder::default().build()?);

    // create and initialize the camera publisher node
    let camera_node = V4lCameraNode::new(ctx.clone(), args.camera_id, args.fps)?;
    let foxglove_node = FoxgloveNode::new(ctx.clone(), args.camera_id)?;
    let logger_node = LoggerNode::new(ctx.clone(), args.camera_id)?;

    // run the nodes
    let nodes = vec![
        tokio::spawn(camera_node.run(shutdown_tx.clone())),
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
