use dora_image_utils::image_to_arrow;
use dora_node_api::{self, dora_core::config::DataId, DoraNode, Event, IntoArrow};
use kornia::io::stream::{RTSPCameraConfig, V4L2CameraConfig};

fn main() -> eyre::Result<()> {
    // parse env variables
    let source_type =
        std::env::var("SOURCE_TYPE").map_err(|e| eyre::eyre!("SOURCE_TYPE error: {}", e))?;
    let source_uri =
        std::env::var("SOURCE_URI").map_err(|e| eyre::eyre!("SOURCE_URI error: {}", e))?;

    // create the camera source
    let mut camera = match source_type.as_str() {
        "webcam" => {
            let image_cols = std::env::var("IMAGE_COLS")
                .map_err(|e| eyre::eyre!("IMAGE_COLS error: {}", e))?
                .parse::<usize>()?;
            let image_rows = std::env::var("IMAGE_ROWS")
                .map_err(|e| eyre::eyre!("IMAGE_ROWS error: {}", e))?
                .parse::<usize>()?;
            let source_fps = std::env::var("SOURCE_FPS")
                .map_err(|e| eyre::eyre!("SOURCE_FPS error: {}", e))?
                .parse::<u32>()?;
            V4L2CameraConfig::new()
                .with_size([image_cols, image_rows].into())
                .with_fps(source_fps)
                .with_device(&source_uri)
                .build()?
        }
        "rtsp" => RTSPCameraConfig::new().with_url(&source_uri).build()?,
        _ => return Err(eyre::eyre!("Invalid source type: {}", source_type)),
    };

    // start the camera source
    camera.start()?;

    let output = DataId::from("frame".to_owned());

    let (mut node, mut events) = DoraNode::init_from_env()?;

    while let Some(event) = events.recv() {
        match event {
            Event::Input {
                id,
                metadata,
                data: _,
            } => match id.as_str() {
                "tick" => {
                    let Some(frame) = camera.grab()? else {
                        continue;
                    };

                    let (meta_parameters, data) = image_to_arrow(frame, metadata)?;

                    node.send_output(output.clone(), meta_parameters, data.into_arrow())?;
                }
                other => eprintln!("Ignoring unexpected input `{other}`"),
            },
            Event::Stop => {
                camera.close()?;
            }
            other => eprintln!("Received unexpected input: {other:?}"),
        }
    }

    Ok(())
}
