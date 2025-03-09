use dora_node_api::{self, dora_core::config::DataId, DoraNode, Event, IntoArrow, Parameter};
use kornia::io::stream::{RTSPCameraConfig, V4L2CameraConfig};

fn main() -> eyre::Result<()> {
    // parse env variables
    let source_type = std::env::var("SOURCE_TYPE")?;

    // create the camera source
    let mut camera = match source_type.as_str() {
        "webcam" => {
            let image_cols = std::env::var("IMAGE_COLS")?.parse::<usize>()?;
            let image_rows = std::env::var("IMAGE_ROWS")?.parse::<usize>()?;
            let source_fps = std::env::var("SOURCE_FPS")?.parse::<u32>()?;
            V4L2CameraConfig::new()
                .with_size([image_cols, image_rows].into())
                .with_fps(source_fps)
                .build()?
        }
        "rtsp" => {
            let source_uri = std::env::var("SOURCE_URI")?;
            RTSPCameraConfig::new().with_url(&source_uri).build()?
        }
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

                    let mut meta_parameters = metadata.parameters;
                    meta_parameters
                        .insert("cols".to_string(), Parameter::Integer(frame.cols() as i64));
                    meta_parameters
                        .insert("rows".to_string(), Parameter::Integer(frame.rows() as i64));

                    node.send_output(
                        output.clone(),
                        meta_parameters,
                        // TODO: avoid to_vec copy
                        frame.as_slice().to_vec().into_arrow(),
                    )?;
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
