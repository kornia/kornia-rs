use dora_node_api::{self, dora_core::config::DataId, DoraNode, Event, Parameter};
use kornia_io::gstreamer::{RTSPCameraConfig, V4L2CameraConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // parse env variables
    let source_type =
        std::env::var("SOURCE_TYPE").map_err(|e| format!("SOURCE_TYPE error: {e}"))?;
    let source_uri = std::env::var("SOURCE_URI").map_err(|e| format!("SOURCE_URI error: {e}"))?;

    // create the camera source
    let mut camera = match source_type.as_str() {
        "webcam" => {
            let image_cols = std::env::var("IMAGE_COLS")
                .map_err(|e| format!("IMAGE_COLS error: {e}"))?
                .parse::<usize>()?;
            let image_rows = std::env::var("IMAGE_ROWS")
                .map_err(|e| format!("IMAGE_ROWS error: {e}"))?
                .parse::<usize>()?;
            let source_fps = std::env::var("SOURCE_FPS")
                .map_err(|e| format!("SOURCE_FPS error: {e}"))?
                .parse::<u32>()?;
            V4L2CameraConfig::new()
                .with_size([image_cols, image_rows].into())
                .with_fps(source_fps)
                .with_device(&source_uri)
                .build()?
        }
        "rtsp" => RTSPCameraConfig::new().with_url(&source_uri).build()?,
        _ => return Err(format!("Invalid source type: {source_type}").into()),
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
                    let Some(frame) = camera.grab_rgb8()? else {
                        continue;
                    };

                    let mut params = metadata.parameters;
                    params.insert("encoding".to_owned(), Parameter::String("RGB8".to_string()));
                    params.insert(
                        "height".to_owned(),
                        Parameter::Integer(frame.size().height as i64),
                    );
                    params.insert(
                        "width".to_owned(),
                        Parameter::Integer(frame.size().width as i64),
                    );

                    // send the frame to the output
                    node.send_output_bytes(
                        output.clone(),
                        params,
                        frame.numel(),
                        frame.as_slice(),
                    )?;
                }
                other => eprintln!("Ignoring unexpected input `{other}`"),
            },
            Event::Stop(_) => {
                camera.close()?;
            }
            other => eprintln!("Received unexpected input: {other:?}"),
        }
    }

    Ok(())
}
