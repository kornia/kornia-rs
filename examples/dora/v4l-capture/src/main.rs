use std::ops::Deref;
use std::sync::Arc;

use dora_image_utils::image_to_arrow;
//use dora_node_api::{self, dora_core::config::DataId, DoraNode, Event, IntoArrow, Parameter};
use dora_node_api::{self, dora_core::config::DataId, DoraNode, Event, Parameter};
use kornia::{
    image::ImageSize,
    io::v4l::{PixelFormat, V4LCameraConfig, V4LVideoCapture},
};

fn main() -> eyre::Result<()> {
    // parse env variables
    let source_device =
        std::env::var("SOURCE_DEVICE").map_err(|e| eyre::eyre!("SOURCE_DEVICE error: {}", e))?;
    let image_cols = std::env::var("IMAGE_COLS")
        .map_err(|e| eyre::eyre!("IMAGE_COLS error: {}", e))?
        .parse::<usize>()?;
    let image_rows = std::env::var("IMAGE_ROWS")
        .map_err(|e| eyre::eyre!("IMAGE_ROWS error: {}", e))?
        .parse::<usize>()?;
    let source_fps = std::env::var("SOURCE_FPS")
        .map_err(|e| eyre::eyre!("SOURCE_FPS error: {}", e))?
        .parse::<u32>()?;
    let pixel_format = std::env::var("PIXEL_FORMAT")
        .map_err(|e| eyre::eyre!("PIXEL_FORMAT error: {}", e))?
        .parse::<String>()?;

    let pixel_format = match pixel_format.as_str() {
        "YUYV" => PixelFormat::YUYV,
        "MJPG" => PixelFormat::MJPG,
        _ => return Err(eyre::eyre!("Invalid pixel format: {}", pixel_format)),
    };

    // create the camera source
    let mut camera = V4LVideoCapture::new(V4LCameraConfig {
        device_path: source_device,
        size: ImageSize {
            width: image_cols,
            height: image_rows,
        },
        fps: source_fps,
        format: pixel_format,
        buffer_size: 4,
    })?;

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

                    // let (meta_parameters, data) = image_to_arrow(frame, metadata)?;
                    let mut meta_parameters = metadata.parameters;
                    meta_parameters.insert(
                        "cols".to_string(),
                        Parameter::Integer(frame.size.width as i64),
                    );
                    meta_parameters.insert(
                        "rows".to_string(),
                        Parameter::Integer(frame.size.height as i64),
                    );

                    // Get data directly from buffer to avoid copying
                    let data = match std::sync::Arc::try_unwrap(frame.buffer.0) {
                        Ok(data) => data,
                        Err(arc_data) => arc_data.as_slice().to_vec(), // Only copy if necessary
                    };
                    node.send_output(output.clone(), meta_parameters, data.into_arrow())?;
                }
                other => eprintln!("Ignoring unexpected input `{other}`"),
            },
            Event::Stop => {
                // camera.close()?;
            }
            other => eprintln!("Received unexpected input: {other:?}"),
        }
    }

    Ok(())
}
