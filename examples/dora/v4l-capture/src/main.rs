use dora_node_api::{self, dora_core::config::DataId, DoraNode, Event};
use kornia::{
    image::{arrow::IntoArrow, Image, ImageSize},
    imgproc::color::{convert_yuyv_to_rgb_u8, YuvToRgbMode},
    io::jpeg,
    io::v4l::{PixelFormat, V4LCameraConfig, V4LVideoCapture},
    tensor::CpuAllocator,
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

                    // allocate the buffer
                    let mut img_rgb8 =
                        Image::<u8, 3, CpuAllocator>::from_size_val(frame.size, 0, CpuAllocator)?;

                    // decode the frame to rgb8
                    match pixel_format {
                        PixelFormat::YUYV => {
                            convert_yuyv_to_rgb_u8(
                                &frame.buffer,
                                &mut img_rgb8,
                                YuvToRgbMode::Bt601Full,
                            )?;
                        }
                        PixelFormat::MJPG => {
                            jpeg::decode_image_jpeg_rgb8(&frame.buffer, &mut img_rgb8)?;
                        }
                        _ => {
                            return Err(eyre::eyre!("Unsupported pixel format: {}", pixel_format));
                        }
                    }

                    node.send_output(output.clone(), metadata.parameters, img_rgb8.into_arrow())?;
                }
                other => eprintln!("Ignoring unexpected input `{other}`"),
            },
            _ => eprintln!("Received unexpected input: {event:?}"),
        }
    }

    Ok(())
}
