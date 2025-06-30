use dora_node_api::{self, dora_core::config::DataId, DoraNode, Event, IntoArrow, Parameter};
use kornia::{
    image::{Image, ImageSize},
    imgproc::{self, color::YuvToRgbMode},
    io,
    tensor::CpuAllocator,
};

fn main() -> eyre::Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;

    let output = DataId::from("output".to_owned());

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => match id.as_str() {
                "frame" => {
                    // decode the frame from arrow
                    let rows = match metadata.parameters.get("height") {
                        Some(Parameter::Integer(rows)) => rows,
                        _ => return Err(eyre::eyre!("Height not found")),
                    };
                    let cols = match metadata.parameters.get("width") {
                        Some(Parameter::Integer(cols)) => cols,
                        _ => return Err(eyre::eyre!("Width not found")),
                    };
                    let encoding = match metadata.parameters.get("encoding") {
                        Some(Parameter::String(encoding)) => encoding,
                        _ => return Err(eyre::eyre!("Encoding not found")),
                    };

                    let data_arr = data.to_data();
                    let data_slice = data_arr.buffer(0);

                    let mut img_rgb8 = Image::from_size_val(
                        ImageSize {
                            width: *cols as usize,
                            height: *rows as usize,
                        },
                        0,
                        CpuAllocator,
                    )?;

                    if encoding == "YUYV" {
                        imgproc::color::convert_yuyv_to_rgb_u8(
                            data_slice,
                            &mut img_rgb8,
                            YuvToRgbMode::Bt601Full,
                        )?;
                    } else if encoding == "MJPG" {
                        io::jpeg::decode_image_jpeg_rgb8(data_slice, &mut img_rgb8)?;
                    } else {
                        return Err(eyre::eyre!("Unsupported encoding: {}", encoding));
                    }

                    // lazily allocate the output image
                    let mut out = Image::from_size_val(img_rgb8.size(), 0f32, CpuAllocator)?;

                    // compute the sobel edge map
                    imgproc::filter::sobel(&img_rgb8.cast()?, &mut out, 3)?;

                    // cast back to u8
                    let out_u8 = out.map(|x| *x as u8)?;

                    let mut params = metadata.parameters;
                    if let Some(p) = params.get_mut("encoding") {
                        *p = Parameter::String("RGB8".to_string());
                    }

                    node.send_output(output.clone(), params, out_u8.into_vec().into_arrow())?;
                }
                other => eprintln!("Ignoring unexpected input `{other}`"),
            },
            Event::Stop => {
                println!("Received manual stop");
            }
            Event::InputClosed { id } => {
                println!("Input `{id}` was closed");
            }
            other => eprintln!("Received unexpected input: {other:?}"),
        }
    }

    Ok(())
}
