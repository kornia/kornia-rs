use dora_node_api::{
    self, arrow::array::UInt8Array, dora_core::config::DataId, DoraNode, Event, Parameter,
};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::{
    color::{self, YuvToRgbMode},
    filter,
};
use kornia_io::jpeg;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut node, mut events) = DoraNode::init_from_env()?;

    let output = DataId::from("output".to_owned());

    let mut img_rgb8 = None;
    let mut out = None;

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => match id.as_str() {
                "frame" => {
                    // decode the frame from arrow
                    let height = match metadata.parameters.get("height") {
                        Some(Parameter::Integer(height)) => height,
                        _ => return Err("Height not found".into()),
                    };
                    let width = match metadata.parameters.get("width") {
                        Some(Parameter::Integer(width)) => width,
                        _ => return Err("Width not found".into()),
                    };
                    let encoding = match metadata.parameters.get("encoding") {
                        Some(Parameter::String(encoding)) => encoding,
                        _ => return Err("Encoding not found".into()),
                    };

                    let data_arr: &UInt8Array = data.as_any().downcast_ref().unwrap();
                    let data_slice = data_arr.values();

                    // lazy init the image
                    if img_rgb8.is_none() {
                        img_rgb8 = Some(Image::from_size_val(
                            ImageSize {
                                width: *width as usize,
                                height: *height as usize,
                            },
                            0,
                            CpuAllocator,
                        )?);
                    }
                    // SAFETY: we know that img_rgb8 is not None
                    let img_rgb8 = img_rgb8.as_mut().unwrap();

                    if encoding == "YUYV" {
                        color::convert_yuyv_to_rgb_u8(
                            data_slice,
                            img_rgb8,
                            YuvToRgbMode::Bt601Full,
                        )?;
                    } else if encoding == "MJPG" {
                        jpeg::decode_image_jpeg_rgb8(data_slice, img_rgb8)?;
                    } else {
                        return Err(format!("Unsupported encoding: {encoding}").into());
                    }

                    // lazily allocate the output image
                    if out.is_none() {
                        out = Some(Image::from_size_val(img_rgb8.size(), 0f32, CpuAllocator)?);
                    }
                    // SAFETY: we know that out is not None
                    let out = out.as_mut().unwrap();

                    // compute the sobel edge map
                    filter::sobel(&img_rgb8.cast()?, out, 3)?;

                    // cast back to u8
                    let out_u8 = out.map(|x| *x as u8)?;

                    let mut params = metadata.parameters;
                    if let Some(p) = params.get_mut("encoding") {
                        *p = Parameter::String("RGB8".to_string());
                    }

                    node.send_output_bytes(
                        output.clone(),
                        params,
                        out_u8.numel(),
                        out_u8.as_slice(),
                    )?;
                }
                other => eprintln!("Ignoring unexpected input `{other}`"),
            },
            Event::Stop(_) => {
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
