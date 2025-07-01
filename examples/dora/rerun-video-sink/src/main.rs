use dora_node_api::{arrow::array::UInt8Array, DoraNode, Event, Parameter};
use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    Image, ImageSize,
};
use kornia_imgproc::color::{self, YuvToRgbMode};
use kornia_io::jpeg;

const RERUN_HOST: &str = "127.0.0.1";
const RERUN_PORT: u16 = 9876;

// TODO: move this to kornia-image crate
#[derive(Debug)]
struct ImageView<'a, const C: usize> {
    pub data: &'a [u8],
    pub size: ImageSize,
}

impl<'a, A: ImageAllocator> From<&'a Image<u8, 3, A>> for ImageView<'a, 3> {
    fn from(img: &'a Image<u8, 3, A>) -> Self {
        ImageView {
            data: img.as_slice(),
            size: img.size(),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut _node, mut events) = DoraNode::init_from_env()?;

    // setup rerun
    let rr_host = std::env::var("RERUN_HOST").unwrap_or_else(|_| RERUN_HOST.to_string());
    let rr_port = std::env::var("RERUN_PORT")
        .unwrap_or_else(|_| RERUN_PORT.to_string())
        .parse::<u16>()?;

    let rr = rerun::RecordingStreamBuilder::new("Camera Sink")
        .connect_grpc_opts(format!("rerun+http://{rr_host}:{rr_port}/proxy"), None)?;

    let mut img_rgb8 = None;

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => {
                // NOTE: we need to convert the timestamp to nanoseconds
                let timestamp_nanos = {
                    let timestamp_secs = metadata.timestamp().get_time().as_secs() as u64;
                    let timestamp_subsec_nanos =
                        metadata.timestamp().get_time().subsec_nanos() as u64;
                    timestamp_secs * 1_000_000_000 + timestamp_subsec_nanos
                };

                let width = match metadata.parameters.get("width") {
                    Some(Parameter::Integer(width)) => width,
                    _ => return Err("Width not found".to_string().into()),
                };
                let height = match metadata.parameters.get("height") {
                    Some(Parameter::Integer(height)) => height,
                    _ => return Err("Height not found".to_string().into()),
                };
                let encoding = match metadata.parameters.get("encoding") {
                    Some(Parameter::String(encoding)) => encoding,
                    _ => return Err("Encoding not found".to_string().into()),
                };

                // create a view into the data
                let data_arr: &UInt8Array = data.as_any().downcast_ref().unwrap();
                let data_slice = data_arr.values();

                let img_rgb8_view: ImageView<3> = {
                    ImageView {
                        data: data_slice,
                        size: ImageSize {
                            width: *width as usize,
                            height: *height as usize,
                        },
                    }
                };

                // lazy init the RGB8 image
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

                rr.set_time(
                    "time",
                    rerun::TimeCell::from_duration_nanos(timestamp_nanos as i64),
                );

                // log directly if the encoding is RGB8
                if encoding == "RGB8" {
                    log_image(&rr, id.as_str(), img_rgb8_view)?;
                } else {
                    if encoding == "YUYV" {
                        color::convert_yuyv_to_rgb_u8(
                            img_rgb8_view.data,
                            img_rgb8,
                            YuvToRgbMode::Bt601Full,
                        )?;
                    } else if encoding == "MJPG" {
                        jpeg::decode_image_jpeg_rgb8(data_slice, img_rgb8)?;
                    } else {
                        return Err(format!("Unsupported encoding: {encoding}").into());
                    }
                    log_image(&rr, id.as_str(), (&*img_rgb8).into())?;
                }
            }
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

fn log_image(
    rr: &rerun::RecordingStream,
    name: &str,
    img: ImageView<3>,
) -> Result<(), Box<dyn std::error::Error>> {
    rr.log(
        name,
        &rerun::Image::from_elements(
            img.data,
            [img.size.width as u32, img.size.height as u32],
            rerun::ColorModel::RGB,
        ),
    )?;
    Ok(())
}
