use dora_node_api::{DoraNode, Event, Parameter};
use kornia::{
    image::{allocator::ImageAllocator, arrow::TryFromArrow, Image, ImageSize},
    imgproc::{self, color::YuvToRgbMode},
    io,
    tensor::CpuAllocator,
};

const RERUN_HOST: &str = "127.0.0.1";
const RERUN_PORT: u16 = 9876;

fn main() -> eyre::Result<()> {
    let (mut _node, mut events) = DoraNode::init_from_env()?;

    // setup rerun
    let rr_host = std::env::var("RERUN_HOST").unwrap_or_else(|_| RERUN_HOST.to_string());
    let rr_port = std::env::var("RERUN_PORT")
        .unwrap_or_else(|_| RERUN_PORT.to_string())
        .parse::<u16>()?;

    let rr = rerun::RecordingStreamBuilder::new("Camera Sink")
        .connect_grpc_opts(format!("rerun+http://{rr_host}:{rr_port}/proxy"), None)?;

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

                //let image = Image::<u8, 3, _>::try_from_arrow(data.into())?;
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
                let data_slice = data_arr.buffer(0).to_vec();

                // log directly if the encoding is RGB8
                if encoding == "RGB8" {
                    let img_rgb8 = Image::new(
                        ImageSize {
                            width: *cols as usize,
                            height: *rows as usize,
                        },
                        data_slice,
                        CpuAllocator,
                    )?;
                    log_image(&rr, id.as_str(), timestamp_nanos, &img_rgb8)?;
                } else {
                    // decode the frame to rgb8
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
                            &data_slice,
                            &mut img_rgb8,
                            YuvToRgbMode::Bt601Full,
                        )?;
                    } else if encoding == "MJPG" {
                        io::jpeg::decode_image_jpeg_rgb8(&data_slice, &mut img_rgb8)?;
                    } else {
                        return Err(eyre::eyre!("Unsupported encoding: {}", encoding));
                    }

                    // log the image to rerun
                    log_image(&rr, id.as_str(), timestamp_nanos, &img_rgb8)?;
                }
            }
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

fn log_image<A: ImageAllocator>(
    rr: &rerun::RecordingStream,
    name: &str,
    timestamp_nanos: u64,
    img: &Image<u8, 3, A>,
) -> eyre::Result<()> {
    rr.set_time(
        "time",
        rerun::TimeCell::from_duration_nanos(timestamp_nanos as i64),
    );
    rr.log(
        name,
        &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::RGB),
    )?;
    Ok(())
}
