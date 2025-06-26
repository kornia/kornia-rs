use dora_image_utils::arrow_to_image;
use dora_node_api::{DoraNode, Event};
use kornia::image::Image;

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

                // log the image to rerun
                log_image(
                    &rr,
                    id.as_str(),
                    timestamp_nanos,
                    &arrow_to_image(data, metadata)?,
                )?;
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

fn log_image(
    rr: &rerun::RecordingStream,
    name: &str,
    timestamp_nanos: u64,
    img: &Image<u8, 3>,
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
