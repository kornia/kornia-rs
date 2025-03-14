use dora_image_utils::arrow_to_image;
use dora_node_api::{DoraNode, Event};
use kornia::image::Image;

fn main() -> eyre::Result<()> {
    let (mut _node, mut events) = DoraNode::init_from_env()?;

    let rr = rerun::RecordingStreamBuilder::new("Camera Sink").connect_tcp()?;

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => match id.as_str() {
                "web-camera/frame" => {
                    log_image(&rr, "web-camera/frame", &arrow_to_image(data, metadata)?)?;
                }
                "rtsp-camera/frame" => {
                    log_image(&rr, "rtsp-camera/frame", &arrow_to_image(data, metadata)?)?;
                }
                "imgproc/output" => {
                    log_image(&rr, "imgproc/output", &arrow_to_image(data, metadata)?)?;
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

fn log_image(rr: &rerun::RecordingStream, name: &str, img: &Image<u8, 3>) -> eyre::Result<()> {
    rr.log_static(
        name,
        &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::RGB),
    )?;

    Ok(())
}
