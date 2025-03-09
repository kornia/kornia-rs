use dora_node_api::{ArrowData, DoraNode, Event, Metadata, Parameter};
use kornia::image::{Image, ImageSize};

fn main() -> eyre::Result<()> {
    let (mut _node, mut events) = DoraNode::init_from_env()?;

    let rr = rerun::RecordingStreamBuilder::new("Camera Sink").connect_tcp()?;

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => match id.as_str() {
                "web-camera/frame" => {
                    log_image(&rr, "web-camera/frame", &deserialize_frame(data, metadata)?)?;
                }
                "rtsp-camera/frame" => {
                    log_image(
                        &rr,
                        "rtsp-camera/frame",
                        &deserialize_frame(data, metadata)?,
                    )?;
                }
                "imgproc/output" => {
                    log_image(&rr, "imgproc/output", &deserialize_frame(data, metadata)?)?;
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

fn deserialize_frame(data: ArrowData, metadata: Metadata) -> eyre::Result<Image<u8, 3>> {
    // SAFETY: we know that the metadata has the "cols" parameter
    let img_cols = metadata.parameters.get("cols").unwrap();
    let img_cols: i64 = match img_cols {
        Parameter::Integer(i) => *i,
        _ => return Err(eyre::eyre!("cols is not an integer")),
    };

    // SAFETY: we know that the metadata has the "rows" parameter
    let img_rows = metadata.parameters.get("rows").unwrap();
    let img_rows: i64 = match img_rows {
        Parameter::Integer(i) => *i,
        _ => return Err(eyre::eyre!("rows is not an integer")),
    };

    let img_data: Vec<u8> = TryFrom::try_from(&data)?;

    Ok(Image::new(
        ImageSize {
            width: img_cols as usize,
            height: img_rows as usize,
        },
        img_data,
    )?)
}
