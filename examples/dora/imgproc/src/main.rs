use dora_node_api::{
    self, dora_core::config::DataId, ArrowData, DoraNode, Event, IntoArrow, Metadata, Parameter,
};
use kornia::{
    image::{Image, ImageSize},
    imgproc,
};

fn main() -> eyre::Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;

    let output = DataId::from("output".to_owned());

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => match id.as_str() {
                "frame" => {
                    // deserialize the frame
                    let img = deserialize_frame(data, metadata.clone())?;

                    // compute the sobel edge map
                    let mut out = Image::from_size_val(img.size(), 0f32)?;
                    imgproc::filter::sobel(&img.cast()?, &mut out, 3)?;

                    let out_u8 = out.map(|x| *x as u8);

                    let mut meta_parameters = metadata.parameters;
                    meta_parameters
                        .insert("cols".to_string(), Parameter::Integer(img.cols() as i64));
                    meta_parameters
                        .insert("rows".to_string(), Parameter::Integer(img.rows() as i64));

                    node.send_output(
                        output.clone(),
                        meta_parameters,
                        out_u8.as_slice().to_vec().into_arrow(),
                    )?;
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

fn deserialize_frame(data: ArrowData, metadata: Metadata) -> eyre::Result<Image<u8, 3>> {
    let img_cols = metadata.parameters.get("cols").unwrap();
    let img_cols: i64 = match img_cols {
        Parameter::Integer(i) => *i,
        _ => return Err(eyre::eyre!("cols is not an integer")),
    };
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
