use dora_node_api::{self, dora_core::config::DataId, DoraNode, Event};
use kornia::{
    image::{
        arrow::{IntoArrow, TryFromArrow},
        Image,
    },
    imgproc,
    tensor::CpuAllocator,
};

fn main() -> eyre::Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;

    let output = DataId::from("output".to_owned());

    let mut out = None;

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => match id.as_str() {
                "frame" => {
                    // convert the frame to an image
                    let img = Image::<u8, 3, CpuAllocator>::try_from_arrow(data.into())?;

                    // lazily allocate the output image
                    let mut out =
                        out.get_or_insert(Image::from_size_val(img.size(), 0f32, CpuAllocator)?);

                    // compute the sobel edge map
                    imgproc::filter::sobel(&img.cast()?, &mut out, 3)?;

                    // cast back to u8
                    let out_u8 = out.map(|x| *x as u8)?;

                    node.send_output(output.clone(), metadata.parameters, out_u8.into_arrow())?;
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
