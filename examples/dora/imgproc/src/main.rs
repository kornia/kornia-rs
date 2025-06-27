use dora_image_utils::{arrow_to_image, image_to_arrow};
use dora_node_api::{self, dora_core::config::DataId, DoraNode, Event, IntoArrow};
use kornia::{image::Image, imgproc, tensor::CpuAllocator};

fn main() -> eyre::Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;

    let output = DataId::from("output".to_owned());

    let mut out = None;

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => match id.as_str() {
                "frame" => {
                    // convert the frame to an image
                    let img = arrow_to_image(data, metadata.clone())?;

                    // compute the sobel edge map
                    let mut out =
                        out.get_or_insert(Image::from_size_val(img.size(), 0f32, CpuAllocator)?);
                    imgproc::filter::sobel(&img.cast()?, &mut out, 3)?;

                    // TODO: make this more efficient in kornia-image crate
                    let out_u8 = out.map(|x| *x as u8)?;

                    let (meta_parameters, data) = image_to_arrow(out_u8, metadata)?;
                    node.send_output(output.clone(), meta_parameters, data.into_arrow())?;
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
