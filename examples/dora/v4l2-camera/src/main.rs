use dora_node_api::{self, dora_core::config::DataId, DoraNode, Event, IntoArrow};
use kornia::io::stream::V4L2CameraConfig;

fn main() -> eyre::Result<()> {
    // TODO: add a config file to set the camera parameters
    let mut camera = V4L2CameraConfig::new().build()?;
    camera.start()?;

    let output = DataId::from("frame".to_owned());

    let (mut node, mut events) = DoraNode::init_from_env()?;

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => match id.as_str() {
                "tick" => {
                    let Some(frame) = camera.grab()? else {
                        continue;
                    };

                    let img_size = frame.size();

                    println!("Received frame: {id} {metadata:?} {data:?} frame: {img_size}");

                    node.send_output(
                        output.clone(),
                        metadata.parameters,
                        // TODO: avoid to_vec copy
                        frame.as_slice().to_vec().into_arrow(),
                    )?;
                }
                other => eprintln!("Ignoring unexpected input `{other}`"),
            },
            Event::Stop => {
                camera.close()?;
            }
            other => eprintln!("Received unexpected input: {other:?}"),
        }
    }

    Ok(())
}
