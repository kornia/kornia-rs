use dora_node_api::{self, DoraNode, Event};

fn main() -> eyre::Result<()> {
    let (mut _node, mut events) = DoraNode::init_from_env()?;

    while let Some(event) = events.recv() {
        match event {
            Event::Input {
                id,
                metadata: _,
                data,
            } => match id.as_str() {
                "frame" => {
                    // TODO: how to deserialize the frame?
                    let received_value: Vec<u8> = TryFrom::try_from(&data)?;
                    println!("sink received message: {}", received_value.len());
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
