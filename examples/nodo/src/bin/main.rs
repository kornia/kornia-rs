use argh::FromArgs;
use nodo::prelude::IntoInstance;
use nodo_runtime::Runtime;

use nodo_pipeline::codelets::webcam::Webcam;

#[derive(FromArgs)]
/// A simple example of a nodo application
struct Args {
    /// the name of the person to greet
    #[argh(option)]
    name: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args: Args = argh::from_env();
    println!("Hello, {}!", args.name);

    let mut rt = Runtime::new();

    let webcam = Webcam::new()?.into_instance("webcam", ());

    rt.add_codelet_schedule(
        nodo::codelet::ScheduleBuilder::new()
            .with_name("camera_task")
            .with(webcam)
            .into(),
    );

    rt.enable_terminate_on_ctrl_c();
    rt.spin();

    Ok(())
}
