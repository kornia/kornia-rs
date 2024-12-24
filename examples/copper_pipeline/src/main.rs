use cu29::prelude::*;
use cu29_helpers::basic_copper_setup;

const SLAB_SIZE: Option<usize> = Some(100 * 1024 * 1024);

#[copper_runtime(config = "kornia_app.ron")]
struct KorniaApplication {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp_dir = tempfile::TempDir::new().expect("could not create a tmp dir");
    let logger_path = tmp_dir.path().join("kornia_app.copper");
    let copper_ctx =
        basic_copper_setup(&logger_path, SLAB_SIZE, false, None).expect("Failed to setup copper.");

    let clock = copper_ctx.clock;

    let mut application = KorniaApplication::new(clock.clone(), copper_ctx.unified_logger.clone())
        .expect("Failed to create application.");

    debug!("Running... starting clock: {}.", clock.now());

    application.start_all_tasks()?;

    application.run()?;

    application.stop_all_tasks()?;

    debug!("End of program: {}.", clock.now());

    Ok(())
}
