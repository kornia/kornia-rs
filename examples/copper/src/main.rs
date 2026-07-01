use cu29::prelude::*;

const SLAB_SIZE: Option<usize> = Some(150 * 1024 * 1024);

#[copper_runtime(config = "kornia_app.ron")]
struct KorniaApplication {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp_dir = tempfile::TempDir::new().expect("could not create a tmp dir");
    let logger_path = tmp_dir.path().join("kornia_app.copper");

    let mut application = KorniaApplication::builder()
        .with_log_path(&logger_path, SLAB_SIZE)?
        .build()
        .expect("Failed to create application.");

    let clock = application.clock();

    debug!("Logger path: {}", path = &logger_path);

    debug!("Running... starting clock: {}.", clock.now());

    application.start_all_tasks()?;

    application.run()?;

    application.stop_all_tasks()?;

    debug!("End of program: {}.", clock.now());

    Ok(())
}
