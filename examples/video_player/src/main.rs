use eframe::egui;
use video_player::MyApp;

fn main() -> eframe::Result {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280., 720.]),
        // hardware_acceleration: eframe::HardwareAcceleration::Off,
        ..Default::default()
    };
    eframe::run_native(
        "Kornia-rs Simple Video Player",
        options,
        Box::new(|_| Ok(Box::<MyApp>::default())),
    )
}
