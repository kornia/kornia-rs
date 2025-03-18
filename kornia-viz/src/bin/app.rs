use eframe::egui::ScrollArea;
use eframe::egui::{self, TextureOptions};

use kornia::io::fps_counter::FpsCounter;
use kornia_viz::keyboard_state::KeyboardState;

fn main() -> eframe::Result<()> {
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Kornia Gui",
        options,
        Box::new(|_cc| Ok(Box::<KorniaApp>::default())),
    )
}
struct KorniaApp {
    name: String,
    fps_counter: FpsCounter,
    keyboard_state: KeyboardState,
    capture: kornia::io::stream::CameraCapture,
    last_image: Option<kornia::image::Image<u8, 3>>,
}

impl Default for KorniaApp {
    fn default() -> Self {
        // Use NVCameraConfig instead of V4L2CameraConfig for GPU acceleration
        let capture = kornia::io::stream::NVCameraConfig::new()
            .with_camera_id(0)
            .build()
            .unwrap();
        capture.start().unwrap();

        Self {
            name: "Kornia".to_string(),
            fps_counter: FpsCounter::default(),
            keyboard_state: KeyboardState::default(),
            capture,
            last_image: None,
        }
    }
}

impl eframe::App for KorniaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(&self.name);
            ui.label(format!("FPS: {:.2}", self.fps_counter.fps()));

            ScrollArea::vertical().show(ui, |ui| {
                ui.label(&self.keyboard_state.text);
            });

            if let Ok(Some(img)) = self.capture.grab() {
                self.last_image = Some(img);
            }

            if let Some(img) = self.last_image.as_ref() {
                let color_image =
                    egui::ColorImage::from_rgb([img.cols(), img.rows()], img.as_slice());
                let texture = ctx.load_texture("test", color_image, TextureOptions::default());
                ui.image(&texture);
            }

            self.fps_counter.update();
            self.keyboard_state.update(ctx, ui);
            ctx.request_repaint();
        });
    }
}
