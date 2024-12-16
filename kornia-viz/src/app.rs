use eframe::egui::ScrollArea;
use eframe::egui::{self, TextureOptions};

use crate::keyboard_state::KeyboardState;
//use kornia::io::fps_counter::FpsCounter;

pub struct KorniaApp {
    name: String,
    //fps_counter: FpsCounter,
    keyboard_state: KeyboardState,
    //capture: kornia::io::stream::CameraCapture,
    //last_image: Option<kornia::image::Image<u8, 3>>,
}

impl KorniaApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }
}

impl Default for KorniaApp {
    fn default() -> Self {
        //let mut capture = kornia::io::stream::V4L2CameraConfig::new()
        //    .with_camera_id(0)
        //    .build()
        //    .unwrap();
        //capture.start().unwrap();

        Self {
            name: "Kornia".to_string(),
            //fps_counter: FpsCounter::default(),
            keyboard_state: KeyboardState::default(),
            //capture,
            //last_image: None,
        }
    }
}

impl eframe::App for KorniaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(&self.name);
            //ui.label(format!("FPS: {:.2}", self.fps_counter.fps()));

            ScrollArea::vertical().show(ui, |ui| {
                ui.label(&self.keyboard_state.text);
            });

            //let img = kornia::io::functional::read_image_any(
            //    "/home/edgar/software/kornia-rs/kornia-viz/assets/icon-256.png",
            //)
            //.expect("could not read image");
            //log::info!("image size: {}x{}", img.cols(), img.rows());

            //let color_image = egui::ColorImage::from_rgb([img.cols(), img.rows()], img.as_slice());
            //let texture = ctx.load_texture("test", color_image, TextureOptions::default());
            //ui.image(&texture);

            //if let Ok(Some(img)) = self.capture.grab() {
            //    self.last_image = Some(img);
            //}

            //if let Some(img) = self.last_image.as_ref() {
            //    let color_image =
            //        egui::ColorImage::from_rgb([img.cols(), img.rows()], img.as_slice());
            //    let texture = ctx.load_texture("test", color_image, TextureOptions::default());
            //    ui.image(&texture);
            //}

            //self.fps_counter.update();
            self.keyboard_state.update(ctx, ui);
            ctx.request_repaint();
        });
    }
}
