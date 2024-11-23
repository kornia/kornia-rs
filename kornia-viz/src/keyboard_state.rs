use eframe::egui;

#[derive(Default)]
pub struct KeyboardState {
    pub text: String,
}

impl KeyboardState {
    pub fn update(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        self.key_pressed(ctx, ui, egui::Key::Space);
        self.key_pressed(ctx, ui, egui::Key::K);
        self.key_pressed(ctx, ui, egui::Key::I);
        self.key_pressed(ctx, ui, egui::Key::J);
        self.key_pressed(ctx, ui, egui::Key::L);
    }

    fn key_pressed(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, key: egui::Key) {
        if ctx.input(|i| i.key_pressed(key)) {
            log::info!("{:?} key pressed", key);
            self.text = format!("{:?} key pressed\n", key);
        }
        if ctx.input(|i| i.key_down(key)) {
            self.text = format!("{:?} key held\n", key);
            ui.ctx().request_repaint();
        }
        if ctx.input(|i| i.key_released(key)) {
            self.text = format!("{:?} key released\n", key);
        }
    }
}
