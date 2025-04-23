use eframe::egui::{self, Grid};
use eframe::egui::{CentralPanel, TextEdit};
use humanize_duration::{prelude::*, Truncate};
use kornia::image::{Image, ImageSize};
use kornia::imgproc::resize::resize_fast;
use kornia::io::stream::video::{ImageFormat, SeekFlags, State, VideoReader};
use std::path::PathBuf;

const SLIDER_HEIGHT: f32 = 20.;

pub struct MyApp {
    file_path: String,
    video_reader: Option<VideoReader>,
    slider_pos: u64,
    playback_speed: u32,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            video_reader: None,
            slider_pos: 0,
            playback_speed: 1,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add_enabled_ui(self.video_reader.is_some(), |ui| {
                    if ui.button("clear").clicked() {
                        self.video_reader = None;
                    }
                });

                ui.add_enabled_ui(self.video_reader.is_none(), |ui| {
                    if ui.button("load").clicked() {
                        log::info!("Loading the Video");
                        let file_path = PathBuf::from(&self.file_path);
                        log::info!("File Path: {}", &self.file_path);
                        let mut video_reader = match VideoReader::new(&file_path, ImageFormat::Rgb8)
                        {
                            Ok(v) => v,
                            Err(err) => {
                                log::error!("Failed to create VideoReader: {}", err);
                                return;
                            }
                        };

                        // Start the video
                        match video_reader.start() {
                            Ok(_) => {}
                            Err(err) => {
                                log::error!("Failed to start the video: {}", err);
                                return;
                            }
                        };
                        self.video_reader = Some(video_reader);
                    }

                    ui.add_sized(
                        [ui.available_width(), ui.available_height()],
                        TextEdit::singleline(&mut self.file_path).hint_text("Click to set path"),
                    );
                });
            });

            if let Some(video_reader) = &mut self.video_reader {
                if let Some(image_frame) = match video_reader.grab() {
                    Ok(f) => f,
                    Err(err) => {
                        log::error!("Failed to grab the image: {}", err);
                        return;
                    }
                } {
                    let (image_size, resized_image) = if image_frame.width() as f32
                        <= ui.available_width()
                        && image_frame.height() as f32 - SLIDER_HEIGHT <= ui.available_height()
                    {
                        ([image_frame.width(), image_frame.height()], image_frame)
                    } else {
                        // The image can't fit, resize it
                        let aspect_ratio = image_frame.width() as f32 / image_frame.height() as f32;

                        // Try scaling by width
                        let mut new_width = ui.available_width();
                        let mut new_height = new_width / aspect_ratio;

                        if new_height > ui.available_height() - SLIDER_HEIGHT {
                            new_height = ui.available_height() - SLIDER_HEIGHT;
                            new_width = new_height * aspect_ratio;
                        }

                        let mut resize_image = match Image::<u8, 3>::from_size_val(
                            ImageSize {
                                width: new_width as usize,
                                height: new_height as usize,
                            },
                            0,
                        ) {
                            Ok(i) => i,
                            Err(err) => {
                                log::error!("Failed to create dst image: {}", err);
                                return;
                            }
                        };

                        if let Err(err) = resize_fast(
                            &image_frame,
                            &mut resize_image,
                            kornia::imgproc::interpolation::InterpolationMode::Nearest,
                        ) {
                            log::error!("Failed to resize video frame: {}", err);
                            return;
                        }

                        ([new_width as usize, new_height as usize], resize_image)
                    };

                    // Render the frame
                    let color_image =
                        egui::ColorImage::from_rgb(image_size, resized_image.as_slice());
                    let texture = ui.ctx().load_texture(
                        "image_frame",
                        color_image,
                        egui::TextureOptions::default(),
                    );

                    let sized_texture =
                        egui::load::SizedTexture::new(texture.id(), texture.size_vec2());
                    ui.image(sized_texture);
                };

                if let Some(duration) = video_reader.get_duration() {
                    if let Some(current_pos) = video_reader.get_pos() {
                        self.slider_pos = current_pos.as_secs();
                    }

                    if ui
                        .add_sized(
                            egui::Vec2::new(ui.available_width(), SLIDER_HEIGHT),
                            egui::Slider::new(&mut self.slider_pos, 0..=duration.as_secs()),
                        )
                        .interact(egui::Sense::DRAG)
                        .drag_stopped()
                    {
                        let seek_time = std::time::Duration::from_secs(self.slider_pos);
                        video_reader.seek(SeekFlags::TRICKMODE, seek_time);
                    }
                }
                // Control Window
                ui.separator();
                egui::Window::new("Control").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        Grid::new("control_grid").show(ui, |ui| {
                            let button_text = match video_reader.get_state() {
                                State::Playing => "Stop",
                                State::Paused => "Play",
                                State::Null => "Restart",
                                _ => "Play",
                            };
                            ui.label("State");
                            if ui.button(button_text).clicked() {
                                match video_reader.get_state() {
                                    State::Playing => match video_reader.pause() {
                                        Ok(_) => {}
                                        Err(err) => {
                                            log::error!("Failed to pause the video: {}", err);
                                            return;
                                        }
                                    },
                                    State::Paused => match video_reader.start() {
                                        Ok(_) => {}
                                        Err(err) => {
                                            log::error!("Failed to start the video: {}", err);
                                            return;
                                        }
                                    },
                                    _ => match video_reader.restart() {
                                        Ok(_) => {}
                                        Err(err) => {
                                            log::error!("Failed to restart the video: {}", err);
                                            return;
                                        }
                                    },
                                }
                            }
                            ui.end_row();

                            ui.label("Playback Speed");
                            if ui
                                .add(egui::Slider::new(&mut self.playback_speed, 0..=8))
                                .changed()
                                && !video_reader.set_playback_speed(self.playback_speed as f64)
                            {
                                log::error!("Failed to set playback speed");
                            };
                            ui.end_row();
                        });
                    })
                });
                // Info Window
                ui.separator();
                egui::Window::new("Info").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        Grid::new("info_grid").show(ui, |ui| {
                            if let Some(current_pos) = video_reader.get_pos() {
                                ui.label("Current Pos:");
                                ui.label(current_pos.human(Truncate::Second).to_string());
                                ui.end_row();
                            }

                            if let Some(duration) = video_reader.get_duration() {
                                ui.label("Duration:");
                                ui.label(duration.human(Truncate::Second).to_string());
                                ui.end_row();
                            }
                        });
                    });
                });
            };
        });
    }
}
