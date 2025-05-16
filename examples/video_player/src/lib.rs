use eframe::egui::{self, Grid};
use eframe::egui::{CentralPanel, TextEdit};
use humanize_duration::{prelude::*, Truncate};
use kornia::image::{Image, ImageSize};
use kornia::imgproc::resize::resize_fast;
use kornia::io::stream::video::{ImageFormat, SeekFlags, VideoReader};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const SLIDER_HEIGHT: f32 = 20.;

pub struct MyApp {
    file_path: String,
    slider_pos: u64,
    playback_speed: u32,
    fps: f64,
    app_state: AppState,
}

struct TextureStore {
    image: Image<u8, 3>,
    texture_handle: egui::TextureHandle,
}

#[derive(Default)]
enum AppState {
    /// The video is currently playing. The tuple contains:
    /// - `VideoReader`
    /// - `Instant` of last loaded frame
    /// - `TextureStore`
    Playing(VideoReader, Instant, Option<TextureStore>),
    /// The video is paused
    Paused(VideoReader),
    /// The video has been finished
    Finished(VideoReader),
    /// The video is not loaded
    #[default]
    Stopped,
}

impl AppState {
    fn inner(&mut self) -> Option<&mut VideoReader> {
        match self {
            AppState::Playing(reader, ..) => Some(reader),
            AppState::Paused(reader) => Some(reader),
            AppState::Finished(reader) => Some(reader),
            AppState::Stopped => None,
        }
    }
}

impl PartialEq for AppState {
    fn eq(&self, other: &Self) -> bool {
        use AppState::*;
        matches!(
            (self, other),
            (Playing(..), Playing(..))
                | (Paused(..), Paused(..))
                | (Finished(..), Finished(..))
                | (Stopped, Stopped)
        )
    }
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            slider_pos: 0,
            playback_speed: 1,
            fps: 8.,
            app_state: AppState::default(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        CentralPanel::default().show(ctx, |ui| {
            render_navbar(self, ui);

            if self.app_state.inner().is_some() {
                render_play_controls(self, ctx, ui);
                render_image(self, ui);
            }
        });
    }
}

fn render_navbar(app: &mut MyApp, ui: &mut eframe::egui::Ui) {
    ui.horizontal(|ui| {
        // Clear Button
        ui.add_enabled_ui(app.app_state != AppState::Stopped, |ui| {
            if ui.button("clear").clicked() {
                app.app_state = AppState::Stopped;
                app.playback_speed = 1;
            }
        });

        // Load Button
        ui.add_enabled_ui(app.app_state == AppState::Stopped, |ui| {
            if ui.button("load").clicked() {
                log::info!("Loading the video, Path: {}", app.file_path);
                let file_path = PathBuf::from(&app.file_path);
                let mut video_reader = VideoReader::new(&file_path, ImageFormat::Rgb8).unwrap();
                video_reader.start().unwrap();

                app.app_state = AppState::Playing(video_reader, Instant::now(), None);
            }

            ui.add_sized(
                [ui.available_width(), ui.available_height()],
                TextEdit::singleline(&mut app.file_path).hint_text("Click to set path"),
            )
        });
    });
}

// This function only runs if `VideoReader` is present
fn render_play_controls(app: &mut MyApp, ctx: &eframe::egui::Context, ui: &mut eframe::egui::Ui) {
    // Slider
    let mut is_finished = false;
    if let AppState::Playing(video_reader, ..) = &mut app.app_state {
        if let Some(duration) = video_reader.get_duration() {
            if let Some(current_pos) = video_reader.get_pos() {
                app.slider_pos = current_pos.as_secs();

                // Video is now finished
                if current_pos == duration {
                    is_finished = true;
                }
            }

            if ui
                .add_sized(
                    egui::Vec2::new(ui.available_width(), SLIDER_HEIGHT),
                    egui::Slider::new(&mut app.slider_pos, 0..=duration.as_secs()),
                )
                .interact(egui::Sense::DRAG)
                .drag_stopped()
            {
                let seek_time = std::time::Duration::from_secs(app.slider_pos);

                #[allow(unused_must_use)]
                video_reader.seek(SeekFlags::TRICKMODE, seek_time);
            }
        }
    }

    if is_finished {
        let app_state = match std::mem::take(&mut app.app_state) {
            AppState::Playing(video_player, ..) => AppState::Finished(video_player),
            _ => panic!("Unexpected AppState"),
        };
        app.app_state = app_state;
    }

    // egui Windows
    // Control Window
    egui::Window::new("Control").show(ctx, |ui| {
        ui.horizontal(|ui| {
            Grid::new("control_grid").show(ui, |ui| {
                let button_text = match app.app_state {
                    AppState::Playing(..) => "Pause",
                    AppState::Paused(..) => "Resume",
                    AppState::Finished(..) => "Restart",
                    _ => "Play",
                };
                ui.label("State");
                if ui.button(button_text).clicked() {
                    let app_state = std::mem::take(&mut app.app_state);
                    match app_state {
                        AppState::Playing(mut video_reader, ..) => {
                            video_reader.pause().expect("Failed to pause video");
                            app.app_state = AppState::Paused(video_reader);
                        }
                        AppState::Paused(mut video_reader) => {
                            video_reader
                                .start()
                                .expect("Failed to start/resume the video");
                            app.app_state = AppState::Playing(video_reader, Instant::now(), None);
                        }
                        AppState::Finished(mut video_reader) => {
                            video_reader.reset().expect("Failed to reset video");
                            video_reader.start().expect("Failed to start the video");
                            app.app_state = AppState::Playing(video_reader, Instant::now(), None);
                        }
                        _ => {
                            log::error!("Unrecognized AppState");
                            app.app_state = app_state
                        }
                    }
                }
                ui.end_row();

                ui.label("Playback Speed");
                if ui
                    .add(egui::Slider::new(&mut app.playback_speed, 0..=8))
                    .changed()
                    && app
                        .app_state
                        .inner()
                        .unwrap()
                        .set_playback_speed(app.playback_speed as f64)
                        .is_err()
                {
                    log::error!("Failed to set playback speed");
                }
                ui.end_row();
            });
        });
    });

    // Info Window
    egui::Window::new("Info").show(ctx, |ui| {
        ui.horizontal(|ui| {
            Grid::new("info_grid").show(ui, |ui| {
                let video_reader = app.app_state.inner().unwrap();

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

                if let Some(fps) = video_reader.get_fps() {
                    ui.label("FPS:");
                    ui.label(format!("{:.2}", fps));
                    ui.end_row();
                }
            })
        });
    });
}

fn render_image(app: &mut MyApp, ui: &mut eframe::egui::Ui) {
    if let AppState::Playing(video_reader, instant, texture_store) = &mut app.app_state {
        if let Some(fps) = video_reader.get_fps() {
            app.fps = fps;
        }

        if instant.elapsed()
            < Duration::from_millis(((1000. / (app.fps * app.playback_speed as f64)) * 1.1) as u64)
            && texture_store.is_some()
        {
            // Don't grab the new image, load the previous one instead
        } else if let Some(image_frame) =
            video_reader.grab_rgb8().expect("Failed to grab the image")
        {
            *instant = Instant::now();

            let mut resize_required = false;
            let new_image_size = if image_frame.width() as f32 <= ui.available_width()
                && image_frame.height() as f32 - SLIDER_HEIGHT <= ui.available_height()
            {
                image_frame.size()
            } else {
                resize_required = true;
                let aspect_ratio = image_frame.width() as f32 / image_frame.height() as f32;
                // Try scaling by width
                let mut new_width = ui.available_width();
                let mut new_height = new_width / aspect_ratio;

                if new_height > ui.available_height() - SLIDER_HEIGHT {
                    new_height = ui.available_height() - SLIDER_HEIGHT;
                    new_width = new_height * aspect_ratio;
                }

                ImageSize {
                    width: new_width as usize,
                    height: new_height as usize,
                }
            };

            if resize_required {
                if let Some(ts) = texture_store {
                    let dst = &mut ts.image;
                    resize_fast(
                        &image_frame,
                        dst,
                        kornia::imgproc::interpolation::InterpolationMode::Nearest,
                    )
                    .expect("Failed to resize frame");

                    let color_image =
                        egui::ColorImage::from_rgb([dst.width(), dst.height()], dst.as_slice());

                    ts.texture_handle
                        .set(color_image, egui::TextureOptions::default());
                } else {
                    let mut dst: Image<u8, 3> =
                        Image::from_size_val(new_image_size, 0).expect("Failed to create Image");
                    resize_fast(
                        &image_frame,
                        &mut dst,
                        kornia::imgproc::interpolation::InterpolationMode::Nearest,
                    )
                    .expect("Failed to resize frame");

                    let color_image = egui::ColorImage::from_rgb(
                        [new_image_size.width, new_image_size.height],
                        dst.as_slice(),
                    );

                    let texture_handle = ui.ctx().load_texture(
                        "video_frame",
                        color_image,
                        egui::TextureOptions::default(),
                    );

                    *texture_store = Some(TextureStore {
                        image: dst,
                        texture_handle,
                    });
                }
            } else {
                let color_image = egui::ColorImage::from_rgb(
                    [image_frame.width(), image_frame.height()],
                    image_frame.as_slice(),
                );

                if let Some(ts) = texture_store {
                    ts.texture_handle
                        .set(color_image, egui::TextureOptions::default());
                } else {
                    let texture = ui.ctx().load_texture(
                        "video_frame",
                        color_image,
                        egui::TextureOptions::default(),
                    );

                    *texture_store = Some(TextureStore {
                        image: image_frame,
                        texture_handle: texture,
                    });
                };
            }
        }

        if let Some(texture_store) = &texture_store {
            let sized_texture = egui::load::SizedTexture::new(
                texture_store.texture_handle.id(),
                egui::Vec2::new(
                    texture_store.image.width() as f32,
                    texture_store.image.height() as f32,
                ),
            );

            ui.image(sized_texture);
        }
    }
}
