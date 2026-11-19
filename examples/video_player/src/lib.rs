use eframe::egui::{self, CentralPanel, Grid, TextEdit};
use humanize_duration::{prelude::*, Truncate};
use kornia::image::{Image, ImageSize};
use kornia::imgproc::resize::resize_fast_rgb;
use kornia::io::stream::error::VideoReaderError;
use kornia::io::stream::video::{ImageFormat, SeekFlags, VideoReader};
use kornia::io::stream::Grab;
use std::path::{Path, PathBuf};

const SLIDER_HEIGHT: f32 = 20.;

fn open_player_reader(path: &Path) -> Result<VideoReader, VideoReaderError> {
    let mut reader = VideoReader::new(path, ImageFormat::Rgb8)?;
    // Speed changes seek from the pipeline position, so decoding must track playback time.
    reader.set_realtime(true);
    Ok(reader)
}

pub struct MyApp {
    file_path: String,
    slider_pos: u64,
    playback_speed: u32,
    active_speed: u32,
    /// Set when `try_grab_rgb8` reports `Grab::Eos`, i.e. after every buffered frame was drawn.
    finished: bool,
    /// Carries a grab error into the Finished/Failed transition.
    error: Option<String>,
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
    /// - Texture for the last displayed frame
    Playing(VideoReader, Option<TextureStore>),
    /// The video is paused
    Paused(VideoReader, Option<TextureStore>),
    /// The video has been finished; the last texture is kept on screen until a restart.
    Finished(VideoReader, Option<TextureStore>),
    /// The stream failed; the error is kept and shown, and a retry reopens the file.
    Failed(VideoReader, String, Option<TextureStore>),
    /// The video is not loaded
    #[default]
    Stopped,
}

impl AppState {
    /// Returns the reader for the current state, if a video is loaded.
    fn inner(&mut self) -> Option<&mut VideoReader> {
        match self {
            AppState::Playing(reader, ..) => Some(reader),
            AppState::Paused(reader, ..) => Some(reader),
            AppState::Finished(reader, _) => Some(reader),
            AppState::Failed(reader, ..) => Some(reader),
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
                | (Failed(..), Failed(..))
                | (Stopped, Stopped)
        )
    }
}

impl Default for MyApp {
    /// Creates the player with default playback settings.
    fn default() -> Self {
        Self {
            file_path: String::new(),
            slider_pos: 0,
            playback_speed: 1,
            active_speed: 1,
            finished: false,
            error: None,
            app_state: AppState::default(),
        }
    }
}

impl eframe::App for MyApp {
    fn ui(&mut self, ui: &mut eframe::egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = &ui.ctx().clone();
        ctx.request_repaint();

        CentralPanel::default().show_inside(ui, |ui| {
            render_navbar(self, ui);

            if self.app_state.inner().is_some() {
                render_play_controls(self, ctx);
                render_image(self, ui);
            }
        });
    }
}

/// Draws the file path field and the load button.
fn render_navbar(app: &mut MyApp, ui: &mut eframe::egui::Ui) {
    ui.horizontal(|ui| {
        // Clear Button
        ui.add_enabled_ui(app.app_state != AppState::Stopped, |ui| {
            if ui.button("clear").clicked() {
                app.app_state = AppState::Stopped;
                app.playback_speed = 1;
                app.active_speed = 1;
            }
        });

        // Load Button
        ui.add_enabled_ui(app.app_state == AppState::Stopped, |ui| {
            if ui.button("load").clicked() {
                log::info!("Loading the video, Path: {}", app.file_path);
                let file_path = PathBuf::from(&app.file_path);
                let mut video_reader = open_player_reader(&file_path).unwrap();
                video_reader.start().unwrap();
                app.finished = false;
                app.error = None;

                app.app_state = AppState::Playing(video_reader, None);
            }

            ui.add_sized(
                [ui.available_width(), ui.available_height()],
                TextEdit::singleline(&mut app.file_path).hint_text("Click to set path"),
            )
        });
    });
}

// This function only runs if `VideoReader` is present
/// Draws the playback controls and switches to `Finished`/`Failed` when playback ends.
fn render_play_controls(app: &mut MyApp, ctx: &eframe::egui::Context) {
    let mut is_playing = false;
    let mut is_finished = false;

    if let AppState::Playing(video_reader, ..) = &app.app_state {
        is_playing = true;

        if let Some(current_pos) = video_reader.get_pos() {
            app.slider_pos = current_pos.as_secs();
        }

        // The reader reports end-of-stream explicitly; the position never reaches the duration
        // (the last frame's PTS is always before it). Wait until every buffered frame was drawn.
        is_finished = app.finished;
    }

    if is_finished {
        let app_state = match std::mem::take(&mut app.app_state) {
            AppState::Playing(video_player, texture_store) => match app.error.take() {
                Some(error) => AppState::Failed(video_player, error, texture_store),
                None => AppState::Finished(video_player, texture_store),
            },
            _ => panic!("Unexpected AppState"),
        };
        app.app_state = app_state;
    }

    // egui Windows
    // Control Window
    egui::Window::new("Control").show(ctx, |ui| {
        if let AppState::Failed(_, error, _) = &app.app_state {
            ui.colored_label(egui::Color32::RED, format!("Playback failed: {error}"));
        }
        ui.horizontal(|ui| {
            Grid::new("control_grid").show(ui, |ui| {
                let button_text = match app.app_state {
                    AppState::Playing(..) => "Pause",
                    AppState::Paused(..) => "Resume",
                    AppState::Finished(..) => "Restart",
                    AppState::Failed(..) => "Retry",
                    _ => "Play",
                };
                ui.label("State");
                if ui.button(button_text).clicked() {
                    let app_state = std::mem::take(&mut app.app_state);
                    match app_state {
                        AppState::Playing(mut video_reader, texture_store) => {
                            video_reader.pause().expect("Failed to pause video");
                            app.app_state = AppState::Paused(video_reader, texture_store);
                        }
                        AppState::Paused(mut video_reader, texture_store) => {
                            video_reader
                                .start()
                                .expect("Failed to start/resume the video");
                            app.app_state = AppState::Playing(video_reader, texture_store);
                        }
                        AppState::Finished(mut video_reader, _) => {
                            video_reader.reset().expect("Failed to reset video");
                            app.active_speed = 1;
                            video_reader.start().expect("Failed to start the video");
                            app.finished = false;
                            app.app_state = AppState::Playing(video_reader, None);
                        }
                        AppState::Failed(video_reader, error, texture_store) => {
                            // The failed pipeline may not recover, so retry reopens the file.
                            // A failure to reopen keeps the failed state (and last frame) visible.
                            let file_path = PathBuf::from(&app.file_path);
                            let reopened = match open_player_reader(&file_path) {
                                Ok(mut reader) => match reader.start() {
                                    Ok(()) => Ok(reader),
                                    // `start` can fail with a bare state-change error; the file's
                                    // own reason is already on the reader's bus.
                                    Err(start_error) => Err(reader
                                        .stream_error()
                                        .map(|error| error.to_string())
                                        .unwrap_or_else(|| start_error.to_string())),
                                },
                                Err(error) => Err(error.to_string()),
                            };
                            match reopened {
                                Ok(video_reader) => {
                                    log::warn!("Retrying after playback error: {error}");
                                    app.active_speed = 1;
                                    app.finished = false;
                                    app.error = None;
                                    app.slider_pos = 0;
                                    // Keep the last frame visible until the new pipeline delivers
                                    // the first one; a later failure keeps it too.
                                    app.app_state = AppState::Playing(video_reader, texture_store);
                                }
                                Err(retry_error) => {
                                    app.app_state =
                                        AppState::Failed(video_reader, retry_error, texture_store);
                                }
                            }
                        }
                        _ => {
                            log::error!("Unrecognized AppState");
                            app.app_state = app_state
                        }
                    }
                }
                ui.end_row();

                // Slider
                // SAFTEY: This function only runs when video_reader is present
                let video_reader = unsafe { app.app_state.inner().unwrap_unchecked() };
                ui.label("Seek");

                if let Some(duration) = video_reader.get_duration() {
                    ui.add_enabled_ui(is_playing, |ui| {
                        if ui
                            .add_sized(
                                egui::Vec2::new(ui.available_width(), SLIDER_HEIGHT),
                                egui::Slider::new(&mut app.slider_pos, 0..=duration.as_secs()),
                            )
                            .interact(egui::Sense::DRAG)
                            .drag_stopped()
                        {
                            let seek_time = std::time::Duration::from_secs(app.slider_pos);

                            if video_reader
                                .seek(SeekFlags::TRICKMODE | SeekFlags::FLUSH, seek_time)
                                .is_ok()
                            {
                                app.active_speed = 1;
                            }
                        }
                    });
                }

                ui.end_row();

                ui.label("Playback Speed");
                ui.add(egui::Slider::new(&mut app.playback_speed, 1..=8));
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
                    ui.label(format!("{fps:.2}"));
                    ui.end_row();
                }
            })
        });
    });
}

/// Applies the selected speed once a frame confirms that the current segment has started.
fn apply_playback_speed(reader: &VideoReader, selected_speed: &mut u32, active_speed: &mut u32) {
    if *selected_speed != *active_speed {
        match reader.set_playback_speed(f64::from(*selected_speed)) {
            Ok(()) => *active_speed = *selected_speed,
            Err(error) => {
                log::error!("Failed to set playback speed: {error}");
                *selected_speed = *active_speed;
            }
        }
    }
}

/// Draws the current frame. The reader's real-time mode paces delivery.
fn render_image(app: &mut MyApp, ui: &mut eframe::egui::Ui) {
    match &mut app.app_state {
        AppState::Playing(reader, texture_store) => {
            if let Some(image_frame) = match reader.try_grab_rgb8() {
                Ok(Grab::Frame { image, .. }) => {
                    apply_playback_speed(reader, &mut app.playback_speed, &mut app.active_speed);
                    Some(image)
                }
                Ok(Grab::Eos) => {
                    app.finished = true;
                    None
                }
                Ok(Grab::Pending) => None,
                Err(error) => {
                    app.finished = true;
                    app.error = Some(error.to_string());
                    None
                }
            } {
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
                        resize_fast_rgb(
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
                        let mut dst: Image<u8, 3> = Image::from_size_val(new_image_size, 0)
                            .expect("Failed to create Image");
                        resize_fast_rgb(
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

                        let owned =
                            Image::from_size_slice(image_frame.size(), image_frame.as_slice())
                                .expect("Failed to copy frame to owned buffer");
                        *texture_store = Some(TextureStore {
                            image: owned,
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
        AppState::Paused(_, Some(texture_store))
        | AppState::Finished(_, Some(texture_store))
        | AppState::Failed(_, _, Some(texture_store)) => {
            let sized_texture = egui::load::SizedTexture::new(
                texture_store.texture_handle.id(),
                egui::Vec2::new(
                    texture_store.image.width() as f32,
                    texture_store.image.height() as f32,
                ),
            );

            ui.image(sized_texture);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn changing_speed_keeps_undisplayed_frames() -> Result<(), Box<dyn std::error::Error>> {
        // Five 16×16 RGB frames at 1 FPS, generated with FFmpeg's testsrc.
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/five-frames.mov");
        let mut reader = open_player_reader(&path)?;
        reader.start()?;

        assert_eq!(wait_for_frame(&mut reader)?, Duration::ZERO);
        // Simulate the UI holding the first frame before the user changes speed.
        std::thread::sleep(Duration::from_millis(200));
        let mut selected_speed = 4;
        let mut active_speed = 1;
        apply_playback_speed(&reader, &mut selected_speed, &mut active_speed);
        assert_eq!(active_speed, 4);

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if wait_for_frame(&mut reader)? >= Duration::from_secs(1) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "no later frame after changing speed"
            );
        }
        reader.close()?;
        Ok(())
    }

    #[test]
    fn restart_uses_speed_selected_before_playback() -> Result<(), Box<dyn std::error::Error>> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/five-frames.mov");
        let mut reader = open_player_reader(&path)?;
        reader.start()?;
        wait_for_frame(&mut reader)?;

        let mut selected_speed = 8;
        let mut active_speed = 1;
        apply_playback_speed(&reader, &mut selected_speed, &mut active_speed);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !reader.is_eos() {
            assert!(Instant::now() < deadline, "playback did not finish");
            let _ = reader.try_grab_rgb8()?;
            std::thread::sleep(Duration::from_millis(5));
        }

        selected_speed = 4;
        reader.reset()?;
        reader.start()?;
        active_speed = 1;
        assert_eq!(wait_for_frame(&mut reader)?, Duration::ZERO);
        let started = Instant::now();
        apply_playback_speed(&reader, &mut selected_speed, &mut active_speed);
        assert_eq!(active_speed, 4);
        loop {
            if wait_for_frame(&mut reader)? >= Duration::from_secs(3) {
                break;
            }
            assert!(started.elapsed() < Duration::from_secs(2));
        }
        assert!(started.elapsed() < Duration::from_secs(2));
        reader.close()?;
        Ok(())
    }

    fn wait_for_frame(reader: &mut VideoReader) -> Result<Duration, Box<dyn std::error::Error>> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match reader.try_grab_rgb8()? {
                Grab::Frame { pts, .. } => return pts.ok_or("missing frame PTS".into()),
                Grab::Eos => return Err("reached EOS before displaying the expected frame".into()),
                Grab::Pending if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Grab::Pending => return Err("timed out waiting for a frame".into()),
            }
        }
    }
}
