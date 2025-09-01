use color_eyre::Result;
use crossterm::event::{self, Event, KeyEventKind};
use kornia_image::Image;
use kornia_io::jpeg::read_image_jpeg_rgb8;
use kornia_io::png::read_image_png_rgb8;
use std::sync::mpsc;
mod model_state;
use model_state::{ModelRequest, ModelResponse, ModelStateHandle};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::{DefaultTerminal, Frame};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
mod model;

fn main() -> Result<()> {
    color_eyre::install()?;
    let terminal = ratatui::init();
    let res = run(terminal);
    ratatui::restore();
    return res;
}

fn get_chat_height(f: &Frame) -> usize {
    let area = f.area();
    let right_chunks = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Horizontal)
        .constraints([
            ratatui::layout::Constraint::Length(30),
            ratatui::layout::Constraint::Min(1),
        ])
        .split(area);
    let vertical = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Min(3),
            ratatui::layout::Constraint::Length(3),
        ])
        .split(right_chunks[1]);
    vertical[0].height.saturating_sub(2) as usize - 2
}

struct AppState {
    prompt_input: String,
    chat_history: Rc<RefCell<Vec<String>>>,
    scroll: usize, // scroll offset for chat
    file_input_mode: bool,
    file_list: Vec<(String, bool)>, // (name, is_folder)
    file_selected: usize,
    file_dir: String,
    ai_streaming: Option<mpsc::Receiver<ModelResponse>>, // streaming channel from model thread
    response_stream: Rc<RefCell<String>>,                // live response buffer

    model_handle: ModelStateHandle,
    delta_image: Vec<PathBuf>, // for image history between the most recentinput and execution
    last_chat_area_width: Option<usize>, // for scroll logic

    // Spinner state
    spinner_index: usize,
    show_spinner: bool,
    spinner_tick: usize, // counts ticks for spinner speed
}

impl AppState {
    /// Returns all visible chat lines, including streaming response, wrapped to the given width.
    fn get_all_chat_lines(&self, wrap_width: usize) -> Vec<String> {
        use textwrap::wrap;
        let mut all_lines: Vec<String> = Vec::new();
        for msg in self.chat_history.borrow().iter() {
            let wrapped = if wrap_width > 0 {
                wrap(msg, wrap_width)
                    .into_iter()
                    .map(|l| l.into_owned())
                    .collect::<Vec<_>>()
            } else {
                vec![msg.clone()]
            };
            all_lines.extend(wrapped);
        }
        // Add streaming response if present
        let partial = self.response_stream.borrow();
        if !partial.is_empty() {
            let wrapped = if wrap_width > 0 {
                wrap(partial.as_str(), wrap_width)
                    .into_iter()
                    .map(|l| l.into_owned())
                    .collect::<Vec<_>>()
            } else {
                vec![partial.to_string()]
            };
            all_lines.extend(wrapped);
        }
        // Add spinner if active
        if self.show_spinner {
            let spinner_chars = ['-', '/', '|', '\\'];
            let c = spinner_chars[self.spinner_index % spinner_chars.len()];
            all_lines.push(format!("{} Thinking...", c));
        }
        all_lines
    }

    // TODO: have it get the live response (programatically) instead of first getting the complete response.
    // TODO: also display an error if image failed to load
    fn start_inference(&mut self) {
        use std::sync::mpsc;
        let (tx, rx) = mpsc::channel();
        let prompt = self.prompt_input.trim().to_string();
        let img = self.delta_image.last().cloned().and_then(|path| {
            match path.extension().and_then(|ext| ext.to_str()) {
                Some("jpg") | Some("jpeg") => read_image_jpeg_rgb8(path).ok(),
                Some("png") => read_image_png_rgb8(path).ok(),
                _ => None,
            }
        });
        self.model_handle
            .tx
            .send(ModelRequest::Inference {
                prompt,
                image: img,
                response_tx: tx,
            })
            .ok();
        self.response_stream.borrow_mut().clear();
        self.ai_streaming = Some(rx);
        self.delta_image.clear();
        // Start spinner
        self.spinner_index = 0;
        self.spinner_tick = 0;
        self.show_spinner = true;
    }

    fn handle_file_input_mode(&mut self, key: crossterm::event::KeyEvent) {
        use crossterm::event::KeyCode;
        match key.code {
            KeyCode::Up => {
                if self.file_selected > 0 {
                    self.file_selected -= 1;
                }
            }
            KeyCode::Down => {
                if self.file_selected + 1 < self.file_list.len() {
                    self.file_selected += 1;
                }
            }
            KeyCode::Enter => {
                if !self.file_list.is_empty() {
                    let (fname, is_folder) = &self.file_list[self.file_selected];
                    if fname == ".." {
                        self.change_to_parent_dir();
                    } else if *is_folder {
                        let fname_cloned = fname.clone();
                        self.change_to_sub_dir(&fname_cloned);
                    } else {
                        let path = if self.file_dir == "." {
                            fname.clone()
                        } else {
                            format!("{}/{}", self.file_dir, fname)
                        };
                        self.chat_history
                            .borrow_mut()
                            .push(format!("[Image inserted: {}]", path));
                        self.delta_image.push(PathBuf::from(path));
                        self.file_input_mode = false;
                        self.scroll = 0;
                    }
                }
            }
            KeyCode::Esc => {
                self.file_input_mode = false;
            }
            _ => {}
        }
    }

    fn handle_main_input(
        &mut self,
        key: crossterm::event::KeyEvent,
        chat_height: usize,
    ) -> Result<bool> {
        use crossterm::event::{KeyCode, KeyModifiers};
        match key.code {
            KeyCode::Char('i') | KeyCode::Char('I')
                if key.modifiers.contains(KeyModifiers::ALT) =>
            {
                self.open_file_picker();
            }
            KeyCode::Char(c) => {
                self.prompt_input.push(c);
            }
            KeyCode::Backspace => {
                self.prompt_input.pop();
            }
            KeyCode::Enter => {
                // Prevent user from submitting while AI is streaming
                if self.ai_streaming.is_some() {
                    // Ignore Enter if AI is still outputting
                    return Ok(false);
                }
                if key.modifiers.contains(KeyModifiers::ALT) {
                    self.chat_history.borrow_mut().clear();
                    self.scroll = 0;
                    // No model context to clear in threaded model
                } else {
                    let user_msg = format!("You: {}", self.prompt_input.trim());
                    if !self.prompt_input.trim().is_empty() {
                        self.chat_history.borrow_mut().push(user_msg);
                        self.start_inference();
                    }
                    self.prompt_input.clear();
                    self.scroll = 0;
                }
            }
            KeyCode::Esc => return Ok(true),
            KeyCode::Up => {
                // Use chat area width for wrapping, match render logic
                let wrap_width = self.last_chat_area_width.unwrap_or(80); // fallback
                let all_lines = self.get_all_chat_lines(wrap_width);
                let total_lines = all_lines.len();
                let max_scroll = total_lines.saturating_sub(chat_height);
                if self.scroll < max_scroll {
                    self.scroll += 1;
                } else if self.scroll == max_scroll && max_scroll > 0 {
                    // Allow reaching the very top
                    self.scroll = max_scroll;
                }
            }
            KeyCode::Down => {
                if self.scroll > 0 {
                    self.scroll -= 1;
                }
            }
            _ => {}
        }
        Ok(false)
    }

    // Store the last chat area width for scroll logic

    fn handle_tick(&mut self) {
        if let Some(rx) = &self.ai_streaming {
            // Collect all available messages first to avoid borrow issues
            let mut responses = Vec::new();
            while let Ok(msg) = rx.try_recv() {
                responses.push(msg);
            }
            for msg in responses {
                match msg {
                    ModelResponse::StreamChunk(chunk) => {
                        *self.response_stream.borrow_mut() = chunk;
                    }
                    ModelResponse::Done => {
                        let ai_msg = format!("SmolVLM: {}", self.response_stream.borrow());
                        self.chat_history.borrow_mut().push(ai_msg);
                        self.response_stream.borrow_mut().clear();
                        self.ai_streaming = None;
                        self.show_spinner = false;
                    }
                    ModelResponse::Error(e) => {
                        self.chat_history.borrow_mut().push(format!("[Error] {e}"));
                        self.response_stream.borrow_mut().clear();
                        self.ai_streaming = None;
                        self.show_spinner = false;
                    }
                }
            }
            // Advance spinner frame if still streaming, but slower (every 5 ticks)
            if self.ai_streaming.is_some() {
                self.spinner_tick += 1;
                if self.spinner_tick >= 5 {
                    self.spinner_index = (self.spinner_index + 1) % 4;
                    self.spinner_tick = 0;
                }
            }
        } else {
            self.show_spinner = false;
        }
    }

    fn open_file_picker(&mut self) {
        use std::fs;
        let dir = &self.file_dir;
        let mut folders = vec![("..".to_string(), true)];
        let mut files = vec![];
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Ok(ft) = entry.file_type() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if ft.is_dir() {
                        folders.push((name, true));
                    } else if ft.is_file() {
                        let lower = name.to_lowercase();
                        if lower.ends_with(".png")
                            || lower.ends_with(".jpg")
                            || lower.ends_with(".jpeg")
                        {
                            files.push((name, false));
                        }
                    }
                }
            }
        }
        folders.sort_by(|a, b| a.0.cmp(&b.0));
        files.sort_by(|a, b| a.0.cmp(&b.0));
        let mut file_list: Vec<(String, bool)> = folders.into_iter().chain(files).collect();
        if file_list.is_empty() {
            file_list.push(("..".to_string(), true));
        }
        self.file_list = file_list;
        self.file_selected = 0;
        self.file_input_mode = true;
    }

    fn change_to_parent_dir(&mut self) {
        let current = PathBuf::from(&self.file_dir);
        let abs = if current.is_absolute() {
            current
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("/"))
                .join(current)
        };
        let parent = abs
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("/"));
        self.file_dir = parent.to_string_lossy().to_string();
        self.update_file_list();
    }

    fn change_to_sub_dir(&mut self, fname: &str) {
        let mut new_dir = PathBuf::from(&self.file_dir);
        new_dir.push(fname);
        self.file_dir = new_dir.to_string_lossy().to_string();
        self.update_file_list();
    }

    fn update_file_list(&mut self) {
        use std::fs;
        let mut folders = vec![("..".to_string(), true)];
        let mut files = vec![];
        if let Ok(entries) = fs::read_dir(&self.file_dir) {
            for entry in entries.flatten() {
                if let Ok(ft) = entry.file_type() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if ft.is_dir() {
                        folders.push((name, true));
                    } else if ft.is_file() {
                        let lower = name.to_lowercase();
                        if lower.ends_with(".png")
                            || lower.ends_with(".jpg")
                            || lower.ends_with(".jpeg")
                        {
                            files.push((name, false));
                        }
                    }
                }
            }
        }
        folders.sort_by(|a, b| a.0.cmp(&b.0));
        files.sort_by(|a, b| a.0.cmp(&b.0));
        self.file_list = folders.into_iter().chain(files).collect();
        self.file_selected = 0;
    }
}

fn run(mut terminal: DefaultTerminal) -> Result<()> {
    let mut state = AppState {
        prompt_input: String::new(),
        chat_history: Rc::new(RefCell::new(Vec::new())),
        scroll: 0,
        file_input_mode: false,
        file_list: Vec::new(),
        file_selected: 0,
        file_dir: ".".to_string(),
        ai_streaming: None,
        response_stream: Rc::new(RefCell::new(String::new())),
        model_handle: ModelStateHandle::new(),
        delta_image: Vec::new(),
        last_chat_area_width: None,
        spinner_index: 0,
        show_spinner: false,
        spinner_tick: 0,
    };
    let mut last_chat_height = 0;
    use std::time::{Duration, Instant};
    let mut last_tick = Instant::now();
    let tick_rate = Duration::from_millis(20);
    loop {
        let mut chat_height = 0;
        let mut chat_area_width = 0;
        terminal.draw(|f| {
            chat_height = get_chat_height(f);
            // Get chat area width for scroll logic
            let area = f.area();
            let right_chunks = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Horizontal)
                .constraints([
                    ratatui::layout::Constraint::Length(30),
                    ratatui::layout::Constraint::Min(1),
                ])
                .split(area);
            let vertical = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Vertical)
                .constraints([
                    ratatui::layout::Constraint::Min(3),
                    ratatui::layout::Constraint::Length(3),
                ])
                .split(right_chunks[1]);
            chat_area_width = vertical[0].width.saturating_sub(2) as usize;
            state.last_chat_area_width = Some(chat_area_width);
            render(f, &state);
        })?;
        if chat_height != last_chat_height {
            state.scroll = 0;
            last_chat_height = chat_height;
        }
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::from_millis(0));
        if event::poll(timeout)? {
            let ev = event::read()?;
            if let Event::Key(key) = ev {
                if state.file_input_mode {
                    state.handle_file_input_mode(key);
                } else if let KeyEventKind::Press = key.kind {
                    if state.handle_main_input(key, chat_height)? {
                        return Ok(());
                    }
                }
            } else if let Event::Mouse(me) = ev {
                use crossterm::event::MouseEventKind;
                match me.kind {
                    MouseEventKind::ScrollUp => {
                        // Scroll up
                        let wrap_width = state.last_chat_area_width.unwrap_or(80);
                        let all_lines = state.get_all_chat_lines(wrap_width);
                        let total_lines = all_lines.len();
                        let max_scroll = total_lines.saturating_sub(chat_height);
                        if state.scroll < max_scroll {
                            state.scroll += 1;
                        } else if state.scroll == max_scroll && max_scroll > 0 {
                            state.scroll = max_scroll;
                        }
                    }
                    MouseEventKind::ScrollDown => {
                        if state.scroll > 0 {
                            state.scroll -= 1;
                        }
                    }
                    _ => {}
                }
            } else if let Event::Resize(_, _) = ev {
                // Clamp scroll to new max_scroll after resize
                let wrap_width = state.last_chat_area_width.unwrap_or(80);
                let all_lines = state.get_all_chat_lines(wrap_width);
                let total_lines = all_lines.len();
                let max_scroll = total_lines.saturating_sub(chat_height);
                state.scroll = state.scroll.min(max_scroll);
                last_chat_height = chat_height;
            }
        } else {
            state.handle_tick();
            last_tick = Instant::now();
        }
    }
}

fn render(frame: &mut Frame, state: &AppState) {
    let area = frame.area();
    // Split horizontally: left (configs), right (main)
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(30), // left pane width
            Constraint::Min(1),     // right pane
        ])
        .split(area);

    use ratatui::style::{Color, Modifier, Style};
    // Windows 95 color palette
    let win95_bg = Color::Rgb(0, 128, 128); // Windows 95 teal (blue-green)
    let win95_border = Color::Rgb(192, 192, 192); // light gray (Win95 border)
    let win95_text = Color::Black;
    let win95_highlight_bg = Color::Rgb(0, 0, 128); // blue
    let win95_highlight_fg = Color::White;
    let win95_header = Color::Rgb(160, 160, 160); // classic win95 gray for headers
    let win95_lightgrey = Color::Rgb(192, 192, 192); // light grey for file dialog
    let win95_file_bg = Color::Rgb(224, 224, 224); // lighter grey for file list background

    // Left pane: Configs
    use ratatui::text::Span;
    let config_block = Block::default()
        .title(Span::styled(
            " Configs ",
            Style::default()
                .bg(win95_header)
                .fg(win95_text)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL | Borders::RIGHT)
        .border_style(Style::default().fg(win95_border))
        .style(Style::default().bg(win95_bg));
    frame.render_widget(config_block, chunks[0]);

    // Right pane: split vertically for chat and prompt
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),    // chat history
            Constraint::Length(5), // prompt input (was 3)
        ])
        .split(chunks[1]);

    // --- Chat rendering: always flush to bottom, bottom-up ---
    let chat_area_width = right_chunks[0].width.saturating_sub(2) as usize;
    let all_lines = state.get_all_chat_lines(chat_area_width);
    let chat_height = right_chunks[0].height.saturating_sub(2) as usize;
    let total_lines = all_lines.len();
    let max_scroll = total_lines.saturating_sub(chat_height);
    let scroll = state.scroll.min(max_scroll);
    let end = total_lines.saturating_sub(scroll);
    let start = end.saturating_sub(chat_height).max(0);
    let visible_lines: Vec<&str> = all_lines
        .get(start..end)
        .map_or(vec![], |slice| slice.iter().map(|s| s.as_str()).collect());
    let chat_text = if total_lines < chat_height {
        // Not enough lines to fill the area, pad the top
        let mut padded_lines = vec![""; chat_height - total_lines];
        padded_lines.extend(visible_lines);
        padded_lines.join("\n")
    } else {
        // Show lines at the very top when scrolled to max
        visible_lines.join("\n")
    };

    let chat_block = Block::default()
        .title(Span::styled(
            " Chat ",
            Style::default()
                .bg(win95_header)
                .fg(win95_text)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(win95_border))
        .style(Style::default().bg(win95_bg));
    let chat_para = Paragraph::new(chat_text)
        .block(chat_block)
        .style(Style::default().fg(win95_text).bg(win95_bg))
        .wrap(ratatui::widgets::Wrap { trim: false });
    frame.render_widget(chat_para, right_chunks[0]);

    // Prompt input area
    let prompt_block = Block::default()
        .title(Span::styled(
            " Prompt ",
            Style::default()
                .bg(win95_header)
                .fg(win95_text)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(win95_border))
        .style(Style::default().bg(win95_bg));
    let prompt_para = Paragraph::new(state.prompt_input.as_str())
        .block(prompt_block)
        .style(Style::default().fg(win95_text).bg(win95_bg))
        .wrap(ratatui::widgets::Wrap { trim: false });
    frame.render_widget(prompt_para, right_chunks[1]);

    // Help line overlay (bottom of screen)
    let help_text = "Alt+Enter: Clear | Alt+I: Insert Image | Esc: Quit | Up/Down: Scroll";
    let area = frame.area();
    let help_rect = Rect {
        x: area.x,
        y: area.y + area.height - 1,
        width: area.width,
        height: 1,
    };
    let help_para =
        Paragraph::new(help_text).style(Style::default().fg(win95_text).bg(win95_border));
    frame.render_widget(help_para, help_rect);

    // --- File picker overlay ---
    if state.file_input_mode {
        // Centered popup
        let area = frame.area();
        let width = (area.width / 2).max(30);
        let height = (area.height / 2).max(10);
        let x = area.x + (area.width - width) / 2;
        let y = area.y + (area.height - height) / 2;
        let popup_rect = Rect {
            x,
            y,
            width,
            height,
        };

        // Fully clear the popup area before rendering the dialog
        use ratatui::widgets::Clear;
        frame.render_widget(Clear, popup_rect);

        let block = Block::default()
            .title(" Select Image File ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(win95_border))
            .style(Style::default().bg(win95_lightgrey));
        frame.render_widget(block, popup_rect);

        // Show current path at the top (inside the popup, above the file list)
        let path_rect = Rect {
            x: popup_rect.x + 1,
            y: popup_rect.y + 1,
            width: popup_rect.width - 2,
            height: 1,
        };
        let path_text = format!("Path: {}", state.file_dir);
        let path_para =
            Paragraph::new(path_text).style(Style::default().fg(Color::Black).bg(win95_lightgrey));
        frame.render_widget(path_para, path_rect);

        // List files/folders below the path
        let max_items = (height as usize).saturating_sub(3); // 1 for border, 1 for path, 1 for border
        let start = if state.file_selected >= max_items {
            state.file_selected + 1 - max_items
        } else {
            0
        };
        let end = (start + max_items).min(state.file_list.len());
        let items = &state.file_list[start..end];
        let list_area_y_start = popup_rect.y + 2;
        let list_area_y_end = popup_rect.y + popup_rect.height - 1;
        let list_area_height = (list_area_y_end - list_area_y_start) as usize;
        let mut lines: Vec<(String, Style)> = vec![];
        for i in 0..list_area_height {
            if i < items.len() {
                let (name, is_folder) = &items[i];
                let selected = start + i == state.file_selected;
                let (icon, style) = (
                    if *is_folder { "📁" } else { "🖼️" },
                    if selected {
                        Style::default()
                            .fg(win95_highlight_fg)
                            .bg(win95_highlight_bg)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::Black).bg(win95_file_bg)
                    },
                );
                let text = format!("{}  {}", icon, name);
                lines.push((text, style));
            } else {
                lines.push((String::new(), Style::default().bg(win95_file_bg)));
            }
        }
        for (i, (text, style)) in lines.iter().enumerate() {
            let y = popup_rect.y + 2 + i as u16;
            if y < popup_rect.y + popup_rect.height - 1 {
                let file_rect = Rect {
                    x: popup_rect.x + 1,
                    y,
                    width: popup_rect.width - 2,
                    height: 1,
                };
                // Always clear the row first with a blank Paragraph with the background color
                let clear_bg = Paragraph::new("")
                    .style(Style::default().bg(style.bg.unwrap_or(win95_file_bg)));
                frame.render_widget(clear_bg, file_rect);
                let para = Paragraph::new(text.as_str()).style(*style);
                frame.render_widget(para, file_rect);
            }
        }
    }
}
