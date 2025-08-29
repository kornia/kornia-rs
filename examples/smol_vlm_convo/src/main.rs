use color_eyre::Result;
use crossterm::event::{self, Event, KeyEventKind};
use kornia_image::Image;
use kornia_io::jpeg::read_image_jpeg_rgb8;
use kornia_io::png::read_image_png_rgb8;
use kornia_tensor::CpuAllocator;
use kornia_vlm::smolvlm::{utils::SmolVlmConfig, SmolVlm};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::{DefaultTerminal, Frame};
use std::path::PathBuf;
mod model;

fn main() -> Result<()> {
    color_eyre::install()?;
    let terminal = ratatui::init();
    let res = run(terminal);
    ratatui::restore();
    return res;

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
        vertical[0].height.saturating_sub(2) as usize
    }
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
    vertical[0].height.saturating_sub(2) as usize
}

struct AppState {
    prompt_input: String,
    chat_history: Vec<String>,
    scroll: usize, // scroll offset for chat
    file_input_mode: bool,
    file_list: Vec<(String, bool)>, // (name, is_folder)
    file_selected: usize,
    file_dir: String,
    ai_streaming: Option<(String, usize)>, // (full_msg, current_index)

    model: SmolVlm<CpuAllocator>,
    delta_image: Vec<PathBuf>, // for image history between the most recentinput and execution
}

impl AppState {
    fn flatten_chat_lines(chat_history: &[String]) -> Vec<String> {
        let mut all_lines = Vec::new();
        for msg in chat_history {
            for line in msg.lines() {
                all_lines.push(line.to_string());
            }
        }
        all_lines
    }

    // TODO: have it get the live response (programatically) instead of first getting the complete response.
    // TODO: also display an error if image failed to load
    fn get_response(&mut self) -> String {
        let img_path = self.delta_image.last().cloned().and_then(|path| {
            match path.extension().and_then(|ext| ext.to_str()) {
                Some("jpg") | Some("jpeg") => read_image_jpeg_rgb8(path).ok(),
                Some("png") => read_image_png_rgb8(path).ok(),
                _ => {
                    eprintln!("Unsupported image format. Only JPEG and PNG are supported.");
                    None
                }
            }
        });

        let response = self
            .model
            .inference(self.prompt_input.trim(), img_path, 200, CpuAllocator)
            .unwrap();

        self.delta_image.clear();

        response
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
                    self.chat_history.clear();
                    self.scroll = 0;
                    self.model.clear_context();
                } else {
                    let user_msg = format!("You: {}", self.prompt_input.trim());
                    if !self.prompt_input.trim().is_empty() {
                        self.chat_history.push(user_msg);
                        let ai_msg = format!("SmolVLM: {}", self.get_response());
                        self.ai_streaming = Some((ai_msg, 0));
                    }
                    self.prompt_input.clear();
                    self.scroll = 0;
                }
            }
            KeyCode::Esc => return Ok(true),
            KeyCode::Up => {
                let all_lines = AppState::flatten_chat_lines(&self.chat_history);
                let total_lines = all_lines.len();
                let max_scroll = total_lines.saturating_sub(chat_height);
                if self.scroll < max_scroll {
                    self.scroll += 1;
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

    fn handle_tick(&mut self) {
        if let Some((full, idx)) = &mut self.ai_streaming {
            if *idx < full.len() {
                *idx += 1;
            }
            if *idx >= full.len() {
                self.chat_history.push(full.clone());
                self.ai_streaming = None;
            }
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
        chat_history: Vec::new(),
        scroll: 0,
        file_input_mode: false,
        file_list: Vec::new(),
        file_selected: 0,
        file_dir: ".".to_string(),
        ai_streaming: None,
        model: SmolVlm::new(SmolVlmConfig::default())?,
        delta_image: Vec::new(),
    };
    let mut last_chat_height = 0;
    use std::time::{Duration, Instant};
    let mut last_tick = Instant::now();
    let tick_rate = Duration::from_millis(20);
    loop {
        let mut chat_height = 0;
        terminal.draw(|f| {
            chat_height = get_chat_height(f);
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
            } else if let Event::Resize(_, _) = ev {
                state.scroll = 0;
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
    // 1. Collect all chat lines (including streaming AI message if present), wrapping each to the chat area width
    use textwrap::wrap;
    let chat_area_width = right_chunks[0].width.saturating_sub(2) as usize;
    let mut all_lines: Vec<String> = Vec::new();
    for msg in &state.chat_history {
        let wrapped = if chat_area_width > 0 {
            wrap(msg, chat_area_width)
                .into_iter()
                .map(|l| l.into_owned())
                .collect::<Vec<_>>()
        } else {
            vec![msg.clone()]
        };
        all_lines.extend(wrapped);
    }
    // For streaming AI output, also wrap as a new block
    if let Some((full, idx)) = &state.ai_streaming {
        if *idx < full.len() {
            let partial = &full[..*idx];
            let wrapped = if chat_area_width > 0 {
                wrap(partial, chat_area_width)
                    .into_iter()
                    .map(|l| l.into_owned())
                    .collect::<Vec<_>>()
            } else {
                vec![partial.to_string()]
            };
            all_lines.extend(wrapped);
        }
    }
    // 2. Determine chat area height (minus borders)
    let chat_height = right_chunks[0].height.saturating_sub(2) as usize;
    // 3. Calculate scroll bounds
    let total_lines = all_lines.len();
    let max_scroll = total_lines.saturating_sub(chat_height);
    let scroll = state.scroll.min(max_scroll);
    // 4. Always show the last chat_height lines (flush to bottom), unless scrolled up
    let end = total_lines.saturating_sub(scroll);
    let start = end.saturating_sub(chat_height);
    let visible_lines: Vec<&str> = all_lines
        .get(start..end)
        .map_or(vec![], |slice| slice.iter().map(|s| s.as_str()).collect());
    // Pad the top with empty lines if not enough messages to fill the chat area
    let mut padded_lines = vec![""; chat_height.saturating_sub(visible_lines.len())];
    padded_lines.extend(visible_lines);
    let chat_text = padded_lines.join("\n");

    let chat_block = Block::default()
        .title(Span::styled(
            " SmolVLM Chat ",
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
