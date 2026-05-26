use std::collections::VecDeque;
use std::time::Instant;

/// A simple frame per second (FPS) counter.
///
/// # Examples
///
/// ```
/// use kornia_io::fps_counter::FpsCounter;
///
/// let mut fps_counter = FpsCounter::new();
///
/// for _ in 0..100 {
///    fps_counter.update();
/// }
/// ```
pub struct FpsCounter {
    frame_times: VecDeque<Instant>,
    window_size: usize,
}

impl FpsCounter {
    /// Creates a new `FpsCounter`.
    pub fn new() -> Self {
        Self::with_window_size(30) // 30 frame window
    }

    /// Creates a new `FpsCounter` with a specified window size.
    pub fn with_window_size(window_size: usize) -> Self {
        Self {
            frame_times: VecDeque::with_capacity(window_size + 1),
            window_size,
        }
    }

    /// Updates the frame count and calculates the FPS.
    pub fn update(&mut self) {
        let now = Instant::now();
        self.frame_times.push_back(now);

        if self.frame_times.len() > self.window_size {
            self.frame_times.pop_front();
        }
    }

    /// Returns the current FPS.
    pub fn fps(&self) -> f32 {
        if self.frame_times.len() < 2 {
            return 0.0;
        }

        if let (Some(first), Some(last)) = (self.frame_times.front(), self.frame_times.back()) {
            let duration = last.duration_since(*first);
            if duration.as_secs_f32() > 0.0 {
                return (self.frame_times.len() - 1) as f32 / duration.as_secs_f32();
            }
        }
        0.0
    }
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_fps_counter() {
        let mut fps_counter = super::FpsCounter::new();
        fps_counter.update();
        fps_counter.update();
        fps_counter.update();
    }
}
