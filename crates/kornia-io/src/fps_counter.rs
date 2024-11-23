use std::time::Instant;

/// The smoothing factor for the FPS calculation.
const SMOOTHING: f32 = 0.95;

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
    last_time: Instant,
    frame_count: u32,
    fps: f32,
}

impl FpsCounter {
    /// Creates a new `FpsCounter`.
    pub fn new() -> Self {
        Self {
            last_time: Instant::now(),
            frame_count: 0,
            fps: 0.0,
        }
    }

    /// Returns the current FPS.
    #[inline]
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// Updates the frame count and calculates the FPS.
    pub fn update(&mut self) {
        self.frame_count += 1;

        let now = Instant::now();
        let duration = now.duration_since(self.last_time);

        // update fps
        let instant_fps = 1.0 / duration.as_secs_f32();
        self.fps = if self.fps == 0.0 {
            instant_fps
        } else {
            self.fps * SMOOTHING + instant_fps * (1.0 - SMOOTHING)
        };
        self.last_time = now;
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
