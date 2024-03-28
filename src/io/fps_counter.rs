use std::time::{Duration, Instant};

pub struct FpsCounter {
    last_time: Instant,
    frame_count: u32,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            last_time: Instant::now(),
            frame_count: 0,
        }
    }

    pub fn new_frame(&mut self) {
        self.frame_count += 1;

        let now = Instant::now();
        let duration = now.duration_since(self.last_time);

        if duration >= Duration::new(1, 0) {
            let fps = self.frame_count as f32 / duration.as_secs_f32();
            println!("FPS: {:.2}", fps);

            // Reset for the next calculation
            self.frame_count = 0;
            self.last_time = now;
        }
    }
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self::new()
    }
}
