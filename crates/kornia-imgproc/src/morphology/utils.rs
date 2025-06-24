/// A grayscale image represented by width, height, and pixel data.
#[derive(Clone, PartialEq, Debug)]
pub struct GrayImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Pixel data in row-major order.
    pub pixels: Vec<u8>,
}

impl GrayImage {
    /// Creates a new GrayImage with given dimensions and empty pixels.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width * height) as usize],
        }
    }

    /// Returns the pixel value at position (x, y).
    pub fn get_pixel(&self, x: u32, y: u32) -> u8 {
        debug_assert!(x < self.width && y < self.height);
        self.pixels[(y * self.width + x) as usize]
    }

    /// Sets the pixel value at position (x, y).
    pub fn put_pixel(&mut self, x: u32, y: u32, val: u8) {
        let idx = (y * self.width + x) as usize;
        self.pixels[idx] = val;
    }

    /// Returns the image dimensions (width, height).
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Creates a `GrayImage` from a given width, height, and a flat vector of pixels.
    ///
    /// # Panics
    /// Panics if the length of the pixel vector does not match `width * height`.
    pub fn from_vec(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        assert_eq!(pixels.len(), (width * height) as usize);
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Returns a string representation of the grayscale image using ASCII characters.
    /// This is useful for visualizing image content in the terminal.
    pub fn to_ascii(&self) -> String {
        let mut out = String::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let v = self.get_pixel(x, y);
                let c = match v {
                    0..=50 => ' ',
                    51..=100 => '.',
                    101..=150 => '*',
                    151..=200 => 'O',
                    _ => '@',
                };
                out.push(c);
            }
            out.push('\n');
        }
        out
    }
}

/// Defines how image borders are handled in filtering operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderMode {
    /// Ignore border pixels outside the image (skip them).
    Ignore,
    /// Clamp to the nearest border pixel.
    Clamp,
    /// Treat border pixels as zero (black).
    Constant(u8),
}
