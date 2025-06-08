#[derive(Default, Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
/// Represents a pixel that can be white, black, or skipped.
pub enum Pixel {
    /// A white pixel.
    White = 255,
    /// A black pixel.
    Black = 0,
    /// A pixel to be skipped.
    #[default]
    Skip = 127,
}

impl Pixel {
    /// Returns the numeric value of the pixel.
    pub fn value(&self) -> u8 {
        match self {
            Pixel::White => 255,
            Pixel::Black => 0,
            Pixel::Skip => 127,
        }
    }
}

impl PartialEq<u8> for Pixel {
    fn eq(&self, other: &u8) -> bool {
        self.value() == *other
    }
}
