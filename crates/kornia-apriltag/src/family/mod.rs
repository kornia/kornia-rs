/// Represents the AprilTag Family
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagFamily {
    /// The width of the tag including the border, in units.
    pub width_at_border: usize,
    /// Whether the border is reversed.
    pub reversed_border: bool,
    /// The total width of the tag, including border, in units.
    pub total_width: usize,
    /// The number of bits in the tag code.
    pub nbits: usize,
    /// The x-coordinates of each bit in the tag.
    pub bit_x: Vec<u8>,
    /// The y-coordinates of each bit in the tag.
    pub bit_y: Vec<u8>,
    /// The code data for the tag family.
    pub code_data: Vec<usize>,
    // TODO: more properties
}

#[doc(hidden)]
pub mod tag36h11;
